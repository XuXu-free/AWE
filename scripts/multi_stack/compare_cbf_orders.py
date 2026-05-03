"""
一阶 CBF vs 二阶 HOCBF 对比绘图

生成并排对比图，展示电流平滑性和 HTO 控制效果。
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from controller.multi_stack.cbf_projection import MultiStackCBFProjection
from controller.multi_stack.cbf_projection_ho import MultiStackCBFProjectionHO
from plant.multi_stack_simulator import MultiStackSimulator


def run_simulation(projector, duration=3600, u_ref=None):
    sim = MultiStackSimulator(dt=0.2)
    sim.reset()
    dt = 0.2
    steps = int(duration / dt)
    ctrl_steps = int(60.0 / dt)

    if u_ref is None:
        u_ref = np.array([600.0] * 4 + [0.05] * 4 + [0.01])
    current_action = u_ref.copy()
    last_action = current_action.copy()

    history = {'t': [], 'I_all': [], 'v_lye_all': [], 'v_c': [], 'HTO': [], 'T_s_all': [], 'projection_success': []}

    for i in range(steps):
        t = i * dt
        if i % ctrl_steps == 0:
            state = sim.state
            current_action, success, info = projector.project(u_ref, state, u_last=last_action)
            last_action = current_action.copy()

        if i % 50 == 0:
            state = sim.state
            n_gas = state[12]
            T_sep = state[5]
            hto = (n_gas * projector.R * T_sep) / (projector.P_sys * projector.V_sep_gas)
            history['t'].append(t / 60)
            history['I_all'].append(current_action[0:4].copy())
            history['v_lye_all'].append(current_action[4:8].copy())
            history['v_c'].append(current_action[8])
            history['HTO'].append(hto * 100)
            history['T_s_all'].append(state[1:5] - 273.15)
            history['projection_success'].append(success)

        sim.step(current_action)
    return history


def main():
    os.makedirs('output/multi_stack/cbf_tests', exist_ok=True)
    u_ref = np.array([600.0] * 4 + [0.05] * 4 + [0.01])

    print("Running first-order CBF (lambda=1000)...")
    cbf1 = MultiStackCBFProjection(
        dt=60.0,
        gamma_vec=[10.0]*4 + [1.0] + [10.0]*4,
        rho_vec=[5000]*9,
        h_margin_vec=[0.0]*4 + [0.007] + [0.0]*4,
        normalize=True,
        lambda_u_scale=1000.0,
    )
    h1 = run_simulation(cbf1, duration=3600, u_ref=u_ref)

    print("Running first-order CBF (lambda=0)...")
    cbf2 = MultiStackCBFProjection(
        dt=60.0,
        gamma_vec=[10.0]*4 + [1.0] + [10.0]*4,
        rho_vec=[5000]*9,
        h_margin_vec=[0.0]*4 + [0.007] + [0.0]*4,
        normalize=True,
        lambda_u_scale=0.0,
    )
    h2 = run_simulation(cbf2, duration=3600, u_ref=u_ref)

    print("Running second-order HOCBF (tiny margin)...")
    cbf3 = MultiStackCBFProjectionHO(
        dt=60.0,
        gamma_vec=[10.0]*4 + [1.0] + [10.0]*4,
        rho_vec=[5000]*9,
        h_margin_vec=[0.0]*4 + [0.0002] + [0.0]*4,
        normalize=True,
        lambda_u_scale=0.0,
        alpha1_hto=2.0,
        alpha2_hto=5.0,
    )
    h3 = run_simulation(cbf3, duration=3600, u_ref=u_ref)

    # Compute metrics
    def metrics(h):
        I = np.array(h['I_all'])
        dI = np.diff(I, axis=0)
        return {
            'max_hto': max(h['HTO']),
            'max_temp': max(np.array(h['T_s_all']).flatten()),
            'max_I': max(I.flatten()),
            'bounce': sum(1 for j in range(4) for i in range(1, len(dI))
                          if dI[i-1, j] * dI[i, j] < 0 and abs(dI[i, j]) > 50),
            'max_dI': float(np.max(np.abs(dI))),
            'mean_dI': float(np.mean(np.abs(dI))),
        }

    m1, m2, m3 = metrics(h1), metrics(h2), metrics(h3)
    hs = [h1, h2, h3]
    ts = [np.array(h['t']) for h in hs]
    labels = ['1st-order CBF (λ=1000)', '1st-order CBF (λ=0)', 'HOCBF tiny margin']
    colors = ['C0', 'C3', 'C1']

    I_ref = u_ref[0]
    v_lye_ref = u_ref[4]
    v_c_ref = u_ref[8]

    # Plot: 2 rows x 3 cols
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 0: Control-related
    # Current comparison
    ax = axes[0, 0]
    for idx, (h, color, label) in enumerate(zip(hs, colors, labels)):
        for j in range(4):
            ax.plot(ts[idx], np.array(h['I_all'])[:, j], color=color, alpha=0.3, linewidth=1.0)
        ax.plot([], [], color=color, label=label, linewidth=2)
    ax.axhline(y=I_ref, color='k', linestyle='--', alpha=0.5, label=f'Ref ({I_ref:.0f}A)')
    ax.set_title('Stack Currents')
    ax.set_ylabel('Current (A)')
    ax.set_xlabel('Time (min)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Lye flow rate comparison
    ax = axes[0, 1]
    for idx, (h, color, label) in enumerate(zip(hs, colors, labels)):
        for j in range(4):
            ax.plot(ts[idx], np.array(h['v_lye_all'])[:, j], color=color, alpha=0.3, linewidth=1.0)
        ax.plot([], [], color=color, label=label, linewidth=2)
    ax.axhline(y=v_lye_ref, color='k', linestyle='--', alpha=0.5, label=f'Ref ({v_lye_ref:.3f})')
    ax.set_title('Lye Flow Rates')
    ax.set_ylabel('Flow Rate (m³/s)')
    ax.set_xlabel('Time (min)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Coolant flow rate comparison
    ax = axes[0, 2]
    for idx, (h, color, label) in enumerate(zip(hs, colors, labels)):
        ax.plot(ts[idx], h['v_c'], color=color, label=label, linewidth=2)
    ax.axhline(y=v_c_ref, color='k', linestyle='--', alpha=0.5, label=f'Ref ({v_c_ref:.3f})')
    ax.set_title('Coolant Flow Rate')
    ax.set_ylabel('Flow Rate (m³/s)')
    ax.set_xlabel('Time (min)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 1: Performance-related
    # HTO comparison
    ax = axes[1, 0]
    for idx, (h, color, label) in enumerate(zip(hs, colors, labels)):
        ax.plot(ts[idx], h['HTO'], color=color, label=label, linewidth=2)
    ax.axhline(y=2.0, color='r', linestyle='--', alpha=0.5, label='Limit (2%)')
    ax.axhline(y=1.8, color='g', linestyle=':', alpha=0.5, label='Target (1.8%)')
    max_hto_all = max(max(h['HTO']) for h in hs)
    ax.fill_between(ts[0], 2.0, max_hto_all * 1.05, alpha=0.1, color='red')
    ax.set_title('HTO (%)')
    ax.set_ylabel('HTO (%)')
    ax.set_xlabel('Time (min)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Temperature comparison
    ax = axes[1, 1]
    for idx, (h, color, label) in enumerate(zip(hs, colors, labels)):
        T = np.array(h['T_s_all'])
        ax.plot(ts[idx], np.mean(T, axis=1), color=color, label=label, linewidth=2)
    ax.axhline(y=90, color='r', linestyle='--', alpha=0.5, label='Tmax (90°C)')
    ax.set_title('Mean Stack Temperature')
    ax.set_ylabel('Temperature (°C)')
    ax.set_xlabel('Time (min)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # dI boxplot (smoothness distribution)
    ax = axes[1, 2]
    dI_data = [np.abs(np.diff(np.array(h['I_all']), axis=0).flatten()) for h in hs]
    short_labels = ['1st λ=1000', '1st λ=0', 'HOCBF']
    bp = ax.boxplot(
        dI_data,
        tick_labels=short_labels,
        patch_artist=True,
        medianprops=dict(color='black', linewidth=1.5),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        showfliers=True,
        flierprops=dict(marker='o', markersize=3, alpha=0.4),
    )
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_title('|dI| Distribution (Smoothness)')
    ax.set_ylabel('|dI| (A/step)')
    ax.set_yscale('log')
    ax.set_ylim(bottom=1e-1)
    ax.tick_params(axis='x', rotation=15)
    ax.grid(True, alpha=0.3, axis='y')

    # Summary text
    summary = (
        f"1st-order CBF (λ=1000): Max HTO={m1['max_hto']:.3f}%  Max I={m1['max_I']:.0f}A  "
        f"Bounces={m1['bounce']}  Mean|dI|={m1['mean_dI']:.1f}\n"
        f"1st-order CBF (λ=0):    Max HTO={m2['max_hto']:.3f}%  Max I={m2['max_I']:.0f}A  "
        f"Bounces={m2['bounce']}  Mean|dI|={m2['mean_dI']:.1f}\n"
        f"HOCBF tiny margin:      Max HTO={m3['max_hto']:.3f}%  Max I={m3['max_I']:.0f}A  "
        f"Bounces={m3['bounce']}  Mean|dI|={m3['mean_dI']:.1f}"
    )
    fig.text(0.5, 0.005, summary, ha='center', fontsize=10, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))

    plt.suptitle('First-Order CBF vs Second-Order HOCBF Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    out_path = 'output/multi_stack/cbf_tests/cbf_order_comparison.png'
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison figure: {out_path}")


if __name__ == "__main__":
    main()
