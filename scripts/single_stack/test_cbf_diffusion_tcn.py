#!/usr/bin/env python3
"""
Single-Stack TCN Diffusion + CBF Model Controller Test

Tests the trained diffusion_tcn policy with CBF safety projection.
Compares three variants:
1. TCN Diffusion (raw, no safety)
2. TCN Diffusion + Safe Projection
3. TCN Diffusion + CBF Projection (tuned params)
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from plant.single_stack_simulator import SingleStackSimulator
from controller.single_stack.cbf_model_controller import SingleStackCBFModelController
from controller.single_stack.safe_model_controller import SingleStackSafeModelController
from controller.single_stack.model_controller import SingleStackModelController


def run_test(controller, name, initial_state, P_ref_profile, duration=7200, dt=60.0):
    """Run a single controller test."""
    print(f"\n{'='*70}")
    print(f"Testing: {name}")
    print(f"{'='*70}")

    sim = SingleStackSimulator(sim_dt=0.2)
    sim.reset(initial_state=initial_state)

    steps = int(duration / dt)

    def get_p_ref(t):
        for t_start, t_end, p_ref in P_ref_profile:
            if t_start <= t < t_end:
                return p_ref
        return P_ref_profile[-1][2]

    history = {
        't': [], 'I': [], 'v_lye': [], 'v_c': [],
        'T_s': [], 'T_sep': [], 'T_c_out': [],
        'P_real': [], 'P_ref': [], 'U_cell': [],
        'HTO': [], 'n_gas': [],
        'HTO_violation': [], 'T_violation': [], 'P_violation': []
    }

    T_max = 363.15
    HTO_max = 2.0
    P_max = 6.0e6

    last_action = np.array([2000.0, 0.01, 0.001])

    for i in range(steps):
        t = i * dt
        P_ref = get_p_ref(t)
        T_ref = 353.15
        P_ref_seq = [get_p_ref(t + j * dt) for j in range(5)]

        state = sim.state
        try:
            if hasattr(controller, 'get_action_with_rollout'):
                I_cmd, v_lye_cmd, v_c_cmd = controller.get_action_with_rollout(
                    state, P_ref_seq, T_ref, last_action, num_candidates=64, verbose=False
                )
            else:
                I_cmd, v_lye_cmd, v_c_cmd = controller.get_action(
                    state, P_ref_seq, T_ref, last_action
                )
        except Exception as e:
            print(f"Error @ t={t}: {e}")
            I_cmd, v_lye_cmd, v_c_cmd = last_action[0], last_action[1], last_action[2]

        action = np.array([I_cmd, v_lye_cmd, v_c_cmd])
        last_action = action.copy()

        T_s_in, T_s, T_sep, T_c_out = state[0], state[1], state[2], state[3]
        n_H2_an, n_liq, n_gas = state[4], state[5], state[6]

        _, U_cell, _ = sim._calculate_electrochemical_properties(I_cmd, T_s)
        P_real = U_cell * I_cmd * sim.N_cell
        hto_pct = (n_gas * sim.R * T_sep) / (sim.P_sys * sim.V_sep_gas) * 100

        history['t'].append(t / 60)
        history['I'].append(I_cmd)
        history['v_lye'].append(v_lye_cmd)
        history['v_c'].append(v_c_cmd)
        history['T_s'].append(T_s - 273.15)
        history['T_sep'].append(T_sep - 273.15)
        history['T_c_out'].append(T_c_out - 273.15)
        history['P_real'].append(P_real / 1e6)
        history['P_ref'].append(P_ref / 1e6)
        history['U_cell'].append(U_cell)
        history['HTO'].append(hto_pct)
        history['n_gas'].append(n_gas)
        history['HTO_violation'].append(hto_pct > HTO_max)
        history['T_violation'].append(T_s > T_max)
        history['P_violation'].append(P_real > P_max)

        for _ in range(int(dt / sim.dt)):
            sim.step(action)

    max_hto = max(history['HTO'])
    max_T = max(history['T_s'])
    max_P = max(history['P_real'])
    hto_violations = sum(history['HTO_violation'])
    T_violations = sum(history['T_violation'])
    P_violations = sum(history['P_violation'])

    print(f"\n  Results:")
    print(f"    Max Temp: {max_T:.2f}°C (limit: 90°C)")
    print(f"    Max Power: {max_P:.3f} MW (limit: 6 MW)")
    print(f"    Max HTO: {max_hto:.4f}% (limit: 2%)")
    print(f"    T violations: {T_violations} / {steps}")
    print(f"    P violations: {P_violations} / {steps}")
    print(f"    HTO violations: {hto_violations} / {steps}")

    return history


def plot_results(histories, names, output_dir):
    """Plot comparison of all controllers."""
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

    # Row 1
    ax = fig.add_subplot(gs[0, 0])
    for hist, name, color in zip(histories, names, colors):
        ax.plot(hist['t'], hist['P_ref'], 'k--', alpha=0.3)
        ax.plot(hist['t'], hist['P_real'], color=color, linewidth=2, label=name)
    ax.axhline(y=6.0, color='r', linestyle='--', alpha=0.5, label='P_max')
    ax.set_title('Power Tracking', fontweight='bold')
    ax.set_ylabel('MW')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    for hist, name, color in zip(histories, names, colors):
        ax.plot(hist['t'], hist['HTO'], color=color, linewidth=2, label=name)
    ax.axhline(y=2.0, color='r', linestyle='--', alpha=0.5, label='HTO_max')
    ax.set_title('HTO', fontweight='bold')
    ax.set_ylabel('%')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    for hist, name, color in zip(histories, names, colors):
        ax.plot(hist['t'], hist['T_s'], color=color, linewidth=2, label=name)
    ax.axhline(y=90, color='r', linestyle='--', alpha=0.5, label='T_max')
    ax.axhline(y=80, color='g', linestyle='--', alpha=0.5, label='T_target')
    ax.set_title('Stack Temperature', fontweight='bold')
    ax.set_ylabel('°C')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Row 2
    ax = fig.add_subplot(gs[1, 0])
    for hist, name, color in zip(histories, names, colors):
        ax.plot(hist['t'], hist['I'], color=color, linewidth=2, label=name)
    ax.set_title('Current', fontweight='bold')
    ax.set_ylabel('A')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    for hist, name, color in zip(histories, names, colors):
        ax.plot(hist['t'], np.array(hist['v_lye']) * 1000, color=color, linewidth=2, label=name)
    ax.set_title('Lye Flow', fontweight='bold')
    ax.set_ylabel('L/s')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 2])
    for hist, name, color in zip(histories, names, colors):
        ax.plot(hist['t'], np.array(hist['v_c']) * 1000, color=color, linewidth=2, label=name)
    ax.set_title('Coolant Flow', fontweight='bold')
    ax.set_ylabel('L/s')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Row 3: Violations
    ax = fig.add_subplot(gs[2, 0])
    for i, (hist, name, color) in enumerate(zip(histories, names, colors)):
        viol = np.array(hist['T_violation'], dtype=float)
        if viol.sum() > 0:
            ax.fill_between(hist['t'], i * 0.3, i * 0.3 + viol * 0.25, alpha=0.6, color=color, label=f'{name}')
    ax.set_title('Temperature Violations', fontweight='bold')
    ax.set_ylabel('Controller')
    ax.set_xlabel('Time (min)')
    ax.set_yticks([0.125, 0.425, 0.725])
    ax.set_yticklabels(names)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, 1])
    for i, (hist, name, color) in enumerate(zip(histories, names, colors)):
        viol = np.array(hist['P_violation'], dtype=float)
        if viol.sum() > 0:
            ax.fill_between(hist['t'], i * 0.3, i * 0.3 + viol * 0.25, alpha=0.6, color=color, label=f'{name}')
    ax.set_title('Power Violations', fontweight='bold')
    ax.set_ylabel('Controller')
    ax.set_xlabel('Time (min)')
    ax.set_yticks([0.125, 0.425, 0.725])
    ax.set_yticklabels(names)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, 2])
    for i, (hist, name, color) in enumerate(zip(histories, names, colors)):
        viol = np.array(hist['HTO_violation'], dtype=float)
        if viol.sum() > 0:
            ax.fill_between(hist['t'], i * 0.3, i * 0.3 + viol * 0.25, alpha=0.6, color=color, label=f'{name}')
    ax.set_title('HTO Violations', fontweight='bold')
    ax.set_ylabel('Controller')
    ax.set_xlabel('Time (min)')
    ax.set_yticks([0.125, 0.425, 0.725])
    ax.set_yticklabels(names)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Row 4: Summary bar chart
    ax = fig.add_subplot(gs[3, :])
    metrics = ['Max T (°C)', 'Max P (MW)', 'Max HTO (%)', 'T Viol', 'P Viol', 'HTO Viol']
    x = np.arange(len(metrics))
    width = 0.25

    for j, (hist, name, color) in enumerate(zip(histories, names, colors)):
        vals = [
            max(hist['T_s']),
            max(hist['P_real']),
            max(hist['HTO']),
            sum(hist['T_violation']),
            sum(hist['P_violation']),
            sum(hist['HTO_violation'])
        ]
        ax.bar(x + j * width, vals, width, label=name, color=color, alpha=0.8, edgecolor='black')

    ax.axhline(y=90, color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=6, color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=2, color='r', linestyle='--', alpha=0.5)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.set_title('Summary Metrics Comparison', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(
        'TCN Diffusion + CBF: Single-Stack Safety Comparison\n'
        'Model: diffusion_tcn | CBF Params: gamma=[3,2,100,100,5], h_margin_HTO=0.002',
        fontsize=14, fontweight='bold'
    )

    out_path = os.path.join(output_dir, 'diffusion_tcn_cbf_comparison.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {out_path}")


def main():
    output_dir = 'output/single_stack/cbf_model_tests'
    os.makedirs(output_dir, exist_ok=True)

    # Verify model exists
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    model_path = os.path.join(base_dir, 'output', 'single_stack', 'policy', 'diffusion_tcn_policy_best.pth')
    stats_path = os.path.join(base_dir, 'output', 'single_stack', 'diffusion_stats.npz')

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
    if not os.path.exists(stats_path):
        print(f"Stats not found: {stats_path}")
        return

    print(f"Model: {model_path}")
    print(f"Stats: {stats_path}")

    # Initial state (warm start around 80°C)
    initial_state = np.array([
        345.0, 353.15, 353.15, 325.0,
        0.52, 0.52, 0.52
    ])

    # Test scenario: aggressive power ramp that pushes boundaries
    P_ref_profile = [
        (0, 600, 1.0e6),
        (600, 1800, 2.5e6),
        (1800, 3000, 4.0e6),
        (3000, 4200, 5.5e6),
        (4200, 5400, 3.0e6),
        (5400, 7200, 1.5e6),
    ]

    print("\n" + "=" * 70)
    print("TCN Diffusion + CBF Safety Test")
    print("=" * 70)

    # 1. Raw TCN Diffusion
    print("\n[1/3] Loading raw TCN Diffusion (no safety)...")
    ctrl_basic = SingleStackModelController(
        dt=60.0, horizon=5, model_type='diffusion_tcn',
        model_path=model_path, stats_path=stats_path
    )
    hist_basic = run_test(ctrl_basic, "TCN Diffusion (raw)", initial_state, P_ref_profile)

    # 2. Safe Projection
    print("\n[2/3] Loading TCN Diffusion + Safe Projection...")
    ctrl_safe = SingleStackSafeModelController(
        dt=60.0, horizon=5, model_type='diffusion_tcn',
        model_path=model_path, stats_path=stats_path,
        use_safe_projection=True
    )
    hist_safe = run_test(ctrl_safe, "TCN Diffusion + Safe", initial_state, P_ref_profile)

    # 3. CBF Projection (tuned params)
    print("\n[3/3] Loading TCN Diffusion + CBF Projection...")
    ctrl_cbf = SingleStackCBFModelController(
        dt=60.0, horizon=5, model_type='diffusion_tcn',
        model_path=model_path, stats_path=stats_path,
        use_cbf_projection=True,
        gamma_vec=[3.0, 2.0, 100.0, 100.0, 5.0],
        rho_vec=[50000, 50000, 50000, 50000, 10000],
        h_margin_vec=[1.0, 0.002, 0.0, 0.0, 0.0],
        lambda_u_scale=[500.0, 200.0, 1000.0],
        soft_mask=[False, True, True, False, True],
        normalize=True,
    )
    hist_cbf = run_test(ctrl_cbf, "TCN Diffusion + CBF", initial_state, P_ref_profile)

    # Plot
    plot_results([hist_basic, hist_safe, hist_cbf],
                 ["Raw", "Safe", "CBF"], output_dir)

    print("\n" + "=" * 70)
    print("Test complete!")
    print(f"Output: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
