"""
一阶CBF HTO约束保护实验图（Figure 3.10）
实验条件：初始3000A稳态，t=60min阶跃降至1000A
参数：gamma=1, rho=5000/15000, h_margin=0.001, lambda_u=0
风格：2x2子图布局，带CBF触发区域标记
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from controller.multi_stack.cbf_projection import MultiStackCBFProjection
from plant.multi_stack_simulator import MultiStackSimulator

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11


def run_steady_state(sim, I_target, duration=7200):
    """Run to steady state with fixed current"""
    u = np.array([I_target] * 4 + [0.05] * 4 + [0.01])
    steps = int(duration / sim.dt)
    for _ in range(steps):
        sim.step(u)
    return sim.get_state().copy()


def run_step_experiment(projector, initial_state, I_before, I_after, step_time_min=60, total_min=180):
    sim = MultiStackSimulator(dt=0.2)
    sim.reset(initial_state=initial_state)
    dt = 0.2
    steps = int(total_min * 60 / dt)
    ctrl_steps = int(60.0 / dt)
    step_step = int(step_time_min * 60 / dt)

    u_ref = np.array([I_before] * 4 + [0.03] * 4 + [0.03])
    current_action = u_ref.copy()
    last_action = current_action.copy()

    history = {'t': [], 'I_all': [], 'v_lye_all': [], 'v_c': [], 'HTO': [], 'T_s_all': [],
               'projection_success': [], 'u_ref': [], 'cbf_triggered': []}

    last_triggered = False
    for i in range(steps):
        t = i * dt
        if i == step_step:
            u_ref = np.array([I_after] * 4 + [0.03] * 4 + [0.03])

        if i % ctrl_steps == 0:
            state = sim.state
            current_action, success, info = projector.project(u_ref, state, u_last=last_action)
            last_action = current_action.copy()

            # Determine if CBF was triggered (action was modified)
            adjustment = float(np.linalg.norm(current_action - u_ref))
            last_triggered = adjustment > 1e-6

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
            history['u_ref'].append(u_ref[0])
            history['cbf_triggered'].append(last_triggered)

        sim.step(current_action)
    return history


def smooth(y, window=51, poly=3):
    from scipy.signal import savgol_filter
    if len(y) < window:
        return y
    return savgol_filter(y, window, poly)


def mark_projection_trigger(ax, t_arr, success_arr, color='#ffcccc', alpha=0.3):
    """Mark CBF projection trigger regions with light background"""
    triggered = False
    start = None
    for i, (t, s) in enumerate(zip(t_arr, success_arr)):
        if s and not triggered:
            triggered = True
            start = t
        elif not s and triggered:
            triggered = False
            if start is not None and t > start:
                ax.axvspan(start, t, alpha=alpha, color=color, zorder=0)
    if triggered and start is not None:
        ax.axvspan(start, t_arr[-1], alpha=alpha, color=color, zorder=0)


def main():
    # 预热到3000A稳态
    print("Warming up to 3000A steady state...")
    sim_warm = MultiStackSimulator(dt=0.2)
    sim_warm.reset()
    initial_state = run_steady_state(sim_warm, 3000.0, duration=7200)
    print(f"Warm-up complete. T_s = {initial_state[1:5] - 273.15}")

    # 一阶CBF：调优后的参数配置
    print("Running first-order CBF step experiment...")
    cbf = MultiStackCBFProjection(
        dt=60.0,
        gamma_vec=[10.0] * 4 + [1.0] + [10.0] * 4,
        rho_vec=[5000.0] * 4 + [15000.0] + [5000.0] * 4,
        h_margin_vec=[0.0] * 4 + [0.001] + [0.0] * 4,
        normalize=True,
        lambda_u_scale=0.0,
        u_weight_scale=[1000.0, 1000.0, 1000.0, 1000.0, 5.0, 5.0, 5.0, 5.0, 0.1],
    )
    h = run_step_experiment(cbf, initial_state, 3000.0, 1000.0, step_time_min=60, total_min=180)

    # 绘图 - 2x2子图布局
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    fig.subplots_adjust(hspace=0.32, wspace=0.28, left=0.07, right=0.97, top=0.90, bottom=0.12)

    t = np.array(h['t'])
    cbf_triggered = np.array(h['cbf_triggered'])

    def setup_ax(ax):
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
        ax.tick_params(axis='both', which='both', length=3)

    # CBF触发区域背景色
    cbf_patch = Patch(facecolor='#ffcccc', alpha=0.5, edgecolor='none', label='CBF触发')

    # (a) 电解槽电流
    ax = axes[0, 0]
    I = np.array(h['I_all']).sum(axis=1) / 1000
    ax.plot(t, I, color='#1f77b4', linewidth=1.5, label='实际电流')
    I_ref_arr = np.where(t < 60, 12.0, 4.0)
    ax.plot(t, I_ref_arr, color='k', linestyle='--', linewidth=1.0, alpha=0.5, label='参考电流')
    mark_projection_trigger(ax, t, cbf_triggered)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(cbf_patch)
    labels.append('CBF触发')
    ax.legend(handles=handles, labels=labels, loc='best', frameon=True)
    ax.set_ylabel('电流 (kA)')
    ax.set_xlabel('时间 (min)')
    setup_ax(ax)

    # (b) 碱液流量
    ax = axes[0, 1]
    v = np.array(h['v_lye_all']) * 1000
    ax.plot(t, smooth(v[:, 0]), color='#1f77b4', linewidth=1.5, label='槽1')
    ax.plot(t, smooth(v[:, 1]), color='#ff7f0e', linewidth=1.5, label='槽2')
    ax.plot(t, smooth(v[:, 2]), color='#2ca02c', linewidth=1.5, label='槽3')
    ax.plot(t, smooth(v[:, 3]), color='#d62728', linewidth=1.5, label='槽4')
    mark_projection_trigger(ax, t, cbf_triggered)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(cbf_patch)
    labels.append('CBF触发')
    ax.legend(handles=handles, labels=labels, loc='upper right', frameon=True, ncol=2)
    ax.set_ylabel('流量 (L/s)')
    ax.set_xlabel('时间 (min)')
    setup_ax(ax)
    ax.set_ylim([0, 50])
    ax.set_yticks([0, 10, 20, 30, 40, 50])

    # (c) 冷却水流量
    ax = axes[1, 0]
    vc = np.array(h['v_c']) * 1000
    ax.plot(t, vc, color='#1f77b4', linewidth=1.5, label='实际流量')
    ax.plot(t, np.full_like(t, 30.0), color='k', linestyle='--', linewidth=1.0, alpha=0.5, label='参考流量')
    mark_projection_trigger(ax, t, cbf_triggered)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(cbf_patch)
    labels.append('CBF触发')
    ax.legend(handles=handles, labels=labels, loc='lower right', frameon=True)
    ax.set_ylabel('流量 (L/s)')
    ax.set_xlabel('时间 (min)')
    setup_ax(ax)
    ax.set_ylim([28, 35])
    ax.set_yticks([29, 30, 31, 32, 33, 34])

    # (d) HTO安全控制
    ax = axes[1, 1]
    ax.plot(t, smooth(np.array(h['HTO'])), color='#1f77b4', linewidth=1.5, label='HTO')
    ax.axhline(2.0, color='r', linestyle='--', linewidth=1.5, label='安全限 (2%)')
    mark_projection_trigger(ax, t, cbf_triggered)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(cbf_patch)
    labels.append('CBF触发')
    ax.legend(handles=handles, labels=labels, loc='upper right', frameon=True)
    ax.set_ylabel('HTO (%)')
    ax.set_xlabel('时间 (min)')
    setup_ax(ax)
    ax.set_ylim([0, 2.3])
    ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0])

    out_path = '../figures/cbf_hto_protection.png'
    plt.savefig(out_path, dpi=600, bbox_inches='tight', facecolor='white')

    # 保存各子图为独立PNG（供LaTeX subfigure环境使用）
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for idx, ax in enumerate(axes.flat):
        label = chr(ord('a') + idx)
        bbox = ax.get_tightbbox(renderer)
        bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())
        bbox_inches = bbox_inches.expanded(1.02, 1.02)
        fig.savefig(f'../figures/cbf_hto_protection_{label}.png', dpi=600, bbox_inches=bbox_inches, facecolor='white')
        print(f'Saved: ../figures/cbf_hto_protection_{label}.png')
    plt.close()
    print(f"Saved: {out_path}")

    # 打印指标
    I_arr = np.array(h['I_all'])
    dI = np.diff(I_arr, axis=0)
    v_arr = np.array(h['v_lye_all']) * 1000
    dv = np.diff(v_arr, axis=0)
    print(f"\n一阶CBF实验指标:")
    print(f"  HTO max: {max(h['HTO']):.3f}%")
    print(f"  Temp max: {max(np.array(h['T_s_all']).flatten()):.2f}°C")
    print(f"  |dI| mean: {np.mean(np.abs(dI)):.1f} A")
    print(f"  |dI| max: {np.max(np.abs(dI)):.1f} A")
    print(f"  |dv_lye| mean: {np.mean(np.abs(dv)):.2f} L/s")
    print(f"  |dv_lye| max: {np.max(np.abs(dv)):.2f} L/s")
    cbf_trigger_count = sum(h['cbf_triggered'])
    print(f"  CBF触发次数: {cbf_trigger_count} / {len(h['cbf_triggered'])} ({cbf_trigger_count/len(h['cbf_triggered'])*100:.1f}%)")


if __name__ == "__main__":
    main()
