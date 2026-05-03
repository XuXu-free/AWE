"""
一阶CBF vs 二阶HOCBF HTO安全实验对比
实验条件：初始3000A稳态，t=60min阶跃降至1000A
参数：alpha1=1, alpha2=2, rho=5000（与论文一致）
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from controller.multi_stack.cbf_projection import MultiStackCBFProjection
from controller.multi_stack.cbf_projection_ho import MultiStackCBFProjectionHO
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
               'projection_success': [], 'u_ref': []}

    for i in range(steps):
        t = i * dt
        if i == step_step:
            u_ref = np.array([I_after] * 4 + [0.03] * 4 + [0.03])

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
            history['u_ref'].append(u_ref[0])

        sim.step(current_action)
    return history


def smooth(y, window=51, poly=3):
    from scipy.signal import savgol_filter
    if len(y) < window:
        return y
    return savgol_filter(y, window, poly)


def mark_projection_trigger(ax, t_arr, success_arr, color='#ffcccc', alpha=0.3):
    """Mark CBF projection trigger regions with light background"""
    # CBF is triggered when projection modifies the reference (success=True means constraint active)
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

    # 一阶CBF：不使用 u_weight_scale，展示其固有震荡缺陷
    print("Running first-order CBF (baseline, no u_weight_scale)...")
    cbf1 = MultiStackCBFProjection(
        dt=60.0,
        gamma_vec=[10.0] * 4 + [1.0] + [10.0] * 4,
        rho_vec=[5000.0] * 9,
        h_margin_vec=[0.0] * 4 + [0.001] + [0.0] * 4,
        normalize=True,
        lambda_u_scale=0.0,
    )
    h1 = run_step_experiment(cbf1, initial_state, 3000.0, 1000.0, step_time_min=60, total_min=180)

    # 二阶HOCBF：混合阶框架，温度通道近似一阶（高alpha），HTO通道二阶
    print("Running second-order HOCBF (with u_weight_scale)...")
    cbf2 = MultiStackCBFProjectionHO(
        dt=60.0,
        gamma_vec=[10.0] * 4 + [5.0] + [10.0] * 4,
        rho_vec=[5000.0] * 9,
        h_margin_vec=[0.0] * 4 + [0.001] + [0.0] * 4,
        normalize=True,
        lambda_u_scale=0.0,
        alpha1_vec=[20.0, 20.0, 20.0, 20.0, 10.0, 20.0, 20.0, 20.0, 20.0],
        alpha2_vec=[20.0] * 4 + [2.0] + [20.0] * 4,
        u_weight_scale=[1000.0, 1000.0, 1000.0, 1000.0, 20.0, 20.0, 20.0, 20.0, 0.1],
    )
    h2 = run_step_experiment(cbf2, initial_state, 3000.0, 1000.0, step_time_min=60, total_min=180)

    # 绘图 - 参考图3.9布局
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    fig.subplots_adjust(hspace=0.32, wspace=0.28, left=0.07, right=0.97, top=0.90, bottom=0.08)
    # 总标题由 LaTeX \caption 控制，此处不设置 suptitle

    t1 = np.array(h1['t'])
    t2 = np.array(h2['t'])

    def setup_ax(ax):
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
        ax.tick_params(axis='both', which='both', length=3)

    # (a) 电解槽电流
    ax = axes[0, 0]
    I1 = np.array(h1['I_all']).sum(axis=1) / 1000
    I2 = np.array(h2['I_all']).sum(axis=1) / 1000
    ax.plot(t1, I1, color='#d62728', linewidth=1.5, label='一阶CBF')
    ax.plot(t2, I2, color='#1f77b4', linewidth=1.5, label='二阶HOCBF')
    I_ref_arr = np.where(t1 < 60, 12.0, 4.0)
    ax.plot(t1, I_ref_arr, color='k', linestyle='--', linewidth=1.0, alpha=0.5, label='参考电流')
    ax.text(0.02, 0.98, '(a)', transform=ax.transAxes, fontsize=14, va='top', ha='left')
    ax.set_ylabel('电流 (kA)')
    ax.legend(loc='best', frameon=True)
    setup_ax(ax)

    # (b) 碱液流量
    ax = axes[0, 1]
    v1 = np.array(h1['v_lye_all']) * 1000
    v2 = np.array(h2['v_lye_all']) * 1000
    ax.plot(t1, smooth(v1[:, 0]), color='#d62728', linewidth=1.5, label='一阶CBF')
    ax.plot(t2, smooth(v2[:, 0]), color='#1f77b4', linewidth=1.5, label='二阶HOCBF')
    ax.plot(t1, np.full_like(t1, 30.0), color='k', linestyle='--', linewidth=1.0, alpha=0.5, label='参考流量')
    ax.text(0.02, 0.98, '(b)', transform=ax.transAxes, fontsize=14, va='top', ha='left')
    ax.set_ylabel('流量 (L/s)')
    ax.legend(loc='best', frameon=True)
    setup_ax(ax)

    # (c) 冷却水流量
    ax = axes[1, 0]
    vc1 = np.array(h1['v_c']) * 1000
    vc2 = np.array(h2['v_c']) * 1000
    ax.plot(t1, vc1, color='#d62728', linewidth=1.5, label='一阶CBF')
    ax.plot(t2, vc2, color='#1f77b4', linewidth=1.5, label='二阶HOCBF')
    ax.plot(t1, np.full_like(t1, 30.0), color='k', linestyle='--', linewidth=1.0, alpha=0.5, label='参考流量')
    ax.text(0.02, 0.98, '(c)', transform=ax.transAxes, fontsize=14, va='top', ha='left')
    ax.set_ylabel('流量 (L/s)')
    ax.set_xlabel('时间 (min)')
    ax.legend(loc='best', frameon=True)
    setup_ax(ax)

    # (d) HTO安全控制
    ax = axes[1, 1]
    ax.plot(t1, smooth(np.array(h1['HTO'])), color='#d62728', linewidth=1.5, label='一阶CBF')
    ax.plot(t2, smooth(np.array(h2['HTO'])), color='#1f77b4', linewidth=1.5, label='二阶HOCBF')
    ax.axhline(2.0, color='r', linestyle='--', linewidth=1.5, label='安全限 (2%)')
    ax.text(0.02, 0.98, '(d)', transform=ax.transAxes, fontsize=14, va='top', ha='left')
    ax.set_ylabel('HTO (%)')
    ax.set_xlabel('时间 (min)')
    ax.legend(loc='best', frameon=True)
    setup_ax(ax)
    ax.set_ylim([0, 2.3])
    ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0])

    out_path = '../figures/cbf_order_comparison.png'
    plt.savefig(out_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out_path}")

    # 打印指标
    for name, h in [('一阶CBF', h1), ('二阶HOCBF', h2)]:
        I = np.array(h['I_all'])
        dI = np.diff(I, axis=0)
        v = np.array(h['v_lye_all']) * 1000
        dv = np.diff(v, axis=0)
        print(f"\n{name}:")
        print(f"  HTO max: {max(h['HTO']):.3f}%")
        print(f"  Temp max: {max(np.array(h['T_s_all']).flatten()):.2f}°C")
        print(f"  |dI| mean: {np.mean(np.abs(dI)):.1f} A")
        print(f"  |dI| max: {np.max(np.abs(dI)):.1f} A")
        print(f"  |dv_lye| mean: {np.mean(np.abs(dv)):.2f} L/s")
        print(f"  |dv_lye| max: {np.max(np.abs(dv)):.2f} L/s")


if __name__ == "__main__":
    main()
