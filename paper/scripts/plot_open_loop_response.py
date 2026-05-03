"""
多槽系统开环测试图（Figure 3-4）
风格：SJTU论文标准，提取单张子图供LaTeX subfigure环境使用
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from plant.multi_stack_simulator import MultiStackSimulator

# ========== 全局配置 ==========
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


def run_open_loop_simulation():
    sim = MultiStackSimulator(dt=0.2)

    # Override initial conditions to match paper
    sim.T_s_init = 358.0
    sim.T_sep_init = 345.0
    sim.T_s_in_init = 345.0
    sim.T_c_out_init = 325.0
    sim.HTO_init = 0.52

    sim.reset()
    state = sim.state.copy()
    print("Paper initial state:")
    print(f"  T_s_in = {state[0]:.2f} K")
    print(f"  T_s = {state[1:5]} K")
    print(f"  T_sep = {state[5]:.2f} K")
    print(f"  T_c_out = {state[6]:.2f} K")
    hto_init = (state[12] * sim.R * state[5]) / (sim.P_sys * sim.V_sep_gas) * 100
    print(f"  HTO = {hto_init:.4f} %")

    dt = 0.2
    pre_steady_duration = 14400.0
    pre_steps = int(pre_steady_duration / dt)

    I = np.array([7000.0, 7000.0, 7000.0, 7000.0])
    v_lye = np.array([3.35e-2, 3.35e-2, 3.35e-2, 3.35e-2])
    v_c = 5.0e-2
    action = np.concatenate([I, v_lye, [v_c]])

    print(f"\nRunning pre-steady phase for {pre_steady_duration}s ({pre_steady_duration/3600:.1f}h)...")
    for i in range(pre_steps):
        state = sim.step(action)

    print("Pre-steady state:")
    print(f"  T_s = {state[1:5]-273.15} C")
    print(f"  T_sep = {state[5]:.2f} C")
    hto_steady = (state[12] * sim.R * state[5]) / (sim.P_sys * sim.V_sep_gas) * 100
    print(f"  HTO = {hto_steady:.4f} %")

    test_duration = 7200.0
    n_steps = int(test_duration / dt)

    I = np.array([7000.0, 7000.0, 7000.0, 7000.0])
    v_lye = np.array([3.35e-2, 3.35e-2, 3.35e-2, 3.35e-2])
    v_c = 5.0e-2

    t_log = []
    T_s_log = []
    T_sep_log = []
    HTO_log = []
    I_log = []
    v_lye_log = []
    v_c_log = []
    P_real_log = []

    for i in range(n_steps):
        t = i * dt
        if t >= 900.0:
            v_lye[0] = 2.5e-2
        if t >= 1800.0:
            I[0] = 3500.0
        if t >= 2700.0:
            I[1] = 3500.0
        if t >= 3600.0:
            I[2] = 3500.0
        if t >= 4500.0:
            v_c = 2.0e-2

        action = np.concatenate([I, v_lye, [v_c]])
        state = sim.step(action)

        if i % 50 == 0:
            T_s = state[1:5]
            T_sep = state[5]
            n_H2_sep_gas = state[12]
            hto_pct = (n_H2_sep_gas * sim.R * T_sep) / (sim.P_sys * sim.V_sep_gas) * 100
            _, U_cell, _ = sim._calculate_electrochemical_properties(I, T_s)
            P_real = np.sum(U_cell * I * sim.N_cell)

            t_log.append(t / 60.0)
            T_s_log.append(T_s - 273.15)
            T_sep_log.append(T_sep - 273.15)
            HTO_log.append(hto_pct)
            I_log.append(I.copy())
            v_lye_log.append(v_lye.copy() * 1e3)
            v_c_log.append(v_c * 1e3)
            P_real_log.append(P_real / 1e6)

    t_log = np.array(t_log)
    T_s_log = np.array(T_s_log)
    T_sep_log = np.array(T_sep_log)
    HTO_log = np.array(HTO_log)
    I_log = np.array(I_log)
    v_lye_log = np.array(v_lye_log)
    v_c_log = np.array(v_c_log)
    P_real_log = np.array(P_real_log)

    print("\nFinal state (t=7200s):")
    print(f"  T_s = {T_s_log[-1]} C")
    print(f"  T_sep = {T_sep_log[-1]:.2f} C")
    print(f"  HTO = {HTO_log[-1]:.4f} %")

    # ========== 绘图 ==========
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    action_times = [900, 1800, 2700, 3600, 4500]
    action_times_min = [t / 60.0 for t in action_times]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    fig.subplots_adjust(hspace=0.32, wspace=0.28, left=0.07, right=0.97, top=0.90, bottom=0.12)

    def setup_ax(ax):
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
        ax.tick_params(axis='both', which='both', length=3)

    # (a) 电解槽电流
    ax = axes[0, 0]
    for j in range(4):
        ax.plot(t_log, I_log[:, j], color=colors[j], linewidth=1.5, label=f'槽{j+1}')
    for ta in action_times_min[:-1]:
        ax.axvline(x=ta, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylabel('电流 (A)')
    ax.set_xlabel('时间 (min)')
    ax.legend(loc='best', frameon=True)
    setup_ax(ax)

    # (b) 碱液流量
    ax = axes[0, 1]
    for j in range(4):
        ax.plot(t_log, v_lye_log[:, j], color=colors[j], linewidth=1.5, label=f'槽{j+1}')
    ax.axvline(x=action_times_min[0], color='gray', linestyle=':', alpha=0.5)
    ax.set_ylabel('流量 (L/s)')
    ax.set_xlabel('时间 (min)')
    ax.legend(loc='best', frameon=True)
    setup_ax(ax)

    # (c) 冷却水流量
    ax = axes[0, 2]
    ax.plot(t_log, v_c_log, color=colors[0], linewidth=1.5, label='冷却水')
    ax.axvline(x=action_times_min[-1], color='gray', linestyle=':', alpha=0.5)
    ax.set_ylabel('流量 (L/s)')
    ax.set_xlabel('时间 (min)')
    ax.legend(loc='best', frameon=True)
    setup_ax(ax)

    # (d) 电解槽与分离器温度
    ax = axes[1, 0]
    for j in range(4):
        ax.plot(t_log, T_s_log[:, j], color=colors[j], linewidth=1.5, label=f'槽{j+1}')
    ax.plot(t_log, T_sep_log, color='k', linestyle='--', linewidth=1.2, alpha=0.7, label='分离器')
    for ta in action_times_min:
        ax.axvline(x=ta, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylabel('温度 (°C)')
    ax.set_xlabel('时间 (min)')
    ax.legend(loc='best', frameon=True, ncol=2)
    setup_ax(ax)

    # (e) HTO浓度
    ax = axes[1, 1]
    ax.plot(t_log, HTO_log, color='#d62728', linewidth=1.5, label='HTO')
    ax.axhline(y=hto_steady, color='gray', linestyle=':', alpha=0.5, label=f'稳态值 ({hto_steady:.3f}%)')
    for ta in action_times_min:
        ax.axvline(x=ta, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylabel('HTO (%)')
    ax.set_xlabel('时间 (min)')
    ax.legend(loc='best', frameon=True)
    setup_ax(ax)

    # (f) 总功率
    ax = axes[1, 2]
    ax.plot(t_log, P_real_log, color='#1f77b4', linewidth=1.5, label='实际功率')
    for ta in action_times_min:
        ax.axvline(x=ta, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylabel('功率 (MW)')
    ax.set_xlabel('时间 (min)')
    ax.legend(loc='best', frameon=True)
    setup_ax(ax)

    # 统一x轴
    for ax in axes.flat:
        ax.set_xlim([t_log.min(), t_log.max()])

    # 保存整图
    plt.savefig('../figures/open_loop_response.png', dpi=600, bbox_inches='tight', facecolor='white')
    print('Saved: ../figures/open_loop_response.png')

    # 保存各子图为独立PNG（供LaTeX subfigure环境使用）
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for idx, ax in enumerate(axes.flat):
        label = chr(ord('a') + idx)
        bbox = ax.get_tightbbox(renderer)
        bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())
        bbox_inches = bbox_inches.expanded(1.02, 1.02)
        fig.savefig(f'../figures/open_loop_response_{label}.png', dpi=600, bbox_inches=bbox_inches, facecolor='white')
        print(f'Saved: ../figures/open_loop_response_{label}.png')
    plt.close()


if __name__ == '__main__':
    run_open_loop_simulation()
