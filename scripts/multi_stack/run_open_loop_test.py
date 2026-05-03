"""
多槽系统开环测试 (Open-Loop Simulation)
参考论文开环仿真设置:
  - 先运行较长稳态阶段使系统达到稳态
  - 初始条件: T_stack = 85°C (358 K), T_sep = 73°C (345 K), HTO = 0.52%
  - Action I  (t=900s):  Stack 1 lye flow: 3.35e-2 -> 2.5e-2 m3/s
  - Action II (t=1800s): Stack 1 current: 7000 -> 3500 A
  - Action III(t=2700s): Stack 2 current: 7000 -> 3500 A
  - Action IV (t=3600s): Stack 3 current: 7000 -> 3500 A
  - Action V  (t=4500s): Coolant flow: 2.2e-3 -> 1.0e-3 m3/s
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from plant.multi_stack_simulator import MultiStackSimulator


def run_open_loop_simulation():
    sim = MultiStackSimulator(dt=0.2)

    # Override initial conditions to match paper
    sim.T_s_init = 358.0      # 85 C
    sim.T_sep_init = 345.0    # 73 C (paper says 73C = 345K)
    sim.T_s_in_init = 345.0   # Approximate
    sim.T_c_out_init = 325.0
    sim.HTO_init = 0.52       # %

    sim.reset()
    state = sim.state.copy()
    print("Paper initial state:")
    print(f"  T_s_in = {state[0]:.2f} K")
    print(f"  T_s = {state[1:5]} K")
    print(f"  T_sep = {state[5]:.2f} K")
    print(f"  T_c_out = {state[6]:.2f} K")
    hto_init = (state[12] * sim.R * state[5]) / (sim.P_sys * sim.V_sep_gas) * 100
    print(f"  HTO = {hto_init:.4f} %")

    # Pre-steady phase: run with initial control inputs for a long time
    dt = 0.2
    pre_steady_duration = 14400.0  # 4 hours
    pre_steps = int(pre_steady_duration / dt)

    I = np.array([7000.0, 7000.0, 7000.0, 7000.0])
    v_lye = np.array([3.35e-2, 3.35e-2, 3.35e-2, 3.35e-2])
    v_c = 5.0e-2
    action = np.concatenate([I, v_lye, [v_c]])

    print(f"\nRunning pre-steady phase for {pre_steady_duration}s ({pre_steady_duration/3600:.1f}h)...")
    for i in range(pre_steps):
        state = sim.step(action)

    print("Pre-steady state:")
    print(f"  T_s_in = {state[0]:.2f} K ({state[0]-273.15:.2f} C)")
    print(f"  T_s = {state[1:5]-273.15} C")
    print(f"  T_sep = {state[5]:.2f} K ({state[5]-273.15:.2f} C)")
    print(f"  T_c_out = {state[6]:.2f} K ({state[6]-273.15:.2f} C)")
    hto_steady = (state[12] * sim.R * state[5]) / (sim.P_sys * sim.V_sep_gas) * 100
    print(f"  HTO = {hto_steady:.4f} %")

    # Open-loop test phase (start from steady state)
    test_duration = 7200.0  # 2 hours
    n_steps = int(test_duration / dt)

    # Reset action values
    I = np.array([7000.0, 7000.0, 7000.0, 7000.0])
    v_lye = np.array([3.35e-2, 3.35e-2, 3.35e-2, 3.35e-2])
    v_c = 5.0e-2

    t_log = []
    T_s_log = []
    T_sep_log = []
    T_s_in_log = []
    T_c_out_log = []
    HTO_log = []
    I_log = []
    v_lye_log = []
    v_c_log = []
    P_real_log = []

    for i in range(n_steps):
        t = i * dt

        # Action I at t=900s: Stack 1 lye flow down
        if t >= 900.0:
            v_lye[0] = 2.5e-2

        # Action II at t=1800s: Stack 1 current down
        if t >= 1800.0:
            I[0] = 3500.0

        # Action III at t=2700s: Stack 2 current down
        if t >= 2700.0:
            I[1] = 3500.0

        # Action IV at t=3600s: Stack 3 current down
        if t >= 3600.0:
            I[2] = 3500.0

        # Action V at t=4500s: Coolant flow down
        if t >= 4500.0:
            v_c = 2.0e-2

        action = np.concatenate([I, v_lye, [v_c]])
        state = sim.step(action)

        # Log every 10s
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
            T_s_in_log.append(state[0] - 273.15)
            T_c_out_log.append(state[6] - 273.15)
            HTO_log.append(hto_pct)
            I_log.append(I.copy())
            v_lye_log.append(v_lye.copy() * 1e3)  # L/s
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

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ax = axes[0, 0]
    for j in range(4):
        ax.plot(t_log, I_log[:, j], label=f'Stack {j+1}')
    for t_action in [900, 1800, 2700, 3600]:
        ax.axvline(x=t_action / 60.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_title('(a) 电解槽电流')
    ax.set_xlabel('时间 (min)')
    ax.set_ylabel('电流 (A)')
    ax.legend()
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)

    ax = axes[0, 1]
    for j in range(4):
        ax.plot(t_log, v_lye_log[:, j], label=f'Stack {j+1}')
    ax.axvline(x=900 / 60.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_title('(b) 碱液流量')
    ax.set_xlabel('时间 (min)')
    ax.set_ylabel('流量 (L/s)')
    ax.legend()
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(t_log, v_c_log, 'c-', label='冷却水')
    ax.axvline(x=4500 / 60.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_title('(c) 冷却水流量')
    ax.set_xlabel('时间 (min)')
    ax.set_ylabel('流量 (L/s)')
    ax.legend()
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)

    ax = axes[1, 0]
    for j in range(4):
        ax.plot(t_log, T_s_log[:, j], label=f'Stack {j+1}')
    ax.plot(t_log, T_sep_log, 'g--', label='Separator', linewidth=1.5)
    for t_action in [900, 1800, 2700, 3600, 4500]:
        ax.axvline(x=t_action / 60.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_title('(d) 电解槽与分离器温度')
    ax.set_xlabel('时间 (min)')
    ax.set_ylabel('温度 (°C)')
    ax.legend(ncol=2)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t_log, HTO_log, 'm-')
    ax.axhline(y=hto_steady, color='gray', linestyle=':', alpha=0.5, label=f'稳态值 ({hto_steady:.3f}%)')
    for t_action in [900, 1800, 2700, 3600, 4500]:
        ax.axvline(x=t_action / 60.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_title('(e) HTO 浓度')
    ax.set_xlabel('时间 (min)')
    ax.set_ylabel('HTO (%)')
    ax.legend()
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)

    ax = axes[1, 2]
    ax.plot(t_log, P_real_log, 'b-')
    for t_action in [900, 1800, 2700, 3600, 4500]:
        ax.axvline(x=t_action / 60.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_title('(f) 总功率')
    ax.set_xlabel('时间 (min)')
    ax.set_ylabel('功率 (MW)')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)

    plt.tight_layout(pad=2.0)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    output_dir = os.path.join(project_root, 'output', 'multi_stack', 'open_loop')
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'open_loop_simulation.png')
    plt.savefig(plot_path, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"\nPlot saved: {plot_path}")
    plt.close()


if __name__ == '__main__':
    run_open_loop_simulation()
