"""
快速测试多槽模型控制器（10分钟测试）
"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import json
from tqdm import tqdm
import time

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from plant.multi_stack_simulator import MultiStackSimulator
from controller.multi_stack.nmpc_controller import MultiStackNMPCController
from controller.multi_stack.model_controller import MultiStackModelController


def run_quick_test(controller_type, model_type, duration=600):
    """快速测试（默认10分钟）"""
    sim_dt = 0.2
    dt_ctrl = 60.0
    horizon = 5
    T_ref = 353.15

    sim = MultiStackSimulator(dt=sim_dt)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    stats_path = os.path.join(project_root, 'output', 'multi_stack', 'diffusion_stats.npz')
    model_path = os.path.join(project_root, 'output', 'multi_stack', 'policy', f'{model_type}_policy_best.pth')

    # 简化的功率曲线（阶跃变化）
    test_steps = int(duration / 60) + horizon + 10
    P_profile = np.ones(test_steps) * 10e6  # 10MW基础
    # 添加一些变化
    for i in range(test_steps):
        if 5 <= i < 10:
            P_profile[i] = 15e6  # 15MW阶跃
        elif 10 <= i < 15:
            P_profile[i] = 8e6   # 8MW阶跃
        elif i >= 15:
            P_profile[i] = 12e6  # 12MW稳定

    # 初始化控制器
    if controller_type == 'nmpc':
        ctrl = MultiStackNMPCController(dt=dt_ctrl, horizon=horizon, dt_sub=sim_dt)
        name = "NMPC"
    else:
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            return None
        ctrl = MultiStackModelController(dt=dt_ctrl, horizon=horizon, model_type=model_type,
                                         model_path=model_path, stats_path=stats_path)
        name = model_type

    print(f"\n{'='*60}")
    print(f"Quick Test: {name}")
    print(f"{'='*60}")

    sim.reset()

    # 简化的历史记录
    history = {'t': [], 'P_ref': [], 'P_real': [], 'T_s_mean': [], 'HTO': [], 'ctrl_time': []}

    last_action = [np.ones(4)*2000, np.ones(4)*0.03, 0.0]

    # 简化的预热（2分钟）
    print("Quick warmup...")
    warmup_steps = int(120 / sim_dt)
    P_future = [10e6] * horizon
    ctrl_steps = int(dt_ctrl / sim_dt)

    for i in range(warmup_steps):
        measured_state = np.copy(sim.state)
        if i % ctrl_steps == 0:
            I_cmd, v_lye_cmd, v_c_cmd = ctrl.get_action(measured_state, P_future, T_ref, last_action)
            last_action = [I_cmd, v_lye_cmd, v_c_cmd]
        action_sim = np.concatenate([last_action[0], last_action[1], [last_action[2]]])
        sim.step(action_sim)

    print(f"Running {duration}s test...")

    # 主测试
    t_eval = np.arange(0, duration, sim_dt)
    total_ctrl_steps = len(t_eval) // ctrl_steps

    with tqdm(total=total_ctrl_steps, desc=name, unit="step") as pbar:
        for i, t in enumerate(t_eval):
            measured_state = np.copy(sim.state)
            T_s_vec = measured_state[1:5]

            Q, U_cell, eta = sim._calculate_electrochemical_properties(last_action[0], sim.state[1:5])
            P_real = np.sum(U_cell * last_action[0] * sim.N_cell)

            n_H2_sep_gas = measured_state[12]
            T_sep = measured_state[5]
            hto_pct = (n_H2_sep_gas * sim.R * T_sep) / (sim.P_sys * sim.V_sep_gas) * 100

            if i % ctrl_steps == 0:
                idx_min = int(t / 60)
                end_idx = idx_min + horizon
                if end_idx <= len(P_profile):
                    P_future = P_profile[idx_min:end_idx]
                else:
                    P_future = np.full(horizon, P_profile[-1])

                start_time = time.time()
                I_cmd, v_lye_cmd, v_c_cmd = ctrl.get_action(measured_state, list(P_future), T_ref, last_action)
                ctrl_time = time.time() - start_time

                last_action = [I_cmd, v_lye_cmd, v_c_cmd]

                history['t'].append(t)
                history['P_ref'].append(P_profile[idx_min])
                history['P_real'].append(P_real)
                history['T_s_mean'].append(np.mean(T_s_vec))
                history['HTO'].append(hto_pct)
                history['ctrl_time'].append(ctrl_time)

                pbar.update(1)

            action_sim = np.concatenate([last_action[0], last_action[1], [last_action[2]]])
            sim.step(action_sim)

    return history


def main():
    duration = 600  # 10分钟测试

    results = {}

    # 测试 NMPC
    try:
        results['nmpc'] = run_quick_test('nmpc', None, duration)
    except Exception as e:
        print(f"NMPC test failed: {e}")
        results['nmpc'] = None

    # 测试 Flow TCN
    try:
        results['flow_tcn'] = run_quick_test('model', 'flow_tcn', duration)
    except Exception as e:
        print(f"Flow TCN test failed: {e}")
        results['flow_tcn'] = None

    # 测试 Diffusion TCN
    try:
        results['diffusion_tcn'] = run_quick_test('model', 'diffusion_tcn', duration)
    except Exception as e:
        print(f"Diffusion TCN test failed: {e}")
        results['diffusion_tcn'] = None

    # 打印结果
    print("\n" + "="*80)
    print("Quick Test Results (10 minutes simulation)")
    print("="*80)
    print(f"{'Controller':<20} {'Power RMSE':<15} {'Avg Ctrl Time':<15} {'Max HTO':<10}")
    print(f"{'':20} {'(MW)':<15} {'(ms)':<15} {'(%)':<10}")
    print("-"*80)

    for name, data in results.items():
        if data is None:
            continue
        P_ref = np.array(data['P_ref'])
        P_real = np.array(data['P_real'])
        rmse_p = np.sqrt(np.mean((P_real - P_ref)**2)) / 1e6
        ctrl_time = np.mean(data['ctrl_time']) * 1000
        hto_max = np.max(data['HTO'])
        print(f"{name:<20} {rmse_p:<15.3f} {ctrl_time:<15.2f} {hto_max:<10.3f}")

    print("="*80)

    # 保存对比图
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    colors = {'nmpc': 'blue', 'flow_tcn': 'green', 'diffusion_tcn': 'orange'}

    for name, data in results.items():
        if data is None:
            continue
        t_arr = np.array(data['t'])
        P_ref = np.array(data['P_ref'])
        P_real = np.array(data['P_real'])
        color = colors.get(name, 'black')

        # 功率
        ax = axes[0, 0]
        if name == 'nmpc':
            ax.plot(t_arr, P_ref/1e6, 'k--', alpha=0.5, label='Reference')
        ax.plot(t_arr, P_real/1e6, color=color, alpha=0.8, label=name)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Power (MW)')
        ax.set_title('Power Tracking')
        ax.legend()
        ax.grid(True)

        # 功率误差
        ax = axes[0, 1]
        ax.plot(t_arr, (P_real - P_ref)/1e6, color=color, alpha=0.8, label=name)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Error (MW)')
        ax.set_title('Power Tracking Error')
        ax.legend()
        ax.grid(True)

        # 温度
        ax = axes[1, 0]
        T_s = np.array(data['T_s_mean']) - 273.15
        ax.plot(t_arr, T_s, color=color, alpha=0.8, label=name)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title('Stack Temperature')
        ax.legend()
        ax.grid(True)

        # 控制时间
        ax = axes[1, 1]
        ctrl_t = np.array(data['ctrl_time']) * 1000
        ax.plot(t_arr, ctrl_t, color=color, alpha=0.8, label=name)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Time (ms)')
        ax.set_title('Control Computation Time')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    output_dir = os.path.join(project_root, 'output', 'multi_stack', 'comparison')
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = os.path.join(output_dir, f'quick_test_{timestamp}.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {plot_path}")
    plt.close()


if __name__ == "__main__":
    main()
