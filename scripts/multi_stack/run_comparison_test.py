"""
多槽控制器对比测试脚本
对比 NMPC、Flow TCN、TCN Diffusion 三种控制器的性能
"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import argparse
import json
from tqdm import tqdm

import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from plant.multi_stack_simulator import MultiStackSimulator
from controller.multi_stack.nmpc_controller import MultiStackNMPCController
from controller.multi_stack.model_controller import MultiStackModelController


def load_december_profile():
    """加载12月风电功率曲线"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    profile_path = os.path.join(project_root, 'output', 'power', 'wind', 'wind_power_2025-02_1min.csv')

    if not os.path.exists(profile_path):
        raise FileNotFoundError(f"Profile not found at {profile_path}")

    print(f"Loading profile from {profile_path}...")
    df = pd.read_csv(profile_path)
    return df['P_ref'].values


def run_warmup(sim, ctrl, last_action, dt, T_ref=353.15):
    """预热阶段"""
    warmup_duration = 14400  # 4小时
    warmup_steps = int(warmup_duration / dt)
    warmup_P_ref = 10.0e6  # 10MW constant
    P_future = [warmup_P_ref] * ctrl.horizon
    ctrl_steps = int(ctrl.dt / dt)

    print(f"Starting Warm-up Phase ({warmup_duration}s at {warmup_P_ref/1e6}MW)...")

    for i in range(warmup_steps):
        measured_state = np.copy(sim.state)

        if i % ctrl_steps == 0:
            I_cmd, v_lye_cmd, v_c_cmd = ctrl.get_action(
                measured_state, P_future, T_ref=T_ref, last_action=last_action
            )
            last_action = [I_cmd, v_lye_cmd, v_c_cmd]

        action_sim = np.concatenate([last_action[0], last_action[1], [last_action[2]]])
        sim.step(action_sim)

    print("Warm-up Complete.")
    return last_action


def run_controller_test(controller_type, model_type, duration, sim_dt=0.2, dt_ctrl=60.0, horizon=5):
    """运行单个控制器测试"""
    T_ref = 353.15  # 80°C

    sim = MultiStackSimulator(dt=sim_dt)

    # 路径设置
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    stats_path = os.path.join(project_root, 'output', 'multi_stack', 'diffusion_stats.npz')
    model_path = os.path.join(project_root, 'output', 'multi_stack', 'policy', f'{model_type}_policy_best.pth')

    # 初始化控制器
    if controller_type == 'nmpc':
        ctrl = MultiStackNMPCController(dt=dt_ctrl, horizon=horizon, dt_sub=sim_dt)
        print(f"\n{'='*60}")
        print(f"Testing NMPC Controller")
        print(f"{'='*60}")
    elif controller_type == 'model':
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}, skipping...")
            return None
        ctrl = MultiStackModelController(dt=dt_ctrl, horizon=horizon, model_type=model_type,
                                         model_path=model_path, stats_path=stats_path)
        print(f"\n{'='*60}")
        print(f"Testing {model_type.upper()} Controller")
        print(f"{'='*60}")
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")

    sim.reset()
    full_profile = load_december_profile()

    if len(full_profile) < duration/60:
        duration = len(full_profile) * 60

    t_eval = np.arange(0, duration, sim_dt)

    # 历史记录
    history = {
        't': [], 'P_ref': [], 'P_real': [],
        'T_s_all': [], 'T_ref': [],
        'I_all': [], 'v_lye_all': [], 'v_c': [],
        'HTO': [], 'ctrl_time': []
    }

    last_action = [
        np.ones(4)*2000,
        np.ones(4)*0.03,
        0.0
    ]

    # 预热
    last_action = run_warmup(sim, ctrl, last_action, sim_dt, T_ref)

    # 主测试循环
    ctrl_steps = int(dt_ctrl / sim_dt)
    profile_indices = (t_eval / 60).astype(int)
    profile_indices = np.clip(profile_indices, 0, len(full_profile)-1)
    total_ctrl_steps = len(t_eval) // ctrl_steps

    print(f"Running test for {duration}s ({duration/3600:.1f}h)...")

    with tqdm(total=total_ctrl_steps, desc=f"Testing {controller_type}", unit="step") as pbar:
        for i, t in enumerate(t_eval):
            measured_state = np.copy(sim.state)
            T_s_vec = measured_state[1:5]

            # 计算实际功率
            Q, U_cell, eta = sim._calculate_electrochemical_properties(last_action[0], sim.state[1:5])
            P_real = np.sum(U_cell * last_action[0] * sim.N_cell)

            # 计算HTO
            n_H2_sep_gas = measured_state[12]
            T_sep = measured_state[5]
            hto_pct = (n_H2_sep_gas * sim.R * T_sep) / (sim.P_sys * sim.V_sep_gas) * 100

            # 控制更新
            if i % ctrl_steps == 0:
                idx_min = int(t / 60)
                end_idx = idx_min + horizon
                if end_idx <= len(full_profile):
                    P_future = full_profile[idx_min:end_idx]
                else:
                    P_future = full_profile[idx_min:]
                    if len(P_future) < horizon:
                        padding = np.full(horizon - len(P_future), full_profile[-1])
                        P_future = np.concatenate([P_future, padding])

                import time
                start_time = time.time()
                I_cmd, v_lye_cmd, v_c_cmd = ctrl.get_action(
                    measured_state, list(P_future), T_ref, last_action
                )
                ctrl_time = time.time() - start_time

                last_action = [I_cmd, v_lye_cmd, v_c_cmd]

                # 记录数据
                history['t'].append(t)
                history['P_ref'].append(full_profile[profile_indices[i]])
                history['P_real'].append(P_real)
                history['T_s_all'].append(T_s_vec)
                history['T_ref'].append(T_ref)
                history['I_all'].append(np.copy(last_action[0]))
                history['v_lye_all'].append(np.copy(last_action[1]))
                history['v_c'].append(last_action[2])
                history['HTO'].append(hto_pct)
                history['ctrl_time'].append(ctrl_time)

                pbar.set_postfix({
                    "P_ref": f"{full_profile[profile_indices[i]]/1e6:.1f}MW",
                    "P_real": f"{P_real/1e6:.1f}MW",
                    "T_s": f"{np.mean(T_s_vec)-273.15:.1f}C",
                    "t_ctrl": f"{ctrl_time*1000:.1f}ms"
                })
                pbar.update(1)

            action_sim = np.concatenate([last_action[0], last_action[1], [last_action[2]]])
            sim.step(action_sim)

    return history


def calculate_metrics(history):
    """计算性能指标"""
    t_arr = np.array(history['t'])
    P_ref_arr = np.array(history['P_ref'])
    P_real_arr = np.array(history['P_real'])
    T_s_all = np.array(history['T_s_all'])
    T_s_mean = np.mean(T_s_all, axis=1)
    T_ref_arr = np.array(history['T_ref'])
    HTO_arr = np.array(history['HTO'])
    ctrl_time_arr = np.array(history['ctrl_time'])

    # 功率跟踪RMSE
    rmse_p = np.sqrt(np.mean((P_real_arr - P_ref_arr)**2)) / 1e6

    # 温度跟踪RMSE
    rmse_t = np.sqrt(np.mean((T_s_mean - T_ref_arr)**2))

    # HTO最大值和平均值
    hto_max = np.max(HTO_arr)
    hto_mean = np.mean(HTO_arr)

    # 控制时间统计
    ctrl_time_mean = np.mean(ctrl_time_arr) * 1000  # ms
    ctrl_time_max = np.max(ctrl_time_arr) * 1000

    return {
        'rmse_power_mw': float(rmse_p),
        'rmse_temp_k': float(rmse_t),
        'hto_max_pct': float(hto_max),
        'hto_mean_pct': float(hto_mean),
        'ctrl_time_mean_ms': float(ctrl_time_mean),
        'ctrl_time_max_ms': float(ctrl_time_max),
        'n_samples': len(t_arr)
    }


def save_comparison_plot(results, output_dir):
    """保存对比图"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    colors = {
        'nmpc': 'blue',
        'flow_tcn': 'green',
        'diffusion_tcn': 'orange'
    }

    labels = {
        'nmpc': 'NMPC',
        'flow_tcn': 'Flow TCN',
        'diffusion_tcn': 'TCN Diffusion'
    }

    for name, data in results.items():
        if data is None:
            continue
        t_arr = np.array(data['t'])
        P_ref = np.array(data['P_ref'])
        P_real = np.array(data['P_real'])
        T_s_all = np.array(data['T_s_all'])
        T_s_mean = np.mean(T_s_all, axis=1)
        T_ref_arr = np.array(data['T_ref'])

        color = colors.get(name, 'black')
        label = labels.get(name, name)

        # 功率跟踪
        ax = axes[0, 0]
        if name == 'nmpc':
            ax.plot(t_arr/3600, P_ref/1e6, 'k--', alpha=0.5, label='Reference')
        ax.plot(t_arr/3600, P_real/1e6, color=color, alpha=0.8, label=label)
        ax.set_xlabel('Time (h)')
        ax.set_ylabel('Power (MW)')
        ax.set_title('Power Tracking')
        ax.legend()
        ax.grid(True)

        # 功率误差
        ax = axes[0, 1]
        power_error = (P_real - P_ref) / 1e6
        ax.plot(t_arr/3600, power_error, color=color, alpha=0.7, label=label)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Time (h)')
        ax.set_ylabel('Power Error (MW)')
        ax.set_title('Power Tracking Error')
        ax.legend()
        ax.grid(True)

        # 温度跟踪
        ax = axes[0, 2]
        if name == 'nmpc':
            ax.plot(t_arr/3600, T_ref_arr-273.15, 'k--', alpha=0.5, label='Reference')
        ax.plot(t_arr/3600, T_s_mean-273.15, color=color, alpha=0.8, label=label)
        ax.set_xlabel('Time (h)')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title('Stack Temperature')
        ax.legend()
        ax.grid(True)

        # 温度误差
        ax = axes[1, 0]
        temp_error = T_s_mean - T_ref_arr
        ax.plot(t_arr/3600, temp_error, color=color, alpha=0.7, label=label)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Time (h)')
        ax.set_ylabel('Temp Error (K)')
        ax.set_title('Temperature Tracking Error')
        ax.legend()
        ax.grid(True)

        # HTO
        ax = axes[1, 1]
        HTO_arr = np.array(data['HTO'])
        ax.plot(t_arr/3600, HTO_arr, color=color, alpha=0.8, label=label)
        ax.set_xlabel('Time (h)')
        ax.set_ylabel('HTO (%)')
        ax.set_title('H2 in O2 (HTO)')
        ax.legend()
        ax.grid(True)

        # 控制时间
        ax = axes[1, 2]
        ctrl_time_arr = np.array(data['ctrl_time']) * 1000
        ax.plot(t_arr/3600, ctrl_time_arr, color=color, alpha=0.8, label=label)
        ax.set_xlabel('Time (h)')
        ax.set_ylabel('Control Time (ms)')
        ax.set_title('Control Computation Time')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'comparison_{timestamp}.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\nComparison plot saved to {plot_path}")
    plt.close(fig)


def print_comparison_table(results):
    """打印对比表格"""
    print("\n" + "="*80)
    print("Controller Performance Comparison")
    print("="*80)
    print(f"{'Controller':<20} {'Power RMSE':<12} {'Temp RMSE':<12} {'HTO Max':<10} {'HTO Mean':<10} {'Ctrl Time':<12}")
    print(f"{'':20} {'(MW)':<12} {'(K)':<12} {'(%)':<10} {'(%)':<10} {'(ms)':<12}")
    print("-"*80)

    for name, data in results.items():
        if data is None:
            continue
        metrics = calculate_metrics(data)
        print(f"{name:<20} {metrics['rmse_power_mw']:<12.3f} {metrics['rmse_temp_k']:<12.3f} "
              f"{metrics['hto_max_pct']:<10.3f} {metrics['hto_mean_pct']:<10.3f} "
              f"{metrics['ctrl_time_mean_ms']:<12.2f}")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Multi-Stack Controller Comparison')
    parser.add_argument('--duration', type=int, default=3600, help='Test duration in seconds (default: 1 hour)')
    parser.add_argument('--controllers', type=str, nargs='+', default=['nmpc', 'flow_tcn', 'diffusion_tcn'],
                        help='Controllers to compare')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    output_dir = os.path.join(project_root, 'output', 'multi_stack', 'comparison')
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    # 测试每个控制器
    for controller in args.controllers:
        if controller == 'nmpc':
            history = run_controller_test('nmpc', None, args.duration)
            results['nmpc'] = history
        elif controller == 'flow_tcn':
            history = run_controller_test('model', 'flow_tcn', args.duration)
            results['flow_tcn'] = history
        elif controller == 'diffusion_tcn':
            history = run_controller_test('model', 'diffusion_tcn', args.duration)
            results['diffusion_tcn'] = history
        else:
            print(f"Unknown controller: {controller}")

    # 打印对比表格
    print_comparison_table(results)

    # 保存对比图
    save_comparison_plot(results, output_dir)

    # 保存详细结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary = {}
    for name, data in results.items():
        if data is not None:
            summary[name] = calculate_metrics(data)

    summary_path = os.path.join(output_dir, f'summary_{timestamp}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
