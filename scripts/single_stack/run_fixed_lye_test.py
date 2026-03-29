"""
单槽固定碱液流速NMPC测试脚本

使用固定碱液流速的控制器，功率参考为确定的阶跃值。
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import argparse
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from plant.single_stack_simulator import SingleStackSimulator
from controller.single_stack.fixed_lye_nmpc_controller import SingleStackFixedLyeNMPCController


def create_step_profile(duration, step_times, step_powers, sim_dt):
    """
    创建阶跃功率参考曲线
    Args:
        duration: 总仿真时长(秒)
        step_times: 阶跃时间点列表(秒)
        step_powers: 各阶段功率值列表(W)
        sim_dt: 仿真步长(秒)
    Returns:
        profile: 功率参考数组
    """
    steps = int(duration / sim_dt)
    profile = np.zeros(steps)

    for i in range(steps):
        t = i * sim_dt
        stage = 0
        for j, step_t in enumerate(step_times):
            if t >= step_t:
                stage = j + 1
        if stage < len(step_powers):
            profile[i] = step_powers[stage]
        else:
            profile[i] = step_powers[-1]

    return profile


def run_fixed_lye_test(v_lye_fixed=0.03, duration=7200, step_scenario='default'):
    """
    运行固定碱液流速NMPC测试
    Args:
        v_lye_fixed: 固定的碱液流速 (m^3/s)
        duration: 测试时长(秒)
        step_scenario: 阶跃场景名称
    """
    sim_dt = 0.2
    dt_ctrl = 60.0
    horizon = 5
    T_ref = 353.15

    sim = SingleStackSimulator(sim_dt=sim_dt)
    ctrl = SingleStackFixedLyeNMPCController(
        dt=dt_ctrl,
        horizon=horizon,
        sim_dt=sim_dt,
        v_lye_fixed=v_lye_fixed
    )

    scenarios = {
        'default': {
            'times': [0, 1800, 3600, 5400],
            'powers': [2.0e6, 4.0e6, 6.0e6, 3.0e6]
        },
        'low_power': {
            'times': [0, 1800, 3600],
            'powers': [1.0e6, 2.5e6, 1.5e6]
        },
        'high_power': {
            'times': [0, 1200, 2400, 3600, 4800],
            'powers': [4.0e6, 6.0e6, 5.0e6, 6.5e6, 4.0e6]
        },
        'ramp_up': {
            'times': [0, 600, 1200, 1800, 2400, 3000],
            'powers': [1.0e6, 2.0e6, 3.0e6, 4.0e6, 5.0e6, 6.0e6]
        },
        'constant_4mw': {
            'times': [0],
            'powers': [4.0e6]
        },
        'constant_3mw': {
            'times': [0],
            'powers': [3.0e6]
        },
        'constant_2mw': {
            'times': [0],
            'powers': [2.0e6]
        },
        'constant_1_5mw': {
            'times': [0],
            'powers': [1.5e6]
        }
    }

    if step_scenario not in scenarios:
        print(f"Unknown scenario: {step_scenario}, using default")
        step_scenario = 'default'

    scenario = scenarios[step_scenario]
    step_times = scenario['times']
    step_powers = scenario['powers']

    print("=" * 60)
    print("单槽固定碱液流速NMPC测试")
    print("=" * 60)
    print(f"固定碱液流速: {v_lye_fixed} m^3/s")
    print(f"测试时长: {duration/60:.1f} 分钟 ({duration/3600:.2f} 小时)")
    print(f"阶跃场景: {step_scenario}")
    print(f"阶跃时间点: {[f'{t/60:.0f}min' for t in step_times]}")
    print(f"功率值: {[f'{p/1e6:.1f}MW' for p in step_powers]}")
    print("=" * 60)

    full_profile = create_step_profile(duration, step_times, step_powers, sim_dt)
    sim.reset()
    t_eval = np.arange(0, duration, sim_dt)

    history = {
        't': [], 'P_ref': [], 'P_real': [],
        'T_s_in': [], 'T_s_all': [], 'T_sep': [], 'T_c_out': [],
        'n_H2_an': [], 'n_liq': [], 'n_gas': [],
        'T_ref': [], 'P_ref_future': [],
        'I_prev': [], 'v_lye_prev': [], 'v_c_prev': [],
        'I': [], 'v_lye': [], 'v_c': [],
        'U_cell_all': [], 'HTO': [], 'H2_rate': []
    }

    last_action = [2000, v_lye_fixed, 0.0]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    data_filename = f"fixed_lye_{step_scenario}_{timestamp}.csv"
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    output_dir = os.path.join(project_root, 'output', 'single_stack', 'fixed_lye_test')
    os.makedirs(output_dir, exist_ok=True)

    plot_every_steps = max(1, int(2000 / sim_dt))
    ctrl_steps = int(dt_ctrl / sim_dt)
    total_ctrl_steps = len(t_eval) // ctrl_steps

    print(f"仿真步数: {len(t_eval)}, 控制步数: {total_ctrl_steps}")
    print("开始仿真...")

    with tqdm(total=total_ctrl_steps, desc="Fixed Lye NMPC", unit="ctrl_step") as pbar:
        for i, t in enumerate(t_eval):
            measured_state = np.copy(sim.state)

            T_s_in = measured_state[0]
            T_s = measured_state[1]
            T_sep = measured_state[2]
            T_c_out = measured_state[3]
            n_H2_an = measured_state[4]
            n_liq = measured_state[5]
            n_gas = measured_state[6]

            Q, U_cell, eta = sim._calculate_electrochemical_properties(last_action[0], sim.state[1])
            P_real = U_cell * last_action[0] * sim.N_cell

            hto_pct = (n_gas * sim.R * T_sep) / (sim.P_sys * sim.V_sep_gas) * 100
            h2_rate = sim.N_cell * last_action[0] * eta / (2 * sim.F)

            prev_action = list(last_action)

            if i % ctrl_steps == 0:
                idx = i
                P_future = []
                for k in range(horizon):
                    future_idx = idx + k * ctrl_steps
                    if future_idx < len(full_profile):
                        P_future.append(full_profile[future_idx])
                    else:
                        P_future.append(full_profile[-1])

                I_cmd, v_lye_cmd, v_c_cmd = ctrl.get_action(
                    measured_state, P_future, T_ref, last_action
                )

                action_sim = np.array([I_cmd, v_lye_cmd, v_c_cmd])
                last_action = [I_cmd, v_lye_cmd, v_c_cmd]

                pbar.set_postfix({
                    "t": f"{t:.0f}s",
                    "P_ref": f"{full_profile[i]/1e6:.1f}MW",
                    "P_real": f"{P_real/1e6:.1f}MW",
                    "T_s": f"{T_s-273.15:.1f}C",
                    "v_lye": f"{v_lye_cmd:.4f}"
                })
                pbar.update(1)
            else:
                action_sim = np.array(last_action)

            sim.step(action_sim)

            if i % 10 == 0:
                history['t'].append(t)
                history['P_ref'].append(full_profile[i])
                history['P_real'].append(P_real)
                history['T_s_in'].append(T_s_in)
                history['T_s_all'].append(T_s)
                history['T_sep'].append(T_sep)
                history['T_c_out'].append(T_c_out)
                history['n_H2_an'].append(n_H2_an)
                history['n_liq'].append(n_liq)
                history['n_gas'].append(n_gas)
                history['T_ref'].append(T_ref)
                history['P_ref_future'].append(P_future)
                history['I_prev'].append(prev_action[0])
                history['v_lye_prev'].append(prev_action[1])
                history['v_c_prev'].append(prev_action[2])
                history['I'].append(last_action[0])
                history['v_lye'].append(last_action[1])
                history['v_c'].append(last_action[2])
                history['U_cell_all'].append(U_cell)
                history['HTO'].append(hto_pct)
                history['H2_rate'].append(h2_rate)

            if plot_every_steps is not None and i > 0 and i % plot_every_steps == 0:
                save_data_csv(history, output_dir, data_filename)
                save_plot(history, output_dir, data_filename.replace('.csv', '_progress.png'),
                         v_lye_fixed, step_scenario)

    save_data_csv(history, output_dir, data_filename)
    save_plot(history, output_dir, data_filename.replace('.csv', '.png'),
             v_lye_fixed, step_scenario)
    calculate_metrics(history, duration, output_dir, data_filename)

    v_lye_values = np.array(history['v_lye'])
    v_lye_variation = np.max(v_lye_values) - np.min(v_lye_values)
    print(f"\n碱液流速验证:")
    print(f"  设定值: {v_lye_fixed} m^3/s")
    print(f"  实际范围: [{np.min(v_lye_values):.6f}, {np.max(v_lye_values):.6f}] m^3/s")
    print(f"  最大偏差: {v_lye_variation:.6e} m^3/s")

    if v_lye_variation < 1e-10:
        print("  [PASS] 碱液流速保持恒定")
    else:
        print("  [WARNING] 碱液流速存在偏差")

    print(f"\n测试完成，结果保存在: {output_dir}")
    return history


def save_data_csv(history, output_dir, filename):
    keys_scalar = ['t', 'P_ref', 'P_real', 'T_s_in', 'T_s_all', 'T_sep',
                   'T_c_out', 'n_H2_an', 'n_liq', 'n_gas', 'T_ref',
                   'I_prev', 'v_lye_prev', 'v_c_prev',
                   'I', 'v_lye', 'v_c', 'HTO', 'H2_rate', 'U_cell_all']

    data_dict = {}
    for key in keys_scalar:
        if key in history and len(history[key]) > 0:
            data_dict[key] = history[key]

    df = pd.DataFrame(data_dict)
    file_path = os.path.join(output_dir, filename)
    df.to_csv(file_path, index=False)
    print(f"数据已保存: {file_path}")


def save_plot(history, output_dir, filename, v_lye_fixed, scenario):
    t_arr = np.array(history['t'])
    if len(t_arr) == 0:
        return

    T_s_all = np.array(history['T_s_all'])
    U_cell_all = np.array(history['U_cell_all'])
    I_all = np.array(history['I'])
    v_lye_all = np.array(history['v_lye'])

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(f'Fixed Lye NMPC Test (v_lye={v_lye_fixed}, scenario={scenario})', fontsize=14)

    ax = axes[0, 0]
    ax.plot(t_arr/60, np.array(history['P_ref'])/1e6, 'k--', label='Ref', linewidth=2)
    ax.plot(t_arr/60, np.array(history['P_real'])/1e6, 'b-', label='Real')
    ax.set_title('Power Tracking')
    ax.set_ylabel('MW')
    ax.set_xlabel('Time (min)')
    ax.legend()
    ax.grid(True)

    ax = axes[0, 1]
    ax.plot(t_arr/60, history['T_sep'], 'g--', label='Separator')
    ax.plot(t_arr/60, history['T_c_out'], 'c:', label='CW Out')
    ax.plot(t_arr/60, T_s_all, 'r-', label='Stack')
    ax.plot(t_arr/60, history['T_ref'], 'k--', linewidth=1.5, label='Ref Temp')
    ax.set_title('Temperatures')
    ax.set_ylabel('K')
    ax.set_xlabel('Time (min)')
    ax.legend()
    ax.grid(True)

    ax = axes[0, 2]
    ax.plot(t_arr/60, history['HTO'], 'm-', label='HTO')
    ax.axhline(y=2.0, color='r', linestyle='--', label='Limit')
    ax.set_title('HTO (H2 in O2)')
    ax.set_ylabel('%')
    ax.set_xlabel('Time (min)')
    ax.legend()
    ax.grid(True)

    ax = axes[1, 0]
    ax.plot(t_arr/60, I_all, 'b-', label='Stack')
    ax.set_title('Stack Current')
    ax.set_ylabel('Amps')
    ax.set_xlabel('Time (min)')
    ax.legend()
    ax.grid(True)

    ax = axes[1, 1]
    ax.plot(t_arr/60, U_cell_all, 'b-', label='Stack')
    ax.set_title('Cell Voltage')
    ax.set_ylabel('Volts')
    ax.set_xlabel('Time (min)')
    ax.legend()
    ax.grid(True)

    ax = axes[1, 2]
    ax.plot(t_arr/60, v_lye_all, 'orange', label=f'v_lye (fixed={v_lye_fixed})')
    ax.axhline(y=v_lye_fixed, color='r', linestyle='--', label='Target')
    ax.set_title('Lye Flow Rate')
    ax.set_ylabel('m3/s')
    ax.set_xlabel('Time (min)')
    ax.legend()
    ax.grid(True)

    ax = axes[2, 0]
    ax.plot(t_arr/60, history['v_c'], 'cyan', label='CW')
    ax.set_title('Coolant Flow')
    ax.set_ylabel('m3/s')
    ax.set_xlabel('Time (min)')
    ax.legend()
    ax.grid(True)

    ax = axes[2, 1]
    ax.plot(t_arr/60, history['H2_rate'], 'g-', label='H2 Rate')
    ax.set_title('H2 Production Rate')
    ax.set_ylabel('mol/s')
    ax.set_xlabel('Time (min)')
    ax.legend()
    ax.grid(True)

    ax = axes[2, 2]
    power_error = (np.array(history['P_real']) - np.array(history['P_ref'])) / 1e6
    ax.plot(t_arr/60, power_error, 'r-', label='Power Error')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax.set_title('Power Tracking Error')
    ax.set_ylabel('MW')
    ax.set_xlabel('Time (min)')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close(fig)


def calculate_metrics(history, duration, output_dir, filename):
    t_arr = np.array(history['t'])
    if len(t_arr) == 0:
        return

    P_ref_arr = np.array(history['P_ref'])
    P_real_arr = np.array(history['P_real'])
    T_s_all = np.array(history['T_s_all'])
    T_ref_arr = np.array(history['T_ref'])

    rmse_p = float(np.sqrt(np.mean((P_real_arr - P_ref_arr)**2)) / 1e6)
    rmse_t = float(np.sqrt(np.mean((T_s_all - T_ref_arr)**2)))
    max_p_error = float(np.max(np.abs(P_real_arr - P_ref_arr)) / 1e6)

    metrics = {
        "duration": duration,
        "rmse_power_mw": rmse_p,
        "rmse_temp_k": rmse_t,
        "max_power_error_mw": max_p_error
    }

    json_filename = filename.replace('.csv', '_metrics.json')
    json_path = os.path.join(output_dir, json_filename)

    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print("-" * 50)
    print(f"性能指标 (时长: {duration}s)")
    print(f"功率RMSE: {rmse_p:.3f} MW")
    print(f"温度RMSE: {rmse_t:.3f} K")
    print(f"最大功率误差: {max_p_error:.3f} MW")
    print(f"指标已保存至: {json_path}")
    print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Single-Stack Fixed Lye Flow NMPC Test')
    parser.add_argument('--v_lye', type=float, default=0.03,
                        help='Fixed lye flow rate (m^3/s), default: 0.03')
    parser.add_argument('--duration', type=int, default=7200,
                        help='Test duration in seconds, default: 7200 (2 hours)')
    parser.add_argument('--scenario', type=str, default='default',
                        choices=['default', 'low_power', 'high_power', 'ramp_up', 'constant_4mw', 'constant_3mw', 'constant_2mw', 'constant_1_5mw'],
                        help='Power step scenario')

    args = parser.parse_args()

    run_fixed_lye_test(
        v_lye_fixed=args.v_lye,
        duration=args.duration,
        step_scenario=args.scenario
    )
