"""
并行风电数据集生成脚本 - 多月份并行处理
基于generate_dataset.py修改，支持多进程并行生成不同月份的数据
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import argparse
import glob
from joblib import Parallel, delayed

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from plant.single_stack_simulator import SingleStackSimulator
from controller.single_stack.nmpc_controller import SingleStackNMPCController


def load_month_profile(month_file):
    """加载单个月份的功率曲线"""
    df = pd.read_csv(month_file)
    profile = df['P_ref'].values
    # Scale for single stack (approx 1/4 of total power)
    profile = profile * 0.25
    return profile


def run_warmup_phase(sim, controller, profile_first_value, dt_ctrl, horizon, sim_dt, T_ref, I_prev, v_lye_prev, v_c_prev):
    """运行warmup阶段，使系统达到稳态"""
    warmup_duration_hours = 4
    warmup_steps = int(warmup_duration_hours * 3600 / dt_ctrl)

    # Warmup starts at 2MW
    P_warmup_start = 2.0e6
    P_warmup_end = profile_first_value

    try:
        for t_w in range(warmup_steps):
            current_time = -(warmup_steps - t_w) * dt_ctrl

            # Interpolate P_ref
            alpha = t_w / max(1, warmup_steps - 1)
            current_P_ref = P_warmup_start + alpha * (P_warmup_end - P_warmup_start)

            # Future P_ref
            P_ref_future = []
            for h in range(horizon):
                future_idx = t_w + h
                if future_idx < warmup_steps:
                    alpha_f = future_idx / max(1, warmup_steps - 1)
                    val = P_warmup_start + alpha_f * (P_warmup_end - P_warmup_start)
                else:
                    val = profile_first_value
                P_ref_future.append(val)

            # NMPC Step
            current_state = sim.state.copy()
            last_action = [I_prev, v_lye_prev, v_c_prev]

            try:
                u_opt_matrix, states_matrix = controller.get_all_actions_states(
                    current_state, P_ref_future, T_ref, last_action
                )
            except Exception as e:
                # Fallback
                u_opt_matrix = np.tile(last_action, (horizon, 1))
                states_matrix = np.tile(current_state, (horizon, 1))

            u0 = u_opt_matrix[0]
            I_cmd, v_lye_cmd, v_c_cmd = u0[0], u0[1], u0[2]

            # Step simulator
            sim_steps = int(dt_ctrl / sim_dt)
            action_sim = np.array([I_cmd, v_lye_cmd, v_c_cmd])
            for _ in range(sim_steps):
                sim.step(action_sim)

            # Update history
            I_prev = I_cmd
            v_lye_prev = v_lye_cmd
            v_c_prev = v_c_cmd

        return I_prev, v_lye_prev, v_c_prev

    except Exception as e:
        print(f"Warmup failed: {e}")
        return I_prev, v_lye_prev, v_c_prev


def process_single_month(args_dict):
    """
    处理单个月份的数据生成

    Args:
        args_dict: 包含所有参数的字典
            - month_idx: 月份序号 (1-12)
            - month_file: 月份数据文件路径
            - timestamp: 统一时间戳
            - output_dir: 输出目录
            - dt_ctrl: 控制周期
            - horizon: NMPC horizon
            - sim_dt: 仿真步长
            - T_ref: 目标温度
            - skip_warmup: 是否跳过warmup
    """
    month_idx = args_dict['month_idx']
    month_file = args_dict['month_file']
    timestamp = args_dict['timestamp']
    output_dir = args_dict['output_dir']
    dt_ctrl = args_dict['dt_ctrl']
    horizon = args_dict['horizon']
    sim_dt = args_dict['sim_dt']
    T_ref = args_dict['T_ref']
    skip_warmup = args_dict.get('skip_warmup', False)

    month_name = os.path.basename(month_file).replace('wind_power_', '').replace('_1min.csv', '')
    print(f"[Month {month_idx:03d}] Starting {month_name}...")

    try:
        # Load month profile
        profile = load_month_profile(month_file)
        total_steps = len(profile) - horizon

        if total_steps <= 0:
            print(f"[Month {month_idx:03d}] Profile too short, skipping")
            return False

        # Initialize simulator and controller
        sim = SingleStackSimulator(sim_dt=sim_dt)
        sim.reset()

        controller = SingleStackNMPCController(dt=dt_ctrl, horizon=horizon, sim_dt=sim_dt)

        # Initial actions
        I_prev = 2000.0
        v_lye_prev = 0.03
        v_c_prev = 0.0

        # Warmup phase
        if not skip_warmup:
            I_prev, v_lye_prev, v_c_prev = run_warmup_phase(
                sim, controller, profile[0], dt_ctrl, horizon, sim_dt, T_ref,
                I_prev, v_lye_prev, v_c_prev
            )

        # Main data generation loop
        data_list = []

        for t in tqdm(range(total_steps), desc=f"Month {month_idx:03d}", leave=False):
            current_state = sim.state.copy()
            T_s_in, T_s, T_sep, T_c_out, n_H2_an, n_liq, n_gas = current_state

            # Get power reference
            current_P_ref = profile[t]
            P_ref_future = profile[t:t+horizon].tolist()

            # Pad if necessary
            while len(P_ref_future) < horizon:
                P_ref_future.append(profile[-1])

            # NMPC solve
            last_action = [I_prev, v_lye_prev, v_c_prev]
            try:
                u_opt_matrix, states_matrix = controller.get_all_actions_states(
                    current_state, P_ref_future, T_ref, last_action
                )
            except Exception as e:
                # Fallback
                u_opt_matrix = np.tile(last_action, (horizon, 1))
                states_matrix = np.tile(current_state, (horizon, 1))

            # Extract first action
            u0 = u_opt_matrix[0]
            I_cmd, v_lye_cmd, v_c_cmd = u0[0], u0[1], u0[2]

            # Calculate derived properties
            _, U_cell, _ = sim._calculate_electrochemical_properties(I_cmd, T_s)
            P_real = I_cmd * U_cell * sim.N_cell
            hto_val = (n_gas * sim.R * T_sep) / (sim.P_sys * sim.V_sep_gas) * 100.0
            H2_rate_total = sim.N_cell * I_cmd / (2 * sim.F)

            # Step simulator
            sim_steps = int(dt_ctrl / sim_dt)
            action_sim = np.array([I_cmd, v_lye_cmd, v_c_cmd])
            for _ in range(sim_steps):
                sim.step(action_sim)

            # Store data
            row = {
                'step': t,
                'time': t * dt_ctrl,
                'P_ref': current_P_ref,
                'P_real': P_real,
                'T_s_in': T_s_in,
                'T_s': T_s,
                'T_sep': T_sep,
                'T_c_out': T_c_out,
                'n_H2_an': n_H2_an,
                'n_liq': n_liq,
                'n_gas': n_gas,
                'T_ref': T_ref,
                'I_prev': I_prev,
                'v_lye_prev': v_lye_prev,
                'v_c_prev': v_c_prev,
                'v_c': v_c_cmd,
                'v_lye': v_lye_cmd,
                'HTO': hto_val,
                'H2_rate': H2_rate_total,
                'I': I_cmd,
                'U_cell': U_cell
            }

            # Save NMPC plan
            for k in range(horizon):
                row[f'P_ref_future_{k}'] = P_ref_future[k]
                u_k = u_opt_matrix[k]
                s_k = states_matrix[k]

                row[f'plan_step_{k}_I'] = u_k[0]
                row[f'plan_step_{k}_v_lye'] = u_k[1]
                row[f'plan_step_{k}_v_c'] = u_k[2]

                row[f'plan_step_{k}_state_T_s_in'] = s_k[0]
                row[f'plan_step_{k}_state_T_s'] = s_k[1]
                row[f'plan_step_{k}_state_T_sep'] = s_k[2]
                row[f'plan_step_{k}_state_T_c_out'] = s_k[3]
                row[f'plan_step_{k}_state_n_H2_an'] = s_k[4]
                row[f'plan_step_{k}_state_n_liq'] = s_k[5]
                row[f'plan_step_{k}_state_n_gas'] = s_k[6]

            data_list.append(row)

            # Update history
            I_prev = I_cmd
            v_lye_prev = v_lye_cmd
            v_c_prev = v_c_cmd

        # Save to CSV
        df = pd.DataFrame(data_list)
        csv_file = os.path.join(output_dir, f"nmpc_dataset_wind_{month_idx:03d}_{timestamp}.csv")
        df.to_csv(csv_file, index=False)
        print(f"[Month {month_idx:03d}] Saved: {csv_file} ({len(data_list)} rows)")

        # Generate plot
        try:
            plot_month_data(df, output_dir, timestamp, month_idx, month_name)
        except Exception as e:
            print(f"[Month {month_idx:03d}] Plot generation failed: {e}")

        return True

    except Exception as e:
        print(f"[Month {month_idx:03d}] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def plot_month_data(df, output_dir, timestamp, month_idx, month_name):
    """生成单个月份的可视化图表"""
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    t_arr = df['time'].values / 3600  # Convert to hours

    # 1. Power
    ax = axes[0, 0]
    ax.plot(t_arr, df['P_ref']/1e6, 'k--', label='Ref', alpha=0.7)
    ax.plot(t_arr, df['P_real']/1e6, 'b-', label='Real')
    ax.set_title(f'Power Tracking - {month_name}')
    ax.set_ylabel('MW')
    ax.set_xlabel('Time (hours)')
    ax.legend()
    ax.grid(True)

    # 2. Temperatures
    ax = axes[0, 1]
    ax.plot(t_arr, df['T_sep'], 'g--', label='Separator')
    ax.plot(t_arr, df['T_c_out'], 'c:', label='CW Out')
    ax.plot(t_arr, df['T_s'], 'r-', label='Stack')
    ax.plot(t_arr, df['T_ref'], 'k--', linewidth=1.5, label='Ref Temp')
    ax.set_title('All Temperatures')
    ax.set_ylabel('K')
    ax.set_xlabel('Time (hours)')
    ax.legend()
    ax.grid(True)

    # 3. HTO
    ax = axes[0, 2]
    ax.plot(t_arr, df['HTO'], 'm-', label='HTO')
    ax.axhline(y=2.0, color='r', linestyle='--', label='Limit')
    ax.set_title('HTO (H2 in O2)')
    ax.set_ylabel('%')
    ax.set_xlabel('Time (hours)')
    ax.legend()
    ax.grid(True)

    # 4. Current
    ax = axes[1, 0]
    ax.plot(t_arr, df['I'], 'b-', label='Stack')
    ax.set_title('Stack Current')
    ax.set_ylabel('Amps')
    ax.set_xlabel('Time (hours)')
    ax.legend()
    ax.grid(True)

    # 5. Voltage
    ax = axes[1, 1]
    ax.plot(t_arr, df['U_cell'], 'b-', label='Stack')
    ax.axhline(y=2.2, color='r', linestyle='--', label='Limit')
    ax.set_title('Cell Voltage')
    ax.set_ylabel('Volts')
    ax.set_xlabel('Time (hours)')
    ax.legend()
    ax.grid(True)

    # 6. Lye Flow
    ax = axes[1, 2]
    ax.plot(t_arr, df['v_lye']*1000, 'orange', label='Stack')
    ax.set_title('Stack Lye Flow')
    ax.set_ylabel('L/s')
    ax.set_xlabel('Time (hours)')
    ax.legend()
    ax.grid(True)

    # 7. Coolant Flow
    ax = axes[2, 0]
    ax.plot(t_arr, df['v_c'], 'cyan', label='CW')
    ax.set_title('Coolant Flow')
    ax.set_ylabel('m³/s')
    ax.set_xlabel('Time (hours)')
    ax.legend()
    ax.grid(True)

    # 8. H2 Rate
    ax = axes[2, 1]
    ax.plot(t_arr, df['H2_rate'], 'g-', label='H2 Rate')
    ax.set_title('H2 Production Rate')
    ax.set_ylabel('mol/s')
    ax.set_xlabel('Time (hours)')
    ax.legend()
    ax.grid(True)

    # 9. H2 States
    ax = axes[2, 2]
    ax.plot(t_arr, df['n_H2_an'], 'b-', label='n_H2_an')
    ax.plot(t_arr, df['n_liq'], 'g-', label='n_liq')
    ax.plot(t_arr, df['n_gas'], 'r-', label='n_gas')
    ax.set_title('H2 States')
    ax.set_ylabel('mol')
    ax.set_xlabel('Time (hours)')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plot_file = os.path.join(output_dir, f"nmpc_dataset_wind_{month_idx:03d}_plot_{timestamp}.png")
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"[Month {month_idx:03d}] Plot saved: {plot_file}")


def main():
    parser = argparse.ArgumentParser(
        description='并行生成风电数据集 - 多月份并行处理'
    )
    parser.add_argument('--max_workers', type=int, default=4,
                        help='并行worker数量 (默认: 4, 设为0使用CPU核心数)')
    parser.add_argument('--dt_ctrl', type=float, default=60.0,
                        help='控制周期 (秒, 默认: 60)')
    parser.add_argument('--horizon', type=int, default=5,
                        help='NMPC预测horizon (默认: 5)')
    parser.add_argument('--sim_dt', type=float, default=0.2,
                        help='仿真步长 (秒, 默认: 0.2)')
    parser.add_argument('--T_ref', type=float, default=353.15,
                        help='目标温度 (K, 默认: 353.15 = 80°C)')
    parser.add_argument('--skip_warmup', action='store_true',
                        help='跳过warmup阶段')
    parser.add_argument('--months', type=str, default='all',
                        help='处理的月份, e.g., "1,2,3" 或 "all" (默认: all)')
    parser.add_argument('--wind_dir', type=str, default=None,
                        help='风电数据目录 (默认: output/power/wind/)')

    args = parser.parse_args()

    # Configuration
    dt_ctrl = args.dt_ctrl
    horizon = args.horizon
    sim_dt = args.sim_dt
    T_ref = args.T_ref

    # Determine max workers
    import multiprocessing
    if args.max_workers <= 0:
        max_workers = multiprocessing.cpu_count()
    else:
        max_workers = args.max_workers

    # Base output directory
    output_base_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..',
        'output', 'single_stack', 'dataset'
    ))

    # Generate unified timestamp and create subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_base_dir, f'wind_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    # Wind data directory
    if args.wind_dir:
        wind_dir = args.wind_dir
    else:
        wind_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..',
            'output', 'power', 'wind'
        ))

    # Get month files
    month_files = sorted(glob.glob(os.path.join(wind_dir, 'wind_power_2025-*_1min.csv')))

    if not month_files:
        print(f"No wind power files found in {wind_dir}")
        return

    print("=" * 60)
    print("并行风电数据集生成")
    print("=" * 60)
    print(f"配置:")
    print(f"  控制周期: {dt_ctrl}s")
    print(f"  NMPC Horizon: {horizon}")
    print(f"  仿真步长: {sim_dt}s")
    print(f"  并行Workers: {max_workers}")
    print(f"  输出目录: {output_dir}")
    print(f"  时间戳: {timestamp}")
    print("=" * 60)

    # Filter months if specified
    if args.months != 'all':
        selected_months = [int(x.strip()) for x in args.months.split(',')]
        month_files = [f for i, f in enumerate(month_files, 1) if i in selected_months]
        print(f"处理选定月份: {selected_months}")
    else:
        print(f"发现 {len(month_files)} 个月份数据文件")

    print("=" * 60)

    # Prepare task arguments
    task_args = []
    for month_idx, month_file in enumerate(month_files, 1):
        task_args.append({
            'month_idx': month_idx,
            'month_file': month_file,
            'timestamp': timestamp,
            'output_dir': output_dir,
            'dt_ctrl': dt_ctrl,
            'horizon': horizon,
            'sim_dt': sim_dt,
            'T_ref': T_ref,
            'skip_warmup': args.skip_warmup
        })

    # Run parallel processing using joblib
    print(f"\nStarting parallel execution with {max_workers} workers...")

    if max_workers == 1:
        # Sequential execution for debugging
        results = [process_single_month(args_dict) for args_dict in task_args]
    else:
        # Parallel execution using joblib
        results = Parallel(n_jobs=max_workers, backend='loky')(
            delayed(process_single_month)(args_dict) for args_dict in task_args
        )

    # Process results
    successful = sum(1 for r in results if r)
    failed = len(results) - successful

    print("\n" + "=" * 60)
    print("生成完成")
    print("=" * 60)
    print(f"成功: {successful}/{len(task_args)}")
    print(f"失败: {failed}/{len(task_args)}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
