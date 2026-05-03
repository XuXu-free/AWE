"""
风电场景 HOCBF 安全控制器测试脚本

基于简化功率跟踪 + 二阶 HOCBF 安全投影，在风电功率曲线下运行 12 小时测试。

Usage:
    python scripts/multi_stack/run_wind_cbf_ho_test.py --duration 43200 --output_subdir wind_cbf_ho_12h
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from plant.multi_stack_simulator import MultiStackSimulator
from controller.multi_stack.cbf_projection_ho import MultiStackCBFProjectionHO


def load_power_profile(profile_path=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    if profile_path is None:
        profile_path = os.path.join(project_root, 'output', 'power', 'wind', 'wind_power_2025-12_1min.csv')
    if not os.path.exists(profile_path):
        raise FileNotFoundError(f"Power profile not found at {profile_path}")
    print(f"Loading power profile from {profile_path}...")
    df = pd.read_csv(profile_path)
    return df['P_ref'].values


def estimate_current_ref(P_ref, n_stacks=4, n_cells=368, U_approx=2.0):
    """Simplified current estimation from power reference."""
    I_est = (P_ref / n_stacks) / (U_approx * n_cells)
    return np.clip(I_est, 0.0, 9360.0)


def run_wind_cbf_ho_test(duration=43200, output_subdir='wind_cbf_ho',
                         alpha1_vec=None, alpha2_vec=None,
                         h_margin_vec=None, u_weight_scale=None,
                         lambda_u_scale=0.0, power_profile=None,
                         v_lye_ref=0.03, v_c_ref=0.0):
    sim_dt = 0.2
    dt_ctrl = 60.0
    T_ref = 353.15
    n_stacks = 4
    n_cells = 368

    sim = MultiStackSimulator(dt=sim_dt)
    sim.reset()

    # Output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    output_dir = os.path.join(project_root, 'output', 'multi_stack', 'test', output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    # Load power profile
    full_profile = load_power_profile(power_profile)
    if len(full_profile) < duration / 60:
        duration = int(len(full_profile) * 60)

    # HOCBF projector
    projector = MultiStackCBFProjectionHO(
        dt=dt_ctrl,
        gamma_vec=[10.0]*4 + [1.0] + [10.0]*4,
        rho_vec=[5000]*9,
        h_margin_vec=h_margin_vec if h_margin_vec is not None else [0.0]*4 + [0.001] + [0.0]*4,
        normalize=True,
        lambda_u_scale=lambda_u_scale,
        soft_mask=[True]*9,
        alpha1_hto=0.8,
        alpha2_hto=2.0,
        alpha1_vec=alpha1_vec,
        alpha2_vec=alpha2_vec,
        u_weight_scale=u_weight_scale,
    )

    # History
    history = {
        't': [], 'P_ref': [], 'P_real': [],
        'T_s_in': [], 'T_s_all': [], 'T_sep': [], 'T_c_out': [],
        'n_H2_an_vec': [], 'n_liq': [], 'n_gas': [],
        'T_ref': [],
        'I_all': [], 'v_lye_all': [], 'v_c': [],
        'U_cell_all': [], 'HTO': [], 'H2_rate': [],
        'solve_time_ms': [], 'projection_success': [],
        'I_ref': [], 'v_c_ref': [],
    }

    last_action = [
        np.ones(4) * 2000.0,
        np.ones(4) * v_lye_ref,
        v_c_ref
    ]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    data_filename = f"wind_cbf_ho_{timestamp}.csv"
    plot_every_steps = max(1, int(3600 / sim_dt))

    t_eval = np.arange(0, duration, sim_dt)
    ctrl_steps = int(dt_ctrl / sim_dt)
    profile_indices = (t_eval / 60).astype(int)
    profile_indices = np.clip(profile_indices, 0, len(full_profile) - 1)

    total_ctrl_steps = len(t_eval) // ctrl_steps

    print(f"Running WIND CBF-HO test for {duration}s ({duration/3600:.1f}h)")
    print(f"  Output dir: {output_dir}")
    print(f"  alpha1_vec: {alpha1_vec}")
    print(f"  alpha2_vec: {alpha2_vec}")
    print(f"  u_weight_scale: {u_weight_scale}")

    with tqdm(total=total_ctrl_steps, desc="Wind CBF-HO", unit="ctrl_step") as pbar:
        for i, t in enumerate(t_eval):
            state = sim.state
            T_s_in = state[0]
            T_s_vec = state[1:5]
            T_sep = state[5]
            T_c_out = state[6]
            n_H2_an_vec = state[7:11]
            n_liq = state[11]
            n_gas = state[12]

            Q, U_cell, eta = sim._calculate_electrochemical_properties(last_action[0], state[1:5])
            P_real = float(np.sum(U_cell * last_action[0] * sim.N_cell))

            hto_pct = float((n_gas * sim.R * T_sep) / (sim.P_sys * sim.V_sep_gas) * 100)
            h2_rate = float(np.sum(sim.N_cell * last_action[0] * eta / (2 * sim.F)))

            if i % ctrl_steps == 0:
                P_ref_val = full_profile[profile_indices[i]]

                # Build reference action from power tracking
                I_ref_val = estimate_current_ref(P_ref_val, n_stacks, n_cells)
                u_ref = np.array([I_ref_val] * n_stacks + [v_lye_ref] * n_stacks + [v_c_ref])

                # HOCBF projection
                last_action_vec = np.concatenate([last_action[0], last_action[1], [last_action[2]]])
                t0 = time.time()
                safe_action, success, info = projector.project(u_ref, state, u_last=last_action_vec)
                solve_time_ms = (time.time() - t0) * 1000.0

                I_cmd = safe_action[0:4]
                v_lye_cmd = safe_action[4:8]
                v_c_cmd = safe_action[8]

                action_sim = safe_action.copy()
                last_action = [I_cmd, v_lye_cmd, v_c_cmd]
            else:
                action_sim = np.concatenate([last_action[0], last_action[1], [last_action[2]]])
                P_ref_val = full_profile[profile_indices[i]]
                solve_time_ms = 0.0
                success = True
                I_ref_val = estimate_current_ref(P_ref_val, n_stacks, n_cells)

            sim.step(action_sim)

            # Log every 10s (50 steps)
            if i % 50 == 0:
                history['t'].append(t)
                history['P_ref'].append(P_ref_val)
                history['P_real'].append(P_real)
                history['T_s_in'].append(T_s_in)
                history['T_s_all'].append(T_s_vec.copy())
                history['T_sep'].append(T_sep)
                history['T_c_out'].append(T_c_out)
                history['n_H2_an_vec'].append(n_H2_an_vec.copy())
                history['n_liq'].append(n_liq)
                history['n_gas'].append(n_gas)
                history['T_ref'].append(T_ref)
                history['I_all'].append(last_action[0].copy())
                history['v_lye_all'].append(last_action[1].copy())
                history['v_c'].append(last_action[2])
                history['U_cell_all'].append(U_cell.copy())
                history['HTO'].append(hto_pct)
                history['H2_rate'].append(h2_rate)
                history['solve_time_ms'].append(solve_time_ms)
                history['projection_success'].append(success)
                history['I_ref'].append(I_ref_val)
                history['v_c_ref'].append(v_c_ref)

            if i % ctrl_steps == 0:
                pbar.set_postfix({
                    "t": f"{t/3600:.1f}h",
                    "P_ref": f"{P_ref_val/1e6:.1f}MW",
                    "P_real": f"{P_real/1e6:.1f}MW",
                    "T_s": f"{np.mean(T_s_vec)-273.15:.1f}C",
                    "HTO": f"{hto_pct:.2f}%"
                })
                pbar.update(1)

            if plot_every_steps is not None and i > 0 and i % plot_every_steps == 0:
                save_data_csv(history, output_dir, data_filename)
                save_plot(history, output_dir, data_filename)

    save_data_csv(history, output_dir, data_filename)
    save_plot(history, output_dir, data_filename)
    calculate_metrics(history, duration, output_dir, data_filename)
    return history


def save_data_csv(history, output_dir, filename):
    if len(history['t']) == 0:
        return
    t_arr = np.array(history['t'])
    T_s_all = np.array(history['T_s_all'])
    df = pd.DataFrame({
        't': t_arr,
        'P_ref': history['P_ref'],
        'P_real': history['P_real'],
        'T_s_in': history['T_s_in'],
        'T_sep': history['T_sep'],
        'T_c_out': history['T_c_out'],
        'T_ref': history['T_ref'],
        'HTO': history['HTO'],
        'H2_rate': history['H2_rate'],
        'v_c': history['v_c'],
        'solve_time_ms': history['solve_time_ms'],
        'projection_success': history['projection_success'],
        'I_ref': history['I_ref'],
        'v_c_ref': history['v_c_ref'],
    })
    for j in range(4):
        df[f'I_{j+1}'] = np.array(history['I_all'])[:, j]
        df[f'v_lye_{j+1}'] = np.array(history['v_lye_all'])[:, j]
        df[f'T_s_{j+1}'] = T_s_all[:, j]
        df[f'U_cell_{j+1}'] = np.array(history['U_cell_all'])[:, j]
    df.to_csv(os.path.join(output_dir, filename), index=False)


def save_plot(history, output_dir, filename):
    t_arr = np.array(history['t'])
    if len(t_arr) == 0:
        return
    T_s_all = np.array(history['T_s_all'])
    U_cell_all = np.array(history['U_cell_all'])
    I_all = np.array(history['I_all'])
    v_lye_all = np.array(history['v_lye_all'])

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    ax = axes[0, 0]
    ax.plot(t_arr / 3600, np.array(history['P_ref']) / 1e6, 'k--', label='Ref')
    ax.plot(t_arr / 3600, np.array(history['P_real']) / 1e6, 'b-', label='Real')
    ax.set_title('Power Tracking')
    ax.set_ylabel('MW')
    ax.set_xlabel('Time (h)')
    ax.legend()
    ax.grid(True)

    ax = axes[0, 1]
    ax.plot(t_arr / 3600, history['T_sep'], 'g--', label='Separator')
    ax.plot(t_arr / 3600, history['T_c_out'], 'c:', label='CW Out')
    for i in range(4):
        ax.plot(t_arr / 3600, T_s_all[:, i], label=f'Stack {i+1}')
    ax.plot(t_arr / 3600, history['T_ref'], 'k--', linewidth=1.5, label='Ref Temp')
    ax.set_title('All Temperatures')
    ax.set_ylabel('K')
    ax.set_xlabel('Time (h)')
    ax.legend()
    ax.grid(True)

    ax = axes[0, 2]
    ax.plot(t_arr / 3600, history['HTO'], 'm-', label='HTO')
    ax.axhline(y=2.0, color='r', linestyle='--', alpha=0.5, label='Limit (2%)')
    ax.set_title('HTO (%)')
    ax.set_ylabel('%')
    ax.set_xlabel('Time (h)')
    ax.legend()
    ax.grid(True)

    ax = axes[1, 0]
    for i in range(4):
        ax.plot(t_arr / 3600, I_all[:, i], label=f'Stack {i+1}')
    ax.set_title('Stack Currents')
    ax.set_ylabel('Amps')
    ax.set_xlabel('Time (h)')
    ax.legend()
    ax.grid(True)

    ax = axes[1, 1]
    for i in range(4):
        ax.plot(t_arr / 3600, U_cell_all[:, i], label=f'Stack {i+1}')
    ax.set_title('Stack Voltages')
    ax.set_ylabel('Volts')
    ax.set_xlabel('Time (h)')
    ax.legend()
    ax.grid(True)

    ax = axes[1, 2]
    for i in range(4):
        ax.plot(t_arr / 3600, v_lye_all[:, i], label=f'Stack {i+1}')
    ax.set_title('Stack Lye Flow')
    ax.set_ylabel('m3/s')
    ax.set_xlabel('Time (h)')
    ax.legend()
    ax.grid(True)

    ax = axes[2, 0]
    ax.plot(t_arr / 3600, history['v_c'], 'cyan', label='CW')
    ax.plot(t_arr / 3600, history['v_c_ref'], 'r--', alpha=0.5, label='CW Ref')
    ax.set_title('Coolant Flow')
    ax.set_ylabel('m3/s')
    ax.set_xlabel('Time (h)')
    ax.legend()
    ax.grid(True)

    ax = axes[2, 1]
    ax.plot(t_arr / 3600, history['H2_rate'], 'g-', label='H2 Rate')
    ax.set_title('H2 Production Rate')
    ax.set_ylabel('mol/s')
    ax.set_xlabel('Time (h)')
    ax.legend()
    ax.grid(True)

    ax = axes[2, 2]
    solve_times = np.array(history['solve_time_ms'])
    if np.max(solve_times) > 0:
        ax.plot(t_arr / 3600, solve_times, 'b-', label='Solve Time')
        ax.set_title('CBF Solve Time')
        ax.set_ylabel('ms')
        ax.set_xlabel('Time (h)')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, filename.replace('.csv', '.png'))
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {plot_path}")


def calculate_metrics(history, duration, output_dir, filename):
    t_arr = np.array(history['t'])
    P_ref_arr = np.array(history['P_ref'])
    P_real_arr = np.array(history['P_real'])
    T_s_all = np.array(history['T_s_all'])
    T_s_mean = np.mean(T_s_all, axis=1) if T_s_all.ndim > 1 else T_s_mean
    T_ref_arr = np.array(history['T_ref'])
    HTO_arr = np.array(history['HTO'])
    I_all = np.array(history['I_all'])
    solve_times = np.array(history['solve_time_ms'])

    rmse_p = float(np.sqrt(np.mean((P_real_arr - P_ref_arr)**2)) / 1e6)
    rmse_t = float(np.sqrt(np.mean((T_s_mean - T_ref_arr)**2)))
    max_hto = float(np.max(HTO_arr))
    max_temp = float(np.max(T_s_all)) if T_s_all.ndim > 1 else 0.0
    max_temp_c = max_temp - 273.15
    mean_solve = float(np.mean(solve_times[solve_times > 0])) if np.any(solve_times > 0) else 0.0

    dI = np.diff(I_all, axis=0)
    mean_dI = float(np.mean(np.abs(dI))) if len(dI) > 0 else 0.0

    metrics = {
        "duration": duration,
        "rmse_power_mw": rmse_p,
        "rmse_temp_k": rmse_t,
        "max_hto_pct": max_hto,
        "max_temp_c": max_temp_c,
        "mean_solve_ms": mean_solve,
        "mean_abs_dI": mean_dI,
    }

    json_filename = filename.replace('.csv', '_metrics.json')
    json_path = os.path.join(output_dir, json_filename)
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print("=" * 60)
    print(f"Wind CBF-HO Test Metrics ({duration/3600:.1f}h)")
    print(f"  Power RMSE:  {rmse_p:.3f} MW")
    print(f"  Temp RMSE:   {rmse_t:.3f} K")
    print(f"  Max HTO:     {max_hto:.3f}%")
    print(f"  Max Temp:    {max_temp_c:.2f} C")
    print(f"  Mean |dI|:   {mean_dI:.1f} A/step")
    print(f"  Mean Solve:  {mean_solve:.2f} ms")
    print(f"  Saved to:    {json_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Wind CBF-HO Test')
    parser.add_argument('--duration', type=int, default=43200, help='Test duration in seconds (default: 43200 = 12h)')
    parser.add_argument('--output_subdir', type=str, default='wind_cbf_ho', help='Output subdirectory')
    parser.add_argument('--power_profile', type=str, default=None, help='Path to power profile CSV')
    parser.add_argument('--alpha1_vec', type=float, nargs=9, default=None)
    parser.add_argument('--alpha2_vec', type=float, nargs=9, default=None)
    parser.add_argument('--h_margin_vec', type=float, nargs=9, default=None)
    parser.add_argument('--u_weight_scale', type=float, nargs=9, default=None)
    parser.add_argument('--lambda_u_scale', type=float, default=0.0)
    parser.add_argument('--v_lye_ref', type=float, default=0.03)
    parser.add_argument('--v_c_ref', type=float, default=0.0)
    args = parser.parse_args()

    alpha1_vec = list(args.alpha1_vec) if args.alpha1_vec is not None else None
    alpha2_vec = list(args.alpha2_vec) if args.alpha2_vec is not None else None
    h_margin_vec = list(args.h_margin_vec) if args.h_margin_vec is not None else None
    u_weight_scale = list(args.u_weight_scale) if args.u_weight_scale is not None else None

    run_wind_cbf_ho_test(
        duration=args.duration,
        output_subdir=args.output_subdir,
        alpha1_vec=alpha1_vec,
        alpha2_vec=alpha2_vec,
        h_margin_vec=h_margin_vec,
        u_weight_scale=u_weight_scale,
        lambda_u_scale=args.lambda_u_scale,
        power_profile=args.power_profile,
        v_lye_ref=args.v_lye_ref,
        v_c_ref=args.v_c_ref,
    )


if __name__ == "__main__":
    main()
