import os
import sys
from typing import Any
import numpy as np
import pandas as pd
from datetime import datetime
import argparse
from tqdm import tqdm

import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from plant.multi_stack_simulator import MultiStackSimulator
from controller.multi_stack.nmpc_controller import MultiStackNMPCController
from controller.multi_stack.nmpc_simplified_controller import MultiStackNMPCSimplifiedController
from controller.multi_stack.model_controller import MultiStackModelController
from controller.multi_stack.model_dynamic_controller import MultiStackModelDynamicController

def add_measurement_noise(state):
    return np.copy(state)

def run_warmup_phase(sim, ctrl, history, last_action, dt, output_dir, filename_prefix="warmup", T_ref=353.15, warmup_ctrl=None):
    warmup_duration = 14400
    warmup_steps = int(warmup_duration / dt)
    warmup_P_ref = 10.0e6

    active_ctrl = warmup_ctrl if warmup_ctrl is not None else ctrl
    print(f"Starting Warm-up Phase ({warmup_duration}s at {warmup_P_ref/1e6}MW) using {active_ctrl.__class__.__name__}...")

    P_future = [warmup_P_ref] * active_ctrl.horizon
    ctrl_steps = int(active_ctrl.dt / dt)
    total_ctrl_steps = warmup_steps // ctrl_steps
    warmup_profile = np.array([warmup_P_ref], dtype=float)
    warmup_profile_indices = np.zeros(warmup_steps, dtype=int)
    plot_every_steps = max(1, int(3600 / dt))
    warmup_data_filename = f"{filename_prefix}.csv"

    with tqdm(total=total_ctrl_steps, desc="Warmup", unit="ctrl_step") as pbar:
        for i in range(warmup_steps):
            t_warmup = -warmup_duration + i * dt
            measured_state = add_measurement_noise(sim.state)
            T_s_in = measured_state[0]
            T_s_vec = measured_state[1:5]
            T_sep = measured_state[5]
            T_c_out = measured_state[6]
            n_H2_an_vec = measured_state[7:11]
            n_liq = measured_state[11]
            n_gas = measured_state[12]
            Q, U_cell, eta = sim._calculate_electrochemical_properties(last_action[0], sim.state[1:5])
            P_real = np.sum(U_cell * last_action[0] * sim.N_cell)
            n_H2_sep_gas = measured_state[12]
            hto_pct = (n_H2_sep_gas * sim.R * T_sep) / (sim.P_sys * sim.V_sep_gas) * 100
            h2_rate = np.sum(sim.N_cell * last_action[0] * eta / (2 * sim.F))
            prev_action = [np.copy(last_action[0]), np.copy(last_action[1]), last_action[2]]

            if i % ctrl_steps == 0:
                I_cmd, v_lye_cmd, v_c_cmd = active_ctrl.get_action(
                    measured_state, P_future, T_ref=T_ref, last_action=last_action
                )
                action_sim = np.concatenate([I_cmd, v_lye_cmd, [v_c_cmd]])
                last_action = [I_cmd, v_lye_cmd, v_c_cmd]
            else:
                action_sim = np.concatenate([last_action[0], last_action[1], [last_action[2]]])

            sim.step(action_sim)
            log_and_visualize(
                i=i, t=t_warmup, history=history, full_profile=warmup_profile,
                profile_indices=warmup_profile_indices, P_real=P_real,
                T_s_in=T_s_in, T_s_vec=T_s_vec, T_sep=T_sep, T_c_out=T_c_out,
                n_H2_an_vec=n_H2_an_vec, n_liq=n_liq, n_gas=n_gas, T_ref=T_ref,
                P_future=P_future, prev_action=prev_action, last_action=last_action,
                U_cell=U_cell, hto_pct=hto_pct, h2_rate=h2_rate,
                ctrl_steps=ctrl_steps, pbar=pbar, data_filename=warmup_data_filename,
                output_dir=output_dir, plot_every_steps=plot_every_steps,
            )

    print("Warm-up Complete. Starting Step Response Test...")
    return last_action

def log_and_visualize(i, t, history, full_profile, profile_indices, P_real, T_s_in, T_s_vec, T_sep, T_c_out, n_H2_an_vec, n_liq, n_gas, T_ref, P_future, prev_action, last_action, U_cell, hto_pct, h2_rate, ctrl_steps, pbar, data_filename, output_dir, plot_every_steps):
    if i % 50 == 0:
        history['t'].append(t)
        history['P_ref'].append(full_profile[profile_indices[i]] if i < len(profile_indices) else full_profile[-1])
        history['P_real'].append(P_real)
        history['T_s_in'].append(T_s_in)
        history['T_s_all'].append(T_s_vec)
        history['T_sep'].append(T_sep)
        history['T_c_out'].append(T_c_out)
        history['n_H2_an_vec'].append(n_H2_an_vec)
        history['n_liq'].append(n_liq)
        history['n_gas'].append(n_gas)
        history['T_ref'].append(T_ref)
        history['P_ref_future'].append(P_future)
        history['I_prev'].append(prev_action[0])
        history['v_lye_prev'].append(prev_action[1])
        history['v_c_prev'].append(prev_action[2])
        history['I_all'].append(last_action[0])
        history['v_lye_all'].append(last_action[1])
        history['v_c'].append(last_action[2])
        history['U_cell_all'].append(U_cell)
        history['HTO'].append(hto_pct)
        history['H2_rate'].append(h2_rate)

    if i % ctrl_steps == 0:
        idx = profile_indices[i] if i < len(profile_indices) else len(full_profile) - 1
        P_ref_val = full_profile[idx]
        pbar.set_postfix({
            "t": f"{t:.0f}s",
            "P_ref": f"{P_ref_val/1e6:.1f}MW",
            "P_real": f"{P_real/1e6:.1f}MW",
            "T_s": f"{np.mean(T_s_vec)-273.15:.1f}C"
        })
        pbar.update(1)

    if plot_every_steps is not None and i > 0 and i % plot_every_steps == 0:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_data_csv(history, output_dir, data_filename)
        save_plot(history, output_dir, data_filename)

def calculate_step_metrics(history, step_time, P_initial, P_final, output_dir, filename):
    t_arr = np.array(history['t'])
    P_ref_arr = np.array(history['P_ref'])
    P_real_arr = np.array(history['P_real'])
    T_s_all = np.array(history['T_s_all'])
    if T_s_all.ndim > 1:
        T_s_mean = np.mean(T_s_all, axis=1)
    else:
        T_s_mean = np.zeros_like(t_arr)
    T_ref_arr = np.array(history['T_ref'])

    # Only evaluate after step
    post_step_mask = t_arr >= step_time
    t_post = t_arr[post_step_mask]
    P_real_post = P_real_arr[post_step_mask]
    T_s_post = T_s_mean[post_step_mask]

    # Power tracking RMSE after step
    rmse_p = float(np.sqrt(np.mean((P_real_post - P_ref_arr[post_step_mask])**2)) / 1e6)
    rmse_t = float(np.sqrt(np.mean((T_s_post - T_ref_arr[post_step_mask])**2)))

    # Step response metrics
    delta_P = P_final - P_initial

    # Rise time: 10% to 90%
    idx_post = np.where(post_step_mask)[0]
    if len(idx_post) > 0 and delta_P > 0:
        P_norm = (P_real_post - P_initial) / delta_P
        idx_10 = None
        idx_90 = None
        for j, val in enumerate(P_norm):
            if idx_10 is None and val >= 0.1:
                idx_10 = idx_post[j]
            if idx_90 is None and val >= 0.9:
                idx_90 = idx_post[j]
                break
        rise_time = float(t_arr[idx_90] - t_arr[idx_10]) if (idx_10 is not None and idx_90 is not None) else None

        # Overshoot
        overshoot_pct = float((np.max(P_real_post) - P_final) / delta_P * 100)

        # Settling time: within +/- 5% of final
        settled = False
        settling_time = None
        for j in range(len(P_norm)):
            if np.all(np.abs(P_norm[j:] - 1.0) <= 0.05):
                settling_time = float(t_post[j] - step_time)
                break
    else:
        rise_time = None
        overshoot_pct = None
        settling_time = None

    metrics = {
        "step_time_s": step_time,
        "P_initial_mw": P_initial / 1e6,
        "P_final_mw": P_final / 1e6,
        "rmse_power_mw": rmse_p,
        "rmse_temp_k": rmse_t,
        "rise_time_s": rise_time,
        "overshoot_pct": overshoot_pct,
        "settling_time_s": settling_time,
    }

    json_filename = filename.replace('.csv', '_metrics.json')
    json_path = os.path.join(output_dir, json_filename)
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print("-" * 50)
    print(f"Step Response Metrics (Step at t={step_time}s)")
    print(f"Power RMSE (post-step): {rmse_p:.3f} MW")
    print(f"Temp RMSE (post-step):  {rmse_t:.3f} K")
    if rise_time is not None:
        print(f"Rise Time (10%-90%):    {rise_time:.1f} s")
    if overshoot_pct is not None:
        print(f"Overshoot:              {overshoot_pct:.2f} %")
    if settling_time is not None:
        print(f"Settling Time (+-5%):   {settling_time:.1f} s")
    print(f"Metrics saved to {json_path}")
    print("-" * 50)

def run_step_response(controller_type='nmpc', model_type='tcn', duration=7200, step_time=3600,
                      P_initial=10.0e6, P_final=20.0e6,
                      warmup_controller='none', output_subdir=None):
    sim_dt = 0.2
    dt_ctrl = 60.0
    horizon = 5
    T_ref = 353.15

    sim = MultiStackSimulator(dt=sim_dt)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    stats_path = os.path.join(project_root, 'output', 'multi_stack', 'diffusion_stats.npz')

    if output_subdir:
        output_dir = os.path.join(project_root, 'output', 'multi_stack', 'test_step', output_subdir)
    else:
        output_dir = os.path.join(project_root, 'output', 'multi_stack', 'test_step')
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(project_root, 'output', 'multi_stack', 'policy', f'{model_type}_policy_best.pth')

    if controller_type == 'nmpc':
        ctrl = MultiStackNMPCController(dt=dt_ctrl, horizon=horizon, dt_sub=sim_dt)
        print("Using Full-Order NMPC Controller")
    elif controller_type == 'nmpc_simplified':
        ctrl = MultiStackNMPCSimplifiedController(dt=dt_ctrl, horizon=horizon, dt_sub=sim_dt)
        print("Using Simplified NMPC Controller")
    elif controller_type == 'model':
        ctrl = MultiStackModelController(dt=dt_ctrl, horizon=horizon, model_type=model_type,
                                         model_path=model_path, stats_path=stats_path)
        print(f"Using Model Controller ({model_type})")
    elif controller_type == 'model_dynamic':
        ctrl = MultiStackModelDynamicController(dt=dt_ctrl, horizon=horizon, model_type=model_type,
                                                model_path=model_path, stats_path=stats_path)
        print(f"Using Model Dynamic Controller ({model_type})")
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")

    sim.reset()
    t_eval = np.arange(0, duration, sim_dt)
    print(f"Running step response test for {duration}s ({duration/3600:.1f}h)")
    print(f"Step at t={step_time}s: {P_initial/1e6:.1f}MW -> {P_final/1e6:.1f}MW")

    # Build step profile
    full_profile = np.full(len(t_eval), P_initial)
    step_idx = int(step_time / sim_dt)
    full_profile[step_idx:] = P_final
    profile_indices = np.arange(len(t_eval))

    history = {
        't': [], 'P_ref': [], 'P_real': [],
        'T_s_in': [], 'T_s_all': [], 'T_sep': [], 'T_c_out': [],
        'n_H2_an_vec': [], 'n_liq': [], 'n_gas': [],
        'T_ref': [], 'P_ref_future': [],
        'I_prev': [], 'v_lye_prev': [], 'v_c_prev': [],
        'I_all': [], 'v_lye_all': [], 'v_c': [],
        'U_cell_all': [], 'HTO': [], 'H2_rate': []
    }

    # Asymmetric initial conditions for each stack
    last_action = [
        np.array([1800.0, 2100.0, 2200.0, 1900.0]),   # I (different per stack)
        np.array([0.025, 0.032, 0.035, 0.028]),        # v_lye (different per stack)
        0.0
    ]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if controller_type == 'model':
        data_filename = f"step_{controller_type}_{model_type}_data_{timestamp}.csv"
    else:
        data_filename = f"step_{controller_type}_data_{timestamp}.csv"
    plot_every_steps = max(1, int(3600 / sim_dt))

    # Warm-up at P_initial
    if warmup_controller == 'none':
        print("Skipping warm-up phase (--warmup_controller=none)")
    else:
        print(f"Creating {warmup_controller} controller for warm-up phase at {P_initial/1e6:.1f}MW...")
        if warmup_controller == 'nmpc_simplified':
            warmup_ctrl = MultiStackNMPCSimplifiedController(dt=dt_ctrl, horizon=horizon, dt_sub=sim_dt)
        elif warmup_controller == 'nmpc':
            warmup_ctrl = MultiStackNMPCController(dt=dt_ctrl, horizon=horizon, dt_sub=sim_dt)
        elif warmup_controller in ['model', 'model_dynamic']:
            warmup_ctrl = MultiStackModelController(dt=dt_ctrl, horizon=horizon, model_type=model_type,
                                                     model_path=model_path, stats_path=stats_path)
        else:
            raise ValueError(f"Unknown warmup controller type: {warmup_controller}")

        last_action = run_warmup_phase(
            sim, ctrl, history, last_action, sim_dt,
            output_dir=output_dir,
            filename_prefix=f"step_warmup_{timestamp}",
            T_ref=T_ref,
            warmup_ctrl=warmup_ctrl
        )

    print(f"Starting Step Response {controller_type.upper()} Test...")
    ctrl_steps = int(dt_ctrl / sim_dt)
    total_ctrl_steps = len(t_eval) // ctrl_steps

    with tqdm(total=total_ctrl_steps, desc="StepResponse", unit="ctrl_step") as pbar:
        for i, t in enumerate(t_eval):
            measured_state = add_measurement_noise(sim.state)
            T_s_in = measured_state[0]
            T_s_vec = measured_state[1:5]
            T_sep = measured_state[5]
            T_c_out = measured_state[6]
            n_H2_an_vec = measured_state[7:11]
            n_liq = measured_state[11]
            n_gas = measured_state[12]
            Q, U_cell, eta = sim._calculate_electrochemical_properties(last_action[0], sim.state[1:5])
            P_real = np.sum(U_cell * last_action[0] * sim.N_cell)
            n_H2_sep_gas = measured_state[12]
            hto_pct = (n_H2_sep_gas * sim.R * T_sep) / (sim.P_sys * sim.V_sep_gas) * 100
            h2_rate = np.sum(sim.N_cell * last_action[0] * eta / (2 * sim.F))
            prev_action = [np.copy(last_action[0]), np.copy(last_action[1]), last_action[2]]

            if i % ctrl_steps == 0:
                idx_min = i
                end_idx = idx_min + horizon
                P_future = full_profile[idx_min:end_idx]
                if len(P_future) < horizon:
                    padding = np.full(horizon - len(P_future), full_profile[-1])
                    P_future = np.concatenate([P_future, padding])
                P_future = list[Any](P_future)

                I_cmd, v_lye_cmd, v_c_cmd = ctrl.get_action(
                    measured_state, P_future, T_ref, last_action
                )
                action_sim = np.concatenate([I_cmd, v_lye_cmd, [v_c_cmd]])
                last_action = [I_cmd, v_lye_cmd, v_c_cmd]
            else:
                action_sim = np.concatenate([last_action[0], last_action[1], [last_action[2]]])

            sim.step(action_sim)
            log_and_visualize(i, t, history, full_profile, profile_indices, P_real, T_s_in, T_s_vec, T_sep, T_c_out, n_H2_an_vec, n_liq, n_gas, T_ref, P_future, prev_action, last_action, U_cell, hto_pct, h2_rate, ctrl_steps, pbar, data_filename, output_dir, plot_every_steps)

    save_data_csv(history, output_dir, data_filename)
    save_plot(history, output_dir, data_filename)
    calculate_step_metrics(history, step_time, P_initial, P_final, output_dir, data_filename)

def save_plot(history, output_dir, filename):
    t_arr = np.array(history['t'])
    if len(t_arr) == 0:
        return

    T_s_all = np.array(history['T_s_all'])
    U_cell_all = np.array(history['U_cell_all'])
    I_all = np.array(history['I_all'])
    v_lye_all = np.array(history['v_lye_all'])

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    ax = axes[0,0]
    ax.plot(t_arr, np.array(history['P_ref'])/1e6, 'k--', label='Ref')
    ax.plot(t_arr, np.array(history['P_real'])/1e6, 'b-', label='Real')
    ax.axvline(x=0, color='gray', linestyle=':', label='Start')
    ax.set_title('Power Tracking (Total)')
    ax.set_ylabel('MW')
    ax.legend()
    ax.grid(True)

    ax = axes[0,1]
    ax.plot(t_arr, history['T_sep'], 'g--', label='Separator')
    ax.plot(t_arr, history['T_c_out'], 'c:', label='CW Out')
    for i in range(4):
        ax.plot(t_arr, T_s_all[:, i], label=f'Stack {i+1}')
    ax.plot(t_arr, history['T_ref'], 'k--', linewidth=1.5, label='Ref Temp')
    ax.axvline(x=0, color='gray', linestyle=':')
    ax.set_title('All Temperatures')
    ax.set_ylabel('K')
    ax.legend()
    ax.grid(True)

    ax = axes[0,2]
    ax.plot(t_arr, history['HTO'], 'm-', label='HTO')
    ax.axvline(x=0, color='gray', linestyle=':')
    ax.set_title('HTO (H2 in O2)')
    ax.set_ylabel('%')
    ax.grid(True)

    ax = axes[1,0]
    for i in range(4):
        ax.plot(t_arr, I_all[:, i], label=f'Stack {i+1}')
    ax.axvline(x=0, color='gray', linestyle=':')
    ax.set_title('Stack Currents')
    ax.set_ylabel('Amps')
    ax.legend()
    ax.grid(True)

    ax = axes[1,1]
    for i in range(4):
        ax.plot(t_arr, U_cell_all[:, i], label=f'Stack {i+1}')
    ax.axvline(x=0, color='gray', linestyle=':')
    ax.set_title('Stack Voltages (Per Cell)')
    ax.set_ylabel('Volts')
    ax.legend()
    ax.grid(True)

    ax = axes[1,2]
    for i in range(4):
        ax.plot(t_arr, v_lye_all[:, i], label=f'Stack {i+1}')
    ax.axvline(x=0, color='gray', linestyle=':')
    ax.set_title('Stack Lye Flow')
    ax.set_ylabel('m3/s')
    ax.legend()
    ax.grid(True)

    ax = axes[2,0]
    ax.plot(t_arr, history['v_c'], 'cyan', label='CW')
    ax.axvline(x=0, color='gray', linestyle=':')
    ax.set_title('Coolant Flow')
    ax.set_ylabel('m3/s')
    ax.grid(True)

    ax = axes[2,1]
    ax.plot(t_arr, history['H2_rate'], 'g-', label='H2 Rate')
    ax.axvline(x=0, color='gray', linestyle=':')
    ax.set_title('Total H2 Production Rate')
    ax.set_ylabel('mol/s')
    ax.grid(True)

    ax = axes[2,2]
    ax.axis('off')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, filename.replace('.csv', '.png'))
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.close(fig)

def save_data_csv(history, output_dir, filename):
    keys_scalar = ['t', 'P_ref', 'P_real', 'T_s_in', 'T_sep', 'T_c_out', 'n_liq', 'n_gas', 'T_ref', 'v_c_prev', 'v_c', 'HTO', 'H2_rate']
    data_dict = {}
    for key in keys_scalar:
        if key in history:
            data_dict[key] = history[key]
    for key in ['T_s_all', 'n_H2_an_vec', 'I_all', 'v_lye_all', 'U_cell_all']:
        if key in history:
            vec_data = np.array(history[key])
            if vec_data.size > 0:
                for k in range(4):
                    data_dict[f"{key}_{k+1}"] = vec_data[:, k]
    df = pd.DataFrame(data_dict)
    file_path = os.path.join(output_dir, filename)
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Multi-Stack Step Response Test')
    parser.add_argument('--controller', type=str, default='model', choices=['nmpc', 'nmpc_simplified', 'model', 'model_dynamic'], help='Controller type')
    parser.add_argument('--model_type', type=str, default='flow_tcn',
                        choices=['diffusion_mlp', 'diffusion_pure_mlp', 'diffusion_tcn', 'diffusion_tcn_l3', 'guided_diffusion_mlp', 'guided_diffusion_tcn', 'deterministic_diffusion_mlp', 'deterministic_diffusion_pure_mlp', 'deterministic_diffusion_tcn', 'deterministic_guided_diffusion_mlp', 'deterministic_guided_diffusion_tcn', 'early_stop_diffusion_mlp', 'early_stop_diffusion_tcn', 'early_stop_guided_diffusion_mlp', 'early_stop_guided_diffusion_tcn', 'flow_mlp', 'flow_tcn', 'flow_mlp_hardflow', 'flow_tcn_hardflow', 'mlp', 'tcn', 'flow_matching', 'pure_mlp', 'pure_tcn', 'lstm'],
                        help='Model type (only for model controller)')
    parser.add_argument('--duration', type=int, default=7200, help='Duration of test in seconds (default 2h)')
    parser.add_argument('--step_time', type=int, default=3600, help='Time of step in seconds (default 1h)')
    parser.add_argument('--P_initial', type=float, default=10.0e6, help='Initial power in W (default 10MW)')
    parser.add_argument('--P_final', type=float, default=20.0e6, help='Final power in W (default 20MW)')
    parser.add_argument('--warmup_controller', type=str, default='none',
                        choices=['none', 'nmpc', 'nmpc_simplified', 'model'],
                        help='Controller type for warm-up phase (default: none)')
    parser.add_argument('--output_subdir', type=str, default=None,
                        help='Subdirectory under output/multi_stack/test_step/ to store results')
    args = parser.parse_args()

    run_step_response(
        controller_type=args.controller, model_type=args.model_type,
        duration=args.duration, step_time=args.step_time,
        P_initial=args.P_initial, P_final=args.P_final,
        warmup_controller=args.warmup_controller, output_subdir=args.output_subdir
    )
