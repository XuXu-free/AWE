
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
from controller.multi_stack.model_controller import MultiStackModelController
from controller.multi_stack.model_dynamic_controller import MultiStackModelDynamicController

def add_measurement_noise(state):
    """
    Returns the state as-is without adding noise.
    """
    return np.copy(state)

def load_december_profile():
    # Get project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    
    profile_path = os.path.join(project_root, 'output', 'power', 'wind', 'wind_power_2025-12_1min.csv')
    
    if not os.path.exists(profile_path):
        raise FileNotFoundError(f"December profile not found at {profile_path}")
        
    print(f"Loading December profile from {profile_path}...")
    df = pd.read_csv(profile_path)
    return df['P_ref'].values

def run_warmup_phase(sim, ctrl, history, last_action, dt, output_dir, filename_prefix="warmup", T_ref=353.15, warmup_ctrl=None):
    """
    Run a warmup phase at constant power to stabilize temperatures.
    If warmup_ctrl is provided, use it for warmup instead of ctrl.
    """
    warmup_duration = 14400 # seconds (4h)
    warmup_steps = int(warmup_duration / dt)
    warmup_P_ref = 10.0e6 # 10MW constant

    # Use warmup_ctrl if provided, otherwise use the main ctrl
    active_ctrl = warmup_ctrl if warmup_ctrl is not None else ctrl

    print(f"Starting Warm-up Phase ({warmup_duration}s at {warmup_P_ref/1e6}MW) using {active_ctrl.__class__.__name__}...")

    # Pre-calculate future profile for warmup (constant)
    P_future = [warmup_P_ref] * active_ctrl.horizon
    
    # Control interval steps (use active_ctrl's dt)
    ctrl_steps = int(active_ctrl.dt / dt)
    total_ctrl_steps = warmup_steps // ctrl_steps
    warmup_profile = np.array([warmup_P_ref], dtype=float)
    warmup_profile_indices = np.zeros(warmup_steps, dtype=int)
    plot_every_steps = max(1, int(3600 / dt))
    warmup_data_filename = f"{filename_prefix}.csv"
    
    with tqdm(total=total_ctrl_steps, desc="Warmup", unit="ctrl_step") as pbar:
        for i in range(warmup_steps):
            t_warmup = -warmup_duration + i * dt
            
            # Measure State (with noise)
            measured_state = add_measurement_noise(sim.state)
            
            # Extract State (from measurements)
            T_s_in = measured_state[0]
            T_s_vec = measured_state[1:5]
            T_sep = measured_state[5]
            T_c_out = measured_state[6]
            n_H2_an_vec = measured_state[7:11]
            n_liq = measured_state[11]
            n_gas = measured_state[12]
            
            # Calculate Real Power (using TRUE state for physics)
            Q, U_cell, eta = sim._calculate_electrochemical_properties(last_action[0], sim.state[1:5])
            P_real = np.sum(U_cell * last_action[0] * sim.N_cell)
            
            # Calculate Extra Metrics (using measurements)
            n_H2_sep_gas = measured_state[12]
            # T_sep is already Kelvin
            hto_pct = (n_H2_sep_gas * sim.R * T_sep) / (sim.P_sys * sim.V_sep_gas) * 100
            
            h2_rate = np.sum(sim.N_cell * last_action[0] * eta / (2 * sim.F))
            
            # Store previous action before update
            prev_action = [np.copy(last_action[0]), np.copy(last_action[1]), last_action[2]]

            # Control Update (using active_ctrl for warmup)
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
                i=i,
                t=t_warmup,
                history=history,
                full_profile=warmup_profile,
                profile_indices=warmup_profile_indices,
                P_real=P_real,
                T_s_in=T_s_in,
                T_s_vec=T_s_vec,
                T_sep=T_sep,
                T_c_out=T_c_out,
                n_H2_an_vec=n_H2_an_vec,
                n_liq=n_liq,
                n_gas=n_gas,
                T_ref=T_ref,
                P_future=P_future,
                prev_action=prev_action,
                last_action=last_action,
                U_cell=U_cell,
                hto_pct=hto_pct,
                h2_rate=h2_rate,
                ctrl_steps=ctrl_steps,
                pbar=pbar,
                data_filename=warmup_data_filename,
                output_dir=output_dir,
                plot_every_steps=plot_every_steps,
            )

    print("Warm-up Complete. Starting Main Test...")
    return last_action

def log_and_visualize(i, t, history, full_profile, profile_indices, P_real, T_s_in, T_s_vec, T_sep, T_c_out, n_H2_an_vec, n_liq, n_gas, T_ref, P_future, prev_action, last_action, U_cell, hto_pct, h2_rate, ctrl_steps, pbar, data_filename, output_dir, plot_every_steps):
    # Log (downsampled) - Every 10s (50 steps)
    if i % 50 == 0:
        history['t'].append(t)
        history['P_ref'].append(full_profile[profile_indices[i]])
        history['P_real'].append(P_real)
        history['T_s_in'].append(T_s_in)
        history['T_s_all'].append(T_s_vec)
        history['T_sep'].append(T_sep)
        history['T_c_out'].append(T_c_out)
        history['n_H2_an_vec'].append(n_H2_an_vec)
        history['n_liq'].append(n_liq)
        history['n_gas'].append(n_gas)
        history['T_ref'].append(T_ref)
        history['P_ref_future'].append(P_future) # Note: P_future from last control step
        history['I_prev'].append(prev_action[0])
        history['v_lye_prev'].append(prev_action[1])
        history['v_c_prev'].append(prev_action[2])
        history['I_all'].append(last_action[0])
        history['v_lye_all'].append(last_action[1])
        history['v_c'].append(last_action[2])
        history['U_cell_all'].append(U_cell)
        history['HTO'].append(hto_pct)
        history['H2_rate'].append(h2_rate)
    
    # Update progress bar (on control steps)
    if i % ctrl_steps == 0:
        P_ref_val = full_profile[profile_indices[i]]
        pbar.set_postfix({
            "t": f"{t:.0f}s",
            "P_ref": f"{P_ref_val/1e6:.1f}MW",
            "P_real": f"{P_real/1e6:.1f}MW",
            "T_s": f"{np.mean(T_s_vec)-273.15:.1f}C"
        })
        pbar.update(1)
    
    # Periodic Plot Update (every 3600s)
    if plot_every_steps is not None and i > 0 and i % plot_every_steps == 0:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_data_csv(history, output_dir, data_filename)
        save_plot(history, output_dir, data_filename)

import json

def calculate_metrics(history, duration, output_dir, filename):
    # Calculate RMSE
    t_arr = np.array(history['t'])
    P_ref_arr = np.array(history['P_ref'])
    P_real_arr = np.array(history['P_real'])
    T_s_all = np.array(history['T_s_all'])
    if T_s_all.ndim > 1:
        T_s_mean = np.mean(T_s_all, axis=1)
    else:
        T_s_mean = np.zeros_like(t_arr)
    T_ref_arr = np.array(history['T_ref'])
    
    # Metrics
    rmse_p = float(np.sqrt(np.mean((P_real_arr - P_ref_arr)**2)) / 1e6)
    rmse_t = float(np.sqrt(np.mean((T_s_mean - T_ref_arr)**2)))
    
    metrics = {
        "duration": duration,
        "rmse_power_mw": rmse_p,
        "rmse_temp_k": rmse_t
    }
    
    json_filename = filename.replace('.csv', '_metrics.json')
    json_path = os.path.join(output_dir, json_filename)
    
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print("-" * 50)
    print(f"Performance Metrics (Duration: {duration}s)")
    print(f"Power RMSE: {rmse_p:.3f} MW")
    print(f"Temp RMSE:  {rmse_t:.3f} K")
    print(f"Metrics saved to {json_path}")
    print("-" * 50)
    
    print("Test Complete. Results saved.")

def run_test(controller_type='nmpc', model_type='tcn', duration=86400):
    # Setup - aligned with generate_dataset.py
    sim_dt = 0.2
    dt_ctrl = 60.0
    horizon = 5
    T_ref = 353.15 # 80°C
    
    sim = MultiStackSimulator(dt=sim_dt)
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    stats_path = os.path.join(project_root, 'output', 'multi_stack', 'diffusion_stats.npz')
    # Try new path structure first
    model_path = os.path.join(project_root, 'output', 'multi_stack', 'policy', f'{model_type}_policy_best.pth')
    
    if controller_type == 'nmpc':
        ctrl = MultiStackNMPCController(dt=dt_ctrl, horizon=horizon, dt_sub=sim_dt)
        print("Using NMPC Controller")
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
    
    # Init
    sim.reset()
    
    # Profile
    full_profile = load_december_profile()
    
    # Use 24 hours for the test to be representative but manageable
    # (The full December data is 31 days which is too long for a single script run)
    if len(full_profile) < duration/60:
         duration = len(full_profile) * 60
    
    # Time array for simulation steps (0.2s)
    t_eval = np.arange(0, duration, sim_dt)
    
    print(f"Running test for {duration}s ({duration/3600:.1f}h) with sim_dt={sim_dt}s, dt_ctrl={dt_ctrl}s")
            
    history = {
        't': [], 'P_ref': [], 'P_real': [],
        'T_s_in': [], 'T_s_all': [], 'T_sep': [], 'T_c_out': [], 
        'n_H2_an_vec': [], 'n_liq': [], 'n_gas': [],
        'T_ref': [], 'P_ref_future': [],
        'I_prev': [], 'v_lye_prev': [], 'v_c_prev': [],
        'I_all': [], 'v_lye_all': [], 'v_c': [],
        'U_cell_all': [], 'HTO': [], 'H2_rate': []
    }
    
    last_action = [
        np.ones(4)*2000, # I
        np.ones(4)*0.03, # v_lye
        0.0 # v_c
    ]
    
    # Generate timestamped filename for data logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    data_filename = f"{controller_type}_data_{timestamp}.csv"
    output_dir = os.path.join(project_root, 'output', 'multi_stack', 'test')
    plot_every_steps = max(1, int(3600 / sim_dt))
    
    # --- Warm-up Phase ---
    # Create NMPC controller specifically for warmup (all controllers use NMPC for warmup)
    print("Creating NMPC controller for warm-up phase...")
    nmpc_ctrl = MultiStackNMPCController(dt=dt_ctrl, horizon=horizon, dt_sub=sim_dt)

    # Pass output_dir and filename_prefix to enable periodic plotting
    last_action = run_warmup_phase(
        sim, ctrl, history, last_action, sim_dt,
        output_dir=output_dir,
        filename_prefix=f"warmup_{timestamp}",
        T_ref=T_ref,
        warmup_ctrl=nmpc_ctrl  # Use NMPC for warmup
    )

    print(f"Starting Multi-Stack {controller_type.upper()} Test...")
    
    # Loop
    ctrl_steps = int(dt_ctrl / sim_dt)
    
    # Pre-calculate profile indices for efficiency
    profile_indices = (t_eval / 60).astype(int)
    profile_indices = np.clip(profile_indices, 0, len(full_profile)-1)
    
    # Calculate total control steps
    total_ctrl_steps = len(t_eval) // ctrl_steps
    
    with tqdm(total=total_ctrl_steps, desc="Testing", unit="ctrl_step") as pbar:
        for i, t in enumerate(t_eval):
            # Measure State (with noise)
            measured_state = add_measurement_noise(sim.state)

            # Extract State (from measurements)
            T_s_in = measured_state[0]
            T_s_vec = measured_state[1:5]
            T_sep = measured_state[5]
            T_c_out = measured_state[6]
            n_H2_an_vec = measured_state[7:11]
            n_liq = measured_state[11]
            n_gas = measured_state[12]
            
            # Calculate Real Power (using TRUE state for physics)
            Q, U_cell, eta = sim._calculate_electrochemical_properties(last_action[0], sim.state[1:5])
            P_real = np.sum(U_cell * last_action[0] * sim.N_cell)
            
            # Calculate Extra Metrics (using measurements)
            n_H2_sep_gas = measured_state[12]
            # T_sep is already Kelvin
            hto_pct = (n_H2_sep_gas * sim.R * T_sep) / (sim.P_sys * sim.V_sep_gas) * 100
            
            # H2 Production Rate (mol/s)
            h2_rate = np.sum(sim.N_cell * last_action[0] * eta / (2 * sim.F))
            
            # Store previous action
            prev_action = [np.copy(last_action[0]), np.copy(last_action[1]), last_action[2]]

            # Control Update
            if i % ctrl_steps == 0:
                # Get P_future from profile
                idx_min = int(t / 60)
                end_idx = idx_min + horizon
                if end_idx <= len(full_profile):
                    P_future = full_profile[idx_min:end_idx]
                else:
                    P_future = full_profile[idx_min:]
                    # Pad with last value if needed
                    if len(P_future) < horizon:
                        padding = np.full(horizon - len(P_future), full_profile[-1])
                        P_future = np.concatenate([P_future, padding])
                
                # Ensure P_future is list or array
                P_future = list[Any](P_future)
                
                I_cmd, v_lye_cmd, v_c_cmd = ctrl.get_action(
                    measured_state, P_future, T_ref, last_action
                )
                
                # Pack action for simulator: [I1..4, v1..4, vc]
                action_sim = np.concatenate([I_cmd, v_lye_cmd, [v_c_cmd]])
                last_action = [I_cmd, v_lye_cmd, v_c_cmd]
            else:
                action_sim = np.concatenate([last_action[0], last_action[1], [last_action[2]]])
                
            sim.step(action_sim)
            
            # Log and Visualize
            log_and_visualize(i, t, history, full_profile, profile_indices, P_real, T_s_in, T_s_vec, T_sep, T_c_out, n_H2_an_vec, n_liq, n_gas, T_ref, P_future, prev_action, last_action, U_cell, hto_pct, h2_rate, ctrl_steps, pbar, data_filename, output_dir, plot_every_steps)
            
    # Save final data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    output_dir = os.path.join(project_root, 'output', 'multi_stack', 'test')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    save_data_csv(history, output_dir, data_filename)
    save_plot(history, output_dir, data_filename)
    calculate_metrics(history, duration, output_dir, data_filename)

def save_plot(history, output_dir, filename):
    t_arr = np.array(history['t'])
    if len(t_arr) == 0:
        return
        
    T_s_all = np.array(history['T_s_all']) # Shape: (N, 4)
    U_cell_all = np.array(history['U_cell_all']) # Shape: (N, 4)
    I_all = np.array(history['I_all'])
    v_lye_all = np.array(history['v_lye_all'])
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    # Power
    ax = axes[0,0]
    ax.plot(t_arr, np.array(history['P_ref'])/1e6, 'k--', label='Ref')
    ax.plot(t_arr, np.array(history['P_real'])/1e6, 'b-', label='Real')
    ax.axvline(x=0, color='gray', linestyle=':', label='Start')
    ax.set_title('Power Tracking (Total)')
    ax.set_ylabel('MW')
    ax.legend()
    ax.grid(True)
    
    # System Temperatures (Combined)
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
    
    # HTO %
    ax = axes[0,2]
    ax.plot(t_arr, history['HTO'], 'm-', label='HTO')
    ax.axvline(x=0, color='gray', linestyle=':')
    ax.set_title('HTO (H2 in O2)')
    ax.set_ylabel('%')
    ax.grid(True)
    
    # Stack Currents
    ax = axes[1,0]
    for i in range(4):
        ax.plot(t_arr, I_all[:, i], label=f'Stack {i+1}')
    ax.axvline(x=0, color='gray', linestyle=':')
    ax.set_title('Stack Currents')
    ax.set_ylabel('Amps')
    ax.legend()
    ax.grid(True)
    
    # Stack Voltages
    ax = axes[1,1]
    for i in range(4):
        ax.plot(t_arr, U_cell_all[:, i], label=f'Stack {i+1}')
    ax.axvline(x=0, color='gray', linestyle=':')
    ax.set_title('Stack Voltages (Per Cell)')
    ax.set_ylabel('Volts')
    ax.legend()
    ax.grid(True)

    # Stack Lye Flow
    ax = axes[1,2]
    for i in range(4):
        ax.plot(t_arr, v_lye_all[:, i], label=f'Stack {i+1}')
    ax.axvline(x=0, color='gray', linestyle=':')
    ax.set_title('Stack Lye Flow')
    ax.set_ylabel('m3/s')
    ax.legend()
    ax.grid(True)

    # Coolant Flow
    ax = axes[2,0]
    ax.plot(t_arr, history['v_c'], 'cyan', label='CW')
    ax.axvline(x=0, color='gray', linestyle=':')
    ax.set_title('Coolant Flow')
    ax.set_ylabel('m3/s')
    ax.grid(True)
    
    # H2 Production Rate
    ax = axes[2,1]
    ax.plot(t_arr, history['H2_rate'], 'g-', label='H2 Rate')
    ax.axvline(x=0, color='gray', linestyle=':')
    ax.set_title('Total H2 Production Rate')
    ax.set_ylabel('mol/s')
    ax.grid(True)
    
    # Empty
    ax = axes[2,2]
    ax.axis('off')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, filename.replace('.csv', '.png'))
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.close(fig)

def save_data_csv(history, output_dir, filename):
    keys_scalar = ['t', 'P_ref', 'P_real', 'T_s_in', 'T_sep', 'T_c_out', 'n_liq', 'n_gas', 'T_ref', 'v_c_prev', 'v_c', 'HTO', 'H2_rate']
    
    # Flatten structure for CSV
    data_dict = {}
    
    # Scalars
    for key in keys_scalar:
        if key in history:
            data_dict[key] = history[key]
            
    # Vectors
    for key in ['T_s_all', 'n_H2_an_vec', 'I_all', 'v_lye_all', 'U_cell_all']:
        if key in history:
            vec_data = np.array(history[key])
            if vec_data.size > 0:
                for k in range(4): # 4 stacks
                    data_dict[f"{key}_{k+1}"] = vec_data[:, k]
            
    df = pd.DataFrame(data_dict)
    file_path = os.path.join(output_dir, filename)
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Multi-Stack Test')
    parser.add_argument('--controller', type=str, default='model', choices=['nmpc', 'model', 'model_dynamic'], help='Controller type')
    parser.add_argument('--model_type', type=str, default='flow_tcn', 
                        choices=['diffusion_mlp', 'diffusion_tcn', 'flow_mlp', 'flow_tcn', 'flow_mlp_hardflow', 'flow_tcn_hardflow', 'mlp', 'tcn', 'flow_matching'], 
                        help='Model type (only for model controller)')
    parser.add_argument('--duration', type=int, default=86400, help='Duration of test in seconds')
    args = parser.parse_args()
    
    run_test(controller_type=args.controller, model_type=args.model_type, duration=args.duration)
