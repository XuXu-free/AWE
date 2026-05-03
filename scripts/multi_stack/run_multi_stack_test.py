
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
from controller.multi_stack.cbf_ho_model_controller import MultiStackCBFHOModelController

def add_measurement_noise(state):
    """
    Returns the state as-is without adding noise.
    """
    return np.copy(state)

def load_power_profile(profile_path=None):
    # Get project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

    if profile_path is None:
        profile_path = os.path.join(project_root, 'output', 'power', 'wind', 'wind_power_2025-12_1min.csv')

    if not os.path.exists(profile_path):
        raise FileNotFoundError(f"Power profile not found at {profile_path}")

    print(f"Loading power profile from {profile_path}...")
    df = pd.read_csv(profile_path)
    return df['P_ref'].values

def run_warmup_phase(sim, ctrl, history, last_action, dt, output_dir, filename_prefix="warmup", T_ref=353.15, warmup_ctrl=None,
                     state_save_path=None, save_at_t=-1200.0, start_t=-14400.0):
    """
    Run a warmup phase at constant power to stabilize temperatures.
    If warmup_ctrl is provided, use it for warmup instead of ctrl.
    If state_save_path is provided, saves simulator state at save_at_t (default: -1200s = 20min before end).
    If start_t is provided (e.g. -1200.0), skip simulation before this time (used with loaded warmup state).
    """
    warmup_duration = 14400 # seconds (4h)
    warmup_steps = int(warmup_duration / dt)
    warmup_P_ref = 10.0e6 # 10MW constant

    # Use warmup_ctrl if provided, otherwise use the main ctrl
    active_ctrl = warmup_ctrl if warmup_ctrl is not None else ctrl

    # Calculate actual simulation range
    if start_t > -warmup_duration:
        actual_start = start_t
        skipped = warmup_duration + start_t
        print(f"Starting Warm-up Phase ({warmup_duration}s at {warmup_P_ref/1e6}MW) using {active_ctrl.__class__.__name__}...")
        print(f"  Skipping first {skipped:.0f}s (starting from t={actual_start:.0f}s)")
    else:
        actual_start = -warmup_duration
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

    # Compute starting index if start_t is specified
    i_start = max(0, int((start_t + warmup_duration) / dt)) if start_t > -warmup_duration else 0
    skipped_ctrl_steps = i_start // ctrl_steps
    remaining_ctrl_steps = total_ctrl_steps - skipped_ctrl_steps

    with tqdm(total=remaining_ctrl_steps, desc="Warmup", unit="ctrl_step") as pbar:
        for i in range(i_start, warmup_steps):
            t_warmup = -warmup_duration + i * dt

            # Save state at specified time (default: 20 min before warmup ends)
            if state_save_path is not None and abs(t_warmup - save_at_t) < dt * 0.5:
                state_dict = {
                    'state': sim.state,
                    'last_action_I': last_action[0],
                    'last_action_v_lye': last_action[1],
                    'last_action_v_c': last_action[2],
                    't_warmup': t_warmup,
                }
                np.savez(state_save_path, **state_dict)
                print(f"[Warmup] Saved state at t={t_warmup:.0f}s to {state_save_path}")
            
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
                cbf_info=None,
            )

    print("Warm-up Complete. Starting Main Test...")
    return last_action

def log_and_visualize(i, t, history, full_profile, profile_indices, P_real, T_s_in, T_s_vec, T_sep, T_c_out, n_H2_an_vec, n_liq, n_gas, T_ref, P_future, prev_action, last_action, U_cell, hto_pct, h2_rate, ctrl_steps, pbar, data_filename, output_dir, plot_every_steps, cbf_info=None):
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

        # Log CBF info if available
        if cbf_info is not None:
            history['cbf_h'].append(cbf_info.get('h', np.zeros(9)).copy())
            history['cbf_val'].append(cbf_info.get('cbf', np.zeros(9)).copy())
            history['cbf_slack'].append(cbf_info.get('slack', np.zeros(9)).copy())
            history['cbf_triggered'].append(cbf_info.get('projection_needed', False))
            history['cbf_adjustment'].append(cbf_info.get('adjustment', 0.0))
            history['cbf_max_slack'].append(cbf_info.get('max_slack', 0.0))
        else:
            history['cbf_h'].append(np.zeros(9))
            history['cbf_val'].append(np.zeros(9))
            history['cbf_slack'].append(np.zeros(9))
            history['cbf_triggered'].append(False)
            history['cbf_adjustment'].append(0.0)
            history['cbf_max_slack'].append(0.0)

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

def run_test(controller_type='nmpc', model_type='tcn', duration=86400, warmup_controller='nmpc', output_subdir=None, power_profile=None,
             save_warmup_state=None, load_warmup_state=None):
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

    # Output directory
    if output_subdir:
        output_dir = os.path.join(project_root, 'output', 'multi_stack', 'test', output_subdir)
    else:
        output_dir = os.path.join(project_root, 'output', 'multi_stack', 'test')
    os.makedirs(output_dir, exist_ok=True)
    # Try new path structure first
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
    elif controller_type == 'cbf_ho_model':
        ctrl = MultiStackCBFHOModelController(
            dt=dt_ctrl, horizon=horizon, model_type=model_type,
            model_path=model_path, stats_path=stats_path,
            gamma_vec=[10.0]*4 + [1.0] + [10.0]*4,
            rho_vec=[5000]*9,
            h_margin_vec=[0.0]*4 + [0.001] + [0.0]*4,
            normalize=True,
            lambda_u_scale=0.0,
            soft_mask=[True]*9,
            alpha1_vec=[10.0, 10.0, 10.0, 10.0, 1.0, 10.0, 10.0, 10.0, 10.0],
            alpha2_vec=[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            u_weight_scale=[80.0, 80.0, 80.0, 80.0, 1.0, 1.0, 1.0, 1.0, 0.1],
            # Adaptive alpha: use 1st-order CBF with smaller alpha1 for HTO below 85°C
            adaptive_alpha=True,
            T_adaptive_threshold=358.15,
            alpha1_vec_low=[10.0, 10.0, 10.0, 10.0, 0.5, 10.0, 10.0, 10.0, 10.0],
            alpha2_vec_low=[2.0, 2.0, 2.0, 2.0, 0.0, 2.0, 2.0, 2.0, 2.0],
            h_margin_vec_low=[0.0]*4 + [0.0] + [0.0]*4,
        )
        print(f"Using HOCBF Model Controller ({model_type}) with adaptive alpha")
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")
    
    # Init
    sim.reset()
    
    # Profile
    full_profile = load_power_profile(power_profile)
    
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
        'U_cell_all': [], 'HTO': [], 'H2_rate': [],
        'cbf_h': [], 'cbf_val': [], 'cbf_slack': [], 'cbf_triggered': [],
        'cbf_adjustment': [], 'cbf_max_slack': []
    }
    
    last_action = [
        np.ones(4)*2000, # I
        np.ones(4)*0.03, # v_lye
        0.0 # v_c
    ]
    
    # Generate timestamped filename for data logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if controller_type == 'model':
        data_filename = f"{controller_type}_{model_type}_data_{timestamp}.csv"
    else:
        data_filename = f"{controller_type}_data_{timestamp}.csv"
    plot_every_steps = max(1, int(3600 / sim_dt))
    
    # --- Warm-up Phase ---
    if warmup_controller == 'none':
        print("Skipping warm-up phase (--warmup_controller=none)")
    elif load_warmup_state is not None and os.path.exists(load_warmup_state):
        # Load saved warmup state to skip most of the warmup
        print(f"Loading warmup state from {load_warmup_state}...")
        state_data = np.load(load_warmup_state)
        sim.state = state_data['state']
        last_action = [
            state_data['last_action_I'],
            state_data['last_action_v_lye'],
            float(state_data['last_action_v_c'])
        ]
        loaded_t = float(state_data['t_warmup']) if 't_warmup' in state_data else -1200.0
        print(f"  Loaded state at t={loaded_t:.0f}s (skipping {14400 + loaded_t:.0f}s of warmup)")

        # Run remaining warmup (e.g., last 20 min)
        if loaded_t < 0:
            print(f"Creating {warmup_controller} controller for remaining warm-up phase...")
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
                filename_prefix=f"warmup_{timestamp}",
                T_ref=T_ref,
                warmup_ctrl=warmup_ctrl,
                state_save_path=None,
                save_at_t=loaded_t,
                start_t=loaded_t
            )
    else:
        print(f"Creating {warmup_controller} controller for warm-up phase...")
        if warmup_controller == 'nmpc_simplified':
            warmup_ctrl = MultiStackNMPCSimplifiedController(dt=dt_ctrl, horizon=horizon, dt_sub=sim_dt)
        elif warmup_controller == 'nmpc':
            warmup_ctrl = MultiStackNMPCController(dt=dt_ctrl, horizon=horizon, dt_sub=sim_dt)
        elif warmup_controller in ['model', 'model_dynamic']:
            warmup_ctrl = MultiStackModelController(dt=dt_ctrl, horizon=horizon, model_type=model_type,
                                                     model_path=model_path, stats_path=stats_path)
        else:
            raise ValueError(f"Unknown warmup controller type: {warmup_controller}")

        # Pass output_dir and filename_prefix to enable periodic plotting
        state_save_path = save_warmup_state if save_warmup_state else None
        last_action = run_warmup_phase(
            sim, ctrl, history, last_action, sim_dt,
            output_dir=output_dir,
            filename_prefix=f"warmup_{timestamp}",
            T_ref=T_ref,
            warmup_ctrl=warmup_ctrl,
            state_save_path=state_save_path,
            save_at_t=-1200.0
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

                result = ctrl.get_action(
                    measured_state, P_future, T_ref, last_action
                )
                if controller_type == 'cbf_ho_model':
                    I_cmd, v_lye_cmd, v_c_cmd, cbf_info = result
                else:
                    I_cmd, v_lye_cmd, v_c_cmd = result
                    cbf_info = None

                # Pack action for simulator: [I1..4, v1..4, vc]
                action_sim = np.concatenate([I_cmd, v_lye_cmd, [v_c_cmd]])
                last_action = [I_cmd, v_lye_cmd, v_c_cmd]
            else:
                action_sim = np.concatenate([last_action[0], last_action[1], [last_action[2]]])
                
            sim.step(action_sim)
            
            # Log and Visualize
            log_and_visualize(i, t, history, full_profile, profile_indices, P_real, T_s_in, T_s_vec, T_sep, T_c_out, n_H2_an_vec, n_liq, n_gas, T_ref, P_future, prev_action, last_action, U_cell, hto_pct, h2_rate, ctrl_steps, pbar, data_filename, output_dir, plot_every_steps, cbf_info)
            
    # Save final data
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

    has_cbf = len(history.get('cbf_h', [])) > 0 and np.any(np.array(history['cbf_triggered']))

    if has_cbf:
        fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    else:
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    # Power
    ax = axes[0,0]
    ax.plot(t_arr/3600, np.array(history['P_ref'])/1e6, 'k--', label='Ref')
    ax.plot(t_arr/3600, np.array(history['P_real'])/1e6, 'b-', label='Real')
    ax.axvline(x=0, color='gray', linestyle=':', label='Start')
    ax.set_title('Power Tracking (Total)')
    ax.set_ylabel('MW')
    ax.set_xlabel('Time (h)')
    ax.legend()
    ax.grid(True)

    # System Temperatures (Combined)
    ax = axes[0,1]
    ax.plot(t_arr/3600, history['T_sep'], 'g--', label='Separator')
    ax.plot(t_arr/3600, history['T_c_out'], 'c:', label='CW Out')
    for i in range(4):
        ax.plot(t_arr/3600, T_s_all[:, i], label=f'Stack {i+1}')
    ax.plot(t_arr/3600, history['T_ref'], 'k--', linewidth=1.5, label='Ref Temp')
    ax.axvline(x=0, color='gray', linestyle=':')
    ax.axhline(y=363.15, color='r', linestyle='--', alpha=0.5, label='Tmax')
    ax.axhline(y=293.15, color='b', linestyle='--', alpha=0.5, label='Tmin')
    ax.set_title('All Temperatures')
    ax.set_ylabel('K')
    ax.set_xlabel('Time (h)')
    ax.legend()
    ax.grid(True)

    # HTO %
    ax = axes[0,2]
    ax.plot(t_arr/3600, history['HTO'], 'm-', label='HTO')
    ax.axhline(y=2.0, color='r', linestyle='--', alpha=0.5, label='Limit (2%)')
    ax.axvline(x=0, color='gray', linestyle=':')
    ax.set_title('HTO (H2 in O2)')
    ax.set_ylabel('%')
    ax.set_xlabel('Time (h)')
    ax.legend()
    ax.grid(True)

    # Stack Currents
    ax = axes[1,0]
    for i in range(4):
        ax.plot(t_arr/3600, I_all[:, i], label=f'Stack {i+1}')
    ax.axvline(x=0, color='gray', linestyle=':')
    ax.set_title('Stack Currents')
    ax.set_ylabel('Amps')
    ax.set_xlabel('Time (h)')
    ax.legend()
    ax.grid(True)

    # Stack Voltages
    ax = axes[1,1]
    for i in range(4):
        ax.plot(t_arr/3600, U_cell_all[:, i], label=f'Stack {i+1}')
    ax.axvline(x=0, color='gray', linestyle=':')
    ax.set_title('Stack Voltages (Per Cell)')
    ax.set_ylabel('Volts')
    ax.set_xlabel('Time (h)')
    ax.legend()
    ax.grid(True)

    # Stack Lye Flow
    ax = axes[1,2]
    for i in range(4):
        ax.plot(t_arr/3600, v_lye_all[:, i], label=f'Stack {i+1}')
    ax.axvline(x=0, color='gray', linestyle=':')
    ax.set_title('Stack Lye Flow')
    ax.set_ylabel('m3/s')
    ax.set_xlabel('Time (h)')
    ax.legend()
    ax.grid(True)

    # Coolant Flow
    ax = axes[2,0]
    ax.plot(t_arr/3600, history['v_c'], 'cyan', label='CW')
    ax.axvline(x=0, color='gray', linestyle=':')
    ax.set_title('Coolant Flow')
    ax.set_ylabel('m3/s')
    ax.set_xlabel('Time (h)')
    ax.legend()
    ax.grid(True)

    # H2 Production Rate
    ax = axes[2,1]
    ax.plot(t_arr/3600, history['H2_rate'], 'g-', label='H2 Rate')
    ax.axvline(x=0, color='gray', linestyle=':')
    ax.set_title('Total H2 Production Rate')
    ax.set_ylabel('mol/s')
    ax.set_xlabel('Time (h)')
    ax.legend()
    ax.grid(True)

    if has_cbf:
        cbf_h = np.array(history['cbf_h'])
        cbf_val = np.array(history['cbf_val'])
        cbf_slack = np.array(history['cbf_slack'])
        cbf_triggered = np.array(history['cbf_triggered'])

        # CBF h values (safety margins)
        ax = axes[2,2]
        for i in range(4):
            ax.plot(t_arr/3600, cbf_h[:, i], label=f'Tmax S{i+1}')
        ax.plot(t_arr/3600, cbf_h[:, 4], 'm-', linewidth=2, label='HTO')
        for i in range(4):
            ax.plot(t_arr/3600, cbf_h[:, 5+i], '--', alpha=0.7, label=f'Tmin S{i+1}')
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.set_title('CBF h(x) - Safety Margins')
        ax.set_ylabel('h')
        ax.set_xlabel('Time (h)')
        ax.legend(fontsize=7)
        ax.grid(True)

        # CBF values (constraint satisfaction)
        ax = axes[3,0]
        for i in range(4):
            ax.plot(t_arr/3600, cbf_val[:, i], label=f'Tmax S{i+1}')
        ax.plot(t_arr/3600, cbf_val[:, 4], 'm-', linewidth=2, label='HTO')
        for i in range(4):
            ax.plot(t_arr/3600, cbf_val[:, 5+i], '--', alpha=0.7, label=f'Tmin S{i+1}')
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.set_title('CBF Values (>=0 Safe)')
        ax.set_ylabel('CBF')
        ax.set_xlabel('Time (h)')
        ax.legend(fontsize=7)
        ax.grid(True)

        # Slack values
        ax = axes[3,1]
        for i in range(4):
            ax.plot(t_arr/3600, cbf_slack[:, i], label=f'Tmax S{i+1}')
        ax.plot(t_arr/3600, cbf_slack[:, 4], 'm-', linewidth=2, label='HTO')
        for i in range(4):
            ax.plot(t_arr/3600, cbf_slack[:, 5+i], '--', alpha=0.7, label=f'Tmin S{i+1}')
        ax.set_title('CBF Slack Values')
        ax.set_ylabel('Slack')
        ax.set_xlabel('Time (h)')
        ax.legend(fontsize=7)
        ax.grid(True)

        # Projection triggered
        ax = axes[3,2]
        trigger_times = t_arr[cbf_triggered] / 3600
        if len(trigger_times) > 0:
            ax.scatter(trigger_times, np.ones(len(trigger_times)), c='red', s=10, label='Triggered', zorder=5)
        ax.fill_between(t_arr/3600, 0, cbf_triggered.astype(float), alpha=0.3, color='red', label='Projection Active')
        ax.set_ylim(-0.1, 1.5)
        ax.set_title('CBF Projection Triggered')
        ax.set_ylabel('Triggered')
        ax.set_xlabel('Time (h)')
        ax.legend()
        ax.grid(True)
    else:
        # Empty
        ax = axes[2,2]
        ax.axis('off')
        if len(axes.shape) > 1 and axes.shape[0] > 3:
            for r in range(3, axes.shape[0]):
                for c in range(axes.shape[1]):
                    axes[r, c].axis('off')

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

    # CBF data
    if len(history.get('cbf_h', [])) > 0:
        cbf_h = np.array(history['cbf_h'])
        cbf_val = np.array(history['cbf_val'])
        cbf_slack = np.array(history['cbf_slack'])
        for k in range(9):
            data_dict[f"cbf_h_{k}"] = cbf_h[:, k]
            data_dict[f"cbf_val_{k}"] = cbf_val[:, k]
            data_dict[f"cbf_slack_{k}"] = cbf_slack[:, k]
        data_dict['cbf_triggered'] = history['cbf_triggered']
        data_dict['cbf_adjustment'] = history['cbf_adjustment']
        data_dict['cbf_max_slack'] = history['cbf_max_slack']

    df = pd.DataFrame(data_dict)
    file_path = os.path.join(output_dir, filename)
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Multi-Stack Test')
    parser.add_argument('--controller', type=str, default='model', choices=['nmpc', 'nmpc_simplified', 'model', 'model_dynamic', 'cbf_ho_model'], help='Controller type')
    parser.add_argument('--model_type', type=str, default='flow_tcn',
                        choices=['diffusion_mlp', 'diffusion_pure_mlp', 'diffusion_tcn', 'diffusion_tcn_l3', 'guided_diffusion_mlp', 'guided_diffusion_tcn', 'deterministic_diffusion_mlp', 'deterministic_diffusion_pure_mlp', 'deterministic_diffusion_tcn', 'deterministic_guided_diffusion_mlp', 'deterministic_guided_diffusion_tcn', 'early_stop_diffusion_mlp', 'early_stop_diffusion_tcn', 'early_stop_guided_diffusion_mlp', 'early_stop_guided_diffusion_tcn', 'flow_mlp', 'flow_tcn', 'flow_mlp_hardflow', 'flow_tcn_hardflow', 'mlp', 'tcn', 'flow_matching', 'pure_mlp', 'pure_tcn', 'lstm'],
                        help='Model type (only for model controller)')
    parser.add_argument('--duration', type=int, default=86400, help='Duration of test in seconds')
    parser.add_argument('--warmup_controller', type=str, default='nmpc_simplified',
                        choices=['none', 'nmpc', 'nmpc_simplified', 'model'],
                        help='Controller type for warm-up phase (default: nmpc_simplified)')
    parser.add_argument('--output_subdir', type=str, default=None,
                        help='Subdirectory under output/multi_stack/test/ to store results')
    parser.add_argument('--power_profile', type=str, default=None,
                        help='Path to power profile CSV (default: December 2025 profile)')
    parser.add_argument('--save_warmup_state', type=str, default=None,
                        help='Path to save warmup state (e.g., output/multi_stack/test/warmup_state.npz)')
    parser.add_argument('--load_warmup_state', type=str, default=None,
                        help='Path to load saved warmup state to skip most of warmup')
    args = parser.parse_args()

    run_test(controller_type=args.controller, model_type=args.model_type, duration=args.duration,
             warmup_controller=args.warmup_controller, output_subdir=args.output_subdir,
             power_profile=args.power_profile,
             save_warmup_state=args.save_warmup_state,
             load_warmup_state=args.load_warmup_state)
