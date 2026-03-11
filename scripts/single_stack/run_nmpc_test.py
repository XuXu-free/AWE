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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from plant.single_stack_simulator import SingleStackSimulator

from controller.single_stack.model_controller import SingleStackModelController
from controller.single_stack.nmpc_controller import SingleStackNMPCController

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
    # Scale down for single stack (0.25 of total)
    return df['P_ref'].values * 0.25

def run_warmup_phase(sim, ctrl, history, last_action, sim_dt, output_dir, filename_prefix="warmup", T_ref=358.15):
    """ 
    Run a warmup phase at constant power to stabilize temperatures.
    """
    warmup_duration = 4 * 60 * 60 # 4h
    warmup_steps = int(warmup_duration / sim_dt)
    warmup_P_ref = 2.0e6 # 2MW constant
    
    print(f"Starting Warm-up Phase ({warmup_duration}s at {warmup_P_ref/1e6}MW)...")
    
    # Pre-calculate future profile for warmup (constant)
    # SingleStackNMPCController uses N_p instead of horizon
    horizon = getattr(ctrl, 'N_p', getattr(ctrl, 'horizon', 5))
    P_future = [warmup_P_ref] * horizon
    
    # Control interval steps
    ctrl_steps = int(ctrl.dt / sim_dt)
    total_ctrl_steps = warmup_steps // ctrl_steps
    warmup_profile = np.array([warmup_P_ref], dtype=float)
    warmup_profile_indices = np.zeros(warmup_steps, dtype=int)
    plot_every_steps = max(1, int(1000 / sim_dt)) # Plot occasionally
    warmup_data_filename = f"{filename_prefix}.csv"
    
    with tqdm(total=total_ctrl_steps, desc="Warmup", unit="ctrl_step") as pbar:
        for i in range(warmup_steps):
            t_warmup = -warmup_duration + i * sim_dt
            
            # Measure State (with noise)
            measured_state = add_measurement_noise(sim.state)
            
            # Extract State
            T_s_in = measured_state[0]
            T_s = measured_state[1]
            T_sep = measured_state[2]
            T_c_out = measured_state[3]
            n_H2_an = measured_state[4]
            n_liq = measured_state[5]
            n_gas = measured_state[6]
            
            # Calculate Real Power
            Q, U_cell, eta = sim._calculate_electrochemical_properties(last_action[0], sim.state[1])
            P_real = U_cell * last_action[0] * sim.N_cell
            
            # Calculate Extra Metrics
            n_H2_sep_gas = measured_state[6]
            # T_sep is already Kelvin
            hto_pct = (n_H2_sep_gas * sim.R * T_sep) / (sim.P_sys * sim.V_sep_gas) * 100 
            
            h2_rate = sim.N_cell * last_action[0] * eta / (2 * sim.F)
            
            # Store previous action before update
            prev_action = list(last_action)

            # Control Update
            if i % ctrl_steps == 0:
                # Controller expects Kelvin T_ref
                I_cmd, v_lye_cmd, v_c_cmd = ctrl.get_action(
                    measured_state, P_future, T_ref=T_ref, last_action=last_action
                )
                action_sim = np.array([I_cmd, v_lye_cmd, v_c_cmd])
                last_action = [I_cmd, v_lye_cmd, v_c_cmd]
                
                # Update progress bar
                pbar.set_postfix({
                    "t": f"{t_warmup:.0f}s",
                    "P_ref": f"{warmup_P_ref/1e6:.1f}MW",
                    "P_real": f"{P_real/1e6:.1f}MW",
                    "T_s": f"{T_s-273.15:.1f}C"
                })
                pbar.update(1)
            else:
                action_sim = np.array(last_action)
                
            sim.step(action_sim)
            
            log_and_visualize(
                i=i,
                t=t_warmup,
                history=history,
                full_profile=warmup_profile,
                profile_indices=warmup_profile_indices,
                P_real=P_real,
                T_s_in=T_s_in,
                T_s=T_s,
                T_sep=T_sep,
                T_c_out=T_c_out,
                n_H2_an=n_H2_an,
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
                data_filename=warmup_data_filename,
                output_dir=output_dir,
                plot_every_steps=plot_every_steps,
            )

    print("Warm-up Complete. Starting Main Test...")
    return last_action

def log_and_visualize(i, t, history, full_profile, profile_indices, P_real, T_s_in, T_s, T_sep, T_c_out, n_H2_an, n_liq, n_gas, T_ref, P_future, prev_action, last_action, U_cell, hto_pct, h2_rate, ctrl_steps, data_filename, output_dir, plot_every_steps):
    # Log (downsampled) - Every 10s (assume dt=1.0s, so every 10 steps)
    if i % 10 == 0:
        history['t'].append(t)
        # Handle profile indexing carefully
        idx = profile_indices[i] if i < len(profile_indices) else profile_indices[-1]
        history['P_ref'].append(full_profile[idx])
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
    
    # Periodic Plot Update
    if plot_every_steps is not None and i > 0 and i % plot_every_steps == 0:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_data_csv(history, output_dir, data_filename)
        save_plot(history, output_dir, data_filename.replace('.csv', '_progress.png'))

def calculate_metrics(history, duration, output_dir, filename):
    # Calculate RMSE
    t_arr = np.array(history['t'])
    if len(t_arr) == 0:
        return

    P_ref_arr = np.array(history['P_ref'])
    P_real_arr = np.array(history['P_real'])
    T_s_all = np.array(history['T_s_all'])
    T_ref_arr = np.array(history['T_ref'])
    
    # Metrics
    rmse_p = float(np.sqrt(np.mean((P_real_arr - P_ref_arr)**2)) / 1e6)
    rmse_t = float(np.sqrt(np.mean((T_s_all - T_ref_arr)**2)))
    
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

def run_test(controller_type='nmpc', model_type='tcn'):
    # Setup
    sim_dt = 0.2
    dt_ctrl = 60.0
    horizon = 5 # N_p
    T_ref = 353.15 # 80°C
    
    sim = SingleStackSimulator(sim_dt=sim_dt)
    
    if controller_type == 'nmpc':
        ctrl = SingleStackNMPCController(dt=dt_ctrl, horizon=horizon, sim_dt=sim_dt) # Use dt_sub default or explicit
        print("Using NMPC Controller")
    elif controller_type == 'model':
        ctrl = SingleStackModelController(model_type=model_type, dt=dt_ctrl)
        print(f"Using Model Controller ({model_type})")
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")
    
    # Init
    sim.reset()
    
    # Profile
    full_profile = load_december_profile()
    
    duration = 86400
        
    t_eval = np.arange(0, duration, sim_dt)
    
    print(f"Running test for {duration}s ({duration/3600:.1f}h) with sim_dt={sim_dt}s, dt_ctrl={dt_ctrl}s")
            
    history = {
        't': [], 'P_ref': [], 'P_real': [],
        'T_s_in': [], 'T_s_all': [], 'T_sep': [], 'T_c_out': [], 
        'n_H2_an': [], 'n_liq': [], 'n_gas': [],
        'T_ref': [], 'P_ref_future': [],
        'I_prev': [], 'v_lye_prev': [], 'v_c_prev': [],
        'I': [], 'v_lye': [], 'v_c': [],
        'U_cell_all': [], 'HTO': [], 'H2_rate': []
    }
    
    last_action = [2000, 0.3, 0.0]
    
    # Generate timestamped filename for data logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    data_filename = f"{controller_type}_data_{timestamp}.csv"
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    output_dir = os.path.join(project_root, 'output', 'single_stack', 'test')
    plot_every_steps = max(1, int(2000 / sim_dt))
    
    # --- Warm-up Phase ---
    last_action = run_warmup_phase(
        sim, ctrl, history, last_action, sim_dt, 
        output_dir=output_dir, 
        filename_prefix=f"warmup_{timestamp}", 
        T_ref=T_ref
    )

    print(f"Starting Single Stack {controller_type.upper()} Test...")
    
    # Loop
    ctrl_steps = int(dt_ctrl / sim_dt)
    
    # Pre-calculate profile indices for efficiency
    # Map each simulation step (sim_dt) to profile index (1min)
    # t = i * sim_dt
    # index = int(t / 60)
    profile_indices = (t_eval / 60).astype(int)
    profile_indices = np.clip(profile_indices, 0, len(full_profile)-1)
    
    # Calculate total control steps
    total_ctrl_steps = len(t_eval) // ctrl_steps
    
    with tqdm(total=total_ctrl_steps, desc="Testing", unit="ctrl_step") as pbar:
        for i, t in enumerate(t_eval):
            # Measure State (with noise)
            measured_state = add_measurement_noise(sim.state)

            # Extract State
            T_s_in = measured_state[0]
            T_s = measured_state[1]
            T_sep = measured_state[2]
            T_c_out = measured_state[3]
            n_H2_an = measured_state[4]
            n_liq = measured_state[5]
            n_gas = measured_state[6]
            
            # Calculate Real Power
            Q, U_cell, eta = sim._calculate_electrochemical_properties(last_action[0], sim.state[1])
            P_real = U_cell * last_action[0] * sim.N_cell
            
            # Calculate Extra Metrics
            n_H2_sep_gas = measured_state[6]
            hto_pct = (n_H2_sep_gas * sim.R * T_sep) / (sim.P_sys * sim.V_sep_gas) * 100 
        
            h2_rate = sim.N_cell * last_action[0] * eta / (2 * sim.F)
            
            # Store previous action
            prev_action = list(last_action)

            # Control Update
            if i % ctrl_steps == 0:
                # Get P_future from profile
                # We need next N_p steps at dt_ctrl intervals
                P_future = []
                for k in range(horizon):
                    t_future = t + k * dt_ctrl
                    idx_future = profile_indices[i + k]
                    if idx_future < len(full_profile):
                        P_future.append(full_profile[idx_future])
                    else:
                        P_future.append(full_profile[-1])
                
                I_cmd, v_lye_cmd, v_c_cmd = ctrl.get_action(
                    measured_state, P_future, T_ref, last_action
                )
                
                action_sim = np.array([I_cmd, v_lye_cmd, v_c_cmd])
                last_action = [I_cmd, v_lye_cmd, v_c_cmd]
                
                # Update progress bar
                idx = profile_indices[i] if i < len(profile_indices) else profile_indices[-1]
                P_ref_val = full_profile[idx]
                pbar.set_postfix({
                    "t": f"{t:.0f}s",
                    "P_ref": f"{P_ref_val/1e6:.1f}MW",
                    "P_real": f"{P_real/1e6:.1f}MW",
                    "T_s": f"{T_s-273.15:.1f}C"
                })
                pbar.update(1)
            else:
                action_sim = np.array(last_action)
                
            sim.step(action_sim)
            
            # Log and Visualize
            log_and_visualize(i, t, history, full_profile, profile_indices, P_real, T_s_in, T_s, T_sep, T_c_out, n_H2_an, n_liq, n_gas, T_ref, P_future, prev_action, last_action, U_cell, hto_pct, h2_rate, ctrl_steps, data_filename, output_dir, plot_every_steps)
            
    # Save final data
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    save_data_csv(history, output_dir, data_filename)
    save_plot(history, output_dir, data_filename.replace('.csv', '.png'))
    calculate_metrics(history, duration, output_dir, data_filename)
    
    print("Test Complete. Results saved.")

def save_plot(history, output_dir, filename):
    t_arr = np.array(history['t'])
    if len(t_arr) == 0:
        return
        
    T_s_all = np.array(history['T_s_all']) # Shape: (N,)
    U_cell_all = np.array(history['U_cell_all']) # Shape: (N,)
    I_all = np.array(history['I']) # Shape: (N,)
    v_lye_all = np.array(history['v_lye']) # Shape: (N,)
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    # Power
    ax = axes[0,0]
    ax.plot(t_arr, np.array(history['P_ref'])/1e6, 'k--', label='Ref')
    ax.plot(t_arr, np.array(history['P_real'])/1e6, 'b-', label='Real')
    ax.axvline(x=0, color='gray', linestyle=':', label='Start')
    ax.set_title('Power Tracking')
    ax.set_ylabel('MW')
    ax.legend()
    ax.grid(True)
    
    # All Temperatures
    ax = axes[0,1]
    ax.plot(t_arr, history['T_sep'], 'g--', label='Separator')
    ax.plot(t_arr, history['T_c_out'], 'c:', label='CW Out')
    ax.plot(t_arr, T_s_all, 'r-', label='Stack')
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
    
    # Stack Current
    ax = axes[1,0]
    ax.plot(t_arr, I_all, 'b-', label='Stack')
    ax.axvline(x=0, color='gray', linestyle=':')
    ax.set_title('Stack Current')
    ax.set_ylabel('Amps')
    ax.legend()
    ax.grid(True)
    
    # Stack Voltage
    ax = axes[1,1]
    ax.plot(t_arr, U_cell_all, 'b-', label='Stack')
    ax.axvline(x=0, color='gray', linestyle=':')
    ax.set_title('Cell Voltage')
    ax.set_ylabel('Volts')
    ax.legend()
    ax.grid(True)
    
    # Stack Lye Flow
    ax = axes[1,2]
    ax.plot(t_arr, v_lye_all, 'orange', label='Stack')
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
    ax.set_title('H2 Production Rate')
    ax.set_ylabel('mol/s')
    ax.grid(True)
    
    # Empty
    ax = axes[2,2]
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close(fig)

def save_data_csv(history, output_dir, filename):
    # Keys for scalar columns
    keys_scalar = ['t', 'P_ref', 'P_real', 'T_s_in', 'T_s_all', 'T_sep', 'T_c_out', 'n_H2_an', 'n_liq', 'n_gas', 'T_ref', 'I_prev', 'v_lye_prev', 'v_c_prev', 'I', 'v_lye', 'v_c', 'HTO', 'H2_rate', 'U_cell_all']
    
    data_dict = {}
    for key in keys_scalar:
        if key in history and len(history[key]) > 0:
            data_dict[key] = history[key]
            
    # Save P_ref_future for model validation (Model Input)
    if 'P_ref_future' in history:
        P_future_arr = np.array(history['P_ref_future'])
        if P_future_arr.size > 0 and len(P_future_arr.shape) > 1:
            horizon = P_future_arr.shape[1]
            for k in range(horizon):
                data_dict[f'P_ref_future_{k}'] = P_future_arr[:, k]
    # keys_scalar = ['t', 'P_ref', 'P_real', 'T_s_in', 'T_sep', 'T_c_out', 'n_liq', 'n_gas', 'T_ref', 'v_c_prev', 'v_c', 'HTO', 'H2_rate']
    # It didn't save future.
    # Single stack previously saved future. I will omit it for now to keep it clean and match multi-stack unless user complains.
    
    df = pd.DataFrame(data_dict)
    file_path = os.path.join(output_dir, filename)
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Single-Stack Test')
    parser.add_argument('--controller', type=str, default='nmpc', choices=['nmpc', 'model'], help='Controller type')
    parser.add_argument('--model_type', type=str, default='diffusion_tcn', choices=['diffusion_tcn', 'diffusion_mlp', 'flow_tcn', 'flow_mlp'], help='Model type for model-based controller')

    args = parser.parse_args()
    
    run_test(
        controller_type=args.controller, 
        model_type=args.model_type,
    )
