
import os
import sys
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
from controller.multi_stack.diffusion_controller import MultiStackDiffusionController
from controller.multi_stack.diffusion_dynamic_controller import MultiStackDiffusionDynamicController

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

def run_warmup_phase(sim, ctrl, history, last_action, dt, T_ref=353.15):
    """
    Run a warmup phase at constant power to stabilize temperatures.
    """
    warmup_duration = 1200 # seconds (20 mins)
    warmup_steps = int(warmup_duration / dt)
    warmup_P_ref = 8.0e6 # 8MW constant
    
    print(f"Starting Warm-up Phase ({warmup_duration}s at {warmup_P_ref/1e6}MW)...")
    
    # Pre-calculate future profile for warmup (constant)
    P_future = [warmup_P_ref] * ctrl.horizon
    
    # Control interval steps
    ctrl_steps = int(ctrl.dt / dt)
    
    with tqdm(total=warmup_steps, desc="Warmup", unit="step") as pbar:
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
            n_gas_total = (sim.p_sys * sim.V_sep_gas) / (sim.R * T_sep)
            hto_pct = (n_H2_sep_gas / n_gas_total) * 100 if n_gas_total > 1e-9 else 0.0
            
            F_const = 96485.0
            h2_rate = np.sum(sim.N_cell * last_action[0] * eta / (2 * F_const))
            
            # Store previous action before update
            prev_action = [np.copy(last_action[0]), np.copy(last_action[1]), last_action[2]]

            # Control Update
            if i % ctrl_steps == 0:
                I_cmd, v_lye_cmd, v_c_cmd = ctrl.get_action(
                    measured_state, P_future, T_ref=T_ref
                )
                action_sim = np.concatenate([I_cmd, v_lye_cmd, [v_c_cmd]])
                last_action = [I_cmd, v_lye_cmd, v_c_cmd]
            else:
                action_sim = np.concatenate([last_action[0], last_action[1], [last_action[2]]])
                
            sim.step(action_sim)
            
            # Log Warmup (downsample logging to avoid huge files? or keep all for now)
            # Log every 1s (approx 5 steps) to save space if dt=0.2
            if i % 5 == 0:
                history['t'].append(t_warmup)
                history['P_ref'].append(warmup_P_ref)
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
            
            if i % 100 == 0:
                pbar.set_postfix({
                    "T_s_mean": f"{np.mean(T_s_vec):.1f}K"
                })
            pbar.update(1)

    print("Warm-up Complete. Starting Main Test...")
    return last_action

def run_test(controller_type='nmpc', model_type='tcn'):
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
    model_path = os.path.join(project_root, 'output', 'multi_stack', 'model', f'best_diffusion_policy_model_{model_type}.pth')
    
    if controller_type == 'nmpc':
        ctrl = MultiStackNMPCController(dt=dt_ctrl, horizon=horizon, dt_sub=sim_dt)
        print("Using NMPC Controller")
    elif controller_type == 'diffusion':
        ctrl = MultiStackDiffusionController(dt=dt_ctrl, horizon=horizon, model_type=model_type, 
                                             model_path=model_path, stats_path=stats_path)
        print(f"Using Diffusion Controller ({model_type})")
    elif controller_type == 'diffusion_dynamic':
        ctrl = MultiStackDiffusionDynamicController(dt=dt_ctrl, horizon=horizon, model_type=model_type, 
                                             model_path=model_path, stats_path=stats_path)
        print(f"Using Diffusion Dynamic Controller ({model_type})")
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")
    
    # Init
    sim.reset()
    
    # Profile
    full_profile = load_december_profile()
    
    # Use 24 hours for the test to be representative but manageable
    # (The full December data is 31 days which is too long for a single script run)
    duration = 86400 # 24 hours
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
        np.zeros(4), # I
        np.ones(4)*0.03, # v_lye
        0.0 # v_c
    ]
    
    # --- Warm-up Phase ---
    last_action = run_warmup_phase(sim, ctrl, history, last_action, sim_dt, T_ref=T_ref)

    print(f"Starting Multi-Stack {controller_type.upper()} Test...")
    
    # Generate timestamped filename for data logging
    data_filename = f"{controller_type}_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    # Loop
    ctrl_steps = int(dt_ctrl / sim_dt)
    
    # Pre-calculate profile indices for efficiency
    profile_indices = (t_eval / 60).astype(int)
    profile_indices = np.clip(profile_indices, 0, len(full_profile)-1)
    
    with tqdm(total=len(t_eval), desc="Testing", unit="step") as pbar:
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
            n_gas_total = (sim.p_sys * sim.V_sep_gas) / (sim.R * T_sep)
            hto_pct = (n_H2_sep_gas / n_gas_total) * 100 if n_gas_total > 1e-9 else 0.0
            
            # H2 Production Rate (mol/s)
            F_const = 96485.0
            h2_rate = np.sum(sim.N_cell * last_action[0] * eta / (2 * F_const))
            
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
                P_future = list(P_future)
                
                I_cmd, v_lye_cmd, v_c_cmd = ctrl.get_action(
                    measured_state, P_future, T_ref=T_ref
                )
                
                # Pack action for simulator: [I1..4, v1..4, vc]
                action_sim = np.concatenate([I_cmd, v_lye_cmd, [v_c_cmd]])
                last_action = [I_cmd, v_lye_cmd, v_c_cmd]
            else:
                action_sim = np.concatenate([last_action[0], last_action[1], [last_action[2]]])
                
            sim.step(action_sim)
            
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
            
            # Update progress bar
            if i % 100 == 0:
                P_ref_val = full_profile[profile_indices[i]]
                pbar.set_postfix({
                    "t": f"{t:.0f}s",
                    "P_ref": f"{P_ref_val/1e6:.1f}MW",
                    "P_real": f"{P_real/1e6:.1f}MW",
                    "T_s": f"{np.mean(T_s_vec)-273.15:.1f}C"
                })
            pbar.update(1)
            
            # Periodic Plot Update (every 3600s)
            if t > 0 and i % (3600 * 5) == 0: # Every hour
                # Use project root relative path
                script_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
                output_dir = os.path.join(project_root, 'output', 'multi_stack', 'test')
                
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                # Save CSV snapshot
                save_data_csv(history, output_dir, data_filename)
                save_plot(history, output_dir, data_filename)
            
    # Save final data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    output_dir = os.path.join(project_root, 'output', 'multi_stack', 'test')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    save_data_csv(history, output_dir, data_filename)
    save_plot(history, output_dir, data_filename)
    
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
    rmse_p = np.sqrt(np.mean((P_real_arr - P_ref_arr)**2)) / 1e6
    rmse_t = np.sqrt(np.mean((T_s_mean - T_ref_arr)**2))
    
    print("-" * 50)
    print(f"Performance Metrics (Duration: {duration}s)")
    print(f"Power RMSE: {rmse_p:.3f} MW")
    print(f"Temp RMSE:  {rmse_t:.3f} K")
    print("-" * 50)
    
    print("Test Complete. Results saved.")

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
    parser.add_argument('--controller', type=str, default='diffusion', choices=['nmpc', 'diffusion', 'diffusion_dynamic'], help='Controller type')
    parser.add_argument('--model_type', type=str, default='tcn', choices=['mlp', 'tcn', 'flow_matching'], help='Diffusion model type (only for diffusion controller)')
    args = parser.parse_args()
    
    run_test(controller_type=args.controller, model_type=args.model_type)
