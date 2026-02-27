
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import argparse

import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from plant.multi_stack_simulator import MultiStackSimulator
from controller.multi_stack.multi_stack_nmpc_controller import MultiStackNMPCController
from controller.multi_stack.multi_stack_diffusion_controller import MultiStackDiffusionController

# Noise Constants
SIGMA_T = 0.5      # Temperature noise (K)
SIGMA_N_H2 = 0.1   # H2 moles noise (mol)
SIGMA_N_SEP = 0.5  # Separator level/gas noise (mol)

def add_measurement_noise(state):
    """
    Returns the state as-is without adding noise.
    """
    return np.copy(state)

def generate_power_profile_values(t_eval):
    for t in t_eval:
        if t < 3600:
            # Normal / Fluctuating Phase (0 - 1h)
            # Base fluctuating signal: Sum of sines to mimic wind/solar variability
            # Range approx 3MW to 22MW
            # Low freq + Med freq components
            val = 12.0 + 6.0 * np.sin(2 * np.pi * t / 11000) + \
                  4.0 * np.sin(2 * np.pi * t / 3700) + \
                  2.0 * np.sin(2 * np.pi * t / 1300)
            
            # Add some noise/roughness
            val += 1.0 * np.sin(2 * np.pi * t / 300)
            
            # Clip to match visual range [3, 22]
            val = np.clip(val, 3.0, 22.0)
            yield val * 1e6
            
        else:
            # Overload Phase (1h - 2h)
            # Reference goes high (~26-33 MW), System limited to ~25MW
            # Plateau with fluctuations
            # Adjust time base for continuity in sine waves or just shift
            t_shift = t - 3600
            val = 28.0 + 3.0 * np.sin(2 * np.pi * t_shift / 2500) + \
                  1.5 * np.sin(2 * np.pi * t / 800)
            
            # Clip [25, 33]
            val = np.clip(val, 25.0, 33.0)
            yield val * 1e6

def run_warmup_phase(sim, ctrl, history, last_action, dt):
    """
    Run a warmup phase at constant power to stabilize temperatures.
    """
    warmup_duration = 1000 # seconds
    warmup_steps = int(warmup_duration / dt)
    warmup_P_ref = 8.0e6 # 8MW constant
    
    print(f"Starting Warm-up Phase ({warmup_duration}s at {warmup_P_ref/1e6}MW)...")
    
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

        # Control
        # Create P_future for warmup (constant) - needed for logging every step
        P_future = [warmup_P_ref] * ctrl.horizon if hasattr(ctrl, 'horizon') else [warmup_P_ref] * ctrl.N

        if i % 10 == 0: # 10s control loop
            I_cmd, v_lye_cmd, v_c_cmd = ctrl.get_action(
                measured_state, P_future, T_ref=358.15
            )
            action_sim = np.concatenate([I_cmd, v_lye_cmd, [v_c_cmd]])
            last_action = [I_cmd, v_lye_cmd, v_c_cmd]
        else:
            action_sim = np.concatenate([last_action[0], last_action[1], [last_action[2]]])
            
        sim.step(action_sim)
        
        # Log Warmup
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
        history['T_ref'].append(358.15) # Log in Kelvin
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
            print(f"Warmup: {(i*dt)/warmup_duration*100:.0f}% | T_s_mean={np.mean(T_s_vec):.1f}K")

    print("Warm-up Complete. Starting Main Test...")
    return last_action

def run_test(controller_type='nmpc', model_type='tcn'):
    # Setup
    dt = 1.0
    sim = MultiStackSimulator(dt=dt)
    
    if controller_type == 'nmpc':
        ctrl = MultiStackNMPCController(dt=10.0, N_p=10)
        print("Using NMPC Controller")
    elif controller_type == 'diffusion':
        ctrl = MultiStackDiffusionController(dt=10.0, horizon=10, model_type=model_type)
        print(f"Using Diffusion Controller ({model_type})")
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")
    
    # Init
    sim.reset()
    
    # Profile
    duration = 7200 # 2 hours
    t_eval = np.arange(0, duration, dt)
    P_ref_profile = np.array(list(generate_power_profile_values(t_eval)))
            
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
    last_action = run_warmup_phase(sim, ctrl, history, last_action, dt)

    print(f"Starting Multi-Stack {controller_type.upper()} Test...")
    
    # Generate timestamped filename for data logging
    data_filename = f"multi_stack_{controller_type}_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    # Loop
    # Initialize P_future_log
    P_future_log = []

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

        # Control
        # Always compute P_future for logging
        P_future = []
        horizon = ctrl.N_p if hasattr(ctrl, 'N') else ctrl.horizon
        for k in range(horizon):
            t_future = t + k * ctrl.dt # ctrl.dt = 10.0
            idx_future = int(t_future / dt) # dt = 1.0
            if idx_future < len(P_ref_profile):
                P_future.append(P_ref_profile[idx_future])
            else:
                P_future.append(P_ref_profile[-1])

        if i % 10 == 0: # 10s control loop
            
            I_cmd, v_lye_cmd, v_c_cmd = ctrl.get_action(
                measured_state, P_future, T_ref=358.15 # Controller expects Kelvin
            )
            
            # Pack action for simulator: [I1..4, v1..4, vc]
            action_sim = np.concatenate([I_cmd, v_lye_cmd, [v_c_cmd]])
            last_action = [I_cmd, v_lye_cmd, v_c_cmd]
        else:
            action_sim = np.concatenate([last_action[0], last_action[1], [last_action[2]]])
            
        sim.step(action_sim)
        
        # Log
        history['t'].append(t)
        history['P_ref'].append(P_ref_profile[i])
        history['P_real'].append(P_real)
        history['T_s_in'].append(T_s_in)
        history['T_s_all'].append(T_s_vec)
        history['T_sep'].append(T_sep)
        history['T_c_out'].append(T_c_out)
        history['n_H2_an_vec'].append(n_H2_an_vec)
        history['n_liq'].append(n_liq)
        history['n_gas'].append(n_gas)
        history['T_ref'].append(358.15) # Log in Kelvin
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
            print(f"t={t:.0f}s | P_ref={P_ref_profile[i]/1e6:.1f}MW | P_real={P_real/1e6:.1f}MW | I_mean={np.mean(last_action[0]):.0f}A | v_lye={np.mean(last_action[1]):.3f} | v_c={last_action[2]:.3f} | T_s_mean={np.mean(T_s_vec)-273.15:.1f}C | HTO={hto_pct:.2f}% | H2={h2_rate:.1f}mol/s")
        
        # Periodic Plot Update (every 1000s)
        if t > 0 and t % 1000 == 0:
            print(f"Updating progress plot at t={t}s...")
            # Use d:\Projects\AWE\output\multi_stack
            output_dir = r"d:\Projects\AWE\output\multi_stack"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Save CSV snapshot
            save_data_csv(history, output_dir, data_filename)
            save_plot(history, output_dir, data_filename)
            
    # Save final data
    output_dir = r"d:\Projects\AWE\output\multi_stack"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    save_data_csv(history, output_dir, data_filename)
    save_plot(history, output_dir, data_filename)
    
    # Calculate RMSE
    t_arr = np.array(history['t'])
    P_ref_arr = np.array(history['P_ref'])
    P_real_arr = np.array(history['P_real'])
    T_s_all = np.array(history['T_s_all'])
    T_s_mean = np.mean(T_s_all, axis=1)
    T_ref_arr = np.array(history['T_ref'])
    
    mask_normal = t_arr < 3600
    mask_overload = t_arr >= 3600
    
    def calc_rmse_power(mask):
        if np.sum(mask) == 0: return 0.0
        return np.sqrt(np.mean((P_real_arr[mask] - P_ref_arr[mask])**2)) / 1e6

    def calc_rmse_temp(mask):
        if np.sum(mask) == 0: return 0.0
        return np.sqrt(np.mean((T_s_mean[mask] - T_ref_arr[mask])**2))

    rmse_p_normal = calc_rmse_power(mask_normal)
    rmse_p_overload = calc_rmse_power(mask_overload)
    rmse_p_total = calc_rmse_power(np.ones_like(t_arr, dtype=bool))
    
    rmse_t_normal = calc_rmse_temp(mask_normal)
    rmse_t_overload = calc_rmse_temp(mask_overload)
    rmse_t_total = calc_rmse_temp(np.ones_like(t_arr, dtype=bool))

    print("-" * 50)
    print(f"Performance Metrics (Duration: {duration}s)")
    print(f"{'Phase':<15} | {'Power RMSE (MW)':<15} | {'Temp RMSE (K)':<15}")
    print("-" * 50)
    print(f"{'Normal':<15} | {rmse_p_normal:<15.3f} | {rmse_t_normal:<15.3f}")
    print(f"{'Overload':<15} | {rmse_p_overload:<15.3f} | {rmse_t_overload:<15.3f}")
    print(f"{'Overall':<15} | {rmse_p_total:<15.3f} | {rmse_t_total:<15.3f}")
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
    parser.add_argument('--controller', type=str, default='diffusion', choices=['nmpc', 'diffusion'], help='Controller type')
    parser.add_argument('--model_type', type=str, default='tcn', choices=['mlp', 'tcn', 'flow_matching'], help='Diffusion model type (only for diffusion controller)')
    args = parser.parse_args()
    
    run_test(controller_type=args.controller, model_type=args.model_type)
