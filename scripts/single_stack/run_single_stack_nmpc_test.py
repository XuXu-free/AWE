
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from plant.single_stack_simulator import SingleStackSimulator
from controller.single_stack.single_stack_nmpc_controller import SingleStackNMPCController

def save_plot(history, output_dir, filename):
    t_arr = np.array(history['t'])
    if len(t_arr) == 0:
        return
        
    T_s_all = np.array(history['T_s_all']) # Shape: (N,) or (N, 1)
    U_cell_all = np.array(history['U_cell_all']) # Shape: (N,) or (N, 1)
    
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
    
    # System Temperatures
    ax = axes[0,1]
    ax.plot(t_arr, history['T_sep'], 'g--', label='Separator')
    ax.plot(t_arr, history['T_c_out'], 'c:', label='CW Out')
    ax.plot(t_arr, history['T_ref'], 'k--', linewidth=1.5, label='Ref Temp')
    ax.axvline(x=0, color='gray', linestyle=':')
    ax.set_title('System Temperatures')
    ax.set_ylabel('°C')
    ax.legend()
    ax.grid(True)
    
    # HTO %
    ax = axes[0,2]
    ax.plot(t_arr, history['HTO'], 'm-', label='HTO')
    ax.axvline(x=0, color='gray', linestyle=':')
    ax.set_title('HTO (H2 in O2)')
    ax.set_ylabel('%')
    ax.grid(True)
    
    # Stack Temperatures
    ax = axes[1,0]
    ax.plot(t_arr, T_s_all, label='Stack')
    ax.plot(t_arr, history['T_ref'], 'k--', linewidth=1.5, label='Ref Temp')
    ax.axvline(x=0, color='gray', linestyle=':')
    ax.set_title('Stack Temperature')
    ax.set_ylabel('°C')
    ax.legend()
    ax.grid(True)
    
    # Current
    ax = axes[1,1]
    ax.plot(t_arr, history['I'], 'b-')
    ax.axvline(x=0, color='gray', linestyle=':')
    ax.set_title('Stack Current')
    ax.set_ylabel('Amps')
    ax.grid(True)
    
    # Stack Voltages
    ax = axes[1,2]
    ax.plot(t_arr, U_cell_all, label='Stack')
    ax.axvline(x=0, color='gray', linestyle=':')
    ax.set_title('Cell Voltage')
    ax.set_ylabel('Volts')
    ax.legend()
    ax.grid(True)
    
    # Lye Flow
    ax = axes[2,0]
    ax.plot(t_arr, history['v_lye'], 'orange', label='Lye')
    ax.axvline(x=0, color='gray', linestyle=':')
    ax.set_title('Lye Flow')
    ax.set_ylabel('m3/s')
    ax.grid(True)

    # CW Flow
    ax = axes[2,1]
    ax.plot(t_arr, history['v_c'], 'cyan', label='CW')
    ax.axvline(x=0, color='gray', linestyle=':')
    ax.set_title('Coolant Flow')
    ax.set_ylabel('m3/s')
    ax.grid(True)
    
    # H2 Production Rate
    ax = axes[2,2]
    ax.plot(t_arr, history['H2_rate'], 'g-', label='H2 Rate')
    ax.axvline(x=0, color='gray', linestyle=':')
    ax.set_title('H2 Production Rate')
    ax.set_ylabel('mol/s')
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close(fig)

def save_data_csv(history, output_dir, filename):
    # Keys for scalar columns
    keys_scalar = ['t', 'P_ref', 'P_real', 'T_s_in', 'T_s', 'T_sep', 'T_c_out', 'n_H2_an', 'n_liq', 'n_gas', 'T_ref', 'I_prev', 'v_lye_prev', 'v_c_prev', 'I', 'v_lye', 'v_c', 'HTO', 'H2_rate', 'U_cell']
    
    if not history['t']:
        return
        
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Determine Horizon N from the first entry of P_ref_future
        if history['P_ref_future']:
            N_horizon = len(history['P_ref_future'][0])
        else:
            N_horizon = 0
            
        # Header
        # Expand P_ref_future into columns
        header = keys_scalar + [f'P_ref_future_{k}' for k in range(N_horizon)]
        writer.writerow(header)
        
        # Rows
        n_steps = len(history['t'])
        for i in range(n_steps):
            row = [
                history['t'][i],
                history['P_ref'][i],
                history['P_real'][i],
                history['T_s_in'][i],
                history['T_s_all'][i],
                history['T_sep'][i],
                history['T_c_out'][i],
                history['n_H2_an'][i],
                history['n_liq'][i],
                history['n_gas'][i],
                history['T_ref'][i],
                history['I_prev'][i],
                history['v_lye_prev'][i],
                history['v_c_prev'][i],
                history['I'][i],
                history['v_lye'][i],
                history['v_c'][i],
                history['HTO'][i],
                history['H2_rate'][i],
                history['U_cell_all'][i]
            ]
            # Extend with P_ref_future vector
            if N_horizon > 0:
                row.extend(history['P_ref_future'][i])
                
            writer.writerow(row)
    print(f"Data saved to {filepath}")

def generate_power_profile_values(t_eval):
    # Scaled down for single stack (approx 1/4 of multi-stack)
    for t in t_eval:
        if t < 3600:
            # Normal / Fluctuating Phase (0 - 1h)
            # Range approx 0.75MW to 5.5MW
            val = 3.0 + 1.5 * np.sin(2 * np.pi * t / 11000) + \
                  1.0 * np.sin(2 * np.pi * t / 3700) + \
                  0.5 * np.sin(2 * np.pi * t / 1300)
            
            val += 0.25 * np.sin(2 * np.pi * t / 300)
            val = np.clip(val, 0.75, 5.5)
            yield val * 1e6
            
        else:
            # Overload Phase (1h - 2h)
            # Range approx 6.25 - 8.25 MW (capped by constraints likely)
            t_shift = t - 3600
            val = 7.0 + 0.75 * np.sin(2 * np.pi * t_shift / 2500) + \
                  0.375 * np.sin(2 * np.pi * t / 800)
            
            val = np.clip(val, 6.25, 8.25)
            yield val * 1e6

def run_warmup_phase(sim, ctrl, history, last_action, dt, warmup_duration=1000, warmup_P_ref=2.0e6):
    print(f"Starting Warm-up Phase ({warmup_duration}s at {warmup_P_ref/1e6}MW)...")
    
    warmup_steps = int(warmup_duration / dt)
    for i in range(warmup_steps):
        t_warmup = -warmup_duration + i * dt
        
        # Extract State
        # x = [T_s_in, T_s, T_sep, T_c_out, n_H2_an, n_H2_sep_liq, n_H2_sep_gas]
        T_s_in = sim.state[0] - 273.15
        T_s = sim.state[1] - 273.15
        T_sep = sim.state[2] - 273.15
        T_c_out = sim.state[3] - 273.15
        n_H2_an = sim.state[4]
        n_liq = sim.state[5]
        n_gas = sim.state[6]
        
        # Calculate Real Power
        Q, U_cell, eta = sim._calculate_electrochemical_properties(last_action[0], sim.state[1])
        P_real = U_cell * last_action[0] * sim.N_cell
        
        # Calculate Extra Metrics
        n_H2_sep_gas = sim.state[6]
        T_sep_K = sim.state[2]
        n_gas_total = (sim.p_sys * sim.V_sep_gas) / (sim.R * T_sep_K)
        hto_pct = (n_H2_sep_gas / n_gas_total) * 100 if n_gas_total > 1e-9 else 0.0
        
        F_const = 96485.0
        h2_rate = sim.N_cell * last_action[0] * eta / (2 * F_const)

        # Store previous action before update
        prev_action = list(last_action)

        # Control
        if i % 10 == 0: # 10s control loop
            I_cmd, v_lye_cmd, v_c_cmd = ctrl.get_action(
                sim.state, warmup_P_ref, T_ref=358.15
            )
            action_sim = np.array([I_cmd, v_lye_cmd, v_c_cmd])
            last_action = [I_cmd, v_lye_cmd, v_c_cmd]
        else:
            action_sim = np.array(last_action)
            
        sim.step(action_sim)
        
        # Log Warmup
        history['t'].append(t_warmup)
        history['P_ref'].append(warmup_P_ref)
        history['P_real'].append(P_real)
        history['T_s_in'].append(T_s_in)
        history['T_s_all'].append(T_s)
        history['T_sep'].append(T_sep)
        history['T_c_out'].append(T_c_out)
        history['n_H2_an'].append(n_H2_an)
        history['n_liq'].append(n_liq)
        history['n_gas'].append(n_gas)
        history['T_ref'].append(85.0 + 273.15) # Log in Kelvin
        # P_ref_future is constant during warmup
        history['P_ref_future'].append([warmup_P_ref] * ctrl.N)
        history['I_prev'].append(prev_action[0])
        history['v_lye_prev'].append(prev_action[1])
        history['v_c_prev'].append(prev_action[2])
        history['I'].append(last_action[0])
        history['v_lye'].append(last_action[1])
        history['v_c'].append(last_action[2])
        history['U_cell_all'].append(U_cell)
        history['HTO'].append(hto_pct)
        history['H2_rate'].append(h2_rate)
        
        if i % 100 == 0:
            print(f"Warmup: {(i*dt)/warmup_duration*100:.0f}% | T_s={T_s:.1f}K")

    print("Warm-up Complete. Starting Main Test...")
    return last_action

def run_test():
    # Setup
    dt = 1.0
    sim = SingleStackSimulator(dt=dt)
    ctrl = SingleStackNMPCController(dt=10.0, horizon=10)
    
    # Init
    sim.reset()
    
    # Profile
    duration = 7200 # 2 hours
    t_eval = np.arange(0, duration, dt)
    
    P_ref_profile = np.array(list(generate_power_profile_values(t_eval)))
            
    history = {
        't': [], 'P_ref': [], 'P_real': [],
        'T_s_in': [], 'T_s_all': [], 'T_sep': [], 'T_c_out': [], 
        'n_H2_an': [], 'n_liq': [], 'n_gas': [],
        'T_ref': [], 'P_ref_future': [],
        'I_prev': [], 'v_lye_prev': [], 'v_c_prev': [],
        'I': [], 'v_lye': [], 'v_c': [],
        'U_cell_all': [], 'HTO': [], 'H2_rate': []
    }
    
    # Initialize previous action
    prev_action_log = [0.0, 0.03, 0.0] # I_prev, v_lye_prev, v_c_prev
    last_action = [0.0, 0.03, 0.0]
    
    # --- Warm-up Phase ---
    # Note: Warmup phase logging needs update too if we want full consistency, 
    # but for now we focus on the main loop or update warmup as well.
    # Let's update warmup signature to accept history keys or just handle it inside.
    # For simplicity, I'll update run_warmup_phase to handle the new keys.
    last_action = run_warmup_phase(sim, ctrl, history, last_action, dt)
    prev_action_log = last_action # For the first step of main loop

    print("Starting Single Stack NMPC Test...")
    
    data_filename = f"single_stack_nmpc_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    # Loop
    # Initialize P_future for the first step if not already available
    # We will compute it inside the loop
    P_future_log = [] # Temporary to hold P_future for logging

    for i, t in enumerate(t_eval):
        # Extract State
        T_s_in = sim.state[0]
        T_s = sim.state[1]
        T_sep = sim.state[2]
        T_c_out = sim.state[3]
        n_H2_an = sim.state[4]
        n_liq = sim.state[5]
        n_gas = sim.state[6]
        
        # Calculate Real Power
        Q, U_cell, eta = sim._calculate_electrochemical_properties(last_action[0], sim.state[1])
        P_real = U_cell * last_action[0] * sim.N_cell
        
        # Calculate Extra Metrics
        n_H2_sep_gas = sim.state[6]
        # T_sep is already Kelvin
        n_gas_total = (sim.p_sys * sim.V_sep_gas) / (sim.R * T_sep)
        hto_pct = (n_H2_sep_gas / n_gas_total) * 100 if n_gas_total > 1e-9 else 0.0
        
        F_const = 96485.0
        h2_rate = sim.N_cell * last_action[0] * eta / (2 * F_const)
        
        # Store previous action before update
        prev_action = list(last_action)

        # Control
        # Always compute P_future for logging, even if control doesn't update
        P_future = []
        for k in range(ctrl.N):
            t_future = t + k * ctrl.dt
            idx_future = int(t_future / dt)
            if idx_future < len(P_ref_profile):
                P_future.append(P_ref_profile[idx_future])
            else:
                P_future.append(P_ref_profile[-1])

        if i % 10 == 0: # 10s control loop
            I_cmd, v_lye_cmd, v_c_cmd = ctrl.get_action(
                sim.state, P_future, T_ref=358.15 # Controller expects Kelvin
            )
            
            action_sim = np.array([I_cmd, v_lye_cmd, v_c_cmd])
            last_action = [I_cmd, v_lye_cmd, v_c_cmd]
        else:
            action_sim = np.array(last_action)
            
        sim.step(action_sim)
        
        # Log
        history['t'].append(t)
        history['P_ref'].append(P_ref_profile[i])
        history['P_real'].append(P_real)
        history['T_s_in'].append(T_s_in)
        history['T_s_all'].append(T_s)
        history['T_sep'].append(T_sep)
        history['T_c_out'].append(T_c_out)
        history['n_H2_an'].append(n_H2_an)
        history['n_liq'].append(n_liq)
        history['n_gas'].append(n_gas)
        history['T_ref'].append(85.0 + 273.15) # Log in Kelvin
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
        
        if i % 100 == 0:
            print(f"t={t:.0f}s | P_ref={P_ref_profile[i]/1e6:.1f}MW | P_real={P_real/1e6:.1f}MW | T_s={T_s:.1f}K")
        
        # Periodic Plot Update (every 2000s)
        if t > 0 and t % 2000 == 0:
            print(f"Updating progress plot at t={t}s...")
            output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            save_plot(history, output_dir, 'single_stack_nmpc_test_progress.png')
            save_data_csv(history, output_dir, data_filename)
            
    # Plot
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    save_plot(history, output_dir, 'single_stack_nmpc_test.png')
    save_data_csv(history, output_dir, data_filename)
    
    # RMSE
    t_arr = np.array(history['t'])
    P_ref_arr = np.array(history['P_ref'])
    P_real_arr = np.array(history['P_real'])
    T_s_all = np.array(history['T_s_all'])
    T_ref_arr = np.array(history['T_ref'])
    
    mask_normal = t_arr < 3600
    mask_overload = t_arr >= 3600
    
    def calc_rmse_power(mask):
        if np.sum(mask) == 0: return 0.0
        return np.sqrt(np.mean((P_real_arr[mask] - P_ref_arr[mask])**2)) / 1e6

    def calc_rmse_temp(mask):
        if np.sum(mask) == 0: return 0.0
        return np.sqrt(np.mean((T_s_all[mask] - T_ref_arr[mask])**2))

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

if __name__ == "__main__":
    run_test()
