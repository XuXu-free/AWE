import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import time
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from plant.single_stack_simulator import SingleStackSimulator
from controller.single_stack.nmpc_controller import SingleStackNMPCController

def randomize_state(sim):
    # Reset to base
    sim.reset()
    
    # Randomize Temperatures
    # T_s: 20C to 90C (293K to 363K)
    T_base = np.random.uniform(293.0, 363.0)
    
    sim.state[0] = T_base - np.random.uniform(0, 5) # T_s_in
    sim.state[1] = T_base # T_s
    sim.state[2] = T_base - np.random.uniform(2, 10) # T_sep
    sim.state[3] = np.random.uniform(293.0, 340.0) # T_c_out
    
    # Gas levels (Start with some gas or empty)
    sim.state[6] = np.random.uniform(0, 100) # n_gas
    
    return sim

def save_and_plot(data_list, output_dir, timestamp):
    if not data_list:
        return
        
    df = pd.DataFrame(data_list)
    
    # Save CSV
    csv_file = os.path.join(output_dir, f"nmpc_dataset_{timestamp}.csv")
    df.to_csv(csv_file, index=False)
    print(f"Dataset saved to: {csv_file}")
    
    # Plot distributions
    print("Generating plots...")
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    t_arr = df['time'].values
    
    # Power
    ax = axes[0,0]
    ax.plot(t_arr, df['P_ref'].values/1e6, 'k--', label='Ref')
    ax.plot(t_arr, df['P_real'].values/1e6, 'b-', label='Real')
    ax.axvline(x=0, color='gray', linestyle=':', label='Start')
    ax.set_title('Power Tracking')
    ax.set_ylabel('MW')
    ax.legend()
    ax.grid(True)
    
    # All Temperatures
    ax = axes[0,1]
    ax.plot(t_arr, df['T_sep'].values, 'g--', label='Separator')
    ax.plot(t_arr, df['T_c_out'].values, 'c:', label='CW Out')
    ax.plot(t_arr, df['T_s'].values, 'r-', label='Stack')
    ax.plot(t_arr, df['T_ref'].values, 'k--', linewidth=1.5, label='Ref Temp')
    ax.axvline(x=0, color='gray', linestyle=':')
    ax.set_title('All Temperatures')
    ax.set_ylabel('K')
    ax.legend()
    ax.grid(True)
    
    # HTO %
    ax = axes[0,2]
    ax.plot(t_arr, df['HTO'].values, 'm-', label='HTO')
    ax.axvline(x=0, color='gray', linestyle=':')
    ax.set_title('HTO (H2 in O2)')
    ax.set_ylabel('%')
    ax.grid(True)
    
    # Stack Current
    ax = axes[1,0]
    ax.plot(t_arr, df['I'].values, 'b-', label='Stack')
    ax.axvline(x=0, color='gray', linestyle=':')
    ax.set_title('Stack Current')
    ax.set_ylabel('Amps')
    ax.legend()
    ax.grid(True)
    
    # Stack Voltage
    ax = axes[1,1]
    ax.plot(t_arr, df['U_cell'].values, 'b-', label='Stack')
    ax.axvline(x=0, color='gray', linestyle=':')
    ax.set_title('Cell Voltage')
    ax.set_ylabel('Volts')
    ax.legend()
    ax.grid(True)
    
    # Stack Lye Flow
    ax = axes[1,2]
    ax.plot(t_arr, df['v_lye'].values, 'orange', label='Stack')
    ax.axvline(x=0, color='gray', linestyle=':')
    ax.set_title('Stack Lye Flow')
    ax.set_ylabel('m3/s')
    ax.legend()
    ax.grid(True)
    
    # Coolant Flow
    ax = axes[2,0]
    ax.plot(t_arr, df['v_c'].values, 'cyan', label='CW')
    ax.axvline(x=0, color='gray', linestyle=':')
    ax.set_title('Coolant Flow')
    ax.set_ylabel('m3/s')
    ax.grid(True)
    
    # H2 Production Rate
    ax = axes[2,1]
    ax.plot(t_arr, df['H2_rate'].values, 'g-', label='H2 Rate')
    ax.axvline(x=0, color='gray', linestyle=':')
    ax.set_title('H2 Production Rate')
    ax.set_ylabel('mol/s')
    ax.grid(True)
    
    # Empty
    ax = axes[2,2]
    ax.axis('off')
            
    plt.tight_layout()
    plot_file = os.path.join(output_dir, f"nmpc_dataset_{timestamp}.png")
    plt.savefig(plot_file)
    print(f"Plots saved to: {plot_file}")
    plt.close()

def load_monthly_profiles():
    import glob
    # Use relative path
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'output', 'power', 'single_stack'))
    # Load January 2025 data
    pattern = os.path.join(base_dir, 'wind_power_2025-01_1min.csv')
    files = sorted(glob.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"No monthly profiles found in {base_dir} matching pattern {pattern}")
        
    print(f"Found {len(files)} monthly profile files: {[os.path.basename(f) for f in files]}")
    
    # Concatenate all profiles
    all_profiles = []
    for f in files:
        df = pd.read_csv(f)
        all_profiles.append(df['P_ref'].values)
        
    full_profile = np.concatenate(all_profiles)
    return full_profile

def generate_dataset():
    # Configuration
    # dt_ctrl = 1 min = 60 s
    dt_ctrl = 60.0
    
    # Output dir relative to script
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'output', 'single_stack', 'dataset'))
    
    # NMPC parameters
    horizon = 5
    sim_dt = 0.2 # 0.2s simulation step
    T_ref = 358.15  # 85°C (Kelvin)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load Real Profile
    try:
        full_profile = load_monthly_profiles()
    except FileNotFoundError as e:
        print(e)
        return


    # Total steps to run
    # We stop early if horizon exceeds data
    # Run for shorter duration for testing if needed, or full month
    # For now run full month as in multi-stack
    total_steps = len(full_profile) - horizon
    
    print(f"Loaded profile with {len(full_profile)} points.")
    print(f"Generating dataset for {total_steps} steps (approx {total_steps * dt_ctrl / 3600:.1f} hours).")
    
    # Initialize Simulator
    # Simulator runs at fine time step (e.g. 0.2s)
    sim = SingleStackSimulator(dt=sim_dt)
    
    # Initialize Controller with Control Interval
    try:
        controller = SingleStackNMPCController(dt=dt_ctrl, N_p=horizon)
    except Exception as e:
        print(f"Failed to initialize controller: {e}")
        return

    # Storage
    data_list = []
    
    # Randomize Initial Condition ONCE
    randomize_state(sim)
    
    # Reset Controller State
    controller.prev_sol_x = None
    controller.last_I = 0.0
    controller.last_v_lye = 0.03
    controller.last_v_c = 0.0
    
    # Reset Action History for logging
    I_prev = 0.0
    v_lye_prev = 0.0
    v_c_prev = 0.0
    
    try:
        # Main Loop with Progress Bar
        for t in tqdm(range(total_steps), desc="Generating Dataset"):
            # Current State (At beginning of interval)
            current_state = sim.state.copy()
            
            # [T_s_in, T_s, T_sep, T_c_out, n_H2_an, n_liq, n_gas]
            T_s_in = current_state[0]
            T_s = current_state[1]
            T_sep = current_state[2]
            T_c_out = current_state[3]
            n_H2_an = current_state[4]
            n_liq = current_state[5]
            n_gas = current_state[6]
            
            # 1. Get P_ref (Current target for this interval)
            current_P_ref = full_profile[t]
            
            # Future P_ref (Next N intervals)
            P_ref_future = []
            for h in range(horizon):
                future_val = full_profile[t + h] 
                P_ref_future.append(future_val)

            # 2. Determine Action via NMPC
            try:
                # Returns (N, 3) and (N+1, 4)
                u_opt_matrix, states_matrix = controller.get_all_actions_states(current_state, P_ref_future, T_ref)
            except Exception as e:
                print(f"NMPC failed at step {t}: {e}")
                break
                
            # Extract first action for execution
            u0 = u_opt_matrix[0]
            I_cmd = u0[0]
            v_lye_cmd = u0[1]
            v_c_cmd = u0[2]
            
            # 3. Calculate derived properties (Instantaneous at start)
            _, U_cell, _ = sim._calculate_electrochemical_properties(I_cmd, T_s)
            P_real = I_cmd * U_cell * sim.N_cell
            
            V_sep_gas = 0.6 * 10.288 # V_sep_gas
            p_sys = 1.6e6
            R = 8.314
            hto_val = (n_gas * R * T_sep) / (p_sys * V_sep_gas) * 100.0
            
            H2_rate_total = sim.N_cell * I_cmd / (2 * 96485.0)
            
            # 4. Step Simulator (Integrate over Control Interval)
            # dt_ctrl = 60s, sim_dt = 0.2s -> 300 steps
            sim_steps = int(dt_ctrl / sim_dt)
            action_sim = np.array([I_cmd, v_lye_cmd, v_c_cmd])
            
            for _ in range(sim_steps):
                sim.step(action_sim)
            
            # 5. Store Data (Snapshot at start of interval)
            # Note: 't' here is step index.
            row = {
                'step': t,
                'time': t * dt_ctrl,
                'P_ref': current_P_ref,
                'P_real': P_real,
                'T_s_in': T_s_in,
                'T_s': T_s,
                'T_sep': T_sep,
                'T_c_out': T_c_out,
                'n_liq': n_liq,
                'n_gas': n_gas,
                'T_ref': T_ref,
                'v_c_prev': v_c_prev,
                'v_c': v_c_cmd,
                'v_lye': v_lye_cmd,
                'HTO': hto_val,
                'H2_rate': H2_rate_total,
                'I': I_cmd,
                'U_cell': U_cell
            }
            
            # Save NMPC Plan
            for k in range(horizon):
                row[f'P_ref_future_{k}'] = P_ref_future[k]
                u_k = u_opt_matrix[k]
                s_k = states_matrix[k]
                
                # Actions [I, v_lye, v_c]
                row[f'plan_step_{k}_I'] = u_k[0]
                row[f'plan_step_{k}_v_lye'] = u_k[1]
                row[f'plan_step_{k}_v_c'] = u_k[2]
                
                # States [T_s_in, T_s, T_sep, T_c_out]
                row[f'plan_step_{k}_state_T_s_in'] = s_k[0]
                row[f'plan_step_{k}_state_T_s'] = s_k[1]
                row[f'plan_step_{k}_state_T_sep'] = s_k[2]
                row[f'plan_step_{k}_state_T_c_out'] = s_k[3]
                
            data_list.append(row)
            
            # Update history
            I_prev = I_cmd
            v_lye_prev = v_lye_cmd
            v_c_prev = v_c_cmd
            
            # Periodic save
            if (t + 1) % 1000 == 0:
                save_and_plot(data_list, output_dir, timestamp)
                
    except KeyboardInterrupt:
        print("\nDataset generation interrupted by user.")
    finally:
        # Final save
        save_and_plot(data_list, output_dir, timestamp)

if __name__ == "__main__":
    generate_dataset()
