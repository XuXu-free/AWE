import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from plant.multi_stack_simulator import MultiStackSimulator
from controller.multi_stack.nmpc_controller import MultiStackNMPCController


def randomize_state(sim):
    # Reset to base
    sim.reset()
    
    # Randomize Temperatures
    # T_s: 20C to 90C (293K to 363K)
    T_base = np.random.uniform(293.0, 363.0)
    # Add small variation between stacks
    sim.state[0] = T_base - np.random.uniform(0, 5) # T_s_in slightly cooler
    sim.state[1:5] = T_base + np.random.uniform(-2, 2, size=4)
    
    # T_sep: similar to T_s
    sim.state[5] = T_base - np.random.uniform(2, 10)
    
    # T_c_out
    sim.state[6] = np.random.uniform(293.0, 340.0)
    
    # Gas levels (Start with some gas or empty)
    sim.state[12] = np.random.uniform(0, 100) # n_gas
    
    return sim

def save_batch(data_list, output_dir, timestamp):
    if not data_list:
        return
        
    df = pd.DataFrame(data_list)
    
    # Save CSV (Append mode)
    csv_file = os.path.join(output_dir, f"nmpc_dataset_{timestamp}.csv")
    
    # Check if file exists to determine header
    header = not os.path.exists(csv_file)
    
    df.to_csv(csv_file, mode='a', header=header, index=False)
    print(f"Appended {len(data_list)} rows to: {csv_file}")

def plot_all(output_dir, timestamp):
    csv_file = os.path.join(output_dir, f"nmpc_dataset_{timestamp}.csv")
    if not os.path.exists(csv_file):
        return

    print("Generating plots...")
    try:
        # Read full file for plotting
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV for plotting: {e}")
        return

    # Create 3x3 subplot
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    # Ensure time is available
    if 'time' in df.columns:
        t_arr = df['time'].values
    else:
        t_arr = df.index.values

    # 1. Power
    ax = axes[0,0]
    if 'P_ref' in df.columns and 'P_real' in df.columns:
        ax.plot(t_arr, df['P_ref']/1e6, 'k--', label='Ref')
        ax.plot(t_arr, df['P_real']/1e6, 'b-', label='Real')
        ax.set_title('Power Tracking (Total)')
        ax.set_ylabel('MW')
        ax.legend()
    ax.grid(True)
    
    # 2. Temperatures
    ax = axes[0,1]
    if 'T_sep' in df.columns:
        ax.plot(t_arr, df['T_sep'], 'g--', label='Separator')
    if 'T_c_out' in df.columns:
        ax.plot(t_arr, df['T_c_out'], 'c:', label='CW Out')
    for i in range(4):
        col = f'T_s_{i+1}'
        if col in df.columns:
            ax.plot(t_arr, df[col], label=f'Stack {i+1}')
    if 'T_ref' in df.columns:
        ax.plot(t_arr, df['T_ref'], 'k--', linewidth=1.5, label='Ref Temp')
    ax.set_title('All Temperatures')
    ax.set_ylabel('K')
    ax.legend()
    ax.grid(True)
    
    # 3. HTO
    ax = axes[0,2]
    if 'HTO' in df.columns:
        ax.plot(t_arr, df['HTO'], 'm-', label='HTO')
    ax.set_title('HTO (H2 in O2)')
    ax.set_ylabel('%')
    ax.grid(True)
    
    # 4. Currents
    ax = axes[1,0]
    for i in range(4):
        col = f'I_{i+1}'
        if col in df.columns:
            ax.plot(t_arr, df[col], label=f'Stack {i+1}')
    ax.set_title('Stack Currents')
    ax.set_ylabel('Amps')
    ax.legend()
    ax.grid(True)
    
    # 5. Voltages
    ax = axes[1,1]
    for i in range(4):
        col = f'U_cell_{i+1}'
        if col in df.columns:
            ax.plot(t_arr, df[col], label=f'Stack {i+1}')
    ax.set_title('Stack Voltages (Per Cell)')
    ax.set_ylabel('Volts')
    ax.legend()
    ax.grid(True)
    
    # 6. Lye Flow
    ax = axes[1,2]
    for i in range(4):
        col = f'v_lye_{i+1}'
        if col in df.columns:
            ax.plot(t_arr, df[col], label=f'Stack {i+1}')
    ax.set_title('Stack Lye Flow')
    ax.set_ylabel('m3/s')
    ax.legend()
    ax.grid(True)
    
    # 7. Coolant Flow
    ax = axes[2,0]
    if 'v_c' in df.columns:
        ax.plot(t_arr, df['v_c'], 'cyan', label='CW')
    ax.set_title('Coolant Flow')
    ax.set_ylabel('m3/s')
    ax.grid(True)
    
    # 8. H2 Rate
    ax = axes[2,1]
    if 'H2_rate' in df.columns:
        ax.plot(t_arr, df['H2_rate'], 'g-', label='H2 Rate')
    ax.set_title('Total H2 Production Rate')
    ax.set_ylabel('mol/s')
    ax.grid(True)
    
    # 9. Empty
    ax = axes[2,2]
    ax.axis('off')
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, f"diffusion_data_{timestamp}.png")
    plt.savefig(plot_file)
    print(f"Plots saved to: {plot_file}")
    plt.close()

def load_monthly_profiles():
    import glob
    # Get project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    
    base_dir = os.path.join(project_root, 'output', 'power', 'wind')
    # Modified to only load January 2025 data
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
    
    # Get project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    
    output_dir = os.path.join(project_root, 'output', 'multi_stack', 'dataset')
    
    # NMPC parameters
    horizon = 5
    sim_dt = 0.2 # 0.2s simulation step
    T_ref = 353.15  # 80°C

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
    total_steps = len(full_profile) - horizon
    
    print(f"Loaded profile with {len(full_profile)} points.")
    print(f"Generating dataset for {total_steps} steps (approx {total_steps * dt_ctrl / 3600:.1f} hours).")
    
    # Initialize Simulator
    # Simulator runs at fine time step (e.g. 0.2s)
    sim = MultiStackSimulator(dt=sim_dt)
    sim.reset() # Initialize state to default values (zeros/initial conditions) to avoid NoneType error
    
    # Initialize Controller with Control Interval
    try:
        controller = MultiStackNMPCController(dt=dt_ctrl, horizon=horizon, dt_sub=sim_dt)
    except Exception as e:
        print(f"Failed to initialize controller: {e}")
        return

    # Storage
    data_list = []
    start_step = 0
    
    # Check for existing data to resume
    # Look for the latest csv in output_dir
    csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv') and 'nmpc_dataset' in f]
    if csv_files:
        latest_csv_name = max(csv_files, key=lambda x: os.path.getctime(os.path.join(output_dir, x)))
        latest_csv_path = os.path.join(output_dir, latest_csv_name)
        
        print(f"Found existing dataset: {latest_csv_name}")
        try:
            df_existing = pd.read_csv(latest_csv_path)
            if not df_existing.empty and 'step' in df_existing.columns:
                last_step = df_existing['step'].max()
                if last_step < total_steps - 1:
                    print(f"Resuming from step {last_step + 1}...")
                    start_step = int(last_step + 1)
                    timestamp = latest_csv_name.replace('nmpc_dataset_', '').replace('.csv', '')
                    
                    # Load existing data into list (optional, but good for consistent saving)
                    # For large datasets, we might just append to file, but here we keep in memory for simplicity/plotting
                    # Only load if memory permits or if needed for continuity (e.g. plotting)
                    # To be safe and simple: We will append new rows to data_list and write in 'a' mode or rewrite.
                    # But the script rewrites the whole file currently.
                    # Let's load it back.
                    # data_list = df_existing.to_dict('records')
                    
                    # Restore Simulator State from last row
                    last_row = df_existing.iloc[-1]
                    
                    # Restore Sim State
                    # State vector: [T_s_in, T_s_vec(4), T_sep, T_c_out, n_H2_an_vec(4), n_liq, n_gas]
                    # We need to make sure sim.state is initialized properly (it should be after sim = MultiStackSimulator())
                    
                    # Update individual elements to avoid shape mismatch or reference issues
                    sim.state[0] = float(last_row['T_s_in'])
                    sim.state[1:5] = np.array([last_row['T_s_1'], last_row['T_s_2'], last_row['T_s_3'], last_row['T_s_4']], dtype=float)
                    sim.state[5] = float(last_row['T_sep'])
                    sim.state[6] = float(last_row['T_c_out'])
                    sim.state[7:11] = np.array([last_row['n_H2_an_1'], last_row['n_H2_an_2'], last_row['n_H2_an_3'], last_row['n_H2_an_4']], dtype=float)
                    sim.state[11] = float(last_row['n_liq'])
                    sim.state[12] = float(last_row['n_gas'])
                    
                    # Restore Controller Last Action (for smoothness cost)
                    controller.last_I = np.array([last_row['I_1'], last_row['I_2'], last_row['I_3'], last_row['I_4']])
                    controller.last_v_lye = np.array([last_row['v_lye_1'], last_row['v_lye_2'], last_row['v_lye_3'], last_row['v_lye_4']])
                    controller.last_v_c = last_row['v_c']
                    
                    # Restore Action History for logging
                    I_prev = controller.last_I
                    v_lye_prev = controller.last_v_lye
                    v_c_prev = controller.last_v_c
                    
                else:
                    print("Existing dataset appears complete or near end. Starting fresh or check horizon.")
                    # If complete, maybe we want to extend? But total_steps is fixed by profile.
                    # Let's assume start fresh if complete.
                    pass
        except Exception as e:
            print(f"Failed to resume from {latest_csv_path}: {e}. Starting fresh.")
    
    if start_step == 0:
        # Randomize Initial Condition ONCE if starting fresh
        randomize_state(sim)
        
        # Initialize previous actions for logging
        I_prev = np.zeros(4)
        v_lye_prev = np.zeros(4)
        v_c_prev = 0.0
    
    # Reset Controller State (only if not resumed, but controller init resets it anyway, so we just restored it above if needed)
    # If starting fresh, prev_sol_x is None. If resumed, we don't have prev_sol_x, so it will warm start from last_I (Cold-ish start)
    # We could theoretically save/load prev_sol_x but it's not critical.
    
    try:
        # Main Loop with Progress Bar
        for t in tqdm(range(start_step, total_steps), desc="Generating Dataset"):
            # Current State (At beginning of interval)
            current_state = sim.state.copy()
            
            T_s_in = current_state[0]
            T_s_vec = current_state[1:5]
            T_sep = current_state[5]
            T_c_out = current_state[6]
            n_H2_an_vec = current_state[7:11]
            n_liq = current_state[11]
            n_gas = current_state[12]
            
            # 1. Get P_ref (Current target for this interval)
            current_P_ref = full_profile[t]
            
            # Future P_ref (Next N intervals)
            P_ref_future = []
            for h in range(horizon):
                future_val = full_profile[t + h] 
                P_ref_future.append(future_val)

            # 2. Determine Action via NMPC
            try:
                u_opt_matrix, states_matrix = controller.get_all_actions_states(current_state, P_ref_future, T_ref)
            except Exception as e:
                print(f"NMPC failed at step {t}: {e}")
                break
                
            # Extract first action for execution
            u0 = u_opt_matrix[0]
            I_cmd = u0[0:4]
            v_lye_cmd = u0[4:8]
            v_c_cmd = u0[8]
            
            # 3. Calculate derived properties (Instantaneous at start)
            _, U_cell_vec, _ = sim._calculate_electrochemical_properties(I_cmd, T_s_vec)
            P_real = np.sum(I_cmd * U_cell_vec * sim.N_cell)
            
            V_sep_gas = 2.0
            p_sys = 1.6e6
            R = 8.314
            hto_val = (n_gas * R * T_sep) / (p_sys * V_sep_gas) * 100.0
            
            H2_rate_vec = sim.N_cell * I_cmd / (2 * 96485.0)
            H2_rate_total = np.sum(H2_rate_vec)
            
            # 4. Step Simulator (Integrate over Control Interval)
            # dt_ctrl = 60s, sim_dt = 0.2s -> 300 steps
            sim_steps = int(dt_ctrl / sim_dt)
            action_sim = np.concatenate([I_cmd, v_lye_cmd, [v_c_cmd]])
            
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
                'T_sep': T_sep,
                'T_c_out': T_c_out,
                'n_liq': n_liq,
                'n_gas': n_gas,
                'T_ref': T_ref,
                'v_c_prev': v_c_prev,
                'v_c': v_c_cmd,
                'HTO': hto_val,
                'H2_rate': H2_rate_total,
                # Vectors
                'T_s_1': T_s_vec[0], 'T_s_2': T_s_vec[1], 'T_s_3': T_s_vec[2], 'T_s_4': T_s_vec[3],
                'n_H2_an_1': n_H2_an_vec[0], 'n_H2_an_2': n_H2_an_vec[1], 'n_H2_an_3': n_H2_an_vec[2], 'n_H2_an_4': n_H2_an_vec[3],
                'I_prev_1': I_prev[0], 'I_prev_2': I_prev[1], 'I_prev_3': I_prev[2], 'I_prev_4': I_prev[3],
                'v_lye_prev_1': v_lye_prev[0], 'v_lye_prev_2': v_lye_prev[1], 'v_lye_prev_3': v_lye_prev[2], 'v_lye_prev_4': v_lye_prev[3],
                'I_1': I_cmd[0], 'I_2': I_cmd[1], 'I_3': I_cmd[2], 'I_4': I_cmd[3],
                'v_lye_1': v_lye_cmd[0], 'v_lye_2': v_lye_cmd[1], 'v_lye_3': v_lye_cmd[2], 'v_lye_4': v_lye_cmd[3],
                'U_cell_1': U_cell_vec[0], 'U_cell_2': U_cell_vec[1], 'U_cell_3': U_cell_vec[2], 'U_cell_4': U_cell_vec[3]
            }
            
            # Save NMPC Plan
            for k in range(horizon):
                row[f'P_ref_future_{k}'] = P_ref_future[k]
                u_k = u_opt_matrix[k]
                s_k = states_matrix[k]
                
                # Actions
                for i in range(4):
                    row[f'plan_step_{k}_I_{i+1}'] = u_k[i]
                for i in range(4):
                    row[f'plan_step_{k}_v_lye_{i+1}'] = u_k[4+i]
                row[f'plan_step_{k}_v_c'] = u_k[8]
                
                # States [T_s_in, T_s_1..4, T_sep, T_c_out, n_H2_an_1..4, n_liq, n_gas]
                # s_k indices: 0: T_s_in, 1-4: T_s, 5: T_sep, 6: T_c_out, 7-10: n_H2_an, 11: n_liq, 12: n_gas
                row[f'plan_step_{k}_state_T_s_in'] = s_k[0]
                for i in range(4):
                    row[f'plan_step_{k}_state_T_s_{i+1}'] = s_k[1+i]
                row[f'plan_step_{k}_state_T_sep'] = s_k[5]
                row[f'plan_step_{k}_state_T_c_out'] = s_k[6]
                
                # Non-thermal states
                for i in range(4):
                    row[f'plan_step_{k}_state_n_H2_an_{i+1}'] = s_k[7+i]
                row[f'plan_step_{k}_state_n_liq'] = s_k[11]
                row[f'plan_step_{k}_state_n_gas'] = s_k[12]
                
            data_list.append(row)
            
            # Update history
            I_prev = I_cmd
            v_lye_prev = v_lye_cmd
            v_c_prev = v_c_cmd
            
            # Periodic save
            if (t + 1) % 1000 == 0:
                save_batch(data_list, output_dir, timestamp)
                data_list = [] # Clear memory
                plot_all(output_dir, timestamp)
                
    except KeyboardInterrupt:
        print("\nDataset generation interrupted by user.")
    finally:
        # Final save
        save_batch(data_list, output_dir, timestamp)
        plot_all(output_dir, timestamp)

if __name__ == "__main__":
    generate_dataset()
