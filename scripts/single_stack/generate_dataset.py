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
import argparse
import glob

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from plant.single_stack_simulator import SingleStackSimulator
from controller.single_stack.nmpc_controller import SingleStackNMPCController

def randomize_state(sim):
    # Reset to base
    sim.reset()
    
    # # Randomize Temperatures
    # # T_s: 20C to 90C (293K to 363K)
    # T_base = np.random.uniform(293.0, 363.0)
    
    # sim.state[0] = T_base - np.random.uniform(0, 5) # T_s_in
    # sim.state[1] = T_base # T_s
    # sim.state[2] = T_base - np.random.uniform(2, 10) # T_sep
    # sim.state[3] = np.random.uniform(293.0, 340.0) # T_c_out
    
    # # Gas levels (Start with some gas or empty)
    # sim.state[4] = np.random.uniform(0, 100) # n_H2_an
    # sim.state[5] = np.random.uniform(0, 100) # n_liq
    # sim.state[6] = np.random.uniform(0, 100) # n_gas
    
    return sim

def save_batch(data_list, output_dir, timestamp, prefix="nmpc_dataset"):
    if not data_list:
        return
        
    df = pd.DataFrame(data_list)
    
    # Save CSV (Append mode)
    csv_file = os.path.join(output_dir, f"{prefix}_{timestamp}.csv")
    
    # Check if file exists to determine header
    header = not os.path.exists(csv_file)
    
    df.to_csv(csv_file, mode='a', header=header, index=False)
    print(f"Appended {len(data_list)} rows to: {csv_file}")

def plot_all(output_dir, timestamp, prefix="nmpc_dataset", df=None):
    if df is None:
        csv_file = os.path.join(output_dir, f"{prefix}_{timestamp}.csv")
        if not os.path.exists(csv_file):
            return

        print(f"Generating plots for {csv_file}...")
        try:
            # Read only necessary columns for plotting to save memory if file is huge
            # For simplicity, read all, but be mindful of large files
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
        ax.set_title('Power Tracking')
        ax.set_ylabel('MW')
        ax.legend()
    ax.grid(True)
    
    # 2. Temperatures
    ax = axes[0,1]
    if 'T_sep' in df.columns:
        ax.plot(t_arr, df['T_sep'], 'g--', label='Separator')
    if 'T_c_out' in df.columns:
        ax.plot(t_arr, df['T_c_out'], 'c:', label='CW Out')
    if 'T_s' in df.columns:
        ax.plot(t_arr, df['T_s'], 'r-', label='Stack')
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
    if 'I' in df.columns:
        ax.plot(t_arr, df['I'], 'b-', label='Stack')
    ax.set_title('Stack Current')
    ax.set_ylabel('Amps')
    ax.legend()
    ax.grid(True)
    
    # 5. Voltages
    ax = axes[1,1]
    if 'U_cell' in df.columns:
        ax.plot(t_arr, df['U_cell'], 'b-', label='Stack')
    ax.set_title('Cell Voltage')
    ax.set_ylabel('Volts')
    ax.legend()
    ax.grid(True)
    
    # 6. Lye Flow
    ax = axes[1,2]
    if 'v_lye' in df.columns:
        ax.plot(t_arr, df['v_lye'], 'orange', label='Stack')
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
    ax.set_title('H2 Production Rate')
    ax.set_ylabel('mol/s')
    ax.grid(True)
    
    # 9. Empty
    ax = axes[2,2]
    ax.axis('off')
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, f"{prefix}_plot_{timestamp}.png")
    plt.savefig(plot_file)
    print(f"Plots saved to: {plot_file}")
    plt.close()

def load_monthly_profiles():
    # Use relative path to wind data
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'output', 'power', 'wind'))
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
    
    # Scale for single stack (approx 1/4 of total power)
    full_profile = full_profile * 0.25
    
    return full_profile

def run_warmup_phase(sim, controller, full_profile, dt_ctrl, horizon, sim_dt, T_ref, output_dir, timestamp, I_prev, v_lye_prev, v_c_prev):
    print("\nStarting Warmup Phase (4 hours)...")
    warmup_duration_hours = 4
    warmup_steps = int(warmup_duration_hours * 3600 / dt_ctrl)
    plot_every_steps = max(1, int(3600 / dt_ctrl))
    
    # Warmup starts at 2MW (approx 1/4 of 8MW)
    P_warmup_start = 2.0e6 
    P_warmup_end = full_profile[0]
    
    warmup_data_list = []
    warmup_plot_history = []
    
    try:
        for t_w in tqdm(range(warmup_steps), desc="Warmup"):
            # Current time (negative)
            current_time = -(warmup_steps - t_w) * dt_ctrl
            
            # Interpolate P_ref
            alpha = t_w / max(1, warmup_steps - 1)
            current_P_ref = P_warmup_start + alpha * (P_warmup_end - P_warmup_start)
            
            # Future P_ref
            P_ref_future = []
            for h in range(horizon):
                future_idx = t_w + h
                if future_idx < warmup_steps:
                    alpha_f = future_idx / max(1, warmup_steps - 1)
                    val = P_warmup_start + alpha_f * (P_warmup_end - P_warmup_start)
                else:
                    # Into the real profile
                    real_idx = future_idx - warmup_steps
                    if real_idx < len(full_profile):
                        val = full_profile[real_idx]
                    else:
                        val = full_profile[-1]
                P_ref_future.append(val)
            
            # --- NMPC Step ---
            current_state = sim.state.copy()
            
            # NMPC Call
            last_action = [I_prev, v_lye_prev, v_c_prev]
            try:
                u_opt_matrix, states_matrix = controller.get_all_actions_states(current_state, P_ref_future, T_ref, last_action)
            except Exception as e:
                print(f"NMPC failed: {e}")
                # Use previous action as fallback
                u_opt_matrix = np.tile(last_action, (horizon, 1))
                states_matrix = np.tile(current_state, (horizon, 1))

            u0 = u_opt_matrix[0]
            I_cmd = u0[0]
            v_lye_cmd = u0[1]
            v_c_cmd = u0[2]
            
            # Derived props
            _, U_cell, _ = sim._calculate_electrochemical_properties(I_cmd, current_state[1])
            P_real = I_cmd * U_cell * sim.N_cell
            
            V_sep_gas = 0.6 * 10.288
            p_sys = 1.6e6
            R = 8.314
            hto_val = (current_state[6] * R * current_state[2]) / (p_sys * V_sep_gas) * 100.0
            
            H2_rate_total = sim.N_cell * I_cmd / (2 * 96485.0)
            
            # Simulator Step
            sim_steps = int(dt_ctrl / sim_dt)
            action_sim = np.array([I_cmd, v_lye_cmd, v_c_cmd])
            
            for _ in range(sim_steps):
                sim.step(action_sim)
            
            # Log Data
            row = {
                'step': t_w - warmup_steps,
                'time': current_time,
                'P_ref': current_P_ref,
                'P_real': P_real,
                'T_s_in': current_state[0],
                'T_s': current_state[1],
                'T_sep': current_state[2],
                'T_c_out': current_state[3],
                'n_H2_an': current_state[4],
                'n_liq': current_state[5],
                'n_gas': current_state[6],
                'T_ref': T_ref,
                'I_prev': I_prev,
                'v_lye_prev': v_lye_prev,
                'v_c_prev': v_c_prev,
                'v_c': v_c_cmd,
                'v_lye': v_lye_cmd,
                'HTO': hto_val,
                'H2_rate': H2_rate_total,
                'I': I_cmd,
                'U_cell': U_cell
            }
            
            # Save Plan
            for k in range(horizon):
                row[f'P_ref_future_{k}'] = P_ref_future[k]
                u_k = u_opt_matrix[k]
                s_k = states_matrix[k]
                
                row[f'plan_step_{k}_I'] = u_k[0]
                row[f'plan_step_{k}_v_lye'] = u_k[1]
                row[f'plan_step_{k}_v_c'] = u_k[2]
                
                row[f'plan_step_{k}_state_T_s_in'] = s_k[0]
                row[f'plan_step_{k}_state_T_s'] = s_k[1]
                row[f'plan_step_{k}_state_T_sep'] = s_k[2]
                row[f'plan_step_{k}_state_T_c_out'] = s_k[3]
                row[f'plan_step_{k}_state_n_H2_an'] = s_k[4]
                row[f'plan_step_{k}_state_n_liq'] = s_k[5]
                row[f'plan_step_{k}_state_n_gas'] = s_k[6]

            warmup_data_list.append(row)
            warmup_plot_history.append({
                'time': current_time,
                'P_ref': current_P_ref,
                'P_real': P_real,
                'T_sep': current_state[2],
                'T_c_out': current_state[3],
                'T_ref': T_ref,
                'T_s': current_state[1],
                'I': I_cmd,
                'v_lye': v_lye_cmd,
                'U_cell': U_cell,
                'v_c': v_c_cmd,
                'HTO': hto_val,
                'H2_rate': H2_rate_total,
            })
            
            if (t_w + 1) % plot_every_steps == 0:
                plot_all(output_dir, timestamp, prefix="warmup", df=pd.DataFrame(warmup_plot_history))
            
            # Update history
            I_prev = I_cmd
            v_lye_prev = v_lye_cmd
            v_c_prev = v_c_cmd
            
        # Save Warmup Data
        save_batch(warmup_data_list, output_dir, timestamp, prefix="nmpc_dataset")
        plot_all(output_dir, timestamp, prefix="warmup", df=pd.DataFrame(warmup_plot_history))
        print("Warmup Phase Completed.\n")
        
        return I_prev, v_lye_prev, v_c_prev
        
    except Exception as e:
        print(f"Warmup failed: {e}")
        import traceback
        traceback.print_exc()
        return I_prev, v_lye_prev, v_c_prev

def generate_dataset():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Generate NMPC dataset for single-stack AWE.')
    parser.add_argument('--continue', dest='continue_gen', action='store_true', help='Continue from existing dataset if available')
    args = parser.parse_args()
    
    # Configuration
    # dt_ctrl = 1 min = 60 s
    dt_ctrl = 60.0
    
    # Output dir relative to script
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'output', 'single_stack', 'dataset'))
    
    # NMPC parameters
    horizon = 5
    sim_dt = 0.2 # 0.2s simulation step
    T_ref = 353.15  # 80°C (Kelvin)

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
    sim = SingleStackSimulator(sim_dt=sim_dt)
    sim.reset()
    
    # Initialize Controller with Control Interval
    try:
        controller = SingleStackNMPCController(dt=dt_ctrl, horizon=horizon, sim_dt=sim_dt)
    except Exception as e:
        print(f"Failed to initialize controller: {e}")
        return

    # Storage
    data_list = []
    start_step = 0
    save_every_steps = max(1, int(3600 / dt_ctrl)) # Save every hour
    
    # Check for existing data to resume
    csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv') and 'nmpc_dataset' in f]
    if csv_files and args.continue_gen:
        latest_csv_name = max(csv_files, key=lambda x: os.path.getctime(os.path.join(output_dir, x)))
        latest_csv_path = os.path.join(output_dir, latest_csv_name)
        
        print(f"Found existing dataset: {latest_csv_name}")
        try:
            # Read last few lines to get state (more efficient than reading whole file)
            # But for simplicity we read whole file for now as dataset size is manageable or we trust pandas
            # Optimization: Read header and last row only
            # However, to get start_step we need 'step' column.
            
            # Let's try to read just the tail
            # Using chunksize is one way, or just reading it all if memory permits (simpler for now)
            df_existing = pd.read_csv(latest_csv_path)
            
            if not df_existing.empty and 'step' in df_existing.columns:
                last_step = df_existing['step'].max()
                if last_step < total_steps - 1:
                    print(f"Resuming from step {last_step + 1}...")
                    start_step = int(last_step + 1)
                    timestamp = latest_csv_name.replace('nmpc_dataset_', '').replace('.csv', '')
                    
                    # Restore Simulator State from last row
                    last_row = df_existing.iloc[-1]
                    
                    # State vector: [T_s_in, T_s, T_sep, T_c_out, n_H2_an, n_liq, n_gas]
                    sim.state[0] = float(last_row['T_s_in'])
                    sim.state[1] = float(last_row['T_s'])
                    sim.state[2] = float(last_row['T_sep'])
                    sim.state[3] = float(last_row['T_c_out'])
                    # n_H2_an not always saved in summary, let's check what we saved.
                    # We didn't explicitly save n_H2_an in previous implementation, 
                    # but we can try to infer or set default if missing.
                    # Wait, the previous implementation did not save n_H2_an in the row dict!
                    # It saved n_liq and n_gas.
                    # We need to be careful. If missing, we might need to randomize or approximate.
                    # Actually, n_H2_an is state[4].
                    # Let's assume 0 if missing or small value.
                    if 'n_H2_an' in last_row:
                        sim.state[4] = float(last_row['n_H2_an'])
                    else:
                        sim.state[4] = 0.0 # Default/Small
                        
                    sim.state[5] = float(last_row['n_liq'])
                    sim.state[6] = float(last_row['n_gas'])
                    
                    # Restore Controller Last Action
                    controller.last_I = float(last_row['I'])
                    controller.last_v_lye = float(last_row['v_lye'])
                    controller.last_v_c = float(last_row['v_c'])
                    
                    # Restore Action History for logging
                    I_prev = controller.last_I
                    v_lye_prev = controller.last_v_lye
                    v_c_prev = controller.last_v_c
                    
                else:
                    print("Existing dataset appears complete or near end. Starting fresh.")
                    pass
        except Exception as e:
            print(f"Failed to resume from {latest_csv_path}: {e}. Starting fresh.")
    
    if start_step == 0:
        # Randomize Initial Condition ONCE if starting fresh
        randomize_state(sim)
        
        # Initialize previous actions for logging
        I_prev = 2000.0
        v_lye_prev = 0.3
        v_c_prev = 0.0

        # ---------------------------------------------------------
        # WARMUP PHASE
        # ---------------------------------------------------------
        I_prev, v_lye_prev, v_c_prev = run_warmup_phase(
            sim, controller, full_profile, dt_ctrl, horizon, sim_dt, T_ref,
            output_dir, timestamp, I_prev, v_lye_prev, v_c_prev
        )
    
    try:
        # Main Loop with Progress Bar
        for t in tqdm(range(start_step, total_steps), desc="Generating Dataset"):
            # Current State (At beginning of interval)
            current_state = sim.state.copy()
            
            T_s_in = current_state[0]
            T_s = current_state[1]
            T_sep = current_state[2]
            T_c_out = current_state[3]
            n_H2_an = current_state[4]
            n_liq = current_state[5]
            n_gas = current_state[6]
            
            # 1. Get P_ref
            current_P_ref = full_profile[t]
            
            # Future P_ref
            P_ref_future = []
            for h in range(horizon):
                future_idx = t + h
                val = full_profile[future_idx] if future_idx < len(full_profile) else full_profile[-1]
                P_ref_future.append(val)

            # 2. Determine Action via NMPC
            last_action = [I_prev, v_lye_prev, v_c_prev]
            try:
                u_opt_matrix, states_matrix = controller.get_all_actions_states(current_state, P_ref_future, T_ref, last_action)
            except Exception as e:
                print(f"NMPC failed at step {t}: {e}")
                import traceback
                traceback.print_exc()
                # Fallback: Tile last action
                u_opt_matrix = np.tile(last_action, (horizon, 1))
                states_matrix = np.tile(current_state, (horizon, 1))
                
            # Extract first action
            u0 = u_opt_matrix[0]
            I_cmd = u0[0]
            v_lye_cmd = u0[1]
            v_c_cmd = u0[2]
            
            # 3. Calculate derived properties
            _, U_cell, _ = sim._calculate_electrochemical_properties(I_cmd, T_s)
            P_real = I_cmd * U_cell * sim.N_cell
            
            hto_val = (n_gas * sim.R * T_sep) / (sim.P_sys * sim.V_sep_gas) * 100.0
            
            H2_rate_total = sim.N_cell * I_cmd / (2 * sim.F)
            
            # 4. Step Simulator
            sim_steps = int(dt_ctrl / sim_dt)
            action_sim = np.array([I_cmd, v_lye_cmd, v_c_cmd])
            
            for _ in range(sim_steps):
                sim.step(action_sim)
            
            # 5. Store Data
            row = {
                'step': t,
                'time': t * dt_ctrl,
                'P_ref': current_P_ref,
                'P_real': P_real,
                'T_s_in': T_s_in,
                'T_s': T_s,
                'T_sep': T_sep,
                'T_c_out': T_c_out,
                'n_H2_an': n_H2_an, # Added explicit save for n_H2_an
                'n_liq': n_liq,
                'n_gas': n_gas,
                'T_ref': T_ref,
                'I_prev': I_prev,
                'v_lye_prev': v_lye_prev,
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
                
                row[f'plan_step_{k}_I'] = u_k[0]
                row[f'plan_step_{k}_v_lye'] = u_k[1]
                row[f'plan_step_{k}_v_c'] = u_k[2]
                
                row[f'plan_step_{k}_state_T_s_in'] = s_k[0]
                row[f'plan_step_{k}_state_T_s'] = s_k[1]
                row[f'plan_step_{k}_state_T_sep'] = s_k[2]
                row[f'plan_step_{k}_state_T_c_out'] = s_k[3]
                row[f'plan_step_{k}_state_n_H2_an'] = s_k[4]
                row[f'plan_step_{k}_state_n_liq'] = s_k[5]
                row[f'plan_step_{k}_state_n_gas'] = s_k[6]
                
            data_list.append(row)
            
            # Update history
            I_prev = I_cmd
            v_lye_prev = v_lye_cmd
            v_c_prev = v_c_cmd
            
            # Periodic save
            if (t + 1) % save_every_steps == 0:
                save_batch(data_list, output_dir, timestamp)
                plot_all(output_dir, timestamp) # Optional: update plots periodically
                data_list = [] # Clear memory after save
                
    except KeyboardInterrupt:
        print("\nDataset generation interrupted by user.")
    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Final save
        save_batch(data_list, output_dir, timestamp)
        plot_all(output_dir, timestamp)

if __name__ == "__main__":
    generate_dataset()
