
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import multiprocessing
import time

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from plant.multi_stack_simulator import MultiStackSimulator
from controller.multi_stack.multi_stack_nmpc_controller import MultiStackNMPCController

def generate_profile(length, mode, p_min, p_max):
    trajectory = np.zeros(length)
    if mode == 'solar':
        # Sine wave pattern with noise (Day/Night cycle)
        period = np.random.randint(400, 800)
        phase = np.random.uniform(0, 2*np.pi)
        t = np.arange(length)
        
        # Base solar curve: max(0, sin)
        base = np.sin(2 * np.pi * t / period + phase)
        base = np.maximum(base, 0)
        
        # Scale to power
        amplitude = np.random.uniform(0.5, 1.0) * (p_max - p_min)
        trajectory = p_min + base * amplitude
        
        # Add cloud noise (random drops)
        cloud_mask = np.random.choice([0, 1], size=length, p=[0.9, 0.1])
        cloud_drop = np.random.uniform(0.3, 0.8, size=length)
        trajectory = trajectory * (1 - cloud_mask * cloud_drop)
        
    elif mode == 'wind':
        # Smoothed Random Walk / Perlin-like
        current = np.random.uniform(p_min, p_max)
        for i in range(length):
            trajectory[i] = current
            # Change
            delta = np.random.normal(0, (p_max - p_min) * 0.05)
            current += delta
            # Tendency to return to mean
            mean_p = (p_min + p_max) / 2
            current += (mean_p - current) * 0.01
            current = np.clip(current, p_min, p_max)
            
    elif mode == 'step':
        # Step changes
        current = np.random.uniform(p_min, p_max)
        steps_to_change = np.random.randint(50, 200)
        for i in range(length):
            if steps_to_change <= 0:
                current = np.random.uniform(p_min, p_max)
                steps_to_change = np.random.randint(50, 200)
            trajectory[i] = current
            steps_to_change -= 1
            
    else: # Mixed / Random Walk (Original)
            # Smooth random walk
            current = np.random.uniform(p_min, p_max)
            target = np.random.uniform(p_min, p_max)
            steps = 0
            for i in range(length):
                if steps <= 0:
                    target = np.random.uniform(p_min, p_max)
                    steps = np.random.randint(50, 200)
                
                move = (target - current) / steps
                current += move + np.random.normal(0, (p_max - p_min)*0.005)
                current = np.clip(current, p_min, p_max)
                trajectory[i] = current
                steps -= 1

    # Add small measurement noise
    trajectory += np.random.normal(0, (p_max - p_min)*0.01, size=length)
    return np.clip(trajectory, p_min, p_max)

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

def generate_single_episode(episode_args):
    episode_idx, steps_per_episode, dt, horizon, P_ref_min, P_ref_max, T_ref = episode_args
    
    # Re-seed random number generator for each process
    np.random.seed(int(time.time() * 1000) % 2**32 + episode_idx)
    
    # Stagger start to avoid simultaneous heavy memory allocation
    time.sleep(np.random.uniform(0.1, 2.0))
    
    # Initialize Simulator
    sim = MultiStackSimulator(dt=dt)
    
    # Initialize Controller
    try:
        controller = MultiStackNMPCController(dt=dt, horizon=horizon)
    except Exception as e:
        print(f"Failed to initialize controller for episode {episode_idx}: {e}")
        return [], np.zeros(steps_per_episode)

    # Storage
    data_list = []
    
    # 1. Select Profile
    profile_type = np.random.choice(['solar', 'wind', 'step', 'mixed'])
    # Generate trajectory with extra horizon
    traj_len = steps_per_episode + horizon + 10
    P_ref_trajectory = generate_profile(traj_len, profile_type, P_ref_min, P_ref_max)
    
    # 2. Randomize Initial Condition
    randomize_state(sim)
    
    # Reset Controller State
    controller.prev_sol_x = None
    controller.last_I = np.zeros(4)
    controller.last_v_lye = np.ones(4) * 0.03
    controller.last_v_c = 0.0
    
    # Reset Action History for logging
    I_prev = np.zeros(4)
    v_lye_prev = np.zeros(4)
    v_c_prev = 0.0
    
    # 3. Episode Loop
    for t in range(steps_per_episode):
        # Current State (Before Step)
        current_state = sim.state.copy()
        
        T_s_in = current_state[0]
        T_s_vec = current_state[1:5]
        T_sep = current_state[5]
        T_c_out = current_state[6]
        n_H2_an_vec = current_state[7:11]
        n_liq = current_state[11]
        n_gas = current_state[12]
        
        # 1. Get P_ref
        current_P_ref = P_ref_trajectory[t]
        
        # Future P_ref
        P_ref_future = []
        for h in range(horizon):
            future_val = P_ref_trajectory[t + 1 + h]
            # Add forecast noise (NMPC sees noisy forecast)
            noise_std = 0.02e6 * (h + 1)
            pred_val = future_val + np.random.normal(0, noise_std)
            P_ref_future.append(np.clip(pred_val, P_ref_min, P_ref_max))

        # 2. Determine Action via NMPC
        try:
            u_opt_matrix = controller.get_all_actions(current_state, P_ref_future, T_ref)
        except Exception as e:
            print(f"NMPC failed at episode {episode_idx} step {t}: {e}")
            break
            
        # Extract first action for execution
        u0 = u_opt_matrix[0]
        I_cmd = u0[0:4]
        v_lye_cmd = u0[4:8]
        v_c_cmd = u0[8]
        
        # 3. Calculate derived properties
        _, U_cell_vec, _ = sim._calculate_electrochemical_properties(I_cmd, T_s_vec)
        P_real = np.sum(I_cmd * U_cell_vec * sim.N_cell)
        
        V_sep_gas = 2.0
        p_sys = 1.6e6
        R = 8.314
        hto_val = (n_gas * R * T_sep) / (p_sys * V_sep_gas) * 100.0
        
        H2_rate_vec = sim.N_cell * I_cmd / (2 * 96485.0)
        H2_rate_total = np.sum(H2_rate_vec)
        
        # 4. Step Simulator
        sim.step(np.concatenate([I_cmd, v_lye_cmd, [v_c_cmd]]))
        
        # 5. Store Data
        # Note: 't' here is relative to episode. We will adjust it in aggregation.
        row = {
            'episode': episode_idx,
            'step': t,
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
            for i in range(4):
                row[f'plan_step_{k}_I_{i+1}'] = u_k[i]
            for i in range(4):
                row[f'plan_step_{k}_v_lye_{i+1}'] = u_k[4+i]
            row[f'plan_step_{k}_v_c'] = u_k[8]
            
        data_list.append(row)
        
        # Update history
        I_prev = I_cmd
        v_lye_prev = v_lye_cmd
        v_c_prev = v_c_cmd
        
    print(f"Episode {episode_idx+1} ({profile_type}) done.")
    return data_list, P_ref_trajectory[:steps_per_episode]

def generate_dataset():
    # Configuration
    num_episodes = 50
    steps_per_episode = 2000 
    dt = 1.0
    output_dir = r"d:\Projects\AWE\output\multi_stack"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"nmpc_dataset_{timestamp}.csv")
    
    # Constraints
    P_ref_min, P_ref_max = 2.0e6, 40.0e6
    T_ref = 358.15
    horizon = 10
    
    print(f"Generating {num_episodes} episodes of {steps_per_episode} steps each using NMPC in PARALLEL...")
    
    # Prepare arguments
    tasks = []
    for i in range(num_episodes):
        tasks.append((i, steps_per_episode, dt, horizon, P_ref_min, P_ref_max, T_ref))
        
    # Parallel Execution
    # Determine CPU count
    num_cores = multiprocessing.cpu_count()
    # Limit to a safe number of workers to avoid memory exhaustion (std::bad_alloc)
    # CasADi/IPOPT instances are heavy.
    num_workers = min(4, max(1, num_cores - 2)) 
    print(f"Using {num_workers} worker processes.")
    
    all_data = []
    all_P_ref_trajectories = []
    
    # Use multiprocessing.Pool
    # Note: On Windows, this must be protected by if __name__ == "__main__"
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(generate_single_episode, tasks)
        
    # Aggregation
    print("Aggregating results...")
    total_step_counter = 0
    
    for episode_data, trajectory in results:
        all_P_ref_trajectories.append(trajectory)
        for row in episode_data:
            row['t'] = total_step_counter * dt
            total_step_counter += 1
        all_data.extend(episode_data)
        
    # Save to CSV
    df = pd.DataFrame(all_data)
    df.to_csv(output_file, index=False)
    print(f"Dataset generated at: {output_file}")
    
    # Plot distributions
    print("Generating distribution plots...")
    plt.figure(figsize=(20, 15))
    
    # Define variables to plot
    plot_vars = [
        ('P_real', 'Power (W)'),
        ('T_sep', 'Separator Temp (K)'),
        ('n_gas', 'Gas Amount (mol)'),
        ('HTO', 'HTO (%)'),
        ('H2_rate', 'H2 Production Rate (mol/s)'),
        ('v_c', 'Cooling Valve')
    ]
    
    for i, (col, label) in enumerate(plot_vars):
        plt.subplot(3, 3, i+1)
        if col in df.columns:
            plt.hist(df[col], bins=50, alpha=0.7, color='blue', edgecolor='black')
            plt.title(f'Distribution of {label}')
            plt.xlabel(label)
            plt.ylabel('Count')
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, f'{col} not found', ha='center')
            
    plt.tight_layout()
    plot_file = os.path.join(output_dir, f"nmpc_dataset_dist_{timestamp}.png")
    plt.savefig(plot_file)
    print(f"Distribution plots saved to: {plot_file}")
    plt.close()

    # Plot P_ref trajectories collection
    print("Generating P_ref trajectory collection plot...")
    plt.figure(figsize=(12, 6))
    
    traj_array = np.array(all_P_ref_trajectories)
    
    # Plot individual trajectories
    for i in range(len(all_P_ref_trajectories)):
        plt.plot(all_P_ref_trajectories[i]/1e6, color='blue', alpha=0.1)
        
    mean_traj = np.mean(traj_array, axis=0)
    plt.plot(mean_traj/1e6, 'r--', linewidth=2, label='Mean Profile')
    
    plt.title(f'Generated P_ref Trajectories (N={num_episodes})')
    plt.xlabel('Step')
    plt.ylabel('Power Reference (MW)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    pref_plot_file = os.path.join(output_dir, f"nmpc_dataset_pref_collection_{timestamp}.png")
    plt.savefig(pref_plot_file)
    print(f"P_ref collection plot saved to: {pref_plot_file}")
    plt.close()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    generate_dataset()
