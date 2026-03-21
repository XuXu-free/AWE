"""
Generate step signal dataset for multi-stack AWE system.
Generates NMPC expert trajectories using random step power references and random initial conditions.
"""
"""
Generate step signal dataset for multi-stack AWE system.
Generates NMPC expert trajectories using random step power references and random initial conditions.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from plant.multi_stack_simulator import MultiStackSimulator
from controller.multi_stack.nmpc_controller import MultiStackNMPCController


def randomize_initial_condition(sim):
    """
    Randomize initial state for step experiment (multi-stack version).
    Returns initial actions suitable for the randomized state.
    """
    # 1. Randomize Temperature States (normal working range 60°C ~ 85°C)
    T_base = np.random.uniform(333.0, 358.0)  # 60°C ~ 85°C
    sim.state[0] = T_base - np.random.uniform(0, 5)  # T_s_in slightly cooler
    # T_s_1..4 with small variations
    for i in range(4):
        sim.state[1+i] = T_base + np.random.uniform(-2, 2)
    sim.state[5] = T_base - np.random.uniform(2, 10)  # T_sep (cooler)
    sim.state[6] = np.random.uniform(293.0, 340.0)  # T_c_out

    # 2. Randomize Hydrogen amounts for all 4 stacks
    # n_H2_an_1..4: start small (0 ~ 2 mol)
    for i in range(4):
        sim.state[7+i] = np.random.uniform(0.0, 2.0)
    # n_liq: start small (0 ~ 2 mol)
    sim.state[11] = np.random.uniform(0.0, 2.0)
    # n_gas: common range (15 ~ 40 mol)
    sim.state[12] = np.random.uniform(15.0, 40.0)

    # 3. Randomize initial actions for all 4 stacks (conservative choices)
    I_init = np.random.uniform(1000, 4000, size=4)  # Current for each stack
    v_lye_init = np.random.uniform(0.02, 0.06, size=4)  # Lye flow for each stack
    v_c_init = np.random.uniform(0.1, 0.4)  # Coolant flow

    return I_init, v_lye_init, v_c_init


def smooth_power_profile(P_ref_array, window=3):
    """Apply moving average smoothing to power profile."""
    smoothed = np.copy(P_ref_array)
    if len(P_ref_array) < window:
        return smoothed
    half_window = window // 2
    for i in range(half_window, len(P_ref_array) - half_window):
        smoothed[i] = np.mean(P_ref_array[i-half_window:i+half_window+1])
    return smoothed


def solve_nmpc_with_retry(controller, state, P_ref_future, T_ref, last_action, max_retries=3):
    """
    Attempt to solve NMPC with retry logic on failure.
    Returns: (u_opt_matrix, states_matrix, success)
    """
    I_prev, v_lye_prev, v_c_prev = last_action
    n_stacks = controller.n_stacks

    for attempt in range(max_retries):
        try:
            u_opt_matrix, states_matrix = controller.get_all_actions_states(
                state, P_ref_future, T_ref, last_action
            )
            if u_opt_matrix is not None and states_matrix is not None:
                return u_opt_matrix, states_matrix, True
        except Exception as e:
            print(f"  NMPC attempt {attempt + 1} failed: {e}")

        # Adjustment strategies
        if attempt == 0:
            # Try smoothing the power profile
            P_ref_future = smooth_power_profile(np.array(P_ref_future)).tolist()
            print("  Retrying with smoothed power profile...")
        elif attempt == 1:
            # Try more conservative initial action
            last_action = [
                I_prev * 0.8,  # Reduce current
                v_lye_prev * 1.1,  # Increase lye flow
                v_c_prev * 1.1  # Increase cooling
            ]
            print("  Retrying with conservative initial action...")
        else:
            # Final attempt: slight state perturbation (handled by caller)
            print("  Final retry failed...")

    return None, None, False


def generate_step_profile(n_steps, power_levels=None, min_duration=10, max_duration=60, noise_std=0.05):
    """
    Generate a step power profile.

    Args:
        n_steps: Total number of steps
        power_levels: List of power levels in Watts (default: 4-20 MW)
        min_duration: Minimum steps per step level
        max_duration: Maximum steps per step level
        noise_std: Relative noise to add (fraction of step magnitude)
    """
    if power_levels is None:
        # Multi-stack: 4-20 MW
        power_levels = [4.0e6, 6.0e6, 8.0e6, 10.0e6, 12.0e6, 14.0e6, 16.0e6, 18.0e6, 20.0e6]

    profile = np.zeros(n_steps)

    current_idx = 0
    last_power = np.random.choice(power_levels)

    while current_idx < n_steps:
        # Random step duration
        step_duration = np.random.randint(min_duration, max_duration + 1)
        end_idx = min(current_idx + step_duration, n_steps)

        # Choose new power level (different from last if possible)
        if len(power_levels) > 1:
            available_powers = [p for p in power_levels if p != last_power]
            power = np.random.choice(available_powers)
        else:
            power = power_levels[0]

        # Add small transition region at the start of the step change
        if current_idx > 0 and current_idx < n_steps:
            # 2-step transition
            transition_steps = min(2, end_idx - current_idx)
            for t in range(transition_steps):
                alpha = t / transition_steps
                profile[current_idx + t] = last_power + alpha * (power - last_power)
            # Fill rest with constant power
            profile[current_idx + transition_steps:end_idx] = power
        else:
            profile[current_idx:end_idx] = power

        last_power = power
        current_idx = end_idx

    # Add small noise for natural variation
    if noise_std > 0:
        noise = np.random.normal(0, noise_std * np.mean(power_levels), n_steps)
        profile = profile + noise
        # Clip to valid range
        profile = np.clip(profile, min(power_levels) * 0.9, max(power_levels) * 1.1)

    return profile


def save_experiment_data(data_list, output_dir, timestamp, exp_id):
    """Save experiment data to CSV file."""
    if not data_list:
        return None

    df = pd.DataFrame(data_list)

    # Filename format: nmpc_dataset_step_{exp_id:03d}_{timestamp}.csv
    csv_file = os.path.join(output_dir, f"nmpc_dataset_step_{exp_id:03d}_{timestamp}.csv")
    df.to_csv(csv_file, index=False)
    print(f"Saved experiment {exp_id} to: {csv_file}")
    return csv_file


def plot_experiment(df, output_dir, timestamp, exp_id):
    """Generate 3x3 grid plot for experiment data (multi-stack version)."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    if 'time' in df.columns:
        t_arr = df['time'].values
    else:
        t_arr = df.index.values

    # 1. Power Tracking
    ax = axes[0, 0]
    if 'P_ref' in df.columns and 'P_real' in df.columns:
        ax.plot(t_arr / 60, df['P_ref'] / 1e6, 'k--', label='Ref')
        ax.plot(t_arr / 60, df['P_real'] / 1e6, 'b-', label='Real')
        ax.set_title('Power Tracking')
        ax.set_ylabel('MW')
        ax.set_xlabel('Time (min)')
        ax.legend()
    ax.grid(True)

    # 2. Temperatures (all 4 stacks)
    ax = axes[0, 1]
    if 'T_sep' in df.columns:
        ax.plot(t_arr / 60, df['T_sep'], 'g--', label='Separator')
    if 'T_c_out' in df.columns:
        ax.plot(t_arr / 60, df['T_c_out'], 'c:', label='CW Out')
    colors = ['r', 'orange', 'brown', 'pink']
    for i in range(4):
        col = f'T_s_{i+1}'
        if col in df.columns:
            ax.plot(t_arr / 60, df[col], color=colors[i], label=f'Stack {i+1}')
    if 'T_ref' in df.columns:
        ax.plot(t_arr / 60, df['T_ref'], 'k--', linewidth=1.5, label='Ref Temp')
    ax.set_title('All Temperatures')
    ax.set_ylabel('K')
    ax.set_xlabel('Time (min)')
    ax.legend()
    ax.grid(True)

    # 3. HTO
    ax = axes[0, 2]
    if 'HTO' in df.columns:
        ax.plot(t_arr / 60, df['HTO'], 'm-', label='HTO')
        ax.axhline(y=2.0, color='r', linestyle='--', label='Limit')
    ax.set_title('HTO (H2 in O2)')
    ax.set_ylabel('%')
    ax.set_xlabel('Time (min)')
    ax.legend()
    ax.grid(True)

    # 4. Currents (4 stacks)
    ax = axes[1, 0]
    colors = ['b', 'g', 'r', 'm']
    for i in range(4):
        col = f'I_{i+1}'
        if col in df.columns:
            ax.plot(t_arr / 60, df[col], color=colors[i], label=f'Stack {i+1}')
    ax.set_title('Stack Currents')
    ax.set_ylabel('Amps')
    ax.set_xlabel('Time (min)')
    ax.legend()
    ax.grid(True)

    # 5. Voltages (4 stacks)
    ax = axes[1, 1]
    colors = ['b', 'g', 'r', 'm']
    for i in range(4):
        col = f'U_cell_{i+1}'
        if col in df.columns:
            ax.plot(t_arr / 60, df[col], color=colors[i], label=f'Stack {i+1}')
    ax.axhline(y=2.2, color='r', linestyle='--', label='Limit')
    ax.set_title('Cell Voltages')
    ax.set_ylabel('Volts')
    ax.set_xlabel('Time (min)')
    ax.legend()
    ax.grid(True)

    # 6. Lye Flows (4 stacks)
    ax = axes[1, 2]
    colors = ['b', 'g', 'r', 'm']
    for i in range(4):
        col = f'v_lye_{i+1}'
        if col in df.columns:
            ax.plot(t_arr / 60, df[col] * 1000, color=colors[i], label=f'Stack {i+1}')  # Convert to L/s
    ax.set_title('Stack Lye Flows')
    ax.set_ylabel('L/s')
    ax.set_xlabel('Time (min)')
    ax.legend()
    ax.grid(True)

    # 7. Coolant Flow
    ax = axes[2, 0]
    if 'v_c' in df.columns:
        ax.plot(t_arr / 60, df['v_c'], 'cyan', label='CW')
    ax.set_title('Coolant Flow')
    ax.set_ylabel('m³/s')
    ax.set_xlabel('Time (min)')
    ax.legend()
    ax.grid(True)

    # 8. H2 Rate
    ax = axes[2, 1]
    if 'H2_rate' in df.columns:
        ax.plot(t_arr / 60, df['H2_rate'], 'g-', label='H2 Rate')
    ax.set_title('H2 Production Rate')
    ax.set_ylabel('mol/s')
    ax.set_xlabel('Time (min)')
    ax.legend()
    ax.grid(True)

    # 9. Hydrogen States
    ax = axes[2, 2]
    colors_n = ['b', 'g', 'r', 'm']
    for i in range(4):
        col = f'n_H2_an_{i+1}'
        if col in df.columns:
            ax.plot(t_arr / 60, df[col], color=colors_n[i], label=f'n_H2_an_{i+1}')
    if 'n_liq' in df.columns:
        ax.plot(t_arr / 60, df['n_liq'], 'c-', label='n_liq')
    if 'n_gas' in df.columns:
        ax.plot(t_arr / 60, df['n_gas'], 'k-', label='n_gas')
    ax.set_title('H2 States')
    ax.set_ylabel('mol')
    ax.set_xlabel('Time (min)')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plot_file = os.path.join(output_dir, f"nmpc_dataset_step_{exp_id:03d}_plot_{timestamp}.png")
    plt.savefig(plot_file, dpi=150)
    print(f"Plot saved to: {plot_file}")
    plt.close()


def run_step_experiment(args_dict):
    """
    Run a single step experiment and save results immediately.
    Args:
        args_dict: Dictionary containing all parameters
    Returns True if successful, False otherwise.
    """
    exp_id = args_dict['exp_id']
    timestamp = args_dict['timestamp']
    n_steps = args_dict['n_steps']
    dt_ctrl = args_dict['dt_ctrl']
    horizon = args_dict['horizon']
    sim_dt = args_dict['sim_dt']
    T_ref = args_dict['T_ref']
    output_dir = args_dict['output_dir']
    power_levels = args_dict['power_levels']
    min_duration = args_dict['min_duration']
    max_duration = args_dict['max_duration']
    max_retries_on_failure = args_dict.get('max_retries', 2)
    skip_plots = args_dict.get('skip_plots', False)

    # Initialize simulator
    sim = MultiStackSimulator(dt=sim_dt)
    sim.reset()

    # Randomize initial condition (NO warmup - start from random state directly)
    I_prev, v_lye_prev, v_c_prev = randomize_initial_condition(sim)

    # Initialize controller
    try:
        controller = MultiStackNMPCController(dt=dt_ctrl, horizon=horizon, dt_sub=sim_dt)
        n_stacks = controller.n_stacks
    except Exception as e:
        print(f"  Failed to initialize controller: {e}")
        return None

    # Generate step profile
    step_profile = generate_step_profile(
        n_steps + horizon,  # Extra for horizon lookahead
        power_levels=power_levels,
        min_duration=min_duration,
        max_duration=max_duration
    )

    # Main experiment loop
    data_list = []

    for t in tqdm(range(n_steps), desc=f"Exp {exp_id}", leave=False):
        # Current state
        current_state = sim.state.copy()

        # Unpack state (13-dim)
        T_s_in = current_state[0]
        T_s_vec = current_state[1:5]  # T_s_1..4
        T_sep = current_state[5]
        T_c_out = current_state[6]
        n_H2_an_vec = current_state[7:11]  # n_H2_an_1..4
        n_liq = current_state[11]
        n_gas = current_state[12]

        # Current and future power references
        current_P_ref = step_profile[t]
        P_ref_future = step_profile[t:t+horizon].tolist()

        # Pad if necessary
        while len(P_ref_future) < horizon:
            P_ref_future.append(step_profile[-1])

        # NMPC solve with retry
        last_action = [I_prev, v_lye_prev, v_c_prev]
        u_opt_matrix, states_matrix, success = solve_nmpc_with_retry(
            controller, current_state, P_ref_future, T_ref, last_action
        )

        if not success:
            print(f"  Experiment {exp_id} failed at step {t}, discarding...")
            return None

        # Extract first action
        u0 = u_opt_matrix[0]
        I_cmd = u0[0:n_stacks]
        v_lye_cmd = u0[n_stacks:2*n_stacks]
        v_c_cmd = u0[2*n_stacks]

        # Calculate derived properties
        _, U_cell_vec, _ = sim._calculate_electrochemical_properties(I_cmd, T_s_vec)
        P_real = np.sum(I_cmd * U_cell_vec * sim.N_cell)

        # HTO calculation
        hto_val = (n_gas * sim.R * T_sep) / (sim.P_sys * sim.V_sep_gas) * 100.0

        # H2 production rate
        H2_rate_vec = sim.N_cell * I_cmd / (2 * sim.F)
        H2_rate_total = np.sum(H2_rate_vec)

        # Step simulator
        sim_steps = int(dt_ctrl / sim_dt)
        action_sim = np.concatenate([I_cmd, v_lye_cmd, [v_c_cmd]])

        for _ in range(sim_steps):
            sim.step(action_sim)

        # Build data row
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
            # Stack temperatures
            'T_s_1': T_s_vec[0], 'T_s_2': T_s_vec[1], 'T_s_3': T_s_vec[2], 'T_s_4': T_s_vec[3],
            # H2 anode amounts
            'n_H2_an_1': n_H2_an_vec[0], 'n_H2_an_2': n_H2_an_vec[1],
            'n_H2_an_3': n_H2_an_vec[2], 'n_H2_an_4': n_H2_an_vec[3],
            # Previous actions
            'I_prev_1': I_prev[0], 'I_prev_2': I_prev[1], 'I_prev_3': I_prev[2], 'I_prev_4': I_prev[3],
            'v_lye_prev_1': v_lye_prev[0], 'v_lye_prev_2': v_lye_prev[1],
            'v_lye_prev_3': v_lye_prev[2], 'v_lye_prev_4': v_lye_prev[3],
            # Current actions
            'I_1': I_cmd[0], 'I_2': I_cmd[1], 'I_3': I_cmd[2], 'I_4': I_cmd[3],
            'v_lye_1': v_lye_cmd[0], 'v_lye_2': v_lye_cmd[1],
            'v_lye_3': v_lye_cmd[2], 'v_lye_4': v_lye_cmd[3],
            # Cell voltages
            'U_cell_1': U_cell_vec[0], 'U_cell_2': U_cell_vec[1],
            'U_cell_3': U_cell_vec[2], 'U_cell_4': U_cell_vec[3]
        }

        # Save NMPC plan
        for k in range(horizon):
            row[f'P_ref_future_{k}'] = P_ref_future[k]
            u_k = u_opt_matrix[k]
            s_k = states_matrix[k]

            # Actions: I_1..4, v_lye_1..4, v_c
            for i in range(n_stacks):
                row[f'plan_step_{k}_I_{i+1}'] = u_k[i]
            for i in range(n_stacks):
                row[f'plan_step_{k}_v_lye_{i+1}'] = u_k[n_stacks+i]
            row[f'plan_step_{k}_v_c'] = u_k[2*n_stacks]

            # States: [T_s_in, T_s_1..4, T_sep, T_c_out, n_H2_an_1..4, n_liq, n_gas]
            row[f'plan_step_{k}_state_T_s_in'] = s_k[0]
            for i in range(n_stacks):
                row[f'plan_step_{k}_state_T_s_{i+1}'] = s_k[1+i]
            row[f'plan_step_{k}_state_T_sep'] = s_k[5]
            row[f'plan_step_{k}_state_T_c_out'] = s_k[6]
            for i in range(n_stacks):
                row[f'plan_step_{k}_state_n_H2_an_{i+1}'] = s_k[7+i]
            row[f'plan_step_{k}_state_n_liq'] = s_k[11]
            row[f'plan_step_{k}_state_n_gas'] = s_k[12]

        data_list.append(row)

        # Update previous actions
        I_prev = I_cmd
        v_lye_prev = v_lye_cmd
        v_c_prev = v_c_cmd

    # Save experiment data immediately after completion
    try:
        df = pd.DataFrame(data_list)
        csv_file = os.path.join(output_dir, f"nmpc_dataset_step_{exp_id:03d}_{timestamp}.csv")
        df.to_csv(csv_file, index=False)
        print(f"[Exp {exp_id:03d}] Saved data to: {csv_file}")

        # Generate plot if not skipped
        if not skip_plots:
            try:
                plot_experiment(df, output_dir, timestamp, exp_id)
                print(f"[Exp {exp_id:03d}] Generated plot")
            except Exception as e:
                print(f"[Exp {exp_id:03d}] Plot generation failed: {e}")

        return True
    except Exception as e:
        print(f"[Exp {exp_id:03d}] Failed to save data: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Generate step signal NMPC dataset for multi-stack AWE.')
    parser.add_argument('--n_experiments', type=int, default=50,
                        help='Number of step experiments to generate (default: 50)')
    parser.add_argument('--n_steps', type=int, default=240,
                        help='Steps per experiment (default: 240 = 4 hours at 60s dt)')
    parser.add_argument('--dt_ctrl', type=float, default=60.0,
                        help='Control period in seconds (default: 60)')
    parser.add_argument('--horizon', type=int, default=5,
                        help='NMPC prediction horizon (default: 5)')
    parser.add_argument('--sim_dt', type=float, default=0.2,
                        help='Simulation time step in seconds (default: 0.2)')
    parser.add_argument('--T_ref', type=float, default=353.15,
                        help='Target temperature in Kelvin (default: 353.15 = 80°C)')
    parser.add_argument('--step_min_duration', type=int, default=10,
                        help='Minimum steps per power level (default: 10)')
    parser.add_argument('--step_max_duration', type=int, default=60,
                        help='Maximum steps per power level (default: 60)')
    parser.add_argument('--power_levels', type=str,
                        default='4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0',
                        help='Comma-separated power levels in MW (default: 4-20 MW, 1MW intervals)')
    parser.add_argument('--max_retries', type=int, default=2,
                        help='Max retries per experiment on failure (default: 2)')
    parser.add_argument('--skip_plots', action='store_true',
                        help='Skip plot generation (default: generate plots)')
    parser.add_argument('--max_workers', type=int, default=4,
                        help='并行worker数量 (默认: 4, 建议不超过4-6，每个worker需要大量内存)')
    args = parser.parse_args()

    # Parse power levels
    power_levels = [float(x) * 1e6 for x in args.power_levels.split(',')]

    # Output directory
    output_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..',
        'output', 'multi_stack', 'dataset', 'step'
    ))
    os.makedirs(output_dir, exist_ok=True)

    # Generate unified timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine max workers
    if args.max_workers <= 0:
        cpu_count = multiprocessing.cpu_count()
        # Limit to 4 or half of CPUs, whichever is smaller, to avoid memory issues
        max_workers = max(1, min(4, cpu_count // 2))
        print(f"Auto-selected {max_workers} workers (limited to avoid memory issues)")
    else:
        max_workers = args.max_workers

    print("=" * 60)
    print("Step Signal Dataset Generation for Multi-Stack AWE")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Experiments: {args.n_experiments}")
    print(f"  Steps per experiment: {args.n_steps} ({args.n_steps * args.dt_ctrl / 3600:.1f} hours)")
    print(f"  Control period: {args.dt_ctrl}s")
    print(f"  NMPC horizon: {args.horizon}")
    print(f"  Power levels: {[f'{p/1e6:.1f}MW' for p in power_levels]}")
    print(f"  Step duration: {args.step_min_duration}-{args.step_max_duration} steps")
    print(f"  Parallel workers: {max_workers}")
    print(f"  Output directory: {output_dir}")
    print(f"  Timestamp: {timestamp}")
    print("=" * 60)

    # Prepare task arguments
    task_args = []
    for exp_id in range(args.n_experiments):
        task_args.append({
            'exp_id': exp_id,
            'timestamp': timestamp,
            'n_steps': args.n_steps,
            'dt_ctrl': args.dt_ctrl,
            'horizon': args.horizon,
            'sim_dt': args.sim_dt,
            'T_ref': args.T_ref,
            'output_dir': output_dir,
            'power_levels': power_levels,
            'min_duration': args.step_min_duration,
            'max_duration': args.step_max_duration,
            'max_retries': args.max_retries,
            'skip_plots': args.skip_plots
        })

    # Run experiments in parallel
    successful_experiments = 0
    failed_experiments = 0

    if max_workers == 1:
        # Sequential execution for debugging
        for args_dict in task_args:
            if run_step_experiment(args_dict):
                successful_experiments += 1
            else:
                failed_experiments += 1
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_exp = {
                executor.submit(run_step_experiment, args_dict): args_dict['exp_id']
                for args_dict in task_args
            }

            # Collect results as they complete
            for future in as_completed(future_to_exp):
                exp_id = future_to_exp[future]
                try:
                    result = future.result()
                    if result:
                        successful_experiments += 1
                        print(f"[Exp {exp_id:03d}] Completed and saved successfully")
                    else:
                        failed_experiments += 1
                        print(f"[Exp {exp_id:03d}] Failed")
                except Exception as e:
                    print(f"[Exp {exp_id:03d}] Exception: {e}")
                    failed_experiments += 1

    print("\n" + "=" * 60)
    print("Generation Complete")
    print("=" * 60)
    print(f"Successful experiments: {successful_experiments}/{args.n_experiments}")
    print(f"Failed experiments: {failed_experiments}/{args.n_experiments}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
