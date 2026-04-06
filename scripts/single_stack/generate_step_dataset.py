"""
Generate step signal dataset for single-stack AWE system.
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
from joblib import Parallel, delayed

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from plant.single_stack_simulator import SingleStackSimulator
from controller.single_stack.nmpc_controller import SingleStackNMPCController


def randomize_initial_condition(sim):
    """
    Randomize initial state for step experiment.
    Returns initial actions suitable for the randomized state.
    """
    # 1. Randomize Temperature States (normal working range 60°C ~ 85°C)
    T_base = np.random.uniform(333.0, 358.0)  # 60°C ~ 85°C
    sim.state[0] = T_base - np.random.uniform(0, 5)    # T_s_in
    sim.state[1] = T_base                              # T_s
    sim.state[2] = T_base - np.random.uniform(2, 10)   # T_sep (cooler)
    sim.state[3] = np.random.uniform(293.0, 340.0)     # T_c_out

    # 2. Randomize Hydrogen amounts (constrained to common ranges to avoid HTO violations)
    # n_H2_an: start small (0 ~ 2 mol)
    sim.state[4] = np.random.uniform(0.0, 2.0)
    # n_liq: start small (0 ~ 2 mol)
    sim.state[5] = np.random.uniform(0.0, 2.0)
    # n_gas: common range (15 ~ 40 mol, corresponds to normal HTO levels)
    sim.state[6] = np.random.uniform(15.0, 40.0)

    # 3. Randomize initial actions (conservative choices for stability)
    I_init = np.random.uniform(1000, 4000)       # Current: not too high
    v_lye_init = np.random.uniform(0.02, 0.06)   # Medium lye flow
    v_c_init = np.random.uniform(0.1, 0.4)       # Medium cooling flow

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
                last_action[0] * 0.8,
                last_action[1] * 1.1,  # Increase lye flow
                last_action[2] * 1.1   # Increase cooling
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
        power_levels: List of power levels in Watts (default: 1-5 MW)
        min_duration: Minimum steps per step level
        max_duration: Maximum steps per step level
        noise_std: Relative noise to add (fraction of step magnitude)
    """
    if power_levels is None:
        power_levels = [1.0e6, 2.0e6, 3.0e6, 4.0e6, 5.0e6]

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
    """Generate 3x3 grid plot for experiment data."""
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

    # 2. Temperatures
    ax = axes[0, 1]
    if 'T_sep' in df.columns:
        ax.plot(t_arr / 60, df['T_sep'], 'g--', label='Separator')
    if 'T_c_out' in df.columns:
        ax.plot(t_arr / 60, df['T_c_out'], 'c:', label='CW Out')
    if 'T_s' in df.columns:
        ax.plot(t_arr / 60, df['T_s'], 'r-', label='Stack')
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

    # 4. Current
    ax = axes[1, 0]
    if 'I' in df.columns:
        ax.plot(t_arr / 60, df['I'], 'b-', label='Stack')
    ax.set_title('Stack Current')
    ax.set_ylabel('Amps')
    ax.set_xlabel('Time (min)')
    ax.legend()
    ax.grid(True)

    # 5. Voltage
    ax = axes[1, 1]
    if 'U_cell' in df.columns:
        ax.plot(t_arr / 60, df['U_cell'], 'b-', label='Stack')
        ax.axhline(y=2.2, color='r', linestyle='--', label='Limit')
    ax.set_title('Cell Voltage')
    ax.set_ylabel('Volts')
    ax.set_xlabel('Time (min)')
    ax.legend()
    ax.grid(True)

    # 6. Lye Flow
    ax = axes[1, 2]
    if 'v_lye' in df.columns:
        ax.plot(t_arr / 60, df['v_lye'] * 1000, 'orange', label='Stack')  # Convert to L/s
    ax.set_title('Stack Lye Flow')
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
    if 'n_H2_an' in df.columns:
        ax.plot(t_arr / 60, df['n_H2_an'], 'b-', label='n_H2_an')
    if 'n_liq' in df.columns:
        ax.plot(t_arr / 60, df['n_liq'], 'g-', label='n_liq')
    if 'n_gas' in df.columns:
        ax.plot(t_arr / 60, df['n_gas'], 'r-', label='n_gas')
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


def run_step_experiment(exp_id, timestamp, n_steps, dt_ctrl, horizon, sim_dt, T_ref,
                         output_dir, power_levels, min_duration, max_duration,
                         max_retries_on_failure=2):
    """
    Run a single step experiment.
    Returns list of data rows or None if failed.
    """
    # Initialize simulator
    sim = SingleStackSimulator(sim_dt=sim_dt)
    sim.reset()

    # Randomize initial condition (NO warmup - start from random state directly)
    I_prev, v_lye_prev, v_c_prev = randomize_initial_condition(sim)

    # Initialize controller
    try:
        controller = SingleStackNMPCController(dt=dt_ctrl, horizon=horizon, sim_dt=sim_dt)
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
        T_s_in, T_s, T_sep, T_c_out, n_H2_an, n_liq, n_gas = current_state

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
        I_cmd = u0[0]
        v_lye_cmd = u0[1]
        v_c_cmd = u0[2]

        # Calculate derived properties
        _, U_cell, _ = sim._calculate_electrochemical_properties(I_cmd, T_s)
        P_real = I_cmd * U_cell * sim.N_cell

        # HTO calculation
        hto_val = (n_gas * sim.R * T_sep) / (sim.P_sys * sim.V_sep_gas) * 100.0

        # H2 production rate
        H2_rate_total = sim.N_cell * I_cmd / (2 * sim.F)

        # Step simulator
        sim_steps = int(dt_ctrl / sim_dt)
        action_sim = np.array([I_cmd, v_lye_cmd, v_c_cmd])

        for _ in range(sim_steps):
            sim.step(action_sim)

        # Build data row
        row = {
            'step': t,
            'time': t * dt_ctrl,
            'P_ref': current_P_ref,
            'P_real': P_real,
            'T_s_in': T_s_in,
            'T_s': T_s,
            'T_sep': T_sep,
            'T_c_out': T_c_out,
            'n_H2_an': n_H2_an,
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

        # Save NMPC plan
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

        # Update previous actions
        I_prev = I_cmd
        v_lye_prev = v_lye_cmd
        v_c_prev = v_c_cmd

    return data_list


def main():
    parser = argparse.ArgumentParser(description='Generate step signal NMPC dataset for single-stack AWE.')
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
    parser.add_argument('--power_levels', type=str, default='1.0,2.0,3.0,4.0,5.0',
                        help='Comma-separated power levels in MW (default: 1.0,2.0,3.0,4.0,5.0)')
    parser.add_argument('--max_retries', type=int, default=2,
                        help='Max retries per experiment on failure (default: 2)')
    parser.add_argument('--skip_plots', action='store_true',
                        help='Skip plot generation (default: generate plots)')
    parser.add_argument('--max_workers', type=int, default=-1,
                        help='Number of parallel workers (default: -1 for all cores)')
    args = parser.parse_args()

    # Parse power levels
    power_levels = [float(x) * 1e6 for x in args.power_levels.split(',')]

    # Base output directory
    output_base_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..',
        'output', 'single_stack', 'dataset'
    ))

    # Generate unified timestamp and create subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_base_dir, f'step_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Step Signal Dataset Generation for Single-Stack AWE")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Experiments: {args.n_experiments}")
    print(f"  Steps per experiment: {args.n_steps} ({args.n_steps * args.dt_ctrl / 3600:.1f} hours)")
    print(f"  Control period: {args.dt_ctrl}s")
    print(f"  NMPC horizon: {args.horizon}")
    print(f"  Power levels: {[f'{p/1e6:.1f}MW' for p in power_levels]}")
    print(f"  Step duration: {args.step_min_duration}-{args.step_max_duration} steps")
    print(f"  Output directory: {output_dir}")
    print(f"  Timestamp: {timestamp}")
    print("=" * 60)

    # Run experiments in parallel using joblib
    print(f"\nStarting parallel execution with {args.max_workers} workers...")

    # Prepare arguments for each experiment
    experiment_results = Parallel(n_jobs=args.max_workers, backend='loky')(
        delayed(run_step_experiment)(
            exp_id=exp_id,
            timestamp=timestamp,
            n_steps=args.n_steps,
            dt_ctrl=args.dt_ctrl,
            horizon=args.horizon,
            sim_dt=args.sim_dt,
            T_ref=args.T_ref,
            output_dir=output_dir,
            power_levels=power_levels,
            min_duration=args.step_min_duration,
            max_duration=args.step_max_duration,
            max_retries_on_failure=args.max_retries
        )
        for exp_id in range(args.n_experiments)
    )

    # Process results
    successful_experiments = 0
    failed_experiments = 0

    for exp_id, result in enumerate(experiment_results):
        if result is not None:
            # Save data
            csv_file = save_experiment_data(result, output_dir, timestamp, exp_id)

            # Generate plot
            if csv_file and not args.skip_plots:
                try:
                    df = pd.read_csv(csv_file)
                    plot_experiment(df, output_dir, timestamp, exp_id)
                except Exception as e:
                    print(f"  Plot generation failed for exp {exp_id}: {e}")

            successful_experiments += 1
        else:
            print(f"  Experiment {exp_id} failed, skipping...")
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
