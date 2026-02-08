
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from plant.multi_stack_simulator import MultiStackSimulator
from controller.multi_stack_nmpc_controller import MultiStackNMPCController

def save_plot(history, output_dir, filename):
    t_arr = np.array(history['t'])
    if len(t_arr) == 0:
        return
        
    T_s_all = np.array(history['T_s_all']) # Shape: (N, 4)
    U_cell_all = np.array(history['U_cell_all']) # Shape: (N, 4)
    
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
    for i in range(4):
        ax.plot(t_arr, T_s_all[:, i], label=f'Stack {i+1}')
    ax.plot(t_arr, history['T_ref'], 'k--', linewidth=1.5, label='Ref Temp')
    ax.axvline(x=0, color='gray', linestyle=':')
    ax.set_title('Individual Stack Temperatures')
    ax.set_ylabel('°C')
    ax.legend()
    ax.grid(True)
    
    # Current
    ax = axes[1,1]
    ax.plot(t_arr, history['I_mean'], 'b-')
    ax.axvline(x=0, color='gray', linestyle=':')
    ax.set_title('Mean Stack Current')
    ax.set_ylabel('Amps')
    ax.grid(True)
    
    # Stack Voltages
    ax = axes[1,2]
    for i in range(4):
        ax.plot(t_arr, U_cell_all[:, i], label=f'Stack {i+1}')
    ax.axvline(x=0, color='gray', linestyle=':')
    ax.set_title('Stack Voltages (Per Cell)')
    ax.set_ylabel('Volts')
    ax.legend()
    ax.grid(True)
    
    # Lye Flow
    ax = axes[2,0]
    ax.plot(t_arr, history['v_lye_mean'], 'orange', label='Mean Lye')
    ax.axvline(x=0, color='gray', linestyle=':')
    ax.set_title('Mean Lye Flow')
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
    ax.set_title('Total H2 Production Rate')
    ax.set_ylabel('mol/s')
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close(fig)

def save_data_csv(history, output_dir, filename):
    keys_scalar = ['t', 'P_ref', 'P_real', 'T_s_mean', 'T_sep', 'T_c_out', 'T_ref', 'I_mean', 'v_lye_mean', 'v_c', 'HTO', 'H2_rate']
    
    if not history['t']:
        return

    # Convert to arrays for shape handling
    T_s_all = np.array(history['T_s_all'])
    U_cell_all = np.array(history['U_cell_all'])
    
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header = keys_scalar + [f'T_s_{i+1}' for i in range(4)] + [f'U_cell_{i+1}' for i in range(4)]
        writer.writerow(header)
        
        # Rows
        n_steps = len(history['t'])
        for i in range(n_steps):
            row = [history[k][i] for k in keys_scalar]
            row.extend(T_s_all[i])
            row.extend(U_cell_all[i])
            writer.writerow(row)
    print(f"Data saved to {filepath}")

def run_test():
    # Setup
    dt = 1.0
    sim = MultiStackSimulator(dt=dt)
    ctrl = MultiStackNMPCController(dt=10.0, horizon=10) # Control interval 10s
    
    # Init
    sim.reset()
    state = sim.get_state()
    # x = [T_s_in, T_s1...4, T_sep, T_c_out, ...]
    
    # Profile
    duration = 30000 # 30,000s to match the reference image x-axis
    t_eval = np.arange(0, duration, dt)
    
    # Power Profile Generation (Mimicking the reference image)
    # The image shows fluctuating power (Renewable-like) + an Overload phase
    P_ref_profile = np.zeros_like(t_eval)
    
    for i, t in enumerate(t_eval):
        if t < 21000:
            # Fluctuating Phase (0 - 2.1e4 s)
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
            P_ref_profile[i] = val * 1e6
            
        elif t < 27000:
            # Overload Phase (2.1e4 - 2.7e4 s)
            # Reference goes high (~26-33 MW), System limited to ~25MW
            # Plateau with fluctuations
            val = 28.0 + 3.0 * np.sin(2 * np.pi * (t - 21000) / 2500) + \
                  1.5 * np.sin(2 * np.pi * t / 800)
            
            # Clip [25, 33]
            val = np.clip(val, 25.0, 33.0)
            P_ref_profile[i] = val * 1e6
            
        else:
            # Ramp Down Phase (> 2.7e4 s)
            decay = (t - 27000) / 3000.0 # 0 to 1
            val = 22.0 - decay * 10.0 # Drop to ~12
            # Add small fluctuation
            val += 1.0 * np.sin(2 * np.pi * t / 1000)
            P_ref_profile[i] = val * 1e6
            
    history = {
        't': [], 'P_ref': [], 'P_real': [],
        'T_s_mean': [], 'T_s_all': [], 'T_sep': [], 'T_c_out': [], 'T_ref': [],
        'I_mean': [], 'v_lye_mean': [], 'v_c': [],
        'U_cell_all': [], 'HTO': [], 'H2_rate': []
    }
    
    last_action = [
        np.zeros(4), # I
        np.ones(4)*0.03, # v_lye
        0.0 # v_c
    ]
    
    # --- Warm-up Phase ---
    warmup_duration = 1000
    warmup_P_ref = 8.0e6
    print(f"Starting Warm-up Phase ({warmup_duration}s at {warmup_P_ref/1e6}MW)...")
    
    warmup_steps = int(warmup_duration / dt)
    for i in range(warmup_steps):
        t_warmup = -warmup_duration + i * dt
        
        # Extract State
        T_s_in = sim.state[0] - 273.15
        T_s_vec = sim.state[1:5] - 273.15
        T_sep = sim.state[5] - 273.15
        T_c_out = sim.state[6] - 273.15
        
        # Calculate Real Power (Approx)
        Q, U_cell, eta = sim._calculate_electrochemical_properties(last_action[0], sim.state[1:5])
        P_real = np.sum(U_cell * last_action[0] * sim.N_cell)
        
        # Calculate Extra Metrics
        # HTO %
        n_H2_sep_gas = sim.state[12]
        T_sep_K = sim.state[5]
        n_gas_total = (sim.p_sys * sim.V_sep_gas) / (sim.R * T_sep_K)
        hto_pct = (n_H2_sep_gas / n_gas_total) * 100 if n_gas_total > 1e-9 else 0.0
        
        # H2 Production Rate (mol/s)
        # n_dot = N * I * eta / (2 * F)
        F_const = 96485.0
        h2_rate = np.sum(sim.N_cell * last_action[0] * eta / (2 * F_const))

        # Control
        if i % 10 == 0: # 10s control loop
            I_cmd, v_lye_cmd, v_c_cmd = ctrl.get_action(
                sim.state, warmup_P_ref
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
        history['T_s_mean'].append(np.mean(T_s_vec))
        history['T_s_all'].append(T_s_vec)
        history['T_sep'].append(T_sep)
        history['T_c_out'].append(T_c_out)
        history['T_ref'].append(ctrl.T_ref)
        history['I_mean'].append(np.mean(last_action[0]))
        history['v_lye_mean'].append(np.mean(last_action[1]))
        history['v_c'].append(last_action[2])
        history['U_cell_all'].append(U_cell)
        history['HTO'].append(hto_pct)
        history['H2_rate'].append(h2_rate)
        
        if i % 100 == 0:
            print(f"Warmup: {(i*dt)/warmup_duration*100:.0f}% | T_s_mean={np.mean(T_s_vec):.1f}C")

    print("Warm-up Complete. Starting Main Test...")

    print("Starting Multi-Stack NMPC Test...")
    
    # Generate timestamped filename for data logging
    data_filename = f"multi_stack_nmpc_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    # Loop
    for i, t in enumerate(t_eval):
        # Extract State
        # x = [T_s_in, T_s1...4, T_sep, T_c_out, ...]
        T_s_in = sim.state[0] - 273.15
        T_s_vec = sim.state[1:5] - 273.15
        T_sep = sim.state[5] - 273.15
        T_c_out = sim.state[6] - 273.15
        
        # Calculate Real Power
        # Need voltage. Simulator step calculates it but doesn't return it directly unless we modify it.
        # Approximation: P = sum(I_i * V_cell_i * N_cell)
        # Or re-call internal method
        Q, U_cell, eta = sim._calculate_electrochemical_properties(last_action[0], sim.state[1:5])
        P_real = np.sum(U_cell * last_action[0] * sim.N_cell)
        
        # Calculate Extra Metrics
        # HTO %
        n_H2_sep_gas = sim.state[12]
        T_sep_K = sim.state[5]
        n_gas_total = (sim.p_sys * sim.V_sep_gas) / (sim.R * T_sep_K)
        hto_pct = (n_H2_sep_gas / n_gas_total) * 100 if n_gas_total > 1e-9 else 0.0
        
        # H2 Production Rate (mol/s)
        F_const = 96485.0
        h2_rate = np.sum(sim.N_cell * last_action[0] * eta / (2 * F_const))
        
        # Control
        if i % 10 == 0: # 10s control loop
            # P_target = P_ref_profile[i] # Scalar
            
            # Construct P_ref vector for horizon N=10
            # Controller uses dt=10.0s, but profile is defined on dt=1.0s
            # We need to sample P_ref_profile at t, t+10, ..., t+(N-1)*10
            P_future = []
            for k in range(ctrl.N):
                t_future = t + k * ctrl.dt # ctrl.dt = 10.0
                idx_future = int(t_future / dt) # dt = 1.0
                if idx_future < len(P_ref_profile):
                    P_future.append(P_ref_profile[idx_future])
                else:
                    P_future.append(P_ref_profile[-1])
            
            I_cmd, v_lye_cmd, v_c_cmd = ctrl.get_action(
                sim.state, P_future
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
        history['T_s_mean'].append(np.mean(T_s_vec))
        history['T_s_all'].append(T_s_vec)
        history['T_sep'].append(T_sep)
        history['T_c_out'].append(T_c_out)
        history['T_ref'].append(ctrl.T_ref)
        history['I_mean'].append(np.mean(last_action[0]))
        history['v_lye_mean'].append(np.mean(last_action[1]))
        history['v_c'].append(last_action[2])
        history['U_cell_all'].append(U_cell)
        history['HTO'].append(hto_pct)
        history['H2_rate'].append(h2_rate)
        
        if i % 100 == 0:
            print(f"t={t:.0f}s | P_ref={P_ref_profile[i]/1e6:.1f}MW | P_real={P_real/1e6:.1f}MW | T_s_mean={np.mean(T_s_vec):.1f}C")
        
        # Periodic Plot Update (every 5000s)
        if t > 0 and t % 5000 == 0:
            print(f"Updating progress plot at t={t}s...")
            output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            save_plot(history, output_dir, 'multi_stack_nmpc_test_progress.png')
            save_data_csv(history, output_dir, data_filename)
            
    # Plot
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    save_plot(history, output_dir, 'multi_stack_nmpc_test.png')
    save_data_csv(history, output_dir, data_filename)
    
    # Calculate RMSE Metrics
    t_arr = np.array(history['t'])
    P_ref_arr = np.array(history['P_ref'])
    P_real_arr = np.array(history['P_real'])
    T_s_all = np.array(history['T_s_all']) # (N, 4)
    T_ref_arr = np.array(history['T_ref']) # (N,)
    
    # 1. Load-tracking RMSE (MW)
    # RMSE = sqrt( mean( (P_real - P_ref)^2 ) )
    rmse_load_mw = np.sqrt(np.mean((P_real_arr - P_ref_arr)**2)) / 1e6
    
    # 2. RMSE for stack temperature control (K)
    # RMSE = sqrt( mean( (T_stack_i - T_ref)^2 ) ) over all stacks and time
    # Expand T_ref to (N, 4) for easy subtraction
    T_ref_expanded = T_ref_arr[:, np.newaxis] # (N, 1)
    rmse_temp_k = np.sqrt(np.mean((T_s_all - T_ref_expanded)**2))
    
    print("-" * 50)
    print(f"Performance Metrics (Duration: {duration}s)")
    print(f"Load-tracking RMSE:             {rmse_load_mw:.3f} MW")
    print(f"RMSE for stack temperature control: {rmse_temp_k:.3f} K")
    print("-" * 50)
    
    print("Test Complete. Results saved.")

if __name__ == "__main__":
    run_test()
