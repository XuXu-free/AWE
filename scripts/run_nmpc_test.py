
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from plant.single_stack_simulator import SingleStackSimulator
from controller.nmpc_controller import NMPCController

def run_nmpc_test():
    # 1. Setup
    dt = 1.0 # Simulation step
    sim = SingleStackSimulator(dt=dt)
    
    # NMPC runs at slower rate (e.g. 60s) or same rate? 
    # Controller default dt=60. Let's align them or use zero-order hold.
    ctrl_dt = 10.0 # Let's run control every 10s for better response
    controller = NMPCController(dt=ctrl_dt, horizon=10)
    
    # 2. Initialization
    sim.reset()
    state = sim.get_state()
    # State: [T_s_in, T_s, T_sep, T_c_out, ...]
    
    # 3. Reference Trajectory (Power Profile)
    # Rated Power ~ 1.8V * 7800A * 368 ~ 5.1 MW
    # Let's do a step test
    duration = 3000 # 30 mins
    t_eval = np.arange(0, duration, dt)
    
    P_ref_profile = np.zeros_like(t_eval)
    for i, t in enumerate(t_eval):
        if t < 300:
            P_ref_profile[i] = 2.0e6 # Start low (2 MW)
        elif t < 900:
            P_ref_profile[i] = 5.0e6 # Step up (5 MW)
        elif t < 1500:
            P_ref_profile[i] = 3.5e6 # Step down (3.5 MW)
        elif t < 2100:
            P_ref_profile[i] = 1.0e6 # Ramp down
        else:
            P_ref_profile[i] = 8.0e6 # End at 2 MW
    # 4. Storage
    history = {
        't': [],
        'P_ref': [],
        'P_real': [],
        'T_stack': [],
        'T_ref': [],
        'I': [],
        'v_lye': [],
        'v_c': []
    }
    
    # Control loop variables
    last_action = [0.0, 0.0335, 0.0] # Initial guess
    action = last_action
    
    print("Starting NMPC Closed-Loop Simulation...")
    
    for i, t in enumerate(t_eval):
        # Current State
        # x = [T_s_in, T_s, T_sep, T_c_out, ...]
        T_stack = sim.state[1] - 273.15 # Kelvin to Celsius? 
        # Wait, Simulator uses Kelvin. Controller likely expects Celsius for T_ref=85 but Kelvin for physics?
        # Let's check NMPC code.
        # NMPC: T_kelvin = T_stack_k + 273.15 => T_stack_k is Celsius.
        # Simulator: T_s is Kelvin (358K ~ 85C).
        
        T_stack_C = sim.state[1] - 273.15
        T_c_out_C = sim.state[3] - 273.15
        
        # Power Calculation
        # P = V * I * N_cell
        _, U_cell, _ = sim._calculate_electrochemical_properties(action[0], sim.state[1])
        P_real = U_cell * action[0] * sim.N_cell
        
        # Control Step
        if i % int(ctrl_dt / dt) == 0:
            P_target = P_ref_profile[i]
            
            # Call NMPC
            # get_action(T_current, T_cw_out, P_ref) all in standard units (C, C, W)
            action = controller.get_action(
                T_current=T_stack_C,
                T_cw_out=T_c_out_C,
                P_ref=P_target
            )
            # action = [I, v_lye, v_c]
            
        # Apply Action
        # Simulator inputs: [I, v_lye, v_c]
        sim.step(action)
        
        # Log
        history['t'].append(t)
        history['P_ref'].append(P_ref_profile[i])
        history['P_real'].append(P_real)
        history['T_stack'].append(T_stack_C)
        history['T_ref'].append(controller.T_ref)
        history['I'].append(action[0])
        history['v_lye'].append(action[1])
        history['v_c'].append(action[2])
        
        if i % 100 == 0:
            print(f"Time={t:.1f}s | P_ref={P_ref_profile[i]/1e6:.2f}MW | P_real={P_real/1e6:.2f}MW | T_stack={T_stack_C:.2f}C")

    # 5. Plotting
    print("Generating plots...")
    
    t_arr = np.array(history['t'])
    
    # Use absolute path relative to this script
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Power Plot (Top Left)
    ax = axes[0, 0]
    ax.plot(t_arr, np.array(history['P_ref'])/1e6, 'k--', label='Reference')
    ax.plot(t_arr, np.array(history['P_real'])/1e6, 'b-', label='Actual')
    ax.set_ylabel('Power (MW)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Power Tracking')
    ax.legend()
    ax.grid(True)
    
    # 2. Temperature Plot (Top Right)
    ax = axes[0, 1]
    ax.plot(t_arr, history['T_stack'], 'r-', label='Stack Temp')
    ax.plot(t_arr, history['T_ref'], 'g--', label='Ref Temp')
    ax.set_ylabel('Temperature (°C)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Temperature Control')
    ax.legend()
    ax.grid(True)
    
    # 3. Current Plot (Bottom Left)
    ax = axes[1, 0]
    ax.plot(t_arr, np.array(history['I'])/1000.0, 'b-', label='Current')
    ax.set_ylabel('Current (kA)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Control: Current')
    ax.legend()
    ax.grid(True)
    
    # 4. Flow Rates Plot (Bottom Right)
    ax = axes[1, 1]
    ax.plot(t_arr, history['v_lye'], 'orange', label='Lye Flow')
    ax.plot(t_arr, history['v_c'], 'cyan', label='Coolant Flow')
    ax.set_ylabel('Flow Rate (m^3/s)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Control: Flow Rates')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'nmpc_test_results.png')
    plt.savefig(output_path)
    plt.close()
    
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    run_nmpc_test()
