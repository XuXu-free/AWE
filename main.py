import time
import numpy as np
import sys
import os

# Add the project root to path so imports work if running from inside or outside
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from plant import AWESimulator
from controller import PIDController, MPCController
from controller.nmpc_controller import NMPCController

def run_simulation(simulator, controller, setpoint, steps=100):
    print(f"\nStarting Simulation with {controller.__class__.__name__}...")
    if isinstance(setpoint, (list, np.ndarray)) and len(setpoint) > 0:
         print(f"Target Temperature: {setpoint[0]} C")
    
    # Storage for plotting/logging
    history = {
        'time': [],
        'T': [],
        'T_cw_out': [],
        'I': [],
        'Power': [],
        'H2_rate': [],
        'T_storage': [],
        'HP_Power_Elec': [],
        'HP_Heat_Recovered': [],
        'HP_COP': [],
        'v_lye': [],
        'v_cw': []
    }
    
    state = simulator.reset()
    controller.reset()
    
    # Current Reference for NMPC (if needed) or Power Reference
    # Main loop assumes setpoint is [T_target]
    # For NMPC, we might need P_ref. Let's assume constant P_ref for main.py demo
    P_ref_demo = 4.0e6 # 4 MW
    
    for i in range(steps):
        # 1. Get Control Action
        if isinstance(controller, NMPCController):
             # NMPC needs specific inputs
             T_curr = state[0]
             T_cw_out_curr = state[1] if len(state) > 1 else 20.0
             action = controller.get_action(T_curr, T_cw_out_curr, P_ref_demo)
        else:
             action = controller.get_action(state, setpoint)
        
        # 2. Simulate System Step
        next_state, info = simulator.step(action)
        
        # 3. Store Data
        history['time'].append(info['time'])
        history['T'].append(info['T'])
        history['T_cw_out'].append(info.get('T_cw_out', 20.0))
        history['I'].append(info['I'])
        history['Power'].append(info['Power'])
        history['H2_rate'].append(info['H2_rate'])
        history['T_storage'].append(info['T_storage'])
        history['HP_Power_Elec'].append(info['HP_Power_Elec'])
        history['HP_Heat_Recovered'].append(info['HP_Heat_Recovered'])
        history['HP_COP'].append(info['HP_COP'])
        
        # Extract flows from action
        if isinstance(action, (list, np.ndarray)) and len(action) > 2:
             history['v_lye'].append(action[1])
             history['v_cw'].append(action[2])
        else:
             history['v_lye'].append(0.04) # Default
             history['v_cw'].append(0.04)  # Default
        
        # Check for shutdown signal
        if info.get('Shutdown', False):
            print(f"\n[CRITICAL] Simulation stopped at Step {i}: Impurity Level {info['Impurity']:.2f}% > Limit!")
            print("System Shutdown Triggered.")
            break
            
        # Update state
        state = next_state
        
        # Optional: Print progress every 10 steps
        if i % 10 == 0:
            print(f"Step {i}: Time={info['time']:.1f}s, T_stack={info['T']:.2f}C, T_cw={info.get('T_cw_out',0):.1f}C")
            
    print("Simulation Complete.")
    return history

def plot_results(history):
    try:
        import matplotlib.pyplot as plt
        
        # 3x3 Grid Layout
        fig, axs = plt.subplots(3, 3, figsize=(18, 12), sharex=True)
        
        # 1. Power
        axs[0, 0].plot(history['time'], np.array(history['Power'])/1e6, 'b-', label='Stack Power')
        axs[0, 0].set_ylabel('Power [MW]')
        axs[0, 0].set_title('Power Consumption')
        axs[0, 0].grid(True)
        
        # 2. Stack Temperature
        axs[0, 1].plot(history['time'], history['T'], 'r-', label='Stack Temp')
        axs[0, 1].axhline(y=85, color='k', linestyle=':', label='Target')
        axs[0, 1].set_ylabel('Temp [C]')
        axs[0, 1].set_title('Stack Temperature')
        axs[0, 1].grid(True)
        
        # 3. Cooling Water Outlet
        axs[0, 2].plot(history['time'], history['T_cw_out'], 'm-', label='CW Out')
        axs[0, 2].set_ylabel('Temp [C]')
        axs[0, 2].set_title('Cooling Water Outlet Temp')
        axs[0, 2].grid(True)
        
        # 4. Current
        axs[1, 0].plot(history['time'], history['I'], 'k-', label='Current')
        axs[1, 0].set_ylabel('Current [A]')
        axs[1, 0].set_title('Current')
        axs[1, 0].grid(True)
        
        # 5. Lye Flow
        axs[1, 1].plot(history['time'], history['v_lye'], 'c-', label='Lye Flow')
        axs[1, 1].set_ylabel('Flow [m3/s]')
        axs[1, 1].set_title('Lye Flow Rate')
        axs[1, 1].grid(True)
        
        # 6. Cooling Water Flow
        axs[1, 2].plot(history['time'], history['v_cw'], 'b-', label='CW Flow')
        axs[1, 2].set_ylabel('Flow [m3/s]')
        axs[1, 2].set_title('Cooling Water Flow Rate')
        axs[1, 2].grid(True)
        
        # 7. Storage Temp
        axs[2, 0].plot(history['time'], history['T_storage'], 'orange', label='Storage Temp')
        axs[2, 0].set_ylabel('Temp [C]')
        axs[2, 0].set_xlabel('Time [s]')
        axs[2, 0].set_title('Thermal Storage Temp')
        axs[2, 0].grid(True)
        
        # 8. HP Power
        axs[2, 1].plot(history['time'], np.array(history['HP_Power_Elec'])/1000.0, 'brown', label='HP Power')
        axs[2, 1].set_ylabel('Power [kW]')
        axs[2, 1].set_xlabel('Time [s]')
        axs[2, 1].set_title('Heat Pump Power')
        axs[2, 1].grid(True)
        
        # 9. HP COP
        axs[2, 2].plot(history['time'], history['HP_COP'], 'purple', label='COP')
        axs[2, 2].set_ylabel('COP [-]')
        axs[2, 2].set_xlabel('Time [s]')
        axs[2, 2].set_title('Heat Pump COP')
        axs[2, 2].grid(True)
        
        plt.tight_layout()
        plt.show()
        print("Plot generated.")
    except ImportError:
        print("Matplotlib not found. Skipping plot.")

def main():
    print("=== AWE Simulation & Control Framework ===")
    
    # 1. Setup Simulation
    dt = 60.0 # Time step in seconds (increased for NMPC stability/speed)
    sim = AWESimulator(dt=dt)
    
    # 2. User Selection
    print("\nSelect Controller:")
    print("1. PID Controller (Basic)")
    print("2. MPC Controller (Simple Linear)")
    print("3. NMPC Controller (Advanced with Cooling Control)")
    
    choice = input("Enter choice (1-3): ").strip()
    
    target_temp = 85.0 # Target operating temperature
    setpoint = np.array([target_temp])
    
    controller = None
    if choice == '1':
        # Simple tuning
        controller = PIDController(dt=dt, kp=10.0, ki=0.5, kd=0.0, u_max=8000.0) # Updated limit
    elif choice == '2':
        try:
            controller = MPCController(dt=dt, N=20, u_max=8000.0)
        except Exception as e:
            print(f"Error initializing MPC: {e}")
            return
    elif choice == '3':
        try:
            controller = NMPCController(dt=dt, horizon=10)
        except Exception as e:
             print(f"Error initializing NMPC: {e}")
             return
    else:
        print("Invalid choice. Defaulting to PID.")
        controller = PIDController(dt=dt, kp=10.0, ki=0.5, kd=0.0)
        
    # 3. Run
    duration = 3600 * 2 # 2 hours
    steps = int(duration / dt)
    
    history = run_simulation(sim, controller, setpoint, steps=steps)
    
    # 4. Show Results
    plot_results(history)

if __name__ == "__main__":
    main()
