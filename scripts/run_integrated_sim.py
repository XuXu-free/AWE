
import numpy as np
import matplotlib.pyplot as plt
import os
from plant.integrated_simulator import IntegratedSimulator

def run_simulation():
    # Instantiate the new integrated simulator
    sim = IntegratedSimulator(dt=1.0)
    
    # Standard Operating Conditions
    I_op = 7800.0  # A
    v_lye_op = 0.0335 # m^3/s
    v_c_op = 0.056  # m^3/s
    
    # Initialize
    print("Initializing Integrated Simulation...")
    sim.reset()
    
    # Run for 2 hours (7200s) to see tank heat up
    duration = 7200
    steps = int(duration / sim.dt)
    
    print(f"Running for {duration} seconds...")
    
    # Action vector (Constant)
    action = np.zeros(9)
    action[0:4] = I_op
    action[4:8] = v_lye_op
    action[8] = v_c_op
    
    # History Storage
    time_hist = []
    T_c_out_hist = []
    Tank_T_hist = []
    HP_Power_hist = []
    HP_Heat_hist = []
    HP_COP_hist = []
    
    for i in range(steps):
        t = i * sim.dt
        
        # Step simulation
        sim.step(action)
        
        # Get extended state
        info = sim.get_full_state()
        
        # Record
        time_hist.append(t)
        T_c_out_hist.append(info["T_c_out_C"])
        Tank_T_hist.append(info["Tank_T_C"])
        HP_Power_hist.append(info["HP_Power_kW"])
        HP_Heat_hist.append(info["HP_Heat_Out_kW"])
        HP_COP_hist.append(info["HP_COP"])
        
        if i % 1000 == 0:
            print(f"t={t:.0f}s | T_c_out={info['T_c_out_C']:.2f}C | Tank={info['Tank_T_C']:.2f}C | HP_Heat={info['HP_Heat_Out_kW']:.2f}kW")
            
    # --- Plotting ---
    print("Generating plots...")
    
    # Convert to arrays
    time_arr = np.array(time_hist) / 60.0 # Minutes
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Plot 1: Temperatures
    ax1.plot(time_arr, T_c_out_hist, label='Cooling Water Out (Source)', color='blue')
    ax1.plot(time_arr, Tank_T_hist, label='Thermal Storage Tank (Sink)', color='red', linewidth=2)
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('Thermal System Temperatures')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Heat Pump Power & Heat
    ax2.plot(time_arr, HP_Heat_hist, label='Heat Delivered to Tank', color='orange')
    ax2.plot(time_arr, HP_Power_hist, label='Heat Pump Elec. Power', color='green', linestyle='--')
    ax2.set_ylabel('Power (kW)')
    ax2.set_title('Heat Pump Performance')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: COP
    ax3.plot(time_arr, HP_COP_hist, label='COP', color='purple')
    ax3.set_ylabel('COP')
    ax3.set_xlabel('Time (minutes)')
    ax3.set_title('Coefficient of Performance')
    ax3.set_ylim(0, 8)
    ax3.grid(True)
    
    plt.tight_layout()
    output_file = os.path.join('output', 'integrated_simulation_results.png')
    plt.savefig(output_file)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    run_simulation()
