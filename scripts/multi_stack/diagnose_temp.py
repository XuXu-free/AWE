
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from plant.multi_stack_simulator import MultiStackSimulator

def run_diagnostic():
    sim = MultiStackSimulator()
    
    # --- Standard Operating Conditions (Original Parameters) ---
    I_op = 7800.0  # A
    v_lye_op = 0.0335 # m^3/s (per stack)
    v_c_op = 0.056  # m^3/s
    
    # --- Initialization ---
    print("Initializing simulation...")
    sim.reset()
    
    # Run for 10000 seconds (Original Duration)
    duration = 10000
    steps = int(duration / sim.dt)
    # Ensure consistent time evaluation
    t_eval = np.linspace(0, duration, steps + 1)
    sim.dt = t_eval[1] - t_eval[0]
    
    # Storage
    y_history = []
    I_hist = []
    v_lye_hist = []
    v_c_hist = []
    
    print(f"Running simulation for {duration}s with constant inputs...")
    print(f"Inputs: I={I_op}A, v_lye={v_lye_op}, v_c={v_c_op}")
    
    # Constant Action vector
    # u = [I1...4, v_lye1...4, v_c]
    action = np.zeros(9)
    action[0:4] = I_op
    action[4:8] = v_lye_op
    action[8] = v_c_op
    
    for t in t_eval:
        # Record current state
        y_history.append(sim.state.copy())
        
        # Store inputs for plotting
        I_hist.append(action[0:4])
        v_lye_hist.append(action[4:8])
        v_c_hist.append(action[8])
        
        # Step simulation
        sim.step(action)
        
    y_history = np.array(y_history).T # Shape (n_states, n_steps)
    I_hist = np.array(I_hist).T
    v_lye_hist = np.array(v_lye_hist).T
    v_c_hist = np.array(v_c_hist)
    
    # --- Data Extraction ---
    time = t_eval / 1000.0 # x10^3 s
    T_s = y_history[1:5, :]
    T_sep = y_history[5, :]
    T_s_in = y_history[0, :]
    
    # HTO Calculation (Hydrogen in Oxygen)
    # n_gas is index 12 (13th element) in state vector
    n_gas_hist = y_history[12, :]
    HTO = (n_gas_hist * sim.R * T_sep) / (sim.p_sys * sim.V_sep_gas) * 100.0
    
    # --- Plotting ---
    print("Generating plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # Adjust spacing
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    
    def setup_axis(ax, xlabel=None):
        ax.set_xlim(0, duration/1000.0)
        if xlabel:
            ax.set_xlabel(xlabel)
            
        # No grid
        ax.grid(True, linestyle=':', alpha=0.6)
        # Ticks direction in
        ax.tick_params(direction='in', top=True, right=True)

    # (a) Current
    ax = axes[0, 0]
    setup_axis(ax, 'Time ($\\times 10^3$ s)')
    ax.set_ylabel('Current (kA)')
    # ax.set_ylim(0, 8.5) # Let it autoscale or set range to show the constant line clearly
    ax.set_ylim(0, 10) 
    
    styles = ['-', '--', '-.', ':'] # Stack 1, 2, 3, 4
    colors = ['steelblue', 'orange', 'green', 'red']
    
    for i in range(4):
        ax.plot(time, I_hist[i]/1000.0, color=colors[i], linestyle=styles[i], label=f'Stack {i+1}')
        
    ax.legend(loc='lower left', bbox_to_anchor=(0.02, 0.05), frameon=True)
    ax.text(-0.1, 1.05, '(a)', transform=ax.transAxes, fontsize=14, fontweight='bold')

    # (b) Flows
    ax = axes[0, 1]
    setup_axis(ax, 'Time ($\\times 10^3$ s)')
    ax.set_ylabel('Lye flow ($\\times 10^{-2}$ m$^3$/s)', color='steelblue')
    # ax.set_ylim(1.8, 3.6)
    
    # Left axis: Lye
    # Stack 1 (Blue solid)
    l1, = ax.plot(time, v_lye_hist[0]*100, color='steelblue', label='Lye flow of stack 1')
    # Stacks 2-4 (Red/Orange dash) - They are identical
    l2, = ax.plot(time, v_lye_hist[1]*100, color='brown', linestyle='--', label='Lye flow of stacks 2-4')
    
    ax.tick_params(axis='y', labelcolor='steelblue')
    
    # Right axis: Coolant
    ax2 = ax.twinx()
    # Plot v_c on right axis with green dashed line
    ax2.set_ylabel('Coolant flow ($\\times 10^{-3}$ m$^3$/s)', color='g')
    # ax2.set_ylim(0, 2.6)
    # v_c_plot is scaled (x1000)
    l3, = ax2.plot(time, v_c_hist*1000, 'g--', linewidth=1.5, label='Coolant flow')
    ax2.tick_params(axis='y', labelcolor='g', direction='in', right=True)
    ax2.grid(False)
    
    # Legend
    lns = [l1, l2, l3]
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='center right', frameon=False, fontsize=9)
    
    ax.text(-0.1, 1.05, '(b)', transform=ax.transAxes, fontsize=14, fontweight='bold')

    # (c) Temperatures
    ax = axes[1, 0]
    setup_axis(ax, 'Time ($\\times 10^3$ s)')
    ax.set_ylabel('Stack temperature (K)')
    # ax.set_ylim(330, 390) 
    
    for i in range(4):
        ax.plot(time, T_s[i], color=colors[i], linestyle=styles[i], label=f'Stack {i+1}')
    
    # Inlet Temperature
    ax.plot(time, T_s_in, color='mediumvioletred', linestyle='--', label='Stack Inlet Temperature')
    
    ax.legend(loc='lower right', ncol=2, fontsize=8)
    ax.text(-0.1, 1.05, '(c)', transform=ax.transAxes, fontsize=14, fontweight='bold')

    # (d) HTO
    ax = axes[1, 1]
    setup_axis(ax, 'Time ($\\times 10^3$ s)')
    ax.set_ylabel('HTO (%)')
    # ax.set_ylim(0.4, 1.05)
    
    ax.plot(time, HTO, color='steelblue', linewidth=2)
    
    ax.text(-0.1, 1.05, '(d)', transform=ax.transAxes, fontsize=14, fontweight='bold')

    output_file = os.path.join('output', 'diagnose_temp_plot.png')
    plt.savefig(output_file, dpi=300)
    print(f"\nPlot saved to {output_file}")

if __name__ == "__main__":
    run_diagnostic()
