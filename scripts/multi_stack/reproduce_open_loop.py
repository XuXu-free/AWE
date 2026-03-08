import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from plant.multi_stack_simulator import MultiStackSimulator

# --- Custom RK4 Solver Removed (Using sim.step) ---

def run_simulation():
    sim = MultiStackSimulator()
    
    # --- Control Actions (Figure 5) ---
    # Initial: I=7.8kA, v_lye=0.0335, v_c=0.0024 (approx from plot)
    # Stage I (0.9e3): v_lye_1 drops to 0.025
    # Stage II (1.8e3): I_1 drops to 3.5kA
    # Stage III (2.7e3): I_2 drops to 3.5kA
    # Stage IV (3.6e3): I_3 drops to 3.5kA
    # Stage V (4.5e3): v_c drops to 0.001 (approx)
    
    def get_inputs(t):
        # Initial Values
        I = np.array([7800.0] * 4)
        v_lye = np.array([0.0335] * 4)
        # v_c (Coolant flow) - Step down at 4500s
        # Note: Paper text says 2.2e-3, but this is physically inconsistent (overheating).
        # We use 2.2e-2 to match thermal balance and assume typo in paper exponent.
        v_c = 0.056
        if t >= 900:
            v_lye[0] = 0.025
        if t >= 1800:
            I[0] = 3500.0
        if t >= 2700:
            I[1] = 3500.0
        if t >= 3600:
            I[2] = 3500.0
        if t >= 4500:
            v_c = 0.010 # Based on plot (1.0 x 10^-3)
            
        return np.concatenate([I, v_lye, [v_c]])

    # --- Initialization ---
    print("Initializing simulation...")
    sim.reset()
    
    x0 = sim.get_state()
    print(f"Initial State Set: T_s_in={sim.T_s_init:.2f} K, T_sep={sim.T_s_in_init:.2f} K, HTO={sim.HTO_init:.3f}%")
    
    # --- Main Simulation ---
    print("Running main simulation...")
    t_end = 7200
    t_eval = np.linspace(0, t_end, 7201)
    
    # --- Simulation Loop (using sim.step) ---
    dt = t_eval[1] - t_eval[0]
    sim.dt = dt
    sim.state = x0.copy()
    sim.current_time = 0.0
    
    # Storage
    y_history = []
    
    for t in t_eval:
        # Record current state
        y_history.append(sim.state.copy())
        
        # Get control input
        u = get_inputs(t)
        
        # Step simulation
        sim.step(u)
        
    y_history = np.array(y_history).T # Shape (n_states, n_steps)
    
    # --- Data Extraction ---
    time = t_eval / 1000.0 # x10^3 s
    T_s = y_history[1:5, :]
    T_sep = y_history[5, :]
    T_s_in = y_history[0, :]
    # Inline HTO calculation (Vectorized)
    n_gas_hist = y_history[12, :]
    HTO = (n_gas_hist * sim.R * T_sep) / (sim.P_sys * sim.V_sep_gas) * 100.0
    
    # Inputs for plotting
    I_plot = np.zeros((4, len(t_eval)))
    v_lye_plot = np.zeros((4, len(t_eval)))
    v_c_plot = np.zeros(len(t_eval))
    
    for i, t in enumerate(t_eval):
        u = get_inputs(t)
        I_plot[:, i] = u[0:4] / 1000.0 # kA
        v_lye_plot[:, i] = u[4:8] * 100.0 # x10^-2
        v_c_plot[i] = u[8] * 1000.0 # x10^-3. If v_c=0.022, result is 2.2. Fits 0-3 scale.
        
    # --- Plotting ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # Adjust spacing
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    
    # Shared settings
    stages = [0.9, 1.8, 2.7, 3.6, 4.5]
    stage_labels = ['I', 'II', 'III', 'IV', 'V']
    
    def setup_axis(ax, xlabel=None):
        ax.set_xlim(0, 7.2)
        # Vertical lines
        for s in stages:
            ax.axvline(x=s, color='gray', linestyle='-.', linewidth=1.0)
        # Stage labels (centered between lines)
        # 0-0.9, 0.9-1.8, etc.
        boundaries = [0] + stages + [7.2]
        
        if xlabel:
            ax.set_xlabel(xlabel)
            
        # No grid
        ax.grid(False)
        
        # Ticks direction in
        ax.tick_params(direction='in', top=True, right=True)

    # (a) Current
    ax = axes[0, 0]
    setup_axis(ax, 'Time ($\\times 10^3$ s)')
    ax.set_ylabel('Current (kA)')
    ax.set_ylim(0, 8.5)
    
    styles = ['-', '--', '-.', ':'] # Stack 1, 2, 3, 4
    colors = ['steelblue', 'orange', 'green', 'red']
    
    for i in range(4):
        ax.plot(time, I_plot[i], color=colors[i], linestyle=styles[i], label=f'Stack {i+1}')
        
    # Legend inside
    ax.legend(loc='lower left', bbox_to_anchor=(0.02, 0.05), frameon=True)
    ax.text(-0.1, 1.05, '(a)', transform=ax.transAxes, fontsize=14, fontweight='bold')
    
    # Stage labels for (a)
    for i, label in enumerate(stage_labels):
        mid = (stages[i] + (stages[i-1] if i>0 else 0)) / 2
        ax.text(mid, 0.5, label, ha='center', fontsize=10)

    # (b) Flows
    ax = axes[0, 1]
    setup_axis(ax, 'Time ($\\times 10^3$ s)')
    ax.set_ylabel('Lye flow ($\\times 10^{-2}$ m$^3$/s)', color='steelblue')
    ax.set_ylim(1.8, 3.6)
    
    # Left axis: Lye
    # Stack 1 (Blue solid)
    l1, = ax.plot(time, v_lye_plot[0], color='steelblue', label='Lye flow of stack 1')
    # Stacks 2-4 (Red/Orange dash) - They are identical
    l2, = ax.plot(time, v_lye_plot[1], color='brown', linestyle='--', label='Lye flow of stacks 2-4')
    
    ax.tick_params(axis='y', labelcolor='steelblue')
    
    # Right axis: Coolant
    ax2 = ax.twinx()
    # Plot v_c on right axis with green dashed line
    ax2.set_ylabel('Coolant flow ($\\times 10^{-3}$ m$^3$/s)', color='g')
    ax2.set_ylim(0, 2.6)
    # v_c_plot is scaled (x100). Plotting it directly on 0-3.0 axis.
    l3, = ax2.plot(time, v_c_plot, 'g--', linewidth=1.5, label='Coolant flow')
    ax2.tick_params(axis='y', labelcolor='g', direction='in', right=True)
    # No grid on twin either
    ax2.grid(False)
    
    # Legend
    lns = [l1, l2, l3]
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='center right', frameon=False, fontsize=9)
    
    ax.text(-0.1, 1.05, '(b)', transform=ax.transAxes, fontsize=14, fontweight='bold')
    
    # Stage labels for (b)
    for i, label in enumerate(stage_labels):
        mid = (stages[i] + (stages[i-1] if i>0 else 0)) / 2
        ax.text(mid, 1.85, label, ha='center', fontsize=10)

    # (c) Temperatures
    ax = axes[1, 0]
    setup_axis(ax, 'Time ($\\times 10^3$ s)')
    ax.set_ylabel('Stack temperature (K)')
    ax.set_ylim(330, 390) # Expanded to cover drift
    
    for i in range(4):
        ax.plot(time, T_s[i], color=colors[i], linestyle=styles[i], label=f'Stack {i+1}')
    
    # Separator/Inlet
    # User requested T_s_in instead of T_sep
    ax.plot(time, T_s_in, color='mediumvioletred', linestyle='--', label='Stack Inlet Temperature')
    
    ax.legend(loc='upper right', ncol=2, fontsize=8)
    ax.text(-0.1, 1.05, '(c)', transform=ax.transAxes, fontsize=14, fontweight='bold')
    
    # Stage labels for (c)
    for i, label in enumerate(stage_labels):
        mid = (stages[i] + (stages[i-1] if i>0 else 0)) / 2
        ax.text(mid, 331, label, ha='center', fontsize=10)

    # (d) HTO
    ax = axes[1, 1]
    setup_axis(ax, 'Time ($\\times 10^3$ s)')
    ax.set_ylabel('HTO (%)')
    ax.set_ylim(0.4, 1.05)
    
    ax.plot(time, HTO, color='steelblue', linewidth=2)
    
    ax.text(-0.1, 1.05, '(d)', transform=ax.transAxes, fontsize=14, fontweight='bold')
    
    # Stage labels for (d)
    for i, label in enumerate(stage_labels):
        mid = (stages[i] + (stages[i-1] if i>0 else 0)) / 2
        ax.text(mid, 0.42, label, ha='center', fontsize=10)

    output_file = os.path.join('output', 'open_loop_reproduction.png')
    plt.savefig(output_file, dpi=300)
    print(f"Saved plot to {output_file}")

if __name__ == "__main__":
    run_simulation()
