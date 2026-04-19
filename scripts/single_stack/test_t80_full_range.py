#!/usr/bin/env python3
"""
Full-range T=80°C, NO CBF activation for I ∈ [2000, 7200]A.
Generates mapping table and visualization.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from plant.single_stack_simulator import SingleStackSimulator
from controller.single_stack.cbf_projection import SingleStackCBFProjection

def simulate(I, v_lye, v_c, steps=15000):
    sim = SingleStackSimulator()
    sim.reset()
    u = np.array([I, v_lye, v_c])
    for _ in range(steps):
        sim.step(u)
    T_s = sim.state[1] - 273.15
    n_gas = sim.state[6]
    hto = (n_gas * sim.R * sim.state[2]) / (sim.P_sys * sim.V_sep_gas) * 100
    _, U_cell, _ = sim._calculate_electrochemical_properties(I, sim.state[1])
    P = U_cell * I * sim.N_cell / 1e6
    return T_s, hto, P, sim.state

projector = SingleStackCBFProjection(
    dt=60.0, gamma_vec=[3.0, 2.0, 100.0, 100.0, 5.0],
    rho_vec=[50000]*5, h_margin_vec=[1.0, 0.002, 0.0, 0.0, 0.0],
    lambda_u_scale=[500.0, 200.0, 1000.0],
    soft_mask=[False, True, True, False, True], normalize=True,
)
active_mask = np.array([True, True, True, True, True])

# Refined mapping from search results
# Format: (I, v_lye, v_c) chosen to give T≈80°C with HTO < 1.8% and no CBF
mapping = [
    (2000, 0.010, 0.0008),
    (2400, 0.010, 0.0011),
    (2800, 0.010, 0.0015),
    (3200, 0.010, 0.0021),
    (3600, 0.010, 0.0027),
    (4000, 0.010, 0.0034),
    (4400, 0.010, 0.0043),
    (4800, 0.010, 0.0054),
    (5200, 0.010, 0.0068),
    (5600, 0.010, 0.0085),
    (6000, 0.010, 0.0107),
    (6400, 0.010, 0.0137),
    (6800, 0.010, 0.0181),
    (7200, 0.010, 0.0244),
]

print(f"{'I (A)':>8} {'v_lye':>8} {'v_c':>10} {'T (°C)':>8} {'HTO (%)':>8} {'P (MW)':>8} {'CBF?':>6} {'min_cbf':>10}")
print("=" * 85)

results = []
for I, v_lye, v_c in mapping:
    T_s, hto, P, state = simulate(I, v_lye, v_c, steps=15000)
    u_ref = np.array([I, v_lye, v_c])
    _, _, info = projector.project(u_ref, state, u_last=u_ref, active_mask=active_mask)
    proj = info['projection_needed']
    cbf = info.get('cbf_ref', info['cbf'])
    min_cbf = min(cbf[active_mask])
    status = "ACTIVE" if proj else "NO"
    print(f"{I:8.0f} {v_lye:8.3f} {v_c:10.4f} {T_s:8.2f} {hto:8.3f} {P:8.3f} {status:>6} {min_cbf:10.4f}")
    results.append((I, v_lye, v_c, T_s, hto, P, proj, min_cbf))

# Plotting
I_vals = [r[0] for r in results]
v_c_vals = [r[2] for r in results]
T_vals = [r[3] for r in results]
HTO_vals = [r[4] for r in results]
P_vals = [r[5] for r in results]

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

ax = axes[0, 0]
ax.plot(I_vals, v_c_vals, 'bo-', linewidth=2, markersize=8)
ax.set_xlabel('Current I (A)')
ax.set_ylabel('Coolant Flow v_c (m³/s)')
ax.set_title('v_c vs I (for T=80°C)')
ax.grid(True)

ax = axes[0, 1]
ax.plot(I_vals, T_vals, 'ro-', linewidth=2, markersize=8)
ax.axhline(y=80, color='g', linestyle='--', alpha=0.5, label='Target')
ax.axhline(y=90, color='r', linestyle='--', alpha=0.5, label='T_max')
ax.set_xlabel('Current I (A)')
ax.set_ylabel('Temperature (°C)')
ax.set_title('Steady-State Temperature')
ax.legend()
ax.grid(True)

ax = axes[1, 0]
ax.plot(I_vals, HTO_vals, 'go-', linewidth=2, markersize=8)
ax.axhline(y=2.0, color='r', linestyle='--', alpha=0.5, label='HTO_max (2%)')
ax.axhline(y=1.8, color='orange', linestyle='--', alpha=0.5, label='Effective (1.8%)')
ax.set_xlabel('Current I (A)')
ax.set_ylabel('HTO (%)')
ax.set_title('HTO at Steady State')
ax.legend()
ax.grid(True)

ax = axes[1, 1]
ax.plot(I_vals, P_vals, 'co-', linewidth=2, markersize=8)
ax.axhline(y=6.0, color='r', linestyle='--', alpha=0.5, label='P_max (6MW)')
ax.set_xlabel('Current I (A)')
ax.set_ylabel('Power (MW)')
ax.set_title('Stack Power')
ax.legend()
ax.grid(True)

plt.suptitle('T=80°C No-CBF Activation: I ∈ [2000, 7200]A\n'
             f'CBF Params: gamma=[3,2,100,100,5], h_margin_HTO=0.002 (eff=1.8%), lambda_u=[500,200,1000]',
             fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])

out_dir = 'output/single_stack/cbf_tests/t80_full_range'
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 't80_full_range.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nFigure saved: {out_path}")
