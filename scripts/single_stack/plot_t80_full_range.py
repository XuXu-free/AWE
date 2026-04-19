#!/usr/bin/env python3
"""
Full visualization suite for T=80°C no-CBF activation across I ∈ [2000, 7200]A.
"""
import os
import sys
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

mapping = [
    (2000, 0.010, 0.0008), (2400, 0.010, 0.0011), (2800, 0.010, 0.0015),
    (3200, 0.010, 0.0021), (3600, 0.010, 0.0027), (4000, 0.010, 0.0034),
    (4400, 0.010, 0.0043), (4800, 0.010, 0.0054), (5200, 0.010, 0.0068),
    (5600, 0.010, 0.0085), (6000, 0.010, 0.0107), (6400, 0.010, 0.0137),
    (6800, 0.010, 0.0181), (7200, 0.010, 0.0244),
]

I_vals, v_c_vals, T_vals, HTO_vals, P_vals = [], [], [], [], []
cbf_T_vals, cbf_HTO_vals, cbf_V_vals, cbf_P_vals, cbf_Tmin_vals = [], [], [], [], []
h_T_vals, h_HTO_vals, h_V_vals, h_P_vals, h_Tmin_vals = [], [], [], [], []

print("Collecting data...")
for I, v_lye, v_c in mapping:
    T_s, hto, P, state = simulate(I, v_lye, v_c, steps=15000)
    u_ref = np.array([I, v_lye, v_c])
    _, _, info = projector.project(u_ref, state, u_last=u_ref, active_mask=active_mask)
    cbf = info.get('cbf_ref', info['cbf'])
    h = info['h']

    I_vals.append(I)
    v_c_vals.append(v_c)
    T_vals.append(T_s)
    HTO_vals.append(hto)
    P_vals.append(P)
    cbf_T_vals.append(cbf[0])
    cbf_HTO_vals.append(cbf[1])
    cbf_V_vals.append(cbf[2])
    cbf_P_vals.append(cbf[3])
    cbf_Tmin_vals.append(cbf[4])
    h_T_vals.append(h[0])
    h_HTO_vals.append(h[1])
    h_V_vals.append(h[2])
    h_P_vals.append(h[3])
    h_Tmin_vals.append(h[4])

I_vals = np.array(I_vals)
v_c_vals = np.array(v_c_vals)
T_vals = np.array(T_vals)
HTO_vals = np.array(HTO_vals)
P_vals = np.array(P_vals)
cbf_T_vals = np.array(cbf_T_vals)
cbf_HTO_vals = np.array(cbf_HTO_vals)
cbf_V_vals = np.array(cbf_V_vals)
cbf_P_vals = np.array(cbf_P_vals)
cbf_Tmin_vals = np.array(cbf_Tmin_vals)
h_T_vals = np.array(h_T_vals)
h_HTO_vals = np.array(h_HTO_vals)
h_V_vals = np.array(h_V_vals)
h_P_vals = np.array(h_P_vals)
h_Tmin_vals = np.array(h_Tmin_vals)

out_dir = 'output/single_stack/cbf_tests/t80_full_range'
os.makedirs(out_dir, exist_ok=True)

# ---- Figure 1: Controls & Physical Quantities (2x3) ----
fig1 = plt.figure(figsize=(16, 10))
gs1 = fig1.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

ax = fig1.add_subplot(gs1[0, 0])
ax.plot(I_vals, v_c_vals * 1000, 'bo-', linewidth=2, markersize=8, markerfacecolor='white', markeredgewidth=2)
ax.set_xlabel('Current I (A)', fontsize=11)
ax.set_ylabel('Coolant Flow v_c (L/s)', fontsize=11)
ax.set_title('Coolant Flow vs Current', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

ax = fig1.add_subplot(gs1[0, 1])
ax.plot(I_vals, T_vals, 'ro-', linewidth=2, markersize=8, markerfacecolor='white', markeredgewidth=2)
ax.axhline(y=80, color='g', linestyle='--', alpha=0.7, label='Target 80°C')
ax.axhline(y=90, color='r', linestyle='--', alpha=0.7, label='T_max = 90°C')
ax.axhline(y=20, color='b', linestyle='--', alpha=0.7, label='T_min = 20°C')
ax.fill_between(I_vals, 78, 82, alpha=0.15, color='g', label='±2°C band')
ax.set_xlabel('Current I (A)', fontsize=11)
ax.set_ylabel('Stack Temperature (°C)', fontsize=11)
ax.set_title('Steady-State Temperature', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = fig1.add_subplot(gs1[0, 2])
ax.plot(I_vals, HTO_vals, 'go-', linewidth=2, markersize=8, markerfacecolor='white', markeredgewidth=2)
ax.axhline(y=2.0, color='r', linestyle='--', alpha=0.7, label='HTO_max = 2.0%')
ax.axhline(y=1.8, color='orange', linestyle='--', alpha=0.7, label='Effective = 1.8%')
ax.fill_between(I_vals, 0, 1.8, alpha=0.1, color='g')
ax.set_xlabel('Current I (A)', fontsize=11)
ax.set_ylabel('HTO (%)', fontsize=11)
ax.set_title('HTO at Steady State', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = fig1.add_subplot(gs1[1, 0])
ax.plot(I_vals, P_vals, 'co-', linewidth=2, markersize=8, markerfacecolor='white', markeredgewidth=2)
ax.axhline(y=6.0, color='r', linestyle='--', alpha=0.7, label='P_max = 6 MW')
ax.fill_between(I_vals, 0, 6.0, alpha=0.1, color='g')
ax.set_xlabel('Current I (A)', fontsize=11)
ax.set_ylabel('Stack Power (MW)', fontsize=11)
ax.set_title('Stack Power', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = fig1.add_subplot(gs1[1, 1])
ax.plot(I_vals, h_T_vals, 'b-o', linewidth=2, markersize=7, label='h_T', markerfacecolor='white', markeredgewidth=1.5)
ax.plot(I_vals, h_Tmin_vals, 'g-s', linewidth=2, markersize=7, label='h_Tmin', markerfacecolor='white', markeredgewidth=1.5)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('Current I (A)', fontsize=11)
ax.set_ylabel('h value', fontsize=11)
ax.set_title('h: Temperature Constraints', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = fig1.add_subplot(gs1[1, 2])
ax.plot(I_vals, h_P_vals / 1e6, 'c-o', linewidth=2, markersize=7, label='h_P (MW)', markerfacecolor='white', markeredgewidth=1.5)
ax.plot(I_vals, h_V_vals, 'm-s', linewidth=2, markersize=7, label='h_V (V)', markerfacecolor='white', markeredgewidth=1.5)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('Current I (A)', fontsize=11)
ax.set_ylabel('h value', fontsize=11)
ax.set_title('h: Power & Voltage Constraints', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

fig1.suptitle(
    'T=80°C No-CBF Activation: Operating Point Mapping\n'
    'I ∈ [2000, 7200]A | v_lye = 0.01 m³/s | h_margin_HTO = 0.002 (eff = 1.8%)',
    fontsize=14, fontweight='bold'
)
out1 = os.path.join(out_dir, 't80_full_range_overview.png')
fig1.savefig(out1, dpi=150, bbox_inches='tight')
plt.close(fig1)
print(f"Saved: {out1}")

# ---- Figure 2: CBF Values (2x3) ----
fig2 = plt.figure(figsize=(16, 10))
gs2 = fig2.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

ax = fig2.add_subplot(gs2[0, 0])
ax.plot(I_vals, cbf_T_vals, 'b-o', linewidth=2, markersize=8, markerfacecolor='white', markeredgewidth=2)
ax.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='CBF boundary')
ax.fill_between(I_vals, 0, cbf_T_vals, where=(cbf_T_vals > 0), alpha=0.2, color='g')
ax.set_xlabel('Current I (A)', fontsize=11)
ax.set_ylabel('CBF_T', fontsize=11)
ax.set_title('CBF: Temperature', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = fig2.add_subplot(gs2[0, 1])
ax.plot(I_vals, cbf_HTO_vals, 'r-o', linewidth=2, markersize=8, markerfacecolor='white', markeredgewidth=2)
ax.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='CBF boundary')
ax.fill_between(I_vals, 0, cbf_HTO_vals, where=(cbf_HTO_vals > 0), alpha=0.2, color='g')
ax.set_xlabel('Current I (A)', fontsize=11)
ax.set_ylabel('CBF_HTO', fontsize=11)
ax.set_title('CBF: HTO', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = fig2.add_subplot(gs2[0, 2])
ax.plot(I_vals, cbf_V_vals, 'm-o', linewidth=2, markersize=8, markerfacecolor='white', markeredgewidth=2)
ax.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='CBF boundary')
ax.fill_between(I_vals, 0, cbf_V_vals, where=(cbf_V_vals > 0), alpha=0.2, color='g')
ax.set_xlabel('Current I (A)', fontsize=11)
ax.set_ylabel('CBF_V', fontsize=11)
ax.set_title('CBF: Voltage', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = fig2.add_subplot(gs2[1, 0])
ax.plot(I_vals, cbf_P_vals, 'c-o', linewidth=2, markersize=8, markerfacecolor='white', markeredgewidth=2)
ax.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='CBF boundary')
ax.fill_between(I_vals, 0, cbf_P_vals, where=(cbf_P_vals > 0), alpha=0.2, color='g')
ax.set_xlabel('Current I (A)', fontsize=11)
ax.set_ylabel('CBF_P', fontsize=11)
ax.set_title('CBF: Power', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = fig2.add_subplot(gs2[1, 1])
ax.plot(I_vals, cbf_Tmin_vals, 'g-o', linewidth=2, markersize=8, markerfacecolor='white', markeredgewidth=2)
ax.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='CBF boundary')
ax.fill_between(I_vals, 0, cbf_Tmin_vals, where=(cbf_Tmin_vals > 0), alpha=0.2, color='g')
ax.set_xlabel('Current I (A)', fontsize=11)
ax.set_ylabel('CBF_Tmin', fontsize=11)
ax.set_title('CBF: Minimum Temperature', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = fig2.add_subplot(gs2[1, 2])
# Min CBF across all constraints
min_cbf = np.min([cbf_T_vals, cbf_HTO_vals, cbf_V_vals, cbf_P_vals, cbf_Tmin_vals], axis=0)
ax.plot(I_vals, min_cbf, 'k-o', linewidth=2.5, markersize=9, markerfacecolor='yellow', markeredgewidth=2, label='Min CBF (all constraints)')
ax.axhline(y=0, color='r', linestyle='--', alpha=0.7, linewidth=2, label='Activation threshold')
ax.fill_between(I_vals, 0, min_cbf, where=(min_cbf > 0), alpha=0.3, color='lime', label='Safe region')
ax.set_xlabel('Current I (A)', fontsize=11)
ax.set_ylabel('Min CBF Value', fontsize=11)
ax.set_title('Minimum CBF Across All Constraints', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

fig2.suptitle(
    'CBF Values at T=80°C Operating Points (All Positive = No Activation)\n'
    'CBF Params: gamma=[3,2,100,100,5] | h_margin_HTO=0.002 | soft_mask=[F,T,T,F,T]',
    fontsize=14, fontweight='bold'
)
out2 = os.path.join(out_dir, 't80_full_range_cbf_values.png')
fig2.savefig(out2, dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f"Saved: {out2}")

# ---- Figure 3: h Values & Safety Margins (2x3) ----
fig3 = plt.figure(figsize=(16, 10))
gs3 = fig3.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

ax = fig3.add_subplot(gs3[0, 0])
ax.plot(I_vals, h_T_vals, 'b-o', linewidth=2, markersize=8, markerfacecolor='white', markeredgewidth=2)
ax.axhline(y=0, color='r', linestyle='--', alpha=0.7)
ax.axhline(y=projector.h_margin_vec[0], color='orange', linestyle=':', alpha=0.7, label=f"h_margin = {projector.h_margin_vec[0]}")
ax.set_xlabel('Current I (A)', fontsize=11)
ax.set_ylabel('h_T (°C)', fontsize=11)
ax.set_title('h: Temperature Margin', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = fig3.add_subplot(gs3[0, 1])
ax.plot(I_vals, h_HTO_vals * 100, 'r-o', linewidth=2, markersize=8, markerfacecolor='white', markeredgewidth=2)
ax.axhline(y=0, color='r', linestyle='--', alpha=0.7)
ax.axhline(y=projector.h_margin_vec[1] * 100, color='orange', linestyle=':', alpha=0.7, label=f"h_margin = {projector.h_margin_vec[1]*100:.2f}%")
ax.set_xlabel('Current I (A)', fontsize=11)
ax.set_ylabel('h_HTO (%)', fontsize=11)
ax.set_title('h: HTO Margin', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = fig3.add_subplot(gs3[0, 2])
ax.plot(I_vals, h_V_vals, 'm-o', linewidth=2, markersize=8, markerfacecolor='white', markeredgewidth=2)
ax.axhline(y=0, color='r', linestyle='--', alpha=0.7)
ax.set_xlabel('Current I (A)', fontsize=11)
ax.set_ylabel('h_V (V)', fontsize=11)
ax.set_title('h: Voltage Margin', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

ax = fig3.add_subplot(gs3[1, 0])
ax.plot(I_vals, h_P_vals / 1e6, 'c-o', linewidth=2, markersize=8, markerfacecolor='white', markeredgewidth=2)
ax.axhline(y=0, color='r', linestyle='--', alpha=0.7)
ax.set_xlabel('Current I (A)', fontsize=11)
ax.set_ylabel('h_P (MW)', fontsize=11)
ax.set_title('h: Power Margin', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

ax = fig3.add_subplot(gs3[1, 1])
ax.plot(I_vals, h_Tmin_vals, 'g-o', linewidth=2, markersize=8, markerfacecolor='white', markeredgewidth=2)
ax.axhline(y=0, color='r', linestyle='--', alpha=0.7)
ax.set_xlabel('Current I (A)', fontsize=11)
ax.set_ylabel('h_Tmin (°C)', fontsize=11)
ax.set_title('h: Min Temperature Margin', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

ax = fig3.add_subplot(gs3[1, 2])
# Combined h margin visualization
min_h = np.min([h_T_vals, h_HTO_vals, h_V_vals, h_P_vals / 1e6, h_Tmin_vals], axis=0)
ax.plot(I_vals, min_h, 'k-o', linewidth=2.5, markersize=9, markerfacecolor='yellow', markeredgewidth=2, label='Min h (all constraints)')
ax.axhline(y=0, color='r', linestyle='--', alpha=0.7, linewidth=2, label='Boundary')
ax.fill_between(I_vals, 0, min_h, where=(min_h > 0), alpha=0.3, color='lime', label='Feasible region')
ax.set_xlabel('Current I (A)', fontsize=11)
ax.set_ylabel('Min h Value', fontsize=11)
ax.set_title('Minimum h Margin Across All Constraints', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

fig3.suptitle(
    'Safety Margins h(x,u) at T=80°C Operating Points\n'
    'All h > 0 confirms constraints are satisfied with margin',
    fontsize=14, fontweight='bold'
)
out3 = os.path.join(out_dir, 't80_full_range_h_margins.png')
fig3.savefig(out3, dpi=150, bbox_inches='tight')
plt.close(fig3)
print(f"Saved: {out3}")

# ---- Figure 4: 3D-style Combined Dashboard ----
fig4 = plt.figure(figsize=(18, 12))
gs4 = fig4.add_gridspec(3, 3, hspace=0.4, wspace=0.35)

# Row 1: Controls
ax1 = fig4.add_subplot(gs4[0, 0])
ax1.plot(I_vals, v_c_vals * 1000, 'b-o', linewidth=2.5, markersize=10, markerfacecolor='white', markeredgewidth=2)
ax1.set_xlabel('Current I (A)', fontsize=11)
ax1.set_ylabel('v_c (L/s)', fontsize=11)
ax1.set_title('Coolant Flow vs Current', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
for i, txt in enumerate(v_c_vals * 1000):
    ax1.annotate(f'{txt:.2f}', (I_vals[i], txt), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=8)

ax2 = fig4.add_subplot(gs4[0, 1])
ax2.plot(I_vals, np.full_like(I_vals, 0.01) * 1000, 'r-o', linewidth=2.5, markersize=10, markerfacecolor='white', markeredgewidth=2)
ax2.set_xlabel('Current I (A)', fontsize=11)
ax2.set_ylabel('v_lye (L/s)', fontsize=11)
ax2.set_title('Lye Flow (Constant)', fontsize=12, fontweight='bold')
ax2.set_ylim(8, 12)
ax2.grid(True, alpha=0.3)

ax3 = fig4.add_subplot(gs4[0, 2])
colors = plt.cm.viridis((P_vals - P_vals.min()) / (P_vals.max() - P_vals.min()))
ax3.scatter(I_vals, P_vals, c=colors, s=150, edgecolors='black', linewidths=1.5, zorder=5)
ax3.plot(I_vals, P_vals, 'k--', alpha=0.3, linewidth=1)
ax3.axhline(y=6.0, color='r', linestyle='--', alpha=0.7, label='P_max = 6MW')
ax3.set_xlabel('Current I (A)', fontsize=11)
ax3.set_ylabel('Power (MW)', fontsize=11)
ax3.set_title('Stack Power (Colored)', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Row 2: Physical
ax4 = fig4.add_subplot(gs4[1, 0])
ax4.plot(I_vals, T_vals, 'r-o', linewidth=2.5, markersize=10, markerfacecolor='white', markeredgewidth=2)
ax4.axhline(y=80, color='g', linestyle='--', alpha=0.7)
ax4.fill_between(I_vals, 78, 82, alpha=0.1, color='g')
ax4.set_xlabel('Current I (A)', fontsize=11)
ax4.set_ylabel('Temperature (°C)', fontsize=11)
ax4.set_title('Stack Temperature', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

ax5 = fig4.add_subplot(gs4[1, 1])
ax5.plot(I_vals, HTO_vals, 'g-o', linewidth=2.5, markersize=10, markerfacecolor='white', markeredgewidth=2)
ax5.axhline(y=2.0, color='r', linestyle='--', alpha=0.7, label='HTO_max = 2.0%')
ax5.axhline(y=1.8, color='orange', linestyle='--', alpha=0.7, label='Effective = 1.8%')
ax5.set_xlabel('Current I (A)', fontsize=11)
ax5.set_ylabel('HTO (%)', fontsize=11)
ax5.set_title('HTO Concentration', fontsize=12, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)

ax6 = fig4.add_subplot(gs4[1, 2])
# Thermal balance: Q_gen vs Q_cool approx
# Use v_c as proxy for cooling
ax6.bar(I_vals, v_c_vals * 1000, width=250, color='steelblue', edgecolor='navy', alpha=0.8)
ax6.set_xlabel('Current I (A)', fontsize=11)
ax6.set_ylabel('Coolant Flow (L/s)', fontsize=11)
ax6.set_title('Cooling Requirement', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3)

# Row 3: CBF & Safety
ax7 = fig4.add_subplot(gs4[2, 0])
ax7.plot(I_vals, cbf_T_vals, 'b-o', linewidth=2.5, markersize=10, markerfacecolor='white', markeredgewidth=2, label='CBF_T')
ax7.plot(I_vals, cbf_HTO_vals, 'r-s', linewidth=2.5, markersize=10, markerfacecolor='white', markeredgewidth=2, label='CBF_HTO')
ax7.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax7.fill_between(I_vals, 0, np.minimum(cbf_T_vals, cbf_HTO_vals), alpha=0.2, color='lime')
ax7.set_xlabel('Current I (A)', fontsize=11)
ax7.set_ylabel('CBF Value', fontsize=11)
ax7.set_title('Key CBF Values (T & HTO)', fontsize=12, fontweight='bold')
ax7.legend(fontsize=9)
ax7.grid(True, alpha=0.3)

ax8 = fig4.add_subplot(gs4[2, 1])
min_cbf_all = np.min([cbf_T_vals, cbf_HTO_vals, cbf_V_vals, cbf_P_vals, cbf_Tmin_vals], axis=0)
ax8.fill_between(I_vals, 0, min_cbf_all, alpha=0.4, color='lime', label='Safety margin')
ax8.plot(I_vals, min_cbf_all, 'k-o', linewidth=3, markersize=12, markerfacecolor='yellow', markeredgewidth=2.5, label='Min CBF')
ax8.axhline(y=0, color='r', linestyle='--', alpha=0.8, linewidth=2, label='Activation threshold')
ax8.set_xlabel('Current I (A)', fontsize=11)
ax8.set_ylabel('Min CBF Value', fontsize=11)
ax8.set_title('Overall CBF Safety Margin', fontsize=12, fontweight='bold')
ax8.legend(fontsize=9)
ax8.grid(True, alpha=0.3)

ax9 = fig4.add_subplot(gs4[2, 2])
# Summary table as text
ax9.axis('off')
table_data = []
for i in range(len(I_vals)):
    table_data.append([f"{I_vals[i]:.0f}", f"{v_c_vals[i]*1000:.3f}", f"{T_vals[i]:.1f}", f"{HTO_vals[i]:.2f}", f"{P_vals[i]:.2f}", f"{min_cbf_all[i]:.3f}"])
table = ax9.table(
    cellText=table_data,
    colLabels=['I (A)', 'v_c (L/s)', 'T (°C)', 'HTO (%)', 'P (MW)', 'min CBF'],
    loc='center',
    cellLoc='center',
    colColours=['#4472C4']*6,
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.8)
for key, cell in table.get_celld().items():
    if key[0] == 0:
        cell.set_text_props(color='white', fontweight='bold')
        cell.set_facecolor('#4472C4')
    else:
        cell.set_facecolor('#E7E6E6' if key[0] % 2 == 0 else 'white')
ax9.set_title('Operating Point Summary', fontsize=12, fontweight='bold', pad=20)

fig4.suptitle(
    'T=80°C No-CBF Activation: Complete Dashboard\n'
    'I ∈ [2000, 7200]A | v_lye = 0.01 m³/s | CBF Params: gamma=[3,2,100,100,5], h_margin_HTO=0.002',
    fontsize=15, fontweight='bold', y=0.98
)
out4 = os.path.join(out_dir, 't80_full_range_dashboard.png')
fig4.savefig(out4, dpi=150, bbox_inches='tight')
plt.close(fig4)
print(f"Saved: {out4}")

print("\nAll figures saved to:", out_dir)
