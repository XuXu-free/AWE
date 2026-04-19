#!/usr/bin/env python3
"""Search for T=80°C with small v_c > 0 (higher I, more O2, lower HTO)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from plant.single_stack_simulator import SingleStackSimulator
from controller.single_stack.cbf_projection import SingleStackCBFProjection

def test_point(I, v_lye, v_c, steps=25000):
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
    rho_vec=[50000]*5, h_margin_vec=[1.0, 0.005, 0.0, 0.0, 0.0],
    lambda_u_scale=[500.0, 200.0, 1000.0],
    soft_mask=[False, True, True, False, True], normalize=True,
)
active_mask = np.array([True, True, True, True, True])

# Strategy: small v_c allows higher I, which increases O2 production (lowers HTO)
# while keeping T ~ 80°C
candidates = []
for v_c in [0.0005, 0.001, 0.002, 0.003, 0.005]:
    for I in [1500, 1800, 2000, 2200, 2500, 2800, 3000]:
        for v_lye in [0.02, 0.03, 0.04, 0.05, 0.06]:
            candidates.append((I, v_lye, v_c))

print(f"Testing {len(candidates)} candidates with v_c > 0...\n")
matches = []
close = []

for I, v_lye, v_c in candidates:
    T_s, HTO, P, state = test_point(I, v_lye, v_c, steps=25000)
    u_ref = np.array([I, v_lye, v_c])
    _, _, info = projector.project(u_ref, state, u_last=u_ref, active_mask=active_mask)
    proj = info['projection_needed']
    cbf = info.get('cbf_ref', info['cbf'])
    h = info['h']

    if 78 <= T_s <= 82 and HTO <= 2.0 and not proj:
        matches.append((I, v_lye, v_c, T_s, HTO, P, proj, cbf, h))
        print(f">>> MATCH: I={I:.0f} v_lye={v_lye:.3f} v_c={v_c:.4f} | T={T_s:.2f}°C HTO={HTO:.3f}% P={P:.3f}MW")
    elif 75 <= T_s <= 85:
        close.append((I, v_lye, v_c, T_s, HTO, P, proj, cbf, h, abs(T_s-80)))

print(f"\nFound {len(matches)} exact matches.")
if matches:
    print("\nAll matches:")
    for m in matches:
        I, v_lye, v_c, T_s, HTO, P, proj, cbf, h = m
        print(f"  I={I:.0f}A, v_lye={v_lye:.3f}, v_c={v_c:.4f} => T={T_s:.2f}°C, HTO={HTO:.3f}%, P={P:.3f}MW, proj={proj}")

# Show top 10 closest non-matches
if close and not matches:
    close.sort(key=lambda x: x[9] + 10*max(0, x[4]-2.0) + (50 if x[6] else 0))
    print("\nTop 10 closest candidates:")
    for c in close[:10]:
        I, v_lye, v_c, T_s, HTO, P, proj, cbf, h, _ = c
        reason = []
        if HTO > 2.0: reason.append("HTO>2%")
        if proj: reason.append("CBF active")
        if not (78 <= T_s <= 82): reason.append(f"T={T_s:.1f}")
        print(f"  I={I:.0f} v_lye={v_lye:.3f} v_c={v_c:.4f} | T={T_s:.2f}°C HTO={HTO:.3f}% P={P:.3f}MW | {', '.join(reason)}")
