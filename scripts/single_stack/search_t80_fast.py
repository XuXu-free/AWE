#!/usr/bin/env python3
"""Quick test of promising T=80°C candidates."""
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

# Most promising candidates from physics: v_c=0, moderate I, moderate-to-high v_lye
candidates = [
    (1200, 0.04, 0.0),
    (1400, 0.05, 0.0),
    (1000, 0.03, 0.0),
    (1200, 0.05, 0.0),
    (1400, 0.04, 0.0),
    (1600, 0.06, 0.0),
    (1000, 0.05, 0.0),
    (800,  0.03, 0.0),
]

print("Testing promising candidates for T≈80°C without CBF activation:\n")
for I, v_lye, v_c in candidates:
    T_s, HTO, P, state = test_point(I, v_lye, v_c, steps=25000)
    u_ref = np.array([I, v_lye, v_c])
    _, _, info = projector.project(u_ref, state, u_last=u_ref, active_mask=active_mask)
    proj = info['projection_needed']
    cbf = info['cbf_ref']
    h = info['h']

    match = (78 <= T_s <= 82) and (HTO <= 2.0) and (not proj)
    marker = " <<< MATCH!" if match else ""
    print(f"I={I:4.0f} v_lye={v_lye:.2f} v_c={v_c:.1f} | T={T_s:5.2f}°C HTO={HTO:5.3f}% P={P:5.3f}MW | "
          f"proj={proj} cbf_T={cbf[0]:6.3f} cbf_HTO={cbf[1]:7.4f}{marker}")
