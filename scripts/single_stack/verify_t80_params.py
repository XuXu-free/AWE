#!/usr/bin/env python3
"""Verify T=80°C without CBF by adjusting h_margin_HTO."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from plant.single_stack_simulator import SingleStackSimulator
from controller.single_stack.cbf_projection import SingleStackCBFProjection

def test(I, v_lye, v_c, steps=25000):
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

# Test both parameter sets
for h_margin_HTO in [0.005, 0.002, 0.0]:
    print(f"\n{'='*70}")
    print(f"h_margin_HTO = {h_margin_HTO} (effective HTO limit = {2.0 - h_margin_HTO*100:.2f}%)")
    print(f"{'='*70}")

    projector = SingleStackCBFProjection(
        dt=60.0, gamma_vec=[3.0, 2.0, 100.0, 100.0, 5.0],
        rho_vec=[50000]*5, h_margin_vec=[1.0, h_margin_HTO, 0.0, 0.0, 0.0],
        lambda_u_scale=[500.0, 200.0, 1000.0],
        soft_mask=[False, True, True, False, True], normalize=True,
    )
    active_mask = np.array([True, True, True, True, True])

    candidates = [
        (2500, 0.04, 0.001),
        (2600, 0.04, 0.001),
        (2700, 0.04, 0.001),
        (2800, 0.04, 0.001),
        (2500, 0.05, 0.001),
        (2600, 0.05, 0.001),
    ]

    for I, v_lye, v_c in candidates:
        T_s, HTO, P, state = test(I, v_lye, v_c, steps=25000)
        u_ref = np.array([I, v_lye, v_c])
        _, _, info = projector.project(u_ref, state, u_last=u_ref, active_mask=active_mask)
        proj = info['projection_needed']
        cbf = info.get('cbf_ref', info['cbf'])
        h = info['h']
        match = (78 <= T_s <= 82) and (HTO <= 2.0) and (not proj)
        marker = " <<< MATCH!" if match else ""
        print(f"I={I:4.0f} v_lye={v_lye:.2f} v_c={v_c:.3f} | T={T_s:5.2f}°C HTO={HTO:5.3f}% P={P:5.3f}MW | "
              f"proj={proj} cbf_T={cbf[0]:6.3f} cbf_HTO={cbf[1]:7.4f}{marker}")
