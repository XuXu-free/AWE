#!/usr/bin/env python3
"""Fast check around I=2000A with reduced steps."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from plant.single_stack_simulator import SingleStackSimulator
from controller.single_stack.cbf_projection import SingleStackCBFProjection

def simulate(I, v_lye, v_c, steps=10000):
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

print("Fast search around I=2000A (10000 steps ≈ 2000s):\n")
for I in [2000, 2200]:
    print(f"\nI={I}A:")
    for v_c in [0.0, 0.0005, 0.0008, 0.001, 0.0015]:
        for v_lye in [0.01, 0.02, 0.03]:
            T_s, hto, P, state = simulate(I, v_lye, v_c)
            if not (75 <= T_s <= 85):
                continue
            u_ref = np.array([I, v_lye, v_c])
            _, _, info = projector.project(u_ref, state, u_last=u_ref, active_mask=active_mask)
            proj = info['projection_needed']
            cbf = info.get('cbf_ref', info['cbf'])
            match = not proj and hto <= 2.0
            marker = " <<< MATCH!" if match else ""
            print(f"  v_c={v_c:.4f} v_lye={v_lye:.3f} | T={T_s:.2f} HTO={hto:.3f}% P={P:.3f}MW proj={proj} cbf_HTO={cbf[1]:.4f}{marker}")
