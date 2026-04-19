#!/usr/bin/env python3
"""
Search for T=80°C, NO CBF activation across I ∈ [2000, 7200]A.
Find v_lye, v_c and CBF params for each I.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from plant.single_stack_simulator import SingleStackSimulator
from controller.single_stack.cbf_projection import SingleStackCBFProjection

def simulate_to_steady(I, v_lye, v_c, steps=25000):
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

def find_vc_for_temp(I, v_lye, target_T=80.0, tol=0.2):
    """Binary search for v_c that gives T ≈ target_T."""
    v_c_low, v_c_high = 0.0, 1.0
    best_v_c, best_T, best_info = None, None, None
    for _ in range(20):
        v_c_mid = (v_c_low + v_c_high) / 2
        T_s, hto, P, state = simulate_to_steady(I, v_lye, v_c_mid)
        if best_T is None or abs(T_s - target_T) < abs(best_T - target_T):
            best_v_c, best_T = v_c_mid, T_s
            best_info = (T_s, hto, P, state)
        if T_s > target_T:
            v_c_low = v_c_mid  # need more cooling
        else:
            v_c_high = v_c_mid  # too cold
        if abs(T_s - target_T) < tol:
            break
    return best_v_c, best_info

def check_cbf(state, u_ref, projector, active_mask):
    _, _, info = projector.project(u_ref, state, u_last=u_ref, active_mask=active_mask)
    return info['projection_needed'], info.get('cbf_ref', info['cbf']), info['h']

# Use h_margin_HTO=0.002 (eff limit=1.8%) as baseline
projector = SingleStackCBFProjection(
    dt=60.0, gamma_vec=[3.0, 2.0, 100.0, 100.0, 5.0],
    rho_vec=[50000]*5, h_margin_vec=[1.0, 0.002, 0.0, 0.0, 0.0],
    lambda_u_scale=[500.0, 200.0, 1000.0],
    soft_mask=[False, True, True, False, True], normalize=True,
)
active_mask = np.array([True, True, True, True, True])

print(f"{'I':>6} {'v_lye':>7} {'v_c':>7} {'T(°C)':>7} {'HTO%':>7} {'P(MW)':>7} {'CBF?':>6} {'min_cbf':>8}")
print("=" * 75)

results = []

for I in np.arange(2000, 7201, 400):
    # For each I, find v_c that gives T=80 with some v_lye
    # HTO increases with v_lye, decreases with I
    # Try different v_lye values to find one where HTO < 1.8%
    found = False
    for v_lye in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]:
        v_c, info = find_vc_for_temp(I, v_lye, target_T=80.0, tol=0.3)
        if v_c is None:
            continue
        T_s, hto, P, state = info
        if abs(T_s - 80.0) > 1.0:
            continue
        u_ref = np.array([I, v_lye, v_c])
        proj_needed, cbf, h = check_cbf(state, u_ref, projector, active_mask)
        min_cbf = min(cbf[active_mask])
        if not proj_needed and hto <= 2.0:
            print(f"{I:6.0f} {v_lye:7.3f} {v_c:7.4f} {T_s:7.2f} {hto:7.3f} {P:7.3f} {'NO':>6} {min_cbf:8.4f}")
            results.append((I, v_lye, v_c, T_s, hto, P, min_cbf))
            found = True
            break
    if not found:
        # Try with relaxed h_margin or report closest
        print(f"{I:6.0f} {'---':>7} {'---':>7} {'---':>7} {'---':>7} {'---':>7} {'YES':>6} {'---':>8}")

print("\n" + "=" * 75)
print("Summary of found operating points (CBF inactive, T≈80°C):")
for r in results:
    I, v_lye, v_c, T_s, hto, P, min_cbf = r
    print(f"  I={I:.0f}A, v_lye={v_lye:.3f}, v_c={v_c:.4f} => T={T_s:.2f}°C, HTO={hto:.3f}%, P={P:.3f}MW, min_cbf={min_cbf:.4f}")
