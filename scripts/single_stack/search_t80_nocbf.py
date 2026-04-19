#!/usr/bin/env python3
"""
Fast search for T=80°C operating point without CBF activation.
Reduced steps for speed, focused grid.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from plant.single_stack_simulator import SingleStackSimulator
from controller.single_stack.cbf_projection import SingleStackCBFProjection


def find_steady_state(I, v_lye, v_c, max_steps=30000, dt=0.2):
    sim = SingleStackSimulator()
    sim.reset()
    u = np.array([I, v_lye, v_c])
    for _ in range(max_steps):
        sim.step(u)
    T_s = sim.state[1] - 273.15
    T_sep = sim.state[2] - 273.15
    n_gas = sim.state[6]
    hto = (n_gas * sim.R * sim.state[2]) / (sim.P_sys * sim.V_sep_gas) * 100
    _, U_cell, _ = sim._calculate_electrochemical_properties(I, sim.state[1])
    Power = U_cell * I * sim.N_cell / 1e6
    return {
        'T_s': T_s, 'T_sep': T_sep, 'HTO': hto,
        'Power': Power, 'U_cell': U_cell,
        'n_gas': n_gas, 'state': sim.state.copy()
    }


def main():
    projector = SingleStackCBFProjection(
        dt=60.0,
        gamma_vec=[3.0, 2.0, 100.0, 100.0, 5.0],
        rho_vec=[50000, 50000, 50000, 50000, 10000],
        h_margin_vec=[1.0, 0.005, 0.0, 0.0, 0.0],
        lambda_u_scale=[500.0, 200.0, 1000.0],
        soft_mask=[False, True, True, False, True],
        active_mask=None,
        normalize=True,
    )

    active_mask = np.array([True, True, True, True, True])

    print("=" * 80)
    print("Fast Search: T≈80°C without CBF activation")
    print("=" * 80)

    # Focus: v_c=0 (no cooling) or very small, low I, varying v_lye
    # Thermal steady state at v_c=0, I~1000-1500A should be around 70-85°C

    best = None
    best_score = float('inf')

    for v_c in [0.0, 0.001]:
        print(f"\n--- v_c = {v_c} ---")
        for I in np.arange(600, 2201, 200):
            for v_lye in [0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]:
                ss = find_steady_state(I, v_lye, v_c, max_steps=30000)
                T_s = ss['T_s']
                HTO = ss['HTO']
                Power = ss['Power']

                u_ref = np.array([I, v_lye, v_c])
                _, _, info = projector.project(u_ref, ss['state'], u_last=u_ref, active_mask=active_mask)
                proj_needed = info['projection_needed']
                cbf_ref = info['cbf_ref']
                h = info['h']

                marker = ""
                if 78 <= T_s <= 82 and HTO <= 2.0 and not proj_needed:
                    marker = " <<< MATCH!"
                elif 78 <= T_s <= 82 and HTO <= 2.0:
                    marker = " < close (CBF active)"
                elif 75 <= T_s <= 85:
                    marker = " ."

                print(f"  I={I:4.0f} v_lye={v_lye:.3f} | T={T_s:5.2f}°C HTO={HTO:5.3f}% P={Power:5.3f}MW "
                      f"cbfT={cbf_ref[0]:6.3f} cbfHTO={cbf_ref[1]:7.4f} proj={proj_needed}{marker}")

                # Score: prefer T close to 80, HTO < 2, no projection
                if HTO <= 2.5 and T_s > 60:
                    score = abs(T_s - 80) + 10 * max(0, HTO - 2.0) + (50 if proj_needed else 0)
                    if score < best_score:
                        best_score = score
                        best = {
                            'I': I, 'v_lye': v_lye, 'v_c': v_c,
                            'T_s': T_s, 'HTO': HTO, 'Power': Power,
                            'proj_needed': proj_needed,
                            'cbf_ref': cbf_ref, 'h': h
                        }

    if best:
        print("\n" + "=" * 80)
        print("BEST CANDIDATE:")
        print("=" * 80)
        print(f"  I={best['I']:.0f}A, v_lye={best['v_lye']:.3f}, v_c={best['v_c']:.3f}")
        print(f"  T={best['T_s']:.2f}°C, HTO={best['HTO']:.3f}%, Power={best['Power']:.3f}MW")
        print(f"  CBF projection needed: {best['proj_needed']}")
        print(f"  cbf_ref = [{', '.join([f'{v:.4f}' for v in best['cbf_ref']])}]")
        print(f"  h       = [{', '.join([f'{v:.4f}' for v in best['h']])}]")


if __name__ == '__main__':
    main()
