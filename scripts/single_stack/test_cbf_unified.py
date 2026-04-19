"""
Unified CBF Test and Visualization Script

Combines single CBF testing and comparison visualization in one script.

Usage:
    # Run single CBF test
    python test_cbf_unified.py --mode single --gamma_vec 1 5 1 1 1

    # Run comparison (original vs normalized)
    python test_cbf_unified.py --mode compare

    # Run with custom parameters
    python test_cbf_unified.py --mode single --gamma_vec 1 5 1 1 1 --duration 7200

    # Run low power HTO test (600A reference, tests CBF reaction to HTO buildup)
    python test_cbf_unified.py --scenario low_power_hto
"""

import os
import sys
import io
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from controller.single_stack.cbf_projection import SingleStackCBFProjection
from plant.single_stack_simulator import SingleStackSimulator
import casadi as ca

# Suppress CasADi/IPOPT solver output
original_opti_solve = ca.Opti.solve
def silent_solve(self):
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        return original_opti_solve(self)
    finally:
        sys.stdout = old_stdout
ca.Opti.solve = silent_solve


def run_simulation(projector, duration=36000, u_ref=None, active_mask=None):
    """Run simulation with given CBF projector and optional per-step active_mask."""
    sim = SingleStackSimulator(sim_dt=0.2)
    sim.reset()

    dt = 0.2
    steps = int(duration / dt)
    ctrl_steps = int(60.0 / dt)

    if u_ref is None:
        u_ref = np.array([600.0, 0.005, 0.0])
    current_action = u_ref.copy()
    last_action = current_action.copy()  # For smoothness penalty

    history = {
        't': [], 'I': [], 'I_ref': [], 'v_lye': [], 'v_lye_ref': [],
        'v_c': [], 'v_c_ref': [],
        'HTO': [], 'T_s': [], 'h_HTO': [], 'projection_success': [],
        'Power': [], 'U_cell': [],
        # CBF values
        'h_raw': [], 'h_norm': [], 'lie_deriv_raw': [], 'lie_deriv_norm': [],
        'cbf_condition': [],
        # Slack values (recorded at control steps)
        'slack': [],
        # Cost composition (recorded at control steps)
        'cost_ref': [], 'cost_delta': [], 'cost_slack': [], 'cost_total': [],
        # Solver timing (recorded at control steps)
        'solve_time_ms': [],
        # Active mask
        'active_mask': []
    }

    # Precompute control weights for cost analysis (matching projector logic)
    u_range = np.array([
        projector.I_max - projector.I_min,
        projector.v_lye_max - projector.v_lye_min,
        projector.v_c_max - projector.v_c_min
    ])
    u_weights = 1.0 / (u_range ** 2)
    lambda_u = projector.lambda_u

    last_costs = {'ref': 0.0, 'delta': 0.0, 'slack': 0.0, 'total': 0.0}
    last_slack = np.zeros(5)

    for i in range(steps):
        t = i * dt

        if i % ctrl_steps == 0:
            state = sim.state
            t0 = time.time()
            current_action, success, info = projector.project(u_ref, state, u_last=last_action, active_mask=active_mask)
            solve_time_ms = (time.time() - t0) * 1000.0

            # Compute cost composition (before updating last_action)
            s = info.get('slack', np.zeros(5))
            last_slack = s.copy()
            ref_cost = np.sum(u_weights * (current_action - u_ref)**2)
            delta_cost = np.sum(lambda_u * (current_action - last_action)**2)
            slack_cost = np.sum(projector.rho_vec * s**2)
            last_costs = {
                'ref': ref_cost,
                'delta': delta_cost,
                'slack': slack_cost,
                'total': ref_cost + delta_cost + slack_cost
            }

            last_action = current_action.copy()

            if not success and i % 500 == 0:  # Print diagnostics every 100s on failure
                # Note: success=False from solver doesn't mean constraint violation
                # The system may still be safe if solution is feasible (check max_slack)
                max_slack = info.get('max_slack', 0)
                if max_slack > 10.0:  # Only warn if severely infeasible
                    print(f"    Warning: CBF solver reported failure at t={t/60:.1f}min, max_slack={max_slack:.4f}")

        if i % 50 == 0:
            state = sim.state
            n_gas = state[6]
            T_sep = state[2]
            hto = (n_gas * projector.R * T_sep) / (projector.P_sys * projector.V_sep_gas)

            # Compute CBF values at current state and action
            lie_deriv_raw, h_raw, _, _ = projector._compute_lie_derivative(state, current_action)

            # Normalize
            h_norm = h_raw / projector.h_scales
            lie_deriv_norm = lie_deriv_raw / np.maximum(np.abs(projector.lie_deriv_scales), 1e-10)

            # CBF condition
            if projector.normalize:
                cbf_cond = lie_deriv_norm + projector.gamma_vec * h_norm
            else:
                cbf_cond = lie_deriv_raw + projector.gamma * h_raw

            history['t'].append(t / 60)
            history['I'].append(current_action[0])
            history['I_ref'].append(u_ref[0])
            history['v_lye'].append(current_action[1])
            history['v_lye_ref'].append(u_ref[1])
            history['v_c'].append(current_action[2])
            history['v_c_ref'].append(u_ref[2])
            # Compute actual power using simulator's electrochemical model
            I_rec = current_action[0]
            T_s_rec = state[1]
            _, U_cell_rec, _ = sim._calculate_electrochemical_properties(I_rec, T_s_rec)
            Power_rec = U_cell_rec * I_rec * sim.N_cell / 1e6

            history['HTO'].append(hto * 100)
            history['T_s'].append(state[1] - 273.15)
            history['h_HTO'].append(0.02 - hto)
            history['Power'].append(Power_rec)
            history['U_cell'].append(U_cell_rec)
            history['projection_success'].append(success)
            history['h_raw'].append(h_raw.copy())
            history['h_norm'].append(h_norm.copy())
            history['lie_deriv_raw'].append(lie_deriv_raw.copy())
            history['lie_deriv_norm'].append(lie_deriv_norm.copy())
            history['cbf_condition'].append(cbf_cond.copy())
            history['cost_ref'].append(last_costs['ref'])
            history['cost_delta'].append(last_costs['delta'])
            history['cost_slack'].append(last_costs['slack'])
            history['cost_total'].append(last_costs['total'])
            history['solve_time_ms'].append(solve_time_ms)
            history['active_mask'].append(active_mask.copy() if active_mask is not None else np.array([True]*5))
            history['slack'].append(last_slack.copy())

        sim.step(current_action)

    return history


def plot_single_test(history, config, output_path):
    """Plot single CBF test results with detailed CBF visualization"""
    fig = plt.figure(figsize=(40, 30))
    gs = fig.add_gridspec(10, 6, hspace=0.35, wspace=0.3)

    t = np.array(history['t'])
    h_raw_arr = np.array(history['h_raw'])
    h_norm_arr = np.array(history['h_norm'])
    lie_deriv_raw_arr = np.array(history['lie_deriv_raw'])
    lie_deriv_norm_arr = np.array(history['lie_deriv_norm'])
    cbf_arr = np.array(history['cbf_condition'])

    # Find failure points
    failure_indices = [i for i, success in enumerate(history['projection_success']) if not success]
    failure_times = [t[i] for i in failure_indices] if failure_indices else []

    # Row 1: Control inputs
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, history['I'], 'b-', linewidth=2, label='Actual')
    ax1.plot(t, history['I_ref'], 'r--', linewidth=1.5, alpha=0.7, label='Reference')
    if failure_times:
        ax1.scatter(failure_times, [history['I'][i] for i in failure_indices],
                   color='red', marker='x', s=100, zorder=5, label='Solve Failed')
    ax1.set_title('Current')
    ax1.set_ylabel('A')
    ax1.legend()
    ax1.grid(True)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, np.array(history['v_lye']) * 1000, 'b-', linewidth=2)
    ax2.plot(t, np.array(history['v_lye_ref']) * 1000, 'r--', linewidth=1.5, alpha=0.7)
    if failure_times:
        ax2.scatter(failure_times, [history['v_lye'][i] * 1000 for i in failure_indices],
                   color='red', marker='x', s=100, zorder=5)
    ax2.set_title('Lye Flow')
    ax2.set_ylabel('L/s')
    ax2.grid(True)

    ax2b = fig.add_subplot(gs[0, 2])
    # Use actual power computed from simulator's electrochemical model
    power_vals = history['Power']
    ax2b.plot(t, power_vals, 'c-', linewidth=2)
    ax2b.axhline(y=6.0, color='r', linestyle='--', label='P_max (6MW)')
    ax2b.set_title('Stack Power')
    ax2b.set_ylabel('MW')
    ax2b.legend()
    ax2b.grid(True)

    ax_vc = fig.add_subplot(gs[0, 3])
    ax_vc.plot(t, np.array(history['v_c']) * 1000, 'b-', linewidth=2, label='Actual')
    ax_vc.plot(t, np.array(history['v_c_ref']) * 1000, 'r--', linewidth=1.5, alpha=0.7, label='Reference')
    if failure_times:
        ax_vc.scatter(failure_times, [history['v_c'][i] * 1000 for i in failure_indices],
                     color='red', marker='x', s=100, zorder=5)
    ax_vc.set_title('Coolant Flow')
    ax_vc.set_ylabel('L/s')
    ax_vc.legend()
    ax_vc.grid(True)

    # Row 2: Physical quantities
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(t, history['HTO'], 'b-', linewidth=2)
    ax3.axhline(y=2.0, color='r', linestyle='--', label='Limit (2%)')
    effective_limit = 2.0 - config.get('h_margin_vec', [0,0,0,0,0])[1] * 100
    if effective_limit < 2.0:
        ax3.axhline(y=effective_limit, color='orange', linestyle=':', label=f'Effective ({effective_limit:.1f}%)')
    ax3.fill_between(t, 2.0, 3.0, alpha=0.1, color='red', where=(np.array(history['HTO']) > 2.0))
    ax3.set_title('HTO (%)')
    ax3.set_ylabel('%')
    ax3.legend()
    ax3.grid(True)
    ax3.set_ylim(0, 3)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(t, history['T_s'], 'r-', linewidth=2)
    ax4.axhline(y=90, color='r', linestyle='--', label='T_max')
    if failure_times:
        ax4.scatter(failure_times, [history['T_s'][i] for i in failure_indices],
                   color='red', marker='x', s=100, zorder=5)
    ax4.set_title('Stack Temperature')
    ax4.set_ylabel('°C')
    ax4.legend()
    ax4.grid(True)

    ax4b = fig.add_subplot(gs[1, 2])
    # Use actual cell voltage from simulator's electrochemical model
    voltage_vals = history['U_cell']
    ax4b.plot(t, voltage_vals, 'm-', linewidth=2)
    ax4b.axhline(y=2.2, color='r', linestyle='--', label='U_max (2.2V)')
    ax4b.set_title('Cell Voltage')
    ax4b.set_ylabel('V')
    ax4b.legend()
    ax4b.grid(True)

    # Row 3-4: Raw h and Normalized h - side by side (2 columns x 3 rows each)
    # Column 1: Raw h
    ax_h_T = fig.add_subplot(gs[2, 0])
    ax_h_T.plot(t, h_raw_arr[:, 0], 'b-', linewidth=1.5)
    ax_h_T.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax_h_T.fill_between(t, -100, 0, alpha=0.2, color='red', where=(h_raw_arr[:, 0] < 0))
    ax_h_T.set_title('Raw h_T (Temperature)')
    ax_h_T.set_ylabel('h')
    ax_h_T.grid(True)

    ax_h_HTO = fig.add_subplot(gs[3, 0])
    ax_h_HTO.plot(t, h_raw_arr[:, 1], 'r-', linewidth=1.5)
    ax_h_HTO.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax_h_HTO.fill_between(t, -0.01, 0, alpha=0.2, color='red', where=(h_raw_arr[:, 1] < 0))
    ax_h_HTO.set_title('Raw h_HTO')
    ax_h_HTO.set_ylabel('h')
    ax_h_HTO.grid(True)

    ax_h_V = fig.add_subplot(gs[4, 0])
    ax_h_V.plot(t, h_raw_arr[:, 2], 'm-', linewidth=1.5)
    ax_h_V.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax_h_V.fill_between(t, -1, 0, alpha=0.2, color='red', where=(h_raw_arr[:, 2] < 0))
    ax_h_V.set_title('Raw h_V (Voltage)')
    ax_h_V.set_ylabel('h')
    ax_h_V.grid(True)

    ax_h_P = fig.add_subplot(gs[5, 0])
    ax_h_P.plot(t, h_raw_arr[:, 3], 'c-', linewidth=1.5)
    ax_h_P.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax_h_P.fill_between(t, -1e7, 0, alpha=0.2, color='red', where=(h_raw_arr[:, 3] < 0))
    ax_h_P.set_title('Raw h_P (Power)')
    ax_h_P.set_ylabel('h')
    ax_h_P.grid(True)

    ax_h_Tmin = fig.add_subplot(gs[6, 0])
    ax_h_Tmin.plot(t, h_raw_arr[:, 4], 'g-', linewidth=1.5)
    ax_h_Tmin.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax_h_Tmin.fill_between(t, -100, 0, alpha=0.2, color='red', where=(h_raw_arr[:, 4] < 0))
    ax_h_Tmin.set_title('Raw h_Tmin (Min Temperature)')
    ax_h_Tmin.set_ylabel('h')
    ax_h_Tmin.set_xlabel('Time (min)')
    ax_h_Tmin.grid(True)

    # Column 2: Normalized h
    ax_hn_T = fig.add_subplot(gs[2, 1])
    ax_hn_T.plot(t, h_norm_arr[:, 0], 'b-', linewidth=1.5)
    ax_hn_T.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax_hn_T.fill_between(t, -2, 0, alpha=0.2, color='red', where=(h_norm_arr[:, 0] < 0))
    ax_hn_T.set_title('Normalized h_T')
    ax_hn_T.set_ylabel('h_norm')
    ax_hn_T.grid(True)

    ax_hn_HTO = fig.add_subplot(gs[3, 1])
    ax_hn_HTO.plot(t, h_norm_arr[:, 1], 'r-', linewidth=1.5)
    ax_hn_HTO.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax_hn_HTO.fill_between(t, -2, 0, alpha=0.2, color='red', where=(h_norm_arr[:, 1] < 0))
    ax_hn_HTO.set_title('Normalized h_HTO')
    ax_hn_HTO.set_ylabel('h_norm')
    ax_hn_HTO.grid(True)

    ax_hn_V = fig.add_subplot(gs[4, 1])
    ax_hn_V.plot(t, h_norm_arr[:, 2], 'm-', linewidth=1.5)
    ax_hn_V.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax_hn_V.fill_between(t, -2, 0, alpha=0.2, color='red', where=(h_norm_arr[:, 2] < 0))
    ax_hn_V.set_title('Normalized h_V')
    ax_hn_V.set_ylabel('h_norm')
    ax_hn_V.grid(True)

    ax_hn_P = fig.add_subplot(gs[5, 1])
    ax_hn_P.plot(t, h_norm_arr[:, 3], 'c-', linewidth=1.5)
    ax_hn_P.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax_hn_P.fill_between(t, -2, 0, alpha=0.2, color='red', where=(h_norm_arr[:, 3] < 0))
    ax_hn_P.set_title('Normalized h_P')
    ax_hn_P.set_ylabel('h_norm')
    ax_hn_P.grid(True)

    ax_hn_Tmin = fig.add_subplot(gs[6, 1])
    ax_hn_Tmin.plot(t, h_norm_arr[:, 4], 'g-', linewidth=1.5)
    ax_hn_Tmin.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax_hn_Tmin.fill_between(t, -2, 0, alpha=0.2, color='red', where=(h_norm_arr[:, 4] < 0))
    ax_hn_Tmin.set_title('Normalized h_Tmin')
    ax_hn_Tmin.set_ylabel('h_norm')
    ax_hn_Tmin.set_xlabel('Time (min)')
    ax_hn_Tmin.grid(True)

    # Column 2: L_f h (raw)
    ax_ld_T = fig.add_subplot(gs[2, 2])
    ax_ld_T.plot(t, lie_deriv_raw_arr[:, 0], 'b-', linewidth=1.5)
    ax_ld_T.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax_ld_T.set_title('L_f h - Temperature')
    ax_ld_T.set_ylabel('L_f h')
    ax_ld_T.grid(True)

    ax_ld_HTO = fig.add_subplot(gs[3, 2])
    ax_ld_HTO.plot(t, lie_deriv_raw_arr[:, 1], 'r-', linewidth=1.5)
    ax_ld_HTO.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax_ld_HTO.set_title('L_f h - HTO')
    ax_ld_HTO.set_ylabel('L_f h')
    ax_ld_HTO.grid(True)

    ax_ld_V = fig.add_subplot(gs[4, 2])
    ax_ld_V.plot(t, lie_deriv_raw_arr[:, 2], 'm-', linewidth=1.5)
    ax_ld_V.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax_ld_V.set_title('L_f h - Voltage')
    ax_ld_V.set_ylabel('L_f h')
    ax_ld_V.grid(True)

    ax_ld_P = fig.add_subplot(gs[5, 2])
    ax_ld_P.plot(t, lie_deriv_raw_arr[:, 3], 'c-', linewidth=1.5)
    ax_ld_P.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax_ld_P.set_title('L_f h - Power')
    ax_ld_P.set_ylabel('L_f h')
    ax_ld_P.grid(True)

    ax_ld_Tmin = fig.add_subplot(gs[6, 2])
    ax_ld_Tmin.plot(t, lie_deriv_raw_arr[:, 4], 'g-', linewidth=1.5)
    ax_ld_Tmin.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax_ld_Tmin.set_title('L_f h - Min Temperature')
    ax_ld_Tmin.set_ylabel('L_f h')
    ax_ld_Tmin.set_xlabel('Time (min)')
    ax_ld_Tmin.grid(True)

    # Column 3: Normalized L_f h
    ax_ldn_T = fig.add_subplot(gs[2, 3])
    ax_ldn_T.plot(t, lie_deriv_norm_arr[:, 0], 'b-', linewidth=1.5)
    ax_ldn_T.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax_ldn_T.set_title('Normalized L_f h - Temperature')
    ax_ldn_T.set_ylabel('L_f h_norm')
    ax_ldn_T.grid(True)

    ax_ldn_HTO = fig.add_subplot(gs[3, 3])
    ax_ldn_HTO.plot(t, lie_deriv_norm_arr[:, 1], 'r-', linewidth=1.5)
    ax_ldn_HTO.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax_ldn_HTO.set_title('Normalized L_f h - HTO')
    ax_ldn_HTO.set_ylabel('L_f h_norm')
    ax_ldn_HTO.grid(True)

    ax_ldn_V = fig.add_subplot(gs[4, 3])
    ax_ldn_V.plot(t, lie_deriv_norm_arr[:, 2], 'm-', linewidth=1.5)
    ax_ldn_V.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax_ldn_V.set_title('Normalized L_f h - Voltage')
    ax_ldn_V.set_ylabel('L_f h_norm')
    ax_ldn_V.grid(True)

    ax_ldn_P = fig.add_subplot(gs[5, 3])
    ax_ldn_P.plot(t, lie_deriv_norm_arr[:, 3], 'c-', linewidth=1.5)
    ax_ldn_P.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax_ldn_P.set_title('Normalized L_f h - Power')
    ax_ldn_P.set_ylabel('L_f h_norm')
    ax_ldn_P.grid(True)

    ax_ldn_Tmin = fig.add_subplot(gs[6, 3])
    ax_ldn_Tmin.plot(t, lie_deriv_norm_arr[:, 4], 'g-', linewidth=1.5)
    ax_ldn_Tmin.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax_ldn_Tmin.set_title('Normalized L_f h - Min Temperature')
    ax_ldn_Tmin.set_ylabel('L_f h_norm')
    ax_ldn_Tmin.set_xlabel('Time (min)')
    ax_ldn_Tmin.grid(True)

    # Column 4: CBF Conditions
    slack_arr = np.array(history['slack']) if len(history['slack']) > 0 else np.zeros((len(t), 5))

    ax_cbf_T = fig.add_subplot(gs[2, 4])
    ax_cbf_T.plot(t, cbf_arr[:, 0], 'b-', linewidth=1.5)
    ax_cbf_T.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax_cbf_T.fill_between(t, -10, 0, alpha=0.2, color='red', where=(cbf_arr[:, 0] < 0))
    ax_cbf_T.set_title('CBF Condition - Temperature (h_T)')
    ax_cbf_T.set_ylabel('L_f h + gamma*h')
    ax_cbf_T.grid(True)

    ax_cbf_HTO = fig.add_subplot(gs[3, 4])
    ax_cbf_HTO.plot(t, cbf_arr[:, 1], 'r-', linewidth=1.5)
    ax_cbf_HTO.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax_cbf_HTO.fill_between(t, -10, 0, alpha=0.2, color='red', where=(cbf_arr[:, 1] < 0))
    ax_cbf_HTO.set_title('CBF Condition - HTO')
    ax_cbf_HTO.set_ylabel('L_f h + gamma*h')
    ax_cbf_HTO.grid(True)

    ax_cbf_V = fig.add_subplot(gs[4, 4])
    ax_cbf_V.plot(t, cbf_arr[:, 2], 'm-', linewidth=1.5)
    ax_cbf_V.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax_cbf_V.fill_between(t, -10, 0, alpha=0.2, color='red', where=(cbf_arr[:, 2] < 0))
    ax_cbf_V.set_title('CBF Condition - Voltage')
    ax_cbf_V.set_ylabel('L_f h + gamma*h')
    ax_cbf_V.grid(True)

    ax_cbf_P = fig.add_subplot(gs[5, 4])
    ax_cbf_P.plot(t, cbf_arr[:, 3], 'c-', linewidth=1.5)
    ax_cbf_P.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax_cbf_P.fill_between(t, -10, 0, alpha=0.2, color='red', where=(cbf_arr[:, 3] < 0))
    ax_cbf_P.set_title('CBF Condition - Power')
    ax_cbf_P.set_ylabel('L_f h + gamma*h')
    ax_cbf_P.grid(True)

    ax_cbf_Tmin = fig.add_subplot(gs[6, 4])
    ax_cbf_Tmin.plot(t, cbf_arr[:, 4], 'g-', linewidth=1.5)
    ax_cbf_Tmin.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax_cbf_Tmin.fill_between(t, -10, 0, alpha=0.2, color='red', where=(cbf_arr[:, 4] < 0))
    ax_cbf_Tmin.set_title('CBF Condition - Min Temperature')
    ax_cbf_Tmin.set_ylabel('L_f h + gamma*h')
    ax_cbf_Tmin.set_xlabel('Time (min)')
    ax_cbf_Tmin.grid(True)

    # Column 5: Slack variables on separate subplots
    ax_slack_T = fig.add_subplot(gs[2, 5])
    ax_slack_T.plot(t, slack_arr[:, 0], 'b-', linewidth=1.5)
    ax_slack_T.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax_slack_T.fill_between(t, 0, np.max(slack_arr[:, 0]) * 1.2 + 1e-6, alpha=0.2, color='red', where=(slack_arr[:, 0] > 0))
    ax_slack_T.set_title('Slack - Temperature')
    ax_slack_T.set_ylabel('slack')
    ax_slack_T.set_ylim(bottom=0)
    ax_slack_T.grid(True)

    ax_slack_HTO = fig.add_subplot(gs[3, 5])
    ax_slack_HTO.plot(t, slack_arr[:, 1], 'r-', linewidth=1.5)
    ax_slack_HTO.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax_slack_HTO.fill_between(t, 0, np.max(slack_arr[:, 1]) * 1.2 + 1e-6, alpha=0.2, color='red', where=(slack_arr[:, 1] > 0))
    ax_slack_HTO.set_title('Slack - HTO')
    ax_slack_HTO.set_ylabel('slack')
    ax_slack_HTO.set_ylim(bottom=0)
    ax_slack_HTO.grid(True)

    ax_slack_V = fig.add_subplot(gs[4, 5])
    ax_slack_V.plot(t, slack_arr[:, 2], 'm-', linewidth=1.5)
    ax_slack_V.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax_slack_V.fill_between(t, 0, np.max(slack_arr[:, 2]) * 1.2 + 1e-6, alpha=0.2, color='red', where=(slack_arr[:, 2] > 0))
    ax_slack_V.set_title('Slack - Voltage')
    ax_slack_V.set_ylabel('slack')
    ax_slack_V.set_ylim(bottom=0)
    ax_slack_V.grid(True)

    ax_slack_P = fig.add_subplot(gs[5, 5])
    ax_slack_P.plot(t, slack_arr[:, 3], 'c-', linewidth=1.5)
    ax_slack_P.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax_slack_P.fill_between(t, 0, np.max(slack_arr[:, 3]) * 1.2 + 1e-6, alpha=0.2, color='red', where=(slack_arr[:, 3] > 0))
    ax_slack_P.set_title('Slack - Power')
    ax_slack_P.set_ylabel('slack')
    ax_slack_P.set_ylim(bottom=0)
    ax_slack_P.grid(True)

    ax_slack_Tmin = fig.add_subplot(gs[6, 5])
    ax_slack_Tmin.plot(t, slack_arr[:, 4], 'g-', linewidth=1.5)
    ax_slack_Tmin.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax_slack_Tmin.fill_between(t, 0, np.max(slack_arr[:, 4]) * 1.2 + 1e-6, alpha=0.2, color='red', where=(slack_arr[:, 4] > 0))
    ax_slack_Tmin.set_title('Slack - Min Temperature')
    ax_slack_Tmin.set_ylabel('slack')
    ax_slack_Tmin.set_xlabel('Time (min)')
    ax_slack_Tmin.set_ylim(bottom=0)
    ax_slack_Tmin.grid(True)

    # Row 7: Cost Composition & Solver Timing
    cost_ref = np.array(history['cost_ref'])
    cost_delta = np.array(history['cost_delta'])
    cost_slack = np.array(history['cost_slack'])
    cost_total = np.array(history['cost_total'])
    solve_time_ms = np.array(history['solve_time_ms'])

    # Stacked area plot of cost components
    ax_cost_stack = fig.add_subplot(gs[7, 0:3])
    ax_cost_stack.stackplot(t, cost_ref, cost_delta, cost_slack,
                            labels=['Ref Tracking', 'Control Change', 'Slack Penalty'],
                            colors=['#1f77b4', '#ff7f0e', '#d62728'], alpha=0.7)
    ax_cost_stack.plot(t, cost_total, 'k--', linewidth=1.5, label='Total Cost')
    ax_cost_stack.set_title('CBF Projection Cost Composition')
    ax_cost_stack.set_ylabel('Cost')
    ax_cost_stack.set_xlabel('Time (min)')
    ax_cost_stack.legend(loc='upper left')
    ax_cost_stack.grid(True, alpha=0.3)

    # Solver time evolution
    ax_solve_time = fig.add_subplot(gs[7, 3:])
    ax_solve_time.plot(t, solve_time_ms, 'g-', linewidth=1.5, label='Solve Time')
    ax_solve_time.axhline(y=np.mean(solve_time_ms), color='r', linestyle='--', label=f'Mean={np.mean(solve_time_ms):.1f}ms')
    ax_solve_time.fill_between(t, 0, solve_time_ms, alpha=0.2, color='green')
    if len(solve_time_ms) > 0:
        ax_solve_time.axhline(y=np.max(solve_time_ms), color='orange', linestyle=':', alpha=0.7, label=f'Max={np.max(solve_time_ms):.1f}ms')
    ax_solve_time.set_title('CBF Optimization Solver Time')
    ax_solve_time.set_ylabel('Time (ms)')
    ax_solve_time.set_xlabel('Time (min)')
    ax_solve_time.legend(loc='upper right')
    ax_solve_time.grid(True, alpha=0.3)

    # Row 8-9: Summary text
    ax_sum = fig.add_subplot(gs[8:10, :])
    ax_sum.axis('off')

    max_hto = max(history['HTO'])
    max_temp = max(history['T_s'])
    max_I = max(history['I'])
    n_failures = len(failure_indices)
    total_points = len(history['projection_success'])

    active_mask_str = config.get('active_mask', [1,1,1,1,1])
    summary = f"""
    CBF Test Configuration:
    -----------------------
    gamma_vec: {config['gamma_vec']}
    rho_vec: {config['rho_vec']}
    h_margin_vec: {config['h_margin_vec']}
    normalize: {config['normalize']}
    duration: {config['duration']}s
    active_mask: {active_mask_str}

    Results:
    --------
    Max HTO: {max_hto:.3f}%
    Max Temperature: {max_temp:.2f}°C
    Max Current: {max_I:.0f}A

    Solver Status:
    --------------
    Total Steps: {total_points}
    Solve Failures: {n_failures} ({100*n_failures/total_points:.1f}%)

    Solver Timing:
    --------------
    Mean Solve Time: {np.mean(history['solve_time_ms']):.2f}ms
    Max Solve Time: {np.max(history['solve_time_ms']):.2f}ms
    Min Solve Time: {np.min(history['solve_time_ms']):.2f}ms
    """
    if n_failures > 0:
        summary += f"    First Failure: t={t[failure_indices[0]]:.1f}min\n"

    ax_sum.text(0.1, 0.5, summary, fontsize=11, fontfamily='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle(f'CBF Test: {config["test_name"]}', fontsize=14, fontweight='bold')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: {output_path}")
def plot_comparison(hist_orig, hist_norm, output_path):
    """Plot comparison between original and normalized CBF"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    t_orig = np.array(hist_orig['t'])
    t_norm = np.array(hist_norm['t'])

    # Row 1: Current
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(t_orig, hist_orig['I'], 'b-', linewidth=2, label='Original CBF', alpha=0.8)
    ax1.plot(t_norm, hist_norm['I'], 'r-', linewidth=2, label='Normalized CBF', alpha=0.8)
    ax1.plot(t_orig, hist_orig['I_ref'], 'k--', linewidth=1.5, label='Reference', alpha=0.5)
    ax1.set_title('Current Comparison')
    ax1.set_ylabel('A')
    ax1.legend()
    ax1.grid(True)

    # Row 2: HTO and Temperature
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t_orig, hist_orig['HTO'], 'b-', linewidth=2, label='Original')
    ax2.plot(t_norm, hist_norm['HTO'], 'r-', linewidth=2, label='Normalized')
    ax2.axhline(y=2.0, color='k', linestyle='--')
    ax2.fill_between(t_orig, 2.0, 3.0, alpha=0.1, color='red')
    ax2.set_title('HTO Comparison')
    ax2.set_ylabel('%')
    ax2.legend()
    ax2.grid(True)
    ax2.set_ylim(0, 3)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(t_orig, hist_orig['T_s'], 'b-', linewidth=2, label='Original')
    ax3.plot(t_norm, hist_norm['T_s'], 'r-', linewidth=2, label='Normalized')
    ax3.axhline(y=90, color='k', linestyle='--')
    ax3.set_title('Temperature Comparison')
    ax3.set_ylabel('°C')
    ax3.legend()
    ax3.grid(True)

    # Summary
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')

    max_hto_orig = max(hist_orig['HTO'])
    max_hto_norm = max(hist_norm['HTO'])
    max_temp_orig = max(hist_orig['T_s'])
    max_temp_norm = max(hist_norm['T_s'])

    summary = f"""
    CBF Comparison Summary:
    -----------------------
    Metric              Original CBF    Normalized CBF
    Max HTO:            {max_hto_orig:.3f}%          {max_hto_norm:.3f}%
    Max Temperature:    {max_temp_orig:.2f}°C        {max_temp_norm:.2f}°C

    Key Insight:
    - Normalized CBF balances |L_f h| and |gamma*h| magnitudes
    - More proactive control (reacts earlier based on derivative trend)
    """
    ax4.text(0.1, 0.5, summary, fontsize=11, fontfamily='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('CBF Comparison: Original vs Normalized', fontsize=14, fontweight='bold')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Comparison figure saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Unified CBF Test and Visualization')
    parser.add_argument('--mode', choices=['single', 'compare'], default='single',
                       help='Test mode: single CBF test or comparison')
    parser.add_argument('--scenario', choices=['default', 'low_power_hto', 'high_temp', 'high_power'], default='default',
                       help='Test scenario: default, low_power_hto (600A low flow, HTO risk test), high_temp (high current + low coolant, temperature risk test), or high_power (max current, power limit test)')
    parser.add_argument('--gamma_vec', type=float, nargs=5, default=[50.0, 0.3, 100.0, 100.0, 10.0],
                       help='Per-constraint gamma values [h_T, h_HTO, h_V, h_P, h_Tmin] (default: [50, 0.3, 100, 100, 10])')
    parser.add_argument('--rho_vec', type=float, nargs=5, default=[1000, 10000, 1000, 1000, 1000],
                       help='Per-constraint rho values (default: [1000, 10000, 1000, 1000, 1000])')
    parser.add_argument('--h_margin_vec', type=float, nargs=5, default=[0.5, 0.010, 0.0, 0.0, 0.0],
                       help='Per-constraint safety margins [h_T, h_HTO, h_V, h_P, h_Tmin] (e.g., 0.010 for HTO means 1.0%% margin)')
    parser.add_argument('--lambda_u_scale', type=float, nargs='+', default=[1000.0],
                       help='Control change penalty scale. Single value applies to all controls, '
                            'or 3 values [I, v_lye, v_c] for per-control tuning (default: 1000.0)')
    parser.add_argument('--soft_mask', type=int, nargs=5, default=[1, 1, 1, 1, 1],
                       help='Per-constraint soft slack mask [h_T, h_HTO, h_V, h_P, h_Tmin]. 1=soft, 0=hard (default: all 1)')
    parser.add_argument('--active_mask', type=int, nargs=5, default=None,
                       help='Per-constraint active mask [h_T, h_HTO, h_V, h_P, h_Tmin]. 1=active, 0=inactive. '
                            'If omitted, scenario defaults are used (high_temp->T only, high_power->T/V/P, low_power->T/HTO).')
    parser.add_argument('--normalize', action='store_true', default=True,
                       help='Use normalized CBF')
    parser.add_argument('--duration', type=int, default=36000,
                       help='Simulation duration in seconds (default: 36000 = 10 hours)')
    parser.add_argument('--output_dir', type=str, default='output/single_stack/cbf_tests',
                       help='Output directory for figures')

    args = parser.parse_args()

    print("="*70)
    print("Unified CBF Test and Visualization")
    print("="*70)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == 'single':
        # Unified CBF tuning parameters across ALL scenarios; only u_ref differs per scenario
        # gamma_T=3.0: temperature reacts faster to drive current down early,
        #               indirectly protecting HTO before it can spike.
        # gamma_HTO=2.0: less conservative than 1.0, so HTO does not clamp current
        #                prematurely, but still reacts well before the 2% limit.
        # h_margin_HTO=0.005: 0.5% buffer -> effective limit 1.5%, reasonable margin.
        args.gamma_vec = [3.0, 2.0, 100.0, 100.0, 5.0]
        args.rho_vec = [50000, 50000, 50000, 50000, 10000]
        args.h_margin_vec = [1.0, 0.005, 0.0, 0.0, 0.0]
        args.soft_mask = [0, 1, 1, 0, 1]  # T hard, P hard, others soft
        # Per-control change penalty: [I, v_lye, v_c]
        # I gets lighter penalty so current can respond quickly to safety constraints;
        # v_lye gets lighter penalty as flow adjustments are less critical;
        # v_c keeps original penalty to avoid rapid coolant swings.
        args.lambda_u_scale = [500.0, 200.0, 1000.0]

        if args.scenario == 'low_power_hto':
            u_ref = np.array([600.0, 0.04, 0.01])  # 600A, 0.04 m3/s, v_c=0.01
            scenario_active_mask = np.array([True, True, False, False, True])
            test_name = 'LowPower_HTO_Tuned'
            print(f"\nRunning LOW POWER HTO test (TUNED)...")
            print(f"  Reference: I=600A, v_lye=0.04 m3/s, v_c=0.01")
        elif args.scenario == 'high_temp':
            u_ref = np.array([9360.0, 0.03, 0.0])  # Max current, normal lye, no coolant -> strong temp drive
            scenario_active_mask = np.array([True, True, False, True, False])
            test_name = 'HighTemperature'
            print(f"\nRunning HIGH TEMPERATURE test...")
            print(f"  Reference: I=9360A, v_lye=0.03 m3/s, v_c=0.0 (Temp risk params)")
        elif args.scenario == 'high_power':
            u_ref = np.array([9360.0, 0.03, 1.0])  # Max current, normal lye flow, very high coolant
            scenario_active_mask = np.array([True, False, False, True, True])
            test_name = 'HighPower'
            print(f"\nRunning HIGH POWER test...")
            print(f"  Reference: I=9360A (max), v_lye=0.03 m3/s, v_c=1.0 (Power limit test)")
        else:
            u_ref = None
            scenario_active_mask = np.array([True, True, True, True, True])
            test_name = f'CBF_test_gamma{args.gamma_vec[1]}'
            print(f"\nRunning single CBF test...")

        # Command-line override takes precedence
        if args.active_mask is not None:
            active_mask = np.array([bool(x) for x in args.active_mask])
        else:
            active_mask = scenario_active_mask

        print(f"  gamma_vec: {args.gamma_vec}")
        print(f"  rho_vec: {args.rho_vec}")
        soft_mask_bool = [bool(x) for x in args.soft_mask]
        print(f"  h_margin_vec: {args.h_margin_vec}")
        print(f"  lambda_u_scale: {args.lambda_u_scale}")
        print(f"  soft_mask: {soft_mask_bool}")
        print(f"  active_mask: {active_mask.tolist()}")
        print(f"  normalize: {args.normalize}")

        projector = SingleStackCBFProjection(
            dt=60.0,
            gamma_vec=args.gamma_vec,
            rho_vec=args.rho_vec,
            h_margin_vec=args.h_margin_vec,
            normalize=args.normalize,
            lambda_u_scale=args.lambda_u_scale,
            soft_mask=soft_mask_bool,
            active_mask=active_mask
        )

        history = run_simulation(projector, duration=args.duration, u_ref=u_ref, active_mask=active_mask)

        config = {
            'gamma_vec': args.gamma_vec,
            'rho_vec': args.rho_vec,
            'h_margin_vec': args.h_margin_vec,
            'normalize': args.normalize,
            'duration': args.duration,
            'test_name': test_name,
            'active_mask': active_mask.tolist()
        }

        if args.scenario == 'low_power_hto':
            output_path = os.path.join(args.output_dir, 'cbf_low_power_hto.png')
        elif args.scenario == 'high_temp':
            output_path = os.path.join(args.output_dir, 'cbf_high_temp.png')
        elif args.scenario == 'high_power':
            output_path = os.path.join(args.output_dir, 'cbf_high_power.png')
        else:
            output_path = os.path.join(args.output_dir, 'cbf_unified_test.png')
        plot_single_test(history, config, output_path)

        print(f"\nResults:")
        print(f"  Max HTO: {max(history['HTO']):.3f}%")
        print(f"  Max Temp: {max(history['T_s']):.2f}°C")
        print(f"  Max Current: {max(history['I']):.0f}A")
        print(f"  Max Power: {max(history['Power']):.3f}MW")

    elif args.mode == 'compare':
        print(f"\nRunning CBF comparison (Original vs Normalized)...")

        # Determine reference control based on scenario
        if args.scenario == 'low_power_hto':
            u_ref = np.array([600.0, 0.005, 0.0])
            print("  Using low power HTO scenario (I=600A, v_lye=0.005)")
        else:
            u_ref = None

        # Original CBF
        print("  Running Original CBF...")
        proj_orig = SingleStackCBFProjection(
            dt=60.0, gamma=1.0,
            gamma_vec=[1.0, 1.0, 1.0, 1.0, 1.0],
            rho_vec=[1e6, 1e15, 1e6, 1e6, 1e6],
            normalize=False
        )
        hist_orig = run_simulation(proj_orig, duration=args.duration, u_ref=u_ref)

        # Normalized CBF
        print("  Running Normalized CBF...")
        proj_norm = SingleStackCBFProjection(
            dt=60.0, gamma=1.0,
            gamma_vec=[1.0, 5.0, 1.0, 1.0, 1.0],
            rho_vec=[1e6, 1e15, 1e6, 1e6, 1e6],
            normalize=True
        )
        hist_norm = run_simulation(proj_norm, duration=args.duration, u_ref=u_ref)

        output_path = os.path.join(args.output_dir, 'cbf_comparison.png')
        plot_comparison(hist_orig, hist_norm, output_path)

        print(f"\nResults:")
        print(f"  Original:   Max HTO={max(hist_orig['HTO']):.3f}%, Max Temp={max(hist_orig['T_s']):.2f}°C")
        print(f"  Normalized: Max HTO={max(hist_norm['HTO']):.3f}%, Max Temp={max(hist_norm['T_s']):.2f}°C")

    print("="*70)


if __name__ == "__main__":
    main()
