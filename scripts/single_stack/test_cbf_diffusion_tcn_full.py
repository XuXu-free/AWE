#!/usr/bin/env python3
"""
Single-Stack TCN Diffusion + CBF Model Controller Test (Full Diagnostics)

Records and visualizes complete CBF constraint reaction:
- h values for all 5 constraints
- CBF conditions (L_f h + gamma*h)
- Projection activation & adjustment magnitude
- Temperature, Power, HTO trajectories with boundaries
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from plant.single_stack_simulator import SingleStackSimulator
from controller.single_stack.cbf_model_controller import SingleStackCBFModelController


def run_test(controller, name, initial_state, P_ref_profile, duration=7200, dt=60.0):
    """Run test with full CBF diagnostics recording."""
    print(f"\n{'='*70}")
    print(f"Testing: {name}")
    print(f"{'='*70}")

    sim = SingleStackSimulator(sim_dt=0.2)
    sim.reset(initial_state=initial_state)
    steps = int(duration / dt)

    def get_p_ref(t):
        for t_start, t_end, p_ref in P_ref_profile:
            if t_start <= t < t_end:
                return p_ref
        return P_ref_profile[-1][2]

    history = {
        't': [], 'I': [], 'v_lye': [], 'v_c': [],
        'T_s': [], 'T_sep': [], 'T_c_out': [],
        'P_real': [], 'P_ref': [], 'U_cell': [],
        'HTO': [], 'n_gas': [],
        # CBF diagnostics
        'cbf_active': [], 'adjustment': [],
        'h_T': [], 'h_HTO': [], 'h_V': [], 'h_P': [], 'h_Tmin': [],
        'cbf_T': [], 'cbf_HTO': [], 'cbf_V': [], 'cbf_P': [], 'cbf_Tmin': [],
        'solver_success': [],
        # Violations
        'T_violation': [], 'P_violation': [], 'HTO_violation': [], 'V_violation': [],
    }

    T_max = 363.15
    HTO_max = 2.0
    P_max = 6.0e6
    V_max = 2.2

    last_action = np.array([2000.0, 0.01, 0.001])
    T_ref = 353.15

    for i in range(steps):
        t = i * dt
        P_ref = get_p_ref(t)
        P_ref_seq = [get_p_ref(t + j * dt) for j in range(5)]

        state = sim.state
        try:
            if hasattr(controller, 'get_action_with_rollout'):
                I_cmd, v_lye_cmd, v_c_cmd = controller.get_action_with_rollout(
                    state, P_ref_seq, T_ref, last_action, num_candidates=64, verbose=False
                )
            else:
                I_cmd, v_lye_cmd, v_c_cmd = controller.get_action(
                    state, P_ref_seq, T_ref, last_action
                )
        except Exception as e:
            print(f"Error @ t={t}: {e}")
            I_cmd, v_lye_cmd, v_c_cmd = last_action[0], last_action[1], last_action[2]

        action = np.array([I_cmd, v_lye_cmd, v_c_cmd])

        # --- CBF diagnostics ---
        cbf_active = 0
        adjustment = 0.0
        h_vals = np.zeros(5)
        cbf_vals = np.zeros(5)
        solver_success = True

        if hasattr(controller, 'use_cbf_projection') and controller.use_cbf_projection:
            try:
                _, success, info = controller.projector.project(
                    action, state, u_last=last_action,
                    active_mask=np.array([True, True, True, True, True])
                )
                h_vals = info.get('h', np.zeros(5))
                cbf_vals = info.get('cbf_ref', info.get('cbf', np.zeros(5)))
                cbf_active = 1 if info.get('projection_needed', False) else 0
                adjustment = np.linalg.norm(info.get('adjustment', 0.0))
                solver_success = success
            except Exception as e:
                print(f"CBF diag error @ t={t}: {e}")

        last_action = action.copy()

        T_s_in, T_s, T_sep, T_c_out = state[0], state[1], state[2], state[3]
        n_H2_an, n_liq, n_gas = state[4], state[5], state[6]

        _, U_cell, _ = sim._calculate_electrochemical_properties(I_cmd, T_s)
        P_real = U_cell * I_cmd * sim.N_cell
        hto_pct = (n_gas * sim.R * T_sep) / (sim.P_sys * sim.V_sep_gas) * 100

        history['t'].append(t / 60)
        history['I'].append(I_cmd)
        history['v_lye'].append(v_lye_cmd)
        history['v_c'].append(v_c_cmd)
        history['T_s'].append(T_s - 273.15)
        history['T_sep'].append(T_sep - 273.15)
        history['T_c_out'].append(T_c_out - 273.15)
        history['P_real'].append(P_real / 1e6)
        history['P_ref'].append(P_ref / 1e6)
        history['U_cell'].append(U_cell)
        history['HTO'].append(hto_pct)
        history['n_gas'].append(n_gas)

        history['cbf_active'].append(cbf_active)
        history['adjustment'].append(adjustment)
        history['h_T'].append(h_vals[0])
        history['h_HTO'].append(h_vals[1])
        history['h_V'].append(h_vals[2])
        history['h_P'].append(h_vals[3])
        history['h_Tmin'].append(h_vals[4])
        history['cbf_T'].append(cbf_vals[0])
        history['cbf_HTO'].append(cbf_vals[1])
        history['cbf_V'].append(cbf_vals[2])
        history['cbf_P'].append(cbf_vals[3])
        history['cbf_Tmin'].append(cbf_vals[4])
        history['solver_success'].append(1 if solver_success else 0)

        history['T_violation'].append(T_s > T_max)
        history['P_violation'].append(P_real > P_max)
        history['HTO_violation'].append(hto_pct > HTO_max)
        history['V_violation'].append(U_cell > V_max)

        for _ in range(int(dt / sim.dt)):
            sim.step(action)

    max_T = max(history['T_s'])
    max_P = max(history['P_real'])
    max_HTO = max(history['HTO'])
    max_V = max(history['U_cell'])
    n_cbf = sum(history['cbf_active'])

    print(f"\n  Results:")
    print(f"    Max Temp: {max_T:.2f}°C (limit: 90°C)")
    print(f"    Max Power: {max_P:.3f} MW (limit: 6 MW)")
    print(f"    Max HTO: {max_HTO:.4f}% (limit: 2%)")
    print(f"    Max Voltage: {max_V:.3f} V (limit: 2.2V)")
    print(f"    CBF activated: {n_cbf} / {steps} ({n_cbf/steps*100:.1f}%)")

    return history


def plot_full_diagnostics(history, output_dir, title_suffix=""):
    """Plot complete CBF diagnostics."""
    t = np.array(history['t'])

    fig = plt.figure(figsize=(20, 18))
    gs = fig.add_gridspec(5, 3, hspace=0.4, wspace=0.3)

    # Row 1: Physical quantities
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, history['P_ref'], 'k--', alpha=0.5, label='Reference')
    ax1.plot(t, history['P_real'], 'b-', linewidth=2, label='Actual')
    ax1.axhline(y=6.0, color='r', linestyle='--', alpha=0.7, label='P_max')
    ax1.set_title('Power Tracking', fontweight='bold')
    ax1.set_ylabel('MW')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, history['T_s'], 'r-', linewidth=2, label='T_s')
    ax2.axhline(y=90, color='r', linestyle='--', alpha=0.7, label='T_max')
    ax2.axhline(y=80, color='g', linestyle='--', alpha=0.5, label='T_target')
    ax2.set_title('Stack Temperature', fontweight='bold')
    ax2.set_ylabel('°C')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(t, history['HTO'], 'g-', linewidth=2)
    ax3.axhline(y=2.0, color='r', linestyle='--', alpha=0.7, label='HTO_max')
    ax3.axhline(y=1.8, color='orange', linestyle='--', alpha=0.5, label='Effective (1.8%)')
    ax3.set_title('HTO', fontweight='bold')
    ax3.set_ylabel('%')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Row 2: Controls
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(t, history['I'], 'b-', linewidth=2)
    ax4.set_title('Current', fontweight='bold')
    ax4.set_ylabel('A')
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(t, np.array(history['v_lye']) * 1000, 'orange', linewidth=2)
    ax5.set_title('Lye Flow', fontweight='bold')
    ax5.set_ylabel('L/s')
    ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(t, np.array(history['v_c']) * 1000, 'c-', linewidth=2)
    ax6.set_title('Coolant Flow', fontweight='bold')
    ax6.set_ylabel('L/s')
    ax6.grid(True, alpha=0.3)

    # Row 3: h values (safety margins)
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(t, history['h_T'], 'b-', linewidth=1.5, label='h_T')
    ax7.plot(t, history['h_Tmin'], 'g-', linewidth=1.5, label='h_Tmin')
    ax7.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax7.set_title('h: Temperature Constraints', fontweight='bold')
    ax7.set_ylabel('h')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(t, history['h_HTO'], 'r-', linewidth=1.5)
    ax8.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax8.set_title('h: HTO', fontweight='bold')
    ax8.set_ylabel('h')
    ax8.grid(True, alpha=0.3)

    ax9 = fig.add_subplot(gs[2, 2])
    ax9.plot(t, np.array(history['h_P']) / 1e6, 'c-', linewidth=1.5, label='h_P (MW)')
    ax9.plot(t, history['h_V'], 'm-', linewidth=1.5, label='h_V')
    ax9.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax9.set_title('h: Power & Voltage', fontweight='bold')
    ax9.set_ylabel('h')
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    # Row 4: CBF conditions
    ax10 = fig.add_subplot(gs[3, 0])
    ax10.plot(t, history['cbf_T'], 'b-', linewidth=1.5)
    ax10.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Activation threshold')
    ax10.fill_between(t, 0, history['cbf_T'], where=(np.array(history['cbf_T']) > 0), alpha=0.2, color='lime')
    ax10.set_title('CBF: Temperature', fontweight='bold')
    ax10.set_ylabel('L_f h + gamma*h')
    ax10.legend()
    ax10.grid(True, alpha=0.3)

    ax11 = fig.add_subplot(gs[3, 1])
    ax11.plot(t, history['cbf_HTO'], 'r-', linewidth=1.5)
    ax11.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax11.fill_between(t, 0, history['cbf_HTO'], where=(np.array(history['cbf_HTO']) > 0), alpha=0.2, color='lime')
    ax11.set_title('CBF: HTO', fontweight='bold')
    ax11.set_ylabel('L_f h + gamma*h')
    ax11.grid(True, alpha=0.3)

    ax12 = fig.add_subplot(gs[3, 2])
    ax12.plot(t, history['cbf_P'], 'c-', linewidth=1.5, label='CBF_P')
    ax12.plot(t, history['cbf_Tmin'], 'g-', linewidth=1.5, label='CBF_Tmin')
    ax12.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax12.fill_between(t, 0, np.minimum(np.array(history['cbf_P']), np.array(history['cbf_Tmin'])),
                      where=(np.array(history['cbf_P']) > 0) & (np.array(history['cbf_Tmin']) > 0), alpha=0.2, color='lime')
    ax12.set_title('CBF: Power & Tmin', fontweight='bold')
    ax12.set_ylabel('L_f h + gamma*h')
    ax12.legend()
    ax12.grid(True, alpha=0.3)

    # Row 5: CBF activation & adjustment
    ax13 = fig.add_subplot(gs[4, 0])
    cbf_active_arr = np.array(history['cbf_active'])
    ax13.fill_between(t, 0, cbf_active_arr, alpha=0.6, color='orange', step='post', label='CBF Active')
    ax13.set_ylim(-0.1, 1.5)
    ax13.set_title('CBF Activation', fontweight='bold')
    ax13.set_ylabel('Active')
    ax13.set_xlabel('Time (min)')
    ax13.legend()
    ax13.grid(True, alpha=0.3)

    ax14 = fig.add_subplot(gs[4, 1])
    ax14.plot(t, history['adjustment'], 'purple', linewidth=1.5)
    ax14.set_title('CBF Control Adjustment', fontweight='bold')
    ax14.set_ylabel('||u - u_ref||')
    ax14.set_xlabel('Time (min)')
    ax14.grid(True, alpha=0.3)

    ax15 = fig.add_subplot(gs[4, 2])
    min_cbf = np.min([
        np.array(history['cbf_T']),
        np.array(history['cbf_HTO']),
        np.array(history['cbf_V']),
        np.array(history['cbf_P']),
        np.array(history['cbf_Tmin'])
    ], axis=0)
    ax15.plot(t, min_cbf, 'k-', linewidth=2, label='Min CBF (all)')
    ax15.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='Activation threshold')
    ax15.fill_between(t, 0, min_cbf, where=(min_cbf > 0), alpha=0.3, color='lime', label='Safe region')
    ax15.set_title('Minimum CBF Across All Constraints', fontweight='bold')
    ax15.set_ylabel('Min CBF')
    ax15.set_xlabel('Time (min)')
    ax15.legend()
    ax15.grid(True, alpha=0.3)

    fig.suptitle(
        f'TCN Diffusion + CBF: Full Constraint Diagnostics{title_suffix}\n'
        'CBF Params: gamma=[3,2,100,100,5], h_margin_HTO=0.002, soft_mask=[F,T,T,F,T]',
        fontsize=15, fontweight='bold'
    )

    out_path = os.path.join(output_dir, 'cbf_full_diagnostics.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {out_path}")


def main():
    output_dir = 'output/single_stack/cbf_model_tests'
    os.makedirs(output_dir, exist_ok=True)

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    model_path = os.path.join(base_dir, 'output', 'single_stack', 'policy', 'diffusion_tcn_policy_best.pth')
    stats_path = os.path.join(base_dir, 'output', 'single_stack', 'diffusion_stats.npz')

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
    if not os.path.exists(stats_path):
        print(f"Stats not found: {stats_path}")
        return

    print(f"Model: {model_path}")
    print(f"Stats: {stats_path}")

    initial_state = np.array([345.0, 353.15, 353.15, 325.0, 0.52, 0.52, 0.52])

    # Aggressive scenario: large power swings to stress all constraints
    P_ref_profile = [
        (0, 600, 1.0e6),
        (600, 1800, 4.5e6),
        (1800, 3000, 1.5e6),
        (3000, 4200, 5.5e6),
        (4200, 5400, 2.0e6),
        (5400, 7200, 4.0e6),
    ]

    print("\n" + "=" * 70)
    print("TCN Diffusion + CBF: Full Diagnostics Test")
    print("=" * 70)

    ctrl = SingleStackCBFModelController(
        dt=60.0, horizon=5, model_type='diffusion_tcn',
        model_path=model_path, stats_path=stats_path,
        use_cbf_projection=True,
        gamma_vec=[3.0, 2.0, 100.0, 100.0, 5.0],
        rho_vec=[50000, 50000, 50000, 50000, 10000],
        h_margin_vec=[1.0, 0.002, 0.0, 0.0, 0.0],
        lambda_u_scale=[500.0, 200.0, 1000.0],
        soft_mask=[False, True, True, False, True],
        normalize=True,
    )

    hist = run_test(ctrl, "TCN Diffusion + CBF", initial_state, P_ref_profile, duration=7200)
    plot_full_diagnostics(hist, output_dir)

    print("\n" + "=" * 70)
    print("Test complete!")
    print(f"Output: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
