#!/usr/bin/env python3
"""
CBF T=80°C Non-Activating Test

Demonstrates an operating point where temperature stabilizes at ~80.5°C
and the CBF projector NEVER activates, by using a small coolant flow (v_c=0.001)
to enable higher current (I=2500A), which produces enough O2 to keep HTO low.

Key parameter adjustment: h_margin_HTO = 0.002 (effective HTO limit = 1.8%)
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


def main():
    output_dir = 'output/single_stack/cbf_tests/t80_nocbf'
    os.makedirs(output_dir, exist_ok=True)

    # Adjusted parameter: h_margin_HTO = 0.002 (effective limit = 1.8%)
    projector = SingleStackCBFProjection(
        dt=60.0,
        gamma_vec=[3.0, 2.0, 100.0, 100.0, 5.0],
        rho_vec=[50000, 50000, 50000, 50000, 10000],
        h_margin_vec=[1.0, 0.002, 0.0, 0.0, 0.0],
        lambda_u_scale=[500.0, 200.0, 1000.0],
        soft_mask=[False, True, True, False, True],
        active_mask=None,
        normalize=True,
    )

    # Operating point: I=2500A, v_lye=0.04, v_c=0.001
    # Small v_c allows higher I, which increases O2 production (lowers HTO)
    # while keeping T ~ 80°C through moderate cooling
    I_ss = 2500.0
    v_lye_ss = 0.04
    v_c_ss = 0.001
    u_ref = np.array([I_ss, v_lye_ss, v_c_ss])

    # Drive simulator to steady state
    print("Driving simulator to steady state...")
    sim = SingleStackSimulator()
    sim.reset()
    for _ in range(30000):  # 6000s
        sim.step(u_ref)
    print(f"Steady-state temperature: {sim.state[1] - 273.15:.2f}°C")

    # Verify CBF does not activate at steady state
    action, success, info = projector.project(
        u_ref, sim.state, u_last=u_ref,
        active_mask=np.array([True, True, True, True, True])
    )
    print(f"\nCBF check at steady state:")
    print(f"  projection_needed: {info['projection_needed']}")
    print(f"  cbf_ref = [{', '.join([f'{v:.4f}' for v in info.get('cbf_ref', info['cbf'])])}]")
    print(f"  h       = [{', '.join([f'{v:.4f}' for v in info['h']])}]")
    print(f"  adjustment: {np.linalg.norm(action - u_ref):.6f}")

    # Run simulation with CBF projection from steady state
    duration = 3600.0  # 1 hour
    dt = 0.2
    ctrl_dt = 60.0
    steps = int(duration / dt)
    ctrl_steps = int(ctrl_dt / dt)

    last_action = u_ref.copy()
    history = {
        't': [], 'I': [], 'v_lye': [], 'v_c': [],
        'T_s': [], 'Power': [], 'HTO': [], 'U_cell': [],
        'proj_needed': [], 'adjustment': [],
        'h_T': [], 'h_HTO': [], 'h_V': [], 'h_P': [], 'h_Tmin': [],
        'cbf_T': [], 'cbf_HTO': [], 'cbf_V': [], 'cbf_P': [], 'cbf_Tmin': [],
    }

    max_adj = 0.0
    n_proj = 0
    n_no_proj = 0

    for i in range(steps):
        if i % ctrl_steps == 0:
            state = sim.state
            action, success, info = projector.project(
                u_ref, state, u_last=u_ref,
                active_mask=np.array([True, True, True, True, True])
            )
            last_action = action.copy()

            h = info['h']
            cbf = info.get('cbf_ref', info['cbf'])
            proj_needed = info.get('projection_needed', True)
            adj = np.linalg.norm(action - u_ref)
            max_adj = max(max_adj, adj)
            if proj_needed:
                n_proj += 1
            else:
                n_no_proj += 1

            history['t'].append(i * dt / 60)
            history['I'].append(action[0])
            history['v_lye'].append(action[1])
            history['v_c'].append(action[2])
            history['T_s'].append(state[1] - 273.15)

            _, U_cell, _ = sim._calculate_electrochemical_properties(action[0], state[1])
            Power = U_cell * action[0] * sim.N_cell / 1e6
            history['Power'].append(Power)
            history['U_cell'].append(U_cell)

            n_gas = state[6]
            T_sep = state[2]
            hto = (n_gas * projector.R * T_sep) / (projector.P_sys * projector.V_sep_gas) * 100
            history['HTO'].append(hto)

            history['proj_needed'].append(proj_needed)
            history['adjustment'].append(adj)
            history['h_T'].append(h[0])
            history['h_HTO'].append(h[1])
            history['h_V'].append(h[2])
            history['h_P'].append(h[3])
            history['h_Tmin'].append(h[4])
            history['cbf_T'].append(cbf[0])
            history['cbf_HTO'].append(cbf[1])
            history['cbf_V'].append(cbf[2])
            history['cbf_P'].append(cbf[3])
            history['cbf_Tmin'].append(cbf[4])

        sim.step(last_action)

    print(f"\n=== T=80°C No-CBF Results ===")
    print(f"Operating point: I={I_ss}A, v_lye={v_lye_ss}, v_c={v_c_ss}")
    print(f"h_margin_HTO = 0.002 (effective HTO limit = 1.8%)")
    print(f"Simulation duration: {duration/60:.0f} min")
    print(f"Total control steps: {n_proj + n_no_proj}")
    print(f"CBF activated: {n_proj} ({n_proj/(n_proj+n_no_proj):.1%})")
    print(f"CBF NOT activated: {n_no_proj} ({n_no_proj/(n_proj+n_no_proj):.1%})")
    print(f"Max control adjustment: {max_adj:.6f}")
    print(f"Max Power: {max(history['Power']):.3f}MW")
    print(f"Max Temp: {max(history['T_s']):.2f}°C")
    print(f"Max HTO: {max(history['HTO']):.3f}%")
    print(f"Min CBF values: T={min(history['cbf_T']):.2f}, HTO={min(history['cbf_HTO']):.4f}, "
          f"V={min(history['cbf_V']):.2f}, P={min(history['cbf_P']):.2f}, Tmin={min(history['cbf_Tmin']):.2f}")

    # Plotting
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

    t = np.array(history['t'])

    # Row 1: Controls
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, history['I'], 'b-', linewidth=2, label='Actual')
    ax1.axhline(y=I_ss, color='r', linestyle='--', alpha=0.5, label='Reference')
    ax1.set_title('Current')
    ax1.set_ylabel('A')
    ax1.legend()
    ax1.grid(True)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, np.array(history['v_lye']) * 1000, 'b-', linewidth=2)
    ax2.axhline(y=v_lye_ss * 1000, color='r', linestyle='--', alpha=0.5)
    ax2.set_title('Lye Flow')
    ax2.set_ylabel('L/s')
    ax2.grid(True)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(t, np.array(history['v_c']) * 1000, 'b-', linewidth=2)
    ax3.axhline(y=v_c_ss * 1000, color='r', linestyle='--', alpha=0.5)
    ax3.set_title('Coolant Flow')
    ax3.set_ylabel('L/s')
    ax3.grid(True)

    # Row 2: Physical quantities
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(t, history['T_s'], 'r-', linewidth=2)
    ax4.axhline(y=80, color='g', linestyle='--', alpha=0.5, label='T_target')
    ax4.axhline(y=90, color='r', linestyle='--', alpha=0.5, label='T_max')
    ax4.axhline(y=20, color='b', linestyle='--', alpha=0.5, label='T_min')
    ax4.set_title('Stack Temperature')
    ax4.set_ylabel('°C')
    ax4.legend()
    ax4.grid(True)

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(t, history['Power'], 'c-', linewidth=2)
    ax5.axhline(y=6.0, color='r', linestyle='--', alpha=0.5, label='P_max')
    ax5.set_title('Stack Power')
    ax5.set_ylabel('MW')
    ax5.legend()
    ax5.grid(True)

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(t, history['HTO'], 'g-', linewidth=2)
    ax6.axhline(y=2.0, color='r', linestyle='--', alpha=0.5, label='HTO_max')
    ax6.axhline(y=1.8, color='orange', linestyle='--', alpha=0.5, label='HTO_eff (1.8%)')
    ax6.set_title('HTO')
    ax6.set_ylabel('%')
    ax6.legend()
    ax6.grid(True)

    # Row 3: h values
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(t, history['h_T'], 'b-', linewidth=1.5, label='h_T')
    ax7.plot(t, history['h_Tmin'], 'g-', linewidth=1.5, label='h_Tmin')
    ax7.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax7.set_title('h: Temperature Constraints')
    ax7.set_ylabel('h')
    ax7.legend()
    ax7.grid(True)

    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(t, history['h_HTO'], 'r-', linewidth=1.5)
    ax8.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax8.set_title('h: HTO')
    ax8.set_ylabel('h')
    ax8.grid(True)

    ax9 = fig.add_subplot(gs[2, 2])
    ax9.plot(t, np.array(history['h_P']) / 1e6, 'c-', linewidth=1.5, label='h_P (MW)')
    ax9.plot(t, history['h_V'], 'm-', linewidth=1.5, label='h_V')
    ax9.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax9.set_title('h: Power & Voltage')
    ax9.set_ylabel('h')
    ax9.legend()
    ax9.grid(True)

    # Row 4: CBF conditions
    ax10 = fig.add_subplot(gs[3, 0])
    ax10.plot(t, history['cbf_T'], 'b-', linewidth=1.5)
    ax10.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax10.set_title('CBF: Temperature')
    ax10.set_ylabel('L_f h + gamma*h')
    ax10.set_xlabel('Time (min)')
    ax10.grid(True)

    ax11 = fig.add_subplot(gs[3, 1])
    ax11.plot(t, history['cbf_HTO'], 'r-', linewidth=1.5)
    ax11.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax11.set_title('CBF: HTO')
    ax11.set_ylabel('L_f h + gamma*h')
    ax11.set_xlabel('Time (min)')
    ax11.grid(True)

    ax12 = fig.add_subplot(gs[3, 2])
    ax12.plot(t, history['cbf_P'], 'c-', linewidth=1.5, label='CBF_P')
    ax12.plot(t, history['cbf_Tmin'], 'g-', linewidth=1.5, label='CBF_Tmin')
    ax12.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax12.set_title('CBF: Power & Tmin')
    ax12.set_ylabel('L_f h + gamma*h')
    ax12.set_xlabel('Time (min)')
    ax12.legend()
    ax12.grid(True)

    plt.suptitle(
        f'T=80°C No-CBF Case: I={I_ss}A, v_lye={v_lye_ss}, v_c={v_c_ss}\n'
        f'h_margin_HTO=0.002 (eff limit=1.8%) | '
        f'CBF activated: {n_proj} times ({n_proj/(n_proj+n_no_proj):.1%}) | '
        f'Max adjustment: {max_adj:.6f}',
        fontsize=13, fontweight='bold'
    )
    out_path = os.path.join(output_dir, 't80_nocbf.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {out_path}")


if __name__ == '__main__':
    main()
