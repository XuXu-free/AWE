#!/usr/bin/env python3
"""
Dynamic CBF Protection Test for Single-Stack AWE

Tests CBF projector in a closed-loop scenario where the baseline controller
drives the system toward constraint boundaries. Demonstrates:
1. Baseline-only: constraints may be violated
2. Baseline + CBF: safe operation with minimal conservatism

Baseline controller: Simple temperature-tracking PI with aggressive ramp
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


class BaselineController:
    """Simple baseline controller that tracks temperature target."""
    def __init__(self, I_init=2000.0, v_lye=0.01, v_c=0.001):
        self.I_init = I_init
        self.v_lye = v_lye
        self.v_c = v_c
        self.Kp = 50.0  # Aggressive gain to push toward boundary
        self.I_last = I_init

    def get_action(self, state, t, I_target_func):
        """Generate reference control that ramps current toward target."""
        I_target = I_target_func(t)
        # Simple ramp: move I toward target
        I_cmd = self.I_last + np.clip(I_target - self.I_last, -100, 100)
        self.I_last = I_cmd
        return np.array([I_cmd, self.v_lye, self.v_c])


def run_simulation(controller, projector, I_target_func, duration=7200.0,
                   dt=0.2, ctrl_dt=60.0, use_cbf=True, active_mask=None):
    """Run closed-loop simulation with optional CBF projection."""
    sim = SingleStackSimulator()
    sim.reset()

    steps = int(duration / dt)
    ctrl_steps = int(ctrl_dt / dt)
    last_action = np.array([controller.I_init, controller.v_lye, controller.v_c])

    history = {
        't': [], 'I_ref': [], 'I_act': [], 'v_lye': [], 'v_c': [],
        'T_s': [], 'Power': [], 'HTO': [], 'U_cell': [],
        'cbf_active': [], 'adjustment': [],
        'h_T': [], 'h_HTO': [], 'h_V': [], 'h_P': [], 'h_Tmin': [],
        'cbf_T': [], 'cbf_HTO': [], 'cbf_V': [], 'cbf_P': [], 'cbf_Tmin': [],
    }

    max_T = 0.0
    max_P = 0.0
    max_HTO = 0.0
    n_proj = 0

    for i in range(steps):
        if i % ctrl_steps == 0:
            state = sim.state
            u_ref = controller.get_action(state, i * dt, I_target_func)

            if use_cbf:
                action, success, info = projector.project(
                    u_ref, state, u_last=last_action, active_mask=active_mask
                )
            else:
                action = u_ref.copy()
                info = {'projection_needed': False, 'h': np.zeros(5),
                        'cbf': np.zeros(5), 'adjustment': 0.0}

            last_action = action.copy()
            proj_needed = info.get('projection_needed', False)
            adj = np.linalg.norm(action - u_ref)
            if proj_needed:
                n_proj += 1

            h = info['h']
            cbf = info.get('cbf_ref', info['cbf'])

            history['t'].append(i * dt / 60)
            history['I_ref'].append(u_ref[0])
            history['I_act'].append(action[0])
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

            history['cbf_active'].append(1 if proj_needed else 0)
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

            max_T = max(max_T, state[1] - 273.15)
            max_P = max(max_P, Power)
            max_HTO = max(max_HTO, hto)

        sim.step(last_action)

    return history, max_T, max_P, max_HTO, n_proj


def plot_comparison(hist_baseline, hist_cbf, output_dir):
    """Plot side-by-side comparison of baseline vs CBF-protected."""
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)

    t_b = np.array(hist_baseline['t'])
    t_c = np.array(hist_cbf['t'])

    # Row 1: Current
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t_b, hist_baseline['I_act'], 'r-', linewidth=2, label='Baseline (no CBF)')
    ax1.plot(t_c, hist_cbf['I_act'], 'b-', linewidth=2, label='With CBF')
    ax1.set_title('Current', fontsize=12, fontweight='bold')
    ax1.set_ylabel('A')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Row 1: Temperature
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t_b, hist_baseline['T_s'], 'r-', linewidth=2, label='Baseline')
    ax2.plot(t_c, hist_cbf['T_s'], 'b-', linewidth=2, label='With CBF')
    ax2.axhline(y=90, color='r', linestyle='--', alpha=0.7, label='T_max=90°C')
    ax2.axhline(y=80, color='g', linestyle='--', alpha=0.5, label='T_target')
    ax2.fill_between(t_c, 88, 90, alpha=0.1, color='r')
    ax2.set_title('Stack Temperature', fontsize=12, fontweight='bold')
    ax2.set_ylabel('°C')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Row 1: Power
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(t_b, hist_baseline['Power'], 'r-', linewidth=2, label='Baseline')
    ax3.plot(t_c, hist_cbf['Power'], 'b-', linewidth=2, label='With CBF')
    ax3.axhline(y=6.0, color='r', linestyle='--', alpha=0.7, label='P_max=6MW')
    ax3.set_title('Stack Power', fontsize=12, fontweight='bold')
    ax3.set_ylabel('MW')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Row 2: HTO
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(t_b, hist_baseline['HTO'], 'r-', linewidth=2, label='Baseline')
    ax4.plot(t_c, hist_cbf['HTO'], 'b-', linewidth=2, label='With CBF')
    ax4.axhline(y=2.0, color='r', linestyle='--', alpha=0.7, label='HTO_max=2%')
    ax4.set_title('HTO', fontsize=12, fontweight='bold')
    ax4.set_ylabel('%')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Row 2: CBF activation indicator
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.fill_between(t_c, 0, hist_cbf['cbf_active'], alpha=0.5, color='orange', step='post')
    ax5.set_ylim(-0.1, 1.5)
    ax5.set_title('CBF Activation', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Active')
    ax5.set_xlabel('Time (min)')
    ax5.grid(True, alpha=0.3)

    # Row 2: Control adjustment
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(t_c, hist_cbf['adjustment'], 'b-', linewidth=1.5)
    ax6.set_title('CBF Control Adjustment', fontsize=12, fontweight='bold')
    ax6.set_ylabel('||u - u_ref||')
    ax6.set_xlabel('Time (min)')
    ax6.grid(True, alpha=0.3)

    # Row 3: h margins (CBF case)
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(t_c, hist_cbf['h_T'], 'b-', linewidth=1.5, label='h_T')
    ax7.plot(t_c, hist_cbf['h_Tmin'], 'g-', linewidth=1.5, label='h_Tmin')
    ax7.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax7.set_title('h: Temperature (CBF)', fontsize=12, fontweight='bold')
    ax7.set_ylabel('h')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(t_c, hist_cbf['h_HTO'], 'r-', linewidth=1.5)
    ax8.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax8.set_title('h: HTO (CBF)', fontsize=12, fontweight='bold')
    ax8.set_ylabel('h')
    ax8.grid(True, alpha=0.3)

    ax9 = fig.add_subplot(gs[2, 2])
    ax9.plot(t_c, np.array(hist_cbf['h_P']) / 1e6, 'c-', linewidth=1.5, label='h_P')
    ax9.plot(t_c, hist_cbf['h_V'], 'm-', linewidth=1.5, label='h_V')
    ax9.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax9.set_title('h: Power & Voltage (CBF)', fontsize=12, fontweight='bold')
    ax9.set_ylabel('h')
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    # Row 4: CBF conditions
    ax10 = fig.add_subplot(gs[3, 0])
    ax10.plot(t_c, hist_cbf['cbf_T'], 'b-', linewidth=1.5)
    ax10.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax10.set_title('CBF: Temperature', fontsize=12, fontweight='bold')
    ax10.set_ylabel('L_f h + gamma*h')
    ax10.set_xlabel('Time (min)')
    ax10.grid(True, alpha=0.3)

    ax11 = fig.add_subplot(gs[3, 1])
    ax11.plot(t_c, hist_cbf['cbf_HTO'], 'r-', linewidth=1.5)
    ax11.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax11.set_title('CBF: HTO', fontsize=12, fontweight='bold')
    ax11.set_ylabel('L_f h + gamma*h')
    ax11.set_xlabel('Time (min)')
    ax11.grid(True, alpha=0.3)

    ax12 = fig.add_subplot(gs[3, 2])
    ax12.plot(t_c, hist_cbf['cbf_P'], 'c-', linewidth=1.5, label='CBF_P')
    ax12.plot(t_c, hist_cbf['cbf_Tmin'], 'g-', linewidth=1.5, label='CBF_Tmin')
    ax12.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax12.set_title('CBF: Power & Tmin', fontsize=12, fontweight='bold')
    ax12.set_ylabel('L_f h + gamma*h')
    ax12.set_xlabel('Time (min)')
    ax12.legend()
    ax12.grid(True, alpha=0.3)

    fig.suptitle(
        'CBF Dynamic Protection Test: Baseline vs CBF-Protected\n'
        'Baseline ramps current from 2000A to 7000A over 2 hours',
        fontsize=15, fontweight='bold'
    )
    out_path = os.path.join(output_dir, 'cbf_dynamic_protection.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {out_path}")


def main():
    output_dir = 'output/single_stack/cbf_tests/dynamic_protection'
    os.makedirs(output_dir, exist_ok=True)

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

    active_mask = np.array([True, True, True, True, True])

    # Scenario: current ramps from 2000A to 7000A over 2 hours
    # This will push temperature and power toward boundaries
    def I_target_func(t):
        """Ramp current: 2000A -> 7000A over 7200s."""
        return 2000.0 + (7000.0 - 2000.0) * min(t / 7200.0, 1.0)

    controller = BaselineController(I_init=2000.0, v_lye=0.01, v_c=0.001)

    print("=" * 70)
    print("RUN 1: Baseline Controller (NO CBF)")
    print("=" * 70)
    hist_baseline, max_T_b, max_P_b, max_HTO_b, n_proj_b = run_simulation(
        controller, projector, I_target_func,
        duration=10800.0, dt=0.2, ctrl_dt=60.0,
        use_cbf=False, active_mask=active_mask
    )
    print(f"Max Temperature: {max_T_b:.2f}°C")
    print(f"Max Power: {max_P_b:.3f} MW")
    print(f"Max HTO: {max_HTO_b:.3f}%")

    print("\n" + "=" * 70)
    print("RUN 2: Baseline + CBF Projection")
    print("=" * 70)
    # Reset controller
    controller = BaselineController(I_init=2000.0, v_lye=0.01, v_c=0.001)
    hist_cbf, max_T_c, max_P_c, max_HTO_c, n_proj_c = run_simulation(
        controller, projector, I_target_func,
        duration=10800.0, dt=0.2, ctrl_dt=60.0,
        use_cbf=True, active_mask=active_mask
    )
    print(f"Max Temperature: {max_T_c:.2f}°C")
    print(f"Max Power: {max_P_c:.3f} MW")
    print(f"Max HTO: {max_HTO_c:.3f}%")
    print(f"CBF activated: {n_proj_c} times")

    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<20} {'Baseline':>12} {'CBF':>12} {'Improvement':>12}")
    print("-" * 60)
    print(f"{'Max Temp (°C)':<20} {max_T_b:>12.2f} {max_T_c:>12.2f} {'Safe' if max_T_c <= 90 else 'UNSAFE':>12}")
    print(f"{'Max Power (MW)':<20} {max_P_b:>12.3f} {max_P_c:>12.3f} {'Safe' if max_P_c <= 6 else 'UNSAFE':>12}")
    print(f"{'Max HTO (%)':<20} {max_HTO_b:>12.3f} {max_HTO_c:>12.3f} {'Safe' if max_HTO_c <= 2 else 'UNSAFE':>12}")

    plot_comparison(hist_baseline, hist_cbf, output_dir)


if __name__ == '__main__':
    main()
