"""
HOCBF 专用测试脚本 —— 纯二阶 HTO 调节效果测试

Usage:
    python test_cbf_ho.py --scenario low_power_hto --lambda_u_scale 0
    python test_cbf_ho.py --alpha1 0.8 --alpha2 2.0 --gamma_hto 1.0
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from controller.multi_stack.cbf_projection_ho import MultiStackCBFProjectionHO
from plant.multi_stack_simulator import MultiStackSimulator


def run_simulation(projector, duration=3600, u_ref=None):
    """Run multi-stack simulation with HOCBF projector"""
    sim = MultiStackSimulator(dt=0.2)
    sim.reset()

    dt = 0.2
    steps = int(duration / dt)
    ctrl_steps = int(60.0 / dt)

    if u_ref is None:
        u_ref = np.array([600.0] * 4 + [0.02] * 4 + [0.01])
    current_action = u_ref.copy()
    last_action = current_action.copy()

    n_cbf = 9
    history = {
        't': [],
        'I_all': [], 'I_ref_all': [],
        'v_lye_all': [], 'v_lye_ref_all': [],
        'v_c': [], 'v_c_ref': [],
        'HTO': [], 'T_s_all': [],
        'h_raw': [], 'h_norm': [],
        'lie1_raw': [], 'lie1_norm': [],
        'lie2_raw': [], 'lie2_norm': [],
        'cbf_condition': [],
        'projection_success': [],
        'cost_ref': [], 'cost_delta': [], 'cost_slack': [], 'cost_total': [],
        'solve_time_ms': []
    }

    u_range = np.array([projector.I_max - projector.I_min] * 4 +
                       [projector.v_lye_max - projector.v_lye_min] * 4 +
                       [projector.v_c_max - projector.v_c_min])
    u_weights = 1.0 / (u_range ** 2)
    lambda_u = projector.lambda_u

    last_costs = {'ref': 0.0, 'delta': 0.0, 'slack': 0.0, 'total': 0.0}

    for i in range(steps):
        t = i * dt

        if i % ctrl_steps == 0:
            state = sim.state
            t0 = time.time()
            current_action, success, info = projector.project(u_ref, state, u_last=last_action)
            solve_time_ms = (time.time() - t0) * 1000.0

            s = info.get('slack', np.zeros(n_cbf))
            ref_cost = np.sum(u_weights * (current_action - u_ref) ** 2)
            delta_cost = np.sum(lambda_u * (current_action - last_action) ** 2)
            slack_cost = np.sum(projector.rho_vec * s ** 2)
            last_costs = {
                'ref': ref_cost,
                'delta': delta_cost,
                'slack': slack_cost,
                'total': ref_cost + delta_cost + slack_cost
            }

            last_action = current_action.copy()

            if not success and i % 500 == 0:
                print(f"    Warning: CBF solve failed at t={t/60:.1f}min, max_slack={info.get('max_slack', 'N/A'):.4f}")

        if i % 50 == 0:
            state = sim.state
            n_gas = state[12]
            T_sep = state[5]
            hto = (n_gas * projector.R * T_sep) / (projector.P_sys * projector.V_sep_gas)
            T_s_vec = state[1:5]

            lie1_raw, lie2_raw, h_raw, _ = projector._compute_lie_derivative(state, current_action)
            h_norm = h_raw / projector.h_scales
            lie1_norm = lie1_raw / np.maximum(np.abs(projector.lie_deriv_scales), 1e-10)
            lie2_norm = lie2_raw / np.maximum(np.abs(projector.lie_deriv_scales)**2, 1e-10)

            # Build CBF condition vector
            cbf_cond = np.zeros(n_cbf)
            for j in range(n_cbf):
                if j == 4:
                    psi1 = lie1_norm[4] + projector.alpha1_hto * h_norm[4]
                    cbf_cond[j] = lie2_norm[4] + projector.alpha2_hto * psi1
                else:
                    cbf_cond[j] = lie1_norm[j] + projector.gamma_vec[j] * h_norm[j]

            history['t'].append(t / 60)
            history['I_all'].append(current_action[0:4].copy())
            history['I_ref_all'].append(u_ref[0:4].copy())
            history['v_lye_all'].append(current_action[4:8].copy())
            history['v_lye_ref_all'].append(u_ref[4:8].copy())
            history['v_c'].append(current_action[8])
            history['v_c_ref'].append(u_ref[8])
            history['HTO'].append(hto * 100)
            history['T_s_all'].append(T_s_vec - 273.15)
            history['h_raw'].append(h_raw.copy())
            history['h_norm'].append(h_norm.copy())
            history['lie1_raw'].append(lie1_raw.copy())
            history['lie1_norm'].append(lie1_norm.copy())
            history['lie2_raw'].append(lie2_raw.copy())
            history['lie2_norm'].append(lie2_norm.copy())
            history['cbf_condition'].append(cbf_cond.copy())
            history['projection_success'].append(success)
            history['cost_ref'].append(last_costs['ref'])
            history['cost_delta'].append(last_costs['delta'])
            history['cost_slack'].append(last_costs['slack'])
            history['cost_total'].append(last_costs['total'])
            history['solve_time_ms'].append(solve_time_ms)

        sim.step(current_action)

    return history


def compute_metrics(history):
    """Compute quantitative metrics from history"""
    I_all = np.array(history['I_all'])
    HTO = np.array(history['HTO'])
    T_s_all = np.array(history['T_s_all'])

    # Max values
    max_hto = float(np.max(HTO))
    max_temp = float(np.max(T_s_all))
    max_I = float(np.max(I_all))

    # Current bounce analysis (sawtooth detection)
    dI = np.diff(I_all, axis=0)
    bounce_count = 0
    for j in range(4):
        for i in range(1, len(dI)):
            if dI[i-1, j] * dI[i, j] < 0 and abs(dI[i, j]) > 50:
                bounce_count += 1

    max_dI = float(np.max(np.abs(dI)))
    mean_dI = float(np.mean(np.abs(dI)))

    # HTO stability (std of last 20% of trajectory)
    n_last = max(1, len(HTO) // 5)
    hto_std_last = float(np.std(HTO[-n_last:]))

    # Solve time
    solve_times = np.array(history['solve_time_ms'])
    mean_solve = float(np.mean(solve_times))
    max_solve = float(np.max(solve_times))

    # Cost composition
    cost_delta = np.array(history['cost_delta'])
    mean_delta_cost = float(np.mean(cost_delta))

    return {
        'max_hto': max_hto,
        'max_temp': max_temp,
        'max_I': max_I,
        'bounce_count': bounce_count,
        'max_dI': max_dI,
        'mean_dI': mean_dI,
        'hto_std_last': hto_std_last,
        'mean_solve_ms': mean_solve,
        'max_solve_ms': max_solve,
        'mean_delta_cost': mean_delta_cost,
    }


def plot_single_test(history, config, output_path):
    """Plot HOCBF test results"""
    fig = plt.figure(figsize=(35, 30))
    gs = fig.add_gridspec(9, 5, hspace=0.35, wspace=0.3)

    t = np.array(history['t'])
    h_raw_arr = np.array(history['h_raw'])
    h_norm_arr = np.array(history['h_norm'])
    lie1_raw_arr = np.array(history['lie1_raw'])
    lie1_norm_arr = np.array(history['lie1_norm'])
    lie2_raw_arr = np.array(history['lie2_raw'])
    lie2_norm_arr = np.array(history['lie2_norm'])
    cbf_arr = np.array(history['cbf_condition'])
    I_all = np.array(history['I_all'])
    T_s_all = np.array(history['T_s_all'])

    failure_indices = [i for i, success in enumerate(history['projection_success']) if not success]
    failure_times = [t[i] for i in failure_indices] if failure_indices else []

    metrics = compute_metrics(history)

    # Row 0: Current
    ax = fig.add_subplot(gs[0, 0])
    for j in range(4):
        ax.plot(t, I_all[:, j], label=f'Stack {j+1}', alpha=0.7)
    ax.plot(t, np.array(history['I_ref_all'])[:, 0], 'k--', linewidth=1.5, alpha=0.5, label='Reference')
    ax.set_title('Stack Currents')
    ax.set_ylabel('A')
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True)

    # Row 0: Lye flow
    ax = fig.add_subplot(gs[0, 1])
    v_lye_all = np.array(history['v_lye_all']) * 1000
    for j in range(4):
        ax.plot(t, v_lye_all[:, j], label=f'Stack {j+1}', alpha=0.7)
    ax.plot(t, np.array(history['v_lye_ref_all'])[:, 0] * 1000, 'k--', linewidth=1.5, alpha=0.5)
    ax.set_title('Lye Flows')
    ax.set_ylabel('L/s')
    ax.grid(True)

    # Row 0: v_c
    ax = fig.add_subplot(gs[0, 2])
    v_c_arr = np.array(history['v_c']) * 1000
    v_c_ref_arr = np.array(history['v_c_ref']) * 1000
    ax.plot(t, v_c_arr, 'b-', linewidth=2, label='Actual')
    ax.plot(t, v_c_ref_arr, 'r--', linewidth=1.5, alpha=0.7, label='Reference')
    if failure_times:
        for ft in failure_times:
            ax.axvline(x=ft, color='red', alpha=0.3, linewidth=0.8)
    ax.set_title('Cooling Water Flow (v_c)')
    ax.set_ylabel('L/s')
    ax.legend(fontsize=8)
    ax.grid(True)

    # Row 1: HTO & Temperatures
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t, history['HTO'], 'b-', linewidth=2)
    ax.axhline(y=2.0, color='r', linestyle='--', label='Limit (2%)')
    effective_limit = 2.0 - config.get('h_margin_vec', [0]*9)[4] * 100
    if effective_limit < 2.0:
        ax.axhline(y=effective_limit, color='orange', linestyle=':', label=f'Effective ({effective_limit:.1f}%)')
    ax.fill_between(t, 2.0, max(history['HTO']) * 1.2, alpha=0.1, color='red', where=(np.array(history['HTO']) > 2.0))
    ax.set_title('HTO (%)')
    ax.set_ylabel('%')
    ax.legend()
    ax.grid(True)

    ax = fig.add_subplot(gs[1, 1])
    for j in range(4):
        ax.plot(t, T_s_all[:, j], label=f'Stack {j+1}', alpha=0.7)
    ax.axhline(y=90, color='r', linestyle='--', label='T_max')
    ax.set_title('Stack Temperatures')
    ax.set_ylabel('°C')
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True)

    # dI/dt for oscillation detection
    ax = fig.add_subplot(gs[1, 2])
    dI = np.diff(I_all, axis=0)
    for j in range(4):
        ax.plot(t[1:], dI[:, j], label=f'Stack {j+1}', alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_title('Current Change per Step (dI)')
    ax.set_ylabel('A/step')
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True)

    # Constraint indices
    idx_T = slice(0, 4)
    idx_HTO = 4
    idx_Tmin = slice(5, 9)
    colors = {'T': 'b', 'HTO': 'r', 'Tmin': 'g'}

    def plot_multi_stack_constraint(ax, data_arr, cidx, color_key, title, ylabel, fill_lower=None, xlabel=None):
        color = colors[color_key]
        if isinstance(cidx, slice):
            for j in range(cidx.start, cidx.stop):
                ax.plot(t, data_arr[:, j], color=color, alpha=0.7, linewidth=1.5,
                        label=f'Stack {j - cidx.start + 1}')
            ax.legend(ncol=2, fontsize=7)
            if fill_lower is not None:
                mask = np.any(data_arr[:, cidx.start:cidx.stop] < 0, axis=1)
                if np.any(mask):
                    ax.fill_between(t, fill_lower, 0, alpha=0.2, color='red', where=mask)
        else:
            ax.plot(t, data_arr[:, cidx], color=color, linewidth=2, label='HTO')
            ax.legend(fontsize=8)
            if fill_lower is not None:
                mask = data_arr[:, cidx] < 0
                if np.any(mask):
                    ax.fill_between(t, fill_lower, 0, alpha=0.2, color='red', where=mask)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if xlabel:
            ax.set_xlabel(xlabel)
        ax.grid(True)

    # Row 2: h_T
    plot_multi_stack_constraint(fig.add_subplot(gs[2, 0]), h_raw_arr, idx_T, 'T', 'Raw h_T', 'h', fill_lower=-100)
    plot_multi_stack_constraint(fig.add_subplot(gs[2, 1]), h_norm_arr, idx_T, 'T', 'Normalized h_T', 'h_norm', fill_lower=-2)
    plot_multi_stack_constraint(fig.add_subplot(gs[2, 2]), lie1_raw_arr, idx_T, 'T', 'L_f h - Temperature', 'L_f h')
    plot_multi_stack_constraint(fig.add_subplot(gs[2, 3]), lie1_norm_arr, idx_T, 'T', 'Normalized L_f h - T', 'L_f h_norm')
    plot_multi_stack_constraint(fig.add_subplot(gs[2, 4]), cbf_arr, idx_T, 'T', 'CBF Condition - T', 'L_f h + gamma*h', fill_lower=-10)

    # Row 3: h_HTO (with lie2)
    plot_multi_stack_constraint(fig.add_subplot(gs[3, 0]), h_raw_arr, idx_HTO, 'HTO', 'Raw h_HTO', 'h', fill_lower=-0.01)
    plot_multi_stack_constraint(fig.add_subplot(gs[3, 1]), h_norm_arr, idx_HTO, 'HTO', 'Normalized h_HTO', 'h_norm', fill_lower=-2)
    plot_multi_stack_constraint(fig.add_subplot(gs[3, 2]), lie1_raw_arr, idx_HTO, 'HTO', 'L_f h - HTO', 'L_f h')
    plot_multi_stack_constraint(fig.add_subplot(gs[3, 3]), lie2_raw_arr, idx_HTO, 'HTO', 'L_f^2 h - HTO', 'L_f^2 h')
    plot_multi_stack_constraint(fig.add_subplot(gs[3, 4]), cbf_arr, idx_HTO, 'HTO', 'CBF Condition - HTO', 'psi_2', fill_lower=-10)

    # Row 4: h_Tmin
    plot_multi_stack_constraint(fig.add_subplot(gs[4, 0]), h_raw_arr, idx_Tmin, 'Tmin', 'Raw h_Tmin', 'h', fill_lower=-100)
    plot_multi_stack_constraint(fig.add_subplot(gs[4, 1]), h_norm_arr, idx_Tmin, 'Tmin', 'Normalized h_Tmin', 'h_norm', fill_lower=-2)
    plot_multi_stack_constraint(fig.add_subplot(gs[4, 2]), lie1_raw_arr, idx_Tmin, 'Tmin', 'L_f h - Tmin', 'L_f h')
    plot_multi_stack_constraint(fig.add_subplot(gs[4, 3]), lie1_norm_arr, idx_Tmin, 'Tmin', 'Normalized L_f h - Tmin', 'L_f h_norm')
    plot_multi_stack_constraint(fig.add_subplot(gs[4, 4]), cbf_arr, idx_Tmin, 'Tmin', 'CBF Condition - Tmin', 'L_f h + gamma*h', fill_lower=-10, xlabel='Time (min)')

    # Row 5: psi_1 for HTO
    psi1_arr = lie1_norm_arr[:, 4] + config['alpha1_hto'] * h_norm_arr[:, 4]
    ax = fig.add_subplot(gs[5, :])
    ax.plot(t, psi1_arr, 'b-', linewidth=2, label='psi_1 = L_f h + alpha1*h')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.set_title('HTO First-Order Intermediate (psi_1)')
    ax.set_ylabel('psi_1')
    ax.set_xlabel('Time (min)')
    ax.legend()
    ax.grid(True)

    # Row 6: Cost Composition & Solver Timing
    cost_ref = np.array(history['cost_ref'])
    cost_delta = np.array(history['cost_delta'])
    cost_slack = np.array(history['cost_slack'])
    cost_total = np.array(history['cost_total'])
    solve_time_ms = np.array(history['solve_time_ms'])

    ax_cost = fig.add_subplot(gs[6, 0:3])
    ax_cost.stackplot(t, cost_ref, cost_delta, cost_slack,
                      labels=['Ref Tracking', 'Control Change', 'Slack Penalty'],
                      colors=['#1f77b4', '#ff7f0e', '#d62728'], alpha=0.7)
    ax_cost.plot(t, cost_total, 'k--', linewidth=1.5, label='Total Cost')
    ax_cost.set_title('CBF Projection Cost Composition')
    ax_cost.set_ylabel('Cost')
    ax_cost.legend(loc='upper left')
    ax_cost.grid(True, alpha=0.3)

    ax_solve = fig.add_subplot(gs[6, 3:])
    ax_solve.plot(t, solve_time_ms, 'g-', linewidth=1.5, label='Solve Time')
    ax_solve.axhline(y=np.mean(solve_time_ms), color='r', linestyle='--', label=f'Mean={np.mean(solve_time_ms):.1f}ms')
    ax_solve.fill_between(t, 0, solve_time_ms, alpha=0.2, color='green')
    if len(solve_time_ms) > 0:
        ax_solve.axhline(y=np.max(solve_time_ms), color='orange', linestyle=':', alpha=0.7, label=f'Max={np.max(solve_time_ms):.1f}ms')
    ax_solve.set_title('CBF Optimization Solver Time')
    ax_solve.set_ylabel('Time (ms)')
    ax_solve.legend(loc='upper right')
    ax_solve.grid(True, alpha=0.3)

    # Summary
    ax_sum = fig.add_subplot(gs[7:, :])
    ax_sum.axis('off')

    max_hto = max(history['HTO'])
    max_temp = max(T_s_all.flatten())
    max_I = max(I_all.flatten())
    n_failures = len(failure_indices)
    total_points = len(history['projection_success'])

    summary = f"""
    HOCBF Test Configuration:
    -------------------------
    lambda_u_scale: {config['lambda_u_scale']}
    alpha1_hto: {config['alpha1_hto']}
    alpha2_hto: {config['alpha2_hto']}
    gamma_vec: {config['gamma_vec']}
    rho_vec: {config['rho_vec']}
    h_margin_vec: {config['h_margin_vec']}
    normalize: {config['normalize']}
    duration: {config['duration']}s

    Quantitative Metrics:
    ---------------------
    Max HTO: {metrics['max_hto']:.3f}%
    Max Temperature: {metrics['max_temp']:.2f}°C
    Max Current: {metrics['max_I']:.0f}A
    Bounce Count: {metrics['bounce_count']}
    Max |dI|: {metrics['max_dI']:.1f} A/step
    Mean |dI|: {metrics['mean_dI']:.1f} A/step
    HTO Std (last 20%): {metrics['hto_std_last']:.4f}

    Solver Status:
    --------------
    Total Steps: {total_points}
    Solve Failures: {n_failures} ({100*n_failures/total_points:.1f}%)

    Solver Timing:
    --------------
    Mean Solve Time: {metrics['mean_solve_ms']:.2f}ms
    Max Solve Time: {metrics['max_solve_ms']:.2f}ms
    Min Solve Time: {np.min(solve_time_ms):.2f}ms
    """
    if n_failures > 0:
        summary += f"    First Failure: t={t[failure_indices[0]]:.1f}min\n"

    ax_sum.text(0.1, 0.5, summary, fontsize=11, fontfamily='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle(f'HOCBF Test: {config["test_name"]}', fontsize=14, fontweight='bold')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: {output_path}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description='HOCBF Test and Visualization')
    parser.add_argument('--scenario', choices=['default', 'low_power_hto', 'high_temp', 'high_power'], default='low_power_hto')
    parser.add_argument('--gamma_vec', type=float, nargs=9,
                       default=[10.0]*4 + [1.0] + [10.0]*4)
    parser.add_argument('--rho_vec', type=float, nargs=9, default=[5000]*9)
    parser.add_argument('--h_margin_vec', type=float, nargs=9,
                       default=[0.0]*4 + [0.007] + [0.0]*4)
    parser.add_argument('--lambda_u_scale', type=float, default=0.0,
                       help='Control change penalty scale (default: 0 for pure HOCBF)')
    parser.add_argument('--alpha1_hto', type=float, default=0.8)
    parser.add_argument('--alpha2_hto', type=float, default=2.0)
    parser.add_argument('--alpha1_vec', type=float, nargs=9, default=None,
                        help='Per-constraint alpha1 (9 elements). Overrides alpha1_hto.')
    parser.add_argument('--alpha2_vec', type=float, nargs=9, default=None,
                        help='Per-constraint alpha2 (9 elements). Set >0 to enable 2nd-order for each constraint.')
    parser.add_argument('--u_weight_scale', type=float, nargs=9, default=None,
                        help='Per-control weight scale for reference tracking (9 elements). '
                             'Larger = more reluctant to deviate from reference. '
                             'Example: prioritize v_c over I: "10 10 10 10 1 1 1 1 0.1"')
    parser.add_argument('--soft_mask', type=int, nargs=9, default=[1]*9)
    parser.add_argument('--normalize', action='store_true', default=True)
    parser.add_argument('--duration', type=int, default=3600)
    parser.add_argument('--output_dir', type=str, default='output/multi_stack/cbf_tests')
    parser.add_argument('--v_lye', type=float, default=None)

    args = parser.parse_args()

    print("="*70)
    print("HOCBF Test — Higher-Order CBF Regulation")
    print("="*70)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.scenario == 'low_power_hto':
        v_lye = args.v_lye if args.v_lye is not None else 0.05
        u_ref = np.array([600.0] * 4 + [v_lye] * 4 + [0.01])
        test_name = 'HOCBF_LowPower_HTO'
        print(f"\nRunning LOW POWER HTO test...")
        print(f"  Reference: I=[600A x4], v_lye=[{v_lye:.2f} x4], v_c=0.01")
    elif args.scenario == 'high_temp':
        v_lye = args.v_lye if args.v_lye is not None else 0.03
        u_ref = np.array([7500.0] * 4 + [v_lye] * 4 + [0.0])
        test_name = 'HOCBF_HighTemperature'
        print(f"\nRunning HIGH TEMPERATURE test...")
    elif args.scenario == 'high_power':
        v_lye = args.v_lye if args.v_lye is not None else 0.03
        u_ref = np.array([9360.0] * 4 + [v_lye] * 4 + [0.5])
        test_name = 'HOCBF_HighPower'
        print(f"\nRunning HIGH POWER test...")
    else:
        test_name = 'HOCBF_Default'

    gamma_vec = list(args.gamma_vec)
    rho_vec = list(args.rho_vec)
    h_margin_vec = list(args.h_margin_vec)
    soft_mask_bool = [bool(x) for x in args.soft_mask]

    alpha1_vec = list(args.alpha1_vec) if args.alpha1_vec is not None else None
    alpha2_vec = list(args.alpha2_vec) if args.alpha2_vec is not None else None
    u_weight_scale = list(args.u_weight_scale) if args.u_weight_scale is not None else None

    print(f"  lambda_u_scale: {args.lambda_u_scale} (0 = pure HOCBF)")
    if alpha1_vec is not None:
        print(f"  alpha1_vec: {alpha1_vec}")
    else:
        print(f"  alpha1_hto: {args.alpha1_hto}")
    if alpha2_vec is not None:
        print(f"  alpha2_vec: {alpha2_vec}")
    else:
        print(f"  alpha2_hto: {args.alpha2_hto}")
    if u_weight_scale is not None:
        print(f"  u_weight_scale: {u_weight_scale}")
    print(f"  gamma_vec: {gamma_vec}")
    print(f"  rho_vec: {rho_vec}")
    print(f"  h_margin_vec: {h_margin_vec}")

    projector = MultiStackCBFProjectionHO(
        dt=60.0,
        gamma_vec=gamma_vec,
        rho_vec=rho_vec,
        h_margin_vec=h_margin_vec,
        normalize=args.normalize,
        lambda_u_scale=args.lambda_u_scale,
        soft_mask=soft_mask_bool,
        alpha1_hto=args.alpha1_hto,
        alpha2_hto=args.alpha2_hto,
        alpha1_vec=alpha1_vec,
        alpha2_vec=alpha2_vec,
        u_weight_scale=u_weight_scale
    )

    history = run_simulation(projector, duration=args.duration, u_ref=u_ref)

    config = {
        'lambda_u_scale': args.lambda_u_scale,
        'alpha1_hto': args.alpha1_hto,
        'alpha2_hto': args.alpha2_hto,
        'alpha1_vec': alpha1_vec,
        'alpha2_vec': alpha2_vec,
        'u_weight_scale': u_weight_scale,
        'gamma_vec': gamma_vec,
        'rho_vec': rho_vec,
        'h_margin_vec': h_margin_vec,
        'normalize': args.normalize,
        'duration': args.duration,
        'test_name': test_name
    }

    output_path = os.path.join(args.output_dir, 'hocbf_test.png')
    metrics = plot_single_test(history, config, output_path)

    print(f"\nQuantitative Results:")
    print(f"  Max HTO: {metrics['max_hto']:.3f}%")
    print(f"  Max Temp: {metrics['max_temp']:.2f}°C")
    print(f"  Max Current: {metrics['max_I']:.0f}A")
    print(f"  Bounce Count: {metrics['bounce_count']}")
    print(f"  Max |dI|: {metrics['max_dI']:.1f} A/step")
    print(f"  Mean |dI|: {metrics['mean_dI']:.1f} A/step")
    print(f"  HTO Std (last 20%): {metrics['hto_std_last']:.4f}")
    print(f"  Mean Solve Time: {metrics['mean_solve_ms']:.2f}ms")
    print("="*70)


if __name__ == "__main__":
    main()
