#!/usr/bin/env python3
"""
CBF Grid Scan Test for Single-Stack AWE

Systematically evaluates CBF projection across a grid of constant reference
controls (I, v_lye, v_c). For each combination, the system is run with a
constant control action, and the CBF projector modifies it as needed.

Metrics recorded per grid point:
  - Max Power, Max Temp, Max HTO
  - Average projected control values
  - Projection success rate
  - Maximum slack per constraint
  - Hard constraint violation flags
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from plant.single_stack_simulator import SingleStackSimulator
from controller.single_stack.cbf_projection import SingleStackCBFProjection


def run_single_test(u_ref, projector, duration=600.0, dt=0.2, ctrl_dt=60.0):
    """Run one grid-point test with constant reference control."""
    sim = SingleStackSimulator()
    sim.reset()

    steps = int(duration / dt)
    ctrl_steps = int(ctrl_dt / dt)

    last_action = u_ref.copy()

    # Recorders
    max_power = 0.0
    max_temp = 0.0
    max_hto = 0.0
    projection_successes = 0
    projection_total = 0
    slack_max = np.zeros(5)
    projected_actions = []

    for i in range(steps):
        if i % ctrl_steps == 0:
            state = sim.state
            action, success, info = projector.project(
                u_ref, state, u_last=last_action,
                active_mask=np.array([True, True, True, True, True])
            )
            last_action = action.copy()
            projection_total += 1
            if success:
                projection_successes += 1

            s = info.get('slack', np.zeros(5))
            slack_max = np.maximum(slack_max, np.abs(s))
            projected_actions.append(action.copy())

        # Record every 50 steps (~10s) to keep overhead low
        if i % 50 == 0:
            state = sim.state
            T_s = state[1] - 273.15
            n_gas = state[6]
            T_sep = state[2]
            hto = (n_gas * projector.R * T_sep) / (projector.P_sys * projector.V_sep_gas) * 100

            I_rec = last_action[0]
            _, U_cell, _ = sim._calculate_electrochemical_properties(I_rec, state[1])
            power = U_cell * I_rec * sim.N_cell / 1e6

            max_power = max(max_power, power)
            max_temp = max(max_temp, T_s)
            max_hto = max(max_hto, hto)

        sim.step(last_action)

    projected_actions = np.array(projected_actions)
    avg_projected = projected_actions.mean(axis=0) if len(projected_actions) > 0 else u_ref

    return {
        'I_ref': u_ref[0],
        'v_lye_ref': u_ref[1],
        'v_c_ref': u_ref[2],
        'max_power_mw': max_power,
        'max_temp_c': max_temp,
        'max_hto_pct': max_hto,
        'avg_I': avg_projected[0],
        'avg_v_lye': avg_projected[1],
        'avg_v_c': avg_projected[2],
        'success_rate': projection_successes / projection_total if projection_total > 0 else 0.0,
        'max_slack_T': slack_max[0],
        'max_slack_HTO': slack_max[1],
        'max_slack_V': slack_max[2],
        'max_slack_P': slack_max[3],
        'max_slack_Tmin': slack_max[4],
        'violated_power': int(max_power > 6.0),
        'violated_temp': int(max_temp > 90.0),
        'violated_hto': int(max_hto > 2.0),
    }


def plot_heatmaps(df, output_dir):
    """Generate heatmap visualizations for key metrics."""
    I_levels = sorted(df['I_ref'].unique())
    v_lye_levels = sorted(df['v_lye_ref'].unique())

    def pivot_for_vc(metric, vc_val):
        sub = df[np.isclose(df['v_c_ref'], vc_val)]
        piv = sub.pivot_table(index='I_ref', columns='v_lye_ref', values=metric, aggfunc='mean')
        # Ensure correct ordering
        piv = piv.reindex(index=I_levels, columns=v_lye_levels)
        return piv

    vc_levels = sorted(df['v_c_ref'].unique())
    metrics = [
        ('max_power_mw', 'Max Power (MW)', 'power'),
        ('max_temp_c', 'Max Temperature (°C)', 'temp'),
        ('max_hto_pct', 'Max HTO (%)', 'hto'),
        ('success_rate', 'Projection Success Rate', 'success'),
    ]

    for metric, title, fname_suffix in metrics:
        n_vc = len(vc_levels)
        fig, axes = plt.subplots(1, n_vc, figsize=(5 * n_vc, 4.5), sharey=True)
        if n_vc == 1:
            axes = [axes]

        vmin = df[metric].min()
        vmax = df[metric].max()

        for ax, vc in zip(axes, vc_levels):
            piv = pivot_for_vc(metric, vc)
            im = ax.imshow(piv.values, aspect='auto', origin='lower',
                           vmin=vmin, vmax=vmax, cmap='viridis')
            ax.set_xticks(range(len(v_lye_levels)))
            ax.set_xticklabels([f'{v:.2f}' for v in v_lye_levels])
            ax.set_yticks(range(len(I_levels)))
            ax.set_yticklabels([f'{I:.0f}' for I in I_levels])
            ax.set_title(f'v_c = {vc:.1f}')
            ax.set_xlabel('v_lye (m³/s)')
            if vc == vc_levels[0]:
                ax.set_ylabel('I (A)')

            # Annotate cells
            for i in range(len(I_levels)):
                for j in range(len(v_lye_levels)):
                    val = piv.values[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                                color='white' if val > (vmin + vmax) / 2 else 'black',
                                fontsize=8)

        fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.02, label=title)
        fig.suptitle(f'CBF Grid Scan: {title}', fontsize=13, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out_path = os.path.join(output_dir, f'gridscan_{fname_suffix}.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Heatmap saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='CBF Grid Scan Test')
    parser.add_argument('--output_dir', type=str, default='output/single_stack/cbf_tests/grid_scan',
                        help='Output directory for results')
    parser.add_argument('--duration', type=float, default=600.0,
                        help='Duration per grid point in seconds (default: 600)')
    parser.add_argument('--I_levels', type=float, nargs='+',
                        default=[0.0, 2500.0, 5000.0, 7500.0, 9360.0],
                        help='Current levels to test (A)')
    parser.add_argument('--v_lye_levels', type=float, nargs='+',
                        default=[0.01, 0.03, 0.05, 0.10],
                        help='Lye flow levels to test (m³/s)')
    parser.add_argument('--v_c_levels', type=float, nargs='+',
                        default=[0.0, 0.3, 0.6, 1.0],
                        help='Coolant flow levels to test (m³/s)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # CBF projector with final unified parameters
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

    total = len(args.I_levels) * len(args.v_lye_levels) * len(args.v_c_levels)
    print(f"CBF Grid Scan: {total} combinations")
    print(f"  I levels: {args.I_levels}")
    print(f"  v_lye levels: {args.v_lye_levels}")
    print(f"  v_c levels: {args.v_c_levels}")
    print(f"  Duration per point: {args.duration}s")
    print("=" * 60)

    results = []
    idx = 0
    t_start = time.time()

    for I in args.I_levels:
        for v_lye in args.v_lye_levels:
            for v_c in args.v_c_levels:
                idx += 1
                u_ref = np.array([I, v_lye, v_c])
                print(f"[{idx:3d}/{total}] I={I:6.0f}A  v_lye={v_lye:.2f}  v_c={v_c:.1f}  ...", end=' ', flush=True)
                t0 = time.time()
                res = run_single_test(u_ref, projector, duration=args.duration)
                elapsed = time.time() - t0
                results.append(res)
                print(f"Pmax={res['max_power_mw']:.2f}MW  Tmax={res['max_temp_c']:.1f}°C  "
                      f"HTOmax={res['max_hto_pct']:.2f}%  SR={res['success_rate']:.1%}  ({elapsed:.1f}s)")

    total_elapsed = time.time() - t_start
    print("=" * 60)
    print(f"Grid scan complete in {total_elapsed:.1f}s")

    df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, 'grid_scan_results.csv')
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"Results CSV saved: {csv_path}")

    # Summary statistics
    print("\nSummary:")
    print(f"  Violations: Power={df['violated_power'].sum()}  Temp={df['violated_temp'].sum()}  HTO={df['violated_hto'].sum()}")
    print(f"  Avg success rate: {df['success_rate'].mean():.2%}")
    print(f"  Worst cell: Pmax={df['max_power_mw'].max():.3f}MW  Tmax={df['max_temp_c'].max():.2f}°C  HTOmax={df['max_hto_pct'].max():.3f}%")

    plot_heatmaps(df, args.output_dir)
    print("\nAll outputs saved to:", args.output_dir)


if __name__ == '__main__':
    main()
