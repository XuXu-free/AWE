#!/usr/bin/env python3
"""
CBF Conservatism Evaluation for Single-Stack AWE

Evaluates whether the CBF projector is overly conservative by checking:
1. How often the projected control differs from the reference control
   when all constraints are far from their boundaries.
2. The relationship between constraint margins and projection activation.
3. The magnitude of control adjustments at different safety margins.

A well-tuned CBF should:
- Return u_ref unchanged when all h(x,u) >> 0 (far from boundary)
- Only activate when constraints are genuinely threatened
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


def run_conservatism_analysis(u_ref, projector, duration=600.0, dt=0.2, ctrl_dt=60.0):
    """Run one test and record detailed projection activation data."""
    sim = SingleStackSimulator()
    sim.reset()

    steps = int(duration / dt)
    ctrl_steps = int(ctrl_dt / dt)
    last_action = u_ref.copy()

    records = []
    total_steps = 0
    activated_steps = 0

    for i in range(steps):
        if i % ctrl_steps == 0:
            state = sim.state
            # Use u_last = u_ref for conservatism evaluation so that delta_cost = 0.
            # This isolates CBF constraint activation from control smoothing effects.
            action, success, info = projector.project(
                u_ref, state, u_last=u_ref,
                active_mask=np.array([True, True, True, True, True])
            )
            last_action = action.copy()

            h_raw = info.get('h', np.zeros(5))
            cbf = info.get('cbf', np.zeros(5))
            cbf_ref = info.get('cbf_ref', cbf)
            slack = info.get('slack', np.zeros(5))
            adjustment = np.linalg.norm(action - u_ref)
            activated = adjustment > 1e-3  # meaningful adjustment

            # True conservatism: cbf_ref >= 0 at u_ref, but optimizer still deviated.
            # Use cbf_ref (original at u_ref) instead of cbf (optimized).
            all_cbf_safe = np.all(cbf_ref >= -1e-6)
            all_h_safe = np.all(h_raw >= 0)

            records.append({
                't': i * dt,
                'I_proj': action[0],
                'v_lye_proj': action[1],
                'v_c_proj': action[2],
                'h_T': h_raw[0],
                'h_HTO': h_raw[1],
                'h_V': h_raw[2],
                'h_P': h_raw[3],
                'h_Tmin': h_raw[4],
                'cbf_T': cbf[0],
                'cbf_HTO': cbf[1],
                'cbf_V': cbf[2],
                'cbf_P': cbf[3],
                'cbf_Tmin': cbf[4],
                'cbf_ref_T': cbf_ref[0],
                'cbf_ref_HTO': cbf_ref[1],
                'cbf_ref_V': cbf_ref[2],
                'cbf_ref_P': cbf_ref[3],
                'cbf_ref_Tmin': cbf_ref[4],
                's_T': slack[0],
                's_HTO': slack[1],
                's_V': slack[2],
                's_P': slack[3],
                's_Tmin': slack[4],
                'activated': int(activated),
                'all_cbf_safe': int(all_cbf_safe),
                'all_h_safe': int(all_h_safe),
                'adjustment_norm': adjustment,
            })

            total_steps += 1
            if activated:
                activated_steps += 1

        sim.step(last_action)

    df = pd.DataFrame(records)
    activation_rate = activated_steps / total_steps if total_steps > 0 else 0.0

    return df, activation_rate


def analyze_grid(output_dir, projector, duration=600.0):
    """Run conservatism analysis across a grid of operating points."""
    I_levels = [0.0, 2500.0, 5000.0, 7500.0, 9360.0]
    v_lye_levels = [0.01, 0.03, 0.05, 0.10]
    v_c_levels = [0.0, 0.3, 0.6, 1.0]

    results = []
    total = len(I_levels) * len(v_lye_levels) * len(v_c_levels)
    idx = 0

    for I in I_levels:
        for v_lye in v_lye_levels:
            for v_c in v_c_levels:
                idx += 1
                u_ref = np.array([I, v_lye, v_c])
                print(f"[{idx:3d}/{total}] I={I:6.0f}A  v_lye={v_lye:.2f}  v_c={v_c:.1f}  ...", end=' ', flush=True)
                t0 = time.time()

                df, activation_rate = run_conservatism_analysis(u_ref, projector, duration=duration)

                # Compute statistics
                safe_but_activated = len(df[(df['all_cbf_safe'] == 1) & (df['activated'] == 1)])
                total_safe = len(df[df['all_cbf_safe'] == 1])
                conservative_rate = safe_but_activated / total_safe if total_safe > 0 else 0.0

                avg_adjustment_when_activated = df[df['activated'] == 1]['adjustment_norm'].mean() if df['activated'].sum() > 0 else 0.0
                max_adjustment = df['adjustment_norm'].max()

                # Which constraint was closest to boundary when activated
                activated_df = df[df['activated'] == 1]
                if len(activated_df) > 0:
                    min_h_when_activated = activated_df[['h_T', 'h_HTO', 'h_V', 'h_P', 'h_Tmin']].min().min()
                    max_cbf_when_activated = activated_df[['cbf_T', 'cbf_HTO', 'cbf_V', 'cbf_P', 'cbf_Tmin']].max().max()
                else:
                    min_h_when_activated = np.nan
                    max_cbf_when_activated = np.nan

                results.append({
                    'I_ref': I,
                    'v_lye_ref': v_lye,
                    'v_c_ref': v_c,
                    'activation_rate': activation_rate,
                    'conservative_rate': conservative_rate,
                    'safe_but_activated': safe_but_activated,
                    'total_safe': total_safe,
                    'avg_adjustment': avg_adjustment_when_activated,
                    'max_adjustment': max_adjustment,
                    'min_h_when_activated': min_h_when_activated,
                    'max_cbf_when_activated': max_cbf_when_activated,
                })

                elapsed = time.time() - t0
                print(f"AR={activation_rate:.1%}  CR={conservative_rate:.1%}  ({elapsed:.1f}s)")

    df_results = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'conservatism_analysis.csv')
    df_results.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\nResults saved: {csv_path}")
    return df_results


def plot_conservatism_heatmaps(df, output_dir):
    """Generate heatmaps for conservatism metrics."""
    I_levels = sorted(df['I_ref'].unique())
    v_lye_levels = sorted(df['v_lye_ref'].unique())
    vc_levels = sorted(df['v_c_ref'].unique())

    metrics = [
        ('activation_rate', 'CBF Activation Rate', 'activation'),
        ('conservative_rate', 'Conservative Rate\n(activated when all cbf>=0)', 'conservative'),
        ('avg_adjustment', 'Avg Adjustment Norm\n(when activated)', 'adjustment'),
        ('max_adjustment', 'Max Adjustment Norm', 'max_adjust'),
    ]

    for metric, title, fname in metrics:
        n_vc = len(vc_levels)
        fig, axes = plt.subplots(1, n_vc, figsize=(5 * n_vc, 4.5), sharey=True)
        if n_vc == 1:
            axes = [axes]

        vmin = df[metric].min()
        vmax = df[metric].max()
        if np.isnan(vmin):
            continue

        for ax, vc in zip(axes, vc_levels):
            sub = df[np.isclose(df['v_c_ref'], vc)]
            piv = sub.pivot_table(index='I_ref', columns='v_lye_ref', values=metric, aggfunc='mean')
            piv = piv.reindex(index=I_levels, columns=v_lye_levels)

            im = ax.imshow(piv.values, aspect='auto', origin='lower',
                           vmin=vmin, vmax=vmax, cmap='RdYlGn_r')
            ax.set_xticks(range(len(v_lye_levels)))
            ax.set_xticklabels([f'{v:.2f}' for v in v_lye_levels])
            ax.set_yticks(range(len(I_levels)))
            ax.set_yticklabels([f'{I:.0f}' for I in I_levels])
            ax.set_title(f'v_c = {vc:.1f}')
            ax.set_xlabel('v_lye (m³/s)')
            if vc == vc_levels[0]:
                ax.set_ylabel('I (A)')

            for i in range(len(I_levels)):
                for j in range(len(v_lye_levels)):
                    val = piv.values[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                                color='white' if val > (vmin + vmax) / 2 else 'black',
                                fontsize=8)

        fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.02, label=title)
        fig.suptitle(f'CBF Conservatism: {title}', fontsize=13, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out_path = os.path.join(output_dir, f'conservatism_{fname}.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Heatmap saved: {out_path}")


def print_summary(df):
    """Print text summary of conservatism issues."""
    print("\n" + "=" * 70)
    print("CBF CONSERVATISM SUMMARY")
    print("=" * 70)

    total_runs = len(df)
    runs_with_conservatism = len(df[df['conservative_rate'] > 0])
    avg_conservative_rate = df['conservative_rate'].mean()

    print(f"\nOverall Statistics:")
    print(f"  Total grid points: {total_runs}")
    print(f"  Points with conservatism issues: {runs_with_conservatism} ({runs_with_conservatism/total_runs:.1%})")
    print(f"  Average conservative rate: {avg_conservative_rate:.2%}")

    print(f"\nTop 10 Most Conservative Operating Points:")
    worst = df.nlargest(10, 'conservative_rate')
    print(worst[['I_ref', 'v_lye_ref', 'v_c_ref', 'conservative_rate',
                 'activation_rate', 'avg_adjustment']].to_string(index=False))

    print(f"\nOperating Points with ZERO Conservatism (ideal behavior):")
    ideal = df[df['conservative_rate'] == 0.0]
    print(f"  Count: {len(ideal)} / {total_runs}")
    if len(ideal) > 0:
        print(ideal[['I_ref', 'v_lye_ref', 'v_c_ref', 'activation_rate']].to_string(index=False))

    # Correlation analysis
    print(f"\nCorrelation between h margins and activation:")
    print(f"  (Lower min_h_when_activated means activated closer to boundary)")
    close_to_boundary = df[df['min_h_when_activated'] < 0]
    far_from_boundary = df[df['min_h_when_activated'] > 100]
    print(f"  Activations with h < 0 (truly needed): {len(close_to_boundary)} points")
    print(f"  Activations with h > 100 (unnecessarily conservative): {len(far_from_boundary)} points")


def main():
    parser = argparse.ArgumentParser(description='CBF Conservatism Evaluation')
    parser.add_argument('--output_dir', type=str,
                        default='output/single_stack/cbf_tests/conservatism',
                        help='Output directory')
    parser.add_argument('--duration', type=float, default=600.0,
                        help='Duration per grid point (default: 600s)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

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

    df = analyze_grid(args.output_dir, projector, duration=args.duration)
    print_summary(df)
    plot_conservatism_heatmaps(df, args.output_dir)

    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
