#!/usr/bin/env python3
"""
Single-Stack TCN Diffusion + CBF with Real Wind Data (Full Diagnostics)

Based on run_nmpc_test.py structure:
- Loads monthly wind power profile
- NMPC warm-up phase
- 24h test with TCN Diffusion + CBF projection
- Records complete CBF constraint reactions
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import argparse
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from plant.single_stack_simulator import SingleStackSimulator
from controller.single_stack.cbf_model_controller import SingleStackCBFModelController
from controller.single_stack.nmpc_controller import SingleStackNMPCController


def load_wind_profile(month='02'):
    """Load monthly wind power data."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    profile_path = os.path.join(project_root, 'output', 'power', 'wind', f'wind_power_2025-{month}_1min.csv')
    if not os.path.exists(profile_path):
        raise FileNotFoundError(f"Wind profile not found at {profile_path}")
    print(f"Loading wind profile from {profile_path}...")
    df = pd.read_csv(profile_path)
    return df['P_ref'].values * 0.25  # Scale for single stack


def run_warmup(sim, ctrl, last_action, sim_dt, T_ref=353.15, month='02'):
    """NMPC warm-up phase."""
    warmup_duration = 4 * 60 * 60
    warmup_steps = int(warmup_duration / sim_dt)
    full_profile = load_wind_profile(month)
    P_warmup_start = 2.0e6
    P_warmup_end = full_profile[0]
    print(f"Starting Warm-up Phase ({warmup_duration}s)...")
    print(f"Ramping from {P_warmup_start/1e6}MW to {P_warmup_end/1e6}MW")
    horizon = getattr(ctrl, 'N_p', getattr(ctrl, 'horizon', 5))
    ctrl_steps = int(ctrl.dt / sim_dt)
    total_ctrl_steps = warmup_steps // ctrl_steps

    # Record warmup history
    wh = {
        't': [], 'P_ref': [], 'P_real': [],
        'T_s': [], 'T_sep': [], 'T_c_out': [],
        'I': [], 'v_lye': [], 'v_c': [],
        'U_cell': [], 'HTO': [],
        'cbf_active': [], 'adjustment': [],
        'h_T': [], 'h_HTO': [], 'h_V': [], 'h_P': [], 'h_Tmin': [],
        'cbf_T': [], 'cbf_HTO': [], 'cbf_V': [], 'cbf_P': [], 'cbf_Tmin': [],
        'T_viol': [], 'P_viol': [], 'HTO_viol': [], 'V_viol': [],
    }

    with tqdm(total=total_ctrl_steps, desc="Warmup", unit="ctrl_step") as pbar:
        for i in range(warmup_steps):
            t_warmup = -warmup_duration + i * sim_dt
            alpha = i / max(1, warmup_steps - 1)
            current_P_ref = P_warmup_start + alpha * (P_warmup_end - P_warmup_start)
            measured_state = sim.state.copy()
            T_s_in, T_s, T_sep, T_c_out = measured_state[0], measured_state[1], measured_state[2], measured_state[3]
            n_H2_an, n_liq, n_gas = measured_state[4], measured_state[5], measured_state[6]

            _, U_cell, _ = sim._calculate_electrochemical_properties(last_action[0], T_s)
            P_real = U_cell * last_action[0] * sim.N_cell
            hto_pct = (n_gas * sim.R * T_sep) / (sim.P_sys * sim.V_sep_gas) * 100

            if i % ctrl_steps == 0:
                P_future = []
                for k in range(horizon):
                    future_idx = i + k * ctrl_steps
                    if future_idx < warmup_steps:
                        alpha_f = future_idx / max(1, warmup_steps - 1)
                        val = P_warmup_start + alpha_f * (P_warmup_end - P_warmup_start)
                    else:
                        real_idx = int((future_idx - warmup_steps) * sim_dt / 60)
                        val = full_profile[real_idx] if real_idx < len(full_profile) else full_profile[-1]
                    P_future.append(val)
                try:
                    I_cmd, v_lye_cmd, v_c_cmd = ctrl.get_action(
                        measured_state, P_future, T_ref=T_ref, last_action=last_action
                    )
                except Exception as e:
                    print(f"Controller failed at warmup step {i}: {e}")
                    I_cmd, v_lye_cmd, v_c_cmd = last_action
                last_action = [I_cmd, v_lye_cmd, v_c_cmd]
                pbar.update(1)

            action_sim = np.array(last_action)
            sim.step(action_sim)

            # Log every 10s
            if i % 10 == 0:
                wh['t'].append(t_warmup)
                wh['P_ref'].append(current_P_ref)
                wh['P_real'].append(P_real)
                wh['T_s'].append(T_s - 273.15)
                wh['T_sep'].append(T_sep - 273.15)
                wh['T_c_out'].append(T_c_out - 273.15)
                wh['I'].append(last_action[0])
                wh['v_lye'].append(last_action[1])
                wh['v_c'].append(last_action[2])
                wh['U_cell'].append(U_cell)
                wh['HTO'].append(hto_pct)
                wh['cbf_active'].append(0)
                wh['adjustment'].append(0.0)
                for k in ['h_T','h_HTO','h_V','h_P','h_Tmin','cbf_T','cbf_HTO','cbf_V','cbf_P','cbf_Tmin']:
                    wh[k].append(0.0)
                wh['T_viol'].append(T_s > 363.15)
                wh['P_viol'].append(P_real > 6.0e6)
                wh['HTO_viol'].append(hto_pct > 2.0)
                wh['V_viol'].append(U_cell > 2.2)

    print("Warm-up Complete.")
    return last_action, wh


def run_test(controller, initial_state, month='02', duration=86400, sim_dt=0.2, dt_ctrl=60.0, T_ref=353.15):
    """Run full test with CBF diagnostics."""
    sim = SingleStackSimulator(sim_dt=sim_dt)
    sim.reset(initial_state=initial_state)
    full_profile = load_wind_profile(month)
    t_eval = np.arange(0, duration, sim_dt)
    profile_indices = np.clip((t_eval / 60).astype(int), 0, len(full_profile) - 1)
    ctrl_steps = int(dt_ctrl / sim_dt)
    total_ctrl_steps = len(t_eval) // ctrl_steps
    horizon = getattr(controller, 'horizon', 5)

    history = {
        't': [], 'P_ref': [], 'P_real': [],
        'T_s': [], 'T_sep': [], 'T_c_out': [],
        'I': [], 'v_lye': [], 'v_c': [],
        'U_cell': [], 'HTO': [],
        # CBF diagnostics
        'cbf_active': [], 'adjustment': [],
        'h_T': [], 'h_HTO': [], 'h_V': [], 'h_P': [], 'h_Tmin': [],
        'cbf_T': [], 'cbf_HTO': [], 'cbf_V': [], 'cbf_P': [], 'cbf_Tmin': [],
        # Violations
        'T_viol': [], 'P_viol': [], 'HTO_viol': [], 'V_viol': [],
    }

    last_action = [2000, 0.03, 0.0]
    active_mask = np.array([True, True, True, True, True])

    # Warm-up with NMPC
    warmup_ctrl = SingleStackNMPCController(dt=dt_ctrl, horizon=horizon, sim_dt=sim_dt)
    last_action, warmup_history = run_warmup(sim, warmup_ctrl, last_action, sim_dt, T_ref=T_ref, month=month)

    print(f"Running main test for {duration}s ({duration/3600:.1f}h)...")

    with tqdm(total=total_ctrl_steps, desc="Testing", unit="ctrl_step") as pbar:
        for i, t in enumerate(t_eval):
            state = sim.state.copy()
            T_s_in, T_s, T_sep, T_c_out = state[0], state[1], state[2], state[3]
            n_H2_an, n_liq, n_gas = state[4], state[5], state[6]

            _, U_cell, _ = sim._calculate_electrochemical_properties(last_action[0], state[1])
            P_real = U_cell * last_action[0] * sim.N_cell
            hto_pct = (n_gas * sim.R * T_sep) / (sim.P_sys * sim.V_sep_gas) * 100

            # Control update
            if i % ctrl_steps == 0:
                idx_min = int(t / 60)
                P_future = [full_profile[min(idx_min + k, len(full_profile) - 1)] for k in range(horizon)]
                try:
                    if hasattr(controller, 'get_action_with_rollout'):
                        I_cmd, v_lye_cmd, v_c_cmd = controller.get_action_with_rollout(
                            state, P_future, T_ref, last_action, num_candidates=64, verbose=False
                        )
                    else:
                        I_cmd, v_lye_cmd, v_c_cmd = controller.get_action(
                            state, P_future, T_ref, last_action
                        )
                except Exception as e:
                    print(f"Controller error @ t={t}: {e}")
                    I_cmd, v_lye_cmd, v_c_cmd = last_action[0], last_action[1], last_action[2]

                action = np.array([I_cmd, v_lye_cmd, v_c_cmd])

                # CBF diagnostics
                cbf_active = 0
                adjustment = 0.0
                h_vals = np.zeros(5)
                cbf_vals = np.zeros(5)
                if hasattr(controller, 'use_cbf_projection') and controller.use_cbf_projection:
                    try:
                        _, success, info = controller.projector.project(
                            action, state, u_last=np.array(last_action), active_mask=active_mask
                        )
                        h_vals = info.get('h', np.zeros(5))
                        cbf_vals = info.get('cbf_ref', info.get('cbf', np.zeros(5)))
                        cbf_active = 1 if info.get('projection_needed', False) else 0
                        adjustment = np.linalg.norm(info.get('adjustment', 0.0))
                    except Exception as e:
                        pass

                last_action = [I_cmd, v_lye_cmd, v_c_cmd]
                P_ref_val = full_profile[profile_indices[i]]
                pbar.set_postfix({
                    "t": f"{t:.0f}s", "P_ref": f"{P_ref_val/1e6:.1f}MW",
                    "P_real": f"{P_real/1e6:.1f}MW", "T_s": f"{T_s-273.15:.1f}C"
                })
                pbar.update(1)
            else:
                action = np.array(last_action)
                cbf_active = history['cbf_active'][-1] if history['cbf_active'] else 0
                adjustment = history['adjustment'][-1] if history['adjustment'] else 0.0
                h_vals = np.array([history['h_T'][-1], history['h_HTO'][-1], history['h_V'][-1], history['h_P'][-1], history['h_Tmin'][-1]]) if history['h_T'] else np.zeros(5)
                cbf_vals = np.array([history['cbf_T'][-1], history['cbf_HTO'][-1], history['cbf_V'][-1], history['cbf_P'][-1], history['cbf_Tmin'][-1]]) if history['cbf_T'] else np.zeros(5)

            sim.step(action)

            # Log every 10s
            if i % 10 == 0:
                history['t'].append(t)
                history['P_ref'].append(full_profile[profile_indices[i]])
                history['P_real'].append(P_real)
                history['T_s'].append(T_s - 273.15)
                history['T_sep'].append(T_sep - 273.15)
                history['T_c_out'].append(T_c_out - 273.15)
                history['I'].append(last_action[0])
                history['v_lye'].append(last_action[1])
                history['v_c'].append(last_action[2])
                history['U_cell'].append(U_cell)
                history['HTO'].append(hto_pct)
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
                history['T_viol'].append(T_s > 363.15)
                history['P_viol'].append(P_real > 6.0e6)
                history['HTO_viol'].append(hto_pct > 2.0)
                history['V_viol'].append(U_cell > 2.2)

    # Merge warmup history
    if warmup_history:
        for key in history:
            if key in warmup_history:
                history[key] = warmup_history[key] + history[key]

    # Summary
    max_T = max(history['T_s'])
    max_P = max(history['P_real'])
    max_HTO = max(history['HTO'])
    max_V = max(history['U_cell'])
    n_cbf = sum(history['cbf_active'])
    print(f"\nResults: Max T={max_T:.2f}°C, Max P={max_P:.3f}MW, Max HTO={max_HTO:.3f}%, CBF active={n_cbf}/{len(history['cbf_active'])}")

    return history


def plot_diagnostics(history, output_dir, filename):
    """Plot full CBF diagnostics."""
    t = np.array(history['t'])
    fig = plt.figure(figsize=(20, 18))
    gs = fig.add_gridspec(5, 3, hspace=0.4, wspace=0.3)

    # Row 1: Physical
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t, np.array(history['P_ref'])/1e6, 'k--', alpha=0.5, label='Ref')
    ax.plot(t, np.array(history['P_real'])/1e6, 'b-', linewidth=2, label='Real')
    ax.axhline(y=6.0, color='r', linestyle='--', alpha=0.7, label='P_max')
    ax.set_title('Power Tracking', fontweight='bold')
    ax.set_ylabel('MW')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t, history['T_s'], 'r-', linewidth=2)
    ax.axhline(y=90, color='r', linestyle='--', alpha=0.7, label='T_max')
    ax.axhline(y=80, color='g', linestyle='--', alpha=0.5, label='T_target')
    ax.set_title('Stack Temperature', fontweight='bold')
    ax.set_ylabel('°C')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(t, history['HTO'], 'g-', linewidth=2)
    ax.axhline(y=2.0, color='r', linestyle='--', alpha=0.7, label='HTO_max')
    ax.axhline(y=1.8, color='orange', linestyle='--', alpha=0.5, label='Effective 1.8%')
    ax.set_title('HTO', fontweight='bold')
    ax.set_ylabel('%')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Row 2: Controls
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t, history['I'], 'b-', linewidth=2)
    ax.set_title('Current', fontweight='bold')
    ax.set_ylabel('A')
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t, np.array(history['v_lye'])*1000, 'orange', linewidth=2)
    ax.set_title('Lye Flow', fontweight='bold')
    ax.set_ylabel('L/s')
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(t, np.array(history['v_c'])*1000, 'c-', linewidth=2)
    ax.set_title('Coolant Flow', fontweight='bold')
    ax.set_ylabel('L/s')
    ax.grid(True, alpha=0.3)

    # Row 3: h values
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(t, history['h_T'], 'b-', linewidth=1.5, label='h_T')
    ax.plot(t, history['h_Tmin'], 'g-', linewidth=1.5, label='h_Tmin')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_title('h: Temperature', fontweight='bold')
    ax.set_ylabel('h')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, 1])
    ax.plot(t, history['h_HTO'], 'r-', linewidth=1.5)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_title('h: HTO', fontweight='bold')
    ax.set_ylabel('h')
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, 2])
    ax.plot(t, np.array(history['h_P'])/1e6, 'c-', linewidth=1.5, label='h_P (MW)')
    ax.plot(t, history['h_V'], 'm-', linewidth=1.5, label='h_V')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_title('h: Power & Voltage', fontweight='bold')
    ax.set_ylabel('h')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Row 4: CBF conditions
    ax = fig.add_subplot(gs[3, 0])
    ax.plot(t, history['cbf_T'], 'b-', linewidth=1.5)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.fill_between(t, 0, history['cbf_T'], where=(np.array(history['cbf_T'])>0), alpha=0.2, color='lime')
    ax.set_title('CBF: Temperature', fontweight='bold')
    ax.set_ylabel('L_f h + gamma*h')
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[3, 1])
    ax.plot(t, history['cbf_HTO'], 'r-', linewidth=1.5)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.fill_between(t, 0, history['cbf_HTO'], where=(np.array(history['cbf_HTO'])>0), alpha=0.2, color='lime')
    ax.set_title('CBF: HTO', fontweight='bold')
    ax.set_ylabel('L_f h + gamma*h')
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[3, 2])
    ax.plot(t, history['cbf_P'], 'c-', linewidth=1.5, label='CBF_P')
    ax.plot(t, history['cbf_Tmin'], 'g-', linewidth=1.5, label='CBF_Tmin')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.set_title('CBF: Power & Tmin', fontweight='bold')
    ax.set_ylabel('L_f h + gamma*h')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Row 5: Activation & summary
    ax = fig.add_subplot(gs[4, 0])
    ax.fill_between(t, 0, history['cbf_active'], alpha=0.6, color='orange', step='post')
    ax.set_ylim(-0.1, 1.5)
    ax.set_title('CBF Activation', fontweight='bold')
    ax.set_ylabel('Active')
    ax.set_xlabel('Time (s)')
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[4, 1])
    ax.plot(t, history['adjustment'], 'purple', linewidth=1.5)
    ax.set_title('Control Adjustment', fontweight='bold')
    ax.set_ylabel('||u - u_ref||')
    ax.set_xlabel('Time (s)')
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[4, 2])
    min_cbf = np.min([history['cbf_T'], history['cbf_HTO'], history['cbf_V'], history['cbf_P'], history['cbf_Tmin']], axis=0)
    ax.plot(t, min_cbf, 'k-', linewidth=2, label='Min CBF')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    ax.fill_between(t, 0, min_cbf, where=(min_cbf>0), alpha=0.3, color='lime')
    ax.set_title('Minimum CBF (All Constraints)', fontweight='bold')
    ax.set_ylabel('Min CBF')
    ax.set_xlabel('Time (s)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mark warmup/test boundary on all subplots
    for ax in fig.axes:
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.6, linewidth=1.2)
        # Only add label on the first axis to avoid duplicate legend entries
        if ax == fig.axes[0]:
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.6, linewidth=1.2, label='Warmup end')
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())

    fig.suptitle('TCN Diffusion + CBF with Wind Data: Full Diagnostics', fontsize=15, fontweight='bold')
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {out_path}")


def save_csv(history, output_dir, filename):
    keys = ['t', 'P_ref', 'P_real', 'T_s', 'T_sep', 'T_c_out',
            'I', 'v_lye', 'v_c', 'U_cell', 'HTO',
            'cbf_active', 'adjustment',
            'h_T', 'h_HTO', 'h_V', 'h_P', 'h_Tmin',
            'cbf_T', 'cbf_HTO', 'cbf_V', 'cbf_P', 'cbf_Tmin',
            'T_viol', 'P_viol', 'HTO_viol', 'V_viol']
    data_dict = {k: history[k] for k in keys if k in history}
    df = pd.DataFrame(data_dict)
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"Data saved: {filepath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--month', type=str, default='02', help='Wind month 01-12')
    parser.add_argument('--duration', type=int, default=86400, help='Test duration in seconds')
    parser.add_argument('--no_cbf', action='store_true', help='Disable CBF projection (pure TCN Diffusion)')
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    output_dir = os.path.join(project_root, 'output', 'single_stack', 'cbf_wind_test')
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(project_root, 'output', 'single_stack', 'policy', 'diffusion_tcn_policy_best.pth')
    stats_path = os.path.join(project_root, 'output', 'single_stack', 'diffusion_stats.npz')

    print(f"Model: {model_path}")
    print(f"Stats: {stats_path}")

    use_cbf = not args.no_cbf
    ctrl = SingleStackCBFModelController(
        dt=60.0, horizon=5, model_type='diffusion_tcn',
        model_path=model_path, stats_path=stats_path,
        use_cbf_projection=use_cbf,
        gamma_vec=[3.0, 2.0, 100.0, 100.0, 5.0],
        rho_vec=[50000]*5,
        h_margin_vec=[1.0, 0.002, 0.0, 0.0, 0.0],
        lambda_u_scale=[500.0, 200.0, 1000.0],
        soft_mask=[False, True, True, False, True],
        normalize=True,
    )

    initial_state = np.array([345.0, 353.15, 353.15, 325.0, 0.52, 0.52, 0.52])

    mode_str = "CBF" if use_cbf else "NO-CBF (Pure Model)"
    print("\n" + "=" * 70)
    print(f"TCN Diffusion + {mode_str} | Wind Month: {args.month} | Duration: {args.duration}s")
    print("=" * 70)

    history = run_test(ctrl, initial_state, month=args.month, duration=args.duration)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    suffix = 'cbf' if use_cbf else 'nocbf'
    plot_diagnostics(history, output_dir, f'{suffix}_wind_{args.month}_{timestamp}.png')
    save_csv(history, output_dir, f'{suffix}_wind_{args.month}_{timestamp}.csv')

    # Metrics
    rmse_p = np.sqrt(np.mean((np.array(history['P_real']) - np.array(history['P_ref'])/1e6)**2))
    rmse_t = np.sqrt(np.mean((np.array(history['T_s']) - 80.0)**2))
    metrics = {'rmse_power_mw': float(rmse_p), 'rmse_temp_c': float(rmse_t),
               'max_temp_c': float(max(history['T_s'])), 'max_hto_pct': float(max(history['HTO'])),
               'cbf_activations': int(sum(history['cbf_active']))}
    json_path = os.path.join(output_dir, f'{suffix}_wind_{args.month}_{timestamp}_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics: {metrics}")
    print(f"Output: {output_dir}")


if __name__ == '__main__':
    main()
