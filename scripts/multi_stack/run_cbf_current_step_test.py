import os
import sys
from typing import Any
import numpy as np
import pandas as pd
from datetime import datetime
import argparse
from tqdm import tqdm

import matplotlib.pyplot as plt
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from plant.multi_stack_simulator import MultiStackSimulator
from controller.multi_stack.nmpc_simplified_controller import MultiStackNMPCSimplifiedController
from controller.multi_stack.cbf_projection_ho import MultiStackCBFProjectionHO
from controller.multi_stack.cbf_projection import MultiStackCBFProjection


def add_measurement_noise(state):
    return np.copy(state)


def run_warmup_phase(sim, ctrl, history, last_action, dt, output_dir, filename_prefix="warmup", T_ref=353.15, warmup_ctrl=None, warmup_duration=14400):
    warmup_steps = int(warmup_duration / dt)
    warmup_P_ref = 10.0e6

    active_ctrl = warmup_ctrl if warmup_ctrl is not None else ctrl
    print(f"Starting Warm-up Phase ({warmup_duration}s at {warmup_P_ref/1e6}MW) using {active_ctrl.__class__.__name__}...")

    P_future = [warmup_P_ref] * active_ctrl.horizon
    ctrl_steps = int(active_ctrl.dt / dt)
    total_ctrl_steps = warmup_steps // ctrl_steps
    warmup_profile = np.array([warmup_P_ref], dtype=float)
    warmup_profile_indices = np.zeros(warmup_steps, dtype=int)
    plot_every_steps = max(1, int(3600 / dt))
    warmup_data_filename = f"{filename_prefix}.csv"

    with tqdm(total=total_ctrl_steps, desc="Warmup", unit="ctrl_step") as pbar:
        for i in range(warmup_steps):
            t_warmup = -warmup_duration + i * dt
            measured_state = add_measurement_noise(sim.state)
            T_s_in = measured_state[0]
            T_s_vec = measured_state[1:5]
            T_sep = measured_state[5]
            T_c_out = measured_state[6]
            n_H2_an_vec = measured_state[7:11]
            n_liq = measured_state[11]
            n_gas = measured_state[12]
            Q, U_cell, eta = sim._calculate_electrochemical_properties(last_action[0], sim.state[1:5])
            P_real = np.sum(U_cell * last_action[0] * sim.N_cell)
            n_H2_sep_gas = measured_state[12]
            hto_pct = (n_H2_sep_gas * sim.R * T_sep) / (sim.P_sys * sim.V_sep_gas) * 100
            h2_rate = np.sum(sim.N_cell * last_action[0] * eta / (2 * sim.F))
            prev_action = [np.copy(last_action[0]), np.copy(last_action[1]), last_action[2]]

            if i % ctrl_steps == 0:
                I_cmd, v_lye_cmd, v_c_cmd = active_ctrl.get_action(
                    measured_state, P_future, T_ref=T_ref, last_action=last_action
                )
                action_sim = np.concatenate([I_cmd, v_lye_cmd, [v_c_cmd]])
                last_action = [I_cmd, v_lye_cmd, v_c_cmd]
            else:
                action_sim = np.concatenate([last_action[0], last_action[1], [last_action[2]]])

            sim.step(action_sim)
            log_and_visualize(
                i=i, t=t_warmup, history=history, full_profile=warmup_profile,
                profile_indices=warmup_profile_indices, P_real=P_real,
                T_s_in=T_s_in, T_s_vec=T_s_vec, T_sep=T_sep, T_c_out=T_c_out,
                n_H2_an_vec=n_H2_an_vec, n_liq=n_liq, n_gas=n_gas, T_ref=T_ref,
                P_future=P_future, prev_action=prev_action, last_action=last_action,
                U_cell=U_cell, hto_pct=hto_pct, h2_rate=h2_rate,
                ctrl_steps=ctrl_steps, pbar=pbar, data_filename=warmup_data_filename,
                output_dir=output_dir, plot_every_steps=plot_every_steps,
                cbf_info=None,
            )

    print("Warm-up Complete. Starting Current Step Test...")
    return last_action


def log_and_visualize(i, t, history, full_profile, profile_indices, P_real, T_s_in, T_s_vec, T_sep, T_c_out,
                      n_H2_an_vec, n_liq, n_gas, T_ref, P_future, prev_action, last_action, U_cell, hto_pct, h2_rate,
                      ctrl_steps, pbar, data_filename, output_dir, plot_every_steps, cbf_info=None):
    if i % 50 == 0:
        history['t'].append(t)
        history['P_ref'].append(full_profile[profile_indices[i]] if i < len(profile_indices) else full_profile[-1])
        history['P_real'].append(P_real)
        history['T_s_in'].append(T_s_in)
        history['T_s_all'].append(T_s_vec)
        history['T_sep'].append(T_sep)
        history['T_c_out'].append(T_c_out)
        history['n_H2_an_vec'].append(n_H2_an_vec)
        history['n_liq'].append(n_liq)
        history['n_gas'].append(n_gas)
        history['T_ref'].append(T_ref)
        history['P_ref_future'].append(P_future)
        history['I_prev'].append(prev_action[0])
        history['v_lye_prev'].append(prev_action[1])
        history['v_c_prev'].append(prev_action[2])
        history['I_all'].append(last_action[0])
        history['v_lye_all'].append(last_action[1])
        history['v_c'].append(last_action[2])
        history['U_cell_all'].append(U_cell)
        history['HTO'].append(hto_pct)
        history['H2_rate'].append(h2_rate)

        # CBF info
        history['cbf_triggered'].append(cbf_info.get('triggered', False) if cbf_info is not None else False)
        history['cbf_adjustment'].append(cbf_info.get('adjustment', 0.0) if cbf_info is not None else 0.0)
        history['cbf_max_slack'].append(cbf_info.get('max_slack', 0.0) if cbf_info is not None else 0.0)
        history['cbf_success'].append(cbf_info.get('success', True) if cbf_info is not None else True)

        # CBF trigger classification: Tmax (idx 0-3), HTO (idx 4), Tmin (idx 5-8)
        if cbf_info is not None and cbf_info.get('triggered', False):
            h_vals = cbf_info.get('h', np.zeros(9))
            if isinstance(h_vals, (list, np.ndarray)) and len(h_vals) == 9:
                # Find which constraint group has the smallest (most violated) h
                h_tmax = np.min(h_vals[0:4])
                h_hto = h_vals[4]
                h_tmin = np.min(h_vals[5:9])
                min_group = np.argmin([h_tmax, h_hto, h_tmin])
                history['cbf_trigger_Tmax'].append(min_group == 0)
                history['cbf_trigger_HTO'].append(min_group == 1)
                history['cbf_trigger_Tmin'].append(min_group == 2)
            else:
                history['cbf_trigger_Tmax'].append(False)
                history['cbf_trigger_HTO'].append(False)
                history['cbf_trigger_Tmin'].append(False)
        else:
            history['cbf_trigger_Tmax'].append(False)
            history['cbf_trigger_HTO'].append(False)
            history['cbf_trigger_Tmin'].append(False)

        # CBF slack per group
        if cbf_info is not None:
            s_vals = cbf_info.get('slack', np.zeros(9))
            if isinstance(s_vals, (list, np.ndarray)) and len(s_vals) == 9:
                history['cbf_slack_Tmax'].append(float(np.max(s_vals[0:4])))
                history['cbf_slack_HTO'].append(float(s_vals[4]))
                history['cbf_slack_Tmin'].append(float(np.max(s_vals[5:9])))
            else:
                history['cbf_slack_Tmax'].append(0.0)
                history['cbf_slack_HTO'].append(0.0)
                history['cbf_slack_Tmin'].append(0.0)
        else:
            history['cbf_slack_Tmax'].append(0.0)
            history['cbf_slack_HTO'].append(0.0)
            history['cbf_slack_Tmin'].append(0.0)

        for idx in range(9):
            h_key = f'cbf_h_{idx}'
            val_key = f'cbf_val_{idx}'
            history[h_key].append(cbf_info.get(h_key, 0.0) if cbf_info is not None else 0.0)
            history[val_key].append(cbf_info.get(val_key, 0.0) if cbf_info is not None else 0.0)

    if i % ctrl_steps == 0:
        idx = profile_indices[i] if i < len(profile_indices) else len(full_profile) - 1
        P_ref_val = full_profile[idx]
        pbar.set_postfix({
            "t": f"{t:.0f}s",
            "P_ref": f"{P_ref_val/1e6:.1f}MW",
            "P_real": f"{P_real/1e6:.1f}MW",
            "T_s": f"{np.mean(T_s_vec)-273.15:.1f}C",
            "I_total": f"{np.sum(last_action[0]):.0f}A"
        })
        pbar.update(1)

    if plot_every_steps is not None and i > 0 and i % plot_every_steps == 0:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_data_csv(history, output_dir, data_filename)
        save_plot(history, output_dir, data_filename)


def run_current_step_test(model_type='diffusion_tcn', duration=7200, step_time=3600,
                          I_initial=3000.0, I_final=1000.0,
                          second_step_time=None, second_step_I=None,
                          warmup_controller='nmpc_simplified', warmup_duration=14400, output_subdir=None,
                          use_cbf_projection=True,
                          v_c_fixed=None, v_lye_fixed=None):
    sim_dt = 0.2
    dt_ctrl = 60.0
    horizon = 5
    T_ref = 353.15

    sim = MultiStackSimulator(dt=sim_dt)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

    if output_subdir:
        output_dir = os.path.join(project_root, 'output', 'multi_stack', 'test_step', output_subdir)
    else:
        output_dir = os.path.join(project_root, 'output', 'multi_stack', 'test_step')
    os.makedirs(output_dir, exist_ok=True)

    # Directly use CBF projection without any model controller
    # Using same parameters as wind test (run_multi_stack_test.py cbf_ho_model)
    # Adjusted params: reduce HTO alpha2 to soften HTO response,
    # increase I weight to reduce current突变 when HTO triggers.
    projector = MultiStackCBFProjectionHO(
        dt=dt_ctrl,
        gamma_vec=[10.0]*4 + [1.0] + [10.0]*4,
        rho_vec=[5000]*9,
        h_margin_vec=[0.0]*4 + [0.001] + [0.0]*4,
        normalize=True,
        lambda_u_scale=0.0,
        soft_mask=[True]*9,
        alpha1_vec=[10.0, 10.0, 10.0, 10.0, 1.0, 10.0, 10.0, 10.0, 10.0],
        alpha2_vec=[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        u_weight_scale=[80.0, 80.0, 80.0, 80.0, 1.0, 1.0, 1.0, 1.0, 0.1],
    )
    print(f"Using MultiStackCBFProjectionHO directly (no model controller)")
    print(f"Parameters aligned with wind test (cbf_ho_model)")
    print(f"CBF projection: {'ON' if use_cbf_projection else 'OFF'}")
    if v_c_fixed is not None:
        print(f"  Fixed v_c = {v_c_fixed:.5f} m3/s")
    if v_lye_fixed is not None:
        v_lye_arr = np.full(4, v_lye_fixed) if np.isscalar(v_lye_fixed) else np.array(v_lye_fixed)
        print(f"  Fixed v_lye = {v_lye_arr} m3/s")

    sim.reset()
    t_eval = np.arange(0, duration, sim_dt)
    print(f"Running current step test for {duration}s ({duration/3600:.1f}h)")
    print(f"Step at t={step_time}s: I_total={I_initial:.0f}A -> {I_final:.0f}A")

    # Build a dummy power profile (not used for control, but for logging)
    full_profile = np.full(len(t_eval), 10.0e6)
    profile_indices = np.arange(len(t_eval))

    history = {
        't': [], 'P_ref': [], 'P_real': [],
        'T_s_in': [], 'T_s_all': [], 'T_sep': [], 'T_c_out': [],
        'n_H2_an_vec': [], 'n_liq': [], 'n_gas': [],
        'T_ref': [], 'P_ref_future': [],
        'I_prev': [], 'v_lye_prev': [], 'v_c_prev': [],
        'I_all': [], 'v_lye_all': [], 'v_c': [],
        'U_cell_all': [], 'HTO': [], 'H2_rate': [],
        'cbf_triggered': [], 'cbf_adjustment': [], 'cbf_max_slack': [], 'cbf_success': [],
        'cbf_trigger_Tmax': [], 'cbf_trigger_HTO': [], 'cbf_trigger_Tmin': [],
        'cbf_slack_Tmax': [], 'cbf_slack_HTO': [], 'cbf_slack_Tmin': [],
    }
    for idx in range(9):
        history[f'cbf_h_{idx}'] = []
        history[f'cbf_val_{idx}'] = []

    # Store reference values for plotting
    history['I_initial'] = [I_initial * 4] * 1000  # Will be trimmed to match t length
    history['I_final'] = [I_final * 4] * 1000
    history['step_time'] = [step_time] * 1000
    history['v_c_ref'] = [v_c_fixed] * 1000
    history['v_lye_ref'] = [v_lye_fixed] * 1000
    history['second_step_time'] = [second_step_time] * 1000 if second_step_time is not None else [None] * 1000
    history['second_step_I'] = [second_step_I * 4] * 1000 if second_step_I is not None else [None] * 1000

    last_action = [
        np.array([1800.0, 2100.0, 2200.0, 1900.0]),
        np.array([0.025, 0.032, 0.035, 0.028]),
        0.0
    ]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    cbf_tag = "cbf_on" if use_cbf_projection else "cbf_off"
    data_filename = f"current_step_{model_type}_{cbf_tag}_data_{timestamp}.csv"
    plot_every_steps = max(1, int(3600 / sim_dt))

    # Warm-up
    if warmup_controller == 'none':
        print("Skipping warm-up phase (--warmup_controller=none)")
    else:
        print(f"Creating {warmup_controller} controller for warm-up phase...")
        if warmup_controller == 'nmpc_simplified':
            warmup_ctrl = MultiStackNMPCSimplifiedController(dt=dt_ctrl, horizon=horizon, dt_sub=sim_dt)
        elif warmup_controller == 'nmpc':
            from controller.multi_stack.nmpc_controller import MultiStackNMPCController
            warmup_ctrl = MultiStackNMPCController(dt=dt_ctrl, horizon=horizon, dt_sub=sim_dt)
        else:
            raise ValueError(f"Unknown warmup controller type: {warmup_controller}")

        last_action = run_warmup_phase(
            sim, None, history, last_action, sim_dt,
            output_dir=output_dir,
            filename_prefix=f"current_step_warmup_{timestamp}",
            T_ref=T_ref,
            warmup_ctrl=warmup_ctrl,
            warmup_duration=warmup_duration
        )

    print(f"Starting Current Step Test (CBF {'ON' if use_cbf_projection else 'OFF'})...")
    ctrl_steps = int(dt_ctrl / sim_dt)
    total_ctrl_steps = len(t_eval) // ctrl_steps

    # Track I target over time
    I_target = I_initial
    has_second_step = second_step_time is not None and second_step_I is not None
    if has_second_step:
        print(f"Second step at t={second_step_time}s: I_total={I_final:.0f}A -> {second_step_I:.0f}A")

    with tqdm(total=total_ctrl_steps, desc="CurrentStep", unit="ctrl_step") as pbar:
        for i, t in enumerate(t_eval):
            if t >= step_time:
                I_target = I_final
            if has_second_step and t >= second_step_time:
                I_target = second_step_I

            measured_state = add_measurement_noise(sim.state)
            T_s_in = measured_state[0]
            T_s_vec = measured_state[1:5]
            T_sep = measured_state[5]
            T_c_out = measured_state[6]
            n_H2_an_vec = measured_state[7:11]
            n_liq = measured_state[11]
            n_gas = measured_state[12]
            Q, U_cell, eta = sim._calculate_electrochemical_properties(last_action[0], sim.state[1:5])
            P_real = np.sum(U_cell * last_action[0] * sim.N_cell)
            n_H2_sep_gas = measured_state[12]
            hto_pct = (n_H2_sep_gas * sim.R * T_sep) / (sim.P_sys * sim.V_sep_gas) * 100
            h2_rate = np.sum(sim.N_cell * last_action[0] * eta / (2 * sim.F))
            prev_action = [np.copy(last_action[0]), np.copy(last_action[1]), last_action[2]]

            if i % ctrl_steps == 0:
                # Dummy P_ref for controller (not actually used since we override current)
                idx_min = i
                end_idx = idx_min + horizon
                P_future = full_profile[idx_min:end_idx]
                if len(P_future) < horizon:
                    padding = np.full(horizon - len(P_future), full_profile[-1])
                    P_future = np.concatenate([P_future, padding])
                P_future = list[Any](P_future)

                # 1. Directly set current target (per stack)
                I_cmd = np.full(4, I_target)

                # 2. Use fixed v_lye and v_c (no controller involved)
                if v_lye_fixed is not None:
                    v_lye_cmd = np.full(4, v_lye_fixed) if np.isscalar(v_lye_fixed) else np.array(v_lye_fixed)
                else:
                    v_lye_cmd = np.array([0.03, 0.03, 0.03, 0.03])
                if v_c_fixed is not None:
                    v_c_cmd = float(v_c_fixed)
                else:
                    v_c_cmd = 0.03

                # 3. Apply CBF projection to the action
                raw_action = np.concatenate([I_cmd, v_lye_cmd, [v_c_cmd]])
                last_action_vec = np.concatenate([last_action[0], last_action[1], [last_action[2]]])

                if use_cbf_projection:
                    safe_action, success, cbf_info = projector.project(
                        raw_action, measured_state, u_last=last_action_vec, verbose=False
                    )

                    # Check if CBF actually modified the action
                    adjustment = float(np.linalg.norm(safe_action - raw_action))
                    cbf_triggered = adjustment > 1e-6

                    cbf_info_out = {
                        'triggered': cbf_triggered,
                        'success': success,
                        'adjustment': adjustment,
                        'max_slack': cbf_info.get('max_slack', 0.0),
                        'h': cbf_info.get('h', np.zeros(9)),
                        'lie1': cbf_info.get('lie1', np.zeros(9)),
                        'lie2': cbf_info.get('lie2', np.zeros(9)),
                        'cbf': cbf_info.get('cbf', np.zeros(9)),
                        'slack': cbf_info.get('slack', np.zeros(9)),
                    }

                    I_cmd = safe_action[0:4]
                    v_lye_cmd = safe_action[4:8]
                    v_c_cmd = safe_action[8]
                else:
                    cbf_info_out = {'triggered': False, 'success': True, 'adjustment': 0.0, 'max_slack': 0.0}

                action_sim = np.concatenate([I_cmd, v_lye_cmd, [v_c_cmd]])
                last_action = [I_cmd, v_lye_cmd, v_c_cmd]
            else:
                action_sim = np.concatenate([last_action[0], last_action[1], [last_action[2]]])
                cbf_info_out = None

            sim.step(action_sim)
            log_and_visualize(i, t, history, full_profile, profile_indices, P_real, T_s_in, T_s_vec, T_sep, T_c_out,
                              n_H2_an_vec, n_liq, n_gas, T_ref, P_future, prev_action, last_action, U_cell, hto_pct, h2_rate,
                              ctrl_steps, pbar, data_filename, output_dir, plot_every_steps, cbf_info_out)

    save_data_csv(history, output_dir, data_filename)
    save_plot(history, output_dir, data_filename)
    calculate_current_step_metrics(history, step_time, I_initial, I_final, output_dir, data_filename,
                                   second_step_time=second_step_time, second_step_I=second_step_I)


def calculate_current_step_metrics(history, step_time, I_initial, I_final, output_dir, filename,
                                   second_step_time=None, second_step_I=None):
    t_arr = np.array(history['t'])
    P_real_arr = np.array(history['P_real'])
    T_s_all = np.array(history['T_s_all'])
    if T_s_all.ndim > 1:
        T_s_mean = np.mean(T_s_all, axis=1)
    else:
        T_s_mean = np.zeros_like(t_arr)
    T_ref_arr = np.array(history['T_ref'])
    I_all = np.array(history['I_all'])
    if I_all.ndim > 1:
        I_total = np.sum(I_all, axis=1)
    else:
        I_total = I_all

    cbf_triggered_arr = np.array(history['cbf_triggered'])

    # Phase 1: after first step
    post_step_mask = t_arr >= step_time
    t_post = t_arr[post_step_mask]
    P_real_post = P_real_arr[post_step_mask]
    T_s_post = T_s_mean[post_step_mask]
    cbf_post = cbf_triggered_arr[post_step_mask]

    rmse_p = float(np.sqrt(np.mean((P_real_post - np.mean(P_real_post))**2)) / 1e6) if len(P_real_post) > 0 else 0.0
    rmse_t = float(np.sqrt(np.mean((T_s_post - T_ref_arr[post_step_mask])**2))) if len(T_s_post) > 0 else 0.0
    cbf_trigger_rate = float(np.mean(cbf_post)) if len(cbf_post) > 0 else 0.0

    metrics = {
        "step_time_s": step_time,
        "I_initial_a": I_initial,
        "I_final_a": I_final,
        "rmse_power_mw": rmse_p,
        "rmse_temp_k": rmse_t,
        "cbf_trigger_rate": cbf_trigger_rate,
        "cbf_trigger_count": int(np.sum(cbf_post)),
        "cbf_trigger_total": int(len(cbf_post)),
    }

    # Phase 2: after second step (if any)
    if second_step_time is not None:
        post_step2_mask = t_arr >= second_step_time
        t_post2 = t_arr[post_step2_mask]
        P_real_post2 = P_real_arr[post_step2_mask]
        T_s_post2 = T_s_mean[post_step2_mask]
        cbf_post2 = cbf_triggered_arr[post_step2_mask]

        rmse_p2 = float(np.sqrt(np.mean((P_real_post2 - np.mean(P_real_post2))**2)) / 1e6) if len(P_real_post2) > 0 else 0.0
        rmse_t2 = float(np.sqrt(np.mean((T_s_post2 - T_ref_arr[post_step2_mask])**2))) if len(T_s_post2) > 0 else 0.0
        cbf_trigger_rate2 = float(np.mean(cbf_post2)) if len(cbf_post2) > 0 else 0.0

        metrics["second_step_time_s"] = second_step_time
        metrics["second_step_I_a"] = second_step_I
        metrics["rmse_power_mw_post_step2"] = rmse_p2
        metrics["rmse_temp_k_post_step2"] = rmse_t2
        metrics["cbf_trigger_rate_post_step2"] = cbf_trigger_rate2
        metrics["cbf_trigger_count_post_step2"] = int(np.sum(cbf_post2))
        metrics["cbf_trigger_total_post_step2"] = int(len(cbf_post2))

    json_filename = filename.replace('.csv', '_metrics.json')
    json_path = os.path.join(output_dir, json_filename)
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print("-" * 50)
    print(f"Current Step Response Metrics (Step at t={step_time}s)")
    print(f"I: {I_initial:.0f}A -> {I_final:.0f}A")
    print(f"Power RMSE (post-step): {rmse_p:.3f} MW")
    print(f"Temp RMSE (post-step):  {rmse_t:.3f} K")
    print(f"CBF Trigger Rate:       {cbf_trigger_rate*100:.1f}% ({metrics['cbf_trigger_count']}/{metrics['cbf_trigger_total']})")
    print(f"Metrics saved to {json_path}")
    print("-" * 50)


def save_plot(history, output_dir, filename):
    t_arr = np.array(history['t'])
    if len(t_arr) == 0:
        return

    T_s_all = np.array(history['T_s_all'])
    U_cell_all = np.array(history['U_cell_all'])
    I_all = np.array(history['I_all'])
    v_lye_all = np.array(history['v_lye_all'])
    cbf_triggered = np.array(history['cbf_triggered'])
    cbf_adjustment = np.array(history['cbf_adjustment'])

    if I_all.ndim > 1:
        I_total = np.sum(I_all, axis=1)
    else:
        I_total = I_all

    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 0.6])

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t_arr, np.array(history['P_ref']) / 1e6, 'k--', label='Ref')
    ax.plot(t_arr, np.array(history['P_real']) / 1e6, 'b-', label='Real')
    ax.set_title('Power Tracking (Total)')
    ax.set_ylabel('MW')
    ax.legend()
    ax.grid(True)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t_arr, history['T_sep'], 'g--', label='Separator')
    ax.plot(t_arr, history['T_c_out'], 'c:', label='CW Out')
    for i in range(4):
        ax.plot(t_arr, T_s_all[:, i], label=f'Stack {i+1}')
    ax.plot(t_arr, history['T_ref'], 'k--', linewidth=1.5, label='Ref Temp')
    ax.axhline(y=363.15, color='r', linestyle='--', alpha=0.5, label='T_max (90°C)')
    ax.axhline(y=293.15, color='b', linestyle='--', alpha=0.5, label='T_min (20°C)')
    ax.set_title('All Temperatures')
    ax.set_ylabel('K')
    ax.legend()
    ax.grid(True)

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(t_arr, history['HTO'], 'm-', label='HTO')
    ax.axhline(y=2.0, color='r', linestyle='--', alpha=0.5, label='HTO_max (2%)')
    ax.set_title('HTO (H2 in O2)')
    ax.set_ylabel('%')
    ax.legend()
    ax.grid(True)

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t_arr, I_total, 'b-', label='I_total')
    for i in range(4):
        ax.plot(t_arr, I_all[:, i], '--', alpha=0.5, label=f'Stack {i+1}')
    # Add stepped current reference line (supports bidirectional step)
    I_init_ref = history.get('I_initial', [None])[0]
    I_fin_ref = history.get('I_final', [None])[0]
    step_time_ref = history.get('step_time', [None])[0]
    second_step_time_ref = history.get('second_step_time', [None])[0]
    second_step_I_ref = history.get('second_step_I', [None])[0]
    if I_init_ref is not None and I_fin_ref is not None and len(t_arr) > 0:
        I_ref_curve = np.full_like(t_arr, float(I_init_ref))
        if step_time_ref is not None:
            I_ref_curve[t_arr >= step_time_ref] = float(I_fin_ref)
        if second_step_time_ref is not None and second_step_I_ref is not None:
            I_ref_curve[t_arr >= second_step_time_ref] = float(second_step_I_ref)
        step_label = f'I_ref ({I_init_ref/4:.0f}→{I_fin_ref/4:.0f}'
        if second_step_time_ref is not None and second_step_I_ref is not None:
            step_label += f'→{second_step_I_ref/4:.0f}'
        step_label += 'A/stk)'
        ax.plot(t_arr, I_ref_curve, 'r--', alpha=0.6, linewidth=2, label=step_label)
    ax.set_title('Stack Currents')
    ax.set_ylabel('Amps')
    ax.legend()
    ax.grid(True)

    ax = fig.add_subplot(gs[1, 1])
    for i in range(4):
        ax.plot(t_arr, U_cell_all[:, i], label=f'Stack {i+1}')
    ax.set_title('Stack Voltages (Per Cell)')
    ax.set_ylabel('Volts')
    ax.legend()
    ax.grid(True)

    ax = fig.add_subplot(gs[1, 2])
    for i in range(4):
        ax.plot(t_arr, v_lye_all[:, i], label=f'Stack {i+1}')
    v_lye_ref = history.get('v_lye_ref', [None])[0]
    if v_lye_ref is not None:
        ax.axhline(y=v_lye_ref, color='r', linestyle='--', alpha=0.5, label=f'v_lye_ref={v_lye_ref}')
    ax.set_title('Stack Lye Flow')
    ax.set_ylabel('m3/s')
    ax.legend()
    ax.grid(True)

    ax = fig.add_subplot(gs[2, 0])
    ax.plot(t_arr, history['v_c'], 'cyan', label='CW')
    v_c_ref = history.get('v_c_ref', [None])[0]
    if v_c_ref is not None:
        ax.axhline(y=v_c_ref, color='r', linestyle='--', alpha=0.5, label=f'v_c_ref={v_c_ref}')
    ax.set_title('Coolant Flow')
    ax.set_ylabel('m3/s')
    ax.legend()
    ax.grid(True)

    ax = fig.add_subplot(gs[2, 1])
    ax.plot(t_arr, history['H2_rate'], 'g-', label='H2 Rate')
    ax.set_title('Total H2 Production Rate')
    ax.set_ylabel('mol/s')
    ax.grid(True)

    # Three CBF slack subplots on the right column
    cbf_slack_Tmax = np.array(history['cbf_slack_Tmax'])
    cbf_slack_HTO = np.array(history['cbf_slack_HTO'])
    cbf_slack_Tmin = np.array(history['cbf_slack_Tmin'])
    cbf_trigger_Tmax = np.array(history['cbf_trigger_Tmax'])
    cbf_trigger_HTO = np.array(history['cbf_trigger_HTO'])
    cbf_trigger_Tmin = np.array(history['cbf_trigger_Tmin'])

    tmax_times = t_arr[cbf_trigger_Tmax]
    tmax_slack = cbf_slack_Tmax[cbf_trigger_Tmax]
    hto_times = t_arr[cbf_trigger_HTO]
    hto_slack = cbf_slack_HTO[cbf_trigger_HTO]
    tmin_times = t_arr[cbf_trigger_Tmin]
    tmin_slack = cbf_slack_Tmin[cbf_trigger_Tmin]

    # Tmax slack subplot
    ax = fig.add_subplot(gs[0, 3])
    ax.plot(t_arr, cbf_slack_Tmax, 'r-', alpha=0.7, label='Slack')
    if len(tmax_times) > 0:
        ax.scatter(tmax_times, tmax_slack, c='red', s=20, marker='o', zorder=5, label='Trigger')
    ax.set_title('CBF Slack: Tmax')
    ax.set_ylabel('Slack Value')
    ax.legend()
    ax.grid(True)

    # HTO slack subplot
    ax = fig.add_subplot(gs[1, 3])
    ax.plot(t_arr, cbf_slack_HTO, 'purple', alpha=0.7, label='Slack')
    if len(hto_times) > 0:
        ax.scatter(hto_times, hto_slack, c='purple', s=20, marker='s', zorder=5, label='Trigger')
    ax.set_title('CBF Slack: HTO')
    ax.set_ylabel('Slack Value')
    ax.legend()
    ax.grid(True)

    # Tmin slack subplot
    ax = fig.add_subplot(gs[2, 3])
    ax.plot(t_arr, cbf_slack_Tmin, 'b-', alpha=0.7, label='Slack')
    if len(tmin_times) > 0:
        ax.scatter(tmin_times, tmin_slack, c='blue', s=20, marker='^', zorder=5, label='Trigger')
    ax.set_title('CBF Slack: Tmin')
    ax.set_ylabel('Slack Value')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, filename.replace('.csv', '.png'))
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.close(fig)


def save_data_csv(history, output_dir, filename):
    keys_scalar = ['t', 'P_ref', 'P_real', 'T_s_in', 'T_sep', 'T_c_out', 'n_liq', 'n_gas', 'T_ref',
                   'v_c_prev', 'v_c', 'HTO', 'H2_rate',
                   'cbf_triggered', 'cbf_adjustment', 'cbf_max_slack', 'cbf_success',
                   'cbf_trigger_Tmax', 'cbf_trigger_HTO', 'cbf_trigger_Tmin',
                   'cbf_slack_Tmax', 'cbf_slack_HTO', 'cbf_slack_Tmin']
    data_dict = {}
    for key in keys_scalar:
        if key in history:
            data_dict[key] = history[key]
    for key in ['T_s_all', 'n_H2_an_vec', 'I_all', 'v_lye_all', 'U_cell_all']:
        if key in history:
            vec_data = np.array(history[key])
            if vec_data.size > 0:
                for k in range(4):
                    data_dict[f"{key}_{k+1}"] = vec_data[:, k]
    # CBF values
    for idx in range(9):
        h_key = f'cbf_h_{idx}'
        val_key = f'cbf_val_{idx}'
        if h_key in history:
            data_dict[h_key] = history[h_key]
        if val_key in history:
            data_dict[val_key] = history[val_key]

    df = pd.DataFrame(data_dict)
    file_path = os.path.join(output_dir, filename)
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Multi-Stack CBF Current Step Test')
    parser.add_argument('--model_type', type=str, default='diffusion_tcn',
                        choices=['diffusion_mlp', 'diffusion_pure_mlp', 'diffusion_tcn', 'diffusion_tcn_l3',
                                 'flow_mlp', 'flow_tcn', 'mlp', 'tcn', 'flow_matching', 'pure_mlp', 'pure_tcn', 'lstm'],
                        help='Model type')
    parser.add_argument('--duration', type=int, default=7200, help='Duration of test in seconds (default 2h)')
    parser.add_argument('--step_time', type=int, default=3600, help='Time of step in seconds (default 1h)')
    parser.add_argument('--I_initial', type=float, default=3000.0, help='Initial total current in A (default 3000)')
    parser.add_argument('--I_final', type=float, default=1000.0, help='Final total current in A (default 1000)')
    parser.add_argument('--warmup_controller', type=str, default='nmpc_simplified',
                        choices=['none', 'nmpc', 'nmpc_simplified'],
                        help='Controller type for warm-up phase')
    parser.add_argument('--warmup_duration', type=int, default=14400,
                        help='Warm-up duration in seconds (default 14400=4h)')
    parser.add_argument('--output_subdir', type=str, default=None,
                        help='Subdirectory under output/multi_stack/test_step/ to store results')
    parser.add_argument('--no_cbf', action='store_true',
                        help='Disable CBF projection (for comparison)')
    parser.add_argument('--v_c', type=float, default=None,
                        help='Fixed coolant flow rate v_c in m3/s (default: use controller output)')
    parser.add_argument('--v_lye', type=float, default=None,
                        help='Fixed lye flow rate v_lye in m3/s per stack (default: use controller output)')
    parser.add_argument('--second_step_time', type=float, default=None,
                        help='Time of second step (bidirectional test) in seconds')
    parser.add_argument('--second_step_I', type=float, default=None,
                        help='Current target after second step in A (default: back to I_initial)')
    args = parser.parse_args()

    run_current_step_test(
        model_type=args.model_type,
        duration=args.duration,
        step_time=args.step_time,
        I_initial=args.I_initial,
        I_final=args.I_final,
        second_step_time=args.second_step_time,
        second_step_I=args.second_step_I,
        warmup_controller=args.warmup_controller,
        warmup_duration=args.warmup_duration,
        output_subdir=args.output_subdir,
        use_cbf_projection=not args.no_cbf,
        v_c_fixed=args.v_c,
        v_lye_fixed=args.v_lye,
    )
