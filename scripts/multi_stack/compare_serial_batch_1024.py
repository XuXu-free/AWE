"""
1024 candidates: Serial vs JIT Batch Rollout 闭环对比 (30步 = 30分钟)
"""
import os, sys, time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from controller.multi_stack.model_controller import MultiStackModelController
from plant.multi_stack_simulator import MultiStackSimulator

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
model_path = os.path.join(project_root, 'output', 'multi_stack', 'policy', 'diffusion_tcn_l3_policy_best.pth')
stats_path = os.path.join(project_root, 'output', 'multi_stack', 'diffusion_stats.npz')

T_ref = 353.15
lambda_track = 1.2
lambda_temp = 0.15
lambda_prod = 1e-6
lambda_I = 0.0002
lambda_lye = 25000.0
lambda_c = 2500.0

def warm_up(ctrl, P_ref, duration_s=14400):
    ctrl.use_batch_rollout = True
    sim = MultiStackSimulator(dt=60.0)
    sim.reset()
    state = sim.state.copy()
    last_action = [np.ones(4)*2000, np.ones(4)*0.03, 0.0]
    n_steps = int(duration_s / 60.0)
    P_future = [P_ref] * ctrl.horizon
    for k in range(n_steps):
        I_cmd, v_lye_cmd, v_c_cmd = ctrl.get_action(state, P_future, T_ref, last_action)
        u_k = np.concatenate([I_cmd, v_lye_cmd, [v_c_cmd]])
        state = sim.step(u_k)
        last_action = [I_cmd, v_lye_cmd, v_c_cmd]
    return state.copy(), [np.copy(last_action[0]), np.copy(last_action[1]), last_action[2]]

def run_test(ctrl, use_batch, label, state0, last_action0, P_ref_val, n_steps=30):
    ctrl.use_batch_rollout = use_batch
    sim = MultiStackSimulator(dt=60.0)
    sim.reset(initial_state=state0.copy())
    state = state0.copy()
    last_action = [np.copy(last_action0[0]), np.copy(last_action0[1]), last_action0[2]]
    total_cost = 0.0
    total_time = 0.0
    P_log = []
    T_log = []
    HTO_log = []
    t_log = []
    P_future = [P_ref_val] * ctrl.horizon

    for k in range(n_steps):
        t = k * 60.0
        t0 = time.perf_counter()
        I_cmd, v_lye_cmd, v_c_cmd = ctrl.get_action(state, P_future, T_ref, last_action)
        t1 = time.perf_counter()
        total_time += (t1 - t0)*1000

        u_k = np.concatenate([I_cmd, v_lye_cmd, [v_c_cmd]])
        I_k = u_k[0:4]
        v_lye_k = u_k[4:8]
        v_c_k = u_k[8]
        I_prev = last_action[0]
        v_lye_prev = last_action[1]
        v_c_prev = last_action[2]

        next_state = sim.step(u_k)
        T_s_next = next_state[1:5]
        _, U_cell_next, _ = sim._calculate_electrochemical_properties(I_k, T_s_next)
        P_real_next = np.sum(U_cell_next * I_k * sim.N_cell)

        total_cost += lambda_track * ((P_real_next - P_ref_val)/1e6)**2
        total_cost += lambda_temp * np.sum((T_s_next - T_ref)**2)
        current_diff_sum = 0.0
        for si in range(4):
            for sj in range(si+1, 4):
                current_diff_sum += (I_k[si] - I_k[sj])**2
        total_cost += lambda_prod * current_diff_sum
        dI = I_k - I_prev
        dv_lye = v_lye_k - v_lye_prev
        dv_c = v_c_k - v_c_prev
        total_cost += lambda_I * np.sum(dI**2)
        total_cost += lambda_lye * np.sum(dv_lye**2)
        total_cost += lambda_c * (dv_c**2)

        state = next_state
        last_action = [I_cmd, v_lye_cmd, v_c_cmd]

        n_H2_sep_gas = state[12]
        T_sep = state[5]
        hto_pct = (n_H2_sep_gas * sim.R * T_sep) / (sim.P_sys * sim.V_sep_gas) * 100
        P_log.append(P_real_next/1e6)
        T_log.append(np.mean(T_s_next)-273.15)
        HTO_log.append(hto_pct)
        t_log.append(t/60.0)

    print(f"{label}: total_time={total_time:.1f}ms, total_cost={total_cost:.4f}")
    return total_cost, total_time, np.array(P_log), np.array(T_log), np.array(HTO_log), np.array(t_log)

ctrl = MultiStackModelController(dt=60.0, horizon=5, model_type='diffusion_tcn_l3',
                                 model_path=model_path, stats_path=stats_path)

print("Warm-up 4h to steady state...")
state_warm, last_action_warm = warm_up(ctrl, 10.0e6, duration_s=14400)
print(f"Warm-up done. T_s_mean={np.mean(state_warm[1:5]):.2f}K")

print("\n" + "=" * 70)
print("1024 candidates: Serial vs JIT Batch (30 steps = 30 min)")
print("=" * 70)

cost_serial, t_serial, P_s, T_s, HTO_s, t_log = run_test(ctrl, False, "Serial   ", state_warm, last_action_warm, 10.0e6, n_steps=30)
cost_batch, t_batch, P_b, T_b, HTO_b, _ = run_test(ctrl, True, "JIT Batch", state_warm, last_action_warm, 10.0e6, n_steps=30)

print("\n" + "-" * 70)
print(f"Cost diff:    {abs(cost_serial - cost_batch):.4f} ({abs(cost_serial-cost_batch)/max(cost_serial,cost_batch)*100:.2f}%)")
print(f"Speedup:      {t_serial/t_batch:.1f}x")
print(f"Max P diff:   {np.abs(P_s - P_b).max():.4f} MW")
print(f"Max T diff:   {np.abs(T_s - T_b).max():.4f} C")
print(f"Max HTO diff: {np.abs(HTO_s - HTO_b).max():.6f} %")

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
ax = axes[0]
ax.plot(t_log, P_s, 'b-', label='Serial', linewidth=1.5)
ax.plot(t_log, P_b, 'r--', label='JIT Batch', linewidth=1.5)
ax.set_title('Power')
ax.set_xlabel('Time (min)')
ax.set_ylabel('MW')
ax.legend()
ax.grid(True)

ax = axes[1]
ax.plot(t_log, T_s, 'b-', label='Serial', linewidth=1.5)
ax.plot(t_log, T_b, 'r--', label='JIT Batch', linewidth=1.5)
ax.set_title('Avg Stack Temperature')
ax.set_xlabel('Time (min)')
ax.set_ylabel('C')
ax.legend()
ax.grid(True)

ax = axes[2]
ax.plot(t_log, HTO_s, 'b-', label='Serial', linewidth=1.5)
ax.plot(t_log, HTO_b, 'r--', label='JIT Batch', linewidth=1.5)
ax.set_title('HTO')
ax.set_xlabel('Time (min)')
ax.set_ylabel('%')
ax.legend()
ax.grid(True)

plt.tight_layout()
out_path = os.path.join(project_root, 'output', 'multi_stack', 'test_step', 'serial_vs_jit_batch_1024_30step.png')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved: {out_path}")
