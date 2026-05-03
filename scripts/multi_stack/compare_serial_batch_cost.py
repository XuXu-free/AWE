"""
对比 Serial Rollout 与 JIT Batch Rollout 的闭环 cost 差异
对固定初始状态，分别用两种模式闭环运行 5 步，比较累计 cost 和轨迹
"""
import os, sys, time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from controller.multi_stack.model_controller import MultiStackModelController
from plant.multi_stack_simulator import MultiStackSimulator

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
model_path = os.path.join(project_root, 'output', 'multi_stack', 'policy', 'diffusion_tcn_l3_policy_best.pth')
stats_path = os.path.join(project_root, 'output', 'multi_stack', 'diffusion_stats.npz')

state0 = np.array([348.0, 353.0, 353.5, 352.8, 353.2, 351.0, 295.0,
                   0.001, 0.0012, 0.0009, 0.0011, 0.005, 0.0001], dtype=np.float32)
P_ref_eval = [10.0e6]*5
T_ref = 353.15
last_action0 = [np.array([2500.0]*4), np.array([0.03]*4), 0.03]

# 权重
lambda_track = 1.2
lambda_temp = 0.15
lambda_prod = 1e-6
lambda_I = 0.0002
lambda_lye = 25000.0
lambda_c = 2500.0

def run_closed_loop(ctrl, use_batch, label):
    ctrl.use_batch_rollout = use_batch
    sim = MultiStackSimulator(dt=60.0)
    sim.reset(initial_state=state0.copy())
    state = state0.copy()
    last_action = [np.copy(last_action0[0]), np.copy(last_action0[1]), last_action0[2]]
    total_cost = 0.0
    total_time = 0.0
    actions_log = []
    states_log = [state.copy()]

    for k in range(5):
        t0 = time.perf_counter()
        I_cmd, v_lye_cmd, v_c_cmd = ctrl.get_action(state, P_ref_eval, T_ref, last_action)
        t1 = time.perf_counter()
        total_time += (t1 - t0)*1000

        u_k = np.concatenate([I_cmd, v_lye_cmd, [v_c_cmd]])
        actions_log.append(u_k.copy())

        # cost for this step
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

        total_cost += lambda_track * ((P_real_next - P_ref_eval[k])/1e6)**2
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
        states_log.append(state.copy())

    print(f"{label}: total_time={total_time:.1f}ms, total_cost={total_cost:.4f}")
    print(f"  Actions:\n" + "\n".join([f"    Step {i}: I={a[0:4].round(1)}, v_lye={a[4:8].round(6)}, v_c={a[8]:.5f}" for i,a in enumerate(actions_log)]))
    return total_cost, total_time, np.array(actions_log), np.array(states_log)

# 加载控制器
ctrl = MultiStackModelController(dt=60.0, horizon=5, model_type='diffusion_tcn_l3',
                                 model_path=model_path, stats_path=stats_path)

# Warm-up
_ = ctrl.get_action(state0, P_ref_eval, T_ref, last_action0)

print("=" * 70)
print("Serial vs JIT Batch Rollout -- Closed-loop 5-step Cost Comparison")
print("=" * 70)

cost_serial, t_serial, acts_serial, states_serial = run_closed_loop(ctrl, False, "Serial   ")
cost_batch, t_batch, acts_batch, states_batch = run_closed_loop(ctrl, True, "JIT Batch")

print("\n" + "-" * 70)
act_diff = np.abs(acts_serial - acts_batch).max()
state_diff = np.abs(states_serial - states_batch).max()
print(f"Max action diff:  {act_diff:.4f}")
print(f"Max state diff:   {state_diff:.6f}")
print(f"Cost diff:        {abs(cost_serial - cost_batch):.6f}")
print(f"Speedup:          {t_serial/t_batch:.1f}x")
