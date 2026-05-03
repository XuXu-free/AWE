"""
测量 TCN Diffusion L3 get_action 内部各阶段耗时细分
"""
import os
import sys
import time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from controller.multi_stack.model_controller import MultiStackModelController
from diffusion.ddpm import DDPMScheduler

# 固定输入
state = np.array([348.0, 353.0, 353.5, 352.8, 353.2, 351.0, 295.0,
                  0.001, 0.0012, 0.0009, 0.0011, 0.005, 0.0001])
P_ref = [10.0e6] * 5
T_ref = 353.15
last_action = [np.array([2500.0]*4), np.array([0.03]*4), 0.03]

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
model_path = os.path.join(project_root, 'output', 'multi_stack', 'policy', 'diffusion_tcn_l3_policy_best.pth')
stats_path = os.path.join(project_root, 'output', 'multi_stack', 'diffusion_stats.npz')

print("加载模型...")
ctrl = MultiStackModelController(
    dt=60.0, horizon=5,
    model_type='diffusion_tcn_l3',
    model_path=model_path,
    stats_path=stats_path
)

last_action_vec = ctrl._as_last_action_vec(last_action)
cond_norm = ctrl._prepare_condition(state, P_ref, T_ref, last_action_vec)

n_warmup = 5
n_runs = 20

# warm-up full pipeline
for i in range(n_warmup):
    _ = ctrl.get_action(state, P_ref, T_ref, last_action)
    print(f"Warm-up {i+1}/{n_warmup}")

times_sample = []
times_post = []
times_rollout = []
times_total = []

for i in range(n_runs):
    # --- 阶段1: 采样 ---
    t0 = time.perf_counter()
    num_candidates = 128
    cond_norm_batch = cond_norm.repeat(num_candidates, 1)
    samples_norm = ctrl.scheduler.sample(ctrl.model, cond_norm_batch, (num_candidates, ctrl.action_dim, ctrl.horizon))
    t1 = time.perf_counter()
    sample_ms = (t1 - t0) * 1000.0

    # --- 阶段2: 后处理 (denormalize + clamp) ---
    actions_denorm = ctrl._denormalize_action(samples_norm)
    actions_proj = ctrl._projection(actions_denorm)
    actions = ctrl._clamp_action(actions_proj)
    actions_np = actions.cpu().numpy()
    t2 = time.perf_counter()
    post_ms = (t2 - t1) * 1000.0

    # --- 阶段3: rollout 评估 ---
    P_ref_eval = np.array(P_ref)[:ctrl.horizon]
    if len(P_ref_eval) < ctrl.horizon:
        P_ref_eval = np.pad(P_ref_eval, (0, ctrl.horizon - len(P_ref_eval)), 'edge')

    best_cost = float('inf')
    for ci in range(num_candidates):
        cost = 0.0
        ctrl.sim_rollout.reset(initial_state=state)
        u_prev = last_action_vec.copy()
        I_prev = u_prev[0:4]
        v_lye_prev = u_prev[4:8]
        v_c_prev = u_prev[8]
        for k in range(ctrl.horizon):
            u_k = actions_np[ci, :, k][:9]
            I_k = u_k[0:4]
            v_lye_k = u_k[4:8]
            v_c_k = u_k[8]
            next_state = ctrl.sim_rollout.step(u_k)
            T_s_vec_next = next_state[1:5]
            _, U_cell_vec_next, _ = ctrl.sim_rollout._calculate_electrochemical_properties(I_k, T_s_vec_next)
            P_real_next = np.sum(U_cell_vec_next * I_k * ctrl.sim_rollout.N_cell)
            cost += ctrl.lambda_track * ((P_real_next - P_ref_eval[k])/1e6)**2
            cost += ctrl.lambda_temp * np.sum((T_s_vec_next - T_ref)**2)
            current_diff_sum = 0.0
            for si in range(4):
                for sj in range(si+1, 4):
                    current_diff_sum += (I_k[si] - I_k[sj])**2
            cost += ctrl.lambda_prod * current_diff_sum
            dI = I_k - I_prev
            dv_lye = v_lye_k - v_lye_prev
            dv_c = v_c_k - v_c_prev
            cost += ctrl.lambda_I * np.sum(dI**2)
            cost += ctrl.lambda_lye * np.sum(dv_lye**2)
            cost += ctrl.lambda_c * (dv_c**2)
            I_prev = I_k
        if cost < best_cost:
            best_cost = cost

    t3 = time.perf_counter()
    rollout_ms = (t3 - t2) * 1000.0
    total_ms = (t3 - t0) * 1000.0

    times_sample.append(sample_ms)
    times_post.append(post_ms)
    times_rollout.append(rollout_ms)
    times_total.append(total_ms)

    print(f"Run {i+1}/{n_runs}: sample={sample_ms:.1f}ms, post={post_ms:.1f}ms, rollout={rollout_ms:.1f}ms, total={total_ms:.1f}ms")

print("\n" + "="*60)
print("TCN Diffusion L3 get_action 内部耗时细分 (ms)")
print("="*60)
for name, arr in [("模型采样 (128 candidates)", times_sample),
                    ("后处理 (denorm+clamp)", times_post),
                    ("Rollout 评估 (128x5步)", times_rollout),
                    ("总计", times_total)]:
    a = np.array(arr)
    print(f"{name:<30} Mean={a.mean():8.2f}  Median={np.median(a):8.2f}  Std={a.std():6.2f}")
