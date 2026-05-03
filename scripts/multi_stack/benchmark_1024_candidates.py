"""
Benchmark 1024 candidates sampling + JIT batch rollout
"""
import os, sys, time, numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
from controller.multi_stack.model_controller import MultiStackModelController

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
model_path = os.path.join(project_root, 'output', 'multi_stack', 'policy', 'diffusion_tcn_l3_policy_best.pth')
stats_path = os.path.join(project_root, 'output', 'multi_stack', 'diffusion_stats.npz')

ctrl = MultiStackModelController(dt=60.0, horizon=5, model_type='diffusion_tcn_l3',
                                 model_path=model_path, stats_path=stats_path)

state = np.array([348.0, 353.0, 353.5, 352.8, 353.2, 351.0, 295.0,
                  0.001, 0.0012, 0.0009, 0.0011, 0.005, 0.0001], dtype=np.float32)
P_ref = [10.0e6]*5
T_ref = 353.15
last_action = [np.array([2500.0]*4), np.array([0.03]*4), 0.03]
last_action_vec = ctrl._as_last_action_vec(last_action)
cond_norm = ctrl._prepare_condition(state, P_ref, T_ref, last_action_vec)

P_ref_eval = np.array(P_ref[:ctrl.horizon])

for num_candidates in [128, 256, 512, 1024]:
    cond_norm_batch = cond_norm.repeat(num_candidates, 1)

    # Warm-up
    with torch.no_grad():
        _ = ctrl.scheduler.sample(ctrl.model, cond_norm_batch, (num_candidates, ctrl.action_dim, ctrl.horizon))
    torch.cuda.synchronize()

    # Sampling time
    t0 = time.perf_counter()
    with torch.no_grad():
        samples_norm = ctrl.scheduler.sample(ctrl.model, cond_norm_batch, (num_candidates, ctrl.action_dim, ctrl.horizon))
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    sampling_ms = (t1-t0)*1000

    actions_denorm = ctrl._denormalize_action(samples_norm)
    actions_proj = ctrl._projection(actions_denorm)
    actions = ctrl._clamp_action(actions_proj)

    # Batch eval
    x = ctrl.batch_sim.reset(initial_state=state, batch_size=num_candidates)
    I_prev = torch.from_numpy(last_action_vec[0:4]).float().to(ctrl.device).unsqueeze(0).expand(num_candidates, -1)
    v_lye_prev = torch.from_numpy(last_action_vec[4:8]).float().to(ctrl.device).unsqueeze(0).expand(num_candidates, -1)
    v_c_prev = torch.tensor(last_action_vec[8], dtype=torch.float32, device=ctrl.device).unsqueeze(0).expand(num_candidates)
    P_ref_t = torch.from_numpy(P_ref_eval).float().to(ctrl.device)
    T_ref_t = torch.tensor(T_ref, dtype=torch.float32, device=ctrl.device)
    costs = torch.zeros(num_candidates, dtype=torch.float32, device=ctrl.device)

    t0 = time.perf_counter()
    for k in range(ctrl.horizon):
        u_k = actions[:, :, k]
        I_k = u_k[:, 0:4]
        v_lye_k = u_k[:, 4:8]
        v_c_k = u_k[:, 8]
        x = ctrl.batch_sim.step(x, u_k)
        T_s_next = x[:, 1:5]
        P_total, _, _ = ctrl.batch_sim.calculate_power(I_k, T_s_next)
        costs += ctrl.lambda_track * ((P_total - P_ref_t[k]) / 1e6) ** 2
        costs += ctrl.lambda_temp * torch.sum((T_s_next - T_ref_t) ** 2, dim=1)
        cost_eq = ((I_k[:, 0] - I_k[:, 1])**2 + (I_k[:, 0] - I_k[:, 2])**2 + (I_k[:, 0] - I_k[:, 3])**2 +
                   (I_k[:, 1] - I_k[:, 2])**2 + (I_k[:, 1] - I_k[:, 3])**2 + (I_k[:, 2] - I_k[:, 3])**2)
        costs += ctrl.lambda_prod * cost_eq
        dI = I_k - I_prev
        dv_lye = v_lye_k - v_lye_prev
        dv_c = v_c_k - v_c_prev
        costs += ctrl.lambda_I * torch.sum(dI**2, dim=1)
        costs += ctrl.lambda_lye * torch.sum(dv_lye**2, dim=1)
        costs += ctrl.lambda_c * (dv_c**2)
        I_prev = I_k
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    rollout_ms = (t1-t0)*1000

    costs_np = costs.cpu().numpy()
    best_idx = int(np.argmin(costs_np))
    print(f'{num_candidates:4d} candidates: sampling={sampling_ms:6.1f}ms, rollout={rollout_ms:6.1f}ms, total={sampling_ms+rollout_ms:6.1f}ms, best_cost={costs_np[best_idx]:.4f}')
