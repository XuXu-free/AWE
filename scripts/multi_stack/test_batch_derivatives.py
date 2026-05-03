"""
对比 Batch 与原始 simulator 在同一点的导数输出
"""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import torch
from plant.multi_stack_simulator import MultiStackSimulator
from plant.multi_stack_simulator_batch import BatchMultiStackSimulator

state_np = np.array([348.0, 353.0, 353.5, 352.8, 353.2, 351.0, 295.0,
                     0.001, 0.0012, 0.0009, 0.0011, 0.005, 0.0001], dtype=np.float32)
action_np = np.array([2500.0, 2500.0, 2500.0, 2500.0,
                      0.03, 0.03, 0.03, 0.03, 0.03], dtype=np.float32)

sim = MultiStackSimulator(dt=60.0)
dxdt_np = sim._get_derivatives(0.0, state_np, action_np)

batch_sim = BatchMultiStackSimulator(dt=60.0, dt_sub=0.2, device='cuda')
state_t = torch.from_numpy(state_np).unsqueeze(0).cuda()
action_t = torch.from_numpy(action_np).unsqueeze(0).cuda()
dxdt_t = batch_sim._derivatives(state_t, action_t)
dxdt_batch = dxdt_t[0].cpu().numpy()

print("原始导数:")
print(dxdt_np)
print("\nBatch 导数:")
print(dxdt_batch)
abs_diff = np.abs(dxdt_np - dxdt_batch)
print("\n绝对误差:")
print(abs_diff)
print(f"\n最大绝对误差: {abs_diff.max():.6e}")

# 先获取原始 simulator 参考结果
sim = MultiStackSimulator(dt=60.0)
sim.reset(initial_state=state_np.copy())
next_state_ref = sim.step(action_np.copy())

# 测试更小 dt_sub 的积分结果
print("\n" + "="*60)
for dt_sub in [0.2, 0.1, 0.05, 0.02]:
    batch_sim2 = BatchMultiStackSimulator(dt=60.0, dt_sub=dt_sub, device='cuda')
    next_t = batch_sim2.step(state_t.clone(), action_t.clone())
    next_np = next_t[0].cpu().numpy()
    diff = np.abs(next_np - next_state_ref)
    print(f"dt_sub={dt_sub}s: max_diff={diff.max():.6e}, mean_diff={diff.mean():.6e}")
