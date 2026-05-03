"""
测试 dt_sub=2.0s 的 batch simulator 与原始 simulator 的数值一致性
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
sim.reset(initial_state=state_np.copy())
next_ref = sim.step(action_np.copy())

batch_sim = BatchMultiStackSimulator(dt=60.0, dt_sub=2.0, device='cuda')
state_t = torch.from_numpy(state_np).unsqueeze(0).cuda()
action_t = torch.from_numpy(action_np).unsqueeze(0).cuda()
next_batch = batch_sim.step(state_t, action_t)[0].cpu().numpy()

abs_diff = np.abs(next_ref - next_batch)
print(f"dt_sub=2.0s 单步 60s 误差: max={abs_diff.max():.6e}, mean={abs_diff.mean():.6e}")
print("原始:", next_ref)
print("Batch:", next_batch)

# 连续 5 步
print("\n连续 5 步对比:")
s_np = state_np.copy()
s_t = state_t.clone()
for i in range(5):
    sim.reset(initial_state=s_np.copy())
    s_np = sim.step(action_np.copy())
    s_t = batch_sim.step(s_t, action_t)
    diff = np.abs(s_np - s_t[0].cpu().numpy())
    print(f"Step {i+1}: max_diff={diff.max():.6e}, mean_diff={diff.mean():.6e}")
