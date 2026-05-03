"""
测试 batch simulator 在不同 dt_sub 下的速度
"""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import torch
import time
from plant.multi_stack_simulator_batch import BatchMultiStackSimulator

state_np = np.array([348.0, 353.0, 353.5, 352.8, 353.2, 351.0, 295.0,
                     0.001, 0.0012, 0.0009, 0.0011, 0.005, 0.0001], dtype=np.float32)
action_np = np.array([2500.0, 2500.0, 2500.0, 2500.0,
                      0.03, 0.03, 0.03, 0.03, 0.03], dtype=np.float32)

state_t = torch.from_numpy(state_np).unsqueeze(0).cuda()
action_t = torch.from_numpy(action_np).unsqueeze(0).cuda()

print("测试 batch simulator 单步 (60s) 速度:\n")
for dt_sub in [2.0, 1.0, 0.5, 0.2]:
    batch_sim = BatchMultiStackSimulator(dt=60.0, dt_sub=dt_sub, device='cuda')
    # Warm-up
    _ = batch_sim.step(state_t.clone(), action_t.clone())
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n_runs = 20
    for _ in range(n_runs):
        _ = batch_sim.step(state_t.clone(), action_t.clone())
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    avg_ms = (t1 - t0) / n_runs * 1000
    print(f"dt_sub={dt_sub}s (n_sub={batch_sim.n_sub}): {avg_ms:.2f} ms/step")

print("\n测试 batch_size=128 的 5 步 rollout 速度:\n")
for dt_sub in [2.0, 1.0, 0.5, 0.2]:
    batch_sim = BatchMultiStackSimulator(dt=60.0, dt_sub=dt_sub, device='cuda')
    batch_size = 128
    x = state_t.expand(batch_size, -1).clone()
    u = action_t.expand(batch_size, -1).clone()
    # Warm-up
    for _ in range(5):
        x = batch_sim.step(x, u)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n_runs = 5
    for _ in range(n_runs):
        x0 = state_t.expand(batch_size, -1).clone()
        for k in range(5):
            x0 = batch_sim.step(x0, u)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    avg_ms = (t1 - t0) / n_runs * 1000
    print(f"dt_sub={dt_sub}s: 128x5 step rollout = {avg_ms:.2f} ms")
