"""
测试 JIT batch simulator 的速度和精度
"""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import torch
import time

from plant.multi_stack_simulator import MultiStackSimulator
from plant.multi_stack_simulator_batch import BatchMultiStackSimulator
from plant.multi_stack_simulator_batch_jit import BatchMultiStackSimulatorJIT

state_np = np.array([348.0, 353.0, 353.5, 352.8, 353.2, 351.0, 295.0,
                     0.001, 0.0012, 0.0009, 0.0011, 0.005, 0.0001], dtype=np.float32)
action_np = np.array([2500.0, 2500.0, 2500.0, 2500.0,
                      0.03, 0.03, 0.03, 0.03, 0.03], dtype=np.float32)

state_t = torch.from_numpy(state_np).unsqueeze(0).cuda()
action_t = torch.from_numpy(action_np).unsqueeze(0).cuda()

# 参考解（原始 simulator，dt=60.0，内部用 solve_ivp）
sim = MultiStackSimulator(dt=60.0)
sim.reset(initial_state=state_np.copy())
next_ref = sim.step(action_np.copy())

print("=" * 60)
print("1. 单步 60s 精度对比 (vs 原始 solve_ivp)")
print("=" * 60)

for dt_sub in [2.0, 1.0, 0.5, 0.2]:
    # Non-JIT batch
    batch_sim = BatchMultiStackSimulator(dt=60.0, dt_sub=dt_sub, device='cuda')
    next_batch = batch_sim.step(state_t.clone(), action_t.clone())[0].cpu().numpy()
    diff_batch = np.abs(next_ref - next_batch)

    # JIT batch
    jit_sim = BatchMultiStackSimulatorJIT(dt=60.0, dt_sub=dt_sub, device='cuda')
    next_jit = jit_sim.step(state_t.clone(), action_t.clone())[0].cpu().numpy()
    diff_jit = np.abs(next_ref - next_jit)

    print(f"dt_sub={dt_sub}s:")
    print(f"  Batch     max={diff_batch.max():.6e} mean={diff_batch.mean():.6e}")
    print(f"  JIT Batch max={diff_jit.max():.6e} mean={diff_jit.mean():.6e}")

print("\n" + "=" * 60)
print("2. 单步速度对比 (batch_size=1)")
print("=" * 60)
for dt_sub in [2.0, 1.0, 0.5, 0.2]:
    # Non-JIT
    batch_sim = BatchMultiStackSimulator(dt=60.0, dt_sub=dt_sub, device='cuda')
    _ = batch_sim.step(state_t.clone(), action_t.clone())
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        _ = batch_sim.step(state_t.clone(), action_t.clone())
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    ms_batch = (t1 - t0) / 20 * 1000

    # JIT
    jit_sim = BatchMultiStackSimulatorJIT(dt=60.0, dt_sub=dt_sub, device='cuda')
    _ = jit_sim.step(state_t.clone(), action_t.clone())
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        _ = jit_sim.step(state_t.clone(), action_t.clone())
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    ms_jit = (t1 - t0) / 20 * 1000

    print(f"dt_sub={dt_sub}s: Batch={ms_batch:.2f} ms, JIT={ms_jit:.2f} ms, 加速比={ms_batch/ms_jit:.1f}x")

print("\n" + "=" * 60)
print("3. 128×5 rollout 速度对比 (controller 实际场景)")
print("=" * 60)

batch_size = 128
x_base = state_t.expand(batch_size, -1).clone()
u_base = action_t.expand(batch_size, -1).clone()

for dt_sub in [2.0, 1.0, 0.5, 0.2]:
    # Non-JIT
    batch_sim = BatchMultiStackSimulator(dt=60.0, dt_sub=dt_sub, device='cuda')
    x = x_base.clone()
    for _ in range(5):
        x = batch_sim.step(x, u_base)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        x = x_base.clone()
        for _ in range(5):
            x = batch_sim.step(x, u_base)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    ms_batch = (t1 - t0) / 5 * 1000

    # JIT
    jit_sim = BatchMultiStackSimulatorJIT(dt=60.0, dt_sub=dt_sub, device='cuda')
    x = x_base.clone()
    for _ in range(5):
        x = jit_sim.step(x, u_base)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        x = x_base.clone()
        for _ in range(5):
            x = jit_sim.step(x, u_base)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    ms_jit = (t1 - t0) / 5 * 1000

    print(f"dt_sub={dt_sub}s: Batch={ms_batch:.2f} ms, JIT={ms_jit:.2f} ms, 加速比={ms_batch/ms_jit:.1f}x")
