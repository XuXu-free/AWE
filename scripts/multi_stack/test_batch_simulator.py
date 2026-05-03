"""
验证 BatchMultiStackSimulator 与原始 MultiStackSimulator 的数值一致性
"""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import torch
from plant.multi_stack_simulator import MultiStackSimulator
from plant.multi_stack_simulator_batch import BatchMultiStackSimulator

def test_consistency():
    # 固定状态与动作
    state_np = np.array([348.0, 353.0, 353.5, 352.8, 353.2, 351.0, 295.0,
                         0.001, 0.0012, 0.0009, 0.0011, 0.005, 0.0001], dtype=np.float32)
    action_np = np.array([2500.0, 2500.0, 2500.0, 2500.0,
                          0.03, 0.03, 0.03, 0.03, 0.03], dtype=np.float32)

    # 原始 simulator
    sim = MultiStackSimulator(dt=60.0)
    sim.reset(initial_state=state_np.copy())
    next_state_np = sim.step(action_np.copy())

    # Batch simulator (batch_size=1)
    batch_sim = BatchMultiStackSimulator(dt=60.0, dt_sub=0.2, device='cuda')
    state_t = torch.from_numpy(state_np).unsqueeze(0).cuda()
    action_t = torch.from_numpy(action_np).unsqueeze(0).cuda()
    next_state_t = batch_sim.step(state_t, action_t)
    next_state_batch = next_state_t[0].cpu().numpy()

    print("原始 simulator 下一步状态:")
    print(next_state_np)
    print("\nBatch simulator 下一步状态:")
    print(next_state_batch)

    abs_diff = np.abs(next_state_np - next_state_batch)
    rel_diff = abs_diff / (np.abs(next_state_np) + 1e-12)
    print("\n绝对误差:")
    print(abs_diff)
    print("\n相对误差:")
    print(rel_diff)
    print(f"\n最大绝对误差: {abs_diff.max():.6e}")
    print(f"最大相对误差: {rel_diff.max():.6e}")

    # 多次步进一致性
    print("\n" + "="*60)
    print("连续 5 步对比:")
    states_orig = [state_np.copy()]
    states_batch = [state_np.copy()]
    s_np = state_np.copy()
    s_t = state_t.clone()
    for i in range(5):
        s_np = sim.step(action_np.copy())
        s_t = batch_sim.step(s_t, action_t)
        states_orig.append(s_np.copy())
        states_batch.append(s_t[0].cpu().numpy().copy())
        diff = np.abs(states_orig[-1] - states_batch[-1])
        print(f"Step {i+1}: max_abs_diff={diff.max():.6e}, mean_abs_diff={diff.mean():.6e}")

if __name__ == '__main__':
    test_consistency()
