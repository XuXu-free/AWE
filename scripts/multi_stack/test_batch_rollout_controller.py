"""
验证 model_controller 中 batch rollout 与 serial rollout 的结果一致性
"""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import time
from controller.multi_stack.model_controller import MultiStackModelController

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
model_path = os.path.join(project_root, 'output', 'multi_stack', 'policy', 'diffusion_tcn_l3_policy_best.pth')
stats_path = os.path.join(project_root, 'output', 'multi_stack', 'diffusion_stats.npz')

print("加载控制器...")
ctrl = MultiStackModelController(
    dt=60.0, horizon=5,
    model_type='diffusion_tcn_l3',
    model_path=model_path,
    stats_path=stats_path
)
print(f"Batch rollout available: {ctrl.use_batch_rollout}")

state = np.array([348.0, 353.0, 353.5, 352.8, 353.2, 351.0, 295.0,
                  0.001, 0.0012, 0.0009, 0.0011, 0.005, 0.0001], dtype=np.float32)
P_ref = [10.0e6] * 5
T_ref = 353.15
last_action = [np.array([2500.0]*4), np.array([0.03]*4), 0.03]

# Warm-up
print("Warm-up...")
_ = ctrl.get_action(state, P_ref, T_ref, last_action)

# Test 1: batch rollout (default)
print("\n测试 batch rollout...")
t0 = time.perf_counter()
I_batch, v_lye_batch, v_c_batch = ctrl.get_action(state, P_ref, T_ref, last_action)
t1 = time.perf_counter()
print(f"Batch rollout time: {(t1-t0)*1000:.1f} ms")
print(f"Batch action: I={I_batch}, v_lye={v_lye_batch}, v_c={v_c_batch:.5f}")

# Test 2: serial rollout (force fallback)
if ctrl.use_batch_rollout:
    print("\n测试 serial rollout...")
    # Temporarily disable batch
    old_use_batch = ctrl.use_batch_rollout
    ctrl.use_batch_rollout = False
    t2 = time.perf_counter()
    I_serial, v_lye_serial, v_c_serial = ctrl.get_action(state, P_ref, T_ref, last_action)
    t3 = time.perf_counter()
    ctrl.use_batch_rollout = old_use_batch
    print(f"Serial rollout time: {(t3-t2)*1000:.1f} ms")
    print(f"Serial action: I={I_serial}, v_lye={v_lye_serial}, v_c={v_c_serial:.5f}")

    diff_I = np.abs(I_batch - I_serial)
    diff_v_lye = np.abs(v_lye_batch - v_lye_serial)
    diff_v_c = abs(v_c_batch - v_c_serial)
    print(f"\n动作差异: max_I_diff={diff_I.max():.4f}, max_v_lye_diff={diff_v_lye.max():.6f}, v_c_diff={diff_v_c:.6f}")
