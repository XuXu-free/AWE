"""
对比不同积分器实现和精度
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
next_state_ref = sim.step(action_np.copy())

state_t = torch.from_numpy(state_np).unsqueeze(0).cuda()
action_t = torch.from_numpy(action_np).unsqueeze(0).cuda()

# 临时修改 batch_sim step 方法测试不同积分器
class BatchSimTest(BatchMultiStackSimulator):
    def step_euler(self, x, u):
        for _ in range(self.n_sub):
            dx = self._derivatives(x, u)
            x = x + self.dt_sub * dx
        return x

    def step_rk4(self, x, u):
        for _ in range(self.n_sub):
            k1 = self._derivatives(x, u)
            k2 = self._derivatives(x + 0.5 * self.dt_sub * k1, u)
            k3 = self._derivatives(x + 0.5 * self.dt_sub * k2, u)
            k4 = self._derivatives(x + self.dt_sub * k3, u)
            x = x + self.dt_sub / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return x

for dt_sub in [0.2, 0.1, 0.05, 0.02]:
    batch_sim = BatchSimTest(dt=60.0, dt_sub=dt_sub, device='cuda')
    next_euler = batch_sim.step_euler(state_t.clone(), action_t.clone())[0].cpu().numpy()
    next_rk4 = batch_sim.step_rk4(state_t.clone(), action_t.clone())[0].cpu().numpy()
    diff_euler = np.abs(next_euler - next_state_ref)
    diff_rk4 = np.abs(next_rk4 - next_state_ref)
    print(f"dt_sub={dt_sub}s  Euler max={diff_euler.max():.6e} mean={diff_euler.mean():.6e}  |  RK4 max={diff_rk4.max():.6e} mean={diff_rk4.mean():.6e}")
