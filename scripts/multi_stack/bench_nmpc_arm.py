"""
Benchmark: Multi-Stack Full NMPC + Simulator on ARM

测量内容:
- Full NMPC.get_action 每步耗时 (dt_sub=0.2s)
- Simulator.step 单步耗时 (dt=0.2s)
- 实时性比 = 实际耗时 / 仿真时间
"""

import os
import sys
import time
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from plant.multi_stack_simulator import MultiStackSimulator
from controller.multi_stack.nmpc_controller import MultiStackNMPCController


def benchmark_nmpc(sim, last_action, P_future, T_REF, n_warmup=1, n_runs=5):
    SIM_DT = 0.2
    DT_CTRL = 60.0
    HORIZON = 3
    n_sub = int(DT_CTRL / SIM_DT)

    print(f"\n[NMPC] Creating FULL NMPC (dt_sub={SIM_DT}s, horizon={HORIZON})...")
    t0 = time.perf_counter()
    ctrl = MultiStackNMPCController(dt=DT_CTRL, horizon=HORIZON, dt_sub=SIM_DT)
    print(f"  done in {time.perf_counter()-t0:.3f}s")

    for i in range(n_warmup):
        t_w0 = time.perf_counter()
        I_cmd, v_lye_cmd, v_c_cmd = ctrl.get_action(sim.state, P_future[:HORIZON], T_REF, last_action)
        t_w1 = time.perf_counter()
        action_sim = np.concatenate([I_cmd, v_lye_cmd, [v_c_cmd]])
        last_action = [I_cmd, v_lye_cmd, v_c_cmd]
        for _ in range(n_sub):
            sim.step(action_sim)
        print(f"  Warm-up {i+1}/{n_warmup}: solve={(t_w1-t_w0)*1000:.1f} ms")

    solve_times = []
    sim_times = []
    for i in range(n_runs):
        t_solve0 = time.perf_counter()
        I_cmd, v_lye_cmd, v_c_cmd = ctrl.get_action(sim.state, P_future[:HORIZON], T_REF, last_action)
        t_solve = time.perf_counter() - t_solve0
        solve_times.append(t_solve)

        action_sim = np.concatenate([I_cmd, v_lye_cmd, [v_c_cmd]])
        last_action = [I_cmd, v_lye_cmd, v_c_cmd]

        t_sim0 = time.perf_counter()
        for _ in range(n_sub):
            sim.step(action_sim)
        t_sim = time.perf_counter() - t_sim0
        sim_times.append(t_sim)

        print(f"  Run {i+1}/{n_runs}: solve={t_solve*1000:7.1f} ms, sim={t_sim*1000:7.1f} ms, total={(t_solve+t_sim)*1000:7.1f} ms")

    solve_arr = np.array(solve_times) * 1000
    sim_arr = np.array(sim_times) * 1000
    cycle_arr = solve_arr + sim_arr
    return solve_arr, sim_arr, cycle_arr


def stats(name, a, unit='ms'):
    print(f"  {name:34s} mean={a.mean():8.3f}  median={np.median(a):8.3f}  "
          f"min={a.min():8.3f}  max={a.max():8.3f}  std={a.std():7.3f}  [{unit}]")


def main():
    DT_CTRL = 60.0
    SIM_DT = 0.2
    T_REF = 353.15
    N_WARMUP = 1
    N_RUNS = 5

    print("=" * 70)
    print(f"Multi-Stack Full NMPC Benchmark on ARM (dt_sub={SIM_DT}s)")
    print(f"Platform: aarch64, dt_ctrl={DT_CTRL}s")
    print(f"Warm-up: {N_WARMUP}, Measured runs: {N_RUNS}")
    print("=" * 70)

    print(f"\n[Setup] Creating simulator (dt={SIM_DT}s)...")
    t0 = time.perf_counter()
    sim = MultiStackSimulator(dt=SIM_DT)
    sim.reset()
    print(f"  done in {time.perf_counter()-t0:.3f}s")

    last_action = [
        np.ones(4) * 2000.0,
        np.ones(4) * 0.03,
        0.0,
    ]
    P_future = [10.0e6] * 5

    nmpc_solve, nmpc_sim, nmpc_cycle = benchmark_nmpc(
        sim, last_action, P_future, T_REF, n_warmup=N_WARMUP, n_runs=N_RUNS)

    print("\n" + "=" * 70)
    print("Summary Results")
    print("=" * 70)
    stats("NMPC solve / cycle", nmpc_solve)
    stats("NMPC sim / cycle",   nmpc_sim)
    stats("NMPC total / cycle", nmpc_cycle)
    print()
    print(f"  NMPC 实时性比率: {nmpc_cycle.mean()/(DT_CTRL*1000):.4f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
