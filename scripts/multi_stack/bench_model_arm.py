"""
Benchmark: Multi-Stack Model Controllers on ARM

测量内容（按列表遍历所有模型）:
- 每个模型 .get_action 推理耗时
- Simulator.step 单步耗时 (dt=0.1s)
- 实时性比 = 实际耗时 / 仿真时间

模型列表:
- diffusion_tcn        : Diffusion + TCN backbone
- diffusion_mlp        : Diffusion + MLP (with residual blocks)
- diffusion_pure_mlp   : Diffusion + 纯 MLP backbone
- pure_mlp             : 纯 MLP, 不带 diffusion
"""

import os
import sys
import time
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from plant.multi_stack_simulator import MultiStackSimulator


MODEL_CONFIGS = [
    {"model_type": "diffusion_tcn",      "weight": "diffusion_tcn_policy_best.pth"},
    {"model_type": "diffusion_mlp",      "weight": "diffusion_mlp_policy_best.pth"},
    {"model_type": "diffusion_pure_mlp", "weight": "diffusion_pure_mlp_policy_best.pth"},
    {"model_type": "pure_mlp",           "weight": "pure_mlp_policy_best.pth"},
]


def benchmark_model(model_type, weight_name, sim, last_action, P_future, T_REF,
                    n_warmup=1, n_runs=5):
    from controller.multi_stack.model_controller import MultiStackModelController

    DT_CTRL = 60.0
    HORIZON = 5
    SIM_DT = 0.1
    n_sub = int(DT_CTRL / SIM_DT)

    model_path = os.path.join(PROJECT_ROOT, 'output', 'multi_stack', 'policy', weight_name)
    stats_path = os.path.join(PROJECT_ROOT, 'output', 'multi_stack', 'diffusion_stats.npz')

    if not os.path.exists(model_path):
        print(f"\n[{model_type}] SKIP — weight not found: {model_path}")
        return None

    print(f"\n[{model_type}] Loading model from {weight_name}...")
    t0 = time.perf_counter()
    ctrl = MultiStackModelController(
        dt=DT_CTRL, horizon=HORIZON,
        model_type=model_type,
        model_path=model_path,
        stats_path=stats_path,
    )
    print(f"  done in {time.perf_counter()-t0:.3f}s")
    print(f"  device: {ctrl.device}, batch_rollout: {ctrl.use_batch_rollout}, candidates: {ctrl.num_candidates}")

    for i in range(n_warmup):
        t_w0 = time.perf_counter()
        I_cmd, v_lye_cmd, v_c_cmd = ctrl.get_action(sim.state, P_future, T_REF, last_action)
        t_w1 = time.perf_counter()
        action_sim = np.concatenate([I_cmd, v_lye_cmd, [v_c_cmd]])
        last_action = [I_cmd, v_lye_cmd, v_c_cmd]
        for _ in range(n_sub):
            sim.step(action_sim)
        print(f"  Warm-up {i+1}/{n_warmup}: inference={(t_w1-t_w0)*1000:.1f} ms")

    solve_times = []
    sim_times = []
    for i in range(n_runs):
        t_solve0 = time.perf_counter()
        I_cmd, v_lye_cmd, v_c_cmd = ctrl.get_action(sim.state, P_future, T_REF, last_action)
        t_solve = time.perf_counter() - t_solve0
        solve_times.append(t_solve)

        action_sim = np.concatenate([I_cmd, v_lye_cmd, [v_c_cmd]])
        last_action = [I_cmd, v_lye_cmd, v_c_cmd]

        t_sim0 = time.perf_counter()
        for _ in range(n_sub):
            sim.step(action_sim)
        t_sim = time.perf_counter() - t_sim0
        sim_times.append(t_sim)

        print(f"  Run {i+1}/{n_runs}: inference={t_solve*1000:7.1f} ms, sim={t_sim*1000:7.1f} ms, total={(t_solve+t_sim)*1000:7.1f} ms")

    solve_arr = np.array(solve_times) * 1000
    sim_arr = np.array(sim_times) * 1000
    cycle_arr = solve_arr + sim_arr
    return {
        "model_type": model_type,
        "solve": solve_arr,
        "sim": sim_arr,
        "cycle": cycle_arr,
    }


def stats(name, a, unit='ms'):
    print(f"  {name:34s} mean={a.mean():8.3f}  median={np.median(a):8.3f}  "
          f"min={a.min():8.3f}  max={a.max():8.3f}  std={a.std():7.3f}  [{unit}]")


def main():
    DT_CTRL = 60.0
    SIM_DT = 0.1
    T_REF = 353.15
    N_WARMUP = 1
    N_RUNS = 5

    print("=" * 70)
    print(f"Multi-Stack Model Benchmark on ARM (dt_sim={SIM_DT}s)")
    print(f"Platform: aarch64, dt_ctrl={DT_CTRL}s")
    print(f"Warm-up: {N_WARMUP}, Measured runs: {N_RUNS}")
    print(f"Models: {[c['model_type'] for c in MODEL_CONFIGS]}")
    print("=" * 70)

    print(f"\n[Setup] Creating simulator (dt={SIM_DT}s)...")
    t0 = time.perf_counter()
    sim = MultiStackSimulator(dt=SIM_DT)
    sim.reset()
    print(f"  done in {time.perf_counter()-t0:.3f}s")

    init_action = [
        np.ones(4) * 2000.0,
        np.ones(4) * 0.03,
        0.0,
    ]
    P_future = [10.0e6] * 5

    results = []
    for cfg in MODEL_CONFIGS:
        sim.reset()
        last_action = [a.copy() if isinstance(a, np.ndarray) else a for a in init_action]
        res = benchmark_model(
            cfg["model_type"], cfg["weight"],
            sim, last_action, P_future, T_REF,
            n_warmup=N_WARMUP, n_runs=N_RUNS,
        )
        if res is not None:
            results.append(res)

    print("\n" + "=" * 70)
    print("Summary Results")
    print("=" * 70)
    for r in results:
        print(f"\n[{r['model_type']}]")
        stats("inference / cycle", r["solve"])
        stats("sim       / cycle", r["sim"])
        stats("total     / cycle", r["cycle"])
        print(f"  实时性比率: {r['cycle'].mean()/(DT_CTRL*1000):.4f}")
    print("=" * 70)

    if results:
        print("\nInference Time Comparison (mean ms):")
        for r in results:
            print(f"  {r['model_type']:24s} {r['solve'].mean():8.3f} ms")
        baseline = results[0]
        print(f"\nSpeedup vs {baseline['model_type']}:")
        for r in results[1:]:
            ratio = baseline["solve"].mean() / r["solve"].mean()
            print(f"  {r['model_type']:24s} {ratio:6.2f}x")
        print("=" * 70)


if __name__ == '__main__':
    main()
