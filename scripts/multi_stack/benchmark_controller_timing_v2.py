"""
多槽控制器生成控制序列时间对比测试 V2
对比方案：
  1. 全NMPC (dt_sub=0.1s)
  2. TCN Diffusion L3, 采样128, Serial Rollout
  3. TCN Diffusion L3, 采样128, JIT Batch Rollout (dt_sub=2.0s)
  4. TCN Diffusion L3 + CBF投射, 采样128, JIT Batch Rollout

测试方法：固定状态下多次调用 get_action，统计平均耗时。
"""

import os
import sys
import time
import numpy as np
import json
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from plant.multi_stack_simulator import MultiStackSimulator
from controller.multi_stack.nmpc_controller import MultiStackNMPCController
from controller.multi_stack.model_controller import MultiStackModelController
from controller.multi_stack.cbf_projection_ho import MultiStackCBFProjectionHO


def build_typical_state():
    """构造一个典型的稳态工作点状态（预热后状态）。"""
    state = np.array([
        348.0,          # T_s_in
        353.0, 353.5, 352.8, 353.2,  # T_s1..4
        351.0,          # T_sep
        295.0,          # T_c_out
        0.001, 0.0012, 0.0009, 0.0011,  # n_H2_an
        0.005,          # n_liq
        0.0001,         # n_gas
    ], dtype=float)
    return state


def benchmark_nmpc(state, P_ref, T_ref, last_action, n_warmup=3, n_runs=5, dt_sub=0.1):
    print(f"\n[1/4] Benchmarking Full NMPC (dt_sub={dt_sub}s)...")
    print("  初始化 NMPC 控制器（首次编译 CasADi 符号图，可能需要几十秒）...")
    ctrl = MultiStackNMPCController(dt=60.0, horizon=5, dt_sub=dt_sub)
    print("  NMPC 初始化完成。")

    for i in range(n_warmup):
        print(f"  Warm-up {i+1}/{n_warmup} ...", end='', flush=True)
        t_w0 = time.perf_counter()
        _ = ctrl.get_action(state, P_ref, T_ref, last_action)
        t_w1 = time.perf_counter()
        print(f" {(t_w1 - t_w0)*1000.0:.1f} ms")

    times = []
    for i in range(n_runs):
        print(f"  Run {i+1}/{n_runs} ...", end='', flush=True)
        t0 = time.perf_counter()
        _ = ctrl.get_action(state, P_ref, T_ref, last_action)
        t1 = time.perf_counter()
        dt_ms = (t1 - t0) * 1000.0
        times.append(dt_ms)
        print(f" {dt_ms:.1f} ms")

    arr = np.array(times)
    stats = {
        'controller': 'nmpc_full',
        'dt_sub': dt_sub,
        'n_runs': n_runs,
        'mean_ms': float(arr.mean()),
        'std_ms': float(arr.std()),
        'min_ms': float(arr.min()),
        'max_ms': float(arr.max()),
        'median_ms': float(np.median(arr)),
        'p95_ms': float(np.percentile(arr, 95)),
        'p99_ms': float(np.percentile(arr, 99)),
    }
    print(f"  Mean: {stats['mean_ms']:.2f} ms, Std: {stats['std_ms']:.2f} ms, "
          f"Min: {stats['min_ms']:.2f} ms, Max: {stats['max_ms']:.2f} ms, "
          f"Median: {stats['median_ms']:.2f} ms")
    return stats, times


def benchmark_diffusion(state, P_ref, T_ref, last_action, use_batch, n_warmup=5, n_runs=50):
    tag = "JIT_Batch" if use_batch else "Serial"
    idx = "3" if use_batch else "2"
    print(f"\n[{idx}/4] Benchmarking TCN Diffusion L3 (samples=128, {tag})...")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    model_path = os.path.join(project_root, 'output', 'multi_stack', 'policy', 'diffusion_tcn_l3_policy_best.pth')
    stats_path = os.path.join(project_root, 'output', 'multi_stack', 'diffusion_stats.npz')

    print("  加载模型...")
    ctrl = MultiStackModelController(
        dt=60.0, horizon=5,
        model_type='diffusion_tcn_l3',
        model_path=model_path,
        stats_path=stats_path
    )

    # Force mode
    ctrl.use_batch_rollout = use_batch
    print(f"  Batch rollout: {ctrl.use_batch_rollout}")

    for i in range(n_warmup):
        print(f"  Warm-up {i+1}/{n_warmup} ...", end='', flush=True)
        t_w0 = time.perf_counter()
        _ = ctrl.get_action(state, P_ref, T_ref, last_action)
        t_w1 = time.perf_counter()
        print(f" {(t_w1 - t_w0)*1000.0:.1f} ms")

    times = []
    for i in range(n_runs):
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  Run {i+1}/{n_runs} ...", end='', flush=True)
        t0 = time.perf_counter()
        _ = ctrl.get_action(state, P_ref, T_ref, last_action)
        t1 = time.perf_counter()
        dt_ms = (t1 - t0) * 1000.0
        times.append(dt_ms)
        if (i + 1) % 5 == 0 or i == 0:
            print(f" {dt_ms:.1f} ms (最近5次平均)")

    arr = np.array(times)
    name = 'diffusion_tcn_l3_batch' if use_batch else 'diffusion_tcn_l3_serial'
    stats = {
        'controller': name,
        'samples': 128,
        'n_runs': n_runs,
        'mean_ms': float(arr.mean()),
        'std_ms': float(arr.std()),
        'min_ms': float(arr.min()),
        'max_ms': float(arr.max()),
        'median_ms': float(np.median(arr)),
        'p95_ms': float(np.percentile(arr, 95)),
        'p99_ms': float(np.percentile(arr, 99)),
    }
    print(f"  Mean: {stats['mean_ms']:.2f} ms, Std: {stats['std_ms']:.2f} ms, "
          f"Min: {stats['min_ms']:.2f} ms, Max: {stats['max_ms']:.2f} ms, "
          f"Median: {stats['median_ms']:.2f} ms")
    return stats, times


def benchmark_diffusion_with_cbf(state, P_ref, T_ref, last_action, n_warmup=5, n_runs=50):
    print(f"\n[4/4] Benchmarking TCN Diffusion L3 + CBF Projection (samples=128, JIT Batch)...")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    model_path = os.path.join(project_root, 'output', 'multi_stack', 'policy', 'diffusion_tcn_l3_policy_best.pth')
    stats_path = os.path.join(project_root, 'output', 'multi_stack', 'diffusion_stats.npz')

    print("  加载模型...")
    ctrl = MultiStackModelController(
        dt=60.0, horizon=5,
        model_type='diffusion_tcn_l3',
        model_path=model_path,
        stats_path=stats_path
    )
    ctrl.use_batch_rollout = True

    projector = MultiStackCBFProjectionHO(
        dt=60.0,
        gamma_vec=[10.0]*4 + [1.0] + [10.0]*4,
        rho_vec=[5000]*9,
        h_margin_vec=[0.0]*4 + [0.001] + [0.0]*4,
        normalize=True,
        lambda_u_scale=0.0,
        soft_mask=[True]*9,
        alpha1_vec=[10.0]*4 + [1.0] + [10.0]*4,
        alpha2_vec=[2.0]*9,
        u_weight_scale=[80.0]*4 + [1.0]*4 + [0.1],
    )

    last_action_vec = np.concatenate([last_action[0], last_action[1], [last_action[2]]])

    for i in range(n_warmup):
        print(f"  Warm-up {i+1}/{n_warmup} ...", end='', flush=True)
        t_w0 = time.perf_counter()
        I_cmd, v_lye_cmd, v_c_cmd = ctrl.get_action(state, P_ref, T_ref, last_action)
        raw_action = np.concatenate([I_cmd, v_lye_cmd, [v_c_cmd]])
        _, _, _ = projector.project(raw_action, state, u_last=last_action_vec, verbose=False)
        t_w1 = time.perf_counter()
        print(f" {(t_w1 - t_w0)*1000.0:.1f} ms")

    times = []
    times_model_only = []
    times_cbf_only = []
    for i in range(n_runs):
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  Run {i+1}/{n_runs} ...", end='', flush=True)

        t0 = time.perf_counter()
        I_cmd, v_lye_cmd, v_c_cmd = ctrl.get_action(state, P_ref, T_ref, last_action)
        t1 = time.perf_counter()
        model_ms = (t1 - t0) * 1000.0

        raw_action = np.concatenate([I_cmd, v_lye_cmd, [v_c_cmd]])
        t2 = time.perf_counter()
        _, _, _ = projector.project(raw_action, state, u_last=last_action_vec, verbose=False)
        t3 = time.perf_counter()
        cbf_ms = (t3 - t2) * 1000.0

        times.append(model_ms + cbf_ms)
        times_model_only.append(model_ms)
        times_cbf_only.append(cbf_ms)

        if (i + 1) % 5 == 0 or i == 0:
            print(f" total={model_ms+cbf_ms:.1f} ms (model={model_ms:.1f}+cbf={cbf_ms:.1f})")

    arr = np.array(times)
    arr_model = np.array(times_model_only)
    arr_cbf = np.array(times_cbf_only)
    stats = {
        'controller': 'diffusion_tcn_l3_cbf_batch',
        'samples': 128,
        'n_runs': n_runs,
        'mean_ms': float(arr.mean()),
        'std_ms': float(arr.std()),
        'min_ms': float(arr.min()),
        'max_ms': float(arr.max()),
        'median_ms': float(np.median(arr)),
        'p95_ms': float(np.percentile(arr, 95)),
        'p99_ms': float(np.percentile(arr, 99)),
        'model_only_mean_ms': float(arr_model.mean()),
        'cbf_only_mean_ms': float(arr_cbf.mean()),
    }
    print(f"  Total Mean: {stats['mean_ms']:.2f} ms, Std: {stats['std_ms']:.2f} ms, "
          f"Min: {stats['min_ms']:.2f} ms, Max: {stats['max_ms']:.2f} ms, "
          f"Median: {stats['median_ms']:.2f} ms")
    print(f"  Breakdown: Model {stats['model_only_mean_ms']:.2f} ms + CBF {stats['cbf_only_mean_ms']:.2f} ms")
    return stats, times


def main():
    state = build_typical_state()
    P_ref = [10.0e6] * 5
    T_ref = 353.15
    last_action = [
        np.array([2500.0, 2500.0, 2500.0, 2500.0]),
        np.array([0.03, 0.03, 0.03, 0.03]),
        0.03
    ]

    n_warmup = 5
    n_runs = 50

    print("=" * 70)
    print("控制器生成控制序列时间对比测试 V2")
    print(f"Warm-up: {n_warmup} 次, 测量: {n_runs} 次")
    print(f"固定状态: T_s_mean={np.mean(state[1:5]):.2f}K, T_sep={state[5]:.2f}K")
    print("=" * 70)

    stats_nmpc, _ = benchmark_nmpc(state, P_ref, T_ref, last_action, n_warmup=3, n_runs=5, dt_sub=0.1)
    stats_diff_serial, _ = benchmark_diffusion(state, P_ref, T_ref, last_action, use_batch=False, n_warmup=5, n_runs=50)
    stats_diff_batch, _ = benchmark_diffusion(state, P_ref, T_ref, last_action, use_batch=True, n_warmup=5, n_runs=50)
    stats_diff_cbf, _ = benchmark_diffusion_with_cbf(state, P_ref, T_ref, last_action, n_warmup=5, n_runs=50)

    # Summary
    print("\n" + "=" * 70)
    print("汇总结果 (单位: ms)")
    print("=" * 70)
    print(f"{'方案':<45} {'Mean':>10} {'Median':>10} {'Std':>10} {'Max':>10}")
    print("-" * 85)
    all_stats = [stats_nmpc, stats_diff_serial, stats_diff_batch, stats_diff_cbf]
    names = {
        'nmpc_full': '全NMPC (dt_sub=0.1s)',
        'diffusion_tcn_l3_serial': 'TCN Diffusion L3 Serial (128)',
        'diffusion_tcn_l3_batch': 'TCN Diffusion L3 JIT Batch (128)',
        'diffusion_tcn_l3_cbf_batch': 'TCN Diffusion L3 + CBF JIT Batch (128)',
    }
    for s in all_stats:
        name = names[s['controller']]
        print(f"{name:<45} {s['mean_ms']:>10.2f} {s['median_ms']:>10.2f} {s['std_ms']:>10.2f} {s['max_ms']:>10.2f}")

    # Speedup
    nmpc_mean = stats_nmpc['mean_ms']
    print("-" * 85)
    print(f"Serial 相对于 NMPC 加速比: {nmpc_mean / stats_diff_serial['mean_ms']:.1f}x")
    print(f"JIT Batch 相对于 NMPC 加速比: {nmpc_mean / stats_diff_batch['mean_ms']:.1f}x")
    print(f"JIT Batch 相对于 Serial 加速比: {stats_diff_serial['mean_ms'] / stats_diff_batch['mean_ms']:.1f}x")
    print(f"JIT Batch + CBF 相对于 NMPC 加速比: {nmpc_mean / stats_diff_cbf['mean_ms']:.1f}x")

    # Save results
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    output_dir = os.path.join(project_root, 'output', 'multi_stack', 'timing_benchmark')
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_path = os.path.join(output_dir, f'timing_benchmark_v2_{timestamp}.json')
    results = {
        'timestamp': timestamp,
        'n_warmup': n_warmup,
        'n_runs': n_runs,
        'state': state.tolist(),
        'stats': all_stats,
    }
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"\n结果已保存: {result_path}")


if __name__ == '__main__':
    main()
