"""
四控制器单次求解速度对比测试
对比方案：
  1. 全NMPC (dt_sub=0.1s)
  2. TCN Diffusion L3, 1024 candidates, JIT Batch Rollout
  3. Diffusion Pure MLP, 128 candidates (DDPM采样)
  4. Pure MLP (直接前向传播, 无采样)

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


def build_typical_state():
    state = np.array([
        348.0,
        353.0, 353.5, 352.8, 353.2,
        351.0,
        295.0,
        0.001, 0.0012, 0.0009, 0.0011,
        0.005,
        0.0001,
    ], dtype=float)
    return state


def benchmark_nmpc(state, P_ref, T_ref, last_action, n_warmup=3, n_runs=10, dt_sub=0.1):
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


def benchmark_diffusion_tcn_l3_batch(state, P_ref, T_ref, last_action, n_warmup=5, n_runs=50):
    print(f"\n[2/4] Benchmarking TCN Diffusion L3 (samples=1024, JIT Batch)...")
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
    ctrl.num_candidates = 1024
    ctrl.use_batch_rollout = True
    print(f"  num_candidates={ctrl.num_candidates}, use_batch_rollout={ctrl.use_batch_rollout}")

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
    stats = {
        'controller': 'diffusion_tcn_l3_batch_1024',
        'samples': 1024,
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


def benchmark_diffusion_pure_mlp(state, P_ref, T_ref, last_action, n_warmup=5, n_runs=50):
    print(f"\n[3/4] Benchmarking Diffusion Pure MLP (samples=128)...")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    model_path = os.path.join(project_root, 'output', 'multi_stack', 'policy', 'diffusion_pure_mlp_policy_best.pth')
    stats_path = os.path.join(project_root, 'output', 'multi_stack', 'diffusion_stats.npz')

    print("  加载模型...")
    ctrl = MultiStackModelController(
        dt=60.0, horizon=5,
        model_type='diffusion_pure_mlp',
        model_path=model_path,
        stats_path=stats_path
    )
    ctrl.num_candidates = 128
    print(f"  num_candidates={ctrl.num_candidates}")

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
    stats = {
        'controller': 'diffusion_pure_mlp',
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


def benchmark_pure_mlp(state, P_ref, T_ref, last_action, n_warmup=5, n_runs=50):
    print(f"\n[4/4] Benchmarking Pure MLP (direct forward, no sampling)...")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    model_path = os.path.join(project_root, 'output', 'multi_stack', 'policy', 'pure_mlp_policy_best.pth')
    stats_path = os.path.join(project_root, 'output', 'multi_stack', 'diffusion_stats.npz')

    print("  加载模型...")
    ctrl = MultiStackModelController(
        dt=60.0, horizon=5,
        model_type='pure_mlp',
        model_path=model_path,
        stats_path=stats_path
    )

    for i in range(n_warmup):
        print(f"  Warm-up {i+1}/{n_warmup} ...", end='', flush=True)
        t_w0 = time.perf_counter()
        _ = ctrl.get_action(state, P_ref, T_ref, last_action)
        t_w1 = time.perf_counter()
        print(f" {(t_w1 - t_w0)*1000.0:.1f} ms")

    times = []
    for i in range(n_runs):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Run {i+1}/{n_runs} ...", end='', flush=True)
        t0 = time.perf_counter()
        _ = ctrl.get_action(state, P_ref, T_ref, last_action)
        t1 = time.perf_counter()
        dt_ms = (t1 - t0) * 1000.0
        times.append(dt_ms)
        if (i + 1) % 10 == 0 or i == 0:
            print(f" {dt_ms:.1f} ms (最近10次平均)")

    arr = np.array(times)
    stats = {
        'controller': 'pure_mlp',
        'samples': 1,
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
    print("四控制器单次求解速度对比测试")
    print(f"Warm-up: {n_warmup} 次, 测量: {n_runs} 次")
    print(f"固定状态: T_s_mean={np.mean(state[1:5]):.2f}K, T_sep={state[5]:.2f}K")
    print("=" * 70)

    stats_nmpc, _ = benchmark_nmpc(state, P_ref, T_ref, last_action, n_warmup=3, n_runs=5, dt_sub=0.1)
    stats_diff_batch, _ = benchmark_diffusion_tcn_l3_batch(state, P_ref, T_ref, last_action, n_warmup=5, n_runs=50)
    stats_diff_pure_mlp, _ = benchmark_diffusion_pure_mlp(state, P_ref, T_ref, last_action, n_warmup=5, n_runs=50)
    stats_pure_mlp, _ = benchmark_pure_mlp(state, P_ref, T_ref, last_action, n_warmup=5, n_runs=50)

    # Summary
    print("\n" + "=" * 70)
    print("汇总结果 (单位: ms)")
    print("=" * 70)
    print(f"{'方案':<45} {'Mean':>10} {'Median':>10} {'Std':>10} {'Max':>10}")
    print("-" * 85)
    all_stats = [stats_nmpc, stats_diff_batch, stats_diff_pure_mlp, stats_pure_mlp]
    names = {
        'nmpc_full': 'Full NMPC (dt_sub=0.1s)',
        'diffusion_tcn_l3_batch_1024': 'TCN Diffusion L3 Batch (1024)',
        'diffusion_pure_mlp': 'Diffusion Pure MLP (128)',
        'pure_mlp': 'Pure MLP (Direct)',
    }
    for s in all_stats:
        name = names[s['controller']]
        print(f"{name:<45} {s['mean_ms']:>10.2f} {s['median_ms']:>10.2f} {s['std_ms']:>10.2f} {s['max_ms']:>10.2f}")

    # Speedup relative to NMPC
    nmpc_mean = stats_nmpc['mean_ms']
    print("-" * 85)
    print(f"TCN Diffusion L3 Batch 相对于 NMPC 加速比: {nmpc_mean / stats_diff_batch['mean_ms']:.1f}x")
    print(f"Diffusion Pure MLP 相对于 NMPC 加速比: {nmpc_mean / stats_diff_pure_mlp['mean_ms']:.1f}x")
    print(f"Pure MLP 相对于 NMPC 加速比: {nmpc_mean / stats_pure_mlp['mean_ms']:.1f}x")

    # Save results
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    output_dir = os.path.join(project_root, 'output', 'multi_stack', 'timing_benchmark')
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_path = os.path.join(output_dir, f'four_controller_benchmark_{timestamp}.json')
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

    # Plot
    plot_comparison(all_stats, output_dir, timestamp)


def plot_comparison(all_stats, output_dir, timestamp):
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    import matplotlib.pyplot as plt

    names_display = {
        'nmpc_full': 'Full NMPC\n(dt_sub=0.1s)',
        'diffusion_tcn_l3_batch_1024': 'TCN Diffusion L3\nBatch (1024)',
        'diffusion_pure_mlp': 'Diffusion Pure MLP\n(128 samples)',
        'pure_mlp': 'Pure MLP\n(Direct)',
    }
    colors = ['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c']

    labels = [names_display[s['controller']] for s in all_stats]
    means = [s['mean_ms'] for s in all_stats]
    stds = [s['std_ms'] for s in all_stats]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(labels, means, yerr=stds, color=colors, edgecolor='black', linewidth=0.8, capsize=5, error_kw={'linewidth': 1.5})

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.annotate(f'{mean:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height + std),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('单次求解时间 (ms)', fontsize=12)
    ax.set_title('多槽AWE控制器单次求解速度对比', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)

    # Log scale if NMPC is much larger
    if max(means) / min(means) > 10:
        ax.set_yscale('log')
        ax.set_ylabel('单次求解时间 (ms, 对数坐标)', fontsize=12)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'four_controller_benchmark_{timestamp}.png')
    plt.savefig(plot_path, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"对比图已保存: {plot_path}")
    plt.close()


if __name__ == '__main__':
    main()
