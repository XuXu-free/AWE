# -*- coding: utf-8 -*-
"""
单槽NMPC (JIT) vs DiffusionTCN 控制策略生成时间对比测试
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import time
import json

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from plant.single_stack_simulator import SingleStackSimulator
from controller.single_stack.model_controller import SingleStackModelController
from controller.single_stack.nmpc_controller import SingleStackNMPCController

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 定义专业配色方案
COLORS = {
    'nmpc': '#e74c3c',        # 红色 - NMPC
    'diffusion': '#1f77b4',   # 蓝色 - Diffusion
    'grid': '#e0e0e0',        # 网格灰
    'text': '#2c3e50',        # 正文深灰
}


class SingleStackNMPCControllerJIT(SingleStackNMPCController):
    """强制使用纯CasADi求解器（不使用预编译C库，不使用JIT）"""

    def _setup_solver(self):
        """重写求解器设置，强制使用纯CasADi（无C编译）"""
        import casadi as ca

        self.n_controls = 3
        self.n_states = 7

        # Build CasADi symbolic optimization problem
        self.U = ca.MX.sym('U', self.n_controls * self.horizon)
        self.P = ca.MX.sym('P', self.n_states + 1 + self.horizon + 3)

        obj = 0
        g = []
        lbg = []
        ubg = []

        p_idx = 0
        T_s_in_K = self.P[p_idx]; p_idx += 1
        T_s_K = self.P[p_idx]; p_idx += 1
        T_sep_K = self.P[p_idx]; p_idx += 1
        T_c_out_K = self.P[p_idx]; p_idx += 1
        n_H2_an = self.P[p_idx]; p_idx += 1
        n_liq = self.P[p_idx]; p_idx += 1
        n_gas = self.P[p_idx]; p_idx += 1

        T_ref_val = self.P[p_idx]; p_idx += 1
        P_ref = self.P[p_idx:p_idx+self.horizon]; p_idx += self.horizon
        I_0 = self.P[p_idx]; p_idx += 1
        v_lye_0 = self.P[p_idx]; p_idx += 1
        v_c_0 = self.P[p_idx]; p_idx += 1

        T_s_in_k = T_s_in_K
        T_s_k = T_s_K
        T_sep_k = T_sep_K
        T_c_out_k = T_c_out_K

        for k in range(self.horizon):
            uk = self.U[k*self.n_controls : (k+1)*self.n_controls]
            I_k = uk[0]
            v_lye_k = uk[1]
            v_c_k = uk[2]

            T_s_in_k, T_s_k, T_sep_k, T_c_out_k, Power_k, V_cell, hto_pct = self._dynamics_step(
                T_s_in_k, T_s_k, T_sep_k, T_c_out_k, I_k, v_lye_k, v_c_k, n_gas
            )

            obj += self.lambda_track * ((Power_k - P_ref[k])/1e6)**2
            obj += self.lambda_temp * (T_s_k - T_ref_val)**2

            if k == 0:
                dI = I_k - I_0
            else:
                uk_prev = self.U[(k-1)*self.n_controls : k*self.n_controls]
                dI = I_k - uk_prev[0]

            dv_lye = v_lye_k - v_lye_0
            dv_c = v_c_k - v_c_0

            obj += self.lambda_I * dI**2
            obj += self.lambda_lye * dv_lye**2
            obj += self.lambda_c * dv_c**2

            g.append(T_s_k)
            lbg.append(self.T_min)
            ubg.append(self.T_max)

            g.append(Power_k)
            lbg.append(self.P_stack_min)
            ubg.append(self.P_stack_max)

            g.append(V_cell)
            lbg.append(self.U_cell_min)
            ubg.append(self.U_cell_max)

            g.append(hto_pct)
            lbg.append(self.HTO_pct_min)
            ubg.append(self.HTO_pct_max)

        lbx = []
        ubx = []
        for k in range(self.horizon):
            lbx.extend([self.I_min, self.v_lye_min, self.v_c_min])
            ubx.extend([self.I_max, self.v_lye_max, self.v_c_max])

        self.lbx = lbx
        self.ubx = ubx
        self.lbg = lbg
        self.ubg = ubg

        nlp = {'x': self.U, 'f': obj, 'g': ca.vertcat(*g), 'p': self.P}

        # 纯CasADi求解器，无JIT，无C编译
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.tol': 1e-4
        }

        print("Initializing NMPC with pure CasADi solver (no JIT, no C compilation)...")
        self.solver = ca.nlpsol('nmpc_solver', 'ipopt', nlp, opts)

        # Setup state prediction function
        pred_states = []
        p_idx = 0
        T_s_in_K = self.P[p_idx]; p_idx += 1
        T_s_K = self.P[p_idx]; p_idx += 1
        T_sep_K = self.P[p_idx]; p_idx += 1
        T_c_out_K = self.P[p_idx]; p_idx += 1
        n_H2_an_dummy = self.P[p_idx]; p_idx += 1
        n_liq_dummy = self.P[p_idx]; p_idx += 1
        n_gas_val = self.P[p_idx]; p_idx += 1

        T_s_in_k = T_s_in_K
        T_s_k = T_s_K
        T_sep_k = T_sep_K
        T_c_out_k = T_c_out_K

        for k in range(self.horizon):
            uk = self.U[k*self.n_controls : (k+1)*self.n_controls]
            I_k = uk[0]
            v_lye_k = uk[1]
            v_c_k = uk[2]

            T_s_in_k, T_s_k, T_sep_k, T_c_out_k, _, _, _ = self._dynamics_step(
                T_s_in_k, T_s_k, T_sep_k, T_c_out_k, I_k, v_lye_k, v_c_k, n_gas_val
            )
            pred_states.append(ca.vertcat(T_s_in_k, T_s_k, T_sep_k, T_c_out_k))

        self.state_func = ca.Function('state_func', [self.U, self.P], [ca.vertcat(*pred_states)])

        self.prev_sol_x = None


def run_test_with_timing(controller_type='nmpc', duration_seconds=3600, month='02'):
    """
    运行测试并记录每个控制步的计算时间

    Args:
        controller_type: 'nmpc' 或 'diffusion_tcn'
        duration_seconds: 测试持续时间（秒）
        month: 风电数据月份
    """
    print(f"\n{'='*60}")
    print(f"Running {controller_type.upper()} Test with Timing")
    print(f"Duration: {duration_seconds/60:.1f} minutes")
    print(f"{'='*60}")

    # 初始化
    sim_dt = 0.2
    ctrl_dt = 60.0
    T_ref = 353.15

    sim = SingleStackSimulator(sim_dt=sim_dt)

    if controller_type == 'nmpc':
        ctrl = SingleStackNMPCControllerJIT(dt=ctrl_dt, horizon=5, sim_dt=sim_dt)
    else:  # diffusion_tcn
        model_path = "output/single_stack/policy/diffusion_tcn_policy_best.pth"
        stats_path = "output/single_stack/diffusion_stats.npz"

        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            return None

        ctrl = SingleStackModelController(
            dt=ctrl_dt,
            horizon=5,
            model_type='diffusion_tcn',
            model_path=model_path,
            stats_path=stats_path
        )

    # 加载风电数据
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    profile_path = os.path.join(project_root, 'output', 'power', 'wind', f'wind_power_2025-{month}_1min.csv')
    df_power = pd.read_csv(profile_path)
    full_profile = df_power['P_ref'].values * 0.25

    # 准备测试
    state = sim.reset()
    last_action = [1000.0, 0.05, 0.5]

    # 简化预热
    print("Warmup...")
    warmup_steps = int(10 * 60 / ctrl_dt)  # 10分钟预热
    P_warmup = 2.0e6

    for _ in range(warmup_steps):
        P_ref_vec = np.full(ctrl.horizon if hasattr(ctrl, 'horizon') else 5, P_warmup)
        action = ctrl.get_action(state, P_ref_vec, T_ref, last_action)
        if action is not None:
            last_action = [action[0], action[1], action[2]]
        state = sim.step(last_action)

    # 正式测试
    print(f"Running test for {duration_seconds/60:.1f} minutes...")
    n_steps = int(duration_seconds / ctrl_dt)

    # 记录数据
    computation_times = []
    step_indices = []
    power_refs = []
    power_reals = []
    temperatures = []
    currents = []
    hto_values = []

    start_time_total = time.time()

    for step in range(n_steps):
        t = step * ctrl_dt
        idx = int(t / 60) % len(full_profile)

        # 构建参考功率向量
        horizon = ctrl.horizon if hasattr(ctrl, 'horizon') else 5
        P_ref_vec = np.array([full_profile[(idx + i) % len(full_profile)] for i in range(horizon)])

        # 记录计算时间
        start_compute = time.time()
        action = ctrl.get_action(state, P_ref_vec, T_ref, last_action)
        compute_time = time.time() - start_compute

        computation_times.append(compute_time * 1000)  # 转换为毫秒
        step_indices.append(step)
        power_refs.append(P_ref_vec[0])

        if action is not None:
            last_action = [action[0], action[1], action[2]]

        # 执行动作
        state = sim.step(last_action)

        # 记录状态
        power_reals.append(state[0])  # P_real
        temperatures.append(state[1])  # T_stack
        currents.append(last_action[0])
        hto_values.append(state[2])  # HTO

        if step % 10 == 0:
            print(f"Step {step}/{n_steps}, Computation time: {compute_time*1000:.2f} ms")

    total_time = time.time() - start_time_total

    # 计算统计信息
    times_array = np.array(computation_times)
    stats = {
        'controller': controller_type,
        'mean_time_ms': np.mean(times_array),
        'std_time_ms': np.std(times_array),
        'min_time_ms': np.min(times_array),
        'max_time_ms': np.max(times_array),
        'median_time_ms': np.median(times_array),
        'total_time_s': total_time,
        'n_steps': n_steps,
        'computation_times': computation_times,
        'step_indices': step_indices
    }

    print(f"\n{controller_type.upper()} Timing Results:")
    print(f"  Mean: {stats['mean_time_ms']:.2f} ms")
    print(f"  Std:  {stats['std_time_ms']:.2f} ms")
    print(f"  Min:  {stats['min_time_ms']:.2f} ms")
    print(f"  Max:  {stats['max_time_ms']:.2f} ms")
    print(f"  Total: {stats['total_time_s']:.2f} s")

    return stats


def plot_timing_comparison(nmpc_stats, diffusion_stats, output_dir):
    """绘制计算时间对比图"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('white')

    # 1. 时间序列对比
    ax = axes[0, 0]
    ax.plot(nmpc_stats['step_indices'], nmpc_stats['computation_times'],
            '-', color=COLORS['nmpc'], linewidth=1.5, alpha=0.7, label='NMPC (JIT)')
    ax.plot(diffusion_stats['step_indices'], diffusion_stats['computation_times'],
            '-', color=COLORS['diffusion'], linewidth=1.5, alpha=0.7, label='DiffusionTCN')
    ax.set_xlabel('Control Step', fontsize=12)
    ax.set_ylabel('Computation Time (ms)', fontsize=12)
    ax.set_title('Computation Time per Control Step', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.4)

    # 2. 箱线图对比
    ax = axes[0, 1]
    data = [nmpc_stats['computation_times'], diffusion_stats['computation_times']]
    labels = ['NMPC (JIT)', 'DiffusionTCN']
    colors = [COLORS['nmpc'], COLORS['diffusion']]

    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel('Computation Time (ms)', fontsize=12)
    ax.set_title('Computation Time Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4, axis='y')

    # 3. 直方图对比
    ax = axes[1, 0]
    ax.hist(nmpc_stats['computation_times'], bins=30, alpha=0.6,
            color=COLORS['nmpc'], label='NMPC (JIT)', edgecolor='white')
    ax.hist(diffusion_stats['computation_times'], bins=30, alpha=0.6,
            color=COLORS['diffusion'], label='DiffusionTCN', edgecolor='white')
    ax.set_xlabel('Computation Time (ms)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Computation Time Histogram', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.4)

    # 4. 统计对比表格
    ax = axes[1, 1]
    ax.axis('off')

    table_data = [
        ['Metric', 'NMPC (JIT)', 'DiffusionTCN', 'Speedup'],
        ['Mean (ms)', f"{nmpc_stats['mean_time_ms']:.2f}",
         f"{diffusion_stats['mean_time_ms']:.2f}",
         f"{nmpc_stats['mean_time_ms']/diffusion_stats['mean_time_ms']:.1f}x"],
        ['Std (ms)', f"{nmpc_stats['std_time_ms']:.2f}",
         f"{diffusion_stats['std_time_ms']:.2f}", '-'],
        ['Min (ms)', f"{nmpc_stats['min_time_ms']:.2f}",
         f"{diffusion_stats['min_time_ms']:.2f}", '-'],
        ['Max (ms)', f"{nmpc_stats['max_time_ms']:.2f}",
         f"{diffusion_stats['max_time_ms']:.2f}", '-'],
        ['Median (ms)', f"{nmpc_stats['median_time_ms']:.2f}",
         f"{diffusion_stats['median_time_ms']:.2f}", '-'],
    ]

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    # 设置表头样式
    for i in range(4):
        table[(0, i)].set_facecolor('#f0f0f0')
        table[(0, i)].set_text_props(weight='bold')

    ax.set_title('Computation Time Statistics', fontsize=14, fontweight='bold', pad=20)

    plt.suptitle('NMPC (JIT) vs DiffusionTCN Computation Time Comparison',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(pad=3.0)

    output_path = os.path.join(output_dir, 'timing_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """主函数：运行对比测试"""

    # 测试配置
    duration_seconds = 1800  # 30分钟测试
    month = '02'

    output_dir = 'output/single_stack/timing_comparison'
    os.makedirs(output_dir, exist_ok=True)

    print("="*60)
    print("Single-Stack Controller Timing Comparison")
    print("NMPC (JIT) vs DiffusionTCN")
    print("="*60)

    # 运行NMPC测试
    nmpc_stats = run_test_with_timing('nmpc', duration_seconds, month)

    # 运行DiffusionTCN测试
    diffusion_stats = run_test_with_timing('diffusion_tcn', duration_seconds, month)

    if nmpc_stats and diffusion_stats:
        # 绘制对比图
        print("\nGenerating comparison plots...")
        plot_timing_comparison(nmpc_stats, diffusion_stats, output_dir)

        # 保存统计信息
        stats_summary = {
            'nmpc_jit': {
                'mean_ms': nmpc_stats['mean_time_ms'],
                'std_ms': nmpc_stats['std_time_ms'],
                'min_ms': nmpc_stats['min_time_ms'],
                'max_ms': nmpc_stats['max_time_ms'],
                'median_ms': nmpc_stats['median_time_ms'],
            },
            'diffusion_tcn': {
                'mean_ms': diffusion_stats['mean_time_ms'],
                'std_ms': diffusion_stats['std_time_ms'],
                'min_ms': diffusion_stats['min_time_ms'],
                'max_ms': diffusion_stats['max_time_ms'],
                'median_ms': diffusion_stats['median_time_ms'],
            },
            'speedup': {
                'mean': nmpc_stats['mean_time_ms'] / diffusion_stats['mean_time_ms'],
                'median': nmpc_stats['median_time_ms'] / diffusion_stats['median_time_ms'],
            }
        }

        with open(os.path.join(output_dir, 'timing_stats.json'), 'w') as f:
            json.dump(stats_summary, f, indent=2)

        print(f"\n{'='*60}")
        print("Summary:")
        print(f"  NMPC (JIT) Mean Time: {nmpc_stats['mean_time_ms']:.2f} ms")
        print(f"  DiffusionTCN Mean Time: {diffusion_stats['mean_time_ms']:.2f} ms")
        print(f"  Speedup: {stats_summary['speedup']['mean']:.1f}x")
        print(f"\nResults saved to: {output_dir}/")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
