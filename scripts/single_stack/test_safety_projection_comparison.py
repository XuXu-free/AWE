"""
单槽系统安全投影对比测试脚本

对比：
1. 原始开环控制量（可能不安全）
2. 经过安全投影后的控制量

验证安全投影能否将不安全的控制量修正为安全控制量。
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from plant.single_stack_simulator import SingleStackSimulator
from controller.single_stack.safe_projection import SingleStackSafeProjection


def run_comparison_test(test_name, raw_actions, initial_state=None, duration=3600):
    """
    运行对比测试：原始控制量 vs 安全投影后的控制量

    Args:
        test_name: 测试名称
        raw_actions: 原始控制动作序列 [(t_start, t_end, [I, v_lye, v_c]), ...]
        initial_state: 初始状态
        duration: 测试时长
    """
    print(f"\n{'='*70}")
    print(f"测试: {test_name}")
    print(f"{'='*70}")

    # 初始化安全投影器
    projector = SingleStackSafeProjection(dt=60.0)

    # 创建两个仿真器（一个用原始控制，一个用投影后控制）
    sim_raw = SingleStackSimulator(sim_dt=0.2)
    sim_safe = SingleStackSimulator(sim_dt=0.2)
    sim_raw.reset(initial_state=initial_state)
    sim_safe.reset(initial_state=initial_state)

    dt = 0.2
    steps = int(duration / dt)
    ctrl_steps = int(60.0 / dt)  # 每60秒控制一次

    # 创建动作查找表
    action_table = {}
    for t_start, t_end, action in raw_actions:
        start_step = int(t_start / dt)
        end_step = int(t_end / dt)
        for s in range(start_step, min(end_step, steps)):
            action_table[s] = np.array(action)

    default_action = np.array([2000.0, 0.03, 0.0])

    # 记录数据
    history_raw = {'t': [], 'I': [], 'v_lye': [], 'v_c': [], 'T_s': [], 'T_sep': [],
                   'P_real': [], 'U_cell': [], 'HTO': [], 'violations': []}
    history_safe = {'t': [], 'I': [], 'v_lye': [], 'v_c': [], 'T_s': [], 'T_sep': [],
                    'P_real': [], 'U_cell': [], 'HTO': [], 'violations': []}
    history_proj_diff = {'t': [], 'dI': [], 'dv_lye': [], 'dv_c': [], 'norm_diff': []}

    # 安全边界
    T_min, T_max = 293.15, 363.15
    U_cell_max = 2.2
    HTO_max = 2.0
    P_stack_max = 6.0e6

    for i in range(steps):
        t = i * dt
        raw_action = action_table.get(i, default_action)

        # 每控制周期应用一次安全投影
        if i % ctrl_steps == 0:
            state_safe = sim_safe.state
            proj_state = np.array([state_safe[0], state_safe[1], state_safe[2], state_safe[3], state_safe[6]])
            safe_action, success = projector.project(raw_action, proj_state)

            # 记录投影差异
            if i % (ctrl_steps * 5) == 0:  # 每5个控制周期记录一次
                history_proj_diff['t'].append(t)
                history_proj_diff['dI'].append(safe_action[0] - raw_action[0])
                history_proj_diff['dv_lye'].append(safe_action[1] - raw_action[1])
                history_proj_diff['dv_c'].append(safe_action[2] - raw_action[2])
                history_proj_diff['norm_diff'].append(np.linalg.norm(safe_action - raw_action))

        # 记录数据（每10秒）
        if i % 50 == 0:
            # 原始控制
            state_r = sim_raw.state
            _, U_r, _ = sim_raw._calculate_electrochemical_properties(raw_action[0], state_r[1])
            P_r = U_r * raw_action[0] * sim_raw.N_cell
            hto_r = (state_r[6] * sim_raw.R * state_r[2]) / (sim_raw.P_sys * sim_raw.V_sep_gas) * 100

            history_raw['t'].append(t)
            history_raw['I'].append(raw_action[0])
            history_raw['v_lye'].append(raw_action[1])
            history_raw['v_c'].append(raw_action[2])
            history_raw['T_s'].append(state_r[1])
            history_raw['T_sep'].append(state_r[2])
            history_raw['P_real'].append(P_r)
            history_raw['U_cell'].append(U_r)
            history_raw['HTO'].append(hto_r)

            # 检查约束违反
            raw_viol = (state_r[1] > T_max or U_r > U_cell_max or
                       hto_r > HTO_max or P_r > P_stack_max)
            history_raw['violations'].append(raw_viol)

            # 安全控制
            state_s = sim_safe.state
            _, U_s, _ = sim_safe._calculate_electrochemical_properties(safe_action[0], state_s[1])
            P_s = U_s * safe_action[0] * sim_safe.N_cell
            hto_s = (state_s[6] * sim_safe.R * state_s[2]) / (sim_safe.P_sys * sim_safe.V_sep_gas) * 100

            history_safe['t'].append(t)
            history_safe['I'].append(safe_action[0])
            history_safe['v_lye'].append(safe_action[1])
            history_safe['v_c'].append(safe_action[2])
            history_safe['T_s'].append(state_s[1])
            history_safe['T_sep'].append(state_s[2])
            history_safe['P_real'].append(P_s)
            history_safe['U_cell'].append(U_s)
            history_safe['HTO'].append(hto_s)

            safe_viol = (state_s[1] > T_max or U_s > U_cell_max or
                        hto_s > HTO_max or P_s > P_stack_max)
            history_safe['violations'].append(safe_viol)

        # 执行仿真步
        sim_raw.step(raw_action)
        sim_safe.step(safe_action)

    # 分析结果
    analyze_comparison(history_raw, history_safe, test_name)

    return history_raw, history_safe, history_proj_diff


def analyze_comparison(history_raw, history_safe, test_name):
    """分析对比结果"""
    print(f"\n对比分析结果:")
    print("-" * 50)

    raw_violations = sum(history_raw['violations'])
    safe_violations = sum(history_safe['violations'])

    print(f"原始控制量:")
    print(f"  最大电流: {max(history_raw['I']):.1f} A")
    print(f"  最高温度: {max(history_raw['T_s']):.2f} K ({max(history_raw['T_s'])-273.15:.1f}C)")
    print(f"  最大电压: {max(history_raw['U_cell']):.3f} V")
    print(f"  最大HTO: {max(history_raw['HTO']):.4f}%")
    print(f"  最大功率: {max(history_raw['P_real'])/1e6:.2f} MW")
    print(f"  约束违反次数: {raw_violations}")

    print(f"\n安全投影后:")
    print(f"  最大电流: {max(history_safe['I']):.1f} A")
    print(f"  最高温度: {max(history_safe['T_s']):.2f} K ({max(history_safe['T_s'])-273.15:.1f}C)")
    print(f"  最大电压: {max(history_safe['U_cell']):.3f} V")
    print(f"  最大HTO: {max(history_safe['HTO']):.4f}%")
    print(f"  最大功率: {max(history_safe['P_real'])/1e6:.2f} MW")
    print(f"  约束违反次数: {safe_violations}")

    if raw_violations > 0 and safe_violations == 0:
        print(f"\n  [SUCCESS] 安全投影成功消除了所有约束违反!")
    elif raw_violations > safe_violations:
        print(f"\n  [PARTIAL] 安全投影减少了约束违反 ({raw_violations} -> {safe_violations})")
    elif safe_violations == 0:
        print(f"\n  [PASS] 原始控制和投影控制都满足约束")
    else:
        print(f"\n  [WARNING] 安全投影未能完全消除约束违反")


def plot_comparison(history_raw, history_safe, history_proj_diff, test_name, output_dir):
    """绘制对比图（原始 vs 安全投影）"""
    t_arr = np.array(history_raw['t'])

    fig = plt.figure(figsize=(16, 20))
    gs = fig.add_gridspec(5, 2, hspace=0.3, wspace=0.3)

    # 电流对比
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t_arr/60, history_raw['I'], 'r-', label='Raw', linewidth=2, alpha=0.7)
    ax1.plot(t_arr/60, history_safe['I'], 'g-', label='Safe Projection', linewidth=2)
    ax1.axhline(y=9360, color='r', linestyle='--', label='I_max')
    ax1.set_title('Current Comparison', fontsize=12, fontweight='bold')
    ax1.set_ylabel('A')
    ax1.legend()
    ax1.grid(True)

    # 投影差异
    ax2 = fig.add_subplot(gs[0, 1])
    if len(history_proj_diff['t']) > 0:
        ax2.plot(np.array(history_proj_diff['t'])/60, history_proj_diff['dI'], 'b-', label='dI', linewidth=2)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_title('Current Projection Difference (Safe - Raw)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('dI (A)')
    ax2.grid(True)

    # 碱液流速对比
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(t_arr/60, history_raw['v_lye'], 'r-', label='Raw', linewidth=2, alpha=0.7)
    ax3.plot(t_arr/60, history_safe['v_lye'], 'g-', label='Safe Projection', linewidth=2)
    ax3.axhline(y=0.1, color='r', linestyle='--', label='v_lye_max')
    ax3.set_title('Lye Flow Comparison', fontsize=12, fontweight='bold')
    ax3.set_ylabel('m3/s')
    ax3.legend()
    ax3.grid(True)

    # 冷却水对比
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(t_arr/60, history_raw['v_c'], 'r-', label='Raw', linewidth=2, alpha=0.7)
    ax4.plot(t_arr/60, history_safe['v_c'], 'g-', label='Safe Projection', linewidth=2)
    ax4.axhline(y=1.0, color='r', linestyle='--', label='v_c_max')
    ax4.set_title('Coolant Flow Comparison', fontsize=12, fontweight='bold')
    ax4.set_ylabel('m3/s')
    ax4.legend()
    ax4.grid(True)

    # 温度对比
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(t_arr/60, np.array(history_raw['T_s'])-273.15, 'r-', label='Raw', linewidth=2, alpha=0.7)
    ax5.plot(t_arr/60, np.array(history_safe['T_s'])-273.15, 'g-', label='Safe Projection', linewidth=2)
    ax5.axhline(y=90, color='r', linestyle='--', label='T_max (90C)')
    ax5.set_title('Stack Temperature Comparison', fontsize=12, fontweight='bold')
    ax5.set_ylabel('C')
    ax5.legend()
    ax5.grid(True)

    # 电压对比
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(t_arr/60, history_raw['U_cell'], 'r-', label='Raw', linewidth=2, alpha=0.7)
    ax6.plot(t_arr/60, history_safe['U_cell'], 'g-', label='Safe Projection', linewidth=2)
    ax6.axhline(y=2.2, color='r', linestyle='--', label='U_max')
    ax6.set_title('Cell Voltage Comparison', fontsize=12, fontweight='bold')
    ax6.set_ylabel('V')
    ax6.legend()
    ax6.grid(True)

    # HTO对比
    ax7 = fig.add_subplot(gs[3, 0])
    ax7.plot(t_arr/60, history_raw['HTO'], 'r-', label='Raw', linewidth=2, alpha=0.7)
    ax7.plot(t_arr/60, history_safe['HTO'], 'g-', label='Safe Projection', linewidth=2)
    ax7.axhline(y=2.0, color='r', linestyle='--', label='HTO_max')
    ax7.set_title('HTO Comparison', fontsize=12, fontweight='bold')
    ax7.set_ylabel('%')
    ax7.legend()
    ax7.grid(True)

    # 功率对比
    ax8 = fig.add_subplot(gs[3, 1])
    ax8.plot(t_arr/60, np.array(history_raw['P_real'])/1e6, 'r-', label='Raw', linewidth=2, alpha=0.7)
    ax8.plot(t_arr/60, np.array(history_safe['P_real'])/1e6, 'g-', label='Safe Projection', linewidth=2)
    ax8.axhline(y=6, color='r', linestyle='--', label='P_max')
    ax8.set_title('Stack Power Comparison', fontsize=12, fontweight='bold')
    ax8.set_ylabel('MW')
    ax8.legend()
    ax8.grid(True)

    # 约束违反对比
    ax9 = fig.add_subplot(gs[4, :])
    raw_viol = np.array(history_raw['violations'], dtype=int)
    safe_viol = np.array(history_safe['violations'], dtype=int)
    ax9.fill_between(t_arr/60, 0, raw_viol, alpha=0.3, color='red', label='Raw Violations')
    ax9.fill_between(t_arr/60, 0, safe_viol, alpha=0.5, color='green', label='Safe Violations')
    ax9.set_title('Constraint Violations Comparison', fontsize=12, fontweight='bold')
    ax9.set_ylabel('Violation (0/1)')
    ax9.set_xlabel('Time (min)')
    ax9.legend()
    ax9.grid(True)

    plt.suptitle(f'Safety Projection Comparison: {test_name}', fontsize=16, fontweight='bold', y=0.995)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"projection_comparison_{test_name}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"对比图已保存: {filepath}")
    return filepath


def main():
    """主函数：运行所有对比测试"""

    output_dir = os.path.join('output', 'single_stack', 'safety_projection_tests')
    os.makedirs(output_dir, exist_ok=True)

    # 测试1: 极端高电流
    print("\n" + "="*70)
    print("测试1: 极端高电流")
    print("="*70)
    history_raw, history_safe, history_diff = run_comparison_test(
        "extreme_high_current",
        raw_actions=[
            (0, 300, [9000, 0.01, 0.1]),
            (300, 3600, [2000, 0.03, 0.1]),
        ],
        duration=3600
    )
    plot_comparison(history_raw, history_safe, history_diff, "extreme_high_current", output_dir)

    # 测试2: 零碱液流速
    print("\n" + "="*70)
    print("测试2: 零碱液流速")
    print("="*70)
    history_raw, history_safe, history_diff = run_comparison_test(
        "zero_lye_flow",
        raw_actions=[
            (0, 600, [5000, 0.0, 0.5]),
            (600, 3600, [3000, 0.03, 0.5]),
        ],
        duration=3600
    )
    plot_comparison(history_raw, history_safe, history_diff, "zero_lye_flow", output_dir)

    # 测试3: 极高碱液流速
    print("\n" + "="*70)
    print("测试3: 极高碱液流速（控制量超限）")
    print("="*70)
    history_raw, history_safe, history_diff = run_comparison_test(
        "extreme_lye_flow",
        raw_actions=[
            (0, 600, [3000, 0.15, 0.3]),
            (600, 3600, [3000, 0.03, 0.3]),
        ],
        duration=3600
    )
    plot_comparison(history_raw, history_safe, history_diff, "extreme_lye_flow", output_dir)

    # 测试4: 组合极端工况
    print("\n" + "="*70)
    print("测试4: 组合极端工况")
    print("="*70)
    history_raw, history_safe, history_diff = run_comparison_test(
        "combined_extreme",
        raw_actions=[
            (0, 200, [9500, 0.0, 0.0]),
            (200, 600, [8000, 0.005, 0.05]),
            (600, 3600, [3000, 0.03, 0.3]),
        ],
        duration=3600
    )
    plot_comparison(history_raw, history_safe, history_diff, "combined_extreme", output_dir)

    print("\n" + "="*70)
    print("所有对比测试完成！")
    print(f"结果保存在: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
