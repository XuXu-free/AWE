"""
单槽系统安全约束开环测试脚本

直接输入极端控制量，测试系统在不安全工况下的行为，
验证安全约束是否会被违反。

测试场景：
1. 控制量超出边界（电流过大、碱液流速过高/过低）
2. 控制量导致状态量超标（温度过高、HTO过高、电压过高）
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


def run_open_loop_test(test_name, actions, initial_state=None, duration=3600):
    """
    运行开环测试

    Args:
        test_name: 测试名称
        actions: 控制动作序列 [(t_start, t_end, [I, v_lye, v_c]), ...]
        initial_state: 初始状态（可选）
        duration: 测试时长（秒）
    """
    sim = SingleStackSimulator(sim_dt=0.2)
    sim.reset(initial_state=initial_state)

    dt = 0.2
    steps = int(duration / dt)

    # 创建动作查找表
    action_table = {}
    for t_start, t_end, action in actions:
        start_step = int(t_start / dt)
        end_step = int(t_end / dt)
        for s in range(start_step, min(end_step, steps)):
            action_table[s] = np.array(action)

    # 默认动作
    default_action = np.array([2000.0, 0.03, 0.0])

    # 记录数据
    history = {
        't': [], 'I': [], 'v_lye': [], 'v_c': [],
        'T_s': [], 'T_sep': [], 'T_c_out': [],
        'P_real': [], 'U_cell': [], 'HTO': [],
        'I_violation': [], 'v_lye_violation': [], 'v_c_violation': [],
        'T_violation': [], 'U_violation': [], 'HTO_violation': [], 'P_violation': []
    }

    # 安全边界
    I_min, I_max = 0.0, 7800.0 * 1.2
    v_lye_min, v_lye_max = 0.0, 0.1
    v_c_min, v_c_max = 0.0, 1.0
    T_min, T_max = 293.15, 363.15
    U_cell_max = 2.2
    HTO_max = 2.0
    P_stack_max = 6.0e6

    print(f"\n{'='*60}")
    print(f"运行测试: {test_name}")
    print(f"{'='*60}")

    for i in range(steps):
        t = i * dt
        action = action_table.get(i, default_action)

        # 记录当前状态
        state = sim.state
        T_s_in, T_s, T_sep, T_c_out = state[0], state[1], state[2], state[3]
        n_H2_an, n_liq, n_gas = state[4], state[5], state[6]

        # 计算电气特性
        _, U_cell, eta = sim._calculate_electrochemical_properties(action[0], T_s)
        P_real = U_cell * action[0] * sim.N_cell

        # 计算HTO
        hto_pct = (n_gas * sim.R * T_sep) / (sim.P_sys * sim.V_sep_gas) * 100

        # 检查约束违反
        I_viol = action[0] < I_min or action[0] > I_max
        v_lye_viol = action[1] < v_lye_min or action[1] > v_lye_max
        v_c_viol = action[2] < v_c_min or action[2] > v_c_max
        T_viol = T_s < T_min or T_s > T_max
        U_viol = U_cell > U_cell_max
        HTO_viol = hto_pct > HTO_max
        P_viol = P_real > P_stack_max

        if i % 50 == 0:  # 每10秒记录一次
            history['t'].append(t)
            history['I'].append(action[0])
            history['v_lye'].append(action[1])
            history['v_c'].append(action[2])
            history['T_s'].append(T_s)
            history['T_sep'].append(T_sep)
            history['T_c_out'].append(T_c_out)
            history['P_real'].append(P_real)
            history['U_cell'].append(U_cell)
            history['HTO'].append(hto_pct)
            history['I_violation'].append(I_viol)
            history['v_lye_violation'].append(v_lye_viol)
            history['v_c_violation'].append(v_c_viol)
            history['T_violation'].append(T_viol)
            history['U_violation'].append(U_viol)
            history['HTO_violation'].append(HTO_viol)
            history['P_violation'].append(P_viol)

        # 执行仿真步
        sim.step(action)

    # 分析结果
    results = analyze_violations(history, test_name)

    return history, results


def analyze_violations(history, test_name):
    """分析约束违反情况"""
    print(f"\n约束违反分析:")
    print("-" * 40)

    violations = {
        'I': sum(history['I_violation']),
        'v_lye': sum(history['v_lye_violation']),
        'v_c': sum(history['v_c_violation']),
        'T': sum(history['T_violation']),
        'U': sum(history['U_violation']),
        'HTO': sum(history['HTO_violation']),
        'P': sum(history['P_violation'])
    }

    max_values = {
        'I': max(history['I']),
        'v_lye': max(history['v_lye']),
        'v_c': max(history['v_c']),
        'T': max(history['T_s']),
        'U': max(history['U_cell']),
        'HTO': max(history['HTO']),
        'P': max(history['P_real']) / 1e6
    }

    min_values = {
        'I': min(history['I']),
        'T': min(history['T_s']),
    }

    any_violation = False
    for var, count in violations.items():
        if count > 0:
            print(f"  [VIOLATION] {var}: {count} 次违反")
            any_violation = True

    print(f"\n最大值记录:")
    print(f"  电流: {max_values['I']:.1f} A (限制: 9360 A)")
    print(f"  碱液流速: {max_values['v_lye']:.4f} m3/s (限制: 0.1)")
    print(f"  冷却水流速: {max_values['v_c']:.3f} m3/s (限制: 1.0)")
    print(f"  温度: {max_values['T']:.2f} K ({max_values['T']-273.15:.1f}°C, 限制: 90°C)")
    print(f"  电压: {max_values['U']:.3f} V (限制: 2.2 V)")
    print(f"  HTO: {max_values['HTO']:.4f}% (限制: 2.0%)")
    print(f"  功率: {max_values['P']:.2f} MW (限制: 6 MW)")

    if not any_violation:
        print("  [PASS] 无约束违反")

    return {
        'test_name': test_name,
        'violations': violations,
        'max_values': max_values,
        'min_values': min_values,
        'any_violation': any_violation
    }


def plot_results(history, test_name, output_dir):
    """绘制测试结果"""
    t_arr = np.array(history['t'])

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    fig.suptitle(f'Safety Constraint Test: {test_name}', fontsize=14)

    # 控制量
    ax = axes[0, 0]
    ax.plot(t_arr/60, history['I'], 'b-', label='I')
    ax.axhline(y=9360, color='r', linestyle='--', label='I_max')
    ax.set_title('Current')
    ax.set_ylabel('A')
    ax.legend()
    ax.grid(True)

    ax = axes[0, 1]
    ax.plot(t_arr/60, history['v_lye'], 'g-', label='v_lye')
    ax.axhline(y=0.1, color='r', linestyle='--', label='v_lye_max')
    ax.set_title('Lye Flow')
    ax.set_ylabel('m3/s')
    ax.legend()
    ax.grid(True)

    # 温度
    ax = axes[1, 0]
    ax.plot(t_arr/60, np.array(history['T_s'])-273.15, 'r-', label='T_s')
    ax.axhline(y=90, color='r', linestyle='--', label='T_max')
    ax.axhline(y=20, color='b', linestyle='--', label='T_min')
    ax.set_title('Stack Temperature')
    ax.set_ylabel('°C')
    ax.legend()
    ax.grid(True)

    # 电压
    ax = axes[1, 1]
    ax.plot(t_arr/60, history['U_cell'], 'm-', label='U_cell')
    ax.axhline(y=2.2, color='r', linestyle='--', label='U_max')
    ax.set_title('Cell Voltage')
    ax.set_ylabel('V')
    ax.legend()
    ax.grid(True)

    # HTO
    ax = axes[2, 0]
    ax.plot(t_arr/60, history['HTO'], 'orange', label='HTO')
    ax.axhline(y=2.0, color='r', linestyle='--', label='HTO_max')
    ax.set_title('HTO (H2 in O2)')
    ax.set_ylabel('%')
    ax.legend()
    ax.grid(True)

    # 功率
    ax = axes[2, 1]
    ax.plot(t_arr/60, np.array(history['P_real'])/1e6, 'purple', label='P_real')
    ax.axhline(y=6, color='r', linestyle='--', label='P_max')
    ax.set_title('Stack Power')
    ax.set_ylabel('MW')
    ax.legend()
    ax.grid(True)

    # 约束违反指示
    ax = axes[3, 0]
    violation_sum = (
        np.array(history['I_violation']) +
        np.array(history['v_lye_violation']) +
        np.array(history['T_violation']) +
        np.array(history['U_violation']) +
        np.array(history['HTO_violation'])
    )
    ax.fill_between(t_arr/60, 0, violation_sum, alpha=0.3, color='red')
    ax.set_title('Constraint Violations (sum)')
    ax.set_ylabel('Count')
    ax.set_xlabel('Time (min)')
    ax.grid(True)

    # 冷却水
    ax = axes[3, 1]
    ax.plot(t_arr/60, history['v_c'], 'c-', label='v_c')
    ax.axhline(y=1.0, color='r', linestyle='--', label='v_c_max')
    ax.set_title('Coolant Flow')
    ax.set_ylabel('m3/s')
    ax.set_xlabel('Time (min)')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"safety_test_{test_name}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()

    print(f"图表已保存: {filepath}")
    return filepath


def main():
    """主函数：运行所有安全测试"""

    output_dir = os.path.join('output', 'single_stack', 'safety_tests')
    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    # 测试1: 电流过大（导致过热）
    print("\n" + "="*60)
    print("测试1: 极端高电流（过热测试）")
    print("="*60)
    history, results = run_open_loop_test(
        "extreme_high_current",
        actions=[
            (0, 300, [9000, 0.01, 0.1]),      # 前5分钟：超大电流
            (300, 3600, [2000, 0.03, 0.1]),   # 后续：正常电流
        ],
        duration=3600
    )
    all_results.append(results)
    plot_results(history, "extreme_high_current", output_dir)

    # 测试2: 零碱液流速（导致过热）
    print("\n" + "="*60)
    print("测试2: 零碱液流速（换热不足）")
    print("="*60)
    history, results = run_open_loop_test(
        "zero_lye_flow",
        actions=[
            (0, 600, [5000, 0.0, 0.5]),       # 前10分钟：零碱液流速
            (600, 3600, [3000, 0.03, 0.5]),   # 后续：正常流速
        ],
        duration=3600
    )
    all_results.append(results)
    plot_results(history, "zero_lye_flow", output_dir)

    # 测试3: 高功率持续运行（HTO累积）
    print("\n" + "="*60)
    print("测试3: 持续高功率（HTO累积测试）")
    print("="*60)
    history, results = run_open_loop_test(
        "sustained_high_power",
        actions=[
            (0, 3600, [6500, 0.02, 0.3]),     # 全程高电流
        ],
        duration=3600
    )
    all_results.append(results)
    plot_results(history, "sustained_high_power", output_dir)

    # 测试4: 极高碱液流速
    print("\n" + "="*60)
    print("测试4: 极高碱液流速（控制量超限）")
    print("="*60)
    history, results = run_open_loop_test(
        "extreme_lye_flow",
        actions=[
            (0, 600, [3000, 0.15, 0.3]),      # 碱液流速超过0.1限制
            (600, 3600, [3000, 0.03, 0.3]),
        ],
        duration=3600
    )
    all_results.append(results)
    plot_results(history, "extreme_lye_flow", output_dir)

    # 测试5: 组合极端工况
    print("\n" + "="*60)
    print("测试5: 组合极端工况（多约束同时违反）")
    print("="*60)
    history, results = run_open_loop_test(
        "combined_extreme",
        actions=[
            (0, 200, [9500, 0.0, 0.0]),       # 超大电流+零流速
            (200, 600, [8000, 0.005, 0.05]),  # 仍然很危险
            (600, 3600, [3000, 0.03, 0.3]),   # 恢复正常
        ],
        duration=3600
    )
    all_results.append(results)
    plot_results(history, "combined_extreme", output_dir)

    # 汇总报告
    print("\n" + "="*60)
    print("安全测试汇总报告")
    print("="*60)

    for r in all_results:
        print(f"\n{r['test_name']}:")
        if r['any_violation']:
            print("  状态: [UNSAFE] 存在约束违反")
            for var, count in r['violations'].items():
                if count > 0:
                    print(f"    - {var}: {count}次")
        else:
            print("  状态: [SAFE] 无约束违反")

    print(f"\n所有测试结果保存在: {output_dir}")


if __name__ == "__main__":
    main()
