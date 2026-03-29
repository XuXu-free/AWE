"""
单槽系统HTO约束专项测试脚本

验证在低功率/低电流情况下HTO约束的违反情况。

背景：在低功率运行时，气体产生速率降低，碱液流速降低，
可能导致分离器内H2积累，HTO升高。

测试场景：
1. 极低功率运行（气体排出不畅）
2. 间歇性低功率（气体积累后突然增加）
3. 低功率+低碱液流速（换热和分离都不足）
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


def run_hto_test(test_name, raw_actions, initial_state=None, duration=7200, use_projection=True):
    """
    运行HTO专项测试

    Args:
        test_name: 测试名称
        raw_actions: 原始控制动作序列 [(t_start, t_end, [I, v_lye, v_c]), ...]
        initial_state: 初始状态
        duration: 测试时长（秒）
        use_projection: 是否使用安全投影
    """
    print(f"\n{'='*70}")
    print(f"HTO测试: {test_name}")
    if use_projection:
        print(f"模式: 带安全投影")
    else:
        print(f"模式: 原始控制（无投影）")
    print(f"{'='*70}")

    sim = SingleStackSimulator(sim_dt=0.2)
    sim.reset(initial_state=initial_state)

    if use_projection:
        projector = SingleStackSafeProjection(dt=60.0)

    dt = 0.2
    steps = int(duration / dt)
    ctrl_steps = int(60.0 / dt)

    # 创建动作查找表
    action_table = {}
    for t_start, t_end, action in raw_actions:
        start_step = int(t_start / dt)
        end_step = int(t_end / dt)
        for s in range(start_step, min(end_step, steps)):
            action_table[s] = np.array(action)

    default_action = np.array([1000.0, 0.01, 0.0])

    # 记录数据
    history = {
        't': [], 'I': [], 'v_lye': [], 'v_c': [],
        'T_s': [], 'T_sep': [], 'T_c_out': [],
        'P_real': [], 'U_cell': [], 'HTO': [],
        'n_gas': [], 'n_H2_an': [],
        'HTO_violation': []
    }

    HTO_max = 2.0
    current_action = default_action.copy()

    for i in range(steps):
        t = i * dt
        raw_action = action_table.get(i, default_action)

        # 每控制周期应用一次安全投影
        if i % ctrl_steps == 0:
            if use_projection:
                state = sim.state
                proj_state = np.array([state[0], state[1], state[2], state[3], state[6]])
                current_action, success = projector.project(raw_action, proj_state)
            else:
                current_action = raw_action

        # 记录数据（每30秒）
        if i % 150 == 0:
            state = sim.state
            T_s_in, T_s, T_sep, T_c_out = state[0], state[1], state[2], state[3]
            n_H2_an, n_liq, n_gas = state[4], state[5], state[6]

            _, U_cell, _ = sim._calculate_electrochemical_properties(current_action[0], T_s)
            P_real = U_cell * current_action[0] * sim.N_cell
            hto_pct = (n_gas * sim.R * T_sep) / (sim.P_sys * sim.V_sep_gas) * 100

            history['t'].append(t)
            history['I'].append(current_action[0])
            history['v_lye'].append(current_action[1])
            history['v_c'].append(current_action[2])
            history['T_s'].append(T_s)
            history['T_sep'].append(T_sep)
            history['T_c_out'].append(T_c_out)
            history['P_real'].append(P_real)
            history['U_cell'].append(U_cell)
            history['HTO'].append(hto_pct)
            history['n_gas'].append(n_gas)
            history['n_H2_an'].append(n_H2_an)
            history['HTO_violation'].append(hto_pct > HTO_max)

        # 执行仿真步
        sim.step(current_action)

    # 分析结果
    max_hto = max(history['HTO'])
    hto_violations = sum(history['HTO_violation'])

    print(f"\nHTO分析结果:")
    print("-" * 50)
    print(f"  最大HTO: {max_hto:.4f}% (限制: 2.0%)")
    print(f"  HTO超标次数: {hto_violations}")
    print(f"  最终HTO: {history['HTO'][-1]:.4f}%")
    print(f"  最大n_gas: {max(history['n_gas']):.2f} mol")
    print(f"  最小功率: {min(history['P_real'])/1e6:.2f} MW")
    print(f"  最大功率: {max(history['P_real'])/1e6:.2f} MW")

    if hto_violations > 0:
        print(f"  [WARNING] HTO约束被违反!")
    else:
        print(f"  [PASS] HTO约束满足")

    return history


def plot_hto_comparison(history_raw, history_safe, test_name, output_dir):
    """绘制HTO对比图"""
    t_raw = np.array(history_raw['t'])
    t_safe = np.array(history_safe['t'])

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    fig.suptitle(f'HTO Constraint Test: {test_name}', fontsize=14, fontweight='bold')

    # 电流
    ax = axes[0, 0]
    ax.plot(t_raw/60, history_raw['I'], 'r-', label='Raw', linewidth=2, alpha=0.7)
    ax.plot(t_safe/60, history_safe['I'], 'g-', label='Safe Projection', linewidth=2)
    ax.set_title('Current')
    ax.set_ylabel('A')
    ax.legend()
    ax.grid(True)

    # HTO（重点）
    ax = axes[0, 1]
    ax.plot(t_raw/60, history_raw['HTO'], 'r-', label='Raw', linewidth=2, alpha=0.7)
    ax.plot(t_safe/60, history_safe['HTO'], 'g-', label='Safe Projection', linewidth=2)
    ax.axhline(y=2.0, color='r', linestyle='--', label='HTO_max', linewidth=2)
    ax.fill_between(t_raw/60, 2.0, max(max(history_raw['HTO']), 2.5), alpha=0.2, color='red')
    ax.set_title('HTO (H2 in O2) - KEY METRIC')
    ax.set_ylabel('%')
    ax.legend()
    ax.grid(True)

    # n_gas
    ax = axes[1, 0]
    ax.plot(t_raw/60, history_raw['n_gas'], 'r-', label='Raw', linewidth=2, alpha=0.7)
    ax.plot(t_safe/60, history_safe['n_gas'], 'g-', label='Safe Projection', linewidth=2)
    ax.set_title('n_gas (Separator Gas Phase)')
    ax.set_ylabel('mol')
    ax.legend()
    ax.grid(True)

    # n_H2_an
    ax = axes[1, 1]
    ax.plot(t_raw/60, history_raw['n_H2_an'], 'r-', label='Raw', linewidth=2, alpha=0.7)
    ax.plot(t_safe/60, history_safe['n_H2_an'], 'g-', label='Safe Projection', linewidth=2)
    ax.set_title('n_H2_an (Anode H2)')
    ax.set_ylabel('mol')
    ax.legend()
    ax.grid(True)

    # 功率
    ax = axes[2, 0]
    ax.plot(t_raw/60, np.array(history_raw['P_real'])/1e6, 'r-', label='Raw', linewidth=2, alpha=0.7)
    ax.plot(t_safe/60, np.array(history_safe['P_real'])/1e6, 'g-', label='Safe Projection', linewidth=2)
    ax.set_title('Power')
    ax.set_ylabel('MW')
    ax.legend()
    ax.grid(True)

    # 温度
    ax = axes[2, 1]
    ax.plot(t_raw/60, np.array(history_raw['T_s'])-273.15, 'r-', label='Raw', linewidth=2, alpha=0.7)
    ax.plot(t_safe/60, np.array(history_safe['T_s'])-273.15, 'g-', label='Safe Projection', linewidth=2)
    ax.axhline(y=90, color='r', linestyle='--', label='T_max')
    ax.set_title('Stack Temperature')
    ax.set_ylabel('C')
    ax.legend()
    ax.grid(True)

    # 碱液流速
    ax = axes[3, 0]
    ax.plot(t_raw/60, history_raw['v_lye'], 'r-', label='Raw', linewidth=2, alpha=0.7)
    ax.plot(t_safe/60, history_safe['v_lye'], 'g-', label='Safe Projection', linewidth=2)
    ax.axhline(y=0.1, color='r', linestyle='--', label='v_lye_max')
    ax.set_title('Lye Flow')
    ax.set_ylabel('m3/s')
    ax.set_xlabel('Time (min)')
    ax.legend()
    ax.grid(True)

    # HTO违反对比
    ax = axes[3, 1]
    raw_viol = np.array(history_raw['HTO_violation'], dtype=int)
    safe_viol = np.array(history_safe['HTO_violation'], dtype=int)
    ax.fill_between(t_raw/60, 0, raw_viol, alpha=0.3, color='red', label='Raw HTO Violations')
    ax.fill_between(t_safe/60, 0, safe_viol, alpha=0.5, color='green', label='Safe HTO Violations')
    ax.set_title('HTO Constraint Violations')
    ax.set_ylabel('Violation (0/1)')
    ax.set_xlabel('Time (min)')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"hto_test_{test_name}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"图表已保存: {filepath}")
    return filepath


def main():
    """主函数：运行所有HTO专项测试"""

    output_dir = os.path.join('output', 'single_stack', 'hto_tests')
    os.makedirs(output_dir, exist_ok=True)

    # 测试1: 极低功率持续运行（气体排出不畅）
    print("\n" + "="*70)
    print("测试1: 极低功率持续运行")
    print("="*70)
    history_raw = run_hto_test(
        "very_low_power",
        raw_actions=[
            (0, 7200, [500, 0.005, 0.0]),  # 极低电流，极低流速
        ],
        duration=7200,
        use_projection=False
    )
    history_safe = run_hto_test(
        "very_low_power",
        raw_actions=[
            (0, 7200, [500, 0.005, 0.0]),
        ],
        duration=7200,
        use_projection=True
    )
    plot_hto_comparison(history_raw, history_safe, "very_low_power", output_dir)

    # 测试2: 间歇性低功率（气体积累后突然增加）
    print("\n" + "="*70)
    print("测试2: 间歇性低功率")
    print("="*70)
    history_raw = run_hto_test(
        "intermittent_low_power",
        raw_actions=[
            (0, 1800, [800, 0.01, 0.0]),     # 低功率30分钟
            (1800, 3600, [500, 0.005, 0.0]),  # 极低功率30分钟（积累气体）
            (3600, 4000, [800, 0.01, 0.0]),  # 稍微增加（检验HTO峰值）
            (4000, 7200, [3000, 0.03, 0.2]), # 恢复正常
        ],
        duration=7200,
        use_projection=False
    )
    history_safe = run_hto_test(
        "intermittent_low_power",
        raw_actions=[
            (0, 1800, [800, 0.01, 0.0]),
            (1800, 3600, [500, 0.005, 0.0]),
            (3600, 4000, [800, 0.01, 0.0]),
            (4000, 7200, [3000, 0.03, 0.2]),
        ],
        duration=7200,
        use_projection=True
    )
    plot_hto_comparison(history_raw, history_safe, "intermittent_low_power", output_dir)

    # 测试3: 低功率+低碱液流速（双重风险）
    print("\n" + "="*70)
    print("测试3: 低功率+低碱液流速")
    print("="*70)
    history_raw = run_hto_test(
        "low_power_low_flow",
        raw_actions=[
            (0, 3600, [600, 0.002, 0.0]),     # 极低功率+极低流速
            (3600, 7200, [3000, 0.03, 0.2]),  # 恢复
        ],
        duration=7200,
        use_projection=False
    )
    history_safe = run_hto_test(
        "low_power_low_flow",
        raw_actions=[
            (0, 3600, [600, 0.002, 0.0]),
            (3600, 7200, [3000, 0.03, 0.2]),
        ],
        duration=7200,
        use_projection=True
    )
    plot_hto_comparison(history_raw, history_safe, "low_power_low_flow", output_dir)

    # 测试4: 功率波动（频繁启停）
    print("\n" + "="*70)
    print("测试4: 功率波动（频繁启停）")
    print("="*70)
    # 创建频繁波动的动作序列
    fluctuating_actions = []
    t = 0
    while t < 7200:
        fluctuating_actions.append((t, min(t+300, 7200), [400, 0.005, 0.0]))  # 低功率5分钟
        t += 300
        if t < 7200:
            fluctuating_actions.append((t, min(t+300, 7200), [2000, 0.02, 0.1]))  # 中等功率5分钟
            t += 300

    history_raw = run_hto_test(
        "power_fluctuation",
        raw_actions=fluctuating_actions,
        duration=7200,
        use_projection=False
    )
    history_safe = run_hto_test(
        "power_fluctuation",
        raw_actions=fluctuating_actions,
        duration=7200,
        use_projection=True
    )
    plot_hto_comparison(history_raw, history_safe, "power_fluctuation", output_dir)

    print("\n" + "="*70)
    print("所有HTO专项测试完成！")
    print(f"结果保存在: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
