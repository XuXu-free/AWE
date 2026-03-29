"""
单槽CBF模型控制器测试脚本

演示如何使用SingleStackCBFModelController进行安全控制。
对比普通ModelController、SafeModelController和CBFModelController的性能。
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from plant.single_stack_simulator import SingleStackSimulator

# 导入控制器
try:
    from controller.single_stack.cbf_model_controller import SingleStackCBFModelController
    from controller.single_stack.safe_model_controller import SingleStackSafeModelController
    from controller.single_stack.model_controller import SingleStackModelController
except ImportError as e:
    print(f"导入控制器失败: {e}")
    sys.exit(1)


def run_controller_test(controller, controller_name, initial_state, P_ref_profile, duration=7200):
    """
    运行单个控制器测试

    Args:
        controller: 控制器实例
        controller_name: 控制器名称
        initial_state: 初始状态
        P_ref_profile: 功率参考序列 [(t_start, t_end, P_ref), ...]
        duration: 测试时长（秒）

    Returns:
        history: 测试历史数据
    """
    print(f"\n{'='*70}")
    print(f"测试控制器: {controller_name}")
    print(f"{'='*70}")

    sim = SingleStackSimulator(sim_dt=0.2)
    sim.reset(initial_state=initial_state)

    dt = 60.0  # 控制周期
    steps = int(duration / dt)

    # 创建功率参考查找表
    def get_p_ref(t):
        for t_start, t_end, p_ref in P_ref_profile:
            if t_start <= t < t_end:
                return p_ref
        return P_ref_profile[-1][2]  # 默认最后一个值

    # 记录数据
    history = {
        't': [], 'I': [], 'v_lye': [], 'v_c': [],
        'T_s': [], 'T_sep': [], 'T_c_out': [],
        'P_real': [], 'P_ref': [], 'U_cell': [],
        'HTO': [], 'n_gas': [],
        'HTO_violation': [], 'T_violation': []
    }

    T_max = 363.15  # 90°C
    HTO_max = 2.0   # 2%

    last_action = np.array([1000.0, 0.01, 0.1])

    for i in range(steps):
        t = i * dt
        P_ref = get_p_ref(t)
        T_ref = 353.15  # 80°C

        # 生成功率参考序列（未来5步）
        P_ref_seq = [get_p_ref(t + j * dt) for j in range(5)]

        # 获取控制动作
        state = sim.state
        try:
            if hasattr(controller, 'get_action_with_rollout'):
                I_cmd, v_lye_cmd, v_c_cmd = controller.get_action_with_rollout(
                    state, P_ref_seq, T_ref, last_action, num_candidates=64, verbose=False
                )
            else:
                I_cmd, v_lye_cmd, v_c_cmd = controller.get_action(
                    state, P_ref_seq, T_ref, last_action
                )
        except Exception as e:
            print(f"控制器出错 @ t={t}: {e}")
            I_cmd, v_lye_cmd, v_c_cmd = last_action

        action = np.array([I_cmd, v_lye_cmd, v_c_cmd])
        last_action = action.copy()

        # 记录数据
        T_s_in, T_s, T_sep, T_c_out = state[0], state[1], state[2], state[3]
        n_H2_an, n_liq, n_gas = state[4], state[5], state[6]

        _, U_cell, _ = sim._calculate_electrochemical_properties(I_cmd, T_s)
        P_real = U_cell * I_cmd * sim.N_cell
        hto_pct = (n_gas * sim.R * T_sep) / (sim.P_sys * sim.V_sep_gas) * 100

        history['t'].append(t)
        history['I'].append(I_cmd)
        history['v_lye'].append(v_lye_cmd)
        history['v_c'].append(v_c_cmd)
        history['T_s'].append(T_s)
        history['T_sep'].append(T_sep)
        history['T_c_out'].append(T_c_out)
        history['P_real'].append(P_real)
        history['P_ref'].append(P_ref)
        history['U_cell'].append(U_cell)
        history['HTO'].append(hto_pct)
        history['n_gas'].append(n_gas)
        history['HTO_violation'].append(hto_pct > HTO_max)
        history['T_violation'].append(T_s > T_max)

        # 仿真步进
        for _ in range(int(dt / sim.dt)):
            sim.step(action)

    # 分析结果
    max_hto = max(history['HTO'])
    max_T = max(history['T_s'])
    hto_violations = sum(history['HTO_violation'])
    T_violations = sum(history['T_violation'])

    print(f"\n测试结果摘要:")
    print("-" * 50)
    print(f"  最大HTO: {max_hto:.4f}% (限制: {HTO_max}%)")
    print(f"  最大温度: {max_T-273.15:.2f}°C (限制: {T_max-273.15}°C)")
    print(f"  HTO违反次数: {hto_violations}")
    print(f"  温度违反次数: {T_violations}")
    print(f"  平均功率跟踪误差: {np.mean([abs(p-r)/1e6 for p,r in zip(history['P_real'], history['P_ref'])])/1e6:.4f} MW")

    return history


def plot_comparison(histories, controller_names, test_name, output_dir):
    """绘制多个控制器的对比图"""
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    fig.suptitle(f'CBF Model Controller Test: {test_name}', fontsize=14, fontweight='bold')

    colors = ['blue', 'green', 'red', 'orange']

    # 功率跟踪
    ax = axes[0, 0]
    for i, (hist, name) in enumerate(zip(histories, controller_names)):
        t = np.array(hist['t']) / 60
        ax.plot(t, np.array(hist['P_ref'])/1e6, 'k--', alpha=0.5, label='Reference' if i==0 else '')
        ax.plot(t, np.array(hist['P_real'])/1e6, color=colors[i % len(colors)],
                label=name, linewidth=2, alpha=0.8)
    ax.set_title('Power Tracking')
    ax.set_ylabel('MW')
    ax.legend()
    ax.grid(True)

    # HTO对比
    ax = axes[0, 1]
    for i, (hist, name) in enumerate(zip(histories, controller_names)):
        t = np.array(hist['t']) / 60
        ax.plot(t, hist['HTO'], color=colors[i % len(colors)],
                label=name, linewidth=2, alpha=0.8)
    ax.axhline(y=2.0, color='r', linestyle='--', label='HTO_max', linewidth=2)
    ax.fill_between(t, 2.0, max(max(h['HTO']) for h in histories) * 1.1,
                    alpha=0.1, color='red')
    ax.set_title('HTO (H2 in O2) - Safety Metric')
    ax.set_ylabel('%')
    ax.legend()
    ax.grid(True)

    # 电流
    ax = axes[1, 0]
    for i, (hist, name) in enumerate(zip(histories, controller_names)):
        t = np.array(hist['t']) / 60
        ax.plot(t, hist['I'], color=colors[i % len(colors)],
                label=name, linewidth=2, alpha=0.8)
    ax.set_title('Current')
    ax.set_ylabel('A')
    ax.legend()
    ax.grid(True)

    # 温度
    ax = axes[1, 1]
    for i, (hist, name) in enumerate(zip(histories, controller_names)):
        t = np.array(hist['t']) / 60
        ax.plot(t, np.array(hist['T_s'])-273.15, color=colors[i % len(colors)],
                label=f'{name} (T_s)', linewidth=2, alpha=0.8)
    ax.axhline(y=90, color='r', linestyle='--', label='T_max')
    ax.set_title('Stack Temperature')
    ax.set_ylabel('°C')
    ax.legend()
    ax.grid(True)

    # 气相氢气量
    ax = axes[2, 0]
    for i, (hist, name) in enumerate(zip(histories, controller_names)):
        t = np.array(hist['t']) / 60
        ax.plot(t, hist['n_gas'], color=colors[i % len(colors)],
                label=name, linewidth=2, alpha=0.8)
    ax.set_title('Gas Phase H2 (n_gas)')
    ax.set_ylabel('mol')
    ax.legend()
    ax.grid(True)

    # 电压
    ax = axes[2, 1]
    for i, (hist, name) in enumerate(zip(histories, controller_names)):
        t = np.array(hist['t']) / 60
        ax.plot(t, hist['U_cell'], color=colors[i % len(colors)],
                label=name, linewidth=2, alpha=0.8)
    ax.axhline(y=2.2, color='r', linestyle='--', label='U_max')
    ax.set_title('Cell Voltage')
    ax.set_ylabel('V')
    ax.legend()
    ax.grid(True)

    # HTO违反
    ax = axes[3, 0]
    for i, (hist, name) in enumerate(zip(histories, controller_names)):
        t = np.array(hist['t']) / 60
        viol = np.array(hist['HTO_violation'], dtype=int)
        ax.fill_between(t, i*0.2, i*0.2 + viol*0.15, alpha=0.5,
                        color=colors[i % len(colors)], label=f'{name} violations')
    ax.set_title('HTO Constraint Violations')
    ax.set_ylabel('Controller')
    ax.set_xlabel('Time (min)')
    ax.set_yticks([0.075, 0.275, 0.475])
    ax.set_yticklabels(controller_names)
    ax.legend()
    ax.grid(True)

    # 碱液流速
    ax = axes[3, 1]
    for i, (hist, name) in enumerate(zip(histories, controller_names)):
        t = np.array(hist['t']) / 60
        ax.plot(t, np.array(hist['v_lye'])*1000, color=colors[i % len(colors)],
                label=name, linewidth=2, alpha=0.8)
    ax.set_title('Lye Flow Rate')
    ax.set_ylabel('L/s')
    ax.set_xlabel('Time (min)')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"cbf_model_test_{test_name}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"图表已保存: {filepath}")
    return filepath


def main():
    """主函数：运行CBF模型控制器测试"""

    output_dir = os.path.join('output', 'single_stack', 'cbf_model_tests')
    os.makedirs(output_dir, exist_ok=True)

    # 检查模型文件是否存在
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    model_path = os.path.join(base_dir, 'output', 'single_stack', 'policy', 'flow_tcn_policy_best.pth')

    if not os.path.exists(model_path):
        print(f"警告: 模型文件不存在: {model_path}")
        print("跳过模型控制器测试，仅演示CBF控制器结构")
        return

    # 初始化状态
    initial_state = np.array([
        345.0,  # T_s_in
        358.0,  # T_s
        358.0,  # T_sep
        325.0,  # T_c_out
        0.52,   # n_H2_an
        0.52,   # n_liq
        0.52    # n_gas
    ])

    # 测试场景：低功率波动（容易导致HTO超标）
    P_ref_profile = [
        (0, 1800, 1.5e6),    # 0-30min: 1.5 MW (低功率)
        (1800, 3600, 0.8e6), # 30-60min: 0.8 MW (极低功率，HTO风险)
        (3600, 5400, 2.0e6), # 60-90min: 2.0 MW (恢复)
        (5400, 7200, 1.0e6), # 90-120min: 1.0 MW (再次降低)
    ]

    print("\n" + "="*70)
    print("CBF模型控制器对比测试")
    print("="*70)
    print(f"输出目录: {output_dir}")

    # 测试1: 基础模型控制器（无安全投影）
    print("\n初始化: 基础模型控制器 (无安全投影)")
    try:
        controller_basic = SingleStackModelController(
            dt=60.0, horizon=5, model_type='flow_tcn'
        )
        history_basic = run_controller_test(
            controller_basic, "Model (No Safety)",
            initial_state, P_ref_profile, duration=7200
        )
    except Exception as e:
        print(f"基础模型控制器测试失败: {e}")
        history_basic = None

    # 测试2: 安全投影模型控制器
    print("\n初始化: 安全投影模型控制器")
    try:
        controller_safe = SingleStackSafeModelController(
            dt=60.0, horizon=5, model_type='flow_tcn',
            use_safe_projection=True
        )
        history_safe = run_controller_test(
            controller_safe, "Model + Safe Projection",
            initial_state, P_ref_profile, duration=7200
        )
    except Exception as e:
        print(f"安全投影控制器测试失败: {e}")
        history_safe = None

    # 测试3: CBF投影模型控制器
    print("\n初始化: CBF投影模型控制器")
    try:
        controller_cbf = SingleStackCBFModelController(
            dt=60.0, horizon=5, model_type='flow_tcn',
            use_cbf_projection=True, gamma=1.0
        )
        history_cbf = run_controller_test(
            controller_cbf, "Model + CBF Projection",
            initial_state, P_ref_profile, duration=7200
        )
    except Exception as e:
        print(f"CBF投影控制器测试失败: {e}")
        history_cbf = None

    # 绘制对比图
    histories = []
    names = []
    if history_basic:
        histories.append(history_basic)
        names.append("Basic")
    if history_safe:
        histories.append(history_safe)
        names.append("Safe")
    if history_cbf:
        histories.append(history_cbf)
        names.append("CBF")

    if histories:
        plot_comparison(histories, names, "low_power_comparison", output_dir)

    print("\n" + "="*70)
    print("CBF模型控制器测试完成！")
    print(f"结果保存在: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
