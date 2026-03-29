"""
多槽CBF模型控制器对比测试脚本

对比 MultiStackModelController、MultiStackSafeModelController 和 MultiStackCBFModelController
在低功率波动场景下的安全控制性能。
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from plant.multi_stack_simulator import MultiStackSimulator

# 导入控制器
try:
    from controller.multi_stack.cbf_model_controller import MultiStackCBFModelController
    from controller.multi_stack.safe_model_controller import MultiStackSafeModelController
    from controller.multi_stack.model_controller import MultiStackModelController
except ImportError as e:
    print(f"导入控制器失败: {e}")
    sys.exit(1)


def run_controller_test(controller, controller_name, initial_state, P_ref_profile, duration=1800):
    """
    运行单个控制器测试

    Args:
        controller: 控制器实例
        controller_name: 控制器名称
        initial_state: 初始状态 (13-dim)
        P_ref_profile: 功率参考序列 [(t_start, t_end, P_ref), ...]
        duration: 测试时长（秒）

    Returns:
        history: 测试历史数据
    """
    print(f"\n{'='*70}")
    print(f"测试控制器: {controller_name}")
    print(f"{'='*70}")

    sim = MultiStackSimulator(dt=0.2)
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
        't': [], 'I_all': [], 'v_lye_all': [], 'v_c': [],
        'T_s_all': [], 'T_sep': [], 'T_c_out': [],
        'P_real': [], 'P_ref': [], 'U_cell_all': [],
        'HTO': [], 'n_gas': [],
        'HTO_violation': [], 'T_violation': []
    }

    T_max = 363.15  # 90°C
    HTO_max = 2.0   # 2%

    last_action = [
        np.ones(4) * 1000.0,   # I
        np.ones(4) * 0.02,     # v_lye
        0.1                      # v_c
    ]

    for i in range(steps):
        if i % 5 == 0:
            print(f"  {controller_name} progress: {i}/{steps} steps (t={i*dt:.0f}s)")
        t = i * dt
        P_ref = get_p_ref(t)
        T_ref = 353.15  # 80°C

        # 生成功率参考序列（未来5步）
        P_ref_seq = [get_p_ref(t + j * dt) for j in range(5)]

        # 获取控制动作
        state = sim.state
        try:
            # 使用 get_action 进行快速对比（CBF/Safe仅采样1个候选并投影）
            I_cmd, v_lye_cmd, v_c_cmd = controller.get_action(
                state, P_ref_seq, T_ref, last_action
            )
        except Exception as e:
            print(f"控制器出错 @ t={t}: {e}")
            I_cmd, v_lye_cmd, v_c_cmd = last_action

        action = np.concatenate([I_cmd, v_lye_cmd, [v_c_cmd]])
        last_action = [I_cmd, v_lye_cmd, v_c_cmd]

        # 记录数据
        T_s_in = state[0]
        T_s_vec = state[1:5]
        T_sep = state[5]
        T_c_out = state[6]
        n_H2_an_vec = state[7:11]
        n_liq = state[11]
        n_gas = state[12]

        _, U_cell_vec, _ = sim._calculate_electrochemical_properties(I_cmd, T_s_vec)
        P_real = np.sum(U_cell_vec * I_cmd * sim.N_cell)
        hto_pct = (n_gas * sim.R * T_sep) / (sim.P_sys * sim.V_sep_gas) * 100

        history['t'].append(t)
        history['I_all'].append(I_cmd.copy())
        history['v_lye_all'].append(v_lye_cmd.copy())
        history['v_c'].append(v_c_cmd)
        history['T_s_all'].append(T_s_vec.copy())
        history['T_sep'].append(T_sep)
        history['T_c_out'].append(T_c_out)
        history['P_real'].append(P_real)
        history['P_ref'].append(P_ref)
        history['U_cell_all'].append(U_cell_vec.copy())
        history['HTO'].append(hto_pct)
        history['n_gas'].append(n_gas)
        history['HTO_violation'].append(hto_pct > HTO_max)
        history['T_violation'].append(np.any(T_s_vec > T_max))

        # 仿真步进
        for _ in range(int(dt / sim.dt)):
            sim.step(action)

    # 分析结果
    hto_array = np.array(history['HTO'])
    T_s_array = np.array(history['T_s_all'])
    max_hto = np.max(hto_array)
    max_T = np.max(T_s_array)
    hto_violations = np.sum(hto_array > HTO_max)
    T_violations = np.sum(np.any(T_s_array > T_max, axis=1))

    print(f"\n测试结果摘要:")
    print("-" * 50)
    print(f"  最大HTO: {max_hto:.4f}% (限制: {HTO_max}%)")
    print(f"  最大温度: {max_T-273.15:.2f}°C (限制: {T_max-273.15}°C)")
    print(f"  HTO违反次数: {hto_violations}")
    print(f"  温度违反次数: {T_violations}")
    mean_err = np.mean([abs(p - r) / 1e6 for p, r in zip(history['P_real'], history['P_ref'])])
    print(f"  平均功率跟踪误差: {mean_err:.4f} MW")

    return history


def plot_comparison(histories, controller_names, test_name, output_dir):
    """绘制多个控制器的对比图"""
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    fig.suptitle(f'Multi-Stack CBF Model Controller Test: {test_name}', fontsize=14, fontweight='bold')

    colors = ['blue', 'green', 'red', 'orange']
    stack_colors = ['C0', 'C1', 'C2', 'C3']

    # 功率跟踪
    ax = axes[0, 0]
    for i, (hist, name) in enumerate(zip(histories, controller_names)):
        t = np.array(hist['t']) / 60
        ax.plot(t, np.array(hist['P_ref']) / 1e6, 'k--', alpha=0.5, label='Reference' if i == 0 else '')
        ax.plot(t, np.array(hist['P_real']) / 1e6, color=colors[i % len(colors)],
                label=name, linewidth=2, alpha=0.8)
    ax.set_title('Total Power Tracking')
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

    # 各槽电流（仅绘制第一个控制器以节省空间，或绘制平均值+标准差）
    ax = axes[1, 0]
    for i, (hist, name) in enumerate(zip(histories, controller_names)):
        t = np.array(hist['t']) / 60
        I_all = np.array(hist['I_all'])
        I_mean = np.mean(I_all, axis=1)
        ax.plot(t, I_mean, color=colors[i % len(colors)],
                label=f'{name} (mean)', linewidth=2, alpha=0.8)
    ax.set_title('Mean Stack Current')
    ax.set_ylabel('A')
    ax.legend()
    ax.grid(True)

    # 各槽温度
    ax = axes[1, 1]
    for i, (hist, name) in enumerate(zip(histories, controller_names)):
        t = np.array(hist['t']) / 60
        T_s_all = np.array(hist['T_s_all'])
        T_mean = np.mean(T_s_all, axis=1) - 273.15
        ax.plot(t, T_mean, color=colors[i % len(colors)],
                label=f'{name} (mean T_s)', linewidth=2, alpha=0.8)
    ax.axhline(y=90, color='r', linestyle='--', label='T_max')
    ax.set_title('Mean Stack Temperature')
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

    # 平均电压
    ax = axes[2, 1]
    for i, (hist, name) in enumerate(zip(histories, controller_names)):
        t = np.array(hist['t']) / 60
        U_all = np.array(hist['U_cell_all'])
        U_mean = np.mean(U_all, axis=1)
        ax.plot(t, U_mean, color=colors[i % len(colors)],
                label=name, linewidth=2, alpha=0.8)
    ax.axhline(y=2.2, color='r', linestyle='--', label='U_max')
    ax.set_title('Mean Cell Voltage')
    ax.set_ylabel('V')
    ax.legend()
    ax.grid(True)

    # HTO违反
    ax = axes[3, 0]
    for i, (hist, name) in enumerate(zip(histories, controller_names)):
        t = np.array(hist['t']) / 60
        viol = np.array(hist['HTO_violation'], dtype=int)
        ax.fill_between(t, i * 0.2, i * 0.2 + viol * 0.15, alpha=0.5,
                        color=colors[i % len(colors)], label=f'{name} violations')
    ax.set_title('HTO Constraint Violations')
    ax.set_ylabel('Controller')
    ax.set_xlabel('Time (min)')
    ax.set_yticks([0.075, 0.275, 0.475])
    ax.set_yticklabels(controller_names)
    ax.legend()
    ax.grid(True)

    # 平均碱液流速
    ax = axes[3, 1]
    for i, (hist, name) in enumerate(zip(histories, controller_names)):
        t = np.array(hist['t']) / 60
        v_lye_all = np.array(hist['v_lye_all'])
        v_mean = np.mean(v_lye_all, axis=1) * 1000
        ax.plot(t, v_mean, color=colors[i % len(colors)],
                label=name, linewidth=2, alpha=0.8)
    ax.set_title('Mean Lye Flow Rate')
    ax.set_ylabel('L/s')
    ax.set_xlabel('Time (min)')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"multi_stack_cbf_model_test_{test_name}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"图表已保存: {filepath}")
    return filepath


def main():
    """主函数：运行多槽CBF模型控制器对比测试"""

    output_dir = os.path.join('output', 'multi_stack', 'cbf_model_tests')
    os.makedirs(output_dir, exist_ok=True)

    # 检查模型文件是否存在
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    model_path = os.path.join(base_dir, 'output', 'multi_stack', 'policy', 'flow_tcn_policy_best.pth')
    stats_path = os.path.join(base_dir, 'output', 'multi_stack', 'diffusion_stats.npz')

    if not os.path.exists(model_path):
        print(f"警告: 模型文件不存在: {model_path}")
        print("跳过模型控制器测试，仅演示CBF控制器结构")
        return

    # 使用仿真器reset获取一致的正确初始状态
    sim_temp = MultiStackSimulator(dt=0.2)
    initial_state = sim_temp.reset()
    print(f"初始状态 n_gas={initial_state[12]:.3f}, n_liq={initial_state[11]:.3f}, n_H2_an_mean={np.mean(initial_state[7:11]):.3f}")

    # 测试场景：低功率波动（容易导致HTO超标）
    P_ref_profile = [
        (0, 1800, 6.0e6),    # 0-30min: 6.0 MW
        (1800, 3600, 3.0e6), # 30-60min: 3.0 MW (低功率，HTO风险)
        (3600, 5400, 8.0e6), # 60-90min: 8.0 MW (恢复)
        (5400, 7200, 2.0e6), # 90-120min: 2.0 MW (再次降低)
    ]

    print("\n" + "="*70)
    print("多槽CBF模型控制器对比测试")
    print("="*70)
    print(f"输出目录: {output_dir}")

    # 测试1: 基础模型控制器（无安全投影）
    print("\n初始化: 基础模型控制器 (无安全投影)")
    try:
        controller_basic = MultiStackModelController(
            dt=60.0, horizon=5, model_type='flow_tcn',
            model_path=model_path, stats_path=stats_path
        )
        history_basic = run_controller_test(
            controller_basic, "Model (No Safety)",
            initial_state, P_ref_profile, duration=1800
        )
    except Exception as e:
        print(f"基础模型控制器测试失败: {e}")
        history_basic = None

    # 测试2: 安全投影模型控制器
    print("\n初始化: 安全投影模型控制器")
    try:
        controller_safe = MultiStackSafeModelController(
            dt=60.0, horizon=5, model_type='flow_tcn',
            model_path=model_path, stats_path=stats_path,
            use_safe_projection=True
        )
        history_safe = run_controller_test(
            controller_safe, "Model + Safe Projection",
            initial_state, P_ref_profile, duration=1800
        )
    except Exception as e:
        print(f"安全投影控制器测试失败: {e}")
        history_safe = None

    # 测试3: CBF投影模型控制器
    print("\n初始化: CBF投影模型控制器")
    try:
        # 使用调优后的推荐参数: gamma=1.0, rho=5000, HTO margin=0.005
        gamma_vec = [10.0] * 4 + [1.0] + [10.0] * 4 + [20.0] * 4 + [10.0] * 4
        rho_vec = [5000] * 17
        h_margin_vec = [0.0] * 4 + [0.005] + [0.0] * 12
        controller_cbf = MultiStackCBFModelController(
            dt=60.0, horizon=5, model_type='flow_tcn',
            model_path=model_path, stats_path=stats_path,
            use_cbf_projection=True, gamma=1.0,
            gamma_vec=gamma_vec, rho_vec=rho_vec,
            h_margin_vec=h_margin_vec,
            lambda_u_scale=1000.0,
            soft_mask=[True]*17
        )
        history_cbf = run_controller_test(
            controller_cbf, "Model + CBF Projection",
            initial_state, P_ref_profile, duration=1800
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
    print("多槽CBF模型控制器测试完成！")
    print(f"结果保存在: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
