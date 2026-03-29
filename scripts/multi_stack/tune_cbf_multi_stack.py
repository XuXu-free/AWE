"""
多槽CBF参数调优脚本

快速扫描不同 gamma / rho / margin 组合，找到能使 HTO < 2% 的参数甜区。
"""

import os
import sys
import numpy as np
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from plant.multi_stack_simulator import MultiStackSimulator
from controller.multi_stack.cbf_model_controller import MultiStackCBFModelController


def run_single_test(controller, initial_state, P_ref_profile, duration=900):
    """运行单个CBF测试，返回关键指标"""
    sim = MultiStackSimulator(dt=0.2)
    sim.reset(initial_state=initial_state)
    dt = 60.0
    steps = int(duration / dt)

    def get_p_ref(t):
        for t_start, t_end, p_ref in P_ref_profile:
            if t_start <= t < t_end:
                return p_ref
        return P_ref_profile[-1][2]

    last_action = [np.ones(4)*1000.0, np.ones(4)*0.02, 0.1]
    max_hto = 0.0
    max_T = 0.0
    hto_violations = 0
    T_violations = 0
    fail_count = 0
    power_errors = []

    for i in range(steps):
        t = i * dt
        P_ref = get_p_ref(t)
        T_ref = 353.15
        P_ref_seq = [get_p_ref(t + j * dt) for j in range(5)]
        state = sim.state

        try:
            I_cmd, v_lye_cmd, v_c_cmd = controller.get_action(state, P_ref_seq, T_ref, last_action)
        except Exception:
            I_cmd, v_lye_cmd, v_c_cmd = last_action
            fail_count += 1

        action = np.concatenate([I_cmd, v_lye_cmd, [v_c_cmd]])
        last_action = [I_cmd, v_lye_cmd, v_c_cmd]

        T_s_vec = state[1:5]
        T_sep = state[5]
        n_gas = state[12]
        _, U_cell_vec, _ = sim._calculate_electrochemical_properties(I_cmd, T_s_vec)
        P_real = np.sum(U_cell_vec * I_cmd * sim.N_cell)
        hto_pct = (n_gas * sim.R * T_sep) / (sim.P_sys * sim.V_sep_gas) * 100

        max_hto = max(max_hto, hto_pct)
        max_T = max(max_T, np.max(T_s_vec))
        if hto_pct > 2.0:
            hto_violations += 1
        if np.any(T_s_vec > 363.15):
            T_violations += 1
        power_errors.append(abs(P_real - P_ref) / 1e6)

        for _ in range(int(dt / sim.dt)):
            sim.step(action)

    mean_power_err = np.mean(power_errors)
    return {
        'max_hto': max_hto,
        'max_T': max_T - 273.15,
        'hto_violations': hto_violations,
        'T_violations': T_violations,
        'mean_power_err_MW': mean_power_err,
        'fail_count': fail_count,
    }


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    model_path = os.path.join(base_dir, 'output', 'multi_stack', 'policy', 'flow_tcn_policy_best.pth')
    stats_path = os.path.join(base_dir, 'output', 'multi_stack', 'diffusion_stats.npz')

    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return

    sim_temp = MultiStackSimulator(dt=0.2)
    initial_state = sim_temp.reset()

    P_ref_profile = [
        (0, 450, 6.0e6),
        (450, 900, 2.0e6),   # 低功率，HTO风险
    ]

    # 参数扫描
    param_sets = [
        {'name': 'base',    'gamma': 1.0,   'rho': 5000,  'margin_hto': 0.0},
        {'name': 'g10',     'gamma': 10.0,  'rho': 5000,  'margin_hto': 0.0},
        {'name': 'm005',    'gamma': 1.0,   'rho': 5000,  'margin_hto': 0.005},
        {'name': 'g10m005', 'gamma': 10.0,  'rho': 5000,  'margin_hto': 0.005},
        {'name': 'g100m005','gamma': 100.0, 'rho': 5000,  'margin_hto': 0.005},
        {'name': 'g10m005r1w', 'gamma': 10.0, 'rho': 10000, 'margin_hto': 0.005},
    ]

    print("="*80)
    print("多槽CBF参数调优")
    print("="*80)
    print(f"测试时长: {len(P_ref_profile)*450/60:.0f} min | 模型: {model_path}")
    print()

    results = []
    for cfg in param_sets:
        gamma = cfg['gamma']
        rho = cfg['rho']
        margin_hto = cfg['margin_hto']

        gamma_vec = [gamma] * 17
        rho_vec = [rho] * 17
        h_margin_vec = [0.0] * 4 + [margin_hto] + [0.0] * 12
        soft_mask = [True] * 17

        print(f"\n测试配置: {cfg['name']} | gamma={gamma}, rho={rho}, margin_hto={margin_hto}")
        print("-"*60)

        controller = MultiStackCBFModelController(
            dt=60.0, horizon=5, model_type='flow_tcn',
            model_path=model_path, stats_path=stats_path,
            use_cbf_projection=True,
            gamma_vec=gamma_vec, rho_vec=rho_vec,
            h_margin_vec=h_margin_vec,
            lambda_u_scale=1000.0,
            soft_mask=soft_mask,
            normalize=True
        )

        metrics = run_single_test(controller, initial_state, P_ref_profile, duration=900)
        results.append((cfg['name'], metrics))

        print(f"  Max HTO:      {metrics['max_hto']:.4f}%")
        print(f"  Max Temp:     {metrics['max_T']:.2f}°C")
        print(f"  HTO Viol:     {metrics['hto_violations']}")
        print(f"  Temp Viol:    {metrics['T_violations']}")
        print(f"  Power Err:    {metrics['mean_power_err_MW']:.3f} MW")
        print(f"  Proj Fails:   {metrics['fail_count']}")

    print("\n" + "="*80)
    print("汇总结果")
    print("="*80)
    print(f"{'Config':<12} {'Max HTO':<10} {'Max T':<8} {'HTO Viol':<10} {'Power Err':<10} {'Fails':<6}")
    print("-"*80)
    for name, m in results:
        print(f"{name:<12} {m['max_hto']:>8.4f}% {m['max_T']:>6.2f}°C {m['hto_violations']:>8} {m['mean_power_err_MW']:>8.3f} {m['fail_count']:>6}")

    # 推荐配置
    safe_configs = [(n, m) for n, m in results if m['max_hto'] <= 2.0 and m['hto_violations'] == 0]
    if safe_configs:
        best = min(safe_configs, key=lambda x: x[1]['mean_power_err_MW'])
        print(f"\n推荐配置: {best[0]} (HTO安全且功率跟踪误差最小)")
    else:
        best = min(results, key=lambda x: x[1]['max_hto'])
        print(f"\n推荐配置: {best[0]} (HTO最低，但仍需进一步优化)")


if __name__ == "__main__":
    main()
