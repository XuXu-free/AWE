"""
自适应 CBF 参数搜索脚本 v3

使用 HTO=1% 的真实状态（模拟 warmup 后），测试不同参数下的 CBF 触发情况。
"""

import os
import sys
import numpy as np
import itertools

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from controller.multi_stack.cbf_projection_ho import MultiStackCBFProjectionHO
from plant.multi_stack_simulator import MultiStackSimulator


def build_state_hto1(T_s_degC, T_s_in_degC=75.0, T_sep_degC=80.0, T_c_out_degC=20.0, hto_pct=1.0):
    """构建一个 HTO=1% 的真实状态。"""
    sim = MultiStackSimulator(dt=0.2)

    state = np.zeros(13)
    T_sep_k = T_sep_degC + 273.15
    state[0] = T_s_in_degC + 273.15
    state[1:5] = T_s_degC + 273.15
    state[5] = T_sep_k
    state[6] = T_c_out_degC + 273.15

    # HTO = (n_gas * R * T_sep) / (P_sys * V_sep_gas) * 100
    # => n_gas = HTO * P_sys * V_sep_gas / (100 * R * T_sep)
    n_gas = (hto_pct * sim.P_sys * sim.V_sep_gas) / (100 * sim.R * T_sep_k)
    state[12] = n_gas
    # 设置 n_H2_an 和 n_liq 为合理值（约占总 H2 的 10% 和 5%）
    state[7:11] = n_gas * 0.1
    state[11] = n_gas * 0.05
    return state


def test_params(state, u_ref, alpha1_hto, alpha2_hto, h_margin_hto, gamma_hto=1.0):
    alpha1_vec = [10.0, 10.0, 10.0, 10.0, alpha1_hto, 10.0, 10.0, 10.0, 10.0]
    alpha2_vec = [2.0, 2.0, 2.0, 2.0, alpha2_hto, 2.0, 2.0, 2.0, 2.0]
    h_margin_vec = [0.0, 0.0, 0.0, 0.0, h_margin_hto, 0.0, 0.0, 0.0, 0.0]
    gamma_vec = [10.0, 10.0, 10.0, 10.0, gamma_hto, 10.0, 10.0, 10.0, 10.0]

    projector = MultiStackCBFProjectionHO(
        dt=60.0,
        gamma_vec=gamma_vec,
        rho_vec=[5000]*9,
        h_margin_vec=h_margin_vec,
        normalize=True,
        lambda_u_scale=0.0,
        alpha1_vec=alpha1_vec,
        alpha2_vec=alpha2_vec,
    )

    u_safe, success, info = projector.project(u_ref, state, u_last=u_ref)

    cbf_temp = info['cbf'][0:4]
    cbf_hto = info['cbf'][4]
    h_hto = info['h'][4]

    return {
        'success': success,
        'cbf_hto': cbf_hto,
        'h_hto': h_hto,
        'temp_triggered': np.any(cbf_temp < -1e-4),
        'hto_triggered': cbf_hto < -1e-4,
    }


def main():
    print("=" * 100)
    print("自适应 CBF 参数搜索 v3 (HTO=1% 真实状态)")
    print("=" * 100)

    temps = [75.0, 80.0, 85.0]
    currents = [3000.0, 5000.0, 7500.0]

    alpha1_vals = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    alpha2_vals = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    h_margin_vals = [-0.005, -0.002, 0.0, 0.001, 0.002, 0.005]
    gamma_vals = [1.0, 10.0, 50.0, 100.0]

    total_combos = len(alpha1_vals) * len(alpha2_vals) * len(h_margin_vals) * len(gamma_vals)
    print(f"搜索空间: {total_combos} 种参数组合 x {len(temps)} 种温度 x {len(currents)} 种电流")
    print("-" * 100)

    results = []
    any_found = False

    for T in temps:
        for I in currents:
            state = build_state_hto1(T)
            u_ref = np.array([I] * 4 + [0.05] * 4 + [0.01])
            found_for_this = False

            for alpha1, alpha2, h_margin, gamma in itertools.product(alpha1_vals, alpha2_vals, h_margin_vals, gamma_vals):
                r = test_params(state, u_ref, alpha1, alpha2, h_margin, gamma)

                if not r['hto_triggered'] and not r['temp_triggered'] and r['h_hto'] > 0:
                    found_for_this = True
                    any_found = True
                    results.append({
                        'T': T, 'I': I,
                        'alpha1_hto': alpha1, 'alpha2_hto': alpha2,
                        'h_margin_hto': h_margin, 'gamma_hto': gamma,
                        'cbf_hto': r['cbf_hto'], 'h_hto': r['h_hto'],
                    })

            status = "FOUND" if found_for_this else "NONE"
            print(f"  T={T:.0f}°C, I={I:.0f}A: {status}")

    print("\n" + "=" * 100)
    if any_found:
        print(f"找到 {len(results)} 组安全参数（h_hto > 0 且不触发 CBF）")
        results_sorted = sorted(results, key=lambda x: x['alpha1_hto'] + x['alpha2_hto'])
        best = results_sorted[0]
        print(f"\n最佳参数（最小 alpha 之和）:")
        print(f"  alpha1_hto = {best['alpha1_hto']:.2f}")
        print(f"  alpha2_hto = {best['alpha2_hto']:.2f}")
        print(f"  h_margin_hto = {best['h_margin_hto']:.4f}")
        print(f"  gamma_hto = {best['gamma_hto']:.2f}")
        print(f"  测试条件: T={best['T']:.0f}°C, I={best['I']:.0f}A")
        print(f"  CBF_hto = {best['cbf_hto']:.6f}")
        print(f"  h_hto = {best['h_hto']:.6f}")

        # 统计 h_margin=0 的情况
        zero_margin = [r for r in results if r['h_margin_hto'] == 0]
        print(f"\nh_margin=0 的安全参数: {len(zero_margin)} 组")
        if zero_margin:
            best_z = sorted(zero_margin, key=lambda x: x['alpha1_hto'] + x['alpha2_hto'])[0]
            print(f"最佳 h_margin=0 参数: alpha1={best_z['alpha1_hto']:.2f}, alpha2={best_z['alpha2_hto']:.2f}, gamma={best_z['gamma_hto']:.2f}")

        import pandas as pd
        df = pd.DataFrame(results_sorted)
        output_path = 'output/multi_stack/test/adaptive_alpha_search_v3.csv'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n详细结果已保存至: {output_path}")
    else:
        print("[结论] 在 HTO=1% 状态下，也没有找到不触发 CBF 的参数组合。")
        print("说明：参考动作（电流）本身导致 HTO 动力学不安全。")
    print("=" * 100)


if __name__ == "__main__":
    main()
