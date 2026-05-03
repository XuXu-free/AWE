"""
自适应 CBF 参数搜索脚本 v2

使用 HTO=0% 的纯净状态（最大安全裕度），测试是否仍存在不触发 CBF 的参数。
如果在这种最乐观状态下仍触发，则说明仅靠调参无法实现目标。
"""

import os
import sys
import numpy as np
import itertools

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from controller.multi_stack.cbf_projection_ho import MultiStackCBFProjectionHO
from plant.multi_stack_simulator import MultiStackSimulator


def build_state_clean(T_s_degC, T_s_in_degC=75.0, T_sep_degC=80.0, T_c_out_degC=20.0):
    """构建一个 HTO=0 的纯净状态，测试最乐观情况。"""
    state = np.zeros(13)
    state[0] = T_s_in_degC + 273.15
    state[1:5] = T_s_degC + 273.15
    state[5] = T_sep_degC + 273.15
    state[6] = T_c_out_degC + 273.15
    state[7:11] = 0.0   # n_H2_an = 0
    state[11] = 0.0     # n_liq = 0
    state[12] = 0.0     # n_gas = 0
    return state


def test_params(state, u_ref, alpha1_hto, alpha2_hto, h_margin_hto, gamma_hto=1.0):
    """测试一组参数是否触发 CBF。"""
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

    temp_triggered = np.any(cbf_temp < -1e-4)
    hto_triggered = cbf_hto < -1e-4

    return {
        'success': success,
        'cbf_hto': cbf_hto,
        'h_hto': h_hto,
        'temp_triggered': temp_triggered,
        'hto_triggered': hto_triggered,
    }


def main():
    print("=" * 100)
    print("自适应 CBF 参数搜索 v2 (HTO=0% 纯净状态)")
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
        state = build_state_clean(T)
        for I in currents:
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
        # 按 alpha 之和排序，优先较小的 alpha
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

        import pandas as pd
        df = pd.DataFrame(results_sorted)
        output_path = 'output/multi_stack/test/adaptive_alpha_search_v2.csv'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n详细结果已保存至: {output_path}")
    else:
        print("[结论] 即使在 HTO=0% 的最乐观状态下，也没有找到不触发 CBF 的参数组合。")
        print("说明：参考动作（电流）本身导致 HTO 动力学不安全，仅靠调整 alpha/h_margin 无法避免触发。")
        print("建议方案：")
        print("  1. 在低温阶段限制电流上升速率")
        print("  2. 使用多候选采样，选择不触发 CBF 的动作")
        print("  3. 在低温阶段使用更保守的参考控制器")
    print("=" * 100)


if __name__ == "__main__":
    main()
