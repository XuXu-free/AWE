"""
自适应 CBF 参数搜索脚本

目标：在 75~85°C 范围内，找到使 HTO CBF 不触发的 alpha1/alpha2/h_margin 参数组合。
同时检查温度 CBF 是否安全。
"""

import os
import sys
import numpy as np
import itertools

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from controller.multi_stack.cbf_projection_ho import MultiStackCBFProjectionHO
from plant.multi_stack_simulator import MultiStackSimulator


def build_state(T_s_degC, T_s_in_degC=75.0, T_sep_degC=80.0, T_c_out_degC=20.0):
    """构建一个典型状态，仅改变温度。"""
    sim = MultiStackSimulator(dt=0.2)
    sim.reset()
    u_low = np.array([500.0] * 4 + [0.05] * 4 + [0.01])
    for _ in range(5000):
        sim.step(u_low)

    state = sim.state.copy()
    state[0] = T_s_in_degC + 273.15
    state[1:5] = T_s_degC + 273.15
    state[5] = T_sep_degC + 273.15
    state[6] = T_c_out_degC + 273.15
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
    lie1_hto = info.get('lie1', [0]*9)[4]

    temp_triggered = np.any(cbf_temp < -1e-4)
    hto_triggered = cbf_hto < -1e-4

    return {
        'success': success,
        'cbf_hto': cbf_hto,
        'h_hto': h_hto,
        'lie1_hto': lie1_hto,
        'temp_triggered': temp_triggered,
        'hto_triggered': hto_triggered,
        'projection_needed': info.get('projection_needed', False),
    }


def main():
    print("=" * 100)
    print("自适应 CBF 参数搜索")
    print("目标：在 75~85°C、典型参考动作下，HTO CBF 不触发")
    print("=" * 100)

    temps = [75.0, 80.0, 85.0]
    currents = [3000.0, 5000.0, 7500.0]

    # 参数搜索空间
    alpha1_vals = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    alpha2_vals = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    h_margin_vals = [-0.005, -0.002, 0.0, 0.001, 0.002, 0.005]
    gamma_vals = [1.0, 10.0, 50.0, 100.0]  # 仅当 alpha2=0 时有效

    total_combos = len(alpha1_vals) * len(alpha2_vals) * len(h_margin_vals) * len(gamma_vals)
    print(f"搜索空间: {total_combos} 种参数组合 x {len(temps)} 种温度 x {len(currents)} 种电流 = {total_combos * len(temps) * len(currents)} 次测试")
    print("-" * 100)

    results = []
    best_result = None
    best_score = -1e9

    for T in temps:
        state = build_state(T)
        for I in currents:
            u_ref = np.array([I] * 4 + [0.05] * 4 + [0.01])
            print(f"\nT={T:.0f}°C, I={I:.0f}A:")

            found_any = False
            for alpha1, alpha2, h_margin, gamma in itertools.product(alpha1_vals, alpha2_vals, h_margin_vals, gamma_vals):
                r = test_params(state, u_ref, alpha1, alpha2, h_margin, gamma)

                if not r['hto_triggered'] and not r['temp_triggered']:
                    found_any = True
                    score = alpha1 + alpha2 * 10 + h_margin * 1000  # 优先选择较小的 alpha
                    if score > best_score:
                        best_score = score
                        best_result = {
                            'T': T, 'I': I,
                            'alpha1_hto': alpha1, 'alpha2_hto': alpha2,
                            'h_margin_hto': h_margin, 'gamma_hto': gamma,
                            'cbf_hto': r['cbf_hto'], 'h_hto': r['h_hto'],
                        }
                    results.append({
                        'T': T, 'I': I,
                        'alpha1_hto': alpha1, 'alpha2_hto': alpha2,
                        'h_margin_hto': h_margin, 'gamma_hto': gamma,
                        'cbf_hto': r['cbf_hto'], 'h_hto': r['h_hto'],
                        'lie1_hto': r['lie1_hto'],
                    })

            if found_any:
                print(f"  [FOUND] 存在不触发 CBF 的参数组合")
            else:
                print(f"  [NONE] 所有参数组合均触发 CBF")

    print("\n" + "=" * 100)
    print("搜索结果汇总")
    print("=" * 100)

    if best_result:
        print(f"最佳参数组合（优先较小 alpha）:")
        print(f"  alpha1_hto = {best_result['alpha1_hto']:.2f}")
        print(f"  alpha2_hto = {best_result['alpha2_hto']:.2f}")
        print(f"  h_margin_hto = {best_result['h_margin_hto']:.4f}")
        print(f"  gamma_hto = {best_result['gamma_hto']:.2f}")
        print(f"  测试条件: T={best_result['T']:.0f}°C, I={best_result['I']:.0f}A")
        print(f"  CBF_hto = {best_result['cbf_hto']:.4f}")
        print(f"  h_hto = {best_result['h_hto']:.6f}")
    else:
        print("[WARNING] 未找到任何不触发 CBF 的参数组合")
        print("建议：需要放宽 h_margin 或修改参考动作")

    # 保存所有成功结果到 CSV
    if results:
        import pandas as pd
        df = pd.DataFrame(results)
        df = df.sort_values(['T', 'I', 'alpha1_hto', 'alpha2_hto'])
        output_path = 'output/multi_stack/test/adaptive_alpha_search_results.csv'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n详细结果已保存至: {output_path}")

    print("=" * 100)


if __name__ == "__main__":
    main()
