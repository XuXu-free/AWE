"""
测试 high_power 场景下 soft 约束 + 极高 rho 的效果。
"""
import os
import sys
import numpy as np
import contextlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from controller.single_stack.cbf_projection import SingleStackCBFProjection
from plant.single_stack_simulator import SingleStackSimulator


def run_fast(duration=7200, u_ref=None, active_mask=None, gamma_vec=None, h_margin_vec=None,
             rho_vec=None, soft_mask=None):
    sim = SingleStackSimulator(sim_dt=0.2)
    sim.reset()
    dt = 0.2
    steps = int(duration / dt)
    ctrl_steps = int(60.0 / dt)
    current_action = u_ref.copy()
    last_action = current_action.copy()

    max_hto = max_temp = max_current = max_power = 0.0
    min_temp = 999.0
    n_failures = 0
    n_temp_violation = 0
    n_power_violation = 0
    n_voltage_violation = 0

    projector = SingleStackCBFProjection(
        dt=60.0, gamma_vec=gamma_vec, rho_vec=rho_vec,
        h_margin_vec=h_margin_vec, normalize=True, lambda_u_scale=1000.0,
        soft_mask=soft_mask, active_mask=active_mask
    )

    for i in range(steps):
        if i % ctrl_steps == 0:
            current_action, success, _ = projector.project(
                u_ref, sim.state, u_last=last_action, active_mask=active_mask
            )
            last_action = current_action.copy()
            if not success:
                n_failures += 1

        state = sim.state
        I = current_action[0]
        T_s = state[1]
        U_cell = 1.229 + 3.202e-5 * I + (8.970e-8 * T_s * I)
        Power = U_cell * I * 368 / 1e6

        hto = (state[6] * projector.R * state[2]) / (projector.P_sys * projector.V_sep_gas) * 100
        max_hto = max(max_hto, hto)
        max_temp = max(max_temp, T_s - 273.15)
        min_temp = min(min_temp, T_s - 273.15)
        max_current = max(max_current, I)
        max_power = max(max_power, Power)

        if T_s - 273.15 > 90:
            n_temp_violation += 1
        if Power > 6.0:
            n_power_violation += 1
        if U_cell > 2.2:
            n_voltage_violation += 1

        sim.step(current_action)

    return max_temp, max_hto, max_current, min_temp, n_failures, max_power, n_temp_violation, n_power_violation, n_voltage_violation


def main():
    print("=" * 120)
    print("HIGH POWER - Soft 约束 + 高 rho 测试")
    print("=" * 120)

    active_mask = np.array([True, False, True, True, True])

    configs = [
        ("原始基线(hard)", [9360.0, 0.03, 1.0], [50, 0.3, 100, 100, 10], [0.5, 0.01, 0, 0, 0], [50000, 10000, 50000, 50000, 10000], [0, 1, 0, 0, 1]),

        # 所有 soft，rho 极高
        ("全soft_rho1e6", [9360.0, 0.03, 1.0], [50, 0.3, 100, 100, 10], [0.5, 0.01, 0, 0, 0], [1e6, 1e6, 1e6, 1e6, 1e6], [1, 1, 1, 1, 1]),
        ("全soft_rho1e8", [9360.0, 0.03, 1.0], [50, 0.3, 100, 100, 10], [0.5, 0.01, 0, 0, 0], [1e8, 1e8, 1e8, 1e8, 1e8], [1, 1, 1, 1, 1]),
        ("全soft_T高rho", [9360.0, 0.03, 1.0], [50, 0.3, 100, 100, 10], [0.5, 0.01, 0, 0, 0], [1e8, 10000, 50000, 1e12, 1e8], [1, 1, 1, 1, 1]),

        # 只有功率 hard，其他 soft + 高 rho
        ("Tsoft_Vsoft_Phard_rho1e8", [9360.0, 0.03, 1.0], [50, 0.3, 100, 100, 10], [0.5, 0.01, 0, 0, 0], [1e8, 10000, 1e8, 50000, 1e8], [1, 1, 1, 0, 1]),
        ("Tsoft_Vsoft_Phard_rho1e8_gamma100", [9360.0, 0.03, 1.0], [100, 0.3, 100, 100, 20], [0.2, 0.01, 0, 0, 0], [1e8, 10000, 1e8, 50000, 1e8], [1, 1, 1, 0, 1]),
        ("Tsoft_Vsoft_Phard_rho1e8_gamma200", [9360.0, 0.03, 1.0], [200, 0.3, 100, 100, 50], [0.1, 0.01, 0, 0, 0], [1e8, 10000, 1e8, 50000, 1e8], [1, 1, 1, 0, 1]),

        # 功率也 soft 但 rho 极高，温度 hard
        ("Thard_Vsoft_Psoft_rho1e12", [9360.0, 0.03, 1.0], [50, 0.3, 100, 100, 10], [0.5, 0.01, 0, 0, 0], [50000, 10000, 1e8, 1e12, 10000], [0, 1, 1, 1, 0]),
        ("Thard_Vsoft_Psoft_rho1e12_gamma100", [9360.0, 0.03, 1.0], [100, 0.3, 100, 100, 20], [0.2, 0.01, 0, 0, 0], [50000, 10000, 1e8, 1e12, 10000], [0, 1, 1, 1, 0]),
    ]

    print(f"\n{'名称':<30s} {'MaxT':>7s} {'MaxI':>7s} {'MaxP':>7s} {'HTO':>7s} {'Fail':>6s} {'T_vio':>6s} {'P_vio':>6s} {'U_vio':>6s}")
    print("-" * 120)

    results = []
    for name, u_ref, g, hm, rho, soft in configs:
        u_ref = np.array(u_ref)
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            max_t, max_h, max_i, min_t, fails, max_p, t_vio, p_vio, u_vio = run_fast(
                7200, u_ref, active_mask, g, hm, rho, [bool(s) for s in soft]
            )
        print(f"{name:<30s} {max_t:7.2f} {max_i:7.0f} {max_p:7.3f} {max_h:7.3f} {fails:6d} {t_vio:6d} {p_vio:6d} {u_vio:6d}")
        results.append((name, max_t, max_i, max_p, max_h, fails, t_vio, p_vio, u_vio))

    print("\n--- 推荐方案（Fail=0 且功率最高）---")
    good_results = [r for r in results if r[5] == 0 and r[7] == 0 and r[6] == 0]
    if good_results:
        sorted_good = sorted(good_results, key=lambda x: -x[3])
        for i, r in enumerate(sorted_good[:3]):
            print(f"#{i+1}: {r[0]} | MaxP={r[3]:.3f}MW MaxT={r[1]:.2f}C MaxI={r[2]:.0f}A")
    else:
        print("没有完全满足 Fail=0 + 无越界的方案。按功率排序：")
        sorted_all = sorted(results, key=lambda x: (-x[3], x[5]))
        for i, r in enumerate(sorted_all[:5]):
            print(f"#{i+1}: {r[0]} | MaxP={r[3]:.3f}MW MaxT={r[1]:.2f}C MaxI={r[2]:.0f}A Fail={r[5]} T_vio={r[6]} P_vio={r[7]}")


if __name__ == "__main__":
    main()
