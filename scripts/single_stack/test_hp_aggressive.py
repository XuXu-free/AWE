"""
测试 high_power 更激进的 CBF 参数组合，减少保守性，提高功率利用率。
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
    n_power_violation = 0

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
        # 简化功率计算
        U_cell = 1.229 + 3.202e-5 * I + (8.970e-8 * T_s * I)
        Power = U_cell * I * 368 / 1e6  # MW

        hto = (state[6] * projector.R * state[2]) / (projector.P_sys * projector.V_sep_gas) * 100
        max_hto = max(max_hto, hto)
        max_temp = max(max_temp, T_s - 273.15)
        min_temp = min(min_temp, T_s - 273.15)
        max_current = max(max_current, I)
        max_power = max(max_power, Power)
        if Power > 6.0:
            n_power_violation += 1

        sim.step(current_action)

    return max_temp, max_hto, max_current, min_temp, n_failures, max_power, n_power_violation


def main():
    print("=" * 110)
    print("HIGH POWER 激进参数测试")
    print("=" * 110)

    active_mask = np.array([True, False, True, True, True])

    configs = [
        # 名称, u_ref, gamma_vec, h_margin_vec, rho_vec, soft_mask
        ("基线(原始)", [9360.0, 0.03, 1.0], [50, 0.3, 100, 100, 10], [0.5, 0.01, 0, 0, 0], [50000, 10000, 50000, 50000, 10000], [0, 1, 0, 0, 1]),
        ("当前保守", [9360.0, 0.05, 1.0], [10, 0.3, 100, 100, 5], [1.0, 0.01, 0, 0, 0], [50000, 10000, 50000, 50000, 10000], [0, 1, 0, 0, 1]),

        # 提高 gamma_T，降低 h_margin
        ("激进A", [9360.0, 0.03, 1.0], [100, 0.3, 100, 100, 20], [0.2, 0.01, 0, 0, 0], [50000, 10000, 50000, 50000, 10000], [0, 1, 0, 0, 1]),
        ("激进B", [9360.0, 0.04, 1.0], [100, 0.3, 100, 100, 20], [0.2, 0.01, 0, 0, 0], [50000, 10000, 50000, 50000, 10000], [0, 1, 0, 0, 1]),
        ("激进C", [9360.0, 0.03, 1.0], [200, 0.3, 100, 200, 50], [0.2, 0.01, 0, 0, 0], [50000, 10000, 50000, 50000, 10000], [0, 1, 0, 0, 1]),

        # 温度 soft + 功率 hard
        ("温度soft_功率hard", [9360.0, 0.03, 1.0], [100, 0.3, 100, 100, 20], [0.2, 0.01, 0, 0, 0], [50000, 10000, 50000, 50000, 10000], [1, 1, 0, 0, 1]),
        ("温度soft_功率hard_vlye04", [9360.0, 0.04, 1.0], [100, 0.3, 100, 100, 20], [0.2, 0.01, 0, 0, 0], [50000, 10000, 50000, 50000, 10000], [1, 1, 0, 0, 1]),

        # 功率也 soft（极高 rho），让系统可以贴着功率边界
        ("全soft高rho", [9360.0, 0.03, 1.0], [100, 0.3, 100, 100, 20], [0.2, 0.01, 0, 0, 0], [50000, 10000, 50000, 1000000, 10000], [1, 1, 1, 1, 1]),
        ("全soft高rho_vlye04", [9360.0, 0.04, 1.0], [100, 0.3, 100, 100, 20], [0.2, 0.01, 0, 0, 0], [50000, 10000, 50000, 1000000, 10000], [1, 1, 1, 1, 1]),

        # 降低 gamma_P（功率约束更严格），但温度约束宽松
        ("严格功率_宽松温度", [9360.0, 0.03, 1.0], [200, 0.3, 100, 10, 50], [0.1, 0.01, 0, 0, 0], [50000, 10000, 50000, 50000, 10000], [1, 1, 0, 0, 1]),
        ("严格功率_宽松温度_vlye04", [9360.0, 0.04, 1.0], [200, 0.3, 100, 10, 50], [0.1, 0.01, 0, 0, 0], [50000, 10000, 50000, 50000, 10000], [1, 1, 0, 0, 1]),

        # 更小 gamma_P（功率更硬）
        ("功率极严格", [9360.0, 0.04, 1.0], [200, 0.3, 100, 5, 50], [0.1, 0.01, 0, 0, 0], [50000, 10000, 50000, 50000, 10000], [1, 1, 0, 0, 1]),

        # 回到原始 gamma，但 soft_mask 不同
        ("原始gamma_温度soft", [9360.0, 0.03, 1.0], [50, 0.3, 100, 100, 10], [0.5, 0.01, 0, 0, 0], [50000, 10000, 50000, 50000, 10000], [1, 1, 0, 0, 1]),
        ("原始gamma_全soft", [9360.0, 0.03, 1.0], [50, 0.3, 100, 100, 10], [0.5, 0.01, 0, 0, 0], [50000, 10000, 50000, 1000000, 10000], [1, 1, 1, 1, 1]),
    ]

    print(f"\n{'名称':<25s} {'MaxT':>7s} {'MaxI':>7s} {'MaxP':>7s} {'HTO':>7s} {'Fail':>6s} {'P_vio':>6s}")
    print("-" * 110)

    results = []
    for name, u_ref, g, hm, rho, soft in configs:
        u_ref = np.array(u_ref)
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            max_t, max_h, max_i, min_t, fails, max_p, p_vio = run_fast(
                7200, u_ref, active_mask, g, hm, rho, [bool(s) for s in soft]
            )
        print(f"{name:<25s} {max_t:7.2f} {max_i:7.0f} {max_p:7.3f} {max_h:7.3f} {fails:6d} {p_vio:6d}")
        results.append((name, max_t, max_i, max_p, max_h, fails, p_vio, u_ref, g, hm, rho, soft))

    # 推荐最佳方案（按 Max Power 排序，同时失败数少）
    print("\n--- 按最大功率排序（Top 5）---")
    sorted_by_power = sorted(results, key=lambda x: (-x[3], x[5]))
    for i, r in enumerate(sorted_by_power[:5]):
        print(f"#{i+1}: {r[0]} | MaxP={r[3]:.3f}MW MaxT={r[1]:.2f}C MaxI={r[2]:.0f}A Fail={r[5]}")


if __name__ == "__main__":
    main()
