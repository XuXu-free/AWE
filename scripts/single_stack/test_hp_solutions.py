"""
测试不同减少 failures 的方案对 high_power 场景的影响。
"""
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from controller.single_stack.cbf_projection import SingleStackCBFProjection
from plant.single_stack_simulator import SingleStackSimulator


def run_fast(duration, u_ref, active_mask, gamma_vec, h_margin_vec, rho_vec, soft_mask):
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

        sim.step(current_action)

    return max_temp, max_hto, max_current, min_temp, n_failures, max_power


def main():
    print("=" * 120)
    print("HIGH POWER - 减少 failures 的方案对比")
    print("=" * 120)

    base_active = np.array([True, False, True, True, True])
    no_v_active = np.array([True, False, False, True, True])

    configs = [
        # 名称, duration, u_ref, active_mask, gamma_vec, h_margin_vec, rho_vec, soft_mask
        ("原始基线_7200s", 7200, [9360.0, 0.03, 1.0], base_active, [50, 0.3, 100, 100, 10], [0.5, 0.01, 0, 0, 0], [50000, 10000, 50000, 50000, 10000], [0, 1, 0, 0, 1]),
        ("去掉电压约束_7200s", 7200, [9360.0, 0.03, 1.0], no_v_active, [50, 0.3, 100, 100, 10], [0.5, 0.01, 0, 0, 0], [50000, 10000, 50000, 50000, 10000], [0, 1, 0, 0, 1]),
        ("去掉电压约束+gamma100_7200s", 7200, [9360.0, 0.03, 1.0], no_v_active, [100, 0.3, 100, 100, 20], [0.2, 0.01, 0, 0, 0], [50000, 10000, 50000, 50000, 10000], [0, 1, 0, 0, 1]),
        ("去掉电压约束+gamma100_hm0.2_7200s", 7200, [9360.0, 0.03, 1.0], no_v_active, [100, 0.3, 100, 100, 20], [0.2, 0.01, 0, 0, 0], [50000, 10000, 50000, 50000, 10000], [0, 1, 0, 0, 1]),
        ("原始基线_3600s", 3600, [9360.0, 0.03, 1.0], base_active, [50, 0.3, 100, 100, 10], [0.5, 0.01, 0, 0, 0], [50000, 10000, 50000, 50000, 10000], [0, 1, 0, 0, 1]),
        ("去掉电压约束_3600s", 3600, [9360.0, 0.03, 1.0], no_v_active, [50, 0.3, 100, 100, 10], [0.5, 0.01, 0, 0, 0], [50000, 10000, 50000, 50000, 10000], [0, 1, 0, 0, 1]),
    ]

    print(f"\n{'名称':<40s} {'MaxT':>7s} {'MaxI':>7s} {'MaxP':>7s} {'HTO':>7s} {'Fail':>6s} {'MinT':>7s}")
    print("-" * 120)

    for name, duration, u_ref, active_mask, g, hm, rho, soft in configs:
        u_ref = np.array(u_ref)
        max_t, max_h, max_i, min_t, fails, max_p = run_fast(
            duration, u_ref, active_mask, g, hm, rho, [bool(s) for s in soft]
        )
        print(f"{name:<40s} {max_t:7.2f} {max_i:7.0f} {max_p:7.3f} {max_h:7.3f} {fails:6d} {min_t:7.2f}")


if __name__ == "__main__":
    main()
