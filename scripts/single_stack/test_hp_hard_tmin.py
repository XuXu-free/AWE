"""
测试把 Tmin 变成 hard 约束对 failures 的影响。
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
    print("HIGH POWER - Tmin hard/soft 对比")
    print("-" * 100)

    active_mask = np.array([True, False, True, True, True])

    configs = [
        ("原始基线_Tmin_soft", [9360.0, 0.03, 1.0], active_mask, [50, 0.3, 100, 100, 10], [0.5, 0.01, 0, 0, 0], [50000, 10000, 50000, 50000, 10000], [0, 1, 0, 0, 1]),
        ("Tmin_hard", [9360.0, 0.03, 1.0], active_mask, [50, 0.3, 100, 100, 10], [0.5, 0.01, 0, 0, 0], [50000, 10000, 50000, 50000, 10000], [0, 1, 0, 0, 0]),
        ("Tmin_hard_gamma100", [9360.0, 0.03, 1.0], active_mask, [100, 0.3, 100, 100, 20], [0.2, 0.01, 0, 0, 0], [50000, 10000, 50000, 50000, 10000], [0, 1, 0, 0, 0]),
        ("Tmin_hard_gamma200", [9360.0, 0.03, 1.0], active_mask, [200, 0.3, 100, 100, 50], [0.1, 0.01, 0, 0, 0], [50000, 10000, 50000, 50000, 10000], [0, 1, 0, 0, 0]),
        ("当前保守_Tmin_soft", [9360.0, 0.05, 1.0], active_mask, [10, 0.3, 100, 100, 5], [1.0, 0.01, 0, 0, 0], [50000, 10000, 50000, 50000, 10000], [0, 1, 0, 0, 1]),
        ("当前保守_Tmin_hard", [9360.0, 0.05, 1.0], active_mask, [10, 0.3, 100, 100, 5], [1.0, 0.01, 0, 0, 0], [50000, 10000, 50000, 50000, 10000], [0, 1, 0, 0, 0]),
    ]

    for name, u_ref, active_mask, g, hm, rho, soft in configs:
        u_ref = np.array(u_ref)
        max_t, max_h, max_i, min_t, fails, max_p = run_fast(7200, u_ref, active_mask, g, hm, rho, [bool(s) for s in soft])
        print(f"{name:30s}: MaxT={max_t:6.2f}C  MaxI={max_i:6.0f}A  MaxP={max_p:6.3f}MW  HTO={max_h:.3f}%  Fail={fails:4d}")


if __name__ == "__main__":
    main()
