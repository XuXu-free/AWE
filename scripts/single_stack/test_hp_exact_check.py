"""
精确检查 high_power 不同参数下的约束满足情况，使用 projector 内部函数。
"""
import os
import sys
import numpy as np
import contextlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from controller.single_stack.cbf_projection import SingleStackCBFProjection
from plant.single_stack_simulator import SingleStackSimulator


def run_check(name, u_ref, gamma_vec, h_margin_vec, rho_vec, soft_mask, active_mask):
    sim = SingleStackSimulator(sim_dt=0.2)
    sim.reset()
    dt = 0.2
    steps = int(7200 / dt)
    ctrl_steps = int(60.0 / dt)
    current_action = u_ref.copy()
    last_action = current_action.copy()

    projector = SingleStackCBFProjection(
        dt=60.0, gamma_vec=gamma_vec, rho_vec=rho_vec,
        h_margin_vec=h_margin_vec, normalize=True, lambda_u_scale=1000.0,
        soft_mask=soft_mask, active_mask=active_mask
    )

    max_T = max_hto = max_U = max_P = 0.0
    n_failures = 0
    n_temp_violation = 0
    n_power_violation = 0
    n_voltage_violation = 0
    max_temp_slack = 0.0
    max_power_slack = 0.0

    for i in range(steps):
        if i % ctrl_steps == 0:
            state = sim.state
            current_action, success, info = projector.project(
                u_ref, state, u_last=last_action, active_mask=active_mask
            )
            last_action = current_action.copy()
            if not success:
                n_failures += 1

            # 每 5 分钟检查一次精确的 CBF 值
            if i % (25 * ctrl_steps) == 0:
                h = info.get('h', np.zeros(5))
                cbf = info.get('cbf', np.zeros(5))
                slack = info.get('slack', np.zeros(5))
                # h[0]=T_max-T_s-hm, h[2]=U_max-U_cell-hm, h[3]=P_max-Power-hm
                T_s = state[1]
                # 从 h[0] 反推温度是否超标: T_s = T_max - h[0] - h_margin
                actual_T = projector.T_max - h[0] - h_margin_vec[0]
                # 从 h[3] 反推功率: Power = P_max - h[3] - h_margin
                actual_P = projector.P_stack_max - h[3] - h_margin_vec[3]
                # 从 h[2] 反推电压
                actual_U = projector.U_cell_max - h[2] - h_margin_vec[2]
                if slack[0] > max_temp_slack:
                    max_temp_slack = slack[0]
                if slack[3] > max_power_slack:
                    max_power_slack = slack[3]

        state = sim.state
        I = current_action[0]
        T_s = state[1]
        U_cell = 1.229 + 3.202e-5 * I + (8.970e-8 * T_s * I)
        Power = U_cell * I * 368 / 1e6

        max_T = max(max_T, T_s - 273.15)
        max_U = max(max_U, U_cell)
        max_P = max(max_P, Power)
        hto = (state[6] * projector.R * state[2]) / (projector.P_sys * projector.V_sep_gas) * 100
        max_hto = max(max_hto, hto)

        if T_s - 273.15 > 90:
            n_temp_violation += 1
        if Power > 6.0:
            n_power_violation += 1
        if U_cell > 2.2:
            n_voltage_violation += 1

        sim.step(current_action)

    print(f"{name:<35s}: MaxT={max_T:6.2f}C  MaxP={max_P:6.3f}MW  MaxU={max_U:5.3f}V  "
          f"HTO={max_hto:.3f}%  Fail={n_failures:4d}  "
          f"T_vio={n_temp_violation:5d}  P_vio={n_power_violation:5d}  U_vio={n_voltage_violation:5d}  "
          f"max_T_slack={max_temp_slack:.4f}  max_P_slack={max_power_slack:.4f}")


def main():
    print("=" * 140)
    print("精确约束检查 - 使用 projector 内部 h 和 cbf 值")
    print("=" * 140)

    active_mask = np.array([True, False, True, True, True])

    configs = [
        ("原始基线", [9360.0, 0.03, 1.0], [50, 0.3, 100, 100, 10], [0.5, 0.01, 0, 0, 0], [50000, 10000, 50000, 50000, 10000], [0, 1, 0, 0, 1]),
        ("当前保守", [9360.0, 0.05, 1.0], [10, 0.3, 100, 100, 5], [1.0, 0.01, 0, 0, 0], [50000, 10000, 50000, 50000, 10000], [0, 1, 0, 0, 1]),
        ("Tsoft高rho_Phard", [9360.0, 0.03, 1.0], [50, 0.3, 100, 100, 10], [0.5, 0.01, 0, 0, 0], [1e8, 10000, 50000, 50000, 10000], [1, 1, 0, 0, 1]),
        ("Tsoft高rho_Phard_gamma100", [9360.0, 0.03, 1.0], [100, 0.3, 100, 100, 20], [0.2, 0.01, 0, 0, 0], [1e8, 10000, 50000, 50000, 10000], [1, 1, 0, 0, 1]),
        ("Tsoft高rho_Psoft高rho", [9360.0, 0.03, 1.0], [50, 0.3, 100, 100, 10], [0.5, 0.01, 0, 0, 0], [1e8, 10000, 1e8, 1e12, 10000], [1, 1, 1, 1, 1]),
        ("Thard_Psoft高rho", [9360.0, 0.03, 1.0], [50, 0.3, 100, 100, 10], [0.5, 0.01, 0, 0, 0], [50000, 10000, 1e8, 1e12, 10000], [0, 1, 1, 1, 0]),
        ("全soft_rho1e8", [9360.0, 0.03, 1.0], [50, 0.3, 100, 100, 10], [0.5, 0.01, 0, 0, 0], [1e8, 1e8, 1e8, 1e8, 1e8], [1, 1, 1, 1, 1]),
    ]

    print(f"\n{'名称':<35s} {'MaxT':>6s} {'MaxP':>7s} {'MaxU':>6s} {'HTO':>6s} {'Fail':>5s} {'T_vio':>6s} {'P_vio':>6s} {'U_vio':>6s} {'T_slack':>8s} {'P_slack':>8s}")
    print("-" * 140)

    for name, u_ref, g, hm, rho, soft in configs:
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            run_check(name, np.array(u_ref), g, hm, rho, [bool(s) for s in soft], active_mask)


if __name__ == "__main__":
    main()
