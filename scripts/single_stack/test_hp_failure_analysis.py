"""
分析 high_power 场景下 solver failures 的原因和分布。
"""
import os
import sys
import numpy as np
import contextlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from controller.single_stack.cbf_projection import SingleStackCBFProjection
from plant.single_stack_simulator import SingleStackSimulator


def analyze():
    sim = SingleStackSimulator(sim_dt=0.2)
    sim.reset()

    # 原始基线参数
    u_ref = np.array([9360.0, 0.03, 1.0])
    active_mask = np.array([True, False, True, True, True])
    gamma_vec = [50, 0.3, 100, 100, 10]
    h_margin_vec = [0.5, 0.01, 0, 0, 0]
    rho_vec = [50000, 10000, 50000, 50000, 10000]
    soft_mask = [False, True, False, False, True]

    projector = SingleStackCBFProjection(
        dt=60.0, gamma_vec=gamma_vec, rho_vec=rho_vec,
        h_margin_vec=h_margin_vec, normalize=True, lambda_u_scale=1000.0,
        soft_mask=soft_mask, active_mask=active_mask
    )

    dt = 0.2
    steps = int(7200 / dt)
    ctrl_steps = int(60.0 / dt)
    current_action = u_ref.copy()
    last_action = current_action.copy()

    failure_times = []
    failure_reasons = []
    max_power = 0.0
    max_temp = 0.0

    for i in range(steps):
        if i % ctrl_steps == 0:
            state = sim.state
            current_action, success, info = projector.project(
                u_ref, state, u_last=last_action, active_mask=active_mask
            )
            last_action = current_action.copy()

            if not success:
                failure_times.append(i * dt / 60)
                # 分析失败原因
                reasons = []
                if not info.get('optimization_success', True):
                    reasons.append('opt_fail')
                if info.get('max_slack', 0) > 1e-3:
                    reasons.append(f"slack={info['max_slack']:.4f}")
                cbf = info.get('cbf', np.zeros(5))
                active = info.get('active_mask', active_mask)
                if np.any(cbf[active] < -1e-3):
                    reasons.append(f"cbf_violation_min={np.min(cbf[active]):.4f}")
                failure_reasons.append(','.join(reasons) if reasons else 'unknown')

        state = sim.state
        I = current_action[0]
        T_s = state[1]
        U_cell = 1.229 + 3.202e-5 * I + (8.970e-8 * T_s * I)
        Power = U_cell * I * 368 / 1e6
        max_power = max(max_power, Power)
        max_temp = max(max_temp, T_s - 273.15)

        sim.step(current_action)

    print(f"Total failures: {len(failure_times)}")
    print(f"Max Power: {max_power:.3f} MW")
    print(f"Max Temp: {max_temp:.2f} C")
    print(f"\nFirst 10 failure times (min): {failure_times[:10]}")
    print(f"First 10 reasons: {failure_reasons[:10]}")
    print(f"\nLast 10 failure times (min): {failure_times[-10:]}")
    print(f"Last 10 reasons: {failure_reasons[-10:]}")

    # 统计原因
    from collections import Counter
    reason_counter = Counter(failure_reasons)
    print(f"\nFailure reason distribution:")
    for reason, count in reason_counter.most_common():
        print(f"  {reason}: {count}")


if __name__ == "__main__":
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        pass  # 只过滤 projector 内部输出？不行，内部直接 print
    analyze()
