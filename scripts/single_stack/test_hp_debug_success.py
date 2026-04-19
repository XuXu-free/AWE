"""
调试 high_power 场景下 actual_success 为 False 的具体原因。
"""
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from controller.single_stack.cbf_projection import SingleStackCBFProjection
from plant.single_stack_simulator import SingleStackSimulator


def analyze():
    sim = SingleStackSimulator(sim_dt=0.2)
    sim.reset()

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

    # Monkey-patch to suppress solver output completely
    original_project = projector.project
    import types
    import casadi as ca

    def silent_project(self, u_ref, state, u_last=None, verbose=False, active_mask=None):
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            result = original_project(u_ref, state, u_last, verbose, active_mask)
        finally:
            sys.stdout = old_stdout
        return result

    projector.project = types.MethodType(silent_project, projector)

    dt = 0.2
    steps = int(7200 / dt)
    ctrl_steps = int(60.0 / dt)
    current_action = u_ref.copy()
    last_action = current_action.copy()

    n_failures = 0
    failure_details = []

    for i in range(steps):
        if i % ctrl_steps == 0:
            state = sim.state
            current_action, success, info = projector.project(
                u_ref, state, u_last=last_action, active_mask=active_mask
            )
            last_action = current_action.copy()

            if not success:
                n_failures += 1
                if len(failure_details) < 5:
                    cbf = info.get('cbf', np.zeros(5))
                    h = info.get('h', np.zeros(5))
                    slack = info.get('slack', np.zeros(5))
                    detail = {
                        't': i * dt / 60,
                        'u_opt': current_action,
                        'cbf': cbf,
                        'h': h,
                        'slack': slack,
                        'solver_msg': info.get('solver_message', ''),
                        'max_slack': info.get('max_slack', 0),
                    }
                    failure_details.append(detail)

        sim.step(current_action)

    print(f"Total failures: {n_failures}")
    print("\nFirst 5 failure details:")
    for d in failure_details:
        print(f"\n  t={d['t']:.0f}min")
        print(f"  u_opt={d['u_opt']}")
        print(f"  h={d['h']}")
        print(f"  cbf={d['cbf']}")
        print(f"  active_mask={active_mask}")
        print(f"  cbf[active]={d['cbf'][active_mask]}")
        print(f"  slack={d['slack']}")
        print(f"  max_slack={d['max_slack']:.4f}")
        print(f"  solver_msg={d['solver_msg'][:200]}")

        # 分析哪个 hard 约束不满足
        hard_mask = (~np.array(soft_mask, dtype=bool)) & active_mask
        for j in range(5):
            if hard_mask[j] and d['cbf'][j] < -1e-3:
                print(f"  -> Hard constraint {j} violated: cbf[{j}]={d['cbf'][j]:.4f}")


if __name__ == "__main__":
    analyze()
