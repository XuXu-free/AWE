import os, sys, io
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from controller.single_stack.cbf_projection import SingleStackCBFProjection
from plant.single_stack_simulator import SingleStackSimulator
import casadi as ca

# Suppress CasADi solver prints
original_opti_solve = ca.Opti.solve
def silent_solve(self):
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        return original_opti_solve(self)
    finally:
        sys.stdout = old_stdout
ca.Opti.solve = silent_solve

def run_fast(duration, u_ref, active_mask, gamma_vec, h_margin_vec, rho_vec, soft_mask):
    sim = SingleStackSimulator(sim_dt=0.2)
    sim.reset()
    dt = 0.2
    steps = int(duration / dt)
    ctrl_steps = int(60.0 / dt)
    current_action = np.array(u_ref).copy()
    last_action = current_action.copy()

    projector = SingleStackCBFProjection(
        dt=60.0, gamma_vec=gamma_vec, rho_vec=rho_vec,
        h_margin_vec=h_margin_vec, normalize=True, lambda_u_scale=1000.0,
        soft_mask=soft_mask, active_mask=active_mask
    )

    max_hto = max_temp = max_current = max_power = 0.0
    min_temp = 999.0
    n_failures = 0

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


base_active = np.array([True, False, True, True, True])
no_v_active = np.array([True, False, False, True, True])

configs = [
    ("baseline_u0.05", 7200, [9360.0, 0.05, 1.0], base_active, [10, 0.3, 100, 100, 5], [1.0, 0.01, 0, 0, 0], [50000, 10000, 50000, 50000, 10000], [0, 1, 0, 0, 1]),
    ("baseline_u0.03", 7200, [9360.0, 0.03, 1.0], base_active, [10, 0.3, 100, 100, 5], [1.0, 0.01, 0, 0, 0], [50000, 10000, 50000, 50000, 10000], [0, 1, 0, 0, 1]),
    ("no_v_constraint", 7200, [9360.0, 0.03, 1.0], no_v_active, [10, 0.3, 100, 100, 5], [1.0, 0.01, 0, 0, 0], [50000, 10000, 50000, 50000, 10000], [0, 1, 0, 0, 1]),
    ("soft_power", 7200, [9360.0, 0.03, 1.0], base_active, [10, 0.3, 100, 100, 5], [1.0, 0.01, 0, 0, 0], [50000, 10000, 50000, 50000, 10000], [0, 1, 1, 1, 1]),
    ("gamma_power_200", 7200, [9360.0, 0.03, 1.0], base_active, [10, 0.3, 100, 200, 5], [1.0, 0.01, 0, 0, 0], [50000, 10000, 50000, 50000, 10000], [0, 1, 0, 0, 1]),
    ("gamma_power_500", 7200, [9360.0, 0.03, 1.0], base_active, [10, 0.3, 100, 500, 5], [1.0, 0.01, 0, 0, 0], [50000, 10000, 50000, 50000, 10000], [0, 1, 0, 0, 1]),
    ("gamma_v_200_p_200", 7200, [9360.0, 0.03, 1.0], base_active, [10, 0.3, 200, 200, 5], [1.0, 0.01, 0, 0, 0], [50000, 10000, 50000, 50000, 10000], [0, 1, 0, 0, 1]),
    ("gamma_v_500_p_500", 7200, [9360.0, 0.03, 1.0], base_active, [10, 0.3, 500, 500, 5], [1.0, 0.01, 0, 0, 0], [50000, 10000, 50000, 50000, 10000], [0, 1, 0, 0, 1]),
    ("hmargin_T_0.5", 7200, [9360.0, 0.03, 1.0], base_active, [10, 0.3, 100, 100, 5], [0.5, 0.01, 0, 0, 0], [50000, 10000, 50000, 50000, 10000], [0, 1, 0, 0, 1]),
    ("hmargin_T_0.2", 7200, [9360.0, 0.03, 1.0], base_active, [10, 0.3, 100, 100, 5], [0.2, 0.01, 0, 0, 0], [50000, 10000, 50000, 50000, 10000], [0, 1, 0, 0, 1]),
    ("gamma_T_50", 7200, [9360.0, 0.03, 1.0], base_active, [50, 0.3, 100, 100, 10], [0.5, 0.01, 0, 0, 0], [50000, 10000, 50000, 50000, 10000], [0, 1, 0, 0, 1]),
    ("gamma_T_50_hm_0.5", 7200, [9360.0, 0.03, 1.0], base_active, [50, 0.3, 100, 100, 10], [0.5, 0.01, 0, 0, 0], [50000, 10000, 50000, 50000, 10000], [0, 1, 0, 0, 1]),
    ("only_power", 7200, [9360.0, 0.03, 1.0], np.array([False, False, False, True, False]), [10, 0.3, 100, 100, 5], [1.0, 0.01, 0, 0, 0], [50000, 10000, 50000, 50000, 10000], [0, 1, 0, 0, 1]),
]

print(f'{"name":<25s} {"MaxT":>7s} {"MaxI":>7s} {"MaxP":>7s} {"HTO":>7s} {"Fail":>6s} {"MinT":>7s}')
print("-" * 80)
for name, duration, u_ref, active_mask, g, hm, rho, soft in configs:
    u_ref = np.array(u_ref)
    max_t, max_h, max_i, min_t, fails, max_p = run_fast(
        duration, u_ref, active_mask, g, hm, rho, [bool(s) for s in soft]
    )
    print(f'{name:<25s} {max_t:7.2f} {max_i:7.0f} {max_p:7.3f} {max_h:7.3f} {fails:6d} {min_t:7.2f}')
