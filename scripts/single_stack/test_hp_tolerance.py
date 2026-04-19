"""
测试放宽 solver tolerance 和 cbf_feasible tolerance 对 high_power failures 的影响。
"""
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from controller.single_stack.cbf_projection import SingleStackCBFProjection
from plant.single_stack_simulator import SingleStackSimulator
import casadi as ca


class TolerantCBFProjection(SingleStackCBFProjection):
    """CBF projection with more tolerant solver settings and feasibility checks."""

    def __init__(self, *args, acceptable_tol=1e-4, cbf_feasible_tol=1e-3, **kwargs):
        self.acceptable_tol = acceptable_tol
        self.cbf_feasible_tol = cbf_feasible_tol
        super().__init__(*args, **kwargs)

    def project(self, u_ref, state, u_last=None, verbose=False, active_mask=None):
        u_ref = np.asarray(u_ref).flatten()
        state = np.asarray(state).flatten()
        u_last = np.asarray(u_last).flatten() if u_last is not None else u_ref.copy()

        if active_mask is None:
            active_mask = self.active_mask.copy()
        else:
            active_mask = np.array(active_mask, dtype=bool)

        lie_deriv_ref, h_ref, grad_h_ref, f_ref = self._compute_lie_derivative(state, u_ref)

        if self.normalize:
            lie_deriv_ref_norm = lie_deriv_ref / np.maximum(np.abs(self.lie_deriv_scales), 1e-10)
            h_ref_norm = h_ref / self.h_scales
            lie_deriv_ref = lie_deriv_ref_norm
            h_ref = h_ref_norm

        if self.normalize:
            cbf_ref = lie_deriv_ref + self.gamma_vec * h_ref
        else:
            cbf_ref = lie_deriv_ref + self.gamma * h_ref

        if np.all(cbf_ref[active_mask] >= 0):
            info = {
                'h': h_ref,
                'lie_deriv': lie_deriv_ref,
                'cbf': cbf_ref,
                'projection_needed': False,
                'slack': np.zeros(5),
                'active_mask': active_mask.copy()
            }
            return u_ref, True, info

        n_cbf_total = 5
        active_soft_mask = self.soft_mask & active_mask
        active_hard_mask = (~self.soft_mask) & active_mask
        n_soft = int(np.sum(active_soft_mask))

        u_weights = 1.0 / (self.u_nominal ** 2)

        opti = ca.Opti()
        u_var = opti.variable(3)
        if n_soft > 0:
            s_var = opti.variable(n_soft)
        else:
            s_var = None

        ref_cost = ca.sum1(u_weights * (u_var - u_ref)**2)
        delta_cost = ca.sum1(self.lambda_u * (u_var - u_last)**2)
        if n_soft > 0:
            rho_active_soft = self.rho_vec[active_soft_mask]
            slack_cost = ca.sum1(rho_active_soft * s_var**2)
        else:
            slack_cost = 0.0
        opti.minimize(ref_cost + delta_cost + slack_cost)

        opti.subject_to(u_var[0] >= self.I_min)
        opti.subject_to(u_var[0] <= self.I_max)
        opti.subject_to(u_var[1] >= self.v_lye_min)
        opti.subject_to(u_var[1] <= self.v_lye_max)
        opti.subject_to(u_var[2] >= self.v_c_min)
        opti.subject_to(u_var[2] <= self.v_c_max)
        if n_soft > 0:
            opti.subject_to(s_var >= 0)

        delta_I_rise = 1000.0
        delta_I_fall = 1000.0
        delta_v_lye_rise = 0.02
        delta_v_lye_fall = 0.02
        delta_v_c_rise = 0.3
        delta_v_c_fall = 0.3
        opti.subject_to(u_var[0] >= u_last[0] - delta_I_fall)
        opti.subject_to(u_var[0] <= u_last[0] + delta_I_rise)
        opti.subject_to(u_var[1] >= u_last[1] - delta_v_lye_fall)
        opti.subject_to(u_var[1] <= u_last[1] + delta_v_lye_rise)
        opti.subject_to(u_var[2] >= u_last[2] - delta_v_c_fall)
        opti.subject_to(u_var[2] <= u_last[2] + delta_v_c_rise)

        lie_deriv_sym, h_sym, _ = self._ca_lie_deriv_fn(state, u_var)
        if self.normalize:
            lie_deriv_sym = lie_deriv_sym / np.maximum(np.abs(self.lie_deriv_scales), 1e-10)
            h_sym = h_sym / self.h_scales
            cbf_sym = lie_deriv_sym + self.gamma_vec * h_sym
        else:
            cbf_sym = lie_deriv_sym + self.gamma * h_sym

        s_idx = 0
        for j in range(n_cbf_total):
            if not active_mask[j]:
                continue
            if self.soft_mask[j]:
                opti.subject_to(cbf_sym[j] + s_var[s_idx] >= 0)
                s_idx += 1
            else:
                opti.subject_to(cbf_sym[j] >= 0)

        u0 = np.clip(u_last, [self.I_min, self.v_lye_min, self.v_c_min],
                    [self.I_max, self.v_lye_max, self.v_c_max])

        if not np.all(cbf_ref[active_mask] >= 0):
            n_search = 100
            I_candidates_up = np.linspace(u0[0], self.I_max, n_search)
            I_candidates_down = np.linspace(u0[0], self.I_min, n_search)
            found_feasible = False

            def check_cbf_feasibility(I_test, v_lye_test=u0[1], v_c_test=u0[2]):
                u_test = np.array([I_test, v_lye_test, v_c_test])
                ld_test, h_test, _, _ = self._compute_lie_derivative(state, u_test)
                if self.normalize:
                    ld_test = ld_test / np.maximum(np.abs(self.lie_deriv_scales), 1e-10)
                    h_test = h_test / self.h_scales
                    cbf_test = ld_test + self.gamma_vec * h_test
                else:
                    cbf_test = ld_test + self.gamma * h_test
                return np.all(cbf_test[active_mask] >= 0)

            for I_test in I_candidates_up:
                if check_cbf_feasibility(I_test):
                    u0[0] = I_test
                    found_feasible = True
                    break

            if not found_feasible:
                for I_test in I_candidates_down:
                    if check_cbf_feasibility(I_test):
                        u0[0] = I_test
                        found_feasible = True
                        break

            if not found_feasible:
                I_grid = np.concatenate([I_candidates_up, I_candidates_down])
                v_lye_grid = np.linspace(self.v_lye_min, self.v_lye_max, 30)
                v_c_grid = np.linspace(self.v_c_min, self.v_c_max, 10)

                for v_lye_test in v_lye_grid:
                    for I_test in I_grid:
                        if check_cbf_feasibility(I_test, v_lye_test, u0[2]):
                            u0[0] = I_test
                            u0[1] = v_lye_test
                            found_feasible = True
                            break
                    if found_feasible:
                        break

                if not found_feasible:
                    for v_c_test in v_c_grid:
                        for v_lye_test in v_lye_grid:
                            for I_test in I_grid:
                                if check_cbf_feasibility(I_test, v_lye_test, v_c_test):
                                    u0[0] = I_test
                                    u0[1] = v_lye_test
                                    u0[2] = v_c_test
                                    found_feasible = True
                                    break
                            if found_feasible:
                                break
                        if found_feasible:
                            break

            if not found_feasible:
                u0 = np.clip(u_ref, [self.I_min, self.v_lye_min, self.v_c_min],
                            [self.I_max, self.v_lye_max, self.v_c_max])

        if n_soft > 0:
            s0 = np.maximum(0, -(cbf_ref[active_soft_mask]))
        else:
            s0 = np.array([])

        opti.set_initial(u_var, u0)
        if n_soft > 0:
            opti.set_initial(s_var, s0)

        s_opts = {
            'print_level': 0,
            'sb': 'yes',
            'max_iter': 300,
            'tol': 1e-6,
            'acceptable_tol': self.acceptable_tol
        }
        if verbose:
            s_opts['print_level'] = 5
            s_opts.pop('sb', None)
        opti.solver('ipopt', {}, s_opts)

        try:
            sol = opti.solve()
            u_opt = sol.value(u_var)
            if n_soft > 0:
                s_opt_raw = sol.value(s_var)
            else:
                s_opt_raw = np.array([])
            solver_success = True
            solver_message = "Solve_Succeeded"
        except RuntimeError as e:
            u_opt = opti.debug.value(u_var)
            if u_opt is None or not np.isfinite(u_opt).all():
                u_opt = u0
            if n_soft > 0:
                s_opt_raw = opti.debug.value(s_var)
            else:
                s_opt_raw = np.array([])
            solver_success = False
            solver_message = str(e)

        if u_opt is None or not np.isfinite(u_opt).all():
            u_opt = u0
        if s_opt_raw is None:
            s_opt_raw = np.array([])
        s_opt_raw = np.asarray(s_opt_raw).flatten()

        s_opt_full = np.zeros(n_cbf_total)
        if n_soft > 0:
            s_opt_full[active_soft_mask] = s_opt_raw
            max_slack = np.max(s_opt_raw) if len(s_opt_raw) > 0 else 0.0
        else:
            max_slack = 0.0

        lie_deriv_opt, h_opt, _, _ = self._compute_lie_derivative(state, u_opt)
        lie_deriv_opt_raw = lie_deriv_opt.copy()
        h_opt_raw = h_opt.copy()

        if self.normalize:
            lie_deriv_opt = lie_deriv_opt / np.maximum(np.abs(self.lie_deriv_scales), 1e-10)
            h_opt = h_opt / self.h_scales
            cbf_opt = lie_deriv_opt + self.gamma_vec * h_opt
        else:
            cbf_opt = lie_deriv_opt + self.gamma * h_opt

        slack_violation = np.maximum(0, -cbf_opt)
        slack_violation[~active_mask] = 0.0

        u_min_arr = np.array([self.I_min, self.v_lye_min, self.v_c_min])
        u_max_arr = np.array([self.I_max, self.v_lye_max, self.v_c_max])
        control_feasible = (
            np.all(u_opt >= u_min_arr - 1e-6) and
            np.all(u_opt <= u_max_arr + 1e-6)
        )
        cbf_feasible = np.all(cbf_opt[active_mask] >= -self.cbf_feasible_tol)
        slack_feasible = np.all(s_opt_full[active_soft_mask] >= -1e-6)
        hard_mask_active = active_hard_mask
        hard_feasible = np.all(cbf_opt[hard_mask_active] >= -self.cbf_feasible_tol) if np.any(hard_mask_active) else True
        actual_success = solver_success or (control_feasible and cbf_feasible and slack_feasible and hard_feasible)

        info = {
            'h': h_opt_raw,
            'h_norm': h_opt if self.normalize else None,
            'lie_deriv': lie_deriv_opt_raw,
            'lie_deriv_norm': lie_deriv_opt if self.normalize else None,
            'cbf': cbf_opt,
            'projection_needed': True,
            'adjustment': np.linalg.norm(u_opt - u_ref),
            'slack': s_opt_full,
            'max_slack': max_slack,
            'slack_violation': slack_violation,
            'optimization_success': actual_success,
            'solver_message': solver_message,
            'active_mask': active_mask.copy()
        }

        return u_opt, actual_success, info


def run_fast(duration, projector, u_ref, active_mask):
    sim = SingleStackSimulator(sim_dt=0.2)
    sim.reset()
    dt = 0.2
    steps = int(duration / dt)
    ctrl_steps = int(60.0 / dt)
    current_action = u_ref.copy()
    last_action = current_action.copy()

    max_temp = max_current = max_power = 0.0
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

        max_temp = max(max_temp, T_s - 273.15)
        max_current = max(max_current, I)
        max_power = max(max_power, Power)

        sim.step(current_action)

    return max_temp, max_current, max_power, n_failures


def main():
    print("=" * 100)
    print("HIGH POWER - 放宽 tolerance 对 failures 的影响")
    print("=" * 100)

    active_mask = np.array([True, False, True, True, True])
    u_ref = np.array([9360.0, 0.03, 1.0])
    gamma_vec = [50, 0.3, 100, 100, 10]
    h_margin_vec = [0.5, 0.01, 0, 0, 0]
    rho_vec = [50000, 10000, 50000, 50000, 10000]
    soft_mask = [False, True, False, False, True]

    configs = [
        ("原始设置 (acceptable=1e-4, cbf_tol=1e-3)", 1e-4, 1e-3),
        ("acceptable=1e-2, cbf_tol=1e-3", 1e-2, 1e-3),
        ("acceptable=1e-1, cbf_tol=1e-3", 1e-1, 1e-3),
        ("acceptable=1e-2, cbf_tol=1e-1", 1e-2, 1e-1),
        ("acceptable=1e-1, cbf_tol=1e-1", 1e-1, 1e-1),
        ("acceptable=1e-1, cbf_tol=1e-2", 1e-1, 1e-2),
    ]

    print(f"\n{'名称':<45s} {'MaxT':>7s} {'MaxI':>7s} {'MaxP':>7s} {'Fail':>6s}")
    print("-" * 100)

    for name, acc_tol, cbf_tol in configs:
        projector = TolerantCBFProjection(
            dt=60.0, gamma_vec=gamma_vec, rho_vec=rho_vec,
            h_margin_vec=h_margin_vec, normalize=True, lambda_u_scale=1000.0,
            soft_mask=soft_mask, active_mask=active_mask,
            acceptable_tol=acc_tol, cbf_feasible_tol=cbf_tol
        )
        max_t, max_i, max_p, fails = run_fast(7200, projector, u_ref, active_mask)
        print(f"{name:<45s} {max_t:7.2f} {max_i:7.0f} {max_p:7.3f} {fails:6d}")


if __name__ == "__main__":
    main()
