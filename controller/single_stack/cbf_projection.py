"""
CBF-based Projection Operator for Single-Stack AWE System

Implements rigorous Control Barrier Function (CBF) constraints with explicit
Lie derivative computation: dh/dt + gamma * h >= 0

Based on: 单槽碱性水电解槽(AWE)系统：非线性非仿射CBF推导（修正版）
"""

import numpy as np
import casadi as ca


class SingleStackCBFProjection:
    """
    CBF projection operator with explicit Lie derivative computation.
    Uses CasADi symbolic differentiation and IPOPT for optimization.

    CBF Condition: L_f h(x,u) + gamma * h(x) >= 0
    """

    def __init__(self, dt=60.0, sim_dt=0.2, gamma=1.0, epsilon=1e-4, rho=1e12,
                 gamma_vec=None, rho_vec=None, h_scales=None, lie_deriv_scales=None, normalize=True,
                 h_margin_vec=None, lambda_u_scale=1000.0, soft_mask=None, active_mask=None):
        """
        Args:
            dt: Control timestep
            sim_dt: Simulation timestep
            gamma: Default gamma (used if gamma_vec not provided)
            epsilon: Numerical differentiation step size (kept for API compatibility)
            rho: Default rho (used if rho_vec not provided)
            gamma_vec: List of gamma values for each CBF constraint [h_T, h_HTO, h_V, h_P, h_Tmin]
            rho_vec: List of rho values for each CBF constraint [h_T, h_HTO, h_V, h_P, h_Tmin]
            h_scales: List of h scales for normalization
            lie_deriv_scales: List of L_f h scales for normalization
            normalize: Whether to use normalized CBF constraints
            h_margin_vec: List of safety margins for each constraint [h_T, h_HTO, h_V, h_P, h_Tmin]
                             (e.g., 0.002 for HTO means effective limit becomes 2% - 0.2% = 1.8%)
            lambda_u_scale: Scale factor for control change penalty (default: 1000.0).
                             Larger values enforce smoother control transitions.
            soft_mask: List of bool indicating whether each constraint uses soft slack
                       [h_T, h_HTO, h_V, h_P, h_Tmin]. Default: all True.
            active_mask: List of bool indicating whether each constraint is active
                         [h_T, h_HTO, h_V, h_P, h_Tmin]. Default: all True.
                         Can be overridden per-call in project().
        """
        self.dt = dt
        self.sim_dt = sim_dt
        self.gamma = gamma
        self.epsilon = epsilon
        self.normalize = normalize

        # Per-constraint gamma values (default: all use gamma)
        if gamma_vec is None:
            self.gamma_vec = np.array([50.0, 0.3, 100.0, 100.0, 10.0])
        else:
            self.gamma_vec = np.array(gamma_vec)

        # Per-constraint rho values for soft constraints (default: all use rho)
        if rho_vec is None:
            # Default: balanced soft constraint penalties
            # h_HTO gets moderately higher penalty as it's the most critical constraint
            self.rho_vec = np.array([1000, 10000, 1000, 1000, 1000])
        else:
            self.rho_vec = np.array(rho_vec)

        # Per-constraint safety margins (default: no margin)
        if h_margin_vec is None:
            self.h_margin_vec = np.array([0.5, 0.010, 0.0, 0.0, 0.0])
        else:
            self.h_margin_vec = np.array(h_margin_vec)

        # Per-constraint soft slack mask (default: all soft)
        if soft_mask is None:
            self.soft_mask = np.array([True, True, True, True, True])
        else:
            self.soft_mask = np.array(soft_mask, dtype=bool)

        # Per-constraint active mask (default: all active)
        if active_mask is None:
            self.active_mask = np.array([True, True, True, True, True])
        else:
            self.active_mask = np.array(active_mask, dtype=bool)

        # Normalization scales
        if h_scales is None:
            # Default: h_T ~ 70K, h_HTO ~ 0.02, h_V ~ 0.5V, h_P ~ 5e6 W, h_Tmin ~ 70K
            self.h_scales = np.array([70.0, 0.02, 0.5, 5e6, 70.0])
        else:
            self.h_scales = np.array(h_scales)

        if lie_deriv_scales is None:
            # Default: L_f h_T ~ 1e-2, L_f h_HTO ~ 1e-3, L_f h_V ~ 1e-5, L_f h_P ~ 1e0, L_f h_Tmin ~ 1e-2
            self.lie_deriv_scales = np.array([1e-2, 1e-3, 1e-5, 1e0, 1e-2])
        else:
            self.lie_deriv_scales = np.array(lie_deriv_scales)

        n_sub = int(dt / sim_dt) if sim_dt > 0 else 1
        self.n_sub = max(1, n_sub)
        self.dt_sub = dt / self.n_sub

        # System Parameters
        self.n_cells = 368
        self.A_cell = 2.0
        self.I_rated = 7800.0
        self.U_rev = 1.229
        self.U_th = 1.481
        self.r1 = 3.202e-5
        self.r2 = 8.970e-8
        self.r3 = -4.193e-12
        self.P_sys = 1.8e6
        self.delta_P = 0.01 * self.P_sys
        self.s = 7.572e-2
        self.t1 = -1.070e-1
        self.t2 = 14.43
        self.t3 = 38.8
        self.F = 96485.0

        # Thermal Parameters
        self.C_s_i = 3.450e7
        self.C_sep = 1e7
        self.C_he = 5e6
        self.C_c = 5e6
        self.T_amb = 298.0
        self.T_c_in = 288.0
        self.k_he = 960.0
        self.A_he = 240.0
        self.c_cw = 4200.0
        self.rho_cw = 1000.0
        self.c_lye = 3200.0
        self.rho_lye = 1280.0
        self.R = 8.314
        self.V_sep = 10.288
        self.V_sep_gas = 0.6 * self.V_sep

        # Radiation/Convection
        self.segma_s = 1000.0
        self.segma_sep = 50.0
        self.A_stack = 80.0
        self.A_sep = 40.0
        self.epsilon_stack = 0.8
        self.epsilon_sep = 0.8
        self.sigma_b = 5.67e-8

        # HTO Parameters
        self.V_an_lye = 2.5
        self.delta = 500e-6
        self.D_eff = 8.569e-10
        self.K_eff = 2e-16
        self.tau_sep = 60.0
        self.mu_lye = 8.76e-4
        self.S_H2_lye = self._calculate_h2_solubility()
        self.K_HTO = self.R / (self.P_sys * self.V_sep_gas)

        # Constraints
        self.I_min, self.I_max = 0.0, 7800.0 * 1.2
        self.v_lye_min, self.v_lye_max = 0.01, 0.1
        self.v_c_min, self.v_c_max = 0.0, 1.0
        self.T_min, self.T_max = 293.15, 363.15
        self.P_stack_max = 6.0e6
        self.U_cell_max = 2.2
        self.HTO_max = 0.02

        # Control change penalty weights (normalized by nominal deltas for balanced scaling)
        self.u_nominal = np.array([2000.0, 0.02, 0.2])
        self.lambda_u = lambda_u_scale / (self.u_nominal ** 2)

        # Build symbolic CasADi functions for h and f
        self._build_casadi_functions()

    def _calculate_h2_solubility(self):
        """Calculate H2 Solubility in Lye (S_H2) [mol/(m^3 Pa)] - exact match with simulator"""
        rho_H2O = 1000.0
        M_H2O = 18e-3
        p_atm = 101325.0
        H_H2 = 7.1698e4 * p_atm
        K_H2 = 3.14
        w_lye = 0.30
        S_H2_H2O = rho_H2O * self.P_sys / (M_H2O * p_atm * H_H2)
        return S_H2_H2O / (10**(K_H2 * w_lye))

    def _build_casadi_functions(self):
        """Build symbolic CasADi functions for h(x,u) and f(x,u)."""
        state = ca.MX.sym('state', 7)
        control = ca.MX.sym('control', 3)

        T_s = state[1]
        T_sep = state[2]
        n_gas = state[6]
        I = control[0]

        # --- Symbolic _cell_voltage ---
        T_C = T_s - 273.15
        R_ohm = self.r1 + self.r2 * T_s + self.r3 * self.P_sys
        V_ohm = R_ohm * I
        term_act = self.t1 + self.t2 / T_C + self.t3 / (T_C**2)
        arg = term_act * I + 1.0
        arg_safe = ca.fmax(arg, 1e-9)
        V_act = self.s * ca.log(arg_safe)
        U_cell = self.U_rev + V_ohm + V_act
        U_cell = ca.fmax(U_cell, self.U_rev)

        Power = U_cell * I * self.n_cells
        hto = (n_gas * self.R * T_sep) / (self.P_sys * self.V_sep_gas)

        h = ca.vertcat(
            self.T_max - T_s - self.h_margin_vec[0],
            self.HTO_max - hto - self.h_margin_vec[1],
            self.U_cell_max - U_cell - self.h_margin_vec[2],
            self.P_stack_max - Power - self.h_margin_vec[3],
            T_s - self.T_min - self.h_margin_vec[4]
        )

        # --- Symbolic _compute_f ---
        T_s_in = state[0]
        T_c_out = state[3]
        n_H2_an = state[4]
        n_liq = state[5]
        v_lye = control[1]
        v_c = control[2]

        f1 = 50.0 + 2.5 * T_C
        f2 = 0.92 - 6.25e-6 * T_C
        I_sq = I**2
        eta = (I_sq / (f1 + I_sq)) * f2

        Q_ele = self.n_cells * I * (U_cell - eta * self.U_th)

        # Stack thermal
        Q_conv = self.segma_s * (T_s - self.T_amb)
        Q_rad = self.epsilon_stack * self.sigma_b * self.A_stack * (T_s**4 - self.T_amb**4)
        Q_diss = Q_conv + Q_rad
        Q_flow = self.c_lye * self.rho_lye * v_lye * (T_s - T_s_in)
        dT_s_dt = (Q_ele - Q_diss - Q_flow) / self.C_s_i

        # Separator thermal
        T_sep_in = T_s
        Q_sep_rad = self.epsilon_sep * self.sigma_b * self.A_sep * (T_sep**4 - self.T_amb**4)
        Q_sep_conv = self.segma_sep * (T_sep - self.T_amb)
        Q_sep_diss = Q_sep_conv + Q_sep_rad
        dT_sep_dt = (0.5 * self.c_lye * self.rho_lye * v_lye * (T_sep_in - T_sep) - Q_sep_diss) / self.C_sep

        # Heat exchanger thermal (LMTD)
        C_rate_lye = self.c_lye * self.rho_lye * v_lye
        C_rate_cw = self.c_cw * self.rho_cw * v_c
        delta_T1 = T_s_in - T_c_out
        delta_T2 = T_sep - self.T_c_in

        # Smooth LMTD using CasADi conditionals
        LMTD = ca.if_else(
            ca.fabs(delta_T1 - delta_T2) < 1e-5,
            delta_T1,
            ca.if_else(
                delta_T1 * delta_T2 <= 0,
                0.0,
                (delta_T1 - delta_T2) / ca.log(delta_T1 / delta_T2)
            )
        )

        Q_hx = self.k_he * self.A_he * LMTD
        dT_s_in_dt = (C_rate_lye * (T_sep - T_s_in) - Q_hx) / self.C_he
        dT_c_out_dt = (C_rate_cw * (self.T_c_in - T_c_out) + Q_hx) / self.C_c

        # HTO derivatives
        n_dot_O2_prod = self.n_cells * I * eta / (4 * self.F)

        n_dot_H2_lye = self.S_H2_lye * self.rho_lye * v_lye / 4.0
        n_dot_H2_diff = self.A_cell * self.n_cells * self.D_eff * self.S_H2_lye * self.P_sys / self.delta
        n_dot_H2_conv = self.A_cell * self.n_cells * (self.K_eff / self.mu_lye) * self.S_H2_lye * self.rho_lye * (self.delta_P / self.delta)
        n_dot_H2_im = n_dot_H2_lye + n_dot_H2_diff + n_dot_H2_conv

        n_dot_H2_im_1 = n_H2_an * v_lye / (2 * self.V_an_lye)
        dn_H2_an_dt = n_dot_H2_im - n_dot_H2_im_1

        n_dot_H2_2 = n_liq / self.tau_sep
        dn_liq_dt = n_dot_H2_im_1 - n_dot_H2_2

        sum_n_dot_O2 = n_dot_O2_prod
        n_dot_H2_out = ca.if_else(
            sum_n_dot_O2 > 1e-9,
            (self.R * T_sep * n_gas * sum_n_dot_O2) / (self.P_sys * self.V_sep_gas),
            0.0
        )

        dn_gas_dt = n_dot_H2_2 - n_dot_H2_out

        f = ca.vertcat(
            dT_s_in_dt, dT_s_dt, dT_sep_dt, dT_c_out_dt,
            dn_H2_an_dt, dn_liq_dt, dn_gas_dt
        )

        self._ca_h_fn = ca.Function('h_fn', [state, control], [h])
        self._ca_f_fn = ca.Function('f_fn', [state, control], [f])

        # Lie derivative: dh/dx * f
        dh_dx = ca.jacobian(h, state)
        lie_deriv = ca.mtimes(dh_dx, f)
        self._ca_lie_deriv_fn = ca.Function('lie_deriv_fn', [state, control], [lie_deriv, h, f])

    def _compute_h(self, state, control):
        """Compute CBF function values h(x) with safety margins"""
        state = np.asarray(state).flatten()
        control = np.asarray(control).flatten()
        h = self._ca_h_fn(state, control).full().flatten()
        return h

    def _compute_f(self, state, control):
        """
        Compute dynamics f(x,u) for Lie derivative.
        Exact match with SingleStackSimulator._get_derivatives.
        """
        state = np.asarray(state).flatten()
        control = np.asarray(control).flatten()
        f = self._ca_f_fn(state, control).full().flatten()
        return f

    def _compute_lie_derivative(self, state, control):
        """
        Compute Lie derivative L_f h = dh/dx * f(x,u) symbolically via CasADi.
        """
        state = np.asarray(state).flatten()
        control = np.asarray(control).flatten()
        lie_deriv, h, f = self._ca_lie_deriv_fn(state, control)
        lie_deriv = lie_deriv.full().flatten()
        h = h.full().flatten()
        f = f.full().flatten()
        # grad_h not used downstream, but kept for API compatibility
        grad_h = np.zeros((len(h), len(state)))
        return lie_deriv, h, grad_h, f

    def project(self, u_ref, state, u_last=None, verbose=False, active_mask=None):
        """
        Project reference control using CBF with soft constraints.

        Soft CBF formulation:
            min ||u - u_ref||^2 + lambda * ||u - u_last||^2 + rho * ||s||^2
            s.t. L_f h_i + gamma * h_i >= -s_i,  s_i >= 0   (for active constraints)
                 u_min <= u <= u_max

        Args:
            u_ref: Reference control [I, v_lye, v_c]
            state: Current state [8-dim]
            u_last: Previous control [I, v_lye, v_c] (for smoothness penalty)
            verbose: Print diagnostics
            active_mask: Optional bool array [h_T, h_HTO, h_V, h_P, h_Tmin] overriding
                         self.active_mask for this call only.

        Returns:
            u_safe: Projected control
            success: Whether projection succeeded (with slack)
            info: Dict with CBF values, derivatives, and slack values
        """
        u_ref = np.asarray(u_ref).flatten()
        state = np.asarray(state).flatten()
        u_last = np.asarray(u_last).flatten() if u_last is not None else u_ref.copy()

        # Resolve active mask for this call
        if active_mask is None:
            active_mask = self.active_mask.copy()
        else:
            active_mask = np.array(active_mask, dtype=bool)

        # Compute current CBF values and Lie derivatives at reference
        lie_deriv_ref, h_ref, grad_h_ref, f_ref = self._compute_lie_derivative(state, u_ref)

        # Normalize if enabled
        if self.normalize:
            lie_deriv_ref_norm = lie_deriv_ref / np.maximum(np.abs(self.lie_deriv_scales), 1e-10)
            h_ref_norm = h_ref / self.h_scales
            lie_deriv_ref = lie_deriv_ref_norm
            h_ref = h_ref_norm

        if self.normalize:
            cbf_ref = lie_deriv_ref + self.gamma_vec * h_ref
        else:
            cbf_ref = lie_deriv_ref + self.gamma * h_ref

        if verbose:
            print(f"Reference point:")
            print(f"  h = {h_ref}")
            print(f"  L_f h = {lie_deriv_ref}")
            print(f"  active_mask = {active_mask}")
            print(f"  L_f h + gamma*h = {cbf_ref}")

        # Check if reference already satisfies active CBF constraints
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

        # CBF constraint optimization: only active constraints participate
        n_cbf_total = 5
        active_soft_mask = self.soft_mask & active_mask
        active_hard_mask = (~self.soft_mask) & active_mask
        n_soft = int(np.sum(active_soft_mask))
        n_active = int(np.sum(active_mask))

        # Normalized control weights using nominal deltas for balanced scaling
        u_weights = 1.0 / (self.u_nominal ** 2)

        # Build CasADi NLP
        opti = ca.Opti()
        u_var = opti.variable(3)
        if n_soft > 0:
            s_var = opti.variable(n_soft)
        else:
            s_var = None

        # Objective
        ref_cost = ca.sum1(u_weights * (u_var - u_ref)**2)
        delta_cost = ca.sum1(self.lambda_u * (u_var - u_last)**2)
        if n_soft > 0:
            # Only penalize slack for active soft constraints
            rho_active_soft = self.rho_vec[active_soft_mask]
            slack_cost = ca.sum1(rho_active_soft * s_var**2)
        else:
            slack_cost = 0.0
        opti.minimize(ref_cost + delta_cost + slack_cost)

        # Bounds
        opti.subject_to(u_var[0] >= self.I_min)
        opti.subject_to(u_var[0] <= self.I_max)
        opti.subject_to(u_var[1] >= self.v_lye_min)
        opti.subject_to(u_var[1] <= self.v_lye_max)
        opti.subject_to(u_var[2] >= self.v_c_min)
        opti.subject_to(u_var[2] <= self.v_c_max)
        if n_soft > 0:
            opti.subject_to(s_var >= 0)

        # Relaxed rate constraints: allow faster current changes to meet safety constraints
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

        # Symbolic CBF constraints (compute all 5, but only add active ones)
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

        # Initial guess
        u0 = np.clip(u_last, [self.I_min, self.v_lye_min, self.v_c_min],
                    [self.I_max, self.v_lye_max, self.v_c_max])

        # If reference violates active CBF, search for feasible initial guess
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

        # Initial slack only for active soft constraints
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
            'acceptable_tol': 1e-4
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

        # Parse slack back to 5-dim: inactive/hard constraints get 0
        s_opt_full = np.zeros(n_cbf_total)
        if n_soft > 0:
            s_opt_full[active_soft_mask] = s_opt_raw
            max_slack = np.max(s_opt_raw) if len(s_opt_raw) > 0 else 0.0
        else:
            max_slack = 0.0

        lie_deriv_opt, h_opt, _, _ = self._compute_lie_derivative(state, u_opt)

        # Store raw (unnormalized) values for info
        lie_deriv_opt_raw = lie_deriv_opt.copy()
        h_opt_raw = h_opt.copy()

        # Normalize for CBF check if enabled
        if self.normalize:
            lie_deriv_opt = lie_deriv_opt / np.maximum(np.abs(self.lie_deriv_scales), 1e-10)
            h_opt = h_opt / self.h_scales
            cbf_opt = lie_deriv_opt + self.gamma_vec * h_opt
        else:
            cbf_opt = lie_deriv_opt + self.gamma * h_opt

        # Actual constraint violation (only meaningful for active constraints)
        slack_violation = np.maximum(0, -cbf_opt)
        slack_violation[~active_mask] = 0.0

        # Feasibility override (only check active constraints)
        u_min_arr = np.array([self.I_min, self.v_lye_min, self.v_c_min])
        u_max_arr = np.array([self.I_max, self.v_lye_max, self.v_c_max])
        control_feasible = (
            np.all(u_opt >= u_min_arr - 1e-6) and
            np.all(u_opt <= u_max_arr + 1e-6)
        )
        cbf_feasible = np.all(cbf_opt[active_mask] >= -1e-3)
        slack_feasible = np.all(s_opt_full[active_soft_mask] >= -1e-6)
        hard_mask_active = active_hard_mask
        hard_feasible = np.all(cbf_opt[hard_mask_active] >= -1e-3) if np.any(hard_mask_active) else True
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

        if verbose:
            print(f"Projected point:")
            print(f"  u = {u_opt}")
            print(f"  s = {s_opt_full}")
            print(f"  h = {h_opt}")
            print(f"  active_mask = {active_mask}")
            print(f"  L_f h + gamma*h = {cbf_opt}")
            print(f"  Max slack: {max_slack:.4f}")
            if not solver_success and actual_success:
                print(f"  [Note: IPOPT reported failure but solution is feasible - accepted]")

        return u_opt, actual_success, info


# Alias for backward compatibility
SingleStackCBFProjectionSimplified = SingleStackCBFProjection
