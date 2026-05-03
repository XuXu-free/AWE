"""
Higher-Order CBF-based Projection Operator for Multi-Stack AWE System

Implements mixed-order Control Barrier Functions:
- First-order CBF for Tmax (x4) and Tmin (x4) constraints
- Second-order HOCBF for HTO constraint

Second-order HTO CBF formulation:
    psi_0 = h
    psi_1 = L_f h + alpha1 * h
    psi_2 = L_f^2 h + alpha2 * psi_1 >= 0

where L_f^2 h = jacobian(L_f h, x) @ f(x,u)

See docs/hocbf_hto_derivation.md for full mathematical derivation.
"""

import numpy as np
import casadi as ca


class MultiStackCBFProjectionHO:
    """
    Mixed-order CBF projection operator.
    HTO uses second-order CBF; temperature uses first-order CBF.
    """

    def __init__(self, dt=60.0, sim_dt=0.2, gamma=1.0, epsilon=1e-4, rho=1e12,
                 gamma_vec=None, rho_vec=None, h_scales=None, lie_deriv_scales=None, normalize=True,
                 h_margin_vec=None, lambda_u_scale=1000.0, soft_mask=None,
                 alpha1_hto=0.8, alpha2_hto=2.0,
                 alpha1_vec=None, alpha2_vec=None,
                 u_weight_scale=None):
        """
        Args:
            dt: Control timestep
            sim_dt: Simulation timestep
            gamma: Default gamma (used if gamma_vec not provided)
            epsilon: Numerical differentiation step size (unused, kept for API compat)
            rho: Default rho (used if rho_vec not provided)
            gamma_vec: List of gamma values for each first-order CBF constraint (fallback if alpha2_vec not set)
            rho_vec: List of rho values for each CBF constraint
            h_scales: List of h scales for normalization
            lie_deriv_scales: List of L_f h scales for normalization
            normalize: Whether to use normalized CBF constraints
            h_margin_vec: List of safety margins for each constraint
            lambda_u_scale: Scale factor for control change penalty (default: 1000.0).
            soft_mask: List of bool indicating whether each constraint uses soft slack
            alpha1_hto: First-order decay rate for HTO HOCBF (default: 0.8)
            alpha2_hto: Second-order damping for HTO HOCBF (default: 2.0)
            alpha1_vec: Per-constraint alpha1 (9 elements). If None, uses gamma_vec for 1st-order and alpha1_hto for HTO.
            alpha2_vec: Per-constraint alpha2 (9 elements). If None, uses 0 for 1st-order and alpha2_hto for HTO.
            u_weight_scale: Per-control weight scale for reference tracking cost (9 elements).
                Larger value = more reluctant to deviate from reference. Default: all 1.0.
                Use case: increase I weights (e.g., 10.0) and decrease v_c weight (e.g., 0.1)
                to prioritize coolant flow adjustment over current adjustment.
        """
        self.dt = dt
        self.sim_dt = sim_dt
        self.gamma = gamma
        self.epsilon = epsilon
        self.normalize = normalize
        self.alpha1_hto = alpha1_hto
        self.alpha2_hto = alpha2_hto

        # Per-constraint gamma values (9 constraints: Tmax x4, HTO x1, Tmin x4)
        if gamma_vec is None:
            defaults = [10.0] * 4 + [gamma] + [10.0] * 4
            self.gamma_vec = np.array(defaults)
        else:
            self.gamma_vec = np.array(gamma_vec)

        # Per-constraint alpha values for HOCBF
        if alpha1_vec is None:
            self.alpha1_vec = np.array(list(self.gamma_vec[:4]) + [alpha1_hto] + list(self.gamma_vec[5:9]))
        else:
            self.alpha1_vec = np.array(alpha1_vec)

        if alpha2_vec is None:
            self.alpha2_vec = np.array([0.0] * 4 + [alpha2_hto] + [0.0] * 4)
        else:
            self.alpha2_vec = np.array(alpha2_vec)

        # Per-constraint rho values for soft constraints
        if rho_vec is None:
            defaults = [1000.0] * 4 + [1e5] + [1000.0] * 4
            self.rho_vec = np.array(defaults)
        else:
            self.rho_vec = np.array(rho_vec)

        # Per-constraint safety margins (default: no margin)
        if h_margin_vec is None:
            self.h_margin_vec = np.array([0.0]*4 + [0.007] + [0.0]*4)
        else:
            self.h_margin_vec = np.array(h_margin_vec)

        # Per-constraint soft slack mask (default: all soft)
        if soft_mask is None:
            self.soft_mask = np.full(9, True, dtype=bool)
        else:
            self.soft_mask = np.array(soft_mask, dtype=bool)

        # Normalization scales
        if h_scales is None:
            defaults = [70.0] * 4 + [0.02] + [70.0] * 4
            self.h_scales = np.array(defaults)
        else:
            self.h_scales = np.array(h_scales)

        if lie_deriv_scales is None:
            defaults = [1e-2] * 4 + [1e-3] + [1e-2] * 4
            self.lie_deriv_scales = np.array(defaults)
        else:
            self.lie_deriv_scales = np.array(lie_deriv_scales)

        n_sub = int(dt / sim_dt) if sim_dt > 0 else 1
        self.n_sub = max(1, n_sub)
        self.dt_sub = dt / self.n_sub

        # System Parameters (MultiStack defaults)
        self.N = 4
        self.n_cells = 368
        self.A_cell = 2.0
        self.I_rated = 7800.0
        self.U_rev = 1.229
        self.U_th = 1.481
        self.r1 = 3.202e-5
        self.r2 = 8.970e-8
        self.r3 = -4.193e-12
        self.P_sys = 1.6e6
        self.delta_P = 0.01 * self.P_sys
        self.s = 7.572e-2
        self.t1 = -1.070e-1
        self.t2 = 14.43
        self.t3 = 38.8
        self.F = 96485.0

        # Thermal Parameters
        self.C_s_i = 3.450e7
        self.C_sep = 5.193e7
        self.C_he = 2.175e7
        self.C_c = 2e7
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
        self.segma_sep = 50.0 * 4
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

        # Control change penalty weights
        u_range = np.array([self.I_max - self.I_min] * 4 +
                           [self.v_lye_max - self.v_lye_min] * 4 +
                           [self.v_c_max - self.v_c_min])
        # Smoothing penalty disabled (set to zero)
        self.lambda_u = np.zeros(9)

        # Per-control reference tracking weight scale
        if u_weight_scale is None:
            self.u_weight_scale = np.ones(9)
        else:
            self.u_weight_scale = np.array(u_weight_scale, dtype=float)

        # Build symbolic CBF function for Lie derivative
        self._build_symbolic_functions()

    def _build_symbolic_functions(self):
        """Build CasADi functions for h(x,u), f(x,u), L_f h, and L_f^2 h."""
        x_sym = ca.MX.sym("x", 13)
        u_sym = ca.MX.sym("u", 9)

        h_sym = self._compute_h_ca(x_sym, u_sym)
        f_sym = self._compute_f_ca(x_sym, u_sym)

        # First-order Lie derivative: L_f h = dh/dx @ f
        dh_dx_sym = ca.jacobian(h_sym, x_sym)
        lie1_sym = dh_dx_sym @ f_sym

        # Second-order Lie derivative: L_f^2 h = d(L_f h)/dx @ f
        # Computed for all constraints; HTO (index 4) will use it
        dlie1_dx_sym = ca.jacobian(lie1_sym, x_sym)
        lie2_sym = dlie1_dx_sym @ f_sym

        self._ca_h = ca.Function("h", [x_sym, u_sym], [h_sym])
        self._ca_f = ca.Function("f", [x_sym, u_sym], [f_sym])
        self._ca_lie = ca.Function("lie", [x_sym, u_sym], [lie1_sym, lie2_sym, h_sym, f_sym])

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

    def _compute_h(self, state, control):
        """Compute CBF function values h(x) with safety margins (9 constraints) using NumPy."""
        T_s_vec = state[1:5]
        T_sep = state[5]
        n_gas = state[12]

        hto = (n_gas * self.R * T_sep) / (self.P_sys * self.V_sep_gas)

        h_T = self.T_max - T_s_vec - self.h_margin_vec[0:4]
        h_HTO = np.array([self.HTO_max - hto - self.h_margin_vec[4]])
        h_Tmin = T_s_vec - self.T_min - self.h_margin_vec[5:9]

        h = np.concatenate([h_T, h_HTO, h_Tmin])
        return h

    def _compute_h_ca(self, state, control):
        """Symbolic CBF computation using CasADi MX."""
        T_s_vec = state[1:5]
        T_sep = state[5]
        n_gas = state[12]

        hto = (n_gas * self.R * T_sep) / (self.P_sys * self.V_sep_gas)

        h_T = self.T_max - T_s_vec - self.h_margin_vec[0:4]
        h_HTO = ca.vertcat(self.HTO_max - hto - self.h_margin_vec[4])
        h_Tmin = T_s_vec - self.T_min - self.h_margin_vec[5:9]

        h = ca.vertcat(h_T, h_HTO, h_Tmin)
        return h

    def _compute_f(self, state, control):
        """
        Compute dynamics f(x,u) for Lie derivative.
        Exact match with MultiStackSimulator._get_derivatives.
        NumPy version.
        """
        T_s_in = state[0]
        T_s_vec = state[1:5]
        T_sep = state[5]
        T_c_out = state[6]
        n_H2_an_vec = state[7:11]
        n_liq = state[11]
        n_gas = state[12]

        I_vec = control[0:4]
        v_lye_vec = control[4:8]
        v_c = control[8]

        # --- Electrochemical properties (vectorized) ---
        T_C_vec = T_s_vec - 273.15
        R_ohm_vec = self.r1 + self.r2 * T_s_vec + self.r3 * self.P_sys
        V_ohm_vec = R_ohm_vec * I_vec
        term_act_vec = self.t1 + self.t2 / T_C_vec + self.t3 / (T_C_vec**2)
        arg_vec = term_act_vec * I_vec + 1.0
        arg_safe_vec = np.maximum(arg_vec, 1e-9)
        V_act_vec = np.where(arg_vec > 1e-9, self.s * np.log(arg_safe_vec), 0.0)
        U_cell_vec = self.U_rev + V_ohm_vec + V_act_vec
        U_cell_vec = np.maximum(U_cell_vec, self.U_rev)

        f1_vec = 50.0 + 2.5 * T_C_vec
        f2_vec = 0.92 - 6.25e-6 * T_C_vec
        I_sq_vec = I_vec**2
        eta_vec = (I_sq_vec / (f1_vec + I_sq_vec)) * f2_vec

        Q_ele_vec = self.n_cells * I_vec * (U_cell_vec - eta_vec * self.U_th)

        # --- Thermal derivatives ---
        v_tot = np.sum(v_lye_vec)

        # Stack thermal
        Q_conv_vec = self.segma_s * (T_s_vec - self.T_amb)
        Q_rad_vec = self.epsilon_stack * self.sigma_b * self.A_stack * (T_s_vec**4 - self.T_amb**4)
        Q_diss_vec = Q_conv_vec + Q_rad_vec
        Q_flow_vec = self.c_lye * self.rho_lye * v_lye_vec * (T_s_vec - T_s_in)
        dT_s_dt_vec = (Q_ele_vec - Q_diss_vec - Q_flow_vec) / self.C_s_i

        # Separator thermal
        if v_tot > 1e-6:
            T_sep_in = np.sum(v_lye_vec * T_s_vec) / v_tot
        else:
            T_sep_in = T_sep
        Q_sep_rad = self.epsilon_sep * self.sigma_b * self.A_sep * (T_sep**4 - self.T_amb**4)
        Q_sep_conv = self.segma_sep * (T_sep - self.T_amb)
        Q_sep_diss = Q_sep_conv + Q_sep_rad
        dT_sep_dt = (0.5 * self.c_lye * self.rho_lye * v_tot * (T_sep_in - T_sep) - Q_sep_diss) / self.C_sep

        # Heat exchanger thermal (LMTD)
        C_rate_lye = self.c_lye * self.rho_lye * v_tot
        C_rate_cw = self.c_cw * self.rho_cw * v_c
        delta_T1 = T_s_in - T_c_out
        delta_T2 = T_sep - self.T_c_in

        if abs(delta_T1 - delta_T2) < 1e-5:
            LMTD = delta_T1
        elif delta_T1 * delta_T2 <= 0:
            LMTD = 0.0
        else:
            LMTD = (delta_T1 - delta_T2) / np.log(delta_T1 / delta_T2)

        Q_hx = self.k_he * self.A_he * LMTD
        dT_s_in_dt = (C_rate_lye * (T_sep - T_s_in) - Q_hx) / self.C_he
        dT_c_out_dt = (C_rate_cw * (self.T_c_in - T_c_out) + Q_hx) / self.C_c

        # --- HTO derivatives ---
        n_dot_O2_prod_vec = self.n_cells * I_vec * eta_vec / (4 * self.F)

        n_dot_H2_lye_vec = self.S_H2_lye * self.rho_lye * v_lye_vec / 4.0
        n_dot_H2_diff_vec = self.A_cell * self.n_cells * self.D_eff * self.S_H2_lye * self.P_sys / self.delta
        n_dot_H2_conv_vec = self.A_cell * self.n_cells * (self.K_eff / self.mu_lye) * self.S_H2_lye * self.rho_lye * (self.delta_P / self.delta)
        n_dot_H2_im_vec = n_dot_H2_lye_vec + n_dot_H2_diff_vec + n_dot_H2_conv_vec

        n_dot_H2_im_1_vec = n_H2_an_vec * v_lye_vec / (2 * self.V_an_lye)
        dn_H2_an_dt_vec = n_dot_H2_im_vec - n_dot_H2_im_1_vec

        n_dot_H2_2 = n_liq / self.tau_sep
        dn_liq_dt = np.sum(n_dot_H2_im_1_vec) - n_dot_H2_2

        sum_n_dot_O2 = np.sum(n_dot_O2_prod_vec)
        if sum_n_dot_O2 > 1e-9:
            n_dot_H2_out = (self.R * T_sep * n_gas * sum_n_dot_O2) / (self.P_sys * self.V_sep_gas)
        else:
            n_dot_H2_out = 0.0

        dn_gas_dt = n_dot_H2_2 - n_dot_H2_out

        return np.concatenate([[dT_s_in_dt], dT_s_dt_vec, [dT_sep_dt], [dT_c_out_dt],
                               dn_H2_an_dt_vec, [dn_liq_dt], [dn_gas_dt]])

    def _compute_f_ca(self, state, control):
        """
        Symbolic dynamics f(x,u) using CasADi MX.
        Exact match with MultiStackSimulator._get_derivatives.
        """
        T_s_in = state[0]
        T_s_vec = state[1:5]
        T_sep = state[5]
        T_c_out = state[6]
        n_H2_an_vec = state[7:11]
        n_liq = state[11]
        n_gas = state[12]

        I_vec = control[0:4]
        v_lye_vec = control[4:8]
        v_c = control[8]

        # --- Electrochemical properties ---
        T_C_vec = T_s_vec - 273.15
        R_ohm_vec = self.r1 + self.r2 * T_s_vec + self.r3 * self.P_sys
        V_ohm_vec = R_ohm_vec * I_vec
        term_act_vec = self.t1 + self.t2 / T_C_vec + self.t3 / (T_C_vec**2)
        arg_vec = term_act_vec * I_vec + 1.0
        arg_safe_vec = ca.fmax(arg_vec, 1e-9)
        V_act_vec = ca.if_else(arg_vec > 1e-9, self.s * ca.log(arg_safe_vec), 0.0)
        U_cell_vec = self.U_rev + V_ohm_vec + V_act_vec
        U_cell_vec = ca.fmax(U_cell_vec, self.U_rev)

        f1_vec = 50.0 + 2.5 * T_C_vec
        f2_vec = 0.92 - 6.25e-6 * T_C_vec
        I_sq_vec = I_vec**2
        eta_vec = (I_sq_vec / (f1_vec + I_sq_vec)) * f2_vec

        Q_ele_vec = self.n_cells * I_vec * (U_cell_vec - eta_vec * self.U_th)

        # --- Thermal derivatives ---
        v_tot = ca.sum1(v_lye_vec)

        # Stack thermal
        Q_conv_vec = self.segma_s * (T_s_vec - self.T_amb)
        Q_rad_vec = self.epsilon_stack * self.sigma_b * self.A_stack * (T_s_vec**4 - self.T_amb**4)
        Q_diss_vec = Q_conv_vec + Q_rad_vec
        Q_flow_vec = self.c_lye * self.rho_lye * v_lye_vec * (T_s_vec - T_s_in)
        dT_s_dt_vec = (Q_ele_vec - Q_diss_vec - Q_flow_vec) / self.C_s_i

        # Separator thermal
        T_sep_in = ca.if_else(v_tot > 1e-6, ca.sum1(v_lye_vec * T_s_vec) / v_tot, T_sep)
        Q_sep_rad = self.epsilon_sep * self.sigma_b * self.A_sep * (T_sep**4 - self.T_amb**4)
        Q_sep_conv = self.segma_sep * (T_sep - self.T_amb)
        Q_sep_diss = Q_sep_conv + Q_sep_rad
        dT_sep_dt = (0.5 * self.c_lye * self.rho_lye * v_tot * (T_sep_in - T_sep) - Q_sep_diss) / self.C_sep

        # Heat exchanger thermal (LMTD)
        C_rate_lye = self.c_lye * self.rho_lye * v_tot
        C_rate_cw = self.c_cw * self.rho_cw * v_c
        delta_T1 = T_s_in - T_c_out
        delta_T2 = T_sep - self.T_c_in

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

        # --- HTO derivatives ---
        n_dot_O2_prod_vec = self.n_cells * I_vec * eta_vec / (4 * self.F)

        n_dot_H2_lye_vec = self.S_H2_lye * self.rho_lye * v_lye_vec / 4.0
        n_dot_H2_diff_vec = self.A_cell * self.n_cells * self.D_eff * self.S_H2_lye * self.P_sys / self.delta
        n_dot_H2_conv_vec = self.A_cell * self.n_cells * (self.K_eff / self.mu_lye) * self.S_H2_lye * self.rho_lye * (self.delta_P / self.delta)
        n_dot_H2_im_vec = n_dot_H2_lye_vec + n_dot_H2_diff_vec + n_dot_H2_conv_vec

        n_dot_H2_im_1_vec = n_H2_an_vec * v_lye_vec / (2 * self.V_an_lye)
        dn_H2_an_dt_vec = n_dot_H2_im_vec - n_dot_H2_im_1_vec

        n_dot_H2_2 = n_liq / self.tau_sep
        dn_liq_dt = ca.sum1(n_dot_H2_im_1_vec) - n_dot_H2_2

        sum_n_dot_O2 = ca.sum1(n_dot_O2_prod_vec)
        n_dot_H2_out = ca.if_else(
            sum_n_dot_O2 > 1e-9,
            (self.R * T_sep * n_gas * sum_n_dot_O2) / (self.P_sys * self.V_sep_gas),
            0.0
        )

        dn_gas_dt = n_dot_H2_2 - n_dot_H2_out

        f = ca.vertcat(dT_s_in_dt, dT_s_dt_vec, dT_sep_dt, dT_c_out_dt,
                       dn_H2_an_dt_vec, dn_liq_dt, dn_gas_dt)
        return f

    def _compute_lie_derivative(self, state, control):
        """
        Compute Lie derivatives using CasADi.
        Returns NumPy arrays for first-order and second-order Lie derivatives.
        """
        state_dm = ca.DM(state)
        control_dm = ca.DM(control)
        lie1, lie2, h_current, f_current = self._ca_lie(state_dm, control_dm)
        return (np.array(lie1).flatten(), np.array(lie2).flatten(),
                np.array(h_current).flatten(), np.array(f_current).flatten())

    def project(self, u_ref, state, u_last=None, verbose=False):
        """
        Project reference control using mixed-order CBF with soft constraints.

        HTO uses second-order CBF:
            psi_1 = L_f h + alpha1 * h
            psi_2 = L_f^2 h + alpha2 * psi_1 >= 0

        Temperature uses first-order CBF:
            L_f h + gamma * h >= 0

        Args:
            u_ref: Reference control [I1..4, v_lye1..4, v_c]
            state: Current state [13-dim]
            u_last: Previous control [I1..4, v_lye1..4, v_c] (for smoothness penalty)
            verbose: Print diagnostics

        Returns:
            u_safe: Projected control
            success: Whether projection succeeded (with slack)
            info: Dict with CBF values, derivatives, and slack values
        """
        u_ref = np.asarray(u_ref).flatten()
        state = np.asarray(state).flatten()
        u_last = np.asarray(u_last).flatten() if u_last is not None else u_ref.copy()

        # Compute current CBF values and Lie derivatives at reference
        lie1_ref, lie2_ref, h_ref_raw, f_ref = self._compute_lie_derivative(state, u_ref)

        # Normalize if enabled
        if self.normalize:
            lie1_ref_norm = lie1_ref / np.maximum(np.abs(self.lie_deriv_scales), 1e-10)
            lie2_ref_norm = lie2_ref / np.maximum(np.abs(self.lie_deriv_scales)**2, 1e-10)
            h_ref_norm = h_ref_raw / self.h_scales
            lie1_ref = lie1_ref_norm
            lie2_ref = lie2_ref_norm
            h_ref = h_ref_norm
        else:
            h_ref = h_ref_raw

        # Build CBF conditions
        # Mixed-order: alpha2_vec[j] > 0 means use second-order for constraint j
        cbf_ref = np.zeros(9)
        for j in range(9):
            if self.alpha2_vec[j] > 0:
                # Second-order HOCBF
                psi1 = lie1_ref[j] + self.alpha1_vec[j] * h_ref[j]
                cbf_ref[j] = lie2_ref[j] + self.alpha2_vec[j] * psi1
            else:
                # First-order CBF (fallback)
                cbf_ref[j] = lie1_ref[j] + self.gamma_vec[j] * h_ref[j]

        if verbose:
            print(f"Reference point:")
            print(f"  h = {h_ref}")
            print(f"  L_f h = {lie1_ref}")
            print(f"  L_f^2 h = {lie2_ref}")
            second_order_idx = np.where(self.alpha2_vec > 0)[0]
            first_order_idx = np.where(self.alpha2_vec <= 0)[0]
            if len(second_order_idx) > 0:
                for j in second_order_idx:
                    psi1_j = lie1_ref[j] + self.alpha1_vec[j] * h_ref[j]
                    psi2_j = lie2_ref[j] + self.alpha2_vec[j] * psi1_j
                    print(f"  psi_1 [{j}] = {psi1_j:.4f}, psi_2 [{j}] = {psi2_j:.4f}")
            if len(first_order_idx) > 0:
                print(f"  First-order CBF {list(first_order_idx)} = {cbf_ref[first_order_idx]}")

        if np.all(cbf_ref >= 0):
            if u_last is not None and np.max(np.abs(u_last[:4] - u_ref[:4])) > 500.0:
                pass
            else:
                info = {
                    'h': h_ref,
                    'lie1': lie1_ref,
                    'lie2': lie2_ref,
                    'cbf': cbf_ref,
                    'projection_needed': False,
                    'slack': np.zeros(9)
                }
                return u_ref, True, info

        # CBF constraint optimization
        n_cbf = 9
        soft_idx = np.where(self.soft_mask)[0]
        n_soft = len(soft_idx)

        u_range = np.array([self.I_max - self.I_min] * 4 +
                           [self.v_lye_max - self.v_lye_min] * 4 +
                           [self.v_c_max - self.v_c_min])
        u_weights = self.u_weight_scale / (u_range ** 2)

        u_min_arr = np.array([self.I_min] * 4 + [self.v_lye_min] * 4 + [self.v_c_min])
        u_max_arr = np.array([self.I_max] * 4 + [self.v_lye_max] * 4 + [self.v_c_max])

        u0 = np.clip(u_ref, u_min_arr, u_max_arr)

        # Search for feasible initial guess
        if not np.all(cbf_ref >= 0):
            n_search = 50
            found_feasible = False

            # Helper to evaluate CBF constraints
            def _eval_cbf_constraints(state_in, u_in):
                lie1_t, lie2_t, h_t, _ = self._compute_lie_derivative(state_in, u_in)
                if self.normalize:
                    lie1_t = lie1_t / np.maximum(np.abs(self.lie_deriv_scales), 1e-10)
                    lie2_t = lie2_t / np.maximum(np.abs(self.lie_deriv_scales)**2, 1e-10)
                    h_t = h_t / self.h_scales
                cbf_t = np.zeros(9)
                for jj in range(9):
                    if self.alpha2_vec[jj] > 0:
                        psi1_t_j = lie1_t[jj] + self.alpha1_vec[jj] * h_t[jj]
                        cbf_t[jj] = lie2_t[jj] + self.alpha2_vec[jj] * psi1_t_j
                    else:
                        cbf_t[jj] = lie1_t[jj] + self.gamma_vec[jj] * h_t[jj]
                return cbf_t

            # Identify violated constraints
            temp_upper_violated = np.any(cbf_ref[0:4] < 0)
            temp_lower_violated = np.any(cbf_ref[5:9] < 0)
            temp_violated = temp_upper_violated or temp_lower_violated

            # Strategy 1: If temperature constraints violated, try increasing v_c first
            if temp_violated and u0[8] < self.v_c_max:
                n_vc_search = min(20, max(3, int((self.v_c_max - u0[8]) / 0.005) + 1))
                for beta in np.linspace(1.0, 2.0, n_vc_search):
                    v_c_test = min(beta * u0[8], self.v_c_max)
                    if v_c_test <= u0[8]:
                        continue
                    u_test = np.concatenate([u0[0:4], u0[4:8], [v_c_test]])
                    cbf_test = _eval_cbf_constraints(state, u_test)
                    if np.all(cbf_test >= 0):
                        u0[8] = v_c_test
                        found_feasible = True
                        break

            # Strategy 2: Try reducing current (original approach)
            if not found_feasible:
                for alpha in np.linspace(1.0, 0.0, n_search):
                    I_test = alpha * u0[0:4]
                    u_test = np.concatenate([I_test, u0[4:8], [u0[8]]])
                    cbf_test = _eval_cbf_constraints(state, u_test)
                    if np.all(cbf_test >= 0):
                        u0[0:4] = I_test
                        found_feasible = True
                        break

            # Strategy 3: Try both reducing current AND increasing v_c
            if not found_feasible and temp_violated and u0[8] < self.v_c_max:
                for alpha in np.linspace(1.0, 0.0, n_search):
                    I_test = alpha * u0[0:4]
                    n_vc_search = min(10, max(2, int((self.v_c_max - u0[8]) / 0.01) + 1))
                    for beta in np.linspace(1.0, 2.0, n_vc_search):
                        v_c_test = min(beta * u0[8], self.v_c_max)
                        if v_c_test <= u0[8]:
                            continue
                        u_test = np.concatenate([I_test, u0[4:8], [v_c_test]])
                        cbf_test = _eval_cbf_constraints(state, u_test)
                        if np.all(cbf_test >= 0):
                            u0[0:4] = I_test
                            u0[8] = v_c_test
                            found_feasible = True
                            break
                    if found_feasible:
                        break

            # Strategy 4: HTO-specific fallback
            if not found_feasible:
                hto_cbf_val = cbf_ref[4]
                if hto_cbf_val < 0:
                    u0[0:4] = np.clip(u0[0:4] + 1000.0, 2000.0, self.I_max)
                    u0[4:8] = np.clip(u0[4:8], 0.02, self.v_lye_max)

        s0 = np.maximum(0, -(cbf_ref[self.soft_mask])) if n_soft > 0 else np.array([])
        us0 = np.concatenate([u0, s0])

        # Build CasADi NLP
        n_us = 9 + n_soft
        us_sym = ca.MX.sym("us", n_us)
        u_sym = us_sym[:9]
        s_sym = us_sym[9:] if n_soft > 0 else ca.MX(0, 1)

        ref_cost = ca.sum1(u_weights * (u_sym - u_ref)**2)
        delta_cost = ca.sum1(self.lambda_u * (u_sym - u_last)**2)
        if n_soft > 0:
            slack_cost = ca.sum1(self.rho_vec[self.soft_mask] * s_sym**2)
        else:
            slack_cost = 0.0
        obj = ref_cost + delta_cost + slack_cost

        # CBF constraints via symbolic function
        state_dm = ca.DM(state)
        lie1_sym, lie2_sym, h_sym, _ = self._ca_lie(state_dm, u_sym)

        if self.normalize:
            lie1_sym = lie1_sym / np.maximum(np.abs(self.lie_deriv_scales), 1e-10)
            lie2_sym = lie2_sym / np.maximum(np.abs(self.lie_deriv_scales)**2, 1e-10)
            h_sym = h_sym / self.h_scales

        # Build constraint vector
        g = []
        lbg = []
        ubg = []
        s_idx = 0
        for j in range(n_cbf):
            if self.alpha2_vec[j] > 0:
                # Second-order HOCBF
                psi1_sym = lie1_sym[j] + self.alpha1_vec[j] * h_sym[j]
                cbf_j = lie2_sym[j] + self.alpha2_vec[j] * psi1_sym
            else:
                # First-order CBF
                cbf_j = lie1_sym[j] + self.gamma_vec[j] * h_sym[j]

            if self.soft_mask[j]:
                g.append(cbf_j + s_sym[s_idx])
                s_idx += 1
            else:
                g.append(cbf_j)
            lbg.append(0.0)
            ubg.append(ca.inf)

        g = ca.vertcat(*g)
        lbg = np.array(lbg)
        ubg = np.array(ubg)

        lbx = np.concatenate([u_min_arr, np.zeros(n_soft)])
        ubx = np.concatenate([u_max_arr, np.full(n_soft, ca.inf)])

        nlp = {"x": us_sym, "f": obj, "g": g}
        ipopt_opts = {
            "ipopt": {
                "print_level": 0,
                "sb": "yes",
                "max_iter": 300,
                "tol": 1e-6,
                "acceptable_tol": 1e-4,
            }
        }
        if verbose:
            ipopt_opts["ipopt"]["print_level"] = 5
            ipopt_opts["ipopt"].pop("sb", None)
        solver = ca.nlpsol("solver", "ipopt", nlp, ipopt_opts)

        result = solver(x0=us0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

        u_opt = np.array(result["x"][:9]).flatten()

        s_opt_full = np.zeros(n_cbf)
        if n_soft > 0:
            s_opt_raw = np.array(result["x"][9:]).flatten()
            s_opt_full[self.soft_mask] = s_opt_raw
            max_slack = float(np.max(s_opt_raw)) if len(s_opt_raw) > 0 else 0.0
        else:
            max_slack = 0.0

        lie1_opt, lie2_opt, h_opt, _ = self._compute_lie_derivative(state, u_opt)

        lie1_opt_raw = lie1_opt.copy()
        lie2_opt_raw = lie2_opt.copy()
        h_opt_raw = h_opt.copy()

        if self.normalize:
            lie1_opt = lie1_opt / np.maximum(np.abs(self.lie_deriv_scales), 1e-10)
            lie2_opt = lie2_opt / np.maximum(np.abs(self.lie_deriv_scales)**2, 1e-10)
            h_opt = h_opt / self.h_scales

        cbf_opt = np.zeros(n_cbf)
        for j in range(n_cbf):
            if self.alpha2_vec[j] > 0:
                psi1_opt = lie1_opt[j] + self.alpha1_vec[j] * h_opt[j]
                cbf_opt[j] = lie2_opt[j] + self.alpha2_vec[j] * psi1_opt
            else:
                cbf_opt[j] = lie1_opt[j] + self.gamma_vec[j] * h_opt[j]

        slack_violation = np.maximum(0, -cbf_opt)

        control_feasible = (
            np.all(u_opt >= u_min_arr - 1e-6) and
            np.all(u_opt <= u_max_arr + 1e-6)
        )
        cbf_feasible = np.all(cbf_opt >= -1e-4)
        slack_feasible = np.all(s_opt_full >= -1e-6)
        hard_feasible = np.all(cbf_opt[~self.soft_mask] >= -1e-4) if np.any(~self.soft_mask) else True

        stats = solver.stats()
        solver_success = bool(stats.get("success", False))
        return_status = stats.get("return_status", "unknown")
        actual_success = solver_success and control_feasible and cbf_feasible and slack_feasible and hard_feasible

        info = {
            'h': h_opt_raw,
            'h_norm': h_opt if self.normalize else None,
            'lie1': lie1_opt_raw,
            'lie1_norm': lie1_opt if self.normalize else None,
            'lie2': lie2_opt_raw,
            'lie2_norm': lie2_opt if self.normalize else None,
            'cbf': cbf_opt,
            'projection_needed': True,
            'adjustment': np.linalg.norm(u_opt - u_ref),
            'slack': s_opt_full,
            'max_slack': max_slack,
            'slack_violation': slack_violation,
            'optimization_success': actual_success,
            'solver_message': str(return_status)
        }

        if verbose:
            print(f"Projected point:")
            print(f"  u = {u_opt}")
            print(f"  s = {s_opt_full}")
            print(f"  h = {h_opt}")
            print(f"  CBF = {cbf_opt}")
            print(f"  Max slack: {max_slack:.4f}")
            if not solver_success and actual_success:
                print(f"  [Note: IPOPT reported failure but solution is feasible - accepted]")

        return u_opt, bool(actual_success), info


# Alias for backward compatibility
MultiStackCBFProjectionHO_Simplified = MultiStackCBFProjectionHO
