"""
CBF-based Projection Operator for Multi-Stack AWE System

Implements rigorous Control Barrier Function (CBF) constraints with explicit
Lie derivative computation: dh/dt + gamma * h >= 0

Adapted from single-stack implementation for 4-stack AWE system.
Uses CasADi/IPOPT for the NLP.
"""

import numpy as np
import casadi as ca


class MultiStackCBFProjection:
    """
    CBF projection operator with explicit Lie derivative computation.
    Uses CasADi symbolic differentiation and IPOPT for optimization.

    CBF Condition: L_f h(x,u) + gamma * h(x) >= 0
    """

    def __init__(self, dt=60.0, sim_dt=0.2, gamma=1.0, epsilon=1e-4, rho=1e12,
                 gamma_vec=None, rho_vec=None, h_scales=None, lie_deriv_scales=None, normalize=True,
                 h_margin_vec=None, lambda_u_scale=1000.0, soft_mask=None):
        """
        Args:
            dt: Control timestep
            sim_dt: Simulation timestep
            gamma: Default gamma (used if gamma_vec not provided)
            epsilon: Numerical differentiation step size (unused, kept for API compat)
            rho: Default rho (used if rho_vec not provided)
            gamma_vec: List of gamma values for each CBF constraint
            rho_vec: List of rho values for each CBF constraint
            h_scales: List of h scales for normalization
            lie_deriv_scales: List of L_f h scales for normalization
            normalize: Whether to use normalized CBF constraints
            h_margin_vec: List of safety margins for each constraint
            lambda_u_scale: Scale factor for control change penalty (default: 1000.0).
            soft_mask: List of bool indicating whether each constraint uses soft slack
        """
        self.dt = dt
        self.sim_dt = sim_dt
        self.gamma = gamma
        self.epsilon = epsilon
        self.normalize = normalize

        # Per-constraint gamma values
        if gamma_vec is None:
            defaults = [10.0] * 4 + [gamma] + [10.0] * 4 + [20.0] * 4 + [10.0] * 4
            self.gamma_vec = np.array(defaults)
        else:
            self.gamma_vec = np.array(gamma_vec)

        # Per-constraint rho values for soft constraints
        if rho_vec is None:
            defaults = [1000.0] * 4 + [1e5] + [1000.0] * 4 + [1000.0] * 4 + [1000.0] * 4
            self.rho_vec = np.array(defaults)
        else:
            self.rho_vec = np.array(rho_vec)

        # Per-constraint safety margins (default: no margin)
        if h_margin_vec is None:
            self.h_margin_vec = np.array([0.0]*4 + [0.007] + [0.0]*12)
        else:
            self.h_margin_vec = np.array(h_margin_vec)

        # Per-constraint soft slack mask (default: all soft)
        if soft_mask is None:
            self.soft_mask = np.full(17, True, dtype=bool)
        else:
            self.soft_mask = np.array(soft_mask, dtype=bool)

        # Normalization scales
        if h_scales is None:
            defaults = [70.0] * 4 + [0.02] + [0.5] * 4 + [5e6] * 4 + [70.0] * 4
            self.h_scales = np.array(defaults)
        else:
            self.h_scales = np.array(h_scales)

        if lie_deriv_scales is None:
            defaults = [1e-2] * 4 + [1e-3] + [1e-5] * 4 + [1e0] * 4 + [1e-2] * 4
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
        self.lambda_u = lambda_u_scale / (u_range ** 2)

        # Build symbolic CBF function for Lie derivative
        self._build_symbolic_functions()

    def _build_symbolic_functions(self):
        """Build CasADi functions for h(x,u), f(x,u), and L_f h(x,u)."""
        x_sym = ca.MX.sym("x", 13)
        u_sym = ca.MX.sym("u", 9)

        h_sym = self._compute_h_ca(x_sym, u_sym)
        f_sym = self._compute_f_ca(x_sym, u_sym)

        dh_dx_sym = ca.jacobian(h_sym, x_sym)
        lie_sym = dh_dx_sym @ f_sym

        self._ca_h = ca.Function("h", [x_sym, u_sym], [h_sym])
        self._ca_f = ca.Function("f", [x_sym, u_sym], [f_sym])
        self._ca_lie = ca.Function("lie", [x_sym, u_sym], [lie_sym, h_sym, f_sym])

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

    def _faraday_efficiency(self, I_vec, T_s_vec):
        """Faraday efficiency - exact match with simulator (vectorized)"""
        T_C = T_s_vec - 273.15
        f1 = 50.0 + 2.5 * T_C
        f2 = 0.92 - 6.25e-6 * T_C
        I_sq = I_vec**2
        return (I_sq / (f1 + I_sq)) * f2

    def _cell_voltage(self, I_vec, T_s_vec):
        """Cell voltage - exact match with simulator (vectorized)"""
        T_C = T_s_vec - 273.15
        R_ohm = self.r1 + self.r2 * T_s_vec + self.r3 * self.P_sys
        V_ohm = R_ohm * I_vec
        term_act = self.t1 + self.t2 / T_C + self.t3 / (T_C**2)
        arg = term_act * I_vec + 1.0
        arg_safe = np.maximum(arg, 1e-9)
        V_act = np.where(arg > 1e-9, self.s * np.log(arg_safe), 0.0)
        U_cell = self.U_rev + V_ohm + V_act
        return np.maximum(U_cell, self.U_rev)

    def _compute_h(self, state, control):
        """Compute CBF function values h(x) with safety margins (17 constraints) using NumPy."""
        T_s_vec = state[1:5]
        T_sep = state[5]
        n_gas = state[12]
        I_vec = control[0:4]

        U_cell_vec = self._cell_voltage(I_vec, T_s_vec)
        Power_vec = U_cell_vec * I_vec * self.n_cells
        hto = (n_gas * self.R * T_sep) / (self.P_sys * self.V_sep_gas)

        h_T = self.T_max - T_s_vec - self.h_margin_vec[0:4]
        h_HTO = np.array([self.HTO_max - hto - self.h_margin_vec[4]])
        h_V = self.U_cell_max - U_cell_vec - self.h_margin_vec[5:9]
        h_P = self.P_stack_max - Power_vec - self.h_margin_vec[9:13]
        h_Tmin = T_s_vec - self.T_min - self.h_margin_vec[13:17]

        h = np.concatenate([h_T, h_HTO, h_V, h_P, h_Tmin])
        return h

    def _compute_h_ca(self, state, control):
        """Symbolic CBF computation using CasADi MX."""
        T_s_vec = state[1:5]
        T_sep = state[5]
        n_gas = state[12]
        I_vec = control[0:4]

        T_C = T_s_vec - 273.15
        R_ohm = self.r1 + self.r2 * T_s_vec + self.r3 * self.P_sys
        V_ohm = R_ohm * I_vec
        term_act = self.t1 + self.t2 / T_C + self.t3 / (T_C**2)
        arg = term_act * I_vec + 1.0
        arg_safe = ca.fmax(arg, 1e-9)
        V_act = ca.if_else(arg > 1e-9, self.s * ca.log(arg_safe), 0.0)
        U_cell = self.U_rev + V_ohm + V_act
        U_cell = ca.fmax(U_cell, self.U_rev)

        Power_vec = U_cell * I_vec * self.n_cells
        hto = (n_gas * self.R * T_sep) / (self.P_sys * self.V_sep_gas)

        h_T = self.T_max - T_s_vec - self.h_margin_vec[0:4]
        h_HTO = ca.vertcat(self.HTO_max - hto - self.h_margin_vec[4])
        h_V = self.U_cell_max - U_cell - self.h_margin_vec[5:9]
        h_P = self.P_stack_max - Power_vec - self.h_margin_vec[9:13]
        h_Tmin = T_s_vec - self.T_min - self.h_margin_vec[13:17]

        h = ca.vertcat(h_T, h_HTO, h_V, h_P, h_Tmin)
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
        Compute Lie derivative L_f h = dh/dx * f(x,u) using CasADi.
        Returns NumPy arrays.
        """
        state_dm = ca.DM(state)
        control_dm = ca.DM(control)
        lie_derivative, h_current, f_current = self._ca_lie(state_dm, control_dm)
        return np.array(lie_derivative).flatten(), np.array(h_current).flatten(), None, np.array(f_current).flatten()

    def project(self, u_ref, state, u_last=None, verbose=False):
        """
        Project reference control using CBF with soft constraints via CasADi/IPOPT.

        Soft CBF formulation:
            min ||u - u_ref||^2 + lambda * ||u - u_last||^2 + rho * ||s||^2
            s.t. L_f h_i + gamma * h_i >= -s_i,  s_i >= 0
                 u_min <= u <= u_max

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
        lie_deriv_ref, h_ref, _, f_ref = self._compute_lie_derivative(state, u_ref)

        # Normalize if enabled
        if self.normalize:
            lie_deriv_ref_norm = lie_deriv_ref / np.maximum(np.abs(self.lie_deriv_scales), 1e-10)
            h_ref_norm = h_ref / self.h_scales
            lie_deriv_ref = lie_deriv_ref_norm
            h_ref = h_ref_norm

        if verbose:
            print(f"Reference point:")
            print(f"  h = {h_ref}")
            print(f"  L_f h = {lie_deriv_ref}")
            gamma_h = self.gamma_vec * h_ref if self.normalize else self.gamma * h_ref
            print(f"  L_f h + gamma*h = {lie_deriv_ref + gamma_h}")

        # Check if reference already satisfies CBF (with per-constraint gamma)
        if self.normalize:
            cbf_ref = lie_deriv_ref + self.gamma_vec * h_ref
        else:
            cbf_ref = lie_deriv_ref + self.gamma * h_ref

        if np.all(cbf_ref >= 0):
            # Reference is already safe
            info = {
                'h': h_ref,
                'lie_deriv': lie_deriv_ref,
                'cbf': cbf_ref,
                'projection_needed': False,
                'slack': np.zeros(17)
            }
            return u_ref, True, info

        # CBF constraint optimization with selectable soft/hard constraints
        n_cbf = 17
        soft_idx = np.where(self.soft_mask)[0]
        n_soft = len(soft_idx)

        # Normalized control weights
        u_range = np.array([self.I_max - self.I_min] * 4 +
                           [self.v_lye_max - self.v_lye_min] * 4 +
                           [self.v_c_max - self.v_c_min])
        u_weights = 1.0 / (u_range ** 2)

        # Bounds: control bounds + slack bounds for soft constraints only
        u_min_arr = np.array([self.I_min] * 4 + [self.v_lye_min] * 4 + [self.v_c_min])
        u_max_arr = np.array([self.I_max] * 4 + [self.v_lye_max] * 4 + [self.v_c_max])

        # Initial guess: start from clipped reference
        u0 = np.clip(u_ref, u_min_arr, u_max_arr)

        # If reference violates CBF, search for a feasible initial guess
        if not np.all(cbf_ref >= 0):
            n_search = 50
            found_feasible = False
            for alpha in np.linspace(1.0, 0.0, n_search):
                I_test = alpha * u0[0:4]
                u_test = np.concatenate([I_test, u0[4:8], [u0[8]]])
                ld_test, h_test, _, _ = self._compute_lie_derivative(state, u_test)
                if self.normalize:
                    ld_test = ld_test / np.maximum(np.abs(self.lie_deriv_scales), 1e-10)
                    h_test = h_test / self.h_scales
                    cbf_test = ld_test + self.gamma_vec * h_test
                else:
                    cbf_test = ld_test + self.gamma * h_test
                if np.all(cbf_test >= 0):
                    u0[0:4] = I_test
                    found_feasible = True
                    break
            if not found_feasible:
                hto_cbf_val = cbf_ref[4] if len(cbf_ref) > 4 else 0
                if hto_cbf_val < 0:
                    u0[0:4] = np.clip(u0[0:4] + 1000.0, 2000.0, self.I_max)
                    u0[4:8] = np.clip(u0[4:8], 0.02, self.v_lye_max)

        # Initial slack only for soft constraints
        s0 = np.maximum(0, -(cbf_ref[self.soft_mask])) if n_soft > 0 else np.array([])
        us0 = np.concatenate([u0, s0])

        # Build CasADi NLP
        n_us = 9 + n_soft
        us_sym = ca.MX.sym("us", n_us)
        u_sym = us_sym[:9]
        s_sym = us_sym[9:] if n_soft > 0 else ca.MX(0, 1)

        # Objective
        ref_cost = ca.sum1(u_weights * (u_sym - u_ref)**2)
        delta_cost = ca.sum1(self.lambda_u * (u_sym - u_last)**2)
        if n_soft > 0:
            slack_cost = ca.sum1(self.rho_vec[self.soft_mask] * s_sym**2)
        else:
            slack_cost = 0.0
        obj = ref_cost + delta_cost + slack_cost

        # CBF constraints via symbolic function
        state_dm = ca.DM(state)
        lie_sym, h_sym, _ = self._ca_lie(state_dm, u_sym)

        if self.normalize:
            lie_sym = lie_sym / np.maximum(np.abs(self.lie_deriv_scales), 1e-10)
            h_sym = h_sym / self.h_scales
            cbf_sym = lie_sym + self.gamma_vec * h_sym
        else:
            cbf_sym = lie_sym + self.gamma * h_sym

        # Build constraint vector: cbf + s >= 0
        g = []
        lbg = []
        ubg = []
        s_idx = 0
        for j in range(n_cbf):
            if self.soft_mask[j]:
                g.append(cbf_sym[j] + s_sym[s_idx])
                s_idx += 1
            else:
                g.append(cbf_sym[j])
            lbg.append(0.0)
            ubg.append(ca.inf)

        g = ca.vertcat(*g)
        lbg = np.array(lbg)
        ubg = np.array(ubg)

        # Variable bounds
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

        # Parse slack: expand back to 17-dim, hard constraints get 0
        s_opt_full = np.zeros(n_cbf)
        if n_soft > 0:
            s_opt_raw = np.array(result["x"][9:]).flatten()
            s_opt_full[self.soft_mask] = s_opt_raw
            max_slack = float(np.max(s_opt_raw)) if len(s_opt_raw) > 0 else 0.0
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

        # Actual constraint violation
        slack_violation = np.maximum(0, -cbf_opt)

        # Feasibility override
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
        actual_success = solver_success or (control_feasible and cbf_feasible and slack_feasible and hard_feasible and max_slack < 1e-3)

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
            'solver_message': str(return_status)
        }

        if verbose:
            print(f"Projected point:")
            print(f"  u = {u_opt}")
            print(f"  s = {s_opt_full}")
            print(f"  h = {h_opt}")
            print(f"  L_f h + gamma*h = {cbf_opt}")
            print(f"  Max slack: {max_slack:.4f}")
            if not solver_success and actual_success:
                print(f"  [Note: IPOPT reported failure but solution is feasible - accepted]")

        return u_opt, bool(actual_success), info


# Alias for backward compatibility
MultiStackCBFProjectionSimplified = MultiStackCBFProjection
