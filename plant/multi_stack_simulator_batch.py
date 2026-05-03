"""
GPU-Batched Multi-Stack AWE Simulator

Re-implements the dynamics from MultiStackSimulator using PyTorch
to enable parallel rollout of N candidate trajectories on GPU.
Uses fixed-step RK4 integration instead of scipy.solve_ivp.
"""

import torch
import numpy as np


class BatchMultiStackSimulator:
    """
    Batched simulator for the 4-stack AWE system.

    All internal operations are PyTorch tensors on the specified device.
    State shape: (batch, 13)
    Action shape: (batch, 9)  [I1..4, v_lye1..4, v_c]
    """

    def __init__(self, dt=60.0, dt_sub=0.2, device='cuda'):
        self.dt = dt
        self.dt_sub = dt_sub
        self.n_sub = int(round(dt / dt_sub))
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # --- Parameters (matched to MultiStackSimulator) ---
        self.N = 4
        self.I_rated = 7800.0
        self.N_cell = 368
        self.A_cell = 2.0
        self.P_sys = 1.6e6
        self.delta_P = 0.01 * self.P_sys

        # Electrochemical
        self.r1 = 3.202e-5
        self.r2 = 8.970e-8
        self.r3 = -4.193e-12
        self.t1 = -1.070e-1
        self.t2 = 14.43
        self.t3 = 38.8
        self.s = 7.572e-2
        self.U_th = 1.481
        self.U_rev = 1.229

        # Thermal
        self.T_am = 298.0
        self.T_c_in = 288.0
        self.C_s_i = 3.450e7
        self.C_sep = 5.193e7
        self.C_he = 2.175e7
        self.C_c = 2.0e7
        self.A_he = 240.0
        self.k_he = 960.0
        self.c_lye = 3200.0
        self.c_cw = 4200.0
        self.rho_lye = 1280.0
        self.rho_cw = 1000.0
        self.segma_s = 1000.0
        self.segma_sep = 200.0
        self.A_stack = 80.0
        self.A_sep = 40.0
        self.epsilon_stack = 0.8
        self.epsilon_sep = 0.8
        self.sigma_b = 5.67e-8

        self.F = 96485.0

        # HTO
        self.V_an_lye = 2.5
        self.V_sep = 10.288
        self.V_sep_gas = 0.6 * self.V_sep
        self.delta = 500e-6
        self.D_eff = 8.569e-10
        self.K_eff = 2e-16
        self.tau_sep = 60.0
        self.R = 8.314
        self.mu_lye = 8.76e-4

        self.S_H2_lye = self._calculate_h2_solubility()

        # Scalar tensors on device
        self._register_scalars()

    def _register_scalars(self):
        """Pre-register scalar constants as tensors on device."""
        def t(v):
            return torch.tensor(v, dtype=torch.float32, device=self.device)
        self.t_r1 = t(self.r1); self.t_r2 = t(self.r2); self.t_r3 = t(self.r3)
        self.t_t1 = t(self.t1); self.t_t2 = t(self.t2); self.t_t3 = t(self.t3)
        self.t_s = t(self.s); self.t_U_th = t(self.U_th); self.t_U_rev = t(self.U_rev)
        self.t_T_am = t(self.T_am); self.t_T_c_in = t(self.T_c_in)
        self.t_C_s_i = t(self.C_s_i); self.t_C_sep = t(self.C_sep)
        self.t_C_he = t(self.C_he); self.t_C_c = t(self.C_c)
        self.t_A_he = t(self.A_he); self.t_k_he = t(self.k_he)
        self.t_c_lye = t(self.c_lye); self.t_c_cw = t(self.c_cw)
        self.t_rho_lye = t(self.rho_lye); self.t_rho_cw = t(self.rho_cw)
        self.t_segma_s = t(self.segma_s); self.t_segma_sep = t(self.segma_sep)
        self.t_A_stack = t(self.A_stack); self.t_A_sep = t(self.A_sep)
        self.t_epsilon_stack = t(self.epsilon_stack); self.t_epsilon_sep = t(self.epsilon_sep)
        self.t_sigma_b = t(self.sigma_b); self.t_F = t(self.F)
        self.t_V_an_lye = t(self.V_an_lye); self.t_V_sep = t(self.V_sep)
        self.t_V_sep_gas = t(self.V_sep_gas); self.t_delta = t(self.delta)
        self.t_D_eff = t(self.D_eff); self.t_K_eff = t(self.K_eff)
        self.t_tau_sep = t(self.tau_sep); self.t_R = t(self.R)
        self.t_mu_lye = t(self.mu_lye); self.t_S_H2_lye = t(self.S_H2_lye)
        self.t_P_sys = t(self.P_sys); self.t_delta_P = t(self.delta_P)
        self.t_N_cell = t(self.N_cell); self.t_A_cell = t(self.A_cell)
        self.t_dt_sub = t(self.dt_sub)

    def _calculate_h2_solubility(self):
        rho_H2O = 1000.0
        M_H2O = 18e-3
        p_atm = 101325.0
        H_H2 = 7.1698e4 * p_atm
        K_H2 = 3.14
        w_lye = 0.30
        S_H2_H2O = rho_H2O * self.P_sys / (M_H2O * p_atm * H_H2)
        return S_H2_H2O / (10**(K_H2 * w_lye))

    def _electrochemical(self, I_i, T_s_i):
        """I_i, T_s_i: (batch, 4) -> Q_ele, U_cell, eta_F: (batch, 4)"""
        T_C = T_s_i - 273.15
        R_ohm = self.t_r1 + self.t_r2 * T_s_i + self.t_r3 * self.t_P_sys
        V_ohm = R_ohm * I_i
        term_act = self.t_t1 + self.t_t2 / T_C + self.t_t3 / (T_C**2)
        arg = term_act * I_i + 1.0
        arg_safe = torch.clamp(arg, min=1e-9)
        V_act = torch.where(arg > 1e-9, self.t_s * torch.log(arg_safe), torch.zeros_like(arg))
        U_cell_i = self.t_U_rev + V_ohm + V_act
        f1 = 50.0 + 2.5 * T_C
        f2 = 0.92 - 6.25e-6 * T_C
        I_sq = I_i**2
        eta_F_i = (I_sq / (f1 + I_sq)) * f2
        Q_ele_i = self.t_N_cell * I_i * (U_cell_i - eta_F_i * self.t_U_th)
        return Q_ele_i, U_cell_i, eta_F_i

    def _derivatives(self, x, u):
        """
        x: (batch, 13)
        u: (batch, 9)
        return dxdt: (batch, 13)
        """
        # Unpack state
        T_s_in = x[:, 0]           # (batch,)
        T_s_i = x[:, 1:5]          # (batch, 4)
        T_sep = x[:, 5]            # (batch,)
        T_c_out = x[:, 6]          # (batch,)
        n_H2_an_i = x[:, 7:11]     # (batch, 4)
        n_H2_sep_liq = x[:, 11]    # (batch,)
        n_H2_sep_gas = x[:, 12]    # (batch,)

        # Unpack action
        I_i = u[:, 0:4]            # (batch, 4)
        v_lye_i = u[:, 4:8]        # (batch, 4)
        v_c = u[:, 8]              # (batch,)

        # --- Electrochemical ---
        Q_ele_i, U_cell_i, eta_F_i = self._electrochemical(I_i, T_s_i)

        # --- Stack Thermal ---
        Q_conv_i = self.t_segma_s * (T_s_i - self.t_T_am)
        Q_rad_i = self.t_epsilon_stack * self.t_sigma_b * self.t_A_stack * (T_s_i**4 - self.t_T_am**4)
        Q_diss_i = Q_conv_i + Q_rad_i
        Q_flow_i = self.t_c_lye * self.t_rho_lye * v_lye_i * (T_s_i - T_s_in.unsqueeze(-1))
        dT_s_dt = (Q_ele_i - Q_diss_i - Q_flow_i) / self.t_C_s_i

        # --- Separator Thermal ---
        v_tot = v_lye_i.sum(dim=-1)  # (batch,)
        # Avoid division by zero
        T_sep_in = torch.where(v_tot > 1e-6, (v_lye_i * T_s_i).sum(dim=-1) / v_tot, T_sep)
        Q_sep_rad = self.t_epsilon_sep * self.t_sigma_b * self.t_A_sep * (T_sep**4 - self.t_T_am**4)
        Q_sep_conv = self.t_segma_sep * (T_sep - self.t_T_am)
        Q_sep_diss = Q_sep_conv + Q_sep_rad
        dT_sep_dt = (0.5 * self.t_c_lye * self.t_rho_lye * v_tot * (T_sep_in - T_sep) - Q_sep_diss) / self.t_C_sep

        # --- Heat Exchanger & Cooling Water ---
        C_rate_lye = self.t_c_lye * self.t_rho_lye * v_tot
        C_rate_cw = self.t_c_cw * self.t_rho_cw * v_c
        delta_T1 = T_s_in - T_c_out
        delta_T2 = T_sep - self.t_T_c_in
        diff = delta_T1 - delta_T2
        prod = delta_T1 * delta_T2
        # Safe LMTD
        log_term = torch.log(torch.abs(delta_T1 / (delta_T2 + 1e-12)) + 1e-12)
        LMTD_general = diff / (log_term + 1e-12)
        LMTD = torch.where(torch.abs(diff) < 1e-5, delta_T1,
                           torch.where(prod <= 0, torch.zeros_like(delta_T1), LMTD_general))
        Q_hx = self.t_k_he * self.t_A_he * LMTD
        dT_s_in_dt = (C_rate_lye * (T_sep - T_s_in) - Q_hx) / self.t_C_he
        dT_c_out_dt = (C_rate_cw * (self.t_T_c_in - T_c_out) + Q_hx) / self.t_C_c

        # --- HTO ---
        n_dot_O2_prod_i = self.t_N_cell * I_i * eta_F_i / (4.0 * self.t_F)
        n_dot_H2_lye_i = self.t_S_H2_lye * self.t_rho_lye * v_lye_i / 4.0
        n_dot_H2_diff_i = self.t_A_cell * self.t_N_cell * self.t_D_eff * self.t_S_H2_lye * self.t_P_sys / self.t_delta
        n_dot_H2_conv_i = (self.t_A_cell * self.t_N_cell * (self.t_K_eff / self.t_mu_lye)
                           * self.t_S_H2_lye * self.t_rho_lye * (self.t_delta_P / self.t_delta))
        n_dot_H2_im_i = n_dot_H2_lye_i + n_dot_H2_diff_i + n_dot_H2_conv_i
        n_dot_H2_im_1_i = n_H2_an_i * v_lye_i / (2.0 * self.t_V_an_lye)
        n_dot_H2_an_i = n_dot_H2_im_i - n_dot_H2_im_1_i

        n_dot_H2_2 = n_H2_sep_liq / self.t_tau_sep
        n_dot_H2_sep_liq = n_dot_H2_im_1_i.sum(dim=-1) - n_dot_H2_2

        sum_n_dot_O2 = n_dot_O2_prod_i.sum(dim=-1)
        n_dot_H2_out = torch.where(sum_n_dot_O2 > 1e-9,
                                   (self.t_R * T_sep * n_H2_sep_gas * sum_n_dot_O2) / (self.t_P_sys * self.t_V_sep_gas),
                                   torch.zeros_like(sum_n_dot_O2))
        n_dot_sep_gas = n_dot_H2_2 - n_dot_H2_out

        # Assemble dxdt
        dxdt = torch.stack([
            dT_s_in_dt,
            dT_s_dt[:, 0], dT_s_dt[:, 1], dT_s_dt[:, 2], dT_s_dt[:, 3],
            dT_sep_dt,
            dT_c_out_dt,
            n_dot_H2_an_i[:, 0], n_dot_H2_an_i[:, 1], n_dot_H2_an_i[:, 2], n_dot_H2_an_i[:, 3],
            n_dot_H2_sep_liq,
            n_dot_sep_gas,
        ], dim=-1)
        return dxdt

    def step(self, x, u):
        """
        RK4 fixed-step integration over self.dt with self.dt_sub sub-steps.
        x: (batch, 13)  float32 tensor on self.device
        u: (batch, 9)   float32 tensor on self.device
        returns x_next: (batch, 13)
        """
        for _ in range(self.n_sub):
            k1 = self._derivatives(x, u)
            k2 = self._derivatives(x + 0.5 * self.dt_sub * k1, u)
            k3 = self._derivatives(x + 0.5 * self.dt_sub * k2, u)
            k4 = self._derivatives(x + self.dt_sub * k3, u)
            x = x + self.dt_sub / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return x

    def reset(self, initial_state=None, batch_size=1):
        """
        Returns initial state tensor of shape (batch_size, 13).
        If initial_state is provided, it should be (batch, 13) or (13,).
        """
        if initial_state is not None:
            x0 = torch.as_tensor(initial_state, dtype=torch.float32, device=self.device)
            if x0.dim() == 1:
                x0 = x0.unsqueeze(0).expand(batch_size, -1)
            elif x0.shape[0] != batch_size:
                # Broadcast if possible
                x0 = x0[0:1].expand(batch_size, -1)
            return x0

        x0 = torch.zeros(batch_size, 13, dtype=torch.float32, device=self.device)
        x0[:, 0] = 345.0
        x0[:, 1:5] = 358.0
        x0[:, 5] = 358.0
        x0[:, 6] = 325.0
        target_HTO_fraction = 0.52 / 100.0
        n_gas = target_HTO_fraction * (self.P_sys * self.V_sep_gas) / (self.R * 358.0)
        x0[:, 7:11] = n_gas / 4.0
        x0[:, 11] = n_gas
        x0[:, 12] = n_gas
        return x0

    def calculate_power(self, I_i, T_s_i):
        """Calculate per-stack and total power."""
        _, U_cell_i, _ = self._electrochemical(I_i, T_s_i)
        P_stack = U_cell_i * I_i * self.t_N_cell
        P_total = P_stack.sum(dim=-1)
        return P_total, P_stack, U_cell_i
