"""
JIT-compiled GPU-Batched Multi-Stack AWE Simulator

Uses torch.jit.script to eliminate Python interpreter overhead
in the RK4 sub-step loop.
"""

import torch
import torch.jit as jit
from typing import List


# Standalone derivative function for JIT compilation
# All physical constants are passed as a List[Tensor] to avoid *args
@jit.script
def _derivatives_jit(x, u, params: List[torch.Tensor]):
    t_r1 = params[0]
    t_r2 = params[1]
    t_r3 = params[2]
    t_t1 = params[3]
    t_t2 = params[4]
    t_t3 = params[5]
    t_s = params[6]
    t_U_th = params[7]
    t_U_rev = params[8]
    t_T_am = params[9]
    t_T_c_in = params[10]
    t_C_s_i = params[11]
    t_C_sep = params[12]
    t_C_he = params[13]
    t_C_c = params[14]
    t_A_he = params[15]
    t_k_he = params[16]
    t_c_lye = params[17]
    t_c_cw = params[18]
    t_rho_lye = params[19]
    t_rho_cw = params[20]
    t_segma_s = params[21]
    t_segma_sep = params[22]
    t_A_stack = params[23]
    t_A_sep = params[24]
    t_epsilon_stack = params[25]
    t_epsilon_sep = params[26]
    t_sigma_b = params[27]
    t_F = params[28]
    t_V_an_lye = params[29]
    t_V_sep = params[30]
    t_V_sep_gas = params[31]
    t_delta = params[32]
    t_D_eff = params[33]
    t_K_eff = params[34]
    t_tau_sep = params[35]
    t_R = params[36]
    t_mu_lye = params[37]
    t_S_H2_lye = params[38]
    t_P_sys = params[39]
    t_delta_P = params[40]
    t_N_cell = params[41]
    t_A_cell = params[42]

    T_s_in = x[:, 0]
    T_s_i = x[:, 1:5]
    T_sep = x[:, 5]
    T_c_out = x[:, 6]
    n_H2_an_i = x[:, 7:11]
    n_H2_sep_liq = x[:, 11]
    n_H2_sep_gas = x[:, 12]

    I_i = u[:, 0:4]
    v_lye_i = u[:, 4:8]
    v_c = u[:, 8]

    T_C = T_s_i - 273.15
    R_ohm = t_r1 + t_r2 * T_s_i + t_r3 * t_P_sys
    V_ohm = R_ohm * I_i
    term_act = t_t1 + t_t2 / T_C + t_t3 / (T_C * T_C)
    arg = term_act * I_i + 1.0
    arg_safe = torch.clamp(arg, min=1e-9)
    V_act = torch.where(arg > 1e-9, t_s * torch.log(arg_safe), torch.zeros_like(arg))
    U_cell_i = t_U_rev + V_ohm + V_act
    f1 = 50.0 + 2.5 * T_C
    f2 = 0.92 - 6.25e-6 * T_C
    I_sq = I_i * I_i
    eta_F_i = (I_sq / (f1 + I_sq)) * f2
    Q_ele_i = t_N_cell * I_i * (U_cell_i - eta_F_i * t_U_th)

    Q_conv_i = t_segma_s * (T_s_i - t_T_am)
    Q_rad_i = t_epsilon_stack * t_sigma_b * t_A_stack * (T_s_i * T_s_i * T_s_i * T_s_i - t_T_am * t_T_am * t_T_am * t_T_am)
    Q_diss_i = Q_conv_i + Q_rad_i
    Q_flow_i = t_c_lye * t_rho_lye * v_lye_i * (T_s_i - T_s_in.unsqueeze(-1))
    dT_s_dt = (Q_ele_i - Q_diss_i - Q_flow_i) / t_C_s_i

    v_tot = v_lye_i.sum(dim=1)
    T_sep_in = torch.where(v_tot > 1e-6, (v_lye_i * T_s_i).sum(dim=1) / v_tot, T_sep)
    Q_sep_rad = t_epsilon_sep * t_sigma_b * t_A_sep * (T_sep * T_sep * T_sep * T_sep - t_T_am * t_T_am * t_T_am * t_T_am)
    Q_sep_conv = t_segma_sep * (T_sep - t_T_am)
    Q_sep_diss = Q_sep_conv + Q_sep_rad
    dT_sep_dt = (0.5 * t_c_lye * t_rho_lye * v_tot * (T_sep_in - T_sep) - Q_sep_diss) / t_C_sep

    C_rate_lye = t_c_lye * t_rho_lye * v_tot
    C_rate_cw = t_c_cw * t_rho_cw * v_c
    delta_T1 = T_s_in - T_c_out
    delta_T2 = T_sep - t_T_c_in
    diff = delta_T1 - delta_T2
    prod = delta_T1 * delta_T2
    log_term = torch.log(torch.abs(delta_T1 / (delta_T2 + 1e-12)) + 1e-12)
    LMTD_general = diff / (log_term + 1e-12)
    LMTD = torch.where(torch.abs(diff) < 1e-5, delta_T1,
                       torch.where(prod <= 0, torch.zeros_like(delta_T1), LMTD_general))
    Q_hx = t_k_he * t_A_he * LMTD
    dT_s_in_dt = (C_rate_lye * (T_sep - T_s_in) - Q_hx) / t_C_he
    dT_c_out_dt = (C_rate_cw * (t_T_c_in - T_c_out) + Q_hx) / t_C_c

    n_dot_O2_prod_i = t_N_cell * I_i * eta_F_i / (4.0 * t_F)
    n_dot_H2_lye_i = t_S_H2_lye * t_rho_lye * v_lye_i / 4.0
    n_dot_H2_diff_i = t_A_cell * t_N_cell * t_D_eff * t_S_H2_lye * t_P_sys / t_delta
    n_dot_H2_conv_i = (t_A_cell * t_N_cell * (t_K_eff / t_mu_lye)
                       * t_S_H2_lye * t_rho_lye * (t_delta_P / t_delta))
    n_dot_H2_im_i = n_dot_H2_lye_i + n_dot_H2_diff_i + n_dot_H2_conv_i
    n_dot_H2_im_1_i = n_H2_an_i * v_lye_i / (2.0 * t_V_an_lye)
    n_dot_H2_an_i = n_dot_H2_im_i - n_dot_H2_im_1_i

    n_dot_H2_2 = n_H2_sep_liq / t_tau_sep
    n_dot_H2_sep_liq = n_dot_H2_im_1_i.sum(dim=1) - n_dot_H2_2

    sum_n_dot_O2 = n_dot_O2_prod_i.sum(dim=1)
    n_dot_H2_out = torch.where(sum_n_dot_O2 > 1e-9,
                               (t_R * T_sep * n_H2_sep_gas * sum_n_dot_O2) / (t_P_sys * t_V_sep_gas),
                               torch.zeros_like(sum_n_dot_O2))
    n_dot_sep_gas = n_dot_H2_2 - n_dot_H2_out

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


@jit.script
def _step_rk4_jit(x, u, dt_sub: float, n_sub: int, params: List[torch.Tensor]):
    """JIT-compiled RK4 fixed-step integration."""
    for _ in range(n_sub):
        k1 = _derivatives_jit(x, u, params)
        k2 = _derivatives_jit(x + 0.5 * dt_sub * k1, u, params)
        k3 = _derivatives_jit(x + 0.5 * dt_sub * k2, u, params)
        k4 = _derivatives_jit(x + dt_sub * k3, u, params)
        x = x + dt_sub / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x


class BatchMultiStackSimulatorJIT:
    def __init__(self, dt=60.0, dt_sub=0.2, device='cuda'):
        self.dt = dt
        self.dt_sub = dt_sub
        self.n_sub = int(round(dt / dt_sub))
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Parameters (matched to MultiStackSimulator)
        self.N = 4
        self.I_rated = 7800.0
        self.N_cell = 368
        self.A_cell = 2.0
        self.P_sys = 1.6e6
        self.delta_P = 0.01 * self.P_sys
        self.r1 = 3.202e-5
        self.r2 = 8.970e-8
        self.r3 = -4.193e-12
        self.t1 = -1.070e-1
        self.t2 = 14.43
        self.t3 = 38.8
        self.s = 7.572e-2
        self.U_th = 1.481
        self.U_rev = 1.229
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

        self._params = self._build_params_list()

    def _calculate_h2_solubility(self):
        rho_H2O = 1000.0
        M_H2O = 18e-3
        p_atm = 101325.0
        H_H2 = 7.1698e4 * p_atm
        K_H2 = 3.14
        w_lye = 0.30
        S_H2_H2O = rho_H2O * self.P_sys / (M_H2O * p_atm * H_H2)
        return S_H2_H2O / (10**(K_H2 * w_lye))

    def _build_params_list(self):
        def t(v):
            return torch.tensor(v, dtype=torch.float32, device=self.device)
        return [
            t(self.r1), t(self.r2), t(self.r3), t(self.t1), t(self.t2), t(self.t3),
            t(self.s), t(self.U_th), t(self.U_rev), t(self.T_am), t(self.T_c_in),
            t(self.C_s_i), t(self.C_sep), t(self.C_he), t(self.C_c),
            t(self.A_he), t(self.k_he), t(self.c_lye), t(self.c_cw),
            t(self.rho_lye), t(self.rho_cw), t(self.segma_s), t(self.segma_sep),
            t(self.A_stack), t(self.A_sep), t(self.epsilon_stack), t(self.epsilon_sep),
            t(self.sigma_b), t(self.F), t(self.V_an_lye), t(self.V_sep),
            t(self.V_sep_gas), t(self.delta), t(self.D_eff), t(self.K_eff),
            t(self.tau_sep), t(self.R), t(self.mu_lye), t(self.S_H2_lye),
            t(self.P_sys), t(self.delta_P), t(self.N_cell), t(self.A_cell)
        ]

    def step(self, x, u):
        return _step_rk4_jit(x, u, self.dt_sub, self.n_sub, self._params)

    def reset(self, initial_state=None, batch_size=1):
        if initial_state is not None:
            x0 = torch.as_tensor(initial_state, dtype=torch.float32, device=self.device)
            if x0.dim() == 1:
                x0 = x0.unsqueeze(0).expand(batch_size, -1)
            elif x0.shape[0] != batch_size:
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
        t_r1 = self._params[0]
        t_r2 = self._params[1]
        t_r3 = self._params[2]
        t_t1 = self._params[3]
        t_t2 = self._params[4]
        t_t3 = self._params[5]
        t_s = self._params[6]
        t_U_rev = self._params[8]
        t_N_cell = self._params[41]
        t_P_sys = self._params[39]

        T_C = T_s_i - 273.15
        R_ohm = t_r1 + t_r2 * T_s_i + t_r3 * t_P_sys
        V_ohm = R_ohm * I_i
        term_act = t_t1 + t_t2 / T_C + t_t3 / (T_C * T_C)
        arg = term_act * I_i + 1.0
        arg_safe = torch.clamp(arg, min=1e-9)
        V_act = torch.where(arg > 1e-9, t_s * torch.log(arg_safe), torch.zeros_like(arg))
        U_cell_i = t_U_rev + V_ohm + V_act
        P_stack = U_cell_i * I_i * t_N_cell
        P_total = P_stack.sum(dim=-1)
        return P_total, P_stack, U_cell_i
