import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .barrier_net import BarrierNet

class AWEBarrierNet(BarrierNet):
    def __init__(
        self,
        n_features: int,
        n_hidden1: int,
        n_hidden21: int,
        n_hidden22: int,
        n_cls: int,
        mean,
        std,
        device=None,
        bn: bool = False,
        u_min=None,
        u_max=None,
    ):
        # Initialize base class
        # Note: obs parameters are not used here but kept for compatibility
        super().__init__(
            n_features=n_features,
            n_hidden1=n_hidden1,
            n_hidden21=n_hidden21,
            n_hidden22=n_hidden22,
            n_cls=n_cls,
            mean=mean,
            std=std,
            device=device,
            bn=bn,
            u_min=u_min,
            u_max=u_max
        )
        
        # --- System Parameters (Matched to MultiStackSimulator) ---
        self.n_stacks = 4
        self.n_cells = 368
        self.A_cell = 2.0
        self.I_rated = 7800.0
        
        # Electrochemical Parameters
        self.U_rev = 1.229
        self.r1 = 3.202e-5
        self.r2 = 8.970e-8
        self.r3 = -4.193e-12
        self.P_sys = 1.6e6
        self.R = 8.314
        self.V_sep = 10.288
        self.V_sep_gas = 0.6 * self.V_sep
        self.s = 7.572e-2
        self.t1 = -1.070e-1
        self.t2 = 14.43
        self.t3 = 38.8
        
        # Thermal Parameters
        self.C_s_i = 3.450e7  # Stack Heat Capacity
        self.C_sep = 5.193e7  # Separator Heat Capacity
        self.C_he = 2.175e7   # HX Lye Side Heat Capacity
        self.C_c = 2.0e7      # HX CW Side Heat Capacity (Estimated)
        
        # self.h_A_stack = 1000.0 # Convective/Rad coeff estimate
        self.T_amb = 298.15
        
        self.kA_hx = 960.0 * 240.0 # k_he * A_he = 230400.0
        self.T_cw_in = 288.15 # 288K
        self.c_cw = 4200.0
        self.rho_cw = 1000.0
        self.c_lye = 3200.0
        self.rho_lye = 1280.0
        
        self.k_lye = self.c_lye * self.rho_lye
        self.k_cw = self.c_cw * self.rho_cw
        
        self.F_const = 96485.0
        self.U_th = 1.481
        
        # Additional Thermal Params from Simulator
        self.epsilon_stack = 0.8
        self.sigma_b = 5.67e-8
        self.A_stack = 80.0
        self.segma_s = 1000.0
        
        self.epsilon_sep = 0.8
        self.A_sep = 40.0
        self.segma_sep = 50.0 * 4
        
        # HTO Params
        self.V_an_lye = 2.5
        self.delta = 500e-6
        self.D_eff = 8.569e-10
        self.K_eff = 2e-16
        self.mu_lye = 8.76e-4
        self.delta_P = 0.01 * self.P_sys
        self.tau_sep = 60.0
        
        # Safety Constraints
        self.T_min = 293.15
        self.T_max = 363.15
        self.HTO_pct_max = 2.0
        self.HTO_pct_min = 0.0
        self.U_cell_max = 2.2
        self.U_cell_min = 0.0
        self.P_stack_max = 6.0e6
        self.P_stack_min = 0.0
        
        # CBF Parameters
        self.gamma_T = 0.01 # Tuned for dt=60s
        self.gamma_HTO = 0.01 # Tuned for dt=60s
        self.gamma_U = 1.0 # Instantaneous
        self.gamma_P = 1.0 # Instantaneous
        
        # Safety Margins (to handle linearization error and time discretization)
        self.margin_T = 5.0 # K
        self.margin_HTO = 0.2 # %
        
    def _calculate_h2_solubility(self):
        rho_H2O = 1000.0
        M_H2O = 18e-3
        p_atm = 101325.0
        H_H2 = 7.1698e4 * p_atm
        K_H2 = 3.14
        w_lye = 0.30
        
        S_H2_H2O = rho_H2O * self.P_sys / (M_H2O * p_atm * H_H2)
        S_H2_lye = S_H2_H2O / (10**(K_H2 * w_lye))
        return S_H2_lye

    def dcbf(self, x0: torch.Tensor, u_ref: torch.Tensor, cbf_params: torch.Tensor, sgn: int) -> torch.Tensor:
        """
        Differentiable CBF layer for AWE system.
        x0: Normalized state (Batch, 13)
        u_ref: Reference control (Batch, 9)
        cbf_params: Learnable parameters for CBF (Batch, 2) - Not used in this explicit formulation yet
        """
        # 1. Denormalize State
        x = x0
        
        batch_size = x.shape[0]
        
        # Unpack state
        # x = [T_s_in, T_s1...4, T_sep, T_c_out, n_H2_an1...4, n_liq, n_gas]
        T_s_in = x[:, 0]
        T_s = x[:, 1:5] # (Batch, 4)
        T_sep = x[:, 5]
        T_c_out = x[:, 6]
        n_H2_an = x[:, 7:11] # (Batch, 4)
        n_liq = x[:, 11]
        n_gas = x[:, 12]
        
        # Current Control Input Guess (u_ref)
        # u = [I1...4, v_lye1...4, v_c]
        u_curr = u_ref.clone()
        I_curr = u_curr[:, 0:4]
        
        # --- 2. Construct CBF Constraints ---
        # We need to form A_cbf * u <= b_cbf
        
        # 2.1 Calculate f(x) and G(x) components
        # LMTD
        delta_T1 = T_s_in - T_c_out
        delta_T2 = T_sep - self.T_cw_in
        
        # Approximate LMTD to avoid singularity in autodiff
        numerator = delta_T1 - delta_T2
        denominator = torch.log(torch.abs(delta_T1 / delta_T2) + 1e-6)
        LMTD = numerator / (denominator + 1e-6)
        # Handle cases where delta_T1 ~ delta_T2
        mask = torch.abs(delta_T1 - delta_T2) < 1e-4
        LMTD[mask] = delta_T1[mask]
        
        Q_hx = self.kA_hx * LMTD
        
        # Q_diss for Stacks
        Q_diss = self.segma_s * (T_s - self.T_amb) + \
                 self.epsilon_stack * self.sigma_b * self.A_stack * (T_s.pow(4) - self.T_amb**4)
                 
        # Q_sep_diss
        Q_sep_diss = self.segma_sep * (T_sep - self.T_amb) + \
                     self.epsilon_sep * self.sigma_b * self.A_sep * (T_sep.pow(4) - self.T_amb**4)
                     
        # HTO Constants
        S_H2_lye = self._calculate_h2_solubility()
        d_H2 = self.A_cell * self.n_cells * (
            self.D_eff * S_H2_lye * self.P_sys / self.delta + 
            (self.K_eff / self.mu_lye) * S_H2_lye * self.rho_lye * self.delta_P / self.delta
        )
        
        phi_x = S_H2_lye * self.rho_lye / 4.0 - n_H2_an / (2.0 * self.V_an_lye)
        kappa_x = self.R * T_sep * n_gas / (self.P_sys * self.V_sep_gas)
        
        # 2.2 Jacobian J_u calculation
        
        # Voltage Model (Vectorized)
        T_C = T_s - 273.15
        
        # U_cell
        R_ohm = self.r1 + self.r2 * T_s + self.r3 * self.P_sys
        V_ohm = R_ohm * I_curr
        
        term_act = self.t1 + self.t2 / T_C + self.t3 / T_C.pow(2)
        arg = term_act * I_curr + 1.0
        arg_safe = torch.clamp(arg, min=1e-6)
        V_act = self.s * torch.log(arg_safe)
        U_cell = self.U_rev + V_ohm + V_act
        
        # Eta_F
        f1 = 50.0 + 2.5 * T_C
        f2 = 0.92 - 6.25e-6 * T_C
        I_sq = I_curr.pow(2)
        eta_F = (I_sq / (f1 + I_sq)) * f2
        
        # Derivatives
        dU_dI = R_ohm + self.s * term_act / arg_safe
        deta_dI = f2 * (2 * I_curr * f1) / (f1 + I_sq).pow(2)
        
        # dQ/dI
        dQ_dI = self.n_cells * ( (U_cell - eta_F * self.U_th) + I_curr * (dU_dI - deta_dI * self.U_th) )
        
        # dn_O2/dI
        const_F = self.n_cells / (4.0 * self.F_const)
        dnO2_dI = const_F * (eta_F + I_curr * deta_dI)
        
        
        G_u = [] # List of G_i * u <= h_i
        H_u = []
        
        # --- Barrier 1: Stack Temperature Max ---
        # h1 = T_max - T_si >= 0
        # dot(h1) >= -gamma * h1
        # -dot(T_si) >= -gamma * h1 => dot(T_si) <= gamma * h1
        # dot(T_si) approx dot_0 + A * (u - u0)
        # dot_0 + A * u - A * u0 <= gamma * h1
        # A * u <= gamma * h1 - dot_0 + A * u0
        
        Q_ele = self.n_cells * I_curr * (U_cell - eta_F * self.U_th)
        Q_flow_term = self.k_lye * (T_s - T_s_in.unsqueeze(1)) # (Batch, 4)
        v_lye_curr = u_curr[:, 4:8]
        T_si_dot_0 = (Q_ele - Q_diss - Q_flow_term * v_lye_curr) / self.C_s_i
        
        for i in range(4):
            # A_row represents gradient of T_si_dot w.r.t u
            # a_Ii = d(T_si_dot)/dI
            a_Ii = 1.0 / self.C_s_i * dQ_dI[:, i]
            # a_vi = d(T_si_dot)/dv_lye = -Q_flow_term / C_s_i
            a_vi = - (self.k_lye / self.C_s_i) * (T_s[:, i] - T_s_in)
            
            A_row = torch.zeros(batch_size, 9, device=x.device)
            A_row[:, i] = a_Ii
            A_row[:, 4+i] = a_vi
            
            h_val = self.T_max - self.margin_T - T_s[:, i]
            # Constraint: A * u <= b
            b_row = self.gamma_T * h_val - T_si_dot_0[:, i] + (A_row * u_curr).sum(dim=1)
            
            G_u.append(A_row)
            H_u.append(b_row)

        # --- Barrier 1b: Stack Temperature Min ---
        # h1b = T_si - T_min >= 0
        # dot(h1b) >= -gamma * h1b
        # dot(T_si) >= -gamma * h1b
        # dot_0 + A * (u - u0) >= -gamma * h1b
        # -A * u <= gamma * h1b + dot_0 - A * u0
        
        for i in range(4):
            # Reuse A_row from Max constraint (recalculate for clarity)
            a_Ii = 1.0 / self.C_s_i * dQ_dI[:, i]
            a_vi = - (self.k_lye / self.C_s_i) * (T_s[:, i] - T_s_in)
            
            A_row = torch.zeros(batch_size, 9, device=x.device)
            A_row[:, i] = a_Ii
            A_row[:, 4+i] = a_vi
            
            h_val_min = T_s[:, i] - self.T_min
            # Constraint: -A * u <= b
            b_row_min = self.gamma_T * h_val_min + T_si_dot_0[:, i] - (A_row * u_curr).sum(dim=1)
            
            G_u.append(-A_row)
            H_u.append(b_row_min)

        # --- Exact HTO Dynamics ---
        # HTO_pct = (n_gas * R * T_sep) / (P_sys * V_sep_gas) * 100
        # dot_HTO = c_HTO * (dot_n_gas * T_sep + n_gas * dot_T_sep)
        c_HTO = 100.0 * self.R / (self.P_sys * self.V_sep_gas)
        HTO_pct = c_HTO * n_gas * T_sep
        
        n_dot_O2 = const_F * I_curr * eta_F # (Batch, 4)
        sum_n_dot_O2 = n_dot_O2.sum(dim=1)
        n_gas_dot_0 = (1.0/self.tau_sep) * n_liq - kappa_x * sum_n_dot_O2
        
        # dot_T_sep = 1/C_sep * ( 0.5 * k_lye * sum(v_lye_i * (T_s_i - T_sep)) - Q_sep_diss )
        v_lye_curr = u_curr[:, 4:8]
        T_sep_diff = T_s - T_sep.unsqueeze(1) # (Batch, 4)
        T_sep_flow_term = 0.5 * self.k_lye * (v_lye_curr * T_sep_diff).sum(dim=1)
        T_sep_dot_0 = (T_sep_flow_term - Q_sep_diss) / self.C_sep
        
        HTO_dot_0 = c_HTO * (n_gas_dot_0 * T_sep + n_gas * T_sep_dot_0)
        
        # Gradients for HTO_dot
        A_row_hto = torch.zeros(batch_size, 9, device=x.device)
        for i in range(4):
            # d(dot_HTO) / dI_i = c_HTO * T_sep * d(dot_n_gas)/dI_i
            A_row_hto[:, i] = c_HTO * T_sep * (-kappa_x * dnO2_dI[:, i])
            # d(dot_HTO) / dv_lye_i = c_HTO * n_gas * d(dot_T_sep)/dv_lye_i
            A_row_hto[:, 4+i] = c_HTO * n_gas * (0.5 * self.k_lye * T_sep_diff[:, i] / self.C_sep)

        # --- Barrier 2: HTO Max ---
        # h2 = HTO_max - HTO_pct >= 0
        # dot(h2) = -dot_HTO >= -gamma * h2 => dot_HTO <= gamma * h2
        # dot_HTO_0 + A * (u - u0) <= gamma * h2
        # A * u <= gamma * h2 - dot_HTO_0 + A * u0
        
        h_val_hto_max = self.HTO_pct_max - self.margin_HTO - HTO_pct
        b_row_hto_max = self.gamma_HTO * h_val_hto_max - HTO_dot_0 + (A_row_hto * u_curr).sum(dim=1)
        
        G_u.append(A_row_hto)
        H_u.append(b_row_hto_max)
        
        # --- Barrier 3: Voltage Max ---
        # h3 = U_max - U_cell_i >= 0
        # h3 is function of (I_i, T_si). Relative degree 0 w.r.t I_i if I_i is control.
        # But wait, I_i is control input u. So relative degree is 0.
        # This is a static constraint on input u: g(u, x) >= 0.
        # h3(u, x) = U_max - U_cell(I_i, T_si) >= 0
        # Linearize h3 around u0:
        # h3(u0) + dh3/du * (u - u0) >= 0
        # h3(u0) - dU/dI * (I - I0) >= 0
        # dU/dI * I <= h3(u0) + dU/dI * I0
        # dU/dI * I <= U_max - U_cell(I0) + dU/dI * I0
        # dU/dI * I <= U_max - (U_cell(I0) - dU/dI * I0)
        
        for i in range(4):
            # A_row * u <= b_row
            A_row_u = torch.zeros(batch_size, 9, device=x.device)
            A_row_u[:, i] = dU_dI[:, i]
            
            # h_val = U_max - U_cell_0
            # Linearization: U_cell(I) ~ U_cell(I0) + dU/dI * (I - I0)
            # U_max - (U0 + J(I-I0)) >= 0
            # J * I <= U_max - U0 + J * I0
            
            b_row_u = self.U_cell_max - U_cell[:, i] + dU_dI[:, i] * I_curr[:, i]
            
            # Apply gamma? No, this is relative degree 0 constraint, direct bound.
            # But we can add it to G_u, H_u list which is solved by QP.
            
            G_u.append(A_row_u)
            H_u.append(b_row_u)
            
        # --- Barrier 4: Voltage Min ---
        # h4 = U_cell_i - U_min >= 0
        # U_cell(I) >= U_min
        # U0 + J(I-I0) >= U_min
        # -J * I <= U0 - J*I0 - U_min
        
        for i in range(4):
            A_row_u_min = torch.zeros(batch_size, 9, device=x.device)
            A_row_u_min[:, i] = -dU_dI[:, i]
            
            b_row_u_min = U_cell[:, i] - dU_dI[:, i] * I_curr[:, i] - self.U_cell_min
            
            G_u.append(A_row_u_min)
            H_u.append(b_row_u_min)
            
        # --- Barrier 5: Power Max ---
        # h5 = P_max - P_stack_i >= 0
        # P_stack_i = N_cell * U_cell_i * I_i
        # dP/dI = N_cell * (U_cell + I * dU/dI)
        dP_dI = self.n_cells * (U_cell + I_curr * dU_dI)
        P_stack = self.n_cells * U_cell * I_curr
        
        # P(I) <= P_max
        # P0 + dP/dI * (I - I0) <= P_max
        # dP/dI * I <= P_max - P0 + dP/dI * I0
        
        for i in range(4):
            A_row_p = torch.zeros(batch_size, 9, device=x.device)
            A_row_p[:, i] = dP_dI[:, i]
            
            b_row_p = self.P_stack_max - P_stack[:, i] + dP_dI[:, i] * I_curr[:, i]
            
            G_u.append(A_row_p)
            H_u.append(b_row_p)
            
        # --- Barrier 5b: Power Min ---
        # h5b = P_stack_i - P_min >= 0
        # P(I) >= P_min
        # P0 + dP/dI * (I - I0) >= P_min
        # -dP/dI * I <= P0 - dP/dI * I0 - P_min
        
        for i in range(4):
            A_row_p_min = torch.zeros(batch_size, 9, device=x.device)
            A_row_p_min[:, i] = -dP_dI[:, i]
            
            b_row_p_min = P_stack[:, i] - dP_dI[:, i] * I_curr[:, i] - self.P_stack_min
            
            G_u.append(A_row_p_min)
            H_u.append(b_row_p_min)
            
        # --- Barrier 6: HTO Min ---
        # h6 = HTO_pct - HTO_min >= 0 (Usually 0, but maybe safety margin)
        # dot(h6) = dot_HTO >= -gamma * h6
        # dot_HTO_0 + A * (u - u0) >= -gamma * h6
        # -A * u <= gamma * h6 + dot_HTO_0 - A * u0
        
        h_val_hto_min = HTO_pct - self.HTO_pct_min
        b_row_hto_min = self.gamma_HTO * h_val_hto_min + HTO_dot_0 - (A_row_hto * u_curr).sum(dim=1)
        
        G_u.append(-A_row_hto)
        H_u.append(b_row_hto_min)
        
        # --- Barrier 7: Actuator Limits (Box Constraints) ---
        # 7a: Max Limits: u <= u_max  => I*u <= u_max
        # 7b: Min Limits: u >= u_min  => -I*u <= -u_min
        
        if self.u_max is not None:
            # u_max can be tensor or float
            # Assuming u_max is tensor (Dim,) or (Batch, Dim)
            # If (Dim,), expand to (Batch, Dim)
            u_max_t = self.u_max
            if u_max_t.dim() == 1:
                u_max_t = u_max_t.unsqueeze(0).expand(batch_size, -1)
                
            for k in range(9): # 9 controls
                # I*u <= u_max
                # e_k * u <= u_max_k
                A_row_box = torch.zeros(batch_size, 9, device=x.device)
                A_row_box[:, k] = 1.0
                b_row_box = u_max_t[:, k]
                
                G_u.append(A_row_box)
                H_u.append(b_row_box)
                
        if self.u_min is not None:
            u_min_t = self.u_min
            if u_min_t.dim() == 1:
                u_min_t = u_min_t.unsqueeze(0).expand(batch_size, -1)
                
            for k in range(9):
                # -I*u <= -u_min
                # -e_k * u <= -u_min_k
                A_row_box = torch.zeros(batch_size, 9, device=x.device)
                A_row_box[:, k] = -1.0
                b_row_box = -u_min_t[:, k]
                
                G_u.append(A_row_box)
                H_u.append(b_row_box)
        
        # --- Stack and Solve ---
        G = torch.stack(G_u, dim=1) # (Batch, Num_Constraints, 9)
        H = torch.stack(H_u, dim=1) # (Batch, Num_Constraints)
        
        # Project u_ref onto G*u <= H using QP solver
        # u_safe = self._project_halfspace(u_ref, G, H)
        u_safe = self._solve_qp(u_ref, G, H)
        
        # Clamp to box constraints (Redundant but safe)
        if self.u_max is not None and self.u_min is not None:
            u_safe = torch.max(torch.min(u_safe, self.u_max), self.u_min)
        
        return u_safe

    def _solve_qp(self, u_ref, G, h, max_iter=100, lr=0.01, tol=1e-4):
        """
        Solve QP: min 0.5 * ||u - u_ref||^2 s.t. G u <= h
        Using Dual Projected Gradient Ascent.
        Dual: max_{\lambda >= 0} -0.5 * \lambda^T (G G^T) \lambda - \lambda^T (G u_ref - h)
        """
        # G: (Batch, Num_Constraints, Dim)
        # h: (Batch, Num_Constraints)
        # u_ref: (Batch, Dim)
        
        # Normalize constraints to improve conditioning
        # scale factors: ||G_i||
        scale = torch.norm(G, dim=2, keepdim=True) # (Batch, Num, 1)
        scale = torch.clamp(scale, min=1e-6)
        
        G_norm = G / scale
        h_norm = h / scale.squeeze(2)
        
        batch_size, num_constraints, dim = G.shape
        
        # Q = G @ G.T -> (Batch, Num_Constraints, Num_Constraints)
        Q = torch.bmm(G_norm, G_norm.transpose(1, 2))
        
        # c = G @ u_ref.unsqueeze(2) - h.unsqueeze(2) -> (Batch, Num_Constraints, 1)
        c = torch.bmm(G_norm, u_ref.unsqueeze(2)).squeeze(2) - h_norm
        
        # Initialize lambda
        lambda_ = torch.zeros(batch_size, num_constraints, device=G.device)
        
        for i in range(max_iter):
            # Gradient of dual objective w.r.t lambda: -Q lambda + c
            # We want to maximize: -0.5 lambda^T Q lambda + lambda^T (G u_ref - h)
            # Gradient is: -Q lambda + (G u_ref - h)
            grad = -torch.bmm(Q, lambda_.unsqueeze(2)).squeeze(2) + c
            
            # Update
            lambda_new = torch.relu(lambda_ + lr * grad)
            
            # Check convergence
            # if torch.norm(lambda_new - lambda_) < tol:
            #     lambda_ = lambda_new
            #     break
                
            lambda_ = lambda_new
            
        # Recover primal: u = u_ref - G^T lambda
        correction = torch.bmm(G_norm.transpose(1, 2), lambda_.unsqueeze(2)).squeeze(2)
        u_star = u_ref - correction
        
        return u_star
