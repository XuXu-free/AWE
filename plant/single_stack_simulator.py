
import numpy as np
from .base_simulator import BaseSimulator
from scipy.integrate import solve_ivp

class SingleStackSimulator(BaseSimulator):
    def __init__(self, dt=1.0):
        super().__init__(dt)
        # --- Parameters ---
        self.N = 1
        self.I_rated = 7800.0
        self.N_cell = 368
        self.A_cell = 2.0
        self.p_sys = 1.6e6  # 1.6 MPa
        self.delta_P = 0.01 * self.p_sys
        
        # Electrochemical
        self.r1, self.r2, self.r3 = 3.202e-5, 8.970e-8, -4.193e-12
        self.t1, self.t2, self.t3 = -1.070e-1, 14.43, 38.8
        self.s = 7.572e-2
        self.U_th = 1.481
        
        # Thermal
        self.T_am = 298.0
        self.T_c_in = 288.0
        self.C_s_i = 3.450e7
        self.C_sep = 5.193e7
        self.C_he = 2.175e7
        self.C_c = 2e7
        self.A_he = 240
        self.k_he = 960
        self.c_lye = 3200
        self.c_cw = 4200
        self.rho_lye = 1280.0
        self.rho_cw = 1000.0
        self.segma_s = 1000
        # Scaled down for single stack system assumption
        self.segma_sep = 50 * self.N 
        
        # Geometry / Radiation
        self.A_stack = 80 
        self.A_sep = 40 
        self.epsilon_stack = 0.8 
        self.epsilon_sep = 0.8 
        self.sigma_b = 5.67e-8 
        
        # HTO
        self.V_an_lye = 2.5
        self.V_sep = 10.288
        self.V_sep_gas = 0.6 * self.V_sep
        self.delta = 500e-6
        self.D_eff = 8.569e-10
        self.K_eff = 2e-16
        self.tau_sep = 60.0
        
        # Solubility
        rho_H2O = 1000
        M_H2O = 18e-3
        p_atm = 101325.0
        H_H2 = 7.1698e4 * p_atm
        K_H2 = 3.14
        w_lye = 0.30
        
        S_H2_H2O = rho_H2O * self.p_sys / (M_H2O * p_atm * H_H2)
        self.S_H2_lye = S_H2_H2O / (10**(K_H2 * w_lye))
        
        self.R = 8.314
        self.mu_lye = 8.76e-4
        
        # Initial Conditions
        self.T_s_init = 358.0 # 85 C
        self.T_sep_init = 345.0 # 72 C
        self.HTO_init = 0.52 # %
        self.T_c_out_init = 325.0 
        
        # Control Limits
        self.v_lye_max = 0.0335
        self.v_lye_min = 0.0101
        self.v_c_max = 0.032
        
    def _calculate_electrochemical_properties(self, I, T_s):
        # T_s in Kelvin
        T_C = T_s - 273.15
        
        # 1. Voltage Model
        U_rev = 1.229
        
        # Ohmic
        R_ohm = self.r1 + self.r2 * T_s + self.r3 * self.p_sys
        V_ohm = R_ohm * I
        
        # Activation
        term_act = self.t1 + self.t2 / T_C + self.t3 / (T_C**2)
        arg = term_act * I + 1.0
        arg_safe = np.maximum(arg, 1e-9)
        V_act = np.where(arg > 1e-9, self.s * np.log(arg_safe), 0.0)
            
        U_cell = U_rev + V_ohm + V_act
        
        # 2. Faraday Efficiency
        f1 = 50.0 + 2.5 * T_C
        f2 = 0.92 - 6.25e-6 * T_C
        
        I_sq = I**2
        eta_F = (I_sq / (f1 + I_sq)) * f2
        
        # 3. Heat Generation
        Q_ele = self.N_cell * I * (U_cell - eta_F * self.U_th)
        
        return Q_ele, U_cell, eta_F

    def _calculate_thermal_derivatives(self, I, v_lye, v_c, T_s_in, T_s, T_sep, T_c_out):
        # All inputs are scalars
        
        # --- 1. Stack Thermal ---
        Q_ele, U_cell, eta_F = self._calculate_electrochemical_properties(I, T_s)
        
        # Dissipation
        Q_conv = self.segma_s * (T_s - self.T_am)
        Q_rad = self.epsilon_stack * self.sigma_b * self.A_stack * (T_s**4 - self.T_am**4)
        Q_diss = Q_conv + Q_rad
        
        # Advection
        Q_flow = self.c_lye * self.rho_lye * v_lye * (T_s - T_s_in)
        
        dT_s_dt = (Q_ele - Q_diss - Q_flow) / self.C_s_i
            
        # --- 2. Separator Thermal ---
        # Single stack: T_sep_in is just T_s
        T_sep_in = T_s
    
        Q_sep_rad = self.epsilon_sep * self.sigma_b * self.A_sep * (T_sep**4 - self.T_am**4)
        Q_sep_conv = self.segma_sep * (T_sep - self.T_am)
        Q_sep_diss = Q_sep_conv + Q_sep_rad

        # v_tot = v_lye for single stack
        dT_sep_dt = (0.5 * self.c_lye * self.rho_lye * v_lye * (T_sep_in - T_sep) - Q_sep_diss) / self.C_sep
        
        # --- 3. Heat Exchanger Thermal ---
        C_rate_lye = self.c_lye * self.rho_lye * v_lye
        C_rate_cw = self.c_cw * self.rho_cw * v_c
        
        # LMTD
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
        
        # Cooling Water Dynamics
        dT_c_out_dt = (C_rate_cw * (self.T_c_in - T_c_out) + Q_hx) / self.C_c
        
        return dT_s_in_dt, dT_s_dt, dT_sep_dt, dT_c_out_dt

    def _calculate_hto_derivatives(self, I, v_lye, n_H2_an, n_H2_sep_liq, n_H2_sep_gas, T_sep, T_s):
        # Scalars
        _, _, eta_F = self._calculate_electrochemical_properties(I, T_s)
        
        # 1. O2 Production Rate (Molar)
        n_dot_O2_prod = self.N_cell * I * eta_F / (4 * 96485.0)
        
        # 2. H2 Impurity Inputs to Anode
        # a) Inflow via Lye Recirculation
        n_dot_H2_lye = self.S_H2_lye * self.rho_lye * v_lye / 4.0
        
        # b) Diffusion
        n_dot_H2_diff = self.A_cell * self.N_cell * self.D_eff * self.S_H2_lye * self.p_sys / self.delta
        
        # c) Convection
        n_dot_H2_conv = self.A_cell * self.N_cell * (self.K_eff / self.mu_lye) * self.S_H2_lye * self.rho_lye * (self.delta_P / self.delta)
        
        n_dot_H2_im = n_dot_H2_lye + n_dot_H2_diff + n_dot_H2_conv
        
        # 3. H2 Outflow from Anode to Separator Liquid
        n_dot_H2_im_1 = n_H2_an * v_lye / (2 * self.V_an_lye)
        
        # Anode H2 Derivative
        n_dot_H2_an = n_dot_H2_im - n_dot_H2_im_1
            
        # --- Separator Liquid Dynamics ---
        # H2 release
        n_dot_H2_2 = n_H2_sep_liq / self.tau_sep
        
        # Liquid H2 Derivative
        n_dot_H2_sep_liq = n_dot_H2_im_1 - n_dot_H2_2
        
        # --- Separator Gas Dynamics ---
        sum_n_dot_O2 = n_dot_O2_prod
        
        if sum_n_dot_O2 > 1e-9:
            n_dot_H2_out = (self.R * T_sep * n_H2_sep_gas * sum_n_dot_O2) / (self.p_sys * self.V_sep_gas)
        else:
            n_dot_H2_out = 0.0
            
        n_dot_sep_gas = n_dot_H2_2 - n_dot_H2_out
        
        return n_dot_H2_an, n_dot_H2_sep_liq, n_dot_sep_gas

    def _get_derivatives(self, t, x, u):
        # x = [T_s_in, T_s, T_sep, T_c_out, n_H2_an, n_H2_sep_liq, n_H2_sep_gas] (7 elements)
        T_s_in = x[0]
        T_s = x[1]
        T_sep = x[2]
        T_c_out = x[3]
        
        n_H2_an = x[4]
        n_H2_sep_liq = x[5]
        n_H2_sep_gas = x[6]
        
        # u = [I, v_lye, v_c] (3 elements)
        I = u[0]
        v_lye = u[1]
        v_c = u[2]
        
        dT_s_in_dt, dT_s_dt, dT_sep_dt, dT_c_out_dt = self._calculate_thermal_derivatives(I, v_lye, v_c, T_s_in, T_s, T_sep, T_c_out)
        n_dot_H2_an, n_dot_H2_sep_liq, n_dot_H2_sep_gas = self._calculate_hto_derivatives(I, v_lye, n_H2_an, n_H2_sep_liq, n_H2_sep_gas, T_sep, T_s)
        
        dxdt = np.array([dT_s_in_dt, dT_s_dt, dT_sep_dt, dT_c_out_dt, n_dot_H2_an, n_dot_H2_sep_liq, n_dot_H2_sep_gas])
        return dxdt

    def reset(self, initial_state=None):
        if initial_state is not None:
            self.state = np.array(initial_state)
        else:
            # x = [T_s_in, T_s, T_sep, T_c_out, n_H2_an, n_H2_sep_liq, n_H2_sep_gas]
            x0 = np.zeros(7)
            
            x0[0] = self.T_s_init
            x0[1] = self.T_s_init
            x0[2] = self.T_sep_init
            x0[3] = self.T_c_out_init
            
            target_HTO_fraction = self.HTO_init / 100.0
            n_gas = target_HTO_fraction * (self.p_sys * self.V_sep_gas) / (self.R * self.T_sep_init)
            
            x0[4] = n_gas / 4 # Approximate anode hold-up scaling
            x0[5] = n_gas
            x0[6] = n_gas
            
            self.state = x0
            
        self.current_time = 0.0
        return self.state

    def step(self, action):
        # action = [I, v_lye, v_c]
        
        fun = lambda t, y: self._get_derivatives(t, y, action)
        t_span = (self.current_time, self.current_time + self.dt)
        
        sol = solve_ivp(fun, t_span, self.state, method='RK45', t_eval=[t_span[1]])
        
        if sol.success:
            self.state = sol.y[:, -1]
        else:
            print(f"Integration failed at t={self.current_time}: {sol.message}")
            self.state = sol.y[:, -1]
            
        self.current_time += self.dt
        return self.state
        
    def get_state(self):
        return self.state
