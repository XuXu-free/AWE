
import numpy as np
from .base_simulator import BaseSimulator
from scipy.integrate import solve_ivp

class MultiStackSimulator(BaseSimulator):
    def __init__(self, dt=0.2):
        super().__init__(dt)
        # --- Parameters from Table 2 and Paper ---
        self.N = 4
        self.I_rated = 7800.0
        self.N_cell = 368
        self.A_cell = 2.0
        self.p_sys = 1.6e6  # 1.6 MPa
        # NOTE unsure
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
        # NOTE unknown
        self.C_c = 2e7
        # NOTE unsure
        self.A_he = 240 # Heat exchange area A_c from Table A1
        # NOTE unsure
        self.k_he = 960 # Heat transfer coefficient k from Table A1
        self.c_lye = 3200  # From text
        self.c_cw = 4200
        self.rho_lye = 1280.0 # From text
        self.rho_cw = 1000.0
        # NOTE unknown
        self.segma_s = 1000
        # NOTE unknown
        self.segma_sep = 50 * 4
        
        # New thermal parameters
        # NOTE unknown
        self.A_stack = 80 # Estimated stack surface area [m^2]
        # NOTE unknown
        self.A_sep = 40 # Estimated separator surface area [m^2]
        # NOTE unsure
        self.epsilon_stack = 0.8 # Emissivity
        # NOTE unsure
        self.epsilon_sep = 0.8 # Emissivity
        self.sigma_b = 5.67e-8 # Stefan-Boltzmann constant [W/m^2 K^4]
        
        
        # HTO
        self.V_an_lye = 2.5
        self.V_sep = 10.288
        # NOTE unsure
        self.V_sep_gas = 0.6 * self.V_sep # Assumption (approx half of V_sep)
        self.delta = 500e-6
        self.D_eff = 8.569e-10
        self.K_eff = 2e-16
        self.tau_sep = 60.0
        
        self.S_H2_lye = self._calculate_h2_solubility()
        
        self.R = 8.314
        self.mu_lye = 8.76e-4
        
        # Initial Conditions (Open Loop)
        self.T_s_init = 358.0 # 85 C
        self.T_s_in_init = 345.0 # 72 C / 345K
        self.T_sep_init = 358.0 # 85 C
        self.HTO_init = 0.52 # %
        self.T_c_out_init = 325.0 
        
        # Control Limits
        self.T_min = 20.0
        self.T_max = 90.0
        self.I_min = 0.0
        self.I_max = 7800.0 * 1.2
        self.v_lye_min = 0.0
        self.v_lye_max = 0.1
        self.v_c_min = 0.0
        self.v_c_max = 1.0
        
        self.P_stack_max = 6.0e6
        self.P_stack_min = 0.0
        self.U_cell_min = 0.0
        self.U_cell_max = 2.2
        
    def _calculate_h2_solubility(self):
        # Calculate H2 Solubility in Lye (S_H2) [mol/(m^3 Pa)]
        # Based on Secchenov equation and Henry's Law at 80C
        rho_H2O = 1000     # kg/m^3 at 80C
        M_H2O = 18e-3   # kg/mol
        p_atm = 101325.0    # Pa = kg/(m*s^2)
        H_H2 = 7.1698e4 * p_atm # Pa = kg/(m*s^2)
        K_H2 = 3.14         # Secchenov parameter
        w_lye = 0.30        # 30 wt% KOH
        
        # S_H2_H2O = rho_H2O / (M_H2O * p_atm * H_H2)
        S_H2_H2O = rho_H2O * self.p_sys / (M_H2O * p_atm * H_H2)
        
        # S_H2_lye = S_H2_H2O / 10^(K_H2 * w_lye)
        return S_H2_H2O / (10**(K_H2 * w_lye))

    def _calculate_electrochemical_properties(self, I_i, T_s_i):
        # T_s_i is in Kelvin (Vector)
        T_C = T_s_i - 273.15
        
        # 1. Voltage Model
        # U_cell = U_rev + (r1 + r2*T + r3*P)*I + s * log( (t1 + t2/T_C + t3/T_C^2)*I + 1 )
        
        U_rev = 1.229
        
        # Ohmic
        R_ohm = self.r1 + self.r2 * T_s_i + self.r3 * self.p_sys
        V_ohm = R_ohm * I_i
        
        # Activation (T in Celsius)
        # Term inside log
        term_act = self.t1 + self.t2 / T_C + self.t3 / (T_C**2)
        
        # Avoid math domain error if term_act * I + 1 <= 0
        arg = term_act * I_i + 1.0
        # Use safe argument for log to avoid warnings, then mask result
        arg_safe = np.maximum(arg, 1e-9)
        V_act = np.where(arg > 1e-9, self.s * np.log(arg_safe), 0.0)
            
        U_cell_i = U_rev + V_ohm + V_act
        
        # 2. Faraday Efficiency
        # f1 = 50 + 2.5 T (A^2). 
        f1 = 50.0 + 2.5 * T_C
        f2 = 0.92 - 6.25e-6 * T_C
        
        # eta = I^2 / ( f1 + I^2 ) * f2
        I_sq = I_i**2
        eta_F_i = (I_sq / (f1 + I_sq)) * f2
        
        # 3. Heat Generation
        # Q = N * I * (U - eta * U_th)
        Q_ele_i = self.N_cell * I_i * (U_cell_i - eta_F_i * self.U_th)
        
        return Q_ele_i, U_cell_i, eta_F_i

    def _calculate_thermal_derivatives(self, I_i, v_lye_i, v_c, T_s_in, T_s_i, T_sep, T_c_out):
        v_tot = np.sum(v_lye_i)
        
        # --- 1. Stack Thermal (Vectorized) ---
        Q_ele_i, U_cell_i, eta_F_i = self._calculate_electrochemical_properties(I_i, T_s_i)
        
        # Dissipation: Q_diss = Q_conv + Q_rad
        # delta_T = np.maximum(T_s_i - self.T_am, 0.0)
        # h_conv = 2.51 * 0.52 * (delta_T / self.phi_stack)**0.25
        Q_conv_i = self.segma_s * (T_s_i - self.T_am)
        
        # Q_rad = epsilon * sigma * A * (T_s^4 - T_am^4)
        Q_rad_i = self.epsilon_stack * self.sigma_b * self.A_stack * (T_s_i**4 - self.T_am**4)
        # Q_rad_i = 0
        
        Q_diss_i = Q_conv_i + Q_rad_i
        
        # Advection: Q_flow = c * rho * v * (T_out - T_in)
        # T_in = T_s_in (from Heat Exchanger)
        Q_flow_i = self.c_lye * self.rho_lye * v_lye_i * (T_s_i - T_s_in)
        
        dT_s_dt = (Q_ele_i - Q_diss_i - Q_flow_i) / self.C_s_i
            
        # --- 2. Separator Thermal ---
        # T_sep_in (Weighted Average)
        if v_tot > 1e-6:
            T_sep_in = np.sum(v_lye_i * T_s_i) / v_tot
        else:
            T_sep_in = T_sep
    
        Q_sep_rad = self.epsilon_sep * self.sigma_b * self.A_sep * (T_sep**4 - self.T_am**4)
        # Q_sep_rad = 0

        Q_sep_conv = self.segma_sep * (T_sep - self.T_am)

        Q_sep_diss = Q_sep_conv + Q_sep_rad

        dT_sep_dt = (0.5 * self.c_lye * self.rho_lye * v_tot * (T_sep_in - T_sep) - Q_sep_diss) / self.C_sep
        
        # --- 3. Heat Exchanger Thermal ---
        # T_sep_out = T_sep (Assuming well mixed / output temp)
        # Eq 17: C_he dT_s_in/dt = c v rho (T_sep_out - T_s_in) - k A DeltaT
        
        C_rate_lye = self.c_lye * self.rho_lye * v_tot
        C_rate_cw = self.c_cw * self.rho_cw * v_c
        
        # Calculate LMTD based on Eq 19
        # Delta T = ((T_s_in - T_c_out) - (T_sep_out - T_c_in)) / ln(...)
        
        delta_T1 = T_s_in - T_c_out
        delta_T2 = T_sep - self.T_c_in
        
        if abs(delta_T1 - delta_T2) < 1e-5:
            LMTD = delta_T1
        elif delta_T1 * delta_T2 <= 0:
            # Handle crossover or invalid log domain (e.g. during initialization or transients)
            LMTD = 0.0
        else:
            LMTD = (delta_T1 - delta_T2) / np.log(delta_T1 / delta_T2)
            
        # Heat transfer rate Q_hx = k * Ac * LMTD
        Q_hx = self.k_he * self.A_he * LMTD
            
        dT_s_in_dt = (C_rate_lye * (T_sep - T_s_in) - Q_hx) / self.C_he
        
        # Cooling Water Dynamics (T_c_out)
        # C_cw * dT_c_out/dt = m_dot_cw * c_cw * (T_c_in - T_c_out) + Q_hx
        dT_c_out_dt = (C_rate_cw * (self.T_c_in - T_c_out) + Q_hx) / self.C_c
        
        return dT_s_in_dt, dT_s_dt, dT_sep_dt, dT_c_out_dt

    def _get_derivatives(self, t, x, u):
        # Unpack State
        # x = [T_s_in, T_s1...4, T_sep, T_c_out, n_H2_an1...4, n_H2_sep_liq, n_H2_sep_gas]
        T_s_in = x[0]
        T_s_i = x[1:5]
        T_sep = x[5]
        T_c_out = x[6]
        
        n_H2_an_i = x[7:11]
        n_H2_sep_liq = x[11]
        n_H2_sep_gas = x[12]
        
        # Unpack Inputs
        # u = [I1...4, v_lye1...4, v_c]
        I_i = u[0:4]
        v_lye_i = u[4:8]
        v_c = u[8]
        
        # Calculate Derivatives
        Q_ele_i, U_cell_i, eta_F_i = self._calculate_electrochemical_properties(I_i, T_s_i)

        dT_s_in_dt, dT_s_dt, dT_sep_dt, dT_c_out_dt = self._calculate_thermal_derivatives(I_i, v_lye_i, v_c, T_s_in, T_s_i, T_sep, T_c_out)
        n_dot_H2_an, n_dot_H2_sep_liq, n_dot_H2_sep_gas = self._calculate_hto_derivatives(I_i, v_lye_i, n_H2_an_i, n_H2_sep_liq, n_H2_sep_gas, T_sep, T_s_i, eta_F_i)
        
        dxdt = np.concatenate([[dT_s_in_dt], dT_s_dt, [dT_sep_dt], [dT_c_out_dt], n_dot_H2_an, [n_dot_H2_sep_liq], [n_dot_H2_sep_gas]])
        return dxdt


    def _calculate_hto_derivatives(self, I_i, v_lye_i, n_H2_an_i, n_H2_sep_liq, n_H2_sep_gas, T_sep, T_s_i, eta_F_i):
        """
        Calculate derivatives for Hydrogen-in-Oxygen (HTO) impurity dynamics.
        Based on mass balance in Anode, Separator Liquid, and Separator Gas phases.
        """
        # Calculate Faraday Efficiency for O2 production (Vectorized)
        # eta_F_i is now passed as argument
        
        # 1. O2 Production Rate (Molar)
        # n_dot_O2 = (N_cell * I * eta_F) / (4 * F)
        n_dot_O2_prod_i = self.N_cell * I_i * eta_F_i / (4 * 96485.0)
        
        # 2. H2 Impurity Inputs to Anode (n_dot_H2_im)
        # a) Inflow via Lye Recirculation
        n_dot_H2_lye_i = self.S_H2_lye * self.rho_lye * v_lye_i / 4.0
        
        # b) Diffusion across membrane (Fick's Law)
        n_dot_H2_diff_i = self.A_cell * self.N_cell * self.D_eff * self.S_H2_lye * self.p_sys / self.delta
        
        # c) Convection due to pressure difference (Darcy's Law)
        n_dot_H2_conv_i = self.A_cell * self.N_cell * (self.K_eff / self.mu_lye) * self.S_H2_lye * self.rho_lye * (self.delta_P / self.delta)
        
        # Total H2 Impurity Inflow to Anode
        n_dot_H2_im_i = n_dot_H2_lye_i + n_dot_H2_diff_i + n_dot_H2_conv_i
        
        # 3. H2 Outflow from Anode to Separator Liquid
        # Assumes perfect mixing in anode volume V_an_lye
        n_dot_H2_im_1_i = n_H2_an_i * v_lye_i / (2 * self.V_an_lye)
        
        # Anode H2 Derivative
        n_dot_H2_an_i = n_dot_H2_im_i - n_dot_H2_im_1_i
            
        # --- Separator Liquid Dynamics ---
        # H2 transfer from Liquid to Gas phase (Desorption/Release)
        n_dot_H2_2 = n_H2_sep_liq / self.tau_sep
        
        # Liquid H2 Derivative: Sum of inflows from stacks - Release to gas
        n_dot_H2_sep_liq = np.sum(n_dot_H2_im_1_i) - n_dot_H2_2
        
        # --- Separator Gas Dynamics ---
        # Total O2 Gas Flow Rate (Carrier Gas)
        sum_n_dot_O2 = np.sum(n_dot_O2_prod_i)
        
        # HTO Output Rate (Carried out by O2 gas stream)
        # Assuming HTO concentration in gas phase is uniform
        # n_dot_out = (n_H2_gas / n_total_gas) * n_dot_total_gas
        # n_total_gas approx P * V / (R * T)
        if sum_n_dot_O2 > 1e-9:
            n_dot_H2_out = (self.R * T_sep * n_H2_sep_gas * sum_n_dot_O2) / (self.p_sys * self.V_sep_gas)
        else:
            n_dot_H2_out = 0.0
            
        # Gas H2 Derivative: Input from liquid - Output via gas flow
        n_dot_sep_gas = n_dot_H2_2 - n_dot_H2_out
        
        return n_dot_H2_an_i, n_dot_H2_sep_liq, n_dot_sep_gas

    def reset(self, initial_state=None):
        if initial_state is not None:
            self.state = np.array(initial_state)
        else:
            # x = [T_s_in, T_s1...4, T_sep, T_c_out, n_H2_an1...4, n_H2_sep_liq, n_H2_sep_gas]
            x0 = np.zeros(13)
            
            # T_s_in (HX Lye Out)
            x0[0] = self.T_s_in_init # Initial guess for Lye In
            
            # T_s (Stacks)
            x0[1:5] = self.T_s_init
            
            # T_sep
            x0[5] = self.T_sep_init
            
            # T_c_out (HX CW Out)
            x0[6] = self.T_c_out_init
            
            # Calculate HTO init
            target_HTO_fraction = self.HTO_init / 100.0
            n_gas = target_HTO_fraction * (self.p_sys * self.V_sep_gas) / (self.R * self.T_sep_init)
            
            # Initialize concentrations (simplified)
            x0[7:11] = n_gas / 4
            x0[11] = n_gas
            
            x0[12] = n_gas
            
            self.state = x0
            
        self.current_time = 0.0
        return self.state

    def step(self, action):
        # action = [I1...4, v_lye1...4, v_c]
        
        # Define derivative function for solve_ivp wrapper
        # fun(t, y) -> dy/dt
        fun = lambda t, y: self._get_derivatives(t, y, action)
        
        t_span = (self.current_time, self.current_time + self.dt)
        
        # Use RK45 (Runge-Kutta 4(5)) as the solver
        sol = solve_ivp(fun, t_span, self.state, method='RK45', t_eval=[t_span[1]])
        
        if sol.success:
            self.state = sol.y[:, -1]
        else:
            # Fallback to manual integration or raise error if critical
            print(f"Integration failed at t={self.current_time}: {sol.message}")
            self.state = sol.y[:, -1] # Try to use the last point anyway
            
        self.current_time += self.dt
        return self.state

    def get_state(self):
        return self.state

