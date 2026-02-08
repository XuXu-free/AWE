
import casadi as ca
import numpy as np

class MultiStackNMPCController:
    def __init__(self, dt=60.0, horizon=10):
        self.dt = dt
        self.N = horizon
        
        # --- System Parameters (Matched to MultiStackSimulator) ---
        self.n_stacks = 4
        self.n_cells = 368
        self.I_rated = 7800.0
        
        # Electrochemical Parameters
        self.U_rev = 1.229
        self.r1 = 3.202e-5
        self.r2 = 8.970e-8
        self.r3 = -4.193e-12
        self.P_sys = 1.6e6
        self.s = 7.572e-2
        self.t1 = -1.070e-1
        self.t2 = 14.43
        self.t3 = 38.8
        
        # Thermal Parameters
        self.C_s_i = 3.450e7  # Stack Heat Capacity
        self.C_sep = 5.193e7  # Separator Heat Capacity
        self.C_he = 2.175e7   # HX Lye Side Heat Capacity
        self.C_c = 2.0e7      # HX CW Side Heat Capacity (Estimated)
        
        self.h_A_stack = 1000.0 # Convective/Rad coeff estimate (simplified form of sigma_s + rad)
        self.T_amb = 25.0
        
        self.kA_hx = 960.0 * 240.0 # k_he * A_he = 230400.0
        self.T_cw_in = 15.0 # 288K
        self.c_cw = 4200.0
        self.rho_cw = 1000.0
        self.c_lye = 3200.0
        self.rho_lye = 1280.0
        
        # Constraints
        self.T_min = 20.0
        self.T_max = 90.0
        self.I_min = 0.0
        # self.I_max = 7800.0 * 1.2
        self.I_max = 7800.0
        self.v_lye_min = 0.0
        self.v_lye_max = 0.1
        self.v_c_min = 0.0
        self.v_c_max = 1.0
        
        self.P_stack_max = 6.0e6
        self.U_cell_max = 2.1
        
        # Targets
        self.T_ref = 85.0
        
        # Weights
        # Adjusted for 4 stacks:
        # Power error is sum of 4 stacks.
        self.lambda_prod = 1.0
        self.lambda_track = 1.2 # 1e-6 scaling in cost
        self.lambda_temp = 0.15
        self.lambda_I = 0.0002
        self.lambda_lye = 25000.0
        self.lambda_c = 0.5
        
        self._setup_solver()
        
        # Internal State Memory
        self.last_I = np.zeros(self.n_stacks)
        self.last_v_lye = np.ones(self.n_stacks) * 0.03
        self.last_v_c = 0.0
        self.prev_sol_x = None

    def _setup_solver(self):
        # Decision Variables Structure:
        # At each step k: [I_1..4, v_lye_1..4, v_c] -> 9 variables
        self.n_controls = self.n_stacks * 2 + 1 # 4+4+1 = 9
        self.U = ca.MX.sym('U', self.n_controls * self.N)
        
        # Parameters: 
        # [T_s_in, T_s_vec, T_sep, T_c_out, n_H2_an_vec, n_liq, n_gas, T_ref, P_ref(N), I_prev, v_lye_prev, v_c_prev]
        # 13 + 1 + N + 9 parameters
        self.n_params = 13 + 1 + self.N + self.n_stacks * 2 + 1
        self.P = ca.MX.sym('P', self.n_params)
        
        # Unpack Initial State
        p_idx = 0
        T_s_in_K = self.P[p_idx]; p_idx += 1
        T_s_K = self.P[p_idx : p_idx+self.n_stacks]; p_idx += self.n_stacks
        T_sep_K = self.P[p_idx]; p_idx += 1
        T_c_out_K = self.P[p_idx]; p_idx += 1
        
        # HTO States
        n_H2_an = self.P[p_idx : p_idx+self.n_stacks]; p_idx += self.n_stacks
        n_liq = self.P[p_idx]; p_idx += 1
        n_gas = self.P[p_idx]; p_idx += 1
        
        T_ref_val = self.P[p_idx]; p_idx += 1
        P_ref = self.P[p_idx : p_idx+self.N]; p_idx += self.N
        
        I_prev = self.P[p_idx : p_idx+self.n_stacks]; p_idx += self.n_stacks
        v_lye_prev = self.P[p_idx : p_idx+self.n_stacks]; p_idx += self.n_stacks
        v_c_prev = self.P[p_idx]; p_idx += 1
        
        # Convert to Celsius for Internal Model
        T_s_in_k = T_s_in_K - 273.15
        T_s_k = T_s_K - 273.15
        T_sep_k = T_sep_K - 273.15
        T_c_out_k = T_c_out_K - 273.15
        
        obj = 0
        g = []
        lbg = []
        ubg = []
        
        # Loop over Horizon
        for k in range(self.N):
            # Extract controls for step k
            uk = self.U[k*self.n_controls : (k+1)*self.n_controls]
            I_k = uk[0 : self.n_stacks]
            v_lye_k = uk[self.n_stacks : 2*self.n_stacks]
            v_c_k = uk[2*self.n_stacks]
            
            # --- System Dynamics (Simplified Thermal Model for Control) ---
            # Sub-stepping for stability
            n_sub = 5
            dt_sub = self.dt / n_sub
            
            T_s_in_sub = T_s_in_k
            T_s_sub = T_s_k
            T_sep_sub = T_sep_k
            T_c_out_sub = T_c_out_k
            
            # Pre-calculate Electro-chemical (assume constant over step k)
            # Voltage & Power
            T_C_vec = T_s_k # Celsius
            T_K_vec = T_C_vec + 273.15
            
            R_ohm = self.r1 + self.r2 * T_K_vec + self.r3 * self.P_sys
            term_act = self.t1 + self.t2 / T_C_vec + self.t3 / (T_C_vec**2 + 1.0) # +1 to avoid div0
            U_act = self.s * ca.log(ca.fmax(term_act * I_k + 1.0, 1e-6))
            U_cell = self.U_rev + R_ohm * I_k + U_act
            V_cell = ca.fmax(U_cell, self.U_rev)
            
            Power_k_vec = V_cell * I_k * self.n_cells
            Total_Power_k = ca.sum1(Power_k_vec)
            
            # Efficiency
            f1 = 50.0 + 2.5 * T_C_vec
            f2 = 0.92 - 6.25e-6 * T_C_vec
            eta = f2 * (I_k**2) / (I_k**2 + f1 + 1e-6)
            
            # Heat Gen
            Q_gen = self.n_cells * I_k * (V_cell - eta * 1.481)
            
            for _ in range(n_sub):
                # 1. Stack Thermal
                # Q_flow = c * rho * v * (T_out - T_in)
                Q_flow_stacks = self.c_lye * self.rho_lye * v_lye_k * (T_s_sub - T_s_in_sub)
                Q_diss_stacks = self.h_A_stack * (T_s_sub - self.T_amb)
                
                dT_s_dt = (Q_gen - Q_diss_stacks - Q_flow_stacks) / self.C_s_i
                T_s_sub = T_s_sub + dT_s_dt * dt_sub
                
                # 2. Separator Thermal
                v_tot = ca.sum1(v_lye_k)
                # Weighted average T_sep_in
                T_sep_in_val = ca.sum1(v_lye_k * T_s_sub) / (v_tot + 1e-6)
                
                Q_sep_diss = 500.0 * (T_sep_sub - self.T_amb) # Simplified loss
                dT_sep_dt = (self.c_lye * self.rho_lye * v_tot * (T_sep_in_val - T_sep_sub) - Q_sep_diss) / self.C_sep
                T_sep_sub = T_sep_sub + dT_sep_dt * dt_sub
                
                # 3. Heat Exchanger
                # LMTD approx: (T_s_in - T_c_out) vs (T_sep - T_c_in)
                # Simplified: Q_hx = kA * (T_hot_avg - T_cold_avg) or standard LMTD
                # Using simpler difference for stability in optimization:
                # Q_hx = kA * (T_lye_in - T_cw_out) # Conservative?
                # Better: Q_hx = kA * (T_sep - T_c_out) # Driving force
                # Let's use linear approx of LMTD
                dT_hot = T_sep_sub - T_c_out_sub
                Q_hx = self.kA_hx * dT_hot 
                # This is a simplification. Real LMTD is hard for NLP.
                
                dT_s_in_dt = (self.c_lye * self.rho_lye * v_tot * (T_sep_sub - T_s_in_sub) - Q_hx) / self.C_he
                T_s_in_sub = T_s_in_sub + dT_s_in_dt * dt_sub
                
                dT_c_out_dt = (self.c_cw * self.rho_cw * v_c_k * (self.T_cw_in - T_c_out_sub) + Q_hx) / self.C_c
                T_c_out_sub = T_c_out_sub + dT_c_out_dt * dt_sub
                
            # Update States
            T_s_k = T_s_sub
            T_s_in_k = T_s_in_sub
            T_sep_k = T_sep_sub
            T_c_out_k = T_c_out_sub
            
            # --- Objective ---
            # 1. Power Tracking
            obj += self.lambda_track * ((Total_Power_k - P_ref[k])/1e6)**2
            
            # 2. Temperature Regulation (All stacks)
            obj += self.lambda_temp * ca.sum1((T_s_k - T_ref_val)**2)
            
            # 3. Production (Maximize)
            # obj -= self.lambda_prod * ca.sum1(I_k) * 1e-4
            
            # 4. Smoothness & Min Effort
            # I: Penalize rate of change (adjacent steps)
            if k == 0:
                dI = I_k - I_prev
            else:
                uk_prev = self.U[(k-1)*self.n_controls : k*self.n_controls]
                dI = I_k - uk_prev[0:self.n_stacks]
            
            # v_lye, v_c: Penalize deviation from current state (initial value of horizon)
            dv_lye = v_lye_k - v_lye_prev
            dv_c = v_c_k - v_c_prev
            
            obj += self.lambda_I * ca.sum1(dI**2)
            obj += self.lambda_lye * ca.sum1(dv_lye**2)
            obj += self.lambda_c * dv_c**2
            
            # --- Constraints ---
            # Stack Temps
            g.append(T_s_k)
            lbg.extend([self.T_min]*self.n_stacks)
            ubg.extend([self.T_max]*self.n_stacks)
            
            # Max Stack Power (0 <= P_i <= 6MW)
            g.append(Power_k_vec)
            lbg.extend([0.0]*self.n_stacks)
            ubg.extend([self.P_stack_max]*self.n_stacks)
            
            # Max Cell Voltage (U_i <= 2.1V)
            g.append(V_cell)
            lbg.extend([-ca.inf]*self.n_stacks)
            ubg.extend([self.U_cell_max]*self.n_stacks)
            
        # Input Bounds
        lbx = []
        ubx = []
        for k in range(self.N):
            # I
            lbx.extend([self.I_min]*self.n_stacks)
            ubx.extend([self.I_max]*self.n_stacks)
            # v_lye
            lbx.extend([self.v_lye_min]*self.n_stacks)
            ubx.extend([self.v_lye_max]*self.n_stacks)
            # v_c
            lbx.append(self.v_c_min)
            ubx.append(self.v_c_max)
            
        self.lbx = lbx
        self.ubx = ubx
        self.lbg = lbg
        self.ubg = ubg
        
        nlp = {'x': self.U, 'f': obj, 'g': ca.vertcat(*g), 'p': self.P}
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-4}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    def get_action(self, state_vec, P_ref_vec):
        # P_ref_vec should be length N. If scalar, repeat.
        if np.isscalar(P_ref_vec):
            P_ref_vec = [P_ref_vec] * self.N
        elif len(P_ref_vec) != self.N:
            # Handle mismatch if any (e.g. pad with last)
            pass 
            
        P_ref_0 = P_ref_vec[0]

        # Initial Guess (Warm Start if available)
        x0 = []
        if getattr(self, 'prev_sol_x', None) is not None:
            # Shift previous solution: u[k] = u[k+1], u[N-1] = u[N-1]
            u_prev = self.prev_sol_x.reshape(self.N, self.n_controls)
            u_guess = np.vstack([u_prev[1:], u_prev[-1:]])
            x0 = u_guess.flatten().tolist()
        else:
            # Cold Start: Distribute P_ref equally
            I_est = (P_ref_0 / self.n_stacks) / (2.0 * self.n_cells)
            I_est = max(self.I_min, min(I_est, self.I_max))
            
            for k in range(self.N):
                x0.extend([I_est]*self.n_stacks)
                x0.extend(self.last_v_lye.tolist())
                x0.append(self.last_v_c)
            
        # Construct Parameters
        p = []
        p.extend(state_vec.tolist()) # Full state vector (13)
        p.append(self.T_ref) # T_ref (Celsius)
        p.extend(P_ref_vec)  # P_ref (N)
        p.extend(self.last_I.tolist())
        p.extend(self.last_v_lye.tolist())
        p.append(self.last_v_c)
        
        try:
            sol = self.solver(x0=x0, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=p)
            u_opt = sol['x'].full().flatten()
            self.prev_sol_x = u_opt # Save for warm start
            
            # Extract first step
            u0 = u_opt[0 : self.n_controls]
            I_cmd = u0[0 : self.n_stacks]
            v_lye_cmd = u0[self.n_stacks : 2*self.n_stacks]
            v_c_cmd = u0[2*self.n_stacks]
            
            self.last_I = I_cmd
            self.last_v_lye = v_lye_cmd
            self.last_v_c = v_c_cmd
            
            return I_cmd, v_lye_cmd, v_c_cmd
            
        except Exception as e:
            print(f"Multi-Stack NMPC Failed: {e}")
            return self.last_I, self.last_v_lye, self.last_v_c
