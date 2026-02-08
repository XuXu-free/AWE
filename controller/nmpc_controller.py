import casadi as ca
import numpy as np

class NMPCController:
    def __init__(self, dt=60.0, horizon=10):
        self.dt = dt
        self.N = horizon
        
        # System Parameters (Must match AWESimulator)
        self.n_cells = 368
        self.U_rev = 1.229
        self.r1 = 3.202e-5
        self.r2 = 8.970e-8
        self.r3 = -4.193e-12
        self.P_sys = 1.6e6
        self.s = 7.572e-2
        self.t1 = -1.070e-1
        self.t2 = 14.43
        self.t3 = 38.8
        self.C_th = 3.450e7
        self.h_A = 1000.0
        self.T_amb = 25.0
        
        # Heat Exchanger & Cooling Water
        self.kA_hx = 230400.0
        self.T_cw_in = 15.0
        self.c_cw = 4200.0
        self.rho_cw = 1000.0
        self.C_cw = 2.0e7
        
        # Constraints
        self.T_min = 25.0
        self.T_max = 90.0
        self.I_min = 0.0
        self.I_max = 7800.0 * 1.2 # Match Simulator 1.2x
        self.v_min = 0.0
        self.v_max = 0.1 # m3/s (Lye)
        self.vc_min = 0.0
        self.vc_max = 1.0 # m3/s (Cooling Water) - assume larger capacity
        
        # Targets
        self.T_ref = 85.0
        
        # Weights (Paper values arXiv:2501.14576)
        # Note: Weights are scaled to maintain relative priority for a single-stack model 
        # compared to the 4-in-1 system in the paper.
        # Ratio Analysis:
        # Paper Term / Code Term (per stack)
        # Prod:  4 * lambda / 1 * w  => w = lambda (if units match)
        # Power: 16 * lambda / 1 * w => w = 4 * lambda (because Power error sums 4 stacks, squared is 16x)
        # Temp:  4 * lambda / 1 * w  => w = lambda
        # I:     4 * lambda / 1 * w  => w = lambda
        # Lye:   4 * lambda / 1 * w  => w = lambda
        # CW:    16 * lambda / 1 * w => w = 4 * lambda (CW flow is shared/total? If per-stack variable, total=4v, sq=16v^2)
        #        Actually, if v_c is total flow, and we model per-stack v_c, then Total=4*v_stack.
        #        Term is lambda_c * (4*v_stack - 4*v0)^2 = 16 * lambda_c * (v_stack - v0)^2
        #        So w_c should be 16 * lambda_c? 
        #        Re-eval: Paper Ratio (CW/Prod) = (lambda_c * (4v)^2) / (lambda_prod * 4n) = 4 (lambda_c/lambda_prod) v^2/n
        #        Code Ratio = (w_c * v^2) / (w_prod * n)
        #        Equating: w_c/w_prod = 4 * lambda_c/lambda_prod
        #        So w_c = 4 * lambda_c = 2.0.
        
        self.lambda_prod = 1.0    # \lambda^{prod}
        self.lambda_track = 1.2 # \lambda^{track}
        self.lambda_temp = 0.15   # \lambda^{temp}
        self.lambda_I = 0.0002    # \lambda^{I}
        self.lambda_lye = 25000.0 # \lambda^{lye}
        self.lambda_c = 0.5 # \lambda^{c} 
        
        # Nominal References / Initial Conditions
        # self.v_lye_0 and self.v_c_0 are now dynamically set to the previous control action
        # to minimize "repeated adjustments" (fatigue), as per paper text.
        
        self._setup_solver()
        
        self.last_I = 0.0
        self.last_v = 0.0335 # Start with valid flow
        self.last_vc = 0.0

    def _setup_solver(self):
        # Decision Variables
        # U: [I_0, v_0, vc_0, I_1, v_1, vc_1, ...]
        self.n_controls = 3
        self.U = ca.MX.sym('U', self.n_controls * self.N)
        
        # Parameters: [T_stack_init, T_cw_out_init, P_ref, I_prev, v_prev, vc_prev]
        self.P = ca.MX.sym('P', 6)
        
        T_stack_k = self.P[0]
        T_cw_out_k = self.P[1]
        P_ref = self.P[2] # Watts
        I_prev = self.P[3]
        v_prev = self.P[4]
        vc_prev = self.P[5]
        
        obj = 0
        g = [] 
        lbg = []
        ubg = []
        
        for k in range(self.N):
            I_k = self.U[self.n_controls*k]
            v_k = self.U[self.n_controls*k+1]
            vc_k = self.U[self.n_controls*k+2]
            
            # --- System Model (Symbolic) ---
            # Voltage
            T_c = ca.fmax(T_stack_k, 0.1)
            T_kelvin = T_stack_k + 273.15
            
            R_ohm = self.r1 + self.r2 * T_kelvin + self.r3 * self.P_sys
            act_coeff = self.t1 + self.t2 / T_c + self.t3 / (T_c**2)
            U_act = self.s * ca.log(ca.fmax(act_coeff * I_k + 1.0, 1e-6))
            U_cell = self.U_rev + R_ohm * I_k + U_act
            V_cell = ca.fmax(U_cell, self.U_rev)
            
            # Power
            Power_k = V_cell * I_k * self.n_cells
            
            # Faraday Efficiency (Approximate constant over step or use initial T)
            f1 = 50.0 + 2.5 * T_stack_k
            f2 = 0.92 - 6.25e-6 * T_stack_k
            eta = f2 * (I_k**2) / (I_k**2 + f1 + 1e-6)
            
            # Sub-stepping for Thermal Dynamics (Stability)
            # dt is large (e.g. 60s), thermal dynamics are fast.
            # Use 10 sub-steps of dt/10.
            n_sub = 10
            dt_sub = self.dt / n_sub
            
            T_s_sub = T_stack_k
            T_c_sub = T_cw_out_k
            
            for _ in range(n_sub):
                # Q_hx = kA * (T_stack - T_cw_out)
                Q_hx = self.kA_hx * (T_s_sub - T_c_sub)
                
                # Cooling Water Temp Dynamics
                # dQ_cw_flow = m_dot_cw * c_cw * (T_cw_in - T_cw_out)
                m_dot_cw = vc_k * self.rho_cw
                dQ_cw_flow = m_dot_cw * self.c_cw * (self.T_cw_in - T_c_sub)
                dT_cw_out_dt = (dQ_cw_flow + Q_hx) / self.C_cw
                
                # Stack Temp Dynamics
                # Q_lye_cooling = Q_hx (assuming adequate flow)
                Q_lye_cooling = Q_hx
                
                # Recalculate Q_gen (Simplified: Assume V and I constant over sub-step)
                # Note: I is decision variable (constant over step). V depends on T, but changing V inside sub-loop increases complexity.
                # Use V_cell from start of step.
                Q_gen = self.n_cells * I_k * (V_cell - eta * 1.481)
                
                Q_nat = self.h_A * (T_s_sub - self.T_amb)
                Q_loss = Q_nat + Q_lye_cooling
                
                dT_stack_dt = (Q_gen - Q_loss) / self.C_th
                
                # Update Sub-step
                T_s_sub = T_s_sub + dT_stack_dt * dt_sub
                T_c_sub = T_c_sub + dT_cw_out_dt * dt_sub
            
            T_stack_next = T_s_sub
            T_cw_out_next = T_c_sub
            
            # --- Objective (Paper Formulation) ---
            # 0. Maximize H2 Production
            # n_dot = N * I * eta / (2 F)
            # Use kmol/s to balance weights (assuming paper uses scaled units or MW/kmol consistency)
            F_const = 96485.33
            n_dot_mol_s = self.n_cells * I_k * eta / (2.0 * F_const)
            n_dot_kmol_s = n_dot_mol_s / 1000.0
            obj -= self.lambda_prod * n_dot_kmol_s * self.dt

            # 1. Track Power (in MW)
            # P_err_MW = (Power_k - P_ref) * 1e-6
            P_err_MW = (Power_k - P_ref)
            obj += self.lambda_track * (P_err_MW)**2
            
            # 2. Maintain Temperature
            obj += self.lambda_temp * (T_stack_next - self.T_ref)**2
            
            # 3. Smoothness (Current only)
            if k == 0:
                obj += self.lambda_I * (I_k - I_prev)**2
            else:
                I_prev_k = self.U[self.n_controls*(k-1)]
                obj += self.lambda_I * (I_k - I_prev_k)**2
                
            # 4. Flow Deviation from Initial/Previous (Minimize Adjustments)
            # Paper: "repeated adjustments... minimized" -> v^0 is the value at current time k (initial state)
            # We use v_prev (last applied action) as the reference for the entire horizon to penalize deviation from current setpoint.
            obj += self.lambda_lye * (v_k - v_prev)**2
            obj += self.lambda_c * (vc_k - vc_prev)**2
            
            # Update State
            T_stack_k = T_stack_next
            T_cw_out_k = T_cw_out_next
            
            # Constraints on State
            g.append(T_stack_next)
            lbg.append(self.T_min)
            ubg.append(self.T_max)
        
        # Constraints on Inputs (Bounds)
        lbx = []
        ubx = []
        for k in range(self.N):
            lbx.append(self.I_min) # I
            lbx.append(self.v_min) # v_lye
            lbx.append(self.vc_min) # v_cw
            
            ubx.append(self.I_max) # I
            ubx.append(self.v_max) # v_lye
            ubx.append(self.vc_max) # v_cw
            
        self.lbx = lbx
        self.ubx = ubx
        
        # Solver
        nlp = {'x': self.U, 'f': obj, 'g': ca.vertcat(*g), 'p': self.P}
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-4}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        self.lbg = lbg
        self.ubg = ubg

    def get_action(self, T_current, T_cw_out, P_ref):
        # Initial guess strategy
        # Estimate I required for P_ref: P = V * I * N => I ~ P / (2.0 * N)
        I_est = P_ref / (2.0 * self.n_cells)
        I_est = max(self.I_min, min(I_est, self.I_max))
        
        x0 = np.zeros(self.n_controls * self.N)
        for k in range(self.N):
            x0[self.n_controls*k] = I_est # Use estimate instead of previous
            x0[self.n_controls*k+1] = self.last_v
            x0[self.n_controls*k+2] = self.last_vc
            
        # Parameters
        p = [T_current, T_cw_out, P_ref, self.last_I, self.last_v, self.last_vc]
        
        # Solve
        try:
            sol = self.solver(x0=x0, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=p)
            u_opt = sol['x'].full().flatten()
            
            I_cmd = u_opt[0]
            v_cmd = u_opt[1]
            vc_cmd = u_opt[2]
            
            # Clip to bounds
            I_cmd = max(self.I_min, min(I_cmd, self.I_max))
            v_cmd = max(self.v_min, min(v_cmd, self.v_max))
            vc_cmd = max(self.vc_min, min(vc_cmd, self.vc_max))
            
            self.last_I = I_cmd
            self.last_v = v_cmd
            self.last_vc = vc_cmd
            
            return [I_cmd, v_cmd, vc_cmd]
        except Exception as e:
            print(f"NMPC Solver Failed: {e}")
            return [self.last_I, self.last_v, self.last_vc] # Fallback

if __name__ == "__main__":
    # Test
    ctl = NMPCController()
    action = ctl.get_action(T_current=25.0, T_cw_out=20.0, P_ref=5.0e6)
    print(f"Action: {action}")
