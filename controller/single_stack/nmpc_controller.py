import casadi as ca
import numpy as np
import os
import subprocess

class SingleStackNMPCController:
    def __init__(self, dt=60.0, horizon=5, sim_dt=0.2):
        self.dt = dt
        self.sim_dt = sim_dt
        self.horizon = horizon
        
        # --- System Parameters (Matched to SingleStackSimulator) ---
        self.n_stacks = 1
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
        self.F = 96485.0  # Faraday constant (C/mol)

        # Thermal Parameters
        self.C_s_i = 3.450e7
        self.C_sep = 1e7
        self.C_he = 5e6
        # NOTE unknown
        self.C_c = 5e6
        
        self.h_A_stack = 1000.0 # Convective/Rad coeff estimate
        self.T_amb = 298.15
        
        # NOTE unsure
        self.A_he = 240
        # NOTE unsure
        self.k_he = 960
        self.kA_hx = 960.0 * 240.0 # k_he * A_he
        self.T_cw_in = 288.15 # 288K
        self.c_cw = 4200.0
        self.rho_cw = 1000.0
        self.c_lye = 3200.0
        self.rho_lye = 1280.0
        
        self.R = 8.314
        self.V_sep = 10.288
        self.V_sep_gas = 0.6 * self.V_sep
        
        # Constraints
        self.T_min = 293.15
        self.T_max = 363.15
        self.I_min = 0.0
        self.I_max = 7800.0 * 1.2
        self.v_lye_min = 0.01
        self.v_lye_max = 0.1
        self.v_c_min = 0.0
        self.v_c_max = 1.0
        
        self.P_stack_max = 6.0e6 # Updated to 6.5 MW
        self.P_stack_min = 0.0
        self.U_cell_min = 0.0
        self.U_cell_max = 2.2    # Updated to 2.2 V
        
        self.HTO_pct_max = 2.0
        self.HTO_pct_min = 0.0
        
        # Targets
        self.T_ref = 358.15 
        
        # Weights
        self.lambda_prod = 1.0
        self.lambda_track = 1.2
        self.lambda_temp = 0.15
        self.lambda_I = 0.0002
        self.lambda_lye = 25000.0
        self.lambda_c = 25.0
        
        self._setup_solver()
        
        self.prev_sol_x = None

    def _dynamics_step(self, T_s_in_k, T_s_k, T_sep_k, T_c_out_k, I_k, v_lye_k, v_c_k, n_gas_k):
        n_sub = int(self.dt / self.sim_dt) if self.sim_dt > 0 else 1
        n_sub = max(1, n_sub)
        dt_sub = self.dt / n_sub

        T_s_in_sub = T_s_in_k
        T_s_sub = T_s_k
        T_sep_sub = T_sep_k
        T_c_out_sub = T_c_out_k

        T_K = T_s_k
        T_C = T_K - 273.15

        R_ohm = self.r1 + self.r2 * T_K + self.r3 * self.P_sys
        term_act = self.t1 + self.t2 / T_C + self.t3 / (T_C**2 + 1.0)
        U_act = self.s * ca.log(ca.fmax(term_act * I_k + 1.0, 1e-6))
        U_cell = self.U_rev + R_ohm * I_k + U_act
        V_cell = ca.fmax(U_cell, self.U_rev)

        f1 = 50.0 + 2.5 * T_C
        f2 = 0.92 - 6.25e-6 * T_C
        eta = f2 * (I_k**2) / (I_k**2 + f1 + 1e-6)

        Q_gen = self.n_cells * I_k * (V_cell - eta * 1.481)
        Power_k = V_cell * I_k * self.n_cells

        for _ in range(n_sub):
            Q_flow_stack = self.c_lye * self.rho_lye * v_lye_k * (T_s_sub - T_s_in_sub)
            Q_diss_stack = self.h_A_stack * (T_s_sub - self.T_amb)

            dT_s_dt = (Q_gen - Q_diss_stack - Q_flow_stack) / self.C_s_i
            T_s_sub = T_s_sub + dT_s_dt * dt_sub

            Q_sep_diss = 500.0 * (T_sep_sub - self.T_amb)
            dT_sep_dt = (self.c_lye * self.rho_lye * v_lye_k * (T_s_sub - T_sep_sub) - Q_sep_diss) / self.C_sep
            T_sep_sub = T_sep_sub + dT_sep_dt * dt_sub

            dT_hot = T_sep_sub - T_c_out_sub
            Q_hx = self.kA_hx * dT_hot

            dT_s_in_dt = (self.c_lye * self.rho_lye * v_lye_k * (T_sep_sub - T_s_in_sub) - Q_hx) / self.C_he
            T_s_in_sub = T_s_in_sub + dT_s_in_dt * dt_sub

            dT_c_out_dt = (self.c_cw * self.rho_cw * v_c_k * (self.T_cw_in - T_c_out_sub) + Q_hx) / self.C_c
            T_c_out_sub = T_c_out_sub + dT_c_out_dt * dt_sub

        # Calculate HTO %
        hto_pct = (n_gas_k * self.R * T_sep_sub) / (self.P_sys * self.V_sep_gas) * 100

        return T_s_in_sub, T_s_sub, T_sep_sub, T_c_out_sub, Power_k, V_cell, hto_pct, eta

    def _setup_solver(self):
        # Decision Variables Structure:
        # At each step k: [I, v_lye, v_c] -> 3 variables
        self.n_controls = 3
        self.U = ca.MX.sym('U', self.n_controls * self.horizon)
        
        # Parameters: 
        # [T_s_in, T_s, T_sep, T_c_out, n_H2_an, n_liq, n_gas, T_ref, P_ref(N), I_prev, v_lye_prev, v_c_prev]
        # 7 + 1 + N + 3 = 11 + N parameters
        self.n_params = 7 + 1 + self.horizon + 3
        self.P = ca.MX.sym('P', self.n_params)
        
        # Unpack Initial State
        p_idx = 0
        T_s_in_K = self.P[p_idx]; p_idx += 1
        T_s_K = self.P[p_idx]; p_idx += 1
        T_sep_K = self.P[p_idx]; p_idx += 1
        T_c_out_K = self.P[p_idx]; p_idx += 1
        
        # HTO States
        n_H2_an = self.P[p_idx]; p_idx += 1
        n_liq = self.P[p_idx]; p_idx += 1
        n_gas = self.P[p_idx]; p_idx += 1
        
        T_ref_val = self.P[p_idx]; p_idx += 1
        P_ref = self.P[p_idx : p_idx+self.horizon]; p_idx += self.horizon
        
        I_0 = self.P[p_idx]; p_idx += 1
        v_lye_0 = self.P[p_idx]; p_idx += 1
        v_c_0 = self.P[p_idx]; p_idx += 1
        
        # Convert to Celsius for Internal Model
        T_s_in_k = T_s_in_K
        T_s_k = T_s_K
        T_sep_k = T_sep_K
        T_c_out_k = T_c_out_K
        
        obj = 0
        g = []
        lbg = []
        ubg = []
        
        # Loop over Horizon
        for k in range(self.horizon):
            # Extract controls for step k
            uk = self.U[k*self.n_controls : (k+1)*self.n_controls]
            I_k = uk[0]
            v_lye_k = uk[1]
            v_c_k = uk[2]
            
            T_s_in_k, T_s_k, T_sep_k, T_c_out_k, Power_k, V_cell, hto_pct, eta = self._dynamics_step(
                T_s_in_k, T_s_k, T_sep_k, T_c_out_k, I_k, v_lye_k, v_c_k, n_gas
            )
            
            # --- Objective ---
            # 1. Power Tracking
            obj += self.lambda_track * ((Power_k - P_ref[k])/1e6)**2
            
            # 2. Temperature Regulation
            obj += self.lambda_temp * (T_s_k - T_ref_val)**2


            # 4. Smoothness
            if k == 0:
                dI = I_k - I_0
            else:
                uk_prev = self.U[(k-1)*self.n_controls : k*self.n_controls]
                dI = I_k - uk_prev[0]
            
            dv_lye = v_lye_k - v_lye_0
            dv_c = v_c_k - v_c_0
            
            obj += self.lambda_I * dI**2
            obj += self.lambda_lye * dv_lye**2
            obj += self.lambda_c * dv_c**2
            
            # --- Constraints ---
            # Stack Temp
            g.append(T_s_k)
            lbg.append(self.T_min)
            ubg.append(self.T_max)
            
            # Max Stack Power
            g.append(Power_k)
            lbg.append(self.P_stack_min)
            ubg.append(self.P_stack_max)
            
            # Max Cell Voltage
            g.append(V_cell)
            lbg.append(self.U_cell_min)
            ubg.append(self.U_cell_max)
            
            # HTO Production Rate
            g.append(hto_pct)
            lbg.append(self.HTO_pct_min)
            ubg.append(self.HTO_pct_max)
            
        # Input Bounds
        lbx = []
        ubx = []
        for k in range(self.horizon):
            lbx.extend([self.I_min, self.v_lye_min, self.v_c_min])
            ubx.extend([self.I_max, self.v_lye_max, self.v_c_max])
            
        self.lbx = lbx
        self.ubx = ubx
        self.lbg = lbg
        self.ubg = ubg
        
        nlp = {'x': self.U, 'f': obj, 'g': ca.vertcat(*g), 'p': self.P}
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-4}
        
        # Define paths for solver artifacts
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_dir = os.path.join(project_root, 'output', 'single_stack')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        c_file = os.path.join(output_dir, 'nmpc_solver.c')
        dll_file = os.path.join(output_dir, 'nmpc_solver.dll')
        
        if os.path.exists(dll_file):
            # print(f"Loading compiled solver from {dll_file}")
            self.solver = ca.nlpsol('nmpc_solver', 'ipopt', dll_file, opts)
        else:
            print("Compiling NMPC solver...")
            # Create JIT solver to generate code
            solver = ca.nlpsol('nmpc_solver', 'ipopt', nlp, opts)
            
            try:
                # Change to output directory to avoid path issues in generate_dependencies name check
                cwd = os.getcwd()
                os.chdir(output_dir)
                try:
                    solver.generate_dependencies('nmpc_solver.c')
                    print(f"C code generated to {c_file}")
                    
                    if os.name == 'posix':
                        cmd = "gcc -fPIC -shared -O0 nmpc_solver.c -o nmpc_solver.dll"
                    else:
                        cmd = "gcc -shared -O0 nmpc_solver.c -o nmpc_solver.dll"
                        
                    subprocess.check_call(cmd.split())
                    print(f"Solver compiled to {dll_file}")
                finally:
                    os.chdir(cwd)
                
                self.solver = ca.nlpsol('nmpc_solver', 'ipopt', dll_file, opts)
            except Exception as e:
                print(f"Compilation failed: {e}. Using JIT solver.")
                self.solver = solver

        pred_states = []

        p_idx = 0
        T_s_in_K = self.P[p_idx]; p_idx += 1
        T_s_K = self.P[p_idx]; p_idx += 1
        T_sep_K = self.P[p_idx]; p_idx += 1
        T_c_out_K = self.P[p_idx]; p_idx += 1
        
        # Skip HTO states for prediction initialization loop (we only need thermal states for T_s_in_k etc.)
        # But we DO need n_gas for dynamics_step
        n_H2_an_dummy = self.P[p_idx]; p_idx += 1
        n_liq_dummy = self.P[p_idx]; p_idx += 1
        n_gas_val = self.P[p_idx]; p_idx += 1

        T_s_in_k = T_s_in_K
        T_s_k = T_s_K
        T_sep_k = T_sep_K
        T_c_out_k = T_c_out_K

        for k in range(self.horizon):
            uk = self.U[k*self.n_controls : (k+1)*self.n_controls]
            I_k = uk[0]
            v_lye_k = uk[1]
            v_c_k = uk[2]

            T_s_in_k, T_s_k, T_sep_k, T_c_out_k, _, _, _, _ = self._dynamics_step(
                T_s_in_k, T_s_k, T_sep_k, T_c_out_k, I_k, v_lye_k, v_c_k, n_gas_val
            )
            pred_states.append(ca.vertcat(T_s_in_k, T_s_k, T_sep_k, T_c_out_k))

        self.state_func = ca.Function('state_func', [self.U, self.P], [ca.vertcat(*pred_states)])

    def solve_nmpc(self, state_vec, P_ref_vec, T_ref, last_action):
        self.T_ref = T_ref

        state_vec = np.asarray(state_vec).flatten()

        last_I = last_action[0]
        last_v_lye = last_action[1]
        last_v_c = last_action[2]

        if np.isscalar(P_ref_vec):
            P_ref_vec = [P_ref_vec] * self.horizon
        elif len(P_ref_vec) != self.horizon:
            pass

        P_ref_0 = P_ref_vec[0]

        x0 = []
        if getattr(self, 'prev_sol_x', None) is not None:
            u_prev = self.prev_sol_x.reshape(self.horizon, self.n_controls)
            u_guess = np.vstack([u_prev[1:], u_prev[-1:]])
            x0 = u_guess.flatten().tolist()
        else:
            I_est = P_ref_0 / (2.0 * self.n_cells)
            I_est = max(self.I_min, min(I_est, self.I_max))
            for _ in range(self.horizon):
                x0.extend([I_est, last_v_lye, last_v_c])

        p = []
        p.extend(state_vec.tolist())
        p.append(self.T_ref)
        p.extend(P_ref_vec)
        p.append(last_I)
        p.append(last_v_lye)
        p.append(last_v_c)

        try:
            sol = self.solver(x0=x0, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=p)
            u_opt = sol['x'].full().flatten()
            self.prev_sol_x = u_opt
            return u_opt
        except Exception as e:
            print(f"Single Stack NMPC Failed: {e}")
            return None

    def get_action(self, state_vec, P_ref_vec, T_ref, last_action):
        u_opt = self.solve_nmpc(state_vec, P_ref_vec, T_ref, last_action)

        if u_opt is not None:
            u0 = u_opt[0 : self.n_controls]
            I_cmd = u0[0]
            v_lye_cmd = u0[1]
            v_c_cmd = u0[2]

            return I_cmd, v_lye_cmd, v_c_cmd
        
        last_I = last_action[0]
        last_v_lye = last_action[1]
        last_v_c = last_action[2]
        return last_I, last_v_lye, last_v_c


    def get_all_actions_states(self, state_vec, P_ref_vec, T_ref, last_action):
        state_vec = np.asarray(state_vec).flatten()
        u_opt = self.solve_nmpc(state_vec, P_ref_vec, T_ref, last_action)

        if u_opt is not None:
            u0 = u_opt[0 : self.n_controls]
            I_cmd = u0[0]
            v_lye_cmd = u0[1]
            v_c_cmd = u0[2]

            actions = u_opt.reshape(self.horizon, self.n_controls)

            if np.isscalar(P_ref_vec):
                P_ref_vec = [P_ref_vec] * self.horizon
            elif len(P_ref_vec) != self.horizon:
                pass

            last_I = last_action[0]
            last_v_lye = last_action[1]
            last_v_c = last_action[2]

            p = []
            p.extend(state_vec.tolist())
            p.append(self.T_ref)
            p.extend(P_ref_vec)
            p.append(last_I)
            p.append(last_v_lye)
            p.append(last_v_c)

            # Evaluate state function
            # Output is flat (N * 4) [T_s_in, T_s, T_sep, T_c_out]
            pred_thermal_vec = self.state_func(u_opt, p).full().flatten()
            pred_thermal = pred_thermal_vec.reshape(self.horizon, 4)

            # Construct Full 7-dim State
            # Non-thermal states from initial condition:
            # n_H2_an, n_liq, n_gas -> indices 4, 5, 6 in state_vec
            non_thermal_initial = state_vec[4:7]

            pred_non_thermal = np.tile(non_thermal_initial, (self.horizon, 1))
            
            # Concatenate: [Thermal(4), Non-Thermal(3)] -> (N, 7)
            pred_states = np.hstack([pred_thermal, pred_non_thermal])

            return actions, pred_states

        else:
            return None, None
