
import casadi as ca
import numpy as np
from ..base_controller import BaseController

class MultiStackNMPCController(BaseController):
    def __init__(self, dt=60.0, horizon=10, dt_sub=0.2):
        self.dt = dt
        self.dt_sub = dt_sub
        self.horizon = horizon
        
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
        
        self.h_A_stack = 1000.0 # Convective/Rad coeff estimate (simplified form of sigma_s + rad)
        self.T_amb = 298.15
        
        self.kA_hx = 960.0 * 240.0 # k_he * A_he = 230400.0
        self.T_cw_in = 288.15 # 288K
        self.c_cw = 4200.0
        self.rho_cw = 1000.0
        self.c_lye = 3200.0
        self.rho_lye = 1280.0
        
        # Constraints
        self.T_min = 293.15
        self.T_max = 363.15
        self.I_min = 0.0
        self.I_max = 7800.0 * 1.2
        # self.I_max = 7800.0
        self.v_lye_min = 0.0
        self.v_lye_max = 0.1
        self.v_c_min = 0.0
        self.v_c_max = 1.0
        
        self.P_stack_max = 6.0e6
        self.P_stack_min = 0.0
        self.U_cell_min = 0.0
        self.U_cell_max = 2.2
        
        self.HTO_pct_max = 2.0
        self.HTO_pct_min = 0.0
        
        # Targets
        self.T_ref = 358.15 # Default value, updated in get_action
        
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
        
        self.prev_sol_x = None

    def _dynamics_step(self, T_s_in_k, T_s_k, T_sep_k, T_c_out_k, I_k, v_lye_k, v_c_k, n_gas_k):
        """
        Calculates the state at the next time step (k+1) given the current state (k) and inputs.
        Includes sub-stepping for numerical stability.
        """
        # Sub-stepping
        n_sub = int(self.dt / self.dt_sub)
        
        T_s_in_sub = T_s_in_k
        T_s_sub = T_s_k
        T_sep_sub = T_sep_k
        T_c_out_sub = T_c_out_k
        
        # Pre-calculate Electro-chemical (assume constant over step k)
        T_K_vec = T_s_k
        T_C_vec = T_K_vec - 273.15
        
        R_ohm = self.r1 + self.r2 * T_K_vec + self.r3 * self.P_sys
        term_act = self.t1 + self.t2 / T_C_vec + self.t3 / (T_C_vec**2 + 1.0)
        U_act = self.s * ca.log(ca.fmax(term_act * I_k + 1.0, 1e-6))
        U_cell = self.U_rev + R_ohm * I_k + U_act
        V_cell = ca.fmax(U_cell, self.U_rev)
        
        # Efficiency
        f1 = 50.0 + 2.5 * T_C_vec
        f2 = 0.92 - 6.25e-6 * T_C_vec
        eta = f2 * (I_k**2) / (I_k**2 + f1 + 1e-6)
        
        Q_gen = self.n_cells * I_k * (V_cell - eta * 1.481)
        
        # Power Calculation (Needed for constraints/obj)
        Power_k_vec = V_cell * I_k * self.n_cells
        
        for _ in range(n_sub):
            Q_flow_stacks = self.c_lye * self.rho_lye * v_lye_k * (T_s_sub - T_s_in_sub)
            Q_diss_stacks = self.h_A_stack * (T_s_sub - self.T_amb)
            
            dT_s_dt = (Q_gen - Q_diss_stacks - Q_flow_stacks) / self.C_s_i
            T_s_sub = T_s_sub + dT_s_dt * self.dt_sub
            
            v_tot = ca.sum1(v_lye_k)
            T_sep_in_val = ca.sum1(v_lye_k * T_s_sub) / (v_tot + 1e-6)
            
            Q_sep_diss = 500.0 * (T_sep_sub - self.T_amb)
            dT_sep_dt = (self.c_lye * self.rho_lye * v_tot * (T_sep_in_val - T_sep_sub) - Q_sep_diss) / self.C_sep
            T_sep_sub = T_sep_sub + dT_sep_dt * self.dt_sub
            
            dT_hot = T_sep_sub - T_c_out_sub
            Q_hx = self.kA_hx * dT_hot 
            
            dT_s_in_dt = (self.c_lye * self.rho_lye * v_tot * (T_sep_sub - T_s_in_sub) - Q_hx) / self.C_he
            T_s_in_sub = T_s_in_sub + dT_s_in_dt * self.dt_sub
            
            dT_c_out_dt = (self.c_cw * self.rho_cw * v_c_k * (self.T_cw_in - T_c_out_sub) + Q_hx) / self.C_c
            T_c_out_sub = T_c_out_sub + dT_c_out_dt * self.dt_sub
        
        # Calculate HTO %
        hto_pct = (n_gas_k * self.R * T_sep_sub) / (self.P_sys * self.V_sep_gas) * 100
        
        return T_s_in_sub, T_s_sub, T_sep_sub, T_c_out_sub, Power_k_vec, V_cell, hto_pct

    def _setup_solver(self):
        import os
        # Decision Variables Structure:
        # At each step k: [I_1..4, v_lye_1..4, v_c] -> 9 variables
        self.n_controls = self.n_stacks * 2 + 1 # 4+4+1 = 9
        self.U = ca.MX.sym('U', self.n_controls * self.horizon)
        
        # Parameters: 
        # [T_s_in, T_s_vec, T_sep, T_c_out, n_H2_an_vec, n_liq, n_gas, T_ref, P_ref(N), I_prev, v_lye_prev, v_c_prev]
        # 13 + 1 + N + 9 parameters
        self.n_params = 13 + 1 + self.horizon + self.n_stacks * 2 + 1
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
        P_ref = self.P[p_idx : p_idx+self.horizon]; p_idx += self.horizon
        
        I_0 = self.P[p_idx : p_idx+self.n_stacks]; p_idx += self.n_stacks
        v_lye_0 = self.P[p_idx : p_idx+self.n_stacks]; p_idx += self.n_stacks
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
            I_k = uk[0 : self.n_stacks]
            v_lye_k = uk[self.n_stacks : 2*self.n_stacks]
            v_c_k = uk[2*self.n_stacks]
            
            # --- System Dynamics (Simplified Thermal Model for Control) ---
            T_s_in_k, T_s_k, T_sep_k, T_c_out_k, Power_k_vec, V_cell, hto_pct = self._dynamics_step(
                T_s_in_k, T_s_k, T_sep_k, T_c_out_k, I_k, v_lye_k, v_c_k, n_gas
            )
            
            Total_Power_k = ca.sum1(Power_k_vec)
            
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
                dI = I_k - I_0
            else:
                uk_prev = self.U[(k-1)*self.n_controls : k*self.n_controls]
                dI = I_k - uk_prev[0:self.n_stacks]
            
            # v_lye, v_c: Penalize deviation from current state (initial value of horizon)
            dv_lye = v_lye_k - v_lye_0
            dv_c = v_c_k - v_c_0
            
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
            lbg.extend([self.P_stack_min]*self.n_stacks)
            ubg.extend([self.P_stack_max]*self.n_stacks)
            
            # Max Cell Voltage (U_i <= 2.1V)
            g.append(V_cell)
            lbg.extend([self.U_cell_min]*self.n_stacks)
            ubg.extend([self.U_cell_max]*self.n_stacks)
            
            # HTO Production Rate (0 <= HTO_pct <= 2%)
            g.append(hto_pct)
            lbg.append(self.HTO_pct_min)
            ubg.append(self.HTO_pct_max)
            
        # Input Bounds
        lbx = []
        ubx = []
        for k in range(self.horizon):
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
        
        # State Function (Extract predicted states)
        # States: T_s_in, T_s(4), T_sep, T_c_out
        # Collect states for trajectory output
        states_traj = []
        
        # Re-construct state trajectory using optimal U (Must match the loop above exactly or capture variables)
        # Since we already have the symbolic variables T_s_k etc. from the loop, we can just collect them.
        # However, the loop above overwrites T_s_k in each iteration.
        # We need to modify the loop to store them or recreate the function.
        # Let's recreate the function to be safe and clean, or we can store them in a list during the loop.
        
        # Actually, capturing them during the loop is better.
        # Let's modify the loop above to store symbolic states.
        
        nlp = {'x': self.U, 'f': obj, 'g': ca.vertcat(*g), 'p': self.P}
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-4}
        
        print("Setting up solver...")
        
        # --- C-Code Generation for Speedup ---
        solver_name = 'nmpc_solver'
        c_file = f'{solver_name}.c'
        dll_file = f'{solver_name}.so' if os.name == 'posix' else f'{solver_name}.dll'
        
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        self.solver = solver
        
        # # Check if compiled solver exists and load it
        # if os.path.exists(dll_file):
        #     # print(f"Loading compiled solver from {dll_file}")
        #     self.solver = ca.nlpsol('solver', 'ipopt', dll_file, opts)
        # else:
        #     print("Compiling NMPC solver...")
        #     # Generate C code
        #     solver.generate_dependencies(c_file)
        #     print(f"C code generated to {c_file}")
            
        #     # Compile
        #     # Requires gcc or cl.exe in path
        #     import subprocess
        #     try:
        #         if os.name == 'posix':
        #             cmd = f"gcc -fPIC -shared -O0 {c_file} -o {dll_file}"
        #             subprocess.check_call(cmd.split())
        #         else:
        #             # Windows (assuming MSVC or MinGW)
        #             # Try gcc (MinGW) first
        #             cmd = f"gcc -shared -O0 {c_file} -o {dll_file}"
        #             subprocess.check_call(cmd.split())
                    
        #         print(f"Solver compiled to {dll_file}")
        #         # Load the compiled solver
        #         self.solver = ca.nlpsol('solver', 'ipopt', dll_file, opts)
        #     except Exception as e:
        #         print(f"Compilation failed: {e}. Using JIT solver.")
        #         self.solver = solver
        
        # Create a separate function for state prediction
        # We need to rebuild the dynamics graph for this function
        # Inputs: U, P. Outputs: Trajectory of [T_s_in, T_s_1..4, T_sep, T_c_out] (7 vars)
        
        pred_states = []
        
        # Re-unpack Initial State for Prediction Function
        p_idx = 0
        T_s_in_K = self.P[p_idx]; p_idx += 1
        T_s_K = self.P[p_idx : p_idx+self.n_stacks]; p_idx += self.n_stacks
        T_sep_K = self.P[p_idx]; p_idx += 1
        T_c_out_K = self.P[p_idx]; p_idx += 1
        
        # Convert to Celsius
        T_s_in_k = T_s_in_K
        T_s_k = T_s_K
        T_sep_k = T_sep_K
        T_c_out_k = T_c_out_K
        
        for k in range(self.horizon):
            uk = self.U[k*self.n_controls : (k+1)*self.n_controls]
            I_k = uk[0 : self.n_stacks]
            v_lye_k = uk[self.n_stacks : 2*self.n_stacks]
            v_c_k = uk[2*self.n_stacks]
            
            # Dynamics (Copy of above)
            T_s_in_k, T_s_k, T_sep_k, T_c_out_k, _, _, _ = self._dynamics_step(
                T_s_in_k, T_s_k, T_sep_k, T_c_out_k, I_k, v_lye_k, v_c_k, n_gas
            )
            
            # Store State: [T_s_in(1), T_s(4), T_sep(1), T_c_out(1)] -> 7
            pred_states.append(ca.vertcat(T_s_in_k, T_s_k, T_sep_k, T_c_out_k))
            
        # Create a function to simulate/predict the full state trajectory based on inputs (U) and parameters (P)
        # self.U: Control actions over the horizon
        # self.P: Initial state and reference parameters
        # Returns: Concatenated vector of predicted states for each step in the horizon
        self.state_func = ca.Function('state_func', [self.U, self.P], [ca.vertcat(*pred_states)])

    def solve_nmpc(self, state_vec, P_ref_vec, T_ref, last_action):
        # Update internal T_ref for logging/reference
        self.T_ref = T_ref
        
        last_I = last_action[0]
        last_v_lye = last_action[1]
        last_v_c = last_action[2]

        # P_ref_vec should be length N. If scalar, repeat.
        if np.isscalar(P_ref_vec):
            P_ref_vec = [P_ref_vec] * self.horizon
        elif len(P_ref_vec) != self.horizon:
            # Handle mismatch if any (e.g. pad with last)
            pass 
            
        P_ref_0 = P_ref_vec[0]

        # Initial Guess (Warm Start if available)
        x0 = []
        if getattr(self, 'prev_sol_x', None) is not None:
            # Shift previous solution: u[k] = u[k+1], u[N-1] = u[N-1]
            u_prev = self.prev_sol_x.reshape(self.horizon, self.n_controls)
            u_guess = np.vstack([u_prev[1:], u_prev[-1:]])
            x0 = u_guess.flatten().tolist()
        else:
            # Cold Start: Distribute P_ref equally
            I_est = (P_ref_0 / self.n_stacks) / (2.0 * self.n_cells)
            I_est = max(self.I_min, min(I_est, self.I_max))
            
            for k in range(self.horizon):
                x0.extend([I_est]*self.n_stacks)
                x0.extend(last_v_lye.tolist())
                x0.append(last_v_c)
            
        # Construct Parameters
        p = []
        p.extend(state_vec.flatten().tolist()) # Full state vector (13)
        p.append(self.T_ref) # T_ref (Kelvin)
        p.extend(P_ref_vec)  # P_ref (N)
        p.extend(last_I.tolist())
        p.extend(last_v_lye.tolist())
        p.append(last_v_c)
        
        try:
            sol = self.solver(x0=x0, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=p)
            u_opt = sol['x'].full().flatten()
            self.prev_sol_x = u_opt # Save for warm start
            return u_opt
        except Exception as e:
            print(f"Multi-Stack NMPC Failed: {e}")
            return None

    def get_action(self, state_vec, P_ref_vec, T_ref, last_action):
        u_opt = self.solve_nmpc(state_vec, P_ref_vec, T_ref, last_action)
        
        last_I = last_action[0]
        last_v_lye = last_action[1]
        last_v_c = last_action[2]
        
        if u_opt is not None:
            # Extract first step
            u0 = u_opt[0 : self.n_controls]
            I_cmd = u0[0 : self.n_stacks]
            v_lye_cmd = u0[self.n_stacks : 2*self.n_stacks]
            v_c_cmd = u0[2*self.n_stacks]
            
            return I_cmd, v_lye_cmd, v_c_cmd
        else:
            return last_I, last_v_lye, last_v_c

    def get_all_actions_states(self, state_vec, P_ref_vec, T_ref, last_action):
        """
        Returns all optimized actions AND predicted states in the horizon.
        Actions shape: (N, n_controls) [I_1..n_stacks, v_lye_1..n_stacks, v_c]
        States shape: (N, 13) [T_s_in, T_s_1..n_stacks, T_sep, T_c_out, n_H2_an_1..n_stacks, n_liq, n_gas]
        Note: The simplified NMPC model only predicts thermal states (7 vars). 
        The other 6 states (n_H2_an, n_liq, n_gas) are not dynamically evolved in the controller's simplified model.
        We will pad them with the initial values (constant assumption for short horizon) or simple integration if possible.
        For now, we return constant values for non-thermal states to match the 13-dim requirement.
        """
        u_opt = self.solve_nmpc(state_vec, P_ref_vec, T_ref, last_action)
        
        # Default last action if not provided (for fallback)
        last_I = last_action[0]
        last_v_lye = last_action[1]
        last_v_c = last_action[2]
        
        if u_opt is not None:
            # Extract first step for internal state update (Side effect: update memory)
            u0 = u_opt[0 : self.n_controls]
            I_cmd = u0[0 : self.n_stacks]
            v_lye_cmd = u0[self.n_stacks : 2*self.n_stacks]
            v_c_cmd = u0[2*self.n_stacks]
            
            # 1. Actions
            # u_opt is flat (N * n_controls)
            actions = u_opt.reshape(self.horizon, self.n_controls)
            
            # 2. Predicted States
            # Re-construct P for state_func (logic from solve_nmpc)
            # P_ref_vec alignment
            if np.isscalar(P_ref_vec):
                P_ref_vec = [P_ref_vec] * self.horizon
            elif len(P_ref_vec) != self.horizon:
                pass
            
            p = []
            p.extend(state_vec.flatten().tolist())
            p.append(self.T_ref)
            p.extend(P_ref_vec)
            p.extend(last_I.tolist())
            p.extend(last_v_lye.tolist())
            p.append(last_v_c)
            
            # Evaluate state function for Thermal States (7 vars)
            # state_func output is flat (N * 7)
            pred_thermal_vec = self.state_func(u_opt, p).full().flatten()
            pred_thermal = pred_thermal_vec.reshape(self.horizon, 7) # [T_s_in, T_s(4), T_sep, T_c_out]
            
            # 3. Construct Full 13-dim State
            # Non-thermal states from initial condition:
            # n_H2_an_vec (4), n_liq (1), n_gas (1) -> indices 7-12 in state_vec
            non_thermal_initial = state_vec[7:13] # Shape (6,)
            
            # Repeat non-thermal states for the whole horizon (Simplification)
            # In reality, n_H2_an changes fast, n_liq/n_gas change slowly.
            # But NMPC simplified model doesn't track them.
            pred_non_thermal = np.tile(non_thermal_initial, (self.horizon, 1))
            
            # Concatenate: [Thermal(7), Non-Thermal(6)] -> (N, 13)
            pred_states = np.hstack([pred_thermal, pred_non_thermal])
            
            return actions, pred_states
        else:
            # Return copies of last action repeated
            last_action_flat = np.concatenate([last_I, last_v_lye, [last_v_c]])
            actions = np.tile(last_action_flat, (self.horizon, 1))
            
            # Repeat current state
            pred_states = np.tile(state_vec.flatten(), (self.horizon, 1))
            return actions, pred_states
