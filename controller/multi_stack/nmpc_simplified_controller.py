import casadi as ca
import numpy as np
from ..base_controller import BaseController

class MultiStackNMPCSimplifiedController(BaseController):
    def __init__(self, dt=60.0, horizon=5, dt_sub=0.2):
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
        self.F = 96485.0  # Faraday constant (C/mol)

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
        self.I_max = 7500.0
        self.v_lye_min = 0.01
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
        self.lambda_prod = 1e-6
        self.lambda_track = 1.2
        self.lambda_temp = 0.15
        self.lambda_I = 0.0002
        self.lambda_lye = 25000.0
        self.lambda_c = 2500.0

        self._setup_solver()

        self.prev_sol_x = None

    def _dynamics_step(self, T_s_in_k, T_s_k, T_sep_k, T_c_out_k, I_k, v_lye_k, v_c_k, n_gas_k):
        """
        Calculates the state at the next time step (k+1) given the current state (k) and inputs.
        Includes sub-stepping for numerical stability.
        This is a SIMPLIFIED thermal-only model; HTO states (n_H2_an, n_liq, n_gas) are NOT dynamically evolved.
        """
        n_sub = int(self.dt / self.dt_sub)

        T_s_in_sub = T_s_in_k
        T_s_sub = T_s_k
        T_sep_sub = T_sep_k
        T_c_out_sub = T_c_out_k

        T_K_vec = T_s_k
        T_C_vec = T_K_vec - 273.15

        R_ohm = self.r1 + self.r2 * T_K_vec + self.r3 * self.P_sys
        term_act = self.t1 + self.t2 / T_C_vec + self.t3 / (T_C_vec**2 + 1.0)
        U_act = self.s * ca.log(ca.fmax(term_act * I_k + 1.0, 1e-6))
        U_cell = self.U_rev + R_ohm * I_k + U_act
        V_cell = ca.fmax(U_cell, self.U_rev)

        f1 = 50.0 + 2.5 * T_C_vec
        f2 = 0.92 - 6.25e-6 * T_C_vec
        eta = f2 * (I_k**2) / (I_k**2 + f1 + 1e-6)

        Q_gen = self.n_cells * I_k * (V_cell - eta * 1.481)

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

        hto_pct = (n_gas_k * self.R * T_sep_sub) / (self.P_sys * self.V_sep_gas) * 100

        return T_s_in_sub, T_s_sub, T_sep_sub, T_c_out_sub, Power_k_vec, V_cell, hto_pct, eta

    def _setup_solver(self):
        import os
        self.n_controls = self.n_stacks * 2 + 1
        self.U = ca.MX.sym('U', self.n_controls * self.horizon)

        self.n_params = 13 + 1 + self.horizon + self.n_stacks * 2 + 1
        self.P = ca.MX.sym('P', self.n_params)

        p_idx = 0
        T_s_in_K = self.P[p_idx]; p_idx += 1
        T_s_K = self.P[p_idx : p_idx+self.n_stacks]; p_idx += self.n_stacks
        T_sep_K = self.P[p_idx]; p_idx += 1
        T_c_out_K = self.P[p_idx]; p_idx += 1

        n_H2_an = self.P[p_idx : p_idx+self.n_stacks]; p_idx += self.n_stacks
        n_liq = self.P[p_idx]; p_idx += 1
        n_gas = self.P[p_idx]; p_idx += 1

        T_ref_val = self.P[p_idx]; p_idx += 1
        P_ref = self.P[p_idx : p_idx+self.horizon]; p_idx += self.horizon

        I_0 = self.P[p_idx : p_idx+self.n_stacks]; p_idx += self.n_stacks
        v_lye_0 = self.P[p_idx : p_idx+self.n_stacks]; p_idx += self.n_stacks
        v_c_0 = self.P[p_idx]; p_idx += 1

        T_s_in_k = T_s_in_K
        T_s_k = T_s_K
        T_sep_k = T_sep_K
        T_c_out_k = T_c_out_K

        obj = 0
        g = []
        lbg = []
        ubg = []

        for k in range(self.horizon):
            uk = self.U[k*self.n_controls : (k+1)*self.n_controls]
            I_k = uk[0 : self.n_stacks]
            v_lye_k = uk[self.n_stacks : 2*self.n_stacks]
            v_c_k = uk[2*self.n_stacks]

            T_s_in_k, T_s_k, T_sep_k, T_c_out_k, Power_k_vec, V_cell, hto_pct, eta = self._dynamics_step(
                T_s_in_k, T_s_k, T_sep_k, T_c_out_k, I_k, v_lye_k, v_c_k, n_gas
            )

            Total_Power_k = ca.sum1(Power_k_vec)

            obj += self.lambda_track * ((Total_Power_k - P_ref[k])/1e6)**2
            obj += self.lambda_temp * ca.sum1((T_s_k - T_ref_val)**2)

            current_diff_sum = 0
            for i in range(self.n_stacks):
                for j in range(i+1, self.n_stacks):
                    current_diff_sum += (I_k[i] - I_k[j])**2
            obj += self.lambda_prod * current_diff_sum

            if k == 0:
                dI = I_k - I_0
            else:
                uk_prev = self.U[(k-1)*self.n_controls : k*self.n_controls]
                dI = I_k - uk_prev[0:self.n_stacks]

            dv_lye = v_lye_k - v_lye_0
            dv_c = v_c_k - v_c_0

            obj += self.lambda_I * ca.sum1(dI**2)
            obj += self.lambda_lye * ca.sum1(dv_lye**2)
            obj += self.lambda_c * dv_c**2

            g.append(T_s_k)
            lbg.extend([self.T_min]*self.n_stacks)
            ubg.extend([self.T_max]*self.n_stacks)

            g.append(Power_k_vec)
            lbg.extend([self.P_stack_min]*self.n_stacks)
            ubg.extend([self.P_stack_max]*self.n_stacks)

            g.append(V_cell)
            lbg.extend([self.U_cell_min]*self.n_stacks)
            ubg.extend([self.U_cell_max]*self.n_stacks)

            g.append(hto_pct)
            lbg.append(self.HTO_pct_min)
            ubg.append(self.HTO_pct_max)

        lbx = []
        ubx = []
        for k in range(self.horizon):
            lbx.extend([self.I_min]*self.n_stacks)
            ubx.extend([self.I_max]*self.n_stacks)
            lbx.extend([self.v_lye_min]*self.n_stacks)
            ubx.extend([self.v_lye_max]*self.n_stacks)
            lbx.append(self.v_c_min)
            ubx.append(self.v_c_max)

        self.lbx = lbx
        self.ubx = ubx
        self.lbg = lbg
        self.ubg = ubg

        pred_states = []

        p_idx = 0
        T_s_in_K = self.P[p_idx]; p_idx += 1
        T_s_K = self.P[p_idx : p_idx+self.n_stacks]; p_idx += self.n_stacks
        T_sep_K = self.P[p_idx]; p_idx += 1
        T_c_out_K = self.P[p_idx]; p_idx += 1
        p_idx += self.n_stacks + 2  # Skip HTO states for thermal prediction

        T_s_in_k = T_s_in_K
        T_s_k = T_s_K
        T_sep_k = T_sep_K
        T_c_out_k = T_c_out_K

        for k in range(self.horizon):
            uk = self.U[k*self.n_controls : (k+1)*self.n_controls]
            I_k = uk[0 : self.n_stacks]
            v_lye_k = uk[self.n_stacks : 2*self.n_stacks]
            v_c_k = uk[2*self.n_stacks]

            T_s_in_k, T_s_k, T_sep_k, T_c_out_k, _, _, _, _ = self._dynamics_step(
                T_s_in_k, T_s_k, T_sep_k, T_c_out_k, I_k, v_lye_k, v_c_k, n_gas
            )

            pred_states.append(ca.vertcat(T_s_in_k, T_s_k, T_sep_k, T_c_out_k))

        nlp = {'x': self.U, 'f': obj, 'g': ca.vertcat(*g), 'p': self.P}
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-4}

        print("Setting up simplified NMPC solver...")
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        self.solver = solver

        self.state_func = ca.Function('state_func', [self.U, self.P], [ca.vertcat(*pred_states)])

    def solve_nmpc(self, state_vec, P_ref_vec, T_ref, last_action):
        self.T_ref = T_ref

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
            I_est = (P_ref_0 / self.n_stacks) / (2.0 * self.n_cells)
            I_est = max(self.I_min, min(I_est, self.I_max))

            for k in range(self.horizon):
                x0.extend([I_est]*self.n_stacks)
                x0.extend(last_v_lye.tolist())
                x0.append(last_v_c)

        p = []
        p.extend(state_vec.flatten().tolist())
        p.append(self.T_ref)
        p.extend(P_ref_vec)
        p.extend(last_I.tolist())
        p.extend(last_v_lye.tolist())
        p.append(last_v_c)

        try:
            sol = self.solver(x0=x0, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=p)
            u_opt = sol['x'].full().flatten()
            self.prev_sol_x = u_opt
            return u_opt
        except Exception as e:
            print(f"Multi-Stack Simplified NMPC Failed: {e}")
            return None

    def get_action(self, state_vec, P_ref_vec, T_ref, last_action):
        u_opt = self.solve_nmpc(state_vec, P_ref_vec, T_ref, last_action)

        last_I = last_action[0]
        last_v_lye = last_action[1]
        last_v_c = last_action[2]

        if u_opt is not None:
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
        Note: The simplified NMPC model only predicts thermal states (7 vars).
        The other 6 states (n_H2_an, n_liq, n_gas) are not dynamically evolved.
        """
        u_opt = self.solve_nmpc(state_vec, P_ref_vec, T_ref, last_action)

        last_I = last_action[0]
        last_v_lye = last_action[1]
        last_v_c = last_action[2]

        if u_opt is not None:
            u0 = u_opt[0 : self.n_controls]
            I_cmd = u0[0 : self.n_stacks]
            v_lye_cmd = u0[self.n_stacks : 2*self.n_stacks]
            v_c_cmd = u0[2*self.n_stacks]

            actions = u_opt.reshape(self.horizon, self.n_controls)

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

            pred_thermal_vec = self.state_func(u_opt, p).full().flatten()
            pred_thermal = pred_thermal_vec.reshape(self.horizon, 7)

            non_thermal_initial = state_vec[7:13]
            pred_non_thermal = np.tile(non_thermal_initial, (self.horizon, 1))

            pred_states = np.hstack([pred_thermal, pred_non_thermal])

            return actions, pred_states
        else:
            last_action_flat = np.concatenate([last_I, last_v_lye, [last_v_c]])
            actions = np.tile(last_action_flat, (self.horizon, 1))

            pred_states = np.tile(state_vec.flatten(), (self.horizon, 1))
            return actions, pred_states
