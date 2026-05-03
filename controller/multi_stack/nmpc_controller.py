import casadi as ca
import numpy as np
from ..base_controller import BaseController

class MultiStackNMPCController(BaseController):
    """
    Full-order NMPC controller for the 4-stack AWE system.
    Unlike the simplified variant, this controller dynamically evolves ALL 13 states
    (7 thermal + 6 HTO) over the prediction horizon using the same physics as the simulator.
    """
    def __init__(self, dt=60.0, horizon=5, dt_sub=0.2):
        self.dt = dt
        self.dt_sub = dt_sub
        self.horizon = horizon

        # --- System Parameters (Matched to MultiStackSimulator) ---
        self.n_stacks = 4
        self.n_cells = 368
        self.I_rated = 7800.0
        self.A_cell = 2.0

        # Electrochemical Parameters
        self.U_rev = 1.229
        self.U_th = 1.481
        self.r1 = 3.202e-5
        self.r2 = 8.970e-8
        self.r3 = -4.193e-12
        self.P_sys = 1.6e6
        self.delta_P = 0.01 * self.P_sys
        self.s = 7.572e-2
        self.t1 = -1.070e-1
        self.t2 = 14.43
        self.t3 = 38.8
        self.F = 96485.0

        # Thermal Parameters
        self.T_amb = 298.15
        self.T_c_in = 288.15
        self.C_s_i = 3.450e7
        self.C_sep = 5.193e7
        self.C_he = 2.175e7
        self.C_c = 2.0e7
        self.k_he = 960.0
        self.A_he = 240.0
        self.c_lye = 3200.0
        self.c_cw = 4200.0
        self.rho_lye = 1280.0
        self.rho_cw = 1000.0

        # Convection / Radiation parameters
        self.segma_s = 1000.0
        self.segma_sep = 200.0
        self.A_stack = 80.0
        self.A_sep = 40.0
        self.epsilon_stack = 0.8
        self.epsilon_sep = 0.8
        self.sigma_b = 5.67e-8

        # HTO Parameters
        self.V_an_lye = 2.5
        self.V_sep = 10.288
        self.V_sep_gas = 0.6 * self.V_sep
        self.delta = 500e-6
        self.D_eff = 8.569e-10
        self.K_eff = 2e-16
        self.tau_sep = 60.0
        self.mu_lye = 8.76e-4
        self.R = 8.314

        # Pre-compute H2 solubility in lye (constant)
        rho_H2O = 1000.0
        M_H2O = 18e-3
        p_atm = 101325.0
        H_H2 = 7.1698e4 * p_atm
        K_H2 = 3.14
        w_lye = 0.30
        S_H2_H2O = rho_H2O * self.P_sys / (M_H2O * p_atm * H_H2)
        self.S_H2_lye = float(S_H2_H2O / (10**(K_H2 * w_lye)))

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
        self.T_ref = 358.15

        # Weights
        self.lambda_prod = 1e-6
        self.lambda_track = 1.2
        self.lambda_temp = 0.15
        self.lambda_I = 0.0002
        self.lambda_lye = 25000.0
        self.lambda_c = 2500.0

        self._setup_solver()
        self.prev_sol_x = None

    def _dynamics_step(self, T_s_in_k, T_s_k, T_sep_k, T_c_out_k,
                       n_H2_an_k, n_H2_sep_liq_k, n_H2_sep_gas_k,
                       I_k, v_lye_k, v_c_k):
        """
        Full 13-state dynamics with sub-stepping.
        Evolves thermal states AND HTO states (n_H2_an, n_liq, n_gas).
        """
        n_sub = int(self.dt / self.dt_sub)

        # State copies for sub-stepping
        T_s_in_sub = T_s_in_k
        T_s_sub = T_s_k
        T_sep_sub = T_sep_k
        T_c_out_sub = T_c_out_k
        n_H2_an_sub = n_H2_an_k
        n_H2_sep_liq_sub = n_H2_sep_liq_k
        n_H2_sep_gas_sub = n_H2_sep_gas_k

        for _ in range(n_sub):
            T_C_vec = T_s_sub - 273.15

            # --- Electrochemistry ---
            R_ohm = self.r1 + self.r2 * T_s_sub + self.r3 * self.P_sys
            term_act = self.t1 + self.t2 / T_C_vec + self.t3 / (T_C_vec**2)
            arg = term_act * I_k + 1.0
            U_act = self.s * ca.log(ca.fmax(arg, 1e-9))
            U_cell = self.U_rev + R_ohm * I_k + U_act
            V_cell = ca.fmax(U_cell, self.U_rev)

            f1 = 50.0 + 2.5 * T_C_vec
            f2 = 0.92 - 6.25e-6 * T_C_vec
            eta_F = f2 * (I_k**2) / (I_k**2 + f1 + 1e-6)

            Q_gen = self.n_cells * I_k * (V_cell - eta_F * self.U_th)
            Power_k_vec = V_cell * I_k * self.n_cells

            # --- Thermal Dynamics (Full model) ---
            # Stack dissipation: convection + radiation
            Q_conv = self.segma_s * (T_s_sub - self.T_amb)
            Q_rad = self.epsilon_stack * self.sigma_b * self.A_stack * (T_s_sub**4 - self.T_amb**4)
            Q_diss = Q_conv + Q_rad

            Q_flow = self.c_lye * self.rho_lye * v_lye_k * (T_s_sub - T_s_in_sub)
            dT_s_dt = (Q_gen - Q_diss - Q_flow) / self.C_s_i
            T_s_sub = T_s_sub + dT_s_dt * self.dt_sub

            # Separator thermal
            v_tot = ca.sum1(v_lye_k)
            T_sep_in = ca.sum1(v_lye_k * T_s_sub) / (v_tot + 1e-6)

            Q_sep_conv = self.segma_sep * (T_sep_sub - self.T_amb)
            Q_sep_rad = self.epsilon_sep * self.sigma_b * self.A_sep * (T_sep_sub**4 - self.T_amb**4)
            Q_sep_diss = Q_sep_conv + Q_sep_rad

            dT_sep_dt = (0.5 * self.c_lye * self.rho_lye * v_tot * (T_sep_in - T_sep_sub) - Q_sep_diss) / self.C_sep
            T_sep_sub = T_sep_sub + dT_sep_dt * self.dt_sub

            # Heat exchanger with LMTD
            delta_T1 = T_s_in_sub - T_c_out_sub
            delta_T2 = T_sep_sub - self.T_c_in
            LMTD = self._safe_lmtd(delta_T1, delta_T2)
            Q_hx = self.k_he * self.A_he * LMTD

            dT_s_in_dt = (self.c_lye * self.rho_lye * v_tot * (T_sep_sub - T_s_in_sub) - Q_hx) / self.C_he
            T_s_in_sub = T_s_in_sub + dT_s_in_dt * self.dt_sub

            C_rate_cw = self.c_cw * self.rho_cw * v_c_k
            dT_c_out_dt = (C_rate_cw * (self.T_c_in - T_c_out_sub) + Q_hx) / self.C_c
            T_c_out_sub = T_c_out_sub + dT_c_out_dt * self.dt_sub

            # --- HTO Dynamics (Full model) ---
            # O2 production rate (carrier gas)
            n_dot_O2_prod = self.n_cells * I_k * eta_F / (4.0 * self.F)

            # H2 impurity inflow to anode
            n_dot_H2_lye = self.S_H2_lye * self.rho_lye * v_lye_k / 4.0
            n_dot_H2_diff = (self.A_cell * self.n_cells * self.D_eff * self.S_H2_lye * self.P_sys) / self.delta
            n_dot_H2_conv = (self.A_cell * self.n_cells * (self.K_eff / self.mu_lye) *
                             self.S_H2_lye * self.rho_lye * (self.delta_P / self.delta))
            n_dot_H2_im = n_dot_H2_lye + n_dot_H2_diff + n_dot_H2_conv

            # H2 outflow from anode to separator liquid
            n_dot_H2_im_1 = n_H2_an_sub * v_lye_k / (2.0 * self.V_an_lye)

            # Anode H2 balance
            n_dot_H2_an = n_dot_H2_im - n_dot_H2_im_1
            n_H2_an_sub = n_H2_an_sub + n_dot_H2_an * self.dt_sub

            # Separator liquid H2 balance
            n_dot_H2_2 = n_H2_sep_liq_sub / self.tau_sep
            n_dot_H2_sep_liq = ca.sum1(n_dot_H2_im_1) - n_dot_H2_2
            n_H2_sep_liq_sub = n_H2_sep_liq_sub + n_dot_H2_sep_liq * self.dt_sub

            # Separator gas H2 balance
            sum_n_dot_O2 = ca.sum1(n_dot_O2_prod)
            n_dot_H2_out = (self.R * T_sep_sub * n_H2_sep_gas_sub * sum_n_dot_O2) / (self.P_sys * self.V_sep_gas)
            n_dot_sep_gas = n_dot_H2_2 - n_dot_H2_out
            n_H2_sep_gas_sub = n_H2_sep_gas_sub + n_dot_sep_gas * self.dt_sub

        # Final HTO percentage
        hto_pct = (n_H2_sep_gas_sub * self.R * T_sep_sub) / (self.P_sys * self.V_sep_gas) * 100.0

        return (T_s_in_sub, T_s_sub, T_sep_sub, T_c_out_sub,
                n_H2_an_sub, n_H2_sep_liq_sub, n_H2_sep_gas_sub,
                Power_k_vec, V_cell, hto_pct, eta_F)

    def _safe_lmtd(self, dt1, dt2):
        """Numerically safe LMTD for CasADi."""
        eps = 1e-6
        # Standard formula with epsilon protection
        ratio = (dt1 + eps) / (dt2 + eps)
        log_ratio = ca.log(ca.fabs(ratio) + eps)
        lmtd = (dt1 - dt2) / (log_ratio + eps)
        return lmtd

    def _setup_solver(self):
        self.n_controls = self.n_stacks * 2 + 1  # 9
        self.U = ca.MX.sym('U', self.n_controls * self.horizon)

        # Parameters: 13 states + T_ref + N P_refs + 9 previous actions
        self.n_params = 13 + 1 + self.horizon + self.n_stacks * 2 + 1
        self.P = ca.MX.sym('P', self.n_params)

        # Unpack parameters
        p_idx = 0
        T_s_in_0 = self.P[p_idx]; p_idx += 1
        T_s_0 = self.P[p_idx : p_idx+self.n_stacks]; p_idx += self.n_stacks
        T_sep_0 = self.P[p_idx]; p_idx += 1
        T_c_out_0 = self.P[p_idx]; p_idx += 1

        n_H2_an_0 = self.P[p_idx : p_idx+self.n_stacks]; p_idx += self.n_stacks
        n_liq_0 = self.P[p_idx]; p_idx += 1
        n_gas_0 = self.P[p_idx]; p_idx += 1

        T_ref_val = self.P[p_idx]; p_idx += 1
        P_ref = self.P[p_idx : p_idx+self.horizon]; p_idx += self.horizon

        I_prev = self.P[p_idx : p_idx+self.n_stacks]; p_idx += self.n_stacks
        v_lye_prev = self.P[p_idx : p_idx+self.n_stacks]; p_idx += self.n_stacks
        v_c_prev = self.P[p_idx]; p_idx += 1

        # State variables for horizon loop
        T_s_in_k = T_s_in_0
        T_s_k = T_s_0
        T_sep_k = T_sep_0
        T_c_out_k = T_c_out_0
        n_H2_an_k = n_H2_an_0
        n_liq_k = n_liq_0
        n_gas_k = n_gas_0

        obj = 0
        g = []
        lbg = []
        ubg = []

        # Store states for trajectory extraction
        stored_states = []

        for k in range(self.horizon):
            uk = self.U[k*self.n_controls : (k+1)*self.n_controls]
            I_k = uk[0 : self.n_stacks]
            v_lye_k = uk[self.n_stacks : 2*self.n_stacks]
            v_c_k = uk[2*self.n_stacks]

            (T_s_in_k, T_s_k, T_sep_k, T_c_out_k,
             n_H2_an_k, n_liq_k, n_gas_k,
             Power_k_vec, V_cell, hto_pct, eta_F) = self._dynamics_step(
                T_s_in_k, T_s_k, T_sep_k, T_c_out_k,
                n_H2_an_k, n_liq_k, n_gas_k,
                I_k, v_lye_k, v_c_k
            )

            Total_Power_k = ca.sum1(Power_k_vec)

            # --- Objective ---
            obj += self.lambda_track * ((Total_Power_k - P_ref[k]) / 1e6)**2
            obj += self.lambda_temp * ca.sum1((T_s_k - T_ref_val)**2)

            current_diff_sum = 0
            for i in range(self.n_stacks):
                for j in range(i+1, self.n_stacks):
                    current_diff_sum += (I_k[i] - I_k[j])**2
            obj += self.lambda_prod * current_diff_sum

            if k == 0:
                dI = I_k - I_prev
            else:
                uk_prev = self.U[(k-1)*self.n_controls : k*self.n_controls]
                dI = I_k - uk_prev[0:self.n_stacks]

            dv_lye = v_lye_k - v_lye_prev
            dv_c = v_c_k - v_c_prev

            obj += self.lambda_I * ca.sum1(dI**2)
            obj += self.lambda_lye * ca.sum1(dv_lye**2)
            obj += self.lambda_c * dv_c**2

            # --- Constraints ---
            g.append(T_s_k)
            lbg.extend([self.T_min] * self.n_stacks)
            ubg.extend([self.T_max] * self.n_stacks)

            g.append(Power_k_vec)
            lbg.extend([self.P_stack_min] * self.n_stacks)
            ubg.extend([self.P_stack_max] * self.n_stacks)

            g.append(V_cell)
            lbg.extend([self.U_cell_min] * self.n_stacks)
            ubg.extend([self.U_cell_max] * self.n_stacks)

            g.append(hto_pct)
            lbg.append(self.HTO_pct_min)
            ubg.append(self.HTO_pct_max)

            # Store full 13-dim state
            stored_states.append(ca.vertcat(T_s_in_k, T_s_k, T_sep_k, T_c_out_k,
                                            n_H2_an_k, n_liq_k, n_gas_k))

        # Input bounds
        lbx = []
        ubx = []
        for k in range(self.horizon):
            lbx.extend([self.I_min] * self.n_stacks)
            ubx.extend([self.I_max] * self.n_stacks)
            lbx.extend([self.v_lye_min] * self.n_stacks)
            ubx.extend([self.v_lye_max] * self.n_stacks)
            lbx.append(self.v_c_min)
            ubx.append(self.v_c_max)

        self.lbx = lbx
        self.ubx = ubx
        self.lbg = lbg
        self.ubg = ubg

        nlp = {'x': self.U, 'f': obj, 'g': ca.vertcat(*g), 'p': self.P}
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-4}

        print("Setting up full-order NMPC solver (13 states)...")
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        # State prediction function (13-dim)
        self.state_func = ca.Function('state_func_full', [self.U, self.P], [ca.vertcat(*stored_states)])

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
                x0.extend([I_est] * self.n_stacks)
                x0.extend(last_v_lye.tolist())
                x0.append(last_v_c)

        p = []
        p.extend(state_vec.flatten().tolist())  # 13 states
        p.append(self.T_ref)
        p.extend(P_ref_vec)
        p.extend(last_I.tolist())
        p.extend(last_v_lye.tolist())
        p.append(last_v_c)

        try:
            sol = self.solver(x0=x0, lbx=self.lbx, ubx=self.ubx,
                              lbg=self.lbg, ubg=self.ubg, p=p)
            u_opt = sol['x'].full().flatten()
            self.prev_sol_x = u_opt
            return u_opt
        except Exception as e:
            print(f"Multi-Stack Full NMPC Failed: {e}")
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
        States shape: (N, 13) [T_s_in, T_s_1..4, T_sep, T_c_out, n_H2_an_1..4, n_liq, n_gas]
        All 13 states are dynamically evolved in this full-order controller.
        """
        u_opt = self.solve_nmpc(state_vec, P_ref_vec, T_ref, last_action)

        last_I = last_action[0]
        last_v_lye = last_action[1]
        last_v_c = last_action[2]

        if u_opt is not None:
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

            pred_vec = self.state_func(u_opt, p).full().flatten()
            pred_states = pred_vec.reshape(self.horizon, 13)

            return actions, pred_states
        else:
            last_action_flat = np.concatenate([last_I, last_v_lye, [last_v_c]])
            actions = np.tile(last_action_flat, (self.horizon, 1))
            pred_states = np.tile(state_vec.flatten(), (self.horizon, 1))
            return actions, pred_states
