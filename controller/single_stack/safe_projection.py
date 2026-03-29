"""
Safe Projection Operator for Single-Stack AWE System

Projects control inputs to the feasible set C defined by:
1. Input constraints (box constraints on I, v_lye, v_c)
2. One-step state constraints (predicted state must be within safe bounds)

Projection formulation:
    P_C(u) = argmin_{y∈C} ||y - u||²

where y = [I, v_lye, v_c] is the projected control input.
"""

import casadi as ca
import numpy as np
from scipy.optimize import minimize


class SingleStackSafeProjection:
    """
    Safe projection operator for single-stack AWE system.
    Ensures projected control keeps the system within constraints for one prediction step.
    """

    def __init__(self, dt=60.0, sim_dt=0.2):
        self.dt = dt
        self.sim_dt = sim_dt
        n_sub = int(dt / sim_dt) if sim_dt > 0 else 1
        self.n_sub = max(1, n_sub)
        self.dt_sub = dt / self.n_sub

        # --- System Parameters (from SingleStackSimulator) ---
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
        self.F = 96485.0

        # Thermal Parameters
        self.C_s_i = 3.450e7
        self.C_sep = 1e7
        self.C_he = 5e6
        self.C_c = 5e6
        self.h_A_stack = 1000.0
        self.T_amb = 298.15
        self.kA_hx = 960.0 * 240.0
        self.T_cw_in = 288.15
        self.c_cw = 4200.0
        self.rho_cw = 1000.0
        self.c_lye = 3200.0
        self.rho_lye = 1280.0
        self.R = 8.314
        self.V_sep = 10.288
        self.V_sep_gas = 0.6 * self.V_sep

        # --- Constraints ---
        # Input constraints
        self.I_min = 0.0
        self.I_max = 7800.0 * 1.2
        self.v_lye_min = 0.01
        self.v_lye_max = 0.1
        self.v_c_min = 0.0
        self.v_c_max = 1.0

        # State constraints (one-step prediction)
        self.T_min = 293.15  # 20°C
        self.T_max = 363.15  # 90°C
        self.P_stack_max = 6.0e6
        self.P_stack_min = 0.0
        self.U_cell_min = 0.0
        self.U_cell_max = 2.2
        self.HTO_pct_max = 2.0
        self.HTO_pct_min = 0.0

        # Setup CasADi optimization problem
        self._setup_projection_solver()

    def _dynamics_step(self, T_s_in, T_s, T_sep, T_c_out, n_gas, I, v_lye, v_c):
        """
        Single-step dynamics prediction (symbolic using CasADi).
        Returns predicted state after dt.
        """
        T_s_in_sub = T_s_in
        T_s_sub = T_s
        T_sep_sub = T_sep
        T_c_out_sub = T_c_out

        T_K = T_s
        T_C = T_K - 273.15

        # Electrochemical calculations
        R_ohm = self.r1 + self.r2 * T_K + self.r3 * self.P_sys
        term_act = self.t1 + self.t2 / T_C + self.t3 / (T_C**2 + 1.0)
        U_act = self.s * ca.log(ca.fmax(term_act * I + 1.0, 1e-6))
        U_cell = self.U_rev + R_ohm * I + U_act
        V_cell = ca.fmax(U_cell, self.U_rev)

        # Faraday efficiency
        f1 = 50.0 + 2.5 * T_C
        f2 = 0.92 - 6.25e-6 * T_C
        eta = f2 * (I**2) / (I**2 + f1 + 1e-6)

        # Heat generation and power
        Q_gen = self.n_cells * I * (V_cell - eta * 1.481)
        Power = V_cell * I * self.n_cells

        # Sub-step integration
        for _ in range(self.n_sub):
            # Stack temperature
            Q_flow_stack = self.c_lye * self.rho_lye * v_lye * (T_s_sub - T_s_in_sub)
            Q_diss_stack = self.h_A_stack * (T_s_sub - self.T_amb)
            dT_s_dt = (Q_gen - Q_diss_stack - Q_flow_stack) / self.C_s_i
            T_s_sub = T_s_sub + dT_s_dt * self.dt_sub

            # Separator temperature
            Q_sep_diss = 500.0 * (T_sep_sub - self.T_amb)
            dT_sep_dt = (self.c_lye * self.rho_lye * v_lye * (T_s_sub - T_sep_sub) - Q_sep_diss) / self.C_sep
            T_sep_sub = T_sep_sub + dT_sep_dt * self.dt_sub

            # HX lye outlet temperature
            dT_hot = T_sep_sub - T_c_out_sub
            Q_hx = self.kA_hx * dT_hot
            dT_s_in_dt = (self.c_lye * self.rho_lye * v_lye * (T_sep_sub - T_s_in_sub) - Q_hx) / self.C_he
            T_s_in_sub = T_s_in_sub + dT_s_in_dt * self.dt_sub

            # Cooling water outlet temperature
            dT_c_out_dt = (self.c_cw * self.rho_cw * v_c * (self.T_cw_in - T_c_out_sub) + Q_hx) / self.C_c
            T_c_out_sub = T_c_out_sub + dT_c_out_dt * self.dt_sub

        # HTO calculation
        hto_pct = (n_gas * self.R * T_sep_sub) / (self.P_sys * self.V_sep_gas) * 100

        return T_s_in_sub, T_s_sub, T_sep_sub, T_c_out_sub, Power, V_cell, hto_pct

    def _setup_projection_solver(self):
        """Setup CasADi NLP solver for projection problem."""
        # Decision variables: projected control [I, v_lye, v_c]
        u = ca.MX.sym('u', 3)
        I = u[0]
        v_lye = u[1]
        v_c = u[2]

        # Parameters: current state [T_s_in, T_s, T_sep, T_c_out, n_gas] and reference control
        state = ca.MX.sym('state', 5)
        T_s_in_0 = state[0]
        T_s_0 = state[1]
        T_sep_0 = state[2]
        T_c_out_0 = state[3]
        n_gas_0 = state[4]

        u_ref = ca.MX.sym('u_ref', 3)

        # Predict next state
        T_s_in_next, T_s_next, T_sep_next, T_c_out_next, Power_next, V_cell_next, hto_pct_next = \
            self._dynamics_step(T_s_in_0, T_s_0, T_sep_0, T_c_out_0, n_gas_0, I, v_lye, v_c)

        # Objective: minimize ||u - u_ref||^2
        obj = ca.sum1((u - u_ref)**2)

        # Constraints
        g = []
        lbg = []
        ubg = []

        # State constraints
        g.extend([T_s_next])           # Stack temperature
        lbg.append(self.T_min)
        ubg.append(self.T_max)

        g.extend([Power_next])         # Stack power
        lbg.append(self.P_stack_min)
        ubg.append(self.P_stack_max)

        g.extend([V_cell_next])        # Cell voltage
        lbg.append(self.U_cell_min)
        ubg.append(self.U_cell_max)

        g.extend([hto_pct_next])       # HTO percentage
        lbg.append(self.HTO_pct_min)
        ubg.append(self.HTO_pct_max)

        # Input bounds
        lbx = [self.I_min, self.v_lye_min, self.v_c_min]
        ubx = [self.I_max, self.v_lye_max, self.v_c_max]

        # Setup NLP
        nlp = {'x': u, 'f': obj, 'g': ca.vertcat(*g), 'p': ca.vertcat(state, u_ref)}

        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.tol': 1e-4,
            'ipopt.max_iter': 100
        }

        self.solver = ca.nlpsol('projector', 'ipopt', nlp, opts)
        self.lbx = lbx
        self.ubx = ubx
        self.lbg = lbg
        self.ubg = ubg

    def project(self, u_ref, state, verbose=False):
        """
        Project reference control to feasible set.

        Args:
            u_ref: Reference control [I, v_lye, v_c]
            state: Current state [T_s_in, T_s, T_sep, T_c_out, n_gas]
            verbose: Print solver info

        Returns:
            u_safe: Projected control within feasible set
            success: Whether projection was successful
        """
        u_ref = np.asarray(u_ref).flatten()
        state = np.asarray(state).flatten()

        # Initial guess
        x0 = u_ref.tolist()

        # Parameters
        p = np.concatenate([state, u_ref]).tolist()

        try:
            sol = self.solver(
                x0=x0,
                lbx=self.lbx,
                ubx=self.ubx,
                lbg=self.lbg,
                ubg=self.ubg,
                p=p
            )

            u_safe = np.array(sol['x'].full()).flatten()
            success = True

            if verbose:
                print(f"Projection: ||u_safe - u_ref|| = {np.linalg.norm(u_safe - u_ref):.4f}")

        except Exception as e:
            if verbose:
                print(f"Projection failed: {e}")
            # Fallback: clip to bounds only
            u_safe = np.clip(u_ref,
                           [self.I_min, self.v_lye_min, self.v_c_min],
                           [self.I_max, self.v_lye_max, self.v_c_max])
            success = False

        return u_safe, success

    def project_batch(self, u_refs, states):
        """
        Batch projection for multiple control/state pairs.

        Args:
            u_refs: Array of shape (N, 3) - reference controls
            states: Array of shape (N, 5) - current states

        Returns:
            u_safes: Projected controls (N, 3)
            successes: Success flags (N,)
        """
        u_refs = np.asarray(u_refs)
        states = np.asarray(states)

        N = u_refs.shape[0]
        u_safes = np.zeros_like(u_refs)
        successes = np.zeros(N, dtype=bool)

        for i in range(N):
            u_safes[i], successes[i] = self.project(u_refs[i], states[i])

        return u_safes, successes


class SingleStackSafeProjectionScipy:
    """
    Alternative implementation using SciPy optimize (no CasADi dependency).
    Suitable for deployment where CasADi is not available.
    """

    def __init__(self, dt=60.0, sim_dt=0.2):
        self.dt = dt
        self.sim_dt = sim_dt
        n_sub = int(dt / sim_dt) if sim_dt > 0 else 1
        self.n_sub = max(1, n_sub)
        self.dt_sub = dt / self.n_sub

        # Copy parameters from CasADi version
        self.n_cells = 368
        self.I_rated = 7800.0
        self.U_rev = 1.229
        self.r1 = 3.202e-5
        self.r2 = 8.970e-8
        self.r3 = -4.193e-12
        self.P_sys = 1.6e6
        self.s = 7.572e-2
        self.t1 = -1.070e-1
        self.t2 = 14.43
        self.t3 = 38.8
        self.F = 96485.0
        self.C_s_i = 3.450e7
        self.C_sep = 1e7
        self.C_he = 5e6
        self.C_c = 5e6
        self.h_A_stack = 1000.0
        self.T_amb = 298.15
        self.kA_hx = 960.0 * 240.0
        self.T_cw_in = 288.15
        self.c_cw = 4200.0
        self.rho_cw = 1000.0
        self.c_lye = 3200.0
        self.rho_lye = 1280.0
        self.R = 8.314
        self.V_sep = 10.288
        self.V_sep_gas = 0.6 * self.V_sep

        # Constraints
        self.I_min, self.I_max = 0.0, 7800.0 * 1.2
        self.v_lye_min, self.v_lye_max = 0.01, 0.1
        self.v_c_min, self.v_c_max = 0.0, 1.0
        self.T_min, self.T_max = 293.15, 363.15
        self.P_stack_max = 6.0e6
        self.U_cell_max = 2.2
        self.HTO_pct_max = 2.0

    def _predict_state(self, state, u):
        """Predict next state using numerical integration."""
        T_s_in, T_s, T_sep, T_c_out, n_gas = state
        I, v_lye, v_c = u

        T_s_in_sub = T_s_in
        T_s_sub = T_s
        T_sep_sub = T_sep
        T_c_out_sub = T_c_out

        for _ in range(self.n_sub):
            T_K = T_s_sub
            T_C = T_K - 273.15

            R_ohm = self.r1 + self.r2 * T_K + self.r3 * self.P_sys
            term_act = self.t1 + self.t2 / T_C + self.t3 / (T_C**2 + 1.0)
            U_act = self.s * np.log(max(term_act * I + 1.0, 1e-6))
            U_cell = self.U_rev + R_ohm * I + U_act
            V_cell = max(U_cell, self.U_rev)

            f1 = 50.0 + 2.5 * T_C
            f2 = 0.92 - 6.25e-6 * T_C
            eta = f2 * (I**2) / (I**2 + f1 + 1e-6)

            Q_gen = self.n_cells * I * (V_cell - eta * 1.481)

            # Stack temperature
            Q_flow_stack = self.c_lye * self.rho_lye * v_lye * (T_s_sub - T_s_in_sub)
            Q_diss_stack = self.h_A_stack * (T_s_sub - self.T_amb)
            dT_s_dt = (Q_gen - Q_diss_stack - Q_flow_stack) / self.C_s_i
            T_s_sub = T_s_sub + dT_s_dt * self.dt_sub

            # Separator temperature
            Q_sep_diss = 500.0 * (T_sep_sub - self.T_amb)
            dT_sep_dt = (self.c_lye * self.rho_lye * v_lye * (T_s_sub - T_sep_sub) - Q_sep_diss) / self.C_sep
            T_sep_sub = T_sep_sub + dT_sep_dt * self.dt_sub

            # HX lye outlet
            dT_hot = T_sep_sub - T_c_out_sub
            Q_hx = self.kA_hx * dT_hot
            dT_s_in_dt = (self.c_lye * self.rho_lye * v_lye * (T_sep_sub - T_s_in_sub) - Q_hx) / self.C_he
            T_s_in_sub = T_s_in_sub + dT_s_in_dt * self.dt_sub

            # Cooling water outlet
            dT_c_out_dt = (self.c_cw * self.rho_cw * v_c * (self.T_cw_in - T_c_out_sub) + Q_hx) / self.C_c
            T_c_out_sub = T_c_out_sub + dT_c_out_dt * self.dt_sub

        hto_pct = (n_gas * self.R * T_sep_sub) / (self.P_sys * self.V_sep_gas) * 100
        Power = V_cell * I * self.n_cells

        return T_s_sub, Power, V_cell, hto_pct

    def project(self, u_ref, state, method='SLSQP'):
        """
        Project using SciPy optimize.

        Args:
            u_ref: Reference control [I, v_lye, v_c]
            state: Current state [T_s_in, T_s, T_sep, T_c_out, n_gas]
            method: Optimization method

        Returns:
            u_safe: Projected control
            success: Whether optimization succeeded
        """
        u_ref = np.asarray(u_ref)
        state = np.asarray(state)

        # Objective: minimize ||u - u_ref||^2
        def objective(u):
            return np.sum((u - u_ref)**2)

        # Constraints
        constraints = []

        def state_constraints(u):
            T_s_next, P_next, V_cell_next, hto_next = self._predict_state(state, u)
            return np.array([T_s_next, P_next, V_cell_next, hto_next])

        # T_s ∈ [T_min, T_max]
        constraints.append({'type': 'ineq', 'fun': lambda u: state_constraints(u)[0] - self.T_min})
        constraints.append({'type': 'ineq', 'fun': lambda u: self.T_max - state_constraints(u)[0]})

        # P ∈ [0, P_max]
        constraints.append({'type': 'ineq', 'fun': lambda u: state_constraints(u)[1]})
        constraints.append({'type': 'ineq', 'fun': lambda u: self.P_stack_max - state_constraints(u)[1]})

        # V_cell ∈ [0, V_max]
        constraints.append({'type': 'ineq', 'fun': lambda u: state_constraints(u)[2]})
        constraints.append({'type': 'ineq', 'fun': lambda u: self.U_cell_max - state_constraints(u)[2]})

        # HTO ∈ [0, HTO_max]
        constraints.append({'type': 'ineq', 'fun': lambda u: state_constraints(u)[3]})
        constraints.append({'type': 'ineq', 'fun': lambda u: self.HTO_pct_max - state_constraints(u)[3]})

        # Bounds
        bounds = [
            (self.I_min, self.I_max),
            (self.v_lye_min, self.v_lye_max),
            (self.v_c_min, self.v_c_max)
        ]

        # Solve
        result = minimize(
            objective,
            u_ref,
            method=method,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 100, 'ftol': 1e-6}
        )

        if result.success:
            return result.x, True
        else:
            # Fallback to clipping
            return np.clip(u_ref,
                         [self.I_min, self.v_lye_min, self.v_c_min],
                         [self.I_max, self.v_lye_max, self.v_c_max]), False


# --- Helper functions for easy use ---

def project_control_safe(u_ref, state, use_casadi=True, dt=60.0):
    """
    Convenience function for single control projection.

    Args:
        u_ref: [I, v_lye, v_c] reference control
        state: [T_s_in, T_s, T_sep, T_c_out, n_gas] current state
        use_casadi: Use CasADi (True) or SciPy (False)
        dt: Control time step

    Returns:
        u_safe: Projected safe control
    """
    if use_casadi:
        projector = SingleStackSafeProjection(dt=dt)
    else:
        projector = SingleStackSafeProjectionScipy(dt=dt)

    u_safe, success = projector.project(u_ref, state)
    return u_safe


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Safe Projection Operator Test")
    print("=" * 60)

    # Initial state
    state = np.array([
        343.15,   # T_s_in: 70°C
        353.15,   # T_s: 80°C
        348.15,   # T_sep: 75°C
        293.15,   # T_c_out: 20°C
        25.0      # n_gas: 25 mol
    ])

    # Reference control (may violate constraints)
    u_ref = np.array([6000.0, 0.05, 0.5])

    print(f"\nCurrent state: T_s={state[1]-273.15:.1f}°C")
    print(f"Reference control: I={u_ref[0]:.1f}A, v_lye={u_ref[1]:.4f}, v_c={u_ref[2]:.3f}")

    # Test CasADi version
    print("\n--- CasADi Version ---")
    proj_casadi = SingleStackSafeProjection()
    u_safe_casadi, success_casadi = proj_casadi.project(u_ref, state, verbose=True)
    print(f"Projected control: I={u_safe_casadi[0]:.1f}A, v_lye={u_safe_casadi[1]:.4f}, v_c={u_safe_casadi[2]:.3f}")
    print(f"Success: {success_casadi}")

    # Test SciPy version
    print("\n--- SciPy Version ---")
    proj_scipy = SingleStackSafeProjectionScipy()
    u_safe_scipy, success_scipy = proj_scipy.project(u_ref, state)
    print(f"Projected control: I={u_safe_scipy[0]:.1f}A, v_lye={u_safe_scipy[1]:.4f}, v_c={u_safe_scipy[2]:.3f}")
    print(f"Success: {success_scipy}")

    print("\n" + "=" * 60)
