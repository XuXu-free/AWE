import casadi as ca
import numpy as np
from .base_controller import BaseController

class MPCController(BaseController):
    def __init__(self, dt, N=10, u_min=0.0, u_max=1000.0):
        super().__init__(dt)
        self.N = N  # Prediction horizon
        self.u_min = u_min
        self.u_max = u_max
        
        # System Parameters (Must match Simulator)
        self.n_cells = 10
        self.area = 0.25
        self.C_th = 50000.0
        self.h_A = 50.0
        self.T_amb = 25.0
        self.V_tn = 1.48
        self.V_rev = 1.23
        self.r = 0.0004
        
        self.setup_mpc()
        self.last_u = 0.0

    def setup_mpc(self):
        # 1. Define Symbolic Variables
        T = ca.MX.sym('T')
        I = ca.MX.sym('I')
        
        # 2. Define Dynamics (Discretized)
        # V_cell = V_rev + (r/A)*I
        v_cell = self.V_rev + (self.r / self.area) * I
        
        # Q_gen = (V_cell - V_tn) * I * n_cells
        q_gen = (v_cell - self.V_tn) * I * self.n_cells
        
        # Q_loss = h_A * (T - T_amb)
        q_loss = self.h_A * (T - self.T_amb)
        
        dT_dt = (q_gen - q_loss) / self.C_th
        
        # Euler discretization
        T_next = T + dT_dt * self.dt
        
        self.f_step = ca.Function('f_step', [T, I], [T_next])
        
        # 3. Optimization Problem
        # Variables over horizon
        self.U = ca.MX.sym('U', self.N) # Control inputs (Current)
        self.X0 = ca.MX.sym('X0')       # Initial state (Temperature)
        self.Ref = ca.MX.sym('Ref')     # Reference state (Target Temp)
        
        obj = 0
        g = [] # Constraints
        
        curr_x = self.X0
        
        # Weights
        w_trk = 10.0   # Tracking weight (Increased)
        w_reg = 1e-5   # Regularization (Decreased to allow higher current)
        w_slew = 0.1  # Slew rate (smoothness)
        
        prev_u = 0 # Or pass as parameter if we want strict continuity
        
        for k in range(self.N):
            u_k = self.U[k]
            
            # Dynamics constraint (Explicit substitution here for simple shooting)
            curr_x = self.f_step(curr_x, u_k)
            
            # Objective
            # Track reference
            obj += w_trk * (curr_x - self.Ref)**2 
            # Minimize energy/control effort (optional)
            obj += w_reg * u_k**2
            # Smoothness
            if k == 0:
                obj += w_slew * (u_k - prev_u)**2
            else:
                obj += w_slew * (u_k - self.U[k-1])**2
                
        # 4. Solver Setup
        nlp = {
            'x': self.U,
            'f': obj,
            'p': ca.vertcat(self.X0, self.Ref)
        }
        
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.tol': 1e-4
        }
        
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    def get_action(self, state, setpoint):
        """
        state: [Temperature]
        setpoint: [Target Temperature]
        """
        x0_val = state[0]
        ref_val = setpoint[0]
        
        # Bounds for U (Current)
        lbx = [self.u_min] * self.N
        ubx = [self.u_max] * self.N
        
        # Initial guess
        # If last_u is too small (endothermic region), solver might get stuck.
        # Initialize with a current that provides heating if T < Ref
        guess_u = self.last_u
        if guess_u < 200.0 and x0_val < ref_val:
             guess_u = 300.0
             
        x_init = [guess_u] * self.N
        
        try:
            sol = self.solver(
                x0=x_init,
                lbx=lbx,
                ubx=ubx,
                p=ca.vertcat(x0_val, ref_val)
            )
            
            u_opt = sol['x'].full().flatten()
            action = u_opt[0]
            self.last_u = action
            return np.array([action])
            
        except Exception as e:
            print(f"MPC Solver failed: {e}")
            return np.array([self.last_u]) # Fallback

    def reset(self):
        self.last_u = 0.0
