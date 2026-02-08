import numpy as np
from .base_controller import BaseController

class PIDController(BaseController):
    def __init__(self, dt, kp, ki, kd, u_min=0.0, u_max=1000.0):
        super().__init__(dt)
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.u_min = u_min
        self.u_max = u_max
        
        self.integral = 0.0
        self.prev_error = 0.0
        
    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        
    def get_action(self, state, setpoint):
        """
        state: [Temperature]
        setpoint: [Target Temperature]
        """
        current_val = state[0]
        target_val = setpoint[0]
        
        error = target_val - current_val
        
        # Proportional
        p_term = self.kp * error
        
        # Integral
        self.integral += error * self.dt
        # Anti-windup (simple clamping)
        # self.integral = np.clip(self.integral, -1000, 1000) 
        i_term = self.ki * self.integral
        
        # Derivative
        d_term = self.kd * (error - self.prev_error) / self.dt
        self.prev_error = error
        
        u = p_term + i_term + d_term
        
        # Saturation
        u = np.clip(u, self.u_min, self.u_max)
        
        return np.array([u])
