import numpy as np

class HeatPump:
    def __init__(self, capacity_kw=5.0, cop_nominal=3.5, max_temp=90.0):
        """
        Simple Heat Pump Model.
        
        Args:
            capacity_kw (float): Nominal heating capacity [kW].
            cop_nominal (float): Nominal COP at standard conditions.
            max_temp (float): Maximum output temperature [C].
        """
        self.capacity = capacity_kw * 1000.0 # Convert to W
        self.cop_nominal = cop_nominal
        self.max_temp = max_temp
        self.is_on = True

    def get_performance(self, T_source, T_sink):
        """
        Calculate COP and Capacity based on operating temperatures.
        Using a simplified Carnot efficiency model with an efficiency factor.
        
        COP_carnot = T_sink_K / (T_sink_K - T_source_K)
        COP_real = eta * COP_carnot
        """
        T_source_K = T_source + 273.15
        T_sink_K = T_sink + 273.15
        
        # Avoid division by zero or unrealistic physics
        if T_sink_K <= T_source_K + 1.0:
            delta_T = 1.0
        else:
            delta_T = T_sink_K - T_source_K
            
        cop_carnot = T_sink_K / delta_T
        
        # Assume nominal defined at dT = 40K (e.g. 10C -> 50C)
        # eta = cop_nominal / ( (273+50) / 40 )
        # eta approx 0.4 - 0.5 usually
        
        eta = self.cop_nominal / ( (273.15 + 55) / 45 ) # Calibrate roughly
        
        cop = eta * cop_carnot
        
        # Cap COP for realism
        cop = np.clip(cop, 1.0, 6.0)
        
        # Derating capacity at high delta T
        # Simple linear derating
        capacity_factor = 1.0 - 0.01 * (delta_T - 30)
        capacity_factor = np.clip(capacity_factor, 0.5, 1.2)
        
        current_capacity = self.capacity * capacity_factor
        
        return cop, current_capacity

    def step(self, T_source, T_sink, dt):
        """
        Run heat pump for one step.
        
        Returns:
            Q_source (float): Heat extracted from source [J]
            Q_sink (float): Heat delivered to sink [J]
            W_comp (float): Work done by compressor [J]
        """
        if not self.is_on or T_sink >= self.max_temp:
            return 0.0, 0.0, 0.0
            
        cop, capacity_w = self.get_performance(T_source, T_sink)
        
        # Energy per step
        q_sink = capacity_w * dt
        w_comp = q_sink / cop
        q_source = q_sink - w_comp
        
        return q_source, q_sink, w_comp

class ThermalStorage:
    def __init__(self, volume_m3=1.0, T_init=25.0, U_loss=2.0):
        """
        Stratified Water Tank Model (simplified as 1-node for now).
        
        Args:
            volume_m3 (float): Tank volume [m3].
            T_init (float): Initial temperature [C].
            U_loss (float): Heat loss coefficient [W/K].
        """
        self.volume = volume_m3
        self.density = 1000.0 # kg/m3
        self.cp = 4184.0 # J/kgK
        self.mass = self.volume * self.density
        self.C_th = self.mass * self.cp # J/K
        
        self.T = T_init
        self.U_loss = U_loss
        self.T_amb = 20.0 # Tank ambient

    def step(self, Q_in, Q_load, dt):
        """
        Update tank temperature.
        
        Args:
            Q_in (float): Energy added [J].
            Q_load (float): Energy removed [J].
            dt (float): Time step [s].
        """
        # Loss to ambient
        q_loss_env = self.U_loss * (self.T - self.T_amb) * dt
        
        # Energy balance
        # M Cp dT = Q_in - Q_load - Q_loss
        delta_E = Q_in - Q_load - q_loss_env
        
        dT = delta_E / self.C_th
        self.T += dT
        
        return self.T
