
import numpy as np
from plant.multi_stack_simulator import MultiStackSimulator
from plant.thermal_systems import HeatPump, ThermalStorage

class IntegratedSimulator(MultiStackSimulator):
    def __init__(self, dt=1.0):
        super().__init__(dt)
        
        # Initialize Integrated Thermal Systems
        # Heat Pump sized to recover some of the waste heat
        # Assuming we want to demonstrate significant heating
        self.hp = HeatPump(capacity_kw=100.0, cop_nominal=4.0, max_temp=90.0)
        
        # Thermal Storage Tank
        # 5 m^3 water tank
        self.storage = ThermalStorage(volume_m3=5.0, T_init=20.0, U_loss=5.0)
        
        # Instantaneous metrics
        self.hp_power = 0.0     # W (Electricity consumption)
        self.hp_heat_out = 0.0  # W (Heat delivered to tank)
        self.hp_heat_in = 0.0   # W (Heat extracted from cooling water)
        
    def step(self, action):
        """
        Step the combined system.
        """
        # 1. Step the Electrolyzer Plant
        super().step(action)
        
        # 2. Extract Interface Variables
        # T_c_out is index 6 in state vector (Kelvin)
        T_c_out_K = self.state[6]
        T_c_out_C = T_c_out_K - 273.15
        
        # 3. Step Heat Pump
        # Source: Cooling Water Output (T_c_out)
        # Sink: Storage Tank (self.storage.T)
        # Returns Energy in Joules for the timestep
        q_source, q_sink, w_work = self.hp.step(T_source=T_c_out_C, T_sink=self.storage.T, dt=self.dt)
        
        # 4. Step Thermal Storage
        # Q_in = Heat from HP
        # Q_load = 0 (No demand modeled yet)
        self.storage.step(Q_in=q_sink, Q_load=0.0, dt=self.dt)
        
        # 5. Store metrics
        if self.dt > 0:
            self.hp_power = w_work / self.dt
            self.hp_heat_out = q_sink / self.dt
            self.hp_heat_in = q_source / self.dt
        
        return self.state

    def get_full_state(self):
        """
        Return a dictionary with both plant and thermal system states.
        """
        return {
            "T_c_out_K": self.state[6],
            "T_c_out_C": self.state[6] - 273.15,
            "Tank_T_C": self.storage.T,
            "HP_Power_kW": self.hp_power / 1000.0,
            "HP_Heat_Out_kW": self.hp_heat_out / 1000.0,
            "HP_COP": (self.hp_heat_out / self.hp_power) if self.hp_power > 1e-3 else 0.0
        }
