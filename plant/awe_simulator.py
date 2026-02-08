import numpy as np
from .base_simulator import BaseSimulator
from .thermal_systems import HeatPump, ThermalStorage

class AWESimulator(BaseSimulator):
    """
    Alkaline Water Electrolysis (AWE) System Simulator.
    
    Integrated with Condensate Heat Pump & Thermal Storage.
    
    Thermal Model:
    dT/dt = (1/C_th) * (Q_gen - Q_loss - Q_recovery)
    Q_gen = (V_cell - V_tn) * I * n_cells
    Q_loss = h * A * (T - T_amb)
    Q_recovery = Heat extracted by Heat Pump
    """
    
    def __init__(self, dt=1.0):
        super().__init__(dt)
        # System Parameters (Based on arXiv:2501.14576 4-in-1 system, single stack parameters)
        self.n_cells = 368          # Number of cells
        self.A_cell = 2.0           # Cell area [m^2]
        self.area = self.A_cell     # Alias
        self.I_rated = 7800.0       # Rated current [A]
        self.I_max = 1.2 * self.I_rated # Max current
        self.P_sys = 1.6e6          # System pressure [Pa] (1.6 MPa)
        
        # Safety Thresholds
        self.T_max_safe = 90.0      # [C] Max safe operating temp
        self.T_shutdown = 95.0      # [C] Shutdown temp
        
        # Electrochemical Parameters
        self.U_rev = 1.229          # Reversible voltage [V]
        self.U_tn = 1.481           # Thermoneutral voltage [V]
        self.V_tn = self.U_tn       # Alias
        
        # Ohmic parameters
        self.r1 = 3.202e-5          # [Ohm]
        self.r2 = 8.970e-8          # [Ohm/K]
        self.r3 = -4.193e-12        # [Ohm/Pa]
        
        # Activation parameters
        self.s = 7.572e-2           # [V]
        self.t1 = -1.070e-1         # [-]
        self.t2 = 14.43             # [K]
        self.t3 = 38.8              # [K^2]
        
        # Safety / Impurity Model
        self.gas_crossover_rate = 2.0e-5  # [mol/s] Constant crossover of H2 into O2
        self.impurity_limit = 2.0         # [%] Max allowed H2 in O2
        
        # Thermal Parameters
        self.C_th = 3.450e7         # [J/K]
        self.T_amb = 25.0           # [C]
        self.h_A = 500.0            # [W/K] (Natural convection)
        
        # Cooling Parameters (Lye side)
        self.c_lye = 3500.0         # [J/kgK] Specific heat of lye (approx 30% KOH)
        self.rho_lye = 1250.0       # [kg/m^3] Density
        self.lye_flow_rated = 0.04  # [m^3/s]
        
        # Heat Exchanger & Cooling Water Parameters (arXiv:2501.14576)
        # Heat transfer coefficient k * Area A
        self.kA_hx = 50000.0        # [W/K] Estimated for MW scale
        self.T_cw_in = 20.0         # [C] Inlet cooling water temp (Chiller)
        self.c_cw = 4180.0          # [J/kgK] Water
        self.rho_cw = 1000.0        # [kg/m3] Water
        self.C_cw = 1.0e6           # [J/K] Heat capacity of HX cooling side
        
        # State: [T_stack, T_cw_out]
        # Note: T_coolant (Lye into stack) is T_lye_out_hx.
        # Simple mixing model: T_lye_out_hx approx T_stack - Q_hx / (flow * cp)
        # Or better: Model T_lye_in_stack explicitly?
        # The paper uses:
        # C_c * dT_c_out/dt = ...
        # Q_hx = kA * LMTD
        
        # Thermal Recovery System
        self.hp = HeatPump(capacity_kw=100.0, cop_nominal=3.5, max_temp=85.0)
        self.storage = ThermalStorage(volume_m3=5.0, T_init=25.0)
        self.hp_setpoint = 60.0 
        
        # Initial State
        self.reset()

    def reset(self, initial_state=None):
        if initial_state is not None:
            self.state = np.array(initial_state)
        else:
            # State: [T_stack, T_cw_out]
            self.state = np.array([self.T_amb, self.T_cw_in]) 
        self.current_time = 0.0
        # Reset sub-systems if needed
        self.storage.T = self.T_amb
        return self.state

    def get_voltage(self, current, T):
        """
        Calculate cell voltage using Sanchez model.
        """
        # Ensure non-zero T for division
        T_c = max(T, 0.1) 
        T_k = T + 273.15
        
        # Ohmic Resistance
        R_ohm = self.r1 + self.r2 * T_k + self.r3 * self.P_sys
        
        # Activation Overpotential
        act_coeff = self.t1 + self.t2 / T_c + self.t3 / (T_c**2)
        
        # Sanity check
        if act_coeff * current <= -1.0:
            act_coeff = 1e-6
            
        U_act = self.s * np.log(act_coeff * current + 1.0)
        
        U_cell = self.U_rev + R_ohm * current + U_act
        return max(U_cell, self.U_rev)

    def get_faraday_efficiency(self, current, T):
        """
        Faraday Efficiency: f2 * I^2 / (I^2 + f1)
        """
        if current < 1e-3:
            return 0.0
            
        f1 = 50.0 + 2.5 * T
        f2 = 0.92 - 6.25e-6 * T
        
        eta = f2 * (current**2) / (current**2 + f1)
        return np.clip(eta, 0.0, 1.0)

    def _calculate_thermal_dynamics(self, T_stack, T_cw_out, current, v_cell, lye_flow, cw_flow, q_recovery=0.0):
        """
        Updates Stack Temperature (T_stack) and Cooling Water Outlet Temperature (T_cw_out).
        Uses sub-stepping to ensure numerical stability.
        """
        # Thermal parameters
        C_th = self.C_th
        C_cw = self.C_cw
        kA_hx = self.kA_hx
        T_cw_in = self.T_cw_in
        c_cw = self.c_cw
        rho_cw = self.rho_cw
        
        # Inputs (assumed constant over step)
        eta = self.get_faraday_efficiency(current, T_stack)
        q_gen = self.n_cells * current * (v_cell - eta * self.U_tn)
        
        # Sub-stepping
        n_substeps = 60 # 1 second steps
        dt_sub = self.dt / n_substeps
        
        T_s = T_stack
        T_c = T_cw_out
        
        for _ in range(n_substeps):
            # Heat Exchanger
            Q_hx = kA_hx * (T_s - T_c)
            
            # CW Dynamics
            m_dot_cw = cw_flow * rho_cw
            dQ_cw_flow = m_dot_cw * c_cw * (T_cw_in - T_c)
            dT_c_dt = (dQ_cw_flow + Q_hx) / C_cw
            
            # Stack Dynamics
            # Lye Cooling (via HX)
            if lye_flow < 1e-6:
                Q_lye_cooling = 0.0
            else:
                Q_lye_cooling = Q_hx
                
            q_natural = self.h_A * (T_s - self.T_amb)
            q_loss = q_natural + Q_lye_cooling
            
            dT_s_dt = (q_gen - q_loss - q_recovery) / C_th
            
            # Update
            T_s += dT_s_dt * dt_sub
            T_c += dT_c_dt * dt_sub
            
        return T_s, T_c

    def _check_safety_status(self, current, T_stack):
        """Calculate gas purity and check safety constraints (Temp, Impurity)."""
        # 1. Impurity Check
        h2_rate = self.calculate_h2_production(current)
        o2_rate = h2_rate / 2.0
        
        total_gas_o2_side = o2_rate + self.gas_crossover_rate
        if total_gas_o2_side > 0:
            impurity_conc = (self.gas_crossover_rate / total_gas_o2_side) * 100.0
        else:
            impurity_conc = 100.0
            
        alarm = False
        shutdown = False
        
        # Impurity Logic
        if impurity_conc > self.impurity_limit:
            alarm = True
            shutdown = True
            
        # 2. Temperature Check
        if T_stack > self.T_max_safe:
            alarm = True
        if T_stack > self.T_shutdown:
            shutdown = True
            
        return {
            "h2_rate": h2_rate,
            "impurity_conc": impurity_conc,
            "alarm": alarm,
            "shutdown": shutdown
        }

    def step(self, action):
        """
        action: [current] or [current, lye_flow, cw_flow]
        """
        # Parse Action
        if isinstance(action, (list, np.ndarray)):
            current = float(action[0])
            if len(action) > 2:
                lye_flow = float(action[1])
                cw_flow = float(action[2])
            elif len(action) > 1:
                lye_flow = float(action[1])
                cw_flow = 0.04 # Default if not provided
            else:
                lye_flow = 0.04
                cw_flow = 0.04
        else:
            current = float(action)
            lye_flow = 0.04
            cw_flow = 0.04
            
        current = max(0.0, current) # Current cannot be negative
        lye_flow = max(0.0, lye_flow)
        cw_flow = max(0.0, cw_flow)
        
        T_stack = self.state[0]
        T_cw_out = self.state[1] if len(self.state) > 1 else self.T_cw_in
        
        v_cell = self.get_voltage(current, T_stack)
        
        # Heat Pump Control & Dynamics
        self.hp.is_on = (T_stack > self.hp_setpoint)
        q_source, q_sink, w_comp = self.hp.step(T_source=T_stack, T_sink=self.storage.T, dt=self.dt)
        q_recovery_power = q_source / self.dt # Convert J to W
        
        # Thermal Storage Dynamics
        self.storage.step(Q_in=q_sink, Q_load=0.0, dt=self.dt)
        
        # Thermal Dynamics (Stack + HX)
        T_stack_new, T_cw_out_new = self._calculate_thermal_dynamics(
            T_stack, T_cw_out, current, v_cell, lye_flow, cw_flow, q_recovery=q_recovery_power
        )
        
        # Impurity & Safety Check
        safety_status = self._check_safety_status(current, T_stack_new)
        
        # Update State
        self.state = np.array([T_stack_new, T_cw_out_new])
        self.current_time += self.dt
        
        # Output info
        info = {
            "time": self.current_time,
            "T": T_stack_new,
            "T_cw_out": T_cw_out_new,
            "I": current,
            "V_cell": v_cell,
            "Power": v_cell * current * self.n_cells,
            "H2_rate": safety_status["h2_rate"],
            "Impurity": safety_status["impurity_conc"],
            "Alarm": safety_status["alarm"],
            "Shutdown": safety_status["shutdown"],
            # Thermal System Info
            "T_storage": self.storage.T,
            "HP_Power_Elec": w_comp / self.dt, # Watts
            "HP_Heat_Recovered": q_recovery_power, # Watts
            "HP_COP": (q_sink/w_comp) if w_comp > 0 else 0.0,
            "HP_Status": 1.0 if self.hp.is_on else 0.0
        }
        
        return self.state, info

    def calculate_h2_production(self, current):
        # Faraday's law
        # Mol/s = (N * I) / (z * F)
        # z = 2 for H2
        F = 96485.33
        mols_per_s = (self.n_cells * current) / (2 * F)
        # Convert to Nm^3/h or kg/h if needed, returning mol/s for now
        return mols_per_s

    def get_state(self):
        return self.state
