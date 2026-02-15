from abc import ABC, abstractmethod

class BaseController(ABC):
    """
    Abstract base class for controllers.
    """
    
    def __init__(self, dt):
        self.dt = dt

    @abstractmethod
    def get_action(self, state, P_ref, T_ref=None):
        """
        Calculate control action based on current state and setpoint.
        
        Args:
            state: Current system state.
            P_ref: Power reference (scalar or vector).
            T_ref: Temperature reference (optional).
            
        Returns:
            tuple: (I_cmd, v_lye_cmd, v_c_cmd)
        """
        pass
    
    def reset(self):
        """Reset controller internal state if any."""
        pass
