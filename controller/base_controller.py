from abc import ABC, abstractmethod

class BaseController(ABC):
    """
    Abstract base class for controllers.
    """
    
    def __init__(self, dt):
        self.dt = dt

    @abstractmethod
    def get_action(self, state, setpoint):
        """
        Calculate control action based on current state and setpoint.
        
        Args:
            state: Current system state.
            setpoint: Target state or reference.
            
        Returns:
            action: Control input to apply.
        """
        pass
    
    def reset(self):
        """Reset controller internal state if any."""
        pass
