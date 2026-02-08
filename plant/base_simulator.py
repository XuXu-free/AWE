from abc import ABC, abstractmethod
import numpy as np

class BaseSimulator(ABC):
    """
    Abstract base class for system simulation.
    """
    
    def __init__(self, dt):
        self.dt = dt
        self.current_time = 0.0
        self.state = None

    @abstractmethod
    def reset(self, initial_state=None):
        """Reset the simulation to initial state."""
        pass

    @abstractmethod
    def step(self, action):
        """
        Advance the simulation by one time step.
        
        Args:
            action: The control input applied to the system.
            
        Returns:
            observation: The new state or observation of the system.
        """
        pass

    @abstractmethod
    def get_state(self):
        """Return the current state."""
        pass
