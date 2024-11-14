"""
Base classes and types used throughout the library
"""

import numpy as np
from typing import Union, TypeVar, Any, Callable, Tuple, Optional
from numpy.typing import ArrayLike

# Type alias for array-like inputs
ArrayType = Union[list, tuple, np.ndarray, ArrayLike]

class BaseClass:
    """Base class with common functionality for all SciForge classes"""
    
    def __init__(self):
        self._history = {'time': [], 'state': []}
        
    def save_state(self, time: float, state: Any):
        """Save current state to history"""
        self._history['time'].append(time)
        self._history['state'].append(state)
        
    def clear_history(self):
        """Clear stored history"""
        self._history = {'time': [], 'state': []}
        
    def get_history(self):
        """Return history dictionary"""
        return self._history

class BaseSolver(BaseClass):
    """Base class for all numerical solvers"""
    
    def __init__(self, store_history: bool = True):
        super().__init__()
        self.store_history = store_history
        
    def solve(self, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the problem. Must be implemented by subclasses.
        
        Returns:
            Tuple of (time points, solution values)
        """
        raise NotImplementedError
        
    def validate_inputs(self, *args, **kwargs):
        """Validate solver inputs. Should be implemented by subclasses."""
        raise NotImplementedError
        
    def estimate_error(self) -> float:
        """
        Estimate numerical error of the solution.
        Should be implemented by subclasses if error estimation is possible.
        """
        raise NotImplementedError

class BaseProcess(BaseClass):
    """Base class for stochastic processes"""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        
    def simulate(self, 
                params: Tuple[float, ...],
                T: float,
                N: int,
                initial_state: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate the stochastic process.
        
        Args:
            params: Process parameters
            T: Final time
            N: Number of time steps
            initial_state: Initial state (optional)
            
        Returns:
            Tuple of (time points, process values)
        """
        raise NotImplementedError
        
    def sample_increment(self, dt: float) -> float:
        """
        Generate a random increment for the process.
        Should be implemented by subclasses.
        
        Args:
            dt: Time step
            
        Returns:
            Random increment value
        """
        raise NotImplementedError
        
    def mean(self, t: float) -> float:
        """
        Calculate theoretical mean at time t.
        Should be implemented by subclasses if analytical solution exists.
        """
        raise NotImplementedError
        
    def variance(self, t: float) -> float:
        """
        Calculate theoretical variance at time t.
        Should be implemented by subclasses if analytical solution exists.
        """
        raise NotImplementedError