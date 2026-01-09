"""
Base classes and types used throughout the library.

This module defines the foundational class hierarchy for SciForge:
- BaseClass: Root class with history tracking
- BaseSolver: Base for numerical solvers
- BaseProcess: Base for stochastic processes
"""

import numpy as np
from typing import Union, TypeVar, Any, Callable, Tuple, Optional, Dict, List
from numpy.typing import ArrayLike


# Type alias for array-like inputs
ArrayType = Union[list, tuple, np.ndarray, ArrayLike]


class BaseClass:
    """
    Base class with common functionality for all SciForge classes.

    Provides:
    - History tracking for state evolution
    - Serialization support
    - Common utility methods

    Attributes:
        _history: Dictionary storing time series of states
        _history_fields: List of field names to track (set by subclasses)
    """

    # Subclasses can override this to specify which fields to track
    _history_fields: List[str] = ["time", "state"]

    def __init__(self):
        self._history: Dict[str, List[Any]] = {}
        self._init_history()

    def _init_history(self) -> None:
        """Initialize empty history containers for all tracked fields."""
        for field in self._history_fields:
            self._history[field] = []

    def save_state(self, time: float, state: Any) -> None:
        """
        Save current state to history.

        Args:
            time: Current simulation time
            state: Current state (can be any serializable object)
        """
        self._history["time"].append(time)
        self._history["state"].append(state)

    def record_state(self, **kwargs) -> None:
        """
        Record multiple state variables to history.

        This is a more flexible alternative to save_state that allows
        recording multiple named quantities.

        Args:
            **kwargs: Named values to record (must match _history_fields)

        Examples:
            >>> system.record_state(time=0.1, position=[1,2,3], velocity=[0,0,1])
        """
        for key, value in kwargs.items():
            if key in self._history:
                # Make a copy if it's an array to avoid reference issues
                if hasattr(value, "copy"):
                    self._history[key].append(value.copy())
                else:
                    self._history[key].append(value)

    def clear_history(self) -> None:
        """Clear all stored history."""
        self._init_history()

    def get_history(self, field: Optional[str] = None) -> Union[Dict, np.ndarray]:
        """
        Return history data.

        Args:
            field: If specified, return only this field as a numpy array.
                   If None, return the entire history dictionary.

        Returns:
            History dictionary or numpy array for specified field

        Examples:
            >>> history = system.get_history()  # Full history dict
            >>> positions = system.get_history("position")  # Just positions
        """
        if field is not None:
            if field not in self._history:
                raise KeyError(f"Unknown history field: {field}")
            return np.array(self._history[field])
        return self._history

    def history_length(self) -> int:
        """Return the number of recorded time steps."""
        return len(self._history.get("time", []))

    @property
    def has_history(self) -> bool:
        """Check if any history has been recorded."""
        return self.history_length() > 0


class BaseSolver(BaseClass):
    """
    Base class for all numerical solvers.

    Provides common interface for:
    - Input validation
    - Solution storage
    - Error estimation

    Attributes:
        store_history: Whether to store solution history
    """

    def __init__(self, store_history: bool = True):
        super().__init__()
        self.store_history = store_history

    def solve(self, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the problem. Must be implemented by subclasses.

        Returns:
            Tuple of (time points, solution values)
        """
        raise NotImplementedError("Subclasses must implement solve()")

    def validate_inputs(self, *args, **kwargs) -> None:
        """
        Validate solver inputs.

        Should raise ValidationError for invalid inputs.
        """
        pass  # Default: no validation

    def estimate_error(self) -> float:
        """
        Estimate numerical error of the solution.

        Returns:
            Estimated error (implementation-specific)

        Raises:
            NotImplementedError: If error estimation is not available
        """
        raise NotImplementedError("Error estimation not available for this solver")


class BaseProcess(BaseClass):
    """
    Base class for stochastic processes.

    Provides common interface for:
    - Random number generation with reproducible seeds
    - Process simulation
    - Theoretical statistics

    Attributes:
        rng: NumPy random number generator
        seed: Random seed (if provided)
    """

    _history_fields = ["time", "value"]

    def __init__(self, seed: Optional[int] = None):
        super().__init__()
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def reset_rng(self, seed: Optional[int] = None) -> None:
        """
        Reset the random number generator.

        Args:
            seed: New seed. If None, uses the original seed.
        """
        self.rng = np.random.default_rng(seed if seed is not None else self.seed)

    def simulate(
        self,
        params: Tuple[float, ...],
        T: float,
        N: int,
        initial_state: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate the stochastic process.

        Args:
            params: Process parameters (process-specific)
            T: Final time
            N: Number of time steps
            initial_state: Initial state (optional, default depends on process)

        Returns:
            Tuple of (time points, process values)
        """
        raise NotImplementedError("Subclasses must implement simulate()")

    def sample_increment(self, dt: float) -> float:
        """
        Generate a random increment for the process.

        Args:
            dt: Time step

        Returns:
            Random increment value
        """
        raise NotImplementedError("Subclasses must implement sample_increment()")

    def mean(self, t: float, **kwargs) -> float:
        """
        Calculate theoretical mean at time t.

        Args:
            t: Time point
            **kwargs: Additional parameters (process-specific)

        Returns:
            Expected value at time t
        """
        raise NotImplementedError("Analytical mean not available")

    def variance(self, t: float, **kwargs) -> float:
        """
        Calculate theoretical variance at time t.

        Args:
            t: Time point
            **kwargs: Additional parameters (process-specific)

        Returns:
            Variance at time t
        """
        raise NotImplementedError("Analytical variance not available")

    def paths(
        self,
        params: Tuple[float, ...],
        T: float,
        N: int,
        n_paths: int,
        initial_state: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate multiple paths of the stochastic process.

        Args:
            params: Process parameters
            T: Final time
            N: Number of time steps
            n_paths: Number of paths to simulate
            initial_state: Initial state for all paths

        Returns:
            Tuple of (time points, paths array of shape (n_paths, N+1))
        """
        paths_list = []
        times = None

        for _ in range(n_paths):
            t, path = self.simulate(params, T, N, initial_state)
            if times is None:
                times = t
            paths_list.append(path)

        return times, np.array(paths_list)