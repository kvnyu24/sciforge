"""
Core module containing shared utilities, base classes, and constants.

This module provides the foundational components used throughout SciForge:
- Base classes for physics systems, solvers, and stochastic processes
- Physical and mathematical constants
- Validation utilities
- Custom exception hierarchy
"""

from .constants import CONSTANTS
from .base import BaseClass, BaseSolver, BaseProcess, ArrayType
from .exceptions import (
    SciForgeError,
    ValidationError,
    DimensionError,
    BoundsError,
    NumericalError,
    ConvergenceError,
    InstabilityError,
    SingularityError,
    PhysicsError,
    EnergyConservationError,
    CausalityError,
    ThermodynamicError,
    ConfigurationError,
    DependencyError,
)
from .utils import (
    validate_array,
    validate_vector,
    validate_positive,
    validate_non_negative,
    validate_finite,
    validate_bounds,
    validate_probability,
    validate_callable,
    validate_type,
    normalize_vector,
    safe_divide,
    ensure_2d,
)
from .integrators import (
    euler_step,
    rk2_step,
    rk4_step,
    rk45_step,
    integrate,
    integrate_adaptive,
    DynamicsIntegrator,
)

__all__ = [
    # Constants
    "CONSTANTS",
    # Base classes
    "BaseClass",
    "BaseSolver",
    "BaseProcess",
    "ArrayType",
    # Exceptions
    "SciForgeError",
    "ValidationError",
    "DimensionError",
    "BoundsError",
    "NumericalError",
    "ConvergenceError",
    "InstabilityError",
    "SingularityError",
    "PhysicsError",
    "EnergyConservationError",
    "CausalityError",
    "ThermodynamicError",
    "ConfigurationError",
    "DependencyError",
    # Validation utilities
    "validate_array",
    "validate_vector",
    "validate_positive",
    "validate_non_negative",
    "validate_finite",
    "validate_bounds",
    "validate_probability",
    "validate_callable",
    "validate_type",
    # Array utilities
    "normalize_vector",
    "safe_divide",
    "ensure_2d",
    # Integrators
    "euler_step",
    "rk2_step",
    "rk4_step",
    "rk45_step",
    "integrate",
    "integrate_adaptive",
    "DynamicsIntegrator",
]