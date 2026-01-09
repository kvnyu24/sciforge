"""
Custom exception hierarchy for SciForge.

This module defines a comprehensive set of exceptions for error handling
throughout the library, enabling precise error identification and recovery.
"""

from typing import Any, Optional, Tuple


class SciForgeError(Exception):
    """
    Base exception for all SciForge errors.

    All custom exceptions in SciForge inherit from this class,
    allowing users to catch all library-specific errors with a single except clause.

    Attributes:
        message: Human-readable error description
        details: Optional additional context about the error
    """

    def __init__(self, message: str, details: Optional[Any] = None):
        self.message = message
        self.details = details
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (details: {self.details})"
        return self.message


# =============================================================================
# Validation Errors
# =============================================================================


class ValidationError(SciForgeError):
    """
    Raised when input validation fails.

    This exception indicates that provided parameters do not meet
    the required constraints (type, range, shape, etc.).

    Examples:
        >>> raise ValidationError("mass", -1.0, "must be positive")
        ValidationError: Invalid value for 'mass': -1.0 (must be positive)
    """

    def __init__(
        self,
        param_name: str,
        value: Any,
        constraint: str,
        details: Optional[Any] = None,
    ):
        self.param_name = param_name
        self.value = value
        self.constraint = constraint
        message = f"Invalid value for '{param_name}': {value} ({constraint})"
        super().__init__(message, details)


class DimensionError(ValidationError):
    """
    Raised when array dimensions don't match expected shape.

    Examples:
        >>> raise DimensionError("position", (2,), (3,))
        DimensionError: Invalid value for 'position': shape (2,) (expected shape (3,))
    """

    def __init__(
        self,
        param_name: str,
        actual_shape: Tuple[int, ...],
        expected_shape: Tuple[int, ...],
    ):
        constraint = f"expected shape {expected_shape}"
        super().__init__(param_name, f"shape {actual_shape}", constraint)
        self.actual_shape = actual_shape
        self.expected_shape = expected_shape


class BoundsError(ValidationError):
    """
    Raised when a value is outside allowed bounds.

    Examples:
        >>> raise BoundsError("temperature", -10.0, (0.0, float('inf')))
        BoundsError: Invalid value for 'temperature': -10.0 (must be in range [0.0, inf])
    """

    def __init__(
        self, param_name: str, value: float, bounds: Tuple[float, float]
    ):
        self.bounds = bounds
        constraint = f"must be in range [{bounds[0]}, {bounds[1]}]"
        super().__init__(param_name, value, constraint)


# =============================================================================
# Numerical Errors
# =============================================================================


class NumericalError(SciForgeError):
    """
    Base exception for numerical computation errors.

    This includes convergence failures, numerical instabilities,
    and other computation-related issues.
    """

    pass


class ConvergenceError(NumericalError):
    """
    Raised when an iterative algorithm fails to converge.

    Attributes:
        iterations: Number of iterations performed before failure
        tolerance: The target tolerance that was not achieved
        final_error: The error at the final iteration
    """

    def __init__(
        self,
        method_name: str,
        iterations: int,
        tolerance: float,
        final_error: Optional[float] = None,
    ):
        self.iterations = iterations
        self.tolerance = tolerance
        self.final_error = final_error

        message = (
            f"{method_name} failed to converge after {iterations} iterations "
            f"(tolerance: {tolerance})"
        )
        if final_error is not None:
            message += f", final error: {final_error}"

        super().__init__(message)


class InstabilityError(NumericalError):
    """
    Raised when numerical instability is detected.

    This can occur when step sizes are too large, when dealing with
    stiff equations, or when floating point limits are exceeded.
    """

    def __init__(self, message: str, step: Optional[int] = None):
        self.step = step
        if step is not None:
            message = f"Instability at step {step}: {message}"
        super().__init__(message)


class SingularityError(NumericalError):
    """
    Raised when a mathematical singularity is encountered.

    Examples include division by zero, evaluation at poles,
    or singular matrices.
    """

    pass


# =============================================================================
# Physics Errors
# =============================================================================


class PhysicsError(SciForgeError):
    """
    Base exception for physics-related constraint violations.

    This is raised when physical laws or constraints would be violated
    by the requested operation.
    """

    pass


class EnergyConservationError(PhysicsError):
    """
    Raised when energy conservation is violated beyond acceptable tolerance.

    This typically indicates a bug in the simulation or numerical issues.
    """

    def __init__(
        self,
        initial_energy: float,
        current_energy: float,
        tolerance: float = 1e-6,
    ):
        self.initial_energy = initial_energy
        self.current_energy = current_energy
        self.tolerance = tolerance

        relative_error = abs(current_energy - initial_energy) / abs(initial_energy)
        message = (
            f"Energy conservation violated: initial={initial_energy:.6e}, "
            f"current={current_energy:.6e}, relative error={relative_error:.2e} "
            f"(tolerance: {tolerance})"
        )
        super().__init__(message)


class CausalityError(PhysicsError):
    """
    Raised when a relativistic causality constraint would be violated.

    Examples include velocities exceeding the speed of light.
    """

    def __init__(self, velocity: float, c: float = 299792458.0):
        self.velocity = velocity
        self.c = c
        message = f"Velocity {velocity:.6e} m/s exceeds speed of light ({c:.6e} m/s)"
        super().__init__(message)


class ThermodynamicError(PhysicsError):
    """
    Raised when thermodynamic constraints are violated.

    Examples include negative absolute temperatures or entropy decreases
    in isolated systems.
    """

    pass


# =============================================================================
# Configuration Errors
# =============================================================================


class ConfigurationError(SciForgeError):
    """
    Raised when there is an error in system configuration.

    This includes incompatible parameter combinations or
    missing required configuration.
    """

    pass


class DependencyError(ConfigurationError):
    """
    Raised when an optional dependency is required but not available.
    """

    def __init__(self, package_name: str, feature: str):
        self.package_name = package_name
        self.feature = feature
        message = (
            f"Optional dependency '{package_name}' is required for {feature}. "
            f"Install with: pip install sciforge[performance]"
        )
        super().__init__(message)