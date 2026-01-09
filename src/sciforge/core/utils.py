"""
Utility functions used throughout the library.

This module provides validation helpers, array utilities, and common
operations used across all SciForge modules.
"""

import numpy as np
from typing import Union, Tuple, Optional, Any
from numpy.typing import ArrayLike

from .base import ArrayType
from .exceptions import (
    ValidationError,
    DimensionError,
    BoundsError,
)


# =============================================================================
# Array Validation
# =============================================================================


def validate_array(
    arr: ArrayType,
    dim: Optional[int] = None,
    expected_shape: Optional[Tuple[int, ...]] = None,
    name: str = "array",
    dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    """
    Validate and convert array-like input to numpy array.

    Args:
        arr: Input array-like object
        dim: Expected number of dimensions (optional)
        expected_shape: Expected shape tuple (optional, more specific than dim)
        name: Parameter name for error messages
        dtype: Expected or target dtype (optional)

    Returns:
        Validated numpy array

    Raises:
        ValidationError: If input cannot be converted to array
        DimensionError: If dimensions don't match expectations

    Examples:
        >>> validate_array([1, 2, 3], dim=1)
        array([1, 2, 3])

        >>> validate_array([1, 2], expected_shape=(3,), name="position")
        DimensionError: Invalid value for 'position': shape (2,) (expected shape (3,))
    """
    try:
        arr = np.asarray(arr, dtype=dtype)
    except (ValueError, TypeError) as e:
        raise ValidationError(name, arr, f"cannot convert to array: {e}")

    if expected_shape is not None:
        if arr.shape != expected_shape:
            raise DimensionError(name, arr.shape, expected_shape)
    elif dim is not None:
        if arr.ndim != dim:
            raise DimensionError(
                name, arr.shape, tuple([-1] * dim)  # -1 indicates any size
            )

    return arr


def validate_vector(
    vec: ArrayType,
    size: int = 3,
    name: str = "vector",
) -> np.ndarray:
    """
    Validate a vector (1D array) of specific size.

    Args:
        vec: Input vector
        size: Expected vector size (default: 3 for 3D)
        name: Parameter name for error messages

    Returns:
        Validated numpy array of shape (size,)

    Examples:
        >>> validate_vector([1, 0, 0])
        array([1, 0, 0])

        >>> validate_vector([1, 0], size=3, name="velocity")
        DimensionError: Invalid value for 'velocity': shape (2,) (expected shape (3,))
    """
    return validate_array(vec, expected_shape=(size,), name=name)


# =============================================================================
# Scalar Validation
# =============================================================================


def validate_positive(
    value: float,
    name: str = "value",
    allow_zero: bool = False,
) -> float:
    """
    Validate that a value is positive.

    Args:
        value: Value to validate
        name: Parameter name for error messages
        allow_zero: If True, zero is allowed

    Returns:
        The validated value

    Raises:
        ValidationError: If value is not positive (or non-negative if allow_zero)

    Examples:
        >>> validate_positive(1.5, "mass")
        1.5

        >>> validate_positive(-1.0, "mass")
        ValidationError: Invalid value for 'mass': -1.0 (must be positive)
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(name, value, "must be a number")

    if not np.isfinite(value):
        raise ValidationError(name, value, "must be finite")

    if allow_zero:
        if value < 0:
            raise ValidationError(name, value, "must be non-negative")
    else:
        if value <= 0:
            raise ValidationError(name, value, "must be positive")

    return float(value)


def validate_non_negative(value: float, name: str = "value") -> float:
    """
    Validate that a value is non-negative (>= 0).

    Args:
        value: Value to validate
        name: Parameter name for error messages

    Returns:
        The validated value

    Raises:
        ValidationError: If value is negative
    """
    return validate_positive(value, name, allow_zero=True)


def validate_finite(value: float, name: str = "value") -> float:
    """
    Validate that a value is finite (not inf or nan).

    Args:
        value: Value to validate
        name: Parameter name for error messages

    Returns:
        The validated value

    Raises:
        ValidationError: If value is not finite
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(name, value, "must be a number")

    if not np.isfinite(value):
        raise ValidationError(name, value, "must be finite")

    return float(value)


def validate_bounds(
    value: float,
    bounds: Tuple[float, float],
    name: str = "value",
    inclusive: Tuple[bool, bool] = (True, True),
) -> float:
    """
    Validate that a value falls within specified bounds.

    Args:
        value: Value to check
        bounds: Tuple of (lower_bound, upper_bound)
        name: Parameter name for error messages
        inclusive: Tuple of (lower_inclusive, upper_inclusive)

    Returns:
        The validated value

    Raises:
        BoundsError: If value is outside bounds

    Examples:
        >>> validate_bounds(0.5, (0, 1), "probability")
        0.5

        >>> validate_bounds(1.5, (0, 1), "probability")
        BoundsError: Invalid value for 'probability': 1.5 (must be in range [0, 1])
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(name, value, "must be a number")

    lower, upper = bounds
    lower_ok = value >= lower if inclusive[0] else value > lower
    upper_ok = value <= upper if inclusive[1] else value < upper

    if not (lower_ok and upper_ok):
        raise BoundsError(name, value, bounds)

    return float(value)


def validate_probability(value: float, name: str = "probability") -> float:
    """
    Validate that a value is a valid probability [0, 1].

    Args:
        value: Value to validate
        name: Parameter name for error messages

    Returns:
        The validated value

    Raises:
        BoundsError: If value is not in [0, 1]
    """
    return validate_bounds(value, (0.0, 1.0), name)


# =============================================================================
# Type Validation
# =============================================================================


def validate_callable(
    func: Any,
    name: str = "function",
) -> callable:
    """
    Validate that an object is callable.

    Args:
        func: Object to validate
        name: Parameter name for error messages

    Returns:
        The validated callable

    Raises:
        ValidationError: If object is not callable
    """
    if not callable(func):
        raise ValidationError(name, type(func).__name__, "must be callable")
    return func


def validate_type(
    value: Any,
    expected_type: type,
    name: str = "value",
) -> Any:
    """
    Validate that a value is of the expected type.

    Args:
        value: Value to validate
        expected_type: Expected type or tuple of types
        name: Parameter name for error messages

    Returns:
        The validated value

    Raises:
        ValidationError: If value is not of expected type
    """
    if not isinstance(value, expected_type):
        type_name = (
            expected_type.__name__
            if isinstance(expected_type, type)
            else str(expected_type)
        )
        raise ValidationError(
            name, type(value).__name__, f"must be of type {type_name}"
        )
    return value


# =============================================================================
# Array Utilities
# =============================================================================


def normalize_vector(v: ArrayLike) -> np.ndarray:
    """
    Normalize a vector to unit length.

    Args:
        v: Input vector

    Returns:
        Unit vector in the same direction

    Raises:
        ValueError: If vector has zero length
    """
    v = np.asarray(v)
    norm = np.linalg.norm(v)
    if norm < 1e-15:
        raise ValueError("Cannot normalize zero vector")
    return v / norm


def safe_divide(
    numerator: ArrayLike,
    denominator: ArrayLike,
    default: float = 0.0,
) -> np.ndarray:
    """
    Perform division with protection against divide-by-zero.

    Args:
        numerator: Numerator array
        denominator: Denominator array
        default: Value to use where denominator is zero

    Returns:
        Result of division with default where denominator is zero
    """
    numerator = np.asarray(numerator)
    denominator = np.asarray(denominator)
    return np.where(
        np.abs(denominator) > 1e-15,
        numerator / denominator,
        default,
    )


def ensure_2d(arr: ArrayLike) -> np.ndarray:
    """
    Ensure array is 2D, expanding dimensions if necessary.

    Args:
        arr: Input array

    Returns:
        2D array (row vector if 1D input)
    """
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr
