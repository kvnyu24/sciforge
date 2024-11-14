"""
Utility functions used throughout the library
"""

import numpy as np
from typing import Union, Tuple
from numpy.typing import ArrayLike
from .base import ArrayType

def validate_array(arr: ArrayType, dim: int = 1) -> np.ndarray:
    """
    Validate and convert array-like input to numpy array
    
    Args:
        arr: Input array-like object
        dim: Expected number of dimensions
        
    Returns:
        Validated numpy array
    """
    arr = np.asarray(arr)
    if arr.ndim != dim:
        raise ValueError(f"Expected {dim}D array, got {arr.ndim}D")
    return arr

def validate_bounds(value: float, bounds: Tuple[float, float], 
                   param_name: str = "Value") -> None:
    """
    Validate that a value falls within specified bounds
    
    Args:
        value: Value to check
        bounds: Tuple of (lower_bound, upper_bound)
        param_name: Name of parameter for error message
    """
    lower, upper = bounds
    if not lower <= value <= upper:
        raise ValueError(
            f"{param_name} must be between {lower} and {upper}, got {value}"
        ) 