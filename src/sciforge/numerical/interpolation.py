"""
Interpolation methods for fitting curves through discrete data points
"""

import numpy as np
from typing import Sequence, Callable, Union, Optional

def linear_interp(x: Sequence[float], y: Sequence[float]) -> Callable[[float], float]:
    """
    Linear interpolation between data points.
    
    Args:
        x: Sequence of x coordinates
        y: Sequence of y coordinates
    
    Returns:
        Callable that interpolates between points
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    if len(x) != len(y):
        raise ValueError("x and y arrays must have same length")
    if len(x) < 2:
        raise ValueError("Need at least 2 points for interpolation")
    if not np.all(np.diff(x) > 0):
        raise ValueError("x values must be strictly increasing")
        
    def interpolate(x_new: float) -> float:
        if x_new < x[0] or x_new > x[-1]:
            raise ValueError("x_new must be within bounds of original data")
            
        # Find index of lower bound
        idx = np.searchsorted(x, x_new) - 1
        idx = np.clip(idx, 0, len(x)-2)
        
        # Linear interpolation formula
        t = (x_new - x[idx]) / (x[idx+1] - x[idx])
        return (1-t) * y[idx] + t * y[idx+1]
        
    return interpolate

def cubic_spline(x: Sequence[float], y: Sequence[float], 
                 bc_type: str = 'natural') -> Callable[[float], float]:
    """
    Cubic spline interpolation between data points.
    
    Args:
        x: Sequence of x coordinates
        y: Sequence of y coordinates
        bc_type: Boundary condition type ('natural' or 'clamped')
    
    Returns:
        Callable that interpolates between points using cubic splines
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    if len(x) != len(y):
        raise ValueError("x and y arrays must have same length")
    if len(x) < 3:
        raise ValueError("Need at least 3 points for cubic spline")
    if not np.all(np.diff(x) > 0):
        raise ValueError("x values must be strictly increasing")
        
    n = len(x)
    h = np.diff(x)
    
    # Build tridiagonal system
    A = np.zeros((n, n))
    b = np.zeros(n)
    
    # Interior points
    for i in range(1, n-1):
        A[i, i-1] = h[i-1]
        A[i, i] = 2*(h[i-1] + h[i])
        A[i, i+1] = h[i]
        b[i] = 3*((y[i+1] - y[i])/h[i] - (y[i] - y[i-1])/h[i-1])
    
    # Boundary conditions
    if bc_type == 'natural':
        A[0,0] = A[-1,-1] = 1
    elif bc_type == 'clamped':
        A[0, :2] = [2*h[0], h[0]]
        A[-1, -2:] = [h[-1], 2*h[-1]]
        b[0] = 3*((y[1] - y[0])/h[0])
        b[-1] = 3*((y[-1] - y[-2])/h[-1])
    else:
        raise ValueError("bc_type must be 'natural' or 'clamped'")
        
    # Solve for second derivatives
    c = np.linalg.solve(A, b)
    
    def interpolate(x_new: float) -> float:
        if x_new < x[0] or x_new > x[-1]:
            raise ValueError("x_new must be within bounds of original data")
            
        # Find interval
        idx = np.searchsorted(x, x_new) - 1
        idx = np.clip(idx, 0, len(x)-2)
        
        # Compute cubic spline coefficients
        dx = x_new - x[idx]
        h_i = h[idx]
        
        a = y[idx]
        b = (y[idx+1] - y[idx])/h_i - h_i*c[idx]/3 - h_i*c[idx+1]/6
        c_i = c[idx]/2
        d = (c[idx+1] - c[idx])/(6*h_i)
        
        # Evaluate cubic polynomial
        return a + b*dx + c_i*dx**2 + d*dx**3
        
    return interpolate
