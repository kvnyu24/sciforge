"""
Optimization algorithms for finding minima/maxima of functions
"""

import numpy as np
from typing import Callable, Tuple, Optional


def newton_optimize(f: Callable[[float], float],
                   df: Callable[[float], float], 
                   x0: float,
                   tol: float = 1e-6,
                   max_iter: int = 100) -> Tuple[float, float]:
    """
    Newton's method for finding local minimum/maximum of a function.
    
    Args:
        f: Function to optimize
        df: First derivative of function
        x0: Initial guess
        tol: Convergence tolerance
        max_iter: Maximum iterations
        
    Returns:
        Tuple of (optimal x value, optimal function value)
    """
    x = x0
    
    for i in range(max_iter):
        dx = df(x)
        if abs(dx) < tol:
            break
            
        # Update using Newton step
        x = x - dx
        
        if i == max_iter - 1:
            raise RuntimeError("Failed to converge within maximum iterations")
            
    return x, f(x)


def gradient_descent(f: Callable[[np.ndarray], float],
                    grad: Callable[[np.ndarray], np.ndarray],
                    x0: np.ndarray,
                    learning_rate: float = 0.1,
                    tol: float = 1e-6, 
                    max_iter: int = 1000) -> Tuple[np.ndarray, float]:
    """
    Gradient descent optimization for multidimensional functions.
    
    Args:
        f: Function to optimize
        grad: Gradient function
        x0: Initial guess as numpy array
        learning_rate: Step size for descent
        tol: Convergence tolerance
        max_iter: Maximum iterations
        
    Returns:
        Tuple of (optimal x vector, optimal function value)
    """
    x = np.asarray(x0, dtype=float)
    
    for i in range(max_iter):
        g = grad(x)
        if np.all(np.abs(g) < tol):
            break
            
        # Update step
        x = x - learning_rate * g
        
        if i == max_iter - 1:
            raise RuntimeError("Failed to converge within maximum iterations")
            
    return x, f(x)


def nelder_mead(f: Callable[[np.ndarray], float],
                x0: np.ndarray,
                step: float = 0.1,
                tol: float = 1e-6,
                max_iter: int = 1000) -> Tuple[np.ndarray, float]:
    """
    Nelder-Mead simplex optimization method.
    Does not require gradient information.
    
    Args:
        f: Function to optimize
        x0: Initial guess as numpy array
        step: Initial step size for simplex
        tol: Convergence tolerance
        max_iter: Maximum iterations
        
    Returns:
        Tuple of (optimal x vector, optimal function value)
    """
    # Parameters
    alpha = 1.0  # reflection
    beta = 0.5   # contraction
    gamma = 2.0  # expansion
    
    dim = len(x0)
    
    # Initialize simplex
    simplex = np.zeros((dim + 1, dim))
    simplex[0] = x0
    for i in range(dim):
        vertex = x0.copy()
        vertex[i] += step
        simplex[i + 1] = vertex
        
    # Evaluate function at all vertices
    values = np.array([f(x) for x in simplex])
    
    for iter in range(max_iter):
        # Order vertices
        order = np.argsort(values)
        values = values[order]
        simplex = simplex[order]
        
        # Check convergence
        if np.max(np.abs(values[1:] - values[0])) < tol:
            break
            
        # Centroid excluding worst point
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        reflected = centroid + alpha * (centroid - simplex[-1])
        reflected_val = f(reflected)
        
        if values[0] <= reflected_val < values[-2]:
            simplex[-1] = reflected
            values[-1] = reflected_val
            continue
            
        # Expansion
        if reflected_val < values[0]:
            expanded = centroid + gamma * (reflected - centroid)
            expanded_val = f(expanded)
            if expanded_val < reflected_val:
                simplex[-1] = expanded
                values[-1] = expanded_val
            else:
                simplex[-1] = reflected
                values[-1] = reflected_val
            continue
            
        # Contraction
        contracted = centroid + beta * (simplex[-1] - centroid)
        contracted_val = f(contracted)
        
        if contracted_val < values[-1]:
            simplex[-1] = contracted
            values[-1] = contracted_val
            continue
            
        # Shrink
        for i in range(1, dim + 1):
            simplex[i] = simplex[0] + beta * (simplex[i] - simplex[0])
            values[i] = f(simplex[i])
            
    if iter == max_iter - 1:
        raise RuntimeError("Failed to converge within maximum iterations")
        
    return simplex[0], values[0]
