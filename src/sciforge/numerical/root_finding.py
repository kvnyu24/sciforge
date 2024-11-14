"""
Root finding algorithms for numerical solutions of equations.
"""

import numpy as np
from typing import Callable, Tuple, Optional


def bisection(f: Callable[[float], float], 
              a: float, 
              b: float, 
              tol: float = 1e-6,
              max_iter: int = 100) -> Tuple[float, float]:
    """
    Find root using bisection method.
    
    Args:
        f: Function to find root of
        a: Left bracket
        b: Right bracket  
        tol: Tolerance for convergence
        max_iter: Maximum iterations
        
    Returns:
        Tuple of (root, function value at root)
    """
    fa = f(a)
    fb = f(b)
    
    if fa * fb > 0:
        raise ValueError("Function must have opposite signs at brackets")
        
    for i in range(max_iter):
        c = (a + b) / 2
        fc = f(c)
        
        if abs(fc) < tol:
            return c, fc
            
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
            
    raise RuntimeError("Failed to converge within maximum iterations")


def newton(f: Callable[[float], float],
          df: Callable[[float], float],
          x0: float,
          tol: float = 1e-6,
          max_iter: int = 100) -> Tuple[float, float]:
    """
    Find root using Newton's method.
    
    Args:
        f: Function to find root of
        df: Derivative of function
        x0: Initial guess
        tol: Tolerance for convergence
        max_iter: Maximum iterations
        
    Returns:
        Tuple of (root, function value at root)
    """
    x = x0
    
    for i in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:
            return x, fx
            
        dfx = df(x)
        if dfx == 0:
            raise ValueError("Derivative is zero")
            
        x = x - fx/dfx
        
    raise RuntimeError("Failed to converge within maximum iterations")


def secant(f: Callable[[float], float],
          x0: float,
          x1: float,
          tol: float = 1e-6, 
          max_iter: int = 100) -> Tuple[float, float]:
    """
    Find root using secant method.
    
    Args:
        f: Function to find root of
        x0: First initial guess
        x1: Second initial guess
        tol: Tolerance for convergence
        max_iter: Maximum iterations
        
    Returns:
        Tuple of (root, function value at root)
    """
    f0 = f(x0)
    f1 = f(x1)
    
    for i in range(max_iter):
        if abs(f1) < tol:
            return x1, f1
            
        if f1 == f0:
            raise ValueError("Function values are equal")
            
        x_new = x1 - f1 * (x1 - x0)/(f1 - f0)
        x0, x1 = x1, x_new
        f0, f1 = f1, f(x1)
        
    raise RuntimeError("Failed to converge within maximum iterations")


def brent(f: Callable[[float], float],
         a: float,
         b: float,
         tol: float = 1e-6,
         max_iter: int = 100) -> Tuple[float, float]:
    """
    Find root using Brent's method.
    
    Args:
        f: Function to find root of
        a: Left bracket
        b: Right bracket
        tol: Tolerance for convergence
        max_iter: Maximum iterations
        
    Returns:
        Tuple of (root, function value at root)
    """
    fa = f(a)
    fb = f(b)
    
    if fa * fb > 0:
        raise ValueError("Function must have opposite signs at brackets")
        
    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa
        
    c = a
    fc = fa
    d = b - a
    e = d
    
    for i in range(max_iter):
        if abs(fb) < tol:
            return b, fb
            
        if fa != fc and fb != fc:
            # Inverse quadratic interpolation
            s = a * fb * fc / ((fa - fb) * (fa - fc)) + \
                b * fa * fc / ((fb - fa) * (fb - fc)) + \
                c * fa * fb / ((fc - fa) * (fc - fb))
        else:
            # Secant method
            s = b - fb * (b - a) / (fb - fa)
            
        # Bounds checking
        if (s < (3*a + b)/4 or s > b) or \
           (abs(s - b) >= abs(b - c)/2) or \
           (abs(b - c) < abs(e)) and (abs(s - b) >= abs(e)/2) or \
           (abs(e) < tol) and (abs(s - b) >= abs(d)/2):
            # Bisection
            s = (a + b)/2
            e = d
        else:
            e = d
            
        d = b - s
        c = b
        fc = fb
        
        fs = f(s)
        if fa * fs < 0:
            b = s
            fb = fs
        else:
            a = s
            fa = fs
            
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa
            
    raise RuntimeError("Failed to converge within maximum iterations")
