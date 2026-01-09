"""
The sciforge.chaos module provides tools for exploring chaotic dynamics,
including fractal generation, simulation of classic attractors, and analysis of
chaotic systems.
"""

# Public API exports from submodules
from .fractals import generate_mandelbrot_set, generate_julia_set

# __all__ defines the public API for 'from sciforge.chaos import *'
__all__ = [
    # from fractals.py
    'generate_mandelbrot_set',
    'generate_julia_set',
] 