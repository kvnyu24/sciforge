"""
Physical and mathematical constants used throughout the library
"""

import numpy as np

CONSTANTS = {
    # Physical constants
    'c': 299792458.0,        # Speed of light (m/s)
    'G': 6.67430e-11,        # Gravitational constant (m³/kg/s²)
    'h': 6.62607015e-34,     # Planck constant (J⋅s)
    'hbar': 1.054571817e-34, # Reduced Planck constant (J⋅s)
    'e': 1.602176634e-19,    # Elementary charge (C)
    'k': 1.380649e-23,       # Boltzmann constant (J/K)
    'eps0': 8.8541878128e-12,# Vacuum permittivity (F/m)
    'mu0': 1.25663706212e-6, # Vacuum permeability (N/A²)
    
    # Mathematical constants
    'pi': np.pi,
    'e': np.e,
    'golden_ratio': (1 + np.sqrt(5)) / 2
} 