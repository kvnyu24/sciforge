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
    'me': 9.1093837015e-31,  # Electron mass (kg)
    'mp': 1.67262192369e-27, # Proton mass (kg)
    'mn': 1.67492749804e-27, # Neutron mass (kg)
    'alpha': 7.297352569e-3, # Fine structure constant
    'Na': 6.02214076e23,     # Avogadro constant (mol⁻¹)
    'R': 8.31446261815324,   # Gas constant (J/mol/K)
    'sigma': 5.670374419e-8, # Stefan-Boltzmann constant (W/m²/K⁴)
    
    # Mathematical constants
    'pi': np.pi,
    'e': np.e,
    'golden_ratio': (1 + np.sqrt(5)) / 2,
    'euler_gamma': 0.57721566490153286, # Euler-Mascheroni constant
    'sqrt2': np.sqrt(2),
    'sqrt3': np.sqrt(3),
    'ln2': np.log(2),
    'ln10': np.log(10)
} 