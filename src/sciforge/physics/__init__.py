"""
Physics module containing classical mechanics and other physics simulations

Modules:
- mechanics: Classical particle mechanics and rigid body dynamics
- fields: Electromagnetic and gravitational field calculations  
- waves: Wave mechanics and propagation
- thermodynamics: Heat transfer and thermal systems
- quantum: Basic quantum mechanical systems
- relativity: Special and general relativity calculations
"""

from .mechanics import Particle
from .fields import ElectricField, MagneticField, GravitationalField
from .waves import Wave, WavePacket
from .thermodynamics import ThermalSystem
from .quantum import Wavefunction
from .relativity import LorentzTransform

__all__ = [
    'Particle',
    'ElectricField', 'MagneticField', 'GravitationalField',
    'Wave', 'WavePacket',
    'ThermalSystem',
    'Wavefunction',
    'LorentzTransform'
]
