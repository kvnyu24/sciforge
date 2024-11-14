"""
Physics module containing classical mechanics and other physics simulations

Modules:
- mechanics: Classical particle mechanics and rigid body dynamics
- fields: Electromagnetic and gravitational field calculations  
- waves: Wave mechanics and propagation
- thermodynamics: Heat transfer and thermal systems
- quantum: Basic quantum mechanical systems
- relativity: Special and general relativity calculations
- fluids: Fluid dynamics simulations
"""

from .mechanics import *
from .fields import *
from .waves import *
from .thermodynamics import *
from .quantum import *
from .relativity import *
from .fluids import *

__all__ = [
    'Particle',
    'ElectricField', 'MagneticField', 'GravitationalField', 
    'Wave', 'WavePacket',
    'ThermalSystem',
    'Wavefunction',
    'LorentzTransform',
    'FluidColumn'
]
