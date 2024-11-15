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
- attosecond: Attosecond optics and strong-field physics simulations
- circuits: Electrical circuit components and simulations
- em_waves: Electromagnetic wave propagation and simulations
"""

from .mechanics import *
from .forces import *
from .fields import *
from .waves import *
from .thermodynamics import *
from .quantum import *
from .relativity import *
from .fluids import *
from .attosecond import *
from .circuits import *
from .em_waves import *
from .oscillations import *
from .statistical import HermitePolynomial, RiceDistribution

__all__ = [
    'Particle', 'Constraint', 'RotationalSpring',
    'ElectricField', 'MagneticField', 'GravitationalField',
    'Wave', 'WavePacket',
    'ThermalSystem',
    'Wavefunction',
    'LorentzTransform',
    'FluidColumn',
    'StrongFieldSystem',
    'AttosecondPulseGenerator',
    'Circuit', 'CircuitElement', 'Resistor', 'Capacitor', 'Inductor',
    'ElectromagneticWave',
    'HarmonicOscillator', 'CoupledOscillator', 'ParametricOscillator',
    'DynamicalSystem', 'PhysicalSystem',
    'RotationalSystem',
    'HermitePolynomial', 'RiceDistribution'
]
