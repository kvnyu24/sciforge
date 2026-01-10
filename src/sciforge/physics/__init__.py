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
from .kinematics import *
from .energy import *
from .orbital import *
from .analytical_mechanics import *
from .electromagnetism import *
from .optics import *

__all__ = [
    # Mechanics
    'Particle', 'Constraint', 'RotationalSpring',
    'DynamicalSystem', 'PhysicalSystem', 'RotationalSystem',
    # Fields
    'ElectricField', 'MagneticField', 'GravitationalField',
    # Waves
    'Wave', 'WavePacket',
    # Thermodynamics
    'ThermalSystem',
    # Quantum
    'Wavefunction',
    # Relativity
    'LorentzTransform',
    # Fluids
    'FluidColumn',
    # Attosecond
    'StrongFieldSystem', 'AttosecondPulseGenerator',
    # Circuits
    'Circuit', 'CircuitElement', 'Resistor', 'Capacitor', 'Inductor',
    # EM Waves
    'ElectromagneticWave',
    # Oscillations
    'HarmonicOscillator', 'CoupledOscillator', 'ParametricOscillator',
    # Statistical
    'HermitePolynomial', 'RiceDistribution',
    # Kinematics (Phase 1.1)
    'ProjectileMotion', 'CircularMotion', 'ReferenceFrame', 'RelativeMotion', 'CurvilinearMotion',
    # Energy (Phase 1.3)
    'WorkCalculator', 'PowerMeter', 'PotentialWell', 'EnergyLandscape', 'KineticEnergy', 'MechanicalEnergy',
    # Orbital (Phase 1.4)
    'OrbitalElements', 'KeplerianOrbit', 'TwoBodyProblem', 'ThreeBodyProblem', 'OrbitalManeuver', 'EscapeTrajectory',
    # Analytical Mechanics (Phase 1.5)
    'GeneralizedCoordinates', 'LagrangianSystem', 'HamiltonianSystem', 'PoissonBracket', 'ActionPrinciple', 'NoetherSymmetry',
    # Electromagnetism (Phase 2)
    'MaxwellSolver1D', 'MaxwellSolver2D', 'MaxwellSolver3D', 'GaussLaw', 'FaradayInduction', 'AmpereMaxwell',
    'ScalarPotential', 'VectorPotential', 'GaugeFreedom', 'RetardedPotential',
    'DipoleRadiation', 'LarmorFormula', 'SynchrotronRadiation', 'CherenkovRadiation', 'Bremsstrahlung', 'AntennaPattern',
    'MultipoleExpansion', 'OctupoleField', 'SphericalHarmonics',
    'DielectricMaterial', 'MagneticMaterial', 'ConductorSkin', 'PlasmaDispersion', 'MetamaterialUnit',
    # Optics (Phase 3)
    # Wave Equation Solvers
    'WaveEquation1D', 'WaveEquation2D', 'WaveEquation3D', 'HelmholtzSolver',
    # Interference & Diffraction
    'TwoSlitInterference', 'MultiSlitInterference', 'SingleSlitDiffraction',
    'CircularAperture', 'ThinFilmInterference', 'FabryPerotInterferometer',
    # Geometric Optics
    'Ray', 'ThinLens', 'ThickLens', 'SphericalMirror', 'OpticalSystem', 'Prism', 'SnellRefraction',
    # Polarization
    'JonesVector', 'JonesMatrix', 'StokesVector', 'MuellerMatrix', 'Waveplate', 'Polarizer',
    # Nonlinear Optics
    'SecondHarmonicGeneration', 'KerrEffect', 'FourWaveMixing', 'SolitonPulse',
    # Acoustics
    'SoundWave', 'AcousticImpedance', 'DopplerShift', 'ResonantCavity', 'Ultrasound',
]
