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
from .statistical_mechanics import *
from .quantum_mechanics import *
from .qft import *
from .condensed_matter import *
from .amo import *
from .plasma import *

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
    # Relativity (Phase 6)
    # Special Relativity
    'LorentzTransform', 'MinkowskiSpacetime', 'RelativisticParticle',
    'FourVector', 'FourMomentum', 'FourVelocity', 'FourForce',
    'ElectromagneticFieldTensor', 'StressEnergyTensor', 'CovariantMaxwell',
    # General Relativity Foundations
    'MetricTensor', 'ChristoffelSymbols', 'RiemannTensor', 'RicciTensor', 'RicciScalar',
    'EinsteinTensor', 'GeodesicEquation',
    # Exact Solutions
    'SchwarzschildMetric', 'KerrMetric', 'ReissnerNordstromMetric', 'KerrNewmanMetric', 'FRWMetric',
    # GR Phenomena
    'GravitationalRedshift', 'PeriastronPrecession', 'GravitationalLensing',
    'FrameDragging', 'EventHorizon', 'HawkingTemperature',
    # Gravitational Waves
    'LinearizedGravity', 'GravitationalWave', 'QuadrupoleFormula', 'ChirpMass', 'GWTemplate',
    # Cosmology
    'FriedmannEquations', 'HubbleParameter', 'CosmicScale', 'RedshiftDistance',
    'DarkEnergy', 'InflationModel',
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
    # Statistical Mechanics (Phase 4)
    # Thermodynamic Laws
    'ThermodynamicProcess', 'CarnotEngine', 'HeatPump', 'EntropyCalculator', 'FreeEnergyMinimizer',
    # Thermodynamic Potentials
    'InternalEnergy', 'Enthalpy', 'HelmholtzFree', 'GibbsFree', 'ChemicalPotential', 'MaxwellRelations',
    # Equations of State
    'VanDerWaalsGas', 'RedlichKwong', 'VirialExpansion', 'IdealMixture',
    # Statistical Ensembles
    'MicrocanonicalEnsemble', 'CanonicalEnsemble', 'GrandCanonicalEnsemble', 'PartitionFunction', 'EquipartitionTheorem',
    # Quantum Statistics
    'BoseEinsteinDistribution', 'FermiDiracDistribution', 'MaxwellBoltzmannDistribution',
    'PhotonGas', 'PhononGas', 'DebyeModel', 'EinsteinModel',
    # Phase Transitions
    'IsingModel1D', 'IsingModel2D', 'LandauTheory', 'CriticalExponents',
    # Non-equilibrium
    'BoltzmannEquation', 'LangevinDynamics', 'FokkerPlanckEquation',
    'FluctuationDissipation', 'JarzynskiEquality', 'CrooksRelation',
    # Quantum Mechanics (Phase 5)
    # Fundamental Operators
    'PositionOperator', 'MomentumOperator', 'AngularMomentumOperator', 'HamiltonianOperator',
    'CreationOperator', 'AnnihilationOperator', 'NumberOperator',
    # Canonical Quantum Systems
    'FiniteWell', 'DoubleWell', 'DeltaPotential', 'StepPotential', 'BarrierTunneling',
    'CoulombPotential', 'HarmonicOscillator3D', 'MorsePotential',
    # Angular Momentum
    'OrbitalAngularMomentum', 'SpinAngularMomentum', 'SpinOrbitCoupling',
    'ClebschGordan', 'WignerDMatrix', 'SphericalHarmonicsQM',
    # Multi-particle Systems
    'TwoParticleSystem', 'IdenticalBosons', 'IdenticalFermions', 'ExchangeInteraction',
    # Approximation Methods
    'TimeIndependentPerturbation', 'VariationalMethod', 'WKBApproximation',
    # Open Quantum Systems
    'DensityMatrix', 'VonNeumannEquation', 'LindbladMasterEquation',
    # QFT Foundations (Phase 7)
    # Classical Field Theory
    'ScalarField', 'VectorField', 'DiracField', 'FieldLagrangian', 'EulerLagrangeField',
    # Canonical Quantization
    'FieldCommutator', 'FockSpace', 'VacuumState', 'NormalOrdering', 'WickTheorem',
    # Propagators & Diagrams
    'FeynmanPropagator', 'FeynmanVertex', 'FeynmanDiagram', 'CrossSection', 'DecayRate',
    # Symmetries
    'GlobalSymmetry', 'LocalGaugeSymmetry', 'SpontaneousSymmetryBreaking',
    'GoldstoneBoson', 'HiggsMechanism',
    # Condensed Matter (Phase 8)
    # Crystal Structure
    'BravaisLattice', 'ReciprocalLattice', 'BrillouinZone', 'CrystalSymmetry', 'MillerIndices',
    # Band Theory
    'BlochWavefunction', 'KronigPenney', 'TightBinding', 'NearlyFreeElectron',
    'EffectiveMass', 'DensityOfStates', 'FermiSurface',
    # Semiconductors
    'IntrinsicSemiconductor', 'DopedSemiconductor', 'PNJunction', 'QuantumWell', 'QuantumDot',
    # Transport
    'DrudeModel', 'BoltzmannTransport', 'HallEffect', 'Mobility',
    # Lattice Dynamics
    'PhononDispersion', 'ThermalConductivity',
    # Magnetism
    'Diamagnetism', 'Paramagnetism', 'Ferromagnetism', 'MagnonDispersion', 'HysteresisLoop',
    # Superconductivity
    'BCSTheory', 'CooperPair', 'MeissnerEffect', 'JosephsonJunction', 'SQUID',
    # Topological Matter
    'BerryPhase', 'BerryCurvature', 'ChernNumber', 'TopologicalInsulator2D', 'IntegerQuantumHall',
    # AMO Physics (Phase 9)
    # Atomic Structure
    'HydrogenAtom', 'MultielectronAtom', 'SlaterDeterminant', 'AtomicTerm', 'SelectionRules',
    # Atom-Light Interaction
    'TwoLevelAtom', 'BlochEquations', 'DipoleMatrixElement', 'EinsteinCoefficients',
    # Laser Physics
    'LaserCavity', 'GainMedium', 'RateEquations', 'ModeLocking',
    # Laser Cooling & Trapping
    'DopplerCooling', 'MagnetoOpticalTrap', 'OpticalDipoleTrap', 'OpticalLattice',
    # Ultracold Atoms
    'BoseEinsteinCondensate', 'GrossPitaevskii', 'FermiGas', 'FeshbachResonance',
    # Molecular Physics
    'MolecularOrbital', 'BornOppenheimer', 'VibrationalSpectrum', 'RotationalSpectrum', 'FranckCondon',
    # Plasma & Astrophysics (Phase 10)
    # Plasma Fundamentals
    'DebyeLength', 'PlasmaFrequency', 'PlasmaParameter', 'VlasovEquation', 'LandauDamping',
    # Magnetohydrodynamics
    'MHDEquations', 'AlfvenWave', 'Magnetosonic', 'MHDInstability', 'MagneticReconnection',
    # Fusion
    'LawsonCriterion', 'TokamakEquilibrium', 'MirrorTrap', 'ICFCapsule',
    # Astrophysics
    'HydrostaticStar', 'LaneEmden', 'StellarEvolution', 'WhiteDwarf', 'NeutronStar',
    'AccretionDisk', 'JetLaunching', 'Nucleosynthesis',
]
