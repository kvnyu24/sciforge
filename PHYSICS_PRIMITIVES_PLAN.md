# SciForge Physics Primitives Implementation Plan

This document provides a comprehensive gap analysis and implementation roadmap for physics primitives in SciForge.

---

## Current Implementation Status

### ✅ Already Implemented

#### Core Infrastructure
- [x] Base classes (BaseClass, BaseSolver, BaseProcess)
- [x] Physical constants (c, G, h, ħ, e, k_B, ε₀, μ₀, me, mp, mn, α, Na, R, σ)
- [x] Mathematical constants (π, e, φ, γ, √2, √3)
- [x] Validation utilities (array, vector, bounds, probability)
- [x] Exception hierarchy (Validation, Numerical, Physics errors)
- [x] Integrators (Euler, RK2, RK4, RK45, adaptive)

#### Classical Mechanics
- [x] Particle dynamics with RK4 integration
- [x] Rigid body dynamics (translation + rotation)
- [x] Simple/double pendulum
- [x] Spring (Hooke's law)
- [x] Friction (static/kinetic)
- [x] Drag force (quadratic)
- [x] Collision resolution (elastic/inelastic)
- [x] Constraints (Lagrange multipliers)
- [x] Lennard-Jones molecular force
- [x] Rotational systems (gyroscope, precession)
- [x] Moment of inertia calculations

#### Fields & Electromagnetism
- [x] Electric field (point charge, Coulomb)
- [x] Magnetic field (current element, Biot-Savart)
- [x] Gravitational field (point mass)
- [x] Uniform field
- [x] Dipole field (electric/magnetic)
- [x] Quadrupole field
- [x] Solenoidal field
- [x] Conservative fields with potential
- [x] EM wave propagation with dispersion

#### Circuits
- [x] Circuit elements (R, L, C)
- [x] Voltage/current sources (DC + AC)
- [x] RLC with parasitic effects
- [x] Network analysis (modified nodal)
- [x] Impedance calculations

#### Oscillations & Waves
- [x] Harmonic oscillator (damped, driven)
- [x] Coupled oscillators + normal modes
- [x] Torsional oscillator
- [x] Resonant systems + frequency scanning
- [x] Parametric oscillator
- [x] Classical waves (sinusoidal)
- [x] Wave packets (Gaussian envelope)
- [x] Standing waves
- [x] Damped waves
- [x] Shock waves

#### Quantum Mechanics
- [x] General wavefunction + probability density
- [x] Expectation values
- [x] Split-operator time evolution
- [x] Quantum harmonic oscillator (energy levels, Hermite eigenstates)
- [x] Particle in box (infinite well)
- [x] Spin systems (spin matrices)

#### Thermodynamics
- [x] Thermal systems (heat capacity, conductivity)
- [x] Heat transfer (conductive)
- [x] Ideal gas (PV=nRT)
- [x] Heat exchangers (effectiveness-NTU)
- [x] Phase change (latent heat)

#### Fluids
- [x] Fluid column (Plateau-Rayleigh instability)
- [x] Fluid jet (Coandă effect)

#### Relativity
- [x] Lorentz transformations
- [x] Time dilation, length contraction
- [x] Relativistic momentum/energy
- [x] Minkowski spacetime (metric, intervals)
- [x] Relativistic particle dynamics

#### Stochastic Processes
- [x] Poisson process
- [x] Wiener process (Brownian motion)
- [x] Ornstein-Uhlenbeck
- [x] Geometric Brownian motion
- [x] Cox-Ingersoll-Ross
- [x] Vasicek model
- [x] Metropolis-Hastings MCMC

#### Numerical Methods
- [x] ODE solvers (Euler, RK2/3/4, Adams-Bashforth, Adaptive RK45)
- [x] Integration (trapezoid, Simpson)
- [x] Optimization (Newton, gradient descent, Nelder-Mead)
- [x] Root finding (bisection, Newton, secant, Brent)
- [x] Interpolation (linear, cubic spline)

#### Chaos & Fractals
- [x] Mandelbrot set (Numba-accelerated)
- [x] Julia set (Numba-accelerated)

#### Advanced/Modern
- [x] Attosecond pulse generation (HHG, ADK tunneling)
- [x] Strong-field laser-matter interaction

---

## 🔴 NOT YET IMPLEMENTED - Comprehensive Gap Analysis

### Phase 1: Core Mechanics Extensions (Foundation)

#### 1.1 Kinematics Primitives
- [ ] `ProjectileMotion` - 2D/3D projectile with drag
- [ ] `CircularMotion` - Uniform/non-uniform circular motion
- [ ] `RelativeMotion` - Reference frame transformations
- [ ] `CurvilinearMotion` - Motion along arbitrary paths

#### 1.2 Force Law Extensions
- [ ] `GravityForce3D` - Full 3D gravitational force
- [ ] `VanDerWaalsForce` - Intermolecular attraction
- [ ] `ElectrostaticForce` - Coulomb force between charges
- [ ] `LorentzForce` - Full F = q(E + v×B)
- [ ] `TidalForce` - Gravitational gradient force
- [ ] `CentrifugalForce` - Non-inertial frame force
- [ ] `CoriolisForce` - Rotating reference frame

#### 1.3 Energy & Work Primitives
- [ ] `WorkCalculator` - Line integral of force
- [ ] `PowerMeter` - Instantaneous/average power
- [ ] `PotentialWell` - General potential energy surfaces
- [ ] `EnergyLandscape` - Multi-dimensional potential surfaces

#### 1.4 Orbital Mechanics
- [ ] `KeplerianOrbit` - Elliptical orbits, orbital elements
- [ ] `TwoBodyProblem` - Reduced mass formulation
- [ ] `ThreeBodyProblem` - Restricted 3-body (Lagrange points)
- [ ] `OrbitalManeuver` - Hohmann transfer, gravity assist
- [ ] `EscapeVelocity` - Escape trajectory calculations

#### 1.5 Lagrangian/Hamiltonian Mechanics
- [ ] `LagrangianSystem` - Generalized coordinates, Euler-Lagrange
- [ ] `HamiltonianSystem` - Phase space, Hamilton's equations
- [ ] `PoissonBracket` - Canonical transformations
- [ ] `ActionPrinciple` - Variational methods
- [ ] `NoetherSymmetry` - Symmetry → conservation mapping

---

### Phase 2: Electromagnetism Deep Dive

#### 2.1 Maxwell's Equations Solvers
- [ ] `MaxwellSolver1D` - 1D FDTD solver
- [ ] `MaxwellSolver2D` - 2D FDTD solver
- [ ] `MaxwellSolver3D` - 3D FDTD solver
- [ ] `GaussLaw` - Divergence equation solver
- [ ] `FaradayInduction` - Time-varying B fields
- [ ] `AmpereMaxwell` - Displacement current

#### 2.2 Electromagnetic Potentials
- [ ] `ScalarPotential` - φ field calculations
- [ ] `VectorPotential` - A field calculations
- [ ] `GaugeFreedom` - Coulomb/Lorenz gauge
- [ ] `RetardedPotential` - Jefimenko's equations

#### 2.3 Radiation
- [ ] `DipoleRadiation` - Oscillating dipole radiation
- [ ] `LarmorFormula` - Accelerating charge power
- [ ] `SynchrotronRadiation` - Relativistic circular motion
- [ ] `CherenkovRadiation` - Superluminal particle
- [ ] `Bremsstrahlung` - Deceleration radiation
- [ ] `AntennaPattern` - Radiation pattern calculations

#### 2.4 Multipole Expansion
- [ ] `MultipoleExpansion` - General expansion
- [ ] `OctupoleField` - Higher-order multipoles
- [ ] `SphericalHarmonics` - Ylm basis functions

#### 2.5 Materials & Media
- [ ] `DielectricMaterial` - Permittivity, polarization
- [ ] `MagneticMaterial` - Permeability, magnetization
- [ ] `ConductorSkin` - Skin depth, AC resistance
- [ ] `PlasmaDispersion` - Plasma frequency, cutoff
- [ ] `MetamaterialUnit` - Negative index materials

---

### Phase 3: Waves & Optics Complete

#### 3.1 Wave Equation Solvers
- [ ] `WaveEquation1D` - String/pipe acoustics
- [ ] `WaveEquation2D` - Membrane vibrations
- [ ] `WaveEquation3D` - 3D wave propagation
- [ ] `HelmholtzSolver` - Time-independent waves

#### 3.2 Interference & Diffraction
- [ ] `TwoSlitInterference` - Young's experiment
- [ ] `MultiSlitInterference` - Diffraction gratings
- [ ] `SingleSlitDiffraction` - Fraunhofer/Fresnel
- [ ] `CircularAperture` - Airy disk pattern
- [ ] `ThinFilmInterference` - Coating design
- [ ] `FabryPerotInterferometer` - Cavity resonances

#### 3.3 Geometric Optics
- [ ] `Ray` - Ray propagation primitive
- [ ] `ThinLens` - Paraxial optics
- [ ] `ThickLens` - Cardinal points
- [ ] `SphericalMirror` - Reflection optics
- [ ] `OpticalSystem` - ABCD matrix propagation
- [ ] `Prism` - Dispersion, deviation
- [ ] `SnellRefraction` - Interface calculations

#### 3.4 Polarization
- [ ] `JonesVector` - Polarization state
- [ ] `JonesMatrix` - Polarization optics
- [ ] `MuellerMatrix` - Partially polarized light
- [ ] `StokesVector` - Stokes parameters
- [ ] `Waveplate` - Retarder elements
- [ ] `Polarizer` - Linear/circular polarizers

#### 3.5 Nonlinear Optics
- [ ] `SecondHarmonicGeneration` - χ² processes
- [ ] `KerrEffect` - χ³ self-focusing
- [ ] `FourWaveMixing` - Parametric processes
- [ ] `SolitonPulse` - Nonlinear wave packets

#### 3.6 Acoustics
- [ ] `SoundWave` - Pressure wave propagation
- [ ] `AcousticImpedance` - Material matching
- [ ] `DopplerShift` - Moving source/observer
- [ ] `ResonantCavity` - Standing wave modes
- [ ] `Ultrasound` - High-frequency acoustics

---

### Phase 4: Thermodynamics & Statistical Mechanics

#### 4.1 Thermodynamic Laws
- [ ] `ThermodynamicProcess` - Isothermal/adiabatic/isobaric/isochoric
- [ ] `CarnotEngine` - Ideal heat engine
- [ ] `HeatPump` - Refrigeration cycle
- [ ] `EntropyCalculator` - dS calculations
- [ ] `FreeEnergyMinimizer` - Helmholtz/Gibbs minimization

#### 4.2 Thermodynamic Potentials
- [ ] `InternalEnergy` - U(S,V,N)
- [ ] `Enthalpy` - H = U + pV
- [ ] `HelmholtzFree` - F = U - TS
- [ ] `GibbsFree` - G = H - TS
- [ ] `ChemicalPotential` - μ = ∂G/∂N
- [ ] `MaxwellRelations` - Cross-derivative identities

#### 4.3 Equations of State
- [ ] `VanDerWaalsGas` - Real gas corrections
- [ ] `RedlichKwong` - Improved real gas
- [ ] `VirialExpansion` - Polynomial EOS
- [ ] `IdealMixture` - Dalton's/Raoult's laws

#### 4.4 Statistical Ensembles
- [ ] `MicrocanonicalEnsemble` - Fixed E, V, N
- [ ] `CanonicalEnsemble` - Fixed T, V, N
- [ ] `GrandCanonicalEnsemble` - Fixed T, V, μ
- [ ] `PartitionFunction` - Z calculations
- [ ] `EquipartitionTheorem` - Degree of freedom energy

#### 4.5 Quantum Statistics
- [ ] `BoseEinsteinDistribution` - Boson statistics
- [ ] `FermiDiracDistribution` - Fermion statistics
- [ ] `MaxwellBoltzmannDistribution` - Classical limit
- [ ] `PhotonGas` - Blackbody radiation
- [ ] `PhononGas` - Lattice vibrations
- [ ] `DebyeModel` - Heat capacity
- [ ] `EinsteinModel` - Simpler heat capacity

#### 4.6 Phase Transitions
- [ ] `IsingModel1D` - Exact solution
- [ ] `IsingModel2D` - Monte Carlo simulation
- [ ] `XYModel` - Continuous spin
- [ ] `HeisenbergModel` - 3D spin
- [ ] `LandauTheory` - Order parameter expansion
- [ ] `CriticalExponents` - Universality classes
- [ ] `CorrelationLength` - Divergence near Tc

#### 4.7 Non-equilibrium
- [ ] `BoltzmannEquation` - Kinetic theory
- [ ] `FluctuationDissipation` - Kubo formula
- [ ] `LangevinDynamics` - Stochastic mechanics
- [ ] `FokkerPlanckEquation` - Probability evolution
- [ ] `JarzynskiEquality` - Work fluctuation theorem
- [ ] `CrooksRelation` - Time-reversal symmetry

---

### Phase 5: Quantum Mechanics Complete

#### 5.1 Fundamental Operators
- [ ] `PositionOperator` - x̂ representation
- [ ] `MomentumOperator` - p̂ = -iħ∇
- [ ] `AngularMomentumOperator` - L̂ = r × p
- [ ] `HamiltonianOperator` - Ĥ construction
- [ ] `CreationOperator` - â†
- [ ] `AnnihilationOperator` - â
- [ ] `NumberOperator` - n̂ = â†â

#### 5.2 Canonical Quantum Systems
- [ ] `FiniteWell` - Finite square well (transcendental solutions)
- [ ] `DoubleWell` - Tunneling, splitting
- [ ] `DeltaPotential` - Dirac delta potential
- [ ] `StepPotential` - Scattering states
- [ ] `BarrierTunneling` - Transmission coefficient
- [ ] `CoulombPotential` - Hydrogen-like atoms
- [ ] `3DHarmonicOscillator` - Isotropic oscillator
- [ ] `MorsePotential` - Anharmonic molecular potential

#### 5.3 Angular Momentum
- [ ] `OrbitalAngularMomentum` - L², Lz eigenstates
- [ ] `SpinAngularMomentum` - Spin-s systems
- [ ] `SpinOrbitCoupling` - L·S interaction
- [ ] `ClebschGordan` - Angular momentum addition
- [ ] `WignerDMatrix` - Rotation matrices
- [ ] `SphericalHarmonicsQM` - Ylm eigenfunctions

#### 5.4 Multi-particle Systems
- [ ] `TwoParticleSystem` - Distinguishable particles
- [ ] `IdenticalBosons` - Symmetric wavefunctions
- [ ] `IdenticalFermions` - Antisymmetric (Slater determinant)
- [ ] `ExchangeInteraction` - Fermionic exchange energy
- [ ] `Helium` - Two-electron atom
- [ ] `SecondQuantization` - Fock space formalism

#### 5.5 Approximation Methods
- [ ] `TimeIndependentPerturbation` - Non-degenerate
- [ ] `DegeneratePerturbation` - Degenerate levels
- [ ] `TimeDependentPerturbation` - Fermi's golden rule
- [ ] `VariationalMethod` - Energy upper bound
- [ ] `WKBApproximation` - Semiclassical
- [ ] `BornApproximation` - Scattering theory
- [ ] `HartreeFock` - Mean-field many-body
- [ ] `DensityFunctional` - DFT basics

#### 5.6 Open Quantum Systems
- [ ] `DensityMatrix` - Mixed state representation
- [ ] `VonNeumannEquation` - Unitary evolution
- [ ] `LindbladMasterEquation` - Dissipative evolution
- [ ] `QuantumChannel` - CPTP maps
- [ ] `Decoherence` - Environment-induced
- [ ] `QuantumMeasurement` - POVM, Kraus operators

#### 5.7 Quantum Phenomena
- [ ] `QuantumTunneling` - Tunneling dynamics
- [ ] `QuantumInterference` - Path integral interference
- [ ] `Entanglement` - Bipartite entanglement measures
- [ ] `BellState` - Maximally entangled pairs
- [ ] `BellInequality` - CHSH test
- [ ] `QuantumTeleportation` - Protocol simulation
- [ ] `QuantumZenoEffect` - Frequent measurement

---

### Phase 6: Special & General Relativity

#### 6.1 Special Relativity Extensions
- [ ] `FourVector` - Spacetime 4-vectors
- [ ] `FourMomentum` - Energy-momentum 4-vector
- [ ] `FourVelocity` - Proper velocity
- [ ] `FourForce` - Relativistic force
- [ ] `ElectromagneticFieldTensor` - Fμν tensor
- [ ] `StressEnergyTensor` - Tμν (SR context)
- [ ] `CovariantMaxwell` - Tensor form of Maxwell

#### 6.2 General Relativity Foundations
- [ ] `MetricTensor` - gμν specification
- [ ] `ChristoffelSymbols` - Connection coefficients
- [ ] `RiemannTensor` - Full curvature tensor
- [ ] `RicciTensor` - Contracted Riemann
- [ ] `RicciScalar` - Scalar curvature
- [ ] `EinsteinTensor` - Gμν = Rμν - ½gμνR
- [ ] `GeodesicEquation` - Free-fall trajectories

#### 6.3 Exact Solutions
- [ ] `SchwarzschildMetric` - Static spherical
- [ ] `KerrMetric` - Rotating black hole
- [ ] `ReissnerNordstromMetric` - Charged black hole
- [ ] `KerrNewmanMetric` - Charged rotating
- [ ] `FRWMetric` - Cosmological metric

#### 6.4 GR Phenomena
- [ ] `GravitationalRedshift` - Frequency shift
- [ ] `PeriastronPrecession` - Mercury advance
- [ ] `GravitationalLensing` - Light bending
- [ ] `FrameDragging` - Lense-Thirring
- [ ] `EventHorizon` - Schwarzschild radius
- [ ] `HawkingTemperature` - Black hole thermodynamics

#### 6.5 Gravitational Waves
- [ ] `LinearizedGravity` - Weak-field perturbation
- [ ] `GravitationalWave` - h+ and h× polarizations
- [ ] `QuadrupoleFormula` - GW luminosity
- [ ] `ChirpMass` - Binary inspiral parameter
- [ ] `GWTemplate` - Matched filtering templates

#### 6.6 Cosmology
- [ ] `FriedmannEquations` - Expansion dynamics
- [ ] `HubbleParameter` - H(z) evolution
- [ ] `CosmicScale` - a(t) scale factor
- [ ] `RedshiftDistance` - Cosmological distances
- [ ] `DarkEnergy` - Equation of state w
- [ ] `InflationModel` - Slow-roll parameters

---

### Phase 7: Quantum Field Theory Foundations

#### 7.1 Classical Field Theory
- [ ] `ScalarField` - Klein-Gordon field
- [ ] `VectorField` - Proca field
- [ ] `DiracField` - Spinor field
- [ ] `FieldLagrangian` - L density construction
- [ ] `EulerLagrangeField` - Field equations

#### 7.2 Canonical Quantization
- [ ] `FieldCommutator` - [φ(x), π(y)]
- [ ] `FockSpace` - Particle number basis
- [ ] `VacuumState` - |0⟩ definition
- [ ] `NormalOrdering` - :operator:
- [ ] `WickTheorem` - Contractions

#### 7.3 Propagators & Feynman Diagrams
- [ ] `FeynmanPropagator` - ⟨0|T{φφ}|0⟩
- [ ] `FeynmanVertex` - Interaction vertex
- [ ] `FeynmanDiagram` - Diagram representation
- [ ] `CrossSection` - σ from |M|²
- [ ] `DecayRate` - Γ calculations

#### 7.4 Symmetries
- [ ] `GlobalSymmetry` - Noether current
- [ ] `LocalGaugeSymmetry` - Gauge field coupling
- [ ] `SpontaneousSymmetryBreaking` - Mexican hat
- [ ] `GoldstoneBoson` - Massless mode
- [ ] `HiggsMechanism` - Mass generation

---

### Phase 8: Condensed Matter Physics

#### 8.1 Crystal Structure
- [ ] `BravaisLattice` - 14 lattice types
- [ ] `ReciprocalLattice` - k-space
- [ ] `BrillouinZone` - First BZ construction
- [ ] `CrystalSymmetry` - Point/space groups
- [ ] `MillerIndices` - Plane notation

#### 8.2 Band Theory
- [ ] `BlochWavefunction` - Periodic potential
- [ ] `KronigPenney` - 1D band structure
- [ ] `TightBinding` - LCAO bands
- [ ] `NearlyFreeElectron` - Weak periodic potential
- [ ] `EffectiveMass` - Band curvature
- [ ] `DensityOfStates` - DOS calculations
- [ ] `FermiSurface` - 3D Fermi level

#### 8.3 Semiconductors
- [ ] `IntrinsicSemiconductor` - Undoped carrier stats
- [ ] `DopedSemiconductor` - n-type, p-type
- [ ] `PNJunction` - Depletion region, I-V
- [ ] `SchottkyBarrier` - Metal-semiconductor
- [ ] `Heterojunction` - Band alignment
- [ ] `QuantumWell` - 2D electron gas
- [ ] `QuantumDot` - 0D confinement

#### 8.4 Transport
- [ ] `DrudeModel` - Classical conductivity
- [ ] `SommerfeldModel` - Quantum corrections
- [ ] `BoltzmannTransport` - Semiclassical transport
- [ ] `HallEffect` - Classical Hall
- [ ] `MagnetoresistanceOrdinary` - B-field resistance
- [ ] `Mobility` - Carrier mobility

#### 8.5 Lattice Dynamics
- [ ] `PhononDispersion` - ω(k) curves
- [ ] `AcousticPhonon` - Linear dispersion
- [ ] `OpticalPhonon` - Gap at zone center
- [ ] `PhononDOS` - Vibrational DOS
- [ ] `ThermalConductivity` - Phonon heat transport
- [ ] `AnharmonicPhonon` - Phonon-phonon scattering

#### 8.6 Magnetism
- [ ] `Diamagnetism` - Larmor diamagnetic
- [ ] `Paramagnetism` - Curie law
- [ ] `Ferromagnetism` - Exchange, domains
- [ ] `Antiferromagnetism` - Néel order
- [ ] `Ferrimagnetism` - Uncompensated AF
- [ ] `MagnonDispersion` - Spin waves
- [ ] `HysteresisLoop` - M-H curves

#### 8.7 Superconductivity
- [ ] `BCSTheory` - Cooper pairing
- [ ] `CooperPair` - Bound state
- [ ] `EnergyGap` - Δ(T) temperature dependence
- [ ] `MeissnerEffect` - Flux expulsion
- [ ] `CoherenceLength` - ξ parameter
- [ ] `PenetrationDepth` - λ parameter
- [ ] `GinzburgLandau` - GL theory
- [ ] `Type2Superconductor` - Vortex lattice
- [ ] `FluxQuantum` - Φ₀ = h/2e
- [ ] `JosephsonJunction` - DC/AC effects
- [ ] `SQUID` - Flux sensor

#### 8.8 Topological Matter
- [ ] `BerryPhase` - Geometric phase
- [ ] `BerryCurvature` - Berry connection
- [ ] `ChernNumber` - Topological invariant
- [ ] `IntegerQuantumHall` - IQHE
- [ ] `FractionalQuantumHall` - FQHE (basics)
- [ ] `TopologicalInsulator2D` - Edge states
- [ ] `TopologicalInsulator3D` - Surface states
- [ ] `WeylSemimetal` - Weyl nodes
- [ ] `MajoranaMode` - Zero modes (toy model)

---

### Phase 9: Atomic, Molecular & Optical (AMO)

#### 9.1 Atomic Structure
- [ ] `HydrogenAtom` - Full radial + angular
- [ ] `MultielectronAtom` - Central field approx
- [ ] `SlaterDeterminant` - Antisymmetrization
- [ ] `AtomicTerm` - LS coupling
- [ ] `JJCoupling` - Heavy atoms
- [ ] `SelectionRules` - Dipole transitions
- [ ] `AtomicSpectrum` - Energy level diagrams

#### 9.2 Atom-Light Interaction
- [ ] `TwoLevelAtom` - Rabi oscillations
- [ ] `BlochEquations` - Optical Bloch
- [ ] `DipoleMatrixElement` - ⟨f|d|i⟩
- [ ] `EinsteinCoefficients` - A, B coefficients
- [ ] `SpontaneousEmission` - Decay rates
- [ ] `StimulatedEmission` - Gain
- [ ] `AbsorptionSpectrum` - Line shapes

#### 9.3 Laser Physics
- [ ] `LaserCavity` - Mode structure
- [ ] `GainMedium` - Population inversion
- [ ] `RateEquations` - N, φ dynamics
- [ ] `LaserThreshold` - Threshold condition
- [ ] `LaserLinewidth` - Schawlow-Townes
- [ ] `ModeLocking` - Ultrashort pulses
- [ ] `FrequencyComb` - Comb spectrum
- [ ] `OpticalParametricOscillator` - OPO

#### 9.4 Laser Cooling & Trapping
- [ ] `DopplerCooling` - Doppler limit
- [ ] `SisyphusCooling` - Sub-Doppler
- [ ] `MagnetoOpticalTrap` - MOT
- [ ] `OpticalDipoleTrap` - Far-detuned trap
- [ ] `OpticalLattice` - Periodic potential
- [ ] `EvaporativeCooling` - BEC route

#### 9.5 Ultracold Atoms
- [ ] `BoseEinsteinCondensate` - Macroscopic occupation
- [ ] `GrossPitaevskii` - GPE dynamics
- [ ] `FermiGas` - Degenerate Fermi gas
- [ ] `FeshbachResonance` - Tunable interactions
- [ ] `BCSBECCrossover` - Pairing crossover
- [ ] `QuantumSimulator` - Lattice models

#### 9.6 Molecular Physics
- [ ] `MolecularOrbital` - LCAO-MO
- [ ] `BornOppenheimer` - Adiabatic separation
- [ ] `PotentialEnergySurface` - PES
- [ ] `VibrationalSpectrum` - IR/Raman
- [ ] `RotationalSpectrum` - Microwave
- [ ] `FranckCondon` - Vibronic transitions

---

### Phase 10: Plasma & Astrophysics

#### 10.1 Plasma Fundamentals
- [ ] `DebyeLength` - Screening distance
- [ ] `PlasmaFrequency` - ωp oscillations
- [ ] `PlasmaParameter` - Coupling strength
- [ ] `VlasovEquation` - Collisionless kinetics
- [ ] `LandauDamping` - Collisionless damping

#### 10.2 Magnetohydrodynamics
- [ ] `MHDEquations` - Ideal MHD system
- [ ] `AlfvenWave` - B-field wave
- [ ] `Magnetosonic` - Fast/slow waves
- [ ] `MHDInstability` - Kink, sausage
- [ ] `MagneticReconnection` - Topology change

#### 10.3 Fusion
- [ ] `LawsonCriterion` - Breakeven condition
- [ ] `TokamakEquilibrium` - Grad-Shafranov
- [ ] `MirrorTrap` - Magnetic mirror
- [ ] `ICFCapsule` - Implosion basics

#### 10.4 Astrophysics
- [ ] `HydrostaticStar` - Stellar structure
- [ ] `LaneEmden` - Polytropic stars
- [ ] `StellarEvolution` - HR diagram tracks
- [ ] `WhiteDwarf` - Electron degeneracy
- [ ] `NeutronStar` - EOS, mass-radius
- [ ] `AccretionDisk` - α-disk model
- [ ] `JetLaunching` - MHD jets basics
- [ ] `Nucleosynthesis` - BBN, stellar

---

### Phase 11: Particle & Nuclear Physics

#### 11.1 Scattering Theory
- [ ] `PartialWave` - Partial wave expansion
- [ ] `ScatteringAmplitude` - f(θ)
- [ ] `OpticalTheorem` - σtot from Im f(0)
- [ ] `RutherfordScattering` - Coulomb scattering
- [ ] `MottScattering` - Relativistic Coulomb

#### 11.2 Nuclear Structure
- [ ] `LiquidDropModel` - SEMF binding energy
- [ ] `ShellModel` - Magic numbers
- [ ] `WoodsSaxon` - Mean-field potential
- [ ] `NuclearRadius` - R = r₀A^(1/3)
- [ ] `NuclearSpin` - Angular momentum

#### 11.3 Radioactivity
- [ ] `AlphaDecay` - Gamow tunneling
- [ ] `BetaDecay` - Fermi theory
- [ ] `GammaDecay` - EM transitions
- [ ] `DecayChain` - Bateman equations
- [ ] `HalfLife` - Decay statistics

#### 11.4 Nuclear Reactions
- [ ] `CrossSection` - σ(E) calculations
- [ ] `QValue` - Energy release
- [ ] `ResonanceFormula` - Breit-Wigner
- [ ] `CompoundNucleus` - Statistical model
- [ ] `FissionYield` - Mass distribution
- [ ] `FusionRate` - Gamow peak

#### 11.5 Particle Physics Basics
- [ ] `DiracEquation` - Relativistic electron
- [ ] `KleinGordonEquation` - Spin-0
- [ ] `DiracSpinor` - 4-component spinor
- [ ] `GammaMatrices` - Clifford algebra
- [ ] `NeutrinoOscillation` - PMNS mixing
- [ ] `QuarkModel` - Hadron spectroscopy

---

### Phase 12: Computational Physics Tools

#### 12.1 Monte Carlo Methods
- [ ] `ImportanceSampling` - Variance reduction
- [ ] `MarkovChainMC` - General MCMC
- [ ] `PathIntegralMC` - Quantum Monte Carlo
- [ ] `MonteCarloIntegration` - High-dimensional integrals

#### 12.2 Molecular Dynamics
- [ ] `VelocityVerlet` - Symplectic integrator
- [ ] `Thermostat` - Nosé-Hoover, Langevin
- [ ] `Barostat` - Pressure control
- [ ] `PeriodicBoundary` - PBC handling
- [ ] `CellList` - Neighbor finding
- [ ] `EwaldSum` - Long-range electrostatics

#### 12.3 PDE Solvers
- [ ] `FiniteDifference1D` - 1D grid methods
- [ ] `FiniteDifference2D` - 2D grid methods
- [ ] `FiniteElement1D` - FEM basics
- [ ] `SpectralMethod` - FFT-based
- [ ] `CrankNicolson` - Implicit time stepping
- [ ] `ADIMethod` - Alternating direction implicit

#### 12.4 Linear Algebra
- [ ] `SparseMatrix` - Sparse storage
- [ ] `ConjugateGradient` - CG solver
- [ ] `GMRES` - Generalized minimal residual
- [ ] `EigenSolver` - Large sparse eigenvalues
- [ ] `SVD` - Singular value decomposition wrapper

---

### Phase 13: Quantum Information & Computing

#### 13.1 Qubits & Gates
- [ ] `Qubit` - Single qubit state
- [ ] `PauliGates` - X, Y, Z gates
- [ ] `HadamardGate` - H gate
- [ ] `PhaseGate` - S, T gates
- [ ] `CNOTGate` - Two-qubit entangling
- [ ] `ToffoliGate` - Three-qubit
- [ ] `UniversalGateSet` - Gate decomposition

#### 13.2 Quantum Circuits
- [ ] `QuantumCircuit` - Circuit representation
- [ ] `CircuitSimulator` - State vector simulation
- [ ] `MeasurementBackend` - Born rule sampling
- [ ] `DensityMatrixSimulator` - Mixed state simulation

#### 13.3 Quantum Algorithms
- [ ] `GroverSearch` - Amplitude amplification
- [ ] `DeutschJozsa` - Deterministic query
- [ ] `QuantumFourierTransform` - QFT
- [ ] `PhaseEstimation` - Eigenvalue estimation
- [ ] `VQE` - Variational quantum eigensolver

#### 13.4 Error Correction
- [ ] `BitFlipCode` - 3-qubit code
- [ ] `PhaseFlipCode` - Phase error
- [ ] `ShorCode` - 9-qubit code
- [ ] `SteaneCode` - 7-qubit CSS
- [ ] `SurfaceCode` - Topological code basics

#### 13.5 Entanglement Measures
- [ ] `VonNeumannEntropy` - S(ρ) = -Tr(ρ log ρ)
- [ ] `Concurrence` - Two-qubit entanglement
- [ ] `Negativity` - Entanglement witness
- [ ] `MutualInformation` - Correlations

---

## Implementation Priority & Phases

### Immediate Priority (Foundation)
1. Phase 1: Core Mechanics Extensions
2. Phase 4: Statistical Mechanics (missing ensembles, distributions)
3. Phase 5.1-5.4: Quantum Mechanics operators & systems

### Medium Priority (Domain Expansion)
4. Phase 2: Electromagnetism deep dive
5. Phase 3: Waves & Optics
6. Phase 8: Condensed Matter basics

### Advanced Topics
7. Phase 6: General Relativity
8. Phase 7: QFT foundations
9. Phase 9: AMO physics
10. Phase 10-11: Plasma, Astro, Nuclear
11. Phase 12-13: Computational tools, Quantum computing

---

## Estimated Scope

| Phase | New Classes | Priority |
|-------|------------|----------|
| 1. Mechanics Extensions | ~25 | HIGH |
| 2. Electromagnetism | ~30 | HIGH |
| 3. Waves & Optics | ~35 | HIGH |
| 4. Thermo & Stat Mech | ~40 | HIGH |
| 5. Quantum Complete | ~50 | HIGH |
| 6. Relativity | ~30 | MEDIUM |
| 7. QFT Foundations | ~20 | MEDIUM |
| 8. Condensed Matter | ~60 | MEDIUM |
| 9. AMO | ~45 | MEDIUM |
| 10. Plasma & Astro | ~25 | LOW |
| 11. Nuclear & Particle | ~25 | LOW |
| 12. Computational | ~20 | LOW |
| 13. Quantum Computing | ~25 | LOW |

**Total new primitives: ~430 classes**

---

## Design Principles for Implementation

### 1. Inheritance Hierarchy
```
BaseClass
├── BaseSolver
├── BaseProcess
└── PhysicalSystem
    ├── DynamicalSystem
    │   └── [mechanics, fluids]
    ├── QuantumSystem
    │   └── [QM, QFT basics]
    ├── ThermodynamicSystem
    │   └── [thermo, stat mech]
    ├── FieldSystem
    │   └── [EM, gravity, gauge]
    └── StatisticalSystem
        └── [ensembles, lattice models]
```

### 2. Composability
- Forces compose additively
- Fields superpose
- Operators compose via tensor products
- Systems couple through interaction terms

### 3. Validation at Boundaries
- Physical parameter validation in `__init__`
- Dimension/unit consistency checks
- Conservation law monitoring

### 4. History Tracking
- All time-dependent systems track state history
- Configurable history depth
- Efficient numpy array storage

### 5. Numerical Backends
- Default: NumPy vectorized
- Optional: Numba JIT for hot loops
- Optional: CuPy/JAX for GPU (future)
