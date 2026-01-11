# SciForge Experiment Catalog

This document tracks all physics experiments implemented in the SciForge library.
Each experiment follows the pattern: **model → simulate → compare to analytic/approx → plot + metrics**

## Status Legend
- [x] Implemented and verified
- [ ] Not yet implemented

---

## 1) Foundations & Numerical Methods (1-25)

### ODE / Integrators (1-10)
- [ ] 1. Euler vs RK2 vs RK4 vs adaptive RK45 on harmonic oscillator
- [ ] 2. Symplectic Euler/Verlet vs RK4 on oscillator (energy drift)
- [ ] 3. Long-time integration stability: pendulum for 10^6 steps
- [ ] 4. Stiff ODE demo: RC circuit (explicit vs implicit)
- [ ] 5. Event detection: bouncing ball with coefficient of restitution
- [ ] 6. Chaotic ODE sensitivity: Lorenz system divergence
- [ ] 7. Constraint stabilization: pendulum as constraint
- [ ] 8. Hamiltonian system: symplectic vs non-symplectic on Kepler
- [ ] 9. Shooting method: brachistochrone boundary value problem
- [ ] 10. Symmetry/invariant check: momentum/energy error trackers

### PDE / Discretizations (11-18)
- [ ] 11. 1D diffusion equation: explicit vs implicit (CFL stability)
- [ ] 12. 1D wave equation: finite difference vs spectral (dispersion)
- [ ] 13. 2D Poisson equation: Gauss-Seidel vs multigrid
- [ ] 14. Advection equation: upwind vs Lax-Friedrichs vs Godunov
- [ ] 15. Burgers' equation: shock formation + viscosity
- [ ] 16. Heat equation on irregular domain (finite elements)
- [ ] 17. Spectral method: Fourier solution of periodic diffusion
- [ ] 18. PDE conservation test: continuity equation mass conservation

### Fourier / Signals (19-21)
- [ ] 19. FFT basics: reconstruct signal, windowing, spectral leakage
- [ ] 20. Convolution theorem: blur kernel spatial vs frequency
- [ ] 21. Power spectral density of noisy oscillator (Welch)

### Monte Carlo / Randomness (22-25)
- [ ] 22. CLT: sum of random vars → Gaussian
- [ ] 23. Random walk → diffusion constant measurement
- [ ] 24. Monte Carlo integration: estimate π / sphere volume
- [ ] 25. MCMC: sample from Boltzmann distribution

---

## 2) Classical Mechanics (26-70)

### Kinematics / Newton (26-34)
- [ ] 26. Constant acceleration: analytic vs numeric
- [ ] 27. Projectile motion: vacuum vs linear vs quadratic drag
- [ ] 28. Terminal velocity: vertical fall with drag
- [ ] 29. Rocket equation (Tsiolkovsky): mass loss + thrust
- [ ] 30. Simple harmonic oscillator: exact vs numeric phase drift
- [ ] 31. Damped oscillator regimes: under/critical/over-damped
- [ ] 32. Driven oscillator: resonance curves, Q factor, phase lag
- [ ] 33. Bead on rotating hoop: stability bifurcation
- [ ] 34. Atwood machine with pulley inertia

### Energy / Potential Landscapes (35-38)
- [ ] 35. Conservative vs non-conservative forces: energy with friction
- [ ] 36. Effective potential: central force orbits classification
- [ ] 37. Escape velocity in Newtonian gravity
- [ ] 38. Small oscillations: Hessian → normal modes

### Collisions (39-42)
- [ ] 39. 1D elastic collision: momentum/energy conservation
- [ ] 40. 1D inelastic collision: energy loss vs restitution
- [ ] 41. 2D billiards: specular reflection, ergodic behavior
- [ ] 42. Gas of hard spheres: Maxwell speed distribution

### Pendula Family (43-48)
- [ ] 43. Simple pendulum: exact vs small-angle
- [ ] 44. Physical pendulum: varying moment of inertia
- [ ] 45. Damped pendulum: decay envelopes
- [ ] 46. Driven damped pendulum: route to chaos (Poincare)
- [ ] 47. Double pendulum: chaotic divergence
- [ ] 48. Coupled pendula: normal modes + beats

### Orbits / Gravitation (49-53)
- [ ] 49. Two-body Kepler: ellipse/hyperbola, Runge-Lenz vector
- [ ] 50. Precession from 1/r^3 perturbation (toy GR)
- [ ] 51. Restricted three-body: Lagrange points stability
- [ ] 52. N-body gravity: leapfrog vs RK energy error
- [ ] 53. Hill sphere for satellite capture

### Rotations / Rigid Bodies (54-58)
- [ ] 54. Tennis racket theorem: intermediate axis instability
- [ ] 55. Gyroscope precession: steady rate vs torque
- [ ] 56. Rolling without slipping: cylinder energy partition
- [ ] 57. Spinning top with friction: nutation damping
- [ ] 58. Moment of inertia: numerical integration

### Variational Mechanics (59-63)
- [ ] 59. Brachistochrone: optimization vs cycloid
- [ ] 60. Fermat principle: least action trajectory
- [ ] 61. Lagrangian EOM derivation and comparison
- [ ] 62. Noether: time-translation → energy conservation
- [ ] 63. Noether: rotation → angular momentum conservation

### Chaos / Nonlinear Dynamics (64-67)
- [ ] 64. Logistic map: bifurcation diagram, Feigenbaum constant
- [ ] 65. Duffing oscillator: hysteresis + chaos
- [ ] 66. Kicked rotor: transition to chaos
- [ ] 67. FPU chain: recurrence phenomenon

### Continuum / Fluids (68-70)
- [ ] 68. 1D string vibration: modes + Fourier series
- [ ] 69. 2D membrane eigenmodes (drum)
- [ ] 70. Shallow water solitons (KdV toy)

---

## 3) Waves & Acoustics (71-79)

- [ ] 71. Wave equation on string: fixed/free boundary reflection
- [ ] 72. Superposition: interference of two pulses
- [ ] 73. Standing waves: nodes/antinodes vs frequency
- [ ] 74. Dispersion: wave packet spreading
- [ ] 75. Doppler effect: moving source vs observer
- [ ] 76. Beats: close-frequency interference
- [ ] 77. Acoustic resonance: open-open vs open-closed
- [ ] 78. Impedance mismatch: reflection/transmission coefficients
- [ ] 79. Fourier synthesis: square wave + Gibbs phenomenon

---

## 4) Electromagnetism (80-100)

### Electrostatics (80-85)
- [ ] 80. Field of point charges: vector field + equipotentials
- [ ] 81. Dipole field + far-field multipole scaling
- [ ] 82. Gauss's law: numerically integrate flux
- [ ] 83. Capacitor: parallel plate edge effects (2D Laplace)
- [ ] 84. Method of images: charge near grounded plane
- [ ] 85. Energy in E-field: density and total energy

### Magnetostatics (86-88)
- [ ] 86. Current loop field (Biot-Savart)
- [ ] 87. Solenoid field profile: finite-length end effects
- [ ] 88. Ampere's law verification

### Electrodynamics (89-95)
- [ ] 89. Faraday induction: emf vs time
- [ ] 90. Mutual inductance between coils
- [ ] 91. Motional EMF: moving conductor in B-field
- [ ] 92. EM wave equation from Maxwell + 1D propagation
- [ ] 93. Dipole radiation pattern
- [ ] 94. Larmor radiation power vs acceleration
- [ ] 95. Synchrotron radiation spectrum

### Circuits (96-98)
- [ ] 96. RC transient
- [ ] 97. RL transient
- [ ] 98. RLC resonance: frequency and phase response

### Waveguides & Cavities (99-100)
- [ ] 99. Waveguide modes: TE/TM cutoff frequencies
- [ ] 100. Cavity resonator: resonant frequencies

---

## 5) Optics & Photonics (101-119)

### Geometric Optics (101-105)
- [ ] 101. Snell's law ray tracing through layers
- [ ] 102. Total internal reflection & critical angle
- [ ] 103. Thin lens ray tracing + focal length
- [ ] 104. Spherical aberration in simple lens
- [ ] 105. GRIN lens ray trajectories

### Wave Optics (106-110)
- [ ] 106. Single-slit diffraction: sinc^2 pattern
- [ ] 107. Double-slit interference: fringe spacing
- [ ] 108. Circular aperture (Airy disk)
- [ ] 109. Diffraction grating: angular peaks
- [ ] 110. Fresnel diffraction: near-field patterns

### Interferometry / Coherence (111-113)
- [ ] 111. Michelson interferometer: fringe shift
- [ ] 112. Fabry-Perot cavity: transmission vs finesse
- [ ] 113. Coherence length: visibility vs bandwidth

### Polarization (114-116)
- [ ] 114. Brewster's angle: p-polarization zero reflection
- [ ] 115. Wave plates: Jones calculus
- [ ] 116. Birefringence: phase retardation

### Nonlinear & Laser Optics (117-119)
- [ ] 117. Four-wave mixing: phase conjugation
- [ ] 118. Kerr lens: self-focusing
- [ ] 119. Optical solitons: NLSE propagation

---

## 6) Thermodynamics (120-132)

### Equations of State & Cycles (120-124)
- [ ] 120. Ideal gas: PV diagrams isothermal/adiabatic
- [ ] 121. Van der Waals: critical point + Maxwell construction
- [ ] 122. Carnot cycle efficiency
- [ ] 123. Otto/Diesel cycle comparison
- [ ] 124. Refrigeration cycle (reverse Carnot)

### Thermodynamic Relations (125-129)
- [ ] 125. Maxwell relations verification
- [ ] 126. Joule expansion: entropy change
- [ ] 127. Clapeyron equation: phase boundary slope
- [ ] 128. Entropy of mixing
- [ ] 129. Blackbody radiation: Wien + Stefan-Boltzmann

### Transport (130-132)
- [ ] 130. Heat conduction: steady-state vs transient
- [ ] 131. Newton's law of cooling
- [ ] 132. Diffusion equation: Gaussian spreading

---

## 7) Statistical Mechanics (133-148)

### Core Distributions (133-136)
- [ ] 133. Maxwell-Boltzmann speed distribution
- [ ] 134. Equipartition via MD
- [ ] 135. Fermi-Dirac occupation vs temperature
- [ ] 136. Bose-Einstein distribution, BEC onset

### Ising / Lattice Models (137-141)
- [ ] 137. 2D Ising Metropolis: magnetization vs T
- [ ] 138. 2D Ising: susceptibility peak, T_c estimate
- [ ] 139. 2D Ising: correlation length vs T
- [ ] 140. Cluster algorithms vs Metropolis near T_c
- [ ] 141. Finite-size scaling: critical exponents

### Random Processes (142-144)
- [ ] 142. Percolation threshold estimation
- [ ] 143. DLA (diffusion-limited aggregation)
- [ ] 144. KPZ growth

### Fluctuation Theorems (145-146)
- [ ] 145. Jarzynski equality
- [ ] 146. Crooks fluctuation theorem

### Kinetic Theory (147-148)
- [ ] 147. Boltzmann equation: BGK relaxation
- [ ] 148. H-theorem: entropy evolution

---

## 8) Quantum Mechanics (149-183)

### Wave Mechanics (149-155)
- [ ] 149. Particle in box: eigenstates + time evolution
- [ ] 150. Finite square well: bound states vs depth
- [ ] 151. Step potential: reflection/transmission
- [ ] 152. Barrier tunneling: transmission vs width/height
- [ ] 153. WKB approximation vs exact tunneling
- [ ] 154. Gaussian wavepacket spreading
- [ ] 155. Wavepacket scattering off barrier

### Harmonic Oscillator (156-158)
- [ ] 156. SHO eigenstates: ladder operators
- [ ] 157. Coherent states: minimal uncertainty
- [ ] 158. Commutator uncertainty relations

### Hydrogen & Angular Momentum (159-161)
- [ ] 159. Hydrogen radial wavefunctions
- [ ] 160. Selection rules via dipole matrix elements
- [ ] 161. Zeeman splitting vs B-field

### Spin & Two-Level Systems (162-165)
- [ ] 162. Spin-1/2 Larmor precession
- [ ] 163. Rabi oscillations with detuning
- [ ] 164. Adiabatic theorem demonstration
- [ ] 165. Landau-Zener transition

### Measurement & Interference (166-168)
- [ ] 166. Double-slit from two-path superposition
- [ ] 167. Which-path decoherence: visibility vs dephasing
- [ ] 168. Heisenberg uncertainty: minimum wavepacket

### Approximation Methods (169-171)
- [ ] 169. Time-independent perturbation: anharmonic oscillator
- [ ] 170. Stark effect: hydrogen in electric field
- [ ] 171. Variational method: quartic potential ground state

### Many-Body / Identical Particles (172-174)
- [ ] 172. Two-fermion antisymmetrization
- [ ] 173. Exchange energy in two-electron model
- [ ] 174. Tight-binding chain (second-quantized)

### Open Quantum Systems (175-178)
- [ ] 175. Density matrix: pure vs mixed states
- [ ] 176. Lindblad dephasing evolution
- [ ] 177. T1/T2 on Bloch sphere
- [ ] 178. Caldeira-Leggett decoherence

### Quantum Information (179-183)
- [ ] 179. Bell inequality (CHSH) violation
- [ ] 180. Quantum teleportation
- [ ] 181. Grover search
- [ ] 182. Quantum phase estimation
- [ ] 183. Shor's algorithm (simplified)

---

## 9) Special Relativity (184-190)

- [ ] 184. Lorentz transformation: simultaneity shift
- [ ] 185. Time dilation: muon lifetime
- [ ] 186. Length contraction
- [ ] 187. Relativistic velocity addition
- [ ] 188. Relativistic Doppler + aberration
- [ ] 189. Energy-momentum: threshold energies
- [ ] 190. Constant proper acceleration worldline

---

## 10) General Relativity & Cosmology (191-203)

### Schwarzschild & Kerr (191-196)
- [ ] 191. Twin paradox visualization
- [ ] 192. Schwarzschild geodesics: perihelion precession
- [ ] 193. Light bending: deflection vs impact parameter
- [ ] 194. Shapiro delay vs closest approach
- [ ] 195. Gravitational redshift vs radius
- [ ] 196. ISCO radius calculation
- [ ] 197. Frame dragging (Kerr)

### Gravitational Waves (198-200)
- [ ] 198. Binary inspiral chirp frequency
- [ ] 199. GW strain h(t) for inspiral
- [ ] 200. Matched filtering: detect chirp in noise

### Cosmology (201-203)
- [ ] 201. Friedmann equations: a(t) for different eras
- [ ] 202. Distance-redshift relations
- [ ] 203. CMB acoustic peaks (simplified)

---

## 11) Particle / Nuclear / QFT (204-219)

### Scattering & Decays (204-208)
- [ ] 204. Rutherford scattering angular distribution
- [ ] 205. Partial wave expansion
- [ ] 206. Breit-Wigner resonance
- [ ] 207. Two-body decay kinematics
- [ ] 208. Cross section vs energy

### Nuclear (209-212)
- [ ] 209. Radioactive decay chains (Bateman)
- [ ] 210. Binding energy curve (SEMF)
- [ ] 211. Gamow tunneling factor
- [ ] 212. Neutron moderation random walk

### QFT Toys (213-219)
- [ ] 213. Klein-Gordon 1D wavepackets
- [ ] 214. Dirac equation: Zitterbewegung
- [ ] 215. Phi^4 kink solutions
- [ ] 216. Lattice field theory energy conservation
- [ ] 217. Block-spin RG on Ising
- [ ] 218. Running coupling: beta function
- [ ] 219. Path integral Monte Carlo

---

## 12) Condensed Matter (220-245)

### Band Structure (220-225)
- [ ] 220. Tight-binding 1D chain
- [ ] 221. Kronig-Penney: band gaps
- [ ] 222. Finite chain: edge states
- [ ] 223. 2D square lattice: Fermi surface
- [ ] 224. Graphene: Dirac cones
- [ ] 225. DOS in 1D/2D/3D

### Semiconductors (226-228)
- [ ] 226. pn junction: depletion width
- [ ] 227. Drift-diffusion: I-V curve
- [ ] 228. Drude conductivity
- [ ] 229. Classical Hall effect
- [ ] 230. Anderson localization

### Phonons / Heat (231-233)
- [ ] 231. MOS capacitor: C-V curve
- [ ] 232. 1D phonon dispersion
- [ ] 233. Debye heat capacity vs T
- [ ] 234. Thermal conductivity via phonon MFP

### Magnetism (235-237)
- [ ] 235. Heisenberg model Monte Carlo
- [ ] 236. XY model: vortex unbinding
- [ ] 237. Hysteresis toy model

### Superconductivity (238-242)
- [ ] 238. Ginzburg-Landau vortices
- [ ] 239. London penetration depth
- [ ] 240. Josephson junction: I-phase relation
- [ ] 241. BCS gap vs temperature
- [ ] 242. Integer QHE edge states

### Topological Phases (243-245)
- [ ] 243. SSH model: Zak phase + edge states
- [ ] 244. Berry curvature integration
- [ ] 245. Chern insulator: Chern number

### Correlated Systems (246-248)
- [ ] 246. Hubbard model: Mott gap
- [ ] 247. t-J model on small lattice
- [ ] 248. Moire flat bands

---

## 13) AMO / Quantum Optics (249-258)

- [ ] 249. Optical Bloch equations
- [ ] 250. Saturation / power broadening
- [ ] 251. Doppler vs natural linewidth
- [ ] 252. Laser cooling force (optical molasses)
- [ ] 253. MOT restoring force
- [ ] 254. Atom interferometer phase shift
- [ ] 255. Ramsey spectroscopy linewidth
- [ ] 256. Spin squeezing
- [ ] 257. Jaynes-Cummings: Rabi splitting
- [ ] 258. Frequency comb generation

---

## 14) Plasma & Fusion (259-267)

- [ ] 259. Debye shielding profile
- [ ] 260. Plasma frequency oscillations
- [ ] 261. Langmuir wave dispersion
- [ ] 262. Two-stream instability
- [ ] 263. Landau damping
- [ ] 264. Alfven waves in MHD
- [ ] 265. Magnetic reconnection (tearing mode)
- [ ] 266. Tokamak guiding center motion
- [ ] 267. Lawson criterion visualization

---

## 15) Astrophysics (268-274)

- [ ] 268. Hydrostatic equilibrium: polytropes
- [ ] 269. Virial theorem in N-body
- [ ] 270. Main-sequence scaling relations
- [ ] 271. Radiative transfer: optical depth
- [ ] 272. Accretion disk temperature profile
- [ ] 273. Eddington luminosity
- [ ] 274. Orbital decay from GW emission

---

## Summary Statistics

| Domain | Range | Count | Implemented |
|--------|-------|-------|-------------|
| Foundations/Numerical | 1-25 | 25 | 0/25 |
| Classical Mechanics | 26-70 | 45 | 0/45 |
| Waves & Acoustics | 71-79 | 9 | 0/9 |
| Electromagnetism | 80-100 | 21 | 0/21 |
| Optics | 101-119 | 19 | 0/19 |
| Thermodynamics | 120-132 | 13 | 0/13 |
| Statistical Mechanics | 133-148 | 16 | 0/16 |
| Quantum Mechanics | 149-183 | 35 | 0/35 |
| Special Relativity | 184-190 | 7 | 0/7 |
| General Relativity | 191-203 | 13 | 0/13 |
| Particle/Nuclear/QFT | 204-219 | 16 | 0/16 |
| Condensed Matter | 220-248 | 29 | 0/29 |
| AMO/Quantum Optics | 249-258 | 10 | 0/10 |
| Plasma & Fusion | 259-267 | 9 | 0/9 |
| Astrophysics | 268-274 | 7 | 0/7 |
| **Total** | 1-274 | **274** | **0/274** |

---

*Last updated: 2026-01-11*
