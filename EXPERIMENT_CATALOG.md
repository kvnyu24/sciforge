# SciForge Experiment Catalog

This document tracks all physics experiments implemented in the SciForge library.
Each experiment follows the pattern: **model → simulate → compare to analytic/approx → plot + metrics**

## Status Legend
- [x] Implemented and verified
- [ ] Not yet implemented

---

## 1) Foundations & Numerical Methods (1-25)

### ODE / Integrators (1-10)
- [x] 1. Euler vs RK2 vs RK4 vs adaptive RK45 on harmonic oscillator → `numerical/integrator_comparison.py`
- [x] 2. Symplectic Euler/Verlet vs RK4 on oscillator (energy drift) → `numerical/symplectic_integrators.py`
- [x] 3. Long-time integration stability: pendulum for 10^6 steps → `numerical/long_time_stability.py`
- [x] 4. Stiff ODE demo: RC circuit (explicit vs implicit) → `numerical/stiff_ode_demo.py`
- [x] 5. Event detection: bouncing ball with coefficient of restitution → `numerical/bouncing_ball.py`
- [x] 6. Chaotic ODE sensitivity: Lorenz system divergence → `numerical/lorenz_sensitivity.py`
- [x] 7. Constraint stabilization: pendulum as constraint → `numerical/constraint_pendulum.py`
- [x] 8. Hamiltonian system: symplectic vs non-symplectic on Kepler → `numerical/kepler_symplectic.py`
- [x] 9. Shooting method: brachistochrone boundary value problem → `numerical/brachistochrone.py`
- [x] 10. Symmetry/invariant check: momentum/energy error trackers → `numerical/conservation_tracker.py`

### PDE / Discretizations (11-18)
- [x] 11. 1D diffusion equation: explicit vs implicit (CFL stability) → `numerical/diffusion_explicit_implicit.py`
- [x] 12. 1D wave equation: finite difference vs spectral (dispersion) → `numerical/wave_dispersion.py`
- [x] 13. 2D Poisson equation: Gauss-Seidel vs multigrid → `numerical/poisson_solvers.py`, `numerical/multigrid_vcycle.py`
- [x] 14. Advection equation: upwind vs Lax-Friedrichs vs Godunov → `numerical/advection_schemes.py`
- [x] 15. Burgers' equation: shock formation + viscosity → `numerical/burgers_shock.py`
- [x] 16. Heat equation / multigrid → `numerical/multigrid_vcycle.py`
- [x] 17. Spectral method: Fourier solution of periodic diffusion → `numerical/spectral_diffusion.py`
- [x] 18. PDE conservation test: continuity equation mass conservation → `numerical/continuity_conservation.py`

### Fourier / Signals (19-21)
- [x] 19. FFT basics: reconstruct signal, windowing, spectral leakage → `numerical/fft_basics.py`
- [x] 20. Convolution theorem: blur kernel spatial vs frequency → `numerical/convolution_theorem.py`
- [x] 21. Power spectral density of noisy oscillator (Welch) → `numerical/psd_noisy_oscillator.py`

### Monte Carlo / Randomness (22-25)
- [x] 22. CLT: sum of random vars → Gaussian → `numerical/clt_demo.py`
- [x] 23. Random walk → diffusion constant measurement → `numerical/random_walk_diffusion.py`
- [x] 24. Monte Carlo integration: estimate π / sphere volume → `numerical/monte_carlo_integration.py`
- [x] 25. MCMC: sample from Boltzmann distribution → `numerical/mcmc_sampling.py`

---

## 2) Classical Mechanics (26-70)

### Kinematics / Newton (26-34)
- [x] 26. Constant acceleration: analytic vs numeric → `mechanics/constant_acceleration.py`
- [x] 27. Projectile motion: vacuum vs linear vs quadratic drag → `mechanics/projectile_drag.py`
- [x] 28. Terminal velocity: vertical fall with drag → `mechanics/terminal_velocity.py`
- [x] 29. Rocket equation (Tsiolkovsky): mass loss + thrust → `mechanics/rocket_equation.py`
- [x] 30. Simple harmonic oscillator: exact vs numeric phase drift → `mechanics/pendulum_comparison.py`
- [x] 31. Damped oscillator regimes: under/critical/over-damped → `oscillations/damped_oscillator.py`
- [x] 32. Driven oscillator: resonance curves, Q factor, phase lag → `oscillations/driven_resonance.py`
- [x] 33. Bead on rotating hoop: stability bifurcation → `mechanics/bead_on_hoop.py`
- [x] 34. Atwood machine with pulley inertia → `mechanics/atwood_machine.py`

### Energy / Potential Landscapes (35-38)
- [x] 35. Conservative vs non-conservative forces: energy with friction → `mechanics/conservative_forces.py`
- [x] 36. Effective potential: central force orbits classification → `mechanics/effective_potential.py`
- [x] 37. Escape velocity in Newtonian gravity → `mechanics/escape_velocity.py`
- [x] 38. Small oscillations: Hessian → normal modes → `mechanics/normal_modes.py`

### Collisions (39-42)
- [x] 39. 1D elastic collision: momentum/energy conservation → `mechanics/collision_1d_elastic.py`
- [x] 40. 1D inelastic collision: energy loss vs restitution → `mechanics/collision_1d_inelastic.py`
- [x] 41. 2D billiards: specular reflection, ergodic behavior → `mechanics/billiards_2d.py`
- [x] 42. Gas of hard spheres: Maxwell speed distribution → `mechanics/hard_sphere_gas.py`

### Pendula Family (43-48)
- [x] 43. Simple pendulum: exact vs small-angle → `mechanics/pendulum_comparison.py`
- [x] 44. Physical pendulum: varying moment of inertia → `mechanics/physical_pendulum.py`
- [x] 45. Damped pendulum: decay envelopes → `oscillations/damped_oscillator.py`
- [x] 46. Driven damped pendulum: route to chaos (Poincare) → `mechanics/driven_pendulum_chaos.py`
- [x] 47. Double pendulum: chaotic divergence → `chaos/double_pendulum.py`
- [x] 48. Coupled pendula: normal modes + beats → `mechanics/coupled_pendula.py`

### Orbits / Gravitation (49-53)
- [x] 49. Two-body Kepler: ellipse/hyperbola, Runge-Lenz vector → `mechanics/kepler_problem.py`
- [x] 50. Precession from 1/r^3 perturbation (toy GR) → `mechanics/precession_perturbation.py`
- [x] 51. Restricted three-body: Lagrange points stability → `mechanics/lagrange_points.py`
- [x] 52. N-body gravity: leapfrog vs RK energy error → `mechanics/n_body_gravity.py`
- [x] 53. Hill sphere for satellite capture → `mechanics/hill_sphere.py`

### Rotations / Rigid Bodies (54-58)
- [x] 54. Tennis racket theorem: intermediate axis instability → `mechanics/tennis_racket_theorem.py`
- [x] 55. Gyroscope precession: steady rate vs torque → `mechanics/gyroscope_precession.py`
- [x] 56. Rolling without slipping: cylinder energy partition → `mechanics/rolling_without_slipping.py`
- [x] 57. Spinning top with friction: nutation damping → `mechanics/spinning_top_friction.py`
- [x] 58. Moment of inertia: numerical integration → `mechanics/moment_of_inertia_numerical.py`

### Additional Mechanics
- [x] 56b. Coriolis deflection → `mechanics/coriolis_deflection.py`
- [x] 57b. Foucault pendulum → `mechanics/foucault_pendulum.py`
- [x] 58b. Tidal locking → `mechanics/tidal_locking.py`

### Variational Mechanics (59-63)
- [x] 59. Brachistochrone: optimization vs cycloid → `numerical/brachistochrone.py`
- [x] 60. Fermat principle: least action trajectory → `mechanics/fermat_principle.py`
- [x] 61. Lagrangian EOM derivation and comparison → `mechanics/lagrangian_eom.py`
- [x] 61b. Poisson brackets → `mechanics/poisson_brackets.py`
- [x] 62. Noether: time-translation → energy conservation → `mechanics/noether_energy.py`
- [x] 62b. Action-angle variables → `mechanics/action_angle.py`
- [x] 63. Noether: rotation → angular momentum conservation → `mechanics/noether_angular_momentum.py`
- [x] 63b. Canonical transformations → `mechanics/canonical_transforms.py`
- [x] 64b. Liouville theorem → `mechanics/liouville_theorem.py`

### Chaos / Nonlinear Dynamics (64-67)
- [x] 64. Logistic map: bifurcation diagram, Feigenbaum constant → `chaos/logistic_map.py`
- [x] 65. Duffing oscillator: hysteresis + chaos → `chaos/duffing_oscillator.py`
- [x] 66. Kicked rotor: transition to chaos → `chaos/kicked_rotor.py`
- [x] 67. FPU chain: recurrence phenomenon → `chaos/fpu_chain.py`

### Continuum / Fluids (68-70)
- [x] 68. 1D string vibration: modes + Fourier series → `mechanics/spring_chain.py`, `waves/string_vibration_modes.py`
- [x] 69. 2D membrane eigenmodes (drum) → `waves/membrane_eigenmodes.py`
- [x] 70. Shallow water solitons (KdV toy) → `waves/kdv_solitons.py`
- [x] 70b. Catenary curve → `mechanics/catenary.py`

---

## 3) Waves & Acoustics (71-79)

- [x] 71. Wave equation on string: fixed/free boundary reflection → `waves/wave_boundary_reflection.py`
- [x] 72. Superposition: interference of two pulses → `waves/pulse_superposition.py`, `waves/wave_interference.py`
- [x] 73. Standing waves: nodes/antinodes vs frequency → `waves/standing_waves.py`
- [x] 74. Dispersion: wave packet spreading → `waves/wave_dispersion.py`
- [x] 75. Doppler effect: moving source vs observer → `waves/doppler_effect.py`
- [x] 76. Beats: close-frequency interference → `waves/beats.py`
- [x] 77. Acoustic resonance: open-open vs open-closed → `waves/acoustic_resonance.py`
- [x] 78. Impedance mismatch: reflection/transmission coefficients → `waves/impedance_mismatch.py`
- [x] 79. Fourier synthesis: square wave + Gibbs phenomenon → `waves/fourier_synthesis.py`

---

## 4) Electromagnetism (80-100)

### Electrostatics (80-85)
- [x] 80. Field of point charges: vector field + equipotentials → `fields/point_charge_field.py`
- [x] 81. Dipole field + far-field multipole scaling → `fields/electric_dipole_field.py`, `fields/dipole_multipole_scaling.py`
- [x] 82. Gauss's law: numerically integrate flux → `electromagnetism/gauss_law_flux.py`
- [x] 83. Capacitor: parallel plate edge effects (2D Laplace) → `electromagnetism/capacitor_edge_effects.py`
- [x] 84. Method of images: charge near grounded plane → `electromagnetism/method_of_images.py`
- [x] 85. Energy in E-field: density and total energy → `electromagnetism/energy_in_efield.py`

### Magnetostatics (86-88)
- [x] 86. Current loop field (Biot-Savart) → `fields/current_loop_biot_savart.py`
- [x] 87. Solenoid field profile: finite-length end effects → `electromagnetism/solenoid_field.py`
- [x] 88. Ampere's law verification → `electromagnetism/ampere_law_verification.py`

### Electrodynamics (89-95)
- [x] 89. Faraday induction: emf vs time → `electromagnetism/faraday_induction.py`
- [x] 90. Mutual inductance between coils → `electromagnetism/mutual_inductance.py`
- [x] 91. Motional EMF: moving conductor in B-field → `electromagnetism/motional_emf.py`
- [x] 92. EM wave equation from Maxwell + 1D propagation → `electromagnetism/em_wave_maxwell.py`
- [x] 93. Dipole radiation pattern → `electromagnetism/dipole_radiation_pattern.py`
- [x] 94. Larmor radiation power vs acceleration → `electromagnetism/larmor_radiation.py`
- [x] 95. Synchrotron radiation spectrum → `electromagnetism/synchrotron_radiation.py`

### Circuits (96-98)
- [x] 96. RC transient → `circuits/rc_circuit.py`
- [x] 97. RL transient → `circuits/rl_transient.py`
- [x] 98. RLC resonance: frequency and phase response → `circuits/rlc_resonance.py`

### Waveguides & Cavities (99-100)
- [x] 99. Waveguide modes: TE/TM cutoff frequencies → `electromagnetism/waveguide_modes.py`
- [x] 100. Cavity resonator: resonant frequencies → `electromagnetism/cavity_resonator.py`

---

## 5) Optics & Photonics (101-119)

### Geometric Optics (101-105)
- [x] 101. Snell's law ray tracing through layers → `optics/snells_law_ray_tracing.py`
- [x] 102. Total internal reflection & critical angle → `optics/total_internal_reflection.py`
- [x] 103. Thin lens ray tracing + focal length → `optics/thin_lens_ray_tracing.py`
- [x] 104. Spherical aberration in simple lens → `optics/spherical_aberration.py`
- [x] 105. GRIN lens ray trajectories → `optics/grin_lens_trajectories.py`

### Wave Optics (106-110)
- [x] 106. Single-slit diffraction: sinc^2 pattern → `optics/single_slit_diffraction.py`
- [x] 107. Double-slit interference: fringe spacing → `optics/double_slit_interference.py`
- [x] 108. Circular aperture (Airy disk) → `optics/airy_disk.py`
- [x] 109. Diffraction grating: angular peaks → `emwaves/diffraction_grating.py`
- [x] 110. Fresnel diffraction: near-field patterns → `optics/fresnel_diffraction.py`

### Interferometry / Coherence (111-113)
- [x] 111. Michelson interferometer: fringe shift → `optics/michelson_interferometer.py`
- [x] 112. Fabry-Perot cavity: transmission vs finesse → `optics/fabry_perot_cavity.py`
- [x] 113. Coherence length: visibility vs bandwidth → `optics/coherence_length_visibility.py`

### Polarization (114-116)
- [x] 114. Brewster's angle: p-polarization zero reflection → `optics/brewster_angle.py`
- [x] 115. Wave plates: Jones calculus → `optics/jones_calculus_waveplates.py`
- [x] 116. Birefringence: phase retardation → `optics/birefringence.py`

### Nonlinear & Laser Optics (117-119)
- [x] 117. Four-wave mixing: phase conjugation → `optics/four_wave_mixing.py`
- [x] 118. Kerr lens: self-focusing → `optics/kerr_lens.py`
- [x] 119. Optical solitons: NLSE propagation → `optics/optical_solitons.py`

---

## 6) Thermodynamics (120-132)

### Equations of State & Cycles (120-124)
- [x] 120. Ideal gas: PV diagrams isothermal/adiabatic → `thermodynamics/ideal_gas_pv_diagrams.py`
- [x] 121. Van der Waals: critical point + Maxwell construction → `thermodynamics/van_der_waals_isotherms.py`
- [x] 122. Carnot cycle efficiency → `thermodynamics/carnot_cycle.py`
- [x] 123. Otto/Diesel cycle comparison → `thermodynamics/otto_diesel_cycles.py`
- [x] 124. Refrigeration cycle (reverse Carnot) → `thermodynamics/refrigeration_cycle.py`

### Thermodynamic Relations (125-129)
- [x] 125. Maxwell relations verification → `thermodynamics/maxwell_relations.py`
- [x] 126. Joule expansion: entropy change → `thermodynamics/joule_expansion.py`
- [x] 127. Clapeyron equation: phase boundary slope → `thermodynamics/clapeyron_equation.py`
- [x] 128. Entropy of mixing → `thermodynamics/entropy_of_mixing.py`
- [x] 129. Blackbody radiation: Wien + Stefan-Boltzmann → `thermodynamics/stefan_boltzmann.py`

### Transport (130-132)
- [x] 130. Heat conduction: steady-state vs transient → `thermodynamics/heat_conduction_comparison.py`
- [x] 131. Newton's law of cooling → `thermodynamics/newton_cooling.py`
- [x] 132. Diffusion equation: Gaussian spreading → `thermodynamics/diffusion_gaussian_spreading.py`

---

## 7) Statistical Mechanics (133-148)

### Core Distributions (133-136)
- [x] 133. Maxwell-Boltzmann speed distribution → `statistical_mechanics/maxwell_boltzmann_distribution.py`
- [x] 134. Equipartition via MD → `statistical_mechanics/equipartition_md.py`
- [x] 135. Fermi-Dirac occupation vs temperature → `statistical_mechanics/fermi_dirac_distribution.py`
- [x] 136. Bose-Einstein distribution, BEC onset → `statistical_mechanics/bose_einstein_distribution.py`

### Ising / Lattice Models (137-141)
- [x] 137. 2D Ising Metropolis: magnetization vs T → `statistical_mechanics/ising_2d_metropolis.py`
- [x] 138. 2D Ising: susceptibility peak, T_c estimate → `statistical_mechanics/ising_2d_susceptibility.py`
- [x] 139. 2D Ising: correlation length vs T → `statistical_mechanics/ising_2d_correlation.py`
- [x] 140. Cluster algorithms vs Metropolis near T_c → `statistical_mechanics/cluster_vs_metropolis.py`
- [x] 141. Finite-size scaling: critical exponents → `statistical_mechanics/finite_size_scaling.py`

### Random Processes (142-144)
- [x] 142. Percolation threshold estimation → `statistical_mechanics/percolation_threshold.py`
- [x] 143. DLA (diffusion-limited aggregation) → `statistical_mechanics/dla_growth.py`
- [x] 144. KPZ growth → `statistical_mechanics/kpz_growth.py`

### Fluctuation Theorems (145-146)
- [x] 145. Jarzynski equality → `statistical_mechanics/jarzynski_equality.py`
- [x] 146. Crooks fluctuation theorem → `statistical_mechanics/crooks_fluctuation.py`

### Kinetic Theory (147-148)
- [x] 147. Boltzmann equation: BGK relaxation → `statistical_mechanics/boltzmann_relaxation.py`
- [x] 148. H-theorem: entropy evolution → `statistical_mechanics/h_theorem_entropy.py`

---

## 8) Quantum Mechanics (149-183)

### Wave Mechanics (149-155)
- [x] 149. Particle in box: eigenstates + time evolution → `quantum/particle_in_box_eigenstates.py`
- [x] 150. Finite square well: bound states vs depth → `quantum/finite_square_well.py`
- [x] 151. Step potential: reflection/transmission → `quantum/step_potential.py`
- [x] 152. Barrier tunneling: transmission vs width/height → `quantum/tunneling.py`
- [x] 153. WKB approximation vs exact tunneling → `quantum/wkb_tunneling.py`
- [x] 154. Gaussian wavepacket spreading → `quantum/gaussian_wavepacket_spreading.py`
- [x] 155. Wavepacket scattering off barrier → `quantum/wavepacket_scattering.py`

### Harmonic Oscillator (156-158)
- [x] 156. SHO eigenstates: ladder operators → `quantum/sho_eigenstates.py`
- [x] 157. Coherent states: minimal uncertainty → `quantum/coherent_states.py`
- [x] 158. Commutator uncertainty relations → `quantum/commutator_uncertainty.py`

### Hydrogen & Angular Momentum (159-161)
- [x] 159. Hydrogen radial wavefunctions → `quantum/hydrogen_radial.py`
- [x] 160. Selection rules via dipole matrix elements → `quantum/selection_rules.py`
- [x] 161. Zeeman splitting vs B-field → `quantum/zeeman_splitting.py`

### Spin & Two-Level Systems (162-165)
- [x] 162. Spin-1/2 Larmor precession → `quantum/larmor_precession.py`
- [x] 163. Rabi oscillations with detuning → `quantum/rabi_oscillations.py`
- [x] 164. Adiabatic theorem demonstration → `quantum/adiabatic_theorem.py`
- [x] 165. Landau-Zener transition → `quantum/landau_zener.py`

### Measurement & Interference (166-168)
- [x] 166. Double-slit from two-path superposition → `quantum/double_slit_superposition.py`
- [x] 167. Which-path decoherence: visibility vs dephasing → `quantum/which_path_decoherence.py`
- [x] 168. Heisenberg uncertainty: minimum wavepacket → `quantum/heisenberg_uncertainty.py`

### Approximation Methods (169-171)
- [x] 169. Time-independent perturbation: anharmonic oscillator → `quantum/perturbation_theory.py`
- [x] 169b. Degenerate perturbation theory → `quantum/degenerate_perturbation.py`
- [x] 170. Stark effect: hydrogen in electric field → `quantum/stark_effect.py`
- [x] 171. Variational method: quartic potential ground state → `quantum/variational_quartic.py`

### Many-Body / Identical Particles (172-174)
- [x] 172. Two-fermion antisymmetrization → `quantum/fermion_antisymmetrization.py`
- [x] 173. Exchange energy in two-electron model → `quantum/exchange_energy.py`
- [x] 174. Tight-binding chain (second-quantized) → `quantum/tight_binding.py`

### Open Quantum Systems (175-178)
- [x] 175. Density matrix: pure vs mixed states → `quantum/density_matrix_intro.py`
- [x] 176. Lindblad dephasing evolution → `quantum/lindblad_dephasing.py`
- [x] 177. T1/T2 on Bloch sphere → `quantum/t1_t2_bloch_sphere.py`
- [x] 178. Caldeira-Leggett decoherence → `quantum/caldeira_leggett.py`

### Quantum Information (179-183)
- [x] 179. Bell inequality (CHSH) violation → `quantum_computing/bell_chsh.py`
- [x] 180. Quantum teleportation → `quantum_computing/quantum_teleportation.py`
- [x] 181. Grover search → `quantum_computing/grover_search.py`
- [x] 182. Quantum phase estimation → `quantum/quantum_phase_estimation.py`
- [x] 183. Shor's algorithm (simplified) → `quantum/shor_algorithm.py`

---

## 9) Special Relativity (184-190)

- [x] 184. Lorentz transformation: simultaneity shift → `relativity/lorentz_simultaneity.py`
- [x] 185. Time dilation: muon lifetime → `relativity/time_dilation.py`
- [x] 186. Length contraction → `relativity/length_contraction.py`
- [x] 187. Relativistic velocity addition → `relativity/velocity_addition.py`
- [x] 188. Relativistic Doppler + aberration → `relativity/doppler_aberration.py`
- [x] 189. Energy-momentum: threshold energies → `relativity/energy_momentum_threshold.py`
- [x] 190. Constant proper acceleration worldline → `relativity/proper_acceleration.py`

---

## 10) General Relativity & Cosmology (191-203)

### Schwarzschild & Kerr (191-197)
- [x] 191. Twin paradox visualization → `relativity/twin_paradox.py`
- [x] 192. Schwarzschild geodesics: perihelion precession → `relativity/schwarzschild_geodesics.py`
- [x] 193. Light bending: deflection vs impact parameter → `relativity/light_bending.py`
- [x] 194. Shapiro delay vs closest approach → `relativity/shapiro_delay.py`
- [x] 195. Gravitational redshift vs radius → `relativity/gravitational_redshift.py`
- [x] 196. ISCO radius calculation → `relativity/isco_radius.py`
- [x] 197. Frame dragging (Kerr) → `relativity/frame_dragging.py`

### Gravitational Waves (198-200)
- [x] 198. Binary inspiral chirp frequency → `relativity/binary_inspiral.py`
- [x] 199. GW strain h(t) for inspiral → `relativity/gw_strain.py`
- [x] 200. Matched filtering: detect chirp in noise → `relativity/matched_filtering.py`

### Cosmology (201-203)
- [x] 201. Friedmann equations: a(t) for different eras → `relativity/friedmann_integration.py`
- [x] 202. Distance-redshift relations → `relativity/distance_redshift.py`
- [x] 203. CMB acoustic peaks (simplified) → `relativity/cmb_acoustic_peaks.py`

---

## 11) Particle / Nuclear / QFT (204-219)

### Scattering & Decays (204-208)
- [x] 204. Rutherford scattering angular distribution → `particle_nuclear/exp204_rutherford_scattering.py`
- [x] 205. Partial wave expansion → `particle_nuclear/exp205_partial_wave_expansion.py`
- [x] 206. Breit-Wigner resonance → `particle_nuclear/exp206_breit_wigner_resonance.py`
- [x] 207. Two-body decay kinematics → `particle_nuclear/exp207_two_body_decay.py`
- [x] 208. Cross section vs energy → `particle_nuclear/exp208_cross_section_energy.py`

### Nuclear (209-212)
- [x] 209. Radioactive decay chains (Bateman) → `particle_nuclear/exp209_decay_chains.py`
- [x] 210. Binding energy curve (SEMF) → `particle_nuclear/exp210_binding_energy.py`
- [x] 211. Gamow tunneling factor → `particle_nuclear/exp211_gamow_tunneling.py`
- [x] 212. Neutron moderation random walk → `particle_nuclear/exp212_neutron_moderation.py`

### QFT Toys (213-219)
- [x] 213. Klein-Gordon 1D wavepackets → `particle_nuclear/exp213_klein_gordon_wavepacket.py`
- [x] 214. Dirac equation: Zitterbewegung → `particle_nuclear/exp214_zitterbewegung.py`
- [x] 215. Phi^4 kink solutions → `particle_nuclear/exp215_phi4_kink.py`
- [x] 216. Lattice field theory energy conservation → `particle_nuclear/exp216_lattice_field_energy.py`
- [x] 217. Block-spin RG on Ising → `particle_nuclear/exp217_block_spin_rg.py`
- [x] 218. Running coupling: beta function → `particle_nuclear/exp218_running_coupling.py`
- [x] 219. Path integral Monte Carlo → `particle_nuclear/exp219_path_integral_mc.py`

---

## 12) Condensed Matter (220-248)

### Band Structure (220-225)
- [x] 220. Tight-binding 1D chain → `condensed_matter/tight_binding_1d_chain.py`
- [x] 221. Kronig-Penney: band gaps → `condensed_matter/kronig_penney_band_gaps.py`
- [x] 222. Finite chain: edge states → `condensed_matter/finite_chain_edge_states.py`
- [x] 223. 2D square lattice: Fermi surface → `condensed_matter/square_lattice_fermi_surface.py`
- [x] 224. Graphene: Dirac cones → `condensed_matter/graphene_dirac_cones.py`
- [x] 225. DOS in 1D/2D/3D → `condensed_matter/dos_1d_2d_3d.py`

### Semiconductors (226-230)
- [x] 226. pn junction: depletion width → `condensed_matter/pn_junction_depletion.py`
- [x] 227. Drift-diffusion: I-V curve → `condensed_matter/drift_diffusion_iv.py`
- [x] 228. Drude conductivity → `condensed_matter/drude_conductivity.py`
- [x] 229. Classical Hall effect → `condensed_matter/hall_effect_classical.py`
- [x] 230. Anderson localization → `condensed_matter/metal_insulator_anderson.py`

### Phonons / Heat (231-234)
- [x] 231. MOS capacitor: C-V curve → `condensed_matter/mos_capacitor_cv.py`
- [x] 232. 1D phonon dispersion → `condensed_matter/phonon_dispersion_1d.py`
- [x] 233. Debye heat capacity vs T → `condensed_matter/debye_heat_capacity.py`
- [x] 234. Thermal conductivity via phonon MFP → `condensed_matter/thermal_conductivity_phonon.py`

### Magnetism (235-237)
- [x] 235. Heisenberg model Monte Carlo → `condensed_matter/heisenberg_monte_carlo.py`
- [x] 236. XY model: vortex unbinding → `condensed_matter/xy_model_vortices.py`
- [x] 237. Hysteresis toy model → `condensed_matter/hysteresis_domain_model.py`

### Superconductivity (238-242)
- [x] 238. Ginzburg-Landau vortices → `condensed_matter/ginzburg_landau_vortices.py`
- [x] 239. London penetration depth → `condensed_matter/london_penetration_depth.py`
- [x] 240. Josephson junction: I-phase relation → `condensed_matter/josephson_junction.py`
- [x] 241. BCS gap vs temperature → `condensed_matter/bcs_gap_temperature.py`
- [x] 242. Integer QHE edge states → `condensed_matter/integer_qhe_edge.py`

### Topological Phases (243-245)
- [x] 243. SSH model: Zak phase + edge states → `condensed_matter/ssh_model_edge_states.py`
- [x] 244. Berry curvature integration → `condensed_matter/berry_curvature_integration.py`
- [x] 245. Chern insulator: Chern number → `condensed_matter/chern_insulator.py`

### Correlated Systems (246-248)
- [x] 246. Hubbard model: Mott gap → `condensed_matter/hubbard_mott_gap.py`
- [x] 247. t-J model on small lattice → `condensed_matter/t_j_model.py`
- [x] 248. Moire flat bands → `condensed_matter/moire_flat_band.py`

---

## 13) AMO / Quantum Optics (249-258)

- [x] 249. Optical Bloch equations → `amo/optical_bloch_equations.py`
- [x] 250. Saturation / power broadening → `amo/saturation_power_broadening.py`
- [x] 251. Doppler vs natural linewidth → `amo/doppler_natural_linewidth.py`
- [x] 252. Laser cooling force (optical molasses) → `amo/laser_cooling_molasses.py`
- [x] 253. MOT restoring force → `amo/mot_restoring_force.py`
- [x] 254. Atom interferometer phase shift → `amo/atom_interferometer.py`
- [x] 255. Ramsey spectroscopy linewidth → `amo/ramsey_spectroscopy.py`
- [x] 256. Spin squeezing → `amo/spin_squeezing.py`
- [x] 257. Jaynes-Cummings: Rabi splitting → `amo/jaynes_cummings.py`
- [x] 258. Frequency comb generation → `amo/frequency_comb.py`

---

## 14) Plasma & Fusion (259-267)

- [x] 259. Debye shielding profile → `plasma_fusion/debye_shielding.py`
- [x] 260. Plasma frequency oscillations → `plasma_fusion/plasma_frequency_oscillations.py`
- [x] 261. Langmuir wave dispersion → `plasma_fusion/langmuir_wave_dispersion.py`
- [x] 262. Two-stream instability → `plasma_fusion/two_stream_instability.py`
- [x] 263. Landau damping → `plasma_fusion/landau_damping.py`
- [x] 264. Alfven waves in MHD → `plasma_fusion/alfven_waves_mhd.py`
- [x] 265. Magnetic reconnection (tearing mode) → `plasma_fusion/magnetic_reconnection_tearing.py`
- [x] 266. Tokamak guiding center motion → `plasma_fusion/tokamak_guiding_center.py`
- [x] 267. Lawson criterion visualization → `plasma_fusion/lawson_criterion_visualization.py`

---

## 15) Astrophysics (268-274)

- [x] 268. Hydrostatic equilibrium: polytropes → `astrophysics/hydrostatic_equilibrium_polytrope.py`
- [x] 269. Virial theorem in N-body → `astrophysics/virial_theorem_nbody.py`
- [x] 270. Main-sequence scaling relations → `astrophysics/main_sequence_scaling.py`
- [x] 271. Radiative transfer: optical depth → `astrophysics/radiative_transfer_optical_depth.py`
- [x] 272. Accretion disk temperature profile → `astrophysics/accretion_disk_temperature.py`
- [x] 273. Eddington luminosity → `astrophysics/eddington_luminosity.py`
- [x] 274. Orbital decay from GW emission → `astrophysics/orbital_decay_gw.py`

---

## Summary Statistics

| Domain | Range | Count | Implemented |
|--------|-------|-------|-------------|
| Foundations/Numerical | 1-25 | 25 | **25/25** |
| Classical Mechanics | 26-70 | 45+ | **55/55** |
| Waves & Acoustics | 71-79 | 9 | **9/9** |
| Electromagnetism | 80-100 | 21 | **21/21** |
| Optics | 101-119 | 19 | **19/19** |
| Thermodynamics | 120-132 | 13 | **13/13** |
| Statistical Mechanics | 133-148 | 16 | **16/16** |
| Quantum Mechanics | 149-183 | 35+ | **40/40** |
| Special Relativity | 184-190 | 7 | **7/7** |
| General Relativity | 191-203 | 13 | **13/13** |
| Particle/Nuclear/QFT | 204-219 | 16 | **16/16** |
| Condensed Matter | 220-248 | 29 | **30/30** |
| AMO/Quantum Optics | 249-258 | 10 | **10/10** |
| Plasma & Fusion | 259-267 | 9 | **9/9** |
| Astrophysics | 268-274 | 7 | **7/7** |
| **Total** | 1-274+ | **274+** | **332 files** |

---

## File Statistics

- **Total Python files:** 332
- **Output PNG plots:** 163+
- **Directories:** 20+

---

*Last updated: 2026-01-12*
*All experiments verified and generating output plots*
