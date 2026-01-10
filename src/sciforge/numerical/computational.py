"""
Computational Physics Tools module.

This module implements advanced computational methods commonly used in
physics simulations: Monte Carlo, molecular dynamics, PDE solvers,
and linear algebra utilities.

Classes:
    Monte Carlo Methods:
    - ImportanceSampling: Variance reduction via importance sampling
    - MarkovChainMC: General Markov chain Monte Carlo
    - PathIntegralMC: Quantum Monte Carlo basics
    - MonteCarloIntegration: High-dimensional integration

    Molecular Dynamics:
    - VelocityVerlet: Symplectic integrator
    - Thermostat: Temperature control (Nose-Hoover, Langevin)
    - Barostat: Pressure control
    - PeriodicBoundary: Periodic boundary conditions
    - CellList: Efficient neighbor finding
    - EwaldSum: Long-range electrostatics

    PDE Solvers:
    - FiniteDifference1D: 1D grid methods
    - FiniteDifference2D: 2D grid methods
    - FiniteElement1D: Finite element basics
    - SpectralMethod: FFT-based solver
    - CrankNicolson: Implicit time stepping
    - ADIMethod: Alternating direction implicit

    Linear Algebra:
    - SparseMatrix: Sparse storage utilities
    - ConjugateGradient: CG solver
    - GMRES: Generalized minimal residual
    - EigenSolver: Large sparse eigenvalues
    - SVDSolver: Singular value decomposition
"""

import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, Tuple, Callable, Dict, List, Union
from scipy import sparse, linalg
from scipy.sparse import linalg as sparse_linalg
from dataclasses import dataclass
import warnings


# =============================================================================
# Monte Carlo Methods
# =============================================================================

class ImportanceSampling:
    """
    Importance sampling for variance reduction.

    Estimates E_p[f(x)] using samples from distribution q(x):
    E_p[f] ≈ (1/N) Σ f(x_i) p(x_i) / q(x_i)

    where x_i ~ q(x).

    Args:
        target_pdf: Target probability density p(x)
        proposal_pdf: Proposal probability density q(x)
        proposal_sampler: Function to generate samples from q
    """

    def __init__(self, target_pdf: Callable, proposal_pdf: Callable,
                 proposal_sampler: Callable):
        self.p = target_pdf
        self.q = proposal_pdf
        self.sample_q = proposal_sampler

    def estimate(self, func: Callable, n_samples: int = 10000,
                 return_variance: bool = False) -> Union[float, Tuple[float, float]]:
        """
        Estimate E_p[f(x)] using importance sampling.

        Args:
            func: Function to compute expectation of
            n_samples: Number of samples
            return_variance: If True, also return variance estimate

        Returns:
            Estimated expectation (and variance if requested)
        """
        samples = np.array([self.sample_q() for _ in range(n_samples)])

        # Importance weights
        weights = np.array([self.p(x) / self.q(x) for x in samples])

        # Function values
        f_values = np.array([func(x) for x in samples])

        # Weighted average
        estimate = np.mean(f_values * weights)

        if return_variance:
            variance = np.var(f_values * weights) / n_samples
            return estimate, variance

        return estimate

    def effective_sample_size(self, n_samples: int = 10000) -> float:
        """
        Effective sample size for the importance sampling.

        ESS = (Σw_i)² / Σw_i²

        Args:
            n_samples: Number of samples

        Returns:
            Effective sample size
        """
        samples = np.array([self.sample_q() for _ in range(n_samples)])
        weights = np.array([self.p(x) / self.q(x) for x in samples])

        return np.sum(weights)**2 / np.sum(weights**2)

    def self_normalized_estimate(self, func: Callable,
                                  n_samples: int = 10000) -> float:
        """
        Self-normalized importance sampling.

        Useful when p(x) is known only up to normalization.

        E_p[f] ≈ Σ w_i f(x_i) / Σ w_i

        Args:
            func: Function to compute expectation of
            n_samples: Number of samples

        Returns:
            Self-normalized estimate
        """
        samples = np.array([self.sample_q() for _ in range(n_samples)])
        weights = np.array([self.p(x) / self.q(x) for x in samples])
        f_values = np.array([func(x) for x in samples])

        return np.sum(weights * f_values) / np.sum(weights)


class MarkovChainMC:
    """
    General Markov chain Monte Carlo sampler.

    Implements Metropolis-Hastings algorithm for sampling from
    arbitrary distributions.

    Args:
        target_log_prob: Log of target probability density
        proposal: Proposal distribution function(current) -> proposed
        proposal_log_prob: Log probability of proposal (optional)
    """

    def __init__(self, target_log_prob: Callable,
                 proposal: Callable,
                 proposal_log_prob: Optional[Callable] = None):
        self.log_p = target_log_prob
        self.propose = proposal
        self.log_q = proposal_log_prob  # For non-symmetric proposals

    def sample(self, x0: ArrayLike, n_samples: int,
               burn_in: int = 1000, thin: int = 1) -> np.ndarray:
        """
        Generate samples from target distribution.

        Args:
            x0: Initial state
            n_samples: Number of samples to collect
            burn_in: Number of initial samples to discard
            thin: Keep every thin-th sample

        Returns:
            Array of samples
        """
        x = np.asarray(x0)
        samples = []
        total_steps = burn_in + n_samples * thin
        n_accepted = 0

        for i in range(total_steps):
            # Propose new state
            x_new = self.propose(x)

            # Log acceptance ratio
            log_alpha = self.log_p(x_new) - self.log_p(x)

            # Correct for asymmetric proposals
            if self.log_q is not None:
                log_alpha += self.log_q(x, x_new) - self.log_q(x_new, x)

            # Accept/reject
            if np.log(np.random.random()) < log_alpha:
                x = x_new
                n_accepted += 1

            # Collect sample after burn-in
            if i >= burn_in and (i - burn_in) % thin == 0:
                samples.append(x.copy())

        self.acceptance_rate = n_accepted / total_steps
        return np.array(samples)

    def adaptive_proposal(self, x0: ArrayLike, n_adapt: int = 1000,
                          target_rate: float = 0.234) -> Callable:
        """
        Adapt proposal distribution to achieve target acceptance rate.

        Uses adaptive scaling of Gaussian proposals.

        Args:
            x0: Initial state
            n_adapt: Number of adaptation steps
            target_rate: Target acceptance rate

        Returns:
            Adapted proposal function
        """
        x = np.asarray(x0)
        dim = len(x) if hasattr(x, '__len__') else 1
        scale = 2.38 / np.sqrt(dim)  # Optimal scaling for Gaussian

        for _ in range(n_adapt):
            x_new = x + scale * np.random.randn(*x.shape)

            log_alpha = self.log_p(x_new) - self.log_p(x)

            if np.log(np.random.random()) < log_alpha:
                x = x_new
                scale *= 1.02  # Increase scale if accepting
            else:
                scale *= 0.98  # Decrease if rejecting

        self._adapted_scale = scale

        def adapted_proposal(current):
            return current + scale * np.random.randn(*current.shape)

        return adapted_proposal


class PathIntegralMC:
    """
    Path integral Monte Carlo for quantum systems.

    Samples quantum paths using the Trotter decomposition:
    Z = Tr[e^{-βH}] ≈ ∫ Π dx_i exp(-S[x])

    where S[x] is the discretized Euclidean action.

    Args:
        potential: Potential energy function V(x)
        mass: Particle mass
        beta: Inverse temperature ℏ/kT
        n_beads: Number of imaginary time slices
    """

    def __init__(self, potential: Callable, mass: float = 1.0,
                 beta: float = 1.0, n_beads: int = 100):
        self.V = potential
        self.m = mass
        self.beta = beta
        self.n_beads = n_beads
        self.tau = beta / n_beads  # Imaginary time step

    def action(self, path: np.ndarray) -> float:
        """
        Calculate Euclidean action for a path.

        S = Σ_i [m(x_{i+1} - x_i)²/(2τ) + τV(x_i)]

        Args:
            path: Array of positions at each imaginary time slice

        Returns:
            Action S
        """
        n = self.n_beads
        S = 0

        for i in range(n):
            i_next = (i + 1) % n  # Periodic boundary
            kinetic = self.m * (path[i_next] - path[i])**2 / (2 * self.tau)
            potential = self.tau * self.V(path[i])
            S += kinetic + potential

        return S

    def sample_paths(self, n_samples: int, x_init: Optional[np.ndarray] = None,
                     burn_in: int = 1000) -> np.ndarray:
        """
        Sample quantum paths using Metropolis algorithm.

        Args:
            n_samples: Number of path samples
            x_init: Initial path configuration
            burn_in: Number of burn-in steps

        Returns:
            Array of path samples
        """
        if x_init is None:
            path = np.zeros(self.n_beads)
        else:
            path = x_init.copy()

        paths = []
        step_size = 0.5

        for step in range(burn_in + n_samples):
            # Update each bead
            for i in range(self.n_beads):
                old_x = path[i]
                new_x = old_x + step_size * (np.random.random() - 0.5)

                # Change in action
                i_prev = (i - 1) % self.n_beads
                i_next = (i + 1) % self.n_beads

                old_S = (self.m * ((path[i_next] - old_x)**2 +
                                   (old_x - path[i_prev])**2) / (2 * self.tau) +
                         self.tau * self.V(old_x))

                new_S = (self.m * ((path[i_next] - new_x)**2 +
                                   (new_x - path[i_prev])**2) / (2 * self.tau) +
                         self.tau * self.V(new_x))

                # Metropolis accept/reject
                if np.random.random() < np.exp(old_S - new_S):
                    path[i] = new_x

            if step >= burn_in:
                paths.append(path.copy())

        return np.array(paths)

    def expectation_position(self, n_samples: int = 10000) -> Tuple[float, float]:
        """
        Compute expectation value <x>.

        Returns:
            (mean, standard error)
        """
        paths = self.sample_paths(n_samples)
        x_values = np.mean(paths, axis=1)  # Average over beads for each sample
        return np.mean(x_values), np.std(x_values) / np.sqrt(n_samples)

    def expectation_position_squared(self, n_samples: int = 10000) -> Tuple[float, float]:
        """
        Compute expectation value <x²>.

        Returns:
            (mean, standard error)
        """
        paths = self.sample_paths(n_samples)
        x2_values = np.mean(paths**2, axis=1)
        return np.mean(x2_values), np.std(x2_values) / np.sqrt(n_samples)


class MonteCarloIntegration:
    """
    Monte Carlo integration for high-dimensional integrals.

    I = ∫ f(x) dx ≈ V/N Σ f(x_i)

    where V is the volume and x_i are random samples.

    Args:
        function: Function to integrate
        bounds: List of (low, high) tuples for each dimension
    """

    def __init__(self, function: Callable, bounds: List[Tuple[float, float]]):
        self.f = function
        self.bounds = bounds
        self.dim = len(bounds)
        self.volume = np.prod([b[1] - b[0] for b in bounds])

    def integrate(self, n_samples: int = 100000) -> Tuple[float, float]:
        """
        Estimate integral using simple Monte Carlo.

        Args:
            n_samples: Number of random samples

        Returns:
            (estimate, standard error)
        """
        samples = np.array([
            [np.random.uniform(b[0], b[1]) for b in self.bounds]
            for _ in range(n_samples)
        ])

        f_values = np.array([self.f(x) for x in samples])

        estimate = self.volume * np.mean(f_values)
        variance = self.volume**2 * np.var(f_values) / n_samples

        return estimate, np.sqrt(variance)

    def stratified_sampling(self, n_samples: int,
                            n_strata: int = 10) -> Tuple[float, float]:
        """
        Stratified sampling for variance reduction.

        Divide domain into strata and sample uniformly from each.

        Args:
            n_samples: Total number of samples
            n_strata: Number of strata per dimension

        Returns:
            (estimate, standard error)
        """
        samples_per_stratum = max(1, n_samples // (n_strata ** self.dim))
        estimates = []

        # Create stratified samples
        for _ in range(samples_per_stratum):
            for indices in np.ndindex(*([n_strata] * self.dim)):
                sample = []
                for d, idx in enumerate(indices):
                    low = self.bounds[d][0] + idx * (self.bounds[d][1] - self.bounds[d][0]) / n_strata
                    high = self.bounds[d][0] + (idx + 1) * (self.bounds[d][1] - self.bounds[d][0]) / n_strata
                    sample.append(np.random.uniform(low, high))

                estimates.append(self.f(np.array(sample)))

        estimate = self.volume * np.mean(estimates)
        variance = self.volume**2 * np.var(estimates) / len(estimates)

        return estimate, np.sqrt(variance)

    def antithetic_sampling(self, n_samples: int) -> Tuple[float, float]:
        """
        Antithetic sampling for variance reduction.

        Uses pairs of negatively correlated samples.

        Args:
            n_samples: Number of samples

        Returns:
            (estimate, standard error)
        """
        n_pairs = n_samples // 2
        estimates = []

        for _ in range(n_pairs):
            u = np.array([np.random.uniform(0, 1) for _ in range(self.dim)])

            # Sample and its antithetic partner
            x1 = np.array([b[0] + u[i] * (b[1] - b[0])
                          for i, b in enumerate(self.bounds)])
            x2 = np.array([b[0] + (1 - u[i]) * (b[1] - b[0])
                          for i, b in enumerate(self.bounds)])

            estimates.append((self.f(x1) + self.f(x2)) / 2)

        estimate = self.volume * np.mean(estimates)
        variance = self.volume**2 * np.var(estimates) / len(estimates)

        return estimate, np.sqrt(variance)


# =============================================================================
# Molecular Dynamics
# =============================================================================

class VelocityVerlet:
    """
    Velocity Verlet integrator for molecular dynamics.

    Symplectic integrator that conserves energy better than
    simple Euler methods.

    x(t+dt) = x(t) + v(t)dt + a(t)dt²/2
    v(t+dt) = v(t) + (a(t) + a(t+dt))dt/2

    Args:
        force_function: Function(positions) -> forces/mass (accelerations)
        dt: Time step
    """

    def __init__(self, force_function: Callable, dt: float):
        self.force = force_function
        self.dt = dt

    def step(self, x: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advance positions and velocities by one time step.

        Args:
            x: Current positions
            v: Current velocities

        Returns:
            (new_positions, new_velocities)
        """
        dt = self.dt

        # Current acceleration
        a = self.force(x)

        # Update positions
        x_new = x + v * dt + 0.5 * a * dt**2

        # New acceleration
        a_new = self.force(x_new)

        # Update velocities
        v_new = v + 0.5 * (a + a_new) * dt

        return x_new, v_new

    def integrate(self, x0: np.ndarray, v0: np.ndarray,
                  n_steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run MD simulation for multiple steps.

        Args:
            x0: Initial positions
            v0: Initial velocities
            n_steps: Number of time steps

        Returns:
            (positions_history, velocities_history, times)
        """
        x = x0.copy()
        v = v0.copy()

        positions = [x.copy()]
        velocities = [v.copy()]
        times = [0]

        for i in range(n_steps):
            x, v = self.step(x, v)
            positions.append(x.copy())
            velocities.append(v.copy())
            times.append((i + 1) * self.dt)

        return np.array(positions), np.array(velocities), np.array(times)


class Thermostat:
    """
    Temperature control for molecular dynamics.

    Implements Nose-Hoover and Langevin thermostats.

    Args:
        temperature: Target temperature
        tau: Coupling time constant
        method: 'nose_hoover' or 'langevin'
    """

    def __init__(self, temperature: float, tau: float = 0.5,
                 method: str = 'nose_hoover'):
        if temperature <= 0:
            raise ValueError("Temperature must be positive")

        self.T = temperature
        self.tau = tau
        self.method = method
        self.xi = 0  # Nose-Hoover friction coefficient

    def apply_nose_hoover(self, v: np.ndarray, masses: np.ndarray,
                          dt: float) -> np.ndarray:
        """
        Apply Nose-Hoover thermostat.

        dξ/dt = (1/Q)(Σm_i v_i² - Nk_BT)

        Args:
            v: Velocities
            masses: Particle masses
            dt: Time step

        Returns:
            Modified velocities
        """
        n_dof = v.size  # Number of degrees of freedom
        K = 0.5 * np.sum(masses * np.sum(v**2, axis=-1))

        # Q is the thermostat mass
        Q = n_dof * self.T * self.tau**2

        # Update friction coefficient
        self.xi += dt * (2 * K - n_dof * self.T) / Q

        # Scale velocities
        scale = np.exp(-0.5 * self.xi * dt)
        return v * scale

    def apply_langevin(self, v: np.ndarray, masses: np.ndarray,
                       dt: float) -> np.ndarray:
        """
        Apply Langevin thermostat (stochastic dynamics).

        v' = v - γv dt + √(2γk_BT/m) dW

        Args:
            v: Velocities
            masses: Particle masses
            dt: Time step

        Returns:
            Modified velocities
        """
        gamma = 1 / self.tau

        # Friction and random force
        friction = np.exp(-gamma * dt)

        # Random kick variance
        sigma = np.sqrt(self.T * (1 - friction**2))

        if v.ndim == 1:
            noise = sigma / np.sqrt(masses) * np.random.randn(len(v))
        else:
            noise = sigma / np.sqrt(masses[:, np.newaxis]) * np.random.randn(*v.shape)

        return v * friction + noise

    def apply(self, v: np.ndarray, masses: np.ndarray, dt: float) -> np.ndarray:
        """
        Apply thermostat based on selected method.

        Args:
            v: Velocities
            masses: Particle masses
            dt: Time step

        Returns:
            Modified velocities
        """
        if self.method == 'nose_hoover':
            return self.apply_nose_hoover(v, masses, dt)
        elif self.method == 'langevin':
            return self.apply_langevin(v, masses, dt)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def instantaneous_temperature(self, v: np.ndarray,
                                   masses: np.ndarray) -> float:
        """
        Calculate instantaneous temperature from velocities.

        T = 2K/(N_dof k_B)

        Args:
            v: Velocities
            masses: Particle masses

        Returns:
            Instantaneous temperature
        """
        n_dof = v.size
        K = 0.5 * np.sum(masses * np.sum(v**2, axis=-1))
        return 2 * K / n_dof


class Barostat:
    """
    Pressure control for molecular dynamics.

    Implements Berendsen and Parrinello-Rahman barostats.

    Args:
        pressure: Target pressure
        tau: Coupling time constant
        compressibility: Isothermal compressibility
    """

    def __init__(self, pressure: float, tau: float = 1.0,
                 compressibility: float = 4.5e-5):
        self.P = pressure
        self.tau = tau
        self.beta = compressibility

    def apply_berendsen(self, positions: np.ndarray, box: np.ndarray,
                        current_pressure: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Berendsen barostat (weak coupling).

        μ = [1 - (dt/τ_P)β(P_0 - P)]^(1/3)

        Args:
            positions: Particle positions
            box: Box dimensions
            current_pressure: Current instantaneous pressure
            dt: Time step

        Returns:
            (scaled_positions, scaled_box)
        """
        mu = (1 - (dt / self.tau) * self.beta * (self.P - current_pressure))**(1/3)

        new_positions = positions * mu
        new_box = box * mu

        return new_positions, new_box

    def compute_pressure(self, positions: np.ndarray, velocities: np.ndarray,
                         forces: np.ndarray, masses: np.ndarray,
                         volume: float) -> float:
        """
        Compute instantaneous pressure.

        P = (Nk_BT + (1/3)Σr_i·f_i) / V

        Args:
            positions: Particle positions
            velocities: Particle velocities
            forces: Forces on particles
            masses: Particle masses
            volume: System volume

        Returns:
            Instantaneous pressure
        """
        n = len(positions)

        # Kinetic contribution
        KE = 0.5 * np.sum(masses * np.sum(velocities**2, axis=-1))
        P_kinetic = 2 * KE / (3 * volume)

        # Virial contribution
        virial = np.sum(positions * forces)
        P_virial = virial / (3 * volume)

        return P_kinetic + P_virial


class PeriodicBoundary:
    """
    Periodic boundary conditions for MD simulations.

    Args:
        box_size: Box dimensions (can be array for non-cubic)
    """

    def __init__(self, box_size: ArrayLike):
        self.box = np.asarray(box_size)

    def wrap(self, positions: np.ndarray) -> np.ndarray:
        """
        Wrap positions into primary cell.

        Args:
            positions: Particle positions

        Returns:
            Wrapped positions
        """
        return positions - self.box * np.floor(positions / self.box)

    def minimum_image(self, r: np.ndarray) -> np.ndarray:
        """
        Apply minimum image convention to displacement.

        Args:
            r: Displacement vector

        Returns:
            Minimum image displacement
        """
        return r - self.box * np.round(r / self.box)

    def distance(self, r1: np.ndarray, r2: np.ndarray) -> float:
        """
        Compute minimum image distance between two points.

        Args:
            r1, r2: Position vectors

        Returns:
            Minimum image distance
        """
        dr = self.minimum_image(r2 - r1)
        return np.linalg.norm(dr)

    def all_images(self, positions: np.ndarray,
                   n_images: int = 1) -> np.ndarray:
        """
        Generate periodic images of all particles.

        Args:
            positions: Particle positions
            n_images: Number of image cells in each direction

        Returns:
            Array including all images
        """
        all_pos = []

        for ix in range(-n_images, n_images + 1):
            for iy in range(-n_images, n_images + 1):
                for iz in range(-n_images, n_images + 1):
                    shift = np.array([ix, iy, iz]) * self.box
                    all_pos.extend(positions + shift)

        return np.array(all_pos)


class CellList:
    """
    Cell list for efficient neighbor finding in MD.

    Divides simulation box into cells for O(N) neighbor finding.

    Args:
        box_size: Simulation box dimensions
        cutoff: Interaction cutoff distance
    """

    def __init__(self, box_size: ArrayLike, cutoff: float):
        self.box = np.asarray(box_size)
        self.cutoff = cutoff

        # Number of cells in each dimension
        self.n_cells = np.maximum(1, np.floor(self.box / cutoff).astype(int))
        self.cell_size = self.box / self.n_cells

        # Initialize cells
        self.cells = {}

    def build(self, positions: np.ndarray) -> None:
        """
        Build cell list from positions.

        Args:
            positions: Particle positions (N x 3)
        """
        self.cells = {}

        for i, pos in enumerate(positions):
            cell_idx = self.get_cell_index(pos)
            if cell_idx not in self.cells:
                self.cells[cell_idx] = []
            self.cells[cell_idx].append(i)

    def get_cell_index(self, position: np.ndarray) -> Tuple:
        """
        Get cell index for a position.

        Args:
            position: Position vector

        Returns:
            Cell index tuple
        """
        idx = np.floor(position / self.cell_size).astype(int)
        idx = idx % self.n_cells  # Handle PBC
        return tuple(idx)

    def get_neighbors(self, particle_idx: int,
                      positions: np.ndarray) -> List[int]:
        """
        Get neighbor list for a particle.

        Args:
            particle_idx: Index of particle
            positions: All positions

        Returns:
            List of neighbor indices
        """
        pos = positions[particle_idx]
        cell_idx = self.get_cell_index(pos)

        neighbors = []

        # Check neighboring cells
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                for dk in [-1, 0, 1]:
                    neighbor_cell = ((cell_idx[0] + di) % self.n_cells[0],
                                    (cell_idx[1] + dj) % self.n_cells[1],
                                    (cell_idx[2] + dk) % self.n_cells[2])

                    if neighbor_cell in self.cells:
                        for j in self.cells[neighbor_cell]:
                            if j != particle_idx:
                                dr = positions[j] - pos
                                dr = dr - self.box * np.round(dr / self.box)
                                if np.linalg.norm(dr) < self.cutoff:
                                    neighbors.append(j)

        return neighbors

    def neighbor_pairs(self, positions: np.ndarray) -> List[Tuple[int, int]]:
        """
        Get all unique neighbor pairs.

        Args:
            positions: All positions

        Returns:
            List of (i, j) pairs
        """
        self.build(positions)
        pairs = []
        seen = set()

        for i in range(len(positions)):
            neighbors = self.get_neighbors(i, positions)
            for j in neighbors:
                pair = (min(i, j), max(i, j))
                if pair not in seen:
                    pairs.append(pair)
                    seen.add(pair)

        return pairs


class EwaldSum:
    """
    Ewald summation for long-range Coulomb interactions.

    Splits 1/r into short-range (real space) and long-range
    (Fourier space) contributions.

    Args:
        box_size: Simulation box dimensions
        n_particles: Number of charged particles
        alpha: Ewald convergence parameter
        k_max: Maximum k-vector magnitude
    """

    def __init__(self, box_size: ArrayLike, n_particles: int,
                 alpha: Optional[float] = None, k_max: int = 5):
        self.box = np.asarray(box_size)
        self.V = np.prod(self.box)
        self.n = n_particles
        self.k_max = k_max

        # Optimal alpha
        if alpha is None:
            self.alpha = np.sqrt(np.pi) * (n_particles / self.V**2)**(1/6)
        else:
            self.alpha = alpha

        # Pre-compute k-vectors
        self._setup_k_vectors()

    def _setup_k_vectors(self) -> None:
        """Set up reciprocal space vectors."""
        self.k_vectors = []
        self.k_factors = []

        for nx in range(-self.k_max, self.k_max + 1):
            for ny in range(-self.k_max, self.k_max + 1):
                for nz in range(-self.k_max, self.k_max + 1):
                    if nx == 0 and ny == 0 and nz == 0:
                        continue

                    k = 2 * np.pi * np.array([nx, ny, nz]) / self.box
                    k_sq = np.dot(k, k)

                    factor = 4 * np.pi / self.V * np.exp(-k_sq / (4 * self.alpha**2)) / k_sq

                    self.k_vectors.append(k)
                    self.k_factors.append(factor)

        self.k_vectors = np.array(self.k_vectors)
        self.k_factors = np.array(self.k_factors)

    def real_space_energy(self, positions: np.ndarray,
                          charges: np.ndarray) -> float:
        """
        Calculate real space part of Ewald sum.

        Args:
            positions: Particle positions
            charges: Particle charges

        Returns:
            Real space energy contribution
        """
        from scipy.special import erfc

        E_real = 0
        n = len(positions)
        pbc = PeriodicBoundary(self.box)

        for i in range(n):
            for j in range(i + 1, n):
                dr = pbc.minimum_image(positions[j] - positions[i])
                r = np.linalg.norm(dr)
                E_real += charges[i] * charges[j] * erfc(self.alpha * r) / r

        return E_real

    def fourier_space_energy(self, positions: np.ndarray,
                             charges: np.ndarray) -> float:
        """
        Calculate Fourier space part of Ewald sum.

        Args:
            positions: Particle positions
            charges: Particle charges

        Returns:
            Fourier space energy contribution
        """
        E_fourier = 0

        for k, factor in zip(self.k_vectors, self.k_factors):
            # Structure factor
            S_k = np.sum(charges * np.exp(1j * np.dot(positions, k)))
            E_fourier += factor * np.abs(S_k)**2

        return 0.5 * E_fourier

    def self_energy(self, charges: np.ndarray) -> float:
        """
        Self-energy correction.

        Args:
            charges: Particle charges

        Returns:
            Self-energy (to be subtracted)
        """
        return -self.alpha / np.sqrt(np.pi) * np.sum(charges**2)

    def total_energy(self, positions: np.ndarray,
                     charges: np.ndarray) -> float:
        """
        Total Ewald energy.

        Args:
            positions: Particle positions
            charges: Particle charges

        Returns:
            Total Coulomb energy
        """
        E_real = self.real_space_energy(positions, charges)
        E_fourier = self.fourier_space_energy(positions, charges)
        E_self = self.self_energy(charges)

        return E_real + E_fourier + E_self


# =============================================================================
# PDE Solvers
# =============================================================================

class FiniteDifference1D:
    """
    1D finite difference solver for PDEs.

    Supports heat equation, wave equation, and Poisson equation.

    Args:
        nx: Number of grid points
        dx: Grid spacing
        bc_type: Boundary condition ('dirichlet', 'neumann', 'periodic')
    """

    def __init__(self, nx: int, dx: float, bc_type: str = 'dirichlet'):
        self.nx = nx
        self.dx = dx
        self.bc_type = bc_type

        self.x = np.linspace(0, (nx - 1) * dx, nx)

    def laplacian_matrix(self) -> sparse.csr_matrix:
        """
        Construct 1D Laplacian matrix (∂²/∂x²).

        Returns:
            Sparse Laplacian matrix
        """
        n = self.nx
        diagonals = [
            np.ones(n - 1),
            -2 * np.ones(n),
            np.ones(n - 1)
        ]

        L = sparse.diags(diagonals, [-1, 0, 1], format='csr') / self.dx**2

        # Apply boundary conditions
        if self.bc_type == 'periodic':
            L = L.tolil()
            L[0, -1] = 1 / self.dx**2
            L[-1, 0] = 1 / self.dx**2
            L = L.tocsr()
        elif self.bc_type == 'neumann':
            L = L.tolil()
            L[0, 0] = -1 / self.dx**2
            L[-1, -1] = -1 / self.dx**2
            L = L.tocsr()

        return L

    def solve_poisson(self, rhs: np.ndarray,
                      bc_left: float = 0, bc_right: float = 0) -> np.ndarray:
        """
        Solve Poisson equation: ∇²u = f.

        Args:
            rhs: Right-hand side f(x)
            bc_left: Left boundary value
            bc_right: Right boundary value

        Returns:
            Solution u(x)
        """
        L = self.laplacian_matrix()

        # Modify RHS for boundary conditions
        b = rhs.copy()
        if self.bc_type == 'dirichlet':
            b[0] -= bc_left / self.dx**2
            b[-1] -= bc_right / self.dx**2
            # Remove boundary rows
            L = L[1:-1, 1:-1]
            b = b[1:-1]

            u_interior = sparse_linalg.spsolve(L, b)

            # Add boundary values
            return np.concatenate([[bc_left], u_interior, [bc_right]])

        return sparse_linalg.spsolve(L, b)

    def solve_heat_equation(self, u0: np.ndarray, dt: float, n_steps: int,
                            alpha: float = 1.0) -> np.ndarray:
        """
        Solve heat equation: ∂u/∂t = α ∂²u/∂x².

        Uses implicit (backward Euler) method.

        Args:
            u0: Initial condition
            dt: Time step
            alpha: Thermal diffusivity
            n_steps: Number of time steps

        Returns:
            Solution at final time
        """
        L = self.laplacian_matrix()
        I = sparse.eye(self.nx)

        # (I - αΔt L) u^{n+1} = u^n
        A = I - alpha * dt * L

        u = u0.copy()
        for _ in range(n_steps):
            u = sparse_linalg.spsolve(A, u)

        return u


class FiniteDifference2D:
    """
    2D finite difference solver for PDEs.

    Args:
        nx, ny: Number of grid points
        dx, dy: Grid spacing
        bc_type: Boundary condition type
    """

    def __init__(self, nx: int, ny: int, dx: float, dy: float,
                 bc_type: str = 'dirichlet'):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.bc_type = bc_type

        self.x = np.linspace(0, (nx - 1) * dx, nx)
        self.y = np.linspace(0, (ny - 1) * dy, ny)

    def laplacian_matrix(self) -> sparse.csr_matrix:
        """
        Construct 2D Laplacian matrix.

        Returns:
            Sparse Laplacian matrix
        """
        nx, ny = self.nx, self.ny
        n = nx * ny

        # 1D Laplacians
        Dxx = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(nx, nx)) / self.dx**2
        Dyy = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(ny, ny)) / self.dy**2

        Ix = sparse.eye(nx)
        Iy = sparse.eye(ny)

        # 2D Laplacian via Kronecker product
        L = sparse.kron(Iy, Dxx) + sparse.kron(Dyy, Ix)

        return L.tocsr()

    def solve_poisson(self, rhs: np.ndarray,
                      bc_func: Optional[Callable] = None) -> np.ndarray:
        """
        Solve 2D Poisson equation: ∇²u = f.

        Args:
            rhs: Right-hand side f(x,y) as 2D array
            bc_func: Boundary condition function(x,y)

        Returns:
            Solution u(x,y) as 2D array
        """
        L = self.laplacian_matrix()

        # Flatten RHS
        b = rhs.flatten()

        # Apply Dirichlet BC
        if self.bc_type == 'dirichlet' and bc_func is not None:
            # Modify RHS for boundary conditions
            for i in range(self.nx):
                for j in range(self.ny):
                    if i == 0 or i == self.nx - 1 or j == 0 or j == self.ny - 1:
                        idx = i * self.ny + j
                        b[idx] = bc_func(self.x[i], self.y[j])
                        # Set row to identity
                        L = L.tolil()
                        L[idx, :] = 0
                        L[idx, idx] = 1
                        L = L.tocsr()

        u = sparse_linalg.spsolve(L, b)
        return u.reshape((self.nx, self.ny))

    def solve_heat_equation(self, u0: np.ndarray, dt: float, n_steps: int,
                            alpha: float = 1.0) -> np.ndarray:
        """
        Solve 2D heat equation.

        Args:
            u0: Initial condition (2D array)
            dt: Time step
            alpha: Thermal diffusivity
            n_steps: Number of time steps

        Returns:
            Solution at final time
        """
        L = self.laplacian_matrix()
        n = self.nx * self.ny
        I = sparse.eye(n)

        A = I - alpha * dt * L

        u = u0.flatten()
        for _ in range(n_steps):
            u = sparse_linalg.spsolve(A, u)

        return u.reshape((self.nx, self.ny))


class FiniteElement1D:
    """
    1D finite element solver using linear elements.

    Args:
        nodes: Node positions
    """

    def __init__(self, nodes: ArrayLike):
        self.nodes = np.asarray(nodes)
        self.n_nodes = len(nodes)
        self.n_elements = len(nodes) - 1

    def stiffness_matrix(self) -> np.ndarray:
        """
        Assemble global stiffness matrix.

        Returns:
            Global stiffness matrix K
        """
        K = np.zeros((self.n_nodes, self.n_nodes))

        for e in range(self.n_elements):
            h = self.nodes[e + 1] - self.nodes[e]

            # Local stiffness matrix
            K_local = np.array([[1, -1], [-1, 1]]) / h

            # Assemble into global matrix
            K[e:e+2, e:e+2] += K_local

        return K

    def mass_matrix(self) -> np.ndarray:
        """
        Assemble global mass matrix.

        Returns:
            Global mass matrix M
        """
        M = np.zeros((self.n_nodes, self.n_nodes))

        for e in range(self.n_elements):
            h = self.nodes[e + 1] - self.nodes[e]

            # Local mass matrix (linear elements)
            M_local = h / 6 * np.array([[2, 1], [1, 2]])

            # Assemble
            M[e:e+2, e:e+2] += M_local

        return M

    def load_vector(self, f: Callable) -> np.ndarray:
        """
        Assemble load vector from source function.

        Args:
            f: Source function f(x)

        Returns:
            Load vector
        """
        b = np.zeros(self.n_nodes)

        for e in range(self.n_elements):
            x1, x2 = self.nodes[e], self.nodes[e + 1]
            h = x2 - x1

            # Simpson's rule for integral
            f1 = f(x1)
            f2 = f((x1 + x2) / 2)
            f3 = f(x2)

            b[e] += h / 6 * (f1 + 2 * f2)
            b[e + 1] += h / 6 * (2 * f2 + f3)

        return b

    def solve_poisson(self, f: Callable, u_left: float = 0,
                      u_right: float = 0) -> np.ndarray:
        """
        Solve -u'' = f with Dirichlet BC.

        Args:
            f: Source function
            u_left, u_right: Boundary values

        Returns:
            Solution at nodes
        """
        K = self.stiffness_matrix()
        b = self.load_vector(f)

        # Apply Dirichlet BC
        # Modify first and last equations
        b -= K[:, 0] * u_left
        b -= K[:, -1] * u_right

        # Solve interior problem
        K_int = K[1:-1, 1:-1]
        b_int = b[1:-1]

        u_int = np.linalg.solve(K_int, b_int)

        return np.concatenate([[u_left], u_int, [u_right]])


class SpectralMethod:
    """
    Spectral method solver using FFT.

    Solves PDEs using Fourier basis functions for
    periodic domains.

    Args:
        n: Number of grid points
        L: Domain length
    """

    def __init__(self, n: int, L: float = 2 * np.pi):
        self.n = n
        self.L = L
        self.dx = L / n
        self.x = np.linspace(0, L, n, endpoint=False)

        # Wavenumbers
        self.k = 2 * np.pi * np.fft.fftfreq(n, self.dx)

    def derivative(self, u: np.ndarray, order: int = 1) -> np.ndarray:
        """
        Compute derivative using FFT.

        Args:
            u: Function values
            order: Derivative order

        Returns:
            Derivative values
        """
        u_hat = np.fft.fft(u)
        du_hat = (1j * self.k)**order * u_hat
        return np.real(np.fft.ifft(du_hat))

    def solve_poisson(self, f: np.ndarray) -> np.ndarray:
        """
        Solve Poisson equation: u'' = f (periodic BC).

        Args:
            f: Right-hand side

        Returns:
            Solution u
        """
        f_hat = np.fft.fft(f)

        # Solve in Fourier space: -k² û = f̂
        with np.errstate(divide='ignore', invalid='ignore'):
            u_hat = -f_hat / self.k**2
            u_hat[0] = 0  # Fix mean value

        return np.real(np.fft.ifft(u_hat))

    def solve_heat_equation(self, u0: np.ndarray, dt: float, n_steps: int,
                            alpha: float = 1.0) -> np.ndarray:
        """
        Solve heat equation spectrally.

        Args:
            u0: Initial condition
            dt: Time step
            alpha: Thermal diffusivity
            n_steps: Number of steps

        Returns:
            Solution at final time
        """
        u_hat = np.fft.fft(u0)

        # Exact solution in Fourier space
        decay = np.exp(-alpha * self.k**2 * dt * n_steps)
        u_hat_final = u_hat * decay

        return np.real(np.fft.ifft(u_hat_final))


class CrankNicolson:
    """
    Crank-Nicolson implicit time stepping.

    (I - θΔt A)u^{n+1} = (I + (1-θ)Δt A)u^n

    with θ = 0.5 for Crank-Nicolson.

    Args:
        matrix_A: Spatial operator matrix A
        theta: Implicitness parameter (0.5 = CN)
    """

    def __init__(self, matrix_A: sparse.spmatrix, theta: float = 0.5):
        self.A = matrix_A
        self.theta = theta
        self.n = matrix_A.shape[0]

    def step(self, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Advance solution by one time step.

        Args:
            u: Current solution
            dt: Time step

        Returns:
            Solution at next time step
        """
        I = sparse.eye(self.n)

        lhs = I - self.theta * dt * self.A
        rhs = I + (1 - self.theta) * dt * self.A

        return sparse_linalg.spsolve(lhs, rhs @ u)

    def solve(self, u0: np.ndarray, dt: float, n_steps: int) -> np.ndarray:
        """
        Solve for multiple time steps.

        Args:
            u0: Initial condition
            dt: Time step
            n_steps: Number of steps

        Returns:
            Final solution
        """
        u = u0.copy()
        for _ in range(n_steps):
            u = self.step(u, dt)
        return u


class ADIMethod:
    """
    Alternating Direction Implicit (ADI) method for 2D problems.

    Splits 2D operator into 1D sweeps for efficiency.

    Args:
        nx, ny: Grid dimensions
        dx, dy: Grid spacing
    """

    def __init__(self, nx: int, ny: int, dx: float, dy: float):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy

    def _tridiagonal_solve(self, a: np.ndarray, b: np.ndarray,
                           c: np.ndarray, d: np.ndarray) -> np.ndarray:
        """
        Thomas algorithm for tridiagonal systems.

        Args:
            a: Lower diagonal
            b: Main diagonal
            c: Upper diagonal
            d: Right-hand side

        Returns:
            Solution vector
        """
        n = len(d)
        c_star = np.zeros(n)
        d_star = np.zeros(n)

        c_star[0] = c[0] / b[0]
        d_star[0] = d[0] / b[0]

        for i in range(1, n):
            denom = b[i] - a[i] * c_star[i - 1]
            c_star[i] = c[i] / denom if i < n - 1 else 0
            d_star[i] = (d[i] - a[i] * d_star[i - 1]) / denom

        x = np.zeros(n)
        x[-1] = d_star[-1]

        for i in range(n - 2, -1, -1):
            x[i] = d_star[i] - c_star[i] * x[i + 1]

        return x

    def step_heat_equation(self, u: np.ndarray, dt: float,
                           alpha: float = 1.0) -> np.ndarray:
        """
        One ADI step for heat equation.

        Args:
            u: Current solution (2D array)
            dt: Time step
            alpha: Thermal diffusivity

        Returns:
            Solution after one step
        """
        rx = alpha * dt / (2 * self.dx**2)
        ry = alpha * dt / (2 * self.dy**2)

        u_half = np.zeros_like(u)
        u_new = np.zeros_like(u)

        # X-sweep (implicit in x, explicit in y)
        for j in range(1, self.ny - 1):
            a = -rx * np.ones(self.nx - 2)
            b = (1 + 2 * rx) * np.ones(self.nx - 2)
            c = -rx * np.ones(self.nx - 2)

            d = ry * u[2:, j] + (1 - 2 * ry) * u[1:-1, j] + ry * u[:-2, j]

            u_half[1:-1, j] = self._tridiagonal_solve(a, b, c, d)

        # Y-sweep (implicit in y, explicit in x)
        for i in range(1, self.nx - 1):
            a = -ry * np.ones(self.ny - 2)
            b = (1 + 2 * ry) * np.ones(self.ny - 2)
            c = -ry * np.ones(self.ny - 2)

            d = rx * u_half[i + 1, 2:] + (1 - 2 * rx) * u_half[i, 1:-1] + rx * u_half[i - 1, :-2]

            u_new[i, 1:-1] = self._tridiagonal_solve(a, b, c, d)

        return u_new


# =============================================================================
# Linear Algebra
# =============================================================================

class SparseMatrix:
    """
    Utilities for sparse matrix operations.

    Args:
        data: Sparse matrix (scipy.sparse format)
    """

    def __init__(self, data: sparse.spmatrix):
        self.matrix = sparse.csr_matrix(data)
        self.shape = self.matrix.shape

    @classmethod
    def from_dense(cls, dense: np.ndarray, threshold: float = 1e-10) -> 'SparseMatrix':
        """
        Create sparse matrix from dense, dropping small entries.

        Args:
            dense: Dense matrix
            threshold: Values below this are dropped

        Returns:
            SparseMatrix instance
        """
        dense[np.abs(dense) < threshold] = 0
        return cls(sparse.csr_matrix(dense))

    @classmethod
    def from_diagonals(cls, diagonals: List[np.ndarray],
                       offsets: List[int], shape: Tuple[int, int]) -> 'SparseMatrix':
        """
        Create sparse matrix from diagonals.

        Args:
            diagonals: List of diagonal arrays
            offsets: Offset of each diagonal
            shape: Matrix shape

        Returns:
            SparseMatrix instance
        """
        return cls(sparse.diags(diagonals, offsets, shape=shape))

    def sparsity(self) -> float:
        """Return fraction of zero entries."""
        return 1 - self.matrix.nnz / np.prod(self.shape)

    def to_dense(self) -> np.ndarray:
        """Convert to dense array."""
        return self.matrix.toarray()

    def __matmul__(self, other: np.ndarray) -> np.ndarray:
        """Matrix-vector multiplication."""
        return self.matrix @ other


class ConjugateGradient:
    """
    Conjugate gradient solver for Ax = b.

    Requires A to be symmetric positive definite.

    Args:
        tol: Convergence tolerance
        max_iter: Maximum iterations
    """

    def __init__(self, tol: float = 1e-10, max_iter: int = 1000):
        self.tol = tol
        self.max_iter = max_iter

    def solve(self, A: sparse.spmatrix, b: np.ndarray,
              x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, int]:
        """
        Solve Ax = b using conjugate gradient.

        Args:
            A: Sparse matrix
            b: Right-hand side
            x0: Initial guess

        Returns:
            (solution, iterations)
        """
        if x0 is None:
            x = np.zeros_like(b)
        else:
            x = x0.copy()

        r = b - A @ x
        p = r.copy()
        rs_old = np.dot(r, r)

        for i in range(self.max_iter):
            Ap = A @ p
            alpha = rs_old / np.dot(p, Ap)

            x += alpha * p
            r -= alpha * Ap

            rs_new = np.dot(r, r)

            if np.sqrt(rs_new) < self.tol:
                return x, i + 1

            p = r + (rs_new / rs_old) * p
            rs_old = rs_new

        warnings.warn(f"CG did not converge in {self.max_iter} iterations")
        return x, self.max_iter

    def solve_preconditioned(self, A: sparse.spmatrix, b: np.ndarray,
                             M_inv: Callable) -> Tuple[np.ndarray, int]:
        """
        Preconditioned conjugate gradient.

        Args:
            A: Sparse matrix
            b: Right-hand side
            M_inv: Function applying preconditioner inverse

        Returns:
            (solution, iterations)
        """
        x = np.zeros_like(b)
        r = b - A @ x
        z = M_inv(r)
        p = z.copy()
        rz_old = np.dot(r, z)

        for i in range(self.max_iter):
            Ap = A @ p
            alpha = rz_old / np.dot(p, Ap)

            x += alpha * p
            r -= alpha * Ap

            if np.linalg.norm(r) < self.tol:
                return x, i + 1

            z = M_inv(r)
            rz_new = np.dot(r, z)

            p = z + (rz_new / rz_old) * p
            rz_old = rz_new

        return x, self.max_iter


class GMRES:
    """
    Generalized Minimal Residual (GMRES) solver.

    Works for general (non-symmetric) matrices.

    Args:
        tol: Convergence tolerance
        max_iter: Maximum iterations
        restart: Restart frequency
    """

    def __init__(self, tol: float = 1e-10, max_iter: int = 1000,
                 restart: int = 50):
        self.tol = tol
        self.max_iter = max_iter
        self.restart = restart

    def solve(self, A: sparse.spmatrix, b: np.ndarray,
              x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, int]:
        """
        Solve Ax = b using GMRES.

        Args:
            A: Sparse matrix
            b: Right-hand side
            x0: Initial guess

        Returns:
            (solution, iterations)
        """
        if x0 is None:
            x0 = np.zeros_like(b)

        x, info = sparse_linalg.gmres(
            A, b, x0=x0, tol=self.tol,
            maxiter=self.max_iter, restart=self.restart
        )

        return x, info


class EigenSolver:
    """
    Large sparse eigenvalue solver.

    Args:
        method: 'arnoldi' or 'lanczos'
    """

    def __init__(self, method: str = 'arnoldi'):
        self.method = method

    def smallest_eigenvalues(self, A: sparse.spmatrix, k: int = 6,
                              which: str = 'SM') -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k smallest eigenvalues.

        Args:
            A: Sparse matrix
            k: Number of eigenvalues
            which: 'SM' (smallest magnitude), 'SA' (smallest algebraic)

        Returns:
            (eigenvalues, eigenvectors)
        """
        vals, vecs = sparse_linalg.eigsh(A, k=k, which=which)
        idx = np.argsort(vals)
        return vals[idx], vecs[:, idx]

    def largest_eigenvalues(self, A: sparse.spmatrix, k: int = 6) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k largest eigenvalues.

        Args:
            A: Sparse matrix
            k: Number of eigenvalues

        Returns:
            (eigenvalues, eigenvectors)
        """
        vals, vecs = sparse_linalg.eigsh(A, k=k, which='LM')
        idx = np.argsort(-vals)
        return vals[idx], vecs[:, idx]

    def spectral_radius(self, A: sparse.spmatrix) -> float:
        """
        Compute spectral radius (largest |λ|).

        Args:
            A: Sparse matrix

        Returns:
            Spectral radius
        """
        vals, _ = sparse_linalg.eigs(A, k=1, which='LM')
        return np.abs(vals[0])


class SVDSolver:
    """
    Singular Value Decomposition utilities.

    Args:
        n_components: Number of singular values to compute
    """

    def __init__(self, n_components: int = 6):
        self.n_components = n_components

    def truncated_svd(self, A: sparse.spmatrix) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute truncated SVD for sparse matrices.

        Args:
            A: Sparse matrix

        Returns:
            (U, s, Vt) where A ≈ U @ diag(s) @ Vt
        """
        U, s, Vt = sparse_linalg.svds(A, k=self.n_components)
        # Sort by decreasing singular value
        idx = np.argsort(-s)
        return U[:, idx], s[idx], Vt[idx, :]

    def low_rank_approximation(self, A: sparse.spmatrix,
                               rank: int) -> np.ndarray:
        """
        Compute low-rank approximation.

        Args:
            A: Sparse matrix
            rank: Target rank

        Returns:
            Low-rank approximation as dense matrix
        """
        self.n_components = rank
        U, s, Vt = self.truncated_svd(A)
        return U @ np.diag(s) @ Vt

    def condition_number(self, A: sparse.spmatrix) -> float:
        """
        Estimate condition number from largest/smallest singular values.

        Args:
            A: Sparse matrix

        Returns:
            Condition number estimate
        """
        s_largest = sparse_linalg.svds(A, k=1, which='LM')[1][0]
        s_smallest = sparse_linalg.svds(A, k=1, which='SM')[1][0]

        return s_largest / s_smallest


# Module exports
__all__ = [
    # Monte Carlo
    'ImportanceSampling', 'MarkovChainMC', 'PathIntegralMC', 'MonteCarloIntegration',
    # Molecular Dynamics
    'VelocityVerlet', 'Thermostat', 'Barostat', 'PeriodicBoundary', 'CellList', 'EwaldSum',
    # PDE Solvers
    'FiniteDifference1D', 'FiniteDifference2D', 'FiniteElement1D',
    'SpectralMethod', 'CrankNicolson', 'ADIMethod',
    # Linear Algebra
    'SparseMatrix', 'ConjugateGradient', 'GMRES', 'EigenSolver', 'SVDSolver',
]
