"""
Quantum Field Theory Foundations

This module provides fundamental QFT tools including:
- Classical Field Theory: Scalar, vector, and spinor fields
- Canonical Quantization: Fock space, commutators, normal ordering
- Propagators & Diagrams: Feynman propagator, vertices, cross sections
- Symmetries: Gauge symmetry, spontaneous symmetry breaking, Higgs mechanism
"""

import numpy as np
from typing import Optional, Callable, Tuple, List, Dict, Union
from numpy.typing import ArrayLike
from scipy.integrate import quad, dblquad
from scipy.special import gamma as gamma_func
import warnings


# =============================================================================
# Physical Constants
# =============================================================================

HBAR = 1.054571817e-34  # Reduced Planck constant (J·s)
C = 299792458.0  # Speed of light (m/s)


# =============================================================================
# Classical Field Theory
# =============================================================================

class ScalarField:
    """Klein-Gordon scalar field φ(x,t)"""

    def __init__(self, mass: float, grid: ArrayLike,
                 hbar: float = 1.0, c: float = 1.0):
        """
        Initialize scalar field

        Args:
            mass: Field mass parameter
            grid: Spatial grid points
            hbar: Reduced Planck constant (natural units default)
            c: Speed of light (natural units default)
        """
        self.mass = mass
        self.grid = np.array(grid)
        self.hbar = hbar
        self.c = c
        self.dx = grid[1] - grid[0] if len(grid) > 1 else 1.0
        self.n_points = len(grid)

        # Field and conjugate momentum
        self.phi = np.zeros(self.n_points, dtype=complex)
        self.pi = np.zeros(self.n_points, dtype=complex)  # π = ∂φ/∂t

    def set_initial_condition(self, phi0: ArrayLike, pi0: ArrayLike):
        """Set initial field configuration"""
        self.phi = np.array(phi0, dtype=complex)
        self.pi = np.array(pi0, dtype=complex)

    def klein_gordon_operator(self) -> np.ndarray:
        """
        Apply Klein-Gordon operator (□ + m²)φ

        Returns:
            Result of (∂²/∂t² - ∇² + m²)φ
        """
        # Laplacian using finite differences
        laplacian = np.zeros_like(self.phi)
        laplacian[1:-1] = (self.phi[2:] - 2*self.phi[1:-1] + self.phi[:-2]) / self.dx**2
        # Periodic boundary conditions
        laplacian[0] = (self.phi[1] - 2*self.phi[0] + self.phi[-1]) / self.dx**2
        laplacian[-1] = (self.phi[0] - 2*self.phi[-1] + self.phi[-2]) / self.dx**2

        return -laplacian + (self.mass * self.c / self.hbar)**2 * self.phi

    def evolve(self, dt: float, n_steps: int = 1):
        """
        Evolve field using leapfrog integration

        Args:
            dt: Time step
            n_steps: Number of steps
        """
        for _ in range(n_steps):
            # Half step for momentum
            kg_op = self.klein_gordon_operator()
            self.pi -= 0.5 * dt * kg_op

            # Full step for field
            self.phi += dt * self.pi

            # Half step for momentum
            kg_op = self.klein_gordon_operator()
            self.pi -= 0.5 * dt * kg_op

    def energy_density(self) -> np.ndarray:
        """
        Calculate energy density T^00

        Returns:
            Energy density at each grid point
        """
        # Gradient term
        grad_phi = np.gradient(self.phi, self.dx)

        # T^00 = (1/2)[π² + (∇φ)² + m²φ²]
        return 0.5 * (np.abs(self.pi)**2 +
                      np.abs(grad_phi)**2 +
                      (self.mass * self.c / self.hbar)**2 * np.abs(self.phi)**2)

    def total_energy(self) -> float:
        """Calculate total field energy"""
        return np.sum(self.energy_density()) * self.dx

    def momentum_density(self) -> np.ndarray:
        """
        Calculate momentum density T^0i

        Returns:
            Momentum density
        """
        grad_phi = np.gradient(self.phi, self.dx)
        return -np.real(self.pi * np.conj(grad_phi))


class VectorField:
    """Proca/Maxwell vector field A^μ(x,t)"""

    def __init__(self, mass: float, grid: ArrayLike,
                 hbar: float = 1.0, c: float = 1.0):
        """
        Initialize vector field (Proca field)

        Args:
            mass: Field mass (0 for Maxwell)
            grid: Spatial grid points
            hbar: Reduced Planck constant
            c: Speed of light
        """
        self.mass = mass
        self.grid = np.array(grid)
        self.hbar = hbar
        self.c = c
        self.dx = grid[1] - grid[0] if len(grid) > 1 else 1.0
        self.n_points = len(grid)

        # 4-potential A^μ = (A^0, A^1, A^2, A^3)
        # For 1D we use A^0, A^1
        self.A = np.zeros((2, self.n_points), dtype=complex)
        self.F = np.zeros((2, 2, self.n_points), dtype=complex)  # Field tensor

    def set_potential(self, A0: ArrayLike, A1: ArrayLike):
        """Set 4-potential components"""
        self.A[0] = np.array(A0, dtype=complex)
        self.A[1] = np.array(A1, dtype=complex)
        self._compute_field_tensor()

    def _compute_field_tensor(self):
        """Compute field strength tensor F_μν = ∂_μ A_ν - ∂_ν A_μ"""
        # F_01 = ∂_0 A_1 - ∂_1 A_0 = -E (electric field in 1D)
        dA0_dx = np.gradient(self.A[0], self.dx)
        self.F[0, 1] = -dA0_dx  # Electric field
        self.F[1, 0] = dA0_dx

    def electric_field(self) -> np.ndarray:
        """Get electric field E = -∂A^0/∂x - ∂A^1/∂t"""
        return -np.gradient(self.A[0], self.dx)

    def proca_equation(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate Proca equation: ∂_μ F^μν + m²A^ν = 0

        Returns:
            Residual for each component
        """
        m2 = (self.mass * self.c / self.hbar)**2

        # Simplified 1D version
        residual_0 = np.gradient(self.F[0, 1], self.dx) + m2 * self.A[0]
        residual_1 = m2 * self.A[1]  # No spatial gradient in 1D

        return residual_0, residual_1

    def energy_density(self) -> np.ndarray:
        """Calculate electromagnetic energy density"""
        E = self.electric_field()
        m2 = (self.mass * self.c / self.hbar)**2
        # T^00 = (1/2)[E² + m²(A^0² + A^1²)]
        return 0.5 * (np.abs(E)**2 + m2 * (np.abs(self.A[0])**2 + np.abs(self.A[1])**2))


class DiracField:
    """Dirac spinor field ψ(x,t)"""

    def __init__(self, mass: float, grid: ArrayLike,
                 hbar: float = 1.0, c: float = 1.0):
        """
        Initialize Dirac field

        Args:
            mass: Fermion mass
            grid: Spatial grid points
            hbar: Reduced Planck constant
            c: Speed of light
        """
        self.mass = mass
        self.grid = np.array(grid)
        self.hbar = hbar
        self.c = c
        self.dx = grid[1] - grid[0] if len(grid) > 1 else 1.0
        self.n_points = len(grid)

        # 4-component Dirac spinor
        self.psi = np.zeros((4, self.n_points), dtype=complex)

        # Gamma matrices (Dirac representation)
        self._init_gamma_matrices()

    def _init_gamma_matrices(self):
        """Initialize Dirac gamma matrices"""
        # γ^0
        self.gamma0 = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, -1]
        ], dtype=complex)

        # γ^1
        self.gamma1 = np.array([
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, -1, 0, 0],
            [-1, 0, 0, 0]
        ], dtype=complex)

        # γ^2
        self.gamma2 = np.array([
            [0, 0, 0, -1j],
            [0, 0, 1j, 0],
            [0, 1j, 0, 0],
            [-1j, 0, 0, 0]
        ], dtype=complex)

        # γ^3
        self.gamma3 = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, -1],
            [-1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=complex)

        # γ^5 = iγ^0γ^1γ^2γ^3
        self.gamma5 = 1j * self.gamma0 @ self.gamma1 @ self.gamma2 @ self.gamma3

    def set_spinor(self, psi: ArrayLike):
        """Set spinor field values"""
        self.psi = np.array(psi, dtype=complex)

    def dirac_operator(self) -> np.ndarray:
        """
        Apply Dirac operator (iγ^μ∂_μ - m)ψ

        Returns:
            Result of Dirac equation operator
        """
        m = self.mass * self.c / self.hbar

        # Spatial derivative
        dpsi_dx = np.zeros_like(self.psi)
        for i in range(4):
            dpsi_dx[i] = np.gradient(self.psi[i], self.dx)

        # (iγ^1 ∂_x - m)ψ  (in 1D, ignoring time derivative)
        result = np.zeros_like(self.psi)
        for i in range(4):
            result[i] = -m * self.psi[i]
            for j in range(4):
                result[i] += 1j * self.gamma1[i, j] * dpsi_dx[j]

        return result

    def probability_density(self) -> np.ndarray:
        """Calculate probability density ψ†ψ"""
        return np.sum(np.abs(self.psi)**2, axis=0)

    def current_density(self) -> np.ndarray:
        """Calculate current density j = ψ†γ^0γ^1ψ"""
        g01 = self.gamma0 @ self.gamma1
        psi_bar = np.conj(self.psi)

        j = np.zeros(self.n_points, dtype=complex)
        for i in range(4):
            for j_idx in range(4):
                j += psi_bar[i] * g01[i, j_idx] * self.psi[j_idx]

        return np.real(j)

    def helicity(self) -> np.ndarray:
        """Calculate helicity σ·p/|p| for each grid point"""
        # Simplified - uses γ^5 in chiral basis
        psi_bar = np.conj(self.psi)
        result = np.zeros(self.n_points, dtype=complex)
        for i in range(4):
            for j in range(4):
                result += psi_bar[i] * self.gamma5[i, j] * self.psi[j]
        return np.real(result) / (self.probability_density() + 1e-10)


class FieldLagrangian:
    """Lagrangian density construction for field theories"""

    def __init__(self, field_type: str = 'scalar'):
        """
        Initialize field Lagrangian

        Args:
            field_type: 'scalar', 'vector', 'dirac', or 'custom'
        """
        self.field_type = field_type

    def scalar_lagrangian(self, phi: np.ndarray, dphi_dt: np.ndarray,
                          grad_phi: np.ndarray, mass: float) -> np.ndarray:
        """
        Klein-Gordon Lagrangian: L = (1/2)(∂_μφ∂^μφ - m²φ²)

        Args:
            phi: Field values
            dphi_dt: Time derivative
            grad_phi: Spatial gradient
            mass: Field mass

        Returns:
            Lagrangian density at each point
        """
        return 0.5 * (np.abs(dphi_dt)**2 - np.sum(np.abs(grad_phi)**2, axis=0) -
                      mass**2 * np.abs(phi)**2)

    def dirac_lagrangian(self, psi: np.ndarray, psi_bar: np.ndarray,
                         dpsi_dxmu: np.ndarray, mass: float,
                         gamma_mu: List[np.ndarray]) -> np.ndarray:
        """
        Dirac Lagrangian: L = ψ̄(iγ^μ∂_μ - m)ψ

        Args:
            psi: Spinor field
            psi_bar: Adjoint spinor
            dpsi_dxmu: Derivatives [∂_0ψ, ∂_1ψ, ∂_2ψ, ∂_3ψ]
            mass: Fermion mass
            gamma_mu: Gamma matrices

        Returns:
            Lagrangian density
        """
        L = np.zeros(psi.shape[1], dtype=complex)

        # Kinetic term
        for mu in range(4):
            for i in range(4):
                for j in range(4):
                    L += 1j * psi_bar[i] * gamma_mu[mu][i, j] * dpsi_dxmu[mu, j]

        # Mass term
        for i in range(4):
            L -= mass * psi_bar[i] * psi[i]

        return np.real(L)

    def maxwell_lagrangian(self, E: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Maxwell Lagrangian: L = -(1/4)F_μν F^μν = (1/2)(E² - B²)

        Args:
            E: Electric field
            B: Magnetic field

        Returns:
            Lagrangian density
        """
        E_sq = np.sum(np.abs(E)**2, axis=0) if E.ndim > 1 else np.abs(E)**2
        B_sq = np.sum(np.abs(B)**2, axis=0) if B.ndim > 1 else np.abs(B)**2
        return 0.5 * (E_sq - B_sq)

    def phi4_lagrangian(self, phi: np.ndarray, dphi_dt: np.ndarray,
                        grad_phi: np.ndarray, mass: float,
                        lambda_coupling: float) -> np.ndarray:
        """
        φ⁴ theory Lagrangian: L = (1/2)(∂φ)² - (1/2)m²φ² - (λ/4!)φ⁴

        Args:
            phi: Field values
            dphi_dt: Time derivative
            grad_phi: Spatial gradient
            mass: Field mass
            lambda_coupling: Self-coupling constant

        Returns:
            Lagrangian density
        """
        kinetic = 0.5 * (np.abs(dphi_dt)**2 - np.sum(np.abs(grad_phi)**2, axis=0))
        mass_term = 0.5 * mass**2 * np.abs(phi)**2
        interaction = (lambda_coupling / 24) * np.abs(phi)**4

        return kinetic - mass_term - interaction


class EulerLagrangeField:
    """Euler-Lagrange equations for field theories"""

    def __init__(self, lagrangian_func: Callable):
        """
        Initialize Euler-Lagrange solver

        Args:
            lagrangian_func: Function L(φ, ∂φ, x) returning Lagrangian density
        """
        self.lagrangian = lagrangian_func

    def field_equation(self, phi: np.ndarray, dL_dphi: Callable,
                       dL_ddphi: Callable, grid: np.ndarray) -> np.ndarray:
        """
        Compute Euler-Lagrange equation: ∂L/∂φ - ∂_μ(∂L/∂(∂_μφ)) = 0

        Args:
            phi: Field values on grid
            dL_dphi: Derivative ∂L/∂φ
            dL_ddphi: Derivative ∂L/∂(∂φ)
            grid: Spatial grid

        Returns:
            Residual of field equation
        """
        dx = grid[1] - grid[0]

        # ∂L/∂φ term
        term1 = dL_dphi(phi)

        # ∂_μ(∂L/∂(∂_μφ)) term
        momentum = dL_ddphi(phi)
        div_momentum = np.gradient(momentum, dx)

        return term1 - div_momentum

    def canonical_momentum(self, dL_ddot_phi: Callable, phi_dot: np.ndarray) -> np.ndarray:
        """
        Calculate canonical momentum π = ∂L/∂(∂_0φ)

        Args:
            dL_ddot_phi: Derivative ∂L/∂φ̇
            phi_dot: Time derivative of field

        Returns:
            Canonical momentum field
        """
        return dL_ddot_phi(phi_dot)


# =============================================================================
# Canonical Quantization
# =============================================================================

class FieldCommutator:
    """Equal-time commutation relations [φ(x), π(y)]"""

    def __init__(self, grid: np.ndarray):
        """
        Initialize commutator calculator

        Args:
            grid: Spatial grid points
        """
        self.grid = np.array(grid)
        self.dx = grid[1] - grid[0] if len(grid) > 1 else 1.0

    def canonical_commutator(self, x_idx: int, y_idx: int) -> complex:
        """
        Calculate [φ(x), π(y)] = iℏδ(x-y)

        Args:
            x_idx: Grid index for x
            y_idx: Grid index for y

        Returns:
            Commutator value (iℏ δ_{xy})
        """
        if x_idx == y_idx:
            return 1j / self.dx  # Delta function normalization
        return 0j

    def commutator_matrix(self) -> np.ndarray:
        """
        Build full commutator matrix [φ(x_i), π(x_j)]

        Returns:
            N×N matrix of commutator values
        """
        n = len(self.grid)
        comm = np.zeros((n, n), dtype=complex)
        for i in range(n):
            comm[i, i] = 1j / self.dx
        return comm

    def creation_annihilation_commutator(self, k1: float, k2: float) -> complex:
        """
        Calculate [a(k), a†(k')] = δ(k-k')

        Args:
            k1, k2: Momentum values

        Returns:
            Commutator (approximation of delta function)
        """
        dk = 2 * np.pi / (self.grid[-1] - self.grid[0])
        if abs(k1 - k2) < dk / 2:
            return 1.0 / dk
        return 0j


class FockSpace:
    """Fock space for quantum field theory"""

    def __init__(self, n_modes: int, max_occupation: int = 10):
        """
        Initialize Fock space

        Args:
            n_modes: Number of momentum modes
            max_occupation: Maximum particles per mode
        """
        self.n_modes = n_modes
        self.max_occupation = max_occupation
        self.dim = (max_occupation + 1) ** n_modes

    def vacuum_state(self) -> np.ndarray:
        """
        Get vacuum state |0⟩

        Returns:
            Vacuum state vector
        """
        state = np.zeros(self.dim, dtype=complex)
        state[0] = 1.0  # |0,0,0,...⟩
        return state

    def occupation_to_index(self, occupation: List[int]) -> int:
        """
        Convert occupation numbers to state index

        Args:
            occupation: List of occupation numbers for each mode

        Returns:
            Index in Fock space
        """
        idx = 0
        for i, n in enumerate(occupation):
            idx += n * (self.max_occupation + 1) ** i
        return idx

    def index_to_occupation(self, idx: int) -> List[int]:
        """
        Convert state index to occupation numbers

        Args:
            idx: Index in Fock space

        Returns:
            List of occupation numbers
        """
        occupation = []
        for _ in range(self.n_modes):
            occupation.append(idx % (self.max_occupation + 1))
            idx //= (self.max_occupation + 1)
        return occupation

    def creation_operator(self, mode: int) -> np.ndarray:
        """
        Get creation operator a†_k for given mode

        Args:
            mode: Mode index

        Returns:
            Creation operator matrix
        """
        a_dag = np.zeros((self.dim, self.dim), dtype=complex)

        for idx in range(self.dim):
            occ = self.index_to_occupation(idx)
            if occ[mode] < self.max_occupation:
                new_occ = occ.copy()
                new_occ[mode] += 1
                new_idx = self.occupation_to_index(new_occ)
                a_dag[new_idx, idx] = np.sqrt(occ[mode] + 1)

        return a_dag

    def annihilation_operator(self, mode: int) -> np.ndarray:
        """
        Get annihilation operator a_k for given mode

        Args:
            mode: Mode index

        Returns:
            Annihilation operator matrix
        """
        a = np.zeros((self.dim, self.dim), dtype=complex)

        for idx in range(self.dim):
            occ = self.index_to_occupation(idx)
            if occ[mode] > 0:
                new_occ = occ.copy()
                new_occ[mode] -= 1
                new_idx = self.occupation_to_index(new_occ)
                a[new_idx, idx] = np.sqrt(occ[mode])

        return a

    def number_operator(self, mode: int) -> np.ndarray:
        """
        Get number operator n_k = a†_k a_k for given mode

        Args:
            mode: Mode index

        Returns:
            Number operator matrix
        """
        a = self.annihilation_operator(mode)
        a_dag = self.creation_operator(mode)
        return a_dag @ a

    def total_number_operator(self) -> np.ndarray:
        """Get total number operator N = Σ_k n_k"""
        N = np.zeros((self.dim, self.dim), dtype=complex)
        for mode in range(self.n_modes):
            N += self.number_operator(mode)
        return N


class VacuumState:
    """Vacuum state properties and calculations"""

    def __init__(self, fock_space: FockSpace):
        """
        Initialize vacuum state calculator

        Args:
            fock_space: Fock space instance
        """
        self.fock = fock_space
        self.state = fock_space.vacuum_state()

    def is_annihilated(self, mode: int) -> bool:
        """Check that a_k |0⟩ = 0"""
        a = self.fock.annihilation_operator(mode)
        result = a @ self.state
        return np.allclose(result, 0)

    def vacuum_energy(self, omega: ArrayLike) -> float:
        """
        Calculate zero-point energy (1/2)Σℏω_k

        Args:
            omega: Mode frequencies

        Returns:
            Vacuum energy (divergent, needs regularization)
        """
        return 0.5 * np.sum(omega)

    def vacuum_fluctuations(self, phi_operator: np.ndarray) -> float:
        """
        Calculate vacuum fluctuations ⟨0|φ²|0⟩

        Args:
            phi_operator: Field operator matrix

        Returns:
            Vacuum expectation value of φ²
        """
        phi_sq = phi_operator @ phi_operator
        return np.real(np.conj(self.state) @ phi_sq @ self.state)


class NormalOrdering:
    """Normal ordering of field operators"""

    def __init__(self, fock_space: FockSpace):
        """
        Initialize normal ordering calculator

        Args:
            fock_space: Fock space instance
        """
        self.fock = fock_space

    def normal_order(self, operator: np.ndarray) -> np.ndarray:
        """
        Normal order an operator (all a† to the left of all a)

        This is an approximate implementation for simple cases.

        Args:
            operator: Operator matrix

        Returns:
            Normal ordered operator
        """
        # For simple cases, subtract vacuum expectation value
        vac = self.fock.vacuum_state()
        vev = np.real(np.conj(vac) @ operator @ vac)
        return operator - vev * np.eye(operator.shape[0])

    def vacuum_subtraction(self, operator: np.ndarray) -> np.ndarray:
        """
        Subtract vacuum expectation value :O: = O - ⟨0|O|0⟩

        Args:
            operator: Operator matrix

        Returns:
            Vacuum-subtracted operator
        """
        vac = self.fock.vacuum_state()
        vev = np.real(np.conj(vac) @ operator @ vac)
        return operator - vev * np.eye(operator.shape[0])


class WickTheorem:
    """Wick's theorem for operator products"""

    def __init__(self, fock_space: FockSpace):
        """
        Initialize Wick's theorem calculator

        Args:
            fock_space: Fock space instance
        """
        self.fock = fock_space

    def contraction(self, a: np.ndarray, b: np.ndarray) -> complex:
        """
        Calculate Wick contraction ⟨0|T{AB}|0⟩ - :AB:

        Args:
            a, b: Operator matrices

        Returns:
            Contraction value
        """
        vac = self.fock.vacuum_state()

        # Time-ordered product (simplified - assumes a before b)
        Tab = a @ b
        Tab_vev = np.conj(vac) @ Tab @ vac

        # Normal ordered product
        normal = NormalOrdering(self.fock)
        ab_normal = normal.normal_order(a @ b)
        ab_normal_vev = np.conj(vac) @ ab_normal @ vac

        return Tab_vev - ab_normal_vev

    def two_point_function(self, phi1: np.ndarray, phi2: np.ndarray) -> complex:
        """
        Calculate two-point function ⟨0|T{φ(x)φ(y)}|0⟩

        Args:
            phi1, phi2: Field operators at positions x, y

        Returns:
            Two-point correlation function
        """
        vac = self.fock.vacuum_state()
        return np.conj(vac) @ (phi1 @ phi2) @ vac


# =============================================================================
# Propagators & Feynman Diagrams
# =============================================================================

class FeynmanPropagator:
    """Feynman propagator for various field theories"""

    def __init__(self, mass: float, hbar: float = 1.0, c: float = 1.0):
        """
        Initialize propagator

        Args:
            mass: Particle mass
            hbar: Reduced Planck constant
            c: Speed of light
        """
        self.mass = mass
        self.hbar = hbar
        self.c = c

    def scalar_propagator(self, p_squared: float, epsilon: float = 1e-10) -> complex:
        """
        Scalar (Klein-Gordon) propagator: i/(p² - m² + iε)

        Args:
            p_squared: 4-momentum squared p² = E²/c² - |p|²
            epsilon: Feynman iε prescription

        Returns:
            Propagator value
        """
        m2 = (self.mass * self.c / self.hbar)**2
        return 1j / (p_squared - m2 + 1j * epsilon)

    def fermion_propagator(self, p: ArrayLike, epsilon: float = 1e-10) -> np.ndarray:
        """
        Fermion (Dirac) propagator: i(γ·p + m)/(p² - m² + iε)

        Args:
            p: 4-momentum [E/c, px, py, pz]
            epsilon: Feynman iε prescription

        Returns:
            4×4 propagator matrix
        """
        p = np.array(p)
        m = self.mass * self.c / self.hbar
        p_sq = p[0]**2 - np.sum(p[1:]**2)

        # Gamma matrices
        gamma = [
            np.diag([1, 1, -1, -1]),  # γ^0
            np.array([[0,0,0,1],[0,0,1,0],[0,-1,0,0],[-1,0,0,0]]),  # γ^1
            np.array([[0,0,0,-1j],[0,0,1j,0],[0,1j,0,0],[-1j,0,0,0]]),  # γ^2
            np.array([[0,0,1,0],[0,0,0,-1],[-1,0,0,0],[0,1,0,0]])  # γ^3
        ]

        # γ·p = γ^μ p_μ
        slash_p = p[0] * gamma[0]
        for i in range(3):
            slash_p -= p[i+1] * gamma[i+1]

        numerator = slash_p + m * np.eye(4)
        denominator = p_sq - m**2 + 1j * epsilon

        return 1j * numerator / denominator

    def photon_propagator(self, k_squared: float, gauge: str = 'feynman',
                          epsilon: float = 1e-10) -> np.ndarray:
        """
        Photon propagator in various gauges

        Args:
            k_squared: 4-momentum squared
            gauge: 'feynman', 'landau', or 'coulomb'
            epsilon: iε prescription

        Returns:
            4×4 propagator tensor
        """
        if gauge == 'feynman':
            # -iη_μν / (k² + iε)
            return -1j * np.diag([1, -1, -1, -1]) / (k_squared + 1j * epsilon)
        elif gauge == 'landau':
            # (η_μν - k_μk_ν/k²) / k²
            # Simplified - returns same as Feynman for demonstration
            return -1j * np.diag([1, -1, -1, -1]) / (k_squared + 1j * epsilon)
        else:
            raise ValueError(f"Unknown gauge: {gauge}")

    def position_space(self, x: float, t: float, d: int = 3) -> complex:
        """
        Propagator in position space (Euclidean, for demonstration)

        D(x,t) = ∫ d^d p/(2π)^d e^{ip·x} D(p)

        Args:
            x: Spatial distance
            t: Time difference
            d: Spatial dimensions

        Returns:
            Position space propagator
        """
        # Simplified: Yukawa-like decay
        m = self.mass * self.c / self.hbar
        r = np.sqrt(x**2 + (self.c * t)**2)

        if d == 3 and r > 0:
            return np.exp(-m * r) / (4 * np.pi * r)
        return 0j


class FeynmanVertex:
    """Feynman diagram vertices and rules"""

    def __init__(self, coupling: float):
        """
        Initialize vertex

        Args:
            coupling: Coupling constant
        """
        self.coupling = coupling

    def phi4_vertex(self) -> complex:
        """
        φ⁴ theory vertex: -iλ

        Returns:
            Vertex factor
        """
        return -1j * self.coupling

    def qed_vertex(self) -> np.ndarray:
        """
        QED vertex: -ieγ^μ

        Returns:
            Vertex factor (4×4 matrix per Lorentz index)
        """
        gamma = [
            np.diag([1, 1, -1, -1], dtype=complex),
            np.array([[0,0,0,1],[0,0,1,0],[0,-1,0,0],[-1,0,0,0]], dtype=complex),
            np.array([[0,0,0,-1j],[0,0,1j,0],[0,1j,0,0],[-1j,0,0,0]], dtype=complex),
            np.array([[0,0,1,0],[0,0,0,-1],[-1,0,0,0],[0,1,0,0]], dtype=complex)
        ]
        return [-1j * self.coupling * g for g in gamma]

    def yukawa_vertex(self) -> np.ndarray:
        """
        Yukawa vertex: -ig

        Returns:
            Vertex factor (4×4 identity)
        """
        return -1j * self.coupling * np.eye(4)

    def symmetry_factor(self, n_identical_lines: int, n_vertices: int) -> float:
        """
        Calculate symmetry factor for diagram

        Args:
            n_identical_lines: Number of identical internal lines
            n_vertices: Number of vertices

        Returns:
            Symmetry factor
        """
        # Simplified calculation
        return 1.0 / (np.math.factorial(n_identical_lines))


class FeynmanDiagram:
    """Feynman diagram representation and calculation"""

    def __init__(self):
        """Initialize Feynman diagram"""
        self.vertices = []
        self.propagators = []
        self.external_legs = []

    def add_vertex(self, vertex: FeynmanVertex, position: int):
        """Add vertex to diagram"""
        self.vertices.append({'vertex': vertex, 'position': position})

    def add_propagator(self, propagator: FeynmanPropagator,
                       from_vertex: int, to_vertex: int):
        """Add internal propagator"""
        self.propagators.append({
            'propagator': propagator,
            'from': from_vertex,
            'to': to_vertex
        })

    def add_external(self, momentum: ArrayLike, incoming: bool = True):
        """Add external leg"""
        self.external_legs.append({
            'momentum': np.array(momentum),
            'incoming': incoming
        })

    def amplitude(self, momenta: Dict[int, np.ndarray]) -> complex:
        """
        Calculate diagram amplitude (simplified)

        Args:
            momenta: Dictionary mapping propagator index to momentum

        Returns:
            Diagram amplitude
        """
        amp = 1.0 + 0j

        # Multiply vertex factors
        for v in self.vertices:
            amp *= v['vertex'].phi4_vertex()

        # Multiply propagator factors
        for i, p in enumerate(self.propagators):
            if i in momenta:
                k = momenta[i]
                k_sq = k[0]**2 - np.sum(k[1:]**2)
                amp *= p['propagator'].scalar_propagator(k_sq)

        return amp


class CrossSection:
    """Cross section calculations from matrix elements"""

    def __init__(self, hbar: float = 1.0, c: float = 1.0):
        """
        Initialize cross section calculator

        Args:
            hbar: Reduced Planck constant
            c: Speed of light
        """
        self.hbar = hbar
        self.c = c

    def from_amplitude(self, M_squared: float, s: float,
                       masses: Tuple[float, float, float, float]) -> float:
        """
        Calculate 2→2 cross section from |M|²

        dσ/dΩ = |M|² / (64π² s) × |p_f|/|p_i|

        Args:
            M_squared: |M|² amplitude squared
            s: Mandelstam s (center of mass energy squared)
            masses: (m1, m2, m3, m4) particle masses

        Returns:
            Differential cross section
        """
        m1, m2, m3, m4 = masses

        # Initial momentum (in CM frame)
        E1_cm = (s + m1**2 - m2**2) / (2 * np.sqrt(s))
        p_i = np.sqrt(E1_cm**2 - m1**2)

        # Final momentum (in CM frame)
        E3_cm = (s + m3**2 - m4**2) / (2 * np.sqrt(s))
        if E3_cm**2 < m3**2:
            return 0.0  # Below threshold
        p_f = np.sqrt(E3_cm**2 - m3**2)

        return M_squared / (64 * np.pi**2 * s) * p_f / p_i

    def total_cross_section(self, M_squared_func: Callable,
                            s: float, masses: Tuple[float, float, float, float],
                            n_angles: int = 100) -> float:
        """
        Integrate differential cross section

        Args:
            M_squared_func: Function |M|²(cos θ)
            s: Mandelstam s
            masses: Particle masses
            n_angles: Integration points

        Returns:
            Total cross section
        """
        cos_theta = np.linspace(-1, 1, n_angles)
        dcos = cos_theta[1] - cos_theta[0]

        sigma_total = 0.0
        for ct in cos_theta:
            M_sq = M_squared_func(ct)
            dsigma = self.from_amplitude(M_sq, s, masses)
            sigma_total += 2 * np.pi * dsigma * dcos  # Integrate over φ and cos θ

        return sigma_total


class DecayRate:
    """Decay rate and lifetime calculations"""

    def __init__(self, hbar: float = HBAR, c: float = C):
        """
        Initialize decay rate calculator

        Args:
            hbar: Reduced Planck constant
            c: Speed of light
        """
        self.hbar = hbar
        self.c = c

    def two_body_decay(self, M_squared: float, M: float,
                       m1: float, m2: float) -> float:
        """
        Calculate 1→2 decay rate

        Γ = |M|² p / (8π M²)

        Args:
            M_squared: |M|² amplitude squared
            M: Decaying particle mass
            m1, m2: Daughter particle masses

        Returns:
            Decay rate (s⁻¹ in SI units, or natural units)
        """
        if M < m1 + m2:
            return 0.0  # Kinematically forbidden

        # Momentum of decay products in rest frame
        E1 = (M**2 + m1**2 - m2**2) / (2 * M)
        p = np.sqrt(E1**2 - m1**2)

        return M_squared * p / (8 * np.pi * M**2)

    def three_body_decay(self, M_squared_func: Callable, M: float,
                         m1: float, m2: float, m3: float,
                         n_points: int = 50) -> float:
        """
        Calculate 1→3 decay rate (phase space integration)

        Args:
            M_squared_func: Function |M|²(m12², m23²)
            M: Decaying particle mass
            m1, m2, m3: Daughter masses
            n_points: Integration points

        Returns:
            Decay rate
        """
        # Dalitz plot limits
        m12_min = (m1 + m2)**2
        m12_max = (M - m3)**2

        rate = 0.0
        dm12 = (m12_max - m12_min) / n_points

        for m12_sq in np.linspace(m12_min, m12_max, n_points):
            # m23² limits
            E2_star = (m12_sq - m1**2 + m2**2) / (2 * np.sqrt(m12_sq))
            E3_star = (M**2 - m12_sq - m3**2) / (2 * np.sqrt(m12_sq))

            if E2_star < m2 or E3_star < m3:
                continue

            p2_star = np.sqrt(E2_star**2 - m2**2)
            p3_star = np.sqrt(E3_star**2 - m3**2)

            m23_min = (E2_star + E3_star)**2 - (p2_star + p3_star)**2
            m23_max = (E2_star + E3_star)**2 - (p2_star - p3_star)**2

            dm23 = (m23_max - m23_min) / n_points

            for m23_sq in np.linspace(m23_min, m23_max, n_points):
                M_sq = M_squared_func(m12_sq, m23_sq)
                rate += M_sq * dm12 * dm23

        # Phase space factor
        return rate / (256 * np.pi**3 * M**3)

    def lifetime(self, decay_rate: float) -> float:
        """
        Calculate lifetime from decay rate

        τ = ℏ/Γ

        Args:
            decay_rate: Total decay rate

        Returns:
            Lifetime
        """
        if decay_rate <= 0:
            return np.inf
        return self.hbar / decay_rate


# =============================================================================
# Symmetries
# =============================================================================

class GlobalSymmetry:
    """Global symmetry and Noether current"""

    def __init__(self, symmetry_type: str = 'U1'):
        """
        Initialize global symmetry

        Args:
            symmetry_type: 'U1', 'SU2', 'SU3', etc.
        """
        self.symmetry_type = symmetry_type

    def u1_transformation(self, phi: np.ndarray, alpha: float) -> np.ndarray:
        """
        Apply U(1) transformation: φ → e^{iα}φ

        Args:
            phi: Complex field
            alpha: Phase angle

        Returns:
            Transformed field
        """
        return np.exp(1j * alpha) * phi

    def noether_current_u1(self, phi: np.ndarray, dphi_dt: np.ndarray,
                           grad_phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate U(1) Noether current

        j^μ = i(φ* ∂^μ φ - φ ∂^μ φ*)

        Args:
            phi: Field values
            dphi_dt: Time derivative
            grad_phi: Spatial gradient

        Returns:
            (j^0, j^i) current components
        """
        # j^0 = charge density
        j0 = 1j * (np.conj(phi) * dphi_dt - phi * np.conj(dphi_dt))

        # j^i = current density
        ji = 1j * (np.conj(phi) * grad_phi - phi * np.conj(grad_phi))

        return np.real(j0), np.real(ji)

    def conserved_charge(self, j0: np.ndarray, dx: float) -> float:
        """
        Calculate conserved charge Q = ∫ j^0 d³x

        Args:
            j0: Charge density
            dx: Grid spacing

        Returns:
            Total charge
        """
        return np.sum(j0) * dx


class LocalGaugeSymmetry:
    """Local gauge symmetry and gauge fields"""

    def __init__(self, gauge_group: str = 'U1', coupling: float = 1.0):
        """
        Initialize gauge symmetry

        Args:
            gauge_group: 'U1' (QED), 'SU2' (weak), 'SU3' (QCD)
            coupling: Gauge coupling constant
        """
        self.gauge_group = gauge_group
        self.coupling = coupling

    def covariant_derivative(self, phi: np.ndarray, A: np.ndarray,
                             grad_phi: np.ndarray) -> np.ndarray:
        """
        Calculate covariant derivative D_μ φ = ∂_μ φ - ig A_μ φ

        Args:
            phi: Matter field
            A: Gauge field
            grad_phi: Ordinary derivative

        Returns:
            Covariant derivative
        """
        return grad_phi - 1j * self.coupling * A * phi

    def gauge_transformation(self, phi: np.ndarray, A: np.ndarray,
                             alpha: np.ndarray, grad_alpha: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply gauge transformation

        φ → e^{igα} φ
        A_μ → A_μ + ∂_μ α

        Args:
            phi: Matter field
            A: Gauge field
            alpha: Gauge parameter
            grad_alpha: Gradient of gauge parameter

        Returns:
            (transformed phi, transformed A)
        """
        phi_new = np.exp(1j * self.coupling * alpha) * phi
        A_new = A + grad_alpha

        return phi_new, A_new

    def field_strength(self, A: np.ndarray, dA: np.ndarray) -> np.ndarray:
        """
        Calculate field strength F_μν = ∂_μ A_ν - ∂_ν A_μ

        For non-Abelian: F_μν = ∂_μ A_ν - ∂_ν A_μ - ig[A_μ, A_ν]

        Args:
            A: Gauge field components
            dA: Derivatives of gauge field

        Returns:
            Field strength tensor components
        """
        # Abelian case
        return dA[0] - dA[1]  # F_01 = ∂_0 A_1 - ∂_1 A_0


class SpontaneousSymmetryBreaking:
    """Spontaneous symmetry breaking (Mexican hat potential)"""

    def __init__(self, mu_squared: float, lambda_param: float):
        """
        Initialize SSB potential

        V(φ) = -μ²|φ|² + λ|φ|⁴

        Args:
            mu_squared: Mass parameter (> 0 for SSB)
            lambda_param: Self-coupling (> 0)
        """
        if mu_squared <= 0:
            raise ValueError("mu² must be positive for SSB")
        if lambda_param <= 0:
            raise ValueError("λ must be positive")

        self.mu_sq = mu_squared
        self.lambda_param = lambda_param

    def potential(self, phi: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate potential V(φ)

        Args:
            phi: Field value(s)

        Returns:
            Potential value(s)
        """
        phi_sq = np.abs(phi)**2
        return -self.mu_sq * phi_sq + self.lambda_param * phi_sq**2

    def vev(self) -> float:
        """
        Calculate vacuum expectation value

        v = √(μ²/(2λ))

        Returns:
            VEV magnitude
        """
        return np.sqrt(self.mu_sq / (2 * self.lambda_param))

    def higgs_mass(self) -> float:
        """
        Calculate Higgs boson mass

        m_H = √(2μ²) = √(2λ)v

        Returns:
            Higgs mass
        """
        return np.sqrt(2 * self.mu_sq)

    def goldstone_mass(self) -> float:
        """
        Goldstone boson mass (= 0 for exact symmetry)

        Returns:
            Goldstone mass (0)
        """
        return 0.0

    def expand_around_vev(self, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Expand field around VEV: φ = (v + h + iπ)/√2

        Args:
            phi: Complex field

        Returns:
            (h, π) Higgs and Goldstone components
        """
        v = self.vev()
        h = np.sqrt(2) * np.real(phi) - v
        pi = np.sqrt(2) * np.imag(phi)
        return h, pi


class GoldstoneBoson:
    """Goldstone boson from spontaneous symmetry breaking"""

    def __init__(self, ssb: SpontaneousSymmetryBreaking):
        """
        Initialize Goldstone boson

        Args:
            ssb: Spontaneous symmetry breaking instance
        """
        self.ssb = ssb

    def mass(self) -> float:
        """Goldstone mass is zero"""
        return 0.0

    def dispersion(self, k: float) -> float:
        """
        Goldstone dispersion relation ω = |k| (massless)

        Args:
            k: Momentum

        Returns:
            Energy
        """
        return np.abs(k)

    def decay_constant(self) -> float:
        """
        Get decay constant f = v

        Returns:
            Decay constant (equals VEV)
        """
        return self.ssb.vev()


class HiggsMechanism:
    """Higgs mechanism for gauge boson mass generation"""

    def __init__(self, gauge: LocalGaugeSymmetry, ssb: SpontaneousSymmetryBreaking):
        """
        Initialize Higgs mechanism

        Args:
            gauge: Gauge symmetry
            ssb: Spontaneous symmetry breaking
        """
        self.gauge = gauge
        self.ssb = ssb

    def gauge_boson_mass(self) -> float:
        """
        Calculate gauge boson mass m_A = gv

        Returns:
            Gauge boson mass
        """
        return self.gauge.coupling * self.ssb.vev()

    def higgs_mass(self) -> float:
        """
        Calculate physical Higgs mass

        Returns:
            Higgs boson mass
        """
        return self.ssb.higgs_mass()

    def goldstone_eaten(self) -> bool:
        """
        Check that Goldstone is eaten (gives gauge boson longitudinal DOF)

        Returns:
            True (Goldstone becomes longitudinal mode)
        """
        return True

    def degrees_of_freedom(self) -> Dict[str, int]:
        """
        Count degrees of freedom before and after SSB

        Returns:
            DOF count
        """
        return {
            'before': {
                'complex_scalar': 2,  # Real + imaginary parts
                'massless_gauge': 2   # Transverse polarizations only
            },
            'after': {
                'real_higgs': 1,
                'massive_gauge': 3    # 2 transverse + 1 longitudinal
            }
        }

    def unitary_gauge_lagrangian(self, h: np.ndarray, A: np.ndarray,
                                  grad_h: np.ndarray, F: np.ndarray) -> np.ndarray:
        """
        Lagrangian in unitary gauge (Goldstone gauged away)

        L = (1/2)(∂h)² - (1/2)m_H²h² - (1/4)F² + (1/2)m_A²A²

        Args:
            h: Higgs field
            A: Gauge field
            grad_h: Higgs gradient
            F: Field strength

        Returns:
            Lagrangian density
        """
        m_H = self.higgs_mass()
        m_A = self.gauge_boson_mass()

        higgs_kinetic = 0.5 * np.sum(np.abs(grad_h)**2, axis=0)
        higgs_mass = 0.5 * m_H**2 * np.abs(h)**2
        gauge_kinetic = 0.25 * np.abs(F)**2
        gauge_mass = 0.5 * m_A**2 * np.abs(A)**2

        return higgs_kinetic - higgs_mass - gauge_kinetic + gauge_mass
