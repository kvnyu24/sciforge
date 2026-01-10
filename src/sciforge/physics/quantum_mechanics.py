"""
Quantum Mechanics Complete Module

This module implements comprehensive quantum mechanics primitives including:
- Fundamental Operators
- Canonical Quantum Systems
- Angular Momentum
- Multi-particle Systems
- Approximation Methods
- Open Quantum Systems

References:
    - Sakurai & Napolitano, "Modern Quantum Mechanics"
    - Cohen-Tannoudji et al., "Quantum Mechanics"
    - Nielsen & Chuang, "Quantum Computation and Quantum Information"
"""

import numpy as np
from typing import Optional, Callable, Tuple, Union, List
from dataclasses import dataclass
from numpy.typing import ArrayLike
from scipy import linalg as la

from ..core.base import BaseClass
from ..core.utils import validate_positive
from ..core.exceptions import ValidationError, PhysicsError


# ==============================================================================
# Physical Constants
# ==============================================================================

HBAR = 1.054571817e-34    # Reduced Planck constant (J·s)
M_E = 9.10938370e-31      # Electron mass (kg)
E_CHARGE = 1.602176634e-19  # Elementary charge (C)
A_0 = 5.29177210903e-11   # Bohr radius (m)
EV_TO_J = 1.602176634e-19  # eV to Joules


# ==============================================================================
# Phase 5.1: Fundamental Operators
# ==============================================================================

class PositionOperator(BaseClass):
    """
    Position operator x̂ in position representation.

    In position basis, x̂ is diagonal: x̂|x⟩ = x|x⟩

    Args:
        grid: Position space grid points
        dimension: Spatial dimension (1, 2, or 3)
    """

    def __init__(self, grid: ArrayLike, dimension: int = 1):
        super().__init__()

        self.grid = np.array(grid)
        self.dimension = dimension
        self.N = len(grid)
        self.dx = grid[1] - grid[0] if len(grid) > 1 else 1.0

        # Position operator is diagonal in position basis
        self.matrix = np.diag(grid)

    def apply(self, psi: np.ndarray) -> np.ndarray:
        """Apply x̂ to wavefunction: x̂ψ(x) = x ψ(x)."""
        return self.grid * psi

    def expectation_value(self, psi: np.ndarray) -> float:
        """Calculate ⟨x⟩ = ⟨ψ|x̂|ψ⟩."""
        prob = np.abs(psi)**2
        return np.sum(self.grid * prob) * self.dx

    def variance(self, psi: np.ndarray) -> float:
        """Calculate ⟨(Δx)²⟩ = ⟨x²⟩ - ⟨x⟩²."""
        prob = np.abs(psi)**2
        x_avg = self.expectation_value(psi)
        x2_avg = np.sum(self.grid**2 * prob) * self.dx
        return x2_avg - x_avg**2

    def uncertainty(self, psi: np.ndarray) -> float:
        """Calculate uncertainty Δx = √⟨(Δx)²⟩."""
        return np.sqrt(self.variance(psi))


class MomentumOperator(BaseClass):
    """
    Momentum operator p̂ = -iℏ∇ in position representation.

    Uses finite differences or FFT for derivatives.

    Args:
        grid: Position space grid points
        hbar: Reduced Planck constant
        method: 'fft' or 'finite_difference'
    """

    def __init__(
        self,
        grid: ArrayLike,
        hbar: float = HBAR,
        method: str = 'fft'
    ):
        super().__init__()

        self.grid = np.array(grid)
        self.hbar = hbar
        self.method = method
        self.N = len(grid)
        self.dx = grid[1] - grid[0] if len(grid) > 1 else 1.0

        # Momentum space grid for FFT
        self.k = 2 * np.pi * np.fft.fftfreq(self.N, self.dx)
        self.p = hbar * self.k

    def apply(self, psi: np.ndarray) -> np.ndarray:
        """Apply p̂ to wavefunction: p̂ψ = -iℏ dψ/dx."""
        if self.method == 'fft':
            psi_k = np.fft.fft(psi)
            dpsi = np.fft.ifft(1j * self.k * psi_k)
            return -1j * self.hbar * dpsi
        else:
            # Central difference
            dpsi = np.zeros_like(psi, dtype=complex)
            dpsi[1:-1] = (psi[2:] - psi[:-2]) / (2 * self.dx)
            dpsi[0] = (psi[1] - psi[-1]) / (2 * self.dx)  # Periodic
            dpsi[-1] = (psi[0] - psi[-2]) / (2 * self.dx)
            return -1j * self.hbar * dpsi

    def expectation_value(self, psi: np.ndarray) -> float:
        """Calculate ⟨p⟩."""
        psi_k = np.fft.fft(psi) * self.dx
        prob_k = np.abs(psi_k)**2
        dk = self.k[1] - self.k[0] if len(self.k) > 1 else 1.0
        return np.sum(self.p * prob_k) * dk / (2 * np.pi)

    def variance(self, psi: np.ndarray) -> float:
        """Calculate ⟨(Δp)²⟩."""
        psi_k = np.fft.fft(psi) * self.dx
        prob_k = np.abs(psi_k)**2
        dk = self.k[1] - self.k[0] if len(self.k) > 1 else 1.0

        p_avg = self.expectation_value(psi)
        p2_avg = np.sum(self.p**2 * prob_k) * dk / (2 * np.pi)
        return p2_avg - p_avg**2

    def uncertainty(self, psi: np.ndarray) -> float:
        """Calculate Δp."""
        return np.sqrt(max(0, self.variance(psi)))


class AngularMomentumOperator(BaseClass):
    """
    Angular momentum operators L̂ = r̂ × p̂.

    L̂² and L̂z in spherical coordinates.

    Args:
        l_max: Maximum angular momentum quantum number
        hbar: Reduced Planck constant
    """

    def __init__(self, l_max: int, hbar: float = HBAR):
        super().__init__()

        self.l_max = l_max
        self.hbar = hbar

        # Dimension of Hilbert space
        self.dim = (l_max + 1)**2

    def L2_eigenvalue(self, l: int) -> float:
        """Return eigenvalue of L̂²: ℏ²l(l+1)."""
        return self.hbar**2 * l * (l + 1)

    def Lz_eigenvalue(self, m: int) -> float:
        """Return eigenvalue of L̂z: ℏm."""
        return self.hbar * m

    def Lz_matrix(self, l: int) -> np.ndarray:
        """Construct L̂z matrix for fixed l (dimension 2l+1)."""
        dim = 2 * l + 1
        m_vals = np.arange(l, -l - 1, -1)
        return self.hbar * np.diag(m_vals)

    def L_plus_matrix(self, l: int) -> np.ndarray:
        """Construct L̂₊ = L̂x + iL̂y raising operator."""
        dim = 2 * l + 1
        L_plus = np.zeros((dim, dim), dtype=complex)
        m_vals = np.arange(l, -l - 1, -1)

        for i in range(dim - 1):
            m = m_vals[i + 1]
            L_plus[i, i + 1] = self.hbar * np.sqrt(l * (l + 1) - m * (m + 1))

        return L_plus

    def L_minus_matrix(self, l: int) -> np.ndarray:
        """Construct L̂₋ = L̂x - iL̂y lowering operator."""
        dim = 2 * l + 1
        L_minus = np.zeros((dim, dim), dtype=complex)
        m_vals = np.arange(l, -l - 1, -1)

        for i in range(dim - 1):
            m = m_vals[i]
            L_minus[i + 1, i] = self.hbar * np.sqrt(l * (l + 1) - m * (m - 1))

        return L_minus

    def Lx_matrix(self, l: int) -> np.ndarray:
        """Construct L̂x = (L̂₊ + L̂₋)/2."""
        return (self.L_plus_matrix(l) + self.L_minus_matrix(l)) / 2

    def Ly_matrix(self, l: int) -> np.ndarray:
        """Construct L̂y = (L̂₊ - L̂₋)/(2i)."""
        return (self.L_plus_matrix(l) - self.L_minus_matrix(l)) / (2j)


class HamiltonianOperator(BaseClass):
    """
    Hamiltonian operator Ĥ = T̂ + V̂.

    Constructs Hamiltonian matrix for various potentials.

    Args:
        grid: Position space grid
        mass: Particle mass
        potential: Potential function V(x) or array
        hbar: Reduced Planck constant
    """

    def __init__(
        self,
        grid: ArrayLike,
        mass: float,
        potential: Union[Callable, ArrayLike],
        hbar: float = HBAR
    ):
        super().__init__()

        self.grid = np.array(grid)
        self.mass = mass
        self.hbar = hbar
        self.N = len(grid)
        self.dx = grid[1] - grid[0] if len(grid) > 1 else 1.0

        # Potential energy
        if callable(potential):
            self.V = np.array([potential(x) for x in grid])
        else:
            self.V = np.array(potential)

        # Construct Hamiltonian matrix
        self._build_matrix()

    def _build_matrix(self):
        """Build Hamiltonian matrix using finite differences."""
        N = self.N
        dx = self.dx

        # Kinetic energy (second derivative)
        T = np.zeros((N, N))
        coeff = -self.hbar**2 / (2 * self.mass * dx**2)

        for i in range(N):
            T[i, i] = -2 * coeff
            if i > 0:
                T[i, i-1] = coeff
            if i < N - 1:
                T[i, i+1] = coeff

        # Potential energy (diagonal)
        V_matrix = np.diag(self.V)

        self.H = T + V_matrix

    def eigenvalues(self, n_states: int = 10) -> np.ndarray:
        """Calculate first n energy eigenvalues."""
        eigenvals = np.linalg.eigvalsh(self.H)
        return np.sort(eigenvals)[:n_states]

    def eigenstates(self, n_states: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate first n energy eigenstates.

        Returns:
            (eigenvalues, eigenvectors) - eigenvectors are columns
        """
        eigenvals, eigenvecs = np.linalg.eigh(self.H)
        idx = np.argsort(eigenvals)[:n_states]
        return eigenvals[idx], eigenvecs[:, idx]

    def apply(self, psi: np.ndarray) -> np.ndarray:
        """Apply Hamiltonian to wavefunction."""
        return self.H @ psi

    def time_evolution_operator(self, dt: float) -> np.ndarray:
        """Calculate U(dt) = exp(-iĤdt/ℏ)."""
        return la.expm(-1j * self.H * dt / self.hbar)


class CreationOperator(BaseClass):
    """
    Creation (raising) operator â† for harmonic oscillator.

    â†|n⟩ = √(n+1)|n+1⟩

    Args:
        n_max: Maximum occupation number
    """

    def __init__(self, n_max: int):
        super().__init__()

        self.n_max = n_max
        self.dim = n_max + 1

        # Build matrix representation
        self.matrix = np.zeros((self.dim, self.dim), dtype=complex)
        for n in range(n_max):
            self.matrix[n + 1, n] = np.sqrt(n + 1)

    def apply(self, state: np.ndarray) -> np.ndarray:
        """Apply â† to state vector."""
        return self.matrix @ state


class AnnihilationOperator(BaseClass):
    """
    Annihilation (lowering) operator â for harmonic oscillator.

    â|n⟩ = √n|n-1⟩, â|0⟩ = 0

    Args:
        n_max: Maximum occupation number
    """

    def __init__(self, n_max: int):
        super().__init__()

        self.n_max = n_max
        self.dim = n_max + 1

        # Build matrix representation
        self.matrix = np.zeros((self.dim, self.dim), dtype=complex)
        for n in range(1, n_max + 1):
            self.matrix[n - 1, n] = np.sqrt(n)

    def apply(self, state: np.ndarray) -> np.ndarray:
        """Apply â to state vector."""
        return self.matrix @ state


class NumberOperator(BaseClass):
    """
    Number operator n̂ = â†â.

    n̂|n⟩ = n|n⟩

    Args:
        n_max: Maximum occupation number
    """

    def __init__(self, n_max: int):
        super().__init__()

        self.n_max = n_max
        self.dim = n_max + 1

        # n̂ = â†â is diagonal with eigenvalues 0, 1, 2, ...
        self.matrix = np.diag(np.arange(n_max + 1, dtype=float))

    def apply(self, state: np.ndarray) -> np.ndarray:
        """Apply n̂ to state vector."""
        return self.matrix @ state

    def expectation_value(self, state: np.ndarray) -> float:
        """Calculate ⟨n̂⟩."""
        return np.real(np.conj(state) @ self.matrix @ state)


# ==============================================================================
# Phase 5.2: Canonical Quantum Systems
# ==============================================================================

class FiniteWell(BaseClass):
    """
    Finite square well potential.

    V(x) = -V₀ for |x| < a, 0 otherwise

    Args:
        V0: Well depth (positive value)
        width: Well half-width a
        mass: Particle mass
        hbar: Reduced Planck constant
        n_grid: Number of grid points
    """

    def __init__(
        self,
        V0: float,
        width: float,
        mass: float = M_E,
        hbar: float = HBAR,
        n_grid: int = 500
    ):
        super().__init__()

        validate_positive(V0, "V0")
        validate_positive(width, "width")

        self.V0 = V0
        self.a = width
        self.mass = mass
        self.hbar = hbar

        # Dimensionless parameter
        self.z0 = width * np.sqrt(2 * mass * V0) / hbar

        # Grid for numerical solution
        self.grid = np.linspace(-3 * width, 3 * width, n_grid)
        self.V = np.where(np.abs(self.grid) < width, -V0, 0.0)

        self.hamiltonian = HamiltonianOperator(self.grid, mass, self.V, hbar)

    def bound_state_count(self) -> int:
        """Estimate number of bound states."""
        return max(1, int(np.ceil(self.z0 / (np.pi / 2))))

    def energy_levels(self, n_states: int = 5) -> np.ndarray:
        """Calculate bound state energies."""
        eigenvals = self.hamiltonian.eigenvalues(n_states + 5)
        # Keep only bound states (E < 0)
        bound = eigenvals[eigenvals < 0]
        return bound[:n_states]

    def eigenstates(self, n_states: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Return bound state wavefunctions."""
        energies, states = self.hamiltonian.eigenstates(n_states + 5)
        mask = energies < 0
        return energies[mask][:n_states], states[:, mask][:, :n_states]


class DoubleWell(BaseClass):
    """
    Double well potential for tunneling studies.

    V(x) = a(x² - b²)²  (quartic double well)

    Args:
        barrier_height: Height of central barrier
        well_separation: Distance between well minima
        mass: Particle mass
        hbar: Reduced Planck constant
    """

    def __init__(
        self,
        barrier_height: float,
        well_separation: float,
        mass: float = M_E,
        hbar: float = HBAR,
        n_grid: int = 500
    ):
        super().__init__()

        self.V0 = barrier_height
        self.b = well_separation / 2
        self.mass = mass
        self.hbar = hbar

        # Coefficient a from V(0) = V0 = a*b^4
        self.a = barrier_height / self.b**4

        # Grid
        self.grid = np.linspace(-2 * well_separation, 2 * well_separation, n_grid)
        self.V = self.a * (self.grid**2 - self.b**2)**2

        self.hamiltonian = HamiltonianOperator(self.grid, mass, self.V, hbar)

    def potential(self, x: float) -> float:
        """Evaluate potential at x."""
        return self.a * (x**2 - self.b**2)**2

    def energy_levels(self, n_states: int = 10) -> np.ndarray:
        """Calculate energy eigenvalues."""
        return self.hamiltonian.eigenvalues(n_states)

    def tunneling_splitting(self) -> float:
        """Calculate ground state tunneling splitting."""
        energies = self.energy_levels(2)
        return energies[1] - energies[0]

    def eigenstates(self, n_states: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """Return energy eigenstates."""
        return self.hamiltonian.eigenstates(n_states)


class DeltaPotential(BaseClass):
    """
    Dirac delta potential V(x) = -αδ(x).

    Has exactly one bound state with E = -mα²/(2ℏ²).

    Args:
        strength: Coupling strength α
        mass: Particle mass
        hbar: Reduced Planck constant
    """

    def __init__(
        self,
        strength: float,
        mass: float = M_E,
        hbar: float = HBAR,
        n_grid: int = 500
    ):
        super().__init__()

        self.alpha = strength
        self.mass = mass
        self.hbar = hbar

        # Analytical bound state energy
        self.E_bound = -mass * strength**2 / (2 * hbar**2)

    def bound_state_energy(self) -> float:
        """Return exact bound state energy."""
        return self.E_bound

    def bound_state_wavefunction(self, x: ArrayLike) -> np.ndarray:
        """
        Return normalized bound state wavefunction.

        ψ(x) = √(κ) exp(-κ|x|) where κ = mα/ℏ²
        """
        x = np.array(x)
        kappa = self.mass * self.alpha / self.hbar**2
        return np.sqrt(kappa) * np.exp(-kappa * np.abs(x))

    def transmission_coefficient(self, E: float) -> float:
        """
        Calculate transmission coefficient for scattering (E > 0).

        T = 1 / (1 + mα²/(2ℏ²E))
        """
        if E <= 0:
            return 0.0
        return 1 / (1 + self.mass * self.alpha**2 / (2 * self.hbar**2 * E))


class StepPotential(BaseClass):
    """
    Step potential for scattering.

    V(x) = 0 for x < 0, V₀ for x ≥ 0

    Args:
        V0: Step height
        mass: Particle mass
        hbar: Reduced Planck constant
    """

    def __init__(
        self,
        V0: float,
        mass: float = M_E,
        hbar: float = HBAR
    ):
        super().__init__()

        self.V0 = V0
        self.mass = mass
        self.hbar = hbar

    def transmission_coefficient(self, E: float) -> float:
        """
        Calculate transmission coefficient.

        T = 4k₁k₂ / (k₁ + k₂)² for E > V₀
        T = 0 for E < V₀ (classically)
        """
        if E <= 0:
            return 0.0

        k1 = np.sqrt(2 * self.mass * E) / self.hbar

        if E > self.V0:
            k2 = np.sqrt(2 * self.mass * (E - self.V0)) / self.hbar
            return 4 * k1 * k2 / (k1 + k2)**2
        else:
            # Evanescent (quantum tunneling through barrier)
            kappa = np.sqrt(2 * self.mass * (self.V0 - E)) / self.hbar
            return 0.0  # Step doesn't tunnel (infinite extent)

    def reflection_coefficient(self, E: float) -> float:
        """Calculate reflection coefficient R = 1 - T."""
        return 1 - self.transmission_coefficient(E)


class BarrierTunneling(BaseClass):
    """
    Rectangular barrier tunneling.

    V(x) = V₀ for 0 < x < L, 0 otherwise

    Args:
        V0: Barrier height
        width: Barrier width L
        mass: Particle mass
        hbar: Reduced Planck constant
    """

    def __init__(
        self,
        V0: float,
        width: float,
        mass: float = M_E,
        hbar: float = HBAR
    ):
        super().__init__()

        self.V0 = V0
        self.L = width
        self.mass = mass
        self.hbar = hbar

    def transmission_coefficient(self, E: float) -> float:
        """
        Calculate tunneling transmission coefficient.

        For E < V₀ (tunneling):
        T = [1 + V₀² sinh²(κL)/(4E(V₀-E))]⁻¹

        For E > V₀ (over barrier):
        T = [1 + V₀² sin²(k₂L)/(4E(E-V₀))]⁻¹
        """
        if E <= 0:
            return 0.0

        if E < self.V0:
            # Tunneling regime
            kappa = np.sqrt(2 * self.mass * (self.V0 - E)) / self.hbar
            sinh_term = np.sinh(kappa * self.L)
            return 1 / (1 + self.V0**2 * sinh_term**2 / (4 * E * (self.V0 - E)))

        elif E == self.V0:
            # At barrier top
            k0 = np.sqrt(2 * self.mass * E) / self.hbar
            return 1 / (1 + (k0 * self.L)**2 / 4)

        else:
            # Over barrier (resonances)
            k2 = np.sqrt(2 * self.mass * (E - self.V0)) / self.hbar
            sin_term = np.sin(k2 * self.L)
            return 1 / (1 + self.V0**2 * sin_term**2 / (4 * E * (E - self.V0)))

    def wkb_transmission(self, E: float) -> float:
        """WKB approximation for tunneling: T ≈ exp(-2∫κdx)."""
        if E >= self.V0:
            return 1.0  # No tunneling needed

        kappa_avg = np.sqrt(2 * self.mass * (self.V0 - E)) / self.hbar
        return np.exp(-2 * kappa_avg * self.L)


class CoulombPotential(BaseClass):
    """
    Coulomb potential for hydrogen-like atoms.

    V(r) = -Ze²/(4πε₀r)

    Args:
        Z: Nuclear charge
        mass: Reduced mass
        hbar: Reduced Planck constant
    """

    def __init__(
        self,
        Z: int = 1,
        mass: float = M_E,
        hbar: float = HBAR
    ):
        super().__init__()

        self.Z = Z
        self.mass = mass
        self.hbar = hbar

        # Bohr radius for this system
        self.a0 = 4 * np.pi * 8.854e-12 * hbar**2 / (mass * E_CHARGE**2 * Z)

        # Rydberg energy
        self.Ry = mass * E_CHARGE**4 * Z**2 / (2 * (4 * np.pi * 8.854e-12)**2 * hbar**2)

    def energy_level(self, n: int) -> float:
        """
        Calculate energy of level n.

        E_n = -Ry/n²
        """
        return -self.Ry / n**2

    def energy_level_eV(self, n: int) -> float:
        """Energy in electron volts."""
        return self.energy_level(n) / EV_TO_J

    def radial_wavefunction(self, n: int, l: int, r: ArrayLike) -> np.ndarray:
        """
        Calculate radial wavefunction R_nl(r).

        Simplified for low n, l.
        """
        r = np.array(r)
        rho = 2 * r / (n * self.a0)

        if n == 1 and l == 0:
            # 1s
            return 2 * (1/self.a0)**1.5 * np.exp(-rho/2)

        elif n == 2 and l == 0:
            # 2s
            return (1/(2*np.sqrt(2))) * (1/self.a0)**1.5 * (2 - rho) * np.exp(-rho/2)

        elif n == 2 and l == 1:
            # 2p
            return (1/(2*np.sqrt(6))) * (1/self.a0)**1.5 * rho * np.exp(-rho/2)

        else:
            # General case would require associated Laguerre polynomials
            return np.zeros_like(r)

    def transition_frequency(self, n_upper: int, n_lower: int) -> float:
        """Calculate transition frequency."""
        dE = self.energy_level(n_lower) - self.energy_level(n_upper)
        return dE / (2 * np.pi * self.hbar)

    def transition_wavelength(self, n_upper: int, n_lower: int) -> float:
        """Calculate transition wavelength."""
        c = 3e8
        freq = self.transition_frequency(n_upper, n_lower)
        return c / freq


class HarmonicOscillator3D(BaseClass):
    """
    3D isotropic harmonic oscillator.

    V(r) = ½mω²r²

    Args:
        omega: Angular frequency
        mass: Particle mass
        hbar: Reduced Planck constant
    """

    def __init__(
        self,
        omega: float,
        mass: float = M_E,
        hbar: float = HBAR
    ):
        super().__init__()

        self.omega = omega
        self.mass = mass
        self.hbar = hbar

        self.length_scale = np.sqrt(hbar / (mass * omega))

    def energy_level(self, n: int) -> float:
        """
        Energy of level with principal quantum number n.

        E_n = ℏω(n + 3/2) where n = 2n_r + l
        """
        return self.hbar * self.omega * (n + 1.5)

    def degeneracy(self, n: int) -> int:
        """
        Calculate degeneracy of energy level n.

        g(n) = (n+1)(n+2)/2
        """
        return (n + 1) * (n + 2) // 2

    def eigenstate_1s(self, r: ArrayLike) -> np.ndarray:
        """Ground state (1s) radial wavefunction."""
        r = np.array(r)
        alpha = self.mass * self.omega / self.hbar
        return (alpha / np.pi)**0.75 * np.exp(-alpha * r**2 / 2)


class MorsePotential(BaseClass):
    """
    Morse potential for anharmonic molecular vibrations.

    V(r) = D_e[1 - exp(-a(r-r_e))]²

    Args:
        De: Well depth
        a: Width parameter
        re: Equilibrium distance
        mass: Reduced mass
        hbar: Reduced Planck constant
    """

    def __init__(
        self,
        De: float,
        a: float,
        re: float,
        mass: float,
        hbar: float = HBAR
    ):
        super().__init__()

        self.De = De
        self.a = a
        self.re = re
        self.mass = mass
        self.hbar = hbar

        # Harmonic frequency at equilibrium
        self.omega_e = a * np.sqrt(2 * De / mass)

        # Anharmonicity parameter
        self.chi_e = hbar * self.omega_e / (4 * De)

        # Maximum vibrational quantum number
        self.n_max = int(np.floor(np.sqrt(2 * mass * De) / (a * hbar) - 0.5))

    def potential(self, r: float) -> float:
        """Evaluate Morse potential."""
        return self.De * (1 - np.exp(-self.a * (r - self.re)))**2

    def energy_level(self, v: int) -> float:
        """
        Vibrational energy level.

        E_v = ℏω_e(v + 1/2) - ℏω_e χ_e(v + 1/2)²
        """
        if v > self.n_max:
            return self.De  # Above dissociation

        return (self.hbar * self.omega_e * (v + 0.5) -
                self.hbar * self.omega_e * self.chi_e * (v + 0.5)**2)

    def dissociation_energy(self) -> float:
        """Return dissociation energy D_e."""
        return self.De

    def number_of_bound_states(self) -> int:
        """Return number of bound vibrational states."""
        return self.n_max + 1


# ==============================================================================
# Phase 5.3: Angular Momentum
# ==============================================================================

class OrbitalAngularMomentum(BaseClass):
    """
    Orbital angular momentum L², Lz eigenstates.

    |l, m⟩ with L²|l,m⟩ = ℏ²l(l+1)|l,m⟩, Lz|l,m⟩ = ℏm|l,m⟩

    Args:
        l: Angular momentum quantum number
        hbar: Reduced Planck constant
    """

    def __init__(self, l: int, hbar: float = HBAR):
        super().__init__()

        self.l = l
        self.hbar = hbar
        self.dim = 2 * l + 1  # Dimension of subspace

    def L2_eigenvalue(self) -> float:
        """Return L² eigenvalue."""
        return self.hbar**2 * self.l * (self.l + 1)

    def Lz_eigenvalue(self, m: int) -> float:
        """Return Lz eigenvalue for given m."""
        if abs(m) > self.l:
            raise ValidationError(f"|m|={abs(m)} cannot exceed l={self.l}")
        return self.hbar * m

    def basis_state(self, m: int) -> np.ndarray:
        """Return |l, m⟩ as column vector in Lz basis."""
        if abs(m) > self.l:
            raise ValidationError(f"|m|={abs(m)} cannot exceed l={self.l}")

        state = np.zeros(self.dim, dtype=complex)
        idx = self.l - m  # m = l is index 0, m = -l is index 2l
        state[idx] = 1.0
        return state


class SpinAngularMomentum(BaseClass):
    """
    Spin angular momentum for spin-s systems.

    Args:
        s: Spin quantum number (half-integer allowed)
        hbar: Reduced Planck constant
    """

    def __init__(self, s: float, hbar: float = HBAR):
        super().__init__()

        # Check that s is non-negative half-integer
        if s < 0 or (2 * s) % 1 != 0:
            raise ValidationError("Spin must be non-negative half-integer")

        self.s = s
        self.hbar = hbar
        self.dim = int(2 * s + 1)

    def Sz_matrix(self) -> np.ndarray:
        """Return Sz matrix."""
        m_vals = np.arange(self.s, -self.s - 1, -1)
        return self.hbar * np.diag(m_vals)

    def Sp_matrix(self) -> np.ndarray:
        """Return S+ (raising) matrix."""
        Sp = np.zeros((self.dim, self.dim), dtype=complex)
        m_vals = np.arange(self.s, -self.s - 1, -1)

        for i in range(self.dim - 1):
            m = m_vals[i + 1]
            Sp[i, i + 1] = self.hbar * np.sqrt(self.s * (self.s + 1) - m * (m + 1))

        return Sp

    def Sm_matrix(self) -> np.ndarray:
        """Return S- (lowering) matrix."""
        return self.Sp_matrix().T.conj()

    def Sx_matrix(self) -> np.ndarray:
        """Return Sx matrix."""
        return (self.Sp_matrix() + self.Sm_matrix()) / 2

    def Sy_matrix(self) -> np.ndarray:
        """Return Sy matrix."""
        return (self.Sp_matrix() - self.Sm_matrix()) / (2j)

    def S2_matrix(self) -> np.ndarray:
        """Return S² matrix (diagonal with eigenvalue s(s+1)ℏ²)."""
        return self.hbar**2 * self.s * (self.s + 1) * np.eye(self.dim)

    def spin_up(self) -> np.ndarray:
        """Return |↑⟩ state (m_s = +s)."""
        state = np.zeros(self.dim, dtype=complex)
        state[0] = 1.0
        return state

    def spin_down(self) -> np.ndarray:
        """Return |↓⟩ state (m_s = -s)."""
        state = np.zeros(self.dim, dtype=complex)
        state[-1] = 1.0
        return state


class SpinOrbitCoupling(BaseClass):
    """
    Spin-orbit coupling L̂·Ŝ interaction.

    H_SO = λ L̂·Ŝ where j = l ± s

    Args:
        l: Orbital angular momentum
        s: Spin (typically 0.5 for electrons)
        coupling: Spin-orbit coupling strength λ
        hbar: Reduced Planck constant
    """

    def __init__(
        self,
        l: int,
        s: float = 0.5,
        coupling: float = 1.0,
        hbar: float = HBAR
    ):
        super().__init__()

        self.l = l
        self.s = s
        self.lam = coupling
        self.hbar = hbar

        # Possible j values
        self.j_values = [abs(l - s), l + s] if l > 0 else [s]

    def energy_shift(self, j: float) -> float:
        """
        Calculate spin-orbit energy shift.

        E_SO = (λ/2)[j(j+1) - l(l+1) - s(s+1)] ℏ²
        """
        return (self.lam * self.hbar**2 / 2) * (
            j * (j + 1) - self.l * (self.l + 1) - self.s * (self.s + 1)
        )

    def fine_structure_splitting(self) -> float:
        """Calculate fine structure splitting between j = l+s and j = l-s."""
        if len(self.j_values) < 2:
            return 0.0

        E_high = self.energy_shift(self.j_values[1])
        E_low = self.energy_shift(self.j_values[0])
        return E_high - E_low


class ClebschGordan(BaseClass):
    """
    Clebsch-Gordan coefficients for angular momentum addition.

    |j, m⟩ = Σ_{m1, m2} C^{j,m}_{j1,m1;j2,m2} |j1, m1⟩|j2, m2⟩

    Args:
        j1: First angular momentum
        j2: Second angular momentum
    """

    def __init__(self, j1: float, j2: float):
        super().__init__()

        self.j1 = j1
        self.j2 = j2

        # Valid j values
        self.j_min = abs(j1 - j2)
        self.j_max = j1 + j2

    def coefficient(self, j: float, m: float, m1: float, m2: float) -> float:
        """
        Calculate Clebsch-Gordan coefficient C^{j,m}_{j1,m1;j2,m2}.

        Uses numerical formula (simplified).
        """
        # Check selection rules
        if m != m1 + m2:
            return 0.0
        if j < self.j_min or j > self.j_max:
            return 0.0
        if abs(m) > j or abs(m1) > self.j1 or abs(m2) > self.j2:
            return 0.0

        # For simple cases
        if self.j1 == 0.5 and self.j2 == 0.5:
            return self._spin_half_cg(j, m, m1, m2)

        # General case would use Racah formula
        # Placeholder for more complex cases
        return 0.0

    def _spin_half_cg(self, j: float, m: float, m1: float, m2: float) -> float:
        """CG coefficients for two spin-1/2 particles."""
        if j == 1:  # Triplet
            if m == 1:
                return 1.0 if m1 == 0.5 and m2 == 0.5 else 0.0
            elif m == 0:
                if (m1 == 0.5 and m2 == -0.5) or (m1 == -0.5 and m2 == 0.5):
                    return 1/np.sqrt(2)
                return 0.0
            elif m == -1:
                return 1.0 if m1 == -0.5 and m2 == -0.5 else 0.0
        elif j == 0:  # Singlet
            if m == 0:
                if m1 == 0.5 and m2 == -0.5:
                    return 1/np.sqrt(2)
                elif m1 == -0.5 and m2 == 0.5:
                    return -1/np.sqrt(2)
            return 0.0
        return 0.0


class WignerDMatrix(BaseClass):
    """
    Wigner D-matrix for rotations of angular momentum states.

    D^j_{m'm}(α, β, γ) = ⟨j,m'|R(α,β,γ)|j,m⟩

    Args:
        j: Angular momentum quantum number
    """

    def __init__(self, j: float):
        super().__init__()
        self.j = j
        self.dim = int(2 * j + 1)

    def small_d(self, m_prime: float, m: float, beta: float) -> float:
        """
        Calculate small Wigner d-matrix element d^j_{m'm}(β).

        Uses recursion or formula for simple cases.
        """
        j = self.j

        # Check validity
        if abs(m_prime) > j or abs(m) > j:
            return 0.0

        # Special cases for j = 1/2
        if j == 0.5:
            c = np.cos(beta / 2)
            s = np.sin(beta / 2)
            if m_prime == 0.5 and m == 0.5:
                return c
            elif m_prime == 0.5 and m == -0.5:
                return -s
            elif m_prime == -0.5 and m == 0.5:
                return s
            elif m_prime == -0.5 and m == -0.5:
                return c

        # General case would use full formula
        return 0.0

    def D_matrix(self, alpha: float, beta: float, gamma: float) -> np.ndarray:
        """
        Calculate full Wigner D-matrix.

        D^j_{m'm}(α,β,γ) = exp(-i m' α) d^j_{m'm}(β) exp(-i m γ)
        """
        D = np.zeros((self.dim, self.dim), dtype=complex)
        m_vals = np.arange(self.j, -self.j - 1, -1)

        for i, m_prime in enumerate(m_vals):
            for k, m in enumerate(m_vals):
                d = self.small_d(m_prime, m, beta)
                D[i, k] = np.exp(-1j * m_prime * alpha) * d * np.exp(-1j * m * gamma)

        return D


class SphericalHarmonicsQM(BaseClass):
    """
    Spherical harmonics Y_l^m(θ, φ) as quantum eigenfunctions.

    Angular part of hydrogen-like wavefunctions.

    Args:
        l_max: Maximum l value to compute
    """

    def __init__(self, l_max: int = 5):
        super().__init__()
        self.l_max = l_max

    def Y(self, l: int, m: int, theta: float, phi: float) -> complex:
        """
        Calculate Y_l^m(θ, φ).

        Uses associated Legendre polynomials.
        """
        if abs(m) > l:
            return 0.0 + 0.0j

        # Normalization
        from math import factorial
        norm = np.sqrt((2*l + 1) * factorial(l - abs(m)) /
                       (4 * np.pi * factorial(l + abs(m))))

        # Associated Legendre polynomial
        P_lm = self._associated_legendre(l, abs(m), np.cos(theta))

        # Spherical harmonic
        if m >= 0:
            return norm * P_lm * np.exp(1j * m * phi)
        else:
            return ((-1)**m) * norm * P_lm * np.exp(1j * m * phi)

    def _associated_legendre(self, l: int, m: int, x: float) -> float:
        """Compute associated Legendre polynomial P_l^m(x)."""
        if m > l:
            return 0.0

        # Start with P_m^m
        pmm = 1.0
        if m > 0:
            somx2 = np.sqrt((1 - x) * (1 + x))
            fact = 1.0
            for i in range(1, m + 1):
                pmm *= -fact * somx2
                fact += 2.0

        if l == m:
            return pmm

        # P_{m+1}^m
        pmmp1 = x * (2 * m + 1) * pmm
        if l == m + 1:
            return pmmp1

        # Recurrence
        for ll in range(m + 2, l + 1):
            pll = (x * (2 * ll - 1) * pmmp1 - (ll + m - 1) * pmm) / (ll - m)
            pmm = pmmp1
            pmmp1 = pll

        return pmmp1


# ==============================================================================
# Phase 5.4: Multi-particle Systems
# ==============================================================================

class TwoParticleSystem(BaseClass):
    """
    Two distinguishable particles quantum system.

    Ψ(x₁, x₂) = ψ_a(x₁)ψ_b(x₂)

    Args:
        grid: Position space grid
        mass1, mass2: Particle masses
        hbar: Reduced Planck constant
    """

    def __init__(
        self,
        grid: ArrayLike,
        mass1: float,
        mass2: float,
        hbar: float = HBAR
    ):
        super().__init__()

        self.grid = np.array(grid)
        self.m1 = mass1
        self.m2 = mass2
        self.hbar = hbar

        # Reduced mass
        self.mu = mass1 * mass2 / (mass1 + mass2)

        # Total mass
        self.M = mass1 + mass2

    def product_state(self, psi1: np.ndarray, psi2: np.ndarray) -> np.ndarray:
        """Create product state Ψ(x₁,x₂) = ψ₁(x₁)⊗ψ₂(x₂)."""
        return np.outer(psi1, psi2)

    def reduced_density_matrix(
        self,
        psi_2d: np.ndarray,
        trace_over: int = 2
    ) -> np.ndarray:
        """
        Calculate reduced density matrix by partial trace.

        Args:
            psi_2d: Two-particle wavefunction on grid
            trace_over: Which particle to trace out (1 or 2)
        """
        rho = np.outer(psi_2d.flatten(), np.conj(psi_2d.flatten()))

        N = len(self.grid)
        if trace_over == 2:
            # Trace over particle 2
            rho_reduced = np.zeros((N, N), dtype=complex)
            for i in range(N):
                for j in range(N):
                    for k in range(N):
                        rho_reduced[i, j] += rho[i*N + k, j*N + k]
        else:
            # Trace over particle 1
            rho_reduced = np.zeros((N, N), dtype=complex)
            for i in range(N):
                for j in range(N):
                    for k in range(N):
                        rho_reduced[i, j] += rho[k*N + i, k*N + j]

        return rho_reduced


class IdenticalBosons(BaseClass):
    """
    System of identical bosons (symmetric wavefunctions).

    Ψ(x₁, x₂) = Ψ(x₂, x₁) (symmetric under exchange)

    Args:
        n_particles: Number of bosons
        n_levels: Number of single-particle levels
    """

    def __init__(self, n_particles: int, n_levels: int):
        super().__init__()

        self.N = n_particles
        self.n_levels = n_levels

    def symmetrize(self, psi1: np.ndarray, psi2: np.ndarray) -> np.ndarray:
        """
        Create symmetrized two-particle state.

        Ψ_S = (1/√2)[ψ_a(x₁)ψ_b(x₂) + ψ_a(x₂)ψ_b(x₁)]
        """
        return (np.outer(psi1, psi2) + np.outer(psi2, psi1)) / np.sqrt(2)

    def permanent(self, matrix: np.ndarray) -> complex:
        """
        Calculate permanent of matrix (symmetric analog of determinant).

        Used for multi-boson states.
        """
        n = matrix.shape[0]
        if n == 0:
            return 1.0

        if n == 1:
            return matrix[0, 0]

        if n == 2:
            return matrix[0, 0] * matrix[1, 1] + matrix[0, 1] * matrix[1, 0]

        # General case: sum over all permutations (expensive!)
        from itertools import permutations
        perm = 0.0
        for p in permutations(range(n)):
            prod = 1.0
            for i, j in enumerate(p):
                prod *= matrix[i, j]
            perm += prod

        return perm


class IdenticalFermions(BaseClass):
    """
    System of identical fermions (antisymmetric wavefunctions).

    Uses Slater determinant for N-particle states.

    Args:
        n_particles: Number of fermions
        n_levels: Number of single-particle levels
    """

    def __init__(self, n_particles: int, n_levels: int):
        super().__init__()

        if n_particles > n_levels:
            raise ValidationError("Cannot have more fermions than available levels")

        self.N = n_particles
        self.n_levels = n_levels

    def antisymmetrize(self, psi1: np.ndarray, psi2: np.ndarray) -> np.ndarray:
        """
        Create antisymmetrized two-particle state.

        Ψ_A = (1/√2)[ψ_a(x₁)ψ_b(x₂) - ψ_a(x₂)ψ_b(x₁)]
        """
        return (np.outer(psi1, psi2) - np.outer(psi2, psi1)) / np.sqrt(2)

    def slater_determinant(self, orbitals: List[np.ndarray], positions: np.ndarray) -> np.ndarray:
        """
        Construct Slater determinant wavefunction.

        Ψ(x₁,...,xN) = (1/√N!) det[φᵢ(xⱼ)]

        Args:
            orbitals: List of single-particle wavefunctions
            positions: Grid positions

        Returns:
            Slater determinant evaluated at each position configuration
        """
        N = len(orbitals)
        n_pos = len(positions)

        # For each position configuration, evaluate determinant
        # Simplified: return matrix of orbitals evaluated at positions
        matrix = np.zeros((N, n_pos), dtype=complex)
        for i, phi in enumerate(orbitals):
            matrix[i, :] = phi

        return matrix / np.sqrt(np.math.factorial(N))


class ExchangeInteraction(BaseClass):
    """
    Exchange interaction energy for fermions.

    J = ∫∫ ψ*_a(r₁)ψ*_b(r₂) V(r₁,r₂) ψ_a(r₂)ψ_b(r₁) dr₁ dr₂

    Args:
        grid: Position grid
        interaction: Two-body interaction V(r₁, r₂)
    """

    def __init__(
        self,
        grid: ArrayLike,
        interaction: Callable[[float, float], float]
    ):
        super().__init__()

        self.grid = np.array(grid)
        self.V = interaction
        self.dx = grid[1] - grid[0] if len(grid) > 1 else 1.0

    def exchange_integral(self, psi_a: np.ndarray, psi_b: np.ndarray) -> complex:
        """
        Calculate exchange integral J.

        J = ⟨ab|V|ba⟩
        """
        N = len(self.grid)
        J = 0.0

        for i in range(N):
            for j in range(N):
                V_ij = self.V(self.grid[i], self.grid[j])
                J += (np.conj(psi_a[i]) * np.conj(psi_b[j]) *
                      V_ij * psi_a[j] * psi_b[i])

        return J * self.dx**2

    def direct_integral(self, psi_a: np.ndarray, psi_b: np.ndarray) -> complex:
        """
        Calculate direct (Coulomb) integral K.

        K = ⟨ab|V|ab⟩
        """
        N = len(self.grid)
        K = 0.0

        for i in range(N):
            for j in range(N):
                V_ij = self.V(self.grid[i], self.grid[j])
                K += (np.conj(psi_a[i]) * np.conj(psi_b[j]) *
                      V_ij * psi_a[i] * psi_b[j])

        return K * self.dx**2


# ==============================================================================
# Phase 5.5: Approximation Methods
# ==============================================================================

class TimeIndependentPerturbation(BaseClass):
    """
    Time-independent perturbation theory (non-degenerate).

    H = H₀ + λV

    Calculates energy corrections to given order.

    Args:
        H0_energies: Unperturbed energies
        H0_states: Unperturbed eigenstates (columns)
        V_matrix: Perturbation matrix in H0 basis
    """

    def __init__(
        self,
        H0_energies: ArrayLike,
        H0_states: np.ndarray,
        V_matrix: np.ndarray
    ):
        super().__init__()

        self.E0 = np.array(H0_energies)
        self.psi0 = H0_states
        self.V = V_matrix
        self.n_states = len(H0_energies)

    def first_order_energy(self, state_index: int) -> float:
        """
        Calculate first-order energy correction.

        E^(1)_n = ⟨n|V|n⟩
        """
        return np.real(self.V[state_index, state_index])

    def second_order_energy(self, state_index: int) -> float:
        """
        Calculate second-order energy correction.

        E^(2)_n = Σ_{m≠n} |⟨m|V|n⟩|² / (E_n - E_m)
        """
        n = state_index
        E2 = 0.0

        for m in range(self.n_states):
            if m != n:
                dE = self.E0[n] - self.E0[m]
                if abs(dE) > 1e-15:
                    E2 += np.abs(self.V[m, n])**2 / dE

        return E2

    def first_order_state(self, state_index: int) -> np.ndarray:
        """
        Calculate first-order state correction.

        |n^(1)⟩ = Σ_{m≠n} (⟨m|V|n⟩/(E_n - E_m)) |m⟩
        """
        n = state_index
        correction = np.zeros(self.n_states, dtype=complex)

        for m in range(self.n_states):
            if m != n:
                dE = self.E0[n] - self.E0[m]
                if abs(dE) > 1e-15:
                    correction += self.V[m, n] / dE * self.psi0[:, m]

        return correction

    def corrected_energy(self, state_index: int, order: int = 2) -> float:
        """Calculate energy up to given order."""
        E = self.E0[state_index]
        E += self.first_order_energy(state_index)
        if order >= 2:
            E += self.second_order_energy(state_index)
        return E


class VariationalMethod(BaseClass):
    """
    Variational method for ground state energy.

    E_trial = ⟨ψ_trial|H|ψ_trial⟩ / ⟨ψ_trial|ψ_trial⟩ ≥ E_0

    Args:
        hamiltonian: Hamiltonian operator
        trial_function: Parameterized trial wavefunction ψ(x; α)
    """

    def __init__(
        self,
        hamiltonian: HamiltonianOperator,
        trial_function: Optional[Callable] = None
    ):
        super().__init__()

        self.H = hamiltonian

    def energy_expectation(self, psi: np.ndarray) -> float:
        """Calculate ⟨H⟩ for trial state."""
        norm = np.sum(np.abs(psi)**2) * self.H.dx
        Hpsi = self.H.apply(psi)
        return np.real(np.sum(np.conj(psi) * Hpsi) * self.H.dx / norm)

    def minimize(
        self,
        trial_params: ArrayLike,
        param_to_psi: Callable[[np.ndarray], np.ndarray],
        method: str = 'powell'
    ) -> Tuple[np.ndarray, float]:
        """
        Minimize energy over trial parameters.

        Args:
            trial_params: Initial parameter guess
            param_to_psi: Function mapping parameters to wavefunction
            method: Optimization method

        Returns:
            (optimal_params, minimum_energy)
        """
        from scipy.optimize import minimize as scipy_minimize

        def objective(params):
            psi = param_to_psi(params)
            return self.energy_expectation(psi)

        result = scipy_minimize(objective, trial_params, method=method)
        return result.x, result.fun


class WKBApproximation(BaseClass):
    """
    WKB (semiclassical) approximation.

    ψ(x) ~ (1/√p(x)) exp(±(i/ℏ)∫p(x')dx')

    where p(x) = √(2m(E-V(x)))

    Args:
        mass: Particle mass
        potential: Potential V(x)
        hbar: Reduced Planck constant
    """

    def __init__(
        self,
        mass: float,
        potential: Callable[[float], float],
        hbar: float = HBAR
    ):
        super().__init__()

        self.mass = mass
        self.V = potential
        self.hbar = hbar

    def classical_momentum(self, x: float, E: float) -> float:
        """Calculate classical momentum p(x) = √(2m(E-V))."""
        KE = E - self.V(x)
        if KE > 0:
            return np.sqrt(2 * self.mass * KE)
        else:
            return 1j * np.sqrt(2 * self.mass * abs(KE))

    def phase_integral(
        self,
        x1: float,
        x2: float,
        E: float,
        n_points: int = 1000
    ) -> complex:
        """Calculate ∫_{x1}^{x2} p(x) dx."""
        x = np.linspace(x1, x2, n_points)
        p = np.array([self.classical_momentum(xi, E) for xi in x])
        return np.trapz(p, x)

    def quantization_condition(self, E: float, x1: float, x2: float) -> float:
        """
        Check Bohr-Sommerfeld quantization.

        ∮ p dx = (n + 1/2) h
        """
        phase = self.phase_integral(x1, x2, E)
        return np.real(phase) / (np.pi * self.hbar) - 0.5

    def tunneling_probability(
        self,
        E: float,
        x1: float,
        x2: float
    ) -> float:
        """
        Calculate WKB tunneling probability.

        T ~ exp(-2∫κdx) where κ = |p|/ℏ for classically forbidden region
        """
        phase = self.phase_integral(x1, x2, E)
        return np.exp(-2 * np.abs(np.imag(phase)) / self.hbar)


# ==============================================================================
# Phase 5.6: Open Quantum Systems
# ==============================================================================

class DensityMatrix(BaseClass):
    """
    Density matrix representation for mixed states.

    ρ = Σᵢ pᵢ |ψᵢ⟩⟨ψᵢ|

    Args:
        rho: Density matrix or pure state vector
    """

    def __init__(self, rho: Union[np.ndarray, None] = None, dim: int = 2):
        super().__init__()

        if rho is None:
            self.rho = np.zeros((dim, dim), dtype=complex)
        elif rho.ndim == 1:
            # Pure state: ρ = |ψ⟩⟨ψ|
            self.rho = np.outer(rho, np.conj(rho))
        else:
            self.rho = np.array(rho, dtype=complex)

        self.dim = self.rho.shape[0]

    @classmethod
    def pure_state(cls, psi: np.ndarray) -> 'DensityMatrix':
        """Create density matrix from pure state."""
        return cls(psi)

    @classmethod
    def mixed_state(cls, states: List[np.ndarray], probabilities: List[float]) -> 'DensityMatrix':
        """Create density matrix from statistical mixture."""
        dim = len(states[0])
        rho = np.zeros((dim, dim), dtype=complex)

        for psi, p in zip(states, probabilities):
            rho += p * np.outer(psi, np.conj(psi))

        return cls(rho)

    @classmethod
    def thermal_state(cls, H: np.ndarray, T: float, kB: float = 1.381e-23) -> 'DensityMatrix':
        """Create thermal equilibrium density matrix."""
        beta = 1 / (kB * T)
        Z = np.trace(la.expm(-beta * H))
        rho = la.expm(-beta * H) / Z
        return cls(rho)

    def trace(self) -> complex:
        """Calculate trace (should be 1)."""
        return np.trace(self.rho)

    def purity(self) -> float:
        """Calculate purity Tr(ρ²). Pure state has purity 1."""
        return np.real(np.trace(self.rho @ self.rho))

    def von_neumann_entropy(self) -> float:
        """Calculate von Neumann entropy S = -Tr(ρ ln ρ)."""
        eigenvals = np.linalg.eigvalsh(self.rho)
        eigenvals = eigenvals[eigenvals > 1e-15]  # Remove zeros
        return -np.sum(eigenvals * np.log(eigenvals))

    def expectation_value(self, operator: np.ndarray) -> complex:
        """Calculate ⟨A⟩ = Tr(ρA)."""
        return np.trace(self.rho @ operator)

    def partial_trace(self, dim_A: int, dim_B: int, trace_over: str = 'B') -> 'DensityMatrix':
        """
        Partial trace over subsystem.

        Args:
            dim_A, dim_B: Dimensions of subsystems A and B
            trace_over: 'A' or 'B'
        """
        rho_full = self.rho.reshape((dim_A, dim_B, dim_A, dim_B))

        if trace_over == 'B':
            rho_A = np.trace(rho_full, axis1=1, axis2=3)
            return DensityMatrix(rho_A)
        else:
            rho_B = np.trace(rho_full, axis1=0, axis2=2)
            return DensityMatrix(rho_B)


class VonNeumannEquation(BaseClass):
    """
    Von Neumann equation for unitary evolution of density matrix.

    dρ/dt = -(i/ℏ)[H, ρ]

    Args:
        hamiltonian: System Hamiltonian
        hbar: Reduced Planck constant
    """

    def __init__(self, hamiltonian: np.ndarray, hbar: float = HBAR):
        super().__init__()

        self.H = hamiltonian
        self.hbar = hbar
        self.dim = hamiltonian.shape[0]

    def commutator(self, rho: np.ndarray) -> np.ndarray:
        """Calculate [H, ρ]."""
        return self.H @ rho - rho @ self.H

    def drho_dt(self, rho: np.ndarray) -> np.ndarray:
        """Calculate time derivative of density matrix."""
        return -1j * self.commutator(rho) / self.hbar

    def evolve(self, rho0: np.ndarray, t: float) -> np.ndarray:
        """
        Evolve density matrix to time t.

        ρ(t) = U(t) ρ(0) U†(t)
        """
        U = la.expm(-1j * self.H * t / self.hbar)
        return U @ rho0 @ U.T.conj()

    def expectation_evolution(
        self,
        rho0: np.ndarray,
        operator: np.ndarray,
        times: ArrayLike
    ) -> np.ndarray:
        """Calculate ⟨A⟩(t) for array of times."""
        expectations = []
        for t in times:
            rho_t = self.evolve(rho0, t)
            expectations.append(np.trace(rho_t @ operator))
        return np.array(expectations)


class LindbladMasterEquation(BaseClass):
    """
    Lindblad master equation for open quantum systems.

    dρ/dt = -(i/ℏ)[H, ρ] + Σₖ γₖ(Lₖ ρ Lₖ† - ½{Lₖ†Lₖ, ρ})

    Args:
        hamiltonian: System Hamiltonian
        lindblad_operators: List of Lindblad (jump) operators Lₖ
        rates: Corresponding decay rates γₖ
        hbar: Reduced Planck constant
    """

    def __init__(
        self,
        hamiltonian: np.ndarray,
        lindblad_operators: List[np.ndarray],
        rates: List[float],
        hbar: float = HBAR
    ):
        super().__init__()

        self.H = hamiltonian
        self.L_ops = lindblad_operators
        self.gamma = rates
        self.hbar = hbar
        self.dim = hamiltonian.shape[0]

    def dissipator(self, rho: np.ndarray) -> np.ndarray:
        """
        Calculate dissipator term.

        D[L]ρ = L ρ L† - ½{L†L, ρ}
        """
        D = np.zeros_like(rho)

        for L, gamma in zip(self.L_ops, self.gamma):
            LdL = L.T.conj() @ L
            D += gamma * (L @ rho @ L.T.conj() -
                          0.5 * (LdL @ rho + rho @ LdL))

        return D

    def drho_dt(self, rho: np.ndarray) -> np.ndarray:
        """Calculate full time derivative."""
        # Coherent (Hamiltonian) evolution
        coherent = -1j * (self.H @ rho - rho @ self.H) / self.hbar

        # Dissipative evolution
        dissipative = self.dissipator(rho)

        return coherent + dissipative

    def evolve(
        self,
        rho0: np.ndarray,
        t_final: float,
        dt: float = 0.01
    ) -> np.ndarray:
        """
        Evolve density matrix using simple Euler integration.

        For better accuracy, use RK4 or matrix exponential methods.
        """
        rho = rho0.copy()
        t = 0

        while t < t_final:
            drho = self.drho_dt(rho)
            rho = rho + drho * dt
            t += dt

        return rho

    def steady_state(self, method: str = 'eigenvalue') -> np.ndarray:
        """
        Find steady state solution where dρ/dt = 0.

        Uses vectorized Liouvillian method.
        """
        # Vectorize the Liouvillian superoperator
        # L|ρ⟩⟩ = |dρ/dt⟩⟩

        dim2 = self.dim**2

        # Build superoperator matrix
        L_super = np.zeros((dim2, dim2), dtype=complex)

        # Hamiltonian part: -i/ℏ (H⊗I - I⊗H^T)
        I = np.eye(self.dim)
        L_super += -1j / self.hbar * (np.kron(self.H, I) - np.kron(I, self.H.T))

        # Dissipator part
        for L, gamma in zip(self.L_ops, self.gamma):
            Ldag = L.T.conj()
            LdL = Ldag @ L

            L_super += gamma * (
                np.kron(L, L.conj()) -
                0.5 * np.kron(LdL, I) -
                0.5 * np.kron(I, LdL.T)
            )

        # Find null space (eigenvalue 0)
        eigenvals, eigenvecs = np.linalg.eig(L_super)

        # Find index closest to zero
        idx = np.argmin(np.abs(eigenvals))
        rho_vec = eigenvecs[:, idx]

        # Reshape to density matrix
        rho = rho_vec.reshape((self.dim, self.dim))

        # Normalize
        rho = rho / np.trace(rho)

        return rho


# ==============================================================================
# Phase 5.4 Additions: Multi-particle Systems
# ==============================================================================

class Helium(BaseClass):
    """
    Two-electron helium atom.

    Uses variational and perturbation methods for the ground state.

    H = -ℏ²/2m (∇₁² + ∇₂²) - Ze²/r₁ - Ze²/r₂ + e²/r₁₂

    Args:
        Z: Nuclear charge (2 for He, can vary for He-like ions)

    Examples:
        >>> he = Helium(Z=2)
        >>> E = he.ground_state_energy_variational()
    """

    def __init__(self, Z: float = 2.0):
        super().__init__()
        self.Z = Z
        self.E_h = 27.211  # Hartree in eV

    def ground_state_energy_zeroth(self) -> float:
        """E⁽⁰⁾ = -2Z² (in Hartree)."""
        return -2 * self.Z**2 * self.E_h / 2

    def first_order_correction(self) -> float:
        """E⁽¹⁾ = 5Z/8 (in Hartree)."""
        return 5 * self.Z / 8 * self.E_h / 2

    def ground_state_energy_perturbation(self) -> float:
        """E = E⁽⁰⁾ + E⁽¹⁾."""
        return self.ground_state_energy_zeroth() + self.first_order_correction()

    def ground_state_energy_variational(self, Z_eff: Optional[float] = None) -> float:
        """Variational energy with Z_eff = Z - 5/16."""
        if Z_eff is None:
            Z_eff = self.Z - 5/16
        return (Z_eff**2 - 2 * self.Z * Z_eff + 5 * Z_eff / 8) * self.E_h

    def optimal_Z_eff(self) -> float:
        """Optimal effective charge."""
        return self.Z - 5/16

    def ionization_energy(self) -> float:
        """First ionization energy."""
        E_He = self.ground_state_energy_variational()
        E_He_plus = -self.Z**2 * self.E_h / 2
        return E_He_plus - E_He


class SecondQuantization(BaseClass):
    """
    Second quantization (Fock space) formalism.

    Args:
        n_modes: Number of single-particle modes
        statistics: 'boson' or 'fermion'

    Examples:
        >>> sq = SecondQuantization(n_modes=4, statistics='fermion')
        >>> state = sq.create(0, sq.vacuum())
    """

    def __init__(self, n_modes: int, statistics: str = 'fermion'):
        super().__init__()
        self.n_modes = n_modes
        self.statistics = statistics
        self.dim = 2**n_modes if statistics == 'fermion' else None
        self._build_basis()

    def _build_basis(self):
        """Build Fock space basis."""
        if self.statistics == 'fermion':
            self.dim = 2**self.n_modes
            self.basis = [tuple((i >> j) & 1 for j in range(self.n_modes))
                         for i in range(self.dim)]
            self.basis_to_idx = {s: i for i, s in enumerate(self.basis)}

    def vacuum(self) -> np.ndarray:
        """Return vacuum state."""
        state = np.zeros(self.dim, dtype=complex)
        state[0] = 1.0
        return state

    def creation_operator(self, mode: int) -> np.ndarray:
        """Build creation operator a†_mode."""
        a_dag = np.zeros((self.dim, self.dim), dtype=complex)
        for i, state in enumerate(self.basis):
            if state[mode] == 0:
                new_state = list(state)
                new_state[mode] = 1
                j = self.basis_to_idx[tuple(new_state)]
                sign = (-1)**sum(state[:mode])
                a_dag[j, i] = sign
        return a_dag

    def annihilation_operator(self, mode: int) -> np.ndarray:
        """Build annihilation operator."""
        return self.creation_operator(mode).T.conj()

    def number_operator(self, mode: int) -> np.ndarray:
        """Build number operator n_mode."""
        a_dag = self.creation_operator(mode)
        a = self.annihilation_operator(mode)
        return a_dag @ a

    def create(self, mode: int, state: np.ndarray) -> np.ndarray:
        """Apply creation operator."""
        return self.creation_operator(mode) @ state

    def annihilate(self, mode: int, state: np.ndarray) -> np.ndarray:
        """Apply annihilation operator."""
        return self.annihilation_operator(mode) @ state


# ==============================================================================
# Phase 5.5 Additions: Approximation Methods
# ==============================================================================

class DegeneratePerturbation(BaseClass):
    """
    Degenerate perturbation theory.

    Args:
        H0: Unperturbed Hamiltonian
        V: Perturbation

    Examples:
        >>> dp = DegeneratePerturbation(H0, V)
        >>> E, states = dp.first_order_correction(E0=1.0)
    """

    def __init__(self, H0: np.ndarray, V: np.ndarray, threshold: float = 1e-10):
        super().__init__()
        self.H0 = np.array(H0, dtype=complex)
        self.V = np.array(V, dtype=complex)
        self.threshold = threshold
        self.E0, self.psi0 = np.linalg.eigh(H0)

    def find_degenerate_subspace(self, E_target: float) -> Tuple[np.ndarray, np.ndarray]:
        """Find degenerate subspace for given energy."""
        indices = np.where(np.abs(self.E0 - E_target) < self.threshold)[0]
        return indices, self.psi0[:, indices]

    def first_order_correction(self, E0: float) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate first-order corrections."""
        indices, states = self.find_degenerate_subspace(E0)
        if len(indices) == 1:
            E1 = np.real(states[:, 0].conj() @ self.V @ states[:, 0])
            return np.array([E1]), states
        V_sub = states.T.conj() @ self.V @ states
        E1, coeffs = np.linalg.eigh(V_sub)
        return E1, states @ coeffs


class TimeDependentPerturbation(BaseClass):
    """
    Time-dependent perturbation theory.

    Args:
        H0: Unperturbed Hamiltonian
        hbar: Reduced Planck constant

    Examples:
        >>> tdp = TimeDependentPerturbation(H0)
        >>> prob = tdp.transition_probability(psi_i, psi_f, V, t)
    """

    def __init__(self, H0: np.ndarray, hbar: float = HBAR):
        super().__init__()
        self.H0 = np.array(H0, dtype=complex)
        self.hbar = hbar
        self.E0, self.psi0 = np.linalg.eigh(H0)

    def transition_amplitude(self, psi_i: np.ndarray, psi_f: np.ndarray,
                            V: np.ndarray, t: float) -> complex:
        """First-order transition amplitude."""
        V_fi = np.vdot(psi_f, V @ psi_i)
        E_i = np.real(np.vdot(psi_i, self.H0 @ psi_i))
        E_f = np.real(np.vdot(psi_f, self.H0 @ psi_f))
        omega_fi = (E_f - E_i) / self.hbar
        if np.abs(omega_fi) < 1e-15:
            return -1j * V_fi * t / self.hbar
        return -1j * V_fi * (np.exp(1j * omega_fi * t) - 1) / (self.hbar * omega_fi)

    def transition_probability(self, psi_i: np.ndarray, psi_f: np.ndarray,
                              V: np.ndarray, t: float) -> float:
        """Calculate transition probability."""
        return np.abs(self.transition_amplitude(psi_i, psi_f, V, t))**2

    def fermi_golden_rule(self, psi_i: np.ndarray, V: np.ndarray,
                         density_of_states: Callable[[float], float]) -> float:
        """Fermi's golden rule rate."""
        Gamma = 0.0
        for n in range(len(self.E0)):
            psi_f = self.psi0[:, n]
            V_fi = np.vdot(psi_f, V @ psi_i)
            Gamma += 2 * np.pi / self.hbar * np.abs(V_fi)**2 * density_of_states(self.E0[n])
        return Gamma


class BornApproximation(BaseClass):
    """
    Born approximation for scattering.

    Args:
        mass: Particle mass
        hbar: Reduced Planck constant
    """

    def __init__(self, mass: float = M_E, hbar: float = HBAR):
        super().__init__()
        self.mass = mass
        self.hbar = hbar

    def momentum_transfer(self, k: float, theta: float) -> float:
        """Calculate momentum transfer q = 2k sin(θ/2)."""
        return 2 * k * np.sin(theta / 2)

    def scattering_amplitude_spherical(self, V: Callable[[float], float],
                                       k: float, theta: float, r_max: float = 100.0) -> complex:
        """Calculate scattering amplitude for spherical potential."""
        q = self.momentum_transfer(k, theta)
        if q < 1e-15:
            r = np.linspace(1e-10, r_max, 1000)
            integrand = r**2 * np.array([V(ri) for ri in r])
            return -self.mass / (np.pi * self.hbar**2) * np.trapz(integrand, r)
        r = np.linspace(1e-10, r_max, 1000)
        integrand = r * np.array([V(ri) for ri in r]) * np.sin(q * r)
        return -2 * self.mass / (self.hbar**2 * q) * np.trapz(integrand, r)

    def differential_cross_section(self, V: Callable[[float], float],
                                   k: float, theta: float) -> float:
        """Calculate dσ/dΩ = |f(θ)|²."""
        return np.abs(self.scattering_amplitude_spherical(V, k, theta))**2


class HartreeFock(BaseClass):
    """
    Hartree-Fock self-consistent field method.

    Args:
        n_basis: Number of basis functions
        n_electrons: Number of electrons
        overlap: Overlap matrix S
        kinetic: Kinetic energy matrix T
        nuclear: Nuclear attraction matrix V_ne
        two_electron: Two-electron integrals
    """

    def __init__(self, n_basis: int, n_electrons: int, overlap: np.ndarray,
                 kinetic: np.ndarray, nuclear: np.ndarray, two_electron: np.ndarray):
        super().__init__()
        self.n_basis = n_basis
        self.n_electrons = n_electrons
        self.n_occ = n_electrons // 2
        self.S = np.array(overlap)
        self.T = np.array(kinetic)
        self.V_ne = np.array(nuclear)
        self.eri = np.array(two_electron)
        self.H_core = self.T + self.V_ne
        self.P = np.zeros((n_basis, n_basis))

    def build_fock_matrix(self) -> np.ndarray:
        """Build Fock matrix."""
        F = self.H_core.copy()
        for mu in range(self.n_basis):
            for nu in range(self.n_basis):
                for lam in range(self.n_basis):
                    for sig in range(self.n_basis):
                        F[mu, nu] += self.P[lam, sig] * self.eri[mu, nu, lam, sig]
                        F[mu, nu] -= 0.5 * self.P[lam, sig] * self.eri[mu, lam, nu, sig]
        return F

    def solve(self, max_iter: int = 100, tol: float = 1e-8) -> Tuple[float, np.ndarray]:
        """Run SCF iteration."""
        s, U = np.linalg.eigh(self.S)
        X = U @ np.diag(1.0 / np.sqrt(s))
        E_old = 0.0
        for _ in range(max_iter):
            F = self.build_fock_matrix()
            F_prime = X.T @ F @ X
            epsilon, C_prime = np.linalg.eigh(F_prime)
            C = X @ C_prime
            C_occ = C[:, :self.n_occ]
            self.P = 2 * C_occ @ C_occ.T
            E = 0.5 * np.trace(self.P @ (self.H_core + F))
            if abs(E - E_old) < tol:
                return E, C
            E_old = E
        return E_old, C


class DensityFunctional(BaseClass):
    """
    Density Functional Theory basics (LDA).

    Args:
        grid: Spatial grid points
        functional: 'lda', 'lda_x', 'lda_c'
    """

    def __init__(self, grid: np.ndarray, functional: str = 'lda'):
        super().__init__()
        self.grid = np.array(grid)
        self.dx = grid[1] - grid[0] if len(grid) > 1 else 1.0
        self.functional = functional

    def lda_exchange(self, rho: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """LDA exchange energy density and potential."""
        C_x = 0.7386
        rho_safe = np.maximum(rho, 1e-30)
        epsilon_x = -C_x * rho_safe**(1/3)
        v_x = -(4/3) * C_x * rho_safe**(1/3)
        return epsilon_x, v_x

    def exchange_correlation_energy(self, rho: np.ndarray) -> float:
        """Calculate XC energy."""
        epsilon_x, _ = self.lda_exchange(rho)
        return np.sum(rho * epsilon_x) * self.dx

    def exchange_correlation_potential(self, rho: np.ndarray) -> np.ndarray:
        """Calculate XC potential."""
        _, v_x = self.lda_exchange(rho)
        return v_x


# ==============================================================================
# Phase 5.6 Additions: Open Quantum Systems
# ==============================================================================

class QuantumChannel(BaseClass):
    """
    Quantum channel (CPTP map).

    Args:
        kraus_operators: List of Kraus operators

    Examples:
        >>> depol = QuantumChannel.depolarizing(p=0.1)
        >>> rho_out = depol.apply(rho_in)
    """

    def __init__(self, kraus_operators: List[np.ndarray]):
        super().__init__()
        self.kraus = [np.array(K, dtype=complex) for K in kraus_operators]
        self.dim = self.kraus[0].shape[0]

    def apply(self, rho: np.ndarray) -> np.ndarray:
        """Apply channel: ε(ρ) = Σ K_i ρ K_i†."""
        return sum(K @ rho @ K.T.conj() for K in self.kraus)

    @classmethod
    def depolarizing(cls, p: float, dim: int = 2) -> 'QuantumChannel':
        """Depolarizing channel."""
        I = np.eye(dim)
        K0 = np.sqrt(1 - 3*p/4) * I
        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]])
        factor = np.sqrt(p / 4)
        return cls([K0, factor * X, factor * Y, factor * Z])

    @classmethod
    def amplitude_damping(cls, gamma: float) -> 'QuantumChannel':
        """Amplitude damping channel."""
        K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
        K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
        return cls([K0, K1])

    @classmethod
    def phase_damping(cls, gamma: float) -> 'QuantumChannel':
        """Phase damping channel."""
        K0 = np.sqrt(1 - gamma) * np.eye(2, dtype=complex)
        K1 = np.sqrt(gamma) * np.array([[1, 0], [0, 0]], dtype=complex)
        K2 = np.sqrt(gamma) * np.array([[0, 0], [0, 1]], dtype=complex)
        return cls([K0, K1, K2])


class Decoherence(BaseClass):
    """
    Decoherence dynamics.

    Args:
        T1: Relaxation time
        T2: Dephasing time
    """

    def __init__(self, T1: float, T2: float):
        super().__init__()
        self.T1 = T1
        self.T2 = T2

    def evolve(self, rho0: np.ndarray, t: float) -> np.ndarray:
        """Evolve density matrix under decoherence."""
        rho = np.array(rho0, dtype=complex)
        if rho.shape == (2, 2):
            decay1 = np.exp(-t / self.T1)
            decay2 = np.exp(-t / self.T2)
            rho_new = np.zeros((2, 2), dtype=complex)
            rho_new[0, 0] = rho[0, 0] + (1 - decay1) * rho[1, 1]
            rho_new[1, 1] = decay1 * rho[1, 1]
            rho_new[0, 1] = decay2 * rho[0, 1]
            rho_new[1, 0] = decay2 * rho[1, 0]
            return rho_new
        raise NotImplementedError("Only qubit decoherence implemented")


class QuantumMeasurement(BaseClass):
    """
    Quantum measurement and POVM.

    Args:
        operators: Measurement operators
    """

    def __init__(self, operators: List[np.ndarray]):
        super().__init__()
        self.operators = [np.array(M, dtype=complex) for M in operators]
        self.n_outcomes = len(operators)
        self.dim = operators[0].shape[0]

    def probabilities(self, rho: np.ndarray) -> np.ndarray:
        """Calculate outcome probabilities."""
        return np.array([np.real(np.trace(M.T.conj() @ M @ rho)) for M in self.operators])

    def post_measurement_states(self, rho: np.ndarray) -> List[np.ndarray]:
        """Calculate post-measurement states."""
        probs = self.probabilities(rho)
        states = []
        for m, M in enumerate(self.operators):
            if probs[m] > 1e-15:
                states.append(M @ rho @ M.T.conj() / probs[m])
            else:
                states.append(np.zeros_like(rho))
        return states

    @classmethod
    def computational_basis(cls, dim: int = 2) -> 'QuantumMeasurement':
        """Projective measurement in computational basis."""
        projectors = [np.zeros((dim, dim), dtype=complex) for _ in range(dim)]
        for i in range(dim):
            projectors[i][i, i] = 1
        return cls(projectors)


# ==============================================================================
# Phase 5.7: Quantum Phenomena
# ==============================================================================

class QuantumTunneling(BaseClass):
    """
    Quantum tunneling through barriers.

    Args:
        mass: Particle mass
        hbar: Reduced Planck constant
    """

    def __init__(self, mass: float = M_E, hbar: float = HBAR):
        super().__init__()
        self.mass = mass
        self.hbar = hbar

    def rectangular_barrier(self, V0: float, width: float, E: float) -> float:
        """Transmission coefficient for rectangular barrier."""
        if E >= V0:
            k1 = np.sqrt(2 * self.mass * E) / self.hbar
            k2 = np.sqrt(2 * self.mass * (E - V0)) / self.hbar
            denom = 1 + (k1**2 - k2**2)**2 * np.sin(k2 * width)**2 / (4 * k1**2 * k2**2)
            return 1 / denom
        kappa = np.sqrt(2 * self.mass * (V0 - E)) / self.hbar
        sinh_ka = np.sinh(kappa * width)
        denom = 1 + V0**2 * sinh_ka**2 / (4 * E * (V0 - E))
        return 1 / denom

    def wkb_transmission(self, V: Callable[[float], float], E: float,
                        x1: float, x2: float) -> float:
        """WKB tunneling probability."""
        x = np.linspace(x1, x2, 1000)
        kappa = np.array([np.sqrt(max(0, 2*self.mass*(V(xi)-E))) / self.hbar for xi in x])
        return np.exp(-2 * np.trapz(kappa, x))


class QuantumInterference(BaseClass):
    """
    Quantum interference effects.

    Args:
        wavelength: de Broglie wavelength
    """

    def __init__(self, wavelength: float):
        super().__init__()
        self.wavelength = wavelength

    def double_slit(self, d: float, D: float, x: np.ndarray) -> np.ndarray:
        """Double-slit interference pattern."""
        phase = np.pi * d * x / (self.wavelength * D)
        return np.cos(phase)**2

    def visibility(self, I_max: float, I_min: float) -> float:
        """Fringe visibility."""
        return (I_max - I_min) / (I_max + I_min)


class Entanglement(BaseClass):
    """
    Bipartite entanglement measures.

    Args:
        rho: Density matrix
        dim_A: Dimension of subsystem A
        dim_B: Dimension of subsystem B
    """

    def __init__(self, rho: np.ndarray, dim_A: int = 2, dim_B: int = 2):
        super().__init__()
        self.rho = np.array(rho, dtype=complex)
        self.dim_A = dim_A
        self.dim_B = dim_B

    def partial_trace_B(self) -> np.ndarray:
        """Trace out subsystem B."""
        rho_full = self.rho.reshape((self.dim_A, self.dim_B, self.dim_A, self.dim_B))
        return np.trace(rho_full, axis1=1, axis2=3)

    def entanglement_entropy(self) -> float:
        """Calculate entanglement entropy."""
        rho_A = self.partial_trace_B()
        eigenvals = np.linalg.eigvalsh(rho_A)
        eigenvals = eigenvals[eigenvals > 1e-15]
        return -np.sum(eigenvals * np.log2(eigenvals))

    def concurrence(self) -> float:
        """Calculate concurrence for two qubits."""
        if self.dim_A * self.dim_B != 4:
            raise ValueError("Concurrence only for two qubits")
        sigma_y = np.array([[0, -1j], [1j, 0]])
        Y_Y = np.kron(sigma_y, sigma_y)
        rho_tilde = Y_Y @ self.rho.conj() @ Y_Y
        R = self.rho @ rho_tilde
        eigenvals = np.sqrt(np.abs(np.linalg.eigvals(R)))
        eigenvals = np.sort(eigenvals)[::-1]
        return max(0, eigenvals[0] - eigenvals[1] - eigenvals[2] - eigenvals[3])

    def negativity(self) -> float:
        """Calculate negativity."""
        rho_reshaped = self.rho.reshape((self.dim_A, self.dim_B, self.dim_A, self.dim_B))
        rho_pt = rho_reshaped.transpose((2, 1, 0, 3)).reshape((self.dim_A * self.dim_B, -1))
        eigenvals = np.linalg.eigvalsh(rho_pt)
        return (np.sum(np.abs(eigenvals)) - 1) / 2


class BellState(BaseClass):
    """
    Bell states and properties.
    """

    STATES = {
        'phi_plus': np.array([1, 0, 0, 1]) / np.sqrt(2),
        'phi_minus': np.array([1, 0, 0, -1]) / np.sqrt(2),
        'psi_plus': np.array([0, 1, 1, 0]) / np.sqrt(2),
        'psi_minus': np.array([0, 1, -1, 0]) / np.sqrt(2),
    }

    def __init__(self, name: str = 'phi_plus'):
        super().__init__()
        self.name = name
        self._state = self.STATES[name].astype(complex)

    def state_vector(self) -> np.ndarray:
        """Return Bell state vector."""
        return self._state.copy()

    def density_matrix(self) -> np.ndarray:
        """Return density matrix."""
        return np.outer(self._state, self._state.conj())


class BellInequality(BaseClass):
    """
    Bell inequality tests (CHSH).
    """

    def __init__(self):
        super().__init__()

    def correlation(self, rho: np.ndarray, theta_A: float, theta_B: float) -> float:
        """Calculate E(a,b)."""
        sigma_a = np.array([[np.cos(theta_A), np.sin(theta_A)],
                           [np.sin(theta_A), -np.cos(theta_A)]])
        sigma_b = np.array([[np.cos(theta_B), np.sin(theta_B)],
                           [np.sin(theta_B), -np.cos(theta_B)]])
        return np.real(np.trace(rho @ np.kron(sigma_a, sigma_b)))

    def chsh_parameter(self, rho: np.ndarray,
                      angles: Tuple = (0, np.pi/4, np.pi/2, 3*np.pi/4)) -> float:
        """Calculate CHSH parameter S."""
        a, a_p, b, b_p = angles
        return (self.correlation(rho, a, b) - self.correlation(rho, a, b_p) +
                self.correlation(rho, a_p, b) + self.correlation(rho, a_p, b_p))

    def violates_bell(self, rho: np.ndarray) -> bool:
        """Check if state violates Bell inequality."""
        return abs(self.chsh_parameter(rho)) > 2


class QuantumTeleportation(BaseClass):
    """
    Quantum teleportation protocol.
    """

    def __init__(self):
        super().__init__()
        self.corrections = {
            (0, 0): np.eye(2),
            (0, 1): np.array([[0, 1], [1, 0]]),
            (1, 0): np.array([[1, 0], [0, -1]]),
            (1, 1): np.array([[0, -1], [1, 0]]),
        }

    def teleport(self, psi_in: np.ndarray) -> np.ndarray:
        """Perform teleportation (ideal)."""
        outcome = tuple(np.random.randint(2, size=2))
        return self.corrections[outcome] @ self.corrections[outcome] @ psi_in

    def fidelity(self, psi_in: np.ndarray, psi_out: np.ndarray) -> float:
        """Calculate fidelity."""
        return np.abs(np.vdot(psi_in, psi_out))**2


class QuantumZenoEffect(BaseClass):
    """
    Quantum Zeno effect.

    Args:
        H: System Hamiltonian
        hbar: Reduced Planck constant
    """

    def __init__(self, H: np.ndarray, hbar: float = HBAR):
        super().__init__()
        self.H = np.array(H, dtype=complex)
        self.hbar = hbar

    def survival_probability(self, psi0: np.ndarray, t: float,
                            n_measurements: int) -> float:
        """Survival probability with n measurements."""
        dt = t / n_measurements
        U = la.expm(-1j * self.H * dt / self.hbar)
        prob = 1.0
        for _ in range(n_measurements):
            psi = U @ psi0
            prob *= np.abs(np.vdot(psi0, psi))**2
        return prob

    def survival_no_measurement(self, psi0: np.ndarray, t: float) -> float:
        """Survival without measurements."""
        U = la.expm(-1j * self.H * t / self.hbar)
        psi_t = U @ psi0
        return np.abs(np.vdot(psi0, psi_t))**2
