"""
Condensed Matter Physics Module

This module provides tools for solid state physics including:
- Crystal Structure: Bravais lattices, reciprocal space, Brillouin zones
- Band Theory: Bloch waves, tight binding, effective mass
- Semiconductors: Doping, junctions, quantum confinement
- Transport: Drude model, Boltzmann transport, Hall effect
- Lattice Dynamics: Phonons, thermal conductivity
- Magnetism: Dia/para/ferromagnetism, spin waves
- Superconductivity: BCS theory, Meissner effect, Josephson junctions
- Topological Matter: Berry phase, quantum Hall, topological insulators
"""

import numpy as np
from typing import Optional, Callable, Tuple, List, Dict, Union
from numpy.typing import ArrayLike
from scipy.integrate import quad, dblquad, solve_ivp
from scipy.linalg import eigh, eig
from scipy.special import factorial
from scipy.optimize import brentq, minimize
import warnings


# =============================================================================
# Physical Constants
# =============================================================================

HBAR = 1.054571817e-34  # Reduced Planck constant (J·s)
KB = 1.380649e-23  # Boltzmann constant (J/K)
E_CHARGE = 1.602176634e-19  # Elementary charge (C)
M_ELECTRON = 9.1093837015e-31  # Electron mass (kg)
EPSILON_0 = 8.8541878128e-12  # Vacuum permittivity (F/m)
MU_B = 9.2740100783e-24  # Bohr magneton (J/T)


# =============================================================================
# Crystal Structure
# =============================================================================

class BravaisLattice:
    """Bravais lattice structure (14 types in 3D)"""

    def __init__(self, lattice_type: str, a: float, b: float = None,
                 c: float = None, alpha: float = 90, beta: float = 90,
                 gamma: float = 90):
        """
        Initialize Bravais lattice

        Args:
            lattice_type: 'cubic', 'fcc', 'bcc', 'hexagonal', 'tetragonal', etc.
            a, b, c: Lattice constants (Angstroms or meters)
            alpha, beta, gamma: Lattice angles (degrees)
        """
        self.lattice_type = lattice_type
        self.a = a
        self.b = b if b is not None else a
        self.c = c if c is not None else a
        self.alpha = np.radians(alpha)
        self.beta = np.radians(beta)
        self.gamma = np.radians(gamma)

        self._compute_lattice_vectors()

    def _compute_lattice_vectors(self):
        """Compute primitive lattice vectors"""
        a, b, c = self.a, self.b, self.c
        alpha, beta, gamma = self.alpha, self.beta, self.gamma

        if self.lattice_type == 'cubic' or self.lattice_type == 'simple_cubic':
            self.a1 = np.array([a, 0, 0])
            self.a2 = np.array([0, a, 0])
            self.a3 = np.array([0, 0, a])

        elif self.lattice_type == 'fcc':
            self.a1 = 0.5 * a * np.array([0, 1, 1])
            self.a2 = 0.5 * a * np.array([1, 0, 1])
            self.a3 = 0.5 * a * np.array([1, 1, 0])

        elif self.lattice_type == 'bcc':
            self.a1 = 0.5 * a * np.array([-1, 1, 1])
            self.a2 = 0.5 * a * np.array([1, -1, 1])
            self.a3 = 0.5 * a * np.array([1, 1, -1])

        elif self.lattice_type == 'hexagonal':
            self.a1 = a * np.array([1, 0, 0])
            self.a2 = a * np.array([-0.5, np.sqrt(3)/2, 0])
            self.a3 = np.array([0, 0, c])

        else:  # General triclinic
            cos_alpha = np.cos(alpha)
            cos_beta = np.cos(beta)
            cos_gamma = np.cos(gamma)
            sin_gamma = np.sin(gamma)

            self.a1 = np.array([a, 0, 0])
            self.a2 = np.array([b * cos_gamma, b * sin_gamma, 0])
            cx = c * cos_beta
            cy = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
            cz = np.sqrt(c**2 - cx**2 - cy**2)
            self.a3 = np.array([cx, cy, cz])

    def volume(self) -> float:
        """Calculate unit cell volume"""
        return np.abs(np.dot(self.a1, np.cross(self.a2, self.a3)))

    def lattice_point(self, n1: int, n2: int, n3: int) -> np.ndarray:
        """Get lattice point R = n1*a1 + n2*a2 + n3*a3"""
        return n1 * self.a1 + n2 * self.a2 + n3 * self.a3

    def generate_lattice(self, n_cells: int = 3) -> np.ndarray:
        """
        Generate lattice points in a region

        Args:
            n_cells: Number of unit cells in each direction

        Returns:
            Array of lattice points
        """
        points = []
        for n1 in range(-n_cells, n_cells + 1):
            for n2 in range(-n_cells, n_cells + 1):
                for n3 in range(-n_cells, n_cells + 1):
                    points.append(self.lattice_point(n1, n2, n3))
        return np.array(points)


class ReciprocalLattice:
    """Reciprocal lattice in k-space"""

    def __init__(self, bravais: BravaisLattice):
        """
        Initialize reciprocal lattice

        Args:
            bravais: Real space Bravais lattice
        """
        self.bravais = bravais
        self._compute_reciprocal_vectors()

    def _compute_reciprocal_vectors(self):
        """Compute reciprocal lattice vectors b_i = 2π (a_j × a_k) / V"""
        a1, a2, a3 = self.bravais.a1, self.bravais.a2, self.bravais.a3
        V = self.bravais.volume()

        self.b1 = 2 * np.pi * np.cross(a2, a3) / V
        self.b2 = 2 * np.pi * np.cross(a3, a1) / V
        self.b3 = 2 * np.pi * np.cross(a1, a2) / V

    def reciprocal_point(self, m1: int, m2: int, m3: int) -> np.ndarray:
        """Get reciprocal lattice vector G = m1*b1 + m2*b2 + m3*b3"""
        return m1 * self.b1 + m2 * self.b2 + m3 * self.b3

    def volume(self) -> float:
        """Calculate reciprocal unit cell volume (2π)³/V"""
        return (2 * np.pi)**3 / self.bravais.volume()


class BrillouinZone:
    """First Brillouin zone construction"""

    def __init__(self, reciprocal: ReciprocalLattice):
        """
        Initialize Brillouin zone

        Args:
            reciprocal: Reciprocal lattice
        """
        self.reciprocal = reciprocal
        self.lattice_type = reciprocal.bravais.lattice_type

    def high_symmetry_points(self) -> Dict[str, np.ndarray]:
        """
        Get high-symmetry points in the BZ

        Returns:
            Dictionary of point names to k-vectors
        """
        b1, b2, b3 = self.reciprocal.b1, self.reciprocal.b2, self.reciprocal.b3

        if self.lattice_type in ['cubic', 'simple_cubic']:
            return {
                'Γ': np.array([0, 0, 0]),
                'X': 0.5 * b1,
                'M': 0.5 * (b1 + b2),
                'R': 0.5 * (b1 + b2 + b3)
            }

        elif self.lattice_type == 'fcc':
            return {
                'Γ': np.array([0, 0, 0]),
                'X': 0.5 * (b1 + b3),
                'L': 0.5 * (b1 + b2 + b3),
                'W': 0.25 * (2*b1 + b2 + b3),
                'K': 0.375 * (2*b1 + b2 + b3)
            }

        elif self.lattice_type == 'bcc':
            return {
                'Γ': np.array([0, 0, 0]),
                'H': 0.5 * (b1 + b2 - b3),
                'N': 0.5 * (b1 + b2),
                'P': 0.25 * (b1 + b2 + b3)
            }

        else:
            return {'Γ': np.array([0, 0, 0])}

    def is_in_first_bz(self, k: np.ndarray) -> bool:
        """
        Check if k-point is in first Brillouin zone

        Args:
            k: k-vector

        Returns:
            True if in first BZ
        """
        # Simplified check: distance to Γ < distance to any G
        k_norm = np.linalg.norm(k)
        for m1 in range(-1, 2):
            for m2 in range(-1, 2):
                for m3 in range(-1, 2):
                    if m1 == 0 and m2 == 0 and m3 == 0:
                        continue
                    G = self.reciprocal.reciprocal_point(m1, m2, m3)
                    if np.linalg.norm(k - G) < k_norm - 1e-10:
                        return False
        return True

    def path_through_bz(self, point_names: List[str],
                        n_points: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate k-path through high-symmetry points

        Args:
            point_names: List of point names (e.g., ['Γ', 'X', 'M', 'Γ'])
            n_points: Points per segment

        Returns:
            (k_path, k_distances) arrays
        """
        points = self.high_symmetry_points()
        k_path = []
        k_dist = []
        total_dist = 0

        for i in range(len(point_names) - 1):
            start = points[point_names[i]]
            end = points[point_names[i + 1]]

            for j in range(n_points):
                t = j / n_points
                k = start + t * (end - start)
                k_path.append(k)
                if j > 0 or i > 0:
                    total_dist += np.linalg.norm(k - k_path[-2]) if len(k_path) > 1 else 0
                k_dist.append(total_dist)

        return np.array(k_path), np.array(k_dist)


class CrystalSymmetry:
    """Point and space group symmetry operations"""

    def __init__(self, point_group: str = 'Oh'):
        """
        Initialize crystal symmetry

        Args:
            point_group: Point group symbol (e.g., 'Oh' for cubic)
        """
        self.point_group = point_group
        self._init_operations()

    def _init_operations(self):
        """Initialize symmetry operations"""
        if self.point_group == 'Oh':  # Cubic full symmetry
            # Identity and inversions
            self.operations = [np.eye(3), -np.eye(3)]

            # C4 rotations around axes
            for axis in [0, 1, 2]:
                for angle in [np.pi/2, np.pi, 3*np.pi/2]:
                    R = self._rotation_matrix(axis, angle)
                    self.operations.append(R)
                    self.operations.append(-R)

        else:
            self.operations = [np.eye(3)]  # Identity only

    def _rotation_matrix(self, axis: int, angle: float) -> np.ndarray:
        """Generate rotation matrix around given axis"""
        c, s = np.cos(angle), np.sin(angle)
        if axis == 0:
            return np.array([[1,0,0], [0,c,-s], [0,s,c]])
        elif axis == 1:
            return np.array([[c,0,s], [0,1,0], [-s,0,c]])
        else:
            return np.array([[c,-s,0], [s,c,0], [0,0,1]])

    def apply(self, vector: np.ndarray) -> List[np.ndarray]:
        """Apply all symmetry operations to a vector"""
        return [R @ vector for R in self.operations]


class MillerIndices:
    """Miller indices for crystal planes"""

    def __init__(self, h: int, k: int, l: int):
        """
        Initialize Miller indices (hkl)

        Args:
            h, k, l: Miller indices
        """
        self.h = h
        self.k = k
        self.l = l
        self.indices = (h, k, l)

    def plane_normal(self, bravais: BravaisLattice) -> np.ndarray:
        """
        Get normal vector to the (hkl) plane

        Args:
            bravais: Bravais lattice

        Returns:
            Normal vector
        """
        reciprocal = ReciprocalLattice(bravais)
        G = self.h * reciprocal.b1 + self.k * reciprocal.b2 + self.l * reciprocal.b3
        return G / np.linalg.norm(G)

    def d_spacing(self, bravais: BravaisLattice) -> float:
        """
        Calculate d-spacing between planes

        Args:
            bravais: Bravais lattice

        Returns:
            Interplanar spacing
        """
        reciprocal = ReciprocalLattice(bravais)
        G = self.h * reciprocal.b1 + self.k * reciprocal.b2 + self.l * reciprocal.b3
        return 2 * np.pi / np.linalg.norm(G)

    def bragg_angle(self, wavelength: float, bravais: BravaisLattice) -> float:
        """
        Calculate Bragg diffraction angle

        2d sin(θ) = nλ (n=1)

        Args:
            wavelength: X-ray wavelength
            bravais: Bravais lattice

        Returns:
            Bragg angle (radians)
        """
        d = self.d_spacing(bravais)
        sin_theta = wavelength / (2 * d)
        if abs(sin_theta) > 1:
            return None  # No diffraction
        return np.arcsin(sin_theta)


# =============================================================================
# Band Theory
# =============================================================================

class BlochWavefunction:
    """Bloch wavefunction ψ_k(r) = e^{ik·r} u_k(r)"""

    def __init__(self, k: np.ndarray, bravais: BravaisLattice):
        """
        Initialize Bloch wavefunction

        Args:
            k: Crystal momentum
            bravais: Lattice structure
        """
        self.k = np.array(k)
        self.bravais = bravais
        self.u_k = None  # Periodic part (set later)

    def set_periodic_part(self, u_k: Callable):
        """Set periodic part u_k(r)"""
        self.u_k = u_k

    def evaluate(self, r: np.ndarray) -> complex:
        """
        Evaluate wavefunction at position r

        Args:
            r: Position vector

        Returns:
            ψ_k(r)
        """
        if self.u_k is None:
            raise ValueError("Periodic part u_k not set")
        return np.exp(1j * np.dot(self.k, r)) * self.u_k(r)

    def probability_density(self, r: np.ndarray) -> float:
        """Calculate |ψ_k(r)|²"""
        return np.abs(self.evaluate(r))**2


class KronigPenney:
    """Kronig-Penney model for 1D band structure"""

    def __init__(self, V0: float, a: float, b: float, m: float = M_ELECTRON):
        """
        Initialize Kronig-Penney model

        Periodic square well: V = 0 for 0 < x < a, V = V0 for a < x < a+b

        Args:
            V0: Barrier height (J)
            a: Well width (m)
            b: Barrier width (m)
            m: Particle mass
        """
        self.V0 = V0
        self.a = a  # Well width
        self.b = b  # Barrier width
        self.period = a + b
        self.m = m

    def dispersion_equation(self, E: float, k: float) -> float:
        """
        Kronig-Penney dispersion relation

        cos(k(a+b)) = cos(αa)cosh(βb) - (β²-α²)/(2αβ) sin(αa)sinh(βb)

        Args:
            E: Energy
            k: Crystal momentum

        Returns:
            Residual (= 0 for allowed energies)
        """
        if E < 0:
            return np.inf

        alpha = np.sqrt(2 * self.m * E) / HBAR

        if E < self.V0:
            beta = np.sqrt(2 * self.m * (self.V0 - E)) / HBAR
            lhs = np.cos(k * self.period)
            rhs = (np.cos(alpha * self.a) * np.cosh(beta * self.b) -
                   (beta**2 - alpha**2) / (2 * alpha * beta) *
                   np.sin(alpha * self.a) * np.sinh(beta * self.b))
        else:
            kappa = np.sqrt(2 * self.m * (E - self.V0)) / HBAR
            lhs = np.cos(k * self.period)
            rhs = (np.cos(alpha * self.a) * np.cos(kappa * self.b) -
                   (kappa**2 + alpha**2) / (2 * alpha * kappa) *
                   np.sin(alpha * self.a) * np.sin(kappa * self.b))

        return lhs - rhs

    def band_energies(self, k: float, n_bands: int = 4) -> List[float]:
        """
        Find band energies at given k

        Args:
            k: Crystal momentum
            n_bands: Number of bands to find

        Returns:
            List of energies
        """
        energies = []
        E_max = 10 * self.V0

        # Search for solutions
        E_search = np.linspace(1e-6, E_max, 1000)
        for i in range(len(E_search) - 1):
            try:
                f1 = self.dispersion_equation(E_search[i], k)
                f2 = self.dispersion_equation(E_search[i+1], k)
                if f1 * f2 < 0:  # Sign change
                    E = brentq(lambda E: self.dispersion_equation(E, k),
                              E_search[i], E_search[i+1])
                    energies.append(E)
                    if len(energies) >= n_bands:
                        break
            except (ValueError, RuntimeError):
                continue

        return energies

    def band_structure(self, n_k: int = 50, n_bands: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate band structure E(k)

        Args:
            n_k: Number of k-points
            n_bands: Number of bands

        Returns:
            (k_values, energies) arrays
        """
        k_bz = np.pi / self.period  # BZ boundary
        k_values = np.linspace(-k_bz, k_bz, n_k)
        energies = np.zeros((n_k, n_bands))

        for i, k in enumerate(k_values):
            E_k = self.band_energies(k, n_bands)
            for j, E in enumerate(E_k):
                if j < n_bands:
                    energies[i, j] = E

        return k_values, energies


class TightBinding:
    """Tight-binding model for band structure"""

    def __init__(self, lattice: BravaisLattice, t: float, epsilon: float = 0):
        """
        Initialize tight-binding model

        Args:
            lattice: Bravais lattice
            t: Hopping parameter (J)
            epsilon: On-site energy (J)
        """
        self.lattice = lattice
        self.t = t
        self.epsilon = epsilon

    def hamiltonian_1d(self, k: float, n_sites: int = 1) -> np.ndarray:
        """
        1D tight-binding Hamiltonian

        H(k) = ε - 2t cos(ka)

        Args:
            k: Crystal momentum
            n_sites: Number of sites in unit cell

        Returns:
            Hamiltonian (scalar for simple case)
        """
        a = self.lattice.a
        return self.epsilon - 2 * self.t * np.cos(k * a)

    def hamiltonian_2d(self, kx: float, ky: float) -> float:
        """
        2D square lattice tight-binding

        H(k) = ε - 2t[cos(kx*a) + cos(ky*a)]

        Args:
            kx, ky: Crystal momentum components

        Returns:
            Energy
        """
        a = self.lattice.a
        return self.epsilon - 2 * self.t * (np.cos(kx * a) + np.cos(ky * a))

    def hamiltonian_3d(self, k: np.ndarray) -> float:
        """
        3D simple cubic tight-binding

        H(k) = ε - 2t[cos(kx*a) + cos(ky*a) + cos(kz*a)]

        Args:
            k: Crystal momentum vector

        Returns:
            Energy
        """
        a = self.lattice.a
        return self.epsilon - 2 * self.t * (np.cos(k[0] * a) +
                                             np.cos(k[1] * a) +
                                             np.cos(k[2] * a))

    def bandwidth(self) -> float:
        """Calculate bandwidth (for simple cubic)"""
        return 12 * self.t  # 6t - (-6t) = 12t

    def dos_1d(self, E: float) -> float:
        """
        1D density of states (van Hove singularities)

        Args:
            E: Energy

        Returns:
            DOS (states per energy per site)
        """
        E_rel = (E - self.epsilon) / (2 * self.t)
        if abs(E_rel) >= 1:
            return 0
        return 1 / (np.pi * self.t * np.sqrt(1 - E_rel**2))


class NearlyFreeElectron:
    """Nearly free electron model (weak periodic potential)"""

    def __init__(self, V_G: Dict[Tuple, float], lattice: BravaisLattice,
                 m: float = M_ELECTRON):
        """
        Initialize NFE model

        Args:
            V_G: Fourier components of potential {G: V_G}
            lattice: Crystal lattice
            m: Electron mass
        """
        self.V_G = V_G
        self.lattice = lattice
        self.m = m
        self.reciprocal = ReciprocalLattice(lattice)

    def free_electron_energy(self, k: np.ndarray) -> float:
        """Free electron energy E = ℏ²k²/(2m)"""
        return HBAR**2 * np.dot(k, k) / (2 * self.m)

    def band_gap_at_zone_boundary(self, G: np.ndarray) -> float:
        """
        Calculate band gap at zone boundary

        E_gap ≈ 2|V_G|

        Args:
            G: Reciprocal lattice vector

        Returns:
            Band gap
        """
        G_tuple = tuple(G.astype(int))
        if G_tuple in self.V_G:
            return 2 * abs(self.V_G[G_tuple])
        return 0

    def dispersion_near_gap(self, k: np.ndarray, G: np.ndarray) -> Tuple[float, float]:
        """
        Calculate energies near zone boundary (two-band model)

        Args:
            k: k-vector near G/2
            G: Reciprocal lattice vector

        Returns:
            (E_lower, E_upper) band energies
        """
        E_k = self.free_electron_energy(k)
        E_kG = self.free_electron_energy(k - G)
        V = self.band_gap_at_zone_boundary(G) / 2

        avg = 0.5 * (E_k + E_kG)
        diff = 0.5 * (E_k - E_kG)
        delta = np.sqrt(diff**2 + V**2)

        return avg - delta, avg + delta


class EffectiveMass:
    """Effective mass from band curvature"""

    def __init__(self, band_func: Callable):
        """
        Initialize effective mass calculator

        Args:
            band_func: Function E(k) returning band energy
        """
        self.E = band_func

    def mass_tensor(self, k: np.ndarray, h: float = 1e-10) -> np.ndarray:
        """
        Calculate effective mass tensor

        (1/m*)_ij = (1/ℏ²) ∂²E/∂k_i∂k_j

        Args:
            k: k-point
            h: Finite difference step

        Returns:
            3×3 inverse mass tensor (×ℏ²)
        """
        inv_mass = np.zeros((3, 3))

        for i in range(3):
            for j in range(3):
                k_pp = k.copy(); k_pp[i] += h; k_pp[j] += h
                k_pm = k.copy(); k_pm[i] += h; k_pm[j] -= h
                k_mp = k.copy(); k_mp[i] -= h; k_mp[j] += h
                k_mm = k.copy(); k_mm[i] -= h; k_mm[j] -= h

                d2E = (self.E(k_pp) - self.E(k_pm) - self.E(k_mp) + self.E(k_mm)) / (4 * h**2)
                inv_mass[i, j] = d2E / HBAR**2

        return inv_mass

    def scalar_mass(self, k: np.ndarray, h: float = 1e-10) -> float:
        """
        Calculate scalar effective mass (for isotropic bands)

        Args:
            k: k-point
            h: Finite difference step

        Returns:
            Effective mass
        """
        inv_m = self.mass_tensor(k, h)
        # Average of diagonal elements
        return 3 / np.trace(inv_m)


class DensityOfStates:
    """Electronic density of states calculations"""

    def __init__(self, band_func: Callable, bz: BrillouinZone):
        """
        Initialize DOS calculator

        Args:
            band_func: Function E(k) returning band energy
            bz: Brillouin zone
        """
        self.E = band_func
        self.bz = bz

    def histogram_dos(self, E_range: Tuple[float, float], n_bins: int = 100,
                      n_k: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate DOS by histogram method

        Args:
            E_range: (E_min, E_max) energy range
            n_bins: Number of energy bins
            n_k: k-points per dimension

        Returns:
            (E_values, DOS) arrays
        """
        E_bins = np.linspace(E_range[0], E_range[1], n_bins + 1)
        dE = E_bins[1] - E_bins[0]
        dos = np.zeros(n_bins)

        # Sample BZ
        b1, b2, b3 = self.bz.reciprocal.b1, self.bz.reciprocal.b2, self.bz.reciprocal.b3
        n_total = 0

        for i1 in range(n_k):
            for i2 in range(n_k):
                for i3 in range(n_k):
                    k = (i1/n_k - 0.5) * b1 + (i2/n_k - 0.5) * b2 + (i3/n_k - 0.5) * b3
                    E = self.E(k)
                    bin_idx = int((E - E_range[0]) / dE)
                    if 0 <= bin_idx < n_bins:
                        dos[bin_idx] += 1
                    n_total += 1

        # Normalize
        V_bz = self.bz.reciprocal.volume()
        dos = dos / (n_total * dE) * V_bz / (2 * np.pi)**3

        E_centers = 0.5 * (E_bins[:-1] + E_bins[1:])
        return E_centers, dos


class FermiSurface:
    """Fermi surface calculations"""

    def __init__(self, band_func: Callable, E_fermi: float, bz: BrillouinZone):
        """
        Initialize Fermi surface

        Args:
            band_func: Function E(k) returning band energy
            E_fermi: Fermi energy
            bz: Brillouin zone
        """
        self.E = band_func
        self.E_f = E_fermi
        self.bz = bz

    def is_on_surface(self, k: np.ndarray, tol: float = 0.01) -> bool:
        """Check if k-point is on Fermi surface"""
        return abs(self.E(k) - self.E_f) / abs(self.E_f) < tol

    def fermi_velocity(self, k: np.ndarray, h: float = 1e-10) -> np.ndarray:
        """
        Calculate Fermi velocity v_F = (1/ℏ)∇_k E

        Args:
            k: k-point on Fermi surface
            h: Finite difference step

        Returns:
            Fermi velocity vector
        """
        grad_E = np.zeros(3)
        for i in range(3):
            k_plus = k.copy(); k_plus[i] += h
            k_minus = k.copy(); k_minus[i] -= h
            grad_E[i] = (self.E(k_plus) - self.E(k_minus)) / (2 * h)
        return grad_E / HBAR

    def generate_surface_points(self, n_points: int = 1000,
                                tol: float = 0.01) -> np.ndarray:
        """
        Generate points on Fermi surface

        Args:
            n_points: Number of points to sample
            tol: Tolerance for being "on" surface

        Returns:
            Array of k-points on Fermi surface
        """
        surface_points = []
        b1 = self.bz.reciprocal.b1
        b2 = self.bz.reciprocal.b2
        b3 = self.bz.reciprocal.b3

        n_search = int(n_points ** (1/3)) + 1

        for i1 in range(n_search):
            for i2 in range(n_search):
                for i3 in range(n_search):
                    k = ((i1/n_search - 0.5) * b1 +
                         (i2/n_search - 0.5) * b2 +
                         (i3/n_search - 0.5) * b3)
                    if self.is_on_surface(k, tol):
                        surface_points.append(k)

        return np.array(surface_points) if surface_points else np.array([]).reshape(0, 3)


# =============================================================================
# Semiconductors
# =============================================================================

class IntrinsicSemiconductor:
    """Intrinsic (undoped) semiconductor statistics"""

    def __init__(self, E_g: float, m_e: float = M_ELECTRON,
                 m_h: float = M_ELECTRON, T: float = 300):
        """
        Initialize intrinsic semiconductor

        Args:
            E_g: Band gap (J)
            m_e: Electron effective mass
            m_h: Hole effective mass
            T: Temperature (K)
        """
        self.E_g = E_g
        self.m_e = m_e
        self.m_h = m_h
        self.T = T

    def intrinsic_carrier_density(self) -> float:
        """
        Calculate intrinsic carrier concentration n_i

        n_i = √(N_c N_v) exp(-E_g/(2kT))

        Returns:
            Carrier density (m⁻³)
        """
        N_c = self.conduction_band_dos()
        N_v = self.valence_band_dos()
        return np.sqrt(N_c * N_v) * np.exp(-self.E_g / (2 * KB * self.T))

    def conduction_band_dos(self) -> float:
        """Calculate effective density of states in conduction band"""
        return 2 * (2 * np.pi * self.m_e * KB * self.T / (2 * np.pi * HBAR)**2)**(3/2)

    def valence_band_dos(self) -> float:
        """Calculate effective density of states in valence band"""
        return 2 * (2 * np.pi * self.m_h * KB * self.T / (2 * np.pi * HBAR)**2)**(3/2)

    def fermi_level(self) -> float:
        """
        Calculate Fermi level (from valence band top)

        E_F = E_g/2 + (3/4)kT ln(m_h/m_e)

        Returns:
            Fermi energy (J)
        """
        return self.E_g / 2 + 0.75 * KB * self.T * np.log(self.m_h / self.m_e)


class DopedSemiconductor:
    """Doped semiconductor (n-type or p-type)"""

    def __init__(self, intrinsic: IntrinsicSemiconductor,
                 N_d: float = 0, N_a: float = 0,
                 E_d: float = 0, E_a: float = 0):
        """
        Initialize doped semiconductor

        Args:
            intrinsic: Intrinsic semiconductor
            N_d: Donor concentration (m⁻³)
            N_a: Acceptor concentration (m⁻³)
            E_d: Donor ionization energy (J)
            E_a: Acceptor ionization energy (J)
        """
        self.intrinsic = intrinsic
        self.N_d = N_d
        self.N_a = N_a
        self.E_d = E_d
        self.E_a = E_a

    @property
    def doping_type(self) -> str:
        """Return 'n' or 'p' type"""
        if self.N_d > self.N_a:
            return 'n'
        return 'p'

    def carrier_concentrations(self) -> Tuple[float, float]:
        """
        Calculate electron and hole concentrations

        Returns:
            (n, p) carrier densities (m⁻³)
        """
        n_i = self.intrinsic.intrinsic_carrier_density()

        if self.doping_type == 'n':
            n = 0.5 * ((self.N_d - self.N_a) +
                       np.sqrt((self.N_d - self.N_a)**2 + 4 * n_i**2))
            p = n_i**2 / n
        else:
            p = 0.5 * ((self.N_a - self.N_d) +
                       np.sqrt((self.N_a - self.N_d)**2 + 4 * n_i**2))
            n = n_i**2 / p

        return n, p

    def fermi_level(self) -> float:
        """Calculate Fermi level position"""
        n, p = self.carrier_concentrations()
        N_c = self.intrinsic.conduction_band_dos()
        kT = KB * self.intrinsic.T

        # E_F from conduction band
        if n > 0:
            return self.intrinsic.E_g - kT * np.log(N_c / n)
        return self.intrinsic.E_g / 2

    def conductivity(self, mu_n: float, mu_p: float) -> float:
        """
        Calculate electrical conductivity

        Args:
            mu_n, mu_p: Electron and hole mobilities (m²/V/s)

        Returns:
            Conductivity (S/m)
        """
        n, p = self.carrier_concentrations()
        return E_CHARGE * (n * mu_n + p * mu_p)


class PNJunction:
    """PN junction physics"""

    def __init__(self, n_side: DopedSemiconductor, p_side: DopedSemiconductor,
                 epsilon_r: float = 11.7):
        """
        Initialize PN junction

        Args:
            n_side: N-type semiconductor
            p_side: P-type semiconductor
            epsilon_r: Relative permittivity
        """
        self.n_side = n_side
        self.p_side = p_side
        self.epsilon = epsilon_r * EPSILON_0

    def built_in_voltage(self) -> float:
        """
        Calculate built-in voltage V_bi

        V_bi = (kT/e) ln(N_a N_d / n_i²)

        Returns:
            Built-in voltage (V)
        """
        n_i = self.n_side.intrinsic.intrinsic_carrier_density()
        N_d = self.n_side.N_d
        N_a = self.p_side.N_a
        kT = KB * self.n_side.intrinsic.T

        return (kT / E_CHARGE) * np.log(N_a * N_d / n_i**2)

    def depletion_width(self, V_a: float = 0) -> float:
        """
        Calculate depletion region width

        W = √(2ε(V_bi - V_a)(1/N_a + 1/N_d) / e)

        Args:
            V_a: Applied voltage (positive = forward bias)

        Returns:
            Depletion width (m)
        """
        V_bi = self.built_in_voltage()
        N_a = self.p_side.N_a
        N_d = self.n_side.N_d

        W_squared = 2 * self.epsilon * (V_bi - V_a) * (1/N_a + 1/N_d) / E_CHARGE
        return np.sqrt(max(W_squared, 0))

    def current_density(self, V_a: float, I_s: float = 1e-12) -> float:
        """
        Calculate current density using ideal diode equation

        J = J_s (exp(eV/kT) - 1)

        Args:
            V_a: Applied voltage (V)
            I_s: Saturation current density (A/m²)

        Returns:
            Current density (A/m²)
        """
        kT = KB * self.n_side.intrinsic.T
        return I_s * (np.exp(E_CHARGE * V_a / kT) - 1)

    def capacitance_per_area(self, V_a: float = 0) -> float:
        """
        Calculate junction capacitance per unit area

        C/A = ε/W

        Args:
            V_a: Applied voltage

        Returns:
            Capacitance per area (F/m²)
        """
        W = self.depletion_width(V_a)
        if W == 0:
            return np.inf
        return self.epsilon / W


class QuantumWell:
    """Quantum well heterostructure (2D electron gas)"""

    def __init__(self, width: float, barrier_height: float,
                 m_well: float = M_ELECTRON, m_barrier: float = M_ELECTRON):
        """
        Initialize quantum well

        Args:
            width: Well width (m)
            barrier_height: Conduction band offset (J)
            m_well: Effective mass in well
            m_barrier: Effective mass in barriers
        """
        self.L = width
        self.V0 = barrier_height
        self.m_well = m_well
        self.m_barrier = m_barrier

    def bound_state_energies(self, n_states: int = 5) -> List[float]:
        """
        Calculate bound state energies

        Returns:
            List of bound state energies (J)
        """
        energies = []

        # Infinite well approximation for search range
        E_inf = HBAR**2 * np.pi**2 / (2 * self.m_well * self.L**2)

        for n in range(1, n_states + 1):
            E_guess = n**2 * E_inf
            if E_guess < self.V0:
                # Refine with finite well transcendental equation
                energies.append(min(E_guess, self.V0 * 0.99))

        return energies

    def subband_dos(self) -> float:
        """
        Calculate 2D density of states (per subband)

        g_2D = m*/(πℏ²)

        Returns:
            DOS (m⁻² J⁻¹)
        """
        return self.m_well / (np.pi * HBAR**2)

    def sheet_density(self, E_F: float, n_subbands: int = 1) -> float:
        """
        Calculate 2D electron sheet density

        Args:
            E_F: Fermi energy
            n_subbands: Number of occupied subbands

        Returns:
            Sheet density (m⁻²)
        """
        energies = self.bound_state_energies(n_subbands)
        g2D = self.subband_dos()

        n_s = 0
        for E_n in energies:
            if E_F > E_n:
                n_s += g2D * (E_F - E_n)

        return n_s


class QuantumDot:
    """Quantum dot (0D confinement)"""

    def __init__(self, size: float, m_eff: float = M_ELECTRON,
                 shape: str = 'spherical'):
        """
        Initialize quantum dot

        Args:
            size: Characteristic size (radius or edge length, m)
            m_eff: Effective mass
            shape: 'spherical' or 'cubic'
        """
        self.size = size
        self.m_eff = m_eff
        self.shape = shape

    def confinement_energy(self, n: int = 1, l: int = 0) -> float:
        """
        Calculate confinement energy

        For spherical: E_nl = ℏ²ξ_nl²/(2m*R²)
        For cubic: E_nml = ℏ²π²(n²+m²+l²)/(2m*L²)

        Args:
            n: Principal quantum number
            l: Angular momentum (for spherical)

        Returns:
            Confinement energy (J)
        """
        if self.shape == 'spherical':
            # Use approximate roots of spherical Bessel functions
            xi_nl = (n + l/2) * np.pi  # Approximation
            return HBAR**2 * xi_nl**2 / (2 * self.m_eff * self.size**2)
        else:  # cubic
            return HBAR**2 * np.pi**2 * (3 * n**2) / (2 * self.m_eff * self.size**2)

    def optical_gap(self, E_bulk_gap: float) -> float:
        """
        Calculate optical gap with confinement

        E_opt = E_g + E_confinement

        Args:
            E_bulk_gap: Bulk band gap (J)

        Returns:
            Optical gap (J)
        """
        return E_bulk_gap + self.confinement_energy(1, 0)

    def charging_energy(self, epsilon_r: float = 10) -> float:
        """
        Calculate single-electron charging energy

        E_c = e²/(2C) ≈ e²/(4πε₀ε_r R)

        Args:
            epsilon_r: Relative permittivity

        Returns:
            Charging energy (J)
        """
        return E_CHARGE**2 / (4 * np.pi * EPSILON_0 * epsilon_r * self.size)


# =============================================================================
# Transport
# =============================================================================

class DrudeModel:
    """Classical Drude model for electrical conductivity"""

    def __init__(self, n: float, m: float = M_ELECTRON, tau: float = 1e-14):
        """
        Initialize Drude model

        Args:
            n: Carrier density (m⁻³)
            m: Carrier effective mass
            tau: Relaxation time (s)
        """
        self.n = n
        self.m = m
        self.tau = tau

    def dc_conductivity(self) -> float:
        """
        Calculate DC conductivity σ = ne²τ/m

        Returns:
            Conductivity (S/m)
        """
        return self.n * E_CHARGE**2 * self.tau / self.m

    def ac_conductivity(self, omega: float) -> complex:
        """
        Calculate AC conductivity σ(ω) = σ_0 / (1 - iωτ)

        Args:
            omega: Angular frequency (rad/s)

        Returns:
            Complex conductivity
        """
        sigma_0 = self.dc_conductivity()
        return sigma_0 / (1 - 1j * omega * self.tau)

    def plasma_frequency(self) -> float:
        """Calculate plasma frequency ω_p = √(ne²/(ε₀m))"""
        return np.sqrt(self.n * E_CHARGE**2 / (EPSILON_0 * self.m))

    def mobility(self) -> float:
        """Calculate carrier mobility μ = eτ/m"""
        return E_CHARGE * self.tau / self.m

    def mean_free_path(self, v_F: float) -> float:
        """
        Calculate mean free path l = v_F τ

        Args:
            v_F: Fermi velocity

        Returns:
            Mean free path (m)
        """
        return v_F * self.tau


class BoltzmannTransport:
    """Boltzmann transport equation for semiclassical transport"""

    def __init__(self, band_func: Callable, tau: float, T: float = 300):
        """
        Initialize Boltzmann transport

        Args:
            band_func: Function E(k) for band dispersion
            tau: Relaxation time (s)
            T: Temperature (K)
        """
        self.E = band_func
        self.tau = tau
        self.T = T

    def group_velocity(self, k: np.ndarray, h: float = 1e-10) -> np.ndarray:
        """Calculate group velocity v = (1/ℏ)∇_k E"""
        v = np.zeros(3)
        for i in range(3):
            k_plus = k.copy(); k_plus[i] += h
            k_minus = k.copy(); k_minus[i] -= h
            v[i] = (self.E(k_plus) - self.E(k_minus)) / (2 * h * HBAR)
        return v

    def fermi_dirac(self, E: float, mu: float) -> float:
        """Calculate Fermi-Dirac distribution"""
        x = (E - mu) / (KB * self.T)
        if x > 100:
            return 0
        if x < -100:
            return 1
        return 1 / (1 + np.exp(x))

    def conductivity_tensor(self, mu: float, bz: BrillouinZone,
                            n_k: int = 20) -> np.ndarray:
        """
        Calculate conductivity tensor σ_ij

        σ_ij = e² ∫ τ v_i v_j (-∂f/∂E) d³k/(2π)³

        Args:
            mu: Chemical potential
            bz: Brillouin zone
            n_k: k-points per dimension

        Returns:
            3×3 conductivity tensor
        """
        sigma = np.zeros((3, 3))
        b1 = bz.reciprocal.b1
        b2 = bz.reciprocal.b2
        b3 = bz.reciprocal.b3
        V_bz = bz.reciprocal.volume()
        dk3 = V_bz / n_k**3

        for i1 in range(n_k):
            for i2 in range(n_k):
                for i3 in range(n_k):
                    k = ((i1/n_k - 0.5) * b1 +
                         (i2/n_k - 0.5) * b2 +
                         (i3/n_k - 0.5) * b3)
                    E = self.E(k)
                    v = self.group_velocity(k)

                    # -∂f/∂E (Fermi window)
                    x = (E - mu) / (KB * self.T)
                    if abs(x) < 30:
                        df_dE = np.exp(x) / (KB * self.T * (1 + np.exp(x))**2)
                    else:
                        df_dE = 0

                    for i in range(3):
                        for j in range(3):
                            sigma[i, j] += self.tau * v[i] * v[j] * df_dE * dk3

        return E_CHARGE**2 * sigma / (2 * np.pi)**3


class HallEffect:
    """Classical Hall effect"""

    def __init__(self, n: float, charge: float = -E_CHARGE):
        """
        Initialize Hall effect

        Args:
            n: Carrier density (m⁻³)
            charge: Carrier charge (negative for electrons)
        """
        self.n = n
        self.q = charge

    def hall_coefficient(self) -> float:
        """
        Calculate Hall coefficient R_H = 1/(nq)

        Returns:
            Hall coefficient (m³/C)
        """
        return 1 / (self.n * self.q)

    def hall_voltage(self, I: float, B: float, t: float) -> float:
        """
        Calculate Hall voltage

        V_H = R_H * I * B / t

        Args:
            I: Current (A)
            B: Magnetic field (T)
            t: Sample thickness (m)

        Returns:
            Hall voltage (V)
        """
        return self.hall_coefficient() * I * B / t

    def hall_mobility(self, sigma: float) -> float:
        """
        Calculate Hall mobility μ_H = |R_H| σ

        Args:
            sigma: Conductivity (S/m)

        Returns:
            Hall mobility (m²/V/s)
        """
        return abs(self.hall_coefficient()) * sigma


class Mobility:
    """Carrier mobility calculations"""

    def __init__(self, m_eff: float = M_ELECTRON, T: float = 300):
        """
        Initialize mobility calculator

        Args:
            m_eff: Effective mass
            T: Temperature (K)
        """
        self.m_eff = m_eff
        self.T = T

    def acoustic_phonon_limited(self, D_ac: float, rho: float,
                                 v_s: float) -> float:
        """
        Calculate acoustic phonon limited mobility

        μ = (2√(2π)eℏ⁴ρv_s²) / (3(m*)^(5/2)(kT)^(3/2)D_ac²)

        Args:
            D_ac: Acoustic deformation potential (J)
            rho: Mass density (kg/m³)
            v_s: Sound velocity (m/s)

        Returns:
            Mobility (m²/V/s)
        """
        numerator = 2 * np.sqrt(2 * np.pi) * E_CHARGE * HBAR**4 * rho * v_s**2
        denominator = 3 * self.m_eff**(5/2) * (KB * self.T)**(3/2) * D_ac**2
        return numerator / denominator

    def ionized_impurity_limited(self, N_i: float, Z: int = 1,
                                  epsilon_r: float = 11.7) -> float:
        """
        Calculate ionized impurity limited mobility (Brooks-Herring)

        Args:
            N_i: Impurity concentration (m⁻³)
            Z: Impurity charge
            epsilon_r: Relative permittivity

        Returns:
            Mobility (m²/V/s)
        """
        epsilon = epsilon_r * EPSILON_0
        v_th = np.sqrt(3 * KB * self.T / self.m_eff)  # Thermal velocity

        # Debye length
        lambda_D = np.sqrt(epsilon * KB * self.T / (N_i * E_CHARGE**2))

        # Brooks-Herring formula (simplified)
        b = 8 * self.m_eff * KB * self.T * lambda_D**2 / HBAR**2
        G = np.log(1 + b) - b / (1 + b)

        return (128 * np.sqrt(2 * np.pi) * epsilon**2 * (KB * self.T)**(3/2)) / \
               (Z**2 * E_CHARGE**3 * np.sqrt(self.m_eff) * N_i * G)


# =============================================================================
# Lattice Dynamics
# =============================================================================

class PhononDispersion:
    """Phonon dispersion relation"""

    def __init__(self, lattice: BravaisLattice, M: float,
                 spring_constant: float):
        """
        Initialize phonon dispersion

        Args:
            lattice: Crystal lattice
            M: Atomic mass (kg)
            spring_constant: Effective spring constant (N/m)
        """
        self.lattice = lattice
        self.M = M
        self.K = spring_constant

    def monatomic_1d(self, k: float) -> float:
        """
        1D monatomic chain dispersion

        ω(k) = 2√(K/M)|sin(ka/2)|

        Args:
            k: Wave vector

        Returns:
            Angular frequency (rad/s)
        """
        a = self.lattice.a
        return 2 * np.sqrt(self.K / self.M) * abs(np.sin(k * a / 2))

    def diatomic_1d(self, k: float, M2: float, K2: float = None) -> Tuple[float, float]:
        """
        1D diatomic chain dispersion (acoustic and optical)

        Args:
            k: Wave vector
            M2: Second atom mass
            K2: Second spring constant (default: same as K)

        Returns:
            (ω_acoustic, ω_optical) frequencies
        """
        if K2 is None:
            K2 = self.K

        a = self.lattice.a
        M1 = self.M

        # Discriminant
        A = self.K / M1 + K2 / M2
        B = np.sqrt((self.K / M1 + K2 / M2)**2 -
                    4 * self.K * K2 * np.sin(k * a)**2 / (M1 * M2))

        omega_optical = np.sqrt(A + B)
        omega_acoustic = np.sqrt(A - B)

        return omega_acoustic, omega_optical

    def debye_frequency(self, v_s: float, n: float) -> float:
        """
        Calculate Debye cutoff frequency

        ω_D = v_s (6π²n)^(1/3)

        Args:
            v_s: Sound velocity (m/s)
            n: Atomic density (m⁻³)

        Returns:
            Debye frequency (rad/s)
        """
        return v_s * (6 * np.pi**2 * n)**(1/3)


class ThermalConductivity:
    """Lattice thermal conductivity"""

    def __init__(self, C_v: float, v_s: float, l_mfp: float):
        """
        Initialize thermal conductivity

        Args:
            C_v: Heat capacity per volume (J/m³/K)
            v_s: Sound velocity (m/s)
            l_mfp: Phonon mean free path (m)
        """
        self.C_v = C_v
        self.v_s = v_s
        self.l_mfp = l_mfp

    def kinetic_theory(self) -> float:
        """
        Calculate thermal conductivity using kinetic theory

        κ = (1/3) C_v v l

        Returns:
            Thermal conductivity (W/m/K)
        """
        return (1/3) * self.C_v * self.v_s * self.l_mfp

    def umklapp_limited(self, T: float, theta_D: float, gamma: float = 2) -> float:
        """
        Estimate Umklapp-limited conductivity

        κ ∝ (M v³/γ²T) exp(θ_D/(3T))

        Args:
            T: Temperature (K)
            theta_D: Debye temperature (K)
            gamma: Grüneisen parameter

        Returns:
            Relative thermal conductivity factor
        """
        return np.exp(theta_D / (3 * T)) / (gamma**2 * T)


# =============================================================================
# Magnetism
# =============================================================================

class Diamagnetism:
    """Diamagnetic response"""

    def __init__(self, n: float, r_squared_avg: float):
        """
        Initialize diamagnetism

        Args:
            n: Electron density (m⁻³)
            r_squared_avg: Average ⟨r²⟩ of electron orbit (m²)
        """
        self.n = n
        self.r2 = r_squared_avg

    def susceptibility(self) -> float:
        """
        Calculate Larmor diamagnetic susceptibility

        χ = -μ₀ n e² ⟨r²⟩ / (6m)

        Returns:
            Magnetic susceptibility (dimensionless)
        """
        from scipy.constants import mu_0
        return -mu_0 * self.n * E_CHARGE**2 * self.r2 / (6 * M_ELECTRON)


class Paramagnetism:
    """Paramagnetic response (Curie law)"""

    def __init__(self, n: float, J: float, g: float = 2.0):
        """
        Initialize paramagnetism

        Args:
            n: Magnetic ion density (m⁻³)
            J: Total angular momentum quantum number
            g: Landé g-factor
        """
        self.n = n
        self.J = J
        self.g = g

    def curie_constant(self) -> float:
        """
        Calculate Curie constant

        C = μ₀ n g² J(J+1) μ_B² / (3k_B)

        Returns:
            Curie constant (K)
        """
        from scipy.constants import mu_0
        return (mu_0 * self.n * self.g**2 * self.J * (self.J + 1) *
                MU_B**2 / (3 * KB))

    def susceptibility(self, T: float) -> float:
        """
        Calculate Curie law susceptibility χ = C/T

        Args:
            T: Temperature (K)

        Returns:
            Susceptibility
        """
        return self.curie_constant() / T

    def brillouin_function(self, x: float) -> float:
        """
        Calculate Brillouin function B_J(x)

        B_J(x) = (2J+1)/(2J) coth((2J+1)x/(2J)) - 1/(2J) coth(x/(2J))

        Args:
            x: Argument (typically gJμ_B B/(k_B T))

        Returns:
            B_J(x)
        """
        J = self.J
        if abs(x) < 1e-10:
            return (J + 1) * x / (3 * J)

        a = (2 * J + 1) / (2 * J)
        b = 1 / (2 * J)
        return a / np.tanh(a * x) - b / np.tanh(b * x)


class Ferromagnetism:
    """Ferromagnetic order"""

    def __init__(self, n: float, J: float, T_c: float, g: float = 2.0):
        """
        Initialize ferromagnetism

        Args:
            n: Magnetic ion density (m⁻³)
            J: Angular momentum
            T_c: Curie temperature (K)
            g: Landé g-factor
        """
        self.n = n
        self.J = J
        self.T_c = T_c
        self.g = g

    def molecular_field_constant(self) -> float:
        """Calculate molecular field constant λ from T_c"""
        return 3 * KB * self.T_c / (self.n * self.g**2 * MU_B**2 *
                                      self.J * (self.J + 1))

    def spontaneous_magnetization(self, T: float, n_iter: int = 100) -> float:
        """
        Calculate spontaneous magnetization M(T) by self-consistent solution

        Args:
            T: Temperature (K)
            n_iter: Number of iterations

        Returns:
            Magnetization (A/m)
        """
        if T >= self.T_c:
            return 0

        M_sat = self.n * self.g * MU_B * self.J
        lam = self.molecular_field_constant()

        # Start with saturation
        M = M_sat * 0.9

        para = Paramagnetism(self.n, self.J, self.g)

        for _ in range(n_iter):
            B_eff = lam * M
            x = self.g * MU_B * self.J * B_eff / (KB * T)
            M_new = M_sat * para.brillouin_function(x)
            if abs(M_new - M) / M_sat < 1e-6:
                break
            M = M_new

        return M

    def critical_exponent_beta(self) -> float:
        """Return mean-field critical exponent β"""
        return 0.5  # Mean-field value


class MagnonDispersion:
    """Spin wave (magnon) dispersion"""

    def __init__(self, J_ex: float, S: float, a: float):
        """
        Initialize magnon dispersion

        Args:
            J_ex: Exchange coupling (J)
            S: Spin quantum number
            a: Lattice constant (m)
        """
        self.J = J_ex
        self.S = S
        self.a = a

    def dispersion_1d(self, k: float) -> float:
        """
        1D ferromagnetic magnon dispersion

        ℏω(k) = 4JS(1 - cos(ka))

        Args:
            k: Wave vector

        Returns:
            Magnon energy (J)
        """
        return 4 * self.J * self.S * (1 - np.cos(k * self.a))

    def dispersion_3d_sc(self, k: np.ndarray) -> float:
        """
        3D simple cubic magnon dispersion

        Args:
            k: Wave vector

        Returns:
            Magnon energy (J)
        """
        return 4 * self.J * self.S * (3 - np.cos(k[0] * self.a) -
                                        np.cos(k[1] * self.a) -
                                        np.cos(k[2] * self.a))

    def stiffness(self) -> float:
        """Calculate spin stiffness D = 2JSa²"""
        return 2 * self.J * self.S * self.a**2


class HysteresisLoop:
    """Magnetic hysteresis loop"""

    def __init__(self, M_s: float, H_c: float, model: str = 'tanh'):
        """
        Initialize hysteresis loop

        Args:
            M_s: Saturation magnetization (A/m)
            H_c: Coercive field (A/m)
            model: 'tanh' or 'linear'
        """
        self.M_s = M_s
        self.H_c = H_c
        self.model = model

    def magnetization(self, H: float, branch: str = 'upper') -> float:
        """
        Calculate magnetization on hysteresis loop

        Args:
            H: Applied field (A/m)
            branch: 'upper' or 'lower' branch

        Returns:
            Magnetization (A/m)
        """
        if self.model == 'tanh':
            if branch == 'upper':
                return self.M_s * np.tanh((H + self.H_c) / self.H_c)
            else:
                return self.M_s * np.tanh((H - self.H_c) / self.H_c)
        else:  # linear
            if H > self.H_c:
                return self.M_s
            elif H < -self.H_c:
                return -self.M_s
            else:
                if branch == 'upper':
                    return self.M_s * H / self.H_c
                else:
                    return self.M_s * H / self.H_c

    def energy_loss_per_cycle(self) -> float:
        """
        Calculate energy loss per hysteresis cycle (approximate)

        Returns:
            Energy per volume per cycle (J/m³)
        """
        # Area ≈ 4 M_s H_c for rectangular loop
        return 4 * self.M_s * self.H_c


# =============================================================================
# Superconductivity
# =============================================================================

class BCSTheory:
    """BCS theory of superconductivity"""

    def __init__(self, N_0: float, V: float, omega_D: float):
        """
        Initialize BCS theory

        Args:
            N_0: Density of states at Fermi level (J⁻¹m⁻³)
            V: Pairing interaction strength (J·m³)
            omega_D: Debye frequency (rad/s)
        """
        self.N_0 = N_0
        self.V = V
        self.omega_D = omega_D
        self.lambda_param = N_0 * V  # Dimensionless coupling

    def critical_temperature(self) -> float:
        """
        Calculate BCS critical temperature

        T_c = 1.13 ℏω_D/k_B exp(-1/λ)

        Returns:
            Critical temperature (K)
        """
        return 1.13 * HBAR * self.omega_D / KB * np.exp(-1 / self.lambda_param)

    def gap_at_zero_T(self) -> float:
        """
        Calculate gap at T=0

        Δ(0) = 2ℏω_D exp(-1/λ)

        Returns:
            Gap energy (J)
        """
        return 2 * HBAR * self.omega_D * np.exp(-1 / self.lambda_param)

    def gap_ratio(self) -> float:
        """Calculate BCS ratio 2Δ(0)/(k_B T_c)"""
        return 2 * self.gap_at_zero_T() / (KB * self.critical_temperature())

    def gap_temperature_dependence(self, T: float) -> float:
        """
        Calculate gap Δ(T) (approximate)

        Δ(T) ≈ Δ(0) tanh(1.74 √(T_c/T - 1))

        Args:
            T: Temperature (K)

        Returns:
            Gap (J)
        """
        T_c = self.critical_temperature()
        if T >= T_c:
            return 0
        Delta_0 = self.gap_at_zero_T()
        return Delta_0 * np.tanh(1.74 * np.sqrt(T_c / T - 1))


class CooperPair:
    """Cooper pair properties"""

    def __init__(self, bcs: BCSTheory, v_F: float):
        """
        Initialize Cooper pair

        Args:
            bcs: BCS theory instance
            v_F: Fermi velocity (m/s)
        """
        self.bcs = bcs
        self.v_F = v_F

    def coherence_length(self, T: float = 0) -> float:
        """
        Calculate BCS coherence length

        ξ₀ = ℏv_F/(πΔ)

        Args:
            T: Temperature (K)

        Returns:
            Coherence length (m)
        """
        Delta = self.bcs.gap_temperature_dependence(T) if T > 0 else self.bcs.gap_at_zero_T()
        if Delta == 0:
            return np.inf
        return HBAR * self.v_F / (np.pi * Delta)

    def pair_size(self) -> float:
        """Estimate Cooper pair size (same as coherence length at T=0)"""
        return self.coherence_length(0)


class MeissnerEffect:
    """Meissner effect and magnetic field expulsion"""

    def __init__(self, lambda_L: float):
        """
        Initialize Meissner effect

        Args:
            lambda_L: London penetration depth (m)
        """
        self.lambda_L = lambda_L

    def penetration_depth(self, T: float, T_c: float) -> float:
        """
        Calculate temperature-dependent penetration depth

        λ(T) = λ(0)/√(1 - (T/T_c)⁴)

        Args:
            T: Temperature (K)
            T_c: Critical temperature (K)

        Returns:
            Penetration depth (m)
        """
        if T >= T_c:
            return np.inf
        return self.lambda_L / np.sqrt(1 - (T / T_c)**4)

    def field_decay(self, x: float, B_0: float) -> float:
        """
        Calculate magnetic field decay into superconductor

        B(x) = B_0 exp(-x/λ)

        Args:
            x: Distance from surface (m)
            B_0: Applied field at surface (T)

        Returns:
            Internal field (T)
        """
        return B_0 * np.exp(-x / self.lambda_L)


class JosephsonJunction:
    """Josephson junction physics"""

    def __init__(self, I_c: float, C: float, R: float = np.inf):
        """
        Initialize Josephson junction

        Args:
            I_c: Critical current (A)
            C: Junction capacitance (F)
            R: Shunt resistance (Ω)
        """
        self.I_c = I_c
        self.C = C
        self.R = R

    def dc_josephson_current(self, phi: float) -> float:
        """
        DC Josephson relation: I = I_c sin(φ)

        Args:
            phi: Phase difference (rad)

        Returns:
            Current (A)
        """
        return self.I_c * np.sin(phi)

    def ac_josephson_frequency(self, V: float) -> float:
        """
        AC Josephson relation: f = 2eV/h

        Args:
            V: Voltage (V)

        Returns:
            Josephson frequency (Hz)
        """
        return 2 * E_CHARGE * V / (2 * np.pi * HBAR)

    def josephson_energy(self, phi: float) -> float:
        """
        Josephson coupling energy

        E_J = (ℏI_c/2e)(1 - cos φ)

        Args:
            phi: Phase difference

        Returns:
            Energy (J)
        """
        E_J = HBAR * self.I_c / (2 * E_CHARGE)
        return E_J * (1 - np.cos(phi))

    def charging_energy(self) -> float:
        """Calculate charging energy E_c = e²/(2C)"""
        return E_CHARGE**2 / (2 * self.C)

    def plasma_frequency(self) -> float:
        """
        Calculate Josephson plasma frequency

        ω_p = √(2eI_c/(ℏC))

        Returns:
            Plasma frequency (rad/s)
        """
        return np.sqrt(2 * E_CHARGE * self.I_c / (HBAR * self.C))


class SQUID:
    """Superconducting Quantum Interference Device"""

    def __init__(self, I_c: float, L: float, area: float):
        """
        Initialize SQUID

        Args:
            I_c: Critical current per junction (A)
            L: Loop inductance (H)
            area: Loop area (m²)
        """
        self.I_c = I_c
        self.L = L
        self.area = area
        self.Phi_0 = 2 * np.pi * HBAR / (2 * E_CHARGE)  # Flux quantum

    def flux_quantum(self) -> float:
        """Return flux quantum Φ₀ = h/(2e)"""
        return self.Phi_0

    def critical_current(self, Phi_ext: float) -> float:
        """
        Calculate maximum critical current as function of flux

        I_max = 2I_c |cos(πΦ/Φ₀)|

        Args:
            Phi_ext: External flux (Wb)

        Returns:
            Maximum supercurrent (A)
        """
        return 2 * self.I_c * abs(np.cos(np.pi * Phi_ext / self.Phi_0))

    def voltage_to_flux(self, V_period: float, f: float) -> float:
        """
        Convert voltage period to flux sensitivity

        Args:
            V_period: Voltage period
            f: Operating frequency

        Returns:
            Flux per voltage
        """
        return self.Phi_0 / V_period


# =============================================================================
# Topological Matter
# =============================================================================

class BerryPhase:
    """Berry phase calculations"""

    def __init__(self, hamiltonian: Callable):
        """
        Initialize Berry phase calculator

        Args:
            hamiltonian: Function H(k) returning Hamiltonian matrix
        """
        self.H = hamiltonian

    def berry_connection(self, k: np.ndarray, band: int,
                         h: float = 1e-6) -> np.ndarray:
        """
        Calculate Berry connection A_n = i⟨n|∇_k|n⟩

        Args:
            k: k-point
            band: Band index
            h: Finite difference step

        Returns:
            Berry connection vector
        """
        A = np.zeros(len(k), dtype=complex)

        # Get eigenstate at k
        _, v = eigh(self.H(k))
        n = v[:, band]

        for i in range(len(k)):
            k_plus = k.copy(); k_plus[i] += h
            k_minus = k.copy(); k_minus[i] -= h

            _, v_plus = eigh(self.H(k_plus))
            _, v_minus = eigh(self.H(k_minus))

            # Derivative of eigenstate
            dn_dk = (v_plus[:, band] - v_minus[:, band]) / (2 * h)

            A[i] = 1j * np.vdot(n, dn_dk)

        return A

    def phase_around_loop(self, k_path: np.ndarray, band: int) -> float:
        """
        Calculate Berry phase around closed loop

        γ = ∮ A·dk

        Args:
            k_path: Array of k-points forming closed loop
            band: Band index

        Returns:
            Berry phase (radians)
        """
        phase = 0 + 0j

        for i in range(len(k_path)):
            k1 = k_path[i]
            k2 = k_path[(i + 1) % len(k_path)]
            dk = k2 - k1

            k_mid = 0.5 * (k1 + k2)
            A = self.berry_connection(k_mid, band)

            phase += np.dot(A, dk)

        return np.real(phase)


class BerryCurvature:
    """Berry curvature (gauge-invariant)"""

    def __init__(self, hamiltonian: Callable):
        """
        Initialize Berry curvature calculator

        Args:
            hamiltonian: Function H(k) returning Hamiltonian matrix
        """
        self.H = hamiltonian

    def curvature(self, k: np.ndarray, band: int, h: float = 1e-5) -> np.ndarray:
        """
        Calculate Berry curvature Ω_n = ∇ × A

        For 2D: Ω = ∂A_y/∂k_x - ∂A_x/∂k_y

        Args:
            k: k-point
            band: Band index
            h: Finite difference step

        Returns:
            Berry curvature
        """
        berry_phase = BerryPhase(self.H)

        if len(k) == 2:
            # 2D case: calculate Ω_z
            k_xp = k.copy(); k_xp[0] += h
            k_xm = k.copy(); k_xm[0] -= h
            k_yp = k.copy(); k_yp[1] += h
            k_ym = k.copy(); k_ym[1] -= h

            A_xp = berry_phase.berry_connection(k_xp, band)
            A_xm = berry_phase.berry_connection(k_xm, band)
            A_yp = berry_phase.berry_connection(k_yp, band)
            A_ym = berry_phase.berry_connection(k_ym, band)

            dAy_dx = (A_yp[1] - A_ym[1]) / (2 * h) if len(A_yp) > 1 else 0
            dAx_dy = (A_xp[0] - A_xm[0]) / (2 * h)

            return np.array([0, 0, np.real(dAy_dx - dAx_dy)])

        return np.zeros(3)


class ChernNumber:
    """Topological Chern number"""

    def __init__(self, hamiltonian: Callable, bz: BrillouinZone):
        """
        Initialize Chern number calculator

        Args:
            hamiltonian: Function H(k) returning Hamiltonian matrix
            bz: Brillouin zone
        """
        self.H = hamiltonian
        self.bz = bz

    def calculate(self, band: int, n_k: int = 30) -> float:
        """
        Calculate Chern number by integrating Berry curvature

        C = (1/2π) ∫_BZ Ω d²k

        Args:
            band: Band index
            n_k: k-points per dimension

        Returns:
            Chern number
        """
        curvature = BerryCurvature(self.H)
        b1, b2 = self.bz.reciprocal.b1[:2], self.bz.reciprocal.b2[:2]

        integral = 0
        dk2 = np.linalg.norm(np.cross(b1, b2)) / n_k**2

        for i1 in range(n_k):
            for i2 in range(n_k):
                k = (i1/n_k - 0.5) * b1 + (i2/n_k - 0.5) * b2
                Omega = curvature.curvature(k, band)
                integral += Omega[2] * dk2

        return integral / (2 * np.pi)


class TopologicalInsulator2D:
    """2D topological insulator (quantum spin Hall)"""

    def __init__(self, M: float, B: float, A: float, C: float = 0, D: float = 0):
        """
        Initialize BHZ model for 2D TI

        H(k) = [M - B(k_x² + k_y²)]τ_z + A(k_x σ_x τ_z + k_y σ_y) + ...

        Args:
            M: Mass parameter
            B: Quadratic term
            A: Linear (Dirac) term
            C: Constant energy shift
            D: Particle-hole asymmetry
        """
        self.M = M
        self.B = B
        self.A = A
        self.C = C
        self.D = D

    def hamiltonian(self, k: np.ndarray) -> np.ndarray:
        """
        Calculate BHZ Hamiltonian

        Args:
            k: k-point (2D)

        Returns:
            4×4 Hamiltonian matrix
        """
        kx, ky = k[0], k[1]
        k2 = kx**2 + ky**2

        # Pauli matrices
        sigma_x = np.array([[0, 1], [1, 0]])
        sigma_y = np.array([[0, -1j], [1j, 0]])
        sigma_z = np.array([[1, 0], [0, -1]])
        sigma_0 = np.eye(2)

        # Tau matrices (orbital)
        tau_z = np.array([[1, 0], [0, -1]])
        tau_0 = np.eye(2)

        # Mass term
        mass = (self.M - self.B * k2)

        # Full Hamiltonian
        H = (self.C - self.D * k2) * np.kron(tau_0, sigma_0)
        H += mass * np.kron(tau_z, sigma_0)
        H += self.A * kx * np.kron(tau_z, sigma_x)
        H += self.A * ky * np.kron(tau_0, sigma_y)

        return H

    def is_topological(self) -> bool:
        """Check if in topological phase (M/B > 0)"""
        return self.M * self.B > 0

    def edge_velocity(self) -> float:
        """Calculate edge state velocity v = A/ℏ"""
        return self.A / HBAR


class IntegerQuantumHall:
    """Integer quantum Hall effect"""

    def __init__(self, n_landau: int, B: float):
        """
        Initialize IQHE

        Args:
            n_landau: Landau level filling
            B: Magnetic field (T)
        """
        self.n = n_landau
        self.B = B

    def hall_conductance(self) -> float:
        """
        Calculate quantized Hall conductance

        σ_xy = ν e²/h

        Returns:
            Hall conductance (S)
        """
        return self.n * E_CHARGE**2 / (2 * np.pi * HBAR)

    def landau_level_energy(self, n: int) -> float:
        """
        Calculate Landau level energy

        E_n = ℏω_c(n + 1/2)

        Args:
            n: Landau level index

        Returns:
            Energy (J)
        """
        omega_c = E_CHARGE * self.B / M_ELECTRON  # Cyclotron frequency
        return HBAR * omega_c * (n + 0.5)

    def magnetic_length(self) -> float:
        """Calculate magnetic length l_B = √(ℏ/(eB))"""
        return np.sqrt(HBAR / (E_CHARGE * self.B))

    def filling_factor(self, n_2d: float) -> float:
        """
        Calculate filling factor ν = n_2D h/(eB)

        Args:
            n_2d: 2D electron density (m⁻²)

        Returns:
            Filling factor
        """
        return n_2d * 2 * np.pi * HBAR / (E_CHARGE * self.B)
