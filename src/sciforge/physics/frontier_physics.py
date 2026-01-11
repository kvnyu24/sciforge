"""
Frontier Physics module

This module implements pedagogical versions of:
- String theory basics
- Holography and AdS/CFT
- Quantum gravity concepts
"""

import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, Callable, Dict, List, Tuple, Union
from dataclasses import dataclass
from scipy import special, integrate


class ClassicalString:
    """
    Classical relativistic string.

    Implements the Nambu-Goto action for a relativistic string:
    S = -T ∫ dτ dσ √(-det(h_αβ))

    where h_αβ = ∂_α X^μ ∂_β X_μ is the induced metric.

    Args:
        tension: String tension T = 1/(2πα')
        length: String length (proper length at rest)
    """

    def __init__(self, tension: float = 1.0, length: float = 1.0):
        self.T = tension
        self.L = length
        self.alpha_prime = 1 / (2 * np.pi * tension)
        self._history = {'time': [], 'energy': []}

    def string_scale(self) -> float:
        """
        String length scale l_s = √α'.

        Returns:
            String length in natural units
        """
        return np.sqrt(self.alpha_prime)

    def energy_at_rest(self) -> float:
        """
        Rest energy of string.

        E = T * L

        Returns:
            String rest energy
        """
        return self.T * self.L

    def mode_frequency(self, n: int) -> float:
        """
        Frequency of n-th oscillation mode.

        ω_n = n π / L  (for open string with fixed ends)

        Args:
            n: Mode number (n > 0)

        Returns:
            Angular frequency of mode
        """
        if n <= 0:
            raise ValueError("Mode number must be positive")
        return n * np.pi / self.L

    def virasoro_constraint(self, alpha_n: ArrayLike, alpha_bar_n: ArrayLike) -> float:
        """
        Virasoro constraint L_0 - a = 0.

        Args:
            alpha_n: Left-moving oscillator amplitudes
            alpha_bar_n: Right-moving oscillator amplitudes

        Returns:
            Constraint violation (should be zero for physical states)
        """
        alpha_n = np.asarray(alpha_n)
        alpha_bar_n = np.asarray(alpha_bar_n)

        # L_0 = α' p²/4 + Σ n |α_n|²
        L_0 = np.sum(np.arange(1, len(alpha_n) + 1) * np.abs(alpha_n)**2)
        L_0_bar = np.sum(np.arange(1, len(alpha_bar_n) + 1) * np.abs(alpha_bar_n)**2)

        # Normal ordering constant a = 1 for bosonic string in 26D
        a = 1.0

        return L_0 + L_0_bar - 2 * a

    def classical_trajectory(
        self,
        times: ArrayLike,
        initial_shape: Optional[Callable] = None
    ) -> np.ndarray:
        """
        Compute classical string trajectory.

        Args:
            times: Time points
            initial_shape: Function σ -> X(σ, 0) for initial shape

        Returns:
            Array of shape (n_times, n_sigma, d) with string positions
        """
        times = np.asarray(times)
        n_sigma = 50
        sigma = np.linspace(0, self.L, n_sigma)

        if initial_shape is None:
            # Default: straight string along x-axis
            def initial_shape(s):
                return np.array([s, 0, 0])

        # Simple wave equation evolution
        d = 3
        X = np.zeros((len(times), n_sigma, d))

        # Initial condition
        X0 = np.array([initial_shape(s) for s in sigma])

        # Wave speed = 1 for relativistic string
        c = 1.0

        for i, t in enumerate(times):
            # D'Alembert solution for wave equation
            for j, s in enumerate(sigma):
                # Standing wave pattern
                X[i, j] = X0[j] * np.cos(np.pi * t / self.L)

        return X


class StringSpectrum:
    """
    String vibration spectrum.

    Computes the mass spectrum of string states.

    Args:
        alpha_prime: Regge slope α'
        dimension: Spacetime dimension D
    """

    def __init__(self, alpha_prime: float = 1.0, dimension: int = 26):
        self.alpha_prime = alpha_prime
        self.D = dimension

        # Normal ordering constant
        # a = (D-2)/24 for bosonic string
        self.a = (dimension - 2) / 24

    def mass_squared(self, N: int, N_bar: Optional[int] = None) -> float:
        """
        Mass squared of string state.

        For open string: M² = (N - a) / α'
        For closed string: M² = 4(N - a) / α' with N = N̄

        Args:
            N: Left-moving oscillator number
            N_bar: Right-moving oscillator number (closed string)

        Returns:
            Mass squared in string units
        """
        if N_bar is None:
            # Open string
            return (N - self.a) / self.alpha_prime
        else:
            # Closed string (level matching: N = N_bar)
            if N != N_bar:
                raise ValueError("Level matching requires N = N_bar")
            return 4 * (N - self.a) / self.alpha_prime

    def tachyon_mass_squared(self) -> float:
        """
        Tachyon mass squared (N = 0 state).

        M² = -a/α' < 0 indicates instability

        Returns:
            Tachyon mass squared
        """
        return self.mass_squared(0)

    def is_tachyon_free(self) -> bool:
        """Check if spectrum is tachyon-free."""
        return self.tachyon_mass_squared() >= 0

    def massless_states(self) -> int:
        """
        Number of oscillator excitations for massless states.

        M² = 0 requires N = a

        Returns:
            N value for massless states
        """
        return int(round(self.a))

    def regge_trajectory(self, J: int) -> float:
        """
        Regge trajectory: J = α' M² + a.

        Relation between spin and mass.

        Args:
            J: Spin (angular momentum)

        Returns:
            Mass squared
        """
        return (J - self.a) / self.alpha_prime

    def degeneracy(self, N: int) -> int:
        """
        Estimate degeneracy of level N.

        d(N) ~ N^{-α} exp(β√N) for large N

        Args:
            N: Oscillator level

        Returns:
            Approximate degeneracy
        """
        if N <= 0:
            return 1

        # Leading Hagedorn behavior
        # d(N) ≈ N^{-(D+1)/4} exp(4π√(N(D-2)/24))
        alpha = (self.D + 1) / 4
        beta = 4 * np.pi * np.sqrt((self.D - 2) / 24)

        return int(N**(-alpha) * np.exp(beta * np.sqrt(N)))

    def hagedorn_temperature(self) -> float:
        """
        Hagedorn temperature.

        T_H = 1/(4π√(α'(D-2)/24))

        Returns:
            Hagedorn temperature
        """
        return 1 / (4 * np.pi * np.sqrt(self.alpha_prime * (self.D - 2) / 24))


class CompactDimension:
    """
    Kaluza-Klein compactification.

    Models extra dimensions compactified on circles or tori.

    Args:
        radius: Compactification radius R
        n_compact: Number of compact dimensions
    """

    def __init__(self, radius: float, n_compact: int = 1):
        self.R = radius
        self.n_compact = n_compact

    def kk_mass(self, n: int) -> float:
        """
        Kaluza-Klein mass for mode n.

        m_n = |n| / R

        Args:
            n: KK mode number

        Returns:
            KK mass
        """
        return abs(n) / self.R

    def kk_spectrum(self, n_max: int = 10) -> np.ndarray:
        """
        KK mass spectrum up to mode n_max.

        Args:
            n_max: Maximum mode number

        Returns:
            Array of KK masses
        """
        return np.array([self.kk_mass(n) for n in range(n_max + 1)])

    def winding_energy(self, w: int, alpha_prime: float = 1.0) -> float:
        """
        String winding mode energy.

        E_w = |w| R / α'

        Args:
            w: Winding number
            alpha_prime: String parameter

        Returns:
            Winding mode energy
        """
        return abs(w) * self.R / alpha_prime

    def t_duality_radius(self, alpha_prime: float = 1.0) -> float:
        """
        T-dual radius R' = α'/R.

        Args:
            alpha_prime: String parameter

        Returns:
            T-dual radius
        """
        return alpha_prime / self.R

    def self_dual_radius(self, alpha_prime: float = 1.0) -> float:
        """
        Self-dual radius where R = R'.

        R_* = √α'

        Args:
            alpha_prime: String parameter

        Returns:
            Self-dual radius
        """
        return np.sqrt(alpha_prime)

    def effective_4d_coupling(
        self,
        higher_d_coupling: float,
        volume: Optional[float] = None
    ) -> float:
        """
        4D effective coupling from higher dimensions.

        g_4² = g_D² / V_compact

        Args:
            higher_d_coupling: Higher-dimensional coupling
            volume: Compact volume (default: (2πR)^n)

        Returns:
            4D effective coupling squared
        """
        if volume is None:
            volume = (2 * np.pi * self.R) ** self.n_compact

        return higher_d_coupling**2 / volume


class AdSMetric:
    """
    Anti-de Sitter spacetime metric.

    ds² = (L/z)²(-dt² + dx² + dz²)  (Poincaré patch)

    Args:
        ads_radius: AdS radius L
        dimension: Spacetime dimension d
    """

    def __init__(self, ads_radius: float = 1.0, dimension: int = 5):
        self.L = ads_radius
        self.d = dimension

    def metric_component(self, z: float) -> float:
        """
        Metric warp factor (L/z)².

        Args:
            z: Radial coordinate (z > 0)

        Returns:
            Warp factor
        """
        if z <= 0:
            raise ValueError("z must be positive")
        return (self.L / z)**2

    def proper_distance(self, z1: float, z2: float) -> float:
        """
        Proper distance between two points at same (t, x).

        d(z1, z2) = L |ln(z2/z1)|

        Args:
            z1, z2: Radial coordinates

        Returns:
            Proper distance
        """
        return self.L * abs(np.log(z2 / z1))

    def geodesic_length(
        self,
        x1: ArrayLike,
        x2: ArrayLike,
        z_boundary: float = 1e-10
    ) -> float:
        """
        Spacelike geodesic length in AdS.

        For boundary points at z → 0 separated by Δx:
        L_geo ≈ 2L ln(|Δx|/ε) + ...

        Args:
            x1, x2: Boundary coordinates
            z_boundary: UV cutoff

        Returns:
            Regularized geodesic length
        """
        x1, x2 = np.asarray(x1), np.asarray(x2)
        delta_x = np.linalg.norm(x2 - x1)

        # Minimal surface dips to z_* = Δx/2
        z_star = delta_x / 2

        # Geodesic length
        return 2 * self.L * np.log(delta_x / z_boundary)

    def cosmological_constant(self) -> float:
        """
        Cosmological constant for AdS.

        Λ = -(d-1)(d-2) / (2L²)

        Returns:
            Negative cosmological constant
        """
        return -(self.d - 1) * (self.d - 2) / (2 * self.L**2)

    def boundary_dimension(self) -> int:
        """Dimension of conformal boundary."""
        return self.d - 1

    def uv_cutoff_to_energy(self, z_uv: float) -> float:
        """
        UV cutoff z to energy scale E.

        E ~ 1/z

        Args:
            z_uv: UV cutoff in z

        Returns:
            Energy scale
        """
        return 1 / z_uv


class CFTCorrelator:
    """
    Conformal field theory correlators.

    Implements 2, 3, and 4-point functions in CFT.

    Args:
        dimension: Spacetime dimension d
    """

    def __init__(self, dimension: int = 4):
        self.d = dimension

    def two_point(
        self,
        x1: ArrayLike,
        x2: ArrayLike,
        delta: float,
        normalization: float = 1.0
    ) -> float:
        """
        Two-point function ⟨O(x1) O(x2)⟩.

        ⟨O(x1) O(x2)⟩ = C / |x1 - x2|^{2Δ}

        Args:
            x1, x2: Position vectors
            delta: Scaling dimension
            normalization: Overall coefficient C

        Returns:
            Two-point function value
        """
        x1, x2 = np.asarray(x1), np.asarray(x2)
        r = np.linalg.norm(x2 - x1)

        if r < 1e-15:
            return float('inf')

        return normalization / r**(2 * delta)

    def three_point(
        self,
        x1: ArrayLike,
        x2: ArrayLike,
        x3: ArrayLike,
        delta1: float,
        delta2: float,
        delta3: float,
        C123: float = 1.0
    ) -> float:
        """
        Three-point function ⟨O1(x1) O2(x2) O3(x3)⟩.

        Args:
            x1, x2, x3: Position vectors
            delta1, delta2, delta3: Scaling dimensions
            C123: OPE coefficient

        Returns:
            Three-point function value
        """
        x1 = np.asarray(x1)
        x2 = np.asarray(x2)
        x3 = np.asarray(x3)

        r12 = np.linalg.norm(x2 - x1)
        r23 = np.linalg.norm(x3 - x2)
        r13 = np.linalg.norm(x3 - x1)

        if min(r12, r23, r13) < 1e-15:
            return float('inf')

        # Exponents from conformal invariance
        e12 = delta1 + delta2 - delta3
        e23 = delta2 + delta3 - delta1
        e13 = delta1 + delta3 - delta2

        return C123 / (r12**e12 * r23**e23 * r13**e13)

    def cross_ratio(
        self,
        x1: ArrayLike,
        x2: ArrayLike,
        x3: ArrayLike,
        x4: ArrayLike
    ) -> Tuple[float, float]:
        """
        Compute conformal cross ratios u and v.

        u = (x12² x34²) / (x13² x24²)
        v = (x14² x23²) / (x13² x24²)

        Args:
            x1, x2, x3, x4: Position vectors

        Returns:
            Tuple (u, v)
        """
        x1, x2 = np.asarray(x1), np.asarray(x2)
        x3, x4 = np.asarray(x3), np.asarray(x4)

        x12_sq = np.sum((x1 - x2)**2)
        x34_sq = np.sum((x3 - x4)**2)
        x13_sq = np.sum((x1 - x3)**2)
        x24_sq = np.sum((x2 - x4)**2)
        x14_sq = np.sum((x1 - x4)**2)
        x23_sq = np.sum((x2 - x3)**2)

        u = (x12_sq * x34_sq) / (x13_sq * x24_sq)
        v = (x14_sq * x23_sq) / (x13_sq * x24_sq)

        return u, v

    def conformal_block_scalar(
        self,
        u: float,
        v: float,
        delta: float,
        delta_ext: float
    ) -> float:
        """
        Scalar conformal block (approximate).

        g_Δ(u, v) for exchange of operator with dimension Δ

        Args:
            u, v: Cross ratios
            delta: Exchanged operator dimension
            delta_ext: External operator dimension

        Returns:
            Conformal block value
        """
        # Leading term in small u expansion
        # g_Δ(u, v) ≈ u^{Δ/2} (1 + O(u, 1-v))
        return u**(delta / 2)


class HolographicEntropy:
    """
    Holographic entanglement entropy via RT formula.

    S_A = Area(γ_A) / (4 G_N)

    Args:
        ads_radius: AdS radius L
        newton_constant: Newton constant G_N
    """

    def __init__(self, ads_radius: float = 1.0, newton_constant: float = 1.0):
        self.L = ads_radius
        self.G_N = newton_constant

    def rt_formula(self, minimal_area: float) -> float:
        """
        Ryu-Takayanagi formula.

        S = Area / (4 G_N)

        Args:
            minimal_area: Area of minimal surface

        Returns:
            Entanglement entropy
        """
        return minimal_area / (4 * self.G_N)

    def interval_entropy_2d(self, interval_length: float, uv_cutoff: float) -> float:
        """
        Entanglement entropy for interval in 2D CFT.

        S = (c/3) ln(l/ε)

        where c = 3L/(2G_N) is the central charge.

        Args:
            interval_length: Length l of interval
            uv_cutoff: UV cutoff ε

        Returns:
            Entanglement entropy
        """
        c = 3 * self.L / (2 * self.G_N)
        return (c / 3) * np.log(interval_length / uv_cutoff)

    def ball_entropy_higher_d(
        self,
        radius: float,
        boundary_dim: int,
        uv_cutoff: float
    ) -> float:
        """
        Entanglement entropy for ball region in higher dimensions.

        S = a_{d-2} (R/ε)^{d-2} + ... + (-1)^{d/2} a_0 ln(R/ε) + ...

        Args:
            radius: Ball radius R
            boundary_dim: Boundary dimension d
            uv_cutoff: UV cutoff ε

        Returns:
            Leading contribution to entropy
        """
        # Area law term
        area_coeff = self.L**(boundary_dim - 1) / (4 * self.G_N)

        # Leading divergence
        return area_coeff * (radius / uv_cutoff)**(boundary_dim - 2)

    def mutual_information(
        self,
        entropy_A: float,
        entropy_B: float,
        entropy_AB: float
    ) -> float:
        """
        Holographic mutual information.

        I(A:B) = S_A + S_B - S_{A∪B}

        Args:
            entropy_A: Entropy of region A
            entropy_B: Entropy of region B
            entropy_AB: Entropy of union

        Returns:
            Mutual information
        """
        return entropy_A + entropy_B - entropy_AB

    def strong_subadditivity(
        self,
        S_A: float,
        S_B: float,
        S_AB: float,
        S_BC: float,
        S_B_only: float,
        S_ABC: float
    ) -> bool:
        """
        Check strong subadditivity.

        S_{AB} + S_{BC} ≥ S_B + S_{ABC}

        Returns:
            True if satisfied
        """
        return S_AB + S_BC >= S_B_only + S_ABC


class PlanckScale:
    """
    Planck scale physics.

    Natural units where quantum gravity effects become important.
    """

    # Fundamental constants (SI units)
    HBAR = 1.054571817e-34  # J·s
    C = 299792458  # m/s
    G_N = 6.67430e-11  # m³/(kg·s²)
    K_B = 1.380649e-23  # J/K

    def __init__(self):
        self._compute_scales()

    def _compute_scales(self):
        """Compute Planck scales from fundamental constants."""
        hbar, c, G = self.HBAR, self.C, self.G_N

        self.length = np.sqrt(hbar * G / c**3)  # Planck length
        self.time = np.sqrt(hbar * G / c**5)    # Planck time
        self.mass = np.sqrt(hbar * c / G)       # Planck mass
        self.energy = np.sqrt(hbar * c**5 / G)  # Planck energy
        self.temperature = np.sqrt(hbar * c**5 / (G * self.K_B**2))

    def planck_length(self) -> float:
        """
        Planck length l_P = √(ℏG/c³) ≈ 1.6×10⁻³⁵ m

        Returns:
            Planck length in meters
        """
        return self.length

    def planck_time(self) -> float:
        """
        Planck time t_P = √(ℏG/c⁵) ≈ 5.4×10⁻⁴⁴ s

        Returns:
            Planck time in seconds
        """
        return self.time

    def planck_mass(self) -> float:
        """
        Planck mass m_P = √(ℏc/G) ≈ 2.2×10⁻⁸ kg

        Returns:
            Planck mass in kg
        """
        return self.mass

    def planck_energy(self) -> float:
        """
        Planck energy E_P = √(ℏc⁵/G) ≈ 1.96×10⁹ J

        Returns:
            Planck energy in Joules
        """
        return self.energy

    def planck_temperature(self) -> float:
        """
        Planck temperature T_P = √(ℏc⁵/(Gk_B²)) ≈ 1.4×10³² K

        Returns:
            Planck temperature in Kelvin
        """
        return self.temperature

    def energy_to_planck(self, energy_joules: float) -> float:
        """Convert energy to Planck units."""
        return energy_joules / self.energy

    def length_to_planck(self, length_meters: float) -> float:
        """Convert length to Planck units."""
        return length_meters / self.length

    def schwarzschild_radius(self, mass: float) -> float:
        """
        Schwarzschild radius of mass.

        r_s = 2GM/c²

        Args:
            mass: Mass in kg

        Returns:
            Schwarzschild radius in meters
        """
        return 2 * self.G_N * mass / self.C**2

    def compton_wavelength(self, mass: float) -> float:
        """
        Compton wavelength of mass.

        λ_C = ℏ/(mc)

        Args:
            mass: Mass in kg

        Returns:
            Compton wavelength in meters
        """
        return self.HBAR / (mass * self.C)

    def quantum_gravity_scale(self, mass: float) -> str:
        """
        Determine which physics applies.

        Compare Schwarzschild radius to Compton wavelength.

        Args:
            mass: Mass in kg

        Returns:
            Description of applicable physics
        """
        r_s = self.schwarzschild_radius(mass)
        lambda_c = self.compton_wavelength(mass)

        if r_s > lambda_c:
            return "Classical gravity (black hole)"
        elif r_s < lambda_c:
            return "Quantum mechanics"
        else:
            return "Quantum gravity (Planck mass)"


class BlackHoleEntropy:
    """
    Black hole thermodynamics.

    Implements Bekenstein-Hawking entropy and related quantities.

    Args:
        mass: Black hole mass in kg
    """

    # Constants
    HBAR = 1.054571817e-34
    C = 299792458
    G_N = 6.67430e-11
    K_B = 1.380649e-23

    def __init__(self, mass: float):
        self.M = mass
        self.r_s = 2 * self.G_N * mass / self.C**2

    def horizon_area(self) -> float:
        """
        Event horizon area.

        A = 4π r_s² = 16π G² M² / c⁴

        Returns:
            Horizon area in m²
        """
        return 4 * np.pi * self.r_s**2

    def bekenstein_hawking_entropy(self) -> float:
        """
        Bekenstein-Hawking entropy.

        S = k_B A / (4 l_P²) = k_B c³ A / (4 ℏ G)

        Returns:
            Entropy in J/K
        """
        A = self.horizon_area()
        l_P_sq = self.HBAR * self.G_N / self.C**3
        return self.K_B * A / (4 * l_P_sq)

    def entropy_in_bits(self) -> float:
        """
        Entropy in bits.

        S_bits = A / (4 l_P² ln 2)

        Returns:
            Entropy in bits
        """
        A = self.horizon_area()
        l_P_sq = self.HBAR * self.G_N / self.C**3
        return A / (4 * l_P_sq * np.log(2))

    def hawking_temperature(self) -> float:
        """
        Hawking temperature.

        T_H = ℏ c³ / (8π G M k_B)

        Returns:
            Temperature in Kelvin
        """
        return (self.HBAR * self.C**3) / (8 * np.pi * self.G_N * self.M * self.K_B)

    def evaporation_time(self) -> float:
        """
        Black hole evaporation time.

        t_evap = 5120 π G² M³ / (ℏ c⁴)

        Returns:
            Evaporation time in seconds
        """
        return (5120 * np.pi * self.G_N**2 * self.M**3) / (self.HBAR * self.C**4)

    def luminosity(self) -> float:
        """
        Hawking radiation luminosity.

        L = ℏ c⁶ / (15360 π G² M²)

        Returns:
            Luminosity in Watts
        """
        return (self.HBAR * self.C**6) / (15360 * np.pi * self.G_N**2 * self.M**2)

    def first_law(self, delta_M: float) -> float:
        """
        First law of black hole thermodynamics.

        dM = T dS / c²

        Args:
            delta_M: Change in mass

        Returns:
            Change in entropy
        """
        T = self.hawking_temperature()
        return delta_M * self.C**2 / T

    @staticmethod
    def minimum_mass_for_stability(temperature: float) -> float:
        """
        Minimum black hole mass to be colder than environment.

        Args:
            temperature: Environment temperature in K

        Returns:
            Minimum mass in kg
        """
        hbar = BlackHoleEntropy.HBAR
        c = BlackHoleEntropy.C
        G = BlackHoleEntropy.G_N
        k_B = BlackHoleEntropy.K_B

        return (hbar * c**3) / (8 * np.pi * G * k_B * temperature)


class InformationParadox:
    """
    Black hole information paradox analysis.

    Implements Page curve and related concepts.
    """

    def __init__(self, initial_entropy: float):
        """
        Args:
            initial_entropy: Initial black hole entropy S_0
        """
        self.S_0 = initial_entropy
        self._history = {'time': [], 'S_BH': [], 'S_rad': []}

    def page_time(self) -> float:
        """
        Page time: when radiation entropy equals black hole entropy.

        t_Page ≈ t_evap / 2

        Returns:
            Normalized Page time (fraction of evaporation)
        """
        return 0.5

    def page_curve(self, times: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Page curve for entanglement entropy.

        Early time: S increases (Hawking calculation)
        Late time: S decreases (unitarity)

        Args:
            times: Normalized time points (0 to 1)

        Returns:
            Tuple of (S_radiation, S_black_hole)
        """
        times = np.asarray(times)

        # Black hole entropy decreases linearly
        S_BH = self.S_0 * (1 - times)

        # Radiation entropy from Page curve
        S_rad = np.zeros_like(times)
        t_page = self.page_time()

        early = times <= t_page
        late = times > t_page

        # Early: follows Hawking (increases)
        S_rad[early] = self.S_0 * times[early] / t_page * t_page

        # Late: follows Page (decreases to zero)
        S_rad[late] = self.S_0 * (1 - times[late])

        return S_rad, S_BH

    def scrambling_time(self, S: float) -> float:
        """
        Scrambling time for information to spread.

        t_scr ~ β ln(S)

        Args:
            S: Black hole entropy

        Returns:
            Scrambling time (in units of β = 1/T)
        """
        return np.log(S)

    def hayden_preskill_time(self, k: int, S: float) -> float:
        """
        Time to recover k qubits after Page time.

        t_HP ~ k + ln(S)

        Args:
            k: Number of qubits to recover
            S: Black hole entropy

        Returns:
            Recovery time
        """
        return k + np.log(S)

    def firewall_paradox_tension(self) -> Dict[str, str]:
        """
        Describe the AMPS firewall paradox.

        Returns:
            Dictionary describing the paradox
        """
        return {
            'assumption_1': 'Unitarity: Information is preserved',
            'assumption_2': 'No drama: Nothing special at horizon',
            'assumption_3': 'EFT valid: Low-energy physics works',
            'assumption_4': 'Locality: No superluminal signaling',
            'paradox': 'These four assumptions are mutually inconsistent',
            'resolution_attempts': [
                'Firewall (AMPS): Give up no drama',
                'ER=EPR: Modify locality/geometry',
                'Complementarity: Observer-dependent',
                'Fuzzballs: Modify horizon structure'
            ]
        }

    def entanglement_wedge(
        self,
        radiation_entropy: float,
        black_hole_entropy: float
    ) -> str:
        """
        Determine entanglement wedge.

        Before Page time: Radiation wedge is small
        After Page time: Radiation wedge includes interior

        Args:
            radiation_entropy: Current radiation entropy
            black_hole_entropy: Current black hole entropy

        Returns:
            Description of wedge
        """
        if radiation_entropy < black_hole_entropy:
            return "Radiation wedge excludes interior"
        else:
            return "Radiation wedge includes interior (island)"


__all__ = [
    'ClassicalString',
    'StringSpectrum',
    'CompactDimension',
    'AdSMetric',
    'CFTCorrelator',
    'HolographicEntropy',
    'PlanckScale',
    'BlackHoleEntropy',
    'InformationParadox',
]
