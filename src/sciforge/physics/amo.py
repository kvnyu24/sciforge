"""
Atomic, Molecular, and Optical (AMO) Physics Module

This module provides tools for AMO physics including:
- Atomic Structure: Hydrogen atom, multi-electron atoms, coupling schemes
- Atom-Light Interaction: Rabi oscillations, Bloch equations, transitions
- Laser Physics: Cavities, gain media, mode locking
- Laser Cooling: Doppler cooling, MOT, optical lattices
- Ultracold Atoms: BEC, Fermi gases, quantum simulation
- Molecular Physics: Molecular orbitals, spectroscopy
"""

import numpy as np
from typing import Optional, Callable, Tuple, List, Dict, Union
from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp, quad
from scipy.special import sph_harm, factorial, genlaguerre, assoc_laguerre
from scipy.linalg import eigh, expm
import warnings


# =============================================================================
# Physical Constants
# =============================================================================

HBAR = 1.054571817e-34  # Reduced Planck constant (J·s)
KB = 1.380649e-23  # Boltzmann constant (J/K)
E_CHARGE = 1.602176634e-19  # Elementary charge (C)
M_ELECTRON = 9.1093837015e-31  # Electron mass (kg)
M_PROTON = 1.67262192369e-27  # Proton mass (kg)
EPSILON_0 = 8.8541878128e-12  # Vacuum permittivity (F/m)
C = 299792458.0  # Speed of light (m/s)
A_BOHR = 5.29177210903e-11  # Bohr radius (m)
E_HARTREE = 4.3597447222071e-18  # Hartree energy (J)
ALPHA_FS = 1/137.035999084  # Fine structure constant


# =============================================================================
# Atomic Structure
# =============================================================================

class HydrogenAtom:
    """Full hydrogen atom wavefunctions and energies"""

    def __init__(self, Z: int = 1):
        """
        Initialize hydrogen-like atom

        Args:
            Z: Nuclear charge
        """
        self.Z = Z
        self.a0 = A_BOHR / Z

    def energy(self, n: int) -> float:
        """
        Calculate energy level

        E_n = -13.6 eV × Z²/n²

        Args:
            n: Principal quantum number

        Returns:
            Energy (J)
        """
        return -E_HARTREE * self.Z**2 / (2 * n**2)

    def radial_wavefunction(self, n: int, l: int, r: float) -> float:
        """
        Calculate radial wavefunction R_nl(r)

        Args:
            n: Principal quantum number
            l: Orbital angular momentum
            r: Radial distance (m)

        Returns:
            R_nl(r)
        """
        if l >= n or l < 0:
            return 0

        rho = 2 * r / (n * self.a0)
        norm = np.sqrt((2/(n*self.a0))**3 * factorial(n-l-1) /
                       (2*n*factorial(n+l)))

        # Associated Laguerre polynomial
        L = assoc_laguerre(rho, n-l-1, 2*l+1)

        return norm * np.exp(-rho/2) * rho**l * L

    def angular_wavefunction(self, l: int, m: int, theta: float, phi: float) -> complex:
        """
        Calculate angular wavefunction Y_lm(θ,φ)

        Args:
            l: Orbital angular momentum
            m: Magnetic quantum number
            theta: Polar angle
            phi: Azimuthal angle

        Returns:
            Y_lm(θ,φ)
        """
        return sph_harm(m, l, phi, theta)

    def wavefunction(self, n: int, l: int, m: int,
                     r: float, theta: float, phi: float) -> complex:
        """
        Calculate full wavefunction ψ_nlm(r,θ,φ)

        Args:
            n, l, m: Quantum numbers
            r, theta, phi: Spherical coordinates

        Returns:
            ψ_nlm
        """
        R = self.radial_wavefunction(n, l, r)
        Y = self.angular_wavefunction(l, m, theta, phi)
        return R * Y

    def probability_density(self, n: int, l: int, m: int,
                           r: float, theta: float) -> float:
        """
        Calculate radial probability density P(r) = r²|R(r)|²

        Args:
            n, l: Quantum numbers
            r: Radial distance
            theta: Polar angle (for angular dependence)

        Returns:
            Probability density
        """
        R = self.radial_wavefunction(n, l, r)
        return r**2 * R**2

    def expectation_r(self, n: int, l: int) -> float:
        """
        Calculate ⟨r⟩ for state |nl⟩

        ⟨r⟩ = (a₀/2)[3n² - l(l+1)]

        Args:
            n, l: Quantum numbers

        Returns:
            ⟨r⟩ (m)
        """
        return (self.a0 / 2) * (3 * n**2 - l * (l + 1))

    def fine_structure_correction(self, n: int, j: float) -> float:
        """
        Calculate fine structure energy correction

        ΔE_fs = E_n × α² × [n/(j+1/2) - 3/4] / n

        Args:
            n: Principal quantum number
            j: Total angular momentum

        Returns:
            Energy correction (J)
        """
        E_n = self.energy(n)
        return E_n * ALPHA_FS**2 * (n / (j + 0.5) - 0.75) / n


class MultielectronAtom:
    """Multi-electron atom in central field approximation"""

    def __init__(self, Z: int, n_electrons: int):
        """
        Initialize multi-electron atom

        Args:
            Z: Nuclear charge
            n_electrons: Number of electrons
        """
        self.Z = Z
        self.N = n_electrons

    def screening_constant(self, n: int, l: int) -> float:
        """
        Calculate Slater screening constant (simplified)

        Args:
            n: Principal quantum number
            l: Orbital angular momentum

        Returns:
            Screening constant σ
        """
        # Simplified Slater rules
        if n == 1:
            return 0.3 * (self.N - 1)
        elif n == 2:
            return 0.85 * min(self.N - 1, 2) + 0.35 * max(0, self.N - 3)
        else:
            return self.Z - self.effective_Z(n, l)

    def effective_Z(self, n: int, l: int) -> float:
        """Calculate effective nuclear charge Z_eff"""
        return self.Z - self.screening_constant(n, l)

    def orbital_energy(self, n: int, l: int) -> float:
        """
        Calculate orbital energy (screened hydrogen)

        E_nl ≈ -13.6 eV × Z_eff² / n²

        Args:
            n, l: Quantum numbers

        Returns:
            Orbital energy (J)
        """
        Z_eff = self.effective_Z(n, l)
        return -E_HARTREE * Z_eff**2 / (2 * n**2)

    def ionization_energy(self) -> float:
        """Estimate first ionization energy"""
        # Highest occupied orbital
        # Simplified - assumes filling order
        n, l = self._highest_occupied()
        return -self.orbital_energy(n, l)

    def _highest_occupied(self) -> Tuple[int, int]:
        """Determine highest occupied orbital"""
        # Aufbau principle (simplified)
        order = [(1,0), (2,0), (2,1), (3,0), (3,1), (4,0), (3,2), (4,1)]
        max_electrons = [2, 2, 6, 2, 6, 2, 10, 6]

        electrons_placed = 0
        for (n, l), max_e in zip(order, max_electrons):
            electrons_placed += max_e
            if electrons_placed >= self.N:
                return n, l
        return 4, 1  # Default


class SlaterDeterminant:
    """Slater determinant for antisymmetric wavefunctions"""

    def __init__(self, orbitals: List[Callable]):
        """
        Initialize Slater determinant

        Args:
            orbitals: List of single-particle orbital functions φ_i(r)
        """
        self.orbitals = orbitals
        self.n_electrons = len(orbitals)

    def evaluate(self, positions: np.ndarray) -> float:
        """
        Evaluate Slater determinant at given positions

        Ψ(r₁, r₂, ...) = (1/√N!) det|φ_i(r_j)|

        Args:
            positions: Array of electron positions (N×3)

        Returns:
            Wavefunction value
        """
        N = self.n_electrons
        matrix = np.zeros((N, N), dtype=complex)

        for i, orbital in enumerate(self.orbitals):
            for j, r in enumerate(positions):
                matrix[i, j] = orbital(r)

        return np.linalg.det(matrix) / np.sqrt(factorial(N))


class AtomicTerm:
    """Atomic term symbol and LS coupling"""

    def __init__(self, L: int, S: float, J: float = None):
        """
        Initialize atomic term

        Args:
            L: Total orbital angular momentum
            S: Total spin
            J: Total angular momentum (optional)
        """
        self.L = L
        self.S = S
        self.J = J if J is not None else abs(L - S)

    @property
    def term_symbol(self) -> str:
        """
        Get term symbol ²S+1 L_J

        Returns:
            Term symbol string (e.g., "³P₂")
        """
        L_symbols = ['S', 'P', 'D', 'F', 'G', 'H', 'I']
        L_sym = L_symbols[self.L] if self.L < len(L_symbols) else f'[{self.L}]'
        multiplicity = int(2 * self.S + 1)
        return f"{multiplicity}{L_sym}_{self.J}"

    def degeneracy(self) -> int:
        """Calculate term degeneracy 2J+1"""
        return int(2 * self.J + 1)

    def lande_g_factor(self) -> float:
        """
        Calculate Landé g-factor

        g_J = 1 + [J(J+1) + S(S+1) - L(L+1)] / [2J(J+1)]

        Returns:
            g-factor
        """
        if self.J == 0:
            return 0
        return 1 + (self.J*(self.J+1) + self.S*(self.S+1) - self.L*(self.L+1)) / \
               (2 * self.J * (self.J + 1))


class SelectionRules:
    """Atomic transition selection rules"""

    def __init__(self, transition_type: str = 'electric_dipole'):
        """
        Initialize selection rules

        Args:
            transition_type: 'electric_dipole', 'magnetic_dipole', 'electric_quadrupole'
        """
        self.type = transition_type

    def is_allowed(self, initial: AtomicTerm, final: AtomicTerm) -> bool:
        """
        Check if transition is allowed

        Args:
            initial: Initial atomic term
            final: Final atomic term

        Returns:
            True if transition is allowed
        """
        dL = abs(final.L - initial.L)
        dS = abs(final.S - initial.S)
        dJ = abs(final.J - initial.J)

        if self.type == 'electric_dipole':
            # ΔL = ±1, ΔS = 0, ΔJ = 0, ±1 (not 0→0)
            L_ok = dL == 1
            S_ok = dS == 0
            J_ok = dJ <= 1 and not (initial.J == 0 and final.J == 0)
            return L_ok and S_ok and J_ok

        elif self.type == 'magnetic_dipole':
            # ΔL = 0, ΔS = 0, ΔJ = 0, ±1
            return dL == 0 and dS == 0 and dJ <= 1

        return False


# =============================================================================
# Atom-Light Interaction
# =============================================================================

class TwoLevelAtom:
    """Two-level atom interacting with light"""

    def __init__(self, omega_0: float, d: float, gamma: float):
        """
        Initialize two-level atom

        Args:
            omega_0: Transition frequency (rad/s)
            d: Dipole matrix element (C·m)
            gamma: Spontaneous emission rate (s⁻¹)
        """
        self.omega_0 = omega_0
        self.d = d
        self.gamma = gamma

    def rabi_frequency(self, E_0: float) -> float:
        """
        Calculate Rabi frequency

        Ω = d·E₀/ℏ

        Args:
            E_0: Electric field amplitude (V/m)

        Returns:
            Rabi frequency (rad/s)
        """
        return self.d * E_0 / HBAR

    def generalized_rabi_frequency(self, Omega: float, delta: float) -> float:
        """
        Calculate generalized Rabi frequency

        Ω' = √(Ω² + δ²)

        Args:
            Omega: Rabi frequency
            delta: Detuning

        Returns:
            Generalized Rabi frequency
        """
        return np.sqrt(Omega**2 + delta**2)

    def excited_probability(self, t: float, Omega: float, delta: float = 0) -> float:
        """
        Calculate excited state probability (no damping)

        P_e(t) = (Ω/Ω')² sin²(Ω't/2)

        Args:
            t: Time (s)
            Omega: Rabi frequency
            delta: Detuning

        Returns:
            Excited state probability
        """
        Omega_prime = self.generalized_rabi_frequency(Omega, delta)
        return (Omega / Omega_prime)**2 * np.sin(Omega_prime * t / 2)**2

    def saturation_parameter(self, I: float) -> float:
        """
        Calculate saturation parameter s = I/I_sat

        Args:
            I: Intensity (W/m²)

        Returns:
            Saturation parameter
        """
        I_sat = self.saturation_intensity()
        return I / I_sat

    def saturation_intensity(self) -> float:
        """
        Calculate saturation intensity

        I_sat = πhcγ/(3λ³)

        Returns:
            Saturation intensity (W/m²)
        """
        omega_0 = self.omega_0
        return np.pi * HBAR * omega_0**3 * self.gamma / (3 * C**2)


class BlochEquations:
    """Optical Bloch equations for two-level system"""

    def __init__(self, atom: TwoLevelAtom):
        """
        Initialize Bloch equations

        Args:
            atom: Two-level atom
        """
        self.atom = atom

    def evolve(self, rho_0: np.ndarray, Omega: float, delta: float,
               t_span: Tuple[float, float], n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve optical Bloch equations

        Args:
            rho_0: Initial density matrix (2×2)
            Omega: Rabi frequency
            delta: Detuning
            t_span: Time range (t_start, t_end)
            n_points: Output points

        Returns:
            (t, rho) time array and density matrices
        """
        gamma = self.atom.gamma

        def bloch_rhs(t, y):
            # y = [rho_gg, rho_ee, Re(rho_ge), Im(rho_ge)]
            rho_gg, rho_ee, u, v = y

            drho_gg = gamma * rho_ee + Omega * v
            drho_ee = -gamma * rho_ee - Omega * v
            du = -gamma/2 * u + delta * v
            dv = -gamma/2 * v - delta * u + Omega * (rho_ee - rho_gg)

            return [drho_gg, drho_ee, du, dv]

        # Initial conditions from density matrix
        y0 = [np.real(rho_0[0,0]), np.real(rho_0[1,1]),
              np.real(rho_0[0,1]), np.imag(rho_0[0,1])]

        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        sol = solve_ivp(bloch_rhs, t_span, y0, t_eval=t_eval)

        # Reconstruct density matrices
        rho_t = []
        for i in range(len(sol.t)):
            rho = np.array([
                [sol.y[0,i], sol.y[2,i] + 1j*sol.y[3,i]],
                [sol.y[2,i] - 1j*sol.y[3,i], sol.y[1,i]]
            ])
            rho_t.append(rho)

        return sol.t, np.array(rho_t)

    def steady_state_population(self, Omega: float, delta: float) -> float:
        """
        Calculate steady-state excited population

        P_e = (s/2) / (1 + s + (2δ/γ)²)

        where s = 2Ω²/γ²

        Args:
            Omega: Rabi frequency
            delta: Detuning

        Returns:
            Steady-state excited population
        """
        gamma = self.atom.gamma
        s = 2 * Omega**2 / gamma**2
        return (s / 2) / (1 + s + (2 * delta / gamma)**2)


class DipoleMatrixElement:
    """Dipole matrix element calculations"""

    def __init__(self, atom: HydrogenAtom):
        """
        Initialize dipole matrix element calculator

        Args:
            atom: Hydrogen atom instance
        """
        self.atom = atom

    def radial_integral(self, n1: int, l1: int, n2: int, l2: int) -> float:
        """
        Calculate radial dipole integral ∫ R₁(r) r R₂(r) r² dr

        Args:
            n1, l1: Initial state quantum numbers
            n2, l2: Final state quantum numbers

        Returns:
            Radial integral (m)
        """
        def integrand(r):
            R1 = self.atom.radial_wavefunction(n1, l1, r)
            R2 = self.atom.radial_wavefunction(n2, l2, r)
            return R1 * r * R2 * r**2

        result, _ = quad(integrand, 0, 100 * self.atom.a0)
        return result

    def angular_integral(self, l1: int, m1: int, l2: int, m2: int, q: int) -> complex:
        """
        Calculate angular dipole integral

        Args:
            l1, m1: Initial state quantum numbers
            l2, m2: Final state quantum numbers
            q: Polarization (-1, 0, +1 for σ⁻, π, σ⁺)

        Returns:
            Angular integral
        """
        # Selection rules
        if abs(l2 - l1) != 1 or m2 - m1 != q:
            return 0

        # 3j symbol (simplified)
        return np.sqrt(3 * (2*l1 + 1) * (2*l2 + 1) / (4 * np.pi))


class EinsteinCoefficients:
    """Einstein A and B coefficients"""

    def __init__(self, omega: float, d: float):
        """
        Initialize Einstein coefficients

        Args:
            omega: Transition angular frequency (rad/s)
            d: Dipole matrix element (C·m)
        """
        self.omega = omega
        self.d = d

    def A_coefficient(self) -> float:
        """
        Calculate Einstein A coefficient (spontaneous emission rate)

        A = ω³|d|² / (3πε₀ℏc³)

        Returns:
            A coefficient (s⁻¹)
        """
        return self.omega**3 * abs(self.d)**2 / (3 * np.pi * EPSILON_0 * HBAR * C**3)

    def B_coefficient(self) -> float:
        """
        Calculate Einstein B coefficient (stimulated emission/absorption)

        B = π|d|² / (3ε₀ℏ²)

        Returns:
            B coefficient (m³/(J·s²))
        """
        return np.pi * abs(self.d)**2 / (3 * EPSILON_0 * HBAR**2)

    def lifetime(self) -> float:
        """Calculate spontaneous emission lifetime τ = 1/A"""
        return 1 / self.A_coefficient()


# =============================================================================
# Laser Physics
# =============================================================================

class LaserCavity:
    """Optical laser cavity"""

    def __init__(self, L: float, R1: float, R2: float, n: float = 1.0):
        """
        Initialize laser cavity

        Args:
            L: Cavity length (m)
            R1, R2: Mirror reflectivities
            n: Refractive index of medium
        """
        self.L = L
        self.R1 = R1
        self.R2 = R2
        self.n = n

    def free_spectral_range(self) -> float:
        """
        Calculate free spectral range

        FSR = c / (2nL)

        Returns:
            FSR (Hz)
        """
        return C / (2 * self.n * self.L)

    def finesse(self) -> float:
        """
        Calculate cavity finesse

        F = π√(R₁R₂) / (1 - √(R₁R₂))

        Returns:
            Finesse
        """
        r = np.sqrt(self.R1 * self.R2)
        return np.pi * np.sqrt(r) / (1 - r)

    def linewidth(self) -> float:
        """
        Calculate cavity linewidth

        δν = FSR / F

        Returns:
            Linewidth (Hz)
        """
        return self.free_spectral_range() / self.finesse()

    def quality_factor(self, omega: float) -> float:
        """
        Calculate Q factor

        Q = ω / δω

        Args:
            omega: Angular frequency (rad/s)

        Returns:
            Q factor
        """
        return omega / (2 * np.pi * self.linewidth())

    def photon_lifetime(self) -> float:
        """
        Calculate photon lifetime in cavity

        τ_c = L / (c × (1 - √(R₁R₂)))

        Returns:
            Photon lifetime (s)
        """
        r = np.sqrt(self.R1 * self.R2)
        return self.L / (C * (1 - r))

    def mode_frequencies(self, q_range: Tuple[int, int]) -> np.ndarray:
        """
        Calculate longitudinal mode frequencies

        ν_q = q × FSR

        Args:
            q_range: Range of mode numbers

        Returns:
            Array of frequencies (Hz)
        """
        FSR = self.free_spectral_range()
        q = np.arange(q_range[0], q_range[1])
        return q * FSR


class GainMedium:
    """Laser gain medium"""

    def __init__(self, sigma_e: float, sigma_a: float, tau: float, n_total: float):
        """
        Initialize gain medium

        Args:
            sigma_e: Emission cross section (m²)
            sigma_a: Absorption cross section (m²)
            tau: Upper state lifetime (s)
            n_total: Total atom density (m⁻³)
        """
        self.sigma_e = sigma_e
        self.sigma_a = sigma_a
        self.tau = tau
        self.n_total = n_total

    def small_signal_gain(self, N_inv: float) -> float:
        """
        Calculate small-signal gain coefficient

        g₀ = (σ_e - σ_a) × N_inv

        Args:
            N_inv: Population inversion density (m⁻³)

        Returns:
            Gain coefficient (m⁻¹)
        """
        return (self.sigma_e - self.sigma_a) * N_inv

    def gain_with_saturation(self, g_0: float, I: float, I_sat: float) -> float:
        """
        Calculate saturated gain

        g = g₀ / (1 + I/I_sat)

        Args:
            g_0: Small-signal gain
            I: Intensity (W/m²)
            I_sat: Saturation intensity (W/m²)

        Returns:
            Saturated gain (m⁻¹)
        """
        return g_0 / (1 + I / I_sat)

    def saturation_intensity(self, omega: float) -> float:
        """
        Calculate saturation intensity

        I_sat = ℏω / (σ_e × τ)

        Args:
            omega: Angular frequency (rad/s)

        Returns:
            Saturation intensity (W/m²)
        """
        return HBAR * omega / (self.sigma_e * self.tau)


class RateEquations:
    """Laser rate equations"""

    def __init__(self, cavity: LaserCavity, gain: GainMedium):
        """
        Initialize rate equations

        Args:
            cavity: Laser cavity
            gain: Gain medium
        """
        self.cavity = cavity
        self.gain = gain

    def evolve(self, N0: float, phi0: float, P_pump: float,
               t_span: Tuple[float, float], n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve laser rate equations

        dN/dt = P_pump - N/τ - c·σ_e·N·φ
        dφ/dt = c·σ_e·N·φ - φ/τ_c

        Args:
            N0: Initial population inversion
            phi0: Initial photon number
            P_pump: Pump rate (s⁻¹ m⁻³)
            t_span: Time range
            n_points: Output points

        Returns:
            (t, N, phi) arrays
        """
        tau = self.gain.tau
        tau_c = self.cavity.photon_lifetime()
        sigma_e = self.gain.sigma_e

        def rhs(t, y):
            N, phi = y
            dN = P_pump - N / tau - C * sigma_e * N * phi
            dphi = C * sigma_e * N * phi - phi / tau_c
            return [dN, dphi]

        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        sol = solve_ivp(rhs, t_span, [N0, phi0], t_eval=t_eval)

        return sol.t, sol.y[0], sol.y[1]

    def threshold_inversion(self) -> float:
        """
        Calculate threshold population inversion

        N_th = 1 / (c·σ_e·τ_c)

        Returns:
            Threshold inversion density (m⁻³)
        """
        tau_c = self.cavity.photon_lifetime()
        return 1 / (C * self.gain.sigma_e * tau_c)


class ModeLocking:
    """Mode-locked laser pulses"""

    def __init__(self, N_modes: int, delta_omega: float, phi_0: float = 0):
        """
        Initialize mode-locked laser

        Args:
            N_modes: Number of locked modes
            delta_omega: Mode spacing (rad/s)
            phi_0: Phase offset
        """
        self.N = N_modes
        self.delta_omega = delta_omega
        self.phi_0 = phi_0

    def pulse_duration(self) -> float:
        """
        Calculate pulse duration (FWHM)

        Δt ≈ 2π / (N × Δω)

        Returns:
            Pulse duration (s)
        """
        return 2 * np.pi / (self.N * self.delta_omega)

    def repetition_rate(self) -> float:
        """
        Calculate repetition rate

        f_rep = Δω / (2π)

        Returns:
            Repetition rate (Hz)
        """
        return self.delta_omega / (2 * np.pi)

    def pulse_train(self, t: np.ndarray, E_0: float = 1.0) -> np.ndarray:
        """
        Calculate electric field of pulse train

        E(t) = E₀ Σ exp(i n Δω t)

        Args:
            t: Time array
            E_0: Field amplitude

        Returns:
            Electric field array
        """
        E = np.zeros_like(t, dtype=complex)
        for n in range(-(self.N//2), self.N//2 + 1):
            E += np.exp(1j * n * self.delta_omega * t + 1j * self.phi_0 * n**2)
        return E_0 * E


# =============================================================================
# Laser Cooling & Trapping
# =============================================================================

class DopplerCooling:
    """Doppler laser cooling"""

    def __init__(self, atom: TwoLevelAtom, wavelength: float):
        """
        Initialize Doppler cooling

        Args:
            atom: Two-level atom
            wavelength: Cooling laser wavelength (m)
        """
        self.atom = atom
        self.wavelength = wavelength
        self.k = 2 * np.pi / wavelength

    def doppler_temperature(self) -> float:
        """
        Calculate Doppler temperature limit

        T_D = ℏγ / (2k_B)

        Returns:
            Doppler temperature (K)
        """
        return HBAR * self.atom.gamma / (2 * KB)

    def scattering_force(self, v: float, delta: float, s: float) -> float:
        """
        Calculate scattering force

        F = ℏk γ (s/2) / (1 + s + (2(δ - kv)/γ)²)

        Args:
            v: Atom velocity (m/s)
            delta: Detuning (rad/s)
            s: Saturation parameter

        Returns:
            Force (N)
        """
        gamma = self.atom.gamma
        delta_eff = delta - self.k * v
        return (HBAR * self.k * gamma * s / 2) / \
               (1 + s + (2 * delta_eff / gamma)**2)

    def cooling_force(self, v: float, delta: float, s: float) -> float:
        """
        Calculate net cooling force from counter-propagating beams

        Args:
            v: Velocity
            delta: Detuning (should be negative for cooling)
            s: Saturation parameter per beam

        Returns:
            Net force
        """
        F_plus = self.scattering_force(v, delta, s)  # k
        F_minus = self.scattering_force(v, delta + 2*self.k*v, s)  # -k beam
        return F_plus - F_minus

    def damping_coefficient(self, delta: float, s: float) -> float:
        """
        Calculate velocity damping coefficient β

        F ≈ -β v  for small v

        Args:
            delta: Detuning
            s: Saturation parameter

        Returns:
            Damping coefficient (kg/s)
        """
        gamma = self.atom.gamma
        return 8 * HBAR * self.k**2 * s * (-delta / gamma) / \
               (1 + s + (2 * delta / gamma)**2)**2


class MagnetoOpticalTrap:
    """Magneto-optical trap (MOT)"""

    def __init__(self, cooling: DopplerCooling, dB_dz: float):
        """
        Initialize MOT

        Args:
            cooling: Doppler cooling instance
            dB_dz: Magnetic field gradient (T/m)
        """
        self.cooling = cooling
        self.dB_dz = dB_dz

    def spring_constant(self, delta: float, s: float, g: float = 1.0) -> float:
        """
        Calculate MOT spring constant

        κ = β × (μ_B g / ℏk) × (dB/dz)

        Args:
            delta: Detuning
            s: Saturation parameter
            g: Landé g-factor

        Returns:
            Spring constant (N/m)
        """
        beta = self.cooling.damping_coefficient(delta, s)
        mu_B = 9.274e-24  # Bohr magneton
        return beta * mu_B * g * self.dB_dz / (HBAR * self.cooling.k)

    def trap_frequency(self, m: float, delta: float, s: float) -> float:
        """
        Calculate trap oscillation frequency

        Args:
            m: Atom mass
            delta: Detuning
            s: Saturation parameter

        Returns:
            Trap frequency (Hz)
        """
        kappa = self.spring_constant(delta, s)
        return np.sqrt(kappa / m) / (2 * np.pi)


class OpticalDipoleTrap:
    """Far-detuned optical dipole trap"""

    def __init__(self, wavelength: float, power: float, waist: float,
                 alpha: float):
        """
        Initialize optical dipole trap

        Args:
            wavelength: Laser wavelength (m)
            power: Laser power (W)
            waist: Beam waist (m)
            alpha: Atomic polarizability (C²·m²/J)
        """
        self.wavelength = wavelength
        self.P = power
        self.w0 = waist
        self.alpha = alpha

    def trap_depth(self) -> float:
        """
        Calculate trap depth

        U₀ = α I₀ / (2ε₀c)

        Returns:
            Trap depth (J)
        """
        I_0 = 2 * self.P / (np.pi * self.w0**2)
        return self.alpha * I_0 / (2 * EPSILON_0 * C)

    def potential(self, r: float, z: float) -> float:
        """
        Calculate trapping potential U(r,z)

        Args:
            r: Radial position
            z: Axial position

        Returns:
            Potential (J)
        """
        z_R = np.pi * self.w0**2 / self.wavelength  # Rayleigh range
        w_z = self.w0 * np.sqrt(1 + (z / z_R)**2)
        I = (2 * self.P / (np.pi * w_z**2)) * np.exp(-2 * r**2 / w_z**2)
        return -self.alpha * I / (2 * EPSILON_0 * C)

    def radial_frequency(self, m: float) -> float:
        """
        Calculate radial trap frequency

        Args:
            m: Atom mass

        Returns:
            Radial frequency (Hz)
        """
        U_0 = self.trap_depth()
        return np.sqrt(4 * U_0 / (m * self.w0**2)) / (2 * np.pi)

    def axial_frequency(self, m: float) -> float:
        """
        Calculate axial trap frequency

        Args:
            m: Atom mass

        Returns:
            Axial frequency (Hz)
        """
        z_R = np.pi * self.w0**2 / self.wavelength
        U_0 = self.trap_depth()
        return np.sqrt(2 * U_0 / (m * z_R**2)) / (2 * np.pi)


class OpticalLattice:
    """Optical lattice potential"""

    def __init__(self, wavelength: float, depth_Er: float):
        """
        Initialize optical lattice

        Args:
            wavelength: Lattice wavelength (m)
            depth_Er: Lattice depth in recoil energies
        """
        self.wavelength = wavelength
        self.a = wavelength / 2  # Lattice constant
        self.V_0_Er = depth_Er

    def recoil_energy(self, m: float) -> float:
        """
        Calculate recoil energy E_r = ℏ²k²/(2m)

        Args:
            m: Atom mass

        Returns:
            Recoil energy (J)
        """
        k = 2 * np.pi / self.wavelength
        return HBAR**2 * k**2 / (2 * m)

    def depth(self, m: float) -> float:
        """Calculate lattice depth in Joules"""
        return self.V_0_Er * self.recoil_energy(m)

    def potential(self, x: float) -> float:
        """
        Calculate lattice potential V(x) = V₀ sin²(kx)

        Args:
            x: Position

        Returns:
            Potential (in units of recoil energy)
        """
        k = 2 * np.pi / self.wavelength
        return self.V_0_Er * np.sin(k * x)**2

    def tunneling_rate(self, m: float) -> float:
        """
        Estimate tunneling rate J (tight-binding)

        J/E_r ≈ (4/√π) (V₀/E_r)^(3/4) exp(-2√(V₀/E_r))

        Args:
            m: Atom mass

        Returns:
            Tunneling rate (Hz)
        """
        s = self.V_0_Er
        E_r = self.recoil_energy(m)
        J_Er = (4 / np.sqrt(np.pi)) * s**(3/4) * np.exp(-2 * np.sqrt(s))
        return J_Er * E_r / (2 * np.pi * HBAR)


# =============================================================================
# Ultracold Atoms
# =============================================================================

class BoseEinsteinCondensate:
    """Bose-Einstein condensate properties"""

    def __init__(self, N: int, m: float, a_s: float,
                 omega_x: float, omega_y: float, omega_z: float):
        """
        Initialize BEC

        Args:
            N: Number of atoms
            m: Atom mass
            a_s: s-wave scattering length (m)
            omega_x, omega_y, omega_z: Trap frequencies (rad/s)
        """
        self.N = N
        self.m = m
        self.a_s = a_s
        self.omega = np.array([omega_x, omega_y, omega_z])

    def critical_temperature(self) -> float:
        """
        Calculate BEC critical temperature

        T_c = (ℏ ω̄/k_B) (N/ζ(3))^(1/3)

        where ω̄ = (ω_x ω_y ω_z)^(1/3)

        Returns:
            Critical temperature (K)
        """
        omega_bar = np.prod(self.omega)**(1/3)
        zeta_3 = 1.202  # Riemann zeta(3)
        return HBAR * omega_bar * (self.N / zeta_3)**(1/3) / KB

    def condensate_fraction(self, T: float) -> float:
        """
        Calculate condensate fraction N₀/N

        N₀/N = 1 - (T/T_c)³

        Args:
            T: Temperature (K)

        Returns:
            Condensate fraction
        """
        T_c = self.critical_temperature()
        if T >= T_c:
            return 0
        return 1 - (T / T_c)**3

    def chemical_potential(self) -> float:
        """
        Calculate Thomas-Fermi chemical potential

        μ = (1/2) ℏω̄ (15 N a_s / a_ho)^(2/5)

        Returns:
            Chemical potential (J)
        """
        omega_bar = np.prod(self.omega)**(1/3)
        a_ho = np.sqrt(HBAR / (self.m * omega_bar))
        return 0.5 * HBAR * omega_bar * (15 * self.N * self.a_s / a_ho)**(2/5)

    def healing_length(self) -> float:
        """
        Calculate healing length

        ξ = ℏ / √(2 m μ)

        Returns:
            Healing length (m)
        """
        mu = self.chemical_potential()
        return HBAR / np.sqrt(2 * self.m * mu)

    def thomas_fermi_radius(self) -> np.ndarray:
        """
        Calculate Thomas-Fermi radii

        R_i = √(2μ/(m ω_i²))

        Returns:
            Array of TF radii (m)
        """
        mu = self.chemical_potential()
        return np.sqrt(2 * mu / (self.m * self.omega**2))


class GrossPitaevskii:
    """Gross-Pitaevskii equation solver"""

    def __init__(self, bec: BoseEinsteinCondensate, grid: np.ndarray):
        """
        Initialize GP equation solver

        Args:
            bec: BEC instance
            grid: Spatial grid
        """
        self.bec = bec
        self.grid = grid
        self.dx = grid[1] - grid[0]

    def interaction_parameter(self) -> float:
        """Calculate interaction parameter g = 4πℏ²a_s/m"""
        return 4 * np.pi * HBAR**2 * self.bec.a_s / self.bec.m

    def hamiltonian(self, psi: np.ndarray) -> np.ndarray:
        """
        Apply GP Hamiltonian

        H ψ = [-ℏ²/(2m) ∇² + V + g|ψ|²] ψ

        Args:
            psi: Wavefunction

        Returns:
            H ψ
        """
        g = self.interaction_parameter()
        m = self.bec.m
        omega_x = self.bec.omega[0]

        # Kinetic term (1D Laplacian)
        kinetic = np.zeros_like(psi)
        kinetic[1:-1] = (psi[2:] - 2*psi[1:-1] + psi[:-2]) / self.dx**2
        kinetic *= -HBAR**2 / (2 * m)

        # Trap potential (harmonic)
        V = 0.5 * m * omega_x**2 * self.grid**2

        # Interaction
        interaction = g * np.abs(psi)**2

        return kinetic + (V + interaction) * psi

    def energy(self, psi: np.ndarray) -> float:
        """
        Calculate total energy

        E = ∫ [ℏ²/(2m)|∇ψ|² + V|ψ|² + (g/2)|ψ|⁴] dx

        Args:
            psi: Normalized wavefunction

        Returns:
            Total energy (J)
        """
        g = self.interaction_parameter()
        m = self.bec.m
        omega_x = self.bec.omega[0]

        # Gradient
        grad_psi = np.gradient(psi, self.dx)

        # Kinetic
        T = HBAR**2 / (2 * m) * np.sum(np.abs(grad_psi)**2) * self.dx

        # Potential
        V = 0.5 * m * omega_x**2 * self.grid**2
        U = np.sum(V * np.abs(psi)**2) * self.dx

        # Interaction
        W = 0.5 * g * np.sum(np.abs(psi)**4) * self.dx

        return T + U + W


class FermiGas:
    """Degenerate Fermi gas"""

    def __init__(self, N: int, m: float, omega: float):
        """
        Initialize Fermi gas

        Args:
            N: Number of atoms
            m: Atom mass
            omega: Trap frequency (rad/s)
        """
        self.N = N
        self.m = m
        self.omega = omega

    def fermi_energy(self) -> float:
        """
        Calculate Fermi energy in harmonic trap

        E_F = ℏω (6N)^(1/3)

        Returns:
            Fermi energy (J)
        """
        return HBAR * self.omega * (6 * self.N)**(1/3)

    def fermi_temperature(self) -> float:
        """Calculate Fermi temperature T_F = E_F / k_B"""
        return self.fermi_energy() / KB

    def thomas_fermi_radius(self) -> float:
        """
        Calculate Thomas-Fermi radius

        R_F = √(2 E_F / (m ω²))

        Returns:
            TF radius (m)
        """
        E_F = self.fermi_energy()
        return np.sqrt(2 * E_F / (self.m * self.omega**2))

    def degeneracy_parameter(self, T: float) -> float:
        """
        Calculate T/T_F

        Args:
            T: Temperature (K)

        Returns:
            Degeneracy parameter
        """
        return T / self.fermi_temperature()


class FeshbachResonance:
    """Feshbach resonance for tunable interactions"""

    def __init__(self, a_bg: float, B_0: float, Delta: float):
        """
        Initialize Feshbach resonance

        Args:
            a_bg: Background scattering length (m)
            B_0: Resonance magnetic field (T)
            Delta: Resonance width (T)
        """
        self.a_bg = a_bg
        self.B_0 = B_0
        self.Delta = Delta

    def scattering_length(self, B: float) -> float:
        """
        Calculate scattering length at magnetic field B

        a(B) = a_bg (1 - Δ/(B - B₀))

        Args:
            B: Magnetic field (T)

        Returns:
            Scattering length (m)
        """
        if abs(B - self.B_0) < 1e-10:
            return np.inf if self.Delta > 0 else -np.inf
        return self.a_bg * (1 - self.Delta / (B - self.B_0))

    def is_attractive(self, B: float) -> bool:
        """Check if interactions are attractive (a < 0)"""
        return self.scattering_length(B) < 0

    def unitarity_field(self) -> float:
        """Get magnetic field for unitarity (|a| → ∞)"""
        return self.B_0


# =============================================================================
# Molecular Physics
# =============================================================================

class MolecularOrbital:
    """LCAO molecular orbital"""

    def __init__(self, coefficients: np.ndarray, atomic_orbitals: List[Callable]):
        """
        Initialize molecular orbital

        ψ_MO = Σ c_i φ_i

        Args:
            coefficients: LCAO coefficients
            atomic_orbitals: List of atomic orbital functions
        """
        self.c = np.array(coefficients)
        self.ao = atomic_orbitals

    def evaluate(self, r: np.ndarray) -> complex:
        """
        Evaluate MO at position r

        Args:
            r: Position vector

        Returns:
            MO value
        """
        result = 0j
        for ci, phi in zip(self.c, self.ao):
            result += ci * phi(r)
        return result

    def normalize(self, grid: np.ndarray):
        """Normalize MO on given grid"""
        norm_sq = 0
        dx = grid[1] - grid[0]
        for x in grid:
            norm_sq += abs(self.evaluate(np.array([x, 0, 0])))**2 * dx
        self.c /= np.sqrt(norm_sq)


class BornOppenheimer:
    """Born-Oppenheimer approximation"""

    def __init__(self, electronic_energy: Callable):
        """
        Initialize Born-Oppenheimer calculation

        Args:
            electronic_energy: Function E(R) returning electronic energy
                              as function of nuclear coordinates
        """
        self.E_elec = electronic_energy

    def potential_energy_surface(self, R_range: np.ndarray) -> np.ndarray:
        """
        Calculate potential energy surface

        Args:
            R_range: Nuclear coordinate values

        Returns:
            PES values
        """
        return np.array([self.E_elec(R) for R in R_range])

    def equilibrium_position(self, R_range: np.ndarray) -> float:
        """
        Find equilibrium nuclear position (PES minimum)

        Args:
            R_range: Nuclear coordinate range

        Returns:
            Equilibrium position
        """
        PES = self.potential_energy_surface(R_range)
        min_idx = np.argmin(PES)
        return R_range[min_idx]

    def vibrational_frequency(self, R_eq: float, m_red: float,
                              h: float = 1e-12) -> float:
        """
        Calculate vibrational frequency from PES curvature

        ω = √(k/μ) where k = d²E/dR²

        Args:
            R_eq: Equilibrium position
            m_red: Reduced mass
            h: Finite difference step

        Returns:
            Vibrational frequency (rad/s)
        """
        # Second derivative
        E_p = self.E_elec(R_eq + h)
        E_m = self.E_elec(R_eq - h)
        E_0 = self.E_elec(R_eq)
        k = (E_p - 2*E_0 + E_m) / h**2

        return np.sqrt(abs(k) / m_red)


class VibrationalSpectrum:
    """Molecular vibrational spectrum"""

    def __init__(self, omega_e: float, x_e: float = 0):
        """
        Initialize vibrational spectrum (Morse oscillator)

        E_v = ℏω_e(v + 1/2) - ℏω_e x_e(v + 1/2)²

        Args:
            omega_e: Harmonic frequency (rad/s)
            x_e: Anharmonicity constant
        """
        self.omega_e = omega_e
        self.x_e = x_e

    def energy(self, v: int) -> float:
        """
        Calculate vibrational energy level

        Args:
            v: Vibrational quantum number

        Returns:
            Energy (J)
        """
        return HBAR * self.omega_e * ((v + 0.5) - self.x_e * (v + 0.5)**2)

    def transition_frequency(self, v_initial: int, v_final: int) -> float:
        """
        Calculate transition frequency

        Args:
            v_initial, v_final: Vibrational quantum numbers

        Returns:
            Frequency (Hz)
        """
        dE = self.energy(v_final) - self.energy(v_initial)
        return abs(dE) / (2 * np.pi * HBAR)

    def dissociation_energy(self) -> float:
        """
        Calculate dissociation energy (Morse)

        D_e = ℏω_e / (4x_e)

        Returns:
            Dissociation energy (J)
        """
        if self.x_e <= 0:
            return np.inf
        return HBAR * self.omega_e / (4 * self.x_e)


class RotationalSpectrum:
    """Molecular rotational spectrum"""

    def __init__(self, B: float, D: float = 0):
        """
        Initialize rotational spectrum

        E_J = B J(J+1) - D J²(J+1)²

        Args:
            B: Rotational constant (J)
            D: Centrifugal distortion constant (J)
        """
        self.B = B
        self.D = D

    @classmethod
    def from_moment_of_inertia(cls, I: float) -> 'RotationalSpectrum':
        """
        Create from moment of inertia

        Args:
            I: Moment of inertia (kg·m²)

        Returns:
            RotationalSpectrum instance
        """
        B = HBAR**2 / (2 * I)
        return cls(B)

    def energy(self, J: int) -> float:
        """
        Calculate rotational energy level

        Args:
            J: Rotational quantum number

        Returns:
            Energy (J)
        """
        return self.B * J * (J + 1) - self.D * J**2 * (J + 1)**2

    def transition_frequency(self, J: int) -> float:
        """
        Calculate J → J+1 transition frequency

        Args:
            J: Initial rotational quantum number

        Returns:
            Frequency (Hz)
        """
        dE = self.energy(J + 1) - self.energy(J)
        return dE / (2 * np.pi * HBAR)


class FranckCondon:
    """Franck-Condon principle for vibronic transitions"""

    def __init__(self, psi_ground: Callable, psi_excited: Callable):
        """
        Initialize Franck-Condon calculation

        Args:
            psi_ground: Ground state vibrational wavefunction
            psi_excited: Excited state vibrational wavefunction
        """
        self.psi_g = psi_ground
        self.psi_e = psi_excited

    def overlap_integral(self, R_range: np.ndarray) -> float:
        """
        Calculate Franck-Condon factor |⟨ψ_e|ψ_g⟩|²

        Args:
            R_range: Nuclear coordinate grid

        Returns:
            FC factor
        """
        dR = R_range[1] - R_range[0]
        overlap = 0j
        for R in R_range:
            overlap += np.conj(self.psi_e(R)) * self.psi_g(R) * dR
        return abs(overlap)**2

    def vertical_transition_energy(self, PES_ground: Callable,
                                    PES_excited: Callable,
                                    R_eq: float) -> float:
        """
        Calculate vertical transition energy

        Args:
            PES_ground: Ground state PES
            PES_excited: Excited state PES
            R_eq: Equilibrium position

        Returns:
            Vertical transition energy (J)
        """
        return PES_excited(R_eq) - PES_ground(R_eq)
