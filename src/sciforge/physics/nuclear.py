"""
Nuclear and Particle Physics module.

This module implements nuclear structure, radioactive decay,
nuclear reactions, and basic particle physics.

Classes:
    Scattering Theory:
    - PartialWave: Partial wave expansion
    - ScatteringAmplitude: Scattering amplitude calculations
    - OpticalTheorem: Optical theorem relations
    - RutherfordScattering: Coulomb scattering
    - MottScattering: Relativistic Coulomb scattering

    Nuclear Structure:
    - LiquidDropModel: Semi-empirical mass formula
    - ShellModel: Nuclear shell model
    - WoodsSaxon: Woods-Saxon potential
    - NuclearRadius: Nuclear radius systematics
    - NuclearSpin: Angular momentum coupling

    Radioactivity:
    - AlphaDecay: Alpha decay via tunneling
    - BetaDecay: Beta decay (Fermi theory)
    - GammaDecay: Electromagnetic transitions
    - DecayChain: Radioactive decay chains
    - HalfLife: Decay statistics

    Nuclear Reactions:
    - NuclearCrossSection: Reaction cross sections
    - QValue: Reaction energetics
    - ResonanceFormula: Breit-Wigner resonances
    - CompoundNucleus: Statistical model
    - FissionYield: Fission product distribution
    - FusionRate: Fusion reaction rates

    Particle Physics:
    - DiracEquation: Relativistic electron equation
    - KleinGordonEquation: Spin-0 field equation
    - DiracSpinor: Four-component spinors
    - GammaMatrices: Dirac algebra
    - NeutrinoOscillation: PMNS mixing
    - QuarkModel: Hadron spectroscopy
"""

import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, Tuple, Callable, Dict, List, Union
from scipy import integrate, special, optimize
from dataclasses import dataclass

# Physical constants
c = 2.998e8           # Speed of light (m/s)
hbar = 1.055e-34      # Reduced Planck constant (J·s)
e = 1.602e-19         # Elementary charge (C)
m_e = 9.109e-31       # Electron mass (kg)
m_p = 1.673e-27       # Proton mass (kg)
m_n = 1.675e-27       # Neutron mass (kg)
k_B = 1.381e-23       # Boltzmann constant (J/K)
epsilon_0 = 8.854e-12 # Vacuum permittivity (F/m)
alpha_fine = 1/137.036  # Fine structure constant

# Nuclear constants
fm = 1e-15            # Femtometer (m)
MeV = 1e6 * e         # MeV in Joules
u = 1.661e-27         # Atomic mass unit (kg)
r0 = 1.2 * fm         # Nuclear radius parameter


# =============================================================================
# Scattering Theory
# =============================================================================

class PartialWave:
    """
    Partial wave expansion for scattering.

    The scattering amplitude is expanded in terms of partial waves:
    f(θ) = Σ_l (2l+1) f_l P_l(cos θ)

    where f_l = (e^(2iδ_l) - 1) / (2ik) = (sin δ_l / k) e^(iδ_l)

    Args:
        k: Wave number
        l_max: Maximum angular momentum
    """

    def __init__(self, k: float, l_max: int = 10):
        if k <= 0:
            raise ValueError("Wave number must be positive")
        if l_max < 0:
            raise ValueError("l_max must be non-negative")

        self.k = k
        self.l_max = l_max
        self._phase_shifts = np.zeros(l_max + 1)

    def set_phase_shift(self, l: int, delta: float) -> None:
        """
        Set phase shift for partial wave l.

        Args:
            l: Angular momentum quantum number
            delta: Phase shift in radians
        """
        if l < 0 or l > self.l_max:
            raise ValueError(f"l must be in [0, {self.l_max}]")
        self._phase_shifts[l] = delta

    def get_phase_shift(self, l: int) -> float:
        """Get phase shift for partial wave l."""
        return self._phase_shifts[l]

    def partial_amplitude(self, l: int) -> complex:
        """
        Partial wave amplitude f_l.

        f_l = (e^(2iδ_l) - 1) / (2ik)

        Args:
            l: Angular momentum

        Returns:
            Complex partial wave amplitude
        """
        delta = self._phase_shifts[l]
        return (np.exp(2j * delta) - 1) / (2j * self.k)

    def scattering_amplitude(self, theta: ArrayLike) -> np.ndarray:
        """
        Total scattering amplitude f(θ).

        Args:
            theta: Scattering angle(s)

        Returns:
            Complex scattering amplitude
        """
        theta = np.asarray(theta)
        cos_theta = np.cos(theta)

        f = np.zeros_like(theta, dtype=complex)

        for l in range(self.l_max + 1):
            f_l = self.partial_amplitude(l)
            P_l = special.eval_legendre(l, cos_theta)
            f += (2 * l + 1) * f_l * P_l

        return f

    def differential_cross_section(self, theta: ArrayLike) -> np.ndarray:
        """
        Differential cross section dσ/dΩ = |f(θ)|².

        Args:
            theta: Scattering angle(s)

        Returns:
            Differential cross section
        """
        f = self.scattering_amplitude(theta)
        return np.abs(f)**2

    def total_cross_section(self) -> float:
        """
        Total cross section from optical theorem.

        σ_tot = (4π/k²) Σ_l (2l+1) sin²(δ_l)
        """
        sigma = 0
        for l in range(self.l_max + 1):
            delta = self._phase_shifts[l]
            sigma += (2 * l + 1) * np.sin(delta)**2

        return 4 * np.pi / self.k**2 * sigma

    def partial_cross_section(self, l: int) -> float:
        """
        Partial wave cross section σ_l.

        σ_l = (4π/k²)(2l+1) sin²(δ_l)
        """
        delta = self._phase_shifts[l]
        return 4 * np.pi / self.k**2 * (2 * l + 1) * np.sin(delta)**2


class ScatteringAmplitude:
    """
    General scattering amplitude calculations.

    Provides methods to compute scattering observables from
    the complex scattering amplitude f(θ).

    Args:
        amplitude_func: Function f(theta) returning complex amplitude
    """

    def __init__(self, amplitude_func: Callable[[ArrayLike], np.ndarray]):
        self.f = amplitude_func

    def differential_cross_section(self, theta: ArrayLike) -> np.ndarray:
        """dσ/dΩ = |f(θ)|²"""
        return np.abs(self.f(theta))**2

    def total_cross_section(self, n_theta: int = 100) -> float:
        """
        Integrate differential cross section.

        σ_tot = ∫ dσ/dΩ dΩ = 2π ∫ |f(θ)|² sin(θ) dθ
        """
        theta = np.linspace(0, np.pi, n_theta)
        dcs = self.differential_cross_section(theta)
        integrand = dcs * np.sin(theta)
        return 2 * np.pi * np.trapezoid(integrand, theta)

    def forward_amplitude(self) -> complex:
        """Forward scattering amplitude f(0)."""
        return self.f(0)

    def optical_theorem_check(self, k: float) -> float:
        """
        Check optical theorem: σ_tot = (4π/k) Im[f(0)].

        Args:
            k: Wave number

        Returns:
            Relative difference between LHS and RHS
        """
        sigma_integrated = self.total_cross_section()
        sigma_optical = 4 * np.pi / k * np.imag(self.forward_amplitude())

        return np.abs(sigma_integrated - sigma_optical) / sigma_integrated


class OpticalTheorem:
    """
    Optical theorem relations for scattering.

    The optical theorem connects total cross section to forward
    scattering amplitude:

    σ_tot = (4π/k) Im[f(0)]

    This is a consequence of unitarity (probability conservation).

    Args:
        k: Wave number
    """

    def __init__(self, k: float):
        if k <= 0:
            raise ValueError("Wave number must be positive")
        self.k = k

    def total_cross_section(self, forward_amplitude: complex) -> float:
        """
        Calculate total cross section from forward amplitude.

        σ_tot = (4π/k) Im[f(0)]

        Args:
            forward_amplitude: f(θ=0)

        Returns:
            Total cross section
        """
        return 4 * np.pi / self.k * np.imag(forward_amplitude)

    def forward_imaginary_part(self, total_cross_section: float) -> float:
        """
        Calculate Im[f(0)] from total cross section.

        Args:
            total_cross_section: σ_tot

        Returns:
            Imaginary part of forward amplitude
        """
        return total_cross_section * self.k / (4 * np.pi)

    def unitarity_bound(self, l: int) -> float:
        """
        Unitarity bound on partial wave amplitude.

        |f_l| ≤ 1/k  (from |S_l| ≤ 1)

        Args:
            l: Partial wave

        Returns:
            Maximum |f_l|
        """
        return 1 / self.k

    def partial_wave_limit(self, l: int) -> float:
        """
        Maximum partial cross section from unitarity.

        σ_l ≤ (4π/k²)(2l+1)

        Args:
            l: Partial wave

        Returns:
            Maximum σ_l
        """
        return 4 * np.pi / self.k**2 * (2 * l + 1)


class RutherfordScattering:
    """
    Rutherford scattering (Coulomb scattering).

    The Rutherford formula for scattering of a charged particle
    by a point Coulomb potential:

    dσ/dΩ = (Z₁Z₂e²/4E)² / sin⁴(θ/2)

    Args:
        Z1: Projectile charge number
        Z2: Target charge number
        energy: Kinetic energy in lab frame (J or MeV if use_MeV=True)
        use_MeV: If True, energy is in MeV
    """

    def __init__(self, Z1: int, Z2: int, energy: float, use_MeV: bool = True):
        if energy <= 0:
            raise ValueError("Energy must be positive")

        self.Z1 = Z1
        self.Z2 = Z2
        self.energy = energy * MeV if use_MeV else energy

    @property
    def distance_of_closest_approach(self) -> float:
        """
        Distance of closest approach (head-on collision).

        d = Z₁Z₂e² / (4πε₀ 2E)
        """
        return self.Z1 * self.Z2 * e**2 / (4 * np.pi * epsilon_0 * 2 * self.energy)

    @property
    def sommerfeld_parameter(self) -> float:
        """
        Sommerfeld parameter η = Z₁Z₂e²/(4πε₀ ħv).

        Characterizes strength of Coulomb interaction.
        """
        # Assume proton projectile for simplicity
        v = np.sqrt(2 * self.energy / m_p)
        return self.Z1 * self.Z2 * e**2 / (4 * np.pi * epsilon_0 * hbar * v)

    def differential_cross_section(self, theta: ArrayLike) -> np.ndarray:
        """
        Rutherford differential cross section.

        dσ/dΩ = (a/2)² / sin⁴(θ/2)

        where a = Z₁Z₂e²/(4πε₀ 2E)

        Args:
            theta: Scattering angle(s)

        Returns:
            Differential cross section
        """
        theta = np.asarray(theta)
        a = self.distance_of_closest_approach
        return (a / 2)**2 / np.sin(theta / 2)**4

    def impact_parameter(self, theta: float) -> float:
        """
        Classical impact parameter for scattering angle theta.

        b = (a/2) cot(θ/2)

        Args:
            theta: Scattering angle

        Returns:
            Impact parameter
        """
        a = self.distance_of_closest_approach
        return (a / 2) / np.tan(theta / 2)

    def total_cross_section(self, theta_min: float = 0.01) -> float:
        """
        Total cross section with minimum angle cutoff.

        Rutherford cross section diverges at θ→0, so we need
        a minimum angle (screening cutoff).

        Args:
            theta_min: Minimum scattering angle (radians)

        Returns:
            Total cross section
        """
        a = self.distance_of_closest_approach
        return np.pi * (a / 2)**2 / np.sin(theta_min / 2)**2


class MottScattering:
    """
    Mott scattering (relativistic Coulomb scattering).

    Extension of Rutherford formula to include relativistic
    and spin effects for electron scattering.

    dσ/dΩ = (dσ/dΩ)_R × [1 - β² sin²(θ/2)]

    Args:
        Z: Target charge number
        energy: Electron kinetic energy (J or MeV)
        use_MeV: If True, energy is in MeV
    """

    def __init__(self, Z: int, energy: float, use_MeV: bool = True):
        if energy <= 0:
            raise ValueError("Energy must be positive")

        self.Z = Z
        self.energy = energy * MeV if use_MeV else energy

    @property
    def beta(self) -> float:
        """Relativistic velocity β = v/c."""
        E_total = self.energy + m_e * c**2
        gamma = E_total / (m_e * c**2)
        return np.sqrt(1 - 1 / gamma**2)

    @property
    def gamma(self) -> float:
        """Lorentz factor γ."""
        E_total = self.energy + m_e * c**2
        return E_total / (m_e * c**2)

    def differential_cross_section(self, theta: ArrayLike) -> np.ndarray:
        """
        Mott differential cross section.

        Args:
            theta: Scattering angle(s)

        Returns:
            Differential cross section
        """
        theta = np.asarray(theta)
        beta = self.beta

        # Rutherford prefactor (for electron)
        E_total = self.energy + m_e * c**2
        p = np.sqrt(E_total**2 - (m_e * c**2)**2) / c

        a = self.Z * e**2 / (4 * np.pi * epsilon_0 * 2 * self.energy)

        dcs_ruth = (a / 2)**2 / np.sin(theta / 2)**4

        # Mott correction
        mott_factor = 1 - beta**2 * np.sin(theta / 2)**2

        return dcs_ruth * mott_factor

    def form_factor_correction(self, theta: ArrayLike, R: float) -> np.ndarray:
        """
        Nuclear form factor correction for extended charge.

        F(q) = 3(sin(qR) - qR cos(qR))/(qR)³

        Args:
            theta: Scattering angle(s)
            R: Nuclear radius

        Returns:
            Form factor |F(q)|²
        """
        theta = np.asarray(theta)

        # Momentum transfer
        p = np.sqrt(2 * m_e * self.energy)  # Non-relativistic approx
        q = 2 * p * np.sin(theta / 2) / hbar

        qR = q * R

        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            F = 3 * (np.sin(qR) - qR * np.cos(qR)) / qR**3
            F = np.where(qR < 1e-6, 1.0, F)

        return F**2


# =============================================================================
# Nuclear Structure
# =============================================================================

class LiquidDropModel:
    """
    Semi-empirical mass formula (liquid drop model).

    The binding energy is given by:
    B(A,Z) = a_V A - a_S A^(2/3) - a_C Z²/A^(1/3) - a_A (A-2Z)²/A + δ(A,Z)

    Args:
        A: Mass number
        Z: Atomic number
    """

    # Empirical coefficients (MeV)
    a_V = 15.8   # Volume term
    a_S = 18.3   # Surface term
    a_C = 0.714  # Coulomb term
    a_A = 23.2   # Asymmetry term
    a_P = 12.0   # Pairing term

    def __init__(self, A: int, Z: int):
        if A < 1:
            raise ValueError("Mass number must be positive")
        if Z < 0 or Z > A:
            raise ValueError("Invalid atomic number")

        self.A = A
        self.Z = Z
        self.N = A - Z

    def binding_energy(self) -> float:
        """
        Calculate binding energy in MeV.

        Returns:
            Binding energy B(A,Z)
        """
        A, Z = self.A, self.Z
        N = self.N

        # Volume term
        B = self.a_V * A

        # Surface term
        B -= self.a_S * A**(2/3)

        # Coulomb term
        B -= self.a_C * Z**2 / A**(1/3)

        # Asymmetry term
        B -= self.a_A * (A - 2*Z)**2 / A

        # Pairing term
        B += self.pairing_term()

        return B

    def pairing_term(self) -> float:
        """
        Pairing energy contribution.

        δ = +a_P/A^(1/2) for even-even
        δ = 0 for odd-A
        δ = -a_P/A^(1/2) for odd-odd
        """
        Z_even = self.Z % 2 == 0
        N_even = self.N % 2 == 0

        if Z_even and N_even:
            return self.a_P / np.sqrt(self.A)
        elif not Z_even and not N_even:
            return -self.a_P / np.sqrt(self.A)
        else:
            return 0

    def mass_excess(self) -> float:
        """
        Mass excess Δ = M - Au.

        Returns:
            Mass excess in MeV
        """
        B = self.binding_energy()
        # M = Z*m_p + N*m_n - B/c²
        # Δ = M - A*u = (Z-A)*m_p + N*m_n - B/c² + (N-Z)*(m_n-m_p)

        # In atomic mass units and MeV:
        # Δ = Z*Δ_H + N*Δ_n - B

        Delta_H = 7.289  # MeV (hydrogen)
        Delta_n = 8.071  # MeV (neutron)

        return self.Z * Delta_H + self.N * Delta_n - B

    def binding_energy_per_nucleon(self) -> float:
        """B/A in MeV."""
        return self.binding_energy() / self.A

    def neutron_separation_energy(self) -> float:
        """
        One-neutron separation energy S_n.

        S_n = B(A,Z) - B(A-1,Z)
        """
        B_AZ = self.binding_energy()

        if self.A > 1:
            ldm_Am1 = LiquidDropModel(self.A - 1, self.Z)
            B_Am1 = ldm_Am1.binding_energy()
            return B_AZ - B_Am1
        return B_AZ

    def proton_separation_energy(self) -> float:
        """
        One-proton separation energy S_p.

        S_p = B(A,Z) - B(A-1,Z-1)
        """
        B_AZ = self.binding_energy()

        if self.A > 1 and self.Z > 0:
            ldm_Am1 = LiquidDropModel(self.A - 1, self.Z - 1)
            B_Am1 = ldm_Am1.binding_energy()
            return B_AZ - B_Am1
        return B_AZ

    def fissility_parameter(self) -> float:
        """
        Fissility parameter x = E_C / (2 E_S).

        x > 1 leads to spontaneous fission.
        """
        E_C = self.a_C * self.Z**2 / self.A**(1/3)
        E_S = self.a_S * self.A**(2/3)
        return E_C / (2 * E_S)

    @classmethod
    def most_stable_Z(cls, A: int) -> int:
        """
        Find most stable Z for given A (valley of stability).

        Args:
            A: Mass number

        Returns:
            Most stable atomic number
        """
        # Minimize binding energy w.r.t. Z
        # Z_opt = A/(2 + a_C A^(2/3)/(2 a_A))

        Z_opt = A / (2 + cls.a_C * A**(2/3) / (2 * cls.a_A))
        return round(Z_opt)


class ShellModel:
    """
    Nuclear shell model.

    Predicts magic numbers and spin-parity from single-particle
    levels in a central potential with spin-orbit coupling.

    Magic numbers: 2, 8, 20, 28, 50, 82, 126

    Args:
        A: Mass number
        Z: Atomic number
    """

    MAGIC_NUMBERS = [2, 8, 20, 28, 50, 82, 126]

    # Single-particle levels (level, 2j, degeneracy)
    # Format: (n, l, 2j) where level is 1s, 1p, etc.
    SP_LEVELS = [
        ('1s1/2', 1, 2),
        ('1p3/2', 3, 4),
        ('1p1/2', 1, 2),
        ('1d5/2', 5, 6),
        ('2s1/2', 1, 2),
        ('1d3/2', 3, 4),
        ('1f7/2', 7, 8),  # Shell closure at 28
        ('2p3/2', 3, 4),
        ('1f5/2', 5, 6),
        ('2p1/2', 1, 2),
        ('1g9/2', 9, 10), # Shell closure at 50
        ('1g7/2', 7, 8),
        ('2d5/2', 5, 6),
        ('2d3/2', 3, 4),
        ('3s1/2', 1, 2),
        ('1h11/2', 11, 12), # Shell closure at 82
    ]

    def __init__(self, A: int, Z: int):
        if A < 1:
            raise ValueError("Mass number must be positive")
        if Z < 0 or Z > A:
            raise ValueError("Invalid atomic number")

        self.A = A
        self.Z = Z
        self.N = A - Z

    def is_magic(self, num: int) -> bool:
        """Check if number is magic."""
        return num in self.MAGIC_NUMBERS

    @property
    def is_doubly_magic(self) -> bool:
        """Check if nucleus is doubly magic."""
        return self.is_magic(self.Z) and self.is_magic(self.N)

    def shell_closure_distance(self) -> Tuple[int, int]:
        """
        Distance to nearest shell closure for protons and neutrons.

        Returns:
            (proton_distance, neutron_distance)
        """
        def dist_to_magic(n):
            if n in self.MAGIC_NUMBERS:
                return 0
            dists = [abs(n - m) for m in self.MAGIC_NUMBERS]
            return min(dists)

        return (dist_to_magic(self.Z), dist_to_magic(self.N))

    def ground_state_spin(self) -> str:
        """
        Predict ground state spin-parity.

        For odd-A nuclei, determined by unpaired nucleon.
        For even-even, always 0+.

        Returns:
            Spin-parity as string (e.g., "5/2+")
        """
        if self.Z % 2 == 0 and self.N % 2 == 0:
            return "0+"

        # Odd-A or odd-odd: need to find last filled level
        # Simplified: just indicate odd-A behavior

        if self.A % 2 == 1:
            # Odd-A: spin from unpaired nucleon
            return "J+"  # Would need full calculation

        return "J+"  # Odd-odd: coupling of both unpaired nucleons

    def shell_correction_energy(self) -> float:
        """
        Estimate shell correction to liquid drop mass.

        Returns:
            Shell correction in MeV (negative for magic nuclei)
        """
        # Simple estimate based on distance to magic numbers
        d_Z, d_N = self.shell_closure_distance()

        if d_Z == 0 and d_N == 0:
            return -10.0  # Doubly magic
        elif d_Z == 0 or d_N == 0:
            return -5.0   # Singly magic
        else:
            # Diminishes away from magic numbers
            return 2.0 * (d_Z + d_N) / (d_Z + d_N + 10)


class WoodsSaxon:
    """
    Woods-Saxon nuclear potential.

    V(r) = -V₀ / (1 + exp((r-R)/a))

    A common parameterization for the nuclear mean field.

    Args:
        V0: Well depth (MeV)
        R: Nuclear radius (fm)
        a: Surface diffuseness (fm)
    """

    def __init__(self, V0: float = 50.0, R: float = None,
                 a: float = 0.65, A: int = None):
        if V0 <= 0:
            raise ValueError("Well depth must be positive")

        self.V0 = V0
        self.a = a * fm

        if R is not None:
            self.R = R * fm
        elif A is not None:
            self.R = r0 * A**(1/3)
        else:
            raise ValueError("Must specify R or A")

    def potential(self, r: ArrayLike) -> np.ndarray:
        """
        Woods-Saxon potential V(r).

        Args:
            r: Radial distance

        Returns:
            Potential in MeV
        """
        r = np.asarray(r)
        return -self.V0 / (1 + np.exp((r - self.R) / self.a))

    def derivative(self, r: ArrayLike) -> np.ndarray:
        """
        Derivative dV/dr.

        Args:
            r: Radial distance

        Returns:
            Potential derivative
        """
        r = np.asarray(r)
        exp_term = np.exp((r - self.R) / self.a)
        return self.V0 * exp_term / (self.a * (1 + exp_term)**2)

    def spin_orbit_potential(self, r: ArrayLike, V_so: float = 20.0) -> np.ndarray:
        """
        Spin-orbit potential proportional to (1/r) dV/dr.

        V_so(r) = V_so * (1/r) dV_WS/dr

        Args:
            r: Radial distance
            V_so: Spin-orbit strength (MeV)

        Returns:
            Spin-orbit potential
        """
        r = np.asarray(r)
        # Avoid division by zero
        r_safe = np.where(r > 1e-15, r, 1e-15)
        return V_so * self.derivative(r) / r_safe

    def density(self, r: ArrayLike, rho0: float = 0.16) -> np.ndarray:
        """
        Nuclear density with Woods-Saxon form.

        ρ(r) = ρ₀ / (1 + exp((r-R)/a))

        Args:
            r: Radial distance
            rho0: Central density (fm⁻³)

        Returns:
            Density
        """
        r = np.asarray(r)
        return rho0 / (1 + np.exp((r - self.R) / self.a))


class NuclearRadius:
    """
    Nuclear radius systematics.

    R = r₀ A^(1/3)

    where r₀ ≈ 1.2 fm for charge radius.

    Args:
        A: Mass number
    """

    def __init__(self, A: int):
        if A < 1:
            raise ValueError("Mass number must be positive")
        self.A = A

    def charge_radius(self, r0: float = 1.2) -> float:
        """
        RMS charge radius.

        Args:
            r0: Radius parameter (fm)

        Returns:
            Charge radius in fm
        """
        return r0 * self.A**(1/3)

    def matter_radius(self, r0: float = 1.15) -> float:
        """
        Nuclear matter radius.

        Args:
            r0: Radius parameter (fm)

        Returns:
            Matter radius in fm
        """
        return r0 * self.A**(1/3)

    def skin_thickness(self) -> float:
        """
        Estimate neutron skin thickness for heavy nuclei.

        Returns:
            Skin thickness in fm
        """
        # Rough estimate based on asymmetry
        return 0.1 * fm

    def central_density(self) -> float:
        """
        Nuclear matter density at center.

        ρ₀ ≈ 0.16 fm⁻³

        Returns:
            Central density in fm⁻³
        """
        return 0.16  # fm^-3

    def binding_radius(self) -> float:
        """
        Radius where binding energy density is maximum.

        Returns:
            Binding radius in fm
        """
        return 1.12 * self.A**(1/3)


class NuclearSpin:
    """
    Nuclear spin and angular momentum.

    Implements angular momentum coupling for nucleons.

    Args:
        J: Total spin quantum number
        parity: Parity (+1 or -1)
    """

    def __init__(self, J: float, parity: int = 1):
        if J < 0:
            raise ValueError("Spin must be non-negative")
        if parity not in [1, -1]:
            raise ValueError("Parity must be +1 or -1")

        self.J = J
        self.parity = parity

    def __str__(self) -> str:
        """String representation like '5/2+'."""
        if self.J == int(self.J):
            J_str = str(int(self.J))
        else:
            J_str = f"{int(2*self.J)}/2"

        parity_str = "+" if self.parity == 1 else "-"
        return f"{J_str}{parity_str}"

    def magnetic_moment_schmidt(self, l: int, nucleon: str = 'proton') -> float:
        """
        Schmidt magnetic moment.

        For j = l + 1/2: μ = g_s/2 + (j - 1/2)g_l
        For j = l - 1/2: μ = j(g_l - g_s/2)/(j+1) + jg_l/(j+1)

        Args:
            l: Orbital angular momentum
            nucleon: 'proton' or 'neutron'

        Returns:
            Magnetic moment in nuclear magnetons
        """
        if nucleon == 'proton':
            g_l = 1.0
            g_s = 5.586
        else:
            g_l = 0.0
            g_s = -3.826

        j = self.J

        if j == l + 0.5:
            return g_s / 2 + (j - 0.5) * g_l
        else:  # j = l - 1/2
            return j * (g_l - g_s / 2) / (j + 1) + j * g_l / (j + 1)

    def quadrupole_moment(self, radius: float, single_particle: bool = True) -> float:
        """
        Electric quadrupole moment.

        For a single particle:
        Q = -(2j-1)/(2j+2) * e * r²

        Args:
            radius: Nuclear radius
            single_particle: If True, use single-particle estimate

        Returns:
            Quadrupole moment
        """
        j = self.J

        if j < 1:
            return 0

        if single_particle:
            return -(2*j - 1) / (2*j + 2) * radius**2

        return 0


# =============================================================================
# Radioactivity
# =============================================================================

class AlphaDecay:
    """
    Alpha decay via quantum tunneling.

    The Geiger-Nuttall law relates decay constant to Q-value:
    log(λ) = a + b/√Q

    This is explained by quantum tunneling through the
    Coulomb barrier.

    Args:
        Z_parent: Parent atomic number
        A_parent: Parent mass number
        Q_value: Q-value in MeV
    """

    def __init__(self, Z_parent: int, A_parent: int, Q_value: float):
        if Q_value <= 0:
            raise ValueError("Q-value must be positive for decay")

        self.Z_p = Z_parent
        self.A_p = A_parent
        self.Q = Q_value * MeV  # Convert to Joules

        # Daughter after alpha emission
        self.Z_d = Z_parent - 2
        self.A_d = A_parent - 4

    @property
    def coulomb_barrier_height(self) -> float:
        """
        Coulomb barrier height V_C.

        V_C = Z_d Z_α e² / (4πε₀ R)

        Returns:
            Barrier height in MeV
        """
        Z_alpha = 2
        R = r0 * (self.A_d**(1/3) + 4**(1/3))  # Touching sphere radius

        V_C = self.Z_d * Z_alpha * e**2 / (4 * np.pi * epsilon_0 * R)
        return V_C / MeV

    @property
    def gamow_factor(self) -> float:
        """
        Gamow factor for tunneling probability.

        G = 2π η

        where η is the Sommerfeld parameter.
        """
        Z_alpha = 2
        m_alpha = 4 * u  # Alpha mass

        # Alpha kinetic energy ≈ Q-value (mass difference small correction)
        E = self.Q
        v = np.sqrt(2 * E / m_alpha)

        eta = self.Z_d * Z_alpha * e**2 / (4 * np.pi * epsilon_0 * hbar * v)
        return 2 * np.pi * eta

    def decay_constant(self) -> float:
        """
        Decay constant λ.

        Uses semi-classical formula with preformation factor.

        Returns:
            Decay constant (s⁻¹)
        """
        R = r0 * self.A_d**(1/3)

        # Assault frequency
        m_alpha = 4 * u
        v = np.sqrt(2 * self.Q / m_alpha)
        f = v / (2 * R)

        # Tunneling probability
        P = np.exp(-self.gamow_factor)

        # Preformation probability (rough estimate)
        S_alpha = 0.1

        return f * P * S_alpha

    def half_life(self) -> float:
        """
        Half-life t_1/2 = ln(2)/λ.

        Returns:
            Half-life in seconds
        """
        return np.log(2) / self.decay_constant()

    def geiger_nuttall(self) -> Tuple[float, float]:
        """
        Geiger-Nuttall law parameters.

        log₁₀(λ) = a + b * Z / √Q

        Returns:
            (a, b) coefficients
        """
        # Empirical values
        a = 28.0
        b = -1.5
        return (a, b)


class BetaDecay:
    """
    Beta decay (Fermi theory).

    Beta-minus: n → p + e⁻ + ν̄_e
    Beta-plus: p → n + e⁺ + ν_e

    Args:
        decay_type: 'minus' or 'plus'
        Q_value: Q-value in MeV
        Z_daughter: Daughter atomic number
        ft_value: ft value (for lifetime calculation)
    """

    def __init__(self, decay_type: str, Q_value: float,
                 Z_daughter: int, ft_value: float = 3000):
        if decay_type not in ['minus', 'plus']:
            raise ValueError("decay_type must be 'minus' or 'plus'")
        if Q_value <= 0:
            raise ValueError("Q-value must be positive")

        self.decay_type = decay_type
        self.Q = Q_value * MeV
        self.Z_d = Z_daughter
        self.ft = ft_value

    def fermi_function(self, W: float) -> float:
        """
        Fermi function F(Z,W) for Coulomb correction.

        F ≈ 2πη / (1 - exp(-2πη))

        where η = ±αZ/β

        Args:
            W: Total electron energy in electron mass units

        Returns:
            Fermi function
        """
        if W <= 1:
            return 0

        p = np.sqrt(W**2 - 1)
        beta = p / W

        sign = 1 if self.decay_type == 'minus' else -1
        eta = sign * alpha_fine * self.Z_d / beta

        return 2 * np.pi * eta / (1 - np.exp(-2 * np.pi * eta))

    def spectrum(self, T_e: ArrayLike) -> np.ndarray:
        """
        Beta electron kinetic energy spectrum.

        N(T) ∝ p E (Q-T)² F(Z,E)

        Args:
            T_e: Electron kinetic energy in MeV

        Returns:
            Spectrum (unnormalized)
        """
        T_e = np.asarray(T_e)
        Q_MeV = self.Q / MeV
        m_e_MeV = m_e * c**2 / MeV

        # Total energy in electron mass units
        W = (T_e + m_e_MeV) / m_e_MeV

        # Phase space
        p = np.sqrt(np.maximum(W**2 - 1, 0))
        E_nu = np.maximum(Q_MeV - T_e, 0)

        # Fermi function
        F = np.array([self.fermi_function(w) for w in W])

        return p * W * E_nu**2 * F

    def average_energy(self) -> float:
        """
        Average electron kinetic energy.

        <T_e> ≈ Q/3 for allowed decays

        Returns:
            Average energy in MeV
        """
        return self.Q / MeV / 3

    def half_life(self) -> float:
        """
        Half-life from ft value.

        t_1/2 = ft / f

        where f is the Fermi integral.

        Returns:
            Half-life in seconds
        """
        # Fermi integral (approximate)
        Q_MeV = self.Q / MeV
        W0 = (Q_MeV + m_e * c**2 / MeV) / (m_e * c**2 / MeV)

        # Simple approximation for f
        f = (W0**5 - 1) / 30

        return self.ft / f


class GammaDecay:
    """
    Gamma decay (electromagnetic transitions).

    Nuclei in excited states can decay by emitting gamma rays.
    The transition rate depends on multipolarity and energy.

    Args:
        energy: Gamma-ray energy in MeV
        multipolarity: 'E1', 'M1', 'E2', 'M2', etc.
        A: Mass number (for single-particle estimates)
    """

    def __init__(self, energy: float, multipolarity: str, A: int = 100):
        if energy <= 0:
            raise ValueError("Energy must be positive")

        self.E_gamma = energy * MeV
        self.multipolarity = multipolarity
        self.A = A

        # Parse multipolarity
        self.type = multipolarity[0]  # 'E' or 'M'
        self.L = int(multipolarity[1])  # Angular momentum

    def weisskopf_estimate(self) -> float:
        """
        Weisskopf single-particle estimate for transition rate.

        Returns:
            Transition rate in s⁻¹
        """
        E = self.E_gamma / MeV  # Energy in MeV
        L = self.L
        R = 1.2 * self.A**(1/3)  # Nuclear radius in fm

        if self.type == 'E':
            # Electric multipole
            if L == 1:
                return 1.0e14 * E**3 * self.A**(2/3)
            elif L == 2:
                return 7.3e7 * E**5 * self.A**(4/3)
            elif L == 3:
                return 3.4e1 * E**7 * self.A**2
            elif L == 4:
                return 1.1e-5 * E**9 * self.A**(8/3)
        else:  # Magnetic
            if L == 1:
                return 3.1e13 * E**3
            elif L == 2:
                return 2.2e7 * E**5 * self.A**(2/3)
            elif L == 3:
                return 1.0e1 * E**7 * self.A**(4/3)
            elif L == 4:
                return 3.3e-6 * E**9 * self.A**2

        return 0

    def half_life(self) -> float:
        """
        Half-life from Weisskopf estimate.

        Returns:
            Half-life in seconds
        """
        rate = self.weisskopf_estimate()
        if rate > 0:
            return np.log(2) / rate
        return np.inf

    def internal_conversion_coefficient(self, shell: str = 'K') -> float:
        """
        Internal conversion coefficient α.

        α = λ_IC / λ_γ

        Approximate scaling with Z and energy.

        Args:
            shell: Atomic shell ('K', 'L', etc.)

        Returns:
            Conversion coefficient
        """
        # Very rough estimate
        # α ∝ Z³ / E_γ³ for K shell

        Z = self.A * 0.4  # Rough Z from A
        E = self.E_gamma / MeV

        if shell == 'K':
            return 0.1 * (Z / 50)**3 * (1 / E)**3
        elif shell == 'L':
            return 0.01 * (Z / 50)**3 * (1 / E)**3
        return 0


class DecayChain:
    """
    Radioactive decay chains (Bateman equations).

    Solves the coupled differential equations for sequential decay:
    dN_i/dt = λ_{i-1} N_{i-1} - λ_i N_i

    Args:
        half_lives: List of half-lives in seconds
        initial_amounts: Initial number of nuclei for each species
    """

    def __init__(self, half_lives: List[float],
                 initial_amounts: Optional[List[float]] = None):
        if any(t <= 0 for t in half_lives):
            raise ValueError("All half-lives must be positive")

        self.half_lives = np.array(half_lives)
        self.lambdas = np.log(2) / self.half_lives

        n = len(half_lives) + 1  # Include stable end product

        if initial_amounts is None:
            self.N0 = np.zeros(n)
            self.N0[0] = 1.0
        else:
            self.N0 = np.array(initial_amounts)

    def bateman_solution(self, t: ArrayLike, species: int) -> np.ndarray:
        """
        Bateman analytical solution for species i.

        N_i(t) = N_1(0) Σ_j c_ij exp(-λ_j t)

        Args:
            t: Time array
            species: Species index (0-indexed)

        Returns:
            Number of nuclei vs time
        """
        t = np.asarray(t)
        n = len(self.lambdas)

        if species == 0:
            return self.N0[0] * np.exp(-self.lambdas[0] * t)

        if species > n:
            # Stable end product
            return self.N0[0] * (1 - np.exp(-self.lambdas[0] * t))

        # Bateman formula for intermediate species
        N = np.zeros_like(t)

        for j in range(species + 1):
            c_ij = 1.0
            for k in range(species + 1):
                if k != j:
                    c_ij *= self.lambdas[k] / (self.lambdas[k] - self.lambdas[j])

            N += c_ij * np.exp(-self.lambdas[j] * t)

        # Product of decay constants
        prod_lambda = np.prod(self.lambdas[:species])

        return self.N0[0] * prod_lambda * N

    def solve_numerical(self, t_span: Tuple[float, float],
                       n_points: int = 1000) -> Dict[str, np.ndarray]:
        """
        Numerical solution of decay chain equations.

        Args:
            t_span: (t_start, t_end)
            n_points: Number of output points

        Returns:
            Dictionary with time and all species populations
        """
        n = len(self.lambdas) + 1

        def equations(t, N):
            dNdt = np.zeros(n)
            dNdt[0] = -self.lambdas[0] * N[0]

            for i in range(1, n - 1):
                dNdt[i] = self.lambdas[i-1] * N[i-1] - self.lambdas[i] * N[i]

            dNdt[-1] = self.lambdas[-1] * N[-2]  # Stable end product
            return dNdt

        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        sol = integrate.solve_ivp(equations, t_span, self.N0,
                                  t_eval=t_eval, method='LSODA')

        return {
            't': sol.t,
            'N': sol.y
        }

    def secular_equilibrium_time(self, daughter: int = 1) -> float:
        """
        Time to reach secular equilibrium.

        Approximately 5-7 daughter half-lives.

        Args:
            daughter: Daughter species index

        Returns:
            Time in seconds
        """
        return 7 * self.half_lives[daughter]

    def activity(self, t: ArrayLike, species: int = 0) -> np.ndarray:
        """
        Activity A = λN.

        Args:
            t: Time array
            species: Species index

        Returns:
            Activity vs time
        """
        N = self.bateman_solution(t, species)
        return self.lambdas[species] * N


class HalfLife:
    """
    Radioactive decay statistics.

    Args:
        half_life: Half-life in seconds
    """

    def __init__(self, half_life: float):
        if half_life <= 0:
            raise ValueError("Half-life must be positive")

        self.t_half = half_life
        self.decay_constant = np.log(2) / half_life
        self.mean_lifetime = 1 / self.decay_constant

    def remaining_fraction(self, t: ArrayLike) -> np.ndarray:
        """
        Fraction of nuclei remaining after time t.

        N(t)/N_0 = exp(-λt) = (1/2)^(t/t_1/2)

        Args:
            t: Time

        Returns:
            Remaining fraction
        """
        t = np.asarray(t)
        return np.exp(-self.decay_constant * t)

    def activity(self, N0: float, t: ArrayLike = 0) -> np.ndarray:
        """
        Activity A = λN.

        Args:
            N0: Initial number of nuclei
            t: Time (default: 0)

        Returns:
            Activity in decays per second
        """
        t = np.asarray(t)
        N = N0 * self.remaining_fraction(t)
        return self.decay_constant * N

    def time_for_fraction(self, fraction: float) -> float:
        """
        Time required for given fraction to decay.

        Args:
            fraction: Fraction decayed (0 to 1)

        Returns:
            Time in seconds
        """
        if fraction <= 0 or fraction >= 1:
            raise ValueError("Fraction must be in (0, 1)")

        return -np.log(1 - fraction) / self.decay_constant

    def probability_decay_in_interval(self, t1: float, t2: float) -> float:
        """
        Probability of decay in time interval [t1, t2].

        P = exp(-λt1) - exp(-λt2)

        Args:
            t1, t2: Time interval

        Returns:
            Decay probability
        """
        return np.exp(-self.decay_constant * t1) - np.exp(-self.decay_constant * t2)


# =============================================================================
# Nuclear Reactions
# =============================================================================

class NuclearCrossSection:
    """
    Nuclear reaction cross sections.

    σ(E) describes the probability of a nuclear reaction.

    Args:
        energy_data: Energy array (MeV)
        cross_section_data: Cross section array (barns)
    """

    def __init__(self, energy_data: ArrayLike, cross_section_data: ArrayLike):
        self.E = np.asarray(energy_data)
        self.sigma = np.asarray(cross_section_data)

        if len(self.E) != len(self.sigma):
            raise ValueError("Energy and cross section arrays must have same length")

    def __call__(self, E: ArrayLike) -> np.ndarray:
        """Interpolate cross section at energy E."""
        E = np.asarray(E)
        return np.interp(E, self.E, self.sigma)

    def astrophysical_S_factor(self) -> np.ndarray:
        """
        Astrophysical S-factor.

        S(E) = σ(E) E exp(2πη)

        where η is the Sommerfeld parameter.

        Returns:
            S-factor array (keV·barn)
        """
        # Assume proton + nucleus for Sommerfeld parameter
        # Would need Z1, Z2, reduced mass for proper calculation
        return self.sigma * self.E

    def maxwellian_average(self, temperature: float) -> float:
        """
        Maxwellian-averaged cross section.

        <σv> = (8/πμ)^(1/2) (k_B T)^(-3/2) ∫ σ(E) E exp(-E/k_B T) dE

        Args:
            temperature: Temperature in K

        Returns:
            Reaction rate <σv> (m³/s)
        """
        kT = k_B * temperature / MeV  # kT in MeV

        # Integration
        E_MeV = self.E
        sigma_m2 = self.sigma * 1e-28  # barns to m²

        integrand = sigma_m2 * E_MeV * MeV * np.exp(-E_MeV / kT)
        integral = np.trapezoid(integrand, E_MeV)

        # Normalization
        mu = m_p / 2  # Rough estimate of reduced mass
        prefactor = np.sqrt(8 / (np.pi * mu)) * (kT * MeV)**(-3/2)

        return prefactor * integral


class QValue:
    """
    Q-value calculations for nuclear reactions.

    Q = (Σ M_initial - Σ M_final) c²

    Args:
        reactants: List of (A, Z) tuples for reactants
        products: List of (A, Z) tuples for products
    """

    def __init__(self, reactants: List[Tuple[int, int]],
                 products: List[Tuple[int, int]]):
        self.reactants = reactants
        self.products = products

    def calculate(self) -> float:
        """
        Calculate Q-value using liquid drop model.

        Returns:
            Q-value in MeV
        """
        # Sum binding energies
        B_reactants = sum(
            LiquidDropModel(A, Z).binding_energy()
            for A, Z in self.reactants
        )

        B_products = sum(
            LiquidDropModel(A, Z).binding_energy()
            for A, Z in self.products
        )

        return B_products - B_reactants

    @property
    def is_exothermic(self) -> bool:
        """Check if reaction is exothermic (Q > 0)."""
        return self.calculate() > 0

    def threshold_energy(self, projectile_A: int, target_A: int) -> float:
        """
        Threshold kinetic energy for endothermic reaction.

        E_th = -Q (1 + m_p/m_t + Q/(2 m_t c²))

        Args:
            projectile_A: Projectile mass number
            target_A: Target mass number

        Returns:
            Threshold energy in MeV
        """
        Q = self.calculate()

        if Q >= 0:
            return 0

        m_p = projectile_A * u
        m_t = target_A * u

        return -Q * (1 + m_p / m_t + Q * MeV / (2 * m_t * c**2)) / MeV


class ResonanceFormula:
    """
    Breit-Wigner resonance formula.

    σ(E) = π ℓ² g_J Γ_in Γ_out / ((E - E_R)² + Γ²/4)

    Args:
        E_resonance: Resonance energy (MeV)
        total_width: Total width Γ (MeV)
        partial_widths: Dict of partial widths {'in': Γ_in, 'out': Γ_out}
        spin_factor: Statistical factor g_J
    """

    def __init__(self, E_resonance: float, total_width: float,
                 partial_widths: Dict[str, float], spin_factor: float = 1.0):
        if E_resonance <= 0:
            raise ValueError("Resonance energy must be positive")
        if total_width <= 0:
            raise ValueError("Width must be positive")

        self.E_R = E_resonance
        self.Gamma = total_width
        self.Gamma_in = partial_widths.get('in', total_width / 2)
        self.Gamma_out = partial_widths.get('out', total_width / 2)
        self.g = spin_factor

    def cross_section(self, E: ArrayLike, reduced_mass: float = m_p) -> np.ndarray:
        """
        Breit-Wigner cross section.

        Args:
            E: Energy in MeV
            reduced_mass: Reduced mass of system

        Returns:
            Cross section in barns
        """
        E = np.asarray(E)

        # de Broglie wavelength
        p = np.sqrt(2 * reduced_mass * E * MeV)
        lambda_bar = hbar / p

        # Breit-Wigner
        numerator = self.g * self.Gamma_in * self.Gamma_out
        denominator = (E - self.E_R)**2 + (self.Gamma / 2)**2

        sigma_m2 = np.pi * lambda_bar**2 * numerator / denominator
        return sigma_m2 / 1e-28  # Convert to barns

    def peak_cross_section(self, reduced_mass: float = m_p) -> float:
        """
        Cross section at resonance peak.

        σ_peak = π ℓ² g Γ_in Γ_out / (Γ/2)²

        Returns:
            Peak cross section in barns
        """
        return self.cross_section(self.E_R, reduced_mass)

    def integrated_cross_section(self) -> float:
        """
        Area under resonance curve.

        ∫ σ dE = 2π² ℓ² g Γ_in Γ_out / Γ

        Returns:
            Integrated cross section (barn·MeV)
        """
        # At resonance energy
        p = np.sqrt(2 * m_p * self.E_R * MeV)
        lambda_bar = hbar / p

        sigma_int = 2 * np.pi**2 * lambda_bar**2 * self.g
        sigma_int *= self.Gamma_in * self.Gamma_out / self.Gamma

        return sigma_int / 1e-28  # barns


class CompoundNucleus:
    """
    Compound nucleus statistical model.

    The compound nucleus model assumes formation and decay
    are independent processes.

    Args:
        A: Compound nucleus mass number
        Z: Compound nucleus atomic number
        excitation_energy: Excitation energy (MeV)
    """

    def __init__(self, A: int, Z: int, excitation_energy: float):
        if A < 1:
            raise ValueError("Mass number must be positive")
        if excitation_energy <= 0:
            raise ValueError("Excitation energy must be positive")

        self.A = A
        self.Z = Z
        self.E_x = excitation_energy

    def level_density_parameter(self) -> float:
        """
        Level density parameter a ≈ A/8 MeV⁻¹.

        Returns:
            Level density parameter
        """
        return self.A / 8

    def level_density(self, E: float = None) -> float:
        """
        Nuclear level density ρ(E).

        ρ(E) ∝ exp(2√(aE)) / E

        Args:
            E: Excitation energy (default: self.E_x)

        Returns:
            Level density
        """
        if E is None:
            E = self.E_x

        a = self.level_density_parameter()
        return np.exp(2 * np.sqrt(a * E)) / E

    def nuclear_temperature(self) -> float:
        """
        Nuclear temperature T = √(E/a).

        Returns:
            Temperature in MeV
        """
        a = self.level_density_parameter()
        return np.sqrt(self.E_x / a)

    def evaporation_spectrum(self, particle: str = 'n') -> Callable:
        """
        Evaporation spectrum for emitted particle.

        dN/dε ∝ ε σ_inv(ε) ρ(E* - B - ε)

        Args:
            particle: 'n', 'p', or 'alpha'

        Returns:
            Function giving spectrum
        """
        # Binding energies (rough)
        B = {'n': 8.0, 'p': 8.0, 'alpha': 7.0}
        binding = B.get(particle, 8.0)

        T = self.nuclear_temperature()

        def spectrum(epsilon):
            if epsilon + binding > self.E_x:
                return 0
            return epsilon * np.exp(-epsilon / T)

        return spectrum


class FissionYield:
    """
    Fission product yield distribution.

    Models the mass distribution of fission products.

    Args:
        A_fissioning: Mass number of fissioning nucleus
        E_excitation: Excitation energy (MeV)
    """

    def __init__(self, A_fissioning: int, E_excitation: float = 6.0):
        self.A_f = A_fissioning
        self.E_x = E_excitation

    def yield_distribution(self, A: ArrayLike) -> np.ndarray:
        """
        Mass yield Y(A) for fission products.

        Double-humped distribution for low-energy fission.

        Args:
            A: Mass number array

        Returns:
            Yield (probability)
        """
        A = np.asarray(A)

        # Symmetric component
        A_sym = self.A_f / 2
        sigma_sym = 7

        Y_sym = np.exp(-(A - A_sym)**2 / (2 * sigma_sym**2))

        # Asymmetric components (heavy and light peaks)
        A_heavy = self.A_f * 0.58
        A_light = self.A_f * 0.42
        sigma_asym = 5

        Y_heavy = np.exp(-(A - A_heavy)**2 / (2 * sigma_asym**2))
        Y_light = np.exp(-(A - A_light)**2 / (2 * sigma_asym**2))

        # Combined
        Y = 0.3 * Y_sym + 0.35 * Y_heavy + 0.35 * Y_light

        # Normalize
        return Y / np.sum(Y)

    def peak_positions(self) -> Tuple[float, float]:
        """
        Positions of heavy and light peaks.

        Returns:
            (A_light, A_heavy)
        """
        return (self.A_f * 0.42, self.A_f * 0.58)

    def average_neutrons(self) -> float:
        """
        Average prompt neutrons per fission ν̄.

        Empirical formula for thermal fission.

        Returns:
            Average neutron number
        """
        # ν ≈ 0.08 + 0.0075(A - 230) + 0.14 E_x
        return 0.08 + 0.0075 * (self.A_f - 230) + 0.14 * self.E_x

    def energy_release(self) -> float:
        """
        Total energy release per fission.

        Returns:
            Energy in MeV
        """
        # Q ≈ 200 MeV for U-235
        return 200.0  # MeV


class FusionRate:
    """
    Fusion reaction rates.

    <σv> for fusion reactions as function of temperature.

    Args:
        reaction: 'DD', 'DT', or 'DHe3'
    """

    def __init__(self, reaction: str = 'DT'):
        if reaction not in ['DD', 'DT', 'DHe3']:
            raise ValueError("Unknown reaction")

        self.reaction = reaction

    def reactivity(self, T: ArrayLike) -> np.ndarray:
        """
        Reactivity <σv> as function of temperature.

        Uses parameterized fits.

        Args:
            T: Temperature in keV

        Returns:
            Reactivity in m³/s
        """
        T = np.asarray(T)

        if self.reaction == 'DT':
            # Bosch-Hale parameterization
            C1 = 1.17e-9
            C2 = 1.51e-2
            C3 = 7.51e-2
            C4 = 4.60e-3
            C5 = 1.35e-2
            C6 = -1.06e-4
            C7 = 1.37e-5

            theta = T / (1 - (T * (C2 + T * (C4 + T * C6))) /
                        (1 + T * (C3 + T * (C5 + T * C7))))
            xi = (6.2696 / theta)**(1/3)

            sigma_v = C1 * theta * np.sqrt(xi / (1.e-3 * T**3)) * np.exp(-3 * xi)

        elif self.reaction == 'DD':
            # DD (both branches)
            sigma_v = 2.33e-14 * T**(-2/3) * np.exp(-18.76 / T**(1/3))

        elif self.reaction == 'DHe3':
            sigma_v = 5.51e-12 * T**(-2/3) * np.exp(-37.2 / T**(1/3))

        return sigma_v * 1e-6  # cm³/s to m³/s

    def power_density(self, n1: float, n2: float, T: float) -> float:
        """
        Fusion power density.

        P = n1 n2 <σv> Q

        Args:
            n1, n2: Reactant densities (m⁻³)
            T: Temperature (keV)

        Returns:
            Power density (W/m³)
        """
        Q = {'DT': 17.6, 'DD': 3.65, 'DHe3': 18.3}  # MeV

        sigma_v = self.reactivity(T)
        return n1 * n2 * sigma_v * Q[self.reaction] * MeV


# =============================================================================
# Particle Physics Basics
# =============================================================================

class DiracEquation:
    """
    Dirac equation for relativistic spin-1/2 particles.

    (iγ^μ ∂_μ - m)ψ = 0

    Args:
        mass: Particle mass (kg)
    """

    def __init__(self, mass: float = m_e):
        if mass <= 0:
            raise ValueError("Mass must be positive")

        self.mass = mass

    def energy_momentum_relation(self, p: ArrayLike) -> np.ndarray:
        """
        Relativistic energy-momentum relation E² = p²c² + m²c⁴.

        Args:
            p: Momentum magnitude

        Returns:
            Energy (positive branch)
        """
        p = np.asarray(p)
        return np.sqrt((p * c)**2 + (self.mass * c**2)**2)

    def positive_energy_spinor(self, p: ArrayLike, spin: int = 1) -> np.ndarray:
        """
        Positive energy spinor solution u(p,s).

        Args:
            p: 3-momentum [px, py, pz]
            spin: Spin projection (+1 or -1)

        Returns:
            4-component spinor (normalized)
        """
        p = np.asarray(p)
        p_mag = np.linalg.norm(p)
        E = self.energy_momentum_relation(p_mag)

        # Two-component spinors
        if spin == 1:
            chi = np.array([1, 0])
        else:
            chi = np.array([0, 1])

        # Four-component spinor
        N = np.sqrt((E + self.mass * c**2) / (2 * self.mass * c**2))

        sigma_dot_p = p[0] * np.array([[0, 1], [1, 0]]) + \
                      p[1] * np.array([[0, -1j], [1j, 0]]) + \
                      p[2] * np.array([[1, 0], [0, -1]])

        lower = sigma_dot_p @ chi / (E + self.mass * c**2)

        u = N * np.concatenate([chi, lower])
        return u

    def negative_energy_spinor(self, p: ArrayLike, spin: int = 1) -> np.ndarray:
        """
        Negative energy spinor solution v(p,s).

        Args:
            p: 3-momentum
            spin: Spin projection

        Returns:
            4-component spinor
        """
        p = np.asarray(p)
        p_mag = np.linalg.norm(p)
        E = self.energy_momentum_relation(p_mag)

        if spin == 1:
            chi = np.array([0, 1])
        else:
            chi = np.array([1, 0])

        N = np.sqrt((E + self.mass * c**2) / (2 * self.mass * c**2))

        sigma_dot_p = p[0] * np.array([[0, 1], [1, 0]]) + \
                      p[1] * np.array([[0, -1j], [1j, 0]]) + \
                      p[2] * np.array([[1, 0], [0, -1]])

        upper = sigma_dot_p @ chi / (E + self.mass * c**2)

        v = N * np.concatenate([upper, chi])
        return v

    def compton_wavelength(self) -> float:
        """Compton wavelength ℓ_C = ℏ/(mc)."""
        return hbar / (self.mass * c)


class KleinGordonEquation:
    """
    Klein-Gordon equation for spin-0 particles.

    (□ + m²)φ = 0

    where □ = ∂²/∂t² - ∇² (with c=ℏ=1).

    Args:
        mass: Particle mass (in energy units or kg)
    """

    def __init__(self, mass: float):
        if mass < 0:
            raise ValueError("Mass must be non-negative")
        self.mass = mass

    def dispersion_relation(self, k: ArrayLike) -> np.ndarray:
        """
        Dispersion relation ω² = k² + m².

        Args:
            k: Wave vector magnitude

        Returns:
            Angular frequency (positive)
        """
        k = np.asarray(k)
        return np.sqrt(k**2 + self.mass**2)

    def group_velocity(self, k: float) -> float:
        """
        Group velocity v_g = dω/dk.

        Args:
            k: Wave vector

        Returns:
            Group velocity
        """
        omega = self.dispersion_relation(k)
        return k / omega

    def phase_velocity(self, k: float) -> float:
        """
        Phase velocity v_p = ω/k.

        Args:
            k: Wave vector

        Returns:
            Phase velocity
        """
        omega = self.dispersion_relation(k)
        return omega / k

    def plane_wave(self, x: ArrayLike, t: float, k: ArrayLike,
                   positive_energy: bool = True) -> np.ndarray:
        """
        Plane wave solution φ = exp(i(k·x - ωt)).

        Args:
            x: Position
            t: Time
            k: Wave vector
            positive_energy: If True, use positive frequency

        Returns:
            Complex field value
        """
        x = np.asarray(x)
        k = np.asarray(k)

        k_dot_x = np.dot(k, x)
        omega = self.dispersion_relation(np.linalg.norm(k))

        if positive_energy:
            return np.exp(1j * (k_dot_x - omega * t))
        else:
            return np.exp(1j * (k_dot_x + omega * t))


class DiracSpinor:
    """
    Four-component Dirac spinor operations.

    Args:
        components: Array of 4 complex components
    """

    def __init__(self, components: ArrayLike):
        self.psi = np.asarray(components, dtype=complex)

        if len(self.psi) != 4:
            raise ValueError("Dirac spinor must have 4 components")

    def adjoint(self) -> np.ndarray:
        """Dirac adjoint ψ̄ = ψ†γ⁰."""
        gamma0 = GammaMatrices.gamma(0)
        return self.psi.conj() @ gamma0

    def probability_density(self) -> float:
        """Probability density ρ = ψ†ψ."""
        return np.real(np.dot(self.psi.conj(), self.psi))

    def current(self) -> np.ndarray:
        """
        Probability current j^μ = ψ̄γ^μψ.

        Returns:
            4-current [j⁰, j¹, j², j³]
        """
        j = np.zeros(4, dtype=complex)
        psi_bar = self.adjoint()

        for mu in range(4):
            gamma_mu = GammaMatrices.gamma(mu)
            j[mu] = psi_bar @ gamma_mu @ self.psi

        return np.real(j)

    def helicity(self, momentum: ArrayLike) -> float:
        """
        Helicity h = s·p/|p|.

        Args:
            momentum: 3-momentum

        Returns:
            Helicity eigenvalue
        """
        p = np.asarray(momentum)
        p_hat = p / np.linalg.norm(p)

        # Spin operator in Dirac representation
        sigma = [
            np.array([[0, 1], [1, 0]]),
            np.array([[0, -1j], [1j, 0]]),
            np.array([[1, 0], [0, -1]])
        ]

        Sigma = [np.block([[s, np.zeros((2, 2))],
                          [np.zeros((2, 2)), s]]) for s in sigma]

        # h = Σ·p_hat
        h_op = sum(p_hat[i] * Sigma[i] for i in range(3))

        return np.real(self.psi.conj() @ h_op @ self.psi) / self.probability_density()


class GammaMatrices:
    """
    Dirac gamma matrices in the Dirac representation.

    {γ^μ, γ^ν} = 2η^μν

    Provides γ⁰, γ¹, γ², γ³, γ⁵.
    """

    @staticmethod
    def gamma(mu: int) -> np.ndarray:
        """
        Return γ^μ matrix.

        Args:
            mu: Index 0, 1, 2, 3 or 5

        Returns:
            4×4 gamma matrix
        """
        sigma_x = np.array([[0, 1], [1, 0]])
        sigma_y = np.array([[0, -1j], [1j, 0]])
        sigma_z = np.array([[1, 0], [0, -1]])
        I2 = np.eye(2)
        Z2 = np.zeros((2, 2))

        if mu == 0:
            return np.block([[I2, Z2], [Z2, -I2]])
        elif mu == 1:
            return np.block([[Z2, sigma_x], [-sigma_x, Z2]])
        elif mu == 2:
            return np.block([[Z2, sigma_y], [-sigma_y, Z2]])
        elif mu == 3:
            return np.block([[Z2, sigma_z], [-sigma_z, Z2]])
        elif mu == 5:
            return GammaMatrices.gamma5()
        else:
            raise ValueError("mu must be 0, 1, 2, 3, or 5")

    @staticmethod
    def gamma5() -> np.ndarray:
        """
        γ⁵ = iγ⁰γ¹γ²γ³.

        Returns:
            γ⁵ matrix
        """
        g0 = GammaMatrices.gamma(0)
        g1 = GammaMatrices.gamma(1)
        g2 = GammaMatrices.gamma(2)
        g3 = GammaMatrices.gamma(3)

        return 1j * g0 @ g1 @ g2 @ g3

    @staticmethod
    def sigma_munu(mu: int, nu: int) -> np.ndarray:
        """
        σ^μν = (i/2)[γ^μ, γ^ν].

        Args:
            mu, nu: Indices

        Returns:
            σ^μν matrix
        """
        g_mu = GammaMatrices.gamma(mu)
        g_nu = GammaMatrices.gamma(nu)

        return (1j / 2) * (g_mu @ g_nu - g_nu @ g_mu)

    @staticmethod
    def slash(four_vector: ArrayLike) -> np.ndarray:
        """
        Feynman slash notation: /a = γ^μ a_μ.

        Args:
            four_vector: [a⁰, a¹, a², a³]

        Returns:
            Slashed matrix
        """
        a = np.asarray(four_vector)
        result = np.zeros((4, 4), dtype=complex)

        eta = np.diag([1, -1, -1, -1])  # Minkowski metric

        for mu in range(4):
            for nu in range(4):
                result += eta[mu, nu] * a[nu] * GammaMatrices.gamma(mu)

        return result


class NeutrinoOscillation:
    """
    Neutrino oscillations (PMNS mixing).

    Probability of ν_α → ν_β:
    P(ν_α→ν_β) = Σ_i,j U*_αi U_βi U_αj U*_βj exp(-i Δm²_ij L/(2E))

    Args:
        theta12: Solar mixing angle (radians)
        theta23: Atmospheric mixing angle (radians)
        theta13: Reactor mixing angle (radians)
        delta_CP: CP-violating phase (radians)
    """

    def __init__(self, theta12: float = 0.59, theta23: float = 0.85,
                 theta13: float = 0.15, delta_CP: float = 0):
        self.theta12 = theta12
        self.theta23 = theta23
        self.theta13 = theta13
        self.delta = delta_CP

        # Mass-squared differences (eV²)
        self.dm21_sq = 7.5e-5
        self.dm32_sq = 2.5e-3

    def pmns_matrix(self) -> np.ndarray:
        """
        PMNS mixing matrix U.

        U = R23 × U13 × R12

        Returns:
            3×3 complex mixing matrix
        """
        c12 = np.cos(self.theta12)
        s12 = np.sin(self.theta12)
        c23 = np.cos(self.theta23)
        s23 = np.sin(self.theta23)
        c13 = np.cos(self.theta13)
        s13 = np.sin(self.theta13)
        delta = self.delta

        U = np.array([
            [c12*c13, s12*c13, s13*np.exp(-1j*delta)],
            [-s12*c23 - c12*s23*s13*np.exp(1j*delta),
             c12*c23 - s12*s23*s13*np.exp(1j*delta), s23*c13],
            [s12*s23 - c12*c23*s13*np.exp(1j*delta),
             -c12*s23 - s12*c23*s13*np.exp(1j*delta), c23*c13]
        ])

        return U

    def oscillation_probability(self, alpha: int, beta: int,
                                 L: float, E: float) -> float:
        """
        Neutrino oscillation probability P(ν_α → ν_β).

        Args:
            alpha: Initial flavor (0=e, 1=μ, 2=τ)
            beta: Final flavor
            L: Baseline in km
            E: Energy in GeV

        Returns:
            Oscillation probability
        """
        U = self.pmns_matrix()

        # Mass-squared differences
        dm_sq = [0, self.dm21_sq, self.dm32_sq]

        # Phase factors (L in km, E in GeV)
        # Δφ = 1.27 Δm² L / E
        factor = 1.27

        P = 0
        for i in range(3):
            for j in range(3):
                dm_ij = dm_sq[i] - dm_sq[j] if i > 0 and j > 0 else 0
                if i > 0:
                    dm_ij = dm_sq[i]
                if j > 0:
                    dm_ij -= dm_sq[j]

                phase = factor * dm_ij * L / E
                P += (np.conj(U[alpha, i]) * U[beta, i] *
                      U[alpha, j] * np.conj(U[beta, j]) *
                      np.exp(-1j * phase))

        return np.real(P)

    def survival_probability(self, alpha: int, L: float, E: float) -> float:
        """
        Survival probability P(ν_α → ν_α).

        Args:
            alpha: Flavor
            L: Baseline (km)
            E: Energy (GeV)

        Returns:
            Survival probability
        """
        return self.oscillation_probability(alpha, alpha, L, E)

    def oscillation_length(self, dm_sq: float, E: float) -> float:
        """
        Oscillation length L_osc = 4πE/Δm².

        Args:
            dm_sq: Mass-squared difference (eV²)
            E: Energy (GeV)

        Returns:
            Oscillation length in km
        """
        return 4 * np.pi * E / (1.27 * dm_sq)


class QuarkModel:
    """
    Quark model for hadron spectroscopy.

    Implements basic SU(3) flavor symmetry and hadron masses.

    Args:
        quark_content: String like 'uud' for proton
    """

    # Quark masses (constituent, MeV)
    QUARK_MASSES = {
        'u': 336,
        'd': 340,
        's': 486,
        'c': 1550,
        'b': 4730,
        't': 171000
    }

    # Quark charges
    QUARK_CHARGES = {
        'u': 2/3, 'd': -1/3, 's': -1/3,
        'c': 2/3, 'b': -1/3, 't': 2/3
    }

    def __init__(self, quark_content: str):
        self.quarks = list(quark_content.lower())

        for q in self.quarks:
            if q not in self.QUARK_MASSES and q.replace('bar', '') not in self.QUARK_MASSES:
                raise ValueError(f"Unknown quark: {q}")

    @property
    def is_baryon(self) -> bool:
        """Check if hadron is a baryon (3 quarks)."""
        return len(self.quarks) == 3

    @property
    def is_meson(self) -> bool:
        """Check if hadron is a meson (quark-antiquark)."""
        return len(self.quarks) == 2

    def charge(self) -> float:
        """Calculate total charge."""
        total = 0
        for q in self.quarks:
            if 'bar' in q:
                total -= self.QUARK_CHARGES[q.replace('bar', '')]
            else:
                total += self.QUARK_CHARGES[q]
        return total

    def constituent_mass(self) -> float:
        """
        Estimate hadron mass from constituent quark masses.

        Returns:
            Mass in MeV
        """
        mass = 0
        for q in self.quarks:
            q_name = q.replace('bar', '')
            mass += self.QUARK_MASSES[q_name]

        # Binding correction (rough)
        if self.is_baryon:
            mass -= 100  # Binding energy
        elif self.is_meson:
            mass -= 50

        return mass

    def strangeness(self) -> int:
        """Calculate strangeness quantum number."""
        S = 0
        for q in self.quarks:
            if q == 's':
                S -= 1
            elif 'sbar' in q:
                S += 1
        return S

    def baryon_number(self) -> float:
        """Calculate baryon number."""
        B = 0
        for q in self.quarks:
            if 'bar' in q:
                B -= 1/3
            else:
                B += 1/3
        return B


# Module exports
__all__ = [
    # Scattering Theory
    'PartialWave', 'ScatteringAmplitude', 'OpticalTheorem',
    'RutherfordScattering', 'MottScattering',
    # Nuclear Structure
    'LiquidDropModel', 'ShellModel', 'WoodsSaxon', 'NuclearRadius', 'NuclearSpin',
    # Radioactivity
    'AlphaDecay', 'BetaDecay', 'GammaDecay', 'DecayChain', 'HalfLife',
    # Nuclear Reactions
    'NuclearCrossSection', 'QValue', 'ResonanceFormula',
    'CompoundNucleus', 'FissionYield', 'FusionRate',
    # Particle Physics
    'DiracEquation', 'KleinGordonEquation', 'DiracSpinor',
    'GammaMatrices', 'NeutrinoOscillation', 'QuarkModel',
]
