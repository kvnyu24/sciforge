"""
Plasma Physics and Astrophysics module.

This module implements plasma physics fundamentals, magnetohydrodynamics,
fusion physics, and astrophysical systems.

Classes:
    Plasma Fundamentals:
    - DebyeLength: Screening distance in plasma
    - PlasmaFrequency: Plasma oscillation frequency
    - PlasmaParameter: Coupling strength parameter
    - VlasovEquation: Collisionless kinetic equation
    - LandauDamping: Collisionless wave damping

    Magnetohydrodynamics:
    - MHDEquations: Ideal MHD system
    - AlfvenWave: Magnetic field waves
    - Magnetosonic: Fast/slow MHD waves
    - MHDInstability: Plasma instabilities
    - MagneticReconnection: Field topology change

    Fusion Physics:
    - LawsonCriterion: Fusion breakeven condition
    - TokamakEquilibrium: Grad-Shafranov equation
    - MirrorTrap: Magnetic mirror confinement
    - ICFCapsule: Inertial confinement basics

    Astrophysics:
    - HydrostaticStar: Stellar structure
    - LaneEmden: Polytropic stars
    - StellarEvolution: HR diagram tracks
    - WhiteDwarf: Degenerate stars
    - NeutronStar: Neutron star structure
    - AccretionDisk: Alpha-disk model
    - JetLaunching: MHD jet basics
    - Nucleosynthesis: Nuclear reactions in stars
"""

import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, Tuple, Callable, Dict, List, Union
from scipy import integrate, special, optimize
from dataclasses import dataclass

# Physical constants
c = 2.998e8           # Speed of light (m/s)
k_B = 1.381e-23       # Boltzmann constant (J/K)
e = 1.602e-19         # Elementary charge (C)
m_e = 9.109e-31       # Electron mass (kg)
m_p = 1.673e-27       # Proton mass (kg)
epsilon_0 = 8.854e-12 # Vacuum permittivity (F/m)
mu_0 = 4 * np.pi * 1e-7  # Vacuum permeability (H/m)
hbar = 1.055e-34      # Reduced Planck constant (J·s)
G = 6.674e-11         # Gravitational constant (m^3/kg/s^2)
sigma_SB = 5.670e-8   # Stefan-Boltzmann constant (W/m^2/K^4)
M_sun = 1.989e30      # Solar mass (kg)
R_sun = 6.96e8        # Solar radius (m)
L_sun = 3.828e26      # Solar luminosity (W)


# =============================================================================
# Plasma Fundamentals
# =============================================================================

class DebyeLength:
    """
    Debye screening length in plasma.

    The Debye length characterizes the distance over which mobile charge
    carriers screen out electric fields in plasmas.

    λ_D = sqrt(ε₀ k_B T / (n_e e²))

    Args:
        temperature: Electron temperature in Kelvin
        density: Electron number density in m^-3
    """

    def __init__(self, temperature: float, density: float):
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        if density <= 0:
            raise ValueError("Density must be positive")

        self.temperature = temperature
        self.density = density

    @property
    def length(self) -> float:
        """Calculate Debye length in meters."""
        return np.sqrt(epsilon_0 * k_B * self.temperature /
                      (self.density * e**2))

    @property
    def sphere_count(self) -> float:
        """Number of particles in Debye sphere."""
        return (4/3) * np.pi * self.density * self.length**3

    def screening_potential(self, r: ArrayLike, q: float = e) -> np.ndarray:
        """
        Debye-screened Coulomb potential.

        φ(r) = (q / 4πε₀r) exp(-r/λ_D)

        Args:
            r: Distance from charge
            q: Point charge value

        Returns:
            Screened potential
        """
        r = np.asarray(r)
        return (q / (4 * np.pi * epsilon_0 * r)) * np.exp(-r / self.length)

    def screening_field(self, r: ArrayLike, q: float = e) -> np.ndarray:
        """
        Electric field with Debye screening.

        Args:
            r: Distance from charge
            q: Point charge value

        Returns:
            Radial electric field magnitude
        """
        r = np.asarray(r)
        prefactor = q / (4 * np.pi * epsilon_0 * r**2)
        screening = 1 + r / self.length
        return prefactor * screening * np.exp(-r / self.length)


class PlasmaFrequency:
    """
    Plasma oscillation frequency.

    The plasma frequency is the natural frequency of electron
    oscillations in a plasma.

    ω_p = sqrt(n_e e² / (ε₀ m_e))

    Args:
        density: Electron number density in m^-3
        species_mass: Mass of oscillating species (default: electron)
    """

    def __init__(self, density: float, species_mass: float = m_e):
        if density <= 0:
            raise ValueError("Density must be positive")
        if species_mass <= 0:
            raise ValueError("Species mass must be positive")

        self.density = density
        self.mass = species_mass

    @property
    def angular_frequency(self) -> float:
        """Plasma angular frequency ω_p in rad/s."""
        return np.sqrt(self.density * e**2 / (epsilon_0 * self.mass))

    @property
    def frequency(self) -> float:
        """Plasma frequency f_p in Hz."""
        return self.angular_frequency / (2 * np.pi)

    @property
    def period(self) -> float:
        """Plasma oscillation period in seconds."""
        return 1 / self.frequency

    def dispersion_relation(self, k: ArrayLike) -> np.ndarray:
        """
        Electromagnetic wave dispersion in plasma.

        ω² = ω_p² + c²k²

        Args:
            k: Wave vector magnitude

        Returns:
            Wave frequency
        """
        k = np.asarray(k)
        omega_p = self.angular_frequency
        return np.sqrt(omega_p**2 + c**2 * k**2)

    def group_velocity(self, omega: ArrayLike) -> np.ndarray:
        """
        Group velocity of EM waves in plasma.

        v_g = c * sqrt(1 - ω_p²/ω²)

        Args:
            omega: Wave angular frequency

        Returns:
            Group velocity
        """
        omega = np.asarray(omega)
        omega_p = self.angular_frequency
        return c * np.sqrt(1 - (omega_p / omega)**2)

    def refractive_index(self, omega: ArrayLike) -> np.ndarray:
        """
        Plasma refractive index.

        n = sqrt(1 - ω_p²/ω²)

        Args:
            omega: Wave angular frequency

        Returns:
            Refractive index (complex if ω < ω_p)
        """
        omega = np.asarray(omega)
        omega_p = self.angular_frequency
        n_squared = 1 - (omega_p / omega)**2
        return np.sqrt(n_squared.astype(complex))


class PlasmaParameter:
    """
    Plasma coupling parameter.

    The plasma parameter Γ measures the ratio of potential to kinetic energy:
    Γ = e² / (4πε₀ a k_B T)

    where a is the mean inter-particle spacing.

    Γ << 1: Weakly coupled (ideal) plasma
    Γ >> 1: Strongly coupled plasma

    Args:
        temperature: Temperature in Kelvin
        density: Number density in m^-3
        charge: Particle charge (default: e)
    """

    def __init__(self, temperature: float, density: float, charge: float = e):
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        if density <= 0:
            raise ValueError("Density must be positive")

        self.temperature = temperature
        self.density = density
        self.charge = charge

    @property
    def wigner_seitz_radius(self) -> float:
        """Mean inter-particle spacing (Wigner-Seitz radius)."""
        return (3 / (4 * np.pi * self.density))**(1/3)

    @property
    def coupling_parameter(self) -> float:
        """Coupling parameter Γ."""
        a = self.wigner_seitz_radius
        return self.charge**2 / (4 * np.pi * epsilon_0 * a * k_B * self.temperature)

    @property
    def is_weakly_coupled(self) -> bool:
        """Check if plasma is weakly coupled (ideal)."""
        return self.coupling_parameter < 0.1

    @property
    def is_strongly_coupled(self) -> bool:
        """Check if plasma is strongly coupled."""
        return self.coupling_parameter > 1.0

    @property
    def thermal_velocity(self) -> float:
        """Thermal velocity of particles."""
        return np.sqrt(k_B * self.temperature / m_e)

    def coulomb_logarithm(self, debye_length: Optional[float] = None) -> float:
        """
        Coulomb logarithm for collision calculations.

        ln Λ = ln(λ_D / b_min)

        where b_min is the minimum impact parameter.

        Args:
            debye_length: Debye length (calculated if not provided)

        Returns:
            Coulomb logarithm
        """
        if debye_length is None:
            debye = DebyeLength(self.temperature, self.density)
            debye_length = debye.length

        # Classical minimum impact parameter
        b_classical = self.charge**2 / (4 * np.pi * epsilon_0 * k_B * self.temperature)
        # Quantum minimum (de Broglie wavelength)
        v_th = self.thermal_velocity
        b_quantum = hbar / (m_e * v_th)

        b_min = max(b_classical, b_quantum)
        return np.log(debye_length / b_min)


class VlasovEquation:
    """
    Vlasov equation solver for collisionless plasma.

    The Vlasov equation describes the evolution of the distribution function
    in phase space:

    ∂f/∂t + v·∇f + (q/m)(E + v×B)·∇_v f = 0

    This implementation uses a 1D electrostatic version.

    Args:
        nx: Number of spatial grid points
        nv: Number of velocity grid points
        Lx: System length
        vmax: Maximum velocity
    """

    def __init__(self, nx: int = 64, nv: int = 64,
                 Lx: float = 4*np.pi, vmax: float = 6.0):
        self.nx = nx
        self.nv = nv
        self.Lx = Lx
        self.vmax = vmax

        # Create grids
        self.x = np.linspace(0, Lx, nx, endpoint=False)
        self.v = np.linspace(-vmax, vmax, nv)
        self.dx = Lx / nx
        self.dv = 2 * vmax / (nv - 1)

        # Wave vector for FFT
        self.kx = 2 * np.pi * np.fft.fftfreq(nx, self.dx)

    def maxwellian(self, density: float = 1.0, temperature: float = 1.0,
                   drift_velocity: float = 0.0) -> np.ndarray:
        """
        Initialize Maxwellian distribution.

        Args:
            density: Particle density
            temperature: Temperature (in velocity units)
            drift_velocity: Bulk drift velocity

        Returns:
            Distribution function f(x, v)
        """
        v = self.v
        f = density / np.sqrt(2 * np.pi * temperature)
        f *= np.exp(-(v - drift_velocity)**2 / (2 * temperature))
        return np.outer(np.ones(self.nx), f)

    def compute_density(self, f: np.ndarray) -> np.ndarray:
        """Compute number density from distribution."""
        return np.trapezoid(f, self.v, axis=1)

    def compute_electric_field(self, f: np.ndarray) -> np.ndarray:
        """
        Compute electric field from Poisson's equation.

        Uses FFT for periodic boundaries.
        """
        n = self.compute_density(f)
        n_avg = np.mean(n)
        rho = n - n_avg  # Charge density (neutralizing background)

        # Solve Poisson equation in Fourier space
        rho_k = np.fft.fft(rho)

        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            phi_k = -rho_k / (1j * self.kx)
            phi_k[0] = 0  # Set k=0 mode
            E_k = -1j * self.kx * np.fft.ifft(phi_k)

        return np.real(np.fft.ifft(-1j * self.kx * phi_k))

    def step(self, f: np.ndarray, dt: float) -> np.ndarray:
        """
        Advance distribution function by one time step.

        Uses splitting method: half advection in x, full advection in v,
        half advection in x.

        Args:
            f: Distribution function
            dt: Time step

        Returns:
            Updated distribution function
        """
        # Half step in x
        f = self._advect_x(f, dt/2)

        # Full step in v with electric field
        E = self.compute_electric_field(f)
        f = self._advect_v(f, E, dt)

        # Half step in x
        f = self._advect_x(f, dt/2)

        return f

    def _advect_x(self, f: np.ndarray, dt: float) -> np.ndarray:
        """Advect in x using FFT."""
        f_k = np.fft.fft(f, axis=0)
        for j in range(self.nv):
            v = self.v[j]
            f_k[:, j] *= np.exp(-1j * self.kx * v * dt)
        return np.real(np.fft.ifft(f_k, axis=0))

    def _advect_v(self, f: np.ndarray, E: np.ndarray, dt: float) -> np.ndarray:
        """Advect in v using upwind scheme."""
        f_new = f.copy()
        for i in range(self.nx):
            a = E[i]  # Acceleration (E/m with m=1, q=1)
            if a > 0:
                # Upwind from left
                for j in range(1, self.nv):
                    f_new[i, j] = f[i, j] - a * dt / self.dv * (f[i, j] - f[i, j-1])
            else:
                # Upwind from right
                for j in range(self.nv - 1):
                    f_new[i, j] = f[i, j] - a * dt / self.dv * (f[i, j+1] - f[i, j])
        return f_new

    def simulate(self, f0: np.ndarray, t_final: float,
                 dt: float = 0.1) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Run simulation and collect diagnostics.

        Args:
            f0: Initial distribution
            t_final: Final time
            dt: Time step

        Returns:
            times: Time array
            field_energy: Electric field energy vs time
            distributions: List of distribution snapshots
        """
        times = []
        field_energy = []
        distributions = [f0.copy()]

        f = f0.copy()
        t = 0

        while t < t_final:
            E = self.compute_electric_field(f)
            energy = 0.5 * np.sum(E**2) * self.dx

            times.append(t)
            field_energy.append(energy)

            f = self.step(f, dt)
            t += dt

        return np.array(times), np.array(field_energy), distributions


class LandauDamping:
    """
    Landau damping of electrostatic waves in plasma.

    Landau damping is a collisionless damping mechanism caused by
    wave-particle interactions where particles with velocities near
    the phase velocity exchange energy with the wave.

    γ = -√(π/8) * (ω_p³/k³v_th³) * exp(-ω²/(2k²v_th²))

    Args:
        plasma_frequency: Plasma frequency ω_p
        thermal_velocity: Thermal velocity v_th
    """

    def __init__(self, plasma_frequency: float, thermal_velocity: float):
        if plasma_frequency <= 0:
            raise ValueError("Plasma frequency must be positive")
        if thermal_velocity <= 0:
            raise ValueError("Thermal velocity must be positive")

        self.omega_p = plasma_frequency
        self.v_th = thermal_velocity

    def dispersion_relation(self, k: float) -> complex:
        """
        Solve dispersion relation for Langmuir waves with Landau damping.

        For k*λ_D << 1:
        ω ≈ ω_p(1 + 3k²λ_D²/2) - i*γ

        Args:
            k: Wave vector magnitude

        Returns:
            Complex frequency ω = ω_r + i*γ
        """
        omega_p = self.omega_p
        v_th = self.v_th

        # Debye length
        lambda_D = v_th / omega_p
        kld = k * lambda_D

        # Real frequency (Bohm-Gross)
        omega_r = omega_p * np.sqrt(1 + 3 * kld**2)

        # Damping rate
        gamma = self.damping_rate(k, omega_r)

        return omega_r + 1j * gamma

    def damping_rate(self, k: float, omega: Optional[float] = None) -> float:
        """
        Calculate Landau damping rate.

        Args:
            k: Wave vector magnitude
            omega: Wave frequency (if not provided, uses Bohm-Gross)

        Returns:
            Damping rate γ (negative for damping)
        """
        omega_p = self.omega_p
        v_th = self.v_th

        if omega is None:
            lambda_D = v_th / omega_p
            omega = omega_p * np.sqrt(1 + 3 * (k * lambda_D)**2)

        # Phase velocity
        v_phase = omega / k

        # Damping rate from derivative of distribution at v_phase
        prefactor = np.sqrt(np.pi / 8) * omega_p**3 / (k**3 * v_th**3)
        exponent = -omega**2 / (2 * k**2 * v_th**2)

        return -prefactor * np.exp(exponent)

    def wave_amplitude(self, k: float, t: ArrayLike, A0: float = 1.0) -> np.ndarray:
        """
        Wave amplitude evolution with Landau damping.

        Args:
            k: Wave vector
            t: Time array
            A0: Initial amplitude

        Returns:
            Wave amplitude vs time
        """
        t = np.asarray(t)
        omega = self.dispersion_relation(k)
        return A0 * np.abs(np.exp(-1j * omega * t))


# =============================================================================
# Magnetohydrodynamics
# =============================================================================

class MHDEquations:
    """
    Ideal magnetohydrodynamics equations.

    The ideal MHD equations combine fluid dynamics with Maxwell's equations:

    ∂ρ/∂t + ∇·(ρv) = 0              (mass continuity)
    ∂(ρv)/∂t + ∇·(ρvv - BB/μ₀ + P*I) = 0   (momentum)
    ∂B/∂t = ∇×(v×B)                  (induction)
    ∂e/∂t + ∇·((e+P)v - B(v·B)/μ₀) = 0    (energy)

    This implements a 1D MHD solver.

    Args:
        nx: Number of grid points
        Lx: Domain length
        gamma: Adiabatic index
    """

    def __init__(self, nx: int = 200, Lx: float = 1.0, gamma: float = 5/3):
        self.nx = nx
        self.Lx = Lx
        self.gamma = gamma

        self.x = np.linspace(0, Lx, nx)
        self.dx = Lx / (nx - 1)

        # State variables: [rho, rho*vx, rho*vy, rho*vz, Bx, By, Bz, e]
        self.state = np.zeros((8, nx))

    def initialize_uniform(self, rho: float, vx: float, vy: float, vz: float,
                          Bx: float, By: float, Bz: float, P: float) -> None:
        """Initialize with uniform state."""
        self.state[0, :] = rho
        self.state[1, :] = rho * vx
        self.state[2, :] = rho * vy
        self.state[3, :] = rho * vz
        self.state[4, :] = Bx
        self.state[5, :] = By
        self.state[6, :] = Bz

        # Total energy
        KE = 0.5 * rho * (vx**2 + vy**2 + vz**2)
        ME = 0.5 * (Bx**2 + By**2 + Bz**2) / mu_0
        self.state[7, :] = P / (self.gamma - 1) + KE + ME

    def compute_pressure(self) -> np.ndarray:
        """Compute thermal pressure from state."""
        rho = self.state[0]
        rho_v = self.state[1:4]
        B = self.state[4:7]
        e = self.state[7]

        KE = 0.5 * np.sum(rho_v**2, axis=0) / rho
        ME = 0.5 * np.sum(B**2, axis=0) / mu_0

        return (self.gamma - 1) * (e - KE - ME)

    def compute_fast_speed(self) -> np.ndarray:
        """Compute fast magnetosonic speed."""
        rho = self.state[0]
        B = self.state[4:7]
        P = self.compute_pressure()

        c_s = np.sqrt(self.gamma * P / rho)  # Sound speed
        v_A = np.sqrt(np.sum(B**2, axis=0) / (mu_0 * rho))  # Alfven speed

        return np.sqrt(c_s**2 + v_A**2)

    def compute_flux(self, U: np.ndarray) -> np.ndarray:
        """Compute flux vector."""
        rho = U[0]
        vx = U[1] / rho
        vy = U[2] / rho
        vz = U[3] / rho
        Bx, By, Bz = U[4], U[5], U[6]
        e = U[7]

        # Pressure
        KE = 0.5 * rho * (vx**2 + vy**2 + vz**2)
        ME = 0.5 * (Bx**2 + By**2 + Bz**2) / mu_0
        P = (self.gamma - 1) * (e - KE - ME)
        P_tot = P + ME

        F = np.zeros_like(U)
        F[0] = rho * vx
        F[1] = rho * vx**2 + P_tot - Bx**2 / mu_0
        F[2] = rho * vx * vy - Bx * By / mu_0
        F[3] = rho * vx * vz - Bx * Bz / mu_0
        F[4] = 0  # ∂Bx/∂t = 0 in 1D
        F[5] = By * vx - Bx * vy
        F[6] = Bz * vx - Bx * vz
        F[7] = (e + P_tot) * vx - Bx * (vx*Bx + vy*By + vz*Bz) / mu_0

        return F

    def lax_friedrichs_flux(self, UL: np.ndarray, UR: np.ndarray,
                           alpha: float) -> np.ndarray:
        """Lax-Friedrichs numerical flux."""
        FL = self.compute_flux(UL)
        FR = self.compute_flux(UR)
        return 0.5 * (FL + FR - alpha * (UR - UL))

    def step(self, dt: float) -> None:
        """
        Advance solution by one time step using finite volume method.

        Args:
            dt: Time step
        """
        U = self.state.copy()

        # Maximum wave speed for Lax-Friedrichs
        alpha = np.max(np.abs(U[1] / U[0]) + self.compute_fast_speed())

        # Compute fluxes at cell interfaces
        dU = np.zeros_like(U)

        for i in range(1, self.nx - 1):
            F_right = self.lax_friedrichs_flux(U[:, i], U[:, i+1], alpha)
            F_left = self.lax_friedrichs_flux(U[:, i-1], U[:, i], alpha)
            dU[:, i] = -(F_right - F_left) / self.dx

        # Update
        self.state = U + dt * dU

        # Enforce boundary conditions (outflow)
        self.state[:, 0] = self.state[:, 1]
        self.state[:, -1] = self.state[:, -2]


class AlfvenWave:
    """
    Alfven wave propagation in magnetized plasma.

    Alfven waves are transverse waves that propagate along magnetic
    field lines with velocity v_A = B/√(μ₀ρ).

    Args:
        B0: Background magnetic field strength (Tesla)
        density: Mass density (kg/m³)
    """

    def __init__(self, B0: float, density: float):
        if B0 <= 0:
            raise ValueError("Magnetic field must be positive")
        if density <= 0:
            raise ValueError("Density must be positive")

        self.B0 = B0
        self.density = density

    @property
    def alfven_velocity(self) -> float:
        """Alfven velocity v_A = B/√(μ₀ρ)."""
        return self.B0 / np.sqrt(mu_0 * self.density)

    def dispersion_relation(self, k: ArrayLike, theta: float = 0) -> np.ndarray:
        """
        Alfven wave dispersion relation.

        ω = k * v_A * |cos(θ)|

        Args:
            k: Wave vector magnitude
            theta: Angle between k and B₀

        Returns:
            Wave frequency
        """
        k = np.asarray(k)
        return k * self.alfven_velocity * np.abs(np.cos(theta))

    def phase_velocity(self, theta: float = 0) -> float:
        """
        Phase velocity.

        Args:
            theta: Propagation angle relative to B₀

        Returns:
            Phase velocity
        """
        return self.alfven_velocity * np.abs(np.cos(theta))

    def group_velocity(self, theta: float = 0) -> Tuple[float, float]:
        """
        Group velocity components.

        For Alfven waves, group velocity is along B₀.

        Args:
            theta: Propagation angle

        Returns:
            (v_parallel, v_perpendicular) components
        """
        v_A = self.alfven_velocity
        return (v_A * np.sign(np.cos(theta)), 0.0)

    def perturbation(self, x: ArrayLike, t: float, k: float,
                    amplitude: float = 0.01) -> Dict[str, np.ndarray]:
        """
        Calculate Alfven wave perturbations.

        Args:
            x: Position array
            t: Time
            k: Wave vector
            amplitude: Relative perturbation amplitude

        Returns:
            Dictionary with velocity and magnetic field perturbations
        """
        x = np.asarray(x)
        omega = self.dispersion_relation(k)
        phase = k * x - omega * t

        # Velocity perturbation (perpendicular to B₀)
        delta_v = amplitude * self.alfven_velocity * np.cos(phase)

        # Magnetic field perturbation
        delta_B = -amplitude * self.B0 * np.cos(phase)

        return {
            'delta_vy': delta_v,
            'delta_By': delta_B,
            'phase': phase
        }


class Magnetosonic:
    """
    Magnetosonic waves (fast and slow MHD waves).

    Magnetosonic waves combine acoustic and magnetic effects:

    v² = ½(c_s² + v_A²) ± ½√((c_s² + v_A²)² - 4c_s²v_A²cos²θ)

    + for fast wave, - for slow wave

    Args:
        B0: Background magnetic field (Tesla)
        density: Mass density (kg/m³)
        pressure: Thermal pressure (Pa)
        gamma: Adiabatic index
    """

    def __init__(self, B0: float, density: float, pressure: float,
                 gamma: float = 5/3):
        if B0 < 0:
            raise ValueError("Magnetic field must be non-negative")
        if density <= 0:
            raise ValueError("Density must be positive")
        if pressure <= 0:
            raise ValueError("Pressure must be positive")

        self.B0 = B0
        self.density = density
        self.pressure = pressure
        self.gamma = gamma

    @property
    def sound_speed(self) -> float:
        """Adiabatic sound speed."""
        return np.sqrt(self.gamma * self.pressure / self.density)

    @property
    def alfven_speed(self) -> float:
        """Alfven speed."""
        return self.B0 / np.sqrt(mu_0 * self.density)

    def wave_speeds(self, theta: float) -> Tuple[float, float, float]:
        """
        Calculate all three MHD wave speeds.

        Args:
            theta: Propagation angle relative to B₀

        Returns:
            (fast, Alfven, slow) wave speeds
        """
        c_s = self.sound_speed
        v_A = self.alfven_speed
        cos_theta = np.cos(theta)

        # Alfven speed for given angle
        v_alfven = v_A * np.abs(cos_theta)

        # Fast and slow magnetosonic
        sum_sq = c_s**2 + v_A**2
        diff = np.sqrt(sum_sq**2 - 4 * c_s**2 * v_A**2 * cos_theta**2)

        v_fast = np.sqrt(0.5 * (sum_sq + diff))
        v_slow = np.sqrt(0.5 * (sum_sq - diff))

        return (v_fast, v_alfven, v_slow)

    def friedrichs_diagram(self, n_angles: int = 100) -> Dict[str, np.ndarray]:
        """
        Generate Friedrichs (phase velocity) diagram.

        Returns:
            Dictionary with angles and wave speeds
        """
        theta = np.linspace(0, 2*np.pi, n_angles)
        fast = np.zeros(n_angles)
        alfven = np.zeros(n_angles)
        slow = np.zeros(n_angles)

        for i, th in enumerate(theta):
            fast[i], alfven[i], slow[i] = self.wave_speeds(th)

        return {
            'theta': theta,
            'fast': fast,
            'alfven': alfven,
            'slow': slow
        }

    @property
    def plasma_beta(self) -> float:
        """Ratio of thermal to magnetic pressure."""
        P_mag = self.B0**2 / (2 * mu_0)
        return self.pressure / P_mag


class MHDInstability:
    """
    Basic MHD instabilities analysis.

    Implements growth rate calculations for common instabilities:
    - Kink instability
    - Sausage instability
    - Rayleigh-Taylor instability

    Args:
        B0: Magnetic field (Tesla)
        density: Density (kg/m³)
        radius: Characteristic radius (m)
    """

    def __init__(self, B0: float, density: float, radius: float):
        self.B0 = B0
        self.density = density
        self.radius = radius

    @property
    def alfven_time(self) -> float:
        """Alfven time scale τ_A = R/v_A."""
        v_A = self.B0 / np.sqrt(mu_0 * self.density)
        return self.radius / v_A

    def kink_growth_rate(self, k: float, qa: float = 1.0) -> float:
        """
        Kink instability growth rate.

        Kink (m=1) instability occurs when safety factor q < 1.

        Args:
            k: Axial wave vector
            qa: Safety factor at edge

        Returns:
            Growth rate γ (positive if unstable)
        """
        v_A = self.B0 / np.sqrt(mu_0 * self.density)

        # Simplified growth rate for kink
        if qa < 1:
            gamma = k * v_A * np.sqrt(1 - qa**2)
        else:
            gamma = 0.0

        return gamma

    def sausage_growth_rate(self, k: float, pressure: float) -> float:
        """
        Sausage (m=0) instability growth rate.

        Args:
            k: Axial wave vector
            pressure: Plasma pressure

        Returns:
            Growth rate
        """
        v_A = self.B0 / np.sqrt(mu_0 * self.density)
        beta = 2 * mu_0 * pressure / self.B0**2

        # Sausage instability for high beta
        if beta > 1:
            gamma = k * v_A * np.sqrt(beta - 1) / np.sqrt(2)
        else:
            gamma = 0.0

        return gamma

    def rayleigh_taylor_growth_rate(self, g: float, k: float,
                                     density_ratio: float) -> float:
        """
        Rayleigh-Taylor instability growth rate.

        Occurs when heavy fluid is supported by light fluid
        in a gravitational field.

        γ = √(g k A)

        where A = (ρ₂ - ρ₁)/(ρ₂ + ρ₁) is the Atwood number.

        Args:
            g: Effective gravity
            k: Wave vector
            density_ratio: ρ₂/ρ₁

        Returns:
            Growth rate
        """
        atwood = (density_ratio - 1) / (density_ratio + 1)
        if atwood > 0:
            return np.sqrt(g * k * atwood)
        return 0.0


class MagneticReconnection:
    """
    Magnetic reconnection model.

    Magnetic reconnection changes the topology of magnetic field lines,
    converting magnetic energy to kinetic and thermal energy.

    Implements Sweet-Parker and Petschek reconnection models.

    Args:
        B_in: Inflow magnetic field (Tesla)
        L: System length scale (m)
        density: Plasma density (kg/m³)
        resistivity: Plasma resistivity (Ω·m)
    """

    def __init__(self, B_in: float, L: float, density: float,
                 resistivity: float):
        if B_in <= 0:
            raise ValueError("Magnetic field must be positive")
        if L <= 0:
            raise ValueError("Length scale must be positive")
        if density <= 0:
            raise ValueError("Density must be positive")
        if resistivity <= 0:
            raise ValueError("Resistivity must be positive")

        self.B_in = B_in
        self.L = L
        self.density = density
        self.eta = resistivity

    @property
    def alfven_velocity(self) -> float:
        """Inflow Alfven velocity."""
        return self.B_in / np.sqrt(mu_0 * self.density)

    @property
    def lundquist_number(self) -> float:
        """
        Lundquist number S = μ₀ L v_A / η.

        Measures ratio of resistive diffusion time to Alfven time.
        """
        v_A = self.alfven_velocity
        return mu_0 * self.L * v_A / self.eta

    def sweet_parker_rate(self) -> float:
        """
        Sweet-Parker reconnection rate.

        M = v_in/v_A = S^(-1/2)

        Returns:
            Reconnection rate M = v_in/v_A
        """
        return 1 / np.sqrt(self.lundquist_number)

    def sweet_parker_thickness(self) -> float:
        """Sweet-Parker current sheet thickness δ = L/√S."""
        return self.L / np.sqrt(self.lundquist_number)

    def petschek_rate(self) -> float:
        """
        Petschek (fast) reconnection rate.

        M ≈ π / (8 ln S)

        Returns:
            Reconnection rate
        """
        S = self.lundquist_number
        return np.pi / (8 * np.log(S))

    def energy_release_rate(self, reconnection_rate: Optional[float] = None) -> float:
        """
        Rate of magnetic energy release.

        P = B²/μ₀ * L² * v_in

        Args:
            reconnection_rate: M = v_in/v_A (default: Sweet-Parker)

        Returns:
            Power released (W)
        """
        if reconnection_rate is None:
            reconnection_rate = self.sweet_parker_rate()

        v_A = self.alfven_velocity
        v_in = reconnection_rate * v_A

        return self.B_in**2 / mu_0 * self.L**2 * v_in

    def reconnection_time(self, reconnection_rate: Optional[float] = None) -> float:
        """
        Time scale for reconnection.

        τ = L / v_in

        Args:
            reconnection_rate: M = v_in/v_A (default: Sweet-Parker)

        Returns:
            Reconnection time (s)
        """
        if reconnection_rate is None:
            reconnection_rate = self.sweet_parker_rate()

        v_A = self.alfven_velocity
        v_in = reconnection_rate * v_A

        return self.L / v_in


# =============================================================================
# Fusion Physics
# =============================================================================

class LawsonCriterion:
    """
    Lawson criterion for fusion breakeven.

    The Lawson criterion specifies the minimum n*τ*T required for
    fusion power to exceed losses:

    n τ_E > 12 k_B T / (<σv> E_fusion)

    For DT fusion at optimal temperature (~15 keV):
    n τ_E > 1.5 × 10²⁰ m⁻³·s

    Args:
        fuel_type: 'DT', 'DD', or 'DHe3'
    """

    # Fusion reaction parameters
    FUSION_DATA = {
        'DT': {
            'E_fusion': 17.6e6 * e,  # 17.6 MeV in Joules
            'T_optimal': 15e3 * e / k_B,  # 15 keV in Kelvin
            'sigma_v_max': 8.5e-22,  # m³/s at optimal T
        },
        'DD': {
            'E_fusion': 3.65e6 * e,  # Average DD
            'T_optimal': 50e3 * e / k_B,  # 50 keV
            'sigma_v_max': 3.0e-23,
        },
        'DHe3': {
            'E_fusion': 18.3e6 * e,  # D-He3
            'T_optimal': 60e3 * e / k_B,
            'sigma_v_max': 1.5e-22,
        }
    }

    def __init__(self, fuel_type: str = 'DT'):
        if fuel_type not in self.FUSION_DATA:
            raise ValueError(f"Unknown fuel type: {fuel_type}")

        self.fuel_type = fuel_type
        self.data = self.FUSION_DATA[fuel_type]

    @property
    def optimal_temperature(self) -> float:
        """Optimal temperature for maximum reactivity (K)."""
        return self.data['T_optimal']

    @property
    def optimal_temperature_keV(self) -> float:
        """Optimal temperature in keV."""
        return self.data['T_optimal'] * k_B / e * 1e-3

    def reactivity(self, T: float) -> float:
        """
        Fusion reactivity <σv> as function of temperature.

        Uses parameterization valid near optimal temperature.

        Args:
            T: Temperature in Kelvin

        Returns:
            Reactivity in m³/s
        """
        T_opt = self.data['T_optimal']
        sigma_v_max = self.data['sigma_v_max']

        # Gaussian approximation around optimal T
        sigma_T = T_opt / 2
        return sigma_v_max * np.exp(-((T - T_opt) / sigma_T)**2)

    def lawson_ntau(self, T: float) -> float:
        """
        Required n*τ for breakeven at given temperature.

        Args:
            T: Temperature in Kelvin

        Returns:
            Required n*τ_E in m⁻³·s
        """
        E_fusion = self.data['E_fusion']
        sigma_v = self.reactivity(T)

        # Breakeven condition
        return 12 * k_B * T / (E_fusion * sigma_v)

    def triple_product(self, T: float) -> float:
        """
        Required triple product n*T*τ for breakeven.

        Args:
            T: Temperature in Kelvin

        Returns:
            Required n*T*τ_E in m⁻³·s·K
        """
        return self.lawson_ntau(T) * T

    def fusion_power_density(self, n: float, T: float) -> float:
        """
        Fusion power density.

        P_fus = (1/4) n² <σv> E_fusion

        Args:
            n: Number density in m⁻³
            T: Temperature in Kelvin

        Returns:
            Power density in W/m³
        """
        E_fusion = self.data['E_fusion']
        sigma_v = self.reactivity(T)

        return 0.25 * n**2 * sigma_v * E_fusion

    def bremsstrahlung_loss(self, n: float, T: float, Z_eff: float = 1.0) -> float:
        """
        Bremsstrahlung radiation power density.

        P_brem ≈ 5.35 × 10⁻³⁷ Z_eff n² √T (W/m³)

        Args:
            n: Electron density in m⁻³
            T: Temperature in Kelvin
            Z_eff: Effective charge

        Returns:
            Loss power density in W/m³
        """
        T_eV = T * k_B / e
        return 5.35e-37 * Z_eff * n**2 * np.sqrt(T_eV)

    def ignition_condition(self, T: float) -> float:
        """
        Required n*τ for ignition (alpha heating sustains burn).

        Args:
            T: Temperature in Kelvin

        Returns:
            Required n*τ for ignition
        """
        # Alpha particle energy fraction for DT
        E_alpha = 3.5e6 * e  # 3.5 MeV
        E_fusion = self.data['E_fusion']

        alpha_fraction = E_alpha / E_fusion

        # Ignition requires alpha heating to exceed losses
        return self.lawson_ntau(T) / alpha_fraction


class TokamakEquilibrium:
    """
    Tokamak MHD equilibrium (Grad-Shafranov equation).

    The Grad-Shafranov equation describes axisymmetric MHD equilibrium:

    Δ*ψ = -μ₀ R² p'(ψ) - F F'(ψ)

    where ψ is the poloidal flux function.

    This implements a simple circular cross-section approximation.

    Args:
        R0: Major radius (m)
        a: Minor radius (m)
        B0: Toroidal field at R0 (Tesla)
        Ip: Plasma current (A)
    """

    def __init__(self, R0: float, a: float, B0: float, Ip: float):
        if R0 <= 0:
            raise ValueError("Major radius must be positive")
        if a <= 0:
            raise ValueError("Minor radius must be positive")
        if a >= R0:
            raise ValueError("Minor radius must be less than major radius")
        if B0 <= 0:
            raise ValueError("Toroidal field must be positive")

        self.R0 = R0
        self.a = a
        self.B0 = B0
        self.Ip = Ip

    @property
    def aspect_ratio(self) -> float:
        """Aspect ratio A = R0/a."""
        return self.R0 / self.a

    @property
    def inverse_aspect_ratio(self) -> float:
        """Inverse aspect ratio ε = a/R0."""
        return self.a / self.R0

    def safety_factor(self, r: ArrayLike) -> np.ndarray:
        """
        Safety factor q(r) profile.

        Uses simple parabolic profile: q(r) = q_0 + (q_a - q_0)(r/a)²

        Args:
            r: Minor radius coordinate

        Returns:
            Safety factor profile
        """
        r = np.asarray(r)
        # Typical q profile
        q_0 = 1.0  # On axis
        q_a = self.edge_safety_factor()

        return q_0 + (q_a - q_0) * (r / self.a)**2

    def edge_safety_factor(self) -> float:
        """
        Edge safety factor q_a.

        q_a = 2π a² B0 / (μ₀ R0 Ip)
        """
        return 2 * np.pi * self.a**2 * self.B0 / (mu_0 * self.R0 * self.Ip)

    def poloidal_field(self, r: float) -> float:
        """
        Poloidal magnetic field at radius r.

        B_θ = μ₀ I(r) / (2π r)

        for enclosed current I(r).

        Args:
            r: Minor radius

        Returns:
            Poloidal field (Tesla)
        """
        if r <= 0:
            return 0
        if r >= self.a:
            return mu_0 * self.Ip / (2 * np.pi * r)

        # Assume parabolic current profile
        I_enclosed = self.Ip * (r / self.a)**2
        return mu_0 * I_enclosed / (2 * np.pi * r)

    def toroidal_field(self, R: float) -> float:
        """
        Toroidal magnetic field at major radius R.

        B_φ = B0 R0 / R

        Args:
            R: Major radius coordinate

        Returns:
            Toroidal field (Tesla)
        """
        return self.B0 * self.R0 / R

    def beta_poloidal(self, n: float, T: float) -> float:
        """
        Poloidal beta.

        β_p = 2μ₀ <p> / B_θ²

        Args:
            n: Density (m⁻³)
            T: Temperature (K)

        Returns:
            Poloidal beta
        """
        p = n * k_B * T
        B_theta = self.poloidal_field(self.a)
        return 2 * mu_0 * p / B_theta**2

    def beta_toroidal(self, n: float, T: float) -> float:
        """
        Toroidal beta.

        β_t = 2μ₀ <p> / B0²

        Args:
            n: Density (m⁻³)
            T: Temperature (K)

        Returns:
            Toroidal beta
        """
        p = n * k_B * T
        return 2 * mu_0 * p / self.B0**2

    def kink_stability_limit(self) -> float:
        """
        Kruskal-Shafranov limit for kink stability.

        Returns:
            Maximum stable current (A)
        """
        return 2 * np.pi * self.a**2 * self.B0 / (mu_0 * self.R0)

    @property
    def is_kink_stable(self) -> bool:
        """Check if plasma is kink stable (q_a > 1)."""
        return self.edge_safety_factor() > 1.0


class MirrorTrap:
    """
    Magnetic mirror confinement.

    Particles are confined by magnetic mirrors where the field
    strength increases at the ends of the device.

    The mirror ratio R_m = B_max / B_min determines the loss cone.

    Args:
        B_min: Minimum field at center (Tesla)
        B_max: Maximum field at mirror (Tesla)
        length: Mirror-to-mirror length (m)
    """

    def __init__(self, B_min: float, B_max: float, length: float):
        if B_min <= 0:
            raise ValueError("B_min must be positive")
        if B_max <= B_min:
            raise ValueError("B_max must be greater than B_min")
        if length <= 0:
            raise ValueError("Length must be positive")

        self.B_min = B_min
        self.B_max = B_max
        self.length = length

    @property
    def mirror_ratio(self) -> float:
        """Mirror ratio R_m = B_max / B_min."""
        return self.B_max / self.B_min

    @property
    def loss_cone_angle(self) -> float:
        """
        Loss cone half-angle.

        sin²(θ_loss) = 1 / R_m

        Returns:
            Loss cone angle in radians
        """
        return np.arcsin(1 / np.sqrt(self.mirror_ratio))

    def trapped_fraction(self) -> float:
        """
        Fraction of particles trapped by mirror.

        For isotropic distribution:
        f_trapped = 1 - 1/R_m
        """
        return 1 - 1 / self.mirror_ratio

    def bounce_frequency(self, energy: float, mass: float) -> float:
        """
        Particle bounce frequency between mirrors.

        Args:
            energy: Particle energy (J)
            mass: Particle mass (kg)

        Returns:
            Bounce frequency (Hz)
        """
        v = np.sqrt(2 * energy / mass)
        return v / (2 * self.length)

    def confinement_parameter(self, density: float, temperature: float,
                               mass: float = m_p) -> float:
        """
        Mirror confinement parameter n*τ.

        τ ~ (R_m - 1) / ν_ii

        where ν_ii is ion-ion collision frequency.

        Args:
            density: Particle density (m⁻³)
            temperature: Temperature (K)
            mass: Ion mass (default: proton)

        Returns:
            Approximate n*τ (m⁻³·s)
        """
        # Collision frequency estimate
        v_th = np.sqrt(k_B * temperature / mass)
        lambda_D = np.sqrt(epsilon_0 * k_B * temperature / (density * e**2))
        ln_Lambda = 10  # Coulomb logarithm approximation

        nu_ii = density * e**4 * ln_Lambda / (
            4 * np.pi * epsilon_0**2 * mass**2 * v_th**3
        )

        tau = (self.mirror_ratio - 1) / nu_ii
        return density * tau

    def magnetic_moment(self, energy_perp: float, B: float) -> float:
        """
        First adiabatic invariant (magnetic moment).

        μ = E_⊥ / B

        Args:
            energy_perp: Perpendicular kinetic energy
            B: Local magnetic field

        Returns:
            Magnetic moment
        """
        return energy_perp / B


class ICFCapsule:
    """
    Inertial confinement fusion capsule basics.

    Models the implosion of a spherical fuel capsule
    driven by external ablation pressure.

    Args:
        initial_radius: Initial capsule radius (m)
        shell_thickness: Initial shell thickness (m)
        fuel_mass: DT fuel mass (kg)
    """

    def __init__(self, initial_radius: float, shell_thickness: float,
                 fuel_mass: float):
        if initial_radius <= 0:
            raise ValueError("Initial radius must be positive")
        if shell_thickness <= 0:
            raise ValueError("Shell thickness must be positive")
        if fuel_mass <= 0:
            raise ValueError("Fuel mass must be positive")

        self.R0 = initial_radius
        self.delta0 = shell_thickness
        self.fuel_mass = fuel_mass

        # DT mass density (solid)
        self.rho_solid = 250  # kg/m³

    @property
    def aspect_ratio(self) -> float:
        """Initial aspect ratio R0/Δ."""
        return self.R0 / self.delta0

    def convergence_ratio(self, compressed_radius: float) -> float:
        """
        Convergence ratio C = R0/R.

        Args:
            compressed_radius: Final compressed radius

        Returns:
            Convergence ratio
        """
        return self.R0 / compressed_radius

    def implosion_velocity(self, ablation_pressure: float) -> float:
        """
        Estimate implosion velocity from rocket equation.

        v_imp ~ √(P/ρ) * ln(C)

        Args:
            ablation_pressure: Ablation pressure (Pa)

        Returns:
            Implosion velocity (m/s)
        """
        rho = self.rho_solid
        return np.sqrt(ablation_pressure / rho)

    def hotspot_temperature(self, implosion_velocity: float) -> float:
        """
        Stagnation hotspot temperature.

        T ~ m_i v²_imp / k_B

        Args:
            implosion_velocity: Implosion velocity (m/s)

        Returns:
            Temperature (K)
        """
        m_DT = 2.5 * m_p  # Average DT mass
        return m_DT * implosion_velocity**2 / k_B

    def areal_density(self, convergence_ratio: float) -> float:
        """
        Areal density ρR of compressed fuel.

        ρR = ρ_0 R_0 C² / 3

        Args:
            convergence_ratio: Convergence ratio

        Returns:
            Areal density (kg/m²)
        """
        return self.rho_solid * self.R0 * convergence_ratio**2 / 3

    def burn_fraction(self, rho_R: float) -> float:
        """
        Approximate burn fraction.

        f_burn ≈ ρR / (ρR + H_B)

        where H_B ≈ 7 g/cm² for DT.

        Args:
            rho_R: Areal density (kg/m²)

        Returns:
            Burn fraction
        """
        H_B = 70  # kg/m² (7 g/cm²)
        return rho_R / (rho_R + H_B)

    def yield_energy(self, burn_fraction: float) -> float:
        """
        Fusion energy yield.

        E = f_burn * m_fuel * E_DT / m_DT

        Args:
            burn_fraction: Fraction of fuel burned

        Returns:
            Yield energy (J)
        """
        E_DT = 17.6e6 * e  # 17.6 MeV
        m_DT = 5 * m_p  # D + T mass

        return burn_fraction * self.fuel_mass * E_DT / m_DT

    def gain(self, yield_energy: float, driver_energy: float) -> float:
        """
        Target gain G = E_yield / E_driver.

        Args:
            yield_energy: Fusion yield (J)
            driver_energy: Driver energy (J)

        Returns:
            Gain factor
        """
        return yield_energy / driver_energy


# =============================================================================
# Astrophysics
# =============================================================================

class HydrostaticStar:
    """
    Hydrostatic equilibrium for stellar structure.

    Solves the stellar structure equations for a star in
    hydrostatic equilibrium:

    dP/dr = -G M(r) ρ / r²
    dM/dr = 4π r² ρ

    Args:
        central_density: Central density (kg/m³)
        central_pressure: Central pressure (Pa)
        composition: Mean molecular weight μ
    """

    def __init__(self, central_density: float, central_pressure: float,
                 composition: float = 0.6):
        if central_density <= 0:
            raise ValueError("Central density must be positive")
        if central_pressure <= 0:
            raise ValueError("Central pressure must be positive")

        self.rho_c = central_density
        self.P_c = central_pressure
        self.mu = composition

    def ideal_gas_temperature(self, P: float, rho: float) -> float:
        """
        Temperature from ideal gas equation of state.

        P = ρ k_B T / (μ m_p)

        Args:
            P: Pressure
            rho: Density

        Returns:
            Temperature
        """
        return P * self.mu * m_p / (rho * k_B)

    def structure_equations(self, r: float, y: np.ndarray) -> np.ndarray:
        """
        Stellar structure ODEs.

        y = [M, P]

        Args:
            r: Radius
            y: State vector [M(r), P(r)]

        Returns:
            Derivatives [dM/dr, dP/dr]
        """
        M, P = y

        if r < 1e-10:
            return np.array([0, 0])

        if P <= 0:
            return np.array([0, 0])

        # Density from polytropic relation
        gamma = 5/3
        K = self.P_c / self.rho_c**gamma
        rho = (P / K)**(1/gamma)

        dMdr = 4 * np.pi * r**2 * rho
        dPdr = -G * M * rho / r**2

        return np.array([dMdr, dPdr])

    def integrate(self, r_max: float = 1e10, n_points: int = 1000) -> Dict[str, np.ndarray]:
        """
        Integrate stellar structure equations.

        Args:
            r_max: Maximum radius to integrate
            n_points: Number of output points

        Returns:
            Dictionary with r, M(r), P(r), rho(r) arrays
        """
        # Initial conditions at small r
        r0 = 1e3
        y0 = [4/3 * np.pi * r0**3 * self.rho_c, self.P_c]

        # Integration
        def stop_surface(r, y):
            return y[1]  # Stop when P = 0
        stop_surface.terminal = True
        stop_surface.direction = -1

        sol = integrate.solve_ivp(
            self.structure_equations,
            [r0, r_max],
            y0,
            events=stop_surface,
            dense_output=True,
            max_step=r_max/100
        )

        # Output on regular grid
        r_surface = sol.t[-1]
        r = np.linspace(r0, r_surface, n_points)
        y = sol.sol(r)

        M = y[0]
        P = y[1]

        # Density from polytropic relation
        gamma = 5/3
        K = self.P_c / self.rho_c**gamma
        rho = np.where(P > 0, (P / K)**(1/gamma), 0)

        return {
            'r': r,
            'M': M,
            'P': P,
            'rho': rho,
            'R_star': r_surface,
            'M_star': M[-1]
        }


class LaneEmden:
    """
    Lane-Emden equation for polytropic stars.

    The Lane-Emden equation describes the structure of a
    self-gravitating polytrope:

    (1/ξ²) d/dξ(ξ² dθ/dξ) + θⁿ = 0

    where P = K ρ^(1+1/n) and θ = (ρ/ρ_c)^(1/n).

    Args:
        n: Polytropic index
    """

    def __init__(self, n: float):
        if n < 0 or n >= 5:
            raise ValueError("Polytropic index must be in [0, 5)")

        self.n = n

    def equation(self, xi: float, y: np.ndarray) -> np.ndarray:
        """
        Lane-Emden equation as first-order system.

        y[0] = θ, y[1] = dθ/dξ

        Args:
            xi: Dimensionless radius
            y: State vector [θ, dθ/dξ]

        Returns:
            Derivatives
        """
        theta, dthetadxi = y

        if xi < 1e-10:
            return np.array([dthetadxi, -1/3])

        if theta <= 0:
            return np.array([dthetadxi, 0])

        d2theta = -2 * dthetadxi / xi - theta**self.n

        return np.array([dthetadxi, d2theta])

    def solve(self, xi_max: float = 20, n_points: int = 1000) -> Dict[str, np.ndarray]:
        """
        Solve Lane-Emden equation.

        Args:
            xi_max: Maximum dimensionless radius
            n_points: Number of output points

        Returns:
            Dictionary with solution
        """
        # Initial conditions: θ(0) = 1, θ'(0) = 0
        xi0 = 1e-6
        theta0 = 1 - xi0**2 / 6  # Taylor expansion
        dtheta0 = -xi0 / 3
        y0 = [theta0, dtheta0]

        # Stop at stellar surface (θ = 0)
        def surface(xi, y):
            return y[0]
        surface.terminal = True
        surface.direction = -1

        sol = integrate.solve_ivp(
            self.equation,
            [xi0, xi_max],
            y0,
            events=surface,
            dense_output=True,
            max_step=xi_max/100
        )

        xi_1 = sol.t[-1]  # Surface dimensionless radius
        xi = np.linspace(xi0, xi_1, n_points)
        y = sol.sol(xi)

        theta = y[0]
        dtheta_dxi = y[1]

        return {
            'xi': xi,
            'theta': theta,
            'dtheta_dxi': dtheta_dxi,
            'xi_1': xi_1,  # First zero of θ
            'minus_xi2_dtheta_1': -xi_1**2 * dtheta_dxi[-1]  # For mass calculation
        }

    def mass_radius_relation(self, K: float, rho_c: float) -> Dict[str, float]:
        """
        Calculate stellar mass and radius from polytropic parameters.

        Args:
            K: Polytropic constant (P = K ρ^(1+1/n))
            rho_c: Central density

        Returns:
            Dictionary with mass and radius
        """
        sol = self.solve()
        xi_1 = sol['xi_1']
        m_factor = sol['minus_xi2_dtheta_1']

        # Length scale
        alpha = np.sqrt((self.n + 1) * K * rho_c**(1/self.n - 1) / (4 * np.pi * G))

        R = alpha * xi_1
        M = 4 * np.pi * alpha**3 * rho_c * m_factor

        return {'M': M, 'R': R, 'rho_c': rho_c}

    @classmethod
    def analytical_solutions(cls, n: int) -> Optional[Callable]:
        """
        Return analytical solution for special cases n = 0, 1, 5.

        Args:
            n: Polytropic index

        Returns:
            Function θ(ξ) if analytical, else None
        """
        if n == 0:
            return lambda xi: 1 - xi**2 / 6
        elif n == 1:
            return lambda xi: np.sin(xi) / xi
        elif n == 5:
            return lambda xi: 1 / np.sqrt(1 + xi**2 / 3)
        else:
            return None


class StellarEvolution:
    """
    Simple stellar evolution model.

    Models main sequence evolution and basic post-main sequence phases
    using scaling relations.

    Args:
        mass: Stellar mass (kg or solar masses if use_solar=True)
        metallicity: Metal fraction Z
        use_solar: If True, mass is in solar masses
    """

    def __init__(self, mass: float, metallicity: float = 0.02,
                 use_solar: bool = True):
        if mass <= 0:
            raise ValueError("Mass must be positive")
        if metallicity < 0 or metallicity > 1:
            raise ValueError("Metallicity must be in [0, 1]")

        self.mass = mass * M_sun if use_solar else mass
        self.mass_solar = self.mass / M_sun
        self.Z = metallicity

    @property
    def luminosity(self) -> float:
        """
        Main sequence luminosity (mass-luminosity relation).

        L/L_sun ≈ (M/M_sun)^α where α varies with mass.
        """
        m = self.mass_solar
        if m < 0.43:
            return 0.23 * m**2.3 * L_sun
        elif m < 2:
            return m**4 * L_sun
        elif m < 55:
            return 1.4 * m**3.5 * L_sun
        else:
            return 32000 * m * L_sun

    @property
    def effective_temperature(self) -> float:
        """
        Main sequence effective temperature.

        T_eff ≈ T_sun * (M/M_sun)^0.5
        """
        T_sun = 5778  # K
        return T_sun * self.mass_solar**0.5

    @property
    def radius(self) -> float:
        """
        Main sequence radius from Stefan-Boltzmann.

        L = 4π R² σ T⁴
        """
        L = self.luminosity
        T = self.effective_temperature
        return np.sqrt(L / (4 * np.pi * sigma_SB * T**4))

    @property
    def main_sequence_lifetime(self) -> float:
        """
        Main sequence lifetime.

        τ_MS ≈ 10¹⁰ yr * (M/M_sun)/(L/L_sun)
        """
        yr = 3.156e7  # seconds
        return 1e10 * yr * self.mass_solar / (self.luminosity / L_sun)

    def spectral_class(self) -> str:
        """Determine spectral class from temperature."""
        T = self.effective_temperature
        if T > 30000:
            return 'O'
        elif T > 10000:
            return 'B'
        elif T > 7500:
            return 'A'
        elif T > 6000:
            return 'F'
        elif T > 5200:
            return 'G'
        elif T > 3700:
            return 'K'
        else:
            return 'M'

    def hr_position(self) -> Tuple[float, float]:
        """
        Position on HR diagram.

        Returns:
            (log(T_eff), log(L/L_sun))
        """
        return (np.log10(self.effective_temperature),
                np.log10(self.luminosity / L_sun))

    def endpoint(self) -> str:
        """Predict stellar endpoint based on mass."""
        m = self.mass_solar
        if m < 0.08:
            return "Brown dwarf (no fusion)"
        elif m < 0.5:
            return "Helium white dwarf"
        elif m < 8:
            return "Carbon-oxygen white dwarf"
        elif m < 25:
            return "Neutron star (via core-collapse supernova)"
        else:
            return "Black hole (via core-collapse supernova)"


class WhiteDwarf:
    """
    White dwarf stellar model.

    White dwarfs are supported by electron degeneracy pressure.
    The mass-radius relation follows from the Chandrasekhar model.

    Args:
        mass: White dwarf mass (kg or solar if use_solar=True)
        composition: Mean molecular weight per electron μ_e (2 for C/O)
        use_solar: If True, mass is in solar masses
    """

    # Chandrasekhar mass
    M_Ch = 1.44 * M_sun

    def __init__(self, mass: float, composition: float = 2.0,
                 use_solar: bool = True):
        self.mass = mass * M_sun if use_solar else mass
        self.mass_solar = self.mass / M_sun
        self.mu_e = composition

        if self.mass > self.M_Ch:
            raise ValueError("Mass exceeds Chandrasekhar limit")

    @property
    def radius(self) -> float:
        """
        White dwarf radius from mass-radius relation.

        R ≈ R_0 * (M/M_Ch)^(-1/3) * (1 - (M/M_Ch)^(4/3))^(1/2)

        where R_0 ≈ 0.0126 R_sun / μ_e^(5/3)
        """
        R_0 = 0.0126 * R_sun / self.mu_e**(5/3)
        x = self.mass / self.M_Ch

        if x >= 0.99:
            return 0  # Collapsed

        return R_0 * x**(-1/3) * np.sqrt(1 - x**(4/3))

    @property
    def central_density(self) -> float:
        """
        Estimate of central density.

        ρ_c ~ M / R³
        """
        R = self.radius
        if R <= 0:
            return np.inf
        return 6 * self.mass / (4 * np.pi * R**3)

    @property
    def surface_gravity(self) -> float:
        """Surface gravitational acceleration."""
        R = self.radius
        if R <= 0:
            return np.inf
        return G * self.mass / R**2

    def cooling_time(self, luminosity: float) -> float:
        """
        Cooling time estimate.

        τ_cool ~ E_thermal / L ~ k_B T N_ions / L

        Args:
            luminosity: Current luminosity

        Returns:
            Cooling time (s)
        """
        # Ion thermal energy
        N_ions = self.mass / (self.mu_e * m_p)
        T_interior = 1e7  # K (typical)
        E_thermal = 1.5 * k_B * T_interior * N_ions

        return E_thermal / luminosity

    def effective_temperature(self, age: float) -> float:
        """
        Effective temperature as function of age (cooling).

        T_eff ~ t^(-2/5) (Mestel cooling)

        Args:
            age: Age since formation (s)

        Returns:
            Effective temperature
        """
        T_0 = 1e5  # K (initial)
        t_0 = 1e8 * 3.156e7  # 100 Myr

        return T_0 * (age / t_0)**(-2/5)


class NeutronStar:
    """
    Neutron star model.

    Neutron stars are supported by neutron degeneracy pressure
    and nuclear forces.

    Args:
        mass: Neutron star mass (kg or solar if use_solar=True)
        use_solar: If True, mass is in solar masses
    """

    # TOV mass limit (approximate)
    M_TOV = 2.2 * M_sun

    def __init__(self, mass: float, use_solar: bool = True):
        self.mass = mass * M_sun if use_solar else mass
        self.mass_solar = self.mass / M_sun

        if self.mass > self.M_TOV:
            raise ValueError("Mass exceeds TOV limit - would collapse to black hole")

    @property
    def radius(self) -> float:
        """
        Neutron star radius (approximately constant ~10-12 km).

        Using simple fit to realistic EOS.
        """
        # R ≈ 11.5 km with weak mass dependence
        return 11.5e3 * (1 - 0.1 * (self.mass_solar - 1.4))

    @property
    def compactness(self) -> float:
        """
        Compactness parameter GM/(Rc²).

        Neutron stars have compactness ~ 0.15-0.25.
        """
        return G * self.mass / (self.radius * c**2)

    @property
    def surface_gravity(self) -> float:
        """Surface gravitational acceleration."""
        return G * self.mass / self.radius**2

    @property
    def gravitational_redshift(self) -> float:
        """
        Surface gravitational redshift.

        z = (1 - 2GM/Rc²)^(-1/2) - 1
        """
        Rs = 2 * G * self.mass / c**2
        return 1 / np.sqrt(1 - Rs / self.radius) - 1

    @property
    def central_density(self) -> float:
        """
        Estimate of central density.

        For neutron stars, ρ_c ~ 5-10 × ρ_nuclear.
        """
        rho_nuclear = 2.3e17  # kg/m³
        return 5 * rho_nuclear * (self.mass_solar / 1.4)

    def spin_down_rate(self, period: float, B_field: float) -> float:
        """
        Magnetic dipole spin-down rate.

        dP/dt = (8π² B² R⁶ sin²α) / (3 I c³ P)

        Args:
            period: Rotation period (s)
            B_field: Surface magnetic field (Tesla)

        Returns:
            Period derivative dP/dt
        """
        # Moment of inertia
        I = 0.4 * self.mass * self.radius**2
        R = self.radius

        # Assuming sin²α = 1 (perpendicular rotator)
        return (8 * np.pi**2 * B_field**2 * R**6) / (3 * I * c**3 * period)

    def characteristic_age(self, period: float, period_dot: float) -> float:
        """
        Characteristic (spin-down) age.

        τ_c = P / (2 dP/dt)

        Args:
            period: Current period
            period_dot: Period derivative

        Returns:
            Characteristic age (s)
        """
        return period / (2 * period_dot)

    def magnetic_field(self, period: float, period_dot: float) -> float:
        """
        Infer magnetic field from spin-down.

        B = 3.2 × 10¹⁹ √(P dP/dt) Gauss

        Args:
            period: Period (s)
            period_dot: Period derivative

        Returns:
            Surface magnetic field (Tesla)
        """
        B_gauss = 3.2e19 * np.sqrt(period * period_dot)
        return B_gauss * 1e-4  # Convert to Tesla


class AccretionDisk:
    """
    Shakura-Sunyaev α-disk model.

    Models a geometrically thin, optically thick accretion disk
    around a compact object.

    Args:
        M: Central object mass (kg)
        mdot: Accretion rate (kg/s)
        alpha: Viscosity parameter (typically 0.01-0.1)
    """

    def __init__(self, M: float, mdot: float, alpha: float = 0.1):
        if M <= 0:
            raise ValueError("Mass must be positive")
        if mdot <= 0:
            raise ValueError("Accretion rate must be positive")
        if alpha <= 0 or alpha > 1:
            raise ValueError("Alpha must be in (0, 1]")

        self.M = M
        self.mdot = mdot
        self.alpha = alpha

    @property
    def eddington_luminosity(self) -> float:
        """Eddington luminosity L_Edd = 4πGMc/κ."""
        kappa = 0.4e-3  # Electron scattering opacity (m²/kg)
        return 4 * np.pi * G * self.M * c / kappa

    @property
    def eddington_mdot(self) -> float:
        """Eddington accretion rate."""
        eta = 0.1  # Radiative efficiency
        return self.eddington_luminosity / (eta * c**2)

    @property
    def inner_radius(self) -> float:
        """Inner disk radius (ISCO for Schwarzschild)."""
        return 6 * G * self.M / c**2

    def temperature(self, r: ArrayLike) -> np.ndarray:
        """
        Disk effective temperature profile.

        T(r) = (3GMṁ/(8πσr³))^(1/4) * (1 - √(r_in/r))^(1/4)

        Args:
            r: Radius array

        Returns:
            Temperature profile
        """
        r = np.asarray(r)
        r_in = self.inner_radius

        T4 = (3 * G * self.M * self.mdot / (8 * np.pi * sigma_SB * r**3))
        f = np.where(r > r_in, 1 - np.sqrt(r_in / r), 0)

        return (T4 * f)**0.25

    def surface_density(self, r: ArrayLike) -> np.ndarray:
        """
        Surface density profile.

        Σ = ṁ / (3π ν)

        where ν = α c_s H is the viscosity.

        Args:
            r: Radius

        Returns:
            Surface density
        """
        r = np.asarray(r)

        # Sound speed from temperature
        T = self.temperature(r)
        c_s = np.sqrt(k_B * T / m_p)

        # Scale height
        H = self.scale_height(r)

        # Viscosity
        nu = self.alpha * c_s * H

        return self.mdot / (3 * np.pi * nu)

    def scale_height(self, r: ArrayLike) -> np.ndarray:
        """
        Disk scale height H = c_s / Ω_K.

        Args:
            r: Radius

        Returns:
            Scale height
        """
        r = np.asarray(r)
        T = self.temperature(r)
        c_s = np.sqrt(k_B * T / m_p)
        Omega_K = np.sqrt(G * self.M / r**3)

        return c_s / Omega_K

    def luminosity(self) -> float:
        """
        Total disk luminosity.

        L = GM ṁ / (2 r_in)
        """
        r_in = self.inner_radius
        return G * self.M * self.mdot / (2 * r_in)

    def spectrum(self, nu: ArrayLike, r_out: float) -> np.ndarray:
        """
        Multi-temperature blackbody spectrum.

        Args:
            nu: Frequency array
            r_out: Outer disk radius

        Returns:
            Flux density (arbitrary units)
        """
        nu = np.asarray(nu)
        r_in = self.inner_radius

        # Integrate over disk
        n_r = 100
        r = np.logspace(np.log10(r_in), np.log10(r_out), n_r)

        F_nu = np.zeros_like(nu)

        for i, ri in enumerate(r[:-1]):
            T = self.temperature(ri)
            dr = r[i+1] - ri
            area = 2 * np.pi * ri * dr

            # Planck function
            x = hbar * 2 * np.pi * nu / (k_B * T)
            B_nu = 2 * hbar * (2*np.pi*nu)**3 / c**2 / (np.exp(x) - 1 + 1e-100)

            F_nu += B_nu * area

        return F_nu


class JetLaunching:
    """
    Basic MHD jet launching model.

    Models jets launched from accretion disks via magnetic fields.
    Implements the Blandford-Payne mechanism basics.

    Args:
        M: Central mass (kg)
        mdot: Disk accretion rate (kg/s)
        B: Magnetic field strength at launch radius (Tesla)
    """

    def __init__(self, M: float, mdot: float, B: float):
        if M <= 0:
            raise ValueError("Mass must be positive")
        if mdot <= 0:
            raise ValueError("Accretion rate must be positive")
        if B <= 0:
            raise ValueError("Magnetic field must be positive")

        self.M = M
        self.mdot = mdot
        self.B = B

    def escape_velocity(self, r: float) -> float:
        """Escape velocity at radius r."""
        return np.sqrt(2 * G * self.M / r)

    def keplerian_velocity(self, r: float) -> float:
        """Keplerian orbital velocity at radius r."""
        return np.sqrt(G * self.M / r)

    def alfven_velocity(self, r: float, density: float) -> float:
        """
        Alfven velocity.

        Args:
            r: Radius (for B(r) ~ r^-1)
            density: Local density
        """
        return self.B / np.sqrt(mu_0 * density)

    def jet_power(self, r_launch: float, efficiency: float = 0.1) -> float:
        """
        Estimate of jet power.

        P_jet ~ η ṁ c²

        Args:
            r_launch: Launch radius
            efficiency: Jet launching efficiency

        Returns:
            Jet power (W)
        """
        return efficiency * self.mdot * c**2

    def magnetic_tower_condition(self, r: float, H: float) -> bool:
        """
        Check if magnetic tower can be launched.

        Requires magnetic pressure to exceed gas pressure.

        Args:
            r: Radius
            H: Disk scale height

        Returns:
            True if magnetically dominated
        """
        # Magnetic pressure
        P_B = self.B**2 / (2 * mu_0)

        # Gas pressure estimate (thin disk)
        disk = AccretionDisk(self.M, self.mdot)
        T = disk.temperature(r)
        rho = disk.surface_density(r) / (2 * H)
        P_gas = rho * k_B * T / m_p

        return P_B > P_gas

    def blandford_znajek_power(self, a: float, r_H: float) -> float:
        """
        Blandford-Znajek power extraction from spinning black hole.

        P_BZ ~ (a/M)² B² r_H² c

        Args:
            a: Dimensionless spin parameter (0 to 1)
            r_H: Horizon radius

        Returns:
            BZ power (W)
        """
        return (a * c / (G * self.M))**2 * self.B**2 * r_H**2 * c


class Nucleosynthesis:
    """
    Basic nucleosynthesis reactions.

    Implements key nuclear reaction rates for stellar
    and primordial nucleosynthesis.

    Args:
        temperature: Temperature in Kelvin
        density: Baryon density in kg/m³
    """

    def __init__(self, temperature: float, density: float):
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        if density <= 0:
            raise ValueError("Density must be positive")

        self.T = temperature
        self.rho = density

    @property
    def T9(self) -> float:
        """Temperature in units of 10⁹ K."""
        return self.T / 1e9

    def pp_chain_rate(self, X_H: float = 0.7) -> float:
        """
        p-p chain energy generation rate.

        ε_pp ≈ ε_0 ρ X² T⁴

        Args:
            X_H: Hydrogen mass fraction

        Returns:
            Energy generation rate (W/kg)
        """
        T6 = self.T / 1e6
        rho_cgs = self.rho * 1e-3  # to g/cm³

        # Approximate p-p rate
        g = 1 + 0.0123 * T6**(1/3) + 0.0109 * T6**(2/3) + 0.000938 * T6
        psi = 1  # Electron screening factor

        f = 0.43 * T6**(-2/3) * np.exp(-33.8 / T6**(1/3))

        return 2.38e6 * psi * f * g * rho_cgs * X_H**2

    def cno_cycle_rate(self, X_H: float = 0.7, X_CNO: float = 0.02) -> float:
        """
        CNO cycle energy generation rate.

        ε_CNO ≈ ε_0 ρ X X_CNO T^16

        Args:
            X_H: Hydrogen mass fraction
            X_CNO: CNO mass fraction

        Returns:
            Energy generation rate (W/kg)
        """
        T6 = self.T / 1e6
        rho_cgs = self.rho * 1e-3

        # Strong temperature dependence
        f = T6**(-2/3) * np.exp(-152.28 / T6**(1/3))

        return 8.67e27 * f * rho_cgs * X_H * X_CNO

    def triple_alpha_rate(self, X_He: float = 0.28) -> float:
        """
        Triple-alpha process rate (3⁴He → ¹²C).

        Args:
            X_He: Helium mass fraction

        Returns:
            Energy generation rate (W/kg)
        """
        T8 = self.T / 1e8
        rho_cgs = self.rho * 1e-3

        # Extreme temperature sensitivity
        f = T8**(-3) * np.exp(-43.2 / T8)

        return 5.1e8 * f * rho_cgs**2 * X_He**3

    def gamow_peak_energy(self, Z1: int, Z2: int, A: float) -> float:
        """
        Gamow peak energy for nuclear reaction.

        E_0 = (π α Z₁ Z₂ k_B T / √(2A))^(2/3)

        Args:
            Z1, Z2: Charge numbers
            A: Reduced mass number

        Returns:
            Gamow peak energy (J)
        """
        alpha_fine = 1/137  # Fine structure constant

        E_G = (np.pi * alpha_fine * Z1 * Z2)**2 * A * m_p * c**2 / 2

        return (E_G * (k_B * self.T)**2)**(1/3)

    def reaction_rate(self, Z1: int, Z2: int, A: float,
                      S_factor: float) -> float:
        """
        Thermonuclear reaction rate.

        <σv> = S(E₀) / (μ E₀) * √(E₀/(3 k_B T)) * exp(-3 E₀/(k_B T))

        Args:
            Z1, Z2: Charge numbers
            A: Reduced mass number
            S_factor: Astrophysical S-factor (keV·barn)

        Returns:
            Reaction rate <σv> (m³/s)
        """
        E_0 = self.gamow_peak_energy(Z1, Z2, A)
        T = self.T

        # Convert S-factor to SI
        S_SI = S_factor * 1e3 * e * 1e-28  # keV·barn to J·m²

        mu = A * m_p

        tau = 3 * E_0 / (k_B * T)
        rate = S_SI / (mu * E_0) * np.sqrt(E_0 / (3 * k_B * T)) * np.exp(-tau)

        return rate


# Module exports
__all__ = [
    # Plasma Fundamentals
    'DebyeLength', 'PlasmaFrequency', 'PlasmaParameter',
    'VlasovEquation', 'LandauDamping',
    # Magnetohydrodynamics
    'MHDEquations', 'AlfvenWave', 'Magnetosonic',
    'MHDInstability', 'MagneticReconnection',
    # Fusion
    'LawsonCriterion', 'TokamakEquilibrium', 'MirrorTrap', 'ICFCapsule',
    # Astrophysics
    'HydrostaticStar', 'LaneEmden', 'StellarEvolution',
    'WhiteDwarf', 'NeutronStar', 'AccretionDisk',
    'JetLaunching', 'Nucleosynthesis',
]
