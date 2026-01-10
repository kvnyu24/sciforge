"""
Waves & Optics Complete Module

This module implements comprehensive wave and optics primitives including:
- Wave Equation Solvers (1D, 2D, 3D, Helmholtz)
- Interference & Diffraction (Young's, gratings, Fraunhofer, Fresnel)
- Geometric Optics (rays, lenses, mirrors, ABCD matrices)
- Polarization (Jones/Mueller/Stokes formalism)
- Nonlinear Optics (SHG, Kerr effect, four-wave mixing, solitons)
- Acoustics (sound waves, impedance, Doppler, resonance)

References:
    - Hecht, "Optics"
    - Born & Wolf, "Principles of Optics"
    - Goodman, "Introduction to Fourier Optics"
    - Boyd, "Nonlinear Optics"
"""

import numpy as np
from typing import Optional, Callable, Tuple, Union, List
from dataclasses import dataclass
from numpy.typing import ArrayLike

from ..core.base import BaseClass, BaseSolver
from ..core.utils import validate_positive, validate_array
from ..core.exceptions import ValidationError, PhysicsError


# ==============================================================================
# Physical Constants
# ==============================================================================

C = 299792458.0              # Speed of light (m/s)
EPSILON_0 = 8.854187817e-12  # Permittivity of free space (F/m)
MU_0 = 4 * np.pi * 1e-7      # Permeability of free space (H/m)


# ==============================================================================
# Phase 3.1: Wave Equation Solvers
# ==============================================================================

class WaveEquation1D(BaseSolver):
    """
    1D Wave equation solver using finite differences.

    ∂²u/∂t² = c² ∂²u/∂x²

    Suitable for strings, pipes, and 1D acoustic propagation.

    Args:
        nx: Number of spatial grid points
        dx: Spatial step size (meters)
        c: Wave speed (m/s)
        dt: Time step (optional, will use CFL if not specified)
        boundary: Boundary condition ('fixed', 'free', 'absorbing', 'periodic')

    Examples:
        >>> wave = WaveEquation1D(nx=200, dx=0.01, c=340)
        >>> wave.set_initial_condition(gaussian_pulse, center=0.5)
        >>> u = wave.run(steps=500)
    """

    def __init__(
        self,
        nx: int,
        dx: float,
        c: float,
        dt: Optional[float] = None,
        boundary: str = 'fixed'
    ):
        super().__init__()

        validate_positive(nx, "nx")
        validate_positive(dx, "dx")
        validate_positive(c, "c")

        self.nx = nx
        self.dx = dx
        self.c = c
        self.boundary = boundary

        # CFL condition: c*dt/dx <= 1
        max_dt = dx / c
        if dt is None:
            self.dt = 0.9 * max_dt
        else:
            if dt > max_dt:
                raise ValidationError(f"dt={dt} exceeds CFL limit {max_dt}")
            self.dt = dt

        self.courant = (c * self.dt / dx)**2

        # Initialize fields: u at time n, n-1
        self.u = np.zeros(nx)
        self.u_prev = np.zeros(nx)

        self._history['time'] = []
        self._history['u'] = []

    def set_initial_condition(
        self,
        displacement: Union[Callable, ArrayLike],
        velocity: Optional[Union[Callable, ArrayLike]] = None
    ):
        """
        Set initial displacement and velocity.

        Args:
            displacement: u(x, 0) - callable or array
            velocity: ∂u/∂t(x, 0) - callable or array (default: 0)
        """
        x = np.linspace(0, (self.nx - 1) * self.dx, self.nx)

        if callable(displacement):
            self.u = np.array([displacement(xi) for xi in x])
        else:
            self.u = np.array(displacement)

        if velocity is not None:
            if callable(velocity):
                v0 = np.array([velocity(xi) for xi in x])
            else:
                v0 = np.array(velocity)
            # Use velocity to compute u at t = -dt
            self.u_prev = self.u - v0 * self.dt
        else:
            self.u_prev = self.u.copy()

    def _apply_boundary(self, u_new: np.ndarray) -> np.ndarray:
        """Apply boundary conditions."""
        if self.boundary == 'fixed':
            u_new[0] = 0
            u_new[-1] = 0
        elif self.boundary == 'free':
            u_new[0] = u_new[1]
            u_new[-1] = u_new[-2]
        elif self.boundary == 'absorbing':
            # Simple absorbing BC
            u_new[0] = self.u[1] + (self.c * self.dt - self.dx) / (self.c * self.dt + self.dx) * (u_new[1] - self.u[0])
            u_new[-1] = self.u[-2] + (self.c * self.dt - self.dx) / (self.c * self.dt + self.dx) * (u_new[-2] - self.u[-1])
        elif self.boundary == 'periodic':
            u_new[0] = u_new[-2]
            u_new[-1] = u_new[1]
        return u_new

    def step(self):
        """Advance one time step."""
        u_new = np.zeros(self.nx)

        # Interior points
        for i in range(1, self.nx - 1):
            u_new[i] = (2 * self.u[i] - self.u_prev[i] +
                        self.courant * (self.u[i+1] - 2*self.u[i] + self.u[i-1]))

        u_new = self._apply_boundary(u_new)

        self.u_prev = self.u.copy()
        self.u = u_new

    def run(self, steps: int, save_every: int = 1) -> np.ndarray:
        """Run simulation for specified steps."""
        for n in range(steps):
            self.step()
            if n % save_every == 0:
                self._history['time'].append(n * self.dt)
                self._history['u'].append(self.u.copy())

        return np.array(self._history['u'])

    def solve(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    def energy(self) -> float:
        """Calculate total energy in the wave."""
        # Kinetic energy: (1/2) ρ (∂u/∂t)²
        # Potential energy: (1/2) τ (∂u/∂x)²
        velocity = (self.u - self.u_prev) / self.dt
        strain = np.gradient(self.u, self.dx)

        KE = 0.5 * np.sum(velocity**2) * self.dx
        PE = 0.5 * self.c**2 * np.sum(strain**2) * self.dx

        return KE + PE


class WaveEquation2D(BaseSolver):
    """
    2D Wave equation solver for membrane vibrations.

    ∂²u/∂t² = c² (∂²u/∂x² + ∂²u/∂y²)

    Args:
        nx, ny: Grid dimensions
        dx, dy: Spatial step sizes
        c: Wave speed
        dt: Time step (optional)
        boundary: Boundary condition type
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        dx: float,
        dy: float,
        c: float,
        dt: Optional[float] = None,
        boundary: str = 'fixed'
    ):
        super().__init__()

        self.nx, self.ny = nx, ny
        self.dx, self.dy = dx, dy
        self.c = c
        self.boundary = boundary

        # 2D CFL condition
        max_dt = 1.0 / (c * np.sqrt(1/dx**2 + 1/dy**2))
        self.dt = 0.9 * max_dt if dt is None else dt

        self.cx2 = (c * self.dt / dx)**2
        self.cy2 = (c * self.dt / dy)**2

        self.u = np.zeros((nx, ny))
        self.u_prev = np.zeros((nx, ny))

        self._history['time'] = []
        self._history['u'] = []

    def set_initial_condition(
        self,
        displacement: Union[Callable, ArrayLike],
        velocity: Optional[Union[Callable, ArrayLike]] = None
    ):
        """Set initial condition u(x, y, 0) and ∂u/∂t(x, y, 0)."""
        if callable(displacement):
            for i in range(self.nx):
                for j in range(self.ny):
                    x = i * self.dx
                    y = j * self.dy
                    self.u[i, j] = displacement(x, y)
        else:
            self.u = np.array(displacement)

        if velocity is not None:
            if callable(velocity):
                v0 = np.zeros((self.nx, self.ny))
                for i in range(self.nx):
                    for j in range(self.ny):
                        x = i * self.dx
                        y = j * self.dy
                        v0[i, j] = velocity(x, y)
            else:
                v0 = np.array(velocity)
            self.u_prev = self.u - v0 * self.dt
        else:
            self.u_prev = self.u.copy()

    def step(self):
        """Advance one time step."""
        u_new = np.zeros((self.nx, self.ny))

        # Interior points
        u_new[1:-1, 1:-1] = (
            2 * self.u[1:-1, 1:-1] - self.u_prev[1:-1, 1:-1] +
            self.cx2 * (self.u[2:, 1:-1] - 2*self.u[1:-1, 1:-1] + self.u[:-2, 1:-1]) +
            self.cy2 * (self.u[1:-1, 2:] - 2*self.u[1:-1, 1:-1] + self.u[1:-1, :-2])
        )

        # Boundary conditions (fixed)
        if self.boundary == 'fixed':
            u_new[0, :] = 0
            u_new[-1, :] = 0
            u_new[:, 0] = 0
            u_new[:, -1] = 0

        self.u_prev = self.u.copy()
        self.u = u_new

    def run(self, steps: int, save_every: int = 10) -> np.ndarray:
        """Run simulation."""
        for n in range(steps):
            self.step()
            if n % save_every == 0:
                self._history['time'].append(n * self.dt)
                self._history['u'].append(self.u.copy())

        return np.array(self._history['u'])

    def solve(self, *args, **kwargs):
        return self.run(*args, **kwargs)


class WaveEquation3D(BaseSolver):
    """
    3D Wave equation solver.

    ∂²u/∂t² = c² ∇²u

    Memory-intensive; use for small domains.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        nz: int,
        dx: float,
        dy: float,
        dz: float,
        c: float,
        dt: Optional[float] = None
    ):
        super().__init__()

        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx, self.dy, self.dz = dx, dy, dz
        self.c = c

        # 3D CFL
        max_dt = 1.0 / (c * np.sqrt(1/dx**2 + 1/dy**2 + 1/dz**2))
        self.dt = 0.9 * max_dt if dt is None else dt

        self.cx2 = (c * self.dt / dx)**2
        self.cy2 = (c * self.dt / dy)**2
        self.cz2 = (c * self.dt / dz)**2

        self.u = np.zeros((nx, ny, nz))
        self.u_prev = np.zeros((nx, ny, nz))

    def step(self):
        """Advance one time step."""
        u_new = np.zeros((self.nx, self.ny, self.nz))

        u_new[1:-1, 1:-1, 1:-1] = (
            2 * self.u[1:-1, 1:-1, 1:-1] - self.u_prev[1:-1, 1:-1, 1:-1] +
            self.cx2 * (self.u[2:, 1:-1, 1:-1] - 2*self.u[1:-1, 1:-1, 1:-1] + self.u[:-2, 1:-1, 1:-1]) +
            self.cy2 * (self.u[1:-1, 2:, 1:-1] - 2*self.u[1:-1, 1:-1, 1:-1] + self.u[1:-1, :-2, 1:-1]) +
            self.cz2 * (self.u[1:-1, 1:-1, 2:] - 2*self.u[1:-1, 1:-1, 1:-1] + self.u[1:-1, 1:-1, :-2])
        )

        self.u_prev = self.u.copy()
        self.u = u_new

    def run(self, steps: int) -> np.ndarray:
        for _ in range(steps):
            self.step()
        return self.u

    def solve(self, *args, **kwargs):
        return self.run(*args, **kwargs)


class HelmholtzSolver(BaseSolver):
    """
    Helmholtz equation solver (time-independent wave equation).

    ∇²u + k²u = f

    where k = ω/c is the wavenumber.

    Args:
        nx, ny: Grid dimensions
        dx, dy: Spatial step sizes
        k: Wavenumber (rad/m)
        boundary: Boundary condition ('dirichlet', 'neumann', 'sommerfeld')
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        dx: float,
        dy: float,
        k: float,
        boundary: str = 'dirichlet'
    ):
        super().__init__()

        self.nx, self.ny = nx, ny
        self.dx, self.dy = dx, dy
        self.k = k
        self.boundary = boundary

        # Solution array
        self.u = np.zeros((nx, ny), dtype=complex)

        # Source term
        self.f = np.zeros((nx, ny), dtype=complex)

    def set_source(self, source: Union[Callable, ArrayLike]):
        """Set source term f(x, y)."""
        if callable(source):
            for i in range(self.nx):
                for j in range(self.ny):
                    x = i * self.dx
                    y = j * self.dy
                    self.f[i, j] = source(x, y)
        else:
            self.f = np.array(source, dtype=complex)

    def solve(self, max_iter: int = 10000, tol: float = 1e-6) -> np.ndarray:
        """
        Solve Helmholtz equation using iterative method (Jacobi).

        Returns:
            Solution array u(x, y)
        """
        dx2 = self.dx**2
        dy2 = self.dy**2
        k2 = self.k**2

        # Coefficient for interior update
        denom = 2/dx2 + 2/dy2 - k2

        for iteration in range(max_iter):
            u_old = self.u.copy()

            # Interior points (Jacobi iteration)
            for i in range(1, self.nx - 1):
                for j in range(1, self.ny - 1):
                    laplacian = (
                        (u_old[i+1, j] + u_old[i-1, j]) / dx2 +
                        (u_old[i, j+1] + u_old[i, j-1]) / dy2
                    )
                    self.u[i, j] = (laplacian - self.f[i, j]) / denom

            # Apply boundary conditions
            if self.boundary == 'dirichlet':
                self.u[0, :] = 0
                self.u[-1, :] = 0
                self.u[:, 0] = 0
                self.u[:, -1] = 0

            # Check convergence
            diff = np.max(np.abs(self.u - u_old))
            if diff < tol:
                break

        return self.u


# ==============================================================================
# Phase 3.2: Interference & Diffraction
# ==============================================================================

class TwoSlitInterference(BaseClass):
    """
    Young's double-slit interference pattern.

    Calculates intensity pattern from two coherent sources.

    Args:
        wavelength: Light wavelength (m)
        slit_separation: Distance between slits (m)
        screen_distance: Distance to observation screen (m)
        slit_width: Width of each slit (m), for finite-width diffraction

    Examples:
        >>> young = TwoSlitInterference(wavelength=500e-9, slit_separation=0.1e-3, screen_distance=1.0)
        >>> x, I = young.intensity_pattern(x_range=(-0.01, 0.01), n_points=500)
    """

    def __init__(
        self,
        wavelength: float,
        slit_separation: float,
        screen_distance: float,
        slit_width: float = 0.0
    ):
        super().__init__()

        validate_positive(wavelength, "wavelength")
        validate_positive(slit_separation, "slit_separation")
        validate_positive(screen_distance, "screen_distance")

        self.wavelength = wavelength
        self.d = slit_separation
        self.L = screen_distance
        self.a = slit_width  # 0 for ideal point sources
        self.k = 2 * np.pi / wavelength

    def path_difference(self, x: float) -> float:
        """Calculate path difference at position x on screen."""
        # For small angles: Δ ≈ d * x / L
        return self.d * x / self.L

    def phase_difference(self, x: float) -> float:
        """Calculate phase difference at position x."""
        return self.k * self.path_difference(x)

    def intensity(self, x: float, include_diffraction: bool = True) -> float:
        """
        Calculate intensity at position x.

        I = I_0 * cos²(π d x / (λ L)) * [sinc(π a x / (λ L))]²
        """
        delta = self.phase_difference(x)

        # Two-slit interference pattern
        I_interference = np.cos(delta / 2)**2

        # Single-slit diffraction envelope
        if include_diffraction and self.a > 0:
            beta = np.pi * self.a * x / (self.wavelength * self.L)
            if abs(beta) < 1e-10:
                I_diffraction = 1.0
            else:
                I_diffraction = (np.sin(beta) / beta)**2
        else:
            I_diffraction = 1.0

        return 4 * I_interference * I_diffraction

    def intensity_pattern(
        self,
        x_range: Tuple[float, float],
        n_points: int = 500
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate intensity pattern over range."""
        x = np.linspace(x_range[0], x_range[1], n_points)
        I = np.array([self.intensity(xi) for xi in x])
        return x, I

    def fringe_spacing(self) -> float:
        """Calculate fringe spacing Δx = λL/d."""
        return self.wavelength * self.L / self.d

    def maxima_positions(self, n_max: int = 10) -> np.ndarray:
        """Find positions of intensity maxima."""
        return np.array([m * self.wavelength * self.L / self.d for m in range(-n_max, n_max + 1)])

    def minima_positions(self, n_max: int = 10) -> np.ndarray:
        """Find positions of intensity minima."""
        return np.array([(m + 0.5) * self.wavelength * self.L / self.d for m in range(-n_max, n_max)])


class MultiSlitInterference(BaseClass):
    """
    Multi-slit (diffraction grating) interference pattern.

    Args:
        wavelength: Light wavelength (m)
        slit_separation: Grating period (m)
        n_slits: Number of slits
        screen_distance: Distance to screen (m)
        slit_width: Individual slit width (m)
    """

    def __init__(
        self,
        wavelength: float,
        slit_separation: float,
        n_slits: int,
        screen_distance: float,
        slit_width: float = 0.0
    ):
        super().__init__()

        self.wavelength = wavelength
        self.d = slit_separation
        self.N = n_slits
        self.L = screen_distance
        self.a = slit_width
        self.k = 2 * np.pi / wavelength

    def intensity(self, theta: float) -> float:
        """
        Calculate intensity at angle θ.

        I = I_0 * [sin(N δ/2) / sin(δ/2)]² * [sinc(β)]²

        where δ = kd sin(θ), β = ka sin(θ)/2
        """
        delta = self.k * self.d * np.sin(theta)
        half_delta = delta / 2

        # N-slit interference factor
        if abs(np.sin(half_delta)) < 1e-10:
            I_Nslit = self.N**2
        else:
            I_Nslit = (np.sin(self.N * half_delta) / np.sin(half_delta))**2

        # Single-slit diffraction envelope
        if self.a > 0:
            beta = self.k * self.a * np.sin(theta) / 2
            if abs(beta) < 1e-10:
                I_diffraction = 1.0
            else:
                I_diffraction = (np.sin(beta) / beta)**2
        else:
            I_diffraction = 1.0

        return I_Nslit * I_diffraction

    def intensity_pattern(
        self,
        theta_range: Tuple[float, float],
        n_points: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate intensity vs angle."""
        theta = np.linspace(theta_range[0], theta_range[1], n_points)
        I = np.array([self.intensity(t) for t in theta])
        return theta, I

    def principal_maxima_angles(self, max_order: int = 5) -> np.ndarray:
        """Find angles of principal maxima: d sin(θ) = m λ."""
        angles = []
        for m in range(-max_order, max_order + 1):
            sin_theta = m * self.wavelength / self.d
            if abs(sin_theta) <= 1:
                angles.append(np.arcsin(sin_theta))
        return np.array(angles)

    def resolving_power(self, order: int = 1) -> float:
        """Calculate resolving power R = mN."""
        return order * self.N

    def angular_dispersion(self, order: int = 1) -> float:
        """Calculate angular dispersion dθ/dλ at order m."""
        # d cos(θ) dθ = m dλ → dθ/dλ = m / (d cos(θ))
        theta_m = np.arcsin(order * self.wavelength / self.d)
        return order / (self.d * np.cos(theta_m))


class SingleSlitDiffraction(BaseClass):
    """
    Single-slit Fraunhofer/Fresnel diffraction.

    Args:
        wavelength: Light wavelength (m)
        slit_width: Slit width (m)
        screen_distance: Distance to screen (m)
        diffraction_type: 'fraunhofer' or 'fresnel'
    """

    def __init__(
        self,
        wavelength: float,
        slit_width: float,
        screen_distance: float,
        diffraction_type: str = 'fraunhofer'
    ):
        super().__init__()

        self.wavelength = wavelength
        self.a = slit_width
        self.L = screen_distance
        self.k = 2 * np.pi / wavelength
        self.diffraction_type = diffraction_type

        # Fresnel number
        self.fresnel_number = slit_width**2 / (wavelength * screen_distance)

    def intensity_fraunhofer(self, x: float) -> float:
        """Calculate Fraunhofer (far-field) diffraction intensity."""
        theta = np.arctan(x / self.L)
        beta = self.k * self.a * np.sin(theta) / 2

        if abs(beta) < 1e-10:
            return 1.0
        return (np.sin(beta) / beta)**2

    def intensity_fresnel(self, x: float, n_zones: int = 100) -> float:
        """
        Calculate Fresnel (near-field) diffraction intensity.

        Uses Fresnel integrals approximation.
        """
        # Fresnel parameters
        sqrt_factor = np.sqrt(2 / (self.wavelength * self.L))

        # Integration limits (Fresnel zones)
        u1 = sqrt_factor * (x - self.a/2)
        u2 = sqrt_factor * (x + self.a/2)

        # Fresnel integrals (numerical approximation)
        C1, S1 = self._fresnel_integrals(u1)
        C2, S2 = self._fresnel_integrals(u2)

        # Intensity
        return 0.5 * ((C2 - C1)**2 + (S2 - S1)**2)

    def _fresnel_integrals(self, u: float, n_terms: int = 50) -> Tuple[float, float]:
        """Compute Fresnel integrals C(u) and S(u)."""
        # Numerical integration (simple trapezoidal)
        t = np.linspace(0, u, max(2, abs(int(u * 100))))
        if len(t) < 2:
            return 0.0, 0.0

        dt = t[1] - t[0]
        C = np.sum(np.cos(np.pi * t**2 / 2)) * dt
        S = np.sum(np.sin(np.pi * t**2 / 2)) * dt

        return C, S

    def intensity(self, x: float) -> float:
        """Calculate intensity at position x."""
        if self.diffraction_type == 'fraunhofer' or self.fresnel_number < 0.1:
            return self.intensity_fraunhofer(x)
        else:
            return self.intensity_fresnel(x)

    def intensity_pattern(
        self,
        x_range: Tuple[float, float],
        n_points: int = 500
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate intensity pattern."""
        x = np.linspace(x_range[0], x_range[1], n_points)
        I = np.array([self.intensity(xi) for xi in x])
        return x, I

    def first_minimum_position(self) -> float:
        """Position of first minimum: x = λL/a."""
        return self.wavelength * self.L / self.a


class CircularAperture(BaseClass):
    """
    Circular aperture diffraction (Airy pattern).

    Args:
        wavelength: Light wavelength (m)
        aperture_diameter: Aperture diameter (m)
        screen_distance: Distance to screen (m)
    """

    def __init__(
        self,
        wavelength: float,
        aperture_diameter: float,
        screen_distance: float
    ):
        super().__init__()

        self.wavelength = wavelength
        self.D = aperture_diameter
        self.L = screen_distance
        self.k = 2 * np.pi / wavelength

    def intensity(self, r: float) -> float:
        """
        Calculate Airy pattern intensity at radial position r.

        I = I_0 * [2 J_1(ka sin θ) / (ka sin θ)]²
        """
        theta = np.arctan(r / self.L)
        x = self.k * self.D * np.sin(theta) / 2

        if abs(x) < 1e-10:
            return 1.0

        # Bessel function J_1 approximation
        J1 = self._bessel_j1(x)

        return (2 * J1 / x)**2

    def _bessel_j1(self, x: float) -> float:
        """Compute Bessel function J_1(x)."""
        # Series expansion for small x
        if abs(x) < 0.01:
            return x / 2

        # Numerical integration
        theta_vals = np.linspace(0, np.pi, 100)
        dtheta = theta_vals[1] - theta_vals[0]
        J1 = np.sum(np.cos(theta_vals - x * np.sin(theta_vals))) * dtheta / np.pi

        return J1

    def airy_disk_radius(self) -> float:
        """
        Calculate radius of first dark ring (Airy disk).

        r_Airy = 1.22 λ L / D
        """
        return 1.22 * self.wavelength * self.L / self.D

    def rayleigh_criterion(self) -> float:
        """Minimum resolvable angle (Rayleigh criterion): θ = 1.22 λ/D."""
        return 1.22 * self.wavelength / self.D

    def intensity_pattern(
        self,
        r_max: float,
        n_points: int = 200
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate radial intensity pattern."""
        r = np.linspace(0, r_max, n_points)
        I = np.array([self.intensity(ri) for ri in r])
        return r, I


class ThinFilmInterference(BaseClass):
    """
    Thin film interference for coating design.

    Args:
        film_thickness: Film thickness (m)
        n_film: Refractive index of film
        n_substrate: Refractive index of substrate
        n_incident: Refractive index of incident medium (default: air)
    """

    def __init__(
        self,
        film_thickness: float,
        n_film: float,
        n_substrate: float,
        n_incident: float = 1.0
    ):
        super().__init__()

        self.d = film_thickness
        self.n1 = n_incident
        self.n2 = n_film
        self.n3 = n_substrate

    def phase_shift(self, wavelength: float, angle: float = 0.0) -> float:
        """Calculate optical phase shift through film."""
        # Angle in film (Snell's law)
        sin_theta2 = self.n1 * np.sin(angle) / self.n2
        cos_theta2 = np.sqrt(1 - sin_theta2**2)

        return 4 * np.pi * self.n2 * self.d * cos_theta2 / wavelength

    def reflectance(self, wavelength: float, angle: float = 0.0, polarization: str = 's') -> float:
        """
        Calculate reflectance at given wavelength and angle.

        Args:
            wavelength: Light wavelength (m)
            angle: Incident angle (radians)
            polarization: 's' or 'p' polarization
        """
        # Fresnel coefficients at each interface
        r12, r23 = self._fresnel_coefficients(angle, polarization)

        # Phase
        delta = self.phase_shift(wavelength, angle)

        # Total reflectance (Airy formula)
        r = (r12 + r23 * np.exp(1j * delta)) / (1 + r12 * r23 * np.exp(1j * delta))

        return np.abs(r)**2

    def _fresnel_coefficients(self, angle: float, polarization: str) -> Tuple[complex, complex]:
        """Calculate Fresnel reflection coefficients."""
        n1, n2, n3 = self.n1, self.n2, self.n3

        # Angles from Snell's law
        sin1 = np.sin(angle)
        cos1 = np.cos(angle)
        sin2 = n1 * sin1 / n2
        cos2 = np.sqrt(1 - sin2**2 + 0j)
        sin3 = n2 * sin2 / n3
        cos3 = np.sqrt(1 - sin3**2 + 0j)

        if polarization == 's':
            r12 = (n1 * cos1 - n2 * cos2) / (n1 * cos1 + n2 * cos2)
            r23 = (n2 * cos2 - n3 * cos3) / (n2 * cos2 + n3 * cos3)
        else:  # p-polarization
            r12 = (n2 * cos1 - n1 * cos2) / (n2 * cos1 + n1 * cos2)
            r23 = (n3 * cos2 - n2 * cos3) / (n3 * cos2 + n2 * cos3)

        return r12, r23

    def antireflection_thickness(self, wavelength: float) -> float:
        """Calculate optimal thickness for antireflection (quarter-wave)."""
        return wavelength / (4 * self.n2)

    def reflectance_spectrum(
        self,
        wavelength_range: Tuple[float, float],
        n_points: int = 200
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate reflectance spectrum."""
        wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], n_points)
        R = np.array([self.reflectance(w) for w in wavelengths])
        return wavelengths, R


class FabryPerotInterferometer(BaseClass):
    """
    Fabry-Perot interferometer / etalon.

    Args:
        spacing: Mirror spacing (m)
        reflectance: Mirror reflectance (0 to 1)
        n_medium: Refractive index between mirrors
    """

    def __init__(
        self,
        spacing: float,
        reflectance: float = 0.9,
        n_medium: float = 1.0
    ):
        super().__init__()

        validate_positive(spacing, "spacing")

        self.d = spacing
        self.R = reflectance
        self.n = n_medium

        # Finesse
        self.finesse = np.pi * np.sqrt(self.R) / (1 - self.R)

    def transmission(self, wavelength: float, angle: float = 0.0) -> float:
        """
        Calculate transmission (Airy function).

        T = 1 / (1 + F sin²(δ/2))

        where F = 4R/(1-R)² and δ = 4πnd cos(θ)/λ
        """
        delta = 4 * np.pi * self.n * self.d * np.cos(angle) / wavelength
        F = 4 * self.R / (1 - self.R)**2

        return 1.0 / (1 + F * np.sin(delta / 2)**2)

    def free_spectral_range(self) -> float:
        """Calculate free spectral range in wavelength (at λ_0)."""
        # FSR = λ²/(2nd)
        # Return in frequency: FSR_f = c/(2nd)
        return C / (2 * self.n * self.d)

    def linewidth(self) -> float:
        """Calculate FWHM linewidth in frequency."""
        return self.free_spectral_range() / self.finesse

    def resolving_power(self, wavelength: float) -> float:
        """Calculate resolving power at given wavelength."""
        return 2 * self.n * self.d / wavelength * self.finesse

    def transmission_spectrum(
        self,
        wavelength_range: Tuple[float, float],
        n_points: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate transmission spectrum."""
        wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], n_points)
        T = np.array([self.transmission(w) for w in wavelengths])
        return wavelengths, T


# ==============================================================================
# Phase 3.3: Geometric Optics
# ==============================================================================

@dataclass
class Ray:
    """
    Ray propagation primitive for geometric optics.

    Attributes:
        position: Ray position [x, y, z]
        direction: Unit direction vector
        wavelength: Light wavelength (for dispersion)
        intensity: Ray intensity (arbitrary units)
    """
    position: np.ndarray
    direction: np.ndarray
    wavelength: float = 550e-9
    intensity: float = 1.0

    def __post_init__(self):
        self.position = np.array(self.position, dtype=float)
        self.direction = np.array(self.direction, dtype=float)
        self.direction = self.direction / np.linalg.norm(self.direction)

    def propagate(self, distance: float) -> 'Ray':
        """Propagate ray by given distance."""
        new_position = self.position + distance * self.direction
        return Ray(new_position, self.direction.copy(), self.wavelength, self.intensity)

    def to_surface(self, surface_z: float) -> 'Ray':
        """Propagate ray to z = surface_z plane."""
        if abs(self.direction[2]) < 1e-15:
            raise PhysicsError("Ray parallel to surface")
        t = (surface_z - self.position[2]) / self.direction[2]
        return self.propagate(t)


class ThinLens(BaseClass):
    """
    Thin lens using paraxial approximation.

    Args:
        focal_length: Focal length (m), positive for converging
        diameter: Lens diameter (m)
        position: Lens position along optical axis (m)

    Examples:
        >>> lens = ThinLens(focal_length=0.1, diameter=0.05)
        >>> ray_out = lens.refract(ray_in)
    """

    def __init__(
        self,
        focal_length: float,
        diameter: float = 0.05,
        position: float = 0.0
    ):
        super().__init__()

        self.f = focal_length
        self.diameter = diameter
        self.z = position

        # Power (diopters)
        self.power = 1.0 / focal_length if focal_length != 0 else 0.0

    def refract(self, ray: Ray) -> Ray:
        """Apply thin lens transformation to ray."""
        # Propagate to lens plane
        ray_at_lens = ray.to_surface(self.z)

        # Check if ray hits lens
        r = np.sqrt(ray_at_lens.position[0]**2 + ray_at_lens.position[1]**2)
        if r > self.diameter / 2:
            raise PhysicsError("Ray misses lens aperture")

        # Paraxial refraction
        # θ_out = θ_in - y/f
        new_direction = ray_at_lens.direction.copy()

        # Small angle approximation
        new_direction[0] -= ray_at_lens.position[0] / self.f
        new_direction[1] -= ray_at_lens.position[1] / self.f

        # Renormalize
        new_direction = new_direction / np.linalg.norm(new_direction)

        return Ray(ray_at_lens.position.copy(), new_direction, ray.wavelength, ray.intensity)

    def image_distance(self, object_distance: float) -> float:
        """Calculate image distance using thin lens equation: 1/f = 1/do + 1/di."""
        if abs(object_distance - self.f) < 1e-15:
            return np.inf  # Object at focal point
        return self.f * object_distance / (object_distance - self.f)

    def magnification(self, object_distance: float) -> float:
        """Calculate lateral magnification m = -di/do."""
        di = self.image_distance(object_distance)
        return -di / object_distance

    def abcd_matrix(self) -> np.ndarray:
        """Return ABCD matrix for the thin lens."""
        return np.array([[1, 0], [-1/self.f, 1]])


class ThickLens(BaseClass):
    """
    Thick lens with cardinal points.

    Args:
        n: Refractive index
        R1: First surface radius of curvature (m)
        R2: Second surface radius of curvature (m)
        thickness: Center thickness (m)
        diameter: Lens diameter (m)
    """

    def __init__(
        self,
        n: float,
        R1: float,
        R2: float,
        thickness: float,
        diameter: float = 0.05
    ):
        super().__init__()

        self.n = n
        self.R1 = R1
        self.R2 = R2
        self.d = thickness
        self.diameter = diameter

        # Lensmaker's equation
        self._compute_cardinal_points()

    def _compute_cardinal_points(self):
        """Compute cardinal points (principal planes, focal points)."""
        n, R1, R2, d = self.n, self.R1, self.R2, self.d

        # Surface powers
        P1 = (n - 1) / R1 if R1 != 0 else 0
        P2 = (1 - n) / R2 if R2 != 0 else 0

        # Effective power
        self.power = P1 + P2 - (d / n) * P1 * P2
        self.f = 1.0 / self.power if self.power != 0 else np.inf

        # Principal plane positions (from first surface)
        if self.power != 0:
            self.H1 = -self.f * (d / n) * P2  # First principal plane
            self.H2 = -self.f * (d / n) * P1  # Second principal plane
        else:
            self.H1 = 0
            self.H2 = d

        # Focal points
        self.F1 = self.H1 - self.f  # Front focal point
        self.F2 = self.H2 + self.f  # Back focal point

    def focal_length(self) -> float:
        """Return effective focal length."""
        return self.f

    def back_focal_length(self) -> float:
        """Return back focal length (from second surface to F2)."""
        return self.f - self.H2 + self.d

    def front_focal_length(self) -> float:
        """Return front focal length (from first surface to F1)."""
        return self.H1 - self.f


class SphericalMirror(BaseClass):
    """
    Spherical mirror for reflection optics.

    Args:
        radius_of_curvature: Mirror radius (m), positive for concave
        diameter: Mirror diameter (m)
    """

    def __init__(
        self,
        radius_of_curvature: float,
        diameter: float = 0.1
    ):
        super().__init__()

        self.R = radius_of_curvature
        self.diameter = diameter
        self.f = radius_of_curvature / 2

    def reflect(self, ray: Ray, mirror_position: float = 0.0) -> Ray:
        """Reflect ray from mirror surface."""
        # Find intersection with sphere (simplified for paraxial)
        ray_at_mirror = ray.to_surface(mirror_position)

        r = np.sqrt(ray_at_mirror.position[0]**2 + ray_at_mirror.position[1]**2)
        if r > self.diameter / 2:
            raise PhysicsError("Ray misses mirror")

        # Paraxial reflection: same as thin lens with f = R/2
        new_direction = ray_at_mirror.direction.copy()
        new_direction[0] -= 2 * ray_at_mirror.position[0] / self.R
        new_direction[1] -= 2 * ray_at_mirror.position[1] / self.R
        new_direction[2] = -new_direction[2]  # Reverse z

        new_direction = new_direction / np.linalg.norm(new_direction)

        return Ray(ray_at_mirror.position.copy(), new_direction, ray.wavelength, ray.intensity)

    def image_distance(self, object_distance: float) -> float:
        """Calculate image distance using mirror equation."""
        return self.f * object_distance / (object_distance - self.f)

    def abcd_matrix(self) -> np.ndarray:
        """Return ABCD matrix for the mirror."""
        return np.array([[1, 0], [-2/self.R, 1]])


class OpticalSystem(BaseClass):
    """
    Optical system using ABCD (ray transfer) matrix formalism.

    Propagates paraxial rays through a sequence of optical elements.

    Examples:
        >>> system = OpticalSystem()
        >>> system.add_element('thin_lens', focal_length=0.1)
        >>> system.add_propagation(0.2)
        >>> y_out, theta_out = system.trace(y_in=0.01, theta_in=0.0)
    """

    def __init__(self):
        super().__init__()
        self.elements: List[Tuple[str, np.ndarray]] = []
        self.total_matrix = np.eye(2)

    def add_element(self, element_type: str, **params):
        """
        Add optical element to system.

        Args:
            element_type: 'thin_lens', 'thick_lens', 'mirror', 'interface'
            **params: Element parameters
        """
        if element_type == 'thin_lens':
            f = params['focal_length']
            M = np.array([[1, 0], [-1/f, 1]])

        elif element_type == 'mirror':
            R = params['radius']
            M = np.array([[1, 0], [-2/R, 1]])

        elif element_type == 'interface':
            n1 = params['n1']
            n2 = params['n2']
            R = params.get('radius', np.inf)
            if R == np.inf:
                M = np.array([[1, 0], [0, n1/n2]])
            else:
                M = np.array([[1, 0], [(n1-n2)/(n2*R), n1/n2]])

        else:
            raise ValidationError(f"Unknown element type: {element_type}")

        self.elements.append((element_type, M))
        self.total_matrix = M @ self.total_matrix

    def add_propagation(self, distance: float, n: float = 1.0):
        """Add free-space propagation."""
        M = np.array([[1, distance/n], [0, 1]])
        self.elements.append(('propagation', M))
        self.total_matrix = M @ self.total_matrix

    def trace(self, y_in: float, theta_in: float) -> Tuple[float, float]:
        """
        Trace ray through system.

        Args:
            y_in: Input ray height
            theta_in: Input ray angle

        Returns:
            (y_out, theta_out)
        """
        ray = np.array([y_in, theta_in])
        ray_out = self.total_matrix @ ray
        return ray_out[0], ray_out[1]

    def effective_focal_length(self) -> float:
        """Calculate effective focal length from ABCD matrix."""
        C = self.total_matrix[1, 0]
        return -1.0 / C if abs(C) > 1e-15 else np.inf

    def magnification(self) -> float:
        """Calculate system magnification."""
        return self.total_matrix[0, 0]


class Prism(BaseClass):
    """
    Prism for dispersion and deviation calculations.

    Args:
        apex_angle: Prism apex angle (radians)
        n: Refractive index (at reference wavelength) or callable n(λ)
        n_sellmeier: Sellmeier coefficients for dispersion [B1, B2, B3, C1, C2, C3]
    """

    def __init__(
        self,
        apex_angle: float,
        n: Union[float, Callable] = 1.5,
        n_sellmeier: Optional[List[float]] = None
    ):
        super().__init__()

        self.A = apex_angle
        self.n_base = n if isinstance(n, float) else n(550e-9)
        self.n_sellmeier = n_sellmeier

        if callable(n):
            self.n_func = n
        elif n_sellmeier is not None:
            self.n_func = self._sellmeier
        else:
            self.n_func = lambda wl: n

    def _sellmeier(self, wavelength: float) -> float:
        """Calculate refractive index using Sellmeier equation."""
        if self.n_sellmeier is None:
            return self.n_base

        B1, B2, B3, C1, C2, C3 = self.n_sellmeier
        wl_um = wavelength * 1e6  # Convert to micrometers
        wl2 = wl_um**2

        n2 = 1 + B1*wl2/(wl2-C1) + B2*wl2/(wl2-C2) + B3*wl2/(wl2-C3)
        return np.sqrt(n2)

    def deviation(self, incident_angle: float, wavelength: float = 550e-9) -> float:
        """
        Calculate total deviation angle.

        δ = i₁ + i₂ - A
        """
        n = self.n_func(wavelength)

        # First refraction
        sin_r1 = np.sin(incident_angle) / n
        if abs(sin_r1) > 1:
            raise PhysicsError("Total internal reflection at first surface")
        r1 = np.arcsin(sin_r1)

        # Inside prism
        r2 = self.A - r1

        # Second refraction
        sin_i2 = n * np.sin(r2)
        if abs(sin_i2) > 1:
            raise PhysicsError("Total internal reflection at second surface")
        i2 = np.arcsin(sin_i2)

        return incident_angle + i2 - self.A

    def minimum_deviation(self, wavelength: float = 550e-9) -> float:
        """Calculate minimum deviation angle."""
        n = self.n_func(wavelength)
        # δ_min = 2 arcsin(n sin(A/2)) - A
        return 2 * np.arcsin(n * np.sin(self.A / 2)) - self.A

    def angular_dispersion(
        self,
        wavelength: float,
        dwavelength: float = 1e-9,
        incident_angle: Optional[float] = None
    ) -> float:
        """Calculate angular dispersion dδ/dλ."""
        if incident_angle is None:
            # Use minimum deviation angle
            n = self.n_func(wavelength)
            incident_angle = np.arcsin(n * np.sin(self.A / 2))

        delta1 = self.deviation(incident_angle, wavelength)
        delta2 = self.deviation(incident_angle, wavelength + dwavelength)

        return (delta2 - delta1) / dwavelength


class SnellRefraction(BaseClass):
    """
    Snell's law refraction at interfaces.

    Args:
        n1: Refractive index of incident medium
        n2: Refractive index of transmitted medium
    """

    def __init__(self, n1: float, n2: float):
        super().__init__()
        self.n1 = n1
        self.n2 = n2

    def refraction_angle(self, incident_angle: float) -> float:
        """Calculate refraction angle from Snell's law."""
        sin_theta2 = self.n1 * np.sin(incident_angle) / self.n2

        if abs(sin_theta2) > 1:
            raise PhysicsError("Total internal reflection")

        return np.arcsin(sin_theta2)

    def critical_angle(self) -> float:
        """Calculate critical angle for total internal reflection (if n1 > n2)."""
        if self.n1 <= self.n2:
            return np.pi / 2  # No TIR possible
        return np.arcsin(self.n2 / self.n1)

    def brewster_angle(self) -> float:
        """Calculate Brewster's angle (no p-polarization reflection)."""
        return np.arctan(self.n2 / self.n1)

    def fresnel_coefficients(
        self,
        incident_angle: float
    ) -> Tuple[complex, complex, complex, complex]:
        """
        Calculate Fresnel reflection and transmission coefficients.

        Returns:
            (r_s, r_p, t_s, t_p) - s and p polarization coefficients
        """
        theta1 = incident_angle
        cos1 = np.cos(theta1)
        sin1 = np.sin(theta1)

        sin2 = self.n1 * sin1 / self.n2
        if abs(sin2) > 1:
            # TIR - complex angle
            cos2 = 1j * np.sqrt(sin2**2 - 1)
        else:
            cos2 = np.sqrt(1 - sin2**2)

        n1, n2 = self.n1, self.n2

        # s-polarization (TE)
        r_s = (n1 * cos1 - n2 * cos2) / (n1 * cos1 + n2 * cos2)
        t_s = 2 * n1 * cos1 / (n1 * cos1 + n2 * cos2)

        # p-polarization (TM)
        r_p = (n2 * cos1 - n1 * cos2) / (n2 * cos1 + n1 * cos2)
        t_p = 2 * n1 * cos1 / (n2 * cos1 + n1 * cos2)

        return r_s, r_p, t_s, t_p

    def reflectance(self, incident_angle: float, polarization: str = 'unpolarized') -> float:
        """Calculate reflectance (power reflection coefficient)."""
        r_s, r_p, _, _ = self.fresnel_coefficients(incident_angle)

        R_s = np.abs(r_s)**2
        R_p = np.abs(r_p)**2

        if polarization == 's':
            return R_s
        elif polarization == 'p':
            return R_p
        else:  # unpolarized
            return 0.5 * (R_s + R_p)


# ==============================================================================
# Phase 3.4: Polarization
# ==============================================================================

class JonesVector(BaseClass):
    """
    Jones vector representation of polarization state.

    E = [Ex, Ey] complex amplitudes

    Args:
        Ex: Complex x-component
        Ey: Complex y-component (or pass array [Ex, Ey])
    """

    def __init__(self, Ex: complex, Ey: Optional[complex] = None):
        super().__init__()

        if Ey is None:
            # Assume Ex is [Ex, Ey]
            self.E = np.array(Ex, dtype=complex)
        else:
            self.E = np.array([Ex, Ey], dtype=complex)

        self._normalize()

    def _normalize(self):
        """Normalize to unit intensity."""
        norm = np.sqrt(np.sum(np.abs(self.E)**2))
        if norm > 0:
            self.E = self.E / norm

    @classmethod
    def horizontal(cls) -> 'JonesVector':
        """Create horizontal linear polarization."""
        return cls(1, 0)

    @classmethod
    def vertical(cls) -> 'JonesVector':
        """Create vertical linear polarization."""
        return cls(0, 1)

    @classmethod
    def diagonal(cls, angle: float = np.pi/4) -> 'JonesVector':
        """Create linear polarization at angle from horizontal."""
        return cls(np.cos(angle), np.sin(angle))

    @classmethod
    def right_circular(cls) -> 'JonesVector':
        """Create right-circular polarization."""
        return cls(1, -1j)

    @classmethod
    def left_circular(cls) -> 'JonesVector':
        """Create left-circular polarization."""
        return cls(1, 1j)

    @classmethod
    def elliptical(cls, a: float, b: float, theta: float = 0) -> 'JonesVector':
        """Create elliptical polarization with semi-axes a, b at angle theta."""
        # Parametric representation
        Ex = a * np.cos(theta) + 1j * b * np.sin(theta)
        Ey = a * np.sin(theta) - 1j * b * np.cos(theta)
        return cls(Ex, Ey)

    def intensity(self) -> float:
        """Calculate intensity."""
        return np.sum(np.abs(self.E)**2)

    def ellipticity(self) -> float:
        """Calculate ellipticity (ratio of minor to major axis)."""
        # From Jones vector to ellipse parameters
        Ex, Ey = self.E
        a2 = np.abs(Ex)**2
        b2 = np.abs(Ey)**2
        ab = Ex * np.conj(Ey)

        # Semi-axes
        A = np.sqrt(0.5 * (a2 + b2 + np.sqrt((a2 - b2)**2 + 4*np.abs(ab)**2)))
        B = np.sqrt(0.5 * (a2 + b2 - np.sqrt((a2 - b2)**2 + 4*np.abs(ab)**2)))

        return B / A if A > 0 else 0

    def orientation_angle(self) -> float:
        """Calculate orientation angle of polarization ellipse."""
        Ex, Ey = self.E
        return 0.5 * np.arctan2(2 * np.real(Ex * np.conj(Ey)),
                                 np.abs(Ex)**2 - np.abs(Ey)**2)


class JonesMatrix(BaseClass):
    """
    Jones matrix for polarization optics.

    Transforms Jones vectors: E_out = M @ E_in

    Args:
        matrix: 2x2 complex matrix or specific element type
    """

    def __init__(self, matrix: Optional[ArrayLike] = None):
        super().__init__()

        if matrix is not None:
            self.M = np.array(matrix, dtype=complex)
        else:
            self.M = np.eye(2, dtype=complex)

    @classmethod
    def linear_polarizer(cls, angle: float = 0) -> 'JonesMatrix':
        """Create linear polarizer at angle θ from horizontal."""
        c = np.cos(angle)
        s = np.sin(angle)
        M = np.array([[c**2, c*s], [c*s, s**2]])
        return cls(M)

    @classmethod
    def quarter_wave_plate(cls, fast_axis_angle: float = 0) -> 'JonesMatrix':
        """Create quarter-wave plate with fast axis at given angle."""
        return cls.waveplate(np.pi/2, fast_axis_angle)

    @classmethod
    def half_wave_plate(cls, fast_axis_angle: float = 0) -> 'JonesMatrix':
        """Create half-wave plate with fast axis at given angle."""
        return cls.waveplate(np.pi, fast_axis_angle)

    @classmethod
    def waveplate(cls, retardance: float, fast_axis_angle: float = 0) -> 'JonesMatrix':
        """Create general waveplate with given retardance."""
        c = np.cos(fast_axis_angle)
        s = np.sin(fast_axis_angle)
        R = np.array([[c, s], [-s, c]])  # Rotation to fast axis frame

        # Phase retardation
        phase = np.exp(1j * retardance / 2)
        W = np.array([[1/phase, 0], [0, phase]])

        M = R.T @ W @ R
        return cls(M)

    @classmethod
    def rotator(cls, angle: float) -> 'JonesMatrix':
        """Create polarization rotator."""
        c = np.cos(angle)
        s = np.sin(angle)
        M = np.array([[c, -s], [s, c]])
        return cls(M)

    def apply(self, jones_vector: JonesVector) -> JonesVector:
        """Apply Jones matrix to Jones vector."""
        E_out = self.M @ jones_vector.E
        return JonesVector(E_out)

    def __matmul__(self, other: Union['JonesMatrix', JonesVector]) -> Union['JonesMatrix', JonesVector]:
        """Matrix multiplication."""
        if isinstance(other, JonesVector):
            return self.apply(other)
        elif isinstance(other, JonesMatrix):
            return JonesMatrix(self.M @ other.M)
        else:
            return JonesMatrix(self.M @ np.array(other))


class StokesVector(BaseClass):
    """
    Stokes vector representation of polarization state.

    S = [S0, S1, S2, S3] = [I, Q, U, V]

    Can represent partially polarized light.
    """

    def __init__(self, S0: float, S1: float = 0, S2: float = 0, S3: float = 0):
        super().__init__()
        self.S = np.array([S0, S1, S2, S3])

    @classmethod
    def from_jones(cls, jones: JonesVector) -> 'StokesVector':
        """Convert Jones vector to Stokes vector."""
        Ex, Ey = jones.E
        S0 = np.abs(Ex)**2 + np.abs(Ey)**2
        S1 = np.abs(Ex)**2 - np.abs(Ey)**2
        S2 = 2 * np.real(Ex * np.conj(Ey))
        S3 = 2 * np.imag(Ex * np.conj(Ey))
        return cls(S0, S1, S2, S3)

    @classmethod
    def unpolarized(cls, intensity: float = 1.0) -> 'StokesVector':
        """Create unpolarized light."""
        return cls(intensity, 0, 0, 0)

    @classmethod
    def horizontal(cls, intensity: float = 1.0) -> 'StokesVector':
        """Create horizontally polarized light."""
        return cls(intensity, intensity, 0, 0)

    @classmethod
    def vertical(cls, intensity: float = 1.0) -> 'StokesVector':
        """Create vertically polarized light."""
        return cls(intensity, -intensity, 0, 0)

    @classmethod
    def right_circular(cls, intensity: float = 1.0) -> 'StokesVector':
        """Create right-circularly polarized light."""
        return cls(intensity, 0, 0, intensity)

    def intensity(self) -> float:
        """Return total intensity S0."""
        return self.S[0]

    def degree_of_polarization(self) -> float:
        """Calculate degree of polarization DOP = sqrt(S1² + S2² + S3²) / S0."""
        if self.S[0] <= 0:
            return 0.0
        return np.sqrt(self.S[1]**2 + self.S[2]**2 + self.S[3]**2) / self.S[0]

    def degree_of_linear_polarization(self) -> float:
        """Calculate degree of linear polarization."""
        if self.S[0] <= 0:
            return 0.0
        return np.sqrt(self.S[1]**2 + self.S[2]**2) / self.S[0]

    def degree_of_circular_polarization(self) -> float:
        """Calculate degree of circular polarization."""
        if self.S[0] <= 0:
            return 0.0
        return abs(self.S[3]) / self.S[0]


class MuellerMatrix(BaseClass):
    """
    Mueller matrix for transformation of Stokes vectors.

    S_out = M @ S_in

    Can handle partially polarized light (unlike Jones matrices).
    """

    def __init__(self, matrix: Optional[ArrayLike] = None):
        super().__init__()

        if matrix is not None:
            self.M = np.array(matrix, dtype=float)
        else:
            self.M = np.eye(4)

    @classmethod
    def from_jones(cls, jones: JonesMatrix) -> 'MuellerMatrix':
        """Convert Jones matrix to Mueller matrix."""
        J = jones.M
        # Mueller-Jones transformation
        U = np.array([
            [1, 0, 0, 1],
            [1, 0, 0, -1],
            [0, 1, 1, 0],
            [0, 1j, -1j, 0]
        ]) / np.sqrt(2)

        U_inv = np.linalg.inv(U)
        M = np.real(U @ np.kron(J, np.conj(J)) @ U_inv)
        return cls(M)

    @classmethod
    def linear_polarizer(cls, angle: float = 0) -> 'MuellerMatrix':
        """Create Mueller matrix for linear polarizer."""
        c2 = np.cos(2 * angle)
        s2 = np.sin(2 * angle)
        M = 0.5 * np.array([
            [1, c2, s2, 0],
            [c2, c2**2, c2*s2, 0],
            [s2, c2*s2, s2**2, 0],
            [0, 0, 0, 0]
        ])
        return cls(M)

    @classmethod
    def quarter_wave_plate(cls, fast_axis_angle: float = 0) -> 'MuellerMatrix':
        """Create Mueller matrix for quarter-wave plate."""
        c2 = np.cos(2 * fast_axis_angle)
        s2 = np.sin(2 * fast_axis_angle)
        M = np.array([
            [1, 0, 0, 0],
            [0, c2**2, c2*s2, -s2],
            [0, c2*s2, s2**2, c2],
            [0, s2, -c2, 0]
        ])
        return cls(M)

    @classmethod
    def depolarizer(cls, depolarization: float = 1.0) -> 'MuellerMatrix':
        """Create depolarizer (reduces DOP)."""
        d = 1 - depolarization
        M = np.diag([1, d, d, d])
        return cls(M)

    def apply(self, stokes: StokesVector) -> StokesVector:
        """Apply Mueller matrix to Stokes vector."""
        S_out = self.M @ stokes.S
        return StokesVector(S_out[0], S_out[1], S_out[2], S_out[3])

    def __matmul__(self, other: Union['MuellerMatrix', StokesVector]) -> Union['MuellerMatrix', StokesVector]:
        """Matrix multiplication."""
        if isinstance(other, StokesVector):
            return self.apply(other)
        elif isinstance(other, MuellerMatrix):
            return MuellerMatrix(self.M @ other.M)
        else:
            return MuellerMatrix(self.M @ np.array(other))


class Waveplate(BaseClass):
    """
    General waveplate (retarder) element.

    Args:
        retardance: Phase retardance (radians)
        fast_axis_angle: Orientation of fast axis (radians)
        wavelength: Design wavelength (m)
    """

    def __init__(
        self,
        retardance: float,
        fast_axis_angle: float = 0,
        wavelength: float = 550e-9
    ):
        super().__init__()

        self.retardance = retardance
        self.fast_axis = fast_axis_angle
        self.wavelength = wavelength

        self.jones = JonesMatrix.waveplate(retardance, fast_axis_angle)
        self.mueller = MuellerMatrix.from_jones(self.jones)

    def transform_jones(self, jones_in: JonesVector) -> JonesVector:
        """Transform Jones vector."""
        return self.jones.apply(jones_in)

    def transform_stokes(self, stokes_in: StokesVector) -> StokesVector:
        """Transform Stokes vector."""
        return self.mueller.apply(stokes_in)

    def retardance_at_wavelength(self, wavelength: float) -> float:
        """Calculate retardance at different wavelength (dispersion)."""
        # Simple dispersion: retardance ∝ 1/λ
        return self.retardance * self.wavelength / wavelength


class Polarizer(BaseClass):
    """
    Polarizer element (linear or circular).

    Args:
        polarizer_type: 'linear', 'right_circular', 'left_circular'
        angle: Orientation angle for linear polarizer (radians)
        extinction_ratio: Ratio of transmitted to blocked (ideal = inf)
    """

    def __init__(
        self,
        polarizer_type: str = 'linear',
        angle: float = 0,
        extinction_ratio: float = 1e6
    ):
        super().__init__()

        self.polarizer_type = polarizer_type
        self.angle = angle
        self.extinction_ratio = extinction_ratio

        if polarizer_type == 'linear':
            self.jones = JonesMatrix.linear_polarizer(angle)
            self.mueller = MuellerMatrix.linear_polarizer(angle)
        elif polarizer_type == 'right_circular':
            # Circular polarizer = linear + QWP
            self.jones = JonesMatrix.quarter_wave_plate(-np.pi/4) @ JonesMatrix.linear_polarizer(0)
            self.mueller = MuellerMatrix.from_jones(self.jones)
        elif polarizer_type == 'left_circular':
            self.jones = JonesMatrix.quarter_wave_plate(np.pi/4) @ JonesMatrix.linear_polarizer(0)
            self.mueller = MuellerMatrix.from_jones(self.jones)
        else:
            raise ValidationError(f"Unknown polarizer type: {polarizer_type}")

    def transform_jones(self, jones_in: JonesVector) -> JonesVector:
        """Transform Jones vector."""
        return self.jones.apply(jones_in)

    def transform_stokes(self, stokes_in: StokesVector) -> StokesVector:
        """Transform Stokes vector."""
        return self.mueller.apply(stokes_in)


# ==============================================================================
# Phase 3.5: Nonlinear Optics
# ==============================================================================

class SecondHarmonicGeneration(BaseClass):
    """
    Second Harmonic Generation (SHG / frequency doubling).

    χ(2) nonlinear process: ω + ω → 2ω

    Args:
        d_eff: Effective nonlinear coefficient (m/V)
        crystal_length: Crystal length (m)
        n_omega: Refractive index at fundamental
        n_2omega: Refractive index at second harmonic
        wavelength: Fundamental wavelength (m)

    Examples:
        >>> shg = SecondHarmonicGeneration(d_eff=2e-12, crystal_length=0.01,
        ...                                 n_omega=1.65, n_2omega=1.68)
        >>> eta = shg.conversion_efficiency(intensity=1e12)  # 1 GW/cm²
    """

    def __init__(
        self,
        d_eff: float,
        crystal_length: float,
        n_omega: float,
        n_2omega: float,
        wavelength: float = 1064e-9
    ):
        super().__init__()

        self.d_eff = d_eff
        self.L = crystal_length
        self.n_omega = n_omega
        self.n_2omega = n_2omega
        self.lambda_omega = wavelength

        # Phase mismatch
        k_omega = 2 * np.pi * n_omega / wavelength
        k_2omega = 2 * np.pi * n_2omega / (wavelength / 2)
        self.delta_k = k_2omega - 2 * k_omega

        # Coherence length
        self.L_coh = np.pi / abs(self.delta_k) if self.delta_k != 0 else np.inf

    def sinc_factor(self) -> float:
        """Calculate phase matching sinc factor."""
        arg = self.delta_k * self.L / 2
        if abs(arg) < 1e-10:
            return 1.0
        return (np.sin(arg) / arg)**2

    def conversion_efficiency(self, intensity: float) -> float:
        """
        Calculate SHG conversion efficiency η = I_2ω / I_ω.

        Low-conversion approximation.

        Args:
            intensity: Fundamental intensity (W/m²)
        """
        # η = (8 π² d_eff² L² / (n_ω² n_2ω λ² ε₀ c)) * I_ω * sinc²(ΔkL/2)
        eta = (8 * np.pi**2 * self.d_eff**2 * self.L**2 /
               (self.n_omega**2 * self.n_2omega * self.lambda_omega**2 *
                EPSILON_0 * C))

        return eta * intensity * self.sinc_factor()

    def second_harmonic_power(self, input_power: float, beam_area: float) -> float:
        """Calculate output SHG power."""
        intensity = input_power / beam_area
        eta = self.conversion_efficiency(intensity)
        return eta * input_power

    def phase_matching_angle(self, crystal_type: str = 'uniaxial') -> float:
        """
        Calculate phase matching angle for birefringent crystal.

        Simplified for Type I phase matching in uniaxial crystal.
        """
        if abs(self.delta_k) < 1e-10:
            return 0.0  # Already phase matched

        # Would need full crystal properties for accurate calculation
        # Return placeholder
        return np.arcsin(self.lambda_omega * self.delta_k / (4 * np.pi * (self.n_2omega - self.n_omega)))


class KerrEffect(BaseClass):
    """
    Kerr effect (optical Kerr / self-focusing).

    χ(3) nonlinear process causing intensity-dependent refractive index:
    n = n_0 + n_2 * I

    Args:
        n0: Linear refractive index
        n2: Nonlinear refractive index (m²/W)
        wavelength: Light wavelength (m)

    Examples:
        >>> kerr = KerrEffect(n0=1.45, n2=2.6e-20, wavelength=800e-9)  # Fused silica
        >>> delta_n = kerr.index_change(intensity=1e16)
        >>> P_cr = kerr.critical_power()
    """

    def __init__(
        self,
        n0: float,
        n2: float,
        wavelength: float = 800e-9
    ):
        super().__init__()

        self.n0 = n0
        self.n2 = n2
        self.wavelength = wavelength

    def index_change(self, intensity: float) -> float:
        """Calculate nonlinear index change Δn = n₂ I."""
        return self.n2 * intensity

    def effective_index(self, intensity: float) -> float:
        """Calculate total effective index n = n₀ + n₂ I."""
        return self.n0 + self.index_change(intensity)

    def nonlinear_phase(self, intensity: float, length: float) -> float:
        """Calculate accumulated nonlinear phase φ_NL = k n₂ I L."""
        return 2 * np.pi * self.n2 * intensity * length / self.wavelength

    def critical_power(self) -> float:
        """
        Calculate critical power for self-focusing.

        P_cr = 3.77 λ² / (8 π n₀ n₂)
        """
        return 3.77 * self.wavelength**2 / (8 * np.pi * self.n0 * self.n2)

    def self_focusing_length(self, power: float, beam_radius: float) -> float:
        """
        Calculate self-focusing collapse distance.

        z_sf ≈ 0.367 k w₀² / sqrt(P/P_cr - 1)
        """
        P_cr = self.critical_power()
        if power <= P_cr:
            return np.inf

        k = 2 * np.pi * self.n0 / self.wavelength
        return 0.367 * k * beam_radius**2 / np.sqrt(power / P_cr - 1)

    def b_integral(self, intensity: float, length: float) -> float:
        """Calculate B-integral (accumulated nonlinear phase in radians)."""
        return self.nonlinear_phase(intensity, length)


class FourWaveMixing(BaseClass):
    """
    Four-wave mixing (FWM) parametric process.

    χ(3) process: ω₁ + ω₂ → ω₃ + ω₄

    For degenerate case: 2ω_p → ω_s + ω_i (signal + idler)

    Args:
        chi3: Third-order susceptibility (m²/V²)
        n: Refractive index
        length: Interaction length (m)
        pump_wavelength: Pump wavelength (m)
    """

    def __init__(
        self,
        chi3: float,
        n: float,
        length: float,
        pump_wavelength: float = 1550e-9
    ):
        super().__init__()

        self.chi3 = chi3
        self.n = n
        self.L = length
        self.lambda_p = pump_wavelength

        # Pump frequency
        self.omega_p = 2 * np.pi * C / pump_wavelength

    def parametric_gain(
        self,
        pump_power: float,
        effective_area: float,
        signal_detuning: float = 0.0
    ) -> float:
        """
        Calculate parametric gain coefficient.

        Args:
            pump_power: Pump power (W)
            effective_area: Effective mode area (m²)
            signal_detuning: Signal frequency detuning from pump (rad/s)
        """
        # Nonlinear coefficient
        gamma = 2 * np.pi * self.n**2 * self.chi3 / (self.lambda_p * effective_area)

        # Simplified gain (no dispersion)
        g = gamma * pump_power / effective_area

        return g

    def idler_wavelength(self, signal_wavelength: float) -> float:
        """Calculate idler wavelength for energy conservation."""
        omega_s = 2 * np.pi * C / signal_wavelength
        omega_i = 2 * self.omega_p - omega_s
        return 2 * np.pi * C / omega_i

    def phase_matching_bandwidth(self, dispersion: float) -> float:
        """
        Calculate phase matching bandwidth.

        Args:
            dispersion: Group velocity dispersion β₂ (s²/m)
        """
        if abs(dispersion) < 1e-30:
            return np.inf
        return np.sqrt(4 / (abs(dispersion) * self.L))


class SolitonPulse(BaseClass):
    """
    Optical soliton in nonlinear dispersive media.

    Fundamental soliton solution of NLSE:
    A(z,t) = A_0 sech(t/T_0) exp(i z / (2 L_D))

    Args:
        peak_power: Soliton peak power (W)
        pulse_duration: Pulse duration T_0 (s)
        n2: Nonlinear refractive index (m²/W)
        beta2: GVD parameter (s²/m)
        effective_area: Effective mode area (m²)
        wavelength: Center wavelength (m)

    Examples:
        >>> soliton = SolitonPulse(peak_power=1e3, pulse_duration=100e-15,
        ...                         n2=2.6e-20, beta2=-20e-27, effective_area=80e-12)
        >>> N = soliton.soliton_order()
    """

    def __init__(
        self,
        peak_power: float,
        pulse_duration: float,
        n2: float,
        beta2: float,
        effective_area: float,
        wavelength: float = 1550e-9
    ):
        super().__init__()

        self.P0 = peak_power
        self.T0 = pulse_duration
        self.n2 = n2
        self.beta2 = beta2
        self.A_eff = effective_area
        self.wavelength = wavelength

        # Derived parameters
        self.gamma = 2 * np.pi * n2 / (wavelength * effective_area)  # Nonlinear coefficient

        # Characteristic lengths
        self.L_D = pulse_duration**2 / abs(beta2)  # Dispersion length
        self.L_NL = 1 / (self.gamma * peak_power)  # Nonlinear length

    def soliton_order(self) -> float:
        """
        Calculate soliton order N.

        N² = L_D / L_NL = γ P₀ T₀² / |β₂|
        """
        return np.sqrt(self.L_D / self.L_NL)

    def is_fundamental(self, tolerance: float = 0.1) -> bool:
        """Check if soliton is fundamental (N ≈ 1)."""
        return abs(self.soliton_order() - 1.0) < tolerance

    def pulse_profile(self, t: ArrayLike) -> np.ndarray:
        """Calculate soliton pulse intensity profile |A(t)|²."""
        t = np.array(t)
        return self.P0 * (1 / np.cosh(t / self.T0))**2

    def soliton_period(self) -> float:
        """Calculate soliton period z_0 = π L_D / 2."""
        return np.pi * self.L_D / 2

    def fundamental_soliton_power(self) -> float:
        """Calculate power required for fundamental soliton (N=1)."""
        return abs(self.beta2) / (self.gamma * self.T0**2)


# ==============================================================================
# Phase 3.6: Acoustics
# ==============================================================================

class SoundWave(BaseClass):
    """
    Sound wave propagation in a medium.

    Args:
        frequency: Wave frequency (Hz)
        medium: Medium type ('air', 'water', 'steel') or dict with properties
        temperature: Temperature (K), for temperature-dependent speed

    Examples:
        >>> sound = SoundWave(frequency=1000, medium='air', temperature=293)
        >>> c = sound.speed()
        >>> lambda_s = sound.wavelength()
    """

    MEDIUM_PROPERTIES = {
        'air': {'density': 1.225, 'bulk_modulus': 1.42e5},
        'water': {'density': 1000, 'bulk_modulus': 2.2e9},
        'steel': {'density': 7850, 'bulk_modulus': 1.6e11},
        'aluminum': {'density': 2700, 'bulk_modulus': 7.6e10},
        'glass': {'density': 2500, 'bulk_modulus': 4.0e10},
    }

    def __init__(
        self,
        frequency: float,
        medium: Union[str, dict] = 'air',
        temperature: float = 293.15
    ):
        super().__init__()

        validate_positive(frequency, "frequency")

        self.frequency = frequency
        self.T = temperature

        if isinstance(medium, str):
            if medium not in self.MEDIUM_PROPERTIES:
                raise ValidationError(f"Unknown medium: {medium}")
            props = self.MEDIUM_PROPERTIES[medium]
            self.density = props['density']
            self.bulk_modulus = props['bulk_modulus']
            self.medium_name = medium
        else:
            self.density = medium['density']
            self.bulk_modulus = medium['bulk_modulus']
            self.medium_name = 'custom'

    def speed(self) -> float:
        """Calculate speed of sound c = √(K/ρ)."""
        if self.medium_name == 'air':
            # Temperature-dependent for air: c ≈ 331.3 √(T/273.15)
            return 331.3 * np.sqrt(self.T / 273.15)
        return np.sqrt(self.bulk_modulus / self.density)

    def wavelength(self) -> float:
        """Calculate wavelength λ = c/f."""
        return self.speed() / self.frequency

    def wavenumber(self) -> float:
        """Calculate wavenumber k = 2π/λ."""
        return 2 * np.pi / self.wavelength()

    def angular_frequency(self) -> float:
        """Calculate angular frequency ω = 2πf."""
        return 2 * np.pi * self.frequency

    def intensity(self, pressure_amplitude: float) -> float:
        """
        Calculate intensity from pressure amplitude.

        I = p²/(2ρc)
        """
        c = self.speed()
        return pressure_amplitude**2 / (2 * self.density * c)

    def pressure_amplitude(self, intensity: float) -> float:
        """Calculate pressure amplitude from intensity."""
        c = self.speed()
        return np.sqrt(2 * self.density * c * intensity)

    def sound_pressure_level(self, intensity: float, I_ref: float = 1e-12) -> float:
        """Calculate sound pressure level in dB."""
        return 10 * np.log10(intensity / I_ref)


class AcousticImpedance(BaseClass):
    """
    Acoustic impedance and material matching.

    Args:
        density: Medium density (kg/m³)
        speed: Speed of sound (m/s)
    """

    def __init__(self, density: float, speed: float):
        super().__init__()

        self.density = density
        self.c = speed
        self.Z = density * speed  # Characteristic impedance

    def impedance(self) -> float:
        """Return acoustic impedance Z = ρc."""
        return self.Z

    def reflection_coefficient(self, Z2: float) -> float:
        """
        Calculate pressure reflection coefficient at interface.

        r = (Z₂ - Z₁) / (Z₂ + Z₁)
        """
        return (Z2 - self.Z) / (Z2 + self.Z)

    def transmission_coefficient(self, Z2: float) -> float:
        """Calculate pressure transmission coefficient."""
        return 2 * Z2 / (Z2 + self.Z)

    def intensity_reflection(self, Z2: float) -> float:
        """Calculate intensity reflection coefficient R = r²."""
        r = self.reflection_coefficient(Z2)
        return r**2

    def intensity_transmission(self, Z2: float) -> float:
        """Calculate intensity transmission coefficient T = 1 - R."""
        return 1 - self.intensity_reflection(Z2)

    @classmethod
    def for_medium(cls, medium: str) -> 'AcousticImpedance':
        """Create impedance for standard medium."""
        props = SoundWave.MEDIUM_PROPERTIES.get(medium)
        if props is None:
            raise ValidationError(f"Unknown medium: {medium}")
        density = props['density']
        speed = np.sqrt(props['bulk_modulus'] / density)
        return cls(density, speed)


class DopplerShift(BaseClass):
    """
    Doppler effect for moving source/observer.

    Args:
        source_frequency: Emitted frequency (Hz)
        speed_of_sound: Sound speed in medium (m/s)
    """

    def __init__(
        self,
        source_frequency: float,
        speed_of_sound: float = 343.0
    ):
        super().__init__()

        self.f_s = source_frequency
        self.c = speed_of_sound

    def observed_frequency(
        self,
        v_source: float = 0.0,
        v_observer: float = 0.0,
        approaching: bool = True
    ) -> float:
        """
        Calculate observed frequency with Doppler shift.

        Args:
            v_source: Source velocity (m/s, positive = toward observer)
            v_observer: Observer velocity (m/s, positive = toward source)
            approaching: If True, source and observer approaching

        f_obs = f_s * (c + v_obs) / (c - v_src)  (approaching)
        """
        if approaching:
            return self.f_s * (self.c + v_observer) / (self.c - v_source)
        else:
            return self.f_s * (self.c - v_observer) / (self.c + v_source)

    def frequency_shift(
        self,
        v_source: float = 0.0,
        v_observer: float = 0.0,
        approaching: bool = True
    ) -> float:
        """Calculate frequency shift Δf = f_obs - f_s."""
        return self.observed_frequency(v_source, v_observer, approaching) - self.f_s

    def mach_number(self, velocity: float) -> float:
        """Calculate Mach number M = v/c."""
        return velocity / self.c

    def shock_cone_angle(self, velocity: float) -> float:
        """
        Calculate shock cone half-angle for supersonic source.

        sin(θ) = c/v = 1/M
        """
        M = self.mach_number(velocity)
        if M <= 1:
            return np.pi / 2  # No shock cone for subsonic
        return np.arcsin(1 / M)


class ResonantCavity(BaseClass):
    """
    Acoustic resonant cavity (standing wave modes).

    Args:
        length: Cavity length (m)
        boundary: Boundary condition ('open-open', 'closed-closed', 'open-closed')
        speed_of_sound: Sound speed (m/s)
    """

    def __init__(
        self,
        length: float,
        boundary: str = 'closed-closed',
        speed_of_sound: float = 343.0
    ):
        super().__init__()

        self.L = length
        self.boundary = boundary
        self.c = speed_of_sound

    def resonant_frequencies(self, n_modes: int = 10) -> np.ndarray:
        """Calculate resonant frequencies for first n modes."""
        frequencies = []

        for n in range(1, n_modes + 1):
            if self.boundary == 'closed-closed' or self.boundary == 'open-open':
                # f_n = n c / (2L)
                f = n * self.c / (2 * self.L)
            elif self.boundary == 'open-closed':
                # Only odd harmonics: f_n = (2n-1) c / (4L)
                f = (2 * n - 1) * self.c / (4 * self.L)
            else:
                raise ValidationError(f"Unknown boundary type: {self.boundary}")
            frequencies.append(f)

        return np.array(frequencies)

    def fundamental_frequency(self) -> float:
        """Return fundamental (lowest) resonant frequency."""
        return self.resonant_frequencies(1)[0]

    def mode_wavelength(self, mode_number: int) -> float:
        """Calculate wavelength for given mode."""
        f = self.resonant_frequencies(mode_number)[mode_number - 1]
        return self.c / f

    def quality_factor(self, damping_coefficient: float) -> float:
        """
        Calculate quality factor Q.

        Q = ω₀ / (2α) where α is damping coefficient
        """
        omega_0 = 2 * np.pi * self.fundamental_frequency()
        return omega_0 / (2 * damping_coefficient)


class Ultrasound(BaseClass):
    """
    Ultrasound wave propagation and attenuation.

    Args:
        frequency: Ultrasound frequency (Hz), typically > 20 kHz
        medium: Propagation medium
    """

    # Attenuation coefficients (dB/cm/MHz)
    ATTENUATION = {
        'water': 0.002,
        'blood': 0.2,
        'soft_tissue': 0.5,
        'muscle': 1.0,
        'bone': 5.0,
        'lung': 40.0,
    }

    def __init__(
        self,
        frequency: float,
        medium: str = 'water'
    ):
        super().__init__()

        if frequency < 20e3:
            raise ValidationError("Ultrasound frequency must be > 20 kHz")

        self.frequency = frequency
        self.f_MHz = frequency / 1e6
        self.medium = medium

        # Get attenuation
        self.alpha_0 = self.ATTENUATION.get(medium, 0.5)  # dB/cm/MHz

    def attenuation_coefficient(self) -> float:
        """Calculate attenuation coefficient in dB/cm."""
        return self.alpha_0 * self.f_MHz

    def intensity_at_depth(self, I_0: float, depth: float) -> float:
        """
        Calculate intensity at given depth.

        Args:
            I_0: Initial intensity
            depth: Penetration depth (m)

        I(z) = I_0 exp(-2αz)
        """
        alpha_per_m = self.attenuation_coefficient() * 100  # Convert to dB/m
        alpha_neper = alpha_per_m / 8.686  # Convert dB to Neper

        return I_0 * np.exp(-2 * alpha_neper * depth)

    def half_value_depth(self) -> float:
        """Calculate depth at which intensity drops to half."""
        alpha_per_m = self.attenuation_coefficient() * 100
        return np.log(2) * 8.686 / (2 * alpha_per_m)

    def wavelength(self, speed: float = 1540.0) -> float:
        """Calculate wavelength (default speed for soft tissue)."""
        return speed / self.frequency

    def resolution(self, speed: float = 1540.0) -> float:
        """Estimate axial resolution (≈ λ/2)."""
        return self.wavelength(speed) / 2
