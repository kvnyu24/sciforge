"""
Electromagnetism Deep Dive Module

This module implements advanced electromagnetic primitives including:
- Maxwell's Equations Solvers (FDTD methods)
- Electromagnetic Potentials (scalar, vector, gauge freedom)
- Radiation (dipole, Larmor, synchrotron, Cherenkov, bremsstrahlung)
- Multipole Expansions
- Material Properties (dielectric, magnetic, conductor, plasma, metamaterial)

References:
    - Griffiths, "Introduction to Electrodynamics"
    - Jackson, "Classical Electrodynamics"
    - Taflove & Hagness, "Computational Electrodynamics: The FDTD Method"
"""

import numpy as np
from typing import Optional, Callable, Tuple, Union, List
from dataclasses import dataclass
from numpy.typing import ArrayLike

from ..core.base import BaseClass, BaseSolver
from ..core.utils import validate_positive, validate_array
from ..core.exceptions import ValidationError, PhysicsError
from .base import Field


# ==============================================================================
# Physical Constants for EM
# ==============================================================================

EPSILON_0 = 8.854187817e-12  # Permittivity of free space (F/m)
MU_0 = 4 * np.pi * 1e-7      # Permeability of free space (H/m)
C = 299792458.0              # Speed of light (m/s)
E_CHARGE = 1.602176634e-19   # Elementary charge (C)


# ==============================================================================
# Phase 2.1: Maxwell's Equations Solvers
# ==============================================================================

class MaxwellSolver1D(BaseSolver):
    """
    1D Finite-Difference Time-Domain (FDTD) solver for Maxwell's equations.

    Solves the 1D wave equation for E and H fields using the Yee algorithm.
    Suitable for transmission line problems and 1D wave propagation.

    Args:
        nx: Number of spatial grid points
        dx: Spatial step size (meters)
        dt: Time step size (seconds), must satisfy CFL condition
        epsilon_r: Relative permittivity (can be array for inhomogeneous media)
        mu_r: Relative permeability (can be array for inhomogeneous media)
        sigma: Conductivity for lossy media (S/m)
        boundary: Boundary condition type ('pec', 'pmc', 'abc', 'periodic')

    Examples:
        >>> solver = MaxwellSolver1D(nx=200, dx=1e-3, dt=1e-12)
        >>> solver.add_source(position=100, source_type='gaussian', params={'width': 20})
        >>> E, H = solver.run(steps=500)
    """

    def __init__(
        self,
        nx: int,
        dx: float,
        dt: Optional[float] = None,
        epsilon_r: Union[float, ArrayLike] = 1.0,
        mu_r: Union[float, ArrayLike] = 1.0,
        sigma: Union[float, ArrayLike] = 0.0,
        boundary: str = 'abc'
    ):
        super().__init__()

        validate_positive(nx, "nx")
        validate_positive(dx, "dx")

        self.nx = nx
        self.dx = dx

        # CFL condition for stability: dt <= dx / c
        max_dt = dx / C
        if dt is None:
            self.dt = 0.99 * max_dt
        else:
            if dt > max_dt:
                raise ValidationError(f"dt={dt} exceeds CFL limit {max_dt}")
            self.dt = dt

        # Material properties (can be spatially varying)
        self.epsilon_r = np.ones(nx) * epsilon_r if np.isscalar(epsilon_r) else np.array(epsilon_r)
        self.mu_r = np.ones(nx) * mu_r if np.isscalar(mu_r) else np.array(mu_r)
        self.sigma = np.zeros(nx) + sigma if np.isscalar(sigma) else np.array(sigma)

        if boundary not in ('pec', 'pmc', 'abc', 'periodic'):
            raise ValidationError(f"Unknown boundary type: {boundary}")
        self.boundary = boundary

        # Initialize fields
        self.E = np.zeros(nx)      # Electric field (Ey)
        self.H = np.zeros(nx - 1)  # Magnetic field (Hz) - staggered grid

        # Update coefficients
        self._compute_coefficients()

        # Source list
        self.sources: List[dict] = []

        # ABC boundary storage
        self._E_left_prev = 0.0
        self._E_right_prev = 0.0

        self._history['time'] = []
        self._history['E'] = []
        self._history['H'] = []

    def _compute_coefficients(self):
        """Compute update coefficients for lossy media."""
        # E-field update coefficients
        loss = self.sigma * self.dt / (2 * EPSILON_0 * self.epsilon_r)
        self.Ca = (1 - loss) / (1 + loss)
        self.Cb = (self.dt / (EPSILON_0 * self.epsilon_r * self.dx)) / (1 + loss)

        # H-field update coefficients (at staggered positions)
        epsilon_avg = 0.5 * (self.epsilon_r[:-1] + self.epsilon_r[1:])
        mu_avg = 0.5 * (self.mu_r[:-1] + self.mu_r[1:])
        self.Da = np.ones(self.nx - 1)
        self.Db = self.dt / (MU_0 * mu_avg * self.dx)

    def add_source(
        self,
        position: int,
        source_type: str = 'gaussian',
        params: Optional[dict] = None
    ):
        """
        Add an electromagnetic source to the simulation.

        Args:
            position: Grid index for source location
            source_type: Type of source ('gaussian', 'sinusoidal', 'ricker', 'custom')
            params: Source parameters (width, frequency, amplitude, etc.)
        """
        if position < 0 or position >= self.nx:
            raise ValidationError(f"Source position {position} out of bounds")

        params = params or {}
        self.sources.append({
            'position': position,
            'type': source_type,
            'params': params
        })

    def _get_source_value(self, source: dict, t: float) -> float:
        """Calculate source value at time t."""
        params = source['params']
        source_type = source['type']

        if source_type == 'gaussian':
            width = params.get('width', 30)
            t0 = params.get('t0', 3 * width * self.dt)
            amplitude = params.get('amplitude', 1.0)
            return amplitude * np.exp(-((t - t0) / (width * self.dt))**2)

        elif source_type == 'sinusoidal':
            freq = params.get('frequency', 1e9)
            amplitude = params.get('amplitude', 1.0)
            return amplitude * np.sin(2 * np.pi * freq * t)

        elif source_type == 'ricker':
            # Ricker wavelet (Mexican hat)
            fp = params.get('peak_frequency', 1e9)
            t0 = params.get('t0', 1.0 / fp)
            amplitude = params.get('amplitude', 1.0)
            arg = (np.pi * fp * (t - t0))**2
            return amplitude * (1 - 2 * arg) * np.exp(-arg)

        elif source_type == 'custom':
            func = params.get('function')
            if func is None:
                raise ValidationError("Custom source requires 'function' parameter")
            return func(t)

        else:
            raise ValidationError(f"Unknown source type: {source_type}")

    def _apply_boundary(self):
        """Apply boundary conditions."""
        if self.boundary == 'pec':
            # Perfect Electric Conductor: E = 0 at boundary
            self.E[0] = 0
            self.E[-1] = 0

        elif self.boundary == 'pmc':
            # Perfect Magnetic Conductor: dE/dn = 0 at boundary
            self.E[0] = self.E[1]
            self.E[-1] = self.E[-2]

        elif self.boundary == 'abc':
            # Absorbing Boundary Condition (Mur's first-order)
            c_ratio = (C * self.dt - self.dx) / (C * self.dt + self.dx)
            E_left_new = self._E_left_prev + c_ratio * (self.E[1] - self.E[0])
            E_right_new = self._E_right_prev + c_ratio * (self.E[-2] - self.E[-1])

            self._E_left_prev = self.E[1]
            self._E_right_prev = self.E[-2]

            self.E[0] = E_left_new
            self.E[-1] = E_right_new

        elif self.boundary == 'periodic':
            # Periodic boundary
            self.E[0] = self.E[-2]
            self.E[-1] = self.E[1]

    def step(self, t: float):
        """
        Perform one FDTD time step.

        Args:
            t: Current simulation time
        """
        # Update H-field
        self.H = self.Da * self.H - self.Db * (self.E[1:] - self.E[:-1])

        # Update E-field
        self.E[1:-1] = (self.Ca[1:-1] * self.E[1:-1] -
                        self.Cb[1:-1] * (self.H[1:] - self.H[:-1]))

        # Add sources
        for source in self.sources:
            pos = source['position']
            self.E[pos] += self._get_source_value(source, t)

        # Apply boundary conditions
        self._apply_boundary()

    def run(self, steps: int, save_every: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the FDTD simulation for specified number of steps.

        Args:
            steps: Number of time steps to run
            save_every: Save fields every N steps (for memory efficiency)

        Returns:
            Tuple of (E_history, H_history) arrays
        """
        for n in range(steps):
            t = n * self.dt
            self.step(t)

            if n % save_every == 0:
                self._history['time'].append(t)
                self._history['E'].append(self.E.copy())
                self._history['H'].append(self.H.copy())

        return np.array(self._history['E']), np.array(self._history['H'])

    def solve(self, *args, **kwargs):
        """Alias for run() to satisfy BaseSolver interface."""
        return self.run(*args, **kwargs)


class MaxwellSolver2D(BaseSolver):
    """
    2D Finite-Difference Time-Domain (FDTD) solver for Maxwell's equations.

    Implements TM mode (Ez, Hx, Hy) or TE mode (Hz, Ex, Ey) propagation.
    Uses the Yee algorithm with staggered grids.

    Args:
        nx, ny: Number of spatial grid points in x and y
        dx, dy: Spatial step sizes (meters)
        dt: Time step size (seconds), must satisfy 2D CFL condition
        mode: Polarization mode ('TM' or 'TE')
        epsilon_r: Relative permittivity (can be 2D array)
        mu_r: Relative permeability (can be 2D array)
        pml_layers: Number of PML absorbing layers (0 for none)

    Examples:
        >>> solver = MaxwellSolver2D(nx=100, ny=100, dx=1e-3, dy=1e-3)
        >>> solver.add_source(position=(50, 50), source_type='gaussian')
        >>> Ez = solver.run(steps=200)
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        dx: float,
        dy: float,
        dt: Optional[float] = None,
        mode: str = 'TM',
        epsilon_r: Union[float, ArrayLike] = 1.0,
        mu_r: Union[float, ArrayLike] = 1.0,
        pml_layers: int = 10
    ):
        super().__init__()

        validate_positive(nx, "nx")
        validate_positive(ny, "ny")
        validate_positive(dx, "dx")
        validate_positive(dy, "dy")

        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.mode = mode
        self.pml_layers = pml_layers

        # 2D CFL condition: dt <= 1 / (c * sqrt(1/dx^2 + 1/dy^2))
        max_dt = 1.0 / (C * np.sqrt(1/dx**2 + 1/dy**2))
        if dt is None:
            self.dt = 0.99 * max_dt
        else:
            if dt > max_dt:
                raise ValidationError(f"dt={dt} exceeds 2D CFL limit {max_dt}")
            self.dt = dt

        # Material properties
        if np.isscalar(epsilon_r):
            self.epsilon_r = np.ones((nx, ny)) * epsilon_r
        else:
            self.epsilon_r = np.array(epsilon_r)

        if np.isscalar(mu_r):
            self.mu_r = np.ones((nx, ny)) * mu_r
        else:
            self.mu_r = np.array(mu_r)

        # Initialize fields based on mode
        if mode == 'TM':
            # TM mode: Ez, Hx, Hy
            self.Ez = np.zeros((nx, ny))
            self.Hx = np.zeros((nx, ny - 1))
            self.Hy = np.zeros((nx - 1, ny))
        else:
            # TE mode: Hz, Ex, Ey
            self.Hz = np.zeros((nx, ny))
            self.Ex = np.zeros((nx, ny - 1))
            self.Ey = np.zeros((nx - 1, ny))

        # PML conductivity profiles
        if pml_layers > 0:
            self._setup_pml()

        self.sources: List[dict] = []
        self._history['time'] = []
        self._history['field'] = []

    def _setup_pml(self):
        """Setup Perfectly Matched Layer absorbing boundaries."""
        n = self.pml_layers

        # Polynomial grading
        sigma_max = 0.8 * (3 + 1) / (np.sqrt(MU_0 / EPSILON_0) * self.dx)

        # Create conductivity profiles
        self.sigma_x = np.zeros(self.nx)
        self.sigma_y = np.zeros(self.ny)

        for i in range(n):
            sigma = sigma_max * ((n - i) / n)**3
            self.sigma_x[i] = sigma
            self.sigma_x[-(i+1)] = sigma
            self.sigma_y[i] = sigma
            self.sigma_y[-(i+1)] = sigma

    def add_source(
        self,
        position: Tuple[int, int],
        source_type: str = 'gaussian',
        params: Optional[dict] = None
    ):
        """Add a point source at the specified grid position."""
        ix, iy = position
        if ix < 0 or ix >= self.nx or iy < 0 or iy >= self.ny:
            raise ValidationError(f"Source position {position} out of bounds")

        params = params or {}
        self.sources.append({
            'position': position,
            'type': source_type,
            'params': params
        })

    def _get_source_value(self, source: dict, t: float) -> float:
        """Calculate source value at time t."""
        params = source['params']
        source_type = source['type']

        if source_type == 'gaussian':
            width = params.get('width', 30)
            t0 = params.get('t0', 3 * width * self.dt)
            amplitude = params.get('amplitude', 1.0)
            return amplitude * np.exp(-((t - t0) / (width * self.dt))**2)

        elif source_type == 'sinusoidal':
            freq = params.get('frequency', 1e9)
            amplitude = params.get('amplitude', 1.0)
            return amplitude * np.sin(2 * np.pi * freq * t)

        else:
            return 0.0

    def step(self, t: float):
        """Perform one 2D FDTD time step."""
        if self.mode == 'TM':
            self._step_tm(t)
        else:
            self._step_te(t)

    def _step_tm(self, t: float):
        """TM mode update (Ez, Hx, Hy)."""
        dt, dx, dy = self.dt, self.dx, self.dy

        # Update Hx: dHx/dt = -(1/mu) * dEz/dy
        self.Hx -= (dt / (MU_0 * self.mu_r[:, :-1])) * (self.Ez[:, 1:] - self.Ez[:, :-1]) / dy

        # Update Hy: dHy/dt = (1/mu) * dEz/dx
        self.Hy += (dt / (MU_0 * self.mu_r[:-1, :])) * (self.Ez[1:, :] - self.Ez[:-1, :]) / dx

        # Update Ez: dEz/dt = (1/eps) * (dHy/dx - dHx/dy)
        dHy_dx = np.zeros_like(self.Ez)
        dHx_dy = np.zeros_like(self.Ez)

        dHy_dx[1:-1, :] = (self.Hy[1:, :] - self.Hy[:-1, :]) / dx
        dHx_dy[:, 1:-1] = (self.Hx[:, 1:] - self.Hx[:, :-1]) / dy

        self.Ez += (dt / (EPSILON_0 * self.epsilon_r)) * (dHy_dx - dHx_dy)

        # Add sources
        for source in self.sources:
            ix, iy = source['position']
            self.Ez[ix, iy] += self._get_source_value(source, t)

        # Simple ABC boundary (zero tangential E)
        self.Ez[0, :] = 0
        self.Ez[-1, :] = 0
        self.Ez[:, 0] = 0
        self.Ez[:, -1] = 0

    def _step_te(self, t: float):
        """TE mode update (Hz, Ex, Ey)."""
        dt, dx, dy = self.dt, self.dx, self.dy

        # Update Ex: dEx/dt = (1/eps) * dHz/dy
        self.Ex += (dt / (EPSILON_0 * self.epsilon_r[:, :-1])) * (self.Hz[:, 1:] - self.Hz[:, :-1]) / dy

        # Update Ey: dEy/dt = -(1/eps) * dHz/dx
        self.Ey -= (dt / (EPSILON_0 * self.epsilon_r[:-1, :])) * (self.Hz[1:, :] - self.Hz[:-1, :]) / dx

        # Update Hz
        dEy_dx = np.zeros_like(self.Hz)
        dEx_dy = np.zeros_like(self.Hz)

        dEy_dx[1:-1, :] = (self.Ey[1:, :] - self.Ey[:-1, :]) / dx
        dEx_dy[:, 1:-1] = (self.Ex[:, 1:] - self.Ex[:, :-1]) / dy

        self.Hz -= (dt / (MU_0 * self.mu_r)) * (dEy_dx - dEx_dy)

        # Add sources
        for source in self.sources:
            ix, iy = source['position']
            self.Hz[ix, iy] += self._get_source_value(source, t)

    def run(self, steps: int, save_every: int = 10) -> np.ndarray:
        """Run the 2D FDTD simulation."""
        for n in range(steps):
            t = n * self.dt
            self.step(t)

            if n % save_every == 0:
                self._history['time'].append(t)
                if self.mode == 'TM':
                    self._history['field'].append(self.Ez.copy())
                else:
                    self._history['field'].append(self.Hz.copy())

        return np.array(self._history['field'])

    def solve(self, *args, **kwargs):
        """Alias for run() to satisfy BaseSolver interface."""
        return self.run(*args, **kwargs)


class MaxwellSolver3D(BaseSolver):
    """
    3D Finite-Difference Time-Domain (FDTD) solver for Maxwell's equations.

    Full 3D implementation of the Yee algorithm with all six field components.
    Memory intensive - use for small domains or with subgridding.

    Args:
        nx, ny, nz: Number of spatial grid points
        dx, dy, dz: Spatial step sizes (meters)
        dt: Time step size (must satisfy 3D CFL condition)
        epsilon_r: Relative permittivity (3D array or scalar)
        mu_r: Relative permeability (3D array or scalar)
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        nz: int,
        dx: float,
        dy: float,
        dz: float,
        dt: Optional[float] = None,
        epsilon_r: Union[float, ArrayLike] = 1.0,
        mu_r: Union[float, ArrayLike] = 1.0
    ):
        super().__init__()

        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx, self.dy, self.dz = dx, dy, dz

        # 3D CFL condition
        max_dt = 1.0 / (C * np.sqrt(1/dx**2 + 1/dy**2 + 1/dz**2))
        self.dt = 0.99 * max_dt if dt is None else dt

        if dt is not None and dt > max_dt:
            raise ValidationError(f"dt exceeds 3D CFL limit {max_dt}")

        # Material arrays
        shape = (nx, ny, nz)
        self.epsilon_r = np.ones(shape) * epsilon_r if np.isscalar(epsilon_r) else np.array(epsilon_r)
        self.mu_r = np.ones(shape) * mu_r if np.isscalar(mu_r) else np.array(mu_r)

        # Initialize all six field components (staggered grid)
        self.Ex = np.zeros((nx, ny-1, nz-1))
        self.Ey = np.zeros((nx-1, ny, nz-1))
        self.Ez = np.zeros((nx-1, ny-1, nz))
        self.Hx = np.zeros((nx-1, ny, nz))
        self.Hy = np.zeros((nx, ny-1, nz))
        self.Hz = np.zeros((nx, ny, nz-1))

        self.sources: List[dict] = []
        self._history['time'] = []

    def add_source(
        self,
        position: Tuple[int, int, int],
        component: str = 'Ez',
        source_type: str = 'gaussian',
        params: Optional[dict] = None
    ):
        """Add a source to a specific field component."""
        params = params or {}
        self.sources.append({
            'position': position,
            'component': component,
            'type': source_type,
            'params': params
        })

    def step(self, t: float):
        """Perform one 3D FDTD time step."""
        dt = self.dt
        dx, dy, dz = self.dx, self.dy, self.dz

        # Update H fields (curl E)
        # Hx: dHx/dt = (1/mu) * (dEy/dz - dEz/dy)
        self.Hx += (dt / (MU_0 * self.mu_r[:-1, :, :])) * (
            (self.Ey[:, :, 1:] - self.Ey[:, :, :-1]) / dz -
            (self.Ez[:, 1:, :] - self.Ez[:, :-1, :]) / dy
        )

        # Hy: dHy/dt = (1/mu) * (dEz/dx - dEx/dz)
        self.Hy += (dt / (MU_0 * self.mu_r[:, :-1, :])) * (
            (self.Ez[1:, :, :] - self.Ez[:-1, :, :]) / dx -
            (self.Ex[:, :, 1:] - self.Ex[:, :, :-1]) / dz
        )

        # Hz: dHz/dt = (1/mu) * (dEx/dy - dEy/dx)
        self.Hz += (dt / (MU_0 * self.mu_r[:, :, :-1])) * (
            (self.Ex[:, 1:, :] - self.Ex[:, :-1, :]) / dy -
            (self.Ey[1:, :, :] - self.Ey[:-1, :, :]) / dx
        )

        # Update E fields (curl H)
        # Ex: dEx/dt = (1/eps) * (dHz/dy - dHy/dz)
        self.Ex[:, :, :] += (dt / (EPSILON_0 * self.epsilon_r[:, :-1, :-1])) * (
            (self.Hz[:, 1:, :] - self.Hz[:, :-1, :]) / dy -
            (self.Hy[:, :, 1:] - self.Hy[:, :, :-1]) / dz
        )

        # Ey: dEy/dt = (1/eps) * (dHx/dz - dHz/dx)
        self.Ey[:, :, :] += (dt / (EPSILON_0 * self.epsilon_r[:-1, :, :-1])) * (
            (self.Hx[:, :, 1:] - self.Hx[:, :, :-1]) / dz -
            (self.Hz[1:, :, :] - self.Hz[:-1, :, :]) / dx
        )

        # Ez: dEz/dt = (1/eps) * (dHy/dx - dHx/dy)
        self.Ez[:, :, :] += (dt / (EPSILON_0 * self.epsilon_r[:-1, :-1, :])) * (
            (self.Hy[1:, :, :] - self.Hy[:-1, :, :]) / dx -
            (self.Hx[:, 1:, :] - self.Hx[:, :-1, :]) / dy
        )

        # Add sources (simplified)
        for source in self.sources:
            ix, iy, iz = source['position']
            component = source['component']
            params = source['params']

            width = params.get('width', 30)
            t0 = params.get('t0', 3 * width * self.dt)
            amplitude = params.get('amplitude', 1.0)
            value = amplitude * np.exp(-((t - t0) / (width * self.dt))**2)

            if component == 'Ez' and ix < self.Ez.shape[0] and iy < self.Ez.shape[1]:
                self.Ez[ix, iy, iz] += value

    def run(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the 3D FDTD simulation."""
        for n in range(steps):
            t = n * self.dt
            self.step(t)
            self._history['time'].append(t)

        return self.Ex, self.Ey, self.Ez

    def solve(self, *args, **kwargs):
        return self.run(*args, **kwargs)


class GaussLaw(BaseClass):
    """
    Gauss's Law solver for electrostatics.

    Computes electric flux through surfaces and charge enclosed.
    ∮ E · dA = Q_enclosed / ε₀

    Args:
        charge_distribution: Callable returning charge density ρ(r)

    Examples:
        >>> gauss = GaussLaw(lambda r: 1e-9 if np.linalg.norm(r) < 0.1 else 0)
        >>> flux = gauss.flux_through_sphere(radius=0.2)
    """

    def __init__(self, charge_distribution: Optional[Callable[[np.ndarray], float]] = None):
        super().__init__()
        self.rho = charge_distribution or (lambda r: 0.0)

    def charge_enclosed(
        self,
        center: ArrayLike,
        radius: float,
        n_points: int = 50
    ) -> float:
        """
        Calculate charge enclosed in a sphere.

        Uses numerical integration over spherical volume.
        """
        center = np.array(center)
        total_charge = 0.0

        # Spherical integration
        r_vals = np.linspace(0, radius, n_points)
        theta_vals = np.linspace(0, np.pi, n_points)
        phi_vals = np.linspace(0, 2*np.pi, 2*n_points)

        dr = r_vals[1] - r_vals[0] if len(r_vals) > 1 else radius
        dtheta = theta_vals[1] - theta_vals[0] if len(theta_vals) > 1 else np.pi
        dphi = phi_vals[1] - phi_vals[0] if len(phi_vals) > 1 else 2*np.pi

        for r in r_vals:
            for theta in theta_vals:
                for phi in phi_vals:
                    x = center[0] + r * np.sin(theta) * np.cos(phi)
                    y = center[1] + r * np.sin(theta) * np.sin(phi)
                    z = center[2] + r * np.cos(theta)

                    rho_val = self.rho(np.array([x, y, z]))
                    dV = r**2 * np.sin(theta) * dr * dtheta * dphi
                    total_charge += rho_val * dV

        return total_charge

    def flux_through_sphere(
        self,
        center: ArrayLike = (0, 0, 0),
        radius: float = 1.0,
        n_points: int = 50
    ) -> float:
        """Calculate electric flux through a spherical surface."""
        Q = self.charge_enclosed(center, radius, n_points)
        return Q / EPSILON_0

    def electric_field_spherical(
        self,
        center: ArrayLike,
        radius: float,
        n_points: int = 50
    ) -> float:
        """
        Calculate radial electric field at radius using Gauss's law.

        Assumes spherical symmetry.
        """
        Q = self.charge_enclosed(center, radius, n_points)
        return Q / (4 * np.pi * EPSILON_0 * radius**2)


class FaradayInduction(BaseClass):
    """
    Faraday's Law of electromagnetic induction.

    EMF = -dΦ_B/dt where Φ_B = ∫ B · dA

    Calculates induced EMF from time-varying magnetic fields.

    Args:
        B_field: Callable returning B(r, t) as 3D vector

    Examples:
        >>> # Linearly increasing B field
        >>> B = lambda r, t: np.array([0, 0, 0.1 * t])
        >>> faraday = FaradayInduction(B)
        >>> emf = faraday.induced_emf_circular(radius=0.1, t=1.0, dt=0.01)
    """

    def __init__(self, B_field: Callable[[np.ndarray, float], np.ndarray]):
        super().__init__()
        self.B = B_field

    def magnetic_flux(
        self,
        t: float,
        loop_center: ArrayLike = (0, 0, 0),
        loop_normal: ArrayLike = (0, 0, 1),
        loop_radius: float = 1.0,
        n_points: int = 30
    ) -> float:
        """Calculate magnetic flux through a circular loop."""
        center = np.array(loop_center)
        normal = np.array(loop_normal)
        normal = normal / np.linalg.norm(normal)

        # Create orthonormal basis
        if abs(normal[2]) < 0.9:
            u = np.cross(normal, np.array([0, 0, 1]))
        else:
            u = np.cross(normal, np.array([1, 0, 0]))
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)

        # Integrate B · dA over circular disk
        flux = 0.0
        r_vals = np.linspace(0, loop_radius, n_points)
        phi_vals = np.linspace(0, 2*np.pi, 2*n_points)

        dr = r_vals[1] - r_vals[0] if len(r_vals) > 1 else loop_radius
        dphi = phi_vals[1] - phi_vals[0] if len(phi_vals) > 1 else 2*np.pi

        for r in r_vals:
            for phi in phi_vals:
                point = center + r * (np.cos(phi) * u + np.sin(phi) * v)
                B_val = self.B(point, t)
                dA = r * dr * dphi
                flux += np.dot(B_val, normal) * dA

        return flux

    def induced_emf(
        self,
        t: float,
        dt: float,
        loop_center: ArrayLike = (0, 0, 0),
        loop_normal: ArrayLike = (0, 0, 1),
        loop_radius: float = 1.0,
        n_points: int = 30
    ) -> float:
        """Calculate induced EMF using finite difference for dΦ/dt."""
        flux_t = self.magnetic_flux(t, loop_center, loop_normal, loop_radius, n_points)
        flux_t_dt = self.magnetic_flux(t + dt, loop_center, loop_normal, loop_radius, n_points)

        return -(flux_t_dt - flux_t) / dt

    def induced_emf_circular(
        self,
        radius: float,
        t: float,
        dt: float,
        center: ArrayLike = (0, 0, 0),
        normal: ArrayLike = (0, 0, 1)
    ) -> float:
        """Convenience method for circular loop in xy-plane."""
        return self.induced_emf(t, dt, center, normal, radius)


class AmpereMaxwell(BaseClass):
    """
    Ampère-Maxwell Law with displacement current.

    ∮ B · dl = μ₀(I_enclosed + ε₀ dΦ_E/dt)

    Computes magnetic field circulation including displacement current.

    Args:
        J_field: Current density J(r, t) as callable
        E_field: Electric field E(r, t) as callable (for displacement current)
    """

    def __init__(
        self,
        J_field: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
        E_field: Optional[Callable[[np.ndarray, float], np.ndarray]] = None
    ):
        super().__init__()
        self.J = J_field or (lambda r, t: np.zeros(3))
        self.E = E_field or (lambda r, t: np.zeros(3))

    def conduction_current(
        self,
        t: float,
        loop_center: ArrayLike,
        loop_normal: ArrayLike,
        loop_radius: float,
        n_points: int = 30
    ) -> float:
        """Calculate conduction current through a loop."""
        center = np.array(loop_center)
        normal = np.array(loop_normal)
        normal = normal / np.linalg.norm(normal)

        # Create basis vectors
        if abs(normal[2]) < 0.9:
            u = np.cross(normal, np.array([0, 0, 1]))
        else:
            u = np.cross(normal, np.array([1, 0, 0]))
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)

        # Integrate J · dA
        current = 0.0
        r_vals = np.linspace(0, loop_radius, n_points)
        phi_vals = np.linspace(0, 2*np.pi, 2*n_points)

        dr = r_vals[1] - r_vals[0] if len(r_vals) > 1 else loop_radius
        dphi = phi_vals[1] - phi_vals[0] if len(phi_vals) > 1 else 2*np.pi

        for r in r_vals:
            for phi in phi_vals:
                point = center + r * (np.cos(phi) * u + np.sin(phi) * v)
                J_val = self.J(point, t)
                dA = r * dr * dphi
                current += np.dot(J_val, normal) * dA

        return current

    def displacement_current(
        self,
        t: float,
        dt: float,
        loop_center: ArrayLike,
        loop_normal: ArrayLike,
        loop_radius: float,
        n_points: int = 30
    ) -> float:
        """Calculate displacement current ε₀ dΦ_E/dt."""
        center = np.array(loop_center)
        normal = np.array(loop_normal)
        normal = normal / np.linalg.norm(normal)

        # Create basis
        if abs(normal[2]) < 0.9:
            u = np.cross(normal, np.array([0, 0, 1]))
        else:
            u = np.cross(normal, np.array([1, 0, 0]))
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)

        def electric_flux(time):
            flux = 0.0
            r_vals = np.linspace(0, loop_radius, n_points)
            phi_vals = np.linspace(0, 2*np.pi, 2*n_points)

            dr = r_vals[1] - r_vals[0] if len(r_vals) > 1 else loop_radius
            dphi = phi_vals[1] - phi_vals[0] if len(phi_vals) > 1 else 2*np.pi

            for r in r_vals:
                for phi in phi_vals:
                    point = center + r * (np.cos(phi) * u + np.sin(phi) * v)
                    E_val = self.E(point, time)
                    dA = r * dr * dphi
                    flux += np.dot(E_val, normal) * dA
            return flux

        dPhi_dt = (electric_flux(t + dt) - electric_flux(t)) / dt
        return EPSILON_0 * dPhi_dt

    def total_enclosed_current(
        self,
        t: float,
        dt: float,
        loop_center: ArrayLike,
        loop_normal: ArrayLike,
        loop_radius: float,
        n_points: int = 30
    ) -> float:
        """Calculate total current (conduction + displacement)."""
        I_cond = self.conduction_current(t, loop_center, loop_normal, loop_radius, n_points)
        I_disp = self.displacement_current(t, dt, loop_center, loop_normal, loop_radius, n_points)
        return I_cond + I_disp


# ==============================================================================
# Phase 2.2: Electromagnetic Potentials
# ==============================================================================

class ScalarPotential(BaseClass):
    """
    Electrostatic scalar potential φ.

    Solves for φ from charge distribution using:
    φ(r) = (1/4πε₀) ∫ ρ(r')/|r-r'| dV'

    E = -∇φ

    Args:
        charge_distribution: Callable returning ρ(r) or list of point charges

    Examples:
        >>> # Point charge
        >>> phi = ScalarPotential(point_charges=[(1e-9, [0, 0, 0])])
        >>> V = phi.potential([1, 0, 0])
    """

    def __init__(
        self,
        charge_distribution: Optional[Callable[[np.ndarray], float]] = None,
        point_charges: Optional[List[Tuple[float, ArrayLike]]] = None
    ):
        super().__init__()
        self.rho = charge_distribution
        self.point_charges = [(q, np.array(r)) for q, r in (point_charges or [])]
        self.k = 1 / (4 * np.pi * EPSILON_0)

    def potential(self, position: ArrayLike) -> float:
        """Calculate scalar potential at position."""
        r = np.array(position)
        phi = 0.0

        # Point charges
        for q, r_q in self.point_charges:
            d = np.linalg.norm(r - r_q)
            if d > 1e-15:
                phi += self.k * q / d

        return phi

    def electric_field(self, position: ArrayLike, dr: float = 1e-8) -> np.ndarray:
        """Calculate E = -∇φ using numerical gradient."""
        r = np.array(position)
        E = np.zeros(3)

        for i in range(3):
            r_plus = r.copy()
            r_plus[i] += dr
            r_minus = r.copy()
            r_minus[i] -= dr

            E[i] = -(self.potential(r_plus) - self.potential(r_minus)) / (2 * dr)

        return E

    def equipotential_surface(
        self,
        V_target: float,
        bounds: Tuple[float, float, float, float],
        n_points: int = 50
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Find equipotential surface in 2D (z=0 plane).

        Returns:
            X, Y, V meshgrid arrays for contour plotting
        """
        x_min, x_max, y_min, y_max = bounds
        x = np.linspace(x_min, x_max, n_points)
        y = np.linspace(y_min, y_max, n_points)
        X, Y = np.meshgrid(x, y)
        V = np.zeros_like(X)

        for i in range(n_points):
            for j in range(n_points):
                V[i, j] = self.potential([X[i, j], Y[i, j], 0])

        return X, Y, V


class VectorPotential(BaseClass):
    """
    Magnetic vector potential A.

    B = ∇ × A
    A(r) = (μ₀/4π) ∫ J(r')/|r-r'| dV'

    For steady currents in magnetostatics.

    Args:
        current_distribution: Callable returning J(r) as 3D vector
        current_loops: List of (current, center, normal, radius) tuples
    """

    def __init__(
        self,
        current_distribution: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        current_loops: Optional[List[Tuple[float, ArrayLike, ArrayLike, float]]] = None
    ):
        super().__init__()
        self.J = current_distribution
        self.current_loops = current_loops or []
        self.k = MU_0 / (4 * np.pi)

    def potential_from_loop(
        self,
        position: ArrayLike,
        current: float,
        center: ArrayLike,
        normal: ArrayLike,
        radius: float,
        n_segments: int = 100
    ) -> np.ndarray:
        """Calculate vector potential from a circular current loop."""
        r = np.array(position)
        center = np.array(center)
        normal = np.array(normal)
        normal = normal / np.linalg.norm(normal)

        # Create loop basis
        if abs(normal[2]) < 0.9:
            u = np.cross(normal, np.array([0, 0, 1]))
        else:
            u = np.cross(normal, np.array([1, 0, 0]))
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)

        A = np.zeros(3)
        dphi = 2 * np.pi / n_segments

        for i in range(n_segments):
            phi = i * dphi
            phi_next = (i + 1) * dphi

            # Current element position and direction
            r_loop = center + radius * (np.cos(phi) * u + np.sin(phi) * v)
            dl = radius * dphi * (-np.sin(phi) * u + np.cos(phi) * v)

            d = np.linalg.norm(r - r_loop)
            if d > 1e-15:
                A += self.k * current * dl / d

        return A

    def potential(self, position: ArrayLike) -> np.ndarray:
        """Calculate total vector potential at position."""
        r = np.array(position)
        A = np.zeros(3)

        for loop in self.current_loops:
            current, center, normal, radius = loop
            A += self.potential_from_loop(r, current, center, normal, radius)

        return A

    def magnetic_field(self, position: ArrayLike, dr: float = 1e-8) -> np.ndarray:
        """Calculate B = ∇ × A using numerical curl."""
        r = np.array(position)

        # Numerical partial derivatives
        def dA_di(i, j):
            r_plus = r.copy()
            r_plus[i] += dr
            r_minus = r.copy()
            r_minus[i] -= dr
            return (self.potential(r_plus)[j] - self.potential(r_minus)[j]) / (2 * dr)

        # Curl components
        Bx = dA_di(1, 2) - dA_di(2, 1)  # dAz/dy - dAy/dz
        By = dA_di(2, 0) - dA_di(0, 2)  # dAx/dz - dAz/dx
        Bz = dA_di(0, 1) - dA_di(1, 0)  # dAy/dx - dAx/dy

        return np.array([Bx, By, Bz])


class GaugeFreedom(BaseClass):
    """
    Electromagnetic gauge transformations.

    A' = A + ∇χ
    φ' = φ - ∂χ/∂t

    Implements Coulomb gauge (∇·A = 0) and Lorenz gauge (∇·A + (1/c²)∂φ/∂t = 0).

    Args:
        scalar_potential: ScalarPotential instance or callable φ(r, t)
        vector_potential: VectorPotential instance or callable A(r, t)
    """

    def __init__(
        self,
        scalar_potential: Union[ScalarPotential, Callable],
        vector_potential: Union[VectorPotential, Callable]
    ):
        super().__init__()

        if isinstance(scalar_potential, ScalarPotential):
            self.phi = lambda r, t: scalar_potential.potential(r)
        else:
            self.phi = scalar_potential

        if isinstance(vector_potential, VectorPotential):
            self.A = lambda r, t: vector_potential.potential(r)
        else:
            self.A = vector_potential

    def divergence_A(self, position: ArrayLike, t: float, dr: float = 1e-8) -> float:
        """Calculate ∇·A at position."""
        r = np.array(position)
        div = 0.0

        for i in range(3):
            r_plus = r.copy()
            r_plus[i] += dr
            r_minus = r.copy()
            r_minus[i] -= dr

            div += (self.A(r_plus, t)[i] - self.A(r_minus, t)[i]) / (2 * dr)

        return div

    def lorenz_gauge_residual(
        self,
        position: ArrayLike,
        t: float,
        dt: float = 1e-10,
        dr: float = 1e-8
    ) -> float:
        """
        Calculate Lorenz gauge residual: ∇·A + (1/c²)∂φ/∂t

        Should be zero if gauge is satisfied.
        """
        r = np.array(position)

        div_A = self.divergence_A(r, t, dr)
        dphi_dt = (self.phi(r, t + dt) - self.phi(r, t)) / dt

        return div_A + dphi_dt / C**2

    def coulomb_gauge_residual(self, position: ArrayLike, t: float, dr: float = 1e-8) -> float:
        """
        Calculate Coulomb gauge residual: ∇·A

        Should be zero if gauge is satisfied.
        """
        return self.divergence_A(position, t, dr)

    def transform(
        self,
        chi: Callable[[np.ndarray, float], float],
        position: ArrayLike,
        t: float,
        dt: float = 1e-10,
        dr: float = 1e-8
    ) -> Tuple[float, np.ndarray]:
        """
        Apply gauge transformation with gauge function χ.

        Returns:
            Tuple of (φ', A') at the given position and time
        """
        r = np.array(position)

        # φ' = φ - ∂χ/∂t
        dchi_dt = (chi(r, t + dt) - chi(r, t)) / dt
        phi_new = self.phi(r, t) - dchi_dt

        # A' = A + ∇χ
        grad_chi = np.zeros(3)
        for i in range(3):
            r_plus = r.copy()
            r_plus[i] += dr
            r_minus = r.copy()
            r_minus[i] -= dr
            grad_chi[i] = (chi(r_plus, t) - chi(r_minus, t)) / (2 * dr)

        A_new = self.A(r, t) + grad_chi

        return phi_new, A_new


class RetardedPotential(BaseClass):
    """
    Retarded potentials (Jefimenko's equations).

    φ(r, t) = (1/4πε₀) ∫ [ρ(r', t_r)] / |r-r'| dV'
    A(r, t) = (μ₀/4π) ∫ [J(r', t_r)] / |r-r'| dV'

    where t_r = t - |r-r'|/c is the retarded time.

    Accounts for finite speed of light in electromagnetic interactions.

    Args:
        rho: Charge density ρ(r, t)
        J: Current density J(r, t)
        source_region: Bounding box [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
    """

    def __init__(
        self,
        rho: Callable[[np.ndarray, float], float],
        J: Callable[[np.ndarray, float], np.ndarray],
        source_region: List[Tuple[float, float]]
    ):
        super().__init__()
        self.rho = rho
        self.J = J
        self.source_region = source_region

    def retarded_time(self, r_field: np.ndarray, r_source: np.ndarray, t: float) -> float:
        """Calculate retarded time t_r = t - |r - r'|/c."""
        return t - np.linalg.norm(r_field - r_source) / C

    def scalar_potential(
        self,
        position: ArrayLike,
        t: float,
        n_points: int = 20
    ) -> float:
        """Calculate retarded scalar potential."""
        r = np.array(position)

        x_range, y_range, z_range = self.source_region
        x_vals = np.linspace(x_range[0], x_range[1], n_points)
        y_vals = np.linspace(y_range[0], y_range[1], n_points)
        z_vals = np.linspace(z_range[0], z_range[1], n_points)

        dx = (x_range[1] - x_range[0]) / max(n_points - 1, 1)
        dy = (y_range[1] - y_range[0]) / max(n_points - 1, 1)
        dz = (z_range[1] - z_range[0]) / max(n_points - 1, 1)
        dV = dx * dy * dz

        phi = 0.0
        k = 1 / (4 * np.pi * EPSILON_0)

        for x in x_vals:
            for y in y_vals:
                for z in z_vals:
                    r_source = np.array([x, y, z])
                    d = np.linalg.norm(r - r_source)

                    if d > 1e-15:
                        t_r = self.retarded_time(r, r_source, t)
                        if t_r >= 0:
                            rho_retarded = self.rho(r_source, t_r)
                            phi += k * rho_retarded * dV / d

        return phi

    def vector_potential(
        self,
        position: ArrayLike,
        t: float,
        n_points: int = 20
    ) -> np.ndarray:
        """Calculate retarded vector potential."""
        r = np.array(position)

        x_range, y_range, z_range = self.source_region
        x_vals = np.linspace(x_range[0], x_range[1], n_points)
        y_vals = np.linspace(y_range[0], y_range[1], n_points)
        z_vals = np.linspace(z_range[0], z_range[1], n_points)

        dx = (x_range[1] - x_range[0]) / max(n_points - 1, 1)
        dy = (y_range[1] - y_range[0]) / max(n_points - 1, 1)
        dz = (z_range[1] - z_range[0]) / max(n_points - 1, 1)
        dV = dx * dy * dz

        A = np.zeros(3)
        k = MU_0 / (4 * np.pi)

        for x in x_vals:
            for y in y_vals:
                for z in z_vals:
                    r_source = np.array([x, y, z])
                    d = np.linalg.norm(r - r_source)

                    if d > 1e-15:
                        t_r = self.retarded_time(r, r_source, t)
                        if t_r >= 0:
                            J_retarded = self.J(r_source, t_r)
                            A += k * J_retarded * dV / d

        return A


# ==============================================================================
# Phase 2.3: Radiation
# ==============================================================================

class DipoleRadiation(BaseClass):
    """
    Radiation from an oscillating electric dipole.

    p(t) = p₀ cos(ωt) ẑ

    Far-field radiation pattern and power calculation.

    Args:
        dipole_moment: Peak dipole moment (C·m)
        frequency: Oscillation frequency (Hz)
        orientation: Unit vector for dipole axis

    Examples:
        >>> dipole = DipoleRadiation(dipole_moment=1e-30, frequency=1e9)
        >>> power = dipole.total_radiated_power()
    """

    def __init__(
        self,
        dipole_moment: float,
        frequency: float,
        orientation: ArrayLike = (0, 0, 1)
    ):
        super().__init__()
        validate_positive(frequency, "frequency")

        self.p0 = dipole_moment
        self.omega = 2 * np.pi * frequency
        self.frequency = frequency
        self.orientation = np.array(orientation)
        self.orientation = self.orientation / np.linalg.norm(self.orientation)

        self.k = self.omega / C  # Wave number
        self.wavelength = C / frequency

    def electric_field(
        self,
        position: ArrayLike,
        t: float,
        far_field: bool = True
    ) -> np.ndarray:
        """
        Calculate electric field at position and time.

        Args:
            position: Observer position (meters)
            t: Time (seconds)
            far_field: If True, use far-field approximation
        """
        r = np.array(position)
        r_mag = np.linalg.norm(r)
        r_hat = r / r_mag

        # Angle between dipole and observation direction
        cos_theta = np.dot(self.orientation, r_hat)
        sin_theta = np.sqrt(1 - cos_theta**2)

        # Retarded time
        t_r = t - r_mag / C

        if far_field:
            # Far-field approximation (kr >> 1)
            # E_theta = (p0 * omega^2 / 4πε0c^2) * (sin θ / r) * cos(ω(t - r/c))
            amplitude = (self.p0 * self.omega**2 * sin_theta) / (4 * np.pi * EPSILON_0 * C**2 * r_mag)
            phase = np.cos(self.omega * t_r)

            # Theta direction (perpendicular to r in the plane of p and r)
            theta_hat = np.cross(np.cross(r_hat, self.orientation), r_hat)
            if np.linalg.norm(theta_hat) > 1e-15:
                theta_hat = theta_hat / np.linalg.norm(theta_hat)

            return amplitude * phase * theta_hat
        else:
            # Near field would include static and induction terms
            # Simplified: just return far-field
            return self.electric_field(position, t, far_field=True)

    def magnetic_field(self, position: ArrayLike, t: float) -> np.ndarray:
        """Calculate magnetic field (B = E/c × r_hat in far field)."""
        r = np.array(position)
        r_hat = r / np.linalg.norm(r)
        E = self.electric_field(position, t)
        return np.cross(r_hat, E) / C

    def poynting_vector(self, position: ArrayLike, t: float) -> np.ndarray:
        """Calculate instantaneous Poynting vector S = E × H."""
        E = self.electric_field(position, t)
        H = self.magnetic_field(position, t) / MU_0
        return np.cross(E, H)

    def time_averaged_intensity(self, position: ArrayLike) -> float:
        """Calculate time-averaged intensity <S> at position."""
        r = np.array(position)
        r_mag = np.linalg.norm(r)
        r_hat = r / r_mag

        cos_theta = np.dot(self.orientation, r_hat)
        sin2_theta = 1 - cos_theta**2

        # <S> = (p0^2 * omega^4 * sin^2(θ)) / (32 π^2 ε0 c^3 r^2)
        return (self.p0**2 * self.omega**4 * sin2_theta) / (32 * np.pi**2 * EPSILON_0 * C**3 * r_mag**2)

    def total_radiated_power(self) -> float:
        """
        Calculate total radiated power.

        P = (p0^2 * omega^4) / (12 π ε0 c^3)
        """
        return (self.p0**2 * self.omega**4) / (12 * np.pi * EPSILON_0 * C**3)

    def radiation_pattern(self, n_theta: int = 50, n_phi: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate normalized radiation pattern.

        Returns:
            theta, phi, pattern arrays for 3D plotting
        """
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2 * np.pi, n_phi)

        THETA, PHI = np.meshgrid(theta, phi)

        # Radiation pattern proportional to sin^2(theta)
        pattern = np.sin(THETA)**2

        return THETA, PHI, pattern


class LarmorFormula(BaseClass):
    """
    Larmor formula for radiation from accelerating charge.

    P = (q^2 * a^2) / (6 π ε0 c^3)  (non-relativistic)

    Calculates instantaneous and time-averaged radiated power.

    Args:
        charge: Particle charge (C)
        mass: Particle mass (kg), used for velocity calculations

    Examples:
        >>> larmor = LarmorFormula(charge=E_CHARGE)
        >>> power = larmor.radiated_power(acceleration=1e15)
    """

    def __init__(self, charge: float, mass: float = 9.109e-31):
        super().__init__()
        self.q = charge
        self.m = mass
        self.prefactor = self.q**2 / (6 * np.pi * EPSILON_0 * C**3)

    def radiated_power(self, acceleration: Union[float, ArrayLike]) -> float:
        """
        Calculate instantaneous radiated power.

        P = (q² a²) / (6πε₀c³)

        Args:
            acceleration: Magnitude or vector of acceleration (m/s²)
        """
        if np.isscalar(acceleration):
            a2 = acceleration**2
        else:
            a2 = np.dot(acceleration, acceleration)

        return self.prefactor * a2

    def power_from_force(self, force: Union[float, ArrayLike]) -> float:
        """Calculate radiated power from applied force."""
        if np.isscalar(force):
            a = force / self.m
        else:
            a = np.linalg.norm(force) / self.m
        return self.radiated_power(a)

    def radiation_reaction_force(self, jerk: ArrayLike) -> np.ndarray:
        """
        Calculate radiation reaction force (Abraham-Lorentz force).

        F_rad = (q² / 6πε₀c³) * da/dt

        Args:
            jerk: Time derivative of acceleration (m/s³)
        """
        jerk = np.array(jerk)
        return self.prefactor * jerk

    def characteristic_time(self) -> float:
        """
        Return characteristic radiation time τ = q²/(6πε₀mc³).

        Time scale for radiation damping.
        """
        return self.q**2 / (6 * np.pi * EPSILON_0 * self.m * C**3)


class SynchrotronRadiation(BaseClass):
    """
    Synchrotron radiation from relativistic circular motion.

    Calculates spectrum, critical frequency, and total power for
    relativistic particles in magnetic fields.

    Args:
        particle_energy: Particle energy (eV)
        magnetic_field: Magnetic field strength (T)
        charge: Particle charge (C)
        mass: Particle rest mass (kg)

    Examples:
        >>> synch = SynchrotronRadiation(particle_energy=1e9, magnetic_field=1.0)
        >>> power = synch.total_power()
        >>> freq_c = synch.critical_frequency()
    """

    def __init__(
        self,
        particle_energy: float,
        magnetic_field: float,
        charge: float = E_CHARGE,
        mass: float = 9.109e-31  # Electron mass
    ):
        super().__init__()

        self.E = particle_energy * E_CHARGE  # Convert eV to J
        self.B = magnetic_field
        self.q = charge
        self.m = mass

        # Lorentz factor
        self.gamma = self.E / (mass * C**2)

        # Velocity (relativistic)
        self.beta = np.sqrt(1 - 1/self.gamma**2)

        # Cyclotron frequency
        self.omega_c = abs(charge) * magnetic_field / (self.gamma * mass)

        # Bending radius
        self.rho = self.gamma * mass * C * self.beta / (abs(charge) * magnetic_field)

    def critical_frequency(self) -> float:
        """
        Calculate critical frequency ω_c.

        Half the power is radiated above and below this frequency.
        ω_c = (3/2) γ³ c / ρ
        """
        return 1.5 * self.gamma**3 * C / self.rho

    def critical_energy(self) -> float:
        """Calculate critical photon energy in eV."""
        return 1.055e-34 * self.critical_frequency() / E_CHARGE

    def total_power(self) -> float:
        """
        Calculate total radiated power (relativistic Larmor).

        P = (q² c / 6πε₀) * γ⁴ / ρ²
        """
        return (self.q**2 * C * self.gamma**4) / (6 * np.pi * EPSILON_0 * self.rho**2)

    def energy_loss_per_revolution(self) -> float:
        """Calculate energy lost per revolution in eV."""
        period = 2 * np.pi * self.rho / (C * self.beta)
        return self.total_power() * period / E_CHARGE

    def spectral_power(self, omega: float) -> float:
        """
        Calculate power spectral density P(ω).

        Uses simplified universal function approximation.
        """
        x = omega / self.critical_frequency()

        if x < 0.01:
            # Low frequency: P ∝ ω^(1/3)
            return self.total_power() * 0.5 * x**(1/3)
        elif x > 5:
            # High frequency: exponential cutoff
            return self.total_power() * 0.5 * np.sqrt(x) * np.exp(-x)
        else:
            # Intermediate: approximate
            return self.total_power() * 0.3 * x**(1/3) * np.exp(-x)

    def opening_angle(self) -> float:
        """
        Calculate characteristic opening angle.

        θ ≈ 1/γ
        """
        return 1.0 / self.gamma


class CherenkovRadiation(BaseClass):
    """
    Cherenkov radiation from superluminal particles in media.

    Occurs when particle velocity v > c/n (phase velocity).

    Args:
        particle_velocity: Particle velocity (m/s) or beta = v/c
        refractive_index: Medium refractive index n
        charge: Particle charge (C)

    Examples:
        >>> cherenkov = CherenkovRadiation(particle_velocity=0.99*C, refractive_index=1.33)
        >>> angle = cherenkov.cone_angle()
        >>> power = cherenkov.radiated_power_per_length(wavelength_range=(400e-9, 700e-9))
    """

    def __init__(
        self,
        particle_velocity: float,
        refractive_index: float,
        charge: float = E_CHARGE
    ):
        super().__init__()

        self.n = refractive_index
        self.q = charge

        # Convert to beta if needed
        if particle_velocity > 1:
            self.v = particle_velocity
            self.beta = particle_velocity / C
        else:
            self.beta = particle_velocity
            self.v = particle_velocity * C

        # Threshold velocity
        self.beta_threshold = 1.0 / refractive_index

        # Check if Cherenkov condition is met
        self.is_radiating = self.beta > self.beta_threshold

    def cone_angle(self) -> float:
        """
        Calculate Cherenkov cone half-angle.

        cos(θ) = 1 / (n β)

        Returns:
            Angle in radians, or 0 if below threshold
        """
        if not self.is_radiating:
            return 0.0

        cos_theta = 1.0 / (self.n * self.beta)
        return np.arccos(cos_theta)

    def cone_angle_degrees(self) -> float:
        """Return cone angle in degrees."""
        return np.degrees(self.cone_angle())

    def threshold_energy(self, mass: float) -> float:
        """
        Calculate threshold kinetic energy for Cherenkov radiation.

        Args:
            mass: Particle rest mass (kg)

        Returns:
            Threshold energy in eV
        """
        gamma_threshold = 1.0 / np.sqrt(1 - self.beta_threshold**2)
        KE = (gamma_threshold - 1) * mass * C**2
        return KE / E_CHARGE

    def photons_per_length(self, lambda_min: float, lambda_max: float) -> float:
        """
        Calculate number of photons emitted per unit length.

        dN/dx = 2π α sin²(θ) ∫ dλ/λ²

        Args:
            lambda_min, lambda_max: Wavelength range (meters)

        Returns:
            Photons per meter
        """
        if not self.is_radiating:
            return 0.0

        alpha = 1 / 137.036  # Fine structure constant
        theta = self.cone_angle()

        # Frank-Tamm formula
        return 2 * np.pi * alpha * np.sin(theta)**2 * (1/lambda_min - 1/lambda_max)

    def radiated_power_per_length(self, wavelength_range: Tuple[float, float]) -> float:
        """
        Calculate power radiated per unit length.

        Args:
            wavelength_range: (lambda_min, lambda_max) in meters
        """
        if not self.is_radiating:
            return 0.0

        lambda_min, lambda_max = wavelength_range

        # Power per length: (q²/4πε₀) * (ω/c²) * (1 - 1/(n²β²))
        # Integrated over frequency range
        omega_max = 2 * np.pi * C / lambda_min
        omega_min = 2 * np.pi * C / lambda_max

        factor = self.q**2 / (4 * np.pi * EPSILON_0 * C**2)
        sin2_theta = 1 - 1 / (self.n * self.beta)**2

        return factor * sin2_theta * 0.5 * (omega_max**2 - omega_min**2)


class Bremsstrahlung(BaseClass):
    """
    Bremsstrahlung (braking radiation) from decelerated charges.

    Radiation emitted when charged particles are decelerated,
    typically in Coulomb fields of atomic nuclei.

    Args:
        particle_energy: Incident particle energy (eV)
        target_Z: Atomic number of target nucleus
        charge: Particle charge (C)
        mass: Particle mass (kg)

    Examples:
        >>> brems = Bremsstrahlung(particle_energy=1e6, target_Z=79)  # Gold target
        >>> spectrum = brems.spectrum(energies=np.linspace(0, 1e6, 100))
    """

    def __init__(
        self,
        particle_energy: float,
        target_Z: int,
        charge: float = E_CHARGE,
        mass: float = 9.109e-31
    ):
        super().__init__()

        self.E0 = particle_energy * E_CHARGE  # Convert to Joules
        self.Z = target_Z
        self.q = charge
        self.m = mass

        # Lorentz factor
        self.gamma = self.E0 / (mass * C**2) + 1

        # Classical electron radius
        self.r_e = E_CHARGE**2 / (4 * np.pi * EPSILON_0 * mass * C**2)

    def kramers_spectrum(self, photon_energy: float) -> float:
        """
        Calculate Kramers' bremsstrahlung spectrum (classical).

        dN/dE ∝ Z² / E

        Args:
            photon_energy: Photon energy in eV

        Returns:
            Relative spectral intensity
        """
        E_photon = photon_energy * E_CHARGE
        E0_J = self.E0

        if E_photon > E0_J or E_photon <= 0:
            return 0.0

        # Kramers law: I(E) ∝ Z² (E0 - E) / E
        return self.Z**2 * (E0_J - E_photon) / E_photon

    def spectrum(self, energies: ArrayLike) -> np.ndarray:
        """
        Calculate spectrum over array of photon energies.

        Args:
            energies: Array of photon energies in eV
        """
        return np.array([self.kramers_spectrum(E) for E in energies])

    def maximum_photon_energy(self) -> float:
        """Return maximum photon energy (equals incident particle energy) in eV."""
        return self.E0 / E_CHARGE

    def radiation_length(self, density: float, A: float) -> float:
        """
        Calculate radiation length in the material.

        X0 = 716.4 A / (Z(Z+1) ln(287/√Z)) g/cm²

        Args:
            density: Material density (kg/m³)
            A: Atomic mass number

        Returns:
            Radiation length in meters
        """
        X0_mass = 716.4 * A / (self.Z * (self.Z + 1) * np.log(287 / np.sqrt(self.Z)))  # g/cm²
        X0_mass *= 10  # Convert to kg/m²
        return X0_mass / density


class AntennaPattern(BaseClass):
    """
    Radiation pattern calculations for antennas.

    Supports common antenna types: dipole, monopole, loop, array.

    Args:
        antenna_type: Type of antenna ('dipole', 'monopole', 'loop', 'array')
        length: Characteristic length (m) - dipole/monopole length or loop radius
        frequency: Operating frequency (Hz)

    Examples:
        >>> antenna = AntennaPattern('dipole', length=0.5, frequency=300e6)
        >>> gain = antenna.directivity()
        >>> theta, phi, pattern = antenna.pattern_3d()
    """

    def __init__(
        self,
        antenna_type: str = 'dipole',
        length: float = 0.5,
        frequency: float = 300e6,
        n_elements: int = 1,
        element_spacing: Optional[float] = None
    ):
        super().__init__()

        self.antenna_type = antenna_type
        self.length = length
        self.frequency = frequency
        self.wavelength = C / frequency
        self.k = 2 * np.pi / self.wavelength

        # Array parameters
        self.n_elements = n_elements
        self.element_spacing = element_spacing or self.wavelength / 2

    def pattern(self, theta: float) -> float:
        """
        Calculate normalized field pattern E(θ) for single element.

        Args:
            theta: Angle from antenna axis (radians)
        """
        if self.antenna_type == 'dipole':
            # Half-wave dipole pattern
            kL = self.k * self.length / 2
            if abs(np.sin(theta)) < 1e-10:
                return 0.0
            return abs(np.cos(kL * np.cos(theta)) - np.cos(kL)) / np.sin(theta)

        elif self.antenna_type == 'monopole':
            # Quarter-wave monopole (half of dipole pattern)
            kL = self.k * self.length
            if abs(np.sin(theta)) < 1e-10:
                return 0.0
            return abs(np.cos(kL * np.cos(theta)) - np.cos(kL)) / np.sin(theta)

        elif self.antenna_type == 'loop':
            # Small loop antenna
            return abs(np.sin(theta))

        else:
            # Isotropic
            return 1.0

    def array_factor(self, theta: float, phi: float = 0.0, scan_angle: float = 0.0) -> float:
        """
        Calculate array factor for linear array.

        Args:
            theta: Elevation angle from array axis
            phi: Azimuth angle
            scan_angle: Beam steering angle
        """
        if self.n_elements == 1:
            return 1.0

        psi = self.k * self.element_spacing * (np.cos(theta) - np.cos(scan_angle))

        # Array factor: sin(N*psi/2) / (N * sin(psi/2))
        if abs(np.sin(psi / 2)) < 1e-10:
            return 1.0

        return abs(np.sin(self.n_elements * psi / 2) / (self.n_elements * np.sin(psi / 2)))

    def total_pattern(self, theta: float, phi: float = 0.0) -> float:
        """Calculate total pattern (element × array factor)."""
        return self.pattern(theta) * self.array_factor(theta, phi)

    def directivity(self) -> float:
        """
        Calculate antenna directivity in dBi.

        D = 4π U_max / P_rad
        """
        # Numerical integration for radiated power
        n_theta = 100
        n_phi = 50

        theta_vals = np.linspace(0.01, np.pi - 0.01, n_theta)
        phi_vals = np.linspace(0, 2 * np.pi, n_phi)

        P_rad = 0.0
        U_max = 0.0

        dtheta = theta_vals[1] - theta_vals[0]
        dphi = phi_vals[1] - phi_vals[0]

        for theta in theta_vals:
            for phi in phi_vals:
                U = self.total_pattern(theta, phi)**2
                U_max = max(U_max, U)
                P_rad += U * np.sin(theta) * dtheta * dphi

        if P_rad < 1e-15:
            return 0.0

        D = 4 * np.pi * U_max / P_rad
        return 10 * np.log10(D)

    def pattern_3d(
        self,
        n_theta: int = 50,
        n_phi: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate 3D radiation pattern.

        Returns:
            theta, phi, pattern arrays for plotting
        """
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2 * np.pi, n_phi)

        THETA, PHI = np.meshgrid(theta, phi)
        pattern = np.zeros_like(THETA)

        for i in range(n_phi):
            for j in range(n_theta):
                pattern[i, j] = self.total_pattern(theta[j], phi[i])

        # Normalize
        if pattern.max() > 0:
            pattern /= pattern.max()

        return THETA, PHI, pattern


# ==============================================================================
# Phase 2.4: Multipole Expansion
# ==============================================================================

class MultipoleExpansion(BaseClass):
    """
    General multipole expansion for charge distributions.

    V(r) = (1/4πε₀) Σ (1/r^(l+1)) Σ q_lm Y_lm(θ,φ)

    Computes monopole, dipole, quadrupole, and higher moments.

    Args:
        charges: List of (charge, position) tuples
        max_l: Maximum multipole order to compute

    Examples:
        >>> charges = [(1e-9, [0.1, 0, 0]), (-1e-9, [-0.1, 0, 0])]
        >>> mp = MultipoleExpansion(charges, max_l=2)
        >>> monopole = mp.monopole_moment()
        >>> dipole = mp.dipole_moment()
    """

    def __init__(
        self,
        charges: List[Tuple[float, ArrayLike]],
        max_l: int = 4
    ):
        super().__init__()
        self.charges = [(q, np.array(r)) for q, r in charges]
        self.max_l = max_l
        self._compute_moments()

    def _compute_moments(self):
        """Compute multipole moments up to max_l."""
        self.moments = {}

        # Monopole (l=0)
        self.moments['monopole'] = sum(q for q, r in self.charges)

        # Dipole (l=1)
        dipole = np.zeros(3)
        for q, r in self.charges:
            dipole += q * r
        self.moments['dipole'] = dipole

        # Quadrupole (l=2) - traceless tensor
        Q = np.zeros((3, 3))
        for q, r in self.charges:
            for i in range(3):
                for j in range(3):
                    Q[i, j] += q * (3 * r[i] * r[j] - (np.dot(r, r) if i == j else 0))
        self.moments['quadrupole'] = Q

        # Higher moments stored as spherical harmonic coefficients
        # (simplified implementation)

    def monopole_moment(self) -> float:
        """Return total charge (monopole moment)."""
        return self.moments['monopole']

    def dipole_moment(self) -> np.ndarray:
        """Return electric dipole moment vector."""
        return self.moments['dipole']

    def quadrupole_tensor(self) -> np.ndarray:
        """Return traceless quadrupole tensor."""
        return self.moments['quadrupole']

    def potential(self, position: ArrayLike, max_l: Optional[int] = None) -> float:
        """
        Calculate potential using multipole expansion.

        Args:
            position: Field point
            max_l: Maximum order to include (default: all computed)
        """
        r = np.array(position)
        r_mag = np.linalg.norm(r)
        r_hat = r / r_mag

        max_l = max_l or self.max_l
        k = 1 / (4 * np.pi * EPSILON_0)

        V = 0.0

        # Monopole
        if max_l >= 0:
            V += k * self.moments['monopole'] / r_mag

        # Dipole
        if max_l >= 1:
            V += k * np.dot(self.moments['dipole'], r_hat) / r_mag**2

        # Quadrupole
        if max_l >= 2:
            Q = self.moments['quadrupole']
            V += k * 0.5 * np.dot(r_hat, Q @ r_hat) / r_mag**3

        return V

    def electric_field(self, position: ArrayLike, dr: float = 1e-8) -> np.ndarray:
        """Calculate electric field from multipole expansion."""
        r = np.array(position)
        E = np.zeros(3)

        for i in range(3):
            r_plus = r.copy()
            r_plus[i] += dr
            r_minus = r.copy()
            r_minus[i] -= dr
            E[i] = -(self.potential(r_plus) - self.potential(r_minus)) / (2 * dr)

        return E


class OctupoleField(Field):
    """
    Octupole field (l=3 multipole).

    Third-order multipole with 7 independent components.

    Args:
        octupole_tensor: 3×3×3 symmetric traceless tensor
        is_electric: True for electric, False for magnetic
    """

    def __init__(self, octupole_tensor: ArrayLike, is_electric: bool = True):
        self.O = np.array(octupole_tensor)
        strength = 8.99e9 if is_electric else 1e-7
        super().__init__(strength * np.linalg.norm(self.O))
        self.is_electric = is_electric

    def field(self, position: ArrayLike) -> np.ndarray:
        """Calculate octupole field at position (approximate)."""
        r = np.array(position)
        r_mag = np.linalg.norm(r)
        r_hat = r / r_mag

        # Simplified: l=3 field falls off as 1/r^5
        # Full expression involves tensor contractions
        O_contracted = np.tensordot(self.O, np.outer(r_hat, r_hat), axes=2)

        return (self.strength / r_mag**5) * (7 * np.dot(O_contracted, r_hat) * r_hat - 2 * O_contracted)


class SphericalHarmonics(BaseClass):
    """
    Spherical harmonics Y_lm(θ, φ) for multipole expansions.

    Computes real and complex spherical harmonics.

    Examples:
        >>> sh = SphericalHarmonics()
        >>> Y_00 = sh.Y(0, 0, theta=np.pi/2, phi=0)
        >>> Y_11 = sh.Y(1, 1, theta=np.pi/4, phi=np.pi/3)
    """

    def __init__(self):
        super().__init__()

    def factorial(self, n: int) -> int:
        """Compute factorial."""
        if n <= 1:
            return 1
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

    def associated_legendre(self, l: int, m: int, x: float) -> float:
        """
        Compute associated Legendre polynomial P_l^m(x).

        Uses recurrence relations for stability.
        """
        m_abs = abs(m)

        if m_abs > l:
            return 0.0

        # Start with P_m^m
        pmm = 1.0
        if m_abs > 0:
            somx2 = np.sqrt((1 - x) * (1 + x))
            fact = 1.0
            for i in range(1, m_abs + 1):
                pmm *= -fact * somx2
                fact += 2.0

        if l == m_abs:
            return pmm

        # P_{m+1}^m
        pmmp1 = x * (2 * m_abs + 1) * pmm
        if l == m_abs + 1:
            return pmmp1

        # Recurrence for higher l
        pll = 0.0
        for ll in range(m_abs + 2, l + 1):
            pll = (x * (2 * ll - 1) * pmmp1 - (ll + m_abs - 1) * pmm) / (ll - m_abs)
            pmm = pmmp1
            pmmp1 = pll

        return pll

    def Y(self, l: int, m: int, theta: float, phi: float, real: bool = False) -> complex:
        """
        Compute spherical harmonic Y_l^m(θ, φ).

        Args:
            l: Degree (l >= 0)
            m: Order (-l <= m <= l)
            theta: Polar angle (0 to π)
            phi: Azimuthal angle (0 to 2π)
            real: If True, return real spherical harmonic
        """
        if abs(m) > l:
            return 0.0

        # Normalization factor
        norm = np.sqrt((2 * l + 1) / (4 * np.pi) *
                       self.factorial(l - abs(m)) / self.factorial(l + abs(m)))

        # Associated Legendre polynomial
        P_lm = self.associated_legendre(l, abs(m), np.cos(theta))

        # Complex exponential
        if real:
            if m > 0:
                return norm * P_lm * np.sqrt(2) * np.cos(m * phi)
            elif m < 0:
                return norm * P_lm * np.sqrt(2) * np.sin(abs(m) * phi)
            else:
                return norm * P_lm
        else:
            if m >= 0:
                return norm * P_lm * np.exp(1j * m * phi)
            else:
                return ((-1)**m) * norm * P_lm * np.exp(1j * m * phi)

    def Y_grid(
        self,
        l: int,
        m: int,
        n_theta: int = 50,
        n_phi: int = 100,
        real: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Y_lm on a grid for visualization.

        Returns:
            theta, phi, Y arrays
        """
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2 * np.pi, n_phi)

        THETA, PHI = np.meshgrid(theta, phi)
        Y_vals = np.zeros_like(THETA, dtype=complex if not real else float)

        for i in range(n_phi):
            for j in range(n_theta):
                Y_vals[i, j] = self.Y(l, m, theta[j], phi[i], real=real)

        return THETA, PHI, Y_vals


# ==============================================================================
# Phase 2.5: Materials & Media
# ==============================================================================

class DielectricMaterial(BaseClass):
    """
    Dielectric material properties.

    Models permittivity, polarization, and frequency-dependent response.

    Args:
        epsilon_r: Relative permittivity (static)
        epsilon_inf: High-frequency permittivity (for dispersion)
        relaxation_time: Debye relaxation time (s)
        conductivity: DC conductivity (S/m)

    Examples:
        >>> glass = DielectricMaterial(epsilon_r=4.0)
        >>> water = DielectricMaterial(epsilon_r=80, epsilon_inf=1.8, relaxation_time=8e-12)
    """

    def __init__(
        self,
        epsilon_r: float = 1.0,
        epsilon_inf: float = 1.0,
        relaxation_time: float = 0.0,
        conductivity: float = 0.0
    ):
        super().__init__()

        self.epsilon_r = epsilon_r
        self.epsilon_inf = epsilon_inf
        self.tau = relaxation_time
        self.sigma = conductivity

    def permittivity(self, frequency: float = 0.0) -> complex:
        """
        Calculate complex permittivity at given frequency.

        Uses Debye model: ε(ω) = ε_∞ + (ε_s - ε_∞)/(1 + iωτ) - iσ/(ωε₀)
        """
        if frequency == 0:
            return self.epsilon_r * EPSILON_0

        omega = 2 * np.pi * frequency

        # Debye relaxation
        if self.tau > 0:
            eps_debye = self.epsilon_inf + (self.epsilon_r - self.epsilon_inf) / (1 + 1j * omega * self.tau)
        else:
            eps_debye = self.epsilon_r

        # Conductivity contribution
        eps_cond = -1j * self.sigma / (omega * EPSILON_0) if omega > 0 else 0

        return (eps_debye + eps_cond) * EPSILON_0

    def refractive_index(self, frequency: float = 0.0) -> complex:
        """Calculate complex refractive index n = √(ε_r μ_r)."""
        eps_r = self.permittivity(frequency) / EPSILON_0
        return np.sqrt(eps_r)

    def skin_depth(self, frequency: float) -> float:
        """Calculate electromagnetic skin depth."""
        n = self.refractive_index(frequency)
        k = n.imag
        if k <= 0:
            return np.inf
        wavelength = C / frequency
        return wavelength / (2 * np.pi * k)

    def polarization(self, E_field: ArrayLike, frequency: float = 0.0) -> np.ndarray:
        """
        Calculate polarization P = ε₀ χ_e E.

        χ_e = ε_r - 1
        """
        E = np.array(E_field)
        eps = self.permittivity(frequency)
        chi_e = eps / EPSILON_0 - 1
        return EPSILON_0 * chi_e * E


class MagneticMaterial(BaseClass):
    """
    Magnetic material properties.

    Models permeability, magnetization, and hysteresis.

    Args:
        mu_r: Relative permeability
        saturation_magnetization: Saturation M_s (A/m)
        coercivity: Coercive field H_c (A/m)
        material_type: 'diamagnetic', 'paramagnetic', 'ferromagnetic'

    Examples:
        >>> iron = MagneticMaterial(mu_r=5000, saturation_magnetization=1.7e6,
        ...                         coercivity=80, material_type='ferromagnetic')
    """

    def __init__(
        self,
        mu_r: float = 1.0,
        saturation_magnetization: float = 0.0,
        coercivity: float = 0.0,
        material_type: str = 'paramagnetic'
    ):
        super().__init__()

        self.mu_r = mu_r
        self.M_s = saturation_magnetization
        self.H_c = coercivity
        self.material_type = material_type

        # Susceptibility
        self.chi_m = mu_r - 1

    def permeability(self) -> float:
        """Return absolute permeability μ = μ_r μ₀."""
        return self.mu_r * MU_0

    def magnetization(self, H_field: ArrayLike, use_hysteresis: bool = False) -> np.ndarray:
        """
        Calculate magnetization M = χ_m H (linear regime).

        Args:
            H_field: Applied magnetic field (A/m)
            use_hysteresis: If True, use simplified hysteresis model
        """
        H = np.array(H_field)
        H_mag = np.linalg.norm(H)

        if self.material_type == 'ferromagnetic' and self.M_s > 0:
            # Langevin-like saturation
            if H_mag > 0:
                M_mag = self.M_s * np.tanh(self.chi_m * H_mag / self.M_s)
            else:
                M_mag = 0.0
            return M_mag * H / H_mag if H_mag > 0 else np.zeros(3)
        else:
            # Linear response
            return self.chi_m * H

    def B_field(self, H_field: ArrayLike) -> np.ndarray:
        """Calculate B = μ₀(H + M)."""
        H = np.array(H_field)
        M = self.magnetization(H)
        return MU_0 * (H + M)

    def energy_density(self, H_field: ArrayLike) -> float:
        """Calculate magnetic energy density u = B·H/2."""
        H = np.array(H_field)
        B = self.B_field(H)
        return 0.5 * np.dot(B, H)


class ConductorSkin(BaseClass):
    """
    Conductor skin effect and AC resistance.

    Models frequency-dependent resistance due to current crowding
    near the surface at high frequencies.

    Args:
        conductivity: DC conductivity (S/m)
        permeability: Relative permeability

    Examples:
        >>> copper = ConductorSkin(conductivity=5.96e7)  # Copper
        >>> delta = copper.skin_depth(frequency=1e6)
        >>> R_ac = copper.ac_resistance(frequency=1e6, radius=1e-3, length=1.0)
    """

    def __init__(
        self,
        conductivity: float,
        permeability: float = 1.0
    ):
        super().__init__()

        validate_positive(conductivity, "conductivity")

        self.sigma = conductivity
        self.mu_r = permeability
        self.mu = permeability * MU_0

    def skin_depth(self, frequency: float) -> float:
        """
        Calculate skin depth δ = √(2/(ωμσ)).

        Args:
            frequency: Operating frequency (Hz)

        Returns:
            Skin depth in meters
        """
        if frequency <= 0:
            return np.inf

        omega = 2 * np.pi * frequency
        return np.sqrt(2 / (omega * self.mu * self.sigma))

    def surface_resistance(self, frequency: float) -> float:
        """
        Calculate surface resistance R_s = 1/(σδ).

        Args:
            frequency: Operating frequency (Hz)

        Returns:
            Surface resistance in Ω
        """
        delta = self.skin_depth(frequency)
        return 1 / (self.sigma * delta)

    def ac_resistance(
        self,
        frequency: float,
        radius: float,
        length: float
    ) -> float:
        """
        Calculate AC resistance of a cylindrical conductor.

        For δ << radius: R_ac ≈ R_s * length / (2πr)
        For δ >> radius: R_ac ≈ R_dc

        Args:
            frequency: Operating frequency (Hz)
            radius: Conductor radius (m)
            length: Conductor length (m)
        """
        R_dc = length / (self.sigma * np.pi * radius**2)

        if frequency <= 0:
            return R_dc

        delta = self.skin_depth(frequency)

        if delta > radius:
            # DC limit
            return R_dc
        else:
            # High frequency: current in thin shell
            R_s = self.surface_resistance(frequency)
            return R_s * length / (2 * np.pi * radius)

    def ac_to_dc_ratio(self, frequency: float, radius: float) -> float:
        """Calculate R_ac / R_dc ratio."""
        R_dc = 1 / (self.sigma * np.pi * radius**2)
        R_ac = self.ac_resistance(frequency, radius, 1.0)
        return R_ac / R_dc


class PlasmaDispersion(BaseClass):
    """
    Plasma dispersion and wave propagation.

    Models electromagnetic wave behavior in plasma including
    plasma frequency, cutoff, and dispersion relations.

    Args:
        electron_density: Electron density (m⁻³)
        collision_frequency: Electron-neutral collision frequency (Hz)
        magnetic_field: Background magnetic field (T) for magnetized plasma

    Examples:
        >>> ionosphere = PlasmaDispersion(electron_density=1e12)
        >>> f_p = ionosphere.plasma_frequency()
        >>> n = ionosphere.refractive_index(frequency=10e6)
    """

    def __init__(
        self,
        electron_density: float,
        collision_frequency: float = 0.0,
        magnetic_field: float = 0.0
    ):
        super().__init__()

        self.n_e = electron_density
        self.nu = collision_frequency
        self.B0 = magnetic_field

        self.m_e = 9.109e-31  # Electron mass
        self.e = E_CHARGE

    def plasma_frequency(self) -> float:
        """
        Calculate plasma frequency ω_p = √(n_e e²/(ε₀ m_e)).

        Returns:
            Plasma frequency in Hz
        """
        omega_p = np.sqrt(self.n_e * self.e**2 / (EPSILON_0 * self.m_e))
        return omega_p / (2 * np.pi)

    def cyclotron_frequency(self) -> float:
        """
        Calculate electron cyclotron frequency ω_c = eB/m_e.

        Returns:
            Cyclotron frequency in Hz
        """
        if self.B0 <= 0:
            return 0.0
        omega_c = self.e * self.B0 / self.m_e
        return omega_c / (2 * np.pi)

    def permittivity(self, frequency: float) -> complex:
        """
        Calculate cold plasma permittivity.

        ε_r = 1 - ω_p²/(ω² + iνω)
        """
        if frequency <= 0:
            return 1.0 - (self.plasma_frequency() / 1)**2  # Limit

        omega = 2 * np.pi * frequency
        omega_p = 2 * np.pi * self.plasma_frequency()

        eps_r = 1 - omega_p**2 / (omega**2 + 1j * self.nu * omega)
        return eps_r

    def refractive_index(self, frequency: float) -> complex:
        """
        Calculate refractive index n = √ε_r.

        n² = 1 - (f_p/f)² for collisionless plasma
        """
        eps_r = self.permittivity(frequency)
        return np.sqrt(eps_r)

    def is_propagating(self, frequency: float) -> bool:
        """Check if wave propagates (f > f_p)."""
        return frequency > self.plasma_frequency()

    def phase_velocity(self, frequency: float) -> float:
        """Calculate phase velocity v_p = c/n."""
        n = self.refractive_index(frequency)
        if n.real <= 0:
            return np.inf
        return C / n.real

    def group_velocity(self, frequency: float, df: float = 1e3) -> float:
        """Calculate group velocity v_g = dω/dk."""
        if not self.is_propagating(frequency):
            return 0.0

        k1 = 2 * np.pi * frequency * self.refractive_index(frequency).real / C
        k2 = 2 * np.pi * (frequency + df) * self.refractive_index(frequency + df).real / C

        if abs(k2 - k1) < 1e-15:
            return C

        return 2 * np.pi * df / (k2 - k1)

    def debye_length(self, temperature: float) -> float:
        """
        Calculate Debye length λ_D = √(ε₀ k_B T / (n_e e²)).

        Args:
            temperature: Electron temperature (K)
        """
        k_B = 1.381e-23  # Boltzmann constant
        return np.sqrt(EPSILON_0 * k_B * temperature / (self.n_e * self.e**2))


class MetamaterialUnit(BaseClass):
    """
    Metamaterial unit cell with effective medium properties.

    Models materials with engineered electromagnetic response,
    including negative index materials.

    Args:
        epsilon_eff: Effective relative permittivity (can be complex/negative)
        mu_eff: Effective relative permeability (can be complex/negative)
        resonance_frequency: Resonance frequency for Lorentz model (Hz)
        damping: Damping factor for loss

    Examples:
        >>> # Double-negative metamaterial
        >>> nim = MetamaterialUnit(epsilon_eff=-1.5, mu_eff=-1.2)
        >>> n = nim.refractive_index()  # Negative index
    """

    def __init__(
        self,
        epsilon_eff: complex = 1.0,
        mu_eff: complex = 1.0,
        resonance_frequency: float = 0.0,
        damping: float = 0.0,
        plasma_frequency: float = 0.0
    ):
        super().__init__()

        self.epsilon_eff = epsilon_eff
        self.mu_eff = mu_eff
        self.f_res = resonance_frequency
        self.gamma = damping
        self.f_p = plasma_frequency

    def permittivity(self, frequency: float = 0.0) -> complex:
        """
        Calculate frequency-dependent permittivity.

        Uses Drude-Lorentz model if resonance parameters given.
        """
        if self.f_p > 0 and frequency > 0:
            # Drude model
            omega = 2 * np.pi * frequency
            omega_p = 2 * np.pi * self.f_p
            return 1 - omega_p**2 / (omega**2 + 1j * self.gamma * omega)

        return self.epsilon_eff

    def permeability(self, frequency: float = 0.0) -> complex:
        """
        Calculate frequency-dependent permeability.

        Uses Lorentz model for magnetic resonance.
        """
        if self.f_res > 0 and frequency > 0:
            # Lorentz resonance model
            omega = 2 * np.pi * frequency
            omega_0 = 2 * np.pi * self.f_res
            F = 0.5  # Oscillator strength
            return 1 + F * omega_0**2 / (omega_0**2 - omega**2 - 1j * self.gamma * omega)

        return self.mu_eff

    def refractive_index(self, frequency: float = 0.0) -> complex:
        """
        Calculate refractive index n = ±√(εμ).

        Sign chosen based on material type (negative for NIMs).
        """
        eps = self.permittivity(frequency)
        mu = self.permeability(frequency)

        n_squared = eps * mu
        n = np.sqrt(n_squared)

        # For double-negative materials, choose negative root
        if np.real(eps) < 0 and np.real(mu) < 0:
            n = -n

        return n

    def impedance(self, frequency: float = 0.0) -> complex:
        """Calculate wave impedance Z = √(μ/ε)."""
        eps = self.permittivity(frequency) * EPSILON_0
        mu = self.permeability(frequency) * MU_0
        return np.sqrt(mu / eps)

    def is_double_negative(self, frequency: float = 0.0) -> bool:
        """Check if both ε and μ are negative (NIM)."""
        eps = self.permittivity(frequency)
        mu = self.permeability(frequency)
        return np.real(eps) < 0 and np.real(mu) < 0

    def is_single_negative(self, frequency: float = 0.0) -> str:
        """Check for single-negative behavior."""
        eps = self.permittivity(frequency)
        mu = self.permeability(frequency)

        if np.real(eps) < 0 and np.real(mu) > 0:
            return 'ENG'  # Epsilon-negative (like metals)
        elif np.real(eps) > 0 and np.real(mu) < 0:
            return 'MNG'  # Mu-negative
        else:
            return 'none'

    def group_velocity_sign(self, frequency: float = 0.0) -> int:
        """
        Determine sign of group velocity.

        Returns:
            -1 for backward wave (NIM), +1 for forward wave
        """
        return -1 if self.is_double_negative(frequency) else 1
