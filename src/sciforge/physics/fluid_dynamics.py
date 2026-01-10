"""
Fluid Dynamics Module

Implements fundamental fluid mechanics equations, flow characterization,
and turbulence modeling for computational fluid dynamics.

Classes:
- NavierStokesSolver: Viscous incompressible flow solver
- EulerFluidSolver: Inviscid flow solver
- StressTensor: Cauchy stress tensor for continuum mechanics
- StrainTensor: Deformation/strain tensor
- ContinuityEquation: Mass conservation solver
- ReynoldsNumber: Flow regime characterization
- VorticityField: Curl of velocity field
- StreamFunction: 2D incompressible flow representation
- VelocityPotential: Irrotational flow potential
- BoundaryLayer: Boundary layer analysis
- TurbulenceModel: RANS turbulence modeling
- KolmogorovScale: Turbulent cascade scales
- EnergySpectrum: Turbulent energy distribution
"""

import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, Tuple, Callable, Union
from scipy import sparse
from scipy.sparse.linalg import spsolve
from ..core.base import BaseClass


class StressTensor(BaseClass):
    """
    Cauchy stress tensor for continuum mechanics.

    The stress tensor σ_ij represents the force per unit area acting
    on a surface with normal in direction j, in direction i.

    Args:
        components: 3x3 stress tensor components (Pa)
                   [[σ_xx, σ_xy, σ_xz],
                    [σ_yx, σ_yy, σ_yz],
                    [σ_zx, σ_zy, σ_zz]]
    """

    def __init__(self, components: ArrayLike):
        super().__init__()
        self.tensor = np.asarray(components, dtype=float)
        if self.tensor.shape != (3, 3):
            raise ValueError("Stress tensor must be 3x3")

    @classmethod
    def from_pressure(cls, pressure: float) -> 'StressTensor':
        """Create isotropic stress tensor from pressure (σ = -p I)."""
        return cls(-pressure * np.eye(3))

    @classmethod
    def from_viscous(cls, velocity_gradient: ArrayLike,
                     mu: float, bulk_viscosity: float = 0.0) -> 'StressTensor':
        """
        Create viscous stress tensor from velocity gradient.

        τ_ij = μ(∂v_i/∂x_j + ∂v_j/∂x_i) + (κ - 2μ/3)(∇·v)δ_ij

        Args:
            velocity_gradient: ∂v_i/∂x_j tensor (3x3)
            mu: Dynamic viscosity (Pa·s)
            bulk_viscosity: Bulk viscosity κ (Pa·s)
        """
        grad_v = np.asarray(velocity_gradient)
        # Symmetric part (strain rate)
        strain_rate = 0.5 * (grad_v + grad_v.T)
        # Divergence
        div_v = np.trace(grad_v)
        # Viscous stress
        tau = 2 * mu * strain_rate + (bulk_viscosity - 2*mu/3) * div_v * np.eye(3)
        return cls(tau)

    @property
    def pressure(self) -> float:
        """Hydrostatic pressure: p = -1/3 * tr(σ)."""
        return -np.trace(self.tensor) / 3

    @property
    def deviatoric(self) -> np.ndarray:
        """Deviatoric (shear) stress: σ' = σ + p*I."""
        return self.tensor + self.pressure * np.eye(3)

    @property
    def von_mises(self) -> float:
        """Von Mises equivalent stress."""
        s = self.deviatoric
        return np.sqrt(1.5 * np.sum(s * s))

    @property
    def principal_stresses(self) -> np.ndarray:
        """Principal stresses (eigenvalues), sorted descending."""
        eigenvalues = np.linalg.eigvalsh(self.tensor)
        return np.sort(eigenvalues)[::-1]

    @property
    def principal_directions(self) -> np.ndarray:
        """Principal stress directions (eigenvectors)."""
        _, eigenvectors = np.linalg.eigh(self.tensor)
        return eigenvectors

    def traction(self, normal: ArrayLike) -> np.ndarray:
        """
        Traction vector on surface with given normal.

        t = σ · n

        Args:
            normal: Unit normal vector to surface

        Returns:
            Traction vector (force per unit area)
        """
        n = np.asarray(normal)
        n = n / np.linalg.norm(n)
        return self.tensor @ n

    def invariants(self) -> Tuple[float, float, float]:
        """
        Stress tensor invariants.

        Returns:
            I1: tr(σ)
            I2: 1/2 * (tr(σ)² - tr(σ²))
            I3: det(σ)
        """
        I1 = np.trace(self.tensor)
        I2 = 0.5 * (I1**2 - np.trace(self.tensor @ self.tensor))
        I3 = np.linalg.det(self.tensor)
        return I1, I2, I3


class StrainTensor(BaseClass):
    """
    Strain tensor for deformation analysis.

    The infinitesimal strain tensor ε_ij = 1/2 (∂u_i/∂x_j + ∂u_j/∂x_i)
    where u is the displacement field.

    Args:
        components: 3x3 strain tensor components (dimensionless)
    """

    def __init__(self, components: ArrayLike):
        super().__init__()
        self.tensor = np.asarray(components, dtype=float)
        if self.tensor.shape != (3, 3):
            raise ValueError("Strain tensor must be 3x3")
        # Symmetrize
        self.tensor = 0.5 * (self.tensor + self.tensor.T)

    @classmethod
    def from_displacement_gradient(cls, displacement_gradient: ArrayLike) -> 'StrainTensor':
        """
        Create strain tensor from displacement gradient.

        ε = 1/2 (∇u + (∇u)ᵀ)
        """
        grad_u = np.asarray(displacement_gradient)
        return cls(0.5 * (grad_u + grad_u.T))

    @classmethod
    def from_velocity_gradient(cls, velocity_gradient: ArrayLike, dt: float) -> 'StrainTensor':
        """Create strain increment from velocity gradient and time step."""
        grad_v = np.asarray(velocity_gradient)
        return cls(0.5 * (grad_v + grad_v.T) * dt)

    @property
    def volumetric(self) -> float:
        """Volumetric strain: ε_v = tr(ε) = ΔV/V."""
        return np.trace(self.tensor)

    @property
    def deviatoric(self) -> np.ndarray:
        """Deviatoric strain: ε' = ε - (ε_v/3)*I."""
        return self.tensor - (self.volumetric / 3) * np.eye(3)

    @property
    def equivalent(self) -> float:
        """Equivalent (von Mises) strain."""
        e = self.deviatoric
        return np.sqrt(2/3 * np.sum(e * e))

    @property
    def principal_strains(self) -> np.ndarray:
        """Principal strains (eigenvalues), sorted descending."""
        eigenvalues = np.linalg.eigvalsh(self.tensor)
        return np.sort(eigenvalues)[::-1]

    def rotate(self, rotation_matrix: ArrayLike) -> 'StrainTensor':
        """Rotate strain tensor: ε' = R ε Rᵀ."""
        R = np.asarray(rotation_matrix)
        return StrainTensor(R @ self.tensor @ R.T)


class ContinuityEquation(BaseClass):
    """
    Mass conservation (continuity) equation solver.

    ∂ρ/∂t + ∇·(ρv) = 0

    For incompressible flow: ∇·v = 0

    Args:
        nx, ny, nz: Grid dimensions
        dx, dy, dz: Grid spacing
        incompressible: If True, solve ∇·v = 0
    """

    def __init__(self, nx: int, ny: int, nz: int = 1,
                 dx: float = 1.0, dy: float = 1.0, dz: float = 1.0,
                 incompressible: bool = True):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.incompressible = incompressible
        self.is_2d = (nz == 1)

    def divergence(self, vx: np.ndarray, vy: np.ndarray,
                   vz: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute velocity divergence ∇·v.

        Args:
            vx, vy, vz: Velocity components on grid

        Returns:
            Divergence field
        """
        div = np.zeros_like(vx)

        # Central differences
        div[1:-1, :] += (vx[2:, :] - vx[:-2, :]) / (2 * self.dx)
        div[:, 1:-1] += (vy[:, 2:] - vy[:, :-2]) / (2 * self.dy)

        if not self.is_2d and vz is not None:
            div[:, :, 1:-1] += (vz[:, :, 2:] - vz[:, :, :-2]) / (2 * self.dz)

        return div

    def check_incompressibility(self, vx: np.ndarray, vy: np.ndarray,
                                vz: Optional[np.ndarray] = None,
                                tolerance: float = 1e-6) -> Tuple[bool, float]:
        """
        Check if velocity field is divergence-free.

        Returns:
            (is_incompressible, max_divergence)
        """
        div = self.divergence(vx, vy, vz)
        max_div = np.max(np.abs(div))
        return max_div < tolerance, max_div

    def project_incompressible(self, vx: np.ndarray, vy: np.ndarray,
                               vz: Optional[np.ndarray] = None,
                               n_iter: int = 100) -> Tuple[np.ndarray, ...]:
        """
        Project velocity field to be divergence-free using pressure correction.

        Solves: ∇²p = ∇·v, then v* = v - ∇p

        Returns:
            Corrected velocity components
        """
        div = self.divergence(vx, vy, vz)

        # Solve Poisson equation for pressure correction using Jacobi iteration
        p = np.zeros_like(div)

        for _ in range(n_iter):
            p_new = np.zeros_like(p)

            # Jacobi iteration for Laplacian
            if self.is_2d:
                p_new[1:-1, 1:-1] = 0.25 * (
                    p[2:, 1:-1] + p[:-2, 1:-1] +
                    p[1:-1, 2:] + p[1:-1, :-2] -
                    self.dx**2 * div[1:-1, 1:-1]
                )
            else:
                p_new[1:-1, 1:-1, 1:-1] = (1/6) * (
                    p[2:, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1] +
                    p[1:-1, 2:, 1:-1] + p[1:-1, :-2, 1:-1] +
                    p[1:-1, 1:-1, 2:] + p[1:-1, 1:-1, :-2] -
                    self.dx**2 * div[1:-1, 1:-1, 1:-1]
                )
            p = p_new

        # Correct velocity
        vx_new = vx.copy()
        vy_new = vy.copy()

        vx_new[1:-1, :] -= (p[2:, :] - p[:-2, :]) / (2 * self.dx)
        vy_new[:, 1:-1] -= (p[:, 2:] - p[:, :-2]) / (2 * self.dy)

        if not self.is_2d and vz is not None:
            vz_new = vz.copy()
            vz_new[:, :, 1:-1] -= (p[:, :, 2:] - p[:, :, :-2]) / (2 * self.dz)
            return vx_new, vy_new, vz_new

        return vx_new, vy_new

    def update_density(self, rho: np.ndarray, vx: np.ndarray, vy: np.ndarray,
                       dt: float, vz: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Update density field using continuity equation (compressible flow).

        ∂ρ/∂t = -∇·(ρv)
        """
        # Compute flux divergence
        flux_x = rho * vx
        flux_y = rho * vy

        div_flux = np.zeros_like(rho)
        div_flux[1:-1, :] += (flux_x[2:, :] - flux_x[:-2, :]) / (2 * self.dx)
        div_flux[:, 1:-1] += (flux_y[:, 2:] - flux_y[:, :-2]) / (2 * self.dy)

        if not self.is_2d and vz is not None:
            flux_z = rho * vz
            div_flux[:, :, 1:-1] += (flux_z[:, :, 2:] - flux_z[:, :, :-2]) / (2 * self.dz)

        return rho - dt * div_flux


class NavierStokesSolver(BaseClass):
    """
    Navier-Stokes equation solver for incompressible viscous flow.

    ∂v/∂t + (v·∇)v = -∇p/ρ + ν∇²v + f
    ∇·v = 0

    Uses fractional step (projection) method.

    Args:
        nx, ny: Grid dimensions
        dx, dy: Grid spacing (m)
        nu: Kinematic viscosity (m²/s)
        rho: Density (kg/m³)
        dt: Time step (s)
    """

    def __init__(self, nx: int, ny: int, dx: float, dy: float,
                 nu: float = 1e-6, rho: float = 1000.0, dt: float = 0.001):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.nu = nu
        self.rho = rho
        self.dt = dt

        # Velocity fields
        self.vx = np.zeros((nx, ny))
        self.vy = np.zeros((nx, ny))
        self.p = np.zeros((nx, ny))

        # Time tracking
        self.time = 0.0

        # Continuity helper
        self.continuity = ContinuityEquation(nx, ny, 1, dx, dy, 1.0, True)

    def set_velocity(self, vx: np.ndarray, vy: np.ndarray):
        """Set initial velocity field."""
        self.vx = np.asarray(vx).copy()
        self.vy = np.asarray(vy).copy()

    def set_boundary_conditions(self, bc_type: str = 'no_slip',
                                 walls: str = 'all'):
        """
        Apply boundary conditions.

        Args:
            bc_type: 'no_slip', 'free_slip', or 'periodic'
            walls: 'all', 'top', 'bottom', 'left', 'right'
        """
        self.bc_type = bc_type
        self.walls = walls

    def _apply_bc(self):
        """Apply boundary conditions to velocity field."""
        bc_type = getattr(self, 'bc_type', 'no_slip')

        if bc_type == 'no_slip':
            # Zero velocity at walls
            self.vx[0, :] = 0; self.vx[-1, :] = 0
            self.vx[:, 0] = 0; self.vx[:, -1] = 0
            self.vy[0, :] = 0; self.vy[-1, :] = 0
            self.vy[:, 0] = 0; self.vy[:, -1] = 0
        elif bc_type == 'free_slip':
            # Zero normal velocity, zero normal gradient of tangential
            self.vx[0, :] = self.vx[1, :]; self.vx[-1, :] = self.vx[-2, :]
            self.vy[:, 0] = self.vy[:, 1]; self.vy[:, -1] = self.vy[:, -2]
            self.vy[0, :] = 0; self.vy[-1, :] = 0
            self.vx[:, 0] = 0; self.vx[:, -1] = 0
        elif bc_type == 'periodic':
            self.vx[0, :] = self.vx[-2, :]; self.vx[-1, :] = self.vx[1, :]
            self.vx[:, 0] = self.vx[:, -2]; self.vx[:, -1] = self.vx[:, 1]
            self.vy[0, :] = self.vy[-2, :]; self.vy[-1, :] = self.vy[1, :]
            self.vy[:, 0] = self.vy[:, -2]; self.vy[:, -1] = self.vy[:, 1]

    def _advection(self, phi: np.ndarray) -> np.ndarray:
        """Compute advection term (v·∇)φ using upwind scheme."""
        advect = np.zeros_like(phi)

        # Upwind in x
        vx_pos = np.maximum(self.vx, 0)
        vx_neg = np.minimum(self.vx, 0)
        advect[1:-1, :] += vx_pos[1:-1, :] * (phi[1:-1, :] - phi[:-2, :]) / self.dx
        advect[1:-1, :] += vx_neg[1:-1, :] * (phi[2:, :] - phi[1:-1, :]) / self.dx

        # Upwind in y
        vy_pos = np.maximum(self.vy, 0)
        vy_neg = np.minimum(self.vy, 0)
        advect[:, 1:-1] += vy_pos[:, 1:-1] * (phi[:, 1:-1] - phi[:, :-2]) / self.dy
        advect[:, 1:-1] += vy_neg[:, 1:-1] * (phi[:, 2:] - phi[:, 1:-1]) / self.dy

        return advect

    def _diffusion(self, phi: np.ndarray) -> np.ndarray:
        """Compute diffusion term ν∇²φ using central differences."""
        diff = np.zeros_like(phi)

        diff[1:-1, 1:-1] = self.nu * (
            (phi[2:, 1:-1] - 2*phi[1:-1, 1:-1] + phi[:-2, 1:-1]) / self.dx**2 +
            (phi[1:-1, 2:] - 2*phi[1:-1, 1:-1] + phi[1:-1, :-2]) / self.dy**2
        )

        return diff

    def _solve_pressure(self, div: np.ndarray, n_iter: int = 50) -> np.ndarray:
        """Solve pressure Poisson equation ∇²p = ρ/dt * ∇·v*."""
        p = self.p.copy()
        rhs = self.rho / self.dt * div

        for _ in range(n_iter):
            p_new = np.zeros_like(p)
            p_new[1:-1, 1:-1] = 0.25 * (
                p[2:, 1:-1] + p[:-2, 1:-1] +
                p[1:-1, 2:] + p[1:-1, :-2] -
                self.dx**2 * rhs[1:-1, 1:-1]
            )
            # Neumann BC for pressure
            p_new[0, :] = p_new[1, :]; p_new[-1, :] = p_new[-2, :]
            p_new[:, 0] = p_new[:, 1]; p_new[:, -1] = p_new[:, -2]
            p = p_new

        return p

    def step(self, force_x: Optional[np.ndarray] = None,
             force_y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advance one time step using projection method.

        Args:
            force_x, force_y: External body forces (N/kg)

        Returns:
            Updated velocity fields (vx, vy)
        """
        # Step 1: Compute intermediate velocity (without pressure)
        vx_star = self.vx.copy()
        vy_star = self.vy.copy()

        # Advection
        vx_star -= self.dt * self._advection(self.vx)
        vy_star -= self.dt * self._advection(self.vy)

        # Diffusion
        vx_star += self.dt * self._diffusion(self.vx)
        vy_star += self.dt * self._diffusion(self.vy)

        # External forces
        if force_x is not None:
            vx_star += self.dt * force_x
        if force_y is not None:
            vy_star += self.dt * force_y

        # Step 2: Solve pressure Poisson equation
        div = self.continuity.divergence(vx_star, vy_star)
        self.p = self._solve_pressure(div)

        # Step 3: Correct velocity to be divergence-free
        self.vx = vx_star.copy()
        self.vy = vy_star.copy()

        self.vx[1:-1, :] -= self.dt / self.rho * (self.p[2:, :] - self.p[:-2, :]) / (2*self.dx)
        self.vy[:, 1:-1] -= self.dt / self.rho * (self.p[:, 2:] - self.p[:, :-2]) / (2*self.dy)

        # Apply boundary conditions
        self._apply_bc()

        self.time += self.dt

        return self.vx, self.vy

    def simulate(self, n_steps: int,
                 force_func: Optional[Callable] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run simulation for multiple steps.

        Args:
            n_steps: Number of time steps
            force_func: Optional function(t, x, y) -> (fx, fy)

        Returns:
            Final velocity fields
        """
        for _ in range(n_steps):
            fx, fy = None, None
            if force_func is not None:
                x = np.arange(self.nx) * self.dx
                y = np.arange(self.ny) * self.dy
                X, Y = np.meshgrid(x, y, indexing='ij')
                fx, fy = force_func(self.time, X, Y)
            self.step(fx, fy)

        return self.vx, self.vy

    def get_vorticity(self) -> np.ndarray:
        """Compute vorticity ω = ∂vy/∂x - ∂vx/∂y."""
        omega = np.zeros((self.nx, self.ny))
        omega[1:-1, 1:-1] = (
            (self.vy[2:, 1:-1] - self.vy[:-2, 1:-1]) / (2*self.dx) -
            (self.vx[1:-1, 2:] - self.vx[1:-1, :-2]) / (2*self.dy)
        )
        return omega

    def get_kinetic_energy(self) -> float:
        """Total kinetic energy per unit depth."""
        return 0.5 * self.rho * np.sum(self.vx**2 + self.vy**2) * self.dx * self.dy

    def get_enstrophy(self) -> float:
        """Enstrophy (integral of vorticity squared)."""
        omega = self.get_vorticity()
        return 0.5 * np.sum(omega**2) * self.dx * self.dy


class EulerFluidSolver(BaseClass):
    """
    Euler equations solver for inviscid compressible flow.

    ∂ρ/∂t + ∇·(ρv) = 0
    ∂(ρv)/∂t + ∇·(ρv⊗v) + ∇p = 0
    ∂E/∂t + ∇·((E+p)v) = 0

    Uses finite volume method with Rusanov (local Lax-Friedrichs) flux.

    Args:
        nx, ny: Grid dimensions
        dx, dy: Grid spacing
        gamma: Ratio of specific heats (default 1.4 for air)
    """

    def __init__(self, nx: int, ny: int, dx: float, dy: float, gamma: float = 1.4):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.gamma = gamma

        # Conservative variables: [ρ, ρu, ρv, E]
        self.rho = np.ones((nx, ny))
        self.rho_u = np.zeros((nx, ny))
        self.rho_v = np.zeros((nx, ny))
        self.E = np.ones((nx, ny)) / (gamma - 1)  # Internal energy

        self.time = 0.0

    def set_initial_conditions(self, rho: np.ndarray, u: np.ndarray,
                               v: np.ndarray, p: np.ndarray):
        """Set initial primitive variables."""
        self.rho = rho.copy()
        self.rho_u = rho * u
        self.rho_v = rho * v
        # Total energy E = p/(γ-1) + 0.5*ρ*(u² + v²)
        self.E = p / (self.gamma - 1) + 0.5 * rho * (u**2 + v**2)

    def get_primitives(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get primitive variables (ρ, u, v, p) from conservative."""
        u = self.rho_u / self.rho
        v = self.rho_v / self.rho
        p = (self.gamma - 1) * (self.E - 0.5 * self.rho * (u**2 + v**2))
        return self.rho, u, v, p

    def get_sound_speed(self) -> np.ndarray:
        """Local sound speed c = √(γp/ρ)."""
        _, u, v, p = self.get_primitives()
        return np.sqrt(self.gamma * p / self.rho)

    def _rusanov_flux_x(self, UL: np.ndarray, UR: np.ndarray) -> np.ndarray:
        """Rusanov flux in x-direction."""
        # Unpack left state
        rhoL, rho_uL, rho_vL, EL = UL[0], UL[1], UL[2], UL[3]
        uL = rho_uL / rhoL
        vL = rho_vL / rhoL
        pL = (self.gamma - 1) * (EL - 0.5 * rhoL * (uL**2 + vL**2))
        cL = np.sqrt(self.gamma * np.abs(pL) / rhoL)

        # Unpack right state
        rhoR, rho_uR, rho_vR, ER = UR[0], UR[1], UR[2], UR[3]
        uR = rho_uR / rhoR
        vR = rho_vR / rhoR
        pR = (self.gamma - 1) * (ER - 0.5 * rhoR * (uR**2 + vR**2))
        cR = np.sqrt(self.gamma * np.abs(pR) / rhoR)

        # Physical fluxes
        FL = np.array([rho_uL, rho_uL*uL + pL, rho_uL*vL, (EL + pL)*uL])
        FR = np.array([rho_uR, rho_uR*uR + pR, rho_uR*vR, (ER + pR)*uR])

        # Maximum wave speed
        smax = np.maximum(np.abs(uL) + cL, np.abs(uR) + cR)

        # Rusanov flux
        return 0.5 * (FL + FR - smax * (UR - UL))

    def step(self, dt: float):
        """Advance one time step using first-order finite volume."""
        U = np.array([self.rho, self.rho_u, self.rho_v, self.E])

        # X-direction fluxes
        dU = np.zeros_like(U)
        for i in range(1, self.nx):
            flux = self._rusanov_flux_x(U[:, i-1, :], U[:, i, :])
            dU[:, i-1, :] += flux / self.dx
            dU[:, i, :] -= flux / self.dx

        # Y-direction fluxes (swap u,v for y-flux)
        for j in range(1, self.ny):
            UL = np.array([U[0, :, j-1], U[2, :, j-1], U[1, :, j-1], U[3, :, j-1]])
            UR = np.array([U[0, :, j], U[2, :, j], U[1, :, j], U[3, :, j]])
            flux_y = self._rusanov_flux_x(UL, UR)
            # Swap back
            flux = np.array([flux_y[0], flux_y[2], flux_y[1], flux_y[3]])
            dU[:, :, j-1] += flux / self.dy
            dU[:, :, j] -= flux / self.dy

        # Update
        U -= dt * dU

        self.rho = U[0]
        self.rho_u = U[1]
        self.rho_v = U[2]
        self.E = U[3]

        # Floor density and energy
        self.rho = np.maximum(self.rho, 1e-10)
        self.E = np.maximum(self.E, 1e-10)

        self.time += dt

    def get_mach_number(self) -> np.ndarray:
        """Local Mach number."""
        _, u, v, _ = self.get_primitives()
        c = self.get_sound_speed()
        return np.sqrt(u**2 + v**2) / c


class ReynoldsNumber(BaseClass):
    """
    Reynolds number calculator and flow regime characterization.

    Re = ρVL/μ = VL/ν

    Characterizes the ratio of inertial to viscous forces.

    Args:
        velocity: Characteristic velocity (m/s)
        length: Characteristic length (m)
        nu: Kinematic viscosity (m²/s), or
        mu: Dynamic viscosity (Pa·s) with rho
        rho: Density (kg/m³), optional if nu is given
    """

    def __init__(self, velocity: float, length: float,
                 nu: Optional[float] = None, mu: Optional[float] = None,
                 rho: float = 1000.0):
        super().__init__()
        self.velocity = velocity
        self.length = length
        self.rho = rho

        if nu is not None:
            self.nu = nu
        elif mu is not None:
            self.nu = mu / rho
        else:
            raise ValueError("Must provide either nu or mu")

        self.mu = self.nu * rho

    @property
    def value(self) -> float:
        """Reynolds number value."""
        return self.velocity * self.length / self.nu

    @property
    def regime(self) -> str:
        """Flow regime classification."""
        Re = self.value
        if Re < 1:
            return "creeping"
        elif Re < 2300:
            return "laminar"
        elif Re < 4000:
            return "transitional"
        else:
            return "turbulent"

    @classmethod
    def pipe_flow(cls, velocity: float, diameter: float, nu: float) -> 'ReynoldsNumber':
        """Reynolds number for pipe flow (L = D)."""
        return cls(velocity, diameter, nu=nu)

    @classmethod
    def flat_plate(cls, velocity: float, length: float, nu: float) -> 'ReynoldsNumber':
        """Reynolds number for flow over flat plate."""
        return cls(velocity, length, nu=nu)

    def critical_velocity(self, Re_crit: float = 2300) -> float:
        """Velocity at which flow becomes critical."""
        return Re_crit * self.nu / self.length

    def boundary_layer_thickness(self, x: float) -> float:
        """
        Laminar boundary layer thickness at distance x.

        δ ≈ 5x/√(Rex) (Blasius solution)
        """
        Re_x = self.velocity * x / self.nu
        return 5 * x / np.sqrt(Re_x) if Re_x > 0 else 0


class VorticityField(BaseClass):
    """
    Vorticity field calculator and analyzer.

    ω = ∇ × v

    In 2D: ω = ∂v/∂x - ∂u/∂y (scalar)
    In 3D: ω = (∂w/∂y - ∂v/∂z, ∂u/∂z - ∂w/∂x, ∂v/∂x - ∂u/∂y)

    Args:
        vx, vy, vz: Velocity components on grid
        dx, dy, dz: Grid spacing
    """

    def __init__(self, vx: np.ndarray, vy: np.ndarray,
                 dx: float, dy: float,
                 vz: Optional[np.ndarray] = None, dz: float = 1.0):
        super().__init__()
        self.vx = np.asarray(vx)
        self.vy = np.asarray(vy)
        self.vz = np.asarray(vz) if vz is not None else None
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.is_2d = (vz is None)

        self._compute_vorticity()

    def _compute_vorticity(self):
        """Compute vorticity field using central differences."""
        if self.is_2d:
            # 2D: ω = ∂vy/∂x - ∂vx/∂y
            self.omega = np.zeros_like(self.vx)
            self.omega[1:-1, 1:-1] = (
                (self.vy[2:, 1:-1] - self.vy[:-2, 1:-1]) / (2*self.dx) -
                (self.vx[1:-1, 2:] - self.vx[1:-1, :-2]) / (2*self.dy)
            )
        else:
            # 3D: ω = ∇ × v
            shape = self.vx.shape
            self.omega_x = np.zeros(shape)
            self.omega_y = np.zeros(shape)
            self.omega_z = np.zeros(shape)

            # ωx = ∂vz/∂y - ∂vy/∂z
            self.omega_x[:, 1:-1, 1:-1] = (
                (self.vz[:, 2:, 1:-1] - self.vz[:, :-2, 1:-1]) / (2*self.dy) -
                (self.vy[:, 1:-1, 2:] - self.vy[:, 1:-1, :-2]) / (2*self.dz)
            )

            # ωy = ∂vx/∂z - ∂vz/∂x
            self.omega_y[1:-1, :, 1:-1] = (
                (self.vx[1:-1, :, 2:] - self.vx[1:-1, :, :-2]) / (2*self.dz) -
                (self.vz[2:, :, 1:-1] - self.vz[:-2, :, 1:-1]) / (2*self.dx)
            )

            # ωz = ∂vy/∂x - ∂vx/∂y
            self.omega_z[1:-1, 1:-1, :] = (
                (self.vy[2:, 1:-1, :] - self.vy[:-2, 1:-1, :]) / (2*self.dx) -
                (self.vx[1:-1, 2:, :] - self.vx[1:-1, :-2, :]) / (2*self.dy)
            )

            self.omega = np.array([self.omega_x, self.omega_y, self.omega_z])

    @property
    def magnitude(self) -> np.ndarray:
        """Vorticity magnitude."""
        if self.is_2d:
            return np.abs(self.omega)
        else:
            return np.sqrt(self.omega_x**2 + self.omega_y**2 + self.omega_z**2)

    @property
    def enstrophy(self) -> float:
        """Enstrophy: integral of ω²."""
        if self.is_2d:
            return 0.5 * np.sum(self.omega**2) * self.dx * self.dy
        else:
            return 0.5 * np.sum(self.magnitude**2) * self.dx * self.dy * self.dz

    def circulation(self, path_x: np.ndarray, path_y: np.ndarray) -> float:
        """
        Compute circulation around a closed path.

        Γ = ∮ v · dl

        Uses trapezoidal integration.
        """
        from scipy.interpolate import RegularGridInterpolator

        x = np.arange(self.vx.shape[0]) * self.dx
        y = np.arange(self.vx.shape[1]) * self.dy

        vx_interp = RegularGridInterpolator((x, y), self.vx)
        vy_interp = RegularGridInterpolator((x, y), self.vy)

        # Compute line integral
        circulation = 0.0
        for i in range(len(path_x) - 1):
            x1, y1 = path_x[i], path_y[i]
            x2, y2 = path_x[i+1], path_y[i+1]

            # Midpoint velocity
            xm, ym = 0.5*(x1+x2), 0.5*(y1+y2)
            vx_m = vx_interp([[xm, ym]])[0]
            vy_m = vy_interp([[xm, ym]])[0]

            # Path increment
            dx = x2 - x1
            dy = y2 - y1

            circulation += vx_m * dx + vy_m * dy

        return circulation


class StreamFunction(BaseClass):
    """
    Stream function for 2D incompressible flow.

    u = ∂ψ/∂y, v = -∂ψ/∂x

    Streamlines are contours of constant ψ.

    Args:
        vx, vy: Velocity components
        dx, dy: Grid spacing
    """

    def __init__(self, vx: np.ndarray, vy: np.ndarray, dx: float, dy: float):
        super().__init__()
        self.vx = np.asarray(vx)
        self.vy = np.asarray(vy)
        self.dx = dx
        self.dy = dy
        self.nx, self.ny = vx.shape

        self._compute_stream_function()

    def _compute_stream_function(self, n_iter: int = 1000):
        """
        Solve for stream function by integrating vorticity.

        ∇²ψ = -ω
        """
        # Compute vorticity
        omega = np.zeros((self.nx, self.ny))
        omega[1:-1, 1:-1] = (
            (self.vy[2:, 1:-1] - self.vy[:-2, 1:-1]) / (2*self.dx) -
            (self.vx[1:-1, 2:] - self.vx[1:-1, :-2]) / (2*self.dy)
        )

        # Solve Poisson equation using Jacobi iteration
        psi = np.zeros((self.nx, self.ny))

        for _ in range(n_iter):
            psi_new = np.zeros_like(psi)
            psi_new[1:-1, 1:-1] = 0.25 * (
                psi[2:, 1:-1] + psi[:-2, 1:-1] +
                psi[1:-1, 2:] + psi[1:-1, :-2] +
                self.dx**2 * omega[1:-1, 1:-1]
            )
            psi = psi_new

        self.psi = psi

    @property
    def values(self) -> np.ndarray:
        """Stream function values."""
        return self.psi

    def get_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        """Recover velocity from stream function."""
        vx = np.zeros_like(self.psi)
        vy = np.zeros_like(self.psi)

        vx[:, 1:-1] = (self.psi[:, 2:] - self.psi[:, :-2]) / (2*self.dy)
        vy[1:-1, :] = -(self.psi[2:, :] - self.psi[:-2, :]) / (2*self.dx)

        return vx, vy

    def mass_flux(self, psi1: float, psi2: float) -> float:
        """Mass flux between two streamlines (per unit depth)."""
        return abs(psi2 - psi1)


class VelocityPotential(BaseClass):
    """
    Velocity potential for irrotational flow.

    v = ∇φ

    For incompressible irrotational flow: ∇²φ = 0 (Laplace equation)

    Args:
        nx, ny: Grid dimensions
        dx, dy: Grid spacing
    """

    def __init__(self, nx: int, ny: int, dx: float, dy: float):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.phi = np.zeros((nx, ny))

    @classmethod
    def uniform_flow(cls, nx: int, ny: int, dx: float, dy: float,
                     U: float, angle: float = 0.0) -> 'VelocityPotential':
        """Create potential for uniform flow."""
        pot = cls(nx, ny, dx, dy)
        x = np.arange(nx) * dx
        y = np.arange(ny) * dy
        X, Y = np.meshgrid(x, y, indexing='ij')
        pot.phi = U * (X * np.cos(angle) + Y * np.sin(angle))
        return pot

    @classmethod
    def source(cls, nx: int, ny: int, dx: float, dy: float,
               strength: float, x0: float, y0: float) -> 'VelocityPotential':
        """Create potential for point source/sink."""
        pot = cls(nx, ny, dx, dy)
        x = np.arange(nx) * dx
        y = np.arange(ny) * dy
        X, Y = np.meshgrid(x, y, indexing='ij')
        r = np.sqrt((X - x0)**2 + (Y - y0)**2)
        r = np.maximum(r, 0.01 * dx)  # Avoid singularity
        pot.phi = strength / (2 * np.pi) * np.log(r)
        return pot

    @classmethod
    def vortex(cls, nx: int, ny: int, dx: float, dy: float,
               circulation: float, x0: float, y0: float) -> 'VelocityPotential':
        """Create potential for point vortex (actually stream function here)."""
        pot = cls(nx, ny, dx, dy)
        x = np.arange(nx) * dx
        y = np.arange(ny) * dy
        X, Y = np.meshgrid(x, y, indexing='ij')
        pot.phi = circulation / (2 * np.pi) * np.arctan2(Y - y0, X - x0)
        return pot

    def get_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute velocity from potential: v = ∇φ."""
        vx = np.zeros_like(self.phi)
        vy = np.zeros_like(self.phi)

        vx[1:-1, :] = (self.phi[2:, :] - self.phi[:-2, :]) / (2*self.dx)
        vy[:, 1:-1] = (self.phi[:, 2:] - self.phi[:, :-2]) / (2*self.dy)

        return vx, vy

    def __add__(self, other: 'VelocityPotential') -> 'VelocityPotential':
        """Superpose potentials (for linear combination of flows)."""
        result = VelocityPotential(self.nx, self.ny, self.dx, self.dy)
        result.phi = self.phi + other.phi
        return result


class BoundaryLayer(BaseClass):
    """
    Boundary layer analysis for flow over surfaces.

    Implements Blasius solution for laminar flat plate and
    turbulent correlations.

    Args:
        U_inf: Free stream velocity (m/s)
        nu: Kinematic viscosity (m²/s)
        x: Streamwise distance (m)
    """

    def __init__(self, U_inf: float, nu: float, x: float):
        super().__init__()
        self.U_inf = U_inf
        self.nu = nu
        self.x = x
        self.Re_x = U_inf * x / nu if x > 0 else 0

    @property
    def is_laminar(self) -> bool:
        """Check if boundary layer is likely laminar (Re_x < 5×10⁵)."""
        return self.Re_x < 5e5

    def thickness_laminar(self) -> float:
        """Laminar boundary layer thickness (Blasius): δ ≈ 5x/√(Re_x)."""
        if self.Re_x <= 0:
            return 0
        return 5 * self.x / np.sqrt(self.Re_x)

    def thickness_turbulent(self) -> float:
        """Turbulent boundary layer thickness: δ ≈ 0.37x/Re_x^0.2."""
        if self.Re_x <= 0:
            return 0
        return 0.37 * self.x / self.Re_x**0.2

    @property
    def thickness(self) -> float:
        """Boundary layer thickness based on flow regime."""
        return self.thickness_laminar() if self.is_laminar else self.thickness_turbulent()

    def displacement_thickness_laminar(self) -> float:
        """Laminar displacement thickness: δ* ≈ 1.72x/√(Re_x)."""
        if self.Re_x <= 0:
            return 0
        return 1.72 * self.x / np.sqrt(self.Re_x)

    def momentum_thickness_laminar(self) -> float:
        """Laminar momentum thickness: θ ≈ 0.664x/√(Re_x)."""
        if self.Re_x <= 0:
            return 0
        return 0.664 * self.x / np.sqrt(self.Re_x)

    def wall_shear_stress_laminar(self, rho: float) -> float:
        """Laminar wall shear stress: τ_w = 0.332ρU²/√(Re_x)."""
        if self.Re_x <= 0:
            return 0
        return 0.332 * rho * self.U_inf**2 / np.sqrt(self.Re_x)

    def skin_friction_coefficient_laminar(self) -> float:
        """Laminar skin friction coefficient: C_f = 0.664/√(Re_x)."""
        if self.Re_x <= 0:
            return 0
        return 0.664 / np.sqrt(self.Re_x)

    def skin_friction_coefficient_turbulent(self) -> float:
        """Turbulent skin friction coefficient (Schlichting): C_f ≈ 0.0592/Re_x^0.2."""
        if self.Re_x <= 0:
            return 0
        return 0.0592 / self.Re_x**0.2

    def velocity_profile_laminar(self, y: np.ndarray) -> np.ndarray:
        """
        Blasius velocity profile u/U_inf as function of η = y√(U/νx).

        Uses polynomial approximation to Blasius solution.
        """
        if self.x <= 0:
            return np.zeros_like(y)

        eta = y * np.sqrt(self.U_inf / (self.nu * self.x))

        # Polynomial approximation to Blasius f'(η)
        # f'(η) ≈ tanh(0.33η) for η < 6, 1 for η > 6
        u_ratio = np.tanh(0.33 * eta)
        u_ratio = np.minimum(u_ratio, 1.0)

        return u_ratio * self.U_inf


class TurbulenceModel(BaseClass):
    """
    RANS (Reynolds-Averaged Navier-Stokes) turbulence model.

    Implements k-ε two-equation turbulence model.

    ∂k/∂t + u·∇k = P_k - ε + ∇·[(ν + ν_t/σ_k)∇k]
    ∂ε/∂t + u·∇ε = C_1ε(ε/k)P_k - C_2ε(ε²/k) + ∇·[(ν + ν_t/σ_ε)∇ε]

    where ν_t = C_μ k²/ε

    Args:
        nx, ny: Grid dimensions
        dx, dy: Grid spacing
        nu: Molecular kinematic viscosity
    """

    # Standard k-ε model constants
    C_mu = 0.09
    C_1e = 1.44
    C_2e = 1.92
    sigma_k = 1.0
    sigma_e = 1.3

    def __init__(self, nx: int, ny: int, dx: float, dy: float, nu: float):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.nu = nu

        # Turbulent kinetic energy and dissipation rate
        self.k = np.ones((nx, ny)) * 1e-4
        self.epsilon = np.ones((nx, ny)) * 1e-5

    def set_initial_turbulence(self, intensity: float, length_scale: float):
        """
        Set initial turbulence from intensity and length scale.

        Args:
            intensity: Turbulence intensity I = u'/U
            length_scale: Turbulent length scale (m)
        """
        # k = 3/2 * (U * I)² but we need U, use default
        self.k = np.ones((self.nx, self.ny)) * (1.5 * intensity**2)
        # ε = C_μ^0.75 * k^1.5 / L
        self.epsilon = self.C_mu**0.75 * self.k**1.5 / length_scale

    @property
    def turbulent_viscosity(self) -> np.ndarray:
        """Turbulent (eddy) viscosity: ν_t = C_μ k²/ε."""
        return self.C_mu * self.k**2 / np.maximum(self.epsilon, 1e-10)

    @property
    def effective_viscosity(self) -> np.ndarray:
        """Effective viscosity: ν_eff = ν + ν_t."""
        return self.nu + self.turbulent_viscosity

    def production(self, vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
        """
        Turbulent kinetic energy production: P_k = ν_t * S²
        where S is the strain rate magnitude.
        """
        # Strain rate tensor components
        S_xx = np.zeros((self.nx, self.ny))
        S_yy = np.zeros((self.nx, self.ny))
        S_xy = np.zeros((self.nx, self.ny))

        S_xx[1:-1, :] = (vx[2:, :] - vx[:-2, :]) / (2*self.dx)
        S_yy[:, 1:-1] = (vy[:, 2:] - vy[:, :-2]) / (2*self.dy)

        S_xy[1:-1, 1:-1] = 0.5 * (
            (vx[1:-1, 2:] - vx[1:-1, :-2]) / (2*self.dy) +
            (vy[2:, 1:-1] - vy[:-2, 1:-1]) / (2*self.dx)
        )

        # S² = 2 * S_ij * S_ij
        S_squared = 2 * (S_xx**2 + S_yy**2 + 2*S_xy**2)

        return self.turbulent_viscosity * S_squared

    def step(self, vx: np.ndarray, vy: np.ndarray, dt: float):
        """Advance k and ε by one time step."""
        nu_t = self.turbulent_viscosity
        P_k = self.production(vx, vy)

        # k equation
        # Diffusion
        diff_k = np.zeros((self.nx, self.ny))
        nu_eff_k = self.nu + nu_t / self.sigma_k
        diff_k[1:-1, 1:-1] = (
            (nu_eff_k[2:, 1:-1] + nu_eff_k[1:-1, 1:-1]) * (self.k[2:, 1:-1] - self.k[1:-1, 1:-1]) -
            (nu_eff_k[1:-1, 1:-1] + nu_eff_k[:-2, 1:-1]) * (self.k[1:-1, 1:-1] - self.k[:-2, 1:-1])
        ) / (2 * self.dx**2)
        diff_k[1:-1, 1:-1] += (
            (nu_eff_k[1:-1, 2:] + nu_eff_k[1:-1, 1:-1]) * (self.k[1:-1, 2:] - self.k[1:-1, 1:-1]) -
            (nu_eff_k[1:-1, 1:-1] + nu_eff_k[1:-1, :-2]) * (self.k[1:-1, 1:-1] - self.k[1:-1, :-2])
        ) / (2 * self.dy**2)

        # Update k
        self.k += dt * (P_k - self.epsilon + diff_k)
        self.k = np.maximum(self.k, 1e-10)

        # ε equation
        diff_e = np.zeros((self.nx, self.ny))
        nu_eff_e = self.nu + nu_t / self.sigma_e
        diff_e[1:-1, 1:-1] = (
            (nu_eff_e[2:, 1:-1] + nu_eff_e[1:-1, 1:-1]) * (self.epsilon[2:, 1:-1] - self.epsilon[1:-1, 1:-1]) -
            (nu_eff_e[1:-1, 1:-1] + nu_eff_e[:-2, 1:-1]) * (self.epsilon[1:-1, 1:-1] - self.epsilon[:-2, 1:-1])
        ) / (2 * self.dx**2)
        diff_e[1:-1, 1:-1] += (
            (nu_eff_e[1:-1, 2:] + nu_eff_e[1:-1, 1:-1]) * (self.epsilon[1:-1, 2:] - self.epsilon[1:-1, 1:-1]) -
            (nu_eff_e[1:-1, 1:-1] + nu_eff_e[1:-1, :-2]) * (self.epsilon[1:-1, 1:-1] - self.epsilon[1:-1, :-2])
        ) / (2 * self.dy**2)

        eps_over_k = self.epsilon / np.maximum(self.k, 1e-10)

        self.epsilon += dt * (
            self.C_1e * eps_over_k * P_k -
            self.C_2e * self.epsilon * eps_over_k +
            diff_e
        )
        self.epsilon = np.maximum(self.epsilon, 1e-10)


class KolmogorovScale(BaseClass):
    """
    Kolmogorov microscales for turbulent flow.

    η = (ν³/ε)^(1/4)  - length scale
    τ = (ν/ε)^(1/2)   - time scale
    v = (νε)^(1/4)    - velocity scale

    Args:
        nu: Kinematic viscosity (m²/s)
        epsilon: Turbulent dissipation rate (m²/s³)
    """

    def __init__(self, nu: float, epsilon: float):
        super().__init__()
        self.nu = nu
        self.epsilon = max(epsilon, 1e-20)

    @property
    def length(self) -> float:
        """Kolmogorov length scale η = (ν³/ε)^(1/4)."""
        return (self.nu**3 / self.epsilon)**0.25

    @property
    def time(self) -> float:
        """Kolmogorov time scale τ = (ν/ε)^(1/2)."""
        return np.sqrt(self.nu / self.epsilon)

    @property
    def velocity(self) -> float:
        """Kolmogorov velocity scale v = (νε)^(1/4)."""
        return (self.nu * self.epsilon)**0.25

    @classmethod
    def from_reynolds(cls, nu: float, L: float, Re: float) -> 'KolmogorovScale':
        """
        Estimate Kolmogorov scales from Reynolds number.

        ε ~ U³/L, η ~ L * Re^(-3/4)
        """
        U = Re * nu / L
        epsilon = U**3 / L
        return cls(nu, epsilon)

    def taylor_microscale(self, u_rms: float) -> float:
        """
        Taylor microscale: λ = u_rms * √(15ν/ε)
        """
        return u_rms * np.sqrt(15 * self.nu / self.epsilon)

    def taylor_reynolds_number(self, u_rms: float) -> float:
        """Taylor-scale Reynolds number: Re_λ = u_rms * λ / ν."""
        lam = self.taylor_microscale(u_rms)
        return u_rms * lam / self.nu


class EnergySpectrum(BaseClass):
    """
    Turbulent energy spectrum analysis.

    E(k) = energy per unit wavenumber

    Implements Kolmogorov -5/3 law and von Kármán spectrum.

    Args:
        k: Wavenumber array (1/m)
        epsilon: Dissipation rate (m²/s³)
        L: Integral length scale (m)
    """

    # Kolmogorov constant
    C_K = 1.5

    def __init__(self, k: ArrayLike, epsilon: float, L: float):
        super().__init__()
        self.k = np.asarray(k)
        self.epsilon = epsilon
        self.L = L

    def kolmogorov_spectrum(self) -> np.ndarray:
        """
        Kolmogorov inertial range spectrum.

        E(k) = C_K * ε^(2/3) * k^(-5/3)
        """
        return self.C_K * self.epsilon**(2/3) * self.k**(-5/3)

    def von_karman_spectrum(self, u_rms: float) -> np.ndarray:
        """
        Von Kármán spectrum (covers all ranges).

        E(k) = A * (kL)^4 / [1 + (kL)²]^(17/6)
        """
        kL = self.k * self.L
        A = 55 * u_rms**2 * self.L / (9 * np.sqrt(np.pi))
        return A * kL**4 / (1 + kL**2)**(17/6)

    def pope_spectrum(self, nu: float) -> np.ndarray:
        """
        Pope's model spectrum (includes dissipation range).

        E(k) = C_K * ε^(2/3) * k^(-5/3) * f_L(kL) * f_η(kη)
        """
        kL = self.k * self.L
        eta = (nu**3 / self.epsilon)**0.25
        k_eta = self.k * eta

        # Low-wavenumber correction
        f_L = (kL / np.sqrt(kL**2 + 6.78))**5.2

        # High-wavenumber (dissipation) correction
        f_eta = np.exp(-5.2 * ((k_eta**4 + 0.4**4)**0.25 - 0.4))

        return self.C_K * self.epsilon**(2/3) * self.k**(-5/3) * f_L * f_eta

    def integral_scale_from_spectrum(self, E: np.ndarray) -> float:
        """Estimate integral scale from spectrum: L ~ ∫E(k)dk / (∫kE(k)dk)."""
        dk = np.gradient(self.k)
        return np.sum(E * dk) / np.sum(self.k * E * dk)

    def dissipation_from_spectrum(self, E: np.ndarray, nu: float) -> float:
        """Estimate dissipation from spectrum: ε = 2ν ∫k²E(k)dk."""
        dk = np.gradient(self.k)
        return 2 * nu * np.sum(self.k**2 * E * dk)


# Export all classes
__all__ = [
    'StressTensor',
    'StrainTensor',
    'ContinuityEquation',
    'NavierStokesSolver',
    'EulerFluidSolver',
    'ReynoldsNumber',
    'VorticityField',
    'StreamFunction',
    'VelocityPotential',
    'BoundaryLayer',
    'TurbulenceModel',
    'KolmogorovScale',
    'EnergySpectrum',
]
