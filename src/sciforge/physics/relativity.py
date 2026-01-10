"""
Relativity Module - Special and General Relativity

This module provides comprehensive tools for relativistic physics:
- Special Relativity: Lorentz transformations, 4-vectors, relativistic dynamics
- General Relativity: Metric tensors, curvature, geodesics
- Exact Solutions: Schwarzschild, Kerr, FRW metrics
- GR Phenomena: Gravitational redshift, lensing, frame dragging
- Gravitational Waves: Linearized gravity, waveforms
- Cosmology: Friedmann equations, cosmological distances
"""

import numpy as np
from typing import Union, Tuple, Optional, Callable, List, Dict
from numpy.typing import ArrayLike
from scipy.integrate import odeint, solve_ivp
from scipy.special import hyp2f1
import warnings


# =============================================================================
# Physical Constants
# =============================================================================

C = 299792458.0  # Speed of light in m/s
G = 6.67430e-11  # Gravitational constant in m^3/(kg*s^2)
HBAR = 1.054571817e-34  # Reduced Planck constant
KB = 1.380649e-23  # Boltzmann constant


# =============================================================================
# Special Relativity - Core Classes
# =============================================================================

class LorentzTransform:
    """Class for special relativity calculations using Lorentz transformations"""

    def __init__(self, velocity: Union[float, ArrayLike]):
        """
        Initialize Lorentz transformation

        Args:
            velocity: Relative velocity between reference frames (as fraction of c)
        """
        self.c = C
        self.velocity = np.array(velocity)
        self.beta = self.velocity / self.c
        self.gamma = 1 / np.sqrt(1 - np.sum(self.beta**2))

    def transform_time(self, t: float, x: ArrayLike) -> float:
        """
        Transform time between reference frames

        Args:
            t: Time in original frame
            x: Position vector in original frame

        Returns:
            Time in new frame
        """
        x = np.array(x)
        return self.gamma * (t - np.dot(self.beta, x) / self.c)

    def transform_position(self, x: ArrayLike, t: float) -> np.ndarray:
        """
        Transform position between reference frames

        Args:
            x: Position vector in original frame
            t: Time in original frame

        Returns:
            Position vector in new frame
        """
        x = np.array(x)
        return x + (self.gamma - 1) * np.dot(self.beta, x) * self.beta / np.sum(self.beta**2) \
               - self.gamma * self.velocity * t

    def proper_time(self, t: float, v: ArrayLike) -> float:
        """
        Calculate proper time for moving object

        Args:
            t: Coordinate time
            v: Velocity vector of object

        Returns:
            Proper time
        """
        v = np.array(v)
        beta = np.linalg.norm(v) / self.c
        gamma = 1 / np.sqrt(1 - beta**2)
        return t / gamma

    def length_contraction(self, length: float) -> float:
        """
        Calculate contracted length along direction of motion

        Args:
            length: Proper length in rest frame

        Returns:
            Contracted length in moving frame
        """
        return length / self.gamma

    def time_dilation(self, time: float) -> float:
        """
        Calculate dilated time

        Args:
            time: Proper time in rest frame

        Returns:
            Dilated time in moving frame
        """
        return self.gamma * time

    def relativistic_mass(self, rest_mass: float) -> float:
        """
        Calculate relativistic mass

        Args:
            rest_mass: Mass in rest frame

        Returns:
            Relativistic mass
        """
        return self.gamma * rest_mass

    def relativistic_momentum(self, mass: float, velocity: ArrayLike) -> np.ndarray:
        """
        Calculate relativistic momentum

        Args:
            mass: Rest mass
            velocity: Velocity vector

        Returns:
            Relativistic momentum vector
        """
        velocity = np.array(velocity)
        beta = np.linalg.norm(velocity) / self.c
        gamma = 1 / np.sqrt(1 - beta**2)
        return mass * gamma * velocity

    def relativistic_energy(self, mass: float) -> float:
        """
        Calculate total relativistic energy

        Args:
            mass: Rest mass

        Returns:
            Total energy (including rest energy)
        """
        return self.gamma * mass * self.c**2


class MinkowskiSpacetime:
    """Class for handling 4D spacetime calculations"""

    def __init__(self):
        """Initialize Minkowski spacetime"""
        self.c = C
        self.metric = np.diag([1, -1, -1, -1])  # Metric tensor (signature +---)

    def interval(self, event1: ArrayLike, event2: ArrayLike) -> float:
        """
        Calculate spacetime interval between two events

        Args:
            event1: First event coordinates (t, x, y, z)
            event2: Second event coordinates (t, x, y, z)

        Returns:
            Spacetime interval
        """
        event1, event2 = np.array(event1), np.array(event2)
        delta = event1 - event2
        delta[0] *= self.c  # Convert time component
        return np.sqrt(np.dot(np.dot(delta, self.metric), delta))

    def proper_time(self, worldline: ArrayLike) -> float:
        """
        Calculate proper time along a worldline

        Args:
            worldline: Array of 4D spacetime points

        Returns:
            Total proper time along worldline
        """
        total_time = 0
        points = np.array(worldline)
        for i in range(len(points)-1):
            total_time += self.interval(points[i], points[i+1])
        return total_time / self.c


class RelativisticParticle:
    """Class representing a relativistic particle"""

    def __init__(self, mass: float, position: ArrayLike, velocity: ArrayLike):
        """
        Initialize relativistic particle

        Args:
            mass: Rest mass
            position: Initial position vector
            velocity: Initial velocity vector
        """
        self.c = C
        self.mass = mass
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.proper_time = 0
        self.history = {'position': [self.position.copy()],
                       'velocity': [self.velocity.copy()],
                       'proper_time': [0]}

    def update(self, force: ArrayLike, dt: float):
        """
        Update particle state under relativistic force

        Args:
            force: Applied force vector
            dt: Time step
        """
        beta = np.linalg.norm(self.velocity) / self.c
        gamma = 1 / np.sqrt(1 - beta**2)

        # Relativistic acceleration
        acceleration = force / (gamma**3 * self.mass)

        # Update velocity and position
        self.velocity += acceleration * dt
        self.position += self.velocity * dt

        # Update proper time
        self.proper_time += dt / gamma

        # Store history
        self.history['position'].append(self.position.copy())
        self.history['velocity'].append(self.velocity.copy())
        self.history['proper_time'].append(self.proper_time)


# =============================================================================
# Special Relativity Extensions - 4-Vectors
# =============================================================================

class FourVector:
    """Spacetime 4-vector with Lorentz transformation support"""

    def __init__(self, components: ArrayLike, signature: str = '+---'):
        """
        Initialize 4-vector

        Args:
            components: (t, x, y, z) or (E/c, px, py, pz) components
            signature: Metric signature, '+---' or '-+++'
        """
        self.components = np.array(components, dtype=float)
        if len(self.components) != 4:
            raise ValueError("4-vector must have exactly 4 components")
        self.signature = signature
        if signature == '+---':
            self.metric = np.diag([1, -1, -1, -1])
        else:
            self.metric = np.diag([-1, 1, 1, 1])

    @property
    def temporal(self) -> float:
        """Return temporal (0th) component"""
        return self.components[0]

    @property
    def spatial(self) -> np.ndarray:
        """Return spatial (1,2,3) components"""
        return self.components[1:]

    def norm_squared(self) -> float:
        """Calculate Lorentz-invariant norm squared"""
        return np.dot(self.components, np.dot(self.metric, self.components))

    def norm(self) -> float:
        """Calculate Lorentz-invariant norm (magnitude)"""
        n2 = self.norm_squared()
        return np.sqrt(abs(n2)) * np.sign(n2)

    def is_timelike(self) -> bool:
        """Check if 4-vector is timelike"""
        if self.signature == '+---':
            return self.norm_squared() > 0
        return self.norm_squared() < 0

    def is_spacelike(self) -> bool:
        """Check if 4-vector is spacelike"""
        if self.signature == '+---':
            return self.norm_squared() < 0
        return self.norm_squared() > 0

    def is_lightlike(self) -> bool:
        """Check if 4-vector is lightlike (null)"""
        return np.abs(self.norm_squared()) < 1e-10

    def boost(self, velocity: ArrayLike) -> 'FourVector':
        """
        Apply Lorentz boost to 4-vector

        Args:
            velocity: Boost velocity (3-vector, as fraction of c)

        Returns:
            Boosted 4-vector
        """
        v = np.array(velocity)
        beta = np.linalg.norm(v)
        if beta < 1e-10:
            return FourVector(self.components.copy(), self.signature)

        gamma = 1 / np.sqrt(1 - beta**2)
        n = v / beta  # Direction unit vector

        # Build boost matrix
        Lambda = np.eye(4)
        Lambda[0, 0] = gamma
        Lambda[0, 1:] = -gamma * beta * n
        Lambda[1:, 0] = -gamma * beta * n
        Lambda[1:, 1:] = np.eye(3) + (gamma - 1) * np.outer(n, n)

        new_components = Lambda @ self.components
        return FourVector(new_components, self.signature)

    def dot(self, other: 'FourVector') -> float:
        """Calculate Minkowski inner product with another 4-vector"""
        return np.dot(self.components, np.dot(self.metric, other.components))

    def __add__(self, other: 'FourVector') -> 'FourVector':
        return FourVector(self.components + other.components, self.signature)

    def __sub__(self, other: 'FourVector') -> 'FourVector':
        return FourVector(self.components - other.components, self.signature)

    def __mul__(self, scalar: float) -> 'FourVector':
        return FourVector(scalar * self.components, self.signature)

    def __rmul__(self, scalar: float) -> 'FourVector':
        return self.__mul__(scalar)


class FourMomentum(FourVector):
    """Energy-momentum 4-vector p^μ = (E/c, px, py, pz)"""

    def __init__(self, energy: float, momentum: ArrayLike, c: float = C):
        """
        Initialize 4-momentum

        Args:
            energy: Total energy
            momentum: 3-momentum vector
            c: Speed of light
        """
        self.c = c
        momentum = np.array(momentum)
        super().__init__([energy/c, momentum[0], momentum[1], momentum[2]])
        self._energy = energy
        self._momentum = momentum

    @classmethod
    def from_mass_velocity(cls, mass: float, velocity: ArrayLike, c: float = C) -> 'FourMomentum':
        """Create 4-momentum from rest mass and velocity"""
        v = np.array(velocity)
        beta = np.linalg.norm(v) / c
        if beta >= 1:
            raise ValueError("Velocity must be less than speed of light")
        gamma = 1 / np.sqrt(1 - beta**2)
        energy = gamma * mass * c**2
        momentum = gamma * mass * v
        return cls(energy, momentum, c)

    @property
    def energy(self) -> float:
        """Get energy component"""
        return self.components[0] * self.c

    @property
    def momentum(self) -> np.ndarray:
        """Get 3-momentum components"""
        return self.components[1:]

    def invariant_mass(self) -> float:
        """Calculate invariant (rest) mass from p^2 = m^2 c^2"""
        p2 = self.norm_squared()
        return np.sqrt(abs(p2)) * self.c

    def velocity(self) -> np.ndarray:
        """Calculate 3-velocity from momentum"""
        E = self.energy
        p = self.momentum
        if E == 0:
            return np.zeros(3)
        return p * self.c**2 / E


class FourVelocity(FourVector):
    """Proper velocity 4-vector u^μ = γ(c, vx, vy, vz)"""

    def __init__(self, velocity: ArrayLike, c: float = C):
        """
        Initialize 4-velocity from 3-velocity

        Args:
            velocity: 3-velocity vector
            c: Speed of light
        """
        self.c = c
        v = np.array(velocity)
        beta = np.linalg.norm(v) / c
        if beta >= 1:
            raise ValueError("Velocity must be less than speed of light")
        gamma = 1 / np.sqrt(1 - beta**2)
        super().__init__([gamma * c, gamma * v[0], gamma * v[1], gamma * v[2]])
        self._gamma = gamma

    @property
    def gamma(self) -> float:
        """Get Lorentz factor"""
        return self.components[0] / self.c

    def three_velocity(self) -> np.ndarray:
        """Get 3-velocity from 4-velocity"""
        return self.spatial / self.gamma


class FourForce(FourVector):
    """Relativistic 4-force K^μ = γ(F·v/c, Fx, Fy, Fz)"""

    def __init__(self, force: ArrayLike, velocity: ArrayLike, c: float = C):
        """
        Initialize 4-force from 3-force and velocity

        Args:
            force: 3-force vector
            velocity: 3-velocity of particle
            c: Speed of light
        """
        self.c = c
        f = np.array(force)
        v = np.array(velocity)
        beta = np.linalg.norm(v) / c
        gamma = 1 / np.sqrt(1 - beta**2)
        power = np.dot(f, v)  # F·v
        super().__init__([gamma * power / c, gamma * f[0], gamma * f[1], gamma * f[2]])

    def power(self) -> float:
        """Get power (rate of energy transfer)"""
        return self.components[0] * self.c


class ElectromagneticFieldTensor:
    """Electromagnetic field tensor F^μν in special relativity"""

    def __init__(self, E: ArrayLike, B: ArrayLike, c: float = C):
        """
        Initialize EM field tensor from E and B fields

        Args:
            E: Electric field 3-vector
            B: Magnetic field 3-vector
            c: Speed of light
        """
        self.c = c
        self.E = np.array(E)
        self.B = np.array(B)

        # Build antisymmetric F^μν tensor
        self.tensor = np.zeros((4, 4))
        # F^0i = -E^i/c (with +--- signature)
        self.tensor[0, 1:] = -self.E / c
        self.tensor[1:, 0] = self.E / c
        # F^ij = -ε_ijk B^k
        self.tensor[1, 2] = -self.B[2]
        self.tensor[1, 3] = self.B[1]
        self.tensor[2, 1] = self.B[2]
        self.tensor[2, 3] = -self.B[0]
        self.tensor[3, 1] = -self.B[1]
        self.tensor[3, 2] = self.B[0]

    def dual_tensor(self) -> np.ndarray:
        """Calculate dual field tensor *F^μν = (1/2)ε^μνρσ F_ρσ"""
        # For EM: swap E <-> B with sign
        dual = np.zeros((4, 4))
        dual[0, 1:] = -self.B
        dual[1:, 0] = self.B
        dual[1, 2] = self.E[2] / self.c
        dual[1, 3] = -self.E[1] / self.c
        dual[2, 1] = -self.E[2] / self.c
        dual[2, 3] = self.E[0] / self.c
        dual[3, 1] = self.E[1] / self.c
        dual[3, 2] = -self.E[0] / self.c
        return dual

    def invariant_1(self) -> float:
        """First Lorentz invariant: (1/2)F_μν F^μν = B² - E²/c²"""
        return np.sum(self.B**2) - np.sum(self.E**2) / self.c**2

    def invariant_2(self) -> float:
        """Second Lorentz invariant: (1/4)F_μν *F^μν = E·B/c"""
        return np.dot(self.E, self.B) / self.c

    def boost(self, velocity: ArrayLike) -> 'ElectromagneticFieldTensor':
        """
        Lorentz boost the EM field

        Args:
            velocity: Boost velocity (3-vector)

        Returns:
            Boosted EM field tensor
        """
        v = np.array(velocity)
        beta = v / self.c
        beta_mag = np.linalg.norm(beta)
        if beta_mag < 1e-10:
            return ElectromagneticFieldTensor(self.E.copy(), self.B.copy(), self.c)

        gamma = 1 / np.sqrt(1 - beta_mag**2)
        n = beta / beta_mag  # Direction

        # Parallel and perpendicular components
        E_par = np.dot(self.E, n) * n
        E_perp = self.E - E_par
        B_par = np.dot(self.B, n) * n
        B_perp = self.B - B_par

        # Boosted fields
        E_new = E_par + gamma * (E_perp + np.cross(v, self.B))
        B_new = B_par + gamma * (B_perp - np.cross(v, self.E) / self.c**2)

        return ElectromagneticFieldTensor(E_new, B_new, self.c)

    def lorentz_force(self, charge: float, four_velocity: FourVelocity) -> FourForce:
        """
        Calculate Lorentz 4-force on a charged particle

        Args:
            charge: Particle charge
            four_velocity: Particle 4-velocity

        Returns:
            Lorentz 4-force
        """
        # K^μ = q F^μν u_ν
        metric = np.diag([1, -1, -1, -1])
        u_lower = metric @ four_velocity.components
        K = charge * self.tensor @ u_lower
        # Convert to FourForce (need to extract 3-force and velocity)
        v = four_velocity.three_velocity()
        gamma = four_velocity.gamma
        F = K[1:] / gamma  # Approximate extraction
        return FourForce(F, v, self.c)


class StressEnergyTensor:
    """Stress-energy tensor T^μν for matter/energy content"""

    def __init__(self, tensor: Optional[ArrayLike] = None):
        """
        Initialize stress-energy tensor

        Args:
            tensor: 4x4 array for T^μν, or None for vacuum
        """
        if tensor is None:
            self.tensor = np.zeros((4, 4))
        else:
            self.tensor = np.array(tensor)
            if self.tensor.shape != (4, 4):
                raise ValueError("Stress-energy tensor must be 4x4")

    @classmethod
    def perfect_fluid(cls, rho: float, p: float,
                      velocity: ArrayLike = None, c: float = C) -> 'StressEnergyTensor':
        """
        Create stress-energy tensor for perfect fluid

        Args:
            rho: Energy density
            p: Pressure
            velocity: 4-velocity of fluid (default: at rest)
            c: Speed of light

        Returns:
            Perfect fluid stress-energy tensor
        """
        if velocity is None:
            u = np.array([1, 0, 0, 0])  # At rest
        else:
            u = np.array(velocity)
            u = u / np.sqrt(abs(u[0]**2 - np.sum(u[1:]**2)))  # Normalize

        metric = np.diag([1, -1, -1, -1])
        # T^μν = (ρ + p/c²)u^μ u^ν - p η^μν
        T = (rho + p/c**2) * np.outer(u, u) - p * np.linalg.inv(metric)
        return cls(T)

    @classmethod
    def dust(cls, rho: float, velocity: ArrayLike = None, c: float = C) -> 'StressEnergyTensor':
        """Create stress-energy tensor for pressureless dust (p=0)"""
        return cls.perfect_fluid(rho, 0.0, velocity, c)

    @classmethod
    def electromagnetic(cls, E: ArrayLike, B: ArrayLike,
                       epsilon_0: float = 8.854e-12, c: float = C) -> 'StressEnergyTensor':
        """
        Create stress-energy tensor for electromagnetic field

        Args:
            E: Electric field 3-vector
            B: Magnetic field 3-vector
            epsilon_0: Permittivity of free space
            c: Speed of light
        """
        E = np.array(E)
        B = np.array(B)
        mu_0 = 1 / (epsilon_0 * c**2)

        # Energy density
        u = 0.5 * (epsilon_0 * np.sum(E**2) + np.sum(B**2) / mu_0)

        # Poynting vector (momentum density * c²)
        S = np.cross(E, B) / mu_0

        # Maxwell stress tensor
        sigma = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                sigma[i,j] = epsilon_0 * (E[i]*E[j] - 0.5*(i==j)*np.sum(E**2))
                sigma[i,j] += (B[i]*B[j] - 0.5*(i==j)*np.sum(B**2)) / mu_0

        # Build T^μν
        T = np.zeros((4, 4))
        T[0, 0] = u
        T[0, 1:] = S / c**2
        T[1:, 0] = S / c**2
        T[1:, 1:] = sigma

        return cls(T)

    def energy_density(self) -> float:
        """Get energy density T^00"""
        return self.tensor[0, 0]

    def momentum_density(self) -> np.ndarray:
        """Get momentum density T^0i"""
        return self.tensor[0, 1:]

    def stress_tensor(self) -> np.ndarray:
        """Get spatial stress tensor T^ij"""
        return self.tensor[1:, 1:]

    def trace(self, metric: Optional[np.ndarray] = None) -> float:
        """Calculate trace T = g_μν T^μν"""
        if metric is None:
            metric = np.diag([1, -1, -1, -1])
        return np.trace(metric @ self.tensor)


class CovariantMaxwell:
    """Maxwell's equations in covariant form"""

    def __init__(self, c: float = C, epsilon_0: float = 8.854e-12):
        """
        Initialize covariant Maxwell equations

        Args:
            c: Speed of light
            epsilon_0: Permittivity of free space
        """
        self.c = c
        self.epsilon_0 = epsilon_0
        self.mu_0 = 1 / (epsilon_0 * c**2)

    def field_equation(self, F: np.ndarray, J: np.ndarray) -> np.ndarray:
        """
        Check inhomogeneous Maxwell equation: ∂_ν F^μν = μ₀ J^μ

        Args:
            F: Field tensor F^μν (4x4)
            J: Current 4-vector (4,)

        Returns:
            Residual (should be zero for valid solution)
        """
        # This is a symbolic check - actual implementation needs spacetime grid
        return self.mu_0 * np.array(J)

    def bianchi_identity(self, F: np.ndarray) -> np.ndarray:
        """
        Check Bianchi identity: ∂_[μ F_νρ] = 0

        This is automatically satisfied for F derived from potential A
        """
        return np.zeros((4, 4, 4))

    def from_potential(self, A: ArrayLike, dx: ArrayLike) -> np.ndarray:
        """
        Calculate F^μν from 4-potential A_μ

        F_μν = ∂_μ A_ν - ∂_ν A_μ

        Args:
            A: 4-potential field values on grid
            dx: Grid spacing (dt, dx, dy, dz)

        Returns:
            Field tensor F_μν
        """
        A = np.array(A)
        dx = np.array(dx)

        # Numerical derivatives using finite differences
        F = np.zeros((4, 4) + A.shape[1:])
        for mu in range(4):
            for nu in range(4):
                # ∂_μ A_ν - ∂_ν A_μ
                dA_mu_nu = np.gradient(A[nu], dx[mu], axis=mu)
                dA_nu_mu = np.gradient(A[mu], dx[nu], axis=nu)
                F[mu, nu] = dA_mu_nu - dA_nu_mu

        return F


# =============================================================================
# General Relativity Foundations
# =============================================================================

class MetricTensor:
    """Metric tensor g_μν for curved spacetime"""

    def __init__(self, metric: ArrayLike):
        """
        Initialize metric tensor

        Args:
            metric: 4x4 array for g_μν
        """
        self.metric = np.array(metric, dtype=float)
        if self.metric.shape != (4, 4):
            raise ValueError("Metric must be 4x4")

        # Compute inverse metric
        self.inverse = np.linalg.inv(self.metric)

        # Compute determinant
        self.determinant = np.linalg.det(self.metric)

    @classmethod
    def minkowski(cls, signature: str = '+---') -> 'MetricTensor':
        """Create Minkowski (flat spacetime) metric"""
        if signature == '+---':
            return cls(np.diag([1, -1, -1, -1]))
        else:
            return cls(np.diag([-1, 1, 1, 1]))

    @classmethod
    def spherical(cls, r: float, theta: float) -> 'MetricTensor':
        """Create Minkowski metric in spherical coordinates (t, r, θ, φ)"""
        return cls(np.diag([1, -1, -r**2, -r**2 * np.sin(theta)**2]))

    def line_element(self, dx: ArrayLike) -> float:
        """
        Calculate line element ds² = g_μν dx^μ dx^ν

        Args:
            dx: Coordinate differential 4-vector

        Returns:
            ds² value
        """
        dx = np.array(dx)
        return np.dot(dx, np.dot(self.metric, dx))

    def raise_index(self, vector: ArrayLike) -> np.ndarray:
        """Raise index of covariant vector: v^μ = g^μν v_ν"""
        return self.inverse @ np.array(vector)

    def lower_index(self, vector: ArrayLike) -> np.ndarray:
        """Lower index of contravariant vector: v_μ = g_μν v^ν"""
        return self.metric @ np.array(vector)

    def proper_distance(self, path: ArrayLike, dt: float = 0.01) -> float:
        """
        Calculate proper distance along a spacelike path

        Args:
            path: Array of 4D coordinates along path
            dt: Parameter step

        Returns:
            Proper distance
        """
        path = np.array(path)
        total = 0.0
        for i in range(len(path) - 1):
            dx = path[i+1] - path[i]
            ds2 = self.line_element(dx)
            if ds2 < 0:  # Spacelike
                total += np.sqrt(-ds2)
        return total


class ChristoffelSymbols:
    """Christoffel symbols Γ^ρ_μν for a given metric"""

    def __init__(self, metric_func: Callable, coords: ArrayLike, h: float = 1e-6):
        """
        Initialize Christoffel symbols

        Args:
            metric_func: Function returning metric tensor at given coordinates
            coords: Current coordinate values
            h: Step size for numerical derivatives
        """
        self.metric_func = metric_func
        self.coords = np.array(coords)
        self.h = h
        self._compute_symbols()

    def _compute_symbols(self):
        """Compute Christoffel symbols numerically"""
        g = self.metric_func(self.coords)
        g_inv = np.linalg.inv(g)

        # Compute metric derivatives
        dg = np.zeros((4, 4, 4))  # dg[μ,ν,σ] = ∂_σ g_μν
        for sigma in range(4):
            x_plus = self.coords.copy()
            x_minus = self.coords.copy()
            x_plus[sigma] += self.h
            x_minus[sigma] -= self.h
            dg[:, :, sigma] = (self.metric_func(x_plus) - self.metric_func(x_minus)) / (2 * self.h)

        # Christoffel symbols: Γ^ρ_μν = (1/2)g^ρσ(∂_μ g_νσ + ∂_ν g_μσ - ∂_σ g_μν)
        self.symbols = np.zeros((4, 4, 4))  # Γ^ρ_μν
        for rho in range(4):
            for mu in range(4):
                for nu in range(4):
                    for sigma in range(4):
                        self.symbols[rho, mu, nu] += 0.5 * g_inv[rho, sigma] * (
                            dg[nu, sigma, mu] + dg[mu, sigma, nu] - dg[mu, nu, sigma]
                        )

    def __getitem__(self, indices: Tuple[int, int, int]) -> float:
        """Get Γ^ρ_μν value"""
        return self.symbols[indices]

    def geodesic_acceleration(self, velocity: ArrayLike) -> np.ndarray:
        """
        Calculate geodesic acceleration: d²x^ρ/dτ² = -Γ^ρ_μν (dx^μ/dτ)(dx^ν/dτ)

        Args:
            velocity: 4-velocity dx^μ/dτ

        Returns:
            Coordinate acceleration
        """
        v = np.array(velocity)
        acc = np.zeros(4)
        for rho in range(4):
            for mu in range(4):
                for nu in range(4):
                    acc[rho] -= self.symbols[rho, mu, nu] * v[mu] * v[nu]
        return acc


class RiemannTensor:
    """Riemann curvature tensor R^ρ_σμν"""

    def __init__(self, metric_func: Callable, coords: ArrayLike, h: float = 1e-5):
        """
        Initialize Riemann tensor

        Args:
            metric_func: Function returning metric tensor at given coordinates
            coords: Current coordinate values
            h: Step size for numerical derivatives
        """
        self.metric_func = metric_func
        self.coords = np.array(coords)
        self.h = h
        self._compute_tensor()

    def _compute_tensor(self):
        """Compute Riemann tensor components"""
        # Get Christoffel symbols at current point and nearby points
        Gamma = ChristoffelSymbols(self.metric_func, self.coords, self.h).symbols

        # Compute derivatives of Christoffel symbols
        dGamma = np.zeros((4, 4, 4, 4))  # dGamma[ρ,σ,μ,ν] = ∂_ν Γ^ρ_σμ
        for nu in range(4):
            x_plus = self.coords.copy()
            x_minus = self.coords.copy()
            x_plus[nu] += self.h
            x_minus[nu] -= self.h
            Gamma_plus = ChristoffelSymbols(self.metric_func, x_plus, self.h).symbols
            Gamma_minus = ChristoffelSymbols(self.metric_func, x_minus, self.h).symbols
            dGamma[:, :, :, nu] = (Gamma_plus - Gamma_minus) / (2 * self.h)

        # Riemann tensor: R^ρ_σμν = ∂_μ Γ^ρ_νσ - ∂_ν Γ^ρ_μσ + Γ^ρ_μλ Γ^λ_νσ - Γ^ρ_νλ Γ^λ_μσ
        self.tensor = np.zeros((4, 4, 4, 4))
        for rho in range(4):
            for sigma in range(4):
                for mu in range(4):
                    for nu in range(4):
                        self.tensor[rho, sigma, mu, nu] = (
                            dGamma[rho, sigma, nu, mu] - dGamma[rho, sigma, mu, nu]
                        )
                        for lam in range(4):
                            self.tensor[rho, sigma, mu, nu] += (
                                Gamma[rho, mu, lam] * Gamma[lam, nu, sigma] -
                                Gamma[rho, nu, lam] * Gamma[lam, mu, sigma]
                            )

    def __getitem__(self, indices: Tuple[int, int, int, int]) -> float:
        """Get R^ρ_σμν value"""
        return self.tensor[indices]

    def kretschmann_scalar(self) -> float:
        """Calculate Kretschmann scalar K = R_αβγδ R^αβγδ"""
        g = self.metric_func(self.coords)
        g_inv = np.linalg.inv(g)

        # Lower all indices
        R_lower = np.zeros((4, 4, 4, 4))
        for rho in range(4):
            for sigma in range(4):
                for mu in range(4):
                    for nu in range(4):
                        for alpha in range(4):
                            R_lower[rho, sigma, mu, nu] += (
                                g[rho, alpha] * self.tensor[alpha, sigma, mu, nu]
                            )

        # Contract with raised tensor
        K = 0.0
        for a in range(4):
            for b in range(4):
                for c in range(4):
                    for d in range(4):
                        R_raised = 0.0
                        for rho in range(4):
                            for sigma in range(4):
                                for mu in range(4):
                                    for nu in range(4):
                                        R_raised += (
                                            g_inv[a, rho] * g_inv[b, sigma] *
                                            g_inv[c, mu] * g_inv[d, nu] *
                                            R_lower[rho, sigma, mu, nu]
                                        )
                        K += R_lower[a, b, c, d] * R_raised
        return K


class RicciTensor:
    """Ricci tensor R_μν = R^ρ_μρν"""

    def __init__(self, riemann: RiemannTensor):
        """
        Initialize Ricci tensor from Riemann tensor

        Args:
            riemann: Riemann curvature tensor
        """
        self.riemann = riemann
        self.tensor = np.zeros((4, 4))

        # Contract Riemann tensor: R_μν = R^ρ_μρν
        for mu in range(4):
            for nu in range(4):
                for rho in range(4):
                    self.tensor[mu, nu] += riemann.tensor[rho, mu, rho, nu]

    def __getitem__(self, indices: Tuple[int, int]) -> float:
        """Get R_μν value"""
        return self.tensor[indices]


class RicciScalar:
    """Ricci scalar R = g^μν R_μν"""

    def __init__(self, ricci: RicciTensor, metric: MetricTensor):
        """
        Initialize Ricci scalar

        Args:
            ricci: Ricci tensor
            metric: Metric tensor
        """
        self.ricci = ricci
        self.metric = metric

        # Contract: R = g^μν R_μν
        self.value = 0.0
        for mu in range(4):
            for nu in range(4):
                self.value += metric.inverse[mu, nu] * ricci.tensor[mu, nu]

    def __float__(self) -> float:
        return self.value


class EinsteinTensor:
    """Einstein tensor G_μν = R_μν - (1/2)g_μν R"""

    def __init__(self, ricci: RicciTensor, ricci_scalar: RicciScalar, metric: MetricTensor):
        """
        Initialize Einstein tensor

        Args:
            ricci: Ricci tensor
            ricci_scalar: Ricci scalar
            metric: Metric tensor
        """
        R = float(ricci_scalar)
        self.tensor = ricci.tensor - 0.5 * metric.metric * R

    def __getitem__(self, indices: Tuple[int, int]) -> float:
        """Get G_μν value"""
        return self.tensor[indices]

    def stress_energy_from_einstein(self, c: float = C, G: float = G) -> np.ndarray:
        """
        Get stress-energy tensor from Einstein equation: G_μν = (8πG/c⁴) T_μν

        Returns:
            T_μν stress-energy tensor
        """
        return self.tensor * c**4 / (8 * np.pi * G)


class GeodesicEquation:
    """Solver for geodesic equation d²x^μ/dτ² + Γ^μ_νρ (dx^ν/dτ)(dx^ρ/dτ) = 0"""

    def __init__(self, metric_func: Callable, initial_position: ArrayLike,
                 initial_velocity: ArrayLike):
        """
        Initialize geodesic solver

        Args:
            metric_func: Function returning metric tensor at given coordinates
            initial_position: Initial 4-position
            initial_velocity: Initial 4-velocity
        """
        self.metric_func = metric_func
        self.position = np.array(initial_position, dtype=float)
        self.velocity = np.array(initial_velocity, dtype=float)
        self.tau = 0.0
        self.history = {
            'tau': [0.0],
            'position': [self.position.copy()],
            'velocity': [self.velocity.copy()]
        }

    def _geodesic_rhs(self, y: np.ndarray, tau: float) -> np.ndarray:
        """RHS of geodesic ODE system"""
        x = y[:4]
        v = y[4:]

        # Get Christoffel symbols
        christoffel = ChristoffelSymbols(self.metric_func, x)

        # Geodesic equation
        dx_dtau = v
        dv_dtau = christoffel.geodesic_acceleration(v)

        return np.concatenate([dx_dtau, dv_dtau])

    def integrate(self, delta_tau: float, n_steps: int = 100) -> None:
        """
        Integrate geodesic equation

        Args:
            delta_tau: Proper time to integrate
            n_steps: Number of integration steps
        """
        tau_span = np.linspace(0, delta_tau, n_steps)
        y0 = np.concatenate([self.position, self.velocity])

        solution = odeint(self._geodesic_rhs, y0, tau_span)

        # Update state
        self.position = solution[-1, :4]
        self.velocity = solution[-1, 4:]
        self.tau += delta_tau

        # Store history
        for i in range(1, len(tau_span)):
            self.history['tau'].append(self.tau - delta_tau + tau_span[i])
            self.history['position'].append(solution[i, :4])
            self.history['velocity'].append(solution[i, 4:])

    def is_timelike(self) -> bool:
        """Check if geodesic is timelike (massive particle)"""
        g = self.metric_func(self.position)
        v2 = np.dot(self.velocity, np.dot(g, self.velocity))
        return v2 > 0

    def is_null(self, tol: float = 1e-6) -> bool:
        """Check if geodesic is null (photon)"""
        g = self.metric_func(self.position)
        v2 = np.dot(self.velocity, np.dot(g, self.velocity))
        return abs(v2) < tol


# =============================================================================
# Exact Solutions
# =============================================================================

class SchwarzschildMetric:
    """Schwarzschild metric for static spherically symmetric spacetime"""

    def __init__(self, M: float, c: float = C, G: float = G):
        """
        Initialize Schwarzschild metric

        Args:
            M: Central mass
            c: Speed of light
            G: Gravitational constant
        """
        self.M = M
        self.c = c
        self.G = G
        self.rs = 2 * G * M / c**2  # Schwarzschild radius

    def metric(self, coords: ArrayLike) -> np.ndarray:
        """
        Get metric tensor at coordinates (t, r, θ, φ)

        Args:
            coords: (t, r, θ, φ) coordinates

        Returns:
            4x4 metric tensor g_μν
        """
        t, r, theta, phi = coords

        if r <= self.rs:
            warnings.warn(f"Inside event horizon (r={r} <= rs={self.rs})")

        f = 1 - self.rs / r

        g = np.zeros((4, 4))
        g[0, 0] = f * self.c**2
        g[1, 1] = -1 / f
        g[2, 2] = -r**2
        g[3, 3] = -r**2 * np.sin(theta)**2

        return g

    def event_horizon(self) -> float:
        """Return Schwarzschild radius (event horizon)"""
        return self.rs

    def isco_radius(self) -> float:
        """Return innermost stable circular orbit (ISCO) radius"""
        return 3 * self.rs

    def photon_sphere_radius(self) -> float:
        """Return photon sphere radius"""
        return 1.5 * self.rs

    def surface_gravity(self) -> float:
        """Return surface gravity at event horizon"""
        return self.c**4 / (4 * self.G * self.M)

    def hawking_temperature(self) -> float:
        """Return Hawking temperature"""
        return HBAR * self.c**3 / (8 * np.pi * self.G * self.M * KB)

    def proper_distance(self, r1: float, r2: float) -> float:
        """
        Calculate proper radial distance between two r coordinates

        Args:
            r1: Inner radial coordinate
            r2: Outer radial coordinate

        Returns:
            Proper distance
        """
        from scipy.integrate import quad

        def integrand(r):
            return 1 / np.sqrt(1 - self.rs / r)

        result, _ = quad(integrand, r1, r2)
        return result

    def time_dilation(self, r: float) -> float:
        """
        Calculate gravitational time dilation factor at radius r

        Returns:
            Factor by which proper time runs slower than coordinate time
        """
        return np.sqrt(1 - self.rs / r)

    def orbital_period(self, r: float) -> float:
        """Calculate orbital period for circular orbit at radius r"""
        return 2 * np.pi * np.sqrt(r**3 / (self.G * self.M))

    def geodesic_solver(self, initial_position: ArrayLike,
                        initial_velocity: ArrayLike) -> GeodesicEquation:
        """Create geodesic solver for this metric"""
        return GeodesicEquation(self.metric, initial_position, initial_velocity)


class KerrMetric:
    """Kerr metric for rotating black holes (Boyer-Lindquist coordinates)"""

    def __init__(self, M: float, a: float, c: float = C, G: float = G):
        """
        Initialize Kerr metric

        Args:
            M: Black hole mass
            a: Specific angular momentum J/(Mc) (0 ≤ a ≤ GM/c²)
            c: Speed of light
            G: Gravitational constant
        """
        self.M = M
        self.a = a
        self.c = c
        self.G = G
        self.rs = 2 * G * M / c**2

        # Check extremal bound
        a_max = G * M / c**2
        if abs(a) > a_max:
            raise ValueError(f"Spin parameter |a| = {abs(a)} exceeds extremal limit {a_max}")

    def _sigma(self, r: float, theta: float) -> float:
        """Σ = r² + a²cos²θ"""
        return r**2 + self.a**2 * np.cos(theta)**2

    def _delta(self, r: float) -> float:
        """Δ = r² - rs*r + a²"""
        return r**2 - self.rs * r + self.a**2

    def metric(self, coords: ArrayLike) -> np.ndarray:
        """
        Get metric tensor at coordinates (t, r, θ, φ)

        Args:
            coords: (t, r, θ, φ) Boyer-Lindquist coordinates

        Returns:
            4x4 metric tensor g_μν
        """
        t, r, theta, phi = coords

        Sigma = self._sigma(r, theta)
        Delta = self._delta(r)
        sin2 = np.sin(theta)**2

        A = (r**2 + self.a**2)**2 - Delta * self.a**2 * sin2

        g = np.zeros((4, 4))
        g[0, 0] = (Delta - self.a**2 * sin2) * self.c**2 / Sigma
        g[0, 3] = self.rs * r * self.a * self.c * sin2 / Sigma
        g[3, 0] = g[0, 3]
        g[1, 1] = -Sigma / Delta
        g[2, 2] = -Sigma
        g[3, 3] = -A * sin2 / Sigma

        return g

    def outer_horizon(self) -> float:
        """Return outer event horizon radius r+"""
        return 0.5 * self.rs + np.sqrt((0.5 * self.rs)**2 - self.a**2)

    def inner_horizon(self) -> float:
        """Return inner (Cauchy) horizon radius r-"""
        return 0.5 * self.rs - np.sqrt((0.5 * self.rs)**2 - self.a**2)

    def ergosphere_radius(self, theta: float) -> float:
        """Return ergosphere outer boundary radius at angle θ"""
        return 0.5 * self.rs + np.sqrt((0.5 * self.rs)**2 - self.a**2 * np.cos(theta)**2)

    def angular_velocity_horizon(self) -> float:
        """Return angular velocity of event horizon"""
        r_plus = self.outer_horizon()
        return self.a * self.c / (r_plus**2 + self.a**2)

    def frame_dragging_velocity(self, r: float, theta: float) -> float:
        """
        Calculate frame dragging angular velocity at (r, θ)

        This is the angular velocity acquired by a ZAMO (zero angular momentum observer)
        """
        Sigma = self._sigma(r, theta)
        A = (r**2 + self.a**2)**2 - self._delta(r) * self.a**2 * np.sin(theta)**2
        return self.rs * r * self.a * self.c / A


class ReissnerNordstromMetric:
    """Reissner-Nordström metric for charged, non-rotating black hole"""

    def __init__(self, M: float, Q: float, c: float = C, G: float = G,
                 epsilon_0: float = 8.854e-12):
        """
        Initialize Reissner-Nordström metric

        Args:
            M: Black hole mass
            Q: Electric charge
            c: Speed of light
            G: Gravitational constant
            epsilon_0: Permittivity of free space
        """
        self.M = M
        self.Q = Q
        self.c = c
        self.G = G
        self.epsilon_0 = epsilon_0

        self.rs = 2 * G * M / c**2
        self.rQ = np.sqrt(G * Q**2 / (4 * np.pi * epsilon_0 * c**4))

        # Check extremal bound
        if self.rQ > 0.5 * self.rs:
            warnings.warn("Charge exceeds extremal limit - naked singularity")

    def metric(self, coords: ArrayLike) -> np.ndarray:
        """
        Get metric tensor at coordinates (t, r, θ, φ)

        Args:
            coords: (t, r, θ, φ) coordinates

        Returns:
            4x4 metric tensor g_μν
        """
        t, r, theta, phi = coords

        f = 1 - self.rs / r + self.rQ**2 / r**2

        g = np.zeros((4, 4))
        g[0, 0] = f * self.c**2
        g[1, 1] = -1 / f
        g[2, 2] = -r**2
        g[3, 3] = -r**2 * np.sin(theta)**2

        return g

    def horizons(self) -> Tuple[float, float]:
        """Return inner and outer horizon radii"""
        discriminant = (0.5 * self.rs)**2 - self.rQ**2
        if discriminant < 0:
            return None, None  # Naked singularity

        r_plus = 0.5 * self.rs + np.sqrt(discriminant)
        r_minus = 0.5 * self.rs - np.sqrt(discriminant)
        return r_minus, r_plus


class KerrNewmanMetric:
    """Kerr-Newman metric for charged, rotating black hole"""

    def __init__(self, M: float, a: float, Q: float, c: float = C, G: float = G,
                 epsilon_0: float = 8.854e-12):
        """
        Initialize Kerr-Newman metric

        Args:
            M: Black hole mass
            a: Specific angular momentum J/(Mc)
            Q: Electric charge
            c: Speed of light
            G: Gravitational constant
            epsilon_0: Permittivity
        """
        self.M = M
        self.a = a
        self.Q = Q
        self.c = c
        self.G = G

        self.rs = 2 * G * M / c**2
        self.rQ = np.sqrt(G * Q**2 / (4 * np.pi * epsilon_0 * c**4))

    def _sigma(self, r: float, theta: float) -> float:
        return r**2 + self.a**2 * np.cos(theta)**2

    def _delta(self, r: float) -> float:
        return r**2 - self.rs * r + self.a**2 + self.rQ**2

    def metric(self, coords: ArrayLike) -> np.ndarray:
        """Get metric tensor at coordinates (t, r, θ, φ)"""
        t, r, theta, phi = coords

        Sigma = self._sigma(r, theta)
        Delta = self._delta(r)
        sin2 = np.sin(theta)**2

        g = np.zeros((4, 4))
        g[0, 0] = (Delta - self.a**2 * sin2) * self.c**2 / Sigma
        g[0, 3] = (self.rs * r - self.rQ**2) * self.a * self.c * sin2 / Sigma
        g[3, 0] = g[0, 3]
        g[1, 1] = -Sigma / Delta
        g[2, 2] = -Sigma
        g[3, 3] = -((r**2 + self.a**2)**2 - Delta * self.a**2 * sin2) * sin2 / Sigma

        return g


class FRWMetric:
    """Friedmann-Robertson-Walker metric for cosmology"""

    def __init__(self, a_func: Callable, k: int = 0, c: float = C):
        """
        Initialize FRW metric

        Args:
            a_func: Scale factor function a(t)
            k: Spatial curvature (+1 closed, 0 flat, -1 open)
            c: Speed of light
        """
        self.a_func = a_func
        self.k = k
        self.c = c

    def metric(self, coords: ArrayLike) -> np.ndarray:
        """
        Get metric tensor at coordinates (t, r, θ, φ) or (t, χ, θ, φ)

        For k≠0, r is comoving radial coordinate with r = sin(χ) or sinh(χ)

        Args:
            coords: (t, r, θ, φ) comoving coordinates

        Returns:
            4x4 metric tensor g_μν
        """
        t, r, theta, phi = coords
        a = self.a_func(t)

        if self.k == 0:
            f_r = 1
        elif self.k == 1:
            f_r = 1 / (1 - r**2)
        else:  # k == -1
            f_r = 1 / (1 + r**2)

        g = np.zeros((4, 4))
        g[0, 0] = self.c**2
        g[1, 1] = -a**2 * f_r
        g[2, 2] = -a**2 * r**2
        g[3, 3] = -a**2 * r**2 * np.sin(theta)**2

        return g

    def hubble_parameter(self, t: float, dt: float = 1e-6) -> float:
        """Calculate Hubble parameter H = (1/a)(da/dt)"""
        a = self.a_func(t)
        da_dt = (self.a_func(t + dt) - self.a_func(t - dt)) / (2 * dt)
        return da_dt / a

    def comoving_distance(self, z: float, H0: float = 70e3,
                          Omega_m: float = 0.3, Omega_Lambda: float = 0.7) -> float:
        """
        Calculate comoving distance to redshift z

        Args:
            z: Redshift
            H0: Hubble constant in m/s/Mpc
            Omega_m: Matter density parameter
            Omega_Lambda: Dark energy density parameter

        Returns:
            Comoving distance in meters
        """
        from scipy.integrate import quad

        def E(z):
            return np.sqrt(Omega_m * (1 + z)**3 + Omega_Lambda)

        def integrand(z):
            return 1 / E(z)

        result, _ = quad(integrand, 0, z)
        return self.c * result / H0


# =============================================================================
# GR Phenomena
# =============================================================================

class GravitationalRedshift:
    """Gravitational redshift and blueshift calculations"""

    def __init__(self, metric: Union[SchwarzschildMetric, KerrMetric]):
        """
        Initialize gravitational redshift calculator

        Args:
            metric: Spacetime metric (Schwarzschild or Kerr)
        """
        self.metric = metric

    def redshift_factor(self, r_emit: float, r_obs: float,
                        theta: float = np.pi/2) -> float:
        """
        Calculate redshift factor z for static emitter/observer

        Args:
            r_emit: Radius of emission
            r_obs: Radius of observation (can be infinity)
            theta: Polar angle (for Kerr metric)

        Returns:
            Redshift factor z = (λ_obs - λ_emit)/λ_emit
        """
        if isinstance(self.metric, SchwarzschildMetric):
            # g_tt at emitter and observer
            g_emit = 1 - self.metric.rs / r_emit
            if r_obs == np.inf:
                g_obs = 1
            else:
                g_obs = 1 - self.metric.rs / r_obs
        else:  # Kerr
            g_emit = self.metric.metric([0, r_emit, theta, 0])[0, 0] / self.metric.c**2
            if r_obs == np.inf:
                g_obs = 1
            else:
                g_obs = self.metric.metric([0, r_obs, theta, 0])[0, 0] / self.metric.c**2

        return np.sqrt(g_obs / g_emit) - 1

    def wavelength_ratio(self, r_emit: float, r_obs: float,
                         theta: float = np.pi/2) -> float:
        """Calculate wavelength ratio λ_obs/λ_emit"""
        z = self.redshift_factor(r_emit, r_obs, theta)
        return 1 + z


class PeriastronPrecession:
    """Perihelion/periastron precession in Schwarzschild geometry"""

    def __init__(self, metric: SchwarzschildMetric):
        """
        Initialize precession calculator

        Args:
            metric: Schwarzschild metric
        """
        self.metric = metric

    def precession_per_orbit(self, a: float, e: float) -> float:
        """
        Calculate precession per orbit (in radians)

        Args:
            a: Semi-major axis
            e: Eccentricity

        Returns:
            Precession angle per orbit (radians)
        """
        # Semi-latus rectum
        p = a * (1 - e**2)

        # GR precession
        delta_phi = 6 * np.pi * self.metric.G * self.metric.M / (self.metric.c**2 * p)

        return delta_phi

    def precession_rate(self, a: float, e: float) -> float:
        """
        Calculate precession rate (radians per second)

        Args:
            a: Semi-major axis
            e: Eccentricity

        Returns:
            Precession rate (rad/s)
        """
        T = 2 * np.pi * np.sqrt(a**3 / (self.metric.G * self.metric.M))
        return self.precession_per_orbit(a, e) / T


class GravitationalLensing:
    """Gravitational lensing calculations"""

    def __init__(self, M: float, c: float = C, G: float = G):
        """
        Initialize gravitational lensing

        Args:
            M: Lens mass
            c: Speed of light
            G: Gravitational constant
        """
        self.M = M
        self.c = c
        self.G = G

    def einstein_radius(self, D_ls: float, D_l: float, D_s: float) -> float:
        """
        Calculate Einstein radius

        Args:
            D_ls: Distance from lens to source
            D_l: Distance to lens
            D_s: Distance to source

        Returns:
            Einstein radius (radians)
        """
        return np.sqrt(4 * self.G * self.M * D_ls / (self.c**2 * D_l * D_s))

    def deflection_angle(self, b: float) -> float:
        """
        Calculate light deflection angle for impact parameter b

        Args:
            b: Impact parameter

        Returns:
            Deflection angle (radians)
        """
        return 4 * self.G * self.M / (self.c**2 * b)

    def image_positions(self, beta: float, theta_E: float) -> Tuple[float, float]:
        """
        Calculate image positions for point mass lens

        Args:
            beta: Source position (angle from optical axis)
            theta_E: Einstein radius

        Returns:
            Two image positions θ_+ and θ_-
        """
        disc = beta**2 + 4 * theta_E**2
        theta_plus = 0.5 * (beta + np.sqrt(disc))
        theta_minus = 0.5 * (beta - np.sqrt(disc))
        return theta_plus, theta_minus

    def magnification(self, beta: float, theta_E: float) -> Tuple[float, float]:
        """
        Calculate magnifications of the two images

        Args:
            beta: Source position
            theta_E: Einstein radius

        Returns:
            Magnifications μ_+ and μ_-
        """
        u = beta / theta_E
        u2 = u * u
        mu_plus = (u2 + 2) / (2 * u * np.sqrt(u2 + 4)) + 0.5
        mu_minus = (u2 + 2) / (2 * u * np.sqrt(u2 + 4)) - 0.5
        return mu_plus, abs(mu_minus)

    def time_delay(self, theta_1: float, theta_2: float, beta: float,
                   D_l: float, D_s: float, D_ls: float) -> float:
        """
        Calculate time delay between two images

        Args:
            theta_1, theta_2: Image positions
            beta: Source position
            D_l, D_s, D_ls: Distances

        Returns:
            Time delay (seconds)
        """
        r_s = 2 * self.G * self.M / self.c**2
        tau = D_l * D_s / (self.c * D_ls)

        # Geometric + Shapiro delay difference
        delay = 0.5 * tau * ((theta_1 - beta)**2 - (theta_2 - beta)**2)
        delay += r_s * D_l * D_s / (D_ls * self.c) * np.log(abs(theta_1 / theta_2))

        return delay


class FrameDragging:
    """Frame dragging (Lense-Thirring effect) calculations"""

    def __init__(self, metric: KerrMetric):
        """
        Initialize frame dragging calculator

        Args:
            metric: Kerr metric
        """
        self.metric = metric

    def precession_rate(self, r: float, theta: float = np.pi/2) -> np.ndarray:
        """
        Calculate frame-dragging precession rate for a gyroscope

        Args:
            r: Radial distance
            theta: Polar angle

        Returns:
            Precession angular velocity vector (rad/s)
        """
        # Lense-Thirring precession
        J = self.metric.a * self.metric.M * self.metric.c
        r3 = r**3

        # Precession in z-direction (aligned with spin axis)
        omega_z = 2 * self.metric.G * J / (self.metric.c**2 * r3)

        return np.array([0, 0, omega_z])

    def geodetic_precession_rate(self, r: float) -> float:
        """
        Calculate geodetic (de Sitter) precession rate

        This is the precession due to curved spacetime (not frame dragging)

        Args:
            r: Orbital radius

        Returns:
            Precession rate (rad/s)
        """
        M = self.metric.M
        omega_orbit = np.sqrt(self.metric.G * M / r**3)
        return 1.5 * (self.metric.G * M / (self.metric.c**2 * r)) * omega_orbit


class EventHorizon:
    """Event horizon properties and calculations"""

    def __init__(self, M: float, a: float = 0, Q: float = 0,
                 c: float = C, G: float = G):
        """
        Initialize event horizon calculator

        Args:
            M: Black hole mass
            a: Spin parameter (for Kerr)
            Q: Charge (for Reissner-Nordström/Kerr-Newman)
            c: Speed of light
            G: Gravitational constant
        """
        self.M = M
        self.a = a
        self.Q = Q
        self.c = c
        self.G = G

        self.rs = 2 * G * M / c**2

    def radius(self) -> float:
        """Calculate event horizon radius"""
        if self.a == 0 and self.Q == 0:
            return self.rs
        else:
            # General Kerr-Newman
            discriminant = (0.5 * self.rs)**2 - self.a**2 - self.Q**2 * self.G / self.c**4
            if discriminant < 0:
                return 0  # Naked singularity
            return 0.5 * self.rs + np.sqrt(discriminant)

    def surface_area(self) -> float:
        """Calculate event horizon surface area"""
        r_h = self.radius()
        if self.a == 0:
            return 4 * np.pi * r_h**2
        else:
            # Kerr
            return 4 * np.pi * (r_h**2 + self.a**2)

    def entropy(self) -> float:
        """Calculate Bekenstein-Hawking entropy S = A/(4 l_P²)"""
        A = self.surface_area()
        l_P_squared = HBAR * self.G / self.c**3  # Planck length squared
        return A / (4 * l_P_squared)


class HawkingTemperature:
    """Hawking radiation temperature"""

    def __init__(self, M: float, a: float = 0, c: float = C, G: float = G):
        """
        Initialize Hawking temperature calculator

        Args:
            M: Black hole mass
            a: Spin parameter
            c: Speed of light
            G: Gravitational constant
        """
        self.M = M
        self.a = a
        self.c = c
        self.G = G

    def temperature(self) -> float:
        """Calculate Hawking temperature"""
        if self.a == 0:
            # Schwarzschild
            return HBAR * self.c**3 / (8 * np.pi * self.G * self.M * KB)
        else:
            # Kerr
            r_plus = self.G * self.M / self.c**2 + np.sqrt(
                (self.G * self.M / self.c**2)**2 - self.a**2
            )
            kappa = (r_plus - self.G * self.M / self.c**2) / (2 * r_plus * (r_plus + self.a))
            kappa *= self.c**2  # Dimensionally correct
            return HBAR * kappa / (2 * np.pi * KB)

    def luminosity(self) -> float:
        """Calculate Hawking luminosity (Stefan-Boltzmann law)"""
        T = self.temperature()
        A = EventHorizon(self.M, self.a, c=self.c, G=self.G).surface_area()
        sigma = np.pi**2 * KB**4 / (60 * HBAR**3 * self.c**2)
        return sigma * A * T**4

    def evaporation_time(self) -> float:
        """Estimate evaporation time (assuming constant temperature - very rough)"""
        # More accurate: t ~ M³
        return 5120 * np.pi * self.G**2 * self.M**3 / (HBAR * self.c**4)


# =============================================================================
# Gravitational Waves
# =============================================================================

class LinearizedGravity:
    """Linearized gravity (weak field approximation)"""

    def __init__(self, c: float = C, G: float = G):
        """
        Initialize linearized gravity

        Args:
            c: Speed of light
            G: Gravitational constant
        """
        self.c = c
        self.G = G

    def metric_perturbation(self, M: float, r: float) -> np.ndarray:
        """
        Calculate metric perturbation h_μν for static mass

        g_μν = η_μν + h_μν where |h| << 1

        Args:
            M: Source mass
            r: Distance from source

        Returns:
            Metric perturbation h_μν
        """
        phi = -self.G * M / r  # Newtonian potential

        h = np.zeros((4, 4))
        h[0, 0] = -2 * phi / self.c**2
        h[1, 1] = -2 * phi / self.c**2
        h[2, 2] = -2 * phi / self.c**2
        h[3, 3] = -2 * phi / self.c**2

        return h

    def trace_reversed(self, h: np.ndarray) -> np.ndarray:
        """
        Calculate trace-reversed perturbation h̄_μν = h_μν - (1/2)η_μν h

        Args:
            h: Metric perturbation

        Returns:
            Trace-reversed perturbation
        """
        eta = np.diag([1, -1, -1, -1])
        trace = np.trace(eta @ h)
        return h - 0.5 * eta * trace

    def lorenz_gauge_residual(self, h_bar: np.ndarray, dx: np.ndarray) -> np.ndarray:
        """
        Check Lorenz gauge condition: ∂^ν h̄_μν = 0

        Args:
            h_bar: Trace-reversed perturbation (on grid)
            dx: Grid spacing

        Returns:
            Gauge residual (should be zero)
        """
        residual = np.zeros(4)
        eta_inv = np.diag([1, -1, -1, -1])

        for mu in range(4):
            for nu in range(4):
                derivative = np.gradient(h_bar[mu, nu], dx[nu], axis=nu)
                residual[mu] += eta_inv[nu, nu] * np.mean(derivative)

        return residual


class GravitationalWave:
    """Gravitational wave calculations and waveforms"""

    def __init__(self, frequency: float = None, h_plus: float = 1e-21,
                 h_cross: float = 0, c: float = C):
        """
        Initialize gravitational wave

        Args:
            frequency: Wave frequency (Hz)
            h_plus: Plus polarization amplitude
            h_cross: Cross polarization amplitude
            c: Speed of light
        """
        self.frequency = frequency
        self.h_plus = h_plus
        self.h_cross = h_cross
        self.c = c

    def strain(self, t: ArrayLike, direction: str = 'plus') -> np.ndarray:
        """
        Calculate strain h(t) for given polarization

        Args:
            t: Time array
            direction: 'plus' or 'cross' polarization

        Returns:
            Strain time series
        """
        t = np.array(t)
        omega = 2 * np.pi * self.frequency

        if direction == 'plus':
            return self.h_plus * np.cos(omega * t)
        else:
            return self.h_cross * np.sin(omega * t)

    def metric_perturbation(self, t: float, z: float = 0) -> np.ndarray:
        """
        Get TT-gauge metric perturbation for wave traveling in z-direction

        Args:
            t: Time
            z: z-coordinate

        Returns:
            4x4 perturbation h_μν^TT
        """
        phase = 2 * np.pi * self.frequency * (t - z / self.c)

        h = np.zeros((4, 4))
        h[1, 1] = self.h_plus * np.cos(phase)
        h[2, 2] = -self.h_plus * np.cos(phase)
        h[1, 2] = self.h_cross * np.sin(phase)
        h[2, 1] = self.h_cross * np.sin(phase)

        return h

    def power_flux(self) -> float:
        """Calculate power flux (W/m²) of the wave"""
        omega = 2 * np.pi * self.frequency
        h_rms = np.sqrt(self.h_plus**2 + self.h_cross**2) / np.sqrt(2)
        return self.c**3 * omega**2 * h_rms**2 / (16 * np.pi * G)

    @classmethod
    def from_binary(cls, m1: float, m2: float, r: float,
                    distance: float, c: float = C, G: float = G) -> 'GravitationalWave':
        """
        Create gravitational wave from binary system

        Args:
            m1, m2: Component masses
            r: Orbital separation
            distance: Distance to source
            c: Speed of light
            G: Gravitational constant

        Returns:
            GravitationalWave instance
        """
        # Chirp mass
        M_c = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)

        # Orbital frequency
        f_orb = np.sqrt(G * (m1 + m2) / r**3) / (2 * np.pi)
        f_gw = 2 * f_orb  # GW frequency is twice orbital

        # Strain amplitude (leading order)
        h = (4 / distance) * (G * M_c / c**2)**(5/3) * (np.pi * f_gw / c)**(2/3)

        return cls(frequency=f_gw, h_plus=h, h_cross=h, c=c)


class QuadrupoleFormula:
    """Quadrupole radiation formula for gravitational waves"""

    def __init__(self, c: float = C, G: float = G):
        """
        Initialize quadrupole formula calculator

        Args:
            c: Speed of light
            G: Gravitational constant
        """
        self.c = c
        self.G = G

    def quadrupole_moment(self, masses: ArrayLike, positions: ArrayLike) -> np.ndarray:
        """
        Calculate quadrupole moment tensor

        Q_ij = Σ m_a (3 x_a^i x_a^j - |x_a|² δ_ij)

        Args:
            masses: Array of masses
            positions: Array of 3D position vectors

        Returns:
            3x3 quadrupole moment tensor
        """
        masses = np.array(masses)
        positions = np.array(positions)

        Q = np.zeros((3, 3))
        for m, x in zip(masses, positions):
            r2 = np.sum(x**2)
            for i in range(3):
                for j in range(3):
                    Q[i, j] += m * (3 * x[i] * x[j] - r2 * (i == j))

        return Q

    def strain_tensor(self, Q_ddot: np.ndarray, distance: float) -> np.ndarray:
        """
        Calculate strain from second time derivative of quadrupole moment

        h_ij^TT = (2G/c⁴r) Q̈_ij^TT

        Args:
            Q_ddot: Second time derivative of quadrupole moment
            distance: Distance to source

        Returns:
            Strain tensor h_ij
        """
        prefactor = 2 * self.G / (self.c**4 * distance)
        return prefactor * Q_ddot

    def luminosity(self, Q_dddot: np.ndarray) -> float:
        """
        Calculate gravitational wave luminosity

        L = (G/5c⁵) ⟨Q⃛_ij Q⃛^ij⟩

        Args:
            Q_dddot: Third time derivative of quadrupole moment

        Returns:
            Power radiated in gravitational waves (W)
        """
        return self.G / (5 * self.c**5) * np.sum(Q_dddot**2)

    def binary_luminosity(self, m1: float, m2: float, r: float) -> float:
        """
        Calculate luminosity for circular binary

        Args:
            m1, m2: Component masses
            r: Orbital separation

        Returns:
            Power radiated (W)
        """
        mu = m1 * m2 / (m1 + m2)  # Reduced mass
        M = m1 + m2

        omega = np.sqrt(self.G * M / r**3)

        return (32/5) * self.G**4 * mu**2 * M**3 / (self.c**5 * r**5)


class ChirpMass:
    """Chirp mass and binary inspiral parameters"""

    def __init__(self, m1: float, m2: float, c: float = C, G: float = G):
        """
        Initialize chirp mass calculator

        Args:
            m1, m2: Component masses
            c: Speed of light
            G: Gravitational constant
        """
        self.m1 = m1
        self.m2 = m2
        self.c = c
        self.G = G

        self.M = m1 + m2  # Total mass
        self.mu = m1 * m2 / self.M  # Reduced mass
        self.eta = self.mu / self.M  # Symmetric mass ratio
        self.Mc = self.M * self.eta**(3/5)  # Chirp mass

    @property
    def chirp_mass(self) -> float:
        """Return chirp mass"""
        return self.Mc

    def frequency_evolution(self, f: float) -> float:
        """
        Calculate df/dt at frequency f (leading order)

        Args:
            f: Gravitational wave frequency

        Returns:
            Frequency derivative (Hz/s)
        """
        prefactor = (96/5) * np.pi**(8/3)
        return prefactor * (self.G * self.Mc / self.c**3)**(5/3) * f**(11/3)

    def time_to_merger(self, f: float) -> float:
        """
        Calculate time to merger from frequency f

        Args:
            f: Current GW frequency

        Returns:
            Time to merger (s)
        """
        prefactor = 5 / (256 * np.pi**(8/3))
        return prefactor * (self.c**3 / (self.G * self.Mc))**(5/3) * f**(-8/3)

    def isco_frequency(self) -> float:
        """Calculate frequency at ISCO (innermost stable circular orbit)"""
        return self.c**3 / (6**(3/2) * np.pi * self.G * self.M)


class GWTemplate:
    """Gravitational wave templates for matched filtering"""

    def __init__(self, chirp_mass: ChirpMass, distance: float = 1e24):
        """
        Initialize GW template generator

        Args:
            chirp_mass: ChirpMass object
            distance: Distance to source (m)
        """
        self.chirp = chirp_mass
        self.distance = distance

    def inspiral_waveform(self, t: ArrayLike, f0: float,
                          phi0: float = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Newtonian-order inspiral waveform

        Args:
            t: Time array (relative to merger)
            f0: Initial frequency
            phi0: Initial phase

        Returns:
            (h_plus, h_cross) strain arrays
        """
        t = np.array(t)

        # Time to merger from f0
        tc = self.chirp.time_to_merger(f0)
        tau = tc - t  # Time remaining to merger

        # Avoid negative tau
        tau = np.maximum(tau, 1e-10)

        # Frequency evolution
        f = ((256/5) * (np.pi * self.chirp.G * self.chirp.Mc / self.chirp.c**3)**(5/3) * tau)**(-3/8)
        f /= np.pi

        # Phase evolution
        phi = phi0 - 2 * ((5 * self.chirp.c**3 * tau) /
                          (256 * self.chirp.G * self.chirp.Mc))**(5/8)

        # Amplitude
        h0 = (4 / self.distance) * (self.chirp.G * self.chirp.Mc / self.chirp.c**2)**(5/3)
        h0 *= (np.pi * f / self.chirp.c)**(2/3)

        h_plus = h0 * np.cos(phi)
        h_cross = h0 * np.sin(phi)

        return h_plus, h_cross

    def frequency_domain(self, f: ArrayLike) -> np.ndarray:
        """
        Calculate frequency-domain waveform (stationary phase approximation)

        Args:
            f: Frequency array

        Returns:
            Complex frequency-domain strain h(f)
        """
        f = np.array(f)
        Mc = self.chirp.Mc

        # Amplitude
        A = np.sqrt(5 * np.pi / 24) * self.chirp.c / self.distance
        A *= (self.chirp.G * Mc / self.chirp.c**3)**(5/6)
        A *= f**(-7/6)

        # Phase (leading order)
        psi = 2 * np.pi * f * self.chirp.time_to_merger(f[0])  # Reference phase
        psi -= np.pi / 4
        psi += (3/128) * (np.pi * self.chirp.G * Mc * f / self.chirp.c**3)**(-5/3)

        return A * np.exp(1j * psi)


# =============================================================================
# Cosmology
# =============================================================================

class FriedmannEquations:
    """Friedmann equations for cosmological dynamics"""

    def __init__(self, H0: float = 70e3, Omega_m: float = 0.3,
                 Omega_r: float = 9e-5, Omega_Lambda: float = 0.7,
                 Omega_k: float = 0, c: float = C):
        """
        Initialize Friedmann equations

        Args:
            H0: Hubble constant (m/s/Mpc)
            Omega_m: Matter density parameter
            Omega_r: Radiation density parameter
            Omega_Lambda: Dark energy density parameter
            Omega_k: Curvature parameter
            c: Speed of light
        """
        self.H0 = H0
        self.Omega_m = Omega_m
        self.Omega_r = Omega_r
        self.Omega_Lambda = Omega_Lambda
        self.Omega_k = Omega_k
        self.c = c

        # Check normalization
        total = Omega_m + Omega_r + Omega_Lambda + Omega_k
        if abs(total - 1) > 1e-6:
            warnings.warn(f"Density parameters sum to {total}, not 1")

    def hubble(self, z: float) -> float:
        """
        Calculate Hubble parameter at redshift z

        H(z) = H0 * E(z)

        Args:
            z: Redshift

        Returns:
            H(z) in same units as H0
        """
        return self.H0 * self.E(z)

    def E(self, z: float) -> float:
        """
        Calculate dimensionless Hubble parameter E(z) = H(z)/H0

        Args:
            z: Redshift

        Returns:
            E(z)
        """
        a = 1 / (1 + z)
        return np.sqrt(
            self.Omega_r * a**(-4) +
            self.Omega_m * a**(-3) +
            self.Omega_k * a**(-2) +
            self.Omega_Lambda
        )

    def deceleration_parameter(self, z: float = 0) -> float:
        """
        Calculate deceleration parameter q(z)

        q = -ä a / ȧ²

        Args:
            z: Redshift

        Returns:
            q(z)
        """
        a = 1 / (1 + z)
        E2 = self.E(z)**2

        q = 0.5 * (self.Omega_r * a**(-4) + self.Omega_m * a**(-3) - 2 * self.Omega_Lambda) / E2
        return q

    def age(self, z: float = 0) -> float:
        """
        Calculate age of universe at redshift z

        Args:
            z: Redshift

        Returns:
            Age in seconds
        """
        from scipy.integrate import quad

        def integrand(zp):
            return 1 / ((1 + zp) * self.E(zp))

        result, _ = quad(integrand, z, np.inf)
        return result / self.H0

    def solve_scale_factor(self, t_span: Tuple[float, float],
                           n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve for scale factor a(t)

        Args:
            t_span: (t_initial, t_final) in seconds
            n_points: Number of output points

        Returns:
            (t, a) arrays
        """
        def rhs(t, a):
            z = 1/a - 1
            return a * self.H0 * self.E(z)

        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        sol = solve_ivp(rhs, t_span, [1/(1 + 1000)], t_eval=t_eval)  # Start at z=1000

        return sol.t, sol.y[0]


class HubbleParameter:
    """Hubble parameter calculations and observations"""

    def __init__(self, friedmann: FriedmannEquations):
        """
        Initialize Hubble parameter calculator

        Args:
            friedmann: Friedmann equations instance
        """
        self.friedmann = friedmann

    def hubble_time(self) -> float:
        """Return Hubble time t_H = 1/H0"""
        return 1 / self.friedmann.H0

    def hubble_distance(self) -> float:
        """Return Hubble distance d_H = c/H0"""
        return self.friedmann.c / self.friedmann.H0

    def at_redshift(self, z: float) -> float:
        """Get H(z)"""
        return self.friedmann.hubble(z)

    def from_distances(self, z: ArrayLike, D_L: ArrayLike) -> np.ndarray:
        """
        Infer H(z) from luminosity distance data (mock observation)

        Args:
            z: Redshift array
            D_L: Luminosity distance array

        Returns:
            Inferred H(z)
        """
        # D_L = (1+z) * integral(c/H(z') dz')
        # Numerical differentiation to get H(z)
        z = np.array(z)
        D_L = np.array(D_L)

        D_C = D_L / (1 + z)  # Comoving distance
        dD_dz = np.gradient(D_C, z)

        return self.friedmann.c / dD_dz


class CosmicScale:
    """Cosmic scale factor calculations"""

    def __init__(self, friedmann: FriedmannEquations):
        """
        Initialize scale factor calculator

        Args:
            friedmann: Friedmann equations instance
        """
        self.friedmann = friedmann

    def from_redshift(self, z: float) -> float:
        """Calculate scale factor from redshift: a = 1/(1+z)"""
        return 1 / (1 + z)

    def to_redshift(self, a: float) -> float:
        """Calculate redshift from scale factor: z = 1/a - 1"""
        return 1/a - 1

    def lookback_time(self, z: float) -> float:
        """
        Calculate lookback time to redshift z

        Args:
            z: Redshift

        Returns:
            Lookback time in seconds
        """
        t_now = self.friedmann.age(0)
        t_then = self.friedmann.age(z)
        return t_now - t_then

    def proper_distance(self, z: float) -> float:
        """
        Calculate proper distance to object at redshift z

        Args:
            z: Redshift

        Returns:
            Proper distance (m)
        """
        from scipy.integrate import quad

        def integrand(zp):
            return 1 / self.friedmann.E(zp)

        result, _ = quad(integrand, 0, z)
        return self.friedmann.c * result / self.friedmann.H0


class RedshiftDistance:
    """Cosmological distance-redshift relations"""

    def __init__(self, friedmann: FriedmannEquations):
        """
        Initialize distance calculator

        Args:
            friedmann: Friedmann equations instance
        """
        self.friedmann = friedmann

    def comoving_distance(self, z: float) -> float:
        """Calculate comoving distance to redshift z"""
        from scipy.integrate import quad

        def integrand(zp):
            return 1 / self.friedmann.E(zp)

        result, _ = quad(integrand, 0, z)
        return self.friedmann.c * result / self.friedmann.H0

    def luminosity_distance(self, z: float) -> float:
        """Calculate luminosity distance to redshift z"""
        D_C = self.comoving_distance(z)
        return (1 + z) * D_C

    def angular_diameter_distance(self, z: float) -> float:
        """Calculate angular diameter distance to redshift z"""
        D_C = self.comoving_distance(z)
        return D_C / (1 + z)

    def distance_modulus(self, z: float) -> float:
        """Calculate distance modulus μ = 5 log10(D_L/10pc)"""
        D_L = self.luminosity_distance(z)
        D_L_pc = D_L / 3.086e16  # Convert to parsecs
        return 5 * np.log10(D_L_pc / 10)


class DarkEnergy:
    """Dark energy equation of state and dynamics"""

    def __init__(self, w0: float = -1, wa: float = 0, c: float = C):
        """
        Initialize dark energy model

        Uses CPL parameterization: w(a) = w0 + wa(1-a)

        Args:
            w0: Present-day equation of state
            wa: Time evolution parameter
            c: Speed of light
        """
        self.w0 = w0
        self.wa = wa
        self.c = c

    def equation_of_state(self, z: float) -> float:
        """
        Calculate equation of state w(z)

        Args:
            z: Redshift

        Returns:
            w = p/ρ
        """
        a = 1 / (1 + z)
        return self.w0 + self.wa * (1 - a)

    def density_ratio(self, z: float) -> float:
        """
        Calculate ρ_DE(z)/ρ_DE(0)

        Args:
            z: Redshift

        Returns:
            Density ratio
        """
        a = 1 / (1 + z)

        if abs(self.wa) < 1e-10:
            # Constant w
            return a**(-3 * (1 + self.w0))
        else:
            # CPL
            return a**(-3 * (1 + self.w0 + self.wa)) * np.exp(-3 * self.wa * (1 - a))

    def is_phantom(self, z: float = 0) -> bool:
        """Check if dark energy is phantom (w < -1)"""
        return self.equation_of_state(z) < -1

    def crossing_redshift(self) -> Optional[float]:
        """
        Find redshift where w = -1 (phantom divide crossing)

        Returns:
            Crossing redshift or None if no crossing
        """
        if self.wa == 0:
            return None

        # w = -1 when w0 + wa(1-a) = -1
        # a = 1 + (w0 + 1)/wa
        a = 1 + (self.w0 + 1) / self.wa

        if 0 < a < 1:
            return 1/a - 1
        return None


class InflationModel:
    """Slow-roll inflation model"""

    def __init__(self, V_func: Callable, dV_func: Callable,
                 d2V_func: Callable, M_pl: float = 2.435e18):
        """
        Initialize inflation model

        Args:
            V_func: Potential V(φ)
            dV_func: First derivative V'(φ)
            d2V_func: Second derivative V''(φ)
            M_pl: Reduced Planck mass (GeV)
        """
        self.V = V_func
        self.dV = dV_func
        self.d2V = d2V_func
        self.M_pl = M_pl

    def epsilon(self, phi: float) -> float:
        """
        Calculate slow-roll parameter ε = (M_pl²/2)(V'/V)²

        Args:
            phi: Field value

        Returns:
            ε parameter
        """
        return 0.5 * self.M_pl**2 * (self.dV(phi) / self.V(phi))**2

    def eta(self, phi: float) -> float:
        """
        Calculate slow-roll parameter η = M_pl² V''/V

        Args:
            phi: Field value

        Returns:
            η parameter
        """
        return self.M_pl**2 * self.d2V(phi) / self.V(phi)

    def n_s(self, phi: float) -> float:
        """
        Calculate scalar spectral index n_s

        n_s ≈ 1 - 6ε + 2η

        Args:
            phi: Field value

        Returns:
            Spectral index
        """
        return 1 - 6 * self.epsilon(phi) + 2 * self.eta(phi)

    def n_t(self, phi: float) -> float:
        """
        Calculate tensor spectral index n_t

        n_t ≈ -2ε

        Args:
            phi: Field value

        Returns:
            Tensor spectral index
        """
        return -2 * self.epsilon(phi)

    def r(self, phi: float) -> float:
        """
        Calculate tensor-to-scalar ratio r

        r ≈ 16ε

        Args:
            phi: Field value

        Returns:
            Tensor-to-scalar ratio
        """
        return 16 * self.epsilon(phi)

    def e_folds(self, phi_start: float, phi_end: float, n_steps: int = 1000) -> float:
        """
        Calculate number of e-folds

        N = ∫ (V/V') dφ / M_pl²

        Args:
            phi_start: Starting field value
            phi_end: Ending field value
            n_steps: Integration steps

        Returns:
            Number of e-folds
        """
        from scipy.integrate import quad

        def integrand(phi):
            return self.V(phi) / (self.M_pl**2 * self.dV(phi))

        result, _ = quad(integrand, phi_start, phi_end)
        return abs(result)

    @classmethod
    def chaotic(cls, m: float, M_pl: float = 2.435e18) -> 'InflationModel':
        """
        Create chaotic inflation model V = (1/2)m²φ²

        Args:
            m: Inflaton mass
            M_pl: Reduced Planck mass

        Returns:
            InflationModel instance
        """
        return cls(
            V_func=lambda phi: 0.5 * m**2 * phi**2,
            dV_func=lambda phi: m**2 * phi,
            d2V_func=lambda phi: m**2,
            M_pl=M_pl
        )

    @classmethod
    def starobinsky(cls, M: float, M_pl: float = 2.435e18) -> 'InflationModel':
        """
        Create Starobinsky (R²) inflation model

        V = (3/4)M²M_pl²(1 - e^(-√(2/3)φ/M_pl))²

        Args:
            M: Mass scale
            M_pl: Reduced Planck mass

        Returns:
            InflationModel instance
        """
        sqrt_23 = np.sqrt(2/3)

        def V(phi):
            x = np.exp(-sqrt_23 * phi / M_pl)
            return 0.75 * M**2 * M_pl**2 * (1 - x)**2

        def dV(phi):
            x = np.exp(-sqrt_23 * phi / M_pl)
            return 0.75 * M**2 * M_pl**2 * 2 * (1 - x) * sqrt_23 / M_pl * x

        def d2V(phi):
            x = np.exp(-sqrt_23 * phi / M_pl)
            return 0.75 * M**2 * M_pl**2 * 2 * (sqrt_23 / M_pl)**2 * x * (2*x - 1)

        return cls(V_func=V, dV_func=dV, d2V_func=d2V, M_pl=M_pl)
