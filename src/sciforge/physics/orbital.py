"""
Orbital mechanics primitives.

This module provides classes for analyzing and simulating orbital motion,
including Keplerian orbits, two-body and three-body problems, orbital
maneuvers, and escape trajectories.
"""

import numpy as np
from typing import Optional, Tuple, Dict, List, Union
from numpy.typing import ArrayLike
from dataclasses import dataclass

from ..core.base import BaseClass
from ..core.constants import CONSTANTS
from ..core.utils import (
    validate_vector,
    validate_positive,
    validate_non_negative,
    validate_finite,
    validate_bounds,
    normalize_vector,
)
from ..core.exceptions import ValidationError, PhysicsError
from .base import DynamicalSystem


@dataclass
class OrbitalElements:
    """
    Classical orbital elements (Keplerian elements).

    Attributes:
        a: Semi-major axis (m)
        e: Eccentricity (dimensionless)
        i: Inclination (rad)
        omega: Argument of periapsis (rad)
        Omega: Longitude of ascending node (rad)
        nu: True anomaly (rad)
        M: Mean anomaly (rad)
        E: Eccentric anomaly (rad)
    """
    a: float  # Semi-major axis
    e: float  # Eccentricity
    i: float = 0.0  # Inclination
    omega: float = 0.0  # Argument of periapsis
    Omega: float = 0.0  # Longitude of ascending node
    nu: float = 0.0  # True anomaly
    M: float = 0.0  # Mean anomaly
    E: float = 0.0  # Eccentric anomaly


class KeplerianOrbit(BaseClass):
    """
    Keplerian orbital motion around a central body.

    Models orbital motion using the classical two-body solution with
    Keplerian elements and provides conversions between state vectors
    and orbital elements.

    Args:
        central_mass: Mass of the central body (kg)
        semi_major_axis: Semi-major axis (m)
        eccentricity: Orbital eccentricity (0 <= e < 1 for bound orbits)
        inclination: Orbital inclination (rad), default 0
        arg_periapsis: Argument of periapsis (rad), default 0
        long_asc_node: Longitude of ascending node (rad), default 0
        true_anomaly: Initial true anomaly (rad), default 0
        G: Gravitational constant

    Examples:
        >>> # Earth orbit around Sun
        >>> orbit = KeplerianOrbit(
        ...     central_mass=1.989e30,  # Solar mass
        ...     semi_major_axis=1.496e11,  # 1 AU
        ...     eccentricity=0.017
        ... )
        >>> print(f"Period: {orbit.period() / (365.25*24*3600):.2f} years")
    """

    def __init__(
        self,
        central_mass: float,
        semi_major_axis: float,
        eccentricity: float,
        inclination: float = 0.0,
        arg_periapsis: float = 0.0,
        long_asc_node: float = 0.0,
        true_anomaly: float = 0.0,
        G: float = CONSTANTS['G'],
    ):
        super().__init__()

        self.M_central = validate_positive(central_mass, "central_mass")
        self.a = validate_positive(semi_major_axis, "semi_major_axis")
        self.e = validate_bounds(eccentricity, (0, 1 - 1e-10), "eccentricity")
        self.i = validate_finite(inclination, "inclination")
        self.omega = validate_finite(arg_periapsis, "arg_periapsis")
        self.Omega = validate_finite(long_asc_node, "long_asc_node")
        self.nu = validate_finite(true_anomaly, "true_anomaly")
        self.G = validate_positive(G, "G")

        # Standard gravitational parameter
        self.mu = self.G * self.M_central

        # Calculate other anomalies
        self.E = self._true_to_eccentric_anomaly(self.nu)
        self.M = self._eccentric_to_mean_anomaly(self.E)

        self.time = 0.0

    def _true_to_eccentric_anomaly(self, nu: float) -> float:
        """Convert true anomaly to eccentric anomaly."""
        return 2 * np.arctan(np.sqrt((1 - self.e) / (1 + self.e)) * np.tan(nu / 2))

    def _eccentric_to_mean_anomaly(self, E: float) -> float:
        """Convert eccentric anomaly to mean anomaly (Kepler's equation)."""
        return E - self.e * np.sin(E)

    def _mean_to_eccentric_anomaly(self, M: float, tolerance: float = 1e-12) -> float:
        """
        Convert mean anomaly to eccentric anomaly using Newton's method.

        Solves Kepler's equation: M = E - e*sin(E)
        """
        E = M  # Initial guess
        for _ in range(100):
            f = E - self.e * np.sin(E) - M
            fp = 1 - self.e * np.cos(E)
            dE = -f / fp
            E += dE
            if abs(dE) < tolerance:
                break
        return E

    def _eccentric_to_true_anomaly(self, E: float) -> float:
        """Convert eccentric anomaly to true anomaly."""
        return 2 * np.arctan(np.sqrt((1 + self.e) / (1 - self.e)) * np.tan(E / 2))

    def period(self) -> float:
        """
        Calculate orbital period.

        T = 2π√(a³/μ)

        Returns:
            Orbital period (s)
        """
        return 2 * np.pi * np.sqrt(self.a**3 / self.mu)

    def mean_motion(self) -> float:
        """
        Calculate mean motion.

        n = √(μ/a³) = 2π/T

        Returns:
            Mean motion (rad/s)
        """
        return np.sqrt(self.mu / self.a**3)

    def periapsis(self) -> float:
        """
        Calculate periapsis distance.

        r_p = a(1 - e)

        Returns:
            Periapsis distance (m)
        """
        return self.a * (1 - self.e)

    def apoapsis(self) -> float:
        """
        Calculate apoapsis distance.

        r_a = a(1 + e)

        Returns:
            Apoapsis distance (m)
        """
        return self.a * (1 + self.e)

    def radius(self, nu: Optional[float] = None) -> float:
        """
        Calculate orbital radius at given true anomaly.

        r = a(1 - e²) / (1 + e*cos(ν))

        Args:
            nu: True anomaly (rad), uses current if None

        Returns:
            Orbital radius (m)
        """
        if nu is None:
            nu = self.nu
        return self.a * (1 - self.e**2) / (1 + self.e * np.cos(nu))

    def velocity_magnitude(self, r: Optional[float] = None) -> float:
        """
        Calculate orbital velocity magnitude using vis-viva equation.

        v = √(μ(2/r - 1/a))

        Args:
            r: Orbital radius (m), uses current if None

        Returns:
            Velocity magnitude (m/s)
        """
        if r is None:
            r = self.radius()
        return np.sqrt(self.mu * (2 / r - 1 / self.a))

    def specific_energy(self) -> float:
        """
        Calculate specific orbital energy.

        ε = -μ/(2a)

        Returns:
            Specific energy (J/kg)
        """
        return -self.mu / (2 * self.a)

    def specific_angular_momentum(self) -> float:
        """
        Calculate specific angular momentum magnitude.

        h = √(μa(1-e²))

        Returns:
            Specific angular momentum (m²/s)
        """
        return np.sqrt(self.mu * self.a * (1 - self.e**2))

    def state_vectors(self, nu: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate position and velocity vectors in the orbital plane.

        Args:
            nu: True anomaly (rad), uses current if None

        Returns:
            Tuple of (position, velocity) vectors in inertial frame
        """
        if nu is None:
            nu = self.nu

        r = self.radius(nu)
        h = self.specific_angular_momentum()

        # Position in perifocal frame
        r_pf = r * np.array([np.cos(nu), np.sin(nu), 0])

        # Velocity in perifocal frame
        v_pf = (self.mu / h) * np.array([
            -np.sin(nu),
            self.e + np.cos(nu),
            0
        ])

        # Rotation matrix from perifocal to inertial frame
        R = self._perifocal_to_inertial_matrix()

        return R @ r_pf, R @ v_pf

    def _perifocal_to_inertial_matrix(self) -> np.ndarray:
        """Calculate rotation matrix from perifocal to inertial frame."""
        cos_O = np.cos(self.Omega)
        sin_O = np.sin(self.Omega)
        cos_i = np.cos(self.i)
        sin_i = np.sin(self.i)
        cos_w = np.cos(self.omega)
        sin_w = np.sin(self.omega)

        R = np.array([
            [cos_O*cos_w - sin_O*sin_w*cos_i, -cos_O*sin_w - sin_O*cos_w*cos_i, sin_O*sin_i],
            [sin_O*cos_w + cos_O*sin_w*cos_i, -sin_O*sin_w + cos_O*cos_w*cos_i, -cos_O*sin_i],
            [sin_w*sin_i, cos_w*sin_i, cos_i]
        ])
        return R

    def propagate(self, dt: float):
        """
        Propagate orbit forward in time.

        Args:
            dt: Time step (s)
        """
        # Update mean anomaly
        n = self.mean_motion()
        self.M += n * dt
        self.M = self.M % (2 * np.pi)

        # Convert to eccentric and true anomaly
        self.E = self._mean_to_eccentric_anomaly(self.M)
        self.nu = self._eccentric_to_true_anomaly(self.E)

        self.time += dt

    def trajectory(
        self,
        n_points: int = 100,
        start_nu: float = 0.0,
        end_nu: float = 2 * np.pi,
    ) -> np.ndarray:
        """
        Generate trajectory points for visualization.

        Args:
            n_points: Number of points
            start_nu: Starting true anomaly
            end_nu: Ending true anomaly

        Returns:
            Array of position vectors (n_points x 3)
        """
        nu_values = np.linspace(start_nu, end_nu, n_points)
        positions = np.array([self.state_vectors(nu)[0] for nu in nu_values])
        return positions

    @classmethod
    def from_state_vectors(
        cls,
        position: ArrayLike,
        velocity: ArrayLike,
        central_mass: float,
        G: float = CONSTANTS['G'],
    ) -> 'KeplerianOrbit':
        """
        Create orbit from position and velocity vectors.

        Args:
            position: Position vector (m)
            velocity: Velocity vector (m/s)
            central_mass: Mass of central body (kg)
            G: Gravitational constant

        Returns:
            KeplerianOrbit instance
        """
        r_vec = np.array(position)
        v_vec = np.array(velocity)
        mu = G * central_mass

        r = np.linalg.norm(r_vec)
        v = np.linalg.norm(v_vec)

        # Specific angular momentum
        h_vec = np.cross(r_vec, v_vec)
        h = np.linalg.norm(h_vec)

        # Eccentricity vector
        e_vec = np.cross(v_vec, h_vec) / mu - r_vec / r
        e = np.linalg.norm(e_vec)

        # Semi-major axis
        energy = v**2 / 2 - mu / r
        if abs(energy) < 1e-15:
            a = float('inf')  # Parabolic
        else:
            a = -mu / (2 * energy)

        # Inclination
        i = np.arccos(h_vec[2] / h)

        # Node vector
        n_vec = np.cross([0, 0, 1], h_vec)
        n = np.linalg.norm(n_vec)

        # Longitude of ascending node
        if n > 1e-15:
            Omega = np.arccos(n_vec[0] / n)
            if n_vec[1] < 0:
                Omega = 2 * np.pi - Omega
        else:
            Omega = 0.0

        # Argument of periapsis
        if n > 1e-15 and e > 1e-15:
            omega = np.arccos(np.dot(n_vec, e_vec) / (n * e))
            if e_vec[2] < 0:
                omega = 2 * np.pi - omega
        else:
            omega = 0.0

        # True anomaly
        if e > 1e-15:
            nu = np.arccos(np.dot(e_vec, r_vec) / (e * r))
            if np.dot(r_vec, v_vec) < 0:
                nu = 2 * np.pi - nu
        else:
            nu = 0.0

        return cls(
            central_mass=central_mass,
            semi_major_axis=a,
            eccentricity=e,
            inclination=i,
            arg_periapsis=omega,
            long_asc_node=Omega,
            true_anomaly=nu,
            G=G,
        )


class TwoBodyProblem(BaseClass):
    """
    Two-body gravitational problem with reduced mass formulation.

    Models the motion of two bodies under mutual gravitational attraction,
    reducing to an equivalent one-body problem with reduced mass.

    Args:
        mass1: Mass of first body (kg)
        mass2: Mass of second body (kg)
        position1: Initial position of first body
        velocity1: Initial velocity of first body
        position2: Initial position of second body
        velocity2: Initial velocity of second body
        G: Gravitational constant

    Examples:
        >>> # Earth-Moon system
        >>> earth_mass = 5.972e24
        >>> moon_mass = 7.34e22
        >>> two_body = TwoBodyProblem(
        ...     mass1=earth_mass,
        ...     mass2=moon_mass,
        ...     position1=[0, 0, 0],
        ...     velocity1=[0, 0, 0],
        ...     position2=[3.84e8, 0, 0],
        ...     velocity2=[0, 1022, 0]
        ... )
    """

    def __init__(
        self,
        mass1: float,
        mass2: float,
        position1: ArrayLike,
        velocity1: ArrayLike,
        position2: ArrayLike,
        velocity2: ArrayLike,
        G: float = CONSTANTS['G'],
    ):
        super().__init__()

        self.m1 = validate_positive(mass1, "mass1")
        self.m2 = validate_positive(mass2, "mass2")
        self.G = validate_positive(G, "G")

        self.r1 = validate_vector(position1, 3, "position1").copy()
        self.v1 = validate_vector(velocity1, 3, "velocity1").copy()
        self.r2 = validate_vector(position2, 3, "position2").copy()
        self.v2 = validate_vector(velocity2, 3, "velocity2").copy()

        # Derived quantities
        self.total_mass = self.m1 + self.m2
        self.reduced_mass = self.m1 * self.m2 / self.total_mass
        self.mu = self.G * self.total_mass

        self.time = 0.0
        self._history = {
            'time': [0.0],
            'r1': [self.r1.copy()],
            'r2': [self.r2.copy()],
            'v1': [self.v1.copy()],
            'v2': [self.v2.copy()],
        }

    def center_of_mass(self) -> np.ndarray:
        """Calculate center of mass position."""
        return (self.m1 * self.r1 + self.m2 * self.r2) / self.total_mass

    def center_of_mass_velocity(self) -> np.ndarray:
        """Calculate center of mass velocity."""
        return (self.m1 * self.v1 + self.m2 * self.v2) / self.total_mass

    def relative_position(self) -> np.ndarray:
        """Calculate relative position r = r2 - r1."""
        return self.r2 - self.r1

    def relative_velocity(self) -> np.ndarray:
        """Calculate relative velocity v = v2 - v1."""
        return self.v2 - self.v1

    def separation(self) -> float:
        """Calculate distance between bodies."""
        return np.linalg.norm(self.relative_position())

    def total_energy(self) -> float:
        """
        Calculate total mechanical energy.

        E = ½m₁v₁² + ½m₂v₂² - Gm₁m₂/r
        """
        KE1 = 0.5 * self.m1 * np.dot(self.v1, self.v1)
        KE2 = 0.5 * self.m2 * np.dot(self.v2, self.v2)
        PE = -self.G * self.m1 * self.m2 / self.separation()
        return KE1 + KE2 + PE

    def angular_momentum(self) -> np.ndarray:
        """
        Calculate total angular momentum vector.

        L = m₁(r₁ × v₁) + m₂(r₂ × v₂)
        """
        return (self.m1 * np.cross(self.r1, self.v1) +
                self.m2 * np.cross(self.r2, self.v2))

    def update(self, dt: float):
        """
        Update positions and velocities using RK4 integration.

        Args:
            dt: Time step (s)
        """
        def acceleration1(r1, r2):
            r = r2 - r1
            r_mag = np.linalg.norm(r)
            return self.G * self.m2 * r / r_mag**3

        def acceleration2(r1, r2):
            r = r1 - r2
            r_mag = np.linalg.norm(r)
            return self.G * self.m1 * r / r_mag**3

        # RK4 integration
        # k1
        a1_1 = acceleration1(self.r1, self.r2)
        a2_1 = acceleration2(self.r1, self.r2)
        k1_r1 = self.v1
        k1_v1 = a1_1
        k1_r2 = self.v2
        k1_v2 = a2_1

        # k2
        r1_2 = self.r1 + 0.5 * dt * k1_r1
        v1_2 = self.v1 + 0.5 * dt * k1_v1
        r2_2 = self.r2 + 0.5 * dt * k1_r2
        v2_2 = self.v2 + 0.5 * dt * k1_v2
        a1_2 = acceleration1(r1_2, r2_2)
        a2_2 = acceleration2(r1_2, r2_2)
        k2_r1 = v1_2
        k2_v1 = a1_2
        k2_r2 = v2_2
        k2_v2 = a2_2

        # k3
        r1_3 = self.r1 + 0.5 * dt * k2_r1
        v1_3 = self.v1 + 0.5 * dt * k2_v1
        r2_3 = self.r2 + 0.5 * dt * k2_r2
        v2_3 = self.v2 + 0.5 * dt * k2_v2
        a1_3 = acceleration1(r1_3, r2_3)
        a2_3 = acceleration2(r1_3, r2_3)
        k3_r1 = v1_3
        k3_v1 = a1_3
        k3_r2 = v2_3
        k3_v2 = a2_3

        # k4
        r1_4 = self.r1 + dt * k3_r1
        v1_4 = self.v1 + dt * k3_v1
        r2_4 = self.r2 + dt * k3_r2
        v2_4 = self.v2 + dt * k3_v2
        a1_4 = acceleration1(r1_4, r2_4)
        a2_4 = acceleration2(r1_4, r2_4)
        k4_r1 = v1_4
        k4_v1 = a1_4
        k4_r2 = v2_4
        k4_v2 = a2_4

        # Update
        self.r1 += (dt / 6) * (k1_r1 + 2*k2_r1 + 2*k3_r1 + k4_r1)
        self.v1 += (dt / 6) * (k1_v1 + 2*k2_v1 + 2*k3_v1 + k4_v1)
        self.r2 += (dt / 6) * (k1_r2 + 2*k2_r2 + 2*k3_r2 + k4_r2)
        self.v2 += (dt / 6) * (k1_v2 + 2*k2_v2 + 2*k3_v2 + k4_v2)

        self.time += dt

        self._history['time'].append(self.time)
        self._history['r1'].append(self.r1.copy())
        self._history['r2'].append(self.r2.copy())
        self._history['v1'].append(self.v1.copy())
        self._history['v2'].append(self.v2.copy())

    def to_keplerian(self) -> KeplerianOrbit:
        """
        Convert to equivalent Keplerian orbit.

        Returns:
            KeplerianOrbit representing the relative motion
        """
        # Use relative position and velocity
        r_rel = self.relative_position()
        v_rel = self.relative_velocity()

        return KeplerianOrbit.from_state_vectors(
            position=r_rel,
            velocity=v_rel,
            central_mass=self.total_mass,
            G=self.G,
        )


class ThreeBodyProblem(BaseClass):
    """
    Restricted three-body problem.

    Models the motion of a small test mass in the gravitational field
    of two massive bodies (primaries) in circular orbit about their
    center of mass. Includes calculation of Lagrange points.

    Args:
        mass1: Mass of primary 1 (kg)
        mass2: Mass of primary 2 (kg)
        separation: Distance between primaries (m)
        G: Gravitational constant

    Examples:
        >>> # Sun-Earth-spacecraft system
        >>> tbp = ThreeBodyProblem(
        ...     mass1=1.989e30,  # Sun
        ...     mass2=5.972e24,   # Earth
        ...     separation=1.496e11  # 1 AU
        ... )
        >>> L1 = tbp.lagrange_point(1)
    """

    def __init__(
        self,
        mass1: float,
        mass2: float,
        separation: float,
        G: float = CONSTANTS['G'],
    ):
        super().__init__()

        self.m1 = validate_positive(mass1, "mass1")
        self.m2 = validate_positive(mass2, "mass2")
        self.R = validate_positive(separation, "separation")
        self.G = validate_positive(G, "G")

        self.total_mass = self.m1 + self.m2
        self.mu = self.m2 / self.total_mass  # Mass ratio

        # Angular velocity of rotating frame
        self.omega = np.sqrt(self.G * self.total_mass / self.R**3)

        # Positions of primaries in rotating frame (centered on COM)
        self.r1 = np.array([-self.mu * self.R, 0, 0])
        self.r2 = np.array([(1 - self.mu) * self.R, 0, 0])

    def effective_potential(self, position: ArrayLike) -> float:
        """
        Calculate effective potential in rotating frame.

        U_eff = -Gm₁/r₁ - Gm₂/r₂ - ½Ω²(x² + y²)

        Args:
            position: Position in rotating frame

        Returns:
            Effective potential (J/kg)
        """
        r = np.array(position)
        r1_dist = np.linalg.norm(r - self.r1)
        r2_dist = np.linalg.norm(r - self.r2)

        gravitational = -self.G * self.m1 / r1_dist - self.G * self.m2 / r2_dist
        centrifugal = -0.5 * self.omega**2 * (r[0]**2 + r[1]**2)

        return gravitational + centrifugal

    def acceleration_rotating(
        self,
        position: ArrayLike,
        velocity: ArrayLike,
    ) -> np.ndarray:
        """
        Calculate acceleration in rotating frame.

        Includes gravitational, Coriolis, and centrifugal terms.

        Args:
            position: Position in rotating frame
            velocity: Velocity in rotating frame

        Returns:
            Acceleration vector
        """
        r = np.array(position)
        v = np.array(velocity)

        # Gravitational acceleration
        r1_vec = r - self.r1
        r2_vec = r - self.r2
        r1_dist = np.linalg.norm(r1_vec)
        r2_dist = np.linalg.norm(r2_vec)

        a_grav = (-self.G * self.m1 * r1_vec / r1_dist**3 -
                  self.G * self.m2 * r2_vec / r2_dist**3)

        # Coriolis: -2ω × v
        omega_vec = np.array([0, 0, self.omega])
        a_coriolis = -2 * np.cross(omega_vec, v)

        # Centrifugal: -ω × (ω × r)
        a_centrifugal = -np.cross(omega_vec, np.cross(omega_vec, r))

        return a_grav + a_coriolis + a_centrifugal

    def lagrange_point(self, point: int) -> np.ndarray:
        """
        Calculate position of Lagrange point.

        Args:
            point: Lagrange point number (1-5)

        Returns:
            Position of Lagrange point in rotating frame
        """
        if point not in [1, 2, 3, 4, 5]:
            raise ValidationError("point", point, "must be 1, 2, 3, 4, or 5")

        mu = self.mu

        if point == 1:
            # L1: Between the two bodies
            # Solve: x - (1-μ)(x+μ)/|x+μ|³ - μ(x-1+μ)/|x-1+μ|³ = 0
            # Approximate solution
            r_hill = self.R * (mu / 3)**(1/3)
            x = self.R * (1 - mu) - r_hill
            return np.array([x, 0, 0])

        elif point == 2:
            # L2: Beyond the smaller body
            r_hill = self.R * (mu / 3)**(1/3)
            x = self.R * (1 - mu) + r_hill
            return np.array([x, 0, 0])

        elif point == 3:
            # L3: Beyond the larger body
            x = -self.R * (1 + 5 * mu / 12)
            return np.array([x, 0, 0])

        elif point == 4:
            # L4: Leading equilateral point
            x = self.R * (0.5 - mu)
            y = self.R * np.sqrt(3) / 2
            return np.array([x, y, 0])

        else:  # point == 5
            # L5: Trailing equilateral point
            x = self.R * (0.5 - mu)
            y = -self.R * np.sqrt(3) / 2
            return np.array([x, y, 0])

    def jacobi_constant(self, position: ArrayLike, velocity: ArrayLike) -> float:
        """
        Calculate Jacobi constant (integral of motion in rotating frame).

        C_J = -2U_eff - v²

        Args:
            position: Position in rotating frame
            velocity: Velocity in rotating frame

        Returns:
            Jacobi constant
        """
        U_eff = self.effective_potential(position)
        v_sq = np.dot(velocity, velocity)
        return -2 * U_eff - v_sq

    def hill_sphere_radius(self, body: int = 2) -> float:
        """
        Calculate Hill sphere radius for one of the bodies.

        r_H = R * (m/3M)^(1/3)

        Args:
            body: Body number (1 or 2)

        Returns:
            Hill sphere radius (m)
        """
        if body == 1:
            m = self.m1
        else:
            m = self.m2

        return self.R * (m / (3 * self.total_mass))**(1/3)


class OrbitalManeuver(BaseClass):
    """
    Orbital maneuver calculations.

    Provides methods for calculating delta-v requirements for various
    orbital maneuvers including Hohmann transfers, bi-elliptic transfers,
    plane changes, and gravity assists.

    Args:
        central_mass: Mass of central body (kg)
        G: Gravitational constant

    Examples:
        >>> maneuver = OrbitalManeuver(central_mass=5.972e24)  # Earth
        >>> dv = maneuver.hohmann_delta_v(r1=6.671e6, r2=4.22e7)  # LEO to GEO
    """

    def __init__(
        self,
        central_mass: float,
        G: float = CONSTANTS['G'],
    ):
        super().__init__()
        self.M = validate_positive(central_mass, "central_mass")
        self.G = validate_positive(G, "G")
        self.mu = self.G * self.M

    def circular_velocity(self, r: float) -> float:
        """
        Calculate circular orbit velocity at radius r.

        v_c = √(μ/r)
        """
        return np.sqrt(self.mu / r)

    def escape_velocity(self, r: float) -> float:
        """
        Calculate escape velocity at radius r.

        v_esc = √(2μ/r) = √2 * v_c
        """
        return np.sqrt(2 * self.mu / r)

    def hohmann_delta_v(self, r1: float, r2: float) -> Tuple[float, float, float]:
        """
        Calculate delta-v for Hohmann transfer between circular orbits.

        Args:
            r1: Initial orbit radius (m)
            r2: Final orbit radius (m)

        Returns:
            Tuple of (delta_v1, delta_v2, total_delta_v) in m/s
        """
        # Semi-major axis of transfer orbit
        a_transfer = (r1 + r2) / 2

        # Velocities
        v1_circular = self.circular_velocity(r1)
        v2_circular = self.circular_velocity(r2)

        # Transfer orbit velocities at periapsis and apoapsis
        v1_transfer = np.sqrt(self.mu * (2/r1 - 1/a_transfer))
        v2_transfer = np.sqrt(self.mu * (2/r2 - 1/a_transfer))

        # Delta-v magnitudes
        dv1 = abs(v1_transfer - v1_circular)
        dv2 = abs(v2_circular - v2_transfer)

        return dv1, dv2, dv1 + dv2

    def hohmann_time(self, r1: float, r2: float) -> float:
        """
        Calculate transfer time for Hohmann transfer.

        T_transfer = π√((r1+r2)³/(8μ))

        Args:
            r1: Initial orbit radius (m)
            r2: Final orbit radius (m)

        Returns:
            Transfer time (s)
        """
        a_transfer = (r1 + r2) / 2
        return np.pi * np.sqrt(a_transfer**3 / self.mu)

    def bi_elliptic_delta_v(
        self,
        r1: float,
        r2: float,
        r_intermediate: float,
    ) -> Tuple[float, float, float, float]:
        """
        Calculate delta-v for bi-elliptic transfer.

        More efficient than Hohmann for r2/r1 > 11.94.

        Args:
            r1: Initial orbit radius (m)
            r2: Final orbit radius (m)
            r_intermediate: Intermediate apoapsis radius (m)

        Returns:
            Tuple of (dv1, dv2, dv3, total) in m/s
        """
        # First transfer ellipse
        a1 = (r1 + r_intermediate) / 2
        v1_circular = self.circular_velocity(r1)
        v1_transfer = np.sqrt(self.mu * (2/r1 - 1/a1))
        dv1 = abs(v1_transfer - v1_circular)

        # At intermediate point
        v_intermediate_1 = np.sqrt(self.mu * (2/r_intermediate - 1/a1))

        # Second transfer ellipse
        a2 = (r_intermediate + r2) / 2
        v_intermediate_2 = np.sqrt(self.mu * (2/r_intermediate - 1/a2))
        dv2 = abs(v_intermediate_2 - v_intermediate_1)

        # Final circularization
        v2_transfer = np.sqrt(self.mu * (2/r2 - 1/a2))
        v2_circular = self.circular_velocity(r2)
        dv3 = abs(v2_circular - v2_transfer)

        return dv1, dv2, dv3, dv1 + dv2 + dv3

    def plane_change_delta_v(
        self,
        v: float,
        angle: float,
    ) -> float:
        """
        Calculate delta-v for simple plane change.

        Δv = 2v sin(θ/2)

        Args:
            v: Orbital velocity (m/s)
            angle: Plane change angle (rad)

        Returns:
            Required delta-v (m/s)
        """
        return 2 * v * np.sin(angle / 2)

    def gravity_assist_delta_v(
        self,
        v_infinity: float,
        periapsis: float,
        assist_body_mass: float,
        turn_angle: float,
    ) -> float:
        """
        Calculate velocity change from gravity assist.

        Args:
            v_infinity: Hyperbolic excess velocity (m/s)
            periapsis: Closest approach distance (m)
            assist_body_mass: Mass of assisting body (kg)
            turn_angle: Trajectory turn angle (rad)

        Returns:
            Magnitude of velocity change (m/s)
        """
        return 2 * v_infinity * np.sin(turn_angle / 2)


class EscapeTrajectory(BaseClass):
    """
    Escape trajectory calculations.

    Calculates parameters for hyperbolic escape trajectories.

    Args:
        central_mass: Mass of central body (kg)
        G: Gravitational constant

    Examples:
        >>> escape = EscapeTrajectory(central_mass=5.972e24)  # Earth
        >>> v_inf = escape.required_velocity(r=6.671e6, target_v_inf=3000)
    """

    def __init__(
        self,
        central_mass: float,
        G: float = CONSTANTS['G'],
    ):
        super().__init__()
        self.M = validate_positive(central_mass, "central_mass")
        self.G = validate_positive(G, "G")
        self.mu = self.G * self.M

    def escape_velocity(self, r: float) -> float:
        """
        Calculate escape velocity at radius r.

        v_esc = √(2μ/r)
        """
        return np.sqrt(2 * self.mu / r)

    def required_velocity(
        self,
        r: float,
        target_v_inf: float,
    ) -> float:
        """
        Calculate required velocity for given hyperbolic excess velocity.

        v² = v_inf² + v_esc²

        Args:
            r: Starting radius (m)
            target_v_inf: Desired velocity at infinity (m/s)

        Returns:
            Required initial velocity (m/s)
        """
        v_esc = self.escape_velocity(r)
        return np.sqrt(target_v_inf**2 + v_esc**2)

    def hyperbolic_excess_velocity(
        self,
        r: float,
        v: float,
    ) -> float:
        """
        Calculate hyperbolic excess velocity from initial conditions.

        v_inf = √(v² - v_esc²)

        Args:
            r: Starting radius (m)
            v: Initial velocity (m/s)

        Returns:
            Hyperbolic excess velocity (m/s), or 0 if bound
        """
        v_esc = self.escape_velocity(r)
        if v < v_esc:
            return 0.0
        return np.sqrt(v**2 - v_esc**2)

    def c3(self, r: float, v: float) -> float:
        """
        Calculate characteristic energy C3.

        C3 = v_inf² = v² - 2μ/r

        Args:
            r: Starting radius (m)
            v: Initial velocity (m/s)

        Returns:
            C3 in m²/s²
        """
        return v**2 - 2 * self.mu / r

    def hyperbolic_elements(
        self,
        position: ArrayLike,
        velocity: ArrayLike,
    ) -> Dict[str, float]:
        """
        Calculate hyperbolic trajectory elements.

        Args:
            position: Position vector (m)
            velocity: Velocity vector (m/s)

        Returns:
            Dictionary with hyperbolic elements
        """
        r = np.array(position)
        v = np.array(velocity)

        r_mag = np.linalg.norm(r)
        v_mag = np.linalg.norm(v)

        # Specific energy
        energy = v_mag**2 / 2 - self.mu / r_mag

        if energy <= 0:
            raise PhysicsError("Orbit is not hyperbolic (energy <= 0)")

        # Semi-major axis (negative for hyperbola)
        a = -self.mu / (2 * energy)

        # Specific angular momentum
        h = np.cross(r, v)
        h_mag = np.linalg.norm(h)

        # Eccentricity
        e = np.sqrt(1 + 2 * energy * h_mag**2 / self.mu**2)

        # Hyperbolic excess velocity
        v_inf = np.sqrt(2 * energy)

        # Semi-latus rectum
        p = h_mag**2 / self.mu

        # Periapsis
        r_p = a * (1 - e)

        # Turn angle (asymptotic deflection)
        turn_angle = 2 * np.arcsin(1 / e)

        return {
            'semi_major_axis': a,
            'eccentricity': e,
            'energy': energy,
            'angular_momentum': h_mag,
            'v_infinity': v_inf,
            'periapsis': r_p,
            'turn_angle': turn_angle,
            'c3': v_inf**2,
        }
