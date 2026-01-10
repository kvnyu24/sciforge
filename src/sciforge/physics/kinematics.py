"""
Kinematics primitives for motion analysis.

This module provides classes for analyzing and simulating different types
of motion including projectile motion, circular motion, relative motion,
and curvilinear motion along arbitrary paths.
"""

import numpy as np
from typing import Optional, Tuple, Callable, Union, List
from numpy.typing import ArrayLike

from ..core.base import BaseClass
from ..core.constants import CONSTANTS
from ..core.utils import (
    validate_array,
    validate_vector,
    validate_positive,
    validate_non_negative,
    validate_finite,
    validate_callable,
    normalize_vector,
)
from ..core.exceptions import ValidationError, PhysicsError
from .base import DynamicalSystem


class ProjectileMotion(DynamicalSystem):
    """
    2D/3D projectile motion with optional air resistance.

    Models the trajectory of a projectile under gravity with optional
    quadratic drag. Supports both 2D and 3D simulations.

    Args:
        mass: Mass of the projectile (kg)
        position: Initial position vector [x, y] or [x, y, z] (m)
        velocity: Initial velocity vector (m/s)
        gravity: Gravitational acceleration (m/s²), default 9.81
        drag_coeff: Drag coefficient (dimensionless), default 0
        cross_section: Cross-sectional area for drag (m²), default 1
        air_density: Air density (kg/m³), default 1.225 (sea level)

    Attributes:
        trajectory: List of position vectors over time
        time_history: List of time values

    Examples:
        >>> proj = ProjectileMotion(
        ...     mass=1.0,
        ...     position=[0, 0, 0],
        ...     velocity=[10, 10, 0],
        ...     gravity=9.81
        ... )
        >>> proj.simulate(dt=0.01, t_max=2.0)
        >>> print(proj.range())  # Horizontal range
    """

    _history_fields = ["time", "position", "velocity", "acceleration"]

    def __init__(
        self,
        mass: float,
        position: ArrayLike,
        velocity: ArrayLike,
        gravity: float = 9.81,
        drag_coeff: float = 0.0,
        cross_section: float = 1.0,
        air_density: float = 1.225,
    ):
        # Validate inputs
        mass = validate_positive(mass, "mass")
        position = np.array(position, dtype=float)
        velocity = np.array(velocity, dtype=float)

        if len(position) not in (2, 3):
            raise ValidationError("position", position.shape, "must be 2D or 3D vector")
        if len(velocity) != len(position):
            raise ValidationError("velocity", velocity.shape,
                                f"must match position dimension {position.shape}")

        gravity = validate_non_negative(gravity, "gravity")
        drag_coeff = validate_non_negative(drag_coeff, "drag_coeff")
        cross_section = validate_positive(cross_section, "cross_section")
        air_density = validate_non_negative(air_density, "air_density")

        super().__init__(mass, position, velocity)

        self.gravity = gravity
        self.drag_coeff = drag_coeff
        self.cross_section = cross_section
        self.air_density = air_density
        self.dim = len(position)

        # Initialize trajectory storage
        self.trajectory = [position.copy()]
        self.velocity_history = [velocity.copy()]
        self.time_history = [0.0]

    def _gravity_vector(self) -> np.ndarray:
        """Return gravity vector (pointing down in y-direction)."""
        g = np.zeros(self.dim)
        g[1] = -self.gravity  # y-component
        return g

    def _drag_force(self, velocity: np.ndarray) -> np.ndarray:
        """Calculate drag force: F_d = -0.5 * C_d * rho * A * |v| * v"""
        if self.drag_coeff == 0:
            return np.zeros(self.dim)

        speed = np.linalg.norm(velocity)
        if speed < 1e-10:
            return np.zeros(self.dim)

        drag_magnitude = 0.5 * self.drag_coeff * self.air_density * self.cross_section * speed**2
        return -drag_magnitude * velocity / speed

    def acceleration(self, position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """Calculate total acceleration at given state."""
        gravity_force = self.mass * self._gravity_vector()
        drag_force = self._drag_force(velocity)
        return (gravity_force + drag_force) / self.mass

    def update(self, dt: float):
        """
        Update projectile state using RK4 integration.

        Args:
            dt: Time step (s)
        """
        # RK4 integration
        pos = self.position
        vel = self.velocity

        # k1
        a1 = self.acceleration(pos, vel)
        k1_v = a1
        k1_x = vel

        # k2
        pos2 = pos + 0.5 * dt * k1_x
        vel2 = vel + 0.5 * dt * k1_v
        a2 = self.acceleration(pos2, vel2)
        k2_v = a2
        k2_x = vel2

        # k3
        pos3 = pos + 0.5 * dt * k2_x
        vel3 = vel + 0.5 * dt * k2_v
        a3 = self.acceleration(pos3, vel3)
        k3_v = a3
        k3_x = vel3

        # k4
        pos4 = pos + dt * k3_x
        vel4 = vel + dt * k3_v
        a4 = self.acceleration(pos4, vel4)
        k4_v = a4
        k4_x = vel4

        # Update state
        self.position = pos + (dt / 6.0) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
        self.velocity = vel + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        self.time += dt

        # Store history
        self.trajectory.append(self.position.copy())
        self.velocity_history.append(self.velocity.copy())
        self.time_history.append(self.time)

    def simulate(
        self,
        dt: float = 0.01,
        t_max: float = 10.0,
        stop_on_ground: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate projectile motion until max time or ground impact.

        Args:
            dt: Time step (s)
            t_max: Maximum simulation time (s)
            stop_on_ground: If True, stop when y <= 0

        Returns:
            Tuple of (times, positions) arrays
        """
        while self.time < t_max:
            if stop_on_ground and self.position[1] < 0:
                break
            self.update(dt)

        return np.array(self.time_history), np.array(self.trajectory)

    def range(self) -> float:
        """Calculate the horizontal range of the projectile."""
        trajectory = np.array(self.trajectory)
        return trajectory[-1, 0] - trajectory[0, 0]

    def max_height(self) -> float:
        """Calculate the maximum height reached."""
        trajectory = np.array(self.trajectory)
        return np.max(trajectory[:, 1])

    def time_of_flight(self) -> float:
        """Calculate total time of flight."""
        return self.time_history[-1]

    def impact_velocity(self) -> np.ndarray:
        """Return the velocity at impact (final velocity)."""
        return self.velocity_history[-1].copy()

    def analytical_range(self) -> float:
        """
        Calculate analytical range (only valid for no drag, flat ground).

        Returns:
            Theoretical range without drag
        """
        v0 = np.linalg.norm(self.velocity_history[0])
        # Angle from horizontal
        theta = np.arctan2(self.velocity_history[0][1], self.velocity_history[0][0])
        return (v0**2 * np.sin(2 * theta)) / self.gravity

    def analytical_max_height(self) -> float:
        """
        Calculate analytical max height (only valid for no drag).

        Returns:
            Theoretical max height without drag
        """
        v0y = self.velocity_history[0][1]
        return (v0y**2) / (2 * self.gravity)

    def analytical_time_of_flight(self) -> float:
        """
        Calculate analytical time of flight (only valid for no drag, flat ground).

        Returns:
            Theoretical time of flight without drag
        """
        v0y = self.velocity_history[0][1]
        return (2 * v0y) / self.gravity


class CircularMotion(DynamicalSystem):
    """
    Uniform and non-uniform circular motion.

    Models motion along a circular path with constant or varying angular velocity.

    Args:
        mass: Mass of the object (kg)
        radius: Radius of circular path (m)
        omega0: Initial angular velocity (rad/s)
        alpha: Angular acceleration (rad/s²), default 0 for uniform motion
        theta0: Initial angle from positive x-axis (rad), default 0
        center: Center of circular path, default [0, 0, 0]

    Attributes:
        theta: Current angle (rad)
        omega: Current angular velocity (rad/s)

    Examples:
        >>> cm = CircularMotion(mass=1.0, radius=2.0, omega0=np.pi)
        >>> for _ in range(100):
        ...     cm.update(dt=0.01)
        >>> print(cm.centripetal_acceleration())
    """

    _history_fields = ["time", "theta", "omega", "position", "velocity"]

    def __init__(
        self,
        mass: float,
        radius: float,
        omega0: float,
        alpha: float = 0.0,
        theta0: float = 0.0,
        center: Optional[ArrayLike] = None,
    ):
        # Validate inputs
        mass = validate_positive(mass, "mass")
        radius = validate_positive(radius, "radius")
        omega0 = validate_finite(omega0, "omega0")
        alpha = validate_finite(alpha, "alpha")
        theta0 = validate_finite(theta0, "theta0")

        self.radius = radius
        self.theta = theta0
        self.omega = omega0
        self.alpha = alpha
        self.center = np.zeros(3) if center is None else validate_vector(center, 3, "center")

        # Calculate initial position and velocity
        position = self._position_from_theta(theta0)
        velocity = self._velocity_from_state(theta0, omega0)

        super().__init__(mass, position, velocity)

        self.history = {
            'time': [0.0],
            'theta': [theta0],
            'omega': [omega0],
            'position': [position.copy()],
            'velocity': [velocity.copy()],
        }

    def _position_from_theta(self, theta: float) -> np.ndarray:
        """Calculate position from angle."""
        return self.center + self.radius * np.array([
            np.cos(theta),
            np.sin(theta),
            0.0
        ])

    def _velocity_from_state(self, theta: float, omega: float) -> np.ndarray:
        """Calculate velocity from angle and angular velocity."""
        # v = omega * r * tangent direction
        return self.radius * omega * np.array([
            -np.sin(theta),
            np.cos(theta),
            0.0
        ])

    def update(self, dt: float):
        """
        Update circular motion state.

        Args:
            dt: Time step (s)
        """
        # Update angular velocity
        self.omega += self.alpha * dt

        # Update angle
        self.theta += self.omega * dt

        # Update position and velocity
        self.position = self._position_from_theta(self.theta)
        self.velocity = self._velocity_from_state(self.theta, self.omega)

        self.time += dt

        # Store history
        self.history['time'].append(self.time)
        self.history['theta'].append(self.theta)
        self.history['omega'].append(self.omega)
        self.history['position'].append(self.position.copy())
        self.history['velocity'].append(self.velocity.copy())

    def period(self) -> float:
        """
        Calculate the period of rotation.

        Returns:
            Period in seconds (only valid for uniform motion)
        """
        if abs(self.alpha) > 1e-10:
            raise PhysicsError("Period undefined for non-uniform circular motion")
        if abs(self.omega) < 1e-10:
            raise PhysicsError("Period undefined for zero angular velocity")
        return (2 * np.pi) / abs(self.omega)

    def frequency(self) -> float:
        """
        Calculate the frequency of rotation.

        Returns:
            Frequency in Hz (only valid for uniform motion)
        """
        return 1.0 / self.period()

    def speed(self) -> float:
        """Calculate the tangential speed."""
        return abs(self.omega * self.radius)

    def centripetal_acceleration(self) -> float:
        """Calculate the centripetal acceleration magnitude."""
        return self.omega**2 * self.radius

    def centripetal_force(self) -> float:
        """Calculate the centripetal force magnitude."""
        return self.mass * self.centripetal_acceleration()

    def tangential_acceleration(self) -> float:
        """Calculate the tangential acceleration magnitude."""
        return abs(self.alpha * self.radius)

    def total_acceleration(self) -> float:
        """Calculate the total acceleration magnitude."""
        ac = self.centripetal_acceleration()
        at = self.tangential_acceleration()
        return np.sqrt(ac**2 + at**2)

    def kinetic_energy(self) -> float:
        """Calculate kinetic energy."""
        return 0.5 * self.mass * self.speed()**2

    def angular_momentum(self) -> float:
        """Calculate angular momentum magnitude about the center."""
        return self.mass * self.radius * self.speed()


class ReferenceFrame(BaseClass):
    """
    Reference frame for coordinate transformations.

    Represents a reference frame that can be translated and rotated
    relative to an inertial frame, with support for non-inertial effects.

    Args:
        origin: Position of frame origin in inertial frame
        velocity: Velocity of frame origin in inertial frame
        angular_velocity: Angular velocity of frame (rad/s), rotation about z-axis

    Examples:
        >>> # Frame moving at constant velocity
        >>> frame = ReferenceFrame(
        ...     origin=[100, 0, 0],
        ...     velocity=[10, 0, 0]
        ... )
        >>> # Transform position from inertial to moving frame
        >>> pos_moving = frame.transform_position([150, 0, 0])
    """

    def __init__(
        self,
        origin: ArrayLike = None,
        velocity: ArrayLike = None,
        angular_velocity: ArrayLike = None,
        acceleration: ArrayLike = None,
        angular_acceleration: ArrayLike = None,
    ):
        super().__init__()

        self.origin = np.zeros(3) if origin is None else validate_vector(origin, 3, "origin")
        self.velocity = np.zeros(3) if velocity is None else validate_vector(velocity, 3, "velocity")
        self.angular_velocity = np.zeros(3) if angular_velocity is None else validate_vector(angular_velocity, 3, "angular_velocity")
        self.acceleration = np.zeros(3) if acceleration is None else validate_vector(acceleration, 3, "acceleration")
        self.angular_acceleration = np.zeros(3) if angular_acceleration is None else validate_vector(angular_acceleration, 3, "angular_acceleration")

        self.time = 0.0

    def update(self, dt: float):
        """Update frame position and orientation."""
        # Update velocity
        self.velocity = self.velocity + self.acceleration * dt

        # Update position
        self.origin = self.origin + self.velocity * dt

        # Update angular velocity
        self.angular_velocity = self.angular_velocity + self.angular_acceleration * dt

        self.time += dt

    def rotation_matrix(self) -> np.ndarray:
        """
        Get current rotation matrix from inertial to this frame.

        Assumes rotation about z-axis for simplicity.
        """
        theta = np.linalg.norm(self.angular_velocity) * self.time
        if abs(theta) < 1e-10:
            return np.eye(3)

        # Rotation about z-axis
        c, s = np.cos(theta), np.sin(theta)
        return np.array([
            [c, s, 0],
            [-s, c, 0],
            [0, 0, 1]
        ])

    def transform_position(
        self,
        position_inertial: ArrayLike,
        time: Optional[float] = None,
    ) -> np.ndarray:
        """
        Transform position from inertial frame to this frame.

        Args:
            position_inertial: Position in inertial frame
            time: Time at which to evaluate (uses current time if None)

        Returns:
            Position in this reference frame
        """
        pos = validate_vector(position_inertial, 3, "position_inertial")

        # Translate to frame origin
        rel_pos = pos - self.origin

        # Rotate to frame orientation
        R = self.rotation_matrix()
        return R @ rel_pos

    def transform_velocity(
        self,
        velocity_inertial: ArrayLike,
        position_inertial: ArrayLike,
    ) -> np.ndarray:
        """
        Transform velocity from inertial frame to this frame.

        Args:
            velocity_inertial: Velocity in inertial frame
            position_inertial: Position in inertial frame

        Returns:
            Velocity in this reference frame
        """
        vel = validate_vector(velocity_inertial, 3, "velocity_inertial")
        pos = validate_vector(position_inertial, 3, "position_inertial")

        # Relative position in inertial frame
        rel_pos = pos - self.origin

        # Velocity transformation: v' = R(v - V - omega × r)
        R = self.rotation_matrix()
        omega_cross_r = np.cross(self.angular_velocity, rel_pos)
        return R @ (vel - self.velocity - omega_cross_r)

    def inverse_transform_position(self, position_frame: ArrayLike) -> np.ndarray:
        """
        Transform position from this frame to inertial frame.

        Args:
            position_frame: Position in this reference frame

        Returns:
            Position in inertial frame
        """
        pos = validate_vector(position_frame, 3, "position_frame")

        # Rotate to inertial orientation
        R = self.rotation_matrix()
        R_inv = R.T  # Orthogonal matrix

        # Translate to inertial origin
        return R_inv @ pos + self.origin


class RelativeMotion(BaseClass):
    """
    Relative motion analysis between reference frames.

    Computes relative position, velocity, and acceleration between
    objects in different reference frames, including non-inertial effects.

    Args:
        frame: Reference frame (ReferenceFrame instance)

    Examples:
        >>> frame = ReferenceFrame(velocity=[10, 0, 0])
        >>> rel = RelativeMotion(frame)
        >>> # Object at rest in inertial frame
        >>> v_rel = rel.relative_velocity([0, 0, 0], [0, 0, 0])
        >>> print(v_rel)  # Should be [-10, 0, 0]
    """

    def __init__(self, frame: ReferenceFrame):
        super().__init__()
        self.frame = frame

    def relative_position(
        self,
        position: ArrayLike,
    ) -> np.ndarray:
        """
        Calculate position relative to the moving frame.

        Args:
            position: Position in inertial frame

        Returns:
            Position in the moving reference frame
        """
        return self.frame.transform_position(position)

    def relative_velocity(
        self,
        velocity: ArrayLike,
        position: ArrayLike,
    ) -> np.ndarray:
        """
        Calculate velocity relative to the moving frame.

        Args:
            velocity: Velocity in inertial frame
            position: Position in inertial frame

        Returns:
            Velocity in the moving reference frame
        """
        return self.frame.transform_velocity(velocity, position)

    def coriolis_acceleration(
        self,
        velocity_relative: ArrayLike,
    ) -> np.ndarray:
        """
        Calculate Coriolis acceleration in the rotating frame.

        a_cor = -2 * omega × v'

        Args:
            velocity_relative: Velocity in the rotating frame

        Returns:
            Coriolis acceleration vector
        """
        v_rel = validate_vector(velocity_relative, 3, "velocity_relative")
        return -2.0 * np.cross(self.frame.angular_velocity, v_rel)

    def centrifugal_acceleration(
        self,
        position_relative: ArrayLike,
    ) -> np.ndarray:
        """
        Calculate centrifugal acceleration in the rotating frame.

        a_cf = -omega × (omega × r')

        Args:
            position_relative: Position in the rotating frame

        Returns:
            Centrifugal acceleration vector
        """
        r_rel = validate_vector(position_relative, 3, "position_relative")
        omega = self.frame.angular_velocity
        return -np.cross(omega, np.cross(omega, r_rel))

    def euler_acceleration(
        self,
        position_relative: ArrayLike,
    ) -> np.ndarray:
        """
        Calculate Euler acceleration (from angular acceleration).

        a_euler = -alpha × r'

        Args:
            position_relative: Position in the rotating frame

        Returns:
            Euler acceleration vector
        """
        r_rel = validate_vector(position_relative, 3, "position_relative")
        return -np.cross(self.frame.angular_acceleration, r_rel)

    def fictitious_acceleration(
        self,
        position_relative: ArrayLike,
        velocity_relative: ArrayLike,
    ) -> np.ndarray:
        """
        Calculate total fictitious (pseudo) acceleration in rotating frame.

        Includes Coriolis, centrifugal, and Euler accelerations.

        Args:
            position_relative: Position in the rotating frame
            velocity_relative: Velocity in the rotating frame

        Returns:
            Total fictitious acceleration vector
        """
        a_cor = self.coriolis_acceleration(velocity_relative)
        a_cf = self.centrifugal_acceleration(position_relative)
        a_euler = self.euler_acceleration(position_relative)

        # Also include linear acceleration of frame origin
        R = self.frame.rotation_matrix()
        a_trans = -R @ self.frame.acceleration

        return a_cor + a_cf + a_euler + a_trans


class CurvilinearMotion(DynamicalSystem):
    """
    Motion along an arbitrary parametric curve.

    Models motion constrained to follow a path defined by parametric
    equations r(s) where s is the arc length parameter.

    Args:
        mass: Mass of the object (kg)
        path_func: Function r(s) returning position for arc length s
        tangent_func: Function T(s) returning unit tangent vector (optional)
        s0: Initial arc length parameter
        v0: Initial speed along the path (m/s)

    Examples:
        >>> # Motion along a helix
        >>> def helix(s):
        ...     return np.array([np.cos(s), np.sin(s), 0.1*s])
        >>> motion = CurvilinearMotion(mass=1.0, path_func=helix, s0=0, v0=1.0)
    """

    _history_fields = ["time", "s", "speed", "position", "velocity"]

    def __init__(
        self,
        mass: float,
        path_func: Callable[[float], np.ndarray],
        s0: float = 0.0,
        v0: float = 0.0,
        tangent_func: Optional[Callable[[float], np.ndarray]] = None,
        acceleration_func: Optional[Callable[[float, float], float]] = None,
    ):
        mass = validate_positive(mass, "mass")
        validate_callable(path_func, "path_func")
        s0 = validate_finite(s0, "s0")
        v0 = validate_finite(v0, "v0")

        self.path_func = path_func
        self.tangent_func = tangent_func
        self.acceleration_func = acceleration_func

        self.s = s0  # Arc length parameter
        self.speed = v0  # Speed along path (ds/dt)

        # Get initial position
        position = np.array(path_func(s0))

        # Get initial velocity (tangent direction * speed)
        tangent = self._get_tangent(s0)
        velocity = v0 * tangent

        super().__init__(mass, position, velocity)

        self.history = {
            'time': [0.0],
            's': [s0],
            'speed': [v0],
            'position': [position.copy()],
            'velocity': [velocity.copy()],
        }

    def _get_tangent(self, s: float, ds: float = 1e-6) -> np.ndarray:
        """
        Calculate unit tangent vector at arc length s.

        Uses provided tangent function or numerical differentiation.
        """
        if self.tangent_func is not None:
            T = np.array(self.tangent_func(s))
            return T / np.linalg.norm(T)

        # Numerical differentiation
        r_plus = np.array(self.path_func(s + ds))
        r_minus = np.array(self.path_func(s - ds))
        dr = r_plus - r_minus
        return dr / np.linalg.norm(dr)

    def _get_curvature(self, s: float, ds: float = 1e-6) -> Tuple[float, np.ndarray]:
        """
        Calculate curvature and normal vector at arc length s.

        Returns:
            Tuple of (curvature, normal_vector)
        """
        # Get tangent vectors
        T_plus = self._get_tangent(s + ds)
        T_minus = self._get_tangent(s - ds)

        # dT/ds
        dT_ds = (T_plus - T_minus) / (2 * ds)

        # Curvature is |dT/ds|
        kappa = np.linalg.norm(dT_ds)

        # Normal vector
        if kappa > 1e-10:
            N = dT_ds / kappa
        else:
            N = np.zeros_like(dT_ds)

        return kappa, N

    def update(self, dt: float, tangential_accel: float = 0.0):
        """
        Update motion along the curve.

        Args:
            dt: Time step (s)
            tangential_accel: Tangential acceleration (m/s²)
        """
        # Get acceleration
        if self.acceleration_func is not None:
            tangential_accel = self.acceleration_func(self.s, self.speed)

        # Update speed
        self.speed += tangential_accel * dt

        # Update arc length parameter
        self.s += self.speed * dt

        # Update position and velocity
        self.position = np.array(self.path_func(self.s))
        tangent = self._get_tangent(self.s)
        self.velocity = self.speed * tangent

        self.time += dt

        # Store history
        self.history['time'].append(self.time)
        self.history['s'].append(self.s)
        self.history['speed'].append(self.speed)
        self.history['position'].append(self.position.copy())
        self.history['velocity'].append(self.velocity.copy())

    def tangential_acceleration(self) -> float:
        """Calculate tangential component of acceleration (rate of speed change)."""
        if self.acceleration_func is not None:
            return self.acceleration_func(self.s, self.speed)
        return 0.0

    def normal_acceleration(self) -> float:
        """Calculate normal (centripetal) component of acceleration."""
        kappa, _ = self._get_curvature(self.s)
        return kappa * self.speed**2

    def total_acceleration(self) -> np.ndarray:
        """Calculate total acceleration vector."""
        tangent = self._get_tangent(self.s)
        kappa, normal = self._get_curvature(self.s)

        a_t = self.tangential_acceleration()
        a_n = self.normal_acceleration()

        return a_t * tangent + a_n * normal

    def curvature(self) -> float:
        """Get curvature at current position."""
        kappa, _ = self._get_curvature(self.s)
        return kappa

    def radius_of_curvature(self) -> float:
        """Get radius of curvature at current position."""
        kappa = self.curvature()
        if kappa < 1e-10:
            return float('inf')
        return 1.0 / kappa

    def frenet_frame(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get Frenet-Serret frame (T, N, B) at current position.

        Returns:
            Tuple of (tangent, normal, binormal) unit vectors
        """
        T = self._get_tangent(self.s)
        kappa, N = self._get_curvature(self.s)

        if np.linalg.norm(N) < 1e-10:
            # Arbitrary normal when curvature is zero
            N = np.array([0, 1, 0]) if abs(T[1]) < 0.9 else np.array([1, 0, 0])
            N = N - np.dot(N, T) * T
            N = N / np.linalg.norm(N)

        B = np.cross(T, N)

        return T, N, B
