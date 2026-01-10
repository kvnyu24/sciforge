"""
Force primitives for mechanical systems.

This module provides force classes that can be composed to build
complex mechanical systems. All forces implement the __call__ interface
returning force vectors given position, velocity, and time.
"""

import numpy as np
from typing import Optional, Union, Callable
from numpy.typing import ArrayLike

from ..core.constants import CONSTANTS
from ..core.utils import (
    validate_vector,
    validate_positive,
    validate_non_negative,
    validate_finite,
    validate_array,
)
from .base import Force



class SpringForce(Force):
    """Hooke's law spring force"""
    def __init__(self, k: float, anchor: ArrayLike, rest_length: float = 0.0):
        self.k = k
        self.anchor = np.array(anchor)
        self.rest_length = rest_length
        
    def __call__(self, position: ArrayLike, velocity: Optional[ArrayLike] = None,
                time: Optional[float] = None) -> np.ndarray:
        displacement = np.array(position) - self.anchor
        stretch = np.linalg.norm(displacement) - self.rest_length
        if np.linalg.norm(displacement) < 1e-10:  # Avoid division by zero
            return np.zeros_like(displacement)
        return -self.k * stretch * displacement / np.linalg.norm(displacement)


class GravityForce(Force):
    """Constant gravitational force"""
    def __init__(self, g: float = 9.81):
        self.g = g
        
    def __call__(self, position: ArrayLike, velocity: Optional[ArrayLike] = None,
                time: Optional[float] = None) -> np.ndarray:
        return np.array([0, -self.g])  # 2D gravity


class DragForce(Force):
    """Quadratic drag force"""
    def __init__(self, drag_coeff: float):
        self.drag_coeff = drag_coeff
        
    def __call__(self, position: ArrayLike, velocity: ArrayLike,
                time: Optional[float] = None) -> np.ndarray:
        if velocity is None:
            return np.zeros_like(position)
        velocity = np.array(velocity)
        speed = np.linalg.norm(velocity)
        if speed < 1e-10:  # Avoid division by zero
            return np.zeros_like(velocity)
        return -0.5 * self.drag_coeff * speed * velocity


class CentralForce(Force):
    """Radial force field with 1/r^2 dependence"""
    def __init__(self, strength: float, center: ArrayLike = None):
        self.strength = strength
        self.center = np.zeros(2) if center is None else np.array(center)  # 2D default
        
    def __call__(self, position: ArrayLike, velocity: Optional[ArrayLike] = None,
                time: Optional[float] = None) -> np.ndarray:
        r = np.array(position) - self.center
        r_mag = np.linalg.norm(r)
        if r_mag < 1e-10:  # Avoid division by zero
            return np.zeros_like(r)
        return self.strength * r / r_mag**3


class FrictionForce(Force):
    """Combined static and kinetic friction"""
    def __init__(self, static_coeff: float, kinetic_coeff: float, normal_force: float):
        self.static_coeff = static_coeff
        self.kinetic_coeff = kinetic_coeff
        self.normal_force = normal_force
        
    def __call__(self, position: ArrayLike, velocity: ArrayLike,
                time: Optional[float] = None) -> np.ndarray:
        if velocity is None:
            return np.zeros_like(position)
        velocity = np.array(velocity)
        speed = np.linalg.norm(velocity)
        
        if speed < 1e-6:  # Static friction regime
            max_static_force = self.static_coeff * self.normal_force
            return -max_static_force * velocity if speed > 0 else np.zeros_like(velocity)
            
        # Kinetic friction regime
        return -self.kinetic_coeff * self.normal_force * velocity / speed


class GravityForce3D(Force):
    """
    3D gravitational force from a point mass or between two bodies.

    F = -G * m1 * m2 * r_hat / r^2

    Args:
        G: Gravitational constant (m³/kg/s²)
        source_mass: Mass of the gravitational source (kg)
        source_position: Position of the source mass
        target_mass: Mass of the target object (kg), default 1.0

    Examples:
        >>> earth_mass = 5.972e24
        >>> gravity = GravityForce3D(source_mass=earth_mass, source_position=[0, 0, 0])
        >>> force = gravity([6.371e6, 0, 0])  # At Earth's surface
    """

    def __init__(
        self,
        source_mass: float,
        source_position: ArrayLike,
        target_mass: float = 1.0,
        G: float = CONSTANTS['G'],
    ):
        self.G = validate_positive(G, "G")
        self.source_mass = validate_positive(source_mass, "source_mass")
        self.source_position = validate_vector(source_position, 3, "source_position")
        self.target_mass = validate_positive(target_mass, "target_mass")

    def __call__(
        self,
        position: ArrayLike,
        velocity: Optional[ArrayLike] = None,
        time: Optional[float] = None,
    ) -> np.ndarray:
        r = np.array(position) - self.source_position
        r_mag = np.linalg.norm(r)

        if r_mag < 1e-10:
            return np.zeros(3)

        # F = -G * M * m / r^2 * r_hat
        force_mag = self.G * self.source_mass * self.target_mass / r_mag**2
        return -force_mag * r / r_mag


class VanDerWaalsForce(Force):
    """
    Van der Waals intermolecular force using Lennard-Jones potential.

    The force is derived from the Lennard-Jones potential:
    U(r) = 4ε[(σ/r)^12 - (σ/r)^6]

    F(r) = 24ε/r [2(σ/r)^12 - (σ/r)^6] * r_hat

    Args:
        epsilon: Depth of potential well (J)
        sigma: Distance at which potential is zero (m)
        center: Position of the other molecule

    Examples:
        >>> vdw = VanDerWaalsForce(epsilon=1e-21, sigma=3.4e-10, center=[0, 0, 0])
        >>> force = vdw([4e-10, 0, 0])
    """

    def __init__(
        self,
        epsilon: float,
        sigma: float,
        center: ArrayLike = None,
    ):
        self.epsilon = validate_positive(epsilon, "epsilon")
        self.sigma = validate_positive(sigma, "sigma")
        self.center = np.zeros(3) if center is None else validate_vector(center, 3, "center")

    def __call__(
        self,
        position: ArrayLike,
        velocity: Optional[ArrayLike] = None,
        time: Optional[float] = None,
    ) -> np.ndarray:
        r_vec = np.array(position) - self.center
        r = np.linalg.norm(r_vec)

        if r < 1e-15:
            return np.zeros(3)

        # F = 24ε/r [2(σ/r)^12 - (σ/r)^6] * r_hat
        sr6 = (self.sigma / r)**6
        sr12 = sr6**2

        force_mag = 24 * self.epsilon / r * (2 * sr12 - sr6)
        return force_mag * r_vec / r

    def potential(self, position: ArrayLike) -> float:
        """Calculate Lennard-Jones potential energy."""
        r_vec = np.array(position) - self.center
        r = np.linalg.norm(r_vec)

        if r < 1e-15:
            return float('inf')

        sr6 = (self.sigma / r)**6
        return 4 * self.epsilon * (sr6**2 - sr6)


class ElectrostaticForce(Force):
    """
    Coulomb electrostatic force between point charges.

    F = k * q1 * q2 / r^2 * r_hat

    where k = 1/(4πε₀) ≈ 8.99e9 N⋅m²/C²

    Args:
        charge1: Charge of the source (C)
        charge2: Charge of the target (C)
        source_position: Position of source charge
        k: Coulomb constant (default: 1/(4πε₀))

    Examples:
        >>> # Two electrons 1 nm apart
        >>> e = 1.6e-19
        >>> coulomb = ElectrostaticForce(charge1=-e, charge2=-e, source_position=[0, 0, 0])
        >>> force = coulomb([1e-9, 0, 0])  # Repulsive force
    """

    def __init__(
        self,
        charge1: float,
        charge2: float,
        source_position: ArrayLike = None,
        k: float = None,
    ):
        self.charge1 = validate_finite(charge1, "charge1")
        self.charge2 = validate_finite(charge2, "charge2")
        self.source_position = np.zeros(3) if source_position is None else validate_vector(source_position, 3, "source_position")

        # Coulomb constant k = 1/(4πε₀)
        if k is None:
            k = 1.0 / (4 * np.pi * CONSTANTS['eps0'])
        self.k = validate_positive(k, "k")

    def __call__(
        self,
        position: ArrayLike,
        velocity: Optional[ArrayLike] = None,
        time: Optional[float] = None,
    ) -> np.ndarray:
        r_vec = np.array(position) - self.source_position
        r = np.linalg.norm(r_vec)

        if r < 1e-15:
            return np.zeros(3)

        # F = k * q1 * q2 / r^2 * r_hat
        force_mag = self.k * self.charge1 * self.charge2 / r**2
        return force_mag * r_vec / r

    def potential_energy(self, position: ArrayLike) -> float:
        """Calculate electrostatic potential energy."""
        r_vec = np.array(position) - self.source_position
        r = np.linalg.norm(r_vec)

        if r < 1e-15:
            return float('inf') if self.charge1 * self.charge2 > 0 else float('-inf')

        return self.k * self.charge1 * self.charge2 / r


class LorentzForce(Force):
    """
    Lorentz force on a charged particle in electric and magnetic fields.

    F = q(E + v × B)

    Args:
        charge: Particle charge (C)
        E_field: Electric field vector or callable(position, time) -> E
        B_field: Magnetic field vector or callable(position, time) -> B

    Examples:
        >>> # Charged particle in uniform fields
        >>> lorentz = LorentzForce(
        ...     charge=1.6e-19,
        ...     E_field=[0, 0, 0],
        ...     B_field=[0, 0, 1.0]  # 1 Tesla in z-direction
        ... )
        >>> force = lorentz([0, 0, 0], velocity=[1e6, 0, 0])  # Deflection
    """

    def __init__(
        self,
        charge: float,
        E_field: Union[ArrayLike, Callable] = None,
        B_field: Union[ArrayLike, Callable] = None,
    ):
        self.charge = validate_finite(charge, "charge")

        # Store field as callable or constant
        if E_field is None:
            self._E_field = lambda pos, t: np.zeros(3)
        elif callable(E_field):
            self._E_field = E_field
        else:
            E_const = validate_vector(E_field, 3, "E_field")
            self._E_field = lambda pos, t: E_const

        if B_field is None:
            self._B_field = lambda pos, t: np.zeros(3)
        elif callable(B_field):
            self._B_field = B_field
        else:
            B_const = validate_vector(B_field, 3, "B_field")
            self._B_field = lambda pos, t: B_const

    def __call__(
        self,
        position: ArrayLike,
        velocity: ArrayLike = None,
        time: float = None,
    ) -> np.ndarray:
        pos = np.array(position)
        vel = np.zeros(3) if velocity is None else np.array(velocity)
        t = 0.0 if time is None else time

        E = self._E_field(pos, t)
        B = self._B_field(pos, t)

        # F = q(E + v × B)
        return self.charge * (E + np.cross(vel, B))

    def cyclotron_frequency(self, mass: float, B_magnitude: float) -> float:
        """
        Calculate cyclotron frequency for given mass and B field magnitude.

        ω_c = |q|B/m

        Args:
            mass: Particle mass (kg)
            B_magnitude: Magnetic field magnitude (T)

        Returns:
            Cyclotron frequency (rad/s)
        """
        return abs(self.charge) * B_magnitude / mass

    def larmor_radius(self, mass: float, v_perp: float, B_magnitude: float) -> float:
        """
        Calculate Larmor (gyro) radius.

        r_L = m * v_perp / (|q| * B)

        Args:
            mass: Particle mass (kg)
            v_perp: Velocity perpendicular to B (m/s)
            B_magnitude: Magnetic field magnitude (T)

        Returns:
            Larmor radius (m)
        """
        return mass * v_perp / (abs(self.charge) * B_magnitude)


class TidalForce(Force):
    """
    Tidal force due to gravitational gradient.

    The tidal force arises from the difference in gravitational acceleration
    across an extended body. For a small displacement dr from the center:

    F_tidal = -G * M * m / r³ * [dr - 3(dr·r_hat)r_hat]

    Args:
        source_mass: Mass of the gravitating body (kg)
        source_position: Position of the gravitating body
        reference_position: Reference position (center of the extended body)
        G: Gravitational constant

    Examples:
        >>> # Moon's tidal force on Earth's surface
        >>> moon_mass = 7.34e22
        >>> moon_dist = 3.84e8
        >>> tidal = TidalForce(
        ...     source_mass=moon_mass,
        ...     source_position=[moon_dist, 0, 0],
        ...     reference_position=[0, 0, 0]
        ... )
    """

    def __init__(
        self,
        source_mass: float,
        source_position: ArrayLike,
        reference_position: ArrayLike = None,
        G: float = CONSTANTS['G'],
    ):
        self.G = validate_positive(G, "G")
        self.source_mass = validate_positive(source_mass, "source_mass")
        self.source_position = validate_vector(source_position, 3, "source_position")
        self.reference_position = np.zeros(3) if reference_position is None else validate_vector(reference_position, 3, "reference_position")

    def __call__(
        self,
        position: ArrayLike,
        velocity: Optional[ArrayLike] = None,
        time: Optional[float] = None,
        mass: float = 1.0,
    ) -> np.ndarray:
        # Vector from source to reference
        r_vec = self.reference_position - self.source_position
        r = np.linalg.norm(r_vec)

        if r < 1e-10:
            return np.zeros(3)

        r_hat = r_vec / r

        # Displacement from reference to position
        dr = np.array(position) - self.reference_position

        # Tidal force coefficient
        coeff = self.G * self.source_mass * mass / r**3

        # F_tidal = -coeff * [dr - 3(dr·r_hat)r_hat]
        return -coeff * (dr - 3 * np.dot(dr, r_hat) * r_hat)

    def tidal_tensor(self) -> np.ndarray:
        """
        Calculate the tidal tensor (gravitational gradient tensor).

        T_ij = -∂²Φ/∂x_i∂x_j = G*M/r³ [δ_ij - 3*r_i*r_j/r²]

        Returns:
            3x3 tidal tensor
        """
        r_vec = self.reference_position - self.source_position
        r = np.linalg.norm(r_vec)

        if r < 1e-10:
            return np.zeros((3, 3))

        r_hat = r_vec / r
        coeff = self.G * self.source_mass / r**3

        # T_ij = coeff * [δ_ij - 3*r_hat_i*r_hat_j]
        return coeff * (np.eye(3) - 3 * np.outer(r_hat, r_hat))


class CentrifugalForce(Force):
    """
    Centrifugal force in a rotating reference frame.

    F_cf = -m * ω × (ω × r)

    where r is the position relative to the rotation axis.

    Args:
        omega: Angular velocity vector of the rotating frame (rad/s)
        axis_point: A point on the rotation axis
        mass: Mass of the object (default 1.0)

    Examples:
        >>> # Earth's rotation
        >>> omega_earth = 7.27e-5  # rad/s
        >>> cf = CentrifugalForce(omega=[0, 0, omega_earth], mass=1.0)
        >>> force = cf([6.371e6, 0, 0])  # At equator
    """

    def __init__(
        self,
        omega: ArrayLike,
        axis_point: ArrayLike = None,
        mass: float = 1.0,
    ):
        self.omega = validate_vector(omega, 3, "omega")
        self.axis_point = np.zeros(3) if axis_point is None else validate_vector(axis_point, 3, "axis_point")
        self.mass = validate_positive(mass, "mass")

    def __call__(
        self,
        position: ArrayLike,
        velocity: Optional[ArrayLike] = None,
        time: Optional[float] = None,
    ) -> np.ndarray:
        pos = np.array(position)

        # Position relative to rotation axis
        r = pos - self.axis_point

        # F = -m * ω × (ω × r) = m * [ω²r - (ω·r)ω]
        omega_cross_r = np.cross(self.omega, r)
        return -self.mass * np.cross(self.omega, omega_cross_r)

    def magnitude_at_radius(self, radius: float) -> float:
        """
        Calculate centrifugal force magnitude at given perpendicular radius.

        |F_cf| = m * ω² * r

        Args:
            radius: Perpendicular distance from rotation axis (m)

        Returns:
            Force magnitude (N)
        """
        omega_mag = np.linalg.norm(self.omega)
        return self.mass * omega_mag**2 * radius


class CoriolisForce(Force):
    """
    Coriolis force in a rotating reference frame.

    F_cor = -2m * ω × v

    where v is the velocity in the rotating frame.

    Args:
        omega: Angular velocity vector of the rotating frame (rad/s)
        mass: Mass of the object (default 1.0)

    Examples:
        >>> # Coriolis effect on Earth
        >>> omega_earth = [0, 0, 7.27e-5]  # rad/s
        >>> coriolis = CoriolisForce(omega=omega_earth, mass=1.0)
        >>> # Northward moving object at equator
        >>> force = coriolis([0, 0, 0], velocity=[0, 100, 0])
    """

    def __init__(
        self,
        omega: ArrayLike,
        mass: float = 1.0,
    ):
        self.omega = validate_vector(omega, 3, "omega")
        self.mass = validate_positive(mass, "mass")

    def __call__(
        self,
        position: ArrayLike,
        velocity: ArrayLike = None,
        time: Optional[float] = None,
    ) -> np.ndarray:
        if velocity is None:
            return np.zeros(3)

        vel = np.array(velocity)

        # F = -2m * ω × v
        return -2 * self.mass * np.cross(self.omega, vel)


class EulerForce(Force):
    """
    Euler force from angular acceleration of the reference frame.

    F_euler = -m * α × r

    where α is the angular acceleration of the frame.

    Args:
        angular_acceleration: Angular acceleration vector (rad/s²)
        axis_point: A point on the rotation axis
        mass: Mass of the object (default 1.0)

    Examples:
        >>> # Spinning up a platform
        >>> euler = EulerForce(angular_acceleration=[0, 0, 0.1], mass=1.0)
        >>> force = euler([1, 0, 0])
    """

    def __init__(
        self,
        angular_acceleration: ArrayLike,
        axis_point: ArrayLike = None,
        mass: float = 1.0,
    ):
        self.alpha = validate_vector(angular_acceleration, 3, "angular_acceleration")
        self.axis_point = np.zeros(3) if axis_point is None else validate_vector(axis_point, 3, "axis_point")
        self.mass = validate_positive(mass, "mass")

    def __call__(
        self,
        position: ArrayLike,
        velocity: Optional[ArrayLike] = None,
        time: Optional[float] = None,
    ) -> np.ndarray:
        pos = np.array(position)

        # Position relative to rotation axis
        r = pos - self.axis_point

        # F = -m * α × r
        return -self.mass * np.cross(self.alpha, r)


class CompositeForce(Force):
    """
    Composite force combining multiple force components.

    Allows building complex force systems by summing individual forces.

    Args:
        forces: List of Force objects to combine

    Examples:
        >>> gravity = GravityForce(g=9.81)
        >>> drag = DragForce(drag_coeff=0.5)
        >>> total = CompositeForce([gravity, drag])
        >>> force = total([0, 10, 0], velocity=[0, -5, 0])
    """

    def __init__(self, forces: list = None):
        self.forces = forces if forces is not None else []

    def add_force(self, force: Force):
        """Add a force to the composite."""
        self.forces.append(force)

    def remove_force(self, force: Force):
        """Remove a force from the composite."""
        self.forces.remove(force)

    def __call__(
        self,
        position: ArrayLike,
        velocity: ArrayLike = None,
        time: float = None,
    ) -> np.ndarray:
        total = np.zeros(3)
        for force in self.forces:
            total = total + force(position, velocity, time)
        return total


class TimeVaryingForce(Force):
    """
    Time-varying force with arbitrary time dependence.

    F(t) = F_0 * f(t)

    Args:
        base_force: Base force vector or Force object
        time_func: Function f(t) that modulates the force

    Examples:
        >>> # Sinusoidal driving force
        >>> oscillating = TimeVaryingForce(
        ...     base_force=[1, 0, 0],
        ...     time_func=lambda t: np.sin(2 * np.pi * t)
        ... )
    """

    def __init__(
        self,
        base_force: Union[ArrayLike, Force],
        time_func: Callable[[float], float],
    ):
        if isinstance(base_force, Force):
            self._base_force = base_force
            self._is_force_obj = True
        else:
            self._base_force = validate_vector(base_force, 3, "base_force")
            self._is_force_obj = False

        self.time_func = time_func

    def __call__(
        self,
        position: ArrayLike,
        velocity: ArrayLike = None,
        time: float = None,
    ) -> np.ndarray:
        t = 0.0 if time is None else time

        if self._is_force_obj:
            base = self._base_force(position, velocity, time)
        else:
            base = self._base_force

        return base * self.time_func(t)
