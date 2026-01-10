"""
Energy and work primitives for mechanical systems.

This module provides classes for calculating work, power, potential energy
surfaces, and energy landscapes in mechanical systems.
"""

import numpy as np
from typing import Optional, Callable, Tuple, List, Union
from numpy.typing import ArrayLike
from scipy import integrate

from ..core.base import BaseClass
from ..core.utils import (
    validate_array,
    validate_vector,
    validate_positive,
    validate_non_negative,
    validate_finite,
    validate_callable,
)
from ..core.exceptions import ValidationError, PhysicsError
from .base import Force


class WorkCalculator(BaseClass):
    """
    Calculate work done by a force along a path.

    Computes the line integral W = ∫ F · dr along a given path.

    Args:
        force: Force object or callable(position, velocity, time) -> force vector

    Examples:
        >>> from sciforge.physics.forces import SpringForce
        >>> spring = SpringForce(k=100, anchor=[0, 0, 0])
        >>> calc = WorkCalculator(spring)
        >>> path = np.array([[1, 0, 0], [0.5, 0, 0], [0, 0, 0]])
        >>> work = calc.work_along_path(path)
    """

    def __init__(self, force: Union[Force, Callable]):
        super().__init__()
        self.force = force

    def work_along_path(
        self,
        path: ArrayLike,
        velocities: Optional[ArrayLike] = None,
        times: Optional[ArrayLike] = None,
    ) -> float:
        """
        Calculate work done along a discrete path.

        Uses trapezoidal integration along the path segments.

        Args:
            path: Array of position vectors along the path (N x dim)
            velocities: Optional array of velocities at each point
            times: Optional array of times at each point

        Returns:
            Total work done (J)
        """
        path = np.atleast_2d(path)
        n_points = len(path)

        if n_points < 2:
            return 0.0

        if velocities is None:
            velocities = [None] * n_points
        if times is None:
            times = [None] * n_points

        work = 0.0
        for i in range(n_points - 1):
            # Displacement vector
            dr = path[i + 1] - path[i]

            # Force at start and end points
            F1 = self.force(path[i], velocities[i], times[i] if times[i] is not None else 0.0)
            F2 = self.force(path[i + 1], velocities[i + 1] if i + 1 < len(velocities) else None,
                          times[i + 1] if times[i + 1] is not None else 0.0)

            # Trapezoidal rule: W ≈ (F1 + F2)/2 · dr
            work += 0.5 * np.dot(F1 + F2, dr)

        return work

    def work_parametric(
        self,
        path_func: Callable[[float], np.ndarray],
        t_start: float,
        t_end: float,
        velocity_func: Optional[Callable[[float], np.ndarray]] = None,
        n_points: int = 1000,
    ) -> float:
        """
        Calculate work along a parametric path.

        Args:
            path_func: Function r(t) returning position at parameter t
            t_start: Start parameter value
            t_end: End parameter value
            velocity_func: Optional function v(t) returning velocity
            n_points: Number of integration points

        Returns:
            Total work done (J)
        """
        t_values = np.linspace(t_start, t_end, n_points)
        path = np.array([path_func(t) for t in t_values])

        velocities = None
        if velocity_func is not None:
            velocities = np.array([velocity_func(t) for t in t_values])

        return self.work_along_path(path, velocities, t_values)

    def work_straight_line(
        self,
        start: ArrayLike,
        end: ArrayLike,
        n_points: int = 100,
    ) -> float:
        """
        Calculate work along a straight line path.

        Args:
            start: Starting position
            end: Ending position
            n_points: Number of integration points

        Returns:
            Total work done (J)
        """
        start = np.array(start)
        end = np.array(end)

        t_values = np.linspace(0, 1, n_points)
        path = np.array([start + t * (end - start) for t in t_values])

        return self.work_along_path(path)

    def is_conservative(
        self,
        closed_path: ArrayLike,
        tolerance: float = 1e-10,
    ) -> bool:
        """
        Test if the force is conservative by checking work around a closed path.

        For a conservative force, ∮ F · dr = 0.

        Args:
            closed_path: Closed path (first and last points should be same)
            tolerance: Tolerance for zero work

        Returns:
            True if force appears conservative
        """
        work = self.work_along_path(closed_path)
        return abs(work) < tolerance


class PowerMeter(BaseClass):
    """
    Calculate instantaneous and average power.

    Power is the rate of work done: P = dW/dt = F · v

    Args:
        force: Force object or callable

    Examples:
        >>> force = lambda pos, vel, t: np.array([0, -9.81, 0])  # Gravity
        >>> meter = PowerMeter(force)
        >>> power = meter.instantaneous([0, 10, 0], [0, -5, 0])
    """

    def __init__(self, force: Union[Force, Callable]):
        super().__init__()
        self.force = force

    def instantaneous(
        self,
        position: ArrayLike,
        velocity: ArrayLike,
        time: float = 0.0,
    ) -> float:
        """
        Calculate instantaneous power.

        P = F · v

        Args:
            position: Current position
            velocity: Current velocity
            time: Current time

        Returns:
            Instantaneous power (W)
        """
        F = self.force(position, velocity, time)
        v = np.array(velocity)
        return np.dot(F, v)

    def average_over_path(
        self,
        path: ArrayLike,
        velocities: ArrayLike,
        times: ArrayLike,
    ) -> float:
        """
        Calculate average power over a path.

        P_avg = W / Δt

        Args:
            path: Array of positions
            velocities: Array of velocities
            times: Array of times

        Returns:
            Average power (W)
        """
        path = np.atleast_2d(path)
        times = np.array(times)

        dt = times[-1] - times[0]
        if abs(dt) < 1e-15:
            return 0.0

        work_calc = WorkCalculator(self.force)
        work = work_calc.work_along_path(path, velocities, times)

        return work / dt

    def power_history(
        self,
        positions: ArrayLike,
        velocities: ArrayLike,
        times: ArrayLike,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate power at each point along a trajectory.

        Args:
            positions: Array of positions (N x dim)
            velocities: Array of velocities (N x dim)
            times: Array of times (N,)

        Returns:
            Tuple of (times, powers) arrays
        """
        positions = np.atleast_2d(positions)
        velocities = np.atleast_2d(velocities)
        times = np.array(times)

        powers = np.array([
            self.instantaneous(pos, vel, t)
            for pos, vel, t in zip(positions, velocities, times)
        ])

        return times, powers


class PotentialWell(BaseClass):
    """
    General potential energy well/surface.

    Represents a potential energy function U(r) and provides methods
    for analyzing its properties.

    Args:
        potential_func: Function U(position) -> potential energy
        gradient_func: Optional function returning -∇U (force). If not provided,
                      computed numerically.

    Examples:
        >>> # Harmonic potential well
        >>> def harmonic(r):
        ...     return 0.5 * 100 * np.sum(r**2)
        >>> well = PotentialWell(harmonic)
        >>> print(well.potential([1, 0, 0]))
        50.0
    """

    def __init__(
        self,
        potential_func: Callable[[np.ndarray], float],
        gradient_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        dim: int = 3,
    ):
        super().__init__()
        self.potential_func = potential_func
        self._gradient_func = gradient_func
        self.dim = dim

    def potential(self, position: ArrayLike) -> float:
        """Calculate potential energy at position."""
        return self.potential_func(np.array(position))

    def force(self, position: ArrayLike, dx: float = 1e-8) -> np.ndarray:
        """
        Calculate force at position (F = -∇U).

        Args:
            position: Position vector
            dx: Step size for numerical gradient

        Returns:
            Force vector
        """
        if self._gradient_func is not None:
            return self._gradient_func(np.array(position))

        # Numerical gradient
        pos = np.array(position)
        grad = np.zeros(len(pos))

        for i in range(len(pos)):
            pos_plus = pos.copy()
            pos_minus = pos.copy()
            pos_plus[i] += dx
            pos_minus[i] -= dx

            grad[i] = (self.potential_func(pos_plus) - self.potential_func(pos_minus)) / (2 * dx)

        return -grad  # F = -∇U

    def find_equilibrium(
        self,
        initial_guess: ArrayLike,
        tolerance: float = 1e-10,
        max_iter: int = 1000,
    ) -> Tuple[np.ndarray, bool]:
        """
        Find equilibrium point (where F = 0).

        Uses gradient descent to find local minimum.

        Args:
            initial_guess: Starting position
            tolerance: Convergence tolerance
            max_iter: Maximum iterations

        Returns:
            Tuple of (equilibrium position, converged)
        """
        from ..numerical.optimization import gradient_descent

        # Define gradient (positive for minimization)
        def grad(x):
            return -self.force(x)

        x_min, converged = gradient_descent(
            f=self.potential_func,
            grad=grad,
            x0=np.array(initial_guess),
            tol=tolerance,
            max_iter=max_iter,
        )

        return x_min, converged

    def is_stable(self, position: ArrayLike, dx: float = 1e-6) -> bool:
        """
        Check if position is a stable equilibrium (local minimum).

        Checks if Hessian is positive definite.

        Args:
            position: Position to check
            dx: Step size for numerical Hessian

        Returns:
            True if position is stable equilibrium
        """
        hessian = self.hessian(position, dx)
        eigenvalues = np.linalg.eigvalsh(hessian)
        return np.all(eigenvalues > 0)

    def hessian(self, position: ArrayLike, dx: float = 1e-6) -> np.ndarray:
        """
        Calculate Hessian matrix of potential at position.

        H_ij = ∂²U/∂x_i∂x_j

        Args:
            position: Position vector
            dx: Step size for numerical differentiation

        Returns:
            Hessian matrix
        """
        pos = np.array(position)
        n = len(pos)
        H = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    # Diagonal: second derivative
                    pos_plus = pos.copy()
                    pos_minus = pos.copy()
                    pos_plus[i] += dx
                    pos_minus[i] -= dx

                    H[i, j] = (self.potential_func(pos_plus) -
                              2 * self.potential_func(pos) +
                              self.potential_func(pos_minus)) / dx**2
                else:
                    # Off-diagonal: mixed partial derivative
                    pos_pp = pos.copy()
                    pos_pm = pos.copy()
                    pos_mp = pos.copy()
                    pos_mm = pos.copy()

                    pos_pp[i] += dx
                    pos_pp[j] += dx
                    pos_pm[i] += dx
                    pos_pm[j] -= dx
                    pos_mp[i] -= dx
                    pos_mp[j] += dx
                    pos_mm[i] -= dx
                    pos_mm[j] -= dx

                    H[i, j] = (self.potential_func(pos_pp) -
                              self.potential_func(pos_pm) -
                              self.potential_func(pos_mp) +
                              self.potential_func(pos_mm)) / (4 * dx**2)

        return H

    def oscillation_frequencies(self, equilibrium: ArrayLike, mass: float = 1.0) -> np.ndarray:
        """
        Calculate small oscillation frequencies about equilibrium.

        For a stable equilibrium, ω_i = √(λ_i/m) where λ_i are
        eigenvalues of the Hessian.

        Args:
            equilibrium: Equilibrium position
            mass: Mass of oscillating particle

        Returns:
            Array of oscillation frequencies
        """
        H = self.hessian(equilibrium)
        eigenvalues = np.linalg.eigvalsh(H)

        # Only positive eigenvalues give real frequencies
        positive_eigenvalues = eigenvalues[eigenvalues > 0]
        return np.sqrt(positive_eigenvalues / mass)


class EnergyLandscape(BaseClass):
    """
    Multi-dimensional potential energy surface analysis.

    Provides tools for visualizing and analyzing complex energy landscapes
    including finding transition states and minimum energy paths.

    Args:
        potential_func: Function U(position) -> energy
        dim: Dimensionality of the space

    Examples:
        >>> # Double well potential
        >>> def double_well(r):
        ...     x, y = r[0], r[1]
        ...     return (x**2 - 1)**2 + y**2
        >>> landscape = EnergyLandscape(double_well, dim=2)
        >>> minima = landscape.find_local_minima(n_starts=10)
    """

    def __init__(
        self,
        potential_func: Callable[[np.ndarray], float],
        dim: int = 2,
        bounds: Optional[Tuple[ArrayLike, ArrayLike]] = None,
    ):
        super().__init__()
        self.potential_func = potential_func
        self.dim = dim
        self.bounds = bounds

        # Default bounds
        if bounds is None:
            self.bounds = (np.full(dim, -10.0), np.full(dim, 10.0))
        else:
            self.bounds = (np.array(bounds[0]), np.array(bounds[1]))

    def potential(self, position: ArrayLike) -> float:
        """Evaluate potential at position."""
        return self.potential_func(np.array(position))

    def gradient(self, position: ArrayLike, dx: float = 1e-8) -> np.ndarray:
        """Calculate gradient numerically."""
        pos = np.array(position)
        grad = np.zeros(self.dim)

        for i in range(self.dim):
            pos_plus = pos.copy()
            pos_minus = pos.copy()
            pos_plus[i] += dx
            pos_minus[i] -= dx
            grad[i] = (self.potential_func(pos_plus) - self.potential_func(pos_minus)) / (2 * dx)

        return grad

    def evaluate_grid(
        self,
        resolution: int = 50,
        bounds: Optional[Tuple[ArrayLike, ArrayLike]] = None,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Evaluate potential on a grid.

        Args:
            resolution: Number of points along each dimension
            bounds: Optional override for bounds

        Returns:
            Tuple of (list of coordinate arrays, potential values)
        """
        if bounds is None:
            bounds = self.bounds

        # Create coordinate arrays
        coords = []
        for i in range(self.dim):
            coords.append(np.linspace(bounds[0][i], bounds[1][i], resolution))

        # Create meshgrid
        grids = np.meshgrid(*coords, indexing='ij')

        # Evaluate potential
        shape = grids[0].shape
        values = np.zeros(shape)

        for idx in np.ndindex(shape):
            pos = np.array([grids[i][idx] for i in range(self.dim)])
            values[idx] = self.potential_func(pos)

        return coords, values

    def find_local_minima(
        self,
        n_starts: int = 10,
        tolerance: float = 1e-8,
        max_iter: int = 1000,
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Find local minima using multiple random starts.

        Args:
            n_starts: Number of random starting points
            tolerance: Convergence tolerance
            max_iter: Maximum iterations per optimization

        Returns:
            List of (position, energy) tuples for found minima
        """
        from ..numerical.optimization import gradient_descent

        minima = []
        found_positions = []

        for _ in range(n_starts):
            # Random starting point within bounds
            x0 = np.random.uniform(self.bounds[0], self.bounds[1])

            try:
                x_min, converged = gradient_descent(
                    f=self.potential_func,
                    grad=self.gradient,
                    x0=x0,
                    tol=tolerance,
                    max_iter=max_iter,
                )

                if converged:
                    # Check if this is a new minimum
                    is_new = True
                    for pos in found_positions:
                        if np.linalg.norm(x_min - pos) < tolerance * 100:
                            is_new = False
                            break

                    if is_new:
                        energy = self.potential_func(x_min)
                        minima.append((x_min, energy))
                        found_positions.append(x_min)

            except Exception:
                continue

        # Sort by energy
        minima.sort(key=lambda x: x[1])
        return minima

    def find_saddle_points(
        self,
        n_starts: int = 20,
        tolerance: float = 1e-6,
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Find saddle points (transition states).

        Uses the method of searching for points where gradient is zero
        but Hessian has both positive and negative eigenvalues.

        Args:
            n_starts: Number of random starting points
            tolerance: Tolerance for zero gradient

        Returns:
            List of (position, energy) tuples for found saddle points
        """
        saddles = []
        found_positions = []

        for _ in range(n_starts):
            # Random starting point
            x0 = np.random.uniform(self.bounds[0], self.bounds[1])

            # Use Newton's method to find where gradient is zero
            x = x0.copy()
            for _ in range(100):
                grad = self.gradient(x)
                if np.linalg.norm(grad) < tolerance:
                    break

                # Hessian
                H = self._hessian(x)
                try:
                    delta = np.linalg.solve(H, -grad)
                    x = x + 0.5 * delta  # Damped step
                except np.linalg.LinAlgError:
                    break

            # Check if this is a saddle point
            if np.linalg.norm(self.gradient(x)) < tolerance:
                H = self._hessian(x)
                eigenvalues = np.linalg.eigvalsh(H)

                # Saddle point has both positive and negative eigenvalues
                if np.any(eigenvalues > 0) and np.any(eigenvalues < 0):
                    # Check if new
                    is_new = True
                    for pos in found_positions:
                        if np.linalg.norm(x - pos) < tolerance * 100:
                            is_new = False
                            break

                    if is_new:
                        energy = self.potential_func(x)
                        saddles.append((x, energy))
                        found_positions.append(x)

        saddles.sort(key=lambda x: x[1])
        return saddles

    def _hessian(self, position: ArrayLike, dx: float = 1e-6) -> np.ndarray:
        """Calculate Hessian matrix."""
        pos = np.array(position)
        H = np.zeros((self.dim, self.dim))

        for i in range(self.dim):
            for j in range(i, self.dim):
                if i == j:
                    pos_plus = pos.copy()
                    pos_minus = pos.copy()
                    pos_plus[i] += dx
                    pos_minus[i] -= dx

                    H[i, j] = (self.potential_func(pos_plus) -
                              2 * self.potential_func(pos) +
                              self.potential_func(pos_minus)) / dx**2
                else:
                    pos_pp = pos.copy()
                    pos_pm = pos.copy()
                    pos_mp = pos.copy()
                    pos_mm = pos.copy()

                    pos_pp[i] += dx
                    pos_pp[j] += dx
                    pos_pm[i] += dx
                    pos_pm[j] -= dx
                    pos_mp[i] -= dx
                    pos_mp[j] += dx
                    pos_mm[i] -= dx
                    pos_mm[j] -= dx

                    H[i, j] = (self.potential_func(pos_pp) -
                              self.potential_func(pos_pm) -
                              self.potential_func(pos_mp) +
                              self.potential_func(pos_mm)) / (4 * dx**2)
                    H[j, i] = H[i, j]

        return H

    def minimum_energy_path(
        self,
        start: ArrayLike,
        end: ArrayLike,
        n_images: int = 20,
        spring_constant: float = 1.0,
        max_iter: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find minimum energy path using Nudged Elastic Band (NEB) method.

        Args:
            start: Starting point (should be a minimum)
            end: Ending point (should be a minimum)
            n_images: Number of images along the path
            spring_constant: Spring constant for elastic band
            max_iter: Maximum NEB iterations

        Returns:
            Tuple of (path positions, energies along path)
        """
        start = np.array(start)
        end = np.array(end)

        # Initialize path with linear interpolation
        path = np.array([
            start + (end - start) * i / (n_images - 1)
            for i in range(n_images)
        ])

        # NEB optimization
        for _ in range(max_iter):
            forces = np.zeros_like(path)

            for i in range(1, n_images - 1):
                # Tangent vector
                tau = path[i + 1] - path[i - 1]
                tau = tau / np.linalg.norm(tau)

                # True force perpendicular to path
                F_true = -self.gradient(path[i])
                F_perp = F_true - np.dot(F_true, tau) * tau

                # Spring force parallel to path
                F_spring = spring_constant * (
                    np.linalg.norm(path[i + 1] - path[i]) -
                    np.linalg.norm(path[i] - path[i - 1])
                ) * tau

                forces[i] = F_perp + F_spring

            # Update path
            step_size = 0.1
            path[1:-1] += step_size * forces[1:-1]

        # Calculate energies
        energies = np.array([self.potential_func(p) for p in path])

        return path, energies


class KineticEnergy(BaseClass):
    """
    Kinetic energy calculations for various systems.

    Provides methods for calculating translational and rotational
    kinetic energy.

    Examples:
        >>> ke = KineticEnergy()
        >>> T = ke.translational(mass=1.0, velocity=[3, 4, 0])
        >>> print(T)  # 12.5 J
    """

    @staticmethod
    def translational(mass: float, velocity: ArrayLike) -> float:
        """
        Calculate translational kinetic energy.

        T = ½mv²

        Args:
            mass: Mass (kg)
            velocity: Velocity vector (m/s)

        Returns:
            Kinetic energy (J)
        """
        v = np.array(velocity)
        return 0.5 * mass * np.dot(v, v)

    @staticmethod
    def rotational(
        moment_of_inertia: Union[float, ArrayLike],
        angular_velocity: Union[float, ArrayLike],
    ) -> float:
        """
        Calculate rotational kinetic energy.

        For scalar I and ω: T = ½Iω²
        For tensor I and vector ω: T = ½ω·I·ω

        Args:
            moment_of_inertia: Scalar or inertia tensor (kg⋅m²)
            angular_velocity: Scalar or angular velocity vector (rad/s)

        Returns:
            Rotational kinetic energy (J)
        """
        if np.isscalar(moment_of_inertia) and np.isscalar(angular_velocity):
            return 0.5 * moment_of_inertia * angular_velocity**2

        I = np.atleast_2d(moment_of_inertia)
        omega = np.atleast_1d(angular_velocity)

        return 0.5 * np.dot(omega, I @ omega)

    @staticmethod
    def total(
        mass: float,
        velocity: ArrayLike,
        moment_of_inertia: Union[float, ArrayLike] = None,
        angular_velocity: Union[float, ArrayLike] = None,
    ) -> float:
        """
        Calculate total kinetic energy (translational + rotational).

        Args:
            mass: Mass (kg)
            velocity: Velocity vector (m/s)
            moment_of_inertia: Optional moment of inertia
            angular_velocity: Optional angular velocity

        Returns:
            Total kinetic energy (J)
        """
        T_trans = KineticEnergy.translational(mass, velocity)

        if moment_of_inertia is not None and angular_velocity is not None:
            T_rot = KineticEnergy.rotational(moment_of_inertia, angular_velocity)
            return T_trans + T_rot

        return T_trans


class MechanicalEnergy(BaseClass):
    """
    Track and analyze mechanical energy in a system.

    Monitors kinetic, potential, and total mechanical energy,
    checking for conservation.

    Args:
        potential_func: Potential energy function U(position)
        mass: System mass

    Examples:
        >>> def spring_potential(r):
        ...     return 0.5 * 100 * np.sum(r**2)
        >>> energy = MechanicalEnergy(spring_potential, mass=1.0)
        >>> E = energy.total([1, 0, 0], [0, 10, 0])
    """

    def __init__(
        self,
        potential_func: Callable[[np.ndarray], float],
        mass: float,
    ):
        super().__init__()
        self.potential_func = potential_func
        self.mass = validate_positive(mass, "mass")

        self._history = {
            'time': [],
            'kinetic': [],
            'potential': [],
            'total': [],
        }

    def kinetic(self, velocity: ArrayLike) -> float:
        """Calculate kinetic energy."""
        return KineticEnergy.translational(self.mass, velocity)

    def potential(self, position: ArrayLike) -> float:
        """Calculate potential energy."""
        return self.potential_func(np.array(position))

    def total(self, position: ArrayLike, velocity: ArrayLike) -> float:
        """Calculate total mechanical energy."""
        return self.kinetic(velocity) + self.potential(position)

    def record(self, time: float, position: ArrayLike, velocity: ArrayLike):
        """Record energy values at a time point."""
        K = self.kinetic(velocity)
        U = self.potential(position)
        E = K + U

        self._history['time'].append(time)
        self._history['kinetic'].append(K)
        self._history['potential'].append(U)
        self._history['total'].append(E)

    def check_conservation(self, tolerance: float = 1e-6) -> Tuple[bool, float]:
        """
        Check if mechanical energy is conserved.

        Args:
            tolerance: Relative tolerance for energy variation

        Returns:
            Tuple of (is_conserved, max_relative_error)
        """
        if len(self._history['total']) < 2:
            return True, 0.0

        energies = np.array(self._history['total'])
        E0 = energies[0]

        if abs(E0) < 1e-15:
            max_error = np.max(np.abs(energies - E0))
            return max_error < tolerance, max_error

        relative_errors = np.abs(energies - E0) / abs(E0)
        max_error = np.max(relative_errors)

        return max_error < tolerance, max_error

    def get_history(self) -> dict:
        """Return energy history."""
        return {k: np.array(v) for k, v in self._history.items()}
