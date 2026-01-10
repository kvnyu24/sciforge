"""
Analytical mechanics primitives: Lagrangian and Hamiltonian formulations.

This module provides classes for working with generalized coordinates,
Lagrangian and Hamiltonian mechanics, canonical transformations, and
symmetry-conservation relationships (Noether's theorem).
"""

import numpy as np
from typing import Optional, Callable, Tuple, List, Dict, Union
from numpy.typing import ArrayLike
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ..core.base import BaseClass
from ..core.utils import (
    validate_array,
    validate_vector,
    validate_positive,
    validate_finite,
    validate_callable,
)
from ..core.exceptions import ValidationError, PhysicsError


class GeneralizedCoordinates(BaseClass):
    """
    Generalized coordinates for mechanical systems.

    Manages a set of generalized coordinates q_i and their time derivatives
    (generalized velocities) q_dot_i.

    Args:
        n_dof: Number of degrees of freedom
        names: Optional names for each coordinate

    Examples:
        >>> coords = GeneralizedCoordinates(n_dof=2, names=['theta', 'phi'])
        >>> coords.set_state([np.pi/4, 0], [0.1, 0.2])
        >>> print(coords.q)
    """

    def __init__(
        self,
        n_dof: int,
        names: Optional[List[str]] = None,
    ):
        super().__init__()

        self.n_dof = n_dof
        self.names = names or [f"q_{i}" for i in range(n_dof)]

        if len(self.names) != n_dof:
            raise ValidationError("names", len(self.names), f"must have {n_dof} elements")

        self.q = np.zeros(n_dof)
        self.q_dot = np.zeros(n_dof)
        self.time = 0.0

    def set_state(self, q: ArrayLike, q_dot: ArrayLike):
        """Set generalized coordinates and velocities."""
        self.q = np.array(q)
        self.q_dot = np.array(q_dot)

        if len(self.q) != self.n_dof or len(self.q_dot) != self.n_dof:
            raise ValidationError("state", (len(self.q), len(self.q_dot)),
                                f"must have {self.n_dof} elements each")

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current state (q, q_dot)."""
        return self.q.copy(), self.q_dot.copy()

    def state_vector(self) -> np.ndarray:
        """Get state as single vector [q, q_dot]."""
        return np.concatenate([self.q, self.q_dot])

    def set_from_state_vector(self, state: ArrayLike):
        """Set state from combined vector [q, q_dot]."""
        state = np.array(state)
        self.q = state[:self.n_dof]
        self.q_dot = state[self.n_dof:]

    def __getitem__(self, key):
        """Get coordinate by index or name."""
        if isinstance(key, str):
            idx = self.names.index(key)
            return self.q[idx]
        return self.q[key]

    def __setitem__(self, key, value):
        """Set coordinate by index or name."""
        if isinstance(key, str):
            idx = self.names.index(key)
            self.q[idx] = value
        else:
            self.q[key] = value


class LagrangianSystem(BaseClass):
    """
    System described by a Lagrangian.

    Implements the Euler-Lagrange equations of motion:
    d/dt(∂L/∂q_dot) - ∂L/∂q = Q

    where L = T - V is the Lagrangian and Q are generalized forces.

    Args:
        n_dof: Number of degrees of freedom
        lagrangian: Function L(q, q_dot, t) -> scalar
        generalized_forces: Optional function Q(q, q_dot, t) -> array
        coordinate_names: Optional names for coordinates

    Examples:
        >>> # Simple pendulum
        >>> def pendulum_lagrangian(q, q_dot, t, m=1.0, l=1.0, g=9.81):
        ...     theta, theta_dot = q[0], q_dot[0]
        ...     T = 0.5 * m * l**2 * theta_dot**2
        ...     V = -m * g * l * np.cos(theta)
        ...     return T - V
        >>> system = LagrangianSystem(n_dof=1, lagrangian=pendulum_lagrangian)
    """

    def __init__(
        self,
        n_dof: int,
        lagrangian: Callable[[np.ndarray, np.ndarray, float], float],
        generalized_forces: Optional[Callable[[np.ndarray, np.ndarray, float], np.ndarray]] = None,
        coordinate_names: Optional[List[str]] = None,
        mass_matrix: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        super().__init__()

        self.n_dof = n_dof
        self.L = lagrangian
        self.Q = generalized_forces
        self.mass_matrix_func = mass_matrix

        self.coords = GeneralizedCoordinates(n_dof, coordinate_names)
        self._history = {
            'time': [],
            'q': [],
            'q_dot': [],
            'energy': [],
        }

    @property
    def q(self) -> np.ndarray:
        return self.coords.q

    @property
    def q_dot(self) -> np.ndarray:
        return self.coords.q_dot

    @property
    def time(self) -> float:
        return self.coords.time

    def set_state(self, q: ArrayLike, q_dot: ArrayLike, t: float = 0.0):
        """Set system state."""
        self.coords.set_state(q, q_dot)
        self.coords.time = t

    def lagrangian(
        self,
        q: Optional[ArrayLike] = None,
        q_dot: Optional[ArrayLike] = None,
        t: Optional[float] = None,
    ) -> float:
        """Evaluate Lagrangian at given or current state."""
        if q is None:
            q = self.q
        if q_dot is None:
            q_dot = self.q_dot
        if t is None:
            t = self.time
        return self.L(np.array(q), np.array(q_dot), t)

    def kinetic_energy(self, q: ArrayLike = None, q_dot: ArrayLike = None) -> float:
        """
        Estimate kinetic energy T = ∂L/∂q_dot · q_dot - L + V

        For L = T - V with T quadratic in q_dot: T = ½ q_dot · M · q_dot
        """
        if q is None:
            q = self.q
        if q_dot is None:
            q_dot = self.q_dot

        # Numerical estimation via derivative with respect to q_dot
        dL_dqdot = self._dL_dqdot(q, q_dot, self.time)
        L_val = self.lagrangian(q, q_dot)

        # For standard Lagrangian T - V with T = ½ q_dot · M · q_dot:
        # T = ½ ∂L/∂q_dot · q_dot
        return 0.5 * np.dot(dL_dqdot, q_dot)

    def _dL_dq(
        self,
        q: np.ndarray,
        q_dot: np.ndarray,
        t: float,
        dx: float = 1e-8,
    ) -> np.ndarray:
        """Calculate ∂L/∂q numerically."""
        dL_dq = np.zeros(self.n_dof)
        for i in range(self.n_dof):
            q_plus = q.copy()
            q_minus = q.copy()
            q_plus[i] += dx
            q_minus[i] -= dx
            dL_dq[i] = (self.L(q_plus, q_dot, t) - self.L(q_minus, q_dot, t)) / (2 * dx)
        return dL_dq

    def _dL_dqdot(
        self,
        q: np.ndarray,
        q_dot: np.ndarray,
        t: float,
        dx: float = 1e-8,
    ) -> np.ndarray:
        """Calculate ∂L/∂q_dot (generalized momentum) numerically."""
        dL_dqdot = np.zeros(self.n_dof)
        for i in range(self.n_dof):
            qdot_plus = q_dot.copy()
            qdot_minus = q_dot.copy()
            qdot_plus[i] += dx
            qdot_minus[i] -= dx
            dL_dqdot[i] = (self.L(q, qdot_plus, t) - self.L(q, qdot_minus, t)) / (2 * dx)
        return dL_dqdot

    def generalized_momenta(
        self,
        q: ArrayLike = None,
        q_dot: ArrayLike = None,
        t: float = None,
    ) -> np.ndarray:
        """
        Calculate generalized momenta.

        p_i = ∂L/∂q_dot_i
        """
        if q is None:
            q = self.q
        if q_dot is None:
            q_dot = self.q_dot
        if t is None:
            t = self.time
        return self._dL_dqdot(np.array(q), np.array(q_dot), t)

    def equations_of_motion(
        self,
        q: np.ndarray,
        q_dot: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Calculate generalized accelerations from Euler-Lagrange equations.

        Returns q_ddot satisfying:
        d/dt(∂L/∂q_dot) = ∂L/∂q + Q
        """
        # Calculate required derivatives
        dL_dq = self._dL_dq(q, q_dot, t)

        # Get generalized forces
        Q = np.zeros(self.n_dof)
        if self.Q is not None:
            Q = self.Q(q, q_dot, t)

        # For standard Lagrangian L = T - V with T = ½ q_dot · M(q) · q_dot:
        # M(q) · q_ddot = ∂L/∂q + Q - dM/dq terms
        # This requires the mass matrix

        if self.mass_matrix_func is not None:
            M = self.mass_matrix_func(q)
            # Simplified: ignore velocity-dependent terms in M
            rhs = dL_dq + Q
            q_ddot = np.linalg.solve(M, rhs)
        else:
            # Numerical approach using second derivatives
            # d/dt(∂L/∂q_dot) = ∂²L/∂q_dot∂q · q_dot + ∂²L/∂q_dot² · q_ddot + ∂²L/∂q_dot∂t
            dx = 1e-6

            # Estimate mass matrix M_ij = ∂²L/∂q_dot_i∂q_dot_j
            M = np.zeros((self.n_dof, self.n_dof))
            for i in range(self.n_dof):
                for j in range(self.n_dof):
                    qdot_pp = q_dot.copy()
                    qdot_pm = q_dot.copy()
                    qdot_mp = q_dot.copy()
                    qdot_mm = q_dot.copy()

                    qdot_pp[i] += dx
                    qdot_pp[j] += dx
                    qdot_pm[i] += dx
                    qdot_pm[j] -= dx
                    qdot_mp[i] -= dx
                    qdot_mp[j] += dx
                    qdot_mm[i] -= dx
                    qdot_mm[j] -= dx

                    M[i, j] = (self.L(q, qdot_pp, t) - self.L(q, qdot_pm, t) -
                              self.L(q, qdot_mp, t) + self.L(q, qdot_mm, t)) / (4 * dx**2)

            # Mixed partial ∂²L/∂q_dot∂q
            mixed = np.zeros((self.n_dof, self.n_dof))
            for i in range(self.n_dof):
                for j in range(self.n_dof):
                    q_p = q.copy()
                    q_m = q.copy()
                    qdot_p = q_dot.copy()
                    qdot_m = q_dot.copy()

                    q_p[j] += dx
                    q_m[j] -= dx
                    qdot_p[i] += dx
                    qdot_m[i] -= dx

                    mixed[i, j] = (self.L(q_p, qdot_p, t) - self.L(q_p, qdot_m, t) -
                                  self.L(q_m, qdot_p, t) + self.L(q_m, qdot_m, t)) / (4 * dx**2)

            # RHS = ∂L/∂q + Q - ∂²L/∂q_dot∂q · q_dot
            rhs = dL_dq + Q - mixed @ q_dot

            # Solve M · q_ddot = rhs
            try:
                q_ddot = np.linalg.solve(M, rhs)
            except np.linalg.LinAlgError:
                q_ddot = np.linalg.lstsq(M, rhs, rcond=None)[0]

        return q_ddot

    def update(self, dt: float):
        """
        Update system state using RK4 integration.

        Args:
            dt: Time step
        """
        q = self.q.copy()
        q_dot = self.q_dot.copy()
        t = self.time

        def deriv(q, q_dot, t):
            q_ddot = self.equations_of_motion(q, q_dot, t)
            return q_dot, q_ddot

        # RK4
        k1_q, k1_qdot = deriv(q, q_dot, t)

        k2_q, k2_qdot = deriv(
            q + 0.5*dt*k1_q,
            q_dot + 0.5*dt*k1_qdot,
            t + 0.5*dt
        )

        k3_q, k3_qdot = deriv(
            q + 0.5*dt*k2_q,
            q_dot + 0.5*dt*k2_qdot,
            t + 0.5*dt
        )

        k4_q, k4_qdot = deriv(
            q + dt*k3_q,
            q_dot + dt*k3_qdot,
            t + dt
        )

        self.coords.q = q + (dt/6) * (k1_q + 2*k2_q + 2*k3_q + k4_q)
        self.coords.q_dot = q_dot + (dt/6) * (k1_qdot + 2*k2_qdot + 2*k3_qdot + k4_qdot)
        self.coords.time = t + dt

        # Record history
        self._history['time'].append(self.time)
        self._history['q'].append(self.q.copy())
        self._history['q_dot'].append(self.q_dot.copy())
        self._history['energy'].append(self.energy())

    def energy(self) -> float:
        """
        Calculate total energy (Hamiltonian for time-independent L).

        H = p · q_dot - L
        """
        p = self.generalized_momenta()
        L = self.lagrangian()
        return np.dot(p, self.q_dot) - L

    def get_history(self) -> Dict[str, np.ndarray]:
        """Get simulation history."""
        return {k: np.array(v) for k, v in self._history.items()}


class HamiltonianSystem(BaseClass):
    """
    System described by a Hamiltonian.

    Implements Hamilton's equations of motion:
    dq/dt = ∂H/∂p
    dp/dt = -∂H/∂q

    Args:
        n_dof: Number of degrees of freedom
        hamiltonian: Function H(q, p, t) -> scalar
        coordinate_names: Optional names for coordinates

    Examples:
        >>> # Simple harmonic oscillator
        >>> def sho_hamiltonian(q, p, t, m=1.0, k=1.0):
        ...     return p[0]**2 / (2*m) + 0.5 * k * q[0]**2
        >>> system = HamiltonianSystem(n_dof=1, hamiltonian=sho_hamiltonian)
    """

    def __init__(
        self,
        n_dof: int,
        hamiltonian: Callable[[np.ndarray, np.ndarray, float], float],
        coordinate_names: Optional[List[str]] = None,
    ):
        super().__init__()

        self.n_dof = n_dof
        self.H = hamiltonian
        self.coord_names = coordinate_names or [f"q_{i}" for i in range(n_dof)]

        self.q = np.zeros(n_dof)
        self.p = np.zeros(n_dof)  # Canonical momenta
        self.time = 0.0

        self._history = {
            'time': [],
            'q': [],
            'p': [],
            'H': [],
        }

    def set_state(self, q: ArrayLike, p: ArrayLike, t: float = 0.0):
        """Set system state."""
        self.q = np.array(q)
        self.p = np.array(p)
        self.time = t

    def hamiltonian(
        self,
        q: ArrayLike = None,
        p: ArrayLike = None,
        t: float = None,
    ) -> float:
        """Evaluate Hamiltonian at given or current state."""
        if q is None:
            q = self.q
        if p is None:
            p = self.p
        if t is None:
            t = self.time
        return self.H(np.array(q), np.array(p), t)

    def _dH_dq(
        self,
        q: np.ndarray,
        p: np.ndarray,
        t: float,
        dx: float = 1e-8,
    ) -> np.ndarray:
        """Calculate ∂H/∂q numerically."""
        dH = np.zeros(self.n_dof)
        for i in range(self.n_dof):
            q_plus = q.copy()
            q_minus = q.copy()
            q_plus[i] += dx
            q_minus[i] -= dx
            dH[i] = (self.H(q_plus, p, t) - self.H(q_minus, p, t)) / (2 * dx)
        return dH

    def _dH_dp(
        self,
        q: np.ndarray,
        p: np.ndarray,
        t: float,
        dx: float = 1e-8,
    ) -> np.ndarray:
        """Calculate ∂H/∂p numerically."""
        dH = np.zeros(self.n_dof)
        for i in range(self.n_dof):
            p_plus = p.copy()
            p_minus = p.copy()
            p_plus[i] += dx
            p_minus[i] -= dx
            dH[i] = (self.H(q, p_plus, t) - self.H(q, p_minus, t)) / (2 * dx)
        return dH

    def equations_of_motion(
        self,
        q: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Hamilton's equations.

        Returns (dq/dt, dp/dt).
        """
        dq_dt = self._dH_dp(q, p, t)
        dp_dt = -self._dH_dq(q, p, t)
        return dq_dt, dp_dt

    def update(self, dt: float):
        """
        Update system using symplectic integrator (leapfrog/Störmer-Verlet).

        Args:
            dt: Time step
        """
        # Symplectic integration preserves phase space volume
        # Half step in p
        dp_dt = -self._dH_dq(self.q, self.p, self.time)
        p_half = self.p + 0.5 * dt * dp_dt

        # Full step in q
        dq_dt = self._dH_dp(self.q, p_half, self.time + 0.5*dt)
        self.q = self.q + dt * dq_dt

        # Half step in p
        dp_dt = -self._dH_dq(self.q, p_half, self.time + dt)
        self.p = p_half + 0.5 * dt * dp_dt

        self.time += dt

        # Record history
        self._history['time'].append(self.time)
        self._history['q'].append(self.q.copy())
        self._history['p'].append(self.p.copy())
        self._history['H'].append(self.hamiltonian())

    def update_rk4(self, dt: float):
        """
        Update system using RK4 (non-symplectic but more accurate short-term).

        Args:
            dt: Time step
        """
        q = self.q.copy()
        p = self.p.copy()
        t = self.time

        def deriv(q, p, t):
            return self.equations_of_motion(q, p, t)

        # RK4
        k1_q, k1_p = deriv(q, p, t)

        k2_q, k2_p = deriv(
            q + 0.5*dt*k1_q,
            p + 0.5*dt*k1_p,
            t + 0.5*dt
        )

        k3_q, k3_p = deriv(
            q + 0.5*dt*k2_q,
            p + 0.5*dt*k2_p,
            t + 0.5*dt
        )

        k4_q, k4_p = deriv(
            q + dt*k3_q,
            p + dt*k3_p,
            t + dt
        )

        self.q = q + (dt/6) * (k1_q + 2*k2_q + 2*k3_q + k4_q)
        self.p = p + (dt/6) * (k1_p + 2*k2_p + 2*k3_p + k4_p)
        self.time = t + dt

        # Record history
        self._history['time'].append(self.time)
        self._history['q'].append(self.q.copy())
        self._history['p'].append(self.p.copy())
        self._history['H'].append(self.hamiltonian())

    def phase_space_trajectory(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get phase space trajectory from history."""
        return np.array(self._history['q']), np.array(self._history['p'])

    def get_history(self) -> Dict[str, np.ndarray]:
        """Get simulation history."""
        return {k: np.array(v) for k, v in self._history.items()}

    @classmethod
    def from_lagrangian(
        cls,
        lagrangian_system: LagrangianSystem,
    ) -> 'HamiltonianSystem':
        """
        Create Hamiltonian system from Lagrangian via Legendre transform.

        H(q, p, t) = p · q_dot(q, p) - L(q, q_dot(q, p), t)

        Note: This requires inverting p = ∂L/∂q_dot to get q_dot(q, p).
        """
        n_dof = lagrangian_system.n_dof

        def hamiltonian(q, p, t):
            # Invert p = ∂L/∂q_dot using Newton's method
            q_dot = np.zeros(n_dof)
            for _ in range(20):
                p_calc = lagrangian_system._dL_dqdot(q, q_dot, t)
                error = p_calc - p
                if np.linalg.norm(error) < 1e-12:
                    break
                # Jacobian approximation
                J = np.eye(n_dof)
                dx = 1e-8
                for i in range(n_dof):
                    qdot_plus = q_dot.copy()
                    qdot_plus[i] += dx
                    p_plus = lagrangian_system._dL_dqdot(q, qdot_plus, t)
                    J[:, i] = (p_plus - p_calc) / dx
                try:
                    delta = np.linalg.solve(J, -error)
                except np.linalg.LinAlgError:
                    break
                q_dot += delta

            L = lagrangian_system.L(q, q_dot, t)
            return np.dot(p, q_dot) - L

        return cls(
            n_dof=n_dof,
            hamiltonian=hamiltonian,
            coordinate_names=lagrangian_system.coords.names,
        )


class PoissonBracket(BaseClass):
    """
    Poisson bracket operations for Hamiltonian mechanics.

    The Poisson bracket of two functions f, g is:
    {f, g} = Σ_i (∂f/∂q_i ∂g/∂p_i - ∂f/∂p_i ∂g/∂q_i)

    Args:
        n_dof: Number of degrees of freedom

    Examples:
        >>> pb = PoissonBracket(n_dof=1)
        >>> # {q, p} = 1 (canonical relation)
        >>> def q_func(q, p): return q[0]
        >>> def p_func(q, p): return p[0]
        >>> result = pb.bracket(q_func, p_func, [0], [1])
    """

    def __init__(self, n_dof: int):
        super().__init__()
        self.n_dof = n_dof

    def bracket(
        self,
        f: Callable[[np.ndarray, np.ndarray], float],
        g: Callable[[np.ndarray, np.ndarray], float],
        q: ArrayLike,
        p: ArrayLike,
        dx: float = 1e-8,
    ) -> float:
        """
        Calculate Poisson bracket {f, g} at given point.

        Args:
            f: First function f(q, p)
            g: Second function g(q, p)
            q: Generalized coordinates
            p: Canonical momenta
            dx: Step size for numerical derivatives

        Returns:
            Value of {f, g}
        """
        q = np.array(q)
        p = np.array(p)
        result = 0.0

        for i in range(self.n_dof):
            # ∂f/∂q_i
            q_plus = q.copy()
            q_minus = q.copy()
            q_plus[i] += dx
            q_minus[i] -= dx
            df_dq = (f(q_plus, p) - f(q_minus, p)) / (2 * dx)

            # ∂g/∂p_i
            p_plus = p.copy()
            p_minus = p.copy()
            p_plus[i] += dx
            p_minus[i] -= dx
            dg_dp = (g(q, p_plus) - g(q, p_minus)) / (2 * dx)

            # ∂f/∂p_i
            df_dp = (f(q, p_plus) - f(q, p_minus)) / (2 * dx)

            # ∂g/∂q_i
            dg_dq = (g(q_plus, p) - g(q_minus, p)) / (2 * dx)

            result += df_dq * dg_dp - df_dp * dg_dq

        return result

    def is_constant_of_motion(
        self,
        f: Callable[[np.ndarray, np.ndarray], float],
        H: Callable[[np.ndarray, np.ndarray], float],
        q: ArrayLike,
        p: ArrayLike,
        tolerance: float = 1e-10,
    ) -> bool:
        """
        Check if f is a constant of motion (commutes with H).

        f is conserved if {f, H} = 0.

        Args:
            f: Function to test
            H: Hamiltonian
            q, p: Phase space point to test at
            tolerance: Tolerance for zero bracket

        Returns:
            True if f appears to be conserved
        """
        return abs(self.bracket(f, H, q, p)) < tolerance

    def canonical_check(
        self,
        q_funcs: List[Callable],
        p_funcs: List[Callable],
        q: ArrayLike,
        p: ArrayLike,
        tolerance: float = 1e-10,
    ) -> bool:
        """
        Check if transformation is canonical using Poisson brackets.

        Canonical transformation preserves:
        {Q_i, Q_j} = 0, {P_i, P_j} = 0, {Q_i, P_j} = δ_ij

        Args:
            q_funcs: Functions Q_i(q, p) for new coordinates
            p_funcs: Functions P_i(q, p) for new momenta
            q, p: Phase space point
            tolerance: Tolerance

        Returns:
            True if transformation appears canonical
        """
        n = len(q_funcs)
        if n != len(p_funcs):
            return False

        # Check {Q_i, Q_j} = 0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(self.bracket(q_funcs[i], q_funcs[j], q, p)) > tolerance:
                    return False

        # Check {P_i, P_j} = 0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(self.bracket(p_funcs[i], p_funcs[j], q, p)) > tolerance:
                    return False

        # Check {Q_i, P_j} = δ_ij
        for i in range(n):
            for j in range(n):
                expected = 1.0 if i == j else 0.0
                if abs(self.bracket(q_funcs[i], p_funcs[j], q, p) - expected) > tolerance:
                    return False

        return True


class ActionPrinciple(BaseClass):
    """
    Action principle and variational methods.

    Implements the principle of least action:
    δS = δ∫L dt = 0

    Args:
        lagrangian: Lagrangian function L(q, q_dot, t)
        n_dof: Number of degrees of freedom

    Examples:
        >>> def L(q, q_dot, t):
        ...     return 0.5 * q_dot[0]**2 - 0.5 * q[0]**2
        >>> action = ActionPrinciple(L, n_dof=1)
        >>> S = action.calculate([0, 1], [[0], [1]], dt=0.01)
    """

    def __init__(
        self,
        lagrangian: Callable[[np.ndarray, np.ndarray, float], float],
        n_dof: int,
    ):
        super().__init__()
        self.L = lagrangian
        self.n_dof = n_dof

    def calculate(
        self,
        time_span: Tuple[float, float],
        trajectory: ArrayLike,
        dt: Optional[float] = None,
    ) -> float:
        """
        Calculate action along a trajectory.

        S = ∫ L dt

        Args:
            time_span: (t_start, t_end)
            trajectory: Array of q values at each time step (n_times x n_dof)
            dt: Time step (calculated from trajectory if None)

        Returns:
            Action value
        """
        trajectory = np.atleast_2d(trajectory)
        n_times = len(trajectory)

        if dt is None:
            dt = (time_span[1] - time_span[0]) / (n_times - 1)

        # Calculate velocities from finite differences
        q_dots = np.zeros_like(trajectory)
        q_dots[0] = (trajectory[1] - trajectory[0]) / dt
        q_dots[-1] = (trajectory[-1] - trajectory[-2]) / dt
        for i in range(1, n_times - 1):
            q_dots[i] = (trajectory[i + 1] - trajectory[i - 1]) / (2 * dt)

        # Integrate Lagrangian
        action = 0.0
        for i in range(n_times):
            t = time_span[0] + i * dt
            L_val = self.L(trajectory[i], q_dots[i], t)
            weight = dt if 0 < i < n_times - 1 else dt / 2
            action += L_val * weight

        return action

    def euler_lagrange_residual(
        self,
        trajectory: ArrayLike,
        time_span: Tuple[float, float],
    ) -> np.ndarray:
        """
        Calculate Euler-Lagrange equation residual along trajectory.

        Returns how well the trajectory satisfies the equations of motion.

        Args:
            trajectory: Array of q values
            time_span: (t_start, t_end)

        Returns:
            Array of residuals at each interior point
        """
        trajectory = np.atleast_2d(trajectory)
        n_times = len(trajectory)
        dt = (time_span[1] - time_span[0]) / (n_times - 1)

        residuals = []

        for i in range(1, n_times - 1):
            t = time_span[0] + i * dt
            q = trajectory[i]

            # Estimate q_dot
            q_dot = (trajectory[i + 1] - trajectory[i - 1]) / (2 * dt)

            # Estimate q_ddot
            q_ddot = (trajectory[i + 1] - 2 * trajectory[i] + trajectory[i - 1]) / dt**2

            # Calculate ∂L/∂q
            dL_dq = np.zeros(self.n_dof)
            dx = 1e-8
            for j in range(self.n_dof):
                q_plus = q.copy()
                q_minus = q.copy()
                q_plus[j] += dx
                q_minus[j] -= dx
                dL_dq[j] = (self.L(q_plus, q_dot, t) - self.L(q_minus, q_dot, t)) / (2 * dx)

            # Calculate ∂²L/∂q_dot² (mass matrix)
            M = np.zeros((self.n_dof, self.n_dof))
            for j in range(self.n_dof):
                for k in range(self.n_dof):
                    qdot_pp = q_dot.copy()
                    qdot_pm = q_dot.copy()
                    qdot_mp = q_dot.copy()
                    qdot_mm = q_dot.copy()
                    qdot_pp[j] += dx
                    qdot_pp[k] += dx
                    qdot_pm[j] += dx
                    qdot_pm[k] -= dx
                    qdot_mp[j] -= dx
                    qdot_mp[k] += dx
                    qdot_mm[j] -= dx
                    qdot_mm[k] -= dx

                    M[j, k] = (self.L(q, qdot_pp, t) - self.L(q, qdot_pm, t) -
                              self.L(q, qdot_mp, t) + self.L(q, qdot_mm, t)) / (4 * dx**2)

            # E-L equation: d/dt(∂L/∂q_dot) - ∂L/∂q = 0
            # ≈ M·q_ddot + ... - ∂L/∂q = 0
            residual = M @ q_ddot - dL_dq
            residuals.append(residual)

        return np.array(residuals)


class NoetherSymmetry(BaseClass):
    """
    Noether's theorem: symmetries and conservation laws.

    For each continuous symmetry of the Lagrangian, there is a
    corresponding conserved quantity.

    Args:
        lagrangian: Lagrangian function L(q, q_dot, t)
        n_dof: Number of degrees of freedom

    Examples:
        >>> # Check time translation symmetry (energy conservation)
        >>> def L(q, q_dot, t):
        ...     return 0.5 * q_dot[0]**2 - 0.5 * q[0]**2
        >>> noether = NoetherSymmetry(L, n_dof=1)
        >>> # dL/dt = 0 implies energy conservation
    """

    def __init__(
        self,
        lagrangian: Callable[[np.ndarray, np.ndarray, float], float],
        n_dof: int,
    ):
        super().__init__()
        self.L = lagrangian
        self.n_dof = n_dof

    def check_time_translation(
        self,
        q: ArrayLike,
        q_dot: ArrayLike,
        t: float = 0.0,
        dt: float = 1e-8,
    ) -> Tuple[bool, float]:
        """
        Check time translation symmetry (∂L/∂t = 0).

        If symmetric, energy H = p·q_dot - L is conserved.

        Returns:
            Tuple of (is_symmetric, dL_dt)
        """
        q = np.array(q)
        q_dot = np.array(q_dot)

        dL_dt = (self.L(q, q_dot, t + dt) - self.L(q, q_dot, t - dt)) / (2 * dt)

        return abs(dL_dt) < 1e-10, dL_dt

    def check_space_translation(
        self,
        direction: ArrayLike,
        q: ArrayLike,
        q_dot: ArrayLike,
        t: float = 0.0,
        dx: float = 1e-8,
    ) -> Tuple[bool, float, float]:
        """
        Check space translation symmetry in given direction.

        If symmetric, momentum in that direction is conserved.

        Args:
            direction: Direction of translation (will be normalized)
            q, q_dot: Current state
            t: Time

        Returns:
            Tuple of (is_symmetric, dL_dq_component, conserved_momentum)
        """
        q = np.array(q)
        q_dot = np.array(q_dot)
        direction = np.array(direction)
        direction = direction / np.linalg.norm(direction)

        # Check ∂L/∂q · direction
        dL_dq = np.zeros(self.n_dof)
        for i in range(self.n_dof):
            q_plus = q.copy()
            q_minus = q.copy()
            q_plus[i] += dx
            q_minus[i] -= dx
            dL_dq[i] = (self.L(q_plus, q_dot, t) - self.L(q_minus, q_dot, t)) / (2 * dx)

        dL_dq_dir = np.dot(dL_dq, direction)

        # Conserved momentum: p · direction
        dL_dqdot = np.zeros(self.n_dof)
        for i in range(self.n_dof):
            qdot_plus = q_dot.copy()
            qdot_minus = q_dot.copy()
            qdot_plus[i] += dx
            qdot_minus[i] -= dx
            dL_dqdot[i] = (self.L(q, qdot_plus, t) - self.L(q, qdot_minus, t)) / (2 * dx)

        p_dir = np.dot(dL_dqdot, direction)

        return abs(dL_dq_dir) < 1e-10, dL_dq_dir, p_dir

    def conserved_quantity(
        self,
        generator: Callable[[np.ndarray], np.ndarray],
        q: ArrayLike,
        q_dot: ArrayLike,
        t: float = 0.0,
        dx: float = 1e-8,
    ) -> float:
        """
        Calculate conserved quantity for given symmetry generator.

        For infinitesimal transformation q → q + ε·ξ(q), the conserved
        quantity is Q = p · ξ.

        Args:
            generator: Function ξ(q) generating the symmetry transformation
            q, q_dot: Current state
            t: Time

        Returns:
            Value of conserved quantity
        """
        q = np.array(q)
        q_dot = np.array(q_dot)

        # Generalized momentum p = ∂L/∂q_dot
        p = np.zeros(self.n_dof)
        for i in range(self.n_dof):
            qdot_plus = q_dot.copy()
            qdot_minus = q_dot.copy()
            qdot_plus[i] += dx
            qdot_minus[i] -= dx
            p[i] = (self.L(q, qdot_plus, t) - self.L(q, qdot_minus, t)) / (2 * dx)

        # Symmetry generator
        xi = generator(q)

        return np.dot(p, xi)

    def angular_momentum(
        self,
        axis: ArrayLike,
        q: ArrayLike,
        q_dot: ArrayLike,
        t: float = 0.0,
    ) -> float:
        """
        Calculate angular momentum about given axis.

        For 3D systems where q represents Cartesian positions.

        Args:
            axis: Rotation axis (will be normalized)
            q, q_dot: State (assumed to be 3D position/velocity or multiple thereof)
            t: Time

        Returns:
            Angular momentum component about axis
        """
        axis = np.array(axis)
        axis = axis / np.linalg.norm(axis)

        # Generator for rotation about axis: ξ = axis × q
        def rotation_generator(q):
            if len(q) == 3:
                return np.cross(axis, q)
            else:
                # Multiple particles
                xi = np.zeros_like(q)
                for i in range(0, len(q), 3):
                    xi[i:i+3] = np.cross(axis, q[i:i+3])
                return xi

        return self.conserved_quantity(rotation_generator, q, q_dot, t)
