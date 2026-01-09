"""
Numerical integrators for solving ordinary differential equations.

This module provides reusable integration methods that can be used
across all physics simulations in SciForge.
"""

import numpy as np
from typing import Callable, Tuple, Optional, Union
from numpy.typing import ArrayLike

from .base import ArrayType


# Type aliases
StateType = Union[np.ndarray, float]
DerivativeFunc = Callable[[float, StateType], StateType]


def euler_step(
    f: DerivativeFunc,
    t: float,
    y: StateType,
    dt: float,
) -> StateType:
    """
    Perform one Euler integration step.

    The Euler method is the simplest integration method:
        y_{n+1} = y_n + dt * f(t_n, y_n)

    Args:
        f: Derivative function f(t, y) -> dy/dt
        t: Current time
        y: Current state
        dt: Time step

    Returns:
        New state after one step

    Note:
        First-order accurate. Use for simple problems or as a baseline.
    """
    return y + dt * f(t, y)


def rk2_step(
    f: DerivativeFunc,
    t: float,
    y: StateType,
    dt: float,
) -> StateType:
    """
    Perform one second-order Runge-Kutta (midpoint method) step.

    The midpoint method:
        k1 = f(t_n, y_n)
        k2 = f(t_n + dt/2, y_n + dt/2 * k1)
        y_{n+1} = y_n + dt * k2

    Args:
        f: Derivative function f(t, y) -> dy/dt
        t: Current time
        y: Current state
        dt: Time step

    Returns:
        New state after one step

    Note:
        Second-order accurate. Good balance of speed and accuracy.
    """
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
    return y + dt * k2


def rk4_step(
    f: DerivativeFunc,
    t: float,
    y: StateType,
    dt: float,
) -> StateType:
    """
    Perform one fourth-order Runge-Kutta step.

    The classic RK4 method:
        k1 = f(t_n, y_n)
        k2 = f(t_n + dt/2, y_n + dt/2 * k1)
        k3 = f(t_n + dt/2, y_n + dt/2 * k2)
        k4 = f(t_n + dt, y_n + dt * k3)
        y_{n+1} = y_n + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

    Args:
        f: Derivative function f(t, y) -> dy/dt
        t: Current time
        y: Current state
        dt: Time step

    Returns:
        New state after one step

    Note:
        Fourth-order accurate. The standard choice for most problems.
    """
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def rk45_step(
    f: DerivativeFunc,
    t: float,
    y: StateType,
    dt: float,
) -> Tuple[StateType, StateType, float]:
    """
    Perform one Runge-Kutta-Fehlberg 4(5) step with error estimation.

    This method computes both 4th and 5th order solutions to estimate
    the local truncation error, enabling adaptive step size control.

    Args:
        f: Derivative function f(t, y) -> dy/dt
        t: Current time
        y: Current state
        dt: Time step

    Returns:
        Tuple of (y4, y5, error) where:
        - y4: 4th order solution (used for stepping)
        - y5: 5th order solution (used for error estimation)
        - error: Estimated error ||y5 - y4||

    Note:
        Use y4 as the solution and the error to adapt step size.
    """
    # Fehlberg coefficients
    k1 = f(t, y)
    k2 = f(t + dt / 4, y + dt * k1 / 4)
    k3 = f(t + 3 * dt / 8, y + dt * (3 * k1 / 32 + 9 * k2 / 32))
    k4 = f(
        t + 12 * dt / 13,
        y + dt * (1932 * k1 / 2197 - 7200 * k2 / 2197 + 7296 * k3 / 2197),
    )
    k5 = f(
        t + dt,
        y
        + dt
        * (439 * k1 / 216 - 8 * k2 + 3680 * k3 / 513 - 845 * k4 / 4104),
    )
    k6 = f(
        t + dt / 2,
        y
        + dt
        * (
            -8 * k1 / 27
            + 2 * k2
            - 3544 * k3 / 2565
            + 1859 * k4 / 4104
            - 11 * k5 / 40
        ),
    )

    # 4th order solution
    y4 = y + dt * (25 * k1 / 216 + 1408 * k3 / 2565 + 2197 * k4 / 4104 - k5 / 5)

    # 5th order solution
    y5 = y + dt * (
        16 * k1 / 135
        + 6656 * k3 / 12825
        + 28561 * k4 / 56430
        - 9 * k5 / 50
        + 2 * k6 / 55
    )

    # Error estimate
    error = np.linalg.norm(y5 - y4) if hasattr(y5, "__len__") else abs(y5 - y4)

    return y4, y5, error


def integrate(
    f: DerivativeFunc,
    y0: StateType,
    t_span: Tuple[float, float],
    dt: float,
    method: str = "rk4",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Integrate an ODE over a time span.

    Args:
        f: Derivative function f(t, y) -> dy/dt
        y0: Initial state
        t_span: Tuple of (t_start, t_end)
        dt: Time step
        method: Integration method ("euler", "rk2", "rk4")

    Returns:
        Tuple of (times, states) arrays

    Examples:
        >>> def f(t, y): return -y  # dy/dt = -y
        >>> t, y = integrate(f, 1.0, (0, 5), dt=0.1)
        >>> y[-1]  # Should be approximately exp(-5)
        0.00673...
    """
    methods = {
        "euler": euler_step,
        "rk2": rk2_step,
        "rk4": rk4_step,
    }

    if method not in methods:
        raise ValueError(f"Unknown method: {method}. Choose from {list(methods.keys())}")

    step_func = methods[method]
    t_start, t_end = t_span

    # Initialize arrays
    n_steps = int((t_end - t_start) / dt) + 1
    times = np.linspace(t_start, t_end, n_steps)

    y0_arr = np.asarray(y0)
    if y0_arr.ndim == 0:
        states = np.zeros(n_steps)
    else:
        states = np.zeros((n_steps, len(y0_arr)))

    states[0] = y0

    # Integration loop
    y = y0_arr.copy() if hasattr(y0_arr, "copy") else y0
    for i in range(1, n_steps):
        y = step_func(f, times[i - 1], y, dt)
        states[i] = y

    return times, states


def integrate_adaptive(
    f: DerivativeFunc,
    y0: StateType,
    t_span: Tuple[float, float],
    tol: float = 1e-6,
    dt_init: float = 0.01,
    dt_min: float = 1e-10,
    dt_max: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Integrate an ODE with adaptive step size control.

    Uses RK45 method with error estimation to automatically adjust
    the step size for efficient integration.

    Args:
        f: Derivative function f(t, y) -> dy/dt
        y0: Initial state
        t_span: Tuple of (t_start, t_end)
        tol: Error tolerance
        dt_init: Initial time step
        dt_min: Minimum allowed time step
        dt_max: Maximum allowed time step

    Returns:
        Tuple of (times, states) arrays

    Note:
        The returned arrays may have variable spacing in time.
    """
    t_start, t_end = t_span
    t = t_start
    dt = dt_init

    times = [t]
    y = np.asarray(y0)
    states = [y.copy() if hasattr(y, "copy") else y]

    while t < t_end:
        # Don't overshoot
        if t + dt > t_end:
            dt = t_end - t

        # Take a step
        y4, y5, error = rk45_step(f, t, y, dt)

        # Check if error is acceptable
        if error < tol or dt <= dt_min:
            # Accept step
            t += dt
            y = y4
            times.append(t)
            states.append(y.copy() if hasattr(y, "copy") else y)

        # Adjust step size
        if error > 0:
            # Safety factor of 0.9
            dt_new = 0.9 * dt * (tol / error) ** 0.2
            dt = max(dt_min, min(dt_max, dt_new))
        else:
            dt = min(dt_max, dt * 2)

    return np.array(times), np.array(states)


class DynamicsIntegrator:
    """
    Specialized integrator for second-order dynamics (position + velocity).

    This class provides convenience methods for integrating Newton's
    equations of motion where the state consists of position and velocity.

    Attributes:
        method: Integration method to use
    """

    def __init__(self, method: str = "rk4"):
        """
        Initialize the dynamics integrator.

        Args:
            method: Integration method ("euler", "rk2", "rk4")
        """
        self.method = method
        self._step_funcs = {
            "euler": euler_step,
            "rk2": rk2_step,
            "rk4": rk4_step,
        }

        if method not in self._step_funcs:
            raise ValueError(
                f"Unknown method: {method}. Choose from {list(self._step_funcs.keys())}"
            )

    def step(
        self,
        acceleration_func: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
        position: np.ndarray,
        velocity: np.ndarray,
        t: float,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform one integration step for second-order dynamics.

        Args:
            acceleration_func: Function(position, velocity, time) -> acceleration
            position: Current position
            velocity: Current velocity
            t: Current time
            dt: Time step

        Returns:
            Tuple of (new_position, new_velocity)
        """
        # Combine state
        state = np.concatenate([position, velocity])
        n = len(position)

        # Define the combined derivative function
        def f(t: float, y: np.ndarray) -> np.ndarray:
            pos = y[:n]
            vel = y[n:]
            acc = acceleration_func(pos, vel, t)
            return np.concatenate([vel, acc])

        # Take a step
        step_func = self._step_funcs[self.method]
        new_state = step_func(f, t, state, dt)

        return new_state[:n], new_state[n:]