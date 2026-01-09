"""Tests for numerical integrators."""

import pytest
import numpy as np

from sciforge.core import (
    euler_step,
    rk2_step,
    rk4_step,
    integrate,
    integrate_adaptive,
    DynamicsIntegrator,
)


class TestIntegrationSteps:
    """Tests for single-step integration methods."""

    def test_euler_constant(self):
        """Test Euler method with constant derivative."""
        # dy/dt = 1, y(0) = 0 => y(t) = t
        f = lambda t, y: 1.0
        y = euler_step(f, 0.0, 0.0, 0.1)
        assert abs(y - 0.1) < 1e-10

    def test_rk4_exponential_decay(self):
        """Test RK4 on exponential decay dy/dt = -y."""
        f = lambda t, y: -y
        y0 = 1.0
        dt = 0.1

        y = rk4_step(f, 0.0, y0, dt)
        expected = np.exp(-dt)

        # RK4 should be very accurate for this problem
        assert abs(y - expected) < 1e-6

    def test_rk4_vs_euler_accuracy(self):
        """Test that RK4 is more accurate than Euler."""
        f = lambda t, y: -y
        y0 = 1.0
        dt = 0.1

        y_euler = euler_step(f, 0.0, y0, dt)
        y_rk4 = rk4_step(f, 0.0, y0, dt)
        expected = np.exp(-dt)

        error_euler = abs(y_euler - expected)
        error_rk4 = abs(y_rk4 - expected)

        # RK4 should be much more accurate
        assert error_rk4 < error_euler / 10


class TestIntegrate:
    """Tests for the integrate function."""

    def test_exponential_decay(self):
        """Test integration of exponential decay."""
        f = lambda t, y: -y
        t, y = integrate(f, 1.0, (0, 5), dt=0.01, method="rk4")

        # y(t) = exp(-t)
        expected = np.exp(-t)
        np.testing.assert_allclose(y, expected, rtol=1e-4)

    def test_harmonic_oscillator(self):
        """Test integration of harmonic oscillator."""
        # d2x/dt2 = -x => [x, v] with dx/dt = v, dv/dt = -x
        def f(t, state):
            x, v = state
            return np.array([v, -x])

        t, y = integrate(f, np.array([1.0, 0.0]), (0, 2 * np.pi), dt=0.01)

        # After one period, should return to initial state
        np.testing.assert_allclose(y[-1], [1.0, 0.0], atol=0.01)

    def test_invalid_method_raises(self):
        """Test that invalid method raises ValueError."""
        f = lambda t, y: y
        with pytest.raises(ValueError):
            integrate(f, 1.0, (0, 1), dt=0.1, method="invalid")


class TestIntegrateAdaptive:
    """Tests for adaptive integration."""

    def test_exponential_decay(self):
        """Test adaptive integration of exponential decay."""
        f = lambda t, y: -y
        t, y = integrate_adaptive(f, 1.0, (0, 5), tol=1e-8)

        # Check final value
        expected_final = np.exp(-5)
        assert abs(y[-1] - expected_final) < 1e-6

    def test_step_size_adapts(self):
        """Test that step sizes are not uniform."""
        f = lambda t, y: -y
        t, _ = integrate_adaptive(f, 1.0, (0, 5), tol=1e-6)

        # With adaptive stepping, time intervals should vary
        dt = np.diff(t)
        assert dt.std() > 0  # Not all the same


class TestDynamicsIntegrator:
    """Tests for the DynamicsIntegrator class."""

    def test_free_fall(self):
        """Test integration of free fall motion."""
        g = 9.81
        accel_func = lambda pos, vel, t: np.array([0, 0, -g])

        integrator = DynamicsIntegrator(method="rk4")

        position = np.array([0.0, 0.0, 100.0])
        velocity = np.array([0.0, 0.0, 0.0])
        dt = 0.01

        # Integrate for 1 second
        for _ in range(100):
            position, velocity = integrator.step(
                accel_func, position, velocity, 0.0, dt
            )

        # After 1 second: z = 100 - 0.5*g*t^2 = 100 - 4.905 = 95.095
        expected_z = 100 - 0.5 * g * 1.0**2
        assert abs(position[2] - expected_z) < 0.01

        # Velocity should be -g*t = -9.81
        assert abs(velocity[2] - (-g)) < 0.01