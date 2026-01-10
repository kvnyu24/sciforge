"""
Experiment 1: Integrator comparison on harmonic oscillator.

Compares Euler, RK2, RK4, and adaptive RK45 methods on the simple
harmonic oscillator, measuring global error vs time step size.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def euler_step(y, t, dt, f):
    """Euler method single step."""
    return y + dt * f(t, y)


def rk2_step(y, t, dt, f):
    """Midpoint (RK2) method single step."""
    k1 = f(t, y)
    k2 = f(t + dt/2, y + dt/2 * k1)
    return y + dt * k2


def rk4_step(y, t, dt, f):
    """Classic RK4 single step."""
    k1 = f(t, y)
    k2 = f(t + dt/2, y + dt/2 * k1)
    k3 = f(t + dt/2, y + dt/2 * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6


def rk45_adaptive(y0, t_span, f, tol=1e-6, dt_init=0.01):
    """
    Adaptive RK45 (Dormand-Prince) integration.

    Returns times, states, and number of steps taken.
    """
    # Dormand-Prince coefficients
    c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])
    a = [
        [],
        [1/5],
        [3/40, 9/40],
        [44/45, -56/15, 32/9],
        [19372/6561, -25360/2187, 64448/6561, -212/729],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
        [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
    ]
    b5 = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
    b4 = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])

    t0, tf = t_span
    t = t0
    y = np.array(y0, dtype=float)
    dt = dt_init

    times = [t]
    states = [y.copy()]
    n_steps = 0

    while t < tf:
        if t + dt > tf:
            dt = tf - t

        # Compute k values
        k = np.zeros((7, len(y)))
        k[0] = f(t, y)
        for i in range(1, 7):
            y_temp = y.copy()
            for j in range(i):
                y_temp += dt * a[i][j] * k[j]
            k[i] = f(t + c[i] * dt, y_temp)

        # 5th and 4th order estimates
        y5 = y + dt * np.dot(b5, k)
        y4 = y + dt * np.dot(b4, k)

        # Error estimate
        error = np.linalg.norm(y5 - y4)

        if error < tol or dt < 1e-10:
            t += dt
            y = y5
            times.append(t)
            states.append(y.copy())
            n_steps += 1

        # Adjust step size
        if error > 0:
            dt = 0.9 * dt * (tol / error) ** 0.2
        dt = min(dt, tf - t) if t < tf else dt
        dt = max(dt, 1e-10)

    return np.array(times), np.array(states), n_steps


def harmonic_oscillator(t, y, omega=1.0):
    """Simple harmonic oscillator: y'' = -omega^2 * y"""
    x, v = y
    return np.array([v, -omega**2 * x])


def exact_solution(t, x0, v0, omega=1.0):
    """Exact solution for harmonic oscillator."""
    A = np.sqrt(x0**2 + (v0/omega)**2)
    phi = np.arctan2(v0/omega, x0)
    x = A * np.cos(omega * t - phi)
    v = -A * omega * np.sin(omega * t - phi)
    return np.array([x, v])


def integrate_fixed_step(y0, t_final, dt, step_func, f):
    """Integrate using fixed step method."""
    t = 0
    y = np.array(y0, dtype=float)
    times = [t]
    states = [y.copy()]

    while t < t_final:
        y = step_func(y, t, dt, f)
        t += dt
        times.append(t)
        states.append(y.copy())

    return np.array(times), np.array(states)


def main():
    # Parameters
    omega = 2 * np.pi  # frequency
    x0, v0 = 1.0, 0.0  # initial conditions
    t_final = 10.0
    y0 = [x0, v0]

    f = lambda t, y: harmonic_oscillator(t, y, omega)

    # Time steps to test
    dts = np.logspace(-4, -1, 20)

    # Collect errors for each method
    errors_euler = []
    errors_rk2 = []
    errors_rk4 = []

    for dt in dts:
        # Euler
        t, states = integrate_fixed_step(y0, t_final, dt, euler_step, f)
        exact = np.array([exact_solution(ti, x0, v0, omega) for ti in t])
        errors_euler.append(np.max(np.abs(states[:, 0] - exact[:, 0])))

        # RK2
        t, states = integrate_fixed_step(y0, t_final, dt, rk2_step, f)
        exact = np.array([exact_solution(ti, x0, v0, omega) for ti in t])
        errors_rk2.append(np.max(np.abs(states[:, 0] - exact[:, 0])))

        # RK4
        t, states = integrate_fixed_step(y0, t_final, dt, rk4_step, f)
        exact = np.array([exact_solution(ti, x0, v0, omega) for ti in t])
        errors_rk4.append(np.max(np.abs(states[:, 0] - exact[:, 0])))

    # Adaptive RK45 for comparison
    t_rk45, states_rk45, n_steps_rk45 = rk45_adaptive(y0, (0, t_final), f, tol=1e-8)
    exact_rk45 = np.array([exact_solution(ti, x0, v0, omega) for ti in t_rk45])
    error_rk45 = np.max(np.abs(states_rk45[:, 0] - exact_rk45[:, 0]))

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Global error vs dt (log-log)
    ax = axes[0, 0]
    ax.loglog(dts, errors_euler, 'o-', label='Euler (O(h))', markersize=4)
    ax.loglog(dts, errors_rk2, 's-', label='RK2 (O(h²))', markersize=4)
    ax.loglog(dts, errors_rk4, '^-', label='RK4 (O(h⁴))', markersize=4)

    # Reference slopes
    ax.loglog(dts, 0.5 * dts, 'k--', alpha=0.5, label='O(h)')
    ax.loglog(dts, 0.1 * dts**2, 'k:', alpha=0.5, label='O(h²)')
    ax.loglog(dts, 0.01 * dts**4, 'k-.', alpha=0.5, label='O(h⁴)')

    ax.set_xlabel('Time step dt')
    ax.set_ylabel('Maximum global error')
    ax.set_title('Global Error vs Time Step')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: Time series comparison at medium dt
    ax = axes[0, 1]
    dt_demo = 0.05
    t_exact = np.linspace(0, t_final, 1000)
    exact_demo = np.array([exact_solution(ti, x0, v0, omega) for ti in t_exact])

    t_e, s_e = integrate_fixed_step(y0, t_final, dt_demo, euler_step, f)
    t_r4, s_r4 = integrate_fixed_step(y0, t_final, dt_demo, rk4_step, f)

    ax.plot(t_exact, exact_demo[:, 0], 'k-', lw=2, label='Exact')
    ax.plot(t_e, s_e[:, 0], 'b--', alpha=0.7, label=f'Euler (dt={dt_demo})')
    ax.plot(t_r4, s_r4[:, 0], 'r-.', alpha=0.7, label=f'RK4 (dt={dt_demo})')
    ax.set_xlabel('Time')
    ax.set_ylabel('Position x(t)')
    ax.set_title(f'Time Series Comparison (dt = {dt_demo})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Phase space
    ax = axes[1, 0]
    ax.plot(exact_demo[:, 0], exact_demo[:, 1], 'k-', lw=2, label='Exact (ellipse)')
    ax.plot(s_e[:, 0], s_e[:, 1], 'b--', alpha=0.7, label='Euler')
    ax.plot(s_r4[:, 0], s_r4[:, 1], 'r-.', alpha=0.7, label='RK4')
    ax.set_xlabel('Position x')
    ax.set_ylabel('Velocity v')
    ax.set_title('Phase Space (Euler spirals out, RK4 conserves)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Plot 4: Adaptive step size
    ax = axes[1, 1]
    ax.plot(t_rk45[:-1], np.diff(t_rk45), 'g-', lw=1.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Step size dt')
    ax.set_title(f'RK45 Adaptive Steps ({n_steps_rk45} total, error={error_rk45:.2e})')
    ax.grid(True, alpha=0.3)

    plt.suptitle('ODE Integrator Comparison: Harmonic Oscillator\n' +
                 r'$\ddot{x} = -\omega^2 x$, $\omega = 2\pi$', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'integrator_comparison.png'), dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/integrator_comparison.png")


if __name__ == "__main__":
    main()
