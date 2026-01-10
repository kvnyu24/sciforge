"""
Experiment 2: Symplectic Euler/Verlet vs RK4 on oscillator (energy drift comparison).

Symplectic integrators preserve the Hamiltonian structure and show
bounded energy error, while non-symplectic methods show secular drift.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def symplectic_euler(x, v, dt, force_func, mass=1.0):
    """
    Symplectic Euler method.
    Updates velocity first, then position (semi-implicit).
    """
    v_new = v + dt * force_func(x) / mass
    x_new = x + dt * v_new
    return x_new, v_new


def velocity_verlet(x, v, dt, force_func, mass=1.0):
    """
    Velocity Verlet (leapfrog) integrator.
    Second order symplectic method.
    """
    a = force_func(x) / mass
    x_new = x + v * dt + 0.5 * a * dt**2
    a_new = force_func(x_new) / mass
    v_new = v + 0.5 * (a + a_new) * dt
    return x_new, v_new


def rk4_step(x, v, dt, force_func, mass=1.0):
    """Standard RK4 for comparison (non-symplectic)."""
    def f(state):
        x, v = state
        return np.array([v, force_func(x) / mass])

    y = np.array([x, v])
    k1 = f(y)
    k2 = f(y + dt/2 * k1)
    k3 = f(y + dt/2 * k2)
    k4 = f(y + dt * k3)
    y_new = y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    return y_new[0], y_new[1]


def explicit_euler(x, v, dt, force_func, mass=1.0):
    """Non-symplectic explicit Euler for comparison."""
    a = force_func(x) / mass
    x_new = x + v * dt
    v_new = v + a * dt
    return x_new, v_new


def harmonic_force(x, k=1.0):
    """Harmonic oscillator force: F = -kx"""
    return -k * x


def energy(x, v, k=1.0, m=1.0):
    """Total energy of harmonic oscillator."""
    return 0.5 * m * v**2 + 0.5 * k * x**2


def simulate(x0, v0, dt, n_steps, integrator, force_func, mass=1.0):
    """Run simulation with given integrator."""
    xs = [x0]
    vs = [v0]
    x, v = x0, v0

    for _ in range(n_steps):
        x, v = integrator(x, v, dt, force_func, mass)
        xs.append(x)
        vs.append(v)

    return np.array(xs), np.array(vs)


def main():
    # Parameters
    k = 4 * np.pi**2  # spring constant (omega = 2*pi)
    m = 1.0
    x0, v0 = 1.0, 0.0
    E0 = energy(x0, v0, k, m)

    dt = 0.02
    n_steps = 10000
    t = np.arange(n_steps + 1) * dt

    force = lambda x: harmonic_force(x, k)

    # Run all integrators
    integrators = {
        'Explicit Euler': explicit_euler,
        'Symplectic Euler': symplectic_euler,
        'Velocity Verlet': velocity_verlet,
        'RK4': rk4_step
    }

    results = {}
    for name, integrator in integrators.items():
        x, v = simulate(x0, v0, dt, n_steps, integrator, force, m)
        E = np.array([energy(xi, vi, k, m) for xi, vi in zip(x, v)])
        results[name] = {'x': x, 'v': v, 'E': E, 'dE': (E - E0) / E0}

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {'Explicit Euler': 'red', 'Symplectic Euler': 'blue',
              'Velocity Verlet': 'green', 'RK4': 'orange'}

    # Plot 1: Energy error vs time
    ax = axes[0, 0]
    for name, data in results.items():
        ax.plot(t, data['dE'] * 100, color=colors[name], label=name, alpha=0.8)
    ax.set_xlabel('Time')
    ax.set_ylabel('Energy error (%)')
    ax.set_title('Relative Energy Error vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, t[-1])

    # Plot 2: Phase space for long integration
    ax = axes[0, 1]
    for name in ['Explicit Euler', 'Velocity Verlet']:
        ax.plot(results[name]['x'], results[name]['v'],
                color=colors[name], alpha=0.5, lw=0.5, label=name)
    # Exact circle
    theta = np.linspace(0, 2*np.pi, 100)
    omega = np.sqrt(k/m)
    ax.plot(x0 * np.cos(theta), -x0 * omega * np.sin(theta), 'k--', lw=2, label='Exact')
    ax.set_xlabel('Position x')
    ax.set_ylabel('Velocity v')
    ax.set_title('Phase Space (Long Integration)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Plot 3: Energy as function of time (zoomed early times)
    ax = axes[1, 0]
    t_zoom = t[:500]
    for name, data in results.items():
        ax.plot(t_zoom, data['E'][:500], color=colors[name], label=name)
    ax.axhline(E0, color='black', linestyle='--', label='Initial E')
    ax.set_xlabel('Time')
    ax.set_ylabel('Total Energy')
    ax.set_title('Energy Conservation (Early Times)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 4: Maximum energy deviation vs dt
    ax = axes[1, 1]
    dts = np.logspace(-3, -1, 15)

    max_dE = {name: [] for name in integrators}

    for dt_test in dts:
        n_test = int(50 / dt_test)  # 50 time units
        for name, integrator in integrators.items():
            x, v = simulate(x0, v0, dt_test, n_test, integrator, force, m)
            E = np.array([energy(xi, vi, k, m) for xi, vi in zip(x, v)])
            max_dE[name].append(np.max(np.abs(E - E0)) / E0)

    for name in integrators:
        ax.loglog(dts, max_dE[name], 'o-', color=colors[name], label=name, markersize=4)

    # Reference slopes
    ax.loglog(dts, 0.5 * dts, 'k--', alpha=0.5, label='O(dt)')
    ax.loglog(dts, 0.3 * dts**2, 'k:', alpha=0.5, label='O(dt²)')

    ax.set_xlabel('Time step dt')
    ax.set_ylabel('Max relative energy error')
    ax.set_title('Energy Error Scaling with Step Size')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Symplectic vs Non-Symplectic Integrators\n' +
                 'Harmonic Oscillator: Bounded vs Secular Energy Drift',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'symplectic_integrators.png'), dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/symplectic_integrators.png")


if __name__ == "__main__":
    main()
