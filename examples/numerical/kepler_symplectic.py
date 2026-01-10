"""
Experiment 8: Hamiltonian system - Symplectic vs non-symplectic on Kepler orbit.

Compares orbital stability and energy/angular momentum conservation
for different integration methods on the gravitational two-body problem.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def kepler_force(r, mu=1.0):
    """Gravitational force: F = -mu * r / |r|^3"""
    r_mag = np.linalg.norm(r)
    if r_mag < 1e-10:
        return np.zeros_like(r)
    return -mu * r / r_mag**3


def symplectic_euler(r, v, dt, mu=1.0):
    """Symplectic Euler (semi-implicit)."""
    a = kepler_force(r, mu)
    v_new = v + dt * a
    r_new = r + dt * v_new
    return r_new, v_new


def velocity_verlet(r, v, dt, mu=1.0):
    """Velocity Verlet (leapfrog)."""
    a = kepler_force(r, mu)
    r_new = r + v * dt + 0.5 * a * dt**2
    a_new = kepler_force(r_new, mu)
    v_new = v + 0.5 * (a + a_new) * dt
    return r_new, v_new


def rk4_step(r, v, dt, mu=1.0):
    """RK4 integration step."""
    def f(state):
        r, v = state[:2], state[2:]
        return np.concatenate([v, kepler_force(r, mu)])

    state = np.concatenate([r, v])
    k1 = f(state)
    k2 = f(state + dt/2 * k1)
    k3 = f(state + dt/2 * k2)
    k4 = f(state + dt * k3)
    state_new = state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    return state_new[:2], state_new[2:]


def explicit_euler(r, v, dt, mu=1.0):
    """Explicit Euler (non-symplectic)."""
    a = kepler_force(r, mu)
    r_new = r + v * dt
    v_new = v + a * dt
    return r_new, v_new


def orbital_energy(r, v, mu=1.0):
    """Specific orbital energy: E = v^2/2 - mu/r"""
    return 0.5 * np.dot(v, v) - mu / np.linalg.norm(r)


def angular_momentum(r, v):
    """Specific angular momentum: L = r x v (z-component for 2D)"""
    return r[0] * v[1] - r[1] * v[0]


def simulate_orbit(r0, v0, n_orbits, dt, method, mu=1.0):
    """Simulate orbit for given number of orbital periods."""
    # Estimate orbital period from initial conditions
    E = orbital_energy(r0, v0, mu)
    a = -mu / (2 * E)  # Semi-major axis
    T = 2 * np.pi * np.sqrt(a**3 / mu)  # Orbital period

    n_steps = int(n_orbits * T / dt)

    r = np.array(r0, dtype=float)
    v = np.array(v0, dtype=float)

    rs = [r.copy()]
    vs = [v.copy()]

    for _ in range(n_steps):
        r, v = method(r, v, dt, mu)
        rs.append(r.copy())
        vs.append(v.copy())

    return np.array(rs), np.array(vs), T


def main():
    # Initial conditions for elliptical orbit (e = 0.5)
    mu = 1.0
    a = 1.0  # Semi-major axis
    e = 0.5  # Eccentricity

    # Start at perihelion
    r_peri = a * (1 - e)
    v_peri = np.sqrt(mu * (2/r_peri - 1/a))  # Vis-viva equation

    r0 = np.array([r_peri, 0.0])
    v0 = np.array([0.0, v_peri])

    E0 = orbital_energy(r0, v0, mu)
    L0 = angular_momentum(r0, v0)

    # Number of orbits and time step
    n_orbits = 50
    dt = 0.01

    # Methods to compare
    methods = {
        'Explicit Euler': explicit_euler,
        'Symplectic Euler': symplectic_euler,
        'Velocity Verlet': velocity_verlet,
        'RK4': rk4_step
    }

    results = {}
    for name, method in methods.items():
        print(f"Simulating {name}...")
        rs, vs, T = simulate_orbit(r0, v0, n_orbits, dt, method, mu)
        Es = np.array([orbital_energy(r, v, mu) for r, v in zip(rs, vs)])
        Ls = np.array([angular_momentum(r, v) for r, v in zip(rs, vs)])
        results[name] = {'r': rs, 'v': vs, 'E': Es, 'L': Ls, 'T': T}

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    colors = {'Explicit Euler': 'red', 'Symplectic Euler': 'blue',
              'Velocity Verlet': 'green', 'RK4': 'orange'}

    # Plot 1: Orbits
    ax = axes[0, 0]

    # Exact ellipse for reference
    theta = np.linspace(0, 2*np.pi, 200)
    r_exact = a * (1 - e**2) / (1 + e * np.cos(theta))
    x_exact = r_exact * np.cos(theta)
    y_exact = r_exact * np.sin(theta)
    ax.plot(x_exact, y_exact, 'k-', lw=2, alpha=0.3, label='Exact')

    for name in ['Explicit Euler', 'Velocity Verlet']:
        rs = results[name]['r']
        ax.plot(rs[:, 0], rs[:, 1], '-', color=colors[name], lw=0.5,
                alpha=0.7, label=name)

    ax.plot(0, 0, 'ko', markersize=10, label='Central body')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Orbital Trajectories ({n_orbits} orbits)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Plot 2: Energy conservation
    ax = axes[0, 1]
    T = results['Velocity Verlet']['T']
    t = np.arange(len(results['Velocity Verlet']['E'])) * dt

    for name, data in results.items():
        dE = (data['E'] - E0) / abs(E0) * 100
        ax.plot(t / T, dE, '-', color=colors[name], lw=1, label=name)

    ax.set_xlabel('Orbits')
    ax.set_ylabel('Energy error (%)')
    ax.set_title('Energy Conservation')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 3: Angular momentum conservation
    ax = axes[1, 0]

    for name, data in results.items():
        dL = (data['L'] - L0) / abs(L0) * 100
        ax.plot(t / T, dL, '-', color=colors[name], lw=1, label=name)

    ax.set_xlabel('Orbits')
    ax.set_ylabel('Angular momentum error (%)')
    ax.set_title('Angular Momentum Conservation')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 4: Phase space (r, v_r)
    ax = axes[1, 1]

    for name in ['Explicit Euler', 'Velocity Verlet', 'RK4']:
        rs = results[name]['r']
        vs = results[name]['v']
        r_mag = np.linalg.norm(rs, axis=1)
        v_r = np.sum(rs * vs, axis=1) / r_mag  # Radial velocity

        ax.plot(r_mag, v_r, '-', color=colors[name], lw=0.5, alpha=0.5, label=name)

    ax.set_xlabel('r')
    ax.set_ylabel('$v_r$')
    ax.set_title('Radial Phase Space')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Kepler Problem: Symplectic vs Non-Symplectic Integration\n' +
                 f'e = {e}, a = {a}, {n_orbits} orbits, dt = {dt}',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'kepler_symplectic.png'), dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/kepler_symplectic.png")

    # Print summary
    print("\nFinal energy errors after", n_orbits, "orbits:")
    for name, data in results.items():
        dE = abs(data['E'][-1] - E0) / abs(E0) * 100
        print(f"  {name}: {dE:.6f}%")


if __name__ == "__main__":
    main()
