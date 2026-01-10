"""
Experiment 36: Effective Potential - Central Force Orbits (Bound/Unbound)

This example demonstrates the effective potential approach to analyzing
central force motion. Shows how angular momentum creates a centrifugal
barrier and determines whether orbits are bound or unbound.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def effective_potential(r, L, m=1.0, G=1.0, M=1.0):
    """
    Calculate effective potential for gravitational central force.

    V_eff(r) = -GMm/r + L^2/(2mr^2)

    Args:
        r: Radial distance
        L: Angular momentum
        m: Orbiting mass
        G: Gravitational constant
        M: Central mass

    Returns:
        Effective potential at radius r
    """
    gravitational = -G * M * m / r
    centrifugal = L**2 / (2 * m * r**2)
    return gravitational + centrifugal


def simulate_orbit(r0, v_r0, v_theta0, L, m=1.0, G=1.0, M=1.0, t_final=20.0, dt=0.001):
    """
    Simulate central force motion using radial equation of motion.

    Args:
        r0: Initial radial distance
        v_r0: Initial radial velocity
        v_theta0: Initial tangential velocity (used to compute L if L not given)
        L: Angular momentum
        m: Orbiting mass
        G: Gravitational constant
        M: Central mass
        t_final: Simulation duration
        dt: Time step

    Returns:
        Dictionary with trajectory data
    """
    # Initial conditions
    r = r0
    v_r = v_r0
    theta = 0.0

    # Calculate angular momentum from initial tangential velocity
    # L = m * r * v_theta
    if L is None:
        L = m * r0 * v_theta0

    times = [0]
    rs = [r]
    thetas = [theta]
    v_rs = [v_r]
    energies = []

    # Total energy
    E = 0.5 * m * v_r**2 + effective_potential(r, L, m, G, M)
    energies.append(E)

    t = 0
    while t < t_final:
        # Radial acceleration: a_r = -dV_eff/dr
        # d/dr(-GMm/r + L^2/(2mr^2)) = GMm/r^2 - L^2/(mr^3)
        a_r = G * M / r**2 - L**2 / (m**2 * r**3)

        # RK4 for radial motion
        k1_r = v_r
        k1_v = a_r

        r_temp = r + 0.5 * dt * k1_r
        if r_temp > 0:
            a_temp = G * M / r_temp**2 - L**2 / (m**2 * r_temp**3)
        else:
            break
        k2_r = v_r + 0.5 * dt * k1_v
        k2_v = a_temp

        r_temp = r + 0.5 * dt * k2_r
        if r_temp > 0:
            a_temp = G * M / r_temp**2 - L**2 / (m**2 * r_temp**3)
        else:
            break
        k3_r = v_r + 0.5 * dt * k2_v
        k3_v = a_temp

        r_temp = r + dt * k3_r
        if r_temp > 0:
            a_temp = G * M / r_temp**2 - L**2 / (m**2 * r_temp**3)
        else:
            break
        k4_r = v_r + dt * k3_v
        k4_v = a_temp

        r_new = r + (dt / 6) * (k1_r + 2*k2_r + 2*k3_r + k4_r)
        v_r_new = v_r + (dt / 6) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

        if r_new <= 0.01:  # Prevent singularity
            break

        # Angular motion: d(theta)/dt = L / (m * r^2)
        omega = L / (m * r**2)
        theta_new = theta + omega * dt

        # Update state
        r = r_new
        v_r = v_r_new
        theta = theta_new
        t += dt

        times.append(t)
        rs.append(r)
        thetas.append(theta)
        v_rs.append(v_r)

        E = 0.5 * m * v_r**2 + effective_potential(r, L, m, G, M)
        energies.append(E)

    # Convert to Cartesian
    rs = np.array(rs)
    thetas = np.array(thetas)
    xs = rs * np.cos(thetas)
    ys = rs * np.sin(thetas)

    return {
        'time': np.array(times),
        'r': rs,
        'theta': thetas,
        'v_r': np.array(v_rs),
        'x': xs,
        'y': ys,
        'energy': np.array(energies),
        'L': L
    }


def main():
    # Parameters
    m = 1.0  # Orbiting mass
    G = 1.0  # Gravitational constant
    M = 1.0  # Central mass

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))

    # Subplot 1: Effective potential for different L values
    ax1 = fig.add_subplot(2, 3, 1)

    r_range = np.linspace(0.3, 5, 500)
    L_values = [0.5, 0.75, 1.0, 1.25, 1.5]

    for L in L_values:
        V_eff = effective_potential(r_range, L, m, G, M)
        ax1.plot(r_range, V_eff, lw=2, label=f'L = {L}')

    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Radial distance r')
    ax1.set_ylabel('Effective Potential V_eff')
    ax1.set_title('Effective Potential for Different L')
    ax1.set_ylim(-3, 2)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Energy levels and turning points
    ax2 = fig.add_subplot(2, 3, 2)

    L = 1.0
    V_eff = effective_potential(r_range, L, m, G, M)
    ax2.plot(r_range, V_eff, 'b-', lw=2, label=f'V_eff (L={L})')

    # Different energy levels
    E_bound = -0.3  # Bound orbit (E < 0)
    E_marginal = 0  # Parabolic orbit (E = 0)
    E_unbound = 0.3  # Hyperbolic orbit (E > 0)

    ax2.axhline(y=E_bound, color='g', linestyle='-', alpha=0.7, label=f'E = {E_bound} (bound)')
    ax2.axhline(y=E_marginal, color='orange', linestyle='-', alpha=0.7, label=f'E = {E_marginal} (marginal)')
    ax2.axhline(y=E_unbound, color='r', linestyle='-', alpha=0.7, label=f'E = {E_unbound} (unbound)')

    # Find and mark turning points for bound orbit
    V_min_idx = np.argmin(V_eff)
    r_min = r_range[V_min_idx]
    V_min = V_eff[V_min_idx]

    ax2.plot(r_min, V_min, 'ko', markersize=8)
    ax2.annotate('r_min (circular orbit)', (r_min, V_min), textcoords="offset points",
                 xytext=(10, 10), fontsize=8)

    ax2.set_xlabel('Radial distance r')
    ax2.set_ylabel('Energy')
    ax2.set_title('Energy Levels and Turning Points')
    ax2.set_ylim(-1, 1)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Bound orbits (elliptical)
    ax3 = fig.add_subplot(2, 3, 3)

    # Different eccentricities achieved by different initial conditions
    orbit_cases = [
        {'r0': 1.0, 'v_r0': 0.0, 'L': 1.0, 'label': 'Circular', 'color': 'blue'},
        {'r0': 1.0, 'v_r0': 0.2, 'L': 0.9, 'label': 'Ellipse (low e)', 'color': 'green'},
        {'r0': 1.0, 'v_r0': 0.4, 'L': 0.8, 'label': 'Ellipse (high e)', 'color': 'orange'},
    ]

    for case in orbit_cases:
        results = simulate_orbit(
            r0=case['r0'], v_r0=case['v_r0'], v_theta0=0,
            L=case['L'], m=m, G=G, M=M, t_final=30.0, dt=0.001
        )
        ax3.plot(results['x'], results['y'], lw=1.5, label=case['label'], color=case['color'])

    ax3.plot(0, 0, 'ko', markersize=10, label='Central mass')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('Bound Orbits')
    ax3.set_aspect('equal')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Subplot 4: Unbound orbit (hyperbolic)
    ax4 = fig.add_subplot(2, 3, 4)

    # Hyperbolic trajectory (positive total energy)
    unbound_cases = [
        {'r0': 3.0, 'v_r0': -0.8, 'L': 1.5, 'label': 'Hyperbola 1', 'color': 'red'},
        {'r0': 3.0, 'v_r0': -0.6, 'L': 1.2, 'label': 'Hyperbola 2', 'color': 'purple'},
    ]

    for case in unbound_cases:
        results = simulate_orbit(
            r0=case['r0'], v_r0=case['v_r0'], v_theta0=0,
            L=case['L'], m=m, G=G, M=M, t_final=15.0, dt=0.001
        )
        ax4.plot(results['x'], results['y'], lw=1.5, label=case['label'], color=case['color'])

    ax4.plot(0, 0, 'ko', markersize=10, label='Central mass')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_title('Unbound (Hyperbolic) Orbits')
    ax4.set_aspect('equal')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Subplot 5: r(t) for bound vs unbound
    ax5 = fig.add_subplot(2, 3, 5)

    # Bound orbit
    results_bound = simulate_orbit(r0=1.0, v_r0=0.3, v_theta0=0, L=0.85,
                                   m=m, G=G, M=M, t_final=30.0, dt=0.001)
    ax5.plot(results_bound['time'], results_bound['r'], 'b-', lw=2, label='Bound orbit')

    # Unbound orbit
    results_unbound = simulate_orbit(r0=2.0, v_r0=-0.5, v_theta0=0, L=1.5,
                                     m=m, G=G, M=M, t_final=15.0, dt=0.001)
    ax5.plot(results_unbound['time'], results_unbound['r'], 'r-', lw=2, label='Unbound orbit')

    ax5.set_xlabel('Time')
    ax5.set_ylabel('Radial distance r')
    ax5.set_title('Radial Distance vs Time')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Subplot 6: Energy conservation
    ax6 = fig.add_subplot(2, 3, 6)

    for results, label, color in [(results_bound, 'Bound', 'blue'),
                                   (results_unbound, 'Unbound', 'red')]:
        E_normalized = results['energy'] / results['energy'][0]
        ax6.plot(results['time'], E_normalized, color=color, lw=2, label=label)

    ax6.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Time')
    ax6.set_ylabel('E(t) / E(0)')
    ax6.set_title('Energy Conservation')
    ax6.set_ylim(0.95, 1.05)
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.suptitle('Effective Potential and Central Force Orbits\n'
                 'V_eff(r) = -GMm/r + L^2/(2mr^2)',
                 fontsize=12, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'effective_potential.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'effective_potential.png')}")


if __name__ == "__main__":
    main()
