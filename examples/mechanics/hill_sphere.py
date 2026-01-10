"""
Example demonstrating the Hill sphere concept.

The Hill sphere is the region around a secondary body (like a planet)
within which the secondary body's gravity dominates over the primary
(like a star). Objects within the Hill sphere can remain bound to the
secondary body, while objects outside will escape to the primary.

The Hill sphere radius is: r_H = a * (m / 3M)^(1/3)

where:
- a = orbital radius of secondary around primary
- m = mass of secondary body
- M = mass of primary body

This example demonstrates:
1. Calculation of Hill sphere radii for solar system bodies
2. Simulation of particle capture and escape at the Hill sphere boundary
3. Comparison of stability inside vs outside the Hill sphere
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def hill_sphere_radius(a, m_secondary, m_primary):
    """
    Calculate Hill sphere radius.

    Args:
        a: Semi-major axis of secondary's orbit around primary
        m_secondary: Mass of secondary body
        m_primary: Mass of primary body

    Returns:
        Hill sphere radius
    """
    return a * (m_secondary / (3 * m_primary))**(1/3)


def simulate_test_particle_three_body(m_primary, m_secondary, a_orbit,
                                       r_particle, v_particle,
                                       t_final, dt, G=1.0):
    """
    Simulate a test particle in a restricted three-body system.

    The primary is at the origin, and the secondary orbits in a circle.

    Args:
        m_primary: Mass of primary (at origin)
        m_secondary: Mass of secondary (orbiting)
        a_orbit: Orbital radius of secondary
        r_particle: Initial position of test particle (relative to secondary)
        v_particle: Initial velocity of test particle (relative to secondary)
        t_final: Simulation duration
        dt: Time step
        G: Gravitational constant

    Returns:
        Dictionary with trajectory data
    """
    # Secondary's orbital angular velocity
    omega = np.sqrt(G * (m_primary + m_secondary) / a_orbit**3)

    # Initial conditions
    # Secondary starts at (a_orbit, 0, 0)
    theta_secondary = 0.0
    r_secondary = np.array([a_orbit, 0.0, 0.0])
    v_secondary = np.array([0.0, a_orbit * omega, 0.0])

    # Particle position in inertial frame
    r = r_secondary + np.array(r_particle)
    v = v_secondary + np.array(v_particle)

    positions = [r.copy()]
    rel_positions = [r - r_secondary]
    times = [0.0]
    secondary_positions = [r_secondary.copy()]

    t = 0
    while t < t_final:
        # Update secondary position (circular orbit)
        theta_secondary += omega * dt
        r_secondary = np.array([
            a_orbit * np.cos(theta_secondary),
            a_orbit * np.sin(theta_secondary),
            0.0
        ])
        v_secondary = np.array([
            -a_orbit * omega * np.sin(theta_secondary),
            a_orbit * omega * np.cos(theta_secondary),
            0.0
        ])

        # Accelerations on test particle
        # From primary
        r_to_primary = -r
        r_p = np.linalg.norm(r_to_primary)
        a_primary = G * m_primary * r_to_primary / r_p**3

        # From secondary
        r_to_secondary = r_secondary - r
        r_s = np.linalg.norm(r_to_secondary)
        if r_s > 0.01:  # Avoid singularity
            a_secondary = G * m_secondary * r_to_secondary / r_s**3
        else:
            a_secondary = np.zeros(3)

        # Total acceleration
        a_total = a_primary + a_secondary

        # Simple Euler integration (sufficient for demonstration)
        v += a_total * dt
        r += v * dt

        t += dt
        positions.append(r.copy())
        rel_positions.append(r - r_secondary)
        times.append(t)
        secondary_positions.append(r_secondary.copy())

    return {
        'positions': np.array(positions),
        'rel_positions': np.array(rel_positions),
        'times': np.array(times),
        'secondary_positions': np.array(secondary_positions)
    }


def main():
    # Use solar system-like units
    # 1 AU, 1 solar mass, G = 4*pi^2 AU^3/(M_sun * year^2)
    G = 1.0
    M_star = 1.0

    # Planet parameters (Earth-like)
    m_planet = 3e-6 * M_star  # Earth/Sun mass ratio
    a_planet = 1.0  # 1 AU

    # Calculate Hill sphere
    r_hill = hill_sphere_radius(a_planet, m_planet, M_star)
    print(f"Hill Sphere Radius: {r_hill:.6f} (orbital radii)")
    print(f"Hill Sphere in AU: {r_hill:.6f} AU")
    print(f"Hill Sphere in km (if 1 AU = 1.5e8 km): {r_hill * 1.5e8:.0f} km")

    fig = plt.figure(figsize=(18, 12))

    # Plot 1: Hill spheres of solar system bodies
    ax1 = fig.add_subplot(2, 3, 1)

    # Solar system data (AU and relative masses)
    solar_system = {
        'Mercury': {'a': 0.387, 'm_ratio': 1.66e-7},
        'Venus': {'a': 0.723, 'm_ratio': 2.45e-6},
        'Earth': {'a': 1.0, 'm_ratio': 3.00e-6},
        'Mars': {'a': 1.524, 'm_ratio': 3.23e-7},
        'Jupiter': {'a': 5.203, 'm_ratio': 9.55e-4},
        'Saturn': {'a': 9.537, 'm_ratio': 2.86e-4},
        'Uranus': {'a': 19.19, 'm_ratio': 4.37e-5},
        'Neptune': {'a': 30.07, 'm_ratio': 5.15e-5},
    }

    names = []
    hill_radii = []
    orbital_radii = []

    for name, data in solar_system.items():
        r_h = hill_sphere_radius(data['a'], data['m_ratio'], 1.0)
        names.append(name)
        hill_radii.append(r_h)
        orbital_radii.append(data['a'])

    # Plot as bar chart (log scale)
    x = np.arange(len(names))
    colors = plt.cm.viridis(np.linspace(0, 1, len(names)))

    bars = ax1.bar(x, hill_radii, color=colors)
    ax1.set_yscale('log')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.set_ylabel('Hill Sphere Radius (AU)')
    ax1.set_title('Hill Sphere Radii of Solar System Planets')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for bar, r_h in zip(bars, hill_radii):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{r_h:.4f}', ha='center', va='bottom', fontsize=7)

    # Plot 2: Schematic of Hill sphere
    ax2 = fig.add_subplot(2, 3, 2)

    # Draw primary (star) at origin
    ax2.plot(0, 0, 'yo', markersize=20, label='Star')

    # Draw secondary (planet) orbit
    theta = np.linspace(0, 2*np.pi, 100)
    ax2.plot(a_planet * np.cos(theta), a_planet * np.sin(theta),
             'k--', lw=1, alpha=0.5)

    # Draw planet
    planet_pos = np.array([a_planet, 0])
    ax2.plot(planet_pos[0], planet_pos[1], 'bo', markersize=10, label='Planet')

    # Draw Hill sphere
    hill_circle = Circle(planet_pos, r_hill, fill=False, color='green',
                         lw=2, linestyle='-', label=f'Hill Sphere (r_H = {r_hill:.4f})')
    ax2.add_patch(hill_circle)

    # Draw Lagrange points L1 and L2
    L1 = planet_pos - np.array([r_hill, 0])
    L2 = planet_pos + np.array([r_hill, 0])
    ax2.plot(L1[0], L1[1], 'r*', markersize=12, label='L1')
    ax2.plot(L2[0], L2[1], 'm*', markersize=12, label='L2')

    ax2.set_xlabel('x (AU)')
    ax2.set_ylabel('y (AU)')
    ax2.set_title('Hill Sphere Schematic')
    ax2.set_aspect('equal')
    ax2.legend(fontsize=8, loc='upper left')
    ax2.set_xlim(-0.5, 1.5)
    ax2.set_ylim(-0.5, 0.5)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Particle trajectories inside Hill sphere (stable)
    ax3 = fig.add_subplot(2, 3, 3)

    # Larger planet for clearer demonstration
    m_planet_demo = 0.001  # Jupiter-like
    a_demo = 1.0
    r_hill_demo = hill_sphere_radius(a_demo, m_planet_demo, M_star)

    T_orbit = 2 * np.pi * np.sqrt(a_demo**3 / (G * M_star))
    dt = T_orbit / 500

    # Test particles at different distances from planet
    distances = [0.3 * r_hill_demo, 0.5 * r_hill_demo, 0.7 * r_hill_demo]
    colors = ['green', 'blue', 'cyan']

    for dist, color in zip(distances, colors):
        # Prograde circular orbit around planet
        v_orb = np.sqrt(G * m_planet_demo / dist)

        result = simulate_test_particle_three_body(
            m_primary=M_star,
            m_secondary=m_planet_demo,
            a_orbit=a_demo,
            r_particle=[dist, 0, 0],
            v_particle=[0, v_orb, 0],
            t_final=10 * T_orbit,
            dt=dt,
            G=G
        )

        rel_pos = result['rel_positions']
        ax3.plot(rel_pos[:, 0] / r_hill_demo, rel_pos[:, 1] / r_hill_demo,
                 color=color, lw=0.5, alpha=0.7,
                 label=f'r = {dist/r_hill_demo:.1f} r_H')

    # Draw Hill sphere boundary
    theta = np.linspace(0, 2*np.pi, 100)
    ax3.plot(np.cos(theta), np.sin(theta), 'r--', lw=2, label='Hill sphere')
    ax3.plot(0, 0, 'bo', markersize=10)  # Planet at origin

    ax3.set_xlabel('x / r_Hill')
    ax3.set_ylabel('y / r_Hill')
    ax3.set_title('Stable Orbits Inside Hill Sphere\n(relative to planet)')
    ax3.set_aspect('equal')
    ax3.legend(fontsize=8)
    ax3.set_xlim(-1.5, 1.5)
    ax3.set_ylim(-1.5, 1.5)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Particle escape outside Hill sphere
    ax4 = fig.add_subplot(2, 3, 4)

    distances = [0.8 * r_hill_demo, 1.0 * r_hill_demo, 1.2 * r_hill_demo]
    colors = ['green', 'orange', 'red']

    for dist, color in zip(distances, colors):
        v_orb = np.sqrt(G * m_planet_demo / dist)

        result = simulate_test_particle_three_body(
            m_primary=M_star,
            m_secondary=m_planet_demo,
            a_orbit=a_demo,
            r_particle=[dist, 0, 0],
            v_particle=[0, v_orb, 0],
            t_final=10 * T_orbit,
            dt=dt,
            G=G
        )

        rel_pos = result['rel_positions']
        ax4.plot(rel_pos[:, 0] / r_hill_demo, rel_pos[:, 1] / r_hill_demo,
                 color=color, lw=0.5, alpha=0.7,
                 label=f'r = {dist/r_hill_demo:.1f} r_H')

    theta = np.linspace(0, 2*np.pi, 100)
    ax4.plot(np.cos(theta), np.sin(theta), 'r--', lw=2, label='Hill sphere')
    ax4.plot(0, 0, 'bo', markersize=10)

    ax4.set_xlabel('x / r_Hill')
    ax4.set_ylabel('y / r_Hill')
    ax4.set_title('Escape Near Hill Sphere Boundary\n(relative to planet)')
    ax4.set_aspect('equal')
    ax4.legend(fontsize=8)
    ax4.set_xlim(-3, 3)
    ax4.set_ylim(-3, 3)
    ax4.grid(True, alpha=0.3)

    # Plot 5: Distance from planet over time
    ax5 = fig.add_subplot(2, 3, 5)

    distances = [0.5 * r_hill_demo, 0.9 * r_hill_demo, 1.1 * r_hill_demo]
    colors = ['green', 'orange', 'red']
    labels = ['Inside (stable)', 'Near boundary', 'Outside (escapes)']

    for dist, color, label in zip(distances, colors, labels):
        v_orb = np.sqrt(G * m_planet_demo / dist)

        result = simulate_test_particle_three_body(
            m_primary=M_star,
            m_secondary=m_planet_demo,
            a_orbit=a_demo,
            r_particle=[dist, 0, 0],
            v_particle=[0, v_orb, 0],
            t_final=20 * T_orbit,
            dt=dt,
            G=G
        )

        rel_pos = result['rel_positions']
        distances_from_planet = np.linalg.norm(rel_pos, axis=1)
        times = result['times'] / T_orbit

        ax5.plot(times, distances_from_planet / r_hill_demo,
                 color=color, lw=1.5, label=label)

    ax5.axhline(y=1.0, color='red', linestyle='--', lw=2,
                label='Hill sphere radius')
    ax5.set_xlabel('Time / Orbital Period')
    ax5.set_ylabel('Distance from Planet / r_Hill')
    ax5.set_title('Particle Distance from Planet Over Time')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, 5)

    # Plot 6: Hill sphere as fraction of orbit
    ax6 = fig.add_subplot(2, 3, 6)

    mass_ratios = np.logspace(-7, -2, 100)
    r_hill_fractions = (mass_ratios / 3)**(1/3)

    ax6.loglog(mass_ratios, r_hill_fractions, 'b-', lw=2)

    # Mark solar system planets
    for name, data in solar_system.items():
        m_ratio = data['m_ratio']
        r_h_frac = (m_ratio / 3)**(1/3)
        ax6.plot(m_ratio, r_h_frac, 'ro', markersize=8)
        ax6.annotate(name, (m_ratio, r_h_frac),
                     xytext=(5, 5), textcoords='offset points', fontsize=7)

    ax6.set_xlabel('Mass Ratio (m/M)')
    ax6.set_ylabel('Hill Sphere / Orbital Radius')
    ax6.set_title('Hill Sphere Size vs Mass Ratio\nr_H/a = (m/3M)^(1/3)')
    ax6.grid(True, alpha=0.3, which='both')

    plt.suptitle('Hill Sphere: Gravitational Sphere of Influence', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'hill_sphere.png'),
                dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {os.path.join(output_dir, 'hill_sphere.png')}")

    # Print solar system Hill sphere data
    print("\nSolar System Hill Sphere Data:")
    print("-" * 50)
    print(f"{'Planet':<10} {'a (AU)':<10} {'r_H (AU)':<12} {'r_H/a':<10}")
    print("-" * 50)
    for name, r_h, a in zip(names, hill_radii, orbital_radii):
        print(f"{name:<10} {a:<10.3f} {r_h:<12.6f} {r_h/a:<10.6f}")


if __name__ == "__main__":
    main()
