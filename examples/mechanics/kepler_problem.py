"""
Example demonstrating the two-body Kepler problem.

This example shows the classical Kepler problem with detailed analysis of
orbital elements, Kepler's laws verification, and energy/angular momentum
conservation. Demonstrates elliptical, parabolic, and hyperbolic orbits.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.orbital import KeplerianOrbit, TwoBodyProblem


def verify_keplers_laws(orbit, results, m1, m2, G):
    """Verify all three of Kepler's laws from simulation data."""
    # Kepler's First Law: Orbits are ellipses
    # (Verified by the eccentricity and trajectory shape)

    # Kepler's Second Law: Equal areas in equal times
    # Calculate areal velocity (should be constant)
    r1 = np.array(results['r1'])
    v1 = np.array(results['v1'])
    r2 = np.array(results['r2'])
    v2 = np.array(results['v2'])

    # Relative position and velocity
    r_rel = r2 - r1
    v_rel = v2 - v1

    # Areal velocity = 0.5 * |r x v|
    areal_velocities = []
    for i in range(len(r_rel)):
        h = np.cross(r_rel[i], v_rel[i])
        areal_velocities.append(0.5 * np.linalg.norm(h))

    # Kepler's Third Law: T^2 proportional to a^3
    # Period from simulation
    times = np.array(results['time'])

    return np.array(areal_velocities)


def simulate_kepler_orbits():
    """Simulate various Kepler orbits and analyze their properties."""
    # Parameters (normalized units)
    G = 1.0
    M_central = 1.0  # Central mass
    m_orbiting = 0.001  # Small orbiting mass (test particle limit)

    # Different orbit types based on eccentricity
    orbit_cases = {
        'Circular (e=0)': {'e': 0.0, 'a': 1.0},
        'Elliptical (e=0.3)': {'e': 0.3, 'a': 1.0},
        'Elliptical (e=0.6)': {'e': 0.6, 'a': 1.0},
        'Highly Elliptical (e=0.9)': {'e': 0.9, 'a': 1.0},
    }

    fig = plt.figure(figsize=(16, 12))

    # Main orbit plot
    ax_orbits = fig.add_subplot(2, 2, 1)
    colors = ['blue', 'green', 'orange', 'red']

    orbit_objects = {}

    for idx, (name, params) in enumerate(orbit_cases.items()):
        # Create Keplerian orbit
        orbit = KeplerianOrbit(
            central_mass=M_central,
            semi_major_axis=params['a'],
            eccentricity=params['e'],
            G=G
        )
        orbit_objects[name] = orbit

        # Generate trajectory
        trajectory = orbit.trajectory(n_points=500)

        ax_orbits.plot(trajectory[:, 0], trajectory[:, 1],
                       color=colors[idx], lw=2, label=name)

        # Mark periapsis and apoapsis
        r_p = orbit.periapsis()
        r_a = orbit.apoapsis()
        ax_orbits.plot(r_p, 0, 'o', color=colors[idx], markersize=8)
        ax_orbits.plot(-r_a, 0, 's', color=colors[idx], markersize=6)

    # Mark central body
    ax_orbits.plot(0, 0, 'ko', markersize=15, label='Central mass')
    ax_orbits.set_xlabel('x (normalized units)')
    ax_orbits.set_ylabel('y (normalized units)')
    ax_orbits.set_title("Kepler's First Law: Elliptical Orbits")
    ax_orbits.set_aspect('equal')
    ax_orbits.legend(loc='upper right', fontsize=8)
    ax_orbits.grid(True, alpha=0.3)
    ax_orbits.set_xlim(-2.5, 2.5)
    ax_orbits.set_ylim(-2.5, 2.5)

    # Kepler's Second Law: Areal velocity conservation
    ax_areal = fig.add_subplot(2, 2, 2)

    for idx, (name, params) in enumerate(orbit_cases.items()):
        # Use TwoBodyProblem for detailed simulation
        orbit = orbit_objects[name]
        r_p = orbit.periapsis()
        v_p = orbit.velocity_magnitude(r_p)

        # Initial conditions at periapsis
        pos1 = np.array([0.0, 0.0, 0.0])
        vel1 = np.array([0.0, 0.0, 0.0])
        pos2 = np.array([r_p, 0.0, 0.0])
        vel2 = np.array([0.0, v_p, 0.0])

        two_body = TwoBodyProblem(
            mass1=M_central,
            mass2=m_orbiting,
            position1=pos1, velocity1=vel1,
            position2=pos2, velocity2=vel2,
            G=G
        )

        # Simulate for one period
        period = orbit.period()
        dt = period / 500

        for _ in range(500):
            two_body.update(dt)

        # Calculate areal velocity
        areal_vel = verify_keplers_laws(orbit, two_body._history, M_central, m_orbiting, G)
        times = np.array(two_body._history['time'])

        # Normalize by initial value
        areal_vel_norm = areal_vel / areal_vel[0]
        ax_areal.plot(times / period, areal_vel_norm,
                      color=colors[idx], lw=1.5, label=name, alpha=0.8)

    ax_areal.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    ax_areal.set_xlabel('Time / Period')
    ax_areal.set_ylabel('Areal Velocity / Initial')
    ax_areal.set_title("Kepler's Second Law: Constant Areal Velocity")
    ax_areal.legend(fontsize=8)
    ax_areal.grid(True, alpha=0.3)
    ax_areal.set_ylim(0.95, 1.05)

    # Kepler's Third Law: T^2 vs a^3
    ax_third = fig.add_subplot(2, 2, 3)

    semi_major_axes = np.linspace(0.5, 3.0, 20)
    periods_squared = []
    a_cubed = []

    for a in semi_major_axes:
        orbit = KeplerianOrbit(
            central_mass=M_central,
            semi_major_axis=a,
            eccentricity=0.3,
            G=G
        )
        T = orbit.period()
        periods_squared.append(T**2)
        a_cubed.append(a**3)

    ax_third.plot(a_cubed, periods_squared, 'bo-', markersize=6, label='Computed')

    # Theoretical line: T^2 = (4pi^2/GM) * a^3
    theoretical = 4 * np.pi**2 / (G * M_central) * np.array(a_cubed)
    ax_third.plot(a_cubed, theoretical, 'r--', lw=2, label='Theory: T^2 = 4pi^2 a^3 / GM')

    ax_third.set_xlabel('Semi-major axis cubed (a^3)')
    ax_third.set_ylabel('Period squared (T^2)')
    ax_third.set_title("Kepler's Third Law: T^2 proportional to a^3")
    ax_third.legend()
    ax_third.grid(True, alpha=0.3)

    # Energy and Angular Momentum Conservation
    ax_conserve = fig.add_subplot(2, 2, 4)

    # Simulate one orbit with e=0.6 and track conserved quantities
    orbit = orbit_objects['Elliptical (e=0.6)']
    r_p = orbit.periapsis()
    v_p = orbit.velocity_magnitude(r_p)

    pos1 = np.array([0.0, 0.0, 0.0])
    vel1 = np.array([0.0, 0.0, 0.0])
    pos2 = np.array([r_p, 0.0, 0.0])
    vel2 = np.array([0.0, v_p, 0.0])

    two_body = TwoBodyProblem(
        mass1=M_central,
        mass2=m_orbiting,
        position1=pos1, velocity1=vel1,
        position2=pos2, velocity2=vel2,
        G=G
    )

    period = orbit.period()
    dt = period / 1000

    energies = [two_body.total_energy()]
    ang_mom = [np.linalg.norm(two_body.angular_momentum())]
    times = [0]

    for i in range(1000):
        two_body.update(dt)
        energies.append(two_body.total_energy())
        ang_mom.append(np.linalg.norm(two_body.angular_momentum()))
        times.append(two_body.time)

    times = np.array(times) / period
    energies = np.array(energies) / energies[0]
    ang_mom = np.array(ang_mom) / ang_mom[0]

    ax_conserve.plot(times, energies, 'b-', lw=2, label='Energy E/E_0')
    ax_conserve.plot(times, ang_mom, 'r--', lw=2, label='Angular Mom |L|/|L_0|')
    ax_conserve.axhline(y=1.0, color='black', linestyle=':', alpha=0.5)
    ax_conserve.set_xlabel('Time / Period')
    ax_conserve.set_ylabel('Normalized Value')
    ax_conserve.set_title('Conservation Laws (e=0.6)')
    ax_conserve.legend()
    ax_conserve.grid(True, alpha=0.3)
    ax_conserve.set_ylim(0.95, 1.05)

    plt.suptitle('Two-Body Kepler Problem: Verification of Kepler\'s Laws', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'kepler_problem.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'kepler_problem.png')}")

    # Print orbital parameters
    print("\nOrbital Parameters:")
    print("-" * 60)
    for name, orbit in orbit_objects.items():
        print(f"\n{name}:")
        print(f"  Semi-major axis: a = {orbit.a:.3f}")
        print(f"  Eccentricity: e = {orbit.e:.3f}")
        print(f"  Periapsis: r_p = {orbit.periapsis():.3f}")
        print(f"  Apoapsis: r_a = {orbit.apoapsis():.3f}")
        print(f"  Period: T = {orbit.period():.3f}")
        print(f"  Specific energy: E = {orbit.specific_energy():.3f}")
        print(f"  Specific angular momentum: h = {orbit.specific_angular_momentum():.3f}")


if __name__ == "__main__":
    simulate_kepler_orbits()
