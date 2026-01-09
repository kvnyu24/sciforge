"""
Example demonstrating the gravitational two-body problem.

This example shows orbital mechanics with two particles interacting
gravitationally, demonstrating Kepler's laws and orbital mechanics.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.mechanics import Particle


def gravitational_force(p1, p2, G=1.0):
    """Calculate gravitational force on p1 due to p2."""
    r = p2.position - p1.position
    r_mag = np.linalg.norm(r)
    if r_mag < 0.1:  # Softening to avoid singularity
        r_mag = 0.1
    return G * p1.mass * p2.mass * r / r_mag**3


def simulate_two_body(m1, m2, pos1, pos2, vel1, vel2, t_final, dt, G=1.0):
    """
    Simulate two-body gravitational interaction.

    Returns:
        Dictionary with trajectory and energy data
    """
    # Create particles without built-in gravity
    p1 = Particle(mass=m1, position=pos1.copy(), velocity=vel1.copy(), gravity=0.0)
    p2 = Particle(mass=m2, position=pos2.copy(), velocity=vel2.copy(), gravity=0.0)

    times = [0]
    x1, y1 = [pos1[0]], [pos1[1]]
    x2, y2 = [pos2[0]], [pos2[1]]

    # Center of mass position
    cm_x = [(m1*pos1[0] + m2*pos2[0])/(m1+m2)]
    cm_y = [(m1*pos1[1] + m2*pos2[1])/(m1+m2)]

    # Energy
    def kinetic_energy():
        return 0.5 * m1 * np.dot(p1.velocity, p1.velocity) + \
               0.5 * m2 * np.dot(p2.velocity, p2.velocity)

    def potential_energy():
        r = np.linalg.norm(p2.position - p1.position)
        r = max(r, 0.1)
        return -G * m1 * m2 / r

    energies = [kinetic_energy() + potential_energy()]

    t = 0
    while t < t_final:
        # Calculate gravitational forces
        F12 = gravitational_force(p1, p2, G)
        F21 = -F12

        # Update positions using the forces
        p1.update(F12, dt)
        p2.update(F21, dt)
        t += dt

        times.append(t)
        x1.append(p1.position[0])
        y1.append(p1.position[1])
        x2.append(p2.position[0])
        y2.append(p2.position[1])
        cm_x.append((m1*p1.position[0] + m2*p2.position[0])/(m1+m2))
        cm_y.append((m1*p1.position[1] + m2*p2.position[1])/(m1+m2))
        energies.append(kinetic_energy() + potential_energy())

    return {
        'time': np.array(times),
        'x1': np.array(x1), 'y1': np.array(y1),
        'x2': np.array(x2), 'y2': np.array(y2),
        'cm_x': np.array(cm_x), 'cm_y': np.array(cm_y),
        'energy': np.array(energies)
    }


def main():
    # Equal mass binary system (circular orbit)
    m1, m2 = 1.0, 1.0
    G = 1.0

    # Initial positions (symmetric about origin)
    pos1 = np.array([-1.0, 0.0, 0.0])
    pos2 = np.array([1.0, 0.0, 0.0])

    # Circular orbit velocity
    r = np.linalg.norm(pos2 - pos1)
    v_circ = np.sqrt(G * m2 / r)  # Circular orbit speed

    # Different eccentricities
    cases = {
        'Circular (e≈0)': 1.0,
        'Elliptical (e≈0.3)': 0.85,
        'Elliptical (e≈0.6)': 0.65
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for idx, (name, v_factor) in enumerate(cases.items()):
        vel1 = np.array([0.0, v_factor * v_circ, 0.0])
        vel2 = np.array([0.0, -v_factor * v_circ, 0.0])

        results = simulate_two_body(m1, m2, pos1, pos2, vel1, vel2,
                                   t_final=20.0, dt=0.01, G=G)

        # Plot orbits
        ax1 = axes[0, idx]
        ax1.plot(results['x1'], results['y1'], 'b-', lw=1, alpha=0.7, label='Body 1')
        ax1.plot(results['x2'], results['y2'], 'r-', lw=1, alpha=0.7, label='Body 2')
        ax1.plot(results['cm_x'], results['cm_y'], 'k--', lw=1, alpha=0.5, label='Center of mass')
        ax1.plot(results['x1'][0], results['y1'][0], 'bo', markersize=10)
        ax1.plot(results['x2'][0], results['y2'][0], 'ro', markersize=10)
        ax1.plot(results['cm_x'][0], results['cm_y'][0], 'k+', markersize=15, mew=2)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title(name)
        ax1.set_aspect('equal')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Plot energy conservation
        ax2 = axes[1, idx]
        E_normalized = results['energy'] / results['energy'][0]
        ax2.plot(results['time'], E_normalized, 'g-', lw=1.5)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('E/E₀')
        ax2.set_title('Energy Conservation')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0.9, 1.1)
        ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)

    plt.suptitle('Gravitational Two-Body Problem (Equal Masses)', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'two_body_problem.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'two_body_problem.png')}")


if __name__ == "__main__":
    main()
