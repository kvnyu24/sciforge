"""
Example demonstrating coupled spring-mass chain dynamics.

This example shows wave propagation through a chain of masses
connected by springs, demonstrating normal modes and dispersion.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.mechanics import Particle


def simulate_spring_chain(n_masses, k, m, initial_displacement, t_final, dt):
    """
    Simulate a chain of masses connected by springs.

    Args:
        n_masses: Number of masses in chain
        k: Spring constant (N/m)
        m: Mass of each particle (kg)
        initial_displacement: Function returning initial displacement for each mass
        t_final: Simulation duration (s)
        dt: Time step (s)

    Returns:
        Dictionary with time and position data
    """
    # Create particles (fixed boundaries at x=0 and x=n_masses+1)
    equilibrium_spacing = 1.0
    particles = []

    for i in range(n_masses):
        x_eq = (i + 1) * equilibrium_spacing
        x0 = x_eq + initial_displacement(i, n_masses)
        particles.append(Particle(
            mass=m,
            position=np.array([x0, 0.0, 0.0]),
            velocity=np.array([0.0, 0.0, 0.0]),
            gravity=0.0
        ))

    # Store results
    times = [0]
    positions = [[p.position[0] - (i+1)*equilibrium_spacing for i, p in enumerate(particles)]]

    t = 0
    while t < t_final:
        # Calculate spring forces
        forces = []
        for i, p in enumerate(particles):
            # Spring force from left
            if i == 0:
                # Fixed wall at x=0
                x_left = 0.0
            else:
                x_left = particles[i-1].position[0]

            # Spring force from right
            if i == n_masses - 1:
                # Fixed wall at x = (n_masses+1) * spacing
                x_right = (n_masses + 1) * equilibrium_spacing
            else:
                x_right = particles[i+1].position[0]

            x_eq = (i + 1) * equilibrium_spacing
            x = p.position[0]

            # Spring forces
            F_left = k * ((x_left - x) + equilibrium_spacing)
            F_right = k * ((x_right - x) - equilibrium_spacing)
            F_total = F_left + F_right

            forces.append(np.array([F_total, 0.0, 0.0]))

        # Update all particles
        for p, F in zip(particles, forces):
            p.update(F, dt)

        t += dt
        times.append(t)
        positions.append([p.position[0] - (i+1)*equilibrium_spacing for i, p in enumerate(particles)])

    return {
        'time': np.array(times),
        'positions': np.array(positions)
    }


def main():
    # Parameters
    n_masses = 20
    k = 100.0  # Spring constant
    m = 1.0    # Mass
    t_final = 10.0
    dt = 0.001

    # Natural frequency of single oscillator
    omega_0 = np.sqrt(k / m)

    # Different initial conditions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Case 1: Single mass displaced (wave pulse)
    def pulse_init(i, n):
        if i == 0:
            return 0.5
        return 0.0

    results1 = simulate_spring_chain(n_masses, k, m, pulse_init, t_final, dt)

    ax1 = axes[0, 0]
    # Create space-time plot
    im1 = ax1.imshow(results1['positions'].T, aspect='auto',
                     extent=[0, t_final, n_masses, 0],
                     cmap='RdBu', vmin=-0.5, vmax=0.5)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Mass index')
    ax1.set_title('Wave Pulse Propagation')
    plt.colorbar(im1, ax=ax1, label='Displacement')

    # Case 2: Sinusoidal initial condition (standing wave - first mode)
    def mode1_init(i, n):
        return 0.3 * np.sin(np.pi * (i + 1) / (n + 1))

    results2 = simulate_spring_chain(n_masses, k, m, mode1_init, t_final, dt)

    ax2 = axes[0, 1]
    im2 = ax2.imshow(results2['positions'].T, aspect='auto',
                     extent=[0, t_final, n_masses, 0],
                     cmap='RdBu', vmin=-0.5, vmax=0.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Mass index')
    ax2.set_title('First Normal Mode')
    plt.colorbar(im2, ax=ax2, label='Displacement')

    # Case 3: Second normal mode
    def mode2_init(i, n):
        return 0.3 * np.sin(2 * np.pi * (i + 1) / (n + 1))

    results3 = simulate_spring_chain(n_masses, k, m, mode2_init, t_final, dt)

    ax3 = axes[1, 0]
    im3 = ax3.imshow(results3['positions'].T, aspect='auto',
                     extent=[0, t_final, n_masses, 0],
                     cmap='RdBu', vmin=-0.5, vmax=0.5)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Mass index')
    ax3.set_title('Second Normal Mode')
    plt.colorbar(im3, ax=ax3, label='Displacement')

    # Case 4: Snapshot comparison at different times
    ax4 = axes[1, 1]
    times_to_plot = [0, 1.0, 2.0, 3.0, 4.0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(times_to_plot)))

    for t_plot, color in zip(times_to_plot, colors):
        idx = int(t_plot / dt)
        if idx < len(results1['positions']):
            ax4.plot(range(n_masses), results1['positions'][idx], 'o-',
                    color=color, label=f't = {t_plot:.1f} s', alpha=0.7, markersize=4)

    ax4.set_xlabel('Mass index')
    ax4.set_ylabel('Displacement')
    ax4.set_title('Wave Pulse at Different Times')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f'Coupled Spring-Mass Chain (N={n_masses}, k={k} N/m, m={m} kg)',
                 fontsize=12, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'spring_chain.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'spring_chain.png')}")


if __name__ == "__main__":
    main()
