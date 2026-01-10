"""
Example demonstrating N-body gravitational simulation.

This example simulates multiple gravitating bodies interacting with each
other, demonstrating:
- Direct N-body simulation with O(N^2) force calculation
- Energy and momentum conservation
- Various configurations (solar system, binary stars, cluster dynamics)
- Visualization of trajectories and conserved quantities
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class NBodySystem:
    """
    N-body gravitational system with direct force calculation.

    Uses RK4 integration for accurate orbit calculation.
    """

    def __init__(self, masses, positions, velocities, G=1.0, softening=0.01):
        """
        Initialize N-body system.

        Args:
            masses: Array of masses (N,)
            positions: Array of position vectors (N, 3)
            velocities: Array of velocity vectors (N, 3)
            G: Gravitational constant
            softening: Softening length to avoid singularities
        """
        self.masses = np.array(masses)
        self.positions = np.array(positions, dtype=float)
        self.velocities = np.array(velocities, dtype=float)
        self.G = G
        self.softening = softening
        self.N = len(masses)
        self.time = 0.0

        self._history = {
            'time': [0.0],
            'positions': [self.positions.copy()],
            'velocities': [self.velocities.copy()]
        }

    def acceleration(self, positions):
        """Calculate gravitational acceleration for all bodies."""
        acc = np.zeros_like(positions)

        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    r_vec = positions[j] - positions[i]
                    r_mag = np.sqrt(np.sum(r_vec**2) + self.softening**2)
                    acc[i] += self.G * self.masses[j] * r_vec / r_mag**3

        return acc

    def update(self, dt):
        """Update system using RK4 integration."""
        # RK4 for coupled position-velocity system
        def derivs(pos, vel):
            return vel, self.acceleration(pos)

        k1_v, k1_a = derivs(self.positions, self.velocities)

        k2_v, k2_a = derivs(
            self.positions + 0.5 * dt * k1_v,
            self.velocities + 0.5 * dt * k1_a
        )

        k3_v, k3_a = derivs(
            self.positions + 0.5 * dt * k2_v,
            self.velocities + 0.5 * dt * k2_a
        )

        k4_v, k4_a = derivs(
            self.positions + dt * k3_v,
            self.velocities + dt * k3_a
        )

        self.positions += (dt / 6) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        self.velocities += (dt / 6) * (k1_a + 2*k2_a + 2*k3_a + k4_a)

        self.time += dt

        self._history['time'].append(self.time)
        self._history['positions'].append(self.positions.copy())
        self._history['velocities'].append(self.velocities.copy())

    def total_energy(self):
        """Calculate total energy (kinetic + potential)."""
        # Kinetic energy
        KE = 0.5 * np.sum(self.masses[:, np.newaxis] * self.velocities**2)

        # Potential energy
        PE = 0.0
        for i in range(self.N):
            for j in range(i + 1, self.N):
                r_vec = self.positions[j] - self.positions[i]
                r_mag = np.sqrt(np.sum(r_vec**2) + self.softening**2)
                PE -= self.G * self.masses[i] * self.masses[j] / r_mag

        return KE + PE

    def total_momentum(self):
        """Calculate total linear momentum."""
        return np.sum(self.masses[:, np.newaxis] * self.velocities, axis=0)

    def total_angular_momentum(self):
        """Calculate total angular momentum."""
        L = np.zeros(3)
        for i in range(self.N):
            L += self.masses[i] * np.cross(self.positions[i], self.velocities[i])
        return L

    def center_of_mass(self):
        """Calculate center of mass position."""
        return np.sum(self.masses[:, np.newaxis] * self.positions, axis=0) / np.sum(self.masses)


def create_solar_system_like():
    """Create a simplified solar system configuration."""
    G = 1.0
    M_star = 1.0

    # Planet data: (mass_ratio, semi_major_axis, eccentricity)
    planets = [
        (0.001, 0.4, 0.1),   # Inner planet
        (0.003, 0.7, 0.05),  # Venus-like
        (0.003, 1.0, 0.02),  # Earth-like
        (0.001, 1.5, 0.1),   # Mars-like
    ]

    masses = [M_star]
    positions = [[0.0, 0.0, 0.0]]
    velocities = [[0.0, 0.0, 0.0]]

    for m_ratio, a, e in planets:
        m = m_ratio * M_star
        r_p = a * (1 - e)  # Start at perihelion
        v_p = np.sqrt(G * M_star * (1 + e) / (a * (1 - e)))

        masses.append(m)
        positions.append([r_p, 0.0, 0.0])
        velocities.append([0.0, v_p, 0.0])

    # Adjust for center of mass motion
    masses = np.array(masses)
    positions = np.array(positions)
    velocities = np.array(velocities)

    com = np.sum(masses[:, np.newaxis] * positions, axis=0) / np.sum(masses)
    com_vel = np.sum(masses[:, np.newaxis] * velocities, axis=0) / np.sum(masses)

    positions -= com
    velocities -= com_vel

    return NBodySystem(masses, positions, velocities, G=G)


def create_binary_star_with_planet():
    """Create a binary star system with a circumbinary planet."""
    G = 1.0

    # Binary stars
    m1, m2 = 1.0, 0.8
    a_binary = 0.3
    mu = m2 / (m1 + m2)

    # Circular binary orbit
    v_binary = np.sqrt(G * (m1 + m2) / a_binary)

    masses = [m1, m2]
    positions = [
        [-mu * a_binary, 0.0, 0.0],
        [(1 - mu) * a_binary, 0.0, 0.0]
    ]
    velocities = [
        [0.0, -mu * v_binary, 0.0],
        [0.0, (1 - mu) * v_binary, 0.0]
    ]

    # Circumbinary planet
    m_planet = 0.001
    a_planet = 1.5  # Must be > 2-3 times binary separation for stability
    v_planet = np.sqrt(G * (m1 + m2) / a_planet)

    masses.append(m_planet)
    positions.append([a_planet, 0.0, 0.0])
    velocities.append([0.0, v_planet, 0.0])

    return NBodySystem(masses, positions, velocities, G=G)


def create_figure_eight():
    """Create the famous figure-8 three-body orbit."""
    # This is a special periodic solution found by Chenciner and Montgomery
    # Using normalized values from the literature
    G = 1.0
    m = 1.0

    # Initial conditions for figure-8 orbit
    x1 = 0.97000436
    y1 = -0.24308753
    vx3 = -0.93240737
    vy3 = -0.86473146

    masses = [m, m, m]
    positions = [
        [x1, y1, 0.0],
        [-x1, -y1, 0.0],
        [0.0, 0.0, 0.0]
    ]
    velocities = [
        [vx3/2, vy3/2, 0.0],
        [vx3/2, vy3/2, 0.0],
        [-vx3, -vy3, 0.0]
    ]

    return NBodySystem(masses, positions, velocities, G=G, softening=0.001)


def main():
    fig = plt.figure(figsize=(18, 12))

    # System 1: Solar system like
    ax1 = fig.add_subplot(2, 3, 1)
    system = create_solar_system_like()

    # Estimate orbital period of outermost planet
    T_outer = 2 * np.pi * np.sqrt(1.5**3 / 1.0)
    dt = T_outer / 500

    for _ in range(2000):
        system.update(dt)

    # Plot trajectories
    history = system._history
    positions = np.array(history['positions'])

    colors = ['gold', 'gray', 'orange', 'blue', 'red']
    labels = ['Star', 'Planet 1', 'Planet 2', 'Planet 3', 'Planet 4']
    sizes = [15, 5, 6, 6, 4]

    for i in range(system.N):
        ax1.plot(positions[:, i, 0], positions[:, i, 1],
                 color=colors[i], lw=0.5, alpha=0.7)
        ax1.plot(positions[-1, i, 0], positions[-1, i, 1],
                 'o', color=colors[i], markersize=sizes[i], label=labels[i])

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Solar System Configuration\n(4 planets)')
    ax1.set_aspect('equal')
    ax1.legend(fontsize=7, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # System 2: Binary with planet
    ax2 = fig.add_subplot(2, 3, 2)
    system = create_binary_star_with_planet()

    T_binary = 2 * np.pi * np.sqrt(0.3**3 / 1.8)
    dt = T_binary / 100

    for _ in range(3000):
        system.update(dt)

    history = system._history
    positions = np.array(history['positions'])

    ax2.plot(positions[:, 0, 0], positions[:, 0, 1], 'y-', lw=0.5, alpha=0.7)
    ax2.plot(positions[:, 1, 0], positions[:, 1, 1], 'orange', lw=0.5, alpha=0.7)
    ax2.plot(positions[:, 2, 0], positions[:, 2, 1], 'b-', lw=0.5, alpha=0.7)

    ax2.plot(positions[-1, 0, 0], positions[-1, 0, 1], 'yo', markersize=12, label='Star 1')
    ax2.plot(positions[-1, 1, 0], positions[-1, 1, 1], 'o', color='orange', markersize=10, label='Star 2')
    ax2.plot(positions[-1, 2, 0], positions[-1, 2, 1], 'bo', markersize=5, label='Planet')

    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Binary Star with Circumbinary Planet')
    ax2.set_aspect('equal')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # System 3: Figure-8 orbit
    ax3 = fig.add_subplot(2, 3, 3)
    system = create_figure_eight()

    # Period of figure-8 orbit is approximately 6.3
    T_fig8 = 6.3
    dt = T_fig8 / 500

    for _ in range(1000):
        system.update(dt)

    history = system._history
    positions = np.array(history['positions'])

    colors = ['red', 'green', 'blue']
    for i in range(3):
        ax3.plot(positions[:, i, 0], positions[:, i, 1],
                 color=colors[i], lw=1, alpha=0.7)
        ax3.plot(positions[-1, i, 0], positions[-1, i, 1],
                 'o', color=colors[i], markersize=10, label=f'Body {i+1}')

    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('Figure-8 Three-Body Orbit\n(Periodic solution)')
    ax3.set_aspect('equal')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Energy conservation for solar system
    ax4 = fig.add_subplot(2, 3, 4)
    system = create_solar_system_like()

    T_outer = 2 * np.pi * np.sqrt(1.5**3 / 1.0)
    dt = T_outer / 200

    times = [0]
    energies = [system.total_energy()]

    for _ in range(1000):
        system.update(dt)
        times.append(system.time)
        energies.append(system.total_energy())

    times = np.array(times) / T_outer
    energies = np.array(energies) / energies[0]

    ax4.plot(times, energies, 'b-', lw=1)
    ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
    ax4.set_xlabel('Time / Outer Period')
    ax4.set_ylabel('Energy / Initial Energy')
    ax4.set_title('Energy Conservation (Solar System)')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0.999, 1.001)

    # Momentum conservation
    ax5 = fig.add_subplot(2, 3, 5)
    system = create_binary_star_with_planet()

    T_binary = 2 * np.pi * np.sqrt(0.3**3 / 1.8)
    dt = T_binary / 100

    times = [0]
    momenta = [np.linalg.norm(system.total_momentum())]
    ang_momenta = [np.linalg.norm(system.total_angular_momentum())]

    for _ in range(1000):
        system.update(dt)
        times.append(system.time)
        momenta.append(np.linalg.norm(system.total_momentum()))
        ang_momenta.append(np.linalg.norm(system.total_angular_momentum()))

    times = np.array(times) / T_binary

    # Normalize (handle zero initial momentum)
    if momenta[0] > 1e-10:
        momenta = np.array(momenta) / momenta[0]
    else:
        momenta = np.array(momenta)

    ang_momenta = np.array(ang_momenta) / ang_momenta[0]

    ax5.plot(times, ang_momenta, 'r-', lw=1.5, label='|L| / |L_0|')
    ax5.axhline(y=1.0, color='black', linestyle='--', alpha=0.7)
    ax5.set_xlabel('Time / Binary Period')
    ax5.set_ylabel('Normalized Value')
    ax5.set_title('Angular Momentum Conservation\n(Binary + Planet)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0.999, 1.001)

    # Hierarchical triple system
    ax6 = fig.add_subplot(2, 3, 6)

    # Inner binary
    m1, m2 = 1.0, 0.5
    a_inner = 0.2
    v_inner = np.sqrt((m1 + m2) / a_inner)
    mu_inner = m2 / (m1 + m2)

    # Outer companion
    m3 = 0.3
    a_outer = 2.0
    v_outer = np.sqrt((m1 + m2 + m3) / a_outer)
    mu_outer = m3 / (m1 + m2 + m3)

    masses = [m1, m2, m3]
    positions = [
        [-mu_inner * a_inner * (1 - mu_outer), 0.0, 0.0],
        [(1 - mu_inner) * a_inner * (1 - mu_outer), 0.0, 0.0],
        [a_outer * (1 - mu_outer), 0.0, 0.0]
    ]
    velocities = [
        [0.0, -mu_inner * v_inner, 0.0],
        [0.0, (1 - mu_inner) * v_inner, 0.0],
        [0.0, v_outer, 0.0]
    ]

    # Adjust for center of mass
    masses_arr = np.array(masses)
    pos_arr = np.array(positions)
    vel_arr = np.array(velocities)

    com = np.sum(masses_arr[:, np.newaxis] * pos_arr, axis=0) / np.sum(masses_arr)
    com_vel = np.sum(masses_arr[:, np.newaxis] * vel_arr, axis=0) / np.sum(masses_arr)

    pos_arr -= com
    vel_arr -= com_vel

    system = NBodySystem(masses_arr, pos_arr, vel_arr, G=1.0)

    T_outer_orb = 2 * np.pi * np.sqrt(a_outer**3 / (m1 + m2 + m3))
    dt = T_outer_orb / 500

    for _ in range(2000):
        system.update(dt)

    history = system._history
    positions = np.array(history['positions'])

    ax6.plot(positions[:, 0, 0], positions[:, 0, 1], 'y-', lw=0.5, alpha=0.7)
    ax6.plot(positions[:, 1, 0], positions[:, 1, 1], 'orange', lw=0.5, alpha=0.7)
    ax6.plot(positions[:, 2, 0], positions[:, 2, 1], 'b-', lw=0.5, alpha=0.7)

    ax6.plot(positions[-1, 0, 0], positions[-1, 0, 1], 'yo', markersize=10, label='Star 1')
    ax6.plot(positions[-1, 1, 0], positions[-1, 1, 1], 'o', color='orange', markersize=8, label='Star 2')
    ax6.plot(positions[-1, 2, 0], positions[-1, 2, 1], 'bo', markersize=6, label='Outer companion')

    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    ax6.set_title('Hierarchical Triple System\n(Inner binary + outer companion)')
    ax6.set_aspect('equal')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    plt.suptitle('N-Body Gravitational Dynamics', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'n_body_gravity.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'n_body_gravity.png')}")


if __name__ == "__main__":
    main()
