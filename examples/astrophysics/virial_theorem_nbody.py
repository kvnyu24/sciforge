"""
Experiment 266: Virial Theorem N-body

Demonstrates the virial theorem for gravitationally bound systems:
2<K> + <U> = 0 in equilibrium

Physical concepts:
- Time-averaged kinetic energy equals -1/2 potential energy
- Used to measure masses of galaxies and clusters
- Virialization is endpoint of gravitational collapse
- Virial temperature from gravitational energy release
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt

# Physical constants
G = 6.674e-11
M_sun = 1.989e30
pc = 3.086e16  # parsec in meters
kpc = 1e3 * pc
k_B = 1.381e-23
m_p = 1.673e-27


def initialize_nbody(N, R, M_total):
    """
    Initialize N-body system with Plummer model distribution.

    Args:
        N: Number of particles
        R: Characteristic radius
        M_total: Total mass

    Returns:
        positions, velocities, masses arrays
    """
    masses = np.ones(N) * M_total / N

    # Plummer model: rho(r) ~ (1 + r^2/a^2)^(-5/2)
    # Cumulative mass: M(r) = M * r^3 / (r^2 + a^2)^(3/2)

    # Sample radii from cumulative distribution
    u = np.random.uniform(0, 1, N)
    r = R / np.sqrt(u**(-2/3) - 1)

    # Limit maximum radius
    r = np.minimum(r, 10 * R)

    # Random directions
    phi = np.random.uniform(0, 2 * np.pi, N)
    cos_theta = np.random.uniform(-1, 1, N)
    sin_theta = np.sqrt(1 - cos_theta**2)

    positions = np.zeros((N, 3))
    positions[:, 0] = r * sin_theta * np.cos(phi)
    positions[:, 1] = r * sin_theta * np.sin(phi)
    positions[:, 2] = r * cos_theta

    # Velocities from virial equilibrium
    # v_rms ~ sqrt(G*M/R)
    v_rms = np.sqrt(G * M_total / (3 * R))

    velocities = np.random.normal(0, v_rms / np.sqrt(3), (N, 3))

    return positions, velocities, masses


def compute_kinetic_energy(velocities, masses):
    """Compute total kinetic energy."""
    v_sq = np.sum(velocities**2, axis=1)
    return 0.5 * np.sum(masses * v_sq)


def compute_potential_energy(positions, masses, softening=1e-3):
    """Compute total gravitational potential energy."""
    N = len(masses)
    U = 0.0

    for i in range(N):
        for j in range(i + 1, N):
            r = np.linalg.norm(positions[i] - positions[j])
            U -= G * masses[i] * masses[j] / np.sqrt(r**2 + softening**2)

    return U


def leapfrog_step(positions, velocities, masses, dt, softening=1e-3):
    """One leapfrog integration step."""
    N = len(masses)

    # Compute accelerations
    acc = np.zeros((N, 3))
    for i in range(N):
        for j in range(N):
            if i != j:
                r_vec = positions[j] - positions[i]
                r = np.linalg.norm(r_vec)
                acc[i] += G * masses[j] * r_vec / (r**2 + softening**2)**1.5

    # Kick
    velocities += acc * dt / 2

    # Drift
    positions += velocities * dt

    # Recompute accelerations
    acc = np.zeros((N, 3))
    for i in range(N):
        for j in range(N):
            if i != j:
                r_vec = positions[j] - positions[i]
                r = np.linalg.norm(r_vec)
                acc[i] += G * masses[j] * r_vec / (r**2 + softening**2)**1.5

    # Kick
    velocities += acc * dt / 2

    return positions, velocities


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # N-body simulation parameters
    N = 100
    R = 1.0  # Characteristic radius (normalized units)
    M_total = 1.0  # Total mass (normalized units)

    # Initialize system slightly out of virial equilibrium
    np.random.seed(42)
    positions, velocities, masses = initialize_nbody(N, R, M_total)

    # Scale velocities to be out of equilibrium (50% too fast)
    velocities *= 1.5

    # Simulation
    t_cross = R / np.sqrt(G * M_total / R)  # Crossing time
    dt = 0.01 * t_cross
    n_steps = 500

    # Storage
    times = []
    kinetic_energies = []
    potential_energies = []
    virial_ratios = []

    for step in range(n_steps):
        K = compute_kinetic_energy(velocities, masses)
        U = compute_potential_energy(positions, masses)

        times.append(step * dt / t_cross)
        kinetic_energies.append(K)
        potential_energies.append(U)
        virial_ratios.append(2 * K / abs(U))

        positions, velocities = leapfrog_step(positions, velocities, masses, dt)

    times = np.array(times)

    # Plot 1: Particle positions at different times
    ax1 = axes[0, 0]

    # Re-run to get snapshots
    positions, velocities, masses = initialize_nbody(N, R, M_total)
    velocities *= 1.5

    # Initial
    ax1.scatter(positions[:, 0], positions[:, 1], s=10, alpha=0.5, label='t=0')

    # Evolve to t = 2 crossing times
    for _ in range(int(2 * t_cross / dt)):
        positions, velocities = leapfrog_step(positions, velocities, masses, dt)

    ax1.scatter(positions[:, 0], positions[:, 1], s=10, alpha=0.5, label='t=2 $t_{cross}$')

    # Evolve more
    for _ in range(int(3 * t_cross / dt)):
        positions, velocities = leapfrog_step(positions, velocities, masses, dt)

    ax1.scatter(positions[:, 0], positions[:, 1], s=10, alpha=0.5, label='t=5 $t_{cross}$')

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('N-body System Evolution')
    ax1.legend()
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-5, 5)

    # Plot 2: Energy evolution and virial equilibrium
    ax2 = axes[0, 1]

    K = np.array(kinetic_energies)
    U = np.array(potential_energies)
    E_total = K + U

    # Normalize
    E0 = E_total[0]
    ax2.plot(times, K / abs(E0), 'r-', lw=1.5, label='Kinetic energy')
    ax2.plot(times, U / abs(E0), 'b-', lw=1.5, label='Potential energy')
    ax2.plot(times, E_total / abs(E0), 'k-', lw=2, label='Total energy')

    # Virial equilibrium values
    ax2.axhline(y=1, color='gray', linestyle=':', alpha=0.7)
    ax2.axhline(y=-2, color='gray', linestyle=':', alpha=0.7)
    ax2.axhline(y=-1, color='gray', linestyle=':', alpha=0.7)

    ax2.set_xlabel('Time (crossing times)')
    ax2.set_ylabel('Energy / |E$_0$|')
    ax2.set_title('Energy Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Virial ratio evolution
    ax3 = axes[1, 0]

    ax3.plot(times, virial_ratios, 'b-', lw=2)
    ax3.axhline(y=1.0, color='red', linestyle='--', lw=2, label='Virial equilibrium (2K/|U| = 1)')
    ax3.axhline(y=virial_ratios[0], color='gray', linestyle=':', alpha=0.7, label='Initial')

    ax3.fill_between(times, 0.9, 1.1, alpha=0.2, color='green', label='Equilibrium zone')

    ax3.set_xlabel('Time (crossing times)')
    ax3.set_ylabel('Virial ratio 2K/|U|')
    ax3.set_title('Approach to Virial Equilibrium')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, times[-1])
    ax3.set_ylim(0, 3)

    # Plot 4: Application to galaxy cluster mass estimation
    ax4 = axes[1, 1]

    # Galaxy cluster parameters
    cluster_radius = 1.0 * kpc  # 1 Mpc
    sigma_v = 1000e3  # 1000 km/s velocity dispersion

    # Virial mass estimator: M = 3 * sigma^2 * R / G
    def virial_mass(sigma, R):
        return 3 * sigma**2 * R / G

    # Calculate for range of radii
    R_range = np.linspace(0.5, 3, 50) * kpc
    M_virial = virial_mass(sigma_v, R_range)

    ax4.plot(R_range / kpc, M_virial / M_sun, 'b-', lw=2)

    # Mark typical cluster
    M_cluster = virial_mass(sigma_v, cluster_radius)
    ax4.plot(1, M_cluster / M_sun, 'ro', markersize=12)
    ax4.annotate(f'$M_{{virial}}$ = {M_cluster/M_sun:.2e} $M_\\odot$',
                 xy=(1, M_cluster / M_sun),
                 xytext=(1.5, M_cluster / M_sun * 2),
                 fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='red'))

    # Observed light (baryonic) mass - much smaller
    M_baryonic = M_cluster / 10  # Typical baryon fraction ~10%
    ax4.axhline(y=M_baryonic / M_sun, color='orange', linestyle='--',
                label='Baryonic mass')

    ax4.set_xlabel('Cluster radius (Mpc)')
    ax4.set_ylabel('Virial Mass ($M_\\odot$)')
    ax4.set_title(f'Virial Mass Estimate ($\\sigma_v$ = {sigma_v/1e3:.0f} km/s)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')

    # Add dark matter note
    ax4.text(0.95, 0.95, 'Gap indicates\nDark Matter!',
             transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Experiment 266: Virial Theorem in Gravitational Systems\n'
                 '$2\\langle K \\rangle + \\langle U \\rangle = 0$ in equilibrium',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'virial_theorem_nbody.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'virial_theorem_nbody.png')}")


if __name__ == "__main__":
    main()
