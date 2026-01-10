"""
Experiment 232: Heisenberg Model Monte Carlo

Demonstrates Monte Carlo simulation of the classical Heisenberg model
on a 2D square lattice, showing the ferromagnetic-to-paramagnetic
phase transition and critical behavior.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def initialize_spins(L, initial='random'):
    """
    Initialize spin configuration on L x L lattice.

    Each spin is a 3D unit vector.

    Args:
        L: Lattice size
        initial: 'random' or 'ordered'

    Returns:
        spins: L x L x 3 array of unit vectors
    """
    if initial == 'ordered':
        # All spins pointing up
        spins = np.zeros((L, L, 3))
        spins[:, :, 2] = 1.0
    else:
        # Random orientations
        # Use spherical coordinates
        theta = np.arccos(2 * np.random.random((L, L)) - 1)
        phi = 2 * np.pi * np.random.random((L, L))

        spins = np.zeros((L, L, 3))
        spins[:, :, 0] = np.sin(theta) * np.cos(phi)
        spins[:, :, 1] = np.sin(theta) * np.sin(phi)
        spins[:, :, 2] = np.cos(theta)

    return spins


def compute_energy(spins, J=1.0):
    """
    Compute total energy of the spin configuration.

    H = -J * sum_{<i,j>} S_i . S_j

    Args:
        spins: L x L x 3 spin array
        J: Exchange coupling

    Returns:
        Total energy
    """
    L = spins.shape[0]
    energy = 0.0

    for i in range(L):
        for j in range(L):
            # Sum over nearest neighbors (with periodic BC)
            neighbors = [
                spins[(i+1) % L, j],
                spins[(i-1) % L, j],
                spins[i, (j+1) % L],
                spins[i, (j-1) % L]
            ]
            for neighbor in neighbors:
                energy -= 0.5 * J * np.dot(spins[i, j], neighbor)

    return energy


def compute_magnetization(spins):
    """
    Compute total magnetization vector and magnitude.

    M = (1/N) * sum_i S_i

    Args:
        spins: L x L x 3 spin array

    Returns:
        M_vec: Magnetization vector
        M: Magnetization magnitude
    """
    L = spins.shape[0]
    M_vec = np.mean(spins, axis=(0, 1))
    M = np.linalg.norm(M_vec)
    return M_vec, M


def metropolis_step(spins, T, J=1.0):
    """
    Perform one Metropolis Monte Carlo sweep.

    Args:
        spins: Current spin configuration
        T: Temperature
        J: Exchange coupling

    Returns:
        Updated spins, acceptance rate
    """
    L = spins.shape[0]
    beta = 1.0 / T if T > 0 else np.inf
    accepted = 0

    for _ in range(L * L):
        # Choose random site
        i = np.random.randint(L)
        j = np.random.randint(L)

        # Propose new random spin direction
        theta = np.arccos(2 * np.random.random() - 1)
        phi = 2 * np.pi * np.random.random()
        new_spin = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])

        # Calculate energy change
        neighbors = [
            spins[(i+1) % L, j],
            spins[(i-1) % L, j],
            spins[i, (j+1) % L],
            spins[i, (j-1) % L]
        ]
        neighbor_sum = sum(neighbors)

        delta_E = -J * np.dot(new_spin - spins[i, j], neighbor_sum)

        # Metropolis acceptance
        if delta_E <= 0 or np.random.random() < np.exp(-beta * delta_E):
            spins[i, j] = new_spin
            accepted += 1

    return spins, accepted / (L * L)


def run_simulation(L, T, n_equilibrate, n_sample, n_skip, J=1.0):
    """
    Run Monte Carlo simulation at temperature T.

    Args:
        L: Lattice size
        T: Temperature
        n_equilibrate: Number of equilibration sweeps
        n_sample: Number of measurement sweeps
        n_skip: Sweeps between measurements
        J: Exchange coupling

    Returns:
        E_samples: Energy samples
        M_samples: Magnetization samples
    """
    spins = initialize_spins(L, 'random')

    # Equilibration
    for _ in range(n_equilibrate):
        spins, _ = metropolis_step(spins, T, J)

    # Measurement
    E_samples = []
    M_samples = []

    for _ in range(n_sample):
        for _ in range(n_skip):
            spins, _ = metropolis_step(spins, T, J)

        E = compute_energy(spins, J) / (L * L)
        _, M = compute_magnetization(spins)
        E_samples.append(E)
        M_samples.append(M)

    return np.array(E_samples), np.array(M_samples)


def main():
    """Main simulation and visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Simulation parameters
    L = 16        # Lattice size
    J = 1.0       # Exchange coupling
    n_equilibrate = 500
    n_sample = 200
    n_skip = 5

    # Temperature range
    # Critical temperature for 2D Heisenberg is T_c = 0 (no long-range order)
    # But for finite size, there's a crossover
    T_range = np.linspace(0.3, 3.0, 15)

    print(f"Running Heisenberg Monte Carlo on {L}x{L} lattice...")

    # Calculate observables vs temperature
    E_mean = []
    E_std = []
    M_mean = []
    M_std = []
    chi_values = []  # Susceptibility
    C_values = []    # Heat capacity

    for T in T_range:
        print(f"  T = {T:.2f}...", end=' ')
        E_samples, M_samples = run_simulation(L, T, n_equilibrate, n_sample, n_skip, J)

        E_mean.append(np.mean(E_samples))
        E_std.append(np.std(E_samples))
        M_mean.append(np.mean(M_samples))
        M_std.append(np.std(M_samples))

        # Susceptibility chi = N * <M^2> / T
        chi = L * L * np.mean(M_samples**2) / T
        chi_values.append(chi)

        # Heat capacity C = N * (<E^2> - <E>^2) / T^2
        C = L * L * (np.mean(E_samples**2) - np.mean(E_samples)**2) / T**2
        C_values.append(C)

        print(f"<E> = {E_mean[-1]:.3f}, <M> = {M_mean[-1]:.3f}")

    E_mean = np.array(E_mean)
    M_mean = np.array(M_mean)
    chi_values = np.array(chi_values)
    C_values = np.array(C_values)

    # Plot 1: Energy vs temperature
    ax1 = axes[0, 0]

    ax1.errorbar(T_range, E_mean, yerr=E_std, fmt='bo-', lw=2, capsize=3,
                label='MC simulation')

    # Low T limit: E -> -2J (ground state)
    ax1.axhline(y=-2*J, color='gray', linestyle='--', alpha=0.5,
               label='Ground state E = -2J')

    ax1.set_xlabel('Temperature (J/kB)')
    ax1.set_ylabel('Energy per spin (J)')
    ax1.set_title('Energy vs Temperature')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Magnetization vs temperature
    ax2 = axes[0, 1]

    ax2.errorbar(T_range, M_mean, yerr=M_std, fmt='ro-', lw=2, capsize=3,
                label='MC simulation')
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    ax2.set_xlabel('Temperature (J/kB)')
    ax2.set_ylabel('Magnetization |M|')
    ax2.set_title('Magnetization vs Temperature')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)

    # Note about Mermin-Wagner theorem
    ax2.text(0.5, 0.95, 'Note: 2D Heisenberg has no long-range order\n(Mermin-Wagner theorem)',
             transform=ax2.transAxes, fontsize=9, va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 3: Susceptibility
    ax3 = axes[1, 0]

    ax3.plot(T_range, chi_values, 'go-', lw=2, label='Susceptibility')
    ax3.set_xlabel('Temperature (J/kB)')
    ax3.set_ylabel('Susceptibility chi')
    ax3.set_title('Magnetic Susceptibility')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Heat capacity
    ax4 = axes[1, 1]

    ax4.plot(T_range, C_values, 'mo-', lw=2, label='Heat capacity')
    ax4.set_xlabel('Temperature (J/kB)')
    ax4.set_ylabel('Heat capacity C')
    ax4.set_title('Heat Capacity')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f'Classical Heisenberg Model Monte Carlo ({L}x{L} lattice)\n'
                 r'$H = -J \sum_{\langle i,j \rangle} \vec{S}_i \cdot \vec{S}_j$',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save main plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'heisenberg_monte_carlo.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'heisenberg_monte_carlo.png')}")

    # Additional figure: Spin configuration snapshots
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))

    temperatures = [0.5, 1.5, 3.0]
    titles = ['Low T (ordered)', 'Intermediate T', 'High T (disordered)']

    for idx, (T, title) in enumerate(zip(temperatures, titles)):
        spins = initialize_spins(L, 'random')

        # Equilibrate
        for _ in range(n_equilibrate):
            spins, _ = metropolis_step(spins, T, J)

        ax = axes2[idx]

        # Plot spins as arrows (quiver plot of z-component)
        X, Y = np.meshgrid(range(L), range(L))
        ax.quiver(X, Y, spins[:, :, 0], spins[:, :, 1],
                  spins[:, :, 2], cmap='coolwarm', clim=[-1, 1])

        ax.set_title(f'{title}\nT = {T} J/kB')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')

    fig2.suptitle('Heisenberg Model: Spin Configurations at Different Temperatures',
                  fontsize=14, y=1.02)
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, 'heisenberg_spins.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'heisenberg_spins.png')}")


if __name__ == "__main__":
    main()
