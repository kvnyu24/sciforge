"""
Experiment 137: 2D Ising Model with Metropolis Algorithm

This example simulates the 2D Ising model on a square lattice using
the Metropolis-Hastings algorithm.

The Ising Hamiltonian is:
H = -J * sum_{<ij>} s_i * s_j - h * sum_i s_i

where:
- J = coupling constant (J > 0 for ferromagnet)
- s_i = +/- 1 spin variables
- h = external magnetic field
- <ij> denotes nearest neighbor pairs

The critical temperature for 2D Ising (h=0) is:
T_c = 2J / (k_B * ln(1 + sqrt(2))) = 2.269 J/k_B
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Critical temperature for 2D Ising
T_C = 2.0 / np.log(1 + np.sqrt(2))  # ≈ 2.269


def initialize_lattice(L, state='random'):
    """
    Initialize a square lattice of spins.

    Args:
        L: Linear size of lattice (L x L)
        state: 'random', 'up', or 'down'

    Returns:
        2D numpy array of spins (+1 or -1)
    """
    if state == 'random':
        return np.random.choice([-1, 1], size=(L, L))
    elif state == 'up':
        return np.ones((L, L), dtype=int)
    elif state == 'down':
        return -np.ones((L, L), dtype=int)
    else:
        raise ValueError(f"Unknown state: {state}")


def compute_energy(spins, J=1.0, h=0.0):
    """
    Compute total energy of the spin configuration.

    Args:
        spins: 2D array of spins
        J: Coupling constant
        h: External field

    Returns:
        Total energy
    """
    L = spins.shape[0]
    energy = 0.0

    # Nearest neighbor interactions (with periodic BC)
    for i in range(L):
        for j in range(L):
            s = spins[i, j]
            # Sum over neighbors (right and down to avoid double counting)
            neighbors = spins[(i+1) % L, j] + spins[i, (j+1) % L]
            energy -= J * s * neighbors

    # External field
    energy -= h * np.sum(spins)

    return energy


def compute_magnetization(spins):
    """Compute magnetization per spin."""
    return np.mean(spins)


def metropolis_step(spins, T, J=1.0, h=0.0):
    """
    Perform one Metropolis sweep (L^2 single-spin flip attempts).

    Args:
        spins: 2D array of spins (modified in place)
        T: Temperature
        J: Coupling constant
        h: External field

    Returns:
        Number of accepted moves
    """
    L = spins.shape[0]
    accepted = 0

    for _ in range(L * L):
        # Select random spin
        i = np.random.randint(L)
        j = np.random.randint(L)

        s = spins[i, j]

        # Compute energy change for flipping this spin
        neighbors_sum = (spins[(i+1) % L, j] + spins[(i-1) % L, j] +
                        spins[i, (j+1) % L] + spins[i, (j-1) % L])

        delta_E = 2 * J * s * neighbors_sum + 2 * h * s

        # Metropolis acceptance
        if delta_E <= 0:
            spins[i, j] = -s
            accepted += 1
        elif np.random.random() < np.exp(-delta_E / T):
            spins[i, j] = -s
            accepted += 1

    return accepted


def run_simulation(L, T, n_equilibrate, n_measure, J=1.0, h=0.0):
    """
    Run Ising model simulation at temperature T.

    Args:
        L: Lattice size
        T: Temperature
        n_equilibrate: Number of equilibration sweeps
        n_measure: Number of measurement sweeps
        J: Coupling constant
        h: External field

    Returns:
        Dictionary with results
    """
    # Initialize
    spins = initialize_lattice(L, state='random')

    # Equilibration
    for _ in range(n_equilibrate):
        metropolis_step(spins, T, J, h)

    # Measurement
    energies = []
    magnetizations = []
    configurations = []

    for step in range(n_measure):
        metropolis_step(spins, T, J, h)

        E = compute_energy(spins, J, h)
        M = compute_magnetization(spins)

        energies.append(E)
        magnetizations.append(M)

        # Save some configurations
        if step % (n_measure // 10) == 0:
            configurations.append(spins.copy())

    return {
        'energies': np.array(energies),
        'magnetizations': np.array(magnetizations),
        'configurations': configurations,
        'final_spins': spins.copy()
    }


def main():
    print("2D Ising Model: Metropolis Algorithm")
    print("=" * 50)
    print(f"Critical temperature T_c = {T_C:.4f} J/k_B")

    # Parameters
    L = 32  # Lattice size
    n_equilibrate = 5000
    n_measure = 10000
    J = 1.0

    # Temperatures to simulate
    temperatures = [1.5, 2.0, T_C, 2.5, 3.0, 4.0]

    print(f"\nLattice size: {L} x {L}")
    print(f"Equilibration sweeps: {n_equilibrate}")
    print(f"Measurement sweeps: {n_measure}")

    # Run simulations
    results = {}
    for T in temperatures:
        print(f"\nSimulating T = {T:.2f}...")
        results[T] = run_simulation(L, T, n_equilibrate, n_measure, J)

        E_mean = np.mean(results[T]['energies']) / (L * L)
        M_mean = np.mean(np.abs(results[T]['magnetizations']))
        print(f"  <E>/N = {E_mean:.4f}")
        print(f"  <|M|> = {M_mean:.4f}")

    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Spin configurations at different temperatures
    ax_configs = axes[0, :]
    cmap = ListedColormap(['blue', 'red'])

    for ax, T in zip(ax_configs, [1.5, T_C, 4.0]):
        spins = results[T]['final_spins']
        ax.imshow(spins, cmap=cmap, interpolation='nearest')
        ax.set_title(f'T = {T:.2f} J/k_B' + (' (T_c)' if T == T_C else ''))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xticks([])
        ax.set_yticks([])

    # Plot 2: Energy time series
    ax_E = axes[1, 0]
    for T in [1.5, T_C, 4.0]:
        E = results[T]['energies'] / (L * L)
        ax_E.plot(E[:2000], alpha=0.7, label=f'T = {T:.2f}')
    ax_E.set_xlabel('Monte Carlo sweep', fontsize=12)
    ax_E.set_ylabel('Energy per spin', fontsize=12)
    ax_E.set_title('Energy Time Series', fontsize=12)
    ax_E.legend()
    ax_E.grid(True, alpha=0.3)

    # Plot 3: Magnetization time series
    ax_M = axes[1, 1]
    for T in [1.5, T_C, 4.0]:
        M = results[T]['magnetizations']
        ax_M.plot(M[:2000], alpha=0.7, label=f'T = {T:.2f}')
    ax_M.set_xlabel('Monte Carlo sweep', fontsize=12)
    ax_M.set_ylabel('Magnetization per spin', fontsize=12)
    ax_M.set_title('Magnetization Time Series', fontsize=12)
    ax_M.legend()
    ax_M.grid(True, alpha=0.3)

    # Plot 4: Phase transition (M vs T)
    ax_phase = axes[1, 2]

    # Run quick simulations at more temperatures
    T_scan = np.linspace(1.0, 4.0, 20)
    M_avg = []
    M_std = []
    E_avg = []

    print("\nTemperature scan for phase diagram...")
    for T in T_scan:
        result = run_simulation(L, T, 2000, 3000, J)
        M_avg.append(np.mean(np.abs(result['magnetizations'])))
        M_std.append(np.std(np.abs(result['magnetizations'])))
        E_avg.append(np.mean(result['energies']) / (L * L))

    ax_phase.errorbar(T_scan, M_avg, yerr=M_std, fmt='o-', capsize=3,
                      label='<|M|>')
    ax_phase.axvline(T_C, color='red', linestyle='--', label=f'$T_c$ = {T_C:.3f}')
    ax_phase.set_xlabel('Temperature (J/k_B)', fontsize=12)
    ax_phase.set_ylabel('|Magnetization| per spin', fontsize=12)
    ax_phase.set_title('Spontaneous Magnetization vs T', fontsize=12)
    ax_phase.legend()
    ax_phase.grid(True, alpha=0.3)

    plt.suptitle('2D Ising Model: Metropolis Monte Carlo Simulation',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Second figure: more detailed analysis
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

    # Energy vs T
    ax_ET = axes2[0]
    ax_ET.plot(T_scan, E_avg, 'bo-', markersize=6)
    ax_ET.axvline(T_C, color='red', linestyle='--', label=f'$T_c$')
    ax_ET.set_xlabel('Temperature (J/k_B)', fontsize=12)
    ax_ET.set_ylabel('Energy per spin', fontsize=12)
    ax_ET.set_title('Energy vs Temperature', fontsize=12)
    ax_ET.legend()
    ax_ET.grid(True, alpha=0.3)

    # Magnetization histogram at T_c
    ax_hist = axes2[1]
    for T, color in zip([1.5, T_C, 4.0], ['blue', 'red', 'green']):
        M = results[T]['magnetizations']
        ax_hist.hist(M, bins=50, alpha=0.5, density=True, label=f'T = {T:.2f}',
                     color=color)
    ax_hist.set_xlabel('Magnetization per spin', fontsize=12)
    ax_hist.set_ylabel('Probability density', fontsize=12)
    ax_hist.set_title('Magnetization Distribution', fontsize=12)
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3)

    plt.tight_layout()

    # Print summary
    print("\n" + "=" * 50)
    print("Results Summary")
    print("=" * 50)
    print(f"{'T':>8} {'<E>/N':>12} {'<|M|>':>12} {'sigma_M':>12}")
    print("-" * 50)
    for T, m, s, e in zip(T_scan, M_avg, M_std, E_avg):
        print(f"{T:>8.3f} {e:>12.4f} {m:>12.4f} {s:>12.4f}")

    # Save plots
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'ising_2d_metropolis.png'),
                dpi=150, bbox_inches='tight')
    fig2.savefig(os.path.join(output_dir, 'ising_2d_analysis.png'),
                 dpi=150, bbox_inches='tight')

    print(f"\nPlots saved to {output_dir}")


if __name__ == "__main__":
    main()
