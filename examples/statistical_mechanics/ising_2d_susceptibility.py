"""
Experiment 138: 2D Ising Model Susceptibility Peak

This example demonstrates the magnetic susceptibility divergence at the
critical temperature in the 2D Ising model.

The magnetic susceptibility is defined as:
chi = (1/T) * (<M^2> - <M>^2) / N = (1/T) * Var(M) / N

At the critical point, chi diverges as:
chi ~ |T - T_c|^(-gamma) where gamma = 7/4 = 1.75 (exact for 2D Ising)

Similarly, the specific heat diverges as:
C_V ~ |T - T_c|^(-alpha) where alpha = 0 (log divergence for 2D Ising)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Critical temperature for 2D Ising
T_C = 2.0 / np.log(1 + np.sqrt(2))  # ≈ 2.269


def initialize_lattice(L, state='random'):
    """Initialize square lattice of spins."""
    if state == 'random':
        return np.random.choice([-1, 1], size=(L, L))
    elif state == 'up':
        return np.ones((L, L), dtype=int)
    return -np.ones((L, L), dtype=int)


def metropolis_sweep(spins, T, J=1.0):
    """Perform one Metropolis sweep with optimized energy calculation."""
    L = spins.shape[0]

    for _ in range(L * L):
        i, j = np.random.randint(L), np.random.randint(L)
        s = spins[i, j]

        # Energy change for spin flip
        neighbors = (spins[(i+1) % L, j] + spins[(i-1) % L, j] +
                    spins[i, (j+1) % L] + spins[i, (j-1) % L])
        delta_E = 2 * J * s * neighbors

        if delta_E <= 0 or np.random.random() < np.exp(-delta_E / T):
            spins[i, j] = -s


def compute_observables(spins, J=1.0):
    """Compute energy and magnetization."""
    L = spins.shape[0]

    # Energy (nearest neighbor)
    E = -J * np.sum(spins * np.roll(spins, 1, axis=0))
    E += -J * np.sum(spins * np.roll(spins, 1, axis=1))

    # Magnetization
    M = np.sum(spins)

    return E, M


def run_measurement(L, T, n_equilibrate, n_measure, n_bins=10, J=1.0):
    """
    Run simulation and compute observables with error estimates.

    Uses binning analysis for error estimation.
    """
    spins = initialize_lattice(L)
    N = L * L

    # Equilibration
    for _ in range(n_equilibrate):
        metropolis_sweep(spins, T, J)

    # Measurement with binning
    bin_size = n_measure // n_bins
    E_bins = []
    M_bins = []
    M2_bins = []
    E2_bins = []

    for b in range(n_bins):
        E_sum, M_sum, E2_sum, M2_sum = 0, 0, 0, 0

        for _ in range(bin_size):
            metropolis_sweep(spins, T, J)
            E, M = compute_observables(spins, J)
            E_sum += E
            M_sum += np.abs(M)
            E2_sum += E**2
            M2_sum += M**2

        E_bins.append(E_sum / bin_size)
        M_bins.append(M_sum / bin_size)
        E2_bins.append(E2_sum / bin_size)
        M2_bins.append(M2_sum / bin_size)

    # Compute averages and errors
    E_mean = np.mean(E_bins) / N
    M_mean = np.mean(M_bins) / N
    E2_mean = np.mean(E2_bins) / N**2
    M2_mean = np.mean(M2_bins) / N**2

    E_err = np.std(E_bins) / (N * np.sqrt(n_bins))
    M_err = np.std(M_bins) / (N * np.sqrt(n_bins))

    # Susceptibility: chi = N * (<M^2> - <|M|>^2) / T
    chi = N * (M2_mean - (np.mean(M_bins) / N)**2) / T

    # Specific heat: C_V = N * (<E^2> - <E>^2) / T^2
    C_V = N * (E2_mean - E_mean**2) / T**2

    # Error estimation for chi and C_V (simple bootstrap)
    chi_samples = []
    C_V_samples = []
    for _ in range(100):
        idx = np.random.choice(n_bins, n_bins, replace=True)
        E_boot = np.mean([E_bins[i] for i in idx]) / N
        M_boot = np.mean([M_bins[i] for i in idx]) / N
        E2_boot = np.mean([E2_bins[i] for i in idx]) / N**2
        M2_boot = np.mean([M2_bins[i] for i in idx]) / N**2

        chi_samples.append(N * (M2_boot - M_boot**2) / T)
        C_V_samples.append(N * (E2_boot - E_boot**2) / T**2)

    chi_err = np.std(chi_samples)
    C_V_err = np.std(C_V_samples)

    return {
        'E': E_mean, 'E_err': E_err,
        'M': M_mean, 'M_err': M_err,
        'chi': chi, 'chi_err': chi_err,
        'C_V': C_V, 'C_V_err': C_V_err
    }


def power_law(t, A, gamma, c):
    """Power law function for fitting: A * |t|^(-gamma) + c"""
    return A * np.abs(t)**(-gamma) + c


def main():
    print("2D Ising Model: Susceptibility Peak")
    print("=" * 50)
    print(f"Critical temperature T_c = {T_C:.4f} J/k_B")
    print(f"Expected exponent gamma = 7/4 = 1.75")

    # Parameters
    L = 32
    n_equilibrate = 5000
    n_measure = 10000
    n_bins = 20

    # Temperature scan near T_c
    T_values = np.concatenate([
        np.linspace(1.5, 2.0, 8),
        np.linspace(2.0, 2.5, 15),
        np.linspace(2.5, 3.5, 8)
    ])
    T_values = np.unique(T_values)

    print(f"\nLattice size: {L} x {L}")
    print(f"Number of temperature points: {len(T_values)}")

    # Run simulations
    results = {'T': [], 'E': [], 'E_err': [], 'M': [], 'M_err': [],
               'chi': [], 'chi_err': [], 'C_V': [], 'C_V_err': []}

    for i, T in enumerate(T_values):
        print(f"  T = {T:.3f} ({i+1}/{len(T_values)})", end='\r')
        res = run_measurement(L, T, n_equilibrate, n_measure, n_bins)
        for key in res:
            results[key].append(res[key])
        results['T'].append(T)

    print("\n")

    # Convert to arrays
    for key in results:
        results[key] = np.array(results[key])

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Magnetization
    ax1 = axes[0, 0]
    ax1.errorbar(results['T'], results['M'], yerr=results['M_err'],
                 fmt='o', capsize=3, label='<|M|>/N')
    ax1.axvline(T_C, color='red', linestyle='--', label=f'$T_c$ = {T_C:.3f}')
    ax1.set_xlabel('Temperature (J/k_B)', fontsize=12)
    ax1.set_ylabel('Magnetization per spin', fontsize=12)
    ax1.set_title('Order Parameter', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Susceptibility
    ax2 = axes[0, 1]
    ax2.errorbar(results['T'], results['chi'], yerr=results['chi_err'],
                 fmt='o', capsize=3, color='green', label=r'$\chi$')
    ax2.axvline(T_C, color='red', linestyle='--', label=f'$T_c$ = {T_C:.3f}')
    ax2.set_xlabel('Temperature (J/k_B)', fontsize=12)
    ax2.set_ylabel(r'Susceptibility $\chi$', fontsize=12)
    ax2.set_title('Magnetic Susceptibility Peak', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Specific heat
    ax3 = axes[1, 0]
    ax3.errorbar(results['T'], results['C_V'], yerr=results['C_V_err'],
                 fmt='o', capsize=3, color='orange', label=r'$C_V$')
    ax3.axvline(T_C, color='red', linestyle='--', label=f'$T_c$ = {T_C:.3f}')
    ax3.set_xlabel('Temperature (J/k_B)', fontsize=12)
    ax3.set_ylabel(r'Specific heat $C_V / k_B$', fontsize=12)
    ax3.set_title('Specific Heat Peak', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Log-log plot for susceptibility
    ax4 = axes[1, 1]

    # Select points on both sides of T_c
    mask_above = (results['T'] > T_C + 0.1) & (results['T'] < 3.2)
    mask_below = (results['T'] < T_C - 0.1) & (results['T'] > 1.6)

    t_above = results['T'][mask_above] - T_C
    chi_above = results['chi'][mask_above]

    t_below = T_C - results['T'][mask_below]
    chi_below = results['chi'][mask_below]

    ax4.loglog(t_above, chi_above, 'bo', label='T > T_c')
    ax4.loglog(t_below, chi_below, 'rs', label='T < T_c')

    # Fit power law on high-T side
    if len(t_above) > 3:
        try:
            popt, _ = curve_fit(lambda t, A, gamma: A * t**(-gamma),
                               t_above, chi_above, p0=[1.0, 1.75], maxfev=5000)
            t_fit = np.logspace(np.log10(t_above.min()), np.log10(t_above.max()), 50)
            ax4.loglog(t_fit, popt[0] * t_fit**(-popt[1]), 'b--',
                      label=f'Fit: $\\gamma$ = {popt[1]:.2f}')
            print(f"Fitted gamma (T > T_c): {popt[1]:.3f}")
        except RuntimeError:
            print("Could not fit power law")

    # Theoretical line
    t_theory = np.logspace(-1, 0, 50)
    ax4.loglog(t_theory, 2 * t_theory**(-1.75), 'k:', alpha=0.5,
              label=r'Theory: $\gamma = 7/4$')

    ax4.set_xlabel('|T - T_c| / J/k_B', fontsize=12)
    ax4.set_ylabel(r'Susceptibility $\chi$', fontsize=12)
    ax4.set_title('Critical Scaling of Susceptibility', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3, which='both')

    plt.suptitle(f'2D Ising Model: Susceptibility Divergence (L = {L})',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Print results near T_c
    print("=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"{'T':>8} {'<E>/N':>10} {'<|M|>':>10} {'chi':>12} {'C_V':>12}")
    print("-" * 60)
    for i in range(len(results['T'])):
        print(f"{results['T'][i]:>8.3f} {results['E'][i]:>10.4f} "
              f"{results['M'][i]:>10.4f} {results['chi'][i]:>12.2f} "
              f"{results['C_V'][i]:>12.2f}")

    # Find peak location
    chi_max_idx = np.argmax(results['chi'])
    T_peak = results['T'][chi_max_idx]
    chi_peak = results['chi'][chi_max_idx]

    print(f"\nSusceptibility peak:")
    print(f"  T_peak = {T_peak:.3f} (T_c = {T_C:.3f})")
    print(f"  chi_max = {chi_peak:.2f}")

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'ising_2d_susceptibility.png'),
                dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {os.path.join(output_dir, 'ising_2d_susceptibility.png')}")


if __name__ == "__main__":
    main()
