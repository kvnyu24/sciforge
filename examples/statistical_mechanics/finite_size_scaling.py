"""
Experiment 141: Finite-Size Scaling and Critical Exponents

This example demonstrates finite-size scaling analysis to extract
critical exponents from Monte Carlo simulations of the 2D Ising model.

Near the critical point, observables scale with system size L as:
- Magnetization: <|M|> ~ L^(-beta/nu) * f_M((T-T_c)*L^(1/nu))
- Susceptibility: chi ~ L^(gamma/nu) * f_chi((T-T_c)*L^(1/nu))
- Specific heat: C_V ~ L^(alpha/nu) * f_C((T-T_c)*L^(1/nu))

For 2D Ising:
- beta = 1/8 (order parameter exponent)
- gamma = 7/4 (susceptibility exponent)
- nu = 1 (correlation length exponent)
- alpha = 0 (specific heat, logarithmic)

At T = T_c:
- <|M|> ~ L^(-beta/nu) = L^(-1/8)
- chi ~ L^(gamma/nu) = L^(7/4)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress

# Critical temperature and exponents for 2D Ising
T_C = 2.0 / np.log(1 + np.sqrt(2))
BETA = 1/8
GAMMA = 7/4
NU = 1
ALPHA = 0  # Logarithmic divergence


def initialize_lattice(L):
    """Initialize random spin configuration."""
    return np.random.choice([-1, 1], size=(L, L))


def wolff_step(spins, T, J=1.0):
    """Perform one Wolff cluster update."""
    L = spins.shape[0]
    p_add = 1 - np.exp(-2 * J / T)

    i0, j0 = np.random.randint(L), np.random.randint(L)
    seed_spin = spins[i0, j0]

    in_cluster = np.zeros((L, L), dtype=bool)
    in_cluster[i0, j0] = True

    stack = [(i0, j0)]
    while stack:
        i, j = stack.pop()
        for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            ni, nj = (i + di) % L, (j + dj) % L
            if not in_cluster[ni, nj] and spins[ni, nj] == seed_spin:
                if np.random.random() < p_add:
                    in_cluster[ni, nj] = True
                    stack.append((ni, nj))

    spins[in_cluster] *= -1


def compute_observables(spins, J=1.0):
    """Compute energy and magnetization."""
    L = spins.shape[0]
    E = -J * np.sum(spins * np.roll(spins, 1, axis=0))
    E += -J * np.sum(spins * np.roll(spins, 1, axis=1))
    M = np.sum(spins)
    return E, M


def run_simulation(L, T, n_equilibrate, n_measure, J=1.0):
    """Run simulation and compute observables with errors."""
    spins = initialize_lattice(L)
    N = L * L

    # Equilibration using Wolff
    for _ in range(n_equilibrate):
        wolff_step(spins, T, J)

    # Measurement
    E_samples = []
    M_samples = []

    for _ in range(n_measure):
        wolff_step(spins, T, J)
        E, M = compute_observables(spins, J)
        E_samples.append(E)
        M_samples.append(M)

    E_arr = np.array(E_samples)
    M_arr = np.array(M_samples)

    # Compute averages
    E_mean = np.mean(E_arr) / N
    M_mean = np.mean(np.abs(M_arr)) / N
    M2_mean = np.mean(M_arr**2) / N**2
    E2_mean = np.mean(E_arr**2) / N**2

    # Susceptibility and specific heat
    chi = N * (M2_mean - M_mean**2) / T
    C_V = N * (E2_mean - E_mean**2) / T**2

    # Error estimation (simple standard error)
    E_err = np.std(E_arr) / (N * np.sqrt(n_measure))
    M_err = np.std(np.abs(M_arr)) / (N * np.sqrt(n_measure))

    return {
        'E': E_mean, 'E_err': E_err,
        'M': M_mean, 'M_err': M_err,
        'chi': chi, 'C_V': C_V
    }


def main():
    print("Finite-Size Scaling: 2D Ising Critical Exponents")
    print("=" * 60)
    print(f"Critical temperature T_c = {T_C:.4f}")
    print(f"Exact exponents: beta = {BETA}, gamma = {GAMMA}, nu = {NU}")

    # System sizes
    L_values = [8, 12, 16, 24, 32, 48]

    # Temperature range near T_c
    n_T = 25
    T_range = np.linspace(1.8, 2.8, n_T)

    # Simulation parameters (use more sweeps for larger L)
    n_equilibrate_base = 500
    n_measure_base = 1000

    # Storage
    results = {L: {'T': [], 'M': [], 'chi': [], 'C_V': []} for L in L_values}

    # Run simulations
    print("\nRunning simulations...")
    for L in L_values:
        print(f"  L = {L}:", end=' ')
        n_eq = n_equilibrate_base
        n_meas = n_measure_base

        for i, T in enumerate(T_range):
            print(f"{i+1}/{n_T}", end='\r')
            res = run_simulation(L, T, n_eq, n_meas)
            results[L]['T'].append(T)
            results[L]['M'].append(res['M'])
            results[L]['chi'].append(res['chi'])
            results[L]['C_V'].append(res['C_V'])

        print(f"  L = {L}: done     ")

    # Convert to arrays
    for L in L_values:
        for key in results[L]:
            results[L][key] = np.array(results[L][key])

    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Plot 1: Magnetization vs T for different L
    ax1 = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(L_values)))
    for L, color in zip(L_values, colors):
        ax1.plot(results[L]['T'], results[L]['M'], 'o-', color=color,
                 markersize=4, label=f'L = {L}')
    ax1.axvline(T_C, color='red', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Temperature', fontsize=12)
    ax1.set_ylabel('<|M|> / N', fontsize=12)
    ax1.set_title('Magnetization vs Temperature', fontsize=12)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Susceptibility vs T
    ax2 = axes[0, 1]
    for L, color in zip(L_values, colors):
        ax2.plot(results[L]['T'], results[L]['chi'], 'o-', color=color,
                 markersize=4, label=f'L = {L}')
    ax2.axvline(T_C, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Temperature', fontsize=12)
    ax2.set_ylabel(r'$\chi$', fontsize=12)
    ax2.set_title('Susceptibility vs Temperature', fontsize=12)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Scaling at T_c - M vs L
    ax3 = axes[0, 2]

    # Get values at T closest to T_c
    M_at_Tc = []
    chi_at_Tc = []
    for L in L_values:
        idx = np.argmin(np.abs(results[L]['T'] - T_C))
        M_at_Tc.append(results[L]['M'][idx])
        chi_at_Tc.append(results[L]['chi'][idx])

    ax3.loglog(L_values, M_at_Tc, 'bo-', markersize=8, label='<|M|> at $T_c$')

    # Fit power law
    log_L = np.log(L_values)
    log_M = np.log(M_at_Tc)
    slope, intercept, _, _, _ = linregress(log_L, log_M)
    L_fit = np.linspace(min(L_values), max(L_values), 50)
    ax3.loglog(L_fit, np.exp(intercept) * L_fit**slope, 'b--',
               label=f'Fit: slope = {-slope:.3f} (theory: {BETA/NU:.3f})')

    ax3.set_xlabel('System size L', fontsize=12)
    ax3.set_ylabel('<|M|> at $T_c$', fontsize=12)
    ax3.set_title(r'$\langle|M|\rangle \sim L^{-\beta/\nu}$', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3, which='both')

    # Plot 4: Chi scaling at T_c
    ax4 = axes[1, 0]
    ax4.loglog(L_values, chi_at_Tc, 'go-', markersize=8, label=r'$\chi$ at $T_c$')

    log_chi = np.log(chi_at_Tc)
    slope_chi, intercept_chi, _, _, _ = linregress(log_L, log_chi)
    ax4.loglog(L_fit, np.exp(intercept_chi) * L_fit**slope_chi, 'g--',
               label=f'Fit: slope = {slope_chi:.3f} (theory: {GAMMA/NU:.3f})')

    ax4.set_xlabel('System size L', fontsize=12)
    ax4.set_ylabel(r'$\chi$ at $T_c$', fontsize=12)
    ax4.set_title(r'$\chi \sim L^{\gamma/\nu}$', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3, which='both')

    # Plot 5: Data collapse for magnetization
    ax5 = axes[1, 1]

    # Scaled variables: x = (T - T_c) * L^(1/nu), y = M * L^(beta/nu)
    for L, color in zip(L_values, colors):
        x = (results[L]['T'] - T_C) * L**(1/NU)
        y = results[L]['M'] * L**(BETA/NU)
        ax5.plot(x, y, 'o-', color=color, markersize=4, label=f'L = {L}')

    ax5.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax5.set_xlabel(r'$(T - T_c) L^{1/\nu}$', fontsize=12)
    ax5.set_ylabel(r'$\langle|M|\rangle L^{\beta/\nu}$', fontsize=12)
    ax5.set_title('Data Collapse: Magnetization', fontsize=12)
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # Plot 6: Data collapse for susceptibility
    ax6 = axes[1, 2]

    for L, color in zip(L_values, colors):
        x = (results[L]['T'] - T_C) * L**(1/NU)
        y = results[L]['chi'] / L**(GAMMA/NU)
        ax6.plot(x, y, 'o-', color=color, markersize=4, label=f'L = {L}')

    ax6.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax6.set_xlabel(r'$(T - T_c) L^{1/\nu}$', fontsize=12)
    ax6.set_ylabel(r'$\chi / L^{\gamma/\nu}$', fontsize=12)
    ax6.set_title('Data Collapse: Susceptibility', fontsize=12)
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    plt.suptitle('Finite-Size Scaling Analysis of 2D Ising Model',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Print summary of extracted exponents
    print("\n" + "=" * 60)
    print("Extracted Critical Exponents")
    print("=" * 60)
    print(f"{'Exponent':>15} {'Measured':>12} {'Exact':>12} {'Error %':>10}")
    print("-" * 50)

    beta_nu_meas = -slope
    gamma_nu_meas = slope_chi

    print(f"{'beta/nu':>15} {beta_nu_meas:>12.4f} {BETA/NU:>12.4f} "
          f"{100*abs(beta_nu_meas - BETA/NU)/(BETA/NU):>10.1f}")
    print(f"{'gamma/nu':>15} {gamma_nu_meas:>12.4f} {GAMMA/NU:>12.4f} "
          f"{100*abs(gamma_nu_meas - GAMMA/NU)/(GAMMA/NU):>10.1f}")

    # Hyperscaling relation check: 2*beta/nu + gamma/nu = d = 2
    hyperscaling = 2 * beta_nu_meas + gamma_nu_meas
    print(f"\nHyperscaling check: 2*beta/nu + gamma/nu = {hyperscaling:.3f} (should be 2)")

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'finite_size_scaling.png'),
                dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {os.path.join(output_dir, 'finite_size_scaling.png')}")


if __name__ == "__main__":
    main()
