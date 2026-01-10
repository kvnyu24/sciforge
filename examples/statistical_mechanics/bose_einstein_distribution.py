"""
Experiment 136: Bose-Einstein Distribution and BEC Onset

This example demonstrates the Bose-Einstein distribution for bosons and
shows the onset of Bose-Einstein Condensation (BEC).

The Bose-Einstein distribution is:
n(E) = 1 / (exp((E - mu)/(k_B*T)) - 1)

Key features:
- mu <= 0 for bosons (chemical potential never exceeds ground state energy)
- At T_c, mu -> 0 and macroscopic occupation of ground state begins
- Below T_c, a finite fraction occupies the ground state (BEC)

Critical temperature for 3D ideal Bose gas:
T_c = (2*pi*hbar^2 / (m*k_B)) * (n / zeta(3/2))^(2/3)

where zeta(3/2) = 2.612...
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import zeta
from scipy.optimize import brentq

# Physical constants
k_B = 1.380649e-23  # Boltzmann constant (J/K)
hbar = 1.054571817e-34  # Reduced Planck constant (J*s)
m_Rb = 87 * 1.66054e-27  # Mass of Rb-87 (kg)


def bose_einstein(E, mu, T, k_B=1.0):
    """
    Bose-Einstein distribution function.

    Args:
        E: Energy (can be array), E >= 0
        mu: Chemical potential (must be < min(E))
        T: Temperature
        k_B: Boltzmann constant

    Returns:
        Average occupation number
    """
    if T <= 0:
        return np.where(E == 0, np.inf, 0.0)

    x = (E - mu) / (k_B * T)
    # Avoid numerical issues
    return np.where(x > 500, 0.0, np.where(x < 1e-10, k_B * T / (E - mu),
                                            1.0 / (np.exp(x) - 1)))


def polylog(s, z):
    """
    Polylogarithm function Li_s(z) using series expansion.

    Li_s(z) = sum_{k=1}^{inf} z^k / k^s

    For |z| < 1, converges. For z=1, Li_s(1) = zeta(s).
    """
    if z <= 0:
        return 0.0
    if z >= 1:
        return zeta(s)

    result = 0.0
    for k in range(1, 1000):
        term = z**k / k**s
        result += term
        if abs(term) < 1e-12 * abs(result):
            break
    return result


def density_of_states_3d(E, prefactor=1.0):
    """
    3D density of states g(E) = prefactor * sqrt(E)

    For free particles: prefactor = (2m)^(3/2) / (4*pi^2*hbar^3)
    """
    if np.isscalar(E):
        return 0.0 if E <= 0 else prefactor * np.sqrt(E)
    result = np.zeros_like(E, dtype=float)
    mask = E > 0
    result[mask] = prefactor * np.sqrt(E[mask])
    return result


def compute_condensate_fraction(T, T_c):
    """
    Compute condensate fraction N_0/N for ideal Bose gas.

    N_0/N = 1 - (T/T_c)^(3/2) for T < T_c
    N_0/N = 0 for T >= T_c
    """
    if T >= T_c:
        return 0.0
    return 1.0 - (T / T_c)**(3/2)


def compute_chemical_potential(T, T_c, E_F_scale=1.0):
    """
    Compute chemical potential for ideal Bose gas.

    For T > T_c: mu < 0, determined by fixing particle number
    For T <= T_c: mu = 0 (condensate forms)

    Using dimensionless units where T_c = 1.
    """
    if T <= T_c:
        return 0.0

    # For T > T_c, solve for mu from n = integral of g(E)*n_BE(E) dE
    # In reduced units with proper normalization
    t = T / T_c

    # Approximate solution using thermal de Broglie wavelength scaling
    # mu/k_B*T_c ≈ ln(1 - (T_c/T)^(3/2)) for T > T_c
    # This is a rough approximation
    x = 1 - (1/t)**(1.5)
    if x > 0:
        mu = T * np.log(x)
    else:
        mu = -0.01  # Small negative value near T_c

    return min(mu, -1e-10)  # Ensure mu < 0


def main():
    print("Bose-Einstein Distribution and BEC Onset")
    print("=" * 50)

    # Using reduced units: k_B = 1, T_c = 1
    T_c = 1.0  # Critical temperature

    print(f"Critical temperature T_c = {T_c}")
    print(f"zeta(3/2) = {zeta(3/2):.4f}")

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Bose-Einstein distribution at different temperatures
    ax1 = axes[0, 0]
    E = np.linspace(0.001, 3.0, 500)
    T_values = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
    colors = plt.cm.coolwarm_r(np.linspace(0, 1, len(T_values)))

    for T, color in zip(T_values, colors):
        mu = compute_chemical_potential(T, T_c)
        n = bose_einstein(E, mu, T)
        label = f'T = {T:.1f} T_c' + (' (BEC)' if T < T_c else '')
        ax1.plot(E, n, color=color, lw=2, label=label)

    ax1.set_xlabel('Energy / $k_B T_c$', fontsize=12)
    ax1.set_ylabel('Occupation number n(E)', fontsize=12)
    ax1.set_title('Bose-Einstein Distribution', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 3)
    ax1.set_ylim(0, 10)

    # Plot 2: Condensate fraction vs temperature
    ax2 = axes[0, 1]
    T_range = np.linspace(0.01, 2.0, 200)
    N0_N = [compute_condensate_fraction(T, T_c) for T in T_range]

    ax2.plot(T_range / T_c, N0_N, 'b-', lw=2, label='$N_0/N = 1 - (T/T_c)^{3/2}$')
    ax2.axvline(1.0, color='red', linestyle='--', lw=1.5, label='$T_c$')
    ax2.fill_between(T_range / T_c, N0_N, alpha=0.3)
    ax2.set_xlabel('$T / T_c$', fontsize=12)
    ax2.set_ylabel('Condensate fraction $N_0 / N$', fontsize=12)
    ax2.set_title('BEC Order Parameter', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 2)
    ax2.set_ylim(0, 1.1)

    # Plot 3: Chemical potential vs temperature
    ax3 = axes[1, 0]
    T_range_mu = np.linspace(0.5, 3.0, 200)
    mu_values = [compute_chemical_potential(T, T_c) for T in T_range_mu]

    ax3.plot(T_range_mu / T_c, mu_values, 'b-', lw=2)
    ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax3.axvline(1.0, color='red', linestyle='--', lw=1.5, label='$T_c$')

    # Shade BEC region
    T_bec = T_range_mu[T_range_mu <= T_c]
    ax3.fill_between([0, 1], [-0.5, -0.5], [0.1, 0.1], alpha=0.2, color='blue',
                     label='BEC phase (mu=0)')

    ax3.set_xlabel('$T / T_c$', fontsize=12)
    ax3.set_ylabel('Chemical potential $\\mu / k_B T_c$', fontsize=12)
    ax3.set_title('Chemical Potential vs Temperature', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 3)
    ax3.set_ylim(-0.5, 0.1)

    # Plot 4: Comparison of Fermi-Dirac, Bose-Einstein, and Maxwell-Boltzmann
    ax4 = axes[1, 1]
    E_compare = np.linspace(0.1, 5, 200)
    T_compare = 1.0
    mu_compare = 0.0  # For comparison at E_F or E=0

    # Bose-Einstein (mu = -0.5 to avoid singularity)
    n_BE = bose_einstein(E_compare, -0.5, T_compare)

    # Fermi-Dirac
    n_FD = 1.0 / (np.exp((E_compare - 1.0) / T_compare) + 1)

    # Maxwell-Boltzmann
    n_MB = np.exp(-(E_compare - 0.5) / T_compare)

    ax4.semilogy(E_compare, n_BE, 'b-', lw=2, label='Bose-Einstein')
    ax4.semilogy(E_compare, n_FD, 'r-', lw=2, label='Fermi-Dirac')
    ax4.semilogy(E_compare, n_MB, 'g--', lw=2, label='Maxwell-Boltzmann')

    ax4.set_xlabel('Energy / $k_B T$', fontsize=12)
    ax4.set_ylabel('Occupation number (log scale)', fontsize=12)
    ax4.set_title('Quantum Statistics Comparison', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3, which='both')
    ax4.set_ylim(1e-3, 100)

    plt.suptitle('Bose-Einstein Condensation: Quantum Statistics of Bosons',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Physical example: Rb-87 BEC
    print("\n" + "=" * 50)
    print("Physical Example: Rb-87 BEC")
    print("=" * 50)

    # Typical BEC parameters
    n_density = 1e14 * 1e6  # atoms/m^3 (10^14 atoms/cm^3)

    # Critical temperature
    zeta_32 = zeta(3/2)  # ≈ 2.612
    T_c_Rb = (2 * np.pi * hbar**2 / (m_Rb * k_B)) * (n_density / zeta_32)**(2/3)

    print(f"Atomic density: {n_density/1e6:.1e} cm^-3")
    print(f"Critical temperature: {T_c_Rb*1e9:.1f} nK")
    print(f"Thermal de Broglie wavelength at T_c: {np.sqrt(2*np.pi*hbar**2/(m_Rb*k_B*T_c_Rb))*1e6:.2f} um")

    # Condensate fraction at various temperatures
    print("\nCondensate fraction:")
    for T_ratio in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        N0_frac = compute_condensate_fraction(T_ratio * T_c_Rb, T_c_Rb)
        T_nK = T_ratio * T_c_Rb * 1e9
        print(f"  T = {T_nK:.1f} nK ({T_ratio:.1f} T_c): N_0/N = {N0_frac:.3f}")

    # Add second figure for energy distribution
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

    # Energy distribution in excited states
    ax5 = axes2[0]
    E_dist = np.linspace(0.001, 5, 500)

    for T_ratio in [0.3, 0.5, 0.7, 1.0, 1.5]:
        T = T_ratio * T_c
        mu = compute_chemical_potential(T, T_c)
        # Energy distribution: g(E) * n(E)
        g_E = density_of_states_3d(E_dist, prefactor=1.0)
        n_E = bose_einstein(E_dist, mu, T)
        dist = g_E * n_E
        # Normalize for visualization
        dist = dist / np.max(dist) if np.max(dist) > 0 else dist
        ax5.plot(E_dist, dist, lw=2, label=f'T = {T_ratio:.1f} T_c')

    ax5.set_xlabel('Energy / $k_B T_c$', fontsize=12)
    ax5.set_ylabel('Energy distribution (normalized)', fontsize=12)
    ax5.set_title('Energy Distribution in Excited States', fontsize=12)
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Phase diagram
    ax6 = axes2[1]
    T_phase = np.linspace(0, 2.5, 100)
    n_phase = np.linspace(0.1, 2, 100)
    T_grid, n_grid = np.meshgrid(T_phase, n_phase)

    # Critical line: T/T_c = (n/n_c)^(2/3) => normalized: T = n^(2/3) at boundary
    # For fixed n, T_c ~ n^(2/3), so phase boundary is T/T_c(n) = 1

    # Simple phase diagram with n/n_c vs T/T_c
    ax6.fill_between([0, 1], [0, 0], [2.5, 2.5], alpha=0.3, color='blue',
                     label='BEC phase')
    ax6.fill_between([1, 2.5], [0, 0], [2.5, 2.5], alpha=0.3, color='red',
                     label='Normal phase')
    ax6.axvline(1.0, color='black', lw=2, label='Phase boundary')

    ax6.set_xlabel('$T / T_c$', fontsize=12)
    ax6.set_ylabel('Density (arb. units)', fontsize=12)
    ax6.set_title('BEC Phase Diagram (fixed density)', fontsize=12)
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plots
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)

    fig.savefig(os.path.join(output_dir, 'bose_einstein_distribution.png'),
                dpi=150, bbox_inches='tight')
    fig2.savefig(os.path.join(output_dir, 'bose_einstein_phase.png'),
                 dpi=150, bbox_inches='tight')

    print(f"\nPlots saved to {output_dir}")


if __name__ == "__main__":
    main()
