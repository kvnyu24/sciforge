"""
Experiment 220: Kronig-Penney Model Band Gaps

Demonstrates the formation of energy bands and gaps in a periodic potential
using the Kronig-Penney model - a foundational model in solid state physics.
The model shows how periodic potentials lead to forbidden energy regions.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def kronig_penney_equation(ka, P, beta_a):
    """
    Kronig-Penney transcendental equation.

    For allowed energies: cos(ka) = cos(beta*a) + (P/beta*a)*sin(beta*a)

    Args:
        ka: Bloch wavevector times lattice constant
        P: Barrier strength parameter (P = m*V0*b*a / hbar^2)
        beta_a: sqrt(2mE/hbar^2) * a

    Returns:
        f(E) - cos(ka), which should be zero for allowed states
    """
    if beta_a == 0:
        return np.inf
    return np.cos(beta_a) + P * np.sinc(beta_a / np.pi) - np.cos(ka)


def find_band_structure(P, n_k=100, n_bands=6):
    """
    Calculate band structure for Kronig-Penney model.

    Args:
        P: Barrier strength parameter
        n_k: Number of k-points
        n_bands: Number of bands to compute

    Returns:
        k_values: Array of k*a values in first Brillouin zone
        energies: Array of energies for each band
    """
    k_values = np.linspace(-np.pi, np.pi, n_k)
    energies = np.zeros((n_bands, n_k))

    # For each k, find energies where f(E) = 0
    for ik, ka in enumerate(k_values):
        # Search for roots in different energy ranges
        beta_a_values = np.linspace(0.01, n_bands * np.pi, 10000)
        f_values = np.cos(beta_a_values) + P * np.sinc(beta_a_values / np.pi) - np.cos(ka)

        # Find sign changes (roots)
        roots = []
        for i in range(len(f_values) - 1):
            if f_values[i] * f_values[i+1] < 0:
                # Refine root by linear interpolation
                beta_root = beta_a_values[i] - f_values[i] * (beta_a_values[i+1] - beta_a_values[i]) / (f_values[i+1] - f_values[i])
                roots.append(beta_root)

        # Energy is proportional to beta^2
        for ib in range(min(len(roots), n_bands)):
            energies[ib, ik] = roots[ib]**2

    return k_values, energies


def main():
    """Main simulation and visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Band structure for different barrier strengths
    ax1 = axes[0, 0]
    P_values = [1.0, 3.0, 10.0]
    colors = ['blue', 'green', 'red']

    for P, color in zip(P_values, colors):
        k_vals, E_bands = find_band_structure(P, n_k=100, n_bands=4)
        for ib in range(4):
            label = f'P = {P}' if ib == 0 else None
            ax1.plot(k_vals / np.pi, E_bands[ib], color=color, lw=2, label=label)

    ax1.set_xlabel('ka / pi')
    ax1.set_ylabel('Energy (arb. units)')
    ax1.set_title('Band Structure: E vs k')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-1, 1)

    # Plot 2: Allowed energy regions (graphical solution)
    ax2 = axes[0, 1]

    P = 3.0
    beta_a = np.linspace(0.01, 4 * np.pi, 1000)
    f_beta = np.cos(beta_a) + P * np.sinc(beta_a / np.pi)

    ax2.plot(beta_a / np.pi, f_beta, 'b-', lw=2, label=r'$\cos(\beta a) + P\frac{\sin(\beta a)}{\beta a}$')
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='cos(ka) = +1')
    ax2.axhline(y=-1, color='gray', linestyle='--', alpha=0.7, label='cos(ka) = -1')
    ax2.fill_between(beta_a / np.pi, -1, 1, alpha=0.2, color='green', label='Allowed region')

    # Mark band edges
    for i, ba in enumerate(beta_a):
        if abs(f_beta[i]) <= 1:
            ax2.axvline(x=ba / np.pi, color='red', alpha=0.1, lw=0.5)

    ax2.set_xlabel(r'$\beta a / \pi$')
    ax2.set_ylabel(r'$f(\beta a)$')
    ax2.set_title(f'Graphical Solution (P = {P})')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-3, 3)

    # Plot 3: Band gaps vs barrier strength
    ax3 = axes[1, 0]

    P_range = np.linspace(0.5, 15, 30)
    gap1 = []
    gap2 = []
    gap3 = []

    for P in P_range:
        k_vals, E_bands = find_band_structure(P, n_k=50, n_bands=4)
        # Gap between bands
        if len(E_bands) > 1:
            gap1.append(np.min(E_bands[1]) - np.max(E_bands[0]))
        else:
            gap1.append(0)
        if len(E_bands) > 2:
            gap2.append(np.min(E_bands[2]) - np.max(E_bands[1]))
        else:
            gap2.append(0)
        if len(E_bands) > 3:
            gap3.append(np.min(E_bands[3]) - np.max(E_bands[2]))
        else:
            gap3.append(0)

    ax3.plot(P_range, gap1, 'b-', lw=2, label='1st gap')
    ax3.plot(P_range, gap2, 'g-', lw=2, label='2nd gap')
    ax3.plot(P_range, gap3, 'r-', lw=2, label='3rd gap')
    ax3.set_xlabel('Barrier Strength P')
    ax3.set_ylabel('Band Gap (arb. units)')
    ax3.set_title('Band Gaps vs Barrier Strength')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Density of states
    ax4 = axes[1, 1]

    P = 5.0
    k_vals, E_bands = find_band_structure(P, n_k=500, n_bands=4)

    # Flatten energies and create histogram
    all_energies = E_bands.flatten()
    all_energies = all_energies[all_energies > 0]  # Remove zeros

    E_bins = np.linspace(0, np.max(all_energies) * 1.1, 100)
    dos, edges = np.histogram(all_energies, bins=E_bins, density=True)
    E_centers = (edges[:-1] + edges[1:]) / 2

    ax4.fill_between(E_centers, dos, alpha=0.5, color='blue')
    ax4.plot(E_centers, dos, 'b-', lw=2)
    ax4.set_xlabel('Energy (arb. units)')
    ax4.set_ylabel('Density of States (arb. units)')
    ax4.set_title(f'DOS from Band Structure (P = {P})')
    ax4.grid(True, alpha=0.3)

    # Mark band gaps
    for ib in range(min(3, len(E_bands) - 1)):
        gap_bottom = np.max(E_bands[ib])
        gap_top = np.min(E_bands[ib + 1])
        ax4.axvspan(gap_bottom, gap_top, alpha=0.3, color='red')

    plt.suptitle('Kronig-Penney Model: Band Structure and Gaps\n'
                 'Periodic delta-function potential demonstration',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'kronig_penney_band_gaps.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'kronig_penney_band_gaps.png')}")


if __name__ == "__main__":
    main()
