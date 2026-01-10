"""
Experiment 244: t-J Model

Demonstrates the t-J model, derived from the Hubbard model in the
large-U limit. Shows the interplay between hole hopping and
antiferromagnetic spin correlations in strongly correlated systems.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg


def t_j_hamiltonian_2site(t, J):
    """
    Exact t-J Hamiltonian for 2 sites with 2 electrons.

    H = -t * P * (c^+_1 c_2 + h.c.) * P + J * (S_1 . S_2 - n_1 n_2 / 4)

    where P projects out double occupancy.

    For 2 electrons on 2 sites with no double occupancy:
    Basis: |up,down>, |down,up>

    Returns:
        Singlet and triplet energies
    """
    # Singlet: |S> = (|up,down> - |down,up>) / sqrt(2)
    # Triplet: |T> = (|up,down> + |down,up>) / sqrt(2)

    # S_1 . S_2 = (1/2)(S_1+ S_2- + S_1- S_2+) + S_1z S_2z
    #           = -3/4 for singlet, +1/4 for triplet
    # n_1 n_2 / 4 = 1/4 (always occupied)

    E_singlet = J * (-3/4 - 1/4)  # = -J
    E_triplet = J * (1/4 - 1/4)   # = 0

    return E_singlet, E_triplet


def t_j_3site_1hole(t, J):
    """
    t-J model for 3 sites with 2 electrons (1 hole).

    Shows how holes move in an antiferromagnetic background.

    Returns:
        Eigenvalues, eigenvectors
    """
    # Basis states: position of hole, spin configuration of remaining 2
    # For S_z = 0: hole at site 0, 1, or 2 with up-down or down-up

    # Simplified: consider just the lowest spin sector
    # Basis: |h0, up1 down2>, |h0, down1 up2>, |h1, up0 down2>, |h1, down0 up2>,
    #        |h2, up0 down1>, |h2, down0 up1>

    dim = 6
    H = np.zeros((dim, dim))

    # Spin exchange on occupied bonds
    # J term acts on adjacent spins, flipping them and adding -J/2 (off-diag)
    # or giving +/- J/4 for parallel/antiparallel spins

    # This is complex - use a simplified effective model
    # Hole hopping with spin constraint

    # For now, use a simpler 3-state effective model
    # |0>, |1>, |2> = hole position
    # In AFM background, hopping is modulated by spin frustration

    H_eff = np.array([
        [0, -t, 0],
        [-t, 0, -t],
        [0, -t, 0]
    ])

    return linalg.eigh(H_eff)


def spin_correlation(J, T, n_sites=2):
    """
    Nearest-neighbor spin correlation <S_i . S_j> vs temperature.

    For Heisenberg dimer: <S1.S2> = -3/4 * tanh(J/(4T))

    Args:
        J: Exchange coupling
        T: Temperature (array)
        n_sites: Number of sites (for scaling)

    Returns:
        Spin correlation
    """
    # Two-site Heisenberg
    E_S = -3*J/4  # Singlet
    E_T = J/4     # Triplet (threefold degenerate)

    # Partition function
    Z = np.exp(-E_S / T) + 3 * np.exp(-E_T / T)

    # <S1.S2>
    S_corr = (-3/4 * np.exp(-E_S / T) + 1/4 * 3 * np.exp(-E_T / T)) / Z

    return S_corr


def hole_mobility(t, J, T):
    """
    Effective hole mobility in t-J model (schematic).

    At low T, holes are mobile carriers in AFM background.
    At high T, spin disorder reduces coherence.

    Returns:
        Qualitative mobility
    """
    # Coherent hopping rate ~ t^2/J at low T
    # Suppressed by thermal fluctuations at high T

    coherent = t**2 / J
    thermal = np.exp(-J / T)

    return coherent * thermal


def main():
    """Main simulation and visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    t = 1.0  # Hopping
    J = 0.3  # Exchange (J ~ 4t^2/U for large U)

    # Plot 1: Energy spectrum of t-J model
    ax1 = axes[0, 0]

    J_range = np.linspace(0, 2, 100)

    E_singlet = []
    E_triplet = []

    for J_val in J_range:
        E_s, E_t = t_j_hamiltonian_2site(t, J_val)
        E_singlet.append(E_s)
        E_triplet.append(E_t)

    ax1.plot(J_range / t, E_singlet, 'b-', lw=2, label='Singlet (AFM)')
    ax1.plot(J_range / t, E_triplet, 'r-', lw=2, label='Triplet (FM)')

    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    ax1.set_xlabel('J / t')
    ax1.set_ylabel('Energy / t')
    ax1.set_title('Two-Site t-J Model: Singlet vs Triplet')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Annotate binding energy
    ax1.fill_between(J_range / t, E_singlet, E_triplet, alpha=0.2, color='blue')
    ax1.text(1, -0.5, 'Singlet-Triplet\nsplitting = J', fontsize=10, ha='center')

    # Plot 2: Spin correlations vs temperature
    ax2 = axes[0, 1]

    J = 0.3
    T_range = np.linspace(0.01, 2, 100)

    S_corr = spin_correlation(J, T_range)

    ax2.plot(T_range / J, S_corr, 'b-', lw=2)
    ax2.axhline(y=-3/4, color='red', linestyle='--', alpha=0.5, label='T=0 limit')
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    ax2.set_xlabel('T / J')
    ax2.set_ylabel(r'$\langle S_1 \cdot S_2 \rangle$')
    ax2.set_title('Spin Correlation vs Temperature')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Mark crossover temperature
    ax2.axvline(x=1, color='green', linestyle=':', alpha=0.5)
    ax2.text(1.1, -0.3, 'T ~ J', fontsize=10)

    # Plot 3: Hole dispersion in AFM background
    ax3 = axes[1, 0]

    # In t-J model, hole dispersion is modified by spin fluctuations
    # E(k) ~ -2t*cos(k) modified to ~ -2t*sqrt(1 + (J/2t)^2)*cos(k)

    k = np.linspace(-np.pi, np.pi, 200)

    J_t_ratios = [0, 0.3, 0.6, 1.0]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(J_t_ratios)))

    for J_t, color in zip(J_t_ratios, colors):
        if J_t == 0:
            E_k = -2 * t * np.cos(k)
            label = 'Free electron'
        else:
            # Simplified: bandwidth reduction
            t_eff = t * np.sqrt(1 + J_t**2)
            E_k = -2 * t_eff * np.cos(k)
            label = f'J/t = {J_t}'

        ax3.plot(k / np.pi, E_k / t, color=color, lw=2, label=label)

    ax3.set_xlabel('k / pi')
    ax3.set_ylabel('E(k) / t')
    ax3.set_title('Hole Dispersion in t-J Model')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Mark zone boundary
    ax3.axvline(x=-1, color='gray', linestyle=':', alpha=0.5)
    ax3.axvline(x=1, color='gray', linestyle=':', alpha=0.5)

    # Plot 4: Phase diagram (schematic for cuprates)
    ax4 = axes[1, 1]

    # Doping x vs Temperature phase diagram
    x_range = np.linspace(0, 0.35, 100)
    T_range = np.linspace(0, 1, 100)
    X, T = np.meshgrid(x_range, T_range)

    # Regions (schematic):
    # AFM: x < 0.05
    # Superconducting dome: 0.05 < x < 0.3, T < T_c(x)
    # Strange metal: above SC dome

    T_c_max = 0.3
    x_opt = 0.16  # Optimal doping

    # SC dome: T_c(x) = T_c_max * (1 - ((x - x_opt)/0.1)^2)
    T_c = T_c_max * (1 - ((X - x_opt) / 0.12)**2)
    T_c = np.maximum(T_c, 0)
    T_c[X < 0.05] = 0
    T_c[X > 0.27] = 0

    # Phase
    phase = np.zeros_like(X)
    phase[X < 0.05] = 0  # AFM
    phase[(X >= 0.05) & (T < T_c)] = 1  # SC
    phase[(X >= 0.05) & (T >= T_c)] = 2  # Strange metal

    colors_map = ['green', 'blue', 'red']
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(colors_map)

    ax4.pcolormesh(X, T, phase, cmap=cmap, alpha=0.3, shading='auto')

    # SC dome boundary
    x_sc = np.linspace(0.05, 0.27, 100)
    T_c_curve = T_c_max * (1 - ((x_sc - x_opt) / 0.12)**2)
    T_c_curve = np.maximum(T_c_curve, 0)
    ax4.plot(x_sc, T_c_curve, 'b-', lw=2)

    # AFM boundary
    ax4.axvline(x=0.05, ymin=0, ymax=0.5, color='green', lw=2)

    ax4.set_xlabel('Hole doping x')
    ax4.set_ylabel('Temperature (arb. units)')
    ax4.set_title('Cuprate Phase Diagram (Schematic)')

    # Labels
    ax4.text(0.02, 0.3, 'AFM', fontsize=12, ha='center', color='darkgreen')
    ax4.text(0.16, 0.1, 'SC', fontsize=12, ha='center', color='darkblue')
    ax4.text(0.16, 0.5, 'Strange\nMetal', fontsize=12, ha='center', color='darkred')

    ax4.set_xlim(0, 0.35)
    ax4.set_ylim(0, 0.8)

    plt.suptitle('t-J Model: Strongly Correlated Electrons\n'
                 r'$H = -t\sum_{\langle ij\rangle\sigma} \tilde{c}^\dagger_{i\sigma}\tilde{c}_{j\sigma} + J\sum_{\langle ij\rangle}(\mathbf{S}_i \cdot \mathbf{S}_j - n_in_j/4)$',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 't_j_model.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 't_j_model.png')}")


if __name__ == "__main__":
    main()
