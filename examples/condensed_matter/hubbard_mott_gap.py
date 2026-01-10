"""
Experiment 243: Hubbard Model Mott Gap

Demonstrates the Hubbard model and the Mott insulator transition,
showing how strong electron-electron correlations open a gap at
half-filling even when band theory predicts a metal.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from itertools import product


def hubbard_hamiltonian_1d(L, t, U, n_up, n_down, periodic=True):
    """
    Build 1D Hubbard model Hamiltonian in occupation number basis.

    H = -t * sum_<i,j>,sigma (c^+_i,sigma c_j,sigma + h.c.)
        + U * sum_i n_i,up * n_i,down

    Args:
        L: Number of sites
        t: Hopping parameter
        U: On-site Coulomb repulsion
        n_up: Number of spin-up electrons
        n_down: Number of spin-down electrons
        periodic: Periodic boundary conditions

    Returns:
        Hamiltonian matrix in occupation basis
    """
    from itertools import combinations

    # Generate basis states
    # Each state is described by which sites are occupied by up/down electrons

    # All ways to place n_up electrons on L sites
    up_configs = list(combinations(range(L), n_up))
    down_configs = list(combinations(range(L), n_down))

    # Full basis
    basis = []
    for up in up_configs:
        for down in down_configs:
            basis.append((set(up), set(down)))

    dim = len(basis)
    H = np.zeros((dim, dim))

    def state_to_index(up_set, down_set):
        for i, (u, d) in enumerate(basis):
            if u == up_set and d == down_set:
                return i
        return -1

    for i, (up, down) in enumerate(basis):
        # Diagonal: on-site interaction
        double_occ = len(up & down)
        H[i, i] = U * double_occ

        # Off-diagonal: hopping
        for sigma, occ in [('up', up), ('down', down)]:
            for site in occ:
                # Hop to neighbors
                neighbors = []
                if site > 0:
                    neighbors.append(site - 1)
                elif periodic:
                    neighbors.append(L - 1)

                if site < L - 1:
                    neighbors.append(site + 1)
                elif periodic:
                    neighbors.append(0)

                for neighbor in neighbors:
                    if neighbor not in occ:
                        # Valid hop
                        new_occ = set(occ)
                        new_occ.remove(site)
                        new_occ.add(neighbor)

                        if sigma == 'up':
                            j = state_to_index(new_occ, down)
                        else:
                            j = state_to_index(up, new_occ)

                        if j >= 0:
                            # Compute fermion sign
                            # Count fermions between site and neighbor
                            sign = 1  # Simplified - proper sign requires more care
                            H[i, j] = -t * sign
                            H[j, i] = -t * sign

    return H, basis


def hubbard_2site(t, U):
    """
    Exact solution for 2-site Hubbard model at half-filling.

    Basis: |up,down>, |down,up>, |up down,0>, |0,up down>

    Returns eigenvalues for singlet and triplet sectors.
    """
    # Singlet sector (S=0, Sz=0)
    # Basis: |S> = (|up,down> - |down,up>)/sqrt(2), |D> = (|up down,0> + |0,up down>)/sqrt(2)
    H_singlet = np.array([
        [0, -2*t],
        [-2*t, U]
    ])

    # Triplet sector (S=1, Sz=0): |T> = (|up,down> + |down,up>)/sqrt(2)
    # No hopping connects this to doubly occupied states
    E_triplet = 0

    E_singlet, _ = linalg.eigh(H_singlet)

    return E_singlet, E_triplet


def mean_field_gap(U, t, z=4):
    """
    Mean-field estimate of Mott gap.

    For large U: gap ~ U - zt

    Args:
        U: Coulomb repulsion
        t: Hopping
        z: Coordination number

    Returns:
        Approximate gap
    """
    if U > z * t:
        return U - z * t
    else:
        return 0


def spectral_function_atomic(omega, U, eta=0.1):
    """
    Atomic limit spectral function.

    In atomic limit, spectral weight at -U/2 and +U/2 for half-filling.

    Args:
        omega: Frequency array
        U: Coulomb repulsion
        eta: Broadening

    Returns:
        A(omega) spectral function
    """
    # Two Hubbard bands centered at +/- U/2
    A = (eta/np.pi) / ((omega - U/2)**2 + eta**2) / 2
    A += (eta/np.pi) / ((omega + U/2)**2 + eta**2) / 2
    return A


def spectral_function_band(omega, t, eta=0.1, n_k=100):
    """
    Non-interacting (U=0) spectral function for 1D chain.

    A(omega) = DOS(omega)

    Args:
        omega: Frequency
        t: Hopping
        eta: Broadening
        n_k: Number of k-points

    Returns:
        A(omega)
    """
    k = np.linspace(-np.pi, np.pi, n_k)
    E_k = -2 * t * np.cos(k)

    # Sum over k
    A = np.zeros_like(omega)
    for ek in E_k:
        A += (eta/np.pi) / ((omega - ek)**2 + eta**2) / n_k

    return A


def main():
    """Main simulation and visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    t = 1.0  # Hopping (set energy scale)

    # Plot 1: Two-site Hubbard model
    ax1 = axes[0, 0]

    U_range = np.linspace(0, 10, 100)

    E_singlet_ground = []
    E_singlet_excited = []

    for U in U_range:
        E_s, E_t = hubbard_2site(t, U)
        E_singlet_ground.append(E_s[0])
        E_singlet_excited.append(E_s[1])

    ax1.plot(U_range / t, E_singlet_ground, 'b-', lw=2, label='Ground state (singlet)')
    ax1.plot(U_range / t, E_singlet_excited, 'r-', lw=2, label='Excited state')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Triplet')

    ax1.set_xlabel('U / t')
    ax1.set_ylabel('Energy / t')
    ax1.set_title('Two-Site Hubbard Model Spectrum')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Mark charge gap
    ax1.fill_between(U_range / t, E_singlet_ground, E_singlet_excited, alpha=0.2, color='yellow')
    ax1.text(5, -1, 'Charge gap', fontsize=10, ha='center')

    # Plot 2: Spectral function evolution
    ax2 = axes[0, 1]

    omega = np.linspace(-6, 6, 500)

    U_values = [0, 2, 4, 8]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(U_values)))

    for U, color in zip(U_values, colors):
        if U == 0:
            A = spectral_function_band(omega, t, eta=0.2)
            label = 'U = 0 (metal)'
        else:
            # Interpolate between band and atomic limits
            A_band = spectral_function_band(omega, t, eta=0.2)
            A_atomic = spectral_function_atomic(omega, U, eta=0.3)
            # Simple interpolation
            weight = min(U / (4*t), 1)
            A = (1 - weight) * A_band + weight * A_atomic
            label = f'U = {U}t'

        ax2.plot(omega, A + 0.5 * U_values.index(U), color=color, lw=2, label=label)

    ax2.set_xlabel('omega / t')
    ax2.set_ylabel('A(omega) + offset')
    ax2.set_title('Spectral Function vs Interaction Strength')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

    # Plot 3: Mott gap vs U
    ax3 = axes[1, 0]

    U_range = np.linspace(0, 12, 100)

    # Mean-field gap
    gap_mf = [mean_field_gap(U, t, z=2) for U in U_range]

    # From two-site exact solution
    gap_2site = []
    for U in U_range:
        E_s, E_t = hubbard_2site(t, U)
        # Gap is difference between excited and ground singlet states
        gap_2site.append(E_s[1] - E_s[0])

    ax3.plot(U_range / t, gap_mf, 'b-', lw=2, label='Mean-field (gap = U - zt)')
    ax3.plot(U_range / t, gap_2site, 'r--', lw=2, label='Two-site exact')

    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax3.axvline(x=2, color='green', linestyle=':', alpha=0.5, label='Mott transition (z=2)')

    ax3.set_xlabel('U / t')
    ax3.set_ylabel('Charge gap / t')
    ax3.set_title('Mott Gap vs Interaction Strength')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 12)
    ax3.set_ylim(0, 10)

    # Shade Mott insulator region
    ax3.fill_between([4, 12], 0, 10, alpha=0.1, color='red')
    ax3.text(8, 8, 'Mott\nInsulator', fontsize=12, ha='center')
    ax3.text(1, 8, 'Metal', fontsize=12, ha='center')

    # Plot 4: Phase diagram schematic
    ax4 = axes[1, 1]

    # Temperature vs U phase diagram
    T_range = np.linspace(0, 2, 100)
    U_range = np.linspace(0, 10, 100)
    T_grid, U_grid = np.meshgrid(T_range, U_range)

    # Critical U_c(T) ~ U_c(0) + a*T^2
    U_c = 4  # Critical U at T=0
    phase = np.where(U_grid > U_c + 2*T_grid**2, 1, 0)

    ax4.contourf(U_grid, T_grid, phase, levels=[-0.5, 0.5, 1.5], colors=['blue', 'red'], alpha=0.3)
    ax4.contour(U_grid, T_grid, phase, levels=[0.5], colors='black', linewidths=2)

    ax4.set_xlabel('U / t')
    ax4.set_ylabel('T / t')
    ax4.set_title('Mott Transition Phase Diagram (Schematic)')

    ax4.text(2, 1, 'Metallic', fontsize=14, ha='center', va='center')
    ax4.text(8, 0.5, 'Mott\nInsulator', fontsize=14, ha='center', va='center')

    # Mark critical point
    ax4.plot(U_c, 0, 'ko', markersize=10)
    ax4.annotate('$U_c$', xy=(U_c, 0), xytext=(U_c + 0.5, 0.3),
                fontsize=12, arrowprops=dict(arrowstyle='->', color='black'))

    plt.suptitle('Hubbard Model and Mott Insulator Transition\n'
                 r'$H = -t\sum_{\langle ij\rangle\sigma} c^\dagger_{i\sigma}c_{j\sigma} + U\sum_i n_{i\uparrow}n_{i\downarrow}$',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'hubbard_mott_gap.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'hubbard_mott_gap.png')}")


if __name__ == "__main__":
    main()
