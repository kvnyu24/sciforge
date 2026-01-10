"""
Experiment 173: Exchange Energy for Two Electrons

Demonstrates the calculation of direct (Coulomb) and exchange integrals
for two-electron systems, such as the helium atom.

Physics:
    For two electrons in orbitals psi_a and psi_b with Coulomb interaction:

    Total energy includes:
    1. Direct (Coulomb) integral J_ab:
       J = integral psi_a*(r1) psi_b*(r2) (e^2/|r1-r2|) psi_a(r1) psi_b(r2) dr1 dr2

    2. Exchange integral K_ab:
       K = integral psi_a*(r1) psi_b*(r2) (e^2/|r1-r2|) psi_b(r1) psi_a(r2) dr1 dr2

    For singlet (S=0, antisymmetric spin, symmetric spatial):
       E_singlet = E_a + E_b + J + K

    For triplet (S=1, symmetric spin, antisymmetric spatial):
       E_triplet = E_a + E_b + J - K

    Exchange energy: Delta_E = E_singlet - E_triplet = 2K

    Hund's rule: Triplet is lower energy when K > 0 (usually true)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad, nquad


# Hydrogen-like orbitals (1D model for simplicity)
def psi_1s(r, Z=1, a0=1):
    """1s orbital: psi = sqrt(Z^3/(pi*a0^3)) * exp(-Z*r/a0)"""
    return np.sqrt(Z**3 / (np.pi * a0**3)) * np.exp(-Z * np.abs(r) / a0)


def psi_2s(r, Z=1, a0=1):
    """2s orbital: psi = (Z/a0)^(3/2) / (4*sqrt(2*pi)) * (2 - Z*r/a0) * exp(-Z*r/(2*a0))"""
    rho = Z * np.abs(r) / a0
    norm = (Z / a0)**(1.5) / (4 * np.sqrt(2 * np.pi))
    return norm * (2 - rho) * np.exp(-rho / 2)


# 1D model wavefunctions for particle in a box
def box_orbital(x, n, L=1.0):
    """Particle in a box eigenfunction."""
    if x < 0 or x > L:
        return 0.0
    return np.sqrt(2/L) * np.sin(n * np.pi * x / L)


def box_orbital_arr(x, n, L=1.0):
    """Array version of box orbital."""
    result = np.sqrt(2/L) * np.sin(n * np.pi * x / L)
    result[(x < 0) | (x > L)] = 0
    return result


def coulomb_1d(r1, r2, softening=0.01):
    """
    Softened 1D Coulomb interaction.

    V(r1, r2) = 1 / sqrt((r1 - r2)^2 + epsilon^2)

    Args:
        r1, r2: Positions
        softening: Regularization parameter

    Returns:
        Interaction energy
    """
    return 1.0 / np.sqrt((r1 - r2)**2 + softening**2)


def direct_integral_1d(psi_a_func, psi_b_func, L, N=100, softening=0.01):
    """
    Calculate direct (Coulomb) integral J_ab in 1D.

    J = integral |psi_a(r1)|^2 V(r1,r2) |psi_b(r2)|^2 dr1 dr2

    Args:
        psi_a_func, psi_b_func: Orbital functions
        L: Integration range [0, L]
        N: Number of grid points
        softening: Coulomb softening

    Returns:
        Direct integral J
    """
    x1 = np.linspace(0.001, L-0.001, N)
    x2 = np.linspace(0.001, L-0.001, N)
    dx = x1[1] - x1[0]

    J = 0.0
    for i, r1 in enumerate(x1):
        psi_a_1 = psi_a_func(r1)
        for j, r2 in enumerate(x2):
            psi_b_2 = psi_b_func(r2)
            V = coulomb_1d(r1, r2, softening)
            J += np.abs(psi_a_1)**2 * V * np.abs(psi_b_2)**2

    return J * dx**2


def exchange_integral_1d(psi_a_func, psi_b_func, L, N=100, softening=0.01):
    """
    Calculate exchange integral K_ab in 1D.

    K = integral psi_a*(r1) psi_b*(r2) V(r1,r2) psi_b(r1) psi_a(r2) dr1 dr2

    Args:
        psi_a_func, psi_b_func: Orbital functions
        L: Integration range
        N: Number of grid points
        softening: Coulomb softening

    Returns:
        Exchange integral K
    """
    x1 = np.linspace(0.001, L-0.001, N)
    x2 = np.linspace(0.001, L-0.001, N)
    dx = x1[1] - x1[0]

    K = 0.0
    for i, r1 in enumerate(x1):
        psi_a_1 = psi_a_func(r1)
        psi_b_1 = psi_b_func(r1)
        for j, r2 in enumerate(x2):
            psi_a_2 = psi_a_func(r2)
            psi_b_2 = psi_b_func(r2)
            V = coulomb_1d(r1, r2, softening)
            K += np.conj(psi_a_1) * np.conj(psi_b_2) * V * psi_b_1 * psi_a_2

    return np.real(K) * dx**2


def two_electron_energies(E_a, E_b, J, K):
    """
    Calculate singlet and triplet energies.

    E_singlet = E_a + E_b + J + K (symmetric spatial)
    E_triplet = E_a + E_b + J - K (antisymmetric spatial)

    Args:
        E_a, E_b: Single-particle energies
        J: Direct integral
        K: Exchange integral

    Returns:
        (E_singlet, E_triplet)
    """
    E_singlet = E_a + E_b + J + K
    E_triplet = E_a + E_b + J - K
    return E_singlet, E_triplet


def main():
    L = 1.0  # Box length
    N_grid = 80  # Grid points for integration
    softening = 0.02

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Define orbital functions
    def psi_1(x): return box_orbital(x, 1, L)
    def psi_2(x): return box_orbital(x, 2, L)
    def psi_3(x): return box_orbital(x, 3, L)

    # Single-particle energies (particle in box, hbar^2/(2m) = 1)
    E_1 = (np.pi / L)**2 / 2
    E_2 = (2 * np.pi / L)**2 / 2
    E_3 = (3 * np.pi / L)**2 / 2

    # ===== Plot 1: Orbitals =====
    ax1 = axes[0, 0]
    x = np.linspace(0, L, 200)

    for n, (E, color) in enumerate([(E_1, 'C0'), (E_2, 'C1'), (E_3, 'C2')], 1):
        psi = box_orbital_arr(x, n, L)
        ax1.plot(x, psi, color=color, lw=2, label=f'n={n}, E={E:.2f}')
        ax1.fill_between(x, 0, psi, alpha=0.2, color=color)

    ax1.set_xlabel('Position x')
    ax1.set_ylabel('Wavefunction psi(x)')
    ax1.set_title('Single-Particle Orbitals\n(Particle in a Box)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='gray', linestyle='-', alpha=0.3)

    # ===== Plot 2: Direct and Exchange integrals vs orbital combinations =====
    ax2 = axes[0, 1]

    orbital_pairs = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]
    pair_labels = ['(1,1)', '(1,2)', '(1,3)', '(2,2)', '(2,3)', '(3,3)']

    J_vals = []
    K_vals = []

    print("Computing direct and exchange integrals...")
    for n_a, n_b in orbital_pairs:
        psi_a = lambda x, n=n_a: box_orbital(x, n, L)
        psi_b = lambda x, n=n_b: box_orbital(x, n, L)

        J = direct_integral_1d(psi_a, psi_b, L, N_grid, softening)
        K = exchange_integral_1d(psi_a, psi_b, L, N_grid, softening)

        J_vals.append(J)
        K_vals.append(K)
        print(f"  ({n_a}, {n_b}): J = {J:.4f}, K = {K:.4f}")

    x_pos = np.arange(len(orbital_pairs))
    width = 0.35

    ax2.bar(x_pos - width/2, J_vals, width, label='Direct J', color='blue', alpha=0.7)
    ax2.bar(x_pos + width/2, K_vals, width, label='Exchange K', color='red', alpha=0.7)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(pair_labels)
    ax2.set_xlabel('Orbital Pair (n_a, n_b)')
    ax2.set_ylabel('Integral Value')
    ax2.set_title('Direct (J) and Exchange (K) Integrals\n(1D Coulomb interaction)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # ===== Plot 3: Singlet vs Triplet energies =====
    ax3 = axes[1, 0]

    # Calculate energies for different configurations
    configs = [(1, 2), (1, 3), (2, 3)]
    config_labels = []
    E_singlets = []
    E_triplets = []

    for n_a, n_b in configs:
        idx = orbital_pairs.index((n_a, n_b))
        J = J_vals[idx]
        K = K_vals[idx]

        E_a = (n_a * np.pi / L)**2 / 2
        E_b = (n_b * np.pi / L)**2 / 2

        E_s, E_t = two_electron_energies(E_a, E_b, J, K)
        E_singlets.append(E_s)
        E_triplets.append(E_t)
        config_labels.append(f'({n_a},{n_b})')

    x_pos = np.arange(len(configs))

    ax3.bar(x_pos - width/2, E_singlets, width, label='Singlet (S=0)', color='blue', alpha=0.7)
    ax3.bar(x_pos + width/2, E_triplets, width, label='Triplet (S=1)', color='red', alpha=0.7)

    # Add exchange splitting lines
    for i, (E_s, E_t) in enumerate(zip(E_singlets, E_triplets)):
        ax3.annotate('', xy=(i+width/2, E_t), xytext=(i-width/2, E_s),
                     arrowprops=dict(arrowstyle='<->', color='green', lw=2))
        ax3.text(i, (E_s + E_t)/2 + 0.5, f'2K={E_s-E_t:.2f}', ha='center', fontsize=9, color='green')

    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(config_labels)
    ax3.set_xlabel('Configuration (n_a, n_b)')
    ax3.set_ylabel('Total Energy')
    ax3.set_title("Singlet vs Triplet Energies\n(Hund's Rule: Triplet Lower)")
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # ===== Plot 4: Exchange interaction vs separation =====
    ax4 = axes[1, 1]

    # For (1,2) configuration, show exchange integrand
    x_vals = np.linspace(0.05, L-0.05, 50)
    K_density = np.zeros((len(x_vals), len(x_vals)))

    for i, r1 in enumerate(x_vals):
        psi_1_r1 = box_orbital(r1, 1, L)
        psi_2_r1 = box_orbital(r1, 2, L)
        for j, r2 in enumerate(x_vals):
            psi_1_r2 = box_orbital(r2, 1, L)
            psi_2_r2 = box_orbital(r2, 2, L)
            V = coulomb_1d(r1, r2, softening)
            K_density[j, i] = np.conj(psi_1_r1) * np.conj(psi_2_r2) * V * psi_2_r1 * psi_1_r2

    im = ax4.imshow(K_density, extent=[0, L, 0, L], origin='lower', cmap='RdBu_r',
                    vmin=-np.max(np.abs(K_density)), vmax=np.max(np.abs(K_density)))
    ax4.set_xlabel('r_1')
    ax4.set_ylabel('r_2')
    ax4.set_title('Exchange Integrand K(r_1, r_2)\n(orbitals n=1 and n=2)')
    ax4.plot([0, L], [0, L], 'k--', alpha=0.5, label='r_1 = r_2')
    plt.colorbar(im, ax=ax4, label='Integrand value')

    plt.suptitle('Exchange Energy in Two-Electron Systems\n'
                 'Direct (J) and Exchange (K) Integrals, Singlet-Triplet Splitting',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'exchange_energy.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'exchange_energy.png')}")

    # Print summary
    print("\n=== Exchange Energy Results ===")
    print(f"\n1D Particle in Box, L = {L}")
    print(f"Coulomb softening = {softening}")

    print(f"\nDirect (J) and Exchange (K) integrals:")
    for (n_a, n_b), J, K in zip(orbital_pairs, J_vals, K_vals):
        print(f"  ({n_a}, {n_b}): J = {J:.4f}, K = {K:.4f}")

    print(f"\nTwo-electron energies (configurations):")
    for (n_a, n_b), E_s, E_t in zip(configs, E_singlets, E_triplets):
        print(f"  ({n_a}, {n_b}): E_singlet = {E_s:.4f}, E_triplet = {E_t:.4f}")
        print(f"           Exchange splitting = {E_s - E_t:.4f}")

    print(f"\nHund's Rule verification:")
    for (n_a, n_b), E_s, E_t in zip(configs, E_singlets, E_triplets):
        if E_t < E_s:
            print(f"  ({n_a}, {n_b}): Triplet lower - Hund's rule satisfied")
        else:
            print(f"  ({n_a}, {n_b}): Singlet lower - Hund's rule violated")


if __name__ == "__main__":
    main()
