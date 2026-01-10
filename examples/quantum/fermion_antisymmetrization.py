"""
Experiment 172: Two-Fermion Antisymmetrization

Demonstrates the antisymmetrization of two-particle wavefunctions for
identical fermions and the consequences for quantum statistics.

Physics:
    For identical fermions, the total wavefunction must be antisymmetric
    under particle exchange:

    Psi(r1, r2) = -Psi(r2, r1)

    For two particles in states |a> and |b>:
    |Psi_A> = (1/sqrt(2)) * (|a>_1 |b>_2 - |b>_1 |a>_2)

    This is the Slater determinant for two particles.

    Key consequences:
    1. Pauli exclusion: If a = b, Psi_A = 0 (no two fermions in same state)
    2. Exchange hole: Reduced probability of finding fermions close together
    3. Fermi-Dirac statistics at finite temperature
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def single_particle_wavefunction(x, n, L=1.0):
    """
    Particle in a box eigenfunction.

    psi_n(x) = sqrt(2/L) * sin(n*pi*x/L)

    Args:
        x: Position
        n: Quantum number (1, 2, 3, ...)
        L: Box length

    Returns:
        Wavefunction value
    """
    return np.sqrt(2/L) * np.sin(n * np.pi * x / L)


def antisymmetrized_wavefunction(x1, x2, n1, n2, L=1.0):
    """
    Antisymmetrized two-particle wavefunction (spatial part).

    Psi_A(x1, x2) = (1/sqrt(2)) * [psi_n1(x1)*psi_n2(x2) - psi_n1(x2)*psi_n2(x1)]

    This is the Slater determinant:
    Psi_A = (1/sqrt(2)) * |psi_n1(x1)  psi_n2(x1)|
                         |psi_n1(x2)  psi_n2(x2)|

    Args:
        x1, x2: Positions of particles 1 and 2
        n1, n2: Quantum numbers
        L: Box length

    Returns:
        Antisymmetrized wavefunction
    """
    psi1_x1 = single_particle_wavefunction(x1, n1, L)
    psi2_x2 = single_particle_wavefunction(x2, n2, L)
    psi1_x2 = single_particle_wavefunction(x2, n1, L)
    psi2_x1 = single_particle_wavefunction(x1, n2, L)

    return (psi1_x1 * psi2_x2 - psi1_x2 * psi2_x1) / np.sqrt(2)


def symmetrized_wavefunction(x1, x2, n1, n2, L=1.0):
    """
    Symmetrized two-particle wavefunction (for bosons).

    Psi_S(x1, x2) = (1/sqrt(2)) * [psi_n1(x1)*psi_n2(x2) + psi_n1(x2)*psi_n2(x1)]

    Args:
        x1, x2: Positions
        n1, n2: Quantum numbers
        L: Box length

    Returns:
        Symmetrized wavefunction
    """
    psi1_x1 = single_particle_wavefunction(x1, n1, L)
    psi2_x2 = single_particle_wavefunction(x2, n2, L)
    psi1_x2 = single_particle_wavefunction(x2, n1, L)
    psi2_x1 = single_particle_wavefunction(x1, n2, L)

    # Normalize properly for n1 = n2 case
    if n1 == n2:
        return psi1_x1 * psi2_x2
    else:
        return (psi1_x1 * psi2_x2 + psi1_x2 * psi2_x1) / np.sqrt(2)


def distinguishable_wavefunction(x1, x2, n1, n2, L=1.0):
    """
    Product wavefunction for distinguishable particles.

    Psi(x1, x2) = psi_n1(x1) * psi_n2(x2)

    Args:
        x1, x2: Positions
        n1, n2: Quantum numbers
        L: Box length

    Returns:
        Product wavefunction
    """
    return single_particle_wavefunction(x1, n1, L) * single_particle_wavefunction(x2, n2, L)


def pair_correlation(x1_grid, x2_grid, wavefunction_func, n1, n2, L=1.0):
    """
    Compute pair correlation function g(x1, x2) = |Psi(x1, x2)|^2.

    Args:
        x1_grid, x2_grid: Position grids
        wavefunction_func: Function computing wavefunction
        n1, n2: Quantum numbers
        L: Box length

    Returns:
        2D probability density
    """
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    Psi = wavefunction_func(X1, X2, n1, n2, L)
    return np.abs(Psi)**2


def diagonal_correlation(x, wavefunction_func, n1, n2, L=1.0):
    """
    Probability density along the diagonal x1 = x2 (exchange hole).

    Args:
        x: Position array
        wavefunction_func: Wavefunction function
        n1, n2: Quantum numbers
        L: Box length

    Returns:
        Probability density |Psi(x, x)|^2
    """
    Psi_diag = wavefunction_func(x, x, n1, n2, L)
    return np.abs(Psi_diag)**2


def main():
    L = 1.0  # Box length
    N = 100  # Grid points
    x = np.linspace(0, L, N)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # States to use
    n1, n2 = 1, 2  # Ground and first excited state

    # ===== Row 1: 2D probability distributions =====

    # Distinguishable particles
    ax1 = axes[0, 0]
    P_dist = pair_correlation(x, x, distinguishable_wavefunction, n1, n2, L)
    im1 = ax1.imshow(P_dist, extent=[0, L, 0, L], origin='lower', cmap='hot')
    ax1.set_xlabel('x_1')
    ax1.set_ylabel('x_2')
    ax1.set_title('Distinguishable Particles\n|psi_1(x_1) psi_2(x_2)|^2')
    ax1.plot([0, L], [0, L], 'w--', alpha=0.5, label='x_1 = x_2')
    plt.colorbar(im1, ax=ax1)

    # Bosons (symmetric)
    ax2 = axes[0, 1]
    P_boson = pair_correlation(x, x, symmetrized_wavefunction, n1, n2, L)
    im2 = ax2.imshow(P_boson, extent=[0, L, 0, L], origin='lower', cmap='hot')
    ax2.set_xlabel('x_1')
    ax2.set_ylabel('x_2')
    ax2.set_title('Bosons (Symmetric)\n|Psi_S(x_1, x_2)|^2')
    ax2.plot([0, L], [0, L], 'w--', alpha=0.5)
    plt.colorbar(im2, ax=ax2)

    # Fermions (antisymmetric)
    ax3 = axes[0, 2]
    P_fermion = pair_correlation(x, x, antisymmetrized_wavefunction, n1, n2, L)
    im3 = ax3.imshow(P_fermion, extent=[0, L, 0, L], origin='lower', cmap='hot')
    ax3.set_xlabel('x_1')
    ax3.set_ylabel('x_2')
    ax3.set_title('Fermions (Antisymmetric)\n|Psi_A(x_1, x_2)|^2')
    ax3.plot([0, L], [0, L], 'w--', alpha=0.5, label='Exchange hole')
    plt.colorbar(im3, ax=ax3)

    # ===== Row 2: Diagonal cuts and analysis =====

    # Exchange hole along diagonal
    ax4 = axes[1, 0]
    P_diag_dist = diagonal_correlation(x, distinguishable_wavefunction, n1, n2, L)
    P_diag_boson = diagonal_correlation(x, symmetrized_wavefunction, n1, n2, L)
    P_diag_fermion = diagonal_correlation(x, antisymmetrized_wavefunction, n1, n2, L)

    ax4.plot(x, P_diag_dist, 'g-', lw=2, label='Distinguishable')
    ax4.plot(x, P_diag_boson, 'b-', lw=2, label='Bosons')
    ax4.plot(x, P_diag_fermion, 'r-', lw=2, label='Fermions')

    ax4.set_xlabel('Position x')
    ax4.set_ylabel('|Psi(x, x)|^2')
    ax4.set_title('Diagonal Slice (x_1 = x_2)\nExchange Hole for Fermions')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Relative coordinate |x1 - x2| distribution
    ax5 = axes[1, 1]

    dx = x[1] - x[0]
    r_vals = np.linspace(0, L/2, 50)
    P_r_dist = np.zeros_like(r_vals)
    P_r_boson = np.zeros_like(r_vals)
    P_r_fermion = np.zeros_like(r_vals)

    for i, x1 in enumerate(x):
        for j, x2 in enumerate(x):
            r = abs(x1 - x2)
            if r < L/2:
                r_idx = int(r / (L/2) * (len(r_vals) - 1))
                if r_idx < len(r_vals):
                    P_r_dist[r_idx] += pair_correlation([x1], [x2], distinguishable_wavefunction, n1, n2, L)[0,0]
                    P_r_boson[r_idx] += pair_correlation([x1], [x2], symmetrized_wavefunction, n1, n2, L)[0,0]
                    P_r_fermion[r_idx] += pair_correlation([x1], [x2], antisymmetrized_wavefunction, n1, n2, L)[0,0]

    # Normalize
    P_r_dist /= np.sum(P_r_dist) * (r_vals[1] - r_vals[0])
    P_r_boson /= np.sum(P_r_boson) * (r_vals[1] - r_vals[0])
    P_r_fermion /= np.sum(P_r_fermion) * (r_vals[1] - r_vals[0])

    ax5.plot(r_vals, P_r_dist, 'g-', lw=2, label='Distinguishable')
    ax5.plot(r_vals, P_r_boson, 'b-', lw=2, label='Bosons')
    ax5.plot(r_vals, P_r_fermion, 'r-', lw=2, label='Fermions')

    ax5.set_xlabel('Separation |x_1 - x_2|')
    ax5.set_ylabel('Probability P(r)')
    ax5.set_title('Pair Separation Distribution\n(Bosons bunch, Fermions avoid)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Pauli exclusion: same state
    ax6 = axes[1, 2]

    # Try to put both particles in state n=1
    n_same = 1
    P_fermion_same = pair_correlation(x, x, antisymmetrized_wavefunction, n_same, n_same, L)

    ax6.imshow(P_fermion_same, extent=[0, L, 0, L], origin='lower', cmap='hot')
    ax6.set_xlabel('x_1')
    ax6.set_ylabel('x_2')
    ax6.set_title('Fermions: Both in n=1 State\n(Pauli Exclusion: |Psi|^2 = 0)')

    # Add text explaining Pauli exclusion
    max_val = np.max(P_fermion_same)
    ax6.text(0.5, 0.5, f'Max |Psi|^2 = {max_val:.2e}',
             transform=ax6.transAxes, fontsize=12, color='white',
             ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

    plt.suptitle('Two-Fermion Antisymmetrization and Quantum Statistics\n'
                 f'States: n_1 = {n1}, n_2 = {n2} (except rightmost: n_1 = n_2 = 1)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'fermion_antisymmetrization.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'fermion_antisymmetrization.png')}")

    # Print numerical analysis
    print("\n=== Two-Fermion Antisymmetrization Results ===")
    print(f"\nParticle in a box, L = {L}")
    print(f"Quantum numbers: n1 = {n1}, n2 = {n2}")

    # Check antisymmetry
    x_test = 0.3
    y_test = 0.7
    Psi_xy = antisymmetrized_wavefunction(x_test, y_test, n1, n2, L)
    Psi_yx = antisymmetrized_wavefunction(y_test, x_test, n1, n2, L)
    print(f"\nAntisymmetry verification:")
    print(f"  Psi({x_test}, {y_test}) = {Psi_xy:.6f}")
    print(f"  Psi({y_test}, {x_test}) = {Psi_yx:.6f}")
    print(f"  Sum = {Psi_xy + Psi_yx:.2e} (should be ~0)")

    # Exchange hole
    x_diag = 0.5
    P_dist_diag = np.abs(distinguishable_wavefunction(x_diag, x_diag, n1, n2, L))**2
    P_ferm_diag = np.abs(antisymmetrized_wavefunction(x_diag, x_diag, n1, n2, L))**2
    print(f"\nExchange hole at x_1 = x_2 = {x_diag}:")
    print(f"  Distinguishable: |Psi|^2 = {P_dist_diag:.6f}")
    print(f"  Fermions: |Psi|^2 = {P_ferm_diag:.2e}")

    # Pauli exclusion
    P_same_max = np.max(pair_correlation(x, x, antisymmetrized_wavefunction, 1, 1, L))
    print(f"\nPauli Exclusion (both in n=1):")
    print(f"  Max |Psi_A|^2 = {P_same_max:.2e}")


if __name__ == "__main__":
    main()
