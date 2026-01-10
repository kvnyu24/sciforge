"""
Experiment 174: Tight-Binding Chain

Demonstrates the tight-binding model for electronic structure in 1D lattices,
showing band formation, density of states, and edge states.

Physics:
    The tight-binding model describes electrons hopping between atomic sites:

    H = -t * sum_<i,j> (c_i^dag c_j + h.c.) + sum_i epsilon_i * c_i^dag c_i

    For a 1D chain with N sites and nearest-neighbor hopping:
    - Energy bands: E(k) = -2t * cos(k*a), where a is lattice constant
    - Band width: W = 4t
    - Density of states: diverges at band edges (van Hove singularities)

    Extensions:
    - SSH model: alternating hopping t1, t2 -> topological edge states
    - Anderson localization: random on-site energies
    - Periodic vs open boundary conditions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh


def tight_binding_hamiltonian(N, t=1.0, epsilon=0.0, periodic=True):
    """
    Construct tight-binding Hamiltonian for 1D chain.

    H = -t * sum_<i,j> |i><j| + epsilon * sum_i |i><i|

    Args:
        N: Number of sites
        t: Hopping parameter
        epsilon: On-site energy (scalar or array)
        periodic: Use periodic boundary conditions

    Returns:
        N x N Hamiltonian matrix
    """
    H = np.zeros((N, N))

    # On-site energies
    if np.isscalar(epsilon):
        np.fill_diagonal(H, epsilon)
    else:
        np.fill_diagonal(H, epsilon)

    # Nearest-neighbor hopping
    for i in range(N - 1):
        H[i, i+1] = -t
        H[i+1, i] = -t

    # Periodic boundary conditions
    if periodic:
        H[0, N-1] = -t
        H[N-1, 0] = -t

    return H


def ssh_hamiltonian(N, t1=1.0, t2=0.5, periodic=False):
    """
    SSH (Su-Schrieffer-Heeger) model Hamiltonian.

    Alternating hopping strengths t1 and t2.
    Topological phase for t1 < t2 (edge states appear).

    Args:
        N: Number of unit cells (total sites = 2N)
        t1: Intra-cell hopping
        t2: Inter-cell hopping
        periodic: Periodic boundary conditions

    Returns:
        2N x 2N Hamiltonian
    """
    dim = 2 * N
    H = np.zeros((dim, dim))

    for i in range(N):
        # Intra-cell hopping (A to B within unit cell i)
        a = 2 * i
        b = 2 * i + 1
        H[a, b] = -t1
        H[b, a] = -t1

        # Inter-cell hopping (B of cell i to A of cell i+1)
        if i < N - 1:
            H[b, b+1] = -t2
            H[b+1, b] = -t2

    # Periodic boundary conditions
    if periodic and N > 1:
        H[dim-1, 0] = -t2
        H[0, dim-1] = -t2

    return H


def dispersion_relation(k, t=1.0, a=1.0):
    """
    Analytical dispersion for 1D tight-binding chain.

    E(k) = -2t * cos(k*a)

    Args:
        k: Wave vector
        t: Hopping parameter
        a: Lattice constant

    Returns:
        Energy
    """
    return -2 * t * np.cos(k * a)


def density_of_states(energies, E_grid, broadening=0.05):
    """
    Calculate density of states from eigenvalues.

    Uses Lorentzian broadening.

    Args:
        energies: Array of eigenvalues
        E_grid: Energy grid for DOS
        broadening: Lorentzian width

    Returns:
        DOS on E_grid
    """
    dos = np.zeros_like(E_grid)
    for E in energies:
        dos += broadening / (np.pi * ((E_grid - E)**2 + broadening**2))
    return dos / len(energies)


def analytical_dos_1d(E, t=1.0):
    """
    Analytical DOS for 1D tight-binding chain.

    rho(E) = 1 / (pi * sqrt(4t^2 - E^2))

    Has van Hove singularities at E = +/- 2t (band edges).

    Args:
        E: Energy
        t: Hopping

    Returns:
        Density of states
    """
    W = 2 * t  # Half bandwidth
    inside_band = np.abs(E) < W

    dos = np.zeros_like(E)
    dos[inside_band] = 1 / (np.pi * np.sqrt(W**2 - E[inside_band]**2 + 1e-10))

    return dos


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ===== Plot 1: Band structure E(k) =====
    ax1 = axes[0, 0]

    t = 1.0
    a = 1.0
    N = 100

    # Analytical dispersion
    k = np.linspace(-np.pi/a, np.pi/a, 500)
    E_k = dispersion_relation(k, t, a)

    ax1.plot(k * a / np.pi, E_k, 'b-', lw=2, label='E(k) = -2t cos(ka)')
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.axhline(2*t, color='red', linestyle=':', alpha=0.7, label='Band edge')
    ax1.axhline(-2*t, color='red', linestyle=':', alpha=0.7)

    ax1.set_xlabel(r'Wave vector $ka/\pi$')
    ax1.set_ylabel('Energy E/t')
    ax1.set_title('1D Tight-Binding Band Structure')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-1, 1)

    # Numerical eigenvalues for comparison
    H = tight_binding_hamiltonian(N, t, periodic=True)
    eigenvalues, _ = eigh(H)
    k_discrete = np.linspace(-np.pi/a, np.pi/a, N)
    ax1.scatter(k_discrete * a / np.pi, eigenvalues, c='red', s=5, alpha=0.5, label='Numerical')

    # ===== Plot 2: Density of States =====
    ax2 = axes[0, 1]

    N_large = 500
    H_large = tight_binding_hamiltonian(N_large, t, periodic=True)
    eigenvalues_large, _ = eigh(H_large)

    E_grid = np.linspace(-2.5*t, 2.5*t, 500)
    dos_numerical = density_of_states(eigenvalues_large, E_grid, broadening=0.03)
    dos_analytical = analytical_dos_1d(E_grid, t)

    ax2.plot(E_grid, dos_numerical, 'b-', lw=2, label='Numerical DOS')
    ax2.plot(E_grid, dos_analytical, 'r--', lw=2, label='Analytical')
    ax2.fill_between(E_grid, 0, dos_numerical, alpha=0.3)

    ax2.axvline(-2*t, color='gray', linestyle=':', alpha=0.5)
    ax2.axvline(2*t, color='gray', linestyle=':', alpha=0.5)
    ax2.text(2*t + 0.1, 0.5, 'Van Hove\nsingularity', fontsize=9)

    ax2.set_xlabel('Energy E/t')
    ax2.set_ylabel('Density of States')
    ax2.set_title('Density of States\n(Van Hove singularities at band edges)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-2.5*t, 2.5*t)
    ax2.set_ylim(0, 1.5)

    # ===== Plot 3: SSH Model - Edge States =====
    ax3 = axes[1, 0]

    N_ssh = 20  # Unit cells
    dim = 2 * N_ssh

    # Trivial phase: t1 > t2
    H_trivial = ssh_hamiltonian(N_ssh, t1=1.0, t2=0.5, periodic=False)
    E_trivial, psi_trivial = eigh(H_trivial)

    # Topological phase: t1 < t2
    H_topo = ssh_hamiltonian(N_ssh, t1=0.5, t2=1.0, periodic=False)
    E_topo, psi_topo = eigh(H_topo)

    # Plot energy levels
    ax3.scatter(np.ones(dim) * 0.5, E_trivial, c='blue', s=50, alpha=0.7, label='Trivial (t1 > t2)')
    ax3.scatter(np.ones(dim) * 1.5, E_topo, c='red', s=50, alpha=0.7, label='Topological (t1 < t2)')

    # Highlight edge states (near zero energy)
    edge_mask = np.abs(E_topo) < 0.1
    ax3.scatter(np.ones(np.sum(edge_mask)) * 1.5, E_topo[edge_mask],
                c='yellow', s=100, edgecolors='black', label='Edge states', zorder=5)

    ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlim(0, 2)
    ax3.set_xticks([0.5, 1.5])
    ax3.set_xticklabels(['Trivial', 'Topological'])
    ax3.set_ylabel('Energy E/t')
    ax3.set_title('SSH Model: Topological Edge States\n(Zero energy states in topological phase)')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # ===== Plot 4: Edge state wavefunction =====
    ax4 = axes[1, 1]

    site_positions = np.arange(dim)

    # Find edge states in topological phase
    edge_indices = np.where(np.abs(E_topo) < 0.1)[0]

    if len(edge_indices) >= 2:
        # Plot the two edge states
        psi_left = np.abs(psi_topo[:, edge_indices[0]])**2
        psi_right = np.abs(psi_topo[:, edge_indices[1]])**2

        ax4.bar(site_positions - 0.2, psi_left, width=0.4, alpha=0.7, label='Left edge state', color='blue')
        ax4.bar(site_positions + 0.2, psi_right, width=0.4, alpha=0.7, label='Right edge state', color='red')

    # Also show bulk state for comparison
    bulk_idx = dim // 2  # Middle of spectrum
    psi_bulk = np.abs(psi_topo[:, bulk_idx])**2
    ax4.plot(site_positions, psi_bulk, 'g-', lw=2, marker='o', markersize=3, label='Bulk state')

    ax4.set_xlabel('Site index')
    ax4.set_ylabel('Probability |psi|^2')
    ax4.set_title('SSH Edge State Wavefunctions\n(Exponentially localized at edges)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-1, dim)

    plt.suptitle('Tight-Binding Model and SSH Topological Insulator\n'
                 '1D chain with nearest-neighbor hopping',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'tight_binding.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'tight_binding.png')}")

    # Print results
    print("\n=== Tight-Binding Chain Results ===")
    print(f"\nSimple 1D chain:")
    print(f"  N = {N_large} sites, t = {t}")
    print(f"  Bandwidth = 4t = {4*t}")
    print(f"  Band center = 0")
    print(f"  Numerical energy range: [{eigenvalues_large.min():.4f}, {eigenvalues_large.max():.4f}]")

    print(f"\nSSH Model:")
    print(f"  N = {N_ssh} unit cells ({dim} sites)")
    print(f"  Trivial phase: t1 = 1.0, t2 = 0.5")
    print(f"  Topological phase: t1 = 0.5, t2 = 1.0")

    if len(edge_indices) >= 2:
        print(f"  Edge state energies: {E_topo[edge_indices]}")
        print(f"  Number of edge states: {len(edge_indices)}")

    # Band gap analysis
    E_sorted = np.sort(E_topo)
    gaps = np.diff(E_sorted)
    max_gap_idx = np.argmax(gaps)
    print(f"  Band gap: {gaps[max_gap_idx]:.4f}")


if __name__ == "__main__":
    main()
