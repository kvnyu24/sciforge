"""
Experiment 222: Finite Chain Edge States

Demonstrates the emergence of edge states in a finite tight-binding chain,
showing how breaking translational symmetry at the boundaries leads to
states localized at the edges - a precursor to topological edge states.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg


def build_tight_binding_hamiltonian(N, t, t_edge=None, on_site=None):
    """
    Build tight-binding Hamiltonian matrix for finite chain.

    Args:
        N: Number of sites
        t: Bulk hopping parameter
        t_edge: Edge hopping parameter (if different from bulk)
        on_site: On-site energies (array of length N or scalar)

    Returns:
        N x N Hamiltonian matrix
    """
    if t_edge is None:
        t_edge = t
    if on_site is None:
        on_site = np.zeros(N)
    elif np.isscalar(on_site):
        on_site = np.full(N, on_site)

    H = np.zeros((N, N))

    # On-site energies
    np.fill_diagonal(H, on_site)

    # Hopping terms
    for i in range(N - 1):
        if i == 0 or i == N - 2:
            # Edge hopping
            H[i, i+1] = -t_edge
            H[i+1, i] = -t_edge
        else:
            # Bulk hopping
            H[i, i+1] = -t
            H[i+1, i] = -t

    return H


def build_dimerized_chain(N, t1, t2):
    """
    Build SSH-like dimerized chain Hamiltonian.

    Args:
        N: Number of sites (should be even)
        t1: Intra-cell hopping
        t2: Inter-cell hopping

    Returns:
        N x N Hamiltonian matrix
    """
    H = np.zeros((N, N))

    for i in range(N - 1):
        if i % 2 == 0:
            # Intra-cell hopping
            H[i, i+1] = -t1
            H[i+1, i] = -t1
        else:
            # Inter-cell hopping
            H[i, i+1] = -t2
            H[i+1, i] = -t2

    return H


def main():
    """Main simulation and visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Parameters
    N = 50  # Number of sites
    t = 1.0  # Hopping parameter

    # Plot 1: Energy spectrum of finite chain
    ax1 = axes[0, 0]

    H = build_tight_binding_hamiltonian(N, t)
    eigenvalues, eigenvectors = linalg.eigh(H)

    ax1.plot(range(1, N+1), eigenvalues, 'bo', markersize=6)
    ax1.set_xlabel('State index')
    ax1.set_ylabel('Energy (units of t)')
    ax1.set_title(f'Energy Spectrum of Finite Chain (N = {N})')
    ax1.grid(True, alpha=0.3)

    # Add expected band edges
    ax1.axhline(y=-2*t, color='red', linestyle='--', alpha=0.7, label='Band edges (+-2t)')
    ax1.axhline(y=2*t, color='red', linestyle='--', alpha=0.7)
    ax1.legend()

    # Plot 2: Wavefunction profiles for different states
    ax2 = axes[0, 1]

    sites = np.arange(1, N+1)

    # Plot lowest, middle, and highest energy states
    states_to_plot = [0, N//4, N//2, 3*N//4, N-1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(states_to_plot)))

    for idx, color in zip(states_to_plot, colors):
        psi = eigenvectors[:, idx]
        ax2.plot(sites, psi**2 + eigenvalues[idx] * 0.1,
                color=color, lw=1.5, label=f'n={idx+1}, E={eigenvalues[idx]:.2f}')

    ax2.set_xlabel('Site index')
    ax2.set_ylabel('|psi|^2 + offset')
    ax2.set_title('Wavefunction Profiles')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Edge potential - creating edge states
    ax3 = axes[1, 0]

    # Add edge potential to create localized states
    on_site = np.zeros(N)
    on_site[0] = -1.5 * t  # Lower energy at left edge
    on_site[-1] = -1.5 * t  # Lower energy at right edge

    H_edge = build_tight_binding_hamiltonian(N, t, on_site=on_site)
    eigenvalues_edge, eigenvectors_edge = linalg.eigh(H_edge)

    # Find edge states (states below the band)
    edge_state_idx = np.where(eigenvalues_edge < -2*t)[0]

    for idx in edge_state_idx:
        psi = eigenvectors_edge[:, idx]
        ax3.plot(sites, psi**2, lw=2, label=f'Edge state E={eigenvalues_edge[idx]:.2f}')

    # Also plot a bulk state for comparison
    bulk_idx = N // 2
    psi_bulk = eigenvectors_edge[:, bulk_idx]
    ax3.plot(sites, psi_bulk**2, 'k--', lw=1.5, alpha=0.7, label='Bulk state')

    ax3.set_xlabel('Site index')
    ax3.set_ylabel('|psi|^2')
    ax3.set_title('Edge States from Edge Potential')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Inset: Energy spectrum with edge states
    ax3_inset = ax3.inset_axes([0.6, 0.6, 0.35, 0.35])
    ax3_inset.plot(range(1, N+1), eigenvalues_edge, 'bo', markersize=3)
    ax3_inset.axhline(y=-2*t, color='red', linestyle='--', alpha=0.7)
    ax3_inset.set_xlabel('n', fontsize=8)
    ax3_inset.set_ylabel('E', fontsize=8)
    ax3_inset.tick_params(labelsize=6)

    # Plot 4: Dimerized chain (SSH model precursor)
    ax4 = axes[1, 1]

    N_dimer = 40
    t1, t2 = 0.5, 1.5  # Dimerization

    H_dimer = build_dimerized_chain(N_dimer, t1, t2)
    eigenvalues_dimer, eigenvectors_dimer = linalg.eigh(H_dimer)

    # Plot spectrum
    ax4.plot(range(1, N_dimer+1), eigenvalues_dimer, 'bo', markersize=6)

    # Find states in the gap (if any)
    gap_center = 0
    gap_states = np.where(np.abs(eigenvalues_dimer) < 0.1)[0]

    if len(gap_states) > 0:
        ax4.plot([i+1 for i in gap_states], eigenvalues_dimer[gap_states],
                'ro', markersize=10, label='Gap states')

        # Plot wavefunction of gap state in inset
        ax4_inset = ax4.inset_axes([0.55, 0.55, 0.4, 0.4])
        for idx in gap_states[:2]:  # Plot up to 2 gap states
            psi = eigenvectors_dimer[:, idx]
            ax4_inset.plot(range(1, N_dimer+1), psi**2, lw=1.5)
        ax4_inset.set_xlabel('Site', fontsize=8)
        ax4_inset.set_ylabel('|psi|^2', fontsize=8)
        ax4_inset.set_title('Gap state', fontsize=8)
        ax4_inset.tick_params(labelsize=6)
        ax4.legend()

    ax4.set_xlabel('State index')
    ax4.set_ylabel('Energy (units of t)')
    ax4.set_title(f'Dimerized Chain (t1={t1}, t2={t2})')
    ax4.grid(True, alpha=0.3)

    # Mark gap
    ax4.axhspan(-abs(t2-t1), abs(t2-t1), alpha=0.2, color='yellow', label='Gap')

    plt.suptitle('Edge States in Finite Tight-Binding Chains\n'
                 'Boundary effects and localization',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'finite_chain_edge_states.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'finite_chain_edge_states.png')}")


if __name__ == "__main__":
    main()
