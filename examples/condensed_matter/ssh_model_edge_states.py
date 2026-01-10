"""
Experiment 239: SSH Model Edge States

Demonstrates the Su-Schrieffer-Heeger (SSH) model - a prototype for
topological insulators. Shows how alternating hopping strengths lead
to topologically protected edge states at zero energy.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg


def ssh_hamiltonian(N, v, w, periodic=False):
    """
    Build SSH model Hamiltonian.

    H = v * sum_m (|m,A><m,B| + h.c.) + w * sum_m (|m,B><m+1,A| + h.c.)

    Args:
        N: Number of unit cells
        v: Intra-cell hopping
        w: Inter-cell hopping
        periodic: Whether to use periodic boundary conditions

    Returns:
        2N x 2N Hamiltonian matrix
    """
    dim = 2 * N  # 2 sites per unit cell (A and B sublattices)
    H = np.zeros((dim, dim))

    for m in range(N):
        # Intra-cell hopping (A -> B within cell m)
        i_A = 2 * m
        i_B = 2 * m + 1
        H[i_A, i_B] = v
        H[i_B, i_A] = v

        # Inter-cell hopping (B of cell m -> A of cell m+1)
        if m < N - 1:
            i_B = 2 * m + 1
            i_A_next = 2 * (m + 1)
            H[i_B, i_A_next] = w
            H[i_A_next, i_B] = w
        elif periodic:
            # Periodic boundary condition
            i_B = 2 * m + 1
            i_A_next = 0
            H[i_B, i_A_next] = w
            H[i_A_next, i_B] = w

    return H


def ssh_bulk_bands(k, v, w):
    """
    Bulk band structure of SSH model.

    E(k) = +/- sqrt(v^2 + w^2 + 2*v*w*cos(k))

    Args:
        k: Wavevector in units where lattice constant a = 1
        v: Intra-cell hopping
        w: Inter-cell hopping

    Returns:
        E_plus, E_minus: Upper and lower band energies
    """
    E_sq = v**2 + w**2 + 2*v*w*np.cos(k)
    E = np.sqrt(E_sq)
    return E, -E


def winding_number(v, w):
    """
    Calculate topological winding number.

    nu = 0 if |v| > |w|  (trivial phase)
    nu = 1 if |v| < |w|  (topological phase)

    The winding number counts how many times h(k) winds around the origin
    as k goes from 0 to 2*pi.
    """
    if np.abs(v) > np.abs(w):
        return 0
    elif np.abs(v) < np.abs(w):
        return 1
    else:
        return 0.5  # Transition point


def compute_winding_number_numerical(v, w, n_k=1000):
    """
    Compute winding number numerically by integrating Berry phase.

    nu = (1/2pi) * integral dk * d(arg(h(k)))/dk

    where h(k) = v + w*exp(ik)
    """
    k = np.linspace(0, 2*np.pi, n_k)

    # h(k) = v + w*exp(ik) = (v + w*cos(k)) + i*w*sin(k)
    h_real = v + w * np.cos(k)
    h_imag = w * np.sin(k)

    # Phase of h(k)
    phase = np.arctan2(h_imag, h_real)

    # Unwrap phase
    phase_unwrapped = np.unwrap(phase)

    # Winding number = total phase change / 2*pi
    return (phase_unwrapped[-1] - phase_unwrapped[0]) / (2 * np.pi)


def main():
    """Main simulation and visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Bulk band structure
    ax1 = axes[0, 0]

    k = np.linspace(-np.pi, np.pi, 500)

    # Different hopping ratios
    params = [
        (1.0, 0.5, 'Trivial (v > w)'),
        (1.0, 1.0, 'Critical (v = w)'),
        (0.5, 1.0, 'Topological (v < w)')
    ]
    colors = ['blue', 'green', 'red']

    for (v, w, label), color in zip(params, colors):
        E_plus, E_minus = ssh_bulk_bands(k, v, w)
        ax1.plot(k/np.pi, E_plus, color=color, lw=2, label=label)
        ax1.plot(k/np.pi, E_minus, color=color, lw=2)

    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.set_xlabel('k / pi')
    ax1.set_ylabel('Energy')
    ax1.set_title('SSH Model Bulk Band Structure')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-1, 1)

    # Mark gap
    ax1.fill_between([-1, 1], -0.5, 0.5, alpha=0.1, color='yellow')

    # Plot 2: Energy spectrum of finite chain
    ax2 = axes[0, 1]

    N = 30  # Number of unit cells

    # Trivial phase
    v, w = 1.0, 0.3
    H_trivial = ssh_hamiltonian(N, v, w, periodic=False)
    E_trivial, V_trivial = linalg.eigh(H_trivial)

    # Topological phase
    v, w = 0.3, 1.0
    H_topo = ssh_hamiltonian(N, v, w, periodic=False)
    E_topo, V_topo = linalg.eigh(H_topo)

    ax2.plot(range(1, 2*N+1), E_trivial, 'bo', markersize=5, label='Trivial (v=1, w=0.3)')
    ax2.plot(range(1, 2*N+1), E_topo, 'r^', markersize=5, alpha=0.7, label='Topological (v=0.3, w=1)')

    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Highlight zero-energy edge states
    zero_states = np.where(np.abs(E_topo) < 0.1)[0]
    if len(zero_states) > 0:
        ax2.scatter([z+1 for z in zero_states], E_topo[zero_states],
                   s=100, c='red', marker='*', zorder=5, label='Edge states')

    ax2.set_xlabel('State index')
    ax2.set_ylabel('Energy')
    ax2.set_title(f'Energy Spectrum (N = {N} unit cells)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Edge state wavefunctions
    ax3 = axes[1, 0]

    # Find and plot edge states
    sites = np.arange(1, 2*N+1)

    # States near zero energy in topological phase
    for idx in zero_states[:2]:  # Plot up to 2 edge states
        psi = V_topo[:, idx]
        psi_sq = np.abs(psi)**2
        ax3.plot(sites, psi_sq, lw=2, marker='o', markersize=4,
                label=f'E = {E_topo[idx]:.4f}')

    # Compare with bulk state
    bulk_idx = N
    psi_bulk = V_topo[:, bulk_idx]
    ax3.plot(sites, np.abs(psi_bulk)**2, 'k--', lw=1.5, alpha=0.5,
            label=f'Bulk state (E = {E_topo[bulk_idx]:.2f})')

    ax3.set_xlabel('Site index')
    ax3.set_ylabel('|psi|^2')
    ax3.set_title('Edge State Wavefunctions (Topological Phase)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Mark sublattices
    for i in range(0, 2*N, 2):
        ax3.axvspan(i+0.5, i+1.5, alpha=0.1, color='blue')
    ax3.text(0.02, 0.95, 'Blue: A sublattice\nWhite: B sublattice',
             transform=ax3.transAxes, fontsize=9, va='top')

    # Plot 4: Phase diagram and winding number
    ax4 = axes[1, 1]

    v_range = np.linspace(0.01, 2, 100)
    w_range = np.linspace(0.01, 2, 100)
    V, W = np.meshgrid(v_range, w_range)

    # Winding number (0 for v > w, 1 for v < w)
    nu = np.where(V < W, 1, 0)

    im = ax4.pcolormesh(V, W, nu, cmap='RdBu', shading='auto')
    plt.colorbar(im, ax=ax4, label='Winding number')

    # Phase boundary
    ax4.plot([0, 2], [0, 2], 'k--', lw=2, label='Phase boundary (v = w)')

    # Mark phases
    ax4.text(1.5, 0.5, 'Trivial\n(nu = 0)', fontsize=12, ha='center',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax4.text(0.5, 1.5, 'Topological\n(nu = 1)', fontsize=12, ha='center',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax4.set_xlabel('Intra-cell hopping v')
    ax4.set_ylabel('Inter-cell hopping w')
    ax4.set_title('SSH Model Phase Diagram')
    ax4.legend(loc='upper right')
    ax4.set_aspect('equal')

    plt.suptitle('Su-Schrieffer-Heeger (SSH) Model\n'
                 'Prototype topological insulator with edge states',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'ssh_model_edge_states.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'ssh_model_edge_states.png')}")


if __name__ == "__main__":
    main()
