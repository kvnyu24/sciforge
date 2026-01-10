"""
Experiment 223: 2D Square Lattice Fermi Surface

Demonstrates the Fermi surface of a 2D square lattice tight-binding model,
showing how the Fermi surface evolves from circular (low filling) to
square (half-filling) due to the underlying lattice symmetry.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def tight_binding_2d(kx, ky, a, t, t_prime=0):
    """
    Tight-binding dispersion for 2D square lattice.

    Args:
        kx, ky: Wavevector components (can be arrays)
        a: Lattice constant
        t: Nearest-neighbor hopping
        t_prime: Next-nearest-neighbor hopping

    Returns:
        Energy E(kx, ky)
    """
    # Nearest-neighbor hopping
    E = -2 * t * (np.cos(kx * a) + np.cos(ky * a))
    # Next-nearest-neighbor hopping (diagonal)
    E += -4 * t_prime * np.cos(kx * a) * np.cos(ky * a)
    return E


def compute_fermi_surface(kx_grid, ky_grid, a, t, E_F, t_prime=0):
    """
    Compute Fermi surface contour.

    Args:
        kx_grid, ky_grid: Meshgrid of k-points
        a: Lattice constant
        t: Nearest-neighbor hopping
        E_F: Fermi energy
        t_prime: Next-nearest-neighbor hopping

    Returns:
        Energy grid for contour plotting
    """
    E = tight_binding_2d(kx_grid, ky_grid, a, t, t_prime)
    return E - E_F


def main():
    """Main simulation and visualization."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Parameters
    a = 1.0  # Lattice constant
    t = 1.0  # Hopping parameter

    # k-space grid (first Brillouin zone)
    kx = np.linspace(-np.pi/a, np.pi/a, 300)
    ky = np.linspace(-np.pi/a, np.pi/a, 300)
    KX, KY = np.meshgrid(kx, ky)

    # Plot Fermi surfaces at different fillings
    E_F_values = [-3.5, -2.0, 0.0, 2.0, 3.5]
    titles = ['Very low filling', 'Low filling', 'Half-filling (Van Hove)',
              'High filling', 'Very high filling']

    for idx, (E_F, title) in enumerate(zip(E_F_values, titles)):
        if idx < 3:
            ax = axes[0, idx]
        else:
            ax = axes[1, idx - 3]

        E_grid = tight_binding_2d(KX, KY, a, t)

        # Plot band energy as colormap
        im = ax.imshow(E_grid, extent=[-np.pi, np.pi, -np.pi, np.pi],
                       origin='lower', cmap='coolwarm', aspect='equal')

        # Plot Fermi surface contour
        contour = ax.contour(KX * a, KY * a, E_grid, levels=[E_F],
                            colors='black', linewidths=2)

        # Mark high-symmetry points
        ax.plot(0, 0, 'wo', markersize=8, markeredgecolor='black')  # Gamma
        ax.plot(np.pi, 0, 'ws', markersize=8, markeredgecolor='black')  # X
        ax.plot(np.pi, np.pi, 'w^', markersize=8, markeredgecolor='black')  # M

        ax.set_xlabel('kx * a')
        ax.set_ylabel('ky * a')
        ax.set_title(f'{title}\nE_F = {E_F:.1f}t')
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi, np.pi)

        # Draw BZ boundary
        ax.axhline(y=np.pi, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=-np.pi, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=np.pi, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=-np.pi, color='gray', linestyle='--', alpha=0.5)

    # Last subplot: Effect of t' (next-nearest-neighbor hopping)
    ax = axes[1, 2]

    E_F = 0  # Half-filling
    t_prime_values = [0, 0.2, 0.4]
    colors = ['black', 'blue', 'red']
    linestyles = ['-', '--', ':']

    for t_prime, color, ls in zip(t_prime_values, colors, linestyles):
        E_grid = tight_binding_2d(KX, KY, a, t, t_prime)
        ax.contour(KX * a, KY * a, E_grid, levels=[E_F],
                  colors=color, linewidths=2, linestyles=ls)

    ax.set_xlabel('kx * a')
    ax.set_ylabel('ky * a')
    ax.set_title("Effect of t' on Fermi Surface\n(E_F = 0)")

    # Create legend manually
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=c, linestyle=ls, linewidth=2,
                              label=f"t' = {tp}t")
                      for c, ls, tp in zip(colors, linestyles, t_prime_values)]
    ax.legend(handles=legend_elements, loc='upper right')

    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_aspect('equal')

    # Draw BZ boundary
    ax.axhline(y=np.pi, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=-np.pi, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=np.pi, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=-np.pi, color='gray', linestyle='--', alpha=0.5)

    plt.suptitle('Fermi Surface of 2D Square Lattice Tight-Binding Model\n'
                 'E(k) = -2t(cos(kx*a) + cos(ky*a)) - 4t\'cos(kx*a)cos(ky*a)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'square_lattice_fermi_surface.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'square_lattice_fermi_surface.png')}")

    # Additional figure: Dispersion along high-symmetry lines
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

    # High-symmetry path: Gamma -> X -> M -> Gamma
    n_points = 100

    # Gamma to X: (0,0) to (pi/a, 0)
    path_GX_kx = np.linspace(0, np.pi/a, n_points)
    path_GX_ky = np.zeros(n_points)
    E_GX = tight_binding_2d(path_GX_kx, path_GX_ky, a, t)

    # X to M: (pi/a, 0) to (pi/a, pi/a)
    path_XM_kx = np.full(n_points, np.pi/a)
    path_XM_ky = np.linspace(0, np.pi/a, n_points)
    E_XM = tight_binding_2d(path_XM_kx, path_XM_ky, a, t)

    # M to Gamma: (pi/a, pi/a) to (0, 0)
    path_MG_kx = np.linspace(np.pi/a, 0, n_points)
    path_MG_ky = np.linspace(np.pi/a, 0, n_points)
    E_MG = tight_binding_2d(path_MG_kx, path_MG_ky, a, t)

    # Combine path
    k_path = np.concatenate([np.linspace(0, 1, n_points),
                             np.linspace(1, 2, n_points),
                             np.linspace(2, 2 + np.sqrt(2), n_points)])
    E_path = np.concatenate([E_GX, E_XM, E_MG])

    ax2 = axes2[0]
    ax2.plot(k_path, E_path, 'b-', lw=2)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='E_F = 0')
    ax2.axvline(x=1, color='gray', linestyle=':', alpha=0.5)
    ax2.axvline(x=2, color='gray', linestyle=':', alpha=0.5)

    ax2.set_xticks([0, 1, 2, 2 + np.sqrt(2)])
    ax2.set_xticklabels(['$\Gamma$', 'X', 'M', '$\Gamma$'])
    ax2.set_xlabel('High-symmetry path')
    ax2.set_ylabel('Energy (units of t)')
    ax2.set_title('Band Structure Along High-Symmetry Path')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # DOS
    ax3 = axes2[1]

    # Compute DOS by histogram of energies over full BZ
    E_all = tight_binding_2d(KX, KY, a, t).flatten()
    E_bins = np.linspace(E_all.min(), E_all.max(), 100)
    dos, edges = np.histogram(E_all, bins=E_bins, density=True)
    E_centers = (edges[:-1] + edges[1:]) / 2

    ax3.fill_between(E_centers, dos, alpha=0.5, color='blue')
    ax3.plot(E_centers, dos, 'b-', lw=2)

    # Mark Van Hove singularities
    ax3.axvline(x=0, color='red', linestyle='--', lw=2, label='Van Hove singularity')

    ax3.set_xlabel('Energy (units of t)')
    ax3.set_ylabel('DOS (arb. units)')
    ax3.set_title('Density of States')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    fig2.suptitle('2D Square Lattice: Band Structure and DOS', fontsize=14, y=1.02)
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, 'square_lattice_band_dos.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'square_lattice_band_dos.png')}")


if __name__ == "__main__":
    main()
