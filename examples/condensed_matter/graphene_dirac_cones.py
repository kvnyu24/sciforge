"""
Experiment 224: Graphene Dirac Cones

Demonstrates the band structure of graphene showing the linear dispersion
(Dirac cones) at the K and K' points. The honeycomb lattice with two
atoms per unit cell leads to this remarkable relativistic-like dispersion.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def graphene_dispersion(kx, ky, a=1.0, t=2.7):
    """
    Tight-binding dispersion for graphene.

    E(k) = +-t * sqrt(3 + 2*cos(sqrt(3)*ky*a) + 4*cos(sqrt(3)*ky*a/2)*cos(3*kx*a/2))

    Args:
        kx, ky: Wavevector components
        a: C-C bond length (default 1.42 Angstrom scaled to 1.0)
        t: Hopping parameter (default 2.7 eV)

    Returns:
        E_plus, E_minus: Upper and lower bands
    """
    # f(k) factor
    f_k = np.sqrt(3 + 2*np.cos(np.sqrt(3)*ky*a) +
                  4*np.cos(np.sqrt(3)*ky*a/2)*np.cos(3*kx*a/2))

    E_plus = t * f_k
    E_minus = -t * f_k

    return E_plus, E_minus


def graphene_dispersion_full(kx, ky, a=1.0, t=2.7):
    """
    Full tight-binding dispersion including complex phase.

    Args:
        kx, ky: Wavevector components
        a: Lattice constant
        t: Hopping parameter

    Returns:
        E_plus, E_minus: Upper and lower bands
    """
    # Lattice vectors for graphene
    a1 = a * np.array([np.sqrt(3)/2, 3/2])
    a2 = a * np.array([-np.sqrt(3)/2, 3/2])

    # Nearest neighbor vectors
    delta1 = a * np.array([0, 1])
    delta2 = a * np.array([np.sqrt(3)/2, -1/2])
    delta3 = a * np.array([-np.sqrt(3)/2, -1/2])

    # f(k) = sum over NN: exp(i k.delta)
    f_k = (np.exp(1j * (kx * delta1[0] + ky * delta1[1])) +
           np.exp(1j * (kx * delta2[0] + ky * delta2[1])) +
           np.exp(1j * (kx * delta3[0] + ky * delta3[1])))

    E = t * np.abs(f_k)
    return E, -E


def main():
    """Main simulation and visualization."""

    fig = plt.figure(figsize=(16, 12))

    # Parameters
    a = 1.0  # Bond length (scaled)
    t = 2.7  # Hopping in eV

    # Reciprocal lattice vectors
    b1 = (2*np.pi/(3*a)) * np.array([np.sqrt(3), 1])
    b2 = (2*np.pi/(3*a)) * np.array([-np.sqrt(3), 1])

    # K and K' points (Dirac points)
    K = (2*np.pi/(3*a)) * np.array([1/np.sqrt(3), 1])
    K_prime = (2*np.pi/(3*a)) * np.array([-1/np.sqrt(3), 1])

    # k-space grid
    kx = np.linspace(-2*np.pi/a, 2*np.pi/a, 300)
    ky = np.linspace(-2*np.pi/a, 2*np.pi/a, 300)
    KX, KY = np.meshgrid(kx, ky)

    E_plus, E_minus = graphene_dispersion_full(KX, KY, a, t)

    # Plot 1: 3D surface plot of bands
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')

    # Downsample for 3D plot
    stride = 5
    ax1.plot_surface(KX[::stride, ::stride], KY[::stride, ::stride],
                     E_plus[::stride, ::stride], cmap='Reds', alpha=0.7)
    ax1.plot_surface(KX[::stride, ::stride], KY[::stride, ::stride],
                     E_minus[::stride, ::stride], cmap='Blues', alpha=0.7)

    ax1.set_xlabel('kx (1/a)')
    ax1.set_ylabel('ky (1/a)')
    ax1.set_zlabel('E (eV)')
    ax1.set_title('Graphene Band Structure (3D)')

    # Mark Dirac points
    ax1.scatter([K[0]], [K[1]], [0], color='green', s=100, marker='o')
    ax1.scatter([K_prime[0]], [K_prime[1]], [0], color='green', s=100, marker='o')

    # Plot 2: Contour plot showing Dirac points
    ax2 = fig.add_subplot(2, 2, 2)

    contour = ax2.contourf(KX, KY, E_plus, levels=50, cmap='RdBu_r')
    plt.colorbar(contour, ax=ax2, label='E (eV)')

    # Mark Dirac points
    ax2.plot(K[0], K[1], 'ko', markersize=10, label='K point')
    ax2.plot(K_prime[0], K_prime[1], 'k^', markersize=10, label="K' point")
    ax2.plot(-K[0], -K[1], 'ko', markersize=10)
    ax2.plot(-K_prime[0], -K_prime[1], 'k^', markersize=10)

    # Draw first Brillouin zone (hexagon)
    bz_vertices = []
    for i in range(6):
        angle = np.pi/6 + i * np.pi/3
        r = 4*np.pi / (3*np.sqrt(3)*a)
        bz_vertices.append([r * np.cos(angle), r * np.sin(angle)])
    bz_vertices.append(bz_vertices[0])
    bz_vertices = np.array(bz_vertices)
    ax2.plot(bz_vertices[:, 0], bz_vertices[:, 1], 'g-', lw=2, label='1st BZ')

    ax2.set_xlabel('kx (1/a)')
    ax2.set_ylabel('ky (1/a)')
    ax2.set_title('Upper Band Contours')
    ax2.legend()
    ax2.set_aspect('equal')

    # Plot 3: Band structure along high-symmetry path
    ax3 = fig.add_subplot(2, 2, 3)

    n_points = 100

    # High-symmetry points
    Gamma = np.array([0, 0])
    M = (b1 + b2) / 2

    # Path: Gamma -> K -> M -> Gamma
    path_GK_kx = np.linspace(Gamma[0], K[0], n_points)
    path_GK_ky = np.linspace(Gamma[1], K[1], n_points)
    E_GK_plus, E_GK_minus = graphene_dispersion_full(path_GK_kx, path_GK_ky, a, t)

    path_KM_kx = np.linspace(K[0], M[0], n_points)
    path_KM_ky = np.linspace(K[1], M[1], n_points)
    E_KM_plus, E_KM_minus = graphene_dispersion_full(path_KM_kx, path_KM_ky, a, t)

    path_MG_kx = np.linspace(M[0], Gamma[0], n_points)
    path_MG_ky = np.linspace(M[1], Gamma[1], n_points)
    E_MG_plus, E_MG_minus = graphene_dispersion_full(path_MG_kx, path_MG_ky, a, t)

    # Combine paths
    path_lengths = [0,
                    np.linalg.norm(K - Gamma),
                    np.linalg.norm(K - Gamma) + np.linalg.norm(M - K),
                    np.linalg.norm(K - Gamma) + np.linalg.norm(M - K) + np.linalg.norm(Gamma - M)]

    k_path = np.concatenate([
        np.linspace(path_lengths[0], path_lengths[1], n_points),
        np.linspace(path_lengths[1], path_lengths[2], n_points),
        np.linspace(path_lengths[2], path_lengths[3], n_points)
    ])

    E_path_plus = np.concatenate([E_GK_plus, E_KM_plus, E_MG_plus])
    E_path_minus = np.concatenate([E_GK_minus, E_KM_minus, E_MG_minus])

    ax3.plot(k_path, E_path_plus, 'r-', lw=2, label='Conduction band')
    ax3.plot(k_path, E_path_minus, 'b-', lw=2, label='Valence band')
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    ax3.axvline(x=path_lengths[1], color='gray', linestyle=':', alpha=0.5)
    ax3.axvline(x=path_lengths[2], color='gray', linestyle=':', alpha=0.5)

    ax3.set_xticks(path_lengths)
    ax3.set_xticklabels(['$\Gamma$', 'K', 'M', '$\Gamma$'])
    ax3.set_xlabel('High-symmetry path')
    ax3.set_ylabel('Energy (eV)')
    ax3.set_title('Band Structure of Graphene')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Linear dispersion near K point (Dirac cone)
    ax4 = fig.add_subplot(2, 2, 4)

    # Zoom in near K point
    delta_k = np.linspace(-0.5, 0.5, 100)
    dkx, dky = np.meshgrid(delta_k, delta_k)

    kx_near_K = K[0] + dkx
    ky_near_K = K[1] + dky

    E_near_K_plus, E_near_K_minus = graphene_dispersion_full(kx_near_K, ky_near_K, a, t)

    # Plot contours
    contour_K = ax4.contourf(dkx, dky, E_near_K_plus, levels=30, cmap='Reds')
    plt.colorbar(contour_K, ax=ax4, label='E (eV)')

    # Overlay circles to show linear dispersion
    for r in [0.1, 0.2, 0.3, 0.4]:
        circle = plt.Circle((0, 0), r, fill=False, color='black', linestyle='--', alpha=0.5)
        ax4.add_patch(circle)

    ax4.set_xlabel('$\delta k_x$ (1/a)')
    ax4.set_ylabel('$\delta k_y$ (1/a)')
    ax4.set_title('Dirac Cone near K Point\n(Linear dispersion E ~ v_F |k|)')
    ax4.set_aspect('equal')

    # Add text showing Fermi velocity
    v_F = 3 * t * a / 2  # Fermi velocity in natural units
    ax4.text(0.02, 0.98, f'$v_F = \\frac{{3ta}}{{2\\hbar}}$\n$\\approx {v_F:.2f}$ (units of a/hbar)',
             transform=ax4.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('Graphene: Honeycomb Lattice and Dirac Fermions\n'
                 'Linear dispersion at K and K\' points leads to massless Dirac behavior',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'graphene_dirac_cones.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'graphene_dirac_cones.png')}")


if __name__ == "__main__":
    main()
