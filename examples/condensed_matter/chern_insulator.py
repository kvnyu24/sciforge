"""
Experiment 240: Chern Insulator Chern Number

Demonstrates the Chern insulator model (Haldane model on honeycomb lattice),
showing how breaking time-reversal symmetry leads to a quantized Hall
conductance characterized by the Chern number.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg


def haldane_hamiltonian(kx, ky, t1, t2, M, phi):
    """
    Haldane model Hamiltonian on honeycomb lattice.

    H(k) = h_0(k)*I + h_x(k)*sigma_x + h_y(k)*sigma_y + h_z(k)*sigma_z

    Args:
        kx, ky: Crystal momenta
        t1: Nearest-neighbor hopping
        t2: Next-nearest-neighbor hopping
        M: Sublattice mass (breaks inversion)
        phi: Phase of NNN hopping (breaks time-reversal)

    Returns:
        2x2 Hamiltonian matrix
    """
    # Lattice vectors
    a1 = np.array([1, 0])
    a2 = np.array([0.5, np.sqrt(3)/2])
    a3 = np.array([-0.5, np.sqrt(3)/2])

    # Nearest-neighbor vectors
    delta1 = np.array([0, 1/np.sqrt(3)])
    delta2 = np.array([0.5, -1/(2*np.sqrt(3))])
    delta3 = np.array([-0.5, -1/(2*np.sqrt(3))])

    # Next-nearest-neighbor vectors (connecting same sublattice)
    b1 = a1
    b2 = a2
    b3 = a3 - a1

    # k vector
    k = np.array([kx, ky])

    # Nearest-neighbor term: t1 * sum exp(i k.delta)
    f_k = (np.exp(1j * np.dot(k, delta1)) +
           np.exp(1j * np.dot(k, delta2)) +
           np.exp(1j * np.dot(k, delta3)))

    # Next-nearest-neighbor term: 2*t2*cos(phi) * sum cos(k.b) - 2*t2*sin(phi) * sum sin(k.b)
    g_k = (np.sin(np.dot(k, b1)) +
           np.sin(np.dot(k, b2)) +
           np.sin(np.dot(k, b3)))

    h_0 = 2 * t2 * np.cos(phi) * (np.cos(np.dot(k, b1)) +
                                   np.cos(np.dot(k, b2)) +
                                   np.cos(np.dot(k, b3)))

    h_x = t1 * np.real(f_k)
    h_y = t1 * np.imag(f_k)
    h_z = M - 2 * t2 * np.sin(phi) * g_k

    # Pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    I = np.eye(2)

    H = h_0 * I + h_x * sigma_x + h_y * sigma_y + h_z * sigma_z

    return H


def compute_berry_curvature(kx, ky, t1, t2, M, phi, dk=1e-5):
    """
    Compute Berry curvature at point (kx, ky).

    Omega = Im[ <du/dkx | du/dky> - <du/dky | du/dkx> ]

    Uses numerical derivatives.

    Args:
        kx, ky: Point in k-space
        t1, t2, M, phi: Model parameters
        dk: Step size for numerical derivative

    Returns:
        Berry curvature for lower band
    """
    # Eigenstates at nearby points
    def get_lower_eigenstate(kx, ky):
        H = haldane_hamiltonian(kx, ky, t1, t2, M, phi)
        eigenvalues, eigenvectors = linalg.eigh(H)
        return eigenvectors[:, 0]  # Lower band

    u = get_lower_eigenstate(kx, ky)
    u_x = get_lower_eigenstate(kx + dk, ky)
    u_y = get_lower_eigenstate(kx, ky + dk)

    # Fix gauge: make overlap with u positive
    if np.real(np.vdot(u, u_x)) < 0:
        u_x = -u_x
    if np.real(np.vdot(u, u_y)) < 0:
        u_y = -u_y

    # Numerical derivatives
    du_dkx = (u_x - u) / dk
    du_dky = (u_y - u) / dk

    # Berry curvature
    omega = 2 * np.imag(np.vdot(du_dkx, du_dky))

    return omega


def compute_chern_number(t1, t2, M, phi, n_k=50):
    """
    Compute Chern number by integrating Berry curvature over BZ.

    C = (1/2pi) * integral Omega(k) d^2k

    Args:
        t1, t2, M, phi: Model parameters
        n_k: Number of k-points per dimension

    Returns:
        Chern number (should be integer)
    """
    # Brillouin zone for honeycomb lattice (rectangular approximation)
    kx_range = np.linspace(-2*np.pi, 2*np.pi, n_k)
    ky_range = np.linspace(-2*np.pi, 2*np.pi, n_k)

    dk = kx_range[1] - kx_range[0]

    # Integrate Berry curvature
    total_curvature = 0
    for kx in kx_range:
        for ky in ky_range:
            omega = compute_berry_curvature(kx, ky, t1, t2, M, phi)
            total_curvature += omega * dk**2

    chern = total_curvature / (2 * np.pi)
    return chern


def main():
    """Main simulation and visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Model parameters
    t1 = 1.0   # Nearest-neighbor hopping
    t2 = 0.3   # Next-nearest-neighbor hopping
    phi = np.pi/2  # Phase that breaks time-reversal

    # Plot 1: Band structure comparison
    ax1 = axes[0, 0]

    # k-path: K -> Gamma -> M -> K'
    n_path = 100

    # High-symmetry points for honeycomb
    K = np.array([4*np.pi/3, 0])
    Gamma = np.array([0, 0])
    M = np.array([np.pi, np.pi/np.sqrt(3)])
    K_prime = np.array([2*np.pi/3, 2*np.pi/np.sqrt(3)])

    def get_bands_along_path(p1, p2, n, t1, t2, M_mass, phi):
        path = [p1 + (p2-p1)*i/(n-1) for i in range(n)]
        E = []
        for k in path:
            H = haldane_hamiltonian(k[0], k[1], t1, t2, M_mass, phi)
            eigenvalues, _ = linalg.eigh(H)
            E.append(eigenvalues)
        return np.array(E)

    # Trivial case (phi = 0)
    E_trivial = np.vstack([
        get_bands_along_path(K, Gamma, n_path, t1, t2, 0, 0),
        get_bands_along_path(Gamma, M, n_path, t1, t2, 0, 0),
        get_bands_along_path(M, K_prime, n_path, t1, t2, 0, 0)
    ])

    # Topological case (phi = pi/2)
    E_topo = np.vstack([
        get_bands_along_path(K, Gamma, n_path, t1, t2, 0, phi),
        get_bands_along_path(Gamma, M, n_path, t1, t2, 0, phi),
        get_bands_along_path(M, K_prime, n_path, t1, t2, 0, phi)
    ])

    k_plot = np.linspace(0, 3, 3*n_path)

    ax1.plot(k_plot, E_trivial[:, 0], 'b-', lw=2, label='Trivial (phi=0)')
    ax1.plot(k_plot, E_trivial[:, 1], 'b-', lw=2)
    ax1.plot(k_plot, E_topo[:, 0], 'r--', lw=2, label='Topological (phi=pi/2)')
    ax1.plot(k_plot, E_topo[:, 1], 'r--', lw=2)

    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.axvline(x=1, color='gray', linestyle=':', alpha=0.5)
    ax1.axvline(x=2, color='gray', linestyle=':', alpha=0.5)

    ax1.set_xticks([0, 1, 2, 3])
    ax1.set_xticklabels(['K', 'Gamma', 'M', "K'"])
    ax1.set_ylabel('Energy')
    ax1.set_title('Haldane Model Band Structure')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Berry curvature map
    ax2 = axes[0, 1]

    n_k = 50
    kx_range = np.linspace(-np.pi, np.pi, n_k)
    ky_range = np.linspace(-np.pi, np.pi, n_k)
    KX, KY = np.meshgrid(kx_range, ky_range)

    Omega = np.zeros((n_k, n_k))
    for i, kx in enumerate(kx_range):
        for j, ky in enumerate(ky_range):
            Omega[j, i] = compute_berry_curvature(kx, ky, t1, t2, 0, phi, dk=0.01)

    im = ax2.pcolormesh(KX, KY, Omega, cmap='RdBu', shading='auto',
                        vmin=-np.max(np.abs(Omega)), vmax=np.max(np.abs(Omega)))
    plt.colorbar(im, ax=ax2, label='Berry curvature')

    ax2.set_xlabel('kx')
    ax2.set_ylabel('ky')
    ax2.set_title('Berry Curvature in k-space')
    ax2.set_aspect('equal')

    # Plot 3: Phase diagram
    ax3 = axes[1, 0]

    M_range = np.linspace(-6*t2, 6*t2, 100)
    phi_range = np.linspace(0, np.pi, 100)
    M_grid, Phi_grid = np.meshgrid(M_range, phi_range)

    # Phase boundaries: |M| = 3*sqrt(3)*t2*|sin(phi)|
    phase = np.zeros_like(M_grid)
    for i, m in enumerate(M_range):
        for j, p in enumerate(phi_range):
            boundary = 3 * np.sqrt(3) * t2 * np.abs(np.sin(p))
            if np.abs(m) < boundary:
                if p > 0:
                    phase[j, i] = np.sign(np.sin(p))
                else:
                    phase[j, i] = 0

    im3 = ax3.pcolormesh(M_grid / t2, Phi_grid / np.pi, phase, cmap='RdBu',
                         shading='auto', vmin=-1, vmax=1)
    plt.colorbar(im3, ax=ax3, label='Chern number')

    # Phase boundaries
    phi_plot = np.linspace(0, np.pi, 100)
    M_boundary = 3 * np.sqrt(3) * t2 * np.abs(np.sin(phi_plot))
    ax3.plot(M_boundary / t2, phi_plot / np.pi, 'k-', lw=2)
    ax3.plot(-M_boundary / t2, phi_plot / np.pi, 'k-', lw=2)

    ax3.set_xlabel('M / t2')
    ax3.set_ylabel('phi / pi')
    ax3.set_title('Haldane Model Phase Diagram')

    # Mark phases
    ax3.text(0, 0.5, 'C = +1', fontsize=14, ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax3.text(4, 0.5, 'C = 0', fontsize=14, ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax3.text(-4, 0.5, 'C = 0', fontsize=14, ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot 4: Edge states in ribbon geometry
    ax4 = axes[1, 1]

    # Simplified: show that Chern number relates to Hall conductance
    # sigma_xy = C * e^2/h

    chern_values = [-1, 0, 1]
    sigma_values = [c for c in chern_values]

    ax4.bar(chern_values, sigma_values, color=['blue', 'gray', 'red'], alpha=0.7)
    ax4.set_xlabel('Chern Number C')
    ax4.set_ylabel('Hall Conductance (e^2/h)')
    ax4.set_title('Quantized Hall Conductance')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_xticks(chern_values)

    # Add explanation
    ax4.text(0.5, 0.95, '$\\sigma_{xy} = C \\times \\frac{e^2}{h}$\n\nQuantized Hall effect\nwithout magnetic field!',
             transform=ax4.transAxes, fontsize=11, va='top', ha='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle('Chern Insulator: Haldane Model\n'
                 'Quantum Hall effect without Landau levels',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'chern_insulator.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'chern_insulator.png')}")


if __name__ == "__main__":
    main()
