"""
Experiment 242: Berry Curvature Integration

Demonstrates Berry phase and Berry curvature in band theory, showing
how the geometric phase accumulated during adiabatic evolution in
parameter space leads to topological invariants.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg


def two_level_hamiltonian(params, h_func):
    """
    General two-level Hamiltonian H = h(params) . sigma

    Args:
        params: Parameter values
        h_func: Function that returns h-vector

    Returns:
        2x2 Hamiltonian matrix
    """
    hx, hy, hz = h_func(params)

    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])

    return hx * sigma_x + hy * sigma_y + hz * sigma_z


def berry_connection(k, h_func, dk=1e-6):
    """
    Compute Berry connection A_i = i * <u| d/dk_i |u>.

    Args:
        k: k-point (array)
        h_func: Function giving h(k)
        dk: Step for numerical derivative

    Returns:
        Berry connection components
    """
    # Get ground state at k
    H = two_level_hamiltonian(k, h_func)
    E, V = linalg.eigh(H)
    u = V[:, 0]  # Ground state

    A = []

    for i in range(len(k)):
        # Numerical derivative
        k_plus = k.copy()
        k_plus[i] += dk

        H_plus = two_level_hamiltonian(k_plus, h_func)
        _, V_plus = linalg.eigh(H_plus)
        u_plus = V_plus[:, 0]

        # Fix gauge
        if np.real(np.vdot(u, u_plus)) < 0:
            u_plus = -u_plus

        du_dk = (u_plus - u) / dk

        # Berry connection: A = i * <u|du/dk>
        A_i = 1j * np.vdot(u, du_dk)
        A.append(np.real(A_i))

    return np.array(A)


def berry_curvature_2d(kx, ky, h_func, dk=1e-5):
    """
    Compute Berry curvature in 2D.

    Omega = dA_y/dkx - dA_x/dky

    Args:
        kx, ky: k-point
        h_func: Function giving h(k)
        dk: Step for numerical derivative

    Returns:
        Berry curvature
    """
    k = np.array([kx, ky])

    # Berry connection at nearby points
    A_00 = berry_connection(k, h_func, dk)
    A_x0 = berry_connection(k + np.array([dk, 0]), h_func, dk)
    A_0y = berry_connection(k + np.array([0, dk]), h_func, dk)

    # Curvature: F_xy = dA_y/dx - dA_x/dy
    dAy_dx = (A_x0[1] - A_00[1]) / dk
    dAx_dy = (A_0y[0] - A_00[0]) / dk

    return dAy_dx - dAx_dy


def analytic_berry_curvature(kx, ky, h_func):
    """
    Analytic Berry curvature for two-band model.

    Omega = (1/2) * h . (dh/dkx x dh/dky) / |h|^3

    Args:
        kx, ky: k-point
        h_func: Function giving h(k)

    Returns:
        Berry curvature
    """
    dk = 1e-6
    k = np.array([kx, ky])

    h = np.array(h_func(k))
    h_norm = np.linalg.norm(h)

    if h_norm < 1e-10:
        return 0

    # Numerical derivatives
    dh_dkx = (np.array(h_func(k + np.array([dk, 0]))) - np.array(h_func(k - np.array([dk, 0])))) / (2*dk)
    dh_dky = (np.array(h_func(k + np.array([0, dk]))) - np.array(h_func(k - np.array([0, dk])))) / (2*dk)

    # Cross product
    cross = np.cross(dh_dkx, dh_dky)

    # Dot with h-hat
    omega = 0.5 * np.dot(h, cross) / h_norm**3

    return omega


def berry_phase_loop(path, h_func, n_points=100):
    """
    Compute Berry phase around a closed loop.

    gamma = -Im log <u(0)|u(1)><u(1)|u(2)>...<u(N-1)|u(0)>

    Args:
        path: Function that returns k(t) for t in [0, 1]
        h_func: Function giving h(k)
        n_points: Number of points on path

    Returns:
        Berry phase
    """
    t_vals = np.linspace(0, 1, n_points + 1)[:-1]  # Exclude endpoint (same as start)

    # Get eigenstates along path
    states = []
    for t in t_vals:
        k = path(t)
        H = two_level_hamiltonian(k, h_func)
        _, V = linalg.eigh(H)
        states.append(V[:, 0])

    # Fix gauge: make overlaps positive
    for i in range(1, len(states)):
        if np.real(np.vdot(states[i-1], states[i])) < 0:
            states[i] = -states[i]

    # Compute product of overlaps
    product = 1.0
    for i in range(len(states)):
        overlap = np.vdot(states[i], states[(i+1) % len(states)])
        product *= overlap

    # Berry phase
    gamma = -np.imag(np.log(product))

    return gamma


def main():
    """Main simulation and visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Model: Massive Dirac fermion H = kx*sigma_x + ky*sigma_y + m*sigma_z
    def h_dirac(k, m=1.0):
        kx, ky = k
        return (kx, ky, m)

    # Plot 1: Berry curvature for Dirac model
    ax1 = axes[0, 0]

    n_k = 50
    kx_range = np.linspace(-3, 3, n_k)
    ky_range = np.linspace(-3, 3, n_k)
    KX, KY = np.meshgrid(kx_range, ky_range)

    m = 0.5
    Omega = np.zeros((n_k, n_k))

    for i, kx in enumerate(kx_range):
        for j, ky in enumerate(ky_range):
            Omega[j, i] = analytic_berry_curvature(kx, ky, lambda k: h_dirac(k, m))

    im1 = ax1.pcolormesh(KX, KY, Omega, cmap='RdBu', shading='auto',
                         vmin=-np.max(np.abs(Omega)), vmax=np.max(np.abs(Omega)))
    plt.colorbar(im1, ax=ax1, label='Berry curvature')

    ax1.set_xlabel('kx')
    ax1.set_ylabel('ky')
    ax1.set_title(f'Berry Curvature: Massive Dirac (m = {m})')
    ax1.set_aspect('equal')

    # Mark origin (where curvature is concentrated)
    ax1.plot(0, 0, 'k+', markersize=15, mew=2)

    # Plot 2: Berry curvature for different masses
    ax2 = axes[0, 1]

    k_radial = np.linspace(0.01, 3, 100)
    masses = [0.1, 0.5, 1.0, 2.0]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(masses)))

    for m, color in zip(masses, colors):
        Omega_radial = [analytic_berry_curvature(k, 0, lambda kp: h_dirac(kp, m)) for k in k_radial]
        ax2.plot(k_radial, Omega_radial, color=color, lw=2, label=f'm = {m}')

    ax2.set_xlabel('k (along kx)')
    ax2.set_ylabel('Berry curvature')
    ax2.set_title('Radial Profile of Berry Curvature')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Berry phase around loops
    ax3 = axes[1, 0]

    # Circular loops of different radii
    radii = np.linspace(0.1, 3, 30)
    m_values = [0.1, 0.5, 1.0]
    colors = ['blue', 'green', 'red']

    for m, color in zip(m_values, colors):
        phases = []
        for r in radii:
            def circle_path(t, r=r):
                theta = 2 * np.pi * t
                return np.array([r * np.cos(theta), r * np.sin(theta)])

            gamma = berry_phase_loop(circle_path, lambda k: h_dirac(k, m), n_points=100)
            phases.append(gamma)

        ax3.plot(radii, np.array(phases) / np.pi, color=color, lw=2, label=f'm = {m}')

    ax3.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label=r'$\pi$ (monopole)')
    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    ax3.set_xlabel('Loop radius')
    ax3.set_ylabel('Berry phase / pi')
    ax3.set_title('Berry Phase Enclosed by Circular Loops')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Integration to get Chern number
    ax4 = axes[1, 1]

    # Integrate Berry curvature over full BZ
    m_range = np.linspace(-3, 3, 50)
    chern_numbers = []

    for m in m_range:
        total = 0
        dk = 6 / n_k
        for kx in kx_range:
            for ky in ky_range:
                omega = analytic_berry_curvature(kx, ky, lambda k: h_dirac(k, m))
                total += omega * dk**2

        chern = total / (2 * np.pi)
        chern_numbers.append(chern)

    ax4.plot(m_range, chern_numbers, 'b-', lw=2)
    ax4.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax4.axhline(y=-0.5, color='gray', linestyle='--', alpha=0.5)
    ax4.axvline(x=0, color='red', linestyle=':', alpha=0.5, label='Gap closing')

    ax4.set_xlabel('Mass m')
    ax4.set_ylabel('Chern number')
    ax4.set_title('Chern Number vs Mass (Topological Phase Transition)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Annotate phases
    ax4.text(-2, 0.3, 'C = +1/2', fontsize=12, ha='center')
    ax4.text(2, -0.3, 'C = -1/2', fontsize=12, ha='center')
    ax4.annotate('Phase\ntransition', xy=(0, 0), xytext=(1, 0.3),
                arrowprops=dict(arrowstyle='->', color='red'), fontsize=10)

    plt.suptitle('Berry Phase and Curvature in Topological Band Theory\n'
                 r'$\Omega = \frac{1}{2}\frac{\hat{h} \cdot (\partial_{k_x}\hat{h} \times \partial_{k_y}\hat{h})}{|h|^2}$',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'berry_curvature_integration.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'berry_curvature_integration.png')}")


if __name__ == "__main__":
    main()
