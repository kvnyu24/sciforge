"""
Experiment 162: Spin Larmor Precession

This experiment demonstrates the Larmor precession of a quantum spin-1/2
particle in a magnetic field, including:
- Time evolution of spin state
- Precession on the Bloch sphere
- Larmor frequency dependence on field strength
- Connection between quantum and classical pictures
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)


def spin_hamiltonian(B: np.ndarray, gamma: float = 1.0, hbar: float = 1.0) -> np.ndarray:
    """
    Create Hamiltonian for spin in magnetic field.

    H = -gamma * B . S = -(hbar*gamma/2) * B . sigma

    Args:
        B: Magnetic field vector [Bx, By, Bz]
        gamma: Gyromagnetic ratio
        hbar: Reduced Planck constant

    Returns:
        2x2 Hamiltonian matrix
    """
    return -hbar * gamma / 2 * (B[0] * sigma_x + B[1] * sigma_y + B[2] * sigma_z)


def evolve_spin(psi0: np.ndarray, H: np.ndarray, t: float,
                hbar: float = 1.0) -> np.ndarray:
    """
    Evolve spin state under Hamiltonian.

    psi(t) = exp(-i*H*t/hbar) * psi(0)

    Args:
        psi0: Initial spinor [c_up, c_down]
        H: Hamiltonian matrix
        t: Time
        hbar: Reduced Planck constant

    Returns:
        Evolved spinor
    """
    # Diagonalize H
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    # Time evolution operator
    U = eigenvectors @ np.diag(np.exp(-1j * eigenvalues * t / hbar)) @ eigenvectors.conj().T

    return U @ psi0


def bloch_vector(psi: np.ndarray) -> np.ndarray:
    """
    Calculate Bloch vector from spinor.

    r = (2*Re(c_up* c_down), 2*Im(c_up* c_down), |c_up|^2 - |c_down|^2)
    """
    c_up, c_down = psi[0], psi[1]

    rx = 2 * np.real(np.conj(c_up) * c_down)
    ry = 2 * np.imag(np.conj(c_up) * c_down)
    rz = np.abs(c_up)**2 - np.abs(c_down)**2

    return np.array([rx, ry, rz])


def spin_up(theta: float = 0, phi: float = 0) -> np.ndarray:
    """
    Create spinor pointing along direction (theta, phi).

    |n> = cos(theta/2)|up> + exp(i*phi)*sin(theta/2)|down>

    Args:
        theta: Polar angle from z-axis
        phi: Azimuthal angle

    Returns:
        Normalized spinor
    """
    return np.array([np.cos(theta/2), np.exp(1j*phi) * np.sin(theta/2)], dtype=complex)


def larmor_frequency(B: float, gamma: float = 1.0) -> float:
    """Calculate Larmor frequency omega_L = gamma * B."""
    return gamma * B


def main():
    # Parameters
    gamma = 1.0  # Gyromagnetic ratio
    hbar = 1.0
    B0 = 1.0  # Field strength

    fig = plt.figure(figsize=(16, 10))

    # Plot 1: Bloch sphere precession trajectory (3D)
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')

    # Magnetic field along z
    B = np.array([0, 0, B0])
    H = spin_hamiltonian(B, gamma, hbar)

    # Initial state: spin along x
    theta0 = np.pi / 2
    phi0 = 0
    psi0 = spin_up(theta0, phi0)

    # Time evolution
    omega_L = larmor_frequency(B0, gamma)
    T = 2 * np.pi / omega_L  # Precession period
    t_arr = np.linspace(0, 2*T, 200)

    trajectory = []
    for t in t_arr:
        psi_t = evolve_spin(psi0, H, t, hbar)
        r = bloch_vector(psi_t)
        trajectory.append(r)

    trajectory = np.array(trajectory)

    # Draw Bloch sphere
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones_like(u), np.cos(v))
    ax1.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='gray')

    # Draw trajectory
    ax1.plot3D(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'b-', lw=2)

    # Draw starting point
    ax1.scatter(*trajectory[0], s=100, c='green', label='Start')
    ax1.scatter(*trajectory[-1], s=100, c='red', label='End')

    # Draw magnetic field direction
    ax1.quiver(0, 0, 0, 0, 0, 1.3, color='orange', arrow_length_ratio=0.1, lw=2)
    ax1.text(0, 0, 1.4, 'B', fontsize=12, color='orange')

    ax1.set_xlabel('Sx')
    ax1.set_ylabel('Sy')
    ax1.set_zlabel('Sz')
    ax1.set_title(f'Larmor Precession on Bloch Sphere\nomega_L = {omega_L:.2f}')
    ax1.legend()

    # Plot 2: Spin components vs time
    ax2 = fig.add_subplot(2, 3, 2)

    ax2.plot(t_arr / T, trajectory[:, 0], 'r-', lw=2, label='<Sx>')
    ax2.plot(t_arr / T, trajectory[:, 1], 'g-', lw=2, label='<Sy>')
    ax2.plot(t_arr / T, trajectory[:, 2], 'b-', lw=2, label='<Sz>')

    ax2.set_xlabel('Time t / T')
    ax2.set_ylabel('Spin expectation value')
    ax2.set_title('Spin Components vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Different initial states
    ax3 = fig.add_subplot(2, 3, 3)

    initial_states = [
        (np.pi/2, 0, '+x'),
        (np.pi/4, 0, '45 deg from z'),
        (np.pi/2, np.pi/4, 'xy plane, phi=45'),
        (np.pi/6, 0, '30 deg from z'),
    ]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(initial_states)))

    for (theta, phi, label), color in zip(initial_states, colors):
        psi0 = spin_up(theta, phi)
        sz_t = []
        for t in t_arr:
            psi_t = evolve_spin(psi0, H, t, hbar)
            r = bloch_vector(psi_t)
            sz_t.append(r[2])
        ax3.plot(t_arr / T, sz_t, color=color, lw=2, label=label)

    ax3.set_xlabel('Time t / T')
    ax3.set_ylabel('<Sz>')
    ax3.set_title('Different Initial States')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Effect of field strength
    ax4 = fig.add_subplot(2, 3, 4)

    B_values = [0.5, 1.0, 2.0, 4.0]
    colors_B = plt.cm.plasma(np.linspace(0.2, 0.9, len(B_values)))

    psi0 = spin_up(np.pi/2, 0)  # Start along +x
    t_arr_long = np.linspace(0, 10, 500)

    for B_test, color in zip(B_values, colors_B):
        B = np.array([0, 0, B_test])
        H = spin_hamiltonian(B, gamma, hbar)
        omega = larmor_frequency(B_test, gamma)

        sx_t = []
        for t in t_arr_long:
            psi_t = evolve_spin(psi0, H, t, hbar)
            r = bloch_vector(psi_t)
            sx_t.append(r[0])

        ax4.plot(t_arr_long, sx_t, color=color, lw=1.5,
                label=f'B = {B_test}, omega_L = {omega:.1f}')

    ax4.set_xlabel('Time t')
    ax4.set_ylabel('<Sx>')
    ax4.set_title('Larmor Frequency vs Field Strength')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Tilted magnetic field
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')

    # B at 45 degrees to z
    B_tilt = np.array([B0/np.sqrt(2), 0, B0/np.sqrt(2)])
    H_tilt = spin_hamiltonian(B_tilt, gamma, hbar)

    psi0 = spin_up(np.pi/2, 0)  # Start along +x

    trajectory_tilt = []
    for t in t_arr:
        psi_t = evolve_spin(psi0, H_tilt, t, hbar)
        r = bloch_vector(psi_t)
        trajectory_tilt.append(r)

    trajectory_tilt = np.array(trajectory_tilt)

    # Draw Bloch sphere
    ax5.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='gray')

    # Draw trajectory
    ax5.plot3D(trajectory_tilt[:, 0], trajectory_tilt[:, 1],
               trajectory_tilt[:, 2], 'b-', lw=2)

    # Draw magnetic field direction
    B_norm = B_tilt / np.linalg.norm(B_tilt)
    ax5.quiver(0, 0, 0, B_norm[0]*1.3, B_norm[1]*1.3, B_norm[2]*1.3,
               color='orange', arrow_length_ratio=0.1, lw=2)
    ax5.text(B_norm[0]*1.4, B_norm[1]*1.4, B_norm[2]*1.4, 'B', fontsize=12,
             color='orange')

    ax5.scatter(*trajectory_tilt[0], s=100, c='green', label='Start')

    ax5.set_xlabel('Sx')
    ax5.set_ylabel('Sy')
    ax5.set_zlabel('Sz')
    ax5.set_title('Precession Around Tilted Field')
    ax5.legend()

    # Plot 6: Spin coherence and phase
    ax6 = fig.add_subplot(2, 3, 6)

    # Phase of the off-diagonal element
    B = np.array([0, 0, B0])
    H = spin_hamiltonian(B, gamma, hbar)
    psi0 = spin_up(np.pi/2, 0)

    phase_evolution = []
    coherence = []

    for t in t_arr:
        psi_t = evolve_spin(psi0, H, t, hbar)

        # Density matrix
        rho = np.outer(psi_t, np.conj(psi_t))

        # Off-diagonal coherence
        coh = rho[0, 1]
        coherence.append(np.abs(coh))
        phase_evolution.append(np.angle(coh))

    ax6.plot(t_arr / T, coherence, 'b-', lw=2, label='|rho_01| (coherence)')
    ax6.plot(t_arr / T, np.array(phase_evolution) / np.pi, 'r--', lw=2,
             label='arg(rho_01) / pi')

    ax6.set_xlabel('Time t / T')
    ax6.set_ylabel('Value')
    ax6.set_title('Coherence and Phase Evolution')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Add formula
    ax6.text(0.5, 0.95, r'$\omega_L = \gamma B$ (Larmor frequency)',
             transform=ax6.transAxes, ha='center', va='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle('Larmor Precession of Spin-1/2\n'
                 r'$H = -\gamma \mathbf{B} \cdot \mathbf{S}$',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'larmor_precession.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'larmor_precession.png')}")


if __name__ == "__main__":
    main()
