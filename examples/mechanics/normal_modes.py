"""
Experiment 38: Small Oscillations Around Equilibrium - Normal Modes

This example demonstrates normal mode analysis for coupled oscillator systems.
Shows how to find normal modes from the eigenvalue problem and how arbitrary
initial conditions decompose into a superposition of normal modes.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh


def build_system_matrices(n_masses, k_values, m_values, k_wall=None):
    """
    Build mass and stiffness matrices for a chain of coupled oscillators.

    Args:
        n_masses: Number of masses
        k_values: List of spring constants (length n_masses+1 for walls on both ends)
        m_values: List of masses
        k_wall: Spring constant for wall springs (if None, uses first/last k_values)

    Returns:
        M: Mass matrix
        K: Stiffness matrix
    """
    M = np.diag(m_values)
    K = np.zeros((n_masses, n_masses))

    for i in range(n_masses):
        # Diagonal terms: sum of springs connected to mass i
        if i == 0:
            K[i, i] = k_values[0] + k_values[1]  # left wall + right spring
        elif i == n_masses - 1:
            K[i, i] = k_values[i] + k_values[i+1]  # left spring + right wall
        else:
            K[i, i] = k_values[i] + k_values[i+1]  # left + right springs

        # Off-diagonal terms: coupling springs
        if i < n_masses - 1:
            K[i, i+1] = -k_values[i+1]
            K[i+1, i] = -k_values[i+1]

    return M, K


def find_normal_modes(M, K):
    """
    Find normal modes by solving the generalized eigenvalue problem.
    K @ v = omega^2 * M @ v

    Returns:
        frequencies: Normal mode frequencies
        modes: Normal mode vectors (columns)
    """
    # Solve generalized eigenvalue problem
    eigenvalues, eigenvectors = eigh(K, M)

    # Frequencies are sqrt of eigenvalues
    frequencies = np.sqrt(np.abs(eigenvalues))

    return frequencies, eigenvectors


def simulate_coupled_oscillators(M, K, x0, v0, t_final, dt):
    """
    Simulate coupled oscillator dynamics.

    Args:
        M: Mass matrix
        K: Stiffness matrix
        x0: Initial displacements
        v0: Initial velocities
        t_final: Simulation duration
        dt: Time step

    Returns:
        Dictionary with time and position data
    """
    n = len(x0)
    M_inv = np.linalg.inv(M)

    # State vector: [x, v]
    x = x0.copy()
    v = v0.copy()

    times = [0]
    positions = [x.copy()]
    velocities = [v.copy()]

    t = 0
    while t < t_final:
        # RK4 integration
        # dx/dt = v
        # dv/dt = -M^(-1) K x

        k1_x = v
        k1_v = -M_inv @ K @ x

        x_temp = x + 0.5 * dt * k1_x
        v_temp = v + 0.5 * dt * k1_v
        k2_x = v_temp
        k2_v = -M_inv @ K @ x_temp

        x_temp = x + 0.5 * dt * k2_x
        v_temp = v + 0.5 * dt * k2_v
        k3_x = v_temp
        k3_v = -M_inv @ K @ x_temp

        x_temp = x + dt * k3_x
        v_temp = v + dt * k3_v
        k4_x = v_temp
        k4_v = -M_inv @ K @ x_temp

        x = x + (dt / 6) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
        v = v + (dt / 6) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        t += dt

        times.append(t)
        positions.append(x.copy())
        velocities.append(v.copy())

    return {
        'time': np.array(times),
        'positions': np.array(positions),
        'velocities': np.array(velocities)
    }


def main():
    # System 1: Two coupled oscillators (simplest case)
    n_masses_2 = 2
    k = 1.0  # Spring constant
    m = 1.0  # Mass
    k_values_2 = [k, k, k]  # wall - mass1 - mass2 - wall
    m_values_2 = [m, m]

    M2, K2 = build_system_matrices(n_masses_2, k_values_2, m_values_2)
    freq2, modes2 = find_normal_modes(M2, K2)

    # System 2: Three coupled oscillators
    n_masses_3 = 3
    k_values_3 = [k, k, k, k]
    m_values_3 = [m, m, m]

    M3, K3 = build_system_matrices(n_masses_3, k_values_3, m_values_3)
    freq3, modes3 = find_normal_modes(M3, K3)

    # Create figure
    fig = plt.figure(figsize=(16, 14))

    # Subplot 1: Normal mode shapes for 2-mass system
    ax1 = fig.add_subplot(3, 3, 1)
    x_pos = [0, 1]
    for i in range(2):
        mode = modes2[:, i]
        mode = mode / np.max(np.abs(mode))  # Normalize for display
        ax1.bar(np.array(x_pos) + i*0.2 - 0.1, mode, width=0.15,
                label=f'Mode {i+1}: f = {freq2[i]/(2*np.pi):.3f} Hz')
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax1.set_xlabel('Mass index')
    ax1.set_ylabel('Normalized displacement')
    ax1.set_title('Normal Modes: 2 Coupled Oscillators')
    ax1.legend()
    ax1.set_xticks([0, 1])
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Normal mode shapes for 3-mass system
    ax2 = fig.add_subplot(3, 3, 2)
    x_pos = [0, 1, 2]
    for i in range(3):
        mode = modes3[:, i]
        mode = mode / np.max(np.abs(mode))
        ax2.bar(np.array(x_pos) + i*0.2 - 0.2, mode, width=0.15,
                label=f'Mode {i+1}: f = {freq3[i]/(2*np.pi):.3f} Hz')
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Mass index')
    ax2.set_ylabel('Normalized displacement')
    ax2.set_title('Normal Modes: 3 Coupled Oscillators')
    ax2.legend(fontsize=8)
    ax2.set_xticks([0, 1, 2])
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Frequency vs mode number
    ax3 = fig.add_subplot(3, 3, 3)
    ax3.plot([1, 2], freq2, 'bo-', markersize=10, lw=2, label='2-mass system')
    ax3.plot([1, 2, 3], freq3, 'rs-', markersize=10, lw=2, label='3-mass system')

    # Theoretical frequencies for identical masses and springs
    # omega_n = 2 * sqrt(k/m) * sin(n*pi / (2*(N+1)))
    for N, color in [(2, 'blue'), (3, 'red')]:
        omega_theory = [2 * np.sqrt(k/m) * np.sin(n * np.pi / (2*(N+1))) for n in range(1, N+1)]
        ax3.plot(range(1, N+1), omega_theory, '--', color=color, alpha=0.5)

    ax3.set_xlabel('Mode number n')
    ax3.set_ylabel('Angular frequency (rad/s)')
    ax3.set_title('Normal Mode Frequencies')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Subplot 4: Pure mode 1 oscillation (2-mass system)
    ax4 = fig.add_subplot(3, 3, 4)
    x0_mode1 = modes2[:, 0].copy()
    results_mode1 = simulate_coupled_oscillators(M2, K2, x0_mode1, np.zeros(2), 20.0, 0.01)

    for i in range(2):
        ax4.plot(results_mode1['time'], results_mode1['positions'][:, i],
                 lw=2, label=f'Mass {i+1}')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Displacement')
    ax4.set_title(f'Mode 1 Oscillation (in-phase)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Subplot 5: Pure mode 2 oscillation (2-mass system)
    ax5 = fig.add_subplot(3, 3, 5)
    x0_mode2 = modes2[:, 1].copy()
    results_mode2 = simulate_coupled_oscillators(M2, K2, x0_mode2, np.zeros(2), 20.0, 0.01)

    for i in range(2):
        ax5.plot(results_mode2['time'], results_mode2['positions'][:, i],
                 lw=2, label=f'Mass {i+1}')
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Displacement')
    ax5.set_title(f'Mode 2 Oscillation (out-of-phase)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Subplot 6: Beating - superposition of modes
    ax6 = fig.add_subplot(3, 3, 6)
    x0_beat = np.array([1.0, 0.0])  # Only first mass displaced
    results_beat = simulate_coupled_oscillators(M2, K2, x0_beat, np.zeros(2), 40.0, 0.01)

    for i in range(2):
        ax6.plot(results_beat['time'], results_beat['positions'][:, i],
                 lw=1.5, label=f'Mass {i+1}')
    ax6.set_xlabel('Time')
    ax6.set_ylabel('Displacement')
    ax6.set_title('Beating: Single Mass Initially Displaced')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Subplot 7: Mode decomposition visualization
    ax7 = fig.add_subplot(3, 3, 7)

    # Decompose initial condition into normal modes
    x0 = np.array([1.0, 0.0])
    # Coefficients: c_n = (mode_n^T @ M @ x0) / (mode_n^T @ M @ mode_n)
    coeffs = []
    for i in range(2):
        mode = modes2[:, i]
        c = (mode.T @ M2 @ x0) / (mode.T @ M2 @ mode)
        coeffs.append(c)

    # Time evolution of each mode contribution
    t_arr = results_beat['time']
    mode_contributions = []
    for i, c in enumerate(coeffs):
        contribution = c * np.cos(freq2[i] * t_arr)[:, np.newaxis] * modes2[:, i]
        mode_contributions.append(contribution)

    # Plot mode 1 contribution to mass 1
    ax7.plot(t_arr, mode_contributions[0][:, 0], 'b-', lw=2, alpha=0.7,
             label='Mode 1 contribution')
    ax7.plot(t_arr, mode_contributions[1][:, 0], 'r-', lw=2, alpha=0.7,
             label='Mode 2 contribution')
    ax7.plot(t_arr, results_beat['positions'][:, 0], 'k--', lw=1.5,
             label='Total (Mass 1)')
    ax7.set_xlabel('Time')
    ax7.set_ylabel('Displacement of Mass 1')
    ax7.set_title('Mode Decomposition')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # Subplot 8: Energy in each mode
    ax8 = fig.add_subplot(3, 3, 8)

    # Calculate energy in each mass
    pos = results_beat['positions']
    vel = results_beat['velocities']

    for i in range(2):
        KE = 0.5 * m * vel[:, i]**2
        ax8.plot(t_arr, KE, lw=2, label=f'KE Mass {i+1}')

    ax8.set_xlabel('Time')
    ax8.set_ylabel('Kinetic Energy')
    ax8.set_title('Energy Exchange Between Masses')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # Subplot 9: 3-mass system arbitrary initial condition
    ax9 = fig.add_subplot(3, 3, 9)
    x0_3 = np.array([1.0, 0.0, -0.5])  # Arbitrary initial condition
    results_3 = simulate_coupled_oscillators(M3, K3, x0_3, np.zeros(3), 30.0, 0.01)

    for i in range(3):
        ax9.plot(results_3['time'], results_3['positions'][:, i],
                 lw=1.5, label=f'Mass {i+1}')
    ax9.set_xlabel('Time')
    ax9.set_ylabel('Displacement')
    ax9.set_title('3-Mass System: Arbitrary Initial Condition')
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    plt.suptitle('Small Oscillations and Normal Modes\n'
                 'Coupled oscillator systems and mode decomposition',
                 fontsize=12, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'normal_modes.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'normal_modes.png')}")

    # Print mode information
    print("\n2-Mass System Normal Modes:")
    print(f"  Mode 1: omega = {freq2[0]:.4f} rad/s, T = {2*np.pi/freq2[0]:.4f} s")
    print(f"          Mode shape: {modes2[:, 0]}")
    print(f"  Mode 2: omega = {freq2[1]:.4f} rad/s, T = {2*np.pi/freq2[1]:.4f} s")
    print(f"          Mode shape: {modes2[:, 1]}")

    print("\n3-Mass System Normal Modes:")
    for i in range(3):
        print(f"  Mode {i+1}: omega = {freq3[i]:.4f} rad/s, T = {2*np.pi/freq3[i]:.4f} s")
        print(f"          Mode shape: {modes3[:, i]}")


if __name__ == "__main__":
    main()
