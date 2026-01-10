"""
Experiment 93: Coupled RLC normal modes.

This example demonstrates the normal modes of coupled RLC circuits,
showing how two oscillators exchange energy and exhibit beating patterns.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def coupled_rlc_equations(y, t, R1, L1, C1, R2, L2, C2, M):
    """
    Equations of motion for two magnetically coupled RLC circuits.

    Circuit 1: L1, R1, C1
    Circuit 2: L2, R2, C2
    Mutual inductance: M

    State vector: [q1, dq1/dt, q2, dq2/dt] = [q1, I1, q2, I2]

    L1 * dI1/dt + M * dI2/dt + R1*I1 + q1/C1 = 0
    L2 * dI2/dt + M * dI1/dt + R2*I2 + q2/C2 = 0
    """
    q1, I1, q2, I2 = y

    # Solve the coupled equations for dI1/dt and dI2/dt
    # [L1  M ] [dI1/dt]   [-R1*I1 - q1/C1]
    # [M   L2] [dI2/dt] = [-R2*I2 - q2/C2]

    det = L1 * L2 - M * M
    if abs(det) < 1e-20:
        det = 1e-20  # Avoid division by zero

    rhs1 = -R1 * I1 - q1 / C1
    rhs2 = -R2 * I2 - q2 / C2

    dI1_dt = (L2 * rhs1 - M * rhs2) / det
    dI2_dt = (L1 * rhs2 - M * rhs1) / det

    return [I1, dI1_dt, I2, dI2_dt]


def calculate_normal_mode_frequencies(L1, C1, L2, C2, M):
    """
    Calculate normal mode frequencies for coupled LC circuits.

    For identical circuits (L1=L2=L, C1=C2=C):
    omega_symmetric = 1/sqrt((L-M)*C)    (currents in phase)
    omega_antisymmetric = 1/sqrt((L+M)*C) (currents out of phase)
    """
    # General case eigenvalue problem
    omega1_sq = 1 / (L1 * C1)
    omega2_sq = 1 / (L2 * C2)
    k = M / np.sqrt(L1 * L2)  # Coupling coefficient

    # For coupled oscillators, the normal mode frequencies are:
    avg_omega_sq = (omega1_sq + omega2_sq) / 2
    delta_omega_sq = np.sqrt((omega1_sq - omega2_sq)**2 / 4 +
                             k**2 * omega1_sq * omega2_sq)

    omega_plus = np.sqrt(avg_omega_sq + delta_omega_sq)
    omega_minus = np.sqrt(max(0, avg_omega_sq - delta_omega_sq))

    return omega_minus, omega_plus


def main():
    # Identical circuit parameters for clearer normal modes
    L = 10e-3       # 10 mH inductance
    R = 1.0         # 1 Ohm resistance (low for underdamped)
    C = 1e-6        # 1 uF capacitance

    # Resonant frequency (uncoupled)
    omega_0 = 1 / np.sqrt(L * C)
    f_0 = omega_0 / (2 * np.pi)

    fig = plt.figure(figsize=(16, 12))

    # Coupling coefficients to test
    k_values = [0, 0.1, 0.3, 0.5]

    # Plot 1: Time evolution for different coupling strengths
    ax1 = fig.add_subplot(2, 2, 1)

    t = np.linspace(0, 0.02, 2000)  # 20 ms

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(k_values)))

    for k, color in zip(k_values, colors):
        M = k * L  # Mutual inductance

        # Initial conditions: charge circuit 1 only
        q0 = 1e-6  # 1 uC initial charge
        y0 = [q0, 0, 0, 0]  # [q1, I1, q2, I2]

        # Solve
        sol = odeint(coupled_rlc_equations, y0, t,
                     args=(R, L, C, R, L, C, M))

        q1 = sol[:, 0]
        q2 = sol[:, 2]

        ax1.plot(t * 1000, q1 * 1e6, color=color, lw=1.5,
                 label=f'Circuit 1, k={k}')

    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Charge q1 (uC)')
    ax1.set_title('Charge on Circuit 1 vs Coupling Strength')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Energy exchange with moderate coupling
    ax2 = fig.add_subplot(2, 2, 2)

    k = 0.3
    M = k * L

    y0 = [1e-6, 0, 0, 0]
    sol = odeint(coupled_rlc_equations, y0, t, args=(R, L, C, R, L, C, M))

    q1, I1, q2, I2 = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3]

    # Energy in each circuit
    E1 = 0.5 * L * I1**2 + 0.5 * q1**2 / C
    E2 = 0.5 * L * I2**2 + 0.5 * q2**2 / C
    E_total = E1 + E2 + M * I1 * I2  # Include mutual inductance energy

    ax2.plot(t * 1000, E1 * 1e9, 'b-', lw=2, label='Energy in Circuit 1')
    ax2.plot(t * 1000, E2 * 1e9, 'r-', lw=2, label='Energy in Circuit 2')
    ax2.plot(t * 1000, E_total * 1e9, 'k--', lw=1, label='Total Energy')

    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Energy (nJ)')
    ax2.set_title(f'Energy Exchange (k = {k})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Beat frequency annotation
    omega_minus, omega_plus = calculate_normal_mode_frequencies(L, C, L, C, M)
    f_beat = np.abs(omega_plus - omega_minus) / (2 * np.pi)
    T_beat = 1 / f_beat if f_beat > 0 else float('inf')
    ax2.text(0.95, 0.95, f'Beat period: {T_beat*1000:.2f} ms',
             transform=ax2.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 3: Normal mode frequencies
    ax3 = fig.add_subplot(2, 2, 3)

    k_range = np.linspace(0, 0.9, 100)
    omega_minus_arr = []
    omega_plus_arr = []

    for k_val in k_range:
        M_val = k_val * L
        om, op = calculate_normal_mode_frequencies(L, C, L, C, M_val)
        omega_minus_arr.append(om)
        omega_plus_arr.append(op)

    omega_minus_arr = np.array(omega_minus_arr)
    omega_plus_arr = np.array(omega_plus_arr)

    ax3.plot(k_range, omega_minus_arr / omega_0, 'b-', lw=2,
             label='Symmetric mode (omega-)')
    ax3.plot(k_range, omega_plus_arr / omega_0, 'r-', lw=2,
             label='Antisymmetric mode (omega+)')
    ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5,
                label='Uncoupled frequency')

    ax3.set_xlabel('Coupling coefficient k = M/L')
    ax3.set_ylabel('omega / omega_0')
    ax3.set_title('Normal Mode Frequencies vs Coupling')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Phase space trajectories
    ax4 = fig.add_subplot(2, 2, 4)

    k = 0.3
    M = k * L

    # Longer simulation for phase space
    t_long = np.linspace(0, 0.05, 5000)

    # Different initial conditions
    initial_conditions = [
        ([1e-6, 0, 0, 0], 'Circuit 1 excited', 'blue'),
        ([1e-6, 0, 1e-6, 0], 'Both in phase', 'green'),
        ([1e-6, 0, -1e-6, 0], 'Out of phase', 'red'),
    ]

    for y0, label, color in initial_conditions:
        sol = odeint(coupled_rlc_equations, y0, t_long,
                     args=(R, L, C, R, L, C, M))
        q1 = sol[:, 0]
        q2 = sol[:, 2]

        ax4.plot(q1 * 1e6, q2 * 1e6, color=color, lw=0.5, alpha=0.7, label=label)

    ax4.set_xlabel('q1 (uC)')
    ax4.set_ylabel('q2 (uC)')
    ax4.set_title('Phase Space: q1 vs q2')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal')

    # Add parameter summary
    fig.text(0.5, 0.02,
             f'Coupled RLC Circuits: L = {L*1000:.0f} mH, C = {C*1e6:.0f} uF, '
             f'R = {R} ohm\n'
             f'Uncoupled resonance: f_0 = {f_0:.0f} Hz, '
             r'Coupling: k = M/L, $\omega_{\pm} = \omega_0/\sqrt{1 \mp k}$',
             ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.suptitle('Coupled RLC Circuit Normal Modes', fontsize=14, y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)

    # Save plot
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'coupled_rlc_modes.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
