"""
Experiment 48: Coupled Pendula - Normal Modes

This example demonstrates coupled pendulum systems and their normal modes.
Shows how two pendulums connected by a spring exhibit two distinct normal
modes (in-phase and out-of-phase) and beating behavior.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh


class CoupledPendula:
    """
    Two identical simple pendulums coupled by a spring.

    Equations of motion (small angle approximation):
    m*L^2*d^2(theta1)/dt^2 = -m*g*L*theta1 - k*d^2*(theta1 - theta2)
    m*L^2*d^2(theta2)/dt^2 = -m*g*L*theta2 - k*d^2*(theta2 - theta1)

    where d is the distance from pivot where spring attaches.
    """

    def __init__(self, m, L, g, k, d, theta1_0, theta2_0, omega1_0=0, omega2_0=0):
        """
        Initialize coupled pendula.

        Args:
            m: Mass of each pendulum bob
            L: Length of each pendulum
            g: Gravitational acceleration
            k: Spring constant
            d: Spring attachment distance from pivot
            theta1_0, theta2_0: Initial angles
            omega1_0, omega2_0: Initial angular velocities
        """
        self.m = m
        self.L = L
        self.g = g
        self.k = k
        self.d = d

        self.theta1 = theta1_0
        self.theta2 = theta2_0
        self.omega1 = omega1_0
        self.omega2 = omega2_0

        # Natural frequency of single pendulum
        self.omega_0 = np.sqrt(g / L)

        # Coupling strength
        self.kappa = k * d**2 / (m * L**2)

        # Normal mode frequencies
        self.omega_plus = self.omega_0  # In-phase mode
        self.omega_minus = np.sqrt(self.omega_0**2 + 2 * self.kappa)  # Out-of-phase mode

    def accelerations(self, theta1, theta2, omega1, omega2):
        """Calculate angular accelerations."""
        alpha1 = -self.omega_0**2 * theta1 - self.kappa * (theta1 - theta2)
        alpha2 = -self.omega_0**2 * theta2 - self.kappa * (theta2 - theta1)
        return alpha1, alpha2

    def update(self, dt):
        """Update system state using RK4."""
        # Current state
        theta1, theta2 = self.theta1, self.theta2
        omega1, omega2 = self.omega1, self.omega2

        # RK4 k1
        alpha1, alpha2 = self.accelerations(theta1, theta2, omega1, omega2)
        k1_theta1, k1_theta2 = omega1, omega2
        k1_omega1, k1_omega2 = alpha1, alpha2

        # RK4 k2
        theta1_temp = theta1 + 0.5 * dt * k1_theta1
        theta2_temp = theta2 + 0.5 * dt * k1_theta2
        omega1_temp = omega1 + 0.5 * dt * k1_omega1
        omega2_temp = omega2 + 0.5 * dt * k1_omega2
        alpha1, alpha2 = self.accelerations(theta1_temp, theta2_temp, omega1_temp, omega2_temp)
        k2_theta1, k2_theta2 = omega1_temp, omega2_temp
        k2_omega1, k2_omega2 = alpha1, alpha2

        # RK4 k3
        theta1_temp = theta1 + 0.5 * dt * k2_theta1
        theta2_temp = theta2 + 0.5 * dt * k2_theta2
        omega1_temp = omega1 + 0.5 * dt * k2_omega1
        omega2_temp = omega2 + 0.5 * dt * k2_omega2
        alpha1, alpha2 = self.accelerations(theta1_temp, theta2_temp, omega1_temp, omega2_temp)
        k3_theta1, k3_theta2 = omega1_temp, omega2_temp
        k3_omega1, k3_omega2 = alpha1, alpha2

        # RK4 k4
        theta1_temp = theta1 + dt * k3_theta1
        theta2_temp = theta2 + dt * k3_theta2
        omega1_temp = omega1 + dt * k3_omega1
        omega2_temp = omega2 + dt * k3_omega2
        alpha1, alpha2 = self.accelerations(theta1_temp, theta2_temp, omega1_temp, omega2_temp)
        k4_theta1, k4_theta2 = omega1_temp, omega2_temp
        k4_omega1, k4_omega2 = alpha1, alpha2

        # Update
        self.theta1 += (dt / 6) * (k1_theta1 + 2*k2_theta1 + 2*k3_theta1 + k4_theta1)
        self.theta2 += (dt / 6) * (k1_theta2 + 2*k2_theta2 + 2*k3_theta2 + k4_theta2)
        self.omega1 += (dt / 6) * (k1_omega1 + 2*k2_omega1 + 2*k3_omega1 + k4_omega1)
        self.omega2 += (dt / 6) * (k1_omega2 + 2*k2_omega2 + 2*k3_omega2 + k4_omega2)


def simulate_coupled_pendula(params, theta1_0, theta2_0, omega1_0, omega2_0, t_final, dt):
    """
    Simulate coupled pendula motion.

    Args:
        params: Dictionary with m, L, g, k, d
        theta1_0, theta2_0: Initial angles
        omega1_0, omega2_0: Initial angular velocities
        t_final: Simulation duration
        dt: Time step

    Returns:
        Dictionary with time and angle data
    """
    pend = CoupledPendula(params['m'], params['L'], params['g'],
                          params['k'], params['d'],
                          theta1_0, theta2_0, omega1_0, omega2_0)

    times = [0]
    theta1s = [theta1_0]
    theta2s = [theta2_0]
    omega1s = [omega1_0]
    omega2s = [omega2_0]

    t = 0
    while t < t_final:
        pend.update(dt)
        t += dt
        times.append(t)
        theta1s.append(pend.theta1)
        theta2s.append(pend.theta2)
        omega1s.append(pend.omega1)
        omega2s.append(pend.omega2)

    return {
        'time': np.array(times),
        'theta1': np.array(theta1s),
        'theta2': np.array(theta2s),
        'omega1': np.array(omega1s),
        'omega2': np.array(omega2s),
        'omega_plus': pend.omega_plus,
        'omega_minus': pend.omega_minus
    }


def analytical_solution(params, theta1_0, theta2_0, t_array):
    """
    Analytical solution for coupled pendula (small angle approximation).

    Normal mode decomposition:
    q_+ = (theta1 + theta2) / 2  (in-phase mode)
    q_- = (theta1 - theta2) / 2  (out-of-phase mode)
    """
    omega_0 = np.sqrt(params['g'] / params['L'])
    kappa = params['k'] * params['d']**2 / (params['m'] * params['L']**2)

    omega_plus = omega_0
    omega_minus = np.sqrt(omega_0**2 + 2 * kappa)

    # Initial conditions in normal mode coordinates
    q_plus_0 = (theta1_0 + theta2_0) / 2
    q_minus_0 = (theta1_0 - theta2_0) / 2

    # Time evolution
    q_plus = q_plus_0 * np.cos(omega_plus * t_array)
    q_minus = q_minus_0 * np.cos(omega_minus * t_array)

    # Back to original coordinates
    theta1_theory = q_plus + q_minus
    theta2_theory = q_plus - q_minus

    return theta1_theory, theta2_theory


def main():
    # Physical parameters
    params = {
        'm': 1.0,    # kg
        'L': 1.0,    # m
        'g': 9.81,   # m/s^2
        'k': 5.0,    # N/m (spring constant)
        'd': 0.5     # m (spring attachment point)
    }

    dt = 0.001
    t_final = 30.0

    # Create figure
    fig = plt.figure(figsize=(16, 14))

    # Case 1: In-phase normal mode (symmetric mode)
    ax1 = fig.add_subplot(3, 3, 1)

    theta0 = 0.2  # rad
    results_in = simulate_coupled_pendula(params, theta0, theta0, 0, 0, t_final, dt)

    ax1.plot(results_in['time'], np.degrees(results_in['theta1']), 'b-', lw=2, label='Pendulum 1')
    ax1.plot(results_in['time'], np.degrees(results_in['theta2']), 'r--', lw=2, label='Pendulum 2')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Angle (degrees)')
    ax1.set_title(f'In-Phase Mode: T = {2*np.pi/results_in["omega_plus"]:.3f}s')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Case 2: Out-of-phase normal mode (antisymmetric mode)
    ax2 = fig.add_subplot(3, 3, 2)

    results_out = simulate_coupled_pendula(params, theta0, -theta0, 0, 0, t_final, dt)

    ax2.plot(results_out['time'], np.degrees(results_out['theta1']), 'b-', lw=2, label='Pendulum 1')
    ax2.plot(results_out['time'], np.degrees(results_out['theta2']), 'r--', lw=2, label='Pendulum 2')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Angle (degrees)')
    ax2.set_title(f'Out-of-Phase Mode: T = {2*np.pi/results_out["omega_minus"]:.3f}s')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Case 3: Beating (one pendulum initially displaced)
    ax3 = fig.add_subplot(3, 3, 3)

    results_beat = simulate_coupled_pendula(params, theta0, 0, 0, 0, 60.0, dt)

    ax3.plot(results_beat['time'], np.degrees(results_beat['theta1']), 'b-', lw=1, label='Pendulum 1')
    ax3.plot(results_beat['time'], np.degrees(results_beat['theta2']), 'r-', lw=1, label='Pendulum 2')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Angle (degrees)')

    # Calculate beat frequency
    omega_beat = abs(results_beat['omega_plus'] - results_beat['omega_minus']) / 2
    T_beat = 2 * np.pi / omega_beat if omega_beat > 0 else np.inf
    ax3.set_title(f'Beating: T_beat = {T_beat:.2f}s')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Case 4: Normal mode frequencies vs coupling strength
    ax4 = fig.add_subplot(3, 3, 4)

    k_values = np.linspace(0, 20, 100)
    omega_0 = np.sqrt(params['g'] / params['L'])

    omega_plus_vals = np.ones_like(k_values) * omega_0
    omega_minus_vals = []

    for k in k_values:
        kappa = k * params['d']**2 / (params['m'] * params['L']**2)
        omega_minus_vals.append(np.sqrt(omega_0**2 + 2 * kappa))

    ax4.plot(k_values, omega_plus_vals, 'b-', lw=2, label='In-phase mode (omega_+)')
    ax4.plot(k_values, omega_minus_vals, 'r-', lw=2, label='Out-of-phase mode (omega_-)')
    ax4.axvline(x=params['k'], color='g', linestyle='--', alpha=0.5,
                label=f'Current k = {params["k"]}')
    ax4.set_xlabel('Spring constant k (N/m)')
    ax4.set_ylabel('Angular frequency (rad/s)')
    ax4.set_title('Normal Mode Frequencies vs Coupling')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Case 5: Energy exchange during beating
    ax5 = fig.add_subplot(3, 3, 5)

    # Calculate energies
    m, L, g = params['m'], params['L'], params['g']
    KE1 = 0.5 * m * L**2 * results_beat['omega1']**2
    KE2 = 0.5 * m * L**2 * results_beat['omega2']**2
    PE1 = 0.5 * m * g * L * results_beat['theta1']**2  # Small angle
    PE2 = 0.5 * m * g * L * results_beat['theta2']**2

    E1 = KE1 + PE1
    E2 = KE2 + PE2

    ax5.plot(results_beat['time'], E1, 'b-', lw=1.5, label='Pendulum 1 energy')
    ax5.plot(results_beat['time'], E2, 'r-', lw=1.5, label='Pendulum 2 energy')
    ax5.plot(results_beat['time'], E1 + E2, 'g--', lw=1.5, label='Total energy')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Energy (J)')
    ax5.set_title('Energy Exchange During Beating')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Case 6: Phase space for both pendulums
    ax6 = fig.add_subplot(3, 3, 6)

    ax6.plot(np.degrees(results_beat['theta1']), results_beat['omega1'],
             'b-', lw=0.5, alpha=0.7, label='Pendulum 1')
    ax6.plot(np.degrees(results_beat['theta2']), results_beat['omega2'],
             'r-', lw=0.5, alpha=0.7, label='Pendulum 2')
    ax6.set_xlabel('Angle (degrees)')
    ax6.set_ylabel('Angular velocity (rad/s)')
    ax6.set_title('Phase Space (Beating)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Case 7: Comparison with analytical solution
    ax7 = fig.add_subplot(3, 3, 7)

    t_arr = results_beat['time']
    theta1_theory, theta2_theory = analytical_solution(params, theta0, 0, t_arr)

    ax7.plot(t_arr, np.degrees(results_beat['theta1']), 'b-', lw=2, label='Numerical (P1)')
    ax7.plot(t_arr, np.degrees(theta1_theory), 'b--', lw=1, alpha=0.7, label='Analytical (P1)')
    ax7.plot(t_arr, np.degrees(results_beat['theta2']), 'r-', lw=2, label='Numerical (P2)')
    ax7.plot(t_arr, np.degrees(theta2_theory), 'r--', lw=1, alpha=0.7, label='Analytical (P2)')
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('Angle (degrees)')
    ax7.set_title('Comparison: Numerical vs Analytical')
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)
    ax7.set_xlim(0, 30)

    # Case 8: Mode visualization
    ax8 = fig.add_subplot(3, 3, 8)

    # Draw the normal modes
    mode_x = [0, 1]
    mode_in = [1, 1]  # In-phase
    mode_out = [1, -1]  # Out-of-phase

    ax8.bar([x - 0.15 for x in mode_x], mode_in, width=0.3, color='blue',
            alpha=0.7, label='In-phase mode')
    ax8.bar([x + 0.15 for x in mode_x], mode_out, width=0.3, color='red',
            alpha=0.7, label='Out-of-phase mode')
    ax8.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax8.set_xticks([0, 1])
    ax8.set_xticklabels(['Pendulum 1', 'Pendulum 2'])
    ax8.set_ylabel('Relative Amplitude')
    ax8.set_title('Normal Mode Shapes')
    ax8.legend()
    ax8.grid(True, alpha=0.3, axis='y')

    # Case 9: Frequency spectrum of beating
    ax9 = fig.add_subplot(3, 3, 9)

    # FFT of pendulum 1 during beating
    theta1_signal = results_beat['theta1']
    N = len(theta1_signal)
    fft_vals = np.abs(np.fft.rfft(theta1_signal))
    freqs = np.fft.rfftfreq(N, dt)

    ax9.semilogy(freqs[:N//4], fft_vals[:N//4], 'b-', lw=1.5)

    # Mark normal mode frequencies
    f_plus = results_beat['omega_plus'] / (2 * np.pi)
    f_minus = results_beat['omega_minus'] / (2 * np.pi)
    ax9.axvline(x=f_plus, color='g', linestyle='--',
                label=f'f+ = {f_plus:.3f} Hz')
    ax9.axvline(x=f_minus, color='r', linestyle='--',
                label=f'f- = {f_minus:.3f} Hz')
    ax9.set_xlabel('Frequency (Hz)')
    ax9.set_ylabel('Power')
    ax9.set_title('Frequency Spectrum (Beating)')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    ax9.set_xlim(0, 1.5)

    plt.suptitle('Coupled Pendula: Normal Modes and Beating\n'
                 f'm={params["m"]}kg, L={params["L"]}m, k={params["k"]}N/m',
                 fontsize=12, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'coupled_pendula.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'coupled_pendula.png')}")

    # Print summary
    omega_0 = np.sqrt(params['g'] / params['L'])
    kappa = params['k'] * params['d']**2 / (params['m'] * params['L']**2)
    omega_plus = omega_0
    omega_minus = np.sqrt(omega_0**2 + 2 * kappa)

    print("\nCoupled Pendula Summary:")
    print("=" * 50)
    print(f"Parameters: m={params['m']}kg, L={params['L']}m, g={params['g']}m/s^2")
    print(f"            k={params['k']}N/m, d={params['d']}m")
    print(f"\nNatural frequency of single pendulum: omega_0 = {omega_0:.4f} rad/s")
    print(f"Coupling parameter: kappa = {kappa:.4f} rad^2/s^2")
    print(f"\nNormal mode frequencies:")
    print(f"  In-phase mode:     omega_+ = {omega_plus:.4f} rad/s, T_+ = {2*np.pi/omega_plus:.4f} s")
    print(f"  Out-of-phase mode: omega_- = {omega_minus:.4f} rad/s, T_- = {2*np.pi/omega_minus:.4f} s")
    print(f"\nBeat frequency: omega_beat = {abs(omega_plus - omega_minus)/2:.4f} rad/s")
    print(f"Beat period: T_beat = {2*np.pi/(abs(omega_plus - omega_minus)/2):.4f} s")


if __name__ == "__main__":
    main()
