"""
Experiment 163: Rabi Oscillations

This experiment demonstrates Rabi oscillations in a two-level quantum system
driven by a resonant or near-resonant field, including:
- On-resonance Rabi oscillations
- Off-resonance (detuned) behavior
- Generalized Rabi frequency
- Population inversion
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)


def two_level_hamiltonian(omega_0: float, Omega: float, delta: float, t: float,
                          hbar: float = 1.0) -> np.ndarray:
    """
    Create Hamiltonian for driven two-level system.

    H = (hbar*omega_0/2) * sigma_z + hbar*Omega * cos(omega_d*t) * sigma_x

    In rotating wave approximation:
    H_RWA = (hbar*delta/2) * sigma_z + (hbar*Omega/2) * sigma_x

    Args:
        omega_0: Transition frequency
        Omega: Rabi frequency (coupling strength)
        delta: Detuning (omega_d - omega_0)
        t: Time (unused in RWA)
        hbar: Reduced Planck constant

    Returns:
        2x2 Hamiltonian matrix (in RWA)
    """
    return hbar * (delta / 2 * sigma_z + Omega / 2 * sigma_x)


def generalized_rabi_frequency(Omega: float, delta: float) -> float:
    """
    Calculate generalized Rabi frequency.

    Omega_R = sqrt(Omega^2 + delta^2)
    """
    return np.sqrt(Omega**2 + delta**2)


def rabi_dynamics_analytical(t: np.ndarray, Omega: float, delta: float,
                              psi0: np.ndarray) -> tuple:
    """
    Analytical solution for Rabi oscillations (RWA).

    Returns excited state population and ground state population vs time.

    Args:
        t: Time array
        Omega: Rabi frequency
        delta: Detuning
        psi0: Initial state [c_g, c_e]

    Returns:
        Tuple of (P_excited, P_ground) arrays
    """
    Omega_R = generalized_rabi_frequency(Omega, delta)

    c_g0, c_e0 = psi0[0], psi0[1]
    P_g0 = np.abs(c_g0)**2
    P_e0 = np.abs(c_e0)**2

    # For starting in ground state:
    if P_g0 > 0.99:
        # P_e(t) = (Omega/Omega_R)^2 * sin^2(Omega_R * t / 2)
        P_e = (Omega / Omega_R)**2 * np.sin(Omega_R * t / 2)**2
        P_g = 1 - P_e
    else:
        # General case - use numerical solution
        # Simplified: assume starting in ground state
        P_e = (Omega / Omega_R)**2 * np.sin(Omega_R * t / 2)**2
        P_g = 1 - P_e

    return P_e, P_g


def rabi_dynamics_numerical(t_span: tuple, Omega: float, delta: float,
                            psi0: np.ndarray, hbar: float = 1.0) -> tuple:
    """
    Numerical solution of Schrodinger equation for Rabi oscillations.

    Args:
        t_span: (t_start, t_end)
        Omega: Rabi frequency
        delta: Detuning
        psi0: Initial state
        hbar: Reduced Planck constant

    Returns:
        Tuple of (times, psi_array)
    """
    H = two_level_hamiltonian(0, Omega, delta, 0, hbar)

    def schrodinger_rhs(t, y):
        psi = y[:2] + 1j * y[2:]
        dpsi_dt = -1j / hbar * H @ psi
        return np.concatenate([np.real(dpsi_dt), np.imag(dpsi_dt)])

    y0 = np.concatenate([np.real(psi0), np.imag(psi0)])

    sol = solve_ivp(schrodinger_rhs, t_span, y0, method='RK45',
                    dense_output=True, max_step=0.01)

    return sol.t, sol.y[:2] + 1j * sol.y[2:]


def main():
    # Parameters
    hbar = 1.0
    Omega = 1.0  # Rabi frequency

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Plot 1: On-resonance Rabi oscillations
    ax1 = axes[0, 0]

    delta = 0  # On resonance
    psi0 = np.array([1.0, 0.0], dtype=complex)  # Start in ground state

    T_Rabi = 2 * np.pi / Omega
    t = np.linspace(0, 4 * T_Rabi, 500)

    P_e, P_g = rabi_dynamics_analytical(t, Omega, delta, psi0)

    ax1.plot(t / T_Rabi, P_g, 'b-', lw=2, label='P(ground)')
    ax1.plot(t / T_Rabi, P_e, 'r-', lw=2, label='P(excited)')

    ax1.set_xlabel('Time t / T_Rabi')
    ax1.set_ylabel('Population')
    ax1.set_title(f'On-Resonance Rabi Oscillations\ndelta = 0, Omega = {Omega}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Mark pi and 2pi pulses
    ax1.axvline(x=0.5, color='green', linestyle='--', alpha=0.5)
    ax1.text(0.5, 0.05, 'pi pulse', rotation=90, va='bottom', fontsize=9)
    ax1.axvline(x=1.0, color='green', linestyle='--', alpha=0.5)
    ax1.text(1.0, 0.05, '2pi pulse', rotation=90, va='bottom', fontsize=9)

    # Plot 2: Off-resonance oscillations
    ax2 = axes[0, 1]

    delta_values = [0, 0.5, 1.0, 2.0]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(delta_values)))

    for delta, color in zip(delta_values, colors):
        Omega_R = generalized_rabi_frequency(Omega, delta)
        P_e, _ = rabi_dynamics_analytical(t, Omega, delta, psi0)
        ax2.plot(t / T_Rabi, P_e, color=color, lw=2,
                label=f'delta = {delta}, Omega_R = {Omega_R:.2f}')

    ax2.set_xlabel('Time t / T_Rabi')
    ax2.set_ylabel('Excited State Population')
    ax2.set_title('Effect of Detuning')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Maximum excitation vs detuning
    ax3 = axes[0, 2]

    delta_range = np.linspace(-3, 3, 100) * Omega
    P_max = (Omega / generalized_rabi_frequency(Omega, delta_range))**2

    ax3.plot(delta_range / Omega, P_max, 'b-', lw=2)
    ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
    ax3.axvline(x=0, color='gray', linestyle=':', alpha=0.5)

    ax3.set_xlabel('Detuning delta / Omega')
    ax3.set_ylabel('Maximum Excitation Probability')
    ax3.set_title('Resonance Lineshape\n(Lorentzian)')
    ax3.grid(True, alpha=0.3)

    # FWHM
    fwhm = 2 * Omega
    ax3.annotate('', xy=(-1, 0.5), xytext=(1, 0.5),
                 arrowprops=dict(arrowstyle='<->', color='red'))
    ax3.text(0, 0.55, f'FWHM = 2*Omega', ha='center', color='red')

    # Plot 4: Bloch sphere trajectory
    ax4 = axes[1, 0]

    # On resonance - rotation about x-axis
    delta = 0
    t_bloch = np.linspace(0, 2*T_Rabi, 200)

    # Bloch vector components
    # u = <sigma_x>, v = <sigma_y>, w = <sigma_z>
    u = np.zeros_like(t_bloch)
    v = -np.sin(Omega * t_bloch)
    w = np.cos(Omega * t_bloch)

    ax4.plot(t_bloch / T_Rabi, u, 'r-', lw=2, label='<sigma_x>')
    ax4.plot(t_bloch / T_Rabi, v, 'g-', lw=2, label='<sigma_y>')
    ax4.plot(t_bloch / T_Rabi, w, 'b-', lw=2, label='<sigma_z>')

    ax4.set_xlabel('Time t / T_Rabi')
    ax4.set_ylabel('Bloch Vector Component')
    ax4.set_title('Bloch Vector Rotation (On Resonance)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Generalized Rabi frequency
    ax5 = axes[1, 1]

    Omega_values = [0.5, 1.0, 2.0]
    colors_Omega = plt.cm.plasma(np.linspace(0.2, 0.9, len(Omega_values)))

    for Omega_test, color in zip(Omega_values, colors_Omega):
        Omega_R = generalized_rabi_frequency(Omega_test, delta_range)
        ax5.plot(delta_range / Omega_test, Omega_R / Omega_test, color=color, lw=2,
                label=f'Omega = {Omega_test}')

    ax5.plot(delta_range / Omega, np.abs(delta_range) / Omega, 'k--', lw=1.5,
             alpha=0.5, label='|delta|')

    ax5.set_xlabel('Detuning delta / Omega')
    ax5.set_ylabel('Omega_R / Omega')
    ax5.set_title('Generalized Rabi Frequency\nOmega_R = sqrt(Omega^2 + delta^2)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Pulse area concept
    ax6 = axes[1, 2]

    # Different pulse areas
    t_pulse = np.linspace(0, 4, 200)

    # Rectangular pulse with area theta
    areas = [np.pi/4, np.pi/2, np.pi, 2*np.pi]
    colors_area = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(areas)))

    for area, color in zip(areas, colors_area):
        # Pulse duration such that Omega * t = area
        Omega_eff = area / 1.0  # 1 unit pulse duration
        delta = 0

        # Final population after pulse
        P_e_final = (Omega_eff / Omega_eff)**2 * np.sin(area / 2)**2

        # Time evolution during pulse
        t_evo = np.linspace(0, 1, 100)
        P_e_t = np.sin(Omega_eff * t_evo / 2)**2

        ax6.plot(t_evo, P_e_t, color=color, lw=2,
                label=f'theta = {area/np.pi:.2f}*pi, P_e = {P_e_final:.2f}')

    ax6.set_xlabel('Time (normalized)')
    ax6.set_ylabel('Excited State Population')
    ax6.set_title('Pulse Area: theta = integral(Omega dt)\n'
                  'pi pulse: complete inversion')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.suptitle('Rabi Oscillations in Two-Level System\n'
                 r'$H = (\hbar\delta/2)\sigma_z + (\hbar\Omega/2)\sigma_x$ (RWA)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'rabi_oscillations.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'rabi_oscillations.png')}")


if __name__ == "__main__":
    main()
