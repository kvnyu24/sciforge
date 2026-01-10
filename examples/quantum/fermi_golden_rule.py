"""
Experiment 170: Fermi's Golden Rule

Demonstrates time-dependent perturbation theory and Fermi's Golden Rule
for transition rates in quantum systems.

Physics:
    Fermi's Golden Rule gives the transition rate from initial state |i>
    to final states |f> due to a time-dependent perturbation:

    Gamma_{i->f} = (2*pi/hbar) |<f|V|i>|^2 * rho(E_f)

    where rho(E_f) is the density of final states at energy E_f.

    For a periodic perturbation V(t) = V_0 * cos(omega*t):
    - Transitions occur when hbar*omega = E_f - E_i (resonance)
    - Energy is conserved (absorbed or emitted photon)

Examples:
    1. Discrete -> Continuum transitions (photoionization)
    2. Bound state decay
    3. Spectral line shape (Lorentzian)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def transition_probability_first_order(V_fi, omega, omega_fi, t):
    """
    First-order time-dependent perturbation theory.

    P_{i->f}(t) = (|V_fi|^2 / hbar^2) * |sin((omega_fi - omega)*t/2)|^2 / ((omega_fi - omega)/2)^2

    This gives the sinc^2 function centered at resonance.

    Args:
        V_fi: Matrix element <f|V|i>
        omega: Perturbation frequency
        omega_fi: Transition frequency (E_f - E_i)/hbar
        t: Time

    Returns:
        Transition probability
    """
    delta_omega = omega_fi - omega

    if abs(delta_omega) < 1e-10:
        # At resonance: P ~ |V_fi|^2 * t^2
        return abs(V_fi)**2 * t**2
    else:
        # Off resonance: sinc^2 behavior
        return abs(V_fi)**2 * (np.sin(delta_omega * t / 2) / (delta_omega / 2))**2


def fermi_golden_rule_rate(V_fi, rho_E):
    """
    Fermi's Golden Rule transition rate.

    Gamma = (2*pi / hbar) * |<f|V|i>|^2 * rho(E)

    In natural units (hbar = 1):
    Gamma = 2*pi * |V_fi|^2 * rho(E)

    Args:
        V_fi: Matrix element
        rho_E: Density of states at transition energy

    Returns:
        Transition rate
    """
    return 2 * np.pi * abs(V_fi)**2 * rho_E


def lorentzian_lineshape(omega, omega_0, gamma):
    """
    Lorentzian line shape from Fermi's Golden Rule.

    L(omega) = (gamma/2*pi) / ((omega - omega_0)^2 + (gamma/2)^2)

    Args:
        omega: Frequency
        omega_0: Resonance frequency
        gamma: Decay rate (FWHM = gamma)

    Returns:
        Normalized line shape
    """
    return (gamma / (2 * np.pi)) / ((omega - omega_0)**2 + (gamma / 2)**2)


def simulate_decay(V, E_i, E_f_values, rho, t_max, dt):
    """
    Simulate decay using multi-level system with Fermi's Golden Rule.

    Solves Schrodinger equation with perturbation turned on.

    Args:
        V: Coupling matrix element
        E_i: Initial state energy
        E_f_values: Array of final state energies
        rho: Density of states function
        t_max: Maximum time
        dt: Time step

    Returns:
        times, survival_probability
    """
    # Use exponential decay model from FGR
    gamma = fermi_golden_rule_rate(V, rho)

    times = np.arange(0, t_max, dt)
    P_survival = np.exp(-gamma * times)

    return times, P_survival


def simulate_rabi_oscillations(V, omega, omega_0, t_max, dt):
    """
    Simulate Rabi oscillations in a two-level system.

    Without decay (pure Rabi), with decay (damped Rabi).

    H = (hbar * omega_0 / 2) * sigma_z + V * cos(omega*t) * sigma_x

    In the rotating frame with RWA:
    H_eff = (hbar * delta / 2) * sigma_z + (V/2) * sigma_x

    where delta = omega_0 - omega is the detuning.

    Args:
        V: Coupling strength
        omega: Drive frequency
        omega_0: Resonance frequency
        t_max: Maximum time
        dt: Time step

    Returns:
        times, P_excited
    """
    delta = omega_0 - omega
    Omega_R = np.sqrt(V**2 + delta**2)  # Generalized Rabi frequency

    times = np.arange(0, t_max, dt)

    # P_excited = (V^2 / Omega_R^2) * sin^2(Omega_R * t / 2)
    P_excited = (V**2 / Omega_R**2) * np.sin(Omega_R * times / 2)**2

    return times, P_excited


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ===== Plot 1: Transition probability vs time (sinc^2) =====
    ax1 = axes[0, 0]

    V_fi = 0.1  # Matrix element
    omega_fi = 1.0  # Transition frequency

    times = np.linspace(0, 50, 500)

    # Different detunings
    detunings = [0, 0.1, 0.2, 0.5]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(detunings)))

    for delta, color in zip(detunings, colors):
        omega = omega_fi - delta
        P = [transition_probability_first_order(V_fi, omega, omega_fi, t) for t in times]
        ax1.plot(times, P, color=color, lw=2, label=f'delta = {delta}')

    ax1.set_xlabel('Time (hbar/E)')
    ax1.set_ylabel('Transition Probability P(t)')
    ax1.set_title("First-Order Perturbation Theory\n" + r"$P(t) = |V_{fi}|^2 \cdot \mathrm{sinc}^2(\Delta\omega \cdot t/2)$")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 50)

    # ===== Plot 2: Spectral lineshape from FGR =====
    ax2 = axes[0, 1]

    omega_0 = 1.0
    omegas = np.linspace(0.5, 1.5, 500)

    # Different decay rates
    gammas = [0.05, 0.1, 0.2, 0.4]

    for gamma in gammas:
        L = lorentzian_lineshape(omegas, omega_0, gamma)
        ax2.plot(omegas, L, lw=2, label=f'gamma = {gamma}')

    ax2.axvline(omega_0, color='gray', linestyle='--', alpha=0.5, label='Resonance')
    ax2.set_xlabel(r'Frequency $\omega$')
    ax2.set_ylabel('Line Shape L(omega)')
    ax2.set_title('Lorentzian Line Shape\n(Natural broadening from FGR)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ===== Plot 3: Exponential decay from continuum =====
    ax3 = axes[1, 0]

    V = 0.1
    rho_values = [0.5, 1.0, 2.0, 4.0]  # Different density of states

    t_max = 50
    dt = 0.1
    times = np.arange(0, t_max, dt)

    for rho in rho_values:
        gamma = fermi_golden_rule_rate(V, rho)
        P_survival = np.exp(-gamma * times)
        ax3.plot(times, P_survival, lw=2, label=f'rho = {rho}, gamma = {gamma:.3f}')

    ax3.set_xlabel('Time (hbar/E)')
    ax3.set_ylabel('Survival Probability')
    ax3.set_title("Fermi's Golden Rule Decay\n" + r"$P(t) = e^{-\Gamma t}$, $\Gamma = 2\pi|V|^2\rho(E)$")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    ax3.set_ylim(1e-3, 1)

    # ===== Plot 4: Rabi oscillations =====
    ax4 = axes[1, 1]

    V = 0.2
    omega_0 = 1.0
    t_max = 100
    dt = 0.1

    # Different detunings
    detunings = [0, 0.1, 0.2, 0.3]

    for delta in detunings:
        omega = omega_0 - delta
        times, P_excited = simulate_rabi_oscillations(V, omega, omega_0, t_max, dt)

        Omega_R = np.sqrt(V**2 + delta**2)
        label = f'delta={delta}, Omega_R={Omega_R:.2f}'
        ax4.plot(times, P_excited, lw=2, label=label, alpha=0.8)

    ax4.set_xlabel('Time (hbar/E)')
    ax4.set_ylabel('Excited State Population')
    ax4.set_title('Rabi Oscillations\n(Coherent driving, no decay)')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, t_max)

    plt.suptitle("Fermi's Golden Rule and Time-Dependent Perturbation Theory",
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'fermi_golden_rule.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'fermi_golden_rule.png')}")

    # Print numerical results
    print("\n=== Fermi's Golden Rule Results ===")
    print(f"\nMatrix element |V| = 0.1")
    print(f"\nTransition rates for different density of states:")
    for rho in [0.5, 1.0, 2.0, 4.0]:
        gamma = fermi_golden_rule_rate(0.1, rho)
        tau = 1 / gamma if gamma > 0 else np.inf
        print(f"  rho = {rho}: Gamma = {gamma:.4f}, tau = {tau:.2f}")

    print(f"\nLorentzian FWHM for gamma = 0.1: {0.1:.3f}")
    print(f"Peak height at resonance: {lorentzian_lineshape(1.0, 1.0, 0.1):.3f}")


if __name__ == "__main__":
    main()
