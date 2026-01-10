"""
Experiment 151: Step Potential Reflection and Transmission

This experiment demonstrates quantum mechanical scattering from a potential step,
including:
- Reflection and transmission coefficients vs energy
- Wavefunction behavior above and below step height
- Probability current conservation
- Classical vs quantum behavior comparison
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def step_coefficients(E: float, V0: float, m: float = 1.0, hbar: float = 1.0) -> tuple:
    """
    Calculate reflection and transmission coefficients for step potential.

    Step potential: V(x) = 0 for x < 0, V(x) = V0 for x >= 0

    Args:
        E: Particle energy
        V0: Step height
        m: Particle mass
        hbar: Reduced Planck constant

    Returns:
        Tuple of (R, T) reflection and transmission coefficients
    """
    if E <= 0:
        return 1.0, 0.0

    k1 = np.sqrt(2 * m * E) / hbar  # Wave number for x < 0

    if E <= V0:
        # Below step: total reflection with evanescent wave
        # k2 is imaginary, so T = 0
        R = 1.0
        T = 0.0
    else:
        # Above step: partial transmission
        k2 = np.sqrt(2 * m * (E - V0)) / hbar

        # Reflection coefficient
        R = ((k1 - k2) / (k1 + k2))**2

        # Transmission coefficient (probability current ratio)
        T = (4 * k1 * k2) / (k1 + k2)**2

    return R, T


def step_wavefunction(x: np.ndarray, E: float, V0: float, m: float = 1.0,
                       hbar: float = 1.0) -> tuple:
    """
    Calculate wavefunction for step potential scattering.

    Returns incident + reflected for x < 0, transmitted for x >= 0.

    Args:
        x: Position array
        E: Particle energy
        V0: Step height
        m: Particle mass
        hbar: Reduced Planck constant

    Returns:
        Tuple of (psi, Re(psi), Im(psi))
    """
    k1 = np.sqrt(2 * m * E) / hbar

    psi = np.zeros_like(x, dtype=complex)

    # Reflection and transmission amplitudes
    if E <= V0:
        # Total reflection with evanescent wave
        kappa = np.sqrt(2 * m * (V0 - E)) / hbar

        # Reflection amplitude (magnitude 1)
        r = (k1 - 1j * kappa) / (k1 + 1j * kappa)
        t = 2 * k1 / (k1 + 1j * kappa)

        # Region I: x < 0
        mask1 = x < 0
        psi[mask1] = np.exp(1j * k1 * x[mask1]) + r * np.exp(-1j * k1 * x[mask1])

        # Region II: x >= 0 (evanescent)
        mask2 = x >= 0
        psi[mask2] = t * np.exp(-kappa * x[mask2])
    else:
        # Partial transmission
        k2 = np.sqrt(2 * m * (E - V0)) / hbar

        # Amplitudes
        r = (k1 - k2) / (k1 + k2)
        t = 2 * k1 / (k1 + k2)

        # Region I: x < 0
        mask1 = x < 0
        psi[mask1] = np.exp(1j * k1 * x[mask1]) + r * np.exp(-1j * k1 * x[mask1])

        # Region II: x >= 0
        mask2 = x >= 0
        psi[mask2] = t * np.exp(1j * k2 * x[mask2])

    return psi, np.real(psi), np.imag(psi)


def probability_current(x: np.ndarray, psi: np.ndarray, m: float = 1.0,
                        hbar: float = 1.0) -> np.ndarray:
    """
    Calculate probability current j = (hbar/2mi)(psi* dpsi/dx - psi dpsi*/dx).
    """
    dx = x[1] - x[0]
    dpsi_dx = np.gradient(psi, dx)
    j = (hbar / (2 * m * 1j)) * (np.conj(psi) * dpsi_dx - psi * np.conj(dpsi_dx))
    return np.real(j)


def main():
    # Parameters (natural units)
    m = 1.0
    hbar = 1.0
    V0 = 1.0  # Step height

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Plot 1: R and T vs Energy
    ax1 = axes[0, 0]

    energies = np.linspace(0.01, 3.0, 500)
    R_values = []
    T_values = []

    for E in energies:
        R, T = step_coefficients(E, V0, m, hbar)
        R_values.append(R)
        T_values.append(T)

    ax1.plot(energies / V0, R_values, 'b-', lw=2, label='Reflection R')
    ax1.plot(energies / V0, T_values, 'r-', lw=2, label='Transmission T')
    ax1.plot(energies / V0, np.array(R_values) + np.array(T_values), 'k--', lw=1,
             label='R + T = 1', alpha=0.7)

    ax1.axvline(x=1.0, color='gray', linestyle=':', alpha=0.7, label='E = V0')
    ax1.fill_betweenx([0, 1], 0, 1, alpha=0.1, color='blue', label='E < V0 (total reflection)')

    ax1.set_xlabel('Energy E / V0')
    ax1.set_ylabel('Probability')
    ax1.set_title('Reflection and Transmission Coefficients')
    ax1.legend(loc='center right')
    ax1.set_xlim(0, 3)
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Wavefunction for E < V0 (evanescent)
    ax2 = axes[0, 1]

    x = np.linspace(-10, 10, 1000)
    E_below = 0.5 * V0

    psi, psi_re, psi_im = step_wavefunction(x, E_below, V0, m, hbar)
    prob = np.abs(psi)**2

    # Draw step potential
    V = np.zeros_like(x)
    V[x >= 0] = V0
    ax2_pot = ax2.twinx()
    ax2_pot.fill_between(x, 0, V, alpha=0.2, color='gray')
    ax2_pot.set_ylabel('Potential V(x)', color='gray')
    ax2_pot.set_ylim(0, 2*V0)

    ax2.plot(x, psi_re, 'b-', lw=1.5, label='Re(psi)', alpha=0.7)
    ax2.plot(x, psi_im, 'g--', lw=1.5, label='Im(psi)', alpha=0.7)
    ax2.plot(x, prob, 'r-', lw=2, label='|psi|^2')

    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Position x')
    ax2.set_ylabel('Wavefunction')
    ax2.set_title(f'E = 0.5 V0 (Total Reflection)')
    ax2.legend(loc='upper right')
    ax2.set_xlim(-10, 10)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Wavefunction for E > V0 (transmission)
    ax3 = axes[0, 2]

    E_above = 1.5 * V0

    psi, psi_re, psi_im = step_wavefunction(x, E_above, V0, m, hbar)
    prob = np.abs(psi)**2

    # Draw step potential
    ax3_pot = ax3.twinx()
    ax3_pot.fill_between(x, 0, V, alpha=0.2, color='gray')
    ax3_pot.set_ylabel('Potential V(x)', color='gray')
    ax3_pot.set_ylim(0, 2*V0)

    ax3.plot(x, psi_re, 'b-', lw=1.5, label='Re(psi)', alpha=0.7)
    ax3.plot(x, psi_im, 'g--', lw=1.5, label='Im(psi)', alpha=0.7)
    ax3.plot(x, prob, 'r-', lw=2, label='|psi|^2')

    ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Position x')
    ax3.set_ylabel('Wavefunction')
    ax3.set_title(f'E = 1.5 V0 (Partial Transmission)')
    ax3.legend(loc='upper right')
    ax3.set_xlim(-10, 10)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Standing wave pattern (E < V0)
    ax4 = axes[1, 0]

    # Different energies below step
    E_cases = [0.2, 0.4, 0.6, 0.8, 0.95]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(E_cases)))

    for E_case, color in zip(E_cases, colors):
        E = E_case * V0
        psi, _, _ = step_wavefunction(x, E, V0, m, hbar)
        prob = np.abs(psi)**2
        ax4.plot(x, prob, color=color, lw=1.5, label=f'E = {E_case:.1f} V0')

    ax4.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Position x')
    ax4.set_ylabel('|psi|^2')
    ax4.set_title('Standing Wave Pattern (E < V0)')
    ax4.legend()
    ax4.set_xlim(-10, 5)
    ax4.grid(True, alpha=0.3)

    # Plot 5: Probability current conservation
    ax5 = axes[1, 1]

    E_test = 1.8 * V0
    psi, _, _ = step_wavefunction(x, E_test, V0, m, hbar)
    j = probability_current(x, psi, m, hbar)

    # Theoretical values
    k1 = np.sqrt(2 * m * E_test) / hbar
    k2 = np.sqrt(2 * m * (E_test - V0)) / hbar
    R, T = step_coefficients(E_test, V0, m, hbar)

    j_incident = hbar * k1 / m
    j_reflected = -R * j_incident
    j_transmitted = T * j_incident * (k2 / k1)  # Adjusted for different k

    ax5.plot(x, j, 'b-', lw=2, label='j(x)')
    ax5.axhline(y=j_incident, color='green', linestyle='--', alpha=0.7,
                label=f'j_incident = {j_incident:.3f}')
    ax5.axhline(y=j_incident * (1 - R), color='red', linestyle='--', alpha=0.7,
                label=f'j_transmitted = {j_incident*(1-R):.3f}')

    ax5.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Position x')
    ax5.set_ylabel('Probability Current j(x)')
    ax5.set_title(f'Probability Current (E = {E_test/V0:.1f} V0)')
    ax5.legend()
    ax5.set_xlim(-10, 10)
    ax5.grid(True, alpha=0.3)

    # Plot 6: Quantum vs Classical comparison
    ax6 = axes[1, 2]

    energies_comp = np.linspace(0.01, 3.0, 500)

    # Quantum
    R_quantum = [step_coefficients(E, V0, m, hbar)[0] for E in energies_comp]

    # Classical: R = 1 for E < V0, R = 0 for E > V0
    R_classical = np.where(energies_comp <= V0, 1.0, 0.0)

    ax6.plot(energies_comp / V0, R_quantum, 'b-', lw=2, label='Quantum R')
    ax6.plot(energies_comp / V0, R_classical, 'r--', lw=2, label='Classical R')

    # Highlight quantum reflection above barrier
    mask = energies_comp > V0
    ax6.fill_between(energies_comp[mask] / V0, 0, np.array(R_quantum)[mask],
                     alpha=0.3, color='blue', label='Quantum reflection (E > V0)')

    ax6.axvline(x=1.0, color='gray', linestyle=':', alpha=0.7)
    ax6.set_xlabel('Energy E / V0')
    ax6.set_ylabel('Reflection Probability R')
    ax6.set_title('Quantum vs Classical Reflection')
    ax6.legend()
    ax6.set_xlim(0, 3)
    ax6.set_ylim(0, 1.1)
    ax6.grid(True, alpha=0.3)

    # Add annotation about quantum reflection
    ax6.annotate('Quantum:\npartial reflection\neven when E > V0',
                 xy=(1.5, 0.15), fontsize=9, ha='center',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('Step Potential Scattering\n'
                 'V(x) = 0 for x < 0, V(x) = V0 for x >= 0',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'step_potential.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'step_potential.png')}")


if __name__ == "__main__":
    main()
