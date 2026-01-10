"""
Example demonstrating the quantum harmonic oscillator.

This example shows the wavefunctions and energy levels of the
quantum harmonic oscillator, comparing to classical behavior.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite
from scipy.special import factorial
from src.sciforge.physics.quantum import Wavefunction


def hermite_function(n, x):
    """
    Calculate the nth Hermite polynomial H_n(x).
    """
    coeffs = np.zeros(n + 1)
    coeffs[n] = 1
    return np.polynomial.hermite.hermval(x, coeffs)


def qho_wavefunction(n, x, m=1.0, omega=1.0, hbar=1.0):
    """
    Calculate the nth eigenstate wavefunction of quantum harmonic oscillator.

    ψ_n(x) = (mω/πℏ)^(1/4) * (1/√(2^n n!)) * H_n(ξ) * exp(-ξ²/2)
    where ξ = √(mω/ℏ) * x

    Args:
        n: Quantum number (n = 0, 1, 2, ...)
        x: Position array
        m: Mass
        omega: Angular frequency
        hbar: Reduced Planck constant

    Returns:
        Wavefunction values
    """
    xi = np.sqrt(m * omega / hbar) * x
    normalization = (m * omega / (np.pi * hbar))**(0.25) / np.sqrt(2**n * factorial(n))
    return normalization * hermite_function(n, xi) * np.exp(-xi**2 / 2)


def qho_energy(n, omega=1.0, hbar=1.0):
    """Calculate energy of nth level: E_n = ℏω(n + 1/2)."""
    return hbar * omega * (n + 0.5)


def classical_amplitude(E, m=1.0, omega=1.0):
    """Calculate classical turning point: x_max = √(2E/mω²)."""
    return np.sqrt(2 * E / (m * omega**2))


def classical_probability(x, E, m=1.0, omega=1.0):
    """
    Classical probability distribution for harmonic oscillator.
    P(x) = 1/(π√(A² - x²)) where A is amplitude
    """
    A = classical_amplitude(E, m, omega)
    prob = np.zeros_like(x)
    valid = np.abs(x) < A
    prob[valid] = 1 / (np.pi * np.sqrt(A**2 - x[valid]**2))
    return prob


def main():
    # Parameters (natural units)
    m = 1.0
    omega = 1.0
    hbar = 1.0

    # Position array
    x = np.linspace(-5, 5, 500)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: First several wavefunctions
    ax1 = axes[0, 0]

    n_states = 6
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_states))

    for n in range(n_states):
        psi = qho_wavefunction(n, x, m, omega, hbar)
        E_n = qho_energy(n, omega, hbar)
        # Offset by energy for visualization
        ax1.plot(x, psi + E_n * 0.3, color=colors[n], lw=2, label=f'n={n}')
        ax1.axhline(y=E_n * 0.3, color=colors[n], linestyle=':', alpha=0.3)

    # Draw potential well
    V = 0.5 * m * omega**2 * x**2
    ax1.plot(x, V * 0.3, 'k--', lw=2, alpha=0.5, label='V(x)')

    ax1.set_xlabel('Position x')
    ax1.set_ylabel('ψ_n(x) + offset')
    ax1.set_title('Wavefunctions')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-5, 5)

    # Plot 2: Probability densities
    ax2 = axes[0, 1]

    for n in range(n_states):
        psi = qho_wavefunction(n, x, m, omega, hbar)
        prob = psi**2
        E_n = qho_energy(n, omega, hbar)
        ax2.plot(x, prob + E_n * 0.2, color=colors[n], lw=2, label=f'n={n}')
        ax2.fill_between(x, E_n * 0.2, prob + E_n * 0.2, color=colors[n], alpha=0.2)

    ax2.set_xlabel('Position x')
    ax2.set_ylabel('|ψ_n(x)|² + offset')
    ax2.set_title('Probability Densities')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-5, 5)

    # Plot 3: Quantum vs Classical probability (high n)
    ax3 = axes[1, 0]

    n_high = 20
    psi_high = qho_wavefunction(n_high, x, m, omega, hbar)
    prob_quantum = psi_high**2

    E_n = qho_energy(n_high, omega, hbar)
    prob_classical = classical_probability(x, E_n, m, omega)

    ax3.plot(x, prob_quantum, 'b-', lw=2, label=f'Quantum (n={n_high})')
    ax3.plot(x, prob_classical, 'r--', lw=2, label='Classical')

    # Mark classical turning points
    A = classical_amplitude(E_n, m, omega)
    ax3.axvline(x=A, color='gray', linestyle=':', alpha=0.5)
    ax3.axvline(x=-A, color='gray', linestyle=':', alpha=0.5)
    ax3.annotate('Classical\nturning points', xy=(A, 0.3), fontsize=9)

    ax3.set_xlabel('Position x')
    ax3.set_ylabel('Probability Density')
    ax3.set_title(f'Correspondence Principle (n = {n_high})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-6, 6)
    ax3.set_ylim(0, 0.6)

    # Plot 4: Energy level diagram
    ax4 = axes[1, 1]

    n_levels = 10
    for n in range(n_levels):
        E_n = qho_energy(n, omega, hbar)
        ax4.hlines(E_n, 0.3, 0.7, colors='blue', lw=2)
        ax4.text(0.72, E_n, f'n={n}, E={(n+0.5):.1f}ℏω', va='center', fontsize=9)

    # Show equal spacing
    for n in range(1, n_levels):
        ax4.annotate('', xy=(0.25, qho_energy(n, omega, hbar)),
                    xytext=(0.25, qho_energy(n-1, omega, hbar)),
                    arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))

    ax4.text(0.15, qho_energy(0.5, omega, hbar), 'ℏω', fontsize=10, color='red', va='center')

    ax4.set_xlim(0, 1)
    ax4.set_ylim(-0.5, qho_energy(n_levels, omega, hbar))
    ax4.set_ylabel('Energy (ℏω)')
    ax4.set_title('Energy Levels: E_n = ℏω(n + ½)')
    ax4.set_xticks([])

    # Draw potential well
    ax4_twin = ax4.twinx()
    x_pot = np.linspace(-3, 3, 100)
    V_pot = 0.5 * m * omega**2 * x_pot**2
    scale = qho_energy(n_levels - 1) / max(V_pot)
    ax4_twin.plot(0.5 + x_pot * 0.1, V_pot * scale, 'k--', lw=1, alpha=0.3)
    ax4_twin.set_ylim(ax4.get_ylim())
    ax4_twin.set_yticks([])

    plt.suptitle('Quantum Harmonic Oscillator\n'
                 'V(x) = ½mω²x², E_n = ℏω(n + ½)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'harmonic_oscillator_quantum.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'harmonic_oscillator_quantum.png')}")


if __name__ == "__main__":
    main()
