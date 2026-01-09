"""
Example demonstrating particle in a box (infinite square well).

This example shows quantum mechanical wavefunctions and energy levels
for a particle confined to a 1D box with infinite potential walls.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.quantum import Wavefunction


def particle_in_box_wavefunction(x, n, L):
    """
    Calculate normalized wavefunction for particle in a box.

    Args:
        x: Position array
        n: Quantum number (n = 1, 2, 3, ...)
        L: Box length

    Returns:
        Wavefunction values (normalized)
    """
    # ψ_n(x) = sqrt(2/L) * sin(nπx/L)
    return np.sqrt(2/L) * np.sin(n * np.pi * x / L)


def energy_level(n, L, m=1.0, hbar=1.0):
    """
    Calculate energy of nth level.

    Args:
        n: Quantum number
        L: Box length
        m: Particle mass
        hbar: Reduced Planck constant

    Returns:
        Energy (in natural units)
    """
    return (n**2 * np.pi**2 * hbar**2) / (2 * m * L**2)


def main():
    # Parameters (natural units: ħ = m = 1)
    L = 1.0      # Box length
    m = 1.0      # Particle mass
    hbar = 1.0   # Reduced Planck constant

    # Position array
    x = np.linspace(0, L, 500)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: First few wavefunctions
    ax1 = axes[0, 0]
    colors = plt.cm.tab10(np.linspace(0, 1, 6))

    for n in range(1, 7):
        psi = particle_in_box_wavefunction(x, n, L)
        E_n = energy_level(n, L, m, hbar)
        # Offset by energy for visualization
        ax1.plot(x, psi + E_n * 0.02, color=colors[n-1], lw=2, label=f'n={n}')

    ax1.set_xlabel('Position x')
    ax1.set_ylabel('ψ(x) + offset')
    ax1.set_title('Wavefunctions ψₙ(x)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    # Plot 2: Probability densities
    ax2 = axes[0, 1]

    for n in range(1, 7):
        psi = particle_in_box_wavefunction(x, n, L)
        prob = psi**2
        E_n = energy_level(n, L, m, hbar)
        ax2.plot(x, prob + E_n * 0.02, color=colors[n-1], lw=2, label=f'n={n}')
        ax2.fill_between(x, E_n * 0.02, prob + E_n * 0.02, color=colors[n-1], alpha=0.2)

    ax2.set_xlabel('Position x')
    ax2.set_ylabel('|ψ(x)|² + offset')
    ax2.set_title('Probability Densities |ψₙ(x)|²')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Energy level diagram
    ax3 = axes[1, 0]

    n_levels = 10
    for n in range(1, n_levels + 1):
        E_n = energy_level(n, L, m, hbar)
        ax3.hlines(E_n, 0.3, 0.7, colors='blue', lw=2)
        ax3.text(0.72, E_n, f'n={n}, E={E_n:.1f}', va='center', fontsize=9)

    # Show E ∝ n² relationship
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, energy_level(n_levels + 1, L, m, hbar))
    ax3.set_ylabel('Energy (ℏ²/2mL²)')
    ax3.set_title('Energy Levels Eₙ = n²π²ℏ²/2mL²')
    ax3.set_xticks([])
    ax3.grid(True, alpha=0.3, axis='y')

    # Draw "walls" of the box
    ax3.axvline(x=0.3, color='black', lw=3)
    ax3.axvline(x=0.7, color='black', lw=3)
    ax3.fill_between([0, 0.3], 0, energy_level(n_levels + 1, L), color='gray', alpha=0.3)
    ax3.fill_between([0.7, 1], 0, energy_level(n_levels + 1, L), color='gray', alpha=0.3)

    # Plot 4: Superposition state
    ax4 = axes[1, 1]

    # Create superposition of n=1 and n=2
    psi_1 = particle_in_box_wavefunction(x, 1, L)
    psi_2 = particle_in_box_wavefunction(x, 2, L)

    # Superposition coefficients
    c1 = 1/np.sqrt(2)
    c2 = 1/np.sqrt(2)

    # Time evolution
    E1 = energy_level(1, L, m, hbar)
    E2 = energy_level(2, L, m, hbar)

    omega_1 = E1 / hbar
    omega_2 = E2 / hbar

    times = np.linspace(0, 2*np.pi/abs(omega_2 - omega_1), 8)
    colors_time = plt.cm.plasma(np.linspace(0, 0.9, len(times)))

    for t, color in zip(times, colors_time):
        # Time-dependent wavefunction
        psi_t = c1 * psi_1 * np.cos(omega_1 * t) + c2 * psi_2 * np.cos(omega_2 * t)
        prob_t = psi_t**2

        ax4.plot(x, prob_t, color=color, lw=1.5, alpha=0.7,
                label=f't = {t/(2*np.pi/abs(omega_2-omega_1)):.2f}T')

    ax4.set_xlabel('Position x')
    ax4.set_ylabel('|ψ(x,t)|²')
    ax4.set_title('Superposition State (n=1 + n=2) Time Evolution')
    ax4.legend(fontsize=8, loc='upper right')
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Quantum Particle in a Box (Infinite Square Well)\n'
                 'L = 1, m = 1, ℏ = 1 (natural units)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'particle_in_box.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'particle_in_box.png')}")


if __name__ == "__main__":
    main()