"""
Example demonstrating quantum tunneling through a potential barrier.

This example shows the transmission and reflection of a quantum
wavepacket encountering a potential barrier.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.quantum import Wavefunction


def transmission_coefficient(E, V0, a, m=1.0, hbar=1.0):
    """
    Calculate transmission coefficient for rectangular barrier.

    Args:
        E: Particle energy
        V0: Barrier height
        a: Barrier width
        m: Particle mass
        hbar: Reduced Planck constant

    Returns:
        Transmission coefficient T
    """
    if E >= V0:
        # Above barrier - oscillatory solution
        k = np.sqrt(2 * m * E) / hbar
        k_prime = np.sqrt(2 * m * (E - V0)) / hbar
        T = 1 / (1 + (k**2 + k_prime**2)**2 * np.sin(k_prime * a)**2 / (4 * k**2 * k_prime**2))
    else:
        # Below barrier - tunneling
        k = np.sqrt(2 * m * E) / hbar
        kappa = np.sqrt(2 * m * (V0 - E)) / hbar

        if kappa * a > 50:  # Very thick barrier
            T = 0
        else:
            sinh_term = np.sinh(kappa * a)
            T = 1 / (1 + (k**2 + kappa**2)**2 * sinh_term**2 / (4 * k**2 * kappa**2))

    return T


def gaussian_wavepacket(x, x0, sigma, k0):
    """Create a Gaussian wavepacket."""
    return np.exp(-(x - x0)**2 / (4 * sigma**2)) * np.exp(1j * k0 * x) / (2 * np.pi * sigma**2)**0.25


def main():
    # Parameters (natural units)
    V0 = 1.0      # Barrier height
    a = 1.0       # Barrier width
    m = 1.0       # Particle mass
    hbar = 1.0    # Reduced Planck constant

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Transmission vs Energy
    ax1 = axes[0, 0]

    energies = np.linspace(0.01, 3.0, 500)
    T_values = [transmission_coefficient(E, V0, a, m, hbar) for E in energies]
    R_values = [1 - T for T in T_values]

    ax1.plot(energies / V0, T_values, 'b-', lw=2, label='Transmission T')
    ax1.plot(energies / V0, R_values, 'r--', lw=2, label='Reflection R')
    ax1.axvline(x=1.0, color='gray', linestyle=':', alpha=0.7, label='E = V₀')
    ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)

    ax1.set_xlabel('Energy / Barrier Height (E/V₀)')
    ax1.set_ylabel('Probability')
    ax1.set_title('Transmission and Reflection Coefficients')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 3)
    ax1.set_ylim(0, 1.1)

    # Highlight tunneling region
    ax1.axvspan(0, 1, alpha=0.1, color='red', label='Tunneling region')

    # Plot 2: Transmission vs Barrier Width
    ax2 = axes[0, 1]

    widths = np.linspace(0.1, 5.0, 100)
    energies_cases = [0.3 * V0, 0.5 * V0, 0.7 * V0, 0.9 * V0]
    colors = ['blue', 'green', 'orange', 'red']

    for E, color in zip(energies_cases, colors):
        T_width = [transmission_coefficient(E, V0, w, m, hbar) for w in widths]
        ax2.semilogy(widths, T_width, color=color, lw=2, label=f'E = {E/V0:.1f}V₀')

    ax2.set_xlabel('Barrier Width (a)')
    ax2.set_ylabel('Transmission Coefficient (log scale)')
    ax2.set_title('Tunneling Probability vs Barrier Width')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Wavefunction illustration
    ax3 = axes[1, 0]

    x = np.linspace(-10, 10, 1000)

    # Barrier region
    barrier_left = -a/2
    barrier_right = a/2

    # Draw potential
    V = np.zeros_like(x)
    V[(x >= barrier_left) & (x <= barrier_right)] = V0
    ax3.fill_between(x, V * 3, alpha=0.3, color='gray', label='Barrier')

    # For E < V0 (tunneling case)
    E = 0.5 * V0
    k = np.sqrt(2 * m * E) / hbar
    kappa = np.sqrt(2 * m * (V0 - E)) / hbar

    # Incident + reflected wave (x < barrier)
    # Transmitted wave (x > barrier)
    # Decaying wave (inside barrier)

    T = transmission_coefficient(E, V0, a, m, hbar)
    R = 1 - T

    # Schematic wavefunction (not exact)
    psi = np.zeros_like(x, dtype=complex)

    # Region I (x < barrier): incident + reflected
    mask1 = x < barrier_left
    psi[mask1] = np.exp(1j * k * x[mask1]) + np.sqrt(R) * np.exp(-1j * k * x[mask1])

    # Region II (inside barrier): exponentially decaying
    mask2 = (x >= barrier_left) & (x <= barrier_right)
    A = psi[mask1][-1]  # Match at boundary
    psi[mask2] = A * np.exp(-kappa * (x[mask2] - barrier_left))

    # Region III (x > barrier): transmitted
    mask3 = x > barrier_right
    B = psi[mask2][-1]  # Match at boundary
    psi[mask3] = B * np.exp(1j * k * (x[mask3] - barrier_right))

    ax3.plot(x, np.abs(psi)**2, 'b-', lw=2, label='|ψ|² (probability density)')
    ax3.plot(x, np.real(psi), 'g--', lw=1, alpha=0.7, label='Re(ψ)')

    ax3.set_xlabel('Position x')
    ax3.set_ylabel('Wavefunction')
    ax3.set_title(f'Tunneling Through Barrier (E = 0.5V₀, T = {T:.3f})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-10, 10)

    # Plot 4: Energy dependence of tunneling
    ax4 = axes[1, 1]

    # Calculate T for different barrier heights
    V0_values = [0.5, 1.0, 2.0, 4.0]
    for V0_case in V0_values:
        T_vs_E = [transmission_coefficient(E, V0_case, a, m, hbar)
                  for E in np.linspace(0.01, 5.0, 200)]
        ax4.plot(np.linspace(0.01, 5.0, 200), T_vs_E, lw=2, label=f'V₀ = {V0_case}')

    ax4.set_xlabel('Energy E')
    ax4.set_ylabel('Transmission Coefficient T')
    ax4.set_title('Transmission for Different Barrier Heights')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1.1)

    plt.suptitle('Quantum Tunneling Through a Rectangular Potential Barrier\n'
                 '(m = ℏ = 1, natural units)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'quantum_tunneling.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'quantum_tunneling.png')}")


if __name__ == "__main__":
    main()