"""
Experiment 149: Particle in Box Eigenstates

This experiment demonstrates the complete eigenstate structure of a particle
confined to a 1D infinite square well, including:
- Orthonormality of eigenstates
- Completeness relation
- Expansion coefficients for arbitrary states
- Time evolution of superposition states
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid


def eigenstate(x: np.ndarray, n: int, L: float) -> np.ndarray:
    """
    Calculate nth eigenstate of particle in a box.

    psi_n(x) = sqrt(2/L) * sin(n*pi*x/L) for 0 < x < L

    Args:
        x: Position array
        n: Quantum number (n = 1, 2, 3, ...)
        L: Box length

    Returns:
        Normalized wavefunction values
    """
    psi = np.zeros_like(x)
    mask = (x >= 0) & (x <= L)
    psi[mask] = np.sqrt(2/L) * np.sin(n * np.pi * x[mask] / L)
    return psi


def energy_level(n: int, L: float, m: float = 1.0, hbar: float = 1.0) -> float:
    """
    Calculate energy of nth level.

    E_n = n^2 * pi^2 * hbar^2 / (2 * m * L^2)
    """
    return (n**2 * np.pi**2 * hbar**2) / (2 * m * L**2)


def expansion_coefficients(psi: np.ndarray, x: np.ndarray, L: float, n_max: int = 20) -> np.ndarray:
    """
    Calculate expansion coefficients c_n = <phi_n | psi>.

    Args:
        psi: Wavefunction to expand
        x: Position array
        L: Box length
        n_max: Maximum quantum number to include

    Returns:
        Array of expansion coefficients
    """
    coeffs = np.zeros(n_max)
    for n in range(1, n_max + 1):
        phi_n = eigenstate(x, n, L)
        coeffs[n-1] = trapezoid(phi_n * psi, x)
    return coeffs


def main():
    # Parameters (natural units)
    L = 1.0      # Box length
    m = 1.0      # Particle mass
    hbar = 1.0   # Reduced Planck constant

    # High-resolution position array
    N = 1000
    x = np.linspace(-0.1*L, 1.1*L, N)
    dx = x[1] - x[0]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Plot 1: First 6 eigenstates
    ax1 = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, 6))

    for n in range(1, 7):
        psi_n = eigenstate(x, n, L)
        E_n = energy_level(n, L, m, hbar)
        offset = (n - 1) * 0.4
        ax1.plot(x, psi_n + offset, color=colors[n-1], lw=2, label=f'n={n}')
        ax1.axhline(y=offset, color=colors[n-1], linestyle=':', alpha=0.3)

    # Draw box walls
    ax1.axvline(x=0, color='black', lw=3)
    ax1.axvline(x=L, color='black', lw=3)
    ax1.fill_betweenx([-0.5, 3], -0.1*L, 0, color='gray', alpha=0.3)
    ax1.fill_betweenx([-0.5, 3], L, 1.1*L, color='gray', alpha=0.3)

    ax1.set_xlabel('Position x/L')
    ax1.set_ylabel('Wavefunction (offset)')
    ax1.set_title('Eigenstates psi_n(x)')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_xlim(-0.1*L, 1.1*L)
    ax1.set_ylim(-0.5, 3)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Orthonormality demonstration
    ax2 = axes[0, 1]

    n_max = 10
    overlap_matrix = np.zeros((n_max, n_max))

    for i in range(1, n_max + 1):
        for j in range(1, n_max + 1):
            psi_i = eigenstate(x, i, L)
            psi_j = eigenstate(x, j, L)
            overlap_matrix[i-1, j-1] = trapezoid(psi_i * psi_j, x)

    im = ax2.imshow(overlap_matrix, cmap='RdBu_r', vmin=-0.1, vmax=1.1,
                    extent=[0.5, n_max+0.5, n_max+0.5, 0.5])
    plt.colorbar(im, ax=ax2, label='<psi_m | psi_n>')
    ax2.set_xlabel('Quantum number n')
    ax2.set_ylabel('Quantum number m')
    ax2.set_title('Orthonormality: <psi_m | psi_n> = delta_mn')
    ax2.set_xticks(range(1, n_max + 1))
    ax2.set_yticks(range(1, n_max + 1))

    # Plot 3: Completeness - reconstructing delta function
    ax3 = axes[0, 2]

    x0 = L / 3  # Position of delta function

    n_terms_list = [5, 10, 20, 50]
    colors_comp = plt.cm.plasma(np.linspace(0.2, 0.9, len(n_terms_list)))

    for n_terms, color in zip(n_terms_list, colors_comp):
        # Sum_{n=1}^{N} psi_n(x) * psi_n(x0) -> delta(x - x0)
        delta_approx = np.zeros_like(x)
        for n in range(1, n_terms + 1):
            psi_n = eigenstate(x, n, L)
            psi_n_x0 = eigenstate(np.array([x0]), n, L)[0]
            delta_approx += psi_n * psi_n_x0

        ax3.plot(x, delta_approx, color=color, lw=1.5, label=f'N={n_terms}')

    ax3.axvline(x=x0, color='red', linestyle='--', alpha=0.7, label=f'x0 = L/3')
    ax3.set_xlabel('Position x/L')
    ax3.set_ylabel('Sum_n psi_n(x) psi_n(x0)')
    ax3.set_title('Completeness: Sum -> delta(x - x0)')
    ax3.legend()
    ax3.set_xlim(0, L)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Expansion of arbitrary state
    ax4 = axes[1, 0]

    # Create a Gaussian-like initial state (localized)
    x_center = L / 2
    sigma = L / 10
    psi_gauss = np.exp(-(x - x_center)**2 / (2 * sigma**2))
    # Normalize
    norm = np.sqrt(trapezoid(psi_gauss**2, x))
    psi_gauss /= norm
    # Zero outside box
    psi_gauss[(x < 0) | (x > L)] = 0

    # Calculate expansion coefficients
    n_expansion = 30
    coeffs = expansion_coefficients(psi_gauss, x, L, n_expansion)

    ax4.bar(range(1, n_expansion + 1), coeffs**2, color='steelblue', alpha=0.7)
    ax4.set_xlabel('Quantum number n')
    ax4.set_ylabel('|c_n|^2')
    ax4.set_title('Expansion Coefficients for Gaussian State')
    ax4.grid(True, alpha=0.3, axis='y')

    # Inset: original state vs reconstruction
    ax4_inset = ax4.inset_axes([0.55, 0.55, 0.4, 0.4])

    # Reconstruct from first N terms
    psi_reconstructed = np.zeros_like(x)
    for n in range(1, n_expansion + 1):
        psi_reconstructed += coeffs[n-1] * eigenstate(x, n, L)

    ax4_inset.plot(x, psi_gauss, 'b-', lw=2, label='Original')
    ax4_inset.plot(x, psi_reconstructed, 'r--', lw=1.5, label='Reconstructed')
    ax4_inset.set_xlim(0, L)
    ax4_inset.legend(fontsize=7)
    ax4_inset.set_title('Wavefunction', fontsize=8)

    # Plot 5: Time evolution of superposition
    ax5 = axes[1, 1]

    # Superposition of n=1, 2, 3
    c1, c2, c3 = 1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)
    E1 = energy_level(1, L, m, hbar)
    E2 = energy_level(2, L, m, hbar)
    E3 = energy_level(3, L, m, hbar)

    psi1 = eigenstate(x, 1, L)
    psi2 = eigenstate(x, 2, L)
    psi3 = eigenstate(x, 3, L)

    # Revival time (when all phases realign)
    T_revival = 4 * m * L**2 / (np.pi * hbar)
    times = np.linspace(0, T_revival, 9)
    colors_time = plt.cm.coolwarm(np.linspace(0, 1, len(times)))

    for t, color in zip(times, colors_time):
        # Time-dependent wavefunction
        psi_t = (c1 * psi1 * np.exp(-1j * E1 * t / hbar) +
                 c2 * psi2 * np.exp(-1j * E2 * t / hbar) +
                 c3 * psi3 * np.exp(-1j * E3 * t / hbar))
        prob = np.abs(psi_t)**2
        ax5.plot(x, prob, color=color, lw=1.5, alpha=0.7,
                 label=f't = {t/T_revival:.2f}T')

    ax5.set_xlabel('Position x/L')
    ax5.set_ylabel('|psi(x,t)|^2')
    ax5.set_title('Time Evolution of Superposition (n=1,2,3)')
    ax5.legend(fontsize=7, loc='upper right', ncol=2)
    ax5.set_xlim(0, L)
    ax5.grid(True, alpha=0.3)

    # Plot 6: Expectation values
    ax6 = axes[1, 2]

    # <x> and <x^2> for each eigenstate
    n_states = 15
    x_expect = np.zeros(n_states)
    x2_expect = np.zeros(n_states)
    delta_x = np.zeros(n_states)

    # Position inside box
    x_in = np.linspace(0, L, 1000)

    for n in range(1, n_states + 1):
        psi_n = np.sqrt(2/L) * np.sin(n * np.pi * x_in / L)
        x_expect[n-1] = trapezoid(psi_n**2 * x_in, x_in)
        x2_expect[n-1] = trapezoid(psi_n**2 * x_in**2, x_in)
        delta_x[n-1] = np.sqrt(x2_expect[n-1] - x_expect[n-1]**2)

    ax6.plot(range(1, n_states + 1), x_expect / L, 'bo-', lw=2, label='<x>/L')
    ax6.axhline(y=0.5, color='blue', linestyle='--', alpha=0.5, label='L/2 (classical)')

    ax6_twin = ax6.twinx()
    ax6_twin.plot(range(1, n_states + 1), delta_x / L, 'rs-', lw=2, label='Delta x / L')
    ax6_twin.set_ylabel('Position Uncertainty (Delta x / L)', color='red')
    ax6_twin.tick_params(axis='y', labelcolor='red')

    # Classical limit
    delta_x_classical = L / np.sqrt(12)
    ax6_twin.axhline(y=delta_x_classical / L, color='red', linestyle='--', alpha=0.5)

    ax6.set_xlabel('Quantum number n')
    ax6.set_ylabel('Mean Position (<x>/L)', color='blue')
    ax6.tick_params(axis='y', labelcolor='blue')
    ax6.set_title('Expectation Values vs Quantum Number')
    ax6.legend(loc='upper left')
    ax6_twin.legend(loc='upper right')
    ax6.grid(True, alpha=0.3)

    plt.suptitle('Particle in Box: Eigenstate Structure and Properties\n'
                 'L = 1, m = 1, hbar = 1 (natural units)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'particle_in_box_eigenstates.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'particle_in_box_eigenstates.png')}")


if __name__ == "__main__":
    main()
