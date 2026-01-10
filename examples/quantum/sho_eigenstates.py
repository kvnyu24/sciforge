"""
Experiment 156: Simple Harmonic Oscillator Eigenstates

This experiment provides a comprehensive study of quantum harmonic oscillator
eigenstates, including:
- Wavefunction structure and nodes
- Ladder operator relationships
- Matrix elements and selection rules
- Position and momentum representations
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
from scipy.integrate import trapezoid


def hermite_polynomial(n: int, x: np.ndarray) -> np.ndarray:
    """
    Calculate physicist's Hermite polynomial H_n(x).

    Uses recurrence relation: H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x)
    """
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x

    H_prev2 = np.ones_like(x)
    H_prev1 = 2 * x

    for k in range(2, n + 1):
        H_curr = 2 * x * H_prev1 - 2 * (k - 1) * H_prev2
        H_prev2 = H_prev1
        H_prev1 = H_curr

    return H_prev1


def sho_eigenstate(n: int, x: np.ndarray, m: float = 1.0, omega: float = 1.0,
                   hbar: float = 1.0) -> np.ndarray:
    """
    Calculate nth eigenstate of quantum harmonic oscillator.

    psi_n(x) = (m*omega/(pi*hbar))^(1/4) * 1/sqrt(2^n * n!) * H_n(xi) * exp(-xi^2/2)
    where xi = sqrt(m*omega/hbar) * x

    Args:
        n: Quantum number (n = 0, 1, 2, ...)
        x: Position array
        m: Particle mass
        omega: Angular frequency
        hbar: Reduced Planck constant

    Returns:
        Normalized wavefunction
    """
    alpha = np.sqrt(m * omega / hbar)
    xi = alpha * x

    # Normalization factor
    norm = (m * omega / (np.pi * hbar))**0.25 / np.sqrt(2**n * factorial(n))

    # Wavefunction
    psi = norm * hermite_polynomial(n, xi) * np.exp(-xi**2 / 2)

    return psi


def sho_energy(n: int, omega: float = 1.0, hbar: float = 1.0) -> float:
    """Energy of nth level: E_n = hbar*omega*(n + 1/2)."""
    return hbar * omega * (n + 0.5)


def ladder_operator_action(psi: np.ndarray, x: np.ndarray, direction: str,
                           m: float = 1.0, omega: float = 1.0,
                           hbar: float = 1.0) -> np.ndarray:
    """
    Apply raising or lowering operator.

    a = sqrt(m*omega/(2*hbar)) * (x + i*p/(m*omega))
    a^dagger = sqrt(m*omega/(2*hbar)) * (x - i*p/(m*omega))

    where p = -i*hbar * d/dx

    Args:
        psi: Wavefunction
        x: Position array
        direction: 'up' (raising) or 'down' (lowering)
        m: Particle mass
        omega: Angular frequency
        hbar: Reduced Planck constant

    Returns:
        Operated wavefunction
    """
    dx = x[1] - x[0]
    alpha = np.sqrt(m * omega / (2 * hbar))

    # Derivative of psi (momentum operator action)
    dpsi_dx = np.gradient(psi, dx)

    if direction == 'down':
        # a = alpha * x + (hbar/m/omega) * alpha * d/dx
        result = alpha * x * psi + (hbar / (m * omega)) * alpha * dpsi_dx
    else:  # up
        # a^dagger = alpha * x - (hbar/m/omega) * alpha * d/dx
        result = alpha * x * psi - (hbar / (m * omega)) * alpha * dpsi_dx

    return result


def main():
    # Parameters (natural units)
    m = 1.0
    omega = 1.0
    hbar = 1.0

    # Spatial grid
    x_max = 8.0
    N = 1000
    x = np.linspace(-x_max, x_max, N)
    dx = x[1] - x[0]

    # Characteristic length scale
    x0 = np.sqrt(hbar / (m * omega))

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Plot 1: First 6 eigenstates
    ax1 = axes[0, 0]

    n_states = 6
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_states))

    for n in range(n_states):
        psi = sho_eigenstate(n, x, m, omega, hbar)
        E_n = sho_energy(n, omega, hbar)
        offset = E_n * 0.5
        ax1.plot(x / x0, psi * x0**0.5 + offset, color=colors[n], lw=2,
                label=f'n={n}')
        ax1.axhline(y=offset, color=colors[n], linestyle=':', alpha=0.3)

    # Draw potential
    V = 0.5 * m * omega**2 * x**2
    ax1.plot(x / x0, V / (hbar * omega) * 0.5, 'k--', lw=2, alpha=0.5,
             label='V(x)')

    ax1.set_xlabel('Position x / x0')
    ax1.set_ylabel('psi_n (offset by E_n)')
    ax1.set_title('Harmonic Oscillator Eigenstates')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-5, 5)

    # Plot 2: Node counting
    ax2 = axes[0, 1]

    for n in range(n_states):
        psi = sho_eigenstate(n, x, m, omega, hbar)
        prob = psi**2

        # Find nodes (zero crossings)
        nodes_x = []
        for i in range(len(psi) - 1):
            if psi[i] * psi[i+1] < 0:
                # Linear interpolation for crossing
                x_node = x[i] - psi[i] * (x[i+1] - x[i]) / (psi[i+1] - psi[i])
                nodes_x.append(x_node)

        ax2.plot(x / x0, prob * x0 + n, color=colors[n], lw=2)
        ax2.fill_between(x / x0, n, prob * x0 + n, color=colors[n], alpha=0.2)

        # Mark nodes
        for node in nodes_x:
            ax2.plot(node / x0, n, 'ko', markersize=5)

        ax2.text(4.5, n + 0.5, f'n={n}: {len(nodes_x)} nodes', fontsize=9)

    ax2.set_xlabel('Position x / x0')
    ax2.set_ylabel('|psi_n|^2 + offset')
    ax2.set_title('Probability Densities and Node Counting')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-5, 6)

    # Plot 3: Ladder operator demonstration
    ax3 = axes[0, 2]

    n_test = 2
    psi_n = sho_eigenstate(n_test, x, m, omega, hbar)
    psi_n_plus_1 = sho_eigenstate(n_test + 1, x, m, omega, hbar)
    psi_n_minus_1 = sho_eigenstate(n_test - 1, x, m, omega, hbar)

    # Apply ladder operators
    a_dagger_psi = ladder_operator_action(psi_n, x, 'up', m, omega, hbar)
    a_psi = ladder_operator_action(psi_n, x, 'down', m, omega, hbar)

    # Normalize for comparison
    norm_up = np.sqrt(n_test + 1)
    norm_down = np.sqrt(n_test)

    ax3.plot(x / x0, psi_n_plus_1, 'b-', lw=2, label=f'psi_{n_test+1} (exact)')
    ax3.plot(x / x0, a_dagger_psi / norm_up, 'b--', lw=2, alpha=0.7,
             label=f'a^dag psi_{n_test} / sqrt({n_test+1})')

    ax3.plot(x / x0, psi_n_minus_1, 'r-', lw=2, label=f'psi_{n_test-1} (exact)')
    ax3.plot(x / x0, a_psi / norm_down, 'r--', lw=2, alpha=0.7,
             label=f'a psi_{n_test} / sqrt({n_test})')

    ax3.set_xlabel('Position x / x0')
    ax3.set_ylabel('Wavefunction')
    ax3.set_title(f'Ladder Operators (starting from n={n_test})')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-5, 5)

    # Plot 4: Matrix elements <m|x|n>
    ax4 = axes[1, 0]

    n_max = 10
    x_matrix = np.zeros((n_max, n_max))

    for m_idx in range(n_max):
        for n_idx in range(n_max):
            psi_m = sho_eigenstate(m_idx, x, m, omega, hbar)
            psi_n = sho_eigenstate(n_idx, x, m, omega, hbar)
            x_matrix[m_idx, n_idx] = trapezoid(psi_m * x * psi_n, x)

    im = ax4.imshow(np.abs(x_matrix), cmap='Blues', extent=[-0.5, n_max-0.5, n_max-0.5, -0.5])
    plt.colorbar(im, ax=ax4, label='|<m|x|n>|')

    ax4.set_xlabel('n')
    ax4.set_ylabel('m')
    ax4.set_title('Position Matrix Elements |<m|x|n>|')

    # Highlight selection rule (m = n +/- 1)
    ax4.text(0.5, 0.95, 'Selection rule:\n<m|x|n> != 0 only if m = n +/- 1',
             transform=ax4.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 5: Position and momentum uncertainties
    ax5 = axes[1, 1]

    n_range = np.arange(0, 15)
    delta_x = []
    delta_p = []
    uncertainty_product = []

    for n in n_range:
        psi = sho_eigenstate(n, x, m, omega, hbar)
        prob = psi**2

        # <x> = 0 by symmetry
        mean_x2 = trapezoid(x**2 * prob, x)
        sigma_x = np.sqrt(mean_x2)
        delta_x.append(sigma_x)

        # <p^2> from kinetic energy
        # For SHO: <T> = E/2, so <p^2>/2m = E/2
        E_n = sho_energy(n, omega, hbar)
        mean_p2 = 2 * m * E_n / 2  # From virial theorem
        sigma_p = np.sqrt(mean_p2)
        delta_p.append(sigma_p)

        uncertainty_product.append(sigma_x * sigma_p)

    ax5.plot(n_range, np.array(delta_x) / x0, 'b-o', lw=2, label='Delta x / x0')
    ax5.plot(n_range, np.array(delta_p) * x0 / hbar, 'r-s', lw=2, label='Delta p * x0 / hbar')

    # Theoretical values: Delta x = x0 * sqrt(n + 1/2)
    delta_x_theory = x0 * np.sqrt(n_range + 0.5)
    ax5.plot(n_range, delta_x_theory / x0, 'b--', alpha=0.5,
             label='Theory: sqrt(n + 1/2)')

    ax5.set_xlabel('Quantum number n')
    ax5.set_ylabel('Uncertainty (dimensionless)')
    ax5.set_title('Position and Momentum Uncertainties')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Momentum space wavefunctions
    ax6 = axes[1, 2]

    # FFT to momentum space
    from scipy.fft import fft, fftfreq, fftshift

    k = 2 * np.pi * fftfreq(N, dx)
    p = hbar * k
    p_sorted_idx = np.argsort(p)
    p_sorted = p[p_sorted_idx]

    # Characteristic momentum
    p0 = hbar / x0

    for n in range(4):
        psi_x = sho_eigenstate(n, x, m, omega, hbar)
        psi_p = fft(psi_x) * dx / np.sqrt(2 * np.pi)
        phi_p = np.abs(psi_p[p_sorted_idx])**2

        ax6.plot(p_sorted / p0, phi_p / max(phi_p) + n, color=colors[n], lw=2,
                label=f'n={n}')

    ax6.set_xlabel('Momentum p / p0')
    ax6.set_ylabel('|phi_n(p)|^2 + offset')
    ax6.set_title('Momentum Space Wavefunctions')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(-5, 5)

    # Note about momentum space form
    ax6.text(0.02, 0.98, 'phi_n(p) has same\nfunctional form as\npsi_n(x)!',
             transform=ax6.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Quantum Harmonic Oscillator: Complete Eigenstate Analysis\n'
                 r'$E_n = \hbar\omega(n + 1/2)$, $x_0 = \sqrt{\hbar/m\omega}$',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'sho_eigenstates.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'sho_eigenstates.png')}")


if __name__ == "__main__":
    main()
