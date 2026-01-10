"""
Experiment 154: Gaussian Wavepacket Spreading

This experiment demonstrates the spreading of a free Gaussian wavepacket,
including:
- Time evolution of wavepacket width
- Relationship between initial width and spreading rate
- Uncertainty principle manifestation
- Momentum and position space representations
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq


def gaussian_wavepacket(x: np.ndarray, x0: float, sigma0: float, k0: float) -> np.ndarray:
    """
    Create a normalized Gaussian wavepacket.

    psi(x, 0) = (2*pi*sigma0^2)^(-1/4) * exp(-(x-x0)^2 / (4*sigma0^2)) * exp(i*k0*x)

    Args:
        x: Position array
        x0: Initial center position
        sigma0: Initial width (standard deviation of |psi|^2)
        k0: Central wave number (momentum = hbar * k0)

    Returns:
        Complex wavefunction array
    """
    norm = (2 * np.pi * sigma0**2)**(-0.25)
    return norm * np.exp(-(x - x0)**2 / (4 * sigma0**2)) * np.exp(1j * k0 * x)


def evolve_wavepacket_fft(psi: np.ndarray, x: np.ndarray, t: float,
                          m: float = 1.0, hbar: float = 1.0) -> np.ndarray:
    """
    Evolve wavepacket using FFT method (exact for free particle).

    Args:
        psi: Initial wavefunction
        x: Position array
        t: Evolution time
        m: Particle mass
        hbar: Reduced Planck constant

    Returns:
        Evolved wavefunction
    """
    dx = x[1] - x[0]
    N = len(x)

    # FFT to momentum space
    psi_k = fft(psi)
    k = 2 * np.pi * fftfreq(N, dx)

    # Apply free particle propagator
    # exp(-i * hbar * k^2 * t / (2m))
    propagator = np.exp(-1j * hbar * k**2 * t / (2 * m))
    psi_k_evolved = psi_k * propagator

    # FFT back to position space
    psi_evolved = ifft(psi_k_evolved)

    return psi_evolved


def analytical_width(t: float, sigma0: float, m: float = 1.0, hbar: float = 1.0) -> float:
    """
    Analytical formula for wavepacket width at time t.

    sigma(t) = sigma0 * sqrt(1 + (hbar * t / (2 * m * sigma0^2))^2)
    """
    return sigma0 * np.sqrt(1 + (hbar * t / (2 * m * sigma0**2))**2)


def calculate_width(x: np.ndarray, prob: np.ndarray) -> float:
    """Calculate standard deviation of probability distribution."""
    dx = x[1] - x[0]
    norm = np.sum(prob) * dx
    mean_x = np.sum(x * prob) * dx / norm
    mean_x2 = np.sum(x**2 * prob) * dx / norm
    return np.sqrt(mean_x2 - mean_x**2)


def main():
    # Parameters (natural units)
    m = 1.0
    hbar = 1.0

    # Spatial grid
    L = 100.0
    N = 2048
    x = np.linspace(-L/2, L/2, N)
    dx = x[1] - x[0]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Plot 1: Wavepacket evolution snapshots
    ax1 = axes[0, 0]

    sigma0 = 1.0
    x0 = -20.0
    k0 = 5.0  # Initial momentum

    psi0 = gaussian_wavepacket(x, x0, sigma0, k0)

    times = [0, 2, 5, 10, 20]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(times)))

    for t, color in zip(times, colors):
        psi_t = evolve_wavepacket_fft(psi0, x, t, m, hbar)
        prob = np.abs(psi_t)**2
        ax1.plot(x, prob, color=color, lw=2, label=f't = {t}')

    ax1.set_xlabel('Position x')
    ax1.set_ylabel('|psi(x,t)|^2')
    ax1.set_title('Wavepacket Evolution (sigma0 = 1.0)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-30, 50)

    # Plot 2: Width vs time
    ax2 = axes[0, 1]

    t_array = np.linspace(0, 30, 100)
    sigmas_numerical = []
    sigmas_analytical = []

    for t in t_array:
        psi_t = evolve_wavepacket_fft(psi0, x, t, m, hbar)
        prob = np.abs(psi_t)**2
        sigma_num = calculate_width(x, prob)
        sigmas_numerical.append(sigma_num)
        sigmas_analytical.append(analytical_width(t, sigma0, m, hbar))

    ax2.plot(t_array, sigmas_numerical, 'b-', lw=2, label='Numerical')
    ax2.plot(t_array, sigmas_analytical, 'r--', lw=2, label='Analytical')

    ax2.set_xlabel('Time t')
    ax2.set_ylabel('Width sigma(t)')
    ax2.set_title('Wavepacket Spreading')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Effect of initial width
    ax3 = axes[0, 2]

    sigma0_values = [0.5, 1.0, 2.0, 4.0]
    colors_sigma = plt.cm.plasma(np.linspace(0.2, 0.9, len(sigma0_values)))

    for sigma0_test, color in zip(sigma0_values, colors_sigma):
        sigma_t = [analytical_width(t, sigma0_test, m, hbar) for t in t_array]
        ax3.plot(t_array, np.array(sigma_t) / sigma0_test, color=color, lw=2,
                label=f'sigma0 = {sigma0_test}')

    ax3.set_xlabel('Time t')
    ax3.set_ylabel('sigma(t) / sigma0')
    ax3.set_title('Spreading Rate vs Initial Width')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Highlight that narrower packets spread faster
    ax3.text(0.5, 0.95, 'Narrower packets spread faster!',
             transform=ax3.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 4: Position and momentum space comparison
    ax4 = axes[1, 0]

    sigma0 = 1.0
    psi0 = gaussian_wavepacket(x, 0, sigma0, k0=0)

    t_compare = 10.0
    psi_t = evolve_wavepacket_fft(psi0, x, t_compare, m, hbar)

    # Position space
    ax4.plot(x, np.abs(psi0)**2, 'b-', lw=2, label='t = 0')
    ax4.plot(x, np.abs(psi_t)**2, 'r-', lw=2, label=f't = {t_compare}')

    ax4.set_xlabel('Position x')
    ax4.set_ylabel('|psi(x)|^2')
    ax4.set_title('Position Space Spreading')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-15, 15)

    # Plot 5: Momentum space (unchanged for free particle)
    ax5 = axes[1, 1]

    # Momentum space representation
    k = 2 * np.pi * fftfreq(N, dx)
    k_sorted_idx = np.argsort(k)
    k_sorted = k[k_sorted_idx]

    psi0_k = fft(psi0)
    psi_t_k = fft(psi_t)

    # Normalize for display
    phi0 = np.abs(psi0_k[k_sorted_idx])**2
    phi_t = np.abs(psi_t_k[k_sorted_idx])**2

    ax5.plot(k_sorted, phi0 / max(phi0), 'b-', lw=2, label='t = 0')
    ax5.plot(k_sorted, phi_t / max(phi_t), 'r--', lw=2, label=f't = {t_compare}')

    ax5.set_xlabel('Wave number k')
    ax5.set_ylabel('|phi(k)|^2 (normalized)')
    ax5.set_title('Momentum Space (Unchanged!)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(-5, 5)

    # Plot 6: Uncertainty product
    ax6 = axes[1, 2]

    # Calculate position and momentum uncertainties
    delta_x = []
    delta_p = []
    uncertainty_product = []

    sigma0 = 1.0
    psi0 = gaussian_wavepacket(x, 0, sigma0, k0=0)

    for t in t_array:
        psi_t = evolve_wavepacket_fft(psi0, x, t, m, hbar)
        prob_x = np.abs(psi_t)**2

        # Position uncertainty
        sigma_x = calculate_width(x, prob_x)
        delta_x.append(sigma_x)

        # Momentum uncertainty (from momentum space)
        psi_k = fft(psi_t)
        prob_k = np.abs(psi_k)**2 * dx**2 / (2 * np.pi)

        # Calculate momentum spread
        mean_k = np.sum(k * prob_k) / np.sum(prob_k)
        mean_k2 = np.sum(k**2 * prob_k) / np.sum(prob_k)
        sigma_k = np.sqrt(mean_k2 - mean_k**2)
        sigma_p = hbar * sigma_k
        delta_p.append(sigma_p)

        uncertainty_product.append(sigma_x * sigma_p)

    ax6.plot(t_array, np.array(uncertainty_product) / (hbar/2), 'b-', lw=2)
    ax6.axhline(y=1.0, color='red', linestyle='--', alpha=0.7,
                label='Minimum (hbar/2)')

    ax6.set_xlabel('Time t')
    ax6.set_ylabel('(Delta x)(Delta p) / (hbar/2)')
    ax6.set_title('Uncertainty Product')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Minimum at t=0, increases with time
    ax6.annotate('Minimum uncertainty\nat t = 0',
                 xy=(0, 1), xytext=(5, 2),
                 arrowprops=dict(arrowstyle='->', color='black'),
                 fontsize=9)

    # Inset: delta_x and delta_p separately
    ax6_inset = ax6.inset_axes([0.5, 0.5, 0.45, 0.45])
    ax6_inset.plot(t_array, delta_x, 'b-', lw=1.5, label='Delta x')
    ax6_inset.plot(t_array, delta_p, 'r-', lw=1.5, label='Delta p')
    ax6_inset.set_xlabel('Time', fontsize=8)
    ax6_inset.set_ylabel('Uncertainty', fontsize=8)
    ax6_inset.legend(fontsize=7)
    ax6_inset.tick_params(labelsize=7)

    plt.suptitle('Free Particle Gaussian Wavepacket Spreading\n'
                 r'$\sigma(t) = \sigma_0\sqrt{1 + (\hbar t / 2m\sigma_0^2)^2}$',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'gaussian_wavepacket_spreading.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'gaussian_wavepacket_spreading.png')}")


if __name__ == "__main__":
    main()
