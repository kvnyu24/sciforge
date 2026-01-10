"""
Experiment 168: Heisenberg Uncertainty Principle

This experiment demonstrates the Heisenberg uncertainty principle, including:
- Position-momentum uncertainty relation
- Minimum uncertainty states (Gaussian)
- Time-energy uncertainty
- Measurement disturbance interpretation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.integrate import trapezoid


def gaussian_wavepacket(x: np.ndarray, x0: float, sigma: float,
                        k0: float = 0) -> np.ndarray:
    """
    Create normalized Gaussian wavepacket.

    psi(x) = (2*pi*sigma^2)^(-1/4) * exp(-(x-x0)^2/(4*sigma^2)) * exp(i*k0*x)

    Args:
        x: Position array
        x0: Center position
        sigma: Width parameter (Delta x = sigma)
        k0: Central wave number

    Returns:
        Normalized wavefunction
    """
    norm = (2 * np.pi * sigma**2)**(-0.25)
    return norm * np.exp(-(x - x0)**2 / (4 * sigma**2)) * np.exp(1j * k0 * x)


def calculate_uncertainties(psi: np.ndarray, x: np.ndarray,
                            hbar: float = 1.0) -> tuple:
    """
    Calculate position and momentum uncertainties.

    Delta x = sqrt(<x^2> - <x>^2)
    Delta p = sqrt(<p^2> - <p>^2)

    Args:
        psi: Wavefunction
        x: Position array
        hbar: Reduced Planck constant

    Returns:
        Tuple of (Delta_x, Delta_p, uncertainty_product)
    """
    dx = x[1] - x[0]
    prob = np.abs(psi)**2

    # Position expectation values
    x_mean = trapezoid(x * prob, x)
    x2_mean = trapezoid(x**2 * prob, x)
    delta_x = np.sqrt(x2_mean - x_mean**2)

    # Momentum space
    N = len(x)
    k = 2 * np.pi * fftfreq(N, dx)
    p = hbar * k

    psi_p = fft(psi) * dx / np.sqrt(2 * np.pi)
    prob_p = np.abs(psi_p)**2

    # Momentum expectation values
    p_mean = np.sum(p * prob_p) / np.sum(prob_p)
    p2_mean = np.sum(p**2 * prob_p) / np.sum(prob_p)
    delta_p = np.sqrt(p2_mean - p_mean**2)

    return delta_x, delta_p, delta_x * delta_p


def non_gaussian_wavefunction(x: np.ndarray, func_type: str = 'box') -> np.ndarray:
    """
    Create non-Gaussian wavefunctions to compare uncertainty products.

    Args:
        x: Position array
        func_type: 'box', 'triangle', 'sech'

    Returns:
        Normalized wavefunction
    """
    if func_type == 'box':
        # Box function
        a = 2.0
        psi = np.zeros_like(x)
        psi[np.abs(x) < a] = 1.0
    elif func_type == 'triangle':
        # Triangular function
        a = 2.0
        psi = np.zeros_like(x)
        mask = np.abs(x) < a
        psi[mask] = 1 - np.abs(x[mask]) / a
    elif func_type == 'sech':
        # Sech function (also minimum uncertainty)
        a = 1.0
        psi = 1 / np.cosh(x / a)
    elif func_type == 'lorentzian':
        # Lorentzian (not square-integrable strictly, but for comparison)
        a = 1.0
        psi = 1 / (1 + (x / a)**2)
    else:
        psi = np.ones_like(x)

    # Normalize
    dx = x[1] - x[0]
    norm = np.sqrt(trapezoid(np.abs(psi)**2, x))
    return psi / norm


def energy_time_uncertainty(E_width: float, hbar: float = 1.0) -> float:
    """
    Calculate minimum lifetime from energy uncertainty.

    Delta E * Delta t >= hbar / 2
    Delta t >= hbar / (2 * Delta E)

    Args:
        E_width: Energy uncertainty
        hbar: Reduced Planck constant

    Returns:
        Minimum time uncertainty
    """
    return hbar / (2 * E_width)


def main():
    # Parameters
    hbar = 1.0

    # Spatial grid
    x_max = 20.0
    N = 2048
    x = np.linspace(-x_max, x_max, N)
    dx = x[1] - x[0]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Plot 1: Position and momentum distributions for Gaussian
    ax1 = axes[0, 0]

    sigma = 1.0
    psi = gaussian_wavepacket(x, 0, sigma)
    prob_x = np.abs(psi)**2

    # Momentum space
    k = 2 * np.pi * fftfreq(N, dx)
    p = hbar * k
    p_sorted_idx = np.argsort(p)
    p_sorted = p[p_sorted_idx]

    psi_p = fft(psi) * dx / np.sqrt(2 * np.pi)
    prob_p = np.abs(psi_p[p_sorted_idx])**2

    ax1_twin = ax1.twinx()

    l1, = ax1.plot(x, prob_x, 'b-', lw=2, label='Position |psi(x)|^2')
    ax1.fill_between(x, 0, prob_x, alpha=0.2, color='blue')

    l2, = ax1_twin.plot(p_sorted, prob_p / max(prob_p) * max(prob_x), 'r-', lw=2,
                        label='Momentum |psi(p)|^2')
    ax1_twin.fill_between(p_sorted, 0, prob_p / max(prob_p) * max(prob_x),
                          alpha=0.2, color='red')

    ax1.set_xlabel('Position x / Momentum p')
    ax1.set_ylabel('Position Probability', color='blue')
    ax1_twin.set_ylabel('Momentum Probability (scaled)', color='red')
    ax1.set_title('Conjugate Distributions (Gaussian)')
    ax1.legend([l1, l2], ['Position', 'Momentum'])
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-5, 5)

    delta_x, delta_p, product = calculate_uncertainties(psi, x, hbar)
    ax1.text(0.02, 0.98, f'Delta x = {delta_x:.3f}\nDelta p = {delta_p:.3f}\n'
             f'Product = {product:.3f} hbar\n(Min = 0.5 hbar)',
             transform=ax1.transAxes, va='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Plot 2: Uncertainty product vs wavepacket width
    ax2 = axes[0, 1]

    sigma_range = np.linspace(0.2, 5, 50)
    products = []
    delta_xs = []
    delta_ps = []

    for sigma in sigma_range:
        psi = gaussian_wavepacket(x, 0, sigma)
        dx_i, dp_i, prod = calculate_uncertainties(psi, x, hbar)
        products.append(prod)
        delta_xs.append(dx_i)
        delta_ps.append(dp_i)

    ax2.plot(sigma_range, products, 'b-', lw=2, label='Delta x * Delta p')
    ax2.axhline(y=hbar/2, color='red', linestyle='--', lw=2,
                label='Minimum = hbar/2')

    ax2.set_xlabel('Gaussian Width sigma')
    ax2.set_ylabel('Uncertainty Product (hbar)')
    ax2.set_title('Gaussian: Minimum Uncertainty State')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Inset: delta_x and delta_p
    ax2_inset = ax2.inset_axes([0.5, 0.5, 0.45, 0.45])
    ax2_inset.plot(sigma_range, delta_xs, 'b-', lw=1.5, label='Delta x')
    ax2_inset.plot(sigma_range, delta_ps, 'r-', lw=1.5, label='Delta p')
    ax2_inset.set_xlabel('sigma', fontsize=8)
    ax2_inset.set_ylabel('Uncertainty', fontsize=8)
    ax2_inset.legend(fontsize=7)
    ax2_inset.tick_params(labelsize=7)

    # Plot 3: Comparison of different wavefunctions
    ax3 = axes[0, 2]

    wavefunction_types = ['box', 'triangle', 'sech', 'lorentzian']
    colors_wf = plt.cm.tab10(np.linspace(0, 0.3, len(wavefunction_types)))

    products_wf = []
    labels_wf = []

    for wf_type, color in zip(wavefunction_types, colors_wf):
        psi = non_gaussian_wavefunction(x, wf_type)
        _, _, prod = calculate_uncertainties(psi, x, hbar)
        products_wf.append(prod)
        labels_wf.append(f'{wf_type}\n(prod={prod:.2f})')

        ax3.plot(x, np.abs(psi)**2, color=color, lw=2, label=wf_type)

    ax3.set_xlabel('Position x')
    ax3.set_ylabel('Probability Density')
    ax3.set_title('Non-Gaussian Wavefunctions')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-5, 5)

    # Plot 4: Bar chart of uncertainty products
    ax4 = axes[1, 0]

    # Add Gaussian for comparison
    psi_gauss = gaussian_wavepacket(x, 0, 1.0)
    _, _, prod_gauss = calculate_uncertainties(psi_gauss, x, hbar)
    products_all = [prod_gauss] + products_wf
    labels_all = ['Gaussian'] + wavefunction_types

    colors_bar = ['green'] + list(colors_wf)
    bars = ax4.bar(range(len(products_all)), products_all, color=colors_bar, alpha=0.7)

    ax4.axhline(y=hbar/2, color='red', linestyle='--', lw=2, label='Minimum hbar/2')
    ax4.set_xticks(range(len(products_all)))
    ax4.set_xticklabels(labels_all, rotation=45, ha='right')
    ax4.set_ylabel('Delta x * Delta p (hbar)')
    ax4.set_title('Uncertainty Product Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # Annotate values
    for bar, prod in zip(bars, products_all):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{prod:.2f}', ha='center', fontsize=9)

    # Plot 5: Narrowing in position increases momentum spread
    ax5 = axes[1, 1]

    sigma_values = [2.0, 1.0, 0.5, 0.25]
    colors_sig = plt.cm.viridis(np.linspace(0.1, 0.9, len(sigma_values)))

    for sigma, color in zip(sigma_values, colors_sig):
        psi = gaussian_wavepacket(x, 0, sigma)
        psi_p = fft(psi) * dx / np.sqrt(2 * np.pi)
        prob_p = np.abs(psi_p[p_sorted_idx])**2

        delta_x, delta_p, _ = calculate_uncertainties(psi, x, hbar)

        ax5.plot(p_sorted, prob_p / max(prob_p), color=color, lw=2,
                label=f'sigma={sigma}, Delta p={delta_p:.2f}')

    ax5.set_xlabel('Momentum p')
    ax5.set_ylabel('Momentum Probability (normalized)')
    ax5.set_title('Narrower Position => Wider Momentum')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(-5, 5)

    # Plot 6: Time-energy uncertainty
    ax6 = axes[1, 2]

    # Show exponential decay of unstable state
    tau = 1.0  # Lifetime
    t = np.linspace(0, 5*tau, 500)

    # Survival probability
    P_survival = np.exp(-t / tau)

    # Energy spectrum (Lorentzian)
    E0 = 5.0  # Central energy
    Gamma = hbar / tau  # Natural width

    E = np.linspace(E0 - 5*Gamma, E0 + 5*Gamma, 500)
    spectral_density = (Gamma / (2*np.pi)) / ((E - E0)**2 + (Gamma/2)**2)

    ax6_twin = ax6.twinx()

    l1, = ax6.plot(t / tau, P_survival, 'b-', lw=2, label='Decay probability')
    l2, = ax6_twin.plot(E, spectral_density, 'r-', lw=2, label='Energy spectrum')

    ax6.set_xlabel('Time t/tau / Energy E')
    ax6.set_ylabel('Survival Probability', color='blue')
    ax6_twin.set_ylabel('Spectral Density', color='red')
    ax6.set_title('Time-Energy Uncertainty\nDelta E * Delta t ~ hbar')
    ax6.legend([l1, l2], ['Decay P(t)', 'Spectrum rho(E)'])
    ax6.grid(True, alpha=0.3)

    # Add annotation
    ax6.text(0.95, 0.95, f'Lifetime tau = {tau}\nWidth Gamma = {Gamma:.2f}\n'
             f'tau * Gamma = {tau * Gamma:.2f} hbar',
             transform=ax6.transAxes, ha='right', va='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle('Heisenberg Uncertainty Principle\n'
                 r'$\Delta x \cdot \Delta p \geq \frac{\hbar}{2}$, '
                 r'$\Delta E \cdot \Delta t \geq \frac{\hbar}{2}$',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'heisenberg_uncertainty.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'heisenberg_uncertainty.png')}")


if __name__ == "__main__":
    main()
