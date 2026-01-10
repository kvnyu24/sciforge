"""
Example demonstrating Fourier synthesis of a square wave with Gibbs phenomenon.

This example shows how a square wave can be approximated by summing sinusoidal
harmonics, and how the Gibbs phenomenon causes overshoot at discontinuities
that doesn't disappear even with infinite terms.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.waves import Wave


def square_wave(t, period=2*np.pi):
    """Ideal square wave with period."""
    return np.sign(np.sin(2 * np.pi * t / period))


def fourier_square_wave(t, n_terms, period=2*np.pi):
    """
    Fourier series approximation of square wave.

    Square wave = (4/pi) * sum_{n=1,3,5,...} sin(n*omega*t) / n

    Args:
        t: Time array
        n_terms: Number of terms (1, 2, 3, ... uses harmonics 1, 3, 5, ...)
        period: Wave period

    Returns:
        Approximate square wave
    """
    omega = 2 * np.pi / period
    result = np.zeros_like(t)

    for i in range(n_terms):
        n = 2 * i + 1  # Odd harmonics: 1, 3, 5, 7, ...
        result += (4 / np.pi) * np.sin(n * omega * t) / n

    return result


def main():
    # Time domain
    t = np.linspace(-np.pi, 3*np.pi, 2000)
    period = 2 * np.pi

    fig = plt.figure(figsize=(16, 14))

    # =========================================================================
    # Panel 1: Progressive approximation (few terms)
    # =========================================================================
    ax1 = fig.add_subplot(3, 3, 1)

    n_terms_list = [1, 2, 3, 5]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(n_terms_list)))

    # Plot ideal square wave
    sq = square_wave(t, period)
    ax1.plot(t / np.pi, sq, 'k--', lw=1, alpha=0.5, label='Ideal')

    for n, color in zip(n_terms_list, colors):
        approx = fourier_square_wave(t, n, period)
        ax1.plot(t / np.pi, approx, color=color, lw=1.5,
                 label=f'{n} term(s)')

    ax1.set_xlabel('t / pi')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Fourier Approximation: 1 to 5 Terms\n(Each term adds an odd harmonic)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-1, 3)
    ax1.set_ylim(-1.5, 1.5)

    # =========================================================================
    # Panel 2: More terms
    # =========================================================================
    ax2 = fig.add_subplot(3, 3, 2)

    n_terms_more = [10, 20, 50, 100]
    colors_more = plt.cm.plasma(np.linspace(0.1, 0.9, len(n_terms_more)))

    ax2.plot(t / np.pi, sq, 'k--', lw=1, alpha=0.5, label='Ideal')

    for n, color in zip(n_terms_more, colors_more):
        approx = fourier_square_wave(t, n, period)
        ax2.plot(t / np.pi, approx, color=color, lw=1,
                 label=f'{n} terms')

    ax2.set_xlabel('t / pi')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('More Terms: 10 to 100\n(Still overshoots at discontinuity!)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-1, 3)
    ax2.set_ylim(-1.5, 1.5)

    # =========================================================================
    # Panel 3: Gibbs phenomenon - zoom at discontinuity
    # =========================================================================
    ax3 = fig.add_subplot(3, 3, 3)

    # Zoom near t = 0
    t_zoom = np.linspace(-0.5, 0.5, 1000)
    sq_zoom = square_wave(t_zoom, period)

    ax3.plot(t_zoom / np.pi, sq_zoom, 'k--', lw=2, alpha=0.5, label='Ideal')

    for n, color in zip([10, 50, 200, 500], colors_more):
        approx = fourier_square_wave(t_zoom, n, period)
        ax3.plot(t_zoom / np.pi, approx, color=color, lw=1.5,
                 label=f'{n} terms')

    # Mark the Gibbs overshoot
    gibbs_overshoot = 1.0 + 2/np.pi * 0.281  # ~9% overshoot
    ax3.axhline(y=gibbs_overshoot, color='red', linestyle=':', alpha=0.7,
                label=f'Gibbs max: ~{gibbs_overshoot:.3f}')
    ax3.axhline(y=-gibbs_overshoot, color='red', linestyle=':', alpha=0.7)

    ax3.set_xlabel('t / pi')
    ax3.set_ylabel('Amplitude')
    ax3.set_title('Gibbs Phenomenon (Zoom at Discontinuity)\nOvershoot ~ 9% persists for any N!')
    ax3.legend(fontsize=8, loc='lower right')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-0.2, 0.2)
    ax3.set_ylim(-1.3, 1.3)

    # =========================================================================
    # Panel 4: Individual harmonics
    # =========================================================================
    ax4 = fig.add_subplot(3, 3, 4)

    t_harm = np.linspace(0, 2*np.pi, 500)
    harmonics = [1, 3, 5, 7, 9]
    colors_harm = plt.cm.rainbow(np.linspace(0, 0.8, len(harmonics)))

    for n, color in zip(harmonics, colors_harm):
        coef = 4 / (n * np.pi)
        harm = coef * np.sin(n * t_harm)
        ax4.plot(t_harm / np.pi, harm, color=color, lw=2,
                 label=f'n={n}: (4/{n}pi)*sin({n}t)')

    ax4.set_xlabel('t / pi')
    ax4.set_ylabel('Amplitude')
    ax4.set_title('Individual Fourier Components\n(Coefficients: 4/(n*pi) for odd n)')
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 2)

    # =========================================================================
    # Panel 5: Coefficient decay
    # =========================================================================
    ax5 = fig.add_subplot(3, 3, 5)

    n_vals = np.arange(1, 52, 2)  # Odd values 1, 3, 5, ..., 51
    coeffs = 4 / (n_vals * np.pi)

    ax5.bar(n_vals, coeffs, width=1.5, color='steelblue', alpha=0.7)
    ax5.plot(n_vals, 4 / (n_vals * np.pi), 'r-', lw=2, label='4/(n*pi)')

    ax5.set_xlabel('Harmonic Number n')
    ax5.set_ylabel('Fourier Coefficient')
    ax5.set_title('Coefficient Decay: 1/n\n(Slow decay causes Gibbs phenomenon)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 6: Error vs number of terms
    # =========================================================================
    ax6 = fig.add_subplot(3, 3, 6)

    n_range = np.arange(1, 101)
    t_err = np.linspace(0, 2*np.pi, 1000)
    sq_err = square_wave(t_err, period)

    rms_errors = []
    max_errors = []

    for n in n_range:
        approx = fourier_square_wave(t_err, n, period)
        error = approx - sq_err

        rms_errors.append(np.sqrt(np.mean(error**2)))
        max_errors.append(np.max(np.abs(error)))

    ax6.semilogy(n_range, rms_errors, 'b-', lw=2, label='RMS error')
    ax6.semilogy(n_range, max_errors, 'r--', lw=2, label='Max error')

    # Gibbs limit for max error
    gibbs_limit = 2/np.pi * 0.281 + 1  # Overshoot amount
    ax6.axhline(y=gibbs_limit - 1, color='orange', linestyle=':',
                label=f'Gibbs limit: {gibbs_limit-1:.3f}')

    ax6.set_xlabel('Number of Terms N')
    ax6.set_ylabel('Error')
    ax6.set_title('Error Convergence\n(Max error plateaus due to Gibbs)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 7: Other waveforms
    # =========================================================================
    ax7 = fig.add_subplot(3, 3, 7)

    t_wave = np.linspace(0, 4*np.pi, 1000)

    # Sawtooth wave Fourier series
    def fourier_sawtooth(t, n_terms):
        result = np.zeros_like(t)
        for n in range(1, n_terms + 1):
            result += ((-1)**(n+1)) * 2 * np.sin(n * t) / (n * np.pi)
        return result

    # Triangle wave Fourier series
    def fourier_triangle(t, n_terms):
        result = np.zeros_like(t)
        for i in range(n_terms):
            n = 2 * i + 1
            result += ((-1)**i) * 8 / (n**2 * np.pi**2) * np.sin(n * t)
        return result

    n_demo = 20
    sq_demo = fourier_square_wave(t_wave, n_demo, period)
    saw_demo = fourier_sawtooth(t_wave, n_demo * 2)
    tri_demo = fourier_triangle(t_wave, n_demo)

    ax7.plot(t_wave / np.pi, sq_demo, 'b-', lw=1.5, label='Square (1/n)')
    ax7.plot(t_wave / np.pi, saw_demo + 2.5, 'r-', lw=1.5, label='Sawtooth (1/n)')
    ax7.plot(t_wave / np.pi, tri_demo - 2.5, 'g-', lw=1.5, label='Triangle (1/n^2)')

    ax7.set_xlabel('t / pi')
    ax7.set_ylabel('Amplitude (offset)')
    ax7.set_title('Different Waveforms (N=20)\n(Smoother = faster convergence)')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.set_xlim(0, 4)

    # =========================================================================
    # Panel 8: Sigma approximation (Lanczos sigma factors)
    # =========================================================================
    ax8 = fig.add_subplot(3, 3, 8)

    def fourier_square_wave_sigma(t, n_terms, period=2*np.pi):
        """Lanczos sigma-approximation to reduce Gibbs phenomenon."""
        omega = 2 * np.pi / period
        result = np.zeros_like(t)

        for i in range(n_terms):
            n = 2 * i + 1
            # Lanczos sigma factor
            sigma = np.sinc((n) / (2 * n_terms + 1))
            result += sigma * (4 / np.pi) * np.sin(n * omega * t) / n

        return result

    n_sigma = 50
    t_sigma = np.linspace(-0.5, 0.5, 500)

    approx_gibbs = fourier_square_wave(t_sigma, n_sigma, period)
    approx_sigma = fourier_square_wave_sigma(t_sigma, n_sigma, period)
    sq_sigma = square_wave(t_sigma, period)

    ax8.plot(t_sigma / np.pi, sq_sigma, 'k--', lw=1, alpha=0.5, label='Ideal')
    ax8.plot(t_sigma / np.pi, approx_gibbs, 'b-', lw=1.5, label='Standard (Gibbs)')
    ax8.plot(t_sigma / np.pi, approx_sigma, 'r-', lw=1.5, label='Sigma factor (reduced)')

    ax8.axhline(y=gibbs_overshoot, color='blue', linestyle=':', alpha=0.5)
    ax8.axhline(y=1, color='red', linestyle=':', alpha=0.5)

    ax8.set_xlabel('t / pi')
    ax8.set_ylabel('Amplitude')
    ax8.set_title('Gibbs Reduction: Lanczos Sigma Factors\n(Smoother but slightly broadened)')
    ax8.legend(fontsize=8)
    ax8.grid(True, alpha=0.3)
    ax8.set_xlim(-0.2, 0.2)
    ax8.set_ylim(-1.3, 1.3)

    # =========================================================================
    # Panel 9: Frequency spectrum
    # =========================================================================
    ax9 = fig.add_subplot(3, 3, 9)

    # Discrete spectrum of square wave
    harmonics_spec = np.arange(1, 22, 2)  # 1, 3, 5, ..., 21
    amplitudes = 4 / (harmonics_spec * np.pi)

    ax9.bar(harmonics_spec, amplitudes, width=0.8, color='steelblue', alpha=0.8,
            label='|c_n| = 4/(n*pi)')

    # Mark fundamental
    ax9.annotate('Fundamental', xy=(1, amplitudes[0]),
                 xytext=(3, amplitudes[0] + 0.2),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=9, color='red')

    ax9.set_xlabel('Harmonic Number n')
    ax9.set_ylabel('Fourier Amplitude |c_n|')
    ax9.set_title('Frequency Spectrum of Square Wave\n(Only odd harmonics present)')
    ax9.legend()
    ax9.grid(True, alpha=0.3, axis='y')
    ax9.set_xlim(0, 22)

    plt.suptitle('Fourier Synthesis of Square Wave with Gibbs Phenomenon\n'
                 'f(t) = (4/pi) * [sin(t) + sin(3t)/3 + sin(5t)/5 + ...]',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'fourier_synthesis.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'fourier_synthesis.png')}")


if __name__ == "__main__":
    main()
