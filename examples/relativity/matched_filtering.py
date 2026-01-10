"""
Experiment 199: Matched Filtering for Gravitational Waves

This experiment demonstrates matched filtering - the technique used
to detect weak gravitational wave signals buried in detector noise.

Physical concepts:
- Signal-to-noise ratio optimization
- Template matching
- Frequency-domain filtering
- Detection statistics
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, fftconvolve
from scipy.fft import fft, ifft, fftfreq


# Physical constants
G = 6.67430e-11  # m^3/(kg*s^2)
c = 299792458.0  # m/s
M_sun = 1.989e30  # kg


def chirp_mass(m1, m2):
    """Calculate chirp mass."""
    return (m1 * m2)**(3/5) / (m1 + m2)**(1/5)


def generate_inspiral_signal(t, m1, m2, D, phi0=0, t_merger=None, G=G, c=c):
    """
    Generate inspiral waveform.

    Args:
        t: Time array
        m1, m2: Component masses
        D: Distance
        phi0: Initial phase
        t_merger: Time of merger in the array (default: end)

    Returns:
        Strain array
    """
    Mc = chirp_mass(m1, m2)

    if t_merger is None:
        t_merger = t[-1]

    # Time to merger
    tau = t_merger - t
    tau = np.maximum(tau, 1e-10)

    # Frequency
    f = (5/256)**(3/8) * (c**3 / (G * Mc))**(5/8) / np.pi * tau**(-3/8)

    # Only include frequencies above some minimum
    f_min = 20  # Hz
    valid = f >= f_min

    # Amplitude
    h0 = np.zeros_like(t)
    h0[valid] = (4/D) * (G * Mc / c**2)**(5/3) * (np.pi * f[valid] / c)**(2/3)

    # Phase
    phi = np.zeros_like(t)
    phi[valid] = phi0 - 2 * (5 * c**3 * tau[valid] / (256 * G * Mc))**(5/8)

    # Strain
    h = h0 * np.cos(phi)

    return h


def generate_noise(n_samples, fs, noise_asd):
    """
    Generate colored Gaussian noise with given ASD.

    Args:
        n_samples: Number of samples
        fs: Sample rate
        noise_asd: Function that returns ASD(f)

    Returns:
        Noise time series
    """
    # Generate white noise
    white = np.random.randn(n_samples)

    # FFT
    white_fft = fft(white)
    freqs = fftfreq(n_samples, 1/fs)

    # Color the noise
    asd = noise_asd(np.abs(freqs))
    asd[0] = 0  # No DC component

    colored_fft = white_fft * asd * np.sqrt(fs / 2)

    # IFFT to get colored noise
    return np.real(ifft(colored_fft))


def ligo_noise_asd(f):
    """
    Simplified LIGO noise ASD.

    Returns ASD in 1/sqrt(Hz)
    """
    f = np.atleast_1d(f).astype(float)
    asd = np.ones_like(f) * 1e-22

    # Seismic wall below 10 Hz
    low_f = f < 10
    asd[low_f] = 1e-19 * (10 / np.maximum(f[low_f], 0.1))**4

    # Low frequency rise
    mid_low = (f >= 10) & (f < 50)
    asd[mid_low] = 3e-23 * (50 / f[mid_low])**2

    # Bucket (most sensitive)
    bucket = (f >= 50) & (f < 300)
    asd[bucket] = 1e-23

    # High frequency rise
    high_f = f >= 300
    asd[high_f] = 1e-23 * (f[high_f] / 300)**2

    return asd


def matched_filter(data, template, psd):
    """
    Perform matched filtering.

    SNR(t) = 4 * Re[IFFT(d(f) * h*(f) / S_n(f))]

    Args:
        data: Time series data (signal + noise)
        template: Template waveform
        psd: Power spectral density

    Returns:
        SNR time series, optimal SNR
    """
    n = len(data)

    # FFT
    data_fft = fft(data)
    template_fft = fft(template)

    # Matched filter in frequency domain
    # SNR = 4 * Re[IFFT(d * h* / Sn)]
    integrand = data_fft * np.conj(template_fft) / psd

    # Normalization: sigma^2 = 4 * integral(|h|^2 / Sn df)
    sigma_sq = 4 * np.sum(np.abs(template_fft)**2 / psd) / n
    sigma = np.sqrt(sigma_sq)

    # SNR time series
    snr_complex = 4 * ifft(integrand) / sigma
    snr = np.abs(snr_complex)

    return snr, sigma


def inner_product(h1, h2, psd, fs):
    """
    Calculate noise-weighted inner product.

    <h1|h2> = 4 * Re[integral(h1*(f) * h2(f) / Sn(f) df)]
    """
    n = len(h1)
    h1_fft = fft(h1)
    h2_fft = fft(h2)

    df = fs / n
    integrand = np.conj(h1_fft) * h2_fft / psd

    return 4 * np.real(np.sum(integrand)) * df


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Parameters
    fs = 4096  # Sample rate (Hz)
    duration = 8  # seconds
    n_samples = int(fs * duration)
    t = np.linspace(0, duration, n_samples)

    # Binary parameters
    m1 = 30 * M_sun
    m2 = 30 * M_sun
    D = 400e6 * 3.086e16  # 400 Mpc
    t_merger = 6.0  # Merger at t = 6s

    # ==========================================================================
    # Generate signal and noise
    # ==========================================================================

    # Signal
    signal = generate_inspiral_signal(t, m1, m2, D, t_merger=t_merger)

    # Noise
    np.random.seed(42)
    noise = generate_noise(n_samples, fs, ligo_noise_asd)

    # Data = signal + noise
    data = signal + noise

    # Template (assume we know the parameters)
    template = generate_inspiral_signal(t, m1, m2, D, t_merger=t_merger)

    # PSD for matched filtering
    freqs = fftfreq(n_samples, 1/fs)
    psd = ligo_noise_asd(np.abs(freqs))**2
    psd[psd == 0] = 1e-50  # Avoid division by zero

    # ==========================================================================
    # Plot 1: Time series
    # ==========================================================================
    ax1 = axes[0, 0]

    ax1.plot(t, data, 'b-', lw=0.3, alpha=0.5, label='Data (signal + noise)')
    ax1.plot(t, signal * 100, 'r-', lw=1, label='Signal (x100)')

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Strain')
    ax1.set_title('Time Series: Signal Buried in Noise')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, duration)

    # Zoom inset
    axins = ax1.inset_axes([0.55, 0.55, 0.4, 0.4])
    mask = (t > 5.8) & (t < 6.2)
    axins.plot(t[mask], data[mask], 'b-', lw=0.5, alpha=0.5)
    axins.plot(t[mask], signal[mask] * 100, 'r-', lw=1)
    axins.set_title('Zoom near merger', fontsize=9)
    axins.grid(True, alpha=0.3)

    # ==========================================================================
    # Plot 2: Noise ASD and signal spectrum
    # ==========================================================================
    ax2 = axes[0, 1]

    f_plot = np.logspace(0, 4, 1000)
    asd = ligo_noise_asd(f_plot)

    ax2.loglog(f_plot, asd, 'b-', lw=2, label='LIGO noise ASD')

    # Signal characteristic strain
    Mc = chirp_mass(m1, m2)
    h_char = 2 * f_plot * (4/D) * (G * Mc / c**2)**(5/3) * (np.pi * f_plot / c)**(2/3)
    h_char[f_plot < 20] = 0

    ax2.loglog(f_plot, h_char, 'r-', lw=2, label='Signal characteristic strain')

    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Strain / sqrt(Hz)')
    ax2.set_title('Sensitivity and Signal')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_xlim(1, 5000)
    ax2.set_ylim(1e-25, 1e-18)

    # Mark sensitive band
    ax2.axvspan(20, 500, alpha=0.1, color='green')
    ax2.text(100, 1e-24, 'Sensitive band', fontsize=10, color='green')

    # ==========================================================================
    # Plot 3: Matched filter SNR
    # ==========================================================================
    ax3 = axes[1, 0]

    snr, sigma = matched_filter(data, template, psd)

    ax3.plot(t, snr, 'b-', lw=1)

    # Find peak
    peak_idx = np.argmax(snr)
    peak_snr = snr[peak_idx]
    peak_time = t[peak_idx]

    ax3.plot(peak_time, peak_snr, 'ro', markersize=10, label=f'Peak SNR = {peak_snr:.1f}')

    # Detection threshold
    threshold = 8
    ax3.axhline(y=threshold, color='green', linestyle='--', lw=2,
               label=f'Detection threshold (SNR = {threshold})')

    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('SNR')
    ax3.set_title('Matched Filter Output')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, duration)

    # ==========================================================================
    # Plot 4: Template bank / parameter recovery
    # ==========================================================================
    ax4 = axes[1, 1]

    # Try templates with different chirp masses
    Mc_true = chirp_mass(m1, m2) / M_sun

    Mc_range = np.linspace(Mc_true * 0.8, Mc_true * 1.2, 50)
    snr_peaks = []

    for Mc_test in Mc_range:
        # Generate template with this chirp mass
        # Approximate: scale masses equally
        m_test = Mc_test * M_sun * 2**(1/5)  # Equal mass case
        template_test = generate_inspiral_signal(t, m_test, m_test, D, t_merger=t_merger)

        snr_test, _ = matched_filter(data, template_test, psd)
        snr_peaks.append(np.max(snr_test))

    ax4.plot(Mc_range, snr_peaks, 'b-', lw=2)
    ax4.axvline(x=Mc_true, color='red', linestyle='--', lw=2,
               label=f'True Mc = {Mc_true:.1f} M_sun')

    # Find recovered value
    best_idx = np.argmax(snr_peaks)
    Mc_recovered = Mc_range[best_idx]
    ax4.plot(Mc_recovered, snr_peaks[best_idx], 'go', markersize=10,
            label=f'Recovered Mc = {Mc_recovered:.1f} M_sun')

    ax4.set_xlabel('Template chirp mass (M_sun)')
    ax4.set_ylabel('Peak SNR')
    ax4.set_title('Parameter Recovery: SNR vs Template Chirp Mass')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Matched Filtering for Gravitational Wave Detection\n'
                 'SNR = <d|h> / sqrt(<h|h>) where <a|b> is noise-weighted inner product',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Print summary
    print("Matched Filtering Summary:")
    print("=" * 60)

    print(f"\nSignal parameters:")
    print(f"  Masses: {m1/M_sun:.0f} + {m2/M_sun:.0f} M_sun")
    print(f"  Chirp mass: {Mc_true:.1f} M_sun")
    print(f"  Distance: {D/3.086e22:.0f} Mpc")

    print(f"\nDetection:")
    print(f"  Peak SNR: {peak_snr:.1f}")
    print(f"  Detection threshold: {threshold}")
    print(f"  Detected: {'Yes' if peak_snr > threshold else 'No'}")
    print(f"  Time of peak: {peak_time:.4f} s (true merger: {t_merger:.4f} s)")

    print(f"\nParameter recovery:")
    print(f"  True chirp mass: {Mc_true:.2f} M_sun")
    print(f"  Recovered chirp mass: {Mc_recovered:.2f} M_sun")
    print(f"  Error: {abs(Mc_recovered - Mc_true)/Mc_true * 100:.1f}%")

    print(f"\nMatched filtering explanation:")
    print(f"  - Correlates data with known template shape")
    print(f"  - Weights by inverse of noise PSD (more weight to quiet frequencies)")
    print(f"  - Optimal for detecting known signal in Gaussian noise")
    print(f"  - LIGO searches ~250,000 templates!")

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'matched_filtering.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")


if __name__ == "__main__":
    main()
