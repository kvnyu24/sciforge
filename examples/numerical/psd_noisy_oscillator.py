"""
Experiment 21: Power spectral density of noisy oscillator (Welch method).

Demonstrates PSD estimation for a damped driven oscillator with noise,
comparing periodogram and Welch methods.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def simulate_noisy_oscillator(t, omega0, gamma, F0, omega_d, noise_level):
    """
    Simulate a damped driven oscillator with noise.

    x'' + 2*gamma*x' + omega0^2*x = F0*cos(omega_d*t) + noise
    """
    dt = t[1] - t[0]
    n = len(t)

    x = np.zeros(n)
    v = np.zeros(n)

    # Random noise forcing
    np.random.seed(42)
    noise = noise_level * np.random.randn(n)

    # Velocity Verlet integration
    for i in range(1, n):
        # Driving force at this time
        F = F0 * np.cos(omega_d * t[i-1]) + noise[i-1]

        # Acceleration
        a = F - 2*gamma*v[i-1] - omega0**2 * x[i-1]

        # Position update
        x[i] = x[i-1] + v[i-1]*dt + 0.5*a*dt**2

        # New acceleration
        F_new = F0 * np.cos(omega_d * t[i]) + noise[i]
        a_new = F_new - 2*gamma*v[i-1] - omega0**2 * x[i]

        # Velocity update
        v[i] = v[i-1] + 0.5*(a + a_new)*dt

    return x, v


def periodogram(signal, dt):
    """Simple periodogram estimate of PSD."""
    n = len(signal)
    fft = np.fft.fft(signal)
    freq = np.fft.fftfreq(n, dt)

    # One-sided PSD
    psd = np.abs(fft)**2 / (n * (1/dt))

    return freq, psd


def welch_psd(signal, dt, nperseg=256, noverlap=None):
    """
    Welch's method for PSD estimation.

    Averages periodograms from overlapping segments with windowing.
    """
    if noverlap is None:
        noverlap = nperseg // 2

    n = len(signal)
    step = nperseg - noverlap

    # Number of segments
    n_segments = (n - noverlap) // step

    # Hann window
    window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(nperseg) / nperseg))
    window_sum = np.sum(window**2)

    # Frequency array
    freq = np.fft.fftfreq(nperseg, dt)

    # Average periodograms
    psd_sum = np.zeros(nperseg)

    for i in range(n_segments):
        start = i * step
        segment = signal[start:start + nperseg] * window
        fft_seg = np.fft.fft(segment)
        psd_sum += np.abs(fft_seg)**2

    psd = psd_sum / (n_segments * window_sum * (1/dt))

    return freq, psd


def theoretical_psd(freq, omega0, gamma, F0, omega_d, noise_var):
    """
    Theoretical PSD for driven damped oscillator.

    For white noise forcing, PSD is related to transfer function.
    """
    omega = 2 * np.pi * freq

    # Transfer function magnitude squared
    H2 = 1 / ((omega0**2 - omega**2)**2 + (2*gamma*omega)**2)

    # PSD from noise
    psd_noise = noise_var * H2

    # Delta function contribution from driving
    # (approximated as narrow peak)
    return psd_noise


def main():
    # Parameters
    omega0 = 2 * np.pi * 10  # Natural frequency 10 Hz
    gamma = 0.5  # Damping coefficient
    F0 = 1.0  # Driving amplitude
    omega_d = 2 * np.pi * 10.5  # Driving frequency 10.5 Hz
    noise_level = 0.5

    # Time array
    fs = 1000  # Sampling frequency
    T = 100    # Total time
    t = np.arange(0, T, 1/fs)
    dt = 1/fs

    # Simulate
    print("Simulating noisy oscillator...")
    x, v = simulate_noisy_oscillator(t, omega0, gamma, F0, omega_d, noise_level)

    # Compute PSDs
    print("Computing PSDs...")
    freq_per, psd_per = periodogram(x, dt)
    freq_welch, psd_welch = welch_psd(x, dt, nperseg=1024, noverlap=512)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Time series
    ax = axes[0, 0]
    t_show = t[:5000]  # First 5 seconds
    ax.plot(t_show, x[:5000], 'b-', lw=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('x(t)')
    ax.set_title('Noisy Oscillator Time Series')
    ax.grid(True, alpha=0.3)

    # Plot 2: Periodogram
    ax = axes[0, 1]
    positive = freq_per > 0
    ax.semilogy(freq_per[positive], psd_per[positive], 'b-', lw=0.5, alpha=0.5,
                label='Periodogram')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD')
    ax.set_title('Periodogram (Raw FFT)')
    ax.set_xlim(0, 50)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 3: Welch method
    ax = axes[1, 0]
    positive = freq_welch > 0
    ax.semilogy(freq_welch[positive], psd_welch[positive], 'r-', lw=1.5,
                label='Welch')

    # Mark frequencies
    f0 = omega0 / (2 * np.pi)
    fd = omega_d / (2 * np.pi)
    ax.axvline(f0, color='green', linestyle='--', alpha=0.7, label=f'f₀ = {f0:.1f} Hz')
    ax.axvline(fd, color='orange', linestyle='--', alpha=0.7, label=f'fᵈ = {fd:.1f} Hz')

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD')
    ax.set_title('Welch Method (Averaged, Windowed)')
    ax.set_xlim(0, 50)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Comparison
    ax = axes[1, 1]

    # Smooth periodogram for comparison
    from scipy.ndimage import uniform_filter1d
    psd_smooth = uniform_filter1d(psd_per[positive], size=50)

    ax.semilogy(freq_per[positive], psd_smooth, 'b-', lw=1, alpha=0.7,
                label='Periodogram (smoothed)')
    ax.semilogy(freq_welch[freq_welch > 0], psd_welch[freq_welch > 0], 'r-', lw=1.5,
                label='Welch')

    # Theoretical curve (approximate)
    f_theory = np.linspace(0.1, 50, 500)
    psd_theory = theoretical_psd(f_theory, omega0, gamma, F0, omega_d, noise_level**2)
    ax.semilogy(f_theory, psd_theory * 1e4, 'k--', lw=1, alpha=0.5,
                label='Theory (scaled)')

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD')
    ax.set_title('Comparison: Periodogram vs Welch')
    ax.set_xlim(0, 50)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Power Spectral Density of Noisy Damped Driven Oscillator\n' +
                 f'f₀ = {omega0/(2*np.pi):.1f} Hz, γ = {gamma}, fᵈ = {omega_d/(2*np.pi):.1f} Hz',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'psd_noisy_oscillator.png'), dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/psd_noisy_oscillator.png")


if __name__ == "__main__":
    main()
