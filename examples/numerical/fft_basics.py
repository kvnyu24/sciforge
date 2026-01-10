"""
Experiment 19: FFT basics - reconstruct a signal; windowing + spectral leakage.

Demonstrates FFT fundamentals including signal reconstruction,
windowing effects, and spectral leakage.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def create_signal(t, frequencies, amplitudes, phases=None):
    """Create a multi-frequency signal."""
    if phases is None:
        phases = np.zeros(len(frequencies))

    signal = np.zeros_like(t)
    for f, A, phi in zip(frequencies, amplitudes, phases):
        signal += A * np.sin(2 * np.pi * f * t + phi)
    return signal


def apply_window(signal, window_type='hann'):
    """Apply window function to signal."""
    n = len(signal)
    if window_type == 'rectangular':
        window = np.ones(n)
    elif window_type == 'hann':
        window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / n))
    elif window_type == 'hamming':
        window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n) / n)
    elif window_type == 'blackman':
        window = 0.42 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n) + \
                 0.08 * np.cos(4 * np.pi * np.arange(n) / n)
    else:
        window = np.ones(n)

    return signal * window, window


def compute_spectrum(signal, dt):
    """Compute power spectrum using FFT."""
    n = len(signal)
    fft = np.fft.fft(signal)
    freq = np.fft.fftfreq(n, dt)

    # Power spectrum (normalized)
    power = np.abs(fft)**2 / n

    return freq, fft, power


def main():
    # Parameters
    fs = 1000  # Sampling frequency
    T = 1.0    # Duration
    n = int(fs * T)
    t = np.linspace(0, T, n, endpoint=False)
    dt = 1 / fs

    # Create signal with known frequencies
    frequencies = [50, 120, 250]
    amplitudes = [1.0, 0.5, 0.3]

    signal_clean = create_signal(t, frequencies, amplitudes)

    # Add noise
    np.random.seed(42)
    noise = 0.5 * np.random.randn(n)
    signal_noisy = signal_clean + noise

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Time domain signal
    ax = axes[0, 0]
    ax.plot(t[:200], signal_clean[:200], 'b-', lw=1.5, label='Clean')
    ax.plot(t[:200], signal_noisy[:200], 'r-', lw=0.5, alpha=0.7, label='Noisy')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Time Domain Signal')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Power spectrum of clean signal
    ax = axes[0, 1]
    freq, fft_clean, power_clean = compute_spectrum(signal_clean, dt)

    positive = freq > 0
    ax.plot(freq[positive], power_clean[positive], 'b-', lw=1.5)

    # Mark true frequencies
    for f, A in zip(frequencies, amplitudes):
        ax.axvline(f, color='red', linestyle='--', alpha=0.5)
        ax.text(f + 5, np.max(power_clean) * 0.9, f'{f} Hz', fontsize=9)

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power')
    ax.set_title('Power Spectrum (Clean Signal)')
    ax.set_xlim(0, fs/2)
    ax.grid(True, alpha=0.3)

    # Plot 3: Signal reconstruction
    ax = axes[0, 2]

    # Keep only significant frequencies for reconstruction
    fft_filtered = fft_clean.copy()
    threshold = np.max(np.abs(fft_clean)) * 0.1
    fft_filtered[np.abs(fft_clean) < threshold] = 0

    signal_reconstructed = np.real(np.fft.ifft(fft_filtered))

    ax.plot(t[:200], signal_clean[:200], 'b-', lw=2, label='Original')
    ax.plot(t[:200], signal_reconstructed[:200], 'r--', lw=1.5, label='Reconstructed')

    error = np.sqrt(np.mean((signal_clean - signal_reconstructed)**2))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Signal Reconstruction (RMSE: {error:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Spectral leakage demonstration
    ax = axes[1, 0]

    # Create signal that doesn't fit exactly in window
    f_leak = 50.5  # Non-integer cycles in window
    signal_leak = np.sin(2 * np.pi * f_leak * t)

    freq_leak, _, power_leak = compute_spectrum(signal_leak, dt)

    ax.semilogy(freq_leak[freq_leak > 0], power_leak[freq_leak > 0] + 1e-10, 'b-', lw=1.5)
    ax.axvline(f_leak, color='red', linestyle='--', alpha=0.7, label=f'True f = {f_leak} Hz')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (log)')
    ax.set_title('Spectral Leakage (Non-integer Periods)')
    ax.set_xlim(30, 70)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Windowing effects
    ax = axes[1, 1]

    windows = ['rectangular', 'hann', 'hamming', 'blackman']
    colors = ['blue', 'red', 'green', 'orange']

    for window_type, color in zip(windows, colors):
        signal_windowed, window = apply_window(signal_leak, window_type)
        freq_w, _, power_w = compute_spectrum(signal_windowed, dt)

        # Normalize for comparison
        power_w = power_w / np.max(power_w)

        ax.semilogy(freq_w[freq_w > 0], power_w[freq_w > 0] + 1e-10,
                    '-', color=color, lw=1.5, label=window_type)

    ax.axvline(f_leak, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Normalized Power (log)')
    ax.set_title('Effect of Window Functions')
    ax.set_xlim(30, 70)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 6: Window functions
    ax = axes[1, 2]

    n_window = 100
    t_window = np.arange(n_window)

    for window_type, color in zip(windows, colors):
        _, window = apply_window(np.ones(n_window), window_type)
        ax.plot(t_window, window, '-', color=color, lw=2, label=window_type)

    ax.set_xlabel('Sample')
    ax.set_ylabel('Window Value')
    ax.set_title('Window Functions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('FFT Fundamentals: Reconstruction, Windowing, and Spectral Leakage',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'fft_basics.png'), dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/fft_basics.png")


if __name__ == "__main__":
    main()
