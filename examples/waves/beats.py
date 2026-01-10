"""
Example demonstrating acoustic beats from close frequency interference.

This example shows how two waves with slightly different frequencies
create a beating pattern, where the amplitude oscillates at the
difference frequency (beat frequency).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.waves import Wave


def main():
    # Time domain
    t = np.linspace(0, 5, 10000)  # 5 seconds with fine resolution

    # Base frequency and different frequency differences
    f1 = 440  # Hz (A4 note)

    fig = plt.figure(figsize=(16, 14))

    # =========================================================================
    # Panel 1: Two close frequencies and their superposition
    # =========================================================================
    ax1 = fig.add_subplot(3, 3, 1)

    f2 = 444  # 4 Hz difference
    delta_f = f2 - f1

    # For visualization, we'll show a shorter time window with lower frequencies
    t_short = np.linspace(0, 1, 2000)
    f1_vis = 20
    f2_vis = 22
    delta_f_vis = f2_vis - f1_vis

    wave1 = np.sin(2 * np.pi * f1_vis * t_short)
    wave2 = np.sin(2 * np.pi * f2_vis * t_short)
    superposition = wave1 + wave2

    ax1.plot(t_short, wave1, 'b-', lw=1, alpha=0.5, label=f'f1 = {f1_vis} Hz')
    ax1.plot(t_short, wave2, 'r-', lw=1, alpha=0.5, label=f'f2 = {f2_vis} Hz')
    ax1.plot(t_short, superposition, 'k-', lw=1.5, label='Sum')

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'Two Waves with Close Frequencies\n(f1 = {f1_vis} Hz, f2 = {f2_vis} Hz)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)

    # =========================================================================
    # Panel 2: Beat pattern with envelope
    # =========================================================================
    ax2 = fig.add_subplot(3, 3, 2)

    # The beat formula: sin(2*pi*f1*t) + sin(2*pi*f2*t) =
    # 2*cos(pi*(f2-f1)*t) * sin(pi*(f1+f2)*t)
    f_avg = (f1_vis + f2_vis) / 2
    f_beat = delta_f_vis / 2

    carrier = np.sin(2 * np.pi * f_avg * t_short)
    envelope = 2 * np.cos(2 * np.pi * f_beat * t_short)

    ax2.fill_between(t_short, -np.abs(envelope), np.abs(envelope),
                     alpha=0.3, color='green', label='Envelope')
    ax2.plot(t_short, superposition, 'k-', lw=0.5, label='Beat pattern')
    ax2.plot(t_short, envelope, 'g-', lw=2, label=f'Beat freq = {delta_f_vis} Hz')
    ax2.plot(t_short, -envelope, 'g-', lw=2)

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('Beat Pattern with Envelope\nf_beat = |f2 - f1|')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-2.5, 2.5)

    # =========================================================================
    # Panel 3: Different beat frequencies
    # =========================================================================
    ax3 = fig.add_subplot(3, 3, 3)

    freq_diffs = [1, 2, 4, 8]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(freq_diffs)))

    for delta, color in zip(freq_diffs, colors):
        f2_var = f1_vis + delta
        wave_beat = np.sin(2 * np.pi * f1_vis * t_short) + \
                    np.sin(2 * np.pi * f2_var * t_short)
        # Offset for visibility
        offset = freq_diffs.index(delta) * 5

        ax3.plot(t_short, wave_beat + offset, color=color, lw=1,
                 label=f'delta f = {delta} Hz')

    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Amplitude (offset)')
    ax3.set_title('Different Beat Frequencies\n(Faster beats with larger frequency difference)')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 1)

    # =========================================================================
    # Panel 4: Frequency spectrum
    # =========================================================================
    ax4 = fig.add_subplot(3, 3, 4)

    # FFT of beat signal
    from scipy.fft import fft, fftfreq

    # Use actual musical frequencies for FFT
    t_fft = np.linspace(0, 1, 44100)  # 1 second at audio sample rate
    f1_music = 440
    f2_music = 444

    signal = np.sin(2 * np.pi * f1_music * t_fft) + \
             np.sin(2 * np.pi * f2_music * t_fft)

    N = len(t_fft)
    yf = fft(signal)
    xf = fftfreq(N, 1/44100)

    # Only positive frequencies
    mask = xf > 0
    ax4.plot(xf[mask], 2/N * np.abs(yf[mask]), 'b-', lw=1)
    ax4.axvline(x=f1_music, color='r', linestyle='--', alpha=0.5,
                label=f'f1 = {f1_music} Hz')
    ax4.axvline(x=f2_music, color='g', linestyle='--', alpha=0.5,
                label=f'f2 = {f2_music} Hz')

    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Amplitude')
    ax4.set_title('Frequency Spectrum of Beat Signal\n(Two distinct peaks)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(420, 460)

    # =========================================================================
    # Panel 5: Musical tuning example
    # =========================================================================
    ax5 = fig.add_subplot(3, 3, 5)

    # Piano tuning: A4 = 440 Hz, nearly-tuned A4 at different offsets
    offsets = [0, 0.5, 1, 2, 5]
    t_tune = np.linspace(0, 2, 4000)

    for i, offset in enumerate(offsets):
        f_ref = 440
        f_test = 440 + offset

        beat = np.sin(2 * np.pi * f_ref * t_tune) + \
               np.sin(2 * np.pi * f_test * t_tune)

        # Calculate envelope for display
        if offset > 0:
            env = 2 * np.cos(np.pi * offset * t_tune)
        else:
            env = 2 * np.ones_like(t_tune)

        ax5.fill_between(t_tune, -np.abs(env) + i*5, np.abs(env) + i*5,
                         alpha=0.5, label=f'Off by {offset} Hz')

    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Envelope (offset)')
    ax5.set_title('Piano Tuning: Beat Rate\n(Slower beats = closer to in-tune)')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0, 2)

    # =========================================================================
    # Panel 6: Beat frequency vs frequency difference
    # =========================================================================
    ax6 = fig.add_subplot(3, 3, 6)

    delta_f_range = np.linspace(0, 20, 100)
    beat_freq = delta_f_range  # Beat frequency = |f2 - f1|

    ax6.plot(delta_f_range, beat_freq, 'b-', lw=3)
    ax6.fill_between(delta_f_range, beat_freq, alpha=0.3)

    # Mark human perception thresholds
    ax6.axvline(x=7, color='r', linestyle='--', alpha=0.7,
                label='~7 Hz: Roughness threshold')
    ax6.axvline(x=15, color='orange', linestyle='--', alpha=0.7,
                label='~15 Hz: Two tones perceived')

    ax6.set_xlabel('Frequency Difference |f2 - f1| (Hz)')
    ax6.set_ylabel('Beat Frequency (Hz)')
    ax6.set_title('Beat Frequency = |f2 - f1|\n(Linear relationship)')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 7: Intensity modulation
    # =========================================================================
    ax7 = fig.add_subplot(3, 3, 7)

    # Intensity = amplitude^2
    t_int = np.linspace(0, 2, 4000)
    f1_int = 20
    delta_int = 2

    signal_int = np.sin(2 * np.pi * f1_int * t_int) + \
                 np.sin(2 * np.pi * (f1_int + delta_int) * t_int)

    # Envelope squared gives intensity
    envelope_int = 2 * np.cos(np.pi * delta_int * t_int)
    intensity = envelope_int**2

    ax7.plot(t_int, signal_int**2, 'b-', lw=0.5, alpha=0.5, label='Instantaneous I')
    ax7.plot(t_int, intensity, 'r-', lw=2, label='Envelope^2')

    # Average intensity
    avg_intensity = 2  # For two waves of unit amplitude
    ax7.axhline(y=avg_intensity, color='green', linestyle='--',
                label=f'Average I = {avg_intensity}')

    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('Intensity (a.u.)')
    ax7.set_title('Intensity Modulation in Beats\nI varies from 0 to 4 (for unit amplitudes)')
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)
    ax7.set_xlim(0, 2)
    ax7.set_ylim(0, 5)

    # =========================================================================
    # Panel 8: Three-frequency beats (complex pattern)
    # =========================================================================
    ax8 = fig.add_subplot(3, 3, 8)

    f_base = 20
    t_3f = np.linspace(0, 2, 4000)

    # Three close frequencies
    freqs = [f_base, f_base + 1, f_base + 3]
    signal_3f = sum(np.sin(2 * np.pi * f * t_3f) for f in freqs)

    ax8.plot(t_3f, signal_3f, 'b-', lw=1)
    ax8.fill_between(t_3f, signal_3f, alpha=0.3)

    ax8.set_xlabel('Time (s)')
    ax8.set_ylabel('Amplitude')
    ax8.set_title(f'Three Frequencies: {freqs}\n(Complex beat pattern)')
    ax8.grid(True, alpha=0.3)
    ax8.set_xlim(0, 2)

    # =========================================================================
    # Panel 9: Space-time diagram of beats
    # =========================================================================
    ax9 = fig.add_subplot(3, 3, 9)

    # Beating waves in space (two counter-propagating waves with different k)
    x = np.linspace(0, 20, 500)
    t_st = np.linspace(0, 5, 200)

    k1 = 2 * np.pi  # wavelength = 1
    k2 = 2 * np.pi * 1.1  # slightly different wavelength

    omega1 = k1  # Same velocity
    omega2 = k2

    X, T = np.meshgrid(x, t_st)
    wave1_st = np.sin(k1 * X - omega1 * T)
    wave2_st = np.sin(k2 * X - omega2 * T)
    beats_st = wave1_st + wave2_st

    im = ax9.imshow(beats_st, aspect='auto',
                    extent=[x.min(), x.max(), t_st.max(), t_st.min()],
                    cmap='RdBu', vmin=-2, vmax=2)
    ax9.set_xlabel('Position x')
    ax9.set_ylabel('Time t')
    ax9.set_title('Spatial Beats (Different Wavelengths)\nBeat wavelength = 2*pi / |k2 - k1|')
    plt.colorbar(im, ax=ax9, label='Amplitude')

    plt.suptitle('Acoustic Beats: Interference of Close Frequencies\n'
                 'u(t) = sin(2*pi*f1*t) + sin(2*pi*f2*t) = 2*cos(pi*(f2-f1)*t)*sin(pi*(f1+f2)*t)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'beats.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'beats.png')}")


if __name__ == "__main__":
    main()
