"""
Experiment 255: Frequency Comb Generation

This example demonstrates optical frequency comb generation from mode-locked
lasers. Frequency combs provide a direct link between optical and radio
frequencies and enable precision spectroscopy. We explore:
- Time-domain: train of ultrashort pulses
- Frequency-domain: comb of equally-spaced lines
- Carrier-envelope offset frequency (CEO)
- Self-referencing and f-2f interferometry
- Applications to optical clocks and spectroscopy

Key physics:
- Comb frequencies: f_n = n * f_rep + f_CEO
- f_rep: repetition rate = c / (2 * L_cavity)
- f_CEO: carrier-envelope offset frequency
- Self-referencing: measure f_CEO by comparing f and 2f
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.amo import C, HBAR

def mode_locked_pulse_train(t, f_rep, f_carrier, tau_pulse, phi_CEO=0, n_pulses=5):
    """
    Generate electric field of mode-locked pulse train.

    E(t) = sum_m A(t - m*T_rep) * exp(i * 2*pi*f_carrier*t + i*m*phi_CEO)

    Args:
        t: Time array
        f_rep: Repetition rate (Hz)
        f_carrier: Carrier (optical) frequency (Hz)
        tau_pulse: Pulse duration FWHM (s)
        phi_CEO: Carrier-envelope phase slip per pulse (rad)
        n_pulses: Number of pulses to include

    Returns:
        Complex electric field
    """
    T_rep = 1 / f_rep

    # Gaussian pulse envelope
    sigma = tau_pulse / (2 * np.sqrt(2 * np.log(2)))

    E = np.zeros_like(t, dtype=complex)

    for m in range(-n_pulses, n_pulses + 1):
        t_shifted = t - m * T_rep
        envelope = np.exp(-t_shifted**2 / (2 * sigma**2))
        carrier = np.exp(2j * np.pi * f_carrier * t + 1j * m * phi_CEO)
        E += envelope * carrier

    return E


def frequency_comb_spectrum(f, f_rep, f_CEO, f_center, bandwidth, n_lines=100):
    """
    Generate frequency comb spectrum.

    S(f) = sum_n delta(f - n*f_rep - f_CEO) * envelope(f)

    Args:
        f: Frequency array
        f_rep: Repetition rate
        f_CEO: Carrier-envelope offset frequency
        f_center: Center frequency of comb envelope
        bandwidth: FWHM bandwidth
        n_lines: Number of comb lines

    Returns:
        Spectral power
    """
    # Find mode number closest to center
    n_center = int((f_center - f_CEO) / f_rep)

    # Gaussian envelope
    sigma_f = bandwidth / (2 * np.sqrt(2 * np.log(2)))
    envelope = np.exp(-(f - f_center)**2 / (2 * sigma_f**2))

    # Sum Lorentzians for each comb line
    linewidth = f_rep / 1000  # Very narrow lines
    spectrum = np.zeros_like(f)

    for dn in range(-n_lines // 2, n_lines // 2 + 1):
        n = n_center + dn
        f_n = n * f_rep + f_CEO
        spectrum += envelope[np.argmin(np.abs(f - f_n))] * \
                   linewidth**2 / ((f - f_n)**2 + linewidth**2)

    return spectrum


def simulate_frequency_comb():
    """Simulate frequency comb generation and properties."""

    results = {}

    # Comb parameters (typical Ti:sapphire)
    f_rep = 1e9  # 1 GHz repetition rate
    f_carrier = 375e12  # ~800 nm carrier
    tau_pulse = 10e-15  # 10 fs pulse duration
    phi_CEO = np.pi / 4  # CEO phase slip

    f_CEO = f_rep * phi_CEO / (2 * np.pi)  # CEO frequency

    results['f_rep'] = f_rep
    results['f_carrier'] = f_carrier
    results['tau_pulse'] = tau_pulse
    results['phi_CEO'] = phi_CEO
    results['f_CEO'] = f_CEO

    print(f"Repetition rate: {f_rep/1e9:.2f} GHz")
    print(f"Pulse duration: {tau_pulse*1e15:.1f} fs")
    print(f"CEO frequency: {f_CEO/1e6:.1f} MHz")

    # 1. Time-domain pulse train
    print("\nGenerating time-domain pulse train...")
    T_rep = 1 / f_rep
    t_span = 5 * T_rep  # Show 5 pulses
    n_points = 10000
    t = np.linspace(-t_span/2, t_span/2, n_points)

    E_t = mode_locked_pulse_train(t, f_rep, f_carrier, tau_pulse, phi_CEO, n_pulses=3)

    results['time_domain'] = {
        't': t,
        'E': E_t,
        'envelope': np.abs(E_t),
        'intensity': np.abs(E_t)**2
    }

    # 2. Frequency-domain spectrum
    print("Computing frequency spectrum...")

    # Bandwidth corresponding to pulse duration
    bandwidth = 0.44 / tau_pulse  # ~44 THz for 10 fs

    # Frequency range around carrier
    f_span = 3 * bandwidth
    n_freq = 2000
    f = np.linspace(f_carrier - f_span/2, f_carrier + f_span/2, n_freq)

    spectrum = frequency_comb_spectrum(f, f_rep, f_CEO, f_carrier, bandwidth, n_lines=50)

    results['frequency_domain'] = {
        'f': f,
        'spectrum': spectrum,
        'bandwidth': bandwidth
    }

    # 3. Self-referencing (f-2f interferometry)
    print("Simulating f-2f self-referencing...")

    # Need octave-spanning spectrum
    # Low-frequency end: f_n = n * f_rep + f_CEO
    # High-frequency (doubled): 2 * f_m = 2 * m * f_rep + 2 * f_CEO

    # Beat between 2*f_m and f_{2m}: gives f_CEO
    # f_beat = 2*f_m - f_{2m} = (2m*f_rep + 2*f_CEO) - (2m*f_rep + f_CEO) = f_CEO

    # Simulate spectrum before and after SHG
    f_low = f_carrier / 2  # Fundamental at red end
    f_high = f_carrier  # SHG of red end = blue end of fundamental

    n_low = int((f_low - f_CEO) / f_rep)
    n_high = 2 * n_low  # Corresponding mode at 2f

    results['self_ref'] = {
        'f_low': f_low,
        'f_high': f_high,
        'n_low': n_low,
        'n_high': n_high,
        'f_beat': f_CEO
    }

    # 4. CEO frequency measurement
    print("Computing CEO frequency measurement...")

    # Time-domain beat signal
    t_beat = np.linspace(0, 10 / f_CEO, 1000)
    beat_signal = np.cos(2 * np.pi * f_CEO * t_beat)

    results['beat'] = {
        't': t_beat,
        'signal': beat_signal
    }

    # 5. Comb line positions
    print("Computing comb line frequencies...")

    # Show a few specific lines
    n_lines_show = 10
    n_center = int((f_carrier - f_CEO) / f_rep)
    n_values = np.arange(n_center - n_lines_show, n_center + n_lines_show + 1)
    f_lines = n_values * f_rep + f_CEO

    results['comb_lines'] = {
        'n': n_values,
        'f': f_lines,
        'n_center': n_center
    }

    # 6. Phase coherence across spectrum
    print("Analyzing phase coherence...")

    # All comb lines have defined phase relationship
    # phi_n = n * (2*pi*f_rep*t + phi_rep) + phi_CEO*t
    # The beat between any two lines is precisely known

    # Linewidth of individual comb line (after stabilization)
    linewidth_comb = 1  # Hz (sub-Hz linewidths possible)

    results['coherence'] = {
        'linewidth': linewidth_comb,
        'n_total': int(bandwidth / f_rep),
        'coverage': (f_carrier - bandwidth/2, f_carrier + bandwidth/2)
    }

    return results


def plot_results(results):
    """Create comprehensive visualization of frequency comb."""

    fig = plt.figure(figsize=(14, 12))

    # Plot 1: Time-domain pulse train
    ax1 = fig.add_subplot(2, 2, 1)
    td = results['time_domain']
    T_rep = 1 / results['f_rep']

    # Plot intensity
    ax1.plot(td['t'] * 1e9, td['intensity'] / np.max(td['intensity']), 'b-', linewidth=1)

    ax1.set_xlabel('Time (ns)', fontsize=11)
    ax1.set_ylabel('Normalized Intensity', fontsize=11)
    ax1.set_title('Mode-Locked Pulse Train (Time Domain)', fontsize=12)
    ax1.set_xlim(-2.5 * T_rep * 1e9, 2.5 * T_rep * 1e9)
    ax1.grid(True, alpha=0.3)

    # Add period annotation
    ax1.annotate('', xy=(T_rep * 1e9, 0.5), xytext=(0, 0.5),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax1.text(T_rep * 1e9 / 2, 0.55, f'$T_{{rep}}$ = {T_rep*1e9:.1f} ns',
            fontsize=10, ha='center', color='red')

    # Inset showing single pulse
    ax1_inset = ax1.inset_axes([0.6, 0.5, 0.35, 0.4])
    mask = np.abs(td['t']) < 100e-15  # +/- 100 fs
    ax1_inset.plot(td['t'][mask] * 1e15, td['intensity'][mask] / np.max(td['intensity']), 'b-')
    ax1_inset.set_xlabel('Time (fs)', fontsize=8)
    ax1_inset.set_title(f'Single pulse: {results["tau_pulse"]*1e15:.0f} fs', fontsize=8)
    ax1_inset.grid(True, alpha=0.3)

    # Plot 2: Frequency-domain comb
    ax2 = fig.add_subplot(2, 2, 2)
    fd = results['frequency_domain']

    # Convert to THz for display
    ax2.plot((fd['f'] - results['f_carrier']) / 1e12, fd['spectrum'] / np.max(fd['spectrum']),
            'b-', linewidth=0.5)

    ax2.set_xlabel('Frequency offset from carrier (THz)', fontsize=11)
    ax2.set_ylabel('Spectral Power (normalized)', fontsize=11)
    ax2.set_title('Frequency Comb Spectrum', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Add bandwidth annotation
    bw_THz = results['frequency_domain']['bandwidth'] / 1e12
    ax2.annotate('', xy=(bw_THz/2, 0.5), xytext=(-bw_THz/2, 0.5),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax2.text(0, 0.55, f'Bandwidth: {bw_THz:.1f} THz',
            fontsize=10, ha='center', color='red')

    # Inset showing individual lines
    ax2_inset = ax2.inset_axes([0.6, 0.5, 0.35, 0.4])
    # Zoom to show line spacing
    f_zoom_center = results['f_carrier']
    f_zoom_span = 20 * results['f_rep']
    f_zoom = np.linspace(f_zoom_center - f_zoom_span/2, f_zoom_center + f_zoom_span/2, 500)
    spec_zoom = frequency_comb_spectrum(f_zoom, results['f_rep'], results['f_CEO'],
                                        results['f_carrier'], results['frequency_domain']['bandwidth'],
                                        n_lines=10)
    ax2_inset.plot((f_zoom - f_zoom_center) / 1e9, spec_zoom / np.max(spec_zoom), 'b-')
    ax2_inset.set_xlabel('Offset (GHz)', fontsize=8)
    ax2_inset.set_title(f'$f_{{rep}}$ = {results["f_rep"]/1e9:.0f} GHz', fontsize=8)
    ax2_inset.grid(True, alpha=0.3)

    # Plot 3: Comb equation visualization
    ax3 = fig.add_subplot(2, 2, 3)

    # Draw comb lines
    cl = results['comb_lines']
    f_rep = results['f_rep']
    f_CEO = results['f_CEO']

    for i, (n, f) in enumerate(zip(cl['n'], cl['f'])):
        height = 0.5 + 0.4 * np.exp(-((n - cl['n_center']) / 5)**2)
        ax3.axvline(x=(f - results['f_carrier']) / 1e9, ymin=0, ymax=height,
                   color='blue', linewidth=1)

    # Mark f_rep
    ax3.annotate('', xy=(f_rep / 1e9, 0.3), xytext=(0, 0.3),
                arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    ax3.text(f_rep / 2e9, 0.35, '$f_{rep}$', fontsize=12, ha='center', color='green')

    # Mark f_CEO (offset from integer multiple)
    ax3.axvline(x=f_CEO / 1e9, color='red', linestyle='--', linewidth=2)
    ax3.text(f_CEO / 1e9 + 0.1, 0.95, '$f_{CEO}$', fontsize=12, color='red')

    ax3.set_xlabel('Frequency offset (GHz)', fontsize=11)
    ax3.set_ylabel('Amplitude', fontsize=11)
    ax3.set_title('Comb Equation: $f_n = n \\cdot f_{rep} + f_{CEO}$', fontsize=12)
    ax3.set_xlim(-10 * f_rep / 1e9, 10 * f_rep / 1e9)
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)

    # Add equation box
    textstr = '\n'.join([
        'Optical frequency comb:',
        f'$f_{{rep}} = {f_rep/1e9:.2f}$ GHz',
        f'$f_{{CEO}} = {f_CEO/1e6:.1f}$ MHz',
        f'Mode number: $n \\sim {cl["n_center"]:.0f}$'
    ])
    ax3.text(0.95, 0.95, textstr, transform=ax3.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 4: Self-referencing scheme
    ax4 = fig.add_subplot(2, 2, 4)

    # Draw octave-spanning comb schematic
    f_low = 0.5  # Normalized
    f_high = 1.0

    # Fundamental comb (lower octave)
    for i in range(10):
        f = f_low + i * 0.05
        ax4.axvline(x=f, ymin=0.1, ymax=0.4, color='blue', linewidth=2)
    ax4.text(f_low + 0.25, 0.5, 'Fundamental', fontsize=10, ha='center', color='blue')

    # SHG comb (2f)
    for i in range(10):
        f = 2 * (f_low + i * 0.05)
        if f <= 1.1:
            ax4.axvline(x=f, ymin=0.6, ymax=0.9, color='red', linewidth=2)
    ax4.text(1.0, 0.95, 'SHG (2f)', fontsize=10, ha='center', color='red')

    # Mark beat note
    ax4.annotate('', xy=(1.0, 0.5), xytext=(1.0, 0.55),
                arrowprops=dict(arrowstyle='-', color='green', lw=2))
    ax4.text(1.02, 0.52, '$f_{CEO}$', fontsize=11, color='green', fontweight='bold')

    ax4.set_xlabel('Frequency (normalized)', fontsize=11)
    ax4.set_title('f-2f Self-Referencing for $f_{CEO}$ Detection', fontsize=12)
    ax4.set_xlim(0.4, 1.2)
    ax4.set_ylim(0, 1.05)
    ax4.set_yticks([])
    ax4.grid(True, alpha=0.3, axis='x')

    # Add explanation
    textstr = '\n'.join([
        'Self-referencing:',
        '1. Octave-spanning spectrum',
        '2. SHG of low frequency end',
        '3. Beat with high frequency end',
        '4. Measure $f_{CEO}$ directly'
    ])
    ax4.text(0.02, 0.95, textstr, transform=ax4.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig


def main():
    print("=" * 60)
    print("Experiment 255: Frequency Comb Generation")
    print("=" * 60)
    print()

    # Run simulation
    print("Running simulations...")
    results = simulate_frequency_comb()

    # Create visualization
    print("\nCreating visualization...")
    fig = plot_results(results)

    # Save figure
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'frequency_comb.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"Repetition rate: f_rep = {results['f_rep']/1e9:.2f} GHz")
    print(f"CEO frequency: f_CEO = {results['f_CEO']/1e6:.1f} MHz")
    print(f"Pulse duration: {results['tau_pulse']*1e15:.0f} fs")
    print(f"Spectral bandwidth: {results['frequency_domain']['bandwidth']/1e12:.1f} THz")
    print()
    print("Key equations:")
    print("  f_n = n * f_rep + f_CEO")
    print("  f_rep = c / (2 * L_cavity) (repetition rate)")
    print("  f_CEO = (d_phi_CE / dt) / (2*pi) (carrier-envelope offset)")
    print()
    print("Number of comb lines: ~{:.0f}".format(results['coherence']['n_total']))
    print("Individual line width: ~{:.0f} Hz (stabilized)".format(results['coherence']['linewidth']))
    print()
    print("Applications:")
    print("  - Optical clocks: link optical to RF frequencies")
    print("  - Precision spectroscopy: absolute frequency measurements")
    print("  - Astronomy: calibrating spectrographs for exoplanet searches")
    print("  - Telecommunications: wavelength division multiplexing")
    print()
    print("Nobel Prize 2005: Hall and Hansch for frequency comb development")

    plt.close()


if __name__ == "__main__":
    main()
