"""
Example 112: Fabry-Perot Cavity

This example demonstrates the Fabry-Perot interferometer/etalon,
showing transmission resonances, linewidth, and spectroscopic applications.

Physics:
    Transmission through a Fabry-Perot cavity:
    T = 1 / (1 + F * sin^2(delta/2))

    where:
    - F = 4R/(1-R)^2 is the coefficient of finesse
    - delta = 4*pi*n*d*cos(theta)/lambda is the round-trip phase
    - R is the mirror reflectance

    Key parameters:
    - Finesse: F_f = pi*sqrt(R)/(1-R) ≈ pi*sqrt(F)/2
    - Free spectral range: FSR = c/(2*n*d)
    - Linewidth: delta_nu = FSR / F_f
    - Resolving power: R_p = m * F_f (where m is the order)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.optics import FabryPerotInterferometer


class EnhancedFabryPerot:
    """Enhanced Fabry-Perot cavity with detailed analysis"""

    def __init__(self, spacing: float, reflectance: float, n_medium: float = 1.0):
        """
        Args:
            spacing: Mirror spacing (m)
            reflectance: Mirror reflectance (0 to 1)
            n_medium: Refractive index between mirrors
        """
        self.d = spacing
        self.R = reflectance
        self.n = n_medium

        # Speed of light
        self.c = 299792458.0

        # Derived quantities
        self.F = 4 * self.R / (1 - self.R)**2  # Coefficient of finesse
        self.finesse = np.pi * np.sqrt(self.R) / (1 - self.R)
        self.fsr_freq = self.c / (2 * self.n * self.d)  # FSR in Hz
        self.linewidth = self.fsr_freq / self.finesse  # FWHM in Hz

    def transmission(self, wavelength: float, angle: float = 0.0) -> float:
        """Calculate transmission at given wavelength and angle"""
        delta = 4 * np.pi * self.n * self.d * np.cos(angle) / wavelength
        return 1.0 / (1 + self.F * np.sin(delta / 2)**2)

    def transmission_vs_frequency(self, frequency: np.ndarray) -> np.ndarray:
        """Calculate transmission spectrum vs frequency"""
        # Phase: delta = 4*pi*n*d*f/c
        delta = 4 * np.pi * self.n * self.d * frequency / self.c
        return 1.0 / (1 + self.F * np.sin(delta / 2)**2)

    def resonance_frequencies(self, n_modes: int = 10, center_freq: float = None) -> np.ndarray:
        """Calculate resonance frequencies around a center frequency"""
        if center_freq is None:
            center_freq = self.c / (2 * self.d)  # First few modes

        # Mode number at center frequency
        m_center = int(2 * self.n * self.d * center_freq / self.c)

        modes = np.arange(m_center - n_modes // 2, m_center + n_modes // 2 + 1)
        return modes * self.fsr_freq

    def resolving_power(self, wavelength: float) -> float:
        """Calculate resolving power at given wavelength"""
        # Order number m = 2*n*d/lambda
        m = 2 * self.n * self.d / wavelength
        return m * self.finesse

    def photon_lifetime(self) -> float:
        """Calculate photon lifetime in cavity"""
        # tau = 2*n*d / (c * (1-R))
        return 2 * self.n * self.d / (self.c * (1 - self.R))

    def quality_factor(self, frequency: float) -> float:
        """Calculate Q factor"""
        return frequency / self.linewidth


def plot_transmission_spectra():
    """Plot Fabry-Perot transmission spectra"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Effect of reflectance
    ax1 = axes[0, 0]

    d = 10e-3  # 10 mm spacing
    reflectances = [0.5, 0.8, 0.95, 0.99]
    colors = plt.cm.viridis(np.linspace(0, 1, len(reflectances)))

    # Frequency range (one FSR)
    fp_ref = EnhancedFabryPerot(d, 0.9)
    f_center = fp_ref.c / (632.8e-9)  # HeNe frequency
    f_range = np.linspace(f_center - 2*fp_ref.fsr_freq, f_center + 2*fp_ref.fsr_freq, 1000)

    for R, color in zip(reflectances, colors):
        fp = EnhancedFabryPerot(d, R)
        T = fp.transmission_vs_frequency(f_range)

        ax1.plot((f_range - f_center) / fp.fsr_freq, T, color=color, linewidth=2,
                label=f'R = {R:.0%}, F = {fp.finesse:.1f}')

    ax1.set_xlabel('Detuning (FSR units)')
    ax1.set_ylabel('Transmission')
    ax1.set_title('Effect of Mirror Reflectance on Transmission')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-2, 2)

    # Plot 2: Effect of spacing (FSR)
    ax2 = axes[0, 1]

    R = 0.9
    spacings = [1e-3, 5e-3, 10e-3, 50e-3]  # 1mm to 50mm

    wavelength = 632.8e-9
    wl_range = np.linspace(wavelength * 0.9999, wavelength * 1.0001, 1000)

    for d, color in zip(spacings, colors):
        fp = EnhancedFabryPerot(d, R)
        T = np.array([fp.transmission(wl) for wl in wl_range])

        ax2.plot((wl_range - wavelength) * 1e12, T, linewidth=2,
                label=f'd = {d*1000:.0f} mm, FSR = {fp.fsr_freq/1e9:.2f} GHz')

    ax2.set_xlabel('Wavelength detuning (pm)')
    ax2.set_ylabel('Transmission')
    ax2.set_title('Effect of Mirror Spacing')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Airy function shape
    ax3 = axes[1, 0]

    R = 0.95
    fp = EnhancedFabryPerot(10e-3, R)

    # Zoom into one resonance
    delta_range = np.linspace(-0.5, 0.5, 1000)  # Phase around resonance
    T = 1.0 / (1 + fp.F * np.sin(delta_range * np.pi)**2)

    ax3.plot(delta_range, T, 'b-', linewidth=2)

    # Mark FWHM
    half_max_delta = np.arcsin(1 / np.sqrt(fp.F)) / np.pi
    ax3.axhline(0.5, color='red', linestyle='--', alpha=0.5)
    ax3.axvline(half_max_delta, color='red', linestyle='--', alpha=0.5)
    ax3.axvline(-half_max_delta, color='red', linestyle='--', alpha=0.5)

    ax3.annotate('', xy=(half_max_delta, 0.5), xytext=(-half_max_delta, 0.5),
                arrowprops=dict(arrowstyle='<->', color='red'))
    ax3.text(0, 0.55, f'FWHM = 1/F = 1/{fp.finesse:.1f}', ha='center', color='red')

    ax3.set_xlabel('Phase detuning (FSR units)')
    ax3.set_ylabel('Transmission')
    ax3.set_title(f'Airy Function: Single Resonance\nR = {R:.0%}, Finesse = {fp.finesse:.1f}')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Angular dependence
    ax4 = axes[1, 1]

    fp = EnhancedFabryPerot(10e-3, 0.9)
    wavelength = 632.8e-9

    angles = np.linspace(0, 0.01, 100)  # 0 to 10 mrad
    wavelengths = np.linspace(wavelength * 0.9999, wavelength * 1.0001, 200)

    T_map = np.zeros((len(angles), len(wavelengths)))
    for i, angle in enumerate(angles):
        for j, wl in enumerate(wavelengths):
            T_map[i, j] = fp.transmission(wl, angle)

    im = ax4.imshow(T_map, extent=[(wavelengths[0]-wavelength)*1e12,
                                   (wavelengths[-1]-wavelength)*1e12,
                                   angles[0]*1e3, angles[-1]*1e3],
                   aspect='auto', cmap='hot', origin='lower')
    plt.colorbar(im, ax=ax4, label='Transmission')

    ax4.set_xlabel('Wavelength detuning (pm)')
    ax4.set_ylabel('Angle (mrad)')
    ax4.set_title('Angular Tuning of Fabry-Perot')

    plt.tight_layout()
    return fig


def plot_spectroscopic_applications():
    """Plot spectroscopic applications of Fabry-Perot"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Resolving a doublet
    ax1 = axes[0, 0]

    fp = EnhancedFabryPerot(50e-3, 0.98)  # High finesse etalon

    # Sodium D lines: 589.0 and 589.6 nm
    lambda1 = 589.0e-9
    lambda2 = 589.6e-9
    delta_lambda = lambda2 - lambda1

    # Frequency splitting
    c = fp.c
    f1 = c / lambda1
    f2 = c / lambda2
    delta_f = abs(f1 - f2)

    print(f"Sodium D line splitting: {delta_f/1e9:.2f} GHz")
    print(f"Fabry-Perot linewidth: {fp.linewidth/1e9:.4f} GHz")
    print(f"Resolving power: {fp.resolving_power(lambda1):.0f}")

    # Generate spectrum
    f_center = (f1 + f2) / 2
    f_range = np.linspace(f_center - 3*delta_f, f_center + 3*delta_f, 1000)

    # Two lines convolved with FP response
    T1 = fp.transmission_vs_frequency(f_range - f1 + f_center)
    T2 = fp.transmission_vs_frequency(f_range - f2 + f_center)

    ax1.plot((f_range - f_center)/1e9, T1, 'b-', linewidth=2, label='D1 (589.0 nm)')
    ax1.plot((f_range - f_center)/1e9, T2, 'r-', linewidth=2, label='D2 (589.6 nm)')
    ax1.plot((f_range - f_center)/1e9, 0.5*(T1 + T2), 'k--', linewidth=2, label='Combined')

    ax1.set_xlabel('Frequency detuning (GHz)')
    ax1.set_ylabel('Transmission')
    ax1.set_title(f'Resolving Sodium D Doublet\n'
                 f'Finesse = {fp.finesse:.1f}, Linewidth = {fp.linewidth/1e6:.1f} MHz')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Scanning Fabry-Perot
    ax2 = axes[0, 1]

    fp = EnhancedFabryPerot(10e-3, 0.95)

    # Simulate multi-mode laser spectrum
    mode_spacing = 100e6  # 100 MHz mode spacing
    n_modes = 11

    modes = (np.arange(n_modes) - n_modes//2) * mode_spacing
    mode_amplitudes = np.exp(-modes**2 / (3*mode_spacing)**2)  # Gaussian envelope

    # Scan FP
    scan_range = np.linspace(-fp.fsr_freq/2, fp.fsr_freq/2, 1000)
    total_T = np.zeros_like(scan_range)

    for mode, amp in zip(modes, mode_amplitudes):
        T = fp.transmission_vs_frequency(scan_range + mode + fp.c/(632.8e-9))
        total_T += amp * T

    ax2.plot(scan_range/1e6, total_T, 'b-', linewidth=2)

    # Mark expected mode positions
    for mode, amp in zip(modes, mode_amplitudes):
        if amp > 0.1:
            ax2.axvline(mode/1e6, color='red', linestyle='--', alpha=0.3)

    ax2.set_xlabel('Frequency (MHz)')
    ax2.set_ylabel('Transmitted signal (arb. units)')
    ax2.set_title('Scanning Fabry-Perot: Laser Mode Analysis')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Finesse vs reflectance
    ax3 = axes[1, 0]

    R_range = np.linspace(0.5, 0.999, 100)
    finesse_values = np.pi * np.sqrt(R_range) / (1 - R_range)

    ax3.semilogy(R_range * 100, finesse_values, 'b-', linewidth=2)

    # Mark common values
    for R_mark in [0.9, 0.95, 0.99, 0.999]:
        F_mark = np.pi * np.sqrt(R_mark) / (1 - R_mark)
        ax3.plot(R_mark * 100, F_mark, 'ro', markersize=10)
        ax3.annotate(f'F={F_mark:.0f}', xy=(R_mark*100, F_mark),
                    xytext=(5, 0), textcoords='offset points')

    ax3.set_xlabel('Mirror reflectance (%)')
    ax3.set_ylabel('Finesse')
    ax3.set_title('Finesse vs Mirror Reflectance')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(50, 100)

    # Plot 4: Quality factor and photon lifetime
    ax4 = axes[1, 1]

    fp_params = [(10e-3, 0.95), (10e-3, 0.99), (50e-3, 0.99), (100e-3, 0.99)]
    labels = ['d=10mm, R=95%', 'd=10mm, R=99%', 'd=50mm, R=99%', 'd=100mm, R=99%']

    wavelength = 632.8e-9
    frequency = c / wavelength

    bar_positions = np.arange(len(fp_params))
    tau_values = []
    Q_values = []

    for (d, R), label in zip(fp_params, labels):
        fp = EnhancedFabryPerot(d, R)
        tau_values.append(fp.photon_lifetime() * 1e9)  # in ns
        Q_values.append(fp.quality_factor(frequency) / 1e6)  # in millions

    ax4_twin = ax4.twinx()

    bars1 = ax4.bar(bar_positions - 0.2, tau_values, 0.4, label='Photon lifetime',
                   color='blue', alpha=0.7)
    bars2 = ax4_twin.bar(bar_positions + 0.2, Q_values, 0.4, label='Q factor',
                        color='red', alpha=0.7)

    ax4.set_xticks(bar_positions)
    ax4.set_xticklabels(labels, rotation=15, ha='right')
    ax4.set_ylabel('Photon lifetime (ns)', color='blue')
    ax4_twin.set_ylabel('Q factor (millions)', color='red')
    ax4.set_title('Cavity Properties')

    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')

    plt.tight_layout()
    return fig


def plot_etalon_modes():
    """Plot cavity modes and their properties"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Cavity modes visualization
    ax1 = axes[0, 0]

    d = 10e-3  # 10 mm
    n_modes = 5
    wavelength_center = 632.8e-9

    # Mode wavelengths: lambda_m = 2*n*d/m
    m_center = int(2 * d / wavelength_center)

    for dm in range(-n_modes, n_modes + 1):
        m = m_center + dm
        lambda_m = 2 * d / m

        # Draw standing wave
        x = np.linspace(0, d, 200)
        amplitude = np.sin(m * np.pi * x / d)

        y_offset = dm * 0.3
        ax1.plot(x * 1e3, amplitude * 0.1 + y_offset, linewidth=1.5)
        ax1.text(d * 1e3 + 0.5, y_offset, f'm = {m}', fontsize=9, va='center')

    # Draw mirrors
    ax1.axvline(0, color='blue', linewidth=3)
    ax1.axvline(d * 1e3, color='blue', linewidth=3)

    ax1.set_xlabel('Position (mm)')
    ax1.set_ylabel('Mode index offset')
    ax1.set_title('Cavity Standing Wave Modes')
    ax1.set_xlim(-1, d * 1e3 + 2)

    # Plot 2: Mode spectrum
    ax2 = axes[0, 1]

    fp = EnhancedFabryPerot(d, 0.95)
    f_center = fp.c / wavelength_center
    f_range = np.linspace(f_center - 5*fp.fsr_freq, f_center + 5*fp.fsr_freq, 2000)

    T = fp.transmission_vs_frequency(f_range)

    ax2.plot((f_range - f_center)/1e9, T, 'b-', linewidth=2)

    # Mark FSR
    ax2.axvline(fp.fsr_freq/1e9, color='red', linestyle='--')
    ax2.axvline(-fp.fsr_freq/1e9, color='red', linestyle='--')
    ax2.annotate('', xy=(fp.fsr_freq/1e9, 0.5), xytext=(0, 0.5),
                arrowprops=dict(arrowstyle='<->', color='red'))
    ax2.text(fp.fsr_freq/(2e9), 0.55, f'FSR = {fp.fsr_freq/1e9:.2f} GHz',
            ha='center', color='red')

    ax2.set_xlabel('Frequency detuning from center (GHz)')
    ax2.set_ylabel('Transmission')
    ax2.set_title(f'Cavity Transmission Spectrum\nFSR = {fp.fsr_freq/1e9:.2f} GHz')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Intensity buildup in cavity
    ax3 = axes[1, 0]

    R_values = [0.9, 0.95, 0.99]

    for R in R_values:
        fp = EnhancedFabryPerot(d, R)

        # Buildup factor = 1/(1-R) at resonance
        buildup = 1 / (1 - R)

        # Time evolution of intensity buildup
        tau = fp.photon_lifetime()
        t = np.linspace(0, 5*tau, 200)
        I = buildup * (1 - np.exp(-t/tau))

        ax3.plot(t * 1e9, I, linewidth=2, label=f'R={R:.0%}, buildup={buildup:.0f}x')

    ax3.set_xlabel('Time (ns)')
    ax3.set_ylabel('Intracavity intensity (relative)')
    ax3.set_title('Cavity Intensity Buildup')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Cavity ringdown
    ax4 = axes[1, 1]

    for R in R_values:
        fp = EnhancedFabryPerot(d, R)
        tau = fp.photon_lifetime()

        t = np.linspace(0, 10*tau, 200)
        I = np.exp(-t/tau)

        ax4.semilogy(t * 1e9, I, linewidth=2,
                    label=f'R={R:.0%}, tau={tau*1e9:.2f} ns')

    ax4.set_xlabel('Time (ns)')
    ax4.set_ylabel('Intensity (log scale)')
    ax4.set_title('Cavity Ringdown Decay')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(1e-3, 1.5)

    plt.tight_layout()
    return fig


def main():
    """Main function to demonstrate Fabry-Perot cavity"""

    # Create figures
    fig1 = plot_transmission_spectra()
    fig2 = plot_spectroscopic_applications()
    fig3 = plot_etalon_modes()

    # Save plots
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)

    fig1.savefig(os.path.join(output_dir, 'fabry_perot_transmission.png'),
                 dpi=150, bbox_inches='tight')
    fig2.savefig(os.path.join(output_dir, 'fabry_perot_spectroscopy.png'),
                 dpi=150, bbox_inches='tight')
    fig3.savefig(os.path.join(output_dir, 'fabry_perot_modes.png'),
                 dpi=150, bbox_inches='tight')

    print(f"Plots saved to {output_dir}/fabry_perot_*.png")

    # Print analysis
    print("\n=== Fabry-Perot Cavity Analysis ===")

    d = 10e-3
    R = 0.95
    fp = EnhancedFabryPerot(d, R)

    print(f"\nParameters: d = {d*1e3:.1f} mm, R = {R:.0%}")
    print(f"\nDerived quantities:")
    print(f"  Finesse: {fp.finesse:.1f}")
    print(f"  Coefficient of finesse F: {fp.F:.1f}")
    print(f"  Free spectral range: {fp.fsr_freq/1e9:.3f} GHz")
    print(f"  Linewidth (FWHM): {fp.linewidth/1e6:.2f} MHz")
    print(f"  Photon lifetime: {fp.photon_lifetime()*1e9:.3f} ns")
    print(f"  Q factor (at 633nm): {fp.quality_factor(fp.c/632.8e-9)/1e6:.1f} million")
    print(f"  Resolving power (at 633nm): {fp.resolving_power(632.8e-9):.0f}")


if __name__ == "__main__":
    main()
