"""
Example 107: Double Slit Interference - Young's Experiment

This example demonstrates Young's double slit experiment, the classic demonstration
of wave interference in optics.

Physics:
    For two slits separated by distance d, the intensity pattern is:
    I = 4 * I_0 * cos^2(phi/2) * sinc^2(beta)

    where:
    - phi = (2*pi*d*sin(theta)) / lambda  (phase difference between slits)
    - beta = (pi*a*sin(theta)) / lambda   (single slit diffraction parameter)
    - a is the slit width

    Interference fringes:
    - Constructive (bright): d*sin(theta) = m*lambda  (m = 0, 1, 2, ...)
    - Destructive (dark): d*sin(theta) = (m + 1/2)*lambda

    Fringe spacing: delta_y = lambda * L / d

    The double slit pattern is the product of:
    1. Single slit diffraction envelope (sinc^2)
    2. Two-slit interference pattern (cos^2)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


class DoubleSlitInterference:
    """Young's double slit interference calculator"""

    def __init__(
        self,
        wavelength: float,
        slit_separation: float,
        slit_width: float,
        screen_distance: float
    ):
        """
        Args:
            wavelength: Light wavelength (m)
            slit_separation: Distance between slit centers (m)
            slit_width: Width of each slit (m)
            screen_distance: Distance to observation screen (m)
        """
        self.wavelength = wavelength
        self.d = slit_separation
        self.a = slit_width
        self.L = screen_distance
        self.k = 2 * np.pi / wavelength

    def interference_intensity(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate the double slit interference pattern.

        Args:
            x: Position on screen (m)

        Returns:
            Normalized intensity pattern
        """
        # Small angle approximation: sin(theta) ~ x/L
        sin_theta = x / self.L

        # Two-slit interference: cos^2(phi/2)
        # phi = k * d * sin(theta)
        phi = self.k * self.d * sin_theta
        interference = np.cos(phi / 2)**2

        # Single slit diffraction envelope: sinc^2(beta)
        # beta = pi * a * sin(theta) / lambda
        beta = np.pi * self.a * sin_theta / self.wavelength

        # Handle beta = 0 (central maximum)
        envelope = np.ones_like(beta)
        nonzero = np.abs(beta) > 1e-10
        envelope[nonzero] = (np.sin(beta[nonzero]) / beta[nonzero])**2

        # Combined intensity: I = 4 * I_0 * cos^2(phi/2) * sinc^2(beta)
        # Normalized so central maximum = 1
        return 4 * interference * envelope

    def single_slit_envelope(self, x: np.ndarray) -> np.ndarray:
        """Calculate single slit diffraction envelope only"""
        sin_theta = x / self.L
        beta = np.pi * self.a * sin_theta / self.wavelength

        envelope = np.ones_like(beta)
        nonzero = np.abs(beta) > 1e-10
        envelope[nonzero] = (np.sin(beta[nonzero]) / beta[nonzero])**2

        return 4 * envelope  # Factor of 4 to match double slit at center

    def interference_only(self, x: np.ndarray) -> np.ndarray:
        """Calculate pure two-slit interference (no diffraction envelope)"""
        sin_theta = x / self.L
        phi = self.k * self.d * sin_theta
        return 4 * np.cos(phi / 2)**2

    def fringe_spacing(self) -> float:
        """Calculate fringe spacing: delta_y = lambda * L / d"""
        return self.wavelength * self.L / self.d

    def maxima_positions(self, n_maxima: int = 10) -> np.ndarray:
        """Calculate positions of interference maxima"""
        # d * sin(theta) = m * lambda
        # For small angles: d * x/L = m * lambda
        # x = m * lambda * L / d
        m = np.arange(-n_maxima, n_maxima + 1)
        return m * self.wavelength * self.L / self.d

    def minima_positions(self, n_minima: int = 10) -> np.ndarray:
        """Calculate positions of interference minima"""
        # d * sin(theta) = (m + 1/2) * lambda
        m = np.arange(-n_minima, n_minima + 1)
        return (m + 0.5) * self.wavelength * self.L / self.d


def plot_basic_interference():
    """Plot basic double slit interference pattern"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Parameters
    wavelength = 632.8e-9  # HeNe laser
    slit_separation = 100e-6  # 100 microns
    slit_width = 20e-6  # 20 microns
    screen_distance = 1.0  # 1 meter

    ds = DoubleSlitInterference(wavelength, slit_separation, slit_width, screen_distance)

    x_range = np.linspace(-0.02, 0.02, 2000)

    # Plot 1: Full pattern with envelope
    ax1 = axes[0, 0]

    intensity = ds.interference_intensity(x_range)
    envelope = ds.single_slit_envelope(x_range)

    ax1.plot(x_range * 1000, intensity, 'b-', linewidth=1, label='Double slit')
    ax1.plot(x_range * 1000, envelope, 'r--', linewidth=2, label='Single slit envelope')
    ax1.plot(x_range * 1000, -envelope, 'r--', linewidth=2)  # Lower envelope

    ax1.set_xlabel('Position x (mm)')
    ax1.set_ylabel('Intensity (I / I_0)')
    ax1.set_title(f"Young's Double Slit Interference\n"
                  f"d = {slit_separation*1e6:.0f} um, a = {slit_width*1e6:.0f} um, "
                  f"lambda = {wavelength*1e9:.1f} nm")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-20, 20)
    ax1.set_ylim(-0.5, 4.5)

    # Plot 2: Decomposition of pattern
    ax2 = axes[0, 1]

    interference_only = ds.interference_only(x_range)
    envelope_normalized = ds.single_slit_envelope(x_range) / 4

    ax2.plot(x_range * 1000, interference_only / 4, 'g-', linewidth=1.5,
             alpha=0.7, label='Interference cos^2(phi/2)')
    ax2.plot(x_range * 1000, envelope_normalized, 'r-', linewidth=1.5,
             alpha=0.7, label='Diffraction sinc^2(beta)')
    ax2.plot(x_range * 1000, intensity / 4, 'b-', linewidth=2,
             label='Combined pattern')

    ax2.set_xlabel('Position x (mm)')
    ax2.set_ylabel('Normalized intensity')
    ax2.set_title('Decomposition of Double Slit Pattern\n'
                  'I = 4 * cos^2(phi/2) * sinc^2(beta)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-20, 20)

    # Plot 3: 2D intensity pattern
    ax3 = axes[1, 0]

    x = np.linspace(-0.015, 0.015, 500)
    y = np.linspace(-0.01, 0.01, 200)
    X, Y = np.meshgrid(x, y)

    # Intensity varies only with x (slits parallel to y)
    I_2d = ds.interference_intensity(X)

    im = ax3.imshow(I_2d, extent=[x.min()*1000, x.max()*1000,
                                   y.min()*1000, y.max()*1000],
                    aspect='auto', cmap='hot', origin='lower', vmax=4)
    plt.colorbar(im, ax=ax3, label='Intensity (I / I_0)')

    ax3.set_xlabel('x (mm)')
    ax3.set_ylabel('y (mm)')
    ax3.set_title('2D Interference Pattern\n(Slits parallel to y-axis)')

    # Plot 4: Log scale showing visibility
    ax4 = axes[1, 1]

    intensity_log = np.maximum(intensity, 1e-6)
    envelope_log = np.maximum(envelope, 1e-6)

    ax4.semilogy(x_range * 1000, intensity_log, 'b-', linewidth=1, label='Double slit')
    ax4.semilogy(x_range * 1000, envelope_log, 'r--', linewidth=2, label='Single slit envelope')

    # Mark missing orders (where envelope = 0)
    for m in range(1, 6):
        x_missing = m * wavelength * screen_distance / slit_width
        if x_missing < x_range.max():
            ax4.axvline(x_missing * 1000, color='orange', linestyle=':', alpha=0.7)
            ax4.axvline(-x_missing * 1000, color='orange', linestyle=':', alpha=0.7)
            if m == 1:
                ax4.text(x_missing * 1000, 2, 'Missing\norders', ha='center', fontsize=9)

    ax4.set_xlabel('Position x (mm)')
    ax4.set_ylabel('Intensity (log scale)')
    ax4.set_title('Log Scale: Missing Orders\n'
                  f'Missing at x = m * lambda * L / a = m * {wavelength*screen_distance/slit_width*1000:.2f} mm')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-20, 20)
    ax4.set_ylim(1e-4, 10)

    plt.tight_layout()
    return fig


def plot_parameter_effects():
    """Plot effect of varying slit separation and wavelength"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    wavelength = 632.8e-9
    slit_width = 20e-6
    screen_distance = 1.0
    x_range = np.linspace(-0.015, 0.015, 2000)

    # Plot 1: Effect of slit separation
    ax1 = axes[0, 0]

    separations = [50e-6, 100e-6, 200e-6, 400e-6]
    colors = plt.cm.viridis(np.linspace(0, 1, len(separations)))

    for d, color in zip(separations, colors):
        ds = DoubleSlitInterference(wavelength, d, slit_width, screen_distance)
        intensity = ds.interference_intensity(x_range)
        fringe = ds.fringe_spacing()

        ax1.plot(x_range * 1000, intensity + (separations.index(d) * 5),
                 color=color, linewidth=1,
                 label=f'd = {d*1e6:.0f} um, spacing = {fringe*1000:.2f} mm')

    ax1.set_xlabel('Position x (mm)')
    ax1.set_ylabel('Intensity (offset for clarity)')
    ax1.set_title('Effect of Slit Separation\n'
                  'Larger d -> More fringes (smaller spacing)')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-15, 15)

    # Plot 2: Effect of wavelength
    ax2 = axes[0, 1]

    slit_separation = 100e-6
    wavelengths = [450e-9, 532e-9, 632.8e-9, 780e-9]
    colors = ['blue', 'green', 'red', 'darkred']
    labels = ['Blue (450nm)', 'Green (532nm)', 'Red (633nm)', 'NIR (780nm)']

    for wl, color, label in zip(wavelengths, colors, labels):
        ds = DoubleSlitInterference(wl, slit_separation, slit_width, screen_distance)
        intensity = ds.interference_intensity(x_range)

        ax2.plot(x_range * 1000, intensity, color=color, linewidth=1.5,
                 alpha=0.7, label=label)

    ax2.set_xlabel('Position x (mm)')
    ax2.set_ylabel('Intensity (I / I_0)')
    ax2.set_title('Effect of Wavelength\n'
                  'Longer wavelength -> Wider fringe spacing')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-15, 15)

    # Plot 3: Fringe spacing vs slit separation
    ax3 = axes[1, 0]

    d_range = np.linspace(20e-6, 500e-6, 100)

    for wl, color, label in zip(wavelengths, colors, labels):
        fringe_spacing = wl * screen_distance / d_range
        ax3.plot(d_range * 1e6, fringe_spacing * 1000, color=color,
                 linewidth=2, label=label)

    ax3.set_xlabel('Slit separation d (um)')
    ax3.set_ylabel('Fringe spacing (mm)')
    ax3.set_title('Fringe Spacing: delta_y = lambda * L / d')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(20, 500)

    # Plot 4: Number of fringes within central diffraction maximum
    ax4 = axes[1, 1]

    # Central diffraction maximum width: 2 * lambda * L / a
    # Number of interference fringes: width / fringe_spacing = 2*d/a

    slit_widths = [10e-6, 20e-6, 50e-6]

    for a in slit_widths:
        n_fringes = 2 * d_range / a
        ax4.plot(d_range * 1e6, n_fringes, linewidth=2,
                 label=f'a = {a*1e6:.0f} um')

    ax4.set_xlabel('Slit separation d (um)')
    ax4.set_ylabel('Number of fringes in central maximum')
    ax4.set_title('Fringes in Central Maximum = 2d/a\n'
                  'More fringes with wider separation or narrower slits')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(20, 500)

    plt.tight_layout()
    return fig


def plot_comparison_with_single_slit():
    """Compare double slit with single slit diffraction"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    wavelength = 632.8e-9
    slit_width = 40e-6
    slit_separation = 200e-6
    screen_distance = 1.0

    x_range = np.linspace(-0.02, 0.02, 2000)

    ds = DoubleSlitInterference(wavelength, slit_separation, slit_width, screen_distance)

    # Single slit with same width
    beta = np.pi * slit_width * (x_range / screen_distance) / wavelength
    single_slit = np.ones_like(beta)
    nonzero = np.abs(beta) > 1e-10
    single_slit[nonzero] = (np.sin(beta[nonzero]) / beta[nonzero])**2

    # Single slit with width = 2a (total width of both slits)
    beta_wide = np.pi * 2 * slit_width * (x_range / screen_distance) / wavelength
    single_wide = np.ones_like(beta_wide)
    nonzero_wide = np.abs(beta_wide) > 1e-10
    single_wide[nonzero_wide] = (np.sin(beta_wide[nonzero_wide]) / beta_wide[nonzero_wide])**2

    # Plot 1: Direct comparison
    ax1 = axes[0, 0]

    double_intensity = ds.interference_intensity(x_range)

    ax1.plot(x_range * 1000, double_intensity, 'b-', linewidth=1.5,
             label='Double slit')
    ax1.plot(x_range * 1000, single_slit, 'r-', linewidth=2, alpha=0.7,
             label=f'Single slit (a = {slit_width*1e6:.0f} um)')
    ax1.plot(x_range * 1000, single_wide * 4, 'g--', linewidth=2, alpha=0.7,
             label=f'Single slit (a = {2*slit_width*1e6:.0f} um) x4')

    ax1.set_xlabel('Position x (mm)')
    ax1.set_ylabel('Intensity (I / I_0)')
    ax1.set_title('Double Slit vs Single Slit\n'
                  f'd = {slit_separation*1e6:.0f} um, a = {slit_width*1e6:.0f} um')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-20, 20)

    # Plot 2: Energy redistribution
    ax2 = axes[0, 1]

    # Compare total power in patterns
    dx = x_range[1] - x_range[0]

    power_double = np.sum(double_intensity) * dx
    power_single = np.sum(single_slit) * dx

    ax2.bar(['Single slit\n(width a)', 'Double slit\n(width a each)'],
            [power_single / power_single, power_double / power_single],
            color=['red', 'blue'], alpha=0.7)

    ax2.set_ylabel('Relative total power (normalized)')
    ax2.set_title('Total Power Comparison\n'
                  'Double slit has ~4x peak but ~2x total power')
    ax2.grid(True, alpha=0.3, axis='y')

    # Annotate
    ax2.text(0, 0.5, f'P = {1:.2f}', ha='center', fontsize=12)
    ax2.text(1, power_double/power_single + 0.1,
             f'P = {power_double/power_single:.2f}', ha='center', fontsize=12)

    # Plot 3: Visibility analysis
    ax3 = axes[1, 0]

    # Fringe visibility: V = (I_max - I_min) / (I_max + I_min)
    # For perfect double slit: V = 1
    # With unequal slits or partial coherence: V < 1

    # Simulate partial coherence by varying amplitude ratio
    amplitude_ratios = [1.0, 0.8, 0.6, 0.4, 0.2]

    for ratio in amplitude_ratios:
        # I = |E1 + E2|^2 with E2 = ratio * E1
        # I = I1 + I2 + 2*sqrt(I1*I2)*cos(phi)
        # I_max = (1 + ratio)^2, I_min = (1 - ratio)^2
        # V = 2*ratio / (1 + ratio^2)

        sin_theta = x_range / screen_distance
        phi = ds.k * ds.d * sin_theta

        I = 1 + ratio**2 + 2 * ratio * np.cos(phi)
        I *= single_slit  # Apply envelope

        visibility = 2 * ratio / (1 + ratio**2)

        ax3.plot(x_range * 1000, I, linewidth=1.5,
                 label=f'A2/A1 = {ratio:.1f}, V = {visibility:.2f}')

    ax3.set_xlabel('Position x (mm)')
    ax3.set_ylabel('Intensity')
    ax3.set_title('Effect of Amplitude Imbalance on Visibility\n'
                  r'V = 2r/(1+r^2) where r = A2/A1')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-10, 10)

    # Plot 4: Multiple slits progression
    ax4 = axes[1, 1]

    # N-slit interference: I = I_0 * sin^2(N*phi/2) / sin^2(phi/2)
    slit_nums = [2, 3, 5, 10]
    d = 100e-6

    sin_theta = x_range / screen_distance
    phi = 2 * np.pi * d * sin_theta / wavelength

    for N in slit_nums:
        # N-slit formula
        numerator = np.sin(N * phi / 2)**2
        denominator = np.sin(phi / 2)**2

        # Handle phi = 0, 2*pi, ...
        intensity_n = np.ones_like(phi) * N**2
        valid = np.abs(denominator) > 1e-10
        intensity_n[valid] = numerator[valid] / denominator[valid]

        # Apply single slit envelope
        intensity_n *= single_slit

        ax4.plot(x_range * 1000, intensity_n / N**2, linewidth=1.5,
                 label=f'N = {N} slits')

    ax4.set_xlabel('Position x (mm)')
    ax4.set_ylabel('Normalized intensity')
    ax4.set_title('Progression: 2 -> N Slits\n'
                  'More slits -> Sharper principal maxima')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-10, 10)

    plt.tight_layout()
    return fig


def plot_intensity_formula():
    """Detailed analysis of the interference intensity formula"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    wavelength = 632.8e-9
    slit_separation = 100e-6
    slit_width = 25e-6
    screen_distance = 1.0

    ds = DoubleSlitInterference(wavelength, slit_separation, slit_width, screen_distance)

    x_range = np.linspace(-0.015, 0.015, 2000)

    # Plot 1: I = 4*I0*cos^2(phi/2) formula visualization
    ax1 = axes[0, 0]

    sin_theta = x_range / screen_distance
    phi = ds.k * slit_separation * sin_theta

    cos2 = np.cos(phi / 2)**2

    ax1.plot(phi / np.pi, 4 * cos2, 'b-', linewidth=2)
    ax1.axhline(4, color='red', linestyle='--', alpha=0.5, label='Max = 4I_0')
    ax1.axhline(0, color='red', linestyle='--', alpha=0.5, label='Min = 0')
    ax1.axhline(2, color='orange', linestyle=':', alpha=0.5, label='Mean = 2I_0')

    # Mark constructive and destructive
    for m in range(-5, 6):
        ax1.axvline(2*m, color='green', linestyle=':', alpha=0.3)
        ax1.axvline(2*m + 1, color='purple', linestyle=':', alpha=0.3)

    ax1.set_xlabel('Phase difference phi/pi')
    ax1.set_ylabel('Intensity I / I_0')
    ax1.set_title(r'Two-Slit Interference: I = 4I_0 cos^2(phi/2)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-10, 10)

    # Annotate
    ax1.text(0, 4.3, 'Constructive\nm = 0', ha='center', fontsize=9, color='green')
    ax1.text(1, -0.5, 'Destructive', ha='center', fontsize=9, color='purple')

    # Plot 2: Path difference visualization
    ax2 = axes[0, 1]

    path_diff = slit_separation * sin_theta
    path_diff_wavelengths = path_diff / wavelength

    ax2.plot(x_range * 1000, path_diff_wavelengths, 'b-', linewidth=2)

    # Mark integer and half-integer wavelengths
    for m in range(-3, 4):
        y_pos = m
        if abs(y_pos) < path_diff_wavelengths.max():
            ax2.axhline(y_pos, color='green', linestyle='--', alpha=0.5)
            if m >= 0:
                ax2.text(x_range.max()*1000*0.8, y_pos + 0.1, f'm={m}', fontsize=9)

        y_pos = m + 0.5
        if abs(y_pos) < path_diff_wavelengths.max():
            ax2.axhline(y_pos, color='purple', linestyle=':', alpha=0.5)

    ax2.set_xlabel('Position x (mm)')
    ax2.set_ylabel('Path difference (wavelengths)')
    ax2.set_title('Path Difference = d * sin(theta)\n'
                  'Green: constructive (m*lambda), Purple: destructive ((m+1/2)*lambda)')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Maxima and minima positions
    ax3 = axes[1, 0]

    intensity = ds.interference_intensity(x_range)
    ax3.plot(x_range * 1000, intensity, 'b-', linewidth=1.5)

    # Mark maxima
    maxima = ds.maxima_positions(5)
    for m, x_max in enumerate(maxima):
        if abs(x_max) < x_range.max():
            I_max = ds.interference_intensity(np.array([x_max]))[0]
            ax3.plot(x_max * 1000, I_max, 'go', markersize=8)
            if x_max >= 0:
                ax3.text(x_max * 1000, I_max + 0.3, f'm={m-5}', ha='center', fontsize=8)

    # Mark minima
    minima = ds.minima_positions(5)
    for x_min in minima:
        if abs(x_min) < x_range.max():
            ax3.axvline(x_min * 1000, color='purple', linestyle=':', alpha=0.3)

    ax3.set_xlabel('Position x (mm)')
    ax3.set_ylabel('Intensity (I / I_0)')
    ax3.set_title('Interference Maxima and Minima\n'
                  f'Maxima at x = m * lambda * L / d = m * {wavelength*screen_distance/slit_separation*1000:.2f} mm')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-15, 15)

    # Plot 4: Central fringe analysis
    ax4 = axes[1, 1]

    # Zoom in on central region
    x_central = np.linspace(-0.003, 0.003, 1000)
    intensity_central = ds.interference_intensity(x_central)

    ax4.plot(x_central * 1000, intensity_central, 'b-', linewidth=2)
    ax4.fill_between(x_central * 1000, 0, intensity_central, alpha=0.3)

    # Mark fringe spacing
    fringe = ds.fringe_spacing()
    ax4.annotate('', xy=(0, 2), xytext=(fringe*1000, 2),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax4.text(fringe*1000/2, 2.3, f'Fringe spacing\n= {fringe*1000:.3f} mm',
             ha='center', fontsize=10)

    ax4.set_xlabel('Position x (mm)')
    ax4.set_ylabel('Intensity (I / I_0)')
    ax4.set_title('Central Fringes\n'
                  f'Fringe spacing = lambda*L/d = {fringe*1e6:.1f} um')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    """Main function to demonstrate double slit interference"""

    # Create figures
    fig1 = plot_basic_interference()
    fig2 = plot_parameter_effects()
    fig3 = plot_comparison_with_single_slit()
    fig4 = plot_intensity_formula()

    # Save plots
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)

    fig1.savefig(os.path.join(output_dir, 'double_slit_interference.png'),
                 dpi=150, bbox_inches='tight')
    fig2.savefig(os.path.join(output_dir, 'double_slit_parameters.png'),
                 dpi=150, bbox_inches='tight')
    fig3.savefig(os.path.join(output_dir, 'double_slit_comparison.png'),
                 dpi=150, bbox_inches='tight')
    fig4.savefig(os.path.join(output_dir, 'double_slit_formula.png'),
                 dpi=150, bbox_inches='tight')

    print(f"Plots saved to {output_dir}/double_slit_*.png")

    # Print analysis
    print("\n=== Young's Double Slit Interference Analysis ===")

    wavelength = 632.8e-9
    slit_separation = 100e-6
    slit_width = 20e-6
    screen_distance = 1.0

    ds = DoubleSlitInterference(wavelength, slit_separation, slit_width, screen_distance)

    print(f"\nParameters:")
    print(f"  Wavelength: {wavelength*1e9:.1f} nm")
    print(f"  Slit separation: {slit_separation*1e6:.0f} um")
    print(f"  Slit width: {slit_width*1e6:.0f} um")
    print(f"  Screen distance: {screen_distance*100:.0f} cm")

    print(f"\nResults:")
    print(f"  Fringe spacing: {ds.fringe_spacing()*1000:.3f} mm")
    print(f"  Fringes in central maximum: {2*slit_separation/slit_width:.0f}")

    print("\nInterference maxima (constructive):")
    print("  d * sin(theta) = m * lambda")
    for m in range(4):
        x_max = m * wavelength * screen_distance / slit_separation
        print(f"  m={m}: x = {x_max*1000:.3f} mm")

    print("\nInterference minima (destructive):")
    print("  d * sin(theta) = (m + 1/2) * lambda")
    for m in range(3):
        x_min = (m + 0.5) * wavelength * screen_distance / slit_separation
        print(f"  m={m}: x = {x_min*1000:.3f} mm")

    print("\nMissing orders (diffraction minima):")
    print("  a * sin(theta) = n * lambda")
    for n in range(1, 4):
        x_missing = n * wavelength * screen_distance / slit_width
        interference_order = n * slit_separation / slit_width
        print(f"  n={n}: x = {x_missing*1000:.3f} mm "
              f"(would be interference order {interference_order:.0f})")


if __name__ == "__main__":
    main()
