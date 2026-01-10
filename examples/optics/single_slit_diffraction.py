"""
Example 107: Single Slit Diffraction

This example demonstrates single-slit diffraction patterns, showing both
Fraunhofer (far-field) and Fresnel (near-field) diffraction regimes.

Physics:
    Fraunhofer diffraction (far field):
    I(theta) = I_0 * [sin(beta) / beta]^2
    where beta = (pi * a * sin(theta)) / lambda

    The intensity has:
    - Central maximum of width 2*lambda/a
    - Minima at sin(theta) = m * lambda / a  (m = +/-1, +/-2, ...)
    - Secondary maxima at beta = tan(beta)

    Fresnel number: N_F = a^2 / (lambda * L)
    - N_F << 1: Fraunhofer regime
    - N_F ~ 1 or larger: Fresnel regime
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import fresnel
from src.sciforge.physics.optics import SingleSlitDiffraction


class EnhancedSingleSlit:
    """Enhanced single slit diffraction with both Fraunhofer and Fresnel patterns"""

    def __init__(self, wavelength: float, slit_width: float, screen_distance: float):
        """
        Args:
            wavelength: Light wavelength (m)
            slit_width: Slit width (m)
            screen_distance: Distance to observation screen (m)
        """
        self.wavelength = wavelength
        self.a = slit_width
        self.L = screen_distance
        self.k = 2 * np.pi / wavelength

        # Fresnel number
        self.fresnel_number = slit_width**2 / (wavelength * screen_distance)

    def fraunhofer_intensity(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate Fraunhofer (far-field) diffraction intensity.

        I = I_0 * sinc^2(pi * a * x / (lambda * L))
        """
        # beta = pi * a * sin(theta) / lambda
        # For small angles: sin(theta) ≈ x/L
        beta = np.pi * self.a * x / (self.wavelength * self.L)

        # Handle beta = 0
        intensity = np.ones_like(beta)
        nonzero = np.abs(beta) > 1e-10
        intensity[nonzero] = (np.sin(beta[nonzero]) / beta[nonzero])**2

        return intensity

    def fresnel_intensity(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate Fresnel (near-field) diffraction intensity.

        Uses Fresnel integrals C(u) and S(u).
        """
        # Fresnel parameters
        sqrt_factor = np.sqrt(2 / (self.wavelength * self.L))

        intensity = np.zeros_like(x)

        for i, xi in enumerate(x):
            # Integration limits
            u1 = sqrt_factor * (xi - self.a/2)
            u2 = sqrt_factor * (xi + self.a/2)

            # Fresnel integrals
            S1, C1 = fresnel(u1)
            S2, C2 = fresnel(u2)

            # Intensity
            intensity[i] = 0.5 * ((C2 - C1)**2 + (S2 - S1)**2)

        return intensity

    def intensity(self, x: np.ndarray) -> np.ndarray:
        """Calculate intensity based on Fresnel number"""
        if self.fresnel_number < 0.1:
            return self.fraunhofer_intensity(x)
        else:
            return self.fresnel_intensity(x)


def plot_fraunhofer_diffraction():
    """Plot Fraunhofer (far-field) diffraction patterns"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    wavelength = 632.8e-9  # HeNe laser wavelength
    screen_distance = 2.0  # 2 meters

    # Plot 1: Different slit widths
    ax1 = axes[0, 0]

    slit_widths = [20e-6, 50e-6, 100e-6, 200e-6]  # 20 to 200 microns
    colors = plt.cm.viridis(np.linspace(0, 1, len(slit_widths)))

    x_range = np.linspace(-0.05, 0.05, 1000)

    for a, color in zip(slit_widths, colors):
        slit = EnhancedSingleSlit(wavelength, a, screen_distance)
        intensity = slit.fraunhofer_intensity(x_range)

        ax1.plot(x_range * 1000, intensity, color=color, linewidth=2,
                label=f'a = {a*1e6:.0f} um')

    ax1.set_xlabel('Position x (mm)')
    ax1.set_ylabel('Normalized intensity')
    ax1.set_title(f'Fraunhofer Diffraction: Effect of Slit Width\n'
                 f'lambda = {wavelength*1e9:.1f} nm, L = {screen_distance} m')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-50, 50)

    # Plot 2: Different wavelengths
    ax2 = axes[0, 1]

    slit_width = 50e-6  # 50 microns
    wavelengths = [450e-9, 532e-9, 632.8e-9, 780e-9]  # Blue to near-IR
    colors = ['blue', 'green', 'red', 'darkred']
    labels = ['Blue (450nm)', 'Green (532nm)', 'Red (633nm)', 'NIR (780nm)']

    for wl, color, label in zip(wavelengths, colors, labels):
        slit = EnhancedSingleSlit(wl, slit_width, screen_distance)
        intensity = slit.fraunhofer_intensity(x_range)

        ax2.plot(x_range * 1000, intensity, color=color, linewidth=2, label=label)

    ax2.set_xlabel('Position x (mm)')
    ax2.set_ylabel('Normalized intensity')
    ax2.set_title(f'Fraunhofer Diffraction: Effect of Wavelength\n'
                 f'a = {slit_width*1e6:.0f} um, L = {screen_distance} m')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-50, 50)

    # Plot 3: 2D intensity pattern
    ax3 = axes[1, 0]

    slit = EnhancedSingleSlit(wavelength, 50e-6, screen_distance)

    # Create 2D pattern (slit is in y-direction, diffraction in x)
    x = np.linspace(-0.03, 0.03, 500)
    y = np.linspace(-0.02, 0.02, 200)
    X, Y = np.meshgrid(x, y)

    # Intensity only varies with x (slit is parallel to y)
    I = slit.fraunhofer_intensity(X)

    im = ax3.imshow(I, extent=[x.min()*1000, x.max()*1000, y.min()*1000, y.max()*1000],
                   aspect='auto', cmap='hot', origin='lower')
    plt.colorbar(im, ax=ax3, label='Normalized intensity')

    ax3.set_xlabel('x (mm)')
    ax3.set_ylabel('y (mm)')
    ax3.set_title('2D Diffraction Pattern\n(Slit parallel to y-axis)')

    # Plot 4: Log scale to show secondary maxima
    ax4 = axes[1, 1]

    x_wide = np.linspace(-0.1, 0.1, 2000)

    for a, color in zip(slit_widths, colors):
        slit = EnhancedSingleSlit(wavelength, a, screen_distance)
        intensity = slit.fraunhofer_intensity(x_wide)
        # Avoid log(0)
        intensity = np.maximum(intensity, 1e-10)

        ax4.semilogy(x_wide * 1000, intensity, color=color, linewidth=1.5,
                    label=f'a = {a*1e6:.0f} um')

    ax4.set_xlabel('Position x (mm)')
    ax4.set_ylabel('Normalized intensity (log scale)')
    ax4.set_title('Fraunhofer Diffraction: Log Scale\nShowing secondary maxima')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-100, 100)
    ax4.set_ylim(1e-6, 1.5)

    plt.tight_layout()
    return fig


def plot_fresnel_diffraction():
    """Plot Fresnel (near-field) diffraction patterns"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    wavelength = 632.8e-9
    slit_width = 200e-6  # 200 microns

    # Plot 1: Transition from Fresnel to Fraunhofer
    ax1 = axes[0, 0]

    distances = [0.01, 0.05, 0.1, 0.5, 2.0]  # 1cm to 2m
    colors = plt.cm.plasma(np.linspace(0, 1, len(distances)))

    for L, color in zip(distances, colors):
        slit = EnhancedSingleSlit(wavelength, slit_width, L)
        x_range = np.linspace(-3 * slit_width, 3 * slit_width, 500)

        if slit.fresnel_number > 0.1:
            intensity = slit.fresnel_intensity(x_range)
            style = '-'
        else:
            intensity = slit.fraunhofer_intensity(x_range)
            style = '--'

        ax1.plot(x_range * 1e6, intensity, color=color, linewidth=2, linestyle=style,
                label=f'L={L*100:.0f}cm, Nf={slit.fresnel_number:.2f}')

    # Mark slit edges
    ax1.axvline(-slit_width*1e6/2, color='black', linestyle=':', alpha=0.5)
    ax1.axvline(slit_width*1e6/2, color='black', linestyle=':', alpha=0.5)

    ax1.set_xlabel('Position x (um)')
    ax1.set_ylabel('Normalized intensity')
    ax1.set_title('Fresnel to Fraunhofer Transition\n'
                 f'Solid: Fresnel (Nf>0.1), Dashed: Fraunhofer (Nf<0.1)')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Fresnel number dependence
    ax2 = axes[0, 1]

    fresnel_numbers = [10, 5, 2, 1, 0.5, 0.1]

    for Nf in fresnel_numbers:
        L = slit_width**2 / (wavelength * Nf)
        slit = EnhancedSingleSlit(wavelength, slit_width, L)

        x_max = 3 * slit_width
        x_range = np.linspace(-x_max, x_max, 500)

        if Nf > 0.1:
            intensity = slit.fresnel_intensity(x_range)
        else:
            intensity = slit.fraunhofer_intensity(x_range)

        ax2.plot(x_range / slit_width, intensity, linewidth=2,
                label=f'Nf = {Nf}')

    ax2.axvline(-0.5, color='black', linestyle=':', alpha=0.5)
    ax2.axvline(0.5, color='black', linestyle=':', alpha=0.5)

    ax2.set_xlabel('Position x/a (normalized to slit width)')
    ax2.set_ylabel('Normalized intensity')
    ax2.set_title('Effect of Fresnel Number\nNf = a^2/(lambda*L)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Near-field pattern evolution
    ax3 = axes[1, 0]

    # Create distance vs position map
    n_distances = 100
    n_positions = 200

    distances = np.linspace(0.005, 0.5, n_distances)
    x_range = np.linspace(-2*slit_width, 2*slit_width, n_positions)

    intensity_map = np.zeros((n_distances, n_positions))

    for i, L in enumerate(distances):
        slit = EnhancedSingleSlit(wavelength, slit_width, L)
        if slit.fresnel_number > 0.1:
            intensity_map[i, :] = slit.fresnel_intensity(x_range)
        else:
            intensity_map[i, :] = slit.fraunhofer_intensity(x_range)

    im = ax3.imshow(intensity_map, extent=[x_range.min()*1e6, x_range.max()*1e6,
                                           distances.min()*100, distances.max()*100],
                   aspect='auto', cmap='hot', origin='lower')
    plt.colorbar(im, ax=ax3, label='Intensity')

    # Mark slit edges
    ax3.axvline(-slit_width*1e6/2, color='white', linestyle='--', alpha=0.5)
    ax3.axvline(slit_width*1e6/2, color='white', linestyle='--', alpha=0.5)

    ax3.set_xlabel('Position x (um)')
    ax3.set_ylabel('Distance L (cm)')
    ax3.set_title('Diffraction Pattern Evolution with Distance')

    # Plot 4: Fresnel zone analysis
    ax4 = axes[1, 1]

    # For a given observation point, show Fresnel zones
    L = 0.05  # 5 cm
    slit = EnhancedSingleSlit(wavelength, slit_width, L)

    # Fresnel zones at the slit
    x_slit = np.linspace(-slit_width/2, slit_width/2, 1000)

    # For on-axis point (x_obs = 0)
    r0 = L  # Distance from slit center to observation point

    # Path length difference
    delta_r = np.sqrt(L**2 + x_slit**2) - L

    # Number of half-wavelengths
    zone_number = 2 * delta_r / wavelength

    ax4.plot(x_slit * 1e6, zone_number, 'b-', linewidth=2)
    ax4.axhline(1, color='red', linestyle='--', alpha=0.5, label='First zone boundary')
    ax4.axhline(2, color='orange', linestyle='--', alpha=0.5, label='Second zone boundary')

    ax4.fill_between(x_slit * 1e6, 0, zone_number, where=zone_number <= 1,
                    color='blue', alpha=0.3, label='First Fresnel zone')
    ax4.fill_between(x_slit * 1e6, 0, zone_number, where=(zone_number > 1) & (zone_number <= 2),
                    color='red', alpha=0.3, label='Second Fresnel zone')

    ax4.set_xlabel('Position across slit (um)')
    ax4.set_ylabel('Fresnel zone number')
    ax4.set_title(f'Fresnel Zones for On-Axis Point\n'
                 f'L = {L*100:.0f} cm, Nf = {slit.fresnel_number:.1f}')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_minima_and_secondary_maxima():
    """Plot positions of minima and secondary maxima"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    wavelength = 632.8e-9
    slit_width = 100e-6
    screen_distance = 2.0

    slit = EnhancedSingleSlit(wavelength, slit_width, screen_distance)

    # Plot 1: Pattern with marked features
    ax1 = axes[0]

    x_range = np.linspace(-0.04, 0.04, 2000)
    intensity = slit.fraunhofer_intensity(x_range)

    ax1.plot(x_range * 1000, intensity, 'b-', linewidth=2)

    # Mark minima: sin(theta) = m * lambda/a
    for m in range(1, 4):
        x_min = m * wavelength * screen_distance / slit_width
        ax1.axvline(x_min * 1000, color='red', linestyle='--', alpha=0.7)
        ax1.axvline(-x_min * 1000, color='red', linestyle='--', alpha=0.7)
        ax1.text(x_min * 1000, 0.5, f'm={m}', ha='center', color='red')

    # Mark secondary maxima (approximately at beta = (m + 0.5) * pi)
    for m in range(1, 3):
        # Secondary maximum at approximately m + 0.5
        x_max = (m + 0.5) * wavelength * screen_distance / slit_width
        I_max = slit.fraunhofer_intensity(np.array([x_max]))[0]
        ax1.plot(x_max * 1000, I_max, 'go', markersize=8)
        ax1.plot(-x_max * 1000, I_max, 'go', markersize=8)

    ax1.set_xlabel('Position x (mm)')
    ax1.set_ylabel('Normalized intensity')
    ax1.set_title('Single Slit Diffraction: Minima and Secondary Maxima\n'
                 f'a = {slit_width*1e6:.0f} um, lambda = {wavelength*1e9:.1f} nm')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-40, 40)

    # Add annotations
    ax1.annotate('Central\nmaximum', xy=(0, 1), xytext=(5, 0.9),
                fontsize=10, ha='left')

    # Plot 2: Relative intensities of maxima
    ax2 = axes[1]

    # Calculate relative intensities
    orders = np.arange(0, 6)
    central_width = 2 * wavelength * screen_distance / slit_width

    relative_intensities = []
    positions = []

    for m in orders:
        if m == 0:
            I = 1.0
            x = 0
        else:
            # Approximate position of mth secondary maximum
            beta_m = (m + 0.5) * np.pi
            x = beta_m * wavelength * screen_distance / (np.pi * slit_width)
            I = (np.sin(beta_m) / beta_m)**2

        relative_intensities.append(I)
        positions.append(x)

    # Theoretical formula: I_m / I_0 = 1 / ((m + 0.5) * pi)^2
    theoretical = [1.0] + [1.0 / ((m + 0.5) * np.pi)**2 for m in range(1, 6)]

    ax2.bar(orders - 0.2, relative_intensities, 0.4, label='Numerical', color='blue', alpha=0.7)
    ax2.bar(orders + 0.2, theoretical, 0.4, label='Theoretical', color='red', alpha=0.7)

    # Add percentage labels
    for i, (val, m) in enumerate(zip(relative_intensities, orders)):
        ax2.text(m, val + 0.02, f'{val*100:.1f}%', ha='center', fontsize=9)

    ax2.set_xlabel('Maximum order m')
    ax2.set_ylabel('Relative intensity I_m / I_0')
    ax2.set_title('Relative Intensities of Diffraction Maxima\n'
                 r'$I_m/I_0 = 1/[(m+0.5)\pi]^2$ for secondary maxima')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(orders)

    plt.tight_layout()
    return fig


def main():
    """Main function to demonstrate single slit diffraction"""

    # Create figures
    fig1 = plot_fraunhofer_diffraction()
    fig2 = plot_fresnel_diffraction()
    fig3 = plot_minima_and_secondary_maxima()

    # Save plots
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)

    fig1.savefig(os.path.join(output_dir, 'single_slit_fraunhofer.png'),
                 dpi=150, bbox_inches='tight')
    fig2.savefig(os.path.join(output_dir, 'single_slit_fresnel.png'),
                 dpi=150, bbox_inches='tight')
    fig3.savefig(os.path.join(output_dir, 'single_slit_analysis.png'),
                 dpi=150, bbox_inches='tight')

    print(f"Plots saved to {output_dir}/single_slit_*.png")

    # Print analysis
    print("\n=== Single Slit Diffraction Analysis ===")

    wavelength = 632.8e-9
    slit_width = 100e-6
    screen_distance = 2.0

    print(f"\nParameters: a = {slit_width*1e6:.0f} um, lambda = {wavelength*1e9:.1f} nm, L = {screen_distance} m")

    # Central maximum width
    central_width = 2 * wavelength * screen_distance / slit_width
    print(f"\nCentral maximum width: {central_width*1000:.2f} mm")

    # Minima positions
    print("\nMinima positions:")
    for m in range(1, 4):
        x_min = m * wavelength * screen_distance / slit_width
        print(f"  m={m}: x = {x_min*1000:.2f} mm")

    # Secondary maxima relative intensities
    print("\nSecondary maxima relative intensities:")
    for m in range(1, 4):
        I_rel = 1.0 / ((m + 0.5) * np.pi)**2
        print(f"  m={m}: I/I_0 = {I_rel*100:.2f}%")


if __name__ == "__main__":
    main()
