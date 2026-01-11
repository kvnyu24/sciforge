"""
Experiment 108: Circular Aperture Diffraction (Airy Disk)

Demonstrates the diffraction pattern from a circular aperture,
producing the characteristic Airy disk pattern.

Physical concepts:
- Fraunhofer diffraction from circular aperture
- Intensity: I(theta) = I_0 * [2*J_1(ka*sin(theta)) / (ka*sin(theta))]^2
- First dark ring at sin(theta) = 1.22 * lambda / D
- Rayleigh criterion for resolution

Applications: telescope resolution, microscopy, optical systems
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1  # Bessel function of first kind, order 1


def airy_pattern_1d(theta, wavelength, diameter):
    """
    Calculate Airy diffraction pattern intensity.

    I(theta) = I_0 * [2*J_1(x) / x]^2
    where x = pi * D * sin(theta) / lambda

    Args:
        theta: Angle from optical axis (radians)
        wavelength: Light wavelength
        diameter: Aperture diameter

    Returns:
        Normalized intensity
    """
    x = np.pi * diameter * np.sin(theta) / wavelength

    # Handle x=0 case (central maximum)
    intensity = np.ones_like(x, dtype=float)
    nonzero = np.abs(x) > 1e-10
    intensity[nonzero] = (2 * j1(x[nonzero]) / x[nonzero])**2

    return intensity


def airy_pattern_2d(x, y, wavelength, diameter, focal_length):
    """
    Calculate 2D Airy pattern in focal plane.

    Args:
        x, y: Coordinates in focal plane
        wavelength: Light wavelength
        diameter: Aperture diameter
        focal_length: Lens focal length

    Returns:
        2D intensity array
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(r, focal_length)
    return airy_pattern_1d(theta, wavelength, diameter)


def rayleigh_criterion(wavelength, diameter):
    """
    Rayleigh criterion: minimum resolvable angle.

    theta_min = 1.22 * lambda / D
    """
    return 1.22 * wavelength / diameter


def airy_disk_radius(wavelength, diameter, focal_length):
    """
    Radius of first dark ring in focal plane.

    r = 1.22 * lambda * f / D
    """
    return 1.22 * wavelength * focal_length / diameter


def airy_zeros():
    """Return positions of first few zeros of the Airy pattern (in units of ka*sin(theta))."""
    # Zeros of J_1(x)/x occur at zeros of J_1(x)
    # First few zeros of J_1: 3.8317, 7.0156, 10.1735, 13.3237, ...
    return np.array([3.8317, 7.0156, 10.1735, 13.3237, 16.4706])


def main():
    """Run Airy disk experiments."""
    fig = plt.figure(figsize=(16, 14))

    # Parameters
    wavelength = 550e-9  # Green light (m)
    diameter = 0.01  # 1 cm aperture
    focal_length = 0.5  # 50 cm focal length

    # Plot 1: 1D Airy pattern
    ax1 = fig.add_subplot(2, 2, 1)

    # Angular range
    theta_max = 5 * rayleigh_criterion(wavelength, diameter)
    theta = np.linspace(-theta_max, theta_max, 1000)

    intensity = airy_pattern_1d(theta, wavelength, diameter)

    # Convert to arcseconds for display
    theta_arcsec = np.degrees(theta) * 3600

    ax1.plot(theta_arcsec, intensity, 'b-', lw=2)
    ax1.fill_between(theta_arcsec, intensity, alpha=0.3)

    # Mark first few zeros
    zeros = airy_zeros()
    for i, z in enumerate(zeros[:3]):
        theta_zero = np.arcsin(z * wavelength / (np.pi * diameter))
        theta_zero_arcsec = np.degrees(theta_zero) * 3600
        ax1.axvline(x=theta_zero_arcsec, color='r', linestyle='--', alpha=0.5)
        ax1.axvline(x=-theta_zero_arcsec, color='r', linestyle='--', alpha=0.5)
        if i == 0:
            ax1.annotate(f'1st min\n{theta_zero_arcsec:.2f}"',
                        xy=(theta_zero_arcsec, 0.02),
                        fontsize=9, ha='center')

    # Rayleigh criterion
    theta_R = rayleigh_criterion(wavelength, diameter)
    theta_R_arcsec = np.degrees(theta_R) * 3600
    ax1.axvline(x=theta_R_arcsec, color='g', linestyle=':', lw=2,
                label=f'Rayleigh: {theta_R_arcsec:.2f}"')

    ax1.set_xlabel('Angle (arcseconds)')
    ax1.set_ylabel('Relative Intensity')
    ax1.set_title(f'Airy Pattern ($\\lambda$ = {wavelength*1e9:.0f} nm, D = {diameter*100:.1f} cm)')
    ax1.set_ylim(0, 1.05)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: 2D Airy disk
    ax2 = fig.add_subplot(2, 2, 2)

    # Focal plane coordinates
    r_airy = airy_disk_radius(wavelength, diameter, focal_length)
    extent = 4 * r_airy

    x = np.linspace(-extent, extent, 500)
    y = np.linspace(-extent, extent, 500)
    X, Y = np.meshgrid(x, y)

    I_2d = airy_pattern_2d(X, Y, wavelength, diameter, focal_length)

    # Use log scale to see rings
    I_display = np.log10(I_2d + 1e-6)

    im = ax2.imshow(I_display, extent=[-extent*1e6, extent*1e6, -extent*1e6, extent*1e6],
                    cmap='hot', origin='lower')

    # Draw circles at minima
    for i, z in enumerate(zeros[:3]):
        r_min = z * wavelength * focal_length / (np.pi * diameter)
        circle = plt.Circle((0, 0), r_min*1e6, fill=False, color='cyan',
                            linestyle='--', linewidth=1.5)
        ax2.add_patch(circle)

    ax2.set_xlabel('x (μm)')
    ax2.set_ylabel('y (μm)')
    ax2.set_title('2D Airy Disk (log intensity)')
    plt.colorbar(im, ax=ax2, label='log₁₀(I)')

    # Plot 3: Resolution comparison for different apertures
    ax3 = fig.add_subplot(2, 2, 3)

    diameters = [0.005, 0.01, 0.02, 0.05]  # 5mm, 1cm, 2cm, 5cm
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(diameters)))

    for D, color in zip(diameters, colors):
        theta_range = np.linspace(0, 10 * rayleigh_criterion(wavelength, D), 500)
        I = airy_pattern_1d(theta_range, wavelength, D)
        theta_arcsec = np.degrees(theta_range) * 3600
        ax3.semilogy(theta_arcsec, I, color=color, lw=2,
                    label=f'D = {D*100:.1f} cm, θ_R = {np.degrees(rayleigh_criterion(wavelength, D))*3600:.2f}"')

    ax3.set_xlabel('Angle (arcseconds)')
    ax3.set_ylabel('Relative Intensity (log)')
    ax3.set_title('Resolution vs Aperture Size')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, which='both')
    ax3.set_ylim(1e-6, 1.5)

    # Plot 4: Rayleigh criterion demonstration - two point sources
    ax4 = fig.add_subplot(2, 2, 4)

    # Two point sources at different separations
    separations = [0.5, 1.0, 1.5, 2.0]  # In units of Rayleigh criterion

    theta_range = np.linspace(-3 * theta_R, 3 * theta_R, 500)
    theta_arcsec = np.degrees(theta_range) * 3600

    for sep in separations:
        theta_sep = sep * theta_R

        # Two sources
        I1 = airy_pattern_1d(theta_range - theta_sep/2, wavelength, diameter)
        I2 = airy_pattern_1d(theta_range + theta_sep/2, wavelength, diameter)
        I_total = I1 + I2

        label = f'{sep:.1f}×θ_R'
        if sep == 1.0:
            label += ' (Rayleigh limit)'
        ax4.plot(theta_arcsec, I_total / I_total.max(), lw=2, label=label)

    ax4.set_xlabel('Angle (arcseconds)')
    ax4.set_ylabel('Normalized Intensity')
    ax4.set_title('Two Point Sources - Rayleigh Criterion')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Add text explaining Rayleigh criterion
    ax4.text(0.95, 0.95,
             'Rayleigh criterion:\nSources resolvable when\nseparation ≥ 1.22λ/D\n(central max at first min)',
             transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Experiment 108: Circular Aperture Diffraction (Airy Disk)\n'
                 '$I(\\theta) = I_0 \\left[\\frac{2 J_1(ka\\sin\\theta)}{ka\\sin\\theta}\\right]^2$',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save output
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'airy_disk.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'airy_disk.png')}")

    # Print summary
    print("\n=== Airy Disk Summary ===")
    print(f"Wavelength: {wavelength*1e9:.0f} nm")
    print(f"Aperture diameter: {diameter*100:.1f} cm")
    print(f"Focal length: {focal_length*100:.0f} cm")
    print(f"\nRayleigh criterion: {np.degrees(theta_R)*3600:.3f} arcseconds")
    print(f"First dark ring radius: {r_airy*1e6:.2f} μm")
    print(f"\nEncircled energy:")
    print(f"  Central disk (to 1st min): 83.8%")
    print(f"  To 2nd minimum: 91.0%")
    print(f"  To 3rd minimum: 93.8%")


if __name__ == "__main__":
    main()
