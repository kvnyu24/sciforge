"""
Example demonstrating diffraction grating patterns.

This example shows how a diffraction grating produces interference
patterns, demonstrating the relationship between grating spacing,
wavelength, and diffraction angles.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.em_waves import EMWave


def grating_intensity(theta, d, wavelength, N):
    """
    Calculate intensity pattern from a diffraction grating.

    I(θ) = I₀ * [sin(N*β)/sin(β)]² * [sin(α)/α]²

    where α = (π*a/λ)*sin(θ), β = (π*d/λ)*sin(θ)
    a = slit width, d = grating spacing, N = number of slits

    For simplicity, assuming slits are very narrow (a << d).

    Args:
        theta: Angle array (radians)
        d: Grating spacing
        wavelength: Light wavelength
        N: Number of slits

    Returns:
        Intensity pattern (normalized)
    """
    beta = np.pi * d * np.sin(theta) / wavelength

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        intensity = (np.sin(N * beta) / np.sin(beta))**2
        intensity = np.nan_to_num(intensity, nan=N**2)

    return intensity / intensity.max()


def grating_maxima_angles(m_max, d, wavelength):
    """
    Calculate angles of diffraction maxima: d*sin(θ) = m*λ

    Args:
        m_max: Maximum order to calculate
        d: Grating spacing
        wavelength: Light wavelength

    Returns:
        Dictionary of order: angle pairs
    """
    angles = {}
    for m in range(-m_max, m_max + 1):
        sin_theta = m * wavelength / d
        if abs(sin_theta) <= 1:
            angles[m] = np.arcsin(sin_theta)
    return angles


def main():
    # Parameters
    d = 1e-6          # Grating spacing (1 μm = 1000 lines/mm)
    wavelengths = {
        'Red': 650e-9,
        'Green': 530e-9,
        'Blue': 470e-9
    }
    N = 100           # Number of slits

    # Angle array
    theta = np.linspace(-np.pi/3, np.pi/3, 5000)
    theta_deg = np.degrees(theta)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Diffraction pattern for single wavelength
    ax1 = axes[0, 0]

    wavelength = 550e-9  # Yellow-green light
    intensity = grating_intensity(theta, d, wavelength, N)

    ax1.plot(theta_deg, intensity, 'b-', lw=1)
    ax1.set_xlabel('Angle (degrees)')
    ax1.set_ylabel('Intensity (normalized)')
    ax1.set_title(f'Diffraction Pattern (λ = {wavelength*1e9:.0f} nm, N = {N})')
    ax1.grid(True, alpha=0.3)

    # Mark principal maxima
    angles = grating_maxima_angles(3, d, wavelength)
    for m, angle in angles.items():
        ax1.axvline(x=np.degrees(angle), color='red', linestyle=':', alpha=0.5)
        ax1.annotate(f'm = {m}', (np.degrees(angle), 0.8), fontsize=8, rotation=90)

    # Plot 2: White light spectrum
    ax2 = axes[0, 1]

    colors = {'Red': 'red', 'Green': 'green', 'Blue': 'blue'}
    for color_name, wl in wavelengths.items():
        I = grating_intensity(theta, d, wl, N)
        ax2.plot(theta_deg, I, color=colors[color_name], lw=1.5, label=f'{color_name} ({wl*1e9:.0f} nm)', alpha=0.7)

    ax2.set_xlabel('Angle (degrees)')
    ax2.set_ylabel('Intensity')
    ax2.set_title('White Light Dispersion')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Effect of number of slits
    ax3 = axes[0, 2]

    N_values = [5, 20, 100, 500]
    colors_N = plt.cm.viridis(np.linspace(0.2, 0.8, len(N_values)))

    # Zoom in around first order maximum
    theta_zoom = np.linspace(0.45, 0.65, 1000)
    theta_zoom_deg = np.degrees(theta_zoom)

    for N_test, color in zip(N_values, colors_N):
        I = grating_intensity(theta_zoom, d, wavelength, N_test)
        ax3.plot(theta_zoom_deg, I, color=color, lw=1.5, label=f'N = {N_test}')

    ax3.set_xlabel('Angle (degrees)')
    ax3.set_ylabel('Intensity')
    ax3.set_title('Effect of Number of Slits on Resolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Grating equation visualization
    ax4 = axes[1, 0]

    # Plot d*sin(θ) = m*λ
    m_orders = [-2, -1, 0, 1, 2]
    wavelength_range = np.linspace(400e-9, 700e-9, 100)

    for m in m_orders:
        angles_m = np.arcsin(m * wavelength_range / d)
        valid = np.isfinite(angles_m)
        if np.any(valid):
            ax4.plot(wavelength_range[valid] * 1e9, np.degrees(angles_m[valid]), lw=2, label=f'm = {m}')

    ax4.set_xlabel('Wavelength (nm)')
    ax4.set_ylabel('Diffraction Angle (degrees)')
    ax4.set_title(f'Grating Equation: d⋅sin(θ) = m⋅λ (d = {d*1e6:.1f} μm)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    # Plot 5: Angular dispersion
    ax5 = axes[1, 1]

    # dθ/dλ = m / (d*cos(θ))
    for m in [1, 2, 3]:
        sin_theta = m * wavelength_range / d
        valid = np.abs(sin_theta) < 1
        cos_theta = np.sqrt(1 - sin_theta[valid]**2)
        dispersion = m / (d * cos_theta)  # rad/m

        ax5.plot(wavelength_range[valid] * 1e9, dispersion / 1e6, lw=2, label=f'm = {m}')

    ax5.set_xlabel('Wavelength (nm)')
    ax5.set_ylabel('Angular Dispersion (rad/μm)')
    ax5.set_title('Angular Dispersion: dθ/dλ = m/(d⋅cos θ)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Resolving power
    ax6 = axes[1, 2]

    # Resolving power R = λ/Δλ = m*N
    N_range = np.logspace(1, 4, 100)

    for m in [1, 2, 3]:
        R = m * N_range
        ax6.loglog(N_range, R, lw=2, label=f'm = {m}')

    # Mark some reference resolutions
    ax6.axhline(y=1000, color='gray', linestyle=':', alpha=0.5)
    ax6.annotate('Δλ = 0.5 nm at 500 nm', (100, 1000), fontsize=9)

    ax6.set_xlabel('Number of Slits N')
    ax6.set_ylabel('Resolving Power R = λ/Δλ')
    ax6.set_title('Resolving Power: R = m⋅N')
    ax6.legend()
    ax6.grid(True, alpha=0.3, which='both')

    plt.suptitle(f'Diffraction Grating (d = {d*1e6:.1f} μm = {1/(d*1000):.0f} lines/mm)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'diffraction_grating.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'diffraction_grating.png')}")


if __name__ == "__main__":
    main()
