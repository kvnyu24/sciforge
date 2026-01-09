"""
Example demonstrating wave interference patterns.

This example shows interference between two coherent wave sources,
demonstrating constructive and destructive interference patterns.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.waves import Wave


def calculate_interference_pattern(x, y, source1_pos, source2_pos, wavelength, amplitude=1.0):
    """
    Calculate interference pattern from two point sources.

    Args:
        x, y: Coordinate arrays (meshgrid)
        source1_pos, source2_pos: Source positions (x, y)
        wavelength: Wavelength of waves
        amplitude: Amplitude of each source

    Returns:
        Interference pattern as 2D array
    """
    k = 2 * np.pi / wavelength

    # Distance from each source
    r1 = np.sqrt((x - source1_pos[0])**2 + (y - source1_pos[1])**2)
    r2 = np.sqrt((x - source2_pos[0])**2 + (y - source2_pos[1])**2)

    # Avoid division by zero
    r1 = np.maximum(r1, 0.01)
    r2 = np.maximum(r2, 0.01)

    # Spherical waves from each source
    wave1 = amplitude * np.sin(k * r1) / np.sqrt(r1)
    wave2 = amplitude * np.sin(k * r2) / np.sqrt(r2)

    return wave1 + wave2


def main():
    # Create coordinate grid
    x = np.linspace(-10, 10, 500)
    y = np.linspace(-10, 10, 500)
    X, Y = np.meshgrid(x, y)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Different source separations
    separations = [0.5, 1.0, 2.0]
    wavelength = 1.0

    for idx, d in enumerate(separations):
        # Source positions (symmetric about origin)
        source1 = (-d/2, 0)
        source2 = (d/2, 0)

        pattern = calculate_interference_pattern(X, Y, source1, source2, wavelength)

        # Plot interference pattern
        ax1 = axes[0, idx]
        im1 = ax1.imshow(pattern, extent=[x.min(), x.max(), y.min(), y.max()],
                         cmap='RdBu', vmin=-2, vmax=2, origin='lower')
        ax1.plot(source1[0], source1[1], 'ko', markersize=8)
        ax1.plot(source2[0], source2[1], 'ko', markersize=8)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title(f'Source separation d = {d}λ')
        plt.colorbar(im1, ax=ax1, label='Amplitude')

        # Plot intensity pattern (square of amplitude)
        intensity = pattern**2
        ax2 = axes[1, idx]
        im2 = ax2.imshow(intensity, extent=[x.min(), x.max(), y.min(), y.max()],
                         cmap='hot', origin='lower')
        ax2.plot(source1[0], source1[1], 'wo', markersize=8)
        ax2.plot(source2[0], source2[1], 'wo', markersize=8)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title('Intensity Pattern')
        plt.colorbar(im2, ax=ax2, label='Intensity')

    plt.suptitle(f'Two-Source Interference Patterns (λ = {wavelength})',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save first figure
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'wave_interference.png'), dpi=150, bbox_inches='tight')

    # Create figure showing intensity profile along screen
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))

    y_screen = 8.0  # Position of observation screen
    x_screen = np.linspace(-10, 10, 500)

    for idx, d in enumerate(separations):
        source1 = (-d/2, 0)
        source2 = (d/2, 0)

        # Calculate path difference at each point on screen
        r1 = np.sqrt((x_screen - source1[0])**2 + y_screen**2)
        r2 = np.sqrt((x_screen - source2[0])**2 + y_screen**2)

        # Phase difference
        k = 2 * np.pi / wavelength
        delta_phi = k * (r2 - r1)

        # Intensity (for equal amplitude sources)
        intensity = 4 * np.cos(delta_phi / 2)**2

        ax = axes2[idx]
        ax.plot(x_screen, intensity, 'b-', lw=2)
        ax.fill_between(x_screen, intensity, alpha=0.3)
        ax.set_xlabel('Position on screen (x)')
        ax.set_ylabel('Intensity (a.u.)')
        ax.set_title(f'd = {d}λ')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 5)

        # Mark maxima positions
        for n in range(-5, 6):
            x_max = n * wavelength * y_screen / d if d != 0 else 0
            if abs(x_max) < 10:
                ax.axvline(x=x_max, color='r', linestyle='--', alpha=0.3)

    fig2.suptitle(f'Intensity Profile at Screen (y = {y_screen})', fontsize=14, y=1.02)
    plt.tight_layout()

    fig2.savefig(os.path.join(output_dir, 'wave_interference_profile.png'), dpi=150, bbox_inches='tight')
    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
