"""
Example demonstrating Moiré pattern visualization.

This example shows how overlapping two sets of parallel lines or wave patterns
with slight rotation creates interesting interference patterns known as Moiré patterns.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics import Wave

def create_wave_pattern(x, y, wavelength, angle):
    """Create a rotated wave pattern"""
    # Rotate coordinates
    x_rot = x * np.cos(angle) - y * np.sin(angle)
    y_rot = x * np.sin(angle) + y * np.cos(angle)
    
    # Create wave pattern
    k = 2 * np.pi / wavelength
    return 0.5 * (1 + np.sin(k * x_rot))

def simulate_moire_pattern():
    # Create coordinate grid
    x = np.linspace(-5, 5, 500)
    y = np.linspace(-5, 5, 500)
    X, Y = np.meshgrid(x, y)
    
    # Parameters for two wave patterns
    wavelength = 0.2
    angle1 = 0
    angle2 = np.pi/16  # Small rotation angle
    
    # Generate patterns
    pattern1 = create_wave_pattern(X, Y, wavelength, angle1)
    pattern2 = create_wave_pattern(X, Y, wavelength, angle2)
    
    # Combine patterns
    moire = pattern1 * pattern2
    
    return x, y, pattern1, pattern2, moire

def plot_results(x, y, pattern1, pattern2, moire):
    """Plot individual patterns and resulting Moiré pattern"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot first pattern
    im1 = ax1.imshow(pattern1, extent=[x[0], x[-1], y[0], y[-1]], 
                     cmap='binary', aspect='equal')
    ax1.set_title('Pattern 1')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    
    # Plot second pattern
    im2 = ax2.imshow(pattern2, extent=[x[0], x[-1], y[0], y[-1]], 
                     cmap='binary', aspect='equal')
    ax2.set_title('Pattern 2')
    ax2.set_xlabel('x')
    
    # Plot Moiré pattern
    im3 = ax3.imshow(moire, extent=[x[0], x[-1], y[0], y[-1]], 
                     cmap='binary', aspect='equal')
    ax3.set_title('Moiré Pattern')
    ax3.set_xlabel('x')
    
    # Add colorbars
    plt.colorbar(im1, ax=ax1, label='Intensity')
    plt.colorbar(im2, ax=ax2, label='Intensity')
    plt.colorbar(im3, ax=ax3, label='Intensity')
    
    plt.tight_layout()

def main():
    # Generate Moiré pattern
    x, y, pattern1, pattern2, moire = simulate_moire_pattern()
    
    # Plot results
    plot_results(x, y, pattern1, pattern2, moire)
    plt.show()

if __name__ == "__main__":
    main() 