"""
Example demonstrating electromagnetic wave polarization.

This example shows different polarization states of light:
linear, circular, and elliptical polarization.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.sciforge.physics.em_waves import EMWave


def polarized_wave(z, t, E0x, E0y, phi_x, phi_y, k, omega):
    """
    Calculate electric field of polarized wave.

    E_x = E0x * cos(kz - ωt + φ_x)
    E_y = E0y * cos(kz - ωt + φ_y)

    Args:
        z: Position array
        t: Time
        E0x, E0y: Amplitude components
        phi_x, phi_y: Phase components
        k: Wave number
        omega: Angular frequency

    Returns:
        Ex, Ey: Electric field components
    """
    Ex = E0x * np.cos(k * z - omega * t + phi_x)
    Ey = E0y * np.cos(k * z - omega * t + phi_y)
    return Ex, Ey


def main():
    # Wave parameters
    wavelength = 1.0
    k = 2 * np.pi / wavelength
    omega = 2 * np.pi  # Frequency

    # Spatial array
    z = np.linspace(0, 3 * wavelength, 500)

    fig = plt.figure(figsize=(16, 12))

    # Define polarization states
    polarizations = {
        'Linear (Horizontal)': (1.0, 0.0, 0.0, 0.0),
        'Linear (Vertical)': (0.0, 1.0, 0.0, 0.0),
        'Linear (45°)': (1.0, 1.0, 0.0, 0.0),
        'Right Circular': (1.0, 1.0, 0.0, np.pi/2),
        'Left Circular': (1.0, 1.0, 0.0, -np.pi/2),
        'Elliptical': (1.0, 0.5, 0.0, np.pi/4)
    }

    # 3D plots for each polarization
    for idx, (name, params) in enumerate(polarizations.items()):
        E0x, E0y, phi_x, phi_y = params

        # Create 3D subplot
        ax = fig.add_subplot(2, 3, idx + 1, projection='3d')

        # Plot wave at t=0
        Ex, Ey = polarized_wave(z, 0, E0x, E0y, phi_x, phi_y, k, omega)

        # Plot the wave in 3D
        ax.plot(z, Ex, Ey, 'b-', lw=1.5, alpha=0.8)

        # Plot projections
        ax.plot(z, Ex, np.min(Ey) - 0.5 * np.ones_like(z), 'r-', lw=0.5, alpha=0.5)
        ax.plot(z, np.min(Ex) - 0.5 * np.ones_like(z), Ey, 'g-', lw=0.5, alpha=0.5)

        # Mark the polarization ellipse at z=0
        t_ellipse = np.linspace(0, 2*np.pi/omega, 100)
        Ex_ellipse, Ey_ellipse = polarized_wave(0, t_ellipse, E0x, E0y, phi_x, phi_y, k, omega)
        ax.plot(np.zeros_like(t_ellipse), Ex_ellipse, Ey_ellipse, 'k-', lw=2)

        ax.set_xlabel('z')
        ax.set_ylabel('Ex')
        ax.set_zlabel('Ey')
        ax.set_title(name)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'polarization_3d.png'),
                dpi=150, bbox_inches='tight')

    # Create 2D figure showing polarization ellipses
    fig2, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (name, params) in enumerate(polarizations.items()):
        E0x, E0y, phi_x, phi_y = params
        ax = axes[idx]

        # Time evolution of field at z=0
        t = np.linspace(0, 2*np.pi/omega, 500)
        Ex, Ey = polarized_wave(0, t, E0x, E0y, phi_x, phi_y, k, omega)

        # Color by time
        colors = plt.cm.viridis(np.linspace(0, 1, len(t)))
        for i in range(len(t) - 1):
            ax.plot(Ex[i:i+2], Ey[i:i+2], color=colors[i], lw=2)

        # Mark start and direction
        ax.plot(Ex[0], Ey[0], 'go', markersize=10, label='Start')
        ax.arrow(Ex[len(t)//4], Ey[len(t)//4],
                Ex[len(t)//4 + 1] - Ex[len(t)//4],
                Ey[len(t)//4 + 1] - Ey[len(t)//4],
                head_width=0.1, head_length=0.05, fc='red', ec='red')

        ax.set_xlabel('Ex')
        ax.set_ylabel('Ey')
        ax.set_title(name)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

    plt.suptitle('Polarization States: Electric Field Vector Traces\n'
                 '(as viewed looking into the oncoming wave)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plots
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    fig2.savefig(os.path.join(output_dir, 'polarization_ellipses.png'), dpi=150, bbox_inches='tight')
    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
