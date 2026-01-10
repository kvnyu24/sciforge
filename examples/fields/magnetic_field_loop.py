"""
Example demonstrating the magnetic field of a current loop.

This example shows the magnetic field pattern around a circular
current loop, approaching a magnetic dipole at large distances.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.fields import MagneticField


def biot_savart_loop(x, y, z, R, I, n_segments=100, mu_0=4*np.pi*1e-7):
    """
    Calculate magnetic field from a circular current loop using Biot-Savart law.

    Loop is centered at origin in the x-y plane with radius R.

    Args:
        x, y, z: Field point coordinates
        R: Loop radius (m)
        I: Current (A)
        n_segments: Number of segments for numerical integration
        mu_0: Permeability of free space

    Returns:
        Bx, By, Bz: Magnetic field components (T)
    """
    # Loop parameterization
    phi = np.linspace(0, 2*np.pi, n_segments)
    dphi = 2*np.pi / n_segments

    # Initialize field
    Bx = np.zeros_like(x)
    By = np.zeros_like(x)
    Bz = np.zeros_like(x)

    for p in phi:
        # Position on loop
        x_loop = R * np.cos(p)
        y_loop = R * np.sin(p)
        z_loop = 0

        # dl vector (tangent to loop)
        dlx = -R * np.sin(p) * dphi
        dly = R * np.cos(p) * dphi
        dlz = 0

        # r vector from loop element to field point
        rx = x - x_loop
        ry = y - y_loop
        rz = z - z_loop

        r_mag = np.sqrt(rx**2 + ry**2 + rz**2)
        r_mag = np.maximum(r_mag, 0.01)  # Avoid singularity

        # Cross product dl × r
        cross_x = dly * rz - dlz * ry
        cross_y = dlz * rx - dlx * rz
        cross_z = dlx * ry - dly * rx

        # Biot-Savart law: dB = (mu_0 I / 4pi) * (dl × r) / |r|^3
        factor = mu_0 * I / (4 * np.pi) / r_mag**3

        Bx += factor * cross_x
        By += factor * cross_y
        Bz += factor * cross_z

    return Bx, By, Bz


def main():
    # Parameters
    R = 1.0           # Loop radius (m)
    I = 1.0           # Current (A)
    mu_0 = 4 * np.pi * 1e-7  # Permeability of free space

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Field in x-z plane (through loop center)
    ax1 = axes[0, 0]

    x = np.linspace(-3, 3, 100)
    z = np.linspace(-3, 3, 100)
    X, Z = np.meshgrid(x, z)
    Y = np.zeros_like(X)

    Bx, By, Bz = biot_savart_loop(X, Y, Z, R, I)
    B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)

    # Mask inside loop region
    mask = (np.abs(Z) < 0.1) & (np.abs(X) < R)

    ax1.streamplot(X, Z, Bx, Bz, color=np.log10(B_mag + 1e-10), cmap='viridis',
                   linewidth=1, density=2)

    # Draw the loop (as cross-section)
    ax1.plot([-R, -R], [-0.05, 0.05], 'r-', lw=5)
    ax1.plot([R, R], [-0.05, 0.05], 'r-', lw=5)

    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('z (m)')
    ax1.set_title('Magnetic Field in x-z Plane (y=0)')
    ax1.set_aspect('equal')

    # Plot 2: Field along axis
    ax2 = axes[0, 1]

    z_axis = np.linspace(-5*R, 5*R, 200)
    _, _, Bz_axis = biot_savart_loop(np.zeros_like(z_axis), np.zeros_like(z_axis), z_axis, R, I)

    # Theoretical on-axis field
    B_theory = mu_0 * I * R**2 / (2 * (R**2 + z_axis**2)**(3/2))

    ax2.plot(z_axis/R, Bz_axis * 1e6, 'b-', lw=2, label='Numerical')
    ax2.plot(z_axis/R, B_theory * 1e6, 'r--', lw=2, label='Analytical')
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

    ax2.set_xlabel('z/R (axial distance in loop radii)')
    ax2.set_ylabel('Bz (μT)')
    ax2.set_title('On-Axis Magnetic Field')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Mark center field
    B_center = mu_0 * I / (2 * R)
    ax2.plot(0, B_center * 1e6, 'go', markersize=10, label=f'B(0) = {B_center*1e6:.2f} μT')

    # Plot 3: Field magnitude contours
    ax3 = axes[1, 0]

    B_mag_log = np.log10(B_mag + 1e-10)
    levels = np.linspace(B_mag_log.min(), B_mag_log.max(), 20)

    contour = ax3.contourf(X, Z, B_mag_log, levels=levels, cmap='plasma')
    ax3.contour(X, Z, B_mag_log, levels=levels, colors='white', linewidths=0.5, alpha=0.3)
    plt.colorbar(contour, ax=ax3, label='log₁₀(|B|)')

    # Draw loop
    ax3.plot([-R, -R], [-0.05, 0.05], 'c-', lw=5)
    ax3.plot([R, R], [-0.05, 0.05], 'c-', lw=5)

    ax3.set_xlabel('x (m)')
    ax3.set_ylabel('z (m)')
    ax3.set_title('Field Magnitude (log scale)')
    ax3.set_aspect('equal')

    # Plot 4: Comparison with dipole approximation
    ax4 = axes[1, 1]

    # Magnetic dipole moment
    m = I * np.pi * R**2

    # Far-field (dipole) approximation on axis
    z_far = np.linspace(2*R, 10*R, 100)
    B_dipole = mu_0 * 2 * m / (4 * np.pi * z_far**3)

    # Exact on-axis field
    B_exact = mu_0 * I * R**2 / (2 * (R**2 + z_far**2)**(3/2))

    ax4.loglog(z_far/R, B_exact * 1e6, 'b-', lw=2, label='Exact')
    ax4.loglog(z_far/R, B_dipole * 1e6, 'r--', lw=2, label='Dipole approx.')

    ax4.set_xlabel('z/R')
    ax4.set_ylabel('Bz (μT)')
    ax4.set_title('Exact vs Dipole Approximation (far field)')
    ax4.legend()
    ax4.grid(True, alpha=0.3, which='both')

    # Add annotation about dipole validity
    ax4.text(0.5, 0.05, 'Dipole approximation valid for z >> R',
             transform=ax4.transAxes, fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'Magnetic Field of a Current Loop\n'
                 f'R = {R} m, I = {I} A, μ = πR²I = {m:.4f} A⋅m²',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'magnetic_field_loop.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'magnetic_field_loop.png')}")


if __name__ == "__main__":
    main()
