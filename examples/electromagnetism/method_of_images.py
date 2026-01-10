"""
Experiment 84: Method of images.

This example demonstrates the method of images for solving electrostatic
problems with conducting boundaries. We show the classic case of a point
charge near a grounded conducting plane.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt


# Constants
K = 8.99e9  # Coulomb constant (N*m^2/C^2)
EPSILON_0 = 8.854e-12


def point_charge_potential(x, y, q, x0, y0):
    """Calculate potential from a point charge."""
    r = np.sqrt((x - x0)**2 + (y - y0)**2)
    r = np.maximum(r, 1e-10)
    return K * q / r


def point_charge_field(x, y, q, x0, y0):
    """Calculate electric field from a point charge."""
    rx = x - x0
    ry = y - y0
    r = np.sqrt(rx**2 + ry**2)
    r = np.maximum(r, 1e-10)

    factor = K * q / r**3
    return factor * rx, factor * ry


def image_solution(x, y, q, d):
    """
    Calculate potential and field for charge near conducting plane.

    Real charge at (0, d), image charge at (0, -d).
    Conducting plane at y = 0.

    Args:
        x, y: Coordinate arrays
        q: Real charge (C)
        d: Distance from charge to plane (m)

    Returns:
        V: Total potential
        Ex, Ey: Total field components
    """
    # Real charge at (0, d)
    V_real = point_charge_potential(x, y, q, 0, d)
    Ex_real, Ey_real = point_charge_field(x, y, q, 0, d)

    # Image charge at (0, -d) with opposite sign
    V_image = point_charge_potential(x, y, -q, 0, -d)
    Ex_image, Ey_image = point_charge_field(x, y, -q, 0, -d)

    # Total (superposition)
    V_total = V_real + V_image
    Ex_total = Ex_real + Ex_image
    Ey_total = Ey_real + Ey_image

    return V_total, Ex_total, Ey_total


def surface_charge_density(x, q, d):
    """
    Induced surface charge density on the conducting plane.

    sigma = -epsilon_0 * E_normal at y = 0
    """
    # At y = 0, the normal field is Ey
    r = np.sqrt(x**2 + d**2)
    # E_y from real charge + image charge at y=0
    Ey_real = K * q * d / r**3
    Ey_image = K * (-q) * (-d) / r**3
    Ey_total = Ey_real + Ey_image

    # Surface charge density (pointing into conductor is negative)
    sigma = EPSILON_0 * Ey_total
    return -sigma  # Convention: sigma is charge on conductor surface


def main():
    # Parameters
    q = 1e-9     # 1 nC charge
    d = 0.5      # 50 cm from plane

    # Create 2D grid
    n = 200
    x = np.linspace(-2, 2, n)
    y = np.linspace(-1, 2, n)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(16, 12))

    # Plot 1: Potential with conducting plane
    ax1 = fig.add_subplot(2, 2, 1)

    V, Ex, Ey = image_solution(X, Y, q, d)

    # Only show region above conducting plane (y > 0)
    V_display = np.where(Y > 0, V, 0)

    # Equipotential contours
    V_clipped = np.clip(V_display, -1e10, 1e10)
    levels = np.linspace(-5e9, 20e9, 26)
    contour = ax1.contourf(X, Y, V_clipped, levels=levels, cmap='RdBu_r', extend='both')
    ax1.contour(X, Y, V_clipped, levels=[0], colors='white', linewidths=2)
    plt.colorbar(contour, ax=ax1, label='Potential (V)')

    # Field lines (only above plane)
    Ex_display = np.where(Y > 0, Ex, 0)
    Ey_display = np.where(Y > 0, Ey, 0)
    ax1.streamplot(X, Y, Ex_display, Ey_display, color='black',
                   linewidth=0.5, density=1.5, arrowsize=1)

    # Draw conducting plane
    ax1.axhline(y=0, color='gray', linewidth=4, label='Conducting plane (V=0)')
    ax1.fill_between(x, -1, 0, color='gray', alpha=0.3)

    # Mark real charge
    ax1.plot(0, d, 'ro', markersize=15, label=f'+q at y={d}m')

    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_title('Charge Near Grounded Conducting Plane\n(Method of Images Solution)')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-1, 2)

    # Plot 2: Image charge interpretation
    ax2 = fig.add_subplot(2, 2, 2)

    # Show the complete picture with image charge
    V_real = point_charge_potential(X, Y, q, 0, d)
    V_image = point_charge_potential(X, Y, -q, 0, -d)
    V_complete = V_real + V_image

    V_clipped2 = np.clip(V_complete, -1e10, 1e10)
    contour2 = ax2.contourf(X, Y, V_clipped2, levels=levels, cmap='RdBu_r', extend='both')
    ax2.contour(X, Y, V_clipped2, levels=[0], colors='white', linewidths=2)
    plt.colorbar(contour2, ax=ax2, label='Potential (V)')

    # Full field streamlines
    Ex_real, Ey_real = point_charge_field(X, Y, q, 0, d)
    Ex_image, Ey_image = point_charge_field(X, Y, -q, 0, -d)
    ax2.streamplot(X, Y, Ex_real + Ex_image, Ey_real + Ey_image,
                   color='black', linewidth=0.5, density=1.5, arrowsize=1)

    # Mark both charges
    ax2.plot(0, d, 'ro', markersize=15, label=f'+q (real)')
    ax2.plot(0, -d, 'bo', markersize=15, label=f'-q (image)')

    # Symmetry plane
    ax2.axhline(y=0, color='green', linewidth=2, linestyle='--',
                label='V=0 (symmetry plane)')

    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')
    ax2.set_title('Equivalent Problem with Image Charge\n(Both charges shown)')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)

    # Plot 3: Surface charge density on conductor
    ax3 = fig.add_subplot(2, 2, 3)

    x_surface = np.linspace(-3, 3, 500)
    sigma = surface_charge_density(x_surface, q, d)

    ax3.plot(x_surface, sigma * 1e9, 'b-', lw=2)
    ax3.fill_between(x_surface, sigma * 1e9, 0, alpha=0.3, color='blue')

    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax3.axvline(x=0, color='gray', linestyle=':', alpha=0.5)

    ax3.set_xlabel('x position on conductor (m)')
    ax3.set_ylabel('Surface charge density (nC/m^2)')
    ax3.set_title('Induced Surface Charge Density on Conductor')
    ax3.grid(True, alpha=0.3)

    # Calculate total induced charge
    dx = x_surface[1] - x_surface[0]
    Q_induced = np.trapz(sigma, x_surface)  # Per unit length in z
    ax3.text(0.95, 0.95, f'Peak density: {np.min(sigma)*1e9:.2f} nC/m^2',
             transform=ax3.transAxes, ha='right', va='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 4: Field strength comparison
    ax4 = fig.add_subplot(2, 2, 4)

    # Field at different heights above conductor
    heights = [0.1, 0.3, 0.5, 0.8]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(heights)))

    x_test = np.linspace(-2, 2, 200)
    for h, color in zip(heights, colors):
        _, Ex_h, Ey_h = image_solution(x_test, np.full_like(x_test, h), q, d)
        E_mag = np.sqrt(Ex_h**2 + Ey_h**2)
        ax4.semilogy(x_test, E_mag, color=color, lw=2, label=f'y = {h} m')

    ax4.axvline(x=0, color='gray', linestyle=':', alpha=0.5)

    ax4.set_xlabel('x position (m)')
    ax4.set_ylabel('Electric field magnitude (V/m)')
    ax4.set_title('Field Strength at Different Heights')
    ax4.legend()
    ax4.grid(True, alpha=0.3, which='both')

    # Add physics explanation
    fig.text(0.5, 0.02,
             'Method of Images: Replace conducting boundary with image charge to satisfy V=0 condition.\n'
             'The solution above the conductor is identical to the real problem.',
             ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.suptitle(f'Method of Images: Point Charge Near Conducting Plane\n'
                 f'q = {q*1e9:.1f} nC, d = {d*100:.0f} cm',
                 fontsize=14, y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)

    # Save plot
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'method_of_images.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
