"""
Experiment 80: Point charge field and equipotentials.

This example demonstrates the electric field from a single point charge,
including field lines, equipotential surfaces, and the 1/r^2 falloff.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def point_charge_field(x, y, z, q, x0=0, y0=0, z0=0, k=8.99e9):
    """
    Calculate electric field from a point charge.

    Args:
        x, y, z: Coordinate arrays
        q: Charge (C)
        x0, y0, z0: Charge position
        k: Coulomb constant (N*m^2/C^2)

    Returns:
        Ex, Ey, Ez: Electric field components (V/m)
        V: Electric potential (V)
    """
    # Distance from charge
    rx = x - x0
    ry = y - y0
    rz = z - z0
    r = np.sqrt(rx**2 + ry**2 + rz**2)

    # Avoid singularity at charge location
    r = np.maximum(r, 1e-10)

    # Electric field: E = kq*r_hat / r^2 = kq*r / r^3
    Ex = k * q * rx / r**3
    Ey = k * q * ry / r**3
    Ez = k * q * rz / r**3

    # Electric potential: V = kq / r
    V = k * q / r

    return Ex, Ey, Ez, V


def main():
    # Parameters
    q = 1e-9  # 1 nC point charge
    k = 8.99e9  # Coulomb constant

    # Create 2D grid for field visualization
    n_points = 200
    x = np.linspace(-2, 2, n_points)
    y = np.linspace(-2, 2, n_points)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    # Calculate field and potential
    Ex, Ey, Ez, V = point_charge_field(X, Y, Z, q)
    E_mag = np.sqrt(Ex**2 + Ey**2 + Ez**2)

    # Create figure
    fig = plt.figure(figsize=(16, 12))

    # Plot 1: Electric field lines with streamplot
    ax1 = fig.add_subplot(2, 2, 1)

    # Use log scale for color
    E_mag_log = np.log10(E_mag + 1)

    stream = ax1.streamplot(X, Y, Ex, Ey, color=E_mag_log, cmap='hot',
                            density=2, linewidth=1, arrowsize=1)
    ax1.plot(0, 0, 'ro', markersize=15, label=f'q = {q*1e9:.1f} nC')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_title('Electric Field Lines')
    ax1.legend()
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(stream.lines, ax=ax1, label='log10(|E|) (V/m)')

    # Plot 2: Equipotential lines
    ax2 = fig.add_subplot(2, 2, 2)

    # Clip potential for visualization
    V_clipped = np.clip(V, -1e10, 1e10)

    # Create equipotential contours
    levels = np.logspace(0, 10, 21)
    contour = ax2.contour(X, Y, V_clipped, levels=levels, cmap='coolwarm')
    ax2.clabel(contour, inline=True, fontsize=8, fmt='%.1e')
    ax2.plot(0, 0, 'ro', markersize=15)
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')
    ax2.set_title('Equipotential Lines (V)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Field magnitude and 1/r^2 falloff
    ax3 = fig.add_subplot(2, 2, 3)

    # Calculate field along radial direction
    r = np.linspace(0.1, 3, 100)
    E_radial = k * q / r**2
    V_radial = k * q / r

    ax3.loglog(r, E_radial, 'b-', lw=2, label='|E| (numerical)')
    ax3.loglog(r, k * q / r**2, 'r--', lw=2, label='kq/r^2 (theory)')

    # Add slope reference line
    r_ref = np.array([0.2, 2])
    ax3.loglog(r_ref, 1e9 * r_ref**(-2), 'g:', lw=1, alpha=0.7, label='slope = -2')

    ax3.set_xlabel('Distance r (m)')
    ax3.set_ylabel('Electric Field |E| (V/m)')
    ax3.set_title('Field Magnitude vs Distance (1/r^2 law)')
    ax3.legend()
    ax3.grid(True, alpha=0.3, which='both')

    # Plot 4: 3D visualization of equipotential surfaces
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')

    # Create spherical equipotential surfaces
    phi = np.linspace(0, 2*np.pi, 50)
    theta = np.linspace(0, np.pi, 50)
    PHI, THETA = np.meshgrid(phi, theta)

    # Plot several equipotential surfaces
    V_values = [k * q / 0.3, k * q / 0.6, k * q / 0.9, k * q / 1.2]
    colors = ['red', 'orange', 'yellow', 'green']

    for V_eq, color in zip(V_values, colors):
        r_eq = k * q / V_eq  # Radius of equipotential
        X_eq = r_eq * np.sin(THETA) * np.cos(PHI)
        Y_eq = r_eq * np.sin(THETA) * np.sin(PHI)
        Z_eq = r_eq * np.cos(THETA)
        ax4.plot_surface(X_eq, Y_eq, Z_eq, alpha=0.3, color=color,
                        label=f'V = {V_eq:.1e} V')

    # Draw field lines in 3D
    n_lines = 8
    for i in range(n_lines):
        phi_line = 2 * np.pi * i / n_lines
        r_line = np.linspace(0.1, 1.5, 20)
        x_line = r_line * np.cos(phi_line)
        y_line = r_line * np.sin(phi_line)
        z_line = np.zeros_like(r_line)
        ax4.plot(x_line, y_line, z_line, 'b-', lw=1, alpha=0.7)

    ax4.scatter([0], [0], [0], c='red', s=100, marker='o')
    ax4.set_xlabel('x (m)')
    ax4.set_ylabel('y (m)')
    ax4.set_zlabel('z (m)')
    ax4.set_title('3D Equipotential Surfaces')

    # Add overall title
    plt.suptitle(f'Point Charge Electric Field\n'
                 f'q = {q*1e9:.1f} nC, k = {k:.2e} N*m^2/C^2',
                 fontsize=14, y=1.02)

    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'point_charge_field.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
