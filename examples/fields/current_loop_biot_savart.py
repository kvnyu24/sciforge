"""
Experiment 86: Current loop field (Biot-Savart).

This example demonstrates the magnetic field of a current loop
calculated using the Biot-Savart law, including on-axis and
off-axis field patterns.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Constants
MU_0 = 4 * np.pi * 1e-7  # Permeability of free space (H/m)


def biot_savart_loop(x, y, z, R, I, n_segments=100):
    """
    Calculate magnetic field from a circular current loop using Biot-Savart law.

    Loop is centered at origin in the x-y plane with radius R.

    Args:
        x, y, z: Field point coordinates (can be arrays)
        R: Loop radius (m)
        I: Current (A)
        n_segments: Number of segments for numerical integration

    Returns:
        Bx, By, Bz: Magnetic field components (T)
    """
    # Ensure arrays
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)

    # Preserve original shape
    orig_shape = x.shape

    # Flatten for calculation
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    # Initialize field components
    Bx = np.zeros_like(x)
    By = np.zeros_like(y)
    Bz = np.zeros_like(z)

    # Loop parameterization
    phi = np.linspace(0, 2*np.pi, n_segments, endpoint=False)
    dphi = 2*np.pi / n_segments

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
        r_mag = np.maximum(r_mag, 1e-10)  # Avoid singularity

        # Cross product dl x r
        cross_x = dly * rz - dlz * ry
        cross_y = dlz * rx - dlx * rz
        cross_z = dlx * ry - dly * rx

        # Biot-Savart law: dB = (mu_0 I / 4pi) * (dl x r) / |r|^3
        factor = MU_0 * I / (4 * np.pi) / r_mag**3

        Bx += factor * cross_x
        By += factor * cross_y
        Bz += factor * cross_z

    # Reshape to original
    Bx = Bx.reshape(orig_shape)
    By = By.reshape(orig_shape)
    Bz = Bz.reshape(orig_shape)

    return Bx, By, Bz


def on_axis_field(z, R, I):
    """
    Analytical on-axis magnetic field for a current loop.

    B_z(z) = mu_0 * I * R^2 / (2 * (R^2 + z^2)^(3/2))
    """
    return MU_0 * I * R**2 / (2 * (R**2 + z**2)**(3/2))


def main():
    # Parameters
    R = 1.0    # Loop radius (m)
    I = 1.0    # Current (A)

    fig = plt.figure(figsize=(16, 12))

    # Plot 1: Field in the x-z plane (through loop center)
    ax1 = fig.add_subplot(2, 2, 1)

    n = 80
    x = np.linspace(-3, 3, n)
    z = np.linspace(-3, 3, n)
    X, Z = np.meshgrid(x, z)
    Y = np.zeros_like(X)

    Bx, By, Bz = biot_savart_loop(X, Y, Z, R, I)
    B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)

    # Field streamlines
    ax1.streamplot(X, Z, Bx, Bz, color=np.log10(B_mag + 1e-10),
                   cmap='viridis', linewidth=1, density=2, arrowsize=1)

    # Draw the loop (as cross-section points)
    ax1.plot([-R, -R], [-0.05, 0.05], 'r-', lw=6, label='Current loop')
    ax1.plot([R, R], [-0.05, 0.05], 'r-', lw=6)

    # Add current direction indicators
    ax1.annotate('', xy=(-R, 0.1), xytext=(-R, -0.1),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax1.annotate('', xy=(R, -0.1), xytext=(R, 0.1),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('z (m)')
    ax1.set_title('Magnetic Field in x-z Plane\n(Biot-Savart Law)')
    ax1.set_aspect('equal')
    ax1.legend(loc='upper right')

    # Plot 2: On-axis field comparison
    ax2 = fig.add_subplot(2, 2, 2)

    z_axis = np.linspace(-5*R, 5*R, 200)
    _, _, Bz_numerical = biot_savart_loop(np.zeros_like(z_axis),
                                          np.zeros_like(z_axis),
                                          z_axis, R, I)
    Bz_analytical = on_axis_field(z_axis, R, I)

    ax2.plot(z_axis/R, Bz_numerical * 1e6, 'b-', lw=2, label='Numerical (Biot-Savart)')
    ax2.plot(z_axis/R, Bz_analytical * 1e6, 'r--', lw=2, label='Analytical')

    # Mark center field
    B_center = MU_0 * I / (2 * R)
    ax2.plot(0, B_center * 1e6, 'go', markersize=10,
             label=f'B(0) = {B_center*1e6:.3f} uT')

    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

    ax2.set_xlabel('z/R (axial distance in loop radii)')
    ax2.set_ylabel('Bz (uT)')
    ax2.set_title('On-Axis Magnetic Field')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Field magnitude contours
    ax3 = fig.add_subplot(2, 2, 3)

    B_mag_log = np.log10(B_mag + 1e-10)
    levels = np.linspace(B_mag_log.min(), B_mag_log.max(), 20)

    contour = ax3.contourf(X, Z, B_mag_log, levels=levels, cmap='plasma')
    ax3.contour(X, Z, B_mag_log, levels=levels, colors='white', linewidths=0.3, alpha=0.5)
    plt.colorbar(contour, ax=ax3, label='log10(|B|) (T)')

    # Draw loop
    ax3.plot([-R, -R], [-0.05, 0.05], 'c-', lw=4)
    ax3.plot([R, R], [-0.05, 0.05], 'c-', lw=4)

    ax3.set_xlabel('x (m)')
    ax3.set_ylabel('z (m)')
    ax3.set_title('Field Magnitude (log scale)')
    ax3.set_aspect('equal')

    # Plot 4: 3D visualization
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')

    # Draw the current loop
    theta = np.linspace(0, 2*np.pi, 100)
    x_loop = R * np.cos(theta)
    y_loop = R * np.sin(theta)
    z_loop = np.zeros_like(theta)
    ax4.plot(x_loop, y_loop, z_loop, 'r-', lw=3, label='Current loop')

    # Sample field lines in 3D
    n_lines = 8
    for i in range(n_lines):
        phi = 2 * np.pi * i / n_lines
        # Start points along axis
        z_start = np.linspace(-2, 2, 20)
        x_start = 0.01 * np.cos(phi) * np.ones_like(z_start)
        y_start = 0.01 * np.sin(phi) * np.ones_like(z_start)

        # Get field at starting points
        Bx, By, Bz = biot_savart_loop(x_start, y_start, z_start, R, I)

        # Simple field line tracing
        for j in range(0, len(z_start), 3):
            x_line = [x_start[j]]
            y_line = [y_start[j]]
            z_line = [z_start[j]]

            for _ in range(50):
                Bx_pt, By_pt, Bz_pt = biot_savart_loop(x_line[-1], y_line[-1],
                                                        z_line[-1], R, I)
                B_mag_pt = np.sqrt(Bx_pt**2 + By_pt**2 + Bz_pt**2)
                if B_mag_pt < 1e-10:
                    break

                ds = 0.1  # Step size
                x_new = x_line[-1] + ds * Bx_pt / B_mag_pt
                y_new = y_line[-1] + ds * By_pt / B_mag_pt
                z_new = z_line[-1] + ds * Bz_pt / B_mag_pt

                # Stop if too far from origin
                if np.sqrt(x_new**2 + y_new**2 + z_new**2) > 4:
                    break

                x_line.append(float(x_new))
                y_line.append(float(y_new))
                z_line.append(float(z_new))

            if len(x_line) > 5:
                ax4.plot(x_line, y_line, z_line, 'b-', lw=0.5, alpha=0.5)

    ax4.set_xlabel('x (m)')
    ax4.set_ylabel('y (m)')
    ax4.set_zlabel('z (m)')
    ax4.set_title('3D Field Line Visualization')
    ax4.set_xlim(-2, 2)
    ax4.set_ylim(-2, 2)
    ax4.set_zlim(-2, 2)

    # Add magnetic moment info
    m = I * np.pi * R**2
    fig.text(0.5, 0.02,
             f'Current Loop: R = {R} m, I = {I} A\n'
             f'Magnetic moment: m = I*pi*R^2 = {m:.4f} A*m^2, '
             f'B(center) = mu_0*I/(2R) = {B_center*1e6:.3f} uT',
             ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.suptitle('Current Loop Magnetic Field (Biot-Savart Law)', fontsize=14, y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)

    # Save plot
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'current_loop_biot_savart.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
