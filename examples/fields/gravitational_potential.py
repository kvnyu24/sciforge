"""
Example demonstrating gravitational potential and fields.

This example shows gravitational potential wells and orbital mechanics
around massive bodies, including escape velocity visualization.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.fields import GravitationalField


def calculate_gravitational_field(x, y, M, G=6.674e-11):
    """
    Calculate gravitational field and potential.

    Args:
        x, y: Coordinate arrays
        M: Central mass (kg)
        G: Gravitational constant

    Returns:
        gx, gy: Gravitational field components
        V: Gravitational potential
    """
    r = np.sqrt(x**2 + y**2)
    r = np.maximum(r, 0.1)  # Avoid singularity

    # Field magnitude and direction
    g_mag = G * M / r**2
    gx = -g_mag * x / r
    gy = -g_mag * y / r

    # Potential (negative)
    V = -G * M / r

    return gx, gy, V


def main():
    # Physical constants
    G = 6.674e-11  # m^3/(kg·s^2)

    # Earth parameters
    M_earth = 5.972e24  # kg
    R_earth = 6.371e6   # m

    # Create coordinate grid (in units of Earth radii)
    r_range = 10  # Plot out to 10 Earth radii
    x = np.linspace(-r_range, r_range, 300) * R_earth
    y = np.linspace(-r_range, r_range, 300) * R_earth
    X, Y = np.meshgrid(x, y)

    # Calculate field and potential
    gx, gy, V = calculate_gravitational_field(X, Y, M_earth, G)

    # Mask inside Earth
    R = np.sqrt(X**2 + Y**2)
    mask = R < R_earth

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Potential well (3D-like visualization)
    ax1 = axes[0, 0]
    V_display = V.copy() / 1e7  # Scale for visualization
    V_display[mask] = np.nan

    im1 = ax1.imshow(V_display, extent=[-r_range, r_range, -r_range, r_range],
                     cmap='viridis', origin='lower')
    circle = plt.Circle((0, 0), 1, color='lightblue', alpha=0.8)
    ax1.add_patch(circle)
    ax1.set_xlabel('x (Earth radii)')
    ax1.set_ylabel('y (Earth radii)')
    ax1.set_title('Gravitational Potential')
    plt.colorbar(im1, ax=ax1, label='Potential (10⁷ J/kg)')

    # Plot 2: Field streamlines
    ax2 = axes[0, 1]
    gx_masked = np.where(mask, np.nan, gx)
    gy_masked = np.where(mask, np.nan, gy)

    # Convert to Earth radii for display
    X_er = X / R_earth
    Y_er = Y / R_earth

    ax2.streamplot(X_er, Y_er, gx_masked, gy_masked, color='blue',
                   linewidth=1, density=2, arrowsize=1)
    circle = plt.Circle((0, 0), 1, color='lightblue', alpha=0.8, label='Earth')
    ax2.add_patch(circle)
    ax2.set_xlabel('x (Earth radii)')
    ax2.set_ylabel('y (Earth radii)')
    ax2.set_title('Gravitational Field Lines')
    ax2.legend()
    ax2.set_aspect('equal')
    ax2.set_xlim(-r_range, r_range)
    ax2.set_ylim(-r_range, r_range)

    # Plot 3: Potential along radial direction
    ax3 = axes[1, 0]
    r_values = np.linspace(1.0, 10.0, 100) * R_earth
    V_r = -G * M_earth / r_values

    # Escape velocity at each radius
    v_escape = np.sqrt(2 * G * M_earth / r_values)

    ax3_twin = ax3.twinx()

    line1 = ax3.plot(r_values / R_earth, V_r / 1e7, 'b-', lw=2, label='Potential')
    line2 = ax3_twin.plot(r_values / R_earth, v_escape / 1000, 'r--', lw=2, label='Escape velocity')

    ax3.set_xlabel('Distance from center (Earth radii)')
    ax3.set_ylabel('Potential (10⁷ J/kg)', color='blue')
    ax3_twin.set_ylabel('Escape Velocity (km/s)', color='red')
    ax3.set_title('Potential and Escape Velocity vs Distance')
    ax3.grid(True, alpha=0.3)

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper right')

    # Mark surface values
    v_escape_surface = np.sqrt(2 * G * M_earth / R_earth)
    ax3.annotate(f'Surface escape velocity:\n{v_escape_surface/1000:.1f} km/s',
                xy=(1, -6.25), xytext=(3, -4),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=9)

    # Plot 4: Orbital velocities and periods
    ax4 = axes[1, 1]

    # Circular orbital velocity
    v_circular = np.sqrt(G * M_earth / r_values)

    # Orbital period
    T_orbit = 2 * np.pi * r_values / v_circular / 3600  # hours

    ax4.plot(r_values / R_earth, v_circular / 1000, 'g-', lw=2, label='Circular orbital velocity')
    ax4.plot(r_values / R_earth, v_escape / 1000, 'r--', lw=2, label='Escape velocity')

    ax4_twin = ax4.twinx()
    ax4_twin.plot(r_values / R_earth, T_orbit, 'b:', lw=2, label='Orbital period')

    ax4.set_xlabel('Distance from center (Earth radii)')
    ax4.set_ylabel('Velocity (km/s)', color='green')
    ax4_twin.set_ylabel('Orbital Period (hours)', color='blue')
    ax4.set_title('Orbital Parameters')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)

    # Mark geostationary orbit (about 6.6 Earth radii)
    r_geo = 42164e3  # meters
    ax4.axvline(x=r_geo/R_earth, color='purple', linestyle=':', alpha=0.7)
    ax4.annotate('Geostationary\norbit', xy=(r_geo/R_earth, 2), fontsize=8,
                ha='center', color='purple')

    plt.suptitle('Gravitational Field Around Earth\n'
                 f'(M = {M_earth:.2e} kg, R = {R_earth/1e6:.2f} Mm)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'gravitational_potential.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'gravitational_potential.png')}")


if __name__ == "__main__":
    main()
