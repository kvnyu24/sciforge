"""
Example demonstrating electric field of a dipole.

This example shows the electric field pattern created by an electric
dipole (two opposite charges), including field lines and equipotential lines.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.fields import ElectricField


def calculate_dipole_field(x, y, q, d, k=8.99e9):
    """
    Calculate electric field from a dipole at given points.

    Args:
        x, y: Coordinate arrays
        q: Charge magnitude (C)
        d: Separation distance (m) - charges at (-d/2, 0) and (d/2, 0)
        k: Coulomb constant

    Returns:
        Ex, Ey: Electric field components
        V: Electric potential
    """
    # Positive charge at (d/2, 0), negative at (-d/2, 0)
    x_pos, y_pos = d/2, 0
    x_neg, y_neg = -d/2, 0

    # Distances from each charge
    r_pos = np.sqrt((x - x_pos)**2 + (y - y_pos)**2)
    r_neg = np.sqrt((x - x_neg)**2 + (y - y_neg)**2)

    # Avoid singularities
    r_pos = np.maximum(r_pos, 0.01)
    r_neg = np.maximum(r_neg, 0.01)

    # Field from positive charge
    Ex_pos = k * q * (x - x_pos) / r_pos**3
    Ey_pos = k * q * (y - y_pos) / r_pos**3

    # Field from negative charge
    Ex_neg = k * (-q) * (x - x_neg) / r_neg**3
    Ey_neg = k * (-q) * (y - y_neg) / r_neg**3

    # Total field
    Ex = Ex_pos + Ex_neg
    Ey = Ey_pos + Ey_neg

    # Potential
    V = k * q / r_pos + k * (-q) / r_neg

    return Ex, Ey, V


def main():
    # Physical parameters
    q = 1e-9  # 1 nC charge
    d = 2.0   # 2 m separation

    # Create coordinate grid
    x = np.linspace(-5, 5, 200)
    y = np.linspace(-5, 5, 200)
    X, Y = np.meshgrid(x, y)

    # Calculate field and potential
    Ex, Ey, V = calculate_dipole_field(X, Y, q, d)

    # Field magnitude for coloring
    E_mag = np.sqrt(Ex**2 + Ey**2)
    E_mag_log = np.log10(E_mag + 1)  # Log scale for better visualization

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Field vectors with streamlines
    ax1 = axes[0, 0]
    # Streamlines
    ax1.streamplot(X, Y, Ex, Ey, color=E_mag_log, cmap='viridis',
                   linewidth=1, density=2, arrowsize=1)
    # Mark charges
    ax1.plot(d/2, 0, 'ro', markersize=15, label='+q')
    ax1.plot(-d/2, 0, 'bo', markersize=15, label='-q')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_title('Electric Field Lines')
    ax1.legend()
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Equipotential lines
    ax2 = axes[0, 1]
    V_clipped = np.clip(V, -1e10, 1e10)
    levels = np.linspace(-5e9, 5e9, 21)
    contour = ax2.contour(X, Y, V_clipped, levels=levels, cmap='RdBu_r')
    ax2.clabel(contour, inline=True, fontsize=8, fmt='%.1e')
    ax2.plot(d/2, 0, 'ro', markersize=15, label='+q')
    ax2.plot(-d/2, 0, 'bo', markersize=15, label='-q')
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')
    ax2.set_title('Equipotential Lines')
    ax2.legend()
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Field magnitude (log scale)
    ax3 = axes[1, 0]
    im = ax3.imshow(E_mag_log, extent=[x.min(), x.max(), y.min(), y.max()],
                    cmap='hot', origin='lower')
    ax3.plot(d/2, 0, 'co', markersize=10)
    ax3.plot(-d/2, 0, 'co', markersize=10)
    ax3.set_xlabel('x (m)')
    ax3.set_ylabel('y (m)')
    ax3.set_title('Field Magnitude (log scale)')
    plt.colorbar(im, ax=ax3, label='log₁₀(|E| + 1)')

    # Plot 4: Field along axes
    ax4 = axes[1, 1]

    # Field along x-axis (y=0)
    x_axis = np.linspace(-5, 5, 500)
    Ex_axis, Ey_axis, _ = calculate_dipole_field(x_axis, np.zeros_like(x_axis), q, d)
    E_x_axis = Ex_axis  # Only x-component along axis

    # Mask near charges
    mask = (np.abs(x_axis - d/2) > 0.2) & (np.abs(x_axis + d/2) > 0.2)

    ax4.plot(x_axis[mask], E_x_axis[mask] / 1e9, 'b-', lw=2, label='Along x-axis (y=0)')

    # Field along perpendicular bisector (x=0)
    y_perp = np.linspace(-5, 5, 500)
    Ex_perp, Ey_perp, _ = calculate_dipole_field(np.zeros_like(y_perp), y_perp, q, d)
    E_perp = -Ex_perp  # Field points in -x direction on bisector

    ax4.plot(y_perp, E_perp / 1e9, 'r--', lw=2, label='Along y-axis (x=0)')

    ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax4.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    ax4.set_xlabel('Position (m)')
    ax4.set_ylabel('Electric Field (GV/m)')
    ax4.set_title('Field Profile Along Principal Axes')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-50, 50)

    plt.suptitle(f'Electric Dipole Field (q = {q*1e9:.1f} nC, d = {d} m)\n'
                 f'Dipole moment p = qd = {q*d*1e9:.1f} nC⋅m',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'electric_dipole_field.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'electric_dipole_field.png')}")


if __name__ == "__main__":
    main()
