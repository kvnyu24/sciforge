"""
Experiment 81: Dipole field and multipole scaling.

This example demonstrates the electric dipole field pattern and
verifies the 1/r^3 falloff characteristic of dipole fields,
along with comparison to monopole and quadrupole scaling.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt


def monopole_field(r, q, k=8.99e9):
    """Monopole field: E ~ 1/r^2"""
    return k * q / r**2


def dipole_field_axial(r, p, k=8.99e9):
    """
    Dipole field along axis (z-axis).
    E_z = 2kp / r^3
    """
    return 2 * k * p / r**3


def dipole_field_perpendicular(r, p, k=8.99e9):
    """
    Dipole field perpendicular to axis (in x-y plane at z=0).
    E_x = -kp / r^3
    """
    return k * p / r**3


def quadrupole_field(r, Q, k=8.99e9):
    """Quadrupole field: E ~ 1/r^4"""
    return k * Q / r**4


def calculate_dipole_field_2d(x, y, q, d, k=8.99e9):
    """
    Calculate 2D dipole field (charges at +/-d/2 on x-axis).

    Args:
        x, y: Coordinate arrays
        q: Charge magnitude (C)
        d: Separation distance (m)
        k: Coulomb constant

    Returns:
        Ex, Ey: Field components
        V: Potential
    """
    # Positive charge at (d/2, 0), negative at (-d/2, 0)
    r_pos = np.sqrt((x - d/2)**2 + y**2)
    r_neg = np.sqrt((x + d/2)**2 + y**2)

    # Avoid singularities
    r_pos = np.maximum(r_pos, 0.01)
    r_neg = np.maximum(r_neg, 0.01)

    # Field from each charge
    Ex = k * q * ((x - d/2) / r_pos**3 - (x + d/2) / r_neg**3)
    Ey = k * q * (y / r_pos**3 - y / r_neg**3)

    # Potential
    V = k * q * (1 / r_pos - 1 / r_neg)

    return Ex, Ey, V


def main():
    # Parameters
    q = 1e-9       # Charge magnitude (1 nC)
    d = 0.2        # Dipole separation (20 cm)
    p = q * d      # Dipole moment
    k = 8.99e9     # Coulomb constant

    # Quadrupole moment (for comparison)
    Q = q * d**2

    fig = plt.figure(figsize=(16, 12))

    # Plot 1: 2D dipole field with streamlines
    ax1 = fig.add_subplot(2, 2, 1)

    x = np.linspace(-1, 1, 150)
    y = np.linspace(-1, 1, 150)
    X, Y = np.meshgrid(x, y)

    Ex, Ey, V = calculate_dipole_field_2d(X, Y, q, d)
    E_mag = np.sqrt(Ex**2 + Ey**2)
    E_mag_log = np.log10(E_mag + 1)

    ax1.streamplot(X, Y, Ex, Ey, color=E_mag_log, cmap='viridis',
                   density=2, linewidth=1, arrowsize=1)
    ax1.plot(d/2, 0, 'ro', markersize=12, label='+q')
    ax1.plot(-d/2, 0, 'bo', markersize=12, label='-q')

    # Draw dipole moment vector
    ax1.arrow(0, 0, d/3, 0, head_width=0.05, head_length=0.03,
              fc='green', ec='green', lw=2)
    ax1.text(0.02, 0.08, r'$\vec{p}$', fontsize=14, color='green')

    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_title(f'Electric Dipole Field\n(d = {d*100:.0f} cm, p = {p*1e9:.2f} nC*m)')
    ax1.legend()
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Equipotential lines
    ax2 = fig.add_subplot(2, 2, 2)

    V_clipped = np.clip(V, -1e10, 1e10)
    levels = np.linspace(-2e9, 2e9, 21)
    contour = ax2.contour(X, Y, V_clipped, levels=levels, cmap='RdBu_r')
    ax2.clabel(contour, inline=True, fontsize=7, fmt='%.1e')
    ax2.plot(d/2, 0, 'ro', markersize=12, label='+q')
    ax2.plot(-d/2, 0, 'bo', markersize=12, label='-q')
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')
    ax2.set_title('Equipotential Lines')
    ax2.legend()
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Multipole scaling comparison
    ax3 = fig.add_subplot(2, 2, 3)

    r = np.logspace(-0.5, 1, 100)  # 0.3 to 10 meters

    # Calculate fields
    E_mono = monopole_field(r, q)
    E_dipole_ax = dipole_field_axial(r, p)
    E_dipole_perp = dipole_field_perpendicular(r, p)
    E_quad = quadrupole_field(r, Q)

    ax3.loglog(r, E_mono, 'b-', lw=2, label=r'Monopole: $E \sim 1/r^2$')
    ax3.loglog(r, E_dipole_ax, 'r-', lw=2, label=r'Dipole (axial): $E \sim 1/r^3$')
    ax3.loglog(r, E_dipole_perp, 'r--', lw=2, label=r'Dipole (perp): $E \sim 1/r^3$')
    ax3.loglog(r, E_quad, 'g-', lw=2, label=r'Quadrupole: $E \sim 1/r^4$')

    # Reference slopes
    ax3.loglog(r, 1e10 * r**(-2), 'b:', lw=1, alpha=0.5)
    ax3.loglog(r, 1e9 * r**(-3), 'r:', lw=1, alpha=0.5)
    ax3.loglog(r, 1e8 * r**(-4), 'g:', lw=1, alpha=0.5)

    ax3.set_xlabel('Distance r (m)')
    ax3.set_ylabel('Electric Field |E| (V/m)')
    ax3.set_title('Multipole Field Scaling')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, which='both')
    ax3.set_ylim(1e3, 1e12)

    # Plot 4: Numerical vs analytical dipole field
    ax4 = fig.add_subplot(2, 2, 4)

    # Calculate numerical field along axis
    r_test = np.linspace(0.3, 3, 50)
    _, _, V_axis = calculate_dipole_field_2d(r_test, np.zeros_like(r_test), q, d)

    # Numerical E from gradient of V
    E_numerical = -np.gradient(V_axis, r_test)

    # Analytical dipole field (far field approximation)
    E_analytical = dipole_field_axial(r_test, p)

    # Exact field (two point charges)
    r_pos = r_test - d/2
    r_neg = r_test + d/2
    E_exact = k * q * (1/r_pos**2 - 1/r_neg**2)

    ax4.semilogy(r_test, np.abs(E_exact), 'b-', lw=2, label='Exact (2 charges)')
    ax4.semilogy(r_test, E_analytical, 'r--', lw=2, label='Dipole approx.')
    ax4.semilogy(r_test, np.abs(E_numerical), 'g:', lw=2, label='Numerical (from V)')

    # Show crossover region
    ax4.axvline(x=d*3, color='gray', linestyle=':', alpha=0.7)
    ax4.text(d*3.2, 1e9, 'r = 3d\n(dipole valid)', fontsize=9, alpha=0.7)

    ax4.set_xlabel('Distance r (m)')
    ax4.set_ylabel('Electric Field |E| (V/m)')
    ax4.set_title('Dipole Approximation Validity')
    ax4.legend()
    ax4.grid(True, alpha=0.3, which='both')

    # Add error analysis
    error = np.abs(E_exact - E_analytical) / E_exact * 100
    ax4_twin = ax4.twinx()
    ax4_twin.plot(r_test, error, 'm-.', lw=1, alpha=0.7, label='Error %')
    ax4_twin.set_ylabel('Error (%)', color='m')
    ax4_twin.tick_params(axis='y', labelcolor='m')
    ax4_twin.set_ylim(0, 100)

    plt.suptitle('Dipole Field and Multipole Scaling\n'
                 r'Dipole moment $p = qd$, Far-field: $E \propto 1/r^3$',
                 fontsize=14, y=1.02)

    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'dipole_multipole_scaling.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
