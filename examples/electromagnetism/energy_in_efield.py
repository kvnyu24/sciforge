"""
Experiment 85: Energy in E-field.

This example demonstrates the energy stored in electric fields,
including energy density u = (1/2) * epsilon_0 * E^2 and total
energy calculations for various charge configurations.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Constants
EPSILON_0 = 8.854e-12  # Permittivity of free space (F/m)
K = 1 / (4 * np.pi * EPSILON_0)  # Coulomb constant


def point_charge_field(x, y, z, q, x0=0, y0=0, z0=0):
    """Electric field from a point charge."""
    rx = x - x0
    ry = y - y0
    rz = z - z0
    r = np.sqrt(rx**2 + ry**2 + rz**2)
    r = np.maximum(r, 1e-10)

    factor = K * q / r**3
    return factor * rx, factor * ry, factor * rz


def energy_density(Ex, Ey, Ez):
    """
    Calculate electric field energy density.

    u = (1/2) * epsilon_0 * E^2
    """
    E_squared = Ex**2 + Ey**2 + Ez**2
    return 0.5 * EPSILON_0 * E_squared


def capacitor_field_energy(V, d, A):
    """
    Calculate energy stored in a parallel plate capacitor.

    U = (1/2) * epsilon_0 * E^2 * Volume = (1/2) * C * V^2

    Args:
        V: Voltage across plates
        d: Plate separation
        A: Plate area

    Returns:
        U: Total energy
        C: Capacitance
    """
    E = V / d
    u = 0.5 * EPSILON_0 * E**2
    U = u * A * d
    C = EPSILON_0 * A / d
    return U, C, u


def spherical_shell_energy(Q, R_inner, R_outer):
    """
    Calculate energy stored in the electric field between two spherical shells.

    U = (Q^2 / 8*pi*epsilon_0) * (1/R_inner - 1/R_outer)
    """
    return K * Q**2 / 2 * (1/R_inner - 1/R_outer)


def main():
    fig = plt.figure(figsize=(16, 12))

    # Plot 1: Energy density around a point charge
    ax1 = fig.add_subplot(2, 2, 1)

    q = 1e-9  # 1 nC
    n = 200
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    Ex, Ey, Ez = point_charge_field(X, Y, Z, q)
    u = energy_density(Ex, Ey, Ez)

    # Log scale for energy density
    u_log = np.log10(u + 1e-20)

    im1 = ax1.imshow(u_log, extent=[x.min(), x.max(), y.min(), y.max()],
                     origin='lower', cmap='hot', aspect='equal')
    plt.colorbar(im1, ax=ax1, label='log10(u) (J/m^3)')

    ax1.plot(0, 0, 'co', markersize=10, label=f'q = {q*1e9:.0f} nC')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_title('Energy Density Around Point Charge\nu = (1/2) * epsilon_0 * E^2')
    ax1.legend()

    # Plot 2: Radial energy density profile
    ax2 = fig.add_subplot(2, 2, 2)

    r = np.linspace(0.05, 2, 100)
    E_radial = K * q / r**2
    u_radial = 0.5 * EPSILON_0 * E_radial**2

    # Energy per unit shell (dU/dr = 4*pi*r^2 * u)
    dU_dr = 4 * np.pi * r**2 * u_radial

    ax2.loglog(r, u_radial, 'b-', lw=2, label='Energy density u(r)')
    ax2.loglog(r, dU_dr, 'r-', lw=2, label='Energy per shell dU/dr')

    # Reference slopes
    ax2.loglog(r, 1e-8 * r**(-4), 'b:', lw=1, alpha=0.5, label='r^-4 (u slope)')
    ax2.loglog(r, 1e-8 * r**(-2), 'r:', lw=1, alpha=0.5, label='r^-2 (dU/dr slope)')

    ax2.set_xlabel('Distance r (m)')
    ax2.set_ylabel('Energy density / Energy per shell')
    ax2.set_title('Radial Energy Distribution')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, which='both')

    # Plot 3: Capacitor energy storage
    ax3 = fig.add_subplot(2, 2, 3)

    # Varying voltage at fixed geometry
    V_range = np.linspace(1, 100, 100)
    d = 0.001  # 1 mm plate separation
    A = 0.01   # 100 cm^2 plate area

    energies = []
    for V in V_range:
        U, C, u = capacitor_field_energy(V, d, A)
        energies.append(U)

    C_val = EPSILON_0 * A / d
    ax3.plot(V_range, np.array(energies) * 1e6, 'b-', lw=2,
             label=f'U = (1/2)CV^2\nC = {C_val*1e12:.2f} pF')

    ax3.set_xlabel('Voltage (V)')
    ax3.set_ylabel('Energy (uJ)')
    ax3.set_title('Capacitor Energy Storage')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Add quadratic fit verification
    U_theory = 0.5 * C_val * V_range**2
    ax3.plot(V_range, U_theory * 1e6, 'r--', lw=2, alpha=0.7, label='Theory')

    # Plot 4: Energy in different configurations
    ax4 = fig.add_subplot(2, 2, 4)

    # Compare energy in different charge configurations
    configs = []

    # Single charge (energy from R_min to R_max)
    R_min = 0.01  # 1 cm inner cutoff
    R_max = 1.0   # 1 m outer limit
    U_single = K * q**2 / 2 * (1/R_min - 1/R_max)
    configs.append(('Single charge\n(0.01-1m)', U_single))

    # Two opposite charges (dipole)
    d_sep = 0.1  # 10 cm separation
    # Energy of dipole (self-energy minus interaction)
    U_dipole = -K * q**2 / d_sep  # Interaction energy (negative = bound)
    configs.append(('Dipole\n(d=10cm)', U_dipole))

    # Parallel plate capacitor
    V_cap = 100
    U_cap, _, _ = capacitor_field_energy(V_cap, d, A)
    configs.append(('Capacitor\n(100V)', U_cap))

    # Spherical capacitor
    R_inner = 0.05
    R_outer = 0.1
    Q_sphere = 1e-9
    U_sphere = spherical_shell_energy(Q_sphere, R_inner, R_outer)
    configs.append(('Spherical cap\n(5-10cm)', U_sphere))

    names = [c[0] for c in configs]
    values = [c[1] for c in configs]

    colors = ['blue' if v >= 0 else 'red' for v in values]
    bars = ax4.bar(names, np.abs(values) * 1e6, color=colors, alpha=0.7)

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        sign = '+' if val >= 0 else '-'
        ax4.annotate(f'{sign}{np.abs(val)*1e6:.2e} uJ',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9)

    ax4.set_ylabel('|Energy| (uJ)')
    ax4.set_title('Energy in Different Configurations\n(blue = positive, red = negative/bound)')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3, axis='y')

    # Add physics summary
    fig.text(0.5, 0.02,
             r'Electric field energy: $U = \int \frac{1}{2}\epsilon_0 E^2 \, dV$ = '
             r'$\frac{1}{2}CV^2$ (capacitor) = $\frac{1}{2}QV$ (general)',
             ha='center', fontsize=11, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.suptitle('Energy Stored in Electric Fields', fontsize=14, y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)

    # Save plot
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'energy_in_efield.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
