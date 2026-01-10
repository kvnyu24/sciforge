"""
Experiment 236: London Penetration Depth

Demonstrates the London equations for superconductivity, showing how
magnetic fields are exponentially screened from the interior of a
superconductor with characteristic penetration depth lambda_L.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


# Physical constants
mu_0 = 4 * np.pi * 1e-7  # Vacuum permeability (H/m)
hbar = 1.055e-34         # Reduced Planck constant (J*s)
e = 1.602e-19            # Electron charge (C)
m_e = 9.109e-31          # Electron mass (kg)


def london_penetration_depth(n_s, m_star=2*m_e, q_star=2*e):
    """
    Calculate London penetration depth.

    lambda_L = sqrt(m* / (mu_0 * n_s * q*^2))

    For Cooper pairs: m* = 2*m_e, q* = 2*e

    Args:
        n_s: Superconducting carrier density (m^-3)
        m_star: Effective mass (default: 2*m_e for Cooper pairs)
        q_star: Effective charge (default: 2*e)

    Returns:
        lambda_L in meters
    """
    return np.sqrt(m_star / (mu_0 * n_s * q_star**2))


def magnetic_field_profile_slab(x, B_0, lambda_L, d):
    """
    Magnetic field inside superconducting slab.

    B(x) = B_0 * cosh(x/lambda_L) / cosh(d/(2*lambda_L))

    for -d/2 <= x <= d/2

    Args:
        x: Position (0 at center)
        B_0: External field
        lambda_L: Penetration depth
        d: Slab thickness

    Returns:
        B(x) magnetic field
    """
    return B_0 * np.cosh(x / lambda_L) / np.cosh(d / (2 * lambda_L))


def magnetic_field_profile_halfspace(x, B_0, lambda_L):
    """
    Magnetic field in semi-infinite superconductor (x > 0).

    B(x) = B_0 * exp(-x/lambda_L)

    Args:
        x: Distance from surface
        B_0: External field
        lambda_L: Penetration depth

    Returns:
        B(x) magnetic field
    """
    return B_0 * np.exp(-x / lambda_L)


def current_density_halfspace(x, B_0, lambda_L):
    """
    Supercurrent density in semi-infinite superconductor.

    J(x) = -(B_0 / (mu_0 * lambda_L)) * exp(-x/lambda_L)

    Args:
        x: Distance from surface
        B_0: External field
        lambda_L: Penetration depth

    Returns:
        J(x) current density (A/m^2)
    """
    return -(B_0 / (mu_0 * lambda_L)) * np.exp(-x / lambda_L)


def temperature_dependence(T, T_c, lambda_0):
    """
    Temperature dependence of penetration depth (two-fluid model).

    lambda(T) = lambda_0 / sqrt(1 - (T/T_c)^4)

    Args:
        T: Temperature
        T_c: Critical temperature
        lambda_0: Zero-temperature penetration depth

    Returns:
        lambda(T)
    """
    t = T / T_c
    return lambda_0 / np.sqrt(np.maximum(1 - t**4, 1e-10))


def main():
    """Main simulation and visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Magnetic field penetration into half-space
    ax1 = axes[0, 0]

    lambda_L = 50e-9  # 50 nm typical for elemental superconductors
    B_0 = 0.01  # 10 mT external field

    x = np.linspace(0, 500e-9, 500)  # 0 to 500 nm
    x_nm = x * 1e9

    B = magnetic_field_profile_halfspace(x, B_0, lambda_L)

    ax1.plot(x_nm, B / B_0, 'b-', lw=2)
    ax1.axhline(y=np.exp(-1), color='gray', linestyle='--', alpha=0.5,
               label=f'B/B_0 = 1/e at x = lambda_L')
    ax1.axvline(x=lambda_L*1e9, color='red', linestyle=':', alpha=0.7,
               label=f'lambda_L = {lambda_L*1e9:.0f} nm')

    ax1.set_xlabel('Distance from surface (nm)')
    ax1.set_ylabel('B / B_0')
    ax1.set_title('Meissner Effect: Magnetic Field Penetration')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 500)
    ax1.fill_between(x_nm, 0, B/B_0, alpha=0.3)

    # Add schematic of superconductor
    ax1.axvspan(-50, 0, alpha=0.3, color='gray', label='Vacuum')
    ax1.text(250, 0.5, 'Superconductor', fontsize=12, ha='center')

    # Plot 2: Supercurrent distribution
    ax2 = axes[0, 1]

    J = current_density_halfspace(x, B_0, lambda_L)
    J_max = np.abs(J).max()

    ax2.plot(x_nm, J / J_max, 'r-', lw=2)
    ax2.fill_between(x_nm, 0, J/J_max, alpha=0.3, color='red')

    ax2.axvline(x=lambda_L*1e9, color='gray', linestyle=':', alpha=0.7)

    ax2.set_xlabel('Distance from surface (nm)')
    ax2.set_ylabel('J / J_max')
    ax2.set_title('Screening Current Distribution')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 500)

    # Explain the physics
    ax2.text(0.95, 0.95, 'Currents flow to cancel\nthe interior field',
             transform=ax2.transAxes, fontsize=10, va='top', ha='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 3: Thin film penetration
    ax3 = axes[1, 0]

    thicknesses = [200e-9, 100e-9, 50e-9, 20e-9]  # Various film thicknesses
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(thicknesses)))

    for d, color in zip(thicknesses, colors):
        x_slab = np.linspace(-d/2, d/2, 200)
        B_slab = magnetic_field_profile_slab(x_slab, B_0, lambda_L, d)

        ax3.plot(x_slab * 1e9, B_slab / B_0, color=color, lw=2,
                label=f'd = {d*1e9:.0f} nm')

    ax3.axhline(y=1, color='gray', linestyle='--', alpha=0.3)
    ax3.set_xlabel('Position in film (nm)')
    ax3.set_ylabel('B / B_0')
    ax3.set_title(f'Thin Film: Field Penetration (lambda_L = {lambda_L*1e9:.0f} nm)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax3.text(0.5, 0.95, 'Thin films show incomplete screening',
             transform=ax3.transAxes, fontsize=10, va='top', ha='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 4: Temperature dependence
    ax4 = axes[1, 1]

    T_c = 9.0  # K (niobium)
    lambda_0 = 40e-9  # 40 nm

    T_range = np.linspace(0.01, 0.99, 100) * T_c

    lambda_T = temperature_dependence(T_range, T_c, lambda_0)

    ax4.plot(T_range / T_c, lambda_T / lambda_0, 'b-', lw=2)
    ax4.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
    ax4.axvline(x=1, color='red', linestyle='--', alpha=0.5, label='T_c')

    ax4.set_xlabel('T / T_c')
    ax4.set_ylabel('lambda(T) / lambda_0')
    ax4.set_title('Temperature Dependence of Penetration Depth')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(1, 10)

    # Add formula
    ax4.text(0.5, 0.95, r'$\lambda(T) = \frac{\lambda_0}{\sqrt{1-(T/T_c)^4}}$',
             transform=ax4.transAxes, fontsize=12, va='top', ha='center',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('London Penetration Depth and Meissner Effect\n'
                 r'London equation: $\nabla^2 \mathbf{B} = \mathbf{B}/\lambda_L^2$',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'london_penetration_depth.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'london_penetration_depth.png')}")

    # Additional figure: London penetration depths for various materials
    fig2, ax = plt.subplots(figsize=(10, 6))

    materials = {
        'Al': {'T_c': 1.2, 'lambda_0': 16},
        'Sn': {'T_c': 3.7, 'lambda_0': 34},
        'Pb': {'T_c': 7.2, 'lambda_0': 37},
        'Nb': {'T_c': 9.3, 'lambda_0': 39},
        'NbTi': {'T_c': 10, 'lambda_0': 300},
        'YBCO': {'T_c': 92, 'lambda_0': 140}
    }

    names = list(materials.keys())
    T_c_vals = [materials[n]['T_c'] for n in names]
    lambda_vals = [materials[n]['lambda_0'] for n in names]

    x_pos = range(len(names))
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(names)))

    bars = ax.bar(x_pos, lambda_vals, color=colors)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names)
    ax.set_ylabel('lambda_0 (nm)')
    ax.set_title('London Penetration Depth for Various Superconductors')
    ax.grid(True, alpha=0.3, axis='y')

    # Add T_c values as text
    for i, (bar, T_c) in enumerate(zip(bars, T_c_vals)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
               f'T_c = {T_c} K', ha='center', fontsize=9)

    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, 'london_materials.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'london_materials.png')}")


if __name__ == "__main__":
    main()
