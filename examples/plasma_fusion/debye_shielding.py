"""
Experiment 256: Debye Shielding Profile

Demonstrates the Debye screening of electric potential in a plasma.
The Debye length characterizes how quickly the potential from a
point charge is screened by the plasma.

phi(r) = (q / 4*pi*epsilon_0*r) * exp(-r/lambda_D)

Physical concepts:
- Debye length scales as sqrt(T/n)
- Charges are screened over distances > lambda_D
- Number of particles in Debye sphere determines plasma behavior
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.plasma import DebyeLength

# Physical constants
epsilon_0 = 8.854e-12
e = 1.602e-19
k_B = 1.381e-23


def coulomb_potential(r, q=e):
    """Unscreened Coulomb potential."""
    return q / (4 * np.pi * epsilon_0 * r)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plasma parameters
    temperatures = [1e4, 1e5, 1e6]  # K (10 eV, 100 eV, 1 keV)
    density = 1e20  # m^-3 (typical fusion plasma)

    # Plot 1: Screened potential for different temperatures
    ax1 = axes[0, 0]

    r = np.linspace(1e-9, 5e-6, 500)

    # Unscreened Coulomb
    phi_coulomb = coulomb_potential(r)
    ax1.semilogy(r * 1e6, phi_coulomb, 'k--', lw=2, label='Coulomb (unscreened)')

    colors = ['blue', 'green', 'red']
    for T, color in zip(temperatures, colors):
        debye = DebyeLength(T, density)
        lambda_D = debye.length
        phi = debye.screening_potential(r)

        T_eV = T * k_B / e
        ax1.semilogy(r * 1e6, phi, color=color, lw=2,
                     label=f'T = {T_eV:.0f} eV, $\\lambda_D$ = {lambda_D*1e6:.2f} $\\mu$m')

        # Mark Debye length
        ax1.axvline(x=lambda_D * 1e6, color=color, linestyle=':', alpha=0.5)

    ax1.set_xlabel('Distance r ($\\mu$m)')
    ax1.set_ylabel('Potential (V)')
    ax1.set_title(f'Debye Screened Potential (n = {density:.0e} m$^{{-3}}$)')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 5)
    ax1.set_ylim(1e-4, 1e4)

    # Plot 2: Screening ratio vs distance
    ax2 = axes[0, 1]

    T = 1e5  # 100 eV
    debye = DebyeLength(T, density)
    lambda_D = debye.length

    r = np.linspace(0.1 * lambda_D, 10 * lambda_D, 200)
    r_normalized = r / lambda_D

    screening_ratio = np.exp(-r / lambda_D)

    ax2.plot(r_normalized, screening_ratio, 'b-', lw=2)
    ax2.fill_between(r_normalized, screening_ratio, alpha=0.3)

    # Mark key points
    ax2.axvline(x=1, color='red', linestyle='--', label='$r = \\lambda_D$')
    ax2.axhline(y=np.exp(-1), color='gray', linestyle=':', alpha=0.7)
    ax2.plot(1, np.exp(-1), 'ro', markersize=10)
    ax2.annotate(f'$e^{{-1}}$ = {np.exp(-1):.3f}', xy=(1, np.exp(-1)),
                 xytext=(2, 0.5), fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='gray'))

    ax2.set_xlabel('Distance $r/\\lambda_D$')
    ax2.set_ylabel('Screening Factor $e^{-r/\\lambda_D}$')
    ax2.set_title('Debye Screening vs Distance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 1.1)

    # Plot 3: Debye length vs temperature and density
    ax3 = axes[1, 0]

    T_range = np.logspace(3, 8, 100)  # 1000 K to 100 MK
    densities = [1e18, 1e19, 1e20, 1e21, 1e22]  # m^-3

    for n in densities:
        lambda_D = np.sqrt(epsilon_0 * k_B * T_range / (n * e**2))
        ax3.loglog(T_range * k_B / e, lambda_D * 1e6, lw=2,
                   label=f'n = {n:.0e} m$^{{-3}}$')

    ax3.set_xlabel('Temperature (eV)')
    ax3.set_ylabel('Debye Length ($\\mu$m)')
    ax3.set_title('Debye Length: $\\lambda_D = \\sqrt{\\epsilon_0 k_B T / n e^2}$')
    ax3.legend()
    ax3.grid(True, alpha=0.3, which='both')

    # Shade typical fusion conditions
    ax3.axvspan(1e3, 1e5, alpha=0.1, color='red', label='Fusion')
    ax3.text(3e3, 1e-1, 'Fusion\nPlasmas', fontsize=9, color='red')

    # Plot 4: Electric field with screening
    ax4 = axes[1, 1]

    T = 1e5  # 100 eV
    debye = DebyeLength(T, density)
    lambda_D = debye.length

    r = np.linspace(0.1 * lambda_D, 5 * lambda_D, 200)

    # Coulomb field
    E_coulomb = e / (4 * np.pi * epsilon_0 * r**2)

    # Screened field
    E_screened = debye.screening_field(r)

    ax4.semilogy(r / lambda_D, E_coulomb, 'k--', lw=2, label='Coulomb $E \\propto 1/r^2$')
    ax4.semilogy(r / lambda_D, E_screened, 'b-', lw=2, label='Screened')
    ax4.fill_between(r / lambda_D, E_screened, E_coulomb, alpha=0.2, color='blue')

    ax4.set_xlabel('Distance $r/\\lambda_D$')
    ax4.set_ylabel('Electric Field (V/m)')
    ax4.set_title('Electric Field Screening')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 5)

    # Add text box with Debye sphere info
    N_D = debye.sphere_count
    textstr = f'Debye sphere particles:\n$N_D$ = {N_D:.2e}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax4.text(0.95, 0.95, textstr, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.suptitle('Experiment 256: Debye Shielding in Plasmas\n'
                 '$\\phi(r) = \\frac{q}{4\\pi\\epsilon_0 r} e^{-r/\\lambda_D}$',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'debye_shielding.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'debye_shielding.png')}")


if __name__ == "__main__":
    main()
