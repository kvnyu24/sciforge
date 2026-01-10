"""
Experiment 269: Accretion Disk Temperature Profile

Demonstrates the temperature structure of geometrically thin,
optically thick accretion disks (Shakura-Sunyaev model).

Physical concepts:
- Viscous dissipation heats the disk
- T(r) ~ r^(-3/4) for standard thin disk
- Inner temperature depends on M and mdot
- Multi-temperature blackbody spectrum
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.plasma import AccretionDisk

# Physical constants
G = 6.674e-11
c = 2.998e8
h = 6.626e-34
k_B = 1.381e-23
sigma_SB = 5.670e-8
M_sun = 1.989e30
R_sun = 6.96e8


def schwarzschild_radius(M):
    """Schwarzschild radius Rs = 2GM/c^2."""
    return 2 * G * M / c**2


def isco_radius(M):
    """Innermost stable circular orbit (Schwarzschild): r_ISCO = 3*Rs."""
    return 3 * schwarzschild_radius(M)


def disk_temperature(r, M, mdot, r_in):
    """
    Standard thin disk temperature profile.

    T^4 = (3*G*M*mdot) / (8*pi*sigma*r^3) * (1 - sqrt(r_in/r))
    """
    T4 = (3 * G * M * mdot / (8 * np.pi * sigma_SB * r**3))
    f = np.where(r > r_in, 1 - np.sqrt(r_in / r), 0)
    return (T4 * f)**0.25


def disk_spectrum(nu, r_in, r_out, M, mdot, distance=10 * 3.086e16):
    """
    Multi-temperature blackbody spectrum from disk.

    F_nu = integral(2*pi*r*B_nu(T(r))*cos(i)*dr) / d^2
    """
    n_r = 100
    r = np.logspace(np.log10(r_in), np.log10(r_out), n_r)

    F_nu = np.zeros_like(nu)

    for i in range(len(r) - 1):
        ri = r[i]
        dr = r[i + 1] - ri
        T = disk_temperature(ri, M, mdot, r_in)

        # Planck function
        x = h * nu / (k_B * T + 1e-100)
        B_nu = 2 * h * nu**3 / c**2 / (np.exp(np.minimum(x, 500)) - 1 + 1e-100)

        # Face-on disk
        area = 2 * np.pi * ri * dr
        F_nu += B_nu * area / distance**2

    return F_nu


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Systems: stellar-mass BH, supermassive BH, white dwarf
    systems = {
        'Stellar BH (10 M$_\\odot$)': {
            'M': 10 * M_sun,
            'mdot': 1e-8 * M_sun / 3.156e7,  # 10^-8 M_sun/yr
            'r_scale': 1,  # In units of R_s
        },
        'SMBH (10$^8$ M$_\\odot$)': {
            'M': 1e8 * M_sun,
            'mdot': 1e-1 * M_sun / 3.156e7,  # 0.1 M_sun/yr
            'r_scale': 1,
        },
        'White Dwarf': {
            'M': 0.6 * M_sun,
            'mdot': 1e-9 * M_sun / 3.156e7,  # 10^-9 M_sun/yr
            'r_scale': 1e-2 * R_sun,  # WD radius scale
        },
    }

    # Plot 1: Temperature profiles
    ax1 = axes[0, 0]

    colors = ['blue', 'red', 'green']

    for (name, params), color in zip(systems.items(), colors):
        M = params['M']
        mdot = params['mdot']

        r_in = isco_radius(M)
        r_out = 1000 * r_in

        r = np.logspace(np.log10(r_in), np.log10(r_out), 200)
        T = disk_temperature(r, M, mdot, r_in)

        ax1.loglog(r / r_in, T, color=color, lw=2, label=name)

    # Show r^(-3/4) scaling
    r_norm = np.logspace(0.5, 3, 50)
    T_scaling = 1e7 * r_norm**(-0.75)
    ax1.loglog(r_norm, T_scaling, 'k--', lw=1.5, alpha=0.5, label='$T \\propto r^{-3/4}$')

    ax1.set_xlabel('Radius ($r / r_{ISCO}$)')
    ax1.set_ylabel('Temperature (K)')
    ax1.set_title('Accretion Disk Temperature Profiles')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_xlim(1, 1000)
    ax1.set_ylim(1e3, 1e8)

    # Plot 2: Disk structure visualization
    ax2 = axes[0, 1]

    M = 10 * M_sun
    mdot = 1e-8 * M_sun / 3.156e7
    disk = AccretionDisk(M, mdot)

    r_in = disk.inner_radius
    r_out = 100 * r_in

    # Create disk image
    x = np.linspace(-r_out, r_out, 200)
    y = np.linspace(-r_out, r_out, 200)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)

    # Temperature map
    T_map = np.where(R >= r_in, disk_temperature(R + 1e-10, M, mdot, r_in), 0)

    # Log scale for visualization
    T_log = np.log10(T_map + 1)

    im = ax2.imshow(T_log, extent=[-100, 100, -100, 100],
                     cmap='inferno', origin='lower')
    plt.colorbar(im, ax=ax2, label='log$_{10}$(T/K)')

    # Draw ISCO
    theta = np.linspace(0, 2 * np.pi, 100)
    ax2.plot(np.cos(theta), np.sin(theta), 'w--', lw=1, label='ISCO')

    ax2.set_xlabel('$r / r_{ISCO}$')
    ax2.set_ylabel('$r / r_{ISCO}$')
    ax2.set_title('Disk Temperature Map (10 M$_\\odot$ BH)')
    ax2.set_aspect('equal')

    # Plot 3: Multi-temperature spectrum
    ax3 = axes[1, 0]

    # Frequency range (X-ray to optical for stellar BH)
    nu = np.logspace(14, 19, 200)  # Hz
    wavelength = c / nu

    for (name, params), color in zip(list(systems.items())[:2], colors[:2]):
        M = params['M']
        mdot = params['mdot']
        r_in = isco_radius(M)
        r_out = 1000 * r_in

        F_nu = disk_spectrum(nu, r_in, r_out, M, mdot)

        # nu * F_nu for SED
        ax3.loglog(nu, nu * F_nu, color=color, lw=2, label=name)

    # Mark key energies
    E_keV = [0.1, 1, 10, 100]
    for E in E_keV:
        nu_E = E * 1e3 * 1.602e-19 / h
        if 1e14 < nu_E < 1e19:
            ax3.axvline(x=nu_E, color='gray', linestyle=':', alpha=0.5)
            ax3.text(nu_E, 1e-20, f'{E} keV', fontsize=8, rotation=90)

    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('$\\nu F_\\nu$ (arbitrary units)')
    ax3.set_title('Accretion Disk Spectra')
    ax3.legend()
    ax3.grid(True, alpha=0.3, which='both')
    ax3.set_xlim(1e14, 1e19)

    # Plot 4: Luminosity and Eddington ratio
    ax4 = axes[1, 1]

    # Mass range
    M_range = np.logspace(0, 10, 100) * M_sun

    # Eddington luminosity
    kappa = 0.4e-3  # m^2/kg (electron scattering)
    L_Edd = 4 * np.pi * G * M_range * c / kappa

    ax4.loglog(M_range / M_sun, L_Edd, 'b-', lw=2, label='Eddington luminosity')

    # Disk luminosity at different Eddington ratios
    eddington_ratios = [0.01, 0.1, 1.0]
    for ratio in eddington_ratios:
        L_disk = ratio * L_Edd
        ax4.loglog(M_range / M_sun, L_disk, '--', lw=1.5,
                   label=f'$\\dot{{m}}$ = {ratio} $\\dot{{m}}_{{Edd}}$')

    # Mark specific sources
    sources = {
        'Cyg X-1': (15, 0.02 * 4 * np.pi * G * 15 * M_sun * c / kappa),
        'M87*': (6.5e9, 1e-5 * 4 * np.pi * G * 6.5e9 * M_sun * c / kappa),
        'Sgr A*': (4e6, 1e-8 * 4 * np.pi * G * 4e6 * M_sun * c / kappa),
    }

    for name, (M, L) in sources.items():
        ax4.plot(M, L, 'r*', markersize=12)
        ax4.annotate(name, (M, L), xytext=(5, 5),
                     textcoords='offset points', fontsize=9)

    ax4.set_xlabel('Black Hole Mass (M$_\\odot$)')
    ax4.set_ylabel('Luminosity (W)')
    ax4.set_title('Accretion Luminosity and Eddington Limit')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, which='both')
    ax4.set_xlim(1, 1e10)
    ax4.set_ylim(1e28, 1e42)

    plt.suptitle('Experiment 269: Accretion Disk Temperature Structure\n'
                 'Shakura-Sunyaev $\\alpha$-disk model',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'accretion_disk_temperature.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'accretion_disk_temperature.png')}")


if __name__ == "__main__":
    main()
