"""
Experiment 270: Eddington Luminosity

Demonstrates the Eddington luminosity limit - the maximum luminosity
at which radiation pressure balances gravity.

Physical concepts:
- L_Edd = 4*pi*G*M*c / kappa
- Above L_Edd, radiation drives mass outflow
- Limits accretion rate onto compact objects
- Super-Eddington sources require special geometry
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt

# Physical constants
G = 6.674e-11
c = 2.998e8
M_sun = 1.989e30
L_sun = 3.828e26
sigma_T = 6.652e-29  # Thomson cross-section (m^2)
m_p = 1.673e-27


def eddington_luminosity(M, kappa=None):
    """
    Eddington luminosity.

    L_Edd = 4*pi*G*M*c / kappa

    Args:
        M: Mass (kg)
        kappa: Opacity (m^2/kg). Default: electron scattering

    Returns:
        Eddington luminosity (W)
    """
    if kappa is None:
        kappa = sigma_T / m_p  # Electron scattering opacity
    return 4 * np.pi * G * M * c / kappa


def eddington_mdot(M, eta=0.1, kappa=None):
    """
    Eddington accretion rate.

    mdot_Edd = L_Edd / (eta * c^2)

    Args:
        M: Mass (kg)
        eta: Radiative efficiency (default 0.1)
        kappa: Opacity

    Returns:
        Eddington accretion rate (kg/s)
    """
    L_Edd = eddington_luminosity(M, kappa)
    return L_Edd / (eta * c**2)


def radiation_pressure_force(L, r, kappa=None):
    """
    Radiation pressure force on unit mass.

    f_rad = kappa * L / (4*pi*r^2*c)

    Args:
        L: Luminosity (W)
        r: Distance (m)
        kappa: Opacity

    Returns:
        Force per unit mass (m/s^2)
    """
    if kappa is None:
        kappa = sigma_T / m_p
    return kappa * L / (4 * np.pi * r**2 * c)


def gravitational_force(M, r):
    """Gravitational force per unit mass."""
    return G * M / r**2


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Eddington luminosity vs mass
    ax1 = axes[0, 0]

    M = np.logspace(-1, 10, 100) * M_sun
    L_Edd = eddington_luminosity(M)

    ax1.loglog(M / M_sun, L_Edd, 'b-', lw=2, label='$L_{Edd}$ (electron scattering)')

    # Different opacities
    kappa_dust = 10 * sigma_T / m_p  # Higher opacity with dust
    L_Edd_dust = eddington_luminosity(M, kappa_dust)
    ax1.loglog(M / M_sun, L_Edd_dust, 'r--', lw=2, label='$L_{Edd}$ (with dust)')

    # Mark specific objects
    objects = {
        'Sun': (1, L_sun),
        'O star (60 M$_\\odot$)': (60, 8e5 * L_sun),
        'ULX': (10, 1e40),
        'AGN (10$^8$ M$_\\odot$)': (1e8, 1e39),
        'Sgr A*': (4e6, 1e33),  # Sub-Eddington
    }

    for name, (M_obj, L_obj) in objects.items():
        L_Edd_obj = eddington_luminosity(M_obj * M_sun)
        ax1.plot(M_obj, L_obj, 'k*', markersize=12)
        ax1.annotate(name, (M_obj, L_obj), xytext=(5, 5),
                     textcoords='offset points', fontsize=9)

        # Show Eddington ratio
        ratio = L_obj / L_Edd_obj
        if ratio > 1:
            ax1.annotate(f'({ratio:.0f}$\\times L_{{Edd}}$)', (M_obj, L_obj),
                         xytext=(5, -10), textcoords='offset points', fontsize=8, color='red')

    ax1.set_xlabel('Mass (M$_\\odot$)')
    ax1.set_ylabel('Luminosity (W)')
    ax1.set_title('Eddington Luminosity: $L_{Edd} = 4\\pi GMc/\\kappa$')
    ax1.legend()
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_xlim(0.1, 1e10)
    ax1.set_ylim(1e24, 1e42)

    # Shade super-Eddington region
    ax1.fill_between(M / M_sun, L_Edd, 1e42, alpha=0.1, color='red')
    ax1.text(1e5, 3e40, 'Super-Eddington', fontsize=10, color='red', ha='center')

    # Plot 2: Force balance
    ax2 = axes[0, 1]

    M_bh = 10 * M_sun
    L_Edd_bh = eddington_luminosity(M_bh)

    r = np.logspace(6, 12, 200)  # meters
    Rs = 2 * G * M_bh / c**2

    f_grav = gravitational_force(M_bh, r)

    L_ratios = [0.1, 0.5, 1.0, 2.0]
    colors = ['blue', 'green', 'orange', 'red']

    for L_ratio, color in zip(L_ratios, colors):
        L = L_ratio * L_Edd_bh
        f_rad = radiation_pressure_force(L, r)
        ax2.loglog(r / Rs, f_rad, color=color, lw=2,
                   label=f'$L = {L_ratio} L_{{Edd}}$')

    ax2.loglog(r / Rs, f_grav, 'k--', lw=2, label='Gravity')

    ax2.set_xlabel('Radius ($r / R_s$)')
    ax2.set_ylabel('Force per unit mass (m/s$^2$)')
    ax2.set_title('Radiation vs Gravitational Force')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_xlim(1, 1e6)

    # Mark balance point
    ax2.axhline(y=gravitational_force(M_bh, 100 * Rs), color='gray', linestyle=':', alpha=0.5)

    # Plot 3: Eddington ratio for different sources
    ax3 = axes[1, 0]

    source_data = {
        'Stellar BH XRBs': {'mass': 10, 'L_range': (0.01, 2)},
        'ULXs': {'mass': 10, 'L_range': (1, 100)},
        'Seyfert galaxies': {'mass': 1e7, 'L_range': (0.01, 0.3)},
        'Quasars': {'mass': 1e9, 'L_range': (0.1, 1)},
        'Blazars': {'mass': 1e9, 'L_range': (0.01, 10)},
    }

    y_pos = np.arange(len(source_data))

    for i, (name, data) in enumerate(source_data.items()):
        L_range = data['L_range']
        ax3.barh(i, L_range[1] - L_range[0], left=L_range[0], height=0.6, alpha=0.7)
        ax3.plot([L_range[0], L_range[1]], [i, i], 'k-', lw=2)

    ax3.axvline(x=1, color='red', linestyle='--', lw=2, label='Eddington limit')
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(source_data.keys())
    ax3.set_xlabel('$L / L_{Edd}$')
    ax3.set_title('Eddington Ratios of Accreting Sources')
    ax3.set_xscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.set_xlim(0.001, 100)

    # Shade super-Eddington
    ax3.axvspan(1, 100, alpha=0.1, color='red')

    # Plot 4: Eddington-limited growth of black holes
    ax4 = axes[1, 1]

    # Initial seed masses
    M_seeds = [100, 1000, 10000]  # M_sun
    colors = ['blue', 'green', 'red']

    # Time evolution at Eddington limit
    eta = 0.1  # Radiative efficiency
    t_Edd = eta * sigma_T * c / (4 * np.pi * G * m_p)  # Eddington time ~ 45 Myr

    t = np.linspace(0, 1e9, 200)  # years
    t_seconds = t * 3.156e7

    for M_seed, color in zip(M_seeds, colors):
        # M(t) = M_0 * exp(t / t_Edd) for Eddington-limited growth
        M_t = M_seed * np.exp(t_seconds / t_Edd)
        ax4.semilogy(t / 1e9, M_t, color=color, lw=2,
                     label=f'$M_0$ = {M_seed} M$_\\odot$')

    # Mark formation of SMBHs
    ax4.axhline(y=1e9, color='gray', linestyle=':', alpha=0.7)
    ax4.text(0.05, 2e9, '$10^9$ M$_\\odot$ SMBH', fontsize=9)

    # Mark z ~ 6 quasars
    t_universe_z6 = 0.9  # Gyr
    ax4.axvline(x=t_universe_z6, color='purple', linestyle='--', alpha=0.7)
    ax4.text(t_universe_z6 + 0.02, 1e3, 'Age at z=6', fontsize=9, rotation=90)

    ax4.set_xlabel('Time (Gyr)')
    ax4.set_ylabel('Black Hole Mass (M$_\\odot$)')
    ax4.set_title('Eddington-Limited BH Growth')
    ax4.legend()
    ax4.grid(True, alpha=0.3, which='both')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(100, 1e11)

    # Add e-folding time note
    t_efold_Myr = t_Edd / 3.156e13
    textstr = f'e-folding time:\n$t_{{Edd}}$ = {t_efold_Myr:.0f} Myr'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax4.text(0.95, 0.05, textstr, transform=ax4.transAxes, fontsize=10,
             verticalalignment='bottom', horizontalalignment='right', bbox=props)

    plt.suptitle('Experiment 270: Eddington Luminosity\n'
                 'The balance between radiation pressure and gravity',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'eddington_luminosity.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'eddington_luminosity.png')}")


if __name__ == "__main__":
    main()
