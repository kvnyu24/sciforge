"""
Experiment 267: Main Sequence Scaling Relations

Demonstrates the scaling relations for main sequence stars:
L ~ M^alpha, R ~ M^beta, T ~ M^gamma

Physical concepts:
- Mass-luminosity relation from nuclear burning
- Mass-radius relation from hydrostatic equilibrium
- Main sequence lifetime scales as M/L
- HR diagram structure from these relations
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.plasma import StellarEvolution

# Physical constants
M_sun = 1.989e30
R_sun = 6.96e8
L_sun = 3.828e26
T_sun = 5778  # K
yr = 3.156e7  # seconds


def mass_luminosity(M):
    """
    Mass-luminosity relation for main sequence stars.

    L/L_sun ~ (M/M_sun)^alpha where alpha varies with mass.
    """
    m = M / M_sun

    if np.isscalar(m):
        if m < 0.43:
            return 0.23 * m**2.3 * L_sun
        elif m < 2:
            return m**4 * L_sun
        elif m < 55:
            return 1.4 * m**3.5 * L_sun
        else:
            return 32000 * m * L_sun
    else:
        L = np.zeros_like(m)
        L[m < 0.43] = 0.23 * m[m < 0.43]**2.3
        L[(m >= 0.43) & (m < 2)] = m[(m >= 0.43) & (m < 2)]**4
        L[(m >= 2) & (m < 55)] = 1.4 * m[(m >= 2) & (m < 55)]**3.5
        L[m >= 55] = 32000 * m[m >= 55]
        return L * L_sun


def mass_radius(M):
    """Mass-radius relation for main sequence stars."""
    m = M / M_sun
    return m**0.8 * R_sun


def mass_temperature(M):
    """Mass-temperature relation from L = 4*pi*R^2*sigma*T^4."""
    L = mass_luminosity(M)
    R = mass_radius(M)
    sigma = 5.67e-8
    return (L / (4 * np.pi * R**2 * sigma))**0.25


def main_sequence_lifetime(M):
    """Main sequence lifetime: tau ~ M/L."""
    L = mass_luminosity(M)
    return 1e10 * yr * (M / M_sun) / (L / L_sun)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Mass range
    M = np.logspace(-1, 2, 100) * M_sun

    # Plot 1: Mass-Luminosity relation
    ax1 = axes[0, 0]

    L = mass_luminosity(M)

    ax1.loglog(M / M_sun, L / L_sun, 'b-', lw=2, label='Empirical relation')

    # Show different power law regimes
    m_low = M[M < 0.43 * M_sun] / M_sun
    m_mid = M[(M >= 0.43 * M_sun) & (M < 2 * M_sun)] / M_sun
    m_high = M[(M >= 2 * M_sun) & (M < 55 * M_sun)] / M_sun

    ax1.plot(m_low, 0.23 * m_low**2.3, 'r--', lw=1.5, alpha=0.7, label='L ~ M$^{2.3}$')
    ax1.plot(m_mid, m_mid**4, 'g--', lw=1.5, alpha=0.7, label='L ~ M$^4$')
    ax1.plot(m_high, 1.4 * m_high**3.5, 'm--', lw=1.5, alpha=0.7, label='L ~ M$^{3.5}$')

    # Mark Sun
    ax1.plot(1, 1, 'yo', markersize=12, markeredgecolor='black', label='Sun')

    # Mark other stars
    stars = {
        'Sirius A': (2.1, 25.4),
        'Vega': (2.1, 40),
        'Proxima': (0.12, 0.0017),
        'Betelgeuse': (20, 1.4e5),  # Not on MS but shown for reference
    }

    for name, (m, L_star) in stars.items():
        ax1.plot(m, L_star, 'k*', markersize=8)
        ax1.annotate(name, (m, L_star), xytext=(5, 5),
                     textcoords='offset points', fontsize=8)

    ax1.set_xlabel('Mass (M$_\\odot$)')
    ax1.set_ylabel('Luminosity (L$_\\odot$)')
    ax1.set_title('Mass-Luminosity Relation')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_xlim(0.1, 100)
    ax1.set_ylim(1e-4, 1e7)

    # Plot 2: HR Diagram
    ax2 = axes[0, 1]

    T = mass_temperature(M)
    L = mass_luminosity(M)

    # Color by mass
    scatter = ax2.scatter(T, L / L_sun, c=np.log10(M / M_sun), cmap='RdYlBu_r',
                          s=50, alpha=0.7)
    plt.colorbar(scatter, ax=ax2, label='log$_{10}$(M/M$_\\odot$)')

    # Draw spectral class regions
    spectral_temps = {
        'O': (30000, 50000),
        'B': (10000, 30000),
        'A': (7500, 10000),
        'F': (6000, 7500),
        'G': (5200, 6000),
        'K': (3700, 5200),
        'M': (2400, 3700),
    }

    for spec, (T_low, T_high) in spectral_temps.items():
        ax2.axvspan(T_low, T_high, alpha=0.1)
        ax2.text((T_low + T_high) / 2, 1e-3, spec, fontsize=10, ha='center')

    # Mark Sun
    ax2.plot(T_sun, 1, 'yo', markersize=12, markeredgecolor='black', zorder=5)

    ax2.set_xlabel('Effective Temperature (K)')
    ax2.set_ylabel('Luminosity (L$_\\odot$)')
    ax2.set_title('Hertzsprung-Russell Diagram (Main Sequence)')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.invert_xaxis()  # Hot on left
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_xlim(50000, 2000)
    ax2.set_ylim(1e-4, 1e7)

    # Plot 3: Main sequence lifetime
    ax3 = axes[1, 0]

    tau = main_sequence_lifetime(M)

    ax3.loglog(M / M_sun, tau / (1e9 * yr), 'b-', lw=2, label='MS lifetime')

    # Show L/M scaling
    m_plot = M / M_sun
    tau_approx = 1e10 * m_plot**(-2.5) / 1e9  # Approximate M^-2.5 for M > 0.5
    ax3.loglog(m_plot, tau_approx, 'r--', lw=1.5, alpha=0.7, label='$\\tau \\propto M^{-2.5}$')

    # Mark Sun
    ax3.plot(1, 10, 'yo', markersize=12, markeredgecolor='black', label='Sun (10 Gyr)')

    # Mark universe age
    t_universe = 13.8  # Gyr
    ax3.axhline(y=t_universe, color='gray', linestyle=':', alpha=0.7)
    ax3.text(50, t_universe * 1.2, 'Age of Universe', fontsize=9)

    ax3.set_xlabel('Mass (M$_\\odot$)')
    ax3.set_ylabel('Main Sequence Lifetime (Gyr)')
    ax3.set_title('Main Sequence Lifetime: $\\tau_{MS} \\sim M/L$')
    ax3.legend()
    ax3.grid(True, alpha=0.3, which='both')
    ax3.set_xlim(0.1, 100)
    ax3.set_ylim(1e-3, 1e4)

    # Shade short-lived region
    ax3.axhspan(0, 0.01, alpha=0.1, color='red')
    ax3.text(30, 0.003, 'Short-lived\n(< 10 Myr)', fontsize=9, ha='center')

    # Plot 4: Mass-Radius and surface gravity
    ax4 = axes[1, 1]

    R = mass_radius(M)

    ax4.loglog(M / M_sun, R / R_sun, 'b-', lw=2, label='Mass-Radius')

    # Power law fit
    m_plot = M / M_sun
    ax4.loglog(m_plot, m_plot**0.8, 'r--', lw=1.5, alpha=0.7, label='R ~ M$^{0.8}$')

    # Mark Sun
    ax4.plot(1, 1, 'yo', markersize=12, markeredgecolor='black', label='Sun')

    # Surface gravity
    ax4_twin = ax4.twinx()
    g = 274 * (M / M_sun) / (R / R_sun)**2  # g in m/s^2, normalized to Sun
    ax4_twin.loglog(M / M_sun, g, 'g:', lw=2, label='Surface gravity')

    ax4.set_xlabel('Mass (M$_\\odot$)')
    ax4.set_ylabel('Radius (R$_\\odot$)', color='blue')
    ax4_twin.set_ylabel('Surface gravity (m/s$^2$)', color='green')

    ax4.tick_params(axis='y', labelcolor='blue')
    ax4_twin.tick_params(axis='y', labelcolor='green')

    ax4.set_title('Mass-Radius Relation and Surface Gravity')

    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    ax4.grid(True, alpha=0.3, which='both')
    ax4.set_xlim(0.1, 100)
    ax4.set_ylim(0.1, 30)

    plt.suptitle('Experiment 267: Main Sequence Scaling Relations\n'
                 'How stellar properties scale with mass',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'main_sequence_scaling.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'main_sequence_scaling.png')}")


if __name__ == "__main__":
    main()
