"""
Experiment 202: Linear Growth Factor

This experiment demonstrates the linear growth factor for density
perturbations in cosmology - how structures grow over time.

Physical concepts:
- Linear perturbation theory
- Growth factor D(z)
- Growth rate f(z)
- Effects of dark energy on structure formation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, odeint, solve_ivp
from scipy.special import hyp2f1


# Physical constants
c = 299792458.0  # m/s
H0 = 70 * 1000 / 3.086e22  # 70 km/s/Mpc in s^-1


def hubble_E(z, Omega_m, Omega_Lambda):
    """E(z) = H(z)/H0 for flat LCDM."""
    return np.sqrt(Omega_m * (1 + z)**3 + Omega_Lambda)


def Omega_m_z(z, Omega_m0, Omega_Lambda0):
    """Matter density parameter at redshift z."""
    E2 = Omega_m0 * (1 + z)**3 + Omega_Lambda0
    return Omega_m0 * (1 + z)**3 / E2


def growth_factor_numerical(z, Omega_m, Omega_Lambda):
    """
    Calculate growth factor D(z) by numerical integration.

    D(z) = (5/2) * Omega_m * E(z) * integral from z to infinity of (1+z')/E(z')^3 dz'

    Normalized so D(0) = 1 for EdS.
    """
    def integrand(zp):
        E3 = hubble_E(zp, Omega_m, Omega_Lambda)**3
        return (1 + zp) / E3

    # Integrate from z to large z (approximate infinity)
    z_max = 1000
    result, _ = quad(integrand, z, z_max)

    # Growth factor
    E_z = hubble_E(z, Omega_m, Omega_Lambda)
    D_unnorm = 2.5 * Omega_m * E_z * result

    return D_unnorm


def growth_factor_approx(z, Omega_m, Omega_Lambda):
    """
    Approximate growth factor using fitting formula.

    D(a) ≈ a * g(a) where g(a) is a correction factor.

    Using Carroll, Press & Turner (1992) approximation:
    g(Omega) ≈ 2.5 * Omega / [Omega^(4/7) - Omega_Lambda + (1 + Omega/2)(1 + Omega_Lambda/70)]
    """
    a = 1 / (1 + z)

    # Omega_m(a)
    Omega = Omega_m_z(z, Omega_m, Omega_Lambda)
    Omega_L = Omega_Lambda / hubble_E(z, Omega_m, Omega_Lambda)**2

    g = 2.5 * Omega / (
        Omega**(4/7) - Omega_L + (1 + Omega/2) * (1 + Omega_L/70)
    )

    return a * g


def growth_factor_EdS(z):
    """Growth factor for Einstein-de Sitter (matter dominated)."""
    return 1 / (1 + z)


def growth_rate_f(z, Omega_m, Omega_Lambda):
    """
    Growth rate f = d ln D / d ln a

    For LCDM, approximately: f ≈ Omega_m(z)^0.55 (gamma ≈ 0.55)
    """
    Omega = Omega_m_z(z, Omega_m, Omega_Lambda)
    gamma = 0.55  # Growth index for LCDM
    return Omega**gamma


def growth_rate_exact(z, Omega_m, Omega_Lambda, dz=0.001):
    """Calculate growth rate f(z) numerically from D(z)."""
    D_z = growth_factor_numerical(z, Omega_m, Omega_Lambda)
    D_z_plus = growth_factor_numerical(z + dz, Omega_m, Omega_Lambda)

    # f = d ln D / d ln a = -(1+z) * d ln D / dz
    dln_D_dz = (np.log(D_z_plus) - np.log(D_z)) / dz
    return -(1 + z) * dln_D_dz


def sigma_8(z, sigma8_0, Omega_m, Omega_Lambda):
    """
    sigma_8(z) = sigma_8(0) * D(z) / D(0)

    sigma_8 is the rms amplitude of matter fluctuations at 8 h^-1 Mpc.
    """
    D_z = growth_factor_numerical(z, Omega_m, Omega_Lambda)
    D_0 = growth_factor_numerical(0, Omega_m, Omega_Lambda)
    return sigma8_0 * D_z / D_0


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Standard parameters
    Omega_m = 0.3
    Omega_Lambda = 0.7

    z_range = np.linspace(0, 5, 200)

    # ==========================================================================
    # Plot 1: Growth factor D(z)
    # ==========================================================================
    ax1 = axes[0, 0]

    # Calculate growth factor for different cosmologies
    cosmologies = [
        {'name': 'LCDM (0.3, 0.7)', 'Omega_m': 0.3, 'Omega_Lambda': 0.7, 'color': 'blue'},
        {'name': 'EdS (1.0, 0)', 'Omega_m': 1.0, 'Omega_Lambda': 0.0, 'color': 'green'},
        {'name': 'Open (0.3, 0)', 'Omega_m': 0.3, 'Omega_Lambda': 0.0, 'color': 'orange'},
    ]

    for cosmo in cosmologies:
        D = np.array([growth_factor_numerical(z, cosmo['Omega_m'], cosmo['Omega_Lambda'])
                     for z in z_range])
        # Normalize to D(0)
        D = D / D[0]

        ax1.plot(z_range, D, '-', color=cosmo['color'], lw=2, label=cosmo['name'])

    # Scale factor for reference
    a = 1 / (1 + z_range)
    ax1.plot(z_range, a, 'k--', lw=1.5, alpha=0.5, label='a = 1/(1+z)')

    ax1.set_xlabel('Redshift z')
    ax1.set_ylabel('Normalized growth factor D(z)/D(0)')
    ax1.set_title('Linear Growth Factor')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add annotation
    ax1.text(0.95, 0.95,
            'Growth is suppressed\nwhen Lambda dominates\n(late times)',
            transform=ax1.transAxes, fontsize=9, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ==========================================================================
    # Plot 2: Growth rate f(z)
    # ==========================================================================
    ax2 = axes[0, 1]

    for cosmo in cosmologies:
        f = np.array([growth_rate_f(z, cosmo['Omega_m'], cosmo['Omega_Lambda'])
                     for z in z_range])
        ax2.plot(z_range, f, '-', color=cosmo['color'], lw=2, label=cosmo['name'])

    # Also plot Omega_m(z)
    Omega_mz = np.array([Omega_m_z(z, 0.3, 0.7) for z in z_range])
    ax2.plot(z_range, Omega_mz, 'b--', lw=1.5, alpha=0.5, label='Omega_m(z)')

    ax2.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
    ax2.axhline(y=0.55, color='gray', linestyle=':', alpha=0.5)

    ax2.set_xlabel('Redshift z')
    ax2.set_ylabel('Growth rate f(z)')
    ax2.set_title('Growth Rate f = d ln D / d ln a')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)

    # Annotation
    ax2.text(3, 0.6, 'f = Omega_m^{0.55}', fontsize=11)
    ax2.text(3, 0.5, '(gamma = 0.55 for LCDM)', fontsize=9)

    # ==========================================================================
    # Plot 3: sigma_8(z) evolution
    # ==========================================================================
    ax3 = axes[1, 0]

    sigma8_0 = 0.8  # Present-day value

    for cosmo in cosmologies:
        s8 = np.array([sigma_8(z, sigma8_0, cosmo['Omega_m'], cosmo['Omega_Lambda'])
                      for z in z_range])
        ax3.plot(z_range, s8, '-', color=cosmo['color'], lw=2, label=cosmo['name'])

    ax3.axhline(y=sigma8_0, color='gray', linestyle='--', alpha=0.7)
    ax3.text(4.5, sigma8_0 + 0.02, f'sigma_8(0) = {sigma8_0}', fontsize=10)

    ax3.set_xlabel('Redshift z')
    ax3.set_ylabel('sigma_8(z)')
    ax3.set_title('Amplitude of Matter Fluctuations')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Mark nonlinear threshold
    ax3.axhline(y=1, color='red', linestyle=':', alpha=0.7)
    ax3.text(0.1, 1.05, 'Nonlinear regime (sigma ~ 1)', fontsize=9, color='red')

    # ==========================================================================
    # Plot 4: f * sigma_8 (commonly used combination)
    # ==========================================================================
    ax4 = axes[1, 1]

    for cosmo in cosmologies:
        f = np.array([growth_rate_f(z, cosmo['Omega_m'], cosmo['Omega_Lambda'])
                     for z in z_range])
        s8 = np.array([sigma_8(z, sigma8_0, cosmo['Omega_m'], cosmo['Omega_Lambda'])
                      for z in z_range])

        f_sigma8 = f * s8

        ax4.plot(z_range, f_sigma8, '-', color=cosmo['color'], lw=2, label=cosmo['name'])

    # Add some "data points" for illustration (simulating RSD measurements)
    z_data = np.array([0.1, 0.35, 0.6, 0.8, 1.0, 1.5])
    f_data = np.array([growth_rate_f(z, 0.3, 0.7) for z in z_data])
    s8_data = np.array([sigma_8(z, sigma8_0, 0.3, 0.7) for z in z_data])
    f_sigma8_data = f_data * s8_data
    errors = 0.05 * f_sigma8_data

    ax4.errorbar(z_data, f_sigma8_data, yerr=errors, fmt='ko', markersize=6,
                capsize=3, label='RSD data (simulated)')

    ax4.set_xlabel('Redshift z')
    ax4.set_ylabel('f(z) * sigma_8(z)')
    ax4.set_title('Growth Rate Diagnostic\n(measured from redshift-space distortions)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Linear Growth Factor for Cosmic Structure Formation\n'
                 'D describes how density perturbations grow over time',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Print summary
    print("Linear Growth Factor Summary:")
    print("=" * 60)

    print(f"\nGrowth factor at key redshifts (LCDM):")
    D_0 = growth_factor_numerical(0, 0.3, 0.7)
    for z in [0.0, 0.5, 1.0, 2.0, 3.0]:
        D = growth_factor_numerical(z, 0.3, 0.7)
        f = growth_rate_f(z, 0.3, 0.7)
        s8 = sigma_8(z, 0.8, 0.3, 0.7)

        print(f"  z = {z}:")
        print(f"    D(z)/D(0) = {D/D_0:.4f}")
        print(f"    f(z) = {f:.4f}")
        print(f"    sigma_8(z) = {s8:.4f}")
        print(f"    f * sigma_8 = {f*s8:.4f}")

    print(f"\nPhysics:")
    print(f"  - Growth factor D(z) describes linear density perturbations")
    print(f"  - In EdS: D proportional to a (simple growth)")
    print(f"  - Dark energy suppresses growth at late times")
    print(f"  - f * sigma_8 is measured via redshift-space distortions (RSD)")
    print(f"  - Different modified gravity theories predict different f(z)")

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'growth_factor.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")


if __name__ == "__main__":
    main()
