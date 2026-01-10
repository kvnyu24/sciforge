"""
Experiment 201: Distance-Redshift Relations

This experiment demonstrates cosmological distance measures and their
dependence on redshift for different cosmological parameters.

Physical concepts:
- Comoving distance
- Luminosity distance
- Angular diameter distance
- Distance modulus (for supernovae)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from src.sciforge.physics.relativity import FriedmannEquations, RedshiftDistance


# Physical constants
c = 299792458.0  # m/s
H0 = 70 * 1000 / 3.086e22  # 70 km/s/Mpc in s^-1
Mpc = 3.086e22  # meters


def hubble_E(z, Omega_m, Omega_Lambda):
    """E(z) = H(z)/H0 for flat LCDM."""
    return np.sqrt(Omega_m * (1 + z)**3 + Omega_Lambda)


def comoving_distance(z, Omega_m, Omega_Lambda, H0=H0, c=c):
    """
    Comoving distance to redshift z.

    D_C(z) = c/H0 * integral from 0 to z of dz'/E(z')
    """
    def integrand(zp):
        return 1 / hubble_E(zp, Omega_m, Omega_Lambda)

    result, _ = quad(integrand, 0, z)
    return c / H0 * result


def luminosity_distance(z, Omega_m, Omega_Lambda, H0=H0, c=c):
    """
    Luminosity distance.

    D_L = (1 + z) * D_C
    """
    D_C = comoving_distance(z, Omega_m, Omega_Lambda, H0, c)
    return (1 + z) * D_C


def angular_diameter_distance(z, Omega_m, Omega_Lambda, H0=H0, c=c):
    """
    Angular diameter distance.

    D_A = D_C / (1 + z)
    """
    D_C = comoving_distance(z, Omega_m, Omega_Lambda, H0, c)
    return D_C / (1 + z)


def distance_modulus(z, Omega_m, Omega_Lambda, H0=H0, c=c):
    """
    Distance modulus for Type Ia supernovae.

    mu = 5 * log10(D_L / 10 pc)
    """
    D_L = luminosity_distance(z, Omega_m, Omega_Lambda, H0, c)
    D_L_pc = D_L / 3.086e16  # Convert to parsecs
    return 5 * np.log10(D_L_pc / 10)


def lookback_time(z, Omega_m, Omega_Lambda, H0=H0):
    """
    Lookback time to redshift z.

    t_L = 1/H0 * integral from 0 to z of dz'/[(1+z') * E(z')]
    """
    def integrand(zp):
        return 1 / ((1 + zp) * hubble_E(zp, Omega_m, Omega_Lambda))

    result, _ = quad(integrand, 0, z)
    return result / H0


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    Gyr = 1e9 * 365.25 * 24 * 3600  # Gigayear in seconds

    # Standard LCDM parameters
    Omega_m = 0.3
    Omega_Lambda = 0.7

    z_range = np.linspace(0.01, 3, 200)

    # ==========================================================================
    # Plot 1: Different distance measures
    # ==========================================================================
    ax1 = axes[0, 0]

    D_C = np.array([comoving_distance(z, Omega_m, Omega_Lambda) for z in z_range])
    D_L = np.array([luminosity_distance(z, Omega_m, Omega_Lambda) for z in z_range])
    D_A = np.array([angular_diameter_distance(z, Omega_m, Omega_Lambda) for z in z_range])

    ax1.plot(z_range, D_C / Mpc, 'b-', lw=2, label='Comoving D_C')
    ax1.plot(z_range, D_L / Mpc, 'r-', lw=2, label='Luminosity D_L')
    ax1.plot(z_range, D_A / Mpc, 'g-', lw=2, label='Angular diameter D_A')

    # Hubble distance
    D_H = c / H0
    ax1.axhline(y=D_H / Mpc, color='gray', linestyle='--', alpha=0.7,
               label=f'Hubble distance ({D_H/Mpc:.0f} Mpc)')

    ax1.set_xlabel('Redshift z')
    ax1.set_ylabel('Distance (Mpc)')
    ax1.set_title('Cosmological Distance Measures (LCDM)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 3)

    # Add relations
    ax1.text(0.95, 0.05,
            'D_L = (1+z) D_C\nD_A = D_C / (1+z)\nD_L = (1+z)^2 D_A',
            transform=ax1.transAxes, fontsize=9, va='bottom', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ==========================================================================
    # Plot 2: Distance modulus (Hubble diagram)
    # ==========================================================================
    ax2 = axes[0, 1]

    # Different cosmologies
    cosmologies = [
        {'name': 'LCDM (0.3, 0.7)', 'Omega_m': 0.3, 'Omega_Lambda': 0.7, 'color': 'blue'},
        {'name': 'EdS (1.0, 0)', 'Omega_m': 1.0, 'Omega_Lambda': 0.0, 'color': 'green'},
        {'name': 'Open (0.3, 0)', 'Omega_m': 0.3, 'Omega_Lambda': 0.0, 'color': 'orange'},
        {'name': 'Lambda (0.1, 0.9)', 'Omega_m': 0.1, 'Omega_Lambda': 0.9, 'color': 'red'},
    ]

    for cosmo in cosmologies:
        mu = np.array([distance_modulus(z, cosmo['Omega_m'], cosmo['Omega_Lambda'])
                      for z in z_range])
        ax2.plot(z_range, mu, '-', color=cosmo['color'], lw=2, label=cosmo['name'])

    # Add some "data points" for illustration
    z_data = np.array([0.1, 0.3, 0.5, 0.7, 1.0, 1.2])
    mu_data = np.array([distance_modulus(z, 0.3, 0.7) for z in z_data])
    mu_err = 0.15 * np.ones_like(mu_data)

    ax2.errorbar(z_data, mu_data, yerr=mu_err, fmt='ko', markersize=6,
                capsize=3, label='Type Ia SNe (simulated)')

    ax2.set_xlabel('Redshift z')
    ax2.set_ylabel('Distance modulus mu')
    ax2.set_title('Hubble Diagram (Type Ia Supernovae)')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ==========================================================================
    # Plot 3: Angular diameter distance - note the maximum!
    # ==========================================================================
    ax3 = axes[1, 0]

    z_extended = np.linspace(0.01, 10, 500)
    D_A_extended = np.array([angular_diameter_distance(z, Omega_m, Omega_Lambda)
                            for z in z_extended])

    ax3.plot(z_extended, D_A_extended / Mpc, 'b-', lw=2)

    # Find maximum
    max_idx = np.argmax(D_A_extended)
    z_max = z_extended[max_idx]
    D_A_max = D_A_extended[max_idx]

    ax3.plot(z_max, D_A_max / Mpc, 'ro', markersize=10)
    ax3.axvline(x=z_max, color='red', linestyle='--', alpha=0.5)
    ax3.annotate(f'Maximum at z = {z_max:.2f}\nD_A = {D_A_max/Mpc:.0f} Mpc',
                xy=(z_max, D_A_max/Mpc), xytext=(z_max + 1, D_A_max/Mpc * 0.9),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax3.set_xlabel('Redshift z')
    ax3.set_ylabel('Angular diameter distance (Mpc)')
    ax3.set_title('Angular Diameter Distance\n(Objects appear larger at high z!)')
    ax3.grid(True, alpha=0.3)

    # Add CMB surface
    z_cmb = 1100
    # D_A at CMB (roughly)
    ax3.text(8, D_A_max/Mpc * 0.5, f'CMB surface at z = {z_cmb}\n(beyond plot range)',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ==========================================================================
    # Plot 4: Lookback time
    # ==========================================================================
    ax4 = axes[1, 1]

    t_lookback = np.array([lookback_time(z, Omega_m, Omega_Lambda)
                          for z in z_extended])

    ax4.plot(z_extended, t_lookback / Gyr, 'b-', lw=2)

    # Age of universe
    t_age = lookback_time(np.inf, Omega_m, Omega_Lambda)
    ax4.axhline(y=t_age / Gyr, color='red', linestyle='--', lw=2,
               label=f'Age of universe = {t_age/Gyr:.1f} Gyr')

    # Mark key epochs
    epochs = [
        (0.5, 'z=0.5'),
        (1.0, 'z=1 (5.9 Gyr ago)'),
        (3.0, 'z=3'),
        (6.0, 'z=6 (first galaxies)'),
    ]

    for z_epoch, label in epochs:
        t_epoch = lookback_time(z_epoch, Omega_m, Omega_Lambda)
        ax4.plot(z_epoch, t_epoch / Gyr, 'go', markersize=6)
        ax4.annotate(label, xy=(z_epoch, t_epoch/Gyr),
                    xytext=(z_epoch + 0.5, t_epoch/Gyr - 1),
                    fontsize=8)

    ax4.set_xlabel('Redshift z')
    ax4.set_ylabel('Lookback time (Gyr)')
    ax4.set_title('Lookback Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 10)

    plt.suptitle('Cosmological Distance-Redshift Relations\n'
                 'H0 = 70 km/s/Mpc, Omega_m = 0.3, Omega_Lambda = 0.7',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Print summary
    print("Distance-Redshift Relations Summary:")
    print("=" * 60)

    print(f"\nCosmological parameters:")
    print(f"  H0 = 70 km/s/Mpc")
    print(f"  Omega_m = 0.3")
    print(f"  Omega_Lambda = 0.7")

    print(f"\nDistances at key redshifts:")
    for z in [0.1, 0.5, 1.0, 2.0, 3.0]:
        D_C = comoving_distance(z, Omega_m, Omega_Lambda)
        D_L = luminosity_distance(z, Omega_m, Omega_Lambda)
        D_A = angular_diameter_distance(z, Omega_m, Omega_Lambda)
        mu = distance_modulus(z, Omega_m, Omega_Lambda)
        t_L = lookback_time(z, Omega_m, Omega_Lambda)

        print(f"\n  z = {z}:")
        print(f"    D_C = {D_C/Mpc:.0f} Mpc")
        print(f"    D_L = {D_L/Mpc:.0f} Mpc")
        print(f"    D_A = {D_A/Mpc:.0f} Mpc")
        print(f"    mu = {mu:.2f} mag")
        print(f"    Lookback = {t_L/Gyr:.2f} Gyr")

    print(f"\nAngular diameter distance maximum:")
    print(f"  z = {z_max:.2f}")
    print(f"  D_A_max = {D_A_max/Mpc:.0f} Mpc")

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'distance_redshift.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")


if __name__ == "__main__":
    main()
