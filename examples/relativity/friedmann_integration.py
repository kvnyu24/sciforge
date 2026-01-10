"""
Experiment 200: Friedmann Equations Integration

This experiment integrates the Friedmann equations to compute the
evolution of the scale factor a(t) for different cosmological models.

Physical concepts:
- Friedmann equations
- Scale factor evolution
- Matter, radiation, and dark energy dominated eras
- Age of the universe
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp, quad
from src.sciforge.physics.relativity import FriedmannEquations


# Physical constants
c = 299792458.0  # m/s
H0_default = 70  # km/s/Mpc in standard units
H0_si = 70 * 1000 / 3.086e22  # Convert to s^-1


def hubble_parameter(a, Omega_m, Omega_r, Omega_Lambda, H0):
    """
    Calculate Hubble parameter H(a).

    H(a) = H0 * sqrt(Omega_r/a^4 + Omega_m/a^3 + Omega_k/a^2 + Omega_Lambda)
    """
    Omega_k = 1 - Omega_m - Omega_r - Omega_Lambda
    E_squared = Omega_r / a**4 + Omega_m / a**3 + Omega_k / a**2 + Omega_Lambda
    return H0 * np.sqrt(E_squared)


def friedmann_ode(a, t, Omega_m, Omega_r, Omega_Lambda, H0):
    """
    ODE for scale factor evolution.

    da/dt = a * H(a)
    """
    H = hubble_parameter(a, Omega_m, Omega_r, Omega_Lambda, H0)
    return a * H


def friedmann_ode_ivp(t, a, Omega_m, Omega_r, Omega_Lambda, H0):
    """Version for solve_ivp (swapped arguments)."""
    return friedmann_ode(a, t, Omega_m, Omega_r, Omega_Lambda, H0)


def age_of_universe(Omega_m, Omega_r, Omega_Lambda, H0):
    """
    Calculate age of universe by integrating.

    t_0 = integral from 0 to 1 of da / (a * H(a))
    """
    def integrand(a):
        if a <= 0:
            return 0
        H = hubble_parameter(a, Omega_m, Omega_r, Omega_Lambda, H0)
        return 1 / (a * H)

    result, _ = quad(integrand, 1e-10, 1)
    return result


def deceleration_parameter(a, Omega_m, Omega_r, Omega_Lambda):
    """
    Calculate deceleration parameter q(a).

    q = -1 - (d ln H / d ln a)
    For LCDM: q = (Omega_r/a^4 + Omega_m/(2*a^3) - Omega_Lambda) / E^2
    """
    Omega_k = 1 - Omega_m - Omega_r - Omega_Lambda
    E2 = Omega_r / a**4 + Omega_m / a**3 + Omega_k / a**2 + Omega_Lambda

    q = (Omega_r / a**4 + 0.5 * Omega_m / a**3 - Omega_Lambda) / E2
    return q


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Standard cosmological parameters
    H0 = H0_si  # Hubble constant in s^-1
    Gyr = 1e9 * 365.25 * 24 * 3600  # Gigayear in seconds

    # ==========================================================================
    # Plot 1: Scale factor evolution for different cosmologies
    # ==========================================================================
    ax1 = axes[0, 0]

    cosmologies = [
        {'name': 'LCDM (standard)', 'Omega_m': 0.3, 'Omega_r': 9e-5, 'Omega_Lambda': 0.7,
         'color': 'blue'},
        {'name': 'EdS (matter only)', 'Omega_m': 1.0, 'Omega_r': 0, 'Omega_Lambda': 0,
         'color': 'green'},
        {'name': 'Open (Omega=0.3)', 'Omega_m': 0.3, 'Omega_r': 0, 'Omega_Lambda': 0,
         'color': 'orange'},
        {'name': 'De Sitter (Lambda only)', 'Omega_m': 0.0, 'Omega_r': 0, 'Omega_Lambda': 1.0,
         'color': 'red'},
    ]

    # Time range (in Hubble times)
    t_H = 1 / H0  # Hubble time

    for cosmo in cosmologies:
        # Calculate age of universe for normalization
        try:
            t_now = age_of_universe(cosmo['Omega_m'], cosmo['Omega_r'],
                                   cosmo['Omega_Lambda'], H0)
        except:
            t_now = t_H

        # Solve forward and backward from today (a=1)
        # Forward: a > 1 (future)
        t_forward = np.linspace(0, 2 * t_H, 500)
        try:
            sol_forward = solve_ivp(
                lambda t, a: friedmann_ode_ivp(t, a, cosmo['Omega_m'],
                                               cosmo['Omega_r'], cosmo['Omega_Lambda'], H0),
                [0, 2 * t_H], [1.0], t_eval=t_forward, method='RK45'
            )
            a_forward = sol_forward.y[0]
        except:
            a_forward = np.ones_like(t_forward)

        # Backward: a < 1 (past)
        t_backward = np.linspace(0, -min(t_now, 2*t_H), 500)
        try:
            sol_backward = solve_ivp(
                lambda t, a: friedmann_ode_ivp(t, a, cosmo['Omega_m'],
                                               cosmo['Omega_r'], cosmo['Omega_Lambda'], H0),
                [0, t_backward[-1]], [1.0], t_eval=t_backward, method='RK45'
            )
            a_backward = sol_backward.y[0]
        except:
            a_backward = np.ones_like(t_backward)

        # Combine
        t_full = np.concatenate([t_backward[::-1], t_forward[1:]])
        a_full = np.concatenate([a_backward[::-1], a_forward[1:]])

        ax1.plot(t_full / Gyr, a_full, '-', color=cosmo['color'], lw=2,
                label=cosmo['name'])

    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=0, color='gray', linestyle=':', alpha=0.5)

    ax1.set_xlabel('Time (Gyr, with t=0 today)')
    ax1.set_ylabel('Scale factor a(t)')
    ax1.set_title('Scale Factor Evolution')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-15, 30)
    ax1.set_ylim(0, 5)

    # ==========================================================================
    # Plot 2: Hubble parameter vs redshift
    # ==========================================================================
    ax2 = axes[0, 1]

    z_range = np.linspace(0, 5, 200)
    a_range = 1 / (1 + z_range)

    for cosmo in cosmologies[:3]:  # Skip de Sitter for clarity
        H = hubble_parameter(a_range, cosmo['Omega_m'], cosmo['Omega_r'],
                            cosmo['Omega_Lambda'], H0)
        H_km_s_Mpc = H * 3.086e22 / 1000  # Convert to km/s/Mpc

        ax2.plot(z_range, H_km_s_Mpc, '-', color=cosmo['color'], lw=2,
                label=cosmo['name'])

    ax2.set_xlabel('Redshift z')
    ax2.set_ylabel('H(z) (km/s/Mpc)')
    ax2.set_title('Hubble Parameter vs Redshift')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Mark H0
    ax2.plot(0, 70, 'ko', markersize=8)
    ax2.annotate('H0 = 70 km/s/Mpc', xy=(0, 70), xytext=(0.5, 90),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=10)

    # ==========================================================================
    # Plot 3: Deceleration parameter q(z)
    # ==========================================================================
    ax3 = axes[1, 0]

    for cosmo in cosmologies[:3]:
        q = deceleration_parameter(a_range, cosmo['Omega_m'], cosmo['Omega_r'],
                                  cosmo['Omega_Lambda'])
        ax3.plot(z_range, q, '-', color=cosmo['color'], lw=2, label=cosmo['name'])

    ax3.axhline(y=0, color='gray', linestyle='--', lw=2, alpha=0.7)
    ax3.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax3.axhline(y=-1, color='gray', linestyle=':', alpha=0.5)

    ax3.set_xlabel('Redshift z')
    ax3.set_ylabel('Deceleration parameter q')
    ax3.set_title('Deceleration Parameter\n(q > 0: decelerating, q < 0: accelerating)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-1.5, 1.5)

    # Mark transition for LCDM
    # q = 0 when Omega_m/(2a^3) = Omega_Lambda
    a_trans = (0.3 / (2 * 0.7))**(1/3)
    z_trans = 1/a_trans - 1
    ax3.axvline(x=z_trans, color='blue', linestyle=':', alpha=0.7)
    ax3.text(z_trans + 0.1, 0.5, f'z_trans = {z_trans:.2f}', fontsize=9, color='blue')

    # ==========================================================================
    # Plot 4: Component densities vs redshift
    # ==========================================================================
    ax4 = axes[1, 1]

    # Standard LCDM
    Omega_m0 = 0.3
    Omega_r0 = 9e-5
    Omega_Lambda0 = 0.7

    # Densities as fraction of critical
    Omega_m = Omega_m0 / a_range**3
    Omega_r = Omega_r0 / a_range**4
    Omega_Lambda = Omega_Lambda0 * np.ones_like(a_range)

    # Normalized to total at each z
    E2 = Omega_m0 / a_range**3 + Omega_r0 / a_range**4 + Omega_Lambda0
    Omega_m_frac = Omega_m / E2
    Omega_r_frac = Omega_r / E2
    Omega_Lambda_frac = Omega_Lambda / E2

    ax4.fill_between(z_range, 0, Omega_r_frac, alpha=0.5, color='red',
                    label='Radiation')
    ax4.fill_between(z_range, Omega_r_frac, Omega_r_frac + Omega_m_frac,
                    alpha=0.5, color='blue', label='Matter')
    ax4.fill_between(z_range, Omega_r_frac + Omega_m_frac, 1,
                    alpha=0.5, color='green', label='Dark energy')

    ax4.set_xlabel('Redshift z')
    ax4.set_ylabel('Fraction of critical density')
    ax4.set_title('Cosmic Component Evolution (LCDM)')
    ax4.legend(loc='center right')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_xlim(0, 5)
    ax4.set_ylim(0, 1)

    # Mark transition redshifts
    # Matter-radiation equality: Omega_m = Omega_r
    z_eq = Omega_m0 / Omega_r0 - 1
    ax4.axvline(x=min(z_eq, 5), color='black', linestyle=':', alpha=0.7)

    plt.suptitle('Friedmann Equations: Cosmic Evolution\n'
                 'H^2 = (8piG/3)rho - kc^2/a^2 + Lambda*c^2/3',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Print summary
    print("Friedmann Equations Summary:")
    print("=" * 60)

    print(f"\nStandard LCDM cosmology:")
    print(f"  H0 = 70 km/s/Mpc")
    print(f"  Omega_m = 0.3 (matter)")
    print(f"  Omega_r = 9e-5 (radiation)")
    print(f"  Omega_Lambda = 0.7 (dark energy)")

    t_age = age_of_universe(0.3, 9e-5, 0.7, H0)
    print(f"\nAge of universe: {t_age/Gyr:.2f} Gyr")

    print(f"\nKey redshifts:")
    z_eq = 0.3 / 9e-5 - 1
    print(f"  Matter-radiation equality: z = {z_eq:.0f}")
    a_trans = (0.3 / (2 * 0.7))**(1/3)
    z_trans = 1/a_trans - 1
    print(f"  Acceleration begins: z = {z_trans:.2f}")

    print(f"\nCompare cosmologies:")
    for cosmo in cosmologies:
        try:
            t = age_of_universe(cosmo['Omega_m'], cosmo['Omega_r'],
                               cosmo['Omega_Lambda'], H0)
            print(f"  {cosmo['name']}: age = {t/Gyr:.2f} Gyr")
        except:
            print(f"  {cosmo['name']}: age = infinite (de Sitter)")

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'friedmann_integration.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")


if __name__ == "__main__":
    main()
