"""
Experiment 203: CMB Acoustic Peaks (Toy Model)

This experiment demonstrates a simplified model of CMB acoustic peaks,
showing how sound waves in the early universe create the characteristic
pattern in the CMB power spectrum.

Physical concepts:
- Baryon-photon fluid oscillations
- Sound horizon at recombination
- Acoustic peak locations
- Sachs-Wolfe effect (qualitative)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, solve_ivp


# Physical constants
c = 299792458.0  # m/s
H0 = 70 * 1000 / 3.086e22  # 70 km/s/Mpc in s^-1
Mpc = 3.086e22  # meters


def hubble_E(z, Omega_m, Omega_r, Omega_Lambda):
    """E(z) = H(z)/H0."""
    return np.sqrt(Omega_m * (1 + z)**3 + Omega_r * (1 + z)**4 + Omega_Lambda)


def sound_speed(z, Omega_b, Omega_r):
    """
    Sound speed in baryon-photon fluid.

    c_s = c / sqrt(3 * (1 + R))

    where R = 3 * rho_b / (4 * rho_gamma) = 3 * Omega_b / (4 * Omega_gamma) * (1+z)^(-1)
    """
    # Ratio of baryon to photon energy density
    # R = (3/4) * (rho_b / rho_gamma) = (3/4) * (Omega_b / Omega_r) * (1+z)^(-1)
    R = 0.75 * Omega_b / Omega_r / (1 + z)

    return c / np.sqrt(3 * (1 + R))


def comoving_sound_horizon(z_rec, Omega_m, Omega_r, Omega_Lambda, Omega_b, H0=H0):
    """
    Comoving sound horizon at recombination.

    r_s = integral from z_rec to infinity of c_s(z) / H(z) dz
    """
    def integrand(z):
        cs = sound_speed(z, Omega_b, Omega_r)
        H = H0 * hubble_E(z, Omega_m, Omega_r, Omega_Lambda)
        return cs / H

    result, _ = quad(integrand, z_rec, 1e6)
    return result


def angular_scale_sound_horizon(z_rec, Omega_m, Omega_r, Omega_Lambda, Omega_b, H0=H0):
    """
    Angular scale of sound horizon.

    theta_s = r_s / D_A(z_rec)
    """
    r_s = comoving_sound_horizon(z_rec, Omega_m, Omega_r, Omega_Lambda, Omega_b, H0)

    # Angular diameter distance to recombination
    def integrand(z):
        return c / (H0 * hubble_E(z, Omega_m, Omega_r, Omega_Lambda))

    D_C, _ = quad(integrand, 0, z_rec)
    D_A = D_C / (1 + z_rec)

    return r_s / D_A


def acoustic_oscillation(k, t, Omega_b, Omega_r):
    """
    Simplified model of acoustic oscillation amplitude.

    For a given wavenumber k, the perturbation oscillates as:
    delta(k, t) ~ cos(k * r_s(t))

    where r_s(t) is the sound horizon.
    """
    # This is a toy model - real calculation involves coupled ODEs
    return np.cos(k * t)  # Simplified


def cmb_power_spectrum_toy(ell, Omega_m, Omega_r, Omega_Lambda, Omega_b,
                           A_s=1e-9, n_s=0.965, z_rec=1090):
    """
    Toy model CMB power spectrum.

    The acoustic peaks appear at multipoles l_n ~ n * pi / theta_s

    This is a VERY simplified model for illustration.
    """
    # Angular scale of sound horizon
    theta_s = angular_scale_sound_horizon(z_rec, Omega_m, Omega_r, Omega_Lambda, Omega_b)

    # Multipole corresponding to sound horizon
    ell_s = np.pi / theta_s

    # Simplified power spectrum with acoustic oscillations
    # C_ell ~ A_s * (ell/ell_pivot)^(n_s-1) * transfer^2 * sin^2(ell * theta_s)

    # Primordial power (approximately)
    ell_pivot = 1000
    P_prim = A_s * (ell / ell_pivot)**(n_s - 1)

    # Acoustic oscillations (simplified)
    # First peak at ell ~ ell_s, subsequent peaks at ell ~ n * ell_s
    oscillation = np.sin(ell * theta_s)**2

    # Damping at high ell (Silk damping)
    ell_damping = 1500  # Approximate damping scale
    damping = np.exp(-(ell / ell_damping)**2)

    # Sachs-Wolfe plateau at low ell
    sw_scale = 200
    sw_plateau = 1 / (1 + (ell / sw_scale)**2)

    # Combine (this is NOT physical, just illustrative)
    Cl = P_prim * (oscillation + 0.3 * sw_plateau) * damping

    # Normalize to typical CMB values
    Cl = Cl * 6e9 / np.max(Cl)

    return Cl, ell_s


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Standard parameters
    Omega_m = 0.3
    Omega_r = 9e-5
    Omega_Lambda = 0.7
    Omega_b = 0.05  # Baryon density

    z_rec = 1090  # Recombination redshift

    # ==========================================================================
    # Plot 1: Sound speed evolution
    # ==========================================================================
    ax1 = axes[0, 0]

    z_range = np.logspace(0, 6, 500)

    cs = np.array([sound_speed(z, Omega_b, Omega_r) for z in z_range])

    ax1.semilogx(z_range, cs / c, 'b-', lw=2)

    ax1.axhline(y=1/np.sqrt(3), color='red', linestyle='--', lw=1.5,
               label='c_s = c/sqrt(3) (photon dominated)')
    ax1.axvline(x=z_rec, color='green', linestyle=':', lw=1.5,
               label=f'Recombination (z = {z_rec})')

    ax1.set_xlabel('Redshift z')
    ax1.set_ylabel('Sound speed c_s / c')
    ax1.set_title('Sound Speed in Baryon-Photon Fluid')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, 1e6)

    # Annotate
    ax1.annotate('Sound speed decreases\nas baryons become important',
                xy=(1e4, 0.5), xytext=(1e2, 0.4),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=10)

    # ==========================================================================
    # Plot 2: Sound horizon evolution
    # ==========================================================================
    ax2 = axes[0, 1]

    # Calculate sound horizon vs z
    r_s = []
    for z in z_range:
        if z < z_rec:
            rs = comoving_sound_horizon(z, Omega_m, Omega_r, Omega_Lambda, Omega_b)
        else:
            rs = comoving_sound_horizon(z, Omega_m, Omega_r, Omega_Lambda, Omega_b)
        r_s.append(rs)

    r_s = np.array(r_s)

    ax2.loglog(z_range, r_s / Mpc, 'b-', lw=2)

    # Mark sound horizon at recombination
    rs_rec = comoving_sound_horizon(z_rec, Omega_m, Omega_r, Omega_Lambda, Omega_b)
    ax2.plot(z_rec, rs_rec / Mpc, 'ro', markersize=10)
    ax2.axhline(y=rs_rec / Mpc, color='red', linestyle='--', alpha=0.5)
    ax2.annotate(f'r_s(z_rec) = {rs_rec/Mpc:.0f} Mpc',
                xy=(z_rec, rs_rec/Mpc), xytext=(1e4, rs_rec/Mpc * 2),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10)

    ax2.axvline(x=z_rec, color='green', linestyle=':', lw=1.5)

    ax2.set_xlabel('Redshift z')
    ax2.set_ylabel('Comoving sound horizon (Mpc)')
    ax2.set_title('Sound Horizon Evolution')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_xlim(1, 1e6)

    # ==========================================================================
    # Plot 3: CMB power spectrum (toy model)
    # ==========================================================================
    ax3 = axes[1, 0]

    ell = np.arange(2, 2500)

    Cl, ell_s = cmb_power_spectrum_toy(ell, Omega_m, Omega_r, Omega_Lambda, Omega_b)

    # Plot ell(ell+1)Cl/(2pi) which is what's usually shown
    Dl = ell * (ell + 1) * Cl / (2 * np.pi)

    ax3.plot(ell, Dl, 'b-', lw=1.5, label='Toy model')

    # Mark acoustic peaks
    peak_locations = ell_s * np.arange(1, 8)
    for i, ell_peak in enumerate(peak_locations):
        if ell_peak < 2500:
            ax3.axvline(x=ell_peak, color='red', linestyle=':', alpha=0.5)
            ax3.text(ell_peak, np.max(Dl) * 1.05, f'{i+1}', fontsize=9,
                    ha='center', color='red')

    ax3.set_xlabel('Multipole l')
    ax3.set_ylabel('D_l = l(l+1)C_l / (2pi) [uK^2]')
    ax3.set_title('CMB Temperature Power Spectrum (Toy Model)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 2500)

    # Add annotation
    ax3.text(0.95, 0.95,
            'Peak positions depend on:\n'
            '- Sound horizon at recombination\n'
            '- Angular diameter distance to CMB\n'
            '- These constrain cosmological parameters!',
            transform=ax3.transAxes, fontsize=9, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ==========================================================================
    # Plot 4: Effect of baryon density on peaks
    # ==========================================================================
    ax4 = axes[1, 1]

    # Different baryon densities
    Omega_b_values = [0.03, 0.05, 0.07]
    colors = ['blue', 'green', 'red']

    for Ob, color in zip(Omega_b_values, colors):
        Cl, _ = cmb_power_spectrum_toy(ell, Omega_m, Omega_r, Omega_Lambda, Ob)
        Dl = ell * (ell + 1) * Cl / (2 * np.pi)
        ax4.plot(ell, Dl, '-', color=color, lw=1.5,
                label=f'Omega_b = {Ob}')

    ax4.set_xlabel('Multipole l')
    ax4.set_ylabel('D_l = l(l+1)C_l / (2pi) [uK^2]')
    ax4.set_title('Effect of Baryon Density on CMB Peaks')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 2500)

    # Annotation about baryon loading
    ax4.text(0.95, 0.05,
            'Higher Omega_b:\n'
            '- Increases odd/even peak ratio\n'
            '- Shifts peak positions slightly\n'
            '- Increases damping',
            transform=ax4.transAxes, fontsize=9, va='bottom', ha='right',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    plt.suptitle('CMB Acoustic Peaks (Toy Model)\n'
                 'Sound waves in the early universe create the CMB power spectrum',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Print summary
    print("CMB Acoustic Peaks Summary:")
    print("=" * 60)

    print(f"\nCosmological parameters:")
    print(f"  Omega_m = {Omega_m}")
    print(f"  Omega_b = {Omega_b}")
    print(f"  Omega_r = {Omega_r}")
    print(f"  Omega_Lambda = {Omega_Lambda}")
    print(f"  z_rec = {z_rec}")

    print(f"\nDerived quantities:")
    rs_rec = comoving_sound_horizon(z_rec, Omega_m, Omega_r, Omega_Lambda, Omega_b)
    theta_s = angular_scale_sound_horizon(z_rec, Omega_m, Omega_r, Omega_Lambda, Omega_b)
    print(f"  Sound horizon at recombination: {rs_rec/Mpc:.1f} Mpc")
    print(f"  Angular scale: theta_s = {np.degrees(theta_s):.4f} degrees")
    print(f"  First peak at l ~ {ell_s:.0f}")

    print(f"\nPhysics of CMB peaks:")
    print(f"  - Acoustic oscillations in baryon-photon plasma")
    print(f"  - Sound waves freeze out at recombination")
    print(f"  - Peak positions encode cosmological parameters")
    print(f"  - First peak: flat geometry (l ~ 220 for flat universe)")
    print(f"  - Peak heights: baryon density, dark matter")
    print(f"  - Damping tail: Silk damping scale")

    print(f"\nNote: This is a TOY MODEL for illustration.")
    print(f"Real CMB calculations require solving Boltzmann equations!")

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'cmb_acoustic_peaks.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")


if __name__ == "__main__":
    main()
