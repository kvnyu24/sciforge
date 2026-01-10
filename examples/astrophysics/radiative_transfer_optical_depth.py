"""
Experiment 268: Radiative Transfer and Optical Depth

Demonstrates radiative transfer through absorbing/emitting media
and the concept of optical depth.

Physical concepts:
- Optical depth: tau = integral(kappa * rho * ds)
- Intensity attenuation: I = I_0 * exp(-tau)
- Photosphere defined where tau ~ 2/3
- Eddington approximation for stellar atmospheres
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt

# Physical constants
c = 2.998e8
h = 6.626e-34
k_B = 1.381e-23
sigma_SB = 5.670e-8


def planck_function(nu, T):
    """Planck blackbody function B_nu(T)."""
    x = h * nu / (k_B * T)
    return 2 * h * nu**3 / c**2 / (np.exp(x) - 1 + 1e-100)


def transfer_equation_solve(tau_grid, S, I_boundary=0):
    """
    Solve radiative transfer equation dI/dtau = I - S

    Args:
        tau_grid: Optical depth grid (increasing inward)
        S: Source function at each point
        I_boundary: Boundary intensity at tau=0

    Returns:
        Intensity at each optical depth
    """
    n = len(tau_grid)
    I = np.zeros(n)
    I[0] = I_boundary

    for i in range(1, n):
        dtau = tau_grid[i] - tau_grid[i-1]
        # Formal solution: I = I_0 * exp(-dtau) + S * (1 - exp(-dtau))
        I[i] = I[i-1] * np.exp(-dtau) + S[i] * (1 - np.exp(-dtau))

    return I


def eddington_limb_darkening(mu, a=0.6):
    """
    Limb darkening from Eddington approximation.

    I(mu) / I(1) = a + (1-a) * mu

    where mu = cos(theta) from normal.
    """
    return a + (1 - a) * mu


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Optical depth and transmission
    ax1 = axes[0, 0]

    tau = np.linspace(0, 5, 200)

    # Transmission
    transmission = np.exp(-tau)

    ax1.plot(tau, transmission, 'b-', lw=2, label='Transmission $e^{-\\tau}$')
    ax1.fill_between(tau, 0, transmission, alpha=0.2)

    # Mark key optical depths
    key_taus = [2/3, 1, 2, 3]
    for t in key_taus:
        trans = np.exp(-t)
        ax1.plot(t, trans, 'ro', markersize=8)
        ax1.annotate(f'$\\tau$ = {t:.2f}\nT = {trans*100:.1f}%',
                     xy=(t, trans), xytext=(t + 0.3, trans + 0.1),
                     fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))

    ax1.axhline(y=np.exp(-2/3), color='green', linestyle='--', alpha=0.7,
                label='$\\tau$ = 2/3 (photosphere)')

    ax1.set_xlabel('Optical Depth $\\tau$')
    ax1.set_ylabel('Transmission $I/I_0$')
    ax1.set_title('Transmission Through Absorbing Medium')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 5)
    ax1.set_ylim(0, 1.1)

    # Plot 2: Emergent intensity from stellar atmosphere
    ax2 = axes[0, 1]

    # Temperature profile (Eddington grey atmosphere)
    tau_atm = np.linspace(0, 10, 200)
    T_eff = 5778  # K (Sun)

    # T^4 = (3/4) * T_eff^4 * (tau + 2/3)
    T = T_eff * (0.75 * (tau_atm + 2/3))**0.25

    ax2.plot(tau_atm, T, 'r-', lw=2, label='Temperature')
    ax2.axhline(y=T_eff, color='gray', linestyle='--', alpha=0.7)
    ax2.axvline(x=2/3, color='green', linestyle='--', alpha=0.7, label='Photosphere')

    ax2.set_xlabel('Optical Depth $\\tau$')
    ax2.set_ylabel('Temperature (K)')
    ax2.set_title('Grey Atmosphere Temperature Profile\n$T^4 = \\frac{3}{4}T_{eff}^4(\\tau + 2/3)$')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 10)

    # Add second y-axis for T/T_eff
    ax2_twin = ax2.twinx()
    ax2_twin.set_ylabel('$T / T_{eff}$', color='blue')
    ax2_twin.set_ylim(ax2.get_ylim()[0] / T_eff, ax2.get_ylim()[1] / T_eff)
    ax2_twin.tick_params(axis='y', labelcolor='blue')

    # Plot 3: Limb darkening
    ax3 = axes[1, 0]

    # Angular positions from center to limb
    r_norm = np.linspace(0, 1, 100)  # r/R
    mu = np.sqrt(1 - r_norm**2)  # cos(theta)

    # Different limb darkening laws
    # Linear: I/I_c = 1 - u*(1 - mu)
    u_values = [0.3, 0.5, 0.7]

    for u in u_values:
        I_ratio = 1 - u * (1 - mu)
        ax3.plot(r_norm, I_ratio, lw=2, label=f'u = {u}')

    # Eddington approximation
    I_edd = eddington_limb_darkening(mu)
    ax3.plot(r_norm, I_edd / I_edd[0], 'k--', lw=2, label='Eddington')

    ax3.set_xlabel('Normalized radius $r/R$')
    ax3.set_ylabel('$I(r) / I(center)$')
    ax3.set_title('Limb Darkening: $I(\\mu) = I_c[1 - u(1 - \\mu)]$')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1.1)

    # Add inset showing solar disk
    ax_inset = ax3.inset_axes([0.55, 0.55, 0.4, 0.4])

    # Create solar disk image
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)

    # Intensity with limb darkening
    u = 0.6
    mu_disk = np.sqrt(np.maximum(0, 1 - R**2))
    I_disk = np.where(R <= 1, 1 - u * (1 - mu_disk), 0)

    ax_inset.imshow(I_disk, extent=[-1, 1, -1, 1], cmap='hot', origin='lower')
    ax_inset.set_title('Solar disk', fontsize=9)
    ax_inset.set_aspect('equal')
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])

    # Plot 4: Spectral line formation
    ax4 = axes[1, 1]

    # Wavelength relative to line center
    dlambda = np.linspace(-2, 2, 200)  # In units of line width

    # Line opacity profile (Gaussian)
    kappa_line = np.exp(-dlambda**2 / 2)

    # Continuum opacity
    kappa_cont = 0.1

    # Total opacity
    kappa_total = kappa_cont + kappa_line

    # Optical depth through atmosphere (simplified)
    column = 1.0  # Column density unit
    tau_total = kappa_total * column

    # Emergent intensity (assuming source function ~ blackbody)
    T_cont = 6000  # K
    T_line = 5000  # K (cooler in line)

    # Contribution function peaks at tau ~ 1
    T_emerge = T_cont * (tau_total < 1) + T_line * (tau_total >= 1)
    I_emerge = np.exp(-tau_total)

    # Normalized to continuum
    I_cont = np.exp(-kappa_cont * column)
    line_profile = I_emerge / I_cont

    ax4.plot(dlambda, line_profile, 'b-', lw=2)
    ax4.fill_between(dlambda, line_profile, 1, alpha=0.3)

    ax4.axhline(y=1.0, color='gray', linestyle=':', alpha=0.7)
    ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50% depth')

    # FWHM
    half_max = (line_profile.min() + 1) / 2
    above_half = dlambda[line_profile > half_max]
    fwhm = above_half.max() - above_half.min()
    ax4.annotate(f'FWHM ~ {fwhm:.1f}', xy=(0, half_max),
                 xytext=(1, 0.7), fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='red'))

    ax4.set_xlabel('$\\Delta\\lambda$ (line widths)')
    ax4.set_ylabel('$I / I_{continuum}$')
    ax4.set_title('Absorption Line Formation')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-2, 2)
    ax4.set_ylim(0, 1.2)

    plt.suptitle('Experiment 268: Radiative Transfer and Optical Depth\n'
                 'How light propagates through absorbing media',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'radiative_transfer_optical_depth.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'radiative_transfer_optical_depth.png')}")


if __name__ == "__main__":
    main()
