"""
Experiment 95: Synchrotron radiation.

This example demonstrates synchrotron radiation from relativistic charged
particles in circular orbits, showing the gamma^4 power dependence,
angular distribution (beaming), and spectrum characteristics.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Physical constants
C = 2.998e8          # Speed of light (m/s)
Q_E = 1.602e-19      # Elementary charge (C)
M_E = 9.109e-31      # Electron mass (kg)
M_P = 1.673e-27      # Proton mass (kg)
EPSILON_0 = 8.854e-12  # Permittivity of free space
HBAR = 1.055e-34     # Reduced Planck constant (J*s)
E_REST_E = M_E * C**2  # Electron rest energy (J)


def lorentz_factor(E, m):
    """
    Calculate Lorentz factor gamma from total energy.

    gamma = E / (m * c^2)

    Args:
        E: Total energy (J)
        m: Rest mass (kg)

    Returns:
        gamma: Lorentz factor
    """
    return E / (m * C**2)


def velocity_from_gamma(gamma):
    """
    Calculate velocity from Lorentz factor.

    v = c * sqrt(1 - 1/gamma^2)
    """
    return C * np.sqrt(1 - 1/gamma**2)


def synchrotron_power(q, m, gamma, rho):
    """
    Total radiated power from synchrotron radiation.

    P = (C_gamma / (2*pi)) * c * E^4 / (rho^2 * (m*c^2)^4)

    where C_gamma = q^2 / (3 * epsilon_0 * (m*c^2)^4)

    Simplified: P = (q^2 * c * gamma^4) / (6 * pi * epsilon_0 * rho^2)

    Args:
        q: Charge (C)
        m: Rest mass (kg)
        gamma: Lorentz factor
        rho: Radius of curvature (m)

    Returns:
        P: Radiated power (W)
    """
    return (q**2 * C * gamma**4) / (6 * np.pi * EPSILON_0 * rho**2)


def synchrotron_power_practical(E_GeV, rho, particle='electron'):
    """
    Synchrotron power in practical units.

    P = C_gamma * c * E^4 / rho^2

    For electrons: C_gamma = 8.85e-5 m / GeV^3

    Args:
        E_GeV: Particle energy (GeV)
        rho: Bending radius (m)
        particle: 'electron' or 'proton'

    Returns:
        P: Radiated power (W)
    """
    if particle == 'electron':
        C_gamma = 8.85e-5  # m/GeV^3
    else:  # proton
        C_gamma = 8.85e-5 * (M_E / M_P)**4  # Much smaller

    return C_gamma * C * E_GeV**4 / rho**2


def critical_frequency(gamma, rho):
    """
    Critical frequency of synchrotron radiation spectrum.

    omega_c = (3/2) * gamma^3 * c / rho

    Half of the radiated power is above this frequency.

    Args:
        gamma: Lorentz factor
        rho: Radius of curvature (m)

    Returns:
        omega_c: Critical angular frequency (rad/s)
    """
    return 1.5 * gamma**3 * C / rho


def critical_energy(gamma, rho):
    """
    Critical photon energy of synchrotron radiation.

    E_c = hbar * omega_c = (3/2) * hbar * gamma^3 * c / rho
    """
    return HBAR * critical_frequency(gamma, rho)


def angular_distribution(theta, gamma):
    """
    Angular distribution of synchrotron radiation power.

    Highly peaked in forward direction with opening angle ~ 1/gamma.

    This is an approximation for the instantaneous angular distribution.

    Args:
        theta: Angle from velocity direction (rad)
        gamma: Lorentz factor

    Returns:
        Relative power per solid angle (normalized)
    """
    # Approximate formula for angular distribution
    x = gamma * theta
    # Distribution peaks at theta=0, width ~ 1/gamma
    return 1 / (1 + x**2)**2.5


def spectrum_function(x):
    """
    Universal synchrotron spectrum function S(x) = x * integral K_{5/3}(t) dt.

    Where x = omega / omega_c.

    Using asymptotic approximations:
    - x << 1: S(x) ~ 2.15 * x^(1/3)
    - x >> 1: S(x) ~ sqrt(pi/2) * sqrt(x) * exp(-x)

    Args:
        x: omega / omega_c (normalized frequency)

    Returns:
        S(x): Normalized spectral function
    """
    # Use interpolation between asymptotic forms
    x = np.atleast_1d(x)
    S = np.zeros_like(x)

    # Low frequency asymptote
    low_mask = x < 0.3
    S[low_mask] = 2.15 * x[low_mask]**(1/3)

    # High frequency asymptote
    high_mask = x > 3
    S[high_mask] = np.sqrt(np.pi/2 * x[high_mask]) * np.exp(-x[high_mask])

    # Intermediate region (interpolation)
    mid_mask = ~low_mask & ~high_mask
    # Approximate peak behavior
    S[mid_mask] = 0.9 * np.exp(-0.7 * (np.log(x[mid_mask]/0.29))**2)

    return S


def beaming_angle(gamma):
    """
    Characteristic beaming angle for synchrotron radiation.

    theta_beam ~ 1/gamma

    Args:
        gamma: Lorentz factor

    Returns:
        theta: Beaming angle (rad)
    """
    return 1.0 / gamma


def main():
    fig = plt.figure(figsize=(16, 12))

    # Plot 1: Power vs energy (gamma^4 dependence)
    ax1 = fig.add_subplot(2, 2, 1)

    rho = 10.0  # 10 m bending radius

    E_GeV = np.logspace(-2, 2, 100)  # 0.01 to 100 GeV

    P_electron = synchrotron_power_practical(E_GeV, rho, 'electron')
    P_proton = synchrotron_power_practical(E_GeV, rho, 'proton')

    ax1.loglog(E_GeV, P_electron, 'b-', lw=2, label='Electron')
    ax1.loglog(E_GeV, P_proton, 'r-', lw=2, label='Proton')

    # Reference line for gamma^4
    E_ref = np.array([0.1, 10])
    P_ref = P_electron[20] * (E_ref / E_GeV[20])**4
    ax1.loglog(E_ref, P_ref, 'g:', lw=2, label=r'$\propto E^4$')

    ax1.set_xlabel('Particle Energy (GeV)')
    ax1.set_ylabel('Radiated Power (W)')
    ax1.set_title(f'Synchrotron Power: P ~ gamma^4\nBending radius = {rho} m')
    ax1.legend()
    ax1.grid(True, alpha=0.3, which='both')

    # Mark LHC and electron synchrotrons
    E_lhc = 7000  # 7 TeV protons
    if E_lhc / 1000 <= E_GeV[-1]:
        pass  # Outside range

    # Add mass ratio annotation
    ratio = (M_P / M_E)**4
    ax1.text(0.95, 0.3, f'At same E:\n$P_e / P_p = (m_p/m_e)^4$\n$= {ratio:.1e}$',
             transform=ax1.transAxes, ha='right', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Angular distribution (beaming)
    ax2 = fig.add_subplot(2, 2, 2, projection='polar')

    gamma_values = [2, 5, 10, 50]
    colors = ['blue', 'green', 'orange', 'red']

    theta = np.linspace(-np.pi/2, np.pi/2, 200)

    for gamma, color in zip(gamma_values, colors):
        pattern = angular_distribution(theta, gamma)
        pattern = pattern / pattern.max()  # Normalize

        # Plot in polar coordinates (shift theta by pi/2 for forward direction)
        ax2.plot(theta + np.pi/2, pattern, color=color, lw=2,
                label=f'gamma = {gamma}')

    ax2.set_title('Angular Distribution of Radiation\n(Forward = up)')
    ax2.set_theta_zero_location('N')
    ax2.set_thetamin(0)
    ax2.set_thetamax(180)
    ax2.legend(loc='lower right')

    # Add beaming angle annotation
    ax2.annotate(r'$\theta_{beam} \sim 1/\gamma$',
                xy=(np.pi/2, 0.5), fontsize=11)

    # Plot 3: Synchrotron spectrum
    ax3 = fig.add_subplot(2, 2, 3)

    x = np.logspace(-3, 1.5, 200)  # omega / omega_c
    S = spectrum_function(x)

    ax3.loglog(x, S, 'b-', lw=2)

    # Mark critical frequency
    ax3.axvline(x=1, color='red', linestyle='--', lw=2, label=r'$\omega = \omega_c$')

    ax3.set_xlabel(r'$\omega / \omega_c$')
    ax3.set_ylabel(r'$S(\omega/\omega_c)$')
    ax3.set_title('Universal Synchrotron Spectrum\n(Normalized)')
    ax3.legend()
    ax3.grid(True, alpha=0.3, which='both')

    # Add spectrum characteristics
    ax3.text(0.02, 0.3, r'Low $\omega$: $S \propto \omega^{1/3}$'
                        '\n'
                        r'High $\omega$: $S \propto \omega^{1/2} e^{-\omega/\omega_c}$',
             transform=ax3.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 4: Critical energy vs gamma and practical applications
    ax4 = fig.add_subplot(2, 2, 4)

    gamma_range = np.logspace(1, 5, 100)
    rho_values = [1, 10, 100, 1000]  # Different bending radii

    for rho in rho_values:
        E_c = critical_energy(gamma_range, rho)
        E_c_keV = E_c / Q_E / 1000  # Convert to keV

        ax4.loglog(gamma_range, E_c_keV, lw=2, label=f'rho = {rho} m')

    ax4.set_xlabel('Lorentz Factor gamma')
    ax4.set_ylabel('Critical Photon Energy (keV)')
    ax4.set_title('Critical Energy vs Lorentz Factor')
    ax4.legend()
    ax4.grid(True, alpha=0.3, which='both')

    # Mark X-ray regime
    ax4.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
    ax4.axhline(y=100, color='gray', linestyle=':', alpha=0.5)
    ax4.text(gamma_range[-1] * 0.8, 1.5, 'Soft X-rays', fontsize=9)
    ax4.text(gamma_range[-1] * 0.8, 150, 'Hard X-rays', fontsize=9)

    # Add electron energy scale (secondary x-axis)
    ax4_top = ax4.twiny()
    E_electron_MeV = gamma_range * 0.511  # E = gamma * m_e * c^2
    ax4_top.set_xscale('log')
    ax4_top.set_xlim(E_electron_MeV[0], E_electron_MeV[-1])
    ax4_top.set_xlabel('Electron Energy (MeV)')

    # Add physics summary
    fig.text(0.5, 0.02,
             r'Synchrotron Power: $P = \frac{C_\gamma c E^4}{\rho^2} \propto \gamma^4 / \rho^2$, '
             r'Critical Frequency: $\omega_c = \frac{3}{2}\frac{\gamma^3 c}{\rho}$'
             + '\n' +
             r'Beaming Angle: $\theta \sim 1/\gamma$, '
             r'Spectrum: $S(\omega/\omega_c)$ universal function',
             ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.suptitle('Synchrotron Radiation from Relativistic Particles', fontsize=14, y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)

    # Save plot
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'synchrotron_radiation.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
