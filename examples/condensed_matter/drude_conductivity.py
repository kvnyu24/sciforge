"""
Experiment 228: Drude Model of Electrical Conductivity

Demonstrates the Drude model for electrical conductivity in metals:
- DC conductivity: sigma = n*e^2*tau/m
- AC conductivity: sigma(omega) = sigma_0 / (1 - i*omega*tau)
- Hall effect from Drude model
- Temperature dependence through scattering time
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


# Physical constants
e = 1.602e-19       # Electron charge (C)
m_e = 9.109e-31     # Electron mass (kg)
k_B = 1.381e-23     # Boltzmann constant (J/K)
hbar = 1.055e-34    # Reduced Planck constant (J*s)


def dc_conductivity(n, tau, m=m_e):
    """
    Drude DC conductivity.

    sigma = n * e^2 * tau / m

    Args:
        n: Carrier density (m^-3)
        tau: Scattering time (s)
        m: Effective mass (kg)

    Returns:
        Conductivity (S/m)
    """
    return n * e**2 * tau / m


def ac_conductivity(omega, sigma_0, tau):
    """
    Drude AC conductivity.

    sigma(omega) = sigma_0 / (1 - i*omega*tau)

    Args:
        omega: Angular frequency (rad/s)
        sigma_0: DC conductivity (S/m)
        tau: Scattering time (s)

    Returns:
        Complex AC conductivity (S/m)
    """
    return sigma_0 / (1 - 1j * omega * tau)


def dielectric_function(omega, omega_p, tau):
    """
    Drude dielectric function.

    epsilon(omega) = 1 - omega_p^2 / (omega^2 + i*omega/tau)

    Args:
        omega: Angular frequency (rad/s)
        omega_p: Plasma frequency (rad/s)
        tau: Scattering time (s)

    Returns:
        Complex dielectric function
    """
    return 1 - omega_p**2 / (omega**2 + 1j * omega / tau)


def plasma_frequency(n, m=m_e):
    """
    Plasma frequency.

    omega_p = sqrt(n * e^2 / (epsilon_0 * m))

    Args:
        n: Carrier density (m^-3)
        m: Effective mass (kg)

    Returns:
        Plasma frequency (rad/s)
    """
    epsilon_0 = 8.854e-12  # Permittivity of free space
    return np.sqrt(n * e**2 / (epsilon_0 * m))


def scattering_time_temperature(tau_0, T, T_D):
    """
    Temperature-dependent scattering time (simplified model).

    For T >> T_D (Debye temperature): tau ~ 1/T (phonon scattering)
    For T << T_D: tau ~ 1/T^5 (Bloch-Gruneisen)

    This uses a simplified interpolation.

    Args:
        tau_0: Reference scattering time at T=T_D
        T: Temperature (K)
        T_D: Debye temperature (K)

    Returns:
        Scattering time (s)
    """
    # Simplified model: interpolate between low-T and high-T regimes
    x = T / T_D
    # Bloch-Gruneisen-like behavior
    if np.isscalar(x):
        if x < 0.1:
            return tau_0 * (T_D / T)**5 * 0.1**4
        else:
            return tau_0 * T_D / T
    else:
        result = np.where(x < 0.1,
                         tau_0 * (T_D / T)**5 * 0.1**4,
                         tau_0 * T_D / T)
        return result


def hall_coefficient(n, carrier_type='electron'):
    """
    Hall coefficient from Drude model.

    R_H = -1/(n*e) for electrons
    R_H = +1/(n*e) for holes

    Args:
        n: Carrier density (m^-3)
        carrier_type: 'electron' or 'hole'

    Returns:
        Hall coefficient (m^3/C)
    """
    if carrier_type == 'electron':
        return -1 / (n * e)
    else:
        return 1 / (n * e)


def hall_mobility(sigma, R_H):
    """
    Hall mobility.

    mu_H = |R_H| * sigma

    Args:
        sigma: Conductivity (S/m)
        R_H: Hall coefficient (m^3/C)

    Returns:
        Hall mobility (m^2/(V*s))
    """
    return np.abs(R_H) * sigma


def magnetoresistance_drude(B, n, tau, m=m_e):
    """
    Magnetoresistance in Drude model.

    In simple Drude model, rho_xx is independent of B (no magnetoresistance).
    rho_xx = m / (n * e^2 * tau)

    But rho_xy (Hall resistivity) = B / (n * e)

    Returns both components.

    Args:
        B: Magnetic field (T)
        n: Carrier density (m^-3)
        tau: Scattering time (s)
        m: Effective mass (kg)

    Returns:
        rho_xx, rho_xy: Resistivity components (Ohm*m)
    """
    rho_xx = m / (n * e**2 * tau)
    rho_xy = B / (n * e)
    return rho_xx, rho_xy


def main():
    """Main simulation and visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Typical metal parameters (copper-like)
    n = 8.5e28      # Carrier density (m^-3)
    tau = 2.5e-14   # Scattering time (s)
    T_D = 343       # Debye temperature for copper (K)

    # Plot 1: AC conductivity
    ax1 = axes[0, 0]

    sigma_0 = dc_conductivity(n, tau)
    omega = np.logspace(10, 17, 500)  # rad/s
    f = omega / (2 * np.pi)

    sigma_ac = ac_conductivity(omega, sigma_0, tau)

    ax1.loglog(f, np.real(sigma_ac), 'b-', lw=2, label=r"Re[$\sigma(\omega)$]")
    ax1.loglog(f, np.abs(np.imag(sigma_ac)), 'r--', lw=2, label=r"|Im[$\sigma(\omega)$]|")

    # Mark characteristic frequency
    f_tau = 1 / (2 * np.pi * tau)
    ax1.axvline(x=f_tau, color='green', linestyle=':', lw=2, label=f'1/(2*pi*tau) = {f_tau:.2e} Hz')

    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Conductivity (S/m)')
    ax1.set_title('Drude AC Conductivity')
    ax1.legend()
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_xlim(1e10, 1e17)

    # Annotate regimes
    ax1.text(1e11, sigma_0 * 0.5, 'DC limit\n(omega*tau << 1)', fontsize=10)
    ax1.text(1e16, sigma_0 * 1e-4, 'High-freq\n(omega*tau >> 1)', fontsize=10)

    # Plot 2: Dielectric function and plasma frequency
    ax2 = axes[0, 1]

    omega_p = plasma_frequency(n)
    f_p = omega_p / (2 * np.pi)

    omega_range = np.linspace(0.01 * omega_p, 3 * omega_p, 500)
    eps = dielectric_function(omega_range, omega_p, tau)

    ax2.plot(omega_range / omega_p, np.real(eps), 'b-', lw=2, label=r"Re[$\epsilon(\omega)$]")
    ax2.plot(omega_range / omega_p, np.imag(eps), 'r--', lw=2, label=r"Im[$\epsilon(\omega)$]")

    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax2.axvline(x=1, color='green', linestyle=':', lw=2, label=r'$\omega = \omega_p$')

    ax2.set_xlabel(r'$\omega / \omega_p$')
    ax2.set_ylabel('Dielectric function')
    ax2.set_title(f'Drude Dielectric Function (f_p = {f_p:.2e} Hz)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 3)
    ax2.set_ylim(-10, 5)

    # Shade metallic region (epsilon < 0)
    omega_norm = omega_range / omega_p
    ax2.fill_between(omega_norm, -10, 5,
                     where=np.real(eps) < 0, alpha=0.2, color='blue',
                     label='Metallic (Re[eps] < 0)')

    # Plot 3: Temperature dependence of conductivity
    ax3 = axes[1, 0]

    T_range = np.linspace(10, 500, 200)
    tau_T = np.array([scattering_time_temperature(tau, T, T_D) for T in T_range])
    sigma_T = dc_conductivity(n, tau_T)

    # Resistivity
    rho_T = 1 / sigma_T

    ax3_rho = ax3
    ax3_rho.semilogy(T_range, rho_T * 1e8, 'b-', lw=2, label='Resistivity')
    ax3_rho.set_xlabel('Temperature (K)')
    ax3_rho.set_ylabel(r'Resistivity ($\mu\Omega\cdot$cm)')
    ax3_rho.set_title('Temperature Dependence of Resistivity')

    # Mark Debye temperature
    ax3_rho.axvline(x=T_D, color='red', linestyle='--', alpha=0.7, label=f'T_D = {T_D} K')

    # Add power law guides
    T_low = T_range[T_range < 50]
    T_high = T_range[T_range > 100]
    ax3_rho.plot(T_low, (T_low/50)**5 * rho_T[T_range < 50][-1] * 1e8,
                'g:', lw=2, alpha=0.7, label=r'$T^5$ (low T)')
    ax3_rho.plot(T_high, (T_high/100) * rho_T[T_range > 100][0] * 1e8,
                'm:', lw=2, alpha=0.7, label=r'$T$ (high T)')

    ax3_rho.legend()
    ax3_rho.grid(True, alpha=0.3)
    ax3_rho.set_xlim(10, 500)

    # Plot 4: Hall effect
    ax4 = axes[1, 1]

    B_range = np.linspace(0, 5, 100)  # Tesla

    # Calculate resistivity components
    rho_xx, rho_xy = magnetoresistance_drude(B_range, n, tau)

    ax4.plot(B_range, rho_xx * 1e8 * np.ones_like(B_range), 'b-', lw=2,
             label=r'$\rho_{xx}$ (longitudinal)')
    ax4.plot(B_range, np.abs(rho_xy) * 1e8, 'r-', lw=2,
             label=r'$|\rho_{xy}|$ (Hall)')

    ax4.set_xlabel('Magnetic Field (T)')
    ax4.set_ylabel(r'Resistivity ($\mu\Omega\cdot$cm)')
    ax4.set_title('Drude Magnetoresistance and Hall Effect')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Add Hall coefficient info
    R_H = hall_coefficient(n)
    mu_H = hall_mobility(sigma_0, R_H)
    ax4.text(0.5, 0.7, f'Hall coefficient R_H = {R_H:.2e} m^3/C\n'
                       f'Hall mobility mu_H = {mu_H*1e4:.1f} cm^2/(V*s)\n'
                       f'n = {n:.2e} m^-3',
             transform=ax4.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('Drude Model of Electrical Conductivity\n'
                 r'$\sigma = ne^2\tau/m$, $\sigma(\omega) = \sigma_0/(1 - i\omega\tau)$',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'drude_conductivity.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'drude_conductivity.png')}")


if __name__ == "__main__":
    main()
