"""
Experiment 96: Dipole radiation pattern.

This example demonstrates the radiation pattern of an oscillating
electric dipole (antenna), showing the angular distribution of
radiated power and the characteristic sin^2(theta) pattern.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Physical constants
C = 2.998e8          # Speed of light (m/s)
MU_0 = 4 * np.pi * 1e-7  # Permeability of free space
EPSILON_0 = 8.854e-12    # Permittivity of free space


def dipole_radiation_pattern(theta):
    """
    Angular radiation pattern of an oscillating electric dipole.

    Power per solid angle: dP/dOmega ~ sin^2(theta)

    Args:
        theta: Polar angle from dipole axis (rad)

    Returns:
        Normalized radiation intensity
    """
    return np.sin(theta)**2


def dipole_radiated_power(p0, omega):
    """
    Total radiated power from oscillating dipole.

    P = (mu_0 * c * omega^4 * p0^2) / (12 * pi)
      = (omega^4 * p0^2) / (12 * pi * epsilon_0 * c^3)

    Args:
        p0: Dipole moment amplitude (C*m)
        omega: Angular frequency (rad/s)

    Returns:
        P: Radiated power (W)
    """
    return MU_0 * C * omega**4 * p0**2 / (12 * np.pi)


def electric_field_radiation_zone(r, theta, p0, omega, t):
    """
    Electric field in the radiation (far) zone.

    E_theta = (mu_0 * omega^2 * p0 * sin(theta) / (4*pi*r)) * cos(omega*t - k*r)

    Args:
        r: Distance from dipole (m)
        theta: Polar angle (rad)
        p0: Dipole moment amplitude (C*m)
        omega: Angular frequency (rad/s)
        t: Time (s)

    Returns:
        E_theta: Theta component of E field (V/m)
    """
    k = omega / C
    return (MU_0 * omega**2 * p0 * np.sin(theta) /
            (4 * np.pi * r)) * np.cos(omega * t - k * r)


def magnetic_field_radiation_zone(r, theta, p0, omega, t):
    """
    Magnetic field in the radiation zone.

    B_phi = E_theta / c
    """
    return electric_field_radiation_zone(r, theta, p0, omega, t) / C


def main():
    # Dipole parameters
    p0 = 1e-9        # Dipole moment amplitude: 1 nC*m
    frequency = 100e6  # 100 MHz (FM radio frequency)
    omega = 2 * np.pi * frequency
    wavelength = C / frequency

    fig = plt.figure(figsize=(16, 12))

    # Plot 1: Polar plot of radiation pattern
    ax1 = fig.add_subplot(2, 2, 1, projection='polar')

    theta = np.linspace(0, 2*np.pi, 360)
    pattern = dipole_radiation_pattern(theta)

    ax1.plot(theta, pattern, 'b-', lw=2)
    ax1.fill(theta, pattern, alpha=0.3)

    ax1.set_title('Dipole Radiation Pattern\n(E-plane, polar plot)')
    ax1.set_ylim(0, 1.2)

    # Mark key features
    ax1.annotate('Null', xy=(0, 0), xytext=(0.3, 0.3),
                textcoords='axes fraction', fontsize=10,
                arrowprops=dict(arrowstyle='->', color='red'))
    ax1.annotate('Maximum', xy=(np.pi/2, 1), xytext=(0.7, 0.7),
                textcoords='axes fraction', fontsize=10,
                arrowprops=dict(arrowstyle='->', color='green'))

    # Plot 2: 3D radiation pattern
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')

    theta_3d = np.linspace(0, np.pi, 50)
    phi_3d = np.linspace(0, 2*np.pi, 100)
    THETA, PHI = np.meshgrid(theta_3d, phi_3d)

    # Pattern is independent of phi for ideal dipole
    R = dipole_radiation_pattern(THETA)

    # Convert to Cartesian
    X = R * np.sin(THETA) * np.cos(PHI)
    Y = R * np.sin(THETA) * np.sin(PHI)
    Z = R * np.cos(THETA)

    # Color by intensity
    colors = plt.cm.hot(R / R.max())

    ax2.plot_surface(X, Y, Z, facecolors=colors, alpha=0.8)

    # Draw dipole axis
    ax2.plot([0, 0], [0, 0], [-1.2, 1.2], 'k-', lw=3, label='Dipole axis')

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('3D Radiation Pattern\n(Toroidal shape)')

    # Plot 3: Field strength vs distance (inverse law)
    ax3 = fig.add_subplot(2, 2, 3)

    r_range = np.logspace(0, 3, 100) * wavelength  # 1 to 1000 wavelengths
    theta_test = np.pi / 2  # Maximum radiation direction

    # Near field (static + induction) ~ 1/r^3, 1/r^2
    # Far field (radiation) ~ 1/r
    E_radiation = np.abs(electric_field_radiation_zone(r_range, theta_test, p0, omega, 0))

    # Approximate near-field terms (order of magnitude)
    k = omega / C
    E_near = MU_0 * C**2 * p0 / (4 * np.pi) * (1/r_range**3 + k/r_range**2)

    ax3.loglog(r_range / wavelength, E_radiation, 'b-', lw=2, label='Radiation field (1/r)')
    ax3.loglog(r_range / wavelength, E_near, 'r--', lw=2, label='Near field (1/r^2, 1/r^3)')

    # Mark transition region
    ax3.axvline(x=1/(2*np.pi), color='green', linestyle=':', lw=2,
                label=r'$r = \lambda/(2\pi)$')

    ax3.set_xlabel('Distance (wavelengths)')
    ax3.set_ylabel('Electric field amplitude (V/m)')
    ax3.set_title('Field Decay: Near vs Far Zone')
    ax3.legend()
    ax3.grid(True, alpha=0.3, which='both')

    # Plot 4: Power and frequency dependence
    ax4 = fig.add_subplot(2, 2, 4)

    # Power vs frequency
    freq_range = np.logspace(6, 10, 100)  # 1 MHz to 10 GHz
    omega_range = 2 * np.pi * freq_range
    power = dipole_radiated_power(p0, omega_range)

    ax4.loglog(freq_range * 1e-6, power, 'b-', lw=2, label='Radiated power')

    # Mark current frequency
    P_current = dipole_radiated_power(p0, omega)
    ax4.plot(frequency * 1e-6, P_current, 'ro', markersize=10,
             label=f'f = {frequency*1e-6:.0f} MHz, P = {P_current:.2e} W')

    ax4.set_xlabel('Frequency (MHz)')
    ax4.set_ylabel('Radiated Power (W)')
    ax4.set_title(r'Radiated Power: $P \propto \omega^4$')
    ax4.legend()
    ax4.grid(True, alpha=0.3, which='both')

    # Add omega^4 reference line
    f_ref = np.array([1e7, 1e9])
    P_ref = 1e-20 * (f_ref)**4
    ax4.loglog(f_ref * 1e-6, P_ref, 'g:', lw=1, label=r'$\propto f^4$')

    # Add physics summary
    fig.text(0.5, 0.02,
             r'Electric Dipole Radiation: $\frac{dP}{d\Omega} \propto \sin^2\theta$, '
             r'$P_{total} = \frac{\mu_0 c \omega^4 p_0^2}{12\pi}$' + '\n'
             f'Parameters: p0 = {p0*1e9:.1f} nC*m, f = {frequency*1e-6:.0f} MHz, '
             f'wavelength = {wavelength:.2f} m',
             ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.suptitle('Electric Dipole Radiation Pattern', fontsize=14, y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)

    # Save plot
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'dipole_radiation_pattern.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
