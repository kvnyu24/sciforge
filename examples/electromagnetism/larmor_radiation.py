"""
Experiment 97: Larmor radiation power.

This example demonstrates the Larmor formula for radiation from
accelerating charged particles, including applications to
synchrotron radiation and cyclotron motion.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt


# Physical constants
C = 2.998e8          # Speed of light (m/s)
Q_E = 1.602e-19      # Elementary charge (C)
M_E = 9.109e-31      # Electron mass (kg)
M_P = 1.673e-27      # Proton mass (kg)
EPSILON_0 = 8.854e-12  # Permittivity of free space


def larmor_power_nonrelativistic(q, a):
    """
    Larmor formula for radiated power (non-relativistic).

    P = q^2 * a^2 / (6 * pi * epsilon_0 * c^3)

    Args:
        q: Charge (C)
        a: Acceleration (m/s^2)

    Returns:
        P: Radiated power (W)
    """
    return q**2 * a**2 / (6 * np.pi * EPSILON_0 * C**3)


def larmor_power_relativistic(q, a_perp, a_para, gamma):
    """
    Relativistic Larmor formula.

    P = (q^2 * gamma^4) / (6 * pi * epsilon_0 * c^3) * (a_perp^2 + gamma^2 * a_para^2)

    where a_perp is perpendicular acceleration and a_para is parallel.

    Args:
        q: Charge (C)
        a_perp: Perpendicular acceleration (m/s^2)
        a_para: Parallel acceleration (m/s^2)
        gamma: Lorentz factor

    Returns:
        P: Radiated power (W)
    """
    return q**2 * gamma**4 / (6 * np.pi * EPSILON_0 * C**3) * (a_perp**2 + gamma**2 * a_para**2)


def synchrotron_power(q, m, E, B):
    """
    Synchrotron radiation power for circular motion.

    P = (q^4 * B^2 * gamma^2 * c) / (6 * pi * epsilon_0 * m^4 * c^8)
      = (C_gamma / (2*pi)) * c * E^4 / (m^4 * c^8 * rho^2)

    For circular motion, a_perp = v^2/rho = qvB/m (cyclotron motion)

    Args:
        q: Charge (C)
        m: Mass (kg)
        E: Total energy (J)
        B: Magnetic field (T)

    Returns:
        P: Radiated power (W)
    """
    gamma = E / (m * C**2)
    v = C * np.sqrt(1 - 1/gamma**2)
    rho = gamma * m * v / (np.abs(q) * B)  # Radius of curvature
    a = v**2 / rho

    return larmor_power_nonrelativistic(q, a) * gamma**4


def cyclotron_radiation(q, m, B, v_perp):
    """
    Power radiated in cyclotron motion.

    Args:
        q: Charge (C)
        m: Mass (kg)
        B: Magnetic field (T)
        v_perp: Velocity perpendicular to B (m/s)
    """
    omega_c = np.abs(q) * B / m  # Cyclotron frequency
    a = omega_c * v_perp  # Centripetal acceleration
    return larmor_power_nonrelativistic(q, a)


def main():
    fig = plt.figure(figsize=(16, 12))

    # Plot 1: Power vs acceleration (non-relativistic)
    ax1 = fig.add_subplot(2, 2, 1)

    a_range = np.logspace(10, 22, 100)  # m/s^2

    # Different particles
    particles = [
        ('Electron', Q_E, M_E, 'blue'),
        ('Proton', Q_E, M_P, 'red'),
    ]

    for name, q, m, color in particles:
        P = larmor_power_nonrelativistic(q, a_range)
        ax1.loglog(a_range, P, color=color, lw=2, label=name)

    ax1.set_xlabel('Acceleration (m/s^2)')
    ax1.set_ylabel('Radiated Power (W)')
    ax1.set_title('Larmor Radiation: Power vs Acceleration\n(Non-relativistic)')
    ax1.legend()
    ax1.grid(True, alpha=0.3, which='both')

    # Mark typical accelerations
    a_earth = 9.8
    ax1.axvline(x=a_earth, color='gray', linestyle=':', alpha=0.5)
    ax1.text(a_earth * 2, 1e-60, 'g (Earth)', fontsize=8, rotation=90)

    # Plot 2: Synchrotron radiation vs energy
    ax2 = fig.add_subplot(2, 2, 2)

    # Energy in GeV
    E_GeV = np.logspace(-1, 2, 100)  # 0.1 to 100 GeV
    E_joules = E_GeV * 1e9 * Q_E

    B = 1.0  # 1 Tesla magnetic field

    # Electron synchrotron power
    P_electron = np.array([synchrotron_power(Q_E, M_E, E, B) for E in E_joules])

    # Proton synchrotron power
    P_proton = np.array([synchrotron_power(Q_E, M_P, E, B) for E in E_joules])

    ax2.loglog(E_GeV, P_electron, 'b-', lw=2, label='Electron')
    ax2.loglog(E_GeV, P_proton, 'r-', lw=2, label='Proton')

    ax2.set_xlabel('Particle Energy (GeV)')
    ax2.set_ylabel('Synchrotron Power (W)')
    ax2.set_title(f'Synchrotron Radiation (B = {B} T)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')

    # Mark mass ratio effect
    ratio = (M_P / M_E)**4
    ax2.text(0.5, 0.1, f'P_e / P_p = (m_p/m_e)^4 = {ratio:.0e}',
             transform=ax2.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 3: Cyclotron radiation
    ax3 = fig.add_subplot(2, 2, 3)

    B_range = np.logspace(-2, 2, 100)  # 0.01 to 100 T
    v_perp = 0.1 * C  # 10% of light speed

    P_cyclotron_e = cyclotron_radiation(Q_E, M_E, B_range, v_perp)
    P_cyclotron_p = cyclotron_radiation(Q_E, M_P, B_range, v_perp)

    ax3.loglog(B_range, P_cyclotron_e, 'b-', lw=2, label='Electron')
    ax3.loglog(B_range, P_cyclotron_p, 'r-', lw=2, label='Proton')

    ax3.set_xlabel('Magnetic Field (T)')
    ax3.set_ylabel('Radiated Power (W)')
    ax3.set_title(f'Cyclotron Radiation (v_perp = 0.1c)')
    ax3.legend()
    ax3.grid(True, alpha=0.3, which='both')

    # Mark typical field strengths
    ax3.axvline(x=1, color='gray', linestyle=':', alpha=0.5)
    ax3.text(1.2, 1e-10, 'MRI (1T)', fontsize=8)
    ax3.axvline(x=10, color='gray', linestyle=':', alpha=0.5)
    ax3.text(12, 1e-10, 'Strong lab (10T)', fontsize=8)

    # Plot 4: Relativistic enhancement
    ax4 = fig.add_subplot(2, 2, 4)

    gamma_range = np.logspace(0, 4, 100)  # 1 to 10000
    a0 = 1e15  # Reference acceleration (m/s^2)

    # Non-relativistic power
    P0 = larmor_power_nonrelativistic(Q_E, a0)

    # Relativistic enhancement for circular motion (perpendicular acceleration)
    P_rel_perp = larmor_power_relativistic(Q_E, a0, 0, gamma_range)

    # Relativistic enhancement for linear acceleration
    P_rel_para = larmor_power_relativistic(Q_E, 0, a0, gamma_range)

    ax4.loglog(gamma_range, P_rel_perp / P0, 'b-', lw=2,
               label=r'Circular motion ($a_\perp$): $\gamma^4$')
    ax4.loglog(gamma_range, P_rel_para / P0, 'r-', lw=2,
               label=r'Linear motion ($a_\parallel$): $\gamma^6$')

    # Reference lines
    ax4.loglog(gamma_range, gamma_range**4, 'b:', lw=1, alpha=0.5)
    ax4.loglog(gamma_range, gamma_range**6, 'r:', lw=1, alpha=0.5)

    ax4.set_xlabel('Lorentz Factor gamma')
    ax4.set_ylabel('P / P_nonrel')
    ax4.set_title('Relativistic Enhancement of Radiation')
    ax4.legend()
    ax4.grid(True, alpha=0.3, which='both')

    # Mark electron at various energies
    E_values = [0.511, 1, 10, 100]  # MeV
    for E in E_values:
        gamma = E / 0.511
        if gamma > 1 and gamma < gamma_range[-1]:
            ax4.axvline(x=gamma, color='gray', linestyle=':', alpha=0.3)
            ax4.text(gamma, 1e-1, f'{E}MeV', fontsize=7, rotation=90)

    # Add physics summary
    fig.text(0.5, 0.02,
             r'Larmor Formula: $P = \frac{q^2 a^2}{6\pi\epsilon_0 c^3}$ (non-rel), '
             r'$P = \frac{q^2 \gamma^4}{6\pi\epsilon_0 c^3}(a_\perp^2 + \gamma^2 a_\parallel^2)$ (rel)'
             + '\n' +
             r'Synchrotron: $P \propto \gamma^4 / \rho^2 \propto E^4 / m^4$',
             ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.suptitle('Larmor Radiation from Accelerating Charges', fontsize=14, y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)

    # Save plot
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'larmor_radiation.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
