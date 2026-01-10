"""
Experiment 258: Langmuir Wave Dispersion

Demonstrates the dispersion relation for Langmuir (electron plasma)
waves, including thermal corrections.

Bohm-Gross dispersion relation:
omega^2 = omega_p^2 + 3*k^2*v_th^2

Physical concepts:
- Langmuir waves are longitudinal electron oscillations
- Thermal motion adds k-dependent correction
- Phase velocity is always greater than thermal velocity
- Group velocity approaches zero at low k
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.plasma import PlasmaFrequency, DebyeLength

# Physical constants
e = 1.602e-19
m_e = 9.109e-31
epsilon_0 = 8.854e-12
k_B = 1.381e-23
c = 2.998e8


def bohm_gross_dispersion(k, omega_p, v_th):
    """Bohm-Gross dispersion relation for Langmuir waves."""
    return np.sqrt(omega_p**2 + 3 * k**2 * v_th**2)


def phase_velocity(k, omega):
    """Phase velocity."""
    return omega / k


def group_velocity(k, omega_p, v_th):
    """Group velocity for Langmuir waves."""
    omega = bohm_gross_dispersion(k, omega_p, v_th)
    return 3 * k * v_th**2 / omega


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plasma parameters
    density = 1e20  # m^-3
    temperatures = [1e6, 1e7, 1e8]  # K (different thermal velocities)

    plasma = PlasmaFrequency(density)
    omega_p = plasma.angular_frequency

    # Plot 1: Dispersion relation for different temperatures
    ax1 = axes[0, 0]

    colors = ['blue', 'green', 'red']

    for T, color in zip(temperatures, colors):
        v_th = np.sqrt(k_B * T / m_e)
        debye = DebyeLength(T, density)
        lambda_D = debye.length

        k = np.linspace(0.01 / lambda_D, 2 / lambda_D, 200)
        omega = bohm_gross_dispersion(k, omega_p, v_th)

        k_normalized = k * lambda_D
        omega_normalized = omega / omega_p

        T_eV = T * k_B / e
        ax1.plot(k_normalized, omega_normalized, color=color, lw=2,
                 label=f'T = {T_eV:.0f} eV')

    # Cold plasma limit
    ax1.axhline(y=1.0, color='black', linestyle='--', label='Cold plasma ($\\omega = \\omega_p$)')

    ax1.set_xlabel('$k \\lambda_D$')
    ax1.set_ylabel('$\\omega / \\omega_p$')
    ax1.set_title('Bohm-Gross Dispersion: $\\omega^2 = \\omega_p^2 + 3 k^2 v_{th}^2$')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 2)
    ax1.set_ylim(0.8, 2.5)

    # Plot 2: Phase and group velocities
    ax2 = axes[0, 1]

    T = 1e7  # K
    v_th = np.sqrt(k_B * T / m_e)
    debye = DebyeLength(T, density)
    lambda_D = debye.length

    k = np.linspace(0.05 / lambda_D, 2 / lambda_D, 200)
    omega = bohm_gross_dispersion(k, omega_p, v_th)

    v_phase = phase_velocity(k, omega)
    v_group = group_velocity(k, omega_p, v_th)

    k_normalized = k * lambda_D

    ax2.plot(k_normalized, v_phase / v_th, 'b-', lw=2, label='Phase velocity $v_\\phi$')
    ax2.plot(k_normalized, v_group / v_th, 'r--', lw=2, label='Group velocity $v_g$')
    ax2.axhline(y=1.0, color='gray', linestyle=':', alpha=0.7, label='Thermal velocity $v_{th}$')

    ax2.set_xlabel('$k \\lambda_D$')
    ax2.set_ylabel('Velocity / $v_{th}$')
    ax2.set_title('Phase and Group Velocities')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 2)
    ax2.set_ylim(0, 20)

    # Add annotation
    ax2.annotate('$v_\\phi v_g = 3 v_{th}^2$', xy=(1.5, 8), fontsize=12,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 3: Wave propagation simulation
    ax3 = axes[1, 0]

    T = 1e7
    v_th = np.sqrt(k_B * T / m_e)
    debye = DebyeLength(T, density)
    lambda_D = debye.length

    # Two waves with different k values
    k_values = [0.3 / lambda_D, 0.8 / lambda_D]
    x = np.linspace(0, 20 * lambda_D, 500)

    t_values = [0, 1e-12, 2e-12]
    linestyles = ['-', '--', ':']

    for k, color in zip(k_values, ['blue', 'red']):
        omega = bohm_gross_dispersion(k, omega_p, v_th)
        v_ph = omega / k

        for t, ls in zip(t_values, linestyles):
            wave = np.cos(k * x - omega * t)
            label = f'$k\\lambda_D$ = {k*lambda_D:.1f}' if t == 0 else None
            ax3.plot(x / lambda_D, wave + 2 * (k == k_values[1]), color=color,
                     ls=ls, lw=1.5, label=label, alpha=0.8)

    ax3.set_xlabel('$x / \\lambda_D$')
    ax3.set_ylabel('Wave Amplitude (offset)')
    ax3.set_title('Langmuir Wave Propagation at Different Times')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 20)
    ax3.set_yticks([])

    # Add time labels
    ax3.text(19, 0.8, 't = 0', fontsize=9)
    ax3.text(19, 0.5, 't = 1 ps', fontsize=9)
    ax3.text(19, 0.2, 't = 2 ps', fontsize=9)

    # Plot 4: Valid range of dispersion relation
    ax4 = axes[1, 1]

    T = 1e7
    v_th = np.sqrt(k_B * T / m_e)
    debye = DebyeLength(T, density)
    lambda_D = debye.length

    k = np.linspace(0.01 / lambda_D, 3 / lambda_D, 300)
    omega = bohm_gross_dispersion(k, omega_p, v_th)

    k_normalized = k * lambda_D
    omega_normalized = omega / omega_p

    # Valid region (k*lambda_D < 1)
    valid = k_normalized < 1
    invalid = k_normalized >= 1

    ax4.plot(k_normalized[valid], omega_normalized[valid], 'b-', lw=2,
             label='Valid (fluid approximation)')
    ax4.plot(k_normalized[invalid], omega_normalized[invalid], 'r--', lw=2,
             label='Invalid (kinetic effects)')

    ax4.axvline(x=1.0, color='gray', linestyle=':', alpha=0.7)
    ax4.text(1.05, 1.5, '$k\\lambda_D = 1$\n(kinetic regime)', fontsize=9)

    # Shade regions
    ax4.axvspan(0, 1, alpha=0.1, color='blue')
    ax4.axvspan(1, 3, alpha=0.1, color='red')

    ax4.set_xlabel('$k \\lambda_D$')
    ax4.set_ylabel('$\\omega / \\omega_p$')
    ax4.set_title('Validity of Bohm-Gross Relation')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 3)
    ax4.set_ylim(0.8, 3)

    # Add physics notes
    textstr = 'Bohm-Gross valid for:\n$k\\lambda_D \\ll 1$\n(long wavelength limit)'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax4.text(0.5, 2.5, textstr, fontsize=10, bbox=props)

    plt.suptitle('Experiment 258: Langmuir Wave Dispersion\n'
                 'Electron Plasma Waves with Thermal Corrections',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'langmuir_wave_dispersion.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'langmuir_wave_dispersion.png')}")


if __name__ == "__main__":
    main()
