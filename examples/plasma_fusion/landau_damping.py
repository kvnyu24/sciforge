"""
Experiment 260: Landau Damping

Demonstrates Landau damping - the collisionless damping of plasma
waves due to wave-particle interactions.

Physical concepts:
- Particles near phase velocity exchange energy with wave
- More slow particles than fast -> wave damped
- Damping rate depends on slope of distribution at v_phase
- gamma = -sqrt(pi/8) * (omega_p / k*v_th)^3 * exp(-1/(2*(k*lambda_D)^2))
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.plasma import LandauDamping, PlasmaFrequency, DebyeLength

# Physical constants
e = 1.602e-19
m_e = 9.109e-31
epsilon_0 = 8.854e-12
k_B = 1.381e-23


def maxwellian(v, v_th):
    """Maxwellian distribution function."""
    return np.exp(-v**2 / (2 * v_th**2)) / (np.sqrt(2 * np.pi) * v_th)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plasma parameters
    density = 1e18  # m^-3
    temperature = 1e7  # K (~1 keV)

    plasma = PlasmaFrequency(density)
    omega_p = plasma.angular_frequency

    v_th = np.sqrt(k_B * temperature / m_e)

    debye = DebyeLength(temperature, density)
    lambda_D = debye.length

    landau = LandauDamping(omega_p, v_th)

    # Plot 1: Distribution function and wave-particle resonance
    ax1 = axes[0, 0]

    v = np.linspace(-5 * v_th, 5 * v_th, 500)
    f = maxwellian(v, v_th)

    ax1.plot(v / v_th, f * v_th, 'b-', lw=2, label='$f(v)$ Maxwellian')

    # Mark phase velocities for different k values
    k_values = [0.2 / lambda_D, 0.3 / lambda_D, 0.5 / lambda_D]
    colors = ['red', 'green', 'orange']

    for k, color in zip(k_values, colors):
        omega = landau.dispersion_relation(k)
        v_phase = np.real(omega) / k

        ax1.axvline(x=v_phase / v_th, color=color, linestyle='--', alpha=0.7,
                    label=f'$v_\\phi$ at $k\\lambda_D$ = {k*lambda_D:.1f}')

        # Mark on distribution
        f_at_vph = maxwellian(v_phase, v_th)
        ax1.plot(v_phase / v_th, f_at_vph * v_th, 'o', color=color, markersize=10)

    # Show slope is negative (damping)
    v_sample = 2.5 * v_th
    f_sample = maxwellian(v_sample, v_th)
    df_dv = -v_sample / v_th**2 * f_sample

    ax1.annotate('', xy=(2.8, f_sample * v_th),
                 xytext=(2.2, f_sample * v_th + 0.05),
                 arrowprops=dict(arrowstyle='->', color='gray'))
    ax1.text(2.5, f_sample * v_th + 0.02, '$df/dv < 0$\n(damping)', fontsize=9, ha='center')

    ax1.set_xlabel('Velocity $v / v_{th}$')
    ax1.set_ylabel('$f(v) \\cdot v_{th}$')
    ax1.set_title('Distribution Function and Phase Velocities')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(0, 0.45)

    # Plot 2: Damping rate vs wavenumber
    ax2 = axes[0, 1]

    k = np.linspace(0.1 / lambda_D, 1.0 / lambda_D, 100)
    k_normalized = k * lambda_D

    gamma = np.array([landau.damping_rate(ki) for ki in k])
    gamma_normalized = -gamma / omega_p  # Make positive for plotting

    ax2.semilogy(k_normalized, gamma_normalized, 'b-', lw=2)
    ax2.fill_between(k_normalized, gamma_normalized, 1e-10, alpha=0.3)

    # Mark key region
    ax2.axvline(x=0.3, color='red', linestyle='--', alpha=0.7, label='Weak damping')
    ax2.axvline(x=0.5, color='red', linestyle=':', alpha=0.7, label='Strong damping')

    ax2.set_xlabel('$k \\lambda_D$')
    ax2.set_ylabel('Damping Rate $|\\gamma| / \\omega_p$')
    ax2.set_title('Landau Damping Rate vs Wavenumber')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(1e-5, 1)

    # Plot 3: Wave amplitude evolution
    ax3 = axes[1, 0]

    k_values = [0.2 / lambda_D, 0.3 / lambda_D, 0.5 / lambda_D]
    colors = ['blue', 'green', 'red']

    t_max = 20 / omega_p
    t = np.linspace(0, t_max, 500)

    for k, color in zip(k_values, colors):
        amplitude = landau.wave_amplitude(k, t)
        ax3.plot(t * omega_p, amplitude, color=color, lw=2,
                 label=f'$k\\lambda_D$ = {k*lambda_D:.1f}')

    ax3.set_xlabel('Time ($\\omega_p^{-1}$)')
    ax3.set_ylabel('Wave Amplitude')
    ax3.set_title('Langmuir Wave Damping in Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 20)
    ax3.set_ylim(0, 1.1)

    # Plot 4: Dispersion relation with damping
    ax4 = axes[1, 1]

    k = np.linspace(0.1 / lambda_D, 0.8 / lambda_D, 50)

    omega_r = []
    gamma_list = []

    for ki in k:
        omega_complex = landau.dispersion_relation(ki)
        omega_r.append(np.real(omega_complex))
        gamma_list.append(np.imag(omega_complex))

    omega_r = np.array(omega_r)
    gamma_list = np.array(gamma_list)

    k_normalized = k * lambda_D

    ax4_twin = ax4.twinx()

    line1, = ax4.plot(k_normalized, omega_r / omega_p, 'b-', lw=2, label='$\\omega_r / \\omega_p$')
    line2, = ax4_twin.plot(k_normalized, -gamma_list / omega_p, 'r--', lw=2, label='$|\\gamma| / \\omega_p$')

    # Bohm-Gross without damping
    omega_BG = np.sqrt(omega_p**2 + 3 * k**2 * v_th**2)
    ax4.plot(k_normalized, omega_BG / omega_p, 'b:', lw=1.5, alpha=0.5, label='Bohm-Gross')

    ax4.set_xlabel('$k \\lambda_D$')
    ax4.set_ylabel('$\\omega_r / \\omega_p$', color='blue')
    ax4_twin.set_ylabel('$|\\gamma| / \\omega_p$', color='red')
    ax4.set_title('Complex Dispersion Relation')

    ax4.tick_params(axis='y', labelcolor='blue')
    ax4_twin.tick_params(axis='y', labelcolor='red')

    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper left')

    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 0.8)
    ax4.set_ylim(0.95, 1.3)
    ax4_twin.set_ylim(0, 0.3)

    plt.suptitle('Experiment 260: Landau Damping\n'
                 'Collisionless damping by wave-particle resonance',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'landau_damping.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'landau_damping.png')}")


if __name__ == "__main__":
    main()
