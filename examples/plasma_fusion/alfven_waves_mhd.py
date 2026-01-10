"""
Experiment 261: Alfven Waves MHD

Demonstrates Alfven waves - transverse waves that propagate along
magnetic field lines in a magnetized plasma.

Physical concepts:
- Alfven velocity: v_A = B / sqrt(mu_0 * rho)
- Dispersion relation: omega = k * v_A * cos(theta)
- Group velocity parallel to B field
- Energy carried by both kinetic and magnetic perturbations
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.plasma import AlfvenWave, Magnetosonic

# Physical constants
mu_0 = 4 * np.pi * 1e-7
m_p = 1.673e-27
k_B = 1.381e-23


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plasma parameters (solar corona-like)
    B0 = 1e-3  # Tesla (10 Gauss)
    n = 1e14   # m^-3
    rho = n * m_p  # Mass density

    alfven = AlfvenWave(B0, rho)
    v_A = alfven.alfven_velocity

    # Plot 1: Alfven wave propagation
    ax1 = axes[0, 0]

    # Simulate wave propagation
    L = 10 * v_A * 1e-3  # 10 wavelengths
    x = np.linspace(0, L, 500)
    k = 2 * np.pi / (v_A * 1e-3)  # Wavelength = v_A * 1 ms

    times = [0, 0.25e-3, 0.5e-3, 0.75e-3]  # seconds
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(times)))

    for t, color in zip(times, colors):
        pert = alfven.perturbation(x, t, k, amplitude=0.05)
        delta_vy = pert['delta_vy']
        ax1.plot(x / (v_A * 1e-3), delta_vy / v_A, color=color, lw=2,
                 label=f't = {t*1e3:.2f} ms')

    ax1.set_xlabel('$x / \\lambda$')
    ax1.set_ylabel('$\\delta v_y / v_A$')
    ax1.set_title(f'Alfven Wave Propagation ($v_A$ = {v_A/1e3:.0f} km/s)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 10)

    # Plot 2: Dispersion relation and phase velocity diagram
    ax2 = axes[0, 1]

    theta = np.linspace(0, 2 * np.pi, 200)

    # Alfven wave phase velocity
    v_phase_A = np.abs(np.cos(theta))

    # Also show magnetosonic waves for comparison
    pressure = n * k_B * 1e6  # 1 MK corona
    magnetosonic = Magnetosonic(B0, rho, pressure)

    friedrichs = magnetosonic.friedrichs_diagram()
    v_fast = friedrichs['fast'] / v_A
    v_alfven = friedrichs['alfven'] / v_A
    v_slow = friedrichs['slow'] / v_A
    theta_ms = friedrichs['theta']

    # Polar plot
    ax2 = plt.subplot(2, 2, 2, projection='polar')
    ax2.plot(theta_ms, v_fast, 'r-', lw=2, label='Fast magnetosonic')
    ax2.plot(theta_ms, v_alfven, 'b-', lw=2, label='Alfven')
    ax2.plot(theta_ms, v_slow, 'g-', lw=2, label='Slow magnetosonic')

    ax2.set_title('Phase Velocity Diagram (Friedrichs)\n$v_\\phi / v_A$', y=1.1)
    ax2.legend(loc='lower right', bbox_to_anchor=(1.3, -0.1))
    ax2.set_theta_zero_location('E')

    # Mark B field direction
    ax2.annotate('B', xy=(0, 1.5), fontsize=12, fontweight='bold')
    ax2.plot([0, 0], [0, 1.5], 'k-', lw=2)
    ax2.annotate('', xy=(0, 1.4), xytext=(0, 1.0),
                 arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Plot 3: MHD wave speeds vs plasma beta
    ax3 = axes[1, 0]

    beta = np.logspace(-2, 2, 100)

    # Sound speed normalized to Alfven speed
    # c_s / v_A = sqrt(gamma * beta / 2)
    gamma = 5 / 3
    cs_over_vA = np.sqrt(gamma * beta / 2)

    # Fast and slow wave speeds at theta = 45 degrees
    cos45 = np.cos(np.pi / 4)
    for angle in [0, np.pi / 4, np.pi / 2]:
        v_fast = np.sqrt(0.5 * (1 + cs_over_vA**2 +
                               np.sqrt((1 + cs_over_vA**2)**2 -
                                       4 * cs_over_vA**2 * np.cos(angle)**2)))
        v_slow = np.sqrt(0.5 * (1 + cs_over_vA**2 -
                                np.sqrt((1 + cs_over_vA**2)**2 -
                                        4 * cs_over_vA**2 * np.cos(angle)**2)))
        v_alfven = np.abs(np.cos(angle)) * np.ones_like(beta)

        angle_deg = int(np.degrees(angle))
        if angle == 0:
            ax3.loglog(beta, v_fast, 'r-', lw=2, label=f'Fast ($\\theta = {angle_deg}$)')
            ax3.loglog(beta, v_slow, 'g-', lw=2, label=f'Slow ($\\theta = {angle_deg}$)')

    ax3.loglog(beta, cs_over_vA, 'k--', lw=2, label='Sound speed $c_s$')
    ax3.axhline(y=1.0, color='blue', linestyle=':', alpha=0.7, label='Alfven speed $v_A$')

    # Mark beta = 1
    ax3.axvline(x=1.0, color='gray', linestyle=':', alpha=0.7)
    ax3.text(1.1, 0.1, '$\\beta = 1$', fontsize=9)

    ax3.set_xlabel('Plasma Beta $\\beta = 2\\mu_0 p / B^2$')
    ax3.set_ylabel('Wave Speed / $v_A$')
    ax3.set_title('MHD Wave Speeds vs Plasma Beta (parallel propagation)')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3, which='both')
    ax3.set_xlim(0.01, 100)
    ax3.set_ylim(0.01, 10)

    # Plot 4: Energy partition in Alfven wave
    ax4 = axes[1, 1]

    # In Alfven wave, kinetic and magnetic energy are equal
    phase = np.linspace(0, 4 * np.pi, 200)

    KE = 0.5 * np.cos(phase)**2  # Kinetic energy
    ME = 0.5 * np.cos(phase)**2  # Magnetic energy
    total = KE + ME

    ax4.plot(phase / np.pi, KE, 'b-', lw=2, label='Kinetic energy $\\frac{1}{2}\\rho (\\delta v)^2$')
    ax4.plot(phase / np.pi, ME, 'r--', lw=2, label='Magnetic energy $\\frac{(\\delta B)^2}{2\\mu_0}$')
    ax4.plot(phase / np.pi, total, 'k:', lw=2, label='Total energy')

    ax4.fill_between(phase / np.pi, 0, KE, alpha=0.2, color='blue')
    ax4.fill_between(phase / np.pi, 0, ME, alpha=0.2, color='red')

    ax4.set_xlabel('Phase ($\\pi$ radians)')
    ax4.set_ylabel('Energy Density (normalized)')
    ax4.set_title('Energy Partition in Alfven Wave')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 4)
    ax4.set_ylim(0, 1.2)

    # Add equipartition annotation
    ax4.annotate('Equipartition:\n$E_K = E_B$', xy=(2, 0.3), fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Experiment 261: Alfven Waves in MHD\n'
                 '$v_A = B / \\sqrt{\\mu_0 \\rho}$, $\\omega = k v_A \\cos\\theta$',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'alfven_waves_mhd.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'alfven_waves_mhd.png')}")


if __name__ == "__main__":
    main()
