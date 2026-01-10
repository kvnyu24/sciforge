"""
Experiment 257: Plasma Frequency Oscillations

Demonstrates plasma oscillations - the natural frequency at which
electrons oscillate in a plasma.

omega_p = sqrt(n_e * e^2 / (epsilon_0 * m_e))

Physical concepts:
- Electrons displaced from ions create restoring electric field
- Oscillation frequency depends only on density
- EM waves below omega_p cannot propagate (plasma is opaque)
- Dispersion relation: omega^2 = omega_p^2 + c^2*k^2
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.plasma import PlasmaFrequency

# Physical constants
e = 1.602e-19
m_e = 9.109e-31
epsilon_0 = 8.854e-12
c = 2.998e8


def simulate_oscillation(omega_p, t, amplitude=0.01):
    """Simulate electron density oscillation."""
    return 1.0 + amplitude * np.cos(omega_p * t)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plasma parameters
    densities = [1e16, 1e18, 1e20]  # m^-3

    # Plot 1: Electron oscillation in time
    ax1 = axes[0, 0]

    density = 1e18  # m^-3
    plasma = PlasmaFrequency(density)
    omega_p = plasma.angular_frequency
    period = plasma.period

    t = np.linspace(0, 5 * period, 500)
    n_normalized = simulate_oscillation(omega_p, t)

    ax1.plot(t / period, n_normalized, 'b-', lw=2)
    ax1.fill_between(t / period, 1, n_normalized, where=n_normalized > 1,
                     alpha=0.3, color='blue', label='Electron excess')
    ax1.fill_between(t / period, n_normalized, 1, where=n_normalized < 1,
                     alpha=0.3, color='red', label='Electron deficit')

    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Time (plasma periods)')
    ax1.set_ylabel('$n_e / n_0$')
    ax1.set_title(f'Plasma Oscillations (n = {density:.0e} m$^{{-3}}$, '
                  f'$\\omega_p$ = {omega_p:.2e} rad/s)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 5)

    # Plot 2: Plasma frequency vs density
    ax2 = axes[0, 1]

    n_range = np.logspace(12, 24, 100)  # m^-3
    omega_p = np.sqrt(n_range * e**2 / (epsilon_0 * m_e))
    f_p = omega_p / (2 * np.pi)

    ax2.loglog(n_range, f_p, 'b-', lw=2)

    # Mark key frequencies
    frequencies = {
        'AM Radio': (1e6, 1e15),
        'FM Radio': (1e8, 1e17),
        'Microwave': (1e10, 1e19),
        'Infrared': (1e14, 1e23),
    }

    for name, (f, n) in frequencies.items():
        ax2.axhline(y=f, color='gray', linestyle=':', alpha=0.5)
        ax2.text(1e13, f * 1.2, name, fontsize=8, color='gray')

    # Mark typical plasma densities
    plasma_types = {
        'Ionosphere': 1e12,
        'Solar wind': 1e7,
        'Tokamak': 1e20,
        'ICF': 1e25,
    }

    for name, n in plasma_types.items():
        if 1e12 <= n <= 1e24:
            f = np.sqrt(n * e**2 / (epsilon_0 * m_e)) / (2 * np.pi)
            ax2.plot(n, f, 'ro', markersize=8)
            ax2.annotate(name, (n, f), xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax2.set_xlabel('Electron Density (m$^{-3}$)')
    ax2.set_ylabel('Plasma Frequency (Hz)')
    ax2.set_title('Plasma Frequency: $f_p = \\frac{1}{2\\pi}\\sqrt{\\frac{n_e e^2}{\\epsilon_0 m_e}}$')
    ax2.grid(True, alpha=0.3, which='both')

    # Plot 3: EM wave dispersion relation
    ax3 = axes[1, 0]

    density = 1e20  # m^-3
    plasma = PlasmaFrequency(density)
    omega_p = plasma.angular_frequency

    k = np.linspace(0, 5 * omega_p / c, 200)

    # Dispersion relation
    omega = plasma.dispersion_relation(k)

    # Light line
    omega_light = c * k

    ax3.plot(k * c / omega_p, omega / omega_p, 'b-', lw=2, label='EM wave in plasma')
    ax3.plot(k * c / omega_p, omega_light / omega_p, 'k--', lw=2, label='Light in vacuum')
    ax3.axhline(y=1.0, color='red', linestyle=':', label='$\\omega = \\omega_p$ (cutoff)')

    # Shade evanescent region
    ax3.fill_between([0, 5], 0, 1, alpha=0.1, color='red')
    ax3.text(2.5, 0.5, 'Evanescent\n($\\omega < \\omega_p$)', ha='center', fontsize=10, color='red')

    ax3.set_xlabel('$kc / \\omega_p$')
    ax3.set_ylabel('$\\omega / \\omega_p$')
    ax3.set_title('EM Wave Dispersion: $\\omega^2 = \\omega_p^2 + c^2 k^2$')
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 5)
    ax3.set_ylim(0, 5)
    ax3.set_aspect('equal')

    # Plot 4: Refractive index and group velocity
    ax4 = axes[1, 1]

    omega = np.linspace(1.1 * omega_p, 5 * omega_p, 200)

    # Refractive index
    n_refr = np.real(plasma.refractive_index(omega))

    # Group velocity
    v_g = plasma.group_velocity(omega) / c

    ax4.plot(omega / omega_p, n_refr, 'b-', lw=2, label='Refractive index n')
    ax4.plot(omega / omega_p, v_g, 'r--', lw=2, label='Group velocity $v_g/c$')

    ax4.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax4.axvline(x=1.0, color='gray', linestyle=':', alpha=0.5)

    # Mark asymptotic behavior
    ax4.annotate('$n \\rightarrow 0$ as $\\omega \\rightarrow \\omega_p$',
                 xy=(1.1, 0.1), fontsize=9)
    ax4.annotate('$n \\rightarrow 1$ as $\\omega \\rightarrow \\infty$',
                 xy=(4, 0.95), fontsize=9)

    ax4.set_xlabel('$\\omega / \\omega_p$')
    ax4.set_ylabel('Value')
    ax4.set_title('Refractive Index and Group Velocity')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(1, 5)
    ax4.set_ylim(0, 1.1)

    plt.suptitle('Experiment 257: Plasma Frequency Oscillations\n'
                 '$\\omega_p = \\sqrt{n_e e^2 / \\epsilon_0 m_e}$',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'plasma_frequency_oscillations.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'plasma_frequency_oscillations.png')}")


if __name__ == "__main__":
    main()
