"""
Experiment 229: 1D Phonon Dispersion

Demonstrates phonon dispersion relations in 1D lattices, including:
- Monatomic chain (acoustic branch only)
- Diatomic chain (acoustic and optical branches)
- Group velocity and density of states
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def monatomic_dispersion(k, a, M, K):
    """
    Phonon dispersion for 1D monatomic chain.

    omega = 2*sqrt(K/M) * |sin(ka/2)|

    Args:
        k: Wavevector
        a: Lattice constant
        M: Atomic mass
        K: Spring constant

    Returns:
        omega: Angular frequency
    """
    omega_max = 2 * np.sqrt(K / M)
    return omega_max * np.abs(np.sin(k * a / 2))


def diatomic_dispersion(k, a, M1, M2, K):
    """
    Phonon dispersion for 1D diatomic chain.

    Returns both acoustic and optical branches.

    Args:
        k: Wavevector
        a: Lattice constant (unit cell = 2a)
        M1, M2: Masses of the two atoms
        K: Spring constant

    Returns:
        omega_acoustic, omega_optical: Two branches
    """
    # Reduced mass terms
    mu = M1 * M2 / (M1 + M2)
    M_sum = M1 + M2

    # Characteristic frequency
    omega_0 = np.sqrt(K / mu)

    # Discriminant
    sin_term = np.sin(k * a / 2)**2
    discriminant = np.sqrt(1 - 4 * M1 * M2 / M_sum**2 * sin_term)

    # Two branches
    omega_optical = omega_0 * np.sqrt(1 + discriminant)
    omega_acoustic = omega_0 * np.sqrt(1 - discriminant)

    return omega_acoustic, omega_optical


def group_velocity_monatomic(k, a, M, K):
    """
    Group velocity for monatomic chain.

    v_g = d(omega)/dk = (a/2)*sqrt(K/M)*cos(ka/2)*sign(sin(ka/2))
    """
    omega_max = 2 * np.sqrt(K / M)
    return omega_max * a / 2 * np.cos(k * a / 2) * np.sign(np.sin(k * a / 2))


def phonon_dos_1d(omega, omega_max):
    """
    Phonon density of states for 1D monatomic chain.

    g(omega) = 2/(pi*sqrt(omega_max^2 - omega^2))

    Args:
        omega: Frequency array
        omega_max: Maximum frequency

    Returns:
        DOS g(omega)
    """
    g = np.zeros_like(omega)
    mask = (omega > 0) & (omega < omega_max)
    g[mask] = 2 / (np.pi * np.sqrt(omega_max**2 - omega[mask]**2 + 1e-10))
    return g


def main():
    """Main simulation and visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Parameters
    a = 1.0      # Lattice constant
    M = 1.0      # Mass (monatomic)
    K = 1.0      # Spring constant

    # For diatomic
    M1 = 1.0
    M2 = 2.0     # Different masses

    # k in first Brillouin zone
    k = np.linspace(-np.pi/a, np.pi/a, 500)

    # Plot 1: Monatomic chain dispersion
    ax1 = axes[0, 0]

    omega_mono = monatomic_dispersion(k, a, M, K)
    omega_max = 2 * np.sqrt(K / M)

    ax1.plot(k * a / np.pi, omega_mono, 'b-', lw=2, label='Acoustic')
    ax1.axhline(y=omega_max, color='gray', linestyle='--', alpha=0.5,
                label=f'$\\omega_{{max}} = 2\\sqrt{{K/M}}$')

    ax1.set_xlabel('ka / pi')
    ax1.set_ylabel('omega (sqrt(K/M))')
    ax1.set_title('Monatomic Chain: Acoustic Branch Only')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-1, 1)

    # Mark zone boundary
    ax1.axvline(x=-1, color='green', linestyle=':', alpha=0.5)
    ax1.axvline(x=1, color='green', linestyle=':', alpha=0.5)

    # Plot 2: Diatomic chain dispersion
    ax2 = axes[0, 1]

    omega_acoustic, omega_optical = diatomic_dispersion(k, a, M1, M2, K)

    ax2.plot(k * a / np.pi, omega_acoustic, 'b-', lw=2, label='Acoustic')
    ax2.plot(k * a / np.pi, omega_optical, 'r-', lw=2, label='Optical')

    # Mark gap
    gap_bottom = np.max(omega_acoustic)
    gap_top = np.min(omega_optical)
    ax2.axhspan(gap_bottom, gap_top, alpha=0.2, color='yellow', label='Phonon gap')

    ax2.set_xlabel('ka / pi')
    ax2.set_ylabel('omega (arb. units)')
    ax2.set_title(f'Diatomic Chain: M2/M1 = {M2/M1}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-1, 1)

    # Add labels for gap edges
    ax2.text(0.6, gap_bottom - 0.1, f'Gap: {gap_top - gap_bottom:.2f}', fontsize=10)

    # Plot 3: Effect of mass ratio
    ax3 = axes[1, 0]

    mass_ratios = [1.0, 1.5, 2.0, 3.0, 5.0]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(mass_ratios)))

    for ratio, color in zip(mass_ratios, colors):
        M2_var = M1 * ratio
        omega_ac, omega_op = diatomic_dispersion(k, a, M1, M2_var, K)

        if ratio == 1.0:
            # Monatomic limit
            ax3.plot(k * a / np.pi, omega_ac, color=color, lw=2,
                    label=f'M2/M1 = {ratio:.1f} (monatomic)')
        else:
            ax3.plot(k * a / np.pi, omega_ac, color=color, lw=2,
                    label=f'M2/M1 = {ratio:.1f}')
            ax3.plot(k * a / np.pi, omega_op, color=color, lw=2, linestyle='--')

    ax3.set_xlabel('ka / pi')
    ax3.set_ylabel('omega (arb. units)')
    ax3.set_title('Effect of Mass Ratio on Dispersion')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-1, 1)

    # Plot 4: Group velocity and DOS
    ax4 = axes[1, 1]

    # Group velocity
    v_g = group_velocity_monatomic(k, a, M, K)

    ax4_twin = ax4.twinx()

    l1, = ax4.plot(k * a / np.pi, v_g, 'b-', lw=2, label='Group velocity')
    ax4.set_xlabel('ka / pi')
    ax4.set_ylabel('Group velocity v_g (a*sqrt(K/M))', color='blue')
    ax4.tick_params(axis='y', labelcolor='blue')
    ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    # DOS on secondary axis
    omega_dos = np.linspace(0, omega_max * 0.999, 100)
    g = phonon_dos_1d(omega_dos, omega_max)

    l2, = ax4_twin.plot(omega_dos / omega_max, g * omega_max, 'r-', lw=2, label='DOS')
    ax4_twin.set_ylabel('DOS g(omega)', color='red')
    ax4_twin.tick_params(axis='y', labelcolor='red')
    ax4_twin.fill_between(omega_dos / omega_max, 0, g * omega_max, alpha=0.2, color='red')

    ax4.set_title('Group Velocity and Density of States')
    ax4.legend([l1, l2], ['Group velocity', 'DOS g(omega)'], loc='upper right')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-1, 1)

    plt.suptitle('1D Phonon Dispersion Relations\n'
                 'Ball-and-spring model for lattice vibrations',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'phonon_dispersion_1d.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'phonon_dispersion_1d.png')}")


if __name__ == "__main__":
    main()
