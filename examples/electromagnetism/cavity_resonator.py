"""
Experiment 99: Cavity resonator.

This example demonstrates electromagnetic cavity resonators, showing
resonant modes in 3D cavities, quality factor Q, mode density,
and hints at the Purcell effect for cavity QED.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Physical constants
C = 2.998e8          # Speed of light (m/s)
MU_0 = 4 * np.pi * 1e-7  # Permeability of free space (H/m)
EPSILON_0 = 8.854e-12    # Permittivity of free space (F/m)
HBAR = 1.055e-34     # Reduced Planck constant (J*s)


def resonant_frequency_rectangular(m, n, p, a, b, c):
    """
    Calculate resonant frequency for a rectangular cavity.

    f_mnp = (c/2) * sqrt((m/a)^2 + (n/b)^2 + (p/c)^2)

    where a, b, c are the cavity dimensions.

    Args:
        m, n, p: Mode indices (integers >= 0)
        a, b, c: Cavity dimensions (m)

    Returns:
        f: Resonant frequency (Hz)
    """
    return (C / 2) * np.sqrt((m / a)**2 + (n / b)**2 + (p / c)**2)


def resonant_frequency_cylindrical(m, n, p, R, L, mode_type='TM'):
    """
    Calculate resonant frequency for a cylindrical cavity.

    For TM modes: f = (c/2pi) * sqrt((x_mn/R)^2 + (p*pi/L)^2)
    For TE modes: f = (c/2pi) * sqrt((x'_mn/R)^2 + (p*pi/L)^2)

    where x_mn is the nth root of J_m(x) and x'_mn is the nth root of J'_m(x).

    Args:
        m: Azimuthal mode number
        n: Radial mode number (>= 1)
        p: Axial mode number (>= 0 for TM, >= 1 for TE with some conditions)
        R: Cavity radius (m)
        L: Cavity length (m)
        mode_type: 'TM' or 'TE'

    Returns:
        f: Resonant frequency (Hz)
    """
    # Bessel function roots (approximate values for first few)
    # J_m(x_mn) = 0 roots:
    j_roots = {
        (0, 1): 2.405, (0, 2): 5.520, (0, 3): 8.654,
        (1, 1): 3.832, (1, 2): 7.016, (1, 3): 10.174,
        (2, 1): 5.136, (2, 2): 8.417, (2, 3): 11.620,
    }

    # J'_m(x'_mn) = 0 roots:
    jp_roots = {
        (0, 1): 3.832, (0, 2): 7.016, (0, 3): 10.174,
        (1, 1): 1.841, (1, 2): 5.331, (1, 3): 8.536,
        (2, 1): 3.054, (2, 2): 6.706, (2, 3): 9.970,
    }

    if mode_type == 'TM':
        x_mn = j_roots.get((m, n), 2.405 + n * np.pi)
    else:  # TE
        x_mn = jp_roots.get((m, n), 1.841 + n * np.pi)

    k_perp = x_mn / R
    k_z = p * np.pi / L

    return C / (2 * np.pi) * np.sqrt(k_perp**2 + k_z**2)


def quality_factor(f, sigma, V, S):
    """
    Calculate quality factor Q for a cavity with conducting walls.

    Q = omega * U / P_loss = omega * (stored energy) / (power loss)

    For a cavity with skin depth delta:
    Q ~ V / (delta * S) ~ (omega * mu_0 * sigma)^(1/2) * V / S

    Args:
        f: Resonant frequency (Hz)
        sigma: Wall conductivity (S/m)
        V: Cavity volume (m^3)
        S: Cavity surface area (m^2)

    Returns:
        Q: Quality factor
    """
    omega = 2 * np.pi * f
    # Skin depth
    delta = np.sqrt(2 / (omega * MU_0 * sigma))
    # Approximate Q
    return V / (delta * S)


def mode_density_3d(f, V):
    """
    Calculate mode density (number of modes per unit frequency) for a 3D cavity.

    D(f) = dN/df = (8 * pi * V / c^3) * f^2

    This is the Weyl formula for electromagnetic modes.

    Args:
        f: Frequency (Hz)
        V: Cavity volume (m^3)

    Returns:
        D: Mode density (modes/Hz)
    """
    return 8 * np.pi * V * f**2 / C**3


def mode_count_3d(f, V):
    """
    Estimate total number of modes below frequency f.

    N(f) = (8 * pi * V / (3 * c^3)) * f^3

    Args:
        f: Frequency (Hz)
        V: Cavity volume (m^3)

    Returns:
        N: Approximate number of modes
    """
    return 8 * np.pi * V * f**3 / (3 * C**3)


def purcell_factor(Q, V, wavelength):
    """
    Calculate Purcell enhancement factor.

    F_p = (3 * Q * lambda^3) / (4 * pi^2 * V)

    This describes the enhancement of spontaneous emission rate
    for an atom in a cavity compared to free space.

    Args:
        Q: Quality factor
        V: Mode volume (m^3)
        wavelength: Transition wavelength (m)

    Returns:
        F_p: Purcell factor
    """
    return 3 * Q * wavelength**3 / (4 * np.pi**2 * V)


def rectangular_cavity_mode_field(x, y, z, m, n, p, a, b, c):
    """
    Electric field pattern for TM_mnp mode in rectangular cavity.

    E_z = E_0 * sin(m*pi*x/a) * sin(n*pi*y/b) * cos(p*pi*z/c)

    Args:
        x, y, z: Position (m)
        m, n, p: Mode indices
        a, b, c: Cavity dimensions (m)

    Returns:
        E_z: Longitudinal electric field (normalized)
    """
    return np.sin(m * np.pi * x / a) * np.sin(n * np.pi * y / b) * np.cos(p * np.pi * z / c)


def main():
    fig = plt.figure(figsize=(16, 12))

    # Rectangular cavity dimensions (microwave oven size)
    a = 0.3   # 30 cm
    b = 0.25  # 25 cm
    c = 0.2   # 20 cm
    V_rect = a * b * c
    S_rect = 2 * (a * b + b * c + c * a)

    # Plot 1: Mode spectrum for rectangular cavity
    ax1 = fig.add_subplot(2, 2, 1)

    # Calculate first several modes
    modes = []
    for m in range(4):
        for n in range(4):
            for p in range(4):
                if m == 0 and n == 0:
                    continue  # Need at least two non-zero indices for TM
                if m == 0 and p == 0:
                    continue
                if n == 0 and p == 0:
                    continue

                f = resonant_frequency_rectangular(m, n, p, a, b, c)
                mode_name = f'{m}{n}{p}'
                modes.append((mode_name, f))

    # Sort by frequency
    modes.sort(key=lambda x: x[1])
    modes = modes[:15]  # First 15 modes

    names = [m[0] for m in modes]
    freqs = [m[1] / 1e9 for m in modes]

    ax1.barh(range(len(modes)), freqs, color='blue', alpha=0.7)
    ax1.set_yticks(range(len(modes)))
    ax1.set_yticklabels([f'TM{n}' for n in names])
    ax1.set_xlabel('Resonant Frequency (GHz)')
    ax1.set_title(f'Rectangular Cavity Mode Spectrum\n'
                  f'{a*100:.0f} x {b*100:.0f} x {c*100:.0f} cm')
    ax1.grid(True, alpha=0.3, axis='x')

    # Mark 2.45 GHz (microwave oven frequency)
    ax1.axvline(x=2.45, color='red', linestyle='--', lw=2, label='2.45 GHz (microwave)')
    ax1.legend()

    # Plot 2: Mode density and count
    ax2 = fig.add_subplot(2, 2, 2)

    f_range = np.linspace(0.1e9, 5e9, 200)

    density = mode_density_3d(f_range, V_rect)
    count = mode_count_3d(f_range, V_rect)

    ax2_twin = ax2.twinx()

    line1, = ax2.plot(f_range / 1e9, density * 1e9, 'b-', lw=2, label='Mode density')
    line2, = ax2_twin.plot(f_range / 1e9, count, 'r-', lw=2, label='Mode count')

    ax2.set_xlabel('Frequency (GHz)')
    ax2.set_ylabel('Mode Density (modes/GHz)', color='b')
    ax2_twin.set_ylabel('Total Mode Count', color='r')
    ax2.set_title('Mode Density and Count\n(Weyl formula)')

    ax2.tick_params(axis='y', labelcolor='b')
    ax2_twin.tick_params(axis='y', labelcolor='r')

    ax2.legend(handles=[line1, line2], loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Add formula annotation
    ax2.text(0.95, 0.3, r'$D(f) = \frac{8\pi V}{c^3} f^2$'
                        '\n'
                        r'$N(f) = \frac{8\pi V}{3c^3} f^3$',
             transform=ax2.transAxes, ha='right', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 3: Quality factor vs frequency
    ax3 = fig.add_subplot(2, 2, 3)

    # Different materials
    materials = [
        ('Copper', 5.8e7, 'orange'),
        ('Aluminum', 3.5e7, 'gray'),
        ('Silver', 6.3e7, 'silver'),
    ]

    f_range = np.logspace(8, 11, 100)  # 100 MHz to 100 GHz

    for name, sigma, color in materials:
        Q = quality_factor(f_range, sigma, V_rect, S_rect)
        ax3.loglog(f_range / 1e9, Q, color=color, lw=2, label=name)

    ax3.set_xlabel('Frequency (GHz)')
    ax3.set_ylabel('Quality Factor Q')
    ax3.set_title('Quality Factor vs Frequency\n(Conductor loss only)')
    ax3.legend()
    ax3.grid(True, alpha=0.3, which='both')

    # Add Q ~ sqrt(f) reference line
    Q_ref = quality_factor(1e9, 5.8e7, V_rect, S_rect)
    ax3.loglog(f_range / 1e9, Q_ref * np.sqrt(f_range / 1e9), 'k:', lw=1,
              label=r'$\propto \sqrt{f}$')

    # Plot 4: Purcell factor for cavity QED
    ax4 = fig.add_subplot(2, 2, 4)

    # Small optical cavity parameters
    wavelength = 800e-9  # 800 nm (typical atomic transition)
    V_mode_range = np.logspace(-18, -12, 100)  # Mode volume from 1 um^3 to 1 mm^3

    Q_values = [100, 1000, 10000, 100000]
    colors = ['blue', 'green', 'orange', 'red']

    for Q, color in zip(Q_values, colors):
        F_p = purcell_factor(Q, V_mode_range, wavelength)
        ax4.loglog(V_mode_range * 1e18, F_p, color=color, lw=2,
                  label=f'Q = {Q:,}')

    ax4.set_xlabel(r'Mode Volume ($\mu m^3$)')
    ax4.set_ylabel('Purcell Factor F_p')
    ax4.set_title(f'Purcell Enhancement (lambda = {wavelength*1e9:.0f} nm)')
    ax4.legend()
    ax4.grid(True, alpha=0.3, which='both')

    # Mark strong coupling regime
    ax4.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax4.text(1e-17 * 1e18, 1.5, 'F_p = 1 (no enhancement)', fontsize=9)

    # Add Purcell formula
    ax4.text(0.95, 0.95, r'$F_p = \frac{3Q\lambda^3}{4\pi^2 V}$',
             transform=ax4.transAxes, ha='right', va='top', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Add physics summary
    fig.text(0.5, 0.02,
             r'Rectangular cavity: $f_{mnp} = \frac{c}{2}\sqrt{(m/a)^2 + (n/b)^2 + (p/c)^2}$'
             + '\n' +
             r'Quality factor: $Q = \omega U / P_{loss}$, '
             r'Purcell effect: $\Gamma_{cav}/\Gamma_0 = F_p = 3Q\lambda^3/(4\pi^2 V)$',
             ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.suptitle('Electromagnetic Cavity Resonators', fontsize=14, y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)

    # Save plot
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'cavity_resonator.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
