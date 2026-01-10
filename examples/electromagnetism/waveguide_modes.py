"""
Experiment 98: Waveguide modes.

This example demonstrates electromagnetic wave propagation in rectangular
waveguides, showing TE and TM modes, cutoff frequencies, field patterns,
and dispersion relations.
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


def cutoff_frequency(m, n, a, b):
    """
    Calculate cutoff frequency for TE_mn or TM_mn mode.

    f_c = (c/2) * sqrt((m/a)^2 + (n/b)^2)

    Args:
        m: Mode index in x-direction
        n: Mode index in y-direction
        a: Waveguide width (m)
        b: Waveguide height (m)

    Returns:
        f_c: Cutoff frequency (Hz)
    """
    return (C / 2) * np.sqrt((m / a)**2 + (n / b)**2)


def cutoff_wavelength(m, n, a, b):
    """
    Calculate cutoff wavelength for mode.

    lambda_c = 2 / sqrt((m/a)^2 + (n/b)^2)

    Args:
        m, n: Mode indices
        a, b: Waveguide dimensions (m)

    Returns:
        lambda_c: Cutoff wavelength (m)
    """
    return 2 / np.sqrt((m / a)**2 + (n / b)**2)


def guide_wavelength(f, f_c):
    """
    Calculate wavelength in waveguide (guide wavelength).

    lambda_g = lambda_0 / sqrt(1 - (f_c/f)^2)

    Valid only for f > f_c.

    Args:
        f: Operating frequency (Hz)
        f_c: Cutoff frequency (Hz)

    Returns:
        lambda_g: Guide wavelength (m), or inf if f < f_c
    """
    if np.isscalar(f):
        if f <= f_c:
            return np.inf
        return (C / f) / np.sqrt(1 - (f_c / f)**2)

    result = np.full_like(f, np.inf, dtype=float)
    mask = f > f_c
    lambda_0 = C / f[mask]
    result[mask] = lambda_0 / np.sqrt(1 - (f_c / f[mask])**2)
    return result


def phase_velocity(f, f_c):
    """
    Calculate phase velocity in waveguide.

    v_p = c / sqrt(1 - (f_c/f)^2)

    Always greater than c for propagating modes.

    Args:
        f: Operating frequency (Hz)
        f_c: Cutoff frequency (Hz)

    Returns:
        v_p: Phase velocity (m/s)
    """
    if np.isscalar(f):
        if f <= f_c:
            return np.inf
        return C / np.sqrt(1 - (f_c / f)**2)

    result = np.full_like(f, np.inf, dtype=float)
    mask = f > f_c
    result[mask] = C / np.sqrt(1 - (f_c / f[mask])**2)
    return result


def group_velocity(f, f_c):
    """
    Calculate group velocity in waveguide.

    v_g = c * sqrt(1 - (f_c/f)^2)

    Always less than c for propagating modes.

    Args:
        f: Operating frequency (Hz)
        f_c: Cutoff frequency (Hz)

    Returns:
        v_g: Group velocity (m/s)
    """
    if np.isscalar(f):
        if f <= f_c:
            return 0
        return C * np.sqrt(1 - (f_c / f)**2)

    result = np.zeros_like(f, dtype=float)
    mask = f > f_c
    result[mask] = C * np.sqrt(1 - (f_c / f[mask])**2)
    return result


def te_mode_fields(x, y, z, m, n, a, b, E0, f, t=0):
    """
    Calculate TE_mn mode field components.

    For TE modes, E_z = 0.

    Args:
        x, y, z: Position (m)
        m, n: Mode indices (m >= 0, n >= 0, not both zero)
        a, b: Waveguide dimensions (m)
        E0: Field amplitude
        f: Frequency (Hz)
        t: Time (s)

    Returns:
        Ex, Ey, Ez, Hx, Hy, Hz: Field components
    """
    f_c = cutoff_frequency(m, n, a, b)
    if f <= f_c:
        return 0, 0, 0, 0, 0, 0

    omega = 2 * np.pi * f
    k = omega / C
    k_c = 2 * np.pi * f_c / C
    beta = np.sqrt(k**2 - k_c**2)  # Propagation constant

    kx = m * np.pi / a
    ky = n * np.pi / b

    # H_z (for TE modes, this is the longitudinal component)
    Hz = E0 * np.cos(kx * x) * np.cos(ky * y) * np.cos(omega * t - beta * z)

    # Transverse components from Hz
    factor = beta / k_c**2

    Hx = factor * kx * E0 * np.sin(kx * x) * np.cos(ky * y) * np.sin(omega * t - beta * z)
    Hy = factor * ky * E0 * np.cos(kx * x) * np.sin(ky * y) * np.sin(omega * t - beta * z)

    # E from curl H
    eta = np.sqrt(MU_0 / EPSILON_0)  # Wave impedance
    Z_te = eta / np.sqrt(1 - (f_c/f)**2)  # TE wave impedance

    Ex = -Z_te * Hy / eta * np.sqrt(1 - (f_c/f)**2)
    Ey = Z_te * Hx / eta * np.sqrt(1 - (f_c/f)**2)
    Ez = 0  # TE mode

    return Ex, Ey, Ez, Hx, Hy, Hz


def tm_mode_fields(x, y, z, m, n, a, b, E0, f, t=0):
    """
    Calculate TM_mn mode field components.

    For TM modes, H_z = 0.

    Args:
        x, y, z: Position (m)
        m, n: Mode indices (m >= 1, n >= 1)
        a, b: Waveguide dimensions (m)
        E0: Field amplitude
        f: Frequency (Hz)
        t: Time (s)

    Returns:
        Ex, Ey, Ez, Hx, Hy, Hz: Field components
    """
    f_c = cutoff_frequency(m, n, a, b)
    if f <= f_c:
        return 0, 0, 0, 0, 0, 0

    omega = 2 * np.pi * f
    k = omega / C
    k_c = 2 * np.pi * f_c / C
    beta = np.sqrt(k**2 - k_c**2)

    kx = m * np.pi / a
    ky = n * np.pi / b

    # E_z (for TM modes, this is the longitudinal component)
    Ez = E0 * np.sin(kx * x) * np.sin(ky * y) * np.cos(omega * t - beta * z)

    # Transverse components from Ez
    factor = beta / k_c**2

    Ex = -factor * kx * E0 * np.cos(kx * x) * np.sin(ky * y) * np.sin(omega * t - beta * z)
    Ey = -factor * ky * E0 * np.sin(kx * x) * np.cos(ky * y) * np.sin(omega * t - beta * z)

    # H from curl E
    eta = np.sqrt(MU_0 / EPSILON_0)
    Z_tm = eta * np.sqrt(1 - (f_c/f)**2)  # TM wave impedance

    Hx = Ey / Z_tm
    Hy = -Ex / Z_tm
    Hz = 0  # TM mode

    return Ex, Ey, Ez, Hx, Hy, Hz


def main():
    # Standard WR-90 waveguide dimensions (X-band, 8-12 GHz)
    a = 22.86e-3  # 22.86 mm width
    b = 10.16e-3  # 10.16 mm height

    fig = plt.figure(figsize=(16, 12))

    # Plot 1: Cutoff frequencies for different modes
    ax1 = fig.add_subplot(2, 2, 1)

    modes = []
    for m in range(4):
        for n in range(3):
            if m == 0 and n == 0:
                continue  # No (0,0) mode
            f_c = cutoff_frequency(m, n, a, b)
            mode_type = 'TE' if (m == 0 or n == 0) else 'TE/TM'
            if m > 0 and n > 0:
                mode_type = 'TM'  # TM modes require both m,n > 0
            modes.append((f'TE{m}{n}', f_c / 1e9, 'blue'))
            if m > 0 and n > 0:
                modes.append((f'TM{m}{n}', f_c / 1e9, 'red'))

    # Sort by cutoff frequency
    modes.sort(key=lambda x: x[1])

    # Remove duplicates (TE and TM with same cutoff)
    seen = set()
    unique_modes = []
    for mode in modes:
        if mode[1] not in seen or mode[0].startswith('TM'):
            unique_modes.append(mode)
            seen.add(mode[1])

    names = [m[0] for m in unique_modes[:10]]
    freqs = [m[1] for m in unique_modes[:10]]
    colors = [m[2] for m in unique_modes[:10]]

    bars = ax1.barh(names, freqs, color=colors, alpha=0.7)

    ax1.set_xlabel('Cutoff Frequency (GHz)')
    ax1.set_title(f'Waveguide Mode Cutoff Frequencies\nWR-90: {a*1000:.2f} x {b*1000:.2f} mm')
    ax1.grid(True, alpha=0.3, axis='x')

    # Mark operating band (X-band)
    ax1.axvspan(8, 12, alpha=0.2, color='green', label='X-band (8-12 GHz)')
    ax1.legend()

    # Add frequency values to bars
    for bar, freq in zip(bars, freqs):
        ax1.text(freq + 0.2, bar.get_y() + bar.get_height()/2,
                f'{freq:.2f}', va='center', fontsize=9)

    # Plot 2: Dispersion relation (omega vs beta)
    ax2 = fig.add_subplot(2, 2, 2)

    f_c_10 = cutoff_frequency(1, 0, a, b)  # TE10 dominant mode
    f_c_20 = cutoff_frequency(2, 0, a, b)  # TE20
    f_c_01 = cutoff_frequency(0, 1, a, b)  # TE01

    f_range = np.linspace(0.5e9, 20e9, 500)

    # Calculate propagation constant beta for each mode
    def beta_from_f(f, f_c):
        result = np.zeros_like(f)
        mask = f > f_c
        result[mask] = (2 * np.pi / C) * np.sqrt(f[mask]**2 - f_c**2)
        return result

    beta_10 = beta_from_f(f_range, f_c_10)
    beta_20 = beta_from_f(f_range, f_c_20)
    beta_01 = beta_from_f(f_range, f_c_01)

    # Light line
    beta_light = 2 * np.pi * f_range / C

    ax2.plot(beta_10, f_range / 1e9, 'b-', lw=2, label='TE10')
    ax2.plot(beta_20, f_range / 1e9, 'g-', lw=2, label='TE20')
    ax2.plot(beta_01, f_range / 1e9, 'r-', lw=2, label='TE01')
    ax2.plot(beta_light, f_range / 1e9, 'k--', lw=1, label='Light line')

    ax2.set_xlabel(r'Propagation Constant $\beta$ (rad/m)')
    ax2.set_ylabel('Frequency (GHz)')
    ax2.set_title('Dispersion Relation')
    ax2.set_xlim(0, 600)
    ax2.set_ylim(0, 20)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Mark cutoff frequencies
    ax2.axhline(y=f_c_10 / 1e9, color='blue', linestyle=':', alpha=0.5)
    ax2.axhline(y=f_c_20 / 1e9, color='green', linestyle=':', alpha=0.5)
    ax2.axhline(y=f_c_01 / 1e9, color='red', linestyle=':', alpha=0.5)

    # Plot 3: TE10 mode field pattern
    ax3 = fig.add_subplot(2, 2, 3)

    x = np.linspace(0, a, 50)
    y = np.linspace(0, b, 30)
    X, Y = np.meshgrid(x, y)

    f_op = 10e9  # 10 GHz operating frequency
    E0 = 1.0

    # TE10 mode: E_y varies as sin(pi*x/a), uniform in y
    Ex, Ey, Ez, Hx, Hy, Hz = te_mode_fields(X, Y, 0, 1, 0, a, b, E0, f_op)

    # For TE10: E_y ~ sin(pi*x/a)
    Ey_pattern = np.sin(np.pi * X / a)

    # Plot E-field magnitude
    im = ax3.contourf(X * 1000, Y * 1000, np.abs(Ey_pattern), levels=20, cmap='hot')
    plt.colorbar(im, ax=ax3, label='|E_y| (normalized)')

    # Add field vectors (H field for TE mode has x and z components)
    skip = 3
    # H_x ~ cos(pi*x/a)
    Hx_pattern = np.cos(np.pi * X / a)
    ax3.quiver(X[::skip, ::skip] * 1000, Y[::skip, ::skip] * 1000,
              Hx_pattern[::skip, ::skip], np.zeros_like(Hx_pattern[::skip, ::skip]),
              color='cyan', alpha=0.7, scale=15)

    ax3.set_xlabel('x (mm)')
    ax3.set_ylabel('y (mm)')
    ax3.set_title(f'TE10 Mode Field Pattern at {f_op/1e9:.0f} GHz\n(Color: |E_y|, Arrows: H_x)')
    ax3.set_aspect('equal')

    # Draw waveguide boundary
    ax3.plot([0, a*1000, a*1000, 0, 0], [0, 0, b*1000, b*1000, 0], 'k-', lw=2)

    # Plot 4: Phase and group velocity
    ax4 = fig.add_subplot(2, 2, 4)

    f_range = np.linspace(f_c_10 * 1.01, 3 * f_c_10, 200)

    v_p = phase_velocity(f_range, f_c_10)
    v_g = group_velocity(f_range, f_c_10)

    ax4.plot(f_range / 1e9, v_p / C, 'b-', lw=2, label=r'Phase velocity $v_p/c$')
    ax4.plot(f_range / 1e9, v_g / C, 'r-', lw=2, label=r'Group velocity $v_g/c$')
    ax4.axhline(y=1, color='gray', linestyle='--', lw=1, label='Speed of light c')

    ax4.axvline(x=f_c_10 / 1e9, color='green', linestyle=':', lw=2)
    ax4.text(f_c_10 / 1e9 + 0.2, 2.5, f'f_c = {f_c_10/1e9:.2f} GHz', fontsize=9)

    ax4.set_xlabel('Frequency (GHz)')
    ax4.set_ylabel('Velocity / c')
    ax4.set_title('Phase and Group Velocity (TE10 Mode)')
    ax4.set_ylim(0, 4)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Add v_p * v_g = c^2 annotation
    ax4.text(0.95, 0.95, r'$v_p \cdot v_g = c^2$',
             transform=ax4.transAxes, ha='right', va='top', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Add physics summary
    fig.text(0.5, 0.02,
             r'Cutoff: $f_c = \frac{c}{2}\sqrt{(m/a)^2 + (n/b)^2}$, '
             r'Dispersion: $\beta = \frac{\omega}{c}\sqrt{1 - (f_c/f)^2}$'
             + '\n' +
             r'Phase velocity: $v_p = c/\sqrt{1-(f_c/f)^2} > c$, '
             r'Group velocity: $v_g = c\sqrt{1-(f_c/f)^2} < c$',
             ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.suptitle('Electromagnetic Wave Propagation in Rectangular Waveguide', fontsize=14, y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)

    # Save plot
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'waveguide_modes.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
