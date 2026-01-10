"""
Experiment 205: Partial Wave Expansion

Demonstrates partial wave analysis for quantum scattering. Expands the
scattering amplitude in terms of Legendre polynomials and phase shifts.

Physics:
- f(θ) = Σ_l (2l+1) f_l P_l(cos θ)
- f_l = (e^(2iδ_l) - 1) / (2ik) = sin(δ_l)/k × e^(iδ_l)
- σ_tot = (4π/k²) Σ_l (2l+1) sin²(δ_l)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from src.sciforge.physics.nuclear import PartialWave


def hard_sphere_phase_shift(l, k, R):
    """
    Phase shift for hard sphere scattering.

    tan(δ_l) = j_l(kR) / n_l(kR)

    Args:
        l: Angular momentum quantum number
        k: Wave number
        R: Sphere radius

    Returns:
        Phase shift in radians
    """
    x = k * R

    if x < 1e-6:
        return 0

    # Spherical Bessel functions
    j_l = special.spherical_jn(l, x)
    n_l = special.spherical_yn(l, x)

    return np.arctan2(j_l, n_l)


def square_well_phase_shift(l, k, V0, R, mu=1.0):
    """
    Phase shift for finite square well.

    Args:
        l: Angular momentum
        k: External wave number
        V0: Well depth (positive for attractive)
        R: Well radius
        mu: Reduced mass

    Returns:
        Phase shift in radians
    """
    hbar = 1.0  # Natural units

    # Internal wave number
    E = hbar**2 * k**2 / (2 * mu)
    if V0 > E:
        k_in = np.sqrt(2 * mu * (E + V0)) / hbar
    else:
        k_in = np.sqrt(2 * mu * (E + V0)) / hbar

    x = k * R
    x_in = k_in * R

    # Matching at boundary using logarithmic derivative
    j_l_in = special.spherical_jn(l, x_in)
    j_l_in_deriv = special.spherical_jn(l, x_in, derivative=True)

    j_l = special.spherical_jn(l, x)
    j_l_deriv = special.spherical_jn(l, x, derivative=True)
    n_l = special.spherical_yn(l, x)
    n_l_deriv = special.spherical_yn(l, x, derivative=True)

    # Logarithmic derivative of internal solution
    if abs(j_l_in) > 1e-10:
        beta = (k_in * j_l_in_deriv) / j_l_in
    else:
        beta = 0

    # Phase shift from matching condition
    numerator = k * j_l_deriv - beta * j_l
    denominator = k * n_l_deriv - beta * n_l

    if abs(denominator) > 1e-10:
        delta = np.arctan2(numerator, denominator)
    else:
        delta = np.pi / 2

    return delta


def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Parameters
    R = 1.0      # fm
    k = 1.0      # fm^-1
    l_max = 10

    # Plot 1: Hard sphere phase shifts vs energy
    ax = axes[0, 0]

    k_values = np.linspace(0.1, 5, 100)
    colors = plt.cm.viridis(np.linspace(0, 0.9, 5))

    for l, color in zip(range(5), colors):
        delta_l = [hard_sphere_phase_shift(l, k_val, R) for k_val in k_values]
        ax.plot(k_values * R, np.degrees(delta_l), '-', color=color,
                lw=2, label=f'l = {l}')

    ax.set_xlabel('kR')
    ax.set_ylabel('Phase Shift δ_l (degrees)')
    ax.set_title('Hard Sphere Phase Shifts')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)

    # Plot 2: Square well phase shifts (resonances)
    ax = axes[0, 1]

    V0 = 50.0  # MeV (deep well)
    k_values_sw = np.linspace(0.1, 5, 200)

    for l, color in zip(range(4), colors):
        delta_l = [square_well_phase_shift(l, k_val, V0, R) for k_val in k_values_sw]
        # Unwrap phase
        delta_l = np.unwrap(delta_l)
        ax.plot(k_values_sw, np.degrees(delta_l), '-', color=color,
                lw=2, label=f'l = {l}')

    ax.set_xlabel('k (fm⁻¹)')
    ax.set_ylabel('Phase Shift δ_l (degrees)')
    ax.set_title(f'Square Well Phase Shifts\nV₀ = {V0} MeV, R = {R} fm')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Differential cross section from partial waves
    ax = axes[0, 2]

    theta = np.linspace(0.01, np.pi, 200)
    cos_theta = np.cos(theta)

    # Set up partial waves
    pw = PartialWave(k=2.0, l_max=6)

    # Case 1: Pure s-wave
    pw.set_phase_shift(0, np.pi/4)
    f_swave = pw.scattering_amplitude(theta)
    dcs_swave = np.abs(f_swave)**2

    # Case 2: s + p wave
    pw.set_phase_shift(0, np.pi/4)
    pw.set_phase_shift(1, np.pi/6)
    f_sp = pw.scattering_amplitude(theta)
    dcs_sp = np.abs(f_sp)**2

    # Case 3: s + p + d wave
    pw.set_phase_shift(0, np.pi/4)
    pw.set_phase_shift(1, np.pi/6)
    pw.set_phase_shift(2, np.pi/8)
    f_spd = pw.scattering_amplitude(theta)
    dcs_spd = np.abs(f_spd)**2

    ax.semilogy(np.degrees(theta), dcs_swave, 'b-', lw=2, label='s-wave only')
    ax.semilogy(np.degrees(theta), dcs_sp, 'g-', lw=2, label='s + p')
    ax.semilogy(np.degrees(theta), dcs_spd, 'r-', lw=2, label='s + p + d')

    ax.set_xlabel('Scattering Angle (degrees)')
    ax.set_ylabel('dσ/dΩ (arb. units)')
    ax.set_title('Differential Cross Section\nPartial Wave Contributions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Partial cross sections
    ax = axes[1, 0]

    # Calculate partial cross sections for different l
    k_scan = np.linspace(0.5, 4, 100)

    for l in range(4):
        sigma_l = []
        for k_val in k_scan:
            delta = square_well_phase_shift(l, k_val, V0=30, R=1.5)
            # Partial cross section: σ_l = (4π/k²)(2l+1)sin²(δ_l)
            sigma = 4 * np.pi / k_val**2 * (2*l + 1) * np.sin(delta)**2
            sigma_l.append(sigma)

        ax.plot(k_scan, sigma_l, '-', lw=2, label=f'l = {l}')

    ax.set_xlabel('k (fm⁻¹)')
    ax.set_ylabel('σ_l (fm²)')
    ax.set_title('Partial Cross Sections\n(Resonance structure visible)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Scattering amplitude in complex plane
    ax = axes[1, 1]

    # Fixed angle, vary energy
    theta_fixed = np.pi / 4  # 45 degrees
    k_range = np.linspace(0.5, 3, 50)

    f_real = []
    f_imag = []

    for k_val in k_range:
        pw_temp = PartialWave(k=k_val, l_max=4)
        for l in range(5):
            delta = square_well_phase_shift(l, k_val, V0=25, R=1.2)
            pw_temp.set_phase_shift(l, delta)
        f = pw_temp.scattering_amplitude(np.array([theta_fixed]))[0]
        f_real.append(np.real(f))
        f_imag.append(np.imag(f))

    # Color by k value
    colors_k = plt.cm.coolwarm(np.linspace(0, 1, len(k_range)))
    for i in range(len(k_range) - 1):
        ax.plot(f_real[i:i+2], f_imag[i:i+2], '-', color=colors_k[i], lw=2)

    ax.scatter(f_real[0], f_imag[0], c='b', s=100, marker='o',
               label=f'k = {k_range[0]:.1f}', zorder=5)
    ax.scatter(f_real[-1], f_imag[-1], c='r', s=100, marker='s',
               label=f'k = {k_range[-1]:.1f}', zorder=5)

    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Re[f(θ)]')
    ax.set_ylabel('Im[f(θ)]')
    ax.set_title(f'Scattering Amplitude (θ = 45°)\nArgand Diagram')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Plot 6: Unitarity circle
    ax = axes[1, 2]

    # For single partial wave: f_l = (e^(2iδ) - 1)/(2ik)
    # Parametrize by phase shift
    delta_range = np.linspace(0, 2*np.pi, 100)
    k_fixed = 1.5

    for l in range(4):
        f_l_real = []
        f_l_imag = []

        for delta in delta_range:
            f_l = (np.exp(2j * delta) - 1) / (2j * k_fixed)
            f_l_real.append(np.real(f_l))
            f_l_imag.append(np.imag(f_l))

        ax.plot(f_l_real, f_l_imag, '-', lw=2, label=f'l = {l}')

    # Unitarity circle
    theta_circ = np.linspace(0, 2*np.pi, 100)
    radius = 1 / (2 * k_fixed)
    ax.plot(radius * np.cos(theta_circ), radius + radius * np.sin(theta_circ),
            'k--', lw=1, label='Unitarity circle')

    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Re[f_l]')
    ax.set_ylabel('Im[f_l]')
    ax.set_title('Partial Wave Unitarity Circle\n|S_l| = 1 (elastic)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.suptitle('Experiment 205: Partial Wave Expansion\n'
                 'Quantum Scattering Analysis', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'exp205_partial_wave_expansion.png'),
                dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/exp205_partial_wave_expansion.png")


if __name__ == "__main__":
    main()
