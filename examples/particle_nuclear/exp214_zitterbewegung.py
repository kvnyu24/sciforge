"""
Experiment 214: Dirac Equation Zitterbewegung

Demonstrates the "trembling motion" (Zitterbewegung) of Dirac particles,
a rapid oscillation due to interference between positive and negative
energy components.

Physics:
- Frequency: ω_Z = 2mc²/ℏ ≈ 1.6 × 10²¹ Hz for electron
- Amplitude: λ_C = ℏ/(mc) ≈ 3.9 × 10⁻¹³ m (Compton wavelength)
- Caused by non-commuting velocity and Hamiltonian
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.nuclear import DiracEquation


def dirac_gamma_matrices():
    """Define Dirac gamma matrices in Dirac representation."""
    gamma0 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, -1]
    ], dtype=complex)

    gamma1 = np.array([
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [-1, 0, 0, 0]
    ], dtype=complex)

    gamma2 = np.array([
        [0, 0, 0, -1j],
        [0, 0, 1j, 0],
        [0, 1j, 0, 0],
        [-1j, 0, 0, 0]
    ], dtype=complex)

    gamma3 = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, -1],
        [-1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype=complex)

    return gamma0, gamma1, gamma2, gamma3


def dirac_hamiltonian(p, m):
    """
    Dirac Hamiltonian H = α·p + βm

    α = γ⁰γ, β = γ⁰
    """
    gamma0, gamma1, gamma2, gamma3 = dirac_gamma_matrices()

    alpha1 = gamma0 @ gamma1
    alpha2 = gamma0 @ gamma2
    alpha3 = gamma0 @ gamma3
    beta = gamma0

    H = alpha1 * p[0] + alpha2 * p[1] + alpha3 * p[2] + beta * m
    return H


def velocity_operator(direction=0):
    """
    Velocity operator v = c α = c γ⁰γ^i

    In natural units, v_i = α_i
    """
    gamma0, gamma1, gamma2, gamma3 = dirac_gamma_matrices()
    gammas = [gamma1, gamma2, gamma3]
    return gamma0 @ gammas[direction]


def positive_energy_spinor(p, m, spin=1):
    """
    Positive energy spinor u(p,s).

    For p along z: u = N × [χ_s, (p_z/(E+m))χ_s]
    """
    p = np.array(p)
    p_mag = np.linalg.norm(p)
    E = np.sqrt(p_mag**2 + m**2)

    if spin == 1:
        chi = np.array([1, 0])
    else:
        chi = np.array([0, 1])

    N = np.sqrt((E + m) / (2 * m))

    # For arbitrary p direction
    sigma_dot_p = np.array([
        [p[2], p[0] - 1j*p[1]],
        [p[0] + 1j*p[1], -p[2]]
    ])

    lower = sigma_dot_p @ chi / (E + m)

    u = N * np.concatenate([chi, lower])
    return u, E


def negative_energy_spinor(p, m, spin=1):
    """
    Negative energy spinor v(p,s).
    """
    p = np.array(p)
    p_mag = np.linalg.norm(p)
    E = np.sqrt(p_mag**2 + m**2)

    if spin == 1:
        chi = np.array([0, 1])
    else:
        chi = np.array([1, 0])

    N = np.sqrt((E + m) / (2 * m))

    sigma_dot_p = np.array([
        [p[2], p[0] - 1j*p[1]],
        [p[0] + 1j*p[1], -p[2]]
    ])

    upper = sigma_dot_p @ chi / (E + m)

    v = N * np.concatenate([upper, chi])
    return v, E


def zitterbewegung_expectation(p, m, t_array):
    """
    Calculate expectation value of position showing Zitterbewegung.

    <x>(t) = <x>(0) + <v>·t + (oscillating term)
    """
    # Create superposition of positive and negative energy states
    u, E = positive_energy_spinor(p, m)
    v, _ = negative_energy_spinor(p, m)

    # Superposition (normalized)
    a = 1 / np.sqrt(2)
    b = 1 / np.sqrt(2)

    # Velocity operator
    alpha_x = velocity_operator(0)

    # Classical velocity (for positive energy only)
    v_classical = np.real(np.conj(u) @ alpha_x @ u)

    # Zitterbewegung amplitude
    # <x> = <v>t + (ℏc/2E) Re[<u|α|v> e^(-2iEt)]
    cross_term = np.conj(u) @ alpha_x @ v

    x_expectation = []
    for t in t_array:
        # Classical drift
        x_class = v_classical * t

        # Zitterbewegung oscillation
        zb_amp = np.real(cross_term * np.exp(-2j * E * t)) / (2 * E)

        x_total = x_class + a * b * zb_amp

        x_expectation.append(np.real(x_total))

    return np.array(x_expectation), v_classical, 1/(2*E)


def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Parameters (natural units: c = ℏ = 1)
    m = 1.0  # Mass

    # Plot 1: Dirac spectrum
    ax = axes[0, 0]

    p_range = np.linspace(-5, 5, 200)

    # Positive and negative energy branches
    E_pos = np.sqrt(p_range**2 + m**2)
    E_neg = -np.sqrt(p_range**2 + m**2)

    ax.plot(p_range, E_pos, 'b-', lw=2, label='Positive energy (particles)')
    ax.plot(p_range, E_neg, 'r-', lw=2, label='Negative energy (antiparticles)')
    ax.axhline(y=m, color='b', linestyle='--', alpha=0.5)
    ax.axhline(y=-m, color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)

    ax.fill_between(p_range, -m, m, alpha=0.2, color='gray', label='Mass gap')

    ax.set_xlabel('Momentum p')
    ax.set_ylabel('Energy E')
    ax.set_title('Dirac Spectrum\nE = ±√(p² + m²)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-6, 6)

    # Plot 2: Zitterbewegung oscillation
    ax = axes[0, 1]

    p = np.array([0.5, 0, 0])  # Small momentum
    t = np.linspace(0, 30, 1000)

    x_exp, v_class, zb_amp = zitterbewegung_expectation(p, m, t)

    ax.plot(t, x_exp, 'b-', lw=1.5, label='<x>(t) with ZB')
    ax.plot(t, v_class * t, 'r--', lw=2, label='Classical drift')

    ax.set_xlabel('Time (ℏ/mc²)')
    ax.set_ylabel('Position (ℏ/mc)')
    ax.set_title(f'Zitterbewegung\np = {p[0]}, m = {m}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: ZB amplitude vs momentum
    ax = axes[0, 2]

    p_values = np.linspace(0.1, 5, 50)
    zb_amplitudes = []
    zb_frequencies = []

    for p_val in p_values:
        E = np.sqrt(p_val**2 + m**2)
        zb_amplitudes.append(1 / (2 * E))  # λ_C / 2 in natural units
        zb_frequencies.append(2 * E)

    ax2 = ax.twinx()

    line1, = ax.plot(p_values, zb_amplitudes, 'b-', lw=2, label='Amplitude')
    line2, = ax2.plot(p_values, zb_frequencies, 'r-', lw=2, label='Frequency')

    ax.set_xlabel('Momentum p')
    ax.set_ylabel('ZB Amplitude (ℏ/mc)', color='b')
    ax2.set_ylabel('ZB Frequency (mc²/ℏ)', color='r')
    ax.set_title('ZB Parameters vs Momentum')
    ax.legend([line1, line2], ['Amplitude', 'Frequency'], loc='center right')
    ax.grid(True, alpha=0.3)

    # Plot 4: Spinor components
    ax = axes[1, 0]

    p_spinor = np.array([2, 0, 0])
    u_pos, E_u = positive_energy_spinor(p_spinor, m)
    v_neg, E_v = negative_energy_spinor(p_spinor, m)

    components = ['ψ₁', 'ψ₂', 'ψ₃', 'ψ₄']
    x_pos = np.arange(4)
    width = 0.35

    ax.bar(x_pos - width/2, np.abs(u_pos)**2, width, label='u (positive E)')
    ax.bar(x_pos + width/2, np.abs(v_neg)**2, width, label='v (negative E)')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(components)
    ax.set_ylabel('|ψ|²')
    ax.set_title(f'Spinor Components\np = {p_spinor[0]}, m = {m}')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 5: Time evolution of probability density
    ax = axes[1, 1]

    # Create wavepacket as superposition
    x = np.linspace(-10, 30, 300)
    sigma = 2.0
    p0 = 1.0

    times = [0, 5, 10, 15]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(times)))

    for t_val, color in zip(times, colors):
        # Positive energy component
        E_pos_t = np.sqrt(p0**2 + m**2)
        v_g = p0 / E_pos_t

        # Envelope moves at group velocity, oscillates at ZB frequency
        envelope = np.exp(-(x - v_g * t_val)**2 / (4 * sigma**2))

        # Add ZB oscillation on top
        zb_osc = 0.1 * np.cos(2 * E_pos_t * t_val)

        rho = envelope * (1 + zb_osc * np.cos(2 * p0 * (x - v_g * t_val)))

        ax.plot(x, rho, '-', color=color, lw=2, label=f't = {t_val}')

    ax.set_xlabel('Position x')
    ax.set_ylabel('Probability Density')
    ax.set_title('Wavepacket with ZB Modulation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Summary
    ax = axes[1, 2]
    ax.axis('off')

    # Physical values
    hbar = 1.055e-34  # J·s
    c = 3e8  # m/s
    m_e = 9.109e-31  # kg
    lambda_C = hbar / (m_e * c)  # Compton wavelength
    omega_ZB = 2 * m_e * c**2 / hbar  # ZB frequency

    summary = f"""
Zitterbewegung (Trembling Motion)
=================================

Origin:
  Interference between positive and
  negative energy states in Dirac eq.

Dirac Equation:
  iℏ∂ψ/∂t = (cα·p + βmc²)ψ
  H = cα·p + βmc²

Velocity Operator:
  v = cα (eigenvalues ±c!)

Heisenberg Equation:
  dx/dt = cα
  [H, α] ≠ 0 → oscillation

ZB Characteristics:
  Frequency: ω_ZB = 2mc²/ℏ
  Amplitude: a_ZB ~ ℏ/(2mc) = λ_C/2

For Electron:
  ω_ZB ≈ {omega_ZB:.2e} rad/s
  f_ZB ≈ {omega_ZB/(2*np.pi):.2e} Hz
  λ_C ≈ {lambda_C:.2e} m

Physical Interpretation:
  • Not directly observable (averages out)
  • Related to particle-antiparticle mixing
  • Appears in semiconductors, graphene
  • Observable in ion traps (simulated)

Damping:
  Realistic wavepackets: ZB damps out
  over time as wavepacket spreads
"""

    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Experiment 214: Dirac Equation Zitterbewegung\n'
                 'Trembling Motion from Particle-Antiparticle Interference', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'exp214_zitterbewegung.png'),
                dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/exp214_zitterbewegung.png")


if __name__ == "__main__":
    main()
