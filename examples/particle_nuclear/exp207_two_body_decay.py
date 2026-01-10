"""
Experiment 207: Two-Body Decay Kinematics

Demonstrates relativistic kinematics for two-body particle decays.
Shows energy-momentum conservation, daughter particle energies,
and angular distributions.

Physics:
- M → m₁ + m₂ (parent mass M decays to daughters m₁, m₂)
- E₁ = (M² + m₁² - m₂²)/(2M)
- p* = λ^(1/2)(M², m₁², m₂²)/(2M) where λ is Kallen function
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def kallen_function(a, b, c):
    """
    Kallen (triangle) function λ(a, b, c).

    λ = a² + b² + c² - 2ab - 2bc - 2ca
    """
    return a**2 + b**2 + c**2 - 2*a*b - 2*b*c - 2*c*a


def two_body_cm_momentum(M, m1, m2):
    """
    Center-of-mass momentum for two-body decay.

    p* = λ^(1/2)(M², m₁², m₂²)/(2M)

    Args:
        M: Parent mass
        m1, m2: Daughter masses

    Returns:
        CM momentum magnitude, or 0 if kinematically forbidden
    """
    lam = kallen_function(M**2, m1**2, m2**2)
    if lam < 0:
        return 0
    return np.sqrt(lam) / (2 * M)


def two_body_cm_energy(M, m1, m2):
    """
    Center-of-mass energies for daughters.

    E₁ = (M² + m₁² - m₂²)/(2M)
    E₂ = (M² + m₂² - m₁²)/(2M)

    Returns:
        (E1, E2) daughter energies
    """
    E1 = (M**2 + m1**2 - m2**2) / (2 * M)
    E2 = (M**2 + m2**2 - m1**2) / (2 * M)
    return E1, E2


def lab_energy_from_angle(E_cm, p_cm, beta_parent, theta_lab, m):
    """
    Lab energy of daughter at lab angle theta.

    E_lab = γ(E* + βp* cos θ*)
    """
    gamma = 1 / np.sqrt(1 - beta_parent**2)
    E_lab = gamma * (E_cm + beta_parent * p_cm * np.cos(theta_lab))
    return E_lab


def lorentz_boost(E, p, beta):
    """
    Lorentz boost along z-axis.

    Returns boosted (E', p')
    """
    gamma = 1 / np.sqrt(1 - beta**2)
    E_prime = gamma * (E - beta * p)
    p_prime = gamma * (p - beta * E)
    return E_prime, p_prime


def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Example: Pion decay π⁺ → μ⁺ + ν_μ
    m_pi = 139.57   # MeV
    m_mu = 105.66   # MeV
    m_nu = 0.0      # MeV (neutrino mass negligible)

    # Plot 1: CM energies vs parent mass
    ax = axes[0, 0]

    # Variable parent mass
    M_range = np.linspace(m_mu + 1, 500, 200)

    E1_vals = []
    E2_vals = []
    p_vals = []

    for M in M_range:
        E1, E2 = two_body_cm_energy(M, m_mu, m_nu)
        p = two_body_cm_momentum(M, m_mu, m_nu)
        E1_vals.append(E1)
        E2_vals.append(E2)
        p_vals.append(p)

    ax.plot(M_range, E1_vals, 'b-', lw=2, label='E_μ (muon)')
    ax.plot(M_range, E2_vals, 'r--', lw=2, label='E_ν (neutrino)')
    ax.plot(M_range, p_vals, 'g:', lw=2, label='p* (CM momentum)')

    ax.axvline(x=m_pi, color='k', linestyle='--', alpha=0.5)
    ax.text(m_pi + 5, 200, 'π mass', fontsize=10)

    ax.set_xlabel('Parent Mass M (MeV)')
    ax.set_ylabel('Energy/Momentum (MeV)')
    ax.set_title('Two-Body Decay: M → μ + ν\nCM Frame')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Pion decay in CM frame
    ax = axes[0, 1]

    E_mu, E_nu = two_body_cm_energy(m_pi, m_mu, m_nu)
    p_cm = two_body_cm_momentum(m_pi, m_mu, m_nu)

    # Angular distribution (isotropic in CM)
    theta_cm = np.linspace(0, 2*np.pi, 100)

    # Muon momentum components
    px_mu = p_cm * np.cos(theta_cm)
    py_mu = p_cm * np.sin(theta_cm)

    # Neutrino (opposite)
    px_nu = -px_mu
    py_nu = -py_mu

    ax.plot(px_mu, py_mu, 'b-', lw=2, label='μ⁺')
    ax.plot(px_nu, py_nu, 'r--', lw=2, label='ν_μ')
    ax.plot(0, 0, 'ko', markersize=10, label='π⁺ at rest')

    ax.set_xlabel('p_x (MeV/c)')
    ax.set_ylabel('p_y (MeV/c)')
    ax.set_title(f'π⁺ → μ⁺ + ν_μ in CM Frame\np* = {p_cm:.1f} MeV/c')
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Lab frame energy distribution
    ax = axes[0, 2]

    # Pion with kinetic energy
    T_pi = 200  # MeV kinetic energy
    E_pi_lab = m_pi + T_pi
    p_pi_lab = np.sqrt(E_pi_lab**2 - m_pi**2)
    beta_pi = p_pi_lab / E_pi_lab

    # Lab angle distribution
    theta_lab = np.linspace(0, np.pi, 100)
    gamma = 1 / np.sqrt(1 - beta_pi**2)

    # Forward/backward muon energies
    E_mu_forward = gamma * (E_mu + beta_pi * p_cm)
    E_mu_backward = gamma * (E_mu - beta_pi * p_cm)

    # Energy vs angle (assuming θ* uniform)
    E_mu_lab = gamma * (E_mu + beta_pi * p_cm * np.cos(theta_lab))

    ax.plot(np.degrees(theta_lab), E_mu_lab, 'b-', lw=2)
    ax.axhline(y=E_mu_forward, color='r', linestyle='--',
               label=f'Max E = {E_mu_forward:.1f} MeV')
    ax.axhline(y=E_mu_backward, color='g', linestyle='--',
               label=f'Min E = {E_mu_backward:.1f} MeV')

    ax.set_xlabel('Lab Angle θ (degrees)')
    ax.set_ylabel('Muon Energy (MeV)')
    ax.set_title(f'Muon Energy in Lab Frame\nπ with T = {T_pi} MeV')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Opening angle for symmetric decay
    ax = axes[1, 0]

    # π⁰ → γγ decay
    m_pi0 = 135.0  # MeV

    # Pion kinetic energy range
    T_range = np.linspace(1, 1000, 200)

    min_angles = []
    for T in T_range:
        E_pi0 = m_pi0 + T
        beta = np.sqrt(1 - (m_pi0/E_pi0)**2)
        gamma = E_pi0 / m_pi0

        # Minimum opening angle when photons symmetric
        # sin(θ_min/2) = m / E_lab for massless daughters
        if gamma > 1:
            sin_half = 1 / gamma
            theta_min = 2 * np.arcsin(sin_half)
        else:
            theta_min = np.pi
        min_angles.append(np.degrees(theta_min))

    ax.semilogy(T_range, min_angles, 'b-', lw=2)
    ax.set_xlabel('π⁰ Kinetic Energy (MeV)')
    ax.set_ylabel('Minimum Opening Angle (degrees)')
    ax.set_title('π⁰ → γγ: Minimum Opening Angle\nCollinearity vs Energy')
    ax.grid(True, alpha=0.3)

    # Plot 5: Three different decays comparison
    ax = axes[1, 1]

    # Different decay modes
    decays = [
        ('π → μν', 139.57, 105.66, 0, 'b'),
        ('K → μν', 493.68, 105.66, 0, 'r'),
        ('K → ππ', 493.68, 139.57, 139.57, 'g'),
    ]

    for name, M, m1, m2, color in decays:
        if M > m1 + m2:
            E1, E2 = two_body_cm_energy(M, m1, m2)
            p = two_body_cm_momentum(M, m1, m2)

            # Kinetic energies
            T1 = E1 - m1
            T2 = E2 - m2 if m2 > 0 else E2

            ax.bar([name], [T1], color=color, alpha=0.7, label=f'T₁ = {T1:.1f}')
            ax.bar([name], [T2], bottom=[T1], color=color, alpha=0.4)

    ax.set_ylabel('Kinetic Energy (MeV)')
    ax.set_title('Kinetic Energy Release\nDifferent Two-Body Decays')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 6: Phase space and kinematics summary
    ax = axes[1, 2]
    ax.axis('off')

    E_mu_cm, E_nu_cm = two_body_cm_energy(m_pi, m_mu, m_nu)
    p_star = two_body_cm_momentum(m_pi, m_mu, m_nu)
    T_mu = E_mu_cm - m_mu
    T_nu = E_nu_cm

    summary = f"""
Two-Body Decay Kinematics
=========================

Conservation Laws:
  M = E₁ + E₂ (energy)
  0 = p₁ + p₂ (momentum in CM)

CM Frame Formulas:
  E₁ = (M² + m₁² - m₂²)/(2M)
  E₂ = (M² + m₂² - m₁²)/(2M)
  p* = λ^(1/2)/(2M)

where λ(a,b,c) = a² + b² + c² - 2(ab + bc + ca)

Example: π⁺ → μ⁺ + ν_μ
  m_π = {m_pi:.2f} MeV
  m_μ = {m_mu:.2f} MeV
  m_ν ≈ 0

  E_μ = {E_mu_cm:.2f} MeV
  E_ν = {E_nu_cm:.2f} MeV
  p*  = {p_star:.2f} MeV/c

  T_μ = {T_mu:.2f} MeV (fixed!)
  T_ν = {T_nu:.2f} MeV

Key Feature:
  In two-body decay, daughter energies
  are FIXED (monoenergetic) in CM frame.

Lorentz Boost to Lab:
  E_lab = γ(E* + βp* cos θ*)
  p_lab = γ(p* cos θ* + βE*)
"""

    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Experiment 207: Two-Body Decay Kinematics\n'
                 'Relativistic Energy-Momentum Conservation', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'exp207_two_body_decay.png'),
                dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/exp207_two_body_decay.png")


if __name__ == "__main__":
    main()
