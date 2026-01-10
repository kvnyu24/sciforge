"""
Experiment 206: Breit-Wigner Resonance

Demonstrates the Breit-Wigner resonance formula for nuclear and particle
physics reactions. Shows resonance line shapes, widths, and interference.

Physics:
- σ(E) = σ₀ Γ²/4 / [(E - E_R)² + Γ²/4]
- Full form: σ = π/k² × g × Γ_in Γ_out / [(E - E_R)² + Γ²/4]
- Resonance width: Γ = ℏ/τ (uncertainty relation)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.nuclear import ResonanceFormula


def breit_wigner(E, E_R, Gamma, sigma_0=1.0):
    """
    Breit-Wigner resonance formula.

    Args:
        E: Energy
        E_R: Resonance energy
        Gamma: Total width
        sigma_0: Peak cross section

    Returns:
        Cross section
    """
    return sigma_0 * (Gamma/2)**2 / ((E - E_R)**2 + (Gamma/2)**2)


def breit_wigner_amplitude(E, E_R, Gamma, Gamma_in=None, Gamma_out=None):
    """
    Breit-Wigner amplitude (complex).

    f(E) = Γ_in^(1/2) Γ_out^(1/2) / (E_R - E - iΓ/2)
    """
    if Gamma_in is None:
        Gamma_in = Gamma / 2
    if Gamma_out is None:
        Gamma_out = Gamma / 2

    return np.sqrt(Gamma_in * Gamma_out) / (E_R - E - 1j * Gamma / 2)


def two_resonance_interference(E, E_R1, Gamma1, E_R2, Gamma2, phase=0):
    """
    Interference between two overlapping resonances.

    |f_1 + e^(iφ) f_2|²
    """
    f1 = breit_wigner_amplitude(E, E_R1, Gamma1)
    f2 = breit_wigner_amplitude(E, E_R2, Gamma2) * np.exp(1j * phase)

    return np.abs(f1 + f2)**2


def fano_resonance(E, E_R, Gamma, q, background=1.0):
    """
    Fano resonance (asymmetric line shape).

    σ(E) = σ_bg × (q + ε)² / (1 + ε²)
    where ε = 2(E - E_R)/Γ
    """
    epsilon = 2 * (E - E_R) / Gamma
    return background * (q + epsilon)**2 / (1 + epsilon**2)


def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Basic Breit-Wigner shape
    ax = axes[0, 0]

    E_R = 10.0   # MeV
    Gamma = 1.0  # MeV

    E = np.linspace(5, 15, 500)

    # Different widths
    widths = [0.5, 1.0, 2.0, 4.0]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(widths)))

    for Gamma_i, color in zip(widths, colors):
        sigma = breit_wigner(E, E_R, Gamma_i)
        ax.plot(E, sigma, '-', color=color, lw=2,
                label=f'Γ = {Gamma_i} MeV')

    ax.axvline(x=E_R, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Energy (MeV)')
    ax.set_ylabel('Cross Section (arb. units)')
    ax.set_title('Breit-Wigner Resonance\nDifferent Widths')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Resonance in S-matrix (Argand diagram)
    ax = axes[0, 1]

    # S = 1 + 2if for elastic scattering
    E_argand = np.linspace(E_R - 3*Gamma, E_R + 3*Gamma, 100)

    for i, Gamma_i in enumerate([0.5, 1.0, 2.0]):
        f = breit_wigner_amplitude(E_argand, E_R, Gamma_i)
        S = 1 + 2j * f  # Unitarity

        ax.plot(np.real(S), np.imag(S), '-', lw=2,
                label=f'Γ = {Gamma_i}')

        # Mark resonance position
        f_res = breit_wigner_amplitude(E_R, E_R, Gamma_i)
        S_res = 1 + 2j * f_res
        ax.plot(np.real(S_res), np.imag(S_res), 'o', markersize=10)

    # Unitarity circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', lw=1, alpha=0.5)

    ax.set_xlabel('Re[S]')
    ax.set_ylabel('Im[S]')
    ax.set_title('S-Matrix Argand Diagram\nCircle represents unitarity')
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Multiple resonances
    ax = axes[0, 2]

    E_wide = np.linspace(0, 30, 500)

    # Three resonances at different energies
    resonances = [
        (5.0, 0.5, 'Low-energy'),
        (12.0, 1.5, 'Medium'),
        (20.0, 3.0, 'Broad')
    ]

    total = np.zeros_like(E_wide)
    for E_R_i, Gamma_i, label in resonances:
        sigma_i = breit_wigner(E_wide, E_R_i, Gamma_i)
        ax.plot(E_wide, sigma_i, '--', lw=1.5, alpha=0.7)
        total += sigma_i

    ax.plot(E_wide, total, 'k-', lw=2, label='Total (incoherent)')

    ax.set_xlabel('Energy (MeV)')
    ax.set_ylabel('Cross Section')
    ax.set_title('Multiple Resonances\n(Incoherent sum)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Interference between resonances
    ax = axes[1, 0]

    E_R1, E_R2 = 8.0, 12.0
    Gamma1, Gamma2 = 2.0, 2.0

    E_int = np.linspace(0, 20, 500)

    phases = [0, np.pi/4, np.pi/2, np.pi]
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(phases)))

    for phase, color in zip(phases, colors):
        sigma_int = two_resonance_interference(E_int, E_R1, Gamma1, E_R2, Gamma2, phase)
        ax.plot(E_int, sigma_int, '-', color=color, lw=2,
                label=f'φ = {np.degrees(phase):.0f}°')

    ax.set_xlabel('Energy (MeV)')
    ax.set_ylabel('Cross Section')
    ax.set_title('Resonance Interference\nTwo overlapping resonances')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Fano resonance (asymmetric)
    ax = axes[1, 1]

    q_values = [-3, -1, 0, 1, 3]
    colors = plt.cm.RdBu(np.linspace(0, 1, len(q_values)))

    E_fano = np.linspace(5, 15, 500)

    for q, color in zip(q_values, colors):
        sigma_fano = fano_resonance(E_fano, E_R, Gamma, q)
        ax.plot(E_fano, sigma_fano, '-', color=color, lw=2,
                label=f'q = {q}')

    ax.axvline(x=E_R, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Energy (MeV)')
    ax.set_ylabel('Cross Section')
    ax.set_title('Fano Resonance\n(Asymmetric line shape)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Resonance parameters summary
    ax = axes[1, 2]
    ax.axis('off')

    summary = """
Breit-Wigner Resonance Formula
==============================

Cross Section (single channel):
  σ(E) = (π/k²) g (Γ_in Γ_out) / [(E-E_R)² + Γ²/4]

Where:
  E_R = Resonance energy
  Γ = Total width = Σ Γ_i (sum over channels)
  Γ_in = Entrance channel width
  Γ_out = Exit channel width
  g = Statistical factor (2J+1)/[(2s₁+1)(2s₂+1)]
  k = Wave number = √(2μE)/ℏ

Properties:
  • Peak at E = E_R
  • FWHM = Γ (Full Width at Half Maximum)
  • Lifetime: τ = ℏ/Γ
  • Peak cross section: σ_max = 4π/k² × g × Γ_in Γ_out/Γ²

Physical Examples:
  • Δ(1232) baryon: E_R ≈ 1232 MeV, Γ ≈ 120 MeV
  • Z boson: E_R ≈ 91.2 GeV, Γ ≈ 2.5 GeV
  • Hoyle state (¹²C*): E_R ≈ 7.65 MeV, Γ ≈ 8.5 eV

S-Matrix Unitarity:
  S_l = e^(2iδ_l) = (E - E_R + iΓ/2)/(E - E_R - iΓ/2)
  |S_l| = 1 (elastic unitarity)
"""

    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Experiment 206: Breit-Wigner Resonance\n'
                 'Nuclear and Particle Physics Resonance Shapes', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'exp206_breit_wigner_resonance.png'),
                dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/exp206_breit_wigner_resonance.png")


if __name__ == "__main__":
    main()
