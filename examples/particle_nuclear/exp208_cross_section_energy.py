"""
Experiment 208: Cross Section vs Energy

Demonstrates energy dependence of nuclear reaction cross sections.
Shows threshold behavior, resonances, and high-energy scaling.

Physics:
- Threshold: σ ∝ √(E - E_th) near threshold
- Resonance: Breit-Wigner peaks
- High energy: σ → const or ∝ log²(s)
- Gamow peak for fusion
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.nuclear import NuclearCrossSection, ResonanceFormula


def threshold_cross_section(E, E_th, sigma_0, power=0.5):
    """
    Cross section near threshold.

    σ(E) = σ₀ × (E - E_th)^power for E > E_th
    """
    E = np.asarray(E)
    sigma = np.zeros_like(E)
    mask = E > E_th
    sigma[mask] = sigma_0 * (E[mask] - E_th)**power
    return sigma


def gamow_peak(E, S_factor, Z1, Z2, T_keV, mu_amu):
    """
    Cross section with Gamow peak for stellar fusion.

    σ(E) = S(E)/E × exp(-2πη)
    where η = Z₁Z₂e²/(ℏv) is Sommerfeld parameter.
    """
    E = np.asarray(E)

    # Gamow energy (keV)
    E_G = 0.979 * (Z1**2 * Z2**2 * mu_amu)**(1/3) * T_keV**(2/3) * 1000

    # Approximate: σ ∝ S(E)/E × exp(-√(E_G/E))
    # More precisely for thermal distribution
    b = 31.28 * Z1 * Z2 * np.sqrt(mu_amu)  # keV^1/2

    with np.errstate(divide='ignore', invalid='ignore'):
        sigma = S_factor / E * np.exp(-b / np.sqrt(E))
        sigma = np.where(E > 0, sigma, 0)

    return sigma


def coulomb_cross_section(E, Z1, Z2, E_c):
    """
    Coulomb-modified cross section.

    σ(E) = σ_geometric × (1 - E_c/E) for E > E_c
    """
    E = np.asarray(E)
    sigma = np.zeros_like(E)
    mask = E > E_c
    sigma[mask] = np.pi * (1.4)**2 * 100 * (1 - E_c / E[mask])  # fm² = 0.01 barn
    return sigma


def compound_nucleus_sigma(E, E_th, sigma_max, a):
    """
    Compound nucleus cross section with many overlapping resonances.
    """
    E = np.asarray(E)
    sigma = np.zeros_like(E)
    mask = E > E_th
    x = (E[mask] - E_th) / a
    sigma[mask] = sigma_max * (1 - np.exp(-x)) * np.exp(-x/10)
    return sigma


def optical_model_sigma(E, R, W):
    """
    Optical model cross section (absorption).

    σ_abs ≈ π R² × (1 - V(E)/E)
    """
    k = np.sqrt(E)  # Simplified
    sigma_geo = np.pi * R**2
    transmission = 1 - np.exp(-W / E)
    return sigma_geo * transmission


def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Threshold behavior
    ax = axes[0, 0]

    E = np.linspace(0, 20, 500)
    E_th = 5.0  # MeV threshold

    # Different power laws
    for power, label in [(0.5, 's-wave'), (1.5, 'p-wave'), (2.5, 'd-wave')]:
        sigma = threshold_cross_section(E, E_th, sigma_0=10, power=power)
        ax.plot(E, sigma, lw=2, label=f'{label} (n = {power})')

    ax.axvline(x=E_th, color='k', linestyle='--', alpha=0.5, label='Threshold')
    ax.set_xlabel('Energy (MeV)')
    ax.set_ylabel('Cross Section (mb)')
    ax.set_title('Threshold Behavior\nσ ∝ (E - E_th)^n')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 20)

    # Plot 2: Resonance structure
    ax = axes[0, 1]

    E_res = np.linspace(0, 30, 1000)

    # Background plus resonances
    sigma_bg = 10 * np.ones_like(E_res)  # mb background

    # Add resonances
    resonances = [
        (5.0, 0.5, 50),   # E_R, Gamma, peak
        (12.0, 1.0, 100),
        (18.0, 2.0, 80),
        (25.0, 0.3, 200),  # Narrow resonance
    ]

    sigma_total = sigma_bg.copy()
    for E_R, Gamma, peak in resonances:
        sigma_res = peak * (Gamma/2)**2 / ((E_res - E_R)**2 + (Gamma/2)**2)
        sigma_total += sigma_res

    ax.semilogy(E_res, sigma_total, 'b-', lw=2)
    ax.semilogy(E_res, sigma_bg, 'k--', lw=1, alpha=0.5, label='Background')

    for E_R, Gamma, peak in resonances:
        ax.axvline(x=E_R, color='r', linestyle=':', alpha=0.3)

    ax.set_xlabel('Energy (MeV)')
    ax.set_ylabel('Cross Section (mb)')
    ax.set_title('Resonance Structure\n(Compound nucleus)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Gamow peak for fusion
    ax = axes[0, 2]

    # D-T fusion
    Z1, Z2 = 1, 1
    mu_amu = 2.5  # Reduced mass in amu
    S_factor = 20e-3  # keV·barn

    E_keV = np.linspace(1, 100, 500)

    temperatures = [5, 10, 20, 50]  # keV
    colors = plt.cm.hot(np.linspace(0.2, 0.8, len(temperatures)))

    for T, color in zip(temperatures, colors):
        # Maxwell-Boltzmann weight
        MB = np.exp(-E_keV / T) / T

        # Cross section
        sigma = gamow_peak(E_keV, S_factor, Z1, Z2, T, mu_amu)

        # Gamow peak is product
        peak = sigma * E_keV * MB
        peak = peak / np.max(peak)  # Normalize

        ax.plot(E_keV, peak, '-', color=color, lw=2, label=f'T = {T} keV')

    ax.set_xlabel('Energy (keV)')
    ax.set_ylabel('Reaction Rate (normalized)')
    ax.set_title('Gamow Peak for D-T Fusion\nσ(E) × E × exp(-E/kT)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: 1/v law for neutron capture
    ax = axes[1, 0]

    v = np.linspace(0.1, 10, 500)  # Relative velocity (arb. units)

    # 1/v law: σ ∝ 1/v
    sigma_1v = 100 / v  # barn

    # With resonance
    v_res = 3.0
    Gamma_v = 0.5
    sigma_res = 500 * (Gamma_v/2)**2 / ((v - v_res)**2 + (Gamma_v/2)**2)
    sigma_with_res = sigma_1v + sigma_res

    ax.loglog(v, sigma_1v, 'b-', lw=2, label='1/v law')
    ax.loglog(v, sigma_with_res, 'r-', lw=2, label='With resonance')

    ax.set_xlabel('Velocity (arb. units)')
    ax.set_ylabel('Cross Section (barn)')
    ax.set_title('Neutron Capture Cross Section\n1/v law + resonance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: High-energy behavior
    ax = axes[1, 1]

    # Proton-proton total cross section
    s = np.logspace(1, 5, 500)  # GeV²
    sqrt_s = np.sqrt(s)

    # Froissart-Martin bound: σ ≤ const × ln²(s/s₀)
    s0 = 1  # GeV²
    sigma_pp_approx = 20 + 0.5 * np.log(s/s0)**2  # mb

    # Pomeron exchange fit
    sigma_pomeron = 21.7 * (s/s0)**0.0808 + 56.1 * (s/s0)**(-0.4525)

    ax.semilogx(sqrt_s, sigma_pp_approx, 'b-', lw=2, label='Froissart bound')
    ax.semilogx(sqrt_s, sigma_pomeron, 'r--', lw=2, label='Regge/Pomeron fit')

    ax.set_xlabel('√s (GeV)')
    ax.set_ylabel('Total Cross Section (mb)')
    ax.set_title('High-Energy pp Cross Section\nσ_tot ∝ ln²(s)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Summary of energy dependences
    ax = axes[1, 2]
    ax.axis('off')

    summary = """
Cross Section Energy Dependence
===============================

1. THRESHOLD BEHAVIOR:
   σ ∝ (E - E_th)^(l+1/2)
   l = 0 (s-wave): σ ∝ √(E - E_th)
   l = 1 (p-wave): σ ∝ (E - E_th)^(3/2)

2. COULOMB BARRIER:
   σ ∝ exp(-2πη)
   η = Z₁Z₂e²/(ℏv) (Sommerfeld parameter)

3. RESONANCES (Breit-Wigner):
   σ = σ_peak × Γ²/4 / [(E-E_R)² + Γ²/4]

4. 1/v LAW (thermal neutrons):
   σ ∝ 1/v for s-wave capture

5. GAMOW PEAK (stellar fusion):
   σ = S(E)/E × exp(-b/√E)
   Peak at E_0 = (bkT/2)^(2/3)

6. HIGH ENERGY (Froissart bound):
   σ_tot ≤ C × ln²(s/s₀)
   Protons: σ rises slowly

7. GEOMETRIC LIMIT:
   σ_geo = πR² ≈ π(r₀A^(1/3))²
   Nuclear radius: R ≈ 1.2 A^(1/3) fm

8. OPTICAL MODEL:
   σ_abs = σ_geo × transmission
   σ_el depends on surface diffuseness
"""

    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Experiment 208: Cross Section vs Energy\n'
                 'Nuclear Reaction Energy Dependences', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'exp208_cross_section_energy.png'),
                dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/exp208_cross_section_energy.png")


if __name__ == "__main__":
    main()
