"""
Experiment 211: Gamow Tunneling and Fusion Rate

Demonstrates quantum tunneling through the Coulomb barrier
and its role in stellar fusion reactions.

Physics:
- Gamow factor: G = 2πη where η = Z₁Z₂e²/(ℏv)
- Tunneling probability: P ∝ exp(-2πη)
- Gamow peak: Maximum of σ(E)·MB(E)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.nuclear import AlphaDecay, FusionRate


def gamow_factor(E_keV, Z1, Z2, mu_amu):
    """
    Calculate Gamow factor G = 2πη.

    η = Z₁Z₂e²/(ℏv) = Z₁Z₂e²/(ℏ)√(μ/2E)

    Args:
        E_keV: Center of mass energy in keV
        Z1, Z2: Charge numbers
        mu_amu: Reduced mass in amu

    Returns:
        Gamow factor (dimensionless)
    """
    # b = 31.28 Z₁Z₂√μ keV^(1/2) (Gamow parameter)
    b = 31.28 * Z1 * Z2 * np.sqrt(mu_amu)
    eta = b / np.sqrt(E_keV)
    return 2 * np.pi * eta


def tunneling_probability(E_keV, Z1, Z2, mu_amu):
    """Tunneling probability through Coulomb barrier."""
    G = gamow_factor(E_keV, Z1, Z2, mu_amu)
    return np.exp(-G)


def cross_section_sfactor(E_keV, S_keVb, Z1, Z2, mu_amu):
    """
    Cross section from S-factor.

    σ(E) = S(E)/E × exp(-2πη)
    """
    G = gamow_factor(E_keV, Z1, Z2, mu_amu)
    return S_keVb / E_keV * np.exp(-G)


def gamow_peak_energy(T_keV, Z1, Z2, mu_amu):
    """
    Gamow peak energy E₀.

    E₀ = (b·kT/2)^(2/3)
    where b = 31.28·Z₁Z₂√μ keV^(1/2)
    """
    b = 31.28 * Z1 * Z2 * np.sqrt(mu_amu)
    return (b * T_keV / 2)**(2/3)


def gamow_peak_width(T_keV, E0):
    """
    Gamow peak width Δ.

    Δ = 4√(E₀·kT/3)
    """
    return 4 * np.sqrt(E0 * T_keV / 3)


def reaction_rate_integrand(E_keV, T_keV, S_keVb, Z1, Z2, mu_amu):
    """
    Reaction rate integrand σ(E)·v·MB(E).

    ∝ σ(E)·E·exp(-E/kT)
    """
    sigma = cross_section_sfactor(E_keV, S_keVb, Z1, Z2, mu_amu)
    maxwell = np.exp(-E_keV / T_keV)
    return sigma * E_keV * maxwell


def coulomb_barrier(Z1, Z2, R_fm):
    """Coulomb barrier height in MeV."""
    e2 = 1.44  # MeV·fm
    return Z1 * Z2 * e2 / R_fm


def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Coulomb barrier and tunneling
    ax = axes[0, 0]

    # Example: proton + proton
    Z1, Z2 = 1, 1
    R = 1.2  # fm, nuclear radius

    r = np.linspace(0.1, 10, 500)

    # Coulomb potential
    V_c = 1.44 * Z1 * Z2 / r  # MeV

    # Nuclear potential (simplified Woods-Saxon-like)
    V_n = -50 * np.exp(-(r - R)**2 / 0.5**2)  # MeV

    V_total = np.where(r > R, V_c, V_c + V_n)

    ax.plot(r, V_c, 'b--', lw=2, label='Coulomb')
    ax.plot(r, V_n, 'g--', lw=2, label='Nuclear')
    ax.plot(r, V_total, 'r-', lw=2, label='Total')

    # Barrier height
    V_barrier = coulomb_barrier(Z1, Z2, R)
    ax.axhline(y=V_barrier, color='k', linestyle=':', alpha=0.5)
    ax.text(5, V_barrier + 0.1, f'Barrier = {V_barrier:.2f} MeV')

    # Typical thermal energy
    kT_Sun = 1.3  # keV (Sun's core)
    ax.axhline(y=kT_Sun/1000, color='orange', linestyle='--',
               label=f'kT (Sun) ≈ {kT_Sun:.1f} keV')

    ax.set_xlabel('Distance (fm)')
    ax.set_ylabel('Potential Energy (MeV)')
    ax.set_title('Coulomb Barrier\np + p fusion')
    ax.set_xlim(0, 10)
    ax.set_ylim(-60, 5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: Tunneling probability vs energy
    ax = axes[0, 1]

    E_range = np.linspace(0.1, 100, 500)

    # Different reactions
    reactions = [
        ('p + p', 1, 1, 0.5, 'b'),
        ('D + T', 1, 1, 1.2, 'r'),
        ('α + α', 2, 2, 2.0, 'g'),
        ('¹²C + ¹²C', 6, 6, 6.0, 'm'),
    ]

    for name, z1, z2, mu, color in reactions:
        P = tunneling_probability(E_range, z1, z2, mu)
        ax.semilogy(E_range, P, '-', color=color, lw=2, label=name)

    ax.set_xlabel('Energy (keV)')
    ax.set_ylabel('Tunneling Probability')
    ax.set_title('Gamow Factor: P = exp(-2πη)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(1e-30, 1)

    # Plot 3: Gamow peak
    ax = axes[0, 2]

    T = 15  # keV (stellar temperature)
    Z1, Z2 = 1, 1
    mu = 0.5  # amu (p+p)
    S = 4e-22  # keV·barn (p+p S-factor)

    E_peak = np.linspace(0.1, 50, 500)

    # Components
    P_tunnel = tunneling_probability(E_peak, Z1, Z2, mu)
    MB = np.exp(-E_peak / T)

    # Gamow peak
    integrand = P_tunnel * MB * E_peak

    # Normalize for plotting
    P_tunnel_norm = P_tunnel / np.max(P_tunnel)
    MB_norm = MB / np.max(MB)
    integrand_norm = integrand / np.max(integrand)

    ax.plot(E_peak, P_tunnel_norm, 'b-', lw=2, label='Tunneling P(E)')
    ax.plot(E_peak, MB_norm, 'r-', lw=2, label='Maxwell-Boltzmann')
    ax.plot(E_peak, integrand_norm, 'g-', lw=3, label='Gamow Peak')

    # Mark peak position
    E0 = gamow_peak_energy(T, Z1, Z2, mu)
    Delta = gamow_peak_width(T, E0)
    ax.axvline(x=E0, color='k', linestyle='--', alpha=0.5)
    ax.axvspan(E0 - Delta/2, E0 + Delta/2, alpha=0.2, color='green')

    ax.text(E0 + 2, 0.8, f'E₀ = {E0:.1f} keV\nΔ = {Delta:.1f} keV')

    ax.set_xlabel('Energy (keV)')
    ax.set_ylabel('Normalized')
    ax.set_title(f'Gamow Peak (T = {T} keV)\np + p fusion')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 50)

    # Plot 4: Temperature dependence of Gamow peak
    ax = axes[1, 0]

    temperatures = [5, 10, 15, 20, 30]  # keV
    colors = plt.cm.hot(np.linspace(0.2, 0.8, len(temperatures)))

    E_range = np.linspace(0.1, 80, 500)

    for T, color in zip(temperatures, colors):
        integrand = reaction_rate_integrand(E_range, T, S, 1, 1, 0.5)
        integrand_norm = integrand / np.max(integrand)
        ax.plot(E_range, integrand_norm, '-', color=color, lw=2,
                label=f'T = {T} keV')

        # Mark peak
        E0 = gamow_peak_energy(T, 1, 1, 0.5)
        ax.axvline(x=E0, color=color, linestyle=':', alpha=0.5)

    ax.set_xlabel('Energy (keV)')
    ax.set_ylabel('Reaction Rate (normalized)')
    ax.set_title('Gamow Peak vs Temperature')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Fusion reactivity
    ax = axes[1, 1]

    T_range = np.linspace(1, 100, 200)

    # Use FusionRate class
    dt = FusionRate('DT')
    dd = FusionRate('DD')
    dhe3 = FusionRate('DHe3')

    sigma_v_dt = dt.reactivity(T_range)
    sigma_v_dd = dd.reactivity(T_range)
    sigma_v_dhe3 = dhe3.reactivity(T_range)

    ax.loglog(T_range, sigma_v_dt, 'r-', lw=2, label='D-T')
    ax.loglog(T_range, sigma_v_dd, 'b-', lw=2, label='D-D')
    ax.loglog(T_range, sigma_v_dhe3, 'g-', lw=2, label='D-³He')

    ax.axvline(x=10, color='k', linestyle='--', alpha=0.5,
               label='Fusion reactor T')

    ax.set_xlabel('Temperature (keV)')
    ax.set_ylabel('<σv> (m³/s)')
    ax.set_title('Fusion Reactivity\n(Maxwellian-averaged)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 100)

    # Plot 6: Summary
    ax = axes[1, 2]
    ax.axis('off')

    summary = """
Gamow Tunneling and Stellar Fusion
==================================

Coulomb Barrier:
  V_c = Z₁Z₂e²/r
  Barrier height: V_B = Z₁Z₂e²/R_nuc

Gamow Factor:
  G = 2πη = b/√E
  b = 31.28·Z₁Z₂√μ  keV^(1/2)
  η = Z₁Z₂e²/(ℏv) (Sommerfeld)

Tunneling Probability:
  P(E) = exp(-G) = exp(-b/√E)

Cross Section:
  σ(E) = S(E)/E × exp(-b/√E)
  S(E) = "astrophysical S-factor"
        (slowly varying)

Gamow Peak:
  Most reactions occur at E₀:
  E₀ = (bkT/2)^(2/3)

  Peak width: Δ = 4√(E₀kT/3)

Stellar Fusion Examples:
  Sun core: T ≈ 15 MK ≈ 1.3 keV
  p+p: E₀ ≈ 6 keV, Δ ≈ 5 keV

  Fusion reactor: T ≈ 10-20 keV
  D+T: E₀ ≈ 64 keV at 10 keV

Key Point:
  Despite E << V_B,
  quantum tunneling enables fusion!
"""

    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Experiment 211: Gamow Tunneling and Fusion Rate\n'
                 'Quantum Tunneling Through Coulomb Barrier', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'exp211_gamow_tunneling.png'),
                dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/exp211_gamow_tunneling.png")


if __name__ == "__main__":
    main()
