"""
Experiment 189: Energy-Momentum Relations and Threshold Energies

This experiment demonstrates relativistic energy-momentum relations
and calculates threshold energies for particle production reactions.

Physical concepts:
- E^2 = (pc)^2 + (mc^2)^2 (energy-momentum relation)
- Invariant mass / center-of-mass energy
- Threshold energy for particle production
- Applications to particle physics experiments
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.relativity import FourMomentum


# Particle masses in MeV/c^2
MASSES = {
    'electron': 0.511,
    'positron': 0.511,
    'muon': 105.7,
    'pion_0': 135.0,
    'pion_pm': 139.6,
    'proton': 938.3,
    'neutron': 939.6,
    'kaon': 493.7,
    'lambda': 1115.7,
    'Z_boson': 91188,
    'W_boson': 80379,
    'higgs': 125100,
}


def energy_from_momentum(p, m, c=1):
    """Calculate energy from momentum: E = sqrt(p^2 c^2 + m^2 c^4)"""
    return np.sqrt(p**2 * c**2 + m**2 * c**4)


def momentum_from_energy(E, m, c=1):
    """Calculate momentum from energy: p = sqrt(E^2 - m^2 c^4) / c"""
    return np.sqrt(E**2 - m**2 * c**4) / c


def lorentz_factor_from_energy(E, m, c=1):
    """Calculate Lorentz factor: gamma = E / (m c^2)"""
    return E / (m * c**2)


def velocity_from_energy(E, m, c=1):
    """Calculate velocity: v = p c^2 / E"""
    p = momentum_from_energy(E, m, c)
    return p * c**2 / E


def threshold_energy_fixed_target(m_projectile, m_target, m_products_total, c=1):
    """
    Calculate threshold energy for particle production in fixed-target collision.

    E_threshold = (sum(m_products)^2 - m_proj^2 - m_targ^2) * c^4 / (2 * m_targ * c^2)

    This is the minimum kinetic energy needed in the lab frame.

    Args:
        m_projectile: Mass of incoming particle
        m_target: Mass of target particle (at rest)
        m_products_total: Sum of product particle masses
        c: Speed of light

    Returns:
        Threshold total energy of projectile in lab frame
    """
    M = m_products_total
    m1 = m_projectile
    m2 = m_target

    # s = (E1 + m2 c^2)^2 - p1^2 c^2 = M^2 c^4 at threshold (all products at rest in CM)
    # E_threshold = (M^2 - m1^2 - m2^2) c^4 / (2 m2 c^2)
    E_threshold = (M**2 - m1**2 - m2**2) * c**4 / (2 * m2 * c**2) + m1 * c**2

    return E_threshold


def invariant_mass(E_total, p_total, c=1):
    """Calculate invariant mass: M = sqrt(E^2 - p^2 c^2) / c^2"""
    return np.sqrt(E_total**2 - (p_total * c)**2) / c**2


def cm_energy_collider(E1, E2, m1, m2, c=1):
    """
    Calculate center-of-mass energy for collider.

    sqrt(s) = sqrt((E1 + E2)^2 - (p1 + p2)^2 c^2)

    For head-on collision with equal momenta (antiparallel):
    sqrt(s) = sqrt(2 E1 E2 + 2 p1 p2 c^2 + m1^2 c^4 + m2^2 c^4)
    """
    p1 = momentum_from_energy(E1, m1, c)
    p2 = momentum_from_energy(E2, m2, c)

    # Head-on collision: p_total = p1 - p2 (if both coming from opposite directions)
    # For symmetric collider: p1 = p2, so p_total = 0

    # E_total = E1 + E2
    E_total = E1 + E2

    # For head-on: s = (E1 + E2)^2 - (p1 - p2)^2 c^2
    # For same particles with same energy: s = 4 E^2
    s = (E1 + E2)**2 - (p1 - p2)**2 * c**2

    return np.sqrt(s)


def main():
    c = 1  # Natural units (MeV for energy, MeV/c for momentum)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ==========================================================================
    # Plot 1: Energy-momentum relation
    # ==========================================================================
    ax1 = axes[0, 0]

    m_proton = MASSES['proton']
    m_electron = MASSES['electron']

    p_range = np.linspace(0, 5000, 500)  # MeV/c

    E_proton = energy_from_momentum(p_range, m_proton)
    E_electron = energy_from_momentum(p_range, m_electron)

    # Also plot massless case (E = pc)
    E_massless = p_range * c

    ax1.plot(p_range, E_proton, 'b-', lw=2, label=f'Proton (m={m_proton:.1f} MeV/c^2)')
    ax1.plot(p_range, E_electron, 'r-', lw=2, label=f'Electron (m={m_electron:.3f} MeV/c^2)')
    ax1.plot(p_range, E_massless, 'g--', lw=2, label='Massless (E=pc)')

    ax1.set_xlabel('Momentum p (MeV/c)')
    ax1.set_ylabel('Energy E (MeV)')
    ax1.set_title('Energy-Momentum Relation: E^2 = (pc)^2 + (mc^2)^2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Mark rest mass energies
    ax1.axhline(y=m_proton, color='blue', linestyle=':', alpha=0.5)
    ax1.axhline(y=m_electron, color='red', linestyle=':', alpha=0.5)

    # Add inset for low momentum
    axins = ax1.inset_axes([0.5, 0.15, 0.45, 0.35])
    p_low = np.linspace(0, 100, 100)
    axins.plot(p_low, energy_from_momentum(p_low, m_proton), 'b-', lw=2)
    axins.plot(p_low, energy_from_momentum(p_low, m_electron), 'r-', lw=2)
    axins.plot(p_low, p_low, 'g--', lw=2)
    axins.set_xlim(0, 100)
    axins.set_ylim(0, 1200)
    axins.set_title('Low momentum regime', fontsize=8)
    axins.grid(True, alpha=0.3)

    # ==========================================================================
    # Plot 2: Kinetic energy vs total energy
    # ==========================================================================
    ax2 = axes[0, 1]

    gamma_range = np.linspace(1, 100, 500)
    beta_range = np.sqrt(1 - 1/gamma_range**2)

    # For proton
    E_total = gamma_range * m_proton
    KE = (gamma_range - 1) * m_proton
    p = gamma_range * m_proton * beta_range

    ax2.loglog(gamma_range, E_total, 'b-', lw=2, label='Total energy E')
    ax2.loglog(gamma_range, KE, 'r-', lw=2, label='Kinetic energy K')
    ax2.loglog(gamma_range, p, 'g-', lw=2, label='Momentum p (MeV/c)')

    ax2.axhline(y=m_proton, color='purple', linestyle='--', lw=1.5,
               label=f'Rest mass m_p = {m_proton:.1f} MeV')

    ax2.set_xlabel('Lorentz factor gamma')
    ax2.set_ylabel('Energy (MeV) or Momentum (MeV/c)')
    ax2.set_title('Proton: Energy Components vs Lorentz Factor')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')

    # Mark transition regions
    ax2.axvline(x=2, color='gray', linestyle=':', alpha=0.5)
    ax2.text(2.2, m_proton*50, 'K ~ mc^2', fontsize=9, rotation=90)

    # ==========================================================================
    # Plot 3: Threshold energies for various reactions
    # ==========================================================================
    ax3 = axes[1, 0]

    reactions = [
        {
            'name': 'p + p -> p + p + pi0',
            'm_proj': MASSES['proton'],
            'm_targ': MASSES['proton'],
            'm_prod': 2*MASSES['proton'] + MASSES['pion_0'],
        },
        {
            'name': 'p + p -> p + n + pi+',
            'm_proj': MASSES['proton'],
            'm_targ': MASSES['proton'],
            'm_prod': MASSES['proton'] + MASSES['neutron'] + MASSES['pion_pm'],
        },
        {
            'name': 'p + p -> p + Lambda + K+',
            'm_proj': MASSES['proton'],
            'm_targ': MASSES['proton'],
            'm_prod': MASSES['proton'] + MASSES['lambda'] + MASSES['kaon'],
        },
        {
            'name': 'gamma + p -> p + e+ + e-',
            'm_proj': 0,  # photon
            'm_targ': MASSES['proton'],
            'm_prod': MASSES['proton'] + 2*MASSES['electron'],
        },
        {
            'name': 'e- + e+ -> mu- + mu+',
            'm_proj': MASSES['electron'],
            'm_targ': MASSES['positron'],
            'm_prod': 2*MASSES['muon'],
        },
    ]

    thresholds = []
    kinetic_thresholds = []
    names = []

    for rxn in reactions:
        E_th = threshold_energy_fixed_target(
            rxn['m_proj'], rxn['m_targ'], rxn['m_prod'], c
        )
        K_th = E_th - rxn['m_proj'] * c**2
        thresholds.append(E_th)
        kinetic_thresholds.append(K_th)
        names.append(rxn['name'])

    y_pos = np.arange(len(reactions))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(reactions)))

    bars = ax3.barh(y_pos, kinetic_thresholds, color=colors, alpha=0.7, edgecolor='black')

    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(names, fontsize=9)
    ax3.set_xlabel('Threshold Kinetic Energy (MeV)')
    ax3.set_title('Threshold Energies for Particle Production\n(Fixed Target)')
    ax3.grid(True, alpha=0.3, axis='x')

    # Add values on bars
    for bar, K in zip(bars, kinetic_thresholds):
        width = bar.get_width()
        ax3.text(width + 50, bar.get_y() + bar.get_height()/2,
                f'{K:.0f} MeV', va='center', fontsize=9)

    ax3.set_xscale('log')
    ax3.set_xlim(1, max(kinetic_thresholds) * 3)

    # ==========================================================================
    # Plot 4: Fixed target vs collider comparison
    # ==========================================================================
    ax4 = axes[1, 1]

    # Compare sqrt(s) achievable with same beam energy
    E_beam = np.logspace(0, 6, 100)  # MeV

    m = MASSES['proton']

    # Fixed target: projectile hits stationary target
    # s = m^2 + m^2 + 2 m E = 2m^2 + 2mE
    sqrt_s_fixed = np.sqrt(2 * m**2 + 2 * m * E_beam)

    # Collider: two beams with equal energy head-on
    # s = (2E)^2 = 4 E^2
    sqrt_s_collider = 2 * E_beam

    ax4.loglog(E_beam/1000, sqrt_s_fixed/1000, 'b-', lw=2,
              label='Fixed target: sqrt(s) ~ sqrt(2mE)')
    ax4.loglog(E_beam/1000, sqrt_s_collider/1000, 'r-', lw=2,
              label='Collider: sqrt(s) = 2E')

    ax4.set_xlabel('Beam Energy (GeV)')
    ax4.set_ylabel('Center-of-Mass Energy sqrt(s) (GeV)')
    ax4.set_title('Collider vs Fixed Target Efficiency')
    ax4.legend()
    ax4.grid(True, alpha=0.3, which='both')

    # Mark LHC energy
    E_LHC = 6500 * 1000  # 6.5 TeV per beam in MeV
    sqrt_s_LHC = 13000 * 1000  # 13 TeV
    ax4.axvline(x=E_LHC/1e6, color='purple', linestyle='--', alpha=0.7)
    ax4.axhline(y=sqrt_s_LHC/1e6, color='purple', linestyle='--', alpha=0.7)
    ax4.plot(E_LHC/1e6, sqrt_s_LHC/1e6, 'p', color='purple', markersize=15)
    ax4.text(E_LHC/1e6 * 1.5, sqrt_s_LHC/1e6, 'LHC\n(13 TeV)', fontsize=10, color='purple')

    # Mark Higgs mass
    ax4.axhline(y=MASSES['higgs']/1000, color='green', linestyle=':', alpha=0.7)
    ax4.text(1, MASSES['higgs']/1000 * 1.2, 'Higgs mass', fontsize=9, color='green')

    plt.suptitle('Relativistic Energy-Momentum and Threshold Energies\n'
                 'E^2 = (pc)^2 + (mc^2)^2',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Print reaction thresholds
    print("Threshold Energies for Particle Production (Fixed Target):")
    print("=" * 70)
    for rxn, E_th, K_th in zip(reactions, thresholds, kinetic_thresholds):
        print(f"\n{rxn['name']}")
        print(f"  Product masses: {rxn['m_prod']:.1f} MeV/c^2")
        print(f"  Threshold total energy: {E_th:.1f} MeV")
        print(f"  Threshold kinetic energy: {K_th:.1f} MeV")
        gamma = E_th / rxn['m_proj'] if rxn['m_proj'] > 0 else np.inf
        print(f"  Required gamma: {gamma:.2f}")

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'energy_momentum_threshold.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")


if __name__ == "__main__":
    main()
