"""
Experiment 210: Binding Energy Curve (Semi-Empirical Mass Formula)

Demonstrates the Weizsacker semi-empirical mass formula (liquid drop model)
for nuclear binding energies. Shows the famous B/A curve and its features.

Physics:
- B(A,Z) = aV·A - aS·A^(2/3) - aC·Z²/A^(1/3) - aA·(A-2Z)²/A + δ(A,Z)
- Volume, surface, Coulomb, asymmetry, and pairing terms
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.nuclear import LiquidDropModel


def semi_empirical_mass_formula(A, Z, a_V=15.8, a_S=18.3, a_C=0.714,
                                 a_A=23.2, a_P=12.0):
    """
    Calculate binding energy using the semi-empirical mass formula.

    Args:
        A: Mass number
        Z: Atomic number
        a_V: Volume coefficient (MeV)
        a_S: Surface coefficient (MeV)
        a_C: Coulomb coefficient (MeV)
        a_A: Asymmetry coefficient (MeV)
        a_P: Pairing coefficient (MeV)

    Returns:
        Binding energy in MeV
    """
    N = A - Z

    # Volume term (bulk nuclear matter)
    B = a_V * A

    # Surface term (surface tension)
    B -= a_S * A**(2/3)

    # Coulomb term (proton repulsion)
    B -= a_C * Z**2 / A**(1/3)

    # Asymmetry term (N ≠ Z penalty)
    B -= a_A * (A - 2*Z)**2 / A

    # Pairing term
    if Z % 2 == 0 and N % 2 == 0:
        delta = a_P / np.sqrt(A)  # even-even
    elif Z % 2 == 1 and N % 2 == 1:
        delta = -a_P / np.sqrt(A)  # odd-odd
    else:
        delta = 0  # odd-A

    B += delta

    return B


def most_stable_Z(A, a_C=0.714, a_A=23.2):
    """
    Find the most stable Z for a given A (valley of stability).

    Minimizing B w.r.t. Z:
    Z_opt = A / (2 + a_C·A^(2/3)/(2·a_A))
    """
    return A / (2 + a_C * A**(2/3) / (2 * a_A))


def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: B/A curve for stable nuclei
    ax = axes[0, 0]

    # Stable nuclei (approximate Z values)
    stable_nuclei = [
        (1, 1), (2, 1), (4, 2), (12, 6), (16, 8), (27, 13), (40, 20),
        (56, 26), (63, 29), (90, 40), (107, 47), (127, 53), (138, 56),
        (181, 73), (197, 79), (208, 82), (238, 92)
    ]

    A_stable = [n[0] for n in stable_nuclei]
    Z_stable = [n[1] for n in stable_nuclei]
    B_A_stable = [semi_empirical_mass_formula(A, Z) / A for A, Z in stable_nuclei]

    ax.plot(A_stable, B_A_stable, 'bo-', markersize=8, lw=2, label='SEMF')

    # Experimental values (approximate)
    B_A_exp = {
        4: 7.07, 12: 7.68, 16: 7.98, 27: 8.33, 40: 8.55, 56: 8.79,
        63: 8.75, 90: 8.71, 107: 8.55, 127: 8.45, 138: 8.37,
        181: 8.02, 197: 7.92, 208: 7.87, 238: 7.57
    }

    A_exp = list(B_A_exp.keys())
    BA_exp = list(B_A_exp.values())
    ax.plot(A_exp, BA_exp, 'rs', markersize=8, label='Experiment (approx)')

    ax.set_xlabel('Mass Number A')
    ax.set_ylabel('Binding Energy per Nucleon B/A (MeV)')
    ax.set_title('Nuclear Binding Energy Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 250)
    ax.set_ylim(0, 9)

    # Mark Fe-56 (maximum)
    ax.axhline(y=8.79, color='g', linestyle='--', alpha=0.5)
    ax.annotate('Fe-56', xy=(56, 8.79), xytext=(80, 8.9),
                arrowprops=dict(arrowstyle='->', color='green'))

    # Plot 2: Individual terms
    ax = axes[0, 1]

    A_range = np.linspace(10, 250, 200)
    Z_opt = [most_stable_Z(A) for A in A_range]

    # Calculate each term
    volume = 15.8 * A_range / A_range
    surface = -18.3 * A_range**(2/3) / A_range
    coulomb = [-0.714 * Z**2 / A**(4/3) for A, Z in zip(A_range, Z_opt)]
    asymmetry = [-23.2 * (A - 2*Z)**2 / A**2 for A, Z in zip(A_range, Z_opt)]

    ax.plot(A_range, volume, 'b-', lw=2, label='Volume (+aV)')
    ax.plot(A_range, surface, 'r-', lw=2, label='Surface (-aS)')
    ax.plot(A_range, coulomb, 'g-', lw=2, label='Coulomb (-aC)')
    ax.plot(A_range, asymmetry, 'm-', lw=2, label='Asymmetry (-aA)')

    B_A_total = [semi_empirical_mass_formula(int(A), round(Z)) / A
                 for A, Z in zip(A_range, Z_opt)]
    ax.plot(A_range, B_A_total, 'k-', lw=2, label='Total B/A')

    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Mass Number A')
    ax.set_ylabel('Contribution to B/A (MeV)')
    ax.set_title('SEMF Terms (per nucleon)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 3: Valley of stability
    ax = axes[0, 2]

    A_valley = np.arange(1, 250)
    Z_valley = [round(most_stable_Z(A)) for A in A_valley]
    N_valley = A_valley - Z_valley

    ax.plot(N_valley, Z_valley, 'b-', lw=2, label='Valley of stability')
    ax.plot(A_valley, A_valley, 'k--', lw=1, alpha=0.5, label='N = Z')

    # Mark magic numbers
    magic = [2, 8, 20, 28, 50, 82, 126]
    for m in magic:
        if m < 130:
            ax.axhline(y=m, color='r', linestyle=':', alpha=0.3)
            ax.axvline(x=m, color='b', linestyle=':', alpha=0.3)

    ax.set_xlabel('Neutron Number N')
    ax.set_ylabel('Proton Number Z')
    ax.set_title('Valley of Stability\n(SEMF prediction)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')

    # Plot 4: Isobaric parabola (mass parabola)
    ax = axes[1, 0]

    A_fixed = 100

    Z_range = np.arange(35, 55)
    B_values = [semi_empirical_mass_formula(A_fixed, Z) for Z in Z_range]

    # Mass excess ∝ -B
    mass_excess = [-B for B in B_values]

    # Even-even and odd-odd
    even_even = [Z for Z in Z_range if Z % 2 == 0 and (A_fixed - Z) % 2 == 0]
    odd_odd = [Z for Z in Z_range if Z % 2 == 1 and (A_fixed - Z) % 2 == 1]

    ax.plot(Z_range, mass_excess, 'b-', lw=1, alpha=0.5)

    for Z in even_even:
        B = semi_empirical_mass_formula(A_fixed, Z)
        ax.plot(Z, -B, 'go', markersize=10, label='even-even' if Z == even_even[0] else '')
    for Z in odd_odd:
        B = semi_empirical_mass_formula(A_fixed, Z)
        ax.plot(Z, -B, 'rs', markersize=10, label='odd-odd' if Z == odd_odd[0] else '')

    ax.set_xlabel('Proton Number Z')
    ax.set_ylabel('Mass Excess ∝ -B (MeV)')
    ax.set_title(f'Mass Parabola for A = {A_fixed}\n(Isobaric analog states)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Separation energies
    ax = axes[1, 1]

    A_sep = np.arange(20, 210)

    S_n = []  # Neutron separation energy
    S_p = []  # Proton separation energy

    for A in A_sep:
        Z = round(most_stable_Z(A))

        B_A = semi_empirical_mass_formula(A, Z)
        B_Am1_n = semi_empirical_mass_formula(A-1, Z)  # Remove neutron
        B_Am1_p = semi_empirical_mass_formula(A-1, Z-1)  # Remove proton

        S_n.append(B_A - B_Am1_n)
        S_p.append(B_A - B_Am1_p)

    ax.plot(A_sep, S_n, 'b-', lw=2, label='Sn (neutron)')
    ax.plot(A_sep, S_p, 'r-', lw=2, label='Sp (proton)')
    ax.axhline(y=8.0, color='k', linestyle='--', alpha=0.5)

    ax.set_xlabel('Mass Number A')
    ax.set_ylabel('Separation Energy (MeV)')
    ax.set_title('Nucleon Separation Energies\nS = B(A,Z) - B(A-1,Z\')')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 15)

    # Plot 6: Summary
    ax = axes[1, 2]
    ax.axis('off')

    summary = """
Semi-Empirical Mass Formula (SEMF)
==================================

Binding Energy:
B(A,Z) = aV·A - aS·A^(2/3) - aC·Z²/A^(1/3)
         - aA·(A-2Z)²/A + δ(A,Z)

Coefficients (MeV):
  aV = 15.8  (Volume - bulk saturation)
  aS = 18.3  (Surface - surface tension)
  aC = 0.714 (Coulomb - proton repulsion)
  aA = 23.2  (Asymmetry - N≠Z penalty)
  aP = 12.0  (Pairing)

Pairing Term:
  δ = +aP/√A  (even-even)
  δ = 0       (odd-A)
  δ = -aP/√A  (odd-odd)

Physical Interpretation:
  • Volume: Nuclear matter is saturated
  • Surface: Nucleons at surface less bound
  • Coulomb: Long-range proton repulsion
  • Asymmetry: Pauli exclusion prefers N≈Z
  • Pairing: Like nucleons pair up

Valley of Stability:
  Z_opt = A / (2 + aC·A^(2/3)/(2aA))

  Light nuclei: N ≈ Z
  Heavy nuclei: N > Z (Coulomb)

Key Features:
  • Maximum B/A at A ≈ 56 (Fe, Ni)
  • Explains fission of heavy nuclei
  • Explains fusion of light nuclei
"""

    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Experiment 210: Nuclear Binding Energy Curve\n'
                 'Semi-Empirical Mass Formula (Liquid Drop Model)', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'exp210_binding_energy.png'),
                dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/exp210_binding_energy.png")


if __name__ == "__main__":
    main()
