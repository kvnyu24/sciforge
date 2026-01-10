"""
Example demonstrating entropy of mixing for ideal gases.

When two ideal gases mix at constant temperature and pressure,
the entropy change is:

Delta_S_mix = -nR * sum(x_i * ln(x_i))

where x_i is the mole fraction of component i.

This example shows:
- Entropy of mixing for binary mixtures
- Effect of number of components
- Comparison with non-ideal mixing
- Gibbs paradox illustration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.core.constants import CONSTANTS


def entropy_of_mixing_binary(x1, n_total=1.0, R=CONSTANTS['R']):
    """
    Calculate entropy of mixing for binary ideal gas mixture.

    Delta_S = -nR * [x1*ln(x1) + x2*ln(x2)]

    Args:
        x1: Mole fraction of component 1 (0 to 1)
        n_total: Total moles
        R: Gas constant

    Returns:
        Entropy of mixing (J/K)
    """
    x1 = np.asarray(x1)
    x2 = 1 - x1

    # Handle edge cases (pure components)
    result = np.zeros_like(x1, dtype=float)
    mask = (x1 > 0) & (x1 < 1)
    result[mask] = -n_total * R * (x1[mask] * np.log(x1[mask]) +
                                    x2[mask] * np.log(x2[mask]))
    return result


def entropy_of_mixing_multicomponent(mole_fractions, n_total=1.0, R=CONSTANTS['R']):
    """
    Calculate entropy of mixing for multicomponent ideal gas mixture.

    Args:
        mole_fractions: Array of mole fractions (must sum to 1)
        n_total: Total moles
        R: Gas constant

    Returns:
        Entropy of mixing (J/K)
    """
    x = np.asarray(mole_fractions)
    # Filter out zero mole fractions to avoid log(0)
    x_nonzero = x[x > 0]
    return -n_total * R * np.sum(x_nonzero * np.log(x_nonzero))


def gibbs_free_energy_of_mixing(x1, n_total=1.0, T=298.15, R=CONSTANTS['R']):
    """
    Calculate Gibbs free energy of mixing for ideal gas.

    Delta_G_mix = nRT * sum(x_i * ln(x_i))
    (negative of T * Delta_S_mix for ideal mixing at constant T, P)
    """
    x1 = np.asarray(x1)
    x2 = 1 - x1
    result = np.zeros_like(x1, dtype=float)
    mask = (x1 > 0) & (x1 < 1)
    result[mask] = n_total * R * T * (x1[mask] * np.log(x1[mask]) +
                                       x2[mask] * np.log(x2[mask]))
    return result


def regular_solution_entropy(x1, W=0, n_total=1.0, T=298.15, R=CONSTANTS['R']):
    """
    Calculate excess entropy for regular solution model.

    For regular solution: S_excess = 0 (mixing is ideal in terms of entropy)
    But G_excess = W * x1 * x2 (non-zero enthalpy of mixing)

    Args:
        x1: Mole fraction of component 1
        W: Interaction parameter (J/mol)

    Returns:
        Total entropy of mixing (same as ideal for regular solution)
    """
    return entropy_of_mixing_binary(x1, n_total, R)


def main():
    R = CONSTANTS['R']
    n_total = 1.0  # 1 mole total
    T = 298.15     # K (25 C)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Entropy of mixing for binary mixture
    ax1 = axes[0, 0]

    x1 = np.linspace(0.001, 0.999, 200)
    delta_S = entropy_of_mixing_binary(x1, n_total, R)

    ax1.plot(x1, delta_S, 'b-', lw=2, label='Ideal mixing')
    ax1.axhline(y=R * np.log(2), color='r', linestyle='--', lw=1.5,
                label=f'Maximum at x=0.5: R*ln(2) = {R*np.log(2):.3f} J/(mol K)')

    # Mark maximum
    ax1.plot(0.5, R * np.log(2), 'ro', markersize=10)

    ax1.set_xlabel('Mole Fraction of Component 1 (x1)', fontsize=12)
    ax1.set_ylabel('Entropy of Mixing (J/(mol K))', fontsize=12)
    ax1.set_title('Entropy of Mixing: Binary Ideal Gas Mixture', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.1 * R * np.log(2))

    # Add annotation
    ax1.text(0.7, 3.0, r'$\Delta S_{mix} = -nR\sum x_i \ln x_i$',
             fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Plot 2: Effect of number of components
    ax2 = axes[0, 1]

    n_components = range(2, 11)
    max_entropies = []

    for n_comp in n_components:
        # Equal molar mixture gives maximum entropy
        x_equal = np.ones(n_comp) / n_comp
        S_max = entropy_of_mixing_multicomponent(x_equal, n_total, R)
        max_entropies.append(S_max)

    ax2.bar(n_components, max_entropies, color='steelblue', alpha=0.7,
            edgecolor='black', label='Maximum entropy (equal molar)')

    # Theoretical: S_max = R * ln(N)
    n_range = np.linspace(2, 10, 100)
    S_theoretical = R * np.log(n_range)
    ax2.plot(n_range, S_theoretical, 'r-', lw=2, label=r'$S_{max} = R \ln N$')

    ax2.set_xlabel('Number of Components (N)', fontsize=12)
    ax2.set_ylabel('Maximum Entropy of Mixing (J/(mol K))', fontsize=12)
    ax2.set_title('Maximum Mixing Entropy vs Number of Components', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for n, s in zip(n_components, max_entropies):
        ax2.text(n, s + 0.2, f'{s:.2f}', ha='center', fontsize=9)

    # Plot 3: Gibbs free energy of mixing
    ax3 = axes[1, 0]

    delta_G = gibbs_free_energy_of_mixing(x1, n_total, T, R)
    T_delta_S = T * delta_S

    ax3.plot(x1, delta_G / 1000, 'b-', lw=2, label=r'$\Delta G_{mix}$')
    ax3.plot(x1, -T_delta_S / 1000, 'r--', lw=2, label=r'$-T\Delta S_{mix}$')
    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)

    ax3.set_xlabel('Mole Fraction of Component 1 (x1)', fontsize=12)
    ax3.set_ylabel('Energy (kJ/mol)', fontsize=12)
    ax3.set_title(f'Gibbs Free Energy of Mixing at T = {T:.1f}K', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 1)

    # Annotation about spontaneity
    ax3.annotate('Mixing is always spontaneous\nfor ideal gases\n(Delta G < 0)',
                xy=(0.5, delta_G[len(x1)//2]/1000),
                xytext=(0.7, -1.0),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # Plot 4: Gibbs Paradox illustration
    ax4 = axes[1, 1]

    # When gases are identical, there should be no entropy of mixing
    x1_range = np.linspace(0.01, 0.99, 100)

    # Classical prediction (treating identical gases as distinguishable)
    S_classical = entropy_of_mixing_binary(x1_range, n_total, R)

    # Quantum/correct treatment (identical gases are indistinguishable)
    S_quantum = np.zeros_like(x1_range)

    ax4.plot(x1_range, S_classical, 'b-', lw=2,
             label='Classical (distinguishable particles)')
    ax4.plot(x1_range, S_quantum, 'r-', lw=2,
             label='Quantum (indistinguishable particles)')

    ax4.fill_between(x1_range, S_classical, S_quantum, alpha=0.3, color='purple',
                     label='Gibbs paradox resolution')

    ax4.set_xlabel('Mole Fraction of Gas A', fontsize=12)
    ax4.set_ylabel('Entropy Change (J/(mol K))', fontsize=12)
    ax4.set_title('Gibbs Paradox: Mixing Identical Gases', fontsize=12)
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 1)

    # Add explanation text
    explanation = ('The Gibbs paradox arises when classical\n'
                   'theory predicts entropy increase for\n'
                   'mixing identical gases. Quantum mechanics\n'
                   'resolves this: identical particles are\n'
                   'fundamentally indistinguishable.')
    ax4.text(0.05, 0.3, explanation, transform=ax4.transAxes, fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('Entropy of Mixing for Ideal Gases\n'
                 f'T = {T:.1f}K, n = {n_total} mol',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'entropy_of_mixing.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'entropy_of_mixing.png')}")


if __name__ == "__main__":
    main()
