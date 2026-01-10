"""
Example demonstrating Joule (free) expansion and entropy change.

In Joule expansion (free expansion into vacuum):
- No work done: W = 0 (expansion into vacuum)
- No heat transfer: Q = 0 (adiabatic process)
- Therefore: Delta U = 0 (internal energy constant)
- For ideal gas: Temperature remains constant
- For real gas: Temperature may change (Joule-Thomson effect)

Key result: Despite Q = 0, entropy INCREASES because process is irreversible.

Delta S = nR * ln(V2/V1) > 0

This example shows:
- Entropy change during free expansion
- Comparison with reversible isothermal expansion
- Ideal vs real gas behavior
- Irreversibility and entropy generation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.core.constants import CONSTANTS


def joule_expansion_entropy(V1, V2, n=1.0, R=CONSTANTS['R']):
    """
    Calculate entropy change for Joule (free) expansion of ideal gas.

    Delta S = nR * ln(V2/V1)

    Args:
        V1: Initial volume (m^3)
        V2: Final volume (m^3)
        n: Number of moles
        R: Gas constant

    Returns:
        Entropy change (J/K)
    """
    return n * R * np.log(V2 / V1)


def reversible_work(V1, V2, n, T, R=CONSTANTS['R']):
    """Work done in reversible isothermal expansion: W = nRT * ln(V2/V1)"""
    return n * R * T * np.log(V2 / V1)


def van_der_waals_temperature_change(V1, V2, T1, n, a, Cv):
    """
    Calculate temperature change for Joule expansion of Van der Waals gas.

    For VdW gas: (dU/dV)_T = a*n^2/V^2
    Since dU = 0 for free expansion: Cv*dT = -a*n^2*dV/V^2

    Integrating: Delta T = (a*n/Cv) * (1/V2 - 1/V1)
    """
    return (a * n / Cv) * (1/V2 - 1/V1)


def entropy_production_rate(V, V_final, k_rate):
    """
    Model entropy production during irreversible expansion.

    Simple exponential approach to equilibrium.
    """
    return k_rate * (V_final - V) / V_final


def main():
    R = CONSTANTS['R']
    n = 1.0           # 1 mole
    T = 300           # K

    # Initial and final volumes
    V1 = 0.001        # m^3 (1 liter)
    V2 = 0.004        # m^3 (4 liters) - 4x expansion

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Entropy change vs volume ratio
    ax1 = axes[0, 0]

    V_ratios = np.linspace(1.01, 10, 200)
    V2_array = V1 * V_ratios
    delta_S = joule_expansion_entropy(V1, V2_array, n, R)

    ax1.plot(V_ratios, delta_S, 'b-', lw=2, label='Joule expansion')

    # Mark specific points
    V_ratio_specific = V2 / V1
    delta_S_specific = joule_expansion_entropy(V1, V2, n, R)
    ax1.plot(V_ratio_specific, delta_S_specific, 'ro', markersize=10)
    ax1.annotate(f'4x expansion\nDelta S = {delta_S_specific:.2f} J/K',
                xy=(V_ratio_specific, delta_S_specific),
                xytext=(V_ratio_specific + 1, delta_S_specific - 3),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10)

    # 2x expansion
    delta_S_2x = joule_expansion_entropy(V1, 2*V1, n, R)
    ax1.plot(2, delta_S_2x, 'go', markersize=8)
    ax1.annotate(f'2x: {delta_S_2x:.2f} J/K', xy=(2, delta_S_2x),
                xytext=(2.5, delta_S_2x + 2), fontsize=9)

    ax1.set_xlabel('Volume Ratio (V2/V1)', fontsize=12)
    ax1.set_ylabel('Entropy Change (J/K)', fontsize=12)
    ax1.set_title('Entropy Increase in Joule Expansion\n'
                  r'$\Delta S = nR \ln(V_2/V_1)$', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, 10)

    # Add physics note
    ax1.text(0.05, 0.95, 'Q = 0, W = 0, but S increases!\n(Irreversible process)',
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Plot 2: Comparison with reversible process
    ax2 = axes[0, 1]

    # For Joule expansion: Q = 0, W = 0
    # For reversible isothermal: Q = W = nRT*ln(V2/V1)

    W_reversible = reversible_work(V1, V2_array, n, T, R)
    Q_reversible = W_reversible  # First law: dU = 0 for isothermal

    ax2.plot(V_ratios, W_reversible / 1000, 'b-', lw=2,
             label='Reversible: W = Q = nRT ln(V2/V1)')
    ax2.axhline(y=0, color='r', linestyle='--', lw=2,
                label='Joule expansion: W = Q = 0')

    # Shade area showing lost work opportunity
    ax2.fill_between(V_ratios, 0, W_reversible / 1000, alpha=0.3, color='blue',
                     label='Lost work potential')

    ax2.set_xlabel('Volume Ratio (V2/V1)', fontsize=12)
    ax2.set_ylabel('Work / Heat (kJ)', fontsize=12)
    ax2.set_title('Lost Work in Free Expansion\n'
                  'vs Reversible Isothermal Expansion', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, 10)

    # Mark specific value
    W_specific = reversible_work(V1, V2, n, T, R)
    ax2.annotate(f'Lost work = {W_specific/1000:.2f} kJ\nfor 4x expansion',
                xy=(4, W_specific/1000), xytext=(6, W_specific/1000 + 1),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=10)

    # Plot 3: Ideal vs Real gas (Van der Waals)
    ax3 = axes[1, 0]

    # Van der Waals parameters for different gases
    gases = {
        'Ideal': {'a': 0, 'color': 'blue'},
        'He': {'a': 0.00346, 'color': 'green'},   # Pa*m^6/mol^2
        'N2': {'a': 0.137, 'color': 'orange'},
        'CO2': {'a': 0.364, 'color': 'red'},
    }

    Cv = 3 * R / 2  # Monatomic (for simplicity)
    T1 = 300

    V1_real = 0.0001  # Start with smaller volume (more effect)
    V2_range = np.linspace(V1_real * 1.1, V1_real * 10, 100)

    for gas_name, params in gases.items():
        a = params['a']
        if a == 0:  # Ideal gas
            delta_T = np.zeros_like(V2_range)
        else:
            delta_T = van_der_waals_temperature_change(V1_real, V2_range, T1, n, a, Cv)

        ax3.plot(V2_range / V1_real, delta_T, color=params['color'], lw=2, label=gas_name)

    ax3.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

    ax3.set_xlabel('Volume Ratio (V2/V1)', fontsize=12)
    ax3.set_ylabel('Temperature Change (K)', fontsize=12)
    ax3.set_title('Joule-Thomson Effect in Real Gases\n'
                  '(Temperature change during free expansion)', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Add explanation
    ax3.text(0.6, 0.3, 'Real gases cool during\nfree expansion due to\n'
             'intermolecular attractions',
             transform=ax3.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Plot 4: P-V diagram showing irreversibility
    ax4 = axes[1, 1]

    # Initial state
    P1 = n * R * T / V1

    # Final state (after equilibrium)
    P2 = n * R * T / V2

    # Plot states
    ax4.plot(V1 * 1000, P1 / 1000, 'bo', markersize=15, label='Initial state')
    ax4.plot(V2 * 1000, P2 / 1000, 'ro', markersize=15, label='Final state')

    # Reversible isothermal path
    V_path = np.linspace(V1, V2, 100)
    P_path = n * R * T / V_path
    ax4.plot(V_path * 1000, P_path / 1000, 'g-', lw=2, label='Reversible path')

    # Shade area under reversible curve (work)
    ax4.fill_between(V_path * 1000, 0, P_path / 1000, alpha=0.2, color='green')

    # Arrow showing irreversible jump
    ax4.annotate('', xy=(V2 * 1000, P2 / 1000), xytext=(V1 * 1000, P1 / 1000),
                arrowprops=dict(arrowstyle='->', color='red', lw=2,
                               connectionstyle='arc3,rad=-0.3'))
    ax4.text((V1 + V2) / 2 * 1000, (P1 + P2) / 2 / 1000 + 200,
             'Free expansion\n(irreversible)', fontsize=10, ha='center', color='red')

    ax4.set_xlabel('Volume (L)', fontsize=12)
    ax4.set_ylabel('Pressure (kPa)', fontsize=12)
    ax4.set_title('P-V Diagram: Free Expansion vs Reversible Path', fontsize=12)
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)

    # Annotations for states
    ax4.annotate('1', (V1 * 1000, P1 / 1000), xytext=(-15, 10),
                textcoords='offset points', fontsize=14, fontweight='bold')
    ax4.annotate('2', (V2 * 1000, P2 / 1000), xytext=(10, 0),
                textcoords='offset points', fontsize=14, fontweight='bold')

    plt.suptitle('Joule Expansion and Entropy Production\n'
                 f'n = {n} mol, T = {T}K, V1 = {V1*1000:.1f}L, V2 = {V2*1000:.1f}L',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'joule_expansion.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'joule_expansion.png')}")


if __name__ == "__main__":
    main()
