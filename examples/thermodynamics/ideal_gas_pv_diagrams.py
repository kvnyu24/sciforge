"""
Example demonstrating ideal gas PV diagrams for isothermal and adiabatic processes.

This example shows:
- Isothermal expansion/compression (constant temperature, PV = nRT)
- Adiabatic expansion/compression (no heat exchange, PV^gamma = constant)
- Comparison of work done in each process
- Temperature changes during adiabatic processes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.core.constants import CONSTANTS


def isothermal_process(V, n, T, R=CONSTANTS['R']):
    """
    Calculate pressure for isothermal process.

    PV = nRT, so P = nRT/V

    Args:
        V: Volume array (m^3)
        n: Number of moles
        T: Temperature (K)
        R: Gas constant

    Returns:
        Pressure array (Pa)
    """
    return n * R * T / V


def adiabatic_process(V, P0, V0, gamma=1.4):
    """
    Calculate pressure for adiabatic process.

    PV^gamma = constant

    Args:
        V: Volume array (m^3)
        P0: Initial pressure (Pa)
        V0: Initial volume (m^3)
        gamma: Heat capacity ratio (Cp/Cv)

    Returns:
        Pressure array (Pa)
    """
    return P0 * (V0 / V)**gamma


def work_isothermal(n, T, V1, V2, R=CONSTANTS['R']):
    """Calculate work done in isothermal process: W = nRT * ln(V2/V1)"""
    return n * R * T * np.log(V2 / V1)


def work_adiabatic(P1, V1, P2, V2, gamma=1.4):
    """Calculate work done in adiabatic process: W = (P1*V1 - P2*V2) / (gamma - 1)"""
    return (P1 * V1 - P2 * V2) / (gamma - 1)


def main():
    # Gas parameters
    n = 1.0              # moles
    R = CONSTANTS['R']   # Gas constant (J/mol-K)
    gamma = 1.4          # Heat capacity ratio for diatomic gas (air)

    # Initial conditions
    T_initial = 300.0    # K
    V1 = 0.001           # m^3 (1 liter)
    P1 = n * R * T_initial / V1  # Pa

    # Volume range for expansion
    V = np.linspace(V1, 4 * V1, 200)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: P-V Diagrams comparison
    ax1 = axes[0, 0]

    # Isothermal process at different temperatures
    temperatures = [200, 300, 400, 500]
    colors_iso = plt.cm.Reds(np.linspace(0.3, 0.9, len(temperatures)))

    for T, color in zip(temperatures, colors_iso):
        P_iso = isothermal_process(V, n, T, R)
        ax1.plot(V * 1000, P_iso / 1000, color=color, lw=2,
                label=f'Isothermal T={T}K')

    # Adiabatic processes starting from different initial states
    V0_values = [V1, 1.5*V1, 2*V1]
    colors_adi = plt.cm.Blues(np.linspace(0.4, 0.9, len(V0_values)))

    for V0, color in zip(V0_values, colors_adi):
        P0 = n * R * T_initial / V0
        V_adi = np.linspace(V0, 4*V1, 200)
        P_adi = adiabatic_process(V_adi, P0, V0, gamma)
        ax1.plot(V_adi * 1000, P_adi / 1000, '--', color=color, lw=2,
                label=f'Adiabatic from V0={V0*1000:.1f}L')

    ax1.set_xlabel('Volume (L)', fontsize=12)
    ax1.set_ylabel('Pressure (kPa)', fontsize=12)
    ax1.set_title('P-V Diagrams: Isothermal vs Adiabatic', fontsize=12)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 4.5)

    # Plot 2: Work comparison
    ax2 = axes[0, 1]

    V2_values = np.linspace(1.1 * V1, 4 * V1, 100)

    # Work for isothermal process
    W_iso = work_isothermal(n, T_initial, V1, V2_values, R)

    # Work for adiabatic process
    P2_adi = adiabatic_process(V2_values, P1, V1, gamma)
    W_adi = work_adiabatic(P1, V1, P2_adi, V2_values, gamma)

    ax2.plot(V2_values / V1, W_iso / 1000, 'r-', lw=2, label='Isothermal')
    ax2.plot(V2_values / V1, W_adi / 1000, 'b-', lw=2, label='Adiabatic')
    ax2.fill_between(V2_values / V1, W_iso / 1000, W_adi / 1000,
                     alpha=0.2, color='purple', label='Work difference')

    ax2.set_xlabel('Volume Ratio (V2/V1)', fontsize=12)
    ax2.set_ylabel('Work Done (kJ)', fontsize=12)
    ax2.set_title('Work Done During Expansion', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add annotation
    ax2.annotate('Isothermal does\nmore work\n(absorbs heat)',
                xy=(3.0, 2.5), fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 3: Temperature during adiabatic process
    ax3 = axes[1, 0]

    # For adiabatic: TV^(gamma-1) = constant
    T_adiabatic = T_initial * (V1 / V)**(gamma - 1)

    ax3.plot(V / V1, T_adiabatic, 'b-', lw=2, label='Adiabatic')
    ax3.axhline(y=T_initial, color='r', linestyle='--', lw=2, label='Isothermal')

    ax3.fill_between(V / V1, T_adiabatic, T_initial, where=T_adiabatic < T_initial,
                     alpha=0.3, color='blue', label='Cooling')

    ax3.set_xlabel('Volume Ratio (V/V1)', fontsize=12)
    ax3.set_ylabel('Temperature (K)', fontsize=12)
    ax3.set_title('Temperature Change During Expansion', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Add specific values
    final_T = T_initial * (V1 / (4*V1))**(gamma - 1)
    ax3.annotate(f'Final T = {final_T:.0f}K\n(4x expansion)',
                xy=(4, final_T), xytext=(3.2, 220),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    # Plot 4: Different gamma values
    ax4 = axes[1, 1]

    gamma_values = [1.0, 1.33, 1.4, 1.67]
    gamma_labels = ['Isothermal (gamma=1)', 'Polyatomic (1.33)',
                    'Diatomic (1.4)', 'Monatomic (1.67)']
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(gamma_values)))

    for gamma_val, label, color in zip(gamma_values, gamma_labels, colors):
        if gamma_val == 1.0:
            P = isothermal_process(V, n, T_initial, R)
        else:
            P = adiabatic_process(V, P1, V1, gamma_val)
        ax4.plot(V * 1000, P / 1000, color=color, lw=2, label=label)

    ax4.set_xlabel('Volume (L)', fontsize=12)
    ax4.set_ylabel('Pressure (kPa)', fontsize=12)
    ax4.set_title('Effect of Heat Capacity Ratio (gamma)', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Add text explaining gamma
    ax4.text(0.95, 0.95,
             'gamma = Cp/Cv\nHigher gamma = steeper curve\n(less compressible)',
             transform=ax4.transAxes, fontsize=9, verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('Ideal Gas PV Diagrams: Isothermal vs Adiabatic Processes\n'
                 f'n = {n} mol, T_initial = {T_initial}K, V1 = {V1*1000:.0f}L',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'ideal_gas_pv_diagrams.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'ideal_gas_pv_diagrams.png')}")


if __name__ == "__main__":
    main()
