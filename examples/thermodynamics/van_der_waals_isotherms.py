"""
Example demonstrating Van der Waals isotherms and the critical point.

The Van der Waals equation of state:
(P + a*n^2/V^2)(V - nb) = nRT

where:
- a: accounts for intermolecular attraction
- b: accounts for molecular volume

This example shows:
- Van der Waals isotherms at various temperatures
- Critical point identification
- Comparison with ideal gas
- Maxwell construction for phase coexistence
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from src.sciforge.core.constants import CONSTANTS


def van_der_waals_pressure(V, n, T, a, b, R=CONSTANTS['R']):
    """
    Calculate pressure using Van der Waals equation.

    (P + a*n^2/V^2)(V - nb) = nRT
    P = nRT/(V - nb) - a*n^2/V^2

    Args:
        V: Volume (m^3)
        n: Number of moles
        T: Temperature (K)
        a: Van der Waals constant a (Pa*m^6/mol^2)
        b: Van der Waals constant b (m^3/mol)
        R: Gas constant

    Returns:
        Pressure (Pa)
    """
    return n * R * T / (V - n * b) - a * n**2 / V**2


def ideal_gas_pressure(V, n, T, R=CONSTANTS['R']):
    """Calculate pressure for ideal gas: P = nRT/V"""
    return n * R * T / V


def critical_point(a, b, R=CONSTANTS['R']):
    """
    Calculate critical point for Van der Waals gas.

    Tc = 8a / (27Rb)
    Pc = a / (27b^2)
    Vc = 3nb

    Returns:
        Tuple of (Tc, Pc, Vc) for 1 mole
    """
    Tc = 8 * a / (27 * R * b)
    Pc = a / (27 * b**2)
    Vc = 3 * b  # per mole
    return Tc, Pc, Vc


def reduced_coordinates(T, P, V, Tc, Pc, Vc):
    """Convert to reduced coordinates: Tr = T/Tc, Pr = P/Pc, Vr = V/Vc"""
    return T / Tc, P / Pc, V / Vc


def main():
    # Van der Waals constants for CO2
    # a = 0.364 Pa*m^6/mol^2, b = 4.27e-5 m^3/mol
    a = 0.364
    b = 4.27e-5
    n = 1.0  # moles
    R = CONSTANTS['R']

    # Calculate critical point
    Tc, Pc, Vc = critical_point(a, b, R)
    print(f"Critical point for CO2:")
    print(f"  Tc = {Tc:.2f} K")
    print(f"  Pc = {Pc/1e6:.2f} MPa")
    print(f"  Vc = {Vc*1e6:.2f} cm^3/mol")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Van der Waals isotherms
    ax1 = axes[0, 0]

    # Volume range
    V_min = 1.1 * n * b  # Slightly above nb to avoid singularity
    V_max = 10 * Vc * n
    V = np.linspace(V_min, V_max, 500)

    # Temperature range around critical temperature
    T_ratios = [0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.20]
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(T_ratios)))

    for T_ratio, color in zip(T_ratios, colors):
        T = T_ratio * Tc
        P = van_der_waals_pressure(V, n, T, a, b, R)
        # Clip negative pressures
        P = np.maximum(P, 0)
        label = f'T/Tc = {T_ratio:.2f} ({T:.0f}K)'
        ax1.plot(V * 1e6, P / 1e6, color=color, lw=2, label=label)

    # Mark critical point
    ax1.plot(Vc * n * 1e6, Pc / 1e6, 'k*', markersize=15, label='Critical point')

    ax1.set_xlabel('Molar Volume (cm$^3$/mol)', fontsize=12)
    ax1.set_ylabel('Pressure (MPa)', fontsize=12)
    ax1.set_title('Van der Waals Isotherms', fontsize=12)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, V_max * 1e6)
    ax1.set_ylim(0, 2 * Pc / 1e6)

    # Plot 2: Comparison with ideal gas at critical temperature
    ax2 = axes[0, 1]

    T = Tc
    P_vdw = van_der_waals_pressure(V, n, T, a, b, R)
    P_ideal = ideal_gas_pressure(V, n, T, R)

    ax2.plot(V * 1e6, P_vdw / 1e6, 'b-', lw=2, label='Van der Waals')
    ax2.plot(V * 1e6, P_ideal / 1e6, 'r--', lw=2, label='Ideal Gas')
    ax2.axhline(y=Pc / 1e6, color='gray', linestyle=':', label='P_c')
    ax2.axvline(x=Vc * n * 1e6, color='gray', linestyle=':')

    ax2.set_xlabel('Molar Volume (cm$^3$/mol)', fontsize=12)
    ax2.set_ylabel('Pressure (MPa)', fontsize=12)
    ax2.set_title(f'Van der Waals vs Ideal Gas at T = Tc = {Tc:.0f}K', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, V_max * 1e6)
    ax2.set_ylim(0, 2.5 * Pc / 1e6)

    # Shade region where vdw differs significantly from ideal
    ax2.fill_between(V * 1e6, P_vdw / 1e6, P_ideal / 1e6,
                     where=P_vdw > 0, alpha=0.2, color='purple',
                     label='Non-ideal behavior')

    # Plot 3: Reduced isotherms (law of corresponding states)
    ax3 = axes[1, 0]

    # In reduced coordinates, all gases follow the same curve
    Vr = np.linspace(0.4, 6, 500)

    # Van der Waals in reduced form: Pr = 8*Tr/(3*Vr - 1) - 3/Vr^2
    Tr_values = [0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.20]
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(Tr_values)))

    for Tr, color in zip(Tr_values, colors):
        Pr = 8 * Tr / (3 * Vr - 1) - 3 / Vr**2
        Pr = np.maximum(Pr, 0)  # Clip negative
        ax3.plot(Vr, Pr, color=color, lw=2, label=f'Tr = {Tr:.2f}')

    # Mark critical point in reduced coordinates
    ax3.plot(1, 1, 'k*', markersize=15)

    ax3.set_xlabel('Reduced Volume (V/Vc)', fontsize=12)
    ax3.set_ylabel('Reduced Pressure (P/Pc)', fontsize=12)
    ax3.set_title('Law of Corresponding States\n(Reduced Van der Waals Equation)',
                  fontsize=12)
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0.3, 6)
    ax3.set_ylim(0, 2)

    # Add annotation about universality
    ax3.text(0.5, 0.15, 'All real gases approximately\nfollow the same reduced curve',
             transform=ax3.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Plot 4: Subcritical isotherm with Maxwell construction
    ax4 = axes[1, 1]

    T_sub = 0.9 * Tc
    V_fine = np.linspace(V_min, V_max, 1000)
    P_sub = van_der_waals_pressure(V_fine, n, T_sub, a, b, R)

    ax4.plot(V_fine * 1e6, P_sub / 1e6, 'b-', lw=2, label='Van der Waals isotherm')

    # Find the Maxwell equal area pressure (approximate)
    # For simplicity, we'll estimate it
    P_maxwell = 0.85 * Pc  # Approximate for T = 0.9*Tc

    ax4.axhline(y=P_maxwell / 1e6, color='r', linestyle='--', lw=2,
                label='Maxwell construction')

    # Shade unstable region
    mask = P_sub < P_maxwell
    ax4.fill_between(V_fine * 1e6, P_sub / 1e6, P_maxwell / 1e6,
                     where=(P_sub < P_maxwell) & (P_sub > 0),
                     alpha=0.3, color='red', label='Unstable region')

    ax4.set_xlabel('Molar Volume (cm$^3$/mol)', fontsize=12)
    ax4.set_ylabel('Pressure (MPa)', fontsize=12)
    ax4.set_title(f'Maxwell Construction at T = 0.9Tc = {T_sub:.0f}K', fontsize=12)
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 600)
    ax4.set_ylim(0, 1.5 * Pc / 1e6)

    # Add annotations for phases
    ax4.annotate('Liquid\nphase', xy=(100, P_maxwell/1e6), fontsize=10,
                 ha='center', va='bottom')
    ax4.annotate('Gas\nphase', xy=(400, P_maxwell/1e6), fontsize=10,
                 ha='center', va='bottom')
    ax4.annotate('Two-phase\ncoexistence', xy=(200, 0.7*P_maxwell/1e6), fontsize=9,
                 ha='center', bbox=dict(boxstyle='round', facecolor='pink', alpha=0.5))

    plt.suptitle('Van der Waals Equation of State and Critical Phenomena\n'
                 f'CO2: a = {a} Pa m$^6$/mol$^2$, b = {b*1e6:.1f} cm$^3$/mol',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'van_der_waals_isotherms.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'van_der_waals_isotherms.png')}")


if __name__ == "__main__":
    main()
