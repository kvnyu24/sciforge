"""
Example demonstrating the Clapeyron equation for phase boundaries.

The Clapeyron equation describes the slope of phase boundaries in P-T space:

dP/dT = Delta_S / Delta_V = L / (T * Delta_V)

where:
- Delta_S = entropy change during phase transition
- Delta_V = volume change during phase transition
- L = latent heat of transition

For liquid-vapor transitions (Clausius-Clapeyron):
dP/dT = P * L / (R * T^2)   (approximating vapor as ideal gas)

This example shows:
- Phase diagram with Clapeyron-derived boundaries
- Comparison with experimental data for water
- Solid-liquid-vapor triple point
- Critical point and supercritical region
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from src.sciforge.core.constants import CONSTANTS


def clausius_clapeyron_vapor_pressure(T, T_ref, P_ref, L, R=CONSTANTS['R']):
    """
    Calculate vapor pressure using integrated Clausius-Clapeyron equation.

    ln(P/P_ref) = (L/R) * (1/T_ref - 1/T)

    Args:
        T: Temperature (K)
        T_ref: Reference temperature (K)
        P_ref: Vapor pressure at reference temperature (Pa)
        L: Latent heat of vaporization (J/mol)
        R: Gas constant

    Returns:
        Vapor pressure (Pa)
    """
    return P_ref * np.exp((L / R) * (1/T_ref - 1/T))


def clapeyron_solid_liquid(T, T_ref, P_ref, L, delta_V):
    """
    Calculate solid-liquid boundary using Clapeyron equation.

    P = P_ref + (L / delta_V) * ln(T / T_ref)

    For approximately constant delta_V and L.

    Args:
        T: Temperature (K)
        T_ref: Reference temperature (K)
        P_ref: Pressure at reference point (Pa)
        L: Latent heat of fusion (J/mol)
        delta_V: Molar volume change (m^3/mol)

    Returns:
        Pressure (Pa)
    """
    # For solid-liquid, more accurate to use:
    # dP/dT = L / (T * delta_V)
    # P - P_ref = (L / delta_V) * ln(T / T_ref)
    return P_ref + (L / delta_V) * np.log(T / T_ref)


def antoine_equation(T, A, B, C):
    """
    Calculate vapor pressure using Antoine equation (empirical).

    log10(P) = A - B / (C + T)

    Args:
        T: Temperature (Celsius for standard coefficients)
        A, B, C: Antoine coefficients

    Returns:
        Pressure (mmHg for standard coefficients)
    """
    return 10**(A - B / (C + T))


def main():
    R = CONSTANTS['R']

    # Water properties
    T_triple = 273.16      # K
    P_triple = 611.657     # Pa
    T_critical = 647.096   # K
    P_critical = 22.064e6  # Pa
    T_boiling = 373.15     # K at 1 atm
    P_atm = 101325         # Pa

    # Latent heats (J/mol)
    L_vaporization = 40650    # At boiling point
    L_fusion = 6010           # At melting point
    L_sublimation = L_fusion + L_vaporization

    # Molar volume changes
    M_water = 0.018          # kg/mol
    rho_ice = 917            # kg/m^3
    rho_water = 1000         # kg/m^3
    V_ice = M_water / rho_ice
    V_water = M_water / rho_water
    delta_V_fusion = V_water - V_ice  # Negative for water!

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Complete phase diagram for water
    ax1 = axes[0, 0]

    # Solid-liquid boundary (ice-water)
    T_sl = np.linspace(250, T_triple, 100)
    P_sl = clapeyron_solid_liquid(T_sl, T_triple, P_triple, L_fusion, delta_V_fusion)

    # Liquid-vapor boundary (water-steam)
    T_lv = np.linspace(T_triple, T_critical * 0.99, 200)
    # Use temperature-dependent latent heat
    L_lv = L_vaporization * (1 - (T_lv / T_critical))**0.38  # Empirical correction
    P_lv = clausius_clapeyron_vapor_pressure(T_lv, T_boiling, P_atm, L_vaporization, R)

    # Solid-vapor boundary (sublimation)
    T_sv = np.linspace(200, T_triple, 100)
    P_sv = clausius_clapeyron_vapor_pressure(T_sv, T_triple, P_triple, L_sublimation, R)

    # Plot boundaries
    ax1.semilogy(T_sl - 273.15, P_sl / 1e6, 'b-', lw=2, label='Ice-Water (solid-liquid)')
    ax1.semilogy(T_lv - 273.15, P_lv / 1e6, 'r-', lw=2, label='Water-Steam (liquid-vapor)')
    ax1.semilogy(T_sv - 273.15, P_sv / 1e6, 'g-', lw=2, label='Ice-Steam (sublimation)')

    # Mark special points
    ax1.plot(T_triple - 273.15, P_triple / 1e6, 'ko', markersize=10, label='Triple point')
    ax1.plot(T_critical - 273.15, P_critical / 1e6, 'r*', markersize=15, label='Critical point')
    ax1.plot(100, P_atm / 1e6, 'b^', markersize=10, label='Boiling point (1 atm)')

    # Region labels
    ax1.text(-50, 1e-4, 'ICE', fontsize=12, fontweight='bold')
    ax1.text(50, 0.01, 'WATER', fontsize=12, fontweight='bold')
    ax1.text(200, 0.001, 'STEAM', fontsize=12, fontweight='bold')
    ax1.text(400, 50, 'Supercritical\nFluid', fontsize=10, ha='center')

    ax1.set_xlabel('Temperature (C)', fontsize=12)
    ax1.set_ylabel('Pressure (MPa)', fontsize=12)
    ax1.set_title('Phase Diagram of Water', fontsize=12)
    ax1.legend(loc='lower right', fontsize=8)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_xlim(-100, 450)
    ax1.set_ylim(1e-6, 100)

    # Plot 2: Vapor pressure curve detail
    ax2 = axes[0, 1]

    T_range = np.linspace(273.15, 373.15, 100)
    P_vapor = clausius_clapeyron_vapor_pressure(T_range, T_boiling, P_atm, L_vaporization, R)

    # Antoine equation for comparison (empirical data)
    # Coefficients for water (valid 1-100 C)
    T_celsius = T_range - 273.15
    A, B, C = 8.07131, 1730.63, 233.426
    P_antoine = antoine_equation(T_celsius, A, B, C) * 133.322  # mmHg to Pa

    ax2.plot(T_celsius, P_vapor / 1000, 'b-', lw=2, label='Clausius-Clapeyron')
    ax2.plot(T_celsius, P_antoine / 1000, 'r--', lw=2, label='Antoine (empirical)')
    ax2.axhline(y=P_atm / 1000, color='gray', linestyle=':', label='1 atm')

    ax2.set_xlabel('Temperature (C)', fontsize=12)
    ax2.set_ylabel('Vapor Pressure (kPa)', fontsize=12)
    ax2.set_title('Water Vapor Pressure: Theory vs Empirical', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Mark boiling point
    ax2.plot(100, P_atm / 1000, 'go', markersize=10)
    ax2.annotate('Boiling point', xy=(100, P_atm / 1000), xytext=(60, 80),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10)

    # Plot 3: Clapeyron slope comparison
    ax3 = axes[1, 0]

    # Calculate dP/dT for different transitions
    T_plot = np.linspace(250, 380, 100)

    # Liquid-vapor: dP/dT = P*L/(R*T^2)
    P_lv_temp = clausius_clapeyron_vapor_pressure(T_plot, T_boiling, P_atm, L_vaporization, R)
    dPdT_lv = P_lv_temp * L_vaporization / (R * T_plot**2)

    # Solid-liquid: dP/dT = L/(T*delta_V)
    dPdT_sl = L_fusion / (T_plot * delta_V_fusion)

    ax3.semilogy(T_plot - 273.15, np.abs(dPdT_lv), 'r-', lw=2, label='Liquid-Vapor')
    ax3.semilogy(T_plot - 273.15, np.abs(dPdT_sl), 'b-', lw=2, label='Solid-Liquid (|dP/dT|)')

    ax3.set_xlabel('Temperature (C)', fontsize=12)
    ax3.set_ylabel('|dP/dT| (Pa/K)', fontsize=12)
    ax3.set_title('Clapeyron Slope: dP/dT = L/(T*Delta_V)', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3, which='both')

    # Add annotation about water anomaly
    ax3.text(0.05, 0.95, 'Water: Solid-liquid slope is NEGATIVE\n'
             '(Ice less dense than water)\n'
             'Ice melts under pressure!',
             transform=ax3.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Plot 4: Effect of latent heat on vapor pressure
    ax4 = axes[1, 1]

    T_ref = 373.15
    P_ref = P_atm
    T_range = np.linspace(300, 450, 100)

    L_values = [30000, 40000, 50000, 60000]  # J/mol
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(L_values)))

    for L, color in zip(L_values, colors):
        P = clausius_clapeyron_vapor_pressure(T_range, T_ref, P_ref, L, R)
        ax4.semilogy(T_range - 273.15, P / 1000, color=color, lw=2,
                    label=f'L = {L/1000:.0f} kJ/mol')

    ax4.axhline(y=P_atm / 1000, color='gray', linestyle=':')

    ax4.set_xlabel('Temperature (C)', fontsize=12)
    ax4.set_ylabel('Vapor Pressure (kPa)', fontsize=12)
    ax4.set_title('Effect of Latent Heat on Vapor Pressure Curve', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3, which='both')

    ax4.text(0.05, 0.05, 'Higher latent heat = steeper curve\n(stronger intermolecular forces)',
             transform=ax4.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle('Clapeyron Equation and Phase Boundaries\n'
                 r'$\frac{dP}{dT} = \frac{\Delta S}{\Delta V} = \frac{L}{T \Delta V}$',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'clapeyron_equation.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'clapeyron_equation.png')}")


if __name__ == "__main__":
    main()
