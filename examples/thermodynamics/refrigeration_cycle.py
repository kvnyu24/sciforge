"""
Experiment 124: Vapor-Compression Refrigeration Cycle

This example demonstrates the vapor-compression refrigeration cycle, the most common
type of refrigeration system used in air conditioners, refrigerators, and heat pumps.

Physics Background:
==================
The vapor-compression cycle operates by using the phase change of a refrigerant to
transfer heat from a cold reservoir (evaporator) to a hot reservoir (condenser).

Four Stages:
1. Compression (1->2): Isentropic compression of low-pressure vapor to high-pressure
   superheated vapor. Work is done on the refrigerant by the compressor.
   - Ideal: Isentropic (constant entropy)
   - Real: Polytropic with losses

2. Condensation (2->3): Isobaric heat rejection. The superheated vapor releases heat
   to the surroundings, condenses to saturated liquid.
   - Heat rejected: Q_H = h_2 - h_3

3. Expansion (3->4): Isenthalpic throttling through an expansion valve. The high-pressure
   liquid expands to low pressure, becoming a two-phase mixture.
   - Ideal: Isenthalpic (h_3 = h_4)
   - This is an irreversible process even in ideal cycle

4. Evaporation (4->1): Isobaric heat absorption. The two-phase mixture absorbs heat
   from the cold space, evaporating to saturated vapor.
   - Heat absorbed: Q_L = h_1 - h_4

Coefficient of Performance (COP):
================================
For a refrigerator: COP_R = Q_L / W = Q_L / (Q_H - Q_L)
For a heat pump: COP_HP = Q_H / W = Q_H / (Q_H - Q_L)

Carnot COP (ideal upper limit):
COP_R,Carnot = T_L / (T_H - T_L)
COP_HP,Carnot = T_H / (T_H - T_L)

Refrigerant Properties:
======================
This example uses an ideal gas approximation with properties inspired by R-134a
(tetrafluoroethane), a common environmentally-friendly refrigerant:
- Critical temperature: ~374 K (101 C)
- Critical pressure: ~4.06 MPa
- Normal boiling point: ~247 K (-26 C)

Real vs Ideal Cycle:
===================
- Ideal: Isentropic compression, no pressure drops, perfect heat exchange
- Real: Polytropic compression, pressure drops in heat exchangers, subcooling/superheating

This example shows:
- T-s (Temperature-Entropy) diagram of the cycle
- P-h (Pressure-Enthalpy) diagram (log scale)
- COP comparison: ideal vs real vs Carnot limit
- Effect of evaporator/condenser temperatures on performance
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


# Refrigerant properties (simplified model inspired by R-134a)
class RefrigerantR134a:
    """
    Simplified thermodynamic properties for R-134a refrigerant.

    Uses ideal gas approximation with phase change properties calibrated
    to approximate R-134a behavior in typical operating ranges.
    """

    # Critical properties
    T_crit = 374.21  # K
    P_crit = 4059.28  # kPa

    # Specific heat capacities (approximate)
    cp_vapor = 1.01  # kJ/(kg·K) - vapor specific heat
    cp_liquid = 1.42  # kJ/(kg·K) - liquid specific heat

    # Latent heat of vaporization (varies with temperature)
    h_fg_ref = 198.6  # kJ/kg at T_ref
    T_ref = 273.15  # K (reference temperature)

    # Gas constant
    R = 0.08149  # kJ/(kg·K)

    # Polytropic exponent for real compression
    n_poly = 1.15  # Polytropic index (1 < n < gamma)
    gamma = 1.13  # Heat capacity ratio for R-134a vapor

    @classmethod
    def saturation_pressure(cls, T):
        """
        Calculate saturation pressure at temperature T using Antoine equation.

        Args:
            T: Temperature in K

        Returns:
            Saturation pressure in kPa
        """
        # Antoine equation parameters (fitted to R-134a data)
        A, B, C = 14.396, 2354.5, -34.0
        return np.exp(A - B / (T + C))

    @classmethod
    def saturation_temperature(cls, P):
        """
        Calculate saturation temperature at pressure P (inverse Antoine).

        Args:
            P: Pressure in kPa

        Returns:
            Saturation temperature in K
        """
        A, B, C = 14.396, 2354.5, -34.0
        return B / (A - np.log(P)) - C

    @classmethod
    def latent_heat(cls, T):
        """
        Calculate latent heat of vaporization at temperature T.

        Uses Watson correlation for temperature dependence.

        Args:
            T: Temperature in K

        Returns:
            Latent heat in kJ/kg
        """
        # Watson correlation
        T_r = T / cls.T_crit
        T_r_ref = cls.T_ref / cls.T_crit
        return cls.h_fg_ref * ((1 - T_r) / (1 - T_r_ref))**0.38

    @classmethod
    def entropy_vapor(cls, T, P, s_ref=1.0):
        """
        Calculate specific entropy of vapor.

        Args:
            T: Temperature in K
            P: Pressure in kPa
            s_ref: Reference entropy (arbitrary)

        Returns:
            Specific entropy in kJ/(kg·K)
        """
        T_sat = cls.saturation_temperature(P)
        P_sat_ref = cls.saturation_pressure(cls.T_ref)

        # Entropy of saturated vapor at reference
        s_g_ref = s_ref

        # Add contributions from temperature and pressure changes
        s = s_g_ref + cls.cp_vapor * np.log(T / cls.T_ref) - cls.R * np.log(P / P_sat_ref)
        return s

    @classmethod
    def enthalpy_vapor(cls, T, T_sat):
        """
        Calculate specific enthalpy of superheated vapor.

        Args:
            T: Actual temperature in K
            T_sat: Saturation temperature in K

        Returns:
            Specific enthalpy in kJ/kg (relative to saturated liquid at T_ref)
        """
        # Enthalpy of saturated vapor at T_sat
        h_g = cls.cp_liquid * (T_sat - cls.T_ref) + cls.latent_heat(T_sat)

        # Add superheat contribution
        h = h_g + cls.cp_vapor * (T - T_sat)
        return h

    @classmethod
    def enthalpy_liquid(cls, T):
        """
        Calculate specific enthalpy of saturated liquid.

        Args:
            T: Temperature in K

        Returns:
            Specific enthalpy in kJ/kg (relative to saturated liquid at T_ref)
        """
        return cls.cp_liquid * (T - cls.T_ref)


def ideal_refrigeration_cycle(T_evap, T_cond, superheat=5.0):
    """
    Calculate state points for ideal vapor-compression refrigeration cycle.

    Args:
        T_evap: Evaporator temperature in K
        T_cond: Condenser temperature in K
        superheat: Superheat at compressor inlet in K

    Returns:
        Dictionary with state points and performance metrics
    """
    ref = RefrigerantR134a

    # State 1: Compressor inlet (superheated vapor at low pressure)
    P_low = ref.saturation_pressure(T_evap)
    T1 = T_evap + superheat
    h1 = ref.enthalpy_vapor(T1, T_evap)
    s1 = ref.entropy_vapor(T1, P_low)

    # State 2: Compressor outlet (superheated vapor at high pressure)
    # Isentropic compression: s2 = s1
    P_high = ref.saturation_pressure(T_cond)
    s2 = s1  # Isentropic

    # Calculate T2 from isentropic compression
    # For ideal gas: T2/T1 = (P2/P1)^((gamma-1)/gamma)
    T2 = T1 * (P_high / P_low)**((ref.gamma - 1) / ref.gamma)
    h2 = ref.enthalpy_vapor(T2, T_cond)

    # State 3: Condenser outlet (saturated liquid at high pressure)
    T3 = T_cond
    h3 = ref.enthalpy_liquid(T3)
    s3 = ref.entropy_vapor(T3, P_high) - ref.latent_heat(T3) / T3

    # State 4: Expansion valve outlet (two-phase mixture at low pressure)
    # Isenthalpic throttling: h4 = h3
    T4 = T_evap
    h4 = h3  # Isenthalpic

    # Quality at state 4
    h_f4 = ref.enthalpy_liquid(T4)
    h_fg4 = ref.latent_heat(T4)
    x4 = (h4 - h_f4) / h_fg4

    s4 = s3 + ref.R * np.log(P_high / P_low)  # Throttling entropy increase

    # Performance calculations
    Q_L = h1 - h4  # Heat absorbed in evaporator
    Q_H = h2 - h3  # Heat rejected in condenser
    W_comp = h2 - h1  # Compressor work

    COP_R = Q_L / W_comp  # Refrigerator COP
    COP_HP = Q_H / W_comp  # Heat pump COP

    # Carnot COP for comparison
    COP_R_carnot = T_evap / (T_cond - T_evap)
    COP_HP_carnot = T_cond / (T_cond - T_evap)

    return {
        'states': {
            1: {'T': T1, 'P': P_low, 'h': h1, 's': s1, 'phase': 'superheated vapor'},
            2: {'T': T2, 'P': P_high, 'h': h2, 's': s2, 'phase': 'superheated vapor'},
            3: {'T': T3, 'P': P_high, 'h': h3, 's': s3, 'phase': 'saturated liquid'},
            4: {'T': T4, 'P': P_low, 'h': h4, 's': s4, 'phase': f'two-phase (x={x4:.2f})'}
        },
        'P_low': P_low,
        'P_high': P_high,
        'Q_L': Q_L,
        'Q_H': Q_H,
        'W_comp': W_comp,
        'COP_R': COP_R,
        'COP_HP': COP_HP,
        'COP_R_carnot': COP_R_carnot,
        'COP_HP_carnot': COP_HP_carnot,
        'quality_4': x4
    }


def real_refrigeration_cycle(T_evap, T_cond, superheat=5.0, eta_isen=0.75,
                              dP_evap=10.0, dP_cond=15.0, subcool=5.0):
    """
    Calculate state points for real vapor-compression refrigeration cycle.

    Includes irreversibilities:
    - Non-isentropic compression (isentropic efficiency < 1)
    - Pressure drops in heat exchangers
    - Subcooling in condenser

    Args:
        T_evap: Evaporator temperature in K
        T_cond: Condenser temperature in K
        superheat: Superheat at compressor inlet in K
        eta_isen: Isentropic efficiency of compressor
        dP_evap: Pressure drop in evaporator in kPa
        dP_cond: Pressure drop in condenser in kPa
        subcool: Subcooling in condenser in K

    Returns:
        Dictionary with state points and performance metrics
    """
    ref = RefrigerantR134a

    # State 1: Compressor inlet
    P_evap = ref.saturation_pressure(T_evap)
    P1 = P_evap - dP_evap  # Account for pressure drop
    T1 = T_evap + superheat
    h1 = ref.enthalpy_vapor(T1, T_evap)
    s1 = ref.entropy_vapor(T1, P1)

    # State 2s: Ideal (isentropic) compressor outlet
    P_cond = ref.saturation_pressure(T_cond)
    P2 = P_cond + dP_cond  # Need higher pressure to overcome drop
    T2s = T1 * (P2 / P1)**((ref.gamma - 1) / ref.gamma)
    h2s = ref.enthalpy_vapor(T2s, T_cond)

    # State 2: Actual compressor outlet (with isentropic efficiency)
    h2 = h1 + (h2s - h1) / eta_isen
    # Estimate T2 from enthalpy
    T2 = T_cond + (h2 - ref.enthalpy_vapor(T_cond, T_cond)) / ref.cp_vapor
    s2 = ref.entropy_vapor(T2, P2)

    # State 3: Condenser outlet (subcooled liquid)
    T3 = T_cond - subcool
    h3 = ref.enthalpy_liquid(T3)
    s3 = ref.entropy_vapor(T_cond, P_cond) - ref.latent_heat(T_cond) / T_cond
    s3 -= ref.cp_liquid * np.log(T_cond / T3)  # Subcooling entropy reduction

    # State 4: Expansion valve outlet
    T4 = T_evap
    h4 = h3  # Isenthalpic

    # Quality at state 4
    h_f4 = ref.enthalpy_liquid(T4)
    h_fg4 = ref.latent_heat(T4)
    x4 = (h4 - h_f4) / h_fg4

    s4 = s3 + ref.R * np.log(P2 / P1)

    # Performance calculations
    Q_L = h1 - h4
    Q_H = h2 - h3
    W_comp = h2 - h1

    COP_R = Q_L / W_comp
    COP_HP = Q_H / W_comp

    # Carnot COP for comparison
    COP_R_carnot = T_evap / (T_cond - T_evap)
    COP_HP_carnot = T_cond / (T_cond - T_evap)

    return {
        'states': {
            1: {'T': T1, 'P': P1, 'h': h1, 's': s1, 'phase': 'superheated vapor'},
            2: {'T': T2, 'P': P2, 'h': h2, 's': s2, 'phase': 'superheated vapor'},
            3: {'T': T3, 'P': P_cond, 'h': h3, 's': s3, 'phase': 'subcooled liquid'},
            4: {'T': T4, 'P': P_evap, 'h': h4, 's': s4, 'phase': f'two-phase (x={x4:.2f})'}
        },
        'P_low': P_evap,
        'P_high': P_cond,
        'Q_L': Q_L,
        'Q_H': Q_H,
        'W_comp': W_comp,
        'COP_R': COP_R,
        'COP_HP': COP_HP,
        'COP_R_carnot': COP_R_carnot,
        'COP_HP_carnot': COP_HP_carnot,
        'quality_4': x4,
        'eta_isentropic': eta_isen
    }


def generate_saturation_dome(T_min=220, T_max=370, n_points=100):
    """
    Generate saturation dome data for T-s and P-h diagrams.

    Args:
        T_min: Minimum temperature in K
        T_max: Maximum temperature in K
        n_points: Number of points

    Returns:
        Dictionary with saturation curves
    """
    ref = RefrigerantR134a

    T = np.linspace(T_min, min(T_max, ref.T_crit - 1), n_points)
    P = np.array([ref.saturation_pressure(t) for t in T])

    # Saturated liquid line
    h_f = np.array([ref.enthalpy_liquid(t) for t in T])
    s_f = np.array([ref.entropy_vapor(t, ref.saturation_pressure(t)) -
                    ref.latent_heat(t) / t for t in T])

    # Saturated vapor line
    h_g = np.array([ref.enthalpy_liquid(t) + ref.latent_heat(t) for t in T])
    s_g = np.array([ref.entropy_vapor(t, ref.saturation_pressure(t)) for t in T])

    return {
        'T': T,
        'P': P,
        'h_f': h_f,
        'h_g': h_g,
        's_f': s_f,
        's_g': s_g
    }


def main():
    """Run refrigeration cycle simulation and generate visualizations."""

    # Operating conditions
    T_evap = 263.15  # K (-10 C) - typical refrigerator evaporator
    T_cond = 313.15  # K (40 C) - typical condenser temperature
    superheat = 5.0  # K

    # Calculate cycles
    ideal = ideal_refrigeration_cycle(T_evap, T_cond, superheat)
    real = real_refrigeration_cycle(T_evap, T_cond, superheat, eta_isen=0.75)

    # Generate saturation dome
    dome = generate_saturation_dome()

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: T-s Diagram
    ax1 = axes[0, 0]

    # Plot saturation dome
    ax1.plot(dome['s_f'], dome['T'] - 273.15, 'k-', lw=1.5, label='Saturation dome')
    ax1.plot(dome['s_g'], dome['T'] - 273.15, 'k-', lw=1.5)
    ax1.fill_betweenx(dome['T'] - 273.15, dome['s_f'], dome['s_g'],
                       alpha=0.1, color='blue')

    # Plot ideal cycle
    s_ideal = [ideal['states'][i]['s'] for i in [1, 2, 3, 4, 1]]
    T_ideal = [ideal['states'][i]['T'] - 273.15 for i in [1, 2, 3, 4, 1]]
    ax1.plot(s_ideal, T_ideal, 'b-o', lw=2, markersize=8, label='Ideal cycle')

    # Plot real cycle
    s_real = [real['states'][i]['s'] for i in [1, 2, 3, 4, 1]]
    T_real = [real['states'][i]['T'] - 273.15 for i in [1, 2, 3, 4, 1]]
    ax1.plot(s_real, T_real, 'r--s', lw=2, markersize=8, label='Real cycle')

    # Label state points
    for i in [1, 2, 3, 4]:
        ax1.annotate(f'{i}', (ideal['states'][i]['s'], ideal['states'][i]['T'] - 273.15),
                    xytext=(5, 5), textcoords='offset points', fontsize=11, fontweight='bold')

    # Add process labels
    ax1.annotate('Compression', xy=(np.mean([s_ideal[0], s_ideal[1]]),
                np.mean([T_ideal[0], T_ideal[1]])), fontsize=9, color='blue')
    ax1.annotate('Condensation', xy=(np.mean([s_ideal[1], s_ideal[2]]),
                np.mean([T_ideal[1], T_ideal[2]]) + 5), fontsize=9, color='blue')
    ax1.annotate('Throttling', xy=(np.mean([s_ideal[2], s_ideal[3]]) + 0.02,
                np.mean([T_ideal[2], T_ideal[3]])), fontsize=9, color='blue')
    ax1.annotate('Evaporation', xy=(np.mean([s_ideal[3], s_ideal[0]]),
                T_ideal[3] - 8), fontsize=9, color='blue')

    ax1.set_xlabel('Specific Entropy s (kJ/kg-K)', fontsize=12)
    ax1.set_ylabel('Temperature (C)', fontsize=12)
    ax1.set_title('T-s Diagram: Vapor-Compression Refrigeration Cycle', fontsize=12)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.6, 1.4)
    ax1.set_ylim(-40, 100)

    # Plot 2: P-h Diagram (log scale)
    ax2 = axes[0, 1]

    # Plot saturation dome
    ax2.semilogy(dome['h_f'], dome['P'], 'k-', lw=1.5, label='Saturation dome')
    ax2.semilogy(dome['h_g'], dome['P'], 'k-', lw=1.5)
    ax2.fill_betweenx(dome['P'], dome['h_f'], dome['h_g'], alpha=0.1, color='blue')

    # Plot ideal cycle
    h_ideal = [ideal['states'][i]['h'] for i in [1, 2, 3, 4, 1]]
    P_ideal = [ideal['states'][i]['P'] for i in [1, 2, 3, 4, 1]]
    ax2.semilogy(h_ideal, P_ideal, 'b-o', lw=2, markersize=8, label='Ideal cycle')

    # Plot real cycle
    h_real = [real['states'][i]['h'] for i in [1, 2, 3, 4, 1]]
    P_real = [real['states'][i]['P'] for i in [1, 2, 3, 4, 1]]
    ax2.semilogy(h_real, P_real, 'r--s', lw=2, markersize=8, label='Real cycle')

    # Label state points
    for i in [1, 2, 3, 4]:
        ax2.annotate(f'{i}', (ideal['states'][i]['h'], ideal['states'][i]['P']),
                    xytext=(5, 5), textcoords='offset points', fontsize=11, fontweight='bold')

    ax2.set_xlabel('Specific Enthalpy h (kJ/kg)', fontsize=12)
    ax2.set_ylabel('Pressure (kPa)', fontsize=12)
    ax2.set_title('P-h Diagram (Log Scale)', fontsize=12)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_xlim(-50, 350)
    ax2.set_ylim(50, 5000)

    # Plot 3: COP vs Temperature
    ax3 = axes[1, 0]

    # Vary evaporator temperature
    T_evap_range = np.linspace(243, 283, 50)  # -30 to 10 C
    COP_ideal_evap = []
    COP_real_evap = []
    COP_carnot_evap = []

    for T_e in T_evap_range:
        try:
            ideal_temp = ideal_refrigeration_cycle(T_e, T_cond, superheat)
            real_temp = real_refrigeration_cycle(T_e, T_cond, superheat, eta_isen=0.75)
            COP_ideal_evap.append(ideal_temp['COP_R'])
            COP_real_evap.append(real_temp['COP_R'])
            COP_carnot_evap.append(ideal_temp['COP_R_carnot'])
        except:
            COP_ideal_evap.append(np.nan)
            COP_real_evap.append(np.nan)
            COP_carnot_evap.append(np.nan)

    ax3.plot(T_evap_range - 273.15, COP_carnot_evap, 'g-', lw=2, label='Carnot limit')
    ax3.plot(T_evap_range - 273.15, COP_ideal_evap, 'b-', lw=2, label='Ideal cycle')
    ax3.plot(T_evap_range - 273.15, COP_real_evap, 'r--', lw=2,
             label=f'Real cycle (eta_isen={real["eta_isentropic"]})')

    # Mark operating point
    ax3.plot(T_evap - 273.15, ideal['COP_R'], 'bo', markersize=10)
    ax3.plot(T_evap - 273.15, real['COP_R'], 'rs', markersize=10)
    ax3.annotate(f'Ideal: COP={ideal["COP_R"]:.2f}',
                xy=(T_evap - 273.15, ideal['COP_R']),
                xytext=(10, 10), textcoords='offset points', fontsize=9)
    ax3.annotate(f'Real: COP={real["COP_R"]:.2f}',
                xy=(T_evap - 273.15, real['COP_R']),
                xytext=(10, -15), textcoords='offset points', fontsize=9)

    ax3.set_xlabel('Evaporator Temperature (C)', fontsize=12)
    ax3.set_ylabel('COP (Refrigeration)', fontsize=12)
    ax3.set_title(f'COP vs Evaporator Temperature\n(T_cond = {T_cond - 273.15:.0f} C)', fontsize=12)
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-30, 10)
    ax3.set_ylim(0, 15)

    # Plot 4: Energy flow and comparison bar chart
    ax4 = axes[1, 1]

    # Bar chart comparing ideal vs real cycle
    x = np.arange(4)
    width = 0.35

    metrics_ideal = [ideal['Q_L'], ideal['W_comp'], ideal['Q_H'], ideal['COP_R']]
    metrics_real = [real['Q_L'], real['W_comp'], real['Q_H'], real['COP_R']]
    labels = ['Q_L\n(kJ/kg)', 'W_comp\n(kJ/kg)', 'Q_H\n(kJ/kg)', 'COP']

    bars1 = ax4.bar(x - width/2, metrics_ideal, width, label='Ideal cycle', color='blue', alpha=0.7)
    bars2 = ax4.bar(x + width/2, metrics_real, width, label='Real cycle', color='red', alpha=0.7)

    # Add value labels
    for bar, val in zip(bars1, metrics_ideal):
        height = bar.get_height()
        ax4.annotate(f'{val:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9)

    for bar, val in zip(bars2, metrics_real):
        height = bar.get_height()
        ax4.annotate(f'{val:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9)

    ax4.set_ylabel('Value', fontsize=12)
    ax4.set_title('Cycle Performance Comparison', fontsize=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, fontsize=10)
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')

    # Add efficiency annotation
    efficiency_text = (
        f'Ideal vs Real Comparison:\n'
        f'Carnot COP: {ideal["COP_R_carnot"]:.2f}\n'
        f'Ideal COP: {ideal["COP_R"]:.2f} ({ideal["COP_R"]/ideal["COP_R_carnot"]*100:.0f}% of Carnot)\n'
        f'Real COP: {real["COP_R"]:.2f} ({real["COP_R"]/ideal["COP_R_carnot"]*100:.0f}% of Carnot)'
    )
    ax4.text(0.02, 0.98, efficiency_text, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Main title
    plt.suptitle(f'Vapor-Compression Refrigeration Cycle Analysis\n'
                 f'T_evap = {T_evap - 273.15:.0f} C, T_cond = {T_cond - 273.15:.0f} C '
                 f'(R-134a approximation)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'refrigeration_cycle.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

    # Print summary
    print("\n" + "="*60)
    print("REFRIGERATION CYCLE ANALYSIS SUMMARY")
    print("="*60)
    print(f"\nOperating Conditions:")
    print(f"  Evaporator temperature: {T_evap - 273.15:.1f} C ({T_evap:.1f} K)")
    print(f"  Condenser temperature:  {T_cond - 273.15:.1f} C ({T_cond:.1f} K)")
    print(f"  Superheat: {superheat:.1f} K")

    print(f"\nIdeal Cycle Performance:")
    print(f"  Heat absorbed (Q_L):    {ideal['Q_L']:.1f} kJ/kg")
    print(f"  Compressor work (W):    {ideal['W_comp']:.1f} kJ/kg")
    print(f"  Heat rejected (Q_H):    {ideal['Q_H']:.1f} kJ/kg")
    print(f"  COP (refrigeration):    {ideal['COP_R']:.2f}")
    print(f"  COP (heat pump):        {ideal['COP_HP']:.2f}")

    print(f"\nReal Cycle Performance (eta_isen = {real['eta_isentropic']}):")
    print(f"  Heat absorbed (Q_L):    {real['Q_L']:.1f} kJ/kg")
    print(f"  Compressor work (W):    {real['W_comp']:.1f} kJ/kg")
    print(f"  Heat rejected (Q_H):    {real['Q_H']:.1f} kJ/kg")
    print(f"  COP (refrigeration):    {real['COP_R']:.2f}")
    print(f"  COP (heat pump):        {real['COP_HP']:.2f}")

    print(f"\nCarnot Limits:")
    print(f"  COP (refrigeration):    {ideal['COP_R_carnot']:.2f}")
    print(f"  COP (heat pump):        {ideal['COP_HP_carnot']:.2f}")

    print(f"\nEfficiency Comparison:")
    print(f"  Ideal / Carnot: {ideal['COP_R']/ideal['COP_R_carnot']*100:.1f}%")
    print(f"  Real / Carnot:  {real['COP_R']/ideal['COP_R_carnot']*100:.1f}%")
    print(f"  Real / Ideal:   {real['COP_R']/ideal['COP_R']*100:.1f}%")


if __name__ == "__main__":
    main()
