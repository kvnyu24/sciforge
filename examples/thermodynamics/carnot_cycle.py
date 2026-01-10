"""
Example demonstrating the Carnot cycle.

This example shows the most efficient heat engine cycle, plotting
P-V and T-S diagrams, and comparing efficiency with other cycles.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.thermodynamics import ThermalSystem


def isothermal_process(V_start, V_end, T, n=1.0, R=8.314):
    """Calculate P-V curve for isothermal process."""
    V = np.linspace(V_start, V_end, 100)
    P = n * R * T / V
    return V, P


def adiabatic_process(V_start, V_end, P_start, gamma=1.4):
    """Calculate P-V curve for adiabatic process."""
    V = np.linspace(V_start, V_end, 100)
    # PV^γ = constant
    P = P_start * (V_start / V)**gamma
    return V, P


def main():
    # Parameters
    n = 1.0          # Moles of gas
    R = 8.314        # Gas constant (J/mol·K)
    gamma = 1.4      # Heat capacity ratio (diatomic gas)

    # Temperatures for hot and cold reservoirs
    T_hot = 600.0    # K
    T_cold = 300.0   # K

    # Define the cycle
    # State 1: Start of isothermal expansion
    V1 = 0.001       # m³
    P1 = n * R * T_hot / V1

    # State 2: End of isothermal expansion, start of adiabatic expansion
    V2 = 0.002       # m³ (expanded isothermally)
    P2 = n * R * T_hot / V2

    # State 3: End of adiabatic expansion
    # T_hot * V2^(γ-1) = T_cold * V3^(γ-1)
    V3 = V2 * (T_hot / T_cold)**(1/(gamma-1))
    P3 = n * R * T_cold / V3

    # State 4: End of isothermal compression, start of adiabatic compression
    V4 = V1 * (T_hot / T_cold)**(1/(gamma-1))
    P4 = n * R * T_cold / V4

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: P-V Diagram
    ax1 = axes[0, 0]

    # Isothermal expansion (1→2)
    V_12, P_12 = isothermal_process(V1, V2, T_hot, n, R)
    ax1.plot(V_12 * 1000, P_12 / 1000, 'r-', lw=2, label=f'Isothermal expansion (T={T_hot}K)')

    # Adiabatic expansion (2→3)
    V_23, P_23 = adiabatic_process(V2, V3, P2, gamma)
    ax1.plot(V_23 * 1000, P_23 / 1000, 'b-', lw=2, label='Adiabatic expansion')

    # Isothermal compression (3→4)
    V_34, P_34 = isothermal_process(V3, V4, T_cold, n, R)
    ax1.plot(V_34 * 1000, P_34 / 1000, 'c-', lw=2, label=f'Isothermal compression (T={T_cold}K)')

    # Adiabatic compression (4→1)
    V_41, P_41 = adiabatic_process(V4, V1, P4, gamma)
    ax1.plot(V_41 * 1000, P_41 / 1000, 'm-', lw=2, label='Adiabatic compression')

    # Mark states
    states = [(V1, P1, '1'), (V2, P2, '2'), (V3, P3, '3'), (V4, P4, '4')]
    for V, P, label in states:
        ax1.plot(V * 1000, P / 1000, 'ko', markersize=10)
        ax1.annotate(label, (V * 1000, P / 1000), xytext=(5, 5),
                    textcoords='offset points', fontsize=12, fontweight='bold')

    ax1.set_xlabel('Volume (L)')
    ax1.set_ylabel('Pressure (kPa)')
    ax1.set_title('P-V Diagram')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: T-S Diagram
    ax2 = axes[0, 1]

    # Calculate entropy changes
    # For isothermal: ΔS = nR ln(V2/V1) = Q/T
    S1 = 0  # Reference
    S2 = S1 + n * R * np.log(V2/V1)  # After isothermal expansion
    S3 = S2  # Adiabatic (no entropy change)
    S4 = S3 + n * R * np.log(V4/V3)  # After isothermal compression
    # S4 should equal S1 (cycle)

    # Plot T-S diagram
    # 1→2: Isothermal expansion at T_hot
    S_12 = np.linspace(S1, S2, 100)
    T_12 = np.ones_like(S_12) * T_hot
    ax2.plot(S_12, T_12, 'r-', lw=2)

    # 2→3: Adiabatic expansion (vertical line)
    ax2.plot([S2, S3], [T_hot, T_cold], 'b-', lw=2)

    # 3→4: Isothermal compression at T_cold
    S_34 = np.linspace(S3, S4, 100)
    T_34 = np.ones_like(S_34) * T_cold
    ax2.plot(S_34, T_34, 'c-', lw=2)

    # 4→1: Adiabatic compression (vertical line)
    ax2.plot([S4, S1], [T_cold, T_hot], 'm-', lw=2)

    # Mark states
    ax2.plot(S1, T_hot, 'ko', markersize=10)
    ax2.plot(S2, T_hot, 'ko', markersize=10)
    ax2.plot(S3, T_cold, 'ko', markersize=10)
    ax2.plot(S4, T_cold, 'ko', markersize=10)

    ax2.annotate('1', (S1, T_hot), xytext=(5, 5), textcoords='offset points', fontsize=12, fontweight='bold')
    ax2.annotate('2', (S2, T_hot), xytext=(5, 5), textcoords='offset points', fontsize=12, fontweight='bold')
    ax2.annotate('3', (S3, T_cold), xytext=(5, -15), textcoords='offset points', fontsize=12, fontweight='bold')
    ax2.annotate('4', (S4, T_cold), xytext=(5, -15), textcoords='offset points', fontsize=12, fontweight='bold')

    # Shade the area (net work)
    ax2.fill([S1, S2, S3, S4], [T_hot, T_hot, T_cold, T_cold], alpha=0.2, color='green')

    ax2.set_xlabel('Entropy S (J/K)')
    ax2.set_ylabel('Temperature (K)')
    ax2.set_title('T-S Diagram')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Efficiency comparison
    ax3 = axes[1, 0]

    T_hot_range = np.linspace(350, 1000, 100)
    eta_carnot = 1 - T_cold / T_hot_range

    # Other cycles for comparison (approximations)
    eta_otto = 1 - (1/8)**(gamma-1)  # Compression ratio of 8
    eta_diesel = 0.45  # Typical diesel efficiency

    ax3.plot(T_hot_range, eta_carnot * 100, 'b-', lw=2, label='Carnot')
    ax3.axhline(y=eta_otto * 100, color='r', linestyle='--', lw=2, label=f'Otto (r=8)')
    ax3.axhline(y=eta_diesel * 100, color='g', linestyle=':', lw=2, label='Typical Diesel')

    # Mark current operating point
    eta_current = 1 - T_cold / T_hot
    ax3.plot(T_hot, eta_current * 100, 'ko', markersize=10)
    ax3.annotate(f'η = {eta_current*100:.1f}%', (T_hot, eta_current * 100),
                xytext=(10, -20), textcoords='offset points', fontsize=10)

    ax3.set_xlabel('Hot Reservoir Temperature (K)')
    ax3.set_ylabel('Efficiency (%)')
    ax3.set_title(f'Carnot Efficiency (T_cold = {T_cold}K)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 100)

    # Plot 4: Energy flow diagram
    ax4 = axes[1, 1]

    # Calculate heat and work
    Q_hot = n * R * T_hot * np.log(V2/V1)  # Heat absorbed
    Q_cold = n * R * T_cold * np.log(V3/V4)  # Heat rejected (negative)
    W_net = Q_hot + Q_cold  # Net work done

    # Bar chart
    labels = ['Q_hot\n(absorbed)', 'W_net\n(work)', 'Q_cold\n(rejected)']
    values = [Q_hot/1000, W_net/1000, -Q_cold/1000]
    colors = ['red', 'green', 'blue']

    bars = ax4.bar(labels, values, color=colors, alpha=0.7, edgecolor='black')

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax4.annotate(f'{val:.1f} kJ',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10)

    ax4.set_ylabel('Energy (kJ)')
    ax4.set_title('Energy Flow in Carnot Cycle')
    ax4.grid(True, alpha=0.3, axis='y')

    # Add efficiency annotation
    ax4.text(0.5, 0.9, f'Efficiency η = W/Q_hot = {W_net/Q_hot*100:.1f}%\n'
             f'Carnot limit: η_max = 1 - T_cold/T_hot = {(1-T_cold/T_hot)*100:.1f}%',
             transform=ax4.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'Carnot Cycle Analysis (T_hot = {T_hot}K, T_cold = {T_cold}K)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'carnot_cycle.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'carnot_cycle.png')}")


if __name__ == "__main__":
    main()
