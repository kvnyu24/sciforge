"""
Example demonstrating Otto and Diesel engine cycles.

Otto Cycle (gasoline engine):
1-2: Isentropic compression
2-3: Constant volume heat addition
3-4: Isentropic expansion
4-1: Constant volume heat rejection

Diesel Cycle (diesel engine):
1-2: Isentropic compression
2-3: Constant pressure heat addition
3-4: Isentropic expansion
4-1: Constant volume heat rejection

This example shows:
- P-V diagrams for both cycles
- Efficiency as a function of compression ratio
- Comparison of thermal efficiencies
- Effect of cutoff ratio on Diesel efficiency
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.core.constants import CONSTANTS


def otto_cycle_efficiency(compression_ratio, gamma=1.4):
    """
    Calculate Otto cycle thermal efficiency.

    eta = 1 - 1/r^(gamma-1)

    Args:
        compression_ratio: r = V1/V2
        gamma: Heat capacity ratio

    Returns:
        Thermal efficiency
    """
    return 1 - 1 / compression_ratio**(gamma - 1)


def diesel_cycle_efficiency(compression_ratio, cutoff_ratio, gamma=1.4):
    """
    Calculate Diesel cycle thermal efficiency.

    eta = 1 - (1/r^(gamma-1)) * (rc^gamma - 1) / (gamma*(rc - 1))

    Args:
        compression_ratio: r = V1/V2
        cutoff_ratio: rc = V3/V2 (volume at end of combustion / volume at start)
        gamma: Heat capacity ratio

    Returns:
        Thermal efficiency
    """
    term1 = 1 / compression_ratio**(gamma - 1)
    term2 = (cutoff_ratio**gamma - 1) / (gamma * (cutoff_ratio - 1))
    return 1 - term1 * term2


def generate_otto_cycle(P1, V1, compression_ratio, pressure_ratio, gamma=1.4):
    """
    Generate Otto cycle states and processes.

    Returns:
        Dictionary with states and process curves
    """
    # State 1: Initial state
    V1, P1 = V1, P1

    # State 2: After isentropic compression
    V2 = V1 / compression_ratio
    P2 = P1 * compression_ratio**gamma

    # State 3: After constant volume heat addition
    V3 = V2
    P3 = P2 * pressure_ratio

    # State 4: After isentropic expansion
    V4 = V1
    P4 = P3 * (V3 / V4)**gamma

    states = {
        'V': [V1, V2, V3, V4, V1],
        'P': [P1, P2, P3, P4, P1]
    }

    # Generate smooth curves for each process
    n_points = 100

    # 1-2: Isentropic compression
    V_12 = np.linspace(V1, V2, n_points)
    P_12 = P1 * (V1 / V_12)**gamma

    # 2-3: Constant volume
    V_23 = np.array([V2, V3])
    P_23 = np.array([P2, P3])

    # 3-4: Isentropic expansion
    V_34 = np.linspace(V3, V4, n_points)
    P_34 = P3 * (V3 / V_34)**gamma

    # 4-1: Constant volume
    V_41 = np.array([V4, V1])
    P_41 = np.array([P4, P1])

    curves = {
        'compression': (V_12, P_12),
        'heat_add': (V_23, P_23),
        'expansion': (V_34, P_34),
        'heat_reject': (V_41, P_41)
    }

    return states, curves


def generate_diesel_cycle(P1, V1, compression_ratio, cutoff_ratio, gamma=1.4):
    """
    Generate Diesel cycle states and processes.

    Returns:
        Dictionary with states and process curves
    """
    # State 1: Initial state
    V1, P1 = V1, P1

    # State 2: After isentropic compression
    V2 = V1 / compression_ratio
    P2 = P1 * compression_ratio**gamma

    # State 3: After constant pressure heat addition
    V3 = V2 * cutoff_ratio
    P3 = P2  # Constant pressure

    # State 4: After isentropic expansion
    V4 = V1
    P4 = P3 * (V3 / V4)**gamma

    states = {
        'V': [V1, V2, V3, V4, V1],
        'P': [P1, P2, P3, P4, P1]
    }

    # Generate smooth curves
    n_points = 100

    # 1-2: Isentropic compression
    V_12 = np.linspace(V1, V2, n_points)
    P_12 = P1 * (V1 / V_12)**gamma

    # 2-3: Constant pressure
    V_23 = np.linspace(V2, V3, n_points)
    P_23 = np.ones_like(V_23) * P2

    # 3-4: Isentropic expansion
    V_34 = np.linspace(V3, V4, n_points)
    P_34 = P3 * (V3 / V_34)**gamma

    # 4-1: Constant volume
    V_41 = np.array([V4, V1])
    P_41 = np.array([P4, P1])

    curves = {
        'compression': (V_12, P_12),
        'heat_add': (V_23, P_23),
        'expansion': (V_34, P_34),
        'heat_reject': (V_41, P_41)
    }

    return states, curves


def main():
    # Parameters
    gamma = 1.4  # Air (diatomic)
    R = CONSTANTS['R']

    # Initial conditions
    P1 = 101325  # Pa (1 atm)
    V1 = 0.001   # m^3 (1 liter)
    T1 = 300     # K

    # Cycle parameters
    compression_ratio_otto = 10
    pressure_ratio_otto = 3.0  # P3/P2
    compression_ratio_diesel = 20
    cutoff_ratio_diesel = 2.0  # V3/V2

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Otto cycle P-V diagram
    ax1 = axes[0, 0]

    states_otto, curves_otto = generate_otto_cycle(
        P1, V1, compression_ratio_otto, pressure_ratio_otto, gamma)

    # Plot cycle curves
    colors = {'compression': 'blue', 'heat_add': 'red',
              'expansion': 'green', 'heat_reject': 'purple'}
    labels = {'compression': '1-2: Isentropic compression',
              'heat_add': '2-3: Const. volume heat addition',
              'expansion': '3-4: Isentropic expansion',
              'heat_reject': '4-1: Const. volume heat rejection'}

    for process, (V, P) in curves_otto.items():
        ax1.plot(V * 1000, P / 1000, color=colors[process], lw=2, label=labels[process])

    # Mark states
    for i, (V, P) in enumerate(zip(states_otto['V'][:-1], states_otto['P'][:-1])):
        ax1.plot(V * 1000, P / 1000, 'ko', markersize=10)
        ax1.annotate(f'{i+1}', (V * 1000, P / 1000), xytext=(5, 5),
                    textcoords='offset points', fontsize=12, fontweight='bold')

    eta_otto = otto_cycle_efficiency(compression_ratio_otto, gamma)
    ax1.set_xlabel('Volume (L)', fontsize=12)
    ax1.set_ylabel('Pressure (kPa)', fontsize=12)
    ax1.set_title(f'Otto Cycle (r = {compression_ratio_otto})\n'
                  f'Thermal Efficiency: {eta_otto*100:.1f}%', fontsize=12)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Diesel cycle P-V diagram
    ax2 = axes[0, 1]

    states_diesel, curves_diesel = generate_diesel_cycle(
        P1, V1, compression_ratio_diesel, cutoff_ratio_diesel, gamma)

    labels_diesel = {'compression': '1-2: Isentropic compression',
                     'heat_add': '2-3: Const. pressure heat addition',
                     'expansion': '3-4: Isentropic expansion',
                     'heat_reject': '4-1: Const. volume heat rejection'}

    for process, (V, P) in curves_diesel.items():
        ax2.plot(V * 1000, P / 1000, color=colors[process], lw=2,
                label=labels_diesel[process])

    # Mark states
    for i, (V, P) in enumerate(zip(states_diesel['V'][:-1], states_diesel['P'][:-1])):
        ax2.plot(V * 1000, P / 1000, 'ko', markersize=10)
        ax2.annotate(f'{i+1}', (V * 1000, P / 1000), xytext=(5, 5),
                    textcoords='offset points', fontsize=12, fontweight='bold')

    eta_diesel = diesel_cycle_efficiency(compression_ratio_diesel,
                                          cutoff_ratio_diesel, gamma)
    ax2.set_xlabel('Volume (L)', fontsize=12)
    ax2.set_ylabel('Pressure (kPa)', fontsize=12)
    ax2.set_title(f'Diesel Cycle (r = {compression_ratio_diesel}, rc = {cutoff_ratio_diesel})\n'
                  f'Thermal Efficiency: {eta_diesel*100:.1f}%', fontsize=12)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Efficiency vs compression ratio
    ax3 = axes[1, 0]

    r = np.linspace(2, 25, 100)
    eta_otto_curve = otto_cycle_efficiency(r, gamma)

    # Diesel for various cutoff ratios
    cutoff_ratios = [1.5, 2.0, 2.5, 3.0]
    colors_rc = plt.cm.Oranges(np.linspace(0.4, 0.9, len(cutoff_ratios)))

    ax3.plot(r, eta_otto_curve * 100, 'b-', lw=2, label='Otto cycle')

    for rc, color in zip(cutoff_ratios, colors_rc):
        eta_d = diesel_cycle_efficiency(r, rc, gamma)
        ax3.plot(r, eta_d * 100, '--', color=color, lw=2, label=f'Diesel (rc={rc})')

    # Mark typical operating points
    ax3.plot(compression_ratio_otto, eta_otto * 100, 'bo', markersize=10)
    ax3.annotate(f'Typical gasoline\n(r={compression_ratio_otto})',
                xy=(compression_ratio_otto, eta_otto * 100),
                xytext=(compression_ratio_otto + 2, eta_otto * 100 - 8),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=9)

    ax3.plot(compression_ratio_diesel, eta_diesel * 100, 'o', color='orange',
             markersize=10)
    ax3.annotate(f'Typical diesel\n(r={compression_ratio_diesel})',
                xy=(compression_ratio_diesel, eta_diesel * 100),
                xytext=(compression_ratio_diesel - 5, eta_diesel * 100 + 5),
                arrowprops=dict(arrowstyle='->', color='orange'),
                fontsize=9)

    ax3.set_xlabel('Compression Ratio (r)', fontsize=12)
    ax3.set_ylabel('Thermal Efficiency (%)', fontsize=12)
    ax3.set_title('Thermal Efficiency vs Compression Ratio', fontsize=12)
    ax3.legend(loc='lower right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(2, 25)
    ax3.set_ylim(20, 75)

    # Plot 4: Effect of cutoff ratio on Diesel efficiency
    ax4 = axes[1, 1]

    rc_range = np.linspace(1.1, 4, 100)
    compression_ratios = [15, 18, 20, 22, 25]
    colors_r = plt.cm.Blues(np.linspace(0.4, 0.9, len(compression_ratios)))

    for r_val, color in zip(compression_ratios, colors_r):
        eta_d = diesel_cycle_efficiency(r_val, rc_range, gamma)
        ax4.plot(rc_range, eta_d * 100, color=color, lw=2, label=f'r = {r_val}')

    # Compare with Otto at same compression ratio
    for r_val in [15, 20, 25]:
        eta_o = otto_cycle_efficiency(r_val, gamma)
        ax4.axhline(y=eta_o * 100, color='gray', linestyle=':', alpha=0.5)

    ax4.set_xlabel('Cutoff Ratio (rc = V3/V2)', fontsize=12)
    ax4.set_ylabel('Diesel Efficiency (%)', fontsize=12)
    ax4.set_title('Diesel Efficiency vs Cutoff Ratio\n'
                  '(Horizontal lines: Otto efficiency at same r)', fontsize=12)
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(1, 4)

    # Add annotation
    ax4.text(0.05, 0.05,
             'Lower cutoff ratio = higher efficiency\nbut less power per stroke',
             transform=ax4.transAxes, fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('Otto and Diesel Engine Cycles Analysis\n'
                 f'gamma = {gamma} (air-standard analysis)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'otto_diesel_cycles.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'otto_diesel_cycles.png')}")


if __name__ == "__main__":
    main()
