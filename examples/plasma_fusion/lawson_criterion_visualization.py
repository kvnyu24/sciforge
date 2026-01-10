"""
Experiment 264: Lawson Criterion Visualization

Demonstrates the Lawson criterion for fusion energy breakeven -
the minimum conditions for fusion power to exceed losses.

Physical concepts:
- Fusion power scales as n^2 <sigma*v>
- Losses include radiation and confinement
- Breakeven requires n*tau > threshold
- Triple product n*T*tau is key figure of merit
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.plasma import LawsonCriterion

# Physical constants
e = 1.602e-19
k_B = 1.381e-23


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Create Lawson objects for different fuels
    lawson_DT = LawsonCriterion('DT')
    lawson_DD = LawsonCriterion('DD')

    # Plot 1: Fusion reactivity vs temperature
    ax1 = axes[0, 0]

    T_keV = np.logspace(0, 3, 200)  # 1 to 1000 keV
    T_K = T_keV * 1e3 * e / k_B

    sigma_v_DT = np.array([lawson_DT.reactivity(T) for T in T_K])
    sigma_v_DD = np.array([lawson_DD.reactivity(T) for T in T_K])

    ax1.loglog(T_keV, sigma_v_DT * 1e6, 'b-', lw=2, label='D-T')
    ax1.loglog(T_keV, sigma_v_DD * 1e6, 'r--', lw=2, label='D-D')

    # Mark optimal temperatures
    T_opt_DT = lawson_DT.optimal_temperature_keV
    T_opt_DD = lawson_DD.optimal_temperature_keV

    sigma_v_max_DT = lawson_DT.reactivity(lawson_DT.optimal_temperature)
    sigma_v_max_DD = lawson_DD.reactivity(lawson_DD.optimal_temperature)

    ax1.plot(T_opt_DT, sigma_v_max_DT * 1e6, 'bo', markersize=10)
    ax1.plot(T_opt_DD, sigma_v_max_DD * 1e6, 'ro', markersize=10)

    ax1.annotate(f'D-T optimal\n{T_opt_DT:.0f} keV',
                 xy=(T_opt_DT, sigma_v_max_DT * 1e6),
                 xytext=(30, sigma_v_max_DT * 1e6 * 3), fontsize=9,
                 arrowprops=dict(arrowstyle='->', color='blue'))

    ax1.set_xlabel('Temperature (keV)')
    ax1.set_ylabel('$\\langle\\sigma v\\rangle$ (10$^{-6}$ m$^3$/s)')
    ax1.set_title('Fusion Reactivity vs Temperature')
    ax1.legend()
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_xlim(1, 1000)
    ax1.set_ylim(1e-28, 1e-20)

    # Plot 2: Required n*tau for breakeven
    ax2 = axes[0, 1]

    T_keV = np.linspace(5, 100, 200)
    T_K = T_keV * 1e3 * e / k_B

    ntau_DT = np.array([lawson_DT.lawson_ntau(T) for T in T_K])
    ntau_DD = np.array([lawson_DD.lawson_ntau(T) for T in T_K])

    # Filter out invalid values
    valid_DT = ntau_DT > 0
    valid_DD = ntau_DD > 0

    ax2.semilogy(T_keV[valid_DT], ntau_DT[valid_DT], 'b-', lw=2, label='D-T breakeven')
    ax2.semilogy(T_keV[valid_DD], ntau_DD[valid_DD], 'r--', lw=2, label='D-D breakeven')

    # Mark minimum n*tau
    min_idx_DT = np.argmin(ntau_DT[valid_DT])
    T_min_DT = T_keV[valid_DT][min_idx_DT]
    ntau_min_DT = ntau_DT[valid_DT][min_idx_DT]

    ax2.plot(T_min_DT, ntau_min_DT, 'bo', markersize=10)
    ax2.annotate(f'Minimum: {ntau_min_DT:.1e} m$^{{-3}}$s\nat {T_min_DT:.0f} keV',
                 xy=(T_min_DT, ntau_min_DT),
                 xytext=(T_min_DT + 20, ntau_min_DT * 3), fontsize=9,
                 arrowprops=dict(arrowstyle='->', color='blue'))

    # Show achieved values (approximate)
    experiments = {
        'JET (1997)': (25, 1.5e20),
        'TFTR': (30, 6e19),
        'JT-60U': (35, 1.5e21),
    }

    for name, (T, ntau) in experiments.items():
        ax2.plot(T, ntau, 's', markersize=8, label=name)

    ax2.set_xlabel('Temperature (keV)')
    ax2.set_ylabel('Required $n\\tau_E$ (m$^{-3}$s)')
    ax2.set_title('Lawson Criterion: $n\\tau_E$ for Breakeven')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_xlim(5, 100)
    ax2.set_ylim(1e18, 1e23)

    # Plot 3: Triple product operating space
    ax3 = axes[1, 0]

    # Create contour of n*T*tau required
    T_range = np.logspace(0, 2, 100)  # 1 to 100 keV
    T_K_range = T_range * 1e3 * e / k_B

    nTtau_required = np.array([lawson_DT.triple_product(T) * k_B / e
                               for T in T_K_range])  # Convert to m^-3 * keV * s

    # Filter valid
    valid = nTtau_required > 0

    ax3.loglog(T_range[valid], nTtau_required[valid] / 1e21, 'b-', lw=2,
               label='D-T breakeven')

    # Ignition condition (alpha heating sustains burn)
    nTtau_ignition = np.array([lawson_DT.ignition_condition(T) * T * k_B / e
                               for T in T_K_range])
    ax3.loglog(T_range[valid], nTtau_ignition[valid] / 1e21, 'r--', lw=2,
               label='Ignition')

    # Mark experimental achievements
    experiments_nTtau = {
        'JET (2021)': (10, 0.6),
        'JT-60U': (15, 1.5),
        'ITER target': (13, 3.5),
        'SPARC target': (12, 2.0),
    }

    markers = ['o', 's', '^', 'D']
    for (name, (T, nTtau)), marker in zip(experiments_nTtau.items(), markers):
        ax3.plot(T, nTtau, marker, markersize=10, label=name)

    ax3.set_xlabel('Temperature (keV)')
    ax3.set_ylabel('Triple product $n T \\tau_E$ (10$^{21}$ m$^{-3}$ keV s)')
    ax3.set_title('Fusion Triple Product')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3, which='both')
    ax3.set_xlim(1, 100)

    # Shade ignition region
    ax3.fill_between(T_range[valid], nTtau_ignition[valid] / 1e21, 100,
                     alpha=0.1, color='green', label='Ignition region')

    # Plot 4: Power balance at different conditions
    ax4 = axes[1, 1]

    # Fixed density, vary temperature
    n = 1e20  # m^-3

    T_keV = np.linspace(5, 50, 100)
    T_K = T_keV * 1e3 * e / k_B

    P_fusion = np.array([lawson_DT.fusion_power_density(n, T) for T in T_K])
    P_brem = np.array([lawson_DT.bremsstrahlung_loss(n, T) for T in T_K])

    # Confinement loss (parametric)
    tau_E = 1.0  # seconds
    P_conf = 3 * n * k_B * T_K / tau_E  # Energy confinement loss

    ax4.semilogy(T_keV, P_fusion / 1e6, 'g-', lw=2, label='Fusion power')
    ax4.semilogy(T_keV, P_brem / 1e6, 'r--', lw=2, label='Bremsstrahlung')
    ax4.semilogy(T_keV, P_conf / 1e6, 'b:', lw=2, label=f'Confinement ($\\tau_E$ = {tau_E} s)')
    ax4.semilogy(T_keV, (P_brem + P_conf) / 1e6, 'k-', lw=1.5, label='Total loss')

    # Find breakeven
    P_net = P_fusion - P_brem - P_conf
    breakeven_idx = np.where(np.diff(np.sign(P_net)))[0]
    if len(breakeven_idx) > 0:
        T_breakeven = T_keV[breakeven_idx[0]]
        ax4.axvline(x=T_breakeven, color='gray', linestyle=':', alpha=0.7)
        ax4.text(T_breakeven + 1, 1, f'Breakeven\n{T_breakeven:.0f} keV', fontsize=9)

    ax4.set_xlabel('Temperature (keV)')
    ax4.set_ylabel('Power Density (MW/m$^3$)')
    ax4.set_title(f'Power Balance (n = {n:.0e} m$^{{-3}}$)')
    ax4.legend()
    ax4.grid(True, alpha=0.3, which='both')
    ax4.set_xlim(5, 50)
    ax4.set_ylim(1e-3, 1e3)

    # Shade ignition region
    ignition_mask = P_fusion > (P_brem + P_conf)
    if np.any(ignition_mask):
        ax4.fill_between(T_keV, 1e-3, 1e3, where=ignition_mask,
                         alpha=0.1, color='green')
        ax4.text(30, 0.1, 'Net gain', fontsize=10, color='green')

    plt.suptitle('Experiment 264: Lawson Criterion for Fusion\n'
                 'Breakeven requires $n\\tau_E > 1.5 \\times 10^{20}$ m$^{-3}$s for D-T at 15 keV',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'lawson_criterion_visualization.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'lawson_criterion_visualization.png')}")


if __name__ == "__main__":
    main()
