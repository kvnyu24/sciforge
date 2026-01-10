"""
Example demonstrating special relativistic time dilation.

This example shows how moving clocks run slower, with applications
to GPS satellites and particle physics.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.relativity import LorentzTransform


def lorentz_factor(v, c=1.0):
    """Calculate the Lorentz factor γ = 1/√(1-v²/c²)."""
    return 1.0 / np.sqrt(1 - (v/c)**2)


def dilated_time(t0, v, c=1.0):
    """Calculate dilated time Δt = γΔt₀."""
    gamma = lorentz_factor(v, c)
    return gamma * t0


def main():
    c = 1.0  # Speed of light (natural units)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Time dilation factor vs velocity
    ax1 = axes[0, 0]

    v_range = np.linspace(0, 0.999*c, 500)
    gamma_range = lorentz_factor(v_range, c)

    ax1.plot(v_range/c, gamma_range, 'b-', lw=2, label='γ = Δt/Δt₀')
    ax1.axhline(y=1, color='gray', linestyle=':', alpha=0.5)

    ax1.set_xlabel('Velocity (v/c)')
    ax1.set_ylabel('Time Dilation Factor γ')
    ax1.set_title('Time Dilation vs Velocity')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 25)

    # Mark notable velocities
    notable = [(0.5, '0.5c'), (0.9, '0.9c'), (0.99, '0.99c')]
    for v, label in notable:
        gamma = lorentz_factor(v*c, c)
        ax1.plot(v, gamma, 'ro', markersize=8)
        ax1.annotate(f'{label}: γ={gamma:.2f}', (v, gamma),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)

    # Plot 2: Twin paradox visualization
    ax2 = axes[0, 1]

    # Scenario: One twin travels at 0.8c for 10 years (Earth time) and returns
    v_travel = 0.8 * c
    T_earth = 20  # Total Earth time (10 years out, 10 back)
    gamma = lorentz_factor(v_travel, c)
    T_traveler = T_earth / gamma

    # Create timeline
    t_earth = np.linspace(0, T_earth, 100)
    t_traveler = t_earth / gamma

    ax2.plot(t_earth, t_earth, 'b-', lw=2, label='Stay-at-home twin (Earth)')
    ax2.plot(t_earth, t_traveler, 'r-', lw=2, label='Traveling twin')
    ax2.plot([T_earth, T_earth], [T_earth, T_traveler], 'g--', lw=2)

    ax2.set_xlabel('Earth Time (years)')
    ax2.set_ylabel('Proper Time Elapsed (years)')
    ax2.set_title(f'Twin Paradox (v = {v_travel/c}c)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Annotate age difference
    ax2.annotate(f'Age difference:\n{T_earth - T_traveler:.1f} years',
                xy=(T_earth, (T_earth + T_traveler)/2), xytext=(T_earth - 5, 15),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 3: GPS satellite time correction
    ax3 = axes[1, 0]

    # GPS satellite parameters
    v_gps = 3870  # m/s (GPS orbital velocity)
    c_real = 3e8  # m/s

    # Time dilation per day (special relativistic effect)
    seconds_per_day = 86400
    gamma_gps = lorentz_factor(v_gps, c_real)
    sr_effect = (gamma_gps - 1) * seconds_per_day * 1e6  # microseconds

    # General relativistic effect (gravitational time dilation) - approximate
    # At GPS altitude (~20,200 km), clocks run faster due to weaker gravity
    gr_effect = 45.7  # microseconds per day (clocks run faster)

    # Net effect
    net_effect = gr_effect - sr_effect

    effects = ['Special Relativity\n(velocity)', 'General Relativity\n(gravity)', 'Net Effect']
    values = [-sr_effect, gr_effect, net_effect]
    colors = ['red', 'blue', 'green']

    bars = ax3.bar(effects, values, color=colors, alpha=0.7, edgecolor='black')

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax3.annotate(f'{val:.1f} μs/day',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -15), textcoords='offset points',
                    ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)

    ax3.axhline(y=0, color='gray', linestyle='-', lw=1)
    ax3.set_ylabel('Time Difference (μs/day)')
    ax3.set_title('GPS Time Corrections')
    ax3.grid(True, alpha=0.3, axis='y')

    ax3.text(0.5, -0.15, f'Without corrections: GPS errors of ~10 km/day!',
             transform=ax3.transAxes, fontsize=10, ha='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 4: Particle accelerator - muon lifetime
    ax4 = axes[1, 1]

    # Muon parameters
    tau_0 = 2.2e-6  # Rest lifetime (s)
    tau_0_us = tau_0 * 1e6  # microseconds

    # Different velocities
    beta_values = np.array([0, 0.5, 0.9, 0.99, 0.999, 0.9999])
    gamma_values = lorentz_factor(beta_values, 1.0)
    tau_dilated = gamma_values * tau_0_us

    x_pos = np.arange(len(beta_values))

    bars = ax4.bar(x_pos, tau_dilated, color=plt.cm.viridis(beta_values), alpha=0.8, edgecolor='black')

    ax4.axhline(y=tau_0_us, color='red', linestyle='--', lw=2, label=f'Rest lifetime = {tau_0_us:.2f} μs')

    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f'{b}c' for b in beta_values])
    ax4.set_xlabel('Muon Velocity')
    ax4.set_ylabel('Observed Lifetime (μs)')
    ax4.set_title('Muon Lifetime Dilation')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_yscale('log')

    # Add gamma values as labels
    for bar, gamma in zip(bars, gamma_values):
        height = bar.get_height()
        ax4.annotate(f'γ={gamma:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8, rotation=45)

    plt.suptitle('Special Relativistic Time Dilation\n'
                 'Δt = γΔt₀ = Δt₀/√(1 - v²/c²)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'time_dilation.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'time_dilation.png')}")


if __name__ == "__main__":
    main()
