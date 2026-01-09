"""
Example demonstrating coupled oscillators and normal modes.

This example shows two masses connected by springs exhibiting
two distinct normal modes: in-phase and anti-phase oscillation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.oscillations import CoupledOscillator


def simulate_coupled_system(m, k1, k2, kc, x1_0, x2_0, v1_0, v2_0, t_final, dt):
    """
    Simulate two coupled oscillators.

    Args:
        m: Mass of each oscillator (kg)
        k1, k2: Spring constants for outer springs (N/m)
        kc: Coupling spring constant (N/m)
        x1_0, x2_0: Initial displacements (m)
        v1_0, v2_0: Initial velocities (m/s)
        t_final: Simulation duration (s)
        dt: Time step (s)

    Returns:
        Dictionary with time and position data
    """
    # CoupledOscillator takes arrays for masses, spring constants, positions, velocities
    system = CoupledOscillator(
        masses=[m, m],
        spring_constants=[k1, k2],
        initial_positions=[np.array([x1_0, 0.0, 0.0]), np.array([x2_0, 0.0, 0.0])],
        initial_velocities=[np.array([v1_0, 0.0, 0.0]), np.array([v2_0, 0.0, 0.0])],
        coupling_constants=[kc]
    )

    times = [0]
    x1 = [x1_0]
    x2 = [x2_0]

    t = 0
    while t < t_final:
        system.update(dt)
        t += dt
        times.append(t)
        x1.append(system.oscillators[0].position[0])
        x2.append(system.oscillators[1].position[0])

    return {
        'time': np.array(times),
        'x1': np.array(x1),
        'x2': np.array(x2)
    }


def main():
    # System parameters
    m = 1.0      # kg
    k = 10.0     # N/m (outer springs)
    kc = 2.0     # N/m (coupling spring)

    # Normal mode frequencies
    omega1 = np.sqrt(k / m)                  # In-phase mode
    omega2 = np.sqrt((k + 2*kc) / m)         # Anti-phase mode
    T1 = 2 * np.pi / omega1
    T2 = 2 * np.pi / omega2

    t_final = 20.0
    dt = 0.001

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Case 1: In-phase normal mode
    results1 = simulate_coupled_system(m, k, k, kc, 1.0, 1.0, 0.0, 0.0, t_final, dt)

    ax1 = axes[0, 0]
    ax1.plot(results1['time'], results1['x1'], 'b-', lw=1.5, label='Mass 1')
    ax1.plot(results1['time'], results1['x2'], 'r--', lw=1.5, label='Mass 2')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Displacement (m)')
    ax1.set_title(f'In-Phase Mode (ω = {omega1:.2f} rad/s)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Case 2: Anti-phase normal mode
    results2 = simulate_coupled_system(m, k, k, kc, 1.0, -1.0, 0.0, 0.0, t_final, dt)

    ax2 = axes[0, 1]
    ax2.plot(results2['time'], results2['x1'], 'b-', lw=1.5, label='Mass 1')
    ax2.plot(results2['time'], results2['x2'], 'r--', lw=1.5, label='Mass 2')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Displacement (m)')
    ax2.set_title(f'Anti-Phase Mode (ω = {omega2:.2f} rad/s)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Case 3: Energy transfer (beat phenomenon)
    results3 = simulate_coupled_system(m, k, k, kc, 1.0, 0.0, 0.0, 0.0, t_final, dt)

    ax3 = axes[0, 2]
    ax3.plot(results3['time'], results3['x1'], 'b-', lw=1, label='Mass 1', alpha=0.8)
    ax3.plot(results3['time'], results3['x2'], 'r-', lw=1, label='Mass 2', alpha=0.8)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Displacement (m)')
    ax3.set_title('Energy Transfer (Beats)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Beat frequency
    omega_beat = abs(omega2 - omega1) / 2
    T_beat = 2 * np.pi / omega_beat
    ax3.text(0.02, 0.98, f'Beat period ≈ {T_beat:.1f} s',
             transform=ax3.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Phase space for beat case
    ax4 = axes[1, 0]
    ax4.plot(results3['x1'], results3['x2'], 'b-', lw=0.5, alpha=0.7)
    ax4.plot(results3['x1'][0], results3['x2'][0], 'go', markersize=10, label='Start')
    ax4.set_xlabel('x₁ (m)')
    ax4.set_ylabel('x₂ (m)')
    ax4.set_title('Phase Space (x₁ vs x₂)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal')

    # Energy in each oscillator
    ax5 = axes[1, 1]
    v1 = np.gradient(results3['x1'], dt)
    v2 = np.gradient(results3['x2'], dt)
    E1 = 0.5 * m * v1**2 + 0.5 * k * results3['x1']**2
    E2 = 0.5 * m * v2**2 + 0.5 * k * results3['x2']**2

    ax5.plot(results3['time'], E1, 'b-', lw=1.5, label='Energy in Mass 1', alpha=0.8)
    ax5.plot(results3['time'], E2, 'r-', lw=1.5, label='Energy in Mass 2', alpha=0.8)
    ax5.plot(results3['time'], E1 + E2, 'k--', lw=1, label='Total Energy', alpha=0.5)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Energy (J)')
    ax5.set_title('Energy Transfer Between Oscillators')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # Normal mode diagram
    ax6 = axes[1, 2]
    ax6.set_xlim(-2, 6)
    ax6.set_ylim(-2, 4)

    # Draw the two modes
    # Mode 1: In-phase
    ax6.annotate('', xy=(1.5, 2.5), xytext=(0.5, 2.5),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax6.annotate('', xy=(3.5, 2.5), xytext=(2.5, 2.5),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax6.plot([0.5, 2.5], [2.5, 2.5], 'ko', markersize=15)
    ax6.text(1.5, 3.2, f'Mode 1: In-phase\nω₁ = {omega1:.2f} rad/s', ha='center', fontsize=10)

    # Mode 2: Anti-phase
    ax6.annotate('', xy=(1.5, 0.5), xytext=(0.5, 0.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax6.annotate('', xy=(1.5, 0.5), xytext=(2.5, 0.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax6.plot([0.5, 2.5], [0.5, 0.5], 'ko', markersize=15)
    ax6.text(1.5, -0.5, f'Mode 2: Anti-phase\nω₂ = {omega2:.2f} rad/s', ha='center', fontsize=10)

    ax6.set_title('Normal Modes')
    ax6.axis('off')

    plt.suptitle(f'Coupled Oscillators (m={m} kg, k={k} N/m, k_c={kc} N/m)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'coupled_oscillators.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'coupled_oscillators.png')}")


if __name__ == "__main__":
    main()
