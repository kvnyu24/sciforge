"""
Example demonstrating damped harmonic oscillator behavior.

This example shows the three damping regimes:
- Underdamped: oscillates with decreasing amplitude
- Critically damped: fastest return to equilibrium without oscillation
- Overdamped: slow exponential return to equilibrium
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.oscillations import HarmonicOscillator


def simulate_damped_oscillator(k, m, damping, x0, v0, t_final, dt):
    """
    Simulate a damped harmonic oscillator.

    Args:
        k: Spring constant (N/m)
        m: Mass (kg)
        damping: Damping coefficient (kg/s)
        x0: Initial position (m)
        v0: Initial velocity (m/s)
        t_final: Simulation duration (s)
        dt: Time step (s)

    Returns:
        Dictionary with time, position, velocity, and energy data
    """
    oscillator = HarmonicOscillator(
        mass=m,
        spring_constant=k,
        position=np.array([x0, 0.0, 0.0]),
        velocity=np.array([v0, 0.0, 0.0]),
        damping=damping
    )

    times = [0]
    positions = [x0]
    velocities = [v0]
    energies = [oscillator.total_energy()]

    t = 0
    while t < t_final:
        oscillator.update(None, dt)  # No external force
        t += dt
        times.append(t)
        positions.append(oscillator.position[0])  # x-component
        velocities.append(oscillator.velocity[0])  # x-component
        energies.append(oscillator.total_energy())

    return {
        'time': np.array(times),
        'position': np.array(positions),
        'velocity': np.array(velocities),
        'energy': np.array(energies)
    }


def calculate_damping_regime(k, m, damping):
    """Determine the damping regime."""
    omega_0 = np.sqrt(k / m)
    gamma = damping / m
    critical_damping = 2 * m * omega_0

    if damping < critical_damping * 0.99:
        regime = "Underdamped"
        omega_d = np.sqrt(omega_0**2 - (gamma/2)**2)
        info = f"ωd = {omega_d:.3f} rad/s"
    elif damping > critical_damping * 1.01:
        regime = "Overdamped"
        info = f"γ/ω₀ = {gamma/omega_0:.2f}"
    else:
        regime = "Critically damped"
        info = f"γ = 2ω₀"

    return regime, info, critical_damping


def main():
    # System parameters
    k = 10.0    # Spring constant (N/m)
    m = 1.0     # Mass (kg)
    x0 = 1.0    # Initial displacement (m)
    v0 = 0.0    # Initial velocity (m/s)

    omega_0 = np.sqrt(k / m)
    critical_damping = 2 * m * omega_0

    # Three damping cases
    damping_cases = {
        'Underdamped': 0.5 * critical_damping,
        'Critically Damped': critical_damping,
        'Overdamped': 3.0 * critical_damping
    }

    t_final = 10.0
    dt = 0.01

    # Run simulations
    results = {}
    for name, damping in damping_cases.items():
        results[name] = simulate_damped_oscillator(k, m, damping, x0, v0, t_final, dt)
        regime, info, _ = calculate_damping_regime(k, m, damping)
        results[name]['regime'] = regime
        results[name]['info'] = info
        results[name]['damping'] = damping

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {'Underdamped': 'blue', 'Critically Damped': 'green', 'Overdamped': 'red'}

    # Plot 1: Position vs time
    ax1 = axes[0, 0]
    for name, data in results.items():
        ax1.plot(data['time'], data['position'], color=colors[name], lw=2, label=name)

    # Add envelope for underdamped case
    gamma = damping_cases['Underdamped'] / m
    envelope = x0 * np.exp(-gamma/2 * results['Underdamped']['time'])
    ax1.plot(results['Underdamped']['time'], envelope, 'b--', alpha=0.5, lw=1)
    ax1.plot(results['Underdamped']['time'], -envelope, 'b--', alpha=0.5, lw=1)

    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (m)')
    ax1.set_title('Position vs Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Phase space (position vs velocity)
    ax2 = axes[0, 1]
    for name, data in results.items():
        ax2.plot(data['position'], data['velocity'], color=colors[name], lw=1.5, label=name, alpha=0.8)
        # Mark starting point
        ax2.plot(data['position'][0], data['velocity'][0], 'o', color=colors[name], markersize=8)

    ax2.set_xlabel('Position (m)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('Phase Space Trajectories')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')

    # Plot 3: Energy decay
    ax3 = axes[1, 0]
    for name, data in results.items():
        ax3.plot(data['time'], data['energy'], color=colors[name], lw=2, label=name)

    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Total Energy (J)')
    ax3.set_title('Energy Dissipation')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')

    # Plot 4: Damping regime diagram
    ax4 = axes[1, 1]
    damping_range = np.linspace(0, 4 * critical_damping, 100)
    gamma_range = damping_range / m

    # Quality factor Q
    Q = omega_0 * m / damping_range
    Q = np.clip(Q, 0, 20)

    ax4.plot(damping_range / critical_damping, Q, 'purple', lw=2)
    ax4.axvline(x=1.0, color='green', linestyle='--', lw=2, label='Critical damping')
    ax4.axvspan(0, 1, alpha=0.2, color='blue', label='Underdamped')
    ax4.axvspan(1, 4, alpha=0.2, color='red', label='Overdamped')

    # Mark the three cases
    for name, damping in damping_cases.items():
        ratio = damping / critical_damping
        Q_val = omega_0 * m / damping if damping > 0 else 0
        Q_val = min(Q_val, 20)
        ax4.plot(ratio, Q_val, 'ko', markersize=10)
        ax4.annotate(name, xy=(ratio, Q_val), xytext=(ratio + 0.1, Q_val + 1), fontsize=8)

    ax4.set_xlabel('Damping Ratio (b/b_critical)')
    ax4.set_ylabel('Quality Factor Q')
    ax4.set_title('Damping Regime Diagram')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 4)
    ax4.set_ylim(0, 15)

    plt.suptitle(f'Damped Harmonic Oscillator (ω₀ = {omega_0:.2f} rad/s, Critical damping = {critical_damping:.2f} kg/s)',
                 fontsize=12, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'damped_oscillator.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'damped_oscillator.png')}")


if __name__ == "__main__":
    main()
