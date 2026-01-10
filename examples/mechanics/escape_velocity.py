"""
Experiment 37: Escape Velocity in Newtonian Gravity

This example demonstrates the concept of escape velocity - the minimum
initial velocity required for an object to escape a gravitational field.
Shows trajectories for sub-escape, escape, and super-escape velocities.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def simulate_radial_trajectory(r0, v0, M=1.0, G=1.0, t_final=50.0, dt=0.001):
    """
    Simulate purely radial motion in a gravitational field.

    Args:
        r0: Initial radial distance from center
        v0: Initial radial velocity (positive = outward)
        M: Central mass
        G: Gravitational constant
        t_final: Simulation duration
        dt: Time step

    Returns:
        Dictionary with trajectory data
    """
    r = r0
    v = v0

    times = [0]
    rs = [r]
    vs = [v]

    # Energy
    E = 0.5 * v**2 - G * M / r
    energies = [E]

    t = 0
    while t < t_final and r > 0.01:
        # Gravitational acceleration (toward center)
        a = -G * M / r**2

        # RK4 integration
        k1_r = v
        k1_v = a

        r_temp = r + 0.5 * dt * k1_r
        if r_temp > 0:
            a_temp = -G * M / r_temp**2
        else:
            break
        k2_r = v + 0.5 * dt * k1_v
        k2_v = a_temp

        r_temp = r + 0.5 * dt * k2_r
        if r_temp > 0:
            a_temp = -G * M / r_temp**2
        else:
            break
        k3_r = v + 0.5 * dt * k2_v
        k3_v = a_temp

        r_temp = r + dt * k3_r
        if r_temp > 0:
            a_temp = -G * M / r_temp**2
        else:
            break
        k4_r = v + dt * k3_v
        k4_v = a_temp

        r_new = r + (dt / 6) * (k1_r + 2*k2_r + 2*k3_r + k4_r)
        v_new = v + (dt / 6) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

        if r_new <= 0.01:
            break

        r = r_new
        v = v_new
        t += dt

        times.append(t)
        rs.append(r)
        vs.append(v)

        E = 0.5 * v**2 - G * M / r
        energies.append(E)

    return {
        'time': np.array(times),
        'r': np.array(rs),
        'v': np.array(vs),
        'energy': np.array(energies)
    }


def escape_velocity(r, M=1.0, G=1.0):
    """Calculate escape velocity at distance r from mass M."""
    return np.sqrt(2 * G * M / r)


def main():
    # Parameters
    G = 1.0  # Gravitational constant
    M = 1.0  # Central mass
    r0 = 1.0  # Starting distance

    # Calculate escape velocity at r0
    v_esc = escape_velocity(r0, M, G)

    # Create figure
    fig = plt.figure(figsize=(16, 12))

    # Subplot 1: Trajectories for different initial velocities
    ax1 = fig.add_subplot(2, 3, 1)

    # Test different fractions of escape velocity
    v_fractions = [0.5, 0.8, 0.95, 1.0, 1.05, 1.2, 1.5]
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(v_fractions)))

    results_list = []
    for v_frac, color in zip(v_fractions, colors):
        v0 = v_frac * v_esc
        results = simulate_radial_trajectory(r0, v0, M, G, t_final=50.0, dt=0.001)
        results['v_frac'] = v_frac
        results_list.append(results)
        ax1.plot(results['time'], results['r'], color=color, lw=2,
                 label=f'v = {v_frac:.2f} v_esc')

    ax1.axhline(y=r0, color='k', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Radial distance r')
    ax1.set_title(f'Trajectories (v_esc = {v_esc:.3f} at r = {r0})')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 15)

    # Subplot 2: Energy vs velocity fraction
    ax2 = fig.add_subplot(2, 3, 2)

    v_frac_range = np.linspace(0.1, 2.0, 100)
    v_range = v_frac_range * v_esc
    E_range = 0.5 * v_range**2 - G * M / r0

    ax2.plot(v_frac_range, E_range, 'b-', lw=2)
    ax2.axhline(y=0, color='r', linestyle='--', lw=2, label='E = 0 (escape threshold)')
    ax2.axvline(x=1.0, color='g', linestyle='--', lw=2, label='v = v_esc')
    ax2.fill_between(v_frac_range, E_range, 0, where=(E_range < 0),
                     alpha=0.3, color='blue', label='Bound (E < 0)')
    ax2.fill_between(v_frac_range, E_range, 0, where=(E_range > 0),
                     alpha=0.3, color='red', label='Unbound (E > 0)')
    ax2.set_xlabel('v / v_esc')
    ax2.set_ylabel('Total Energy E')
    ax2.set_title('Energy as Function of Initial Velocity')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Velocity vs position
    ax3 = fig.add_subplot(2, 3, 3)

    for results, color in zip(results_list, colors):
        ax3.plot(results['r'], results['v'], color=color, lw=2,
                 label=f'v = {results["v_frac"]:.2f} v_esc')

    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax3.set_xlabel('Radial distance r')
    ax3.set_ylabel('Radial velocity v')
    ax3.set_title('Phase Space (v vs r)')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 10)

    # Subplot 4: Escape velocity vs distance
    ax4 = fig.add_subplot(2, 3, 4)

    r_range = np.linspace(0.2, 5, 100)
    v_esc_range = escape_velocity(r_range, M, G)

    ax4.plot(r_range, v_esc_range, 'b-', lw=2, label='v_esc(r)')
    ax4.fill_between(r_range, 0, v_esc_range, alpha=0.3, color='blue', label='Bound region')
    ax4.fill_between(r_range, v_esc_range, v_esc_range.max() * 1.2, alpha=0.3,
                     color='red', label='Escape region')
    ax4.set_xlabel('Distance from center r')
    ax4.set_ylabel('Escape velocity')
    ax4.set_title('Escape Velocity vs Distance: v_esc = sqrt(2GM/r)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Subplot 5: Energy conservation
    ax5 = fig.add_subplot(2, 3, 5)

    for results, color in zip(results_list, colors):
        E_normalized = results['energy'] / results['energy'][0] if results['energy'][0] != 0 else results['energy']
        ax5.plot(results['time'], E_normalized, color=color, lw=1.5,
                 label=f'v = {results["v_frac"]:.2f} v_esc')

    ax5.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Time')
    ax5.set_ylabel('E(t) / E(0)')
    ax5.set_title('Energy Conservation Check')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # Subplot 6: Maximum height reached (for bound trajectories)
    ax6 = fig.add_subplot(2, 3, 6)

    v_frac_test = np.linspace(0.1, 0.99, 50)
    max_heights = []

    for v_frac in v_frac_test:
        v0 = v_frac * v_esc
        # For bound trajectories, max height from energy conservation:
        # E = 0.5*v0^2 - GM/r0 = -GM/r_max
        # r_max = r0 / (1 - v0^2*r0/(2GM)) = r0 / (1 - (v/v_esc)^2)
        E = 0.5 * v0**2 - G * M / r0
        if E < 0:
            r_max = -G * M / E
        else:
            r_max = np.inf
        max_heights.append(r_max)

    ax6.plot(v_frac_test, max_heights, 'b-', lw=2)
    ax6.axhline(y=r0, color='g', linestyle='--', label='Initial distance')
    ax6.set_xlabel('v / v_esc')
    ax6.set_ylabel('Maximum height r_max')
    ax6.set_title('Maximum Height for Bound Trajectories')
    ax6.set_ylim(0, 20)
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.suptitle('Escape Velocity in Newtonian Gravity\n'
                 'v_esc = sqrt(2GM/r), E = (1/2)mv^2 - GMm/r',
                 fontsize=12, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'escape_velocity.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'escape_velocity.png')}")


if __name__ == "__main__":
    main()
