"""
Example comparing simple pendulum with small angle approximation.

This example demonstrates the difference between exact pendulum motion
and the small angle (harmonic) approximation, showing when the
linearization breaks down.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.mechanics import Pendulum


def simulate_pendulum(length, initial_angle, t_final, dt):
    """
    Simulate pendulum motion.

    Args:
        length: Pendulum length (m)
        initial_angle: Initial angle from vertical (radians)
        t_final: Simulation duration (s)
        dt: Time step (s)

    Returns:
        Dictionary with time, angle, and angular velocity data
    """
    pendulum = Pendulum(
        mass=1.0,
        length=length,
        theta0=initial_angle,
        omega0=0.0
    )

    times = [0]
    angles = [initial_angle]
    angular_velocities = [0.0]

    t = 0
    while t < t_final:
        pendulum.update(dt)
        t += dt
        times.append(t)
        angles.append(pendulum.theta)
        angular_velocities.append(pendulum.omega)

    return {
        'time': np.array(times),
        'angle': np.array(angles),
        'omega': np.array(angular_velocities)
    }


def small_angle_solution(initial_angle, length, t, g=9.81):
    """Analytical solution for small angle approximation."""
    omega_0 = np.sqrt(g / length)
    return initial_angle * np.cos(omega_0 * t)


def main():
    # Parameters
    length = 1.0  # m
    g = 9.81  # m/s^2
    t_final = 10.0
    dt = 0.001

    # Different initial angles
    angles_deg = [5, 30, 90, 150]
    angles_rad = [np.radians(a) for a in angles_deg]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (angle_deg, angle_rad) in enumerate(zip(angles_deg, angles_rad)):
        # Simulate exact pendulum
        results = simulate_pendulum(length, angle_rad, t_final, dt)

        # Calculate small angle approximation
        small_angle = small_angle_solution(angle_rad, length, results['time'])

        # Plot comparison
        ax = axes[idx]
        ax.plot(results['time'], np.degrees(results['angle']), 'b-',
                lw=2, label='Exact')
        ax.plot(results['time'], np.degrees(small_angle), 'r--',
                lw=2, alpha=0.7, label='Small angle approx.')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (degrees)')
        ax.set_title(f'Initial angle: {angle_deg}°')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Calculate period comparison
        omega_0 = np.sqrt(g / length)
        T_approx = 2 * np.pi / omega_0

        # Find actual period by looking for zero crossings
        angles_array = results['angle']
        crossings = []
        for i in range(1, len(angles_array)):
            if angles_array[i-1] > 0 and angles_array[i] <= 0:
                crossings.append(results['time'][i])
        if len(crossings) >= 2:
            T_actual = 2 * (crossings[1] - crossings[0])
        else:
            T_actual = T_approx

        # Add period info
        ax.text(0.02, 0.98, f'T_approx = {T_approx:.3f} s\nT_actual = {T_actual:.3f} s\nError = {100*(T_actual-T_approx)/T_approx:.1f}%',
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Simple Pendulum: Exact vs Small Angle Approximation\n(L = 1 m, g = 9.81 m/s²)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Create period vs amplitude plot
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    angles_sweep = np.linspace(1, 170, 50)
    periods = []
    T_approx = 2 * np.pi / np.sqrt(g / length)

    for angle_deg in angles_sweep:
        results = simulate_pendulum(length, np.radians(angle_deg), 20.0, 0.001)
        angles_array = results['angle']

        # Find period from zero crossings
        crossings = []
        for i in range(1, len(angles_array)):
            if angles_array[i-1] > 0 and angles_array[i] <= 0:
                crossings.append(results['time'][i])
        if len(crossings) >= 2:
            T = 2 * (crossings[1] - crossings[0])
        else:
            T = T_approx
        periods.append(T)

    ax2.plot(angles_sweep, periods, 'b-', lw=2, label='Exact period')
    ax2.axhline(y=T_approx, color='r', linestyle='--', lw=2,
                label=f'Small angle: T = {T_approx:.3f} s')
    ax2.set_xlabel('Initial Amplitude (degrees)')
    ax2.set_ylabel('Period (s)')
    ax2.set_title('Pendulum Period vs Amplitude')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Save plots
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'pendulum_comparison.png'), dpi=150, bbox_inches='tight')
    fig2.savefig(os.path.join(output_dir, 'pendulum_period_vs_amplitude.png'), dpi=150, bbox_inches='tight')
    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
