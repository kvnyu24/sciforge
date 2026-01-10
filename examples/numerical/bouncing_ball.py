"""
Experiment 5: Event detection - Bouncing ball with coefficient of restitution.

Demonstrates event detection for discontinuous systems,
tracking impact times and energy loss at each bounce.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def simulate_bouncing_ball(y0, v0, g, e, t_max, dt=0.0001):
    """
    Simulate a bouncing ball with coefficient of restitution.

    Args:
        y0: Initial height (m)
        v0: Initial velocity (m/s, positive upward)
        g: Gravitational acceleration (m/s^2)
        e: Coefficient of restitution (0 < e <= 1)
        t_max: Maximum simulation time (s)
        dt: Time step for integration

    Returns:
        times, positions, velocities, impact_times, impact_velocities
    """
    y = y0
    v = v0
    t = 0

    times = [t]
    positions = [y]
    velocities = [v]
    impact_times = []
    impact_velocities = []

    while t < t_max:
        # Simple Euler step (exact for constant g)
        v_new = v - g * dt
        y_new = y + v * dt + 0.5 * (-g) * dt**2

        # Check for ground collision
        if y_new < 0 and v_new < 0:
            # Find exact collision time using quadratic formula
            # y + v*t_c - 0.5*g*t_c^2 = 0
            a = -0.5 * g
            b = v
            c = y
            discriminant = b**2 - 4*a*c

            if discriminant >= 0:
                t_c = (-b - np.sqrt(discriminant)) / (2*a)
                if t_c < 0:
                    t_c = (-b + np.sqrt(discriminant)) / (2*a)

                # Advance to collision point
                t_impact = t + t_c
                v_impact = v - g * t_c  # velocity just before impact

                impact_times.append(t_impact)
                impact_velocities.append(abs(v_impact))

                # Apply coefficient of restitution
                v_after = -e * v_impact

                # If velocity is too small, ball has stopped
                if abs(v_after) < 0.001:
                    # Fill rest with zeros
                    n_remaining = int((t_max - t_impact) / dt)
                    times.extend([t_impact + i*dt for i in range(n_remaining)])
                    positions.extend([0.0] * n_remaining)
                    velocities.extend([0.0] * n_remaining)
                    break

                # Record impact
                times.append(t_impact)
                positions.append(0.0)
                velocities.append(v_after)

                # Continue from impact with remaining time
                dt_remaining = dt - t_c
                y = 0.0
                v = v_after
                t = t_impact

                # Complete the step
                y = y + v * dt_remaining - 0.5 * g * dt_remaining**2
                v = v - g * dt_remaining
                t = t + dt_remaining
            else:
                y = y_new
                v = v_new
                t += dt
        else:
            y = y_new
            v = v_new
            t += dt

        times.append(t)
        positions.append(max(0, y))
        velocities.append(v)

    return (np.array(times), np.array(positions), np.array(velocities),
            np.array(impact_times), np.array(impact_velocities))


def analytical_bounce_times(y0, g, e, n_bounces):
    """Calculate analytical bounce times for ball dropped from rest."""
    times = []
    velocities = []

    # First drop: t1 = sqrt(2*y0/g), v1 = sqrt(2*g*y0)
    t1 = np.sqrt(2 * y0 / g)
    v1 = np.sqrt(2 * g * y0)

    times.append(t1)
    velocities.append(v1)

    t_current = t1
    v_current = v1

    for _ in range(n_bounces - 1):
        v_up = e * v_current
        dt = 2 * v_up / g  # Time up + down
        t_current += dt
        v_current = v_up
        times.append(t_current)
        velocities.append(v_current)

    return np.array(times), np.array(velocities)


def main():
    # Parameters
    y0 = 10.0  # Initial height (m)
    v0 = 0.0   # Dropped from rest
    g = 9.81   # Gravity (m/s^2)
    t_max = 10.0

    # Different coefficients of restitution
    e_values = [1.0, 0.9, 0.7, 0.5]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Height vs time for different e
    ax = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(e_values)))

    for e, color in zip(e_values, colors):
        t, y, v, t_imp, v_imp = simulate_bouncing_ball(y0, v0, g, e, t_max)
        ax.plot(t, y, '-', color=color, lw=1.5, label=f'e = {e}')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Height (m)')
    ax.set_title('Bouncing Ball: Height vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, t_max)
    ax.set_ylim(0, y0 * 1.1)

    # Plot 2: Energy dissipation
    ax = axes[0, 1]

    for e, color in zip(e_values, colors):
        t, y, v, t_imp, v_imp = simulate_bouncing_ball(y0, v0, g, e, t_max)
        E = 0.5 * v**2 + g * y  # Total mechanical energy (per unit mass)
        E0 = g * y0  # Initial energy
        ax.plot(t, E / E0, '-', color=color, lw=1.5, label=f'e = {e}')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('E / E₀')
    ax.set_title('Energy Dissipation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, t_max)
    ax.set_ylim(0, 1.1)

    # Plot 3: Impact velocity decay
    ax = axes[1, 0]

    for e, color in zip(e_values, colors):
        t, y, v, t_imp, v_imp = simulate_bouncing_ball(y0, v0, g, e, t_max)
        if len(t_imp) > 0:
            bounce_num = np.arange(1, len(t_imp) + 1)
            ax.semilogy(bounce_num, v_imp, 'o-', color=color, markersize=5,
                        label=f'e = {e}')

            # Analytical prediction: v_n = e^(n-1) * v_1
            v1 = np.sqrt(2 * g * y0)
            v_analytical = v1 * e ** (bounce_num - 1)
            ax.semilogy(bounce_num, v_analytical, '--', color=color, alpha=0.5)

    ax.set_xlabel('Bounce number')
    ax.set_ylabel('Impact velocity (m/s)')
    ax.set_title('Impact Velocity Decay (solid: sim, dashed: analytical)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Time between bounces
    ax = axes[1, 1]

    for e, color in zip(e_values, colors):
        t, y, v, t_imp, v_imp = simulate_bouncing_ball(y0, v0, g, e, t_max)
        if len(t_imp) > 1:
            dt_bounces = np.diff(t_imp)
            bounce_num = np.arange(1, len(dt_bounces) + 1)
            ax.semilogy(bounce_num, dt_bounces, 'o-', color=color, markersize=5,
                        label=f'e = {e}')

            # Analytical: dt_n = 2 * v_n / g = 2 * e^(n-1) * v_1 / g
            v1 = np.sqrt(2 * g * y0)
            dt_analytical = 2 * v1 * e ** bounce_num / g
            ax.semilogy(bounce_num, dt_analytical, '--', color=color, alpha=0.5)

    ax.set_xlabel('Interval number')
    ax.set_ylabel('Time between bounces (s)')
    ax.set_title('Time Between Bounces (geometric decay)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Bouncing Ball with Coefficient of Restitution\n' +
                 f'y₀ = {y0} m, g = {g} m/s²', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'bouncing_ball.png'), dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/bouncing_ball.png")

    # Print impact times for e=0.7
    print("\nImpact times for e=0.7:")
    t, y, v, t_imp, v_imp = simulate_bouncing_ball(y0, v0, g, 0.7, t_max)
    for i, (ti, vi) in enumerate(zip(t_imp[:10], v_imp[:10])):
        print(f"  Bounce {i+1}: t = {ti:.4f} s, v = {vi:.4f} m/s")


if __name__ == "__main__":
    main()
