"""
Example demonstrating projectile motion with air resistance.

This example compares the ideal parabolic trajectory (no air resistance)
with realistic trajectories including quadratic drag, showing how air
resistance affects range, maximum height, and flight time.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics import Particle


def simulate_projectile(v0, angle, drag_coeff=0.0, dt=0.001, mass=1.0):
    """
    Simulate projectile motion with optional drag.

    Args:
        v0: Initial velocity magnitude (m/s)
        angle: Launch angle (degrees)
        drag_coeff: Drag coefficient (0 for ideal)
        dt: Time step (s)
        mass: Projectile mass (kg)

    Returns:
        Dictionary with trajectory data
    """
    angle_rad = np.radians(angle)
    vx0 = v0 * np.cos(angle_rad)
    vy0 = v0 * np.sin(angle_rad)

    particle = Particle(
        mass=mass,
        position=np.array([0.0, 0.0, 0.0]),
        velocity=np.array([vx0, 0.0, vy0]),
        drag_coeff=drag_coeff,
        gravity=9.81
    )

    positions = [particle.position.copy()]
    velocities = [particle.velocity.copy()]
    times = [0.0]

    t = 0.0
    while particle.position[2] >= 0 or t < dt:
        # No external force needed - gravity and drag are handled internally
        external_force = np.array([0.0, 0.0, 0.0])
        particle.update(external_force, dt)

        positions.append(particle.position.copy())
        velocities.append(particle.velocity.copy())
        t += dt
        times.append(t)

        # Stop if projectile hits ground after rising
        if particle.position[2] < 0 and len(positions) > 10:
            break

        # Safety limit
        if t > 100:
            break

    return {
        'time': np.array(times),
        'position': np.array(positions),
        'velocity': np.array(velocities)
    }


def analytical_trajectory(v0, angle, t_array):
    """Calculate ideal (no drag) trajectory analytically."""
    angle_rad = np.radians(angle)
    vx0 = v0 * np.cos(angle_rad)
    vy0 = v0 * np.sin(angle_rad)
    g = 9.81

    x = vx0 * t_array
    z = vy0 * t_array - 0.5 * g * t_array**2

    return x, z


def plot_results(trajectories, v0, angles):
    """Create comprehensive visualization of projectile motion."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = plt.cm.viridis(np.linspace(0, 1, len(angles)))

    # Plot 1: Trajectories comparison
    ax1 = axes[0, 0]
    for i, angle in enumerate(angles):
        # With drag
        traj = trajectories[f'drag_{angle}']
        ax1.plot(traj['position'][:, 0], traj['position'][:, 2],
                 '-', color=colors[i], lw=2, label=f'{angle}° (with drag)')

        # Ideal (no drag)
        traj_ideal = trajectories[f'ideal_{angle}']
        ax1.plot(traj_ideal['position'][:, 0], traj_ideal['position'][:, 2],
                 '--', color=colors[i], lw=1, alpha=0.7)

    ax1.set_xlabel('Horizontal Distance (m)')
    ax1.set_ylabel('Height (m)')
    ax1.set_title(f'Projectile Trajectories (v₀ = {v0} m/s)\nSolid: with drag, Dashed: ideal')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # Plot 2: Range comparison
    ax2 = axes[0, 1]
    ranges_drag = []
    ranges_ideal = []
    for angle in angles:
        traj_drag = trajectories[f'drag_{angle}']
        traj_ideal = trajectories[f'ideal_{angle}']
        ranges_drag.append(traj_drag['position'][-1, 0])
        ranges_ideal.append(traj_ideal['position'][-1, 0])

    x_pos = np.arange(len(angles))
    width = 0.35
    ax2.bar(x_pos - width/2, ranges_ideal, width, label='Ideal (no drag)', alpha=0.7)
    ax2.bar(x_pos + width/2, ranges_drag, width, label='With drag', alpha=0.7)
    ax2.set_xlabel('Launch Angle (degrees)')
    ax2.set_ylabel('Range (m)')
    ax2.set_title('Range Comparison')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(angles)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Velocity magnitude vs time
    ax3 = axes[1, 0]
    angle = 45  # Show for 45 degrees
    traj_drag = trajectories[f'drag_{angle}']
    traj_ideal = trajectories[f'ideal_{angle}']

    v_drag = np.linalg.norm(traj_drag['velocity'], axis=1)
    v_ideal = np.linalg.norm(traj_ideal['velocity'], axis=1)

    ax3.plot(traj_drag['time'], v_drag, 'b-', lw=2, label='With drag')
    ax3.plot(traj_ideal['time'], v_ideal, 'r--', lw=2, label='Ideal')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Speed (m/s)')
    ax3.set_title(f'Speed vs Time (45° launch)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Maximum height comparison
    ax4 = axes[1, 1]
    heights_drag = []
    heights_ideal = []
    for angle in angles:
        traj_drag = trajectories[f'drag_{angle}']
        traj_ideal = trajectories[f'ideal_{angle}']
        heights_drag.append(np.max(traj_drag['position'][:, 2]))
        heights_ideal.append(np.max(traj_ideal['position'][:, 2]))

    ax4.bar(x_pos - width/2, heights_ideal, width, label='Ideal', alpha=0.7)
    ax4.bar(x_pos + width/2, heights_drag, width, label='With drag', alpha=0.7)
    ax4.set_xlabel('Launch Angle (degrees)')
    ax4.set_ylabel('Maximum Height (m)')
    ax4.set_title('Maximum Height Comparison')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(angles)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Projectile Motion: Effect of Air Resistance', fontsize=14, y=1.02)
    plt.tight_layout()


def main():
    # Simulation parameters
    v0 = 50.0  # Initial velocity (m/s)
    angles = [15, 30, 45, 60, 75]  # Launch angles
    drag_coeff = 0.1  # Drag coefficient

    trajectories = {}

    # Run simulations for each angle
    for angle in angles:
        # With drag
        trajectories[f'drag_{angle}'] = simulate_projectile(
            v0, angle, drag_coeff=drag_coeff
        )
        # Without drag (ideal)
        trajectories[f'ideal_{angle}'] = simulate_projectile(
            v0, angle, drag_coeff=0.0
        )

    # Plot results
    plot_results(trajectories, v0, angles)

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'projectile_drag.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'projectile_drag.png')}")


if __name__ == "__main__":
    main()
