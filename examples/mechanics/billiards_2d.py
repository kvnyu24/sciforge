"""
Experiment 41: 2D Billiards

This example demonstrates 2D elastic collisions on a billiard table.
Shows ball-ball collisions and ball-wall collisions with realistic physics.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import PatchCollection


class Ball:
    """Simple 2D ball for billiards simulation."""

    def __init__(self, pos, vel, radius=0.0286, mass=0.17):
        """
        Initialize a billiard ball.

        Args:
            pos: Initial position [x, y]
            vel: Initial velocity [vx, vy]
            radius: Ball radius (default: standard pool ball ~28.6mm)
            mass: Ball mass (default: standard pool ball ~170g)
        """
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.radius = radius
        self.mass = mass

    def update(self, dt):
        """Update ball position."""
        self.pos += self.vel * dt


def resolve_ball_collision(ball1, ball2, restitution=1.0):
    """
    Resolve elastic collision between two balls.

    Args:
        ball1, ball2: Ball objects
        restitution: Coefficient of restitution
    """
    # Vector from ball1 to ball2
    delta = ball2.pos - ball1.pos
    distance = np.linalg.norm(delta)

    if distance < ball1.radius + ball2.radius:
        # Normal vector (from ball1 to ball2)
        normal = delta / distance

        # Relative velocity
        rel_vel = ball1.vel - ball2.vel

        # Relative velocity along normal
        vel_along_normal = np.dot(rel_vel, normal)

        # Only resolve if balls are approaching
        if vel_along_normal > 0:
            # Impulse magnitude
            j = -(1 + restitution) * vel_along_normal
            j /= (1/ball1.mass + 1/ball2.mass)

            # Update velocities
            ball1.vel += (j / ball1.mass) * normal
            ball2.vel -= (j / ball2.mass) * normal

            # Separate balls to prevent overlap
            overlap = (ball1.radius + ball2.radius) - distance
            ball1.pos -= (overlap / 2) * normal
            ball2.pos += (overlap / 2) * normal


def resolve_wall_collision(ball, table_width, table_height, restitution=0.95):
    """
    Resolve collision between ball and walls.

    Args:
        ball: Ball object
        table_width, table_height: Table dimensions
        restitution: Coefficient of restitution for wall bounce
    """
    # Left wall
    if ball.pos[0] - ball.radius < 0:
        ball.pos[0] = ball.radius
        ball.vel[0] = -restitution * ball.vel[0]

    # Right wall
    if ball.pos[0] + ball.radius > table_width:
        ball.pos[0] = table_width - ball.radius
        ball.vel[0] = -restitution * ball.vel[0]

    # Bottom wall
    if ball.pos[1] - ball.radius < 0:
        ball.pos[1] = ball.radius
        ball.vel[1] = -restitution * ball.vel[1]

    # Top wall
    if ball.pos[1] + ball.radius > table_height:
        ball.pos[1] = table_height - ball.radius
        ball.vel[1] = -restitution * ball.vel[1]


def simulate_billiards(balls, table_width, table_height, t_final, dt):
    """
    Simulate billiards motion.

    Args:
        balls: List of Ball objects
        table_width, table_height: Table dimensions
        t_final: Simulation duration
        dt: Time step

    Returns:
        Dictionary with trajectory data
    """
    n_balls = len(balls)

    times = [0]
    positions = [[ball.pos.copy() for ball in balls]]
    velocities = [[ball.vel.copy() for ball in balls]]

    t = 0
    while t < t_final:
        # Update positions
        for ball in balls:
            ball.update(dt)

        # Check ball-ball collisions
        for i in range(n_balls):
            for j in range(i + 1, n_balls):
                resolve_ball_collision(balls[i], balls[j])

        # Check wall collisions
        for ball in balls:
            resolve_wall_collision(ball, table_width, table_height)

        t += dt
        times.append(t)
        positions.append([ball.pos.copy() for ball in balls])
        velocities.append([ball.vel.copy() for ball in balls])

    return {
        'time': np.array(times),
        'positions': np.array(positions),
        'velocities': np.array(velocities)
    }


def main():
    # Table dimensions (standard pool table: 9ft x 4.5ft ≈ 2.74m x 1.37m)
    table_width = 2.74
    table_height = 1.37
    ball_radius = 0.0286  # ~28.6mm

    # Create figure
    fig = plt.figure(figsize=(16, 12))

    # Case 1: Simple two-ball collision
    ax1 = fig.add_subplot(2, 3, 1)

    balls_case1 = [
        Ball([0.5, 0.7], [1.0, 0.0], ball_radius),
        Ball([1.5, 0.7], [0.0, 0.0], ball_radius)
    ]

    results1 = simulate_billiards(balls_case1, table_width, table_height, 3.0, 0.001)

    # Plot trajectories
    ax1.plot(results1['positions'][:, 0, 0], results1['positions'][:, 0, 1],
             'b-', lw=1.5, label='Cue ball')
    ax1.plot(results1['positions'][:, 1, 0], results1['positions'][:, 1, 1],
             'r-', lw=1.5, label='Target ball')

    # Plot initial and final positions
    ax1.plot(results1['positions'][0, 0, 0], results1['positions'][0, 0, 1],
             'bo', markersize=12)
    ax1.plot(results1['positions'][0, 1, 0], results1['positions'][0, 1, 1],
             'ro', markersize=12)

    ax1.set_xlim(0, table_width)
    ax1.set_ylim(0, table_height)
    ax1.set_aspect('equal')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_title('Head-on Collision')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Case 2: Angled collision
    ax2 = fig.add_subplot(2, 3, 2)

    balls_case2 = [
        Ball([0.3, 0.5], [2.0, 0.5], ball_radius),
        Ball([1.5, 0.8], [0.0, 0.0], ball_radius)
    ]

    results2 = simulate_billiards(balls_case2, table_width, table_height, 3.0, 0.001)

    ax2.plot(results2['positions'][:, 0, 0], results2['positions'][:, 0, 1],
             'b-', lw=1.5, label='Cue ball')
    ax2.plot(results2['positions'][:, 1, 0], results2['positions'][:, 1, 1],
             'r-', lw=1.5, label='Target ball')

    ax2.plot(results2['positions'][0, 0, 0], results2['positions'][0, 0, 1],
             'bo', markersize=12)
    ax2.plot(results2['positions'][0, 1, 0], results2['positions'][0, 1, 1],
             'ro', markersize=12)

    ax2.set_xlim(0, table_width)
    ax2.set_ylim(0, table_height)
    ax2.set_aspect('equal')
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')
    ax2.set_title('Angled Collision')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Case 3: Break shot (multiple balls)
    ax3 = fig.add_subplot(2, 3, 3)

    # Triangle rack setup
    rack_x = 2.0
    rack_y = table_height / 2
    spacing = 2.1 * ball_radius

    balls_case3 = [
        Ball([0.5, rack_y], [3.0, 0.0], ball_radius)  # Cue ball
    ]

    # Create triangle rack
    row_positions = [
        [(rack_x, rack_y)],
        [(rack_x + spacing * 0.866, rack_y - spacing/2),
         (rack_x + spacing * 0.866, rack_y + spacing/2)],
        [(rack_x + 2*spacing * 0.866, rack_y - spacing),
         (rack_x + 2*spacing * 0.866, rack_y),
         (rack_x + 2*spacing * 0.866, rack_y + spacing)],
    ]

    for row in row_positions:
        for pos in row:
            balls_case3.append(Ball(list(pos), [0.0, 0.0], ball_radius))

    results3 = simulate_billiards(balls_case3, table_width, table_height, 5.0, 0.0005)

    # Plot all trajectories
    colors = plt.cm.tab10(np.linspace(0, 1, len(balls_case3)))
    for i, (color) in enumerate(colors):
        ax3.plot(results3['positions'][:, i, 0], results3['positions'][:, i, 1],
                 color=color, lw=0.5, alpha=0.7)

    # Draw table boundary
    ax3.add_patch(Rectangle((0, 0), table_width, table_height,
                             fill=False, edgecolor='brown', lw=2))

    ax3.set_xlim(-0.1, table_width + 0.1)
    ax3.set_ylim(-0.1, table_height + 0.1)
    ax3.set_aspect('equal')
    ax3.set_xlabel('x (m)')
    ax3.set_ylabel('y (m)')
    ax3.set_title('Break Shot')
    ax3.grid(True, alpha=0.3)

    # Case 4: Kinetic energy conservation
    ax4 = fig.add_subplot(2, 3, 4)

    # Calculate total KE for case 3
    n_frames = len(results3['time'])
    KE_total = []
    for frame in range(n_frames):
        ke = 0
        for i in range(len(balls_case3)):
            v = results3['velocities'][frame, i]
            ke += 0.5 * balls_case3[i].mass * np.dot(v, v)
        KE_total.append(ke)

    ax4.plot(results3['time'], KE_total, 'b-', lw=2)
    ax4.axhline(y=KE_total[0], color='r', linestyle='--', alpha=0.5,
                label='Initial KE')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Total Kinetic Energy (J)')
    ax4.set_title('Energy Conservation (Break Shot)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Case 5: Cushion (wall) bounces
    ax5 = fig.add_subplot(2, 3, 5)

    balls_case5 = [
        Ball([0.3, 0.3], [2.0, 1.5], ball_radius)
    ]

    results5 = simulate_billiards(balls_case5, table_width, table_height, 8.0, 0.001)

    ax5.plot(results5['positions'][:, 0, 0], results5['positions'][:, 0, 1],
             'b-', lw=1, alpha=0.7)

    # Mark bounce points (where velocity changes sign)
    for i in range(1, len(results5['time'])):
        if (results5['velocities'][i, 0, 0] * results5['velocities'][i-1, 0, 0] < 0 or
            results5['velocities'][i, 0, 1] * results5['velocities'][i-1, 0, 1] < 0):
            ax5.plot(results5['positions'][i, 0, 0], results5['positions'][i, 0, 1],
                     'ro', markersize=6)

    ax5.add_patch(Rectangle((0, 0), table_width, table_height,
                             fill=False, edgecolor='brown', lw=2))

    ax5.set_xlim(-0.1, table_width + 0.1)
    ax5.set_ylim(-0.1, table_height + 0.1)
    ax5.set_aspect('equal')
    ax5.set_xlabel('x (m)')
    ax5.set_ylabel('y (m)')
    ax5.set_title('Multiple Wall Bounces (red = cushion hit)')
    ax5.grid(True, alpha=0.3)

    # Case 6: Collision angle analysis
    ax6 = fig.add_subplot(2, 3, 6)

    # For collision between moving ball and stationary ball,
    # the angle between final velocity vectors should be 90 degrees (elastic)
    impact_angles = np.linspace(5, 85, 17)  # Degrees
    final_angle_differences = []

    for impact_deg in impact_angles:
        impact_rad = np.radians(impact_deg)

        # Cue ball comes in at angle, target at origin
        cue_start = np.array([0.0, 0.5])
        target_pos = np.array([1.0, 0.5])

        # Velocity direction toward target, but offset by impact angle
        offset = 0.05 * np.tan(impact_rad)
        target_pos[1] += offset

        velocity_dir = target_pos - cue_start
        velocity_dir = velocity_dir / np.linalg.norm(velocity_dir)

        balls_angle = [
            Ball(cue_start, 2.0 * velocity_dir, ball_radius),
            Ball([1.0, 0.5], [0.0, 0.0], ball_radius)
        ]

        results_angle = simulate_billiards(balls_angle, table_width, table_height, 2.0, 0.0005)

        # Get velocities after collision (when target starts moving)
        for i in range(1, len(results_angle['time'])):
            if np.linalg.norm(results_angle['velocities'][i, 1]) > 0.1:
                v1_after = results_angle['velocities'][i + 10, 0]
                v2_after = results_angle['velocities'][i + 10, 1]

                # Calculate angle between velocity vectors
                if np.linalg.norm(v1_after) > 0.01 and np.linalg.norm(v2_after) > 0.01:
                    cos_angle = np.dot(v1_after, v2_after) / (np.linalg.norm(v1_after) * np.linalg.norm(v2_after))
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle = np.degrees(np.arccos(cos_angle))
                    final_angle_differences.append(angle)
                else:
                    final_angle_differences.append(90)  # Head-on collision
                break
        else:
            final_angle_differences.append(np.nan)

    ax6.plot(impact_angles, final_angle_differences, 'bo-', lw=2, markersize=8)
    ax6.axhline(y=90, color='r', linestyle='--', lw=2, label='Theoretical (90 deg)')
    ax6.set_xlabel('Impact Parameter Angle (degrees)')
    ax6.set_ylabel('Angle Between Final Velocities (degrees)')
    ax6.set_title('90-Degree Rule for Elastic Collisions')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(80, 100)

    plt.suptitle('2D Billiards Simulation\n'
                 f'Table: {table_width:.2f}m x {table_height:.2f}m, Ball radius: {ball_radius*1000:.1f}mm',
                 fontsize=12, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'billiards_2d.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'billiards_2d.png')}")


if __name__ == "__main__":
    main()
