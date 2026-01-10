"""
Example demonstrating rolling without slipping dynamics.

Rolling without slipping is a constraint that couples translational and
rotational motion. For a wheel of radius R:
    v_cm = omega * R

Key physics concepts:
1. The point of contact is instantaneously at rest
2. Friction provides the torque but does no work
3. Energy is distributed between translation and rotation
4. Different shapes (disk, sphere, hoop) have different effective inertia

This example demonstrates:
- Rolling down an incline (race of different shapes)
- The role of friction in enabling rolling
- Energy partition between translation and rotation
- Comparison with sliding motion
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


class RollingBody:
    """
    A rigid body that can roll without slipping.

    The moment of inertia is expressed as I = c * m * R^2 where c depends
    on the shape:
    - Solid sphere: c = 2/5
    - Solid cylinder/disk: c = 1/2
    - Hollow sphere: c = 2/3
    - Hoop/ring: c = 1
    """

    def __init__(self, mass, radius, inertia_coeff, name):
        self.mass = mass
        self.radius = radius
        self.c = inertia_coeff  # I = c * m * R^2
        self.I = inertia_coeff * mass * radius**2
        self.name = name

        # State
        self.x = 0.0  # Position along incline
        self.v = 0.0  # Velocity (of CM)
        self.theta = 0.0  # Rotation angle
        self.omega = 0.0  # Angular velocity

        self._history = {
            'time': [0.0],
            'x': [0.0],
            'v': [0.0],
            'theta': [0.0],
            'omega': [0.0]
        }

    def effective_mass_ratio(self):
        """
        Calculate ratio of translational acceleration to point mass.

        For rolling: a = g*sin(theta) / (1 + I/(mR^2)) = g*sin(theta) / (1 + c)
        """
        return 1.0 / (1.0 + self.c)

    def update_rolling(self, incline_angle, dt, g=9.81):
        """
        Update state for rolling without slipping down an incline.

        Constraint: v = omega * R
        Equation of motion: a = g*sin(theta) / (1 + c)
        """
        # Acceleration down the incline
        a = g * np.sin(incline_angle) / (1.0 + self.c)

        # Update velocity and position
        self.v += a * dt
        self.x += self.v * dt

        # Rolling constraint
        self.omega = self.v / self.radius
        self.theta += self.omega * dt

        # Store history
        self._history['time'].append(self._history['time'][-1] + dt)
        self._history['x'].append(self.x)
        self._history['v'].append(self.v)
        self._history['theta'].append(self.theta)
        self._history['omega'].append(self.omega)

    def kinetic_energy(self):
        """Calculate total kinetic energy (translational + rotational)."""
        KE_trans = 0.5 * self.mass * self.v**2
        KE_rot = 0.5 * self.I * self.omega**2
        return KE_trans, KE_rot

    def reset(self):
        """Reset to initial state."""
        self.x = 0.0
        self.v = 0.0
        self.theta = 0.0
        self.omega = 0.0
        self._history = {
            'time': [0.0],
            'x': [0.0],
            'v': [0.0],
            'theta': [0.0],
            'omega': [0.0]
        }


def simulate_sliding(mass, incline_angle, friction_coeff, t_final, dt, g=9.81):
    """
    Simulate a block sliding down an incline with friction.

    Args:
        mass: Mass of block
        incline_angle: Angle of incline (radians)
        friction_coeff: Kinetic friction coefficient
        t_final: Simulation duration
        dt: Time step
        g: Gravitational acceleration

    Returns:
        Dictionary with trajectory data
    """
    x = 0.0
    v = 0.0

    times = [0.0]
    positions = [0.0]
    velocities = [0.0]

    # Acceleration: a = g*(sin(theta) - mu*cos(theta))
    a = g * (np.sin(incline_angle) - friction_coeff * np.cos(incline_angle))
    if a < 0:
        a = 0  # Object doesn't slide if friction is too high

    t = 0
    while t < t_final:
        v += a * dt
        x += v * dt

        t += dt
        times.append(t)
        positions.append(x)
        velocities.append(v)

    return {
        'time': np.array(times),
        'x': np.array(positions),
        'v': np.array(velocities)
    }


def minimum_friction_for_rolling(inertia_coeff, incline_angle):
    """
    Calculate minimum friction coefficient required for pure rolling.

    For rolling without slipping:
    mu_min = tan(theta) / (1 + 1/c)

    where c = I / (mR^2)
    """
    c = inertia_coeff
    return np.tan(incline_angle) / (1.0 + 1.0/c)


def main():
    # Common parameters
    mass = 1.0  # kg
    radius = 0.1  # m
    g = 9.81
    incline_angle = np.radians(30)
    t_final = 2.0
    dt = 0.001

    # Create different rolling bodies
    bodies = [
        RollingBody(mass, radius, 1.0, 'Hoop (c=1)'),
        RollingBody(mass, radius, 2/3, 'Hollow Sphere (c=2/3)'),
        RollingBody(mass, radius, 0.5, 'Solid Cylinder (c=1/2)'),
        RollingBody(mass, radius, 0.4, 'Solid Sphere (c=2/5)'),
    ]

    fig = plt.figure(figsize=(18, 12))

    # --- Race down incline ---
    ax1 = fig.add_subplot(2, 3, 1)

    colors = ['red', 'orange', 'blue', 'green']

    for body, color in zip(bodies, colors):
        body.reset()
        while body._history['time'][-1] < t_final:
            body.update_rolling(incline_angle, dt, g)

        times = np.array(body._history['time'])
        positions = np.array(body._history['x'])

        ax1.plot(times, positions, color=color, lw=2, label=body.name)

    # Add sliding block for comparison (frictionless)
    slide_result = simulate_sliding(mass, incline_angle, 0.0, t_final, dt, g)
    ax1.plot(slide_result['time'], slide_result['x'], 'k--', lw=2,
             label='Sliding (frictionless)')

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Distance down incline (m)')
    ax1.set_title(f'Race Down Incline (theta = {np.degrees(incline_angle):.0f} deg)\n'
                  'Solid sphere wins! (lowest moment of inertia)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Velocity comparison ---
    ax2 = fig.add_subplot(2, 3, 2)

    for body, color in zip(bodies, colors):
        times = np.array(body._history['time'])
        velocities = np.array(body._history['v'])

        ax2.plot(times, velocities, color=color, lw=2, label=body.name)

    ax2.plot(slide_result['time'], slide_result['v'], 'k--', lw=2,
             label='Sliding (frictionless)')

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('Velocity Down Incline')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- Energy partition ---
    ax3 = fig.add_subplot(2, 3, 3)

    # Use solid cylinder as example
    body = bodies[2]  # Solid cylinder
    body.reset()

    times = [0]
    KE_trans = [0]
    KE_rot = [0]
    PE = [mass * g * 0]  # Initial height = 0 (relative)

    while body._history['time'][-1] < t_final:
        body.update_rolling(incline_angle, dt, g)

        times.append(body._history['time'][-1])
        KE_t, KE_r = body.kinetic_energy()
        KE_trans.append(KE_t)
        KE_rot.append(KE_r)

        # Height decreases as x increases
        h = -body.x * np.sin(incline_angle)
        PE.append(mass * g * h)

    times = np.array(times)
    KE_trans = np.array(KE_trans)
    KE_rot = np.array(KE_rot)
    PE = np.array(PE)
    total = KE_trans + KE_rot + PE

    ax3.fill_between(times, 0, KE_trans, alpha=0.5, label='KE translational')
    ax3.fill_between(times, KE_trans, KE_trans + KE_rot, alpha=0.5, label='KE rotational')
    ax3.plot(times, -PE, 'g-', lw=2, label='Energy from gravity')
    ax3.plot(times, KE_trans + KE_rot, 'k--', lw=2, label='Total KE')

    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Energy (J)')
    ax3.set_title(f'Energy Partition (Solid Cylinder, c=0.5)\n'
                  f'KE_trans = {1/(1+body.c)*100:.0f}%, KE_rot = {body.c/(1+body.c)*100:.0f}%')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # --- Effective acceleration comparison ---
    ax4 = fig.add_subplot(2, 3, 4)

    c_values = np.linspace(0, 1.5, 100)
    eff_acc_ratio = 1.0 / (1.0 + c_values)

    ax4.plot(c_values, eff_acc_ratio, 'b-', lw=2)

    # Mark the standard shapes
    shapes = [
        (0.4, 'Solid Sphere'),
        (0.5, 'Solid Cylinder'),
        (2/3, 'Hollow Sphere'),
        (1.0, 'Hoop')
    ]

    for c, name in shapes:
        ratio = 1.0 / (1.0 + c)
        ax4.plot(c, ratio, 'ro', markersize=10)
        ax4.annotate(name, (c, ratio), xytext=(5, 5), textcoords='offset points',
                     fontsize=8)

    ax4.set_xlabel('Inertia Coefficient c = I/(mR^2)')
    ax4.set_ylabel('Acceleration / g sin(theta)')
    ax4.set_title('Effective Acceleration vs Moment of Inertia\n'
                  'a = g*sin(theta) / (1 + c)')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 1.5)
    ax4.set_ylim(0, 1)

    # --- Minimum friction for rolling ---
    ax5 = fig.add_subplot(2, 3, 5)

    angles = np.linspace(0, np.radians(60), 100)

    for c, name, color in [(0.4, 'Solid Sphere', 'green'),
                            (0.5, 'Solid Cylinder', 'blue'),
                            (1.0, 'Hoop', 'red')]:
        mu_min = np.tan(angles) / (1.0 + 1.0/c)
        ax5.plot(np.degrees(angles), mu_min, color=color, lw=2, label=name)

    ax5.axhline(y=0.3, color='gray', linestyle='--', label='mu = 0.3 (typical)')
    ax5.axhline(y=0.5, color='gray', linestyle=':', label='mu = 0.5')

    ax5.set_xlabel('Incline Angle (degrees)')
    ax5.set_ylabel('Minimum Friction Coefficient')
    ax5.set_title('Minimum Friction for Pure Rolling\n'
                  'mu_min = tan(theta) / (1 + 1/c)')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0, 60)
    ax5.set_ylim(0, 0.8)

    # --- Rolling vs sliding energy efficiency ---
    ax6 = fig.add_subplot(2, 3, 6)

    # Compare final velocity for same height drop
    h_drop = 1.0  # 1 meter drop

    # Rolling (no energy loss)
    c_values = [0.4, 0.5, 2/3, 1.0]
    names = ['Solid Sphere', 'Solid Cylinder', 'Hollow Sphere', 'Hoop']

    final_v_rolling = []
    for c in c_values:
        # v^2 = 2*g*h / (1 + c)
        v = np.sqrt(2 * g * h_drop / (1 + c))
        final_v_rolling.append(v)

    # Sliding with friction
    mu_values = [0.0, 0.1, 0.2, 0.3]
    final_v_sliding = []
    for mu in mu_values:
        # Using 30 degree incline
        # Energy: mgh = 0.5*m*v^2 + mu*m*g*cos(theta)*d
        # where d = h/sin(theta)
        # v^2 = 2*g*h*(1 - mu*cot(theta))
        v_sq = 2 * g * h_drop * (1 - mu / np.tan(incline_angle))
        if v_sq > 0:
            final_v_sliding.append(np.sqrt(v_sq))
        else:
            final_v_sliding.append(0)

    x_pos = np.arange(len(c_values))
    width = 0.35

    bars1 = ax6.bar(x_pos - width/2, final_v_rolling, width, label='Rolling',
                    color='blue', alpha=0.7)
    bars2 = ax6.bar(x_pos + width/2, final_v_sliding[:len(c_values)], width,
                    label='Sliding (mu=0)', color='orange', alpha=0.7)

    ax6.set_ylabel('Final Velocity (m/s)')
    ax6.set_xlabel('Shape')
    ax6.set_title(f'Final Velocity After {h_drop}m Drop')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(names, rotation=15, ha='right')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Rolling Without Slipping: Physics of Rolling Motion',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'rolling_without_slipping.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'rolling_without_slipping.png')}")

    # Print analysis
    print("\nRolling Without Slipping Analysis:")
    print("-" * 60)
    print(f"Incline angle: {np.degrees(incline_angle):.0f} degrees")
    print(f"\nShape Comparison (after t={t_final}s):")
    print(f"{'Shape':<20} {'c':<8} {'Distance (m)':<15} {'Velocity (m/s)':<15}")
    print("-" * 60)
    for body in bodies:
        print(f"{body.name:<20} {body.c:<8.2f} {body._history['x'][-1]:<15.3f} "
              f"{body._history['v'][-1]:<15.3f}")

    print(f"\nMinimum friction for pure rolling at {np.degrees(incline_angle):.0f} deg:")
    for c, name in shapes:
        mu_min = minimum_friction_for_rolling(c, incline_angle)
        print(f"  {name}: mu_min = {mu_min:.3f}")


if __name__ == "__main__":
    main()
