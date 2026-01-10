"""
Experiment 42: Gas of Hard Spheres (Maxwell Distribution)

This example demonstrates the emergence of the Maxwell-Boltzmann velocity
distribution from a gas of hard spheres undergoing elastic collisions.
Shows how thermal equilibrium and the characteristic distribution arise
from simple collision mechanics.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


class HardSphere:
    """A hard sphere particle for gas simulation."""

    def __init__(self, pos, vel, radius, mass=1.0):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.radius = radius
        self.mass = mass


def maxwell_boltzmann_speed(v, m, kT):
    """
    Maxwell-Boltzmann speed distribution in 2D.

    f(v) = (m*v / kT) * exp(-m*v^2 / (2*kT))
    """
    return (m * v / kT) * np.exp(-m * v**2 / (2 * kT))


def maxwell_boltzmann_velocity_component(vx, m, kT):
    """
    Maxwell-Boltzmann distribution for one velocity component.

    f(vx) = sqrt(m / (2*pi*kT)) * exp(-m*vx^2 / (2*kT))
    """
    return np.sqrt(m / (2 * np.pi * kT)) * np.exp(-m * vx**2 / (2 * kT))


def simulate_hard_sphere_gas(n_particles, box_size, radius, initial_speed,
                              t_final, dt, mass=1.0):
    """
    Simulate a 2D gas of hard spheres.

    Args:
        n_particles: Number of particles
        box_size: Side length of square box
        radius: Particle radius
        initial_speed: Initial speed for all particles
        t_final: Simulation duration
        dt: Time step
        mass: Particle mass

    Returns:
        Dictionary with simulation data
    """
    # Initialize particles with random positions and velocities
    particles = []

    # Create grid to avoid initial overlaps
    grid_size = int(np.ceil(np.sqrt(n_particles)))
    spacing = box_size / grid_size

    particle_idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if particle_idx >= n_particles:
                break

            x = spacing/2 + i * spacing
            y = spacing/2 + j * spacing

            # Random velocity direction with fixed initial speed
            angle = np.random.uniform(0, 2*np.pi)
            vx = initial_speed * np.cos(angle)
            vy = initial_speed * np.sin(angle)

            particles.append(HardSphere([x, y], [vx, vy], radius, mass))
            particle_idx += 1
        if particle_idx >= n_particles:
            break

    # Storage for snapshots
    times = [0]
    velocity_snapshots = [[p.vel.copy() for p in particles]]
    speed_snapshots = [[np.linalg.norm(p.vel) for p in particles]]

    n_collisions = 0
    t = 0

    while t < t_final:
        # Update positions
        for p in particles:
            p.pos += p.vel * dt

        # Wall collisions
        for p in particles:
            if p.pos[0] - p.radius < 0:
                p.pos[0] = p.radius
                p.vel[0] = -p.vel[0]
            if p.pos[0] + p.radius > box_size:
                p.pos[0] = box_size - p.radius
                p.vel[0] = -p.vel[0]
            if p.pos[1] - p.radius < 0:
                p.pos[1] = p.radius
                p.vel[1] = -p.vel[1]
            if p.pos[1] + p.radius > box_size:
                p.pos[1] = box_size - p.radius
                p.vel[1] = -p.vel[1]

        # Particle-particle collisions
        for i in range(n_particles):
            for j in range(i + 1, n_particles):
                p1, p2 = particles[i], particles[j]

                delta = p2.pos - p1.pos
                dist = np.linalg.norm(delta)

                if dist < 2 * radius and dist > 0:
                    # Normal vector
                    normal = delta / dist

                    # Relative velocity
                    rel_vel = p1.vel - p2.vel
                    vel_along_normal = np.dot(rel_vel, normal)

                    # Only resolve if approaching
                    if vel_along_normal > 0:
                        # Elastic collision (equal masses)
                        p1.vel -= vel_along_normal * normal
                        p2.vel += vel_along_normal * normal

                        # Separate particles
                        overlap = 2 * radius - dist
                        p1.pos -= (overlap / 2) * normal
                        p2.pos += (overlap / 2) * normal

                        n_collisions += 1

        t += dt

        # Store snapshots periodically
        if len(times) < 1000 or t - times[-1] > t_final / 100:
            times.append(t)
            velocity_snapshots.append([p.vel.copy() for p in particles])
            speed_snapshots.append([np.linalg.norm(p.vel) for p in particles])

    return {
        'times': np.array(times),
        'velocities': np.array(velocity_snapshots),
        'speeds': np.array(speed_snapshots),
        'n_collisions': n_collisions,
        'final_particles': particles
    }


def main():
    # Parameters
    n_particles = 200
    box_size = 10.0
    radius = 0.15
    initial_speed = 1.0  # All particles start with same speed
    mass = 1.0
    t_final = 100.0
    dt = 0.01

    print("Simulating hard sphere gas...")
    print(f"  Particles: {n_particles}")
    print(f"  Box size: {box_size} x {box_size}")
    print(f"  Particle radius: {radius}")

    results = simulate_hard_sphere_gas(n_particles, box_size, radius,
                                        initial_speed, t_final, dt, mass)

    print(f"  Total collisions: {results['n_collisions']}")

    # Calculate temperature from average kinetic energy
    # In 2D: <KE> = kT (equipartition: 2 * (1/2)kT)
    speeds_final = results['speeds'][-1]
    avg_KE = 0.5 * mass * np.mean(np.array(speeds_final)**2)
    kT = avg_KE  # In 2D

    print(f"  Final temperature (kT): {kT:.4f}")

    # Create figure
    fig = plt.figure(figsize=(16, 12))

    # Subplot 1: Initial velocity distribution
    ax1 = fig.add_subplot(2, 3, 1)

    speeds_initial = results['speeds'][0]
    ax1.hist(speeds_initial, bins=20, density=True, alpha=0.7, color='blue',
             label='Initial', edgecolor='black')
    ax1.axvline(x=initial_speed, color='r', linestyle='--', lw=2,
                label=f'Initial speed = {initial_speed}')
    ax1.set_xlabel('Speed')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('Initial Speed Distribution\n(All particles same speed)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Final velocity distribution with Maxwell-Boltzmann fit
    ax2 = fig.add_subplot(2, 3, 2)

    ax2.hist(speeds_final, bins=30, density=True, alpha=0.7, color='green',
             label='Simulation', edgecolor='black')

    # Theoretical Maxwell-Boltzmann distribution
    v_range = np.linspace(0, max(speeds_final) * 1.2, 100)
    mb_theory = maxwell_boltzmann_speed(v_range, mass, kT)
    ax2.plot(v_range, mb_theory, 'r-', lw=2, label='Maxwell-Boltzmann theory')

    ax2.set_xlabel('Speed')
    ax2.set_ylabel('Probability Density')
    ax2.set_title(f'Final Speed Distribution (t = {t_final})\nkT = {kT:.3f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Velocity component distribution
    ax3 = fig.add_subplot(2, 3, 3)

    vx_final = [v[0] for v in results['velocities'][-1]]
    vy_final = [v[1] for v in results['velocities'][-1]]

    ax3.hist(vx_final, bins=25, density=True, alpha=0.6, color='blue',
             label='vx', edgecolor='black')
    ax3.hist(vy_final, bins=25, density=True, alpha=0.6, color='red',
             label='vy', edgecolor='black')

    # Theoretical Gaussian
    vx_range = np.linspace(min(vx_final) * 1.2, max(vx_final) * 1.2, 100)
    gaussian_theory = maxwell_boltzmann_velocity_component(vx_range, mass, kT)
    ax3.plot(vx_range, gaussian_theory, 'k-', lw=2, label='Theory (Gaussian)')

    ax3.set_xlabel('Velocity Component')
    ax3.set_ylabel('Probability Density')
    ax3.set_title('Velocity Component Distribution\n(Should be Gaussian)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Subplot 4: Evolution of speed distribution
    ax4 = fig.add_subplot(2, 3, 4)

    # Plot distributions at different times
    time_indices = [0, len(results['times'])//4, len(results['times'])//2,
                    3*len(results['times'])//4, -1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(time_indices)))

    for idx, color in zip(time_indices, colors):
        speeds = results['speeds'][idx]
        t = results['times'][idx]
        ax4.hist(speeds, bins=20, density=True, alpha=0.4, color=color,
                 label=f't = {t:.1f}', histtype='step', lw=2)

    ax4.plot(v_range, mb_theory, 'k--', lw=2, label='Equilibrium MB')
    ax4.set_xlabel('Speed')
    ax4.set_ylabel('Probability Density')
    ax4.set_title('Evolution Toward Equilibrium')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # Subplot 5: Temperature evolution
    ax5 = fig.add_subplot(2, 3, 5)

    temperatures = []
    for speeds in results['speeds']:
        avg_v2 = np.mean(np.array(speeds)**2)
        T = 0.5 * mass * avg_v2  # kT in 2D
        temperatures.append(T)

    ax5.plot(results['times'], temperatures, 'b-', lw=1.5)
    ax5.axhline(y=np.mean(temperatures[-10:]), color='r', linestyle='--',
                label='Equilibrium T')
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Temperature (kT)')
    ax5.set_title('Temperature Equilibration')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Subplot 6: Final particle positions
    ax6 = fig.add_subplot(2, 3, 6)

    for p in results['final_particles']:
        circle = plt.Circle(p.pos, p.radius, fill=False, color='blue', alpha=0.5)
        ax6.add_patch(circle)

        # Velocity arrows
        scale = 0.3
        ax6.arrow(p.pos[0], p.pos[1], scale*p.vel[0], scale*p.vel[1],
                  head_width=0.1, head_length=0.05, fc='red', ec='red', alpha=0.5)

    ax6.set_xlim(0, box_size)
    ax6.set_ylim(0, box_size)
    ax6.set_aspect('equal')
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    ax6.set_title('Final Configuration\n(arrows show velocities)')
    ax6.grid(True, alpha=0.3)

    plt.suptitle('Hard Sphere Gas: Emergence of Maxwell-Boltzmann Distribution\n'
                 f'{n_particles} particles, {results["n_collisions"]} collisions',
                 fontsize=12, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'hard_sphere_gas.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'hard_sphere_gas.png')}")

    # Statistical analysis
    print("\nStatistical Analysis:")
    print("=" * 50)

    # Chi-squared test for Maxwell-Boltzmann
    speeds_arr = np.array(speeds_final)
    # Compare with theoretical MB distribution
    mean_speed = np.mean(speeds_arr)
    std_speed = np.std(speeds_arr)
    theoretical_mean = np.sqrt(np.pi * kT / (2 * mass))  # 2D MB mean
    theoretical_std = np.sqrt((4 - np.pi) * kT / (2 * mass))  # 2D MB std

    print(f"Mean speed - Simulation: {mean_speed:.4f}, Theory: {theoretical_mean:.4f}")
    print(f"Std speed  - Simulation: {std_speed:.4f}, Theory: {theoretical_std:.4f}")

    # Normality test for velocity components
    vx_arr = np.array(vx_final)
    vy_arr = np.array(vy_final)
    _, p_vx = stats.normaltest(vx_arr)
    _, p_vy = stats.normaltest(vy_arr)
    print(f"Normality test p-values: vx = {p_vx:.4f}, vy = {p_vy:.4f}")
    print("  (p > 0.05 suggests distribution is consistent with Gaussian)")


if __name__ == "__main__":
    main()
