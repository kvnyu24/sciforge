"""
Experiment 39: 1D Elastic Collision

This example demonstrates perfectly elastic 1D collisions where both
momentum and kinetic energy are conserved. Shows analytical solutions
and compares with numerical simulation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.mechanics import Collision


def elastic_collision_analytical(m1, m2, v1, v2):
    """
    Calculate final velocities for 1D elastic collision analytically.

    Conservation of momentum: m1*v1 + m2*v2 = m1*v1' + m2*v2'
    Conservation of energy: m1*v1^2 + m2*v2^2 = m1*v1'^2 + m2*v2'^2

    Solution:
    v1' = ((m1-m2)*v1 + 2*m2*v2) / (m1+m2)
    v2' = ((m2-m1)*v2 + 2*m1*v1) / (m1+m2)
    """
    v1_final = ((m1 - m2) * v1 + 2 * m2 * v2) / (m1 + m2)
    v2_final = ((m2 - m1) * v2 + 2 * m1 * v1) / (m1 + m2)
    return v1_final, v2_final


def simulate_1d_collision(m1, m2, x1_0, x2_0, v1_0, v2_0, t_final, dt):
    """
    Simulate 1D collision between two particles.

    Args:
        m1, m2: Masses
        x1_0, x2_0: Initial positions
        v1_0, v2_0: Initial velocities
        t_final: Simulation duration
        dt: Time step

    Returns:
        Dictionary with trajectory data
    """
    x1, x2 = x1_0, x2_0
    v1, v2 = v1_0, v2_0

    collision = Collision(restitution_coeff=1.0)  # Perfectly elastic
    collided = False

    times = [0]
    x1s, x2s = [x1], [x2]
    v1s, v2s = [v1], [v2]

    t = 0
    while t < t_final:
        # Check for collision (particles meet)
        if not collided and x2 - x1 <= 0:
            # Resolve collision
            normal = np.array([1.0, 0.0, 0.0])
            v1_3d = np.array([v1, 0.0, 0.0])
            v2_3d = np.array([v2, 0.0, 0.0])
            v1_new_3d, v2_new_3d = collision.resolve(m1, m2, v1_3d, v2_3d, normal)
            v1 = v1_new_3d[0]
            v2 = v2_new_3d[0]
            collided = True

        # Update positions
        x1 += v1 * dt
        x2 += v2 * dt
        t += dt

        times.append(t)
        x1s.append(x1)
        x2s.append(x2)
        v1s.append(v1)
        v2s.append(v2)

    return {
        'time': np.array(times),
        'x1': np.array(x1s),
        'x2': np.array(x2s),
        'v1': np.array(v1s),
        'v2': np.array(v2s)
    }


def main():
    # Create figure
    fig = plt.figure(figsize=(16, 12))

    # Case 1: Equal masses
    m1, m2 = 1.0, 1.0
    v1_0, v2_0 = 3.0, -1.0
    x1_0, x2_0 = 0.0, 5.0

    results1 = simulate_1d_collision(m1, m2, x1_0, x2_0, v1_0, v2_0, 3.0, 0.001)
    v1_f_theory, v2_f_theory = elastic_collision_analytical(m1, m2, v1_0, v2_0)

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(results1['time'], results1['x1'], 'b-', lw=2, label=f'm1 = {m1} kg')
    ax1.plot(results1['time'], results1['x2'], 'r-', lw=2, label=f'm2 = {m2} kg')
    ax1.axvline(x=1.25, color='k', linestyle='--', alpha=0.3, label='Collision')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (m)')
    ax1.set_title(f'Equal Masses: Velocity Exchange\nv1: {v1_0} -> {v1_f_theory:.2f}, v2: {v2_0} -> {v2_f_theory:.2f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Case 2: Heavy particle hits light particle at rest
    m1, m2 = 4.0, 1.0
    v1_0, v2_0 = 2.0, 0.0
    x1_0, x2_0 = 0.0, 5.0

    results2 = simulate_1d_collision(m1, m2, x1_0, x2_0, v1_0, v2_0, 4.0, 0.001)
    v1_f_theory, v2_f_theory = elastic_collision_analytical(m1, m2, v1_0, v2_0)

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(results2['time'], results2['x1'], 'b-', lw=2, label=f'm1 = {m1} kg')
    ax2.plot(results2['time'], results2['x2'], 'r-', lw=2, label=f'm2 = {m2} kg')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position (m)')
    ax2.set_title(f'Heavy -> Light at Rest\nv1: {v1_0} -> {v1_f_theory:.2f}, v2: {v2_0} -> {v2_f_theory:.2f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Case 3: Light particle hits heavy particle at rest
    m1, m2 = 1.0, 4.0
    v1_0, v2_0 = 3.0, 0.0
    x1_0, x2_0 = 0.0, 5.0

    results3 = simulate_1d_collision(m1, m2, x1_0, x2_0, v1_0, v2_0, 4.0, 0.001)
    v1_f_theory, v2_f_theory = elastic_collision_analytical(m1, m2, v1_0, v2_0)

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(results3['time'], results3['x1'], 'b-', lw=2, label=f'm1 = {m1} kg')
    ax3.plot(results3['time'], results3['x2'], 'r-', lw=2, label=f'm2 = {m2} kg')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Position (m)')
    ax3.set_title(f'Light -> Heavy at Rest (Rebound)\nv1: {v1_0} -> {v1_f_theory:.2f}, v2: {v2_0} -> {v2_f_theory:.2f}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Case 4: Velocity vs mass ratio plot
    ax4 = fig.add_subplot(2, 3, 4)

    mass_ratios = np.linspace(0.1, 10, 100)
    v1_0 = 1.0
    v2_0 = 0.0  # Target at rest

    v1_finals = []
    v2_finals = []

    for ratio in mass_ratios:
        m1 = 1.0
        m2 = ratio
        v1_f, v2_f = elastic_collision_analytical(m1, m2, v1_0, v2_0)
        v1_finals.append(v1_f)
        v2_finals.append(v2_f)

    ax4.plot(mass_ratios, v1_finals, 'b-', lw=2, label="v1' (projectile)")
    ax4.plot(mass_ratios, v2_finals, 'r-', lw=2, label="v2' (target)")
    ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax4.axhline(y=v1_0, color='b', linestyle='--', alpha=0.3, label='Initial v1')
    ax4.axvline(x=1.0, color='g', linestyle='--', alpha=0.3, label='m1 = m2')
    ax4.set_xlabel('Mass ratio m2/m1')
    ax4.set_ylabel('Final velocity')
    ax4.set_title('Final Velocities vs Mass Ratio\n(Target initially at rest)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Case 5: Energy and momentum conservation check
    ax5 = fig.add_subplot(2, 3, 5)

    # Use case 1 data
    m1, m2 = 1.0, 1.0
    p_total = m1 * results1['v1'] + m2 * results1['v2']
    KE_total = 0.5 * m1 * results1['v1']**2 + 0.5 * m2 * results1['v2']**2

    ax5.plot(results1['time'], p_total / p_total[0], 'b-', lw=2, label='Momentum')
    ax5.plot(results1['time'], KE_total / KE_total[0], 'r--', lw=2, label='Kinetic Energy')
    ax5.axhline(y=1.0, color='k', linestyle=':', alpha=0.5)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Normalized quantity')
    ax5.set_title('Conservation Laws (Equal Mass Collision)')
    ax5.set_ylim(0.95, 1.05)
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Case 6: Multiple collisions (Newton's cradle effect)
    ax6 = fig.add_subplot(2, 3, 6)

    # Simplified Newton's cradle: 5 equal masses in a line
    n_balls = 5
    m_ball = 1.0
    spacing = 2.0

    # Initial: first ball moving toward others at rest
    x_init = [i * spacing for i in range(n_balls)]
    v_init = [2.0] + [0.0] * (n_balls - 1)

    # Simulate collisions
    times_cradle = [0]
    positions_cradle = [[x for x in x_init]]

    x = list(x_init)
    v = list(v_init)
    t = 0
    dt_cradle = 0.001
    t_final_cradle = 5.0

    while t < t_final_cradle:
        # Check for collisions between adjacent balls
        for i in range(n_balls - 1):
            if x[i+1] - x[i] <= 0.01 and v[i] > v[i+1]:
                # Elastic collision between equal masses = velocity exchange
                v[i], v[i+1] = v[i+1], v[i]

        # Update positions
        for i in range(n_balls):
            x[i] += v[i] * dt_cradle

        t += dt_cradle
        times_cradle.append(t)
        positions_cradle.append([pos for pos in x])

    positions_cradle = np.array(positions_cradle)

    for i in range(n_balls):
        ax6.plot(times_cradle, positions_cradle[:, i], lw=2, label=f'Ball {i+1}')

    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Position (m)')
    ax6.set_title("Newton's Cradle (5 Equal Masses)")
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    plt.suptitle("1D Elastic Collisions\nConservation of Momentum and Kinetic Energy",
                 fontsize=12, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'collision_1d_elastic.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'collision_1d_elastic.png')}")

    # Print summary
    print("\nElastic Collision Summary:")
    print("=" * 50)
    print("Key formulas for 1D elastic collision:")
    print("  v1' = ((m1-m2)*v1 + 2*m2*v2) / (m1+m2)")
    print("  v2' = ((m2-m1)*v2 + 2*m1*v1) / (m1+m2)")
    print("\nSpecial cases:")
    print("  Equal masses: Complete velocity exchange")
    print("  m1 >> m2: Projectile barely slowed, target gets ~2*v1")
    print("  m1 << m2: Projectile rebounds, target barely moves")


if __name__ == "__main__":
    main()
