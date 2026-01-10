"""
Experiment 40: 1D Inelastic Collision

This example demonstrates inelastic 1D collisions where momentum is conserved
but kinetic energy is not. Shows the spectrum from perfectly elastic (e=1)
to perfectly inelastic (e=0) collisions.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.mechanics import Collision


def inelastic_collision(m1, m2, v1, v2, e):
    """
    Calculate final velocities for 1D inelastic collision.

    Coefficient of restitution: e = -(v1' - v2') / (v1 - v2)
    Combined with momentum conservation gives:
    v1' = (m1*v1 + m2*v2 + m2*e*(v2 - v1)) / (m1 + m2)
    v2' = (m1*v1 + m2*v2 + m1*e*(v1 - v2)) / (m1 + m2)

    Args:
        m1, m2: Masses
        v1, v2: Initial velocities
        e: Coefficient of restitution (0 to 1)

    Returns:
        Final velocities (v1', v2')
    """
    v1_final = (m1*v1 + m2*v2 + m2*e*(v2 - v1)) / (m1 + m2)
    v2_final = (m1*v1 + m2*v2 + m1*e*(v1 - v2)) / (m1 + m2)
    return v1_final, v2_final


def perfectly_inelastic_collision(m1, m2, v1, v2):
    """
    Calculate final velocity for perfectly inelastic collision (e=0).
    Both objects stick together and move with common velocity.

    v_final = (m1*v1 + m2*v2) / (m1 + m2)
    """
    return (m1*v1 + m2*v2) / (m1 + m2)


def simulate_collision(m1, m2, x1_0, x2_0, v1_0, v2_0, e, t_final, dt):
    """
    Simulate 1D collision with given coefficient of restitution.
    """
    x1, x2 = x1_0, x2_0
    v1, v2 = v1_0, v2_0

    collision = Collision(restitution_coeff=e)
    collided = False

    times = [0]
    x1s, x2s = [x1], [x2]
    v1s, v2s = [v1], [v2]

    t = 0
    while t < t_final:
        # Check for collision
        if not collided and x2 - x1 <= 0:
            normal = np.array([1.0, 0.0, 0.0])
            v1_3d = np.array([v1, 0.0, 0.0])
            v2_3d = np.array([v2, 0.0, 0.0])
            v1_new_3d, v2_new_3d = collision.resolve(m1, m2, v1_3d, v2_3d, normal)
            v1 = v1_new_3d[0]
            v2 = v2_new_3d[0]
            collided = True

        # Update positions (for inelastic, they may stick together)
        if e == 0 and collided:
            # Perfectly inelastic: move together
            v_common = (m1*v1 + m2*v2) / (m1 + m2)
            x1 += v_common * dt
            x2 = x1  # They're stuck together
        else:
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

    # Parameters
    m1, m2 = 2.0, 1.0
    v1_0, v2_0 = 3.0, -1.0
    x1_0, x2_0 = 0.0, 4.0

    # Subplot 1: Position comparison for different e values
    ax1 = fig.add_subplot(2, 3, 1)

    e_values = [1.0, 0.8, 0.5, 0.2, 0.0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(e_values)))

    for e, color in zip(e_values, colors):
        results = simulate_collision(m1, m2, x1_0, x2_0, v1_0, v2_0, e, 3.0, 0.001)
        ax1.plot(results['time'], results['x1'], '-', color=color, lw=2,
                 label=f'e = {e}')

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position of m1 (m)')
    ax1.set_title('Projectile Position for Different e')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Velocity comparison
    ax2 = fig.add_subplot(2, 3, 2)

    for e, color in zip(e_values, colors):
        results = simulate_collision(m1, m2, x1_0, x2_0, v1_0, v2_0, e, 3.0, 0.001)
        ax2.plot(results['time'], results['v1'], '-', color=color, lw=2,
                 label=f'e = {e}')

    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity of m1 (m/s)')
    ax2.set_title('Projectile Velocity for Different e')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Energy loss vs coefficient of restitution
    ax3 = fig.add_subplot(2, 3, 3)

    e_range = np.linspace(0, 1, 100)
    energy_loss_fraction = []

    KE_initial = 0.5 * m1 * v1_0**2 + 0.5 * m2 * v2_0**2

    for e in e_range:
        v1_f, v2_f = inelastic_collision(m1, m2, v1_0, v2_0, e)
        KE_final = 0.5 * m1 * v1_f**2 + 0.5 * m2 * v2_f**2
        loss = (KE_initial - KE_final) / KE_initial
        energy_loss_fraction.append(loss)

    ax3.plot(e_range, energy_loss_fraction, 'b-', lw=2)
    ax3.fill_between(e_range, energy_loss_fraction, alpha=0.3)
    ax3.set_xlabel('Coefficient of Restitution e')
    ax3.set_ylabel('Fraction of KE Lost')
    ax3.set_title('Energy Dissipation vs Restitution')
    ax3.grid(True, alpha=0.3)

    # Mark special cases
    ax3.axvline(x=1.0, color='g', linestyle='--', alpha=0.5, label='Elastic')
    ax3.axvline(x=0.0, color='r', linestyle='--', alpha=0.5, label='Perfectly inelastic')
    ax3.legend()

    # Subplot 4: Perfectly inelastic collision demonstration
    ax4 = fig.add_subplot(2, 3, 4)

    # Two objects collide and stick
    m1_pi, m2_pi = 3.0, 2.0
    v1_pi, v2_pi = 4.0, -2.0

    v_final = perfectly_inelastic_collision(m1_pi, m2_pi, v1_pi, v2_pi)
    p_initial = m1_pi * v1_pi + m2_pi * v2_pi
    KE_initial_pi = 0.5 * m1_pi * v1_pi**2 + 0.5 * m2_pi * v2_pi**2
    KE_final_pi = 0.5 * (m1_pi + m2_pi) * v_final**2

    # Bar chart
    labels = ['Before', 'After']
    momentum_vals = [p_initial, (m1_pi + m2_pi) * v_final]
    KE_vals = [KE_initial_pi, KE_final_pi]

    x_bar = np.arange(len(labels))
    width = 0.35

    bars1 = ax4.bar(x_bar - width/2, momentum_vals, width, label='Momentum (kg m/s)', color='blue', alpha=0.7)
    bars2 = ax4.bar(x_bar + width/2, KE_vals, width, label='Kinetic Energy (J)', color='red', alpha=0.7)

    ax4.set_xlabel('State')
    ax4.set_ylabel('Value')
    ax4.set_title(f'Perfectly Inelastic Collision\nm1={m1_pi}kg @ {v1_pi}m/s, m2={m2_pi}kg @ {v2_pi}m/s')
    ax4.set_xticks(x_bar)
    ax4.set_xticklabels(labels)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # Add text annotations
    ax4.text(0, momentum_vals[0] * 1.05, f'{momentum_vals[0]:.1f}', ha='center', fontsize=9)
    ax4.text(1, momentum_vals[1] * 1.05, f'{momentum_vals[1]:.1f}', ha='center', fontsize=9)
    ax4.text(0.35, KE_vals[0] * 0.5, f'{KE_vals[0]:.1f}', ha='center', fontsize=9, color='white')
    ax4.text(1.35, KE_vals[1] * 0.5, f'{KE_vals[1]:.1f}', ha='center', fontsize=9, color='white')

    # Subplot 5: Final velocities as function of e
    ax5 = fig.add_subplot(2, 3, 5)

    v1_finals = []
    v2_finals = []

    for e in e_range:
        v1_f, v2_f = inelastic_collision(m1, m2, v1_0, v2_0, e)
        v1_finals.append(v1_f)
        v2_finals.append(v2_f)

    ax5.plot(e_range, v1_finals, 'b-', lw=2, label="v1' (m1)")
    ax5.plot(e_range, v2_finals, 'r-', lw=2, label="v2' (m2)")
    ax5.axhline(y=v1_0, color='b', linestyle='--', alpha=0.3, label='v1 initial')
    ax5.axhline(y=v2_0, color='r', linestyle='--', alpha=0.3, label='v2 initial')

    # For e=0, both should have same final velocity (center of mass velocity)
    v_cm = (m1 * v1_0 + m2 * v2_0) / (m1 + m2)
    ax5.axhline(y=v_cm, color='g', linestyle=':', alpha=0.7, label=f'v_cm = {v_cm:.2f}')

    ax5.set_xlabel('Coefficient of Restitution e')
    ax5.set_ylabel('Final Velocity (m/s)')
    ax5.set_title('Final Velocities vs Restitution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Subplot 6: Real-world examples with typical e values
    ax6 = fig.add_subplot(2, 3, 6)

    # Real materials and their typical coefficients
    materials = {
        'Super ball (rubber)': 0.9,
        'Golf ball': 0.83,
        'Tennis ball': 0.75,
        'Basketball': 0.76,
        'Baseball': 0.55,
        'Softball': 0.40,
        'Clay (modeling)': 0.10,
        'Putty': 0.0
    }

    # Simulate collision for each material
    m1_real, m2_real = 1.0, 10.0  # Ball hitting wall (m2 >> m1)
    v1_real, v2_real = 5.0, 0.0

    material_names = list(materials.keys())
    rebound_velocities = []
    energy_retained = []

    for name, e in materials.items():
        v1_f, v2_f = inelastic_collision(m1_real, m2_real, v1_real, v2_real, e)
        rebound_velocities.append(abs(v1_f))
        KE_i = 0.5 * m1_real * v1_real**2
        KE_f = 0.5 * m1_real * v1_f**2
        energy_retained.append(100 * KE_f / KE_i)

    y_pos = np.arange(len(material_names))
    ax6.barh(y_pos, energy_retained, color=plt.cm.RdYlGn(np.array(list(materials.values()))))
    ax6.set_yticks(y_pos)
    ax6.set_yticklabels(material_names)
    ax6.set_xlabel('Kinetic Energy Retained (%)')
    ax6.set_title('Real Materials: Ball Bouncing off Wall')
    ax6.grid(True, alpha=0.3, axis='x')

    # Add e values as text
    for i, (name, e) in enumerate(materials.items()):
        ax6.text(energy_retained[i] + 2, i, f'e={e}', va='center', fontsize=8)

    plt.suptitle(f"1D Inelastic Collisions\nm1={m1}kg, m2={m2}kg, v1={v1_0}m/s, v2={v2_0}m/s",
                 fontsize=12, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'collision_1d_inelastic.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'collision_1d_inelastic.png')}")

    # Print summary
    print("\nInelastic Collision Summary:")
    print("=" * 50)
    print("Coefficient of restitution: e = |v_rel_after| / |v_rel_before|")
    print("  e = 1: Perfectly elastic (KE conserved)")
    print("  0 < e < 1: Inelastic (some KE lost)")
    print("  e = 0: Perfectly inelastic (maximum KE loss, objects stick)")
    print(f"\nFor m1={m1}kg, m2={m2}kg, v1={v1_0}m/s, v2={v2_0}m/s:")
    print(f"  Elastic (e=1): KE loss = 0%")
    print(f"  Inelastic (e=0): KE loss = {energy_loss_fraction[0]*100:.1f}%")


if __name__ == "__main__":
    main()
