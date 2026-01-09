"""
Example demonstrating elastic vs inelastic collisions.

This example shows the difference between perfectly elastic and inelastic
collisions, demonstrating conservation of momentum and kinetic energy.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.mechanics import Collision


def simulate_collision(m1, m2, v1, v2, restitution):
    """
    Simulate 1D collision between two particles.

    Args:
        m1, m2: Masses of particles (kg)
        v1, v2: Initial velocities (m/s)
        restitution: Coefficient of restitution (1=elastic, 0=perfectly inelastic)

    Returns:
        Final velocities after collision
    """
    collision = Collision(restitution_coeff=restitution)

    # For 1D collision, use normal along x-axis
    normal = np.array([1.0, 0.0, 0.0])
    v1_3d = np.array([v1, 0.0, 0.0])
    v2_3d = np.array([v2, 0.0, 0.0])

    v1_final, v2_final = collision.resolve(m1, m2, v1_3d, v2_3d, normal)

    return v1_final[0], v2_final[0]


def calculate_energies(m1, m2, v1, v2):
    """Calculate total kinetic energy."""
    return 0.5 * m1 * v1**2 + 0.5 * m2 * v2**2


def calculate_momentum(m1, m2, v1, v2):
    """Calculate total momentum."""
    return m1 * v1 + m2 * v2


def run_collision_sweep():
    """Run collisions for various restitution coefficients."""
    m1, m2 = 2.0, 1.0  # kg
    v1_initial, v2_initial = 5.0, -3.0  # m/s

    restitutions = np.linspace(0, 1, 11)
    results = []

    for e in restitutions:
        v1_f, v2_f = simulate_collision(m1, m2, v1_initial, v2_initial, e)
        ke_initial = calculate_energies(m1, m2, v1_initial, v2_initial)
        ke_final = calculate_energies(m1, m2, v1_f, v2_f)
        p_initial = calculate_momentum(m1, m2, v1_initial, v2_initial)
        p_final = calculate_momentum(m1, m2, v1_f, v2_f)

        results.append({
            'e': e,
            'v1_f': v1_f,
            'v2_f': v2_f,
            'ke_initial': ke_initial,
            'ke_final': ke_final,
            'ke_ratio': ke_final / ke_initial,
            'p_initial': p_initial,
            'p_final': p_final
        })

    return results, m1, m2, v1_initial, v2_initial


def plot_results(results, m1, m2, v1_i, v2_i):
    """Create comprehensive collision visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    restitutions = [r['e'] for r in results]

    # Plot 1: Final velocities vs restitution
    ax1 = axes[0, 0]
    v1_finals = [r['v1_f'] for r in results]
    v2_finals = [r['v2_f'] for r in results]

    ax1.plot(restitutions, v1_finals, 'b-o', lw=2, label=f'Particle 1 (m={m1} kg)')
    ax1.plot(restitutions, v2_finals, 'r-s', lw=2, label=f'Particle 2 (m={m2} kg)')
    ax1.axhline(y=v1_i, color='b', linestyle='--', alpha=0.5, label=f'v₁ initial = {v1_i} m/s')
    ax1.axhline(y=v2_i, color='r', linestyle='--', alpha=0.5, label=f'v₂ initial = {v2_i} m/s')
    ax1.set_xlabel('Coefficient of Restitution (e)')
    ax1.set_ylabel('Final Velocity (m/s)')
    ax1.set_title('Final Velocities After Collision')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Energy conservation
    ax2 = axes[0, 1]
    ke_ratios = [r['ke_ratio'] for r in results]
    ax2.plot(restitutions, ke_ratios, 'g-o', lw=2, markersize=8)
    ax2.fill_between(restitutions, ke_ratios, alpha=0.3)
    ax2.set_xlabel('Coefficient of Restitution (e)')
    ax2.set_ylabel('KE_final / KE_initial')
    ax2.set_title('Kinetic Energy Conservation')
    ax2.set_ylim(0, 1.1)
    ax2.axhline(y=1.0, color='r', linestyle='--', label='Perfect conservation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Momentum conservation (should always be conserved)
    ax3 = axes[1, 0]
    p_initials = [r['p_initial'] for r in results]
    p_finals = [r['p_final'] for r in results]

    ax3.plot(restitutions, p_initials, 'b-o', lw=2, label='Initial momentum')
    ax3.plot(restitutions, p_finals, 'r--s', lw=2, label='Final momentum')
    ax3.set_xlabel('Coefficient of Restitution (e)')
    ax3.set_ylabel('Momentum (kg⋅m/s)')
    ax3.set_title('Momentum Conservation (Always Conserved)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Energy loss vs restitution
    ax4 = axes[1, 1]
    energy_loss = [(1 - r['ke_ratio']) * 100 for r in results]
    ax4.bar(restitutions, energy_loss, width=0.08, color='orange', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Coefficient of Restitution (e)')
    ax4.set_ylabel('Energy Loss (%)')
    ax4.set_title('Energy Dissipated in Collision')
    ax4.grid(True, alpha=0.3, axis='y')

    # Add annotations
    ax4.annotate('Perfectly\nInelastic', xy=(0, energy_loss[0]),
                 xytext=(0.15, energy_loss[0] + 5),
                 fontsize=9, ha='center')
    ax4.annotate('Perfectly\nElastic', xy=(1, energy_loss[-1]),
                 xytext=(0.85, 10),
                 fontsize=9, ha='center')

    plt.suptitle(f'Collision Analysis: m₁={m1} kg, m₂={m2} kg, v₁={v1_i} m/s, v₂={v2_i} m/s',
                 fontsize=12, y=1.02)
    plt.tight_layout()


def main():
    # Run collision sweep
    results, m1, m2, v1_i, v2_i = run_collision_sweep()

    # Plot results
    plot_results(results, m1, m2, v1_i, v2_i)

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'elastic_collision.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'elastic_collision.png')}")


if __name__ == "__main__":
    main()
