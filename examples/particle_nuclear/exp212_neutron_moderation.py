"""
Experiment 212: Neutron Moderation Random Walk

Demonstrates neutron slowing down (moderation) in nuclear reactors
using random walk simulations. Shows energy loss and thermalization.

Physics:
- Elastic scattering: E' = E × ((A-1)/(A+1))² to E
- Average logarithmic decrement: ξ = 1 + (A-1)²/(2A) × ln((A-1)/(A+1))
- Slowing down: n collisions = ln(E₀/E_th) / ξ
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def scattering_angle_cm():
    """
    Random isotropic scattering angle in CM frame.
    cos(θ_cm) uniform in [-1, 1]
    """
    return 2 * np.random.random() - 1


def energy_after_collision(E, A, cos_theta_cm):
    """
    Neutron energy after elastic collision with nucleus of mass A.

    E' = E × (1 + α + (1 - α)cos(θ_cm))/2
    where α = ((A-1)/(A+1))²
    """
    alpha = ((A - 1) / (A + 1))**2
    return E * (1 + alpha + (1 - alpha) * cos_theta_cm) / 2


def lethargy(E, E0):
    """Lethargy u = ln(E₀/E)"""
    return np.log(E0 / E)


def average_log_decrement(A):
    """
    Average logarithmic energy decrement ξ per collision.

    ξ = 1 - (A-1)²/(2A) × ln((A+1)/(A-1))
    For A=1 (hydrogen): ξ = 1
    """
    if A == 1:
        return 1.0
    return 1 - (A - 1)**2 / (2 * A) * np.log((A + 1) / (A - 1))


def mean_collisions_to_thermalize(E0, E_th, A):
    """Number of collisions to thermalize from E₀ to E_th."""
    xi = average_log_decrement(A)
    return np.log(E0 / E_th) / xi


def simulate_neutron_moderation(E0, E_thermal, A, n_neutrons=1000):
    """
    Simulate neutron slowing down.

    Returns:
        List of collision counts to reach thermal energy
        Energy histories for some neutrons
    """
    collision_counts = []
    energy_histories = []

    for i in range(n_neutrons):
        E = E0
        history = [E]
        n_collisions = 0

        while E > E_thermal and n_collisions < 1000:
            cos_theta = scattering_angle_cm()
            E = energy_after_collision(E, A, cos_theta)
            history.append(E)
            n_collisions += 1

        collision_counts.append(n_collisions)
        if i < 10:  # Save first 10 histories
            energy_histories.append(history)

    return collision_counts, energy_histories


def simulate_3d_random_walk(n_steps, mean_free_path, scattering_sigma):
    """
    3D random walk for neutron diffusion.
    """
    positions = np.zeros((n_steps + 1, 3))

    for i in range(n_steps):
        # Random direction
        theta = np.arccos(2 * np.random.random() - 1)
        phi = 2 * np.pi * np.random.random()

        # Step length from exponential distribution
        step = -mean_free_path * np.log(np.random.random())

        # Direction vector
        dx = step * np.sin(theta) * np.cos(phi)
        dy = step * np.sin(theta) * np.sin(phi)
        dz = step * np.cos(theta)

        positions[i + 1] = positions[i] + np.array([dx, dy, dz])

    return positions


def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Parameters
    E0 = 2e6  # eV (2 MeV fission neutron)
    E_thermal = 0.025  # eV (thermal at 300 K)
    n_neutrons = 5000

    # Plot 1: Energy vs collision number for different moderators
    ax = axes[0, 0]

    moderators = [
        ('H (A=1)', 1, 'b'),
        ('D (A=2)', 2, 'r'),
        ('C (A=12)', 12, 'g'),
        ('U (A=238)', 238, 'm'),
    ]

    for name, A, color in moderators:
        collisions, histories = simulate_neutron_moderation(E0, E_thermal, A,
                                                             n_neutrons=100)
        # Plot first trajectory
        if histories:
            ax.semilogy(range(len(histories[0])), histories[0], '-', color=color,
                        lw=2, label=name)

    ax.axhline(y=E_thermal, color='k', linestyle='--', alpha=0.5,
               label='Thermal')
    ax.set_xlabel('Collision Number')
    ax.set_ylabel('Neutron Energy (eV)')
    ax.set_title('Energy vs Collision\n(Single trajectories)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(E_thermal/2, E0*2)

    # Plot 2: Distribution of collisions to thermalize
    ax = axes[0, 1]

    for name, A, color in moderators[:3]:  # Skip U (too many)
        collisions, _ = simulate_neutron_moderation(E0, E_thermal, A, n_neutrons)
        ax.hist(collisions, bins=30, alpha=0.5, color=color, label=name,
                density=True)

        # Theoretical mean
        n_mean = mean_collisions_to_thermalize(E0, E_thermal, A)
        ax.axvline(x=n_mean, color=color, linestyle='--', lw=2)

    ax.set_xlabel('Number of Collisions')
    ax.set_ylabel('Probability Density')
    ax.set_title('Collisions to Thermalize\n(2 MeV → 0.025 eV)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Average log decrement and collisions vs A
    ax = axes[0, 2]

    A_range = np.arange(1, 250)
    xi_values = [average_log_decrement(A) for A in A_range]
    n_collisions = [mean_collisions_to_thermalize(E0, E_thermal, A) for A in A_range]

    ax2 = ax.twinx()

    line1, = ax.plot(A_range, xi_values, 'b-', lw=2, label='ξ')
    line2, = ax2.plot(A_range, n_collisions, 'r-', lw=2, label='n collisions')

    ax.set_xlabel('Mass Number A')
    ax.set_ylabel('Average Log Decrement ξ', color='b')
    ax2.set_ylabel('Collisions to Thermalize', color='r')
    ax.set_title('Moderation Efficiency vs Mass')

    # Mark common moderators
    for name, A in [('H', 1), ('D', 2), ('Be', 9), ('C', 12), ('O', 16)]:
        ax.axvline(x=A, color='gray', linestyle=':', alpha=0.5)
        ax.text(A, 0.9, name, fontsize=8, ha='center')

    ax.legend([line1, line2], ['ξ', 'n collisions'], loc='right')
    ax.grid(True, alpha=0.3)

    # Plot 4: Lethargy distribution
    ax = axes[1, 0]

    # Simulate with hydrogen
    collisions, histories = simulate_neutron_moderation(E0, E_thermal, A=1,
                                                         n_neutrons=2000)

    all_energies = []
    for hist in histories:
        all_energies.extend(hist)

    lethargies = [lethargy(E, E0) for E in all_energies if E > E_thermal]

    ax.hist(lethargies, bins=50, density=True, alpha=0.7)
    ax.set_xlabel('Lethargy u = ln(E₀/E)')
    ax.set_ylabel('Probability Density')
    ax.set_title('Lethargy Distribution\n(Hydrogen moderator)')
    ax.grid(True, alpha=0.3)

    # Plot 5: 3D random walk (spatial diffusion)
    ax = axes[1, 1]

    # Simulate 3D walk
    mean_free_path = 1.0  # cm (typical for moderator)
    n_walks = 5
    n_steps = 100

    for i in range(n_walks):
        positions = simulate_3d_random_walk(n_steps, mean_free_path, 0.1)
        ax.plot(positions[:, 0], positions[:, 1], '-', lw=1, alpha=0.7)

    ax.plot(0, 0, 'ko', markersize=10, label='Start')
    ax.set_xlabel('x (mean free paths)')
    ax.set_ylabel('y (mean free paths)')
    ax.set_title('Neutron Diffusion\n(3D random walk, projected)')
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Summary
    ax = axes[1, 2]
    ax.axis('off')

    # Calculate values for summary
    xi_H = average_log_decrement(1)
    xi_D = average_log_decrement(2)
    xi_C = average_log_decrement(12)
    n_H = mean_collisions_to_thermalize(E0, E_thermal, 1)
    n_D = mean_collisions_to_thermalize(E0, E_thermal, 2)
    n_C = mean_collisions_to_thermalize(E0, E_thermal, 12)

    summary = f"""
Neutron Moderation Physics
==========================

Elastic Scattering:
  E' = E × (1 + α + (1-α)cos θ_cm)/2
  where α = ((A-1)/(A+1))²

Energy Limits:
  Maximum loss (θ_cm = π): E' = αE
  No energy loss (θ_cm = 0): E' = E

Average Logarithmic Decrement:
  ξ = <ln(E/E')> = 1 - (A-1)²/(2A) × ln((A+1)/(A-1))

Collisions to Thermalize:
  n = ln(E₀/E_th) / ξ

Moderator Comparison:
  Material   A    ξ      n (2 MeV→thermal)
  H         1    {xi_H:.3f}   {n_H:.0f}
  D         2    {xi_D:.3f}   {n_D:.0f}
  C        12    {xi_C:.3f}   {n_C:.0f}

Slowing Down:
  • Fission neutrons: ~2 MeV
  • Thermal energy: ~0.025 eV
  • Need to lose factor of ~10⁸

Good Moderators:
  • Light (low A) → large ξ
  • Low absorption cross section
  • Cheap and available

  H₂O: ξ_eff ≈ 0.92, high σ_abs
  D₂O: ξ_eff ≈ 0.51, low σ_abs
  Graphite: ξ ≈ 0.158, low σ_abs
"""

    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Experiment 212: Neutron Moderation Random Walk\n'
                 'Slowing Down in Nuclear Reactors', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'exp212_neutron_moderation.png'),
                dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/exp212_neutron_moderation.png")


if __name__ == "__main__":
    main()
