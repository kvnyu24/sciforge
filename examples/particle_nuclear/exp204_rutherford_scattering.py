"""
Experiment 204: Rutherford Scattering Angular Distribution

Demonstrates the classical Rutherford scattering formula for charged particle
scattering off a Coulomb potential. Shows how the differential cross section
varies as 1/sin^4(theta/2) and compares with modified screened potentials.

Physics:
- dσ/dΩ = (Z₁Z₂e²/4E)² / sin⁴(θ/2)
- Classical trajectory: hyperbolic orbit
- Impact parameter: b = (a/2)cot(θ/2)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.nuclear import RutherfordScattering


def rutherford_cross_section(theta, Z1, Z2, E_MeV):
    """
    Calculate Rutherford differential cross section.

    Args:
        theta: Scattering angle in radians
        Z1, Z2: Charge numbers
        E_MeV: Kinetic energy in MeV

    Returns:
        dσ/dΩ in fm²/sr
    """
    # Distance of closest approach (fm)
    e2 = 1.44  # MeV·fm (e² in natural units)
    a = Z1 * Z2 * e2 / (2 * E_MeV)

    # Rutherford formula
    return (a / 2)**2 / np.sin(theta / 2)**4


def screened_cross_section(theta, Z1, Z2, E_MeV, screening_angle):
    """
    Screened Rutherford cross section with atomic screening.

    Accounts for finite atomic size screening effects.
    """
    ruth = rutherford_cross_section(theta, Z1, Z2, E_MeV)
    screening_factor = 1 / (1 + (screening_angle / theta)**2)**2
    return ruth * screening_factor


def impact_parameter(theta, Z1, Z2, E_MeV):
    """Calculate classical impact parameter for given scattering angle."""
    e2 = 1.44  # MeV·fm
    a = Z1 * Z2 * e2 / (2 * E_MeV)
    return (a / 2) / np.tan(theta / 2)


def trajectory(b, Z1, Z2, E_MeV, n_points=500):
    """
    Calculate classical hyperbolic trajectory.

    Args:
        b: Impact parameter (fm)
        Z1, Z2: Charge numbers
        E_MeV: Kinetic energy in MeV
        n_points: Number of trajectory points

    Returns:
        x, y coordinates of trajectory
    """
    e2 = 1.44  # MeV·fm
    a = Z1 * Z2 * e2 / (2 * E_MeV)

    # Eccentricity of hyperbola
    epsilon = np.sqrt(1 + (2 * b / a)**2)

    # Scattering angle
    theta_scatter = 2 * np.arctan(a / (2 * b)) if b > 0 else np.pi

    # Asymptotic angle
    theta_asymp = np.pi - theta_scatter / 2

    # Parametric angle
    phi = np.linspace(-theta_asymp + 0.1, theta_asymp - 0.1, n_points)

    # Distance (hyperbolic orbit)
    r = a * (epsilon**2 - 1) / (2 * (1 + epsilon * np.cos(phi)))

    # Cartesian coordinates
    x = r * np.cos(phi)
    y = r * np.sin(phi)

    return x, y


def main():
    # Alpha particle scattering on gold (Geiger-Marsden experiment)
    Z1 = 2       # Alpha particle
    Z2 = 79      # Gold nucleus
    E_MeV = 5.0  # 5 MeV alpha particles

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Differential cross section vs angle
    ax = axes[0, 0]
    theta = np.linspace(0.1, np.pi, 500)
    theta_deg = np.degrees(theta)

    dcs = rutherford_cross_section(theta, Z1, Z2, E_MeV)

    ax.semilogy(theta_deg, dcs, 'b-', lw=2, label='Rutherford')
    ax.semilogy(theta_deg, screened_cross_section(theta, Z1, Z2, E_MeV, 0.05),
                'r--', lw=2, label='Screened')

    ax.set_xlabel('Scattering Angle (degrees)')
    ax.set_ylabel('dσ/dΩ (fm²/sr)')
    ax.set_title(f'Rutherford Cross Section\nα on Au, E = {E_MeV} MeV')
    ax.set_xlim(0, 180)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Angular distribution (polar)
    ax = axes[0, 1]
    ax = plt.subplot(232, projection='polar')

    theta_polar = np.linspace(0.1, np.pi, 100)
    dcs_polar = rutherford_cross_section(theta_polar, Z1, Z2, E_MeV)
    # Normalize for visualization
    dcs_norm = dcs_polar / np.max(dcs_polar)

    ax.plot(theta_polar, dcs_norm**0.2, 'b-', lw=2)  # Power for visualization
    ax.plot(-theta_polar, dcs_norm**0.2, 'b-', lw=2)
    ax.set_title('Angular Distribution\n(dσ/dΩ)^0.2 for visibility', pad=20)

    # Plot 3: Impact parameter vs scattering angle
    ax = axes[0, 2]
    theta_b = np.linspace(0.1, np.pi - 0.1, 100)
    b_values = impact_parameter(theta_b, Z1, Z2, E_MeV)

    ax.plot(np.degrees(theta_b), b_values, 'g-', lw=2)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Scattering Angle (degrees)')
    ax.set_ylabel('Impact Parameter (fm)')
    ax.set_title('Impact Parameter vs Angle')
    ax.grid(True, alpha=0.3)

    # Distance of closest approach for head-on collision
    e2 = 1.44
    d_closest = Z1 * Z2 * e2 / E_MeV
    ax.axhline(y=d_closest/2, color='r', linestyle='--',
               label=f'd_min/2 = {d_closest/2:.1f} fm')
    ax.legend()

    # Plot 4: Classical trajectories
    ax = axes[1, 0]

    # Various impact parameters
    b_values_traj = [10, 20, 40, 80, 150]  # fm
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(b_values_traj)))

    for b, color in zip(b_values_traj, colors):
        x, y = trajectory(b, Z1, Z2, E_MeV)
        ax.plot(x, y, '-', color=color, lw=1.5,
                label=f'b = {b} fm')

    # Mark target nucleus
    ax.plot(0, 0, 'ko', markersize=10)
    circle = plt.Circle((0, 0), 7.0, fill=False, color='k', linestyle='--')
    ax.add_patch(circle)

    ax.set_xlim(-300, 100)
    ax.set_ylim(-150, 150)
    ax.set_xlabel('x (fm)')
    ax.set_ylabel('y (fm)')
    ax.set_title('Classical Hyperbolic Trajectories')
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 5: Energy dependence
    ax = axes[1, 1]

    energies = [2, 5, 10, 20, 50]  # MeV
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(energies)))
    theta_e = np.linspace(5, 175, 100)
    theta_e_rad = np.radians(theta_e)

    for E, color in zip(energies, colors):
        dcs_e = rutherford_cross_section(theta_e_rad, Z1, Z2, E)
        # Normalize to 90 degrees for comparison
        dcs_90 = rutherford_cross_section(np.pi/2, Z1, Z2, E)
        ax.semilogy(theta_e, dcs_e / dcs_90, '-', color=color,
                    lw=2, label=f'E = {E} MeV')

    ax.set_xlabel('Scattering Angle (degrees)')
    ax.set_ylabel('dσ/dΩ normalized to 90°')
    ax.set_title('Energy Independence of Shape\n(Same 1/sin⁴(θ/2) dependence)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Comparison with quantum corrections
    ax = axes[1, 2]

    # Different projectiles at same energy
    projectiles = [
        ('α (Z=2)', 2, 'b'),
        ('C (Z=6)', 6, 'g'),
        ('O (Z=8)', 8, 'r'),
    ]

    theta_p = np.linspace(5, 175, 100)
    theta_p_rad = np.radians(theta_p)

    for name, Z, color in projectiles:
        dcs_p = rutherford_cross_section(theta_p_rad, Z, Z2, 10.0)
        # Normalize
        dcs_p_norm = dcs_p / dcs_p[len(dcs_p)//2]
        ax.semilogy(theta_p, dcs_p, '-', color=color, lw=2, label=name)

    ax.set_xlabel('Scattering Angle (degrees)')
    ax.set_ylabel('dσ/dΩ (fm²/sr)')
    ax.set_title(f'Different Projectiles on Au\nE = 10 MeV')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Experiment 204: Rutherford Scattering Angular Distribution\n'
                 'Classical Coulomb Scattering (Geiger-Marsden)', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'exp204_rutherford_scattering.png'),
                dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/exp204_rutherford_scattering.png")


if __name__ == "__main__":
    main()
