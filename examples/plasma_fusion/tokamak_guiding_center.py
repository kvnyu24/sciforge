"""
Experiment 263: Tokamak Guiding Center Motion

Demonstrates charged particle motion in tokamak magnetic geometry
using the guiding center approximation.

Physical concepts:
- Particles gyrate rapidly around field lines
- Guiding center drifts: grad-B, curvature, ExB
- Banana orbits from magnetic trapping
- Neoclassical transport from orbit effects
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.plasma import TokamakEquilibrium

# Physical constants
e = 1.602e-19
m_p = 1.673e-27
m_e = 9.109e-31


def toroidal_to_cartesian(R, Z, phi):
    """Convert tokamak coordinates to Cartesian."""
    X = R * np.cos(phi)
    Y = R * np.sin(phi)
    return X, Y, Z


def grad_B_drift(v_perp, B, grad_B_mag, m, q):
    """Grad-B drift velocity magnitude."""
    return m * v_perp**2 / (2 * q * B**2) * grad_B_mag


def curvature_drift(v_par, R_c, B, m, q):
    """Curvature drift velocity magnitude."""
    return m * v_par**2 / (q * B * R_c)


def simulate_particle_orbit(tokamak, energy_keV, pitch_angle, mass=m_p,
                            charge=e, n_orbits=3):
    """
    Simulate particle guiding center orbit in tokamak.

    Args:
        tokamak: TokamakEquilibrium object
        energy_keV: Particle energy in keV
        pitch_angle: Initial pitch angle (cos(theta) = v_par/v)
        mass: Particle mass
        charge: Particle charge
        n_orbits: Number of poloidal transits

    Returns:
        Trajectory data
    """
    R0 = tokamak.R0
    a = tokamak.a
    B0 = tokamak.B0

    # Initial conditions
    r = 0.3 * a  # Start at r/a = 0.3
    theta = 0.0  # Poloidal angle
    phi = 0.0    # Toroidal angle

    # Particle velocity
    energy = energy_keV * 1e3 * e  # Joules
    v = np.sqrt(2 * energy / mass)
    v_par = v * pitch_angle
    v_perp = v * np.sqrt(1 - pitch_angle**2)

    # Magnetic moment (conserved)
    R = R0 + r * np.cos(theta)
    B_local = B0 * R0 / R
    mu = mass * v_perp**2 / (2 * B_local)

    # Time step
    omega_c = charge * B0 / mass
    dt = 0.01 * 2 * np.pi / omega_c

    # Storage
    trajectory = {'R': [], 'Z': [], 'phi': [], 'v_par': [], 'time': []}

    t = 0
    n_poloidal = 0

    while n_poloidal < n_orbits:
        R = R0 + r * np.cos(theta)
        Z = r * np.sin(theta)

        trajectory['R'].append(R)
        trajectory['Z'].append(Z)
        trajectory['phi'].append(phi)
        trajectory['v_par'].append(v_par)
        trajectory['time'].append(t)

        # Local magnetic field
        B_local = B0 * R0 / R
        B_theta = tokamak.poloidal_field(r) if r > 0 else 0

        # Update v_perp from magnetic moment conservation
        v_perp_sq = 2 * mu * B_local / mass
        if v_perp_sq < 0:
            v_perp_sq = 0

        # Check for mirror bounce
        v_sq = v**2
        if v_perp_sq > v_sq:
            v_par = -v_par  # Mirror reflection
            v_perp_sq = v_sq - v_par**2

        v_perp = np.sqrt(v_perp_sq)

        # Parallel velocity from energy conservation
        v_par_sq = v_sq - v_perp_sq
        if v_par_sq < 0:
            v_par_sq = 0
            v_par = -abs(v_par)
        else:
            v_par = np.sign(v_par) * np.sqrt(v_par_sq)

        # Guiding center drifts
        # Toroidal precession
        v_drift_phi = (v_par**2 + 0.5 * v_perp**2) / (omega_c * R)

        # Parallel motion along field
        q_local = tokamak.safety_factor(r)

        # Update positions
        phi += v_drift_phi * dt / R
        theta += v_par * dt / (q_local * R)

        # Track poloidal transits
        if theta > 2 * np.pi:
            theta -= 2 * np.pi
            n_poloidal += 1
        elif theta < 0:
            theta += 2 * np.pi
            n_poloidal += 1

        # Radial drift (simplified)
        dr = 0  # First order guiding center has no radial drift
        r = max(0.01 * a, min(0.9 * a, r + dr * dt))

        t += dt

    for key in trajectory:
        trajectory[key] = np.array(trajectory[key])

    return trajectory


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Tokamak parameters (ITER-like)
    R0 = 6.2  # Major radius (m)
    a = 2.0   # Minor radius (m)
    B0 = 5.3  # Toroidal field (T)
    Ip = 15e6 # Plasma current (A)

    tokamak = TokamakEquilibrium(R0, a, B0, Ip)

    # Plot 1: Tokamak cross-section with magnetic surfaces
    ax1 = axes[0, 0]

    # Draw flux surfaces
    theta = np.linspace(0, 2 * np.pi, 100)
    r_surfaces = np.linspace(0.1 * a, 0.9 * a, 8)

    for r in r_surfaces:
        R = R0 + r * np.cos(theta)
        Z = r * np.sin(theta)
        ax1.plot(R, Z, 'b-', lw=0.8, alpha=0.5)

    # Draw tokamak outline
    R_out = R0 + a * np.cos(theta)
    Z_out = a * np.sin(theta)
    ax1.plot(R_out, Z_out, 'k-', lw=2)

    # Passing particle orbit
    traj_passing = simulate_particle_orbit(tokamak, energy_keV=10, pitch_angle=0.8)
    ax1.plot(traj_passing['R'], traj_passing['Z'], 'r-', lw=1.5, label='Passing')

    # Trapped (banana) particle orbit
    traj_trapped = simulate_particle_orbit(tokamak, energy_keV=10, pitch_angle=0.3)
    ax1.plot(traj_trapped['R'], traj_trapped['Z'], 'g-', lw=1.5, label='Trapped (banana)')

    ax1.set_xlabel('R (m)')
    ax1.set_ylabel('Z (m)')
    ax1.set_title(f'Tokamak Orbits (R$_0$ = {R0} m, a = {a} m, B$_0$ = {B0} T)')
    ax1.legend()
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # Mark magnetic axis
    ax1.plot(R0, 0, 'ko', markersize=8)
    ax1.text(R0, 0.2, 'Axis', ha='center')

    # Plot 2: Safety factor profile
    ax2 = axes[0, 1]

    r = np.linspace(0, a, 100)
    q = tokamak.safety_factor(r)

    ax2.plot(r / a, q, 'b-', lw=2)
    ax2.axhline(y=1, color='red', linestyle='--', label='q = 1 (kink limit)')
    ax2.axhline(y=2, color='orange', linestyle=':', label='q = 2')

    # Mark edge q
    q_edge = tokamak.edge_safety_factor()
    ax2.plot(1.0, q_edge, 'ko', markersize=10)
    ax2.annotate(f'$q_a$ = {q_edge:.1f}', xy=(1.0, q_edge),
                 xytext=(0.8, q_edge + 0.5), fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='gray'))

    ax2.set_xlabel('Normalized radius r/a')
    ax2.set_ylabel('Safety factor q')
    ax2.set_title('Safety Factor Profile')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, max(q_edge + 1, 4))

    # Plot 3: Particle velocity vs poloidal angle
    ax3 = axes[1, 0]

    ax3.plot(traj_passing['time'] * 1e6, traj_passing['v_par'] / 1e6, 'r-', lw=1.5,
             label='Passing')
    ax3.plot(traj_trapped['time'] * 1e6, traj_trapped['v_par'] / 1e6, 'g-', lw=1.5,
             label='Trapped')

    ax3.axhline(y=0, color='gray', linestyle=':', alpha=0.7)

    ax3.set_xlabel('Time ($\\mu$s)')
    ax3.set_ylabel('Parallel velocity (10$^6$ m/s)')
    ax3.set_title('Parallel Velocity Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Trapping fraction and banana width
    ax4 = axes[1, 1]

    # Trapping fraction as function of radius
    r_norm = np.linspace(0.1, 0.9, 50)
    r_abs = r_norm * a

    # Inverse aspect ratio
    epsilon = r_abs / R0

    # Trapped fraction for Maxwellian
    f_trapped = np.sqrt(2 * epsilon)

    # Banana width
    q_r = tokamak.safety_factor(r_abs)
    rho_pol = 0.003  # Poloidal Larmor radius for 10 keV proton (approximate)
    delta_b = rho_pol * q_r / np.sqrt(epsilon)

    ax4.plot(r_norm, f_trapped, 'b-', lw=2, label='Trapped fraction')
    ax4_twin = ax4.twinx()
    ax4_twin.plot(r_norm, delta_b / a * 100, 'r--', lw=2, label='Banana width')

    ax4.set_xlabel('Normalized radius r/a')
    ax4.set_ylabel('Trapped fraction', color='blue')
    ax4_twin.set_ylabel('Banana width (% of a)', color='red')

    ax4.tick_params(axis='y', labelcolor='blue')
    ax4_twin.tick_params(axis='y', labelcolor='red')

    ax4.set_title('Particle Trapping in Tokamak')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)

    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.suptitle('Experiment 263: Tokamak Guiding Center Motion\n'
                 'Passing and trapped particle orbits in toroidal geometry',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'tokamak_guiding_center.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'tokamak_guiding_center.png')}")


if __name__ == "__main__":
    main()
