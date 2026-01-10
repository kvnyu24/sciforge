"""
Experiment 134: Equipartition Theorem Check via Molecular Dynamics

This example verifies the equipartition theorem using a molecular dynamics
simulation. The theorem states that each quadratic degree of freedom
contributes (1/2)*k_B*T to the average energy.

For a particle with 3 translational degrees of freedom:
<E_kinetic> = (3/2)*k_B*T

Each velocity component contributes:
<(1/2)*m*v_i^2> = (1/2)*k_B*T

The simulation uses a Lennard-Jones potential between particles.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt

# Physical constants (using reduced units)
# In reduced units: epsilon = 1, sigma = 1, m = 1, k_B = 1


def lennard_jones_force(r, epsilon=1.0, sigma=1.0):
    """
    Calculate Lennard-Jones force magnitude.

    F(r) = 24*epsilon/r * [2*(sigma/r)^12 - (sigma/r)^6]

    Args:
        r: Distance between particles
        epsilon: Depth of potential well
        sigma: Distance at which potential is zero

    Returns:
        Force magnitude (positive = repulsive)
    """
    sr6 = (sigma / r)**6
    sr12 = sr6**2
    return 24 * epsilon / r * (2 * sr12 - sr6)


def lennard_jones_potential(r, epsilon=1.0, sigma=1.0):
    """Calculate Lennard-Jones potential energy."""
    sr6 = (sigma / r)**6
    sr12 = sr6**2
    return 4 * epsilon * (sr12 - sr6)


def compute_forces(positions, box_size, r_cut=2.5):
    """
    Compute forces on all particles using Lennard-Jones potential.
    Uses minimum image convention for periodic boundary conditions.

    Args:
        positions: (N, 3) array of particle positions
        box_size: Size of cubic simulation box
        r_cut: Cutoff distance for interactions

    Returns:
        forces: (N, 3) array of forces on each particle
        potential_energy: Total potential energy
    """
    n_particles = len(positions)
    forces = np.zeros_like(positions)
    potential_energy = 0.0

    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            # Minimum image convention
            rij = positions[j] - positions[i]
            rij = rij - box_size * np.round(rij / box_size)

            r = np.linalg.norm(rij)

            if r < r_cut and r > 0.1:  # Avoid singularity
                # Force
                f_mag = lennard_jones_force(r)
                f_vec = f_mag * rij / r

                forces[i] -= f_vec
                forces[j] += f_vec

                # Potential energy
                potential_energy += lennard_jones_potential(r)

    return forces, potential_energy


def velocity_verlet_step(positions, velocities, forces, box_size, dt, mass=1.0):
    """
    Perform one velocity Verlet integration step.

    Args:
        positions: Current positions
        velocities: Current velocities
        forces: Current forces
        box_size: Simulation box size
        dt: Time step
        mass: Particle mass

    Returns:
        new_positions, new_velocities, new_forces, potential_energy
    """
    # Half-step velocity update
    velocities_half = velocities + 0.5 * dt * forces / mass

    # Full position update
    new_positions = positions + dt * velocities_half

    # Apply periodic boundary conditions
    new_positions = np.mod(new_positions, box_size)

    # Compute new forces
    new_forces, potential_energy = compute_forces(new_positions, box_size)

    # Complete velocity update
    new_velocities = velocities_half + 0.5 * dt * new_forces / mass

    return new_positions, new_velocities, new_forces, potential_energy


def initialize_system(n_particles, box_size, temperature, mass=1.0):
    """
    Initialize particle positions and velocities.

    Args:
        n_particles: Number of particles
        box_size: Size of cubic box
        temperature: Initial temperature (k_B = 1)
        mass: Particle mass

    Returns:
        positions, velocities
    """
    # Initialize on a simple cubic lattice
    n_side = int(np.ceil(n_particles**(1/3)))
    spacing = box_size / n_side

    positions = []
    for i in range(n_side):
        for j in range(n_side):
            for k in range(n_side):
                if len(positions) < n_particles:
                    positions.append([
                        (i + 0.5) * spacing,
                        (j + 0.5) * spacing,
                        (k + 0.5) * spacing
                    ])
    positions = np.array(positions)

    # Initialize velocities from Maxwell-Boltzmann distribution
    sigma = np.sqrt(temperature / mass)
    velocities = np.random.normal(0, sigma, (n_particles, 3))

    # Remove center of mass velocity
    velocities -= np.mean(velocities, axis=0)

    # Scale to exact temperature
    kinetic_energy = 0.5 * mass * np.sum(velocities**2)
    current_temp = 2 * kinetic_energy / (3 * n_particles)
    velocities *= np.sqrt(temperature / current_temp)

    return positions, velocities


def berendsen_thermostat(velocities, current_temp, target_temp, tau=0.1, dt=0.001):
    """
    Apply Berendsen thermostat for temperature control.

    Args:
        velocities: Current velocities
        current_temp: Current temperature
        target_temp: Target temperature
        tau: Coupling time constant
        dt: Time step

    Returns:
        Scaled velocities
    """
    if current_temp > 0:
        lambda_factor = np.sqrt(1 + (dt / tau) * (target_temp / current_temp - 1))
        return velocities * lambda_factor
    return velocities


def run_md_simulation(n_particles, target_temp, n_equilibration, n_production, dt=0.002):
    """
    Run molecular dynamics simulation.

    Args:
        n_particles: Number of particles
        target_temp: Target temperature (reduced units)
        n_equilibration: Number of equilibration steps
        n_production: Number of production steps
        dt: Time step

    Returns:
        Dictionary with simulation results
    """
    # System setup
    density = 0.5  # Reduced density
    box_size = (n_particles / density)**(1/3)

    print(f"  Box size: {box_size:.2f}")
    print(f"  Density: {density:.3f}")

    # Initialize system
    positions, velocities = initialize_system(n_particles, box_size, target_temp)
    forces, _ = compute_forces(positions, box_size)

    # Storage for observables
    kinetic_energies = []
    potential_energies = []
    temperatures = []
    velocity_components = {'vx': [], 'vy': [], 'vz': []}
    kinetic_components = {'x': [], 'y': [], 'z': []}

    # Equilibration with thermostat
    print("  Equilibrating...")
    for step in range(n_equilibration):
        positions, velocities, forces, pe = velocity_verlet_step(
            positions, velocities, forces, box_size, dt
        )

        # Compute temperature
        ke = 0.5 * np.sum(velocities**2)
        current_temp = 2 * ke / (3 * n_particles)

        # Apply thermostat
        velocities = berendsen_thermostat(velocities, current_temp, target_temp,
                                          tau=0.5, dt=dt)

    # Production run (no thermostat for proper sampling)
    print("  Production run...")
    for step in range(n_production):
        positions, velocities, forces, pe = velocity_verlet_step(
            positions, velocities, forces, box_size, dt
        )

        # Compute observables
        ke = 0.5 * np.sum(velocities**2)
        current_temp = 2 * ke / (3 * n_particles)

        # Store data
        kinetic_energies.append(ke)
        potential_energies.append(pe)
        temperatures.append(current_temp)

        # Store velocity components for analysis
        velocity_components['vx'].extend(velocities[:, 0])
        velocity_components['vy'].extend(velocities[:, 1])
        velocity_components['vz'].extend(velocities[:, 2])

        # Store kinetic energy per component
        kinetic_components['x'].append(0.5 * np.sum(velocities[:, 0]**2))
        kinetic_components['y'].append(0.5 * np.sum(velocities[:, 1]**2))
        kinetic_components['z'].append(0.5 * np.sum(velocities[:, 2]**2))

    return {
        'kinetic_energies': np.array(kinetic_energies),
        'potential_energies': np.array(potential_energies),
        'temperatures': np.array(temperatures),
        'velocity_components': velocity_components,
        'kinetic_components': kinetic_components,
        'n_particles': n_particles,
        'target_temp': target_temp
    }


def main():
    print("Equipartition Theorem Check via Molecular Dynamics")
    print("=" * 50)

    # Simulation parameters
    n_particles = 64
    target_temps = [0.5, 1.0, 1.5, 2.0]  # Reduced units
    n_equilibration = 2000
    n_production = 5000

    results = {}

    for T in target_temps:
        print(f"\nSimulation at T* = {T}")
        results[T] = run_md_simulation(n_particles, T, n_equilibration, n_production)

    # Analysis and plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Temperature equilibration
    ax1 = axes[0, 0]
    for T in target_temps:
        temps = results[T]['temperatures']
        ax1.plot(temps, alpha=0.7, label=f'T* = {T}')
        ax1.axhline(T, linestyle='--', color='gray', alpha=0.5)
    ax1.set_xlabel('Time step', fontsize=12)
    ax1.set_ylabel('Temperature (reduced units)', fontsize=12)
    ax1.set_title('Temperature During Production Run', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Equipartition check - KE per particle vs T
    ax2 = axes[0, 1]
    measured_temps = []
    ke_per_particle = []
    theoretical_ke = []

    for T in target_temps:
        avg_ke = np.mean(results[T]['kinetic_energies'])
        n = results[T]['n_particles']
        measured_temps.append(np.mean(results[T]['temperatures']))
        ke_per_particle.append(avg_ke / n)
        theoretical_ke.append(1.5 * T)  # (3/2)k_B*T with k_B=1

    ax2.scatter(target_temps, ke_per_particle, s=100, c='blue',
                label='Simulation', zorder=5)
    ax2.plot(target_temps, theoretical_ke, 'r-', lw=2,
             label='Equipartition: (3/2)T')
    ax2.set_xlabel('Target Temperature', fontsize=12)
    ax2.set_ylabel('<KE> per particle', fontsize=12)
    ax2.set_title('Equipartition Theorem Verification\n<KE> = (3/2)k_B T', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Velocity component distributions at T*=1
    ax3 = axes[1, 0]
    T_test = 1.0
    vx = np.array(results[T_test]['velocity_components']['vx'])
    vy = np.array(results[T_test]['velocity_components']['vy'])
    vz = np.array(results[T_test]['velocity_components']['vz'])

    v_range = np.linspace(-4, 4, 200)
    gaussian = (1 / np.sqrt(2 * np.pi * T_test)) * np.exp(-v_range**2 / (2 * T_test))

    ax3.hist(vx, bins=50, density=True, alpha=0.4, label='$v_x$')
    ax3.hist(vy, bins=50, density=True, alpha=0.4, label='$v_y$')
    ax3.hist(vz, bins=50, density=True, alpha=0.4, label='$v_z$')
    ax3.plot(v_range, gaussian, 'k-', lw=2, label='Maxwell-Boltzmann')
    ax3.set_xlabel('Velocity component', fontsize=12)
    ax3.set_ylabel('Probability density', fontsize=12)
    ax3.set_title(f'Velocity Distributions at T* = {T_test}', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Energy per degree of freedom
    ax4 = axes[1, 1]

    for T in target_temps:
        ke_x = np.array(results[T]['kinetic_components']['x'])
        ke_y = np.array(results[T]['kinetic_components']['y'])
        ke_z = np.array(results[T]['kinetic_components']['z'])
        n = results[T]['n_particles']

        # Energy per particle per direction
        ke_x_avg = np.mean(ke_x) / n
        ke_y_avg = np.mean(ke_y) / n
        ke_z_avg = np.mean(ke_z) / n

        x_pos = T
        width = 0.1
        ax4.bar(x_pos - width, ke_x_avg, width=width, color='red', alpha=0.7)
        ax4.bar(x_pos, ke_y_avg, width=width, color='green', alpha=0.7)
        ax4.bar(x_pos + width, ke_z_avg, width=width, color='blue', alpha=0.7)

    # Theoretical line
    T_line = np.linspace(0.3, 2.2, 100)
    ax4.plot(T_line, 0.5 * T_line, 'k--', lw=2, label='Equipartition: (1/2)T')

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='x-component'),
        Patch(facecolor='green', alpha=0.7, label='y-component'),
        Patch(facecolor='blue', alpha=0.7, label='z-component'),
        plt.Line2D([0], [0], color='black', linestyle='--', label='(1/2)T')
    ]
    ax4.legend(handles=legend_elements)
    ax4.set_xlabel('Temperature', fontsize=12)
    ax4.set_ylabel('<KE> per particle per direction', fontsize=12)
    ax4.set_title('Equipartition: Energy per Degree of Freedom', fontsize=12)
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Equipartition Theorem Verification via Molecular Dynamics\n'
                 '(Lennard-Jones fluid in reduced units)', fontsize=14, y=1.02)
    plt.tight_layout()

    # Print numerical results
    print("\n" + "=" * 50)
    print("Equipartition Theorem Results")
    print("=" * 50)
    print(f"{'T_target':>10} {'T_measured':>12} {'<KE>/N':>12} {'(3/2)T':>12} {'Error %':>10}")
    print("-" * 60)
    for T in target_temps:
        T_meas = np.mean(results[T]['temperatures'])
        ke_n = np.mean(results[T]['kinetic_energies']) / n_particles
        theory = 1.5 * T
        error = 100 * abs(ke_n - theory) / theory
        print(f"{T:>10.2f} {T_meas:>12.3f} {ke_n:>12.4f} {theory:>12.4f} {error:>10.2f}")

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'equipartition_md.png'),
                dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {os.path.join(output_dir, 'equipartition_md.png')}")


if __name__ == "__main__":
    main()
