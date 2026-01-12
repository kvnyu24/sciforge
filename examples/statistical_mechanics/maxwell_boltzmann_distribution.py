"""
Experiment 133: Maxwell-Boltzmann Speed Distribution

This example demonstrates the Maxwell-Boltzmann speed distribution for an ideal gas
using molecular dynamics simulation and compares with the theoretical distribution.

The Maxwell-Boltzmann speed distribution is:
f(v) = 4*pi * (m / (2*pi*k_B*T))^(3/2) * v^2 * exp(-m*v^2 / (2*k_B*T))

where:
- m = molecular mass
- k_B = Boltzmann constant
- T = temperature
- v = speed
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

# Physical constants
k_B = 1.380649e-23  # Boltzmann constant (J/K)
m_Ar = 6.6335e-26   # Mass of argon atom (kg)


def maxwell_boltzmann_pdf(v, m, T):
    """
    Maxwell-Boltzmann speed distribution probability density function.

    Args:
        v: Speed values (m/s)
        m: Particle mass (kg)
        T: Temperature (K)

    Returns:
        Probability density at each speed
    """
    prefactor = 4 * np.pi * (m / (2 * np.pi * k_B * T))**(3/2)
    return prefactor * v**2 * np.exp(-m * v**2 / (2 * k_B * T))


def maxwell_boltzmann_cdf_inv(u, m, T):
    """
    Generate Maxwell-Boltzmann distributed speeds using rejection sampling.

    Args:
        u: Uniform random samples
        m: Particle mass (kg)
        T: Temperature (K)

    Returns:
        Array of speeds following Maxwell-Boltzmann distribution
    """
    # Characteristic speed
    v_p = np.sqrt(2 * k_B * T / m)  # Most probable speed

    # Use rejection sampling with envelope
    speeds = []
    max_pdf = maxwell_boltzmann_pdf(v_p, m, T)

    while len(speeds) < len(u):
        # Generate candidate speeds (up to 5*v_p covers essentially all distribution)
        v_candidate = np.random.uniform(0, 5 * v_p, len(u))
        y = np.random.uniform(0, 1.1 * max_pdf, len(u))

        # Accept if below the PDF
        accept = y < maxwell_boltzmann_pdf(v_candidate, m, T)
        speeds.extend(v_candidate[accept])

    return np.array(speeds[:len(u)])


def generate_mb_velocities(n_particles, m, T):
    """
    Generate 3D velocity vectors following Maxwell-Boltzmann distribution.

    Each velocity component follows a Gaussian with variance = k_B*T/m.

    Args:
        n_particles: Number of particles
        m: Particle mass (kg)
        T: Temperature (K)

    Returns:
        Array of shape (n_particles, 3) with velocity components
    """
    sigma = np.sqrt(k_B * T / m)
    velocities = np.random.normal(0, sigma, (n_particles, 3))
    return velocities


def simple_md_simulation(n_particles, T, n_steps, dt):
    """
    Simple MD simulation of ideal gas in a periodic box.
    Particles interact only via elastic collisions with walls.

    Args:
        n_particles: Number of particles
        T: Temperature (K)
        n_steps: Number of simulation steps
        dt: Time step (s)

    Returns:
        Final speeds of all particles
    """
    m = m_Ar
    box_size = 1e-8  # 10 nm box

    # Initialize positions uniformly
    positions = np.random.uniform(0, box_size, (n_particles, 3))

    # Initialize velocities from Maxwell-Boltzmann
    velocities = generate_mb_velocities(n_particles, m, T)

    # Simple simulation with periodic boundary conditions
    for _ in range(n_steps):
        # Update positions
        positions += velocities * dt

        # Apply periodic boundary conditions
        positions = np.mod(positions, box_size)

    # Return speeds
    speeds = np.linalg.norm(velocities, axis=1)
    return speeds


def main():
    # Simulation parameters
    n_particles = 10000
    T = 300  # Temperature (K) - room temperature
    n_steps = 100
    dt = 1e-15  # 1 femtosecond

    print("Maxwell-Boltzmann Speed Distribution")
    print("=" * 50)
    print(f"Number of particles: {n_particles}")
    print(f"Temperature: {T} K")
    print(f"Particle mass (Ar): {m_Ar:.4e} kg")

    # Theoretical characteristic speeds
    v_p = np.sqrt(2 * k_B * T / m_Ar)  # Most probable speed
    v_mean = np.sqrt(8 * k_B * T / (np.pi * m_Ar))  # Mean speed
    v_rms = np.sqrt(3 * k_B * T / m_Ar)  # RMS speed

    print(f"\nTheoretical speeds for Ar at {T} K:")
    print(f"  Most probable speed (v_p): {v_p:.1f} m/s")
    print(f"  Mean speed <v>: {v_mean:.1f} m/s")
    print(f"  RMS speed (v_rms): {v_rms:.1f} m/s")

    # Generate speeds from Maxwell-Boltzmann distribution directly
    velocities = generate_mb_velocities(n_particles, m_Ar, T)
    speeds_direct = np.linalg.norm(velocities, axis=1)

    # Run MD simulation
    speeds_md = simple_md_simulation(n_particles, T, n_steps, dt)

    # Compute statistics from simulation
    print(f"\nSimulation results:")
    print(f"  Mean speed: {np.mean(speeds_direct):.1f} m/s (expected: {v_mean:.1f})")
    print(f"  RMS speed: {np.sqrt(np.mean(speeds_direct**2)):.1f} m/s (expected: {v_rms:.1f})")

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Speed range for theoretical curve
    v_range = np.linspace(0, 5 * v_p, 500)
    pdf_theory = maxwell_boltzmann_pdf(v_range, m_Ar, T)

    # Plot 1: Speed distribution comparison
    ax1 = axes[0, 0]
    ax1.hist(speeds_direct, bins=60, density=True, alpha=0.7,
             label='Generated velocities', color='steelblue', edgecolor='white')
    ax1.plot(v_range, pdf_theory, 'r-', lw=2, label='Maxwell-Boltzmann theory')
    ax1.axvline(v_p, color='green', linestyle='--', lw=1.5, label=f'$v_p$ = {v_p:.0f} m/s')
    ax1.axvline(v_mean, color='orange', linestyle='--', lw=1.5, label=f'$<v>$ = {v_mean:.0f} m/s')
    ax1.axvline(v_rms, color='purple', linestyle='--', lw=1.5, label=f'$v_{{rms}}$ = {v_rms:.0f} m/s')
    ax1.set_xlabel('Speed (m/s)', fontsize=12)
    ax1.set_ylabel('Probability Density', fontsize=12)
    ax1.set_title(f'Maxwell-Boltzmann Speed Distribution\nArgon at T = {T} K', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1600)

    # Plot 2: Velocity component distributions
    ax2 = axes[0, 1]
    sigma = np.sqrt(k_B * T / m_Ar)
    v_comp_range = np.linspace(-1500, 1500, 500)
    gaussian_theory = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-v_comp_range**2 / (2 * sigma**2))

    for i, (comp, label, color) in enumerate(zip(velocities.T, ['$v_x$', '$v_y$', '$v_z$'],
                                                   ['red', 'green', 'blue'])):
        ax2.hist(comp, bins=60, density=True, alpha=0.3, label=label, color=color)
    ax2.plot(v_comp_range, gaussian_theory, 'k-', lw=2, label='Gaussian theory')
    ax2.set_xlabel('Velocity component (m/s)', fontsize=12)
    ax2.set_ylabel('Probability Density', fontsize=12)
    ax2.set_title('Velocity Component Distributions\n(Each follows Gaussian)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Temperature dependence
    ax3 = axes[1, 0]
    temperatures = [100, 200, 300, 500, 800]
    colors = plt.cm.hot(np.linspace(0.2, 0.8, len(temperatures)))

    for T_i, color in zip(temperatures, colors):
        pdf_i = maxwell_boltzmann_pdf(v_range, m_Ar, T_i)
        ax3.plot(v_range, pdf_i, lw=2, color=color, label=f'T = {T_i} K')

    ax3.set_xlabel('Speed (m/s)', fontsize=12)
    ax3.set_ylabel('Probability Density', fontsize=12)
    ax3.set_title('Temperature Dependence of Speed Distribution', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 2000)

    # Plot 4: Q-Q plot for verification
    ax4 = axes[1, 1]
    speeds_sorted = np.sort(speeds_direct)
    n = len(speeds_sorted)

    # Theoretical quantiles using rejection sampling
    theoretical_speeds = maxwell_boltzmann_cdf_inv(np.random.random(n), m_Ar, T)
    theoretical_sorted = np.sort(theoretical_speeds)

    ax4.scatter(theoretical_sorted[::100], speeds_sorted[::100], alpha=0.5, s=20)
    max_v = max(theoretical_sorted.max(), speeds_sorted.max())
    ax4.plot([0, max_v], [0, max_v], 'r--', lw=2, label='Perfect agreement')
    ax4.set_xlabel('Theoretical Quantiles (m/s)', fontsize=12)
    ax4.set_ylabel('Sample Quantiles (m/s)', fontsize=12)
    ax4.set_title('Q-Q Plot: Sample vs Theory', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal')

    plt.suptitle('Maxwell-Boltzmann Speed Distribution Analysis', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'maxwell_boltzmann_distribution.png'),
                dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {os.path.join(output_dir, 'maxwell_boltzmann_distribution.png')}")


if __name__ == "__main__":
    main()
