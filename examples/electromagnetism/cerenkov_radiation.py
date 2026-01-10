"""
Experiment 98: Cerenkov cone.

This example demonstrates Cerenkov radiation, which occurs when a
charged particle travels faster than the phase velocity of light
in a dielectric medium.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Physical constants
C = 2.998e8          # Speed of light in vacuum (m/s)
Q_E = 1.602e-19      # Elementary charge (C)
M_E = 9.109e-31      # Electron mass (kg)


def cerenkov_angle(n, beta):
    """
    Calculate Cerenkov angle.

    cos(theta_c) = 1 / (n * beta)

    where n is refractive index and beta = v/c.

    Args:
        n: Refractive index
        beta: Particle velocity / c

    Returns:
        theta_c: Cerenkov angle (rad), or None if below threshold
    """
    if n * beta <= 1:
        return None  # Below threshold
    cos_theta = 1 / (n * beta)
    if cos_theta > 1:
        return None
    return np.arccos(cos_theta)


def cerenkov_threshold_velocity(n):
    """
    Calculate threshold velocity for Cerenkov radiation.

    v_threshold = c / n

    Args:
        n: Refractive index

    Returns:
        beta_threshold: v/c threshold
    """
    return 1 / n


def cerenkov_energy_threshold(n, m):
    """
    Calculate threshold kinetic energy for Cerenkov radiation.

    gamma_threshold = 1 / sqrt(1 - 1/n^2)

    Args:
        n: Refractive index
        m: Particle mass (kg)

    Returns:
        KE_threshold: Threshold kinetic energy (J)
    """
    gamma = 1 / np.sqrt(1 - 1/n**2)
    return (gamma - 1) * m * C**2


def cerenkov_photons_per_length(n, beta, lambda1, lambda2):
    """
    Number of Cerenkov photons emitted per unit length in wavelength range.

    dN/dx = (2*pi*alpha) * sin^2(theta_c) * (1/lambda1 - 1/lambda2)

    where alpha is fine structure constant.

    Args:
        n: Refractive index
        beta: v/c
        lambda1, lambda2: Wavelength range (m)

    Returns:
        dN_dx: Photons per meter
    """
    alpha = 1 / 137  # Fine structure constant
    theta = cerenkov_angle(n, beta)
    if theta is None:
        return 0

    return 2 * np.pi * alpha * np.sin(theta)**2 * (1/lambda1 - 1/lambda2)


def main():
    fig = plt.figure(figsize=(16, 12))

    # Common materials
    materials = {
        'Water': 1.33,
        'Glass': 1.5,
        'Diamond': 2.42,
        'Air (STP)': 1.0003,
        'Aerogel': 1.03,
    }

    # Plot 1: Cerenkov cone geometry
    ax1 = fig.add_subplot(2, 2, 1)

    # Particle trajectory
    n_water = materials['Water']
    beta = 0.9
    theta_c = cerenkov_angle(n_water, beta)

    # Draw particle path
    x_particle = np.linspace(0, 5, 100)
    ax1.plot(x_particle, np.zeros_like(x_particle), 'b-', lw=3, label='Particle path')

    # Draw Cerenkov cone (wavefronts)
    for x0 in np.linspace(0.5, 4, 8):
        # Cone from point x0
        t_elapsed = (5 - x0) / (beta * C)  # Time since particle was at x0
        r_wave = (C / n_water) * t_elapsed  # Radius of wavefront

        # Tangent line from current position to circle at x0
        x_current = 5
        if theta_c is not None:
            y_cone = (x_current - x0) * np.tan(theta_c)
            ax1.plot([x0, x_current], [0, y_cone], 'r-', lw=1, alpha=0.5)
            ax1.plot([x0, x_current], [0, -y_cone], 'r-', lw=1, alpha=0.5)

    # Draw final cone edge
    if theta_c is not None:
        y_max = 5 * np.tan(theta_c)
        ax1.plot([0, 5], [0, y_max], 'r-', lw=2, label=f'Cerenkov cone (theta = {np.degrees(theta_c):.1f}deg)')
        ax1.plot([0, 5], [0, -y_max], 'r-', lw=2)

    ax1.annotate(r'$\theta_c$', xy=(4, 0.5), fontsize=14, color='red')
    ax1.arrow(4.5, 0, 0.4, 0, head_width=0.1, head_length=0.05, fc='blue', ec='blue')
    ax1.text(4.7, -0.3, 'v', fontsize=12, color='blue')

    ax1.set_xlabel('Distance (arb. units)')
    ax1.set_ylabel('Transverse distance')
    ax1.set_title(f'Cerenkov Cone Geometry (n = {n_water}, beta = {beta})')
    ax1.legend()
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Cerenkov angle vs velocity
    ax2 = fig.add_subplot(2, 2, 2)

    beta_range = np.linspace(0, 1, 500)

    for name, n in materials.items():
        beta_threshold = cerenkov_threshold_velocity(n)
        angles = []
        for b in beta_range:
            theta = cerenkov_angle(n, b)
            angles.append(np.degrees(theta) if theta is not None else np.nan)

        ax2.plot(beta_range, angles, lw=2, label=f'{name} (n={n})')
        ax2.axvline(x=beta_threshold, linestyle=':', alpha=0.3)

    ax2.set_xlabel('Particle velocity (beta = v/c)')
    ax2.set_ylabel('Cerenkov angle (degrees)')
    ax2.set_title('Cerenkov Angle vs Velocity')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 90)

    # Plot 3: Threshold energy for different particles and media
    ax3 = fig.add_subplot(2, 2, 3)

    particles = [
        ('Electron', M_E),
        ('Muon', 207 * M_E),
        ('Pion', 273 * M_E),
        ('Proton', 1836 * M_E),
    ]

    n_range = np.linspace(1.01, 2.5, 100)

    for name, mass in particles:
        KE_threshold = np.array([cerenkov_energy_threshold(n, mass) for n in n_range])
        # Convert to MeV
        KE_MeV = KE_threshold / (Q_E * 1e6)
        ax3.semilogy(n_range, KE_MeV, lw=2, label=name)

    # Mark common materials
    for mat_name, n in materials.items():
        if n > 1.01:
            ax3.axvline(x=n, linestyle=':', alpha=0.5, color='gray')
            ax3.text(n, 1e4, mat_name, rotation=90, fontsize=8, va='bottom')

    ax3.set_xlabel('Refractive index n')
    ax3.set_ylabel('Threshold kinetic energy (MeV)')
    ax3.set_title('Cerenkov Threshold Energy')
    ax3.legend()
    ax3.grid(True, alpha=0.3, which='both')

    # Plot 4: 3D cone visualization
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')

    n = 1.33  # Water
    beta = 0.95
    theta_c = cerenkov_angle(n, beta)

    if theta_c is not None:
        # Cone surface
        z = np.linspace(0, 5, 50)
        phi = np.linspace(0, 2*np.pi, 100)
        Z, PHI = np.meshgrid(z, phi)

        R = Z * np.tan(theta_c)
        X = R * np.cos(PHI)
        Y = R * np.sin(PHI)

        ax4.plot_surface(X, Y, Z, alpha=0.5, cmap='Blues')

        # Particle trajectory
        ax4.plot([0, 0], [0, 0], [0, 5], 'r-', lw=3, label='Particle')

        # Light rays
        for phi_ray in np.linspace(0, 2*np.pi, 12, endpoint=False):
            z_ray = np.linspace(0, 5, 50)
            r_ray = z_ray * np.tan(theta_c)
            x_ray = r_ray * np.cos(phi_ray)
            y_ray = r_ray * np.sin(phi_ray)
            ax4.plot(x_ray, y_ray, z_ray, 'b-', lw=0.5, alpha=0.5)

    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z (particle direction)')
    ax4.set_title('3D Cerenkov Cone')

    # Add physics summary
    fig.text(0.5, 0.02,
             r'Cerenkov Radiation: $\cos\theta_c = \frac{1}{n\beta}$, '
             r'Threshold: $\beta > 1/n$ (particle faster than light in medium)' + '\n'
             r'$\frac{dN}{dx} \propto \sin^2\theta_c \propto (1 - 1/(n\beta)^2)$ photons/length',
             ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.suptitle('Cerenkov Radiation', fontsize=14, y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)

    # Save plot
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'cerenkov_radiation.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
