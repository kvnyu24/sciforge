"""
Experiment 82: Gauss's law numerical flux integration.

This example demonstrates Gauss's law by numerically computing the electric
flux through various closed surfaces around charge distributions, verifying
that the flux equals Q_enclosed / epsilon_0.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Constants
EPSILON_0 = 8.854e-12  # Permittivity of free space (F/m)
K = 1 / (4 * np.pi * EPSILON_0)  # Coulomb constant


def point_charge_field(x, y, z, q, x0=0, y0=0, z0=0):
    """
    Electric field from a point charge.

    Args:
        x, y, z: Field point coordinates
        q: Charge (C)
        x0, y0, z0: Charge position

    Returns:
        Ex, Ey, Ez: Field components
    """
    rx = x - x0
    ry = y - y0
    rz = z - z0
    r = np.sqrt(rx**2 + ry**2 + rz**2)
    r = np.maximum(r, 1e-10)

    factor = K * q / r**3
    return factor * rx, factor * ry, factor * rz


def calculate_flux_sphere(q_charges, r_sphere, n_theta=50, n_phi=100):
    """
    Calculate electric flux through a spherical surface.

    Args:
        q_charges: List of (q, x, y, z) tuples for each charge
        r_sphere: Radius of the Gaussian sphere
        n_theta: Number of polar angle divisions
        n_phi: Number of azimuthal angle divisions

    Returns:
        flux: Total electric flux through the sphere
        E_dot_dA: Array of flux contributions (for visualization)
    """
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi)
    THETA, PHI = np.meshgrid(theta, phi)

    # Spherical surface points
    X = r_sphere * np.sin(THETA) * np.cos(PHI)
    Y = r_sphere * np.sin(THETA) * np.sin(PHI)
    Z = r_sphere * np.cos(THETA)

    # Outward normal (radial direction on sphere)
    n_x = np.sin(THETA) * np.cos(PHI)
    n_y = np.sin(THETA) * np.sin(PHI)
    n_z = np.cos(THETA)

    # Calculate total field at each surface point
    Ex_total = np.zeros_like(X)
    Ey_total = np.zeros_like(Y)
    Ez_total = np.zeros_like(Z)

    for q, x0, y0, z0 in q_charges:
        Ex, Ey, Ez = point_charge_field(X, Y, Z, q, x0, y0, z0)
        Ex_total += Ex
        Ey_total += Ey
        Ez_total += Ez

    # E dot n (normal component)
    E_dot_n = Ex_total * n_x + Ey_total * n_y + Ez_total * n_z

    # Surface element dA = r^2 sin(theta) dtheta dphi
    dtheta = theta[1] - theta[0]
    dphi = phi[1] - phi[0]
    dA = r_sphere**2 * np.sin(THETA) * dtheta * dphi

    # Flux integral
    E_dot_dA = E_dot_n * dA
    flux = np.sum(E_dot_dA)

    return flux, E_dot_dA, X, Y, Z


def calculate_flux_cube(q_charges, half_side, n_points=50):
    """
    Calculate electric flux through a cubic surface.

    Args:
        q_charges: List of (q, x, y, z) tuples
        half_side: Half the side length of the cube
        n_points: Number of points per edge

    Returns:
        flux: Total flux through all faces
    """
    a = half_side
    x = np.linspace(-a, a, n_points)
    y = np.linspace(-a, a, n_points)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dA = dx * dy

    flux_total = 0

    # Six faces: +x, -x, +y, -y, +z, -z
    faces = [
        (a, 'x', +1),   # +x face
        (-a, 'x', -1),  # -x face
        (a, 'y', +1),   # +y face
        (-a, 'y', -1),  # -y face
        (a, 'z', +1),   # +z face
        (-a, 'z', -1),  # -z face
    ]

    for pos, axis, sign in faces:
        X, Y = np.meshgrid(x, y)

        if axis == 'x':
            X_face = pos * np.ones_like(X)
            Y_face = X
            Z_face = Y
            n_vec = (sign, 0, 0)
        elif axis == 'y':
            X_face = X
            Y_face = pos * np.ones_like(X)
            Z_face = Y
            n_vec = (0, sign, 0)
        else:  # z
            X_face = X
            Y_face = Y
            Z_face = pos * np.ones_like(X)
            n_vec = (0, 0, sign)

        # Calculate field
        Ex_total = np.zeros_like(X_face)
        Ey_total = np.zeros_like(Y_face)
        Ez_total = np.zeros_like(Z_face)

        for q, x0, y0, z0 in q_charges:
            Ex, Ey, Ez = point_charge_field(X_face, Y_face, Z_face, q, x0, y0, z0)
            Ex_total += Ex
            Ey_total += Ey
            Ez_total += Ez

        # E dot n
        E_dot_n = Ex_total * n_vec[0] + Ey_total * n_vec[1] + Ez_total * n_vec[2]

        flux_total += np.sum(E_dot_n * dA)

    return flux_total


def main():
    fig = plt.figure(figsize=(16, 12))

    # Test case 1: Single point charge
    q1 = 1e-9  # 1 nC
    charges_single = [(q1, 0, 0, 0)]

    # Plot 1: Flux vs sphere radius (single charge at center)
    ax1 = fig.add_subplot(2, 2, 1)

    radii = np.linspace(0.1, 2, 20)
    fluxes = []
    theoretical_flux = q1 / EPSILON_0

    for r in radii:
        flux, _, _, _, _ = calculate_flux_sphere(charges_single, r)
        fluxes.append(flux)

    ax1.axhline(y=theoretical_flux, color='r', linestyle='--', lw=2,
                label=f'Theory: Q/e0 = {theoretical_flux:.2f} V*m')
    ax1.plot(radii, fluxes, 'bo-', lw=2, markersize=8, label='Numerical flux')

    ax1.set_xlabel('Sphere Radius (m)')
    ax1.set_ylabel('Electric Flux (V*m)')
    ax1.set_title('Gauss\'s Law: Flux vs Gaussian Surface Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Error analysis
    error = np.abs(np.array(fluxes) - theoretical_flux) / theoretical_flux * 100
    ax1.text(0.95, 0.05, f'Max error: {np.max(error):.2f}%',
             transform=ax1.transAxes, ha='right', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: 3D visualization of flux density on sphere
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')

    flux, E_dot_dA, X, Y, Z = calculate_flux_sphere(charges_single, 0.5)

    # Color by flux density
    colors = plt.cm.RdBu_r((E_dot_dA / np.max(np.abs(E_dot_dA)) + 1) / 2)

    ax2.plot_surface(X, Y, Z, facecolors=colors, alpha=0.8)
    ax2.scatter([0], [0], [0], c='red', s=100, marker='o', label='Charge')

    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')
    ax2.set_zlabel('z (m)')
    ax2.set_title('Flux Density on Gaussian Sphere\n(red = outward, blue = inward)')

    # Plot 3: Multiple charges - flux depends on enclosed charge
    ax3 = fig.add_subplot(2, 2, 3)

    q2 = 2e-9  # 2 nC
    q3 = -1e-9  # -1 nC

    # Different charge configurations
    configs = [
        ('Single +1nC at origin', [(1e-9, 0, 0, 0)]),
        ('Single +2nC at origin', [(2e-9, 0, 0, 0)]),
        ('Dipole (+1nC, -1nC)', [(1e-9, 0.1, 0, 0), (-1e-9, -0.1, 0, 0)]),
        ('+1nC at origin, +1nC outside', [(1e-9, 0, 0, 0), (1e-9, 2, 0, 0)]),
        ('Three charges', [(1e-9, 0, 0, 0), (1e-9, 0.1, 0, 0), (-1e-9, -0.1, 0, 0)]),
    ]

    r_gaussian = 0.5  # Gaussian sphere radius
    bar_positions = range(len(configs))
    theoretical_values = []
    numerical_values = []

    for name, charges in configs:
        flux, _, _, _, _ = calculate_flux_sphere(charges, r_gaussian)
        numerical_values.append(flux)

        # Calculate enclosed charge
        q_enclosed = sum(q for q, x, y, z in charges
                        if np.sqrt(x**2 + y**2 + z**2) < r_gaussian)
        theoretical_values.append(q_enclosed / EPSILON_0)

    width = 0.35
    ax3.bar([p - width/2 for p in bar_positions], theoretical_values, width,
            label='Q_enc / e0', color='blue', alpha=0.7)
    ax3.bar([p + width/2 for p in bar_positions], numerical_values, width,
            label='Numerical flux', color='red', alpha=0.7)

    ax3.set_xticks(bar_positions)
    ax3.set_xticklabels([c[0] for c in configs], rotation=15, ha='right', fontsize=8)
    ax3.set_ylabel('Electric Flux (V*m)')
    ax3.set_title(f'Gauss\'s Law: Different Charge Configurations\n(Gaussian sphere r = {r_gaussian} m)')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Comparison of sphere vs cube Gaussian surfaces
    ax4 = fig.add_subplot(2, 2, 4)

    sizes = np.linspace(0.2, 1.5, 15)
    flux_spheres = []
    flux_cubes = []

    for size in sizes:
        flux_s, _, _, _, _ = calculate_flux_sphere(charges_single, size)
        flux_c = calculate_flux_cube(charges_single, size)
        flux_spheres.append(flux_s)
        flux_cubes.append(flux_c)

    ax4.axhline(y=theoretical_flux, color='gray', linestyle='--', lw=2,
                label=f'Theory: {theoretical_flux:.2f} V*m')
    ax4.plot(sizes, flux_spheres, 'bo-', lw=2, label='Spherical surface')
    ax4.plot(sizes, flux_cubes, 'rs-', lw=2, label='Cubic surface')

    ax4.set_xlabel('Surface Size (m)')
    ax4.set_ylabel('Electric Flux (V*m)')
    ax4.set_title('Flux Through Different Gaussian Surface Shapes')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Add summary text
    fig.text(0.5, 0.02,
             r"Gauss's Law: $\Phi_E = \oint \vec{E} \cdot d\vec{A} = Q_{enclosed}/\epsilon_0$",
             ha='center', fontsize=12, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.suptitle('Numerical Verification of Gauss\'s Law', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)

    # Save plot
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'gauss_law_flux.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
