"""
Experiment 87: Solenoid field profile.

This example demonstrates the magnetic field inside and outside a solenoid,
showing the approximately uniform internal field and the rapid decay outside.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt


# Constants
MU_0 = 4 * np.pi * 1e-7  # Permeability of free space (H/m)


def solenoid_field_loop_sum(x, z, R, L, N, I, n_loops=50):
    """
    Calculate solenoid field by summing contributions from N current loops.

    Solenoid axis along z, centered at origin, length L, radius R.

    Args:
        x, z: Field point coordinates (y=0 assumed)
        R: Solenoid radius (m)
        L: Solenoid length (m)
        N: Number of turns
        I: Current (A)
        n_loops: Number of loops to sum (approximation of continuous winding)

    Returns:
        Bx, Bz: Magnetic field components (T)
    """
    # Ensure arrays
    x = np.atleast_1d(x)
    z = np.atleast_1d(z)

    orig_shape = x.shape
    x = x.flatten()
    z = z.flatten()

    Bx = np.zeros_like(x)
    Bz = np.zeros_like(z)

    # Loop positions along solenoid
    z_loops = np.linspace(-L/2, L/2, n_loops)
    dz_loop = L / n_loops

    # Current per loop (total N turns carrying current I)
    I_loop = I * N / n_loops

    for z_loop in z_loops:
        # Field from single loop at z_loop (using approximation for off-axis)
        rho = np.abs(x)  # Distance from axis
        zz = z - z_loop  # Axial distance from loop

        # On-axis contribution (dominant for small rho)
        r2 = R**2 + zz**2
        r = np.sqrt(r2)

        # Bz from single loop
        Bz_loop = MU_0 * I_loop * R**2 / (2 * r**3)

        # Off-axis correction (first-order)
        if np.any(rho > 0):
            # Radial field component (approximate)
            Br_loop = MU_0 * I_loop * R**2 * zz * rho / (2 * r**5) * 3

            # Convert to Bx (assuming y=0)
            Bx += Br_loop * np.sign(x)

        Bz += Bz_loop

    Bx = Bx.reshape(orig_shape)
    Bz = Bz.reshape(orig_shape)

    return Bx, Bz


def solenoid_field_analytical(z, R, L, n, I):
    """
    Analytical on-axis field for a finite solenoid.

    B(z) = (mu_0 * n * I / 2) * [cos(theta1) - cos(theta2)]

    where theta1, theta2 are angles to the solenoid ends.

    Args:
        z: Axial position (m)
        R: Solenoid radius (m)
        L: Solenoid length (m)
        n: Turns per unit length (1/m)
        I: Current (A)

    Returns:
        Bz: Axial field component (T)
    """
    # Angles to ends of solenoid
    z1 = L/2 - z   # Distance to far end
    z2 = -L/2 - z  # Distance to near end (negative if z > -L/2)

    cos_theta1 = z1 / np.sqrt(z1**2 + R**2)
    cos_theta2 = z2 / np.sqrt(z2**2 + R**2)

    return MU_0 * n * I / 2 * (cos_theta1 - cos_theta2)


def infinite_solenoid_field(n, I):
    """Field inside an infinite solenoid: B = mu_0 * n * I"""
    return MU_0 * n * I


def main():
    # Solenoid parameters
    R = 0.05      # Radius: 5 cm
    L = 0.5       # Length: 50 cm
    N = 500       # Total turns
    I = 1.0       # Current: 1 A
    n = N / L     # Turns per unit length

    # Expected field (infinite solenoid limit)
    B_inf = infinite_solenoid_field(n, I)

    fig = plt.figure(figsize=(16, 12))

    # Plot 1: Field in x-z plane
    ax1 = fig.add_subplot(2, 2, 1)

    n_pts = 60
    x = np.linspace(-0.15, 0.15, n_pts)
    z = np.linspace(-0.4, 0.4, n_pts)
    X, Z = np.meshgrid(x, z)

    Bx, Bz = solenoid_field_loop_sum(X, Z, R, L, N, I, n_loops=100)
    B_mag = np.sqrt(Bx**2 + Bz**2)

    # Field lines
    ax1.streamplot(X, Z, Bx, Bz, color=np.log10(B_mag + 1e-10),
                   cmap='viridis', linewidth=1, density=2)

    # Draw solenoid outline
    ax1.add_patch(plt.Rectangle((-R, -L/2), 2*R, L, fill=False,
                                 edgecolor='red', linewidth=2))
    ax1.axhline(y=L/2, xmin=0.5-R/0.3, xmax=0.5+R/0.3, color='red', lw=2)
    ax1.axhline(y=-L/2, xmin=0.5-R/0.3, xmax=0.5+R/0.3, color='red', lw=2)

    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('z (m)')
    ax1.set_title('Magnetic Field Lines Around Solenoid')
    ax1.set_aspect('equal')

    # Plot 2: On-axis field profile
    ax2 = fig.add_subplot(2, 2, 2)

    z_axis = np.linspace(-0.5, 0.5, 200)
    _, Bz_numerical = solenoid_field_loop_sum(np.zeros_like(z_axis), z_axis,
                                               R, L, N, I, n_loops=200)
    Bz_analytical = solenoid_field_analytical(z_axis, R, L, n, I)

    ax2.plot(z_axis * 100, Bz_numerical * 1e3, 'b-', lw=2, label='Numerical')
    ax2.plot(z_axis * 100, Bz_analytical * 1e3, 'r--', lw=2, label='Analytical')
    ax2.axhline(y=B_inf * 1e3, color='gray', linestyle=':', lw=2,
                label=f'Infinite solenoid: {B_inf*1e3:.3f} mT')

    # Mark solenoid ends
    ax2.axvline(x=-L/2 * 100, color='green', linestyle=':', alpha=0.7)
    ax2.axvline(x=L/2 * 100, color='green', linestyle=':', alpha=0.7)
    ax2.axvspan(-L/2 * 100, L/2 * 100, alpha=0.1, color='green')

    ax2.set_xlabel('z position (cm)')
    ax2.set_ylabel('Bz (mT)')
    ax2.set_title('On-Axis Field Profile')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add field uniformity annotation
    B_center = Bz_analytical[len(z_axis)//2]
    B_edge = Bz_analytical[np.argmin(np.abs(z_axis - L/2))]
    uniformity = (B_center - B_edge) / B_center * 100
    ax2.text(0.95, 0.05, f'Field at center: {B_center*1e3:.3f} mT\n'
                         f'Field at edge: {B_edge*1e3:.3f} mT\n'
                         f'Edge drop: {uniformity:.1f}%',
             transform=ax2.transAxes, ha='right', va='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 3: Radial field profile at center
    ax3 = fig.add_subplot(2, 2, 3)

    r_profile = np.linspace(0, 0.15, 100)
    Bx_r, Bz_r = solenoid_field_loop_sum(r_profile, np.zeros_like(r_profile),
                                          R, L, N, I, n_loops=200)
    B_r = np.sqrt(Bx_r**2 + Bz_r**2)

    ax3.plot(r_profile * 100, Bz_r * 1e3, 'b-', lw=2, label='Bz (axial)')
    ax3.plot(r_profile * 100, np.abs(Bx_r) * 1e3, 'r-', lw=2, label='|Bx| (radial)')
    ax3.plot(r_profile * 100, B_r * 1e3, 'k--', lw=2, label='|B| (total)')

    # Mark solenoid radius
    ax3.axvline(x=R * 100, color='green', linestyle=':', lw=2, label='Solenoid radius')

    ax3.set_xlabel('Radial distance from axis (cm)')
    ax3.set_ylabel('Magnetic field (mT)')
    ax3.set_title('Radial Field Profile at z = 0')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Field magnitude map
    ax4 = fig.add_subplot(2, 2, 4)

    B_mag_normalized = B_mag / B_inf
    levels = np.linspace(0, 1.5, 16)

    contour = ax4.contourf(X * 100, Z * 100, B_mag_normalized, levels=levels,
                           cmap='hot_r', extend='max')
    ax4.contour(X * 100, Z * 100, B_mag_normalized, levels=[0.5, 0.9, 0.95, 1.0],
                colors='white', linewidths=1)
    plt.colorbar(contour, ax=ax4, label='|B| / B_infinite')

    # Draw solenoid
    ax4.add_patch(plt.Rectangle((-R*100, -L/2*100), 2*R*100, L*100,
                                 fill=False, edgecolor='cyan', linewidth=2))

    ax4.set_xlabel('x (cm)')
    ax4.set_ylabel('z (cm)')
    ax4.set_title('Normalized Field Magnitude')
    ax4.set_aspect('equal')

    # Add parameter summary
    fig.text(0.5, 0.02,
             f'Solenoid: R = {R*100:.0f} cm, L = {L*100:.0f} cm, '
             f'N = {N} turns, I = {I} A\n'
             f'n = N/L = {n:.0f} turns/m, '
             f'B(infinite) = mu_0*n*I = {B_inf*1e3:.3f} mT',
             ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.suptitle('Solenoid Magnetic Field Profile', fontsize=14, y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)

    # Save plot
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'solenoid_field.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
