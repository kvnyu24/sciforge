"""
Experiment 90: Mutual inductance of coupled coils.

This example demonstrates the mutual inductance between two coupled coils,
computing M numerically via flux linkage and showing how M depends on
geometry and orientation. Results are compared to analytical formulas.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Physical constants
MU_0 = 4 * np.pi * 1e-7  # Permeability of free space (H/m)


def magnetic_field_loop(x, y, z, R, I, n_phi=100):
    """
    Calculate magnetic field from a circular current loop at origin in xy-plane.

    Uses Biot-Savart law numerically.

    Args:
        x, y, z: Field point coordinates (m)
        R: Loop radius (m)
        I: Current (A)
        n_phi: Number of integration points

    Returns:
        Bx, By, Bz: Magnetic field components (T)
    """
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi

    # Current element positions on loop
    x_loop = R * np.cos(phi)
    y_loop = R * np.sin(phi)
    z_loop = np.zeros_like(phi)

    # dl vector (tangent to loop)
    dlx = -R * np.sin(phi) * dphi
    dly = R * np.cos(phi) * dphi
    dlz = np.zeros_like(phi)

    # Vector from source to field point
    rx = x - x_loop
    ry = y - y_loop
    rz = z - z_loop

    r_mag = np.sqrt(rx**2 + ry**2 + rz**2)
    r_mag = np.maximum(r_mag, 1e-10)  # Avoid division by zero

    # Biot-Savart: dB = (mu_0 * I / 4pi) * (dl x r) / |r|^3
    # Cross product dl x r
    cross_x = dly * rz - dlz * ry
    cross_y = dlz * rx - dlx * rz
    cross_z = dlx * ry - dly * rx

    factor = MU_0 * I / (4 * np.pi * r_mag**3)

    Bx = np.sum(factor * cross_x)
    By = np.sum(factor * cross_y)
    Bz = np.sum(factor * cross_z)

    return Bx, By, Bz


def magnetic_field_loop_on_axis(z, R, I):
    """
    Analytical on-axis magnetic field from a current loop.

    B_z = mu_0 * I * R^2 / (2 * (R^2 + z^2)^(3/2))

    Args:
        z: Axial distance from loop center (m)
        R: Loop radius (m)
        I: Current (A)

    Returns:
        Bz: Axial magnetic field (T)
    """
    return MU_0 * I * R**2 / (2 * (R**2 + z**2)**1.5)


def flux_through_loop(R_secondary, z_offset, R_primary, I_primary, n_r=20, n_phi=20, n_integration=100):
    """
    Calculate magnetic flux through a secondary loop due to primary loop current.

    Primary loop at z=0 in xy-plane, secondary loop at z=z_offset, parallel.

    Args:
        R_secondary: Radius of secondary loop (m)
        z_offset: Axial separation between loops (m)
        R_primary: Radius of primary loop (m)
        I_primary: Current in primary loop (A)
        n_r: Radial integration points
        n_phi: Angular integration points
        n_integration: Biot-Savart integration points

    Returns:
        Phi: Magnetic flux through secondary loop (Wb)
    """
    # Integrate B_z over the area of secondary loop
    r_vals = np.linspace(0, R_secondary, n_r)
    phi_vals = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

    flux = 0.0
    for r in r_vals[1:]:  # Skip r=0
        dr = R_secondary / n_r
        dphi = 2 * np.pi / n_phi
        for phi in phi_vals:
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            z = z_offset

            _, _, Bz = magnetic_field_loop(x, y, z, R_primary, I_primary, n_integration)
            dA = r * dr * dphi
            flux += Bz * dA

    return flux


def mutual_inductance_neumann(R1, R2, d, n_phi=100):
    """
    Calculate mutual inductance using Neumann formula.

    M = (mu_0 / 4pi) * integral integral (dl1 . dl2) / |r12|

    For coaxial circular loops, this has an analytical solution in terms
    of elliptic integrals.

    Args:
        R1: Radius of first loop (m)
        R2: Radius of second loop (m)
        d: Axial separation (m)
        n_phi: Number of integration points

    Returns:
        M: Mutual inductance (H)
    """
    phi1 = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    phi2 = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi

    M = 0.0
    for p1 in phi1:
        for p2 in phi2:
            # Positions on loops
            x1, y1, z1 = R1 * np.cos(p1), R1 * np.sin(p1), 0
            x2, y2, z2 = R2 * np.cos(p2), R2 * np.sin(p2), d

            # dl vectors
            dl1 = np.array([-R1 * np.sin(p1), R1 * np.cos(p1), 0]) * dphi
            dl2 = np.array([-R2 * np.sin(p2), R2 * np.cos(p2), 0]) * dphi

            # Separation
            r12 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
            r12 = max(r12, 1e-10)

            # Neumann formula contribution
            M += np.dot(dl1, dl2) / r12

    return MU_0 * M / (4 * np.pi)


def mutual_inductance_analytical_coaxial(R1, R2, d):
    """
    Analytical mutual inductance for coaxial loops using elliptic integrals.

    M = mu_0 * sqrt(R1 * R2) * ((2/k - k) * K(k) - (2/k) * E(k))

    where k^2 = 4*R1*R2 / ((R1 + R2)^2 + d^2)
    K, E are complete elliptic integrals of first and second kind.

    Args:
        R1: Radius of first loop (m)
        R2: Radius of second loop (m)
        d: Axial separation (m)

    Returns:
        M: Mutual inductance (H)
    """
    from scipy.special import ellipk, ellipe

    k2 = 4 * R1 * R2 / ((R1 + R2)**2 + d**2)
    k = np.sqrt(k2)

    if k >= 1:
        return np.inf

    K = ellipk(k2)
    E = ellipe(k2)

    M = MU_0 * np.sqrt(R1 * R2) * ((2/k - k) * K - 2/k * E)

    return M


def main():
    fig = plt.figure(figsize=(16, 12))

    # Plot 1: Mutual inductance vs separation for coaxial loops
    ax1 = fig.add_subplot(2, 2, 1)

    R1 = 0.05  # 5 cm primary loop
    R2 = 0.05  # 5 cm secondary loop
    I = 1.0    # 1 A current

    d_range = np.linspace(0.01, 0.3, 30)  # 1 cm to 30 cm separation

    M_numerical = []
    M_analytical = []

    for d in d_range:
        # Numerical via flux linkage: M = Phi / I
        flux = flux_through_loop(R2, d, R1, I, n_r=15, n_phi=20, n_integration=80)
        M_numerical.append(flux / I)

        # Analytical formula
        M_analytical.append(mutual_inductance_analytical_coaxial(R1, R2, d))

    M_numerical = np.array(M_numerical)
    M_analytical = np.array(M_analytical)

    ax1.semilogy(d_range * 100, M_analytical * 1e6, 'b-', lw=2, label='Analytical (elliptic)')
    ax1.semilogy(d_range * 100, M_numerical * 1e6, 'ro', markersize=6, label='Numerical (flux)')

    ax1.set_xlabel('Axial Separation d (cm)')
    ax1.set_ylabel('Mutual Inductance M (uH)')
    ax1.set_title(f'Mutual Inductance vs Separation\nR1 = R2 = {R1*100:.0f} cm')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Effect of radius ratio
    ax2 = fig.add_subplot(2, 2, 2)

    d_fixed = 0.05  # 5 cm separation
    R1_fixed = 0.05  # 5 cm primary

    R2_ratios = np.linspace(0.2, 3.0, 30)
    R2_range = R1_fixed * R2_ratios

    M_vs_ratio = []
    for R2 in R2_range:
        M = mutual_inductance_analytical_coaxial(R1_fixed, R2, d_fixed)
        M_vs_ratio.append(M)

    M_vs_ratio = np.array(M_vs_ratio)

    ax2.plot(R2_ratios, M_vs_ratio * 1e6, 'b-', lw=2)
    ax2.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)

    ax2.set_xlabel('Radius Ratio R2/R1')
    ax2.set_ylabel('Mutual Inductance M (uH)')
    ax2.set_title(f'Effect of Radius Ratio\nR1 = {R1_fixed*100:.0f} cm, d = {d_fixed*100:.0f} cm')
    ax2.grid(True, alpha=0.3)

    # Find and mark maximum
    idx_max = np.argmax(M_vs_ratio)
    ax2.plot(R2_ratios[idx_max], M_vs_ratio[idx_max] * 1e6, 'ro', markersize=10)
    ax2.text(R2_ratios[idx_max], M_vs_ratio[idx_max] * 1e6 * 1.1,
             f'Max at R2/R1 = {R2_ratios[idx_max]:.2f}', ha='center')

    # Plot 3: Effect of orientation (tilted secondary)
    ax3 = fig.add_subplot(2, 2, 3)

    d_fixed = 0.1  # 10 cm separation
    R = 0.05       # 5 cm radius for both loops

    tilt_angles = np.linspace(0, 90, 19)  # 0 to 90 degrees

    # For tilted loops, flux depends on cos(theta) for small tilt
    # More accurately, we need to compute the flux through the tilted loop

    M_vs_tilt = []
    for theta_deg in tilt_angles:
        theta = np.radians(theta_deg)
        # Approximate: M(theta) approx M(0) * cos(theta) for small tilts
        # More accurate for larger tilts requires 3D integration

        # Compute flux through tilted loop numerically
        n_points = 15
        r_vals = np.linspace(0, R, n_points)
        phi_vals = np.linspace(0, 2 * np.pi, 20, endpoint=False)

        flux = 0.0
        for r in r_vals[1:]:
            dr = R / n_points
            dphi = 2 * np.pi / 20
            for phi in phi_vals:
                # Position on tilted secondary loop (tilted around x-axis)
                x_local = r * np.cos(phi)
                y_local = r * np.sin(phi) * np.cos(theta)
                z_local = d_fixed + r * np.sin(phi) * np.sin(theta)

                Bx, By, Bz = magnetic_field_loop(x_local, y_local, z_local, R, I, 60)

                # Normal vector to tilted loop
                normal = np.array([0, -np.sin(theta), np.cos(theta)])
                B_vec = np.array([Bx, By, Bz])

                # Flux through area element
                dA = r * dr * dphi
                flux += np.dot(B_vec, normal) * dA

        M_vs_tilt.append(flux / I)

    M_vs_tilt = np.array(M_vs_tilt)

    ax3.plot(tilt_angles, M_vs_tilt * 1e6, 'b-', lw=2, label='Numerical')
    ax3.plot(tilt_angles, M_vs_tilt[0] * np.cos(np.radians(tilt_angles)) * 1e6,
             'r--', lw=2, label=r'$M_0 \cos\theta$')

    ax3.set_xlabel('Tilt Angle (degrees)')
    ax3.set_ylabel('Mutual Inductance M (uH)')
    ax3.set_title(f'Effect of Secondary Loop Tilt\nR = {R*100:.0f} cm, d = {d_fixed*100:.0f} cm')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Coupling coefficient k = M / sqrt(L1 * L2)
    ax4 = fig.add_subplot(2, 2, 4)

    # Self-inductance of a circular loop (approximate): L = mu_0 * R * (ln(8R/a) - 2)
    # where a is wire radius. For thin wire, use L approx mu_0 * R

    R = 0.05  # 5 cm
    a = 0.001  # 1 mm wire radius

    L_self = MU_0 * R * (np.log(8 * R / a) - 2)

    d_range = np.linspace(0.001, 0.3, 50)
    k_coupling = []

    for d in d_range:
        M = mutual_inductance_analytical_coaxial(R, R, d)
        k = M / L_self  # For identical loops, L1 = L2
        k_coupling.append(k)

    k_coupling = np.array(k_coupling)

    ax4.plot(d_range * 100, k_coupling, 'b-', lw=2)
    ax4.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='k = 1 (perfect coupling)')

    ax4.set_xlabel('Separation d (cm)')
    ax4.set_ylabel('Coupling Coefficient k')
    ax4.set_title(f'Coupling Coefficient k = M/sqrt(L1*L2)\nR = {R*100:.0f} cm, wire radius = {a*1000:.0f} mm')
    ax4.set_ylim(0, 1.1)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Add self-inductance annotation
    ax4.text(0.95, 0.5, f'Self-inductance:\nL = {L_self*1e6:.2f} uH',
             transform=ax4.transAxes, ha='right', va='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Add physics summary
    fig.text(0.5, 0.02,
             r'Mutual Inductance: $M = \Phi_{12}/I_1$ = '
             r'$\frac{\mu_0}{4\pi}\oint\oint \frac{d\vec{l}_1 \cdot d\vec{l}_2}{|\vec{r}_{12}|}$'
             + '\n' +
             r'Coaxial loops: $M = \mu_0\sqrt{R_1 R_2}\left[\left(\frac{2}{k}-k\right)K(k^2) - \frac{2}{k}E(k^2)\right]$',
             ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.suptitle('Mutual Inductance of Coupled Coils', fontsize=14, y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)

    # Save plot
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'mutual_inductance.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
