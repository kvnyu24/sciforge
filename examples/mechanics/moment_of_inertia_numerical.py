"""
Example demonstrating numerical integration for moment of inertia.

The moment of inertia (rotational inertia) is calculated as:
    I = integral(r^2 * dm) = integral(r^2 * rho * dV)

where r is the perpendicular distance from the rotation axis.

For complex shapes, analytical solutions may not exist, and numerical
integration is required. This example demonstrates:

1. Monte Carlo integration for arbitrary 3D shapes
2. Numerical integration using quadrature for 2D shapes
3. Comparison with analytical results for standard shapes
4. Parallel axis theorem verification
5. Principal moments of inertia calculation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def monte_carlo_moment_of_inertia(shape_func, bounds, density, axis, n_samples=100000):
    """
    Calculate moment of inertia using Monte Carlo integration.

    Args:
        shape_func: Function that returns True if point is inside shape
        bounds: ((x_min, x_max), (y_min, y_max), (z_min, z_max))
        density: Mass density (uniform)
        axis: Rotation axis ('x', 'y', 'z', or unit vector)
        n_samples: Number of random samples

    Returns:
        Tuple of (moment_of_inertia, mass, uncertainty)
    """
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = bounds
    volume_box = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)

    # Generate random points
    points = np.random.uniform(
        [x_min, y_min, z_min],
        [x_max, y_max, z_max],
        (n_samples, 3)
    )

    # Check which points are inside the shape
    inside = np.array([shape_func(p) for p in points])
    points_inside = points[inside]

    if len(points_inside) == 0:
        return 0.0, 0.0, 0.0

    # Calculate perpendicular distance from axis
    if isinstance(axis, str):
        if axis == 'x':
            axis_vec = np.array([1, 0, 0])
        elif axis == 'y':
            axis_vec = np.array([0, 1, 0])
        else:  # 'z'
            axis_vec = np.array([0, 0, 1])
    else:
        axis_vec = np.array(axis) / np.linalg.norm(axis)

    # For each point, calculate perpendicular distance to axis
    # r_perp = |r - (r . axis_vec) * axis_vec|
    projections = np.outer(np.dot(points_inside, axis_vec), axis_vec)
    r_perp_vec = points_inside - projections
    r_perp_sq = np.sum(r_perp_vec**2, axis=1)

    # Volume fraction and mass
    volume_fraction = len(points_inside) / n_samples
    volume = volume_fraction * volume_box
    mass = density * volume

    # Moment of inertia
    # I = rho * integral(r^2 dV) = rho * V * <r^2>
    r_sq_mean = np.mean(r_perp_sq)
    I = density * volume * r_sq_mean

    # Estimate uncertainty (statistical)
    r_sq_std = np.std(r_perp_sq) / np.sqrt(len(points_inside))
    I_uncertainty = density * volume * r_sq_std

    return I, mass, I_uncertainty


def quadrature_moment_of_inertia_2d(shape_func, bounds, thickness, density, axis='z', n_points=200):
    """
    Calculate moment of inertia using 2D quadrature for a flat shape.

    Args:
        shape_func: Function that returns True if (x, y) is inside shape
        bounds: ((x_min, x_max), (y_min, y_max))
        thickness: Thickness of the shape (z-direction)
        density: Mass density
        axis: Rotation axis (only 'z' for flat shapes about z-axis)
        n_points: Number of grid points per dimension

    Returns:
        Tuple of (moment_of_inertia, mass)
    """
    (x_min, x_max), (y_min, y_max) = bounds

    x = np.linspace(x_min, x_max, n_points)
    y = np.linspace(y_min, y_max, n_points)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    X, Y = np.meshgrid(x, y)

    # Check which points are inside
    inside = np.zeros_like(X, dtype=bool)
    for i in range(n_points):
        for j in range(n_points):
            inside[i, j] = shape_func((X[i, j], Y[i, j]))

    # For z-axis rotation, r^2 = x^2 + y^2
    R_sq = X**2 + Y**2

    # Area and mass
    area = np.sum(inside) * dx * dy
    mass = density * area * thickness

    # Moment of inertia
    I = density * thickness * np.sum(R_sq[inside]) * dx * dy

    return I, mass


def full_inertia_tensor(shape_func, bounds, density, n_samples=100000):
    """
    Calculate full 3x3 inertia tensor using Monte Carlo.

    I_ij = integral(rho * (delta_ij * r^2 - r_i * r_j) dV)

    Args:
        shape_func: Function returning True if point is inside shape
        bounds: Bounding box
        density: Mass density
        n_samples: Number of samples

    Returns:
        3x3 inertia tensor, mass
    """
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = bounds
    volume_box = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)

    # Generate random points
    points = np.random.uniform(
        [x_min, y_min, z_min],
        [x_max, y_max, z_max],
        (n_samples, 3)
    )

    # Check which points are inside
    inside = np.array([shape_func(p) for p in points])
    points_inside = points[inside]

    if len(points_inside) == 0:
        return np.zeros((3, 3)), 0.0

    volume_fraction = len(points_inside) / n_samples
    volume = volume_fraction * volume_box
    mass = density * volume

    # Calculate inertia tensor components
    x, y, z = points_inside[:, 0], points_inside[:, 1], points_inside[:, 2]
    r_sq = x**2 + y**2 + z**2

    I_tensor = np.zeros((3, 3))

    # Diagonal elements
    I_tensor[0, 0] = np.mean(y**2 + z**2)
    I_tensor[1, 1] = np.mean(x**2 + z**2)
    I_tensor[2, 2] = np.mean(x**2 + y**2)

    # Off-diagonal elements (negative of products)
    I_tensor[0, 1] = I_tensor[1, 0] = -np.mean(x * y)
    I_tensor[0, 2] = I_tensor[2, 0] = -np.mean(x * z)
    I_tensor[1, 2] = I_tensor[2, 1] = -np.mean(y * z)

    I_tensor *= density * volume

    return I_tensor, mass


def principal_moments(I_tensor):
    """
    Calculate principal moments of inertia and principal axes.

    Returns:
        principal_moments: Eigenvalues (sorted)
        principal_axes: Eigenvectors (columns)
    """
    eigenvalues, eigenvectors = np.linalg.eigh(I_tensor)
    # Sort by magnitude
    idx = np.argsort(eigenvalues)
    return eigenvalues[idx], eigenvectors[:, idx]


def main():
    np.random.seed(42)  # For reproducibility

    fig = plt.figure(figsize=(18, 12))

    # --- Comparison with analytical results ---
    ax1 = fig.add_subplot(2, 3, 1)

    # Solid sphere
    R = 0.1  # radius
    rho = 1000  # density (kg/m^3)
    M_analytical = 4/3 * np.pi * R**3 * rho
    I_analytical_sphere = 2/5 * M_analytical * R**2

    def sphere(p):
        return p[0]**2 + p[1]**2 + p[2]**2 <= R**2

    bounds_sphere = ((-R, R), (-R, R), (-R, R))

    samples_list = [1000, 5000, 10000, 50000, 100000, 500000]
    I_computed = []
    uncertainties = []

    for n in samples_list:
        I, M, uncertainty = monte_carlo_moment_of_inertia(
            sphere, bounds_sphere, rho, 'z', n_samples=n
        )
        I_computed.append(I)
        uncertainties.append(uncertainty)

    I_computed = np.array(I_computed)
    uncertainties = np.array(uncertainties)

    ax1.errorbar(samples_list, I_computed / I_analytical_sphere,
                 yerr=uncertainties / I_analytical_sphere,
                 fmt='bo-', capsize=3, label='Monte Carlo')
    ax1.axhline(y=1.0, color='red', linestyle='--', lw=2, label='Analytical')
    ax1.set_xscale('log')
    ax1.set_xlabel('Number of Samples')
    ax1.set_ylabel('I_computed / I_analytical')
    ax1.set_title('Monte Carlo Convergence (Solid Sphere)\n'
                  f'I_analytical = {I_analytical_sphere:.6f} kg*m^2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.9, 1.1)

    # --- Different shapes comparison ---
    ax2 = fig.add_subplot(2, 3, 2)

    shapes = {
        'Solid Sphere': {
            'func': lambda p, R=R: p[0]**2 + p[1]**2 + p[2]**2 <= R**2,
            'bounds': ((-R, R), (-R, R), (-R, R)),
            'I_theory': lambda M, R: 2/5 * M * R**2
        },
        'Solid Cylinder': {
            'func': lambda p, R=R, H=2*R: p[0]**2 + p[1]**2 <= R**2 and abs(p[2]) <= H/2,
            'bounds': ((-R, R), (-R, R), (-R, R)),
            'I_theory': lambda M, R: 1/2 * M * R**2
        },
        'Hollow Cylinder': {
            'func': lambda p, R_out=R, R_in=R*0.7, H=2*R: (
                R_in**2 <= p[0]**2 + p[1]**2 <= R_out**2 and abs(p[2]) <= H/2
            ),
            'bounds': ((-R, R), (-R, R), (-R, R)),
            'I_theory': lambda M, R, R_in=R*0.7: 1/2 * M * (R**2 + R_in**2)
        },
        'Cube': {
            'func': lambda p, L=2*R: abs(p[0]) <= L/2 and abs(p[1]) <= L/2 and abs(p[2]) <= L/2,
            'bounds': ((-R, R), (-R, R), (-R, R)),
            'I_theory': lambda M, L=2*R: 1/6 * M * L**2
        }
    }

    names = []
    ratios = []
    I_computed_shapes = []
    I_theory_shapes = []

    for name, shape in shapes.items():
        I, M, _ = monte_carlo_moment_of_inertia(
            shape['func'], shape['bounds'], rho, 'z', n_samples=100000
        )
        I_theory = shape['I_theory'](M, R)

        names.append(name)
        ratios.append(I / I_theory if I_theory > 0 else 0)
        I_computed_shapes.append(I)
        I_theory_shapes.append(I_theory)

    x_pos = np.arange(len(names))
    colors = ['blue', 'green', 'orange', 'red']

    bars = ax2.bar(x_pos, ratios, color=colors, alpha=0.7)
    ax2.axhline(y=1.0, color='black', linestyle='--', lw=2)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(names, rotation=15, ha='right')
    ax2.set_ylabel('I_computed / I_theory')
    ax2.set_title('Monte Carlo vs Analytical (100,000 samples)')
    ax2.set_ylim(0.95, 1.05)
    ax2.grid(True, alpha=0.3, axis='y')

    # --- Complex shape: T-beam ---
    ax3 = fig.add_subplot(2, 3, 3)

    # T-beam dimensions
    web_height = 0.1
    web_width = 0.02
    flange_width = 0.08
    flange_height = 0.02

    def t_beam(p):
        x, y, z = p
        # Web
        in_web = abs(x) <= web_width/2 and 0 <= y <= web_height and abs(z) <= 0.05
        # Flange
        in_flange = abs(x) <= flange_width/2 and web_height <= y <= web_height + flange_height and abs(z) <= 0.05
        return in_web or in_flange

    bounds_t = ((-flange_width, flange_width),
                (-0.01, web_height + flange_height + 0.01),
                (-0.06, 0.06))

    # Calculate I about different axes
    I_x, M_t, _ = monte_carlo_moment_of_inertia(t_beam, bounds_t, rho, 'x', n_samples=200000)
    I_y, _, _ = monte_carlo_moment_of_inertia(t_beam, bounds_t, rho, 'y', n_samples=200000)
    I_z, _, _ = monte_carlo_moment_of_inertia(t_beam, bounds_t, rho, 'z', n_samples=200000)

    axes_labels = ['X (horizontal)', 'Y (vertical)', 'Z (length)']
    I_values = [I_x, I_y, I_z]

    ax3.bar(axes_labels, I_values, color=['red', 'green', 'blue'], alpha=0.7)
    ax3.set_ylabel('Moment of Inertia (kg*m^2)')
    ax3.set_title(f'T-Beam Moments of Inertia\n'
                  f'Mass = {M_t:.4f} kg')
    ax3.grid(True, alpha=0.3, axis='y')

    # Add T-beam visualization
    ax3_inset = ax3.inset_axes([0.6, 0.5, 0.35, 0.45])
    # Draw T-beam cross-section
    web = plt.Rectangle((-web_width/2, 0), web_width, web_height, fill=True, color='blue', alpha=0.5)
    flange = plt.Rectangle((-flange_width/2, web_height), flange_width, flange_height, fill=True, color='blue', alpha=0.5)
    ax3_inset.add_patch(web)
    ax3_inset.add_patch(flange)
    ax3_inset.set_xlim(-0.05, 0.05)
    ax3_inset.set_ylim(-0.01, 0.13)
    ax3_inset.set_aspect('equal')
    ax3_inset.set_title('Cross-section', fontsize=8)

    # --- Parallel axis theorem verification ---
    ax4 = fig.add_subplot(2, 3, 4)

    # Disk about center vs offset axis
    disk_R = 0.1
    disk_thickness = 0.01

    def disk(p):
        return p[0]**2 + p[1]**2 <= disk_R**2 and abs(p[2]) <= disk_thickness/2

    bounds_disk = ((-disk_R, disk_R), (-disk_R, disk_R), (-disk_thickness, disk_thickness))

    # I about z-axis through center
    I_center, M_disk, _ = monte_carlo_moment_of_inertia(disk, bounds_disk, rho, 'z', n_samples=100000)

    # I about offset z-axes
    offsets = [0, 0.02, 0.04, 0.06, 0.08, 0.1]
    I_offset_numerical = []
    I_offset_theory = []

    for d in offsets:
        # Shift the disk
        def disk_shifted(p, d=d):
            return (p[0]-d)**2 + p[1]**2 <= disk_R**2 and abs(p[2]) <= disk_thickness/2

        bounds_shifted = ((-disk_R + d, disk_R + d), (-disk_R, disk_R), (-disk_thickness, disk_thickness))
        I_off, _, _ = monte_carlo_moment_of_inertia(disk_shifted, bounds_shifted, rho, 'z', n_samples=50000)
        I_offset_numerical.append(I_off)

        # Parallel axis theorem: I_offset = I_center + M * d^2
        I_offset_theory.append(I_center + M_disk * d**2)

    ax4.plot(offsets, I_offset_numerical, 'bo-', markersize=8, label='Monte Carlo')
    ax4.plot(offsets, I_offset_theory, 'r--', lw=2, label='Parallel Axis Theorem')
    ax4.set_xlabel('Offset Distance (m)')
    ax4.set_ylabel('Moment of Inertia (kg*m^2)')
    ax4.set_title('Parallel Axis Theorem Verification\n'
                  f'I_offset = I_center + M*d^2')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # --- Principal axes for asymmetric object ---
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')

    # L-shaped object
    def L_shape(p):
        x, y, z = p
        vertical = abs(x) <= 0.01 and 0 <= y <= 0.1 and abs(z) <= 0.02
        horizontal = 0 <= x <= 0.08 and abs(y) <= 0.01 and abs(z) <= 0.02
        return vertical or horizontal

    bounds_L = ((-0.02, 0.1), (-0.02, 0.12), (-0.03, 0.03))

    I_tensor, M_L = full_inertia_tensor(L_shape, bounds_L, rho, n_samples=200000)
    principal_I, principal_axes = principal_moments(I_tensor)

    # Visualize L-shape
    n_vis = 5000
    points = np.random.uniform(
        [bounds_L[0][0], bounds_L[1][0], bounds_L[2][0]],
        [bounds_L[0][1], bounds_L[1][1], bounds_L[2][1]],
        (n_vis, 3)
    )
    inside = np.array([L_shape(p) for p in points])
    points_inside = points[inside]

    ax5.scatter(points_inside[:, 0]*100, points_inside[:, 1]*100, points_inside[:, 2]*100,
                c='blue', alpha=0.3, s=1)

    # Draw principal axes
    cm = np.mean(points_inside, axis=0) * 100  # Center of mass (approx)
    colors = ['red', 'green', 'blue']
    for i in range(3):
        axis = principal_axes[:, i] * 5  # Scale for visibility
        ax5.quiver(cm[0], cm[1], cm[2], axis[0], axis[1], axis[2],
                   color=colors[i], arrow_length_ratio=0.2, lw=2,
                   label=f'I{i+1} = {principal_I[i]:.6f}')

    ax5.set_xlabel('X (cm)')
    ax5.set_ylabel('Y (cm)')
    ax5.set_zlabel('Z (cm)')
    ax5.set_title('L-Shape with Principal Axes')
    ax5.legend(fontsize=8)

    # --- Convergence analysis ---
    ax6 = fig.add_subplot(2, 3, 6)

    # Show how error scales with sample size
    n_trials = 10
    sample_sizes = [100, 500, 1000, 5000, 10000, 50000]

    means = []
    stds = []

    for n in sample_sizes:
        I_trials = []
        for _ in range(n_trials):
            I, _, _ = monte_carlo_moment_of_inertia(sphere, bounds_sphere, rho, 'z', n_samples=n)
            I_trials.append(I)
        means.append(np.mean(I_trials))
        stds.append(np.std(I_trials))

    # Plot relative error
    rel_error = np.array(stds) / I_analytical_sphere

    ax6.loglog(sample_sizes, rel_error, 'bo-', markersize=8, label='Measured std')
    ax6.loglog(sample_sizes, 1/np.sqrt(sample_sizes), 'r--', lw=2, label='1/sqrt(N)')
    ax6.set_xlabel('Number of Samples')
    ax6.set_ylabel('Relative Standard Deviation')
    ax6.set_title('Monte Carlo Convergence Rate\n'
                  'Error ~ 1/sqrt(N)')
    ax6.legend()
    ax6.grid(True, alpha=0.3, which='both')

    plt.suptitle('Numerical Integration for Moment of Inertia', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'moment_of_inertia_numerical.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'moment_of_inertia_numerical.png')}")

    # Print results
    print("\nMoment of Inertia Numerical Integration Results:")
    print("-" * 60)
    print(f"\nSolid Sphere (R = {R*100:.0f} cm, rho = {rho} kg/m^3):")
    print(f"  Analytical: I = {I_analytical_sphere:.6f} kg*m^2")
    print(f"  Monte Carlo (500k samples): I = {I_computed[-1]:.6f} kg*m^2")
    print(f"  Relative error: {abs(I_computed[-1]/I_analytical_sphere - 1)*100:.2f}%")

    print(f"\nL-Shape Principal Moments:")
    print(f"  I1 = {principal_I[0]:.8f} kg*m^2")
    print(f"  I2 = {principal_I[1]:.8f} kg*m^2")
    print(f"  I3 = {principal_I[2]:.8f} kg*m^2")

    print(f"\nInertia Tensor:")
    print(I_tensor)


if __name__ == "__main__":
    main()
