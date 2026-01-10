"""
Experiment 83: Capacitor edge effects (2D Laplace).

This example solves the 2D Laplace equation to visualize the
electric potential and field distribution in a parallel plate
capacitor, revealing edge effects (fringing fields).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt


def solve_laplace_2d(nx, ny, V_top, V_bottom, plate_x_start, plate_x_end,
                     plate_y_top, plate_y_bottom, tolerance=1e-6, max_iter=10000):
    """
    Solve 2D Laplace equation using finite difference relaxation method.

    Args:
        nx, ny: Grid dimensions
        V_top, V_bottom: Plate voltages
        plate_x_start, plate_x_end: Plate x-extent (as fraction of domain)
        plate_y_top, plate_y_bottom: Plate y-positions (as fraction)
        tolerance: Convergence criterion
        max_iter: Maximum iterations

    Returns:
        V: Potential array
        iterations: Number of iterations to converge
    """
    V = np.zeros((ny, nx))

    # Plate indices
    j_top = int(plate_y_top * ny)
    j_bottom = int(plate_y_bottom * ny)
    i_start = int(plate_x_start * nx)
    i_end = int(plate_x_end * nx)

    # Set plate boundary conditions
    V[j_top, i_start:i_end] = V_top
    V[j_bottom, i_start:i_end] = V_bottom

    # Relaxation iteration
    for iteration in range(max_iter):
        V_old = V.copy()

        # Update interior points (Jacobi method)
        for j in range(1, ny-1):
            for i in range(1, nx-1):
                # Skip plate regions
                if (j == j_top or j == j_bottom) and (i_start <= i < i_end):
                    continue
                V[j, i] = 0.25 * (V_old[j+1, i] + V_old[j-1, i] +
                                  V_old[j, i+1] + V_old[j, i-1])

        # Check convergence
        diff = np.max(np.abs(V - V_old))
        if diff < tolerance:
            return V, iteration + 1

    return V, max_iter


def solve_laplace_fast(nx, ny, V_top, V_bottom, plate_x_start, plate_x_end,
                       plate_y_top, plate_y_bottom, tolerance=1e-6, max_iter=50000):
    """
    Faster Laplace solver using vectorized SOR (Successive Over-Relaxation).
    """
    V = np.zeros((ny, nx))

    # Plate indices
    j_top = int(plate_y_top * ny)
    j_bottom = int(plate_y_bottom * ny)
    i_start = int(plate_x_start * nx)
    i_end = int(plate_x_end * nx)

    # Create mask for fixed points (plates)
    mask = np.ones((ny, nx), dtype=bool)
    mask[j_top, i_start:i_end] = False
    mask[j_bottom, i_start:i_end] = False
    mask[0, :] = False  # Boundaries
    mask[-1, :] = False
    mask[:, 0] = False
    mask[:, -1] = False

    # Set plate potentials
    V[j_top, i_start:i_end] = V_top
    V[j_bottom, i_start:i_end] = V_bottom

    # Optimal relaxation parameter for Laplace
    omega = 2 / (1 + np.sin(np.pi / nx))

    for iteration in range(max_iter):
        V_old = V.copy()

        # Vectorized update for interior points
        V_new = 0.25 * (np.roll(V, 1, axis=0) + np.roll(V, -1, axis=0) +
                        np.roll(V, 1, axis=1) + np.roll(V, -1, axis=1))

        # Apply SOR update only at non-fixed points
        V = np.where(mask, V_old + omega * (V_new - V_old), V)

        # Restore boundary conditions
        V[j_top, i_start:i_end] = V_top
        V[j_bottom, i_start:i_end] = V_bottom
        V[0, :] = 0
        V[-1, :] = 0
        V[:, 0] = 0
        V[:, -1] = 0

        # Check convergence
        diff = np.max(np.abs(V[mask] - V_old[mask]))
        if diff < tolerance:
            return V, iteration + 1

    return V, max_iter


def calculate_electric_field(V, dx, dy):
    """Calculate electric field from potential gradient."""
    Ey, Ex = np.gradient(-V, dy, dx)
    return Ex, Ey


def main():
    # Domain parameters
    nx, ny = 150, 150
    Lx, Ly = 1.0, 1.0  # Domain size (m)
    dx = Lx / nx
    dy = Ly / ny

    # Capacitor parameters
    V_top = 100      # Top plate voltage (V)
    V_bottom = 0     # Bottom plate voltage (V)

    # Create coordinate arrays
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(16, 12))

    # Case 1: Full-width plates (ideal capacitor)
    ax1 = fig.add_subplot(2, 2, 1)

    V1, iter1 = solve_laplace_fast(nx, ny, V_top, V_bottom,
                                    0.05, 0.95, 0.7, 0.3)
    Ex1, Ey1 = calculate_electric_field(V1, dx, dy)
    E_mag1 = np.sqrt(Ex1**2 + Ey1**2)

    # Potential contours
    levels = np.linspace(0, 100, 21)
    contour1 = ax1.contourf(X, Y, V1, levels=levels, cmap='RdBu_r')
    ax1.contour(X, Y, V1, levels=levels, colors='white', linewidths=0.5, alpha=0.5)
    plt.colorbar(contour1, ax=ax1, label='Potential (V)')

    # Field lines
    ax1.streamplot(X, Y, Ex1, Ey1, color='black', linewidth=0.5,
                   density=1.5, arrowsize=1)

    # Draw plates
    ax1.hlines(0.7, 0.05, 0.95, colors='red', linewidths=4, label='Top plate (+)')
    ax1.hlines(0.3, 0.05, 0.95, colors='blue', linewidths=4, label='Bottom plate (-)')

    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_title(f'Near-Ideal Capacitor (full plates)\nConverged in {iter1} iterations')
    ax1.set_aspect('equal')
    ax1.legend(loc='upper right', fontsize=8)

    # Case 2: Short plates (strong edge effects)
    ax2 = fig.add_subplot(2, 2, 2)

    V2, iter2 = solve_laplace_fast(nx, ny, V_top, V_bottom,
                                    0.3, 0.7, 0.7, 0.3)
    Ex2, Ey2 = calculate_electric_field(V2, dx, dy)
    E_mag2 = np.sqrt(Ex2**2 + Ey2**2)

    contour2 = ax2.contourf(X, Y, V2, levels=levels, cmap='RdBu_r')
    ax2.contour(X, Y, V2, levels=levels, colors='white', linewidths=0.5, alpha=0.5)
    plt.colorbar(contour2, ax=ax2, label='Potential (V)')

    ax2.streamplot(X, Y, Ex2, Ey2, color='black', linewidth=0.5,
                   density=1.5, arrowsize=1)

    ax2.hlines(0.7, 0.3, 0.7, colors='red', linewidths=4)
    ax2.hlines(0.3, 0.3, 0.7, colors='blue', linewidths=4)

    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')
    ax2.set_title(f'Finite Plates (strong fringing)\nConverged in {iter2} iterations')
    ax2.set_aspect('equal')

    # Case 3: Field magnitude comparison
    ax3 = fig.add_subplot(2, 2, 3)

    # Field along vertical centerline
    center_x = nx // 2
    y_line = y

    E_ideal = V_top / (0.7 - 0.3)  # Ideal uniform field between plates
    E_center1 = E_mag1[:, center_x]
    E_center2 = E_mag2[:, center_x]

    ax3.plot(y_line, E_center1, 'b-', lw=2, label='Full plates')
    ax3.plot(y_line, E_center2, 'r-', lw=2, label='Short plates')
    ax3.axhline(y=E_ideal, color='gray', linestyle='--', lw=2,
                label=f'Ideal: {E_ideal:.1f} V/m')

    # Mark plate positions
    ax3.axvline(x=0.3, color='blue', linestyle=':', alpha=0.5)
    ax3.axvline(x=0.7, color='red', linestyle=':', alpha=0.5)
    ax3.axvspan(0.3, 0.7, alpha=0.1, color='gray', label='Between plates')

    ax3.set_xlabel('y position (m)')
    ax3.set_ylabel('Electric Field Magnitude (V/m)')
    ax3.set_title('Field Strength Along Vertical Centerline')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Case 4: Field uniformity analysis
    ax4 = fig.add_subplot(2, 2, 4)

    # Calculate field uniformity in the capacitor region
    j_bottom_idx = int(0.3 * ny)
    j_top_idx = int(0.7 * ny)
    i_left_idx = int(0.3 * nx)
    i_right_idx = int(0.7 * nx)

    E_region1 = E_mag1[j_bottom_idx:j_top_idx, :]
    E_region2 = E_mag2[j_bottom_idx:j_top_idx, :]

    # Field uniformity across horizontal slices
    y_mid = (0.3 + 0.7) / 2
    j_mid = int(y_mid * ny)

    ax4.plot(x, E_mag1[j_mid, :], 'b-', lw=2, label='Full plates (y=0.5)')
    ax4.plot(x, E_mag2[j_mid, :], 'r-', lw=2, label='Short plates (y=0.5)')
    ax4.axhline(y=E_ideal, color='gray', linestyle='--', lw=2, label='Ideal')

    # Mark plate extent for short plates
    ax4.axvline(x=0.3, color='green', linestyle=':', alpha=0.7)
    ax4.axvline(x=0.7, color='green', linestyle=':', alpha=0.7)
    ax4.axvspan(0.3, 0.7, alpha=0.1, color='green', label='Plate region')

    ax4.set_xlabel('x position (m)')
    ax4.set_ylabel('Electric Field Magnitude (V/m)')
    ax4.set_title('Field Uniformity at Midplane (y = 0.5)')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # Add info about edge effects
    edge_enhancement = np.max(E_mag2) / E_ideal
    ax4.text(0.95, 0.95, f'Edge enhancement: {edge_enhancement:.2f}x',
             transform=ax4.transAxes, ha='right', va='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Capacitor Edge Effects: 2D Laplace Equation Solution\n'
                 r'$\nabla^2 V = 0$ with boundary conditions',
                 fontsize=14, y=1.02)

    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'capacitor_edge_effects.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
