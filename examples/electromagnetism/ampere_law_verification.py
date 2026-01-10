"""
Experiment 88: Ampere's law verification.

This example verifies Ampere's law by computing line integrals of the
magnetic field around different paths and comparing with the enclosed current.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt


# Constants
MU_0 = 4 * np.pi * 1e-7  # Permeability of free space (H/m)


def infinite_wire_field(x, y, I, x0=0, y0=0):
    """
    Magnetic field from an infinite straight wire along z-axis.

    B = mu_0 * I / (2 * pi * r) in the phi direction.

    Args:
        x, y: Field point coordinates
        I: Current (A), positive for +z direction
        x0, y0: Wire position

    Returns:
        Bx, By: Magnetic field components (T)
    """
    dx = x - x0
    dy = y - y0
    r = np.sqrt(dx**2 + dy**2)
    r = np.maximum(r, 1e-10)

    # B = (mu_0 * I / 2*pi*r) * phi_hat
    # phi_hat = (-sin(phi), cos(phi)) = (-y/r, x/r)
    B_mag = MU_0 * I / (2 * np.pi * r)

    Bx = -B_mag * dy / r
    By = B_mag * dx / r

    return Bx, By


def compute_line_integral(path_x, path_y, Bx_func, By_func):
    """
    Compute line integral of B along a closed path.

    integral(B . dl)

    Args:
        path_x, path_y: Arrays of path coordinates
        Bx_func, By_func: Functions returning field components

    Returns:
        integral: Value of line integral
    """
    integral = 0.0

    for i in range(len(path_x) - 1):
        # Midpoint for field evaluation
        x_mid = (path_x[i] + path_x[i+1]) / 2
        y_mid = (path_y[i] + path_y[i+1]) / 2

        # dl vector
        dx = path_x[i+1] - path_x[i]
        dy = path_y[i+1] - path_y[i]

        # Field at midpoint
        Bx = Bx_func(x_mid, y_mid)
        By = By_func(x_mid, y_mid)

        # B . dl
        integral += Bx * dx + By * dy

    return integral


def create_circular_path(radius, center=(0, 0), n_points=200):
    """Create a circular path."""
    theta = np.linspace(0, 2*np.pi, n_points)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    return x, y


def create_rectangular_path(x_min, x_max, y_min, y_max, n_points=200):
    """Create a rectangular path."""
    n_side = n_points // 4

    # Bottom: left to right
    x1 = np.linspace(x_min, x_max, n_side)
    y1 = np.full(n_side, y_min)

    # Right: bottom to top
    x2 = np.full(n_side, x_max)
    y2 = np.linspace(y_min, y_max, n_side)

    # Top: right to left
    x3 = np.linspace(x_max, x_min, n_side)
    y3 = np.full(n_side, y_max)

    # Left: top to bottom
    x4 = np.full(n_side, x_min)
    y4 = np.linspace(y_max, y_min, n_side)

    x = np.concatenate([x1, x2, x3, x4])
    y = np.concatenate([y1, y2, y3, y4])

    return x, y


def main():
    # Wire parameters
    I = 10.0  # Current: 10 A

    fig = plt.figure(figsize=(16, 12))

    # Plot 1: Field around single wire with Amperian loops
    ax1 = fig.add_subplot(2, 2, 1)

    n = 100
    x = np.linspace(-2, 2, n)
    y = np.linspace(-2, 2, n)
    X, Y = np.meshgrid(x, y)

    Bx, By = infinite_wire_field(X, Y, I)
    B_mag = np.sqrt(Bx**2 + By**2)

    # Field streamlines
    ax1.streamplot(X, Y, Bx, By, color=np.log10(B_mag + 1e-10),
                   cmap='viridis', linewidth=1, density=2)

    # Draw wire (cross-section)
    ax1.plot(0, 0, 'ro', markersize=15, label='Wire (I out of page)')

    # Draw Amperian loops
    for r in [0.5, 1.0, 1.5]:
        theta = np.linspace(0, 2*np.pi, 100)
        ax1.plot(r * np.cos(theta), r * np.sin(theta), 'g--', lw=2)

    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_title('Magnetic Field Around Infinite Wire\nwith Amperian Loops')
    ax1.set_aspect('equal')
    ax1.legend(loc='upper right')

    # Plot 2: Ampere's law verification for different loop radii
    ax2 = fig.add_subplot(2, 2, 2)

    radii = np.linspace(0.1, 2, 20)
    line_integrals = []
    theoretical = MU_0 * I

    for r in radii:
        path_x, path_y = create_circular_path(r)

        def Bx_func(x, y):
            return infinite_wire_field(x, y, I)[0]

        def By_func(x, y):
            return infinite_wire_field(x, y, I)[1]

        integral = compute_line_integral(path_x, path_y, Bx_func, By_func)
        line_integrals.append(integral)

    ax2.axhline(y=theoretical * 1e6, color='r', linestyle='--', lw=2,
                label=f'Theory: mu_0*I = {theoretical*1e6:.3f} uT*m')
    ax2.plot(radii, np.array(line_integrals) * 1e6, 'bo-', lw=2,
             label='Numerical integral')

    ax2.set_xlabel('Loop radius (m)')
    ax2.set_ylabel('Line integral (uT*m)')
    ax2.set_title("Ampere's Law: Circular Loops\nintegral(B.dl) = mu_0 * I_enclosed")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Error analysis
    error = np.abs(np.array(line_integrals) - theoretical) / theoretical * 100
    ax2.text(0.95, 0.05, f'Max error: {np.max(error):.3f}%',
             transform=ax2.transAxes, ha='right', va='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 3: Different path shapes
    ax3 = fig.add_subplot(2, 2, 3)

    # Test different paths enclosing the wire
    paths = [
        ('Circle r=0.5', create_circular_path(0.5)),
        ('Circle r=1.0', create_circular_path(1.0)),
        ('Square 2x2', create_rectangular_path(-1, 1, -1, 1)),
        ('Rectangle 3x1', create_rectangular_path(-1.5, 1.5, -0.5, 0.5)),
    ]

    def Bx_func(x, y):
        return infinite_wire_field(x, y, I)[0]

    def By_func(x, y):
        return infinite_wire_field(x, y, I)[1]

    integrals = []
    for name, (path_x, path_y) in paths:
        integral = compute_line_integral(path_x, path_y, Bx_func, By_func)
        integrals.append(integral)
        ax3.plot(path_x, path_y, lw=2, label=f'{name}: {integral*1e6:.3f} uT*m')

    ax3.plot(0, 0, 'ro', markersize=10)
    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax3.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

    ax3.set_xlabel('x (m)')
    ax3.set_ylabel('y (m)')
    ax3.set_title(f"Different Amperian Paths (All enclose I = {I}A)")
    ax3.legend(fontsize=9)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Path not enclosing current
    ax4 = fig.add_subplot(2, 2, 4)

    # Create paths that don't enclose the wire
    paths_no_current = [
        ('Circle at (2,0), r=0.5', create_circular_path(0.5, center=(2, 0))),
        ('Circle at (0,2), r=0.5', create_circular_path(0.5, center=(0, 2))),
        ('Square not enclosing', create_rectangular_path(0.5, 1.5, 0.5, 1.5)),
    ]

    # Also test two-wire system
    def Bx_two_wires(x, y):
        Bx1, By1 = infinite_wire_field(x, y, I, x0=-0.5)
        Bx2, By2 = infinite_wire_field(x, y, -I, x0=0.5)  # Opposite current
        return Bx1 + Bx2

    def By_two_wires(x, y):
        Bx1, By1 = infinite_wire_field(x, y, I, x0=-0.5)
        Bx2, By2 = infinite_wire_field(x, y, -I, x0=0.5)
        return By1 + By2

    # Draw field for two wires
    Bx_tw = Bx_two_wires(X, Y)
    By_tw = By_two_wires(X, Y)
    B_mag_tw = np.sqrt(Bx_tw**2 + By_tw**2)

    ax4.streamplot(X, Y, Bx_tw, By_tw, color=np.log10(B_mag_tw + 1e-10),
                   cmap='plasma', linewidth=0.8, density=1.5)

    # Draw wires
    ax4.plot(-0.5, 0, 'ro', markersize=12, label='+I (out)')
    ax4.plot(0.5, 0, 'bo', markersize=12, label='-I (into page)')

    # Test paths
    test_cases = [
        ('Enclose both (net=0)', create_circular_path(1.5), Bx_two_wires, By_two_wires),
        ('Enclose +I only', create_circular_path(0.3, center=(-0.5, 0)), Bx_two_wires, By_two_wires),
        ('Enclose -I only', create_circular_path(0.3, center=(0.5, 0)), Bx_two_wires, By_two_wires),
    ]

    results_text = []
    for name, (path_x, path_y), Bx_f, By_f in test_cases:
        integral = compute_line_integral(path_x, path_y, Bx_f, By_f)
        ax4.plot(path_x, path_y, '--', lw=2)
        results_text.append(f'{name}: {integral*1e6:.3f} uT*m')

    ax4.set_xlabel('x (m)')
    ax4.set_ylabel('y (m)')
    ax4.set_title('Two Wires with Opposite Currents')
    ax4.legend(loc='upper right')
    ax4.set_aspect('equal')

    # Add results
    ax4.text(0.02, 0.02, '\n'.join(results_text), transform=ax4.transAxes,
             fontsize=9, va='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Add physics summary
    fig.text(0.5, 0.02,
             r"Ampere's Law: $\oint \vec{B} \cdot d\vec{l} = \mu_0 I_{enclosed}$" + '\n'
             f'For I = {I} A: mu_0*I = {MU_0*I*1e6:.3f} uT*m',
             ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.suptitle("Verification of Ampere's Law", fontsize=14, y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)

    # Save plot
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'ampere_law_verification.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
