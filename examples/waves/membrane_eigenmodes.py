"""
Experiment 69: 2D Membrane Eigenmodes.

Study of vibrating rectangular and circular membranes:

1. Rectangular membrane: sine modes (Chladni patterns)
2. Circular membrane: Bessel function modes (drum head)
3. Nodal lines and degeneracy
4. Time evolution
5. Mode superposition

The 2D wave equation: d^2u/dt^2 = c^2 * (d^2u/dx^2 + d^2u/dy^2)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jn, jn_zeros
from mpl_toolkits.mplot3d import Axes3D


# =============================================================================
# Rectangular Membrane
# =============================================================================

def rectangular_mode(x, y, m, n, Lx, Ly):
    """
    (m, n) mode shape for rectangular membrane.

    phi_{m,n}(x, y) = sin(m*pi*x/Lx) * sin(n*pi*y/Ly)
    """
    return np.sin(m * np.pi * x / Lx) * np.sin(n * np.pi * y / Ly)


def rectangular_frequency(m, n, Lx, Ly, c):
    """
    (m, n) mode frequency for rectangular membrane.

    omega_{m,n} = pi * c * sqrt((m/Lx)^2 + (n/Ly)^2)
    """
    return np.pi * c * np.sqrt((m / Lx)**2 + (n / Ly)**2)


# =============================================================================
# Circular Membrane
# =============================================================================

def circular_mode_radial(r, theta, m, n, R):
    """
    (m, n) mode shape for circular membrane in polar coordinates.

    phi_{m,n}(r, theta) = J_m(k_{m,n} * r) * cos(m * theta)

    where k_{m,n} * R = z_{m,n} (nth zero of J_m)
    """
    # Get nth zero of Bessel function J_m
    zeros = jn_zeros(m, n)
    k_mn = zeros[n-1] / R

    return jn(m, k_mn * r) * np.cos(m * theta)


def circular_mode_cartesian(X, Y, m, n, R):
    """Circular mode in Cartesian coordinates."""
    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)

    # Get Bessel zero
    zeros = jn_zeros(m, n)
    k_mn = zeros[n-1] / R

    mode = jn(m, k_mn * r) * np.cos(m * theta)

    # Mask outside the circle
    mode[r > R] = np.nan

    return mode


def circular_frequency(m, n, R, c):
    """
    (m, n) mode frequency for circular membrane.

    omega_{m,n} = c * z_{m,n} / R

    where z_{m,n} is nth zero of Bessel J_m
    """
    zeros = jn_zeros(m, n)
    return c * zeros[n-1] / R


# =============================================================================
# Main
# =============================================================================

def main():
    fig = plt.figure(figsize=(16, 12))

    # Parameters
    Lx, Ly = 1.0, 1.0  # Rectangular membrane dimensions
    R = 1.0            # Circular membrane radius
    c = 1.0            # Wave speed

    # Grid for rectangular membrane
    nx, ny = 100, 100
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X_rect, Y_rect = np.meshgrid(x, y)

    # Grid for circular membrane
    x_circ = np.linspace(-R, R, 100)
    y_circ = np.linspace(-R, R, 100)
    X_circ, Y_circ = np.meshgrid(x_circ, y_circ)

    # Plot 1-6: Rectangular membrane modes (Chladni patterns)
    modes_rect = [(1, 1), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1)]

    for idx, (m, n) in enumerate(modes_rect):
        ax = fig.add_subplot(3, 4, idx + 1)

        Z = rectangular_mode(X_rect, Y_rect, m, n, Lx, Ly)
        omega = rectangular_frequency(m, n, Lx, Ly, c)

        # Plot mode shape
        im = ax.contourf(X_rect, Y_rect, Z, levels=50, cmap='RdBu')
        ax.contour(X_rect, Y_rect, Z, levels=[0], colors='black', linewidths=2)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'({m},{n}) mode\nomega = {omega:.2f}')
        ax.set_aspect('equal')

    # Plot 7-10: Circular membrane modes
    modes_circ = [(0, 1), (1, 1), (0, 2), (2, 1)]

    for idx, (m, n) in enumerate(modes_circ):
        ax = fig.add_subplot(3, 4, 7 + idx)

        Z = circular_mode_cartesian(X_circ, Y_circ, m, n, R)
        omega = circular_frequency(m, n, R, c)

        # Plot mode shape
        im = ax.contourf(X_circ, Y_circ, Z, levels=50, cmap='RdBu')
        ax.contour(X_circ, Y_circ, Z, levels=[0], colors='black', linewidths=2)

        # Draw circle boundary
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(R*np.cos(theta), R*np.sin(theta), 'k-', lw=2)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'({m},{n}) Bessel mode\nomega = {omega:.2f}')
        ax.set_aspect('equal')
        ax.set_xlim(-1.1*R, 1.1*R)
        ax.set_ylim(-1.1*R, 1.1*R)

    # Plot 11: Frequency spectrum (rectangular)
    ax = fig.add_subplot(3, 4, 11)

    # Compute frequencies for many modes
    freqs_rect = []
    labels_rect = []
    for m in range(1, 6):
        for n in range(1, 6):
            omega = rectangular_frequency(m, n, Lx, Ly, c)
            freqs_rect.append(omega)
            labels_rect.append(f'({m},{n})')

    # Sort by frequency
    sorted_idx = np.argsort(freqs_rect)
    freqs_sorted = [freqs_rect[i] for i in sorted_idx[:15]]
    labels_sorted = [labels_rect[i] for i in sorted_idx[:15]]

    ax.barh(range(len(freqs_sorted)), freqs_sorted, color='steelblue')
    ax.set_yticks(range(len(freqs_sorted)))
    ax.set_yticklabels(labels_sorted)
    ax.set_xlabel('Frequency omega')
    ax.set_ylabel('Mode (m, n)')
    ax.set_title('Rectangular Membrane\nFrequency Spectrum')

    # Highlight degenerate modes
    ax.axvline(x=rectangular_frequency(1, 2, Lx, Ly, c), color='red',
               linestyle='--', alpha=0.5, label='Degenerate')
    ax.legend(fontsize=8)

    # Plot 12: Summary
    ax = fig.add_subplot(3, 4, 12)
    ax.axis('off')

    summary = """2D Membrane Eigenmodes
======================

WAVE EQUATION:
  d^2u/dt^2 = c^2 * nabla^2 u

RECTANGULAR MEMBRANE (Lx x Ly):
  phi_{m,n}(x,y) = sin(m*pi*x/Lx) * sin(n*pi*y/Ly)
  omega_{m,n} = pi*c*sqrt((m/Lx)^2 + (n/Ly)^2)

CIRCULAR MEMBRANE (radius R):
  phi_{m,n}(r,theta) = J_m(k_{m,n}*r) * cos(m*theta)
  k_{m,n} = z_{m,n} / R (Bessel zeros)
  omega_{m,n} = c * z_{m,n} / R

KEY FEATURES:

1. NODAL LINES (black):
   Lines where u = 0 always
   (Chladni patterns on plates)

2. DEGENERACY:
   Square membrane: (m,n) and (n,m)
   have same frequency (if m != n)

3. HIGHER MODES:
   More nodal lines, higher frequency

4. BOUNDARY SHAPE:
   Rectangle -> sine products
   Circle -> Bessel functions

APPLICATIONS:
- Drum acoustics
- Loudspeaker cones
- MEMS devices
- Quantum dots (2D confinement)

Bessel zeros z_{m,n}:
  J_0: 2.405, 5.520, 8.654...
  J_1: 3.832, 7.016, 10.17..."""

    ax.text(0.02, 0.98, summary, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    plt.suptitle("2D Membrane Eigenmodes: Chladni Patterns and Bessel Functions",
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'membrane_eigenmodes.png'), dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/membrane_eigenmodes.png")

    # Create additional figure with 3D mode visualizations
    fig2 = plt.figure(figsize=(15, 10))

    # 3D plots of selected modes
    modes_3d = [((1, 1), 'rect'), ((2, 2), 'rect'), ((0, 1), 'circ'), ((1, 2), 'circ')]

    for idx, ((m, n), shape) in enumerate(modes_3d):
        ax = fig2.add_subplot(2, 2, idx + 1, projection='3d')

        if shape == 'rect':
            Z = rectangular_mode(X_rect, Y_rect, m, n, Lx, Ly)
            surf = ax.plot_surface(X_rect, Y_rect, Z, cmap='RdBu', alpha=0.8)
            ax.set_title(f'Rectangular ({m},{n}) mode')
        else:
            Z = circular_mode_cartesian(X_circ, Y_circ, m, n, R)
            Z_masked = np.ma.masked_where(np.isnan(Z), Z)
            surf = ax.plot_surface(X_circ, Y_circ, Z_masked, cmap='RdBu', alpha=0.8)
            ax.set_title(f'Circular ({m},{n}) mode')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u')

    plt.suptitle("3D Visualization of Membrane Modes", fontsize=14)
    plt.tight_layout()

    fig2.savefig(os.path.join(output_dir, 'membrane_eigenmodes_3d.png'), dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/membrane_eigenmodes_3d.png")


if __name__ == "__main__":
    main()
