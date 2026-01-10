"""
Experiment 235: Ginzburg-Landau Vortices

Demonstrates the Ginzburg-Landau theory of superconductivity, showing
vortex solutions in type-II superconductors where magnetic flux
penetrates in quantized vortex lines.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


def initialize_order_parameter(Nx, Ny, n_vortices=1, positions=None):
    """
    Initialize superconducting order parameter with vortices.

    Args:
        Nx, Ny: Grid dimensions
        n_vortices: Number of vortices to place
        positions: List of (x, y) vortex positions (optional)

    Returns:
        psi: Complex order parameter field
    """
    x = np.linspace(-10, 10, Nx)
    y = np.linspace(-10, 10, Ny)
    X, Y = np.meshgrid(x, y)

    psi = np.ones((Ny, Nx), dtype=complex)

    if positions is None:
        # Place vortices at random positions
        positions = []
        for _ in range(n_vortices):
            positions.append((np.random.uniform(-5, 5), np.random.uniform(-5, 5)))

    for x0, y0 in positions:
        r = np.sqrt((X - x0)**2 + (Y - y0)**2)
        theta = np.arctan2(Y - y0, X - x0)

        # Vortex ansatz: psi ~ f(r) * exp(i*theta)
        # f(r) ~ r for small r, f(r) -> 1 for large r
        f = r / np.sqrt(r**2 + 1)
        psi *= f * np.exp(1j * theta)

    return psi, x, y


def gl_free_energy_density(psi, A, alpha, beta, kappa, dx, dy):
    """
    Compute Ginzburg-Landau free energy density.

    f = alpha*|psi|^2 + (beta/2)*|psi|^4 + |(grad - iA)psi|^2/2 + (curl A)^2/2

    Args:
        psi: Order parameter
        A: Vector potential (Ax, Ay)
        alpha: GL parameter (< 0 in superconducting state)
        beta: GL parameter (> 0)
        kappa: GL parameter (ratio of penetration depth to coherence length)
        dx, dy: Grid spacing

    Returns:
        Free energy density field
    """
    Ax, Ay = A

    # Covariant derivatives
    psi_x = (np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1)) / (2 * dx)
    psi_y = (np.roll(psi, -1, axis=0) - np.roll(psi, 1, axis=0)) / (2 * dy)

    D_x_psi = psi_x - 1j * Ax * psi
    D_y_psi = psi_y - 1j * Ay * psi

    # Magnetic field B = curl A
    Ax_y = (np.roll(Ax, -1, axis=0) - np.roll(Ax, 1, axis=0)) / (2 * dy)
    Ay_x = (np.roll(Ay, -1, axis=1) - np.roll(Ay, 1, axis=1)) / (2 * dx)
    B = Ay_x - Ax_y

    # Free energy density
    f = (alpha * np.abs(psi)**2 +
         beta / 2 * np.abs(psi)**4 +
         np.abs(D_x_psi)**2 / 2 + np.abs(D_y_psi)**2 / 2 +
         B**2 / (2 * kappa**2))

    return np.real(f), B


def solve_gl_relaxation(psi_init, alpha, beta, kappa, dx, dy, n_steps=1000, dt=0.01):
    """
    Solve GL equations using relaxation method.

    dpsi/dt = -dF/dpsi*

    Args:
        psi_init: Initial order parameter
        alpha, beta, kappa: GL parameters
        dx, dy: Grid spacing
        n_steps: Number of iterations
        dt: Time step

    Returns:
        psi: Relaxed order parameter
    """
    psi = psi_init.copy()
    Ny, Nx = psi.shape

    # No applied field (self-consistent A)
    Ax = np.zeros((Ny, Nx))
    Ay = np.zeros((Ny, Nx))

    for step in range(n_steps):
        # Laplacian of psi
        psi_lap = ((np.roll(psi, 1, axis=1) + np.roll(psi, -1, axis=1) - 2*psi) / dx**2 +
                   (np.roll(psi, 1, axis=0) + np.roll(psi, -1, axis=0) - 2*psi) / dy**2)

        # GL equation: dpsi/dt = lap(psi) - alpha*psi - beta*|psi|^2*psi
        dpsi = psi_lap - alpha * psi - beta * np.abs(psi)**2 * psi

        psi += dt * dpsi

    return psi


def abrikosov_lattice_positions(n_vortices, a_lattice):
    """
    Generate positions for triangular Abrikosov vortex lattice.

    Args:
        n_vortices: Approximate number of vortices
        a_lattice: Lattice constant

    Returns:
        List of (x, y) positions
    """
    positions = []
    n_side = int(np.sqrt(n_vortices))

    for i in range(-n_side, n_side):
        for j in range(-n_side, n_side):
            # Triangular lattice
            x = a_lattice * (i + 0.5 * (j % 2))
            y = a_lattice * j * np.sqrt(3) / 2
            positions.append((x, y))

    return positions


def main():
    """Main simulation and visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Grid parameters
    Nx, Ny = 100, 100
    Lx, Ly = 20, 20
    dx, dy = Lx/Nx, Ly/Ny

    # GL parameters (Type-II superconductor: kappa > 1/sqrt(2))
    alpha = -1.0
    beta = 1.0
    kappa = 2.0  # Type-II

    # Plot 1: Single vortex structure
    ax1 = axes[0, 0]

    psi_single, x, y = initialize_order_parameter(Nx, Ny, n_vortices=1, positions=[(0, 0)])
    psi_relaxed = solve_gl_relaxation(psi_single, alpha, beta, kappa, dx, dy, n_steps=500)

    X, Y = np.meshgrid(x, y)
    im1 = ax1.pcolormesh(X, Y, np.abs(psi_relaxed)**2, cmap='viridis', shading='auto')
    plt.colorbar(im1, ax=ax1, label='|psi|^2')

    # Overlay phase contours
    phase = np.angle(psi_relaxed)
    ax1.contour(X, Y, phase, levels=np.linspace(-np.pi, np.pi, 9), colors='white', alpha=0.5)

    ax1.set_xlabel('x (xi)')
    ax1.set_ylabel('y (xi)')
    ax1.set_title('Single Vortex: Order Parameter |psi|^2')
    ax1.set_aspect('equal')

    # Plot 2: Vortex profile (radial cut)
    ax2 = axes[0, 1]

    # Extract radial profile
    r_vals = np.sqrt(X**2 + Y**2)
    r_bins = np.linspace(0, 8, 50)
    psi_profile = []

    for i in range(len(r_bins) - 1):
        mask = (r_vals >= r_bins[i]) & (r_vals < r_bins[i+1])
        if np.any(mask):
            psi_profile.append(np.mean(np.abs(psi_relaxed[mask])**2))
        else:
            psi_profile.append(0)

    r_centers = (r_bins[:-1] + r_bins[1:]) / 2
    ax2.plot(r_centers, psi_profile, 'b-', lw=2, label='|psi|^2 (numerical)')

    # Theoretical profile: |psi|^2 ~ tanh^2(r/xi)
    xi = 1 / np.sqrt(-2 * alpha)
    psi_theory = np.tanh(r_centers / xi)**2
    ax2.plot(r_centers, psi_theory, 'r--', lw=2, label='tanh^2(r/xi)')

    ax2.set_xlabel('r (xi)')
    ax2.set_ylabel('|psi|^2')
    ax2.set_title('Vortex Core Profile')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Vortex lattice
    ax3 = axes[1, 0]

    # Create vortex lattice
    a_lattice = 3.0
    positions = abrikosov_lattice_positions(20, a_lattice)
    psi_lattice, x_l, y_l = initialize_order_parameter(150, 150, positions=positions)

    X_l, Y_l = np.meshgrid(x_l, y_l)
    im3 = ax3.pcolormesh(X_l, Y_l, np.abs(psi_lattice)**2, cmap='viridis', shading='auto')
    plt.colorbar(im3, ax=ax3, label='|psi|^2')

    # Mark vortex positions
    for x0, y0 in positions:
        if -10 < x0 < 10 and -10 < y0 < 10:
            ax3.plot(x0, y0, 'rx', markersize=5)

    ax3.set_xlabel('x (xi)')
    ax3.set_ylabel('y (xi)')
    ax3.set_title('Abrikosov Vortex Lattice')
    ax3.set_aspect('equal')
    ax3.set_xlim(-10, 10)
    ax3.set_ylim(-10, 10)

    # Plot 4: Phase diagram (H vs kappa)
    ax4 = axes[1, 1]

    kappa_range = np.linspace(0.1, 5, 100)
    kappa_c = 1 / np.sqrt(2)  # Type-I/Type-II boundary

    # Critical fields (schematic)
    H_c = np.ones_like(kappa_range)  # Thermodynamic critical field
    H_c1 = H_c * np.log(kappa_range) / (np.sqrt(2) * kappa_range)  # Lower critical field
    H_c2 = H_c * np.sqrt(2) * kappa_range  # Upper critical field

    # Only valid for Type-II
    H_c1[kappa_range < kappa_c] = np.nan
    H_c2[kappa_range < kappa_c] = np.nan

    ax4.semilogy(kappa_range, H_c, 'k-', lw=2, label='$H_c$ (thermodynamic)')
    ax4.semilogy(kappa_range, H_c1, 'b-', lw=2, label='$H_{c1}$ (lower)')
    ax4.semilogy(kappa_range, H_c2, 'r-', lw=2, label='$H_{c2}$ (upper)')

    ax4.axvline(x=kappa_c, color='green', linestyle='--', lw=2,
               label=f'Type-I/II boundary ($\\kappa_c = 1/\\sqrt{{2}}$)')

    # Shade regions
    ax4.fill_between(kappa_range, 0.01, H_c1, where=kappa_range > kappa_c,
                    alpha=0.2, color='blue', label='Meissner state')
    ax4.fill_between(kappa_range, H_c1, H_c2, where=kappa_range > kappa_c,
                    alpha=0.2, color='yellow', label='Mixed state (vortices)')

    ax4.set_xlabel('GL parameter kappa = lambda/xi')
    ax4.set_ylabel('Magnetic field H')
    ax4.set_title('Type-II Superconductor Phase Diagram')
    ax4.legend(loc='upper left', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 5)
    ax4.set_ylim(0.01, 10)

    ax4.text(0.3, 0.5, 'Type-I', fontsize=12, ha='center')
    ax4.text(2.5, 0.3, 'Type-II', fontsize=12, ha='center')

    plt.suptitle('Ginzburg-Landau Theory: Vortices in Type-II Superconductors\n'
                 r'$F = \alpha|\psi|^2 + \frac{\beta}{2}|\psi|^4 + \frac{1}{2}|(\nabla - i\mathbf{A})\psi|^2 + \frac{B^2}{2\mu_0}$',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'ginzburg_landau_vortices.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'ginzburg_landau_vortices.png')}")


if __name__ == "__main__":
    main()
