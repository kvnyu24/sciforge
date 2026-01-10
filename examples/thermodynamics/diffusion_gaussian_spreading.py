"""
Example demonstrating the diffusion equation and Gaussian spreading.

The diffusion equation (Fick's second law):
dc/dt = D * d^2c/dx^2

where D is the diffusion coefficient.

For an initial delta function (point source), the solution is a Gaussian
that spreads with time:

c(x,t) = (N / sqrt(4*pi*D*t)) * exp(-x^2 / (4*D*t))

This example shows:
- Gaussian spreading of an initial concentration pulse
- Width increases as sqrt(t)
- Comparison with numerical solution
- Effect of diffusion coefficient
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.core.constants import CONSTANTS


def gaussian_solution(x, t, D, N=1.0):
    """
    Analytical solution for diffusion from a point source.

    c(x,t) = (N / sqrt(4*pi*D*t)) * exp(-x^2 / (4*D*t))

    Args:
        x: Position array (m)
        t: Time (s)
        D: Diffusion coefficient (m^2/s)
        N: Total amount of diffusing substance

    Returns:
        Concentration array
    """
    if t <= 0:
        # Return delta-like initial condition
        c = np.zeros_like(x)
        c[len(x)//2] = N / (x[1] - x[0])
        return c

    return (N / np.sqrt(4 * np.pi * D * t)) * np.exp(-x**2 / (4 * D * t))


def gaussian_width(t, D):
    """
    Calculate the standard deviation (width) of the Gaussian.

    sigma = sqrt(2*D*t)

    Args:
        t: Time (s)
        D: Diffusion coefficient (m^2/s)

    Returns:
        Standard deviation (m)
    """
    return np.sqrt(2 * D * t)


def solve_diffusion_ftcs(c0, D, dx, dt, nt):
    """
    Solve 1D diffusion equation using FTCS method.

    Args:
        c0: Initial concentration distribution
        D: Diffusion coefficient (m^2/s)
        dx: Spatial step (m)
        dt: Time step (s)
        nt: Number of time steps

    Returns:
        Concentration evolution array (nt+1 x nx)
    """
    nx = len(c0)
    c = np.zeros((nt + 1, nx))
    c[0, :] = c0

    r = D * dt / dx**2
    if r > 0.5:
        print(f"Warning: r = {r:.3f} > 0.5, solution may be unstable")

    for n in range(nt):
        # Interior points
        c[n+1, 1:-1] = c[n, 1:-1] + r * (c[n, 2:] - 2*c[n, 1:-1] + c[n, :-2])
        # Zero-flux boundary conditions
        c[n+1, 0] = c[n+1, 1]
        c[n+1, -1] = c[n+1, -2]

    return c


def mean_squared_displacement(x, c):
    """
    Calculate mean squared displacement.

    <x^2> = integral(x^2 * c(x) dx) / integral(c(x) dx)

    For diffusion: <x^2> = 2*D*t
    """
    dx = x[1] - x[0]
    total_mass = np.sum(c) * dx
    if total_mass > 0:
        return np.sum(x**2 * c) * dx / total_mass
    return 0


def main():
    # Diffusion coefficients (m^2/s)
    D_values = {
        'Small molecule in water': 1e-9,
        'Protein in water': 1e-11,
        'Atom in solid': 1e-14,
    }

    # Use a moderate D for visualization
    D = 1e-5  # m^2/s (typical for gases)

    # Domain
    L = 0.02  # 2 cm total
    nx = 201
    x = np.linspace(-L/2, L/2, nx)
    dx = L / (nx - 1)

    # Time parameters
    dt = 1e-4  # s
    total_time = 0.5  # s
    nt = int(total_time / dt)

    # Initial condition: narrow Gaussian
    sigma0 = 0.0005  # 0.5 mm initial width
    c0 = np.exp(-x**2 / (2 * sigma0**2)) / (sigma0 * np.sqrt(2 * np.pi))

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Gaussian spreading over time (analytical)
    ax1 = axes[0, 0]

    times = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(times)))

    for t, color in zip(times, colors):
        c = gaussian_solution(x, t, D)
        sigma = gaussian_width(t, D)
        ax1.plot(x * 100, c * 100, color=color, lw=2,
                label=f't = {t*1000:.0f}ms, sigma = {sigma*1000:.2f}mm')

    ax1.set_xlabel('Position (cm)', fontsize=12)
    ax1.set_ylabel('Concentration (arb. units)', fontsize=12)
    ax1.set_title(f'Gaussian Spreading (D = {D*1e6:.0f} x 10^-6 m^2/s)', fontsize=12)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax1.text(0.05, 0.95, r'$c(x,t) = \frac{N}{\sqrt{4\pi Dt}} e^{-x^2/4Dt}$',
             transform=ax1.transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Plot 2: Width vs time
    ax2 = axes[0, 1]

    t_range = np.linspace(0.001, 0.5, 100)
    sigma_range = gaussian_width(t_range, D)

    ax2.plot(t_range * 1000, sigma_range * 1000, 'b-', lw=2, label=r'$\sigma = \sqrt{2Dt}$')

    # Also plot sqrt(t) fit
    ax2.plot(t_range * 1000, np.sqrt(t_range * 1000) * sigma_range[-1] / np.sqrt(t_range[-1] * 1000),
             'r--', lw=1.5, label=r'$\propto \sqrt{t}$')

    ax2.set_xlabel('Time (ms)', fontsize=12)
    ax2.set_ylabel('Gaussian Width sigma (mm)', fontsize=12)
    ax2.set_title('Diffusive Spreading: Width Grows as sqrt(t)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Log-log inset to show sqrt(t) scaling
    ax2_inset = ax2.inset_axes([0.55, 0.15, 0.4, 0.4])
    ax2_inset.loglog(t_range * 1000, sigma_range * 1000, 'b-', lw=2)
    ax2_inset.set_xlabel('t (ms)', fontsize=8)
    ax2_inset.set_ylabel('sigma (mm)', fontsize=8)
    ax2_inset.set_title('Log-log plot\nslope = 0.5', fontsize=8)
    ax2_inset.grid(True, alpha=0.3, which='both')

    # Plot 3: Numerical vs analytical solution
    ax3 = axes[1, 0]

    # Solve numerically
    c_numerical = solve_diffusion_ftcs(c0, D, dx, dt, nt)

    # Compare at specific time
    t_compare = 0.1  # s
    n_compare = int(t_compare / dt)

    c_analytical = gaussian_solution(x, t_compare, D)

    ax3.plot(x * 100, c_numerical[n_compare], 'b-', lw=2, label='Numerical (FTCS)')
    ax3.plot(x * 100, c_analytical, 'r--', lw=2, label='Analytical')
    ax3.plot(x * 100, c0, 'g:', lw=1.5, alpha=0.7, label='Initial (t=0)')

    ax3.set_xlabel('Position (cm)', fontsize=12)
    ax3.set_ylabel('Concentration', fontsize=12)
    ax3.set_title(f'Numerical vs Analytical at t = {t_compare*1000:.0f}ms', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Effect of diffusion coefficient
    ax4 = axes[1, 1]

    t_fixed = 0.1  # s
    D_range = [1e-6, 5e-6, 1e-5, 5e-5]
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(D_range)))

    for D_val, color in zip(D_range, colors):
        c = gaussian_solution(x, t_fixed, D_val)
        sigma = gaussian_width(t_fixed, D_val)
        ax4.plot(x * 100, c, color=color, lw=2,
                label=f'D = {D_val*1e6:.0f} x 10^-6 m^2/s')

    ax4.set_xlabel('Position (cm)', fontsize=12)
    ax4.set_ylabel('Concentration', fontsize=12)
    ax4.set_title(f'Effect of Diffusion Coefficient at t = {t_fixed*1000:.0f}ms', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Add annotation
    ax4.text(0.05, 0.05, 'Higher D = faster spreading\nLower peak concentration',
             transform=ax4.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Add conservation note
    ax4.text(0.95, 0.95, 'Total amount conserved:\n' + r'$\int c(x,t) dx = N$',
             transform=ax4.transAxes, fontsize=10, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('Diffusion Equation and Gaussian Spreading\n'
                 r'$\frac{\partial c}{\partial t} = D \frac{\partial^2 c}{\partial x^2}$',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'diffusion_gaussian_spreading.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'diffusion_gaussian_spreading.png')}")


if __name__ == "__main__":
    main()
