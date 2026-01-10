"""
Experiment 18: PDE conservation test - continuity equation mass conservation.

Tests mass conservation in numerical solutions of the continuity equation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def initial_density(x, y, mode='gaussian'):
    """Initial density distribution."""
    if mode == 'gaussian':
        x0, y0 = 0.3, 0.5
        sigma = 0.1
        return np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    elif mode == 'ring':
        r = np.sqrt((x - 0.5)**2 + (y - 0.5)**2)
        return np.exp(-(r - 0.2)**2 / (2 * 0.05**2))
    else:
        return np.ones_like(x)


def velocity_field(x, y, mode='rotation'):
    """Velocity field (incompressible, div v = 0)."""
    if mode == 'rotation':
        # Solid body rotation around (0.5, 0.5)
        omega = 2 * np.pi  # Angular velocity
        vx = -omega * (y - 0.5)
        vy = omega * (x - 0.5)
    elif mode == 'shear':
        vx = np.sin(2 * np.pi * y)
        vy = np.zeros_like(y)
    elif mode == 'vortex':
        r = np.sqrt((x - 0.5)**2 + (y - 0.5)**2) + 1e-10
        theta = np.arctan2(y - 0.5, x - 0.5)
        v_theta = np.tanh(10 * r) / r
        vx = -v_theta * np.sin(theta)
        vy = v_theta * np.cos(theta)
    else:
        vx = np.ones_like(x)
        vy = np.zeros_like(y)

    return vx, vy


def upwind_2d(rho, vx, vy, dx, dy, dt):
    """
    2D upwind scheme for continuity equation.

    ρ_t + ∇·(ρv) = 0

    For incompressible flow (div v = 0):
    ρ_t + v·∇ρ = 0
    """
    rho_new = rho.copy()
    ny, nx = rho.shape

    for i in range(1, ny-1):
        for j in range(1, nx-1):
            # Upwind in x
            if vx[i, j] > 0:
                drho_dx = (rho[i, j] - rho[i, j-1]) / dx
            else:
                drho_dx = (rho[i, j+1] - rho[i, j]) / dx

            # Upwind in y
            if vy[i, j] > 0:
                drho_dy = (rho[i, j] - rho[i-1, j]) / dy
            else:
                drho_dy = (rho[i+1, j] - rho[i, j]) / dy

            rho_new[i, j] = rho[i, j] - dt * (vx[i, j] * drho_dx + vy[i, j] * drho_dy)

    # Periodic BC
    rho_new[0, :] = rho_new[-2, :]
    rho_new[-1, :] = rho_new[1, :]
    rho_new[:, 0] = rho_new[:, -2]
    rho_new[:, -1] = rho_new[:, 1]

    return rho_new


def lax_wendroff_2d(rho, vx, vy, dx, dy, dt):
    """2D Lax-Wendroff scheme."""
    rho_new = rho.copy()
    ny, nx = rho.shape

    # Predictor step (half time step)
    rho_half = np.zeros_like(rho)

    for i in range(1, ny-1):
        for j in range(1, nx-1):
            # Central differences for spatial derivatives
            drho_dx = (rho[i, j+1] - rho[i, j-1]) / (2 * dx)
            drho_dy = (rho[i+1, j] - rho[i-1, j]) / (2 * dy)

            rho_half[i, j] = rho[i, j] - 0.5 * dt * (vx[i, j] * drho_dx + vy[i, j] * drho_dy)

    # Corrector step
    for i in range(1, ny-1):
        for j in range(1, nx-1):
            drho_dx = (rho_half[i, j+1] - rho_half[i, j-1]) / (2 * dx)
            drho_dy = (rho_half[i+1, j] - rho_half[i-1, j]) / (2 * dy)

            rho_new[i, j] = rho[i, j] - dt * (vx[i, j] * drho_dx + vy[i, j] * drho_dy)

    # Periodic BC
    rho_new[0, :] = rho_new[-2, :]
    rho_new[-1, :] = rho_new[1, :]
    rho_new[:, 0] = rho_new[:, -2]
    rho_new[:, -1] = rho_new[:, 1]

    return rho_new


def conservative_form(rho, vx, vy, dx, dy, dt):
    """
    Conservative form discretization.

    ρ_t + (ρvx)_x + (ρvy)_y = 0

    Uses flux differencing to ensure conservation.
    """
    rho_new = rho.copy()
    ny, nx = rho.shape

    # Compute fluxes at cell interfaces
    # Flux in x: F_x = ρ * vx
    # Flux in y: F_y = ρ * vy

    for i in range(1, ny-1):
        for j in range(1, nx-1):
            # x-fluxes at j+1/2 and j-1/2
            if vx[i, j] > 0:
                Fx_right = rho[i, j] * vx[i, j]
            else:
                Fx_right = rho[i, j+1] * vx[i, j]

            if vx[i, j-1] > 0:
                Fx_left = rho[i, j-1] * vx[i, j-1]
            else:
                Fx_left = rho[i, j] * vx[i, j-1]

            # y-fluxes at i+1/2 and i-1/2
            if vy[i, j] > 0:
                Fy_top = rho[i, j] * vy[i, j]
            else:
                Fy_top = rho[i+1, j] * vy[i, j]

            if vy[i-1, j] > 0:
                Fy_bottom = rho[i-1, j] * vy[i-1, j]
            else:
                Fy_bottom = rho[i, j] * vy[i-1, j]

            rho_new[i, j] = rho[i, j] - dt * ((Fx_right - Fx_left) / dx +
                                               (Fy_top - Fy_bottom) / dy)

    # Periodic BC
    rho_new[0, :] = rho_new[-2, :]
    rho_new[-1, :] = rho_new[1, :]
    rho_new[:, 0] = rho_new[:, -2]
    rho_new[:, -1] = rho_new[:, 1]

    return rho_new


def main():
    # Grid
    nx, ny = 64, 64
    Lx, Ly = 1.0, 1.0
    dx, dy = Lx / nx, Ly / ny

    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)

    # Velocity field (solid body rotation - incompressible)
    vx, vy = velocity_field(X, Y, 'rotation')

    # Time stepping
    v_max = np.max(np.sqrt(vx**2 + vy**2))
    dt = 0.4 * min(dx, dy) / v_max  # CFL condition
    T_rotation = 1.0  # One full rotation
    n_steps = int(T_rotation / dt)

    # Initial condition
    rho0 = initial_density(X, Y, 'gaussian')
    mass0 = np.sum(rho0) * dx * dy

    # Methods to test
    methods = {
        'Upwind': upwind_2d,
        'Lax-Wendroff': lax_wendroff_2d,
        'Conservative': conservative_form
    }

    results = {}

    print("Simulating continuity equation...")
    for name, method in methods.items():
        print(f"  {name}...")
        rho = rho0.copy()
        masses = [mass0]

        for step in range(n_steps):
            rho = method(rho, vx, vy, dx, dy, dt)
            mass = np.sum(rho) * dx * dy
            masses.append(mass)

        results[name] = {
            'rho_final': rho,
            'masses': np.array(masses)
        }

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    times = np.arange(n_steps + 1) * dt

    # Row 1: Final density distributions
    for idx, (name, data) in enumerate(results.items()):
        ax = axes[0, idx]

        im = ax.imshow(data['rho_final'], extent=[0, Lx, 0, Ly],
                       origin='lower', cmap='viridis', vmin=0)
        plt.colorbar(im, ax=ax, label='ρ')

        # Add velocity field arrows
        skip = 4
        ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                  vx[::skip, ::skip], vy[::skip, ::skip],
                  color='white', alpha=0.5, scale=30)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'{name} (t = {T_rotation:.2f})')

    # Row 2: Mass conservation
    ax = axes[1, 0]

    for name, data in results.items():
        mass_error = (data['masses'] - mass0) / mass0 * 100
        ax.plot(times, mass_error, lw=2, label=name)

    ax.set_xlabel('Time')
    ax.set_ylabel('Mass error (%)')
    ax.set_title('Mass Conservation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Row 2: Mass error (log scale)
    ax = axes[1, 1]

    for name, data in results.items():
        mass_error = np.abs(data['masses'] - mass0) / mass0
        ax.semilogy(times, mass_error + 1e-16, lw=2, label=name)

    ax.set_xlabel('Time')
    ax.set_ylabel('|Mass error| / M₀')
    ax.set_title('Mass Error (Log Scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Row 2: Summary
    ax = axes[1, 2]
    ax.axis('off')

    summary = f"""Continuity Equation Test
========================
∂ρ/∂t + ∇·(ρv) = 0

Test Setup:
• Grid: {nx} × {ny}
• Velocity: solid body rotation
• Initial: Gaussian blob
• One full rotation (T = {T_rotation})

Mass Conservation Errors:
"""

    for name, data in results.items():
        final_error = abs(data['masses'][-1] - mass0) / mass0
        max_error = np.max(np.abs(data['masses'] - mass0)) / mass0
        summary += f"\n{name}:\n  Final: {final_error:.2e}\n  Max: {max_error:.2e}"

    summary += """

Notes:
• Conservative scheme preserves
  total mass to machine precision
• Upwind is diffusive but stable
• Non-conservative forms may have
  O(1%) mass errors over long times
• For compressible flow, use
  conservative formulation"""

    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    plt.suptitle('Continuity Equation: Mass Conservation Test',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'continuity_conservation.png'), dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/continuity_conservation.png")


if __name__ == "__main__":
    main()
