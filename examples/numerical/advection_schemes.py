"""
Experiment 14: Advection equation - upwind vs Lax-Friedrichs vs Godunov (numerical diffusion).

Compares different finite difference schemes for the advection equation,
demonstrating numerical diffusion and its dependence on the scheme.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def initial_square_wave(x, x1=0.1, x2=0.3):
    """Square wave initial condition."""
    return np.where((x >= x1) & (x <= x2), 1.0, 0.0)


def initial_gaussian(x, x0=0.2, sigma=0.04):
    """Gaussian initial condition."""
    return np.exp(-(x - x0)**2 / (2 * sigma**2))


def upwind_scheme(u, c, dx, dt):
    """
    First-order upwind scheme.

    For c > 0: u_i^{n+1} = u_i^n - c*dt/dx * (u_i^n - u_{i-1}^n)
    """
    nu = c * dt / dx
    u_new = np.zeros_like(u)

    if c > 0:
        u_new[1:] = u[1:] - nu * (u[1:] - u[:-1])
        u_new[0] = u[0] - nu * (u[0] - u[-1])  # Periodic BC
    else:
        u_new[:-1] = u[:-1] - nu * (u[1:] - u[:-1])
        u_new[-1] = u[-1] - nu * (u[0] - u[-1])

    return u_new


def lax_friedrichs_scheme(u, c, dx, dt):
    """
    Lax-Friedrichs scheme.

    u_i^{n+1} = 0.5*(u_{i+1}^n + u_{i-1}^n) - c*dt/(2*dx) * (u_{i+1}^n - u_{i-1}^n)
    """
    nu = c * dt / dx
    u_new = np.zeros_like(u)

    # Interior
    u_new[1:-1] = 0.5 * (u[2:] + u[:-2]) - nu/2 * (u[2:] - u[:-2])

    # Periodic BC
    u_new[0] = 0.5 * (u[1] + u[-1]) - nu/2 * (u[1] - u[-1])
    u_new[-1] = u_new[0]

    return u_new


def lax_wendroff_scheme(u, c, dx, dt):
    """
    Lax-Wendroff scheme (second-order accurate).

    u_i^{n+1} = u_i^n - c*dt/(2*dx) * (u_{i+1}^n - u_{i-1}^n)
              + (c*dt)^2/(2*dx^2) * (u_{i+1}^n - 2*u_i^n + u_{i-1}^n)
    """
    nu = c * dt / dx
    u_new = np.zeros_like(u)

    # Interior
    u_new[1:-1] = u[1:-1] - nu/2 * (u[2:] - u[:-2]) + \
                  nu**2/2 * (u[2:] - 2*u[1:-1] + u[:-2])

    # Periodic BC
    u_new[0] = u[0] - nu/2 * (u[1] - u[-1]) + nu**2/2 * (u[1] - 2*u[0] + u[-1])
    u_new[-1] = u_new[0]

    return u_new


def godunov_scheme(u, c, dx, dt):
    """
    Godunov scheme (exact Riemann solver for linear advection).

    For linear advection, this reduces to upwind.
    """
    return upwind_scheme(u, c, dx, dt)


def exact_solution(x, t, c, initial_func, L=1.0):
    """Exact solution (shift initial condition)."""
    x_shifted = np.mod(x - c * t, L)
    return initial_func(x_shifted)


def main():
    # Parameters
    c = 1.0  # Wave speed
    L = 1.0  # Domain length
    nx = 200

    x = np.linspace(0, L, nx, endpoint=False)
    dx = x[1] - x[0]

    # CFL number
    nu = 0.8
    dt = nu * dx / abs(c)

    t_final = 0.5
    n_steps = int(t_final / dt)

    # Schemes to compare
    schemes = {
        'Upwind': upwind_scheme,
        'Lax-Friedrichs': lax_friedrichs_scheme,
        'Lax-Wendroff': lax_wendroff_scheme,
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Test with square wave
    u0_square = initial_square_wave(x)

    for idx, (name, scheme) in enumerate(schemes.items()):
        ax = axes[0, idx]

        u = u0_square.copy()
        for _ in range(n_steps):
            u = scheme(u, c, dx, dt)

        u_exact = exact_solution(x, t_final, c, initial_square_wave, L)

        ax.plot(x, u0_square, 'k--', lw=1, alpha=0.5, label='Initial')
        ax.plot(x, u_exact, 'g-', lw=2, alpha=0.5, label='Exact')
        ax.plot(x, u, 'b-', lw=1.5, label=name)

        # Compute L2 error
        l2_error = np.sqrt(np.mean((u - u_exact)**2))

        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.set_title(f'{name} (Square Wave)\nL² error: {l2_error:.4f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.3, 1.5)

    # Test with Gaussian
    u0_gauss = initial_gaussian(x)

    for idx, (name, scheme) in enumerate(schemes.items()):
        ax = axes[1, idx]

        u = u0_gauss.copy()
        for _ in range(n_steps):
            u = scheme(u, c, dx, dt)

        u_exact = exact_solution(x, t_final, c, initial_gaussian, L)

        ax.plot(x, u0_gauss, 'k--', lw=1, alpha=0.5, label='Initial')
        ax.plot(x, u_exact, 'g-', lw=2, alpha=0.5, label='Exact')
        ax.plot(x, u, 'b-', lw=1.5, label=name)

        # Compute L2 error
        l2_error = np.sqrt(np.mean((u - u_exact)**2))

        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.set_title(f'{name} (Gaussian)\nL² error: {l2_error:.4f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.2)

    plt.suptitle(f'Advection Equation: Comparison of Finite Difference Schemes\n' +
                 f'c = {c}, CFL = {nu}, t = {t_final}, nx = {nx}',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'advection_schemes.png'), dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/advection_schemes.png")


if __name__ == "__main__":
    main()
