"""
Experiment 11: 1D diffusion equation - explicit vs implicit (stability CFL).

Compares explicit and implicit finite difference methods for the heat/diffusion
equation, demonstrating the CFL stability condition.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded


def initial_gaussian(x, x0=0.5, sigma=0.05):
    """Gaussian initial condition."""
    return np.exp(-(x - x0)**2 / (2 * sigma**2))


def analytical_diffusion(x, t, D, x0=0.5, sigma0=0.05):
    """Analytical solution for Gaussian diffusion."""
    sigma_t = np.sqrt(sigma0**2 + 2 * D * t)
    return (sigma0 / sigma_t) * np.exp(-(x - x0)**2 / (2 * sigma_t**2))


def explicit_ftcs(u, D, dx, dt):
    """
    Forward Time Central Space (explicit) scheme.

    Stability condition: r = D*dt/dx^2 <= 0.5
    """
    r = D * dt / dx**2
    u_new = u.copy()
    u_new[1:-1] = u[1:-1] + r * (u[2:] - 2*u[1:-1] + u[:-2])
    return u_new


def implicit_btcs(u, D, dx, dt):
    """
    Backward Time Central Space (implicit) scheme.

    Unconditionally stable but requires solving tridiagonal system.
    """
    r = D * dt / dx**2
    n = len(u)

    # Tridiagonal matrix: -r, (1+2r), -r
    # Using scipy's banded matrix solver
    ab = np.zeros((3, n))
    ab[0, 2:] = -r      # Upper diagonal
    ab[1, :] = 1 + 2*r  # Main diagonal
    ab[2, :-2] = -r     # Lower diagonal

    # Boundary conditions (Dirichlet u=0)
    ab[1, 0] = 1
    ab[1, -1] = 1

    rhs = u.copy()
    rhs[0] = 0
    rhs[-1] = 0

    return solve_banded((1, 1), ab, rhs)


def crank_nicolson(u, D, dx, dt):
    """
    Crank-Nicolson (implicit) scheme.

    Second order in time and space, unconditionally stable.
    """
    r = D * dt / dx**2 / 2
    n = len(u)

    # LHS matrix: -r, (1+2r), -r
    ab = np.zeros((3, n))
    ab[0, 2:] = -r
    ab[1, :] = 1 + 2*r
    ab[2, :-2] = -r
    ab[1, 0] = 1
    ab[1, -1] = 1

    # RHS: explicit part
    rhs = np.zeros(n)
    rhs[1:-1] = r * u[2:] + (1 - 2*r) * u[1:-1] + r * u[:-2]
    rhs[0] = 0
    rhs[-1] = 0

    return solve_banded((1, 1), ab, rhs)


def main():
    # Parameters
    D = 0.01  # Diffusion coefficient
    L = 1.0   # Domain length
    nx = 51   # Number of spatial points

    x = np.linspace(0, L, nx)
    dx = x[1] - x[0]

    # Different time steps to test stability
    dt_stable = 0.4 * dx**2 / D  # r = 0.4 (stable)
    dt_marginal = 0.5 * dx**2 / D  # r = 0.5 (marginally stable)
    dt_unstable = 0.6 * dx**2 / D  # r = 0.6 (unstable for explicit)

    t_final = 0.5

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Test each time step
    for col, (dt, label) in enumerate([(dt_stable, 'r=0.4 (stable)'),
                                        (dt_marginal, 'r=0.5 (marginal)'),
                                        (dt_unstable, 'r=0.6 (unstable)')]):
        r = D * dt / dx**2
        n_steps = int(t_final / dt)

        # Initial condition
        u_expl = initial_gaussian(x)
        u_impl = initial_gaussian(x)
        u_cn = initial_gaussian(x)

        # Time evolution
        for _ in range(n_steps):
            u_expl = explicit_ftcs(u_expl, D, dx, dt)
            u_impl = implicit_btcs(u_impl, D, dx, dt)
            u_cn = crank_nicolson(u_cn, D, dx, dt)

        # Analytical solution
        u_exact = analytical_diffusion(x, t_final, D)

        # Plot solutions
        ax = axes[0, col]
        ax.plot(x, u_exact, 'k-', lw=2, label='Exact')
        ax.plot(x, u_expl, 'r--', lw=1.5, label='Explicit FTCS')
        ax.plot(x, u_impl, 'b-.', lw=1.5, label='Implicit BTCS')
        ax.plot(x, u_cn, 'g:', lw=1.5, label='Crank-Nicolson')

        ax.set_xlabel('x')
        ax.set_ylabel('u(x,t)')
        ax.set_title(f'{label}\ndt = {dt:.5f}, {n_steps} steps')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.5, 1.5)

        # Plot errors
        ax = axes[1, col]
        err_expl = np.abs(u_expl - u_exact)
        err_impl = np.abs(u_impl - u_exact)
        err_cn = np.abs(u_cn - u_exact)

        # Clip for visualization if unstable
        err_expl = np.clip(err_expl, 0, 10)

        ax.semilogy(x, err_expl + 1e-16, 'r-', lw=1.5, label=f'Explicit (max={np.max(err_expl):.2e})')
        ax.semilogy(x, err_impl + 1e-16, 'b-', lw=1.5, label=f'Implicit (max={np.max(err_impl):.2e})')
        ax.semilogy(x, err_cn + 1e-16, 'g-', lw=1.5, label=f'CN (max={np.max(err_cn):.2e})')

        ax.set_xlabel('x')
        ax.set_ylabel('|Error|')
        ax.set_title('Absolute Error')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('1D Diffusion Equation: Explicit vs Implicit Methods\n' +
                 f'D = {D}, dx = {dx:.4f}, CFL stability: r = D·dt/dx² ≤ 0.5',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'diffusion_explicit_implicit.png'), dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/diffusion_explicit_implicit.png")


if __name__ == "__main__":
    main()
