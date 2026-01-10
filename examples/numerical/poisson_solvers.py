"""
Experiment 13: 2D Poisson equation - Gauss-Seidel vs multigrid toy (convergence rates).

Compares iterative solvers for the Poisson equation,
demonstrating the acceleration from multigrid methods.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def poisson_rhs(x, y):
    """Right-hand side of Poisson equation: -∇²u = f"""
    # Choose f such that exact solution is known
    # u = sin(πx)sin(πy) => f = 2π²sin(πx)sin(πy)
    return 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)


def exact_solution(x, y):
    """Exact solution: u = sin(πx)sin(πy)"""
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def jacobi_iteration(u, f, dx):
    """One Jacobi iteration for Poisson equation."""
    u_new = np.zeros_like(u)
    u_new[1:-1, 1:-1] = 0.25 * (
        u[2:, 1:-1] + u[:-2, 1:-1] +
        u[1:-1, 2:] + u[1:-1, :-2] +
        dx**2 * f[1:-1, 1:-1]
    )
    return u_new


def gauss_seidel_iteration(u, f, dx):
    """One Gauss-Seidel iteration (in-place update)."""
    u_new = u.copy()
    for i in range(1, u.shape[0] - 1):
        for j in range(1, u.shape[1] - 1):
            u_new[i, j] = 0.25 * (
                u_new[i+1, j] + u_new[i-1, j] +
                u_new[i, j+1] + u_new[i, j-1] +
                dx**2 * f[i, j]
            )
    return u_new


def sor_iteration(u, f, dx, omega=1.5):
    """Successive Over-Relaxation (SOR) iteration."""
    u_new = u.copy()
    for i in range(1, u.shape[0] - 1):
        for j in range(1, u.shape[1] - 1):
            gs_value = 0.25 * (
                u_new[i+1, j] + u_new[i-1, j] +
                u_new[i, j+1] + u_new[i, j-1] +
                dx**2 * f[i, j]
            )
            u_new[i, j] = (1 - omega) * u_new[i, j] + omega * gs_value
    return u_new


def residual(u, f, dx):
    """Compute residual r = f - Au."""
    r = np.zeros_like(u)
    r[1:-1, 1:-1] = f[1:-1, 1:-1] - (
        -4 * u[1:-1, 1:-1] +
        u[2:, 1:-1] + u[:-2, 1:-1] +
        u[1:-1, 2:] + u[1:-1, :-2]
    ) / dx**2
    return r


def restrict(r_fine):
    """Restrict to coarse grid (full weighting)."""
    n_coarse = (r_fine.shape[0] + 1) // 2
    r_coarse = np.zeros((n_coarse, n_coarse))

    # Full weighting restriction
    for i in range(1, n_coarse - 1):
        for j in range(1, n_coarse - 1):
            i_f, j_f = 2*i, 2*j
            r_coarse[i, j] = 0.25 * r_fine[i_f, j_f] + \
                             0.125 * (r_fine[i_f+1, j_f] + r_fine[i_f-1, j_f] +
                                     r_fine[i_f, j_f+1] + r_fine[i_f, j_f-1]) + \
                             0.0625 * (r_fine[i_f+1, j_f+1] + r_fine[i_f+1, j_f-1] +
                                      r_fine[i_f-1, j_f+1] + r_fine[i_f-1, j_f-1])
    return r_coarse


def interpolate(e_coarse):
    """Interpolate to fine grid (bilinear)."""
    n_fine = 2 * e_coarse.shape[0] - 1
    e_fine = np.zeros((n_fine, n_fine))

    # Copy coarse grid points
    e_fine[::2, ::2] = e_coarse

    # Interpolate in x-direction
    e_fine[1:-1:2, ::2] = 0.5 * (e_fine[:-2:2, ::2] + e_fine[2::2, ::2])

    # Interpolate in y-direction
    e_fine[:, 1:-1:2] = 0.5 * (e_fine[:, :-2:2] + e_fine[:, 2::2])

    return e_fine


def v_cycle(u, f, dx, n_pre=2, n_post=2, level=0, max_level=4):
    """V-cycle multigrid iteration."""
    n = u.shape[0]

    # At coarsest level, solve directly
    if level == max_level or n <= 5:
        for _ in range(50):
            u = gauss_seidel_iteration(u, f, dx)
        return u

    # Pre-smoothing
    for _ in range(n_pre):
        u = gauss_seidel_iteration(u, f, dx)

    # Compute residual
    r = residual(u, f, dx)

    # Restrict residual to coarse grid
    r_coarse = restrict(r)
    dx_coarse = 2 * dx

    # Solve error equation on coarse grid
    e_coarse = np.zeros_like(r_coarse)
    e_coarse = v_cycle(e_coarse, r_coarse, dx_coarse, n_pre, n_post, level+1, max_level)

    # Interpolate correction to fine grid
    e_fine = interpolate(e_coarse)

    # Correct
    u = u + e_fine

    # Post-smoothing
    for _ in range(n_post):
        u = gauss_seidel_iteration(u, f, dx)

    return u


def solve_poisson(n, method='jacobi', max_iter=10000, tol=1e-8):
    """Solve Poisson equation with specified method."""
    dx = 1.0 / (n - 1)
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)

    f = poisson_rhs(X, Y)
    u_exact = exact_solution(X, Y)

    u = np.zeros((n, n))  # Initial guess
    errors = []
    residuals = []

    for iteration in range(max_iter):
        if method == 'jacobi':
            u = jacobi_iteration(u, f, dx)
        elif method == 'gauss_seidel':
            u = gauss_seidel_iteration(u, f, dx)
        elif method == 'sor':
            # Optimal omega for Poisson on square grid
            omega = 2 / (1 + np.sin(np.pi * dx))
            u = sor_iteration(u, f, dx, omega)
        elif method == 'multigrid':
            u = v_cycle(u, f, dx)

        err = np.max(np.abs(u - u_exact))
        res = np.max(np.abs(residual(u, f, dx)))
        errors.append(err)
        residuals.append(res)

        if res < tol:
            break

    return u, errors, residuals, X, Y, u_exact


def main():
    n = 33  # Grid size (2^k + 1 for multigrid)
    max_iter = 500

    methods = ['jacobi', 'gauss_seidel', 'sor', 'multigrid']
    colors = {'jacobi': 'blue', 'gauss_seidel': 'green', 'sor': 'orange', 'multigrid': 'red'}

    results = {}
    print("Solving Poisson equation...")
    for method in methods:
        print(f"  {method}...")
        u, errors, residuals, X, Y, u_exact = solve_poisson(n, method, max_iter)
        results[method] = {
            'u': u, 'errors': errors, 'residuals': residuals,
            'X': X, 'Y': Y, 'exact': u_exact
        }

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Convergence of residual
    ax = axes[0, 0]
    for method in methods:
        residuals = results[method]['residuals']
        ax.semilogy(residuals, color=colors[method], lw=2, label=method)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Max Residual')
    ax.set_title('Residual Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Convergence of error
    ax = axes[0, 1]
    for method in methods:
        errors = results[method]['errors']
        ax.semilogy(errors, color=colors[method], lw=2, label=method)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Max Error')
    ax.set_title('Error Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Final solution (multigrid)
    ax = axes[1, 0]
    X = results['multigrid']['X']
    Y = results['multigrid']['Y']
    u = results['multigrid']['u']

    contour = ax.contourf(X, Y, u, levels=20, cmap='viridis')
    plt.colorbar(contour, ax=ax, label='u')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Multigrid Solution')
    ax.set_aspect('equal')

    # Plot 4: Error distribution (multigrid)
    ax = axes[1, 1]
    error = np.abs(u - results['multigrid']['exact'])

    contour = ax.contourf(X, Y, error, levels=20, cmap='hot')
    plt.colorbar(contour, ax=ax, label='|Error|')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Multigrid Error (max = {np.max(error):.2e})')
    ax.set_aspect('equal')

    plt.suptitle('2D Poisson Equation Solvers Comparison\n' +
                 f'Grid: {n}×{n}, -∇²u = f, u = sin(πx)sin(πy)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'poisson_solvers.png'), dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/poisson_solvers.png")

    # Print summary
    print("\nConvergence summary (iterations to reach tol=1e-8):")
    for method in methods:
        n_iter = len(results[method]['residuals'])
        final_res = results[method]['residuals'][-1]
        print(f"  {method}: {n_iter} iterations, final residual = {final_res:.2e}")


if __name__ == "__main__":
    main()
