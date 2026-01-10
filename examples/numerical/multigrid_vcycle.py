"""
Experiment 16: Multigrid V-cycle for Poisson equation.

Demonstrates the V-cycle multigrid algorithm for solving the 2D Poisson equation,
comparing convergence rates of Jacobi, Gauss-Seidel, and Multigrid methods,
and showing how multigrid achieves O(N) complexity through level-by-level analysis.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter


def poisson_rhs(x, y):
    """
    Right-hand side of Poisson equation: -nabla^2 u = f

    Choose f such that exact solution is u = sin(pi*x)*sin(pi*y)
    Then f = 2*pi^2*sin(pi*x)*sin(pi*y)
    """
    return 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)


def exact_solution(x, y):
    """Exact solution: u = sin(pi*x)*sin(pi*y)"""
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def compute_residual(u, f, h):
    """
    Compute residual r = f - A*u for the discrete Laplacian.

    The discrete Laplacian uses:
    (nabla^2 u)_ij = (u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4*u_{i,j}) / h^2
    """
    r = np.zeros_like(u)
    r[1:-1, 1:-1] = f[1:-1, 1:-1] - (
        u[2:, 1:-1] + u[:-2, 1:-1] +
        u[1:-1, 2:] + u[1:-1, :-2] -
        4 * u[1:-1, 1:-1]
    ) / h**2
    return r


def residual_norm(u, f, h):
    """Compute L2 norm of residual."""
    r = compute_residual(u, f, h)
    return np.sqrt(np.sum(r**2) / r.size)


def jacobi_iteration(u, f, h, num_iter=1):
    """
    Weighted Jacobi iteration for Poisson equation.

    u_new = (1-omega)*u + omega * (h^2*f + u_neighbors) / 4
    """
    omega = 2.0 / 3.0  # Optimal damping for multigrid smoothing
    for _ in range(num_iter):
        u_new = u.copy()
        u_new[1:-1, 1:-1] = (1 - omega) * u[1:-1, 1:-1] + omega * 0.25 * (
            u[2:, 1:-1] + u[:-2, 1:-1] +
            u[1:-1, 2:] + u[1:-1, :-2] +
            h**2 * f[1:-1, 1:-1]
        )
        u = u_new
    return u


def gauss_seidel_iteration(u, f, h, num_iter=1):
    """
    Red-black Gauss-Seidel iteration for Poisson equation.

    Updates in checkerboard pattern for potential parallelization.
    """
    for _ in range(num_iter):
        # Red sweep (i+j even)
        for i in range(1, u.shape[0] - 1):
            for j in range(1, u.shape[1] - 1):
                if (i + j) % 2 == 0:
                    u[i, j] = 0.25 * (
                        u[i+1, j] + u[i-1, j] +
                        u[i, j+1] + u[i, j-1] +
                        h**2 * f[i, j]
                    )
        # Black sweep (i+j odd)
        for i in range(1, u.shape[0] - 1):
            for j in range(1, u.shape[1] - 1):
                if (i + j) % 2 == 1:
                    u[i, j] = 0.25 * (
                        u[i+1, j] + u[i-1, j] +
                        u[i, j+1] + u[i, j-1] +
                        h**2 * f[i, j]
                    )
    return u


def restrict_full_weighting(r_fine):
    """
    Restrict fine grid to coarse grid using full weighting operator.

    Stencil (weights multiply by 1/16):
        1  2  1
        2  4  2
        1  2  1
    """
    n_fine = r_fine.shape[0]
    n_coarse = (n_fine + 1) // 2
    r_coarse = np.zeros((n_coarse, n_coarse))

    for i in range(1, n_coarse - 1):
        for j in range(1, n_coarse - 1):
            i_f = 2 * i
            j_f = 2 * j
            # Full weighting stencil
            r_coarse[i, j] = (
                4 * r_fine[i_f, j_f] +
                2 * (r_fine[i_f+1, j_f] + r_fine[i_f-1, j_f] +
                     r_fine[i_f, j_f+1] + r_fine[i_f, j_f-1]) +
                1 * (r_fine[i_f+1, j_f+1] + r_fine[i_f+1, j_f-1] +
                     r_fine[i_f-1, j_f+1] + r_fine[i_f-1, j_f-1])
            ) / 16.0

    return r_coarse


def interpolate_bilinear(e_coarse):
    """
    Interpolate coarse grid correction to fine grid using bilinear interpolation.
    """
    n_coarse = e_coarse.shape[0]
    n_fine = 2 * n_coarse - 1
    e_fine = np.zeros((n_fine, n_fine))

    # Direct injection of coarse points
    e_fine[::2, ::2] = e_coarse

    # Interpolate in x-direction (horizontal edges)
    e_fine[1:-1:2, ::2] = 0.5 * (e_fine[:-2:2, ::2] + e_fine[2::2, ::2])

    # Interpolate in y-direction (vertical edges and centers)
    e_fine[:, 1:-1:2] = 0.5 * (e_fine[:, :-2:2] + e_fine[:, 2::2])

    return e_fine


def v_cycle(u, f, h, level=0, max_level=4, nu1=2, nu2=2, level_residuals=None):
    """
    V-cycle multigrid iteration.

    Args:
        u: Current approximation
        f: Right-hand side
        h: Grid spacing
        level: Current level (0 = finest)
        max_level: Maximum coarsening level
        nu1: Number of pre-smoothing iterations
        nu2: Number of post-smoothing iterations
        level_residuals: Dict to store residuals at each level (for visualization)

    Returns:
        Updated approximation u
    """
    n = u.shape[0]

    # Record residual at this level (before smoothing)
    if level_residuals is not None and level not in level_residuals:
        level_residuals[level] = []
    if level_residuals is not None:
        level_residuals[level].append(residual_norm(u, f, h))

    # Coarsest level: solve directly with many iterations
    if level == max_level or n <= 5:
        for _ in range(50):
            u = gauss_seidel_iteration(u, f, h, num_iter=1)
        return u

    # Pre-smoothing
    u = gauss_seidel_iteration(u, f, h, num_iter=nu1)

    # Compute residual
    r = compute_residual(u, f, h)

    # Restrict residual to coarse grid
    r_coarse = restrict_full_weighting(r)
    h_coarse = 2 * h

    # Solve error equation A*e = r on coarse grid
    e_coarse = np.zeros_like(r_coarse)
    e_coarse = v_cycle(e_coarse, r_coarse, h_coarse, level + 1, max_level,
                       nu1, nu2, level_residuals)

    # Interpolate correction to fine grid
    e_fine = interpolate_bilinear(e_coarse)

    # Correct
    u = u + e_fine

    # Post-smoothing
    u = gauss_seidel_iteration(u, f, h, num_iter=nu2)

    return u


def solve_with_method(n, method, max_iter=1000, tol=1e-10):
    """
    Solve Poisson equation with specified method.

    Returns:
        u: Solution
        residual_history: List of residuals at each iteration
        time_elapsed: Time taken
    """
    h = 1.0 / (n - 1)
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)

    f = poisson_rhs(X, Y)
    u = np.zeros((n, n))

    residual_history = []
    level_residuals = {} if method == 'multigrid' else None

    start_time = perf_counter()

    for iteration in range(max_iter):
        if method == 'jacobi':
            u = jacobi_iteration(u, f, h, num_iter=1)
        elif method == 'gauss_seidel':
            u = gauss_seidel_iteration(u, f, h, num_iter=1)
        elif method == 'multigrid':
            level_residuals_cycle = {}
            u = v_cycle(u, f, h, level=0, max_level=4, nu1=2, nu2=2,
                       level_residuals=level_residuals_cycle)
            # Aggregate level residuals
            for lvl, res_list in level_residuals_cycle.items():
                if lvl not in level_residuals:
                    level_residuals[lvl] = []
                level_residuals[lvl].extend(res_list)

        res_norm = residual_norm(u, f, h)
        residual_history.append(res_norm)

        if res_norm < tol:
            break

    elapsed = perf_counter() - start_time

    return u, residual_history, elapsed, X, Y, level_residuals


def complexity_analysis(grid_sizes):
    """
    Analyze complexity by measuring iterations and time to converge
    for different grid sizes.
    """
    results = {
        'jacobi': {'iters': [], 'times': []},
        'gauss_seidel': {'iters': [], 'times': []},
        'multigrid': {'iters': [], 'times': []}
    }

    tol = 1e-6

    for n in grid_sizes:
        print(f"  Grid size {n}x{n}...")

        for method in ['jacobi', 'gauss_seidel', 'multigrid']:
            # Limit iterations for slow methods on large grids
            max_iter = 10000 if method != 'multigrid' else 100
            if method == 'jacobi' and n > 65:
                max_iter = min(max_iter, 2000)  # Jacobi is very slow

            _, residuals, elapsed, _, _, _ = solve_with_method(n, method, max_iter, tol)

            # Count iterations to reach tolerance
            n_iter = len(residuals)
            if residuals[-1] > tol:
                n_iter = max_iter  # Did not converge

            results[method]['iters'].append(n_iter)
            results[method]['times'].append(elapsed)

    return results


def main():
    print("Multigrid V-cycle for Poisson Equation")
    print("=" * 50)

    # Main solution demonstration
    n = 65  # Grid size (2^k + 1 for multigrid compatibility)

    print("\nSolving Poisson equation on 65x65 grid...")

    # Solve with different methods
    methods = ['jacobi', 'gauss_seidel', 'multigrid']
    results = {}

    for method in methods:
        print(f"  {method.capitalize()}...")
        u, res_hist, elapsed, X, Y, level_res = solve_with_method(
            n, method, max_iter=2000, tol=1e-10
        )
        results[method] = {
            'u': u,
            'residuals': res_hist,
            'time': elapsed,
            'level_residuals': level_res
        }
        print(f"    Converged in {len(res_hist)} iterations ({elapsed:.3f}s)")

    # Complexity analysis
    print("\nAnalyzing complexity scaling...")
    grid_sizes = [17, 33, 65, 129]
    complexity = complexity_analysis(grid_sizes)

    # Create 4-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel 1: Solution visualization
    ax = axes[0, 0]
    u_mg = results['multigrid']['u']
    u_exact = exact_solution(X, Y)

    contour = ax.contourf(X, Y, u_mg, levels=30, cmap='viridis')
    plt.colorbar(contour, ax=ax, label='u(x,y)')
    ax.contour(X, Y, u_mg, levels=10, colors='white', alpha=0.3, linewidths=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Multigrid Solution\n(max error = {np.max(np.abs(u_mg - u_exact)):.2e})')
    ax.set_aspect('equal')

    # Panel 2: Convergence comparison
    ax = axes[0, 1]
    colors = {'jacobi': 'blue', 'gauss_seidel': 'green', 'multigrid': 'red'}
    linestyles = {'jacobi': '-', 'gauss_seidel': '--', 'multigrid': '-'}

    for method in methods:
        residuals = results[method]['residuals']
        ax.semilogy(residuals, color=colors[method], linestyle=linestyles[method],
                    lw=2, label=f'{method.replace("_", " ").title()} ({len(residuals)} iter)')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Residual Norm (L2)')
    ax.set_title('Convergence Rate Comparison\n(Multigrid achieves rapid convergence)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, min(500, max(len(r['residuals']) for r in results.values())))

    # Panel 3: Level-by-level residual behavior
    ax = axes[1, 0]

    # Run a single V-cycle with detailed level tracking
    h = 1.0 / (n - 1)
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    Xg, Yg = np.meshgrid(x, y)
    f = poisson_rhs(Xg, Yg)
    u_test = np.zeros((n, n))

    # Track residuals through multiple V-cycles
    level_history = {0: [], 1: [], 2: [], 3: []}
    for cycle in range(10):
        level_res_cycle = {}
        u_test = v_cycle(u_test, f, h, level=0, max_level=4, nu1=2, nu2=2,
                        level_residuals=level_res_cycle)
        for lvl in range(4):
            if lvl in level_res_cycle and level_res_cycle[lvl]:
                level_history[lvl].append(level_res_cycle[lvl][0])

    level_colors = plt.cm.plasma(np.linspace(0.2, 0.8, 4))
    level_labels = ['Level 0 (finest, 65x65)', 'Level 1 (33x33)',
                    'Level 2 (17x17)', 'Level 3 (9x9)']

    for lvl in range(4):
        if level_history[lvl]:
            ax.semilogy(level_history[lvl], 'o-', color=level_colors[lvl],
                       lw=1.5, markersize=4, label=level_labels[lvl])

    ax.set_xlabel('V-cycle Number')
    ax.set_ylabel('Residual Norm at Level Entry')
    ax.set_title('Level-by-Level Residual Reduction\n(Coarse grids eliminate low-frequency errors)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 4: Complexity scaling (O(N) demonstration)
    ax = axes[1, 1]

    N_values = [g**2 for g in grid_sizes]  # Total unknowns

    # Plot iterations vs N
    ax_iter = ax
    ax_time = ax.twinx()

    # Iterations (left y-axis)
    for method in ['jacobi', 'gauss_seidel', 'multigrid']:
        iters = complexity[method]['iters']
        ax_iter.plot(N_values, iters, 'o-', color=colors[method], lw=2,
                    label=f'{method.replace("_", " ").title()} iterations')

    ax_iter.set_xlabel('Number of Unknowns (N)')
    ax_iter.set_ylabel('Iterations to Converge (tol=1e-6)', color='black')
    ax_iter.set_xscale('log')
    ax_iter.set_yscale('log')

    # Add reference lines for complexity
    N_ref = np.array(N_values)
    # O(N) reference
    ax_iter.plot(N_ref, N_ref / N_ref[0] * 10, 'k:', alpha=0.5, label='O(N) reference')
    # O(N^2) reference
    ax_iter.plot(N_ref, (N_ref / N_ref[0])**2 * 10, 'k--', alpha=0.5, label='O(N^2) reference')

    ax_iter.set_title('Complexity Analysis: Multigrid = O(N)\n(Classical methods scale as O(N^2))')
    ax_iter.legend(loc='upper left', fontsize=8)
    ax_iter.grid(True, alpha=0.3)

    # Add work units annotation
    ax_iter.annotate('Multigrid:\nConstant iterations\n= O(N) work',
                     xy=(N_values[-1], complexity['multigrid']['iters'][-1]),
                     xytext=(N_values[-2], complexity['multigrid']['iters'][-1] * 10),
                     fontsize=9, ha='center',
                     arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))

    plt.suptitle('Multigrid V-cycle for 2D Poisson Equation\n' +
                 r'$-\nabla^2 u = f$, Dirichlet BC, exact solution: $u = \sin(\pi x)\sin(\pi y)$',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'multigrid_vcycle.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to {output_path}")

    # Print summary
    print("\nConvergence Summary:")
    print("-" * 60)
    print(f"{'Method':<20} {'Iterations':<15} {'Time (s)':<15}")
    print("-" * 60)
    for method in methods:
        n_iter = len(results[method]['residuals'])
        time_s = results[method]['time']
        print(f"{method.replace('_', ' ').title():<20} {n_iter:<15} {time_s:<15.4f}")

    print("\nComplexity Analysis (iterations to converge):")
    print("-" * 70)
    print(f"{'Grid':<10} {'N unknowns':<15} {'Jacobi':<15} {'Gauss-Seidel':<15} {'Multigrid':<15}")
    print("-" * 70)
    for i, g in enumerate(grid_sizes):
        N = g**2
        j_iter = complexity['jacobi']['iters'][i]
        gs_iter = complexity['gauss_seidel']['iters'][i]
        mg_iter = complexity['multigrid']['iters'][i]
        print(f"{g}x{g:<6} {N:<15} {j_iter:<15} {gs_iter:<15} {mg_iter:<15}")

    print("\nKey Insight: Multigrid achieves O(N) complexity because:")
    print("  1. Work at each level is proportional to grid points at that level")
    print("  2. Coarse grids efficiently eliminate low-frequency error components")
    print("  3. Number of V-cycles needed is independent of grid size")


if __name__ == "__main__":
    main()
