"""
Experiment 16: Multigrid V-cycle for Poisson equation.

Demonstrates the V-cycle multigrid algorithm for solving the 2D Poisson equation,
comparing convergence rates of Jacobi, Gauss-Seidel, and Multigrid methods,
and showing how multigrid achieves O(N) complexity through level-by-level analysis.

The key insight is that iterative methods like Jacobi and Gauss-Seidel efficiently
reduce high-frequency error components but struggle with low-frequency errors.
Multigrid exploits the fact that low-frequency errors on a fine grid appear as
high-frequency errors on a coarser grid, enabling rapid convergence.
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
    Compute residual r = f + nabla^2 u for the discrete Laplacian.

    For equation -nabla^2 u = f, we have A*u = f where A = -nabla^2
    Residual is r = f - A*u = f + nabla^2 u
    """
    r = np.zeros_like(u)
    # Discrete Laplacian: (u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4*u_{i,j}) / h^2
    r[1:-1, 1:-1] = f[1:-1, 1:-1] + (
        u[2:, 1:-1] + u[:-2, 1:-1] +
        u[1:-1, 2:] + u[1:-1, :-2] -
        4 * u[1:-1, 1:-1]
    ) / h**2
    return r


def residual_norm(u, f, h):
    """Compute L2 norm of residual."""
    r = compute_residual(u, f, h)
    return np.sqrt(np.sum(r**2) * h**2)


def jacobi_smooth(u, f, h, num_iter=1, omega=0.8):
    """
    Weighted Jacobi iteration for smoothing.

    Solves -nabla^2 u = f
    Update: u_new = u + omega * (residual) / (4/h^2)
    """
    for _ in range(num_iter):
        u_new = u.copy()
        u_new[1:-1, 1:-1] = (1 - omega) * u[1:-1, 1:-1] + omega * 0.25 * (
            u[2:, 1:-1] + u[:-2, 1:-1] +
            u[1:-1, 2:] + u[1:-1, :-2] +
            h**2 * f[1:-1, 1:-1]
        )
        u = u_new
    return u


def gauss_seidel_smooth(u, f, h, num_iter=1):
    """
    Gauss-Seidel iteration with red-black ordering.

    Solves -nabla^2 u = f
    """
    u = u.copy()
    for _ in range(num_iter):
        # Red sweep (i+j even)
        for i in range(1, u.shape[0] - 1):
            for j in range(1 + (i % 2), u.shape[1] - 1, 2):
                u[i, j] = 0.25 * (
                    u[i+1, j] + u[i-1, j] +
                    u[i, j+1] + u[i, j-1] +
                    h**2 * f[i, j]
                )
        # Black sweep (i+j odd)
        for i in range(1, u.shape[0] - 1):
            for j in range(1 + ((i + 1) % 2), u.shape[1] - 1, 2):
                u[i, j] = 0.25 * (
                    u[i+1, j] + u[i-1, j] +
                    u[i, j+1] + u[i, j-1] +
                    h**2 * f[i, j]
                )
    return u


def restrict(r_fine):
    """
    Restrict fine grid to coarse grid using full weighting.

    Coarse grid has (n_fine + 1) / 2 points.
    """
    n_fine = r_fine.shape[0]
    n_coarse = (n_fine + 1) // 2
    r_coarse = np.zeros((n_coarse, n_coarse))

    # Full weighting restriction
    for i in range(1, n_coarse - 1):
        for j in range(1, n_coarse - 1):
            i_f = 2 * i
            j_f = 2 * j
            r_coarse[i, j] = (
                4 * r_fine[i_f, j_f] +
                2 * (r_fine[i_f+1, j_f] + r_fine[i_f-1, j_f] +
                     r_fine[i_f, j_f+1] + r_fine[i_f, j_f-1]) +
                (r_fine[i_f+1, j_f+1] + r_fine[i_f+1, j_f-1] +
                 r_fine[i_f-1, j_f+1] + r_fine[i_f-1, j_f-1])
            ) / 16.0

    return r_coarse


def prolongate(e_coarse):
    """
    Prolongate (interpolate) coarse grid to fine grid using bilinear interpolation.
    """
    n_coarse = e_coarse.shape[0]
    n_fine = 2 * n_coarse - 1
    e_fine = np.zeros((n_fine, n_fine))

    # Copy coarse points directly
    e_fine[::2, ::2] = e_coarse

    # Interpolate horizontal edges
    for i in range(0, n_fine, 2):
        for j in range(1, n_fine - 1, 2):
            e_fine[i, j] = 0.5 * (e_fine[i, j-1] + e_fine[i, j+1])

    # Interpolate vertical edges and centers
    for i in range(1, n_fine - 1, 2):
        for j in range(n_fine):
            e_fine[i, j] = 0.5 * (e_fine[i-1, j] + e_fine[i+1, j])

    return e_fine


def v_cycle(u, f, h, nu1=2, nu2=2, level=0, max_levels=10, level_data=None):
    """
    V-cycle multigrid iteration.

    Args:
        u: Current approximation
        f: Right-hand side
        h: Grid spacing
        nu1: Pre-smoothing iterations
        nu2: Post-smoothing iterations
        level: Current recursion level
        max_levels: Maximum number of levels
        level_data: Optional dict to record level-wise data

    Returns:
        Updated solution u
    """
    n = u.shape[0]

    # Record data at this level if requested
    if level_data is not None:
        if level not in level_data:
            level_data[level] = {'residuals': [], 'grid_size': n}
        level_data[level]['residuals'].append(residual_norm(u, f, h))

    # Base case: solve directly at coarsest level
    if n <= 3 or level >= max_levels:
        # Direct solve with many iterations
        for _ in range(100):
            u = gauss_seidel_smooth(u, f, h, num_iter=1)
        return u

    # Pre-smoothing
    u = gauss_seidel_smooth(u, f, h, num_iter=nu1)

    # Compute residual r = f - Au
    r = compute_residual(u, f, h)

    # Restrict residual to coarse grid
    r_coarse = restrict(r)

    # Solve coarse grid error equation: A * e = r
    e_coarse = np.zeros_like(r_coarse)
    h_coarse = 2 * h
    e_coarse = v_cycle(e_coarse, r_coarse, h_coarse, nu1, nu2,
                       level + 1, max_levels, level_data)

    # Prolongate (interpolate) error to fine grid
    e_fine = prolongate(e_coarse)

    # Correct: u = u + e
    u = u + e_fine

    # Post-smoothing
    u = gauss_seidel_smooth(u, f, h, num_iter=nu2)

    return u


def solve_poisson(n, method, max_iter=1000, tol=1e-10):
    """
    Solve Poisson equation with specified method.

    Args:
        n: Grid size (must be 2^k + 1 for multigrid)
        method: 'jacobi', 'gauss_seidel', or 'multigrid'
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        u: Solution array
        residual_history: List of residual norms
        elapsed_time: Wall-clock time
        X, Y: Meshgrid coordinates
    """
    h = 1.0 / (n - 1)
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)

    f = poisson_rhs(X, Y)
    u = np.zeros((n, n))  # Initial guess with zero boundary conditions

    residual_history = []
    level_data_all = {}

    start_time = perf_counter()

    for iteration in range(max_iter):
        if method == 'jacobi':
            u = jacobi_smooth(u, f, h, num_iter=1)
        elif method == 'gauss_seidel':
            u = gauss_seidel_smooth(u, f, h, num_iter=1)
        elif method == 'multigrid':
            level_data = {}
            u = v_cycle(u, f, h, nu1=2, nu2=2, level=0, max_levels=10,
                       level_data=level_data)
            # Collect level data
            for lvl, data in level_data.items():
                if lvl not in level_data_all:
                    level_data_all[lvl] = []
                level_data_all[lvl].append(data['residuals'][0])

        res = residual_norm(u, f, h)
        residual_history.append(res)

        if res < tol:
            break

    elapsed = perf_counter() - start_time

    return u, residual_history, elapsed, X, Y, level_data_all


def analyze_complexity(grid_sizes, tol=1e-6):
    """
    Analyze iteration count and timing for different grid sizes.
    """
    results = {
        'jacobi': {'iters': [], 'times': []},
        'gauss_seidel': {'iters': [], 'times': []},
        'multigrid': {'iters': [], 'times': []}
    }

    for n in grid_sizes:
        print(f"  Analyzing {n}x{n} grid...")

        for method in ['jacobi', 'gauss_seidel', 'multigrid']:
            # Limit max iterations based on method
            if method == 'multigrid':
                max_iter = 50
            elif method == 'gauss_seidel':
                max_iter = 5000
            else:  # jacobi
                max_iter = 10000

            _, residuals, elapsed, _, _, _ = solve_poisson(n, method, max_iter, tol)

            n_iter = len(residuals)
            converged = residuals[-1] < tol if residuals else False

            results[method]['iters'].append(n_iter if converged else max_iter)
            results[method]['times'].append(elapsed)

    return results


def main():
    print("=" * 60)
    print("Experiment 16: Multigrid V-cycle for Poisson Equation")
    print("=" * 60)

    # Parameters
    n = 65  # Grid size (2^6 + 1)
    tol = 1e-10

    print(f"\nSolving -nabla^2 u = f on {n}x{n} grid")
    print(f"Exact solution: u = sin(pi*x)*sin(pi*y)")

    # Solve with each method
    methods = ['jacobi', 'gauss_seidel', 'multigrid']
    results = {}

    for method in methods:
        print(f"\n  {method.replace('_', ' ').title()}...", end=" ", flush=True)
        if method == 'multigrid':
            max_iter = 20
        elif method == 'gauss_seidel':
            max_iter = 1000
        else:
            max_iter = 2000

        u, res_hist, elapsed, X, Y, level_data = solve_poisson(
            n, method, max_iter=max_iter, tol=tol
        )

        results[method] = {
            'u': u, 'residuals': res_hist, 'time': elapsed,
            'X': X, 'Y': Y, 'level_data': level_data
        }

        converged = "converged" if res_hist[-1] < tol else "max iter"
        print(f"{len(res_hist)} iter, {elapsed:.3f}s ({converged})")

    # Complexity analysis
    print("\nComplexity Analysis:")
    grid_sizes = [17, 33, 65, 129]
    complexity = analyze_complexity(grid_sizes, tol=1e-6)

    # Create 4-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Colors for methods
    colors = {'jacobi': '#1f77b4', 'gauss_seidel': '#2ca02c', 'multigrid': '#d62728'}

    # Panel 1: Solution and error
    ax = axes[0, 0]
    X, Y = results['multigrid']['X'], results['multigrid']['Y']
    u_mg = results['multigrid']['u']
    u_exact = exact_solution(X, Y)

    cf = ax.contourf(X, Y, u_mg, levels=25, cmap='RdBu_r')
    plt.colorbar(cf, ax=ax, label='u(x,y)', shrink=0.8)
    ax.contour(X, Y, u_mg, levels=10, colors='k', alpha=0.3, linewidths=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    max_err = np.max(np.abs(u_mg - u_exact))
    ax.set_title(f'Multigrid Solution (max error = {max_err:.2e})')
    ax.set_aspect('equal')

    # Panel 2: Convergence comparison
    ax = axes[0, 1]

    for method in methods:
        res = results[method]['residuals']
        n_iter = len(res)
        label = f'{method.replace("_", " ").title()} ({n_iter} iter)'
        ax.semilogy(res, '-', color=colors[method], lw=2, label=label)

    ax.axhline(y=tol, color='gray', linestyle='--', alpha=0.5, label=f'Tolerance = {tol}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Residual Norm (L2)')
    ax.set_title('Convergence Rate Comparison')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    # Panel 3: Level-by-level residual reduction (V-cycle anatomy)
    ax = axes[1, 0]

    # Track residuals at each level during V-cycles
    h = 1.0 / (n - 1)
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    Xg, Yg = np.meshgrid(x, y)
    f = poisson_rhs(Xg, Yg)
    u_track = np.zeros((n, n))

    n_cycles = 10
    level_history = {}

    for cycle in range(n_cycles):
        level_data = {}
        u_track = v_cycle(u_track, f, h, nu1=2, nu2=2, level=0, max_levels=6,
                          level_data=level_data)
        for lvl, data in level_data.items():
            if lvl not in level_history:
                level_history[lvl] = []
            if data['residuals']:
                level_history[lvl].append(data['residuals'][0])

    level_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(level_history)))
    grid_sizes_lvl = [n // (2**lvl) + (1 if n // (2**lvl) > 0 else 0)
                      for lvl in range(len(level_history))]

    for lvl in sorted(level_history.keys()):
        if level_history[lvl]:
            gs = n // (2**lvl)
            label = f'Level {lvl} ({gs}x{gs})'
            ax.semilogy(range(1, len(level_history[lvl]) + 1),
                       level_history[lvl], 'o-', color=level_colors[lvl],
                       lw=1.5, markersize=5, label=label)

    ax.set_xlabel('V-cycle Number')
    ax.set_ylabel('Residual Norm at Level Entry')
    ax.set_title('Level-by-Level Residual Reduction')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: Complexity analysis - O(N) demonstration
    ax = axes[1, 1]

    N_values = np.array([g**2 for g in grid_sizes])

    for method in methods:
        iters = complexity[method]['iters']
        marker = 'o' if method != 'multigrid' else 's'
        ax.loglog(N_values, iters, marker + '-', color=colors[method], lw=2,
                  markersize=8, label=method.replace('_', ' ').title())

    # Reference lines
    N_ref = np.array([N_values[0], N_values[-1]])
    # O(1) - constant iterations
    ax.loglog(N_ref, [10, 10], 'k:', alpha=0.4, lw=1.5, label='O(1) reference')
    # O(N) scaling
    ax.loglog(N_ref, N_ref / N_ref[0] * 100, 'k--', alpha=0.4, lw=1.5, label='O(N) reference')

    ax.set_xlabel('Number of Unknowns (N)')
    ax.set_ylabel('Iterations to Converge')
    ax.set_title('Complexity: Multigrid Achieves O(N) Total Work')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')

    # Add annotation
    mg_iters = complexity['multigrid']['iters']
    ax.annotate('Multigrid: constant\niterations = O(N) work',
                xy=(N_values[-1], mg_iters[-1]),
                xytext=(N_values[-2] * 0.5, mg_iters[-1] * 5),
                fontsize=9, color=colors['multigrid'],
                arrowprops=dict(arrowstyle='->', color=colors['multigrid']))

    plt.suptitle(r'Multigrid V-cycle: $-\nabla^2 u = f$ with Dirichlet BC',
                 fontsize=14, y=0.98)
    plt.tight_layout()

    # Save figure
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'multigrid_vcycle.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("Summary: Iterations to reach tolerance 1e-6")
    print("=" * 70)
    print(f"{'Grid':<12} {'N':<10} {'Jacobi':<12} {'Gauss-Seidel':<14} {'Multigrid':<12}")
    print("-" * 70)
    for i, g in enumerate(grid_sizes):
        N = g**2
        print(f"{g}x{g:<8} {N:<10} {complexity['jacobi']['iters'][i]:<12} "
              f"{complexity['gauss_seidel']['iters'][i]:<14} "
              f"{complexity['multigrid']['iters'][i]:<12}")

    print("\n" + "=" * 70)
    print("Key Insights:")
    print("=" * 70)
    print("1. Multigrid achieves O(N) complexity - iterations are nearly constant")
    print("2. Classical methods (Jacobi/GS) scale as O(N) iterations = O(N^2) work")
    print("3. Smoothing eliminates high-frequency errors; coarse grids fix low-frequency")
    print("4. V-cycle recursively applies this idea at all scales")


if __name__ == "__main__":
    main()
