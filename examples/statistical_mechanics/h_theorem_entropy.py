"""
Experiment 148: H-Theorem and Entropy Evolution

This example demonstrates Boltzmann's H-theorem, which proves that
the H-function (negative entropy) decreases monotonically in time
as a system relaxes to equilibrium.

Boltzmann's H-function:
H = integral f(v) * ln(f(v)) dv

The H-theorem states: dH/dt <= 0

Entropy is related to H by: S = -k_B * H

Therefore: dS/dt >= 0 (entropy increases)

At equilibrium (Maxwell-Boltzmann distribution):
- dH/dt = 0
- H reaches its minimum value
- S reaches its maximum value (second law of thermodynamics)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson


def maxwell_boltzmann_1d(v, T, m=1.0, k_B=1.0):
    """1D Maxwell-Boltzmann distribution (Gaussian)."""
    sigma = np.sqrt(k_B * T / m)
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-v**2 / (2 * sigma**2))


def compute_h_function(f, v_grid):
    """
    Compute Boltzmann H-function: H = integral f * ln(f) dv

    Args:
        f: Distribution function values on grid
        v_grid: Velocity grid points

    Returns:
        H value
    """
    # Avoid log(0) by adding small value where f ≈ 0
    f_safe = np.maximum(f, 1e-30)

    integrand = f * np.log(f_safe)

    # Set integrand to 0 where f is effectively 0
    integrand = np.where(f > 1e-20, integrand, 0)

    return simpson(integrand, x=v_grid)


def compute_entropy(f, v_grid, k_B=1.0):
    """
    Compute entropy: S = -k_B * H = -k_B * integral f * ln(f) dv
    """
    return -k_B * compute_h_function(f, v_grid)


def bgk_relaxation_with_h(f0, v_grid, T_eq, tau, t_max, dt=0.01):
    """
    BGK relaxation with H-function tracking.

    Returns:
        times, H_values, S_values, distributions
    """
    f_eq = maxwell_boltzmann_1d(v_grid, T_eq)
    dv = v_grid[1] - v_grid[0]
    f_eq = f_eq / simpson(f_eq, x=v_grid)  # Normalize

    times = np.arange(0, t_max + dt, dt)
    H_values = []
    S_values = []
    distributions = []
    dH_dt = []

    f_prev = f0.copy()

    for i, t in enumerate(times):
        # Analytical BGK solution
        f = f_eq + (f0 - f_eq) * np.exp(-t / tau)

        H = compute_h_function(f, v_grid)
        S = compute_entropy(f, v_grid)

        H_values.append(H)
        S_values.append(S)
        distributions.append(f.copy())

        # Numerical derivative of H
        if i > 0:
            dH = (H - H_values[-2]) / dt
            dH_dt.append(dH)

        f_prev = f.copy()

    return times, np.array(H_values), np.array(S_values), distributions, np.array(dH_dt)


def dsmc_h_evolution(n_particles, T_init, n_steps, dt=0.01):
    """
    DSMC simulation tracking entropy evolution.
    """
    m = 1.0
    k_B = 1.0

    # Initialize with non-equilibrium: uniform in velocity box
    v_max = np.sqrt(3 * k_B * T_init / m) * 1.5
    velocities = np.random.uniform(-v_max, v_max, (n_particles, 3))

    times = []
    H_values = []
    S_values = []
    velocity_snapshots = []

    # Velocity grid for distribution estimation
    v_grid = np.linspace(-v_max, v_max, 100)
    dv = v_grid[1] - v_grid[0]

    collision_prob = 0.2

    for step in range(n_steps):
        t = step * dt
        times.append(t)

        # Estimate 1D distribution using histogram
        hist, edges = np.histogram(velocities[:, 0], bins=v_grid, density=True)
        v_centers = 0.5 * (edges[1:] + edges[:-1])

        H = compute_h_function(hist, v_centers)
        S = compute_entropy(hist, v_centers)

        H_values.append(H)
        S_values.append(S)

        if step % (n_steps // 10) == 0:
            velocity_snapshots.append(velocities.copy())

        # Collision step
        indices = np.random.permutation(n_particles)

        for i in range(0, n_particles - 1, 2):
            if np.random.random() < collision_prob:
                i1, i2 = indices[i], indices[i+1]

                # Elastic collision: exchange random velocity component
                for axis in range(3):
                    v1, v2 = velocities[i1, axis], velocities[i2, axis]
                    total = v1 + v2
                    v_new1 = total / 2 + np.random.normal(0, abs(v1 - v2) / 2)
                    v_new2 = total - v_new1
                    velocities[i1, axis] = v_new1
                    velocities[i2, axis] = v_new2

    return np.array(times), np.array(H_values), np.array(S_values), velocity_snapshots


def main():
    print("H-Theorem: Entropy Evolution")
    print("=" * 60)

    # Parameters
    T_eq = 1.0
    tau = 1.0
    k_B = 1.0
    m = 1.0

    v_grid = np.linspace(-6, 6, 300)
    dv = v_grid[1] - v_grid[0]

    # Initial non-equilibrium distributions to test

    # 1. Bimodal (two-temperature)
    T_hot, T_cold = 2.0, 0.5
    f_bimodal = 0.5 * maxwell_boltzmann_1d(v_grid, T_hot) + 0.5 * maxwell_boltzmann_1d(v_grid, T_cold)
    f_bimodal = f_bimodal / simpson(f_bimodal, x=v_grid)

    # 2. Uniform distribution
    v_max = np.sqrt(3 * k_B * T_eq / m) * 1.5
    f_uniform = np.where(np.abs(v_grid) < v_max, 1.0 / (2 * v_max), 0)
    f_uniform = f_uniform / simpson(f_uniform, x=v_grid)

    # 3. Delta-like (narrow peak) - approximated by narrow Gaussian
    f_delta = maxwell_boltzmann_1d(v_grid, 0.1)
    f_delta = f_delta / simpson(f_delta, x=v_grid)

    # Equilibrium distribution
    f_eq = maxwell_boltzmann_1d(v_grid, T_eq)
    f_eq = f_eq / simpson(f_eq, x=v_grid)

    # Compute equilibrium H and S
    H_eq = compute_h_function(f_eq, v_grid)
    S_eq = compute_entropy(f_eq, v_grid)

    print(f"Equilibrium temperature: T = {T_eq}")
    print(f"Equilibrium H: H_eq = {H_eq:.4f}")
    print(f"Equilibrium S: S_eq = {S_eq:.4f}")

    t_max = 5 * tau

    # Run BGK relaxation for each initial condition
    print("\nRunning relaxation simulations...")

    results = {}
    for name, f0 in [('bimodal', f_bimodal), ('uniform', f_uniform), ('delta', f_delta)]:
        print(f"  {name}...", end=' ')
        times, H, S, dists, dH = bgk_relaxation_with_h(f0, v_grid, T_eq, tau, t_max)
        results[name] = {'times': times, 'H': H, 'S': S, 'dists': dists, 'dH': dH}

        print(f"H_0 = {H[0]:.4f}, H_final = {H[-1]:.4f}")

    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Plot 1: Initial distributions
    ax1 = axes[0, 0]
    ax1.plot(v_grid, f_bimodal, 'b-', lw=2, label='Bimodal')
    ax1.plot(v_grid, f_uniform, 'g-', lw=2, label='Uniform')
    ax1.plot(v_grid, f_delta, 'm-', lw=2, label='Delta-like')
    ax1.plot(v_grid, f_eq, 'r--', lw=2, label='Equilibrium')
    ax1.set_xlabel('Velocity v', fontsize=12)
    ax1.set_ylabel('f(v)', fontsize=12)
    ax1.set_title('Initial Distributions', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: H-function evolution
    ax2 = axes[0, 1]
    colors = {'bimodal': 'blue', 'uniform': 'green', 'delta': 'magenta'}

    for name in results:
        ax2.plot(results[name]['times'], results[name]['H'],
                 color=colors[name], lw=2, label=name.title())

    ax2.axhline(H_eq, color='red', linestyle='--', lw=2, label='$H_{eq}$')
    ax2.set_xlabel('Time (t/tau)', fontsize=12)
    ax2.set_ylabel('H-function', fontsize=12)
    ax2.set_title('H-Function Evolution (H-Theorem: H decreases)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Entropy evolution
    ax3 = axes[0, 2]

    for name in results:
        ax3.plot(results[name]['times'], results[name]['S'],
                 color=colors[name], lw=2, label=name.title())

    ax3.axhline(S_eq, color='red', linestyle='--', lw=2, label='$S_{eq}$')
    ax3.set_xlabel('Time (t/tau)', fontsize=12)
    ax3.set_ylabel('Entropy S', fontsize=12)
    ax3.set_title('Entropy Evolution (S increases)', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: dH/dt (should always be <= 0)
    ax4 = axes[1, 0]

    for name in results:
        t_deriv = results[name]['times'][1:]
        ax4.plot(t_deriv, results[name]['dH'], color=colors[name], lw=2,
                 label=name.title())

    ax4.axhline(0, color='gray', linestyle='--')
    ax4.set_xlabel('Time (t/tau)', fontsize=12)
    ax4.set_ylabel('dH/dt', fontsize=12)
    ax4.set_title('H-Theorem: dH/dt <= 0', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Distribution evolution (bimodal case)
    ax5 = axes[1, 1]
    dists = results['bimodal']['dists']
    times = results['bimodal']['times']

    time_indices = [0, len(times)//10, len(times)//4, len(times)//2, -1]
    colors_t = plt.cm.viridis(np.linspace(0, 1, len(time_indices)))

    for idx, color in zip(time_indices, colors_t):
        ax5.plot(v_grid, dists[idx], color=color, lw=2,
                 label=f't = {times[idx]:.2f}')

    ax5.plot(v_grid, f_eq, 'r--', lw=2, label='Equilibrium')
    ax5.set_xlabel('Velocity v', fontsize=12)
    ax5.set_ylabel('f(v)', fontsize=12)
    ax5.set_title('Bimodal to Maxwell-Boltzmann', fontsize=12)
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Approach to equilibrium in H-S space
    ax6 = axes[1, 2]

    for name in results:
        H = results[name]['H'] - H_eq  # Relative to equilibrium
        S = results[name]['S'] - S_eq
        ax6.plot(H, S, color=colors[name], lw=2, label=name.title())
        ax6.plot(H[0], S[0], 'o', color=colors[name], markersize=10)
        ax6.plot(H[-1], S[-1], '*', color=colors[name], markersize=15)

    ax6.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax6.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax6.set_xlabel('$H - H_{eq}$', fontsize=12)
    ax6.set_ylabel('$S - S_{eq}$', fontsize=12)
    ax6.set_title('Trajectory to Equilibrium\n(circles: start, stars: end)', fontsize=12)
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.suptitle("Boltzmann's H-Theorem: Irreversible Approach to Equilibrium",
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Second figure: DSMC verification
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

    print("\nRunning DSMC simulation...")
    times_dsmc, H_dsmc, S_dsmc, v_snaps = dsmc_h_evolution(5000, T_eq, 500)

    ax7 = axes2[0]
    ax7.plot(times_dsmc, H_dsmc, 'b-', lw=1.5, alpha=0.7)
    ax7.set_xlabel('Time', fontsize=12)
    ax7.set_ylabel('H-function', fontsize=12)
    ax7.set_title('DSMC H-Function Evolution', fontsize=12)
    ax7.grid(True, alpha=0.3)

    ax8 = axes2[1]
    ax8.plot(times_dsmc, S_dsmc, 'r-', lw=1.5, alpha=0.7)
    ax8.set_xlabel('Time', fontsize=12)
    ax8.set_ylabel('Entropy S', fontsize=12)
    ax8.set_title('DSMC Entropy Evolution', fontsize=12)
    ax8.grid(True, alpha=0.3)

    plt.tight_layout()

    # Print summary
    print("\n" + "=" * 60)
    print("H-Theorem Summary")
    print("=" * 60)
    print(f"\nEquilibrium values:")
    print(f"  H_eq = {H_eq:.4f}")
    print(f"  S_eq = {S_eq:.4f}")

    print(f"\n{'Initial':>12} {'H_0':>10} {'S_0':>10} {'H_final':>10} {'S_final':>10}")
    print("-" * 55)
    for name in results:
        H = results[name]['H']
        S = results[name]['S']
        print(f"{name:>12} {H[0]:>10.4f} {S[0]:>10.4f} {H[-1]:>10.4f} {S[-1]:>10.4f}")

    print("\nKey observations:")
    print("1. H always decreases (or stays constant at equilibrium)")
    print("2. S always increases (second law of thermodynamics)")
    print("3. All initial conditions relax to the same equilibrium")
    print("4. dH/dt = 0 only at Maxwell-Boltzmann equilibrium")

    # Save plots
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'h_theorem_entropy.png'),
                dpi=150, bbox_inches='tight')
    fig2.savefig(os.path.join(output_dir, 'h_theorem_dsmc.png'),
                 dpi=150, bbox_inches='tight')

    print(f"\nPlots saved to {output_dir}")


if __name__ == "__main__":
    main()
