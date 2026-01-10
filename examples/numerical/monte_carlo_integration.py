"""
Experiment 24: Monte Carlo integration - estimate π and volume of sphere in high dimension.

Demonstrates Monte Carlo integration techniques and curse of dimensionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy import special


def estimate_pi(n_samples):
    """Estimate π using Monte Carlo (area of quarter circle)."""
    np.random.seed(None)
    x = np.random.uniform(0, 1, n_samples)
    y = np.random.uniform(0, 1, n_samples)

    inside = (x**2 + y**2) <= 1
    pi_estimate = 4 * np.sum(inside) / n_samples

    return pi_estimate, x, y, inside


def sphere_volume_mc(n_samples, dimension):
    """
    Estimate volume of unit sphere in d dimensions using Monte Carlo.

    Volume of d-dimensional unit sphere: V_d = π^(d/2) / Γ(d/2 + 1)
    """
    # Generate random points in [-1, 1]^d
    points = np.random.uniform(-1, 1, (n_samples, dimension))

    # Check if inside unit sphere
    r2 = np.sum(points**2, axis=1)
    inside = r2 <= 1

    # Volume estimate = fraction inside × volume of cube
    cube_volume = 2**dimension
    volume_estimate = np.sum(inside) / n_samples * cube_volume

    return volume_estimate


def sphere_volume_exact(dimension):
    """Exact volume of unit sphere in d dimensions."""
    return np.pi**(dimension/2) / special.gamma(dimension/2 + 1)


def convergence_analysis(max_samples, n_trials=10):
    """Analyze convergence of π estimate with sample size."""
    sample_sizes = np.logspace(2, int(np.log10(max_samples)), 20, dtype=int)
    errors = np.zeros((len(sample_sizes), n_trials))

    for i, n in enumerate(sample_sizes):
        for j in range(n_trials):
            pi_est, _, _, _ = estimate_pi(n)
            errors[i, j] = abs(pi_est - np.pi) / np.pi

    return sample_sizes, errors


def main():
    np.random.seed(42)

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Visual demonstration of π estimation
    ax = axes[0, 0]
    n = 5000
    pi_est, x, y, inside = estimate_pi(n)

    ax.scatter(x[inside], y[inside], c='blue', s=1, alpha=0.5, label='Inside')
    ax.scatter(x[~inside], y[~inside], c='red', s=1, alpha=0.5, label='Outside')

    theta = np.linspace(0, np.pi/2, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', lw=2)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'π ≈ {pi_est:.4f} (n={n})\nTrue: π = {np.pi:.4f}')
    ax.set_aspect('equal')
    ax.legend(fontsize=8)

    # Plot 2: Convergence with sample size
    ax = axes[0, 1]
    sample_sizes, errors = convergence_analysis(1000000, n_trials=5)

    # Plot individual trials
    for j in range(errors.shape[1]):
        ax.loglog(sample_sizes, errors[:, j], 'b-', alpha=0.3, lw=0.5)

    # Mean and expected
    ax.loglog(sample_sizes, np.mean(errors, axis=1), 'b-', lw=2, label='Mean error')
    ax.loglog(sample_sizes, 1/np.sqrt(sample_sizes), 'r--', lw=2, label='1/√N')

    ax.set_xlabel('Number of samples')
    ax.set_ylabel('Relative error')
    ax.set_title('Convergence Rate: O(1/√N)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Sphere volume vs dimension
    ax = axes[0, 2]
    dimensions = np.arange(1, 21)
    n_samples_sphere = 100000

    volume_exact = [sphere_volume_exact(d) for d in dimensions]
    volume_mc = [sphere_volume_mc(n_samples_sphere, d) for d in dimensions]

    ax.semilogy(dimensions, volume_exact, 'ko-', lw=2, markersize=6, label='Exact')
    ax.semilogy(dimensions, volume_mc, 'rs--', lw=1.5, markersize=6, label='Monte Carlo')

    ax.set_xlabel('Dimension')
    ax.set_ylabel('Volume')
    ax.set_title(f'Unit Sphere Volume (n={n_samples_sphere})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Fraction inside sphere vs dimension
    ax = axes[1, 0]

    fractions = [sphere_volume_mc(n_samples_sphere, d) / (2**d) for d in dimensions]
    fractions_exact = [sphere_volume_exact(d) / (2**d) for d in dimensions]

    ax.semilogy(dimensions, fractions_exact, 'k-', lw=2, label='Exact')
    ax.semilogy(dimensions, fractions, 'rs', markersize=6, label='MC estimate')

    ax.set_xlabel('Dimension')
    ax.set_ylabel('Fraction in sphere')
    ax.set_title('Curse of Dimensionality:\nSphere shrinks relative to cube')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Required samples for fixed accuracy
    ax = axes[1, 1]

    target_accuracy = 0.01  # 1% error
    dimensions_test = [2, 5, 10, 15]
    n_samples_test = np.logspace(3, 6, 20, dtype=int)

    for d in dimensions_test:
        errors_d = []
        V_exact = sphere_volume_exact(d)
        for n in n_samples_test:
            V_mc = np.mean([sphere_volume_mc(n, d) for _ in range(5)])
            errors_d.append(abs(V_mc - V_exact) / V_exact)
        ax.loglog(n_samples_test, errors_d, 'o-', markersize=3, label=f'd={d}')

    ax.axhline(target_accuracy, color='gray', linestyle='--', label='1% target')
    ax.set_xlabel('Number of samples')
    ax.set_ylabel('Relative error')
    ax.set_title('Error vs Samples by Dimension')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 6: Summary
    ax = axes[1, 2]
    ax.axis('off')

    summary = f"""Monte Carlo Integration
=======================
Basic Idea:
  ∫f(x)dx ≈ V·<f(x)>_random
  where V is domain volume

π Estimation:
  π = 4 × (area of quarter circle)
  π = 4 × (points inside / total)

Convergence:
  Error ~ 1/√N (CLT)
  Independent of dimension!

Curse of Dimensionality:
  Volume ratio: V_sphere/V_cube
  → 0 exponentially as d → ∞

  Dim  V_sphere  V_cube  Ratio
  2    π         4       0.785
  5    5.26      32      0.164
  10   2.55      1024    0.0025
  20   0.026     ~10⁶    ~10⁻⁸

Practical Limits:
  Need many more samples in
  high dimensions to get
  non-zero "hits" inside sphere.

Key Applications:
• High-dimensional integrals
• Bayesian inference
• Statistical physics
• Option pricing"""

    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Monte Carlo Integration: From π to High Dimensions',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'monte_carlo_integration.png'), dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/monte_carlo_integration.png")


if __name__ == "__main__":
    main()
