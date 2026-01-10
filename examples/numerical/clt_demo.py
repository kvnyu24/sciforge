"""
Experiment 22: Central Limit Theorem - sum of random vars converges to Gaussian.

Demonstrates the CLT by showing how sums of various distributions
converge to a Gaussian distribution.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def sum_of_uniforms(n_samples, n_sum):
    """Sum of n_sum uniform random variables."""
    samples = np.random.uniform(0, 1, (n_samples, n_sum))
    return np.sum(samples, axis=1)


def sum_of_exponentials(n_samples, n_sum, rate=1.0):
    """Sum of n_sum exponential random variables."""
    samples = np.random.exponential(1/rate, (n_samples, n_sum))
    return np.sum(samples, axis=1)


def sum_of_bernoulli(n_samples, n_sum, p=0.5):
    """Sum of n_sum Bernoulli random variables (binomial)."""
    samples = np.random.binomial(1, p, (n_samples, n_sum))
    return np.sum(samples, axis=1)


def sum_of_poisson(n_samples, n_sum, lam=1.0):
    """Sum of n_sum Poisson random variables."""
    samples = np.random.poisson(lam, (n_samples, n_sum))
    return np.sum(samples, axis=1)


def standardize(data):
    """Standardize to zero mean and unit variance."""
    return (data - np.mean(data)) / np.std(data)


def main():
    np.random.seed(42)

    n_samples = 50000
    n_sums = [1, 2, 5, 10, 30, 100]

    distributions = {
        'Uniform': (sum_of_uniforms, {}),
        'Exponential': (sum_of_exponentials, {'rate': 1.0}),
        'Bernoulli': (sum_of_bernoulli, {'p': 0.3}),
        'Poisson': (sum_of_poisson, {'lam': 2.0})
    }

    fig, axes = plt.subplots(len(distributions), len(n_sums), figsize=(18, 12))

    # Standard normal for comparison
    x_norm = np.linspace(-4, 4, 100)
    pdf_norm = stats.norm.pdf(x_norm)

    for row, (dist_name, (func, kwargs)) in enumerate(distributions.items()):
        for col, n in enumerate(n_sums):
            ax = axes[row, col]

            # Generate sums
            data = func(n_samples, n, **kwargs)
            data_std = standardize(data)

            # Histogram
            ax.hist(data_std, bins=50, density=True, alpha=0.7,
                    color='steelblue', edgecolor='black', linewidth=0.5)

            # Standard normal overlay
            ax.plot(x_norm, pdf_norm, 'r-', lw=2, label='N(0,1)')

            # KS test
            ks_stat, p_value = stats.kstest(data_std, 'norm')

            ax.set_xlim(-4, 4)
            ax.set_ylim(0, 0.6)

            if row == 0:
                ax.set_title(f'n = {n}')
            if col == 0:
                ax.set_ylabel(dist_name)
            if row == len(distributions) - 1:
                ax.set_xlabel('Standardized Value')

            # Add KS statistic
            ax.text(0.95, 0.95, f'KS: {ks_stat:.3f}',
                    transform=ax.transAxes, ha='right', va='top', fontsize=8)

            ax.grid(True, alpha=0.3)

    plt.suptitle('Central Limit Theorem: Sum of n Independent Random Variables\n' +
                 'Standardized to zero mean and unit variance',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'clt_demo.png'), dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/clt_demo.png")


if __name__ == "__main__":
    main()
