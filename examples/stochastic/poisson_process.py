"""
Example demonstrating the Poisson process.

This example shows random arrivals following a Poisson process,
with applications to radioactive decay and queuing theory.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, expon
from src.sciforge.stochastic.processes import PoissonProcess


def simulate_poisson_process(rate, T, n_realizations=1):
    """
    Simulate Poisson process arrivals.

    Args:
        rate: Average arrival rate (λ)
        T: Time horizon
        n_realizations: Number of independent processes

    Returns:
        List of arrival times for each realization
    """
    arrivals_list = []

    for _ in range(n_realizations):
        arrivals = []
        t = 0
        while t < T:
            # Inter-arrival times are exponentially distributed
            dt = np.random.exponential(1/rate)
            t += dt
            if t < T:
                arrivals.append(t)
        arrivals_list.append(np.array(arrivals))

    return arrivals_list


def main():
    # Parameters
    rate = 2.0    # Average arrivals per unit time (λ)
    T = 20.0      # Time horizon
    n_paths = 100  # Number of realizations

    # Simulate
    arrivals_list = simulate_poisson_process(rate, T, n_paths)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Sample paths (counting process)
    ax1 = axes[0, 0]

    for i in range(min(10, n_paths)):
        arrivals = arrivals_list[i]
        times = np.concatenate([[0], arrivals, [T]])
        counts = np.arange(len(times))
        counts[-1] = counts[-2]  # Last point continues horizontally

        ax1.step(times, counts, where='post', lw=1, alpha=0.7)

    # Expected value
    t_line = np.linspace(0, T, 100)
    ax1.plot(t_line, rate * t_line, 'r--', lw=2, label=f'E[N(t)] = λt = {rate}t')

    ax1.set_xlabel('Time')
    ax1.set_ylabel('Number of Arrivals N(t)')
    ax1.set_title('Poisson Counting Process')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Inter-arrival time distribution
    ax2 = axes[0, 1]

    all_inter_arrivals = []
    for arrivals in arrivals_list:
        if len(arrivals) > 1:
            inter_arrivals = np.diff(arrivals)
            all_inter_arrivals.extend(inter_arrivals)

    all_inter_arrivals = np.array(all_inter_arrivals)

    ax2.hist(all_inter_arrivals, bins=30, density=True, alpha=0.7, edgecolor='black')

    # Theoretical exponential distribution
    x_exp = np.linspace(0, all_inter_arrivals.max(), 100)
    pdf_exp = expon.pdf(x_exp, scale=1/rate)
    ax2.plot(x_exp, pdf_exp, 'r-', lw=2, label=f'Exp(λ={rate})')

    ax2.set_xlabel('Inter-arrival Time')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('Inter-arrival Time Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Count distribution at fixed time
    ax3 = axes[0, 2]

    t_fixed = 10.0
    counts_at_t = [sum(arr < t_fixed) for arr in arrivals_list]

    ax3.hist(counts_at_t, bins=range(int(max(counts_at_t)) + 2), density=True,
             alpha=0.7, edgecolor='black', align='left')

    # Theoretical Poisson distribution
    k_range = np.arange(0, int(max(counts_at_t)) + 1)
    pmf_poisson = poisson.pmf(k_range, rate * t_fixed)
    ax3.bar(k_range + 0.3, pmf_poisson, width=0.3, color='red', alpha=0.7, label=f'Poisson(λt={rate*t_fixed})')

    ax3.set_xlabel('Count N(t)')
    ax3.set_ylabel('Probability')
    ax3.set_title(f'Count Distribution at t = {t_fixed}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Variance equals mean
    ax4 = axes[1, 0]

    time_points = np.linspace(1, T, 20)
    means = []
    variances = []

    for t_point in time_points:
        counts = [sum(arr < t_point) for arr in arrivals_list]
        means.append(np.mean(counts))
        variances.append(np.var(counts))

    ax4.plot(means, variances, 'bo', markersize=8, label='Sample')
    ax4.plot([0, max(means)], [0, max(means)], 'r--', lw=2, label='Var = Mean')

    ax4.set_xlabel('Sample Mean')
    ax4.set_ylabel('Sample Variance')
    ax4.set_title('Poisson Property: Var[N(t)] = E[N(t)] = λt')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal')

    # Plot 5: Superposition of independent processes
    ax5 = axes[1, 1]

    # Two independent Poisson processes
    rate1, rate2 = 1.5, 0.5

    arrivals1 = simulate_poisson_process(rate1, T, 1)[0]
    arrivals2 = simulate_poisson_process(rate2, T, 1)[0]
    arrivals_merged = np.sort(np.concatenate([arrivals1, arrivals2]))

    # Plot all three
    for arr, label, color in [(arrivals1, f'Process 1 (λ={rate1})', 'blue'),
                               (arrivals2, f'Process 2 (λ={rate2})', 'green'),
                               (arrivals_merged, f'Merged (λ={rate1+rate2})', 'red')]:
        times = np.concatenate([[0], arr, [T]])
        counts = np.arange(len(times))
        counts[-1] = counts[-2]
        ax5.step(times, counts, where='post', lw=2, label=label, alpha=0.8)

    ax5.set_xlabel('Time')
    ax5.set_ylabel('Count')
    ax5.set_title('Superposition: Merged Process is Poisson')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # Plot 6: Thinning (splitting)
    ax6 = axes[1, 2]

    # Original process
    arrivals_original = simulate_poisson_process(rate, T, 1)[0]

    # Thin with probability p
    p = 0.3
    kept = np.random.random(len(arrivals_original)) < p
    arrivals_thinned = arrivals_original[kept]
    arrivals_removed = arrivals_original[~kept]

    ax6.eventplot([arrivals_original], lineoffsets=3, linelengths=0.5, colors='blue',
                  label=f'Original (λ={rate})')
    ax6.eventplot([arrivals_thinned], lineoffsets=2, linelengths=0.5, colors='green',
                  label=f'Kept (λ={rate*p:.1f})')
    ax6.eventplot([arrivals_removed], lineoffsets=1, linelengths=0.5, colors='red',
                  label=f'Removed (λ={rate*(1-p):.1f})')

    ax6.set_xlabel('Time')
    ax6.set_yticks([1, 2, 3])
    ax6.set_yticklabels(['Removed', 'Kept', 'Original'])
    ax6.set_title(f'Thinning with p = {p}')
    ax6.legend(fontsize=8, loc='upper right')
    ax6.grid(True, alpha=0.3)

    plt.suptitle(f'Poisson Process (λ = {rate} arrivals/unit time)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'poisson_process.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'poisson_process.png')}")


if __name__ == "__main__":
    main()
