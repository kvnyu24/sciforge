"""
Experiment 25: MCMC - sample from Boltzmann distribution in 1D potential.

Demonstrates Markov Chain Monte Carlo using Metropolis-Hastings
to sample from non-trivial distributions.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def double_well_potential(x, a=1.0, b=4.0):
    """Double-well potential: V(x) = a*x^4 - b*x^2"""
    return a * x**4 - b * x**2


def harmonic_potential(x, k=1.0):
    """Harmonic potential: V(x) = k*x^2/2"""
    return 0.5 * k * x**2


def asymmetric_potential(x):
    """Asymmetric multi-well potential."""
    return x**4 - 3*x**2 + x


def boltzmann_distribution(x, potential, T):
    """Boltzmann distribution: P(x) ∝ exp(-V(x)/(k_B*T))"""
    V = potential(x)
    return np.exp(-V / T)


def metropolis_hastings(potential, T, n_samples, x0=0.0, step_size=0.5, burn_in=1000):
    """
    Metropolis-Hastings MCMC sampling.

    Args:
        potential: Potential energy function V(x)
        T: Temperature (k_B*T)
        n_samples: Number of samples to generate
        x0: Initial position
        step_size: Proposal step size
        burn_in: Number of initial samples to discard

    Returns:
        samples, acceptance_rate
    """
    x = x0
    samples = []
    n_accepted = 0
    total_samples = n_samples + burn_in

    for i in range(total_samples):
        # Propose new position (symmetric random walk)
        x_new = x + step_size * np.random.randn()

        # Acceptance probability
        dV = potential(x_new) - potential(x)
        acceptance = min(1.0, np.exp(-dV / T))

        # Accept or reject
        if np.random.random() < acceptance:
            x = x_new
            if i >= burn_in:
                n_accepted += 1

        if i >= burn_in:
            samples.append(x)

    acceptance_rate = n_accepted / n_samples
    return np.array(samples), acceptance_rate


def autocorrelation(samples, max_lag=100):
    """Compute autocorrelation function."""
    n = len(samples)
    mean = np.mean(samples)
    var = np.var(samples)

    acf = np.zeros(max_lag)
    for lag in range(max_lag):
        acf[lag] = np.mean((samples[lag:] - mean) * (samples[:-lag if lag > 0 else n] - mean)) / var

    return acf


def main():
    np.random.seed(42)

    # Parameters
    n_samples = 50000
    burn_in = 5000

    # Test different temperatures for double well
    temperatures = [0.5, 1.0, 2.0, 5.0]
    step_size = 0.5

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Double-well potential and samples at different T
    ax = axes[0, 0]
    x_range = np.linspace(-2.5, 2.5, 200)
    V = double_well_potential(x_range)
    ax.plot(x_range, V, 'k-', lw=2, label='V(x)')

    ax2 = ax.twinx()
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(temperatures)))

    for T, color in zip(temperatures, colors):
        samples, _ = metropolis_hastings(double_well_potential, T, n_samples,
                                          step_size=step_size, burn_in=burn_in)
        ax2.hist(samples, bins=50, density=True, alpha=0.5, color=color,
                 label=f'T={T}')

        # Theoretical distribution
        P = boltzmann_distribution(x_range, double_well_potential, T)
        P = P / np.trapz(P, x_range)  # Normalize
        ax2.plot(x_range, P, '-', color=color, lw=2)

    ax.set_xlabel('x')
    ax.set_ylabel('V(x)', color='black')
    ax2.set_ylabel('P(x)', color='blue')
    ax.set_title('Double-Well: Samples at Different T')
    ax2.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Plot 2: Trace plot
    ax = axes[0, 1]
    T_trace = 1.0
    samples_trace, acc_rate = metropolis_hastings(double_well_potential, T_trace, 5000,
                                                   step_size=step_size, burn_in=0)
    ax.plot(samples_trace[:2000], 'b-', lw=0.5)
    ax.axvline(burn_in, color='red', linestyle='--', label='Burn-in end')
    ax.set_xlabel('Sample')
    ax.set_ylabel('x')
    ax.set_title(f'Trace Plot (T={T_trace}, acc={acc_rate:.2f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Autocorrelation
    ax = axes[0, 2]
    samples_acf, _ = metropolis_hastings(double_well_potential, T_trace, n_samples,
                                          step_size=step_size, burn_in=burn_in)
    acf = autocorrelation(samples_acf, max_lag=200)

    ax.plot(acf, 'b-', lw=1.5)
    ax.axhline(0, color='gray', linestyle='--')
    ax.axhline(0.1, color='red', linestyle=':', label='0.1 threshold')
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Autocorrelation Function')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Find correlation time
    corr_time = np.where(acf < 0.1)[0]
    if len(corr_time) > 0:
        corr_time = corr_time[0]
        ax.axvline(corr_time, color='green', linestyle='--',
                   label=f'τ_corr ≈ {corr_time}')
        ax.legend()

    # Plot 4: Effect of step size
    ax = axes[1, 0]
    step_sizes = [0.1, 0.5, 1.0, 2.0, 5.0]
    acceptance_rates = []
    effective_samples = []

    T_step = 1.0
    for step in step_sizes:
        samples_s, acc = metropolis_hastings(double_well_potential, T_step, 10000,
                                              step_size=step, burn_in=1000)
        acceptance_rates.append(acc)

        # Estimate effective sample size from autocorrelation
        acf_s = autocorrelation(samples_s, max_lag=100)
        tau = np.sum(acf_s[acf_s > 0])  # Integrated autocorrelation time
        n_eff = len(samples_s) / (2 * tau) if tau > 0 else len(samples_s)
        effective_samples.append(n_eff)

    ax.plot(step_sizes, acceptance_rates, 'bo-', lw=2, markersize=8)
    ax.axhline(0.234, color='red', linestyle='--', label='Optimal (0.234)')
    ax.set_xlabel('Step size')
    ax.set_ylabel('Acceptance rate')
    ax.set_title('Acceptance Rate vs Step Size')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Asymmetric potential
    ax = axes[1, 1]
    x_asym = np.linspace(-2, 2.5, 200)
    V_asym = asymmetric_potential(x_asym)

    samples_asym, _ = metropolis_hastings(asymmetric_potential, T=1.0, n_samples=n_samples,
                                           step_size=0.5, burn_in=burn_in)

    ax.plot(x_asym, V_asym, 'k-', lw=2, label='V(x)')
    ax2 = ax.twinx()
    ax2.hist(samples_asym, bins=50, density=True, alpha=0.7, color='blue')

    P_asym = boltzmann_distribution(x_asym, asymmetric_potential, T=1.0)
    P_asym = P_asym / np.trapz(P_asym, x_asym)
    ax2.plot(x_asym, P_asym, 'r-', lw=2, label='Theory')

    ax.set_xlabel('x')
    ax.set_ylabel('V(x)')
    ax2.set_ylabel('P(x)')
    ax.set_title('Asymmetric Multi-Well (T=1)')
    ax.grid(True, alpha=0.3)

    # Plot 6: Summary
    ax = axes[1, 2]
    ax.axis('off')

    summary = """MCMC: Metropolis-Hastings
=========================
Algorithm:
1. Propose: x' = x + N(0,σ)
2. Accept with prob:
   α = min(1, exp(-ΔV/T))
3. Repeat N times

Key Parameters:
• Step size σ: controls mixing
• Temperature T: controls spread
• Burn-in: discard initial samples

Optimal Acceptance:
• ~23.4% for high dimensions
• ~44% for 1D

Diagnostics:
• Trace plot: visual mixing
• Autocorrelation: independence
• Effective sample size

Boltzmann Distribution:
  P(x) ∝ exp(-V(x)/(k_B·T))

Applications:
• Statistical physics
• Bayesian inference
• Optimization (simulated annealing)
• Protein folding"""

    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    plt.suptitle('Markov Chain Monte Carlo: Sampling from Boltzmann Distribution',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'mcmc_sampling.png'), dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/mcmc_sampling.png")


if __name__ == "__main__":
    main()
