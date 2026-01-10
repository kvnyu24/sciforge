"""
Example demonstrating Geometric Brownian Motion (GBM).

This example shows stock price modeling using GBM, the foundation
of the Black-Scholes option pricing model.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.stochastic.processes import GeometricBrownianMotion


def simulate_gbm(S0, mu, sigma, T, n_steps, n_paths):
    """
    Simulate Geometric Brownian Motion paths.

    Args:
        S0: Initial price
        mu: Drift (expected return)
        sigma: Volatility
        T: Time horizon
        n_steps: Number of time steps
        n_paths: Number of simulation paths

    Returns:
        Time array and price paths
    """
    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)

    # Generate paths
    S = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0

    for i in range(n_steps):
        Z = np.random.standard_normal(n_paths)
        S[:, i+1] = S[:, i] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

    return t, S


def main():
    # Parameters
    S0 = 100        # Initial stock price
    mu = 0.10       # Expected annual return (10%)
    sigma = 0.20    # Annual volatility (20%)
    T = 1.0         # Time horizon (1 year)
    n_steps = 252   # Daily steps (trading days)
    n_paths = 1000  # Number of simulations

    # Run simulation
    t, S = simulate_gbm(S0, mu, sigma, T, n_steps, n_paths)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Sample paths
    ax1 = axes[0, 0]
    for i in range(min(50, n_paths)):
        ax1.plot(t, S[i], lw=0.5, alpha=0.5)
    ax1.axhline(y=S0, color='black', linestyle='--', lw=1, label=f'S₀ = {S0}')

    # Expected value line
    E_S = S0 * np.exp(mu * t)
    ax1.plot(t, E_S, 'r-', lw=2, label=f'E[S(t)] = S₀e^(μt)')

    ax1.set_xlabel('Time (years)')
    ax1.set_ylabel('Stock Price')
    ax1.set_title('Geometric Brownian Motion Paths')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Log price paths
    ax2 = axes[0, 1]
    for i in range(min(50, n_paths)):
        ax2.plot(t, np.log(S[i]), lw=0.5, alpha=0.5)

    # Expected log price
    E_log_S = np.log(S0) + (mu - 0.5 * sigma**2) * t
    ax2.plot(t, E_log_S, 'r-', lw=2, label=f'E[ln S] = ln S₀ + (μ-σ²/2)t')

    ax2.set_xlabel('Time (years)')
    ax2.set_ylabel('Log Price')
    ax2.set_title('Log Price Follows Brownian Motion with Drift')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Final price distribution
    ax3 = axes[0, 2]
    S_final = S[:, -1]

    ax3.hist(S_final, bins=50, density=True, alpha=0.7, edgecolor='black')

    # Theoretical lognormal distribution
    from scipy.stats import lognorm
    s_theory = np.linspace(S_final.min(), S_final.max(), 100)
    sigma_total = sigma * np.sqrt(T)
    mu_total = np.log(S0) + (mu - 0.5 * sigma**2) * T
    pdf = lognorm.pdf(s_theory, s=sigma_total, scale=np.exp(mu_total))
    ax3.plot(s_theory, pdf, 'r-', lw=2, label='Lognormal PDF')

    ax3.axvline(x=np.mean(S_final), color='blue', linestyle='--', label=f'Mean = {np.mean(S_final):.1f}')
    ax3.axvline(x=np.median(S_final), color='green', linestyle=':', label=f'Median = {np.median(S_final):.1f}')

    ax3.set_xlabel('Final Stock Price')
    ax3.set_ylabel('Probability Density')
    ax3.set_title('Final Price Distribution (Lognormal)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Returns distribution
    ax4 = axes[1, 0]
    daily_returns = np.diff(np.log(S), axis=1)  # Log returns

    ax4.hist(daily_returns.flatten(), bins=100, density=True, alpha=0.7, edgecolor='black')

    # Theoretical normal distribution
    from scipy.stats import norm
    returns_range = np.linspace(daily_returns.min(), daily_returns.max(), 100)
    dt = T / n_steps
    pdf_returns = norm.pdf(returns_range, loc=(mu - 0.5*sigma**2)*dt, scale=sigma*np.sqrt(dt))
    ax4.plot(returns_range, pdf_returns, 'r-', lw=2, label='Normal PDF')

    ax4.set_xlabel('Daily Log Return')
    ax4.set_ylabel('Probability Density')
    ax4.set_title('Log Returns Distribution (Normal)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Quantile paths
    ax5 = axes[1, 1]

    percentiles = [5, 25, 50, 75, 95]
    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(percentiles)))

    for p, color in zip(percentiles, colors):
        quantile_path = np.percentile(S, p, axis=0)
        ax5.plot(t, quantile_path, color=color, lw=2, label=f'{p}th percentile')

    ax5.fill_between(t, np.percentile(S, 5, axis=0), np.percentile(S, 95, axis=0),
                     alpha=0.2, color='blue', label='90% CI')
    ax5.axhline(y=S0, color='black', linestyle='--', lw=1)

    ax5.set_xlabel('Time (years)')
    ax5.set_ylabel('Stock Price')
    ax5.set_title('Price Distribution Over Time')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # Plot 6: Probability of different outcomes
    ax6 = axes[1, 2]

    thresholds = np.linspace(50, 200, 50)
    prob_above = [np.mean(S_final > thresh) for thresh in thresholds]
    prob_below = [np.mean(S_final < thresh) for thresh in thresholds]

    ax6.plot(thresholds, prob_above, 'g-', lw=2, label='P(S > threshold)')
    ax6.plot(thresholds, prob_below, 'r-', lw=2, label='P(S < threshold)')
    ax6.axvline(x=S0, color='black', linestyle='--', label=f'S₀ = {S0}')
    ax6.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)

    ax6.set_xlabel('Threshold Price')
    ax6.set_ylabel('Probability')
    ax6.set_title('Probability of Price Levels at T=1')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.suptitle(f'Geometric Brownian Motion: dS = μS dt + σS dW\n'
                 f'S₀ = {S0}, μ = {mu:.0%}, σ = {sigma:.0%}, T = {T} year',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'geometric_brownian.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'geometric_brownian.png')}")


if __name__ == "__main__":
    main()
