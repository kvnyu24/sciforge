"""
Example demonstrating the Ornstein-Uhlenbeck process.

This example shows mean-reverting stochastic behavior, used in
interest rate models, population dynamics, and physics.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.stochastic.processes import OrnsteinUhlenbeck


def simulate_ou_process(X0, theta, mu, sigma, T, n_steps, n_paths):
    """
    Simulate Ornstein-Uhlenbeck process paths.

    dX = theta(mu - X)dt + sigma*dW

    Args:
        X0: Initial value
        theta: Mean reversion speed
        mu: Long-term mean
        sigma: Volatility
        T: Time horizon
        n_steps: Number of time steps
        n_paths: Number of simulation paths

    Returns:
        Time array and process paths
    """
    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)

    X = np.zeros((n_paths, n_steps + 1))
    X[:, 0] = X0

    for i in range(n_steps):
        dW = np.random.standard_normal(n_paths) * np.sqrt(dt)
        X[:, i+1] = X[:, i] + theta * (mu - X[:, i]) * dt + sigma * dW

    return t, X


def main():
    # Parameters
    X0 = 0.0        # Initial value
    theta = 2.0     # Mean reversion speed
    mu = 1.0        # Long-term mean
    sigma = 0.5     # Volatility
    T = 5.0         # Time horizon
    n_steps = 500   # Number of steps
    n_paths = 500   # Number of paths

    # Run simulation
    t, X = simulate_ou_process(X0, theta, mu, sigma, T, n_steps, n_paths)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Sample paths
    ax1 = axes[0, 0]
    for i in range(min(50, n_paths)):
        ax1.plot(t, X[i], lw=0.5, alpha=0.5)

    # Expected value (deterministic solution)
    E_X = mu + (X0 - mu) * np.exp(-theta * t)
    ax1.plot(t, E_X, 'r-', lw=2, label='E[X(t)]')
    ax1.axhline(y=mu, color='black', linestyle='--', lw=1, label=f'μ = {mu}')

    ax1.set_xlabel('Time')
    ax1.set_ylabel('X(t)')
    ax1.set_title('Ornstein-Uhlenbeck Process Paths')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Mean and variance over time
    ax2 = axes[0, 1]

    sample_mean = np.mean(X, axis=0)
    sample_std = np.std(X, axis=0)

    # Theoretical variance
    var_theory = (sigma**2 / (2 * theta)) * (1 - np.exp(-2 * theta * t))
    std_theory = np.sqrt(var_theory)

    ax2.plot(t, sample_mean, 'b-', lw=2, label='Sample mean')
    ax2.plot(t, E_X, 'r--', lw=2, label='Theoretical mean')
    ax2.fill_between(t, sample_mean - sample_std, sample_mean + sample_std,
                     alpha=0.3, color='blue', label='±1 std (sample)')
    ax2.plot(t, E_X + std_theory, 'g:', lw=1.5)
    ax2.plot(t, E_X - std_theory, 'g:', lw=1.5, label='±1 std (theory)')

    ax2.axhline(y=mu, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('X(t)')
    ax2.set_title('Mean and Standard Deviation')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Stationary distribution
    ax3 = axes[0, 2]

    # Use late-time values (should be close to stationary)
    X_stationary = X[:, -100:].flatten()

    ax3.hist(X_stationary, bins=50, density=True, alpha=0.7, edgecolor='black')

    # Theoretical stationary distribution (normal)
    from scipy.stats import norm
    x_range = np.linspace(X_stationary.min(), X_stationary.max(), 100)
    var_stationary = sigma**2 / (2 * theta)
    pdf = norm.pdf(x_range, loc=mu, scale=np.sqrt(var_stationary))
    ax3.plot(x_range, pdf, 'r-', lw=2, label=f'N(μ={mu}, σ²={var_stationary:.3f})')

    ax3.set_xlabel('X')
    ax3.set_ylabel('Probability Density')
    ax3.set_title('Stationary Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Different mean reversion speeds
    ax4 = axes[1, 0]

    theta_values = [0.5, 1.0, 2.0, 5.0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(theta_values)))

    for theta_test, color in zip(theta_values, colors):
        t_test, X_test = simulate_ou_process(0, theta_test, mu, sigma, T, n_steps, 1)
        ax4.plot(t_test, X_test[0], color=color, lw=1.5, label=f'θ = {theta_test}')

    ax4.axhline(y=mu, color='black', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Time')
    ax4.set_ylabel('X(t)')
    ax4.set_title('Effect of Mean Reversion Speed θ')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Autocorrelation function
    ax5 = axes[1, 1]

    # Calculate sample autocorrelation
    max_lag = 100
    lags = np.arange(max_lag)
    dt = T / n_steps

    # Use one long path for autocorrelation
    t_long, X_long = simulate_ou_process(mu, theta, mu, sigma, T*10, n_steps*10, 1)
    X_centered = X_long[0] - np.mean(X_long[0])

    autocorr = np.correlate(X_centered, X_centered, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr[:max_lag] / autocorr[0]

    # Theoretical autocorrelation
    tau = lags * dt
    autocorr_theory = np.exp(-theta * tau)

    ax5.plot(tau, autocorr, 'b-', lw=2, label='Sample')
    ax5.plot(tau, autocorr_theory, 'r--', lw=2, label=r'Theory: $e^{-\theta\tau}$')
    ax5.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    ax5.set_xlabel('Lag τ')
    ax5.set_ylabel('Autocorrelation')
    ax5.set_title('Autocorrelation Function')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Half-life of mean reversion
    ax6 = axes[1, 2]

    theta_range = np.linspace(0.1, 5.0, 100)
    half_life = np.log(2) / theta_range

    ax6.plot(theta_range, half_life, 'b-', lw=2)
    ax6.axhline(y=np.log(2)/theta, color='r', linestyle='--',
               label=f'Current: τ₁/₂ = {np.log(2)/theta:.2f}')
    ax6.plot(theta, np.log(2)/theta, 'ro', markersize=10)

    ax6.set_xlabel('Mean Reversion Speed θ')
    ax6.set_ylabel('Half-life τ₁/₂')
    ax6.set_title('Half-life = ln(2)/θ')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.suptitle(f'Ornstein-Uhlenbeck Process: dX = θ(μ - X)dt + σdW\n'
                 f'θ = {theta}, μ = {mu}, σ = {sigma}, X₀ = {X0}',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'ornstein_uhlenbeck.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'ornstein_uhlenbeck.png')}")


if __name__ == "__main__":
    main()
