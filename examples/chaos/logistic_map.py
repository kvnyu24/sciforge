"""
Example demonstrating the logistic map and period doubling route to chaos.

This example shows the famous logistic map x_{n+1} = r*x_n*(1-x_n),
demonstrating bifurcations and the onset of chaos.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def logistic_map(x, r):
    """Logistic map: x_{n+1} = r*x_n*(1-x_n)."""
    return r * x * (1 - x)


def iterate_logistic(x0, r, n_iterations, n_transient=500):
    """
    Iterate the logistic map.

    Args:
        x0: Initial condition
        r: Control parameter
        n_iterations: Number of iterations to return
        n_transient: Number of transient iterations to discard

    Returns:
        Array of x values after transient
    """
    x = x0
    # Discard transients
    for _ in range(n_transient):
        x = logistic_map(x, r)

    # Collect iterations
    result = np.zeros(n_iterations)
    for i in range(n_iterations):
        result[i] = x
        x = logistic_map(x, r)

    return result


def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Time series for different r values
    ax1 = axes[0, 0]

    r_values = [2.8, 3.2, 3.5, 3.8]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(r_values)))

    for r, color in zip(r_values, colors):
        x = iterate_logistic(0.5, r, 50, n_transient=0)
        ax1.plot(range(50), x, 'o-', color=color, lw=1, markersize=3, label=f'r = {r}')

    ax1.set_xlabel('Iteration n')
    ax1.set_ylabel('x_n')
    ax1.set_title('Time Series for Different r Values')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Bifurcation diagram
    ax2 = axes[0, 1]

    r_range = np.linspace(2.5, 4.0, 1000)
    n_iterations = 200

    for r in r_range:
        x_vals = iterate_logistic(0.5, r, n_iterations)
        ax2.plot([r] * n_iterations, x_vals, 'k,', markersize=0.1)

    ax2.set_xlabel('r')
    ax2.set_ylabel('x (attractor)')
    ax2.set_title('Bifurcation Diagram')
    ax2.set_xlim(2.5, 4.0)

    # Mark Feigenbaum points
    ax2.axvline(x=3.0, color='blue', linestyle=':', alpha=0.5)
    ax2.axvline(x=3.449, color='green', linestyle=':', alpha=0.5)
    ax2.axvline(x=3.5699, color='red', linestyle=':', alpha=0.5)

    # Plot 3: Zoomed bifurcation (self-similarity)
    ax3 = axes[0, 2]

    r_range_zoom = np.linspace(3.84, 3.856, 500)

    for r in r_range_zoom:
        x_vals = iterate_logistic(0.5, r, n_iterations)
        ax3.plot([r] * n_iterations, x_vals, 'k,', markersize=0.2)

    ax3.set_xlabel('r')
    ax3.set_ylabel('x')
    ax3.set_title('Zoomed Bifurcation (Self-Similarity)')

    # Plot 4: Cobweb diagram
    ax4 = axes[1, 0]

    r_cobweb = 3.5
    x0 = 0.2
    n_cobweb = 50

    # Plot the map and diagonal
    x_plot = np.linspace(0, 1, 200)
    ax4.plot(x_plot, logistic_map(x_plot, r_cobweb), 'b-', lw=2, label=f'f(x) = {r_cobweb}x(1-x)')
    ax4.plot(x_plot, x_plot, 'k--', lw=1, label='y = x')

    # Cobweb
    x = x0
    for _ in range(n_cobweb):
        x_new = logistic_map(x, r_cobweb)
        ax4.plot([x, x], [x, x_new], 'r-', lw=0.5, alpha=0.7)
        ax4.plot([x, x_new], [x_new, x_new], 'r-', lw=0.5, alpha=0.7)
        x = x_new

    ax4.plot(x0, 0, 'go', markersize=8, label='Start')
    ax4.set_xlabel('x_n')
    ax4.set_ylabel('x_{n+1}')
    ax4.set_title(f'Cobweb Diagram (r = {r_cobweb})')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)

    # Plot 5: Lyapunov exponent
    ax5 = axes[1, 1]

    r_lyap = np.linspace(2.5, 4.0, 500)
    lyapunov = np.zeros_like(r_lyap)

    for i, r in enumerate(r_lyap):
        x = 0.5
        lyap_sum = 0
        n_iter = 1000

        for _ in range(100):  # Transient
            x = logistic_map(x, r)

        for _ in range(n_iter):
            x = logistic_map(x, r)
            derivative = abs(r * (1 - 2*x))
            if derivative > 0:
                lyap_sum += np.log(derivative)

        lyapunov[i] = lyap_sum / n_iter

    ax5.plot(r_lyap, lyapunov, 'b-', lw=1)
    ax5.axhline(y=0, color='red', linestyle='--', lw=1.5, label='λ = 0 (boundary of chaos)')
    ax5.fill_between(r_lyap, lyapunov, 0, where=(lyapunov > 0), alpha=0.3, color='red', label='Chaotic (λ > 0)')
    ax5.fill_between(r_lyap, lyapunov, 0, where=(lyapunov < 0), alpha=0.3, color='blue', label='Stable (λ < 0)')

    ax5.set_xlabel('r')
    ax5.set_ylabel('Lyapunov Exponent λ')
    ax5.set_title('Lyapunov Exponent vs r')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(-3, 1)

    # Plot 6: Sensitive dependence
    ax6 = axes[1, 2]

    r_sens = 3.9  # Chaotic regime
    x1_0 = 0.5
    x2_0 = 0.5 + 1e-10

    n_sens = 50
    x1 = np.zeros(n_sens)
    x2 = np.zeros(n_sens)
    x1[0], x2[0] = x1_0, x2_0

    for i in range(1, n_sens):
        x1[i] = logistic_map(x1[i-1], r_sens)
        x2[i] = logistic_map(x2[i-1], r_sens)

    ax6.plot(range(n_sens), x1, 'b-', lw=1.5, label='x₀ = 0.5')
    ax6.plot(range(n_sens), x2, 'r--', lw=1.5, label='x₀ = 0.5 + 10⁻¹⁰')

    ax6.set_xlabel('Iteration n')
    ax6.set_ylabel('x_n')
    ax6.set_title(f'Sensitive Dependence (r = {r_sens})')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Add text about Feigenbaum constant
    ax6.text(0.02, 0.02, 'Trajectories diverge\nafter ~30 iterations!',
             transform=ax6.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Logistic Map: x_{n+1} = rx_n(1 - x_n)\n'
                 'Period-Doubling Route to Chaos',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'logistic_map.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'logistic_map.png')}")


if __name__ == "__main__":
    main()
