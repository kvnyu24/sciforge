"""
Experiment 146: Crooks Fluctuation Theorem

This example demonstrates the Crooks Fluctuation Theorem, which relates
the probability distributions of work in forward and reverse processes.

Crooks' Theorem states:
P_F(W) / P_R(-W) = exp(beta * (W - Delta_F))

where:
- P_F(W) = probability of work W in forward process
- P_R(-W) = probability of work -W in reverse process
- Delta_F = free energy difference (F_final - F_initial for forward)
- beta = 1/(k_B * T)

At W = Delta_F, the forward and reverse work distributions cross:
P_F(Delta_F) = P_R(-Delta_F)

This provides a powerful method to extract free energy differences
from the crossing point of forward and reverse work distributions.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.interpolate import interp1d


def harmonic_langevin_step(x, v, k, gamma, T, m, dt):
    """Perform one Langevin dynamics step for harmonic oscillator."""
    k_B = 1.0
    sigma = np.sqrt(2 * gamma * k_B * T / dt)
    R = np.random.normal(0, sigma)

    v_half = v + 0.5 * dt * (-k * x - gamma * v) / m + 0.5 * dt * R / m
    x_new = x + dt * v_half
    R_new = np.random.normal(0, sigma)
    v_new = (v_half + 0.5 * dt * (-k * x_new + R_new) / m) / (1 + 0.5 * gamma * dt / m)

    return x_new, v_new


def simulate_forward_reverse(k_init, k_final, n_steps, tau, T, n_realizations=5000):
    """
    Simulate both forward and reverse protocols.

    Forward: k goes from k_init to k_final
    Reverse: k goes from k_final to k_init

    Returns work distributions for both.
    """
    dt = tau / n_steps
    m = 1.0
    gamma = 1.0
    k_B = 1.0

    works_forward = []
    works_reverse = []

    for real in range(n_realizations):
        # Forward process
        x_f = np.random.normal(0, np.sqrt(k_B * T / k_init))
        v_f = np.random.normal(0, np.sqrt(k_B * T / m))

        W_f = 0.0
        k_prev = k_init

        for step in range(n_steps):
            t = (step + 1) * dt
            k_curr = k_init + (k_final - k_init) * t / tau
            W_f += 0.5 * (k_curr - k_prev) * x_f**2
            x_f, v_f = harmonic_langevin_step(x_f, v_f, k_curr, gamma, T, m, dt)
            k_prev = k_curr

        works_forward.append(W_f)

        # Reverse process
        x_r = np.random.normal(0, np.sqrt(k_B * T / k_final))
        v_r = np.random.normal(0, np.sqrt(k_B * T / m))

        W_r = 0.0
        k_prev = k_final

        for step in range(n_steps):
            t = (step + 1) * dt
            k_curr = k_final + (k_init - k_final) * t / tau
            W_r += 0.5 * (k_curr - k_prev) * x_r**2
            x_r, v_r = harmonic_langevin_step(x_r, v_r, k_curr, gamma, T, m, dt)
            k_prev = k_curr

        works_reverse.append(W_r)

    return np.array(works_forward), np.array(works_reverse)


def compute_work_distributions(works_forward, works_reverse, n_bins=100):
    """
    Compute probability distributions P_F(W) and P_R(-W).
    """
    # Combined range for consistent binning
    all_works = np.concatenate([works_forward, -works_reverse])
    W_min, W_max = np.min(all_works), np.max(all_works)
    bins = np.linspace(W_min - 0.1, W_max + 0.1, n_bins)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    # Forward distribution P_F(W)
    hist_f, _ = np.histogram(works_forward, bins=bins, density=True)

    # Reverse distribution P_R(-W) - note the sign flip
    hist_r, _ = np.histogram(-works_reverse, bins=bins, density=True)

    return bin_centers, hist_f, hist_r


def find_crossing_point(W, P_F, P_R):
    """
    Find the crossing point where P_F(W) = P_R(-W).

    This gives Delta_F directly from Crooks theorem.
    """
    # Interpolate to find crossing
    mask = (P_F > 1e-10) & (P_R > 1e-10)
    if np.sum(mask) < 5:
        return np.nan

    W_valid = W[mask]
    log_ratio = np.log(P_F[mask]) - np.log(P_R[mask])

    try:
        # Find where log(P_F/P_R) = 0
        f = interp1d(W_valid, log_ratio, kind='linear')
        # Search for zero crossing
        W_cross = brentq(f, W_valid[0], W_valid[-1])
        return W_cross
    except (ValueError, RuntimeError):
        return np.nan


def verify_crooks(W, P_F, P_R, Delta_F, T):
    """
    Verify Crooks theorem: P_F(W)/P_R(-W) = exp(beta*(W - Delta_F))
    """
    beta = 1.0 / T
    mask = (P_F > 1e-10) & (P_R > 1e-10)

    log_ratio_measured = np.log(P_F[mask] / P_R[mask])
    log_ratio_theory = beta * (W[mask] - Delta_F)

    return W[mask], log_ratio_measured, log_ratio_theory


def main():
    print("Crooks Fluctuation Theorem Demonstration")
    print("=" * 60)

    # Parameters
    k_init = 1.0
    k_final = 4.0
    T = 1.0
    k_B = 1.0
    beta = 1.0 / (k_B * T)

    n_steps = 500
    n_realizations = 10000

    # Exact free energy difference
    Delta_F_exact = 0.5 * k_B * T * np.log(k_final / k_init)
    print(f"Exact Delta_F = {Delta_F_exact:.4f}")

    # Different switching times
    tau_values = [0.5, 1.0, 2.0, 5.0]

    results = {}

    print("\nRunning forward and reverse simulations...")
    for tau in tau_values:
        print(f"  tau = {tau}...", end=' ')
        W_f, W_r = simulate_forward_reverse(k_init, k_final, n_steps, tau, T,
                                             n_realizations)

        # Compute distributions
        W, P_F, P_R = compute_work_distributions(W_f, W_r)

        # Find crossing point
        Delta_F_crossing = find_crossing_point(W, P_F, P_R)

        results[tau] = {
            'W_forward': W_f,
            'W_reverse': W_r,
            'W_bins': W,
            'P_F': P_F,
            'P_R': P_R,
            'Delta_F_crossing': Delta_F_crossing
        }

        print(f"Delta_F (crossing) = {Delta_F_crossing:.4f}")

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Forward and reverse work distributions
    ax1 = axes[0, 0]
    tau_demo = 1.0
    W = results[tau_demo]['W_bins']
    P_F = results[tau_demo]['P_F']
    P_R = results[tau_demo]['P_R']

    ax1.plot(W, P_F, 'b-', lw=2, label='$P_F(W)$')
    ax1.plot(W, P_R, 'r-', lw=2, label='$P_R(-W)$')
    ax1.axvline(Delta_F_exact, color='green', linestyle='--', lw=2,
                label=f'$\\Delta F$ = {Delta_F_exact:.3f}')
    ax1.axvline(results[tau_demo]['Delta_F_crossing'], color='purple',
                linestyle=':', lw=2, label=f'Crossing = {results[tau_demo]["Delta_F_crossing"]:.3f}')

    ax1.set_xlabel('Work W', fontsize=12)
    ax1.set_ylabel('Probability density', fontsize=12)
    ax1.set_title(f'Work Distributions (tau = {tau_demo})', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Log ratio verification
    ax2 = axes[0, 1]

    W_valid, log_meas, log_theory = verify_crooks(W, P_F, P_R, Delta_F_exact, T)

    ax2.plot(W_valid, log_meas, 'bo', markersize=4, alpha=0.5, label='Measured')
    ax2.plot(W_valid, log_theory, 'r-', lw=2, label='Theory')
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(Delta_F_exact, color='green', linestyle='--', alpha=0.5)

    ax2.set_xlabel('Work W', fontsize=12)
    ax2.set_ylabel(r'$\ln(P_F(W) / P_R(-W))$', fontsize=12)
    ax2.set_title('Crooks Theorem Verification', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Distributions for different tau
    ax3 = axes[1, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(tau_values)))

    for tau, color in zip(tau_values, colors):
        W = results[tau]['W_bins']
        P_F = results[tau]['P_F']
        P_R = results[tau]['P_R']

        ax3.plot(W, P_F, '-', color=color, lw=1.5, alpha=0.7)
        ax3.plot(W, P_R, '--', color=color, lw=1.5, alpha=0.7)

    # Add legend entries
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', linestyle='-', label='Forward'),
        Line2D([0], [0], color='black', linestyle='--', label='Reverse')
    ]
    for tau, color in zip(tau_values, colors):
        legend_elements.append(
            Line2D([0], [0], color=color, label=f'tau = {tau}'))

    ax3.axvline(Delta_F_exact, color='red', linestyle=':', lw=2)
    ax3.set_xlabel('Work W', fontsize=12)
    ax3.set_ylabel('Probability density', fontsize=12)
    ax3.set_title('Work Distributions for Different Switching Times', fontsize=12)
    ax3.legend(handles=legend_elements, fontsize=8, ncol=2)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Crossing point vs exact Delta_F
    ax4 = axes[1, 1]
    taus = list(results.keys())
    crossings = [results[tau]['Delta_F_crossing'] for tau in taus]

    ax4.plot(taus, crossings, 'bo-', markersize=10, label='Crossing point')
    ax4.axhline(Delta_F_exact, color='red', linestyle='--', lw=2,
                label=f'Exact $\\Delta F$ = {Delta_F_exact:.4f}')
    ax4.fill_between(taus, Delta_F_exact - 0.02, Delta_F_exact + 0.02,
                     alpha=0.2, color='red')

    ax4.set_xlabel('Switching time tau', fontsize=12)
    ax4.set_ylabel('$\\Delta F$ from crossing', fontsize=12)
    ax4.set_title('Free Energy from Crooks Crossing', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Crooks Fluctuation Theorem: Work Distributions in Forward/Reverse Processes',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Second figure: detailed analysis
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

    # Histograms overlay
    ax5 = axes2[0]
    tau = 1.0
    W_f = results[tau]['W_forward']
    W_r = results[tau]['W_reverse']

    ax5.hist(W_f, bins=50, density=True, alpha=0.5, color='blue',
             label='Forward $P_F(W)$')
    ax5.hist(-W_r, bins=50, density=True, alpha=0.5, color='red',
             label='Reverse $P_R(-W)$')
    ax5.axvline(Delta_F_exact, color='green', linestyle='--', lw=2,
                label=f'$\\Delta F$')
    ax5.set_xlabel('Work W', fontsize=12)
    ax5.set_ylabel('Probability density', fontsize=12)
    ax5.set_title('Forward vs Reverse Work Histograms', fontsize=12)
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Acceptance ratio method (CFT-based)
    ax6 = axes2[1]

    # Bennett acceptance ratio style analysis
    def acceptance_function(W, Delta_F_guess, beta):
        return 1.0 / (1.0 + np.exp(beta * (W - Delta_F_guess)))

    Delta_F_range = np.linspace(0, 1.5, 100)
    ratio = []

    for dF in Delta_F_range:
        f_forward = np.mean(acceptance_function(W_f, dF, beta))
        f_reverse = np.mean(acceptance_function(-W_r, -dF, beta))
        ratio.append(f_forward / f_reverse if f_reverse > 0 else np.nan)

    ax6.semilogy(Delta_F_range, ratio, 'b-', lw=2)
    ax6.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax6.axvline(Delta_F_exact, color='red', linestyle='--', lw=2,
                label=f'$\\Delta F$ = {Delta_F_exact:.3f}')

    # Find where ratio = 1
    ratio_arr = np.array(ratio)
    valid = ~np.isnan(ratio_arr)
    if np.any(ratio_arr[valid] > 1) and np.any(ratio_arr[valid] < 1):
        f_interp = interp1d(Delta_F_range[valid], np.log(ratio_arr[valid]))
        try:
            dF_bennett = brentq(f_interp, Delta_F_range[valid][0],
                               Delta_F_range[valid][-1])
            ax6.axvline(dF_bennett, color='purple', linestyle=':', lw=2,
                       label=f'Bennett = {dF_bennett:.3f}')
        except ValueError:
            pass

    ax6.set_xlabel('$\\Delta F$ guess', fontsize=12)
    ax6.set_ylabel('Acceptance ratio', fontsize=12)
    ax6.set_title('Bennett Acceptance Ratio Method', fontsize=12)
    ax6.legend()
    ax6.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    # Print summary
    print("\n" + "=" * 60)
    print("Crooks Fluctuation Theorem Results")
    print("=" * 60)
    print(f"Exact Delta_F = {Delta_F_exact:.4f}")
    print(f"\n{'tau':>8} {'Delta_F (cross)':>16} {'Error':>12}")
    print("-" * 40)
    for tau in taus:
        dF = results[tau]['Delta_F_crossing']
        err = abs(dF - Delta_F_exact)
        print(f"{tau:>8.2f} {dF:>16.4f} {err:>12.4f}")

    # Save plots
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'crooks_fluctuation.png'),
                dpi=150, bbox_inches='tight')
    fig2.savefig(os.path.join(output_dir, 'crooks_analysis.png'),
                 dpi=150, bbox_inches='tight')

    print(f"\nPlots saved to {output_dir}")


if __name__ == "__main__":
    main()
