"""
Experiment 145: Jarzynski Equality

This example demonstrates the Jarzynski equality, which relates
nonequilibrium work measurements to equilibrium free energy differences.

The Jarzynski Equality states:
<exp(-beta * W)> = exp(-beta * Delta_F)

where:
- W = work done on the system during a nonequilibrium process
- Delta_F = free energy difference between final and initial equilibrium states
- beta = 1/(k_B * T)
- <...> = average over many realizations of the protocol

This remarkable result allows extracting equilibrium information from
arbitrarily far-from-equilibrium measurements.

For a harmonic oscillator with time-dependent spring constant k(t):
- The free energy is F = (1/2) * k_B * T * ln(k/k_0)
- We can verify Jarzynski by comparing <exp(-W/k_B*T)> with exp(-Delta_F/k_B*T)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def harmonic_langevin_step(x, v, k, gamma, T, m, dt):
    """
    Perform one Langevin dynamics step for a harmonic oscillator.

    dx/dt = v
    m*dv/dt = -k*x - gamma*v + sqrt(2*gamma*k_B*T)*eta

    Uses velocity Verlet with Langevin thermostat (BBK integrator).
    """
    k_B = 1.0  # Using reduced units

    # Random force
    sigma = np.sqrt(2 * gamma * k_B * T / dt)
    R = np.random.normal(0, sigma)

    # Half step velocity update
    v_half = v + 0.5 * dt * (-k * x - gamma * v) / m + 0.5 * dt * R / m

    # Full position update
    x_new = x + dt * v_half

    # New force
    R_new = np.random.normal(0, sigma)

    # Complete velocity update
    v_new = (v_half + 0.5 * dt * (-k * x_new + R_new) / m) / (1 + 0.5 * gamma * dt / m)

    return x_new, v_new


def simulate_protocol(k_init, k_final, n_steps, tau, T, n_realizations=1000):
    """
    Simulate the work distribution for a switching protocol.

    The spring constant changes linearly from k_init to k_final over time tau.

    Args:
        k_init: Initial spring constant
        k_final: Final spring constant
        n_steps: Number of time steps
        tau: Total switching time
        T: Temperature
        n_realizations: Number of independent trajectories

    Returns:
        works: Array of work values for each realization
        trajectories: Sample trajectories for visualization
    """
    dt = tau / n_steps
    m = 1.0
    gamma = 1.0  # Friction coefficient
    k_B = 1.0

    works = []
    trajectories = []

    for real in range(n_realizations):
        # Initialize from equilibrium distribution
        x = np.random.normal(0, np.sqrt(k_B * T / k_init))
        v = np.random.normal(0, np.sqrt(k_B * T / m))

        W = 0.0  # Work accumulator
        traj_x = [x]

        k_prev = k_init

        for step in range(n_steps):
            # Current spring constant (linear protocol)
            t = (step + 1) * dt
            k_curr = k_init + (k_final - k_init) * t / tau

            # Work done by changing k: dW = (dk/dt) * x^2 / 2
            dW = 0.5 * (k_curr - k_prev) * x**2
            W += dW

            # Langevin step
            x, v = harmonic_langevin_step(x, v, k_curr, gamma, T, m, dt)

            k_prev = k_curr
            traj_x.append(x)

        works.append(W)

        if real < 5:  # Save first few trajectories
            trajectories.append(traj_x)

    return np.array(works), trajectories


def compute_free_energy_difference(k_init, k_final, T):
    """
    Compute exact free energy difference for harmonic oscillator.

    F = (1/2) * k_B * T * ln(k / (k_B * T))

    Delta_F = F(k_final) - F(k_init) = (1/2) * k_B * T * ln(k_final / k_init)
    """
    k_B = 1.0
    return 0.5 * k_B * T * np.log(k_final / k_init)


def main():
    print("Jarzynski Equality Demonstration")
    print("=" * 60)
    print("For harmonic oscillator with time-dependent spring constant")

    # Parameters
    k_init = 1.0
    k_final = 4.0  # Increase spring constant by factor of 4
    T = 1.0
    k_B = 1.0
    beta = 1.0 / (k_B * T)

    n_realizations = 5000

    # Exact free energy difference
    Delta_F_exact = compute_free_energy_difference(k_init, k_final, T)
    print(f"\nExact free energy difference: Delta_F = {Delta_F_exact:.4f}")

    # Study different switching times (fast vs slow)
    tau_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    n_steps = 500

    results = {}

    print("\nRunning simulations for different switching times...")
    for tau in tau_values:
        print(f"  tau = {tau:.1f}...", end=' ')
        works, trajectories = simulate_protocol(k_init, k_final, n_steps, tau, T,
                                                 n_realizations)

        # Jarzynski estimate
        exp_W = np.exp(-beta * works)
        jarzynski_avg = np.mean(exp_W)
        Delta_F_jarzynski = -np.log(jarzynski_avg) / beta

        # Standard work averages
        mean_W = np.mean(works)
        std_W = np.std(works)

        results[tau] = {
            'works': works,
            'trajectories': trajectories,
            'mean_W': mean_W,
            'std_W': std_W,
            'Delta_F_jarzynski': Delta_F_jarzynski,
            'jarzynski_error': abs(Delta_F_jarzynski - Delta_F_exact) / abs(Delta_F_exact)
        }

        print(f"<W> = {mean_W:.3f}, Delta_F_J = {Delta_F_jarzynski:.4f}")

    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Plot 1: Work distributions for different tau
    ax1 = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(tau_values)))

    for tau, color in zip(tau_values[::2], colors[::2]):
        works = results[tau]['works']
        ax1.hist(works, bins=50, density=True, alpha=0.4, color=color,
                 label=f'tau = {tau}')

    ax1.axvline(Delta_F_exact, color='red', linestyle='--', lw=2,
                label=f'$\\Delta F$ = {Delta_F_exact:.3f}')
    ax1.set_xlabel('Work W', fontsize=12)
    ax1.set_ylabel('Probability density', fontsize=12)
    ax1.set_title('Work Distributions', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Jarzynski estimate vs tau
    ax2 = axes[0, 1]
    taus = list(results.keys())
    Delta_F_values = [results[tau]['Delta_F_jarzynski'] for tau in taus]
    mean_W_values = [results[tau]['mean_W'] for tau in taus]

    ax2.semilogx(taus, Delta_F_values, 'bo-', markersize=8, label='$\\Delta F$ (Jarzynski)')
    ax2.semilogx(taus, mean_W_values, 'gs-', markersize=8, label='$\\langle W \\rangle$')
    ax2.axhline(Delta_F_exact, color='red', linestyle='--', lw=2,
                label=f'$\\Delta F$ exact = {Delta_F_exact:.3f}')
    ax2.set_xlabel('Switching time tau', fontsize=12)
    ax2.set_ylabel('Energy', fontsize=12)
    ax2.set_title('Jarzynski Estimate vs Switching Time', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')

    # Plot 3: Sample trajectories
    ax3 = axes[0, 2]
    tau_demo = 1.0
    time = np.linspace(0, tau_demo, n_steps + 1)
    for traj in results[tau_demo]['trajectories'][:5]:
        ax3.plot(time, traj, alpha=0.7)
    ax3.set_xlabel('Time', fontsize=12)
    ax3.set_ylabel('Position x', fontsize=12)
    ax3.set_title(f'Sample Trajectories (tau = {tau_demo})', fontsize=12)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Verification of Jarzynski equality
    ax4 = axes[1, 0]
    exp_W_vals = [np.mean(np.exp(-beta * results[tau]['works'])) for tau in taus]
    exp_F_exact = np.exp(-beta * Delta_F_exact)

    ax4.semilogx(taus, exp_W_vals, 'bo-', markersize=8,
                 label=r'$\langle e^{-\beta W} \rangle$')
    ax4.axhline(exp_F_exact, color='red', linestyle='--', lw=2,
                label=f'$e^{{-\\beta \\Delta F}}$ = {exp_F_exact:.4f}')
    ax4.set_xlabel('Switching time tau', fontsize=12)
    ax4.set_ylabel(r'$\langle e^{-\beta W} \rangle$', fontsize=12)
    ax4.set_title('Jarzynski Equality Verification', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3, which='both')

    # Plot 5: Dissipated work (W - Delta_F)
    ax5 = axes[1, 1]
    W_diss = [results[tau]['mean_W'] - Delta_F_exact for tau in taus]

    ax5.loglog(taus, W_diss, 'ro-', markersize=8)
    ax5.set_xlabel('Switching time tau', fontsize=12)
    ax5.set_ylabel(r'$\langle W \rangle - \Delta F$ (dissipated work)', fontsize=12)
    ax5.set_title('Dissipation vs Switching Speed', fontsize=12)
    ax5.grid(True, alpha=0.3, which='both')

    # Theoretical: W_diss ~ 1/tau for slow switching
    tau_fit = np.array(taus)
    ax5.loglog(tau_fit, W_diss[0] * taus[0] / tau_fit, 'k--', alpha=0.5,
               label=r'$\sim 1/\tau$')
    ax5.legend()

    # Plot 6: Convergence with number of realizations
    ax6 = axes[1, 2]
    tau_conv = 1.0
    works_full = results[tau_conv]['works']

    n_vals = np.logspace(1, np.log10(len(works_full)), 30).astype(int)
    Delta_F_estimates = []

    for n in n_vals:
        exp_W = np.exp(-beta * works_full[:n])
        Delta_F_est = -np.log(np.mean(exp_W)) / beta
        Delta_F_estimates.append(Delta_F_est)

    ax6.semilogx(n_vals, Delta_F_estimates, 'b-', lw=2)
    ax6.axhline(Delta_F_exact, color='red', linestyle='--', lw=2,
                label=f'$\\Delta F$ exact')
    ax6.fill_between([n_vals[0], n_vals[-1]],
                     [Delta_F_exact - 0.05, Delta_F_exact - 0.05],
                     [Delta_F_exact + 0.05, Delta_F_exact + 0.05],
                     alpha=0.2, color='red')
    ax6.set_xlabel('Number of realizations', fontsize=12)
    ax6.set_ylabel('$\\Delta F$ (Jarzynski estimate)', fontsize=12)
    ax6.set_title('Convergence of Jarzynski Estimate', fontsize=12)
    ax6.legend()
    ax6.grid(True, alpha=0.3, which='both')

    plt.suptitle('Jarzynski Equality: Connecting Nonequilibrium Work to Free Energy',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Print summary
    print("\n" + "=" * 60)
    print("Jarzynski Equality Results")
    print("=" * 60)
    print(f"Exact Delta_F = {Delta_F_exact:.4f}")
    print(f"\n{'tau':>8} {'<W>':>10} {'Delta_F_J':>12} {'Error %':>10}")
    print("-" * 45)
    for tau in taus:
        print(f"{tau:>8.2f} {results[tau]['mean_W']:>10.4f} "
              f"{results[tau]['Delta_F_jarzynski']:>12.4f} "
              f"{results[tau]['jarzynski_error']*100:>10.2f}")

    print(f"\nNote: <W> >= Delta_F (second law)")
    print(f"The Jarzynski equality recovers Delta_F from <exp(-W/kT)>")

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'jarzynski_equality.png'),
                dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {os.path.join(output_dir, 'jarzynski_equality.png')}")


if __name__ == "__main__":
    main()
