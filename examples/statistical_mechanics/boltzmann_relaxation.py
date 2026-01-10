"""
Experiment 147: Boltzmann Equation Relaxation

This example demonstrates the relaxation of a velocity distribution
to the Maxwell-Boltzmann equilibrium, governed by the Boltzmann equation.

The Boltzmann equation describes the evolution of the distribution function f(v,t):
df/dt = (df/dt)_collision

For the BGK (Bhatnagar-Gross-Krook) relaxation approximation:
df/dt = -(f - f_eq) / tau

where:
- f = current distribution function
- f_eq = Maxwell-Boltzmann equilibrium distribution
- tau = relaxation time

The distribution relaxes exponentially: f(t) = f_eq + (f_0 - f_eq) * exp(-t/tau)

This example uses Direct Simulation Monte Carlo (DSMC) to model
velocity relaxation in a gas.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import maxwell


def maxwell_boltzmann_1d(v, T, m=1.0, k_B=1.0):
    """1D Maxwell-Boltzmann distribution (Gaussian)."""
    sigma = np.sqrt(k_B * T / m)
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-v**2 / (2 * sigma**2))


def maxwell_boltzmann_speed(v, T, m=1.0, k_B=1.0):
    """3D Maxwell-Boltzmann speed distribution."""
    prefactor = 4 * np.pi * (m / (2 * np.pi * k_B * T))**(3/2)
    return prefactor * v**2 * np.exp(-m * v**2 / (2 * k_B * T))


def bgk_relaxation(f0, v_grid, T_eq, tau, t_max, dt=0.01):
    """
    BGK relaxation model: df/dt = -(f - f_eq) / tau

    Analytical solution: f(t) = f_eq + (f_0 - f_eq) * exp(-t/tau)

    Args:
        f0: Initial distribution on velocity grid
        v_grid: Velocity grid points
        T_eq: Equilibrium temperature
        tau: Relaxation time
        t_max: Maximum simulation time
        dt: Time step

    Returns:
        times, distributions (time series of f)
    """
    f_eq = maxwell_boltzmann_1d(v_grid, T_eq)
    f_eq = f_eq / np.trapz(f_eq, v_grid)  # Normalize

    times = np.arange(0, t_max, dt)
    distributions = []

    f = f0.copy()
    for t in times:
        # Analytical BGK solution
        f = f_eq + (f0 - f_eq) * np.exp(-t / tau)
        distributions.append(f.copy())

    return times, np.array(distributions)


def dsmc_relaxation(n_particles, T_init, T_eq, n_steps, dt=0.01):
    """
    Direct Simulation Monte Carlo (DSMC) for velocity relaxation.

    Uses collision operator to relax distribution to equilibrium.
    Simple approach: random pairs collide and exchange velocity components.

    Args:
        n_particles: Number of particles
        T_init: Initial temperature (determines initial velocity spread)
        T_eq: Equilibrium temperature
        n_steps: Number of time steps
        dt: Time step

    Returns:
        times, velocity_history (for analysis)
    """
    m = 1.0
    k_B = 1.0

    # Initialize with non-equilibrium distribution
    # Two-temperature distribution (hot and cold populations)
    n_hot = n_particles // 2
    n_cold = n_particles - n_hot

    v_hot = np.random.normal(0, np.sqrt(k_B * 2 * T_init / m), (n_hot, 3))
    v_cold = np.random.normal(0, np.sqrt(k_B * 0.5 * T_init / m), (n_cold, 3))

    velocities = np.vstack([v_hot, v_cold])
    np.random.shuffle(velocities)

    times = []
    temp_history = []
    velocity_snapshots = []

    # Collision probability per step
    collision_prob = 0.1

    for step in range(n_steps):
        t = step * dt
        times.append(t)

        # Record temperature
        T_current = np.mean(velocities**2) * m / (3 * k_B)
        temp_history.append(T_current)

        # Save velocity snapshots
        if step % (n_steps // 20) == 0:
            velocity_snapshots.append(velocities.copy())

        # Collision step: random pairs exchange energy
        indices = np.random.permutation(n_particles)

        for i in range(0, n_particles - 1, 2):
            if np.random.random() < collision_prob:
                i1, i2 = indices[i], indices[i+1]

                # Simple elastic collision model
                # Exchange random component of velocity
                axis = np.random.randint(3)
                v1, v2 = velocities[i1, axis], velocities[i2, axis]

                # Conserve momentum and energy on average
                # Random redistribution
                total = v1 + v2
                v_new1 = total / 2 + np.random.normal(0, abs(v1 - v2) / 2)
                v_new2 = total - v_new1

                velocities[i1, axis] = v_new1
                velocities[i2, axis] = v_new2

    return np.array(times), np.array(temp_history), velocity_snapshots


def compute_distribution_distance(f1, f2, v_grid):
    """Compute L2 distance between two distributions."""
    return np.sqrt(np.trapz((f1 - f2)**2, v_grid))


def main():
    print("Boltzmann Equation Relaxation")
    print("=" * 60)

    # Parameters
    T_eq = 1.0  # Equilibrium temperature
    tau = 1.0   # Relaxation time
    m = 1.0
    k_B = 1.0

    # Velocity grid for BGK
    v_grid = np.linspace(-5, 5, 200)

    # Initial non-equilibrium distribution (bimodal)
    T_hot = 2.0
    T_cold = 0.5
    f0 = 0.5 * maxwell_boltzmann_1d(v_grid, T_hot) + 0.5 * maxwell_boltzmann_1d(v_grid, T_cold)
    f0 = f0 / np.trapz(f0, v_grid)  # Normalize

    print(f"Equilibrium temperature: T_eq = {T_eq}")
    print(f"Relaxation time: tau = {tau}")
    print(f"Initial distribution: Bimodal (T_hot={T_hot}, T_cold={T_cold})")

    # BGK relaxation
    t_max = 5 * tau
    times_bgk, dists_bgk = bgk_relaxation(f0, v_grid, T_eq, tau, t_max)

    # DSMC simulation
    n_particles = 10000
    n_steps = 500
    print(f"\nRunning DSMC with {n_particles} particles...")
    times_dsmc, temps_dsmc, v_snapshots = dsmc_relaxation(
        n_particles, T_eq, T_eq, n_steps, dt=t_max/n_steps
    )

    # Compute relaxation metrics
    f_eq = maxwell_boltzmann_1d(v_grid, T_eq)
    f_eq = f_eq / np.trapz(f_eq, v_grid)

    distances = [compute_distribution_distance(f, f_eq, v_grid) for f in dists_bgk]

    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Plot 1: Distribution evolution (BGK)
    ax1 = axes[0, 0]
    time_indices = [0, len(times_bgk)//10, len(times_bgk)//4,
                    len(times_bgk)//2, len(times_bgk)-1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(time_indices)))

    for idx, color in zip(time_indices, colors):
        t = times_bgk[idx]
        f = dists_bgk[idx]
        ax1.plot(v_grid, f, color=color, lw=2, label=f't = {t:.2f}')

    ax1.plot(v_grid, f_eq, 'r--', lw=2, label='Equilibrium')
    ax1.set_xlabel('Velocity v', fontsize=12)
    ax1.set_ylabel('f(v)', fontsize=12)
    ax1.set_title('BGK Relaxation', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Distance to equilibrium
    ax2 = axes[0, 1]
    ax2.semilogy(times_bgk, distances, 'b-', lw=2, label='$||f - f_{eq}||$')

    # Theoretical exponential decay
    d0 = distances[0]
    ax2.semilogy(times_bgk, d0 * np.exp(-np.array(times_bgk) / tau), 'r--',
                 lw=2, label=f'$e^{{-t/\\tau}}$, $\\tau$ = {tau}')

    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Distance to equilibrium', fontsize=12)
    ax2.set_title('Relaxation Dynamics', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')

    # Plot 3: DSMC temperature evolution
    ax3 = axes[0, 2]
    ax3.plot(times_dsmc, temps_dsmc, 'b-', lw=1.5, alpha=0.7)
    ax3.axhline(T_eq, color='red', linestyle='--', label=f'$T_{{eq}}$ = {T_eq}')
    ax3.set_xlabel('Time', fontsize=12)
    ax3.set_ylabel('Temperature', fontsize=12)
    ax3.set_title('DSMC Temperature Relaxation', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: DSMC velocity distribution snapshots
    ax4 = axes[1, 0]
    if len(v_snapshots) >= 4:
        snap_indices = [0, len(v_snapshots)//3, 2*len(v_snapshots)//3, -1]
        colors = plt.cm.plasma(np.linspace(0, 1, len(snap_indices)))

        for i, (idx, color) in enumerate(zip(snap_indices, colors)):
            speeds = np.linalg.norm(v_snapshots[idx], axis=1)
            ax4.hist(speeds, bins=50, density=True, alpha=0.4, color=color,
                     label=f'Snapshot {idx+1}')

        # Theoretical Maxwell speed distribution
        v_theory = np.linspace(0, 4, 100)
        f_theory = maxwell_boltzmann_speed(v_theory, T_eq)
        ax4.plot(v_theory, f_theory, 'k-', lw=2, label='Maxwell-Boltzmann')

    ax4.set_xlabel('Speed |v|', fontsize=12)
    ax4.set_ylabel('Probability density', fontsize=12)
    ax4.set_title('DSMC Speed Distribution Evolution', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Different relaxation times
    ax5 = axes[1, 1]
    tau_values = [0.5, 1.0, 2.0]

    for tau_test in tau_values:
        times_test, dists_test = bgk_relaxation(f0, v_grid, T_eq, tau_test, t_max)
        distances_test = [compute_distribution_distance(f, f_eq, v_grid)
                         for f in dists_test]
        ax5.semilogy(times_test, distances_test, lw=2, label=f'tau = {tau_test}')

    ax5.set_xlabel('Time', fontsize=12)
    ax5.set_ylabel('Distance to equilibrium', fontsize=12)
    ax5.set_title('Effect of Relaxation Time', fontsize=12)
    ax5.legend()
    ax5.grid(True, alpha=0.3, which='both')

    # Plot 6: Approach to equilibrium (phase space)
    ax6 = axes[1, 2]

    # Track moments of distribution
    mean_v2 = [np.trapz(v_grid**2 * f, v_grid) for f in dists_bgk]
    mean_v4 = [np.trapz(v_grid**4 * f, v_grid) for f in dists_bgk]

    # Kurtosis = <v^4> / <v^2>^2
    kurtosis = np.array(mean_v4) / np.array(mean_v2)**2

    # For Gaussian: kurtosis = 3
    ax6.plot(times_bgk, kurtosis, 'b-', lw=2, label='Kurtosis')
    ax6.axhline(3.0, color='red', linestyle='--', label='Gaussian (K=3)')
    ax6.set_xlabel('Time', fontsize=12)
    ax6.set_ylabel('Kurtosis', fontsize=12)
    ax6.set_title('Distribution Shape Evolution', fontsize=12)
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.suptitle('Boltzmann Equation: Relaxation to Maxwell-Boltzmann Equilibrium',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Print summary
    print("\n" + "=" * 60)
    print("Relaxation Summary")
    print("=" * 60)

    # Characteristic times
    for tau_test in tau_values:
        t_half = tau_test * np.log(2)
        t_99 = tau_test * np.log(100)
        print(f"\ntau = {tau_test}:")
        print(f"  Half-relaxation time: t_1/2 = {t_half:.3f}")
        print(f"  99% relaxation time: t_99 = {t_99:.3f}")

    # H-function check (will be detailed in next experiment)
    print("\nNote: The H-theorem guarantees entropy increases monotonically")
    print("during relaxation (see H-theorem experiment).")

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'boltzmann_relaxation.png'),
                dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {os.path.join(output_dir, 'boltzmann_relaxation.png')}")


if __name__ == "__main__":
    main()
