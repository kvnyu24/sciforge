"""
Experiment 140: Cluster Algorithms vs Metropolis

This example compares the efficiency of cluster algorithms (Wolff, Swendsen-Wang)
versus the local Metropolis algorithm for the 2D Ising model.

Cluster algorithms avoid critical slowing down by flipping entire clusters of
aligned spins, leading to:
- Dynamic exponent z ≈ 0 for cluster algorithms vs z ≈ 2 for Metropolis
- Much better sampling near the critical temperature

Wolff algorithm:
1. Choose random seed spin
2. Grow cluster by adding aligned neighbors with probability p = 1 - exp(-2J/k_B*T)
3. Flip entire cluster

Swendsen-Wang algorithm:
1. Bond all aligned neighbor pairs with probability p = 1 - exp(-2J/k_B*T)
2. Identify connected clusters
3. Flip each cluster with probability 1/2
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time

# Critical temperature for 2D Ising
T_C = 2.0 / np.log(1 + np.sqrt(2))


def initialize_lattice(L, state='random'):
    """Initialize square lattice of spins."""
    if state == 'random':
        return np.random.choice([-1, 1], size=(L, L))
    return np.ones((L, L), dtype=int)


def metropolis_sweep(spins, T, J=1.0):
    """Perform one Metropolis sweep (L^2 single-spin flips)."""
    L = spins.shape[0]

    for _ in range(L * L):
        i, j = np.random.randint(L), np.random.randint(L)
        s = spins[i, j]

        neighbors = (spins[(i+1) % L, j] + spins[(i-1) % L, j] +
                    spins[i, (j+1) % L] + spins[i, (j-1) % L])
        delta_E = 2 * J * s * neighbors

        if delta_E <= 0 or np.random.random() < np.exp(-delta_E / T):
            spins[i, j] = -s


def wolff_step(spins, T, J=1.0):
    """
    Perform one Wolff cluster update.

    Returns:
        cluster_size: Number of spins flipped
    """
    L = spins.shape[0]
    p_add = 1 - np.exp(-2 * J / T)  # Probability to add aligned neighbor

    # Choose random seed
    i0, j0 = np.random.randint(L), np.random.randint(L)
    seed_spin = spins[i0, j0]

    # Track which sites are in the cluster
    in_cluster = np.zeros((L, L), dtype=bool)
    in_cluster[i0, j0] = True

    # BFS to grow cluster
    queue = deque([(i0, j0)])

    while queue:
        i, j = queue.popleft()

        # Check all neighbors
        for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            ni, nj = (i + di) % L, (j + dj) % L

            if not in_cluster[ni, nj] and spins[ni, nj] == seed_spin:
                if np.random.random() < p_add:
                    in_cluster[ni, nj] = True
                    queue.append((ni, nj))

    # Flip all spins in cluster
    spins[in_cluster] *= -1

    return np.sum(in_cluster)


def swendsen_wang_step(spins, T, J=1.0):
    """
    Perform one Swendsen-Wang cluster update.

    Returns:
        n_clusters: Number of clusters formed
    """
    L = spins.shape[0]
    p_bond = 1 - np.exp(-2 * J / T)

    # Create bonds between aligned neighbors
    h_bonds = np.zeros((L, L), dtype=bool)  # Horizontal bonds
    v_bonds = np.zeros((L, L), dtype=bool)  # Vertical bonds

    for i in range(L):
        for j in range(L):
            # Horizontal bond (to right neighbor)
            if spins[i, j] == spins[i, (j+1) % L]:
                if np.random.random() < p_bond:
                    h_bonds[i, j] = True
            # Vertical bond (to bottom neighbor)
            if spins[i, j] == spins[(i+1) % L, j]:
                if np.random.random() < p_bond:
                    v_bonds[i, j] = True

    # Find connected clusters using union-find
    parent = np.arange(L * L)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Union sites connected by bonds
    for i in range(L):
        for j in range(L):
            idx = i * L + j
            if h_bonds[i, j]:
                union(idx, i * L + (j + 1) % L)
            if v_bonds[i, j]:
                union(idx, ((i + 1) % L) * L + j)

    # Collect clusters and flip each with probability 1/2
    cluster_roots = {}
    for i in range(L):
        for j in range(L):
            root = find(i * L + j)
            if root not in cluster_roots:
                cluster_roots[root] = np.random.random() < 0.5

    # Flip clusters
    for i in range(L):
        for j in range(L):
            root = find(i * L + j)
            if cluster_roots[root]:
                spins[i, j] *= -1

    return len(cluster_roots)


def compute_autocorrelation(data, max_lag=None):
    """Compute autocorrelation function of time series."""
    n = len(data)
    if max_lag is None:
        max_lag = n // 4

    data_centered = data - np.mean(data)
    var = np.var(data)

    if var == 0:
        return np.zeros(max_lag)

    autocorr = np.correlate(data_centered, data_centered, mode='full')
    autocorr = autocorr[n-1:n-1+max_lag]
    autocorr /= var * np.arange(n, n - max_lag, -1)

    return autocorr


def estimate_autocorrelation_time(autocorr):
    """Estimate integrated autocorrelation time."""
    # Sum until autocorrelation becomes negative or very small
    tau = 0.5  # Start with 0.5 for lag=0 term
    for i, rho in enumerate(autocorr[1:], 1):
        if rho < 0.01:
            break
        tau += rho
    return tau


def run_comparison(L, T, n_sweeps, algorithm):
    """Run simulation and measure magnetization time series."""
    spins = initialize_lattice(L)
    N = L * L

    magnetizations = []
    cluster_sizes = []
    times = []

    start_time = time.time()

    for sweep in range(n_sweeps):
        if algorithm == 'metropolis':
            metropolis_sweep(spins, T)
        elif algorithm == 'wolff':
            cluster_sizes.append(wolff_step(spins, T))
        elif algorithm == 'swendsen_wang':
            cluster_sizes.append(swendsen_wang_step(spins, T))

        magnetizations.append(np.abs(np.sum(spins)) / N)

    elapsed = time.time() - start_time

    return np.array(magnetizations), np.array(cluster_sizes), elapsed


def main():
    print("Cluster Algorithms vs Metropolis for 2D Ising Model")
    print("=" * 60)
    print(f"Critical temperature T_c = {T_C:.4f}")

    # Parameters
    L = 32
    n_sweeps = 5000
    n_equilibrate = 1000

    print(f"\nLattice size: {L} x {L}")
    print(f"Number of sweeps: {n_sweeps}")

    # Run at T_c where difference is most pronounced
    T = T_C

    print(f"\nRunning simulations at T = T_c = {T:.4f}...")

    results = {}
    for algo in ['metropolis', 'wolff', 'swendsen_wang']:
        print(f"  {algo}...", end=' ')
        M, clusters, elapsed = run_comparison(L, T, n_sweeps, algo)
        results[algo] = {
            'M': M[n_equilibrate:],  # Discard equilibration
            'clusters': clusters[n_equilibrate:] if len(clusters) > 0 else None,
            'time': elapsed
        }
        print(f"done ({elapsed:.2f}s)")

    # Compute autocorrelation
    max_lag = 500
    autocorr = {}
    tau_int = {}

    for algo in results:
        autocorr[algo] = compute_autocorrelation(results[algo]['M'], max_lag)
        tau_int[algo] = estimate_autocorrelation_time(autocorr[algo])

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Magnetization time series
    ax1 = axes[0, 0]
    for algo, color in [('metropolis', 'blue'), ('wolff', 'red'),
                        ('swendsen_wang', 'green')]:
        ax1.plot(results[algo]['M'][:500], color=color, alpha=0.7,
                 label=algo.replace('_', '-').title())
    ax1.set_xlabel('Sweep', fontsize=12)
    ax1.set_ylabel('|M| / N', fontsize=12)
    ax1.set_title(f'Magnetization Time Series at T_c (L={L})', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Autocorrelation function
    ax2 = axes[0, 1]
    lags = np.arange(max_lag)
    for algo, color in [('metropolis', 'blue'), ('wolff', 'red'),
                        ('swendsen_wang', 'green')]:
        ax2.plot(lags[:200], autocorr[algo][:200], color=color,
                 label=f'{algo.replace("_", "-").title()}: $\\tau$ = {tau_int[algo]:.1f}')
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Lag (sweeps)', fontsize=12)
    ax2.set_ylabel('Autocorrelation', fontsize=12)
    ax2.set_title('Autocorrelation of Magnetization', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Cluster size distribution (Wolff)
    ax3 = axes[1, 0]
    if results['wolff']['clusters'] is not None:
        clusters = results['wolff']['clusters']
        ax3.hist(clusters / (L*L), bins=50, density=True, alpha=0.7,
                 color='red', label='Wolff')
        ax3.axvline(np.mean(clusters) / (L*L), color='darkred', linestyle='--',
                    label=f'Mean = {np.mean(clusters)/(L*L):.3f}')
    ax3.set_xlabel('Cluster size / N', fontsize=12)
    ax3.set_ylabel('Probability density', fontsize=12)
    ax3.set_title('Wolff Cluster Size Distribution at T_c', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Autocorrelation time vs system size
    ax4 = axes[1, 1]

    # Study size dependence
    L_values = [8, 16, 24, 32, 48]
    tau_L = {'metropolis': [], 'wolff': []}

    print("\nStudying size dependence...")
    for L_test in L_values:
        print(f"  L = {L_test}...", end=' ')
        for algo in ['metropolis', 'wolff']:
            M, _, _ = run_comparison(L_test, T_C, 3000, algo)
            ac = compute_autocorrelation(M[500:], 300)
            tau_L[algo].append(estimate_autocorrelation_time(ac))
        print("done")

    ax4.loglog(L_values, tau_L['metropolis'], 'bo-', markersize=8,
               label='Metropolis')
    ax4.loglog(L_values, tau_L['wolff'], 'rs-', markersize=8,
               label='Wolff')

    # Fit power laws
    from scipy.stats import linregress
    log_L = np.log(L_values)

    # Metropolis: tau ~ L^z with z ≈ 2
    slope_met, intercept, _, _, _ = linregress(log_L, np.log(tau_L['metropolis']))
    L_fit = np.linspace(L_values[0], L_values[-1], 50)
    ax4.loglog(L_fit, np.exp(intercept) * L_fit**slope_met, 'b--', alpha=0.5,
               label=f'Metropolis fit: z = {slope_met:.2f}')

    # Wolff: tau ~ L^z with z ≈ 0
    if min(tau_L['wolff']) > 0:
        slope_wolff, intercept_w, _, _, _ = linregress(log_L, np.log(tau_L['wolff']))
        ax4.loglog(L_fit, np.exp(intercept_w) * L_fit**slope_wolff, 'r--', alpha=0.5,
                   label=f'Wolff fit: z = {slope_wolff:.2f}')

    ax4.set_xlabel('System size L', fontsize=12)
    ax4.set_ylabel('Autocorrelation time $\\tau$', fontsize=12)
    ax4.set_title('Critical Slowing Down: $\\tau \\sim L^z$', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3, which='both')

    plt.suptitle('Cluster Algorithms vs Local Updates: Efficiency Comparison',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Print summary
    print("\n" + "=" * 60)
    print("Efficiency Summary at T_c")
    print("=" * 60)
    print(f"{'Algorithm':>20} {'tau_int':>12} {'Time (s)':>12} {'Eff. samples':>15}")
    print("-" * 60)

    n_prod = n_sweeps - n_equilibrate
    for algo in ['metropolis', 'wolff', 'swendsen_wang']:
        eff_samples = n_prod / (2 * tau_int[algo]) if tau_int[algo] > 0 else n_prod
        print(f"{algo:>20} {tau_int[algo]:>12.1f} {results[algo]['time']:>12.2f} "
              f"{eff_samples:>15.1f}")

    print(f"\nSpeedup factor (Wolff vs Metropolis): {tau_int['metropolis']/tau_int['wolff']:.1f}x")

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'cluster_vs_metropolis.png'),
                dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {os.path.join(output_dir, 'cluster_vs_metropolis.png')}")


if __name__ == "__main__":
    main()
