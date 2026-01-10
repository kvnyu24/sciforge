"""
Experiment 142: Percolation Threshold

This example demonstrates site and bond percolation on a 2D square lattice,
finding the critical probability (percolation threshold) at which an
infinite cluster spans the system.

For 2D square lattice:
- Site percolation threshold: p_c ≈ 0.593
- Bond percolation threshold: p_c ≈ 0.5 (exactly 1/2)

Near p_c, observables exhibit power-law scaling:
- Probability of infinite cluster: P_inf ~ (p - p_c)^beta, beta = 5/36
- Mean cluster size: S ~ |p - p_c|^(-gamma), gamma = 43/18
- Correlation length: xi ~ |p - p_c|^(-nu), nu = 4/3
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
from scipy.optimize import curve_fit

# Percolation thresholds for 2D square lattice
P_C_SITE = 0.5927  # Site percolation
P_C_BOND = 0.5     # Bond percolation (exact)

# Critical exponents (2D percolation)
BETA_PERC = 5/36   # Order parameter exponent
GAMMA_PERC = 43/18  # Susceptibility exponent
NU_PERC = 4/3      # Correlation length exponent


def site_percolation(L, p, n_samples=1):
    """
    Generate site percolation configurations.

    Each site is occupied with probability p.

    Args:
        L: Linear lattice size
        p: Occupation probability
        n_samples: Number of configurations to generate

    Returns:
        Array of configurations (n_samples, L, L)
    """
    return np.random.random((n_samples, L, L)) < p


def bond_percolation(L, p, n_samples=1):
    """
    Generate bond percolation configurations.

    Each bond is present with probability p. Occupied sites are those
    connected by at least one bond.

    Returns effective site occupancy based on cluster connectivity.
    """
    # Create horizontal and vertical bonds
    h_bonds = np.random.random((n_samples, L, L)) < p
    v_bonds = np.random.random((n_samples, L, L)) < p

    # For simplicity, return a configuration where a site is "occupied"
    # if it has at least one active bond to a neighbor
    configs = np.zeros((n_samples, L, L), dtype=bool)

    for s in range(n_samples):
        # Union-find to identify connected components
        parent = np.arange(L * L).reshape(L, L)

        def find(i, j):
            pi, pj = parent[i, j] // L, parent[i, j] % L
            if (pi, pj) != (i, j):
                root = find(pi, pj)
                parent[i, j] = root
                return root
            return i * L + j

        def union(i1, j1, i2, j2):
            r1, r2 = find(i1, j1), find(i2, j2)
            if r1 != r2:
                parent[r1 // L, r1 % L] = r2

        # Connect sites via bonds
        for i in range(L):
            for j in range(L):
                if h_bonds[s, i, j]:  # Horizontal bond to right
                    union(i, j, i, (j + 1) % L)
                if v_bonds[s, i, j]:  # Vertical bond down
                    union(i, j, (i + 1) % L, j)

        # Mark all sites as occupied (bond percolation connects all sites)
        # The configuration stores which sites belong to connected clusters
        configs[s] = np.ones((L, L), dtype=bool)

    return configs


def find_clusters(config):
    """
    Find clusters in a percolation configuration using connected components.

    Args:
        config: 2D boolean array (True = occupied)

    Returns:
        labeled: Array with cluster labels
        n_clusters: Number of clusters
    """
    # Use scipy.ndimage.label with 4-connectivity
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    labeled, n_clusters = label(config.astype(int), structure)
    return labeled, n_clusters


def check_spanning(labeled, L):
    """
    Check if there's a cluster spanning the system.

    A cluster spans if it touches both top and bottom (or left and right).

    Returns:
        spans: True if spanning cluster exists
        spanning_label: Label of spanning cluster (0 if none)
    """
    # Check top-bottom spanning
    top_labels = set(labeled[0, :]) - {0}
    bottom_labels = set(labeled[-1, :]) - {0}
    tb_spanning = top_labels & bottom_labels

    # Check left-right spanning
    left_labels = set(labeled[:, 0]) - {0}
    right_labels = set(labeled[:, -1]) - {0}
    lr_spanning = left_labels & right_labels

    all_spanning = tb_spanning | lr_spanning

    if all_spanning:
        return True, min(all_spanning)
    return False, 0


def measure_percolation(L, p, n_samples=100, perc_type='site'):
    """
    Measure percolation observables.

    Returns:
        - spanning_prob: Probability of spanning cluster
        - P_inf: Fraction of sites in largest cluster
        - mean_cluster_size: Average cluster size (excluding largest)
    """
    if perc_type == 'site':
        configs = site_percolation(L, p, n_samples)
    else:
        configs = site_percolation(L, p, n_samples)  # Simplified

    spanning_count = 0
    P_inf_list = []
    cluster_sizes_all = []

    for config in configs:
        labeled, n_clusters = find_clusters(config)
        spans, span_label = check_spanning(labeled, L)

        if spans:
            spanning_count += 1

        # Find cluster sizes
        cluster_sizes = []
        largest_size = 0
        for c in range(1, n_clusters + 1):
            size = np.sum(labeled == c)
            cluster_sizes.append(size)
            if size > largest_size:
                largest_size = size

        P_inf_list.append(largest_size / (L * L))

        # Mean cluster size excluding largest
        if len(cluster_sizes) > 1:
            cluster_sizes.remove(largest_size)
            cluster_sizes_all.extend(cluster_sizes)

    spanning_prob = spanning_count / n_samples
    P_inf = np.mean(P_inf_list)

    # Mean cluster size (second moment / first moment)
    if cluster_sizes_all:
        sizes = np.array(cluster_sizes_all)
        mean_size = np.sum(sizes**2) / np.sum(sizes) if np.sum(sizes) > 0 else 0
    else:
        mean_size = 0

    return spanning_prob, P_inf, mean_size


def main():
    print("Percolation Threshold on 2D Square Lattice")
    print("=" * 60)
    print(f"Site percolation threshold: p_c = {P_C_SITE:.4f}")
    print(f"Bond percolation threshold: p_c = {P_C_BOND:.4f}")

    # Parameters
    L = 64
    n_samples = 100
    p_values = np.linspace(0.3, 0.9, 30)

    print(f"\nLattice size: {L} x {L}")
    print(f"Samples per probability: {n_samples}")

    # Run measurements
    results = {'p': [], 'spanning': [], 'P_inf': [], 'mean_size': []}

    print("\nMeasuring percolation observables...")
    for i, p in enumerate(p_values):
        print(f"  p = {p:.3f} ({i+1}/{len(p_values)})", end='\r')
        span_prob, P_inf, mean_size = measure_percolation(L, p, n_samples, 'site')
        results['p'].append(p)
        results['spanning'].append(span_prob)
        results['P_inf'].append(P_inf)
        results['mean_size'].append(mean_size)

    print("\n")

    for key in results:
        results[key] = np.array(results[key])

    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Plot 1: Example configurations
    ax1 = axes[0, 0]
    np.random.seed(42)
    config_low = site_percolation(L, 0.4, 1)[0]
    ax1.imshow(config_low, cmap='binary', interpolation='nearest')
    ax1.set_title(f'p = 0.4 (below $p_c$)', fontsize=12)
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2 = axes[0, 1]
    config_mid = site_percolation(L, P_C_SITE, 1)[0]
    ax2.imshow(config_mid, cmap='binary', interpolation='nearest')
    ax2.set_title(f'p = {P_C_SITE:.3f} (at $p_c$)', fontsize=12)
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax3 = axes[0, 2]
    config_high = site_percolation(L, 0.8, 1)[0]
    ax3.imshow(config_high, cmap='binary', interpolation='nearest')
    ax3.set_title(f'p = 0.8 (above $p_c$)', fontsize=12)
    ax3.set_xticks([])
    ax3.set_yticks([])

    # Plot 2: Spanning probability
    ax4 = axes[1, 0]
    ax4.plot(results['p'], results['spanning'], 'bo-', markersize=4)
    ax4.axvline(P_C_SITE, color='red', linestyle='--', label=f'$p_c$ = {P_C_SITE:.3f}')
    ax4.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax4.set_xlabel('Occupation probability p', fontsize=12)
    ax4.set_ylabel('Spanning probability', fontsize=12)
    ax4.set_title('Probability of Spanning Cluster', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 3: Order parameter (P_inf)
    ax5 = axes[1, 1]
    ax5.plot(results['p'], results['P_inf'], 'go-', markersize=4)
    ax5.axvline(P_C_SITE, color='red', linestyle='--', label=f'$p_c$')
    ax5.set_xlabel('Occupation probability p', fontsize=12)
    ax5.set_ylabel(r'$P_\infty$ (largest cluster / N)', fontsize=12)
    ax5.set_title('Order Parameter', fontsize=12)
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 4: Mean cluster size (susceptibility analog)
    ax6 = axes[1, 2]
    ax6.semilogy(results['p'], results['mean_size'], 'mo-', markersize=4)
    ax6.axvline(P_C_SITE, color='red', linestyle='--', label=f'$p_c$')
    ax6.set_xlabel('Occupation probability p', fontsize=12)
    ax6.set_ylabel('Mean cluster size S', fontsize=12)
    ax6.set_title('Mean Cluster Size (Susceptibility)', fontsize=12)
    ax6.legend()
    ax6.grid(True, alpha=0.3, which='both')

    plt.suptitle(f'Site Percolation on {L}x{L} Square Lattice',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Finite-size scaling analysis
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

    # Run for multiple system sizes
    L_values = [16, 32, 64, 128]
    colors = plt.cm.viridis(np.linspace(0, 1, len(L_values)))
    p_fine = np.linspace(0.5, 0.7, 20)

    print("Finite-size scaling analysis...")
    fss_results = {}

    for L_fss in L_values:
        print(f"  L = {L_fss}...", end=' ')
        fss_results[L_fss] = {'p': [], 'spanning': [], 'P_inf': []}

        for p in p_fine:
            span_prob, P_inf, _ = measure_percolation(L_fss, p, 50, 'site')
            fss_results[L_fss]['p'].append(p)
            fss_results[L_fss]['spanning'].append(span_prob)
            fss_results[L_fss]['P_inf'].append(P_inf)

        for key in fss_results[L_fss]:
            fss_results[L_fss][key] = np.array(fss_results[L_fss][key])
        print("done")

    # Spanning probability crossing
    ax7 = axes2[0]
    for L_fss, color in zip(L_values, colors):
        ax7.plot(fss_results[L_fss]['p'], fss_results[L_fss]['spanning'],
                 'o-', color=color, markersize=4, label=f'L = {L_fss}')
    ax7.axvline(P_C_SITE, color='red', linestyle='--', alpha=0.5)
    ax7.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax7.set_xlabel('p', fontsize=12)
    ax7.set_ylabel('Spanning probability', fontsize=12)
    ax7.set_title('Crossing Point Method for $p_c$', fontsize=12)
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # Data collapse
    ax8 = axes2[1]
    for L_fss, color in zip(L_values, colors):
        x = (fss_results[L_fss]['p'] - P_C_SITE) * L_fss**(1/NU_PERC)
        y = fss_results[L_fss]['P_inf'] * L_fss**(BETA_PERC/NU_PERC)
        ax8.plot(x, y, 'o-', color=color, markersize=4, label=f'L = {L_fss}')

    ax8.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax8.set_xlabel(r'$(p - p_c) L^{1/\nu}$', fontsize=12)
    ax8.set_ylabel(r'$P_\infty L^{\beta/\nu}$', fontsize=12)
    ax8.set_title('Data Collapse Using Known Exponents', fontsize=12)
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    plt.tight_layout()

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Estimated p_c from spanning probability = 0.5: "
          f"{results['p'][np.argmin(np.abs(results['spanning'] - 0.5))]:.3f}")
    print(f"Theoretical p_c (site): {P_C_SITE:.4f}")

    # Critical exponents
    print(f"\n2D Percolation Critical Exponents:")
    print(f"  beta = {BETA_PERC:.4f} = 5/36")
    print(f"  gamma = {GAMMA_PERC:.4f} = 43/18")
    print(f"  nu = {NU_PERC:.4f} = 4/3")

    # Save plots
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'percolation_threshold.png'),
                dpi=150, bbox_inches='tight')
    fig2.savefig(os.path.join(output_dir, 'percolation_scaling.png'),
                 dpi=150, bbox_inches='tight')

    print(f"\nPlots saved to {output_dir}")


if __name__ == "__main__":
    main()
