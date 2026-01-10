"""
Experiment 217: Block-Spin Renormalization Group

Demonstrates the block-spin (real-space) renormalization group for the
1D Ising model. Shows how coarse-graining generates effective couplings.

Physics:
- Block spins: S'_i = sign(s_{2i} + s_{2i+1})
- RG flow: K' = f(K)
- Fixed points: K* = 0 (high T), K* = ∞ (low T)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def block_spin_1d(spins):
    """
    Block spin transformation for 1D.

    Combine pairs of spins into single block spin.
    S'_i = sign(s_{2i} + s_{2i+1})
    """
    n = len(spins)
    n_new = n // 2

    blocked = np.zeros(n_new)
    for i in range(n_new):
        sum_spin = spins[2*i] + spins[2*i + 1]
        blocked[i] = np.sign(sum_spin) if sum_spin != 0 else np.random.choice([-1, 1])

    return blocked


def majority_rule_2d(spins, block_size=2):
    """
    Majority rule blocking for 2D.

    S'_{i,j} = sign(sum of spins in block)
    """
    n = spins.shape[0]
    n_new = n // block_size

    blocked = np.zeros((n_new, n_new))

    for i in range(n_new):
        for j in range(n_new):
            block = spins[i*block_size:(i+1)*block_size,
                         j*block_size:(j+1)*block_size]
            block_sum = np.sum(block)
            blocked[i, j] = np.sign(block_sum) if block_sum != 0 else np.random.choice([-1, 1])

    return blocked


def ising_energy_1d(spins, J):
    """1D Ising energy: E = -J Σ s_i s_{i+1}"""
    n = len(spins)
    energy = 0
    for i in range(n - 1):
        energy -= J * spins[i] * spins[i + 1]
    return energy


def ising_correlation(spins, r):
    """Calculate <s_i s_{i+r}>"""
    n = len(spins)
    if r >= n:
        return 0
    corr = 0
    count = 0
    for i in range(n - r):
        corr += spins[i] * spins[i + r]
        count += 1
    return corr / count if count > 0 else 0


def metropolis_step_1d(spins, beta, J):
    """Single Metropolis update for 1D Ising."""
    n = len(spins)
    i = np.random.randint(n)

    # Energy change for flipping spin i
    neighbors = 0
    if i > 0:
        neighbors += spins[i - 1]
    if i < n - 1:
        neighbors += spins[i + 1]

    delta_E = 2 * J * spins[i] * neighbors

    if delta_E <= 0 or np.random.random() < np.exp(-beta * delta_E):
        spins[i] *= -1

    return spins


def simulate_and_block(n_sites, beta, J, n_sweeps, n_blocks):
    """
    Simulate and apply blocking transformation.

    Returns correlation functions at each blocking level.
    """
    # Initialize random spins
    spins = np.random.choice([-1, 1], size=n_sites)

    # Thermalize
    for _ in range(n_sweeps):
        for _ in range(n_sites):
            spins = metropolis_step_1d(spins, beta, J)

    # Store blocked configurations
    blocked_spins = [spins.copy()]

    for _ in range(n_blocks):
        spins = block_spin_1d(spins)
        blocked_spins.append(spins.copy())

    return blocked_spins


def rg_flow_1d_exact(K):
    """
    Exact RG equation for 1D Ising.

    K' = ½ ln(cosh(2K))

    where K = βJ
    """
    return 0.5 * np.log(np.cosh(2 * K))


def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: RG flow for 1D Ising
    ax = axes[0, 0]

    K = np.linspace(0.01, 3, 100)
    K_prime = rg_flow_1d_exact(K)

    ax.plot(K, K_prime, 'b-', lw=2, label="K' = f(K)")
    ax.plot(K, K, 'k--', lw=1, label="K' = K")

    # Fixed points
    ax.plot([0], [0], 'go', markersize=10, label='High-T fixed point')

    ax.set_xlabel('K = βJ')
    ax.set_ylabel("K' (renormalized)")
    ax.set_title('1D Ising RG Flow\nK\' = ½ ln(cosh 2K)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 2)

    # Plot 2: RG trajectory
    ax = axes[0, 1]

    K0_values = [0.2, 0.5, 1.0, 1.5, 2.0]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(K0_values)))

    for K0, color in zip(K0_values, colors):
        K_traj = [K0]
        for _ in range(10):
            K_traj.append(rg_flow_1d_exact(K_traj[-1]))

        ax.plot(range(len(K_traj)), K_traj, 'o-', color=color,
                lw=2, markersize=6, label=f'K₀ = {K0}')

    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('RG Iteration')
    ax.set_ylabel('K')
    ax.set_title('RG Trajectories\nAll flow to K=0 (disordered)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Block spin visualization
    ax = axes[0, 2]

    n_sites = 64
    beta = 1.0
    J = 1.0

    blocked_spins = simulate_and_block(n_sites, beta, J, n_sweeps=1000, n_blocks=4)

    # Plot as images
    levels = len(blocked_spins)
    for i, spins in enumerate(blocked_spins):
        y = levels - i
        x = np.arange(len(spins)) * (n_sites / len(spins))
        width = n_sites / len(spins)

        for j, s in enumerate(spins):
            color = 'blue' if s > 0 else 'red'
            ax.barh(y, width, left=j * width, color=color, alpha=0.7,
                    edgecolor='white', height=0.8)

    ax.set_xlabel('Site')
    ax.set_ylabel('Block Level')
    ax.set_title('Block Spin Coarse Graining\n(Blue: +1, Red: -1)')
    ax.set_yticks(range(1, levels + 1))
    ax.set_yticklabels([f'Level {i}' for i in range(levels)])
    ax.grid(True, alpha=0.3)

    # Plot 4: Correlation length scaling
    ax = axes[1, 0]

    n_sites = 256
    betas = [0.3, 0.5, 0.8, 1.0]
    J = 1.0
    n_sweeps = 2000

    for beta in betas:
        blocked = simulate_and_block(n_sites, beta, J, n_sweeps, n_blocks=5)

        # Calculate correlation at r=1 for each level
        corr_vs_level = []
        for level, spins in enumerate(blocked):
            if len(spins) > 1:
                c = ising_correlation(spins, 1)
                corr_vs_level.append(abs(c))

        ax.plot(range(len(corr_vs_level)), corr_vs_level, 'o-', lw=2,
                markersize=8, label=f'β = {beta}')

    ax.set_xlabel('Block Level')
    ax.set_ylabel('|<s_i s_{i+1}>|')
    ax.set_title('Nearest-Neighbor Correlation\nUnder Blocking')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Magnetization under blocking
    ax = axes[1, 1]

    n_sites = 512
    beta = 0.8
    J = 1.0

    # Multiple samples
    n_samples = 20
    mag_vs_level = [[] for _ in range(7)]

    for _ in range(n_samples):
        blocked = simulate_and_block(n_sites, beta, J, n_sweeps=500, n_blocks=6)

        for level, spins in enumerate(blocked):
            m = np.mean(spins)
            mag_vs_level[level].append(abs(m))

    # Plot mean and std
    levels = range(len(mag_vs_level))
    means = [np.mean(m) for m in mag_vs_level]
    stds = [np.std(m) for m in mag_vs_level]

    ax.errorbar(levels, means, yerr=stds, fmt='o-', lw=2, markersize=8,
                capsize=5)
    ax.set_xlabel('Block Level')
    ax.set_ylabel('|<m>|')
    ax.set_title(f'Magnetization Under Blocking\nβ = {beta}')
    ax.grid(True, alpha=0.3)

    # Plot 6: Summary
    ax = axes[1, 2]
    ax.axis('off')

    summary = """
Block-Spin Renormalization Group
================================

Idea:
  Coarse-grain the lattice by grouping
  spins into "blocks" and defining
  effective block variables.

1D Ising Block Spin:
  S'_i = sign(s_{2i} + s_{2i+1})

RG Transformation:
  Trace over short-distance DOF
  → Effective action for blocks

Exact 1D Result:
  K' = ½ ln(cosh 2K)
  where K = βJ

Fixed Points:
  K* = 0: High-T (disordered)
          Stable fixed point
  K* = ∞: T = 0 (ordered)
          Unstable in 1D

1D Ising Properties:
  • No phase transition (T_c = 0)
  • All K flow to K* = 0
  • Correlation length finite

2D Ising (Kadanoff):
  • Critical fixed point at K_c
  • ν = 1 (correlation length exponent)
  • Block spin → decimation

Scaling Near Fixed Point:
  K - K* ~ b^(1/ν) (K₀ - K*)
  ξ ~ |T - T_c|^(-ν)

Applications:
  • Critical phenomena
  • Universality classes
  • Effective field theories
"""

    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Experiment 217: Block-Spin Renormalization Group\n'
                 'Real-Space RG for Ising Model', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'exp217_block_spin_rg.png'),
                dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/exp217_block_spin_rg.png")


if __name__ == "__main__":
    main()
