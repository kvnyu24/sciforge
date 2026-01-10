"""
Experiment 23: Random walk converges to diffusion constant measurement.

Shows that random walks lead to diffusive behavior and how to
extract the diffusion coefficient from MSD.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def random_walk_1d(n_steps, n_walkers, step_size=1.0):
    """Simulate 1D random walks."""
    steps = np.random.choice([-1, 1], size=(n_walkers, n_steps)) * step_size
    positions = np.cumsum(steps, axis=1)
    # Add initial position (0)
    positions = np.hstack([np.zeros((n_walkers, 1)), positions])
    return positions


def random_walk_2d(n_steps, n_walkers, step_size=1.0):
    """Simulate 2D random walks."""
    angles = np.random.uniform(0, 2*np.pi, size=(n_walkers, n_steps))
    dx = step_size * np.cos(angles)
    dy = step_size * np.sin(angles)

    x = np.cumsum(dx, axis=1)
    y = np.cumsum(dy, axis=1)

    x = np.hstack([np.zeros((n_walkers, 1)), x])
    y = np.hstack([np.zeros((n_walkers, 1)), y])

    return x, y


def compute_msd(positions, max_lag=None):
    """
    Compute mean squared displacement.

    MSD(τ) = <(x(t+τ) - x(t))²>
    """
    n_walkers, n_steps = positions.shape
    if max_lag is None:
        max_lag = n_steps // 2

    msd = np.zeros(max_lag)
    msd_err = np.zeros(max_lag)

    for lag in range(1, max_lag):
        displacements = positions[:, lag:] - positions[:, :-lag]
        squared_disp = displacements**2
        msd[lag] = np.mean(squared_disp)
        msd_err[lag] = np.std(np.mean(squared_disp, axis=1)) / np.sqrt(n_walkers)

    return msd, msd_err


def compute_msd_2d(x, y, max_lag=None):
    """Compute MSD for 2D random walk."""
    n_walkers, n_steps = x.shape
    if max_lag is None:
        max_lag = n_steps // 2

    msd = np.zeros(max_lag)

    for lag in range(1, max_lag):
        dx = x[:, lag:] - x[:, :-lag]
        dy = y[:, lag:] - y[:, :-lag]
        r2 = dx**2 + dy**2
        msd[lag] = np.mean(r2)

    return msd


def main():
    np.random.seed(42)

    # Parameters
    n_steps = 1000
    n_walkers = 1000
    step_size = 1.0
    dt = 1.0  # Time step

    # 1D random walk
    positions_1d = random_walk_1d(n_steps, n_walkers, step_size)

    # 2D random walk
    x_2d, y_2d = random_walk_2d(n_steps, n_walkers, step_size)

    # Compute MSD
    msd_1d, msd_1d_err = compute_msd(positions_1d, max_lag=500)
    msd_2d = compute_msd_2d(x_2d, y_2d, max_lag=500)

    # Theoretical MSD
    # 1D: MSD = 2*D*t where D = step_size^2 / (2*dt) for discrete walk
    D_1d_theory = step_size**2 / (2 * dt)
    # 2D: MSD = 4*D*t where D = step_size^2 / (4*dt)
    D_2d_theory = step_size**2 / (4 * dt)

    t = np.arange(len(msd_1d)) * dt

    # Fit to extract D
    # Linear fit: MSD = 2*d*D*t where d is dimension
    fit_range = slice(10, 400)  # Avoid early transients and late noise

    # 1D fit
    coeffs_1d = np.polyfit(t[fit_range], msd_1d[fit_range], 1)
    D_1d_fit = coeffs_1d[0] / 2

    # 2D fit
    coeffs_2d = np.polyfit(t[fit_range], msd_2d[fit_range], 1)
    D_2d_fit = coeffs_2d[0] / 4

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Sample 1D trajectories
    ax = axes[0, 0]
    n_show = 20
    time = np.arange(n_steps + 1) * dt
    for i in range(n_show):
        ax.plot(time, positions_1d[i], '-', lw=0.5, alpha=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Position')
    ax.set_title('1D Random Walk Trajectories')
    ax.grid(True, alpha=0.3)

    # Plot 2: Position distribution evolution
    ax = axes[0, 1]
    times_to_show = [100, 300, 500, 800]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(times_to_show)))

    for t_idx, color in zip(times_to_show, colors):
        positions = positions_1d[:, t_idx]
        ax.hist(positions, bins=30, density=True, alpha=0.5, color=color,
                label=f't = {t_idx}')

        # Theoretical Gaussian
        sigma = np.sqrt(2 * D_1d_theory * t_idx * dt)
        x = np.linspace(-3*sigma, 3*sigma, 100)
        ax.plot(x, np.exp(-x**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi)),
                '-', color=color, lw=2)

    ax.set_xlabel('Position')
    ax.set_ylabel('Probability Density')
    ax.set_title('Position Distribution (solid: theory)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 3: 1D MSD
    ax = axes[0, 2]
    t_msd = np.arange(len(msd_1d)) * dt

    ax.plot(t_msd, msd_1d, 'b-', lw=2, label='Measured')
    ax.plot(t_msd, 2 * D_1d_theory * t_msd, 'r--', lw=2, label=f'Theory: D={D_1d_theory:.3f}')
    ax.plot(t_msd, 2 * D_1d_fit * t_msd, 'g:', lw=2, label=f'Fit: D={D_1d_fit:.3f}')

    ax.set_xlabel('Time')
    ax.set_ylabel('MSD')
    ax.set_title('1D Mean Squared Displacement')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Sample 2D trajectories
    ax = axes[1, 0]
    for i in range(10):
        ax.plot(x_2d[i], y_2d[i], '-', lw=0.5, alpha=0.7)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('2D Random Walk Trajectories')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    # Plot 5: 2D MSD
    ax = axes[1, 1]

    ax.plot(t_msd, msd_2d, 'b-', lw=2, label='Measured')
    ax.plot(t_msd, 4 * D_2d_theory * t_msd, 'r--', lw=2, label=f'Theory: D={D_2d_theory:.3f}')
    ax.plot(t_msd, 4 * D_2d_fit * t_msd, 'g:', lw=2, label=f'Fit: D={D_2d_fit:.3f}')

    ax.set_xlabel('Time')
    ax.set_ylabel('MSD')
    ax.set_title('2D Mean Squared Displacement')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Summary
    ax = axes[1, 2]
    ax.axis('off')

    summary = f"""Random Walk to Diffusion
========================
Parameters:
  N steps = {n_steps}
  N walkers = {n_walkers}
  Step size = {step_size}
  Time step = {dt}

Einstein Relation:
  MSD = 2dDt (d = dimension)

1D Results:
  D_theory = {D_1d_theory:.4f}
  D_fit = {D_1d_fit:.4f}
  Error = {abs(D_1d_fit - D_1d_theory)/D_1d_theory*100:.2f}%

2D Results:
  D_theory = {D_2d_theory:.4f}
  D_fit = {D_2d_fit:.4f}
  Error = {abs(D_2d_fit - D_2d_theory)/D_2d_theory*100:.2f}%

Key Insights:
• Random walk → diffusion
• <x²> grows linearly with t
• Position distribution → Gaussian
• D relates to step size & time:
  D = Δx² / (2d·Δt)

Applications:
• Brownian motion
• Polymer physics
• Transport phenomena"""

    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.suptitle('Random Walk and Diffusion: MSD Analysis',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'random_walk_diffusion.png'), dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/random_walk_diffusion.png")


if __name__ == "__main__":
    main()
