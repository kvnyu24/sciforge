"""
Example demonstrating Brownian motion (Wiener process).

This example shows random walk behavior of particles in a fluid,
demonstrating the statistical properties of Brownian motion.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.stochastic.processes import WienerProcess


def simulate_brownian_motion_2d(n_particles, n_steps, dt, D=1.0):
    """
    Simulate 2D Brownian motion for multiple particles.

    Args:
        n_particles: Number of particles to simulate
        n_steps: Number of time steps
        dt: Time step
        D: Diffusion coefficient

    Returns:
        Arrays of x and y positions over time
    """
    # Standard deviation of displacement per step
    sigma = np.sqrt(2 * D * dt)

    # Initialize positions at origin
    x = np.zeros((n_particles, n_steps))
    y = np.zeros((n_particles, n_steps))

    # Random walk
    for i in range(1, n_steps):
        x[:, i] = x[:, i-1] + np.random.normal(0, sigma, n_particles)
        y[:, i] = y[:, i-1] + np.random.normal(0, sigma, n_particles)

    return x, y


def main():
    # Parameters
    n_particles = 100
    n_steps = 1000
    dt = 0.01
    D = 1.0  # Diffusion coefficient

    # Run simulation
    x, y = simulate_brownian_motion_2d(n_particles, n_steps, dt, D)
    t = np.arange(n_steps) * dt

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Trajectories (first 10 particles)
    ax1 = axes[0, 0]
    for i in range(min(10, n_particles)):
        ax1.plot(x[i], y[i], lw=0.5, alpha=0.7)
    ax1.plot(0, 0, 'ko', markersize=10, label='Start')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('2D Brownian Motion Trajectories')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Plot 2: Final positions distribution
    ax2 = axes[0, 1]
    ax2.scatter(x[:, -1], y[:, -1], alpha=0.5, s=20)
    ax2.plot(0, 0, 'r+', markersize=15, mew=3, label='Start')

    # Add theoretical distribution circle (1 std dev)
    r_std = np.sqrt(2 * D * t[-1])  # Expected RMS displacement
    theta = np.linspace(0, 2*np.pi, 100)
    ax2.plot(r_std * np.cos(theta), r_std * np.sin(theta), 'r--', lw=2,
            label=f'1σ (r = {r_std:.2f})')
    ax2.plot(2*r_std * np.cos(theta), 2*r_std * np.sin(theta), 'r:', lw=1,
            label=f'2σ (r = {2*r_std:.2f})')

    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title(f'Final Positions (t = {t[-1]:.1f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    # Plot 3: Mean squared displacement
    ax3 = axes[0, 2]

    # Calculate MSD
    r_squared = x**2 + y**2
    msd = np.mean(r_squared, axis=0)
    msd_std = np.std(r_squared, axis=0)

    ax3.plot(t, msd, 'b-', lw=2, label='Simulated MSD')
    ax3.fill_between(t, msd - msd_std, msd + msd_std, alpha=0.3)
    ax3.plot(t, 4 * D * t, 'r--', lw=2, label=f'Theory: <r²> = 4Dt')

    ax3.set_xlabel('Time')
    ax3.set_ylabel('Mean Squared Displacement')
    ax3.set_title('MSD vs Time (Diffusion Law)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Displacement histogram at different times
    ax4 = axes[1, 0]

    time_indices = [100, 300, 500, 999]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(time_indices)))

    for idx, color in zip(time_indices, colors):
        r = np.sqrt(x[:, idx]**2 + y[:, idx]**2)
        ax4.hist(r, bins=20, alpha=0.5, color=color, label=f't = {t[idx]:.2f}', density=True)

        # Theoretical Rayleigh distribution for 2D
        r_theory = np.linspace(0, np.max(r), 100)
        sigma_r = np.sqrt(2 * D * t[idx])
        rayleigh = r_theory / sigma_r**2 * np.exp(-r_theory**2 / (2 * sigma_r**2))
        ax4.plot(r_theory, rayleigh, color=color, lw=2, linestyle='--')

    ax4.set_xlabel('Distance from origin')
    ax4.set_ylabel('Probability density')
    ax4.set_title('Displacement Distribution (Rayleigh)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Single coordinate time series
    ax5 = axes[1, 1]

    for i in range(5):
        ax5.plot(t, x[i], lw=1, alpha=0.7)

    ax5.set_xlabel('Time')
    ax5.set_ylabel('x position')
    ax5.set_title('X-coordinate vs Time (Wiener Process)')
    ax5.grid(True, alpha=0.3)

    # Plot 6: Diffusion coefficient estimation
    ax6 = axes[1, 2]

    # Estimate D from MSD slope
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(t[10:], msd[10:])
    D_estimated = slope / 4

    ax6.plot(t, msd, 'b.', alpha=0.5, label='Data')
    ax6.plot(t, slope * t + intercept, 'r-', lw=2,
            label=f'Linear fit: D = {D_estimated:.3f}')
    ax6.plot(t, 4 * D * t, 'g--', lw=2, label=f'True: D = {D:.3f}')

    ax6.set_xlabel('Time')
    ax6.set_ylabel('Mean Squared Displacement')
    ax6.set_title(f'Diffusion Coefficient Estimation\nD_est = {D_estimated:.3f}, D_true = {D:.3f}')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.suptitle(f'Brownian Motion (N={n_particles} particles, D={D}, dt={dt})',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'brownian_motion.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'brownian_motion.png')}")


if __name__ == "__main__":
    main()
