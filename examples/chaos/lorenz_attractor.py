"""
Example demonstrating the Lorenz attractor.

This example shows chaotic dynamics in the famous Lorenz system,
demonstrating sensitive dependence on initial conditions (butterfly effect).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Using custom RK4 implementation (no external import needed)


def lorenz_system(state, t, sigma=10.0, rho=28.0, beta=8/3):
    """
    Lorenz system differential equations.

    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z

    Args:
        state: [x, y, z] state vector
        t: Time (not used, system is autonomous)
        sigma, rho, beta: Lorenz parameters

    Returns:
        Derivatives [dx/dt, dy/dt, dz/dt]
    """
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return np.array([dxdt, dydt, dzdt])


def simulate_lorenz(initial_state, t_max, dt, sigma=10.0, rho=28.0, beta=8/3):
    """Simulate the Lorenz system using RK4."""
    n_steps = int(t_max / dt)
    t = np.linspace(0, t_max, n_steps)

    state = np.zeros((n_steps, 3))
    state[0] = initial_state

    for i in range(1, n_steps):
        # RK4 step
        k1 = lorenz_system(state[i-1], t[i-1], sigma, rho, beta)
        k2 = lorenz_system(state[i-1] + dt/2 * k1, t[i-1] + dt/2, sigma, rho, beta)
        k3 = lorenz_system(state[i-1] + dt/2 * k2, t[i-1] + dt/2, sigma, rho, beta)
        k4 = lorenz_system(state[i-1] + dt * k3, t[i-1] + dt, sigma, rho, beta)
        state[i] = state[i-1] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    return t, state


def main():
    # Parameters
    sigma = 10.0
    rho = 28.0
    beta = 8/3
    t_max = 50.0
    dt = 0.01

    # Initial condition
    initial_state = np.array([1.0, 1.0, 1.0])

    # Simulate
    t, state = simulate_lorenz(initial_state, t_max, dt, sigma, rho, beta)
    x, y, z = state[:, 0], state[:, 1], state[:, 2]

    fig = plt.figure(figsize=(16, 12))

    # Plot 1: 3D attractor
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(x, y, z, lw=0.5, alpha=0.8)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Lorenz Attractor (3D)')

    # Plot 2: X-Z projection
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(x, z, lw=0.5, alpha=0.7)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.set_title('X-Z Projection')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Time series
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(t, x, 'b-', lw=0.5, label='X', alpha=0.8)
    ax3.plot(t, y, 'r-', lw=0.5, label='Y', alpha=0.8)
    ax3.plot(t, z, 'g-', lw=0.5, label='Z', alpha=0.8)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Value')
    ax3.set_title('Time Series')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Butterfly effect (sensitivity to initial conditions)
    ax4 = fig.add_subplot(2, 2, 4)

    # Two trajectories with slightly different initial conditions
    initial_1 = np.array([1.0, 1.0, 1.0])
    initial_2 = np.array([1.0 + 1e-10, 1.0, 1.0])  # Tiny perturbation

    t1, state1 = simulate_lorenz(initial_1, t_max, dt, sigma, rho, beta)
    t2, state2 = simulate_lorenz(initial_2, t_max, dt, sigma, rho, beta)

    # Calculate divergence
    divergence = np.sqrt(np.sum((state1 - state2)**2, axis=1))

    ax4.semilogy(t1, divergence, 'b-', lw=1.5)
    ax4.axhline(y=1e-10, color='r', linestyle='--', label='Initial separation')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Distance (log scale)')
    ax4.set_title('Butterfly Effect: Trajectory Divergence\n(Initial separation = 10⁻¹⁰)')
    ax4.grid(True, alpha=0.3, which='both')
    ax4.legend()

    # Add Lyapunov exponent estimate annotation
    # Rough estimate from exponential growth phase
    growth_region = (t1 > 5) & (t1 < 25) & (divergence > 1e-8) & (divergence < 10)
    if np.any(growth_region):
        from scipy.stats import linregress
        slope, _, _, _, _ = linregress(t1[growth_region], np.log(divergence[growth_region]))
        ax4.text(0.05, 0.95, f'λ ≈ {slope:.2f} (Lyapunov exponent)',
                transform=ax4.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'Lorenz System: σ = {sigma}, ρ = {rho}, β = {beta:.2f}\n'
                 'A paradigm of deterministic chaos',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'lorenz_attractor.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'lorenz_attractor.png')}")


if __name__ == "__main__":
    main()
