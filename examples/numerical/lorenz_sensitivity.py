"""
Experiment 6: Chaotic ODE sensitivity - Lorenz system divergence.

Demonstrates sensitive dependence on initial conditions in
the Lorenz system, measuring divergence rate (Lyapunov exponent).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def lorenz(state, sigma=10.0, rho=28.0, beta=8.0/3.0):
    """Lorenz system equations."""
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return np.array([dxdt, dydt, dzdt])


def rk4_step(state, dt, f):
    """RK4 integration step."""
    k1 = f(state)
    k2 = f(state + dt/2 * k1)
    k3 = f(state + dt/2 * k2)
    k4 = f(state + dt * k3)
    return state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6


def integrate_lorenz(state0, t_final, dt, sigma=10.0, rho=28.0, beta=8.0/3.0):
    """Integrate Lorenz system."""
    f = lambda s: lorenz(s, sigma, rho, beta)

    n_steps = int(t_final / dt)
    states = np.zeros((n_steps + 1, 3))
    states[0] = state0

    for i in range(n_steps):
        states[i+1] = rk4_step(states[i], dt, f)

    times = np.arange(n_steps + 1) * dt
    return times, states


def estimate_lyapunov(state0, t_final, dt, delta=1e-8, sigma=10.0, rho=28.0, beta=8.0/3.0):
    """
    Estimate largest Lyapunov exponent by tracking
    separation of nearby trajectories.
    """
    f = lambda s: lorenz(s, sigma, rho, beta)

    n_steps = int(t_final / dt)

    # Reference trajectory
    state_ref = np.array(state0, dtype=float)

    # Perturbed trajectory
    perturbation = delta * np.array([1, 0, 0])
    state_pert = state_ref + perturbation

    lyapunov_sum = 0.0
    separations = []
    times = []

    renorm_interval = 10  # Renormalize every N steps

    for i in range(n_steps):
        state_ref = rk4_step(state_ref, dt, f)
        state_pert = rk4_step(state_pert, dt, f)

        if (i + 1) % renorm_interval == 0:
            d = np.linalg.norm(state_pert - state_ref)
            separations.append(d)
            times.append((i + 1) * dt)

            if d > 0:
                lyapunov_sum += np.log(d / delta)

                # Renormalize
                state_pert = state_ref + delta * (state_pert - state_ref) / d

    lyapunov = lyapunov_sum / (n_steps * dt)
    return lyapunov, np.array(times), np.array(separations)


def main():
    # Lorenz parameters
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0

    # Initial conditions
    state0 = np.array([1.0, 1.0, 1.0])

    # Simulation parameters
    t_final = 50.0
    dt = 0.01

    # Different perturbation sizes
    perturbations = [1e-10, 1e-8, 1e-6, 1e-4]

    fig = plt.figure(figsize=(15, 12))

    # Plot 1: 3D attractor
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    t, states = integrate_lorenz(state0, t_final, dt, sigma, rho, beta)
    ax1.plot(states[:, 0], states[:, 1], states[:, 2], 'b-', lw=0.3, alpha=0.7)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Lorenz Attractor')

    # Plot 2: Trajectory divergence for different perturbations
    ax2 = fig.add_subplot(2, 2, 2)
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(perturbations)))

    for eps, color in zip(perturbations, colors):
        # Reference trajectory
        t_ref, states_ref = integrate_lorenz(state0, t_final, dt, sigma, rho, beta)

        # Perturbed trajectory
        state0_pert = state0 + eps * np.array([1, 0, 0])
        t_pert, states_pert = integrate_lorenz(state0_pert, t_final, dt, sigma, rho, beta)

        # Calculate separation
        separation = np.linalg.norm(states_pert - states_ref, axis=1)

        ax2.semilogy(t_ref, separation, '-', color=color, lw=1.5,
                     label=f'δ₀ = {eps:.0e}')

    # Theoretical exponential growth
    lyap_est = 0.9  # Approximate largest Lyapunov exponent
    t_theory = np.linspace(0, 30, 100)
    ax2.semilogy(t_theory, 1e-10 * np.exp(lyap_est * t_theory), 'k--',
                 alpha=0.5, label=f'exp(λt), λ≈{lyap_est}')

    ax2.axhline(1.0, color='gray', linestyle=':', alpha=0.5, label='|δ|=1')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Separation |δ(t)|')
    ax2.set_title('Exponential Divergence of Nearby Trajectories')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, t_final)
    ax2.set_ylim(1e-12, 1e3)

    # Plot 3: Time series comparison
    ax3 = fig.add_subplot(2, 2, 3)

    eps_demo = 1e-8
    state0_pert = state0 + eps_demo * np.array([1, 0, 0])
    t_pert, states_pert = integrate_lorenz(state0_pert, t_final, dt, sigma, rho, beta)

    ax3.plot(t, states[:, 0], 'b-', lw=1, label='Reference')
    ax3.plot(t_pert, states_pert[:, 0], 'r--', lw=1, alpha=0.7, label=f'Perturbed (δ={eps_demo:.0e})')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('X coordinate')
    ax3.set_title('X(t) Comparison - Identical then Divergent')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Mark approximate divergence time
    separation = np.linalg.norm(states_pert - states, axis=1)
    diverge_idx = np.where(separation > 1.0)[0]
    if len(diverge_idx) > 0:
        t_diverge = t[diverge_idx[0]]
        ax3.axvline(t_diverge, color='gray', linestyle=':', alpha=0.7)
        ax3.text(t_diverge + 0.5, ax3.get_ylim()[1] * 0.9,
                 f't_diverge ≈ {t_diverge:.1f}', fontsize=9)

    # Plot 4: Lyapunov exponent estimation
    ax4 = fig.add_subplot(2, 2, 4)

    # Estimate Lyapunov exponent over time
    lyap, t_lyap, seps = estimate_lyapunov(state0, t_final, dt, delta=1e-9)

    # Running average of Lyapunov exponent
    n_window = 50
    if len(seps) > n_window:
        log_seps = np.log(seps / 1e-9)
        lyap_running = np.zeros(len(t_lyap) - n_window)
        for i in range(len(lyap_running)):
            lyap_running[i] = (log_seps[i + n_window] - log_seps[i]) / (t_lyap[i + n_window] - t_lyap[i])

        ax4.plot(t_lyap[n_window//2:-n_window//2], lyap_running, 'b-', lw=1)

    ax4.axhline(lyap, color='r', linestyle='--', lw=2, label=f'λ ≈ {lyap:.3f}')
    ax4.axhline(0.906, color='gray', linestyle=':', alpha=0.7, label='Literature: λ₁ ≈ 0.906')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Local Lyapunov exponent')
    ax4.set_title('Lyapunov Exponent Estimation')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Lorenz System: Sensitive Dependence on Initial Conditions\n' +
                 f'σ={sigma}, ρ={rho}, β={beta:.2f}', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'lorenz_sensitivity.png'), dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/lorenz_sensitivity.png")

    print(f"\nEstimated largest Lyapunov exponent: λ ≈ {lyap:.4f}")
    print(f"Literature value: λ₁ ≈ 0.906")


if __name__ == "__main__":
    main()
