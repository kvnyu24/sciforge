"""
Experiment 234: Hysteresis Domain Model

Demonstrates magnetic hysteresis using the Stoner-Wohlfarth model and
domain wall dynamics, showing how magnetic domains lead to the
characteristic S-shaped hysteresis loop.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def stoner_wohlfarth_energy(theta, h, psi=0):
    """
    Stoner-Wohlfarth energy for single-domain particle.

    E = -K*cos^2(theta) - M_s*H*cos(theta - psi)

    Normalized: e = -cos^2(theta) - h*cos(theta - psi)

    Args:
        theta: Magnetization angle
        h: Reduced field H/H_k where H_k = 2K/M_s
        psi: Field angle relative to easy axis

    Returns:
        Normalized energy
    """
    return -np.cos(theta)**2 - h * np.cos(theta - psi)


def find_equilibrium(h, psi=0, initial_theta=0, tol=1e-8, max_iter=1000):
    """
    Find equilibrium magnetization angle for given field.

    Uses gradient descent to find local minimum.

    Args:
        h: Reduced field
        psi: Field angle
        initial_theta: Starting angle
        tol: Convergence tolerance
        max_iter: Maximum iterations

    Returns:
        Equilibrium angle theta
    """
    theta = initial_theta

    for _ in range(max_iter):
        # Gradient of energy
        dE = 2 * np.cos(theta) * np.sin(theta) + h * np.sin(theta - psi)

        # Newton step
        d2E = 2 * np.cos(2*theta) + h * np.cos(theta - psi)

        if abs(d2E) > 1e-10:
            step = -dE / d2E
        else:
            step = -0.1 * np.sign(dE)

        theta_new = theta + 0.5 * step  # Damped update

        if abs(theta_new - theta) < tol:
            break
        theta = theta_new

    return theta


def stoner_wohlfarth_loop(h_max=2.0, n_points=200, psi=0):
    """
    Calculate Stoner-Wohlfarth hysteresis loop.

    Args:
        h_max: Maximum reduced field
        n_points: Number of field points
        psi: Field angle

    Returns:
        h_values, m_values: Field and magnetization arrays
    """
    # Forward sweep
    h_forward = np.linspace(-h_max, h_max, n_points)
    m_forward = np.zeros(n_points)

    theta = np.pi if psi < np.pi/4 else 0  # Start near saturation

    for i, h in enumerate(h_forward):
        theta = find_equilibrium(h, psi, theta)
        m_forward[i] = np.cos(theta)

    # Backward sweep
    h_backward = np.linspace(h_max, -h_max, n_points)
    m_backward = np.zeros(n_points)

    theta = 0 if psi < np.pi/4 else np.pi

    for i, h in enumerate(h_backward):
        theta = find_equilibrium(h, psi, theta)
        m_backward[i] = np.cos(theta)

    return (np.concatenate([h_forward, h_backward]),
            np.concatenate([m_forward, m_backward]))


def preisach_model(H, H_c_mean, H_c_std, M_s=1.0):
    """
    Simplified Preisach model for polycrystalline hysteresis.

    Assumes distribution of switching fields.

    Args:
        H: Applied field array (must start from saturation)
        H_c_mean: Mean coercive field
        H_c_std: Standard deviation of coercive fields
        M_s: Saturation magnetization

    Returns:
        M: Magnetization array
    """
    M = np.zeros_like(H)
    n_hysterons = 100

    # Distribution of switching fields
    H_c_up = np.random.normal(H_c_mean, H_c_std, n_hysterons)
    H_c_down = np.random.normal(-H_c_mean, H_c_std, n_hysterons)

    # State of each hysteron (+1 or -1)
    states = np.ones(n_hysterons)  # Start saturated

    for i, h in enumerate(H):
        # Update states based on field
        states = np.where(h > H_c_up, 1, states)
        states = np.where(h < H_c_down, -1, states)

        M[i] = M_s * np.mean(states)

    return M


def domain_wall_dynamics(t, H_applied, M0=-1.0, H_c=0.5, tau=1.0, alpha=0.1):
    """
    Domain wall propagation model.

    dM/dt = (M_eq - M) / tau

    where M_eq depends on field history.

    Args:
        t: Time array
        H_applied: Applied field vs time
        M0: Initial magnetization
        H_c: Coercive field
        tau: Relaxation time
        alpha: Switching sharpness

    Returns:
        M: Magnetization vs time
    """
    dt = t[1] - t[0]
    M = np.zeros_like(t)
    M[0] = M0

    for i in range(1, len(t)):
        H = H_applied[i]

        # Equilibrium magnetization (tanh switching)
        M_eq = np.tanh(alpha * (H - H_c * np.sign(M[i-1])))

        # Relaxation dynamics
        dM = (M_eq - M[i-1]) / tau
        M[i] = M[i-1] + dM * dt

    return M


def main():
    """Main simulation and visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Stoner-Wohlfarth model for different angles
    ax1 = axes[0, 0]

    psi_values = [0, np.pi/6, np.pi/4, np.pi/3]
    labels = ['0', 'pi/6', 'pi/4', 'pi/3']
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(psi_values)))

    for psi, label, color in zip(psi_values, labels, colors):
        h, m = stoner_wohlfarth_loop(h_max=2.0, n_points=200, psi=psi)

        # Split into two halves for proper plotting
        n = len(h) // 2
        ax1.plot(h[:n], m[:n], color=color, lw=2, label=f'psi = {label}')
        ax1.plot(h[n:], m[n:], color=color, lw=2)

    ax1.set_xlabel('Reduced field h = H/H_k')
    ax1.set_ylabel('Reduced magnetization m = M/M_s')
    ax1.set_title('Stoner-Wohlfarth Model: Effect of Field Angle')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-1.2, 1.2)
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

    # Plot 2: Preisach model (polycrystalline)
    ax2 = axes[0, 1]

    H_max = 2.0
    n_points = 500
    H_c_mean = 0.5

    # Forward and backward sweep
    H_forward = np.linspace(-H_max, H_max, n_points)
    H_backward = np.linspace(H_max, -H_max, n_points)
    H = np.concatenate([H_forward, H_backward])

    for H_c_std, color, label in [(0.1, 'blue', 'Sharp'), (0.3, 'green', 'Moderate'), (0.5, 'red', 'Broad')]:
        np.random.seed(42)  # Reproducibility
        M = preisach_model(H, H_c_mean, H_c_std)
        ax2.plot(H[:n_points], M[:n_points], color=color, lw=2, label=f'{label} distribution')
        ax2.plot(H[n_points:], M[n_points:], color=color, lw=2)

    ax2.set_xlabel('Applied field H')
    ax2.set_ylabel('Magnetization M/M_s')
    ax2.set_title('Preisach Model: Distribution of Switching Fields')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

    # Plot 3: Dynamic hysteresis (rate-dependent)
    ax3 = axes[1, 0]

    frequencies = [0.1, 0.5, 1.0, 2.0]
    colors = plt.cm.plasma(np.linspace(0, 0.8, len(frequencies)))

    for freq, color in zip(frequencies, colors):
        t = np.linspace(0, 10/freq, 1000)
        H_applied = 2.0 * np.sin(2 * np.pi * freq * t)
        M = domain_wall_dynamics(t, H_applied, M0=-1.0, H_c=0.5, tau=0.5)

        # Plot last few cycles
        n_skip = len(t) // 2
        ax3.plot(H_applied[n_skip:], M[n_skip:], color=color, lw=1.5, alpha=0.8,
                label=f'f = {freq} Hz')

    ax3.set_xlabel('Applied field H')
    ax3.set_ylabel('Magnetization M')
    ax3.set_title('Rate-Dependent Hysteresis')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Energy landscape and switching
    ax4 = axes[1, 1]

    theta = np.linspace(0, 2*np.pi, 200)

    h_values = [-0.3, 0, 0.3, 0.6, 1.0]
    colors = plt.cm.RdYlBu(np.linspace(0, 1, len(h_values)))

    for h, color in zip(h_values, colors):
        E = stoner_wohlfarth_energy(theta, h, psi=0)
        ax4.plot(theta * 180 / np.pi, E, color=color, lw=2, label=f'h = {h}')

    ax4.set_xlabel('Magnetization angle theta (degrees)')
    ax4.set_ylabel('Energy (normalized)')
    ax4.set_title('Stoner-Wohlfarth Energy Landscape')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 360)

    # Mark minima
    ax4.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax4.axvline(x=180, color='gray', linestyle=':', alpha=0.5)

    plt.suptitle('Magnetic Hysteresis and Domain Models\n'
                 'Single-domain and polycrystalline behavior',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'hysteresis_domain_model.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'hysteresis_domain_model.png')}")


if __name__ == "__main__":
    main()
