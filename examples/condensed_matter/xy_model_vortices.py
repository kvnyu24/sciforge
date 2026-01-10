"""
Experiment 233: XY Model Vortex Unbinding

Demonstrates the 2D XY model and the Berezinskii-Kosterlitz-Thouless (BKT)
transition, where vortex-antivortex pairs unbind at the critical temperature.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def initialize_angles(L, initial='random'):
    """
    Initialize spin angles on L x L lattice.

    XY model: spins are planar, characterized by angle theta.

    Args:
        L: Lattice size
        initial: 'random', 'ordered', or 'vortex'

    Returns:
        theta: L x L array of angles [0, 2*pi)
    """
    if initial == 'ordered':
        return np.zeros((L, L))
    elif initial == 'vortex':
        # Create a single vortex at center
        theta = np.zeros((L, L))
        cx, cy = L // 2, L // 2
        for i in range(L):
            for j in range(L):
                theta[i, j] = np.arctan2(j - cy, i - cx)
        return theta
    else:
        return 2 * np.pi * np.random.random((L, L))


def compute_energy(theta, J=1.0):
    """
    Compute total energy of XY model.

    H = -J * sum_{<i,j>} cos(theta_i - theta_j)

    Args:
        theta: L x L angle array
        J: Coupling constant

    Returns:
        Total energy
    """
    L = theta.shape[0]

    # Use numpy roll for efficient neighbor access
    dtheta_x = theta - np.roll(theta, -1, axis=0)
    dtheta_y = theta - np.roll(theta, -1, axis=1)

    energy = -J * (np.sum(np.cos(dtheta_x)) + np.sum(np.cos(dtheta_y)))
    return energy


def compute_magnetization(theta):
    """
    Compute magnetization (average spin vector).

    M = (1/N) * sum_i (cos(theta_i), sin(theta_i))

    Returns magnitude of magnetization.
    """
    L = theta.shape[0]
    mx = np.mean(np.cos(theta))
    my = np.mean(np.sin(theta))
    return np.sqrt(mx**2 + my**2)


def compute_vorticity(theta):
    """
    Compute vorticity (topological charge) at each plaquette.

    q = (1/2pi) * sum over plaquette of d(theta)

    Returns:
        vorticity: (L-1) x (L-1) array of topological charges
    """
    L = theta.shape[0]
    vorticity = np.zeros((L, L))

    for i in range(L):
        for j in range(L):
            # Plaquette corners (with periodic BC)
            theta1 = theta[i, j]
            theta2 = theta[(i+1) % L, j]
            theta3 = theta[(i+1) % L, (j+1) % L]
            theta4 = theta[i, (j+1) % L]

            # Sum of angle differences around plaquette
            d1 = wrap_angle(theta2 - theta1)
            d2 = wrap_angle(theta3 - theta2)
            d3 = wrap_angle(theta4 - theta3)
            d4 = wrap_angle(theta1 - theta4)

            vorticity[i, j] = (d1 + d2 + d3 + d4) / (2 * np.pi)

    return vorticity


def wrap_angle(angle):
    """Wrap angle to [-pi, pi)."""
    return ((angle + np.pi) % (2 * np.pi)) - np.pi


def metropolis_step(theta, T, J=1.0, delta_max=0.5):
    """
    Perform one Metropolis Monte Carlo sweep.

    Args:
        theta: Current angle configuration
        T: Temperature
        J: Coupling constant
        delta_max: Maximum angle change

    Returns:
        Updated theta, acceptance rate
    """
    L = theta.shape[0]
    beta = 1.0 / T if T > 0 else np.inf
    accepted = 0

    for _ in range(L * L):
        i = np.random.randint(L)
        j = np.random.randint(L)

        # Propose new angle
        delta = delta_max * (2 * np.random.random() - 1)
        new_theta = (theta[i, j] + delta) % (2 * np.pi)

        # Calculate energy change
        neighbors = [
            theta[(i+1) % L, j],
            theta[(i-1) % L, j],
            theta[i, (j+1) % L],
            theta[i, (j-1) % L]
        ]

        old_energy = -J * sum(np.cos(theta[i, j] - n) for n in neighbors)
        new_energy = -J * sum(np.cos(new_theta - n) for n in neighbors)
        delta_E = new_energy - old_energy

        if delta_E <= 0 or np.random.random() < np.exp(-beta * delta_E):
            theta[i, j] = new_theta
            accepted += 1

    return theta, accepted / (L * L)


def helicity_modulus(theta, T, J=1.0):
    """
    Compute helicity modulus (superfluid stiffness).

    This is the key order parameter for the BKT transition.
    """
    L = theta.shape[0]

    # <cos(theta_i - theta_j)> along x
    dtheta_x = theta - np.roll(theta, -1, axis=0)
    gamma_x = np.mean(np.cos(dtheta_x))

    # <sin(theta_i - theta_j)>^2 along x
    sin_term = np.mean(np.sin(dtheta_x))**2

    rho_s = J * (gamma_x - J * L * L / T * sin_term)
    return rho_s


def main():
    """Main simulation and visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Simulation parameters
    L = 32        # Lattice size
    J = 1.0       # Coupling constant
    n_equilibrate = 1000
    n_sample = 500
    n_skip = 5

    # BKT transition temperature: T_BKT ~ 0.89 J
    T_BKT = 0.893

    # Temperature range
    T_range = np.linspace(0.3, 1.5, 12)

    print(f"Running XY model Monte Carlo on {L}x{L} lattice...")

    # Calculate observables vs temperature
    M_mean = []
    rho_s_mean = []
    n_vortex = []

    for T in T_range:
        print(f"  T = {T:.2f}...", end=' ')

        theta = initialize_angles(L, 'random')

        # Equilibration
        for _ in range(n_equilibrate):
            theta, _ = metropolis_step(theta, T, J)

        # Measurement
        M_samples = []
        rho_samples = []
        vortex_samples = []

        for _ in range(n_sample):
            for _ in range(n_skip):
                theta, _ = metropolis_step(theta, T, J)

            M_samples.append(compute_magnetization(theta))
            rho_samples.append(helicity_modulus(theta, T, J))

            vorticity = compute_vorticity(theta)
            n_v = np.sum(np.abs(vorticity) > 0.5)
            vortex_samples.append(n_v)

        M_mean.append(np.mean(M_samples))
        rho_s_mean.append(np.mean(rho_samples))
        n_vortex.append(np.mean(vortex_samples))

        print(f"M = {M_mean[-1]:.3f}, rho_s = {rho_s_mean[-1]:.3f}, n_v = {n_vortex[-1]:.1f}")

    M_mean = np.array(M_mean)
    rho_s_mean = np.array(rho_s_mean)
    n_vortex = np.array(n_vortex)

    # Plot 1: Magnetization
    ax1 = axes[0, 0]

    ax1.plot(T_range, M_mean, 'bo-', lw=2, markersize=8)
    ax1.axvline(x=T_BKT, color='red', linestyle='--', alpha=0.7,
               label=f'T_BKT ~ {T_BKT:.2f}')

    ax1.set_xlabel('Temperature (J/kB)')
    ax1.set_ylabel('Magnetization |M|')
    ax1.set_title('Magnetization vs Temperature')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Helicity modulus (superfluid stiffness)
    ax2 = axes[0, 1]

    ax2.plot(T_range, rho_s_mean, 'go-', lw=2, markersize=8)
    ax2.axvline(x=T_BKT, color='red', linestyle='--', alpha=0.7)

    # Universal jump prediction: rho_s(T_BKT^-) = 2T_BKT/pi
    ax2.axhline(y=2*T_BKT/np.pi, color='gray', linestyle=':',
               label=r'Universal jump: $\rho_s = 2T/\pi$')

    ax2.set_xlabel('Temperature (J/kB)')
    ax2.set_ylabel('Helicity modulus rho_s')
    ax2.set_title('Superfluid Stiffness (Order Parameter)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Vortex density
    ax3 = axes[1, 0]

    ax3.plot(T_range, n_vortex / (L*L), 'mo-', lw=2, markersize=8)
    ax3.axvline(x=T_BKT, color='red', linestyle='--', alpha=0.7,
               label=f'T_BKT ~ {T_BKT:.2f}')

    ax3.set_xlabel('Temperature (J/kB)')
    ax3.set_ylabel('Vortex density (n_v / N)')
    ax3.set_title('Free Vortex Density')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax3.text(0.5, 0.95, 'Vortex-antivortex unbinding at T_BKT',
             transform=ax3.transAxes, fontsize=10, va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 4: Spin configuration with vortices
    ax4 = axes[1, 1]

    # Generate a configuration at T ~ T_BKT
    theta = initialize_angles(L, 'random')
    T = 0.9

    for _ in range(n_equilibrate):
        theta, _ = metropolis_step(theta, T, J)

    # Plot spin arrows
    X, Y = np.meshgrid(range(L), range(L))
    step = 2  # Show every other spin for clarity
    ax4.quiver(X[::step, ::step], Y[::step, ::step],
              np.cos(theta[::step, ::step]), np.sin(theta[::step, ::step]),
              color='blue', alpha=0.6)

    # Mark vortices
    vorticity = compute_vorticity(theta)
    vortex_pos = np.where(vorticity > 0.5)
    antivortex_pos = np.where(vorticity < -0.5)

    ax4.scatter(vortex_pos[1] + 0.5, vortex_pos[0] + 0.5,
               s=100, c='red', marker='^', label='Vortex (+)', zorder=5)
    ax4.scatter(antivortex_pos[1] + 0.5, antivortex_pos[0] + 0.5,
               s=100, c='green', marker='v', label='Antivortex (-)', zorder=5)

    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_title(f'Spin Configuration at T = {T} J/kB')
    ax4.legend()
    ax4.set_xlim(-0.5, L-0.5)
    ax4.set_ylim(-0.5, L-0.5)
    ax4.set_aspect('equal')

    plt.suptitle('2D XY Model: Berezinskii-Kosterlitz-Thouless Transition\n'
                 r'$H = -J \sum_{\langle i,j \rangle} \cos(\theta_i - \theta_j)$',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'xy_model_vortices.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'xy_model_vortices.png')}")


if __name__ == "__main__":
    main()
