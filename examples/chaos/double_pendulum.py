"""
Example demonstrating double pendulum chaos.

This example shows how a simple mechanical system (double pendulum)
exhibits chaotic behavior and sensitive dependence on initial conditions.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def double_pendulum_derivatives(state, t, L1, L2, m1, m2, g=9.81):
    """
    Calculate derivatives for double pendulum system.

    State: [theta1, omega1, theta2, omega2]

    Returns: [dtheta1/dt, domega1/dt, dtheta2/dt, domega2/dt]
    """
    theta1, omega1, theta2, omega2 = state
    delta = theta2 - theta1

    # Denominators
    den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta)**2
    den2 = (L2 / L1) * den1

    # Derivatives
    dtheta1 = omega1
    dtheta2 = omega2

    domega1 = ((m2 * L1 * omega1**2 * np.sin(delta) * np.cos(delta) +
                m2 * g * np.sin(theta2) * np.cos(delta) +
                m2 * L2 * omega2**2 * np.sin(delta) -
                (m1 + m2) * g * np.sin(theta1)) / den1)

    domega2 = ((-m2 * L2 * omega2**2 * np.sin(delta) * np.cos(delta) +
                (m1 + m2) * g * np.sin(theta1) * np.cos(delta) -
                (m1 + m2) * L1 * omega1**2 * np.sin(delta) -
                (m1 + m2) * g * np.sin(theta2)) / den2)

    return np.array([dtheta1, domega1, dtheta2, domega2])


def simulate_double_pendulum(initial_state, L1, L2, m1, m2, t_max, dt, g=9.81):
    """Simulate double pendulum using RK4."""
    n_steps = int(t_max / dt)
    t = np.linspace(0, t_max, n_steps)

    state = np.zeros((n_steps, 4))
    state[0] = initial_state

    for i in range(1, n_steps):
        # RK4 step
        k1 = double_pendulum_derivatives(state[i-1], t[i-1], L1, L2, m1, m2, g)
        k2 = double_pendulum_derivatives(state[i-1] + dt/2 * k1, t[i-1] + dt/2, L1, L2, m1, m2, g)
        k3 = double_pendulum_derivatives(state[i-1] + dt/2 * k2, t[i-1] + dt/2, L1, L2, m1, m2, g)
        k4 = double_pendulum_derivatives(state[i-1] + dt * k3, t[i-1] + dt, L1, L2, m1, m2, g)
        state[i] = state[i-1] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    return t, state


def get_cartesian_coords(state, L1, L2):
    """Convert angles to Cartesian coordinates."""
    theta1, _, theta2, _ = state.T

    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)

    return x1, y1, x2, y2


def main():
    # Parameters
    L1, L2 = 1.0, 1.0  # Pendulum lengths (m)
    m1, m2 = 1.0, 1.0  # Masses (kg)
    g = 9.81           # Gravity (m/s^2)
    t_max = 30.0       # Simulation time (s)
    dt = 0.001         # Time step (s)

    # Initial condition
    theta1_0 = np.pi / 2  # 90 degrees
    theta2_0 = np.pi / 2
    omega1_0 = 0.0
    omega2_0 = 0.0

    initial_state = np.array([theta1_0, omega1_0, theta2_0, omega2_0])

    # Simulate
    t, state = simulate_double_pendulum(initial_state, L1, L2, m1, m2, t_max, dt, g)
    x1, y1, x2, y2 = get_cartesian_coords(state, L1, L2)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Trajectory of second mass
    ax1 = axes[0, 0]
    ax1.plot(x2, y2, lw=0.3, alpha=0.7)
    ax1.plot(0, 0, 'ko', markersize=10, label='Pivot')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_title('Trajectory of Second Mass')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Angle time series
    ax2 = axes[0, 1]
    ax2.plot(t, np.degrees(state[:, 0]), 'b-', lw=0.5, label='θ₁', alpha=0.8)
    ax2.plot(t, np.degrees(state[:, 2]), 'r-', lw=0.5, label='θ₂', alpha=0.8)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Angle (degrees)')
    ax2.set_title('Angular Displacement vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Phase space (theta1 vs omega1)
    ax3 = axes[0, 2]
    ax3.plot(np.degrees(state[:, 0]), state[:, 1], lw=0.3, alpha=0.7)
    ax3.set_xlabel('θ₁ (degrees)')
    ax3.set_ylabel('ω₁ (rad/s)')
    ax3.set_title('Phase Space (Pendulum 1)')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Sensitive dependence on initial conditions
    ax4 = axes[1, 0]

    # Simulate with slightly different initial condition
    perturbation = 1e-6
    initial_perturbed = initial_state.copy()
    initial_perturbed[0] += perturbation

    t2, state2 = simulate_double_pendulum(initial_perturbed, L1, L2, m1, m2, t_max, dt, g)

    # Calculate divergence
    divergence = np.sqrt(np.sum((state - state2)**2, axis=1))

    ax4.semilogy(t, divergence, 'b-', lw=1)
    ax4.axhline(y=perturbation, color='r', linestyle='--', label=f'Initial: {perturbation}')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Divergence (log scale)')
    ax4.set_title(f'Butterfly Effect (Δθ₁ = {perturbation})')
    ax4.legend()
    ax4.grid(True, alpha=0.3, which='both')

    # Plot 5: Compare trajectories
    ax5 = axes[1, 1]
    x1_2, y1_2, x2_2, y2_2 = get_cartesian_coords(state2, L1, L2)

    ax5.plot(x2[:len(t)//3], y2[:len(t)//3], 'b-', lw=0.5, alpha=0.7, label='Original')
    ax5.plot(x2_2[:len(t)//3], y2_2[:len(t)//3], 'r--', lw=0.5, alpha=0.7, label='Perturbed')
    ax5.set_xlabel('x (m)')
    ax5.set_ylabel('y (m)')
    ax5.set_title('Early Trajectories (first 1/3 of simulation)')
    ax5.legend()
    ax5.set_aspect('equal')
    ax5.grid(True, alpha=0.3)

    # Plot 6: Energy conservation check
    ax6 = axes[1, 2]

    # Calculate total energy
    theta1, omega1, theta2, omega2 = state.T

    # Kinetic energy
    T = (0.5 * m1 * L1**2 * omega1**2 +
         0.5 * m2 * (L1**2 * omega1**2 + L2**2 * omega2**2 +
                     2 * L1 * L2 * omega1 * omega2 * np.cos(theta1 - theta2)))

    # Potential energy
    V = -(m1 + m2) * g * L1 * np.cos(theta1) - m2 * g * L2 * np.cos(theta2)

    E_total = T + V
    E_normalized = (E_total - E_total[0]) / abs(E_total[0])

    ax6.plot(t, E_normalized * 100, 'b-', lw=1)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Energy Error (%)')
    ax6.set_title('Energy Conservation')
    ax6.grid(True, alpha=0.3)

    plt.suptitle(f'Double Pendulum Chaos (L₁=L₂={L1}m, m₁=m₂={m1}kg)\n'
                 f'Initial: θ₁=θ₂=90°, ω₁=ω₂=0',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'double_pendulum.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'double_pendulum.png')}")


if __name__ == "__main__":
    main()
