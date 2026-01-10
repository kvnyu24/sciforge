"""
Experiment 3: Long-time integration stability.

Tests pendulum integration for 10^6 steps to demonstrate
energy drift and phase error accumulation over long times.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def pendulum_rhs(theta, omega, g_over_L):
    """Right-hand side of pendulum equations."""
    return omega, -g_over_L * np.sin(theta)


def rk4_step(theta, omega, dt, g_over_L):
    """RK4 step for pendulum."""
    def f(th, om):
        return pendulum_rhs(th, om, g_over_L)

    k1_th, k1_om = f(theta, omega)
    k2_th, k2_om = f(theta + dt/2*k1_th, omega + dt/2*k1_om)
    k3_th, k3_om = f(theta + dt/2*k2_th, omega + dt/2*k2_om)
    k4_th, k4_om = f(theta + dt*k3_th, omega + dt*k3_om)

    theta_new = theta + dt * (k1_th + 2*k2_th + 2*k3_th + k4_th) / 6
    omega_new = omega + dt * (k1_om + 2*k2_om + 2*k3_om + k4_om) / 6

    return theta_new, omega_new


def verlet_step(theta, omega, dt, g_over_L):
    """Velocity Verlet for pendulum."""
    a = -g_over_L * np.sin(theta)
    theta_new = theta + omega * dt + 0.5 * a * dt**2
    a_new = -g_over_L * np.sin(theta_new)
    omega_new = omega + 0.5 * (a + a_new) * dt
    return theta_new, omega_new


def pendulum_energy(theta, omega, g_over_L):
    """Dimensionless energy: E = 0.5*omega^2 + g/L*(1-cos(theta))"""
    return 0.5 * omega**2 + g_over_L * (1 - np.cos(theta))


def estimate_period_numerically(theta0, g_over_L):
    """Estimate period by integrating one cycle with high precision."""
    dt_fine = 1e-5
    theta, omega = theta0, 0.0
    t = 0

    # Integrate until theta crosses zero going negative
    while theta >= 0:
        theta, omega = verlet_step(theta, omega, dt_fine, g_over_L)
        t += dt_fine

    # Period is approximately 4 * time to reach zero
    return 4 * t


def main():
    # Parameters
    g_over_L = 9.81  # g/L
    theta0 = 0.1  # Small angle (radians)
    omega0 = 0.0

    dt = 0.001
    n_steps = 1000000  # 10^6 steps

    E0 = pendulum_energy(theta0, omega0, g_over_L)
    omega_0 = np.sqrt(g_over_L)  # Natural frequency
    T_approx = 2 * np.pi / omega_0  # Small angle period

    # Storage for sampled data (sample every 1000 steps to save memory)
    sample_rate = 1000
    n_samples = n_steps // sample_rate + 1

    times = np.zeros(n_samples)
    energies_rk4 = np.zeros(n_samples)
    energies_verlet = np.zeros(n_samples)
    phases_rk4 = np.zeros(n_samples)
    phases_verlet = np.zeros(n_samples)

    # Initialize
    theta_rk4, omega_rk4 = theta0, omega0
    theta_verlet, omega_verlet = theta0, omega0

    print(f"Running {n_steps:,} steps with dt={dt}...")
    print(f"Total simulated time: {n_steps * dt:.1f} s ({n_steps * dt / T_approx:.1f} periods)")

    # Integration loop
    sample_idx = 0
    for i in range(n_steps + 1):
        if i % sample_rate == 0:
            t = i * dt
            times[sample_idx] = t
            energies_rk4[sample_idx] = pendulum_energy(theta_rk4, omega_rk4, g_over_L)
            energies_verlet[sample_idx] = pendulum_energy(theta_verlet, omega_verlet, g_over_L)

            # Phase = expected phase from linear theory
            expected_phase = theta0 * np.cos(omega_0 * t)
            phases_rk4[sample_idx] = theta_rk4 - expected_phase
            phases_verlet[sample_idx] = theta_verlet - expected_phase

            sample_idx += 1

            if i % 100000 == 0:
                print(f"  Step {i:,}/{n_steps:,}")

        if i < n_steps:
            theta_rk4, omega_rk4 = rk4_step(theta_rk4, omega_rk4, dt, g_over_L)
            theta_verlet, omega_verlet = verlet_step(theta_verlet, omega_verlet, dt, g_over_L)

    # Calculate relative energy errors
    dE_rk4 = (energies_rk4 - E0) / E0
    dE_verlet = (energies_verlet - E0) / E0

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Energy error vs time
    ax = axes[0, 0]
    ax.plot(times, dE_rk4 * 100, 'b-', lw=0.5, alpha=0.7, label='RK4')
    ax.plot(times, dE_verlet * 100, 'g-', lw=0.5, alpha=0.7, label='Verlet')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Energy error (%)')
    ax.set_title(f'Energy Drift over {n_steps:,} Steps')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Energy error (log scale of absolute value)
    ax = axes[0, 1]
    ax.semilogy(times, np.abs(dE_rk4) + 1e-16, 'b-', lw=0.5, alpha=0.7, label='RK4')
    ax.semilogy(times, np.abs(dE_verlet) + 1e-16, 'g-', lw=0.5, alpha=0.7, label='Verlet')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('|Relative energy error|')
    ax.set_title('Energy Error (Log Scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Phase error vs time
    ax = axes[1, 0]
    ax.plot(times, phases_rk4, 'b-', lw=0.5, alpha=0.7, label='RK4')
    ax.plot(times, phases_verlet, 'g-', lw=0.5, alpha=0.7, label='Verlet')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Phase error (rad)')
    ax.set_title('Phase Error (Deviation from Linear Theory)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Final summary statistics
    ax = axes[1, 1]
    ax.axis('off')

    summary = f"""Long-Time Integration Summary
================================
Total steps: {n_steps:,}
Time step dt: {dt}
Total time: {times[-1]:.1f} s
Number of periods: {times[-1] / T_approx:.1f}

Initial conditions:
  theta_0 = {theta0:.3f} rad ({np.degrees(theta0):.1f} deg)
  omega_0 = {omega0:.3f} rad/s
  E_0 = {E0:.6f}

Final Energy Errors:
  RK4:    {dE_rk4[-1]*100:+.4e} %
  Verlet: {dE_verlet[-1]*100:+.4e} %

Max Energy Errors:
  RK4:    {np.max(np.abs(dE_rk4))*100:.4e} %
  Verlet: {np.max(np.abs(dE_verlet))*100:.4e} %

Energy Drift Rate:
  RK4:    {dE_rk4[-1]/times[-1]*100:.2e} %/s
  Verlet: {dE_verlet[-1]/times[-1]*100:.2e} %/s

Conclusion: Verlet (symplectic) shows bounded energy
oscillation, while RK4 shows secular drift."""

    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Long-Time Integration Stability Test\n' +
                 f'Pendulum: {n_steps:,} steps, dt={dt}', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'long_time_stability.png'), dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/long_time_stability.png")


if __name__ == "__main__":
    main()
