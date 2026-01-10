"""
Experiment 26: Constant acceleration - analytic vs numeric integration.

Compares analytical kinematic equations with numerical integration
for motion under constant acceleration.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def analytical_kinematics(t, x0, v0, a):
    """Analytical solution for constant acceleration."""
    x = x0 + v0 * t + 0.5 * a * t**2
    v = v0 + a * t
    return x, v


def euler_integration(x0, v0, a, dt, n_steps):
    """Euler method integration."""
    x = np.zeros(n_steps + 1)
    v = np.zeros(n_steps + 1)
    x[0], v[0] = x0, v0

    for i in range(n_steps):
        v[i+1] = v[i] + a * dt
        x[i+1] = x[i] + v[i] * dt

    return x, v


def verlet_integration(x0, v0, a, dt, n_steps):
    """Velocity Verlet integration."""
    x = np.zeros(n_steps + 1)
    v = np.zeros(n_steps + 1)
    x[0], v[0] = x0, v0

    for i in range(n_steps):
        x[i+1] = x[i] + v[i] * dt + 0.5 * a * dt**2
        v[i+1] = v[i] + a * dt

    return x, v


def main():
    # Parameters
    x0 = 0.0  # Initial position
    v0 = 10.0  # Initial velocity (m/s)
    a = -9.81  # Acceleration (gravity)
    t_final = 3.0

    # Different time steps
    dts = [0.5, 0.1, 0.01]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for col, dt in enumerate(dts):
        n_steps = int(t_final / dt)
        t_num = np.arange(n_steps + 1) * dt
        t_ana = np.linspace(0, t_final, 500)

        # Analytical
        x_ana, v_ana = analytical_kinematics(t_ana, x0, v0, a)

        # Numerical
        x_euler, v_euler = euler_integration(x0, v0, a, dt, n_steps)
        x_verlet, v_verlet = verlet_integration(x0, v0, a, dt, n_steps)

        # Position plot
        ax = axes[0, col]
        ax.plot(t_ana, x_ana, 'k-', lw=2, label='Analytical')
        ax.plot(t_num, x_euler, 'bo-', markersize=4, label='Euler')
        ax.plot(t_num, x_verlet, 'rs--', markersize=4, label='Verlet')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (m)')
        ax.set_title(f'Position (dt = {dt} s)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Velocity plot
        ax = axes[1, col]
        ax.plot(t_ana, v_ana, 'k-', lw=2, label='Analytical')
        ax.plot(t_num, v_euler, 'bo-', markersize=4, label='Euler')
        ax.plot(t_num, v_verlet, 'rs--', markersize=4, label='Verlet')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity (m/s)')
        ax.set_title(f'Velocity (dt = {dt} s)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Calculate errors at final time
        x_ana_final, v_ana_final = analytical_kinematics(t_num[-1], x0, v0, a)
        err_x_euler = abs(x_euler[-1] - x_ana_final)
        err_x_verlet = abs(x_verlet[-1] - x_ana_final)

        print(f"dt = {dt}: Euler error = {err_x_euler:.4e}, Verlet error = {err_x_verlet:.4e}")

    plt.suptitle('Constant Acceleration: Analytical vs Numerical\n' +
                 f'x₀ = {x0} m, v₀ = {v0} m/s, a = {a} m/s²',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'constant_acceleration.png'), dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/constant_acceleration.png")


if __name__ == "__main__":
    main()
