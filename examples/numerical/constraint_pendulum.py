"""
Experiment 7: Constraint stabilization - pendulum as constraint.

Compares explicit constraint handling (Lagrange multipliers)
vs direct integration, showing constraint drift.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def pendulum_cartesian_explicit(x0, y0, vx0, vy0, L, g, dt, n_steps):
    """
    Pendulum in Cartesian coordinates without explicit constraint enforcement.
    Constraint: x^2 + y^2 = L^2
    """
    x, y = x0, y0
    vx, vy = vx0, vy0
    m = 1.0

    xs, ys = [x], [y]
    vxs, vys = [vx], [vy]
    constraints = [np.sqrt(x**2 + y**2) - L]

    for _ in range(n_steps):
        # Only gravity, no constraint force
        ax = 0
        ay = -g

        # Velocity Verlet without constraint
        x_new = x + vx * dt + 0.5 * ax * dt**2
        y_new = y + vy * dt + 0.5 * ay * dt**2
        vx_new = vx + ax * dt
        vy_new = vy + ay * dt

        x, y = x_new, y_new
        vx, vy = vx_new, vy_new

        xs.append(x)
        ys.append(y)
        vxs.append(vx)
        vys.append(vy)
        constraints.append(np.sqrt(x**2 + y**2) - L)

    return (np.array(xs), np.array(ys), np.array(vxs), np.array(vys),
            np.array(constraints))


def pendulum_lagrange_multiplier(x0, y0, vx0, vy0, L, g, dt, n_steps):
    """
    Pendulum using Lagrange multipliers to enforce constraint.

    Constraint: C = x^2 + y^2 - L^2 = 0
    Gradient: grad C = (2x, 2y)

    Equations of motion:
    m*ax = lambda * 2x
    m*ay = -mg + lambda * 2y

    Plus constraint equation: x^2 + y^2 = L^2
    """
    x, y = x0, y0
    vx, vy = vx0, vy0
    m = 1.0

    xs, ys = [x], [y]
    vxs, vys = [vx], [vy]
    constraints = [np.sqrt(x**2 + y**2) - L]
    lambdas = []

    for _ in range(n_steps):
        # Compute Lagrange multiplier
        # From constraint at acceleration level:
        # d^2/dt^2 (x^2 + y^2) = 2(x*ax + y*ay + vx^2 + vy^2) = 0
        # Substituting ax = 2*lambda*x/m, ay = -g + 2*lambda*y/m:
        # 2(x * 2*lambda*x/m + y*(-g + 2*lambda*y/m) + vx^2 + vy^2) = 0
        # 4*lambda*(x^2 + y^2)/m - 2*g*y + 2*(vx^2 + vy^2) = 0
        # lambda = (g*y - vx^2 - vy^2) * m / (2*(x^2 + y^2))

        r_sq = x**2 + y**2
        if r_sq > 0:
            lam = m * (g * y - vx**2 - vy**2) / (2 * r_sq)
        else:
            lam = 0

        lambdas.append(lam)

        # Compute accelerations with constraint force
        ax = 2 * lam * x / m
        ay = -g + 2 * lam * y / m

        # Velocity Verlet
        x_new = x + vx * dt + 0.5 * ax * dt**2
        y_new = y + vy * dt + 0.5 * ay * dt**2

        # Recompute lambda at new position for velocity update
        r_sq_new = x_new**2 + y_new**2
        vx_half = vx + 0.5 * ax * dt
        vy_half = vy + 0.5 * ay * dt

        if r_sq_new > 0:
            lam_new = m * (g * y_new - vx_half**2 - vy_half**2) / (2 * r_sq_new)
        else:
            lam_new = 0

        ax_new = 2 * lam_new * x_new / m
        ay_new = -g + 2 * lam_new * y_new / m

        vx_new = vx + 0.5 * (ax + ax_new) * dt
        vy_new = vy + 0.5 * (ay + ay_new) * dt

        x, y = x_new, y_new
        vx, vy = vx_new, vy_new

        xs.append(x)
        ys.append(y)
        vxs.append(vx)
        vys.append(vy)
        constraints.append(np.sqrt(x**2 + y**2) - L)

    return (np.array(xs), np.array(ys), np.array(vxs), np.array(vys),
            np.array(constraints), np.array(lambdas))


def pendulum_projection(x0, y0, vx0, vy0, L, g, dt, n_steps):
    """
    Pendulum with projection method - project back to constraint after each step.
    """
    x, y = x0, y0
    vx, vy = vx0, vy0

    xs, ys = [x], [y]
    vxs, vys = [vx], [vy]
    constraints = [np.sqrt(x**2 + y**2) - L]

    for _ in range(n_steps):
        # Simple Euler step with gravity
        ax, ay = 0, -g

        x_new = x + vx * dt
        y_new = y + vy * dt
        vx_new = vx + ax * dt
        vy_new = vy + ay * dt

        # Project position onto constraint manifold
        r = np.sqrt(x_new**2 + y_new**2)
        if r > 0:
            x_new = L * x_new / r
            y_new = L * y_new / r

            # Project velocity to be tangent to constraint
            # v_tangent = v - (v . r_hat) * r_hat
            r_hat = np.array([x_new, y_new]) / L
            v = np.array([vx_new, vy_new])
            v_radial = np.dot(v, r_hat)
            v_tangent = v - v_radial * r_hat
            vx_new, vy_new = v_tangent

        x, y = x_new, y_new
        vx, vy = vx_new, vy_new

        xs.append(x)
        ys.append(y)
        vxs.append(vx)
        vys.append(vy)
        constraints.append(np.sqrt(x**2 + y**2) - L)

    return (np.array(xs), np.array(ys), np.array(vxs), np.array(vys),
            np.array(constraints))


def pendulum_theta(theta0, omega0, L, g, dt, n_steps):
    """Pendulum in angular coordinates (exact constraint)."""
    theta = theta0
    omega = omega0

    thetas = [theta]
    omegas = [omega]

    for _ in range(n_steps):
        # d^2theta/dt^2 = -(g/L) * sin(theta)
        alpha = -(g / L) * np.sin(theta)

        theta_new = theta + omega * dt + 0.5 * alpha * dt**2
        alpha_new = -(g / L) * np.sin(theta_new)
        omega_new = omega + 0.5 * (alpha + alpha_new) * dt

        theta, omega = theta_new, omega_new
        thetas.append(theta)
        omegas.append(omega)

    thetas = np.array(thetas)
    xs = L * np.sin(thetas)
    ys = -L * np.cos(thetas)

    return xs, ys, thetas, np.array(omegas)


def main():
    # Parameters
    L = 1.0  # Pendulum length
    g = 9.81  # Gravity
    theta0 = np.pi / 4  # 45 degrees

    # Initial conditions (Cartesian)
    x0 = L * np.sin(theta0)
    y0 = -L * np.cos(theta0)
    vx0 = 0.0
    vy0 = 0.0

    dt = 0.01
    n_steps = 5000
    t = np.arange(n_steps + 1) * dt

    # Run all methods
    x_expl, y_expl, vx_expl, vy_expl, c_expl = pendulum_cartesian_explicit(
        x0, y0, vx0, vy0, L, g, dt, n_steps)

    x_lagr, y_lagr, vx_lagr, vy_lagr, c_lagr, lam = pendulum_lagrange_multiplier(
        x0, y0, vx0, vy0, L, g, dt, n_steps)

    x_proj, y_proj, vx_proj, vy_proj, c_proj = pendulum_projection(
        x0, y0, vx0, vy0, L, g, dt, n_steps)

    x_theta, y_theta, thetas, omegas = pendulum_theta(
        theta0, 0.0, L, g, dt, n_steps)

    # Calculate energies
    def energy(x, y, vx, vy, g=9.81, m=1.0):
        return 0.5 * m * (vx**2 + vy**2) + m * g * (y + L)

    E0 = energy(x0, y0, vx0, vy0)
    E_lagr = np.array([energy(x, y, vx, vy) for x, y, vx, vy in
                       zip(x_lagr, y_lagr, vx_lagr, vy_lagr)])
    E_proj = np.array([energy(x, y, vx, vy) for x, y, vx, vy in
                       zip(x_proj, y_proj, vx_proj, vy_proj)])
    E_theta = 0.5 * (L * omegas)**2 + g * (y_theta + L)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Trajectories
    ax = axes[0, 0]
    ax.plot(x_expl, y_expl, 'r-', lw=1, alpha=0.5, label='No constraint')
    ax.plot(x_lagr, y_lagr, 'b-', lw=1, alpha=0.7, label='Lagrange mult.')
    ax.plot(x_proj, y_proj, 'g--', lw=1, alpha=0.7, label='Projection')
    ax.plot(x_theta, y_theta, 'k:', lw=2, alpha=0.5, label='Angular (exact)')

    # Draw constraint circle
    theta_circle = np.linspace(0, 2*np.pi, 100)
    ax.plot(L * np.cos(theta_circle), L * np.sin(theta_circle), 'gray',
            linestyle='--', alpha=0.3)

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Pendulum Trajectories')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 0.5)

    # Plot 2: Constraint violation
    ax = axes[0, 1]
    ax.semilogy(t, np.abs(c_expl) + 1e-16, 'r-', lw=1, label='No constraint')
    ax.semilogy(t, np.abs(c_lagr) + 1e-16, 'b-', lw=1, label='Lagrange mult.')
    ax.semilogy(t, np.abs(c_proj) + 1e-16, 'g--', lw=1, label='Projection')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('|r - L| (m)')
    ax.set_title('Constraint Violation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Energy conservation
    ax = axes[1, 0]
    ax.plot(t, (E_lagr - E0) / E0 * 100, 'b-', lw=1, label='Lagrange mult.')
    ax.plot(t, (E_proj - E0) / E0 * 100, 'g--', lw=1, label='Projection')
    ax.plot(t, (E_theta - E0) / E0 * 100, 'k:', lw=2, alpha=0.5, label='Angular')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Energy error (%)')
    ax.set_title('Energy Conservation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Lagrange multiplier (constraint force)
    ax = axes[1, 1]

    # Theoretical: lambda = tension = m * (g*cos(theta) + L*omega^2)
    # In our formulation: constraint force = 2*lambda*(x, y)
    # Tension T = |2*lambda| * L = |2*lambda*L|

    tension_numerical = np.abs(2 * lam * L)
    tension_analytical = g * np.cos(thetas[:-1]) + L * omegas[:-1]**2

    ax.plot(t[:-1], tension_numerical, 'b-', lw=1, label='Numerical (Lagrange)')
    ax.plot(t[:-1], tension_analytical, 'r--', lw=1, alpha=0.7, label='Analytical')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Tension / m (m/s²)')
    ax.set_title('Constraint Force (String Tension)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Constraint Handling Methods for Pendulum\n' +
                 f'L = {L} m, θ₀ = {np.degrees(theta0):.0f}°, dt = {dt} s',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'constraint_pendulum.png'), dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/constraint_pendulum.png")


if __name__ == "__main__":
    main()
