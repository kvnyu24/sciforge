"""
Experiment 60: Fermat's Principle Analog for Mechanics.

Demonstrates how the principle of least action in mechanics is analogous to
Fermat's principle in optics. Both systems minimize an integral quantity
(action or optical path length) leading to analogous trajectories.

In optics: Light travels the path that minimizes the optical path length.
In mechanics: Particles follow paths that extremize the action integral.

We simulate a particle moving through a region with varying "potential" and
show the analogy with light refraction at interfaces.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def optical_path_length(path_y, x_points, n_func):
    """
    Calculate optical path length for a given path.

    OPL = integral of n(x, y) * ds

    Args:
        path_y: y-coordinates of path points
        x_points: x-coordinates of path points
        n_func: refractive index function n(x, y)

    Returns:
        Total optical path length
    """
    opl = 0
    for i in range(len(x_points) - 1):
        dx = x_points[i+1] - x_points[i]
        dy = path_y[i+1] - path_y[i]
        ds = np.sqrt(dx**2 + dy**2)

        # Average refractive index along segment
        x_mid = (x_points[i] + x_points[i+1]) / 2
        y_mid = (path_y[i] + path_y[i+1]) / 2
        n_avg = n_func(x_mid, y_mid)

        opl += n_avg * ds

    return opl


def mechanical_action(path_y, x_points, v_func, mass=1.0, E_total=1.0):
    """
    Calculate abbreviated action for mechanical system.

    For a system with energy E and potential V(x,y):
    S = integral of sqrt(2m(E - V)) * ds

    This is the mechanical analog of optical path length.

    Args:
        path_y: y-coordinates of path points
        x_points: x-coordinates of path points
        v_func: potential energy function V(x, y)
        mass: particle mass
        E_total: total energy

    Returns:
        Abbreviated action
    """
    action = 0
    for i in range(len(x_points) - 1):
        dx = x_points[i+1] - x_points[i]
        dy = path_y[i+1] - path_y[i]
        ds = np.sqrt(dx**2 + dy**2)

        x_mid = (x_points[i] + x_points[i+1]) / 2
        y_mid = (path_y[i] + path_y[i+1]) / 2
        V = v_func(x_mid, y_mid)

        # sqrt(2m(E - V)) is the local momentum magnitude
        kinetic = E_total - V
        if kinetic > 0:
            p = np.sqrt(2 * mass * kinetic)
            action += p * ds
        else:
            action += 1e10  # Forbidden region

    return action


def find_optimal_path(x_start, y_start, x_end, y_end, func, n_points=20):
    """
    Find the path that minimizes the given functional.

    Args:
        x_start, y_start: starting point
        x_end, y_end: ending point
        func: functional to minimize (optical path or action)
        n_points: number of interior points

    Returns:
        x_points, y_points: optimized path
    """
    x_points = np.linspace(x_start, x_end, n_points + 2)

    # Initial guess: straight line
    y_init = np.linspace(y_start, y_end, n_points + 2)

    # Only optimize interior points
    def objective(y_interior):
        y_full = np.concatenate([[y_start], y_interior, [y_end]])
        return func(y_full, x_points)

    result = minimize(objective, y_init[1:-1], method='BFGS')

    y_optimal = np.concatenate([[y_start], result.x, [y_end]])
    return x_points, y_optimal


def snells_law_verification():
    """
    Verify Snell's law emerges from Fermat's principle.

    At an interface between media with n1 and n2:
    n1 * sin(theta1) = n2 * sin(theta2)
    """
    # Two-medium system
    def n_two_media(x, y):
        if x < 0.5:
            return 1.0
        else:
            return 1.5

    # Find optimal path
    x_opt, y_opt = find_optimal_path(0, 0, 1, 0.3,
                                      lambda y, x: optical_path_length(y, x, n_two_media),
                                      n_points=30)

    # Calculate angles at interface
    # Find point closest to interface
    idx = np.argmin(np.abs(x_opt - 0.5))

    if idx > 0 and idx < len(x_opt) - 1:
        # Incident ray direction
        dx1 = x_opt[idx] - x_opt[idx-1]
        dy1 = y_opt[idx] - y_opt[idx-1]
        theta1 = np.arctan2(np.abs(dy1), dx1)

        # Refracted ray direction
        dx2 = x_opt[idx+1] - x_opt[idx]
        dy2 = y_opt[idx+1] - y_opt[idx]
        theta2 = np.arctan2(np.abs(dy2), dx2)

        # Verify Snell's law
        n1, n2 = 1.0, 1.5
        snell_lhs = n1 * np.sin(theta1)
        snell_rhs = n2 * np.sin(theta2)

        return x_opt, y_opt, theta1, theta2, snell_lhs, snell_rhs

    return x_opt, y_opt, 0, 0, 0, 0


def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Fermat's principle - refraction at interface
    ax = axes[0, 0]

    def n_interface(x, y):
        return 1.0 if x < 0.5 else 1.5

    x_opt, y_opt, theta1, theta2, snell_l, snell_r = snells_law_verification()

    # Draw media
    ax.axvspan(0, 0.5, alpha=0.2, color='blue', label='n=1.0')
    ax.axvspan(0.5, 1.0, alpha=0.2, color='red', label='n=1.5')
    ax.axvline(x=0.5, color='black', linestyle='--', lw=2)

    # Draw optimal path
    ax.plot(x_opt, y_opt, 'g-', lw=3, label='Fermat path')

    # Draw straight line for comparison
    ax.plot([0, 1], [0, 0.3], 'k:', lw=2, alpha=0.5, label='Straight line')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f"Fermat's Principle: Refraction\n"
                 f"n1*sin(theta1) = {snell_l:.3f}, n2*sin(theta2) = {snell_r:.3f}")
    ax.legend(fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Plot 2: Mechanical analog - particle in potential step
    ax = axes[0, 1]

    def V_step(x, y):
        return 0.0 if x < 0.5 else 0.3

    E_total = 1.0
    x_mech, y_mech = find_optimal_path(0, 0, 1, 0.3,
                                        lambda y, x: mechanical_action(y, x, V_step, E_total=E_total),
                                        n_points=30)

    # Draw potential regions
    ax.axvspan(0, 0.5, alpha=0.2, color='green', label='V=0')
    ax.axvspan(0.5, 1.0, alpha=0.2, color='orange', label='V=0.3')
    ax.axvline(x=0.5, color='black', linestyle='--', lw=2)

    ax.plot(x_mech, y_mech, 'b-', lw=3, label='Least action path')
    ax.plot([0, 1], [0, 0.3], 'k:', lw=2, alpha=0.5, label='Straight line')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Mechanical Analog: Potential Step\n'
                 'Particle bends like light at interface')
    ax.legend(fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Plot 3: Optical vs Mechanical comparison
    ax = axes[0, 2]

    ax.plot(x_opt, y_opt, 'g-', lw=2, label='Optical (Fermat)')
    ax.plot(x_mech, y_mech, 'b--', lw=2, label='Mechanical (Maupertuis)')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Comparison: Optical vs Mechanical\n'
                 'Both extremize path integrals')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Plot 4: Gradient index medium
    ax = axes[1, 0]

    def n_gradient(x, y):
        return 1.0 + 0.5 * y

    # Show refractive index field
    x_grid = np.linspace(0, 1, 50)
    y_grid = np.linspace(-0.5, 0.5, 50)
    X, Y = np.meshgrid(x_grid, y_grid)
    N = n_gradient(X, Y)

    im = ax.contourf(X, Y, N, levels=20, cmap='viridis', alpha=0.7)
    plt.colorbar(im, ax=ax, label='Refractive index n')

    # Find curved path
    x_grad, y_grad = find_optimal_path(0, -0.3, 1, 0.3,
                                        lambda y, x: optical_path_length(y, x, n_gradient),
                                        n_points=30)

    ax.plot(x_grad, y_grad, 'r-', lw=3, label='Fermat path')
    ax.plot([0, 1], [-0.3, 0.3], 'w:', lw=2, label='Straight line')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Gradient Index Medium\n'
                 'Path curves toward higher n')
    ax.legend(fontsize=8)

    # Plot 5: Mechanical analog - gravity field
    ax = axes[1, 1]

    def V_gravity(x, y):
        return -y  # Potential increases upward

    x_grav, y_grav = find_optimal_path(0, -0.3, 1, 0.3,
                                        lambda y, x: mechanical_action(y, x, V_gravity, E_total=2.0),
                                        n_points=30)

    # Show potential field
    V = V_gravity(X, Y)
    im = ax.contourf(X, Y, -V, levels=20, cmap='coolwarm', alpha=0.7)
    plt.colorbar(im, ax=ax, label='Potential V')

    ax.plot(x_grav, y_grav, 'k-', lw=3, label='Least action path')
    ax.plot([0, 1], [-0.3, 0.3], 'w:', lw=2, label='Straight line')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Gravity Analog\n'
                 'Parabolic trajectory emerges')
    ax.legend(fontsize=8)

    # Plot 6: Summary diagram
    ax = axes[1, 2]
    ax.axis('off')

    summary = """Fermat-Maupertuis Correspondence
=====================================

OPTICS (Fermat's Principle):
  - Minimize: OPL = integral n(x,y) ds
  - n = refractive index = c/v
  - Result: Snell's law at interfaces
  - Curved paths in gradient media

MECHANICS (Maupertuis' Principle):
  - Minimize: Action S = integral p ds
  - p = sqrt(2m(E-V)) = momentum
  - Result: Refraction at potential steps
  - Curved paths in force fields

CORRESPONDENCE:
  n(x,y)  <-->  sqrt(2m(E-V(x,y)))

  Optical path length <--> Abbreviated action

  Speed of light v <--> Particle velocity

  This analogy led Hamilton to develop
  his wave-mechanical formulation,
  which inspired Schroedinger's equation!

Historical note: This correspondence was
key to the development of quantum mechanics."""

    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle("Fermat's Principle and the Mechanical Analog\n"
                 "Variational Principles Unite Optics and Mechanics",
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'fermat_principle.png'), dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/fermat_principle.png")


if __name__ == "__main__":
    main()
