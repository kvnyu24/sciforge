"""
Experiment 9: Shooting method for boundary value problem - Brachistochrone.

Finds the curve of fastest descent using numerical optimization
and compares to the analytical cycloid solution.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, brentq


def brachistochrone_ode(y, x, dydx):
    """
    Brachistochrone ODE from calculus of variations.

    The Euler-Lagrange equation for minimizing time gives:
    y * (1 + (dy/dx)^2) = C (constant)

    Rearranging: dy/dx = sqrt((C - y) / y)
    """
    if y <= 0 or y >= C_global:
        return 0
    return -np.sqrt((C_global - y) / y)


def cycloid_parametric(theta, R):
    """Cycloid in parametric form: x = R(theta - sin(theta)), y = R(1 - cos(theta))"""
    x = R * (theta - np.sin(theta))
    y = R * (1 - np.cos(theta))
    return x, y


def find_cycloid_parameters(x_end, y_end):
    """Find cycloid radius R and angle theta_end for given endpoint."""
    def objective(params):
        R, theta_end = params
        x, y = cycloid_parametric(theta_end, R)
        return (x - x_end)**2 + (y - y_end)**2

    # Initial guess
    R_guess = np.sqrt(x_end**2 + y_end**2) / 2
    theta_guess = np.pi

    from scipy.optimize import minimize
    result = minimize(objective, [R_guess, theta_guess], method='Nelder-Mead')
    return result.x


def descent_time(y_path, x_path, g=9.81):
    """Calculate descent time along a path."""
    if len(y_path) < 2:
        return np.inf

    T = 0
    for i in range(len(y_path) - 1):
        dx = x_path[i+1] - x_path[i]
        dy = y_path[i+1] - y_path[i]
        ds = np.sqrt(dx**2 + dy**2)

        # Average height
        y_avg = (y_path[i] + y_path[i+1]) / 2
        if y_avg <= 0:
            y_avg = 1e-10

        # Speed from energy conservation: v = sqrt(2*g*y)
        v = np.sqrt(2 * g * y_avg)
        if v > 0:
            T += ds / v

    return T


def shooting_method(y_end, x_end, n_points=100, g=9.81):
    """
    Use shooting method to find brachistochrone curve.

    We parameterize by the initial slope and shoot to hit the endpoint.
    """
    def integrate_ode(slope_init, C):
        """Integrate brachistochrone ODE with given initial slope and constant."""
        x = np.linspace(0, x_end, n_points)
        y = np.zeros(n_points)
        y[0] = 1e-6  # Start just below origin

        for i in range(1, n_points):
            dx = x[i] - x[i-1]
            # y * (1 + y'^2) = C => y' = -sqrt((C-y)/y) (going down)
            if y[i-1] > 0 and y[i-1] < C:
                dydx = -np.sqrt((C - y[i-1]) / y[i-1])
            else:
                dydx = slope_init
            y[i] = y[i-1] + dydx * dx
            y[i] = max(y[i], 1e-6)  # Keep positive

        return x, y

    def residual(C):
        """Residual: how far we miss the target y_end."""
        x, y = integrate_ode(-1.0, C)
        return y[-1] - y_end

    # Find C that hits the endpoint
    try:
        C_opt = brentq(residual, y_end * 1.01, y_end * 100)
        x_path, y_path = integrate_ode(-1.0, C_opt)
        return x_path, y_path
    except:
        return None, None


def straight_line(x_end, y_end, n_points=100):
    """Straight line path."""
    x = np.linspace(0, x_end, n_points)
    y = y_end * x / x_end
    return x, y


def parabolic_path(x_end, y_end, n_points=100):
    """Parabolic path: y = a*x^2"""
    a = y_end / x_end**2
    x = np.linspace(0, x_end, n_points)
    y = a * x**2
    return x, y


def circular_arc(x_end, y_end, n_points=100):
    """Circular arc path."""
    # Find circle passing through origin and endpoint with center on perpendicular bisector
    # Simplified: use arc of circle
    R = (x_end**2 + y_end**2) / (2 * y_end)
    center_x, center_y = 0, R

    theta_start = -np.pi/2
    theta_end = np.arctan2(x_end - center_x, center_y - y_end) - np.pi/2

    theta = np.linspace(theta_start, theta_end, n_points)
    x = center_x + R * np.cos(theta + np.pi/2)
    y = center_y - R * np.sin(theta + np.pi/2)

    # Normalize to hit endpoint exactly
    x = x * x_end / x[-1]
    y = y * y_end / y[-1]
    y[0] = 0

    return x, y


def main():
    # Endpoint
    x_end = 2.0
    y_end = 1.0  # Dropping 1 meter
    g = 9.81

    n_points = 200

    # Cycloid (analytical solution)
    R, theta_end = find_cycloid_parameters(x_end, y_end)
    theta = np.linspace(0, theta_end, n_points)
    x_cycloid, y_cycloid = cycloid_parametric(theta, R)

    # Other paths
    x_straight, y_straight = straight_line(x_end, y_end, n_points)
    x_parab, y_parab = parabolic_path(x_end, y_end, n_points)

    # Numerical shooting
    x_shoot, y_shoot = shooting_method(y_end, x_end, n_points, g)

    # Calculate descent times
    T_cycloid = descent_time(y_cycloid, x_cycloid, g)
    T_straight = descent_time(y_straight, x_straight, g)
    T_parab = descent_time(y_parab, x_parab, g)
    T_shoot = descent_time(y_shoot, x_shoot, g) if x_shoot is not None else np.inf

    # Analytical time for cycloid
    T_analytical = theta_end * np.sqrt(R / g)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: All paths
    ax = axes[0, 0]
    ax.plot(x_cycloid, -y_cycloid, 'b-', lw=2, label=f'Cycloid (T={T_cycloid:.4f}s)')
    ax.plot(x_straight, -y_straight, 'r--', lw=2, label=f'Straight (T={T_straight:.4f}s)')
    ax.plot(x_parab, -y_parab, 'g-.', lw=2, label=f'Parabola (T={T_parab:.4f}s)')
    if x_shoot is not None:
        ax.plot(x_shoot, -y_shoot, 'm:', lw=2, label=f'Shooting (T={T_shoot:.4f}s)')

    ax.plot(0, 0, 'ko', markersize=10, label='Start')
    ax.plot(x_end, -y_end, 'k^', markersize=10, label='End')

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Brachistochrone: Curve of Fastest Descent')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Plot 2: Speed vs position
    ax = axes[0, 1]

    for name, (x, y) in [('Cycloid', (x_cycloid, y_cycloid)),
                          ('Straight', (x_straight, y_straight)),
                          ('Parabola', (x_parab, y_parab))]:
        v = np.sqrt(2 * g * y)
        ax.plot(x, v, lw=2, label=name)

    ax.set_xlabel('x (m)')
    ax.set_ylabel('Speed (m/s)')
    ax.set_title('Speed Along Path (v = √(2gy))')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Cumulative time along path
    ax = axes[1, 0]

    for name, (x, y) in [('Cycloid', (x_cycloid, y_cycloid)),
                          ('Straight', (x_straight, y_straight)),
                          ('Parabola', (x_parab, y_parab))]:
        times = [0]
        for i in range(1, len(y)):
            dx = x[i] - x[i-1]
            dy = y[i] - y[i-1]
            ds = np.sqrt(dx**2 + dy**2)
            y_avg = (y[i] + y[i-1]) / 2
            v = np.sqrt(2 * g * max(y_avg, 1e-10))
            times.append(times[-1] + ds / v)
        ax.plot(x, times, lw=2, label=name)

    ax.set_xlabel('x (m)')
    ax.set_ylabel('Elapsed time (s)')
    ax.set_title('Cumulative Descent Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Summary
    ax = axes[1, 1]
    ax.axis('off')

    summary = f"""Brachistochrone Problem Summary
================================
Start: (0, 0)
End: ({x_end}, -{y_end}) m
g = {g} m/s²

Descent Times:
--------------
Cycloid (optimal):  {T_cycloid:.6f} s
Analytical:         {T_analytical:.6f} s
Straight line:      {T_straight:.6f} s (+{(T_straight/T_cycloid-1)*100:.1f}%)
Parabola:          {T_parab:.6f} s (+{(T_parab/T_cycloid-1)*100:.1f}%)

Cycloid Parameters:
  Radius R = {R:.4f} m
  Angle θ_end = {np.degrees(theta_end):.1f}°

The brachistochrone (Greek: "shortest time") is
a cycloid - the curve traced by a point on a
rolling circle. This was one of the first
problems solved using calculus of variations
(Johann Bernoulli, 1696).

Key insight: The optimal path starts steep
(gaining speed quickly) then levels out,
rather than following the shortest distance."""

    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Brachistochrone: The Curve of Fastest Descent',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'brachistochrone.png'), dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/brachistochrone.png")


if __name__ == "__main__":
    main()
