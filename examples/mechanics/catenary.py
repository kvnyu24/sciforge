"""
Experiment 70: Catenary Curve.

Demonstrates the catenary curve - the shape assumed by a uniform chain
or cable hanging under its own weight, supported at both ends.

Key concepts:
1. The catenary equation: y = a * cosh(x/a) where a = T_0 / (rho * g)
2. Not a parabola! (but close for small sag)
3. Derived from energy minimization or force balance
4. Historical: Galileo thought it was a parabola

The catenary minimizes the potential energy of the hanging chain.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, brentq


def catenary(x, a, x0=0, y0=0):
    """
    Catenary curve: y = a * cosh((x - x0) / a) + C

    Args:
        x: x-coordinates
        a: Catenary parameter (horizontal tension / weight per length)
        x0: x-coordinate of lowest point
        y0: y-coordinate of lowest point

    Returns:
        y-coordinates
    """
    return a * (np.cosh((x - x0) / a) - 1) + y0


def parabola_approx(x, a, x0=0, y0=0):
    """
    Parabolic approximation to catenary.

    For small x/a: cosh(x/a) - 1 approx (x/a)^2 / 2

    So y approx (x - x0)^2 / (2*a)
    """
    return (x - x0)**2 / (2 * a) + y0


def find_catenary_params(x1, y1, x2, y2, L):
    """
    Find catenary parameters given endpoints and total length.

    The catenary passes through (x1, y1) and (x2, y2) with arc length L.

    For simplicity, assume x1 < x2 and solve for a and the minimum point.
    """
    # The arc length of catenary from x=x1 to x=x2 is:
    # L = a * (sinh((x2-x0)/a) - sinh((x1-x0)/a))

    # For symmetric case (x1 = -d, x2 = d, y1 = y2 = h, lowest at x=0):
    # L = 2 * a * sinh(d/a)
    # h = a * (cosh(d/a) - 1)

    d = (x2 - x1) / 2
    x_mid = (x1 + x2) / 2

    # Need to solve: L = 2 * a * sinh(d/a) for a
    # This is transcendental, use numerical root finding

    def length_equation(a):
        if a <= 0:
            return float('inf')
        return 2 * a * np.sinh(d / a) - L

    # Find bounds for a
    a_min = L / 100  # Very curved
    a_max = L * 10   # Nearly straight

    try:
        a = brentq(length_equation, a_min, a_max)
    except Exception:
        a = d  # Fallback

    # Now find the y-offset
    y_lowest = y1 - a * (np.cosh((x1 - x_mid) / a) - 1)

    return a, x_mid, y_lowest


def chain_energy(positions, g=9.81):
    """
    Calculate potential energy of a discrete chain.

    Args:
        positions: Array of shape (N, 2) with (x, y) coordinates
        g: Gravitational acceleration

    Returns:
        Total potential energy (assuming uniform mass per segment)
    """
    # Assume unit mass per segment
    y = positions[:, 1]
    return np.sum(y) * g


def simulate_chain_relaxation(x1, x2, y1, y2, n_points, n_steps=1000, dt=0.01):
    """
    Simulate chain relaxation to equilibrium using damped dynamics.

    Args:
        x1, x2: x-coordinates of endpoints
        y1, y2: y-coordinates of endpoints
        n_points: Number of points in chain (including endpoints)
        n_steps: Number of simulation steps
        dt: Time step

    Returns:
        Final positions and energy history
    """
    # Initialize as straight line
    x = np.linspace(x1, x2, n_points)
    y = np.linspace(y1, y2, n_points)
    positions = np.column_stack([x, y])

    # Fixed endpoints
    fixed = np.zeros(n_points, dtype=bool)
    fixed[0] = True
    fixed[-1] = True

    # Segment length (constraint)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    segment_length = np.sqrt(dx**2 + dy**2)

    velocities = np.zeros_like(positions)
    g = 9.81
    damping = 5.0
    spring_k = 1000.0  # Strong spring to maintain segment lengths

    energy_history = []

    for step in range(n_steps):
        forces = np.zeros_like(positions)

        # Gravity
        forces[:, 1] -= g

        # Spring forces to maintain segment lengths
        for i in range(n_points - 1):
            diff = positions[i+1] - positions[i]
            dist = np.linalg.norm(diff)
            if dist > 0:
                direction = diff / dist
                stretch = dist - segment_length
                force = spring_k * stretch * direction

                if not fixed[i]:
                    forces[i] += force
                if not fixed[i+1]:
                    forces[i+1] -= force

        # Damping
        forces -= damping * velocities

        # Update (simple Euler for damped system)
        for i in range(n_points):
            if not fixed[i]:
                velocities[i] += forces[i] * dt
                positions[i] += velocities[i] * dt

        # Calculate energy
        energy_history.append(chain_energy(positions, g))

    return positions, np.array(energy_history)


def arc_length(x, a, x0=0):
    """
    Arc length of catenary from x0 to x.

    s = a * sinh((x - x0) / a)
    """
    return a * np.sinh((x - x0) / a)


def main():
    fig = plt.figure(figsize=(16, 12))

    # --- Plot 1: Catenary vs Parabola ---
    ax1 = fig.add_subplot(2, 3, 1)

    # Parameters
    a = 2.0  # Catenary parameter
    x = np.linspace(-3, 3, 200)

    y_cat = catenary(x, a)
    y_par = parabola_approx(x, a)

    ax1.plot(x, y_cat, 'b-', lw=2, label='Catenary: a*cosh(x/a)')
    ax1.plot(x, y_par, 'r--', lw=2, label='Parabola: x^2/(2a)')

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Catenary vs Parabola\n(Same curvature at minimum)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Mark the difference
    idx = np.argmax(np.abs(y_cat - y_par))
    ax1.annotate(f'Max diff: {np.max(np.abs(y_cat - y_par)):.3f}',
                 xy=(x[idx], y_cat[idx]), xytext=(x[idx]+0.5, y_cat[idx]+0.5),
                 arrowprops=dict(arrowstyle='->', color='green'),
                 fontsize=10, color='green')

    # --- Plot 2: Different catenary parameters ---
    ax2 = fig.add_subplot(2, 3, 2)

    a_values = [0.5, 1.0, 2.0, 4.0]
    colors = ['red', 'orange', 'green', 'blue']

    for a, color in zip(a_values, colors):
        x = np.linspace(-3, 3, 200)
        y = catenary(x, a)
        ax2.plot(x, y, color=color, lw=2, label=f'a = {a}')

    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Catenaries with Different Parameters\n(a = T_0 / (rho * g))')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 6)

    # --- Plot 3: Chain simulation ---
    ax3 = fig.add_subplot(2, 3, 3)

    # Simulate a hanging chain
    x1, x2 = -2.0, 2.0
    y1, y2 = 3.0, 3.0
    n_points = 30

    positions, energy_hist = simulate_chain_relaxation(
        x1, x2, y1, y2, n_points, n_steps=2000, dt=0.01
    )

    # Plot simulated chain
    ax3.plot(positions[:, 0], positions[:, 1], 'bo-', markersize=4,
             label='Simulated chain')

    # Find best-fit catenary
    # Total length of simulated chain
    L_sim = np.sum(np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1)))

    a_fit, x0_fit, y0_fit = find_catenary_params(x1, y1, x2, y2, L_sim)
    x_fit = np.linspace(x1, x2, 100)
    y_fit = catenary(x_fit, a_fit, x0_fit, y0_fit)

    ax3.plot(x_fit, y_fit, 'r-', lw=2, label=f'Catenary fit (a={a_fit:.2f})')

    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('Simulated Chain Relaxation\n(Converges to catenary)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')

    # --- Plot 4: Energy minimization ---
    ax4 = fig.add_subplot(2, 3, 4)

    # Plot energy during relaxation
    ax4.semilogy(energy_hist - energy_hist[-1] + 1e-10, 'b-', lw=1)
    ax4.set_xlabel('Simulation step')
    ax4.set_ylabel('Energy - Final Energy')
    ax4.set_title('Energy Minimization During Relaxation\n(Chain finds minimum energy shape)')
    ax4.grid(True, alpha=0.3)

    # --- Plot 5: Force balance derivation ---
    ax5 = fig.add_subplot(2, 3, 5)

    # Show the force diagram for a small element
    a = 2.0
    x_elem = 1.0

    # Draw catenary
    x_cat = np.linspace(-2, 2, 100)
    y_cat = catenary(x_cat, a)
    ax5.plot(x_cat, y_cat, 'b-', lw=2)

    # Mark element
    dx = 0.3
    x_left = x_elem - dx/2
    x_right = x_elem + dx/2
    y_left = catenary(x_left, a)
    y_right = catenary(x_right, a)

    ax5.plot([x_left, x_right], [y_left, y_right], 'ro-', markersize=8, lw=3)

    # Tension vectors
    slope_left = np.sinh(x_left / a)
    slope_right = np.sinh(x_right / a)

    T0 = 1.0  # Horizontal tension
    scale = 0.5

    # Left tension (pointing into element)
    T_left_x = T0 * scale
    T_left_y = T0 * slope_left * scale
    ax5.arrow(x_left, y_left, -T_left_x, -T_left_y,
              head_width=0.1, head_length=0.05, fc='green', ec='green')

    # Right tension (pointing out of element)
    T_right_x = T0 * scale
    T_right_y = T0 * slope_right * scale
    ax5.arrow(x_right, y_right, T_right_x, T_right_y,
              head_width=0.1, head_length=0.05, fc='green', ec='green')

    # Weight
    x_mid = (x_left + x_right) / 2
    y_mid = (y_left + y_right) / 2
    ax5.arrow(x_mid, y_mid, 0, -0.3,
              head_width=0.1, head_length=0.05, fc='red', ec='red')

    ax5.annotate('T', (x_left - 0.5, y_left - 0.3), fontsize=12, color='green')
    ax5.annotate('T', (x_right + 0.2, y_right + 0.2), fontsize=12, color='green')
    ax5.annotate('W', (x_mid + 0.1, y_mid - 0.4), fontsize=12, color='red')

    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    ax5.set_title('Force Balance on Element\nT*sin(theta) balances weight')
    ax5.set_aspect('equal')
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(-2.5, 2.5)
    ax5.set_ylim(-0.5, 3)

    # --- Plot 6: Theory summary ---
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    summary = """The Catenary Curve
==================

EQUATION:
---------
    y = a * cosh(x/a)   or   y = a * (cosh(x/a) - 1)

where a = T_0 / (rho * g)
    T_0 = horizontal tension
    rho = mass per unit length
    g   = gravitational acceleration

DERIVATION (Force Balance):
---------------------------
Consider small element ds at angle theta:
- Horizontal: T*cos(theta) = T_0 (constant)
- Vertical:   d(T*sin(theta)) = rho*g*ds

Since tan(theta) = dy/dx and ds = sqrt(1 + (dy/dx)^2)*dx:

    d^2y/dx^2 = (rho*g/T_0) * sqrt(1 + (dy/dx)^2)

Solution: y = a * cosh(x/a)

ENERGY MINIMIZATION:
--------------------
The catenary minimizes potential energy:
    U = integral(rho*g*y*ds)

subject to fixed length constraint.

COMPARISON WITH PARABOLA:
-------------------------
Parabola: y = x^2/(2a)  (uniform horizontal load)
Catenary: y = a*(cosh(x/a) - 1)  (uniform load along arc)

For small x/a, they are nearly identical!
Difference grows as sag increases.

APPLICATIONS:
-------------
- Suspension bridge cables
- Power lines
- Architectural arches (inverted catenary)
- Gateway Arch in St. Louis

HISTORICAL NOTE:
----------------
Galileo (1638): Thought it was a parabola
Leibniz, Huygens, Bernoulli (1691): Solved correctly
Name: From Latin "catena" meaning chain"""

    ax6.text(0.02, 0.98, summary, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle('The Catenary Curve: Shape of a Hanging Chain (Experiment 70)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'catenary.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {output_path}")

    # Print analysis
    print("\nCatenary Analysis:")
    print("-" * 50)

    print("\nCatenary vs Parabola comparison (a = 2):")
    x_test = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    for x in x_test:
        y_c = catenary(x, 2.0)
        y_p = parabola_approx(x, 2.0)
        diff = abs(y_c - y_p)
        pct = 100 * diff / y_c if y_c > 0 else 0
        print(f"  x = {x:.1f}: catenary = {y_c:.4f}, parabola = {y_p:.4f}, "
              f"diff = {diff:.4f} ({pct:.1f}%)")

    print(f"\nSimulated chain:")
    print(f"  Arc length: {L_sim:.3f}")
    print(f"  Best-fit catenary parameter a: {a_fit:.3f}")

    # Calculate sag
    sag = y1 - np.min(positions[:, 1])
    span = x2 - x1
    print(f"  Span: {span:.1f}, Sag: {sag:.3f}, Sag/Span ratio: {sag/span:.3f}")


if __name__ == "__main__":
    main()
