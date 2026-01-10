"""
Experiment 64: Liouville's Theorem.

Demonstrates Liouville's theorem: the phase space volume (density)
is conserved under Hamiltonian time evolution.

Key concepts:
1. Phase space density is incompressible: d(rho)/dt = 0
2. Equivalent: divergence of flow vanishes in phase space
3. Phase space volume of any region is constant
4. Foundation of statistical mechanics

div(v) = sum_i (d(q_dot_i)/dq_i + d(p_dot_i)/dp_i) = 0
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


def simulate_ensemble(H_derivatives, initial_states, t_span, dt):
    """
    Evolve an ensemble of initial conditions under Hamiltonian dynamics.

    Args:
        H_derivatives: Function returning (dq/dt, dp/dt) given (q, p)
        initial_states: Array of shape (N, 2) with initial (q, p) pairs
        t_span: (t_start, t_end)
        dt: Time step

    Returns:
        Dictionary with trajectory data
    """
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / dt)
    n_particles = len(initial_states)

    times = np.linspace(t_start, t_end, n_steps)
    trajectories = np.zeros((n_steps, n_particles, 2))
    trajectories[0] = initial_states

    for i in range(1, n_steps):
        for j in range(n_particles):
            state = trajectories[i-1, j]

            # RK4 step
            k1 = np.array(H_derivatives(state[0], state[1]))
            k2 = np.array(H_derivatives(state[0] + 0.5*dt*k1[0],
                                        state[1] + 0.5*dt*k1[1]))
            k3 = np.array(H_derivatives(state[0] + 0.5*dt*k2[0],
                                        state[1] + 0.5*dt*k2[1]))
            k4 = np.array(H_derivatives(state[0] + dt*k3[0],
                                        state[1] + dt*k3[1]))

            trajectories[i, j] = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

    return {
        'times': times,
        'trajectories': trajectories
    }


def polygon_area(vertices):
    """Calculate area of polygon using shoelace formula."""
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * abs(np.sum(x[:-1]*y[1:] - x[1:]*y[:-1]) +
                     x[-1]*y[0] - x[0]*y[-1])


def convex_hull_area(points):
    """Calculate area of convex hull of points."""
    from scipy.spatial import ConvexHull
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, 'volume' is actually area
    except Exception:
        return 0.0


# --- Hamiltonian Systems ---

def sho_derivatives(q, p, m=1.0, k=1.0):
    """Simple harmonic oscillator: H = p^2/(2m) + k*q^2/2"""
    dq_dt = p / m
    dp_dt = -k * q
    return dq_dt, dp_dt


def pendulum_derivatives(q, p, m=1.0, L=1.0, g=9.81):
    """Pendulum: H = p^2/(2mL^2) - mgL*cos(q)"""
    dq_dt = p / (m * L**2)
    dp_dt = -m * g * L * np.sin(q)
    return dq_dt, dp_dt


def duffing_derivatives(q, p, m=1.0, alpha=1.0, beta=0.1):
    """Duffing oscillator: H = p^2/(2m) + alpha*q^2/2 + beta*q^4/4"""
    dq_dt = p / m
    dp_dt = -alpha * q - beta * q**3
    return dq_dt, dp_dt


def henon_heiles_derivatives(state, a=1.0, b=1.0):
    """
    Henon-Heiles system (2 DOF).

    H = (px^2 + py^2)/2 + (x^2 + y^2)/2 + x^2*y - y^3/3
    """
    x, y, px, py = state
    dx_dt = px
    dy_dt = py
    dpx_dt = -x - 2*a*x*y
    dpy_dt = -y - a*x**2 + b*y**2
    return np.array([dx_dt, dy_dt, dpx_dt, dpy_dt])


def main():
    fig = plt.figure(figsize=(16, 12))

    # --- Plot 1: Phase space flow for SHO ---
    ax1 = fig.add_subplot(2, 3, 1)

    # Create a blob of initial conditions
    n_points = 200
    q_center, p_center = 1.5, 0.5
    radius = 0.3

    theta = np.random.uniform(0, 2*np.pi, n_points)
    r = np.sqrt(np.random.uniform(0, 1, n_points)) * radius
    q_init = q_center + r * np.cos(theta)
    p_init = p_center + r * np.sin(theta)

    initial_states = np.column_stack([q_init, p_init])

    # Evolve
    def sho_deriv(q, p):
        return sho_derivatives(q, p, m=1.0, k=1.0)

    result = simulate_ensemble(sho_deriv, initial_states, (0, 10), 0.01)

    # Plot at different times
    times_to_plot = [0, 2.5, 5.0, 7.5]
    colors = ['blue', 'green', 'orange', 'red']

    for t, color in zip(times_to_plot, colors):
        idx = int(t / 0.01)
        points = result['trajectories'][idx]
        ax1.scatter(points[:, 0], points[:, 1], s=5, c=color, alpha=0.5,
                    label=f't = {t}')

    ax1.set_xlabel('q (position)')
    ax1.set_ylabel('p (momentum)')
    ax1.set_title('SHO: Ensemble Evolution\n(Shape changes, area preserved)')
    ax1.legend(fontsize=8)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Area conservation ---
    ax2 = fig.add_subplot(2, 3, 2)

    # Track convex hull area over time
    times = result['times']
    areas = []

    for i in range(len(times)):
        points = result['trajectories'][i]
        area = convex_hull_area(points)
        areas.append(area)

    ax2.plot(times, np.array(areas) / areas[0], 'b-', lw=2)
    ax2.axhline(1.0, color='r', linestyle='--', lw=2, label='Expected (constant)')

    ax2.set_xlabel('Time')
    ax2.set_ylabel('Area / Initial Area')
    ax2.set_title("Liouville's Theorem: Area Conservation\n(Convex hull area)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.9, 1.1)

    # --- Plot 3: Pendulum (nonlinear) ---
    ax3 = fig.add_subplot(2, 3, 3)

    # Pendulum ensemble
    q_center, p_center = 0.5, 2.0
    q_init = q_center + r * np.cos(theta)
    p_init = p_center + r * np.sin(theta)

    initial_states_pend = np.column_stack([q_init, p_init])

    def pend_deriv(q, p):
        return pendulum_derivatives(q, p, m=1.0, L=1.0, g=1.0)

    result_pend = simulate_ensemble(pend_deriv, initial_states_pend, (0, 15), 0.01)

    # Plot evolution
    for t, color in zip([0, 3.0, 7.5, 12.0], colors):
        idx = min(int(t / 0.01), len(result_pend['times'])-1)
        points = result_pend['trajectories'][idx]
        ax3.scatter(points[:, 0], points[:, 1], s=5, c=color, alpha=0.5,
                    label=f't = {t}')

    ax3.set_xlabel('theta (angle)')
    ax3.set_ylabel('p_theta (angular momentum)')
    ax3.set_title('Pendulum: Phase Space Evolution\n(Stretching/folding, area preserved)')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # --- Plot 4: Divergence of flow field ---
    ax4 = fig.add_subplot(2, 3, 4)

    # For Hamiltonian flow: div(v) = d(q_dot)/dq + d(p_dot)/dp = 0

    q_range = np.linspace(-2, 2, 20)
    p_range = np.linspace(-2, 2, 20)
    Q, P = np.meshgrid(q_range, p_range)

    # SHO flow field
    dQ, dP = sho_derivatives(Q, P)

    # Compute divergence numerically
    h = q_range[1] - q_range[0]
    div_v = np.zeros_like(Q)

    for i in range(1, Q.shape[0]-1):
        for j in range(1, Q.shape[1]-1):
            dqdot_dq = (sho_derivatives(Q[i,j]+h, P[i,j])[0] -
                        sho_derivatives(Q[i,j]-h, P[i,j])[0]) / (2*h)
            dpdot_dp = (sho_derivatives(Q[i,j], P[i,j]+h)[1] -
                        sho_derivatives(Q[i,j], P[i,j]-h)[1]) / (2*h)
            div_v[i,j] = dqdot_dq + dpdot_dp

    # Streamplot
    ax4.streamplot(Q, P, dQ, dP, density=1.5, color='blue', linewidth=0.5)

    # Show divergence (should be zero)
    div_max = np.max(np.abs(div_v[1:-1, 1:-1]))

    ax4.set_xlabel('q')
    ax4.set_ylabel('p')
    ax4.set_title(f'Hamiltonian Flow (SHO)\ndiv(v) = {div_max:.2e} (should be 0)')
    ax4.set_xlim(-2, 2)
    ax4.set_ylim(-2, 2)
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)

    # --- Plot 5: Compare Hamiltonian vs dissipative ---
    ax5 = fig.add_subplot(2, 3, 5)

    # Damped oscillator (NOT Hamiltonian)
    def damped_derivatives(q, p, m=1.0, k=1.0, gamma=0.3):
        dq_dt = p / m
        dp_dt = -k * q - gamma * p  # Damping term
        return dq_dt, dp_dt

    # Initial ensemble
    q_init = q_center + r * np.cos(theta)
    p_init = p_center + r * np.sin(theta)
    initial_states_damp = np.column_stack([q_init, p_init])

    # Simulate both
    def damp_deriv(q, p):
        return damped_derivatives(q, p)

    result_damp = simulate_ensemble(damp_deriv, initial_states_damp, (0, 10), 0.01)

    # Calculate areas over time
    areas_ham = []
    areas_damp = []

    for i in range(len(result['times'])):
        areas_ham.append(convex_hull_area(result['trajectories'][i]))
        areas_damp.append(convex_hull_area(result_damp['trajectories'][i]))

    ax5.plot(result['times'], np.array(areas_ham) / areas_ham[0],
             'b-', lw=2, label='Hamiltonian (SHO)')
    ax5.plot(result_damp['times'], np.array(areas_damp) / areas_damp[0],
             'r-', lw=2, label='Dissipative (damped)')
    ax5.axhline(1.0, color='k', linestyle='--', alpha=0.5)

    ax5.set_xlabel('Time')
    ax5.set_ylabel('Area / Initial Area')
    ax5.set_title('Hamiltonian vs Dissipative Systems\n(Only Hamiltonian preserves area)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, 1.2)

    # --- Plot 6: Theory summary ---
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    summary = """Liouville's Theorem
===================

STATEMENT:
----------
The phase space distribution function rho(q, p, t)
satisfies the Liouville equation:

    d(rho)/dt = drho/dt + {rho, H} = 0

This means rho is constant along phase trajectories.

EQUIVALENT STATEMENTS:
----------------------
1. Phase space volume is conserved
2. Phase flow is incompressible
3. Divergence of flow vanishes:
   div(v) = d(q_dot)/dq + d(p_dot)/dp = 0

PROOF (for 1 DOF):
------------------
q_dot = dH/dp     =>  d(q_dot)/dq = d^2H/(dq dp)
p_dot = -dH/dq    =>  d(p_dot)/dp = -d^2H/(dp dq)

Sum = 0  (mixed partials cancel!)

IMPLICATIONS:
-------------
1. Cannot have attractors in Hamiltonian systems
2. Microcanonical ensemble is stationary
3. Entropy is constant (information preserved)
4. Recurrence: system returns close to initial state

NON-HAMILTONIAN SYSTEMS:
------------------------
Dissipative: Phase space volume SHRINKS
             (trajectories converge to attractors)

Expanding:   Volume grows
             (unstable, energy input)

STATISTICAL MECHANICS:
----------------------
Liouville's theorem justifies:
- Microcanonical ensemble (energy shell)
- Equal a priori probabilities
- Ergodic hypothesis connection

The theorem is the classical analog of
unitary evolution in quantum mechanics!"""

    ax6.text(0.02, 0.98, summary, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))

    plt.suptitle("Liouville's Theorem: Phase Space Volume Conservation (Experiment 64)",
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'liouville_theorem.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {output_path}")

    # Print numerical verification
    print("\nLiouville's Theorem Verification:")
    print("-" * 50)

    print("\nPhase space area conservation (SHO):")
    print(f"  Initial area: {areas_ham[0]:.4f}")
    print(f"  Final area: {areas_ham[-1]:.4f}")
    print(f"  Ratio: {areas_ham[-1]/areas_ham[0]:.6f}")

    print("\nPhase space area (Damped oscillator):")
    print(f"  Initial area: {areas_damp[0]:.4f}")
    print(f"  Final area: {areas_damp[-1]:.4f}")
    print(f"  Ratio: {areas_damp[-1]/areas_damp[0]:.6f}")
    print("  (Area shrinks due to dissipation)")

    print("\nDivergence of Hamiltonian flow:")
    print(f"  Max |div(v)|: {div_max:.2e}")
    print("  (Should be exactly 0 for Hamiltonian systems)")


if __name__ == "__main__":
    main()
