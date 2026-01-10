"""
Example demonstrating the restricted three-body problem and Lagrange points.

This example visualizes the five Lagrange points in a circular restricted
three-body problem, shows the effective potential in the rotating frame,
and demonstrates particle motion near the Lagrange points (stable vs unstable).

The five Lagrange points are:
- L1: Between the two bodies
- L2: Beyond the smaller body
- L3: Beyond the larger body
- L4: Leading equilateral point (60 degrees ahead)
- L5: Trailing equilateral point (60 degrees behind)

L4 and L5 are stable for mass ratio mu < 0.0385 (Routh's criterion).
L1, L2, L3 are always unstable but useful for space missions.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.orbital import ThreeBodyProblem


def simulate_test_particle(tbp, initial_pos, initial_vel, t_final, dt):
    """
    Simulate a test particle in the rotating frame of the three-body problem.

    Args:
        tbp: ThreeBodyProblem instance
        initial_pos: Initial position in rotating frame
        initial_vel: Initial velocity in rotating frame
        t_final: Total simulation time
        dt: Time step

    Returns:
        Dictionary with trajectory data
    """
    r = np.array(initial_pos, dtype=float)
    v = np.array(initial_vel, dtype=float)

    positions = [r.copy()]
    velocities = [v.copy()]
    times = [0.0]
    jacobi_constants = [tbp.jacobi_constant(r, v)]

    t = 0
    while t < t_final:
        # RK4 integration in rotating frame
        k1_v = tbp.acceleration_rotating(r, v)
        k1_r = v

        k2_v = tbp.acceleration_rotating(r + 0.5*dt*k1_r, v + 0.5*dt*k1_v)
        k2_r = v + 0.5*dt*k1_v

        k3_v = tbp.acceleration_rotating(r + 0.5*dt*k2_r, v + 0.5*dt*k2_v)
        k3_r = v + 0.5*dt*k2_v

        k4_v = tbp.acceleration_rotating(r + dt*k3_r, v + dt*k3_v)
        k4_r = v + dt*k3_v

        v = v + (dt/6) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        r = r + (dt/6) * (k1_r + 2*k2_r + 2*k3_r + k4_r)

        t += dt
        positions.append(r.copy())
        velocities.append(v.copy())
        times.append(t)
        jacobi_constants.append(tbp.jacobi_constant(r, v))

    return {
        'positions': np.array(positions),
        'velocities': np.array(velocities),
        'times': np.array(times),
        'jacobi': np.array(jacobi_constants)
    }


def main():
    # Create a Sun-Earth like system (normalized units)
    # Mass ratio mu = m2/(m1+m2) ~ 3e-6 for Sun-Earth
    # Using larger ratio for visualization
    G = 1.0
    m1 = 1.0  # Primary (Sun-like)
    m2 = 0.01  # Secondary (Earth-like, exaggerated mass)
    R = 1.0  # Separation

    tbp = ThreeBodyProblem(mass1=m1, mass2=m2, separation=R, G=G)

    # Get Lagrange point positions
    L_points = {}
    for i in range(1, 6):
        L_points[f'L{i}'] = tbp.lagrange_point(i)

    print("Lagrange Point Positions (rotating frame):")
    print("-" * 40)
    for name, pos in L_points.items():
        print(f"  {name}: ({pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f})")
    print(f"\nMass ratio mu = {tbp.mu:.4f}")
    print(f"Primary at: ({tbp.r1[0]:+.4f}, 0, 0)")
    print(f"Secondary at: ({tbp.r2[0]:+.4f}, 0, 0)")

    fig = plt.figure(figsize=(18, 12))

    # Plot 1: Lagrange points and effective potential contours
    ax1 = fig.add_subplot(2, 3, 1)

    # Create grid for effective potential
    x = np.linspace(-1.5, 1.5, 200)
    y = np.linspace(-1.5, 1.5, 200)
    X, Y = np.meshgrid(x, y)
    U = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            pos = np.array([X[i, j], Y[i, j], 0.0])
            U[i, j] = tbp.effective_potential(pos)

    # Clip extreme values for better visualization
    U = np.clip(U, -10, 0)

    # Plot potential contours
    levels = np.linspace(-5, -0.5, 30)
    contour = ax1.contourf(X, Y, U, levels=levels, cmap='viridis', alpha=0.7)
    ax1.contour(X, Y, U, levels=levels, colors='white', linewidths=0.3, alpha=0.5)

    # Mark primaries
    ax1.plot(tbp.r1[0], tbp.r1[1], 'yo', markersize=20, label='Primary (M1)')
    ax1.plot(tbp.r2[0], tbp.r2[1], 'co', markersize=10, label='Secondary (M2)')

    # Mark Lagrange points
    colors = {'L1': 'red', 'L2': 'orange', 'L3': 'magenta', 'L4': 'lime', 'L5': 'lime'}
    for name, pos in L_points.items():
        ax1.plot(pos[0], pos[1], '*', color=colors[name], markersize=15,
                 markeredgecolor='black', markeredgewidth=0.5, label=name)

    ax1.set_xlabel('x (rotating frame)')
    ax1.set_ylabel('y (rotating frame)')
    ax1.set_title('Effective Potential and Lagrange Points')
    ax1.set_aspect('equal')
    ax1.legend(loc='upper right', fontsize=7)
    plt.colorbar(contour, ax=ax1, label='Effective Potential')

    # Plot 2: Zero-velocity curves (Hill's regions)
    ax2 = fig.add_subplot(2, 3, 2)

    # Calculate Jacobi constants at Lagrange points
    C_L = {}
    for name, pos in L_points.items():
        C_L[name] = tbp.jacobi_constant(pos, np.zeros(3))

    # Plot zero-velocity curves for different energy levels
    C_values = [C_L['L1'], C_L['L2'], C_L['L3'], (C_L['L4'] + C_L['L5'])/2]
    C_labels = ['C(L1)', 'C(L2)', 'C(L3)', 'C(L4,L5)']

    for C, label in zip(C_values, C_labels):
        ax2.contour(X, Y, -2*U, levels=[C], colors='blue', linewidths=1.5)

    # Mark features
    ax2.plot(tbp.r1[0], tbp.r1[1], 'yo', markersize=15)
    ax2.plot(tbp.r2[0], tbp.r2[1], 'co', markersize=8)
    for name, pos in L_points.items():
        ax2.plot(pos[0], pos[1], '*', color=colors[name], markersize=12,
                 markeredgecolor='black', markeredgewidth=0.5)

    ax2.set_xlabel('x (rotating frame)')
    ax2.set_ylabel('y (rotating frame)')
    ax2.set_title('Zero-Velocity Curves (Hill Regions)')
    ax2.set_aspect('equal')
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Motion near unstable L1 point
    ax3 = fig.add_subplot(2, 3, 3)

    L1 = L_points['L1']
    T_orbit = 2 * np.pi / tbp.omega  # Orbital period

    # Small perturbations from L1
    perturbations = [
        (0.001, 0.0, 'red', 'x-perturbation'),
        (0.0, 0.001, 'blue', 'y-perturbation'),
        (0.001, 0.001, 'green', 'xy-perturbation')
    ]

    for dx, dy, color, label in perturbations:
        initial_pos = L1 + np.array([dx, dy, 0])
        result = simulate_test_particle(tbp, initial_pos, np.zeros(3),
                                        t_final=2*T_orbit, dt=T_orbit/500)
        pos = result['positions']
        ax3.plot(pos[:, 0] - L1[0], pos[:, 1] - L1[1], color=color,
                 lw=1.5, label=label, alpha=0.8)

    ax3.plot(0, 0, 'r*', markersize=15, label='L1')
    ax3.set_xlabel('x - L1_x')
    ax3.set_ylabel('y - L1_y')
    ax3.set_title('Motion Near L1 (Unstable)\nSmall perturbations grow exponentially')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')

    # Plot 4: Motion near stable L4 point
    ax4 = fig.add_subplot(2, 3, 4)

    L4 = L_points['L4']

    # Small perturbations from L4
    perturbations = [
        (0.01, 0.0, 'red', 'x-perturbation'),
        (0.0, 0.01, 'blue', 'y-perturbation'),
        (0.01, 0.01, 'green', 'xy-perturbation')
    ]

    for dx, dy, color, label in perturbations:
        initial_pos = L4 + np.array([dx, dy, 0])
        result = simulate_test_particle(tbp, initial_pos, np.zeros(3),
                                        t_final=10*T_orbit, dt=T_orbit/200)
        pos = result['positions']
        ax4.plot(pos[:, 0] - L4[0], pos[:, 1] - L4[1], color=color,
                 lw=1, label=label, alpha=0.7)

    ax4.plot(0, 0, 'g*', markersize=15, label='L4')
    ax4.set_xlabel('x - L4_x')
    ax4.set_ylabel('y - L4_y')
    ax4.set_title('Motion Near L4 (Stable)\nPerturbations lead to bounded tadpole orbits')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal')

    # Plot 5: Tadpole and horseshoe orbits
    ax5 = fig.add_subplot(2, 3, 5)

    # Tadpole orbit (small amplitude around L4)
    initial_pos = L4 + np.array([0.02, 0.0, 0])
    initial_vel = np.array([0.0, 0.01, 0])
    result = simulate_test_particle(tbp, initial_pos, initial_vel,
                                    t_final=30*T_orbit, dt=T_orbit/200)
    pos = result['positions']
    ax5.plot(pos[:, 0], pos[:, 1], 'b-', lw=0.8, label='Tadpole orbit', alpha=0.7)

    # Mark primaries and L-points
    ax5.plot(tbp.r1[0], tbp.r1[1], 'yo', markersize=15)
    ax5.plot(tbp.r2[0], tbp.r2[1], 'co', markersize=8)
    for name, lp in L_points.items():
        ax5.plot(lp[0], lp[1], '*', color=colors[name], markersize=10,
                 markeredgecolor='black', markeredgewidth=0.5)

    ax5.set_xlabel('x (rotating frame)')
    ax5.set_ylabel('y (rotating frame)')
    ax5.set_title('Tadpole Orbit Near L4')
    ax5.set_aspect('equal')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(-1.5, 1.5)
    ax5.set_ylim(-1.5, 1.5)

    # Plot 6: Jacobi constant conservation
    ax6 = fig.add_subplot(2, 3, 6)

    # Simulate a more complex trajectory
    initial_pos = L_points['L1'] + np.array([0.1, 0.05, 0])
    initial_vel = np.array([0.0, 0.05, 0])
    result = simulate_test_particle(tbp, initial_pos, initial_vel,
                                    t_final=5*T_orbit, dt=T_orbit/500)

    times = result['times'] / T_orbit
    jacobi = result['jacobi'] / result['jacobi'][0]

    ax6.plot(times, jacobi, 'b-', lw=1.5)
    ax6.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
    ax6.set_xlabel('Time / Orbital Period')
    ax6.set_ylabel('Jacobi Constant / Initial')
    ax6.set_title('Jacobi Constant Conservation')
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(0.999, 1.001)

    plt.suptitle('Restricted Three-Body Problem: Lagrange Points\n'
                 f'Mass ratio mu = {tbp.mu:.4f}', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'lagrange_points.png'),
                dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {os.path.join(output_dir, 'lagrange_points.png')}")

    # Print Jacobi constants at Lagrange points
    print("\nJacobi Constants at Lagrange Points:")
    print("-" * 40)
    for name, C in sorted(C_L.items()):
        print(f"  {name}: C = {C:.4f}")
    print(f"\nL4 and L5 have equal Jacobi constants (both are equilateral)")
    print(f"Stability: L4, L5 stable for mu < 0.0385 (current mu = {tbp.mu:.4f})")


if __name__ == "__main__":
    main()
