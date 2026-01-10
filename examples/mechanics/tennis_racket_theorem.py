"""
Example demonstrating the Tennis Racket Theorem (Intermediate Axis Theorem).

The Tennis Racket Theorem (also known as Dzhanibekov effect) states that
rotation of a rigid body about its intermediate principal axis is unstable,
while rotation about the axes with maximum and minimum moments of inertia
is stable.

For a rigid body with principal moments of inertia I1 < I2 < I3:
- Rotation about axis 1 (smallest I): STABLE
- Rotation about axis 2 (intermediate I): UNSTABLE
- Rotation about axis 3 (largest I): STABLE

This counterintuitive result explains why a tennis racket flipped in the
air will unexpectedly flip 180 degrees about its intermediate axis.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def euler_equations(omega, I):
    """
    Euler's equations for torque-free rigid body rotation.

    d(omega)/dt = I^(-1) * (I*omega x omega)

    where omega is the angular velocity and I is the inertia tensor.
    """
    I1, I2, I3 = I[0, 0], I[1, 1], I[2, 2]

    domega1 = (I2 - I3) / I1 * omega[1] * omega[2]
    domega2 = (I3 - I1) / I2 * omega[2] * omega[0]
    domega3 = (I1 - I2) / I3 * omega[0] * omega[1]

    return np.array([domega1, domega2, domega3])


def simulate_rigid_body_rotation(I, omega0, t_final, dt):
    """
    Simulate torque-free rigid body rotation using RK4.

    Args:
        I: Diagonal inertia tensor (3x3)
        omega0: Initial angular velocity
        t_final: Simulation duration
        dt: Time step

    Returns:
        Dictionary with time history of angular velocity and body orientation
    """
    omega = np.array(omega0, dtype=float)

    times = [0.0]
    omega_history = [omega.copy()]

    # Track body orientation using rotation matrix
    R = np.eye(3)  # Body frame to space frame
    R_history = [R.copy()]

    # Calculate conserved quantities
    L = I @ omega  # Angular momentum (constant in space frame)
    E = 0.5 * omega @ I @ omega  # Rotational kinetic energy

    L_history = [np.linalg.norm(L)]
    E_history = [E]

    t = 0
    while t < t_final:
        # RK4 for omega
        k1 = euler_equations(omega, I)
        k2 = euler_equations(omega + 0.5*dt*k1, I)
        k3 = euler_equations(omega + 0.5*dt*k2, I)
        k4 = euler_equations(omega + dt*k3, I)

        omega = omega + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

        # Update rotation matrix
        # Using exponential map approximation for small dt
        omega_mag = np.linalg.norm(omega)
        if omega_mag > 1e-10:
            axis = omega / omega_mag
            angle = omega_mag * dt

            # Rodrigues' rotation formula
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            dR = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
            R = R @ dR

        t += dt
        times.append(t)
        omega_history.append(omega.copy())
        R_history.append(R.copy())

        # Recalculate conserved quantities (should be constant)
        L = I @ omega
        E = 0.5 * omega @ I @ omega
        L_history.append(np.linalg.norm(L))
        E_history.append(E)

    return {
        'times': np.array(times),
        'omega': np.array(omega_history),
        'R': np.array(R_history),
        'L': np.array(L_history),
        'E': np.array(E_history)
    }


def track_body_axis(R_history, axis_index):
    """
    Track the orientation of a body axis in space frame.

    Args:
        R_history: History of rotation matrices
        axis_index: Which body axis to track (0, 1, or 2)

    Returns:
        Array of axis directions in space frame
    """
    body_axis = np.zeros(3)
    body_axis[axis_index] = 1.0

    space_axes = np.array([R @ body_axis for R in R_history])
    return space_axes


def main():
    # Define asymmetric rigid body (like a tennis racket or book)
    # I1 < I2 < I3
    I1, I2, I3 = 1.0, 2.0, 3.0
    I = np.diag([I1, I2, I3])

    # Simulation parameters
    t_final = 30.0
    dt = 0.001

    # Initial angular velocities with small perturbations
    omega_magnitude = 5.0
    perturbation = 0.1

    fig = plt.figure(figsize=(18, 12))

    # --- Rotation about SMALLEST moment of inertia (STABLE) ---
    ax1 = fig.add_subplot(2, 3, 1)

    omega0 = np.array([omega_magnitude, perturbation, perturbation])
    result = simulate_rigid_body_rotation(I, omega0, t_final, dt)

    ax1.plot(result['times'], result['omega'][:, 0], 'r-', lw=1.5, label='omega_1 (main)')
    ax1.plot(result['times'], result['omega'][:, 1], 'g-', lw=1.5, label='omega_2')
    ax1.plot(result['times'], result['omega'][:, 2], 'b-', lw=1.5, label='omega_3')

    ax1.set_xlabel('Time')
    ax1.set_ylabel('Angular Velocity')
    ax1.set_title(f'Rotation about Axis 1 (I1={I1}, smallest)\nSTABLE')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Rotation about INTERMEDIATE moment of inertia (UNSTABLE) ---
    ax2 = fig.add_subplot(2, 3, 2)

    omega0 = np.array([perturbation, omega_magnitude, perturbation])
    result_unstable = simulate_rigid_body_rotation(I, omega0, t_final, dt)

    ax2.plot(result_unstable['times'], result_unstable['omega'][:, 0], 'r-', lw=1.5, label='omega_1')
    ax2.plot(result_unstable['times'], result_unstable['omega'][:, 1], 'g-', lw=1.5, label='omega_2 (main)')
    ax2.plot(result_unstable['times'], result_unstable['omega'][:, 2], 'b-', lw=1.5, label='omega_3')

    ax2.set_xlabel('Time')
    ax2.set_ylabel('Angular Velocity')
    ax2.set_title(f'Rotation about Axis 2 (I2={I2}, intermediate)\nUNSTABLE - Tennis Racket Effect!')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- Rotation about LARGEST moment of inertia (STABLE) ---
    ax3 = fig.add_subplot(2, 3, 3)

    omega0 = np.array([perturbation, perturbation, omega_magnitude])
    result = simulate_rigid_body_rotation(I, omega0, t_final, dt)

    ax3.plot(result['times'], result['omega'][:, 0], 'r-', lw=1.5, label='omega_1')
    ax3.plot(result['times'], result['omega'][:, 1], 'g-', lw=1.5, label='omega_2')
    ax3.plot(result['times'], result['omega'][:, 2], 'b-', lw=1.5, label='omega_3 (main)')

    ax3.set_xlabel('Time')
    ax3.set_ylabel('Angular Velocity')
    ax3.set_title(f'Rotation about Axis 3 (I3={I3}, largest)\nSTABLE')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # --- Phase space for unstable rotation ---
    ax4 = fig.add_subplot(2, 3, 4)

    # Plot phase portrait omega_1 vs omega_3 for different initial conditions
    colors = plt.cm.viridis(np.linspace(0, 1, 5))

    for i, pert in enumerate([0.05, 0.1, 0.2, 0.3, 0.5]):
        omega0 = np.array([pert * omega_magnitude, omega_magnitude, pert * omega_magnitude])
        result = simulate_rigid_body_rotation(I, omega0, t_final, dt)

        ax4.plot(result['omega'][:, 0], result['omega'][:, 2],
                 color=colors[i], lw=0.8, alpha=0.7,
                 label=f'pert = {pert}')

    ax4.set_xlabel('omega_1')
    ax4.set_ylabel('omega_3')
    ax4.set_title('Phase Space (omega_1 vs omega_3)\nfor rotation near intermediate axis')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal')

    # --- Track body axis orientation (unstable case) ---
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')

    omega0 = np.array([perturbation, omega_magnitude, perturbation])
    result = simulate_rigid_body_rotation(I, omega0, t_final, dt)

    # Track the intermediate axis (axis 2) in space frame
    axis2_space = track_body_axis(result['R'], 1)

    # Subsample for clearer visualization
    step = 50
    ax5.plot(axis2_space[::step, 0], axis2_space[::step, 1], axis2_space[::step, 2],
             'b-', lw=1, alpha=0.7)

    # Draw unit sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax5.plot_surface(x, y, z, alpha=0.1, color='gray')

    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    ax5.set_zlabel('Z')
    ax5.set_title('Intermediate Axis Trajectory on Unit Sphere\n(Body axis 2 traced in space frame)')

    # --- Conservation laws check ---
    ax6 = fig.add_subplot(2, 3, 6)

    omega0 = np.array([perturbation, omega_magnitude, perturbation])
    result = simulate_rigid_body_rotation(I, omega0, t_final, dt)

    L_norm = result['L'] / result['L'][0]
    E_norm = result['E'] / result['E'][0]

    ax6.plot(result['times'], L_norm, 'b-', lw=1.5, label='|L| / |L_0|')
    ax6.plot(result['times'], E_norm, 'r--', lw=1.5, label='E / E_0')
    ax6.axhline(y=1.0, color='black', linestyle=':', alpha=0.5)

    ax6.set_xlabel('Time')
    ax6.set_ylabel('Normalized Value')
    ax6.set_title('Conservation Laws\n(Angular Momentum and Energy)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(0.98, 1.02)

    plt.suptitle('Tennis Racket Theorem (Intermediate Axis Theorem)\n'
                 f'Inertia: I1={I1} < I2={I2} < I3={I3}',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'tennis_racket_theorem.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'tennis_racket_theorem.png')}")

    # Create additional figure showing flip period
    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Flip dynamics visualization
    omega0 = np.array([0.1, 5.0, 0.1])
    result = simulate_rigid_body_rotation(I, omega0, t_final, dt)

    axes[0].plot(result['times'], result['omega'][:, 1], 'g-', lw=2)
    axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('omega_2 (intermediate axis)')
    axes[0].set_title('Sign Flips of Intermediate Axis Angular Velocity\n'
                      'Each zero crossing = 180 degree flip')
    axes[0].grid(True, alpha=0.3)

    # Polhode (angular velocity trajectory in body frame)
    axes[1].plot(result['omega'][:, 0], result['omega'][:, 2], 'b-', lw=0.5)
    axes[1].set_xlabel('omega_1')
    axes[1].set_ylabel('omega_3')
    axes[1].set_title('Polhode: Angular Velocity in Body Frame')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tennis_racket_flips.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'tennis_racket_flips.png')}")

    # Print analysis
    print("\nTennis Racket Theorem Analysis:")
    print("-" * 50)
    print(f"Moments of inertia: I1={I1}, I2={I2}, I3={I3}")
    print(f"\nStability:")
    print(f"  Rotation about axis 1 (I1={I1}, smallest): STABLE")
    print(f"  Rotation about axis 2 (I2={I2}, intermediate): UNSTABLE")
    print(f"  Rotation about axis 3 (I3={I3}, largest): STABLE")
    print(f"\nThe instability about the intermediate axis is caused by")
    print(f"exponential growth of perturbations in the other two axes.")


if __name__ == "__main__":
    main()
