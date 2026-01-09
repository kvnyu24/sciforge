"""
Example demonstrating rigid body rotation dynamics.

This example shows free rotation of a rigid body (torque-free motion),
demonstrating conservation of angular momentum and energy.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.mechanics import RigidBody


def simulate_free_rotation(I, omega0, t_final, dt):
    """
    Simulate free rotation of a rigid body.

    Args:
        I: Moment of inertia tensor (3x3 diagonal)
        omega0: Initial angular velocity vector
        t_final: Simulation duration (s)
        dt: Time step (s)

    Returns:
        Dictionary with time, angular velocity, and energy data
    """
    body = RigidBody(
        mass=1.0,
        position=np.array([0.0, 0.0, 0.0]),
        velocity=np.array([0.0, 0.0, 0.0]),
        inertia_tensor=I,
        angular_velocity=omega0
    )

    times = [0]
    omega_x = [omega0[0]]
    omega_y = [omega0[1]]
    omega_z = [omega0[2]]
    angular_momentum = [np.linalg.norm(I @ omega0)]
    kinetic_energy = [0.5 * omega0 @ I @ omega0]

    t = 0
    while t < t_final:
        # Free rotation: no external torque
        body.update(force=np.array([0.0, 0.0, 0.0]),
                   torque=np.array([0.0, 0.0, 0.0]), dt=dt)
        t += dt

        omega = body.angular_velocity
        times.append(t)
        omega_x.append(omega[0])
        omega_y.append(omega[1])
        omega_z.append(omega[2])
        angular_momentum.append(np.linalg.norm(I @ omega))
        kinetic_energy.append(0.5 * omega @ I @ omega)

    return {
        'time': np.array(times),
        'omega_x': np.array(omega_x),
        'omega_y': np.array(omega_y),
        'omega_z': np.array(omega_z),
        'L': np.array(angular_momentum),
        'E': np.array(kinetic_energy)
    }


def main():
    # Asymmetric rigid body (like a book or phone)
    # Different moments of inertia along each axis
    I1, I2, I3 = 1.0, 2.0, 3.0  # Principal moments
    I = np.diag([I1, I2, I3])

    # Three cases: rotation primarily about each principal axis
    cases = {
        'About smallest I (stable)': np.array([10.0, 0.1, 0.1]),
        'About middle I (unstable)': np.array([0.1, 10.0, 0.1]),
        'About largest I (stable)': np.array([0.1, 0.1, 10.0])
    }

    t_final = 10.0
    dt = 0.001

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    colors = {'About smallest I (stable)': 'blue',
              'About middle I (unstable)': 'red',
              'About largest I (stable)': 'green'}

    for idx, (name, omega0) in enumerate(cases.items()):
        results = simulate_free_rotation(I, omega0, t_final, dt)

        # Plot angular velocity components
        ax1 = axes[0, idx]
        ax1.plot(results['time'], results['omega_x'], 'r-', label='ωx', alpha=0.8)
        ax1.plot(results['time'], results['omega_y'], 'g-', label='ωy', alpha=0.8)
        ax1.plot(results['time'], results['omega_z'], 'b-', label='ωz', alpha=0.8)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Angular Velocity (rad/s)')
        ax1.set_title(name)
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Plot conservation quantities
        ax2 = axes[1, idx]
        L_normalized = results['L'] / results['L'][0]
        E_normalized = results['E'] / results['E'][0]
        ax2.plot(results['time'], L_normalized, 'b-', label='|L|/|L₀|', lw=2)
        ax2.plot(results['time'], E_normalized, 'r--', label='E/E₀', lw=2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Normalized Value')
        ax2.set_title('Conservation Laws')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0.95, 1.05)

    plt.suptitle('Rigid Body Free Rotation (I₁=1, I₂=2, I₃=3 kg⋅m²)\n'
                 'Rotation about intermediate axis is unstable!',
                 fontsize=12, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'rigid_body_rotation.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'rigid_body_rotation.png')}")


if __name__ == "__main__":
    main()
