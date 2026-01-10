"""
Example demonstrating gyroscope precession and nutation.

A gyroscope is a spinning symmetric rigid body that exhibits remarkable
behavior when subjected to external torques. Key phenomena include:

1. Precession: The spin axis slowly rotates around the vertical when
   gravity exerts a torque. The precession rate is Omega = tau / (I*omega)
   where tau is the torque, I is the moment of inertia, and omega is
   the spin rate.

2. Nutation: Small oscillations superimposed on the precession, caused
   by the initial conditions not matching the steady precession state.

3. Stability: Fast spinning gyroscopes are more stable against tilting.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def simulate_gyroscope(I_spin, I_perp, mass, pivot_to_cm, spin_rate,
                       theta0, phi0, theta_dot0, phi_dot0,
                       t_final, dt, g=9.81):
    """
    Simulate gyroscope motion using Euler angles.

    Uses the Lagrangian formulation with Euler angles (phi, theta, psi):
    - phi: Precession angle (rotation about vertical)
    - theta: Nutation angle (tilt from vertical)
    - psi: Spin angle (rotation about symmetry axis)

    Args:
        I_spin: Moment of inertia about spin axis
        I_perp: Moment of inertia perpendicular to spin axis
        mass: Total mass
        pivot_to_cm: Distance from pivot to center of mass
        spin_rate: Constant spin rate (psi_dot)
        theta0: Initial tilt angle from vertical
        phi0: Initial precession angle
        theta_dot0: Initial rate of change of theta
        phi_dot0: Initial rate of change of phi
        t_final: Simulation duration
        dt: Time step
        g: Gravitational acceleration

    Returns:
        Dictionary with trajectory data
    """
    # State: [theta, phi, theta_dot, phi_dot]
    state = np.array([theta0, phi0, theta_dot0, phi_dot0], dtype=float)

    times = [0.0]
    theta_history = [theta0]
    phi_history = [phi0]
    theta_dot_history = [theta_dot0]
    phi_dot_history = [phi_dot0]

    def derivatives(s, t):
        """Calculate derivatives of state variables."""
        theta, phi, theta_dot, phi_dot = s

        # Prevent singularity at theta = 0
        sin_theta = np.sin(theta) if abs(theta) > 1e-6 else 1e-6
        cos_theta = np.cos(theta)

        # Effective potential derivative
        # E = 0.5*I_perp*(theta_dot^2 + sin^2(theta)*phi_dot^2)
        #   + 0.5*I_spin*(psi_dot + cos(theta)*phi_dot)^2
        #   + m*g*r*cos(theta)

        # Angular momentum about vertical (conserved when no friction)
        # L_z = I_perp * sin^2(theta) * phi_dot + I_spin * cos(theta) * (psi_dot + cos(theta)*phi_dot)

        # For constant spin (psi_dot = spin_rate), we can write equations of motion:

        # Effective angular momentum about z
        L3 = I_spin * spin_rate  # Spin angular momentum

        # Theta equation
        # I_perp * theta_ddot = I_perp * sin(theta)*cos(theta)*phi_dot^2
        #                     - L3 * sin(theta) * phi_dot
        #                     + m*g*r*sin(theta)

        theta_ddot = (sin_theta * cos_theta * phi_dot**2
                      - L3 * sin_theta * phi_dot / I_perp
                      + mass * g * pivot_to_cm * sin_theta / I_perp)

        # Phi equation (from conservation of vertical angular momentum)
        # For steady precession: phi_dot = m*g*r / (I_spin * psi_dot) = m*g*r / L3
        # For general motion, we use:
        # d/dt(I_perp * sin^2(theta) * phi_dot + L3 * cos(theta)) = 0

        # This gives:
        # phi_ddot = (2 * I_perp * sin(theta) * cos(theta) * theta_dot * phi_dot
        #           + L3 * sin(theta) * theta_dot) / (I_perp * sin^2(theta))

        if abs(sin_theta) > 1e-6:
            phi_ddot = (-2 * cos_theta * theta_dot * phi_dot / sin_theta
                        + L3 * theta_dot / (I_perp * sin_theta))
        else:
            phi_ddot = 0.0

        return np.array([theta_dot, phi_dot, theta_ddot, phi_ddot])

    t = 0
    while t < t_final:
        # RK4 integration
        k1 = derivatives(state, t)
        k2 = derivatives(state + 0.5*dt*k1, t + 0.5*dt)
        k3 = derivatives(state + 0.5*dt*k2, t + 0.5*dt)
        k4 = derivatives(state + dt*k3, t + dt)

        state = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

        t += dt
        times.append(t)
        theta_history.append(state[0])
        phi_history.append(state[1])
        theta_dot_history.append(state[2])
        phi_dot_history.append(state[3])

    return {
        'times': np.array(times),
        'theta': np.array(theta_history),
        'phi': np.array(phi_history),
        'theta_dot': np.array(theta_dot_history),
        'phi_dot': np.array(phi_dot_history)
    }


def spin_axis_trajectory(theta, phi):
    """
    Calculate the trajectory of the spin axis tip on unit sphere.

    Args:
        theta: Array of nutation angles
        phi: Array of precession angles

    Returns:
        x, y, z coordinates of spin axis tip
    """
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z


def steady_precession_rate(I_spin, spin_rate, mass, pivot_to_cm, theta, g=9.81):
    """
    Calculate steady precession rate.

    Omega_p = tau / L = (m*g*r*sin(theta)) / (I_spin * omega * sin(theta))
            = m*g*r / (I_spin * omega)
    """
    L3 = I_spin * spin_rate
    return mass * g * pivot_to_cm / L3


def main():
    # Gyroscope parameters (typical toy gyroscope scale)
    mass = 0.2  # kg
    radius = 0.03  # m (disk radius)
    pivot_to_cm = 0.05  # m

    # Moments of inertia for a disk
    I_spin = 0.5 * mass * radius**2  # About spin axis
    I_perp = 0.25 * mass * radius**2 + mass * pivot_to_cm**2  # About pivot

    g = 9.81
    t_final = 5.0
    dt = 0.0001

    fig = plt.figure(figsize=(18, 12))

    # --- Different spin rates ---
    ax1 = fig.add_subplot(2, 3, 1)

    spin_rates = [50, 100, 200, 400]  # rad/s
    colors = ['red', 'orange', 'green', 'blue']

    for spin_rate, color in zip(spin_rates, colors):
        theta0 = np.radians(30)  # Initial tilt
        phi0 = 0.0
        theta_dot0 = 0.0
        phi_dot0 = 0.0

        result = simulate_gyroscope(
            I_spin, I_perp, mass, pivot_to_cm, spin_rate,
            theta0, phi0, theta_dot0, phi_dot0,
            t_final, dt, g
        )

        # Theoretical precession rate
        omega_p = steady_precession_rate(I_spin, spin_rate, mass, pivot_to_cm, theta0, g)

        ax1.plot(result['times'], np.degrees(result['phi']),
                 color=color, lw=1.5,
                 label=f'omega={spin_rate} rad/s (Omega_p={np.degrees(omega_p):.1f} deg/s)')

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Precession Angle (degrees)')
    ax1.set_title('Precession vs Spin Rate\n(Higher spin = slower precession)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Nutation and precession ---
    ax2 = fig.add_subplot(2, 3, 2)

    spin_rate = 100  # rad/s
    theta0 = np.radians(30)
    phi0 = 0.0
    theta_dot0 = 0.5  # Non-zero to induce nutation
    phi_dot0 = 0.0

    result = simulate_gyroscope(
        I_spin, I_perp, mass, pivot_to_cm, spin_rate,
        theta0, phi0, theta_dot0, phi_dot0,
        t_final, dt, g
    )

    ax2.plot(result['times'], np.degrees(result['theta']), 'b-', lw=1.5, label='theta (nutation)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Nutation Angle (degrees)')
    ax2.set_title('Nutation: Oscillation of Tilt Angle\n(theta_dot0 = 0.5 rad/s)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # --- Spin axis trajectory on sphere ---
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')

    # Case 1: Pure precession (steady state)
    theta0 = np.radians(30)
    omega_p = steady_precession_rate(I_spin, spin_rate, mass, pivot_to_cm, theta0, g)
    phi_dot0 = omega_p  # Start at steady precession rate
    theta_dot0 = 0.0

    result = simulate_gyroscope(
        I_spin, I_perp, mass, pivot_to_cm, spin_rate,
        theta0, 0.0, theta_dot0, phi_dot0,
        t_final, dt, g
    )

    x, y, z = spin_axis_trajectory(result['theta'], result['phi'])
    ax3.plot(x, y, z, 'b-', lw=1.5, label='Steady precession')

    # Case 2: With nutation
    theta_dot0 = 0.5
    phi_dot0 = 0.0

    result = simulate_gyroscope(
        I_spin, I_perp, mass, pivot_to_cm, spin_rate,
        theta0, 0.0, theta_dot0, phi_dot0,
        t_final, dt, g
    )

    x, y, z = spin_axis_trajectory(result['theta'], result['phi'])
    ax3.plot(x, y, z, 'r-', lw=1, alpha=0.7, label='With nutation')

    # Draw hemisphere
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi/2, 30)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones(np.size(u)), np.cos(v))
    ax3.plot_surface(xs, ys, zs, alpha=0.1, color='gray')

    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('Spin Axis Trajectory')
    ax3.legend(fontsize=8)

    # --- Precession rate vs spin rate ---
    ax4 = fig.add_subplot(2, 3, 4)

    spin_rates_sweep = np.linspace(20, 500, 50)
    precession_rates = []

    for omega in spin_rates_sweep:
        omega_p = steady_precession_rate(I_spin, omega, mass, pivot_to_cm, np.radians(30), g)
        precession_rates.append(omega_p)

    ax4.plot(spin_rates_sweep, np.degrees(precession_rates), 'b-', lw=2)
    ax4.set_xlabel('Spin Rate (rad/s)')
    ax4.set_ylabel('Precession Rate (deg/s)')
    ax4.set_title('Precession Rate vs Spin Rate\nOmega_p = mgr / (I*omega)')
    ax4.grid(True, alpha=0.3)

    # Mark the inverse relationship
    ax4.annotate('Omega_p ~ 1/omega', xy=(300, 5), fontsize=12)

    # --- Energy analysis ---
    ax5 = fig.add_subplot(2, 3, 5)

    spin_rate = 100
    theta0 = np.radians(30)
    theta_dot0 = 0.5
    phi_dot0 = 0.0

    result = simulate_gyroscope(
        I_spin, I_perp, mass, pivot_to_cm, spin_rate,
        theta0, 0.0, theta_dot0, phi_dot0,
        t_final, dt, g
    )

    # Calculate energies
    KE_precession = 0.5 * I_perp * np.sin(result['theta'])**2 * result['phi_dot']**2
    KE_nutation = 0.5 * I_perp * result['theta_dot']**2
    PE = mass * g * pivot_to_cm * np.cos(result['theta'])

    # Total (excluding constant spin kinetic energy)
    E_total = KE_precession + KE_nutation + PE

    ax5.plot(result['times'], KE_precession, 'b-', lw=1.5, label='KE (precession)')
    ax5.plot(result['times'], KE_nutation, 'r-', lw=1.5, label='KE (nutation)')
    ax5.plot(result['times'], PE - PE[0], 'g-', lw=1.5, label='PE - PE_0')
    ax5.plot(result['times'], E_total - E_total[0], 'k--', lw=2, label='Total E - E_0')

    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Energy (J)')
    ax5.set_title('Energy Exchange During Nutation\n(Total energy conserved)')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # --- Different initial tilt angles ---
    ax6 = fig.add_subplot(2, 3, 6)

    tilt_angles = [10, 30, 60, 80]  # degrees
    colors = ['blue', 'green', 'orange', 'red']
    spin_rate = 100

    for tilt, color in zip(tilt_angles, colors):
        theta0 = np.radians(tilt)
        omega_p = steady_precession_rate(I_spin, spin_rate, mass, pivot_to_cm, theta0, g)

        result = simulate_gyroscope(
            I_spin, I_perp, mass, pivot_to_cm, spin_rate,
            theta0, 0.0, 0.0, omega_p,  # Start at steady precession
            t_final, dt, g
        )

        ax6.plot(result['times'], np.degrees(result['phi']),
                 color=color, lw=1.5, label=f'theta = {tilt} deg')

    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Precession Angle (degrees)')
    ax6.set_title('Precession at Different Tilt Angles\n(Same spin rate)')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    plt.suptitle('Gyroscope Precession and Nutation', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'gyroscope_precession.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'gyroscope_precession.png')}")

    # Print analysis
    print("\nGyroscope Analysis:")
    print("-" * 50)
    print(f"Mass: {mass} kg")
    print(f"Disk radius: {radius*100:.1f} cm")
    print(f"Pivot to CM: {pivot_to_cm*100:.1f} cm")
    print(f"I_spin: {I_spin:.6f} kg*m^2")
    print(f"I_perp: {I_perp:.6f} kg*m^2")
    print(f"\nSteady precession rates (theta = 30 deg):")
    for omega in [50, 100, 200]:
        omega_p = steady_precession_rate(I_spin, omega, mass, pivot_to_cm, np.radians(30), g)
        print(f"  Spin {omega} rad/s: Precession {np.degrees(omega_p):.2f} deg/s")


if __name__ == "__main__":
    main()
