"""
Example demonstrating a spinning top with friction.

A spinning top (symmetric top) exhibits complex behavior when friction
is present at the contact point. Key phenomena include:

1. Energy dissipation: Friction causes the spin to slow down over time
2. Rising motion: A fast-spinning top can "rise" to a more vertical position
3. Sleeping top: At high spin rates, the top can balance vertically
4. Wobble and fall: As spin decreases, wobble increases until the top falls

The friction at the contact point provides:
- Energy dissipation (slowing the spin)
- Torque that can cause the top to rise

This example simulates a tippe-top-like behavior where friction can cause
counterintuitive rising motion.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def simulate_spinning_top(I_spin, I_perp, mass, cm_height, contact_radius,
                          spin_rate0, theta0, friction_coeff,
                          t_final, dt, g=9.81):
    """
    Simulate a spinning top with friction at the contact point.

    Uses simplified Euler angle dynamics with friction modeling.

    Args:
        I_spin: Moment of inertia about spin axis
        I_perp: Moment of inertia perpendicular to spin axis
        mass: Total mass
        cm_height: Height of center of mass above contact point
        contact_radius: Radius at which friction acts
        spin_rate0: Initial spin rate
        theta0: Initial tilt angle from vertical
        friction_coeff: Friction coefficient at contact
        t_final: Simulation duration
        dt: Time step
        g: Gravitational acceleration

    Returns:
        Dictionary with time history
    """
    # State: [theta, phi, psi, theta_dot, phi_dot, psi_dot]
    # theta: nutation angle (tilt)
    # phi: precession angle
    # psi: spin angle

    theta = theta0
    phi = 0.0
    psi = 0.0
    theta_dot = 0.0
    phi_dot = 0.0
    psi_dot = spin_rate0

    times = [0.0]
    theta_history = [theta]
    phi_history = [phi]
    psi_dot_history = [psi_dot]
    theta_dot_history = [theta_dot]
    phi_dot_history = [phi_dot]
    energy_history = []

    # Calculate initial energy
    KE_spin = 0.5 * I_spin * psi_dot**2
    KE_prec = 0.5 * I_perp * (theta_dot**2 + np.sin(theta)**2 * phi_dot**2)
    PE = mass * g * cm_height * np.cos(theta)
    energy_history.append(KE_spin + KE_prec + PE)

    t = 0
    while t < t_final and psi_dot > 0.1:  # Stop when spin rate drops too low
        sin_theta = np.sin(theta) if abs(theta) > 1e-6 else 1e-6
        cos_theta = np.cos(theta)

        # Angular momentum about spin axis
        L_spin = I_spin * psi_dot

        # Friction torque (acts to slow spin and can cause rising)
        # Friction force at contact point
        # F_friction = mu * N, where N ~ mg for small theta
        F_friction = friction_coeff * mass * g

        # Torque from friction about pivot
        tau_friction = F_friction * contact_radius

        # Friction causes spin to slow down
        psi_ddot = -tau_friction / I_spin

        # Gravity torque (causes precession)
        tau_gravity = mass * g * cm_height * sin_theta

        # For a heavy symmetric top with friction, equations of motion:

        # Simplified model: steady precession with friction effects
        if abs(L_spin) > 1e-6:
            # Precession rate from gyroscopic effect
            phi_dot_steady = tau_gravity / (L_spin * sin_theta) if abs(sin_theta) > 1e-6 else 0

            # Friction can cause rising (theta decreases)
            # This is the "tippe top" effect
            # Friction torque component that affects theta
            tau_theta = -tau_friction * sin_theta * 0.5  # Simplified model

            theta_ddot = (sin_theta * cos_theta * phi_dot**2
                          - L_spin * sin_theta * phi_dot / I_perp
                          + tau_gravity / I_perp
                          + tau_theta / I_perp)

            # Adjust precession rate
            phi_ddot = (-2 * cos_theta * theta_dot * phi_dot / sin_theta
                        + L_spin * theta_dot / (I_perp * sin_theta)
                        if abs(sin_theta) > 1e-6 else 0)
        else:
            phi_dot_steady = 0
            theta_ddot = tau_gravity / I_perp
            phi_ddot = 0

        # Update velocities
        psi_dot += psi_ddot * dt
        theta_dot += theta_ddot * dt
        phi_dot += phi_ddot * dt

        # Damping on theta_dot to prevent instability
        theta_dot *= 0.99

        # Update angles
        psi += psi_dot * dt
        theta += theta_dot * dt
        phi += phi_dot * dt

        # Keep theta in valid range
        theta = max(0.01, min(np.pi - 0.01, theta))

        # Prevent negative spin
        if psi_dot < 0:
            psi_dot = 0

        t += dt

        # Store history
        times.append(t)
        theta_history.append(theta)
        phi_history.append(phi)
        psi_dot_history.append(psi_dot)
        theta_dot_history.append(theta_dot)
        phi_dot_history.append(phi_dot)

        # Energy
        KE_spin = 0.5 * I_spin * psi_dot**2
        KE_prec = 0.5 * I_perp * (theta_dot**2 + np.sin(theta)**2 * phi_dot**2)
        PE = mass * g * cm_height * np.cos(theta)
        energy_history.append(KE_spin + KE_prec + PE)

    return {
        'times': np.array(times),
        'theta': np.array(theta_history),
        'phi': np.array(phi_history),
        'psi_dot': np.array(psi_dot_history),
        'theta_dot': np.array(theta_dot_history),
        'phi_dot': np.array(phi_dot_history),
        'energy': np.array(energy_history)
    }


def sleeping_top_stability(I_spin, I_perp, mass, cm_height, spin_rate, g=9.81):
    """
    Check if a top can "sleep" (remain vertical) at given spin rate.

    The sleeping condition requires:
    omega^2 > 4 * m * g * h * I_perp / I_spin^2
    """
    critical_omega_sq = 4 * mass * g * cm_height * I_perp / I_spin**2
    return spin_rate**2 > critical_omega_sq, np.sqrt(critical_omega_sq)


def main():
    # Top parameters (typical toy top)
    mass = 0.1  # kg
    radius = 0.03  # m
    cm_height = 0.04  # m
    contact_radius = 0.005  # m

    # Moments of inertia (cone-like)
    I_spin = 0.5 * mass * radius**2
    I_perp = 0.25 * mass * radius**2 + mass * cm_height**2

    g = 9.81
    t_final = 30.0
    dt = 0.0005

    fig = plt.figure(figsize=(18, 12))

    # --- Effect of initial spin rate ---
    ax1 = fig.add_subplot(2, 3, 1)

    spin_rates = [20, 50, 100, 200]  # rad/s
    colors = ['red', 'orange', 'green', 'blue']
    theta0 = np.radians(30)
    friction = 0.3

    for omega, color in zip(spin_rates, colors):
        result = simulate_spinning_top(
            I_spin, I_perp, mass, cm_height, contact_radius,
            omega, theta0, friction, t_final, dt, g
        )

        ax1.plot(result['times'], np.degrees(result['theta']),
                 color=color, lw=1.5, label=f'omega_0 = {omega} rad/s')

    ax1.axhline(y=90, color='black', linestyle='--', alpha=0.5, label='Horizontal')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Tilt Angle (degrees)')
    ax1.set_title('Tilt Angle vs Time\n(Higher spin = more stable)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)

    # --- Spin rate decay ---
    ax2 = fig.add_subplot(2, 3, 2)

    for omega, color in zip(spin_rates, colors):
        result = simulate_spinning_top(
            I_spin, I_perp, mass, cm_height, contact_radius,
            omega, theta0, friction, t_final, dt, g
        )

        ax2.plot(result['times'], result['psi_dot'],
                 color=color, lw=1.5, label=f'omega_0 = {omega} rad/s')

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Spin Rate (rad/s)')
    ax2.set_title('Spin Rate Decay Due to Friction')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- Effect of friction coefficient ---
    ax3 = fig.add_subplot(2, 3, 3)

    frictions = [0.1, 0.2, 0.3, 0.5]
    colors = ['blue', 'green', 'orange', 'red']
    omega0 = 100  # rad/s

    for mu, color in zip(frictions, colors):
        result = simulate_spinning_top(
            I_spin, I_perp, mass, cm_height, contact_radius,
            omega0, theta0, mu, t_final, dt, g
        )

        ax3.plot(result['times'], np.degrees(result['theta']),
                 color=color, lw=1.5, label=f'mu = {mu}')

    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Tilt Angle (degrees)')
    ax3.set_title('Effect of Friction Coefficient\n(Higher friction = faster fall)')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 100)

    # --- Precession and nutation ---
    ax4 = fig.add_subplot(2, 3, 4)

    omega0 = 100
    result = simulate_spinning_top(
        I_spin, I_perp, mass, cm_height, contact_radius,
        omega0, theta0, 0.2, min(t_final, 10.0), dt, g
    )

    ax4.plot(result['times'], np.degrees(result['phi']),
             'b-', lw=1.5, label='Precession (phi)')
    ax4_twin = ax4.twinx()
    ax4_twin.plot(result['times'], np.degrees(result['theta']),
                  'r-', lw=1.5, label='Nutation (theta)')

    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Precession Angle (degrees)', color='blue')
    ax4_twin.set_ylabel('Nutation Angle (degrees)', color='red')
    ax4.set_title('Precession and Nutation Over Time')
    ax4.grid(True, alpha=0.3)

    # --- Spin axis trajectory ---
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')

    omega0 = 100
    result = simulate_spinning_top(
        I_spin, I_perp, mass, cm_height, contact_radius,
        omega0, np.radians(20), 0.2, min(t_final, 15.0), dt, g
    )

    # Calculate spin axis tip position
    x = np.sin(result['theta']) * np.cos(result['phi'])
    y = np.sin(result['theta']) * np.sin(result['phi'])
    z = np.cos(result['theta'])

    # Color by time
    t_norm = result['times'] / result['times'][-1]
    colors = plt.cm.viridis(t_norm)

    for i in range(len(x) - 1):
        ax5.plot([x[i], x[i+1]], [y[i], y[i+1]], [z[i], z[i+1]],
                 color=colors[i], lw=0.5)

    # Draw sphere wireframe
    u = np.linspace(0, 2*np.pi, 20)
    v = np.linspace(0, np.pi, 10)
    xs = np.outer(np.cos(u), np.sin(v)) * 0.5
    ys = np.outer(np.sin(u), np.sin(v)) * 0.5
    zs = np.outer(np.ones(np.size(u)), np.cos(v)) * 0.5 + 0.5
    ax5.plot_wireframe(xs, ys, zs, alpha=0.1, color='gray')

    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    ax5.set_zlabel('Z')
    ax5.set_title('Spin Axis Trajectory\n(Color: blue=start, yellow=end)')

    # --- Energy dissipation ---
    ax6 = fig.add_subplot(2, 3, 6)

    omega0 = 100
    frictions = [0.1, 0.3, 0.5]
    colors = ['green', 'orange', 'red']

    for mu, color in zip(frictions, colors):
        result = simulate_spinning_top(
            I_spin, I_perp, mass, cm_height, contact_radius,
            omega0, theta0, mu, t_final, dt, g
        )

        E_norm = result['energy'] / result['energy'][0]
        ax6.plot(result['times'], E_norm, color=color, lw=1.5,
                 label=f'mu = {mu}')

    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Energy / Initial Energy')
    ax6.set_title('Energy Dissipation Due to Friction')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(0, 1.1)

    plt.suptitle('Spinning Top with Friction\n'
                 'Friction causes spin decay and eventual fall',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'spinning_top_friction.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'spinning_top_friction.png')}")

    # Create additional figure for sleeping top analysis
    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Sleeping top stability diagram
    ax = axes[0]

    spin_rates = np.linspace(10, 300, 100)
    thetas_stable = []
    thetas_unstable = []

    for omega in spin_rates:
        can_sleep, omega_crit = sleeping_top_stability(I_spin, I_perp, mass, cm_height, omega, g)
        if can_sleep:
            thetas_stable.append(omega)
        else:
            thetas_unstable.append(omega)

    # Critical spin rate
    _, omega_critical = sleeping_top_stability(I_spin, I_perp, mass, cm_height, 0, g)

    ax.axvline(x=omega_critical, color='red', linestyle='--', lw=2,
               label=f'Critical omega = {omega_critical:.1f} rad/s')
    ax.axvspan(0, omega_critical, alpha=0.3, color='red', label='Unstable (falls)')
    ax.axvspan(omega_critical, 300, alpha=0.3, color='green', label='Can sleep (stable)')

    ax.set_xlabel('Spin Rate (rad/s)')
    ax.set_ylabel('Stability')
    ax.set_title('Sleeping Top Stability Condition\n'
                 'omega^2 > 4*m*g*h*I_perp / I_spin^2')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 300)
    ax.set_yticks([])

    # Time to fall vs initial spin rate
    ax = axes[1]

    spin_rates_test = [20, 40, 60, 80, 100, 150, 200]
    fall_times = []

    for omega in spin_rates_test:
        result = simulate_spinning_top(
            I_spin, I_perp, mass, cm_height, contact_radius,
            omega, np.radians(10), 0.3, 60.0, dt, g
        )

        # Find when theta exceeds 80 degrees (essentially fallen)
        fall_idx = np.where(result['theta'] > np.radians(80))[0]
        if len(fall_idx) > 0:
            fall_times.append(result['times'][fall_idx[0]])
        else:
            fall_times.append(result['times'][-1])

    ax.plot(spin_rates_test, fall_times, 'bo-', markersize=8, lw=2)
    ax.set_xlabel('Initial Spin Rate (rad/s)')
    ax.set_ylabel('Time to Fall (s)')
    ax.set_title('Time Until Top Falls vs Initial Spin\n'
                 '(Higher spin = longer spin time)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spinning_top_stability.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'spinning_top_stability.png')}")

    # Print analysis
    print("\nSpinning Top Analysis:")
    print("-" * 50)
    print(f"Mass: {mass*1000:.0f} g")
    print(f"Radius: {radius*100:.1f} cm")
    print(f"CM height: {cm_height*100:.1f} cm")
    print(f"I_spin: {I_spin:.6f} kg*m^2")
    print(f"I_perp: {I_perp:.6f} kg*m^2")
    print(f"\nCritical spin rate for sleeping: {omega_critical:.1f} rad/s")
    print(f"  = {omega_critical * 60 / (2*np.pi):.0f} RPM")


if __name__ == "__main__":
    main()
