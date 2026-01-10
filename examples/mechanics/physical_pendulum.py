"""
Experiment 44: Physical Pendulum

This example demonstrates the physical (compound) pendulum, which is a rigid
body rotating about a fixed pivot point. Unlike the simple pendulum, the
moment of inertia and center of mass position affect the motion.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


class PhysicalPendulum:
    """
    Physical pendulum - a rigid body oscillating about a pivot.

    The equation of motion is:
    I * alpha = -m * g * d * sin(theta)

    where I is moment of inertia about pivot, m is mass, g is gravity,
    d is distance from pivot to center of mass.
    """

    def __init__(self, mass, I_pivot, d_cm, theta0, omega0=0.0, damping=0.0, g=9.81):
        """
        Initialize physical pendulum.

        Args:
            mass: Total mass of pendulum
            I_pivot: Moment of inertia about pivot point
            d_cm: Distance from pivot to center of mass
            theta0: Initial angle from vertical (radians)
            omega0: Initial angular velocity (rad/s)
            damping: Damping coefficient
            g: Gravitational acceleration
        """
        self.mass = mass
        self.I_pivot = I_pivot
        self.d_cm = d_cm
        self.theta = theta0
        self.omega = omega0
        self.damping = damping
        self.g = g

    def angular_acceleration(self, theta, omega):
        """Calculate angular acceleration."""
        torque = -self.mass * self.g * self.d_cm * np.sin(theta)
        damping_torque = -self.damping * omega
        return (torque + damping_torque) / self.I_pivot

    def update(self, dt):
        """Update pendulum state using RK4 integration."""
        # RK4 for theta and omega
        k1_theta = self.omega
        k1_omega = self.angular_acceleration(self.theta, self.omega)

        k2_theta = self.omega + 0.5 * dt * k1_omega
        k2_omega = self.angular_acceleration(self.theta + 0.5 * dt * k1_theta,
                                              self.omega + 0.5 * dt * k1_omega)

        k3_theta = self.omega + 0.5 * dt * k2_omega
        k3_omega = self.angular_acceleration(self.theta + 0.5 * dt * k2_theta,
                                              self.omega + 0.5 * dt * k2_omega)

        k4_theta = self.omega + dt * k3_omega
        k4_omega = self.angular_acceleration(self.theta + dt * k3_theta,
                                              self.omega + dt * k3_omega)

        self.theta += (dt / 6) * (k1_theta + 2*k2_theta + 2*k3_theta + k4_theta)
        self.omega += (dt / 6) * (k1_omega + 2*k2_omega + 2*k3_omega + k4_omega)

    def small_angle_period(self):
        """Calculate period for small oscillations."""
        return 2 * np.pi * np.sqrt(self.I_pivot / (self.mass * self.g * self.d_cm))

    def equivalent_simple_length(self):
        """Calculate equivalent simple pendulum length."""
        return self.I_pivot / (self.mass * self.d_cm)


def simulate_physical_pendulum(pendulum, t_final, dt):
    """
    Simulate physical pendulum motion.

    Returns:
        Dictionary with time, angle, and angular velocity data
    """
    times = [0]
    thetas = [pendulum.theta]
    omegas = [pendulum.omega]

    t = 0
    while t < t_final:
        pendulum.update(dt)
        t += dt
        times.append(t)
        thetas.append(pendulum.theta)
        omegas.append(pendulum.omega)

    return {
        'time': np.array(times),
        'theta': np.array(thetas),
        'omega': np.array(omegas)
    }


def rod_pendulum(length, mass, pivot_offset=0):
    """
    Create a uniform rod pendulum.

    Args:
        length: Rod length
        mass: Rod mass
        pivot_offset: Distance from one end to pivot (0 = pivot at end)
    """
    # Center of mass at L/2 from one end
    d_cm = length / 2 - pivot_offset

    # Moment of inertia about CM: I_cm = (1/12) * m * L^2
    I_cm = (1.0 / 12) * mass * length**2

    # Parallel axis theorem: I_pivot = I_cm + m * d_cm^2
    I_pivot = I_cm + mass * d_cm**2

    return I_pivot, d_cm


def disk_pendulum(radius, mass, pivot_offset=0):
    """
    Create a disk pendulum (pivot at edge or offset from center).

    Args:
        radius: Disk radius
        mass: Disk mass
        pivot_offset: Distance from center to pivot (radius = pivot at edge)
    """
    d_cm = pivot_offset

    # Moment of inertia about CM: I_cm = (1/2) * m * R^2
    I_cm = 0.5 * mass * radius**2

    # Parallel axis theorem
    I_pivot = I_cm + mass * d_cm**2

    return I_pivot, d_cm


def main():
    # Parameters
    g = 9.81
    theta0 = np.radians(30)  # 30 degrees initial angle
    t_final = 10.0
    dt = 0.001

    # Create figure
    fig = plt.figure(figsize=(16, 14))

    # Case 1: Uniform rod pivoted at one end
    ax1 = fig.add_subplot(3, 3, 1)

    rod_length = 1.0
    rod_mass = 1.0
    I_rod, d_rod = rod_pendulum(rod_length, rod_mass, pivot_offset=0)

    pendulum_rod = PhysicalPendulum(rod_mass, I_rod, d_rod, theta0, g=g)
    results_rod = simulate_physical_pendulum(pendulum_rod, t_final, dt)

    # Equivalent simple pendulum
    L_eq = pendulum_rod.equivalent_simple_length()

    ax1.plot(results_rod['time'], np.degrees(results_rod['theta']), 'b-', lw=2)
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Angle (degrees)')
    ax1.set_title(f'Uniform Rod (L={rod_length}m, pivot at end)\n'
                  f'T = {pendulum_rod.small_angle_period():.3f}s, L_eq = {L_eq:.3f}m')
    ax1.grid(True, alpha=0.3)

    # Case 2: Comparison of different pendulum types
    ax2 = fig.add_subplot(3, 3, 2)

    # Simple pendulum of length 1m
    simple_period = 2 * np.pi * np.sqrt(1.0 / g)

    # Different physical pendulums with same total length
    pendulum_configs = [
        ('Simple (L=1m)', 1.0, 1.0, 1.0),  # Equivalent to point mass at L
        ('Rod (L=1m, end)', rod_length, I_rod, d_rod),
        ('Disk (R=0.5m, edge)', *disk_pendulum(0.5, rod_mass, pivot_offset=0.5), 0.5),
    ]

    for name, length, I, d in pendulum_configs:
        if name.startswith('Simple'):
            # For simple pendulum, I = m*L^2, d = L
            pend = PhysicalPendulum(rod_mass, rod_mass * 1.0**2, 1.0, theta0, g=g)
        else:
            pend = PhysicalPendulum(rod_mass, I, d, theta0, g=g)

        results = simulate_physical_pendulum(pend, t_final, dt)
        ax2.plot(results['time'], np.degrees(results['theta']),
                 lw=2, label=f'{name}: T={pend.small_angle_period():.3f}s')

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Angle (degrees)')
    ax2.set_title('Comparison of Pendulum Types')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Case 3: Period vs pivot position for rod
    ax3 = fig.add_subplot(3, 3, 3)

    pivot_positions = np.linspace(0.01, rod_length - 0.01, 100)
    periods = []

    for pivot_pos in pivot_positions:
        I, d = rod_pendulum(rod_length, rod_mass, pivot_offset=pivot_pos)
        if d > 0:  # Valid configuration
            T = 2 * np.pi * np.sqrt(I / (rod_mass * g * d))
            periods.append(T)
        else:
            periods.append(np.nan)

    ax3.plot(pivot_positions / rod_length, periods, 'b-', lw=2)

    # Mark minimum period
    valid_periods = [p for p in periods if not np.isnan(p)]
    min_idx = np.nanargmin(periods)
    ax3.plot(pivot_positions[min_idx] / rod_length, periods[min_idx], 'ro',
             markersize=10, label=f'Minimum T at x/L = {pivot_positions[min_idx]/rod_length:.2f}')

    ax3.set_xlabel('Pivot position / Length')
    ax3.set_ylabel('Period (s)')
    ax3.set_title('Period vs Pivot Position (Uniform Rod)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Case 4: Large amplitude oscillations
    ax4 = fig.add_subplot(3, 3, 4)

    initial_angles = [10, 30, 60, 90, 120]
    colors = plt.cm.viridis(np.linspace(0, 1, len(initial_angles)))

    for angle_deg, color in zip(initial_angles, colors):
        pend = PhysicalPendulum(rod_mass, I_rod, d_rod, np.radians(angle_deg), g=g)
        results = simulate_physical_pendulum(pend, 15.0, dt)
        ax4.plot(results['time'], np.degrees(results['theta']),
                 color=color, lw=1.5, label=f'{angle_deg} deg')

    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Angle (degrees)')
    ax4.set_title('Large Amplitude Oscillations')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Case 5: Phase space
    ax5 = fig.add_subplot(3, 3, 5)

    for angle_deg, color in zip([30, 60, 90, 150], colors[:4]):
        pend = PhysicalPendulum(rod_mass, I_rod, d_rod, np.radians(angle_deg), g=g)
        results = simulate_physical_pendulum(pend, 20.0, dt)
        ax5.plot(np.degrees(results['theta']), results['omega'],
                 color=color, lw=1, alpha=0.7, label=f'{angle_deg} deg')

    ax5.set_xlabel('Angle (degrees)')
    ax5.set_ylabel('Angular velocity (rad/s)')
    ax5.set_title('Phase Space')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Case 6: Energy conservation
    ax6 = fig.add_subplot(3, 3, 6)

    pend = PhysicalPendulum(rod_mass, I_rod, d_rod, np.radians(60), g=g)
    results = simulate_physical_pendulum(pend, 10.0, dt)

    # Calculate energies
    # PE = m * g * d * (1 - cos(theta))
    # KE = (1/2) * I * omega^2
    PE = rod_mass * g * d_rod * (1 - np.cos(results['theta']))
    KE = 0.5 * I_rod * results['omega']**2
    total_E = PE + KE

    ax6.plot(results['time'], KE, 'b-', lw=1.5, label='Kinetic Energy')
    ax6.plot(results['time'], PE, 'r-', lw=1.5, label='Potential Energy')
    ax6.plot(results['time'], total_E, 'g--', lw=2, label='Total Energy')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Energy (J)')
    ax6.set_title('Energy Conservation')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Case 7: Damped physical pendulum
    ax7 = fig.add_subplot(3, 3, 7)

    damping_coeffs = [0, 0.05, 0.1, 0.2]
    colors_damp = plt.cm.Reds(np.linspace(0.3, 1, len(damping_coeffs)))

    for damp, color in zip(damping_coeffs, colors_damp):
        pend = PhysicalPendulum(rod_mass, I_rod, d_rod, np.radians(45),
                                damping=damp, g=g)
        results = simulate_physical_pendulum(pend, 20.0, dt)
        ax7.plot(results['time'], np.degrees(results['theta']),
                 color=color, lw=1.5, label=f'b = {damp}')

    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('Angle (degrees)')
    ax7.set_title('Damped Physical Pendulum')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # Case 8: Period vs amplitude
    ax8 = fig.add_subplot(3, 3, 8)

    amplitudes = np.linspace(5, 170, 34)
    measured_periods = []
    small_angle_T = 2 * np.pi * np.sqrt(I_rod / (rod_mass * g * d_rod))

    for amp in amplitudes:
        pend = PhysicalPendulum(rod_mass, I_rod, d_rod, np.radians(amp), g=g)
        results = simulate_physical_pendulum(pend, 30.0, 0.0005)

        # Find period from zero crossings
        thetas = results['theta']
        crossings = []
        for i in range(1, len(thetas)):
            if thetas[i-1] > 0 and thetas[i] <= 0:
                crossings.append(results['time'][i])

        if len(crossings) >= 2:
            T = 2 * (crossings[1] - crossings[0])
        else:
            T = np.nan
        measured_periods.append(T)

    ax8.plot(amplitudes, measured_periods, 'b.-', lw=1.5, label='Measured')
    ax8.axhline(y=small_angle_T, color='r', linestyle='--',
                label=f'Small angle: T = {small_angle_T:.3f}s')
    ax8.set_xlabel('Initial Amplitude (degrees)')
    ax8.set_ylabel('Period (s)')
    ax8.set_title('Period vs Amplitude')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # Case 9: Visual representation of pendulum motion
    ax9 = fig.add_subplot(3, 3, 9)

    # Draw snapshots at different times
    pend = PhysicalPendulum(rod_mass, I_rod, d_rod, np.radians(60), g=g)
    results = simulate_physical_pendulum(pend, 5.0, dt)

    times_to_show = np.linspace(0, 5.0, 20)
    colors_snap = plt.cm.cool(np.linspace(0, 1, len(times_to_show)))

    for t_show, color in zip(times_to_show, colors_snap):
        idx = int(t_show / dt)
        if idx < len(results['theta']):
            theta = results['theta'][idx]

            # Draw rod
            x_end = rod_length * np.sin(theta)
            y_end = -rod_length * np.cos(theta)
            ax9.plot([0, x_end], [0, y_end], color=color, lw=2, alpha=0.5)

            # Mark center of mass
            x_cm = d_rod * np.sin(theta)
            y_cm = -d_rod * np.cos(theta)
            ax9.plot(x_cm, y_cm, 'o', color=color, markersize=4)

    ax9.plot(0, 0, 'ko', markersize=10)  # Pivot
    ax9.set_xlim(-1.5, 1.5)
    ax9.set_ylim(-1.5, 0.5)
    ax9.set_aspect('equal')
    ax9.set_xlabel('x (m)')
    ax9.set_ylabel('y (m)')
    ax9.set_title('Pendulum Motion (color = time)')
    ax9.grid(True, alpha=0.3)

    plt.suptitle('Physical (Compound) Pendulum\n'
                 'Uniform rod: L=1m, pivot at one end',
                 fontsize=12, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'physical_pendulum.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'physical_pendulum.png')}")

    # Print summary
    print("\nPhysical Pendulum Summary:")
    print("=" * 50)
    print(f"Uniform rod: L = {rod_length}m, M = {rod_mass}kg")
    print(f"Pivot at end: I = {I_rod:.4f} kg*m^2, d_cm = {d_rod:.4f}m")
    print(f"Small angle period: T = {small_angle_T:.4f}s")
    print(f"Equivalent simple pendulum length: L_eq = {I_rod/(rod_mass*d_rod):.4f}m")
    print("\nFor uniform rod pivoted at end: L_eq = (2/3)*L")


if __name__ == "__main__":
    main()
