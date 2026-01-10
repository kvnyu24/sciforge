"""
Experiment 35: Conservative vs Non-Conservative Forces - Energy Bookkeeping with Friction

This example demonstrates the difference between conservative and non-conservative
forces by tracking energy in a system with gravity (conservative) and friction
(non-conservative). Shows how mechanical energy is dissipated by friction.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.mechanics import Particle


def simulate_sliding_block(mass, initial_velocity, mu_k, incline_angle, t_final, dt, g=9.81):
    """
    Simulate a block sliding on an inclined plane with friction.

    Args:
        mass: Block mass (kg)
        initial_velocity: Initial velocity along incline (m/s)
        mu_k: Kinetic friction coefficient
        incline_angle: Angle of incline from horizontal (radians)
        t_final: Simulation duration (s)
        dt: Time step (s)
        g: Gravitational acceleration (m/s^2)

    Returns:
        Dictionary with time, position, velocity, and energy data
    """
    # Set up coordinate system: x along incline (positive = down), y perpendicular
    # Initial position at top of incline
    x0 = 0.0

    # Forces
    # Gravity component along incline: mg*sin(theta) (down the incline)
    # Normal force: N = mg*cos(theta)
    # Friction force: f = mu_k * N (opposes motion)

    F_gravity = mass * g * np.sin(incline_angle)  # down the incline
    N = mass * g * np.cos(incline_angle)  # normal force
    F_friction_mag = mu_k * N  # friction magnitude

    # Data storage
    times = [0]
    positions = [x0]
    velocities = [initial_velocity]

    # Energy tracking
    KE = [0.5 * mass * initial_velocity**2]
    PE = [0]  # Reference at initial position
    work_friction = [0]
    total_mechanical = [KE[0] + PE[0]]

    x = x0
    v = initial_velocity
    t = 0
    W_fric = 0  # Cumulative work done by friction

    height_ref = 0  # Track height change for PE calculation

    while t < t_final and v > 1e-10:  # Stop if velocity becomes zero or negative
        # Friction opposes motion
        if v > 0:
            F_net = F_gravity - F_friction_mag
        else:
            F_net = F_gravity + F_friction_mag

        # Simple Euler integration
        a = F_net / mass
        v_new = v + a * dt

        # Check if block stops due to friction
        if v_new <= 0 and F_gravity < F_friction_mag:
            # Block comes to rest (static friction holds it)
            v_new = 0

        dx = v * dt + 0.5 * a * dt**2
        x_new = x + dx

        # Work done by friction (negative, energy dissipated)
        W_fric -= F_friction_mag * abs(dx)

        # Height change (negative when going down incline)
        dh = -dx * np.sin(incline_angle)
        height_ref += dh

        # Update state
        x = x_new
        v = max(0, v_new)  # Velocity can't go negative
        t += dt

        # Record data
        times.append(t)
        positions.append(x)
        velocities.append(v)

        # Calculate energies
        current_KE = 0.5 * mass * v**2
        current_PE = mass * g * height_ref
        KE.append(current_KE)
        PE.append(current_PE)
        work_friction.append(W_fric)
        total_mechanical.append(current_KE + current_PE)

        if v == 0:
            break

    return {
        'time': np.array(times),
        'position': np.array(positions),
        'velocity': np.array(velocities),
        'kinetic_energy': np.array(KE),
        'potential_energy': np.array(PE),
        'work_friction': np.array(work_friction),
        'mechanical_energy': np.array(total_mechanical)
    }


def simulate_frictionless(mass, initial_velocity, incline_angle, t_final, dt, g=9.81):
    """
    Simulate frictionless sliding for comparison (conservative system).
    """
    F_gravity = mass * g * np.sin(incline_angle)

    times = [0]
    positions = [0]
    velocities = [initial_velocity]
    KE = [0.5 * mass * initial_velocity**2]
    PE = [0]
    total_mechanical = [KE[0]]

    x = 0
    v = initial_velocity
    t = 0
    height_ref = 0

    while t < t_final:
        a = F_gravity / mass
        v_new = v + a * dt
        dx = v * dt + 0.5 * a * dt**2
        x_new = x + dx

        dh = -dx * np.sin(incline_angle)
        height_ref += dh

        x = x_new
        v = v_new
        t += dt

        times.append(t)
        positions.append(x)
        velocities.append(v)

        current_KE = 0.5 * mass * v**2
        current_PE = mass * g * height_ref
        KE.append(current_KE)
        PE.append(current_PE)
        total_mechanical.append(current_KE + current_PE)

    return {
        'time': np.array(times),
        'position': np.array(positions),
        'velocity': np.array(velocities),
        'kinetic_energy': np.array(KE),
        'potential_energy': np.array(PE),
        'mechanical_energy': np.array(total_mechanical)
    }


def main():
    # Parameters
    mass = 1.0  # kg
    initial_velocity = 2.0  # m/s
    incline_angle = np.radians(30)  # 30 degree incline
    g = 9.81  # m/s^2
    t_final = 5.0  # s
    dt = 0.001  # s

    # Different friction coefficients
    friction_coeffs = [0.0, 0.1, 0.3, 0.5]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Simulate for different friction coefficients
    results_list = []
    for mu in friction_coeffs:
        if mu == 0:
            results = simulate_frictionless(mass, initial_velocity, incline_angle, t_final, dt, g)
            results['work_friction'] = np.zeros_like(results['time'])
        else:
            results = simulate_sliding_block(mass, initial_velocity, mu, incline_angle, t_final, dt, g)
        results['mu'] = mu
        results_list.append(results)

    # Plot 1: Position vs Time
    ax1 = axes[0, 0]
    for results in results_list:
        label = f'mu = {results["mu"]}'
        ax1.plot(results['time'], results['position'], lw=2, label=label)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position along incline (m)')
    ax1.set_title('Position vs Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Velocity vs Time
    ax2 = axes[0, 1]
    for results in results_list:
        label = f'mu = {results["mu"]}'
        ax2.plot(results['time'], results['velocity'], lw=2, label=label)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('Velocity vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Kinetic Energy vs Time
    ax3 = axes[0, 2]
    for results in results_list:
        label = f'mu = {results["mu"]}'
        ax3.plot(results['time'], results['kinetic_energy'], lw=2, label=label)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Kinetic Energy (J)')
    ax3.set_title('Kinetic Energy vs Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Mechanical Energy vs Time (showing dissipation)
    ax4 = axes[1, 0]
    for results in results_list:
        label = f'mu = {results["mu"]}'
        ax4.plot(results['time'], results['mechanical_energy'], lw=2, label=label)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Mechanical Energy (J)')
    ax4.set_title('Total Mechanical Energy (KE + PE)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Energy Components for mu = 0.3 (detailed view)
    ax5 = axes[1, 1]
    results_mid = results_list[2]  # mu = 0.3
    ax5.plot(results_mid['time'], results_mid['kinetic_energy'], 'b-', lw=2, label='Kinetic Energy')
    ax5.plot(results_mid['time'], results_mid['potential_energy'], 'r-', lw=2, label='Potential Energy')
    ax5.plot(results_mid['time'], results_mid['mechanical_energy'], 'g--', lw=2, label='Mechanical Energy')
    ax5.plot(results_mid['time'], -results_mid['work_friction'], 'm:', lw=2, label='Energy Dissipated')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Energy (J)')
    ax5.set_title(f'Energy Components (mu = {results_mid["mu"]})')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Energy Conservation Check
    ax6 = axes[1, 2]
    for results in results_list:
        # Total energy = mechanical + dissipated (should be constant)
        total_energy = results['mechanical_energy'] - results['work_friction']
        # Normalize by initial energy
        if total_energy[0] > 0:
            normalized = total_energy / total_energy[0]
        else:
            normalized = np.ones_like(total_energy)
        label = f'mu = {results["mu"]}'
        ax6.plot(results['time'], normalized, lw=2, label=label)
    ax6.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('(Mechanical + Dissipated) / Initial Energy')
    ax6.set_title('Energy Conservation Check')
    ax6.set_ylim(0.95, 1.05)
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.suptitle('Conservative vs Non-Conservative Forces: Energy Bookkeeping\n'
                 f'Block sliding down 30 deg incline, m = {mass} kg, v0 = {initial_velocity} m/s',
                 fontsize=12, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'conservative_forces.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'conservative_forces.png')}")


if __name__ == "__main__":
    main()
