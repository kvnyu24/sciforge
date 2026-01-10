"""
Experiment 29: Rocket equation (Tsiolkovsky) - mass loss + thrust profile.

Demonstrates the Tsiolkovsky rocket equation and simulates
rocket motion with varying mass.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def tsiolkovsky_delta_v(v_exhaust, m_initial, m_final):
    """
    Tsiolkovsky rocket equation.

    Δv = v_exhaust * ln(m_initial / m_final)
    """
    return v_exhaust * np.log(m_initial / m_final)


def simulate_rocket(m_initial, m_fuel, m_dot, v_exhaust, g, dt, t_max):
    """
    Simulate rocket ascent with mass loss.

    Args:
        m_initial: Initial total mass (kg)
        m_fuel: Fuel mass (kg)
        m_dot: Fuel burn rate (kg/s)
        v_exhaust: Exhaust velocity (m/s)
        g: Gravitational acceleration (m/s^2)
        dt: Time step
        t_max: Maximum simulation time

    Returns:
        times, positions, velocities, masses, thrusts
    """
    # Burn time
    t_burn = m_fuel / m_dot

    t = 0
    x = 0
    v = 0
    m = m_initial

    times = [t]
    positions = [x]
    velocities = [v]
    masses = [m]
    thrusts = [m_dot * v_exhaust]
    accelerations = [(m_dot * v_exhaust / m) - g]

    while t < t_max:
        # Thrust (if fuel remaining)
        if m > m_initial - m_fuel:
            thrust = m_dot * v_exhaust
            dm = -m_dot * dt
        else:
            thrust = 0
            dm = 0

        # Acceleration
        a = thrust / m - g

        # Update (Euler)
        v_new = v + a * dt
        x_new = x + v * dt

        # Update mass
        m_new = m + dm

        t += dt
        x = x_new
        v = v_new
        m = m_new

        times.append(t)
        positions.append(x)
        velocities.append(v)
        masses.append(m)
        thrusts.append(thrust)
        accelerations.append(a)

        # Stop if returned to ground with negative velocity
        if x < 0 and v < 0 and t > 1:
            break

    return (np.array(times), np.array(positions), np.array(velocities),
            np.array(masses), np.array(thrusts), np.array(accelerations))


def multi_stage_rocket(stages, g, dt, t_max):
    """
    Simulate multi-stage rocket.

    stages: list of (m_structure, m_fuel, m_dot, v_exhaust)
    """
    t = 0
    x = 0
    v = 0

    # Total initial mass
    m = sum(s[0] + s[1] for s in stages)  # structure + fuel for all stages

    times = [t]
    positions = [x]
    velocities = [v]
    masses = [m]
    stage_events = []

    current_stage = 0

    while t < t_max and current_stage < len(stages):
        m_structure, m_fuel, m_dot, v_exhaust = stages[current_stage]

        # Fuel remaining in current stage
        m_above = sum(s[0] + s[1] for s in stages[current_stage+1:])  # mass of upper stages
        m_current_fuel = m - m_structure - m_above

        if m_current_fuel > 0:
            thrust = m_dot * v_exhaust
            dm = -m_dot * dt
        else:
            # Stage separation
            stage_events.append((t, x, v, current_stage))
            m -= m_structure  # Drop empty stage
            current_stage += 1
            if current_stage < len(stages):
                continue
            else:
                thrust = 0
                dm = 0

        # Acceleration
        a = thrust / m - g if m > 0 else -g

        # Update
        v += a * dt
        x += v * dt
        m += dm
        t += dt

        times.append(t)
        positions.append(x)
        velocities.append(v)
        masses.append(m)

        if x < 0 and v < 0 and t > 1:
            break

    return np.array(times), np.array(positions), np.array(velocities), np.array(masses), stage_events


def main():
    # Single stage rocket parameters
    m_initial = 1000  # kg
    m_fuel = 800  # kg
    m_dot = 10  # kg/s
    v_exhaust = 3000  # m/s (typical chemical rocket)
    g = 9.81  # m/s^2
    dt = 0.1
    t_max = 300

    # Simulate
    times, pos, vel, mass, thrust, acc = simulate_rocket(
        m_initial, m_fuel, m_dot, v_exhaust, g, dt, t_max)

    # Theoretical delta-v
    dv_theory = tsiolkovsky_delta_v(v_exhaust, m_initial, m_initial - m_fuel)

    # Actual delta-v (considering gravity losses)
    t_burn = m_fuel / m_dot
    burn_end_idx = np.searchsorted(times, t_burn)
    dv_actual = vel[burn_end_idx]

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Altitude vs time
    ax = axes[0, 0]
    ax.plot(times, pos / 1000, 'b-', lw=2)
    ax.axvline(t_burn, color='red', linestyle='--', label=f'Burnout t={t_burn:.0f}s')
    max_alt_idx = np.argmax(pos)
    ax.plot(times[max_alt_idx], pos[max_alt_idx]/1000, 'go', markersize=10,
            label=f'Max alt: {pos[max_alt_idx]/1000:.1f} km')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Altitude (km)')
    ax.set_title('Rocket Altitude')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Velocity vs time
    ax = axes[0, 1]
    ax.plot(times, vel, 'b-', lw=2)
    ax.axvline(t_burn, color='red', linestyle='--', label='Burnout')
    ax.axhline(dv_theory, color='green', linestyle=':', label=f'Δv theory: {dv_theory:.0f} m/s')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title(f'Velocity (Δv actual: {dv_actual:.0f} m/s)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Mass vs time
    ax = axes[0, 2]
    ax.plot(times, mass, 'b-', lw=2)
    ax.axhline(m_initial - m_fuel, color='red', linestyle='--', label='Dry mass')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mass (kg)')
    ax.set_title('Rocket Mass')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Multi-stage comparison
    ax = axes[1, 0]

    # Single stage
    ax.plot(times, pos / 1000, 'b-', lw=2, label='Single stage')

    # Two-stage rocket with same total mass
    # Stage 1: 600 kg total, 500 kg fuel
    # Stage 2: 400 kg total, 300 kg fuel
    stages = [
        (100, 500, 8, 3000),   # Stage 1: structure, fuel, burn rate, exhaust vel
        (100, 300, 5, 3200),   # Stage 2
    ]
    t2, p2, v2, m2, events = multi_stage_rocket(stages, g, dt, t_max)
    ax.plot(t2, p2 / 1000, 'r-', lw=2, label='Two stage')

    for event in events:
        ax.axvline(event[0], color='orange', linestyle=':', alpha=0.5)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Altitude (km)')
    ax.set_title('Single vs Multi-Stage')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Delta-v budget
    ax = axes[1, 1]

    mass_ratios = np.linspace(1.5, 10, 50)
    delta_vs = v_exhaust * np.log(mass_ratios)

    ax.plot(mass_ratios, delta_vs / 1000, 'b-', lw=2)
    ax.axhline(9.4, color='red', linestyle='--', label='LEO (9.4 km/s)')
    ax.axhline(11.2, color='orange', linestyle='--', label='Earth escape (11.2 km/s)')

    our_ratio = m_initial / (m_initial - m_fuel)
    our_dv = v_exhaust * np.log(our_ratio)
    ax.plot(our_ratio, our_dv / 1000, 'go', markersize=10, label=f'Our rocket')

    ax.set_xlabel('Mass ratio (m₀/m_f)')
    ax.set_ylabel('Δv (km/s)')
    ax.set_title('Tsiolkovsky Rocket Equation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Summary
    ax = axes[1, 2]
    ax.axis('off')

    gravity_loss = dv_theory - dv_actual

    summary = f"""Rocket Dynamics Summary
=======================
Tsiolkovsky Equation:
  Δv = v_e × ln(m₀/m_f)

Single Stage Rocket:
  Initial mass: {m_initial} kg
  Fuel mass: {m_fuel} kg
  Exhaust velocity: {v_exhaust} m/s
  Burn rate: {m_dot} kg/s
  Burn time: {t_burn:.1f} s

Performance:
  Δv (theory): {dv_theory:.0f} m/s
  Δv (actual): {dv_actual:.0f} m/s
  Gravity loss: {gravity_loss:.0f} m/s
  Max altitude: {pos[max_alt_idx]/1000:.1f} km

Mass Ratio:
  m₀/m_f = {our_ratio:.2f}

Staging Benefits:
  Single stage max alt: {np.max(pos)/1000:.1f} km
  Two stage max alt: {np.max(p2)/1000:.1f} km

Key Insight: Staging helps because
each stage carries less dead weight
(empty tanks) after burnout."""

    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Rocket Equation and Multi-Stage Rockets',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'rocket_equation.png'), dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/rocket_equation.png")


if __name__ == "__main__":
    main()
