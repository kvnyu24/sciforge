"""
Experiment 58: Tidal Locking Timescales.

Demonstrates the physics of tidal locking, where a moon's rotation
synchronizes with its orbital period due to tidal torques.

Key physics:
1. Tidal bulge raised by gravitational gradient
2. Tidal torque slows rotation toward synchronous state
3. Energy dissipation through tidal heating
4. Timescale depends on mass ratio, distance, and dissipation

The Moon is tidally locked to Earth, always showing the same face.
This experiment shows the evolution toward that state.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


# Physical constants
G = 6.674e-11  # Gravitational constant (m^3 kg^-1 s^-2)

# Earth-Moon system parameters
M_EARTH = 5.972e24  # kg
M_MOON = 7.342e22  # kg
R_MOON = 1.737e6  # m
A_MOON = 3.844e8  # m (semi-major axis)

# Love number and dissipation factor
k2_MOON = 0.024  # Tidal Love number for the Moon
Q_MOON = 30  # Quality factor (dissipation)


def tidal_torque(M_primary, R_secondary, a, omega_spin, omega_orb, k2, Q):
    """
    Calculate tidal torque on a tidally-deformed body.

    The torque is:
    tau = (3/2) * k2 * (G * M_p^2 * R_s^5) / (Q * a^6) * sign(omega_spin - omega_orb)

    This simplified formula gives the secular average torque.

    Args:
        M_primary: Mass of the primary (causing tides)
        R_secondary: Radius of the secondary (experiencing tides)
        a: Orbital semi-major axis
        omega_spin: Spin angular velocity of secondary
        omega_orb: Orbital angular velocity
        k2: Tidal Love number
        Q: Quality factor (lower = more dissipation)

    Returns:
        Tidal torque (N*m)
    """
    # Magnitude of torque
    tau_mag = (3/2) * k2 * (G * M_primary**2 * R_secondary**5) / (Q * a**6)

    # Sign depends on whether spinning faster or slower than orbit
    sign = np.sign(omega_spin - omega_orb)

    return -tau_mag * sign  # Negative to slow down if spinning too fast


def tidal_heating_rate(M_primary, R_secondary, a, omega_spin, omega_orb, k2, Q):
    """
    Calculate tidal heating rate (power dissipated).

    P = tau * |omega_spin - omega_orb|

    Returns:
        Heating power (W)
    """
    tau = abs(tidal_torque(M_primary, R_secondary, a, omega_spin, omega_orb, k2, Q))
    return tau * abs(omega_spin - omega_orb)


def moment_of_inertia_sphere(M, R):
    """Moment of inertia of a uniform sphere: I = (2/5) M R^2"""
    return 0.4 * M * R**2


def orbital_angular_velocity(M_total, a):
    """Kepler's third law: omega = sqrt(G*M/a^3)"""
    return np.sqrt(G * M_total / a**3)


def tidal_locking_timescale(M_primary, M_secondary, R_secondary, a, omega_spin_0,
                            k2, Q):
    """
    Estimate the tidal locking timescale.

    tau_lock ~ (Q / k2) * (M_s / M_p) * (a / R_s)^6 * (1 / omega_orb)

    This is an order-of-magnitude estimate.
    """
    omega_orb = orbital_angular_velocity(M_primary + M_secondary, a)
    I = moment_of_inertia_sphere(M_secondary, R_secondary)

    # Average torque magnitude
    tau_avg = abs(tidal_torque(M_primary, R_secondary, a, omega_spin_0, omega_orb, k2, Q))

    # Time to change angular momentum significantly
    L = I * abs(omega_spin_0 - omega_orb)
    tau_lock = L / tau_avg if tau_avg > 0 else np.inf

    return tau_lock


def simulate_tidal_evolution(M_primary, M_secondary, R_secondary, a,
                             omega_spin_0, k2, Q, t_max, n_steps=1000):
    """
    Simulate the evolution of spin rate toward synchronous rotation.

    Args:
        M_primary: Mass of primary
        M_secondary: Mass of secondary (the spinning body)
        R_secondary: Radius of secondary
        a: Orbital distance
        omega_spin_0: Initial spin rate
        k2: Love number
        Q: Quality factor
        t_max: Maximum simulation time
        n_steps: Number of time steps

    Returns:
        Dictionary with evolution data
    """
    omega_orb = orbital_angular_velocity(M_primary + M_secondary, a)
    I = moment_of_inertia_sphere(M_secondary, R_secondary)

    times = np.linspace(0, t_max, n_steps)
    dt = times[1] - times[0]

    omega_spin = np.zeros(n_steps)
    omega_spin[0] = omega_spin_0

    torques = np.zeros(n_steps)
    heating = np.zeros(n_steps)

    for i in range(1, n_steps):
        tau = tidal_torque(M_primary, R_secondary, a, omega_spin[i-1], omega_orb, k2, Q)
        torques[i-1] = tau

        heating[i-1] = tidal_heating_rate(M_primary, R_secondary, a,
                                          omega_spin[i-1], omega_orb, k2, Q)

        # Angular acceleration: alpha = tau / I
        alpha = tau / I

        # Update spin rate
        omega_spin[i] = omega_spin[i-1] + alpha * dt

        # Check if locked (within 0.1% of synchronous)
        if abs(omega_spin[i] - omega_orb) / omega_orb < 0.001:
            omega_spin[i:] = omega_orb
            break

    torques[-1] = torques[-2]
    heating[-1] = heating[-2]

    return {
        'times': times,
        'omega_spin': omega_spin,
        'omega_orb': omega_orb,
        'torques': torques,
        'heating': heating,
        'period_spin': 2 * np.pi / omega_spin,
        'period_orb': 2 * np.pi / omega_orb
    }


def main():
    fig = plt.figure(figsize=(16, 12))

    # --- Plot 1: Tidal evolution of a fast-spinning moon ---
    ax1 = fig.add_subplot(2, 3, 1)

    # Start with Moon spinning once per 6 hours (like early Moon)
    omega_spin_0 = 2 * np.pi / (6 * 3600)  # rad/s
    t_max = 1e9 * 365.25 * 24 * 3600  # 1 billion years in seconds

    result = simulate_tidal_evolution(
        M_EARTH, M_MOON, R_MOON, A_MOON,
        omega_spin_0, k2_MOON, Q_MOON, t_max, n_steps=2000
    )

    times_Gyr = result['times'] / (1e9 * 365.25 * 24 * 3600)
    period_days = result['period_spin'] / (24 * 3600)
    period_orb_days = result['period_orb'] / (24 * 3600)

    ax1.semilogy(times_Gyr, period_days, 'b-', lw=2, label='Spin period')
    ax1.axhline(period_orb_days, color='r', linestyle='--', lw=2,
                label=f'Orbital period ({period_orb_days:.1f} days)')

    ax1.set_xlabel('Time (billion years)')
    ax1.set_ylabel('Spin Period (days)')
    ax1.set_title('Moon Spin Evolution\n(Starting at 6-hour rotation)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Different initial spin rates ---
    ax2 = fig.add_subplot(2, 3, 2)

    initial_periods_hours = [4, 6, 12, 24, 100]  # hours
    colors = plt.cm.viridis(np.linspace(0, 1, len(initial_periods_hours)))

    for T0, color in zip(initial_periods_hours, colors):
        omega0 = 2 * np.pi / (T0 * 3600)
        result = simulate_tidal_evolution(
            M_EARTH, M_MOON, R_MOON, A_MOON,
            omega0, k2_MOON, Q_MOON, t_max, n_steps=1000
        )

        times_Gyr = result['times'] / (1e9 * 365.25 * 24 * 3600)
        period_days = result['period_spin'] / (24 * 3600)

        ax2.semilogy(times_Gyr, period_days, '-', color=color, lw=2,
                     label=f'T0 = {T0} hr')

    ax2.axhline(period_orb_days, color='red', linestyle='--', lw=2)
    ax2.set_xlabel('Time (billion years)')
    ax2.set_ylabel('Spin Period (days)')
    ax2.set_title('Effect of Initial Spin Rate')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- Plot 3: Locking timescale vs orbital distance ---
    ax3 = fig.add_subplot(2, 3, 3)

    distances = np.logspace(8, 9.5, 50)  # 100,000 km to 3 million km
    timescales = []

    omega_init = 2 * np.pi / (10 * 3600)  # 10 hour initial rotation

    for a in distances:
        tau = tidal_locking_timescale(M_EARTH, M_MOON, R_MOON, a,
                                      omega_init, k2_MOON, Q_MOON)
        timescales.append(tau / (1e9 * 365.25 * 24 * 3600))  # Convert to Gyr

    ax3.loglog(distances/1e6, timescales, 'b-', lw=2)
    ax3.axvline(A_MOON/1e6, color='r', linestyle='--',
                label=f'Current Moon distance ({A_MOON/1e6:.0f} km)')
    ax3.axhline(4.5, color='g', linestyle=':', label='Age of Solar System')

    ax3.set_xlabel('Orbital Distance (1000 km)')
    ax3.set_ylabel('Locking Timescale (Gyr)')
    ax3.set_title('Locking Time vs Distance\n(~a^6 dependence)')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # --- Plot 4: Tidal heating over time ---
    ax4 = fig.add_subplot(2, 3, 4)

    omega_spin_0 = 2 * np.pi / (6 * 3600)
    result = simulate_tidal_evolution(
        M_EARTH, M_MOON, R_MOON, A_MOON,
        omega_spin_0, k2_MOON, Q_MOON, t_max, n_steps=2000
    )

    times_Gyr = result['times'] / (1e9 * 365.25 * 24 * 3600)
    heating_TW = result['heating'] / 1e12  # Convert to TW

    ax4.semilogy(times_Gyr, heating_TW + 1e-10, 'r-', lw=2)
    ax4.set_xlabel('Time (billion years)')
    ax4.set_ylabel('Tidal Heating Power (TW)')
    ax4.set_title('Tidal Heating During Evolution\n(Dissipation in Moon interior)')
    ax4.grid(True, alpha=0.3)

    # Add Io for comparison
    ax4.axhline(100, color='orange', linestyle='--', alpha=0.7)
    ax4.annotate("Io's current heating (~100 TW)", xy=(0.5, 100),
                 fontsize=9, color='orange')

    # --- Plot 5: Effect of Q factor ---
    ax5 = fig.add_subplot(2, 3, 5)

    Q_values = [10, 30, 100, 300]
    colors = ['red', 'orange', 'green', 'blue']

    for Q, color in zip(Q_values, colors):
        omega0 = 2 * np.pi / (6 * 3600)
        result = simulate_tidal_evolution(
            M_EARTH, M_MOON, R_MOON, A_MOON,
            omega0, k2_MOON, Q, t_max, n_steps=1000
        )

        times_Gyr = result['times'] / (1e9 * 365.25 * 24 * 3600)
        period_days = result['period_spin'] / (24 * 3600)

        ax5.semilogy(times_Gyr, period_days, '-', color=color, lw=2,
                     label=f'Q = {Q}')

    ax5.axhline(period_orb_days, color='black', linestyle='--', lw=1)
    ax5.set_xlabel('Time (billion years)')
    ax5.set_ylabel('Spin Period (days)')
    ax5.set_title('Effect of Quality Factor Q\n(Lower Q = faster locking)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # --- Plot 6: Theory summary ---
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    summary = """Tidal Locking: Moon Always Shows Same Face
==========================================

TIDAL TORQUE:
-------------
The gravitational gradient creates tidal bulges.
If spinning faster than orbit, bulge leads the
planet and is pulled back, slowing rotation.

tau = (3/2) * k2 * (G*M_p^2 * R_s^5) / (Q*a^6)

KEY PARAMETERS:
---------------
k2: Tidal Love number (~0.024 for Moon)
    Measures body's response to tidal force

Q:  Quality factor (~30 for Moon)
    Lower Q = more dissipation = faster locking

LOCKING TIMESCALE:
------------------
tau_lock ~ (Q/k2) * (M_s/M_p) * (a/R_s)^6 / omega_orb

Strong dependence on distance: ~a^6

EXAMPLES IN SOLAR SYSTEM:
-------------------------
Body          Locked?   Reason
Moon          Yes       Close, old
Phobos        Yes       Very close
Io            Yes       Very close to Jupiter
Titan         Yes       Moderate distance, old
Pluto-Charon  Both!     Mutually locked

NOT LOCKED:
Mercury       3:2 resonance instead
Mars (moons)  Phobos yes, Deimos almost
Most asteroids  Too far or irregular

CONSEQUENCES:
-------------
1. Same face always visible (near side)
2. Tidal heating during spin-down
3. Libration allows seeing 59% of Moon
4. Contributes to early lunar volcanism"""

    ax6.text(0.02, 0.98, summary, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('Tidal Locking Timescales (Experiment 58)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'tidal_locking.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {output_path}")

    # Print analysis
    print("\nTidal Locking Analysis:")
    print("-" * 50)
    print(f"Moon orbital period: {2*np.pi/orbital_angular_velocity(M_EARTH+M_MOON, A_MOON)/(24*3600):.2f} days")
    print(f"Moon k2 (Love number): {k2_MOON}")
    print(f"Moon Q (quality factor): {Q_MOON}")

    tau_lock = tidal_locking_timescale(M_EARTH, M_MOON, R_MOON, A_MOON,
                                       2*np.pi/(6*3600), k2_MOON, Q_MOON)
    print(f"\nEstimated locking timescale: {tau_lock/(1e9*365.25*24*3600):.2f} Gyr")
    print("(Actual locking occurred ~4 billion years ago)")


if __name__ == "__main__":
    main()
