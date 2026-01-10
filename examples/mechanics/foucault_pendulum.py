"""
Experiment 57: Foucault Pendulum Precession.

Demonstrates the Foucault pendulum, which provides direct evidence of
Earth's rotation. The pendulum's plane of oscillation precesses due
to the Coriolis effect.

Key physics:
1. Precession rate: Omega_p = Omega_Earth * sin(latitude)
2. At the poles: full rotation in one sidereal day
3. At equator: no precession
4. Precession is clockwise in Northern hemisphere

The equations of motion in the rotating frame include the Coriolis force,
which causes the apparent precession of the swing plane.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Earth parameters
OMEGA_EARTH = 7.2921e-5  # rad/s
SIDEREAL_DAY = 86164  # seconds


def simulate_foucault_pendulum(length, latitude, x0, y0, vx0, vy0,
                               t_max, dt=0.01, g=9.81):
    """
    Simulate a Foucault pendulum using small-angle approximation.

    In the rotating Earth frame with Coriolis effect:
    x'' + omega0^2 * x = 2*Omega_z * y'
    y'' + omega0^2 * y = -2*Omega_z * x'

    where omega0 = sqrt(g/L) and Omega_z = Omega * sin(latitude)

    Args:
        length: Pendulum length (m)
        latitude: Geographic latitude (degrees)
        x0, y0: Initial position (m)
        vx0, vy0: Initial velocity (m/s)
        t_max: Simulation time (s)
        dt: Time step (s)
        g: Gravitational acceleration (m/s^2)

    Returns:
        Dictionary with trajectory data
    """
    omega0 = np.sqrt(g / length)  # Natural frequency
    Omega_z = OMEGA_EARTH * np.sin(np.radians(latitude))  # Vertical component

    # State: [x, y, vx, vy]
    state = np.array([x0, y0, vx0, vy0], dtype=float)

    times = [0.0]
    x_hist = [x0]
    y_hist = [y0]
    vx_hist = [vx0]
    vy_hist = [vy0]

    def derivatives(s):
        x, y, vx, vy = s
        ax = -omega0**2 * x + 2 * Omega_z * vy
        ay = -omega0**2 * y - 2 * Omega_z * vx
        return np.array([vx, vy, ax, ay])

    t = 0
    while t < t_max:
        # RK4 integration
        k1 = derivatives(state)
        k2 = derivatives(state + 0.5*dt*k1)
        k3 = derivatives(state + 0.5*dt*k2)
        k4 = derivatives(state + dt*k3)

        state = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        t += dt

        times.append(t)
        x_hist.append(state[0])
        y_hist.append(state[1])
        vx_hist.append(state[2])
        vy_hist.append(state[3])

    return {
        'times': np.array(times),
        'x': np.array(x_hist),
        'y': np.array(y_hist),
        'vx': np.array(vx_hist),
        'vy': np.array(vy_hist),
        'latitude': latitude,
        'length': length,
        'omega0': omega0,
        'Omega_z': Omega_z
    }


def analytic_precession_rate(latitude):
    """
    Analytic precession rate of Foucault pendulum.

    Omega_precession = Omega_Earth * sin(latitude)
    """
    return OMEGA_EARTH * np.sin(np.radians(latitude))


def precession_period(latitude):
    """
    Time for one complete precession (rotation of swing plane).

    T = 24 hours / sin(latitude) (in sidereal time)
    """
    if abs(latitude) < 0.1:
        return np.inf
    return SIDEREAL_DAY / abs(np.sin(np.radians(latitude)))


def main():
    fig = plt.figure(figsize=(16, 12))

    # Pendulum parameters
    length = 67.0  # meters (like the original Pantheon pendulum)
    g = 9.81

    # --- Plot 1: 3D trajectory at Paris latitude ---
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')

    latitude = 48.8  # Paris
    amplitude = 5.0  # meters
    t_max = 3600 * 4  # 4 hours

    result = simulate_foucault_pendulum(
        length, latitude, amplitude, 0, 0, 0, t_max, dt=0.05
    )

    # Sample every 100th point for clarity
    skip = 100
    x, y = result['x'][::skip], result['y'][::skip]
    t = result['times'][::skip]

    # Color by time
    colors = plt.cm.viridis(t / t.max())
    for i in range(len(x)-1):
        ax1.plot([x[i], x[i+1]], [y[i], y[i+1]], [0, 0],
                 color=colors[i], lw=0.5)

    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_zlabel('z (m)')
    ax1.set_title(f'Foucault Pendulum at {latitude}N (Paris)\n'
                  f'4 hours of oscillation')

    # --- Plot 2: Top-down view showing precession ---
    ax2 = fig.add_subplot(2, 3, 2)

    # Show full trajectory as light gray
    ax2.plot(result['x'], result['y'], 'gray', alpha=0.1, lw=0.2)

    # Highlight specific times
    omega0 = result['omega0']
    period = 2 * np.pi / omega0
    n_periods = int(t_max / period)

    # Mark swing plane at different times
    times_to_mark = [0, 1, 2, 3, 4]  # hours
    colors = ['red', 'orange', 'green', 'blue', 'purple']

    for hour, color in zip(times_to_mark, colors):
        t_mark = hour * 3600
        idx = int(t_mark / 0.05)
        if idx < len(result['x']):
            # Find the swing direction at this time
            # Average over a few periods to get the swing axis
            idx_start = max(0, idx - int(period/0.05))
            idx_end = min(len(result['x']), idx + int(period/0.05))

            x_seg = result['x'][idx_start:idx_end]
            y_seg = result['y'][idx_start:idx_end]

            # Principal axis via covariance
            if len(x_seg) > 10:
                cov = np.cov(x_seg, y_seg)
                eigenvalues, eigenvectors = np.linalg.eig(cov)
                major_axis = eigenvectors[:, np.argmax(eigenvalues)]

                # Draw the swing axis
                ax2.arrow(0, 0, major_axis[0]*amplitude, major_axis[1]*amplitude,
                          head_width=0.2, head_length=0.1, fc=color, ec=color,
                          lw=2, label=f't={hour}h')
                ax2.arrow(0, 0, -major_axis[0]*amplitude, -major_axis[1]*amplitude,
                          head_width=0.2, head_length=0.1, fc=color, ec=color, lw=2)

    ax2.set_xlabel('x (East-West, m)')
    ax2.set_ylabel('y (North-South, m)')
    ax2.set_title('Precession of Swing Plane\n(Clockwise in Northern Hemisphere)')
    ax2.set_xlim(-amplitude*1.2, amplitude*1.2)
    ax2.set_ylim(-amplitude*1.2, amplitude*1.2)
    ax2.set_aspect('equal')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.3)

    # --- Plot 3: Precession rate vs latitude ---
    ax3 = fig.add_subplot(2, 3, 3)

    latitudes = np.linspace(-90, 90, 181)
    precession_rates = [analytic_precession_rate(lat) for lat in latitudes]
    precession_periods = [precession_period(lat) / 3600 for lat in latitudes]  # hours

    ax3.plot(latitudes, np.degrees(precession_rates) * 3600, 'b-', lw=2)
    ax3.axhline(0, color='k', linestyle='--', alpha=0.5)

    ax3.set_xlabel('Latitude (degrees)')
    ax3.set_ylabel('Precession Rate (deg/hour)')
    ax3.set_title('Precession Rate = Omega * sin(latitude)')
    ax3.grid(True, alpha=0.3)

    # Mark key latitudes
    key_lats = [90, 48.8, 30, 0]
    labels = ['Pole', 'Paris', '30N', 'Equator']
    for lat, label in zip(key_lats, labels):
        rate = np.degrees(analytic_precession_rate(lat)) * 3600
        ax3.plot(lat, rate, 'ro', markersize=8)
        ax3.annotate(label, (lat, rate), textcoords="offset points",
                     xytext=(5, 5), fontsize=9)

    # --- Plot 4: Comparison at different latitudes ---
    ax4 = fig.add_subplot(2, 3, 4)

    test_latitudes = [90, 60, 45, 30, 0]
    t_max_compare = 3600 * 2  # 2 hours

    for lat in test_latitudes:
        result = simulate_foucault_pendulum(
            length, lat, amplitude, 0, 0, 0, t_max_compare, dt=0.05
        )

        # Calculate swing angle over time
        swing_angles = np.arctan2(result['y'], result['x'])

        # Unwrap and average to get precession
        # (complex due to oscillation - we'll use envelope)

        ax4.plot(result['x'][::50], result['y'][::50], lw=0.5,
                 label=f'{lat}N', alpha=0.7)

    ax4.set_xlabel('x (m)')
    ax4.set_ylabel('y (m)')
    ax4.set_title('Trajectories at Different Latitudes (2 hours)')
    ax4.set_xlim(-amplitude*1.2, amplitude*1.2)
    ax4.set_ylim(-amplitude*1.2, amplitude*1.2)
    ax4.set_aspect('equal')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # --- Plot 5: Period of precession vs latitude ---
    ax5 = fig.add_subplot(2, 3, 5)

    latitudes_pos = np.linspace(5, 90, 86)
    periods_hours = [precession_period(lat) / 3600 for lat in latitudes_pos]

    ax5.semilogy(latitudes_pos, periods_hours, 'b-', lw=2)
    ax5.axhline(24, color='r', linestyle='--', label='Sidereal day (23.93h)')

    ax5.set_xlabel('Latitude (degrees)')
    ax5.set_ylabel('Precession Period (hours)')
    ax5.set_title('Time for Full Rotation of Swing Plane\n'
                  'T = 24h / sin(latitude)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Mark famous Foucault pendulum locations
    locations = [
        (48.8, 'Paris'),
        (40.7, 'New York'),
        (51.5, 'London'),
        (35.7, 'Tokyo'),
    ]
    for lat, city in locations:
        T = precession_period(lat) / 3600
        ax5.plot(lat, T, 'ro', markersize=8)
        ax5.annotate(city, (lat, T), textcoords="offset points",
                     xytext=(5, 5), fontsize=9)

    # --- Plot 6: Theory summary ---
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    summary = """Foucault Pendulum: Proof of Earth's Rotation
=============================================

The Foucault pendulum demonstrates Earth's rotation through
the precession of its swing plane.

EQUATIONS OF MOTION (small angle):
----------------------------------
    x'' + omega_0^2 * x = 2*Omega_z * y'
    y'' + omega_0^2 * y = -2*Omega_z * x'

where:
    omega_0 = sqrt(g/L)     (natural frequency)
    Omega_z = Omega * sin(phi)  (vertical Earth rotation)

PRECESSION RATE:
----------------
    Omega_precession = Omega_Earth * sin(latitude)

    At pole:    360 deg / sidereal day
    At Paris:   ~11.3 deg/hour
    At equator: 0 deg/hour

PRECESSION PERIOD:
------------------
    T = (sidereal day) / sin(latitude)

    At 90N:  23.93 hours (one day)
    At 48.8N: 32.0 hours (Paris)
    At 30N:  47.9 hours
    At 0:    infinite (no precession)

HISTORICAL NOTE:
----------------
Leon Foucault demonstrated this in 1851 at the
Pantheon in Paris with a 67m pendulum, providing
the first direct visual proof of Earth's rotation.

The precession is CLOCKWISE in the Northern
hemisphere and COUNTERCLOCKWISE in the Southern."""

    ax6.text(0.02, 0.98, summary, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.suptitle('Foucault Pendulum Precession (Experiment 57)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'foucault_pendulum.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {output_path}")

    # Print analysis
    print("\nFoucault Pendulum Analysis:")
    print("-" * 50)
    print(f"Pendulum length: {length} m")
    print(f"Natural period: {2*np.pi/np.sqrt(g/length):.2f} s")
    print(f"\nPrecession rates at key latitudes:")
    for lat in [90, 60, 48.8, 45, 30, 0]:
        rate = np.degrees(analytic_precession_rate(lat)) * 3600
        period = precession_period(lat) / 3600
        print(f"  {lat:5.1f}N: {rate:6.2f} deg/hour (period: {period:.1f} hours)")


if __name__ == "__main__":
    main()
