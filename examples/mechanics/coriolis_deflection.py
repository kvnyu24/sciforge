"""
Experiment 56: Coriolis Effect on Projectile Motion.

Demonstrates the Coriolis effect on projectile motion on a rotating Earth.
The Coriolis acceleration is: a_cor = -2 * omega x v

Key effects:
1. Deflection to the right (left) in the Northern (Southern) hemisphere
2. Deflection magnitude depends on latitude and velocity
3. No deflection at equator for horizontal motion
4. Maximum deflection at the poles

The equations in the rotating frame:
    a = g - 2*omega x v - omega x (omega x r)

where the last term is centrifugal (usually small for Earth).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


# Earth parameters
OMEGA_EARTH = 7.2921e-5  # rad/s (one rotation per sidereal day)
R_EARTH = 6.371e6  # meters
g = 9.81  # m/s^2


def omega_vector(latitude):
    """
    Earth's angular velocity vector at given latitude.

    In local coordinates (East, North, Up):
    omega = Omega * (0, cos(lat), sin(lat))
    """
    lat_rad = np.radians(latitude)
    return OMEGA_EARTH * np.array([0, np.cos(lat_rad), np.sin(lat_rad)])


def simulate_projectile(v0, launch_angle, azimuth, latitude, t_max, dt=0.01,
                        include_coriolis=True):
    """
    Simulate projectile motion in rotating Earth frame.

    Args:
        v0: Initial speed (m/s)
        launch_angle: Angle from horizontal (degrees)
        azimuth: Launch direction, 0=East, 90=North (degrees)
        latitude: Geographic latitude (degrees)
        t_max: Maximum simulation time (s)
        dt: Time step (s)
        include_coriolis: Whether to include Coriolis force

    Returns:
        Dictionary with trajectory data
    """
    # Initial velocity in local ENU coordinates (East, North, Up)
    launch_rad = np.radians(launch_angle)
    azimuth_rad = np.radians(azimuth)

    vx0 = v0 * np.cos(launch_rad) * np.sin(azimuth_rad)  # East
    vy0 = v0 * np.cos(launch_rad) * np.cos(azimuth_rad)  # North
    vz0 = v0 * np.sin(launch_rad)  # Up

    state = np.array([0.0, 0.0, 0.0, vx0, vy0, vz0])  # [x, y, z, vx, vy, vz]
    omega = omega_vector(latitude)

    times = [0.0]
    positions = [state[:3].copy()]
    velocities = [state[3:].copy()]

    t = 0
    while t < t_max and state[2] >= 0:  # Stop when projectile hits ground
        r = state[:3]
        v = state[3:]

        # Gravity (downward)
        a_gravity = np.array([0.0, 0.0, -g])

        # Coriolis acceleration: -2 * omega x v
        if include_coriolis:
            a_coriolis = -2 * np.cross(omega, v)
        else:
            a_coriolis = np.zeros(3)

        # Total acceleration
        a = a_gravity + a_coriolis

        # RK4 integration
        def derivatives(s):
            v = s[3:]
            if include_coriolis:
                a_cor = -2 * np.cross(omega, v)
            else:
                a_cor = np.zeros(3)
            a = a_gravity + a_cor
            return np.concatenate([v, a])

        k1 = derivatives(state)
        k2 = derivatives(state + 0.5*dt*k1)
        k3 = derivatives(state + 0.5*dt*k2)
        k4 = derivatives(state + dt*k3)

        state = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        t += dt

        times.append(t)
        positions.append(state[:3].copy())
        velocities.append(state[3:].copy())

    return {
        'times': np.array(times),
        'positions': np.array(positions),
        'velocities': np.array(velocities),
        'latitude': latitude
    }


def calculate_deflection(positions):
    """
    Calculate the deflection from a straight-line trajectory.

    Returns:
        East-West deflection (positive = East)
    """
    # Deflection is the x-component (East) at landing
    return positions[-1, 0]


def main():
    fig = plt.figure(figsize=(16, 12))

    # --- Plot 1: Trajectory comparison with/without Coriolis ---
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')

    v0 = 500  # m/s
    launch_angle = 45
    azimuth = 0  # Fire East
    latitude = 45

    traj_with = simulate_projectile(v0, launch_angle, azimuth, latitude, 100,
                                    include_coriolis=True)
    traj_without = simulate_projectile(v0, launch_angle, azimuth, latitude, 100,
                                       include_coriolis=False)

    pos_w = traj_with['positions']
    pos_wo = traj_without['positions']

    ax1.plot(pos_w[:, 0], pos_w[:, 1], pos_w[:, 2], 'b-', lw=2,
             label='With Coriolis')
    ax1.plot(pos_wo[:, 0], pos_wo[:, 1], pos_wo[:, 2], 'r--', lw=2,
             label='Without Coriolis')

    ax1.set_xlabel('East (m)')
    ax1.set_ylabel('North (m)')
    ax1.set_zlabel('Up (m)')
    ax1.set_title(f'Projectile at {latitude}N\nv0={v0} m/s, angle={launch_angle} deg')
    ax1.legend()

    # --- Plot 2: Top-down view showing deflection ---
    ax2 = fig.add_subplot(2, 3, 2)

    ax2.plot(pos_w[:, 0], pos_w[:, 1], 'b-', lw=2, label='With Coriolis')
    ax2.plot(pos_wo[:, 0], pos_wo[:, 1], 'r--', lw=2, label='Without Coriolis')
    ax2.plot(pos_w[0, 0], pos_w[0, 1], 'go', markersize=10, label='Launch')
    ax2.plot(pos_w[-1, 0], pos_w[-1, 1], 'bx', markersize=10)
    ax2.plot(pos_wo[-1, 0], pos_wo[-1, 1], 'rx', markersize=10)

    deflection = pos_w[-1, 1] - pos_wo[-1, 1]
    ax2.annotate(f'Deflection: {deflection:.1f} m',
                 xy=(pos_w[-1, 0], pos_w[-1, 1]),
                 xytext=(pos_w[-1, 0] - 5000, pos_w[-1, 1] + 500),
                 arrowprops=dict(arrowstyle='->', color='blue'),
                 fontsize=10)

    ax2.set_xlabel('East (m)')
    ax2.set_ylabel('North (m)')
    ax2.set_title('Top-Down View\n(Deflection to the right in Northern Hemisphere)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    # --- Plot 3: Deflection vs latitude ---
    ax3 = fig.add_subplot(2, 3, 3)

    latitudes = np.linspace(-90, 90, 37)
    deflections = []

    for lat in latitudes:
        traj_w = simulate_projectile(v0, launch_angle, 0, lat, 100, include_coriolis=True)
        traj_wo = simulate_projectile(v0, launch_angle, 0, lat, 100, include_coriolis=False)

        # North-south deflection (y-component)
        defl = traj_w['positions'][-1, 1] - traj_wo['positions'][-1, 1]
        deflections.append(defl)

    ax3.plot(latitudes, deflections, 'b-', lw=2)
    ax3.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax3.axvline(0, color='k', linestyle='--', alpha=0.5)

    ax3.set_xlabel('Latitude (degrees)')
    ax3.set_ylabel('Deflection (m)')
    ax3.set_title('Deflection vs Latitude\n(Firing East, v0=500 m/s)')
    ax3.grid(True, alpha=0.3)

    # Mark hemispheres
    ax3.fill_between([0, 90], [min(deflections)]*2, [max(deflections)]*2,
                     alpha=0.1, color='blue', label='Northern (deflects right)')
    ax3.fill_between([-90, 0], [min(deflections)]*2, [max(deflections)]*2,
                     alpha=0.1, color='red', label='Southern (deflects left)')
    ax3.legend(fontsize=8)

    # --- Plot 4: Different launch speeds ---
    ax4 = fig.add_subplot(2, 3, 4)

    speeds = [100, 300, 500, 800, 1000]
    colors = plt.cm.viridis(np.linspace(0, 1, len(speeds)))

    for v, color in zip(speeds, colors):
        latitudes_subset = np.linspace(0, 90, 19)
        defls = []
        for lat in latitudes_subset:
            traj_w = simulate_projectile(v, 45, 0, lat, 200, include_coriolis=True)
            traj_wo = simulate_projectile(v, 45, 0, lat, 200, include_coriolis=False)
            defl = np.abs(traj_w['positions'][-1, 1] - traj_wo['positions'][-1, 1])
            defls.append(defl)

        ax4.plot(latitudes_subset, defls, '-', color=color, lw=2, label=f'v0={v} m/s')

    ax4.set_xlabel('Latitude (degrees)')
    ax4.set_ylabel('|Deflection| (m)')
    ax4.set_title('Deflection Magnitude vs Latitude\nfor Different Launch Speeds')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # --- Plot 5: Long-range artillery example ---
    ax5 = fig.add_subplot(2, 3, 5)

    # Paris Gun parameters (WWI long-range artillery)
    v0_gun = 1600  # m/s (approximate muzzle velocity)
    angle_gun = 50  # degrees

    # Simulate for different azimuths
    azimuths = [0, 45, 90, 135, 180, 225, 270, 315]

    for az in azimuths:
        traj = simulate_projectile(v0_gun, angle_gun, az, 49, 200,
                                   include_coriolis=True, dt=0.1)
        pos = traj['positions']
        ax5.plot(pos[:, 0]/1000, pos[:, 1]/1000, lw=1.5,
                 label=f'Azimuth={az} deg')

    ax5.plot(0, 0, 'ko', markersize=10)
    ax5.set_xlabel('East (km)')
    ax5.set_ylabel('North (km)')
    ax5.set_title('Long-Range Artillery (Paris Gun)\n'
                  f'v0={v0_gun} m/s, lat=49N, angle={angle_gun} deg')
    ax5.legend(fontsize=7, ncol=2)
    ax5.grid(True, alpha=0.3)
    ax5.set_aspect('equal')

    # --- Plot 6: Theoretical explanation ---
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    summary = """Coriolis Effect on Projectile Motion
=====================================

In a rotating reference frame, the equation of motion is:

    a = g - 2*omega x v - omega x (omega x r)
                |             |
           Coriolis      Centrifugal

For Earth:
- omega = 7.29 x 10^-5 rad/s
- Centrifugal term is small (already in "g")
- Coriolis dominates for moving objects

KEY RESULTS:
------------

1. DEFLECTION DIRECTION
   - Northern hemisphere: deflect RIGHT
   - Southern hemisphere: deflect LEFT
   - At equator: minimal horizontal deflection

2. MAGNITUDE
   - Proportional to velocity
   - Proportional to sin(latitude) for horizontal motion
   - delta ~ 2 * omega * v * t * sin(lat)

3. PRACTICAL EXAMPLES
   - Artillery: 100-1000m deflection at 100km range
   - Missiles: Must correct for Coriolis
   - Hurricanes: Cyclonic rotation

4. THE CORIOLIS PARAMETER
   f = 2 * omega * sin(latitude)
   f = 0 at equator
   f = 1.46 x 10^-4 /s at poles

The Coriolis force is essential for:
- Weather systems
- Ocean currents
- Long-range ballistics
- Sniper shots at extreme range"""

    ax6.text(0.02, 0.98, summary, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('Coriolis Effect on Projectile Motion (Experiment 56)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'coriolis_deflection.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {output_path}")

    # Print numerical results
    print("\nCoriolis Deflection Analysis:")
    print("-" * 50)
    print(f"Earth's angular velocity: {OMEGA_EARTH:.4e} rad/s")
    print(f"\nExample: v0=500 m/s, angle=45 deg, lat=45N")
    print(f"  Range: {pos_w[-1, 0]/1000:.2f} km")
    print(f"  Deflection: {pos_w[-1, 1] - pos_wo[-1, 1]:.1f} m (right/south)")

    # Theoretical estimate
    flight_time = traj_with['times'][-1]
    v_avg = v0 * np.cos(np.radians(launch_angle))  # Approximate
    lat_rad = np.radians(latitude)
    defl_theory = OMEGA_EARTH * v_avg * flight_time**2 * np.sin(lat_rad)
    print(f"  Theoretical estimate: {defl_theory:.1f} m")


if __name__ == "__main__":
    main()
