"""
Experiment 194: Gravitational Redshift

This experiment demonstrates gravitational redshift - the shift in frequency
of light as it climbs out of or falls into a gravitational potential.

Physical concepts:
- Gravitational time dilation
- Pound-Rebka experiment
- GPS satellite corrections
- Black hole redshift (extreme case)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.relativity import SchwarzschildMetric, GravitationalRedshift


# Physical constants
G = 6.67430e-11  # m^3/(kg*s^2)
c = 299792458.0  # m/s
M_sun = 1.989e30  # kg
M_earth = 5.972e24  # kg
R_earth = 6.371e6  # m


def gravitational_redshift(r_emit, r_obs, M, G=G, c=c):
    """
    Calculate gravitational redshift z.

    z = (lambda_obs - lambda_emit) / lambda_emit
    z = sqrt(g_tt(r_obs) / g_tt(r_emit)) - 1

    For Schwarzschild:
    z = sqrt((1 - r_s/r_obs) / (1 - r_s/r_emit)) - 1

    For weak field:
    z ≈ GM(1/r_emit - 1/r_obs) / c^2

    Args:
        r_emit: Radius of emission
        r_obs: Radius of observation
        M: Central mass

    Returns:
        Redshift z (positive means observed wavelength is longer)
    """
    rs = 2 * G * M / c**2

    g_tt_emit = 1 - rs / r_emit
    g_tt_obs = 1 - rs / r_obs

    if g_tt_emit <= 0:
        return np.inf  # At or inside horizon

    z = np.sqrt(g_tt_obs / g_tt_emit) - 1
    return z


def gravitational_redshift_weak(r_emit, r_obs, M, G=G, c=c):
    """Weak field approximation for gravitational redshift."""
    return G * M * (1/r_emit - 1/r_obs) / c**2


def time_dilation_factor(r, M, G=G, c=c):
    """
    Calculate gravitational time dilation factor.

    d_tau / dt = sqrt(1 - r_s/r)

    Clocks run slower (d_tau/dt < 1) deeper in gravitational well.
    """
    rs = 2 * G * M / c**2
    return np.sqrt(1 - rs / r)


def gps_time_correction(h, M=M_earth, R=R_earth, v_orbit=3870, G=G, c=c):
    """
    Calculate GPS satellite clock correction.

    GPS satellites experience:
    1. Gravitational effect: clocks run FASTER (higher altitude)
    2. Special relativistic effect: clocks run SLOWER (velocity)

    Args:
        h: Orbital altitude above surface
        M: Central mass (Earth)
        R: Central body radius
        v_orbit: Orbital velocity

    Returns:
        Net time difference per day (in microseconds)
    """
    r = R + h

    # Gravitational time dilation (satellite vs surface)
    # dt_sat/dt_surface = sqrt((1-rs/r)/(1-rs/R))
    rs = 2 * G * M / c**2
    gravitational = np.sqrt((1 - rs/r) / (1 - rs/R)) - 1

    # Special relativistic time dilation
    # At surface, rotation velocity is ~465 m/s at equator
    v_surface = 465  # m/s
    sr_satellite = np.sqrt(1 - (v_orbit/c)**2) - 1
    sr_surface = np.sqrt(1 - (v_surface/c)**2) - 1
    special_rel = sr_satellite - sr_surface

    # Total: positive means satellite clock runs faster
    total = gravitational + special_rel

    # Convert to microseconds per day
    seconds_per_day = 86400
    return total * seconds_per_day * 1e6, gravitational * seconds_per_day * 1e6, special_rel * seconds_per_day * 1e6


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ==========================================================================
    # Plot 1: Gravitational redshift vs radius (Schwarzschild)
    # ==========================================================================
    ax1 = axes[0, 0]

    # Use solar mass
    M = M_sun
    rs = 2 * G * M / c**2

    r_range = np.linspace(1.1 * rs, 100 * rs, 500)

    # Redshift from r to infinity
    z_exact = []
    z_weak = []
    for r in r_range:
        z_exact.append(gravitational_redshift(r, np.inf, M))
        z_weak.append(gravitational_redshift_weak(r, np.inf, M))

    z_exact = np.array(z_exact)
    z_weak = np.array(z_weak)

    ax1.semilogy(r_range/rs, z_exact, 'b-', lw=2, label='Exact (Schwarzschild)')
    ax1.semilogy(r_range/rs, z_weak, 'r--', lw=2, label='Weak field approx.')

    ax1.axvline(x=1.5, color='green', linestyle=':', alpha=0.7,
               label='Photon sphere (1.5 r_s)')
    ax1.axvline(x=3, color='purple', linestyle=':', alpha=0.7,
               label='ISCO (3 r_s)')

    ax1.set_xlabel('Emission radius r / r_s')
    ax1.set_ylabel('Redshift z')
    ax1.set_title('Gravitational Redshift to Infinity')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, 100)

    # Mark event horizon
    ax1.fill_betweenx([1e-4, 1e3], 0, 1, alpha=0.2, color='black')
    ax1.text(0.5, 0.1, 'Horizon', fontsize=10, ha='center')

    # ==========================================================================
    # Plot 2: Pound-Rebka experiment
    # ==========================================================================
    ax2 = axes[0, 1]

    # Pound-Rebka: tower height = 22.5 m
    h_tower = 22.5  # m
    r_bottom = R_earth
    r_top = R_earth + h_tower

    z_tower = gravitational_redshift(r_bottom, r_top, M_earth)

    # Show fractional frequency shift
    heights = np.linspace(0, 100, 100)  # m
    z_heights = []
    for h in heights:
        z_heights.append(gravitational_redshift(R_earth, R_earth + h, M_earth))

    ax2.plot(heights, np.array(z_heights) * 1e15, 'b-', lw=2)

    # Mark Pound-Rebka
    ax2.plot(22.5, z_tower * 1e15, 'ro', markersize=10)
    ax2.annotate(f'Pound-Rebka (1959)\nh = 22.5 m\nz = {z_tower:.2e}',
                xy=(22.5, z_tower * 1e15), xytext=(40, z_tower * 1e15 * 0.7),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax2.set_xlabel('Height above surface (m)')
    ax2.set_ylabel('Fractional frequency shift (x10^-15)')
    ax2.set_title('Gravitational Redshift on Earth\n(Pound-Rebka Experiment)')
    ax2.grid(True, alpha=0.3)

    # Add physics note
    ax2.text(0.95, 0.05,
            'Measured using Mossbauer effect\n'
            'with Fe-57 gamma rays.\n\n'
            'Result: z = (2.57 +/- 0.26) x 10^-15\n'
            'GR prediction: z = 2.46 x 10^-15',
            transform=ax2.transAxes, fontsize=9, va='bottom', ha='right',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # ==========================================================================
    # Plot 3: GPS satellite time corrections
    # ==========================================================================
    ax3 = axes[1, 0]

    # GPS altitude
    h_gps = 20200e3  # 20,200 km altitude
    v_gps = 3870  # m/s orbital velocity

    total, grav, sr = gps_time_correction(h_gps, v_orbit=v_gps)

    effects = ['Gravitational\n(altitude)', 'Special Relativistic\n(velocity)', 'Net Effect']
    values = [grav, sr, total]
    colors = ['blue', 'red', 'green']

    bars = ax3.bar(effects, values, color=colors, alpha=0.7, edgecolor='black')

    ax3.axhline(y=0, color='gray', linestyle='-', lw=1)

    for bar, val in zip(bars, values):
        height = bar.get_height()
        y_pos = height + 2 if height >= 0 else height - 5
        ax3.text(bar.get_x() + bar.get_width()/2, y_pos, f'{val:.1f} us/day',
                ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)

    ax3.set_ylabel('Clock difference (microseconds/day)')
    ax3.set_title('GPS Satellite Clock Corrections')
    ax3.grid(True, alpha=0.3, axis='y')

    ax3.text(0.5, -0.15,
            f'GPS altitude: {h_gps/1e3:.0f} km, velocity: {v_gps:.0f} m/s\n'
            f'Without corrections: ~10 km/day position error!',
            transform=ax3.transAxes, fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # ==========================================================================
    # Plot 4: Redshift from different stellar objects
    # ==========================================================================
    ax4 = axes[1, 1]

    # Different objects
    objects = [
        {'name': 'Sun (surface)', 'M': M_sun, 'R': 6.96e8, 'color': 'yellow'},
        {'name': 'White dwarf', 'M': 0.6 * M_sun, 'R': 7e6, 'color': 'white'},
        {'name': 'Neutron star', 'M': 1.4 * M_sun, 'R': 10e3, 'color': 'purple'},
        {'name': 'r = 3 r_s (near BH)', 'M': 10 * M_sun, 'R': None, 'color': 'black'},
    ]

    x_pos = np.arange(len(objects))
    z_values = []
    names = []

    for obj in objects:
        if obj['R'] is None:
            # For black hole, use 3 r_s
            rs = 2 * G * obj['M'] / c**2
            r = 3 * rs
        else:
            r = obj['R']
        z = gravitational_redshift(r, np.inf, obj['M'])
        z_values.append(z)
        names.append(obj['name'])

    colors = [obj['color'] for obj in objects]

    bars = ax4.bar(x_pos, z_values, color=colors, alpha=0.7, edgecolor='black')

    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(names, rotation=15, ha='right')
    ax4.set_ylabel('Redshift z')
    ax4.set_title('Gravitational Redshift from Stellar Objects')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, z in zip(bars, z_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, height * 1.5, f'z = {z:.2e}',
                ha='center', fontsize=9)

    plt.suptitle('Gravitational Redshift\n'
                 'z = sqrt((1 - r_s/r_obs) / (1 - r_s/r_emit)) - 1',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Print summary
    print("Gravitational Redshift Summary:")
    print("=" * 60)

    print(f"\nPound-Rebka Experiment (1959):")
    print(f"  Height: 22.5 m")
    print(f"  Predicted z: {z_tower:.2e}")
    print(f"  This was first direct measurement of gravitational redshift!")

    print(f"\nGPS Corrections:")
    print(f"  Gravitational (altitude): +{grav:.1f} us/day (faster)")
    print(f"  Special relativistic (velocity): {sr:.1f} us/day (slower)")
    print(f"  Net effect: +{total:.1f} us/day")
    print(f"  Position error if uncorrected: ~{total * 300 / 1e6 * 86400 / 1000:.0f} km/day")

    print(f"\nRedshift from stellar surfaces:")
    for obj, z in zip(objects, z_values):
        print(f"  {obj['name']}: z = {z:.4e}")

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'gravitational_redshift.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")


if __name__ == "__main__":
    main()
