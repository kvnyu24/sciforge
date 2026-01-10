"""
Experiment 196: Frame Dragging (Lense-Thirring Effect)

This experiment demonstrates frame dragging - how a rotating massive
body drags spacetime around with it.

Physical concepts:
- Kerr metric and frame dragging
- Lense-Thirring precession
- ZAMO (zero angular momentum observer)
- Gravity Probe B experiment
- Ergosphere and Penrose process
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.relativity import KerrMetric, FrameDragging


# Physical constants
G = 6.67430e-11  # m^3/(kg*s^2)
c = 299792458.0  # m/s
M_sun = 1.989e30  # kg
M_earth = 5.972e24  # kg
R_earth = 6.371e6  # m


def schwarzschild_radius(M, G=G, c=c):
    """Calculate Schwarzschild radius"""
    return 2 * G * M / c**2


def frame_dragging_omega(r, theta, a, M, G=G, c=c):
    """
    Calculate frame dragging angular velocity.

    omega = 2*a*M*r / Sigma^2 * (c / r_g)

    where Sigma^2 = (r^2 + a^2)^2 - Delta * a^2 * sin^2(theta)
    and Delta = r^2 - r_s*r + a^2

    Args:
        r: Radial coordinate
        theta: Polar angle
        a: Spin parameter (dimensionless, 0 to 1)
        M: Black hole mass

    Returns:
        Frame dragging angular velocity omega (rad/s)
    """
    r_g = G * M / c**2
    rs = 2 * r_g
    a_dim = a * r_g  # Dimensionful spin parameter

    # Kerr metric functions
    Sigma = r**2 + a_dim**2 * np.cos(theta)**2
    Delta = r**2 - rs * r + a_dim**2

    # Frame dragging angular velocity
    # omega = g_t_phi / g_phi_phi
    A = (r**2 + a_dim**2)**2 - Delta * a_dim**2 * np.sin(theta)**2

    omega = rs * r * a_dim * c / A

    return omega


def lense_thirring_precession(r, J, M, G=G, c=c):
    """
    Calculate Lense-Thirring precession rate for a gyroscope.

    Omega_LT = 2*G*J / (c^2 * r^3) for equatorial orbit

    Args:
        r: Orbital radius
        J: Angular momentum of central body
        M: Mass of central body

    Returns:
        Precession rate (rad/s)
    """
    return 2 * G * J / (c**2 * r**3)


def geodetic_precession(r, M, G=G, c=c):
    """
    Calculate geodetic (de Sitter) precession rate.

    Omega_geo = 3*G*M / (2*c^2*r) * sqrt(G*M/r^3)

    This is precession due to curved spacetime, not frame dragging.
    """
    return 1.5 * G * M / (c**2 * r) * np.sqrt(G * M / r**3)


def ergosphere_radius(theta, a, M, G=G, c=c):
    """
    Calculate ergosphere outer boundary radius.

    r_ergo = r_g + sqrt(r_g^2 - a^2 * cos^2(theta))

    where r_g = GM/c^2
    """
    r_g = G * M / c**2
    a_dim = a * r_g
    return r_g + np.sqrt(r_g**2 - a_dim**2 * np.cos(theta)**2)


def outer_horizon_kerr(a, M, G=G, c=c):
    """Calculate outer horizon radius for Kerr black hole."""
    r_g = G * M / c**2
    return r_g + np.sqrt(r_g**2 - (a * r_g)**2)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ==========================================================================
    # Plot 1: Frame dragging around a Kerr black hole
    # ==========================================================================
    ax1 = axes[0, 0]

    M = 10 * M_sun
    r_g = G * M / c**2

    a = 0.9  # High spin
    theta = np.pi / 2  # Equatorial plane

    r_range = np.linspace(1.5 * r_g, 30 * r_g, 100)

    # Frame dragging angular velocity
    omega = frame_dragging_omega(r_range, theta, a, M)

    # Convert to Hz
    f_drag = omega / (2 * np.pi)

    ax1.semilogy(r_range / r_g, f_drag, 'b-', lw=2, label=f'a = {a}')

    # Compare with other spin values
    for spin, color in [(0.5, 'green'), (0.99, 'red')]:
        omega_spin = frame_dragging_omega(r_range, theta, spin, M)
        ax1.semilogy(r_range / r_g, omega_spin / (2*np.pi), '--',
                    color=color, lw=1.5, label=f'a = {spin}')

    # Mark ergosphere
    r_ergo = ergosphere_radius(theta, a, M)
    ax1.axvline(x=r_ergo / r_g, color='purple', linestyle=':', lw=1.5)
    ax1.text(r_ergo/r_g + 0.5, f_drag[0], 'Ergosphere', fontsize=9, color='purple')

    # Mark horizon
    r_horizon = outer_horizon_kerr(a, M)
    ax1.axvline(x=r_horizon / r_g, color='black', linestyle='-', lw=2, alpha=0.5)
    ax1.text(r_horizon/r_g + 0.2, f_drag[-1], 'Horizon', fontsize=9)

    ax1.set_xlabel('r / r_g')
    ax1.set_ylabel('Frame dragging frequency (Hz)')
    ax1.set_title(f'Frame Dragging Around Kerr Black Hole (M = 10 M_sun)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, 30)

    # ==========================================================================
    # Plot 2: Ergosphere cross-section
    # ==========================================================================
    ax2 = axes[0, 1]

    a = 0.9  # High spin
    M = 10 * M_sun
    r_g = G * M / c**2

    theta_range = np.linspace(0, 2*np.pi, 200)

    # Ergosphere boundary
    r_ergo = np.array([ergosphere_radius(t, a, M) for t in theta_range])

    # Outer horizon
    r_horizon = outer_horizon_kerr(a, M)

    # Plot in Cartesian-like coordinates (r, theta) -> (x, z)
    x_ergo = r_ergo * np.sin(theta_range)
    z_ergo = r_ergo * np.cos(theta_range)

    x_horizon = r_horizon * np.sin(theta_range)
    z_horizon = r_horizon * np.cos(theta_range)

    ax2.plot(x_ergo / r_g, z_ergo / r_g, 'purple', lw=2, label='Ergosphere')
    ax2.fill(x_ergo / r_g, z_ergo / r_g, alpha=0.2, color='purple')

    ax2.plot(x_horizon / r_g, z_horizon / r_g, 'black', lw=2, label='Event horizon')
    ax2.fill(x_horizon / r_g, z_horizon / r_g, alpha=0.8, color='black')

    # Rotation axis
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax2.annotate('', xy=(0, 2.5), xytext=(0, 1.8),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax2.text(0.2, 2.2, 'Spin', fontsize=10, color='blue')

    ax2.set_xlabel('x / r_g')
    ax2.set_ylabel('z / r_g')
    ax2.set_title(f'Ergosphere Cross-Section (a = {a})')
    ax2.legend(loc='upper right')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)

    # Add note about Penrose process
    ax2.text(-2.5, -2.5,
            'In ergosphere:\n'
            'Nothing can stay\n'
            'stationary - forced\n'
            'to co-rotate with BH\n\n'
            'Penrose process can\n'
            'extract rotational\n'
            'energy from BH',
            fontsize=8, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # ==========================================================================
    # Plot 3: Gravity Probe B experiment
    # ==========================================================================
    ax3 = axes[1, 0]

    # Earth's parameters
    J_earth = 7.07e33  # Earth's angular momentum (kg m^2/s)
    I_earth = 8.04e37  # Moment of inertia
    omega_earth = 7.29e-5  # Angular velocity (rad/s)

    # Gravity Probe B orbit
    altitude_gpb = 640e3  # 640 km
    r_gpb = R_earth + altitude_gpb

    # Calculate precession rates
    omega_lt = lense_thirring_precession(r_gpb, J_earth, M_earth)
    omega_geo = geodetic_precession(r_gpb, M_earth)

    # Convert to milliarcseconds per year
    mas_per_rad = 180 * 3600 * 1000 / np.pi
    seconds_per_year = 365.25 * 24 * 3600

    lt_mas_yr = omega_lt * mas_per_rad * seconds_per_year
    geo_mas_yr = omega_geo * mas_per_rad * seconds_per_year

    effects = ['Geodetic\n(de Sitter)', 'Frame Dragging\n(Lense-Thirring)']
    values = [geo_mas_yr, lt_mas_yr]
    colors = ['blue', 'red']

    bars = ax3.bar(effects, values, color=colors, alpha=0.7, edgecolor='black')

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, height + 100,
                f'{val:.0f} mas/yr', ha='center', fontsize=11)

    ax3.set_ylabel('Precession rate (milliarcseconds/year)')
    ax3.set_title('Gravity Probe B Precession Rates')
    ax3.grid(True, alpha=0.3, axis='y')

    # Add experimental results
    ax3.text(0.5, 0.95,
            'GP-B Results (2011):\n'
            f'Geodetic: 6601.8 +/- 18.3 mas/yr\n'
            f'Frame-dragging: 37.2 +/- 7.2 mas/yr\n\n'
            f'GR predictions:\n'
            f'Geodetic: ~6606 mas/yr\n'
            f'Frame-dragging: ~39 mas/yr',
            transform=ax3.transAxes, fontsize=9, va='top', ha='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # ==========================================================================
    # Plot 4: Frame dragging vs radius for different objects
    # ==========================================================================
    ax4 = axes[1, 1]

    # Different spinning objects
    objects = [
        {'name': 'Earth (r = R_earth)', 'M': M_earth, 'J': J_earth,
         'r_surface': R_earth, 'color': 'blue'},
        {'name': 'Sun (r = R_sun)', 'M': M_sun, 'J': 1.9e41,
         'r_surface': 6.96e8, 'color': 'orange'},
        {'name': 'Neutron star (a=0.1)', 'M': 1.4*M_sun, 'J': None,
         'r_surface': 10e3, 'a': 0.1, 'color': 'purple'},
    ]

    r_normalized = np.linspace(1, 100, 200)

    for obj in objects:
        if obj.get('J') is not None:
            # Non-relativistic case
            r = r_normalized * obj['r_surface']
            omega = lense_thirring_precession(r, obj['J'], obj['M'])
        else:
            # Kerr case
            a = obj['a']
            r = r_normalized * obj['r_surface']
            omega = frame_dragging_omega(r, np.pi/2, a, obj['M'])

        ax4.loglog(r_normalized, omega, '-', color=obj['color'], lw=2,
                  label=obj['name'])

    ax4.set_xlabel('r / r_surface')
    ax4.set_ylabel('Frame dragging rate (rad/s)')
    ax4.set_title('Frame Dragging for Different Objects')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, which='both')

    plt.suptitle('Frame Dragging (Lense-Thirring Effect)\n'
                 'Rotating masses drag spacetime around with them',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Print summary
    print("Frame Dragging Summary:")
    print("=" * 60)

    print(f"\nGravity Probe B (Earth orbit, 640 km altitude):")
    print(f"  Geodetic precession: {geo_mas_yr:.1f} mas/year")
    print(f"  Frame-dragging precession: {lt_mas_yr:.1f} mas/year")
    print(f"  Ratio: Geodetic is ~{geo_mas_yr/lt_mas_yr:.0f}x larger")

    print(f"\nKerr black hole (a = 0.9, M = 10 M_sun):")
    r_test = 3 * G * M / c**2  # 3 r_g
    omega_fd = frame_dragging_omega(r_test, np.pi/2, 0.9, M)
    print(f"  Frame dragging at r = 3 r_g: {omega_fd:.1e} rad/s")
    print(f"  Corresponding frequency: {omega_fd/(2*np.pi):.1f} Hz")

    print(f"\nErgosphere properties (a = 0.9):")
    r_ergo_eq = ergosphere_radius(np.pi/2, 0.9, M)
    r_ergo_pole = ergosphere_radius(0, 0.9, M)
    print(f"  Equatorial radius: {r_ergo_eq/r_g:.2f} r_g")
    print(f"  Polar radius: {r_ergo_pole/r_g:.2f} r_g")
    print(f"  (Horizon radius: {outer_horizon_kerr(0.9, M)/r_g:.2f} r_g)")

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'frame_dragging.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")


if __name__ == "__main__":
    main()
