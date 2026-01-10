"""
Experiment 192: Light Bending and Null Geodesics

This experiment demonstrates the bending of light in curved spacetime,
tracing null geodesics around massive objects.

Physical concepts:
- Null geodesics in Schwarzschild spacetime
- Gravitational light deflection
- Einstein ring and gravitational lensing
- Photon sphere and critical impact parameter
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


# Physical constants
G = 6.67430e-11  # m^3/(kg*s^2)
c = 299792458.0  # m/s
M_sun = 1.989e30  # kg


def schwarzschild_radius(M, G=G, c=c):
    """Calculate Schwarzschild radius r_s = 2GM/c^2"""
    return 2 * G * M / c**2


def light_deflection_angle(b, M, G=G, c=c):
    """
    Calculate light deflection angle for impact parameter b.

    For weak field (b >> r_s):
    alpha = 4GM / (c^2 * b)

    Args:
        b: Impact parameter
        M: Mass of deflector

    Returns:
        Deflection angle in radians
    """
    return 4 * G * M / (c**2 * b)


def photon_geodesic_equation(phi, y, rs, b):
    """
    Null geodesic equation in Schwarzschild spacetime.

    Using u = 1/r:
    d^2u/dphi^2 + u = 3*rs*u^2/2

    Args:
        phi: Angle parameter
        y: [u, du/dphi]
        rs: Schwarzschild radius
        b: Impact parameter

    Returns:
        [du/dphi, d^2u/dphi^2]
    """
    u, du_dphi = y
    d2u_dphi2 = -u + 1.5 * rs * u**2
    return [du_dphi, d2u_dphi2]


def trace_light_ray(b, M, phi_start=-np.pi, phi_end=np.pi, n_points=1000, G=G, c=c):
    """
    Trace a light ray with given impact parameter.

    Args:
        b: Impact parameter
        M: Central mass
        phi_start, phi_end: Angular range
        n_points: Number of points

    Returns:
        (phi, r) arrays
    """
    rs = schwarzschild_radius(M, G, c)

    # Initial conditions: light coming from infinity
    # At phi = phi_start, r -> infinity, u -> 0
    # du/dphi = -1/b for incoming light from phi = -pi
    u0 = 1e-10  # Small value (far from mass)
    du_dphi0 = -1 / b  # Coming in

    phi_span = (phi_start, phi_end)
    phi_eval = np.linspace(phi_start, phi_end, n_points)

    sol = solve_ivp(
        lambda phi, y: photon_geodesic_equation(phi, y, rs, b),
        phi_span, [u0, du_dphi0], t_eval=phi_eval,
        method='RK45', rtol=1e-10, atol=1e-12
    )

    # Filter out points where r < rs (inside horizon)
    r = 1 / np.maximum(sol.y[0], 1e-15)
    valid = r > rs

    return sol.t[valid], r[valid]


def critical_impact_parameter(M, G=G, c=c):
    """
    Calculate critical impact parameter for photon capture.

    b_crit = 3*sqrt(3)*GM/c^2 = (3*sqrt(3)/2) * r_s
    """
    rs = schwarzschild_radius(M, G, c)
    return 1.5 * np.sqrt(3) * rs


def einstein_radius(M, D_l, D_s, D_ls, G=G, c=c):
    """
    Calculate Einstein radius for gravitational lensing.

    theta_E = sqrt(4GM * D_ls / (c^2 * D_l * D_s))
    """
    return np.sqrt(4 * G * M * D_ls / (c**2 * D_l * D_s))


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    M = M_sun  # Solar mass
    rs = schwarzschild_radius(M)

    # ==========================================================================
    # Plot 1: Light ray trajectories for different impact parameters
    # ==========================================================================
    ax1 = axes[0, 0]

    b_crit = critical_impact_parameter(M)

    # Different impact parameters
    b_values = np.array([1.5, 2.0, 2.6, 3.0, 4.0, 6.0, 10.0]) * rs
    colors = plt.cm.viridis(np.linspace(0.9, 0.1, len(b_values)))

    for b, color in zip(b_values, colors):
        try:
            phi, r = trace_light_ray(b, M, phi_start=-2.5, phi_end=2.5, n_points=2000)

            x = r * np.cos(phi)
            y = r * np.sin(phi)

            ax1.plot(x/rs, y/rs, '-', color=color, lw=1.5,
                    label=f'b = {b/rs:.1f} r_s')
        except:
            continue

    # Draw black hole
    circle = plt.Circle((0, 0), 1, color='black', fill=True)
    ax1.add_patch(circle)

    # Draw photon sphere
    photon_sphere = plt.Circle((0, 0), 1.5, color='red', fill=False,
                               linestyle='--', lw=1.5)
    ax1.add_patch(photon_sphere)

    # Mark critical impact parameter
    ax1.axhline(y=b_crit/rs, color='red', linestyle=':', alpha=0.5)
    ax1.text(-8, b_crit/rs + 0.5, f'b_crit = {b_crit/rs:.2f} r_s', fontsize=9, color='red')

    ax1.set_xlabel('x / r_s')
    ax1.set_ylabel('y / r_s')
    ax1.set_title('Light Ray Deflection in Schwarzschild Spacetime')
    ax1.set_xlim(-10, 10)
    ax1.set_ylim(-5, 10)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # ==========================================================================
    # Plot 2: Deflection angle vs impact parameter
    # ==========================================================================
    ax2 = axes[0, 1]

    b_range = np.linspace(2 * rs, 100 * rs, 200)

    # Weak field approximation
    alpha_weak = light_deflection_angle(b_range, M)

    # Exact formula (numerical integration would be needed for full accuracy)
    # For now, use higher-order approximation
    alpha_exact = light_deflection_angle(b_range, M) * (
        1 + 15 * np.pi * (rs / b_range)**2 / 16
    )

    ax2.plot(b_range/rs, np.degrees(alpha_weak), 'b-', lw=2,
            label='Weak field: 4GM/(c^2 b)')
    ax2.plot(b_range/rs, np.degrees(alpha_exact), 'r--', lw=2,
            label='Higher order correction')

    # Mark critical impact parameter
    ax2.axvline(x=b_crit/rs, color='green', linestyle=':', lw=1.5)
    ax2.text(b_crit/rs + 1, 2, f'b_crit\n= {b_crit/rs:.2f} r_s', fontsize=9, color='green')

    # Mark solar deflection (b = R_sun)
    R_sun = 6.96e8  # m
    alpha_sun = light_deflection_angle(R_sun, M)
    ax2.axvline(x=R_sun/rs, color='orange', linestyle=':', lw=1.5)
    ax2.annotate(f'Sun\'s limb\nalpha = {np.degrees(alpha_sun)*3600:.2f}"',
                xy=(R_sun/rs, np.degrees(alpha_sun)), xytext=(R_sun/rs*1.2, 0.1),
                fontsize=9, color='orange',
                arrowprops=dict(arrowstyle='->', color='orange'))

    ax2.set_xlabel('Impact parameter b / r_s')
    ax2.set_ylabel('Deflection angle (degrees)')
    ax2.set_title('Light Deflection Angle')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 100)
    ax2.set_yscale('log')

    # ==========================================================================
    # Plot 3: Gravitational lensing (Einstein ring)
    # ==========================================================================
    ax3 = axes[1, 0]

    # Create a simple lensing diagram
    # Source behind lens, multiple images form

    # Lens at origin
    ax3.plot(0, 0, 'ko', markersize=15, label='Lens (mass M)')

    # Source position (slightly off-axis)
    beta = 0.3  # Source position in units of Einstein radius
    ax3.plot(0, beta, 'r*', markersize=15, label='Source')

    # Einstein radius (normalized to 1)
    theta_E = 1.0

    # Image positions for point mass lens
    # theta^2 - beta*theta - theta_E^2 = 0
    # theta = (beta +/- sqrt(beta^2 + 4*theta_E^2)) / 2
    theta_plus = 0.5 * (beta + np.sqrt(beta**2 + 4 * theta_E**2))
    theta_minus = 0.5 * (beta - np.sqrt(beta**2 + 4 * theta_E**2))

    # Draw images
    ax3.plot(0, theta_plus, 'b^', markersize=12, label=f'Image 1 (theta = {theta_plus:.2f})')
    ax3.plot(0, theta_minus, 'gv', markersize=12, label=f'Image 2 (theta = {theta_minus:.2f})')

    # Draw Einstein ring
    circle = plt.Circle((0, 0), theta_E, color='gold', fill=False,
                        linestyle='--', lw=2, label='Einstein ring')
    ax3.add_patch(circle)

    # Light paths (schematic)
    ax3.annotate('', xy=(0, theta_plus), xytext=(0, beta),
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5,
                              connectionstyle='arc3,rad=0.3'))
    ax3.annotate('', xy=(0, theta_minus), xytext=(0, beta),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5,
                              connectionstyle='arc3,rad=-0.3'))

    ax3.set_xlabel('Angular position')
    ax3.set_ylabel('Angular position')
    ax3.set_title('Gravitational Lensing Geometry\n(Source at beta = 0.3 theta_E)')
    ax3.set_xlim(-2, 2)
    ax3.set_ylim(-2, 2)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')

    # ==========================================================================
    # Plot 4: Black hole shadow (photon capture)
    # ==========================================================================
    ax4 = axes[1, 1]

    # Simulate what an observer would see looking at a black hole
    # with a bright background

    # Create grid of impact parameters
    n = 300
    x = np.linspace(-5, 5, n)
    y = np.linspace(-5, 5, n)
    X, Y = np.meshgrid(x, y)
    B = np.sqrt(X**2 + Y**2) * rs  # Impact parameter

    # Photons with b < b_crit are captured
    captured = B < b_crit

    # Create image: black where captured, bright gradient elsewhere
    image = np.ones((n, n))
    image[captured] = 0

    # Add some structure to background (e.g., accretion disk glow)
    # Simple radial gradient
    background = 1 / (1 + (B / (5 * rs))**2)
    image = image * background

    im = ax4.imshow(image, extent=[-5, 5, -5, 5], cmap='hot',
                   origin='lower', vmin=0, vmax=1)

    # Draw shadow boundary
    theta = np.linspace(0, 2*np.pi, 100)
    ax4.plot(b_crit/rs * np.cos(theta), b_crit/rs * np.sin(theta),
            'c-', lw=2, label=f'Shadow boundary (b = {b_crit/rs:.2f} r_s)')

    # Draw event horizon (would not be visible, but shown for reference)
    ax4.plot(np.cos(theta), np.sin(theta), 'r--', lw=1, label='Event horizon')

    ax4.set_xlabel('x / r_s')
    ax4.set_ylabel('y / r_s')
    ax4.set_title('Black Hole Shadow')
    ax4.legend(loc='upper right', fontsize=9)

    plt.suptitle('Light Bending and Null Geodesics in General Relativity\n'
                 'Deflection: alpha = 4GM/(c^2 b) for weak field',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Print summary
    print("Light Bending Summary:")
    print("=" * 60)
    print(f"\nPhoton sphere radius: 1.5 r_s")
    print(f"Critical impact parameter: {b_crit/rs:.4f} r_s = {1.5*np.sqrt(3):.4f} r_s")
    print(f"\nSolar deflection at limb:")
    print(f"  Deflection angle: {np.degrees(alpha_sun)*3600:.4f} arcseconds")
    print(f"  (First measured by Eddington, 1919)")
    print(f"\nEinstein's prediction: 1.75 arcseconds")

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'light_bending.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")


if __name__ == "__main__":
    main()
