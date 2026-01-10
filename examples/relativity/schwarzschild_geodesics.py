"""
Experiment 191: Schwarzschild Geodesics and Orbital Precession

This experiment demonstrates geodesic motion in Schwarzschild spacetime,
including orbital precession (perihelion advance) as predicted by GR.

Physical concepts:
- Schwarzschild metric and its properties
- Geodesic equations in curved spacetime
- Effective potential for orbital motion
- Perihelion precession (Mercury's orbit)
- Comparison with Newtonian orbits
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp


# Physical constants
G = 6.67430e-11  # m^3/(kg*s^2)
c = 299792458.0  # m/s
M_sun = 1.989e30  # kg


def schwarzschild_radius(M, G=G, c=c):
    """Calculate Schwarzschild radius r_s = 2GM/c^2"""
    return 2 * G * M / c**2


def effective_potential_gr(r, L, M, mu=1, G=G, c=c):
    """
    GR effective potential for Schwarzschild geodesics.

    V_eff = -GMmu/r + L^2/(2*mu*r^2) - GML^2/(mu*c^2*r^3)

    Args:
        r: Radial coordinate
        L: Angular momentum
        M: Central mass
        mu: Orbiting mass (or 1 for test particle)

    Returns:
        Effective potential
    """
    rs = schwarzschild_radius(M, G, c)
    term1 = -G * M * mu / r
    term2 = L**2 / (2 * mu * r**2)
    term3 = -G * M * L**2 / (mu * c**2 * r**3)
    return term1 + term2 + term3


def effective_potential_newton(r, L, M, mu=1, G=G):
    """
    Newtonian effective potential.

    V_eff = -GMmu/r + L^2/(2*mu*r^2)
    """
    return -G * M * mu / r + L**2 / (2 * mu * r**2)


def orbit_equations_schwarzschild(y, phi, rs, L, E, c=c):
    """
    Geodesic equations in Schwarzschild spacetime using phi as parameter.

    Using u = 1/r, the equation becomes:
    d^2u/dphi^2 + u = GM/L^2 + 3GM*u^2/c^2

    For orbits, we track (u, du/dphi).
    """
    u, du_dphi = y

    # d^2u/dphi^2 = -u + rs/(2*L^2/c^2) + 3*rs*u^2/2
    # where L is specific angular momentum (L/m)
    d2u_dphi2 = -u + rs * c**2 / (2 * L**2) + 1.5 * rs * u**2

    return [du_dphi, d2u_dphi2]


def integrate_orbit(M, L, r0, dr_dphi0, n_orbits=5, n_points=2000, G=G, c=c):
    """
    Integrate orbit in Schwarzschild spacetime.

    Args:
        M: Central mass
        L: Specific angular momentum (L/m)
        r0: Initial radius
        dr_dphi0: Initial dr/dphi
        n_orbits: Number of orbits to compute
        n_points: Number of points

    Returns:
        (phi, r) arrays
    """
    rs = schwarzschild_radius(M, G, c)

    u0 = 1 / r0
    du_dphi0 = -dr_dphi0 / r0**2

    phi_span = (0, 2 * np.pi * n_orbits)
    phi_eval = np.linspace(0, 2 * np.pi * n_orbits, n_points)

    sol = solve_ivp(
        lambda phi, y: orbit_equations_schwarzschild(y, phi, rs, L, 0, c),
        phi_span, [u0, du_dphi0], t_eval=phi_eval,
        method='RK45', rtol=1e-10, atol=1e-12
    )

    r = 1 / sol.y[0]
    return sol.t, r


def precession_per_orbit(a, e, M, G=G, c=c):
    """
    Calculate GR precession per orbit.

    Delta_phi = 6*pi*G*M / (c^2 * a * (1-e^2))

    Args:
        a: Semi-major axis
        e: Eccentricity
        M: Central mass

    Returns:
        Precession angle per orbit (radians)
    """
    return 6 * np.pi * G * M / (c**2 * a * (1 - e**2))


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ==========================================================================
    # Plot 1: Effective potential comparison
    # ==========================================================================
    ax1 = axes[0, 0]

    # Use normalized units for clarity
    M = M_sun
    rs = schwarzschild_radius(M)

    # Angular momentum for circular orbit at r = 6*rs
    r_circ = 6 * rs
    L_circ = np.sqrt(G * M * r_circ)  # Specific angular momentum for circular orbit

    r_range = np.linspace(3 * rs, 30 * rs, 500)

    V_gr = effective_potential_gr(r_range, L_circ, M, mu=1)
    V_newton = effective_potential_newton(r_range, L_circ, M, mu=1)

    # Normalize by some energy scale
    E_scale = G * M / r_circ
    ax1.plot(r_range/rs, V_gr/E_scale, 'b-', lw=2, label='GR (Schwarzschild)')
    ax1.plot(r_range/rs, V_newton/E_scale, 'r--', lw=2, label='Newtonian')

    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.axvline(x=3, color='purple', linestyle=':', lw=1.5,
               label='Photon sphere (3 r_s)')
    ax1.axvline(x=6, color='green', linestyle=':', lw=1.5,
               label='ISCO (6 r_s)')

    ax1.set_xlabel('Radius r / r_s')
    ax1.set_ylabel('Effective Potential (normalized)')
    ax1.set_title('Effective Potential: GR vs Newtonian')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(2, 30)

    # ==========================================================================
    # Plot 2: Precessing orbit
    # ==========================================================================
    ax2 = axes[0, 1]

    # Use exaggerated parameters for visibility
    # For Mercury: a = 5.79e10 m, e = 0.206
    # Scale to show visible precession

    M = 1e8 * M_sun  # Supermassive black hole for visible effect
    a = 100 * schwarzschild_radius(M)  # Semi-major axis
    e = 0.8  # High eccentricity

    # Calculate orbital parameters
    r_peri = a * (1 - e)
    r_apo = a * (1 + e)

    # Specific angular momentum for this orbit
    # L^2 = GM * a * (1 - e^2)
    L = np.sqrt(G * M * a * (1 - e**2))

    # Initial conditions: start at perihelion
    r0 = r_peri
    dr_dphi0 = 0  # At perihelion/aphelion

    phi, r = integrate_orbit(M, L, r0, dr_dphi0, n_orbits=8, n_points=5000)

    # Convert to Cartesian
    x = r * np.cos(phi)
    y = r * np.sin(phi)

    rs = schwarzschild_radius(M)
    ax2.plot(x/rs, y/rs, 'b-', lw=0.5, alpha=0.7)

    # Mark perihelion positions
    peri_indices = []
    for i in range(1, len(r) - 1):
        if r[i] < r[i-1] and r[i] < r[i+1]:
            peri_indices.append(i)

    if len(peri_indices) > 0:
        peri_x = r[peri_indices] * np.cos(phi[peri_indices])
        peri_y = r[peri_indices] * np.sin(phi[peri_indices])
        ax2.plot(peri_x/rs, peri_y/rs, 'ro', markersize=6, label='Perihelion')

    # Draw central mass
    circle = plt.Circle((0, 0), 1, color='black', fill=True)
    ax2.add_patch(circle)

    ax2.set_xlabel('x / r_s')
    ax2.set_ylabel('y / r_s')
    ax2.set_title(f'Precessing Orbit (e = {e})')
    ax2.set_aspect('equal')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Calculate precession
    delta_phi = precession_per_orbit(a, e, M)
    ax2.text(0.05, 0.95, f'Precession per orbit:\n{np.degrees(delta_phi):.2f} deg',
            transform=ax2.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ==========================================================================
    # Plot 3: Mercury's precession
    # ==========================================================================
    ax3 = axes[1, 0]

    # Mercury orbital parameters
    a_mercury = 5.79e10  # m
    e_mercury = 0.2056
    T_mercury = 87.97 * 24 * 3600  # orbital period in seconds

    # GR precession per orbit
    delta_phi_orbit = precession_per_orbit(a_mercury, e_mercury, M_sun)

    # Convert to arcseconds per century
    orbits_per_century = 100 * 365.25 * 24 * 3600 / T_mercury
    precession_century = delta_phi_orbit * orbits_per_century * 180 * 3600 / np.pi

    # Compare different effects on Mercury's precession
    effects = {
        'GR prediction': 42.98,
        'Other planets': 531.63,
        'Solar oblateness': 0.03,
        'Total observed': 574.64,
    }

    positions = np.arange(len(effects))
    values = list(effects.values())
    labels = list(effects.keys())
    colors = ['red', 'blue', 'green', 'purple']

    bars = ax3.bar(positions, values, color=colors, alpha=0.7, edgecolor='black')

    ax3.set_xticks(positions)
    ax3.set_xticklabels(labels, rotation=15)
    ax3.set_ylabel('Precession (arcseconds/century)')
    ax3.set_title("Mercury's Perihelion Precession")
    ax3.grid(True, alpha=0.3, axis='y')

    # Add computed GR value
    ax3.axhline(y=precession_century, color='red', linestyle='--', lw=2, alpha=0.7)
    ax3.text(0.5, precession_century + 10, f'Computed: {precession_century:.2f}"/century',
            color='red', fontsize=10)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, height + 5, f'{val:.2f}"',
                ha='center', fontsize=9)

    # ==========================================================================
    # Plot 4: Orbit comparison (GR vs Newtonian)
    # ==========================================================================
    ax4 = axes[1, 1]

    # Simple comparison orbit
    M = 1e6 * M_sun
    a = 50 * schwarzschild_radius(M)
    e = 0.5

    r_peri = a * (1 - e)
    L = np.sqrt(G * M * a * (1 - e**2))

    phi, r_gr = integrate_orbit(M, L, r_peri, 0, n_orbits=3, n_points=3000)

    # Newtonian orbit (exact ellipse)
    p = a * (1 - e**2)  # Semi-latus rectum
    r_newton = p / (1 + e * np.cos(phi))

    rs = schwarzschild_radius(M)

    # Plot orbits
    x_gr = r_gr * np.cos(phi)
    y_gr = r_gr * np.sin(phi)
    x_newton = r_newton * np.cos(phi)
    y_newton = r_newton * np.sin(phi)

    ax4.plot(x_newton/rs, y_newton/rs, 'b-', lw=1, alpha=0.5, label='Newtonian')
    ax4.plot(x_gr/rs, y_gr/rs, 'r-', lw=1, alpha=0.7, label='GR (Schwarzschild)')

    # Central mass
    circle = plt.Circle((0, 0), 1, color='black', fill=True)
    ax4.add_patch(circle)

    ax4.set_xlabel('x / r_s')
    ax4.set_ylabel('y / r_s')
    ax4.set_title('Orbit Comparison: GR vs Newtonian')
    ax4.set_aspect('equal')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    delta_phi = precession_per_orbit(a, e, M)
    ax4.text(0.05, 0.95, f'e = {e}, a = {a/rs:.0f} r_s\n'
            f'Precession: {np.degrees(delta_phi):.2f} deg/orbit',
            transform=ax4.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Schwarzschild Geodesics and Orbital Precession\n'
                 'GR precession: Delta_phi = 6*pi*GM/(c^2*a*(1-e^2))',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Print summary
    print("Schwarzschild Orbital Precession Summary:")
    print("=" * 60)
    print(f"\nMercury's perihelion precession due to GR:")
    print(f"  Formula: 6*pi*G*M_sun / (c^2 * a * (1-e^2))")
    print(f"  Per orbit: {delta_phi_orbit * 180 * 3600 / np.pi:.4f} arcseconds")
    print(f"  Per century: {precession_century:.2f} arcseconds")
    print(f"  Observed: 43.11 +/- 0.45 arcseconds/century")
    print(f"\n  This was one of the first tests of General Relativity!")

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'schwarzschild_geodesics.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")


if __name__ == "__main__":
    main()
