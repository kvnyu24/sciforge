"""
Experiment 193: Shapiro Time Delay

This experiment demonstrates the Shapiro time delay - the gravitational
time delay experienced by light passing near a massive object.

Physical concepts:
- Shapiro delay formula
- Radar ranging experiments
- Tests of General Relativity
- Applications to pulsar timing
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt


# Physical constants
G = 6.67430e-11  # m^3/(kg*s^2)
c = 299792458.0  # m/s
M_sun = 1.989e30  # kg
AU = 1.496e11  # Astronomical unit in meters


def schwarzschild_radius(M, G=G, c=c):
    """Calculate Schwarzschild radius"""
    return 2 * G * M / c**2


def shapiro_delay(r_1, r_2, b, M, G=G, c=c):
    """
    Calculate Shapiro time delay for light traveling from r_1 to r_2
    with closest approach distance b to mass M.

    Delta_t = (2GM/c^3) * ln((r_1 + sqrt(r_1^2 - b^2)) * (r_2 + sqrt(r_2^2 - b^2)) / b^2)

    For r_1, r_2 >> b:
    Delta_t ≈ (4GM/c^3) * (1 + ln(4*r_1*r_2/b^2))

    Args:
        r_1: Distance from mass to first point
        r_2: Distance from mass to second point
        b: Impact parameter (closest approach)
        M: Mass causing delay

    Returns:
        Time delay in seconds
    """
    rs = schwarzschild_radius(M, G, c)

    # Full formula
    term1 = r_1 + np.sqrt(np.maximum(r_1**2 - b**2, 0))
    term2 = r_2 + np.sqrt(np.maximum(r_2**2 - b**2, 0))

    delay = (rs / c) * np.log(term1 * term2 / b**2)

    return delay


def shapiro_delay_simplified(r_1, r_2, b, M, G=G, c=c):
    """
    Simplified Shapiro delay for r_1, r_2 >> b.

    Delta_t ≈ (2GM/c^3) * (1 + ln(4*r_1*r_2/b^2))
    """
    rs = schwarzschild_radius(M, G, c)
    return (rs / c) * (1 + np.log(4 * r_1 * r_2 / b**2))


def round_trip_delay(r_earth, r_planet, b, M, G=G, c=c):
    """
    Calculate round-trip Shapiro delay for radar ranging.

    Total delay is approximately double the one-way delay
    (exact calculation would account for different paths).
    """
    return 2 * shapiro_delay(r_earth, r_planet, b, M, G, c)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ==========================================================================
    # Plot 1: Shapiro delay vs impact parameter
    # ==========================================================================
    ax1 = axes[0, 0]

    # Consider Earth-Mars radar ranging
    r_earth = 1.0 * AU
    r_mars = 1.524 * AU
    R_sun = 6.96e8  # Solar radius

    # Range of impact parameters (from grazing Sun to large distances)
    b_range = np.linspace(1.1 * R_sun, 50 * R_sun, 200)

    delay = shapiro_delay(r_earth, r_mars, b_range, M_sun)
    delay_simplified = shapiro_delay_simplified(r_earth, r_mars, b_range, M_sun)

    ax1.plot(b_range / R_sun, delay * 1e6, 'b-', lw=2, label='Full formula')
    ax1.plot(b_range / R_sun, delay_simplified * 1e6, 'r--', lw=2,
            label='Simplified (r >> b)')

    ax1.set_xlabel('Impact parameter b / R_sun')
    ax1.set_ylabel('Shapiro delay (microseconds)')
    ax1.set_title('Shapiro Delay: Earth to Mars')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Mark grazing incidence
    b_graze = 1.01 * R_sun
    delay_graze = shapiro_delay(r_earth, r_mars, b_graze, M_sun)
    ax1.axvline(x=1, color='orange', linestyle=':', alpha=0.7)
    ax1.annotate(f'Grazing Sun:\ndelay = {delay_graze*1e6:.1f} us',
                xy=(1.1, delay_graze*1e6), xytext=(5, delay_graze*1e6*0.8),
                arrowprops=dict(arrowstyle='->', color='orange'),
                fontsize=10, color='orange')

    # ==========================================================================
    # Plot 2: Superior conjunction geometry
    # ==========================================================================
    ax2 = axes[0, 1]

    # Draw Sun-Earth-Planet geometry
    theta = np.linspace(0, 2*np.pi, 100)

    # Earth orbit
    ax2.plot(np.cos(theta), np.sin(theta), 'b--', alpha=0.5, label='Earth orbit')

    # Mars orbit
    ax2.plot(1.524 * np.cos(theta), 1.524 * np.sin(theta), 'r--', alpha=0.5,
            label='Mars orbit')

    # Sun at center
    ax2.plot(0, 0, 'yo', markersize=30, label='Sun')

    # Earth position
    ax2.plot(1, 0, 'bo', markersize=10, label='Earth')

    # Mars at superior conjunction (opposite side of Sun)
    ax2.plot(-1.524, 0, 'ro', markersize=8, label='Mars (superior conjunction)')

    # Light path (bent around Sun)
    # Draw straight line for simplicity
    ax2.plot([1, -1.524], [0, 0], 'g-', lw=2, alpha=0.7, label='Radar signal')

    # Impact parameter
    ax2.annotate('', xy=(0, 0.1), xytext=(0, 0),
                arrowprops=dict(arrowstyle='<->', color='purple'))
    ax2.text(0.1, 0.15, 'b (impact\nparameter)', fontsize=9, color='purple')

    ax2.set_xlabel('Distance (AU)')
    ax2.set_ylabel('Distance (AU)')
    ax2.set_title('Superior Conjunction Geometry')
    ax2.set_xlim(-2, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.legend(loc='upper left', fontsize=8)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    # Add annotation about Shapiro delay
    ax2.text(0.05, 0.95,
            'Shapiro (1964) proposed using\n'
            'radar ranging to test GR.\n\n'
            'Signal is delayed by curved\n'
            'spacetime near the Sun.',
            transform=ax2.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # ==========================================================================
    # Plot 3: Historical measurements
    # ==========================================================================
    ax3 = axes[1, 0]

    # Simulated measurement data (based on actual experiments)
    experiments = [
        {'name': 'Mariner 6/7 (1970)', 'precision': 0.05, 'result': 1.00},
        {'name': 'Mariner 9 (1971)', 'precision': 0.02, 'result': 1.00},
        {'name': 'Viking (1979)', 'precision': 0.002, 'result': 1.000},
        {'name': 'Cassini (2003)', 'precision': 2.3e-5, 'result': 1.000021},
    ]

    # GR prediction
    gamma_GR = 1.0

    x_pos = np.arange(len(experiments))
    widths = 0.6

    # Plot results with error bars
    results = [e['result'] for e in experiments]
    errors = [e['precision'] for e in experiments]
    names = [e['name'] for e in experiments]

    bars = ax3.bar(x_pos, results, width=widths, color='steelblue', alpha=0.7,
                  edgecolor='black')
    ax3.errorbar(x_pos, results, yerr=errors, fmt='none', color='black',
                capsize=5, capthick=2)

    ax3.axhline(y=1.0, color='red', linestyle='--', lw=2, label='GR prediction (gamma = 1)')

    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(names, rotation=15, ha='right')
    ax3.set_ylabel('Measured gamma / GR prediction')
    ax3.set_title('Shapiro Delay Tests of GR')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(0.95, 1.05)

    # Add precision annotation
    for i, exp in enumerate(experiments):
        ax3.text(i, 0.96, f'precision:\n{exp["precision"]*100:.3f}%',
                ha='center', fontsize=8)

    # ==========================================================================
    # Plot 4: Pulsar timing application
    # ==========================================================================
    ax4 = axes[1, 1]

    # Binary pulsar - Shapiro delay varies with orbital phase
    # Simulate a binary pulsar like PSR J1614-2230

    # Orbital parameters (simplified)
    P_orb = 8.7 * 24 * 3600  # Orbital period in seconds
    a = 1.2e9  # Semi-major axis in meters (about 8 light-seconds)
    M_companion = 0.5 * M_sun  # White dwarf companion
    inclination = np.radians(89.17)  # Nearly edge-on

    # Orbital phase
    phase = np.linspace(0, 2*np.pi, 1000)

    # Impact parameter varies with orbital phase
    # At conjunction, b is minimum
    b_min = a * np.cos(inclination)  # Closest approach

    # For edge-on orbit, b varies as b = a * |sin(phase - pi/2)| approximately
    b = np.abs(a * np.sin(phase - np.pi/2) * np.cos(inclination) +
              a * np.sqrt(1 - np.sin(phase - np.pi/2)**2) * np.sin(inclination))
    b = np.maximum(b, 1e6)  # Avoid singularity

    # Calculate Shapiro delay (simplified - pulsar much further than orbit)
    # For pulsar timing, the relevant delay is from the companion
    delay_pulsar = (2 * G * M_companion / c**3) * np.log(1 + 1/(b/a))

    # Convert to microseconds
    delay_us = delay_pulsar * 1e6

    ax4.plot(phase * 180 / np.pi, delay_us, 'b-', lw=2)

    ax4.set_xlabel('Orbital phase (degrees)')
    ax4.set_ylabel('Shapiro delay (microseconds)')
    ax4.set_title('Shapiro Delay in Binary Pulsar\n(PSR J1614-2230 type)')
    ax4.grid(True, alpha=0.3)

    # Mark superior conjunction
    ax4.axvline(x=90, color='red', linestyle='--', alpha=0.7)
    ax4.axvline(x=270, color='red', linestyle='--', alpha=0.7)
    ax4.text(90, np.max(delay_us) * 0.9, 'Superior\nconjunction',
            ha='center', fontsize=9, color='red')

    # Add physics note
    ax4.text(0.95, 0.95,
            'Shapiro delay in binary pulsars\n'
            'allows precise mass measurements.\n\n'
            'PSR J1614-2230: Neutron star\n'
            'mass = 1.97 M_sun measured\n'
            'to < 0.5% precision!',
            transform=ax4.transAxes, fontsize=9, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Shapiro Time Delay\n'
                 'Delta t = (2GM/c^3) * ln[(r_1 + sqrt(r_1^2-b^2))(r_2 + sqrt(r_2^2-b^2))/b^2]',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Print summary
    print("Shapiro Delay Summary:")
    print("=" * 60)

    # Solar grazing
    b_graze = R_sun * 1.01
    delay = shapiro_delay(1.0*AU, 1.524*AU, b_graze, M_sun)
    print(f"\nEarth-Mars radar at solar grazing:")
    print(f"  One-way delay: {delay*1e6:.2f} microseconds")
    print(f"  Round-trip delay: {2*delay*1e6:.2f} microseconds")

    # Cassini result
    print(f"\nCassini (2003) measurement:")
    print(f"  gamma = 1 + (2.1 +/- 2.3) x 10^-5")
    print(f"  This is the most precise test of GR")

    # Viking result (from 1970s)
    print(f"\nViking (1979) measurement:")
    print(f"  gamma = 1.000 +/- 0.002")
    print(f"  Confirmed GR to 0.2% precision")

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'shapiro_delay.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")


if __name__ == "__main__":
    main()
