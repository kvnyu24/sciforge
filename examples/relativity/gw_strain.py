"""
Experiment 198: Gravitational Wave Strain h(t)

This experiment demonstrates gravitational wave strain calculations,
including waveform generation and detector response.

Physical concepts:
- GW strain tensor in TT gauge
- Plus and cross polarizations
- Detector response function
- Antenna patterns
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Physical constants
G = 6.67430e-11  # m^3/(kg*s^2)
c = 299792458.0  # m/s
M_sun = 1.989e30  # kg


def chirp_mass(m1, m2):
    """Calculate chirp mass."""
    return (m1 * m2)**(3/5) / (m1 + m2)**(1/5)


def gw_strain_components(t, f, h0, phi0=0, iota=0):
    """
    Calculate GW strain components h_+ and h_x.

    For circular orbit:
    h_+ = h0 * (1 + cos^2(iota)) / 2 * cos(phi(t))
    h_x = h0 * cos(iota) * sin(phi(t))

    Args:
        t: Time array
        f: Frequency (can be time-dependent)
        h0: Strain amplitude
        phi0: Initial phase
        iota: Inclination angle (0 = face-on)

    Returns:
        (h_plus, h_cross) strain components
    """
    # Phase evolution for monochromatic wave
    if isinstance(f, (int, float)):
        phi = 2 * np.pi * f * t + phi0
    else:
        # Time-dependent frequency (chirp)
        phi = phi0 + np.cumsum(2 * np.pi * f * np.gradient(t))

    # Strain components
    h_plus = h0 * (1 + np.cos(iota)**2) / 2 * np.cos(phi)
    h_cross = h0 * np.cos(iota) * np.sin(phi)

    return h_plus, h_cross


def detector_response(h_plus, h_cross, psi, theta, phi):
    """
    Calculate detector response.

    h = F_+ * h_+ + F_x * h_x

    where F_+ and F_x are the antenna pattern functions.

    Args:
        h_plus, h_cross: GW strain components
        psi: Polarization angle
        theta: Zenith angle of source
        phi: Azimuth angle of source

    Returns:
        Detected strain
    """
    # Antenna pattern functions for L-shaped detector
    F_plus = 0.5 * (1 + np.cos(theta)**2) * np.cos(2*phi) * np.cos(2*psi) \
             - np.cos(theta) * np.sin(2*phi) * np.sin(2*psi)

    F_cross = 0.5 * (1 + np.cos(theta)**2) * np.cos(2*phi) * np.sin(2*psi) \
              + np.cos(theta) * np.sin(2*phi) * np.cos(2*psi)

    return F_plus * h_plus + F_cross * h_cross


def antenna_pattern_plus(theta, phi, psi=0):
    """Antenna pattern F_+ for a single arm pair."""
    return 0.5 * (1 + np.cos(theta)**2) * np.cos(2*phi) * np.cos(2*psi) \
           - np.cos(theta) * np.sin(2*phi) * np.sin(2*psi)


def antenna_pattern_cross(theta, phi, psi=0):
    """Antenna pattern F_x for a single arm pair."""
    return 0.5 * (1 + np.cos(theta)**2) * np.cos(2*phi) * np.sin(2*psi) \
           + np.cos(theta) * np.sin(2*phi) * np.cos(2*psi)


def inspiral_strain(t, m1, m2, D, iota=0, phi0=0, f0=30, G=G, c=c):
    """
    Generate full inspiral waveform with both polarizations.

    Args:
        t: Time array (relative to merger at t=0)
        m1, m2: Component masses
        D: Distance to source
        iota: Inclination angle
        phi0: Initial phase
        f0: Reference frequency

    Returns:
        (h_plus, h_cross, frequency) arrays
    """
    Mc = chirp_mass(m1, m2)
    M = m1 + m2

    # Time to merger from f0
    tc = 5/256 / np.pi**(8/3) * (c**3 / (G * Mc))**(5/3) * f0**(-8/3)

    # Time remaining to merger
    tau = tc - t
    tau = np.maximum(tau, 1e-10)

    # Frequency evolution
    f = (5/256)**(3/8) * (c**3 / (G * Mc))**(5/8) / np.pi * tau**(-3/8)

    # Amplitude
    h0 = (4/D) * (G * Mc / c**2)**(5/3) * (np.pi * f / c)**(2/3)

    # Phase
    phi = phi0 - 2 * (5 * c**3 * tau / (256 * G * Mc))**(5/8)

    # Strain components
    h_plus = h0 * (1 + np.cos(iota)**2) / 2 * np.cos(phi)
    h_cross = h0 * np.cos(iota) * np.sin(phi)

    return h_plus, h_cross, f


def strain_tensor_tt(h_plus, h_cross, direction='z'):
    """
    Construct TT-gauge strain tensor for wave propagating in given direction.

    For z-direction:
    h_ij = | h_+   h_x   0 |
           | h_x  -h_+   0 |
           |  0    0    0 |
    """
    h = np.zeros((3, 3))

    if direction == 'z':
        h[0, 0] = h_plus
        h[1, 1] = -h_plus
        h[0, 1] = h[1, 0] = h_cross
    elif direction == 'x':
        h[1, 1] = h_plus
        h[2, 2] = -h_plus
        h[1, 2] = h[2, 1] = h_cross
    else:  # y
        h[0, 0] = h_plus
        h[2, 2] = -h_plus
        h[0, 2] = h[2, 0] = h_cross

    return h


def main():
    fig = plt.figure(figsize=(16, 12))

    # Binary parameters
    m1 = 30 * M_sun
    m2 = 30 * M_sun
    D = 400e6 * 3.086e16  # 400 Mpc

    # ==========================================================================
    # Plot 1: h_+ and h_x polarizations
    # ==========================================================================
    ax1 = fig.add_subplot(2, 2, 1)

    # Monochromatic wave for clarity
    f = 100  # Hz
    h0 = 1e-21
    t = np.linspace(0, 0.1, 5000)  # 100 ms

    h_plus, h_cross = gw_strain_components(t, f, h0, iota=np.pi/4)

    ax1.plot(t * 1000, h_plus * 1e21, 'b-', lw=1.5, label='h_+ (plus)')
    ax1.plot(t * 1000, h_cross * 1e21, 'r--', lw=1.5, label='h_x (cross)')

    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Strain (x10^-21)')
    ax1.set_title('GW Polarizations (f = 100 Hz, iota = 45 deg)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ==========================================================================
    # Plot 2: Effect of test mass ring
    # ==========================================================================
    ax2 = fig.add_subplot(2, 2, 2)

    # Show deformation of a ring of test masses
    n_masses = 32
    theta_ring = np.linspace(0, 2*np.pi, n_masses, endpoint=False)
    r0 = 1.0  # Initial radius

    # Different phases
    phases = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(phases)))

    for phase, color in zip(phases, colors):
        # Strain tensor at this phase
        h_p = 0.3 * np.cos(phase)  # Exaggerated for visibility
        h_c = 0.3 * np.sin(phase)

        # Deformed positions
        x = r0 * np.cos(theta_ring)
        y = r0 * np.sin(theta_ring)

        # Apply strain: dx_i = h_ij * x_j
        dx = h_p * x + h_c * y
        dy = h_c * x - h_p * y

        x_def = x + dx
        y_def = y + dy

        ax2.plot(np.append(x_def, x_def[0]), np.append(y_def, y_def[0]),
                'o-', color=color, lw=1.5, markersize=3,
                label=f'phase = {np.degrees(phase):.0f} deg')

    # Original ring
    ax2.plot(r0 * np.cos(theta_ring), r0 * np.sin(theta_ring),
            'k--', lw=1, alpha=0.5, label='Original')

    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Ring of Test Masses Deformed by GW\n(strain exaggerated)')
    ax2.set_aspect('equal')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ==========================================================================
    # Plot 3: Antenna pattern
    # ==========================================================================
    ax3 = fig.add_subplot(2, 2, 3, projection='polar')

    # Antenna pattern in equatorial plane (theta = pi/2)
    phi_range = np.linspace(0, 2*np.pi, 200)
    theta = np.pi / 2

    F_plus = np.abs(antenna_pattern_plus(theta, phi_range, psi=0))
    F_cross = np.abs(antenna_pattern_cross(theta, phi_range, psi=0))
    F_total = np.sqrt(F_plus**2 + F_cross**2)

    ax3.plot(phi_range, F_plus, 'b-', lw=2, label='|F_+|')
    ax3.plot(phi_range, F_cross, 'r--', lw=2, label='|F_x|')
    ax3.plot(phi_range, F_total, 'k-', lw=1.5, alpha=0.7, label='Total')

    ax3.set_title('Detector Antenna Pattern\n(equatorial plane)', pad=20)
    ax3.legend(loc='upper right')

    # Mark arm directions (for L-shaped detector)
    ax3.annotate('Arm 1', xy=(0, 0.6), fontsize=10, color='green')
    ax3.annotate('Arm 2', xy=(np.pi/2, 0.6), fontsize=10, color='green')

    # ==========================================================================
    # Plot 4: Full inspiral waveform
    # ==========================================================================
    ax4 = fig.add_subplot(2, 2, 4)

    Mc = chirp_mass(m1, m2)

    # Calculate merger time from 30 Hz
    f0 = 30
    tc = 5/256 / np.pi**(8/3) * (c**3 / (G * Mc))**(5/3) * f0**(-8/3)

    # Time array
    t = np.linspace(0, tc * 0.999, 50000)

    # Get waveform
    h_plus, h_cross, freq = inspiral_strain(t, m1, m2, D, iota=0, f0=f0)

    # Plot strain (time relative to merger)
    ax4.plot((t - tc), h_plus, 'b-', lw=0.3, alpha=0.7)

    ax4.set_xlabel('Time to merger (s)')
    ax4.set_ylabel('Strain h_+')
    ax4.set_title(f'Inspiral Waveform: BBH {m1/M_sun:.0f}+{m2/M_sun:.0f} M_sun at {D/3.086e22:.0f} Mpc')
    ax4.grid(True, alpha=0.3)

    # Add inset for last moments
    axins = ax4.inset_axes([0.55, 0.55, 0.4, 0.4])
    mask = (t - tc) > -0.1
    axins.plot((t - tc)[mask], h_plus[mask], 'b-', lw=0.5)
    axins.set_xlabel('Time (s)', fontsize=8)
    axins.set_ylabel('h_+', fontsize=8)
    axins.set_title('Last 0.1 s', fontsize=9)
    axins.grid(True, alpha=0.3)

    plt.suptitle('Gravitational Wave Strain h(t)\n'
                 'TT-gauge: h_ij stretches space perpendicular to propagation',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Print summary
    print("GW Strain Summary:")
    print("=" * 60)

    print(f"\nBinary: {m1/M_sun:.0f} + {m2/M_sun:.0f} M_sun at {D/3.086e22:.0f} Mpc")
    print(f"Chirp mass: {Mc/M_sun:.1f} M_sun")

    # Peak strain
    h_peak = np.max(np.abs(h_plus))
    print(f"Peak strain: {h_peak:.2e}")

    # Signal duration
    print(f"Signal duration (30 Hz to merger): {tc:.2f} s")

    print(f"\nPolarizations:")
    print(f"  h_+ (plus): quadrupolar pattern aligned with detector arms")
    print(f"  h_x (cross): quadrupolar pattern rotated 45 degrees")

    print(f"\nStrain effects:")
    arm_length = 4000  # LIGO arm length in meters
    print(f"  LIGO arm length: {arm_length} m")
    print(f"  At peak strain {h_peak:.2e}:")
    print(f"    Arm length change: {h_peak * arm_length * 1e18:.3f} attometers")
    print(f"    (That's {h_peak * arm_length / 1e-15:.1e} times smaller than a proton!)")

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'gw_strain.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")


if __name__ == "__main__":
    main()
