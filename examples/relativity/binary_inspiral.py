"""
Experiment 197: Binary Inspiral and Chirp Signal

This experiment demonstrates the gravitational wave signal from
inspiraling compact binaries, including the characteristic chirp.

Physical concepts:
- Chirp mass and frequency evolution
- Peters-Mathews orbital decay
- Inspiral waveform
- Time to merger calculation
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


def chirp_mass(m1, m2):
    """
    Calculate chirp mass.

    M_c = (m1 * m2)^(3/5) / (m1 + m2)^(1/5)
    """
    return (m1 * m2)**(3/5) / (m1 + m2)**(1/5)


def total_mass(m1, m2):
    """Total mass M = m1 + m2"""
    return m1 + m2


def symmetric_mass_ratio(m1, m2):
    """Symmetric mass ratio eta = m1*m2 / (m1+m2)^2"""
    M = m1 + m2
    return m1 * m2 / M**2


def orbital_frequency(r, M, G=G):
    """Keplerian orbital frequency."""
    return np.sqrt(G * M / r**3)


def gw_frequency(f_orb):
    """GW frequency is twice orbital frequency for quadrupole radiation."""
    return 2 * f_orb


def frequency_derivative(f, Mc, G=G, c=c):
    """
    Time derivative of GW frequency (chirp).

    df/dt = (96/5) * pi^(8/3) * (G*Mc/c^3)^(5/3) * f^(11/3)
    """
    prefactor = (96/5) * np.pi**(8/3)
    return prefactor * (G * Mc / c**3)**(5/3) * f**(11/3)


def time_to_merger(f, Mc, G=G, c=c):
    """
    Time to merger from GW frequency f.

    tau = (5/256) * (c^3/(G*Mc))^(5/3) * (pi*f)^(-8/3)
    """
    prefactor = 5 / 256 / np.pi**(8/3)
    return prefactor * (c**3 / (G * Mc))**(5/3) * f**(-8/3)


def orbital_separation_from_frequency(f_gw, M, G=G, c=c):
    """Calculate orbital separation from GW frequency."""
    f_orb = f_gw / 2
    return (G * M / (4 * np.pi**2 * f_orb**2))**(1/3)


def gw_strain_amplitude(f, Mc, D, G=G, c=c):
    """
    GW strain amplitude (leading order).

    h = (4/D) * (G*Mc/c^2)^(5/3) * (pi*f/c)^(2/3)
    """
    return (4/D) * (G * Mc / c**2)**(5/3) * (np.pi * f / c)**(2/3)


def inspiral_frequency_evolution(t, f0, Mc, G=G, c=c):
    """
    Frequency as function of time during inspiral.

    f(t) = (1/pi) * (5/256)^(3/8) * (c^3/(G*Mc))^(5/8) * (tc-t)^(-3/8)

    Args:
        t: Time array (t=0 is some reference, tc is merger time)
        f0: Initial frequency at t=0
        Mc: Chirp mass

    Returns:
        Frequency array
    """
    # Time to merger from f0
    tc = time_to_merger(f0, Mc, G, c)

    # Time remaining to merger
    tau = tc - t

    # Avoid negative tau
    tau = np.maximum(tau, 1e-10)

    # Frequency evolution
    factor = (5/256)**(3/8) * (c**3 / (G * Mc))**(5/8) / np.pi
    f = factor * tau**(-3/8)

    return f


def inspiral_waveform(t, f0, Mc, D, phi0=0, G=G, c=c):
    """
    Generate inspiral waveform h(t).

    Returns h_plus polarization (cross is 90 degrees out of phase).
    """
    # Get frequency evolution
    f = inspiral_frequency_evolution(t, f0, Mc, G, c)

    # Strain amplitude
    h = gw_strain_amplitude(f, Mc, D, G, c)

    # Phase evolution
    tc = time_to_merger(f0, Mc, G, c)
    tau = np.maximum(tc - t, 1e-10)

    phi = phi0 - 2 * (5 * c**3 * tau / (256 * G * Mc))**(5/8)

    return h * np.cos(2 * np.pi * f * t + phi), f


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Binary neutron star system (like GW170817)
    m1 = 1.4 * M_sun
    m2 = 1.4 * M_sun
    Mc = chirp_mass(m1, m2)
    M = total_mass(m1, m2)
    D = 40e6 * 3.086e16  # 40 Mpc in meters

    # ==========================================================================
    # Plot 1: Frequency evolution (chirp)
    # ==========================================================================
    ax1 = axes[0, 0]

    f0 = 30  # Hz - starting frequency
    tc = time_to_merger(f0, Mc)

    # Time array (starting at t=0, merger at t=tc)
    t = np.linspace(0, tc * 0.999, 10000)

    f_evolution = inspiral_frequency_evolution(t, f0, Mc)

    ax1.semilogy(t, f_evolution, 'b-', lw=1)

    # Mark key frequencies
    f_isco = c**3 / (6**(3/2) * np.pi * G * M)  # ISCO frequency
    t_isco = time_to_merger(f_isco, Mc)

    ax1.axhline(y=f_isco, color='red', linestyle='--', alpha=0.7,
               label=f'ISCO frequency ({f_isco:.0f} Hz)')
    ax1.axhline(y=f0, color='green', linestyle=':', alpha=0.7,
               label=f'Initial frequency ({f0} Hz)')

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('GW Frequency (Hz)')
    ax1.set_title(f'Inspiral Frequency Evolution (Mc = {Mc/M_sun:.2f} M_sun)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Annotate chirp
    ax1.annotate('Chirp: frequency\nincreases rapidly\nnear merger',
                xy=(tc*0.95, f_evolution[-100]), xytext=(tc*0.6, 200),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ==========================================================================
    # Plot 2: Inspiral waveform
    # ==========================================================================
    ax2 = axes[0, 1]

    # Show last few seconds
    t_plot = np.linspace(tc - 5, tc * 0.999, 10000)
    h, f = inspiral_waveform(t_plot, f0, Mc, D)

    ax2.plot(t_plot - tc, h, 'b-', lw=0.5)

    ax2.set_xlabel('Time to merger (s)')
    ax2.set_ylabel('Strain h')
    ax2.set_title(f'Inspiral Waveform (D = 40 Mpc)')
    ax2.grid(True, alpha=0.3)

    # Show envelope
    h_envelope = gw_strain_amplitude(f, Mc, D)
    ax2.plot(t_plot - tc, h_envelope, 'r--', lw=1, alpha=0.7, label='Amplitude envelope')
    ax2.plot(t_plot - tc, -h_envelope, 'r--', lw=1, alpha=0.7)
    ax2.legend()

    # ==========================================================================
    # Plot 3: Different binary systems
    # ==========================================================================
    ax3 = axes[1, 0]

    # Different systems
    systems = [
        {'name': 'BNS (1.4+1.4)', 'm1': 1.4*M_sun, 'm2': 1.4*M_sun, 'color': 'blue'},
        {'name': 'NSBH (1.4+10)', 'm1': 1.4*M_sun, 'm2': 10*M_sun, 'color': 'green'},
        {'name': 'BBH (30+30)', 'm1': 30*M_sun, 'm2': 30*M_sun, 'color': 'red'},
        {'name': 'BBH (10+10)', 'm1': 10*M_sun, 'm2': 10*M_sun, 'color': 'orange'},
    ]

    f_range = np.logspace(0, 4, 500)  # 1 Hz to 10 kHz

    for sys in systems:
        Mc_sys = chirp_mass(sys['m1'], sys['m2'])
        h_sys = gw_strain_amplitude(f_range, Mc_sys, D)

        ax3.loglog(f_range, h_sys, '-', color=sys['color'], lw=2,
                  label=sys['name'])

    # LIGO sensitivity curve (approximate)
    f_ligo = np.logspace(1, 4, 100)
    h_ligo = 1e-23 * (f_ligo / 100)**(-1/2)  # Simplified
    ax3.loglog(f_ligo, h_ligo, 'k--', lw=1.5, alpha=0.7, label='LIGO noise (approx)')

    ax3.set_xlabel('GW Frequency (Hz)')
    ax3.set_ylabel('Strain h')
    ax3.set_title('Characteristic Strain for Different Binary Systems')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, which='both')
    ax3.set_xlim(1, 1e4)
    ax3.set_ylim(1e-25, 1e-20)

    # ==========================================================================
    # Plot 4: Time to merger vs frequency
    # ==========================================================================
    ax4 = axes[1, 1]

    for sys in systems:
        Mc_sys = chirp_mass(sys['m1'], sys['m2'])
        tau = time_to_merger(f_range, Mc_sys)

        # Convert to useful units
        tau_days = tau / (24 * 3600)

        ax4.loglog(f_range, tau_days, '-', color=sys['color'], lw=2,
                  label=sys['name'])

    # Mark key times
    ax4.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax4.text(2, 1.5, '1 day', fontsize=9, color='gray')

    ax4.axhline(y=1/24, color='gray', linestyle=':', alpha=0.5)
    ax4.text(2, 1/24*1.5, '1 hour', fontsize=9, color='gray')

    ax4.axhline(y=1/24/60, color='gray', linestyle=':', alpha=0.5)
    ax4.text(2, 1/24/60*1.5, '1 minute', fontsize=9, color='gray')

    ax4.set_xlabel('GW Frequency (Hz)')
    ax4.set_ylabel('Time to merger (days)')
    ax4.set_title('Time to Merger vs GW Frequency')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, which='both')
    ax4.set_xlim(1, 1e4)
    ax4.set_ylim(1e-8, 1e4)

    plt.suptitle('Binary Inspiral and Chirp Signal\n'
                 'df/dt proportional to f^(11/3) - the "chirp"',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Print summary
    print("Binary Inspiral Summary:")
    print("=" * 60)

    for sys in systems:
        Mc_sys = chirp_mass(sys['m1'], sys['m2'])
        M_sys = total_mass(sys['m1'], sys['m2'])
        f_isco_sys = c**3 / (6**(3/2) * np.pi * G * M_sys)

        print(f"\n{sys['name']}:")
        print(f"  Chirp mass: {Mc_sys/M_sun:.2f} M_sun")
        print(f"  ISCO frequency: {f_isco_sys:.0f} Hz")
        print(f"  Time from 10 Hz to merger: {time_to_merger(10, Mc_sys)/60:.1f} minutes")
        print(f"  Time from 100 Hz to merger: {time_to_merger(100, Mc_sys):.1f} seconds")
        print(f"  Strain at 100 Hz (40 Mpc): {gw_strain_amplitude(100, Mc_sys, D):.2e}")

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'binary_inspiral.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")


if __name__ == "__main__":
    main()
