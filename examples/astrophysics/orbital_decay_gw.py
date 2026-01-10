"""
Experiment 271: Orbital Decay from Gravitational Waves

Demonstrates how gravitational wave emission causes binary systems
to lose energy and spiral inward.

Physical concepts:
- GW power: P = (32/5) * (G^4/c^5) * (m1*m2)^2 * (m1+m2) / a^5
- Orbital decay: da/dt proportional to GW luminosity
- Chirp mass determines inspiral rate
- Hulse-Taylor pulsar: first indirect GW detection
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt

# Physical constants
G = 6.674e-11
c = 2.998e8
M_sun = 1.989e30
pc = 3.086e16


def gw_power(m1, m2, a):
    """
    Gravitational wave luminosity from circular binary.

    P = (32/5) * (G^4/c^5) * (m1*m2)^2 * (m1+m2) / a^5
    """
    return (32/5) * G**4 / c**5 * (m1 * m2)**2 * (m1 + m2) / a**5


def chirp_mass(m1, m2):
    """
    Chirp mass: determines GW amplitude and frequency evolution.

    M_c = (m1 * m2)^(3/5) / (m1 + m2)^(1/5)
    """
    return (m1 * m2)**(3/5) / (m1 + m2)**(1/5)


def orbital_frequency(m1, m2, a):
    """Orbital frequency from Kepler's law."""
    return np.sqrt(G * (m1 + m2) / a**3) / (2 * np.pi)


def gw_frequency(f_orb):
    """GW frequency is twice orbital frequency for circular binary."""
    return 2 * f_orb


def merger_time(m1, m2, a0):
    """
    Time to merger from initial separation a0.

    t_merger = (5/256) * c^5 * a0^4 / (G^3 * m1 * m2 * (m1+m2))
    """
    return (5/256) * c**5 * a0**4 / (G**3 * m1 * m2 * (m1 + m2))


def separation_evolution(m1, m2, a0, t):
    """
    Separation as function of time.

    a(t) = a0 * (1 - t/t_merger)^(1/4)
    """
    t_merge = merger_time(m1, m2, a0)
    valid = t < t_merge
    a = np.where(valid, a0 * (1 - t / t_merge)**(1/4), 0)
    return a


def gw_strain(m1, m2, a, d):
    """
    GW strain amplitude (leading order).

    h ~ (G/c^4) * (M_c * omega^2)^(5/3) / d
    """
    M_c = chirp_mass(m1, m2)
    omega = np.sqrt(G * (m1 + m2) / a**3)
    return (G / c**4) * (M_c * omega**2)**(5/3) / d * (G * M_c / c**2)**(5/3)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Binary parameters
    m1 = 30 * M_sun  # First black hole
    m2 = 30 * M_sun  # Second black hole
    M_total = m1 + m2
    M_c = chirp_mass(m1, m2)

    # Plot 1: GW power vs separation
    ax1 = axes[0, 0]

    # Separation range (in Schwarzschild radii)
    Rs = 2 * G * M_total / c**2
    a = np.logspace(1, 4, 200) * Rs

    P_gw = gw_power(m1, m2, a)
    L_sun_watts = 3.828e26

    ax1.loglog(a / Rs, P_gw, 'b-', lw=2)

    # Mark key separations
    a_ISCO = 6 * Rs
    P_ISCO = gw_power(m1, m2, a_ISCO)
    ax1.plot(6, P_ISCO, 'ro', markersize=10, label='ISCO')

    # Solar luminosity reference
    ax1.axhline(y=L_sun_watts, color='yellow', linestyle='--', alpha=0.7, label='$L_\\odot$')

    ax1.set_xlabel('Separation ($R_s$)')
    ax1.set_ylabel('GW Power (W)')
    ax1.set_title(f'GW Luminosity (M$_1$ = M$_2$ = {m1/M_sun:.0f} M$_\\odot$)')
    ax1.legend()
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_xlim(5, 1e4)

    # Annotate peak power
    ax1.annotate(f'$P_{{GW}}$ ~ {P_ISCO:.1e} W\nat ISCO',
                 xy=(6, P_ISCO), xytext=(20, P_ISCO * 10),
                 fontsize=10, arrowprops=dict(arrowstyle='->', color='red'))

    # Plot 2: Orbital evolution
    ax2 = axes[0, 1]

    # Initial separation
    a0 = 1000 * Rs
    t_merge = merger_time(m1, m2, a0)

    # Time array (log scale near merger)
    t = np.concatenate([
        np.linspace(0, 0.9 * t_merge, 100),
        np.linspace(0.9 * t_merge, 0.9999 * t_merge, 100)
    ])

    a_t = separation_evolution(m1, m2, a0, t)
    f_gw_t = gw_frequency(orbital_frequency(m1, m2, a_t + 1e-10))

    ax2.semilogy(t / t_merge, a_t / Rs, 'b-', lw=2, label='Separation')
    ax2_twin = ax2.twinx()
    ax2_twin.semilogy(t / t_merge, f_gw_t, 'r--', lw=2, label='GW frequency')

    ax2.set_xlabel('Time ($t / t_{merger}$)')
    ax2.set_ylabel('Separation ($R_s$)', color='blue')
    ax2_twin.set_ylabel('GW Frequency (Hz)', color='red')

    ax2.tick_params(axis='y', labelcolor='blue')
    ax2_twin.tick_params(axis='y', labelcolor='red')

    ax2.set_title(f'Inspiral Evolution ($t_{{merge}}$ = {t_merge/3.156e7:.1e} yr)')

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)

    # Plot 3: Hulse-Taylor pulsar comparison
    ax3 = axes[1, 0]

    # Hulse-Taylor pulsar parameters
    m1_HT = 1.4398 * M_sun  # Pulsar mass
    m2_HT = 1.3886 * M_sun  # Companion mass
    P_orb_0 = 7.75 * 3600  # Initial orbital period (seconds)
    a0_HT = (G * (m1_HT + m2_HT) * P_orb_0**2 / (4 * np.pi**2))**(1/3)

    # Observed period decay
    dP_dt_obs = -2.4e-12  # s/s (observed)

    # GR prediction
    P_gw_HT = gw_power(m1_HT, m2_HT, a0_HT)
    E_orb = -G * m1_HT * m2_HT / (2 * a0_HT)
    dE_dt = -P_gw_HT

    # da/dt from energy loss
    da_dt = 2 * a0_HT * dE_dt / E_orb

    # Period decay from da/dt
    dP_dt_GR = 1.5 * P_orb_0 * da_dt / a0_HT

    # Time series
    years = np.linspace(1975, 2020, 100)
    t_sec = (years - 1975) * 3.156e7

    # Cumulative period shift
    P_shift_obs = 0.5 * dP_dt_obs * t_sec  # Cumulative shift
    P_shift_GR = 0.5 * dP_dt_GR * t_sec

    ax3.plot(years, -P_shift_obs, 'bo', markersize=4, alpha=0.5, label='Observations')
    ax3.plot(years, -P_shift_GR, 'r-', lw=2, label='GR prediction')

    ax3.set_xlabel('Year')
    ax3.set_ylabel('Cumulative period shift (s)')
    ax3.set_title('Hulse-Taylor Pulsar (PSR B1913+16)\n'
                  'First evidence for gravitational waves')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Add merger time
    t_merge_HT = merger_time(m1_HT, m2_HT, a0_HT)
    ax3.text(0.95, 0.05, f'Merger in {t_merge_HT/(3.156e7*1e6):.0f} Myr',
             transform=ax3.transAxes, fontsize=10,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 4: Merger times for different systems
    ax4 = axes[1, 1]

    # Different binary types
    systems = {
        'NS-NS (1.4+1.4 M$_\\odot$)': (1.4, 1.4),
        'BH-NS (10+1.4 M$_\\odot$)': (10, 1.4),
        'BH-BH (10+10 M$_\\odot$)': (10, 10),
        'BH-BH (30+30 M$_\\odot$)': (30, 30),
        'SMBH (10$^6$+10$^6$ M$_\\odot$)': (1e6, 1e6),
    }

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(systems)))

    # Initial separation range
    a_range = np.logspace(9, 12, 100)  # meters

    for (name, (m1_s, m2_s)), color in zip(systems.items(), colors):
        m1_kg = m1_s * M_sun
        m2_kg = m2_s * M_sun

        t_merge = merger_time(m1_kg, m2_kg, a_range)

        ax4.loglog(a_range / 1e9, t_merge / (3.156e7 * 1e9), color=color, lw=2,
                   label=name)

    # Mark key separations
    ax4.axhline(y=13.8, color='gray', linestyle=':', alpha=0.7)  # Age of universe
    ax4.text(10, 15, 'Age of Universe', fontsize=9)

    ax4.axhline(y=0.001, color='gray', linestyle=':', alpha=0.5)  # 1 Myr
    ax4.text(10, 0.0015, '1 Myr', fontsize=9)

    ax4.set_xlabel('Initial Separation (10$^9$ m)')
    ax4.set_ylabel('Merger Time (Gyr)')
    ax4.set_title('Merger Timescales for Different Binaries')
    ax4.legend(loc='lower right', fontsize=9)
    ax4.grid(True, alpha=0.3, which='both')
    ax4.set_xlim(1, 1000)
    ax4.set_ylim(1e-6, 1e6)

    plt.suptitle('Experiment 271: Orbital Decay from Gravitational Waves\n'
                 '$P_{GW} = \\frac{32}{5}\\frac{G^4}{c^5}\\frac{(m_1 m_2)^2(m_1+m_2)}{a^5}$',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'orbital_decay_gw.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'orbital_decay_gw.png')}")


if __name__ == "__main__":
    main()
