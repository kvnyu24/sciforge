"""
Experiment 195: Innermost Stable Circular Orbit (ISCO)

This experiment demonstrates the concept of ISCO - the smallest stable
circular orbit around a black hole or compact object.

Physical concepts:
- ISCO in Schwarzschild and Kerr spacetimes
- Effective potential analysis
- Binding energy at ISCO
- Accretion disk inner edge
- Black hole spin measurement
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq


# Physical constants
G = 6.67430e-11  # m^3/(kg*s^2)
c = 299792458.0  # m/s
M_sun = 1.989e30  # kg


def schwarzschild_radius(M, G=G, c=c):
    """Calculate Schwarzschild radius"""
    return 2 * G * M / c**2


def effective_potential(r, L, M, G=G, c=c):
    """
    Effective potential for test particle in Schwarzschild spacetime.

    V_eff = (1 - r_s/r)(1 + L^2/(r^2 c^2))

    where L is specific angular momentum.
    """
    rs = schwarzschild_radius(M, G, c)
    return (1 - rs/r) * (1 + L**2 / (r**2 * c**2))


def isco_radius_schwarzschild(M, G=G, c=c):
    """
    ISCO radius for Schwarzschild black hole.

    r_ISCO = 6 GM/c^2 = 3 r_s
    """
    return 6 * G * M / c**2


def isco_radius_kerr(a, M, prograde=True, G=G, c=c):
    """
    ISCO radius for Kerr black hole.

    For prograde orbits (co-rotating with BH spin):
    r_ISCO decreases with spin, minimum at 1 GM/c^2 for a = M

    For retrograde orbits:
    r_ISCO increases with spin, maximum at 9 GM/c^2 for a = M

    Args:
        a: Dimensionless spin parameter (0 <= a <= 1)
        M: Black hole mass
        prograde: True for prograde (co-rotating) orbits

    Returns:
        ISCO radius
    """
    # Using r_g = GM/c^2
    r_g = G * M / c**2

    # Calculate Z1 and Z2
    Z1 = 1 + (1 - a**2)**(1/3) * ((1 + a)**(1/3) + (1 - a)**(1/3))
    Z2 = np.sqrt(3 * a**2 + Z1**2)

    if prograde:
        r_isco = r_g * (3 + Z2 - np.sqrt((3 - Z1) * (3 + Z1 + 2*Z2)))
    else:
        r_isco = r_g * (3 + Z2 + np.sqrt((3 - Z1) * (3 + Z1 + 2*Z2)))

    return r_isco


def angular_momentum_isco(r_isco, M, G=G, c=c):
    """
    Specific angular momentum at ISCO.

    For Schwarzschild:
    L_ISCO = sqrt(12) * GM/c
    """
    rs = schwarzschild_radius(M, G, c)
    r_g = G * M / c**2

    # L/m at ISCO
    L = r_isco * c / np.sqrt(r_isco / r_g - 3)
    return L


def binding_energy_isco(r_isco, M, G=G, c=c):
    """
    Binding energy per unit rest mass at ISCO.

    E_bind/mc^2 = 1 - sqrt(1 - r_s/r_ISCO) * sqrt(1 - 3*r_s/(2*r_ISCO))

    For Schwarzschild (r_ISCO = 6 r_g):
    E_bind/mc^2 = 1 - sqrt(8/9) ≈ 0.0572 (5.72% of rest mass)
    """
    rs = schwarzschild_radius(M, G, c)

    # Energy per unit rest mass at ISCO
    E_over_mc2 = np.sqrt(1 - rs/r_isco) * np.sqrt(1 - 1.5*rs/r_isco) / \
                 np.sqrt(1 - 1.5*rs/r_isco)

    # For proper calculation using circular orbit energy
    r_g = G * M / c**2
    E = (1 - 2*r_g/r_isco) / np.sqrt(1 - 3*r_g/r_isco)

    return 1 - E


def orbital_frequency_isco(r_isco, M, G=G, c=c):
    """
    Orbital frequency at ISCO.

    Omega = sqrt(GM/r^3) for Newtonian
    Omega = c^3/(r + r_g*a) * sqrt(r_g/r^3) for Kerr
    """
    return np.sqrt(G * M / r_isco**3)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    M = 10 * M_sun  # 10 solar mass black hole
    rs = schwarzschild_radius(M)
    r_g = rs / 2

    # ==========================================================================
    # Plot 1: Effective potential showing ISCO
    # ==========================================================================
    ax1 = axes[0, 0]

    r_range = np.linspace(1.1 * rs, 30 * r_g, 500)

    # Calculate L values for different circular orbits
    L_values = [3.0, 3.46, 4.0, 5.0]  # In units of GM/c

    for L_ratio in L_values:
        L = L_ratio * G * M / c
        V = effective_potential(r_range, L, M)
        label = f'L = {L_ratio:.2f} GM/c'
        if L_ratio == 3.46:
            label += ' (ISCO)'
        ax1.plot(r_range/r_g, V, '-', lw=2, label=label)

    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=6, color='red', linestyle=':', lw=2, label='ISCO (r = 6 r_g)')
    ax1.axvline(x=3, color='purple', linestyle=':', alpha=0.7, label='Photon sphere')
    ax1.axvline(x=2, color='black', linestyle='-', lw=2, alpha=0.5, label='Event horizon')

    ax1.set_xlabel('r / r_g  (r_g = GM/c^2)')
    ax1.set_ylabel('Effective potential V_eff / c^2')
    ax1.set_title('Effective Potential for Different Angular Momenta')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1.5, 30)
    ax1.set_ylim(0.88, 1.05)

    # Mark the inflection point at ISCO
    ax1.annotate('ISCO: V\'=V\'\'=0',
                xy=(6, effective_potential(6*r_g, 3.46*G*M/c, M)),
                xytext=(12, 0.94),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')

    # ==========================================================================
    # Plot 2: ISCO radius vs spin (Kerr)
    # ==========================================================================
    ax2 = axes[0, 1]

    a_range = np.linspace(0, 0.998, 100)

    r_isco_prograde = [isco_radius_kerr(a, M, prograde=True) for a in a_range]
    r_isco_retrograde = [isco_radius_kerr(a, M, prograde=False) for a in a_range]

    ax2.plot(a_range, np.array(r_isco_prograde)/r_g, 'b-', lw=2,
            label='Prograde (co-rotating)')
    ax2.plot(a_range, np.array(r_isco_retrograde)/r_g, 'r-', lw=2,
            label='Retrograde (counter-rotating)')

    ax2.axhline(y=6, color='gray', linestyle='--', alpha=0.7,
               label='Schwarzschild (a=0)')
    ax2.axhline(y=1, color='green', linestyle=':', alpha=0.7,
               label='Horizon (extreme Kerr)')

    ax2.set_xlabel('Spin parameter a = J/(Mc)')
    ax2.set_ylabel('ISCO radius / r_g')
    ax2.set_title('ISCO Radius vs Black Hole Spin')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 10)

    # Add annotation about accretion efficiency
    ax2.text(0.5, 8.5,
            'ISCO determines accretion disk inner edge\n'
            'and radiative efficiency:\n'
            'a=0: eta = 5.7%\n'
            'a=1 (prograde): eta = 42%',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ==========================================================================
    # Plot 3: Binding energy at ISCO
    # ==========================================================================
    ax3 = axes[1, 0]

    # Binding energy vs spin
    binding_prograde = []
    binding_retrograde = []

    for a in a_range:
        r_pro = isco_radius_kerr(a, M, prograde=True)
        r_ret = isco_radius_kerr(a, M, prograde=False)

        # Simplified binding energy calculation
        # For Kerr, this is more complex, using approximation
        eta_pro = 1 - np.sqrt(1 - 2*r_g/(3*r_pro))
        eta_ret = 1 - np.sqrt(1 - 2*r_g/(3*r_ret))

        binding_prograde.append(eta_pro)
        binding_retrograde.append(eta_ret)

    ax3.plot(a_range, np.array(binding_prograde) * 100, 'b-', lw=2,
            label='Prograde')
    ax3.plot(a_range, np.array(binding_retrograde) * 100, 'r-', lw=2,
            label='Retrograde')

    ax3.axhline(y=5.72, color='gray', linestyle='--', alpha=0.7,
               label='Schwarzschild (5.72%)')

    ax3.set_xlabel('Spin parameter a')
    ax3.set_ylabel('Binding energy at ISCO (%)')
    ax3.set_title('Radiative Efficiency of Accretion')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 1)

    # Compare with nuclear fusion
    ax3.axhline(y=0.7, color='orange', linestyle=':', alpha=0.7)
    ax3.text(0.5, 1.5, 'Nuclear fusion: 0.7%', fontsize=9, color='orange')

    # ==========================================================================
    # Plot 4: ISCO orbital properties
    # ==========================================================================
    ax4 = axes[1, 1]

    # Calculate orbital frequency at ISCO for different masses
    masses = np.logspace(0, 10, 100) * M_sun  # 1 to 10^10 solar masses

    r_isco = isco_radius_schwarzschild(masses)
    omega_isco = orbital_frequency_isco(r_isco, masses)
    f_isco = omega_isco / (2 * np.pi)

    ax4.loglog(masses/M_sun, f_isco, 'b-', lw=2)

    ax4.set_xlabel('Black hole mass (solar masses)')
    ax4.set_ylabel('ISCO orbital frequency (Hz)')
    ax4.set_title('Orbital Frequency at ISCO')
    ax4.grid(True, alpha=0.3, which='both')

    # Mark different types of black holes
    stellar_mass = 10
    intermediate = 1e4
    supermassive = 4e6  # Sgr A*

    for m, name in [(stellar_mass, 'Stellar (10 M_sun)'),
                    (intermediate, 'Intermediate (10^4 M_sun)'),
                    (supermassive, 'Sgr A* (4x10^6 M_sun)')]:
        r = isco_radius_schwarzschild(m * M_sun)
        f = orbital_frequency_isco(r, m * M_sun) / (2 * np.pi)
        ax4.plot(m, f, 'ro', markersize=8)
        ax4.annotate(f'{name}\nf = {f:.1e} Hz',
                    xy=(m, f), xytext=(m*3, f*2),
                    fontsize=8, arrowprops=dict(arrowstyle='->', color='red'))

    # Mark LIGO band
    ax4.axhspan(10, 1000, alpha=0.2, color='green')
    ax4.text(1e2, 100, 'LIGO band', fontsize=10, color='green')

    plt.suptitle('Innermost Stable Circular Orbit (ISCO)\n'
                 'Schwarzschild: r_ISCO = 6 GM/c^2 = 3 r_s',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Print summary
    print("ISCO Summary:")
    print("=" * 60)

    print(f"\nSchwarzschild (non-spinning) black hole:")
    r_isco = isco_radius_schwarzschild(M)
    print(f"  ISCO radius: 6 r_g = 3 r_s = {r_isco:.3e} m")
    print(f"  ISCO radius: {r_isco/1e3:.1f} km for 10 M_sun BH")

    E_bind = binding_energy_isco(r_isco, M)
    print(f"  Binding energy: {E_bind*100:.2f}% of rest mass")

    L_isco = angular_momentum_isco(r_isco, M)
    print(f"  Specific angular momentum: {L_isco:.3e} m^2/s")

    omega = orbital_frequency_isco(r_isco, M)
    print(f"  Orbital frequency: {omega/(2*np.pi):.1f} Hz")

    print(f"\nKerr (spinning) black hole limits:")
    print(f"  a = 0 (Schwarzschild): r_ISCO = 6 r_g")
    print(f"  a = 1 (prograde): r_ISCO = 1 r_g (at horizon)")
    print(f"  a = 1 (retrograde): r_ISCO = 9 r_g")

    print(f"\nAccretion efficiency:")
    print(f"  Schwarzschild: 5.72%")
    print(f"  Kerr a=1 (prograde): ~42%")
    print(f"  Compare: nuclear fusion is only 0.7%!")

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'isco_radius.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")


if __name__ == "__main__":
    main()
