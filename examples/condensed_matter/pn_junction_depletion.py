"""
Experiment 226: PN Junction Depletion Width

Demonstrates the physics of pn junction formation, including the
depletion region, built-in potential, electric field distribution,
and how depletion width varies with doping and applied bias.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


# Physical constants (SI units)
q = 1.602e-19        # Electron charge (C)
epsilon_0 = 8.854e-12  # Vacuum permittivity (F/m)
kB = 1.381e-23       # Boltzmann constant (J/K)
T = 300              # Temperature (K)
V_T = kB * T / q     # Thermal voltage (~26 mV at 300K)


def depletion_width(Na, Nd, epsilon_r, Vbi, Va=0):
    """
    Calculate depletion region width.

    W = sqrt(2*epsilon*(Vbi - Va)*(Na + Nd)/(q*Na*Nd))

    Args:
        Na: Acceptor concentration (m^-3)
        Nd: Donor concentration (m^-3)
        epsilon_r: Relative permittivity
        Vbi: Built-in potential (V)
        Va: Applied voltage (V), positive = forward bias

    Returns:
        W: Total depletion width (m)
        xp: Depletion width on p-side (m)
        xn: Depletion width on n-side (m)
    """
    epsilon = epsilon_0 * epsilon_r
    V_eff = Vbi - Va

    if V_eff <= 0:
        return 0, 0, 0

    W = np.sqrt(2 * epsilon * V_eff * (Na + Nd) / (q * Na * Nd))
    xp = W * Nd / (Na + Nd)
    xn = W * Na / (Na + Nd)

    return W, xp, xn


def built_in_potential(Na, Nd, ni):
    """
    Calculate built-in potential.

    Vbi = (kT/q) * ln(Na * Nd / ni^2)

    Args:
        Na: Acceptor concentration
        Nd: Donor concentration
        ni: Intrinsic carrier concentration

    Returns:
        Vbi: Built-in potential (V)
    """
    return V_T * np.log(Na * Nd / ni**2)


def electric_field_profile(x, xp, xn, Na, Nd, epsilon_r):
    """
    Electric field distribution in depletion region.

    Args:
        x: Position array (0 at junction)
        xp, xn: Depletion widths
        Na, Nd: Doping concentrations
        epsilon_r: Relative permittivity

    Returns:
        E: Electric field array (V/m)
    """
    epsilon = epsilon_0 * epsilon_r
    E = np.zeros_like(x)

    # p-side (-xp < x < 0)
    mask_p = (x >= -xp) & (x < 0)
    E[mask_p] = -q * Na / epsilon * (x[mask_p] + xp)

    # n-side (0 < x < xn)
    mask_n = (x >= 0) & (x <= xn)
    E[mask_n] = q * Nd / epsilon * (x[mask_n] - xn)

    return E


def potential_profile(x, xp, xn, Na, Nd, epsilon_r, Vbi):
    """
    Electrostatic potential distribution.

    Args:
        x: Position array
        xp, xn: Depletion widths
        Na, Nd: Doping concentrations
        epsilon_r: Relative permittivity
        Vbi: Built-in potential

    Returns:
        V: Potential array (V)
    """
    epsilon = epsilon_0 * epsilon_r
    V = np.zeros_like(x)

    # Reference: V = 0 at x = -xp (p-side neutral region)

    # p-side neutral region (x < -xp)
    mask_p_neutral = x < -xp
    V[mask_p_neutral] = 0

    # p-side depletion (-xp <= x < 0)
    mask_p = (x >= -xp) & (x < 0)
    V[mask_p] = q * Na / (2 * epsilon) * (x[mask_p] + xp)**2

    # n-side depletion (0 <= x <= xn)
    mask_n = (x >= 0) & (x <= xn)
    V_at_0 = q * Na * xp**2 / (2 * epsilon)
    V[mask_n] = V_at_0 + q * Nd / epsilon * (xn * x[mask_n] - x[mask_n]**2 / 2)

    # n-side neutral region (x > xn)
    mask_n_neutral = x > xn
    V[mask_n_neutral] = Vbi

    return V


def main():
    """Main simulation and visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Silicon parameters
    epsilon_r = 11.7  # Relative permittivity of Si
    ni = 1.5e16       # Intrinsic carrier concentration (m^-3)

    # Doping concentrations (asymmetric junction)
    Na = 1e23         # Acceptor concentration (m^-3) = 10^17 cm^-3
    Nd = 1e22         # Donor concentration (m^-3) = 10^16 cm^-3

    # Built-in potential
    Vbi = built_in_potential(Na, Nd, ni)
    print(f"Built-in potential: {Vbi:.3f} V")

    # Depletion width at equilibrium
    W, xp, xn = depletion_width(Na, Nd, epsilon_r, Vbi)
    print(f"Depletion width: {W*1e9:.1f} nm (xp={xp*1e9:.1f} nm, xn={xn*1e9:.1f} nm)")

    # Plot 1: Charge, field, and potential profiles
    ax1 = axes[0, 0]

    x = np.linspace(-1.5 * xp, 1.5 * xn, 1000)
    x_nm = x * 1e9  # Convert to nm

    # Charge density
    rho = np.zeros_like(x)
    rho[(x >= -xp) & (x < 0)] = -q * Na
    rho[(x >= 0) & (x <= xn)] = q * Nd

    # Normalize for plotting
    rho_norm = rho / q / 1e23

    ax1.fill_between(x_nm, rho_norm, where=rho_norm < 0, alpha=0.5, color='blue', label='p-side (acceptors)')
    ax1.fill_between(x_nm, rho_norm, where=rho_norm > 0, alpha=0.5, color='red', label='n-side (donors)')
    ax1.axvline(x=0, color='black', linestyle='-', lw=2)
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.5)

    ax1.set_xlabel('Position (nm)')
    ax1.set_ylabel('Charge density (10^23 q/m^3)')
    ax1.set_title('Charge Distribution in Depletion Region')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Electric field
    ax2 = axes[0, 1]

    E = electric_field_profile(x, xp, xn, Na, Nd, epsilon_r)
    E_kV_cm = E / 1e5  # Convert to kV/cm

    ax2.plot(x_nm, E_kV_cm, 'b-', lw=2)
    ax2.fill_between(x_nm, E_kV_cm, alpha=0.3)
    ax2.axvline(x=0, color='black', linestyle=':', alpha=0.7)
    ax2.axvline(x=-xp*1e9, color='gray', linestyle='--', alpha=0.5, label='Depletion edges')
    ax2.axvline(x=xn*1e9, color='gray', linestyle='--', alpha=0.5)

    ax2.set_xlabel('Position (nm)')
    ax2.set_ylabel('Electric Field (kV/cm)')
    ax2.set_title('Electric Field Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Mark maximum field
    E_max = np.min(E_kV_cm)
    ax2.annotate(f'E_max = {abs(E_max):.1f} kV/cm',
                 xy=(0, E_max), xytext=(50, E_max*0.5),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=10)

    # Plot 3: Potential and band diagram
    ax3 = axes[1, 0]

    V = potential_profile(x, xp, xn, Na, Nd, epsilon_r, Vbi)

    ax3.plot(x_nm, V, 'b-', lw=2, label='Electrostatic potential')
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.axhline(y=Vbi, color='gray', linestyle='--', alpha=0.5)

    # Mark built-in potential
    ax3.annotate('', xy=(xn*1e9*1.2, Vbi), xytext=(xn*1e9*1.2, 0),
                 arrowprops=dict(arrowstyle='<->', color='red'))
    ax3.text(xn*1e9*1.3, Vbi/2, f'$V_{{bi}}$ = {Vbi:.2f} V', fontsize=10, color='red')

    ax3.set_xlabel('Position (nm)')
    ax3.set_ylabel('Potential (V)')
    ax3.set_title('Electrostatic Potential')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Depletion width vs applied voltage
    ax4 = axes[1, 1]

    Va_range = np.linspace(-10, Vbi - 0.05, 100)
    W_values = []
    xp_values = []
    xn_values = []

    for Va in Va_range:
        W_va, xp_va, xn_va = depletion_width(Na, Nd, epsilon_r, Vbi, Va)
        W_values.append(W_va * 1e9)
        xp_values.append(xp_va * 1e9)
        xn_values.append(xn_va * 1e9)

    ax4.plot(Va_range, W_values, 'k-', lw=2, label='Total W')
    ax4.plot(Va_range, xp_values, 'b--', lw=1.5, label='p-side (xp)')
    ax4.plot(Va_range, xn_values, 'r--', lw=1.5, label='n-side (xn)')

    ax4.axvline(x=0, color='gray', linestyle=':', alpha=0.7)
    ax4.axhspan(0, W*1e9, xmin=0.5, alpha=0.1, color='green', label='Equilibrium W')

    # Mark regions
    ax4.text(-5, max(W_values)*0.8, 'Reverse bias\n(W increases)', fontsize=10, ha='center')
    ax4.text(Vbi/2, max(W_values)*0.3, 'Forward bias\n(W decreases)', fontsize=10, ha='center')

    ax4.set_xlabel('Applied Voltage (V)')
    ax4.set_ylabel('Depletion Width (nm)')
    ax4.set_title('Depletion Width vs Applied Voltage')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f'PN Junction Depletion Region\n'
                 f'Si junction: Na = 10^17 cm^-3, Nd = 10^16 cm^-3, Vbi = {Vbi:.2f} V',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'pn_junction_depletion.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'pn_junction_depletion.png')}")


if __name__ == "__main__":
    main()
