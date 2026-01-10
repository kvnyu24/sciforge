"""
Experiment 228: MOS Capacitor C-V Curve

Demonstrates the capacitance-voltage characteristics of a Metal-Oxide-
Semiconductor (MOS) capacitor, showing accumulation, depletion, and
inversion regimes as the gate voltage is swept.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


# Physical constants
q = 1.602e-19        # Electron charge (C)
epsilon_0 = 8.854e-12  # Vacuum permittivity (F/m)
kB = 1.381e-23       # Boltzmann constant (J/K)


def fermi_potential(Na, ni, T=300):
    """
    Calculate Fermi potential phi_F = (kT/q) * ln(Na/ni)

    Args:
        Na: Acceptor concentration (m^-3)
        ni: Intrinsic concentration (m^-3)
        T: Temperature (K)

    Returns:
        phi_F (V)
    """
    V_T = kB * T / q
    return V_T * np.log(Na / ni)


def threshold_voltage(phi_F, Na, epsilon_s, Cox, phi_ms=0):
    """
    Calculate threshold voltage for strong inversion.

    Vth = phi_ms + 2*phi_F + sqrt(4*epsilon_s*q*Na*phi_F) / Cox

    Args:
        phi_F: Fermi potential
        Na: Acceptor concentration
        epsilon_s: Semiconductor permittivity
        Cox: Oxide capacitance per unit area
        phi_ms: Metal-semiconductor work function difference

    Returns:
        Vth (V)
    """
    Qd_max = np.sqrt(4 * epsilon_s * q * Na * phi_F)
    return phi_ms + 2 * phi_F + Qd_max / Cox


def flatband_voltage(phi_ms, Qf, Cox):
    """
    Calculate flatband voltage.

    Vfb = phi_ms - Qf/Cox

    Args:
        phi_ms: Work function difference
        Qf: Fixed oxide charge density (C/m^2)
        Cox: Oxide capacitance per unit area

    Returns:
        Vfb (V)
    """
    return phi_ms - Qf / Cox


def surface_potential_from_Vg(Vg, Vfb, phi_F, Na, epsilon_s, Cox, T=300):
    """
    Calculate surface potential psi_s from gate voltage (iterative solution).

    Vg = Vfb + psi_s + sqrt(2*epsilon_s*q*Na/Cox^2 * (psi_s + VT*exp((psi_s-2*phi_F)/VT)))

    Args:
        Vg: Gate voltage (array)
        Vfb: Flatband voltage
        phi_F: Fermi potential
        Na: Acceptor concentration
        epsilon_s: Semiconductor permittivity
        Cox: Oxide capacitance
        T: Temperature

    Returns:
        psi_s: Surface potential
    """
    V_T = kB * T / q
    psi_s = np.zeros_like(Vg)

    gamma = np.sqrt(2 * epsilon_s * q * Na) / Cox  # Body effect coefficient

    for i, vg in enumerate(Vg):
        # Newton-Raphson to solve for psi_s
        psi = 0.5  # Initial guess
        for _ in range(100):
            # F function
            if psi > 0:
                arg = min((psi - 2 * phi_F) / V_T, 50)
                Q_term = psi + V_T * np.exp(arg)
            else:
                Q_term = abs(psi)

            Q_term = max(Q_term, 1e-10)
            F = vg - Vfb - psi - gamma * np.sqrt(Q_term)

            # Derivative
            if psi > 0:
                arg = min((psi - 2 * phi_F) / V_T, 50)
                dQ_dpsi = 1 + np.exp(arg)
            else:
                dQ_dpsi = -np.sign(psi) if psi != 0 else 1

            dF_dpsi = -1 - gamma * dQ_dpsi / (2 * np.sqrt(Q_term + 1e-10))

            psi_new = psi - F / dF_dpsi
            if abs(psi_new - psi) < 1e-9:
                break
            psi = psi_new

        psi_s[i] = psi

    return psi_s


def mos_capacitance(Vg, Vfb, phi_F, Na, epsilon_s, Cox, T=300):
    """
    Calculate MOS capacitance vs gate voltage.

    C = Cox / (1 + Cox/Cs) where Cs is semiconductor capacitance

    Args:
        Vg: Gate voltage array
        Vfb: Flatband voltage
        phi_F: Fermi potential
        Na: Acceptor concentration
        epsilon_s: Semiconductor permittivity
        Cox: Oxide capacitance
        T: Temperature

    Returns:
        C: Total capacitance array
        psi_s: Surface potential array
    """
    V_T = kB * T / q
    C = np.zeros_like(Vg)
    psi_s = surface_potential_from_Vg(Vg, Vfb, phi_F, Na, epsilon_s, Cox, T)

    Ld = np.sqrt(epsilon_s * V_T / (q * Na))  # Debye length

    for i, (vg, psi) in enumerate(zip(Vg, psi_s)):
        if psi < 0:
            # Accumulation
            # High capacitance, approaches Cox
            Cs = epsilon_s / Ld * np.exp(-psi / (2 * V_T))
        elif psi < 2 * phi_F:
            # Depletion
            # Wd = sqrt(2*epsilon_s*psi/(q*Na))
            Wd = np.sqrt(2 * epsilon_s * abs(psi) / (q * Na)) if psi > 0 else Ld
            Wd = max(Wd, Ld)
            Cs = epsilon_s / Wd
        else:
            # Inversion
            arg = min((psi - 2 * phi_F) / V_T, 50)
            Cs = epsilon_s / Ld * np.sqrt(np.exp(arg))

        # Total capacitance: series combination
        C[i] = 1 / (1/Cox + 1/Cs)

    return C, psi_s


def main():
    """Main simulation and visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Silicon MOS parameters
    epsilon_s = 11.7 * epsilon_0   # Si permittivity
    epsilon_ox = 3.9 * epsilon_0   # SiO2 permittivity
    tox = 10e-9                    # Oxide thickness (10 nm)
    Cox = epsilon_ox / tox         # Oxide capacitance per area

    Na = 1e23  # Acceptor concentration (10^17 cm^-3)
    ni = 1.5e16  # Intrinsic concentration

    T = 300
    V_T = kB * T / q

    phi_F = fermi_potential(Na, ni, T)
    phi_ms = -0.3  # Work function difference (V)
    Qf = 0  # No fixed charge

    Vfb = flatband_voltage(phi_ms, Qf, Cox)
    Vth = threshold_voltage(phi_F, Na, epsilon_s, Cox, phi_ms)

    print(f"Fermi potential phi_F = {phi_F:.3f} V")
    print(f"Flatband voltage Vfb = {Vfb:.3f} V")
    print(f"Threshold voltage Vth = {Vth:.3f} V")

    # Gate voltage sweep
    Vg = np.linspace(-2, 3, 500)

    # Calculate C-V curve
    C, psi_s = mos_capacitance(Vg, Vfb, phi_F, Na, epsilon_s, Cox, T)

    # Normalize to Cox
    C_norm = C / Cox

    # Plot 1: Normalized C-V curve
    ax1 = axes[0, 0]

    ax1.plot(Vg, C_norm, 'b-', lw=2)
    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='$C_{ox}$')
    ax1.axvline(x=Vfb, color='green', linestyle=':', alpha=0.7, label=f'$V_{{fb}}$ = {Vfb:.2f}V')
    ax1.axvline(x=Vth, color='red', linestyle=':', alpha=0.7, label=f'$V_{{th}}$ = {Vth:.2f}V')

    # Shade regions
    ax1.axvspan(Vg[0], Vfb, alpha=0.1, color='blue', label='Accumulation')
    ax1.axvspan(Vfb, Vth, alpha=0.1, color='yellow', label='Depletion')
    ax1.axvspan(Vth, Vg[-1], alpha=0.1, color='red', label='Inversion')

    ax1.set_xlabel('Gate Voltage (V)')
    ax1.set_ylabel('C / C_ox')
    ax1.set_title('MOS Capacitor C-V Curve')
    ax1.legend(loc='right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)

    # Plot 2: Surface potential
    ax2 = axes[0, 1]

    ax2.plot(Vg, psi_s, 'b-', lw=2)
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax2.axhline(y=phi_F, color='green', linestyle='--', alpha=0.7, label=f'$\\phi_F$ = {phi_F:.2f}V')
    ax2.axhline(y=2*phi_F, color='red', linestyle='--', alpha=0.7, label=f'$2\\phi_F$ = {2*phi_F:.2f}V')

    ax2.axvline(x=Vfb, color='green', linestyle=':', alpha=0.5)
    ax2.axvline(x=Vth, color='red', linestyle=':', alpha=0.5)

    ax2.set_xlabel('Gate Voltage (V)')
    ax2.set_ylabel('Surface Potential $\\psi_s$ (V)')
    ax2.set_title('Surface Potential vs Gate Voltage')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Effect of oxide thickness
    ax3 = axes[1, 0]

    tox_values = [5e-9, 10e-9, 20e-9, 50e-9]
    colors = ['blue', 'green', 'orange', 'red']

    for tox_val, color in zip(tox_values, colors):
        Cox_val = epsilon_ox / tox_val
        Vth_val = threshold_voltage(phi_F, Na, epsilon_s, Cox_val, phi_ms)
        C_val, _ = mos_capacitance(Vg, Vfb, phi_F, Na, epsilon_s, Cox_val, T)
        C_norm_val = C_val / Cox_val

        ax3.plot(Vg, C_norm_val, color=color, lw=2, label=f'tox = {tox_val*1e9:.0f} nm')

    ax3.set_xlabel('Gate Voltage (V)')
    ax3.set_ylabel('C / C_ox')
    ax3.set_title('Effect of Oxide Thickness')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.1)

    # Plot 4: Effect of doping concentration
    ax4 = axes[1, 1]

    Na_values = [1e22, 1e23, 1e24]  # m^-3
    labels = ['10^16', '10^17', '10^18']

    for Na_val, label, color in zip(Na_values, labels, colors[:3]):
        phi_F_val = fermi_potential(Na_val, ni, T)
        Vth_val = threshold_voltage(phi_F_val, Na_val, epsilon_s, Cox, phi_ms)
        Vfb_val = flatband_voltage(phi_ms, Qf, Cox)
        C_val, _ = mos_capacitance(Vg, Vfb_val, phi_F_val, Na_val, epsilon_s, Cox, T)
        C_norm_val = C_val / Cox

        ax4.plot(Vg, C_norm_val, color=color, lw=2, label=f'Na = {label} cm^-3')

    ax4.set_xlabel('Gate Voltage (V)')
    ax4.set_ylabel('C / C_ox')
    ax4.set_title('Effect of Substrate Doping')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1.1)

    plt.suptitle('MOS Capacitor: Capacitance-Voltage Characteristics\n'
                 'Accumulation - Depletion - Inversion regimes',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'mos_capacitor_cv.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'mos_capacitor_cv.png')}")


if __name__ == "__main__":
    main()
