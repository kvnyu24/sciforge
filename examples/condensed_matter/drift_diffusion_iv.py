"""
Experiment 227: Drift-Diffusion I-V Curve

Demonstrates the drift-diffusion model for semiconductor device
current-voltage characteristics, including the ideal diode equation
and deviations due to recombination, series resistance, and breakdown.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


# Physical constants
q = 1.602e-19        # Electron charge (C)
kB = 1.381e-23       # Boltzmann constant (J/K)


def thermal_voltage(T):
    """Thermal voltage V_T = kT/q"""
    return kB * T / q


def ideal_diode_current(V, I0, T=300):
    """
    Ideal diode (Shockley) equation.

    I = I0 * (exp(qV/kT) - 1)

    Args:
        V: Applied voltage (array)
        I0: Saturation current
        T: Temperature (K)

    Returns:
        Current I
    """
    V_T = thermal_voltage(T)
    # Clip exponent to avoid overflow
    exponent = np.clip(V / V_T, -100, 100)
    return I0 * (np.exp(exponent) - 1)


def diode_current_with_ideality(V, I0, n, T=300):
    """
    Diode equation with ideality factor.

    I = I0 * (exp(qV/nkT) - 1)

    Args:
        V: Applied voltage
        I0: Saturation current
        n: Ideality factor (1 for ideal, ~2 for recombination-dominated)
        T: Temperature

    Returns:
        Current I
    """
    V_T = thermal_voltage(T)
    exponent = np.clip(V / (n * V_T), -100, 100)
    return I0 * (np.exp(exponent) - 1)


def diode_current_with_series_resistance(V, I0, n, Rs, T=300, max_iter=100):
    """
    Diode with series resistance (implicit equation solved iteratively).

    I = I0 * (exp((V - I*Rs)/(n*V_T)) - 1)

    Args:
        V: Applied voltage
        I0: Saturation current
        n: Ideality factor
        Rs: Series resistance
        T: Temperature
        max_iter: Maximum iterations for Newton-Raphson

    Returns:
        Current I
    """
    V_T = thermal_voltage(T)

    # Initial guess: ideal diode current
    I = diode_current_with_ideality(V, I0, n, T)

    # Newton-Raphson iteration
    for _ in range(max_iter):
        V_d = V - I * Rs  # Voltage across diode
        exponent = np.clip(V_d / (n * V_T), -100, 100)
        f = I - I0 * (np.exp(exponent) - 1)
        df_dI = 1 + I0 * Rs / (n * V_T) * np.exp(exponent)
        I_new = I - f / df_dI
        if np.max(np.abs(I_new - I)) < 1e-12:
            break
        I = I_new

    return I


def reverse_breakdown_current(V, I0, Vbr, n_br=3, T=300):
    """
    Diode current including avalanche breakdown in reverse bias.

    Args:
        V: Applied voltage
        I0: Saturation current
        Vbr: Breakdown voltage (negative)
        n_br: Breakdown steepness parameter
        T: Temperature

    Returns:
        Current I
    """
    V_T = thermal_voltage(T)

    # Forward current
    I_forward = ideal_diode_current(V, I0, T)

    # Breakdown contribution (empirical model)
    mask = V < 0
    I_breakdown = np.zeros_like(V)
    I_breakdown[mask] = -I0 * (np.abs(V[mask]) / np.abs(Vbr))**n_br

    return I_forward + I_breakdown


def drift_diffusion_1d(x, n, p, E, Dn, Dp, mu_n, mu_p):
    """
    Compute electron and hole currents from drift-diffusion.

    Jn = q * (mu_n * n * E + Dn * dn/dx)
    Jp = q * (mu_p * p * E - Dp * dp/dx)

    Args:
        x: Position array
        n, p: Electron and hole concentrations
        E: Electric field
        Dn, Dp: Diffusion coefficients
        mu_n, mu_p: Mobilities

    Returns:
        Jn, Jp: Current densities
    """
    dx = x[1] - x[0]

    # Gradients (central difference)
    dn_dx = np.gradient(n, dx)
    dp_dx = np.gradient(p, dx)

    # Current densities
    Jn = q * (mu_n * n * E + Dn * dn_dx)
    Jp = q * (mu_p * p * E - Dp * dp_dx)

    return Jn, Jp


def main():
    """Main simulation and visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    T = 300  # Temperature (K)
    V_T = thermal_voltage(T)
    I0 = 1e-12  # Saturation current (A)

    # Voltage range
    V = np.linspace(-1, 0.8, 1000)

    # Plot 1: Ideal diode I-V curve (linear and log scale)
    ax1 = axes[0, 0]

    I_ideal = ideal_diode_current(V, I0, T)

    ax1.plot(V, I_ideal * 1e3, 'b-', lw=2, label='Ideal diode')
    ax1.set_xlabel('Voltage (V)')
    ax1.set_ylabel('Current (mA)')
    ax1.set_title('Ideal Diode I-V Characteristic (Linear Scale)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 10)

    # Mark turn-on voltage (~0.6V for Si)
    ax1.axvline(x=0.6, color='gray', linestyle='--', alpha=0.5)
    ax1.text(0.62, 5, 'Turn-on\n~0.6V', fontsize=10)

    # Inset: log scale
    ax1_inset = ax1.inset_axes([0.15, 0.5, 0.35, 0.4])
    V_forward = np.linspace(0.1, 0.7, 100)
    I_forward = ideal_diode_current(V_forward, I0, T)
    ax1_inset.semilogy(V_forward, I_forward, 'b-', lw=2)
    ax1_inset.set_xlabel('V (V)', fontsize=8)
    ax1_inset.set_ylabel('I (A)', fontsize=8)
    ax1_inset.set_title('Log scale', fontsize=8)
    ax1_inset.tick_params(labelsize=6)
    ax1_inset.grid(True, alpha=0.3)

    # Plot 2: Effect of ideality factor
    ax2 = axes[0, 1]

    n_values = [1.0, 1.3, 1.6, 2.0]
    colors = ['blue', 'green', 'orange', 'red']

    V_forward = np.linspace(0, 0.8, 200)

    for n, color in zip(n_values, colors):
        I = diode_current_with_ideality(V_forward, I0, n, T)
        ax2.semilogy(V_forward, I, color=color, lw=2, label=f'n = {n}')

    ax2.set_xlabel('Voltage (V)')
    ax2.set_ylabel('Current (A)')
    ax2.set_title('Effect of Ideality Factor n')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(1e-14, 1)

    # Add explanation
    ax2.text(0.1, 1e-3, 'n = 1: Diffusion-dominated\nn = 2: Recombination-dominated',
             fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot 3: Effect of series resistance
    ax3 = axes[1, 0]

    Rs_values = [0, 10, 50, 100]  # Ohms
    V_full = np.linspace(-0.5, 1.5, 300)

    for Rs in Rs_values:
        if Rs == 0:
            I = ideal_diode_current(V_full, I0, T)
        else:
            I = diode_current_with_series_resistance(V_full, I0, 1.0, Rs, T)

        ax3.plot(V_full, I * 1e3, lw=2, label=f'Rs = {Rs} Ohm')

    ax3.set_xlabel('Voltage (V)')
    ax3.set_ylabel('Current (mA)')
    ax3.set_title('Effect of Series Resistance')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-0.5, 15)

    # Plot 4: Complete I-V with breakdown
    ax4 = axes[1, 1]

    V_wide = np.linspace(-20, 1, 500)
    Vbr = -15  # Breakdown voltage

    I_complete = reverse_breakdown_current(V_wide, I0, Vbr, n_br=5, T=T)

    ax4.plot(V_wide, I_complete * 1e3, 'b-', lw=2)

    # Mark regions
    ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax4.axvline(x=Vbr, color='red', linestyle='--', alpha=0.5)

    ax4.fill_between([Vbr-2, Vbr], -100, 10, alpha=0.2, color='red')
    ax4.text(Vbr-1, -50, 'Breakdown', fontsize=10, rotation=90, va='center')

    ax4.text(-7, 0.5, 'Reverse bias\n(leakage)', fontsize=10, ha='center')
    ax4.text(0.5, 5, 'Forward\nbias', fontsize=10, ha='center')

    ax4.set_xlabel('Voltage (V)')
    ax4.set_ylabel('Current (mA)')
    ax4.set_title(f'Complete I-V Curve with Breakdown (Vbr = {Vbr} V)')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-100, 15)

    plt.suptitle('Drift-Diffusion Model: Diode I-V Characteristics\n'
                 f'I = I0(exp(V/nVT) - 1), T = {T}K, VT = {V_T*1000:.1f} mV',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'drift_diffusion_iv.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'drift_diffusion_iv.png')}")

    # Additional figure: Temperature dependence
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    temperatures = [250, 300, 350, 400]  # K
    V_forward = np.linspace(0, 0.8, 200)

    for T_val in temperatures:
        # I0 also depends on T, roughly as T^3 * exp(-Eg/kT)
        I0_T = I0 * (T_val/300)**3 * np.exp(-1.12*q/(kB*300) * (1 - 300/T_val))
        I = ideal_diode_current(V_forward, I0_T, T_val)
        ax2.semilogy(V_forward, I, lw=2, label=f'T = {T_val} K')

    ax2.set_xlabel('Voltage (V)')
    ax2.set_ylabel('Current (A)')
    ax2.set_title('Temperature Dependence of Diode I-V')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, 'drift_diffusion_temperature.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'drift_diffusion_temperature.png')}")


if __name__ == "__main__":
    main()
