"""
Example demonstrating Stefan-Boltzmann law and blackbody radiation.

This example shows how radiated power depends on temperature (T^4),
and compares blackbody spectra at different temperatures.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.thermodynamics import ThermalSystem


def planck_distribution(wavelength, T):
    """
    Calculate spectral radiance using Planck's law.

    Args:
        wavelength: Wavelength in meters
        T: Temperature in Kelvin

    Returns:
        Spectral radiance in W/(m^2·sr·m)
    """
    h = 6.626e-34   # Planck constant
    c = 3e8         # Speed of light
    k_B = 1.381e-23 # Boltzmann constant

    # Avoid overflow/underflow
    exponent = h * c / (wavelength * k_B * T)
    exponent = np.clip(exponent, -700, 700)

    return (2 * h * c**2 / wavelength**5) / (np.exp(exponent) - 1)


def wien_displacement(T):
    """Calculate peak wavelength using Wien's law."""
    b = 2.898e-3  # Wien's displacement constant (m·K)
    return b / T


def stefan_boltzmann_power(T, emissivity=1.0):
    """Calculate total radiated power per unit area."""
    sigma = 5.67e-8  # Stefan-Boltzmann constant (W/(m^2·K^4))
    return emissivity * sigma * T**4


def main():
    # Constants
    sigma = 5.67e-8  # Stefan-Boltzmann constant

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Blackbody spectra at different temperatures
    ax1 = axes[0, 0]

    wavelengths = np.linspace(1e-9, 3e-6, 1000)  # 1 nm to 3 μm
    temperatures = [3000, 4000, 5000, 6000, 7000]  # Kelvin
    colors = plt.cm.hot(np.linspace(0.3, 0.9, len(temperatures)))

    for T, color in zip(temperatures, colors):
        B = planck_distribution(wavelengths, T)
        ax1.plot(wavelengths * 1e9, B / 1e12, color=color, lw=2, label=f'T = {T} K')

        # Mark peak wavelength
        lambda_max = wien_displacement(T)
        B_max = planck_distribution(lambda_max, T)
        ax1.plot(lambda_max * 1e9, B_max / 1e12, 'o', color=color, markersize=6)

    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Spectral Radiance (TW/(m²·sr·m))')
    ax1.set_title('Planck Blackbody Spectra')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 2000)

    # Add visible spectrum band
    ax1.axvspan(380, 700, alpha=0.2, color='yellow', label='Visible')

    # Plot 2: Wien's displacement law
    ax2 = axes[0, 1]

    T_range = np.linspace(1000, 10000, 100)
    lambda_max = wien_displacement(T_range)

    ax2.plot(T_range, lambda_max * 1e9, 'b-', lw=2)
    ax2.fill_between(T_range, 380, 700, where=(lambda_max*1e9 >= 380) & (lambda_max*1e9 <= 700),
                     alpha=0.3, color='yellow', label='Visible peak')

    ax2.set_xlabel('Temperature (K)')
    ax2.set_ylabel('Peak Wavelength (nm)')
    ax2.set_title("Wien's Displacement Law: λ_max × T = 2898 μm·K")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 3000)

    # Mark notable temperatures
    stars = {'Sun (~5778K)': 5778, 'Sirius (~9940K)': 9940, 'Red Giant (~3500K)': 3500}
    for name, T in stars.items():
        lam = wien_displacement(T)
        ax2.plot(T, lam * 1e9, 'ro', markersize=8)
        ax2.annotate(name, (T, lam * 1e9), xytext=(5, 10), textcoords='offset points', fontsize=8)

    # Plot 3: Stefan-Boltzmann law (power vs temperature)
    ax3 = axes[1, 0]

    T_range = np.linspace(200, 2000, 100)
    P_range = stefan_boltzmann_power(T_range)

    ax3.semilogy(T_range, P_range, 'r-', lw=2, label='P = σT⁴')
    ax3.set_xlabel('Temperature (K)')
    ax3.set_ylabel('Radiated Power (W/m²)')
    ax3.set_title('Stefan-Boltzmann Law: P = σT⁴')
    ax3.grid(True, alpha=0.3, which='both')

    # Mark some reference points
    references = {
        'Room temp (300K)': 300,
        'Boiling water (373K)': 373,
        'Iron melting (1811K)': 1811
    }
    for name, T in references.items():
        P = stefan_boltzmann_power(T)
        ax3.plot(T, P, 'ko', markersize=8)
        ax3.annotate(f'{name}\nP = {P:.0f} W/m²', (T, P),
                    xytext=(10, 0), textcoords='offset points', fontsize=8)

    # Plot 4: T^4 dependence illustration
    ax4 = axes[1, 1]

    T_ratio = np.linspace(1, 3, 50)  # T/T0 ratio
    P_ratio = T_ratio**4

    ax4.plot(T_ratio, P_ratio, 'b-', lw=2, label='P ∝ T⁴')
    ax4.plot(T_ratio, T_ratio, 'g--', lw=1.5, label='Linear (T)')
    ax4.plot(T_ratio, T_ratio**2, 'r:', lw=1.5, label='Quadratic (T²)')

    ax4.set_xlabel('Temperature Ratio (T/T₀)')
    ax4.set_ylabel('Power Ratio (P/P₀)')
    ax4.set_title('Power Scales as T⁴')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Add annotation
    ax4.annotate('Doubling T increases\npower by 16×!',
                xy=(2, 16), xytext=(2.2, 40),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Stefan-Boltzmann Law and Blackbody Radiation\n'
                 f'σ = {sigma:.3e} W/(m²·K⁴)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'stefan_boltzmann.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'stefan_boltzmann.png')}")


if __name__ == "__main__":
    main()
