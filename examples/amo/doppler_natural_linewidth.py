"""
Experiment 248: Doppler Broadening vs Natural Linewidth

This example compares Doppler broadening (due to thermal motion) with the
natural linewidth (due to spontaneous emission). We explore:
- Gaussian Doppler profile for thermal atoms
- Lorentzian natural lineshape
- Voigt profile (convolution of both effects)
- Temperature dependence of Doppler width
- Crossover between Doppler-limited and natural-limited regimes

Key physics:
- Doppler width: Delta_D = omega_0 * sqrt(2 k_B T / m c^2) ~ sqrt(T)
- Natural width: gamma = 1 / tau (fixed for each transition)
- Cold atoms: natural-limited; Hot atoms: Doppler-limited
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import voigt_profile
from src.sciforge.physics.amo import HBAR, KB, C, M_PROTON

def doppler_width(omega_0, T, m):
    """
    Calculate Doppler FWHM width.

    Delta_D = omega_0 * sqrt(8 * k_B * T * ln(2) / (m * c^2))

    Args:
        omega_0: Transition angular frequency (rad/s)
        T: Temperature (K)
        m: Atomic mass (kg)

    Returns:
        Doppler FWHM (rad/s)
    """
    return omega_0 * np.sqrt(8 * KB * T * np.log(2) / (m * C**2))


def gaussian_profile(delta, sigma):
    """Gaussian (Doppler) lineshape normalized to peak = 1."""
    return np.exp(-delta**2 / (2 * sigma**2))


def lorentzian_profile(delta, gamma):
    """Lorentzian (natural) lineshape normalized to peak = 1."""
    return (gamma/2)**2 / (delta**2 + (gamma/2)**2)


def voigt_lineshape(delta, sigma, gamma):
    """
    Voigt profile: convolution of Gaussian and Lorentzian.

    Normalized to peak = 1.
    """
    # scipy's voigt_profile uses sigma and gamma parameters directly
    v = voigt_profile(delta, sigma, gamma/2)
    return v / np.max(v)


def calculate_lineshapes():
    """Calculate and compare lineshapes at various conditions."""

    # Rubidium-87 D2 line parameters
    wavelength = 780e-9  # m
    omega_0 = 2 * np.pi * C / wavelength  # rad/s
    gamma = 2 * np.pi * 6.07e6  # Natural linewidth (rad/s)
    m_Rb = 87 * M_PROTON  # Rb-87 mass

    results = {}
    results['gamma'] = gamma
    results['omega_0'] = omega_0
    results['m'] = m_Rb

    # 1. Temperature-dependent Doppler width
    print("Computing Doppler widths vs temperature...")
    T_range = np.logspace(0, 4, 100)  # 1 K to 10000 K
    Delta_D = np.array([doppler_width(omega_0, T, m_Rb) for T in T_range])

    results['temp_dependence'] = {
        'T': T_range,
        'Delta_D': Delta_D,
        'Delta_D_MHz': Delta_D / (2 * np.pi * 1e6),
        'gamma_MHz': gamma / (2 * np.pi * 1e6)
    }

    # Find crossover temperature (Doppler = natural)
    T_crossover = m_Rb * C**2 * (gamma / omega_0)**2 / (8 * KB * np.log(2))
    results['T_crossover'] = T_crossover
    print(f"Crossover temperature: {T_crossover*1e3:.2f} mK")

    # 2. Lineshape comparison at different temperatures
    print("Computing lineshapes at various temperatures...")
    temperatures = [300, 1000, 1e-3, 1e-6]  # Room temp, hot, cold, ultracold
    temp_labels = ['300 K (room temp)', '1000 K (hot)', '1 mK (cold)', '1 uK (ultracold)']

    delta_range = np.linspace(-5e9 * 2 * np.pi, 5e9 * 2 * np.pi, 1000)  # +/- 5 GHz

    lineshapes = []
    for T, label in zip(temperatures, temp_labels):
        Delta_D_T = doppler_width(omega_0, T, m_Rb)
        sigma_D = Delta_D_T / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma

        # Calculate profiles
        gauss = gaussian_profile(delta_range, sigma_D)
        lorentz = lorentzian_profile(delta_range, gamma)
        voigt = voigt_lineshape(delta_range, sigma_D, gamma)

        lineshapes.append({
            'T': T,
            'label': label,
            'Delta_D': Delta_D_T,
            'delta': delta_range,
            'gaussian': gauss,
            'lorentzian': lorentz,
            'voigt': voigt
        })

    results['lineshapes'] = lineshapes

    # 3. Zoom on natural linewidth for cold atoms
    print("Computing cold atom lineshape (natural-limited)...")
    delta_narrow = np.linspace(-50e6 * 2 * np.pi, 50e6 * 2 * np.pi, 500)  # +/- 50 MHz

    T_cold = 1e-6  # 1 microkelvin
    Delta_D_cold = doppler_width(omega_0, T_cold, m_Rb)
    sigma_cold = Delta_D_cold / (2 * np.sqrt(2 * np.log(2)))

    results['cold_atoms'] = {
        'T': T_cold,
        'Delta_D': Delta_D_cold,
        'delta': delta_narrow,
        'lorentzian': lorentzian_profile(delta_narrow, gamma),
        'voigt': voigt_lineshape(delta_narrow, sigma_cold, gamma)
    }

    # 4. Ratio of Doppler to natural width
    ratio = Delta_D / gamma
    results['temp_dependence']['ratio'] = ratio

    return results


def plot_results(results):
    """Create comprehensive visualization."""

    fig = plt.figure(figsize=(14, 12))
    gamma_MHz = results['gamma'] / (2 * np.pi * 1e6)

    # Plot 1: Doppler width vs temperature
    ax1 = fig.add_subplot(2, 2, 1)
    td = results['temp_dependence']
    ax1.loglog(td['T'], td['Delta_D_MHz'], 'b-', linewidth=2, label='Doppler width')
    ax1.axhline(y=gamma_MHz, color='r', linestyle='--', linewidth=2, label='Natural width')

    # Mark crossover
    T_cross = results['T_crossover']
    ax1.axvline(x=T_cross, color='green', linestyle=':', alpha=0.7)
    ax1.plot(T_cross, gamma_MHz, 'go', markersize=10, label=f'Crossover: {T_cross*1e3:.1f} mK')

    # Mark common temperatures
    for T, label in [(300, 'Room'), (1, '1 K'), (1e-3, '1 mK'), (1e-6, '1 uK')]:
        if 0.5 < T < 5000:
            ax1.axvline(x=T, color='gray', linestyle=':', alpha=0.3)

    ax1.set_xlabel('Temperature (K)', fontsize=11)
    ax1.set_ylabel('Linewidth (MHz)', fontsize=11)
    ax1.set_title('Doppler Width vs Temperature (Rb-87 D2 line)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_xlim(1e-7, 1e4)
    ax1.set_ylim(1e-3, 1e4)
    ax1.grid(True, alpha=0.3, which='both')

    # Add regime labels
    ax1.text(1e-5, 1e3, 'Doppler-limited', fontsize=10, style='italic')
    ax1.text(1e-5, 1, 'Natural-limited', fontsize=10, style='italic')

    # Plot 2: Lineshapes at room temperature (Doppler-dominated)
    ax2 = fig.add_subplot(2, 2, 2)
    ls_room = results['lineshapes'][0]  # 300 K
    delta_GHz = ls_room['delta'] / (2 * np.pi * 1e9)

    ax2.plot(delta_GHz, ls_room['gaussian'], 'b-', linewidth=2, label='Gaussian (Doppler)')
    ax2.plot(delta_GHz, ls_room['lorentzian'], 'r--', linewidth=2, label='Lorentzian (Natural)')
    ax2.plot(delta_GHz, ls_room['voigt'], 'g-', linewidth=2, label='Voigt (Combined)')

    ax2.set_xlabel('Detuning (GHz)', fontsize=11)
    ax2.set_ylabel('Normalized Intensity', fontsize=11)
    ax2.set_title(f'Lineshapes at 300 K (Doppler FWHM = {ls_room["Delta_D"]/(2*np.pi*1e9):.2f} GHz)', fontsize=12)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_xlim(-5, 5)
    ax2.grid(True, alpha=0.3)

    # Indicate FWHM
    fwhm_GHz = ls_room['Delta_D'] / (2 * np.pi * 1e9)
    ax2.annotate('', xy=(fwhm_GHz/2, 0.5), xytext=(-fwhm_GHz/2, 0.5),
                arrowprops=dict(arrowstyle='<->', color='blue', lw=1.5))
    ax2.text(0, 0.55, f'FWHM = {fwhm_GHz*1000:.0f} MHz', fontsize=9, ha='center')

    # Plot 3: Cold atom lineshape (natural-limited)
    ax3 = fig.add_subplot(2, 2, 3)
    cold = results['cold_atoms']
    delta_MHz = cold['delta'] / (2 * np.pi * 1e6)

    ax3.plot(delta_MHz, cold['lorentzian'], 'r-', linewidth=2, label='Lorentzian (Natural)')
    ax3.plot(delta_MHz, cold['voigt'], 'g--', linewidth=2, label='Voigt (nearly identical)')

    ax3.set_xlabel('Detuning (MHz)', fontsize=11)
    ax3.set_ylabel('Normalized Intensity', fontsize=11)
    ax3.set_title(f'Ultracold Atoms at 1 $\\mu$K (Doppler << Natural)', fontsize=12)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.set_xlim(-50, 50)
    ax3.grid(True, alpha=0.3)

    # Indicate natural linewidth
    ax3.annotate('', xy=(gamma_MHz/2, 0.5), xytext=(-gamma_MHz/2, 0.5),
                arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
    ax3.text(0, 0.55, f'$\\gamma$ = {gamma_MHz:.1f} MHz', fontsize=9, ha='center')

    # Add text box
    Delta_D_cold_kHz = cold['Delta_D'] / (2 * np.pi * 1e3)
    textstr = f'Doppler width: {Delta_D_cold_kHz:.1f} kHz\nNatural width: {gamma_MHz:.1f} MHz\nRatio: {gamma_MHz*1e3/Delta_D_cold_kHz:.0f}'
    ax3.text(0.05, 0.95, textstr, transform=ax3.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 4: Comparison of all regimes
    ax4 = fig.add_subplot(2, 2, 4)

    # Ratio of Doppler to natural width
    td = results['temp_dependence']
    ax4.loglog(td['T'], td['ratio'], 'b-', linewidth=2)
    ax4.axhline(y=1, color='r', linestyle='--', linewidth=1.5, label='Doppler = Natural')

    # Fill regions
    ax4.fill_between(td['T'], 1, td['ratio'], where=(td['ratio'] > 1),
                    alpha=0.2, color='red', label='Doppler-limited')
    ax4.fill_between(td['T'], 1, td['ratio'], where=(td['ratio'] < 1),
                    alpha=0.2, color='blue', label='Natural-limited')

    ax4.set_xlabel('Temperature (K)', fontsize=11)
    ax4.set_ylabel(r'$\Delta_D / \gamma$', fontsize=11)
    ax4.set_title('Ratio of Doppler to Natural Linewidth', fontsize=12)
    ax4.legend(loc='upper left', fontsize=9)
    ax4.set_xlim(1e-7, 1e4)
    ax4.grid(True, alpha=0.3, which='both')

    # Mark key temperatures
    for T, label, color in [(300, 'Room temp', 'orange'),
                            (1e-3, 'Doppler limit', 'green'),
                            (1e-6, 'Sub-Doppler', 'purple')]:
        ax4.axvline(x=T, color=color, linestyle=':', alpha=0.5)
        idx = np.argmin(np.abs(td['T'] - T))
        ax4.plot(T, td['ratio'][idx], 'o', color=color, markersize=8)

    plt.tight_layout()
    return fig


def main():
    print("=" * 60)
    print("Experiment 248: Doppler Broadening vs Natural Linewidth")
    print("=" * 60)
    print()

    # Run calculations
    print("Running calculations...")
    results = calculate_lineshapes()

    # Create visualization
    print("\nCreating visualization...")
    fig = plot_results(results)

    # Save figure
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'doppler_natural_linewidth.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    # Print summary
    gamma = results['gamma']
    print("\n" + "=" * 60)
    print("Summary (Rb-87 D2 line at 780 nm):")
    print("=" * 60)
    print(f"Natural linewidth gamma: {gamma/(2*np.pi*1e6):.2f} MHz")
    print(f"Spontaneous lifetime: {1/gamma*1e9:.1f} ns")
    print()
    print("Doppler widths at various temperatures:")
    for ls in results['lineshapes']:
        print(f"  {ls['label']}: {ls['Delta_D']/(2*np.pi*1e6):.3g} MHz")
    print()
    print(f"Crossover temperature (Doppler = Natural): {results['T_crossover']*1e3:.2f} mK")
    print()
    print("Regimes:")
    print("  Room temperature: Doppler-limited (FWHM ~ 500 MHz >> gamma)")
    print("  Laser cooling: Doppler-limited until Doppler temperature")
    print("  Ultracold atoms: Natural-limited (resolved atomic structure)")

    plt.close()


if __name__ == "__main__":
    main()
