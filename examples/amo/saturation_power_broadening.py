"""
Experiment 247: Saturation Intensity and Power Broadening

This example demonstrates the saturation behavior of a two-level atom and
the power broadening effect in atomic spectroscopy. We explore:
- Saturation intensity I_sat and saturation parameter s = I/I_sat
- How steady-state population depends on intensity
- Power broadening of the Lorentzian lineshape
- The transition from linear to saturated response

Key physics:
- At low intensity (s << 1): linear response, natural linewidth
- At high intensity (s >> 1): saturation, population approaches 1/2, line broadens
- Effective linewidth: gamma_eff = gamma * sqrt(1 + s)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.amo import TwoLevelAtom, BlochEquations, HBAR, C, EPSILON_0

def calculate_saturation_physics():
    """Calculate saturation and power broadening effects."""

    # Rubidium D2 line parameters
    wavelength = 780e-9  # 780 nm
    omega_0 = 2 * np.pi * C / wavelength
    gamma = 2 * np.pi * 6.07e6  # Natural linewidth (rad/s)
    d = 2.5e-29  # Dipole moment (C*m)

    # Create atom
    atom = TwoLevelAtom(omega_0=omega_0, d=d, gamma=gamma)
    bloch = BlochEquations(atom)

    # Calculate saturation intensity
    I_sat = atom.saturation_intensity()
    print(f"Saturation intensity: {I_sat:.2f} W/m^2 = {I_sat*1e-4:.2f} mW/cm^2")

    results = {}
    results['I_sat'] = I_sat
    results['gamma'] = gamma

    # 1. Population vs saturation parameter (on resonance)
    print("Computing saturation curve...")
    s_values = np.logspace(-2, 3, 200)  # s from 0.01 to 1000
    delta = 0  # On resonance

    # For each s, calculate Rabi frequency from s = 2*Omega^2/gamma^2
    # (on resonance, s = I/I_sat = 2*Omega^2/gamma^2)
    P_e_vs_s = []
    for s in s_values:
        Omega = gamma * np.sqrt(s / 2)
        P_e = bloch.steady_state_population(Omega, delta)
        P_e_vs_s.append(P_e)

    results['saturation'] = {
        's': s_values,
        'P_e': np.array(P_e_vs_s),
        'theoretical': (s_values / 2) / (1 + s_values)  # On resonance
    }

    # 2. Power broadening - lineshape at various intensities
    print("Computing power-broadened lineshapes...")
    delta_range = np.linspace(-20 * gamma, 20 * gamma, 400)
    s_values_lineshape = [0.1, 1, 10, 100]

    lineshapes = []
    for s in s_values_lineshape:
        Omega = gamma * np.sqrt(s / 2)
        P_e_delta = np.array([bloch.steady_state_population(Omega, d) for d in delta_range])
        lineshapes.append({
            's': s,
            'delta': delta_range,
            'P_e': P_e_delta,
            'P_e_normalized': P_e_delta / np.max(P_e_delta)
        })

    results['lineshapes'] = lineshapes

    # 3. Effective linewidth vs saturation parameter
    print("Computing effective linewidth...")
    s_range = np.logspace(-1, 2, 50)
    fwhm_natural = gamma  # Natural FWHM

    fwhm_power_broadened = []
    for s in s_range:
        Omega = gamma * np.sqrt(s / 2)
        P_e_profile = np.array([bloch.steady_state_population(Omega, d)
                               for d in np.linspace(-10*gamma, 10*gamma, 200)])
        max_P = np.max(P_e_profile)
        half_max = max_P / 2

        # Find FWHM numerically
        delta_grid = np.linspace(-10*gamma, 10*gamma, 200)
        above_half = delta_grid[P_e_profile >= half_max]
        if len(above_half) > 0:
            fwhm = above_half[-1] - above_half[0]
        else:
            fwhm = gamma

        fwhm_power_broadened.append(fwhm)

    results['linewidth'] = {
        's': s_range,
        'fwhm': np.array(fwhm_power_broadened) / gamma,
        'theoretical': np.sqrt(1 + s_range)  # gamma * sqrt(1 + s)
    }

    # 4. Scattering rate vs intensity
    print("Computing scattering rate...")
    # R_sc = gamma * P_e (on resonance, R_sc = gamma * s / (2 + 2s))
    scattering_rate = gamma * results['saturation']['P_e']

    results['scattering'] = {
        's': s_values,
        'rate': scattering_rate / gamma,  # Normalize to gamma
        'theoretical': s_values / (2 * (1 + s_values))  # = P_e (normalized)
    }

    return results


def plot_results(results):
    """Create comprehensive visualization of saturation and power broadening."""

    fig = plt.figure(figsize=(14, 12))
    gamma = results['gamma']
    I_sat = results['I_sat']

    # Plot 1: Population vs saturation parameter
    ax1 = fig.add_subplot(2, 2, 1)
    sat = results['saturation']
    ax1.semilogx(sat['s'], sat['P_e'], 'b-', linewidth=2, label='Numerical')
    ax1.semilogx(sat['s'], sat['theoretical'], 'r--', linewidth=1.5, label='Theory: $s/(2+2s)$')
    ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7, label='Saturation limit')
    ax1.axvline(x=1, color='green', linestyle=':', alpha=0.7, label='$s = 1$')

    ax1.set_xlabel('Saturation Parameter $s = I/I_{sat}$', fontsize=11)
    ax1.set_ylabel('Excited State Population $P_e$', fontsize=11)
    ax1.set_title('Saturation of Two-Level Atom (On Resonance)', fontsize=12)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.set_xlim(0.01, 1000)
    ax1.set_ylim(0, 0.55)
    ax1.grid(True, alpha=0.3)

    # Add inset showing low-intensity linear regime
    ax1_inset = ax1.inset_axes([0.15, 0.55, 0.35, 0.35])
    linear_mask = sat['s'] < 0.3
    ax1_inset.plot(sat['s'][linear_mask], sat['P_e'][linear_mask], 'b-', linewidth=2)
    ax1_inset.plot(sat['s'][linear_mask], sat['s'][linear_mask]/2, 'r--', linewidth=1.5)
    ax1_inset.set_xlabel('$s$', fontsize=8)
    ax1_inset.set_ylabel('$P_e$', fontsize=8)
    ax1_inset.set_title('Linear regime: $P_e \\approx s/2$', fontsize=8)
    ax1_inset.grid(True, alpha=0.3)

    # Plot 2: Power-broadened lineshapes
    ax2 = fig.add_subplot(2, 2, 2)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, ls in enumerate(results['lineshapes']):
        label = f'$s = {ls["s"]:.1f}$'
        ax2.plot(ls['delta'] / gamma, ls['P_e_normalized'],
                color=colors[i], linewidth=1.5, label=label)

    ax2.set_xlabel(r'Detuning $\delta/\gamma$', fontsize=11)
    ax2.set_ylabel('Normalized Population', fontsize=11)
    ax2.set_title('Power Broadening of Absorption Line', fontsize=12)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_xlim(-15, 15)
    ax2.grid(True, alpha=0.3)

    # Add FWHM annotations
    for i, ls in enumerate(results['lineshapes']):
        fwhm = gamma * np.sqrt(1 + ls['s'])
        ax2.annotate(f'FWHM={fwhm/gamma:.1f}$\\gamma$',
                    xy=(fwhm/(2*gamma), 0.5),
                    xytext=(fwhm/(2*gamma) + 2, 0.5 + 0.1*(3-i)),
                    color=colors[i], fontsize=8,
                    arrowprops=dict(arrowstyle='->', color=colors[i], alpha=0.5))

    # Plot 3: Effective linewidth vs saturation
    ax3 = fig.add_subplot(2, 2, 3)
    lw = results['linewidth']
    ax3.loglog(lw['s'], lw['fwhm'], 'bo', markersize=4, label='Numerical FWHM')
    ax3.loglog(lw['s'], lw['theoretical'], 'r-', linewidth=2, label=r'Theory: $\sqrt{1+s}$')

    ax3.set_xlabel('Saturation Parameter $s$', fontsize=11)
    ax3.set_ylabel(r'FWHM / $\gamma$', fontsize=11)
    ax3.set_title('Power Broadening of Linewidth', fontsize=12)
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3, which='both')

    # Add text box
    textstr = '\n'.join([
        'Power Broadening:',
        r'$\gamma_{eff} = \gamma\sqrt{1+s}$',
        '',
        'At $s=1$: 41% broadening',
        'At $s=10$: 3.3x broader',
        'At $s=100$: 10x broader'
    ])
    ax3.text(0.95, 0.05, textstr, transform=ax3.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 4: Scattering rate vs intensity
    ax4 = fig.add_subplot(2, 2, 4)
    sc = results['scattering']
    ax4.semilogx(sc['s'], sc['rate'], 'b-', linewidth=2, label=r'$R_{sc}/\gamma$')
    ax4.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7, label=r'Maximum: $\gamma/2$')

    # Show linear and saturated regimes
    ax4.fill_between(sc['s'][sc['s'] < 1], 0, sc['rate'][sc['s'] < 1],
                    alpha=0.2, color='green', label='Linear regime')
    ax4.fill_between(sc['s'][sc['s'] > 10], 0, sc['rate'][sc['s'] > 10],
                    alpha=0.2, color='red', label='Saturated regime')

    ax4.set_xlabel('Saturation Parameter $s = I/I_{sat}$', fontsize=11)
    ax4.set_ylabel(r'Scattering Rate $R_{sc}/\gamma$', fontsize=11)
    ax4.set_title('Photon Scattering Rate vs Intensity', fontsize=12)
    ax4.legend(loc='right', fontsize=9)
    ax4.set_xlim(0.01, 1000)
    ax4.set_ylim(0, 0.55)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    print("=" * 60)
    print("Experiment 247: Saturation Intensity and Power Broadening")
    print("=" * 60)
    print()

    # Run calculations
    print("Running calculations...")
    results = calculate_saturation_physics()

    # Create visualization
    print("\nCreating visualization...")
    fig = plot_results(results)

    # Save figure
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'saturation_power_broadening.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    # Print summary
    gamma = results['gamma']
    I_sat = results['I_sat']
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"Saturation intensity I_sat: {I_sat:.2f} W/m^2 ({I_sat*0.1:.2f} mW/cm^2)")
    print(f"Natural linewidth gamma: {gamma/(2*np.pi)/1e6:.2f} MHz")
    print()
    print("Key results:")
    print(f"  At s = 0.1: P_e = {0.1/2/(1+0.1):.3f}, FWHM = {np.sqrt(1.1):.2f} * gamma")
    print(f"  At s = 1:   P_e = {1/2/(1+1):.3f}, FWHM = {np.sqrt(2):.2f} * gamma")
    print(f"  At s = 10:  P_e = {10/2/(1+10):.3f}, FWHM = {np.sqrt(11):.2f} * gamma")
    print(f"  At s = 100: P_e = {100/2/(1+100):.3f}, FWHM = {np.sqrt(101):.2f} * gamma")

    plt.close()


if __name__ == "__main__":
    main()
