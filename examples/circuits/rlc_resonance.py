"""
Example demonstrating RLC circuit resonance.

This example shows the resonance behavior of a series RLC circuit,
including frequency response and quality factor.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def rlc_impedance(omega, R, L, C):
    """
    Calculate complex impedance of series RLC circuit.

    Z = R + j(ωL - 1/ωC)
    """
    return R + 1j * (omega * L - 1 / (omega * C))


def rlc_transfer_function(omega, R, L, C, V_in=1.0):
    """
    Calculate voltage across capacitor (voltage divider).

    H(ω) = V_C / V_in = (1/jωC) / Z
    """
    Z_C = 1 / (1j * omega * C)
    Z_total = rlc_impedance(omega, R, L, C)
    return V_in * Z_C / Z_total


def main():
    # Circuit parameters
    R = 100       # Resistance (Ohms)
    L = 10e-3     # Inductance (H) = 10 mH
    C = 1e-6      # Capacitance (F) = 1 μF

    # Resonant frequency
    omega_0 = 1 / np.sqrt(L * C)
    f_0 = omega_0 / (2 * np.pi)

    # Quality factor
    Q = omega_0 * L / R

    # Bandwidth
    bandwidth = omega_0 / Q

    # Frequency range
    f = np.logspace(2, 5, 1000)  # 100 Hz to 100 kHz
    omega = 2 * np.pi * f

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Magnitude response (Bode plot)
    ax1 = axes[0, 0]

    H = rlc_transfer_function(omega, R, L, C)
    H_mag = np.abs(H)
    H_db = 20 * np.log10(H_mag)

    ax1.semilogx(f, H_db, 'b-', lw=2)
    ax1.axvline(x=f_0, color='r', linestyle='--', label=f'f₀ = {f_0:.0f} Hz')
    ax1.axhline(y=H_db.max() - 3, color='g', linestyle=':', label='-3 dB bandwidth')

    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Magnitude (dB)')
    ax1.set_title('Frequency Response (Magnitude)')
    ax1.legend()
    ax1.grid(True, alpha=0.3, which='both')

    # Plot 2: Phase response
    ax2 = axes[0, 1]

    H_phase = np.angle(H, deg=True)

    ax2.semilogx(f, H_phase, 'b-', lw=2)
    ax2.axvline(x=f_0, color='r', linestyle='--', label=f'f₀ = {f_0:.0f} Hz')
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax2.axhline(y=-90, color='gray', linestyle=':', alpha=0.5)

    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Phase (degrees)')
    ax2.set_title('Frequency Response (Phase)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_ylim(-180, 90)

    # Plot 3: Effect of Q factor (different R values)
    ax3 = axes[1, 0]

    R_values = [50, 100, 200, 500]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(R_values)))

    for R_test, color in zip(R_values, colors):
        Q_test = omega_0 * L / R_test
        H_test = rlc_transfer_function(omega, R_test, L, C)
        H_mag_test = np.abs(H_test)
        ax3.semilogx(f/f_0, H_mag_test, color=color, lw=2,
                    label=f'R = {R_test}Ω, Q = {Q_test:.1f}')

    ax3.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Normalized Frequency (f/f₀)')
    ax3.set_ylabel('|H(f)|')
    ax3.set_title('Effect of Resistance on Q Factor')
    ax3.legend()
    ax3.grid(True, alpha=0.3, which='both')

    # Plot 4: Impedance components
    ax4 = axes[1, 1]

    Z_R = R * np.ones_like(omega)
    Z_L = omega * L
    Z_C = 1 / (omega * C)
    Z_total = np.abs(rlc_impedance(omega, R, L, C))

    ax4.loglog(f, Z_R, 'r-', lw=2, label='R')
    ax4.loglog(f, Z_L, 'b--', lw=2, label='ωL')
    ax4.loglog(f, Z_C, 'g-.', lw=2, label='1/ωC')
    ax4.loglog(f, Z_total, 'k-', lw=2, label='|Z|')

    ax4.axvline(x=f_0, color='purple', linestyle=':', alpha=0.7)

    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Impedance (Ω)')
    ax4.set_title('Impedance Components')
    ax4.legend()
    ax4.grid(True, alpha=0.3, which='both')

    # Add parameter annotations
    info_text = (f'R = {R} Ω\n'
                 f'L = {L*1000:.0f} mH\n'
                 f'C = {C*1e6:.0f} μF\n'
                 f'f₀ = {f_0:.0f} Hz\n'
                 f'Q = {Q:.1f}\n'
                 f'BW = {bandwidth/(2*np.pi):.0f} Hz')

    ax4.text(0.02, 0.98, info_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Series RLC Circuit Resonance\n'
                 f'ω₀ = 1/√(LC), Q = ω₀L/R = 1/(ω₀RC)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'rlc_resonance.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'rlc_resonance.png')}")


if __name__ == "__main__":
    main()
