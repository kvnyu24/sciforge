"""
Example demonstrating RC circuit transient response.

This example shows charging and discharging of a capacitor through
a resistor, including the time constant and energy storage.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def rc_charging(t, V0, R, C):
    """
    Voltage across capacitor during charging.

    V(t) = V0 * (1 - e^(-t/RC))

    Args:
        t: Time array
        V0: Source voltage
        R: Resistance (Ohms)
        C: Capacitance (Farads)

    Returns:
        Capacitor voltage
    """
    tau = R * C
    return V0 * (1 - np.exp(-t / tau))


def rc_discharging(t, V0, R, C):
    """
    Voltage across capacitor during discharging.

    V(t) = V0 * e^(-t/RC)
    """
    tau = R * C
    return V0 * np.exp(-t / tau)


def main():
    # Circuit parameters
    V0 = 10.0      # Source voltage (V)
    R = 1000       # Resistance (Ohms) = 1 kΩ
    C = 100e-6     # Capacitance (F) = 100 μF
    tau = R * C    # Time constant (s)

    # Time array (5 time constants)
    t = np.linspace(0, 5 * tau, 1000)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Charging curve
    ax1 = axes[0, 0]

    V_charge = rc_charging(t, V0, R, C)
    ax1.plot(t * 1000, V_charge, 'b-', lw=2, label='V_C(t)')
    ax1.axhline(y=V0, color='gray', linestyle='--', label=f'V₀ = {V0}V')
    ax1.axhline(y=V0 * (1 - np.exp(-1)), color='r', linestyle=':', alpha=0.7)
    ax1.axvline(x=tau * 1000, color='r', linestyle=':', alpha=0.7)

    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Capacitor Voltage (V)')
    ax1.set_title('Capacitor Charging')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax1.annotate(f'τ = RC = {tau*1000:.1f} ms', xy=(tau * 1000, V0 * 0.632),
                xytext=(tau * 1000 + 20, V0 * 0.5),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10)

    # Plot 2: Discharging curve
    ax2 = axes[0, 1]

    V_discharge = rc_discharging(t, V0, R, C)
    ax2.plot(t * 1000, V_discharge, 'r-', lw=2, label='V_C(t)')
    ax2.axhline(y=V0 * np.exp(-1), color='b', linestyle=':', alpha=0.7)
    ax2.axvline(x=tau * 1000, color='b', linestyle=':', alpha=0.7)

    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Capacitor Voltage (V)')
    ax2.set_title('Capacitor Discharging')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax2.annotate(f'V(τ) = V₀/e ≈ 0.37V₀', xy=(tau * 1000, V0 * np.exp(-1)),
                xytext=(tau * 1000 + 50, V0 * 0.5),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=10)

    # Plot 3: Current during charging
    ax3 = axes[0, 2]

    I_charge = (V0 / R) * np.exp(-t / tau)
    ax3.plot(t * 1000, I_charge * 1000, 'g-', lw=2, label='I(t)')
    ax3.axhline(y=V0 / R * 1000, color='gray', linestyle='--',
               label=f'I₀ = V₀/R = {V0/R*1000:.0f} mA')

    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Current (mA)')
    ax3.set_title('Current During Charging')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Energy storage
    ax4 = axes[1, 0]

    # Energy stored in capacitor
    E_capacitor = 0.5 * C * V_charge**2

    # Energy dissipated in resistor
    E_resistor = 0.5 * C * V0**2 - E_capacitor

    # Total energy from source
    E_source = C * V0 * V_charge

    ax4.plot(t * 1000, E_capacitor * 1e6, 'b-', lw=2, label='Energy in capacitor')
    ax4.plot(t * 1000, E_resistor * 1e6, 'r-', lw=2, label='Energy dissipated in resistor')
    ax4.plot(t * 1000, E_source * 1e6, 'g--', lw=2, label='Energy from source')

    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('Energy (μJ)')
    ax4.set_title('Energy Distribution During Charging')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Different time constants
    ax5 = axes[1, 1]

    R_values = [500, 1000, 2000, 5000]  # Ohms
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(R_values)))

    for R_test, color in zip(R_values, colors):
        tau_test = R_test * C
        V_test = rc_charging(t, V0, R_test, C)
        ax5.plot(t * 1000, V_test, color=color, lw=2,
                label=f'R = {R_test/1000:.1f} kΩ, τ = {tau_test*1000:.0f} ms')

    ax5.axhline(y=V0, color='gray', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Time (ms)')
    ax5.set_ylabel('Capacitor Voltage (V)')
    ax5.set_title('Effect of Resistance on Charging Speed')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # Plot 6: Step response (square wave input)
    ax6 = axes[1, 2]

    # Create square wave input
    t_extended = np.linspace(0, 10 * tau, 2000)
    period = 4 * tau
    V_input = V0 * (((t_extended % period) < period/2).astype(float))

    # Simulate RC response
    dt = t_extended[1] - t_extended[0]
    V_output = np.zeros_like(t_extended)

    for i in range(1, len(t_extended)):
        dV = (V_input[i-1] - V_output[i-1]) / tau * dt
        V_output[i] = V_output[i-1] + dV

    ax6.plot(t_extended * 1000, V_input, 'b-', lw=1, alpha=0.5, label='Input (square wave)')
    ax6.plot(t_extended * 1000, V_output, 'r-', lw=2, label='Output (RC filtered)')

    ax6.set_xlabel('Time (ms)')
    ax6.set_ylabel('Voltage (V)')
    ax6.set_title('RC Low-Pass Filter Response')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.suptitle(f'RC Circuit Analysis (R = {R/1000:.1f} kΩ, C = {C*1e6:.0f} μF, τ = {tau*1000:.1f} ms)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'rc_circuit.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'rc_circuit.png')}")


if __name__ == "__main__":
    main()
