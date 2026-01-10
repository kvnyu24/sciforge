"""
Experiment 90: RL transient.

This example demonstrates the transient response of an RL (resistor-inductor)
circuit, including the exponential growth and decay of current.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt


def rl_current_growth(t, V0, R, L):
    """
    Current in RL circuit when voltage is applied.

    I(t) = (V0/R) * (1 - e^(-t*R/L))

    Args:
        t: Time array
        V0: Applied voltage (V)
        R: Resistance (Ohms)
        L: Inductance (H)

    Returns:
        Current (A)
    """
    tau = L / R  # Time constant
    I_max = V0 / R  # Steady-state current
    return I_max * (1 - np.exp(-t / tau))


def rl_current_decay(t, I0, R, L):
    """
    Current decay in RL circuit after voltage is removed.

    I(t) = I0 * e^(-t*R/L)
    """
    tau = L / R
    return I0 * np.exp(-t / tau)


def inductor_voltage_growth(t, V0, R, L):
    """
    Voltage across inductor during current growth.

    V_L(t) = V0 * e^(-t*R/L)
    """
    tau = L / R
    return V0 * np.exp(-t / tau)


def resistor_voltage_growth(t, V0, R, L):
    """
    Voltage across resistor during current growth.

    V_R(t) = V0 * (1 - e^(-t*R/L))
    """
    tau = L / R
    return V0 * (1 - np.exp(-t / tau))


def energy_in_inductor(I, L):
    """Energy stored in inductor: E = (1/2)*L*I^2"""
    return 0.5 * L * I**2


def main():
    # Circuit parameters
    V0 = 10.0      # Source voltage (V)
    R = 100        # Resistance (Ohms)
    L = 0.5        # Inductance (H) = 500 mH
    tau = L / R    # Time constant (s)
    I_max = V0 / R  # Steady-state current (A)

    # Time array (5 time constants)
    t = np.linspace(0, 5 * tau, 1000)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Current growth
    ax1 = axes[0, 0]

    I_growth = rl_current_growth(t, V0, R, L)
    ax1.plot(t * 1000, I_growth * 1000, 'b-', lw=2, label='I(t)')
    ax1.axhline(y=I_max * 1000, color='gray', linestyle='--',
                label=f'I_max = V/R = {I_max*1000:.0f} mA')
    ax1.axhline(y=I_max * 1000 * (1 - np.exp(-1)), color='r', linestyle=':', alpha=0.7)
    ax1.axvline(x=tau * 1000, color='r', linestyle=':', alpha=0.7)

    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Current (mA)')
    ax1.set_title('RL Circuit: Current Growth')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax1.annotate(f'tau = L/R = {tau*1000:.1f} ms', xy=(tau * 1000, I_max * 1000 * 0.632),
                xytext=(tau * 1000 + 5, I_max * 1000 * 0.4),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10)

    # Plot 2: Current decay
    ax2 = axes[0, 1]

    I_decay = rl_current_decay(t, I_max, R, L)
    ax2.plot(t * 1000, I_decay * 1000, 'r-', lw=2, label='I(t)')
    ax2.axhline(y=I_max * 1000 * np.exp(-1), color='b', linestyle=':', alpha=0.7)
    ax2.axvline(x=tau * 1000, color='b', linestyle=':', alpha=0.7)

    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Current (mA)')
    ax2.set_title('RL Circuit: Current Decay')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax2.annotate(f'I(tau) = I0/e = {I_max*1000*np.exp(-1):.1f} mA',
                xy=(tau * 1000, I_max * 1000 * np.exp(-1)),
                xytext=(tau * 1000 + 5, I_max * 1000 * 0.6),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=10)

    # Plot 3: Voltage distribution
    ax3 = axes[0, 2]

    V_L = inductor_voltage_growth(t, V0, R, L)
    V_R = resistor_voltage_growth(t, V0, R, L)

    ax3.plot(t * 1000, V_L, 'b-', lw=2, label='V_L (inductor)')
    ax3.plot(t * 1000, V_R, 'r-', lw=2, label='V_R (resistor)')
    ax3.plot(t * 1000, V_L + V_R, 'k--', lw=1, label='V_L + V_R = V0')

    ax3.axhline(y=V0, color='gray', linestyle=':', alpha=0.5)
    ax3.axvline(x=tau * 1000, color='gray', linestyle=':', alpha=0.5)

    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Voltage (V)')
    ax3.set_title('Voltage Distribution During Growth')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Energy storage
    ax4 = axes[1, 0]

    E_L = energy_in_inductor(I_growth, L)
    E_max = energy_in_inductor(I_max, L)

    ax4.plot(t * 1000, E_L * 1000, 'g-', lw=2, label='Energy in L')
    ax4.axhline(y=E_max * 1000, color='gray', linestyle='--',
                label=f'E_max = (1/2)LI^2 = {E_max*1000:.2f} mJ')

    # Energy dissipated in resistor
    E_R = E_max - E_L  # From conservation (simplified)
    ax4.plot(t * 1000, E_R * 1000, 'r-', lw=2, label='Energy dissipated in R')

    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('Energy (mJ)')
    ax4.set_title('Energy Storage in Inductor')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Different time constants
    ax5 = axes[1, 1]

    L_values = [0.1, 0.25, 0.5, 1.0]  # H
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(L_values)))

    for L_test, color in zip(L_values, colors):
        tau_test = L_test / R
        I_test = rl_current_growth(t, V0, R, L_test)
        ax5.plot(t * 1000, I_test * 1000, color=color, lw=2,
                label=f'L = {L_test*1000:.0f} mH, tau = {tau_test*1000:.1f} ms')

    ax5.axhline(y=I_max * 1000, color='gray', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Time (ms)')
    ax5.set_ylabel('Current (mA)')
    ax5.set_title('Effect of Inductance on Rise Time')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # Plot 6: Step response (square wave input)
    ax6 = axes[1, 2]

    # Create square wave input
    t_extended = np.linspace(0, 10 * tau, 2000)
    period = 4 * tau
    V_input = V0 * (((t_extended % period) < period/2).astype(float))

    # Simulate RL response
    dt = t_extended[1] - t_extended[0]
    I_output = np.zeros_like(t_extended)

    for i in range(1, len(t_extended)):
        # di/dt = (V - IR) / L
        dI = (V_input[i-1] - I_output[i-1] * R) / L * dt
        I_output[i] = I_output[i-1] + dI

    ax6.plot(t_extended * 1000, V_input, 'b-', lw=1, alpha=0.5, label='Input voltage')
    ax6.plot(t_extended * 1000, I_output * R, 'r-', lw=2, label='V_R = I*R')

    ax6.set_xlabel('Time (ms)')
    ax6.set_ylabel('Voltage (V)')
    ax6.set_title('RL Response to Square Wave')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.suptitle(f'RL Circuit Transient Analysis\n'
                 f'R = {R} ohm, L = {L*1000:.0f} mH, tau = L/R = {tau*1000:.1f} ms',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'rl_transient.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
