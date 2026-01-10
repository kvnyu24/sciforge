"""
Experiment 237: Josephson Junction I-Phase Relationship

Demonstrates the Josephson effect, showing the current-phase relationship
I = I_c * sin(phi) and the dynamics of driven Josephson junctions,
including Shapiro steps and the RSJ model.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# Physical constants
Phi_0 = 2.067e-15  # Magnetic flux quantum (Wb)
hbar = 1.055e-34   # Reduced Planck constant
e = 1.602e-19      # Electron charge


def josephson_dc_current(phi, I_c):
    """
    DC Josephson current.

    I = I_c * sin(phi)

    Args:
        phi: Phase difference
        I_c: Critical current

    Returns:
        Supercurrent I
    """
    return I_c * np.sin(phi)


def josephson_ac_voltage(dphi_dt):
    """
    AC Josephson voltage.

    V = (hbar / 2e) * dphi/dt = (Phi_0 / 2pi) * dphi/dt

    Args:
        dphi_dt: Time derivative of phase

    Returns:
        Voltage V
    """
    return Phi_0 / (2 * np.pi) * dphi_dt


def rsj_model(y, t, I_bias, I_c, R, C, omega_rf=0, I_rf=0):
    """
    RSJ (Resistively Shunted Junction) model equations.

    I_bias = I_c*sin(phi) + V/R + C*dV/dt + I_rf*cos(omega_rf*t)

    where V = (Phi_0/2pi) * dphi/dt

    Args:
        y: State [phi, dphi/dt]
        t: Time
        I_bias: DC bias current
        I_c: Critical current
        R: Shunt resistance
        C: Junction capacitance
        omega_rf: RF drive frequency
        I_rf: RF drive amplitude

    Returns:
        [dphi/dt, d2phi/dt2]
    """
    phi, dphi_dt = y

    # Voltage
    V = Phi_0 / (2 * np.pi) * dphi_dt

    # Phase dynamics
    tau_RC = R * C if C > 0 else 1e-15
    omega_p = np.sqrt(2 * np.pi * I_c / (Phi_0 * C)) if C > 0 else 1

    # Include RF drive
    I_total = I_bias + I_rf * np.cos(omega_rf * t)

    if C > 0:
        # Full RSJ model with capacitance
        d2phi_dt2 = (2 * np.pi / Phi_0) * (
            (I_total - I_c * np.sin(phi)) * R - V
        ) / (R * C)
    else:
        # Overdamped limit (RC << 1)
        d2phi_dt2 = 0

    return [dphi_dt, d2phi_dt2]


def rsj_overdamped(I_bias, I_c, R):
    """
    Overdamped RSJ model: Time-averaged voltage.

    For I > I_c: <V> = R * sqrt(I^2 - I_c^2)

    Args:
        I_bias: Bias current (array)
        I_c: Critical current
        R: Shunt resistance

    Returns:
        Average voltage
    """
    V = np.zeros_like(I_bias)
    mask = np.abs(I_bias) > I_c
    V[mask] = R * np.sqrt(I_bias[mask]**2 - I_c**2) * np.sign(I_bias[mask])
    return V


def shapiro_step_voltage(n, omega_rf):
    """
    Shapiro step voltage.

    V_n = n * (hbar * omega_rf) / (2e) = n * Phi_0 * omega_rf / (2pi)

    Args:
        n: Step number (integer)
        omega_rf: RF drive frequency

    Returns:
        Step voltage
    """
    return n * Phi_0 * omega_rf / (2 * np.pi)


def main():
    """Main simulation and visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Junction parameters
    I_c = 1e-6      # Critical current (1 uA)
    R = 10          # Shunt resistance (10 Ohm)
    C = 1e-12       # Capacitance (1 pF)

    # Plot 1: Current-Phase Relationship
    ax1 = axes[0, 0]

    phi = np.linspace(-2*np.pi, 2*np.pi, 500)
    I = josephson_dc_current(phi, I_c)

    ax1.plot(phi / np.pi, I / I_c, 'b-', lw=2)
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

    # Mark critical current
    ax1.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='I_c')
    ax1.axhline(y=-1, color='red', linestyle='--', alpha=0.5)

    ax1.set_xlabel('Phase difference phi / pi')
    ax1.set_ylabel('Current I / I_c')
    ax1.set_title('DC Josephson Effect: I = I_c sin(phi)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-1.5, 1.5)

    # Plot 2: IV Characteristic (RSJ model)
    ax2 = axes[0, 1]

    I_bias_range = np.linspace(-3*I_c, 3*I_c, 500)
    V_avg = rsj_overdamped(I_bias_range, I_c, R)

    ax2.plot(V_avg * 1e6, I_bias_range / I_c, 'b-', lw=2, label='RSJ model')

    # Normal state resistance line
    V_normal = I_bias_range * R
    ax2.plot(V_normal * 1e6, I_bias_range / I_c, 'r--', lw=1.5, alpha=0.5,
            label='Normal state (Ohmic)')

    ax2.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
    ax2.axhline(y=-1, color='gray', linestyle=':', alpha=0.5)

    ax2.set_xlabel('Voltage (uV)')
    ax2.set_ylabel('Current I / I_c')
    ax2.set_title('Overdamped RSJ I-V Characteristic')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Time dynamics under constant bias
    ax3 = axes[1, 0]

    # Time in units of 1/omega_p
    omega_p = np.sqrt(2 * np.pi * I_c / (Phi_0 * C))
    t_end = 100 / omega_p
    t = np.linspace(0, t_end, 2000)

    I_bias_values = [0.5 * I_c, 1.5 * I_c, 3.0 * I_c]
    colors = ['blue', 'green', 'red']

    for I_b, color in zip(I_bias_values, colors):
        # Solve RSJ equations
        y0 = [0, 0]  # Initial [phi, dphi/dt]
        sol = odeint(rsj_model, y0, t, args=(I_b, I_c, R, C))

        phi_t = sol[:, 0]
        V_t = Phi_0 / (2 * np.pi) * sol[:, 1]

        ax3.plot(t * omega_p, V_t * 1e6, color=color, lw=1.5, alpha=0.8,
                label=f'I/I_c = {I_b/I_c:.1f}')

    ax3.set_xlabel('Time (1/omega_p)')
    ax3.set_ylabel('Voltage (uV)')
    ax3.set_title('Voltage Oscillations (AC Josephson Effect)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Shapiro steps
    ax4 = axes[1, 1]

    # RF drive parameters
    f_rf = 10e9  # 10 GHz
    omega_rf = 2 * np.pi * f_rf
    I_rf = 0.5 * I_c

    # Calculate expected Shapiro step voltages
    V_steps = [shapiro_step_voltage(n, omega_rf) for n in range(-3, 4)]

    # Sweep bias current and measure time-averaged voltage
    I_bias_sweep = np.linspace(-2*I_c, 2*I_c, 200)
    V_avg_rf = []

    for I_b in I_bias_sweep:
        # Shorter simulation for averaging
        t_rf = np.linspace(0, 100 / omega_rf, 5000)
        y0 = [0, 0]
        sol = odeint(rsj_model, y0, t_rf, args=(I_b, I_c, R, C, omega_rf, I_rf))

        # Time-average voltage (skip transient)
        V_t = Phi_0 / (2 * np.pi) * sol[len(t_rf)//2:, 1]
        V_avg_rf.append(np.mean(V_t))

    V_avg_rf = np.array(V_avg_rf)

    ax4.plot(V_avg_rf * 1e6, I_bias_sweep / I_c, 'b-', lw=2)

    # Mark Shapiro steps
    for n, V_n in enumerate(V_steps):
        if n < len(V_steps) // 2:
            continue
        ax4.axvline(x=V_n * 1e6, color='red', linestyle='--', alpha=0.5)
        ax4.text(V_n * 1e6 + 1, 1.5, f'n={n - len(V_steps)//2}', fontsize=9, rotation=90)

    ax4.set_xlabel('Average Voltage (uV)')
    ax4.set_ylabel('Current I / I_c')
    ax4.set_title(f'Shapiro Steps (f_rf = {f_rf/1e9:.0f} GHz)')
    ax4.grid(True, alpha=0.3)

    ax4.text(0.05, 0.95, f'$V_n = n h f_{{rf}} / 2e$\n$= n \\times {V_steps[4]*1e6:.2f}$ uV',
             transform=ax4.transAxes, fontsize=10, va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle('Josephson Junction Physics\n'
                 'DC and AC Josephson effects, RSJ model, Shapiro steps',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'josephson_junction.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'josephson_junction.png')}")


if __name__ == "__main__":
    main()
