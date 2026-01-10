"""
Experiment 94: Transmission line reflections.

This example demonstrates wave propagation on transmission lines,
including impedance matching and reflection coefficients.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt


def reflection_coefficient(Z_L, Z_0):
    """
    Calculate reflection coefficient at a load.

    Gamma = (Z_L - Z_0) / (Z_L + Z_0)

    Args:
        Z_L: Load impedance (complex, Ohms)
        Z_0: Characteristic impedance (real, Ohms)

    Returns:
        Gamma: Reflection coefficient (complex)
    """
    return (Z_L - Z_0) / (Z_L + Z_0)


def transmission_coefficient(Z_L, Z_0):
    """
    Calculate transmission coefficient.

    T = 2 * Z_L / (Z_L + Z_0)
    """
    return 2 * Z_L / (Z_L + Z_0)


def simulate_transmission_line(nx, nt, Z_0, Z_L, v, dx, dt, pulse_width):
    """
    Simulate wave propagation on a transmission line using FDTD.

    Args:
        nx: Number of spatial points
        nt: Number of time steps
        Z_0: Characteristic impedance
        Z_L: Load impedance
        v: Wave velocity
        dx: Spatial step
        dt: Time step
        pulse_width: Width of input pulse

    Returns:
        V: Voltage history (nt, nx)
        I: Current history (nt, nx)
    """
    # Transmission line parameters
    L = Z_0 / v   # Inductance per unit length
    C = 1 / (Z_0 * v)  # Capacitance per unit length

    # Initialize
    V = np.zeros((nt, nx))
    I = np.zeros((nt, nx))

    # Courant number
    c = v * dt / dx

    for n in range(nt - 1):
        # Source: Gaussian pulse at left end
        t = n * dt
        V[n, 0] = np.exp(-((t - 3*pulse_width) / pulse_width)**2)

        # Update interior voltage
        for i in range(1, nx):
            V[n+1, i] = V[n, i] - c * Z_0 * (I[n, i] - I[n, i-1])

        # Update interior current
        for i in range(nx - 1):
            I[n+1, i] = I[n, i] - c / Z_0 * (V[n+1, i+1] - V[n+1, i])

        # Load boundary condition
        Gamma = (Z_L - Z_0) / (Z_L + Z_0)
        V[n+1, -1] = V[n, -1] + (1 + Gamma) * c * Z_0 * I[n, -2] - c * Z_0 * I[n, -2]

    return V, I


def voltage_along_line(z, Z_L, Z_0, beta, length, V_plus=1):
    """
    Calculate voltage phasor along transmission line.

    V(z) = V+ * (e^(-j*beta*z) + Gamma * e^(j*beta*z))

    where z is measured from the load.
    """
    Gamma = reflection_coefficient(Z_L, Z_0)
    d = length - z  # Distance from load
    return V_plus * (np.exp(-1j * beta * z) + Gamma * np.exp(-1j * beta * (2*length - z)))


def standing_wave_pattern(d, Gamma, beta):
    """
    Calculate VSWR pattern (voltage magnitude vs distance from load).

    |V(d)| = |V+| * |1 + Gamma * e^(-2j*beta*d)|
    """
    return np.abs(1 + Gamma * np.exp(-2j * beta * d))


def main():
    # Transmission line parameters
    Z_0 = 50  # 50 Ohm characteristic impedance (typical coax)

    fig = plt.figure(figsize=(16, 12))

    # Plot 1: Reflection coefficient vs load impedance
    ax1 = fig.add_subplot(2, 2, 1)

    Z_L_real = np.linspace(0, 200, 100)

    Gamma_vals = reflection_coefficient(Z_L_real, Z_0)

    ax1.plot(Z_L_real, np.abs(Gamma_vals), 'b-', lw=2, label='|Gamma|')
    ax1.axvline(x=Z_0, color='green', linestyle='--', lw=2, label=f'Z_L = Z_0 = {Z_0} ohm (matched)')
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    # Mark special cases
    ax1.plot(0, 1, 'ro', markersize=10, label='Short circuit (Gamma = -1)')
    ax1.plot(200, abs(reflection_coefficient(200, Z_0)), 'rs', markersize=10)

    ax1.set_xlabel('Load Impedance Z_L (Ohms)')
    ax1.set_ylabel('|Reflection Coefficient|')
    ax1.set_title('Reflection Coefficient vs Load Impedance\n(Real loads only)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Standing wave patterns
    ax2 = fig.add_subplot(2, 2, 2)

    wavelength = 1  # Normalized wavelength
    beta = 2 * np.pi / wavelength
    d = np.linspace(0, 2 * wavelength, 500)

    load_conditions = [
        ('Matched (Z_L = Z_0)', Z_0),
        ('Open circuit (Z_L = inf)', 1e10),
        ('Short circuit (Z_L = 0)', 1e-10),
        ('Z_L = 2*Z_0', 2*Z_0),
        ('Z_L = Z_0/2', Z_0/2),
    ]

    colors = plt.cm.tab10(np.linspace(0, 0.5, len(load_conditions)))

    for (name, Z_L), color in zip(load_conditions, colors):
        Gamma = reflection_coefficient(Z_L, Z_0)
        pattern = standing_wave_pattern(d, Gamma, beta)
        VSWR = (1 + np.abs(Gamma)) / (1 - np.abs(Gamma)) if np.abs(Gamma) < 1 else float('inf')
        ax2.plot(d / wavelength, pattern, color=color, lw=2,
                label=f'{name}, VSWR={VSWR:.2f}')

    ax2.set_xlabel('Distance from load (wavelengths)')
    ax2.set_ylabel('Normalized |V|')
    ax2.set_title('Standing Wave Patterns')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Time-domain pulse reflection simulation
    ax3 = fig.add_subplot(2, 2, 3)

    # Simulation parameters
    nx = 200
    nt = 400
    v = 2e8  # Wave velocity (m/s)
    line_length = 10  # 10 m line
    dx = line_length / nx
    dt = 0.8 * dx / v  # CFL condition
    pulse_width = 5 * dt

    # Different load conditions
    load_tests = [
        ('Matched', Z_0),
        ('Open', 1e6),
        ('Short', 1e-6),
    ]

    x = np.linspace(0, line_length, nx)
    t_plot = [50, 100, 150, 200, 250]  # Time steps to plot

    for idx, (name, Z_L) in enumerate(load_tests):
        V, I = simulate_transmission_line(nx, nt, Z_0, Z_L, v, dx, dt, pulse_width)

        # Plot voltage at selected times
        for t_idx in t_plot:
            alpha = 0.3 + 0.7 * t_idx / max(t_plot)
            ax3.plot(x, V[t_idx, :] + 3*idx, color=f'C{idx}', alpha=alpha, lw=1)

        ax3.axhline(y=3*idx, color='gray', linestyle=':', alpha=0.3)
        ax3.text(-0.5, 3*idx + 0.5, name, fontsize=10, va='center')

    ax3.set_xlabel('Position along line (m)')
    ax3.set_ylabel('Voltage (offset for clarity)')
    ax3.set_title('Pulse Propagation and Reflection')
    ax3.set_xlim(-1, line_length + 1)
    ax3.grid(True, alpha=0.3)

    # Plot 4: VSWR and power reflection
    ax4 = fig.add_subplot(2, 2, 4)

    Z_L_range = np.logspace(-1, 2, 100) * Z_0

    Gamma_arr = reflection_coefficient(Z_L_range, Z_0)
    VSWR_arr = (1 + np.abs(Gamma_arr)) / (1 - np.abs(Gamma_arr))
    Power_reflected = np.abs(Gamma_arr)**2 * 100  # Percentage

    ax4.semilogx(Z_L_range / Z_0, VSWR_arr, 'b-', lw=2, label='VSWR')
    ax4_twin = ax4.twinx()
    ax4_twin.semilogx(Z_L_range / Z_0, Power_reflected, 'r--', lw=2, label='Power reflected (%)')

    ax4.axvline(x=1, color='green', linestyle=':', lw=2, label='Z_L = Z_0')

    ax4.set_xlabel('Normalized Load Impedance Z_L / Z_0')
    ax4.set_ylabel('VSWR', color='b')
    ax4_twin.set_ylabel('Power Reflected (%)', color='r')
    ax4.tick_params(axis='y', labelcolor='b')
    ax4_twin.tick_params(axis='y', labelcolor='r')

    ax4.set_title('VSWR and Power Reflection')
    ax4.set_ylim(1, 10)
    ax4_twin.set_ylim(0, 100)

    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax4.grid(True, alpha=0.3)

    # Add physics summary
    fig.text(0.5, 0.02,
             r'Transmission Line: $\Gamma = \frac{Z_L - Z_0}{Z_L + Z_0}$, '
             r'VSWR = $\frac{1 + |\Gamma|}{1 - |\Gamma|}$, '
             r'Power reflected = $|\Gamma|^2$',
             ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.suptitle(f'Transmission Line Reflections (Z_0 = {Z_0} ohm)', fontsize=14, y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)

    # Save plot
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'transmission_line_reflections.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
