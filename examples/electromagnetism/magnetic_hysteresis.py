"""
Experiment 100: Magnetic hysteresis.

This example demonstrates magnetic hysteresis in ferromagnetic materials,
showing the B-H loop, coercivity, remanence, and energy loss per cycle.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt


# Physical constants
MU_0 = 4 * np.pi * 1e-7  # Permeability of free space (H/m)


def jiles_atherton_model(H, H_prev, M_prev, params):
    """
    Simplified Jiles-Atherton model for hysteresis.

    This model captures the key physics of domain wall motion
    and domain rotation in ferromagnetic materials.

    Args:
        H: Applied field (A/m)
        H_prev: Previous field value
        M_prev: Previous magnetization
        params: Dictionary of material parameters

    Returns:
        M: Magnetization (A/m)
    """
    Ms = params['Ms']      # Saturation magnetization
    a = params['a']        # Domain wall density parameter
    k = params['k']        # Pinning coefficient
    c = params['c']        # Reversible coefficient
    alpha = params['alpha']  # Domain coupling

    # Effective field
    He = H + alpha * M_prev

    # Anhysteretic magnetization (Langevin function approximation)
    if abs(He / a) < 0.01:
        Man = Ms * He / (3 * a)
    else:
        Man = Ms * (1/np.tanh(He/a) - a/He)

    # Direction of field change
    delta = 1 if H > H_prev else -1

    # Irreversible magnetization change
    dM_irr = (Man - M_prev) / (k * delta - alpha * (Man - M_prev))

    # Total magnetization change (simplified)
    dH = H - H_prev
    if abs(dH) < 1e-10:
        dM = 0
    else:
        dM = (1 - c) * dM_irr * dH + c * (Man - M_prev) * dH / (H + 1e-10)

    M = M_prev + dM

    # Clamp to saturation
    M = np.clip(M, -Ms, Ms)

    return M


def simple_hysteresis_loop(H_max, n_points, params):
    """
    Generate a complete hysteresis loop using simplified model.

    Args:
        H_max: Maximum applied field (A/m)
        n_points: Number of points per branch
        params: Material parameters

    Returns:
        H_loop: Field values
        B_loop: Flux density values
    """
    Ms = params['Ms']
    Hc = params['Hc']  # Coercivity
    Br = params['Br']  # Remanence

    # Create smooth hysteresis using tanh model
    H_up = np.linspace(-H_max, H_max, n_points)
    H_down = np.linspace(H_max, -H_max, n_points)

    def hysteresis_branch(H, direction):
        """
        direction: +1 for increasing H, -1 for decreasing
        """
        # Shifted tanh to create hysteresis
        H_shift = Hc * direction
        M = Ms * np.tanh((H - H_shift) / (Hc * 0.5))
        B = MU_0 * (H + M)
        return B

    B_up = hysteresis_branch(H_up, +1)
    B_down = hysteresis_branch(H_down, -1)

    H_loop = np.concatenate([H_up, H_down])
    B_loop = np.concatenate([B_up, B_down])

    return H_loop, B_loop


def calculate_hysteresis_loss(H_loop, B_loop):
    """
    Calculate energy loss per hysteresis cycle.

    Energy loss = integral(H dB) over closed loop
    """
    # Numerical integration using trapezoidal rule
    loss = 0
    for i in range(len(H_loop) - 1):
        dB = B_loop[i+1] - B_loop[i]
        H_avg = (H_loop[i] + H_loop[i+1]) / 2
        loss += H_avg * dB

    return abs(loss)


def main():
    # Material parameters for different ferromagnets
    materials = {
        'Soft Iron': {
            'Ms': 1.7e6,      # Saturation magnetization (A/m)
            'Hc': 100,        # Coercivity (A/m)
            'Br': 1.0,        # Remanence (T)
            'color': 'blue'
        },
        'Hard Steel': {
            'Ms': 1.4e6,
            'Hc': 5000,
            'Br': 0.9,
            'color': 'red'
        },
        'Permalloy': {
            'Ms': 0.8e6,
            'Hc': 4,
            'Br': 0.5,
            'color': 'green'
        },
        'Alnico': {
            'Ms': 1.0e6,
            'Hc': 50000,
            'Br': 0.8,
            'color': 'orange'
        }
    }

    fig = plt.figure(figsize=(16, 12))

    # Plot 1: B-H loops for different materials
    ax1 = fig.add_subplot(2, 2, 1)

    for name, params in materials.items():
        H_max = params['Hc'] * 5
        H_loop, B_loop = simple_hysteresis_loop(H_max, 200, params)

        # Normalize for comparison
        ax1.plot(H_loop / params['Hc'], B_loop / (MU_0 * params['Ms']),
                 color=params['color'], lw=2, label=name)

    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

    ax1.set_xlabel('H / Hc (normalized field)')
    ax1.set_ylabel('B / Bs (normalized flux density)')
    ax1.set_title('Magnetic Hysteresis Loops\n(normalized to coercivity and saturation)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Mark key features
    ax1.annotate('Coercivity Hc', xy=(-1, 0), xytext=(-2, 0.3),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=10)
    ax1.annotate('Remanence Br', xy=(0, 0.5), xytext=(1, 0.7),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=10)

    # Plot 2: Soft iron detailed loop
    ax2 = fig.add_subplot(2, 2, 2)

    soft_iron = materials['Soft Iron']
    H_max = soft_iron['Hc'] * 10
    H_loop, B_loop = simple_hysteresis_loop(H_max, 500, soft_iron)

    ax2.plot(H_loop, B_loop, 'b-', lw=2)
    ax2.fill(H_loop, B_loop, alpha=0.3, color='blue', label='Energy loss area')

    # Mark key points
    ax2.plot(0, soft_iron['Br'], 'ro', markersize=10, label=f"Br = {soft_iron['Br']} T")
    ax2.plot(soft_iron['Hc'], 0, 'go', markersize=10, label=f"Hc = {soft_iron['Hc']} A/m")

    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

    ax2.set_xlabel('Magnetic field H (A/m)')
    ax2.set_ylabel('Magnetic flux density B (T)')
    ax2.set_title('Soft Iron Hysteresis Loop')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Calculate and display energy loss
    loss = calculate_hysteresis_loss(H_loop, B_loop)
    ax2.text(0.95, 0.05, f'Energy loss per cycle:\n{loss:.2f} J/m^3',
             transform=ax2.transAxes, ha='right', va='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 3: Initial magnetization curves
    ax3 = fig.add_subplot(2, 2, 3)

    for name, params in materials.items():
        Ms = params['Ms']
        Hc = params['Hc']

        H = np.linspace(0, Hc * 10, 200)
        # Virgin curve (from demagnetized state)
        M = Ms * np.tanh(H / (Hc * 2))
        B = MU_0 * (H + M)

        ax3.plot(H / Hc, B / (MU_0 * Ms), color=params['color'], lw=2, label=name)

    ax3.set_xlabel('H / Hc')
    ax3.set_ylabel('B / Bs')
    ax3.set_title('Initial (Virgin) Magnetization Curves')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Energy loss vs frequency (Steinmetz equation)
    ax4 = fig.add_subplot(2, 2, 4)

    # Steinmetz equation: P = k * f * B_max^n
    # Typical n ~ 1.6 for steel

    f_range = np.logspace(0, 4, 100)  # 1 Hz to 10 kHz
    B_max_values = [0.5, 1.0, 1.5]  # Tesla

    k_steinmetz = 0.001  # Steinmetz coefficient (material dependent)
    n = 1.6  # Steinmetz exponent

    for B_max in B_max_values:
        P = k_steinmetz * f_range * B_max**n
        ax4.loglog(f_range, P, lw=2, label=f'B_max = {B_max} T')

    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Power loss (W/m^3)')
    ax4.set_title('Core Loss vs Frequency\n(Steinmetz empirical law)')
    ax4.legend()
    ax4.grid(True, alpha=0.3, which='both')

    # Mark typical frequencies
    ax4.axvline(x=50, color='gray', linestyle=':', alpha=0.5)
    ax4.text(55, 1e-2, '50 Hz\n(power)', fontsize=8)
    ax4.axvline(x=1000, color='gray', linestyle=':', alpha=0.5)
    ax4.text(1100, 1e-2, '1 kHz', fontsize=8)

    # Add physics summary
    fig.text(0.5, 0.02,
             r'Magnetic Hysteresis: Area of B-H loop = energy loss per cycle per unit volume'
             + '\n' +
             r'Key parameters: Coercivity $H_c$ (width), Remanence $B_r$ (height at H=0), '
             r'Saturation $B_s$',
             ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.suptitle('Magnetic Hysteresis in Ferromagnetic Materials', fontsize=14, y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)

    # Save plot
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'magnetic_hysteresis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
