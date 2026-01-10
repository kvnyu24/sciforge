"""
Experiment 238: BCS Gap vs Temperature

Demonstrates the BCS theory of superconductivity, showing the temperature
dependence of the superconducting gap Delta(T), the density of states,
and the gap equation solution.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize


# Physical constants
kB = 1.381e-23  # Boltzmann constant


def fermi_function(E, T):
    """
    Fermi-Dirac distribution.

    f(E) = 1 / (1 + exp(E/kT))
    """
    if T == 0:
        return np.where(E < 0, 1.0, 0.0)
    x = E / (kB * T)
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(x))


def bcs_dos(E, Delta):
    """
    BCS density of states.

    N(E) / N(0) = |E| / sqrt(E^2 - Delta^2) for |E| > Delta
               = 0 for |E| < Delta

    Args:
        E: Energy (relative to Fermi level)
        Delta: Superconducting gap

    Returns:
        Normalized DOS
    """
    E = np.atleast_1d(E)
    dos = np.zeros_like(E, dtype=float)
    mask = np.abs(E) > np.abs(Delta)
    dos[mask] = np.abs(E[mask]) / np.sqrt(E[mask]**2 - Delta**2)
    return dos


def gap_integral(Delta, T, omega_D, N0_V):
    """
    BCS gap equation integral.

    1 = N(0)*V * integral_0^omega_D dE / sqrt(E^2 + Delta^2) * tanh(sqrt(E^2+Delta^2)/(2kT))

    Returns the difference (LHS - RHS) for root finding.

    Args:
        Delta: Gap value to test
        T: Temperature
        omega_D: Debye frequency cutoff
        N0_V: Dimensionless coupling constant N(0)*V

    Returns:
        Residual for gap equation
    """
    if Delta <= 0:
        return 1.0  # Gap must be positive

    def integrand(E):
        xi = np.sqrt(E**2 + Delta**2)
        if T == 0:
            return 1 / xi
        else:
            return np.tanh(xi / (2 * kB * T)) / xi

    result, _ = integrate.quad(integrand, 0, omega_D, limit=100)
    return 1 - N0_V * result


def solve_gap_equation(T, omega_D, N0_V, Delta_0):
    """
    Solve BCS gap equation at temperature T.

    Args:
        T: Temperature
        omega_D: Debye cutoff
        N0_V: Coupling constant
        Delta_0: Zero-temperature gap (initial guess)

    Returns:
        Delta(T)
    """
    if T == 0:
        # Zero temperature - use asymptotic formula
        return Delta_0

    # Start with T=0 gap as initial guess
    try:
        Delta = optimize.brentq(gap_integral, 1e-10, 1.5*Delta_0,
                               args=(T, omega_D, N0_V))
    except ValueError:
        # No solution - above T_c
        Delta = 0

    return Delta


def bcs_gap_approximate(t):
    """
    Approximate BCS gap temperature dependence.

    Delta(T)/Delta(0) ~ 1.74 * sqrt(1 - t) for t -> 1
    Delta(T)/Delta(0) ~ 1 - sqrt(2*pi*t) * exp(-1/t) for t -> 0

    where t = T/T_c

    Args:
        t: Reduced temperature T/T_c

    Returns:
        Delta/Delta_0
    """
    t = np.atleast_1d(t)
    result = np.zeros_like(t)

    # Near T_c
    mask_high = t > 0.5
    result[mask_high] = 1.74 * np.sqrt(1 - t[mask_high])

    # Low temperature
    mask_low = (t > 0) & (t <= 0.5)
    result[mask_low] = 1 - np.sqrt(2 * np.pi * t[mask_low]) * np.exp(-1/t[mask_low])

    result[t <= 0] = 1.0
    result[t >= 1] = 0.0

    return result


def critical_temperature(Delta_0, gamma=1.764):
    """
    BCS critical temperature.

    T_c = Delta_0 / (gamma * kB)

    where gamma = pi/exp(C_Euler) ~ 1.764

    Args:
        Delta_0: Zero-temperature gap
        gamma: BCS ratio

    Returns:
        T_c
    """
    return Delta_0 / (gamma * kB)


def main():
    """Main simulation and visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # BCS parameters (Aluminum-like)
    Delta_0 = 0.34e-3 * 1.602e-19  # Gap at T=0 (0.34 meV in Joules)
    T_c = Delta_0 / (1.764 * kB)   # Critical temperature (~1.2 K)
    omega_D = 10 * Delta_0         # Debye cutoff

    # Coupling constant from T_c equation
    N0_V = 1 / np.log(2 * omega_D / Delta_0)

    print(f"Delta_0 = {Delta_0/1.602e-22:.3f} meV")
    print(f"T_c = {T_c:.2f} K")
    print(f"N(0)V = {N0_V:.3f}")

    # Plot 1: Gap vs Temperature
    ax1 = axes[0, 0]

    t_range = np.linspace(0, 1.1, 100)
    T_range = t_range * T_c

    # Approximate solution
    Delta_approx = Delta_0 * bcs_gap_approximate(t_range)

    ax1.plot(t_range, Delta_approx / Delta_0, 'b-', lw=2, label='BCS theory')

    # BCS ratio
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.axvline(x=1, color='red', linestyle='--', alpha=0.5, label='T_c')

    # Mark key points
    ax1.plot(0, 1, 'ko', markersize=8)
    ax1.annotate('Delta(0)', xy=(0, 1), xytext=(0.1, 1.05), fontsize=10)

    ax1.set_xlabel('T / T_c')
    ax1.set_ylabel('Delta(T) / Delta(0)')
    ax1.set_title('BCS Gap Temperature Dependence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1.1)
    ax1.set_ylim(0, 1.2)

    # Add BCS ratio text
    ax1.text(0.5, 0.95, f'$2\\Delta(0)/k_B T_c = {2*Delta_0/(kB*T_c):.2f}$\n(BCS: 3.53)',
             transform=ax1.transAxes, fontsize=11, va='top', ha='center',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot 2: Density of States
    ax2 = axes[0, 1]

    E_range = np.linspace(-3*Delta_0, 3*Delta_0, 1000)
    E_meV = E_range / (1.602e-22)  # Convert to meV

    # DOS at different temperatures
    t_values = [0, 0.5, 0.8, 0.95]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(t_values)))

    for t, color in zip(t_values, colors):
        Delta_t = Delta_0 * bcs_gap_approximate(t)
        dos = bcs_dos(E_range, Delta_t)
        # Clip for visualization
        dos = np.clip(dos, 0, 5)
        ax2.plot(E_meV, dos, color=color, lw=2, label=f'T/T_c = {t}')

    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Normal state')
    ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

    ax2.set_xlabel('Energy E (meV)')
    ax2.set_ylabel('N(E) / N(0)')
    ax2.set_title('BCS Density of States')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 5)

    # Plot 3: Coherence peaks
    ax3 = axes[1, 0]

    # Thermally smeared DOS
    T_values = [0.1*T_c, 0.3*T_c, 0.5*T_c]
    colors = ['blue', 'green', 'red']

    for T_val, color in zip(T_values, colors):
        Delta_t = Delta_0 * bcs_gap_approximate(T_val/T_c)
        dos = bcs_dos(E_range, Delta_t)

        # Thermal smearing: convolve with derivative of Fermi function
        # Simplified: just multiply by (1 - 2*f(E))
        thermal_factor = 1 - 2 * fermi_function(E_range, T_val)
        dos_thermal = dos * np.abs(thermal_factor)
        dos_thermal = np.clip(dos_thermal, 0, 5)

        ax3.plot(E_meV, dos_thermal, color=color, lw=2,
                label=f'T = {T_val/T_c:.1f} T_c')

    ax3.axvline(x=Delta_0/(1.602e-22), color='gray', linestyle=':', alpha=0.5, label='Delta(0)')
    ax3.axvline(x=-Delta_0/(1.602e-22), color='gray', linestyle=':', alpha=0.5)

    ax3.set_xlabel('Energy E (meV)')
    ax3.set_ylabel('Thermally weighted DOS')
    ax3.set_title('Coherence Peaks in Tunneling DOS')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Comparison with experimental data for various materials
    ax4 = axes[1, 1]

    # Different superconductors
    materials = {
        'Al': {'T_c': 1.2, 'Delta_0': 0.17, 'color': 'blue'},
        'Sn': {'T_c': 3.7, 'Delta_0': 0.58, 'color': 'green'},
        'Pb': {'T_c': 7.2, 'Delta_0': 1.35, 'color': 'red'},
        'Nb': {'T_c': 9.3, 'Delta_0': 1.55, 'color': 'purple'}
    }

    t_plot = np.linspace(0, 1, 100)
    Delta_bcs = bcs_gap_approximate(t_plot)

    # Universal BCS curve
    ax4.plot(t_plot, Delta_bcs, 'k-', lw=3, label='BCS universal curve')

    # Mark materials
    for name, params in materials.items():
        ratio = 2 * params['Delta_0'] / (1.764 * params['T_c'])
        ax4.scatter([0], [1], color=params['color'], s=100, marker='o',
                   label=f"{name}: $2\\Delta/k_BT_c$ = {ratio:.2f}")

    ax4.set_xlabel('T / T_c')
    ax4.set_ylabel('Delta(T) / Delta(0)')
    ax4.set_title('Universal BCS Gap Curve')
    ax4.legend(loc='lower left')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 1.1)
    ax4.set_ylim(0, 1.2)

    # Add BCS prediction text
    ax4.text(0.95, 0.95, 'BCS prediction:\n$2\\Delta(0)/k_BT_c = 3.53$',
             transform=ax4.transAxes, fontsize=10, va='top', ha='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle('BCS Theory of Superconductivity\n'
                 r'Gap equation: $1 = N(0)V \int_0^{\omega_D} \frac{\tanh(\beta\xi/2)}{\xi} d\epsilon$',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'bcs_gap_temperature.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'bcs_gap_temperature.png')}")


if __name__ == "__main__":
    main()
