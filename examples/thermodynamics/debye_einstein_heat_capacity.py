"""
Example comparing Debye and Einstein models of heat capacity.

Einstein Model (1907):
- Treats all atoms as independent quantum harmonic oscillators
- All oscillators have the same frequency
- C_V = 3Nk * (theta_E/T)^2 * exp(theta_E/T) / (exp(theta_E/T) - 1)^2

Debye Model (1912):
- Treats solid as elastic continuum with phonons
- Distribution of frequencies up to cutoff (Debye frequency)
- C_V = 9Nk * (T/theta_D)^3 * integral[x^4*exp(x)/(exp(x)-1)^2 dx] from 0 to theta_D/T

This example shows:
- Comparison of both models with experimental data
- Low and high temperature behavior
- Debye T^3 law at low temperatures
- Dulong-Petit limit at high temperatures
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from src.sciforge.core.constants import CONSTANTS


def einstein_heat_capacity(T, theta_E, n_atoms=1):
    """
    Calculate heat capacity using Einstein model.

    C_V = 3Nk * (theta_E/T)^2 * exp(theta_E/T) / (exp(theta_E/T) - 1)^2

    Args:
        T: Temperature (K)
        theta_E: Einstein temperature (K)
        n_atoms: Number of atoms (in moles * Avogadro)

    Returns:
        Heat capacity per mole (J/(mol K))
    """
    T = np.asarray(T)
    k_B = CONSTANTS['k']
    N_A = CONSTANTS['Na']

    # Avoid division by zero and overflow
    x = np.clip(theta_E / T, 1e-10, 700)

    exp_x = np.exp(x)
    C_V = 3 * N_A * k_B * x**2 * exp_x / (exp_x - 1)**2

    return C_V


def debye_integrand(x):
    """Integrand for Debye heat capacity: x^4 * exp(x) / (exp(x) - 1)^2"""
    if x < 1e-10:
        return 0
    if x > 700:
        return 0
    exp_x = np.exp(x)
    return x**4 * exp_x / (exp_x - 1)**2


def debye_heat_capacity(T, theta_D, n_atoms=1):
    """
    Calculate heat capacity using Debye model.

    C_V = 9Nk * (T/theta_D)^3 * integral[x^4*exp(x)/(exp(x)-1)^2 dx] from 0 to theta_D/T

    Args:
        T: Temperature (K)
        theta_D: Debye temperature (K)
        n_atoms: Number of atoms (in moles * Avogadro)

    Returns:
        Heat capacity per mole (J/(mol K))
    """
    T = np.atleast_1d(T)
    k_B = CONSTANTS['k']
    N_A = CONSTANTS['Na']
    R = CONSTANTS['R']

    C_V = np.zeros_like(T, dtype=float)

    for i, temp in enumerate(T):
        if temp < 1e-10:
            C_V[i] = 0
            continue

        x_max = theta_D / temp
        if x_max > 500:  # Low T limit: use T^3 approximation
            C_V[i] = (12/5) * np.pi**4 * R * (temp / theta_D)**3
        else:
            integral, _ = quad(debye_integrand, 0, x_max)
            C_V[i] = 9 * R * (temp / theta_D)**3 * integral

    return C_V if len(C_V) > 1 else C_V[0]


def dulong_petit_limit():
    """Classical Dulong-Petit limit: C_V = 3R"""
    return 3 * CONSTANTS['R']


def main():
    R = CONSTANTS['R']

    # Material parameters (approximate values)
    materials = {
        'Diamond': {'theta_D': 2230, 'theta_E': 1320, 'color': 'blue'},
        'Silicon': {'theta_D': 645, 'theta_E': 380, 'color': 'green'},
        'Copper': {'theta_D': 343, 'theta_E': 200, 'color': 'red'},
        'Lead': {'theta_D': 105, 'theta_E': 62, 'color': 'purple'},
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Comparison of Debye and Einstein models for one material (Cu)
    ax1 = axes[0, 0]

    theta_D = materials['Copper']['theta_D']
    theta_E = materials['Copper']['theta_E']

    T = np.linspace(1, 500, 200)

    C_debye = np.array([debye_heat_capacity(t, theta_D) for t in T])
    C_einstein = einstein_heat_capacity(T, theta_E)

    ax1.plot(T, C_debye, 'b-', lw=2, label='Debye model')
    ax1.plot(T, C_einstein, 'r--', lw=2, label='Einstein model')
    ax1.axhline(y=dulong_petit_limit(), color='gray', linestyle=':',
                lw=2, label=f'Dulong-Petit limit: 3R = {3*R:.1f} J/(mol K)')

    ax1.set_xlabel('Temperature (K)', fontsize=12)
    ax1.set_ylabel('Heat Capacity C$_V$ (J/(mol K))', fontsize=12)
    ax1.set_title(f'Heat Capacity of Copper\n'
                  f'(theta_D = {theta_D}K, theta_E = {theta_E}K)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 500)
    ax1.set_ylim(0, 30)

    # Plot 2: Low temperature behavior (T^3 law)
    ax2 = axes[0, 1]

    T_low = np.linspace(1, 100, 200)

    C_debye_low = np.array([debye_heat_capacity(t, theta_D) for t in T_low])
    C_einstein_low = einstein_heat_capacity(T_low, theta_E)

    # Theoretical T^3 law
    C_T3 = (12/5) * np.pi**4 * R * (T_low / theta_D)**3

    ax2.semilogy(T_low, C_debye_low, 'b-', lw=2, label='Debye')
    ax2.semilogy(T_low, C_einstein_low, 'r--', lw=2, label='Einstein')
    ax2.semilogy(T_low, C_T3, 'g:', lw=2, label='T$^3$ law')

    ax2.set_xlabel('Temperature (K)', fontsize=12)
    ax2.set_ylabel('Heat Capacity C$_V$ (J/(mol K), log scale)', fontsize=12)
    ax2.set_title('Low Temperature Behavior\n'
                  '(Debye follows T$^3$ law, Einstein decays exponentially)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_xlim(0, 100)

    # Add annotation
    ax2.annotate('Einstein model fails\nat low T', xy=(20, 0.1),
                fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Plot 3: Different materials (Debye model)
    ax3 = axes[1, 0]

    T_range = np.linspace(1, 600, 200)

    for name, params in materials.items():
        theta_D = params['theta_D']
        C = np.array([debye_heat_capacity(t, theta_D) for t in T_range])
        ax3.plot(T_range, C, color=params['color'], lw=2,
                label=f'{name} (theta_D = {theta_D}K)')

    ax3.axhline(y=dulong_petit_limit(), color='gray', linestyle=':', lw=1.5)

    ax3.set_xlabel('Temperature (K)', fontsize=12)
    ax3.set_ylabel('Heat Capacity C$_V$ (J/(mol K))', fontsize=12)
    ax3.set_title('Debye Heat Capacity for Various Materials', fontsize=12)
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 600)
    ax3.set_ylim(0, 30)

    # Add annotation about Debye temperature
    ax3.text(0.05, 0.95, 'Higher Debye temperature =\nharder material\n(stiffer bonds)',
             transform=ax3.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Plot 4: Reduced temperature plot (universal curve)
    ax4 = axes[1, 1]

    T_reduced = np.linspace(0.01, 3, 200)

    # Universal Debye function (using theta_D = 1 as reference)
    C_universal = np.array([debye_heat_capacity(t, 1.0) for t in T_reduced])

    # Einstein with theta_E = theta_D (for comparison)
    C_einstein_universal = einstein_heat_capacity(T_reduced, 1.0)

    ax4.plot(T_reduced, C_universal / R, 'b-', lw=2, label='Debye')
    ax4.plot(T_reduced, C_einstein_universal / R, 'r--', lw=2, label='Einstein')
    ax4.axhline(y=3, color='gray', linestyle=':', lw=1.5, label='Classical limit (3R)')

    ax4.set_xlabel('Reduced Temperature (T/theta)', fontsize=12)
    ax4.set_ylabel('C$_V$/R (dimensionless)', fontsize=12)
    ax4.set_title('Universal Heat Capacity Curve\n(Dimensionless form)', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 3)
    ax4.set_ylim(0, 3.5)

    # Add key values
    ax4.annotate(f'C/R = 3 at T >> theta',
                xy=(2.5, 3), xytext=(2.0, 2.5),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=10)

    ax4.annotate('Debye: C ~ T$^3$\nEinstein: C ~ exp(-theta/T)',
                xy=(0.2, 0.5), fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.suptitle('Debye vs Einstein Models of Heat Capacity in Solids',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'debye_einstein_heat_capacity.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'debye_einstein_heat_capacity.png')}")


if __name__ == "__main__":
    main()
