"""
Experiment 230: Debye Heat Capacity

Demonstrates the Debye model for lattice heat capacity, showing the
transition from T^3 behavior at low temperatures to the classical
Dulong-Petit limit at high temperatures.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


# Physical constants
kB = 1.381e-23       # Boltzmann constant (J/K)
hbar = 1.055e-34     # Reduced Planck constant (J*s)
R = 8.314            # Gas constant (J/mol/K)


def debye_function(x):
    """
    Debye function D(x) = 3*(x^(-3)) * integral_0^x t^4*e^t/(e^t-1)^2 dt

    Args:
        x: Debye parameter Theta_D / T

    Returns:
        D(x) value
    """
    if x < 1e-10:
        return 1.0  # High T limit

    if x > 100:
        # Low T limit: D(x) ~ (4*pi^4/5) / x^3
        return 4 * np.pi**4 / (5 * x**3)

    def integrand(t):
        if t < 1e-10:
            return t**2
        exp_t = np.exp(t)
        return t**4 * exp_t / (exp_t - 1)**2

    result, _ = integrate.quad(integrand, 0, x, limit=100)
    return 3 * result / x**3


def debye_heat_capacity(T, Theta_D, n_atoms=1):
    """
    Debye model heat capacity.

    C_V = 9*n*R * (T/Theta_D)^3 * integral_0^{Theta_D/T} x^4*e^x/(e^x-1)^2 dx

    Args:
        T: Temperature (array)
        Theta_D: Debye temperature
        n_atoms: Number of atoms per formula unit

    Returns:
        Heat capacity C_V (J/mol/K)
    """
    T = np.atleast_1d(T)
    C = np.zeros_like(T, dtype=float)

    for i, temp in enumerate(T):
        if temp < 1e-10:
            C[i] = 0
        else:
            x = Theta_D / temp
            C[i] = 9 * n_atoms * R * debye_function(x)

    return C if len(C) > 1 else C[0]


def einstein_heat_capacity(T, Theta_E, n_atoms=1):
    """
    Einstein model heat capacity for comparison.

    C_V = 3*n*R * (Theta_E/T)^2 * exp(Theta_E/T) / (exp(Theta_E/T) - 1)^2

    Args:
        T: Temperature
        Theta_E: Einstein temperature
        n_atoms: Number of atoms per formula unit

    Returns:
        Heat capacity C_V
    """
    T = np.atleast_1d(T)
    C = np.zeros_like(T, dtype=float)

    for i, temp in enumerate(T):
        if temp < 1e-10:
            C[i] = 0
        else:
            x = Theta_E / temp
            if x > 100:
                C[i] = 0
            else:
                exp_x = np.exp(min(x, 500))
                C[i] = 3 * n_atoms * R * x**2 * exp_x / (exp_x - 1)**2

    return C if len(C) > 1 else C[0]


def low_T_approximation(T, Theta_D, n_atoms=1):
    """
    Low temperature (T << Theta_D) approximation: C_V ~ T^3
    """
    return 12 * np.pi**4 / 5 * n_atoms * R * (T / Theta_D)**3


def main():
    """Main simulation and visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Debye heat capacity vs reduced temperature
    ax1 = axes[0, 0]

    # Use reduced temperature T/Theta_D
    T_reduced = np.linspace(0.01, 2.5, 200)
    Theta_D = 1.0  # Normalize

    T = T_reduced * Theta_D
    C_debye = debye_heat_capacity(T, Theta_D)
    C_dulong_petit = 3 * R  # Classical limit

    ax1.plot(T_reduced, C_debye / R, 'b-', lw=2, label='Debye model')
    ax1.axhline(y=3, color='red', linestyle='--', alpha=0.7, label='Dulong-Petit (3R)')

    # Low-T approximation
    T_low = np.linspace(0.01, 0.3, 50)
    C_low_T = low_T_approximation(T_low, Theta_D) / R
    ax1.plot(T_low, C_low_T, 'g--', lw=2, alpha=0.7, label=r'$T^3$ law')

    ax1.set_xlabel(r'$T / \Theta_D$')
    ax1.set_ylabel(r'$C_V / R$')
    ax1.set_title('Debye Heat Capacity')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 2.5)
    ax1.set_ylim(0, 3.5)

    # Plot 2: Comparison with Einstein model
    ax2 = axes[0, 1]

    Theta_D = 400  # K (typical for metals)
    Theta_E = 0.8 * Theta_D  # Einstein temperature (empirical relation)

    T_range = np.linspace(1, 500, 200)

    C_debye = debye_heat_capacity(T_range, Theta_D)
    C_einstein = einstein_heat_capacity(T_range, Theta_E)

    ax2.plot(T_range, C_debye / R, 'b-', lw=2, label='Debye')
    ax2.plot(T_range, C_einstein / R, 'r--', lw=2, label='Einstein')
    ax2.axhline(y=3, color='gray', linestyle=':', alpha=0.5)
    ax2.axvline(x=Theta_D, color='blue', linestyle=':', alpha=0.5, label=r'$\Theta_D$')

    ax2.set_xlabel('Temperature (K)')
    ax2.set_ylabel(r'$C_V / R$')
    ax2.set_title(f'Debye vs Einstein Model ($\\Theta_D$ = {Theta_D} K)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Heat capacity of different materials
    ax3 = axes[1, 0]

    # Debye temperatures for various materials
    materials = {
        'Lead (Pb)': 105,
        'Gold (Au)': 165,
        'Copper (Cu)': 343,
        'Aluminum (Al)': 428,
        'Silicon (Si)': 645,
        'Diamond (C)': 2230
    }

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(materials)))

    T_range = np.linspace(1, 1000, 300)

    for (name, Theta_D), color in zip(materials.items(), colors):
        C = debye_heat_capacity(T_range, Theta_D)
        ax3.plot(T_range, C / R, color=color, lw=2, label=f'{name} ({Theta_D} K)')

    ax3.axhline(y=3, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Temperature (K)')
    ax3.set_ylabel(r'$C_V / R$')
    ax3.set_title('Heat Capacity of Various Materials')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 1000)
    ax3.set_ylim(0, 3.5)

    # Plot 4: Low temperature T^3 behavior (log-log plot)
    ax4 = axes[1, 1]

    Theta_D = 400

    T_range = np.logspace(0, 2.5, 100)  # 1 to ~300 K

    C_debye = debye_heat_capacity(T_range, Theta_D)
    C_T3 = low_T_approximation(T_range, Theta_D)

    ax4.loglog(T_range, C_debye / R, 'b-', lw=2, label='Debye model')
    ax4.loglog(T_range, C_T3 / R, 'r--', lw=2, alpha=0.7, label=r'$\propto T^3$')

    # Mark regimes
    ax4.axvline(x=Theta_D / 10, color='green', linestyle=':', alpha=0.7)
    ax4.text(Theta_D / 20, 0.01, r'$T \ll \Theta_D$', fontsize=10)
    ax4.text(Theta_D / 2, 2, r'$T \sim \Theta_D$', fontsize=10)

    ax4.set_xlabel('Temperature (K)')
    ax4.set_ylabel(r'$C_V / R$')
    ax4.set_title(f'Low Temperature Behavior ($\\Theta_D$ = {Theta_D} K)')
    ax4.legend()
    ax4.grid(True, alpha=0.3, which='both')
    ax4.set_xlim(1, 500)

    plt.suptitle('Debye Model for Lattice Heat Capacity\n'
                 r'$C_V = 9nR(T/\Theta_D)^3 \int_0^{x} \frac{t^4 e^t}{(e^t-1)^2} dt$, where $x = \Theta_D/T$',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'debye_heat_capacity.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'debye_heat_capacity.png')}")


if __name__ == "__main__":
    main()
