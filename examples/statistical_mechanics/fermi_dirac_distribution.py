"""
Experiment 135: Fermi-Dirac Occupation vs Temperature

This example demonstrates the Fermi-Dirac distribution for fermions and
shows how occupation probability changes with temperature.

The Fermi-Dirac distribution is:
f(E) = 1 / (exp((E - mu)/(k_B*T)) + 1)

where:
- E = energy of the state
- mu = chemical potential (Fermi level at T=0)
- k_B = Boltzmann constant
- T = temperature

Key features:
- At T=0: Step function (all states below E_F are occupied)
- At T>0: States near E_F have partial occupation
- Width of transition region ~ k_B*T
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import brentq

# Physical constants
k_B = 1.380649e-23  # Boltzmann constant (J/K)
eV = 1.602176634e-19  # Electron volt (J)


def fermi_dirac(E, mu, T, k_B=1.0):
    """
    Fermi-Dirac distribution function.

    Args:
        E: Energy (can be array)
        mu: Chemical potential
        T: Temperature
        k_B: Boltzmann constant (default 1 for reduced units)

    Returns:
        Occupation probability
    """
    if T == 0:
        return np.where(E < mu, 1.0, np.where(E == mu, 0.5, 0.0))

    # Avoid overflow in exponential
    x = (E - mu) / (k_B * T)
    # For large positive x, f -> 0; for large negative x, f -> 1
    return np.where(x > 500, 0.0, np.where(x < -500, 1.0, 1.0 / (np.exp(x) + 1)))


def density_of_states_3d(E, m=1.0, hbar=1.0):
    """
    3D free electron density of states.

    g(E) = (2m)^(3/2) / (2*pi^2*hbar^3) * sqrt(E)

    Args:
        E: Energy
        m: Electron mass
        hbar: Reduced Planck constant

    Returns:
        Density of states
    """
    if np.isscalar(E):
        return 0.0 if E < 0 else (2*m)**(1.5) / (2 * np.pi**2 * hbar**3) * np.sqrt(E)
    result = np.zeros_like(E)
    mask = E >= 0
    result[mask] = (2*m)**(1.5) / (2 * np.pi**2 * hbar**3) * np.sqrt(E[mask])
    return result


def compute_chemical_potential(T, E_F, k_B=1.0, n_electrons=1.0):
    """
    Compute chemical potential at temperature T for a 3D free electron gas.
    Uses the constraint that total particle number is conserved.

    At low T: mu ≈ E_F * [1 - (pi^2/12) * (k_B*T/E_F)^2]

    Args:
        T: Temperature
        E_F: Fermi energy at T=0
        k_B: Boltzmann constant
        n_electrons: Number of electrons (normalized to 1)

    Returns:
        Chemical potential mu(T)
    """
    if T == 0:
        return E_F

    # Low temperature expansion
    x = k_B * T / E_F
    if x < 0.1:
        return E_F * (1 - (np.pi**2 / 12) * x**2)

    # For higher temperatures, solve numerically
    def particle_number(mu):
        E_max = max(10 * E_F, mu + 50 * k_B * T)
        E_range = np.linspace(0, E_max, 1000)
        dE = E_range[1] - E_range[0]
        n = np.sum(density_of_states_3d(E_range) * fermi_dirac(E_range, mu, T, k_B)) * dE
        return n

    n_target = particle_number(E_F)  # Number at T=0

    try:
        mu = brentq(lambda m: particle_number(m) - n_target, -5*E_F, 5*E_F)
    except ValueError:
        # Fallback to low-T approximation
        mu = E_F * (1 - (np.pi**2 / 12) * x**2)

    return mu


def sommerfeld_expansion(T, E_F, k_B=1.0):
    """
    Sommerfeld expansion for low-temperature properties.

    Returns chemical potential, internal energy, and heat capacity.
    """
    x = k_B * T / E_F

    # Chemical potential: mu = E_F * [1 - (pi^2/12)(k_B*T/E_F)^2]
    mu = E_F * (1 - (np.pi**2 / 12) * x**2)

    # Internal energy: U = U_0 + (pi^2/4) * N * (k_B*T)^2 / E_F
    # Heat capacity: C_V = (pi^2/2) * N * k_B * (k_B*T/E_F)
    gamma = np.pi**2 / 2  # Sommerfeld coefficient
    C_V = gamma * k_B * x

    return mu, C_V


def main():
    print("Fermi-Dirac Distribution vs Temperature")
    print("=" * 50)

    # Using reduced units: E_F = 1, k_B = 1
    E_F = 1.0  # Fermi energy

    # Energy range
    E = np.linspace(-0.5, 2.5, 1000)

    # Temperatures to study (in units of E_F/k_B)
    T_values = [0, 0.01, 0.05, 0.1, 0.2, 0.5]

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Fermi-Dirac distribution at different temperatures
    ax1 = axes[0, 0]
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(T_values)))

    for T, color in zip(T_values, colors):
        if T == 0:
            # Special handling for T=0 (step function)
            f = fermi_dirac(E, E_F, T)
            label = 'T = 0 (T=0 K)'
        else:
            # Compute chemical potential at this T
            mu = compute_chemical_potential(T, E_F)
            f = fermi_dirac(E, mu, T)
            label = f'T = {T:.2f} $E_F/k_B$'
        ax1.plot(E, f, color=color, lw=2, label=label)

    ax1.axvline(E_F, color='gray', linestyle='--', alpha=0.7, label='$E_F$')
    ax1.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Energy / $E_F$', fontsize=12)
    ax1.set_ylabel('Occupation probability f(E)', fontsize=12)
    ax1.set_title('Fermi-Dirac Distribution at Various Temperatures', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.5, 2.5)
    ax1.set_ylim(-0.05, 1.05)

    # Plot 2: Temperature smearing of the step
    ax2 = axes[0, 1]
    E_zoom = np.linspace(0.5, 1.5, 500)

    T_zoom = [0.01, 0.02, 0.05, 0.1, 0.2]
    for T in T_zoom:
        mu = compute_chemical_potential(T, E_F)
        f = fermi_dirac(E_zoom, mu, T)
        ax2.plot(E_zoom, f, lw=2, label=f'T = {T:.2f}')

    ax2.axvline(E_F, color='gray', linestyle='--', alpha=0.7)
    ax2.axhline(0.5, color='gray', linestyle=':', alpha=0.5)

    # Add annotation for thermal width
    T_demo = 0.1
    width = 4 * T_demo  # Width ~ 4*k_B*T
    ax2.annotate('', xy=(E_F - width/2, 0.5), xytext=(E_F + width/2, 0.5),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax2.text(E_F, 0.55, f'~4$k_B$T', ha='center', fontsize=10, color='red')

    ax2.set_xlabel('Energy / $E_F$', fontsize=12)
    ax2.set_ylabel('Occupation probability f(E)', fontsize=12)
    ax2.set_title('Thermal Smearing of Fermi Edge', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Chemical potential vs temperature
    ax3 = axes[1, 0]
    T_range = np.linspace(0.001, 0.5, 100)
    mu_values = [compute_chemical_potential(T, E_F) for T in T_range]
    mu_sommerfeld = [E_F * (1 - (np.pi**2 / 12) * (T/E_F)**2) for T in T_range]

    ax3.plot(T_range, mu_values, 'b-', lw=2, label='Numerical')
    ax3.plot(T_range, mu_sommerfeld, 'r--', lw=2, label='Sommerfeld expansion')
    ax3.axhline(E_F, color='gray', linestyle='--', alpha=0.5, label='$E_F$')
    ax3.set_xlabel('Temperature / ($E_F/k_B$)', fontsize=12)
    ax3.set_ylabel('Chemical potential / $E_F$', fontsize=12)
    ax3.set_title('Chemical Potential vs Temperature', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Comparison with Maxwell-Boltzmann (classical limit)
    ax4 = axes[1, 1]

    # At high energies, F-D approaches M-B
    E_high = np.linspace(E_F, 5 * E_F, 200)
    T_compare = 0.5
    mu = compute_chemical_potential(T_compare, E_F)

    # Fermi-Dirac
    f_fd = fermi_dirac(E_high, mu, T_compare)

    # Maxwell-Boltzmann
    f_mb = np.exp(-(E_high - mu) / T_compare)

    ax4.semilogy(E_high / E_F, f_fd, 'b-', lw=2, label='Fermi-Dirac')
    ax4.semilogy(E_high / E_F, f_mb, 'r--', lw=2, label='Maxwell-Boltzmann')

    ax4.set_xlabel('Energy / $E_F$', fontsize=12)
    ax4.set_ylabel('Occupation probability (log scale)', fontsize=12)
    ax4.set_title(f'F-D vs M-B at High Energies (T = {T_compare} $E_F/k_B$)', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3, which='both')
    ax4.set_ylim(1e-6, 10)

    plt.suptitle('Fermi-Dirac Distribution: Quantum Statistics of Fermions',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Print numerical analysis
    print("\nChemical Potential vs Temperature:")
    print(f"{'T/(E_F/k_B)':>12} {'mu/E_F (num)':>15} {'mu/E_F (Somm)':>15} {'Diff %':>10}")
    print("-" * 55)
    for T in [0.01, 0.05, 0.1, 0.2, 0.3]:
        mu_num = compute_chemical_potential(T, E_F)
        mu_som = E_F * (1 - (np.pi**2 / 12) * T**2)
        diff = 100 * abs(mu_num - mu_som) / E_F
        print(f"{T:>12.2f} {mu_num/E_F:>15.4f} {mu_som/E_F:>15.4f} {diff:>10.3f}")

    # Realistic example: metals
    print("\n" + "=" * 50)
    print("Physical Example: Copper (Cu)")
    print("=" * 50)
    E_F_Cu = 7.0 * eV  # Fermi energy of copper
    T_F_Cu = E_F_Cu / k_B  # Fermi temperature
    print(f"Fermi energy: {E_F_Cu/eV:.1f} eV")
    print(f"Fermi temperature: {T_F_Cu:.0f} K = {T_F_Cu/1000:.1f} kK")

    for T_K in [300, 1000, 3000]:
        ratio = T_K / T_F_Cu
        print(f"\nAt T = {T_K} K (T/T_F = {ratio:.4f}):")
        print(f"  Thermal smearing width: ~{4*k_B*T_K/eV*1000:.1f} meV")
        print(f"  mu/E_F change: {(np.pi**2/12)*ratio**2*100:.4f}%")

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'fermi_dirac_distribution.png'),
                dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {os.path.join(output_dir, 'fermi_dirac_distribution.png')}")


if __name__ == "__main__":
    main()
