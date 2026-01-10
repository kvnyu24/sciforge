"""
Experiment 231: Thermal Conductivity from Phonons

Demonstrates phonon-mediated thermal conductivity using the kinetic theory
approach: kappa = (1/3) * C_V * v * l, showing temperature dependence and
the competition between phonon heat capacity and scattering mechanisms.
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


def debye_heat_capacity_per_volume(T, Theta_D, n_density):
    """
    Debye heat capacity per unit volume.

    Args:
        T: Temperature (K)
        Theta_D: Debye temperature (K)
        n_density: Atom number density (1/m^3)

    Returns:
        C_V in J/(m^3 K)
    """
    if T < 1e-10:
        return 0

    x = Theta_D / T

    def integrand(t):
        if t < 1e-10:
            return t**2
        exp_t = np.exp(min(t, 500))
        return t**4 * exp_t / (exp_t - 1)**2

    if x > 100:
        # Low T limit
        integral = 4 * np.pi**4 / 15
    else:
        integral, _ = integrate.quad(integrand, 0, x, limit=100)

    return 9 * n_density * kB * (T / Theta_D)**3 * integral


def phonon_mean_free_path(T, Theta_D, L_boundary, A_umklapp):
    """
    Phonon mean free path considering multiple scattering mechanisms.

    1/l = 1/L + A*T^3*exp(-Theta_D/bT)  (boundary + Umklapp)

    Args:
        T: Temperature (K)
        Theta_D: Debye temperature (K)
        L_boundary: Boundary scattering length (m)
        A_umklapp: Umklapp scattering coefficient

    Returns:
        Mean free path l (m)
    """
    # Boundary scattering (T-independent)
    tau_boundary_inv = 1 / L_boundary

    # Umklapp scattering (dominates at high T)
    # l_U ~ 1/(T^3 * exp(-Theta_D/(3T)))
    b = 3.0  # Fitting parameter
    if T > Theta_D / 50:
        tau_umklapp_inv = A_umklapp * T**3 * np.exp(-Theta_D / (b * T))
    else:
        tau_umklapp_inv = 0

    return 1 / (tau_boundary_inv + tau_umklapp_inv)


def debye_velocity(Theta_D, a):
    """
    Average Debye velocity.

    v_D = Theta_D * kB / (hbar * (6*pi^2/a^3)^(1/3))
    """
    return Theta_D * kB * a / (hbar * (6 * np.pi**2)**(1/3))


def thermal_conductivity(T, Theta_D, v_sound, n_density, L_boundary, A_umklapp):
    """
    Thermal conductivity from kinetic theory.

    kappa = (1/3) * C_V * v * l

    Args:
        T: Temperature (K)
        Theta_D: Debye temperature (K)
        v_sound: Sound velocity (m/s)
        n_density: Atom density (1/m^3)
        L_boundary: Boundary scattering length (m)
        A_umklapp: Umklapp scattering coefficient

    Returns:
        Thermal conductivity kappa (W/m/K)
    """
    C_V = debye_heat_capacity_per_volume(T, Theta_D, n_density)
    l = phonon_mean_free_path(T, Theta_D, L_boundary, A_umklapp)

    return (1/3) * C_V * v_sound * l


def callaway_model(T, Theta_D, v_sound, n_density, tau_N, tau_U, tau_b):
    """
    Callaway model for thermal conductivity (simplified).

    Includes normal and Umklapp processes with different relaxation times.

    Args:
        T: Temperature
        Theta_D: Debye temperature
        v_sound: Sound velocity
        n_density: Atom density
        tau_N: Normal process relaxation time scale
        tau_U: Umklapp process relaxation time scale
        tau_b: Boundary scattering time

    Returns:
        Thermal conductivity
    """
    # This is a simplified version
    x_max = Theta_D / T

    def integrand(x):
        if x < 1e-10:
            return 0
        exp_x = np.exp(min(x, 500))
        # Relaxation time approximation
        omega = x * kB * T / hbar
        tau_combined = 1 / (1/tau_b + omega**2/tau_N + omega**4/tau_U * np.exp(-Theta_D/(3*T)))
        return x**4 * exp_x / (exp_x - 1)**2 * tau_combined

    if x_max > 100:
        x_max = 100

    result, _ = integrate.quad(integrand, 0, x_max, limit=100)

    prefactor = kB / (2 * np.pi**2 * v_sound) * (kB * T / hbar)**3
    return prefactor * result


def main():
    """Main simulation and visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Material parameters (silicon-like)
    Theta_D = 645      # Debye temperature (K)
    v_sound = 8000     # Average sound velocity (m/s)
    a = 5.43e-10       # Lattice constant (m)
    n_density = 8 / a**3  # Si has 8 atoms per unit cell

    # Scattering parameters
    L_boundary = 1e-6  # Boundary scattering length (1 um)
    A_umklapp = 1e-15  # Umklapp coefficient

    # Temperature range
    T_range = np.linspace(1, 500, 200)

    # Plot 1: Heat capacity and mean free path vs T
    ax1 = axes[0, 0]

    C_V = np.array([debye_heat_capacity_per_volume(T, Theta_D, n_density) for T in T_range])
    l_mfp = np.array([phonon_mean_free_path(T, Theta_D, L_boundary, A_umklapp) for T in T_range])

    ax1_twin = ax1.twinx()

    l1, = ax1.plot(T_range, C_V / 1e6, 'b-', lw=2, label='Heat capacity C_V')
    ax1.set_xlabel('Temperature (K)')
    ax1.set_ylabel('C_V (MJ/m^3/K)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    l2, = ax1_twin.semilogy(T_range, l_mfp * 1e6, 'r-', lw=2, label='Mean free path')
    ax1_twin.set_ylabel('Mean free path (um)', color='red')
    ax1_twin.tick_params(axis='y', labelcolor='red')

    ax1.axvline(x=Theta_D, color='gray', linestyle='--', alpha=0.5)
    ax1.text(Theta_D + 10, C_V.max()/1e6 * 0.9, r'$\Theta_D$', fontsize=10)

    ax1.legend([l1, l2], ['Heat capacity', 'Mean free path'], loc='right')
    ax1.set_title('Heat Capacity and Mean Free Path')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Thermal conductivity vs temperature
    ax2 = axes[0, 1]

    kappa = np.array([thermal_conductivity(T, Theta_D, v_sound, n_density,
                                           L_boundary, A_umklapp)
                      for T in T_range])

    ax2.semilogy(T_range, kappa, 'b-', lw=2, label='Total')

    # Show limiting behaviors
    # Low T: kappa ~ T^3 (boundary limited, C_V ~ T^3)
    T_low = T_range[T_range < Theta_D/10]
    kappa_low = kappa[:len(T_low)]
    ax2.semilogy(T_low, kappa_low[0] * (T_low/T_low[0])**3, 'g--', lw=1.5,
                alpha=0.7, label=r'$\propto T^3$ (low T)')

    # High T: kappa ~ 1/T (Umklapp limited)
    T_high = T_range[T_range > Theta_D]
    if len(T_high) > 0:
        kappa_high = kappa[-len(T_high):]
        idx_ref = len(T_range) - len(T_high)
        kappa_1_T = kappa[idx_ref] * (T_range[idx_ref] / T_high)
        ax2.semilogy(T_high, kappa_1_T, 'r--', lw=1.5, alpha=0.7,
                    label=r'$\propto 1/T$ (high T)')

    ax2.axvline(x=Theta_D, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Temperature (K)')
    ax2.set_ylabel('Thermal conductivity (W/m/K)')
    ax2.set_title('Thermal Conductivity vs Temperature')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')

    # Plot 3: Effect of sample size (boundary scattering)
    ax3 = axes[1, 0]

    L_values = [100e-9, 1e-6, 10e-6, 100e-6]  # Different sample sizes
    labels = ['100 nm', '1 um', '10 um', '100 um']
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(L_values)))

    for L, label, color in zip(L_values, labels, colors):
        kappa_L = np.array([thermal_conductivity(T, Theta_D, v_sound, n_density, L, A_umklapp)
                           for T in T_range])
        ax3.semilogy(T_range, kappa_L, color=color, lw=2, label=f'L = {label}')

    ax3.set_xlabel('Temperature (K)')
    ax3.set_ylabel('Thermal conductivity (W/m/K)')
    ax3.set_title('Effect of Sample Size (Boundary Scattering)')
    ax3.legend()
    ax3.grid(True, alpha=0.3, which='both')

    # Plot 4: Comparison of different materials
    ax4 = axes[1, 1]

    materials = {
        'Diamond': {'Theta_D': 2230, 'v': 12000, 'L': 1e-5, 'A': 1e-18},
        'Silicon': {'Theta_D': 645, 'v': 8000, 'L': 1e-6, 'A': 1e-15},
        'Aluminum': {'Theta_D': 428, 'v': 5100, 'L': 1e-6, 'A': 1e-14},
        'Lead': {'Theta_D': 105, 'v': 1200, 'L': 1e-6, 'A': 1e-13}
    }

    T_range_mat = np.linspace(1, 400, 150)
    colors = plt.cm.tab10(np.linspace(0, 0.4, len(materials)))

    for (name, params), color in zip(materials.items(), colors):
        kappa_mat = np.array([thermal_conductivity(T, params['Theta_D'], params['v'],
                                                   n_density, params['L'], params['A'])
                              for T in T_range_mat])
        ax4.semilogy(T_range_mat, kappa_mat, color=color, lw=2,
                    label=f"{name} ($\\Theta_D$={params['Theta_D']}K)")

    ax4.set_xlabel('Temperature (K)')
    ax4.set_ylabel('Thermal conductivity (W/m/K)')
    ax4.set_title('Thermal Conductivity of Different Materials')
    ax4.legend()
    ax4.grid(True, alpha=0.3, which='both')

    plt.suptitle('Phonon Thermal Conductivity\n'
                 r'$\kappa = \frac{1}{3} C_V v \ell$ (Kinetic theory)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'thermal_conductivity_phonon.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'thermal_conductivity_phonon.png')}")


if __name__ == "__main__":
    main()
