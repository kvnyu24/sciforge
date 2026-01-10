"""
Experiment 225: Density of States in 1D/2D/3D

Compares the density of states (DOS) for free electrons in different
dimensions, demonstrating the characteristic power-law behaviors:
1D: g(E) ~ E^(-1/2), 2D: g(E) ~ constant, 3D: g(E) ~ E^(1/2)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def dos_1d_free(E, m=1.0, hbar=1.0, L=1.0):
    """
    Density of states for 1D free electrons.

    g(E) = L/(pi*hbar) * sqrt(m/(2E))

    Args:
        E: Energy (array)
        m: Electron mass
        hbar: Reduced Planck constant
        L: System length

    Returns:
        DOS g(E)
    """
    E = np.asarray(E)
    g = np.zeros_like(E)
    mask = E > 0
    g[mask] = L / (np.pi * hbar) * np.sqrt(m / (2 * E[mask]))
    return g


def dos_2d_free(E, m=1.0, hbar=1.0, A=1.0):
    """
    Density of states for 2D free electrons.

    g(E) = A*m / (pi*hbar^2) for E > 0

    Args:
        E: Energy (array)
        m: Electron mass
        hbar: Reduced Planck constant
        A: System area

    Returns:
        DOS g(E)
    """
    E = np.asarray(E)
    g = np.zeros_like(E)
    g[E > 0] = A * m / (np.pi * hbar**2)
    return g


def dos_3d_free(E, m=1.0, hbar=1.0, V=1.0):
    """
    Density of states for 3D free electrons.

    g(E) = V/(2*pi^2) * (2m/hbar^2)^(3/2) * sqrt(E)

    Args:
        E: Energy (array)
        m: Electron mass
        hbar: Reduced Planck constant
        V: System volume

    Returns:
        DOS g(E)
    """
    E = np.asarray(E)
    g = np.zeros_like(E)
    mask = E > 0
    prefactor = V / (2 * np.pi**2) * (2 * m / hbar**2)**(3/2)
    g[mask] = prefactor * np.sqrt(E[mask])
    return g


def dos_1d_tight_binding(E, t=1.0, a=1.0):
    """
    DOS for 1D tight-binding model.

    g(E) = 1/(pi*a*t*sqrt(1-(E/2t)^2)) for |E| < 2t

    Args:
        E: Energy (array)
        t: Hopping parameter
        a: Lattice constant

    Returns:
        DOS g(E)
    """
    E = np.asarray(E)
    g = np.zeros_like(E)
    mask = np.abs(E) < 2*t
    x = E[mask] / (2*t)
    g[mask] = 1 / (np.pi * a * t * np.sqrt(1 - x**2 + 1e-10))
    return g


def dos_2d_square_tight_binding(E, t=1.0, n_k=500):
    """
    DOS for 2D square lattice tight-binding (numerical).

    Args:
        E: Energy values for DOS
        t: Hopping parameter
        n_k: Number of k-points per dimension

    Returns:
        DOS g(E)
    """
    # Generate k-points in first BZ
    kx = np.linspace(-np.pi, np.pi, n_k)
    ky = np.linspace(-np.pi, np.pi, n_k)
    KX, KY = np.meshgrid(kx, ky)

    # Energy dispersion
    E_k = -2 * t * (np.cos(KX) + np.cos(KY))
    E_flat = E_k.flatten()

    # Compute DOS by histogram
    dos, edges = np.histogram(E_flat, bins=E, density=True)
    E_centers = (edges[:-1] + edges[1:]) / 2

    return E_centers, dos


def fermi_dirac(E, mu, T, kB=1.0):
    """
    Fermi-Dirac distribution.

    Args:
        E: Energy
        mu: Chemical potential
        T: Temperature
        kB: Boltzmann constant

    Returns:
        f(E) occupation probability
    """
    if T == 0:
        return np.where(E < mu, 1.0, 0.0)
    x = (E - mu) / (kB * T)
    # Avoid overflow
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(x))


def main():
    """Main simulation and visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Energy range
    E = np.linspace(0.01, 5, 500)

    # Plot 1: Free electron DOS in 1D, 2D, 3D
    ax1 = axes[0, 0]

    g_1d = dos_1d_free(E)
    g_2d = dos_2d_free(E)
    g_3d = dos_3d_free(E)

    # Normalize for comparison
    g_1d_norm = g_1d / np.max(g_1d[E > 0.1])
    g_2d_norm = g_2d / np.max(g_2d)
    g_3d_norm = g_3d / np.max(g_3d)

    ax1.plot(E, g_1d_norm, 'b-', lw=2, label='1D: $g(E) \propto E^{-1/2}$')
    ax1.plot(E, g_2d_norm, 'g-', lw=2, label='2D: $g(E) \propto$ const')
    ax1.plot(E, g_3d_norm, 'r-', lw=2, label='3D: $g(E) \propto E^{1/2}$')

    ax1.set_xlabel('Energy (arb. units)')
    ax1.set_ylabel('DOS (normalized)')
    ax1.set_title('Free Electron DOS in Different Dimensions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 5)
    ax1.set_ylim(0, 2)

    # Plot 2: Tight-binding DOS in 1D and 2D
    ax2 = axes[0, 1]

    E_tb = np.linspace(-4.5, 4.5, 500)
    g_tb_1d = dos_1d_tight_binding(E_tb, t=1.0)

    ax2.plot(E_tb, g_tb_1d, 'b-', lw=2, label='1D tight-binding')

    # 2D tight-binding (numerical)
    E_bins = np.linspace(-4.5, 4.5, 200)
    E_centers, g_tb_2d = dos_2d_square_tight_binding(E_bins, t=1.0, n_k=300)

    ax2.plot(E_centers, g_tb_2d * 3, 'g-', lw=2, label='2D square lattice (scaled)')

    ax2.set_xlabel('Energy (units of t)')
    ax2.set_ylabel('DOS (arb. units)')
    ax2.set_title('Tight-Binding DOS: Van Hove Singularities')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Van Hove singularity (2D)')

    # Mark band edges
    ax2.axvline(x=-2, color='gray', linestyle=':', alpha=0.5)
    ax2.axvline(x=2, color='gray', linestyle=':', alpha=0.5)

    # Plot 3: Filling at finite temperature
    ax3 = axes[1, 0]

    E_range = np.linspace(-3, 5, 500)
    g_3d_full = dos_3d_free(E_range)
    g_3d_full[E_range < 0] = 0

    mu = 2.0  # Fermi level
    temperatures = [0.01, 0.2, 0.5, 1.0]
    colors = ['blue', 'green', 'orange', 'red']

    for T, color in zip(temperatures, colors):
        f = fermi_dirac(E_range, mu, T)
        occupied = g_3d_full * f
        ax3.plot(E_range, occupied, color=color, lw=2, label=f'T = {T}')

    ax3.plot(E_range, g_3d_full, 'k--', lw=1.5, alpha=0.5, label='DOS (T=0)')
    ax3.axvline(x=mu, color='gray', linestyle=':', alpha=0.7, label=f'$\mu$ = {mu}')

    ax3.set_xlabel('Energy (arb. units)')
    ax3.set_ylabel('Occupied DOS')
    ax3.set_title('Thermal Smearing of Fermi Surface (3D)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Integrated DOS (number of states)
    ax4 = axes[1, 1]

    # Number of states up to energy E
    dE = E[1] - E[0]
    N_1d = np.cumsum(g_1d) * dE
    N_2d = np.cumsum(g_2d) * dE
    N_3d = np.cumsum(g_3d) * dE

    # Normalize
    N_1d_norm = N_1d / np.max(N_1d)
    N_2d_norm = N_2d / np.max(N_2d)
    N_3d_norm = N_3d / np.max(N_3d)

    ax4.plot(E, N_1d_norm, 'b-', lw=2, label='1D: $N(E) \propto E^{1/2}$')
    ax4.plot(E, N_2d_norm, 'g-', lw=2, label='2D: $N(E) \propto E$')
    ax4.plot(E, N_3d_norm, 'r-', lw=2, label='3D: $N(E) \propto E^{3/2}$')

    ax4.set_xlabel('Energy (arb. units)')
    ax4.set_ylabel('N(E) / N_max (integrated DOS)')
    ax4.set_title('Integrated Density of States')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Density of States: Dimension Dependence\n'
                 'Free electrons and tight-binding models',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'dos_1d_2d_3d.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'dos_1d_2d_3d.png')}")


if __name__ == "__main__":
    main()
