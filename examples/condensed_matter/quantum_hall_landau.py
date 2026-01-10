"""
Experiment 241: Quantum Hall Effect Landau Levels

Demonstrates the quantum Hall effect, showing Landau level quantization
in a 2D electron gas under a magnetic field and the resulting quantized
Hall conductance.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite
from scipy import linalg


# Physical constants
hbar = 1.055e-34    # Reduced Planck constant
e = 1.602e-19       # Electron charge
m_e = 9.109e-31     # Electron mass


def landau_level_energy(n, B, m_eff=m_e):
    """
    Landau level energy.

    E_n = hbar * omega_c * (n + 1/2)

    where omega_c = eB/m is the cyclotron frequency.

    Args:
        n: Landau level index (0, 1, 2, ...)
        B: Magnetic field (T)
        m_eff: Effective mass

    Returns:
        Energy in Joules
    """
    omega_c = e * B / m_eff
    return hbar * omega_c * (n + 0.5)


def magnetic_length(B):
    """
    Magnetic length l_B = sqrt(hbar / (eB)).

    This is the characteristic length scale in quantum Hall systems.

    Args:
        B: Magnetic field (T)

    Returns:
        Magnetic length in meters
    """
    return np.sqrt(hbar / (e * B))


def landau_wavefunction(x, y, n, k_y, B, m_eff=m_e):
    """
    Landau level wavefunction in Landau gauge A = (0, Bx, 0).

    psi_n,ky(x, y) = exp(i*ky*y) * phi_n(x - x_0)

    where phi_n is the harmonic oscillator wavefunction and x_0 = -l_B^2 * k_y.

    Args:
        x, y: Position (can be arrays)
        n: Landau level index
        k_y: y-momentum
        B: Magnetic field
        m_eff: Effective mass

    Returns:
        Wavefunction value (complex)
    """
    l_B = magnetic_length(B)
    x_0 = -l_B**2 * k_y

    # Shifted coordinate
    xi = (x - x_0) / l_B

    # Hermite polynomial
    H_n = hermite(n)

    # Harmonic oscillator eigenfunction
    prefactor = 1 / np.sqrt(2**n * np.math.factorial(n)) * (1/(np.pi * l_B**2))**0.25
    phi_n = prefactor * H_n(xi) * np.exp(-xi**2 / 2)

    # Full wavefunction
    psi = np.exp(1j * k_y * y) * phi_n

    return psi


def landau_dos(E, B, m_eff=m_e, gamma=0.01):
    """
    Density of states with Landau level broadening.

    g(E) = sum_n (1 / (pi*gamma)) / (1 + ((E - E_n) / gamma)^2)

    Args:
        E: Energy (array)
        B: Magnetic field
        m_eff: Effective mass
        gamma: Level broadening (in units of hbar*omega_c)

    Returns:
        DOS
    """
    omega_c = e * B / m_eff
    gamma_J = gamma * hbar * omega_c

    g = np.zeros_like(E)

    for n in range(20):  # Sum over Landau levels
        E_n = landau_level_energy(n, B, m_eff)
        # Lorentzian broadening
        g += (1 / (np.pi * gamma_J)) / (1 + ((E - E_n) / gamma_J)**2)

    return g


def hall_conductance(n_filled, e=1.0, h=1.0):
    """
    Quantized Hall conductance.

    sigma_xy = n * e^2 / h

    Args:
        n_filled: Number of filled Landau levels
        e: Electron charge (can use 1 for natural units)
        h: Planck constant (can use 1 for natural units)

    Returns:
        Hall conductance in units of e^2/h
    """
    return n_filled * e**2 / h


def filling_factor(n_e, B):
    """
    Landau level filling factor.

    nu = n_e * h / (eB) = n_e / n_B

    where n_B = eB/h is the degeneracy per Landau level per unit area.

    Args:
        n_e: Electron density (m^-2)
        B: Magnetic field (T)

    Returns:
        Filling factor
    """
    n_B = e * B / (2 * np.pi * hbar)
    return n_e / n_B


def main():
    """Main simulation and visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Parameters
    B = 10  # Magnetic field (T)
    m_eff = 0.067 * m_e  # GaAs effective mass

    # Plot 1: Landau level energies
    ax1 = axes[0, 0]

    B_range = np.linspace(0.1, 20, 100)
    n_levels = 6

    colors = plt.cm.viridis(np.linspace(0, 0.9, n_levels))

    for n, color in zip(range(n_levels), colors):
        E_n = landau_level_energy(n, B_range, m_eff)
        E_meV = E_n / (1.602e-22)  # Convert to meV
        ax1.plot(B_range, E_meV, color=color, lw=2, label=f'n = {n}')

    # Mark Fermi energy sweeping through levels
    E_F = 15  # meV
    ax1.axhline(y=E_F, color='red', linestyle='--', alpha=0.7, label='Fermi energy')

    ax1.set_xlabel('Magnetic Field (T)')
    ax1.set_ylabel('Energy (meV)')
    ax1.set_title('Landau Level Fan Diagram')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 20)
    ax1.set_ylim(0, 50)

    # Plot 2: Density of states
    ax2 = axes[0, 1]

    B = 5  # Tesla
    omega_c = e * B / m_eff
    E_range = np.linspace(0, 10 * hbar * omega_c, 500)
    E_meV = E_range / (1.602e-22)

    # Different broadening
    for gamma, style, label in [(0.01, '-', 'Clean'), (0.05, '--', 'Moderate'), (0.1, ':', 'Dirty')]:
        dos = landau_dos(E_range, B, m_eff, gamma)
        ax2.plot(E_meV, dos * 1.602e-22, style, lw=2, label=label)

    # Mark Landau levels
    for n in range(6):
        E_n = landau_level_energy(n, B, m_eff)
        ax2.axvline(x=E_n / 1.602e-22, color='gray', linestyle=':', alpha=0.5)

    ax2.set_xlabel('Energy (meV)')
    ax2.set_ylabel('DOS (arb. units)')
    ax2.set_title(f'Density of States (B = {B} T)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Hall conductance plateau
    ax3 = axes[1, 0]

    # Simulate sweeping Fermi energy through Landau levels
    B = 5
    n_e_range = np.linspace(1e14, 2e15, 500)  # Electron density (m^-2)

    nu = filling_factor(n_e_range, B)
    sigma_xy = np.floor(nu)  # Integer plateaus (simplified)

    # Add some rounding at transitions
    for i in range(1, len(sigma_xy)):
        if sigma_xy[i] != sigma_xy[i-1]:
            # Smooth transition
            pass

    ax3.plot(1/B * np.ones_like(nu), sigma_xy, 'b-', lw=2)

    # Better: plot as function of filling factor
    ax3.clear()
    ax3.step(nu, np.floor(nu), where='mid', lw=2, color='blue')

    ax3.set_xlabel('Filling factor nu')
    ax3.set_ylabel('Hall conductance (e^2/h)')
    ax3.set_title('Integer Quantum Hall Effect')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)

    # Mark plateaus
    for n in range(1, 10):
        ax3.axhline(y=n, color='gray', linestyle=':', alpha=0.3)
        ax3.text(n - 0.2, n + 0.2, f'nu={n}', fontsize=9)

    # Plot 4: Wavefunction probability density
    ax4 = axes[1, 1]

    B = 10
    l_B = magnetic_length(B)

    x = np.linspace(-5*l_B, 5*l_B, 200)
    y = np.zeros_like(x)  # y = 0 slice
    k_y = 0  # k_y = 0

    x_nm = x * 1e9  # Convert to nm

    for n in range(4):
        psi = landau_wavefunction(x, y, n, k_y, B, m_eff)
        prob = np.abs(psi)**2
        ax4.plot(x_nm, prob / np.max(prob) + n, lw=2, label=f'n = {n}')
        ax4.fill_between(x_nm, n, prob / np.max(prob) + n, alpha=0.3)

    ax4.set_xlabel('x (nm)')
    ax4.set_ylabel('|psi|^2 + offset')
    ax4.set_title(f'Landau Level Wavefunctions (l_B = {l_B*1e9:.1f} nm)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Mark magnetic length
    ax4.axvline(x=l_B*1e9, color='red', linestyle='--', alpha=0.5)
    ax4.axvline(x=-l_B*1e9, color='red', linestyle='--', alpha=0.5)

    plt.suptitle('Quantum Hall Effect and Landau Levels\n'
                 r'$E_n = \hbar\omega_c(n + 1/2)$, $\sigma_{xy} = \nu e^2/h$',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'quantum_hall_landau.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'quantum_hall_landau.png')}")


if __name__ == "__main__":
    main()
