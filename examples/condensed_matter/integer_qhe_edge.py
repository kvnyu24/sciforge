"""
Experiment 235: Integer Quantum Hall Effect Edge States

Demonstrates the integer quantum Hall effect (IQHE) with focus on edge states:
- Landau levels and edge state dispersion
- Chiral edge channels
- Hall conductance quantization sigma_xy = n*e^2/h
- Disorder and robustness of quantization
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.special import hermite


# Physical constants
hbar = 1.055e-34    # Reduced Planck constant (J*s)
e = 1.602e-19       # Electron charge (C)
m_e = 9.109e-31     # Electron mass (kg)


def magnetic_length(B):
    """
    Magnetic length l_B = sqrt(hbar/(eB)).

    Args:
        B: Magnetic field (T)

    Returns:
        Magnetic length (m)
    """
    return np.sqrt(hbar / (e * B))


def cyclotron_frequency(B, m_eff=m_e):
    """
    Cyclotron frequency omega_c = eB/m.

    Args:
        B: Magnetic field (T)
        m_eff: Effective mass (kg)

    Returns:
        Cyclotron frequency (rad/s)
    """
    return e * B / m_eff


def landau_level_energy(n, B, m_eff=m_e):
    """
    Landau level energy.

    E_n = hbar * omega_c * (n + 1/2)

    Args:
        n: Landau level index (0, 1, 2, ...)
        B: Magnetic field (T)
        m_eff: Effective mass (kg)

    Returns:
        Energy (J)
    """
    omega_c = cyclotron_frequency(B, m_eff)
    return hbar * omega_c * (n + 0.5)


def edge_state_dispersion(k_y, n, B, V_edge, m_eff=m_e, W=None):
    """
    Edge state dispersion in a Hall bar with confining potential.

    For a hard-wall potential at x = 0 and x = W, the guiding center
    X = -l_B^2 * k_y must be within the sample.

    The dispersion curves up near the edges, creating chiral edge states.

    Args:
        k_y: Wavevector along edge (rad/m)
        n: Landau level index
        B: Magnetic field (T)
        V_edge: Edge potential strength (J/m^2)
        m_eff: Effective mass (kg)
        W: Sample width (m), if None uses 20*l_B

    Returns:
        Energy (J)
    """
    l_B = magnetic_length(B)
    if W is None:
        W = 20 * l_B

    # Guiding center position
    X = -l_B**2 * k_y

    # Bulk Landau level energy
    E_n = landau_level_energy(n, B, m_eff)

    # Add confining potential (parabolic near edges)
    # V(X) = V_edge * (X/l_B)^2 when X < 0 or X > W
    if np.isscalar(X):
        if X < 0:
            E = E_n + V_edge * (X / l_B)**2
        elif X > W:
            E = E_n + V_edge * ((X - W) / l_B)**2
        else:
            E = E_n
    else:
        E = np.where(X < 0,
                    E_n + V_edge * (X / l_B)**2,
                    np.where(X > W,
                            E_n + V_edge * ((X - W) / l_B)**2,
                            E_n))

    return E


def edge_state_velocity(k_y, n, B, V_edge, m_eff=m_e, W=None, dk=1e6):
    """
    Edge state group velocity v = (1/hbar) * dE/dk.

    Args:
        k_y: Wavevector (rad/m)
        n: Landau level index
        B: Magnetic field (T)
        V_edge: Edge potential strength (J/m^2)
        m_eff: Effective mass (kg)
        W: Sample width (m)
        dk: Finite difference step

    Returns:
        Group velocity (m/s)
    """
    E_plus = edge_state_dispersion(k_y + dk/2, n, B, V_edge, m_eff, W)
    E_minus = edge_state_dispersion(k_y - dk/2, n, B, V_edge, m_eff, W)

    return (E_plus - E_minus) / (hbar * dk)


def hall_conductance(n_filled):
    """
    Quantized Hall conductance.

    sigma_xy = n * e^2 / h

    Args:
        n_filled: Number of filled Landau levels

    Returns:
        Hall conductance (S)
    """
    h = 2 * np.pi * hbar
    return n_filled * e**2 / h


def hall_resistance(n_filled):
    """
    Quantized Hall resistance.

    R_H = h / (n * e^2)

    Args:
        n_filled: Number of filled Landau levels

    Returns:
        Hall resistance (Ohm)
    """
    h = 2 * np.pi * hbar
    return h / (n_filled * e**2)


def lattice_qhe_hamiltonian(Nx, Ny, t, phi):
    """
    Tight-binding Hamiltonian for 2D lattice in magnetic field (Harper-Hofstadter model).

    H = -t sum_{<i,j>} exp(i*A_{ij}) c_i^dag c_j

    Uses Landau gauge: A_y = B*x

    Args:
        Nx: Number of sites in x direction
        Ny: Number of sites in y direction
        t: Hopping parameter (eV)
        phi: Magnetic flux per plaquette in units of flux quantum (phi_0 = h/e)

    Returns:
        Hamiltonian matrix (Nx*Ny x Nx*Ny)
    """
    N = Nx * Ny
    H = np.zeros((N, N), dtype=complex)

    def idx(ix, iy):
        """Convert 2D indices to 1D."""
        return ix * Ny + iy

    for ix in range(Nx):
        for iy in range(Ny):
            i = idx(ix, iy)

            # Hopping in x direction (no phase in Landau gauge)
            if ix < Nx - 1:
                j = idx(ix + 1, iy)
                H[i, j] = -t
                H[j, i] = -t

            # Hopping in y direction (Peierls phase)
            if iy < Ny - 1:
                j = idx(ix, iy + 1)
                phase = 2 * np.pi * phi * ix
                H[i, j] = -t * np.exp(1j * phase)
                H[j, i] = -t * np.exp(-1j * phase)

    return H


def add_disorder(H, W_disorder, seed=None):
    """
    Add Anderson disorder to Hamiltonian.

    Args:
        H: Hamiltonian matrix
        W_disorder: Disorder strength (eV)
        seed: Random seed

    Returns:
        Disordered Hamiltonian
    """
    if seed is not None:
        np.random.seed(seed)

    N = H.shape[0]
    disorder = W_disorder * (np.random.random(N) - 0.5)
    H_disordered = H.copy()
    np.fill_diagonal(H_disordered, np.diag(H) + disorder)

    return H_disordered


def compute_chern_number(Nx, Ny, t, phi, E_fermi, n_k=20):
    """
    Compute Chern number by integrating Berry curvature.

    C = (1/2pi) * integral F dk_x dk_y

    where F is the Berry curvature.

    Args:
        Nx, Ny: System size (for unit cell, should be 1/phi periods)
        t: Hopping parameter
        phi: Flux per plaquette
        E_fermi: Fermi energy
        n_k: Number of k-points

    Returns:
        Chern number (integer)
    """
    # For a proper calculation, we need the magnetic unit cell
    # Here we use a simplified approach based on spectral flow

    # This is a simplified calculation for demonstration
    q = int(1 / phi) if phi > 0 else 1  # Magnetic unit cell size

    if q > 20:  # Too large for full calculation
        return 0

    # Build k-space Hamiltonian for magnetic unit cell
    chern = 0

    # Calculate from edge state counting (simplified)
    # Number of edge states = Chern number
    H = lattice_qhe_hamiltonian(30, q, t, phi)
    E, V = linalg.eigh(H)

    # Count states below E_fermi
    n_filled = np.sum(E < E_fermi)

    # For phi = 1/q, expect Landau-level-like structure
    # First Landau level has Chern = 1

    return int(np.round(n_filled * phi))


def main():
    """Main simulation and visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Parameters
    B = 10  # Tesla
    m_eff = 0.067 * m_e  # GaAs effective mass
    l_B = magnetic_length(B)
    omega_c = cyclotron_frequency(B, m_eff)

    # Plot 1: Edge state dispersion
    ax1 = axes[0, 0]

    # k-range (in units of 1/l_B)
    k_range = np.linspace(-3 / l_B, 3 / l_B, 500)

    # Sample width
    W = 15 * l_B
    V_edge = 0.5 * hbar * omega_c  # Edge potential strength

    # Landau levels with edge dispersion
    n_levels = 4
    colors = plt.cm.viridis(np.linspace(0, 0.8, n_levels))

    for n, color in zip(range(n_levels), colors):
        E = edge_state_dispersion(k_range, n, B, V_edge, m_eff, W)
        E_meV = E / (1.602e-22)  # Convert to meV
        ax1.plot(k_range * l_B, E_meV, color=color, lw=2, label=f'n = {n}')

    # Mark sample boundaries (guiding center positions)
    ax1.axvline(x=0, color='red', linestyle='--', lw=2, alpha=0.7, label='Left edge')
    ax1.axvline(x=-W/l_B, color='blue', linestyle='--', lw=2, alpha=0.7, label='Right edge')

    # Mark bulk region
    ax1.axvspan(-W/l_B, 0, alpha=0.1, color='gray')

    ax1.set_xlabel(r'$k_y \cdot l_B$')
    ax1.set_ylabel('Energy (meV)')
    ax1.set_title('Edge State Dispersion (Landau Gauge)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Add chirality arrows
    ax1.annotate('', xy=(1.5, 25), xytext=(1.5, 15),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax1.text(1.7, 20, 'Right-moving', fontsize=10, color='red')

    ax1.annotate('', xy=(-17, 15), xytext=(-17, 25),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax1.text(-20, 20, 'Left-moving', fontsize=10, color='blue')

    # Plot 2: Hall conductance plateaus
    ax2 = axes[0, 1]

    # Simulate sweeping B field at fixed density
    n_e = 3e15  # Electron density (m^-2)
    B_range = np.linspace(1, 20, 500)

    # Filling factor
    nu = np.array([n_e * 2 * np.pi * hbar / (e * B_val) for B_val in B_range])

    # Hall conductance (quantized plateaus)
    sigma_xy = np.floor(nu) * e**2 / (2 * np.pi * hbar) * 1e6  # In microsiemens

    # Longitudinal resistivity (SdH oscillations with zeros at plateaus)
    # Simplified model: peaks at half-integer filling
    rho_xx = np.abs(np.sin(np.pi * nu)) * 0.5  # Arbitrary units

    ax2_twin = ax2.twinx()

    line1, = ax2.plot(1/B_range, sigma_xy / (e**2 / (2*np.pi*hbar) * 1e6), 'b-', lw=2)
    line2, = ax2_twin.plot(1/B_range, rho_xx, 'r-', lw=2, alpha=0.7)

    ax2.set_xlabel('1/B (1/T)')
    ax2.set_ylabel(r'$\sigma_{xy}$ (e^2/h)', color='blue')
    ax2_twin.set_ylabel(r'$\rho_{xx}$ (arb. units)', color='red')

    ax2.set_title('Integer Quantum Hall Effect')
    ax2.grid(True, alpha=0.3)

    # Mark plateaus
    for n in range(1, 6):
        ax2.axhline(y=n, color='gray', linestyle=':', alpha=0.5)
        ax2.text(0.02, n + 0.1, f'n={n}', fontsize=9)

    ax2.set_ylim(0, 6)

    # Plot 3: Tight-binding model spectrum
    ax3 = axes[1, 0]

    Nx, Ny = 30, 30
    t = 1.0  # hopping in eV (arbitrary units)
    phi = 1/10  # Flux quantum per plaquette (1/q)

    H = lattice_qhe_hamiltonian(Nx, Ny, t, phi)
    E, V = linalg.eigh(H)

    # Sort by energy
    idx_sorted = np.argsort(E)
    E_sorted = E[idx_sorted]

    ax3.plot(range(len(E_sorted)), E_sorted, 'b.', markersize=2)

    ax3.set_xlabel('State index')
    ax3.set_ylabel('Energy (t)')
    ax3.set_title(f'Harper-Hofstadter Spectrum (phi = 1/{int(1/phi)})')
    ax3.grid(True, alpha=0.3)

    # Mark Landau-level-like gaps
    # In the Hofstadter model, there are q-1 gaps
    ax3.text(0.7, 0.9, f'Flux: phi = {phi:.2f} phi_0\n'
                       f'Expect {int(1/phi)} sub-bands',
            transform=ax3.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # Plot 4: Edge state probability density
    ax4 = axes[1, 1]

    # Use the tight-binding model to show edge states
    # Find states in a gap
    E_mid = (E_sorted[Nx*Ny//2 - 10] + E_sorted[Nx*Ny//2 + 10]) / 2

    # Find states near this energy
    gap_states = np.where(np.abs(E - E_mid) < 0.1 * t)[0]

    if len(gap_states) > 0:
        # Plot probability density summed over y
        x_prob = np.zeros(Nx)

        for state_idx in gap_states[:min(5, len(gap_states))]:
            psi = V[:, state_idx]
            psi_sq = np.abs(psi)**2

            # Reshape to 2D and sum over y
            psi_2d = psi_sq.reshape(Nx, Ny)
            x_prob += np.sum(psi_2d, axis=1)

        x_prob /= np.max(x_prob)

        ax4.bar(range(Nx), x_prob, color='blue', alpha=0.7)
        ax4.set_xlabel('x position')
        ax4.set_ylabel('Probability density (summed over y)')
        ax4.set_title('Edge State Localization')

        # Mark edges
        ax4.axvspan(0, 3, alpha=0.3, color='red', label='Left edge')
        ax4.axvspan(Nx-3, Nx, alpha=0.3, color='green', label='Right edge')
        ax4.legend()
    else:
        # Alternative: show bulk vs edge
        ax4.text(0.5, 0.5, 'No gap states found\n(try different parameters)',
                transform=ax4.transAxes, ha='center', fontsize=12)

    ax4.grid(True, alpha=0.3)

    # Add quantization info
    R_H_1 = hall_resistance(1)
    R_H_2 = hall_resistance(2)

    fig.text(0.5, 0.02, f'von Klitzing constant: h/e^2 = {R_H_1:.6f} Ohm  |  '
                        f'R_H(n=1) = {R_H_1:.1f} Ohm, R_H(n=2) = {R_H_2:.1f} Ohm',
            ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.suptitle('Integer Quantum Hall Effect Edge States\n'
                 r'Chiral edge channels, $\sigma_{xy} = n \cdot e^2/h$',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'integer_qhe_edge.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'integer_qhe_edge.png')}")


if __name__ == "__main__":
    main()
