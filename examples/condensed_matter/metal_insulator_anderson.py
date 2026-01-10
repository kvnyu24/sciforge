"""
Experiment 230: Anderson Localization / Metal-Insulator Transition

Demonstrates Anderson localization in disordered systems:
- 1D tight binding with disorder
- Localization length vs disorder strength
- Transmission coefficient vs system size
- Mobility edge concept
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg


def anderson_hamiltonian_1d(N, t, W, seed=None):
    """
    1D Anderson model Hamiltonian.

    H = sum_i epsilon_i |i><i| + t * sum_<i,j> (|i><j| + h.c.)

    where epsilon_i are random on-site energies uniformly distributed in [-W/2, W/2].

    Args:
        N: Number of sites
        t: Hopping parameter
        W: Disorder strength (width of random distribution)
        seed: Random seed for reproducibility

    Returns:
        N x N Hamiltonian matrix
    """
    if seed is not None:
        np.random.seed(seed)

    H = np.zeros((N, N))

    # Random on-site energies
    epsilon = W * (np.random.random(N) - 0.5)
    np.fill_diagonal(H, epsilon)

    # Hopping terms (nearest neighbor)
    for i in range(N - 1):
        H[i, i+1] = -t
        H[i+1, i] = -t

    return H


def localization_length_lyapunov(E, t, W, N_samples=1000, seed=None):
    """
    Calculate localization length using Lyapunov exponent (transfer matrix method).

    For 1D systems, xi = 1/gamma where gamma is the Lyapunov exponent.

    Args:
        E: Energy
        t: Hopping parameter
        W: Disorder strength
        N_samples: Number of sites for averaging
        seed: Random seed

    Returns:
        Localization length xi
    """
    if seed is not None:
        np.random.seed(seed)

    # Transfer matrix method
    gamma = 0  # Lyapunov exponent

    # Random on-site energies
    epsilon = W * (np.random.random(N_samples) - 0.5)

    # Initialize
    psi_prev = 1.0
    psi_curr = 1.0

    sum_log = 0

    for i in range(1, N_samples):
        # Recursion: psi_{i+1} = [(E - epsilon_i)/t] * psi_i - psi_{i-1}
        psi_next = ((E - epsilon[i]) / t) * psi_curr - psi_prev

        # Normalize to prevent overflow
        norm = np.abs(psi_next)
        if norm > 0:
            sum_log += np.log(norm)
            psi_next /= norm

        psi_prev = psi_curr / norm if norm > 0 else psi_curr
        psi_curr = psi_next

    gamma = sum_log / N_samples

    # Localization length
    if gamma > 0:
        return 1 / gamma
    else:
        return np.inf


def transmission_coefficient(E, t, W, N, seed=None):
    """
    Calculate transmission coefficient through 1D disordered chain.

    Uses the Green's function method.

    Args:
        E: Energy
        t: Hopping parameter
        W: Disorder strength
        N: System size
        seed: Random seed

    Returns:
        Transmission coefficient T
    """
    if seed is not None:
        np.random.seed(seed)

    # Build Hamiltonian (use complex dtype for self-energies)
    H = anderson_hamiltonian_1d(N, t, W, seed).astype(complex)

    # Self-energies for semi-infinite leads
    # Sigma = -t * exp(i*k*a) where E = -2t*cos(k*a)
    # For |E| < 2t, k is real
    if np.abs(E) < 2 * t:
        cos_ka = -E / (2 * t)
        sin_ka = np.sqrt(1 - cos_ka**2)
        # Self-energy
        sigma = -t * (cos_ka + 1j * sin_ka)
    else:
        # Evanescent mode
        return 0.0

    # Add self-energies to first and last sites
    H[0, 0] += sigma
    H[N-1, N-1] += sigma

    # Green's function G = (E - H)^(-1)
    try:
        G = linalg.inv((E + 1e-10j) * np.eye(N) - H)
    except linalg.LinAlgError:
        return 0.0

    # Transmission: T = Gamma_L * |G_{1N}|^2 * Gamma_R
    # where Gamma = -2 * Im(Sigma)
    gamma = -2 * np.imag(sigma)

    T = gamma**2 * np.abs(G[0, N-1])**2

    return min(T, 1.0)  # Clip to physical range


def inverse_participation_ratio(psi):
    """
    Calculate inverse participation ratio (IPR).

    IPR = sum_i |psi_i|^4

    For extended state: IPR ~ 1/N
    For localized state: IPR ~ 1/xi (localization length)

    Args:
        psi: Wavefunction (normalized)

    Returns:
        IPR
    """
    psi_normalized = psi / np.sqrt(np.sum(np.abs(psi)**2))
    return np.sum(np.abs(psi_normalized)**4)


def participation_ratio(psi):
    """
    Participation ratio PR = 1/IPR.

    Measures the number of sites the wavefunction extends over.

    Args:
        psi: Wavefunction (normalized)

    Returns:
        PR
    """
    ipr = inverse_participation_ratio(psi)
    return 1 / ipr if ipr > 0 else 0


def analyze_eigenstates(H):
    """
    Analyze all eigenstates of a Hamiltonian.

    Args:
        H: Hamiltonian matrix

    Returns:
        energies: Eigenvalues
        iprs: Inverse participation ratios
        prs: Participation ratios
    """
    energies, eigenvectors = linalg.eigh(H)

    iprs = np.array([inverse_participation_ratio(eigenvectors[:, i])
                     for i in range(len(energies))])
    prs = 1 / iprs

    return energies, iprs, prs


def thouless_conductance(E, t, W, N, n_disorder=10, seed=None):
    """
    Calculate dimensionless Thouless conductance.

    g = delta_E / Delta

    where delta_E is the level sensitivity to boundary conditions
    and Delta is the mean level spacing.

    Args:
        E: Energy (not used directly, averaged over)
        t: Hopping parameter
        W: Disorder strength
        N: System size
        n_disorder: Number of disorder realizations
        seed: Random seed

    Returns:
        Thouless conductance
    """
    if seed is not None:
        np.random.seed(seed)

    g_values = []

    for _ in range(n_disorder):
        # Open boundary conditions
        H_open = anderson_hamiltonian_1d(N, t, W)
        E_open = linalg.eigvalsh(H_open)

        # Periodic boundary conditions
        H_periodic = H_open.copy()
        H_periodic[0, N-1] = -t
        H_periodic[N-1, 0] = -t
        E_periodic = linalg.eigvalsh(H_periodic)

        # Level shift
        delta_E = np.mean(np.abs(E_open - E_periodic))

        # Mean level spacing
        Delta = (E_open[-1] - E_open[0]) / N

        if Delta > 0:
            g_values.append(delta_E / Delta)

    return np.mean(g_values) if g_values else 0


def main():
    """Main simulation and visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Parameters
    t = 1.0  # Hopping parameter

    # Plot 1: Localization length vs disorder strength
    ax1 = axes[0, 0]

    W_range = np.linspace(0.1, 8, 30)
    E_values = [0.0, 0.5, 1.0, 1.5]  # Different energies
    colors = ['blue', 'green', 'orange', 'red']

    for E, color in zip(E_values, colors):
        xi_values = []
        for W in W_range:
            xi = localization_length_lyapunov(E, t, W, N_samples=5000, seed=42)
            xi_values.append(min(xi, 1000))  # Cap for visualization

        ax1.semilogy(W_range, xi_values, 'o-', color=color, lw=2,
                    markersize=4, label=f'E = {E}')

    # Theoretical: xi ~ 105 * (t/W)^2 for W << t at band center
    W_theory = np.linspace(0.5, 4, 50)
    xi_theory = 105 * (t / W_theory)**2
    ax1.semilogy(W_theory, xi_theory, 'k--', lw=2, alpha=0.5,
                label=r'Theory: $\xi \sim (t/W)^2$')

    ax1.set_xlabel('Disorder Strength W/t')
    ax1.set_ylabel('Localization Length (lattice units)')
    ax1.set_title('Localization Length vs Disorder')
    ax1.legend()
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_ylim(1, 1000)

    # Plot 2: Transmission vs system size (exponential decay)
    ax2 = axes[0, 1]

    N_range = np.arange(10, 200, 10)
    W_values = [1.0, 2.0, 4.0]
    E = 0.0  # Band center

    for W, color in zip(W_values, ['blue', 'green', 'red']):
        T_avg = []
        for N in N_range:
            # Average over disorder realizations
            T_samples = []
            for seed in range(20):
                T = transmission_coefficient(E, t, W, N, seed=seed)
                T_samples.append(T)
            T_avg.append(np.mean(T_samples))

        ax2.semilogy(N_range, T_avg, 'o-', color=color, lw=2,
                    markersize=4, label=f'W = {W}')

    ax2.set_xlabel('System Size N')
    ax2.set_ylabel('Transmission Coefficient T')
    ax2.set_title('Transmission vs System Size (E = 0)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')

    # Add exponential decay guide
    ax2.text(0.6, 0.8, r'$T \sim e^{-2N/\xi}$',
            transform=ax2.transAxes, fontsize=12,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot 3: Eigenstate localization (IPR vs energy)
    ax3 = axes[1, 0]

    N = 200
    W_cases = [0.5, 2.0, 6.0]

    for W, color in zip(W_cases, ['blue', 'green', 'red']):
        H = anderson_hamiltonian_1d(N, t, W, seed=42)
        energies, iprs, prs = analyze_eigenstates(H)

        # Normalize by N for comparison
        ax3.scatter(energies, prs / N, s=10, c=color, alpha=0.5,
                   label=f'W = {W}')

    ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5,
               label='Extended (PR = N)')

    ax3.set_xlabel('Energy')
    ax3.set_ylabel('Participation Ratio / N')
    ax3.set_title('Eigenstate Localization (N = 200)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.1)

    # Mark band edges
    ax3.axvline(x=-2*t, color='black', linestyle=':', alpha=0.5)
    ax3.axvline(x=2*t, color='black', linestyle=':', alpha=0.5)

    # Plot 4: Wavefunction visualization
    ax4 = axes[1, 1]

    N = 100
    x = np.arange(N)

    # Clean system (small disorder)
    W_clean = 0.5
    H_clean = anderson_hamiltonian_1d(N, t, W_clean, seed=42)
    E_clean, V_clean = linalg.eigh(H_clean)

    # Strongly disordered system
    W_dirty = 5.0
    H_dirty = anderson_hamiltonian_1d(N, t, W_dirty, seed=42)
    E_dirty, V_dirty = linalg.eigh(H_dirty)

    # Plot mid-band states
    mid_idx = N // 2

    psi_clean = np.abs(V_clean[:, mid_idx])**2
    psi_dirty = np.abs(V_dirty[:, mid_idx])**2

    ax4.plot(x, psi_clean / np.max(psi_clean), 'b-', lw=2,
            label=f'Weak disorder (W={W_clean})')
    ax4.plot(x, psi_dirty / np.max(psi_dirty) + 1.2, 'r-', lw=2,
            label=f'Strong disorder (W={W_dirty})')

    ax4.fill_between(x, 0, psi_clean / np.max(psi_clean), alpha=0.3, color='blue')
    ax4.fill_between(x, 1.2, psi_dirty / np.max(psi_dirty) + 1.2, alpha=0.3, color='red')

    ax4.set_xlabel('Site Index')
    ax4.set_ylabel(r'$|\psi|^2$ (normalized)')
    ax4.set_title('Wavefunction Localization')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    ax4.text(70, 0.5, 'Extended', fontsize=11, color='blue')
    ax4.text(70, 1.7, 'Localized', fontsize=11, color='red')

    # Add localization length estimates
    pr_clean = participation_ratio(V_clean[:, mid_idx])
    pr_dirty = participation_ratio(V_dirty[:, mid_idx])
    ax4.text(0.02, 0.95, f'PR(weak) = {pr_clean:.1f}\nPR(strong) = {pr_dirty:.1f}',
            transform=ax4.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.suptitle('Anderson Localization in 1D\n'
                 'All states localized in 1D for any disorder W > 0',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'metal_insulator_anderson.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'metal_insulator_anderson.png')}")


if __name__ == "__main__":
    main()
