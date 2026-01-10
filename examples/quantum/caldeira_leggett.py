"""
Experiment 178: Caldeira-Leggett Decoherence Model

Demonstrates the Caldeira-Leggett model of quantum decoherence for a particle
coupled to a bath of harmonic oscillators.

Physics:
    The Caldeira-Leggett model describes a quantum system coupled to an environment:

    H = H_S + H_B + H_SB

    H_S = p^2/(2M) + V(x)              (System)
    H_B = sum_n (p_n^2/(2m_n) + m_n*omega_n^2 * x_n^2/2)  (Bath)
    H_SB = -x * sum_n c_n * x_n         (Coupling)

    After tracing out the bath, the system density matrix evolves with:
    - Dissipation (friction): proportional to velocity
    - Decoherence: spatial superpositions decay exponentially

    Key predictions:
    1. Decoherence rate: Gamma_dec ~ (delta_x)^2 / lambda_th^2 * Gamma_relax
       where lambda_th = hbar / sqrt(M * k_B * T) is thermal de Broglie wavelength

    2. Macroscopic superpositions decohere extremely fast
       (explains emergence of classical behavior)

    3. Quantum-to-classical transition
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm


def caldeira_leggett_master_equation(rho, x_grid, p_op, gamma, D, hbar=1.0):
    """
    Caldeira-Leggett master equation in position representation.

    d rho/dt = -i/hbar [H, rho] - i*gamma/hbar [x, {p, rho}] - D/hbar^2 [x, [x, rho]]

    The last term causes decoherence: decay of off-diagonal elements.

    Simplified to just the decoherence term for demonstration:
    d rho(x, x')/dt = -D/hbar^2 * (x - x')^2 * rho(x, x')

    Args:
        rho: Density matrix in position basis
        x_grid: Position grid
        p_op: Momentum operator (unused in simplified version)
        gamma: Friction coefficient
        D: Diffusion coefficient (related to temperature)
        hbar: Reduced Planck constant

    Returns:
        drho/dt
    """
    N = len(x_grid)

    # Decoherence term: -D/hbar^2 * (x - x')^2 * rho(x, x')
    X, Xp = np.meshgrid(x_grid, x_grid)
    decoherence_rate = D / hbar**2 * (X - Xp)**2

    drho = -decoherence_rate * rho

    return drho


def evolve_caldeira_leggett(rho0, x_grid, gamma, D, t_final, dt, hbar=1.0):
    """
    Evolve density matrix under Caldeira-Leggett decoherence.

    Args:
        rho0: Initial density matrix
        x_grid: Position grid
        gamma: Friction
        D: Diffusion coefficient
        t_final: Final time
        dt: Time step
        hbar: Reduced Planck constant

    Returns:
        times, rho_history
    """
    times = [0]
    rho_history = [rho0.copy()]

    rho = rho0.copy()
    t = 0

    while t < t_final:
        drho = caldeira_leggett_master_equation(rho, x_grid, None, gamma, D, hbar)
        rho = rho + drho * dt
        t += dt

        times.append(t)
        rho_history.append(rho.copy())

    return np.array(times), rho_history


def gaussian_wavepacket(x, x0, sigma, k0=0):
    """
    Gaussian wavepacket.

    psi(x) = (2*pi*sigma^2)^(-1/4) * exp(-(x-x0)^2/(4*sigma^2) + i*k0*x)

    Args:
        x: Position array
        x0: Center position
        sigma: Width
        k0: Initial momentum/hbar

    Returns:
        Normalized wavefunction
    """
    psi = (2 * np.pi * sigma**2)**(-0.25) * np.exp(-(x - x0)**2 / (4 * sigma**2) + 1j * k0 * x)
    return psi


def superposition_state(x, x1, x2, sigma, phase=0):
    """
    Spatial superposition of two Gaussian wavepackets.

    |psi> = (|x1> + e^(i*phi)|x2>) / sqrt(2)

    Args:
        x: Position array
        x1, x2: Center positions
        sigma: Width
        phase: Relative phase

    Returns:
        Superposition wavefunction
    """
    psi1 = gaussian_wavepacket(x, x1, sigma)
    psi2 = gaussian_wavepacket(x, x2, sigma)
    psi = (psi1 + np.exp(1j * phase) * psi2) / np.sqrt(2)
    return psi / np.linalg.norm(psi)


def coherence_measure(rho, x_grid, x1, x2, width=0.5):
    """
    Measure coherence between two spatial regions.

    Integrates |rho(x, x')| in off-diagonal regions.

    Args:
        rho: Density matrix
        x_grid: Position grid
        x1, x2: Centers of the two regions
        width: Region width

    Returns:
        Coherence measure
    """
    X, Xp = np.meshgrid(x_grid, x_grid)
    dx = x_grid[1] - x_grid[0]

    # Mask for off-diagonal region
    mask = ((np.abs(X - x1) < width) & (np.abs(Xp - x2) < width)) | \
           ((np.abs(X - x2) < width) & (np.abs(Xp - x1) < width))

    return np.sum(np.abs(rho[mask])) * dx**2


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Parameters
    N = 100
    x_max = 10
    x_grid = np.linspace(-x_max, x_max, N)
    dx = x_grid[1] - x_grid[0]

    sigma = 1.0  # Wavepacket width
    separation = 4.0  # Distance between superposed states
    x1, x2 = -separation/2, separation/2

    hbar = 1.0

    # Initial superposition state
    psi0 = superposition_state(x_grid, x1, x2, sigma)
    rho0 = np.outer(psi0, np.conj(psi0))

    # ===== Plot 1: Initial density matrix =====
    ax1 = axes[0, 0]

    im1 = ax1.imshow(np.abs(rho0), extent=[-x_max, x_max, -x_max, x_max],
                    origin='lower', cmap='hot', aspect='auto')
    ax1.set_xlabel("x'")
    ax1.set_ylabel("x")
    ax1.set_title('Initial Density Matrix |rho(x, x\')|\n(Cat state: superposition)')
    plt.colorbar(im1, ax=ax1)

    # Mark the off-diagonal coherence peaks
    ax1.scatter([x1, x2], [x2, x1], color='cyan', marker='o', s=100, label='Coherence peaks')
    ax1.legend()

    # ===== Plot 2: Decoherence dynamics =====
    ax2 = axes[0, 1]

    D_values = [0.1, 0.5, 1.0, 2.0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(D_values)))
    t_max = 5.0
    dt = 0.01

    for D, color in zip(D_values, colors):
        times, rho_history = evolve_caldeira_leggett(rho0, x_grid, gamma=0.1, D=D, t_final=t_max, dt=dt, hbar=hbar)

        # Track coherence decay
        coherences = [coherence_measure(rho, x_grid, x1, x2, sigma) for rho in rho_history]

        ax2.plot(times, coherences / coherences[0], color=color, lw=2, label=f'D = {D}')

        # Analytical: exp(-D * (x1-x2)^2 * t / hbar^2)
        dec_rate = D * separation**2 / hbar**2
        ax2.plot(times, np.exp(-dec_rate * times), '--', color=color, lw=1, alpha=0.7)

    ax2.set_xlabel('Time')
    ax2.set_ylabel('Normalized Coherence')
    ax2.set_title('Decoherence of Spatial Superposition\n(solid: numerical, dashed: analytical)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    ax2.set_ylim(1e-3, 1)

    # ===== Plot 3: Density matrix at different times =====
    ax3 = axes[1, 0]

    D = 0.5
    times_snapshot = [0, 0.5, 1.0, 2.0]

    for i, t_snap in enumerate(times_snapshot):
        times, rho_history = evolve_caldeira_leggett(rho0, x_grid, gamma=0.1, D=D, t_final=t_snap + 0.01, dt=dt, hbar=hbar)
        rho_final = rho_history[-1]

        # Plot diagonal (probability density)
        prob = np.real(np.diag(rho_final))
        ax3.plot(x_grid, prob + i * 0.3, lw=2, label=f't = {t_snap}')

    ax3.set_xlabel('Position x')
    ax3.set_ylabel('P(x) + offset')
    ax3.set_title('Probability Density Evolution\n(Populations preserved, coherence lost)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # ===== Plot 4: Decoherence rate vs separation =====
    ax4 = axes[1, 1]

    D = 1.0
    separations = np.linspace(1, 8, 20)
    decoherence_rates = []

    for sep in separations:
        x1_temp, x2_temp = -sep/2, sep/2
        psi_temp = superposition_state(x_grid, x1_temp, x2_temp, sigma)
        rho_temp = np.outer(psi_temp, np.conj(psi_temp))

        # Measure decoherence rate
        times, rho_hist = evolve_caldeira_leggett(rho_temp, x_grid, gamma=0.1, D=D, t_final=2.0, dt=dt, hbar=hbar)
        coherences = [coherence_measure(rho, x_grid, x1_temp, x2_temp, sigma) for rho in rho_hist]

        # Fit exponential decay rate
        if coherences[0] > 0 and coherences[-1] > 0:
            rate = -np.log(coherences[-1] / coherences[0]) / times[-1]
            decoherence_rates.append(rate)
        else:
            decoherence_rates.append(np.nan)

    # Analytical prediction: Gamma_dec = D * (Delta_x)^2 / hbar^2
    analytical_rates = D * separations**2 / hbar**2

    ax4.plot(separations, decoherence_rates, 'bo-', lw=2, markersize=5, label='Numerical')
    ax4.plot(separations, analytical_rates, 'r--', lw=2, label=r'$\Gamma = D \cdot (\Delta x)^2 / \hbar^2$')

    ax4.set_xlabel('Superposition separation (Delta x)')
    ax4.set_ylabel('Decoherence rate')
    ax4.set_title('Decoherence Rate vs Superposition Size\n(Larger = faster decoherence)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Add macroscopic scale annotation
    ax4.annotate('Macroscopic objects\ndecohere instantly!',
                 xy=(6, analytical_rates[15]), xytext=(4, analytical_rates[15] + 10),
                 arrowprops=dict(arrowstyle='->', color='gray'),
                 fontsize=10, color='gray')

    plt.suptitle('Caldeira-Leggett Model of Quantum Decoherence\n'
                 'Emergence of Classicality from Environmental Interaction',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'caldeira_leggett.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'caldeira_leggett.png')}")

    # Print results
    print("\n=== Caldeira-Leggett Decoherence Results ===")
    print(f"\nParameters:")
    print(f"  Wavepacket width sigma = {sigma}")
    print(f"  Initial separation = {separation}")

    print(f"\nDecoherence rates for D = 1.0:")
    for sep, rate in zip(separations[::4], decoherence_rates[::4]):
        if not np.isnan(rate):
            print(f"  Delta_x = {sep:.1f}: Gamma = {rate:.3f} (T_dec = {1/rate:.3f})")

    print(f"\nKey insight: Decoherence rate scales as (Delta_x)^2")
    print(f"This explains why macroscopic superpositions are never observed!")

    # Estimate for macroscopic object
    print(f"\nMacroscopic estimate (1 gram object, 1 mm separation):")
    print(f"  At room temperature: T_dec ~ 10^-40 seconds!")


if __name__ == "__main__":
    main()
