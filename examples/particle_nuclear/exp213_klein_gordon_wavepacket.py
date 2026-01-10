"""
Experiment 213: Klein-Gordon 1D Wavepackets

Demonstrates relativistic wave propagation using the Klein-Gordon equation.
Shows dispersion, group velocity, and negative energy components.

Physics:
- (□ + m²)φ = 0, where □ = ∂²/∂t² - ∇²
- Dispersion: ω² = k² + m² (natural units, c = ℏ = 1)
- Group velocity: v_g = k/ω < 1
- Phase velocity: v_p = ω/k > 1
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.qft import ScalarField


def klein_gordon_dispersion(k, m):
    """Dispersion relation: ω = √(k² + m²)"""
    return np.sqrt(k**2 + m**2)


def group_velocity(k, m):
    """Group velocity: v_g = dω/dk = k/ω"""
    omega = klein_gordon_dispersion(k, m)
    return k / omega


def phase_velocity(k, m):
    """Phase velocity: v_p = ω/k"""
    omega = klein_gordon_dispersion(k, m)
    return omega / k


def gaussian_wavepacket(x, x0, sigma, k0, m, t):
    """
    Gaussian wavepacket solution to Klein-Gordon equation.

    At t=0: φ(x,0) = exp(-(x-x0)²/(4σ²)) × exp(ik₀x)

    Time evolution via Fourier transform.
    """
    # Fourier transform of initial condition
    dk = 0.02
    k = np.arange(-20, 20, dk)

    # Gaussian in k-space centered at k0
    psi_k = np.exp(-(k - k0)**2 * sigma**2) * np.exp(-1j * k * x0)

    # Time evolution
    omega = klein_gordon_dispersion(k, m)

    # Reconstruct wavepacket
    phi = np.zeros_like(x, dtype=complex)
    for i, xi in enumerate(x):
        phi[i] = np.sum(psi_k * np.exp(1j * (k * xi - omega * t))) * dk

    # Normalize
    phi = phi / np.sqrt(2 * np.pi)
    return phi


def evolve_kg_finite_difference(phi0, pi0, x, m, dt, n_steps):
    """
    Evolve Klein-Gordon equation using finite differences.

    ∂²φ/∂t² = ∂²φ/∂x² - m²φ

    Uses leapfrog method.
    """
    dx = x[1] - x[0]
    n = len(x)

    phi = phi0.copy()
    pi = pi0.copy()  # ∂φ/∂t

    history = [phi.copy()]

    for _ in range(n_steps):
        # Laplacian with periodic BC
        laplacian = np.zeros_like(phi)
        laplacian[1:-1] = (phi[2:] - 2*phi[1:-1] + phi[:-2]) / dx**2
        laplacian[0] = (phi[1] - 2*phi[0] + phi[-1]) / dx**2
        laplacian[-1] = (phi[0] - 2*phi[-1] + phi[-2]) / dx**2

        # Update π (half step)
        pi = pi + 0.5 * dt * (laplacian - m**2 * phi)

        # Update φ
        phi = phi + dt * pi

        # Recalculate laplacian
        laplacian[1:-1] = (phi[2:] - 2*phi[1:-1] + phi[:-2]) / dx**2
        laplacian[0] = (phi[1] - 2*phi[0] + phi[-1]) / dx**2
        laplacian[-1] = (phi[0] - 2*phi[-1] + phi[-2]) / dx**2

        # Update π (half step)
        pi = pi + 0.5 * dt * (laplacian - m**2 * phi)

        history.append(phi.copy())

    return history


def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Parameters (natural units: c = ℏ = 1)
    m = 1.0  # Mass

    # Plot 1: Dispersion relation
    ax = axes[0, 0]

    k = np.linspace(-5, 5, 200)

    for mass in [0, 0.5, 1.0, 2.0]:
        omega = klein_gordon_dispersion(k, mass)
        ax.plot(k, omega, lw=2, label=f'm = {mass}')

    ax.plot(k, np.abs(k), 'k--', lw=1, label='Massless (ω = |k|)')
    ax.set_xlabel('Wave vector k')
    ax.set_ylabel('Angular frequency ω')
    ax.set_title('Klein-Gordon Dispersion\nω² = k² + m²')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 5)
    ax.set_ylim(0, 6)

    # Plot 2: Group and phase velocities
    ax = axes[0, 1]

    k_pos = np.linspace(0.1, 5, 100)

    for mass, color in [(0.5, 'b'), (1.0, 'r'), (2.0, 'g')]:
        vg = group_velocity(k_pos, mass)
        vp = phase_velocity(k_pos, mass)

        ax.plot(k_pos, vg, '-', color=color, lw=2, label=f'v_g (m={mass})')
        ax.plot(k_pos, vp, '--', color=color, lw=2, label=f'v_p (m={mass})')

    ax.axhline(y=1, color='k', linestyle=':', alpha=0.5, label='c = 1')
    ax.set_xlabel('Wave vector k')
    ax.set_ylabel('Velocity (c = 1)')
    ax.set_title('Group and Phase Velocities\nv_g < c < v_p')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 3)

    # Plot 3: Wavepacket propagation
    ax = axes[0, 2]

    x = np.linspace(-50, 50, 1000)
    x0 = -20
    sigma = 3.0
    k0 = 2.0

    times = [0, 10, 20, 30]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(times)))

    for t, color in zip(times, colors):
        phi = gaussian_wavepacket(x, x0, sigma, k0, m, t)
        ax.plot(x, np.abs(phi)**2, '-', color=color, lw=2, label=f't = {t}')

        # Mark expected peak position from group velocity
        vg = group_velocity(k0, m)
        x_peak = x0 + vg * t
        ax.axvline(x=x_peak, color=color, linestyle='--', alpha=0.3)

    ax.set_xlabel('Position x')
    ax.set_ylabel('|φ|²')
    ax.set_title(f'Wavepacket Propagation\nk₀ = {k0}, m = {m}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Dispersion and spreading
    ax = axes[1, 0]

    # Compare massless and massive
    for mass, label in [(0.0, 'Massless'), (1.0, 'Massive (m=1)')]:
        x0_d = -15
        sigma_d = 2.0
        k0_d = 3.0

        phi_t0 = gaussian_wavepacket(x, x0_d, sigma_d, k0_d, mass, t=0)
        phi_t20 = gaussian_wavepacket(x, x0_d, sigma_d, k0_d, mass, t=20)

        ax.plot(x, np.abs(phi_t0)**2, '--', lw=1, alpha=0.5)
        ax.plot(x, np.abs(phi_t20)**2, '-', lw=2, label=f'{label}, t=20')

    ax.set_xlabel('Position x')
    ax.set_ylabel('|φ|²')
    ax.set_title('Wavepacket Spreading\nMassive vs Massless')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Finite difference evolution
    ax = axes[1, 1]

    # Initial condition
    x_fd = np.linspace(-30, 30, 300)
    dx = x_fd[1] - x_fd[0]

    # Gaussian wavepacket
    x0_fd = -10
    sigma_fd = 2.0
    k0_fd = 2.0

    phi0_fd = np.exp(-(x_fd - x0_fd)**2 / (4 * sigma_fd**2)) * np.exp(1j * k0_fd * x_fd)
    phi0_fd = np.real(phi0_fd)

    # Initial velocity (from group velocity)
    omega0 = klein_gordon_dispersion(k0_fd, m)
    pi0_fd = omega0 * np.imag(np.exp(-(x_fd - x0_fd)**2 / (4 * sigma_fd**2)) *
                               np.exp(1j * k0_fd * x_fd))

    # Evolve
    dt = 0.02
    n_steps = 500

    history = evolve_kg_finite_difference(phi0_fd, pi0_fd, x_fd, m, dt, n_steps)

    # Plot snapshots
    times_fd = [0, 100, 300, 500]
    colors = plt.cm.plasma(np.linspace(0, 0.8, len(times_fd)))

    for t_idx, color in zip(times_fd, colors):
        ax.plot(x_fd, history[t_idx], '-', color=color, lw=2,
                label=f't = {t_idx * dt:.1f}')

    ax.set_xlabel('Position x')
    ax.set_ylabel('φ(x,t)')
    ax.set_title('Finite Difference Evolution\n(Real Klein-Gordon)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Summary
    ax = axes[1, 2]
    ax.axis('off')

    vg_example = group_velocity(k0, m)
    vp_example = phase_velocity(k0, m)

    summary = f"""
Klein-Gordon Equation
=====================

Equation (c = ℏ = 1):
  (∂²/∂t² - ∇² + m²)φ = 0
  or: (□ + m²)φ = 0

Dispersion Relation:
  ω² = k² + m²
  ω = √(k² + m²)

Velocities:
  Group: v_g = dω/dk = k/√(k² + m²) < 1
  Phase: v_p = ω/k = √(k² + m²)/k > 1

  v_g × v_p = 1 (always)

For k = {k0}, m = {m}:
  v_g = {vg_example:.3f} c
  v_p = {vp_example:.3f} c

Wavepacket Behavior:
  • Travels at group velocity
  • Spreads due to dispersion
  • Massive: more dispersion
  • Massless: no spreading

Energy Density:
  ρ = (1/2)[π² + (∇φ)² + m²φ²]

Issues with KG Equation:
  • Negative probability density
  • Negative energy solutions
  → Leads to QFT interpretation

Resolution:
  • φ is a field operator
  • Particles and antiparticles
  • Probability → charge density
"""

    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Experiment 213: Klein-Gordon 1D Wavepackets\n'
                 'Relativistic Scalar Wave Propagation', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'exp213_klein_gordon_wavepacket.png'),
                dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/exp213_klein_gordon_wavepacket.png")


if __name__ == "__main__":
    main()
