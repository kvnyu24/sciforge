"""
Experiment 216: Lattice Field Energy Conservation

Demonstrates energy conservation in lattice field theory simulations.
Uses discretized scalar field with symplectic integrators.

Physics:
- Lattice action: S = Σ_n [½(φ_n+1 - φ_n)² + ½m²φ_n² + (λ/4!)φ_n⁴]
- Energy: E = Σ_n [½π_n² + ½(∇φ)² + V(φ)]
- Symplectic integrator preserves phase space volume
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def lattice_potential(phi, m, lam):
    """Lattice potential: V = ½m²φ² + (λ/4!)φ⁴"""
    return 0.5 * m**2 * phi**2 + (lam / 24) * phi**4


def lattice_force(phi, m, lam):
    """Force: -dV/dφ = -m²φ - (λ/6)φ³"""
    return -m**2 * phi - (lam / 6) * phi**3


def compute_energy(phi, pi, m, lam, dx):
    """
    Compute total energy on lattice.

    E = Σ_n [½π_n² + ½(∇φ)² + V(φ)]
    """
    n = len(phi)

    # Kinetic energy
    kinetic = 0.5 * np.sum(pi**2) * dx

    # Gradient energy (periodic BC)
    gradient = 0
    for i in range(n):
        dphi = (phi[(i+1) % n] - phi[i]) / dx
        gradient += 0.5 * dphi**2 * dx

    # Potential energy
    potential = np.sum(lattice_potential(phi, m, lam)) * dx

    return kinetic, gradient, potential


def euler_step(phi, pi, m, lam, dx, dt):
    """Simple Euler integrator (NOT symplectic)."""
    # Force from lattice Laplacian and potential
    n = len(phi)
    laplacian = np.zeros(n)
    for i in range(n):
        laplacian[i] = (phi[(i+1) % n] - 2*phi[i] + phi[(i-1) % n]) / dx**2

    force = laplacian + lattice_force(phi, m, lam)

    phi_new = phi + dt * pi
    pi_new = pi + dt * force

    return phi_new, pi_new


def leapfrog_step(phi, pi, m, lam, dx, dt):
    """Leapfrog (Stormer-Verlet) symplectic integrator."""
    n = len(phi)

    # Half step for pi
    laplacian = np.zeros(n)
    for i in range(n):
        laplacian[i] = (phi[(i+1) % n] - 2*phi[i] + phi[(i-1) % n]) / dx**2
    force = laplacian + lattice_force(phi, m, lam)
    pi_half = pi + 0.5 * dt * force

    # Full step for phi
    phi_new = phi + dt * pi_half

    # Half step for pi (with new phi)
    for i in range(n):
        laplacian[i] = (phi_new[(i+1) % n] - 2*phi_new[i] + phi_new[(i-1) % n]) / dx**2
    force = laplacian + lattice_force(phi_new, m, lam)
    pi_new = pi_half + 0.5 * dt * force

    return phi_new, pi_new


def rk4_step(phi, pi, m, lam, dx, dt):
    """4th order Runge-Kutta (NOT symplectic)."""
    n = len(phi)

    def derivatives(phi_in, pi_in):
        laplacian = np.zeros(n)
        for i in range(n):
            laplacian[i] = (phi_in[(i+1) % n] - 2*phi_in[i] + phi_in[(i-1) % n]) / dx**2
        force = laplacian + lattice_force(phi_in, m, lam)
        return pi_in, force

    k1_phi, k1_pi = derivatives(phi, pi)
    k2_phi, k2_pi = derivatives(phi + 0.5*dt*k1_phi, pi + 0.5*dt*k1_pi)
    k3_phi, k3_pi = derivatives(phi + 0.5*dt*k2_phi, pi + 0.5*dt*k2_pi)
    k4_phi, k4_pi = derivatives(phi + dt*k3_phi, pi + dt*k3_pi)

    phi_new = phi + (dt/6) * (k1_phi + 2*k2_phi + 2*k3_phi + k4_phi)
    pi_new = pi + (dt/6) * (k1_pi + 2*k2_pi + 2*k3_pi + k4_pi)

    return phi_new, pi_new


def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Lattice parameters
    n_sites = 100
    L = 10.0
    dx = L / n_sites
    x = np.linspace(0, L - dx, n_sites)

    # Field parameters
    m = 1.0
    lam = 0.5

    # Initial condition: localized wavepacket
    phi0 = np.exp(-(x - L/2)**2 / 2) * np.sin(3 * 2 * np.pi * x / L)
    pi0 = np.zeros(n_sites)

    # Plot 1: Initial configuration
    ax = axes[0, 0]

    ax.plot(x, phi0, 'b-', lw=2, label='φ(x,0)')
    ax.plot(x, lattice_potential(phi0, m, lam), 'r--', lw=1.5, label='V(φ)')

    ax.set_xlabel('Position x')
    ax.set_ylabel('Field Value')
    ax.set_title('Initial Field Configuration')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Energy conservation comparison
    ax = axes[0, 1]

    dt = 0.01
    n_steps = 2000

    integrators = [
        ('Euler', euler_step, 'r'),
        ('Leapfrog', leapfrog_step, 'b'),
        ('RK4', rk4_step, 'g'),
    ]

    for name, integrator, color in integrators:
        phi = phi0.copy()
        pi = pi0.copy()

        E_total = []
        E0 = sum(compute_energy(phi, pi, m, lam, dx))

        for _ in range(n_steps):
            phi, pi = integrator(phi, pi, m, lam, dx, dt)
            E = sum(compute_energy(phi, pi, m, lam, dx))
            rel_err = (E - E0) / E0 if np.isfinite(E) else np.nan
            E_total.append(np.clip(rel_err, -0.5, 0.5))  # Clip for visualization

        t_array = np.arange(n_steps) * dt
        ax.plot(t_array, E_total, '-', color=color, lw=1.5, label=name)

    ax.set_xlabel('Time')
    ax.set_ylabel('Relative Energy Error (E-E₀)/E₀')
    ax.set_title('Energy Conservation\nDifferent Integrators')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 0.1)

    # Plot 3: Long-time energy drift
    ax = axes[0, 2]

    n_steps_long = 10000

    for name, integrator, color in [('Leapfrog', leapfrog_step, 'b'),
                                     ('RK4', rk4_step, 'g')]:
        phi = phi0.copy()
        pi = pi0.copy()

        E_total = []
        E0 = sum(compute_energy(phi, pi, m, lam, dx))

        for _ in range(n_steps_long):
            phi, pi = integrator(phi, pi, m, lam, dx, dt)
            E = sum(compute_energy(phi, pi, m, lam, dx))
            E_total.append(E)

        t_array = np.arange(n_steps_long) * dt
        ax.plot(t_array, E_total, '-', color=color, lw=1, alpha=0.7, label=name)

    ax.axhline(y=E0, color='k', linestyle='--', alpha=0.5, label='Initial E')
    ax.set_xlabel('Time')
    ax.set_ylabel('Total Energy')
    ax.set_title('Long-Time Energy Behavior\n(Leapfrog bounded, RK4 drifts)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Energy components
    ax = axes[1, 0]

    phi = phi0.copy()
    pi = pi0.copy()

    kinetic_hist = []
    gradient_hist = []
    potential_hist = []
    total_hist = []

    for _ in range(3000):
        phi, pi = leapfrog_step(phi, pi, m, lam, dx, dt)
        K, G, V = compute_energy(phi, pi, m, lam, dx)
        kinetic_hist.append(K)
        gradient_hist.append(G)
        potential_hist.append(V)
        total_hist.append(K + G + V)

    t_array = np.arange(3000) * dt

    ax.plot(t_array, kinetic_hist, 'r-', lw=1, alpha=0.7, label='Kinetic')
    ax.plot(t_array, gradient_hist, 'g-', lw=1, alpha=0.7, label='Gradient')
    ax.plot(t_array, potential_hist, 'b-', lw=1, alpha=0.7, label='Potential')
    ax.plot(t_array, total_hist, 'k-', lw=2, label='Total')

    ax.set_xlabel('Time')
    ax.set_ylabel('Energy')
    ax.set_title('Energy Components\n(Leapfrog integrator)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Field evolution snapshots
    ax = axes[1, 1]

    phi = phi0.copy()
    pi = pi0.copy()

    times = [0, 50, 100, 150]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(times)))

    snapshot_idx = 0
    for step in range(200):
        if step in [0, 50, 100, 150]:
            ax.plot(x, phi, '-', color=colors[snapshot_idx], lw=2,
                    label=f't = {step * dt:.1f}')
            snapshot_idx += 1

        phi, pi = leapfrog_step(phi, pi, m, lam, dx, dt * 10)

    ax.set_xlabel('Position x')
    ax.set_ylabel('Field φ(x,t)')
    ax.set_title('Field Evolution Snapshots')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Summary
    ax = axes[1, 2]
    ax.axis('off')

    E0_val = sum(compute_energy(phi0, pi0, m, lam, dx))

    summary = f"""
Lattice Field Theory Energy Conservation
========================================

Lattice Hamiltonian:
  H = Sigma_i [pi_i^2/2 + (phi_i+1 - phi_i)^2/(2a^2) + V(phi_i)] a

Potential:
  V(phi) = m^2*phi^2/2 + (lambda/4!)*phi^4

Equations of Motion:
  dphi_i/dt = pi_i
  dpi_i/dt = (phi_i+1 - 2*phi_i + phi_i-1)/a^2 - dV/dphi

Integrators:
-----------
1. Euler (1st order):
   - NOT symplectic
   - Energy drifts systematically
   - Avoid for long simulations

2. Leapfrog (2nd order):
   - Symplectic (preserves phase space)
   - Energy bounded, oscillates
   - Best for long-time behavior

3. RK4 (4th order):
   - NOT symplectic
   - High local accuracy
   - Energy drifts over time

Symplectic Property:
  Preserves: det(d(q',p')/d(q,p)) = 1
  Bounded energy error for all time

Lattice Parameters:
  Sites: {n_sites}
  Spacing: dx = {dx:.3f}
  m = {m}, lambda = {lam}
  Initial E = {E0_val:.3f}

Application:
  - Lattice QCD uses similar methods
  - Critical for Monte Carlo
"""

    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Experiment 216: Lattice Field Energy Conservation\n'
                 'Symplectic vs Non-Symplectic Integrators', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'exp216_lattice_field_energy.png'),
                dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/exp216_lattice_field_energy.png")


if __name__ == "__main__":
    main()
