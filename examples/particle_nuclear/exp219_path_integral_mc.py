"""
Experiment 219: Path Integral Monte Carlo (1D)

Demonstrates path integral Monte Carlo for computing quantum mechanical
ground state properties. Uses Euclidean time discretization.

Physics:
- ⟨x_f|e^{-βH}|x_i⟩ = ∫Dx(τ) exp(-S_E[x]/ℏ)
- Euclidean action: S_E = ∫₀^β [½m(dx/dτ)² + V(x)] dτ
- Ground state energy: E₀ = -limₐ→∞ (1/β) ln Z
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def harmonic_potential(x, omega=1.0, m=1.0):
    """Harmonic oscillator potential: V = ½mω²x²"""
    return 0.5 * m * omega**2 * x**2


def anharmonic_potential(x, omega=1.0, m=1.0, g=0.1):
    """Anharmonic oscillator: V = ½mω²x² + gx⁴"""
    return 0.5 * m * omega**2 * x**2 + g * x**4


def double_well_potential(x, a=1.0, b=1.0):
    """Double well: V = -ax² + bx⁴"""
    return -a * x**2 + b * x**4


def euclidean_action(path, potential, dt, m=1.0):
    """
    Calculate Euclidean action for a path.

    S_E = Σ_i [½m(x_{i+1} - x_i)²/dt + dt × V(x_i)]
    """
    n = len(path)
    kinetic = 0
    potential_sum = 0

    for i in range(n):
        # Kinetic term (periodic BC)
        dx = path[(i+1) % n] - path[i]
        kinetic += 0.5 * m * dx**2 / dt

        # Potential term
        potential_sum += dt * potential(path[i])

    return kinetic + potential_sum


def metropolis_step(path, potential, dt, m=1.0, delta=0.5):
    """
    Single Metropolis update step.

    Propose local change to path and accept/reject.
    """
    n = len(path)
    i = np.random.randint(n)

    # Propose new position
    x_old = path[i]
    x_new = x_old + np.random.uniform(-delta, delta)

    # Calculate action change (only affected terms)
    # Kinetic terms: (i-1,i) and (i,i+1)
    ip1 = (i + 1) % n
    im1 = (i - 1) % n

    dS_kinetic = m / (2 * dt) * (
        (path[ip1] - x_new)**2 + (x_new - path[im1])**2
        - (path[ip1] - x_old)**2 - (x_old - path[im1])**2
    )

    # Potential term
    dS_potential = dt * (potential(x_new) - potential(x_old))

    dS = dS_kinetic + dS_potential

    # Metropolis accept/reject
    if dS <= 0 or np.random.random() < np.exp(-dS):
        path[i] = x_new
        return True
    return False


def pimc_simulation(n_time, beta, potential, n_sweeps, n_therm=1000,
                    m=1.0, delta=0.5):
    """
    Path Integral Monte Carlo simulation.

    Args:
        n_time: Number of time slices
        beta: Inverse temperature (imaginary time extent)
        potential: Potential function V(x)
        n_sweeps: Number of measurement sweeps
        n_therm: Thermalization sweeps
        m: Mass
        delta: Metropolis step size

    Returns:
        Dictionary with observables
    """
    dt = beta / n_time

    # Initialize random path
    path = np.random.uniform(-1, 1, n_time)

    # Thermalization
    for _ in range(n_therm):
        for _ in range(n_time):
            metropolis_step(path, potential, dt, m, delta)

    # Measurement
    x_samples = []
    x2_samples = []
    action_samples = []
    accept_count = 0
    total_count = 0

    for _ in range(n_sweeps):
        for _ in range(n_time):
            if metropolis_step(path, potential, dt, m, delta):
                accept_count += 1
            total_count += 1

        # Measure observables
        x_samples.append(np.mean(path))
        x2_samples.append(np.mean(path**2))
        action_samples.append(euclidean_action(path, potential, dt, m))

    return {
        'x': np.array(x_samples),
        'x2': np.array(x2_samples),
        'action': np.array(action_samples),
        'path': path.copy(),
        'acceptance': accept_count / total_count
    }


def ground_state_energy_virial(x2, omega, m=1.0):
    """
    Estimate ground state energy using virial theorem.

    For harmonic oscillator: E = ⟨T⟩ + ⟨V⟩ = 2⟨V⟩ = mω²⟨x²⟩
    """
    return m * omega**2 * x2


def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Parameters
    omega = 1.0
    m = 1.0
    beta = 10.0  # Low temperature
    n_time = 100
    n_sweeps = 5000
    n_therm = 2000

    # Plot 1: Sample paths
    ax = axes[0, 0]

    potential = lambda x: harmonic_potential(x, omega, m)

    # Run short simulations to get sample paths
    for i in range(5):
        result = pimc_simulation(n_time, beta, potential, n_sweeps=100,
                                 n_therm=500, m=m)
        tau = np.linspace(0, beta, n_time)
        ax.plot(tau, result['path'], '-', alpha=0.7, lw=1)

    ax.set_xlabel('Imaginary Time τ')
    ax.set_ylabel('Position x(τ)')
    ax.set_title('Sample Paths (Harmonic Oscillator)\nPeriodic boundary conditions')
    ax.grid(True, alpha=0.3)

    # Plot 2: Position distribution
    ax = axes[0, 1]

    result = pimc_simulation(n_time, beta, potential, n_sweeps, n_therm, m=m)

    # Collect all positions from path
    all_x = []
    path = np.random.uniform(-1, 1, n_time)
    dt = beta / n_time

    for _ in range(2000):
        for _ in range(n_time):
            metropolis_step(path, potential, dt, m, delta=0.5)
        all_x.extend(path)

    all_x = np.array(all_x)

    # Histogram
    x_range = np.linspace(-3, 3, 100)
    ax.hist(all_x, bins=50, density=True, alpha=0.7, label='PIMC')

    # Exact ground state |ψ₀|²
    psi_sq = np.exp(-m * omega * x_range**2) / np.sqrt(np.pi / (m * omega))
    ax.plot(x_range, psi_sq, 'r-', lw=2, label='Exact |ψ₀|²')

    ax.set_xlabel('Position x')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'Ground State Distribution\nβ = {beta}, n_τ = {n_time}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: ⟨x²⟩ vs β (temperature)
    ax = axes[0, 2]

    betas = [2, 4, 6, 8, 10, 15]
    x2_means = []
    x2_stds = []

    for b in betas:
        result = pimc_simulation(n_time, b, potential, n_sweeps=3000,
                                 n_therm=1000, m=m)
        x2_means.append(np.mean(result['x2']))
        x2_stds.append(np.std(result['x2']) / np.sqrt(len(result['x2'])))

    ax.errorbar(betas, x2_means, yerr=x2_stds, fmt='o-', lw=2,
                markersize=8, capsize=5, label='PIMC')

    # Exact: ⟨x²⟩ = (ℏ/2mω) coth(βℏω/2) ≈ ℏ/2mω at low T
    x2_exact = [0.5 / (m * omega) * (1 / np.tanh(b * omega / 2)) for b in betas]
    ax.plot(betas, x2_exact, 'r--', lw=2, label='Exact')

    ax.axhline(y=0.5 / (m * omega), color='gray', linestyle=':', alpha=0.5,
               label='T=0 limit')

    ax.set_xlabel('β = 1/(k_B T)')
    ax.set_ylabel('⟨x²⟩')
    ax.set_title('Position Variance vs Inverse Temperature')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Anharmonic oscillator
    ax = axes[1, 0]

    g_values = [0, 0.1, 0.3, 0.5]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(g_values)))

    for g, color in zip(g_values, colors):
        pot = lambda x, g=g: anharmonic_potential(x, omega, m, g)

        # Collect positions
        path = np.random.uniform(-1, 1, n_time)
        dt = beta / n_time
        all_x = []

        for _ in range(2000):
            for _ in range(n_time):
                metropolis_step(path, pot, dt, m, delta=0.5)
            all_x.extend(path)

        ax.hist(all_x, bins=50, density=True, alpha=0.5, color=color,
                label=f'g = {g}')

    ax.set_xlabel('Position x')
    ax.set_ylabel('Probability Density')
    ax.set_title('Anharmonic Oscillator\nV = ½ω²x² + gx⁴')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Double well
    ax = axes[1, 1]

    pot_dw = lambda x: double_well_potential(x, a=2.0, b=1.0)

    # Collect positions
    path = np.random.uniform(-2, 2, n_time)
    dt = beta / n_time
    all_x = []

    for _ in range(3000):
        for _ in range(n_time):
            metropolis_step(path, pot_dw, dt, m, delta=0.5)
        all_x.extend(path)

    ax.hist(all_x, bins=50, density=True, alpha=0.7, label='PIMC')

    # Show potential
    x_pot = np.linspace(-2, 2, 100)
    V_pot = pot_dw(x_pot)
    V_pot = V_pot - np.min(V_pot)
    V_pot = V_pot / np.max(V_pot) * 1.5  # Scale for visibility
    ax.plot(x_pot, V_pot, 'r--', lw=2, label='V(x) (scaled)')

    ax.set_xlabel('Position x')
    ax.set_ylabel('Probability Density')
    ax.set_title('Double Well Potential\nV = -ax² + bx⁴')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Summary
    ax = axes[1, 2]
    ax.axis('off')

    exact_x2 = 0.5 / (m * omega)
    E0_exact = 0.5 * omega

    summary = f"""
Path Integral Monte Carlo (PIMC)
================================

Idea:
  Quantum mechanics → classical
  statistical mechanics in d+1 dim

Path Integral:
  ⟨x_f|e^{{-βH}}|x_i⟩ = ∫Dx exp(-S_E/ℏ)

Euclidean Action:
  S_E = ∫₀^β [½m(dx/dτ)² + V(x)] dτ

Discretization:
  β = N_τ × Δτ
  x(τ) → {{x_0, x_1, ..., x_{{N_τ-1}}}}

Monte Carlo Weight:
  exp(-S_E) where
  S_E = Σᵢ [½m(x_{{i+1}}-x_i)²/Δτ + Δτ×V(x_i)]

Observables:
  ⟨O⟩ = (1/Z) ∫Dx O[x] exp(-S_E)

Harmonic Oscillator (ω = {omega}):
  Exact E₀ = ℏω/2 = {E0_exact:.3f}
  Exact ⟨x²⟩ = ℏ/(2mω) = {exact_x2:.3f}

Simulation Parameters:
  β = {beta}, N_τ = {n_time}
  Δτ = {beta/n_time:.3f}

Applications:
  • Ground state properties
  • Quantum fluids (He-4)
  • Lattice QCD
  • Quantum annealing
"""

    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Experiment 219: Path Integral Monte Carlo\n'
                 'Quantum Mechanics via Statistical Mechanics', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'exp219_path_integral_mc.png'),
                dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/exp219_path_integral_mc.png")


if __name__ == "__main__":
    main()
