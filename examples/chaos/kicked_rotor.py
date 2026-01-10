"""
Experiment 66: Kicked Rotor - Classical and Quantum Chaos.

The kicked rotor (or standard map) is a fundamental model of Hamiltonian chaos:

    p_{n+1} = p_n + K * sin(theta_n)
    theta_{n+1} = theta_n + p_{n+1}   (mod 2*pi)

Where K is the kick strength parameter.

This demonstrates:
1. Transition from integrability to chaos as K increases
2. KAM (Kolmogorov-Arnold-Moser) tori destruction
3. Chaotic sea with surviving islands
4. Diffusion in momentum space
5. Connection to quantum localization (Anderson localization)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def standard_map(theta, p, K):
    """
    Standard map (kicked rotor map).

    p_{n+1} = p_n + K * sin(theta_n)
    theta_{n+1} = theta_n + p_{n+1}   (mod 2*pi)
    """
    p_new = p + K * np.sin(theta)
    theta_new = (theta + p_new) % (2 * np.pi)
    return theta_new, p_new


def iterate_map(theta0, p0, K, n_iterations):
    """Iterate the standard map n times."""
    theta = np.zeros(n_iterations + 1)
    p = np.zeros(n_iterations + 1)

    theta[0] = theta0
    p[0] = p0

    for i in range(n_iterations):
        theta[i+1], p[i+1] = standard_map(theta[i], p[i], K)

    return theta, p


def generate_phase_space(K, n_orbits=50, n_iterations=1000):
    """Generate phase space by following many orbits."""
    all_theta = []
    all_p = []

    for _ in range(n_orbits):
        theta0 = np.random.uniform(0, 2*np.pi)
        p0 = np.random.uniform(-np.pi, np.pi)

        theta, p = iterate_map(theta0, p0, K, n_iterations)

        # Fold p into [-pi, pi]
        p_folded = ((p + np.pi) % (2 * np.pi)) - np.pi

        all_theta.extend(theta)
        all_p.extend(p_folded)

    return np.array(all_theta), np.array(all_p)


def compute_lyapunov(K, n_iterations=10000):
    """Estimate Lyapunov exponent for the standard map."""
    # Start from a typical point
    theta = 0.5
    p = 0.5

    lyap_sum = 0

    for _ in range(n_iterations):
        # Jacobian of the map
        # d(theta', p')/d(theta, p) = [[1+K*cos(theta), 1], [K*cos(theta), 1]]
        J = np.array([[1 + K * np.cos(theta), 1],
                      [K * np.cos(theta), 1]])

        # Largest singular value contributes to Lyapunov
        _, s, _ = np.linalg.svd(J)
        lyap_sum += np.log(s[0])

        theta, p = standard_map(theta, p, K)

    return lyap_sum / n_iterations


def momentum_diffusion(K, n_particles=100, n_iterations=1000):
    """Track momentum diffusion (spreading of p over time)."""
    # Start all particles at theta random, p = 0
    theta = np.random.uniform(0, 2*np.pi, n_particles)
    p = np.zeros(n_particles)

    p_variance = [0]

    for _ in range(n_iterations):
        for j in range(n_particles):
            theta[j], p[j] = standard_map(theta[j], p[j], K)

        p_variance.append(np.var(p))

    return np.array(p_variance)


def kam_tori_destruction():
    """
    Demonstrate KAM tori destruction as K increases.

    Track specific initial conditions through the transition.
    """
    K_values = [0.1, 0.5, 0.9, 0.97, 1.2, 2.0]
    results = {}

    # Track orbits starting near the "golden ratio" torus
    p0 = np.pi * (np.sqrt(5) - 1) / 2  # Golden ratio * pi

    for K in K_values:
        theta, p = iterate_map(0.0, p0, K, 5000)
        p_folded = ((p + np.pi) % (2 * np.pi)) - np.pi
        results[K] = (theta, p_folded)

    return results


def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Phase space for small K (nearly integrable)
    ax = axes[0, 0]

    K = 0.5
    theta, p = generate_phase_space(K, n_orbits=30, n_iterations=500)
    ax.scatter(theta, p, s=0.1, c='blue', alpha=0.5)

    ax.set_xlabel('theta')
    ax.set_ylabel('p')
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_title(f'K = {K} (Nearly Integrable)\n'
                 'KAM tori still intact')
    ax.grid(True, alpha=0.3)

    # Plot 2: Phase space for K = 0.97 (near critical)
    ax = axes[0, 1]

    K = 0.97
    theta, p = generate_phase_space(K, n_orbits=50, n_iterations=1000)
    ax.scatter(theta, p, s=0.1, c='blue', alpha=0.5)

    ax.set_xlabel('theta')
    ax.set_ylabel('p')
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_title(f'K = {K} (Near Critical)\n'
                 'Last KAM torus breaking')
    ax.grid(True, alpha=0.3)

    # Plot 3: Phase space for large K (fully chaotic)
    ax = axes[0, 2]

    K = 5.0
    theta, p = generate_phase_space(K, n_orbits=50, n_iterations=1000)
    ax.scatter(theta, p, s=0.1, c='red', alpha=0.3)

    ax.set_xlabel('theta')
    ax.set_ylabel('p')
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_title(f'K = {K} (Fully Chaotic)\n'
                 'Global chaos, no barriers')
    ax.grid(True, alpha=0.3)

    # Plot 4: Lyapunov exponent vs K
    ax = axes[1, 0]

    K_range = np.linspace(0.1, 5.0, 50)
    lyap = [compute_lyapunov(K, n_iterations=5000) for K in K_range]

    ax.plot(K_range, lyap, 'b-', lw=2)
    ax.axhline(y=0, color='r', linestyle='--', lw=1)
    ax.axvline(x=0.9716, color='g', linestyle=':', lw=2, label='K_c ~ 0.9716')

    ax.set_xlabel('Kick strength K')
    ax.set_ylabel('Lyapunov exponent')
    ax.set_title('Lyapunov Exponent vs K\n'
                 'Transition to global chaos at K_c')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Momentum diffusion
    ax = axes[1, 1]

    n_iter = 1000
    t = np.arange(n_iter + 1)

    for K, color in [(0.5, 'blue'), (1.0, 'green'), (2.0, 'orange'), (5.0, 'red')]:
        p_var = momentum_diffusion(K, n_particles=200, n_iterations=n_iter)
        ax.plot(t, p_var, color=color, lw=1.5, label=f'K={K}')

    # Theoretical diffusion: <p^2> ~ D*t where D ~ K^2/2 for large K
    ax.plot(t, 2.5**2/2 * t, 'k--', lw=1, alpha=0.5, label='Diffusion ~ t')

    ax.set_xlabel('Iteration n')
    ax.set_ylabel('Momentum variance <p^2>')
    ax.set_title('Momentum Diffusion\n'
                 'Chaotic motion leads to diffusion')
    ax.legend(fontsize=8)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')

    # Plot 6: Summary and critical K
    ax = axes[1, 2]
    ax.axis('off')

    summary = """Kicked Rotor (Standard Map)
===========================

The standard map is the archetypal model
of Hamiltonian chaos:

  p_{n+1} = p_n + K * sin(theta_n)
  theta_{n+1} = theta_n + p_{n+1}  (mod 2*pi)

KEY PHENOMENA:

1. KAM THEOREM:
   For small K, most invariant tori survive
   (slightly deformed). System remains
   quasi-integrable.

2. CRITICAL K (K_c ~ 0.9716):
   Last KAM torus (golden ratio frequency)
   breaks at K_c. Below K_c: transport
   barriers exist. Above: global chaos.

3. RESONANCE ISLANDS:
   At rational frequency ratios p/2*pi,
   resonance zones create island chains.
   These persist even in chaotic regime.

4. DIFFUSION:
   For K > K_c, momentum diffuses:
   <p^2> ~ (K^2/2) * n  (quasi-linear)

5. QUANTUM KICKED ROTOR:
   Quantum version shows "dynamical
   localization" - suppression of
   classical diffusion (like Anderson
   localization in disordered systems).

PHYSICAL REALIZATIONS:
- Atoms in standing light waves
- Particle accelerators
- Comet dynamics
- Billiard balls

This model is central to chaos theory,
quantum chaos, and has connections to
random matrix theory and localization."""

    ax.text(0.02, 0.98, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    plt.suptitle("Kicked Rotor: Standard Map and Hamiltonian Chaos\n"
                 "p' = p + K*sin(theta), theta' = theta + p'",
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'kicked_rotor.png'), dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/kicked_rotor.png")


if __name__ == "__main__":
    main()
