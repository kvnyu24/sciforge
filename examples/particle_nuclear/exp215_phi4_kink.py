"""
Experiment 215: Phi^4 Kink Solutions

Demonstrates topological soliton solutions (kinks) in the φ⁴ theory,
a classic example of non-perturbative field theory.

Physics:
- Double-well potential: V(φ) = (λ/4)(φ² - v²)²
- Kink solution: φ(x) = v × tanh((x-x₀)/w)
- Width: w = √(2/λ)/v
- Energy: E = (4/3)√(2λ)v³
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def phi4_potential(phi, v, lam):
    """
    φ⁴ potential (Mexican hat / double well).

    V(φ) = (λ/4)(φ² - v²)²
    """
    return (lam / 4) * (phi**2 - v**2)**2


def phi4_force(phi, v, lam):
    """
    Force from φ⁴ potential.

    -dV/dφ = -λφ(φ² - v²)
    """
    return -lam * phi * (phi**2 - v**2)


def kink_solution(x, x0, v, lam):
    """
    Analytical kink solution.

    φ(x) = v × tanh((x - x₀) × √(λ/2) × v)
    """
    width = np.sqrt(2 / lam) / v
    return v * np.tanh((x - x0) / width)


def antikink_solution(x, x0, v, lam):
    """
    Antikink solution.

    φ(x) = -v × tanh((x - x₀) × √(λ/2) × v)
    """
    return -kink_solution(x, x0, v, lam)


def kink_energy_density(x, x0, v, lam):
    """
    Energy density of kink.

    ε(x) = (1/2)(∂φ/∂x)² + V(φ)
    """
    width = np.sqrt(2 / lam) / v
    sech2 = 1 / np.cosh((x - x0) / width)**2

    # Gradient energy
    dphi_dx = v * sech2 / width
    kinetic = 0.5 * dphi_dx**2

    # Potential energy
    phi = kink_solution(x, x0, v, lam)
    potential = phi4_potential(phi, v, lam)

    return kinetic + potential


def kink_total_energy(v, lam):
    """
    Total kink energy (mass).

    M = (4/3) × √(2λ) × v³
    """
    return (4/3) * np.sqrt(2 * lam) * v**3


def kink_width(v, lam):
    """Kink width w = √(2/λ)/v"""
    return np.sqrt(2 / lam) / v


def evolve_phi4_field(phi0, dphi0, x, v, lam, t_span, t_eval):
    """
    Evolve φ⁴ field using finite differences.
    """
    dx = x[1] - x[0]
    n = len(x)

    def field_eqn(t, y):
        phi = y[:n]
        dphi = y[n:]

        # Laplacian with Dirichlet BC
        laplacian = np.zeros(n)
        laplacian[1:-1] = (phi[2:] - 2*phi[1:-1] + phi[:-2]) / dx**2

        # Boundary conditions (fixed at vacuum)
        laplacian[0] = (phi[1] - 2*phi[0] + (-v)) / dx**2
        laplacian[-1] = ((v) - 2*phi[-1] + phi[-2]) / dx**2

        ddphi = laplacian + phi4_force(phi, v, lam)

        return np.concatenate([dphi, ddphi])

    y0 = np.concatenate([phi0, dphi0])
    sol = solve_ivp(field_eqn, t_span, y0, t_eval=t_eval, method='RK45',
                    max_step=0.01)

    return sol.t, sol.y[:n]


def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Parameters
    v = 1.0    # Vacuum expectation value
    lam = 1.0  # Coupling constant

    # Plot 1: Potential and vacua
    ax = axes[0, 0]

    phi_range = np.linspace(-1.5, 1.5, 200)
    V = phi4_potential(phi_range, v, lam)

    ax.plot(phi_range, V, 'b-', lw=2)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)

    # Mark vacua
    ax.plot([-v, v], [0, 0], 'ro', markersize=10, label='Vacua')
    ax.plot([0], [phi4_potential(0, v, lam)], 'go', markersize=10,
            label='Unstable max')

    ax.set_xlabel('Field φ')
    ax.set_ylabel('V(φ)')
    ax.set_title('φ⁴ Double-Well Potential\nV = (λ/4)(φ² - v²)²')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Kink and antikink profiles
    ax = axes[0, 1]

    x = np.linspace(-10, 10, 500)
    x0 = 0

    phi_kink = kink_solution(x, x0, v, lam)
    phi_antikink = antikink_solution(x, x0, v, lam)

    ax.plot(x, phi_kink, 'b-', lw=2, label='Kink')
    ax.plot(x, phi_antikink, 'r--', lw=2, label='Antikink')
    ax.axhline(y=v, color='k', linestyle=':', alpha=0.5)
    ax.axhline(y=-v, color='k', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)

    # Mark kink width
    w = kink_width(v, lam)
    ax.axvspan(-w, w, alpha=0.2, color='blue')

    ax.set_xlabel('Position x')
    ax.set_ylabel('Field φ(x)')
    ax.set_title(f'Kink Solution\nφ = v tanh(x/w), w = {w:.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Energy density
    ax = axes[0, 2]

    epsilon = kink_energy_density(x, x0, v, lam)

    ax.plot(x, epsilon, 'g-', lw=2)
    ax.fill_between(x, 0, epsilon, alpha=0.3)

    total_E = kink_total_energy(v, lam)
    ax.text(5, 0.4, f'Total E = {total_E:.3f}', fontsize=12)

    ax.set_xlabel('Position x')
    ax.set_ylabel('Energy Density ε(x)')
    ax.set_title('Kink Energy Distribution\nLocalized at center')
    ax.grid(True, alpha=0.3)

    # Plot 4: Kink-antikink configuration
    ax = axes[1, 0]

    # Kink at -5, antikink at +5
    phi_kk = kink_solution(x, -5, v, lam) + antikink_solution(x, 5, v, lam) + v

    ax.plot(x, phi_kk, 'b-', lw=2)
    ax.axhline(y=v, color='k', linestyle=':', alpha=0.5)
    ax.axhline(y=-v, color='k', linestyle=':', alpha=0.5)

    ax.set_xlabel('Position x')
    ax.set_ylabel('Field φ(x)')
    ax.set_title('Kink-Antikink Pair\n(Approximate superposition)')
    ax.grid(True, alpha=0.3)

    # Plot 5: Parameter dependence
    ax = axes[1, 1]

    lambdas = [0.5, 1.0, 2.0, 4.0]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(lambdas)))

    for lam_val, color in zip(lambdas, colors):
        phi_l = kink_solution(x, 0, v, lam_val)
        ax.plot(x, phi_l, '-', color=color, lw=2,
                label=f'λ = {lam_val}')

    ax.set_xlabel('Position x')
    ax.set_ylabel('Field φ(x)')
    ax.set_title('Kink Width vs Coupling\nw ∝ 1/√λ')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Summary
    ax = axes[1, 2]
    ax.axis('off')

    w_val = kink_width(v, lam)
    E_val = kink_total_energy(v, lam)

    summary = f"""
φ⁴ Kink Solutions
=================

Lagrangian:
  L = (1/2)(∂μφ)² - (λ/4)(φ² - v²)²

Vacuum States:
  φ = ±v (spontaneously broken Z₂)

Static Equation:
  d²φ/dx² = λφ(φ² - v²)

Kink Solution:
  φ(x) = v tanh[(x-x₀)/w]
  w = √(2/λ)/v = {w_val:.3f}

Antikink:
  φ(x) = -v tanh[(x-x₀)/w]

Kink Energy (Mass):
  M = (4/3)√(2λ)v³ = {E_val:.3f}

Topological Charge:
  Q = (φ(+∞) - φ(-∞))/(2v)
  Kink: Q = +1
  Antikink: Q = -1

Properties:
  • Topologically stable
  • Cannot decay to vacuum
  • Interpolates between vacua
  • Localized energy density

Dynamics:
  • Kink-antikink attraction
  • Can annihilate to radiation
  • Lorentz contracted when moving
  • Moving kink: γ factor

Applications:
  • Domain walls in cosmology
  • Magnetic domain walls
  • Dislocations in crystals
"""

    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Experiment 215: φ⁴ Kink Solutions\n'
                 'Topological Solitons in Field Theory', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'exp215_phi4_kink.png'),
                dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/exp215_phi4_kink.png")


if __name__ == "__main__":
    main()
