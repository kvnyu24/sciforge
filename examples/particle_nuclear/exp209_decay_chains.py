"""
Experiment 209: Radioactive Decay Chains (Bateman Equations)

Demonstrates the Bateman equations for sequential radioactive decay.
Shows parent-daughter equilibrium, secular equilibrium, and activity.

Physics:
- dN₁/dt = -λ₁ N₁
- dN₂/dt = λ₁ N₁ - λ₂ N₂
- Bateman solution: N_n(t) = N₁(0) × Σⱼ cⱼ exp(-λⱼt)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from src.sciforge.physics.nuclear import DecayChain


def bateman_solution_2(t, lambda1, lambda2, N10):
    """
    Analytical Bateman solution for 2-member chain.

    A → B → C (stable)

    Args:
        t: Time array
        lambda1, lambda2: Decay constants
        N10: Initial parent nuclei

    Returns:
        N1, N2, N3 (parent, daughter, stable)
    """
    N1 = N10 * np.exp(-lambda1 * t)

    if abs(lambda1 - lambda2) > 1e-10:
        N2 = N10 * lambda1 / (lambda2 - lambda1) * (np.exp(-lambda1 * t) - np.exp(-lambda2 * t))
    else:
        # Degenerate case
        N2 = N10 * lambda1 * t * np.exp(-lambda1 * t)

    N3 = N10 - N1 - N2

    return N1, N2, N3


def bateman_solution_3(t, lambdas, N10):
    """
    Bateman solution for 3-member chain.

    A → B → C → D (stable)
    """
    l1, l2, l3 = lambdas

    N1 = N10 * np.exp(-l1 * t)

    # N2 term
    c21 = l1 / (l2 - l1)
    c22 = -l1 / (l2 - l1)
    N2 = N10 * (c21 * np.exp(-l1 * t) + c22 * np.exp(-l2 * t))

    # N3 term (more complex)
    c31 = l1 * l2 / ((l2 - l1) * (l3 - l1))
    c32 = l1 * l2 / ((l1 - l2) * (l3 - l2))
    c33 = l1 * l2 / ((l1 - l3) * (l2 - l3))
    N3 = N10 * (c31 * np.exp(-l1 * t) + c32 * np.exp(-l2 * t) + c33 * np.exp(-l3 * t))

    N4 = N10 - N1 - N2 - N3

    return N1, N2, N3, N4


def numerical_decay_chain(t_span, lambdas, N0, t_eval):
    """
    Numerical solution for arbitrary decay chain.
    """
    n = len(lambdas) + 1  # Number of species (including stable end)

    def equations(t, N):
        dNdt = np.zeros(n)
        dNdt[0] = -lambdas[0] * N[0]

        for i in range(1, n - 1):
            dNdt[i] = lambdas[i-1] * N[i-1] - lambdas[i] * N[i]

        dNdt[-1] = lambdas[-1] * N[-2]  # Stable end product
        return dNdt

    N0_full = np.zeros(n)
    N0_full[:len(N0)] = N0

    sol = solve_ivp(equations, t_span, N0_full, t_eval=t_eval, method='LSODA')
    return sol.t, sol.y


def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Example: Uranium-238 decay chain (simplified)
    # Half-lives (years)
    t_half_U238 = 4.47e9  # years
    t_half_Th234 = 24.1 / 365  # years (24 days)
    t_half_Ra226 = 1600  # years
    t_half_Rn222 = 3.82 / 365  # years (3.82 days)

    # Plot 1: Simple two-member chain
    ax = axes[0, 0]

    lambda1 = np.log(2) / 10  # Parent t_1/2 = 10 units
    lambda2 = np.log(2) / 5   # Daughter t_1/2 = 5 units
    N10 = 1000

    t = np.linspace(0, 50, 500)
    N1, N2, N3 = bateman_solution_2(t, lambda1, lambda2, N10)

    ax.plot(t, N1, 'b-', lw=2, label='Parent (t₁/₂ = 10)')
    ax.plot(t, N2, 'r-', lw=2, label='Daughter (t₁/₂ = 5)')
    ax.plot(t, N3, 'g-', lw=2, label='Stable product')
    ax.axhline(y=N10, color='k', linestyle='--', alpha=0.3)

    ax.set_xlabel('Time (arb. units)')
    ax.set_ylabel('Number of Nuclei')
    ax.set_title('Two-Member Decay Chain\nA → B → C')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Activity curves
    ax = axes[0, 1]

    A1 = lambda1 * N1  # Parent activity
    A2 = lambda2 * N2  # Daughter activity

    ax.plot(t, A1, 'b-', lw=2, label='A(parent)')
    ax.plot(t, A2, 'r-', lw=2, label='A(daughter)')

    # Find transient equilibrium time
    t_eq_approx = np.log(lambda2 / lambda1) / (lambda2 - lambda1)
    ax.axvline(x=t_eq_approx, color='k', linestyle='--', alpha=0.5)

    ax.set_xlabel('Time (arb. units)')
    ax.set_ylabel('Activity (λN)')
    ax.set_title('Activity Curves\nTransient Equilibrium')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Secular equilibrium (long-lived parent)
    ax = axes[0, 2]

    lambda_parent = np.log(2) / 1000   # Very long-lived parent
    lambda_daughter = np.log(2) / 1    # Short-lived daughter
    N0_secular = 1000

    t_sec = np.linspace(0, 20, 500)
    N_p, N_d, N_s = bateman_solution_2(t_sec, lambda_parent, lambda_daughter, N0_secular)

    A_parent = lambda_parent * N_p
    A_daughter = lambda_daughter * N_d

    ax.plot(t_sec, A_parent, 'b-', lw=2, label='A(parent)')
    ax.plot(t_sec, A_daughter, 'r-', lw=2, label='A(daughter)')

    ax.set_xlabel('Time (daughter half-lives)')
    ax.set_ylabel('Activity')
    ax.set_title('Secular Equilibrium\n(Parent t₁/₂ >> Daughter t₁/₂)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Three-member chain
    ax = axes[1, 0]

    lambdas_3 = [np.log(2)/10, np.log(2)/3, np.log(2)/8]
    t_3 = np.linspace(0, 60, 500)
    N1_3, N2_3, N3_3, N4_3 = bateman_solution_3(t_3, lambdas_3, 1000)

    ax.plot(t_3, N1_3, 'b-', lw=2, label='N₁ (parent)')
    ax.plot(t_3, N2_3, 'r-', lw=2, label='N₂ (1st daughter)')
    ax.plot(t_3, N3_3, 'g-', lw=2, label='N₃ (2nd daughter)')
    ax.plot(t_3, N4_3, 'k-', lw=2, label='N₄ (stable)')

    ax.set_xlabel('Time')
    ax.set_ylabel('Number of Nuclei')
    ax.set_title('Three-Member Chain\nA → B → C → D')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Numerical solution for longer chain
    ax = axes[1, 1]

    # 5-member chain with different half-lives
    half_lives = [10, 2, 5, 1, 8]  # arb. units
    lambdas_5 = [np.log(2) / t for t in half_lives]
    N0_5 = [1000, 0, 0, 0, 0, 0]  # Only parent initially

    t_5, N_5 = numerical_decay_chain((0, 50), lambdas_5, N0_5, np.linspace(0, 50, 500))

    colors = plt.cm.viridis(np.linspace(0, 0.9, 6))
    labels = ['Parent', 'D₁', 'D₂', 'D₃', 'D₄', 'Stable']

    for i, (color, label) in enumerate(zip(colors, labels)):
        ax.plot(t_5, N_5[i], '-', color=color, lw=2, label=label)

    ax.set_xlabel('Time')
    ax.set_ylabel('Number of Nuclei')
    ax.set_title('5-Member Decay Chain\n(Numerical Solution)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 6: Summary and equilibrium conditions
    ax = axes[1, 2]
    ax.axis('off')

    summary = """
Bateman Equations for Decay Chains
==================================

Differential Equations:
  dN₁/dt = -λ₁N₁
  dNᵢ/dt = λᵢ₋₁Nᵢ₋₁ - λᵢNᵢ  (i = 2,...,n)
  dNₙ₊₁/dt = λₙNₙ  (stable end)

General Solution (Bateman, 1910):
  Nᵢ(t) = N₁(0) × Σⱼ cᵢⱼ exp(-λⱼt)

  where cᵢⱼ = Π_{k=1}^{i-1} λₖ / Π_{k=1,k≠j}^{i} (λₖ - λⱼ)

Equilibrium Conditions:
-----------------------
1. SECULAR EQUILIBRIUM:
   When λ₁ << λ₂ (parent long-lived)
   A₁ ≈ A₂ after t >> t₁/₂(daughter)
   N₂/N₁ = λ₁/λ₂

2. TRANSIENT EQUILIBRIUM:
   When λ₁ < λ₂ (parent shorter but not much)
   A₂/A₁ = λ₂/(λ₂ - λ₁) > 1
   Daughter activity exceeds parent

3. NO EQUILIBRIUM:
   When λ₁ > λ₂ (daughter longer-lived)
   Daughter accumulates continuously

Activity:
  A = λN = N₀λe^{-λt}
  Total activity = Σ λᵢNᵢ

Applications:
  • Dating (¹⁴C, U-Pb, K-Ar)
  • Nuclear medicine generators
  • Environmental monitoring
  • Reactor physics
"""

    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Experiment 209: Radioactive Decay Chains\n'
                 'Bateman Equations and Equilibrium', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'exp209_decay_chains.png'),
                dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/exp209_decay_chains.png")


if __name__ == "__main__":
    main()
