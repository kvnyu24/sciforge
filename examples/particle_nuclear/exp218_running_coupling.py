"""
Experiment 218: Running Coupling and Beta Function

Demonstrates the scale dependence of coupling constants in quantum field theory.
Shows the beta function, asymptotic freedom, and Landau poles.

Physics:
- β(g) = μ dg/dμ (beta function)
- QCD: β < 0 → asymptotic freedom
- QED: β > 0 → Landau pole
- Fixed points: β(g*) = 0
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def beta_phi4_1loop(g, n=1):
    """
    One-loop beta function for φ⁴ theory.

    β(g) = (n+8)/(16π²) g² + O(g³)

    Args:
        g: Coupling constant
        n: Number of field components
    """
    return (n + 8) / (16 * np.pi**2) * g**2


def beta_qed_1loop(alpha):
    """
    One-loop beta function for QED.

    β(α) = (2α²)/(3π)

    where α = e²/(4π)
    """
    return (2 * alpha**2) / (3 * np.pi)


def beta_qcd_1loop(alpha_s, nf=6):
    """
    One-loop beta function for QCD.

    β(α_s) = -b₀ α_s² / (2π)
    b₀ = 11 - 2n_f/3

    Args:
        alpha_s: Strong coupling
        nf: Number of active flavors
    """
    b0 = 11 - 2 * nf / 3
    return -b0 * alpha_s**2 / (2 * np.pi)


def beta_qcd_2loop(alpha_s, nf=6):
    """
    Two-loop beta function for QCD.

    β(α_s) = -b₀ α_s²/(2π) - b₁ α_s³/(4π²)
    """
    b0 = 11 - 2 * nf / 3
    b1 = 51 - 19 * nf / 3
    return -b0 * alpha_s**2 / (2 * np.pi) - b1 * alpha_s**3 / (4 * np.pi**2)


def solve_rg_equation(beta_func, g0, mu_range, **kwargs):
    """
    Solve the RG equation dg/d(ln μ) = β(g).

    Args:
        beta_func: Beta function β(g)
        g0: Initial coupling at μ_0
        mu_range: (μ_min, μ_max)
        kwargs: Additional arguments for beta_func

    Returns:
        Arrays of (μ, g(μ))
    """
    def rg_eq(t, g):
        return [beta_func(g[0], **kwargs)]

    t_span = (np.log(mu_range[0]), np.log(mu_range[1]))
    t_eval = np.linspace(t_span[0], t_span[1], 200)

    sol = solve_ivp(rg_eq, t_span, [g0], t_eval=t_eval, method='RK45',
                    max_step=0.1)

    mu = np.exp(sol.t)
    g = sol.y[0]

    return mu, g


def alpha_qcd_analytic(mu, Lambda_QCD=0.2, nf=6):
    """
    Analytic one-loop running of α_s.

    α_s(μ) = 4π / (b₀ ln(μ²/Λ²))
    """
    b0 = 11 - 2 * nf / 3
    with np.errstate(divide='ignore', invalid='ignore'):
        result = 4 * np.pi / (b0 * np.log(mu**2 / Lambda_QCD**2))
        return np.where(mu > Lambda_QCD, result, np.nan)


def alpha_qed_analytic(mu, alpha_0=1/137, mu_0=0.511e-3):
    """
    Analytic one-loop running of α_QED.

    α(μ) = α₀ / (1 - (2α₀)/(3π) ln(μ/μ₀))
    """
    return alpha_0 / (1 - (2 * alpha_0) / (3 * np.pi) * np.log(mu / mu_0))


def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: QCD beta function
    ax = axes[0, 0]

    alpha_s = np.linspace(0, 0.5, 100)

    for nf in [3, 4, 5, 6]:
        beta = beta_qcd_1loop(alpha_s, nf)
        ax.plot(alpha_s, beta, lw=2, label=f'n_f = {nf}')

    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('α_s')
    ax.set_ylabel('β(α_s)')
    ax.set_title('QCD Beta Function (1-loop)\nβ < 0 → Asymptotic Freedom')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: QED beta function
    ax = axes[0, 1]

    alpha = np.linspace(0, 0.02, 100)
    beta_qed = beta_qed_1loop(alpha)

    ax.plot(alpha, beta_qed, 'r-', lw=2)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('α')
    ax.set_ylabel('β(α)')
    ax.set_title('QED Beta Function (1-loop)\nβ > 0 → Landau Pole')
    ax.grid(True, alpha=0.3)

    # Plot 3: Running α_s (QCD)
    ax = axes[0, 2]

    mu_range = np.logspace(-1, 3, 200)  # GeV

    # Different Λ_QCD
    for Lambda in [0.15, 0.2, 0.3]:
        alpha_s = alpha_qcd_analytic(mu_range, Lambda_QCD=Lambda)
        ax.semilogx(mu_range, alpha_s, lw=2, label=f'Λ = {Lambda} GeV')

    ax.axhline(y=0.118, color='k', linestyle='--', alpha=0.5,
               label='α_s(M_Z) ≈ 0.118')
    ax.axvline(x=91.2, color='gray', linestyle=':', alpha=0.5)

    ax.set_xlabel('Energy Scale μ (GeV)')
    ax.set_ylabel('α_s(μ)')
    ax.set_title('Running of Strong Coupling\nα_s decreases at high energy')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    ax.set_xlim(0.1, 1000)

    # Plot 4: Running α_QED
    ax = axes[1, 0]

    mu_range_qed = np.logspace(-3, 17, 200)  # GeV

    alpha_qed = alpha_qed_analytic(mu_range_qed)

    ax.semilogx(mu_range_qed, alpha_qed, 'r-', lw=2)
    ax.axhline(y=1/137, color='k', linestyle='--', alpha=0.5,
               label='α(0) = 1/137')
    ax.axhline(y=1/128, color='gray', linestyle=':', alpha=0.5,
               label='α(M_Z) ≈ 1/128')

    # Landau pole
    mu_Landau = 0.511e-3 * np.exp(3 * np.pi / (2 * 1/137))
    ax.axvline(x=mu_Landau, color='red', linestyle='--', alpha=0.5,
               label=f'Landau pole')

    ax.set_xlabel('Energy Scale μ (GeV)')
    ax.set_ylabel('α(μ)')
    ax.set_title('Running of QED Coupling\nα increases at high energy')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 0.02)

    # Plot 5: Coupling unification (schematic)
    ax = axes[1, 1]

    mu_range_unif = np.logspace(2, 16, 100)  # GeV

    # Simplified running
    alpha_1 = 1/60 + (1/60) * 0.02 * np.log(mu_range_unif / 91.2)
    alpha_2 = 1/30 - (1/30) * 0.01 * np.log(mu_range_unif / 91.2)
    alpha_3 = 0.118 - 0.118 * 0.05 * np.log(mu_range_unif / 91.2)
    alpha_3 = np.maximum(alpha_3, 0.01)

    ax.semilogx(mu_range_unif, 1/alpha_1, 'b-', lw=2, label='U(1)')
    ax.semilogx(mu_range_unif, 1/alpha_2, 'g-', lw=2, label='SU(2)')
    ax.semilogx(mu_range_unif, 1/alpha_3, 'r-', lw=2, label='SU(3)')

    # Mark unification scale
    ax.axvline(x=2e16, color='purple', linestyle='--', alpha=0.5,
               label='GUT scale')

    ax.set_xlabel('Energy Scale μ (GeV)')
    ax.set_ylabel('1/α')
    ax.set_title('Gauge Coupling Unification\n(Schematic)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1e2, 1e17)

    # Plot 6: Summary
    ax = axes[1, 2]
    ax.axis('off')

    summary = """
Running Coupling and Beta Function
==================================

Renormalization Group Equation:
  μ dg/dμ = β(g)
  or: d(lnμ) = dg/β(g)

One-Loop Beta Functions:
-----------------------
QED:
  β(α) = (2α²)/(3π) > 0
  α(μ) increases with energy
  Landau pole at μ ~ 10²⁸⁵ GeV

QCD:
  β(α_s) = -b₀α_s²/(2π), b₀ = 11 - 2n_f/3 > 0
  α_s(μ) decreases with energy
  ASYMPTOTIC FREEDOM (Nobel 2004)

φ⁴ Theory:
  β(g) = (n+8)/(16π²)g²
  Trivial in 4D (free field)

Implications:
-------------
1. Asymptotic Freedom (QCD):
   • High energy: quarks weakly coupled
   • Perturbation theory works
   • Jets, hard scattering

2. Confinement (QCD):
   • Low energy: α_s → large
   • Non-perturbative
   • Hadrons, color confinement

3. Landau Pole (QED):
   • Theory breaks down at high μ
   • Need UV completion

Running at M_Z = 91.2 GeV:
  α_QED ≈ 1/128
  α_s ≈ 0.118
  sin²θ_W ≈ 0.231

Grand Unification:
  Couplings may unify at ~10¹⁶ GeV
  → Single gauge group (GUT)
"""

    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Experiment 218: Running Coupling and Beta Function\n'
                 'Scale Dependence in Quantum Field Theory', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'exp218_running_coupling.png'),
                dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/exp218_running_coupling.png")


if __name__ == "__main__":
    main()
