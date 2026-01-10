"""
Experiment 171: Variational Method for Quartic Potential

Demonstrates the variational principle for finding approximate ground state
energies of quantum systems, applied to the quartic (anharmonic) oscillator.

Physics:
    The Variational Principle states that for any trial wavefunction |psi_trial>:

    E_trial = <psi_trial|H|psi_trial> / <psi_trial|psi_trial> >= E_0

    where E_0 is the true ground state energy.

    For the quartic oscillator V(x) = lambda * x^4:
    - No exact analytical solution
    - Good test case for variational methods
    - Compare Gaussian and other trial functions

Trial wavefunctions:
    1. Gaussian: psi(x) = (alpha/pi)^(1/4) exp(-alpha*x^2/2)
    2. Gaussian with quartic correction: psi(x) ~ exp(-alpha*x^2/2 - beta*x^4/4)
    3. Linear combination of HO eigenstates
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar
from scipy.integrate import quad


def gaussian_trial(x, alpha):
    """
    Gaussian trial wavefunction.

    psi(x) = (alpha/pi)^(1/4) * exp(-alpha*x^2/2)

    Args:
        x: Position
        alpha: Width parameter

    Returns:
        Wavefunction value
    """
    return (alpha / np.pi)**0.25 * np.exp(-alpha * x**2 / 2)


def gaussian_energy_quartic(alpha, lam):
    """
    Expectation value of H = -d^2/dx^2 + lambda*x^4 with Gaussian trial function.

    <T> = alpha/2 (kinetic energy for Gaussian, hbar = m = 1)
    <V> = lambda * <x^4> = lambda * (3/(4*alpha^2))

    E(alpha) = alpha/2 + 3*lambda/(4*alpha^2)

    Args:
        alpha: Width parameter
        lam: Quartic potential strength

    Returns:
        Trial energy
    """
    T = alpha / 2  # Kinetic energy
    V = 3 * lam / (4 * alpha**2)  # Potential energy <x^4> for Gaussian
    return T + V


def optimal_alpha_quartic(lam):
    """
    Find optimal Gaussian width for quartic potential.

    dE/d(alpha) = 1/2 - 3*lambda/(2*alpha^3) = 0
    => alpha_opt = (3*lambda)^(1/3)

    Args:
        lam: Quartic potential strength

    Returns:
        Optimal alpha
    """
    return (3 * lam)**(1/3)


def variational_ground_state_quartic(lam):
    """
    Variational ground state energy for quartic potential.

    E_var = (3/4) * (3*lambda)^(1/3)

    Args:
        lam: Quartic potential strength

    Returns:
        Variational energy
    """
    alpha_opt = optimal_alpha_quartic(lam)
    return gaussian_energy_quartic(alpha_opt, lam)


def gaussian_with_quartic_correction(x, alpha, beta):
    """
    Improved trial function with quartic correction.

    psi(x) ~ exp(-alpha*x^2/2 - beta*x^4/4)

    Normalized numerically.
    """
    psi = np.exp(-alpha * x**2 / 2 - beta * x**4 / 4)
    return psi


def numerical_energy(params, lam, x_grid):
    """
    Compute energy numerically for arbitrary trial function.

    Args:
        params: Trial function parameters
        lam: Potential strength
        x_grid: Position grid

    Returns:
        Trial energy
    """
    if len(params) == 1:
        alpha = params[0]
        psi = gaussian_trial(x_grid, alpha)
    else:
        alpha, beta = params
        psi = gaussian_with_quartic_correction(x_grid, alpha, beta)

    dx = x_grid[1] - x_grid[0]

    # Normalize
    norm = np.trapz(psi**2, x_grid)
    psi = psi / np.sqrt(norm)

    # Kinetic energy: -d^2/dx^2 using finite differences
    d2psi = np.zeros_like(psi)
    d2psi[1:-1] = (psi[2:] - 2*psi[1:-1] + psi[:-2]) / dx**2

    T = -np.trapz(psi * d2psi, x_grid)

    # Potential energy: lambda * x^4
    V = lam * np.trapz(psi**2 * x_grid**4, x_grid)

    return T + V


def exact_diagonalization_quartic(lam, n_basis=50, x_max=10, n_grid=1000):
    """
    Find exact ground state energy by diagonalizing in position basis.

    Args:
        lam: Quartic potential strength
        n_basis: Basis size (unused, using grid method)
        x_max: Grid extent
        n_grid: Number of grid points

    Returns:
        Ground state energy
    """
    x = np.linspace(-x_max, x_max, n_grid)
    dx = x[1] - x[0]

    # Hamiltonian matrix (finite difference)
    H = np.zeros((n_grid, n_grid))

    # Kinetic energy: -d^2/dx^2
    for i in range(n_grid):
        H[i, i] = 1 / dx**2 + lam * x[i]**4  # Diagonal
        if i > 0:
            H[i, i-1] = -0.5 / dx**2
        if i < n_grid - 1:
            H[i, i+1] = -0.5 / dx**2

    eigenvalues = np.linalg.eigvalsh(H)
    return eigenvalues[0]


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ===== Plot 1: Energy vs alpha for fixed lambda =====
    ax1 = axes[0, 0]

    lam = 1.0
    alphas = np.linspace(0.5, 3.0, 200)

    E_trial = [gaussian_energy_quartic(a, lam) for a in alphas]
    alpha_opt = optimal_alpha_quartic(lam)
    E_opt = variational_ground_state_quartic(lam)

    # Exact energy for comparison
    E_exact = exact_diagonalization_quartic(lam)

    ax1.plot(alphas, E_trial, 'b-', lw=2, label='Variational E(alpha)')
    ax1.axhline(E_exact, color='r', linestyle='--', lw=2, label=f'Exact E = {E_exact:.4f}')
    ax1.axvline(alpha_opt, color='g', linestyle=':', lw=2, label=f'Optimal alpha = {alpha_opt:.3f}')
    ax1.plot(alpha_opt, E_opt, 'go', markersize=10, label=f'E_var = {E_opt:.4f}')

    ax1.set_xlabel(r'Width Parameter $\alpha$')
    ax1.set_ylabel(r'Energy')
    ax1.set_title(r'Variational Energy vs $\alpha$ for $V(x) = \lambda x^4$' + f'\n(lambda = {lam})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ===== Plot 2: Comparison of trial wavefunctions =====
    ax2 = axes[0, 1]

    x = np.linspace(-3, 3, 500)

    # Exact ground state (numerical)
    H_matrix = np.zeros((len(x), len(x)))
    dx = x[1] - x[0]
    for i in range(len(x)):
        H_matrix[i, i] = 1 / dx**2 + lam * x[i]**4
        if i > 0:
            H_matrix[i, i-1] = -0.5 / dx**2
        if i < len(x) - 1:
            H_matrix[i, i+1] = -0.5 / dx**2

    eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)
    psi_exact = eigenvectors[:, 0]
    psi_exact = psi_exact / np.sqrt(np.trapz(psi_exact**2, x))
    if psi_exact[len(x)//2] < 0:
        psi_exact = -psi_exact

    # Gaussian trial
    psi_gauss = gaussian_trial(x, alpha_opt)

    # Better trial with quartic correction
    x_grid = np.linspace(-5, 5, 500)
    result = minimize(numerical_energy, [alpha_opt, 0.1], args=(lam, x_grid),
                     method='Nelder-Mead')
    alpha_best, beta_best = result.x
    psi_improved = gaussian_with_quartic_correction(x, alpha_best, beta_best)
    psi_improved = psi_improved / np.sqrt(np.trapz(psi_improved**2, x))

    ax2.plot(x, psi_exact**2, 'k-', lw=2, label='Exact |psi|^2')
    ax2.plot(x, psi_gauss**2, 'b--', lw=2, label=f'Gaussian (alpha={alpha_opt:.2f})')
    ax2.plot(x, psi_improved**2, 'r:', lw=2, label='Improved trial')

    # Show potential (scaled)
    V = lam * x**4
    ax2.fill_between(x, 0, V/20, alpha=0.2, color='gray', label='V(x)/20')

    ax2.set_xlabel('Position x')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('Trial Wavefunctions vs Exact Ground State\n(Quartic potential)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(0, 1.2)

    # ===== Plot 3: Variational energy vs lambda =====
    ax3 = axes[1, 0]

    lambdas = np.logspace(-2, 1, 50)

    E_variational = [variational_ground_state_quartic(l) for l in lambdas]
    E_exact_vals = [exact_diagonalization_quartic(l) for l in lambdas]

    ax3.loglog(lambdas, E_variational, 'b-', lw=2, label='Variational (Gaussian)')
    ax3.loglog(lambdas, E_exact_vals, 'r--', lw=2, label='Exact')

    # Scaling: E ~ lambda^(1/3) for quartic potential
    ax3.loglog(lambdas, 0.68 * lambdas**(1/3), 'g:', lw=2, label=r'$\propto \lambda^{1/3}$')

    ax3.set_xlabel(r'Coupling $\lambda$')
    ax3.set_ylabel('Ground State Energy')
    ax3.set_title(r'Ground State Energy Scaling: $E \propto \lambda^{1/3}$')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # ===== Plot 4: Variational accuracy =====
    ax4 = axes[1, 1]

    relative_error = [(E_v - E_e) / E_e * 100 for E_v, E_e in zip(E_variational, E_exact_vals)]

    ax4.semilogx(lambdas, relative_error, 'b-', lw=2)
    ax4.axhline(0, color='gray', linestyle='--', alpha=0.5)

    ax4.set_xlabel(r'Coupling $\lambda$')
    ax4.set_ylabel('Relative Error (%)')
    ax4.set_title('Variational Method Accuracy\n(Gaussian trial function)')
    ax4.grid(True, alpha=0.3)

    # Add text annotation
    avg_error = np.mean(relative_error)
    ax4.text(0.5, 0.9, f'Average error: {avg_error:.2f}%',
             transform=ax4.transAxes, fontsize=12,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Variational Method for Quartic Potential\n' +
                 r'$H = -\frac{\hbar^2}{2m}\frac{d^2}{dx^2} + \lambda x^4$',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'variational_quartic.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'variational_quartic.png')}")

    # Print numerical results
    print("\n=== Variational Method Results ===")
    print(f"\nQuartic potential V(x) = lambda * x^4")
    print(f"\nGaussian trial function: psi(x) ~ exp(-alpha*x^2/2)")
    print(f"\nFor lambda = 1.0:")
    print(f"  Optimal alpha = {alpha_opt:.4f}")
    print(f"  Variational energy = {E_opt:.6f}")
    print(f"  Exact energy = {E_exact:.6f}")
    print(f"  Relative error = {(E_opt - E_exact)/E_exact * 100:.3f}%")

    print(f"\nImproved trial with quartic correction:")
    print(f"  alpha = {alpha_best:.4f}, beta = {beta_best:.4f}")
    print(f"  Energy = {result.fun:.6f}")
    print(f"  Relative error = {(result.fun - E_exact)/E_exact * 100:.3f}%")


if __name__ == "__main__":
    main()
