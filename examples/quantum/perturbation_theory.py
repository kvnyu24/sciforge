"""
Experiment 169: Time-Independent Perturbation Theory

Demonstrates non-degenerate perturbation theory by computing energy
corrections to first and second order for a perturbed quantum harmonic
oscillator and anharmonic perturbations.

Physics:
    For H = H_0 + lambda*V where lambda is small:

    First-order energy: E_n^(1) = <n|V|n>
    Second-order energy: E_n^(2) = sum_{m!=n} |<m|V|n>|^2 / (E_n^0 - E_m^0)

    First-order state: |n^(1)> = sum_{m!=n} <m|V|n>/(E_n^0 - E_m^0) |m>

Examples:
    1. Anharmonic oscillator: V = x^3 (cubic) or V = x^4 (quartic)
    2. Linear perturbation: V = x (electric field)
    3. Comparison with exact diagonalization
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial


def harmonic_oscillator_matrix_element(n, m, operator='x'):
    """
    Calculate matrix elements <n|O|m> for harmonic oscillator.

    Using a = (x + ip)/sqrt(2), a† = (x - ip)/sqrt(2) in units hbar=m=omega=1
    So x = (a + a†)/sqrt(2)

    <n|a|m> = sqrt(m) delta_{n,m-1}
    <n|a†|m> = sqrt(m+1) delta_{n,m+1}
    """
    if operator == 'x':
        # <n|x|m> = (1/sqrt(2))(<n|a|m> + <n|a†|m>)
        #         = (1/sqrt(2))(sqrt(m) delta_{n,m-1} + sqrt(m+1) delta_{n,m+1})
        if n == m - 1:
            return np.sqrt(m / 2)
        elif n == m + 1:
            return np.sqrt((m + 1) / 2)
        else:
            return 0.0
    elif operator == 'x2':
        # x^2 = (a + a†)^2 / 2 = (a^2 + a†^2 + 2a†a + 1)/2
        result = 0.0
        if n == m:
            result = (2 * m + 1) / 2  # (2a†a + 1)/2 = (2n + 1)/2
        elif n == m - 2:
            result = np.sqrt(m * (m - 1)) / 2
        elif n == m + 2:
            result = np.sqrt((m + 1) * (m + 2)) / 2
        return result
    elif operator == 'x3':
        # x^3 can be computed from x and x^2
        result = 0.0
        for k in range(max(0, m-3), min(m+4, 100)):
            for j in range(max(0, k-1), min(k+2, 100)):
                result += harmonic_oscillator_matrix_element(n, j, 'x') * \
                         harmonic_oscillator_matrix_element(j, k, 'x') * \
                         harmonic_oscillator_matrix_element(k, m, 'x')
        return result
    elif operator == 'x4':
        # x^4 = x^2 * x^2
        result = 0.0
        for k in range(max(0, m-4), min(m+5, 100)):
            result += harmonic_oscillator_matrix_element(n, k, 'x2') * \
                     harmonic_oscillator_matrix_element(k, m, 'x2')
        return result
    return 0.0


def build_matrix(n_states, operator):
    """Build matrix representation of operator in HO basis."""
    V = np.zeros((n_states, n_states))
    for i in range(n_states):
        for j in range(n_states):
            V[i, j] = harmonic_oscillator_matrix_element(i, j, operator)
    return V


def first_order_energy(V, state_idx):
    """First-order energy correction: E^(1)_n = <n|V|n>."""
    return V[state_idx, state_idx]


def second_order_energy(V, E0, state_idx):
    """
    Second-order energy correction.
    E^(2)_n = sum_{m!=n} |<m|V|n>|^2 / (E_n^0 - E_m^0)
    """
    n = state_idx
    E2 = 0.0
    n_states = len(E0)

    for m in range(n_states):
        if m != n:
            dE = E0[n] - E0[m]
            if abs(dE) > 1e-10:
                E2 += abs(V[m, n])**2 / dE

    return E2


def exact_diagonalization(H0, V, lam):
    """Diagonalize H = H0 + lam*V exactly."""
    H = H0 + lam * V
    eigenvalues = np.linalg.eigvalsh(H)
    return eigenvalues


def main():
    # Number of basis states
    n_states = 30

    # Unperturbed Hamiltonian: harmonic oscillator E_n = (n + 1/2) hbar*omega
    # Using natural units: hbar = m = omega = 1
    E0 = np.array([n + 0.5 for n in range(n_states)])
    H0 = np.diag(E0)

    # Build perturbation matrices
    V_x3 = build_matrix(n_states, 'x3')  # Cubic anharmonicity
    V_x4 = build_matrix(n_states, 'x4')  # Quartic anharmonicity

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ===== Plot 1: Energy corrections for quartic perturbation =====
    ax1 = axes[0, 0]

    lambdas = np.linspace(0, 0.1, 50)
    n_levels = 5

    for n in range(n_levels):
        E1 = first_order_energy(V_x4, n)
        E2 = second_order_energy(V_x4, E0, n)

        E_pert1 = E0[n] + lambdas * E1
        E_pert2 = E0[n] + lambdas * E1 + lambdas**2 * E2

        # Exact values
        E_exact = np.array([exact_diagonalization(H0, V_x4, lam)[n] for lam in lambdas])

        ax1.plot(lambdas, E_exact, '-', color=f'C{n}', lw=2, label=f'n={n} (exact)')
        ax1.plot(lambdas, E_pert1, '--', color=f'C{n}', lw=1.5, alpha=0.7)
        ax1.plot(lambdas, E_pert2, ':', color=f'C{n}', lw=2, alpha=0.7)

    ax1.set_xlabel(r'Coupling $\lambda$')
    ax1.set_ylabel(r'Energy $(h\omega)$')
    ax1.set_title(r'Quartic Perturbation $V = \lambda x^4$' + '\n(solid: exact, dashed: 1st order, dotted: 2nd order)')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ===== Plot 2: Perturbation theory breakdown =====
    ax2 = axes[0, 1]

    lambdas_wide = np.linspace(0, 0.5, 100)
    n = 0  # Ground state

    E1 = first_order_energy(V_x4, n)
    E2 = second_order_energy(V_x4, E0, n)

    E_pert1 = E0[n] + lambdas_wide * E1
    E_pert2 = E0[n] + lambdas_wide * E1 + lambdas_wide**2 * E2
    E_exact = np.array([exact_diagonalization(H0, V_x4, lam)[n] for lam in lambdas_wide])

    ax2.plot(lambdas_wide, E_exact, 'k-', lw=2, label='Exact')
    ax2.plot(lambdas_wide, E_pert1, 'b--', lw=2, label='1st order')
    ax2.plot(lambdas_wide, E_pert2, 'r:', lw=2, label='2nd order')

    # Show breakdown region
    ax2.axvspan(0.2, 0.5, alpha=0.2, color='red', label='Breakdown region')

    ax2.set_xlabel(r'Coupling $\lambda$')
    ax2.set_ylabel(r'Ground State Energy $(h\omega)$')
    ax2.set_title('Perturbation Theory Breakdown\n(Ground state with quartic perturbation)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ===== Plot 3: Cubic vs Quartic perturbation =====
    ax3 = axes[1, 0]

    lambdas = np.linspace(0, 0.05, 50)

    for n in range(3):
        # Cubic perturbation
        E1_cubic = first_order_energy(V_x3, n)
        E2_cubic = second_order_energy(V_x3, E0, n)
        E_cubic = E0[n] + lambdas * E1_cubic + lambdas**2 * E2_cubic

        # Quartic perturbation
        E1_quartic = first_order_energy(V_x4, n)
        E2_quartic = second_order_energy(V_x4, E0, n)
        E_quartic = E0[n] + lambdas * E1_quartic + lambdas**2 * E2_quartic

        ax3.plot(lambdas, E_cubic - E0[n], '-', color=f'C{n}', lw=2, label=f'n={n} (cubic)')
        ax3.plot(lambdas, E_quartic - E0[n], '--', color=f'C{n}', lw=2, label=f'n={n} (quartic)')

    ax3.set_xlabel(r'Coupling $\lambda$')
    ax3.set_ylabel(r'Energy Shift $\Delta E$ $(h\omega)$')
    ax3.set_title(r'Energy Shifts: $x^3$ (solid) vs $x^4$ (dashed)' + '\n(2nd order perturbation theory)')
    ax3.legend(ncol=2, fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ===== Plot 4: Matrix elements and selection rules =====
    ax4 = axes[1, 1]

    # Show structure of perturbation matrices
    V_display = V_x4[:10, :10]
    im = ax4.imshow(np.abs(V_display), cmap='Blues', aspect='auto')
    plt.colorbar(im, ax=ax4, label=r'$|V_{nm}|$')

    ax4.set_xlabel('State m')
    ax4.set_ylabel('State n')
    ax4.set_title(r'Matrix Elements $|\langle n|x^4|m\rangle|$' + '\n(Selection rules visible)')
    ax4.set_xticks(range(10))
    ax4.set_yticks(range(10))

    # Annotate non-zero elements
    for i in range(min(10, n_states)):
        for j in range(min(10, n_states)):
            if abs(V_display[i, j]) > 0.1:
                ax4.text(j, i, f'{V_display[i,j]:.1f}', ha='center', va='center',
                        fontsize=7, color='white' if V_display[i,j] > 2 else 'black')

    plt.suptitle('Time-Independent Perturbation Theory\n'
                 r'$H = H_0 + \lambda V$ for Quantum Harmonic Oscillator',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'perturbation_theory.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'perturbation_theory.png')}")

    # Print numerical results
    print("\n=== Perturbation Theory Results ===")
    print(f"\nQuartic perturbation (lambda = 0.05):")
    for n in range(5):
        E1 = first_order_energy(V_x4, n)
        E2 = second_order_energy(V_x4, E0, n)
        E_exact = exact_diagonalization(H0, V_x4, 0.05)[n]
        E_pert = E0[n] + 0.05 * E1 + 0.05**2 * E2
        print(f"  n={n}: E^(0)={E0[n]:.3f}, E^(1)={E1:.3f}, E^(2)={E2:.3f}")
        print(f"        E_pert={E_pert:.5f}, E_exact={E_exact:.5f}, error={abs(E_pert-E_exact):.6f}")


if __name__ == "__main__":
    main()
