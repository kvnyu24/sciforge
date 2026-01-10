"""
Experiment 175: Density Matrix Dephasing (Lindblad Master Equation)

Demonstrates open quantum system dynamics using the Lindblad master equation
for pure dephasing of a qubit.

Physics:
    The Lindblad master equation describes non-unitary evolution:

    d rho/dt = -i/hbar [H, rho] + sum_k gamma_k * D[L_k] rho

    where D[L] rho = L rho L^dag - (1/2){L^dag L, rho}

    For pure dephasing (T2 process):
    - Lindblad operator: L = sqrt(gamma_phi) * sigma_z
    - Off-diagonal elements decay: rho_01(t) -> rho_01(0) * exp(-gamma_phi * t)
    - Diagonal elements (populations) unchanged
    - Bloch vector shrinks in xy-plane

    For amplitude damping (T1 process):
    - Lindblad operator: L = sqrt(gamma_1) * sigma_-
    - Population decays to ground state
    - T2 <= 2*T1 always (dephasing is at least half of relaxation)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm


# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
sigma_plus = np.array([[0, 1], [0, 0]], dtype=complex)
sigma_minus = np.array([[0, 0], [1, 0]], dtype=complex)
I = np.eye(2, dtype=complex)


def commutator(A, B):
    """Compute [A, B] = AB - BA."""
    return A @ B - B @ A


def anticommutator(A, B):
    """Compute {A, B} = AB + BA."""
    return A @ B + B @ A


def dissipator(L, rho):
    """
    Lindblad dissipator.

    D[L] rho = L rho L^dag - (1/2){L^dag L, rho}

    Args:
        L: Lindblad (jump) operator
        rho: Density matrix

    Returns:
        D[L] rho
    """
    L_dag = L.T.conj()
    L_dag_L = L_dag @ L

    return L @ rho @ L_dag - 0.5 * anticommutator(L_dag_L, rho)


def lindblad_rhs(rho, H, lindblad_ops, hbar=1.0):
    """
    Right-hand side of Lindblad master equation.

    d rho/dt = -i/hbar [H, rho] + sum_k D[L_k] rho

    Args:
        rho: Density matrix
        H: Hamiltonian
        lindblad_ops: List of (L, gamma) tuples
        hbar: Reduced Planck constant

    Returns:
        d rho/dt
    """
    # Coherent evolution
    drho = -1j / hbar * commutator(H, rho)

    # Dissipation
    for L, gamma in lindblad_ops:
        drho += gamma * dissipator(L, rho)

    return drho


def evolve_lindblad(rho0, H, lindblad_ops, t_final, dt, hbar=1.0):
    """
    Evolve density matrix using Lindblad master equation.

    Uses simple Euler integration.

    Args:
        rho0: Initial density matrix
        H: Hamiltonian
        lindblad_ops: List of (L, gamma) tuples
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
        drho = lindblad_rhs(rho, H, lindblad_ops, hbar)
        rho = rho + drho * dt
        t += dt

        times.append(t)
        rho_history.append(rho.copy())

    return np.array(times), rho_history


def bloch_vector(rho):
    """
    Extract Bloch vector from density matrix.

    rho = (I + r.sigma) / 2

    r = (Tr(rho sigma_x), Tr(rho sigma_y), Tr(rho sigma_z))

    Args:
        rho: 2x2 density matrix

    Returns:
        Bloch vector (rx, ry, rz)
    """
    rx = np.real(np.trace(rho @ sigma_x))
    ry = np.real(np.trace(rho @ sigma_y))
    rz = np.real(np.trace(rho @ sigma_z))
    return np.array([rx, ry, rz])


def purity(rho):
    """
    Calculate purity Tr(rho^2).

    Pure state: purity = 1
    Maximally mixed: purity = 1/d
    """
    return np.real(np.trace(rho @ rho))


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Parameters
    omega = 1.0  # Qubit frequency
    hbar = 1.0
    dt = 0.01
    t_final = 10.0

    # Hamiltonian: H = (hbar omega / 2) sigma_z
    H = hbar * omega / 2 * sigma_z

    # Initial state: |+> = (|0> + |1>) / sqrt(2)
    psi_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho0 = np.outer(psi_plus, psi_plus.conj())

    # ===== Plot 1: Pure dephasing evolution =====
    ax1 = axes[0, 0]

    gamma_phi_values = [0.1, 0.3, 0.5, 1.0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(gamma_phi_values)))

    for gamma_phi, color in zip(gamma_phi_values, colors):
        L_dephasing = np.sqrt(gamma_phi) * sigma_z
        lindblad_ops = [(L_dephasing / np.sqrt(gamma_phi), gamma_phi)]

        times, rho_history = evolve_lindblad(rho0, H, lindblad_ops, t_final, dt, hbar)

        # Extract off-diagonal element magnitude
        coherence = [np.abs(rho[0, 1]) for rho in rho_history]

        # Analytical solution
        coherence_analytical = np.abs(rho0[0, 1]) * np.exp(-gamma_phi * times)

        ax1.plot(times, coherence, '-', color=color, lw=2, label=f'gamma_phi = {gamma_phi}')
        ax1.plot(times, coherence_analytical, '--', color=color, lw=1, alpha=0.7)

    ax1.set_xlabel('Time')
    ax1.set_ylabel('Coherence |rho_01|')
    ax1.set_title('Pure Dephasing (T2* process)\nSolid: numerical, Dashed: analytical')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    ax1.set_ylim(1e-2, 1)

    # ===== Plot 2: Bloch vector dynamics =====
    ax2 = axes[0, 1]

    gamma_phi = 0.3
    L_dephasing = sigma_z  # Un-normalized, gamma included separately
    lindblad_ops = [(L_dephasing, gamma_phi)]

    times, rho_history = evolve_lindblad(rho0, H, lindblad_ops, t_final, dt, hbar)

    bloch_vectors = [bloch_vector(rho) for rho in rho_history]
    rx = [b[0] for b in bloch_vectors]
    ry = [b[1] for b in bloch_vectors]
    rz = [b[2] for b in bloch_vectors]
    r_magnitude = [np.sqrt(b[0]**2 + b[1]**2 + b[2]**2) for b in bloch_vectors]

    ax2.plot(times, rx, 'r-', lw=2, label='r_x')
    ax2.plot(times, ry, 'g-', lw=2, label='r_y')
    ax2.plot(times, rz, 'b-', lw=2, label='r_z')
    ax2.plot(times, r_magnitude, 'k--', lw=2, label='|r|')

    ax2.set_xlabel('Time')
    ax2.set_ylabel('Bloch Vector Components')
    ax2.set_title(f'Bloch Vector Evolution (gamma_phi = {gamma_phi})\nDephasing shrinks xy-components')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='gray', linestyle='-', alpha=0.3)

    # ===== Plot 3: Amplitude damping (T1) =====
    ax3 = axes[1, 0]

    # Initial state: excited state |1>
    rho0_excited = np.array([[0, 0], [0, 1]], dtype=complex)

    gamma_1_values = [0.1, 0.3, 0.5, 1.0]

    for gamma_1, color in zip(gamma_1_values, colors):
        L_decay = sigma_minus  # Jump operator for decay
        lindblad_ops = [(L_decay, gamma_1)]

        times, rho_history = evolve_lindblad(rho0_excited, np.zeros((2,2)), lindblad_ops, t_final, dt, hbar)

        # Excited state population
        P_excited = [np.real(rho[1, 1]) for rho in rho_history]

        # Analytical: P_1(t) = exp(-gamma_1 * t)
        P_analytical = np.exp(-gamma_1 * times)

        ax3.plot(times, P_excited, '-', color=color, lw=2, label=f'gamma_1 = {gamma_1}')
        ax3.plot(times, P_analytical, '--', color=color, lw=1, alpha=0.7)

    ax3.set_xlabel('Time')
    ax3.set_ylabel('Excited State Population P_1')
    ax3.set_title('Amplitude Damping (T1 process)\nDecay to ground state')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # ===== Plot 4: Purity evolution =====
    ax4 = axes[1, 1]

    # Compare pure dephasing vs amplitude damping
    gamma_phi = 0.3
    gamma_1 = 0.3

    # Pure dephasing from |+>
    L_deph = sigma_z
    lindblad_ops_deph = [(L_deph, gamma_phi)]
    times, rho_deph = evolve_lindblad(rho0, H, lindblad_ops_deph, t_final, dt, hbar)
    purity_deph = [purity(rho) for rho in rho_deph]

    # Amplitude damping from |+>
    L_decay = sigma_minus
    lindblad_ops_decay = [(L_decay, gamma_1)]
    _, rho_decay = evolve_lindblad(rho0, np.zeros((2,2)), lindblad_ops_decay, t_final, dt, hbar)
    purity_decay = [purity(rho) for rho in rho_decay]

    # Combined: both T1 and T2
    lindblad_ops_both = [(L_deph, gamma_phi), (L_decay, gamma_1)]
    _, rho_both = evolve_lindblad(rho0, H, lindblad_ops_both, t_final, dt, hbar)
    purity_both = [purity(rho) for rho in rho_both]

    ax4.plot(times, purity_deph, 'b-', lw=2, label='Pure dephasing (T2)')
    ax4.plot(times, purity_decay, 'r-', lw=2, label='Amplitude damping (T1)')
    ax4.plot(times, purity_both, 'g-', lw=2, label='Both T1 and T2')

    ax4.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Maximally mixed')
    ax4.axhline(1.0, color='gray', linestyle=':', alpha=0.5, label='Pure state')

    ax4.set_xlabel('Time')
    ax4.set_ylabel('Purity Tr(rho^2)')
    ax4.set_title('Purity Evolution under Different Decoherence\n(gamma = 0.3)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0.4, 1.05)

    plt.suptitle('Lindblad Master Equation: Dephasing and Relaxation\n'
                 r'$\frac{d\rho}{dt} = -\frac{i}{\hbar}[H,\rho] + \sum_k \gamma_k D[L_k]\rho$',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'lindblad_dephasing.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'lindblad_dephasing.png')}")

    # Print results
    print("\n=== Lindblad Master Equation Results ===")
    print(f"\nQubit frequency omega = {omega}")
    print(f"\nPure dephasing (T2):")
    for gamma_phi in gamma_phi_values:
        T2 = 1 / gamma_phi
        print(f"  gamma_phi = {gamma_phi}: T2 = {T2:.2f}")

    print(f"\nAmplitude damping (T1):")
    for gamma_1 in gamma_1_values:
        T1 = 1 / gamma_1
        print(f"  gamma_1 = {gamma_1}: T1 = {T1:.2f}")

    print(f"\nRelation: T2 <= 2*T1 always holds")
    print(f"Physical interpretation:")
    print(f"  - T1: Energy relaxation (population decay)")
    print(f"  - T2: Coherence decay (off-diagonal elements)")
    print(f"  - T2* = 1/(1/T2 + 1/(2*T1)): Total dephasing rate")


if __name__ == "__main__":
    main()
