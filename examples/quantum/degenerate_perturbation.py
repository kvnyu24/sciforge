"""
Experiment 166: Degenerate Perturbation Theory

This experiment demonstrates degenerate perturbation theory, which is needed
when unperturbed energy levels are degenerate (multiple states with same energy).

Physics:
    When E_n^(0) = E_m^(0) for n != m (degeneracy), standard perturbation theory
    fails because denominators (E_n - E_m) vanish.

    Solution: Diagonalize the perturbation V within the degenerate subspace.

    Procedure:
    1. Identify degenerate subspace D with states {|n_1>, |n_2>, ...}
    2. Construct perturbation matrix V_ij = <n_i|V|n_j>
    3. Diagonalize V in this subspace to find "good" states |n_i'>
    4. First-order energy corrections are eigenvalues of V in D
    5. Good states are the eigenvectors

    Example: Linear Stark effect in hydrogen
    - n=2 level has 4-fold degeneracy (2s, 2p_x, 2p_y, 2p_z)
    - Electric field perturbation V = eEz lifts degeneracy
    - First-order energy shift proportional to E (linear Stark)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag


def create_degenerate_hamiltonian(energies, degeneracies):
    """
    Create block-diagonal Hamiltonian with specified degeneracies.

    Args:
        energies: List of energy levels
        degeneracies: List of degeneracy for each level

    Returns:
        Hamiltonian matrix
    """
    blocks = []
    for E, d in zip(energies, degeneracies):
        blocks.append(E * np.eye(d))
    return block_diag(*blocks)


def perturbation_in_subspace(V, indices):
    """
    Extract perturbation matrix in a subspace.

    Args:
        V: Full perturbation matrix
        indices: Indices of states in subspace

    Returns:
        Subspace perturbation matrix
    """
    n = len(indices)
    V_sub = np.zeros((n, n), dtype=complex)
    for i, idx_i in enumerate(indices):
        for j, idx_j in enumerate(indices):
            V_sub[i, j] = V[idx_i, idx_j]
    return V_sub


def degenerate_perturbation_corrections(H0, V, deg_indices, lam=1.0):
    """
    Compute first-order corrections using degenerate perturbation theory.

    Args:
        H0: Unperturbed Hamiltonian
        V: Perturbation
        deg_indices: Indices of degenerate states
        lam: Perturbation strength

    Returns:
        Tuple of (energy_corrections, good_states)
    """
    # Extract V in degenerate subspace
    V_sub = perturbation_in_subspace(V, deg_indices)

    # Diagonalize V in this subspace
    eigenvalues, eigenvectors = np.linalg.eigh(V_sub)

    # First-order energy corrections
    E1 = lam * eigenvalues

    # Good states (linear combinations of original degenerate states)
    # expressed in full Hilbert space
    n_total = H0.shape[0]
    n_deg = len(deg_indices)
    good_states = np.zeros((n_total, n_deg), dtype=complex)

    for i in range(n_deg):
        for j, idx in enumerate(deg_indices):
            good_states[idx, i] = eigenvectors[j, i]

    return E1, good_states


def non_degenerate_second_order(H0, V, state_idx, E0, lam=1.0):
    """
    Second-order energy correction for non-degenerate case.

    E^(2)_n = sum_{m != n} |<m|V|n>|^2 / (E_n - E_m)

    Args:
        H0: Unperturbed Hamiltonian
        V: Perturbation
        state_idx: Index of state
        E0: Unperturbed energies
        lam: Perturbation strength

    Returns:
        Second-order energy correction
    """
    n = state_idx
    E2 = 0.0

    for m in range(len(E0)):
        if m != n and not np.isclose(E0[n], E0[m]):
            E2 += np.abs(V[m, n])**2 / (E0[n] - E0[m])

    return lam**2 * E2


def hydrogen_n2_stark():
    """
    Model hydrogen n=2 level in electric field (linear Stark effect).

    The n=2 level has states: |2s>, |2p_z>, |2p_x>, |2p_y>
    Only |2s> and |2p_z> are mixed by the perturbation V = eEz

    Matrix element <2s|z|2p_z> = -3*a0 (in atomic units)

    Returns:
        Tuple of (H0, V, labels)
    """
    # States: 0=2s, 1=2p_z, 2=2p_x, 3=2p_y
    # All have same unperturbed energy E_2 = -1/(2*2^2) = -0.125 Ry
    E2 = -0.125  # In Rydbergs

    H0 = E2 * np.eye(4, dtype=complex)

    # Perturbation V = eEz (in atomic units, taking E=1 for simplicity)
    # Only mixes s and p_z (selection rule: Delta m = 0)
    # <2s|z|2p_z> = -3*a0 (using a0 = 1)
    dipole_element = 3.0  # In atomic units

    V = np.zeros((4, 4), dtype=complex)
    V[0, 1] = dipole_element  # <2s|V|2p_z>
    V[1, 0] = dipole_element  # Hermitian

    labels = ['2s', '2p_z', '2p_x', '2p_y']

    return H0, V, labels


def two_level_crossing():
    """
    Simple two-level system showing degeneracy lifting.

    H0 has two degenerate states at E=0.
    V couples them, splitting the degeneracy.

    Returns:
        Tuple of (H0, V)
    """
    H0 = np.zeros((2, 2), dtype=complex)

    # Symmetric perturbation
    coupling = 1.0
    V = np.array([[0, coupling],
                  [coupling, 0]], dtype=complex)

    return H0, V


def compare_with_exact(H0, V, lam_values):
    """
    Compare perturbation theory with exact diagonalization.

    Args:
        H0: Unperturbed Hamiltonian
        V: Perturbation
        lam_values: Array of perturbation strengths

    Returns:
        Tuple of (exact_energies, pert_energies)
    """
    n = H0.shape[0]
    exact = np.zeros((len(lam_values), n))
    pert_1st = np.zeros((len(lam_values), n))

    # Get degenerate subspace (all states with same energy)
    E0 = np.diag(H0)
    unique_E = np.unique(np.round(E0, 10))

    for i, lam in enumerate(lam_values):
        # Exact
        H = H0 + lam * V
        exact[i] = np.sort(np.linalg.eigvalsh(H))

        # First-order degenerate PT
        E_pert = []
        for E_level in unique_E:
            deg_idx = np.where(np.isclose(E0, E_level))[0]
            if len(deg_idx) > 1:
                # Degenerate case
                corrections, _ = degenerate_perturbation_corrections(H0, V, deg_idx, lam)
                for corr in corrections:
                    E_pert.append(E_level + corr)
            else:
                # Non-degenerate case
                idx = deg_idx[0]
                E1 = lam * V[idx, idx]
                E2 = non_degenerate_second_order(H0, V, idx, E0, lam)
                E_pert.append(E_level + E1 + E2)

        pert_1st[i] = np.sort(E_pert)

    return exact, pert_1st


def main():
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # ===== Plot 1: Simple two-level degeneracy lifting =====
    ax1 = axes[0, 0]

    H0_2lev, V_2lev = two_level_crossing()

    lam_vals = np.linspace(0, 1, 50)
    exact_2lev, pert_2lev = compare_with_exact(H0_2lev, V_2lev, lam_vals)

    ax1.plot(lam_vals, exact_2lev[:, 0], 'b-', lw=2, label='Exact E-')
    ax1.plot(lam_vals, exact_2lev[:, 1], 'r-', lw=2, label='Exact E+')
    ax1.plot(lam_vals, pert_2lev[:, 0], 'b--', lw=2, alpha=0.7, label='PT E-')
    ax1.plot(lam_vals, pert_2lev[:, 1], 'r--', lw=2, alpha=0.7, label='PT E+')

    ax1.set_xlabel('Perturbation strength lambda')
    ax1.set_ylabel('Energy')
    ax1.set_title('Two-Level Degeneracy Lifting\n(Exact vs Degenerate PT)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ===== Plot 2: Good states (eigenstates of V in subspace) =====
    ax2 = axes[0, 1]

    corrections, good_states = degenerate_perturbation_corrections(
        H0_2lev, V_2lev, [0, 1], lam=1.0)

    # Visualize good states
    state_labels = ['|1>', '|2>']
    x_pos = np.arange(2)
    width = 0.35

    ax2.bar(x_pos - width/2, np.abs(good_states[:, 0])**2, width,
           label=f'Good state 1 (E1={corrections[0]:.2f})', alpha=0.7)
    ax2.bar(x_pos + width/2, np.abs(good_states[:, 1])**2, width,
           label=f'Good state 2 (E1={corrections[1]:.2f})', alpha=0.7)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(state_labels)
    ax2.set_ylabel('Probability')
    ax2.set_title('Good States (Eigenstates of V in subspace)\n|+> = (|1>+|2>)/sqrt(2), |-> = (|1>-|2>)/sqrt(2)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # ===== Plot 3: Hydrogen n=2 Stark effect =====
    ax3 = axes[0, 2]

    H0_H, V_H, labels_H = hydrogen_n2_stark()

    # Electric field strength (in atomic units)
    E_field = np.linspace(0, 0.1, 50)

    exact_H = np.zeros((len(E_field), 4))
    for i, E in enumerate(E_field):
        H = H0_H + E * V_H
        exact_H[i] = np.sort(np.linalg.eigvalsh(H))

    E0 = np.real(H0_H[0, 0])
    colors = ['b', 'r', 'gray', 'gray']

    for j in range(4):
        ax3.plot(E_field, (exact_H[:, j] - E0) * 27.2, color=colors[j], lw=2)

    ax3.axhline(0, color='black', linestyle=':', alpha=0.5)
    ax3.set_xlabel('Electric field (a.u.)')
    ax3.set_ylabel('Energy shift (eV)')
    ax3.set_title('Linear Stark Effect (Hydrogen n=2)\nDegeneracy lifted by electric field')
    ax3.grid(True, alpha=0.3)

    # Add annotations
    ax3.text(0.08, 0.15, '|2s> + |2p_z>', color='r', fontsize=9)
    ax3.text(0.08, -0.15, '|2s> - |2p_z>', color='b', fontsize=9)
    ax3.text(0.08, 0.01, '|2p_x>, |2p_y>', color='gray', fontsize=9)

    # ===== Plot 4: 3-fold degeneracy example =====
    ax4 = axes[1, 0]

    # Create system with 3-fold degenerate ground state
    H0_3 = np.diag([0, 0, 0, 2, 3])

    # Perturbation that mixes degenerate states
    V_3 = np.zeros((5, 5), dtype=complex)
    V_3[0, 1] = V_3[1, 0] = 0.5
    V_3[0, 2] = V_3[2, 0] = 0.3
    V_3[1, 2] = V_3[2, 1] = 0.4
    # Also some non-degenerate mixing
    V_3[0, 3] = V_3[3, 0] = 0.1

    lam_vals_3 = np.linspace(0, 1, 50)
    exact_3, pert_3 = compare_with_exact(H0_3, V_3, lam_vals_3)

    for j in range(5):
        ax4.plot(lam_vals_3, exact_3[:, j], '-', lw=2, label=f'Level {j+1}')

    ax4.set_xlabel('Perturbation strength lambda')
    ax4.set_ylabel('Energy')
    ax4.set_title('3-fold Degeneracy Lifting\n(Three degenerate states at E=0)')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ===== Plot 5: Comparison with non-degenerate PT =====
    ax5 = axes[1, 1]

    # Two close but non-degenerate levels
    H0_close = np.diag([0, 0.001])  # Nearly degenerate
    V_close = np.array([[0.1, 0.5],
                        [0.5, -0.1]], dtype=complex)

    lam_close = np.linspace(0, 0.5, 50)

    # Exact
    exact_close = np.zeros((len(lam_close), 2))
    for i, lam in enumerate(lam_close):
        exact_close[i] = np.sort(np.linalg.eigvalsh(H0_close + lam * V_close))

    # Non-degenerate PT (will fail for small splitting)
    E0_close = np.diag(H0_close)
    nondegenPT = np.zeros((len(lam_close), 2))
    for i, lam in enumerate(lam_close):
        for j in range(2):
            E1 = lam * V_close[j, j]
            E2 = non_degenerate_second_order(H0_close, V_close, j, E0_close, lam)
            nondegenPT[i, j] = E0_close[j] + E1 + E2

    # Degenerate PT (treating as degenerate)
    degenPT = np.zeros((len(lam_close), 2))
    for i, lam in enumerate(lam_close):
        corrections, _ = degenerate_perturbation_corrections(
            np.zeros((2, 2)), V_close, [0, 1], lam)
        degenPT[i] = np.sort(corrections)

    ax5.plot(lam_close, exact_close[:, 0], 'b-', lw=2, label='Exact (lower)')
    ax5.plot(lam_close, exact_close[:, 1], 'r-', lw=2, label='Exact (upper)')
    ax5.plot(lam_close, nondegenPT[:, 0], 'b:', lw=2, label='Non-deg PT')
    ax5.plot(lam_close, nondegenPT[:, 1], 'r:', lw=2)
    ax5.plot(lam_close, degenPT[:, 0], 'b--', lw=2, label='Degen PT')
    ax5.plot(lam_close, degenPT[:, 1], 'r--', lw=2)

    ax5.set_xlabel('Perturbation strength lambda')
    ax5.set_ylabel('Energy')
    ax5.set_title('Nearly Degenerate Case\n(Non-degenerate PT fails, degenerate PT works)')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # ===== Plot 6: Perturbation matrix visualization =====
    ax6 = axes[1, 2]

    # Show structure of perturbation in Stark effect
    # Full matrix with degeneracy structure
    H0_vis = create_degenerate_hamiltonian([0, 2], [4, 2])
    V_vis = np.random.randn(6, 6)
    V_vis = (V_vis + V_vis.T) / 2  # Symmetrize

    # Show the perturbation matrix structure
    im = ax6.imshow(np.abs(V_vis), cmap='Blues', aspect='auto')
    plt.colorbar(im, ax=ax6, label='|V_ij|')

    # Mark degenerate subspaces
    ax6.axhline(3.5, color='red', linestyle='-', lw=2)
    ax6.axvline(3.5, color='red', linestyle='-', lw=2)

    ax6.set_xlabel('State j')
    ax6.set_ylabel('State i')
    ax6.set_title('Perturbation Matrix Structure\n(Red lines separate degenerate subspaces)')
    ax6.set_xticks(range(6))
    ax6.set_yticks(range(6))

    # Add text annotations
    ax6.text(1.5, 1.5, 'Degenerate\nsubspace 1', ha='center', va='center',
            fontsize=10, color='white', weight='bold')
    ax6.text(4.5, 4.5, 'Subspace 2', ha='center', va='center',
            fontsize=10, color='black')

    plt.suptitle('Degenerate Perturbation Theory\n'
                 'Diagonalize V in degenerate subspace to find "good" states',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'degenerate_perturbation.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'degenerate_perturbation.png')}")

    # Print numerical results
    print("\n=== Degenerate Perturbation Theory Results ===")

    print("\n1. Two-level system (|1>, |2> degenerate at E=0):")
    corrections, good_states = degenerate_perturbation_corrections(
        H0_2lev, V_2lev, [0, 1], lam=1.0)
    print(f"   Energy corrections: E1 = {corrections[0]:.4f}, {corrections[1]:.4f}")
    print(f"   Good state 1: {good_states[0, 0]:.3f}|1> + {good_states[1, 0]:.3f}|2>")
    print(f"   Good state 2: {good_states[0, 1]:.3f}|1> + {good_states[1, 1]:.3f}|2>")

    print("\n2. Hydrogen n=2 Stark effect:")
    H0_H, V_H, labels_H = hydrogen_n2_stark()
    corrections_H, good_states_H = degenerate_perturbation_corrections(
        H0_H, V_H, [0, 1, 2, 3], lam=1.0)
    print(f"   First-order energy shifts (a.u.): {corrections_H}")
    print("   Only |2s> and |2p_z> are mixed (selection rules)")
    print("   |2p_x> and |2p_y> remain unshifted to first order")

    print("\n3. Key insight:")
    print("   Standard PT fails when E_n = E_m (division by zero)")
    print("   Solution: diagonalize V in degenerate subspace first")
    print("   The eigenstates of V are the 'good' basis states")


if __name__ == "__main__":
    main()
