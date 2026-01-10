"""
Experiment 181: Quantum Phase Estimation

This experiment demonstrates the quantum phase estimation (QPE) algorithm,
which estimates the eigenvalue (phase) of a unitary operator.

Physics/Algorithm:
    Given a unitary U with eigenstate |psi> and eigenvalue e^{2*pi*i*phi}:
        U|psi> = e^{2*pi*i*phi}|psi>

    QPE estimates phi to n-bit precision using:
    1. n ancilla qubits initialized to |0>
    2. Hadamard on all ancillas to create superposition
    3. Controlled-U^{2^k} operations for k = 0, 1, ..., n-1
    4. Inverse Quantum Fourier Transform (QFT)
    5. Measure ancillas to get binary approximation of phi

    The final state encodes phi in the ancilla register:
        |measured> = |phi_1 phi_2 ... phi_n>
        phi approx = 0.phi_1 phi_2 ... phi_n

    Resources:
    - n ancilla qubits for n-bit precision
    - O(n^2) controlled-U operations
    - Polynomial in n (exponential speedup over classical phase estimation)

    Applications:
    - Shor's algorithm (order finding)
    - Quantum simulation (energy estimation)
    - HHL algorithm (solving linear systems)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def hadamard():
    """Hadamard gate."""
    return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def controlled_phase(theta):
    """
    Controlled phase gate.

    |0><0| tensor I + |1><1| tensor P(theta)
    where P(theta) = diag(1, e^{i*theta})
    """
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, np.exp(1j * theta)]
    ], dtype=complex)


def tensor_product(*matrices):
    """Compute tensor product of multiple matrices."""
    result = matrices[0]
    for m in matrices[1:]:
        result = np.kron(result, m)
    return result


def apply_qft(state, n_qubits):
    """
    Apply Quantum Fourier Transform to state.

    QFT|j> = (1/sqrt(N)) * sum_k e^{2*pi*i*j*k/N} |k>

    Args:
        state: State vector
        n_qubits: Number of qubits

    Returns:
        Transformed state
    """
    N = 2**n_qubits
    # Build QFT matrix
    omega = np.exp(2j * np.pi / N)
    qft_matrix = np.zeros((N, N), dtype=complex)

    for j in range(N):
        for k in range(N):
            qft_matrix[j, k] = omega**(j * k)

    qft_matrix /= np.sqrt(N)
    return qft_matrix @ state


def apply_inverse_qft(state, n_qubits):
    """
    Apply inverse Quantum Fourier Transform.

    Args:
        state: State vector
        n_qubits: Number of qubits

    Returns:
        Transformed state
    """
    N = 2**n_qubits
    omega = np.exp(-2j * np.pi / N)  # Negative sign for inverse
    qft_inv_matrix = np.zeros((N, N), dtype=complex)

    for j in range(N):
        for k in range(N):
            qft_inv_matrix[j, k] = omega**(j * k)

    qft_inv_matrix /= np.sqrt(N)
    return qft_inv_matrix @ state


def run_phase_estimation(U, eigenstate, n_precision_qubits):
    """
    Run quantum phase estimation algorithm.

    Args:
        U: Unitary matrix (d x d)
        eigenstate: Eigenstate of U (d-dimensional)
        n_precision_qubits: Number of precision qubits

    Returns:
        Tuple of (measurement_probabilities, estimated_phases)
    """
    d = len(eigenstate)  # Dimension of eigenstate
    n_eigenstate_qubits = int(np.log2(d))

    N = 2**n_precision_qubits
    total_dim = N * d

    # Initialize state: |0>^n tensor |eigenstate>
    # |0...0> is the first basis state
    state = np.zeros(total_dim, dtype=complex)
    for i, amp in enumerate(eigenstate):
        state[i] = amp  # |0...0>|eigenstate>

    # Step 1: Apply Hadamard to all precision qubits
    # H^n = H tensor H tensor ... tensor H
    # Applied to first n qubits
    H_n = np.eye(1, dtype=complex)
    for _ in range(n_precision_qubits):
        H_n = np.kron(H_n, hadamard())

    # Full Hadamard on precision register
    H_full = np.kron(H_n, np.eye(d, dtype=complex))
    state = H_full @ state

    # Step 2: Apply controlled-U^{2^k} for k = 0, 1, ..., n-1
    # Control qubit k applies U^{2^k}
    for k in range(n_precision_qubits):
        # U^{2^k}
        U_power = np.linalg.matrix_power(U, 2**k)

        # Build controlled-U^{2^k}
        # Control is qubit k (0-indexed from left)
        # |0><0|_k tensor I_rest + |1><1|_k tensor (I_other tensor U)

        # Reshape state to apply controlled operation
        # The k-th qubit controls U on the eigenstate register

        # Total system: n precision qubits + eigenstate register
        # Qubit k controls, acts on last n_eigenstate_qubits

        # Build the controlled gate more explicitly
        control_qubit = n_precision_qubits - 1 - k  # Reverse order for QFT

        new_state = np.zeros_like(state)
        for idx in range(total_dim):
            # Decompose index
            precision_idx = idx // d
            eigenstate_idx = idx % d

            # Check if control qubit is |1>
            if (precision_idx >> control_qubit) & 1:
                # Apply U to eigenstate part
                for j in range(d):
                    new_idx = (precision_idx) * d + j
                    new_state[new_idx] += U_power[j, eigenstate_idx] * state[idx]
            else:
                # Identity
                new_state[idx] += state[idx]

        state = new_state

    # Step 3: Apply inverse QFT to precision register
    # Reshape and apply
    state_reshaped = state.reshape(N, d)

    for eigenstate_idx in range(d):
        precision_state = state_reshaped[:, eigenstate_idx]
        state_reshaped[:, eigenstate_idx] = apply_inverse_qft(precision_state, n_precision_qubits)

    state = state_reshaped.flatten()

    # Step 4: Compute measurement probabilities on precision register
    probs = np.zeros(N)
    for m in range(N):
        for e in range(d):
            idx = m * d + e
            probs[m] += np.abs(state[idx])**2

    # Estimated phases: m/N
    phases = np.arange(N) / N

    return probs, phases


def simple_phase_gate(phi):
    """
    Create a phase gate with eigenvalue e^{2*pi*i*phi}.

    P|1> = e^{2*pi*i*phi}|1>
    """
    return np.array([[1, 0], [0, np.exp(2j * np.pi * phi)]], dtype=complex)


def main():
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # ===== Plot 1: Basic QPE for known phase =====
    ax1 = axes[0, 0]

    # Known phase
    true_phi = 0.25  # = 1/4
    U = simple_phase_gate(true_phi)
    eigenstate = np.array([0, 1], dtype=complex)  # |1> is eigenstate

    n_qubits = 3
    probs, phases = run_phase_estimation(U, eigenstate, n_qubits)

    ax1.bar(phases, probs, width=0.05, alpha=0.7)
    ax1.axvline(true_phi, color='red', linestyle='--', lw=2, label=f'True phi = {true_phi}')

    ax1.set_xlabel('Estimated phase')
    ax1.set_ylabel('Probability')
    ax1.set_title(f'QPE with {n_qubits} precision qubits\nTrue phase = {true_phi}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ===== Plot 2: Effect of precision qubits =====
    ax2 = axes[0, 1]

    true_phi = 0.3  # Not exactly representable
    U = simple_phase_gate(true_phi)

    for n in [2, 3, 4, 5]:
        probs, phases = run_phase_estimation(U, eigenstate, n)
        ax2.plot(phases, probs, 'o-', lw=1.5, markersize=4, label=f'n={n}', alpha=0.7)

    ax2.axvline(true_phi, color='red', linestyle='--', lw=2, label=f'True = {true_phi}')

    ax2.set_xlabel('Estimated phase')
    ax2.set_ylabel('Probability')
    ax2.set_title(f'Effect of Precision Qubits\nTrue phase = {true_phi} (irrational binary)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ===== Plot 3: Peak probability vs n_qubits =====
    ax3 = axes[0, 2]

    true_phi = 1/3  # Irrational in binary

    n_range = range(2, 9)
    peak_probs = []
    errors = []

    for n in n_range:
        U = simple_phase_gate(true_phi)
        probs, phases = run_phase_estimation(U, eigenstate, n)
        peak_idx = np.argmax(probs)
        peak_probs.append(probs[peak_idx])
        errors.append(np.abs(phases[peak_idx] - true_phi))

    ax3.plot(list(n_range), peak_probs, 'bo-', lw=2, markersize=8, label='Peak probability')
    ax3.axhline(1.0, color='green', linestyle='--', alpha=0.5)

    ax3.set_xlabel('Number of precision qubits')
    ax3.set_ylabel('Peak probability')
    ax3.set_title(f'Success Probability vs Precision\nTrue phase = 1/3')
    ax3.grid(True, alpha=0.3)

    # Add error on secondary axis
    ax3_twin = ax3.twinx()
    ax3_twin.semilogy(list(n_range), errors, 'rs--', lw=2, markersize=8, label='Error')
    ax3_twin.set_ylabel('Phase estimation error', color='red')
    ax3_twin.tick_params(axis='y', labelcolor='red')

    # ===== Plot 4: Exact vs approximate phases =====
    ax4 = axes[1, 0]

    n_qubits = 4

    # Exact phases (can be represented exactly)
    exact_phases = [0, 0.25, 0.5, 0.75, 0.125]
    # Approximate phases (cannot be represented exactly)
    approx_phases = [0.1, 0.3, 0.7, 1/3, 1/7]

    exact_peaks = []
    approx_peaks = []

    for phi in exact_phases:
        U = simple_phase_gate(phi)
        probs, _ = run_phase_estimation(U, eigenstate, n_qubits)
        exact_peaks.append(np.max(probs))

    for phi in approx_phases:
        U = simple_phase_gate(phi)
        probs, _ = run_phase_estimation(U, eigenstate, n_qubits)
        approx_peaks.append(np.max(probs))

    x_pos = np.arange(len(exact_phases))
    width = 0.35

    ax4.bar(x_pos - width/2, exact_peaks, width, label='Exact (k/2^n)', alpha=0.7)
    ax4.bar(x_pos + width/2, approx_peaks, width, label='Approximate', alpha=0.7)

    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f'{p:.2f}' for p in exact_phases])
    ax4.set_xlabel('True phase')
    ax4.set_ylabel('Peak probability')
    ax4.set_title('Exact vs Approximate Binary Representations')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # ===== Plot 5: QPE for 2x2 rotation gate =====
    ax5 = axes[1, 1]

    # Rotation gate Rz(theta) has eigenvalues e^{+/-i*theta/2}
    theta = 2 * np.pi * 0.15

    Rz = np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)]
    ], dtype=complex)

    # Two eigenstates: |0> and |1>
    eigenstate_0 = np.array([1, 0], dtype=complex)
    eigenstate_1 = np.array([0, 1], dtype=complex)

    n_qubits = 5
    probs_0, phases = run_phase_estimation(Rz, eigenstate_0, n_qubits)
    probs_1, _ = run_phase_estimation(Rz, eigenstate_1, n_qubits)

    ax5.bar(phases - 0.01, probs_0, width=0.02, label='|0> eigenstate', alpha=0.7)
    ax5.bar(phases + 0.01, probs_1, width=0.02, label='|1> eigenstate', alpha=0.7)

    # True phases
    phi_0 = (1 - theta / (2*np.pi)) % 1  # e^{-i*theta/2} = e^{2pi*i*(1-theta/2pi)}
    phi_1 = (theta / (2*np.pi)) % 1

    ax5.axvline(phi_0, color='blue', linestyle='--', lw=2, alpha=0.7)
    ax5.axvline(phi_1, color='orange', linestyle='--', lw=2, alpha=0.7)

    ax5.set_xlabel('Estimated phase')
    ax5.set_ylabel('Probability')
    ax5.set_title('QPE for Rotation Gate Rz(theta)\nTwo eigenstates with phases +/- theta/2')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # ===== Plot 6: Quantum speedup illustration =====
    ax6 = axes[1, 2]

    # Classical: O(1/epsilon) to estimate phase to precision epsilon
    # Quantum: O(1/epsilon) controlled-U calls, but with n = log(1/epsilon) qubits

    precisions = 2**np.arange(2, 10)
    classical_cost = precisions  # O(1/epsilon)
    quantum_cost = np.log2(precisions)**2  # O(log^2(1/epsilon)) for controlled-U

    ax6.semilogy(np.log2(precisions), classical_cost, 'r-o', lw=2, markersize=8,
                label='Classical: O(1/epsilon)')
    ax6.semilogy(np.log2(precisions), quantum_cost, 'b-o', lw=2, markersize=8,
                label='Quantum: O(log^2(1/epsilon))')

    ax6.set_xlabel('Precision bits n = log2(1/epsilon)')
    ax6.set_ylabel('Number of operations (queries)')
    ax6.set_title('Quantum Phase Estimation Speedup\nExponential advantage in query complexity')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.suptitle('Quantum Phase Estimation (QPE) Algorithm\n'
                 r'Estimate eigenvalue $e^{2\pi i \phi}$ of unitary U',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'quantum_phase_estimation.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'quantum_phase_estimation.png')}")

    # Print numerical results
    print("\n=== Quantum Phase Estimation Results ===")

    print("\n1. Basic QPE (phi = 0.25, n = 3 qubits):")
    probs, phases = run_phase_estimation(simple_phase_gate(0.25), eigenstate, 3)
    best_idx = np.argmax(probs)
    print(f"   Estimated phase: {phases[best_idx]:.4f}")
    print(f"   Probability: {probs[best_idx]:.4f}")

    print("\n2. Precision analysis (phi = 1/3):")
    true_phi = 1/3
    for n in [3, 5, 7]:
        probs, phases = run_phase_estimation(simple_phase_gate(true_phi), eigenstate, n)
        best_idx = np.argmax(probs)
        error = abs(phases[best_idx] - true_phi)
        print(f"   n = {n}: est = {phases[best_idx]:.6f}, error = {error:.6f}, prob = {probs[best_idx]:.4f}")

    print("\n3. Algorithm steps:")
    print("   1. Initialize |0>^n |eigenstate>")
    print("   2. Apply H^n to precision register")
    print("   3. Apply controlled-U^{2^k} for k = 0,...,n-1")
    print("   4. Apply inverse QFT to precision register")
    print("   5. Measure to get binary approximation of phi")

    print("\n4. Complexity:")
    print("   - Precision: epsilon = 2^{-n}")
    print("   - Qubits: n = O(log(1/epsilon))")
    print("   - Controlled-U calls: O(2^n) = O(1/epsilon)")
    print("   - Query complexity: O(1/epsilon) (optimal)")


if __name__ == "__main__":
    main()
