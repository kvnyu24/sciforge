"""
Experiment 182: 3-Qubit Bit-Flip Error Correction Code

Demonstrates the simplest quantum error correction code that protects against
single bit-flip (X) errors.

Physics:
    The 3-qubit bit-flip code encodes one logical qubit in three physical qubits:

    |0>_L = |000>
    |1>_L = |111>

    General logical state: |psi>_L = alpha|000> + beta|111>

    Error correction:
    - Can detect and correct any single bit-flip (X) error
    - Uses majority voting (syndrome measurement)
    - Cannot correct phase errors or multiple errors

    Syndrome measurement (without destroying quantum info):
    - Measure Z_0 Z_1 parity (same or different?)
    - Measure Z_1 Z_2 parity

    Syndrome table:
    Syndrome | Error | Correction
    (0, 0)   | None  | None
    (1, 0)   | X_0   | Apply X_0
    (1, 1)   | X_1   | Apply X_1
    (0, 1)   | X_2   | Apply X_2
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


# Pauli gates
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def tensor(*args):
    """Tensor product of multiple matrices."""
    result = args[0]
    for m in args[1:]:
        result = np.kron(result, m)
    return result


def encode_bit_flip(psi):
    """
    Encode a single qubit state into 3-qubit bit-flip code.

    |0> -> |000>
    |1> -> |111>
    |psi> = alpha|0> + beta|1> -> alpha|000> + beta|111>

    Args:
        psi: Single qubit state [alpha, beta]

    Returns:
        Encoded 3-qubit state (8-element array)
    """
    alpha, beta = psi
    encoded = np.zeros(8, dtype=complex)
    encoded[0] = alpha  # |000> has index 0
    encoded[7] = beta   # |111> has index 7
    return encoded


def decode_bit_flip(encoded_state):
    """
    Decode 3-qubit state back to single qubit.

    Projects onto code space and extracts logical qubit.

    Args:
        encoded_state: 3-qubit state

    Returns:
        Decoded single qubit state
    """
    # Project onto code space: |000> and |111>
    alpha = encoded_state[0]  # Coefficient of |000>
    beta = encoded_state[7]   # Coefficient of |111>

    # Normalize
    norm = np.sqrt(np.abs(alpha)**2 + np.abs(beta)**2)
    if norm > 0:
        return np.array([alpha, beta]) / norm
    return np.array([1, 0])


def apply_error(state, error_type, qubit):
    """
    Apply error to specified qubit.

    Args:
        state: 3-qubit state
        error_type: 'X', 'Y', 'Z', or 'I'
        qubit: Qubit index (0, 1, or 2)

    Returns:
        State after error
    """
    error_ops = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
    E = error_ops[error_type]

    # Build full error operator
    ops = [I, I, I]
    ops[qubit] = E
    E_full = tensor(*ops)

    return E_full @ state


def measure_syndrome(state):
    """
    Measure error syndrome without destroying the state.

    Syndrome bits:
    s_01 = parity of qubits 0 and 1 (same=0, different=1)
    s_12 = parity of qubits 1 and 2

    In practice, uses ancilla qubits. Here we compute syndrome directly.

    Args:
        state: 3-qubit state

    Returns:
        (s_01, s_12): Syndrome bits
    """
    # Z0Z1 and Z1Z2 parity measurements
    # Calculate expectation values

    # Z0Z1 operator
    Z0Z1 = tensor(Z, Z, I)
    parity_01 = np.real(np.conj(state) @ Z0Z1 @ state)

    # Z1Z2 operator
    Z1Z2 = tensor(I, Z, Z)
    parity_12 = np.real(np.conj(state) @ Z1Z2 @ state)

    # Convert to binary syndrome
    # If parity is close to +1, syndrome bit is 0
    # If parity is close to -1, syndrome bit is 1
    s_01 = 0 if parity_01 > 0 else 1
    s_12 = 0 if parity_12 > 0 else 1

    return (s_01, s_12)


def correct_error(state, syndrome):
    """
    Apply correction based on syndrome.

    Syndrome table:
    (0, 0) -> No error
    (1, 0) -> Error on qubit 0, apply X_0
    (1, 1) -> Error on qubit 1, apply X_1
    (0, 1) -> Error on qubit 2, apply X_2

    Args:
        state: Corrupted 3-qubit state
        syndrome: (s_01, s_12)

    Returns:
        Corrected state
    """
    s_01, s_12 = syndrome

    if s_01 == 0 and s_12 == 0:
        return state  # No error
    elif s_01 == 1 and s_12 == 0:
        return apply_error(state, 'X', 0)  # Correct qubit 0
    elif s_01 == 1 and s_12 == 1:
        return apply_error(state, 'X', 1)  # Correct qubit 1
    else:  # s_01 == 0 and s_12 == 1
        return apply_error(state, 'X', 2)  # Correct qubit 2


def fidelity(psi1, psi2):
    """Calculate fidelity between two states."""
    return np.abs(np.conj(psi1) @ psi2)**2


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ===== Plot 1: Error correction demonstration =====
    ax1 = axes[0, 0]

    # Test state: arbitrary superposition
    psi_input = np.array([0.6, 0.8], dtype=complex)
    psi_input = psi_input / np.linalg.norm(psi_input)

    print("=== Bit-Flip Code Demonstration ===")
    print(f"\nInput state: |psi> = {psi_input}")

    # Encode
    encoded = encode_bit_flip(psi_input)
    print(f"Encoded: {np.round(encoded, 4)}")

    # Test different errors
    errors = [('None', None, None), ('X_0', 'X', 0), ('X_1', 'X', 1), ('X_2', 'X', 2)]
    fidelities_uncorrected = []
    fidelities_corrected = []

    for error_name, error_type, qubit in errors:
        if error_type is not None:
            corrupted = apply_error(encoded, error_type, qubit)
        else:
            corrupted = encoded.copy()

        # Decode without correction
        decoded_uncorrected = decode_bit_flip(corrupted)
        fid_uncorr = fidelity(psi_input, decoded_uncorrected)
        fidelities_uncorrected.append(fid_uncorr)

        # Measure syndrome and correct
        syndrome = measure_syndrome(corrupted)
        corrected = correct_error(corrupted, syndrome)
        decoded_corrected = decode_bit_flip(corrected)
        fid_corr = fidelity(psi_input, decoded_corrected)
        fidelities_corrected.append(fid_corr)

        print(f"\n{error_name}:")
        print(f"  Syndrome: {syndrome}")
        print(f"  Fidelity (no correction): {fid_uncorr:.4f}")
        print(f"  Fidelity (corrected): {fid_corr:.4f}")

    x_pos = np.arange(len(errors))
    width = 0.35

    ax1.bar(x_pos - width/2, fidelities_uncorrected, width, label='Uncorrected', color='red', alpha=0.7)
    ax1.bar(x_pos + width/2, fidelities_corrected, width, label='Corrected', color='green', alpha=0.7)

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([e[0] for e in errors])
    ax1.set_ylabel('Fidelity')
    ax1.set_title('Single Bit-Flip Error Correction\n(3-qubit code)')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1.1)

    # ===== Plot 2: Two errors (uncorrectable) =====
    ax2 = axes[0, 1]

    # Test two-error cases
    two_errors = [
        ('X_0, X_1', [(0, 'X'), (1, 'X')]),
        ('X_0, X_2', [(0, 'X'), (2, 'X')]),
        ('X_1, X_2', [(1, 'X'), (2, 'X')])
    ]

    fid_two_uncorr = []
    fid_two_corr = []

    for name, error_list in two_errors:
        corrupted = encoded.copy()
        for qubit, error_type in error_list:
            corrupted = apply_error(corrupted, error_type, qubit)

        # Without correction
        decoded = decode_bit_flip(corrupted)
        fid_two_uncorr.append(fidelity(psi_input, decoded))

        # With (wrong) correction
        syndrome = measure_syndrome(corrupted)
        corrected = correct_error(corrupted, syndrome)
        decoded = decode_bit_flip(corrected)
        fid_two_corr.append(fidelity(psi_input, decoded))

    x_pos = np.arange(len(two_errors))

    ax2.bar(x_pos - width/2, fid_two_uncorr, width, label='Uncorrected', color='red', alpha=0.7)
    ax2.bar(x_pos + width/2, fid_two_corr, width, label='After correction', color='orange', alpha=0.7)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([e[0] for e in two_errors])
    ax2.set_ylabel('Fidelity')
    ax2.set_title('Two Bit-Flip Errors (UNCORRECTABLE)\n(Code fails: wrong syndrome)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 1.1)

    # ===== Plot 3: Phase errors (not protected) =====
    ax3 = axes[1, 0]

    phase_errors = [('None', None, None), ('Z_0', 'Z', 0), ('Z_1', 'Z', 1), ('Z_2', 'Z', 2)]
    fid_phase_uncorr = []
    fid_phase_corr = []

    for name, error_type, qubit in phase_errors:
        if error_type is not None:
            corrupted = apply_error(encoded, error_type, qubit)
        else:
            corrupted = encoded.copy()

        # Without correction
        decoded = decode_bit_flip(corrupted)
        fid_phase_uncorr.append(fidelity(psi_input, decoded))

        # With correction
        syndrome = measure_syndrome(corrupted)
        corrected = correct_error(corrupted, syndrome)
        decoded = decode_bit_flip(corrected)
        fid_phase_corr.append(fidelity(psi_input, decoded))

    x_pos = np.arange(len(phase_errors))

    ax3.bar(x_pos - width/2, fid_phase_uncorr, width, label='Uncorrected', color='red', alpha=0.7)
    ax3.bar(x_pos + width/2, fid_phase_corr, width, label='After correction', color='orange', alpha=0.7)

    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([e[0] for e in phase_errors])
    ax3.set_ylabel('Fidelity')
    ax3.set_title('Phase (Z) Errors (NOT PROTECTED)\n(Code only handles bit flips)')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(0, 1.1)

    # ===== Plot 4: Logical error rate vs physical error rate =====
    ax4 = axes[1, 1]

    # For bit-flip code with error prob p per qubit:
    # Logical error = P(2+ errors) ≈ 3*p^2 for small p

    p_physical = np.linspace(0, 0.3, 100)

    # Without coding: just p
    p_logical_no_code = p_physical

    # With 3-qubit code: P(2+) = C(3,2)*p^2*(1-p) + C(3,3)*p^3
    p_logical_coded = 3 * p_physical**2 * (1 - p_physical) + p_physical**3

    ax4.plot(p_physical, p_logical_no_code, 'r-', lw=2, label='No coding')
    ax4.plot(p_physical, p_logical_coded, 'b-', lw=2, label='3-qubit code')
    ax4.plot(p_physical, p_physical, 'r--', alpha=0.5)
    ax4.plot(p_physical, 3*p_physical**2, 'b--', alpha=0.5, label=r'$\approx 3p^2$')

    # Threshold
    p_threshold = 0.5 * (1 - 1/np.sqrt(3))
    ax4.axvline(p_threshold, color='green', linestyle=':', lw=2, label=f'Threshold ≈ {p_threshold:.3f}')

    ax4.set_xlabel('Physical error probability p')
    ax4.set_ylabel('Logical error probability')
    ax4.set_title('Logical vs Physical Error Rate\n(Code helps when p < threshold)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 0.3)
    ax4.set_ylim(0, 0.3)

    plt.suptitle('3-Qubit Bit-Flip Error Correction Code\n'
                 '|0>_L = |000>, |1>_L = |111>',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'bit_flip_code.png'), dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {os.path.join(output_dir, 'bit_flip_code.png')}")

    # Syndrome table
    print("\n=== Syndrome Table ===")
    print("Syndrome (s01, s12) | Error | Correction")
    print("--------------------|-------|------------")
    print("      (0, 0)        | None  | None")
    print("      (1, 0)        | X_0   | Apply X_0")
    print("      (1, 1)        | X_1   | Apply X_1")
    print("      (0, 1)        | X_2   | Apply X_2")


if __name__ == "__main__":
    main()
