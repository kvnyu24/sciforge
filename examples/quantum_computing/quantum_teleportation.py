"""
Experiment 180: Quantum Teleportation Circuit

Demonstrates the quantum teleportation protocol for transferring an arbitrary
quantum state using an entangled pair and classical communication.

Physics:
    Quantum teleportation transfers a qubit state from Alice to Bob:

    |psi>_in = alpha|0> + beta|1>  (unknown state to teleport)

    Protocol:
    1. Prepare entangled pair |Phi+> = (|00> + |11>)/sqrt(2) (shared by Alice & Bob)
    2. Alice performs Bell measurement on her input qubit and her half of pair
    3. Alice sends classical bits (2 bits) to Bob
    4. Bob applies correction based on classical bits

    Circuit:
    |psi>  --[CNOT]--[H]--[M]---> Classical bit m1
               |              \
    |0>  -----+--------[M]---> Classical bit m2
                              \
    |0>  -------[H]----[CNOT]--[X^m2]--[Z^m1]--> |psi> (teleported)

    Key points:
    - No information travels faster than light (classical bits required)
    - Original state is destroyed (no cloning)
    - Requires pre-shared entanglement
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


# Quantum gates
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

# CNOT gate (control on first qubit)
CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=complex)


def tensor(*args):
    """Compute tensor product of multiple matrices."""
    result = args[0]
    for m in args[1:]:
        result = np.kron(result, m)
    return result


def apply_gate(state, gate, qubit_indices, n_qubits):
    """
    Apply gate to specified qubits in multi-qubit state.

    Args:
        state: State vector
        gate: Gate matrix
        qubit_indices: Tuple of qubit indices (for multi-qubit gates)
        n_qubits: Total number of qubits

    Returns:
        New state vector
    """
    if isinstance(qubit_indices, int):
        qubit_indices = (qubit_indices,)

    # For single qubit gate
    if len(qubit_indices) == 1:
        q = qubit_indices[0]
        ops = [I] * n_qubits
        ops[q] = gate
        full_gate = tensor(*ops)
        return full_gate @ state

    # For CNOT (2-qubit gate)
    elif len(qubit_indices) == 2:
        control, target = qubit_indices
        new_state = np.zeros_like(state)
        dim = 2**n_qubits

        for i in range(dim):
            control_bit = (i >> (n_qubits - 1 - control)) & 1
            if control_bit == 1:
                # Flip target
                j = i ^ (1 << (n_qubits - 1 - target))
                new_state[j] = state[i]
            else:
                new_state[i] = state[i]

        return new_state

    return state


def measure_qubit(state, qubit, n_qubits, outcome=None):
    """
    Measure a single qubit.

    Args:
        state: State vector
        qubit: Qubit index to measure
        n_qubits: Total number of qubits
        outcome: Force specific outcome (for demonstration)

    Returns:
        (outcome, post_measurement_state)
    """
    dim = 2**n_qubits

    # Calculate probabilities
    prob_0 = 0
    prob_1 = 0

    for i in range(dim):
        qubit_bit = (i >> (n_qubits - 1 - qubit)) & 1
        if qubit_bit == 0:
            prob_0 += np.abs(state[i])**2
        else:
            prob_1 += np.abs(state[i])**2

    # Determine outcome
    if outcome is None:
        outcome = 0 if np.random.random() < prob_0 else 1

    # Collapse state
    new_state = np.zeros_like(state)
    for i in range(dim):
        qubit_bit = (i >> (n_qubits - 1 - qubit)) & 1
        if qubit_bit == outcome:
            new_state[i] = state[i]

    # Normalize
    norm = np.linalg.norm(new_state)
    if norm > 0:
        new_state = new_state / norm

    return outcome, new_state


def quantum_teleportation(psi_in, verbose=False):
    """
    Perform quantum teleportation protocol.

    Args:
        psi_in: Input state to teleport (2-element array)
        verbose: Print intermediate steps

    Returns:
        psi_out: Teleported state (2-element array)
        m1, m2: Measurement outcomes
    """
    # Normalize input
    psi_in = psi_in / np.linalg.norm(psi_in)

    # Initialize 3-qubit state: |psi>_A |0>_B |0>_C
    # Qubit 0: Alice's input (the state to teleport)
    # Qubit 1: Alice's half of Bell pair
    # Qubit 2: Bob's half of Bell pair

    state = tensor(psi_in, np.array([1, 0]), np.array([1, 0]))

    if verbose:
        print("Initial state: |psi>|0>|0>")

    # Step 1: Create Bell pair between qubits 1 and 2
    # H on qubit 1
    state = apply_gate(state, H, 1, 3)
    # CNOT with control=1, target=2
    state = apply_gate(state, CNOT, (1, 2), 3)

    if verbose:
        print("After Bell pair creation: |psi>|Phi+>")

    # Step 2: Alice's Bell measurement
    # CNOT with control=0, target=1
    state = apply_gate(state, CNOT, (0, 1), 3)
    # H on qubit 0
    state = apply_gate(state, H, 0, 3)

    if verbose:
        print("After Alice's operations")

    # Measure qubits 0 and 1
    m1, state = measure_qubit(state, 0, 3)
    m2, state = measure_qubit(state, 1, 3)

    if verbose:
        print(f"Measurement outcomes: m1={m1}, m2={m2}")

    # Step 3: Bob's corrections
    # Apply X^m2 to qubit 2
    if m2 == 1:
        state = apply_gate(state, X, 2, 3)
    # Apply Z^m1 to qubit 2
    if m1 == 1:
        state = apply_gate(state, Z, 2, 3)

    if verbose:
        print("After Bob's corrections")

    # Extract Bob's qubit (qubit 2)
    # The state should now be |m1>|m2>|psi>
    # Extract the amplitude of qubit 2

    psi_out = np.zeros(2, dtype=complex)
    dim = 8  # 2^3

    for i in range(dim):
        q2_bit = i & 1  # Qubit 2 is the last bit
        q0_bit = (i >> 2) & 1
        q1_bit = (i >> 1) & 1

        # Only look at terms with correct measurement outcomes
        if q0_bit == m1 and q1_bit == m2:
            psi_out[q2_bit] = state[i]

    # Normalize
    psi_out = psi_out / np.linalg.norm(psi_out)

    return psi_out, m1, m2


def fidelity(psi1, psi2):
    """Calculate fidelity |<psi1|psi2>|^2."""
    return np.abs(np.conj(psi1) @ psi2)**2


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ===== Plot 1: Teleportation of specific states =====
    ax1 = axes[0, 0]

    test_states = [
        (np.array([1, 0]), '|0>'),
        (np.array([0, 1]), '|1>'),
        (np.array([1, 1]) / np.sqrt(2), '|+>'),
        (np.array([1, -1]) / np.sqrt(2), '|->'),
        (np.array([1, 1j]) / np.sqrt(2), '|i>'),
        (np.array([1, -1j]) / np.sqrt(2), '|-i>')
    ]

    n_trials = 100
    fidelities = []

    for psi_in, name in test_states:
        trial_fidelities = []
        for _ in range(n_trials):
            psi_out, _, _ = quantum_teleportation(psi_in)
            f = fidelity(psi_in, psi_out)
            trial_fidelities.append(f)
        fidelities.append(trial_fidelities)

    positions = np.arange(len(test_states))
    bp = ax1.boxplot(fidelities, positions=positions, widths=0.6, patch_artist=True)

    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')

    ax1.set_xticks(positions)
    ax1.set_xticklabels([name for _, name in test_states])
    ax1.set_ylabel('Fidelity')
    ax1.set_title('Teleportation Fidelity for Different Input States\n(100 trials each)')
    ax1.axhline(1.0, color='red', linestyle='--', label='Perfect fidelity')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0.95, 1.05)

    # ===== Plot 2: Measurement outcomes distribution =====
    ax2 = axes[0, 1]

    # Teleport |+> state many times and collect measurement outcomes
    psi_plus = np.array([1, 1]) / np.sqrt(2)
    n_teleport = 1000

    outcomes = {'00': 0, '01': 0, '10': 0, '11': 0}

    for _ in range(n_teleport):
        _, m1, m2 = quantum_teleportation(psi_plus)
        key = f'{m1}{m2}'
        outcomes[key] += 1

    labels = list(outcomes.keys())
    counts = list(outcomes.values())

    ax2.bar(labels, counts, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
    ax2.axhline(n_teleport/4, color='gray', linestyle='--', label='Expected (uniform)')

    ax2.set_xlabel('Measurement outcomes (m1, m2)')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Bell Measurement Outcomes\n(N={n_teleport} teleportations of |+>)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # ===== Plot 3: Bloch sphere representation =====
    ax3 = axes[1, 0]

    # Generate random states on Bloch sphere and teleport
    n_random = 50
    thetas = np.random.uniform(0, np.pi, n_random)
    phis = np.random.uniform(0, 2*np.pi, n_random)

    input_bloch = []
    output_bloch = []

    for theta, phi in zip(thetas, phis):
        # Input state on Bloch sphere
        psi_in = np.array([np.cos(theta/2), np.exp(1j*phi)*np.sin(theta/2)])

        # Bloch vector of input
        rx_in = 2 * np.real(np.conj(psi_in[0]) * psi_in[1])
        ry_in = 2 * np.imag(np.conj(psi_in[0]) * psi_in[1])
        rz_in = np.abs(psi_in[0])**2 - np.abs(psi_in[1])**2
        input_bloch.append([rx_in, ry_in, rz_in])

        # Teleport
        psi_out, _, _ = quantum_teleportation(psi_in)

        # Bloch vector of output
        rx_out = 2 * np.real(np.conj(psi_out[0]) * psi_out[1])
        ry_out = 2 * np.imag(np.conj(psi_out[0]) * psi_out[1])
        rz_out = np.abs(psi_out[0])**2 - np.abs(psi_out[1])**2
        output_bloch.append([rx_out, ry_out, rz_out])

    input_bloch = np.array(input_bloch)
    output_bloch = np.array(output_bloch)

    # Plot x and z components
    ax3.scatter(input_bloch[:, 0], input_bloch[:, 2], c='blue', s=50, alpha=0.7, label='Input')
    ax3.scatter(output_bloch[:, 0], output_bloch[:, 2], c='red', s=50, marker='x', alpha=0.7, label='Output')

    # Draw unit circle
    theta_circle = np.linspace(0, 2*np.pi, 100)
    ax3.plot(np.cos(theta_circle), np.sin(theta_circle), 'k-', alpha=0.3)

    ax3.set_xlabel('Bloch vector r_x')
    ax3.set_ylabel('Bloch vector r_z')
    ax3.set_title('Bloch Sphere Projection (x-z plane)\n(Input and teleported output match)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    ax3.set_xlim(-1.2, 1.2)
    ax3.set_ylim(-1.2, 1.2)

    # ===== Plot 4: Circuit diagram (text-based) =====
    ax4 = axes[1, 1]
    ax4.axis('off')

    circuit_text = """
    Quantum Teleportation Circuit

    |psi>_in ----[.]-----[H]----[M]----> m1
                 |                      |
    |0> --------[X]------------[M]----> m2    (Classical
                 |                      |      communication)
                [H]-----[.]-------------+
                        |               |
    |0> ---------------[X]----[X^m2]--[Z^m1]----> |psi>_out


    Steps:
    1. Create Bell pair (H, CNOT on qubits 1,2)
    2. Alice: CNOT(0->1), H(0), Measure(0,1)
    3. Bob: Apply X if m2=1, Apply Z if m1=1

    Corrections Table:
    m1  m2  |  Bob's Operation
    --------|------------------
     0   0  |  Nothing
     0   1  |  X gate
     1   0  |  Z gate
     1   1  |  ZX = iY gate
    """

    ax4.text(0.05, 0.95, circuit_text, transform=ax4.transAxes,
             fontfamily='monospace', fontsize=10, verticalalignment='top')

    plt.suptitle('Quantum Teleportation Protocol\n'
                 'Transferring Quantum States Using Entanglement',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'quantum_teleportation.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'quantum_teleportation.png')}")

    # Print example
    print("\n=== Quantum Teleportation Example ===")
    psi_test = np.array([1, 1j]) / np.sqrt(2)
    print(f"\nInput state: |psi> = {psi_test}")
    print(f"Bloch vector: ({2*np.real(np.conj(psi_test[0])*psi_test[1]):.3f}, "
          f"{2*np.imag(np.conj(psi_test[0])*psi_test[1]):.3f}, "
          f"{np.abs(psi_test[0])**2 - np.abs(psi_test[1])**2:.3f})")

    psi_out, m1, m2 = quantum_teleportation(psi_test, verbose=True)
    print(f"\nOutput state: |psi> = {psi_out}")
    print(f"Fidelity: {fidelity(psi_test, psi_out):.6f}")


if __name__ == "__main__":
    main()
