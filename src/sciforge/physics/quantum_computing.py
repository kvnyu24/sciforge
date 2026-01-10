"""
Quantum Information and Computing module.

This module implements quantum information primitives, quantum gates,
circuits, algorithms, and error correction.

Classes:
    Qubits & Gates:
    - Qubit: Single qubit state
    - PauliGates: X, Y, Z gates
    - HadamardGate: Hadamard gate
    - PhaseGate: S, T phase gates
    - CNOTGate: Controlled-NOT gate
    - ToffoliGate: Three-qubit Toffoli gate
    - UniversalGateSet: Gate decomposition

    Quantum Circuits:
    - QuantumCircuit: Circuit representation
    - CircuitSimulator: State vector simulation
    - MeasurementBackend: Born rule sampling
    - DensityMatrixSimulator: Mixed state simulation

    Quantum Algorithms:
    - GroverSearch: Amplitude amplification
    - DeutschJozsa: Deterministic query algorithm
    - QuantumFourierTransform: QFT
    - PhaseEstimation: Eigenvalue estimation
    - VQE: Variational quantum eigensolver

    Error Correction:
    - BitFlipCode: 3-qubit bit flip code
    - PhaseFlipCode: Phase error code
    - ShorCode: 9-qubit Shor code
    - SteaneCode: 7-qubit CSS code
    - SurfaceCode: Topological code basics

    Entanglement Measures:
    - VonNeumannEntropy: Quantum entropy
    - Concurrence: Two-qubit entanglement
    - Negativity: Entanglement witness
    - MutualInformation: Quantum correlations
"""

import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, Tuple, Callable, Dict, List, Union
from scipy import linalg
from dataclasses import dataclass


# =============================================================================
# Basic Quantum States
# =============================================================================

class Qubit:
    """
    Single qubit quantum state.

    |ψ⟩ = α|0⟩ + β|1⟩ where |α|² + |β|² = 1

    Args:
        state: State vector [α, β] or None for |0⟩
    """

    def __init__(self, state: Optional[ArrayLike] = None):
        if state is None:
            self._state = np.array([1, 0], dtype=complex)
        else:
            self._state = np.asarray(state, dtype=complex)

        self._normalize()

    def _normalize(self) -> None:
        """Normalize state vector."""
        norm = np.linalg.norm(self._state)
        if norm > 0:
            self._state /= norm

    @property
    def state(self) -> np.ndarray:
        """Get state vector."""
        return self._state.copy()

    @classmethod
    def zero(cls) -> 'Qubit':
        """Create |0⟩ state."""
        return cls([1, 0])

    @classmethod
    def one(cls) -> 'Qubit':
        """Create |1⟩ state."""
        return cls([0, 1])

    @classmethod
    def plus(cls) -> 'Qubit':
        """Create |+⟩ = (|0⟩ + |1⟩)/√2 state."""
        return cls([1, 1])

    @classmethod
    def minus(cls) -> 'Qubit':
        """Create |-⟩ = (|0⟩ - |1⟩)/√2 state."""
        return cls([1, -1])

    @classmethod
    def from_bloch(cls, theta: float, phi: float) -> 'Qubit':
        """
        Create qubit from Bloch sphere angles.

        |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩

        Args:
            theta: Polar angle [0, π]
            phi: Azimuthal angle [0, 2π]

        Returns:
            Qubit instance
        """
        alpha = np.cos(theta / 2)
        beta = np.exp(1j * phi) * np.sin(theta / 2)
        return cls([alpha, beta])

    def bloch_vector(self) -> np.ndarray:
        """
        Get Bloch vector representation.

        n = (⟨X⟩, ⟨Y⟩, ⟨Z⟩)

        Returns:
            3D Bloch vector
        """
        rho = np.outer(self._state, self._state.conj())

        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]])

        return np.array([
            np.real(np.trace(rho @ X)),
            np.real(np.trace(rho @ Y)),
            np.real(np.trace(rho @ Z))
        ])

    def probability_zero(self) -> float:
        """Probability of measuring |0⟩."""
        return np.abs(self._state[0])**2

    def probability_one(self) -> float:
        """Probability of measuring |1⟩."""
        return np.abs(self._state[1])**2

    def measure(self) -> int:
        """
        Perform measurement in computational basis.

        Collapses state and returns outcome.

        Returns:
            0 or 1
        """
        if np.random.random() < self.probability_zero():
            self._state = np.array([1, 0], dtype=complex)
            return 0
        else:
            self._state = np.array([0, 1], dtype=complex)
            return 1

    def apply_gate(self, gate: np.ndarray) -> 'Qubit':
        """
        Apply single-qubit gate.

        Args:
            gate: 2×2 unitary matrix

        Returns:
            Self (for chaining)
        """
        self._state = gate @ self._state
        self._normalize()
        return self

    def density_matrix(self) -> np.ndarray:
        """Get density matrix ρ = |ψ⟩⟨ψ|."""
        return np.outer(self._state, self._state.conj())

    def fidelity(self, other: 'Qubit') -> float:
        """
        Fidelity with another pure state.

        F = |⟨ψ|φ⟩|²

        Args:
            other: Another qubit

        Returns:
            Fidelity [0, 1]
        """
        return np.abs(np.dot(self._state.conj(), other._state))**2


# =============================================================================
# Quantum Gates
# =============================================================================

class PauliGates:
    """
    Pauli gates X, Y, Z.

    X = [[0, 1], [1, 0]]    (bit flip)
    Y = [[0, -i], [i, 0]]   (bit and phase flip)
    Z = [[1, 0], [0, -1]]   (phase flip)
    """

    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)

    @classmethod
    def rotation_x(cls, theta: float) -> np.ndarray:
        """
        Rotation around X-axis.

        R_x(θ) = exp(-iθX/2) = cos(θ/2)I - i sin(θ/2)X

        Args:
            theta: Rotation angle

        Returns:
            2×2 rotation matrix
        """
        return np.cos(theta/2) * cls.I - 1j * np.sin(theta/2) * cls.X

    @classmethod
    def rotation_y(cls, theta: float) -> np.ndarray:
        """
        Rotation around Y-axis.

        R_y(θ) = cos(θ/2)I - i sin(θ/2)Y
        """
        return np.cos(theta/2) * cls.I - 1j * np.sin(theta/2) * cls.Y

    @classmethod
    def rotation_z(cls, theta: float) -> np.ndarray:
        """
        Rotation around Z-axis.

        R_z(θ) = exp(-iθZ/2) = diag(e^(-iθ/2), e^(iθ/2))
        """
        return np.array([
            [np.exp(-1j * theta / 2), 0],
            [0, np.exp(1j * theta / 2)]
        ], dtype=complex)


class HadamardGate:
    """
    Hadamard gate.

    H = (1/√2) [[1, 1], [1, -1]]

    Creates superposition: H|0⟩ = |+⟩, H|1⟩ = |-⟩
    """

    matrix = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

    @classmethod
    def apply(cls, qubit: Qubit) -> Qubit:
        """Apply Hadamard gate to qubit."""
        return qubit.apply_gate(cls.matrix)

    @classmethod
    def tensor_n(cls, n: int) -> np.ndarray:
        """
        n-qubit Hadamard H⊗n.

        Args:
            n: Number of qubits

        Returns:
            2^n × 2^n matrix
        """
        H_n = cls.matrix
        for _ in range(n - 1):
            H_n = np.kron(H_n, cls.matrix)
        return H_n


class PhaseGate:
    """
    Phase gates S and T.

    S = [[1, 0], [0, i]]      (π/2 phase)
    T = [[1, 0], [0, e^(iπ/4)]]  (π/4 phase)

    S = T²
    """

    S = np.array([[1, 0], [0, 1j]], dtype=complex)
    T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

    @classmethod
    def phase(cls, phi: float) -> np.ndarray:
        """
        General phase gate P(φ).

        P(φ) = [[1, 0], [0, e^(iφ)]]

        Args:
            phi: Phase angle

        Returns:
            Phase gate matrix
        """
        return np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=complex)

    @classmethod
    def apply_S(cls, qubit: Qubit) -> Qubit:
        """Apply S gate."""
        return qubit.apply_gate(cls.S)

    @classmethod
    def apply_T(cls, qubit: Qubit) -> Qubit:
        """Apply T gate."""
        return qubit.apply_gate(cls.T)


class CNOTGate:
    """
    Controlled-NOT (CNOT) gate.

    CNOT = [[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]]

    |00⟩ → |00⟩, |01⟩ → |01⟩, |10⟩ → |11⟩, |11⟩ → |10⟩
    """

    matrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=complex)

    @classmethod
    def apply(cls, state: np.ndarray, control: int, target: int,
              n_qubits: int) -> np.ndarray:
        """
        Apply CNOT to multi-qubit state.

        Args:
            state: State vector
            control: Control qubit index
            target: Target qubit index
            n_qubits: Total number of qubits

        Returns:
            New state vector
        """
        dim = 2**n_qubits
        new_state = np.zeros(dim, dtype=complex)

        for i in range(dim):
            # Check if control bit is 1
            if (i >> (n_qubits - 1 - control)) & 1:
                # Flip target bit
                j = i ^ (1 << (n_qubits - 1 - target))
                new_state[j] = state[i]
            else:
                new_state[i] = state[i]

        return new_state

    @classmethod
    def create_bell_state(cls, which: str = 'phi_plus') -> np.ndarray:
        """
        Create Bell state using CNOT.

        Args:
            which: 'phi_plus', 'phi_minus', 'psi_plus', 'psi_minus'

        Returns:
            Two-qubit state vector
        """
        H = HadamardGate.matrix
        I = np.eye(2)

        if which == 'phi_plus':
            # |Φ+⟩ = (|00⟩ + |11⟩)/√2
            state = np.array([1, 0, 0, 0], dtype=complex)  # |00⟩
        elif which == 'phi_minus':
            # |Φ-⟩ = (|00⟩ - |11⟩)/√2
            state = np.array([1, 0, 0, 0], dtype=complex)
        elif which == 'psi_plus':
            # |Ψ+⟩ = (|01⟩ + |10⟩)/√2
            state = np.array([0, 1, 0, 0], dtype=complex)  # |01⟩
        elif which == 'psi_minus':
            # |Ψ-⟩ = (|01⟩ - |10⟩)/√2
            state = np.array([0, 1, 0, 0], dtype=complex)
        else:
            raise ValueError(f"Unknown Bell state: {which}")

        # Apply H⊗I then CNOT
        H_I = np.kron(H, I)
        state = H_I @ state
        state = cls.matrix @ state

        if which in ['phi_minus', 'psi_minus']:
            Z_I = np.kron(PauliGates.Z, I)
            state = Z_I @ state

        return state


class ToffoliGate:
    """
    Toffoli (CCNOT) gate.

    Three-qubit gate: flips target only if both controls are |1⟩.
    """

    @classmethod
    def matrix(cls) -> np.ndarray:
        """Get 8×8 Toffoli matrix."""
        T = np.eye(8, dtype=complex)
        # Swap |110⟩ ↔ |111⟩
        T[6, 6] = 0
        T[6, 7] = 1
        T[7, 6] = 1
        T[7, 7] = 0
        return T

    @classmethod
    def apply(cls, state: np.ndarray) -> np.ndarray:
        """
        Apply Toffoli gate to 3-qubit state.

        Args:
            state: 8-element state vector

        Returns:
            New state vector
        """
        return cls.matrix() @ state


class UniversalGateSet:
    """
    Universal gate set and decompositions.

    Any unitary can be approximated using {H, T, CNOT}.
    """

    @classmethod
    def decompose_single_qubit(cls, U: np.ndarray,
                               precision: float = 1e-10) -> List[Tuple[str, float]]:
        """
        Decompose single-qubit unitary into rotations.

        U = e^(iα) R_z(β) R_y(γ) R_z(δ)

        Args:
            U: 2×2 unitary matrix
            precision: Numerical precision

        Returns:
            List of (gate_name, angle) tuples
        """
        # Extract global phase
        det = np.linalg.det(U)
        alpha = np.angle(det) / 2
        U = U * np.exp(-1j * alpha)

        # ZYZ decomposition
        # U = R_z(β) R_y(γ) R_z(δ)

        if np.abs(U[1, 0]) < precision:
            gamma = 0
            beta = np.angle(U[0, 0])
            delta = np.angle(U[1, 1])
        elif np.abs(U[0, 0]) < precision:
            gamma = np.pi
            beta = np.angle(U[0, 1])
            delta = np.angle(-U[1, 0])
        else:
            gamma = 2 * np.arccos(np.abs(U[0, 0]))
            beta = np.angle(U[1, 1]) - np.angle(U[1, 0])
            delta = np.angle(U[1, 1]) + np.angle(U[1, 0])

        return [
            ('global_phase', alpha),
            ('Rz', beta),
            ('Ry', gamma),
            ('Rz', delta)
        ]

    @classmethod
    def controlled_unitary(cls, U: np.ndarray) -> np.ndarray:
        """
        Create controlled version of single-qubit unitary.

        CU = |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ U

        Args:
            U: 2×2 unitary

        Returns:
            4×4 controlled unitary
        """
        I = np.eye(2, dtype=complex)
        P0 = np.array([[1, 0], [0, 0]], dtype=complex)  # |0⟩⟨0|
        P1 = np.array([[0, 0], [0, 1]], dtype=complex)  # |1⟩⟨1|

        return np.kron(P0, I) + np.kron(P1, U)


# =============================================================================
# Quantum Circuits
# =============================================================================

class QuantumCircuit:
    """
    Quantum circuit representation.

    Args:
        n_qubits: Number of qubits
    """

    def __init__(self, n_qubits: int):
        if n_qubits < 1:
            raise ValueError("Need at least 1 qubit")

        self.n_qubits = n_qubits
        self.gates = []  # List of (gate_name, qubits, params)

    def h(self, qubit: int) -> 'QuantumCircuit':
        """Add Hadamard gate."""
        self._validate_qubit(qubit)
        self.gates.append(('H', [qubit], {}))
        return self

    def x(self, qubit: int) -> 'QuantumCircuit':
        """Add Pauli-X gate."""
        self._validate_qubit(qubit)
        self.gates.append(('X', [qubit], {}))
        return self

    def y(self, qubit: int) -> 'QuantumCircuit':
        """Add Pauli-Y gate."""
        self._validate_qubit(qubit)
        self.gates.append(('Y', [qubit], {}))
        return self

    def z(self, qubit: int) -> 'QuantumCircuit':
        """Add Pauli-Z gate."""
        self._validate_qubit(qubit)
        self.gates.append(('Z', [qubit], {}))
        return self

    def s(self, qubit: int) -> 'QuantumCircuit':
        """Add S gate."""
        self._validate_qubit(qubit)
        self.gates.append(('S', [qubit], {}))
        return self

    def t(self, qubit: int) -> 'QuantumCircuit':
        """Add T gate."""
        self._validate_qubit(qubit)
        self.gates.append(('T', [qubit], {}))
        return self

    def rx(self, qubit: int, theta: float) -> 'QuantumCircuit':
        """Add R_x rotation."""
        self._validate_qubit(qubit)
        self.gates.append(('RX', [qubit], {'theta': theta}))
        return self

    def ry(self, qubit: int, theta: float) -> 'QuantumCircuit':
        """Add R_y rotation."""
        self._validate_qubit(qubit)
        self.gates.append(('RY', [qubit], {'theta': theta}))
        return self

    def rz(self, qubit: int, theta: float) -> 'QuantumCircuit':
        """Add R_z rotation."""
        self._validate_qubit(qubit)
        self.gates.append(('RZ', [qubit], {'theta': theta}))
        return self

    def cnot(self, control: int, target: int) -> 'QuantumCircuit':
        """Add CNOT gate."""
        self._validate_qubit(control)
        self._validate_qubit(target)
        if control == target:
            raise ValueError("Control and target must be different")
        self.gates.append(('CNOT', [control, target], {}))
        return self

    def cz(self, control: int, target: int) -> 'QuantumCircuit':
        """Add controlled-Z gate."""
        self._validate_qubit(control)
        self._validate_qubit(target)
        self.gates.append(('CZ', [control, target], {}))
        return self

    def toffoli(self, c1: int, c2: int, target: int) -> 'QuantumCircuit':
        """Add Toffoli gate."""
        for q in [c1, c2, target]:
            self._validate_qubit(q)
        self.gates.append(('TOFFOLI', [c1, c2, target], {}))
        return self

    def measure(self, qubit: int) -> 'QuantumCircuit':
        """Add measurement."""
        self._validate_qubit(qubit)
        self.gates.append(('MEASURE', [qubit], {}))
        return self

    def measure_all(self) -> 'QuantumCircuit':
        """Add measurement on all qubits."""
        for q in range(self.n_qubits):
            self.gates.append(('MEASURE', [q], {}))
        return self

    def _validate_qubit(self, qubit: int) -> None:
        """Validate qubit index."""
        if qubit < 0 or qubit >= self.n_qubits:
            raise ValueError(f"Qubit {qubit} out of range [0, {self.n_qubits})")

    def depth(self) -> int:
        """Calculate circuit depth."""
        # Simplified: count sequential gate layers
        return len(self.gates)

    def __str__(self) -> str:
        """String representation of circuit."""
        lines = [f"QuantumCircuit({self.n_qubits} qubits):"]
        for gate, qubits, params in self.gates:
            param_str = ', '.join(f"{k}={v:.4f}" for k, v in params.items())
            lines.append(f"  {gate}({', '.join(map(str, qubits))}) {param_str}")
        return '\n'.join(lines)


class CircuitSimulator:
    """
    State vector simulator for quantum circuits.

    Args:
        circuit: QuantumCircuit to simulate
    """

    def __init__(self, circuit: QuantumCircuit):
        self.circuit = circuit
        self.n_qubits = circuit.n_qubits
        self.dim = 2**self.n_qubits

        # Initialize to |0...0⟩
        self._state = np.zeros(self.dim, dtype=complex)
        self._state[0] = 1

    @property
    def state(self) -> np.ndarray:
        """Get current state vector."""
        return self._state.copy()

    def reset(self) -> None:
        """Reset to |0...0⟩ state."""
        self._state = np.zeros(self.dim, dtype=complex)
        self._state[0] = 1

    def run(self) -> np.ndarray:
        """
        Execute circuit and return final state.

        Returns:
            Final state vector
        """
        self.reset()

        for gate, qubits, params in self.circuit.gates:
            if gate == 'MEASURE':
                continue  # Skip measurements for state vector simulation

            self._apply_gate(gate, qubits, params)

        return self._state.copy()

    def _apply_gate(self, gate: str, qubits: List[int], params: Dict) -> None:
        """Apply a gate to the state."""
        if gate == 'H':
            self._apply_single_qubit_gate(HadamardGate.matrix, qubits[0])
        elif gate == 'X':
            self._apply_single_qubit_gate(PauliGates.X, qubits[0])
        elif gate == 'Y':
            self._apply_single_qubit_gate(PauliGates.Y, qubits[0])
        elif gate == 'Z':
            self._apply_single_qubit_gate(PauliGates.Z, qubits[0])
        elif gate == 'S':
            self._apply_single_qubit_gate(PhaseGate.S, qubits[0])
        elif gate == 'T':
            self._apply_single_qubit_gate(PhaseGate.T, qubits[0])
        elif gate == 'RX':
            self._apply_single_qubit_gate(
                PauliGates.rotation_x(params['theta']), qubits[0]
            )
        elif gate == 'RY':
            self._apply_single_qubit_gate(
                PauliGates.rotation_y(params['theta']), qubits[0]
            )
        elif gate == 'RZ':
            self._apply_single_qubit_gate(
                PauliGates.rotation_z(params['theta']), qubits[0]
            )
        elif gate == 'CNOT':
            self._apply_cnot(qubits[0], qubits[1])
        elif gate == 'CZ':
            self._apply_cz(qubits[0], qubits[1])
        elif gate == 'TOFFOLI':
            self._apply_toffoli(qubits[0], qubits[1], qubits[2])

    def _apply_single_qubit_gate(self, U: np.ndarray, qubit: int) -> None:
        """Apply single-qubit gate to specified qubit."""
        n = self.n_qubits

        # Build full operator via tensor products
        if qubit == 0:
            full_U = U
        else:
            full_U = np.eye(2)

        for i in range(1, n):
            if i == qubit:
                full_U = np.kron(full_U, U)
            else:
                full_U = np.kron(full_U, np.eye(2))

        self._state = full_U @ self._state

    def _apply_cnot(self, control: int, target: int) -> None:
        """Apply CNOT gate."""
        new_state = np.zeros_like(self._state)
        n = self.n_qubits

        for i in range(self.dim):
            control_bit = (i >> (n - 1 - control)) & 1
            if control_bit:
                j = i ^ (1 << (n - 1 - target))
                new_state[j] = self._state[i]
            else:
                new_state[i] = self._state[i]

        self._state = new_state

    def _apply_cz(self, control: int, target: int) -> None:
        """Apply controlled-Z gate."""
        n = self.n_qubits

        for i in range(self.dim):
            control_bit = (i >> (n - 1 - control)) & 1
            target_bit = (i >> (n - 1 - target)) & 1
            if control_bit and target_bit:
                self._state[i] *= -1

    def _apply_toffoli(self, c1: int, c2: int, target: int) -> None:
        """Apply Toffoli gate."""
        new_state = np.zeros_like(self._state)
        n = self.n_qubits

        for i in range(self.dim):
            c1_bit = (i >> (n - 1 - c1)) & 1
            c2_bit = (i >> (n - 1 - c2)) & 1

            if c1_bit and c2_bit:
                j = i ^ (1 << (n - 1 - target))
                new_state[j] = self._state[i]
            else:
                new_state[i] = self._state[i]

        self._state = new_state

    def probabilities(self) -> np.ndarray:
        """Get measurement probabilities for all basis states."""
        return np.abs(self._state)**2

    def sample(self, n_shots: int = 1024) -> Dict[str, int]:
        """
        Sample measurement outcomes.

        Args:
            n_shots: Number of measurements

        Returns:
            Dictionary of outcome counts
        """
        probs = self.probabilities()
        outcomes = np.random.choice(self.dim, size=n_shots, p=probs)

        counts = {}
        for outcome in outcomes:
            bitstring = format(outcome, f'0{self.n_qubits}b')
            counts[bitstring] = counts.get(bitstring, 0) + 1

        return counts


class MeasurementBackend:
    """
    Measurement backend implementing Born rule sampling.

    Args:
        n_qubits: Number of qubits
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.dim = 2**n_qubits

    def measure_computational_basis(self, state: np.ndarray,
                                    qubits: Optional[List[int]] = None) -> Tuple[str, np.ndarray]:
        """
        Measure in computational basis.

        Args:
            state: State vector
            qubits: Which qubits to measure (None = all)

        Returns:
            (outcome_string, post_measurement_state)
        """
        if qubits is None:
            qubits = list(range(self.n_qubits))

        probs = np.abs(state)**2
        outcome = np.random.choice(self.dim, p=probs)

        # Collapse state
        new_state = np.zeros_like(state)
        new_state[outcome] = 1

        # Get measured bits
        bits = format(outcome, f'0{self.n_qubits}b')
        measured_bits = ''.join(bits[q] for q in qubits)

        return measured_bits, new_state

    def measure_pauli(self, state: np.ndarray, qubit: int,
                      basis: str = 'Z') -> Tuple[int, np.ndarray]:
        """
        Measure single qubit in Pauli basis.

        Args:
            state: State vector
            qubit: Qubit to measure
            basis: 'X', 'Y', or 'Z'

        Returns:
            (outcome, post_measurement_state)
        """
        # Rotate to computational basis if needed
        if basis == 'X':
            H = HadamardGate.matrix
            state = self._apply_single_qubit(state, H, qubit)
        elif basis == 'Y':
            # Rotate Y eigenbasis to Z eigenbasis
            S_dag = PhaseGate.S.T.conj()
            H = HadamardGate.matrix
            state = self._apply_single_qubit(state, S_dag, qubit)
            state = self._apply_single_qubit(state, H, qubit)

        # Measure in Z basis
        probs = np.abs(state)**2
        outcome = np.random.choice(self.dim, p=probs)

        # Extract measured qubit value
        result = (outcome >> (self.n_qubits - 1 - qubit)) & 1

        # Collapse state
        new_state = np.zeros_like(state)
        new_state[outcome] = 1

        # Rotate back if needed
        if basis == 'X':
            new_state = self._apply_single_qubit(new_state, H, qubit)
        elif basis == 'Y':
            new_state = self._apply_single_qubit(new_state, H, qubit)
            new_state = self._apply_single_qubit(new_state, PhaseGate.S, qubit)

        return result, new_state

    def _apply_single_qubit(self, state: np.ndarray, U: np.ndarray,
                            qubit: int) -> np.ndarray:
        """Helper to apply single-qubit gate."""
        n = self.n_qubits
        full_U = np.eye(1)

        for i in range(n):
            if i == qubit:
                full_U = np.kron(full_U, U)
            else:
                full_U = np.kron(full_U, np.eye(2))

        return full_U @ state


class DensityMatrixSimulator:
    """
    Density matrix simulator for mixed states.

    ρ = Σ_i p_i |ψ_i⟩⟨ψ_i|

    Args:
        n_qubits: Number of qubits
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.dim = 2**n_qubits

        # Initialize to |0...0⟩⟨0...0|
        self._rho = np.zeros((self.dim, self.dim), dtype=complex)
        self._rho[0, 0] = 1

    @property
    def density_matrix(self) -> np.ndarray:
        """Get current density matrix."""
        return self._rho.copy()

    def set_state(self, state: np.ndarray) -> None:
        """Set pure state |ψ⟩⟨ψ|."""
        self._rho = np.outer(state, state.conj())

    def set_density_matrix(self, rho: np.ndarray) -> None:
        """Set density matrix directly."""
        self._rho = rho.copy()

    def apply_unitary(self, U: np.ndarray) -> None:
        """
        Apply unitary evolution ρ → UρU†.

        Args:
            U: Unitary matrix
        """
        self._rho = U @ self._rho @ U.T.conj()

    def apply_channel(self, kraus_ops: List[np.ndarray]) -> None:
        """
        Apply quantum channel via Kraus operators.

        ρ → Σ_k K_k ρ K_k†

        Args:
            kraus_ops: List of Kraus operators
        """
        new_rho = np.zeros_like(self._rho)
        for K in kraus_ops:
            new_rho += K @ self._rho @ K.T.conj()
        self._rho = new_rho

    def depolarizing_channel(self, p: float) -> None:
        """
        Apply depolarizing channel.

        ρ → (1-p)ρ + (p/3)(XρX + YρY + ZρZ)

        Args:
            p: Depolarization probability
        """
        if self.n_qubits != 1:
            raise ValueError("Single qubit depolarizing only")

        X, Y, Z = PauliGates.X, PauliGates.Y, PauliGates.Z

        self._rho = ((1 - p) * self._rho +
                     (p / 3) * (X @ self._rho @ X +
                               Y @ self._rho @ Y +
                               Z @ self._rho @ Z))

    def amplitude_damping(self, gamma: float) -> None:
        """
        Apply amplitude damping (spontaneous emission).

        K_0 = [[1, 0], [0, √(1-γ)]]
        K_1 = [[0, √γ], [0, 0]]

        Args:
            gamma: Damping probability
        """
        if self.n_qubits != 1:
            raise ValueError("Single qubit amplitude damping only")

        K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
        K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)

        self.apply_channel([K0, K1])

    def purity(self) -> float:
        """
        Calculate purity Tr(ρ²).

        Pure state: Tr(ρ²) = 1
        Maximally mixed: Tr(ρ²) = 1/d
        """
        return np.real(np.trace(self._rho @ self._rho))

    def trace(self) -> float:
        """Trace of density matrix (should be 1)."""
        return np.real(np.trace(self._rho))


# =============================================================================
# Quantum Algorithms
# =============================================================================

class GroverSearch:
    """
    Grover's search algorithm.

    Finds marked item in unstructured database with O(√N) queries.

    Args:
        n_qubits: Number of qubits (database size N = 2^n)
        oracle: Function marking target state(s)
    """

    def __init__(self, n_qubits: int, oracle: Callable[[int], bool]):
        self.n_qubits = n_qubits
        self.N = 2**n_qubits
        self.oracle = oracle

        # Count marked items
        self.n_marked = sum(1 for i in range(self.N) if oracle(i))

    def optimal_iterations(self) -> int:
        """
        Optimal number of Grover iterations.

        k_opt ≈ (π/4)√(N/M)

        where M is number of marked items.
        """
        if self.n_marked == 0:
            return 0
        return int(np.pi / 4 * np.sqrt(self.N / self.n_marked))

    def oracle_matrix(self) -> np.ndarray:
        """
        Construct oracle matrix U_f.

        U_f|x⟩ = (-1)^f(x)|x⟩
        """
        U = np.eye(self.N, dtype=complex)
        for i in range(self.N):
            if self.oracle(i):
                U[i, i] = -1
        return U

    def diffusion_matrix(self) -> np.ndarray:
        """
        Construct diffusion operator.

        U_s = 2|s⟩⟨s| - I

        where |s⟩ = H^⊗n|0⟩^⊗n
        """
        # |s⟩ is uniform superposition
        s = np.ones(self.N) / np.sqrt(self.N)
        return 2 * np.outer(s, s) - np.eye(self.N)

    def run(self, n_iterations: Optional[int] = None) -> int:
        """
        Run Grover's algorithm.

        Args:
            n_iterations: Number of iterations (default: optimal)

        Returns:
            Found marked item
        """
        if n_iterations is None:
            n_iterations = self.optimal_iterations()

        # Initialize to uniform superposition
        state = np.ones(self.N, dtype=complex) / np.sqrt(self.N)

        U_f = self.oracle_matrix()
        U_s = self.diffusion_matrix()

        # Grover iterations
        for _ in range(n_iterations):
            state = U_f @ state  # Oracle
            state = U_s @ state  # Diffusion

        # Measure
        probs = np.abs(state)**2
        return np.random.choice(self.N, p=probs)

    def success_probability(self, n_iterations: int) -> float:
        """
        Calculate success probability after k iterations.

        P_success = sin²((2k+1)θ)

        where sin²(θ) = M/N
        """
        theta = np.arcsin(np.sqrt(self.n_marked / self.N))
        return np.sin((2 * n_iterations + 1) * theta)**2


class DeutschJozsa:
    """
    Deutsch-Jozsa algorithm.

    Determines if function f:{0,1}^n → {0,1} is constant or balanced
    with a single query.

    Args:
        n_qubits: Number of input qubits
        oracle: Oracle function
    """

    def __init__(self, n_qubits: int, oracle: Callable[[int], int]):
        self.n_qubits = n_qubits
        self.N = 2**n_qubits
        self.oracle = oracle

    def run(self) -> str:
        """
        Run Deutsch-Jozsa algorithm.

        Returns:
            'constant' or 'balanced'
        """
        n = self.n_qubits
        N = self.N

        # Initialize |0⟩^n |1⟩
        state = np.zeros(2 * N, dtype=complex)
        state[1] = 1  # |0...01⟩

        # Apply H^⊗(n+1)
        H = HadamardGate.matrix
        H_full = H
        for _ in range(n):
            H_full = np.kron(H_full, H)
        state = H_full @ state

        # Apply oracle U_f|x⟩|y⟩ = |x⟩|y⊕f(x)⟩
        new_state = np.zeros_like(state)
        for i in range(N):
            for y in [0, 1]:
                idx_in = 2 * i + y
                f_x = self.oracle(i)
                idx_out = 2 * i + (y ^ f_x)
                new_state[idx_out] += state[idx_in]
        state = new_state

        # Apply H^⊗n to first n qubits
        H_n = HadamardGate.tensor_n(n)
        state_reshaped = state.reshape(N, 2)
        state_reshaped = (H_n @ state_reshaped.T).T
        state = state_reshaped.flatten()

        # Measure first n qubits
        probs = np.abs(state[:N])**2 + np.abs(state[N:])**2

        # If P(0...0) ≈ 1, function is constant
        if probs[0] > 0.5:
            return 'constant'
        return 'balanced'


class QuantumFourierTransform:
    """
    Quantum Fourier Transform.

    QFT|j⟩ = (1/√N) Σ_k e^(2πijk/N)|k⟩

    Args:
        n_qubits: Number of qubits
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.N = 2**n_qubits

    def matrix(self) -> np.ndarray:
        """
        Get QFT matrix.

        QFT[j,k] = ω^(jk) / √N

        where ω = e^(2πi/N)
        """
        N = self.N
        omega = np.exp(2j * np.pi / N)

        QFT = np.zeros((N, N), dtype=complex)
        for j in range(N):
            for k in range(N):
                QFT[j, k] = omega**(j * k)

        return QFT / np.sqrt(N)

    def inverse_matrix(self) -> np.ndarray:
        """Get inverse QFT matrix."""
        return self.matrix().T.conj()

    def apply(self, state: np.ndarray) -> np.ndarray:
        """
        Apply QFT to state.

        Args:
            state: Input state vector

        Returns:
            Transformed state
        """
        return self.matrix() @ state

    def apply_inverse(self, state: np.ndarray) -> np.ndarray:
        """Apply inverse QFT."""
        return self.inverse_matrix() @ state

    def circuit(self) -> QuantumCircuit:
        """
        Build QFT circuit.

        Returns:
            QuantumCircuit implementing QFT
        """
        n = self.n_qubits
        qc = QuantumCircuit(n)

        for i in range(n):
            qc.h(i)
            for j in range(i + 1, n):
                angle = np.pi / 2**(j - i)
                # Controlled phase (approximated)
                qc.rz(j, angle / 2)
                qc.cnot(i, j)
                qc.rz(j, -angle / 2)
                qc.cnot(i, j)

        # Swap qubits
        for i in range(n // 2):
            # SWAP via 3 CNOTs
            qc.cnot(i, n - 1 - i)
            qc.cnot(n - 1 - i, i)
            qc.cnot(i, n - 1 - i)

        return qc


class PhaseEstimation:
    """
    Quantum Phase Estimation algorithm.

    Estimates eigenvalue e^(2πiφ) of unitary U given eigenstate |u⟩.

    Args:
        n_precision: Number of precision qubits
        unitary: Unitary operator matrix
    """

    def __init__(self, n_precision: int, unitary: np.ndarray):
        self.n_precision = n_precision
        self.U = unitary
        self.dim = unitary.shape[0]
        self.n_target = int(np.log2(self.dim))

    def run(self, eigenstate: np.ndarray) -> float:
        """
        Run phase estimation.

        Args:
            eigenstate: Approximate eigenstate of U

        Returns:
            Estimated phase φ ∈ [0, 1)
        """
        n = self.n_precision
        N = 2**n

        # Initialize state: |0⟩^n ⊗ |u⟩
        dim_total = N * self.dim
        state = np.zeros(dim_total, dtype=complex)

        # |0...0⟩ ⊗ |u⟩
        for i, amp in enumerate(eigenstate):
            state[i] = amp

        # Apply H to precision qubits
        for i in range(n):
            # Apply H ⊗ I_rest
            state_reshaped = state.reshape(2**i, 2, -1)
            H = HadamardGate.matrix
            state_reshaped = np.tensordot(H, state_reshaped, axes=([1], [1]))
            state = state_reshaped.transpose(1, 0, 2).reshape(-1)

        # Apply controlled-U^(2^k)
        for k in range(n):
            U_pow = np.linalg.matrix_power(self.U, 2**k)

            # Controlled-U^(2^k) on qubit (n-1-k)
            control = n - 1 - k
            state_reshaped = state.reshape(2**control, 2, -1, self.dim)

            # Apply U to target when control is 1
            state_reshaped[:, 1, :, :] = np.tensordot(
                state_reshaped[:, 1, :, :], U_pow, axes=([2], [0])
            )
            state = state_reshaped.reshape(-1)

        # Apply inverse QFT to precision qubits
        QFT_inv = QuantumFourierTransform(n).inverse_matrix()

        state_reshaped = state.reshape(N, self.dim)
        state_reshaped = QFT_inv @ state_reshaped
        state = state_reshaped.reshape(-1)

        # Measure precision qubits
        probs = np.sum(np.abs(state.reshape(N, self.dim))**2, axis=1)
        outcome = np.random.choice(N, p=probs / probs.sum())

        # Convert to phase
        return outcome / N


class VQE:
    """
    Variational Quantum Eigensolver.

    Finds ground state energy of Hamiltonian using variational approach.

    Args:
        hamiltonian: Hamiltonian matrix
        n_qubits: Number of qubits
        ansatz: Variational ansatz type
    """

    def __init__(self, hamiltonian: np.ndarray, n_qubits: int,
                 ansatz: str = 'hardware_efficient'):
        self.H = hamiltonian
        self.n_qubits = n_qubits
        self.ansatz = ansatz
        self.dim = 2**n_qubits

    def ansatz_circuit(self, params: np.ndarray) -> np.ndarray:
        """
        Build parameterized ansatz state.

        Args:
            params: Variational parameters

        Returns:
            State vector
        """
        state = np.zeros(self.dim, dtype=complex)
        state[0] = 1

        n = self.n_qubits
        n_layers = len(params) // (3 * n)

        param_idx = 0

        for layer in range(n_layers):
            # Single-qubit rotations
            for q in range(n):
                Rx = PauliGates.rotation_x(params[param_idx])
                Ry = PauliGates.rotation_y(params[param_idx + 1])
                Rz = PauliGates.rotation_z(params[param_idx + 2])
                param_idx += 3

                # Apply to state
                full_gate = np.eye(1)
                for i in range(n):
                    if i == q:
                        full_gate = np.kron(full_gate, Rz @ Ry @ Rx)
                    else:
                        full_gate = np.kron(full_gate, np.eye(2))
                state = full_gate @ state

            # Entangling layer (linear CNOT chain)
            for q in range(n - 1):
                # CNOT
                new_state = np.zeros_like(state)
                for i in range(self.dim):
                    control_bit = (i >> (n - 1 - q)) & 1
                    if control_bit:
                        j = i ^ (1 << (n - 2 - q))
                        new_state[j] = state[i]
                    else:
                        new_state[i] = state[i]
                state = new_state

        return state

    def energy(self, params: np.ndarray) -> float:
        """
        Calculate expectation value ⟨ψ(θ)|H|ψ(θ)⟩.

        Args:
            params: Variational parameters

        Returns:
            Energy expectation value
        """
        state = self.ansatz_circuit(params)
        return np.real(state.conj() @ self.H @ state)

    def optimize(self, n_layers: int = 1, max_iter: int = 100,
                 method: str = 'COBYLA') -> Tuple[float, np.ndarray]:
        """
        Optimize variational parameters.

        Args:
            n_layers: Number of ansatz layers
            max_iter: Maximum iterations
            method: Optimization method

        Returns:
            (minimum_energy, optimal_params)
        """
        from scipy.optimize import minimize

        n_params = 3 * self.n_qubits * n_layers
        x0 = np.random.randn(n_params) * 0.1

        result = minimize(self.energy, x0, method=method,
                         options={'maxiter': max_iter})

        return result.fun, result.x

    def exact_ground_state_energy(self) -> float:
        """Get exact ground state energy for comparison."""
        eigenvalues = np.linalg.eigvalsh(self.H)
        return eigenvalues[0]


# =============================================================================
# Error Correction
# =============================================================================

class BitFlipCode:
    """
    3-qubit bit flip code.

    Encodes |0⟩ → |000⟩, |1⟩ → |111⟩

    Corrects single bit flip errors.
    """

    @classmethod
    def encode(cls, state: np.ndarray) -> np.ndarray:
        """
        Encode single qubit into 3-qubit code.

        Args:
            state: Single qubit state [α, β]

        Returns:
            Encoded 3-qubit state
        """
        alpha, beta = state[0], state[1]
        encoded = np.zeros(8, dtype=complex)
        encoded[0] = alpha  # |000⟩
        encoded[7] = beta   # |111⟩
        return encoded

    @classmethod
    def decode(cls, state: np.ndarray) -> np.ndarray:
        """
        Decode 3-qubit state after error correction.

        Args:
            state: 3-qubit state

        Returns:
            Decoded single qubit state
        """
        # Project onto code space
        alpha = state[0]  # |000⟩ coefficient
        beta = state[7]   # |111⟩ coefficient

        return np.array([alpha, beta], dtype=complex)

    @classmethod
    def syndrome_measurement(cls, state: np.ndarray) -> Tuple[int, int]:
        """
        Measure error syndrome.

        Syndromes:
        (0,0): no error
        (1,0): error on qubit 0
        (0,1): error on qubit 2
        (1,1): error on qubit 1

        Args:
            state: Potentially corrupted state

        Returns:
            (syndrome_01, syndrome_12) parity measurements
        """
        # Z₀Z₁ parity
        parity_01 = 0
        for i in range(8):
            bit0 = (i >> 2) & 1
            bit1 = (i >> 1) & 1
            if (bit0 ^ bit1) == 1:
                parity_01 += np.abs(state[i])**2

        # Z₁Z₂ parity
        parity_12 = 0
        for i in range(8):
            bit1 = (i >> 1) & 1
            bit2 = i & 1
            if (bit1 ^ bit2) == 1:
                parity_12 += np.abs(state[i])**2

        s01 = 1 if parity_01 > 0.5 else 0
        s12 = 1 if parity_12 > 0.5 else 0

        return (s01, s12)

    @classmethod
    def correct(cls, state: np.ndarray) -> np.ndarray:
        """
        Perform error correction.

        Args:
            state: Potentially corrupted state

        Returns:
            Corrected state
        """
        s01, s12 = cls.syndrome_measurement(state)

        if s01 == 0 and s12 == 0:
            return state  # No error
        elif s01 == 1 and s12 == 0:
            # Error on qubit 0 - apply X₀
            X0 = np.kron(np.kron(PauliGates.X, np.eye(2)), np.eye(2))
            return X0 @ state
        elif s01 == 1 and s12 == 1:
            # Error on qubit 1 - apply X₁
            X1 = np.kron(np.kron(np.eye(2), PauliGates.X), np.eye(2))
            return X1 @ state
        else:  # s01 == 0 and s12 == 1
            # Error on qubit 2 - apply X₂
            X2 = np.kron(np.kron(np.eye(2), np.eye(2)), PauliGates.X)
            return X2 @ state


class PhaseFlipCode:
    """
    3-qubit phase flip code.

    Encodes |0⟩ → |+++⟩, |1⟩ → |---⟩

    Corrects single phase flip (Z) errors.
    """

    @classmethod
    def encode(cls, state: np.ndarray) -> np.ndarray:
        """
        Encode single qubit.

        Args:
            state: Single qubit state

        Returns:
            Encoded 3-qubit state
        """
        alpha, beta = state[0], state[1]

        # |+⟩ = (|0⟩ + |1⟩)/√2
        plus = np.array([1, 1]) / np.sqrt(2)
        minus = np.array([1, -1]) / np.sqrt(2)

        plus3 = np.kron(np.kron(plus, plus), plus)
        minus3 = np.kron(np.kron(minus, minus), minus)

        return alpha * plus3 + beta * minus3

    @classmethod
    def decode(cls, state: np.ndarray) -> np.ndarray:
        """
        Decode after error correction.

        Args:
            state: 3-qubit state

        Returns:
            Decoded single qubit
        """
        # Apply H^⊗3 to convert to bit flip basis
        H3 = HadamardGate.tensor_n(3)
        state_z = H3 @ state

        # Now in bit flip code space
        return BitFlipCode.decode(state_z)

    @classmethod
    def correct(cls, state: np.ndarray) -> np.ndarray:
        """
        Perform error correction.

        Works by converting to bit flip basis, correcting, converting back.
        """
        H3 = HadamardGate.tensor_n(3)

        # Convert to bit flip basis
        state_z = H3 @ state

        # Correct bit flip (which was phase flip)
        state_corrected = BitFlipCode.correct(state_z)

        # Convert back
        return H3 @ state_corrected


class ShorCode:
    """
    9-qubit Shor code.

    Concatenation of bit flip and phase flip codes.
    Corrects arbitrary single-qubit errors.

    Encodes |0⟩ → (|000⟩ + |111⟩)^⊗3 / 2√2
    """

    @classmethod
    def encode(cls, state: np.ndarray) -> np.ndarray:
        """
        Encode single qubit into 9-qubit code.

        Args:
            state: Single qubit state

        Returns:
            Encoded 9-qubit state
        """
        alpha, beta = state[0], state[1]

        # |0_L⟩ = (|000⟩ + |111⟩)⊗3 / 2√2
        block_0 = (np.array([1, 0, 0, 0, 0, 0, 0, 1]) / np.sqrt(2))  # |000⟩ + |111⟩
        logical_0 = np.kron(np.kron(block_0, block_0), block_0)

        # |1_L⟩ = (|000⟩ - |111⟩)⊗3 / 2√2
        block_1 = (np.array([1, 0, 0, 0, 0, 0, 0, -1]) / np.sqrt(2))  # |000⟩ - |111⟩
        logical_1 = np.kron(np.kron(block_1, block_1), block_1)

        return alpha * logical_0 + beta * logical_1

    @classmethod
    def decode(cls, state: np.ndarray) -> np.ndarray:
        """
        Decode 9-qubit state.

        Returns:
            Single qubit state
        """
        # Project onto logical subspace
        logical_0 = cls.encode(np.array([1, 0]))
        logical_1 = cls.encode(np.array([0, 1]))

        alpha = np.dot(logical_0.conj(), state)
        beta = np.dot(logical_1.conj(), state)

        norm = np.sqrt(np.abs(alpha)**2 + np.abs(beta)**2)
        return np.array([alpha, beta]) / norm


class SteaneCode:
    """
    7-qubit Steane code.

    CSS code based on classical [7,4,3] Hamming code.
    Corrects arbitrary single-qubit errors.
    """

    # Parity check matrices for [7,4,3] Hamming code
    H_MATRIX = np.array([
        [1, 0, 0, 1, 1, 0, 1],
        [0, 1, 0, 1, 0, 1, 1],
        [0, 0, 1, 0, 1, 1, 1]
    ])

    @classmethod
    def encode(cls, state: np.ndarray) -> np.ndarray:
        """
        Encode single qubit.

        Args:
            state: Single qubit state

        Returns:
            Encoded 7-qubit state
        """
        alpha, beta = state[0], state[1]

        # Codewords of [7,4,3] code
        codewords = []
        for i in range(16):  # 4 information bits
            info = [(i >> j) & 1 for j in range(4)]
            # Generate codeword (simplified)
            codeword = [0] * 7
            # This is a placeholder - real encoding is more complex
            codeword[0] = info[0]
            codeword[1] = info[1]
            codeword[2] = info[2]
            codeword[3] = info[3]
            codeword[4] = (info[0] + info[1] + info[3]) % 2
            codeword[5] = (info[0] + info[2] + info[3]) % 2
            codeword[6] = (info[1] + info[2] + info[3]) % 2
            codewords.append(codeword)

        # |0_L⟩ = superposition of even-weight codewords
        # |1_L⟩ = superposition of odd-weight codewords
        logical_0 = np.zeros(128, dtype=complex)
        logical_1 = np.zeros(128, dtype=complex)

        for cw in codewords:
            idx = sum(cw[i] * 2**(6-i) for i in range(7))
            weight = sum(cw)
            if weight % 2 == 0:
                logical_0[idx] = 1
            else:
                logical_1[idx] = 1

        logical_0 /= np.linalg.norm(logical_0)
        logical_1 /= np.linalg.norm(logical_1)

        return alpha * logical_0 + beta * logical_1


class SurfaceCode:
    """
    Surface code basics (toric code variant).

    Topological error correcting code with high threshold.

    Args:
        distance: Code distance d
    """

    def __init__(self, distance: int = 3):
        if distance < 3:
            raise ValueError("Distance must be at least 3")

        self.d = distance
        self.n_data = distance**2
        self.n_ancilla = (distance - 1)**2 + (distance - 1)**2

    def logical_operators(self) -> Dict[str, np.ndarray]:
        """
        Get logical X and Z operators.

        Returns:
            Dictionary with 'X' and 'Z' logical operators
        """
        d = self.d

        # Logical X: chain of X along one direction
        X_chain = np.zeros(self.n_data, dtype=int)
        for i in range(d):
            X_chain[i * d] = 1  # First column

        # Logical Z: chain of Z along perpendicular direction
        Z_chain = np.zeros(self.n_data, dtype=int)
        for i in range(d):
            Z_chain[i] = 1  # First row

        return {'X': X_chain, 'Z': Z_chain}

    def stabilizer_generators(self) -> Dict[str, List[np.ndarray]]:
        """
        Get X and Z stabilizer generators.

        Returns:
            Dictionary with 'X' and 'Z' stabilizer lists
        """
        d = self.d
        X_stabilizers = []
        Z_stabilizers = []

        # X stabilizers (plaquettes)
        for i in range(d - 1):
            for j in range(d - 1):
                stab = np.zeros(self.n_data, dtype=int)
                stab[i * d + j] = 1
                stab[i * d + j + 1] = 1
                stab[(i + 1) * d + j] = 1
                stab[(i + 1) * d + j + 1] = 1
                X_stabilizers.append(stab)

        # Z stabilizers (vertices) - similar construction
        for i in range(d - 1):
            for j in range(d - 1):
                stab = np.zeros(self.n_data, dtype=int)
                stab[i * d + j] = 1
                stab[i * d + j + 1] = 1
                stab[(i + 1) * d + j] = 1
                stab[(i + 1) * d + j + 1] = 1
                Z_stabilizers.append(stab)

        return {'X': X_stabilizers, 'Z': Z_stabilizers}

    def threshold(self) -> float:
        """
        Error threshold for surface code.

        Returns:
            Approximate threshold probability
        """
        return 0.01  # ~1% threshold


# =============================================================================
# Entanglement Measures
# =============================================================================

class VonNeumannEntropy:
    """
    Von Neumann entropy S(ρ) = -Tr(ρ log ρ).

    For pure bipartite state, measures entanglement.
    """

    @staticmethod
    def compute(rho: np.ndarray) -> float:
        """
        Calculate von Neumann entropy.

        Args:
            rho: Density matrix

        Returns:
            Entropy (in bits, using log2)
        """
        eigenvalues = np.linalg.eigvalsh(rho)
        # Filter out zero eigenvalues
        eigenvalues = eigenvalues[eigenvalues > 1e-15]

        return -np.sum(eigenvalues * np.log2(eigenvalues))

    @staticmethod
    def entanglement_entropy(state: np.ndarray, subsystem_dims: Tuple[int, int]) -> float:
        """
        Entanglement entropy of bipartite pure state.

        S_A = S_B = -Tr(ρ_A log ρ_A)

        Args:
            state: Pure state vector
            subsystem_dims: (dim_A, dim_B)

        Returns:
            Entanglement entropy
        """
        d_A, d_B = subsystem_dims

        # Reshape to matrix and compute reduced density matrix
        psi = state.reshape(d_A, d_B)
        rho_A = psi @ psi.T.conj()

        return VonNeumannEntropy.compute(rho_A)


class Concurrence:
    """
    Concurrence for two-qubit entanglement.

    C(ρ) = max(0, λ₁ - λ₂ - λ₃ - λ₄)

    where λᵢ are eigenvalues of √(√ρ ρ̃ √ρ) in decreasing order.
    """

    @staticmethod
    def compute(rho: np.ndarray) -> float:
        """
        Calculate concurrence for two-qubit state.

        Args:
            rho: 4×4 density matrix

        Returns:
            Concurrence C ∈ [0, 1]
        """
        if rho.shape != (4, 4):
            raise ValueError("Concurrence requires 4×4 density matrix")

        # σ_y ⊗ σ_y
        Y = PauliGates.Y
        YY = np.kron(Y, Y)

        # Spin-flipped state
        rho_tilde = YY @ rho.conj() @ YY

        # R = √(√ρ ρ̃ √ρ)
        sqrt_rho = linalg.sqrtm(rho)
        R = linalg.sqrtm(sqrt_rho @ rho_tilde @ sqrt_rho)

        # Eigenvalues in decreasing order
        eigenvalues = np.sort(np.real(np.linalg.eigvals(R)))[::-1]

        return max(0, eigenvalues[0] - eigenvalues[1] - eigenvalues[2] - eigenvalues[3])

    @staticmethod
    def from_pure_state(state: np.ndarray) -> float:
        """
        Concurrence for pure two-qubit state.

        C = 2|αδ - βγ| for |ψ⟩ = α|00⟩ + β|01⟩ + γ|10⟩ + δ|11⟩
        """
        if len(state) != 4:
            raise ValueError("State must be 4-dimensional")

        return 2 * np.abs(state[0] * state[3] - state[1] * state[2])


class Negativity:
    """
    Negativity as entanglement measure.

    N(ρ) = (||ρ^(T_B)||₁ - 1) / 2

    where T_B is partial transpose over subsystem B.
    """

    @staticmethod
    def partial_transpose(rho: np.ndarray, dims: Tuple[int, int],
                          which: int = 1) -> np.ndarray:
        """
        Compute partial transpose.

        Args:
            rho: Density matrix
            dims: (dim_A, dim_B)
            which: 0 for A, 1 for B

        Returns:
            Partially transposed matrix
        """
        d_A, d_B = dims

        rho_reshaped = rho.reshape(d_A, d_B, d_A, d_B)

        if which == 1:
            rho_pt = np.transpose(rho_reshaped, (0, 3, 2, 1))
        else:
            rho_pt = np.transpose(rho_reshaped, (2, 1, 0, 3))

        return rho_pt.reshape(d_A * d_B, d_A * d_B)

    @staticmethod
    def compute(rho: np.ndarray, dims: Tuple[int, int] = (2, 2)) -> float:
        """
        Calculate negativity.

        Args:
            rho: Density matrix
            dims: Subsystem dimensions

        Returns:
            Negativity N ≥ 0
        """
        rho_pt = Negativity.partial_transpose(rho, dims)

        eigenvalues = np.linalg.eigvalsh(rho_pt)
        negative_eigenvalues = eigenvalues[eigenvalues < 0]

        return -np.sum(negative_eigenvalues)

    @staticmethod
    def logarithmic_negativity(rho: np.ndarray,
                               dims: Tuple[int, int] = (2, 2)) -> float:
        """
        Logarithmic negativity E_N = log₂(||ρ^(T_B)||₁).

        Args:
            rho: Density matrix
            dims: Subsystem dimensions

        Returns:
            Logarithmic negativity
        """
        rho_pt = Negativity.partial_transpose(rho, dims)
        trace_norm = np.sum(np.abs(np.linalg.eigvalsh(rho_pt)))

        return np.log2(trace_norm)


class MutualInformation:
    """
    Quantum mutual information.

    I(A:B) = S(ρ_A) + S(ρ_B) - S(ρ_AB)

    Measures total correlations (classical + quantum).
    """

    @staticmethod
    def compute(rho: np.ndarray, dims: Tuple[int, int]) -> float:
        """
        Calculate quantum mutual information.

        Args:
            rho: Joint density matrix ρ_AB
            dims: (dim_A, dim_B)

        Returns:
            Mutual information I(A:B)
        """
        d_A, d_B = dims

        # Reduced density matrices
        rho_reshaped = rho.reshape(d_A, d_B, d_A, d_B)

        rho_A = np.trace(rho_reshaped, axis1=1, axis2=3)
        rho_B = np.trace(rho_reshaped, axis1=0, axis2=2)

        # Entropies
        S_A = VonNeumannEntropy.compute(rho_A)
        S_B = VonNeumannEntropy.compute(rho_B)
        S_AB = VonNeumannEntropy.compute(rho)

        return S_A + S_B - S_AB

    @staticmethod
    def conditional_entropy(rho: np.ndarray, dims: Tuple[int, int]) -> float:
        """
        Conditional entropy S(A|B) = S(ρ_AB) - S(ρ_B).

        Can be negative for entangled states.

        Args:
            rho: Joint density matrix
            dims: Subsystem dimensions

        Returns:
            Conditional entropy
        """
        d_A, d_B = dims

        rho_reshaped = rho.reshape(d_A, d_B, d_A, d_B)
        rho_B = np.trace(rho_reshaped, axis1=0, axis2=2)

        S_AB = VonNeumannEntropy.compute(rho)
        S_B = VonNeumannEntropy.compute(rho_B)

        return S_AB - S_B


# Module exports
__all__ = [
    # Qubits & Gates
    'Qubit', 'PauliGates', 'HadamardGate', 'PhaseGate',
    'CNOTGate', 'ToffoliGate', 'UniversalGateSet',
    # Circuits
    'QuantumCircuit', 'CircuitSimulator', 'MeasurementBackend', 'DensityMatrixSimulator',
    # Algorithms
    'GroverSearch', 'DeutschJozsa', 'QuantumFourierTransform',
    'PhaseEstimation', 'VQE',
    # Error Correction
    'BitFlipCode', 'PhaseFlipCode', 'ShorCode', 'SteaneCode', 'SurfaceCode',
    # Entanglement
    'VonNeumannEntropy', 'Concurrence', 'Negativity', 'MutualInformation',
]
