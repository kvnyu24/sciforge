"""
Experiment 181: Grover's Search Algorithm

Demonstrates Grover's quantum search algorithm for finding marked items in an
unsorted database with quadratic speedup over classical search.

Physics/Algorithm:
    Given N = 2^n items with M marked solutions:
    - Classical: O(N) queries needed
    - Quantum: O(sqrt(N/M)) queries (quadratic speedup!)

    Algorithm:
    1. Initialize: |s> = H^n|0> = (1/sqrt(N)) * sum_x |x>

    2. Repeat O(sqrt(N)) times:
       a. Oracle: O|x> = (-1)^f(x)|x>  (flip sign of marked states)
       b. Diffusion: D = 2|s><s| - I   (reflect about mean)

    3. Measure to find marked item with high probability

    Geometric interpretation:
    - State evolves in 2D plane spanned by |marked> and |unmarked>
    - Each iteration rotates by angle 2*arcsin(sqrt(M/N))
    - Stop when close to |marked> state
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def initialize_superposition(n_qubits):
    """
    Create uniform superposition |s> = H^n|0>.

    |s> = (1/sqrt(N)) * sum_{x=0}^{N-1} |x>

    Args:
        n_qubits: Number of qubits

    Returns:
        State vector
    """
    N = 2**n_qubits
    return np.ones(N, dtype=complex) / np.sqrt(N)


def oracle(state, marked_items):
    """
    Apply oracle O that flips the sign of marked items.

    O|x> = -|x> if x is marked, |x> otherwise

    Args:
        state: State vector
        marked_items: List of marked item indices

    Returns:
        New state vector
    """
    new_state = state.copy()
    for item in marked_items:
        new_state[item] = -new_state[item]
    return new_state


def diffusion(state):
    """
    Apply Grover diffusion operator D = 2|s><s| - I.

    This reflects the state about the mean amplitude.
    Amplifies amplitude of marked states.

    Args:
        state: State vector

    Returns:
        New state vector
    """
    N = len(state)
    mean_amplitude = np.mean(state)
    return 2 * mean_amplitude - state


def grover_iteration(state, marked_items):
    """
    Single Grover iteration: Oracle followed by Diffusion.

    G = D @ O

    Args:
        state: State vector
        marked_items: Marked item indices

    Returns:
        New state vector
    """
    state = oracle(state, marked_items)
    state = diffusion(state)
    return state


def optimal_iterations(N, M):
    """
    Calculate optimal number of Grover iterations.

    k_opt = round(pi/(4*arcsin(sqrt(M/N))) - 1/2)
         ≈ pi/4 * sqrt(N/M)

    Args:
        N: Total number of items
        M: Number of marked items

    Returns:
        Optimal iteration count
    """
    theta = np.arcsin(np.sqrt(M / N))
    return int(np.round(np.pi / (4 * theta) - 0.5))


def success_probability(n_iterations, N, M):
    """
    Calculate success probability after k iterations.

    P_success = sin^2((2k+1) * theta)

    where theta = arcsin(sqrt(M/N))

    Args:
        n_iterations: Number of Grover iterations
        N: Total items
        M: Marked items

    Returns:
        Probability of finding marked item
    """
    theta = np.arcsin(np.sqrt(M / N))
    return np.sin((2 * n_iterations + 1) * theta)**2


def run_grover(n_qubits, marked_items, n_iterations=None, track_history=False):
    """
    Run Grover's algorithm.

    Args:
        n_qubits: Number of qubits
        marked_items: List of marked items
        n_iterations: Number of iterations (default: optimal)
        track_history: Whether to track state evolution

    Returns:
        Final state (and history if requested)
    """
    N = 2**n_qubits
    M = len(marked_items)

    if n_iterations is None:
        n_iterations = optimal_iterations(N, M)

    state = initialize_superposition(n_qubits)

    if track_history:
        history = [state.copy()]

    for _ in range(n_iterations):
        state = grover_iteration(state, marked_items)
        if track_history:
            history.append(state.copy())

    if track_history:
        return state, history
    return state


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ===== Plot 1: Amplitude evolution during Grover search =====
    ax1 = axes[0, 0]

    n_qubits = 4
    N = 2**n_qubits
    marked_items = [7]  # Search for item 7

    # Run with tracking
    n_iter = optimal_iterations(N, len(marked_items)) + 3
    state, history = run_grover(n_qubits, marked_items, n_iter, track_history=True)

    # Plot amplitude evolution
    iterations = range(len(history))

    # Marked item amplitude
    marked_amp = [np.real(h[marked_items[0]]) for h in history]
    # Unmarked item amplitude (average)
    unmarked_amp = [np.real(np.mean([h[i] for i in range(N) if i not in marked_items])) for h in history]

    ax1.plot(iterations, marked_amp, 'b-', lw=2, marker='o', label='Marked item amplitude')
    ax1.plot(iterations, unmarked_amp, 'r-', lw=2, marker='s', label='Unmarked item amplitude')

    ax1.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax1.axhline(0.0, color='gray', linestyle='-', alpha=0.3)

    k_opt = optimal_iterations(N, 1)
    ax1.axvline(k_opt, color='green', linestyle=':', lw=2, label=f'Optimal k = {k_opt}')

    ax1.set_xlabel('Grover Iterations')
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'Amplitude Evolution in Grover Search\n(N={N}, marking item {marked_items[0]})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ===== Plot 2: Success probability vs iterations =====
    ax2 = axes[0, 1]

    iterations = np.arange(0, 15)
    P_success = [success_probability(k, N, 1) for k in iterations]

    ax2.plot(iterations, P_success, 'b-', lw=2, marker='o')
    ax2.axhline(1.0, color='green', linestyle='--', alpha=0.5, label='P=1')
    ax2.axvline(k_opt, color='red', linestyle=':', lw=2, label=f'Optimal k = {k_opt}')

    ax2.fill_between(iterations, 0, P_success, alpha=0.3)

    ax2.set_xlabel('Number of Grover Iterations')
    ax2.set_ylabel('Success Probability')
    ax2.set_title(f'Success Probability vs Iterations\n(N={N}, 1 marked item)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max(iterations))
    ax2.set_ylim(0, 1.1)

    # ===== Plot 3: Scaling with database size =====
    ax3 = axes[1, 0]

    n_qubits_range = range(2, 12)
    classical_queries = []
    quantum_queries = []

    for n in n_qubits_range:
        N = 2**n
        classical_queries.append(N / 2)  # Expected classical queries
        quantum_queries.append(optimal_iterations(N, 1))

    ax3.semilogy(n_qubits_range, classical_queries, 'r-', lw=2, marker='s', label='Classical: O(N)')
    ax3.semilogy(n_qubits_range, quantum_queries, 'b-', lw=2, marker='o', label='Quantum: O(sqrt(N))')

    # Reference lines
    n_ref = np.array(list(n_qubits_range))
    ax3.semilogy(n_ref, 2**n_ref / 2, 'r:', alpha=0.5)
    ax3.semilogy(n_ref, np.pi/4 * np.sqrt(2**n_ref), 'b:', alpha=0.5)

    ax3.set_xlabel('Number of Qubits (n)')
    ax3.set_ylabel('Number of Queries')
    ax3.set_title('Query Complexity: Quadratic Speedup\n(Database size N = 2^n)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # ===== Plot 4: Final probability distribution =====
    ax4 = axes[1, 1]

    # Search for multiple items
    n_qubits = 5
    N = 2**n_qubits
    marked_items = [7, 15, 23]

    k_opt = optimal_iterations(N, len(marked_items))
    final_state = run_grover(n_qubits, marked_items, k_opt)

    probabilities = np.abs(final_state)**2

    colors = ['red' if i in marked_items else 'blue' for i in range(N)]
    ax4.bar(range(N), probabilities, color=colors, alpha=0.7)

    ax4.set_xlabel('Item Index')
    ax4.set_ylabel('Probability')
    ax4.set_title(f'Final Probability Distribution (N={N}, k={k_opt})\n(Red = marked items: {marked_items})')
    ax4.grid(True, alpha=0.3, axis='y')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', alpha=0.7, label='Marked'),
                       Patch(facecolor='blue', alpha=0.7, label='Unmarked')]
    ax4.legend(handles=legend_elements)

    plt.suptitle("Grover's Quantum Search Algorithm\n"
                 r'Quadratic Speedup: O(N) $\rightarrow$ O($\sqrt{N}$)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'grover_search.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'grover_search.png')}")

    # Print results
    print("\n=== Grover's Search Algorithm Results ===")

    print(f"\nExample: N = {N}, searching for items {marked_items}")
    print(f"Optimal iterations: {k_opt}")

    total_prob_marked = sum(probabilities[i] for i in marked_items)
    print(f"Total probability of marked items: {total_prob_marked:.4f}")

    print(f"\nSpeedup comparison:")
    for n in [4, 8, 10, 16]:
        N = 2**n
        classical = N / 2
        quantum = np.pi/4 * np.sqrt(N)
        speedup = classical / quantum
        print(f"  n={n:2d} (N={N:6d}): Classical~{classical:.0f}, Quantum~{quantum:.1f}, Speedup~{speedup:.1f}x")

    # Simulate search
    print(f"\nSimulated searches (100 trials):")
    n_trials = 100
    successes = 0
    N_search = 2**n_qubits
    for _ in range(n_trials):
        state = run_grover(n_qubits, marked_items)
        probs = np.abs(state)**2
        measured = np.random.choice(N_search, p=probs)
        if measured in marked_items:
            successes += 1
    print(f"  Success rate: {successes}/{n_trials} = {successes/n_trials:.2%}")


if __name__ == "__main__":
    main()
