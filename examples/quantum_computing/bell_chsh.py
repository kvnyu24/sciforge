"""
Experiment 179: Bell Inequality (CHSH) Violation

Demonstrates the CHSH (Clauser-Horne-Shimony-Holt) form of Bell's inequality
and its violation by entangled quantum states.

Physics:
    The CHSH inequality constrains local hidden variable theories:

    |S| = |E(a,b) - E(a,b') + E(a',b) + E(a',b')| <= 2

    where E(a,b) is the correlation function for measurement settings a, b.

    For quantum mechanics with maximally entangled Bell state:
    |Phi+> = (|00> + |11>) / sqrt(2)

    Maximum violation: S = 2*sqrt(2) ≈ 2.828 (Tsirelson bound)
    Achieved with settings: a=0, a'=pi/2, b=pi/4, b'=3*pi/4

    This proves quantum mechanics cannot be explained by local hidden variables.

    Historical significance:
    - Bell (1964): Original inequality
    - CHSH (1969): Experimentally testable form
    - Aspect et al. (1982): First convincing experimental test
    - Nobel Prize 2022: Aspect, Clauser, Zeilinger
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.eye(2, dtype=complex)


def create_bell_state(which='phi_plus'):
    """
    Create Bell state.

    |Phi+> = (|00> + |11>) / sqrt(2)
    |Phi-> = (|00> - |11>) / sqrt(2)
    |Psi+> = (|01> + |10>) / sqrt(2)
    |Psi-> = (|01> - |10>) / sqrt(2)

    Args:
        which: 'phi_plus', 'phi_minus', 'psi_plus', 'psi_minus'

    Returns:
        4-element state vector
    """
    if which == 'phi_plus':
        return np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    elif which == 'phi_minus':
        return np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2)
    elif which == 'psi_plus':
        return np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)
    elif which == 'psi_minus':
        return np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
    else:
        raise ValueError(f"Unknown Bell state: {which}")


def measurement_operator(theta):
    """
    Measurement operator in x-z plane.

    M(theta) = cos(theta) * sigma_z + sin(theta) * sigma_x

    This measures spin along direction (sin(theta), 0, cos(theta)).

    Args:
        theta: Angle from z-axis

    Returns:
        2x2 measurement operator
    """
    return np.cos(theta) * sigma_z + np.sin(theta) * sigma_x


def correlation(state, theta_a, theta_b):
    """
    Calculate quantum correlation function E(a,b).

    E(a,b) = <psi| M_a tensor M_b |psi>

    Args:
        state: Two-qubit state
        theta_a: Alice's measurement angle
        theta_b: Bob's measurement angle

    Returns:
        Correlation value in [-1, 1]
    """
    M_a = measurement_operator(theta_a)
    M_b = measurement_operator(theta_b)

    # Joint operator
    M_ab = np.kron(M_a, M_b)

    # Expectation value
    return np.real(np.conj(state) @ M_ab @ state)


def chsh_parameter(state, a, a_prime, b, b_prime):
    """
    Calculate CHSH parameter S.

    S = E(a,b) - E(a,b') + E(a',b) + E(a',b')

    Local hidden variable theories: |S| <= 2
    Quantum mechanics: |S| <= 2*sqrt(2) ≈ 2.828

    Args:
        state: Two-qubit state
        a, a_prime: Alice's measurement angles
        b, b_prime: Bob's measurement angles

    Returns:
        CHSH parameter S
    """
    E_ab = correlation(state, a, b)
    E_ab_prime = correlation(state, a, b_prime)
    E_a_prime_b = correlation(state, a_prime, b)
    E_a_prime_b_prime = correlation(state, a_prime, b_prime)

    return E_ab - E_ab_prime + E_a_prime_b + E_a_prime_b_prime


def simulate_experiment(state, theta_a, theta_b, n_shots=1000, seed=None):
    """
    Simulate Bell experiment with finite statistics.

    Args:
        state: Two-qubit state
        theta_a, theta_b: Measurement angles
        n_shots: Number of measurement shots
        seed: Random seed

    Returns:
        Empirical correlation
    """
    if seed is not None:
        np.random.seed(seed)

    # Measurement projectors
    # For spin along theta: |+> = cos(theta/2)|0> + sin(theta/2)|1>
    #                       |-> = -sin(theta/2)|0> + cos(theta/2)|1>

    def get_prob(state, theta_a, theta_b, outcome_a, outcome_b):
        """Probability of specific outcome pair."""
        # Build projector
        if outcome_a == +1:
            P_a = np.array([[np.cos(theta_a/2)**2, np.cos(theta_a/2)*np.sin(theta_a/2)],
                           [np.cos(theta_a/2)*np.sin(theta_a/2), np.sin(theta_a/2)**2]], dtype=complex)
        else:
            P_a = np.array([[np.sin(theta_a/2)**2, -np.cos(theta_a/2)*np.sin(theta_a/2)],
                           [-np.cos(theta_a/2)*np.sin(theta_a/2), np.cos(theta_a/2)**2]], dtype=complex)

        if outcome_b == +1:
            P_b = np.array([[np.cos(theta_b/2)**2, np.cos(theta_b/2)*np.sin(theta_b/2)],
                           [np.cos(theta_b/2)*np.sin(theta_b/2), np.sin(theta_b/2)**2]], dtype=complex)
        else:
            P_b = np.array([[np.sin(theta_b/2)**2, -np.cos(theta_b/2)*np.sin(theta_b/2)],
                           [-np.cos(theta_b/2)*np.sin(theta_b/2), np.cos(theta_b/2)**2]], dtype=complex)

        P_ab = np.kron(P_a, P_b)
        return np.real(np.conj(state) @ P_ab @ state)

    # Compute probabilities
    p_pp = get_prob(state, theta_a, theta_b, +1, +1)
    p_pm = get_prob(state, theta_a, theta_b, +1, -1)
    p_mp = get_prob(state, theta_a, theta_b, -1, +1)
    p_mm = get_prob(state, theta_a, theta_b, -1, -1)

    probs = np.array([p_pp, p_pm, p_mp, p_mm])
    probs = probs / np.sum(probs)  # Normalize

    # Sample
    outcomes_idx = np.random.choice(4, size=n_shots, p=probs)

    # Convert to correlations: (++): +1, (+-): -1, (-+): -1, (--): +1
    correlations = np.array([+1, -1, -1, +1])
    sampled_correlations = correlations[outcomes_idx]

    return np.mean(sampled_correlations)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Bell state
    phi_plus = create_bell_state('phi_plus')

    # ===== Plot 1: Correlation function E(a,b) =====
    ax1 = axes[0, 0]

    # Fix Alice's angle, vary Bob's
    theta_a = 0
    theta_b_range = np.linspace(0, 2*np.pi, 100)

    # Quantum correlation
    E_quantum = [correlation(phi_plus, theta_a, theta_b) for theta_b in theta_b_range]

    # Classical bound: |E| <= 1
    ax1.fill_between(theta_b_range / np.pi, -1, 1, alpha=0.2, color='green', label='Classical range')

    ax1.plot(theta_b_range / np.pi, E_quantum, 'b-', lw=2, label='Quantum: E(0, theta_b)')

    # Analytical: E(a, b) = -cos(b - a) for |Phi+>
    E_analytical = -np.cos(theta_b_range - theta_a)
    ax1.plot(theta_b_range / np.pi, E_analytical, 'r--', lw=2, alpha=0.7, label='Analytical: -cos(b-a)')

    ax1.set_xlabel('Bob\'s angle theta_b / pi')
    ax1.set_ylabel('Correlation E(a, b)')
    ax1.set_title('Quantum Correlation Function\n|Phi+> = (|00> + |11>)/sqrt(2)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 2)

    # ===== Plot 2: CHSH parameter vs settings =====
    ax2 = axes[0, 1]

    # Vary b while keeping optimal a, a', b' relationship
    # Optimal: a=0, a'=pi/2, b=pi/4, b'=-pi/4 gives maximum violation

    b_range = np.linspace(-np.pi/2, np.pi/2, 100)
    S_values = []

    for b in b_range:
        # Optimal settings given b
        a = 0
        a_prime = np.pi/2
        b_prime = b + np.pi/2  # Orthogonal to b

        S = chsh_parameter(phi_plus, a, a_prime, b, b_prime)
        S_values.append(abs(S))

    ax2.plot(b_range / np.pi, S_values, 'b-', lw=2, label='|S| (quantum)')
    ax2.axhline(2, color='green', linestyle='--', lw=2, label='Classical bound |S| <= 2')
    ax2.axhline(2*np.sqrt(2), color='red', linestyle=':', lw=2, label=f'Tsirelson bound |S| <= 2sqrt(2) = {2*np.sqrt(2):.3f}')

    # Mark optimal angle
    optimal_b = np.pi/4
    ax2.axvline(optimal_b / np.pi, color='gray', linestyle=':', alpha=0.5)
    ax2.annotate(f'Maximum at b = pi/4', xy=(0.25, 2.82), fontsize=10)

    ax2.fill_between(b_range / np.pi, 0, 2, alpha=0.2, color='green')
    ax2.fill_between(b_range / np.pi, 2, 2*np.sqrt(2), alpha=0.2, color='red')

    ax2.set_xlabel('Bob\'s angle b / pi')
    ax2.set_ylabel('CHSH parameter |S|')
    ax2.set_title('CHSH Inequality Violation\n(Red region: quantum advantage)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 3)

    # ===== Plot 3: Experimental simulation with finite statistics =====
    ax3 = axes[1, 0]

    # Optimal settings
    a, a_prime = 0, np.pi/2
    b, b_prime = np.pi/4, 3*np.pi/4

    n_shots_range = [10, 50, 100, 500, 1000, 5000]
    n_trials = 50

    # Collect statistics
    S_means = []
    S_stds = []

    for n_shots in n_shots_range:
        S_trials = []
        for trial in range(n_trials):
            E_ab = simulate_experiment(phi_plus, a, b, n_shots, seed=trial*4)
            E_ab_prime = simulate_experiment(phi_plus, a, b_prime, n_shots, seed=trial*4+1)
            E_a_prime_b = simulate_experiment(phi_plus, a_prime, b, n_shots, seed=trial*4+2)
            E_a_prime_b_prime = simulate_experiment(phi_plus, a_prime, b_prime, n_shots, seed=trial*4+3)

            S = E_ab - E_ab_prime + E_a_prime_b + E_a_prime_b_prime
            S_trials.append(abs(S))

        S_means.append(np.mean(S_trials))
        S_stds.append(np.std(S_trials))

    ax3.errorbar(n_shots_range, S_means, yerr=S_stds, fmt='bo-', lw=2, markersize=8, capsize=5)
    ax3.axhline(2, color='green', linestyle='--', lw=2, label='Classical bound')
    ax3.axhline(2*np.sqrt(2), color='red', linestyle=':', lw=2, label='Tsirelson bound')

    ax3.set_xlabel('Number of measurement shots')
    ax3.set_ylabel('CHSH parameter |S|')
    ax3.set_title('Statistical Violation of CHSH Inequality\n(Error bars: std over 50 trials)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    ax3.set_ylim(1.5, 3.0)

    # ===== Plot 4: Different Bell states =====
    ax4 = axes[1, 1]

    states = {
        'phi_plus': create_bell_state('phi_plus'),
        'phi_minus': create_bell_state('phi_minus'),
        'psi_plus': create_bell_state('psi_plus'),
        'psi_minus': create_bell_state('psi_minus'),
        'separable': np.array([1, 0, 0, 0], dtype=complex),  # |00>
        'mixed': np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)  # Same as phi_plus
    }

    state_names = list(states.keys())[:-1]  # Skip 'mixed' (duplicate)
    S_optimal = []

    for name in state_names:
        state = states[name]

        # Find maximum |S| over all angle settings
        best_S = 0
        for a in np.linspace(0, np.pi, 20):
            for a_p in np.linspace(0, np.pi, 20):
                for b in np.linspace(0, np.pi, 20):
                    b_p = b + np.pi/2
                    S = abs(chsh_parameter(state, a, a_p, b, b_p))
                    if S > best_S:
                        best_S = S

        S_optimal.append(best_S)

    colors = ['blue', 'red', 'green', 'purple', 'gray']
    bars = ax4.bar(state_names, S_optimal, color=colors, alpha=0.7)

    ax4.axhline(2, color='green', linestyle='--', lw=2, label='Classical bound')
    ax4.axhline(2*np.sqrt(2), color='red', linestyle=':', lw=2, label='Tsirelson bound')

    ax4.set_ylabel('Maximum |S|')
    ax4.set_title('CHSH Violation for Different States\n(Only entangled states violate)')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # Annotate
    for bar, S in zip(bars, S_optimal):
        ax4.text(bar.get_x() + bar.get_width()/2, S + 0.1, f'{S:.2f}',
                ha='center', fontsize=10)

    plt.suptitle('Bell Inequality (CHSH) Violation\n'
                 'Proof of Quantum Nonlocality',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'bell_chsh.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'bell_chsh.png')}")

    # Print results
    print("\n=== Bell Inequality (CHSH) Results ===")
    print(f"\nOptimal measurement settings:")
    print(f"  Alice: a = 0, a' = pi/2")
    print(f"  Bob: b = pi/4, b' = 3*pi/4")

    print(f"\nCHSH parameter with |Phi+>:")
    S_opt = chsh_parameter(phi_plus, 0, np.pi/2, np.pi/4, 3*np.pi/4)
    print(f"  S = {S_opt:.4f}")
    print(f"  Classical bound: |S| <= 2")
    print(f"  Tsirelson bound: |S| <= 2*sqrt(2) = {2*np.sqrt(2):.4f}")
    print(f"  Violation: {abs(S_opt) - 2:.4f} above classical bound")

    print(f"\nCorrelation values:")
    print(f"  E(a, b) = {correlation(phi_plus, 0, np.pi/4):.4f}")
    print(f"  E(a, b') = {correlation(phi_plus, 0, 3*np.pi/4):.4f}")
    print(f"  E(a', b) = {correlation(phi_plus, np.pi/2, np.pi/4):.4f}")
    print(f"  E(a', b') = {correlation(phi_plus, np.pi/2, 3*np.pi/4):.4f}")


if __name__ == "__main__":
    main()
