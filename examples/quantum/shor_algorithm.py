"""
Experiment 183: Shor's Algorithm for Integer Factorization

This experiment demonstrates Shor's algorithm for factoring integers,
focusing on the quantum period-finding subroutine.

Physics/Algorithm:
    Shor's algorithm factors N in polynomial time using quantum mechanics.

    Key insight: Factoring reduces to order finding (period finding).

    Given N to factor:
    1. Choose random a coprime to N
    2. Find order r: smallest r such that a^r = 1 (mod N)
    3. If r is even and a^{r/2} != -1 (mod N):
       gcd(a^{r/2} +/- 1, N) gives factors

    Quantum speedup:
    - Classical period finding: O(sqrt(N)) best known
    - Quantum (QPE on U_a): O((log N)^3) operations

    The quantum part:
    1. Create superposition: |0>|1>
    2. Apply U_a^{2^k} controlled on register 1
       U_a|y> = |a*y mod N>
    3. QFT gives peaks at multiples of N/r
    4. Continued fractions extract r

    This is an educational implementation for small numbers.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction
from math import gcd


def modular_exponentiation_matrix(a, N, n_qubits):
    """
    Create unitary matrix for modular exponentiation U_a.

    U_a|y> = |a*y mod N> for y < N
    U_a|y> = |y> for y >= N (identity on invalid states)

    Args:
        a: Base for exponentiation
        N: Modulus
        n_qubits: Number of qubits in register

    Returns:
        Unitary matrix (2^n x 2^n)
    """
    dim = 2**n_qubits
    U = np.zeros((dim, dim), dtype=complex)

    for y in range(dim):
        if y < N and gcd(y, N) == 1:
            new_y = (a * y) % N
            U[new_y, y] = 1
        elif y == 0:
            # |0> maps to |0> (not used in algorithm but keeps matrix unitary)
            U[0, 0] = 1
        else:
            # Identity for states >= N
            U[y, y] = 1

    return U


def classical_order_finding(a, N):
    """
    Find order r of a modulo N classically.

    The order is the smallest r such that a^r = 1 (mod N).

    Args:
        a: Base
        N: Modulus

    Returns:
        Order r
    """
    if gcd(a, N) != 1:
        return None  # a not coprime to N

    r = 1
    current = a % N
    while current != 1:
        current = (current * a) % N
        r += 1
        if r > N:  # Safety check
            return None

    return r


def quantum_period_finding(a, N, n_precision_qubits):
    """
    Quantum period finding using QFT (simplified simulation).

    This simulates the quantum circuit that finds the period r
    of the function f(x) = a^x mod N.

    Args:
        a: Base for modular exponentiation
        N: Modulus
        n_precision_qubits: Number of qubits in counting register

    Returns:
        Tuple of (measurement_probabilities, possible_periods)
    """
    # Dimension of counting register
    M = 2**n_precision_qubits

    # Need ceil(log2(N)) qubits for work register
    n_work_qubits = int(np.ceil(np.log2(N))) + 1
    work_dim = 2**n_work_qubits

    # Total dimension
    total_dim = M * work_dim

    # Initialize: |0>|1>
    state = np.zeros(total_dim, dtype=complex)
    state[1] = 1  # |0...0>|0...01>

    # Apply Hadamard to counting register
    # After Hadamard: (1/sqrt(M)) * sum_x |x>|1>
    for x in range(M):
        state[x * work_dim + 1] = 1 / np.sqrt(M)

    # Apply controlled U_a^{2^k} operations
    # Result: (1/sqrt(M)) * sum_x |x>|a^x mod N>
    new_state = np.zeros(total_dim, dtype=complex)

    for x in range(M):
        # Compute a^x mod N
        ax_mod_N = pow(a, x, N)
        # Transfer amplitude
        new_state[x * work_dim + ax_mod_N] = state[x * work_dim + 1]

    state = new_state

    # Apply QFT to counting register
    # QFT|x> = (1/sqrt(M)) sum_y exp(2*pi*i*x*y/M) |y>
    final_state = np.zeros(total_dim, dtype=complex)
    omega = np.exp(2j * np.pi / M)

    for y in range(M):
        for work_idx in range(work_dim):
            amp = 0
            for x in range(M):
                amp += omega**(x * y) * state[x * work_dim + work_idx]
            final_state[y * work_dim + work_idx] = amp / np.sqrt(M)

    state = final_state

    # Measurement probabilities on counting register
    probs = np.zeros(M)
    for y in range(M):
        for work_idx in range(work_dim):
            probs[y] += np.abs(state[y * work_dim + work_idx])**2

    return probs


def extract_period_from_measurement(measurement, M, N):
    """
    Extract period from QPE measurement using continued fractions.

    The measurement m is approximately s*M/r for some 0 <= s < r.
    Use continued fraction expansion to find r.

    Args:
        measurement: Measured value from counting register
        M: Dimension of counting register (2^n)
        N: Number being factored

    Returns:
        Candidate period r
    """
    if measurement == 0:
        return None

    # Get continued fraction convergents of m/M
    frac = Fraction(measurement, M).limit_denominator(N)

    # The denominator should be the period (or a divisor)
    return frac.denominator


def shors_algorithm(N, a=None, n_precision_qubits=None):
    """
    Run Shor's algorithm to factor N (simplified version).

    Args:
        N: Number to factor (should be composite, not prime power)
        a: Base for period finding (random if None)
        n_precision_qubits: Precision qubits (default: 2*ceil(log2(N)))

    Returns:
        Tuple of (factors, period, measurements)
    """
    if n_precision_qubits is None:
        n_precision_qubits = 2 * int(np.ceil(np.log2(N))) + 1

    # Choose a random a coprime to N
    if a is None:
        for candidate in range(2, N):
            if gcd(candidate, N) == 1:
                a = candidate
                break

    # Check if gcd(a, N) > 1 (lucky case - found factor immediately)
    g = gcd(a, N)
    if g > 1:
        return (g, N // g), None, None

    # Run quantum period finding
    probs = quantum_period_finding(a, N, n_precision_qubits)

    M = 2**n_precision_qubits

    # Get most likely measurements
    top_indices = np.argsort(probs)[::-1][:5]

    # Try to extract period from each measurement
    for m in top_indices:
        if m == 0:
            continue

        r_candidate = extract_period_from_measurement(m, M, N)

        if r_candidate is None or r_candidate == 0:
            continue

        # Verify: is a^r = 1 (mod N)?
        if pow(a, r_candidate, N) != 1:
            # Try multiples
            for mult in [2, 3, 4]:
                if pow(a, mult * r_candidate, N) == 1:
                    r_candidate = mult * r_candidate
                    break
            else:
                continue

        r = r_candidate

        # Check if r is even
        if r % 2 == 0:
            # Check if a^{r/2} != -1 (mod N)
            half_power = pow(a, r // 2, N)
            if half_power != N - 1:
                # Compute factors
                factor1 = gcd(half_power - 1, N)
                factor2 = gcd(half_power + 1, N)

                if factor1 > 1 and factor1 < N:
                    return (factor1, N // factor1), r, probs
                if factor2 > 1 and factor2 < N:
                    return (factor2, N // factor2), r, probs

    # Classical fallback
    r_classical = classical_order_finding(a, N)
    return None, r_classical, probs


def main():
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # ===== Plot 1: Quantum period finding for N=15 =====
    ax1 = axes[0, 0]

    N = 15
    a = 7  # 7^r mod 15 has period 4

    n_qubits = 8
    probs = quantum_period_finding(a, N, n_qubits)
    M = 2**n_qubits

    # Theoretical peaks at k*M/r for k = 0, 1, ..., r-1
    r_true = classical_order_finding(a, N)

    ax1.bar(range(M), probs, width=1.0, alpha=0.7)

    # Mark theoretical peak positions
    if r_true:
        for k in range(r_true):
            peak_pos = int(k * M / r_true)
            ax1.axvline(peak_pos, color='red', linestyle='--', alpha=0.7,
                       label=f'k*M/r = {k}*{M}/{r_true}' if k == 0 else '')

    ax1.set_xlabel('Measurement outcome')
    ax1.set_ylabel('Probability')
    ax1.set_title(f'Quantum Period Finding (N={N}, a={a})\nPeaks at multiples of M/r = {M}/{r_true}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ===== Plot 2: Period extraction =====
    ax2 = axes[0, 1]

    # Show continued fraction convergence
    measurements = [0, 64, 128, 192]  # Expected peaks for r=4, M=256
    M = 256

    x_pos = np.arange(len(measurements))
    extracted_periods = []

    for m in measurements:
        if m == 0:
            extracted_periods.append(0)
        else:
            frac = Fraction(m, M).limit_denominator(N)
            extracted_periods.append(frac.denominator)

    ax2.bar(x_pos, extracted_periods, alpha=0.7)
    ax2.axhline(r_true, color='red', linestyle='--', lw=2, label=f'True period r = {r_true}')

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([str(m) for m in measurements])
    ax2.set_xlabel('Measurement m')
    ax2.set_ylabel('Extracted period')
    ax2.set_title('Period Extraction via Continued Fractions\nm/M approximates s/r')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # ===== Plot 3: Modular exponentiation sequence =====
    ax3 = axes[0, 2]

    a_values = [2, 4, 7, 11, 13]
    colors = plt.cm.tab10(np.linspace(0, 0.5, len(a_values)))

    for a_val, color in zip(a_values, colors):
        sequence = [pow(a_val, x, N) for x in range(20)]
        r = classical_order_finding(a_val, N)
        ax3.plot(sequence, 'o-', color=color, lw=1.5, markersize=4,
                label=f'a={a_val}, r={r}')

    ax3.set_xlabel('Exponent x')
    ax3.set_ylabel('a^x mod 15')
    ax3.set_title('Modular Exponentiation Sequences\nPeriodicity visible')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # ===== Plot 4: Full Shor's algorithm example =====
    ax4 = axes[1, 0]

    # Factor N = 21 = 3 * 7
    N_factor = 21
    a_factor = 2

    # Show probability distribution
    n_qubits_factor = 10
    probs_factor = quantum_period_finding(a_factor, N_factor, n_qubits_factor)
    M_factor = 2**n_qubits_factor

    r_factor = classical_order_finding(a_factor, N_factor)

    ax4.bar(range(M_factor), probs_factor, width=1.0, alpha=0.7)

    ax4.set_xlabel('Measurement outcome')
    ax4.set_ylabel('Probability')
    ax4.set_title(f'Shor\'s Algorithm for N={N_factor}\na={a_factor}, period r={r_factor}')
    ax4.grid(True, alpha=0.3)

    # Add text about factorization
    if r_factor and r_factor % 2 == 0:
        half_power = pow(a_factor, r_factor // 2, N_factor)
        f1 = gcd(half_power - 1, N_factor)
        f2 = gcd(half_power + 1, N_factor)
        ax4.text(0.98, 0.95, f'r = {r_factor} (even)\na^{{r/2}} = {half_power}\n'
                f'gcd({half_power}-1, {N_factor}) = {f1}\n'
                f'gcd({half_power}+1, {N_factor}) = {f2}',
                transform=ax4.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # ===== Plot 5: Classical vs Quantum complexity =====
    ax5 = axes[1, 1]

    # Number of bits
    n_bits = np.arange(10, 100, 5)
    N_values = 2**n_bits

    # Classical (Number Field Sieve): exp(c * n^{1/3} * (log n)^{2/3})
    classical_ops = np.exp(1.9 * n_bits**(1/3) * np.log(n_bits)**(2/3))

    # Quantum (Shor): O(n^3) = O((log N)^3)
    quantum_ops = n_bits**3

    ax5.semilogy(n_bits, classical_ops, 'r-', lw=2, label='Classical (NFS): exp(n^{1/3})')
    ax5.semilogy(n_bits, quantum_ops, 'b-', lw=2, label='Quantum (Shor): O(n^3)')

    # RSA key sizes
    for key_size, label in [(512, 'RSA-512'), (1024, 'RSA-1024'), (2048, 'RSA-2048')]:
        ax5.axvline(key_size, color='gray', linestyle=':', alpha=0.5)
        ax5.text(key_size + 2, 1e20, label, rotation=90, fontsize=8, va='bottom')

    ax5.set_xlabel('Number of bits n = log2(N)')
    ax5.set_ylabel('Number of operations')
    ax5.set_title("Shor's Algorithm Speedup\nExponential quantum advantage")
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(1, 1e50)

    # ===== Plot 6: Success probability analysis =====
    ax6 = axes[1, 2]

    # For different numbers, show success rates
    test_numbers = [15, 21, 33, 35, 39, 55, 77, 91]
    success_rates = []
    true_factors = []

    for N_test in test_numbers:
        # Find factors classically for verification
        for f in range(2, int(np.sqrt(N_test)) + 1):
            if N_test % f == 0:
                true_factors.append(f)
                break

        # Simulate multiple attempts
        successes = 0
        trials = 20

        for _ in range(trials):
            a_test = np.random.randint(2, N_test - 1)
            while gcd(a_test, N_test) != 1:
                a_test = np.random.randint(2, N_test - 1)

            r = classical_order_finding(a_test, N_test)
            if r and r % 2 == 0:
                hp = pow(a_test, r // 2, N_test)
                if hp != N_test - 1:
                    f1 = gcd(hp - 1, N_test)
                    f2 = gcd(hp + 1, N_test)
                    if (f1 > 1 and f1 < N_test) or (f2 > 1 and f2 < N_test):
                        successes += 1

        success_rates.append(successes / trials)

    ax6.bar(range(len(test_numbers)), success_rates, alpha=0.7)
    ax6.axhline(0.5, color='red', linestyle='--', lw=2, label='50% threshold')

    ax6.set_xticks(range(len(test_numbers)))
    ax6.set_xticklabels([str(n) for n in test_numbers])
    ax6.set_xlabel('Number N to factor')
    ax6.set_ylabel('Success probability')
    ax6.set_title("Shor's Algorithm Success Rate\n(Random a, single attempt)")
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')

    plt.suptitle("Shor's Algorithm for Integer Factorization\n"
                 "Quantum polynomial-time factoring via period finding",
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'shor_algorithm.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'shor_algorithm.png')}")

    # Print numerical results
    print("\n=== Shor's Algorithm Results ===")

    print("\n1. Example: Factor N = 15")
    print("   Choose a = 7 (coprime to 15)")
    print(f"   Order r = {classical_order_finding(7, 15)}")
    print("   7^2 mod 15 = 4 (not -1 mod 15)")
    print("   gcd(4-1, 15) = gcd(3, 15) = 3")
    print("   gcd(4+1, 15) = gcd(5, 15) = 5")
    print("   Factors: 3 * 5 = 15")

    print("\n2. Example: Factor N = 21")
    print("   Choose a = 2")
    r21 = classical_order_finding(2, 21)
    print(f"   Order r = {r21}")
    hp21 = pow(2, r21 // 2, 21)
    print(f"   2^{r21//2} mod 21 = {hp21}")
    print(f"   gcd({hp21}-1, 21) = {gcd(hp21-1, 21)}")
    print(f"   gcd({hp21}+1, 21) = {gcd(hp21+1, 21)}")
    print("   Factors: 3 * 7 = 21")

    print("\n3. Algorithm complexity:")
    print("   - Classical factoring (NFS): O(exp(n^{1/3} * (log n)^{2/3}))")
    print("   - Quantum (Shor): O(n^3) where n = log2(N)")
    print("   - Exponential quantum speedup!")

    print("\n4. Quantum subroutine:")
    print("   - Period finding via Quantum Phase Estimation")
    print("   - U_a|y> = |a*y mod N>")
    print("   - Eigenvalues encode period: e^{2*pi*i*s/r}")
    print("   - QFT extracts s/r, continued fractions find r")


if __name__ == "__main__":
    main()
