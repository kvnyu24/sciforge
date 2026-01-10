"""
Experiment 158: Commutator and Uncertainty Relations

This experiment demonstrates the connection between commutators and the
Heisenberg uncertainty principle through the Robertson uncertainty relation.

Physics:
    For two observables A and B with commutator [A,B] = AB - BA:

    Robertson uncertainty relation:
        sigma_A * sigma_B >= |<[A,B]>| / 2

    where sigma_A = sqrt(<A^2> - <A>^2) is the standard deviation.

    Key commutators:
        [x, p] = i*hbar           (position-momentum)
        [L_x, L_y] = i*hbar*L_z   (angular momentum)
        [a, a^dag] = 1            (ladder operators)

    Minimum uncertainty states saturate the inequality.
    For [x,p], Gaussian wavepackets are minimum uncertainty states.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid


# Pauli matrices for spin-1/2
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)


def commutator(A, B):
    """
    Compute the commutator [A, B] = AB - BA.

    Args:
        A: Matrix or operator
        B: Matrix or operator

    Returns:
        Commutator matrix
    """
    return A @ B - B @ A


def expectation_value(operator, state):
    """
    Compute expectation value <psi|O|psi>.

    Args:
        operator: Hermitian operator (matrix)
        state: State vector (normalized)

    Returns:
        Expectation value (real)
    """
    return np.real(np.conj(state) @ operator @ state)


def variance(operator, state):
    """
    Compute variance <A^2> - <A>^2.

    Args:
        operator: Hermitian operator
        state: State vector

    Returns:
        Variance
    """
    exp_A = expectation_value(operator, state)
    exp_A2 = expectation_value(operator @ operator, state)
    return exp_A2 - exp_A**2


def robertson_inequality(A, B, state):
    """
    Check Robertson uncertainty relation.

    sigma_A * sigma_B >= |<[A,B]>| / 2

    Args:
        A, B: Operators (matrices)
        state: State vector

    Returns:
        Tuple of (sigma_A, sigma_B, product, lower_bound, saturated)
    """
    sigma_A = np.sqrt(max(0, variance(A, state)))
    sigma_B = np.sqrt(max(0, variance(B, state)))

    comm_AB = commutator(A, B)
    exp_comm = expectation_value(comm_AB, state)

    product = sigma_A * sigma_B
    lower_bound = np.abs(exp_comm) / 2

    # Check if approximately saturated (minimum uncertainty)
    saturated = np.isclose(product, lower_bound, rtol=0.01)

    return sigma_A, sigma_B, product, lower_bound, saturated


def angular_momentum_matrices(j):
    """
    Construct angular momentum matrices J_x, J_y, J_z for spin j.

    Uses: J_+ = J_x + iJ_y, J_- = J_x - iJ_y
          J_+|j,m> = sqrt(j(j+1) - m(m+1)) * |j,m+1>
          J_-|j,m> = sqrt(j(j+1) - m(m-1)) * |j,m-1>

    Args:
        j: Angular momentum quantum number (half-integer or integer)

    Returns:
        Tuple of (J_x, J_y, J_z) matrices
    """
    dim = int(2*j + 1)
    m_values = np.arange(j, -j-1, -1)  # m = j, j-1, ..., -j

    J_plus = np.zeros((dim, dim), dtype=complex)
    J_minus = np.zeros((dim, dim), dtype=complex)
    J_z = np.zeros((dim, dim), dtype=complex)

    for i, m in enumerate(m_values):
        J_z[i, i] = m
        if i > 0:  # J_+ raises m
            J_plus[i-1, i] = np.sqrt(j*(j+1) - m*(m+1))
        if i < dim-1:  # J_- lowers m
            J_minus[i+1, i] = np.sqrt(j*(j+1) - m*(m-1))

    J_x = (J_plus + J_minus) / 2
    J_y = (J_plus - J_minus) / (2j * 1j) if j != 0 else (J_plus - J_minus) / (2 * 1j)

    return J_x, J_y, J_z


def ladder_operators(n_states):
    """
    Create annihilation and creation operators for harmonic oscillator.

    a|n> = sqrt(n)|n-1>
    a^dag|n> = sqrt(n+1)|n+1>
    [a, a^dag] = 1

    Args:
        n_states: Number of Fock states to include

    Returns:
        Tuple of (a, a_dag) matrices
    """
    a = np.zeros((n_states, n_states), dtype=complex)
    for n in range(1, n_states):
        a[n-1, n] = np.sqrt(n)

    a_dag = a.conj().T

    return a, a_dag


def position_momentum_grid(n_points=512, x_max=10.0, hbar=1.0):
    """
    Create position and momentum operators on a grid.

    Args:
        n_points: Number of grid points
        x_max: Maximum position
        hbar: Reduced Planck constant

    Returns:
        Tuple of (x_grid, x_op, p_op)
    """
    x = np.linspace(-x_max, x_max, n_points)
    dx = x[1] - x[0]

    # Position operator (diagonal)
    x_op = np.diag(x)

    # Momentum operator (via finite differences)
    # p = -i*hbar * d/dx
    p_op = np.zeros((n_points, n_points), dtype=complex)
    for i in range(n_points):
        if i > 0:
            p_op[i, i-1] = 1j * hbar / (2 * dx)
        if i < n_points - 1:
            p_op[i, i+1] = -1j * hbar / (2 * dx)

    return x, x_op, p_op


def gaussian_state_on_grid(x, x0, sigma, k0=0, hbar=1.0):
    """
    Create normalized Gaussian wavepacket on grid.

    Args:
        x: Position grid
        x0: Center position
        sigma: Width parameter
        k0: Central momentum
        hbar: Reduced Planck constant

    Returns:
        Normalized state vector
    """
    dx = x[1] - x[0]
    psi = (2 * np.pi * sigma**2)**(-0.25) * np.exp(-(x - x0)**2 / (4 * sigma**2))
    psi = psi.astype(complex) * np.exp(1j * k0 * x / hbar)

    # Normalize
    norm = np.sqrt(trapezoid(np.abs(psi)**2, x))
    return psi / norm


def squeezed_state_on_grid(x, x0, sigma_x, sigma_p, hbar=1.0):
    """
    Create squeezed Gaussian state (non-minimum uncertainty in general).

    Args:
        x: Position grid
        x0: Center position
        sigma_x: Position uncertainty
        sigma_p: Momentum uncertainty (not enforced)
        hbar: Reduced Planck constant

    Returns:
        State vector
    """
    dx = x[1] - x[0]
    psi = (2 * np.pi * sigma_x**2)**(-0.25) * np.exp(-(x - x0)**2 / (4 * sigma_x**2))

    # Normalize
    norm = np.sqrt(trapezoid(np.abs(psi)**2, x))
    return psi / norm


def main():
    # Set up figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # ===== Plot 1: [x, p] = i*hbar verification =====
    ax1 = axes[0, 0]

    hbar = 1.0
    n_points = 256
    x_max = 10.0
    x, x_op, p_op = position_momentum_grid(n_points, x_max, hbar)

    # Compute [x, p]
    comm_xp = commutator(x_op, p_op)

    # Should be i*hbar * I
    expected = 1j * hbar * np.eye(n_points)

    # Compare diagonal elements (boundary effects at edges)
    diag_comm = np.diag(comm_xp)
    diag_expected = np.diag(expected)

    interior = slice(n_points//4, 3*n_points//4)
    ax1.plot(x[interior], np.real(diag_comm[interior]), 'b-', lw=2, label='Re([x,p])')
    ax1.plot(x[interior], np.imag(diag_comm[interior]), 'r-', lw=2, label='Im([x,p])')
    ax1.axhline(hbar, color='green', linestyle='--', lw=1.5, label=f'Expected: i*hbar = i*{hbar}')

    ax1.set_xlabel('Position x')
    ax1.set_ylabel('Commutator diagonal')
    ax1.set_title('[x, p] = i*hbar (Canonical Commutation)\n(Grid representation)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ===== Plot 2: [L_x, L_y] = i*hbar*L_z verification =====
    ax2 = axes[0, 1]

    j_values = [0.5, 1, 1.5, 2]

    for j in j_values:
        J_x, J_y, J_z = angular_momentum_matrices(j)

        # Compute [J_x, J_y]
        comm_xy = commutator(J_x, J_y)
        expected_comm = 1j * hbar * J_z

        # Compare Frobenius norm
        diff = np.linalg.norm(comm_xy - expected_comm, 'fro')

        # Take sample matrix element
        dim = int(2*j + 1)
        ax2.scatter([j], [diff], s=100, label=f'j={j}, dim={dim}')

    ax2.axhline(0, color='green', linestyle='--', lw=1.5, label='Expected: 0')
    ax2.set_xlabel('Angular momentum j')
    ax2.set_ylabel('||[Jx,Jy] - i*hbar*Jz||')
    ax2.set_title('[L_x, L_y] = i*hbar*L_z\n(Verification for different j)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ===== Plot 3: [a, a^dag] = 1 =====
    ax3 = axes[0, 2]

    n_states_list = [5, 10, 20, 50]

    for n_states in n_states_list:
        a, a_dag = ladder_operators(n_states)

        comm_aa = commutator(a, a_dag)
        expected_comm = np.eye(n_states)

        # Diagonal elements
        diag = np.real(np.diag(comm_aa))
        ax3.plot(range(n_states), diag, 'o-', lw=1.5,
                label=f'N={n_states}', alpha=0.7)

    ax3.axhline(1, color='green', linestyle='--', lw=2, label='Expected: 1')
    ax3.set_xlabel('Fock state |n>')
    ax3.set_ylabel('Diagonal of [a, a^dag]')
    ax3.set_title('[a, a^dag] = 1 (Bosonic Commutator)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # ===== Plot 4: Robertson inequality for spin-1/2 =====
    ax4 = axes[1, 0]

    # Test various spin states (use module-level Pauli matrices)
    hbar = 1.0
    pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
    pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
    S_x = hbar/2 * pauli_x
    S_y = hbar/2 * pauli_y
    S_z = hbar/2 * pauli_z

    # Parametrize states on Bloch sphere
    theta_vals = np.linspace(0, np.pi, 50)
    phi = 0  # Fix phi

    products = []
    lower_bounds = []

    for theta in theta_vals:
        # State |psi> = cos(theta/2)|0> + sin(theta/2)e^{i*phi}|1>
        state = np.array([np.cos(theta/2), np.sin(theta/2) * np.exp(1j * phi)], dtype=complex)

        sigma_x_val, sigma_y_val, prod, lb, sat = robertson_inequality(S_x, S_y, state)
        products.append(prod)
        lower_bounds.append(lb)

    ax4.plot(theta_vals / np.pi, products, 'b-', lw=2, label='sigma_x * sigma_y')
    ax4.plot(theta_vals / np.pi, lower_bounds, 'r--', lw=2, label='|<[Sx,Sy]>|/2')
    ax4.fill_between(theta_vals / np.pi, lower_bounds, products, alpha=0.3, color='blue')

    ax4.set_xlabel('Polar angle theta/pi')
    ax4.set_ylabel('Uncertainty product (hbar^2)')
    ax4.set_title('Robertson Inequality: Spin-1/2\nsigma_x * sigma_y >= |<[Sx,Sy]>|/2')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # ===== Plot 5: Minimum uncertainty states (x-p) =====
    ax5 = axes[1, 1]

    # Different Gaussian widths
    sigma_values = np.linspace(0.5, 3, 30)
    products_xp = []
    min_bound = hbar / 2

    for sigma in sigma_values:
        psi = gaussian_state_on_grid(x, 0, sigma, k0=0, hbar=hbar)

        # Compute uncertainties
        exp_x = np.real(trapezoid(x * np.abs(psi)**2, x))
        exp_x2 = np.real(trapezoid(x**2 * np.abs(psi)**2, x))
        var_x = exp_x2 - exp_x**2

        # Analytical for Gaussian: sigma_x = sigma, sigma_p = hbar/(2*sigma)
        sigma_x = np.sqrt(var_x)
        sigma_p = hbar / (2 * sigma)  # Analytical

        products_xp.append(sigma_x * sigma_p)

    ax5.plot(sigma_values, products_xp, 'b-', lw=2, marker='o', markersize=4,
            label='Gaussian wavepackets')
    ax5.axhline(min_bound, color='red', linestyle='--', lw=2,
                label=f'Minimum = hbar/2 = {min_bound:.2f}')

    ax5.set_xlabel('Gaussian width sigma')
    ax5.set_ylabel('sigma_x * sigma_p')
    ax5.set_title('Minimum Uncertainty States\nGaussians saturate sigma_x*sigma_p = hbar/2')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, 1)

    # ===== Plot 6: Non-minimum uncertainty comparison =====
    ax6 = axes[1, 2]

    # Compare different wavefunction shapes
    wf_types = ['gaussian', 'box', 'triangle', 'exponential']
    colors = plt.cm.tab10(np.linspace(0, 0.4, len(wf_types)))

    products_wf = []

    for wf_type, color in zip(wf_types, colors):
        dx = x[1] - x[0]

        if wf_type == 'gaussian':
            psi = (2 * np.pi)**(-0.25) * np.exp(-x**2 / 4)
        elif wf_type == 'box':
            psi = np.zeros_like(x)
            psi[np.abs(x) < 2] = 1.0
        elif wf_type == 'triangle':
            psi = np.zeros_like(x)
            mask = np.abs(x) < 2
            psi[mask] = 1 - np.abs(x[mask]) / 2
        elif wf_type == 'exponential':
            psi = np.exp(-np.abs(x))

        # Normalize
        psi = psi / np.sqrt(trapezoid(np.abs(psi)**2, x))

        # Position uncertainty
        exp_x = trapezoid(x * np.abs(psi)**2, x)
        exp_x2 = trapezoid(x**2 * np.abs(psi)**2, x)
        sigma_x = np.sqrt(exp_x2 - exp_x**2)

        # Momentum uncertainty via Fourier
        psi_fft = np.fft.fft(psi) * dx / np.sqrt(2 * np.pi)
        k = 2 * np.pi * np.fft.fftfreq(len(x), dx)
        p = hbar * k
        prob_p = np.abs(psi_fft)**2
        prob_p = prob_p / np.sum(prob_p)  # Normalize

        exp_p = np.sum(p * prob_p)
        exp_p2 = np.sum(p**2 * prob_p)
        sigma_p = np.sqrt(exp_p2 - exp_p**2)

        prod = sigma_x * sigma_p
        products_wf.append(prod)

        ax6.bar(wf_type, prod, color=color, alpha=0.7)

    ax6.axhline(hbar/2, color='red', linestyle='--', lw=2, label='Minimum hbar/2')

    ax6.set_ylabel('sigma_x * sigma_p')
    ax6.set_title('Uncertainty Product Comparison\n(Only Gaussian is minimum uncertainty)')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for i, (wf_type, prod) in enumerate(zip(wf_types, products_wf)):
        ax6.text(i, prod + 0.05, f'{prod:.2f}', ha='center', fontsize=9)

    plt.suptitle('Commutators and the Robertson Uncertainty Relation\n'
                 r'$\sigma_A \sigma_B \geq \frac{1}{2}|\langle[A,B]\rangle|$',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'commutator_uncertainty.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'commutator_uncertainty.png')}")

    # Print numerical results
    print("\n=== Commutator and Uncertainty Relations ===")

    print("\n1. Canonical commutators:")
    a, a_dag = ladder_operators(10)
    comm = commutator(a, a_dag)
    print(f"   [a, a^dag] = {np.real(np.diag(comm)[0]):.4f} (expected: 1)")

    J_x, J_y, J_z = angular_momentum_matrices(1)
    comm_jxy = commutator(J_x, J_y)
    expected_jz = 1j * hbar * J_z
    print(f"   [Jx, Jy] = i*hbar*Jz: error = {np.linalg.norm(comm_jxy - expected_jz):.2e}")

    print("\n2. Uncertainty products for different states:")
    for wf_type, prod in zip(wf_types, products_wf):
        ratio = prod / (hbar/2)
        print(f"   {wf_type:12s}: sigma_x*sigma_p = {prod:.4f} = {ratio:.2f} * (hbar/2)")

    print("\n3. Robertson inequality verification (spin-1/2):")
    state_z = np.array([1, 0], dtype=complex)  # |+z>
    state_x = np.array([1, 1], dtype=complex) / np.sqrt(2)  # |+x>
    state_gen = np.array([1, 1j], dtype=complex) / np.sqrt(2)  # General

    for name, state in [('|+z>', state_z), ('|+x>', state_x), ('general', state_gen)]:
        sx, sy, prod, lb, sat = robertson_inequality(S_x, S_y, state)
        print(f"   {name:8s}: sigma_x*sigma_y = {prod:.4f} >= {lb:.4f} (saturated: {sat})")


if __name__ == "__main__":
    main()
