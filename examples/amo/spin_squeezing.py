"""
Experiment 253: Spin Squeezing Model

This example demonstrates spin squeezing in atomic ensembles, where quantum
correlations reduce fluctuations below the standard quantum limit (SQL).
We explore:
- Collective spin states and Bloch sphere representation
- Standard quantum limit (SQL) for N uncorrelated atoms
- One-axis twisting Hamiltonian for squeezing
- Squeezing parameter and Wineland criterion
- Application to atomic clocks and quantum metrology

Key physics:
- Collective spin: J = sum_i s_i, with |J| = N/2 for N spin-1/2 particles
- SQL: Delta_theta = 1/sqrt(N)
- Squeezing: reduce variance in one quadrature, increase in conjugate
- Squeezing parameter: xi^2 = N * (Delta J_perp)^2 / <J>^2
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from src.sciforge.physics.amo import HBAR

def pauli_matrices():
    """Return Pauli matrices."""
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    return sigma_x, sigma_y, sigma_z


def collective_spin_operators(N):
    """
    Construct collective spin operators for N spin-1/2 particles.

    Uses the symmetric (Dicke) subspace for computational efficiency.
    Dimension: N+1 (from m = -N/2 to +N/2)

    Args:
        N: Number of atoms

    Returns:
        Jx, Jy, Jz, J2 matrices
    """
    j = N / 2  # Total spin
    dim = N + 1

    # m values from +j to -j
    m_vals = np.arange(j, -j - 1, -1)

    # Jz is diagonal
    Jz = np.diag(m_vals)

    # J+ and J- ladder operators
    J_plus = np.zeros((dim, dim), dtype=complex)
    J_minus = np.zeros((dim, dim), dtype=complex)

    for i in range(dim - 1):
        m = m_vals[i + 1]
        J_plus[i, i + 1] = np.sqrt(j * (j + 1) - m * (m + 1))
        J_minus[i + 1, i] = np.sqrt(j * (j + 1) - m * (m - 1))

    Jx = (J_plus + J_minus) / 2
    Jy = (J_plus - J_minus) / (2 * 1j)

    J2 = Jx @ Jx + Jy @ Jy + Jz @ Jz

    return Jx, Jy, Jz, J2


def coherent_spin_state(N, theta, phi):
    """
    Create coherent spin state |theta, phi>.

    CSS = product state of N atoms all pointing in (theta, phi) direction.

    Args:
        N: Number of atoms
        theta: Polar angle
        phi: Azimuthal angle

    Returns:
        State vector in Dicke basis
    """
    j = N / 2
    dim = N + 1

    # CSS in terms of |j, m> states
    state = np.zeros(dim, dtype=complex)

    for idx, m in enumerate(np.arange(j, -j - 1, -1)):
        # Binomial coefficient
        k = int(j + m)
        from math import factorial
        binom = factorial(int(2 * j)) / (factorial(k) * factorial(int(2 * j - k)))

        amplitude = np.sqrt(binom) * (np.cos(theta / 2))**(j + m) * (np.sin(theta / 2))**(j - m)
        amplitude *= np.exp(-1j * m * phi)

        state[idx] = amplitude

    return state / np.linalg.norm(state)


def one_axis_twisting(Jz, chi, t):
    """
    One-axis twisting Hamiltonian evolution.

    H_OAT = chi * Jz^2

    Args:
        Jz: Collective Jz operator
        chi: Twisting strength
        t: Evolution time

    Returns:
        Unitary evolution operator
    """
    H = chi * Jz @ Jz
    return expm(-1j * H * t)


def calculate_squeezing(state, Jx, Jy, Jz, N):
    """
    Calculate squeezing parameter.

    Args:
        state: Quantum state vector
        Jx, Jy, Jz: Collective spin operators
        N: Number of atoms

    Returns:
        Dictionary with squeezing metrics
    """
    j = N / 2

    # Expectation values
    Jx_avg = np.real(np.conj(state) @ Jx @ state)
    Jy_avg = np.real(np.conj(state) @ Jy @ state)
    Jz_avg = np.real(np.conj(state) @ Jz @ state)

    # Mean spin length and direction
    J_mean = np.sqrt(Jx_avg**2 + Jy_avg**2 + Jz_avg**2)

    # Variances
    Jx2_avg = np.real(np.conj(state) @ Jx @ Jx @ state)
    Jy2_avg = np.real(np.conj(state) @ Jy @ Jy @ state)
    Jz2_avg = np.real(np.conj(state) @ Jz @ Jz @ state)

    var_Jx = Jx2_avg - Jx_avg**2
    var_Jy = Jy2_avg - Jy_avg**2
    var_Jz = Jz2_avg - Jz_avg**2

    # Covariance for finding minimum variance quadrature
    JyJz_avg = np.real(np.conj(state) @ (Jy @ Jz + Jz @ Jy) @ state / 2)
    cov_yz = JyJz_avg - Jy_avg * Jz_avg

    # Eigenvalues of variance matrix in y-z plane (perpendicular to x)
    # V = [[var_y, cov_yz], [cov_yz, var_z]]
    trace = var_Jy + var_Jz
    det = var_Jy * var_Jz - cov_yz**2
    discriminant = trace**2 - 4 * det
    if discriminant >= 0:
        sqrt_disc = np.sqrt(discriminant)
        var_perp_min = (trace - sqrt_disc) / 2
        var_perp_max = (trace + sqrt_disc) / 2
    else:
        var_perp_min = min(var_Jy, var_Jz)
        var_perp_max = max(var_Jy, var_Jz)

    # Ensure non-negative
    var_perp_min = max(0, var_perp_min)
    var_perp_max = max(0, var_perp_max)

    # SQL variance for CSS: var = j/2 = N/4
    var_SQL = j / 2

    # Squeezing parameter (Wineland)
    # xi^2 = N * var_perp_min / J_mean^2
    if J_mean > 0:
        xi_sq = N * var_perp_min / J_mean**2
    else:
        xi_sq = 1

    # Squeezing in dB
    xi_dB = 10 * np.log10(xi_sq) if xi_sq > 0 else 0

    return {
        'J_mean': J_mean,
        'J_max': j,
        'var_Jy': var_Jy,
        'var_Jz': var_Jz,
        'var_perp_min': var_perp_min,
        'var_perp_max': var_perp_max,
        'var_SQL': var_SQL,
        'xi_sq': xi_sq,
        'xi_dB': xi_dB,
        'Jx_avg': Jx_avg,
        'Jy_avg': Jy_avg,
        'Jz_avg': Jz_avg
    }


def simulate_spin_squeezing():
    """Simulate spin squeezing dynamics."""

    results = {}

    # Number of atoms (keep small for full calculation)
    N = 20  # 20 atoms for tractable Hilbert space
    j = N / 2

    results['N'] = N
    results['j'] = j

    print(f"Simulating {N} spin-1/2 particles")
    print(f"Hilbert space dimension: {N + 1}")

    # Construct collective spin operators
    print("Constructing collective spin operators...")
    Jx, Jy, Jz, J2 = collective_spin_operators(N)

    # Initial coherent spin state pointing along +x
    print("Creating initial coherent spin state...")
    psi_0 = coherent_spin_state(N, np.pi / 2, 0)  # theta=pi/2, phi=0 -> +x

    # Verify initial state
    init_metrics = calculate_squeezing(psi_0, Jx, Jy, Jz, N)
    print(f"Initial state: <Jx> = {init_metrics['Jx_avg']:.1f}, var = {init_metrics['var_SQL']:.2f}")

    results['initial'] = init_metrics

    # 1. Time evolution under one-axis twisting
    print("\nSimulating one-axis twisting evolution...")
    chi = 1.0  # Twisting parameter (normalized)
    t_max = np.pi / 2  # Maximum evolution time
    n_times = 100
    times = np.linspace(0, t_max, n_times)

    evolution = []
    for t in times:
        U = one_axis_twisting(Jz, chi, t)
        psi_t = U @ psi_0
        metrics = calculate_squeezing(psi_t, Jx, Jy, Jz, N)
        metrics['t'] = t
        evolution.append(metrics)

    results['evolution'] = evolution

    # Extract time series
    results['times'] = times
    results['xi_sq_t'] = np.array([e['xi_sq'] for e in evolution])
    results['xi_dB_t'] = np.array([e['xi_dB'] for e in evolution])
    results['var_min_t'] = np.array([e['var_perp_min'] for e in evolution])
    results['var_max_t'] = np.array([e['var_perp_max'] for e in evolution])
    results['J_mean_t'] = np.array([e['J_mean'] for e in evolution])

    # Find optimal squeezing (exclude t=0 and look for actual squeezing)
    # The minimum should occur at finite time due to OAT dynamics
    xi_sq_nonzero = results['xi_sq_t'].copy()
    xi_sq_nonzero[0] = 2  # Exclude t=0 from search
    idx_opt = np.argmin(xi_sq_nonzero)
    if idx_opt == 0:
        # If still at 0, find first local minimum after t>0
        for i in range(2, len(xi_sq_nonzero)):
            if xi_sq_nonzero[i-1] < xi_sq_nonzero[i] and xi_sq_nonzero[i-1] < xi_sq_nonzero[i-2]:
                idx_opt = i - 1
                break
    results['t_opt'] = times[idx_opt]
    results['xi_opt'] = results['xi_sq_t'][idx_opt]
    results['xi_opt_dB'] = results['xi_dB_t'][idx_opt]

    print(f"Optimal squeezing at t = {results['t_opt']:.3f}")
    print(f"Squeezing: xi^2 = {results['xi_opt']:.3f} ({results['xi_opt_dB']:.1f} dB)")

    # 2. Scaling with N
    print("\nComputing scaling with atom number...")
    N_values = np.array([10, 20, 50, 100, 200])
    xi_opt_scaling = []

    for N_i in N_values:
        if N_i <= 100:  # Full calculation for small N
            Jx_i, Jy_i, Jz_i, _ = collective_spin_operators(N_i)
            psi_i = coherent_spin_state(N_i, np.pi / 2, 0)

            # Optimal time scales as N^(-2/3)
            t_opt_i = (1 / N_i)**(2/3) * results['t_opt'] * N**(2/3)

            U_i = one_axis_twisting(Jz_i, chi, t_opt_i)
            psi_squeezed = U_i @ psi_i
            metrics_i = calculate_squeezing(psi_squeezed, Jx_i, Jy_i, Jz_i, N_i)
            xi_opt_scaling.append(metrics_i['xi_sq'])
        else:
            # Theoretical scaling: xi^2_opt ~ N^(-2/3)
            xi_opt_scaling.append(N_i**(-2/3) * N**(2/3) * results['xi_opt'])

    results['scaling'] = {
        'N': N_values,
        'xi_sq': np.array(xi_opt_scaling),
        'SQL': np.ones_like(N_values)  # SQL is xi^2 = 1
    }

    # 3. Wigner function visualization data
    print("Computing Wigner function representation...")
    # Sample points on Bloch sphere for squeezed state
    n_theta = 50
    n_phi = 100
    theta_grid = np.linspace(0, np.pi, n_theta)
    phi_grid = np.linspace(0, 2 * np.pi, n_phi)

    # Get squeezed state at optimal time
    U_opt = one_axis_twisting(Jz, chi, results['t_opt'])
    psi_squeezed = U_opt @ psi_0

    results['squeezed_state'] = psi_squeezed
    results['operators'] = {'Jx': Jx, 'Jy': Jy, 'Jz': Jz}

    return results


def plot_results(results):
    """Create comprehensive visualization of spin squeezing."""

    fig = plt.figure(figsize=(14, 12))
    N = results['N']

    # Plot 1: Squeezing parameter vs time
    ax1 = fig.add_subplot(2, 2, 1)

    ax1.semilogy(results['times'], results['xi_sq_t'], 'b-', linewidth=2,
                label=r'$\xi^2$ (squeezing)')
    ax1.axhline(y=1, color='gray', linestyle='--', linewidth=1.5,
               label='SQL ($\\xi^2 = 1$)')

    # Mark optimal squeezing
    ax1.plot(results['t_opt'], results['xi_opt'], 'ro', markersize=10,
            label=f'Optimal: {results["xi_opt_dB"]:.1f} dB')

    ax1.set_xlabel('Evolution time $\\chi t$', fontsize=11)
    ax1.set_ylabel('Squeezing parameter $\\xi^2$', fontsize=11)
    ax1.set_title(f'One-Axis Twisting Dynamics (N = {N} atoms)', fontsize=12)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_xlim(0, results['times'][-1])
    ax1.set_ylim(0.01, 10)
    ax1.grid(True, alpha=0.3, which='both')

    # Plot 2: Variance dynamics
    ax2 = fig.add_subplot(2, 2, 2)

    ax2.semilogy(results['times'], results['var_min_t'], 'b-', linewidth=2,
                label=r'$\min(\Delta J_\perp^2)$')
    ax2.semilogy(results['times'], results['var_max_t'], 'r-', linewidth=2,
                label=r'$\max(\Delta J_\perp^2)$')
    ax2.axhline(y=results['initial']['var_SQL'], color='gray', linestyle='--',
               linewidth=1.5, label='SQL variance')

    ax2.set_xlabel('Evolution time $\\chi t$', fontsize=11)
    ax2.set_ylabel('Variance', fontsize=11)
    ax2.set_title('Variance Squeezing and Anti-Squeezing', fontsize=12)
    ax2.legend(loc='right', fontsize=9)
    ax2.set_xlim(0, results['times'][-1])
    ax2.grid(True, alpha=0.3, which='both')

    # Add uncertainty product
    product = results['var_min_t'] * results['var_max_t']
    ax2_twin = ax2.twinx()
    ax2_twin.plot(results['times'], product / results['initial']['var_SQL']**2,
                 'g:', linewidth=1.5, alpha=0.7, label='Uncertainty product')
    ax2_twin.set_ylabel('Normalized product', fontsize=10, color='g')
    ax2_twin.tick_params(axis='y', labelcolor='g')

    # Plot 3: Scaling with N
    ax3 = fig.add_subplot(2, 2, 3)
    sc = results['scaling']

    ax3.loglog(sc['N'], sc['xi_sq'], 'bo-', markersize=8, linewidth=2,
              label=r'$\xi^2_{opt}$')
    ax3.loglog(sc['N'], sc['SQL'], 'r--', linewidth=2, label='SQL')

    # Theoretical scaling
    N_theory = np.logspace(1, 2.5, 50)
    xi_theory = 0.5 * N_theory**(-2/3)  # Approximate OAT scaling
    ax3.loglog(N_theory, xi_theory, 'g:', linewidth=1.5, alpha=0.7,
              label=r'$\propto N^{-2/3}$')

    ax3.set_xlabel('Number of atoms N', fontsize=11)
    ax3.set_ylabel('Optimal squeezing $\\xi^2$', fontsize=11)
    ax3.set_title('Scaling of Optimal Squeezing', fontsize=12)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3, which='both')

    # Add text about Heisenberg limit
    ax3.text(0.05, 0.05,
            'OAT scaling: $\\xi^2 \\propto N^{-2/3}$\nHeisenberg limit: $\\xi^2 \\propto N^{-1}$',
            transform=ax3.transAxes, fontsize=9,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 4: Mean spin length decay
    ax4 = fig.add_subplot(2, 2, 4)

    ax4.plot(results['times'], results['J_mean_t'] / results['j'], 'b-', linewidth=2)
    ax4.axhline(y=1, color='gray', linestyle='--', alpha=0.5)

    ax4.set_xlabel('Evolution time $\\chi t$', fontsize=11)
    ax4.set_ylabel(r'Mean spin $|\langle \mathbf{J} \rangle| / j$', fontsize=11)
    ax4.set_title('Mean Spin Length During Squeezing', fontsize=12)
    ax4.set_xlim(0, results['times'][-1])
    ax4.set_ylim(0, 1.1)
    ax4.grid(True, alpha=0.3)

    # Add annotation about trade-off
    ax4.annotate('Squeezing reduces\nmean spin length',
                xy=(results['t_opt'], results['J_mean_t'][np.argmin(np.abs(results['times']-results['t_opt']))] / results['j']),
                xytext=(results['t_opt'] + 0.3, 0.7),
                fontsize=9,
                arrowprops=dict(arrowstyle='->', color='red'))

    # Add text about applications
    textstr = '\n'.join([
        'Applications:',
        '- Atomic clocks: improve stability',
        '- Magnetometry: beyond SQL',
        '- Gravitational wave detection',
        '- Quantum information'
    ])
    ax4.text(0.95, 0.05, textstr, transform=ax4.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig


def main():
    print("=" * 60)
    print("Experiment 253: Spin Squeezing Model")
    print("=" * 60)
    print()

    # Run simulation
    print("Running simulations...")
    results = simulate_spin_squeezing()

    # Create visualization
    print("\nCreating visualization...")
    fig = plot_results(results)

    # Save figure
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'spin_squeezing.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"Number of atoms: {results['N']}")
    print(f"Initial state: Coherent spin state along +x")
    print()
    print("Squeezing results (one-axis twisting):")
    print(f"  Optimal time: chi*t = {results['t_opt']:.3f}")
    print(f"  Optimal xi^2: {results['xi_opt']:.3f}")
    print(f"  Squeezing: {results['xi_opt_dB']:.1f} dB below SQL")
    print()
    print("Key concepts:")
    print("  - SQL: Phase sensitivity delta_theta = 1/sqrt(N)")
    print("  - Squeezed: delta_theta < 1/sqrt(N)")
    print("  - Heisenberg limit: delta_theta = 1/N")
    print()
    print("One-axis twisting scaling:")
    print("  - xi^2_opt ~ N^(-2/3)")
    print("  - Better than SQL, but not Heisenberg-limited")
    print("  - Two-axis twisting can reach N^(-1) scaling")

    plt.close()


if __name__ == "__main__":
    main()
