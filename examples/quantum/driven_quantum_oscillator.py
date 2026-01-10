"""
Experiment 158: Driven Quantum Oscillator

This experiment demonstrates the driven quantum harmonic oscillator, including:
- Response to periodic driving force
- Resonance behavior
- Transition rates between states
- Floquet theory basics
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
from scipy.integrate import solve_ivp
from scipy.linalg import expm


def create_oscillator_matrices(n_max: int, m: float = 1.0, omega: float = 1.0,
                                hbar: float = 1.0) -> tuple:
    """
    Create Hamiltonian and position operator matrices for truncated Hilbert space.

    Args:
        n_max: Maximum quantum number (matrix dimension)
        m: Particle mass
        omega: Angular frequency
        hbar: Reduced Planck constant

    Returns:
        Tuple of (H0, x_op, a, a_dagger) matrices
    """
    # Creation and annihilation operators
    a = np.zeros((n_max, n_max), dtype=complex)
    for n in range(n_max - 1):
        a[n, n+1] = np.sqrt(n + 1)

    a_dagger = a.T.conj()

    # Position operator: x = sqrt(hbar/(2*m*omega)) * (a + a^dagger)
    x0 = np.sqrt(hbar / (2 * m * omega))
    x_op = x0 * (a + a_dagger)

    # Unperturbed Hamiltonian
    H0 = np.diag([hbar * omega * (n + 0.5) for n in range(n_max)])

    return H0, x_op, a, a_dagger


def time_evolution_driven(H0: np.ndarray, x_op: np.ndarray, F0: float,
                          omega_d: float, t_span: tuple, psi0: np.ndarray,
                          hbar: float = 1.0) -> tuple:
    """
    Evolve quantum state under driven oscillator Hamiltonian.

    H(t) = H0 - F0 * x * cos(omega_d * t)

    Args:
        H0: Unperturbed Hamiltonian
        x_op: Position operator
        F0: Driving force amplitude
        omega_d: Driving frequency
        t_span: (t_start, t_end)
        psi0: Initial state vector
        hbar: Reduced Planck constant

    Returns:
        Tuple of (times, states)
    """
    n_max = len(psi0)

    def schrodinger_rhs(t, psi_flat):
        psi = psi_flat[:n_max] + 1j * psi_flat[n_max:]
        H_t = H0 - F0 * x_op * np.cos(omega_d * t)
        dpsi_dt = -1j / hbar * H_t @ psi
        return np.concatenate([np.real(dpsi_dt), np.imag(dpsi_dt)])

    psi0_flat = np.concatenate([np.real(psi0), np.imag(psi0)])

    sol = solve_ivp(schrodinger_rhs, t_span, psi0_flat,
                    method='RK45', dense_output=True,
                    max_step=0.01 * 2*np.pi/omega_d)

    return sol.t, sol.y[:n_max] + 1j * sol.y[n_max:]


def perturbation_transition_rate(n_initial: int, n_final: int, F0: float,
                                  omega: float, omega_d: float, t: float,
                                  m: float = 1.0, hbar: float = 1.0) -> float:
    """
    Calculate transition probability using first-order perturbation theory.

    For n_final = n_initial + 1 (absorption) or n_initial - 1 (emission).
    """
    if n_final == n_initial + 1:
        # Absorption
        x_matrix_element = np.sqrt(hbar * (n_initial + 1) / (2 * m * omega))
        delta_omega = omega_d - omega
    elif n_final == n_initial - 1 and n_initial > 0:
        # Emission
        x_matrix_element = np.sqrt(hbar * n_initial / (2 * m * omega))
        delta_omega = omega_d + omega
    else:
        return 0.0

    # Transition amplitude from first-order perturbation theory
    if np.abs(delta_omega) < 1e-10:
        # On resonance
        amplitude = F0 * x_matrix_element * t / (2 * hbar)
    else:
        amplitude = F0 * x_matrix_element * np.sin(delta_omega * t / 2) / (hbar * delta_omega)

    return np.abs(amplitude)**2


def main():
    # Parameters (natural units)
    m = 1.0
    omega = 1.0
    hbar = 1.0
    n_max = 20  # Truncation of Hilbert space

    # Create operators
    H0, x_op, a, a_dagger = create_oscillator_matrices(n_max, m, omega, hbar)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Plot 1: Resonant vs off-resonant driving
    ax1 = axes[0, 0]

    F0 = 0.1 * hbar * omega  # Weak driving
    T = 2 * np.pi / omega

    # Start in ground state
    psi0 = np.zeros(n_max, dtype=complex)
    psi0[0] = 1.0

    # On resonance
    omega_d_res = omega
    t_span = (0, 20*T)
    t_res, psi_res = time_evolution_driven(H0, x_op, F0, omega_d_res, t_span, psi0, hbar)

    # Off resonance
    omega_d_off = 1.5 * omega
    t_off, psi_off = time_evolution_driven(H0, x_op, F0, omega_d_off, t_span, psi0, hbar)

    # Ground state population
    P0_res = np.abs(psi_res[0, :])**2
    P0_off = np.abs(psi_off[0, :])**2

    ax1.plot(t_res / T, P0_res, 'b-', lw=2, label=f'On resonance (omega_d = omega)')
    ax1.plot(t_off / T, P0_off, 'r-', lw=2, label=f'Off resonance (omega_d = 1.5*omega)')

    ax1.set_xlabel('Time t / T')
    ax1.set_ylabel('Ground State Population |c_0|^2')
    ax1.set_title('Ground State Depletion')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)

    # Plot 2: Transition to n=1 state
    ax2 = axes[0, 1]

    P1_res = np.abs(psi_res[1, :])**2
    P1_off = np.abs(psi_off[1, :])**2

    ax2.plot(t_res / T, P1_res, 'b-', lw=2, label='On resonance')
    ax2.plot(t_off / T, P1_off, 'r-', lw=2, label='Off resonance')

    # Perturbation theory prediction (on resonance)
    P1_pert = perturbation_transition_rate(0, 1, F0, omega, omega, t_res, m, hbar)
    ax2.plot(t_res / T, P1_pert, 'g--', lw=2, alpha=0.7, label='Perturbation theory')

    ax2.set_xlabel('Time t / T')
    ax2.set_ylabel('First Excited State Population |c_1|^2')
    ax2.set_title('Excitation to n=1')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Frequency scan (resonance curve)
    ax3 = axes[0, 2]

    omega_d_range = np.linspace(0.5, 1.5, 50) * omega
    t_fixed = 10 * T
    P1_final = []

    for omega_d in omega_d_range:
        psi0 = np.zeros(n_max, dtype=complex)
        psi0[0] = 1.0
        t, psi = time_evolution_driven(H0, x_op, F0, omega_d, (0, t_fixed), psi0, hbar)
        P1_final.append(np.abs(psi[1, -1])**2)

    ax3.plot(omega_d_range / omega, P1_final, 'b-', lw=2)
    ax3.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='omega_d = omega')

    ax3.set_xlabel('Driving Frequency omega_d / omega')
    ax3.set_ylabel('P(n=1) at t = 10T')
    ax3.set_title('Resonance Curve')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Strong driving - multi-photon transitions
    ax4 = axes[1, 0]

    F0_strong = 0.5 * hbar * omega  # Stronger driving
    t_span = (0, 30*T)
    psi0 = np.zeros(n_max, dtype=complex)
    psi0[0] = 1.0

    t_strong, psi_strong = time_evolution_driven(H0, x_op, F0_strong, omega, t_span, psi0, hbar)

    # Population in several states
    for n in range(5):
        Pn = np.abs(psi_strong[n, :])**2
        ax4.plot(t_strong / T, Pn, lw=1.5, label=f'n={n}')

    ax4.set_xlabel('Time t / T')
    ax4.set_ylabel('Population |c_n|^2')
    ax4.set_title(f'Strong Driving (F0 = {F0_strong:.1f})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Mean energy vs time
    ax5 = axes[1, 1]

    # Energy for weak driving
    E_res = np.real(np.sum(np.conj(psi_res) * (H0 @ psi_res), axis=0))
    E_off = np.real(np.sum(np.conj(psi_off) * (H0 @ psi_off), axis=0))

    ax5.plot(t_res / T, E_res / (hbar * omega), 'b-', lw=2, label='On resonance')
    ax5.plot(t_off / T, E_off / (hbar * omega), 'r-', lw=2, label='Off resonance')

    ax5.set_xlabel('Time t / T')
    ax5.set_ylabel('Mean Energy <E> / (hbar*omega)')
    ax5.set_title('Energy Absorption')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Floquet spectrum (simplified)
    ax6 = axes[1, 2]

    # Calculate quasi-energies from Floquet theory
    # Using one-period propagator
    dt_floquet = 0.001 * T
    n_steps = int(T / dt_floquet)

    # Propagator over one period
    U = np.eye(n_max, dtype=complex)

    for step in range(n_steps):
        t = step * dt_floquet
        H_t = H0 - F0 * x_op * np.cos(omega * t)
        U = expm(-1j * H_t * dt_floquet / hbar) @ U

    # Floquet quasi-energies
    eigenvalues = np.linalg.eigvals(U)
    quasi_energies = -np.angle(eigenvalues) * hbar / T

    # Sort and unwrap
    quasi_energies = np.sort(np.real(quasi_energies))

    # Plot in first Brillouin zone
    bz_half = hbar * omega / 2
    quasi_energies_bz = np.mod(quasi_energies + bz_half, hbar * omega) - bz_half

    ax6.plot(range(n_max), quasi_energies_bz / (hbar * omega), 'bo', markersize=8,
             label='Floquet quasi-energies')

    # Compare with unperturbed energies (mod hbar*omega)
    E_unpert = np.array([hbar * omega * (n + 0.5) for n in range(n_max)])
    E_unpert_bz = np.mod(E_unpert + bz_half, hbar * omega) - bz_half
    ax6.plot(range(n_max), E_unpert_bz / (hbar * omega), 'r+', markersize=10,
             label='Unperturbed (mod omega)')

    ax6.set_xlabel('State index')
    ax6.set_ylabel('Quasi-energy / (hbar*omega)')
    ax6.set_title('Floquet Spectrum (First Brillouin Zone)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax6.axhline(y=-0.5, color='gray', linestyle=':', alpha=0.5)

    plt.suptitle('Driven Quantum Harmonic Oscillator\n'
                 r'$H(t) = H_0 - F_0 x \cos(\omega_d t)$',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'driven_quantum_oscillator.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'driven_quantum_oscillator.png')}")


if __name__ == "__main__":
    main()
