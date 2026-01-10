"""
Experiment 164: Adiabatic Theorem

This experiment demonstrates the quantum adiabatic theorem: a system stays
in its instantaneous eigenstate if the Hamiltonian changes slowly enough.

Physics:
    The adiabatic theorem states that if H(t) changes slowly, a system
    initially in eigenstate |n(0)> remains in the instantaneous eigenstate
    |n(t)> (up to a phase).

    Adiabatic condition:
        |<m(t)|dH/dt|n(t)>| << (E_n - E_m)^2 / hbar

    or equivalently, the characteristic time T of the change must satisfy:
        T >> hbar / (Delta E)^2 * |<m|dH/dt|n>|

    Example: Spin-1/2 in a slowly rotating magnetic field
        H(t) = -gamma * B(t) . S

    If B rotates slowly, the spin follows it adiabatically.
    Fast rotation leads to non-adiabatic transitions (Landau-Zener).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
identity = np.eye(2, dtype=complex)


def rotating_field_hamiltonian(t, omega, B0, theta_max, hbar=1.0):
    """
    Hamiltonian for spin in a rotating magnetic field.

    B(t) = B0 * (sin(theta(t)), 0, cos(theta(t)))
    where theta(t) = theta_max * sin(omega * t)

    H = -(hbar/2) * (B_x*sigma_x + B_z*sigma_z)

    Args:
        t: Time
        omega: Angular frequency of rotation
        B0: Field magnitude (sets energy scale)
        theta_max: Maximum tilt angle
        hbar: Reduced Planck constant

    Returns:
        2x2 Hamiltonian matrix
    """
    theta = theta_max * np.sin(omega * t)
    B_x = B0 * np.sin(theta)
    B_z = B0 * np.cos(theta)

    return -hbar / 2 * (B_x * sigma_x + B_z * sigma_z)


def instantaneous_eigenstate(H):
    """
    Get ground state (lowest eigenvalue) of Hamiltonian.

    Args:
        H: Hamiltonian matrix

    Returns:
        Tuple of (eigenvalue, eigenstate)
    """
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    idx = np.argmin(eigenvalues)
    return eigenvalues[idx], eigenvectors[:, idx]


def solve_schrodinger(H_func, t_span, psi0, hbar=1.0, n_points=1000):
    """
    Solve time-dependent Schrodinger equation.

    i*hbar * d|psi>/dt = H(t)|psi>

    Args:
        H_func: Function returning H(t)
        t_span: (t_start, t_end)
        psi0: Initial state
        hbar: Reduced Planck constant
        n_points: Number of time points

    Returns:
        Tuple of (times, states)
    """
    def schrodinger_rhs(t, y):
        psi = y[:2] + 1j * y[2:]
        H = H_func(t)
        dpsi_dt = -1j / hbar * H @ psi
        return np.concatenate([np.real(dpsi_dt), np.imag(dpsi_dt)])

    y0 = np.concatenate([np.real(psi0), np.imag(psi0)])

    t_eval = np.linspace(t_span[0], t_span[1], n_points)

    sol = solve_ivp(schrodinger_rhs, t_span, y0, method='RK45',
                    t_eval=t_eval, max_step=0.01)

    states = sol.y[:2] + 1j * sol.y[2:]
    return sol.t, states.T


def adiabatic_parameter(omega, B0, theta_max, hbar=1.0):
    """
    Calculate adiabatic parameter.

    The adiabatic condition is Q = hbar*omega*theta_max / (2*B0) << 1

    Args:
        omega: Rotation frequency
        B0: Field strength
        theta_max: Maximum angle
        hbar: Reduced Planck constant

    Returns:
        Adiabatic parameter Q (small = adiabatic)
    """
    # Energy gap is 2 * (hbar/2) * B0 = hbar * B0
    # Rate of change is ~ omega * theta_max * B0
    # Q ~ hbar * omega * theta_max / gap^2 * gap
    return hbar * omega * theta_max / (hbar * B0)


def linear_sweep_hamiltonian(t, T_sweep, Delta_i, Delta_f, Omega, hbar=1.0):
    """
    Hamiltonian for Landau-Zener problem.

    H(t) = (Delta(t)/2) * sigma_z + (Omega/2) * sigma_x

    where Delta(t) = Delta_i + (Delta_f - Delta_i) * t / T_sweep

    Args:
        t: Time
        T_sweep: Total sweep time
        Delta_i: Initial detuning
        Delta_f: Final detuning
        Omega: Coupling strength
        hbar: Reduced Planck constant

    Returns:
        Hamiltonian matrix
    """
    Delta = Delta_i + (Delta_f - Delta_i) * t / T_sweep
    return hbar / 2 * (Delta * sigma_z + Omega * sigma_x)


def landau_zener_probability(v, Omega, hbar=1.0):
    """
    Landau-Zener transition probability.

    P_LZ = exp(-pi * Omega^2 / (2 * hbar * v))

    where v = |d(Delta)/dt| is the sweep rate.

    Adiabatic limit: slow sweep (v -> 0), P_LZ -> 0
    Diabatic limit: fast sweep (v -> inf), P_LZ -> 1

    Args:
        v: Sweep rate
        Omega: Coupling
        hbar: Reduced Planck constant

    Returns:
        Transition probability
    """
    return np.exp(-np.pi * Omega**2 / (2 * hbar * v))


def main():
    hbar = 1.0
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # ===== Plot 1: Spin following rotating field (adiabatic) =====
    ax1 = axes[0, 0]

    B0 = 1.0
    theta_max = np.pi / 4
    omega_slow = 0.1  # Slow rotation (adiabatic)

    T = 2 * np.pi / omega_slow * 2  # Two periods

    def H_slow(t):
        return rotating_field_hamiltonian(t, omega_slow, B0, theta_max, hbar)

    # Start in ground state of H(0)
    _, psi0 = instantaneous_eigenstate(H_slow(0))

    t, states = solve_schrodinger(H_slow, (0, T), psi0, hbar, n_points=500)

    # Calculate overlap with instantaneous ground state
    overlaps = []
    for ti, psi in zip(t, states):
        _, gs = instantaneous_eigenstate(H_slow(ti))
        overlap = np.abs(np.vdot(gs, psi))**2
        overlaps.append(overlap)

    ax1.plot(t * omega_slow / (2*np.pi), overlaps, 'b-', lw=2)
    ax1.axhline(1.0, color='green', linestyle='--', alpha=0.5)

    Q = adiabatic_parameter(omega_slow, B0, theta_max, hbar)
    ax1.set_xlabel('Time (periods)')
    ax1.set_ylabel('|<psi|ground>|^2')
    ax1.set_title(f'Adiabatic Evolution (omega = {omega_slow})\nQ = {Q:.3f} << 1 (adiabatic)')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.9, 1.01)

    # ===== Plot 2: Non-adiabatic case (fast rotation) =====
    ax2 = axes[0, 1]

    omega_fast = 2.0  # Fast rotation (non-adiabatic)
    T_fast = 2 * np.pi / omega_fast * 2

    def H_fast(t):
        return rotating_field_hamiltonian(t, omega_fast, B0, theta_max, hbar)

    _, psi0_fast = instantaneous_eigenstate(H_fast(0))
    t_fast, states_fast = solve_schrodinger(H_fast, (0, T_fast), psi0_fast, hbar, n_points=500)

    overlaps_fast = []
    for ti, psi in zip(t_fast, states_fast):
        _, gs = instantaneous_eigenstate(H_fast(ti))
        overlap = np.abs(np.vdot(gs, psi))**2
        overlaps_fast.append(overlap)

    ax2.plot(t_fast * omega_fast / (2*np.pi), overlaps_fast, 'r-', lw=2)
    ax2.axhline(1.0, color='green', linestyle='--', alpha=0.5)

    Q_fast = adiabatic_parameter(omega_fast, B0, theta_max, hbar)
    ax2.set_xlabel('Time (periods)')
    ax2.set_ylabel('|<psi|ground>|^2')
    ax2.set_title(f'Non-Adiabatic Evolution (omega = {omega_fast})\nQ = {Q_fast:.3f} ~ 1 (transitions)')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)

    # ===== Plot 3: Transition probability vs. omega =====
    ax3 = axes[0, 2]

    omega_range = np.linspace(0.01, 3, 50)
    final_overlaps = []

    for omega in omega_range:
        def H_omega(t):
            return rotating_field_hamiltonian(t, omega, B0, theta_max, hbar)

        T_omega = 2 * np.pi / omega  # One period
        _, psi0_omega = instantaneous_eigenstate(H_omega(0))
        _, states_omega = solve_schrodinger(H_omega, (0, T_omega), psi0_omega, hbar, n_points=200)

        _, gs = instantaneous_eigenstate(H_omega(T_omega))
        final_overlap = np.abs(np.vdot(gs, states_omega[-1]))**2
        final_overlaps.append(final_overlap)

    ax3.plot(omega_range, final_overlaps, 'b-', lw=2)
    ax3.axhline(1.0, color='green', linestyle='--', alpha=0.5, label='Adiabatic limit')
    ax3.axvline(B0, color='red', linestyle=':', alpha=0.5, label='omega = B0')

    ax3.set_xlabel('Rotation frequency omega')
    ax3.set_ylabel('Final ground state overlap')
    ax3.set_title('Adiabaticity vs. Rotation Speed\n(One period evolution)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # ===== Plot 4: Landau-Zener transition =====
    ax4 = axes[1, 0]

    Omega_LZ = 0.5  # Coupling
    Delta_range = 5.0
    T_sweeps = [50, 20, 10, 5, 2]  # Different sweep times
    colors_lz = plt.cm.viridis(np.linspace(0.1, 0.9, len(T_sweeps)))

    for T_sweep, color in zip(T_sweeps, colors_lz):
        def H_LZ(t):
            return linear_sweep_hamiltonian(t, T_sweep, -Delta_range, Delta_range, Omega_LZ, hbar)

        # Start in ground state (high-field limit)
        _, psi0_lz = instantaneous_eigenstate(H_LZ(0))

        t_lz, states_lz = solve_schrodinger(H_LZ, (0, T_sweep), psi0_lz, hbar, n_points=300)

        # Ground state population
        populations = []
        for ti, psi in zip(t_lz, states_lz):
            _, gs = instantaneous_eigenstate(H_LZ(ti))
            populations.append(np.abs(np.vdot(gs, psi))**2)

        v = 2 * Delta_range / T_sweep
        P_LZ = landau_zener_probability(v, Omega_LZ, hbar)

        ax4.plot(t_lz / T_sweep, populations, color=color, lw=2,
                label=f'T={T_sweep}, P_LZ={P_LZ:.3f}')

    ax4.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Time (normalized)')
    ax4.set_ylabel('Ground state population')
    ax4.set_title('Landau-Zener Transitions\n(Linear sweep through avoided crossing)')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ===== Plot 5: Landau-Zener probability =====
    ax5 = axes[1, 1]

    v_range = np.linspace(0.01, 2, 100)
    Omega_values = [0.2, 0.5, 1.0]
    colors_om = plt.cm.plasma(np.linspace(0.2, 0.8, len(Omega_values)))

    for Omega, color in zip(Omega_values, colors_om):
        P_LZ = landau_zener_probability(v_range, Omega, hbar)
        # Adiabatic: stay in ground state -> P_diabatic = P_LZ
        # So P_adiabatic = 1 - P_LZ
        ax5.plot(v_range, 1 - P_LZ, color=color, lw=2, label=f'Omega = {Omega}')

    ax5.set_xlabel('Sweep rate v = |dDelta/dt|')
    ax5.set_ylabel('Adiabatic transition probability')
    ax5.set_title('Landau-Zener Formula\n$P_{adiab} = 1 - \\exp(-\\pi\\Omega^2/2v)$')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, 1.1)

    # ===== Plot 6: Energy level diagram =====
    ax6 = axes[1, 2]

    Delta_plot = np.linspace(-3, 3, 100)
    Omega_plot = 0.5

    # Eigenvalues
    E_plus = np.sqrt(Delta_plot**2 + Omega_plot**2) / 2
    E_minus = -E_plus

    # Diabatic levels (without coupling)
    E_diab_1 = Delta_plot / 2
    E_diab_2 = -Delta_plot / 2

    ax6.plot(Delta_plot, E_plus, 'b-', lw=2, label='Adiabatic +')
    ax6.plot(Delta_plot, E_minus, 'r-', lw=2, label='Adiabatic -')
    ax6.plot(Delta_plot, E_diab_1, 'b--', lw=1, alpha=0.5, label='Diabatic 1')
    ax6.plot(Delta_plot, E_diab_2, 'r--', lw=1, alpha=0.5, label='Diabatic 2')

    # Mark avoided crossing
    ax6.annotate('', xy=(0, Omega_plot/2), xytext=(0, -Omega_plot/2),
                arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    ax6.text(0.3, 0, f'Gap = {Omega_plot}', color='green', fontsize=10)

    ax6.set_xlabel('Detuning Delta')
    ax6.set_ylabel('Energy')
    ax6.set_title('Avoided Crossing\n(Adiabatic states avoid, diabatic states cross)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.suptitle('Quantum Adiabatic Theorem\n'
                 'System stays in instantaneous eigenstate if change is slow enough',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'adiabatic_theorem.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'adiabatic_theorem.png')}")

    # Print numerical results
    print("\n=== Adiabatic Theorem Results ===")

    print("\n1. Adiabatic parameter Q = hbar*omega*theta / gap:")
    print(f"   Slow case (omega={omega_slow}): Q = {Q:.4f} << 1 (adiabatic)")
    print(f"   Fast case (omega={omega_fast}): Q = {Q_fast:.4f} ~ 1 (non-adiabatic)")

    print("\n2. Landau-Zener transition probabilities:")
    for T_sweep in T_sweeps:
        v = 2 * Delta_range / T_sweep
        P_LZ = landau_zener_probability(v, Omega_LZ, hbar)
        print(f"   T_sweep = {T_sweep:4.1f}: v = {v:.3f}, P_diabatic = {P_LZ:.4f}")

    print("\n3. Adiabatic condition:")
    print("   |<m|dH/dt|n>| << (E_n - E_m)^2 / hbar")
    print("   Slow changes relative to energy gaps ensure adiabaticity")


if __name__ == "__main__":
    main()
