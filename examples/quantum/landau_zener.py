"""
Experiment 165: Landau-Zener Transition

This experiment demonstrates the Landau-Zener transition - the dynamics of a
two-level system with time-varying energy splitting, including:
- Adiabatic vs diabatic transitions
- Landau-Zener formula for transition probability
- Effect of sweep rate
- Multiple crossings and interference
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def landau_zener_hamiltonian(t: float, v: float, Delta: float,
                              hbar: float = 1.0) -> np.ndarray:
    """
    Create Landau-Zener Hamiltonian.

    H(t) = [[v*t, Delta], [Delta, -v*t]]

    Energy levels: E_+/- = +/- sqrt((v*t)^2 + Delta^2)

    Args:
        t: Time
        v: Sweep rate (d(E1-E2)/dt = 2v)
        Delta: Coupling strength (half gap at crossing)
        hbar: Reduced Planck constant

    Returns:
        2x2 Hamiltonian matrix
    """
    return hbar * np.array([[v * t, Delta], [Delta, -v * t]], dtype=complex)


def landau_zener_probability(v: float, Delta: float, hbar: float = 1.0) -> float:
    """
    Landau-Zener transition probability (diabatic transition).

    P_LZ = exp(-2*pi*Delta^2 / (hbar * v))

    This is the probability to stay in the diabatic state (not follow adiabatic).

    Args:
        v: Sweep rate
        Delta: Coupling strength
        hbar: Reduced Planck constant

    Returns:
        Transition probability
    """
    return np.exp(-2 * np.pi * Delta**2 / (hbar * np.abs(v)))


def solve_landau_zener(t_span: tuple, v: float, Delta: float, psi0: np.ndarray,
                        hbar: float = 1.0) -> tuple:
    """
    Numerically solve Landau-Zener dynamics.

    Args:
        t_span: (t_initial, t_final)
        v: Sweep rate
        Delta: Coupling strength
        psi0: Initial state [c1, c2]
        hbar: Reduced Planck constant

    Returns:
        Tuple of (times, states)
    """
    def schrodinger_rhs(t, y):
        psi = y[:2] + 1j * y[2:]
        H = landau_zener_hamiltonian(t, v, Delta, hbar)
        dpsi_dt = -1j / hbar * H @ psi
        return np.concatenate([np.real(dpsi_dt), np.imag(dpsi_dt)])

    y0 = np.concatenate([np.real(psi0), np.imag(psi0)])

    sol = solve_ivp(schrodinger_rhs, t_span, y0, method='RK45',
                    dense_output=True, max_step=0.001 / np.sqrt(v + 0.01))

    psi = sol.y[:2] + 1j * sol.y[2:]

    return sol.t, psi


def adiabatic_states(t: float, v: float, Delta: float,
                     hbar: float = 1.0) -> tuple:
    """
    Calculate instantaneous adiabatic eigenstates.

    Returns:
        Tuple of (eigenvalues, eigenvectors)
    """
    H = landau_zener_hamiltonian(t, v, Delta, hbar)
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    return eigenvalues, eigenvectors


def main():
    # Parameters
    hbar = 1.0
    Delta = 1.0  # Coupling strength

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Plot 1: Energy level diagram
    ax1 = axes[0, 0]

    v = 1.0
    t_range = np.linspace(-5, 5, 200)

    # Diabatic energies
    E_diabatic_1 = hbar * v * t_range
    E_diabatic_2 = -hbar * v * t_range

    # Adiabatic energies
    E_adiabatic_plus = np.sqrt((hbar * v * t_range)**2 + (hbar * Delta)**2)
    E_adiabatic_minus = -E_adiabatic_plus

    ax1.plot(t_range, E_diabatic_1, 'b--', lw=1.5, alpha=0.5, label='Diabatic 1')
    ax1.plot(t_range, E_diabatic_2, 'r--', lw=1.5, alpha=0.5, label='Diabatic 2')
    ax1.plot(t_range, E_adiabatic_plus, 'k-', lw=2, label='Adiabatic +')
    ax1.plot(t_range, E_adiabatic_minus, 'k-', lw=2, label='Adiabatic -')

    # Mark gap at crossing
    ax1.annotate('', xy=(0, hbar*Delta), xytext=(0, -hbar*Delta),
                 arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    ax1.text(0.3, 0, f'2*Delta = {2*Delta}', color='green', fontsize=10)

    ax1.set_xlabel('Time t')
    ax1.set_ylabel('Energy')
    ax1.set_title('Landau-Zener Energy Levels')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Dynamics for different sweep rates
    ax2 = axes[0, 1]

    v_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(v_values)))

    # Start in diabatic state 1 (lower energy at t < 0)
    psi0 = np.array([1.0, 0.0], dtype=complex)
    t_span = (-10, 10)

    for v, color in zip(v_values, colors):
        t, psi = solve_landau_zener(t_span, v, Delta, psi0, hbar)
        P1 = np.abs(psi[0, :])**2  # Probability to stay in diabatic state 1

        P_LZ = landau_zener_probability(v, Delta, hbar)
        ax2.plot(t, P1, color=color, lw=2,
                label=f'v = {v}, P_LZ = {P_LZ:.3f}')

    ax2.set_xlabel('Time t')
    ax2.set_ylabel('Probability in Diabatic State 1')
    ax2.set_title('Landau-Zener Dynamics')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Plot 3: LZ formula verification
    ax3 = axes[0, 2]

    v_range = np.logspace(-2, 1, 50)
    P_LZ_theory = [landau_zener_probability(v, Delta, hbar) for v in v_range]

    # Numerical results
    P_LZ_numerical = []
    for v in v_range:
        t, psi = solve_landau_zener((-20/np.sqrt(v+0.01), 20/np.sqrt(v+0.01)),
                                     v, Delta, psi0, hbar)
        P_LZ_numerical.append(np.abs(psi[0, -1])**2)

    ax3.semilogx(v_range, P_LZ_theory, 'b-', lw=2, label='LZ formula')
    ax3.semilogx(v_range, P_LZ_numerical, 'ro', markersize=5, alpha=0.7,
                label='Numerical')

    ax3.set_xlabel('Sweep Rate v')
    ax3.set_ylabel('Diabatic Transition Probability')
    ax3.set_title('Landau-Zener Formula Verification\nP = exp(-2*pi*Delta^2/v)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Mark adiabatic and diabatic limits
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax3.text(0.02, 0.1, 'Adiabatic\n(slow)', fontsize=9)
    ax3.text(5, 0.9, 'Diabatic\n(fast)', fontsize=9)

    # Plot 4: Adiabatic vs diabatic probabilities
    ax4 = axes[1, 0]

    v = 0.5  # Intermediate regime
    t, psi = solve_landau_zener((-15, 15), v, Delta, psi0, hbar)

    # Project onto instantaneous adiabatic states
    P_adiabatic_minus = []
    P_adiabatic_plus = []

    for i, ti in enumerate(t):
        _, eigvecs = adiabatic_states(ti, v, Delta, hbar)
        psi_i = psi[:, i]

        # Overlap with adiabatic states
        P_adiabatic_minus.append(np.abs(np.vdot(eigvecs[:, 0], psi_i))**2)
        P_adiabatic_plus.append(np.abs(np.vdot(eigvecs[:, 1], psi_i))**2)

    ax4.plot(t, np.abs(psi[0, :])**2, 'b-', lw=2, label='Diabatic 1')
    ax4.plot(t, np.abs(psi[1, :])**2, 'r-', lw=2, label='Diabatic 2')
    ax4.plot(t, P_adiabatic_minus, 'g--', lw=2, label='Adiabatic -')
    ax4.plot(t, P_adiabatic_plus, 'm--', lw=2, label='Adiabatic +')

    ax4.set_xlabel('Time t')
    ax4.set_ylabel('Probability')
    ax4.set_title(f'Diabatic vs Adiabatic Basis (v = {v})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Multiple crossings (Stuckelberg oscillations)
    ax5 = axes[1, 1]

    # Oscillating detuning: E(t) = A*sin(omega*t)
    # This creates multiple LZ crossings

    A = 5.0  # Amplitude
    omega = 0.5  # Frequency

    def oscillating_hamiltonian(t, hbar=1.0):
        epsilon = A * np.sin(omega * t)
        return hbar * np.array([[epsilon, Delta], [Delta, -epsilon]], dtype=complex)

    def schrodinger_osc(t, y):
        psi = y[:2] + 1j * y[2:]
        H = oscillating_hamiltonian(t, hbar)
        dpsi_dt = -1j / hbar * H @ psi
        return np.concatenate([np.real(dpsi_dt), np.imag(dpsi_dt)])

    t_span_osc = (0, 6 * 2 * np.pi / omega)  # 6 periods
    y0 = np.concatenate([np.real(psi0), np.imag(psi0)])

    sol_osc = solve_ivp(schrodinger_osc, t_span_osc, y0, method='RK45',
                        dense_output=True, max_step=0.01)

    t_osc = sol_osc.t
    psi_osc = sol_osc.y[:2] + 1j * sol_osc.y[2:]

    ax5.plot(t_osc * omega / (2*np.pi), np.abs(psi_osc[0, :])**2, 'b-', lw=2,
             label='P(state 1)')
    ax5.plot(t_osc * omega / (2*np.pi), np.abs(psi_osc[1, :])**2, 'r-', lw=2,
             label='P(state 2)')

    ax5.set_xlabel('Time (in driving periods)')
    ax5.set_ylabel('Probability')
    ax5.set_title('Stuckelberg Oscillations\n(Multiple LZ Crossings)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Phase diagram
    ax6 = axes[1, 2]

    # Adiabaticity parameter: gamma = 2*pi*Delta^2 / (hbar*v)
    # gamma >> 1: adiabatic
    # gamma << 1: diabatic

    v_range_2d = np.logspace(-1, 1, 50)
    Delta_range = np.logspace(-1, 1, 50)
    V, D = np.meshgrid(v_range_2d, Delta_range)

    # Transition probability (diabatic)
    P_diabatic = np.exp(-2 * np.pi * D**2 / (hbar * V))

    im = ax6.contourf(np.log10(V), np.log10(D), P_diabatic, levels=20,
                      cmap='RdBu_r')
    plt.colorbar(im, ax=ax6, label='Diabatic transition probability')

    # Mark gamma = 1 line
    gamma_1_line = np.sqrt(hbar * v_range_2d / (2 * np.pi))
    ax6.plot(np.log10(v_range_2d), np.log10(gamma_1_line), 'k--', lw=2,
             label='gamma = 1')

    ax6.set_xlabel('log10(Sweep Rate v)')
    ax6.set_ylabel('log10(Coupling Delta)')
    ax6.set_title('Landau-Zener Phase Diagram')
    ax6.legend()

    ax6.text(-0.8, 0.5, 'ADIABATIC\n(follow\neigenstates)', fontsize=9,
             ha='center', color='blue')
    ax6.text(0.5, -0.5, 'DIABATIC\n(pass through)', fontsize=9,
             ha='center', color='red')

    plt.suptitle('Landau-Zener Transitions\n'
                 r'$H(t) = v t \sigma_z + \Delta \sigma_x$, '
                 r'$P_{LZ} = e^{-2\pi\Delta^2/v}$',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'landau_zener.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'landau_zener.png')}")


if __name__ == "__main__":
    main()
