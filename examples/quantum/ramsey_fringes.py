"""
Experiment 164: Ramsey Fringes

This experiment demonstrates Ramsey interferometry with separated oscillatory
fields, including:
- Two pi/2 pulses separated by free evolution
- Ramsey fringe pattern
- Sensitivity to detuning
- Decay due to decoherence (T2)
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
identity = np.eye(2, dtype=complex)


def rotation_x(theta: float) -> np.ndarray:
    """
    Rotation operator about x-axis.

    R_x(theta) = exp(-i * theta/2 * sigma_x)
    = cos(theta/2) * I - i * sin(theta/2) * sigma_x
    """
    return np.cos(theta/2) * identity - 1j * np.sin(theta/2) * sigma_x


def rotation_z(phi: float) -> np.ndarray:
    """
    Rotation operator about z-axis.

    R_z(phi) = exp(-i * phi/2 * sigma_z)
    = [[exp(-i*phi/2), 0], [0, exp(i*phi/2)]]
    """
    return np.array([[np.exp(-1j*phi/2), 0], [0, np.exp(1j*phi/2)]], dtype=complex)


def free_evolution(delta: float, T: float, hbar: float = 1.0) -> np.ndarray:
    """
    Free evolution operator for detuned system.

    U_free = exp(-i * delta/2 * sigma_z * T)
    """
    return rotation_z(delta * T)


def ramsey_sequence(psi0: np.ndarray, delta: float, T: float,
                    T2: float = np.inf, hbar: float = 1.0) -> np.ndarray:
    """
    Apply Ramsey sequence: pi/2 pulse - wait T - pi/2 pulse.

    Args:
        psi0: Initial state
        delta: Detuning
        T: Free evolution time
        T2: Coherence time (optional decay)
        hbar: Reduced Planck constant

    Returns:
        Final state
    """
    # First pi/2 pulse (rotation about x)
    R_pi2 = rotation_x(np.pi/2)

    # Free evolution
    U_free = free_evolution(delta, T, hbar)

    # Second pi/2 pulse
    R_pi2_2 = rotation_x(np.pi/2)

    # Apply sequence
    psi = R_pi2 @ psi0
    psi = U_free @ psi
    psi = R_pi2_2 @ psi

    # Optional: apply T2 decay to off-diagonal elements
    if np.isfinite(T2):
        # Decay factor for coherence
        decay = np.exp(-T / T2)
        # In density matrix representation, this would decay rho_01
        # For pure state, we approximate by reducing the oscillation amplitude
        # This is a simplified model

    return psi


def ramsey_probability(delta: float, T: float, T2: float = np.inf) -> float:
    """
    Calculate excited state probability after Ramsey sequence.

    P_e = (1/2) * (1 - cos(delta * T)) for ideal case

    With T2 decay:
    P_e = (1/2) * (1 - exp(-T/T2) * cos(delta * T))
    """
    decay = np.exp(-T / T2) if np.isfinite(T2) else 1.0
    return 0.5 * (1 - decay * np.cos(delta * T))


def ramsey_fringe_visibility(T: float, T2: float) -> float:
    """Calculate fringe visibility with T2 decay."""
    return np.exp(-T / T2) if np.isfinite(T2) else 1.0


def main():
    # Parameters
    hbar = 1.0

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Plot 1: Ramsey fringes vs detuning
    ax1 = axes[0, 0]

    T_values = [1.0, 2.0, 5.0]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(T_values)))

    delta_range = np.linspace(-5, 5, 500)

    for T, color in zip(T_values, colors):
        P_e = [ramsey_probability(delta, T) for delta in delta_range]
        ax1.plot(delta_range, P_e, color=color, lw=2, label=f'T = {T}')

    ax1.set_xlabel('Detuning delta')
    ax1.set_ylabel('Excited State Probability')
    ax1.set_title('Ramsey Fringes vs Detuning')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    # Plot 2: Ramsey fringes vs time (fixed detuning)
    ax2 = axes[0, 1]

    delta_values = [0.5, 1.0, 2.0]
    colors_delta = plt.cm.plasma(np.linspace(0.2, 0.9, len(delta_values)))

    T_range = np.linspace(0, 20, 500)

    for delta, color in zip(delta_values, colors_delta):
        P_e = [ramsey_probability(delta, T) for T in T_range]
        ax2.plot(T_range, P_e, color=color, lw=2, label=f'delta = {delta}')

    ax2.set_xlabel('Free Evolution Time T')
    ax2.set_ylabel('Excited State Probability')
    ax2.set_title('Ramsey Fringes vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    # Plot 3: Effect of T2 decay
    ax3 = axes[0, 2]

    T2_values = [5.0, 10.0, 20.0, np.inf]
    colors_T2 = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(T2_values)))

    delta = 1.0

    for T2, color in zip(T2_values, colors_T2):
        P_e = [ramsey_probability(delta, T, T2) for T in T_range]
        label = f'T2 = {T2}' if np.isfinite(T2) else 'T2 = inf'
        ax3.plot(T_range, P_e, color=color, lw=2, label=label)

    ax3.set_xlabel('Free Evolution Time T')
    ax3.set_ylabel('Excited State Probability')
    ax3.set_title(f'T2 Decoherence Effect (delta = {delta})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Bloch sphere visualization
    ax4 = axes[1, 0]

    # Show pulse sequence on Bloch sphere
    delta = 0
    T = 3.0

    psi0 = np.array([1, 0], dtype=complex)  # Ground state

    # After first pi/2 pulse
    psi1 = rotation_x(np.pi/2) @ psi0

    # During free evolution (sample points)
    n_free = 20
    t_free = np.linspace(0, T, n_free)
    bloch_free = []

    for t in t_free:
        psi_t = free_evolution(delta, t) @ psi1
        # Bloch vector
        u = 2 * np.real(np.conj(psi_t[0]) * psi_t[1])
        v = 2 * np.imag(np.conj(psi_t[0]) * psi_t[1])
        w = np.abs(psi_t[0])**2 - np.abs(psi_t[1])**2
        bloch_free.append([u, v, w])

    bloch_free = np.array(bloch_free)

    # After second pi/2 pulse
    psi2 = rotation_x(np.pi/2) @ psi1

    # Plot trajectory
    ax4.plot(t_free, bloch_free[:, 0], 'r-', lw=2, label='<sigma_x>')
    ax4.plot(t_free, bloch_free[:, 1], 'g-', lw=2, label='<sigma_y>')
    ax4.plot(t_free, bloch_free[:, 2], 'b-', lw=2, label='<sigma_z>')

    ax4.axvline(x=0, color='purple', linestyle='--', alpha=0.7)
    ax4.text(0.1, 0.9, 'pi/2 pulse', fontsize=9, color='purple')
    ax4.axvline(x=T, color='purple', linestyle='--', alpha=0.7)
    ax4.text(T + 0.1, 0.9, 'pi/2 pulse', fontsize=9, color='purple')

    ax4.set_xlabel('Time during free evolution')
    ax4.set_ylabel('Bloch Vector Component')
    ax4.set_title('Bloch Vector During Ramsey Sequence\n(on resonance)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Fringe spacing
    ax5 = axes[1, 1]

    # Demonstrate how fringe spacing decreases with T
    T_cases = [1.0, 2.0, 4.0]
    delta_fine = np.linspace(-10, 10, 1000)

    for i, T in enumerate(T_cases):
        P_e = [ramsey_probability(delta, T) for delta in delta_fine]
        ax5.plot(delta_fine, np.array(P_e) + i * 1.2, lw=2,
                label=f'T = {T}, spacing = 2*pi/T = {2*np.pi/T:.2f}')

    ax5.set_xlabel('Detuning delta')
    ax5.set_ylabel('P_e (offset for clarity)')
    ax5.set_title('Fringe Spacing = 2*pi/T')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Frequency measurement precision
    ax6 = axes[1, 2]

    # Standard quantum limit for frequency measurement
    # Delta omega ~ 1 / (sqrt(N) * T)

    T_range_prec = np.linspace(0.1, 20, 100)
    N = 1000  # Number of measurements

    # Precision (inverse of sensitivity)
    precision = 1 / (np.sqrt(N) * T_range_prec)

    # With T2 decay, optimal T ~ T2
    T2 = 10.0
    precision_T2 = 1 / (np.sqrt(N) * T_range_prec * np.exp(-T_range_prec / T2))

    ax6.semilogy(T_range_prec, precision, 'b-', lw=2, label='Ideal (no decay)')
    ax6.semilogy(T_range_prec, precision_T2, 'r-', lw=2, label=f'With T2 = {T2}')

    # Optimal point with T2
    T_opt = T2 / 2  # Approximate optimal
    idx_opt = np.argmin(precision_T2)
    ax6.axvline(x=T_range_prec[idx_opt], color='green', linestyle='--', alpha=0.7)
    ax6.scatter([T_range_prec[idx_opt]], [precision_T2[idx_opt]], s=100, c='green',
               zorder=5, label=f'Optimal T ~ T2/2')

    ax6.set_xlabel('Interrogation Time T')
    ax6.set_ylabel('Frequency Uncertainty (arb.)')
    ax6.set_title('Ramsey Spectroscopy Precision\n(N = 1000 measurements)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.suptitle('Ramsey Interferometry\n'
                 r'Sequence: $\pi/2$ pulse - free evolution T - $\pi/2$ pulse',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'ramsey_fringes.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'ramsey_fringes.png')}")


if __name__ == "__main__":
    main()
