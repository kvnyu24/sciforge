"""
Experiment 155: Wavepacket Scattering

This experiment demonstrates the scattering of a Gaussian wavepacket from
various potential barriers, including:
- Time evolution of scattered wavepacket
- Splitting into reflected and transmitted parts
- Energy dependence of scattering
- Comparison with stationary state results
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq


def gaussian_wavepacket(x: np.ndarray, x0: float, sigma: float, k0: float) -> np.ndarray:
    """Create a normalized Gaussian wavepacket."""
    norm = (2 * np.pi * sigma**2)**(-0.25)
    return norm * np.exp(-(x - x0)**2 / (4 * sigma**2)) * np.exp(1j * k0 * x)


def split_operator_step(psi: np.ndarray, V: np.ndarray, k: np.ndarray,
                        dt: float, m: float = 1.0, hbar: float = 1.0) -> np.ndarray:
    """
    One step of split-operator method for time evolution.

    Args:
        psi: Wavefunction
        V: Potential array
        k: Wave number array
        dt: Time step
        m: Particle mass
        hbar: Reduced Planck constant

    Returns:
        Evolved wavefunction
    """
    # Half step in position space (potential)
    psi = psi * np.exp(-1j * V * dt / (2 * hbar))

    # Full step in momentum space (kinetic)
    psi_k = fft(psi)
    T_k = hbar**2 * k**2 / (2 * m)
    psi_k = psi_k * np.exp(-1j * T_k * dt / hbar)
    psi = ifft(psi_k)

    # Half step in position space (potential)
    psi = psi * np.exp(-1j * V * dt / (2 * hbar))

    return psi


def evolve_wavepacket(psi: np.ndarray, V: np.ndarray, x: np.ndarray,
                       t_total: float, n_steps: int,
                       m: float = 1.0, hbar: float = 1.0) -> list:
    """
    Evolve wavepacket using split-operator method.

    Args:
        psi: Initial wavefunction
        V: Potential array
        x: Position array
        t_total: Total evolution time
        n_steps: Number of time steps
        m: Particle mass
        hbar: Reduced Planck constant

    Returns:
        List of (time, wavefunction) tuples at selected times
    """
    dx = x[1] - x[0]
    N = len(x)
    dt = t_total / n_steps

    # Wave number array
    k = 2 * np.pi * fftfreq(N, dx)

    snapshots = [(0, psi.copy())]
    n_save = min(10, n_steps)
    save_interval = n_steps // n_save

    for step in range(1, n_steps + 1):
        psi = split_operator_step(psi, V, k, dt, m, hbar)
        if step % save_interval == 0:
            snapshots.append((step * dt, psi.copy()))

    return snapshots


def rectangular_barrier(x: np.ndarray, x_center: float, width: float, height: float) -> np.ndarray:
    """Create rectangular barrier potential."""
    V = np.zeros_like(x)
    V[np.abs(x - x_center) < width / 2] = height
    return V


def gaussian_barrier(x: np.ndarray, x_center: float, sigma: float, height: float) -> np.ndarray:
    """Create Gaussian barrier potential."""
    return height * np.exp(-(x - x_center)**2 / (2 * sigma**2))


def calculate_transmission_reflection(psi: np.ndarray, x: np.ndarray,
                                       barrier_position: float) -> tuple:
    """Calculate transmission and reflection probabilities."""
    dx = x[1] - x[0]
    prob = np.abs(psi)**2

    # Probability in transmitted region (x > barrier)
    T = np.sum(prob[x > barrier_position]) * dx

    # Probability in reflected region (x < barrier)
    R = np.sum(prob[x < barrier_position]) * dx

    return R, T


def stationary_transmission(E: float, V0: float, a: float,
                            m: float = 1.0, hbar: float = 1.0) -> float:
    """
    Transmission coefficient from stationary state analysis.
    """
    if E <= 0:
        return 0.0
    if E >= V0:
        k = np.sqrt(2 * m * E) / hbar
        k_prime = np.sqrt(2 * m * (E - V0)) / hbar
        sin_term = np.sin(k_prime * a)
        T = 1 / (1 + (k**2 + k_prime**2)**2 * sin_term**2 / (4 * k**2 * k_prime**2))
    else:
        k = np.sqrt(2 * m * E) / hbar
        kappa = np.sqrt(2 * m * (V0 - E)) / hbar
        if kappa * a > 50:
            return 0.0
        sinh_term = np.sinh(kappa * a)
        T = 1 / (1 + (k**2 + kappa**2)**2 * sinh_term**2 / (4 * k**2 * kappa**2))
    return T


def main():
    # Parameters (natural units)
    m = 1.0
    hbar = 1.0

    # Spatial grid
    L = 100.0
    N = 2048
    x = np.linspace(-L/2, L/2, N)
    dx = x[1] - x[0]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Plot 1: Wavepacket scattering snapshots
    ax1 = axes[0, 0]

    # Initial wavepacket
    x0 = -15.0
    sigma = 2.0
    k0 = 3.0  # Mean momentum
    E_mean = hbar**2 * k0**2 / (2 * m)

    # Barrier
    V0 = 1.5 * E_mean  # Above mean energy for partial transmission
    barrier_width = 2.0
    barrier_center = 0.0

    V = rectangular_barrier(x, barrier_center, barrier_width, V0)
    psi0 = gaussian_wavepacket(x, x0, sigma, k0)

    # Evolve
    t_total = 15.0
    n_steps = 3000
    snapshots = evolve_wavepacket(psi0, V, x, t_total, n_steps, m, hbar)

    # Plot snapshots
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(snapshots)))

    for (t, psi), color in zip(snapshots, colors):
        prob = np.abs(psi)**2
        ax1.plot(x, prob, color=color, lw=1.5, alpha=0.8, label=f't = {t:.1f}')

    # Draw barrier
    ax1.fill_between(x, 0, V / max(V) * max(np.abs(psi0)**2) * 0.5,
                     alpha=0.3, color='gray')
    ax1.axvline(x=barrier_center, color='black', linestyle='--', alpha=0.3)

    ax1.set_xlabel('Position x')
    ax1.set_ylabel('|psi(x,t)|^2')
    ax1.set_title(f'Wavepacket Scattering (E_mean = {E_mean:.2f}, V0 = {V0:.2f})')
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-40, 40)

    # Plot 2: R and T vs time
    ax2 = axes[0, 1]

    R_t = []
    T_t = []
    times = []

    for t, psi in snapshots:
        R, T = calculate_transmission_reflection(psi, x, barrier_center)
        R_t.append(R)
        T_t.append(T)
        times.append(t)

    ax2.plot(times, R_t, 'b-', lw=2, label='Reflection R')
    ax2.plot(times, T_t, 'r-', lw=2, label='Transmission T')
    ax2.plot(times, np.array(R_t) + np.array(T_t), 'k--', lw=1, label='R + T')

    ax2.set_xlabel('Time t')
    ax2.set_ylabel('Probability')
    ax2.set_title('R and T vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)

    # Plot 3: Comparison with different barrier heights
    ax3 = axes[0, 2]

    V0_values = [0.5 * E_mean, 1.0 * E_mean, 1.5 * E_mean, 2.0 * E_mean]
    colors_V0 = plt.cm.plasma(np.linspace(0.2, 0.9, len(V0_values)))

    final_T = []
    final_R = []

    for V0_test, color in zip(V0_values, colors_V0):
        V_test = rectangular_barrier(x, barrier_center, barrier_width, V0_test)
        psi0_test = gaussian_wavepacket(x, x0, sigma, k0)
        snapshots_test = evolve_wavepacket(psi0_test, V_test, x, t_total, n_steps, m, hbar)

        # Final state
        _, psi_final = snapshots_test[-1]
        R, T = calculate_transmission_reflection(psi_final, x, barrier_center)
        final_T.append(T)
        final_R.append(R)

        # Plot final probability
        prob_final = np.abs(psi_final)**2
        ax3.plot(x, prob_final, color=color, lw=1.5,
                label=f'V0 = {V0_test/E_mean:.1f} E (T={T:.2f})')

    ax3.set_xlabel('Position x')
    ax3.set_ylabel('|psi(x,t_final)|^2')
    ax3.set_title('Final State for Different Barrier Heights')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-40, 40)

    # Plot 4: Energy distribution of wavepacket
    ax4 = axes[1, 0]

    # Momentum space representation
    k_grid = 2 * np.pi * fftfreq(N, dx)
    k_sorted_idx = np.argsort(k_grid)
    k_sorted = k_grid[k_sorted_idx]

    psi0_k = fft(psi0)
    prob_k = np.abs(psi0_k[k_sorted_idx])**2
    prob_k = prob_k / np.max(prob_k)

    # Energy distribution
    E_k = hbar**2 * k_sorted**2 / (2 * m)

    ax4.plot(k_sorted, prob_k, 'b-', lw=2, label='|phi(k)|^2')
    ax4.axvline(x=k0, color='red', linestyle='--', alpha=0.7, label=f'k0 = {k0}')

    # Show energy range
    ax4_twin = ax4.twinx()
    ax4_twin.plot(k_sorted, E_k, 'g--', lw=1, alpha=0.5)
    ax4_twin.set_ylabel('Energy E(k)', color='green')

    ax4.set_xlabel('Wave number k')
    ax4.set_ylabel('|phi(k)|^2 (normalized)')
    ax4.set_title('Momentum Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-5, 10)

    # Plot 5: Wavepacket vs stationary state transmission
    ax5 = axes[1, 1]

    # Calculate stationary state transmission weighted by momentum distribution
    T_stationary_weighted = 0
    prob_k_full = np.abs(psi0_k)**2
    norm_k = np.sum(prob_k_full)

    for i, ki in enumerate(k_grid):
        if ki > 0:  # Only right-moving components
            E_i = hbar**2 * ki**2 / (2 * m)
            T_i = stationary_transmission(E_i, V0, barrier_width, m, hbar)
            T_stationary_weighted += prob_k_full[i] * T_i

    T_stationary_weighted /= norm_k

    # Compare with wavepacket result
    ax5.bar(['Wavepacket\n(numerical)', 'Stationary\n(weighted avg)'],
            [final_T[2], T_stationary_weighted],  # Use V0 = 1.5 E_mean case
            color=['steelblue', 'coral'], alpha=0.7)

    ax5.set_ylabel('Transmission Probability')
    ax5.set_title('Wavepacket vs Stationary State Prediction')
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_ylim(0, 1)

    for i, v in enumerate([final_T[2], T_stationary_weighted]):
        ax5.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)

    # Plot 6: Gaussian vs rectangular barrier
    ax6 = axes[1, 2]

    # Gaussian barrier with same effective area
    sigma_barrier = barrier_width / (2 * np.sqrt(2 * np.log(2)))  # FWHM = barrier_width
    V_gauss = gaussian_barrier(x, barrier_center, sigma_barrier, V0_values[2])
    V_rect = rectangular_barrier(x, barrier_center, barrier_width, V0_values[2])

    # Evolve through both
    psi0_test = gaussian_wavepacket(x, x0, sigma, k0)
    snapshots_gauss = evolve_wavepacket(psi0_test, V_gauss, x, t_total, n_steps, m, hbar)
    psi0_test = gaussian_wavepacket(x, x0, sigma, k0)
    snapshots_rect = evolve_wavepacket(psi0_test, V_rect, x, t_total, n_steps, m, hbar)

    _, psi_final_gauss = snapshots_gauss[-1]
    _, psi_final_rect = snapshots_rect[-1]

    ax6.plot(x, np.abs(psi_final_gauss)**2, 'b-', lw=2, label='Gaussian barrier')
    ax6.plot(x, np.abs(psi_final_rect)**2, 'r--', lw=2, label='Rectangular barrier')

    # Show barriers (scaled for visibility)
    scale = max(np.abs(psi_final_gauss)**2) * 0.5 / max(V_gauss)
    ax6.fill_between(x, 0, V_gauss * scale, alpha=0.2, color='blue')
    ax6.fill_between(x, 0, V_rect * scale, alpha=0.2, color='red')

    R_g, T_g = calculate_transmission_reflection(psi_final_gauss, x, barrier_center)
    R_r, T_r = calculate_transmission_reflection(psi_final_rect, x, barrier_center)

    ax6.set_xlabel('Position x')
    ax6.set_ylabel('|psi(x,t_final)|^2')
    ax6.set_title(f'Barrier Shape Comparison\n(T_gauss={T_g:.3f}, T_rect={T_r:.3f})')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(-40, 40)

    plt.suptitle('Wavepacket Scattering from Potential Barriers\n'
                 'Split-Operator FFT Method',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'wavepacket_scattering.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'wavepacket_scattering.png')}")


if __name__ == "__main__":
    main()
