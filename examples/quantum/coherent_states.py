"""
Experiment 157: Coherent States Evolution

This experiment demonstrates coherent states of the quantum harmonic oscillator,
including:
- Definition and properties of coherent states
- Time evolution maintaining minimum uncertainty
- Classical-like oscillatory behavior
- Phase space (Wigner function) representation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
from scipy.integrate import trapezoid


def hermite_polynomial(n: int, x: np.ndarray) -> np.ndarray:
    """Calculate physicist's Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x

    H_prev2 = np.ones_like(x)
    H_prev1 = 2 * x

    for k in range(2, n + 1):
        H_curr = 2 * x * H_prev1 - 2 * (k - 1) * H_prev2
        H_prev2 = H_prev1
        H_prev1 = H_curr

    return H_prev1


def sho_eigenstate(n: int, x: np.ndarray, m: float = 1.0, omega: float = 1.0,
                   hbar: float = 1.0) -> np.ndarray:
    """Calculate nth eigenstate of quantum harmonic oscillator."""
    alpha = np.sqrt(m * omega / hbar)
    xi = alpha * x
    norm = (m * omega / (np.pi * hbar))**0.25 / np.sqrt(2**n * factorial(n))
    return norm * hermite_polynomial(n, xi) * np.exp(-xi**2 / 2)


def coherent_state(alpha_param: complex, x: np.ndarray, n_max: int = 50,
                   m: float = 1.0, omega: float = 1.0, hbar: float = 1.0) -> np.ndarray:
    """
    Calculate coherent state |alpha>.

    |alpha> = exp(-|alpha|^2/2) * sum_n (alpha^n / sqrt(n!)) |n>

    Args:
        alpha_param: Complex amplitude
        x: Position array
        n_max: Maximum n for series truncation
        m: Particle mass
        omega: Angular frequency
        hbar: Reduced Planck constant

    Returns:
        Coherent state wavefunction
    """
    psi = np.zeros_like(x, dtype=complex)

    prefactor = np.exp(-np.abs(alpha_param)**2 / 2)

    for n in range(n_max):
        c_n = prefactor * alpha_param**n / np.sqrt(factorial(n))
        psi_n = sho_eigenstate(n, x, m, omega, hbar)
        psi += c_n * psi_n

    return psi


def coherent_state_analytical(alpha_param: complex, x: np.ndarray, t: float = 0,
                               m: float = 1.0, omega: float = 1.0,
                               hbar: float = 1.0) -> np.ndarray:
    """
    Analytical form of coherent state (Gaussian).

    psi(x, t) = exp(-(x - x_cl(t))^2 / (2*x0^2)) * exp(i * phase)

    where x_cl(t) = sqrt(2) * x0 * Re(alpha * exp(-i*omega*t))
    """
    x0 = np.sqrt(hbar / (m * omega))

    # Time-evolved alpha
    alpha_t = alpha_param * np.exp(-1j * omega * t)

    # Classical position and momentum
    x_cl = np.sqrt(2) * x0 * np.real(alpha_t)
    p_cl = np.sqrt(2) * (hbar / x0) * np.imag(alpha_t)

    # Gaussian wavepacket centered at classical position
    norm = (m * omega / (np.pi * hbar))**0.25

    # Phase includes momentum kick and dynamical phase
    phase = (p_cl * x / hbar - omega * t / 2 -
             np.abs(alpha_param)**2 * np.sin(omega * t) / 2)

    psi = norm * np.exp(-(x - x_cl)**2 / (2 * x0**2)) * np.exp(1j * phase)

    return psi


def wigner_function(psi: np.ndarray, x: np.ndarray, p: np.ndarray,
                    hbar: float = 1.0) -> np.ndarray:
    """
    Calculate Wigner quasi-probability distribution.

    W(x, p) = (1/pi*hbar) * integral psi*(x+y) psi(x-y) exp(2ipy/hbar) dy

    Args:
        psi: Wavefunction
        x: Position grid for W
        p: Momentum grid for W
        hbar: Reduced Planck constant

    Returns:
        Wigner function W(x, p)
    """
    dx_psi = x[1] - x[0]
    N_psi = len(psi)
    x_psi = x

    W = np.zeros((len(p), len(x)))

    for i, xi in enumerate(x):
        for j, pj in enumerate(p):
            # Integration variable
            y_max = min(xi - x_psi[0], x_psi[-1] - xi)
            if y_max <= 0:
                continue

            n_pts = min(100, int(y_max / dx_psi) * 2 + 1)
            y = np.linspace(-y_max, y_max, n_pts)

            # Interpolate psi at x+y and x-y
            psi_plus = np.interp(xi + y, x_psi, psi)
            psi_minus = np.interp(xi - y, x_psi, psi)

            integrand = np.conj(psi_plus) * psi_minus * np.exp(2j * pj * y / hbar)
            W[j, i] = np.real(trapezoid(integrand, y)) / (np.pi * hbar)

    return W


def main():
    # Parameters (natural units)
    m = 1.0
    omega = 1.0
    hbar = 1.0
    x0 = np.sqrt(hbar / (m * omega))  # Characteristic length

    # Spatial grid
    x_max = 8.0
    N = 500
    x = np.linspace(-x_max, x_max, N)
    dx = x[1] - x[0]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Plot 1: Coherent state as superposition of number states
    ax1 = axes[0, 0]

    alpha = 2.0  # Real alpha for simplicity
    n_max = 20

    # Poisson distribution of number states
    n_arr = np.arange(0, n_max)
    P_n = np.exp(-np.abs(alpha)**2) * np.abs(alpha)**(2*n_arr) / factorial(n_arr)

    ax1.bar(n_arr, P_n, color='steelblue', alpha=0.7)
    ax1.axvline(x=np.abs(alpha)**2, color='red', linestyle='--', lw=2,
                label=f'<n> = |alpha|^2 = {np.abs(alpha)**2}')

    ax1.set_xlabel('Number state n')
    ax1.set_ylabel('Probability |c_n|^2')
    ax1.set_title(f'Number State Distribution (alpha = {alpha})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Coherent state wavefunction
    ax2 = axes[0, 1]

    # Compare numerical and analytical
    psi_coherent = coherent_state(alpha, x, n_max=50, m=m, omega=omega, hbar=hbar)
    psi_analytical = coherent_state_analytical(alpha, x, t=0, m=m, omega=omega, hbar=hbar)

    prob_num = np.abs(psi_coherent)**2
    prob_ana = np.abs(psi_analytical)**2

    ax2.plot(x / x0, prob_num, 'b-', lw=2, label='Numerical (sum)')
    ax2.plot(x / x0, prob_ana, 'r--', lw=2, label='Analytical (Gaussian)')

    # Compare with ground state
    psi_0 = sho_eigenstate(0, x, m, omega, hbar)
    ax2.plot(x / x0, np.abs(psi_0)**2, 'g:', lw=2, label='Ground state n=0')

    ax2.axvline(x=np.sqrt(2) * np.real(alpha), color='orange', linestyle='--',
                alpha=0.7, label='Classical position')

    ax2.set_xlabel('Position x / x0')
    ax2.set_ylabel('|psi|^2')
    ax2.set_title(f'Coherent State Wavefunction (alpha = {alpha})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-6, 8)

    # Plot 3: Time evolution
    ax3 = axes[0, 2]

    alpha = 2.5
    T = 2 * np.pi / omega  # Period
    times = np.linspace(0, T, 9)
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(times)))

    for t, color in zip(times, colors):
        psi_t = coherent_state_analytical(alpha, x, t, m, omega, hbar)
        prob_t = np.abs(psi_t)**2
        ax3.plot(x / x0, prob_t, color=color, lw=1.5, alpha=0.8,
                label=f't = {t/T:.2f}T')

    # Show oscillation range
    x_max_classical = np.sqrt(2) * x0 * alpha
    ax3.axvline(x=x_max_classical / x0, color='black', linestyle=':', alpha=0.5)
    ax3.axvline(x=-x_max_classical / x0, color='black', linestyle=':', alpha=0.5)

    ax3.set_xlabel('Position x / x0')
    ax3.set_ylabel('|psi(x,t)|^2')
    ax3.set_title('Coherent State Time Evolution')
    ax3.legend(fontsize=7, ncol=2)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Classical trajectory comparison
    ax4 = axes[1, 0]

    t_arr = np.linspace(0, 3*T, 200)
    x_expect = []
    p_expect = []
    delta_x = []

    for t in t_arr:
        psi_t = coherent_state_analytical(alpha, x, t, m, omega, hbar)
        prob_t = np.abs(psi_t)**2

        # Expectation values
        x_mean = trapezoid(x * prob_t, x)
        x2_mean = trapezoid(x**2 * prob_t, x)

        x_expect.append(x_mean)
        delta_x.append(np.sqrt(x2_mean - x_mean**2))

    # Classical trajectory
    x_classical = np.sqrt(2) * x0 * alpha * np.cos(omega * t_arr)

    ax4.plot(t_arr / T, np.array(x_expect) / x0, 'b-', lw=2, label='Quantum <x>')
    ax4.plot(t_arr / T, x_classical / x0, 'r--', lw=2, label='Classical x(t)')

    ax4.set_xlabel('Time t / T')
    ax4.set_ylabel('Position x / x0')
    ax4.set_title('Quantum vs Classical Motion')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Minimum uncertainty verification
    ax5 = axes[1, 1]

    # Calculate uncertainties over time
    delta_x_arr = []
    delta_p_arr = []

    for t in t_arr[:50]:  # Fewer points for speed
        psi_t = coherent_state_analytical(alpha, x, t, m, omega, hbar)
        prob_t = np.abs(psi_t)**2

        x_mean = trapezoid(x * prob_t, x)
        x2_mean = trapezoid(x**2 * prob_t, x)
        sigma_x = np.sqrt(x2_mean - x_mean**2)
        delta_x_arr.append(sigma_x)

        # Momentum uncertainty from kinetic energy
        # For coherent state: Delta p = hbar / (2 * Delta x) = p0 / sqrt(2)
        # where p0 = hbar / x0
        delta_p_arr.append(hbar / (2 * sigma_x))

    uncertainty_product = np.array(delta_x_arr) * np.array(delta_p_arr)

    ax5.plot(t_arr[:50] / T, uncertainty_product / (hbar / 2), 'b-', lw=2)
    ax5.axhline(y=1.0, color='red', linestyle='--', lw=2,
                label='Minimum (hbar/2)')

    ax5.set_xlabel('Time t / T')
    ax5.set_ylabel('(Delta x)(Delta p) / (hbar/2)')
    ax5.set_title('Uncertainty Product (Minimum for All Time)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0.9, 1.1)

    # Plot 6: Phase space (simplified Wigner function)
    ax6 = axes[1, 2]

    # Use analytical Gaussian Wigner function for coherent state
    alpha = 2.0
    t = 0

    x_w = np.linspace(-6*x0, 6*x0, 100)
    p_w = np.linspace(-6*hbar/x0, 6*hbar/x0, 100)
    X_W, P_W = np.meshgrid(x_w, p_w)

    # Classical phase space point
    x_cl = np.sqrt(2) * x0 * np.real(alpha * np.exp(-1j * omega * t))
    p_cl = np.sqrt(2) * (hbar / x0) * np.imag(alpha * np.exp(-1j * omega * t))

    # Gaussian Wigner function
    W = (1 / (np.pi * hbar)) * np.exp(
        -((X_W - x_cl)**2 / x0**2 + (P_W - p_cl)**2 * x0**2 / hbar**2)
    )

    im = ax6.contourf(X_W / x0, P_W * x0 / hbar, W * np.pi * hbar, levels=30,
                      cmap='RdBu_r')
    plt.colorbar(im, ax=ax6, label='W(x, p) * pi * hbar')

    # Show classical trajectory circle
    theta = np.linspace(0, 2*np.pi, 100)
    x_circle = np.sqrt(2) * alpha * np.cos(theta)
    p_circle = -np.sqrt(2) * alpha * np.sin(theta)
    ax6.plot(x_circle, p_circle, 'k--', lw=2, label='Classical orbit')

    ax6.set_xlabel('Position x / x0')
    ax6.set_ylabel('Momentum p * x0 / hbar')
    ax6.set_title('Wigner Function (Phase Space)')
    ax6.legend()
    ax6.set_aspect('equal')

    plt.suptitle('Coherent States of the Quantum Harmonic Oscillator\n'
                 r'$|\alpha\rangle = e^{-|\alpha|^2/2} \sum_n \frac{\alpha^n}{\sqrt{n!}}|n\rangle$',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'coherent_states.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'coherent_states.png')}")


if __name__ == "__main__":
    main()
