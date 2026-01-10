"""
Experiment 259: Two-Stream Instability

Demonstrates the two-stream instability that occurs when two
electron populations counter-stream through each other.

Physical concepts:
- Counter-streaming electrons create unstable modes
- Growth rate depends on beam velocities and densities
- Maximum growth at k ~ omega_p / v_beam
- Important in beam-plasma interactions and astrophysics
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Physical constants
e = 1.602e-19
m_e = 9.109e-31
epsilon_0 = 8.854e-12


def two_stream_dispersion(omega, k, n1, n2, v1, v2):
    """
    Dispersion relation for two-stream instability.

    D(omega, k) = 1 - omega_p1^2/(omega - k*v1)^2 - omega_p2^2/(omega - k*v2)^2 = 0
    """
    omega_p1 = np.sqrt(n1 * e**2 / (epsilon_0 * m_e))
    omega_p2 = np.sqrt(n2 * e**2 / (epsilon_0 * m_e))

    term1 = omega_p1**2 / (omega - k * v1)**2
    term2 = omega_p2**2 / (omega - k * v2)**2

    return 1 - term1 - term2


def solve_dispersion(k, n1, n2, v1, v2, initial_guess=None):
    """Solve for complex frequency at given k."""
    omega_p1 = np.sqrt(n1 * e**2 / (epsilon_0 * m_e))
    omega_p2 = np.sqrt(n2 * e**2 / (epsilon_0 * m_e))

    omega_p = np.sqrt(omega_p1**2 + omega_p2**2)

    if initial_guess is None:
        initial_guess = [omega_p, 0.1 * omega_p]  # [omega_r, gamma]

    def equations(x):
        omega = x[0] + 1j * x[1]
        D = two_stream_dispersion(omega, k, n1, n2, v1, v2)
        return [np.real(D), np.imag(D)]

    try:
        solution = fsolve(equations, initial_guess, full_output=True)
        omega_r, gamma = solution[0]
        return omega_r, gamma
    except Exception:
        return np.nan, np.nan


def cold_beam_growth_rate(k, n1, n2, v1, v2):
    """
    Approximate growth rate for symmetric cold beams.

    gamma_max ~ omega_p * (n_beam / 2*n_total)^(1/3)
    """
    omega_p1 = np.sqrt(n1 * e**2 / (epsilon_0 * m_e))
    omega_p2 = np.sqrt(n2 * e**2 / (epsilon_0 * m_e))

    omega_p = np.sqrt(omega_p1**2 + omega_p2**2)
    v_beam = abs(v1 - v2) / 2

    # Resonance at k = omega_p / v_beam
    k_res = omega_p / v_beam
    delta_k = abs(k - k_res) / k_res

    # Approximate growth rate
    if n1 == n2:
        gamma_max = omega_p * (np.sqrt(3) / 2**(4/3))
    else:
        ratio = min(n1, n2) / (n1 + n2)
        gamma_max = omega_p * ratio**(1/3)

    # Gaussian profile around resonance
    gamma = gamma_max * np.exp(-delta_k**2 / 0.5)

    return gamma


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plasma parameters
    n_total = 1e18  # Total density
    v_beam = 1e7    # Beam velocity (m/s)

    # Plot 1: Growth rate vs k for symmetric beams
    ax1 = axes[0, 0]

    n1 = n_total / 2
    n2 = n_total / 2
    v1 = v_beam
    v2 = -v_beam

    omega_p = np.sqrt(n_total * e**2 / (epsilon_0 * m_e))
    k_res = omega_p / v_beam

    k = np.linspace(0.1 * k_res, 3 * k_res, 100)
    gamma = np.array([cold_beam_growth_rate(ki, n1, n2, v1, v2) for ki in k])

    ax1.plot(k / k_res, gamma / omega_p, 'b-', lw=2)
    ax1.fill_between(k / k_res, 0, gamma / omega_p, alpha=0.3)

    ax1.axvline(x=1.0, color='red', linestyle='--', label='Resonance $k = \\omega_p/v_b$')
    ax1.set_xlabel('$k / k_{res}$')
    ax1.set_ylabel('Growth Rate $\\gamma / \\omega_p$')
    ax1.set_title('Two-Stream Instability Growth Rate (Symmetric Beams)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 3)

    # Plot 2: Effect of beam density ratio
    ax2 = axes[0, 1]

    density_ratios = [0.5, 0.3, 0.1, 0.05]  # n_beam / n_plasma
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(density_ratios)))

    for ratio, color in zip(density_ratios, colors):
        n_beam = n_total * ratio
        n_plasma = n_total * (1 - ratio)

        k = np.linspace(0.1 * k_res, 3 * k_res, 100)
        gamma = np.array([cold_beam_growth_rate(ki, n_beam, n_plasma, v1, v2)
                          for ki in k])

        ax2.plot(k / k_res, gamma / omega_p, color=color, lw=2,
                 label=f'$n_b/n_0$ = {ratio}')

    ax2.set_xlabel('$k / k_{res}$')
    ax2.set_ylabel('Growth Rate $\\gamma / \\omega_p$')
    ax2.set_title('Growth Rate vs Beam Density Ratio')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 3)

    # Plot 3: Instability evolution in time
    ax3 = axes[1, 0]

    n1 = n_total / 2
    n2 = n_total / 2

    gamma_max = omega_p * (np.sqrt(3) / 2**(4/3))

    t = np.linspace(0, 20 / gamma_max, 500)

    # Initial perturbation grows exponentially
    amplitude = 0.01 * np.exp(gamma_max * t)
    amplitude_sat = np.minimum(amplitude, 1.0)  # Saturation at nonlinear level

    # Electric field energy
    E_field = amplitude_sat**2

    ax3.semilogy(t * gamma_max, amplitude, 'b--', lw=1.5, alpha=0.5, label='Linear growth')
    ax3.semilogy(t * gamma_max, amplitude_sat, 'b-', lw=2, label='With saturation')
    ax3.semilogy(t * gamma_max, E_field, 'r-', lw=2, label='Field energy $\\propto E^2$')

    ax3.axhline(y=1.0, color='gray', linestyle=':', alpha=0.7)
    ax3.axvline(x=np.log(100), color='green', linestyle='--', alpha=0.7,
                label='Saturation time')

    ax3.set_xlabel('Time ($\\gamma_{max}^{-1}$)')
    ax3.set_ylabel('Amplitude / Energy')
    ax3.set_title('Instability Time Evolution')
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 20)
    ax3.set_ylim(1e-4, 10)

    # Plot 4: Phase space portrait of instability
    ax4 = axes[1, 1]

    # Show distribution function modification
    v = np.linspace(-2 * v_beam, 2 * v_beam, 500)

    # Initial two-stream distribution
    sigma_v = v_beam / 10
    f_initial = (np.exp(-(v - v_beam)**2 / (2 * sigma_v**2)) +
                 np.exp(-(v + v_beam)**2 / (2 * sigma_v**2)))
    f_initial /= np.max(f_initial)

    # After instability: plateau formation
    v_trap = v_beam * 0.3  # Trapping width
    f_final = f_initial.copy()
    # Create plateau between beams
    plateau_region = np.abs(v) < v_beam
    f_final[plateau_region] = np.maximum(f_final[plateau_region], 0.3)

    ax4.plot(v / v_beam, f_initial, 'b-', lw=2, label='Initial (unstable)')
    ax4.plot(v / v_beam, f_final, 'r--', lw=2, label='After saturation')
    ax4.fill_between(v / v_beam, 0, f_initial, alpha=0.2, color='blue')

    # Mark resonance velocities
    k_mode = k_res
    v_phase = omega_p / k_mode
    ax4.axvline(x=v_phase / v_beam, color='green', linestyle=':', alpha=0.7)
    ax4.axvline(x=-v_phase / v_beam, color='green', linestyle=':', alpha=0.7)
    ax4.text(0.95, 0.8, 'Phase velocity', fontsize=9, color='green')

    ax4.set_xlabel('Velocity $v / v_{beam}$')
    ax4.set_ylabel('Distribution Function f(v)')
    ax4.set_title('Velocity Distribution Evolution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-2, 2)
    ax4.set_ylim(0, 1.2)

    # Add physics explanation
    textstr = ('Two-stream instability:\n'
               '- Resonant particles: $v \\approx \\omega/k$\n'
               '- Extract energy from beams\n'
               '- Saturates by plateau formation')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax4.text(0.98, 0.55, textstr, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.suptitle('Experiment 259: Two-Stream Instability\n'
                 'Counter-streaming beams create growing electrostatic waves',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'two_stream_instability.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'two_stream_instability.png')}")


if __name__ == "__main__":
    main()
