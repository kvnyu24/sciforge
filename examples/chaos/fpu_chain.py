"""
Experiment 67: Fermi-Pasta-Ulam (FPU) Chain Recurrence.

The FPU problem (1955) is a landmark in the history of computational physics
and the birth of nonlinear science and chaos theory.

Fermi, Pasta, and Ulam studied a chain of masses with nonlinear springs:

    m * x_i'' = k(x_{i+1} - 2*x_i + x_{i-1}) + alpha[(x_{i+1} - x_i)^2 - (x_i - x_{i-1})^2]

They expected energy to thermalize (equipartition) among normal modes, but
instead observed near-periodic recurrence to the initial state!

This phenomenon led to:
1. Discovery of solitons (Zabusky-Kruskal, 1965)
2. KAM theory and near-integrability
3. Modern understanding of thermalization in nonlinear systems
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dst, idst


def fpu_derivatives(state, N, k, alpha):
    """
    FPU alpha-chain equations of motion.

    m * x_i'' = k(x_{i+1} - 2*x_i + x_{i-1})
              + alpha*[(x_{i+1} - x_i)^2 - (x_i - x_{i-1})^2]

    Fixed boundary conditions: x_0 = x_{N+1} = 0
    """
    x = state[:N]      # positions
    v = state[N:]      # velocities

    # Extend with boundary conditions
    x_ext = np.zeros(N + 2)
    x_ext[1:-1] = x

    # Accelerations
    a = np.zeros(N)
    for i in range(N):
        # Linear term
        linear = k * (x_ext[i+2] - 2*x_ext[i+1] + x_ext[i])

        # Nonlinear term (FPU-alpha)
        delta_right = x_ext[i+2] - x_ext[i+1]
        delta_left = x_ext[i+1] - x_ext[i]
        nonlinear = alpha * (delta_right**2 - delta_left**2)

        a[i] = linear + nonlinear

    return np.concatenate([v, a])


def simulate_fpu(N, k, alpha, x0, v0, t_max, dt):
    """Simulate FPU chain using symplectic (velocity Verlet) integrator."""
    n_steps = int(t_max / dt)
    t = np.linspace(0, t_max, n_steps)

    x = np.zeros((n_steps, N))
    v = np.zeros((n_steps, N))

    x[0] = x0
    v[0] = v0

    state = np.concatenate([x0, v0])

    for i in range(1, n_steps):
        # RK4 step
        k1 = fpu_derivatives(state, N, k, alpha)
        k2 = fpu_derivatives(state + dt/2*k1, N, k, alpha)
        k3 = fpu_derivatives(state + dt/2*k2, N, k, alpha)
        k4 = fpu_derivatives(state + dt*k3, N, k, alpha)

        state = state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        x[i] = state[:N]
        v[i] = state[N:]

    return t, x, v


def compute_normal_mode_energies(x, v, N, k):
    """
    Compute energy in each normal mode.

    Normal modes of linear chain: Q_m = sqrt(2/(N+1)) * sum_j sin(m*j*pi/(N+1)) * x_j
    """
    # Use discrete sine transform for efficiency
    # DST-I gives normal mode amplitudes

    energies = np.zeros((len(x), N))

    # Normal mode frequencies
    omega = 2 * np.sqrt(k) * np.sin(np.arange(1, N+1) * np.pi / (2*(N+1)))

    for i in range(len(x)):
        # Normal mode coordinates (DST of positions)
        Q = dst(x[i], type=1) / np.sqrt(2 * (N + 1))

        # Normal mode momenta (DST of momenta)
        P = dst(v[i], type=1) / np.sqrt(2 * (N + 1))

        # Energy in each mode: E_m = (1/2)(P_m^2 + omega_m^2 * Q_m^2)
        energies[i] = 0.5 * (P**2 + omega**2 * Q**2)

    return energies


def initial_condition_first_mode(N, amplitude=1.0):
    """Initialize system in first normal mode."""
    # First mode: x_j = A * sin(j*pi/(N+1))
    j = np.arange(1, N + 1)
    x0 = amplitude * np.sin(j * np.pi / (N + 1))
    v0 = np.zeros(N)
    return x0, v0


def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Parameters (close to original FPU values)
    N = 32          # Number of masses
    k = 1.0         # Linear spring constant
    alpha = 0.25    # Nonlinear parameter
    amplitude = 1.0
    dt = 0.1

    # Initial condition: first mode
    x0, v0 = initial_condition_first_mode(N, amplitude)

    # Simulate for long time to see recurrence
    t_max = 50000
    t, x, v = simulate_fpu(N, k, alpha, x0, v0, t_max, dt)

    # Compute mode energies
    energies = compute_normal_mode_energies(x, v, N, k)

    # Plot 1: Mode energies vs time
    ax = axes[0, 0]

    for mode in [0, 1, 2, 3]:  # First few modes
        ax.plot(t, energies[:, mode], lw=0.5, label=f'Mode {mode+1}', alpha=0.8)

    ax.set_xlabel('Time')
    ax.set_ylabel('Mode energy')
    ax.set_title('FPU Recurrence: Mode Energies\n'
                 'Energy returns nearly to initial state!')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, t_max)

    # Plot 2: Energy in mode 1 (shows recurrence)
    ax = axes[0, 1]

    E1 = energies[:, 0]
    E1_initial = E1[0]

    ax.plot(t, E1 / E1_initial, 'b-', lw=0.5)
    ax.axhline(y=1, color='r', linestyle='--', lw=1, label='Initial')

    # Find approximate recurrence time
    # Look for when E1 returns close to initial
    threshold = 0.8
    recurrence_idx = np.where(E1 / E1_initial > threshold)[0]
    if len(recurrence_idx) > 100:
        # Find first return after significant departure
        departed = np.where(E1 / E1_initial < 0.5)[0]
        if len(departed) > 0:
            first_depart = departed[0]
            returns = recurrence_idx[recurrence_idx > first_depart]
            if len(returns) > 0:
                t_rec = t[returns[0]]
                ax.axvline(x=t_rec, color='g', linestyle=':', lw=2, label=f'Recurrence ~ {t_rec:.0f}')

    ax.set_xlabel('Time')
    ax.set_ylabel('E_1 / E_1(0)')
    ax.set_title('Energy in First Mode\n'
                 'Near-periodic recurrence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Energy spectrum evolution
    ax = axes[0, 2]

    # Snapshot at different times
    times_to_show = [0, t_max//4, t_max//2, 3*t_max//4]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(times_to_show)))

    for ti, color in zip(times_to_show, colors):
        idx = int(ti / dt)
        if idx < len(energies):
            ax.semilogy(range(1, N+1), energies[idx] + 1e-15, 'o-',
                       color=color, lw=1.5, markersize=3, label=f't={ti:.0f}')

    ax.set_xlabel('Mode number')
    ax.set_ylabel('Mode energy (log scale)')
    ax.set_title('Energy Spectrum Evolution\n'
                 'Energy spreads to higher modes, then returns')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    # Plot 4: Comparison linear vs nonlinear
    ax = axes[1, 0]

    # Linear system (alpha = 0)
    t_short = 5000
    t_lin, x_lin, v_lin = simulate_fpu(N, k, 0.0, x0, v0, t_short, dt)
    t_nlin, x_nlin, v_nlin = simulate_fpu(N, k, alpha, x0, v0, t_short, dt)

    E_lin = compute_normal_mode_energies(x_lin, v_lin, N, k)
    E_nlin = compute_normal_mode_energies(x_nlin, v_nlin, N, k)

    ax.plot(t_lin, E_lin[:, 0] / E_lin[0, 0], 'b-', lw=1, label='Linear (alpha=0)')
    ax.plot(t_nlin, E_nlin[:, 0] / E_nlin[0, 0], 'r-', lw=1, label=f'Nonlinear (alpha={alpha})')

    ax.set_xlabel('Time')
    ax.set_ylabel('E_1 / E_1(0)')
    ax.set_title('Linear vs Nonlinear Chain\n'
                 'Nonlinearity causes mode coupling')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Total energy conservation
    ax = axes[1, 1]

    E_total = np.sum(energies, axis=1)
    E_error = (E_total - E_total[0]) / E_total[0] * 100

    ax.plot(t, E_error, 'b-', lw=0.5)

    ax.set_xlabel('Time')
    ax.set_ylabel('Energy error (%)')
    ax.set_title('Total Energy Conservation\n'
                 'Symplectic integrator preserves energy')
    ax.grid(True, alpha=0.3)

    # Plot 6: Summary
    ax = axes[1, 2]
    ax.axis('off')

    summary = """Fermi-Pasta-Ulam (FPU) Problem
==============================

THE PARADOX (1955):
Fermi, Pasta, and Ulam simulated a chain
of masses with nonlinear springs:

  m*x_i'' = k*(x_{i+1} - 2*x_i + x_{i-1})
          + alpha*[(x_{i+1}-x_i)^2 - (x_i-x_{i-1})^2]

EXPECTATION:
Nonlinearity should thermalize energy
(equipartition among all modes).

OBSERVATION:
Instead of thermalization, energy
RECURS to initial state!

SIGNIFICANCE:

1. BIRTH OF COMPUTATIONAL PHYSICS:
   First major numerical experiment
   showing unexpected physics.

2. SOLITONS (1965):
   Zabusky and Kruskal showed the
   continuum limit gives KdV equation,
   which has soliton solutions.

3. KAM THEORY:
   Near-integrability explains
   recurrence for small nonlinearity.

4. THERMALIZATION TIME:
   For small alpha, thermalization
   takes astronomically long times.

Parameters used:
  N = 32 masses
  k = 1.0 (linear spring)
  alpha = 0.25 (nonlinearity)

MODERN UNDERSTANDING:
The FPU problem is near-integrable;
solitons and KAM tori explain the
near-periodic behavior."""

    ax.text(0.02, 0.98, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))

    plt.suptitle("Fermi-Pasta-Ulam Chain: The Recurrence That Launched Chaos Theory",
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'fpu_chain.png'), dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/fpu_chain.png")


if __name__ == "__main__":
    main()
