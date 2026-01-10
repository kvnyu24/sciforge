"""
Experiment 68: 1D String Vibration Modes.

Comprehensive study of vibrating string physics:

1. Normal modes and eigenfrequencies
2. Time evolution from initial conditions
3. Fourier analysis of arbitrary plucked strings
4. Effect of damping
5. Comparison of pluck positions

The wave equation: d^2u/dt^2 = c^2 * d^2u/dx^2

with fixed boundary conditions: u(0,t) = u(L,t) = 0
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dst, idst


def mode_shape(x, n, L):
    """
    nth normal mode shape.

    phi_n(x) = sin(n * pi * x / L)
    """
    return np.sin(n * np.pi * x / L)


def mode_frequency(n, L, c):
    """
    nth normal mode frequency.

    f_n = n * c / (2 * L)
    omega_n = n * pi * c / L
    """
    return n * np.pi * c / L


def plucked_string_coefficients(x_pluck, L, n_modes=50):
    """
    Compute Fourier coefficients for a plucked string.

    Initial shape: triangular with peak at x_pluck
    u(x, 0) = (h * x / x_pluck)                    for x <= x_pluck
            = (h * (L - x) / (L - x_pluck))        for x > x_pluck

    Coefficients: a_n = (2/L) * integral u(x,0) * sin(n*pi*x/L) dx
    """
    h = 1.0  # Height of pluck

    a = np.zeros(n_modes + 1)

    for n in range(1, n_modes + 1):
        # Analytical result for triangular pluck
        a[n] = (2 * h * L**2 / (n**2 * np.pi**2 * x_pluck * (L - x_pluck))) * np.sin(n * np.pi * x_pluck / L)

    return a


def struck_string_coefficients(x_strike, L, width, n_modes=50):
    """
    Compute Fourier coefficients for a struck string.

    Initial velocity is localized near x_strike.
    u(x, 0) = 0
    du/dt(x, 0) = v0 * exp(-(x - x_strike)^2 / (2 * width^2))
    """
    v0 = 1.0  # Initial velocity amplitude
    x = np.linspace(0, L, 1000)
    dx = x[1] - x[0]

    v_init = v0 * np.exp(-(x - x_strike)**2 / (2 * width**2))

    b = np.zeros(n_modes + 1)

    for n in range(1, n_modes + 1):
        phi_n = np.sin(n * np.pi * x / L)
        # b_n = (2 / (n * pi * c)) * integral v(x,0) * sin(n*pi*x/L) dx
        # We'll return omega_n * b_n (easier to use)
        b[n] = (2 / L) * np.sum(v_init * phi_n) * dx

    return b


def string_solution(x, t, L, c, a_coeffs, b_coeffs=None, damping=0.0):
    """
    Compute string displacement at positions x and times t.

    u(x, t) = sum_n [a_n * cos(omega_n * t) + b_n * sin(omega_n * t)] * sin(n*pi*x/L)

    With damping: amplitude decays as exp(-damping * omega_n * t)
    """
    if b_coeffs is None:
        b_coeffs = np.zeros_like(a_coeffs)

    u = np.zeros((len(t), len(x)))

    n_modes = len(a_coeffs) - 1

    for n in range(1, n_modes + 1):
        omega_n = n * np.pi * c / L
        phi_n = np.sin(n * np.pi * x / L)

        for i, ti in enumerate(t):
            decay = np.exp(-damping * omega_n * ti)
            time_factor = a_coeffs[n] * np.cos(omega_n * ti) + b_coeffs[n] * np.sin(omega_n * ti)
            u[i] += decay * time_factor * phi_n

    return u


def simulate_wave_equation(L, c, u0, v0, t_max, dx, dt):
    """
    Simulate wave equation using finite differences.

    d^2u/dt^2 = c^2 * d^2u/dx^2

    Uses explicit central differences (FTCS scheme).
    """
    Nx = int(L / dx) + 1
    Nt = int(t_max / dt) + 1

    x = np.linspace(0, L, Nx)
    t = np.linspace(0, t_max, Nt)

    # Courant number
    r = (c * dt / dx)**2

    if r > 1:
        print(f"Warning: CFL condition violated (r = {r:.2f} > 1)")

    u = np.zeros((Nt, Nx))

    # Initial conditions
    u[0] = u0(x)

    # First time step (using initial velocity)
    u[1, 1:-1] = (u[0, 1:-1] + dt * v0(x[1:-1]) +
                  0.5 * r * (u[0, 2:] - 2*u[0, 1:-1] + u[0, :-2]))

    # Time stepping
    for n in range(1, Nt - 1):
        u[n+1, 1:-1] = (2*u[n, 1:-1] - u[n-1, 1:-1] +
                        r * (u[n, 2:] - 2*u[n, 1:-1] + u[n, :-2]))

    return x, t, u


def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # String parameters
    L = 1.0    # Length (m)
    c = 100.0  # Wave speed (m/s)
    n_modes = 20

    # Plot 1: Normal mode shapes
    ax = axes[0, 0]

    x = np.linspace(0, L, 200)

    for n in range(1, 7):
        y = mode_shape(x, n, L)
        f_n = mode_frequency(n, L, c) / (2 * np.pi)
        ax.plot(x, y + 2*(6-n), lw=2, label=f'n={n}, f={f_n:.1f} Hz')

    ax.set_xlabel('Position x (m)')
    ax.set_ylabel('Mode shape (offset)')
    ax.set_title('Normal Modes of Vibrating String\n'
                 'phi_n(x) = sin(n*pi*x/L)')
    ax.legend(fontsize=8, loc='right')
    ax.grid(True, alpha=0.3)

    # Plot 2: Plucked string evolution
    ax = axes[0, 1]

    x_pluck = L / 4  # Pluck at 1/4 length
    a_coeffs = plucked_string_coefficients(x_pluck, L, n_modes)

    # Time evolution
    T1 = 2 * L / c  # Fundamental period
    times = np.linspace(0, T1, 9)

    x = np.linspace(0, L, 200)
    colors = plt.cm.viridis(np.linspace(0, 1, len(times)))

    for ti, color in zip(times, colors):
        u = string_solution(x, np.array([ti]), L, c, a_coeffs)
        ax.plot(x, u[0], color=color, lw=1.5, alpha=0.8)

    ax.plot([x_pluck], [1.0], 'ro', markersize=10, label='Pluck point')
    ax.set_xlabel('Position x (m)')
    ax.set_ylabel('Displacement')
    ax.set_title(f'Plucked String Time Evolution\n'
                 f'Pluck at x = L/4, showing one period')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Mode energy distribution for different pluck positions
    ax = axes[0, 2]

    pluck_positions = [L/8, L/4, L/3, L/2]
    markers = ['o', 's', '^', 'D']

    for x_p, marker in zip(pluck_positions, markers):
        a = plucked_string_coefficients(x_p, L, n_modes)
        energy = a[1:]**2  # Energy proportional to a_n^2

        ax.semilogy(range(1, n_modes+1), energy, f'{marker}-', lw=1.5,
                   markersize=4, label=f'x_pluck = {x_p/L:.2f}L')

    ax.set_xlabel('Mode number n')
    ax.set_ylabel('Relative energy (log scale)')
    ax.set_title('Mode Energy vs Pluck Position\n'
                 'Pluck at node -> mode absent')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    # Plot 4: Effect of damping
    ax = axes[1, 0]

    x = np.linspace(0, L, 200)
    x_pluck = L / 3
    a_coeffs = plucked_string_coefficients(x_pluck, L, n_modes)

    damping_values = [0, 0.01, 0.05, 0.1]
    T1 = 2 * L / c

    t_damp = np.linspace(0, 10*T1, 100)

    for damping in damping_values:
        # Track amplitude at x = L/3
        u = string_solution(np.array([L/3]), t_damp, L, c, a_coeffs, damping=damping)
        ax.plot(t_damp / T1, u[:, 0], lw=1.5, label=f'gamma = {damping}')

    ax.set_xlabel('Time (periods)')
    ax.set_ylabel('Displacement at x = L/3')
    ax.set_title('Effect of Damping\n'
                 'Higher modes damp faster')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Comparison of pluck vs strike
    ax = axes[1, 1]

    x = np.linspace(0, L, 200)

    # Plucked at center
    a_pluck = plucked_string_coefficients(L/2, L, n_modes)
    b_pluck = np.zeros_like(a_pluck)

    # Struck at center
    a_strike = np.zeros(n_modes + 1)
    b_strike = struck_string_coefficients(L/2, L, L/20, n_modes)

    t_show = np.array([0, T1/4, T1/2])

    for i, ti in enumerate(t_show):
        u_pluck = string_solution(x, np.array([ti]), L, c, a_pluck)
        u_strike = string_solution(x, np.array([ti]), L, c, a_strike, b_strike)

        offset = 2 * (2 - i)
        ax.plot(x, u_pluck[0] + offset, 'b-', lw=2, label='Plucked' if i == 0 else '')
        ax.plot(x, u_strike[0] + offset, 'r--', lw=2, label='Struck' if i == 0 else '')
        ax.text(1.02, offset, f't = {ti/T1:.2f}T', transform=ax.get_yaxis_transform())

    ax.set_xlabel('Position x (m)')
    ax.set_ylabel('Displacement (offset)')
    ax.set_title('Plucked vs Struck String\n'
                 'Different harmonic content')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Plot 6: Summary
    ax = axes[1, 2]
    ax.axis('off')

    summary = """1D String Vibration Modes
=========================

WAVE EQUATION:
  d^2u/dt^2 = c^2 * d^2u/dx^2

BOUNDARY CONDITIONS:
  u(0, t) = u(L, t) = 0  (fixed ends)

NORMAL MODES:
  phi_n(x) = sin(n * pi * x / L)
  omega_n = n * pi * c / L
  f_n = n * c / (2 * L)

GENERAL SOLUTION:
  u(x,t) = SUM_n [a_n*cos(omega_n*t)
                + b_n*sin(omega_n*t)]
                * sin(n*pi*x/L)

KEY RESULTS:

1. HARMONICS:
   Frequencies are integer multiples
   of fundamental f_1 = c/(2L)

2. PLUCK POSITION:
   Pluck at x = L/m kills modes n = m, 2m, 3m...
   (pluck at node -> mode not excited)

3. PLUCK vs STRIKE:
   - Pluck: a_n ~ 1/n^2 (softer sound)
   - Strike: b_n ~ 1/n   (brighter sound)

4. DAMPING:
   Higher modes damp faster
   (gamma * omega_n decay rate)

5. WAVE SPEED:
   c = sqrt(T / mu)
   T = tension, mu = mass/length

APPLICATIONS:
- Musical instruments (guitar, violin, piano)
- Vibration engineering
- Seismic waves in structures"""

    ax.text(0.02, 0.98, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    plt.suptitle("1D String Vibration: Normal Modes and Time Evolution",
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'string_vibration_modes.png'), dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/string_vibration_modes.png")


if __name__ == "__main__":
    main()
