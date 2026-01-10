"""
Experiment 12: 1D wave equation - finite difference vs spectral (dispersion error).

Compares numerical dispersion in finite difference and spectral methods
for the wave equation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def wave_fd_step(u, u_prev, c, dx, dt):
    """
    Finite difference step for wave equation.

    u_tt = c^2 * u_xx

    Using central differences:
    (u^{n+1} - 2u^n + u^{n-1})/dt^2 = c^2 * (u_{i+1} - 2u_i + u_{i-1})/dx^2
    """
    r = (c * dt / dx)**2
    u_new = np.zeros_like(u)

    # Interior points
    u_new[1:-1] = 2*u[1:-1] - u_prev[1:-1] + r * (u[2:] - 2*u[1:-1] + u[:-2])

    # Periodic boundary conditions
    u_new[0] = 2*u[0] - u_prev[0] + r * (u[1] - 2*u[0] + u[-2])
    u_new[-1] = u_new[0]

    return u_new


def wave_spectral_step(u_hat, u_hat_prev, c, k, dt):
    """
    Spectral method step for wave equation.

    In Fourier space: d^2u_hat/dt^2 = -c^2 * k^2 * u_hat
    This is a simple harmonic oscillator for each mode.
    """
    omega = c * np.abs(k)

    # Exact solution for simple harmonic oscillator
    cos_omega_dt = np.cos(omega * dt)
    sin_omega_dt = np.sin(omega * dt)

    # u_hat(t+dt) = 2*cos(omega*dt)*u_hat(t) - u_hat(t-dt)
    u_hat_new = 2 * cos_omega_dt * u_hat - u_hat_prev

    return u_hat_new


def initial_wavepacket(x, x0=0.25, k0=20, sigma=0.03):
    """Gaussian wave packet initial condition."""
    return np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.cos(k0 * x)


def main():
    # Parameters
    c = 1.0  # Wave speed
    L = 1.0  # Domain length (periodic)
    nx = 128  # Number of points

    x = np.linspace(0, L, nx, endpoint=False)
    dx = x[1] - x[0]

    # Wave numbers for spectral method
    k = 2 * np.pi * np.fft.fftfreq(nx, dx)

    # CFL condition: r = c*dt/dx should be <= 1 for stability
    dt = 0.5 * dx / c  # r = 0.5
    t_final = 2.0  # Enough time for wave to travel around twice
    n_steps = int(t_final / dt)

    # Initial conditions
    u0 = initial_wavepacket(x)
    # For wave equation, also need initial velocity (zero for this example)
    # This means u at t=-dt equals u at t=0 for zero initial velocity

    # Finite difference
    u_fd = u0.copy()
    u_fd_prev = u0.copy()

    # Spectral
    u_hat = np.fft.fft(u0)
    u_hat_prev = u_hat.copy()

    # Storage for snapshots
    times = [0]
    u_fd_history = [u0.copy()]
    u_sp_history = [u0.copy()]

    # Time evolution
    for step in range(n_steps):
        # Finite difference
        u_fd_new = wave_fd_step(u_fd, u_fd_prev, c, dx, dt)
        u_fd_prev = u_fd
        u_fd = u_fd_new

        # Spectral
        u_hat_new = wave_spectral_step(u_hat, u_hat_prev, c, k, dt)
        u_hat_prev = u_hat
        u_hat = u_hat_new

        # Store snapshots
        if (step + 1) % (n_steps // 8) == 0:
            times.append((step + 1) * dt)
            u_fd_history.append(u_fd.copy())
            u_sp_history.append(np.real(np.fft.ifft(u_hat)))

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot snapshots
    n_snapshots = min(6, len(times))
    indices = np.linspace(0, len(times)-1, n_snapshots, dtype=int)

    for idx, i in enumerate(indices[:3]):
        ax = axes[0, idx]
        ax.plot(x, u_fd_history[i], 'b-', lw=1.5, label='Finite Diff')
        ax.plot(x, u_sp_history[i], 'r--', lw=1.5, label='Spectral')
        ax.set_xlabel('x')
        ax.set_ylabel('u(x,t)')
        ax.set_title(f't = {times[i]:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.2, 1.2)

    # Dispersion analysis
    ax = axes[1, 0]

    # Theoretical dispersion relation: omega = c * k
    k_normalized = k * dx
    omega_exact = c * np.abs(k)

    # Numerical dispersion for finite difference
    # omega_numerical = (2/dt) * arcsin(r * sin(k*dx/2))
    r = c * dt / dx
    omega_fd = (2/dt) * np.arcsin(np.clip(r * np.sin(np.abs(k) * dx / 2), -1, 1))

    # Phase velocity
    c_fd = np.where(np.abs(k) > 1e-10, omega_fd / np.abs(k), c)
    c_exact = c * np.ones_like(k)

    valid = (np.abs(k_normalized) < np.pi) & (np.abs(k_normalized) > 0)

    ax.plot(k_normalized[valid], c_fd[valid]/c, 'b-', lw=2, label='Finite Diff')
    ax.axhline(1.0, color='r', linestyle='--', lw=2, label='Exact (Spectral)')
    ax.set_xlabel('k·dx')
    ax.set_ylabel('c_numerical / c_exact')
    ax.set_title('Phase Velocity Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, np.pi)
    ax.set_ylim(0.7, 1.1)

    # Group velocity
    ax = axes[1, 1]

    # dω/dk for finite difference
    c_g_fd = c * r * np.cos(np.abs(k) * dx / 2) / np.sqrt(np.clip(1 - (r * np.sin(np.abs(k) * dx / 2))**2, 1e-10, 1))
    c_g_fd = np.where(np.abs(k_normalized) < np.pi, c_g_fd, 0)

    ax.plot(k_normalized[valid], c_g_fd[valid]/c, 'b-', lw=2, label='Finite Diff')
    ax.axhline(1.0, color='r', linestyle='--', lw=2, label='Exact (Spectral)')
    ax.set_xlabel('k·dx')
    ax.set_ylabel('c_g / c_exact')
    ax.set_title('Group Velocity Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, np.pi)
    ax.set_ylim(0, 1.2)

    # Final comparison
    ax = axes[1, 2]
    ax.plot(x, u_fd_history[-1], 'b-', lw=1.5, label='Finite Diff (final)')
    ax.plot(x, u_sp_history[-1], 'r--', lw=1.5, label='Spectral (final)')
    ax.plot(x, u0, 'k:', lw=1, alpha=0.5, label='Initial')

    ax.set_xlabel('x')
    ax.set_ylabel('u(x,t)')
    ax.set_title(f'Final State (t = {times[-1]:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('1D Wave Equation: Finite Difference vs Spectral Methods\n' +
                 f'c = {c}, nx = {nx}, CFL r = {r:.2f}',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'wave_dispersion.png'), dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/wave_dispersion.png")


if __name__ == "__main__":
    main()
