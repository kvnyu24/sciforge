"""
Experiment 17: Spectral method - Fourier solution of periodic diffusion.

Demonstrates the spectral method for solving the diffusion equation
with periodic boundary conditions.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def initial_condition(x, mode='multi'):
    """Various initial conditions."""
    if mode == 'gaussian':
        return np.exp(-((x - 0.5) ** 2) / (2 * 0.05**2))
    elif mode == 'multi':
        return np.sin(2*np.pi*x) + 0.5*np.sin(6*np.pi*x) + 0.3*np.cos(10*np.pi*x)
    elif mode == 'step':
        return np.where((x > 0.25) & (x < 0.75), 1.0, 0.0)
    else:
        return np.sin(2*np.pi*x)


def spectral_diffusion_solve(u0, D, t_final, dt, L=1.0):
    """
    Solve diffusion equation using spectral method.

    u_t = D * u_xx

    In Fourier space: du_hat/dt = -D * k^2 * u_hat
    Exact solution: u_hat(k, t) = u_hat(k, 0) * exp(-D * k^2 * t)
    """
    n = len(u0)
    dx = L / n

    # Wave numbers
    k = 2 * np.pi * np.fft.fftfreq(n, dx)

    # Transform to Fourier space
    u_hat0 = np.fft.fft(u0)

    # Time evolution (exact in Fourier space)
    n_steps = int(t_final / dt)
    times = np.linspace(0, t_final, n_steps + 1)

    solutions = []
    spectra = []

    for t in times:
        # Exact solution in Fourier space
        decay = np.exp(-D * k**2 * t)
        u_hat = u_hat0 * decay

        # Transform back
        u = np.real(np.fft.ifft(u_hat))
        solutions.append(u)
        spectra.append(np.abs(u_hat))

    return times, np.array(solutions), np.array(spectra), k


def finite_diff_diffusion(u0, D, t_final, dt, dx):
    """FTCS finite difference for comparison."""
    r = D * dt / dx**2

    if r > 0.5:
        print(f"Warning: r = {r} > 0.5, scheme may be unstable")

    n_steps = int(t_final / dt)
    u = u0.copy()

    for _ in range(n_steps):
        u_new = u.copy()
        u_new[1:-1] = u[1:-1] + r * (u[2:] - 2*u[1:-1] + u[:-2])
        # Periodic BC
        u_new[0] = u[0] + r * (u[1] - 2*u[0] + u[-1])
        u_new[-1] = u_new[0]
        u = u_new

    return u


def main():
    # Parameters
    D = 0.01  # Diffusion coefficient
    L = 1.0
    n = 128

    x = np.linspace(0, L, n, endpoint=False)
    dx = x[1] - x[0]

    dt = 0.001
    t_final = 0.5

    # Initial condition
    u0 = initial_condition(x, 'multi')

    # Spectral solution
    times, solutions, spectra, k = spectral_diffusion_solve(u0, D, t_final, dt, L)

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Time evolution in physical space
    ax = axes[0, 0]
    n_snapshots = 6
    indices = np.linspace(0, len(times)-1, n_snapshots, dtype=int)
    colors = plt.cm.viridis(np.linspace(0, 0.9, n_snapshots))

    for i, idx in enumerate(indices):
        ax.plot(x, solutions[idx], '-', color=colors[i], lw=1.5,
                label=f't = {times[idx]:.3f}')

    ax.set_xlabel('x')
    ax.set_ylabel('u(x,t)')
    ax.set_title('Solution Evolution (Spectral Method)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: Spectrum evolution
    ax = axes[0, 1]

    k_positive = k[k >= 0]
    for i, idx in enumerate(indices):
        spectrum_pos = spectra[idx, k >= 0]
        ax.semilogy(k_positive[:n//4], spectrum_pos[:n//4] + 1e-16, '-',
                    color=colors[i], lw=1.5, label=f't = {times[idx]:.3f}')

    ax.set_xlabel('Wave number k')
    ax.set_ylabel('|û(k)|')
    ax.set_title('Fourier Spectrum Decay')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 3: Mode decay comparison
    ax = axes[0, 2]

    # Track specific modes
    modes = [1, 3, 5, 10]
    mode_indices = [np.argmin(np.abs(k - 2*np.pi*m)) for m in modes]

    for m, idx in zip(modes, mode_indices):
        mode_amplitude = spectra[:, idx]
        # Theoretical decay
        k_m = 2 * np.pi * m
        theory = spectra[0, idx] * np.exp(-D * k_m**2 * times)

        ax.semilogy(times, mode_amplitude, '-', lw=2, label=f'Mode {m} (numerical)')
        ax.semilogy(times, theory, '--', lw=1, alpha=0.7, label=f'Mode {m} (theory)')

    ax.set_xlabel('Time')
    ax.set_ylabel('Mode amplitude')
    ax.set_title('Individual Mode Decay')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Plot 4: Comparison with finite difference
    ax = axes[1, 0]

    # FD solution at final time
    u_fd = finite_diff_diffusion(u0, D, t_final, dt, dx)
    u_spectral = solutions[-1]

    ax.plot(x, u_spectral, 'b-', lw=2, label='Spectral')
    ax.plot(x, u_fd, 'r--', lw=1.5, label='Finite Diff')
    ax.plot(x, u0, 'k:', lw=1, alpha=0.5, label='Initial')

    ax.set_xlabel('x')
    ax.set_ylabel('u(x,t)')
    ax.set_title(f'Spectral vs FD at t = {t_final}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Error analysis
    ax = axes[1, 1]

    # Analytical solution in Fourier space
    u_hat0 = np.fft.fft(u0)
    decay = np.exp(-D * k**2 * t_final)
    u_exact = np.real(np.fft.ifft(u_hat0 * decay))

    error_spectral = np.abs(u_spectral - u_exact)
    error_fd = np.abs(u_fd - u_exact)

    ax.semilogy(x, error_spectral + 1e-16, 'b-', lw=1.5, label=f'Spectral (max={np.max(error_spectral):.2e})')
    ax.semilogy(x, error_fd + 1e-16, 'r-', lw=1.5, label=f'FD (max={np.max(error_fd):.2e})')

    ax.set_xlabel('x')
    ax.set_ylabel('|Error|')
    ax.set_title('Error Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Summary
    ax = axes[1, 2]
    ax.axis('off')

    summary = f"""Spectral Method Summary
========================
Diffusion: u_t = D·u_xx

Physical Space:
  u(x,t) = Σ û_k(t) exp(ikx)

Fourier Space:
  dû_k/dt = -D·k²·û_k

Exact Solution:
  û_k(t) = û_k(0)·exp(-D·k²·t)

Key Properties:
• Exponential convergence for
  smooth periodic problems
• No numerical dispersion
• High-k modes decay fastest
• Machine precision accuracy
  for smooth initial data

Parameters Used:
  D = {D}
  N = {n} points
  dt = {dt}
  t_final = {t_final}

Comparison:
  Spectral error: ~{np.max(error_spectral):.2e}
  FD error: ~{np.max(error_fd):.2e}"""

    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.suptitle('Spectral Method for Periodic Diffusion Equation',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'spectral_diffusion.png'), dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/spectral_diffusion.png")


if __name__ == "__main__":
    main()
