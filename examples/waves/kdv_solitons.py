"""
Experiment 70: Shallow Water Solitons (KdV Equation).

The Korteweg-de Vries (KdV) equation models shallow water waves:

    u_t + 6*u*u_x + u_xxx = 0

This is the canonical equation for solitons, featuring:
1. Soliton solutions that maintain shape
2. Soliton collisions that pass through each other
3. Conservation of infinitely many quantities
4. Connection to inverse scattering transform
5. Balance of nonlinearity and dispersion

Historical note: This equation connects to the FPU problem through
the continuum limit, as discovered by Zabusky and Kruskal (1965).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq


def soliton(x, t, c, x0=0):
    """
    Single soliton solution of KdV.

    u(x, t) = (c/2) * sech^2(sqrt(c)/2 * (x - c*t - x0))

    Amplitude and speed are related: taller solitons travel faster!
    """
    arg = np.sqrt(c) / 2 * (x - c * t - x0)
    return (c / 2) / np.cosh(arg)**2


def two_soliton(x, t, c1, c2, x1=0, x2=0):
    """
    Two-soliton solution (exact).

    This is the remarkable feature: two solitons pass through
    each other and emerge unchanged (except for a phase shift).
    """
    if c1 == c2:
        return soliton(x, t, c1, x1) + soliton(x, t, c2, x2)

    # Exact two-soliton formula
    k1 = np.sqrt(c1) / 2
    k2 = np.sqrt(c2) / 2

    eta1 = k1 * (x - c1 * t - x1)
    eta2 = k2 * (x - c2 * t - x2)

    # Two-soliton solution via Hirota's method
    delta = (k1 - k2)**2 / (k1 + k2)**2

    f = 1 + np.exp(2*eta1) + np.exp(2*eta2) + delta * np.exp(2*(eta1 + eta2))

    # u = 2 * d^2/dx^2 [log(f)]
    # Computed numerically for simplicity
    dx = x[1] - x[0] if len(x) > 1 else 1e-6
    log_f = np.log(f + 1e-15)

    u = np.zeros_like(x)
    u[1:-1] = 2 * (log_f[2:] - 2*log_f[1:-1] + log_f[:-2]) / dx**2

    return u


def kdv_pseudospectral(u0, x, t_max, dt, nu=0):
    """
    Solve KdV equation using pseudospectral method.

    u_t + 6*u*u_x + u_xxx = nu * u_xx  (with optional viscosity)

    Uses split-step Fourier method:
    - Linear part (u_xxx) in Fourier space
    - Nonlinear part (6*u*u_x) in physical space
    """
    N = len(x)
    L = x[-1] - x[0]
    dx = L / N

    # Wavenumbers
    k = 2 * np.pi * fftfreq(N, dx)

    # Store solution
    n_steps = int(t_max / dt)
    n_save = min(500, n_steps)
    save_every = max(1, n_steps // n_save)

    u_history = np.zeros((n_save, N))
    t_history = np.zeros(n_save)

    u = u0.copy()
    u_hat = fft(u)

    save_idx = 0
    u_history[0] = u
    t_history[0] = 0

    # Linear operator: exp(-i*k^3*dt) for u_xxx, exp(-nu*k^2*dt) for dissipation
    linear_op = np.exp((-1j * k**3 - nu * k**2) * dt)
    linear_op_half = np.exp((-1j * k**3 - nu * k**2) * dt / 2)

    for n in range(1, n_steps):
        # Split-step method (Strang splitting)

        # Half step of linear part
        u_hat = linear_op_half * u_hat

        # Full step of nonlinear part
        u = np.real(ifft(u_hat))
        u_x = np.real(ifft(1j * k * u_hat))
        nonlinear = -6 * u * u_x
        u_hat = u_hat + dt * fft(nonlinear)

        # Half step of linear part
        u_hat = linear_op_half * u_hat
        u = np.real(ifft(u_hat))

        # Save periodically
        if n % save_every == 0 and save_idx < n_save - 1:
            save_idx += 1
            u_history[save_idx] = u
            t_history[save_idx] = n * dt

    return t_history[:save_idx+1], u_history[:save_idx+1]


def conserved_quantities(u, x):
    """
    Compute first three conserved quantities of KdV.

    I1 = integral u dx           (mass)
    I2 = integral u^2 dx         (momentum)
    I3 = integral (u^3 - u_x^2/2) dx  (energy)
    """
    dx = x[1] - x[0]

    I1 = np.sum(u) * dx

    I2 = np.sum(u**2) * dx

    u_x = np.gradient(u, dx)
    I3 = np.sum(u**3 - 0.5 * u_x**2) * dx

    return I1, I2, I3


def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Domain
    L = 40
    N = 512
    x = np.linspace(-L/2, L/2, N)
    dx = x[1] - x[0]

    # Plot 1: Single soliton propagation
    ax = axes[0, 0]

    c = 4.0  # Soliton speed (and determines amplitude)
    times = [0, 2, 4, 6]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(times)))

    for t, color in zip(times, colors):
        u = soliton(x, t, c, x0=-10)
        ax.plot(x, u, color=color, lw=2, label=f't = {t}')

    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.set_title('Single Soliton Propagation\n'
                 f'u = (c/2) * sech^2(sqrt(c)/2 * (x - ct)), c = {c}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-L/2, L/2)

    # Plot 2: Two-soliton collision
    ax = axes[0, 1]

    # Initial condition: two solitons of different speeds
    c1, c2 = 8.0, 2.0  # Different speeds
    x1, x2 = -15, 5    # Initial positions

    u0 = soliton(x, 0, c1, x1) + soliton(x, 0, c2, x2)

    # Simulate
    t_max = 6
    dt = 0.0001
    t_hist, u_hist = kdv_pseudospectral(u0, x, t_max, dt)

    # Plot snapshots
    times_show = [0, 2, 3, 4, 6]
    for t_show in times_show:
        idx = np.argmin(np.abs(t_hist - t_show))
        color = plt.cm.plasma(t_show / t_max)
        ax.plot(x, u_hist[idx], color=color, lw=1.5, alpha=0.8, label=f't={t_hist[idx]:.1f}')

    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.set_title('Two-Soliton Collision\n'
                 'Solitons pass through unchanged!')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-L/2, L/2)

    # Plot 3: Space-time diagram of collision
    ax = axes[0, 2]

    im = ax.imshow(u_hist, extent=[-L/2, L/2, 0, t_hist[-1]],
                   aspect='auto', origin='lower', cmap='RdBu_r')
    plt.colorbar(im, ax=ax, label='u')

    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_title('Space-Time Diagram\n'
                 'Solitons exchange positions')

    # Plot 4: Conservation of invariants
    ax = axes[1, 0]

    # Track conserved quantities
    I1_hist = []
    I2_hist = []
    I3_hist = []

    for u in u_hist:
        I1, I2, I3 = conserved_quantities(u, x)
        I1_hist.append(I1)
        I2_hist.append(I2)
        I3_hist.append(I3)

    I1_hist = np.array(I1_hist)
    I2_hist = np.array(I2_hist)
    I3_hist = np.array(I3_hist)

    ax.plot(t_hist, (I1_hist - I1_hist[0]) / (np.abs(I1_hist[0]) + 1e-10) * 100,
            'b-', lw=1.5, label='I1 (mass)')
    ax.plot(t_hist, (I2_hist - I2_hist[0]) / (np.abs(I2_hist[0]) + 1e-10) * 100,
            'r-', lw=1.5, label='I2 (momentum)')
    ax.plot(t_hist, (I3_hist - I3_hist[0]) / (np.abs(I3_hist[0]) + 1e-10) * 100,
            'g-', lw=1.5, label='I3 (energy)')

    ax.set_xlabel('Time')
    ax.set_ylabel('Relative error (%)')
    ax.set_title('Conservation of Invariants\n'
                 'KdV has infinitely many conserved quantities')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Initial condition breaking into solitons
    ax = axes[1, 1]

    # Gaussian initial condition
    u0_gauss = 4 * np.exp(-(x/3)**2)

    t_max_gauss = 8
    t_hist_g, u_hist_g = kdv_pseudospectral(u0_gauss, x, t_max_gauss, dt)

    times_show = [0, 2, 4, 6, 8]
    for t_show in times_show:
        idx = np.argmin(np.abs(t_hist_g - t_show))
        if idx < len(u_hist_g):
            color = plt.cm.viridis(t_show / t_max_gauss)
            ax.plot(x, u_hist_g[idx], color=color, lw=1.5, label=f't={t_hist_g[idx]:.1f}')

    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.set_title('Gaussian -> Soliton Train\n'
                 'Initial hump breaks into solitons')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-L/2, L/2)

    # Plot 6: Summary
    ax = axes[1, 2]
    ax.axis('off')

    summary = """Korteweg-de Vries (KdV) Equation
================================

THE EQUATION:
  u_t + 6*u*u_x + u_xxx = 0

TERMS:
  u_t     : time evolution
  6*u*u_x : nonlinear steepening
  u_xxx   : dispersion

SOLITON SOLUTION:
  u(x,t) = (c/2) * sech^2[sqrt(c)/2 * (x - ct)]

KEY FEATURES:
- Amplitude ~ speed (taller = faster)
- Solitons collide and emerge unchanged
- Only acquire phase shift in collision

CONSERVATION LAWS:
  I1 = integral u dx         (mass)
  I2 = integral u^2 dx       (momentum)
  I3 = integral (u^3 - u_x^2/2) dx  (energy)
  ... infinitely many more!

PHYSICAL ORIGINS:
- Shallow water waves (original derivation)
- Ion-acoustic waves in plasma
- Continuum limit of FPU chain
- Many other nonlinear wave systems

MATHEMATICAL SIGNIFICANCE:
- First equation solved by inverse
  scattering transform (1967)
- Foundation of soliton theory
- Integrable Hamiltonian system
- Connection to algebraic geometry

HISTORY:
1895: Korteweg & de Vries derived equation
1965: Zabusky & Kruskal found "solitons"
1967: Gardner, Greene, Kruskal, Miura
      solved by inverse scattering"""

    ax.text(0.02, 0.98, summary, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.suptitle("KdV Equation: Solitons in Shallow Water",
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'kdv_solitons.png'), dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/kdv_solitons.png")


if __name__ == "__main__":
    main()
