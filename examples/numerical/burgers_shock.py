"""
Experiment 15: Burgers' equation - shock formation + viscosity regularization.

Demonstrates shock formation in the inviscid Burgers equation and
how viscosity regularizes the solution.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def initial_sine(x):
    """Sinusoidal initial condition."""
    return np.sin(2 * np.pi * x)


def initial_step(x, x0=0.5):
    """Step function initial condition."""
    return np.where(x < x0, 1.0, 0.0)


def inviscid_burgers_godunov(u, dx, dt):
    """
    Godunov scheme for inviscid Burgers equation: u_t + (u^2/2)_x = 0

    Uses exact Riemann solver for Burgers equation.
    """
    u_new = np.zeros_like(u)
    n = len(u)

    for i in range(n):
        # Left and right states at cell interfaces
        u_L = u[i-1]  # Periodic BC
        u_R = u[i]
        u_LL = u[i-2]
        u_RR = u[(i+1) % n]

        # Godunov flux at left interface (i-1/2)
        if u_L >= u_R:  # Shock or expansion fan
            if u_L > 0 and u_R > 0:
                f_left = 0.5 * u_L**2
            elif u_L < 0 and u_R < 0:
                f_left = 0.5 * u_R**2
            else:  # Transonic
                if u_L >= 0 >= u_R:
                    f_left = 0.0  # Sonic point
                else:
                    f_left = min(0.5 * u_L**2, 0.5 * u_R**2)
        else:  # Expansion
            if u_L >= 0:
                f_left = 0.5 * u_L**2
            elif u_R <= 0:
                f_left = 0.5 * u_R**2
            else:
                f_left = 0.0

        # Godunov flux at right interface (i+1/2)
        u_L2 = u[i]
        u_R2 = u[(i+1) % n]

        if u_L2 >= u_R2:
            if u_L2 > 0 and u_R2 > 0:
                f_right = 0.5 * u_L2**2
            elif u_L2 < 0 and u_R2 < 0:
                f_right = 0.5 * u_R2**2
            else:
                if u_L2 >= 0 >= u_R2:
                    f_right = 0.0
                else:
                    f_right = min(0.5 * u_L2**2, 0.5 * u_R2**2)
        else:
            if u_L2 >= 0:
                f_right = 0.5 * u_L2**2
            elif u_R2 <= 0:
                f_right = 0.5 * u_R2**2
            else:
                f_right = 0.0

        u_new[i] = u[i] - dt/dx * (f_right - f_left)

    return u_new


def viscous_burgers(u, dx, dt, nu):
    """
    Viscous Burgers equation: u_t + u*u_x = nu * u_xx

    Using operator splitting: advection (upwind) + diffusion (FTCS)
    """
    n = len(u)
    u_new = u.copy()

    # Advection step (upwind)
    for i in range(n):
        if u[i] > 0:
            u_new[i] = u[i] - dt/dx * u[i] * (u[i] - u[i-1])
        else:
            u_new[i] = u[i] - dt/dx * u[i] * (u[(i+1) % n] - u[i])

    # Diffusion step
    u = u_new.copy()
    r = nu * dt / dx**2
    for i in range(1, n-1):
        u_new[i] = u[i] + r * (u[i+1] - 2*u[i] + u[i-1])
    u_new[0] = u[0] + r * (u[1] - 2*u[0] + u[-1])
    u_new[-1] = u_new[0]

    return u_new


def characteristics_burgers(x0, t, u0_func):
    """
    Compute characteristics for inviscid Burgers equation.

    Characteristic: x = x0 + u0(x0) * t
    """
    u0 = u0_func(x0)
    x = x0 + u0 * t
    return x, u0


def main():
    # Parameters
    L = 1.0
    nx = 200
    x = np.linspace(0, L, nx, endpoint=False)
    dx = x[1] - x[0]

    # Time stepping (CFL condition)
    dt = 0.5 * dx  # CFL ~ 0.5 for max |u| ~ 1

    # Viscosity values
    viscosities = [0, 0.001, 0.005, 0.02]

    # Time snapshots
    times = [0, 0.1, 0.2, 0.3]

    fig, axes = plt.subplots(2, 4, figsize=(16, 10))

    # Row 1: Evolution with different viscosities at t=0.3
    u0 = initial_sine(x)

    for col, nu in enumerate(viscosities):
        ax = axes[0, col]

        u = u0.copy()
        t = 0
        n_steps = int(times[-1] / dt)

        for _ in range(n_steps):
            if nu == 0:
                u = inviscid_burgers_godunov(u, dx, dt)
            else:
                u = viscous_burgers(u, dx, dt, nu)
            t += dt

        ax.plot(x, u0, 'k--', lw=1, alpha=0.5, label='Initial')
        ax.plot(x, u, 'b-', lw=2, label=f't = {times[-1]}')

        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.set_title(f'ν = {nu}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.5, 1.5)

    # Row 2: Time evolution for inviscid case with characteristics
    ax = axes[1, 0]

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(times)))

    for i, t_snap in enumerate(times):
        u = u0.copy()
        n_steps = int(t_snap / dt) if t_snap > 0 else 0

        for _ in range(n_steps):
            u = inviscid_burgers_godunov(u, dx, dt)

        ax.plot(x, u, '-', color=colors[i], lw=1.5, label=f't = {t_snap}')

    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.set_title('Shock Formation (Inviscid)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.5, 1.5)

    # Row 2, col 2: Characteristics
    ax = axes[1, 1]

    x0_chars = np.linspace(0, 1, 20, endpoint=False)
    t_range = np.linspace(0, 0.3, 50)

    for x0_c in x0_chars:
        u0_c = initial_sine(np.array([x0_c]))[0]
        x_char = x0_c + u0_c * t_range
        x_char = np.mod(x_char, L)  # Periodic wrapping

        color = 'red' if u0_c > 0 else 'blue'
        ax.plot(x_char, t_range, '-', color=color, alpha=0.5, lw=0.8)

    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_title('Characteristics (red: u>0, blue: u<0)')
    ax.set_xlim(0, L)
    ax.set_ylim(0, 0.3)
    ax.grid(True, alpha=0.3)

    # Row 2, col 3: Shock profile comparison
    ax = axes[1, 2]

    u0 = initial_step(x)
    t_shock = 0.2

    for nu in [0, 0.005, 0.02]:
        u = u0.copy()
        n_steps = int(t_shock / dt)

        for _ in range(n_steps):
            if nu == 0:
                u = inviscid_burgers_godunov(u, dx, dt)
            else:
                u = viscous_burgers(u, dx, dt, nu)

        label = 'Inviscid' if nu == 0 else f'ν = {nu}'
        ax.plot(x, u, '-', lw=1.5, label=label)

    # Exact shock location
    shock_speed = 0.5  # (u_L + u_R) / 2 = 0.5
    x_shock = 0.5 + shock_speed * t_shock
    ax.axvline(x_shock, color='gray', linestyle='--', alpha=0.5, label='Shock location')

    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.set_title(f'Shock Profile (t = {t_shock})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 2, col 4: Summary
    ax = axes[1, 3]
    ax.axis('off')

    summary = """Burgers' Equation Summary
=========================
Inviscid: u_t + u·u_x = 0

• Nonlinear advection causes
  wave steepening
• Characteristics cross where
  u_x < 0 initially
• Shock forms at breaking time:
  t_b = -1 / min(u_x(x,0))

Viscous: u_t + u·u_x = ν·u_xx

• Viscosity regularizes shocks
• Shock width ~ ν/Δu
• Balances steepening with
  diffusive smoothing

Numerical Methods:
• Godunov: shock-capturing,
  1st order accurate
• Viscous: operator splitting,
  advection + diffusion

Conservation:
∫ u dx is conserved (exact)
Numerical schemes may have
small conservation errors."""

    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle("Burgers' Equation: Shock Formation and Viscous Regularization",
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'burgers_shock.png'), dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/burgers_shock.png")


if __name__ == "__main__":
    main()
