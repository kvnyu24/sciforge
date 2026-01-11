"""
Experiment 28: Terminal Velocity

Demonstrates vertical fall with drag and the approach to steady-state
terminal velocity.

Physical concepts:
- Newton's second law: m*dv/dt = mg - b*v (linear drag) or mg - c*v^2 (quadratic)
- Terminal velocity: v_t = mg/b (linear) or sqrt(mg/c) (quadratic)
- Exponential approach for linear drag: v(t) = v_t*(1 - exp(-t/tau))
- Time constant tau = m/b = v_t/g

Applications: skydivers, raindrops, sedimenting particles
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt


def terminal_velocity_linear(m, g, b):
    """Terminal velocity for linear drag F = -b*v."""
    return m * g / b


def terminal_velocity_quadratic(m, g, c):
    """Terminal velocity for quadratic drag F = -c*v^2."""
    return np.sqrt(m * g / c)


def solve_linear_drag(m, g, b, v0, t_max, dt=0.001):
    """
    Solve vertical fall with linear drag: m*dv/dt = mg - b*v

    Analytic solution: v(t) = v_t + (v0 - v_t)*exp(-t/tau)
    where v_t = mg/b and tau = m/b
    """
    t = np.arange(0, t_max, dt)
    v_t = terminal_velocity_linear(m, g, b)
    tau = m / b

    # Analytic solution
    v_analytic = v_t + (v0 - v_t) * np.exp(-t / tau)

    # Numerical solution (RK4)
    def dv_dt(v):
        return g - (b / m) * v

    v_numeric = np.zeros_like(t)
    v_numeric[0] = v0

    for i in range(1, len(t)):
        v = v_numeric[i-1]
        k1 = dv_dt(v)
        k2 = dv_dt(v + 0.5 * dt * k1)
        k3 = dv_dt(v + 0.5 * dt * k2)
        k4 = dv_dt(v + dt * k3)
        v_numeric[i] = v + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

    # Position by integration
    y_analytic = v_t * t - tau * (v0 - v_t) * (1 - np.exp(-t / tau))

    return t, v_analytic, v_numeric, y_analytic, v_t, tau


def solve_quadratic_drag(m, g, c, v0, t_max, dt=0.001):
    """
    Solve vertical fall with quadratic drag: m*dv/dt = mg - c*v^2

    Analytic solution: v(t) = v_t * tanh(g*t/v_t + arctanh(v0/v_t))
    """
    t = np.arange(0, t_max, dt)
    v_t = terminal_velocity_quadratic(m, g, c)

    # Analytic solution (for v0 < v_t)
    if v0 < v_t:
        v_analytic = v_t * np.tanh(g * t / v_t + np.arctanh(v0 / v_t))
    else:
        # Handle v0 >= v_t case numerically
        v_analytic = None

    # Numerical solution (RK4)
    def dv_dt(v):
        return g - (c / m) * v * abs(v)

    v_numeric = np.zeros_like(t)
    v_numeric[0] = v0

    for i in range(1, len(t)):
        v = v_numeric[i-1]
        k1 = dv_dt(v)
        k2 = dv_dt(v + 0.5 * dt * k1)
        k3 = dv_dt(v + 0.5 * dt * k2)
        k4 = dv_dt(v + dt * k3)
        v_numeric[i] = v + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

    return t, v_analytic, v_numeric, v_t


def main():
    """Run terminal velocity experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Parameters
    m = 80.0  # mass (kg) - typical skydiver
    g = 9.81  # gravity (m/s^2)

    # Linear drag coefficient (for comparison)
    b_linear = 15.0  # kg/s

    # Quadratic drag coefficient (realistic for skydiver)
    # F = 0.5 * rho * Cd * A * v^2, so c = 0.5 * rho * Cd * A
    rho = 1.2  # air density kg/m^3
    Cd = 1.0   # drag coefficient (spread-eagle)
    A = 0.7    # cross-sectional area m^2
    c = 0.5 * rho * Cd * A

    v0 = 0.0  # starting from rest
    t_max = 30.0  # seconds

    # Plot 1: Linear drag velocity vs time
    ax1 = axes[0, 0]
    t, v_ana, v_num, y, v_t, tau = solve_linear_drag(m, g, b_linear, v0, t_max)

    ax1.plot(t, v_ana, 'b-', lw=2, label='Analytic')
    ax1.plot(t[::50], v_num[::50], 'ro', markersize=4, label='Numerical (RK4)')
    ax1.axhline(y=v_t, color='g', linestyle='--', lw=2,
                label=f'Terminal velocity = {v_t:.2f} m/s')
    ax1.axhline(y=0.99*v_t, color='g', linestyle=':', alpha=0.5)

    # Mark time constant
    idx_tau = np.argmin(np.abs(t - tau))
    v_at_tau = v_ana[idx_tau]
    ax1.plot(tau, v_at_tau, 'ko', markersize=10)
    ax1.annotate(f'$\\tau$ = {tau:.2f} s\nv = {v_at_tau:.2f} m/s\n(63% of $v_t$)',
                 xy=(tau, v_at_tau), xytext=(tau + 2, v_at_tau - 10),
                 fontsize=10, arrowprops=dict(arrowstyle='->', color='black'))

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Velocity (m/s)')
    ax1.set_title('Linear Drag: $F = -bv$')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, t_max)
    ax1.set_ylim(0, v_t * 1.1)

    # Plot 2: Quadratic drag velocity vs time
    ax2 = axes[0, 1]
    t, v_ana_q, v_num_q, v_t_q = solve_quadratic_drag(m, g, c, v0, t_max)

    if v_ana_q is not None:
        ax2.plot(t, v_ana_q, 'b-', lw=2, label='Analytic')
    ax2.plot(t[::50], v_num_q[::50], 'ro', markersize=4, label='Numerical (RK4)')
    ax2.axhline(y=v_t_q, color='g', linestyle='--', lw=2,
                label=f'Terminal velocity = {v_t_q:.2f} m/s')

    # Time to reach 99% of terminal velocity
    idx_99 = np.argmin(np.abs(v_num_q - 0.99 * v_t_q))
    t_99 = t[idx_99]
    ax2.axvline(x=t_99, color='orange', linestyle=':', alpha=0.7)
    ax2.annotate(f'99% at t = {t_99:.1f} s',
                 xy=(t_99, 0.99 * v_t_q), xytext=(t_99 + 2, 0.8 * v_t_q),
                 fontsize=10, arrowprops=dict(arrowstyle='->', color='orange'))

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('Quadratic Drag: $F = -cv^2$ (Skydiver)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, t_max)
    ax2.set_ylim(0, v_t_q * 1.1)

    # Plot 3: Comparison of linear vs quadratic drag
    ax3 = axes[1, 0]

    # Adjust linear drag to have same terminal velocity
    b_adjusted = m * g / v_t_q
    t, v_lin, _, _, _, tau_adj = solve_linear_drag(m, g, b_adjusted, v0, t_max)

    ax3.plot(t, v_lin, 'b-', lw=2, label=f'Linear drag ($\\tau$ = {tau_adj:.2f} s)')
    ax3.plot(t, v_num_q, 'r-', lw=2, label='Quadratic drag')
    ax3.axhline(y=v_t_q, color='g', linestyle='--', lw=2, label=f'$v_t$ = {v_t_q:.2f} m/s')

    # Highlight difference in approach
    ax3.fill_between(t, v_lin, v_num_q, alpha=0.3, color='purple')

    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.set_title('Comparison: Same Terminal Velocity')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 15)

    # Plot 4: Terminal velocity vs mass for different objects
    ax4 = axes[1, 1]

    masses = np.linspace(0.001, 100, 200)  # kg

    # Different objects with different drag characteristics
    objects = {
        'Raindrop (r=2mm)': {'c': 0.5 * 1.2 * 0.47 * np.pi * 0.002**2},
        'Golf ball': {'c': 0.5 * 1.2 * 0.25 * np.pi * 0.0214**2},
        'Baseball': {'c': 0.5 * 1.2 * 0.35 * np.pi * 0.037**2},
        'Skydiver (spread)': {'c': 0.5 * 1.2 * 1.0 * 0.7},
        'Skydiver (dive)': {'c': 0.5 * 1.2 * 0.7 * 0.15},
    }

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(objects)))

    for (name, params), color in zip(objects.items(), colors):
        c_obj = params['c']
        v_t_obj = np.sqrt(masses * g / c_obj)
        ax4.loglog(masses, v_t_obj, lw=2, color=color, label=name)

    # Mark typical values
    ax4.axhline(y=9.0, color='gray', linestyle=':', alpha=0.5)  # raindrop
    ax4.text(0.002, 10, '~9 m/s (raindrop)', fontsize=9)
    ax4.axhline(y=55, color='gray', linestyle=':', alpha=0.5)  # skydiver spread
    ax4.text(50, 60, '~55 m/s (spread)', fontsize=9)
    ax4.axhline(y=90, color='gray', linestyle=':', alpha=0.5)  # skydiver dive
    ax4.text(50, 95, '~90 m/s (dive)', fontsize=9)

    ax4.set_xlabel('Mass (kg)')
    ax4.set_ylabel('Terminal Velocity (m/s)')
    ax4.set_title('Terminal Velocity vs Mass')
    ax4.legend(loc='lower right', fontsize=9)
    ax4.grid(True, alpha=0.3, which='both')
    ax4.set_xlim(0.001, 100)
    ax4.set_ylim(1, 500)

    plt.suptitle('Experiment 28: Terminal Velocity\n'
                 'Vertical fall with drag approaching steady state',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save output
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'terminal_velocity.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'terminal_velocity.png')}")

    # Print summary
    print("\n=== Terminal Velocity Summary ===")
    print(f"Linear drag: v_t = mg/b = {terminal_velocity_linear(m, g, b_linear):.2f} m/s")
    print(f"Quadratic drag (skydiver): v_t = sqrt(mg/c) = {v_t_q:.2f} m/s")
    print(f"Time to reach 99% of v_t (quadratic): {t_99:.1f} s")


if __name__ == "__main__":
    main()
