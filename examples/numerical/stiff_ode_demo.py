"""
Experiment 4: Stiff ODE demo - RC circuit with tiny time constant.

Demonstrates the stability issues with explicit methods on stiff
systems, and compares to implicit (backward Euler) approach.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def rc_circuit_rhs(V, t, R, C, V_source):
    """
    RC circuit charging: dV/dt = (V_source - V) / (RC)

    This becomes stiff when RC is very small (fast decay).
    """
    return (V_source(t) - V) / (R * C)


def explicit_euler(V, t, dt, R, C, V_source):
    """Explicit (forward) Euler - unstable for stiff systems with large dt."""
    return V + dt * rc_circuit_rhs(V, t, R, C, V_source)


def implicit_euler(V, t, dt, R, C, V_source):
    """
    Implicit (backward) Euler - stable for stiff systems.

    For RC circuit: V_{n+1} = V_n + dt * (V_s(t+dt) - V_{n+1}) / (RC)
    Solving: V_{n+1} = (V_n + dt * V_s(t+dt) / (RC)) / (1 + dt / (RC))
    """
    tau = R * C
    return (V + dt * V_source(t + dt) / tau) / (1 + dt / tau)


def rk4_step(V, t, dt, R, C, V_source):
    """RK4 step."""
    def f(V, t):
        return rc_circuit_rhs(V, t, R, C, V_source)

    k1 = f(V, t)
    k2 = f(V + dt/2 * k1, t + dt/2)
    k3 = f(V + dt/2 * k2, t + dt/2)
    k4 = f(V + dt * k3, t + dt)
    return V + dt * (k1 + 2*k2 + 2*k3 + k4) / 6


def exact_rc_response(t, R, C, V0, V_final):
    """Exact solution for step input: V(t) = V_final + (V0 - V_final) * exp(-t/RC)"""
    tau = R * C
    return V_final + (V0 - V_final) * np.exp(-t / tau)


def simulate(V0, t_final, dt, R, C, V_source, method):
    """Run simulation with given method."""
    methods = {
        'explicit': explicit_euler,
        'implicit': implicit_euler,
        'rk4': rk4_step
    }

    step = methods[method]
    t = 0
    V = V0
    times = [t]
    voltages = [V]

    while t < t_final:
        V = step(V, t, dt, R, C, V_source)
        t += dt
        times.append(t)
        voltages.append(V)

    return np.array(times), np.array(voltages)


def main():
    # Circuit parameters - very small time constant (stiff)
    R = 1.0  # Ohms
    C = 1e-6  # Farads (1 microfarad)
    tau = R * C  # = 1e-6 s = 1 microsecond

    # Step input
    V_step = lambda t: 5.0  # 5V step

    V0 = 0.0
    V_final = 5.0
    t_final = 10 * tau  # 10 time constants

    # Time steps to test (relative to tau)
    dt_ratios = [0.1, 0.5, 1.0, 2.0, 5.0]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Exact solution for comparison
    t_exact = np.linspace(0, t_final, 1000)
    V_exact = exact_rc_response(t_exact, R, C, V0, V_final)

    # Test different time steps
    for idx, ratio in enumerate(dt_ratios):
        ax = axes.flatten()[idx]
        dt = ratio * tau

        # Explicit Euler
        try:
            t_exp, V_exp = simulate(V0, t_final, dt, R, C, V_step, 'explicit')
            # Check for instability (values going crazy)
            if np.max(np.abs(V_exp)) > 1e10:
                label_exp = f'Explicit (UNSTABLE)'
                V_exp = np.clip(V_exp, -100, 100)
            else:
                label_exp = 'Explicit Euler'
            ax.plot(t_exp / tau, V_exp, 'b-', lw=1.5, alpha=0.7, label=label_exp)
        except:
            pass

        # Implicit Euler
        t_imp, V_imp = simulate(V0, t_final, dt, R, C, V_step, 'implicit')
        ax.plot(t_imp / tau, V_imp, 'r--', lw=1.5, alpha=0.7, label='Implicit Euler')

        # RK4
        try:
            t_rk4, V_rk4 = simulate(V0, t_final, dt, R, C, V_step, 'rk4')
            if np.max(np.abs(V_rk4)) > 1e10:
                label_rk4 = f'RK4 (UNSTABLE)'
                V_rk4 = np.clip(V_rk4, -100, 100)
            else:
                label_rk4 = 'RK4'
            ax.plot(t_rk4 / tau, V_rk4, 'g-.', lw=1.5, alpha=0.7, label=label_rk4)
        except:
            pass

        # Exact
        ax.plot(t_exact / tau, V_exact, 'k-', lw=2, alpha=0.5, label='Exact')

        ax.set_xlabel('Time (t/τ)')
        ax.set_ylabel('Voltage (V)')
        ax.set_title(f'dt/τ = {ratio}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1, 7)
        ax.set_xlim(0, 10)

    # Summary panel
    ax = axes[1, 2]
    ax.axis('off')

    summary = f"""Stiff ODE Demonstration
========================
RC Circuit Parameters:
  R = {R} Ω
  C = {C*1e6:.1f} μF
  τ = RC = {tau*1e6:.1f} μs

Stability Analysis:
==================
For explicit Euler on y' = λy:
  Stable if |1 + λdt| < 1
  For RC: λ = -1/τ = -{1/tau:.0e}
  Stability requires dt < 2τ

Observations:
- dt/τ < 2: All methods stable
- dt/τ > 2: Explicit methods unstable
- Implicit Euler: Unconditionally stable
  (A-stable, L-stable)

Stiffness Ratio:
  |λ_fast/λ_slow| = large for stiff systems
  Here we only have one eigenvalue,
  but fast decay makes explicit
  methods need tiny time steps.

Practical Implications:
  For stiff systems (chemical kinetics,
  electronic circuits, etc.), implicit
  methods allow much larger time steps."""

    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Stiff ODE: RC Circuit with τ = 1 μs\n' +
                 'Explicit vs Implicit Method Stability', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'stiff_ode_demo.png'), dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/stiff_ode_demo.png")


if __name__ == "__main__":
    main()
