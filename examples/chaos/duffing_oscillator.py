"""
Experiment 65: Duffing Oscillator - Chaos in a Driven Nonlinear System.

The Duffing oscillator is a paradigm of chaos in driven nonlinear systems:

    x'' + delta*x' + alpha*x + beta*x^3 = gamma*cos(omega*t)

Where:
- delta: damping coefficient
- alpha: linear stiffness (can be negative for double-well potential)
- beta: nonlinear stiffness (cubic term)
- gamma: driving amplitude
- omega: driving frequency

This demonstrates:
1. Period-doubling route to chaos
2. Strange attractors in Poincare sections
3. Sensitivity to initial conditions
4. Bifurcation diagrams
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def duffing_derivatives(state, t, delta, alpha, beta, gamma, omega):
    """
    Duffing oscillator equations of motion.

    x'' + delta*x' + alpha*x + beta*x^3 = gamma*cos(omega*t)

    State: [x, v] where v = x'
    """
    x, v = state
    dxdt = v
    dvdt = gamma * np.cos(omega * t) - delta * v - alpha * x - beta * x**3
    return np.array([dxdt, dvdt])


def simulate_duffing(x0, v0, t_max, dt, delta, alpha, beta, gamma, omega):
    """Simulate Duffing oscillator using RK4."""
    n_steps = int(t_max / dt)
    t = np.linspace(0, t_max, n_steps)

    state = np.zeros((n_steps, 2))
    state[0] = [x0, v0]

    for i in range(1, n_steps):
        s = state[i-1]
        ti = t[i-1]

        k1 = duffing_derivatives(s, ti, delta, alpha, beta, gamma, omega)
        k2 = duffing_derivatives(s + dt/2*k1, ti + dt/2, delta, alpha, beta, gamma, omega)
        k3 = duffing_derivatives(s + dt/2*k2, ti + dt/2, delta, alpha, beta, gamma, omega)
        k4 = duffing_derivatives(s + dt*k3, ti + dt, delta, alpha, beta, gamma, omega)

        state[i] = s + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    return t, state


def poincare_section(t, state, omega):
    """
    Extract Poincare section: sample state at driving period.

    Sample when omega*t = 2*pi*n (stroboscopic map)
    """
    T = 2 * np.pi / omega
    dt = t[1] - t[0]
    period_steps = int(T / dt)

    # Skip transient (first 100 periods)
    start_idx = 100 * period_steps
    if start_idx >= len(t):
        start_idx = len(t) // 2

    indices = np.arange(start_idx, len(t), period_steps)
    return state[indices, 0], state[indices, 1]


def bifurcation_diagram(gamma_range, delta, alpha, beta, omega, n_transient=200, n_sample=100):
    """
    Compute bifurcation diagram varying gamma (driving amplitude).
    """
    gamma_values = []
    x_values = []

    dt = 0.01
    T = 2 * np.pi / omega

    for gamma in gamma_range:
        # Simulate
        t_max = (n_transient + n_sample) * T
        t, state = simulate_duffing(0.1, 0.0, t_max, dt, delta, alpha, beta, gamma, omega)

        # Get Poincare section after transient
        period_steps = int(T / dt)
        start_idx = n_transient * period_steps

        if start_idx < len(t):
            for i in range(n_sample):
                idx = start_idx + i * period_steps
                if idx < len(t):
                    gamma_values.append(gamma)
                    x_values.append(state[idx, 0])

    return np.array(gamma_values), np.array(x_values)


def lyapunov_exponent(delta, alpha, beta, gamma, omega, t_max=1000, dt=0.01):
    """
    Estimate largest Lyapunov exponent.
    """
    # Main trajectory
    x0, v0 = 0.1, 0.0
    t, state = simulate_duffing(x0, v0, t_max, dt, delta, alpha, beta, gamma, omega)

    # Perturbed trajectory
    eps = 1e-8
    t2, state2 = simulate_duffing(x0 + eps, v0, t_max, dt, delta, alpha, beta, gamma, omega)

    # Track divergence
    d = np.sqrt((state[:, 0] - state2[:, 0])**2 + (state[:, 1] - state2[:, 1])**2)

    # Estimate Lyapunov from log growth
    # Skip transient
    skip = len(t) // 10
    d = d[skip:]
    t_lyap = t[skip:]

    # Linear fit to log(d)
    valid = (d > 1e-15) & (d < 10)
    if np.sum(valid) > 100:
        from scipy.stats import linregress
        slope, _, _, _, _ = linregress(t_lyap[valid], np.log(d[valid]))
        return slope
    return 0.0


def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Parameters for double-well Duffing oscillator
    delta = 0.3   # Damping
    alpha = -1.0  # Negative linear stiffness (double-well)
    beta = 1.0    # Positive cubic stiffness
    omega = 1.2   # Driving frequency

    dt = 0.01

    # Plot 1: Time series - periodic vs chaotic
    ax = axes[0, 0]

    # Low amplitude - periodic
    gamma_low = 0.3
    t, state_low = simulate_duffing(0.1, 0.0, 200, dt, delta, alpha, beta, gamma_low, omega)

    # High amplitude - chaotic
    gamma_high = 0.5
    t, state_high = simulate_duffing(0.1, 0.0, 200, dt, delta, alpha, beta, gamma_high, omega)

    ax.plot(t[-2000:], state_low[-2000:, 0], 'b-', lw=0.5, label=f'gamma={gamma_low} (periodic)')
    ax.plot(t[-2000:], state_high[-2000:, 0], 'r-', lw=0.5, alpha=0.7, label=f'gamma={gamma_high} (chaotic)')

    ax.set_xlabel('Time')
    ax.set_ylabel('x(t)')
    ax.set_title('Duffing Oscillator: Periodic vs Chaotic')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: Phase space
    ax = axes[0, 1]

    ax.plot(state_low[-5000:, 0], state_low[-5000:, 1], 'b-', lw=0.3, alpha=0.7, label='Periodic')
    ax.plot(state_high[-5000:, 0], state_high[-5000:, 1], 'r-', lw=0.3, alpha=0.5, label='Chaotic')

    ax.set_xlabel('x')
    ax.set_ylabel('dx/dt')
    ax.set_title('Phase Space Trajectories')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Poincare section (strange attractor)
    ax = axes[0, 2]

    gamma_chaos = 0.5
    t, state = simulate_duffing(0.1, 0.0, 2000, dt, delta, alpha, beta, gamma_chaos, omega)
    x_poincare, v_poincare = poincare_section(t, state, omega)

    ax.scatter(x_poincare, v_poincare, s=0.5, c='blue', alpha=0.5)

    ax.set_xlabel('x')
    ax.set_ylabel('dx/dt')
    ax.set_title(f'Poincare Section (gamma={gamma_chaos})\n'
                 'Strange attractor with fractal structure')
    ax.grid(True, alpha=0.3)

    # Plot 4: Bifurcation diagram
    ax = axes[1, 0]

    gamma_range = np.linspace(0.2, 0.6, 200)
    gamma_bif, x_bif = bifurcation_diagram(gamma_range, delta, alpha, beta, omega)

    ax.scatter(gamma_bif, x_bif, s=0.1, c='black', alpha=0.5)

    ax.set_xlabel('Driving amplitude gamma')
    ax.set_ylabel('x (at Poincare section)')
    ax.set_title('Bifurcation Diagram\n'
                 'Period-doubling route to chaos')
    ax.grid(True, alpha=0.3)

    # Plot 5: Sensitive dependence
    ax = axes[1, 1]

    gamma = 0.5
    t_max = 100

    # Two nearby trajectories
    x0 = 0.1
    eps = 1e-6

    t, state1 = simulate_duffing(x0, 0.0, t_max, dt, delta, alpha, beta, gamma, omega)
    t, state2 = simulate_duffing(x0 + eps, 0.0, t_max, dt, delta, alpha, beta, gamma, omega)

    divergence = np.sqrt((state1[:, 0] - state2[:, 0])**2 + (state1[:, 1] - state2[:, 1])**2)

    ax.semilogy(t, divergence, 'b-', lw=1)
    ax.axhline(y=eps, color='r', linestyle='--', label=f'Initial separation: {eps}')

    ax.set_xlabel('Time')
    ax.set_ylabel('Trajectory separation (log scale)')
    ax.set_title('Butterfly Effect in Duffing Oscillator\n'
                 'Exponential divergence of nearby trajectories')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    # Estimate and annotate Lyapunov exponent
    try:
        lyap = lyapunov_exponent(delta, alpha, beta, gamma, omega)
        ax.text(0.05, 0.95, f'Lyapunov exp ~ {lyap:.3f}',
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    except Exception:
        pass

    # Plot 6: Potential and summary
    ax = axes[1, 2]

    # Draw double-well potential
    x = np.linspace(-2, 2, 200)
    V = alpha/2 * x**2 + beta/4 * x**4

    ax_pot = ax.inset_axes([0.05, 0.55, 0.4, 0.4])
    ax_pot.plot(x, V, 'b-', lw=2)
    ax_pot.set_xlabel('x', fontsize=8)
    ax_pot.set_ylabel('V(x)', fontsize=8)
    ax_pot.set_title('Double-well potential', fontsize=8)
    ax_pot.grid(True, alpha=0.3)

    ax.axis('off')

    summary = """Duffing Oscillator
==================

Equation of motion:
  x'' + delta*x' + alpha*x + beta*x^3 = gamma*cos(omega*t)

Parameters used:
  delta = 0.3  (damping)
  alpha = -1.0 (negative -> double-well)
  beta  = 1.0  (restoring nonlinearity)
  omega = 1.2  (driving frequency)

KEY PHENOMENA:

1. Double-Well Potential:
   V(x) = (alpha/2)*x^2 + (beta/4)*x^4
   Two stable equilibria at x = +/- sqrt(-alpha/beta)

2. Period-Doubling Cascade:
   As gamma increases:
   Period 1 -> Period 2 -> Period 4 -> ... -> Chaos

3. Strange Attractor:
   In chaotic regime, Poincare section shows
   fractal structure (self-similar at all scales)

4. Intermittency:
   Near bifurcation points, system alternates
   between regular and chaotic behavior

5. Hysteresis:
   System can have multiple attractors;
   final state depends on initial conditions

APPLICATIONS:
- Vibration engineering
- Ship rolling dynamics
- Electrical circuits (Josephson junctions)
- Buckling of beams"""

    ax.text(0.35, 0.95, summary, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle("Duffing Oscillator: Chaos in a Driven Nonlinear System\n"
                 "x'' + delta*x' + alpha*x + beta*x^3 = gamma*cos(omega*t)",
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'duffing_oscillator.png'), dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/duffing_oscillator.png")


if __name__ == "__main__":
    main()
