"""
Experiment 61: Poisson Bracket Dynamics.

Demonstrates the Poisson bracket formulation of classical mechanics.
The Poisson bracket is fundamental to Hamiltonian mechanics and
provides a bridge to quantum mechanics.

Key concepts:
1. {q, H} = dq/dt (Hamilton's first equation)
2. {p, H} = dp/dt (Hamilton's second equation)
3. {f, H} = 0 implies f is a constant of motion
4. Canonical relations: {q_i, p_j} = delta_ij

The Poisson bracket of two phase space functions f and g is:
    {f, g} = sum_i (df/dq_i * dg/dp_i - df/dp_i * dg/dq_i)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def numerical_derivative(f, x, i, h=1e-7):
    """
    Numerical partial derivative of f with respect to x[i].

    Args:
        f: Function f(x) where x is an array
        x: Point at which to evaluate
        i: Index of variable to differentiate
        h: Step size

    Returns:
        Partial derivative df/dx_i
    """
    x_plus = x.copy()
    x_minus = x.copy()
    x_plus[i] += h
    x_minus[i] -= h
    return (f(x_plus) - f(x_minus)) / (2 * h)


def poisson_bracket(f, g, state, n_dof):
    """
    Compute the Poisson bracket {f, g} at a given phase space point.

    {f, g} = sum_i (df/dq_i * dg/dp_i - df/dp_i * dg/dq_i)

    Args:
        f: Function f(state) where state = [q1,...,qn, p1,...,pn]
        g: Function g(state)
        state: Phase space point [q1,...,qn, p1,...,pn]
        n_dof: Number of degrees of freedom

    Returns:
        Poisson bracket value
    """
    result = 0.0

    for i in range(n_dof):
        # df/dq_i
        df_dq = numerical_derivative(f, state, i)
        # df/dp_i
        df_dp = numerical_derivative(f, state, n_dof + i)
        # dg/dq_i
        dg_dq = numerical_derivative(g, state, i)
        # dg/dp_i
        dg_dp = numerical_derivative(g, state, n_dof + i)

        result += df_dq * dg_dp - df_dp * dg_dq

    return result


class HamiltonianSystem:
    """
    A Hamiltonian system with Poisson bracket dynamics.
    """

    def __init__(self, H, n_dof):
        """
        Initialize the system.

        Args:
            H: Hamiltonian function H(state) where state = [q..., p...]
            n_dof: Number of degrees of freedom
        """
        self.H = H
        self.n_dof = n_dof

    def equations_of_motion(self, state):
        """
        Compute dq/dt and dp/dt using Poisson brackets.

        dq_i/dt = {q_i, H}
        dp_i/dt = {p_i, H}
        """
        derivatives = np.zeros_like(state)

        for i in range(self.n_dof):
            # {q_i, H} = dH/dp_i
            derivatives[i] = numerical_derivative(self.H, state, self.n_dof + i)
            # {p_i, H} = -dH/dq_i
            derivatives[self.n_dof + i] = -numerical_derivative(self.H, state, i)

        return derivatives

    def simulate(self, state0, t_span, dt):
        """Simulate using RK4."""
        t_start, t_end = t_span
        n_steps = int((t_end - t_start) / dt)
        times = np.linspace(t_start, t_end, n_steps)

        states = np.zeros((n_steps, len(state0)))
        states[0] = state0

        for i in range(1, n_steps):
            s = states[i-1]
            k1 = self.equations_of_motion(s)
            k2 = self.equations_of_motion(s + 0.5*dt*k1)
            k3 = self.equations_of_motion(s + 0.5*dt*k2)
            k4 = self.equations_of_motion(s + dt*k3)
            states[i] = s + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

        return times, states

    def check_poisson_bracket(self, f, state):
        """
        Compute {f, H} to check if f is a constant of motion.

        If {f, H} = 0, then df/dt = 0 (f is conserved).
        """
        return poisson_bracket(f, self.H, state, self.n_dof)


def main():
    fig = plt.figure(figsize=(16, 12))

    # --- Example 1: Simple Harmonic Oscillator ---
    # H = p^2/(2m) + (1/2)k*q^2

    m = 1.0
    k = 1.0
    omega = np.sqrt(k/m)

    def H_sho(state):
        q, p = state
        return p**2 / (2*m) + 0.5*k*q**2

    sho = HamiltonianSystem(H_sho, n_dof=1)

    # --- Plot 1: Verify Hamilton's equations via Poisson brackets ---
    ax1 = fig.add_subplot(2, 3, 1)

    # Test at various phase space points
    q_test = np.linspace(-2, 2, 10)
    p_test = np.linspace(-2, 2, 10)

    # {q, H} should equal dH/dp = p/m
    # {p, H} should equal -dH/dq = -k*q

    dq_dt_theory = []
    dq_dt_bracket = []
    dp_dt_theory = []
    dp_dt_bracket = []

    for q in q_test:
        for p in p_test:
            state = np.array([q, p])

            # Theoretical values
            dq_dt_theory.append(p/m)
            dp_dt_theory.append(-k*q)

            # From Poisson brackets
            def q_func(s):
                return s[0]

            def p_func(s):
                return s[1]

            dq_dt_bracket.append(poisson_bracket(q_func, H_sho, state, 1))
            dp_dt_bracket.append(poisson_bracket(p_func, H_sho, state, 1))

    ax1.scatter(dq_dt_theory, dq_dt_bracket, alpha=0.5, label='{q, H} vs p/m')
    ax1.scatter(dp_dt_theory, dp_dt_bracket, alpha=0.5, label='{p, H} vs -kq')
    ax1.plot([-3, 3], [-3, 3], 'k--', label='Perfect agreement')

    ax1.set_xlabel('Theoretical dq/dt or dp/dt')
    ax1.set_ylabel('Poisson Bracket Value')
    ax1.set_title("Hamilton's Equations from Poisson Brackets\n{q,H}=dq/dt, {p,H}=dp/dt")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Phase space trajectory ---
    ax2 = fig.add_subplot(2, 3, 2)

    state0 = np.array([2.0, 0.0])  # Start at q=2, p=0
    times, states = sho.simulate(state0, (0, 10), 0.01)

    ax2.plot(states[:, 0], states[:, 1], 'b-', lw=1.5)
    ax2.plot(state0[0], state0[1], 'go', markersize=10, label='Start')

    # Draw equipotential lines (constant H)
    q_grid = np.linspace(-3, 3, 100)
    p_grid = np.linspace(-3, 3, 100)
    Q, P = np.meshgrid(q_grid, p_grid)
    H_grid = P**2/(2*m) + 0.5*k*Q**2

    ax2.contour(Q, P, H_grid, levels=10, colors='gray', alpha=0.5)

    ax2.set_xlabel('q (position)')
    ax2.set_ylabel('p (momentum)')
    ax2.set_title('Phase Space Trajectory\n(Along constant H contour)')
    ax2.legend()
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    # --- Plot 3: Conservation of energy {H, H} = 0 ---
    ax3 = fig.add_subplot(2, 3, 3)

    # Calculate H along trajectory
    energies = [H_sho(s) for s in states]
    energy_error = np.array(energies) - energies[0]

    # Also check {H, H} at each point
    HH_brackets = [poisson_bracket(H_sho, H_sho, s, 1) for s in states[::10]]

    ax3.plot(times, energy_error, 'b-', lw=1.5, label='H(t) - H(0)')
    ax3.axhline(0, color='k', linestyle='--', alpha=0.5)

    ax3.set_xlabel('Time')
    ax3.set_ylabel('Energy deviation')
    ax3.set_title('Energy Conservation\n{H, H} = 0 implies dH/dt = 0')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Add annotation about bracket
    ax3.text(0.5, 0.95, f'Max |{{H,H}}| = {max(abs(np.array(HH_brackets))):.2e}',
             transform=ax3.transAxes, fontsize=10)

    # --- Example 2: Kepler Problem (Angular Momentum Conservation) ---
    ax4 = fig.add_subplot(2, 3, 4)

    # H = (p_r^2)/(2m) + L^2/(2mr^2) - k/r
    # But let's use 2D Cartesian: H = (px^2 + py^2)/(2m) - k/r

    m_kepler = 1.0
    k_kepler = 1.0

    def H_kepler(state):
        x, y, px, py = state
        r = np.sqrt(x**2 + y**2) + 1e-10
        return (px**2 + py**2)/(2*m_kepler) - k_kepler/r

    def L_z(state):
        """Angular momentum L_z = x*py - y*px"""
        x, y, px, py = state
        return x*py - y*px

    kepler = HamiltonianSystem(H_kepler, n_dof=2)

    # Initial conditions for elliptical orbit
    r0 = 1.0
    v0 = 0.8 * np.sqrt(k_kepler / (m_kepler * r0))  # Less than circular velocity
    state0_kepler = np.array([r0, 0.0, 0.0, m_kepler * v0])

    times_k, states_k = kepler.simulate(state0_kepler, (0, 20), 0.01)

    # Calculate L_z over time
    L_values = [L_z(s) for s in states_k]

    ax4.plot(times_k, L_values, 'b-', lw=1.5)
    ax4.axhline(L_values[0], color='r', linestyle='--', label='Initial L_z')

    ax4.set_xlabel('Time')
    ax4.set_ylabel('Angular Momentum L_z')
    ax4.set_title('Angular Momentum Conservation\n{L_z, H} = 0 for central force')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Verify {L, H} = 0
    L_H_brackets = [kepler.check_poisson_bracket(L_z, s) for s in states_k[::20]]
    ax4.text(0.5, 0.05, f'|{{L_z, H}}| < {max(abs(np.array(L_H_brackets))):.2e}',
             transform=ax4.transAxes, fontsize=10)

    # --- Plot 5: Canonical Poisson bracket relations ---
    ax5 = fig.add_subplot(2, 3, 5)

    # For 2 DOF system, verify:
    # {q_i, q_j} = 0
    # {p_i, p_j} = 0
    # {q_i, p_j} = delta_ij

    def q1(s):
        return s[0]

    def q2(s):
        return s[1]

    def p1(s):
        return s[2]

    def p2(s):
        return s[3]

    # Test at a random point
    test_state = np.array([1.5, -0.5, 2.0, -1.0])

    brackets = {
        '{q1, q2}': poisson_bracket(q1, q2, test_state, 2),
        '{p1, p2}': poisson_bracket(p1, p2, test_state, 2),
        '{q1, p1}': poisson_bracket(q1, p1, test_state, 2),
        '{q1, p2}': poisson_bracket(q1, p2, test_state, 2),
        '{q2, p1}': poisson_bracket(q2, p1, test_state, 2),
        '{q2, p2}': poisson_bracket(q2, p2, test_state, 2),
    }

    expected = {
        '{q1, q2}': 0,
        '{p1, p2}': 0,
        '{q1, p1}': 1,
        '{q1, p2}': 0,
        '{q2, p1}': 0,
        '{q2, p2}': 1,
    }

    names = list(brackets.keys())
    computed = list(brackets.values())
    expected_vals = [expected[n] for n in names]

    x_pos = np.arange(len(names))
    width = 0.35

    bars1 = ax5.bar(x_pos - width/2, computed, width, label='Computed')
    bars2 = ax5.bar(x_pos + width/2, expected_vals, width, label='Expected', alpha=0.7)

    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(names, rotation=45, ha='right')
    ax5.set_ylabel('Bracket Value')
    ax5.set_title('Canonical Poisson Bracket Relations\n{q_i, p_j} = delta_ij')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')

    # --- Plot 6: Theory summary ---
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    summary = """Poisson Brackets in Classical Mechanics
========================================

DEFINITION:
-----------
    {f, g} = sum_i (df/dq_i * dg/dp_i - df/dp_i * dg/dq_i)

PROPERTIES:
-----------
1. Antisymmetry:    {f, g} = -{g, f}
2. Linearity:       {af + bg, h} = a{f,h} + b{g,h}
3. Leibniz rule:    {fg, h} = f{g,h} + g{f,h}
4. Jacobi identity: {{f,g},h} + {{g,h},f} + {{h,f},g} = 0

HAMILTON'S EQUATIONS:
---------------------
    dq_i/dt = {q_i, H} = dH/dp_i
    dp_i/dt = {p_i, H} = -dH/dq_i

For any function f(q, p, t):
    df/dt = {f, H} + df/dt_explicit

CANONICAL RELATIONS:
--------------------
    {q_i, q_j} = 0
    {p_i, p_j} = 0
    {q_i, p_j} = delta_ij

CONSTANTS OF MOTION:
--------------------
If {f, H} = 0, then df/dt = 0 (f is conserved)

Examples:
- {H, H} = 0  =>  Energy conserved
- {L_z, H} = 0 for central force  =>  Angular momentum
- {p_i, H} = 0 if dH/dq_i = 0  =>  Momentum (if symmetry)

CONNECTION TO QUANTUM MECHANICS:
--------------------------------
    {f, g}  <-->  (1/ih)[F, G]  (commutator)

The Poisson bracket becomes the quantum commutator!"""

    ax6.text(0.02, 0.98, summary, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.suptitle('Poisson Bracket Dynamics (Experiment 61)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'poisson_brackets.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {output_path}")

    # Print numerical verification
    print("\nPoisson Bracket Verification:")
    print("-" * 50)
    print("\nCanonical bracket relations:")
    for name, value in brackets.items():
        print(f"  {name} = {value:.6f} (expected: {expected[name]})")

    print("\nConservation laws verified:")
    print(f"  |{{H, H}}| < {max(abs(np.array(HH_brackets))):.2e} (energy)")
    print(f"  |{{L_z, H}}| < {max(abs(np.array(L_H_brackets))):.2e} (angular momentum)")


if __name__ == "__main__":
    main()
