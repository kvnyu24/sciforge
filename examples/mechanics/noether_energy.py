"""
Experiment 62: Noether's Theorem - Time Translation Symmetry and Energy Conservation.

Demonstrates Noether's theorem: every continuous symmetry of the Lagrangian
corresponds to a conserved quantity.

For time-translation invariance (L does not depend explicitly on t):
  -> Energy is conserved

We show this by:
1. Verifying dE/dt = 0 when L has no explicit time dependence
2. Showing dE/dt != 0 when L depends explicitly on t (time-varying potential)
3. Deriving the conserved quantity (Hamiltonian) from the Noether current
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def rk4_step(f, t, state, dt):
    """Single RK4 step."""
    k1 = f(t, state)
    k2 = f(t + dt/2, state + dt/2 * k1)
    k3 = f(t + dt/2, state + dt/2 * k2)
    k4 = f(t + dt, state + dt * k3)
    return state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)


def simulate(f, state0, t_span, dt):
    """Simulate ODE using RK4."""
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / dt)
    t = np.linspace(t_start, t_end, n_steps)
    states = np.zeros((n_steps, len(state0)))
    states[0] = state0

    for i in range(1, n_steps):
        states[i] = rk4_step(f, t[i-1], states[i-1], dt)

    return t, states


# System 1: Simple harmonic oscillator (time-independent L)
class HarmonicOscillator:
    """
    L = (1/2)m*v^2 - (1/2)k*x^2

    Time-translation invariant -> Energy conserved
    """

    def __init__(self, m=1.0, k=1.0):
        self.m = m
        self.k = k

    def lagrangian(self, x, v, t):
        T = 0.5 * self.m * v**2
        V = 0.5 * self.k * x**2
        return T - V

    def hamiltonian(self, x, v, t):
        """H = p*v - L = T + V (for natural Lagrangians)"""
        T = 0.5 * self.m * v**2
        V = 0.5 * self.k * x**2
        return T + V

    def equations_of_motion(self, t, state):
        x, v = state
        a = -self.k * x / self.m
        return np.array([v, a])

    def dL_dt_explicit(self, x, v, t):
        """Explicit time derivative of L (should be 0)."""
        return 0.0


# System 2: Oscillator with time-varying spring constant
class TimeVaryingOscillator:
    """
    L = (1/2)m*v^2 - (1/2)k(t)*x^2

    where k(t) = k0 * (1 + A*sin(omega*t))

    NOT time-translation invariant -> Energy NOT conserved
    """

    def __init__(self, m=1.0, k0=1.0, A=0.3, omega=0.5):
        self.m = m
        self.k0 = k0
        self.A = A
        self.omega = omega

    def k(self, t):
        return self.k0 * (1 + self.A * np.sin(self.omega * t))

    def dk_dt(self, t):
        return self.k0 * self.A * self.omega * np.cos(self.omega * t)

    def lagrangian(self, x, v, t):
        T = 0.5 * self.m * v**2
        V = 0.5 * self.k(t) * x**2
        return T - V

    def hamiltonian(self, x, v, t):
        T = 0.5 * self.m * v**2
        V = 0.5 * self.k(t) * x**2
        return T + V

    def equations_of_motion(self, t, state):
        x, v = state
        a = -self.k(t) * x / self.m
        return np.array([v, a])

    def dL_dt_explicit(self, x, v, t):
        """Explicit time derivative of L."""
        # dL/dt|explicit = -dV/dt|explicit = -(1/2)*(dk/dt)*x^2
        return -0.5 * self.dk_dt(t) * x**2


# System 3: Particle in gravitational field (time-independent)
class FreeFall:
    """
    L = (1/2)m*v^2 - m*g*y

    Time-translation invariant -> Energy conserved
    """

    def __init__(self, m=1.0, g=9.81):
        self.m = m
        self.g = g

    def lagrangian(self, y, v, t):
        T = 0.5 * self.m * v**2
        V = self.m * self.g * y
        return T - V

    def hamiltonian(self, y, v, t):
        T = 0.5 * self.m * v**2
        V = self.m * self.g * y
        return T + V

    def equations_of_motion(self, t, state):
        y, v = state
        a = -self.g
        return np.array([v, a])


# System 4: Parametric oscillator (explicit time dependence in mass term)
class ParametricOscillator:
    """
    L = (1/2)m(t)*v^2 - (1/2)k*x^2

    where m(t) = m0 * exp(-gamma*t) (damping-like effect)

    NOT time-translation invariant -> Energy NOT conserved
    But Noether's theorem tells us WHAT is conserved!
    """

    def __init__(self, m0=1.0, gamma=0.1, k=1.0):
        self.m0 = m0
        self.gamma = gamma
        self.k = k

    def m(self, t):
        return self.m0 * np.exp(-self.gamma * t)

    def dm_dt(self, t):
        return -self.gamma * self.m(t)

    def lagrangian(self, x, v, t):
        T = 0.5 * self.m(t) * v**2
        V = 0.5 * self.k * x**2
        return T - V

    def hamiltonian(self, x, v, t):
        T = 0.5 * self.m(t) * v**2
        V = 0.5 * self.k * x**2
        return T + V

    def equations_of_motion(self, t, state):
        x, v = state
        # EOM: m(t)*a + dm/dt*v + k*x = 0
        a = (-self.dm_dt(t) * v - self.k * x) / self.m(t)
        return np.array([v, a])

    def dL_dt_explicit(self, x, v, t):
        """Explicit time derivative of L."""
        return 0.5 * self.dm_dt(t) * v**2


def compute_noether_current_time(system, x, v, t, epsilon=1e-7):
    """
    Compute Noether current for time translation.

    For time translation symmetry:
    J = sum_i (dL/dq_dot_i * q_dot_i) - L = H

    This is the Hamiltonian (energy) for natural Lagrangians.
    """
    L = system.lagrangian(x, v, t)

    # dL/dv * v
    dL_dv = (system.lagrangian(x, v + epsilon, t) - system.lagrangian(x, v - epsilon, t)) / (2 * epsilon)

    # Noether current = p*v - L = H
    J = dL_dv * v - L

    return J


def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Simulation parameters
    dt = 0.01
    t_span = (0, 30)

    # Initial conditions
    x0, v0 = 1.0, 0.0
    state0 = np.array([x0, v0])

    # System 1: Time-independent harmonic oscillator
    ax = axes[0, 0]

    ho = HarmonicOscillator(m=1.0, k=1.0)
    t, states = simulate(ho.equations_of_motion, state0, t_span, dt)

    E = np.array([ho.hamiltonian(s[0], s[1], ti) for s, ti in zip(states, t)])
    dE = (E - E[0]) / E[0] * 100

    ax.plot(t, states[:, 0], 'b-', lw=1, label='Position x')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position', color='b')
    ax.tick_params(axis='y', labelcolor='b')

    ax2 = ax.twinx()
    ax2.plot(t, dE, 'r-', lw=1, label='Energy error')
    ax2.set_ylabel('Energy error (%)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_ylim(-1, 1)

    ax.set_title('Harmonic Oscillator (Time-Independent L)\n'
                 'Energy conserved: dE/dt = 0')
    ax.grid(True, alpha=0.3)

    # System 2: Time-varying spring
    ax = axes[0, 1]

    tvo = TimeVaryingOscillator(m=1.0, k0=1.0, A=0.5, omega=0.3)
    t, states = simulate(tvo.equations_of_motion, state0, t_span, dt)

    E = np.array([tvo.hamiltonian(s[0], s[1], ti) for s, ti in zip(states, t)])
    k_t = np.array([tvo.k(ti) for ti in t])

    ax.plot(t, states[:, 0], 'b-', lw=1, label='Position x')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position', color='b')
    ax.tick_params(axis='y', labelcolor='b')

    ax2 = ax.twinx()
    ax2.plot(t, E, 'r-', lw=1.5, label='Energy')
    ax2.plot(t, k_t * 0.5, 'g--', lw=1, alpha=0.7, label='k(t)/2')
    ax2.set_ylabel('Energy (J)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.legend(loc='upper right', fontsize=8)

    ax.set_title('Time-Varying Spring: k(t) = k0(1 + A*sin(wt))\n'
                 'Energy NOT conserved: dL/dt != 0')
    ax.grid(True, alpha=0.3)

    # System 3: Noether current verification
    ax = axes[0, 2]

    # For time-independent system, Noether current = H = constant
    ho = HarmonicOscillator()
    t, states = simulate(ho.equations_of_motion, state0, t_span, dt)

    # Compute Noether current (should equal Hamiltonian)
    noether = np.array([compute_noether_current_time(ho, s[0], s[1], ti)
                        for s, ti in zip(states, t)])
    hamiltonian = np.array([ho.hamiltonian(s[0], s[1], ti) for s, ti in zip(states, t)])

    ax.plot(t, noether, 'b-', lw=2, label='Noether current J')
    ax.plot(t, hamiltonian, 'r--', lw=2, label='Hamiltonian H')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Value')
    ax.set_title("Noether Current = Hamiltonian\n"
                 "J = sum(dL/dv * v) - L = H")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # System 4: Compare conservative vs non-conservative
    ax = axes[1, 0]

    ho = HarmonicOscillator()
    tvo = TimeVaryingOscillator(A=0.5, omega=0.5)

    t1, states1 = simulate(ho.equations_of_motion, state0, t_span, dt)
    t2, states2 = simulate(tvo.equations_of_motion, state0, t_span, dt)

    E1 = np.array([ho.hamiltonian(s[0], s[1], ti) for s, ti in zip(states1, t1)])
    E2 = np.array([tvo.hamiltonian(s[0], s[1], ti) for s, ti in zip(states2, t2)])

    ax.plot(t1, E1, 'b-', lw=2, label='Time-independent (conserved)')
    ax.plot(t2, E2, 'r-', lw=2, label='Time-dependent (not conserved)')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Energy (J)')
    ax.set_title('Energy Conservation Comparison\n'
                 'Symmetry breaking -> Conservation breaking')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # System 5: dH/dt = -dL/dt verification
    ax = axes[1, 1]

    tvo = TimeVaryingOscillator(A=0.3, omega=0.5)
    t, states = simulate(tvo.equations_of_motion, state0, t_span, dt)

    H = np.array([tvo.hamiltonian(s[0], s[1], ti) for s, ti in zip(states, t)])
    dH_dt = np.gradient(H, t)

    dL_dt_explicit = np.array([tvo.dL_dt_explicit(s[0], s[1], ti)
                               for s, ti in zip(states, t)])

    ax.plot(t, dH_dt, 'b-', lw=1.5, label='dH/dt (numerical)')
    ax.plot(t, -dL_dt_explicit, 'r--', lw=1.5, label='-dL/dt|explicit')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Rate of change')
    ax.set_title('Noether Identity: dH/dt = -dL/dt|explicit\n'
                 'Energy change equals explicit time dependence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Summary
    ax = axes[1, 2]
    ax.axis('off')

    summary = """Noether's Theorem: Time Translation Symmetry
=============================================

THEOREM:
If the Lagrangian L(q, q_dot, t) does not
depend explicitly on time (dL/dt = 0), then
the system has a conserved quantity:

       n
  H = SUM (dL/dq_dot_i * q_dot_i) - L
      i=1

This is the HAMILTONIAN (total energy for
natural Lagrangians where L = T - V).

KEY RELATIONSHIPS:

1. Time symmetry  <=>  Energy conservation
   dL/dt = 0      <=>  dH/dt = 0

2. If dL/dt != 0 (explicit time dependence):
   dH/dt = -dL/dt|explicit

PHYSICAL INTERPRETATION:

- Time-independent L means physics doesn't
  change with the clock -> energy conserved

- Pumping energy into system (time-varying
  potential) breaks time symmetry and
  energy conservation

THIS IS EXPERIMENT 62 OF THE CATALOG:
Noether theorem: time translation -> energy

This deep connection between symmetry and
conservation underlies all of modern physics."""

    ax.text(0.02, 0.98, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.suptitle("Noether's Theorem: Time Translation Invariance and Energy Conservation",
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'noether_energy.png'), dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/noether_energy.png")


if __name__ == "__main__":
    main()
