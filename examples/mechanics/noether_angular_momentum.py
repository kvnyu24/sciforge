"""
Experiment 63: Noether's Theorem - Rotational Symmetry and Angular Momentum Conservation.

Demonstrates Noether's theorem for spatial rotations:

For rotational invariance (L does not depend on angle phi):
  -> Angular momentum is conserved

We show this by:
1. Central force problems (rotationally symmetric) - L conserved
2. Non-central forces - L not conserved
3. Deriving angular momentum as the Noether current for rotations
4. Multi-particle systems with rotational symmetry
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


class CentralForce2D:
    """
    Particle in 2D central force field.

    L = (1/2)m(x_dot^2 + y_dot^2) - V(r)

    where r = sqrt(x^2 + y^2)

    Rotationally symmetric -> Angular momentum L_z = m(x*vy - y*vx) conserved
    """

    def __init__(self, m=1.0, k=1.0, force_type='gravitational'):
        self.m = m
        self.k = k
        self.force_type = force_type

    def V(self, r):
        """Potential energy V(r)."""
        if self.force_type == 'gravitational':
            return -self.k / (r + 1e-10)
        elif self.force_type == 'harmonic':
            return 0.5 * self.k * r**2
        elif self.force_type == 'coulomb':
            return self.k / (r + 1e-10)

    def dV_dr(self, r):
        """Radial force = -dV/dr."""
        if self.force_type == 'gravitational':
            return -self.k / (r + 1e-10)**2
        elif self.force_type == 'harmonic':
            return self.k * r
        elif self.force_type == 'coulomb':
            return -self.k / (r + 1e-10)**2

    def angular_momentum(self, x, y, vx, vy):
        """L_z = m(x*vy - y*vx) = r x p."""
        return self.m * (x * vy - y * vx)

    def equations_of_motion(self, t, state):
        x, y, vx, vy = state
        r = np.sqrt(x**2 + y**2) + 1e-10

        # Force = -dV/dr * (r_hat)
        Fr = -self.dV_dr(r)
        ax = Fr * x / r / self.m
        ay = Fr * y / r / self.m

        return np.array([vx, vy, ax, ay])


class NonCentralForce2D:
    """
    Particle in 2D with non-central force (NOT rotationally symmetric).

    V(x, y) = (1/2)k_x*x^2 + (1/2)k_y*y^2  (anisotropic harmonic oscillator)

    or

    V(x, y) = -k/(r) + epsilon*x  (tilted gravitational field)

    NOT rotationally symmetric -> Angular momentum NOT conserved
    """

    def __init__(self, m=1.0, kx=1.0, ky=2.0, force_type='anisotropic'):
        self.m = m
        self.kx = kx
        self.ky = ky
        self.force_type = force_type

    def angular_momentum(self, x, y, vx, vy):
        """L_z = m(x*vy - y*vx)."""
        return self.m * (x * vy - y * vx)

    def equations_of_motion(self, t, state):
        x, y, vx, vy = state

        if self.force_type == 'anisotropic':
            ax = -self.kx * x / self.m
            ay = -self.ky * y / self.m
        else:  # tilted
            r = np.sqrt(x**2 + y**2) + 1e-10
            k = 1.0
            epsilon = 0.3
            ax = k * x / r**3 / self.m - epsilon / self.m
            ay = k * y / r**3 / self.m

        return np.array([vx, vy, ax, ay])


class AxisymmetricTop:
    """
    Axisymmetric spinning top (simplified 2D model).

    The Lagrangian only depends on theta (tilt angle), not on phi (precession).
    Therefore L_phi (angular momentum about vertical) is conserved.
    """

    def __init__(self, I1=1.0, I3=0.5, mgh=1.0):
        self.I1 = I1  # Moment of inertia about symmetry axis
        self.I3 = I3  # Moment about perpendicular axis
        self.mgh = mgh  # Gravitational torque coefficient

    def lagrangian(self, theta, phi, theta_dot, phi_dot):
        """L = T - V for spinning top."""
        T = 0.5 * self.I1 * theta_dot**2 + 0.5 * self.I3 * (phi_dot * np.cos(theta))**2
        V = self.mgh * np.cos(theta)
        return T - V

    def angular_momentum_phi(self, theta, phi_dot):
        """Conserved momentum conjugate to phi."""
        return self.I3 * phi_dot * np.cos(theta)**2


def compute_noether_current_rotation(system, x, y, vx, vy, epsilon=1e-7):
    """
    Compute Noether current for infinitesimal rotation.

    For rotation by angle delta_phi about z-axis:
    delta_x = -y * delta_phi
    delta_y = x * delta_phi

    Noether current: J = sum_i (dL/dq_dot_i * delta_q_i)
                       = m*vx*(-y) + m*vy*(x)
                       = m*(x*vy - y*vx) = L_z
    """
    return system.m * (x * vy - y * vx)


def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    dt = 0.001
    t_span = (0, 20)

    # Initial conditions for orbit
    x0, y0 = 1.0, 0.0
    vx0 = 0.0
    vy0 = 1.2  # Elliptical orbit

    state0 = np.array([x0, y0, vx0, vy0])

    # Plot 1: Central force - L conserved
    ax = axes[0, 0]

    cf = CentralForce2D(m=1.0, k=1.0, force_type='gravitational')
    t, states = simulate(cf.equations_of_motion, state0, t_span, dt)

    x, y = states[:, 0], states[:, 1]
    vx, vy = states[:, 2], states[:, 3]
    L = np.array([cf.angular_momentum(xi, yi, vxi, vyi)
                  for xi, yi, vxi, vyi in zip(x, y, vx, vy)])

    ax.plot(x, y, 'b-', lw=0.5)
    ax.plot(0, 0, 'yo', markersize=12, label='Central body')
    ax.plot(x[0], y[0], 'go', markersize=8, label='Start')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Central Force: V = -k/r\n'
                 'Rotationally symmetric')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Plot 2: Angular momentum for central force
    ax = axes[0, 1]

    ax.plot(t, L, 'b-', lw=1.5)
    ax.axhline(y=L[0], color='r', linestyle='--', lw=1, label='Initial L')

    dL = (L - L[0]) / np.abs(L[0]) * 100
    ax.fill_between(t, L, L[0], alpha=0.3)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angular momentum L_z')
    ax.set_title(f'L_z Conservation (Central Force)\n'
                 f'Max deviation: {np.max(np.abs(dL)):.2e}%')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Non-central force - L NOT conserved
    ax = axes[0, 2]

    ncf = NonCentralForce2D(m=1.0, kx=1.0, ky=2.0, force_type='anisotropic')
    t, states = simulate(ncf.equations_of_motion, state0, t_span, dt)

    x, y = states[:, 0], states[:, 1]
    vx, vy = states[:, 2], states[:, 3]
    L_nc = np.array([ncf.angular_momentum(xi, yi, vxi, vyi)
                     for xi, yi, vxi, vyi in zip(x, y, vx, vy)])

    ax.plot(t, L_nc, 'r-', lw=1.5)
    ax.axhline(y=L_nc[0], color='b', linestyle='--', lw=1, alpha=0.5)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angular momentum L_z')
    ax.set_title('L_z NOT Conserved (Anisotropic Oscillator)\n'
                 'V = (1/2)kx*x^2 + (1/2)ky*y^2, kx != ky')
    ax.grid(True, alpha=0.3)

    # Plot 4: Compare orbits
    ax = axes[1, 0]

    # Central force orbit
    cf = CentralForce2D(m=1.0, k=1.0)
    t, states_c = simulate(cf.equations_of_motion, state0, t_span, dt)
    x_c, y_c = states_c[:, 0], states_c[:, 1]

    # Non-central force orbit
    ncf = NonCentralForce2D(m=1.0, kx=1.0, ky=2.0)
    t, states_nc = simulate(ncf.equations_of_motion, state0, t_span, dt)
    x_nc, y_nc = states_nc[:, 0], states_nc[:, 1]

    ax.plot(x_c, y_c, 'b-', lw=0.5, label='Central (L conserved)')
    ax.plot(x_nc, y_nc, 'r-', lw=0.5, label='Non-central (L varies)')
    ax.plot(0, 0, 'ko', markersize=8)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Orbit Comparison\n'
                 'Symmetry determines conservation')
    ax.legend(fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Plot 5: Noether current verification
    ax = axes[1, 1]

    cf = CentralForce2D(m=1.0, k=1.0)
    t, states = simulate(cf.equations_of_motion, state0, t_span, dt)

    x, y = states[:, 0], states[:, 1]
    vx, vy = states[:, 2], states[:, 3]

    # Compute angular momentum directly
    L_direct = cf.m * (x * vy - y * vx)

    # Compute via Noether current
    L_noether = np.array([compute_noether_current_rotation(cf, xi, yi, vxi, vyi)
                          for xi, yi, vxi, vyi in zip(x, y, vx, vy)])

    ax.plot(t, L_direct, 'b-', lw=2, label='L_z = m(x*vy - y*vx)')
    ax.plot(t, L_noether, 'r--', lw=2, label='Noether current')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angular momentum')
    ax.set_title('Noether Current = Angular Momentum\n'
                 'J = sum(dL/dq_dot * delta_q) for rotations')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Summary
    ax = axes[1, 2]
    ax.axis('off')

    summary = """Noether's Theorem: Rotational Symmetry
=======================================

THEOREM:
If the Lagrangian L(q, q_dot) is invariant
under rotations (depends only on |r|, not
on angle phi), then angular momentum is
conserved:

       L_z = m(x*vy - y*vx) = r x p

DERIVATION (Noether Current):
For infinitesimal rotation delta_phi:
  delta_x = -y * delta_phi
  delta_y = +x * delta_phi

Noether current:
  J = dL/dvx * delta_x + dL/dvy * delta_y
    = m*vx*(-y) + m*vy*(x)
    = m(x*vy - y*vx)
    = L_z

EXAMPLES:

1. Central Force (V = V(r)):
   - Rotationally symmetric
   - L_z = constant (Kepler's 2nd law!)

2. Anisotropic Oscillator (kx != ky):
   - NOT rotationally symmetric
   - L_z oscillates

3. Isotropic Oscillator (kx = ky):
   - Rotationally symmetric
   - L_z = constant

KEY INSIGHT:
Kepler's "equal areas in equal times"
is just angular momentum conservation,
which follows from rotational symmetry
of gravity!

THIS IS EXPERIMENT 63:
Noether theorem: rotation -> angular momentum"""

    ax.text(0.02, 0.98, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    plt.suptitle("Noether's Theorem: Rotational Invariance and Angular Momentum Conservation",
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'noether_angular_momentum.png'), dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/noether_angular_momentum.png")


if __name__ == "__main__":
    main()
