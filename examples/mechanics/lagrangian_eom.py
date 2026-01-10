"""
Experiment 61: Derive Equations of Motion from Lagrangian.

Demonstrates how the Euler-Lagrange equations derive equations of motion
from a Lagrangian L = T - V. We implement symbolic-like differentiation
numerically and verify against known analytical solutions for:

1. Simple pendulum
2. Double pendulum
3. Particle in central force
4. Coupled oscillators

The Euler-Lagrange equation: d/dt(dL/dq_dot) - dL/dq = 0
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def numerical_derivative(f, x, h=1e-7):
    """Central difference numerical derivative."""
    return (f(x + h) - f(x - h)) / (2 * h)


class LagrangianSystem:
    """
    Base class for systems defined by a Lagrangian.

    Subclasses implement the Lagrangian L(q, q_dot, t).
    This class provides the Euler-Lagrange equations numerically.
    """

    def __init__(self, n_dof):
        """
        Initialize system.

        Args:
            n_dof: Number of degrees of freedom
        """
        self.n_dof = n_dof

    def lagrangian(self, q, q_dot, t):
        """
        Compute Lagrangian L = T - V.

        Args:
            q: Generalized coordinates (array of length n_dof)
            q_dot: Generalized velocities
            t: Time

        Returns:
            Scalar Lagrangian value
        """
        raise NotImplementedError

    def dL_dq(self, q, q_dot, t, h=1e-7):
        """Partial derivative of L with respect to q."""
        result = np.zeros(self.n_dof)
        for i in range(self.n_dof):
            def L_qi(qi):
                q_temp = q.copy()
                q_temp[i] = qi
                return self.lagrangian(q_temp, q_dot, t)
            result[i] = numerical_derivative(L_qi, q[i], h)
        return result

    def dL_dqdot(self, q, q_dot, t, h=1e-7):
        """Partial derivative of L with respect to q_dot (generalized momenta)."""
        result = np.zeros(self.n_dof)
        for i in range(self.n_dof):
            def L_qdoti(qdoti):
                qdot_temp = q_dot.copy()
                qdot_temp[i] = qdoti
                return self.lagrangian(q, qdot_temp, t)
            result[i] = numerical_derivative(L_qdoti, q_dot[i], h)
        return result

    def equations_of_motion(self, t, state):
        """
        Compute accelerations from Euler-Lagrange equations.

        The Euler-Lagrange equation:
        d/dt(dL/dq_dot) = dL/dq

        Expanding: d/dt(dL/dq_dot) = d²L/(dq_dot dq) * q_dot + d²L/(dq_dot)² * q_ddot

        We solve for q_ddot using the generalized mass matrix.
        """
        q = state[:self.n_dof]
        q_dot = state[self.n_dof:]

        # Compute generalized mass matrix M_ij = d²L / dq_dot_i dq_dot_j
        h = 1e-5
        M = np.zeros((self.n_dof, self.n_dof))
        for i in range(self.n_dof):
            for j in range(self.n_dof):
                def d2L_dqdoti_dqdotj(qi, qj):
                    qdot_temp = q_dot.copy()
                    qdot_temp[i] = qi
                    qdot_temp[j] = qj
                    return self.lagrangian(q, qdot_temp, t)

                # Mixed second derivative
                M[i, j] = (d2L_dqdoti_dqdotj(q_dot[i]+h, q_dot[j]+h)
                           - d2L_dqdoti_dqdotj(q_dot[i]+h, q_dot[j]-h)
                           - d2L_dqdoti_dqdotj(q_dot[i]-h, q_dot[j]+h)
                           + d2L_dqdoti_dqdotj(q_dot[i]-h, q_dot[j]-h)) / (4*h**2)

        # Compute dL/dq
        dLdq = self.dL_dq(q, q_dot, t)

        # Compute d²L/(dq_dot dq) * q_dot term
        Coriolis = np.zeros(self.n_dof)
        for i in range(self.n_dof):
            for j in range(self.n_dof):
                def d2L_dqdoti_dqj(qdoti, qj):
                    q_temp = q.copy()
                    q_temp[j] = qj
                    qdot_temp = q_dot.copy()
                    qdot_temp[i] = qdoti
                    return self.lagrangian(q_temp, qdot_temp, t)

                d2 = (d2L_dqdoti_dqj(q_dot[i]+h, q[j]+h)
                      - d2L_dqdoti_dqj(q_dot[i]+h, q[j]-h)
                      - d2L_dqdoti_dqj(q_dot[i]-h, q[j]+h)
                      + d2L_dqdoti_dqj(q_dot[i]-h, q[j]-h)) / (4*h**2)
                Coriolis[i] += d2 * q_dot[j]

        # Euler-Lagrange: M * q_ddot = dL/dq - Coriolis
        # Solve for q_ddot
        rhs = dLdq - Coriolis
        try:
            q_ddot = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            q_ddot = np.zeros(self.n_dof)

        return np.concatenate([q_dot, q_ddot])

    def simulate(self, q0, q_dot0, t_span, dt):
        """Simulate using RK4."""
        t_start, t_end = t_span
        n_steps = int((t_end - t_start) / dt)
        t = np.linspace(t_start, t_end, n_steps)

        state = np.zeros((n_steps, 2 * self.n_dof))
        state[0] = np.concatenate([q0, q_dot0])

        for i in range(1, n_steps):
            s = state[i-1]
            k1 = self.equations_of_motion(t[i-1], s)
            k2 = self.equations_of_motion(t[i-1] + dt/2, s + dt/2 * k1)
            k3 = self.equations_of_motion(t[i-1] + dt/2, s + dt/2 * k2)
            k4 = self.equations_of_motion(t[i-1] + dt, s + dt * k3)
            state[i] = s + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        return t, state


class SimplePendulum(LagrangianSystem):
    """Simple pendulum: L = (1/2)ml^2*theta_dot^2 + mgl*cos(theta)"""

    def __init__(self, length=1.0, mass=1.0, g=9.81):
        super().__init__(n_dof=1)
        self.length = length
        self.mass = mass
        self.g = g

    def lagrangian(self, q, q_dot, t):
        theta = q[0]
        theta_dot = q_dot[0]
        T = 0.5 * self.mass * self.length**2 * theta_dot**2
        V = -self.mass * self.g * self.length * np.cos(theta)
        return T - V


class DoublePendulum(LagrangianSystem):
    """Double pendulum in Lagrangian formulation."""

    def __init__(self, L1=1.0, L2=1.0, m1=1.0, m2=1.0, g=9.81):
        super().__init__(n_dof=2)
        self.L1, self.L2 = L1, L2
        self.m1, self.m2 = m1, m2
        self.g = g

    def lagrangian(self, q, q_dot, t):
        theta1, theta2 = q
        omega1, omega2 = q_dot

        # Kinetic energy
        T = (0.5 * self.m1 * self.L1**2 * omega1**2 +
             0.5 * self.m2 * (self.L1**2 * omega1**2 +
                              self.L2**2 * omega2**2 +
                              2 * self.L1 * self.L2 * omega1 * omega2 * np.cos(theta1 - theta2)))

        # Potential energy
        V = (-(self.m1 + self.m2) * self.g * self.L1 * np.cos(theta1) -
              self.m2 * self.g * self.L2 * np.cos(theta2))

        return T - V


class CentralForce(LagrangianSystem):
    """Particle in central force (2D polar coordinates)."""

    def __init__(self, mass=1.0, k=1.0):
        super().__init__(n_dof=2)  # r, phi
        self.mass = mass
        self.k = k  # Force constant (e.g., gravitational GM*m)

    def lagrangian(self, q, q_dot, t):
        r, phi = q
        r_dot, phi_dot = q_dot

        # Kinetic energy in polar coordinates
        T = 0.5 * self.mass * (r_dot**2 + r**2 * phi_dot**2)

        # Gravitational potential
        V = -self.k / (r + 1e-10)

        return T - V


class CoupledOscillators(LagrangianSystem):
    """Two coupled harmonic oscillators."""

    def __init__(self, m1=1.0, m2=1.0, k1=1.0, k2=1.0, k_coupling=0.5):
        super().__init__(n_dof=2)
        self.m1, self.m2 = m1, m2
        self.k1, self.k2 = k1, k2
        self.k_c = k_coupling

    def lagrangian(self, q, q_dot, t):
        x1, x2 = q
        v1, v2 = q_dot

        T = 0.5 * self.m1 * v1**2 + 0.5 * self.m2 * v2**2
        V = 0.5 * self.k1 * x1**2 + 0.5 * self.k2 * x2**2 + 0.5 * self.k_c * (x2 - x1)**2

        return T - V


def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Simple pendulum - compare with analytical
    ax = axes[0, 0]

    pendulum = SimplePendulum(length=1.0, mass=1.0, g=9.81)
    q0 = np.array([0.3])  # Small angle
    q_dot0 = np.array([0.0])

    t, state = pendulum.simulate(q0, q_dot0, (0, 10), 0.01)

    # Analytical solution for small angles: theta = theta0 * cos(omega*t)
    omega = np.sqrt(pendulum.g / pendulum.length)
    theta_analytical = q0[0] * np.cos(omega * t)

    ax.plot(t, state[:, 0], 'b-', lw=2, label='Lagrangian EOM')
    ax.plot(t, theta_analytical, 'r--', lw=2, label='Analytical (small angle)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (rad)')
    ax.set_title('Simple Pendulum\nL = (1/2)ml^2*omega^2 + mgl*cos(theta)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Energy conservation
    ax = axes[0, 1]

    # Calculate energy
    theta = state[:, 0]
    theta_dot = state[:, 1]
    T = 0.5 * pendulum.mass * pendulum.length**2 * theta_dot**2
    V = -pendulum.mass * pendulum.g * pendulum.length * np.cos(theta)
    E = T + V

    ax.plot(t, T, 'r-', lw=1.5, label='Kinetic T')
    ax.plot(t, V - V.min(), 'b-', lw=1.5, label='Potential V (shifted)')
    ax.plot(t, E, 'k--', lw=2, label='Total E')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Energy (J)')
    ax.set_title('Energy Conservation\n(Euler-Lagrange preserves energy)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Double pendulum
    ax = axes[0, 2]

    dp = DoublePendulum(L1=1.0, L2=1.0, m1=1.0, m2=1.0)
    q0 = np.array([np.pi/4, np.pi/4])
    q_dot0 = np.array([0.0, 0.0])

    t_dp, state_dp = dp.simulate(q0, q_dot0, (0, 10), 0.005)

    ax.plot(t_dp, np.degrees(state_dp[:, 0]), 'b-', lw=1, label='theta1')
    ax.plot(t_dp, np.degrees(state_dp[:, 1]), 'r-', lw=1, label='theta2')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (degrees)')
    ax.set_title('Double Pendulum (from Lagrangian)\n'
                 'L = T(theta1, theta2, omega1, omega2) - V(theta1, theta2)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Central force - orbital motion
    ax = axes[1, 0]

    cf = CentralForce(mass=1.0, k=1.0)
    # Circular orbit conditions
    r0 = 1.0
    v_circ = np.sqrt(cf.k / (cf.mass * r0))

    q0 = np.array([r0, 0.0])
    q_dot0 = np.array([0.0, v_circ * 1.1 / r0])  # Slightly faster -> ellipse

    t_cf, state_cf = cf.simulate(q0, q_dot0, (0, 20), 0.01)

    r = state_cf[:, 0]
    phi = state_cf[:, 1]
    x = r * np.cos(phi)
    y = r * np.sin(phi)

    ax.plot(x, y, 'b-', lw=1)
    ax.plot(0, 0, 'yo', markersize=15, label='Central body')
    ax.plot(x[0], y[0], 'go', markersize=8, label='Start')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Central Force Orbit\n'
                 'L = (1/2)m(r_dot^2 + r^2*phi_dot^2) + k/r')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Plot 5: Coupled oscillators
    ax = axes[1, 1]

    osc = CoupledOscillators(m1=1.0, m2=1.0, k1=1.0, k2=1.0, k_coupling=0.3)
    q0 = np.array([1.0, 0.0])  # Only first oscillator displaced
    q_dot0 = np.array([0.0, 0.0])

    t_osc, state_osc = osc.simulate(q0, q_dot0, (0, 30), 0.01)

    ax.plot(t_osc, state_osc[:, 0], 'b-', lw=1, label='x1')
    ax.plot(t_osc, state_osc[:, 1], 'r-', lw=1, label='x2')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Displacement')
    ax.set_title('Coupled Oscillators\n'
                 'Energy transfer via coupling spring')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Summary
    ax = axes[1, 2]
    ax.axis('off')

    summary = """Euler-Lagrange Equations
========================

Given Lagrangian L(q, q_dot, t) = T - V

The equations of motion are:

    d   dL      dL
   -- (----) - ---- = 0
   dt  dq_dot   dq

For each generalized coordinate q_i.

EXAMPLES DEMONSTRATED:
----------------------

1. Simple Pendulum (1 DOF):
   L = (1/2)ml^2*theta_dot^2 + mgl*cos(theta)
   => theta_ddot = -(g/l)*sin(theta)

2. Double Pendulum (2 DOF):
   Complex coupled equations emerge
   automatically from L(theta1, theta2, ...)

3. Central Force (2 DOF, polar):
   L = (1/2)m(r_dot^2 + r^2*phi_dot^2) + k/r
   => Kepler orbits

4. Coupled Oscillators (2 DOF):
   L = (1/2)m1*v1^2 + (1/2)m2*v2^2
     - (1/2)k1*x1^2 - (1/2)k2*x2^2
     - (1/2)k_c*(x2-x1)^2
   => Normal modes and energy transfer

The Lagrangian formulation is powerful because:
- Works in any coordinate system
- Automatically handles constraints
- Conserved quantities emerge naturally
- Foundation of quantum mechanics (path integral)"""

    ax.text(0.02, 0.98, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Deriving Equations of Motion from the Lagrangian\n'
                 'd/dt(dL/dq_dot) - dL/dq = 0',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'lagrangian_eom.png'), dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/lagrangian_eom.png")


if __name__ == "__main__":
    main()
