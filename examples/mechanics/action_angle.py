"""
Experiment 62: Action-Angle Variables.

Demonstrates the transformation to action-angle variables, which are
particularly useful for integrable systems and perturbation theory.

Key concepts:
1. Action variable J = (1/2pi) * integral(p dq) over one period
2. Angle variable theta increases linearly: theta = omega * t
3. Action is an adiabatic invariant (conserved under slow changes)
4. Powerful for the Kepler problem and other integrable systems

For a harmonic oscillator:
    J = E / omega
    theta = omega * t
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


def action_integral_sho(E, m, k):
    """
    Compute action variable for simple harmonic oscillator.

    J = (1/2pi) * integral(p dq) = (1/2pi) * integral(sqrt(2m(E - V)) dq)

    For SHO: J = E / omega = E * sqrt(m/k)

    Args:
        E: Total energy
        m: Mass
        k: Spring constant

    Returns:
        Action variable J
    """
    omega = np.sqrt(k / m)
    return E / omega


def action_from_trajectory(q, p):
    """
    Compute action by integrating p dq around closed orbit.

    J = (1/2pi) * integral(p dq)

    Args:
        q: Array of position values over one cycle
        p: Array of momentum values

    Returns:
        Action variable J
    """
    # Use trapezoidal integration
    dq = np.diff(q)
    p_avg = 0.5 * (p[:-1] + p[1:])
    integral = np.sum(p_avg * dq)

    # For a closed orbit, add contribution to complete the loop
    integral += 0.5 * (p[-1] + p[0]) * (q[0] - q[-1])

    return integral / (2 * np.pi)


def simulate_sho(E, m, k, t_max, dt=0.001):
    """Simulate simple harmonic oscillator."""
    omega = np.sqrt(k / m)
    A = np.sqrt(2 * E / k)  # Amplitude from energy

    # Initial conditions: start at turning point
    q0 = A
    p0 = 0.0

    state = np.array([q0, p0])
    times = [0.0]
    qs = [q0]
    ps = [p0]

    def derivatives(s):
        q, p = s
        return np.array([p/m, -k*q])

    t = 0
    while t < t_max:
        k1 = derivatives(state)
        k2 = derivatives(state + 0.5*dt*k1)
        k3 = derivatives(state + 0.5*dt*k2)
        k4 = derivatives(state + dt*k3)
        state = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

        t += dt
        times.append(t)
        qs.append(state[0])
        ps.append(state[1])

    return np.array(times), np.array(qs), np.array(ps)


def simulate_sho_adiabatic(E0, m, k0, k_final, t_ramp, t_total, dt=0.001):
    """
    Simulate SHO with slowly changing spring constant.

    Tests adiabatic invariance of the action.
    """
    omega0 = np.sqrt(k0 / m)
    A0 = np.sqrt(2 * E0 / k0)

    state = np.array([A0, 0.0])
    times = [0.0]
    qs = [A0]
    ps = [0.0]
    ks = [k0]
    Es = [E0]
    Js = [E0 / omega0]

    def k_of_t(t):
        """Spring constant changes linearly during ramp."""
        if t < t_ramp:
            return k0 + (k_final - k0) * (t / t_ramp)
        else:
            return k_final

    def derivatives(s, k):
        q, p = s
        return np.array([p/m, -k*q])

    t = 0
    while t < t_total:
        k = k_of_t(t)

        k1 = derivatives(state, k)
        k2 = derivatives(state + 0.5*dt*k1, k)
        k3 = derivatives(state + 0.5*dt*k2, k)
        k4 = derivatives(state + dt*k3, k)
        state = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

        t += dt
        times.append(t)
        qs.append(state[0])
        ps.append(state[1])
        ks.append(k)

        # Calculate energy and action
        E = 0.5*ps[-1]**2/m + 0.5*k*qs[-1]**2
        omega = np.sqrt(k/m)
        J = E / omega

        Es.append(E)
        Js.append(J)

    return {
        'times': np.array(times),
        'q': np.array(qs),
        'p': np.array(ps),
        'k': np.array(ks),
        'E': np.array(Es),
        'J': np.array(Js)
    }


def kepler_action(E, L, m, k):
    """
    Compute radial action for Kepler problem.

    For bound orbits (E < 0):
    J_r = -L + k * sqrt(m / (-2E))

    where k = G*M*m for gravity.

    The total action J_total = J_r + |L| gives:
    J_total = k * sqrt(m / (-2E)) = k * sqrt(m/|2E|)

    This leads to E = -m*k^2 / (2*(J_r + |L|)^2)

    Args:
        E: Total energy (negative for bound orbit)
        L: Angular momentum magnitude
        m: Reduced mass
        k: Force constant (G*M*m)

    Returns:
        Radial action J_r
    """
    if E >= 0:
        return np.inf  # Unbound orbit

    J_r = -L + k * np.sqrt(m / (-2 * E))
    return J_r


def simulate_kepler(E, L, m, k, t_max, dt=0.001):
    """
    Simulate Kepler problem in polar coordinates.

    Using effective potential approach.
    """
    # Turning points from effective potential
    # E = 0.5*m*r_dot^2 + L^2/(2*m*r^2) - k/r
    # At turning points, r_dot = 0

    # Find r_min and r_max by solving quadratic
    # E = L^2/(2*m*r^2) - k/r
    # => E*r^2 + k*r - L^2/(2m) = 0

    a_coef = E
    b_coef = k
    c_coef = -L**2 / (2*m)

    disc = b_coef**2 - 4*a_coef*c_coef
    if disc < 0 or E >= 0:
        print("Invalid orbit parameters")
        return None

    r1 = (-b_coef + np.sqrt(disc)) / (2*a_coef)
    r2 = (-b_coef - np.sqrt(disc)) / (2*a_coef)

    r_min = min(r1, r2)
    r_max = max(r1, r2)

    # Start at r_max (aphelion)
    r = r_max
    r_dot = 0.0
    phi = 0.0
    phi_dot = L / (m * r**2)

    state = np.array([r, phi, r_dot, phi_dot])

    times = [0.0]
    rs = [r]
    phis = [phi]
    r_dots = [r_dot]
    phi_dots = [phi_dot]

    def derivatives(s):
        r, phi, r_dot, phi_dot = s
        r_ddot = r * phi_dot**2 - k / (m * r**2)
        phi_ddot = -2 * r_dot * phi_dot / r
        return np.array([r_dot, phi_dot, r_ddot, phi_ddot])

    t = 0
    while t < t_max:
        k1 = derivatives(state)
        k2 = derivatives(state + 0.5*dt*k1)
        k3 = derivatives(state + 0.5*dt*k2)
        k4 = derivatives(state + dt*k3)
        state = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

        t += dt
        times.append(t)
        rs.append(state[0])
        phis.append(state[1])
        r_dots.append(state[2])
        phi_dots.append(state[3])

    return {
        'times': np.array(times),
        'r': np.array(rs),
        'phi': np.array(phis),
        'r_dot': np.array(r_dots),
        'phi_dot': np.array(phi_dots),
        'r_min': r_min,
        'r_max': r_max
    }


def main():
    fig = plt.figure(figsize=(16, 12))

    # Parameters
    m = 1.0
    k = 4.0  # omega = 2

    # --- Plot 1: Action-angle for SHO ---
    ax1 = fig.add_subplot(2, 3, 1)

    energies = [0.5, 1.0, 2.0, 4.0]
    colors = ['blue', 'green', 'orange', 'red']

    omega = np.sqrt(k / m)
    period = 2 * np.pi / omega

    for E, color in zip(energies, colors):
        times, qs, ps = simulate_sho(E, m, k, 2*period)

        ax1.plot(qs, ps, color=color, lw=1.5, label=f'E={E:.1f}, J={E/omega:.2f}')

        # Compute action from trajectory
        # Find one complete period
        period_idx = int(period / 0.001)
        J_computed = action_from_trajectory(qs[:period_idx+1], ps[:period_idx+1])

    ax1.set_xlabel('q (position)')
    ax1.set_ylabel('p (momentum)')
    ax1.set_title('Phase Space Orbits\nAction J = E/omega = area/(2*pi)')
    ax1.legend(fontsize=8)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Angle variable theta = omega*t ---
    ax2 = fig.add_subplot(2, 3, 2)

    E = 2.0
    times, qs, ps = simulate_sho(E, m, k, 4*period)

    # Compute angle variable from q, p
    # For SHO: q = A*cos(theta), p = -m*omega*A*sin(theta)
    # => theta = atan2(-p/(m*omega*A), q/A) = atan2(-p/(m*omega), q)

    A = np.sqrt(2*E/k)
    theta_computed = np.arctan2(-ps/(m*omega), qs)
    theta_computed = np.unwrap(theta_computed)  # Remove 2pi jumps

    # Theoretical: theta = omega * t (with phase offset)
    theta_theory = omega * times + theta_computed[0]

    ax2.plot(times, theta_computed, 'b-', lw=2, label='Computed from (q,p)')
    ax2.plot(times, theta_theory, 'r--', lw=2, label='theta = omega*t')

    ax2.set_xlabel('Time')
    ax2.set_ylabel('Angle theta (rad)')
    ax2.set_title('Angle Variable\ntheta increases linearly with time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # --- Plot 3: Adiabatic invariance of action ---
    ax3 = fig.add_subplot(2, 3, 3)

    E0 = 2.0
    k0 = 1.0
    k_final = 4.0
    t_ramp = 50.0  # Slow change
    t_total = 100.0

    result = simulate_sho_adiabatic(E0, m, k0, k_final, t_ramp, t_total)

    ax3_twin = ax3.twinx()

    ax3.plot(result['times'], result['E'], 'b-', lw=1.5, label='Energy E')
    ax3.plot(result['times'], result['J'], 'r-', lw=2, label='Action J')
    ax3_twin.plot(result['times'], result['k'], 'g--', lw=1.5, alpha=0.5)

    ax3.set_xlabel('Time')
    ax3.set_ylabel('E (blue) and J (red)', color='blue')
    ax3_twin.set_ylabel('Spring constant k', color='green')
    ax3.set_title('Adiabatic Invariance\nAction J conserved as k changes slowly')
    ax3.legend(loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Mark the ramp region
    ax3.axvspan(0, t_ramp, alpha=0.1, color='yellow', label='Ramp region')

    # --- Plot 4: Fast vs slow parameter change ---
    ax4 = fig.add_subplot(2, 3, 4)

    ramp_times = [5.0, 20.0, 50.0, 100.0]
    colors = ['red', 'orange', 'green', 'blue']

    for t_ramp, color in zip(ramp_times, colors):
        result = simulate_sho_adiabatic(E0, m, k0, k_final, t_ramp, 2*t_ramp)
        J_relative = result['J'] / result['J'][0]
        ax4.plot(result['times']/t_ramp, J_relative, color=color, lw=1.5,
                 label=f't_ramp = {t_ramp}')

    ax4.axhline(1.0, color='k', linestyle='--', alpha=0.5)
    ax4.set_xlabel('t / t_ramp')
    ax4.set_ylabel('J / J_0')
    ax4.set_title('Action Conservation vs Ramp Speed\n(Slower = better conservation)')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0.8, 1.2)

    # --- Plot 5: Kepler problem action-angle ---
    ax5 = fig.add_subplot(2, 3, 5)

    # Kepler parameters (units where G*M = 1)
    m_kepler = 1.0
    k_kepler = 1.0  # G*M*m = 1

    # Different orbits
    L = 0.8
    energies_kepler = [-0.5, -0.4, -0.3, -0.25]
    colors = ['blue', 'green', 'orange', 'red']

    for E, color in zip(energies_kepler, colors):
        result = simulate_kepler(E, L, m_kepler, k_kepler, 30)
        if result is not None:
            x = result['r'] * np.cos(result['phi'])
            y = result['r'] * np.sin(result['phi'])

            J_r = kepler_action(E, L, m_kepler, k_kepler)

            ax5.plot(x, y, color=color, lw=1.5, label=f'E={E:.2f}, J_r={J_r:.2f}')

    ax5.plot(0, 0, 'yo', markersize=15)
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    ax5.set_title('Kepler Orbits\nDifferent energies, same L')
    ax5.legend(fontsize=8)
    ax5.set_aspect('equal')
    ax5.grid(True, alpha=0.3)

    # --- Plot 6: Theory summary ---
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    summary = """Action-Angle Variables
======================

ACTION VARIABLE:
----------------
    J = (1/2pi) * integral(p dq)

The integral is over one complete period of motion.
For SHO: J = E / omega

ANGLE VARIABLE:
---------------
    theta = omega * t    (modulo 2*pi)

where omega = dH/dJ (frequency of motion)

For SHO: theta = arctan(-p/(m*omega*q))

IN ACTION-ANGLE COORDINATES:
----------------------------
    H = H(J)       (no theta dependence)
    dJ/dt = 0      (action is constant)
    dtheta/dt = omega(J)  (angle increases uniformly)

ADIABATIC INVARIANCE:
---------------------
When system parameters change SLOWLY compared
to the period of motion, the action J is conserved
even though the energy E changes!

Example: pendulum with slowly changing length
    omega ~ sqrt(g/L) changes
    E = J * omega changes
    But J stays constant!

KEPLER PROBLEM:
---------------
Two actions: J_r (radial) and L (angular momentum)

    J_r = -L + k*sqrt(m/(-2E))

Energy depends only on total action:
    E = -m*k^2 / (2*(J_r + |L|)^2)

This leads to quantization in quantum mechanics!

APPLICATIONS:
-------------
- Perturbation theory (small deviations from integrable)
- KAM theorem (stability of tori)
- Semiclassical quantization (Bohr-Sommerfeld)
- Chaos theory (action diffusion)"""

    ax6.text(0.02, 0.98, summary, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('Action-Angle Variables (Experiment 62)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'action_angle.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {output_path}")

    # Print numerical results
    print("\nAction-Angle Variable Analysis:")
    print("-" * 50)

    omega = np.sqrt(k / m)
    print(f"SHO parameters: m={m}, k={k}, omega={omega:.3f}")

    for E in energies:
        J_theory = E / omega
        times, qs, ps = simulate_sho(E, m, k, 2*np.pi/omega)
        period_idx = int((2*np.pi/omega) / 0.001)
        J_computed = action_from_trajectory(qs[:period_idx+1], ps[:period_idx+1])
        print(f"  E={E:.1f}: J_theory={J_theory:.4f}, J_computed={J_computed:.4f}")

    print(f"\nAdiabatic invariance test:")
    result_slow = simulate_sho_adiabatic(E0, m, k0, k_final, 100.0, 200.0)
    result_fast = simulate_sho_adiabatic(E0, m, k0, k_final, 5.0, 20.0)
    print(f"  Slow ramp: J_final/J_initial = {result_slow['J'][-1]/result_slow['J'][0]:.4f}")
    print(f"  Fast ramp: J_final/J_initial = {result_fast['J'][-1]/result_fast['J'][0]:.4f}")


if __name__ == "__main__":
    main()
