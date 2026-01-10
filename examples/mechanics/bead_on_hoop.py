"""
Experiment 33: Bead on rotating hoop - stability bifurcation vs rotation speed.

Demonstrates how a rotating hoop exhibits a pitchfork bifurcation
as the rotation speed increases past a critical value.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def effective_potential(theta, omega, g, R):
    """
    Effective potential for bead on rotating hoop.

    V_eff = -m*g*R*cos(θ) - (1/2)*m*ω²*R²*sin²(θ)

    Normalized: V_eff/mgR = -cos(θ) - (ω²R/2g)*sin²(θ)
    """
    gamma = omega**2 * R / g  # Dimensionless parameter
    return -np.cos(theta) - 0.5 * gamma * np.sin(theta)**2


def find_equilibria(omega, g, R):
    """
    Find equilibrium positions.

    dV/dθ = 0 gives: sin(θ)[1 - γ*cos(θ)] = 0

    Solutions:
    - θ = 0 always (bottom)
    - θ = π always (top, unstable)
    - cos(θ) = 1/γ if γ > 1
    """
    gamma = omega**2 * R / g

    equilibria = [0.0, np.pi]  # Bottom and top

    if gamma > 1:
        # Additional equilibria
        theta_stable = np.arccos(1 / gamma)
        equilibria.append(theta_stable)
        equilibria.append(-theta_stable)

    return np.array(equilibria), gamma


def bead_eom(state, t, omega, g, R, b=0.1):
    """
    Equations of motion for bead on rotating hoop with damping.

    θ'' = sin(θ)[γ*cos(θ) - 1] - b*θ'

    where γ = ω²R/g
    """
    theta, theta_dot = state
    gamma = omega**2 * R / g

    theta_ddot = np.sin(theta) * (gamma * np.cos(theta) - 1) - b * theta_dot

    return np.array([theta_dot, theta_ddot])


def simulate_bead(theta0, omega, g, R, t_max, dt, b=0.1):
    """Simulate bead motion using RK4."""
    n_steps = int(t_max / dt)
    t = np.linspace(0, t_max, n_steps)

    state = np.array([theta0, 0.0])  # Start at rest
    thetas = [state[0]]
    theta_dots = [state[1]]

    for i in range(n_steps - 1):
        # RK4
        k1 = bead_eom(state, t[i], omega, g, R, b)
        k2 = bead_eom(state + dt/2 * k1, t[i] + dt/2, omega, g, R, b)
        k3 = bead_eom(state + dt/2 * k2, t[i] + dt/2, omega, g, R, b)
        k4 = bead_eom(state + dt * k3, t[i] + dt, omega, g, R, b)

        state = state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

        thetas.append(state[0])
        theta_dots.append(state[1])

    return t, np.array(thetas), np.array(theta_dots)


def main():
    # Parameters
    g = 9.81
    R = 1.0

    # Critical angular velocity
    omega_c = np.sqrt(g / R)

    # Different rotation speeds
    omegas = [0.5 * omega_c, omega_c, 1.5 * omega_c, 2.0 * omega_c]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    theta = np.linspace(-np.pi, np.pi, 200)

    # Plot 1: Effective potential for different ω
    ax = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(omegas)))

    for omega, color in zip(omegas, colors):
        V = effective_potential(theta, omega, g, R)
        gamma = omega**2 * R / g
        ax.plot(np.degrees(theta), V, '-', color=color, lw=2,
                label=f'ω/ω_c = {omega/omega_c:.1f} (γ={gamma:.1f})')

    ax.set_xlabel('θ (degrees)')
    ax.set_ylabel('V_eff / (mgR)')
    ax.set_title('Effective Potential')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: Bifurcation diagram
    ax = axes[0, 1]

    omega_range = np.linspace(0, 3 * omega_c, 200)
    stable_bottom = []
    unstable_bottom = []
    stable_off_center = []

    for omega in omega_range:
        gamma = omega**2 * R / g

        if gamma < 1:
            stable_bottom.append(0)
        else:
            unstable_bottom.append(0)
            theta_eq = np.arccos(1 / gamma)
            stable_off_center.append(theta_eq)

    # Plot branches
    ax.plot(omega_range[:len(stable_bottom)] / omega_c, stable_bottom, 'b-', lw=3,
            label='Stable (bottom)')
    ax.plot(omega_range[len(stable_bottom):] / omega_c, unstable_bottom, 'b--', lw=2,
            label='Unstable (bottom)')

    omega_super = omega_range[len(stable_bottom):]
    ax.plot(omega_super / omega_c, stable_off_center, 'r-', lw=3,
            label='Stable (off-center)')
    ax.plot(omega_super / omega_c, [-t for t in stable_off_center], 'r-', lw=3)

    ax.axvline(1.0, color='gray', linestyle=':', label='Critical ω/ω_c = 1')
    ax.set_xlabel('ω / ω_c')
    ax.set_ylabel('Equilibrium θ (rad)')
    ax.set_title('Pitchfork Bifurcation')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 3: Time evolution below critical
    ax = axes[0, 2]
    omega_sub = 0.8 * omega_c
    t, theta_t, _ = simulate_bead(0.3, omega_sub, g, R, t_max=20, dt=0.01)
    ax.plot(t, np.degrees(theta_t), 'b-', lw=1.5)
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('θ (degrees)')
    ax.set_title(f'Below Critical: ω/ω_c = {omega_sub/omega_c:.1f}')
    ax.grid(True, alpha=0.3)

    # Plot 4: Time evolution above critical
    ax = axes[1, 0]
    omega_super = 1.5 * omega_c
    t, theta_t, _ = simulate_bead(0.3, omega_super, g, R, t_max=20, dt=0.01)

    # Theoretical equilibrium
    gamma = omega_super**2 * R / g
    theta_eq = np.degrees(np.arccos(1 / gamma))

    ax.plot(t, np.degrees(theta_t), 'r-', lw=1.5)
    ax.axhline(theta_eq, color='green', linestyle='--', label=f'θ_eq = {theta_eq:.1f}°')
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('θ (degrees)')
    ax.set_title(f'Above Critical: ω/ω_c = {omega_super/omega_c:.1f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Phase portraits
    ax = axes[1, 1]

    for omega, color in zip([0.8 * omega_c, 1.5 * omega_c], ['blue', 'red']):
        for theta0 in np.linspace(-2.5, 2.5, 10):
            t, theta_t, theta_dot_t = simulate_bead(theta0, omega, g, R,
                                                      t_max=30, dt=0.01, b=0.05)
            ax.plot(np.degrees(theta_t), theta_dot_t, '-', color=color, lw=0.5, alpha=0.5)

    ax.set_xlabel('θ (degrees)')
    ax.set_ylabel('dθ/dt (rad/s)')
    ax.set_title('Phase Portraits (blue: ω<ω_c, red: ω>ω_c)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-180, 180)

    # Plot 6: Summary
    ax = axes[1, 2]
    ax.axis('off')

    summary = f"""Bead on Rotating Hoop
=====================
Setup:
  Hoop radius R = {R} m
  Rotation rate ω
  Critical: ω_c = √(g/R) = {omega_c:.2f} rad/s

Effective Potential:
  V_eff = -mgR[cos(θ) + (γ/2)sin²(θ)]
  where γ = ω²R/g

Equilibria:
  θ = 0 (bottom): always exists
    - stable if ω < ω_c
    - unstable if ω > ω_c

  θ = ±arccos(1/γ): exists if ω > ω_c
    - always stable

Bifurcation:
  Supercritical pitchfork at ω = ω_c

  ω < ω_c: single stable equilibrium at θ=0
  ω > ω_c: two stable equilibria appear
           symmetrically off-center

Physical Interpretation:
  Centrifugal force pushes bead outward.
  Above critical ω, outward force
  overcomes gravity at the bottom."""

    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.suptitle('Bead on Rotating Hoop: Bifurcation Analysis',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'bead_on_hoop.png'), dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/bead_on_hoop.png")


if __name__ == "__main__":
    main()
