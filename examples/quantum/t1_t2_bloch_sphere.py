"""
Experiment 176: T1/T2 Relaxation on the Bloch Sphere

Visualizes qubit decoherence processes (T1 and T2) on the Bloch sphere,
showing how the Bloch vector evolves under different decoherence mechanisms.

Physics:
    The Bloch vector r = (rx, ry, rz) describes the qubit state:
    rho = (I + r . sigma) / 2

    T1 (longitudinal relaxation):
    - Bloch vector component rz decays toward equilibrium
    - drz/dt = -(rz - rz_eq) / T1
    - rx, ry also decay due to energy relaxation

    T2 (transverse relaxation):
    - Bloch vector components rx, ry decay
    - drx/dt = -rx/T2, dry/dt = -ry/T2
    - T2 <= 2*T1 always (pure dephasing contribution)

    Combined dynamics:
    drx/dt = -rx/T2 - omega*ry
    dry/dt = -ry/T2 + omega*rx
    drz/dt = -(rz - rz_eq)/T1
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def bloch_equations(r, t, T1, T2, omega=0.0, rz_eq=-1.0):
    """
    Bloch equations for qubit relaxation.

    drx/dt = -rx/T2 - omega*ry
    dry/dt = -ry/T2 + omega*rx
    drz/dt = -(rz - rz_eq)/T1

    Args:
        r: Bloch vector [rx, ry, rz]
        t: Time (not used, autonomous system)
        T1: Longitudinal relaxation time
        T2: Transverse relaxation time
        omega: Larmor precession frequency
        rz_eq: Equilibrium z-component (thermal)

    Returns:
        dr/dt
    """
    rx, ry, rz = r

    drx = -rx / T2 - omega * ry
    dry = -ry / T2 + omega * rx
    drz = -(rz - rz_eq) / T1

    return np.array([drx, dry, drz])


def evolve_bloch(r0, T1, T2, t_max, dt, omega=0.0, rz_eq=-1.0):
    """
    Integrate Bloch equations.

    Args:
        r0: Initial Bloch vector
        T1, T2: Relaxation times
        t_max: Maximum time
        dt: Time step
        omega: Precession frequency
        rz_eq: Equilibrium z

    Returns:
        times, trajectory (array of Bloch vectors)
    """
    times = [0]
    trajectory = [r0.copy()]

    r = r0.copy()
    t = 0

    while t < t_max:
        dr = bloch_equations(r, t, T1, T2, omega, rz_eq)
        r = r + dr * dt
        t += dt

        times.append(t)
        trajectory.append(r.copy())

    return np.array(times), np.array(trajectory)


def draw_bloch_sphere(ax, alpha=0.1):
    """Draw a Bloch sphere wireframe."""
    # Sphere
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x, y, z, alpha=alpha, color='lightblue')

    # Equator
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), np.zeros_like(theta), 'k-', alpha=0.3)

    # Meridians
    for phi in [0, np.pi/2]:
        ax.plot(np.cos(phi)*np.sin(v), np.sin(phi)*np.sin(v), np.cos(v), 'k-', alpha=0.3)

    # Axes
    ax.plot([-1.2, 1.2], [0, 0], [0, 0], 'k-', lw=0.5, alpha=0.5)
    ax.plot([0, 0], [-1.2, 1.2], [0, 0], 'k-', lw=0.5, alpha=0.5)
    ax.plot([0, 0], [0, 0], [-1.2, 1.2], 'k-', lw=0.5, alpha=0.5)

    # Labels
    ax.text(1.3, 0, 0, 'x', fontsize=10)
    ax.text(0, 1.3, 0, 'y', fontsize=10)
    ax.text(0, 0, 1.3, '|0>', fontsize=10)
    ax.text(0, 0, -1.3, '|1>', fontsize=10)


def main():
    fig = plt.figure(figsize=(16, 12))

    # Initial state: |+> (equator, x-axis)
    r0_plus = np.array([1.0, 0.0, 0.0])
    # Initial state: superposition tilted up
    r0_tilted = np.array([0.5, 0.5, 0.707])

    dt = 0.01

    # ===== Plot 1: Pure T2 dephasing =====
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    draw_bloch_sphere(ax1)

    T2_values = [2.0, 5.0, 10.0]
    colors = ['red', 'green', 'blue']

    for T2, color in zip(T2_values, colors):
        times, traj = evolve_bloch(r0_plus, T1=np.inf, T2=T2, t_max=15, dt=dt, omega=1.0)
        ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=color, lw=2,
                label=f'T2 = {T2}', alpha=0.8)
        ax1.scatter([traj[0, 0]], [traj[0, 1]], [traj[0, 2]], color=color, s=50, marker='o')
        ax1.scatter([traj[-1, 0]], [traj[-1, 1]], [traj[-1, 2]], color=color, s=50, marker='x')

    ax1.set_title('Pure T2 Dephasing\n(Spiral toward z-axis)')
    ax1.legend(loc='upper left')
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-1.2, 1.2)
    ax1.set_zlim(-1.2, 1.2)

    # ===== Plot 2: Pure T1 relaxation =====
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    draw_bloch_sphere(ax2)

    T1_values = [2.0, 5.0, 10.0]
    r0_excited = np.array([0.0, 0.0, 1.0])  # |0> state

    for T1, color in zip(T1_values, colors):
        # T2 = 2*T1 (minimum allowed by physics)
        times, traj = evolve_bloch(r0_excited, T1=T1, T2=2*T1, t_max=20, dt=dt, omega=0.0, rz_eq=-1.0)
        ax2.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=color, lw=2,
                label=f'T1 = {T1}', alpha=0.8)
        ax2.scatter([traj[0, 0]], [traj[0, 1]], [traj[0, 2]], color=color, s=50, marker='o')
        ax2.scatter([traj[-1, 0]], [traj[-1, 1]], [traj[-1, 2]], color=color, s=50, marker='x')

    ax2.set_title('Pure T1 Relaxation\n(Decay from |0> to |1>)')
    ax2.legend(loc='upper left')
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_zlim(-1.2, 1.2)

    # ===== Plot 3: Combined T1 and T2 =====
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    draw_bloch_sphere(ax3)

    # Starting from |+> with both relaxation processes
    T1 = 10.0
    T2_values = [5.0, 10.0, 20.0]  # T2 <= 2*T1

    for T2, color in zip(T2_values, colors):
        times, traj = evolve_bloch(r0_plus, T1=T1, T2=T2, t_max=30, dt=dt, omega=0.5, rz_eq=-1.0)
        ax3.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=color, lw=2,
                label=f'T2 = {T2}', alpha=0.8)
        ax3.scatter([traj[0, 0]], [traj[0, 1]], [traj[0, 2]], color=color, s=50, marker='o')
        ax3.scatter([traj[-1, 0]], [traj[-1, 1]], [traj[-1, 2]], color=color, s=50, marker='x')

    ax3.set_title(f'Combined T1={T1} and T2 Relaxation\n(Spiral to ground state)')
    ax3.legend(loc='upper left')
    ax3.set_xlim(-1.2, 1.2)
    ax3.set_ylim(-1.2, 1.2)
    ax3.set_zlim(-1.2, 1.2)

    # ===== Plot 4: Time evolution of Bloch components =====
    ax4 = fig.add_subplot(2, 2, 4)

    T1, T2 = 10.0, 5.0
    times, traj = evolve_bloch(r0_plus, T1=T1, T2=T2, t_max=30, dt=dt, omega=1.0, rz_eq=-1.0)

    ax4.plot(times, traj[:, 0], 'r-', lw=2, label='r_x')
    ax4.plot(times, traj[:, 1], 'g-', lw=2, label='r_y')
    ax4.plot(times, traj[:, 2], 'b-', lw=2, label='r_z')

    # Magnitude
    r_mag = np.sqrt(traj[:, 0]**2 + traj[:, 1]**2 + traj[:, 2]**2)
    ax4.plot(times, r_mag, 'k--', lw=2, label='|r|')

    # Transverse magnitude
    r_xy = np.sqrt(traj[:, 0]**2 + traj[:, 1]**2)
    ax4.plot(times, r_xy, 'm:', lw=2, label='r_xy')

    # Theoretical envelopes
    ax4.plot(times, np.exp(-times/T2), 'c--', lw=1, alpha=0.5, label=f'exp(-t/T2)')
    ax4.plot(times, -1 + 2*np.exp(-times/T1), 'y--', lw=1, alpha=0.5, label='rz_eq + exp(-t/T1)')

    ax4.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax4.axhline(-1, color='gray', linestyle=':', alpha=0.3, label='Equilibrium')

    ax4.set_xlabel('Time')
    ax4.set_ylabel('Bloch Vector Component')
    ax4.set_title(f'Time Evolution (T1={T1}, T2={T2}, omega=1)')
    ax4.legend(fontsize=8, ncol=2)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 30)
    ax4.set_ylim(-1.1, 1.1)

    plt.suptitle('T1 and T2 Relaxation on the Bloch Sphere\n'
                 'Visualizing Qubit Decoherence',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 't1_t2_bloch_sphere.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 't1_t2_bloch_sphere.png')}")

    # Print summary
    print("\n=== T1/T2 Relaxation Summary ===")
    print("\nT1 (Longitudinal/Energy Relaxation):")
    print("  - Decay of population difference")
    print("  - rz -> rz_eq exponentially")
    print("  - Physical: spontaneous emission, thermal equilibration")

    print("\nT2 (Transverse/Phase Relaxation):")
    print("  - Decay of quantum coherence")
    print("  - rx, ry -> 0 exponentially")
    print("  - Physical: dephasing, T1 contribution")
    print("  - T2 <= 2*T1 always!")

    print("\nTypical values in superconducting qubits:")
    print("  - T1 ~ 10-100 microseconds")
    print("  - T2 ~ 1-50 microseconds")
    print("  - T2/T1 ~ 0.1-1 (often limited by dephasing)")


if __name__ == "__main__":
    main()
