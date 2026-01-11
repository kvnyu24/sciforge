"""
Experiment 34: Atwood Machine

Simulates the classic Atwood machine with pulley inertia.

Physical concepts:
- Two masses connected by a string over a pulley
- Without pulley inertia: a = g*(m1-m2)/(m1+m2)
- With pulley inertia: a = g*(m1-m2)/(m1+m2+I/R^2)
- Tension: T1 = m1*(g-a), T2 = m2*(g+a)
- Energy conservation with rotational KE

The pulley's moment of inertia reduces the system's acceleration.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt


def atwood_acceleration_no_inertia(m1, m2, g=9.81):
    """Acceleration without pulley inertia."""
    return g * (m1 - m2) / (m1 + m2)


def atwood_acceleration_with_inertia(m1, m2, I, R, g=9.81):
    """Acceleration with pulley moment of inertia I and radius R."""
    return g * (m1 - m2) / (m1 + m2 + I / R**2)


def atwood_tensions(m1, m2, a, g=9.81):
    """Calculate tensions in the string on each side."""
    T1 = m1 * (g - a)  # Tension on heavier mass side
    T2 = m2 * (g + a)  # Tension on lighter mass side
    return T1, T2


def simulate_atwood(m1, m2, I, R, t_max, dt=0.001, g=9.81):
    """
    Simulate Atwood machine dynamics.

    Returns time, position of m1, velocity, acceleration.
    """
    t = np.arange(0, t_max, dt)
    n = len(t)

    # Acceleration is constant
    if I > 0:
        a = atwood_acceleration_with_inertia(m1, m2, I, R, g)
    else:
        a = atwood_acceleration_no_inertia(m1, m2, g)

    # Kinematic equations
    v = a * t  # velocity
    y = 0.5 * a * t**2  # displacement

    # Energy tracking
    KE_m1 = 0.5 * m1 * v**2
    KE_m2 = 0.5 * m2 * v**2
    KE_rot = 0.5 * I * (v / R)**2 if I > 0 else np.zeros_like(v)

    # Potential energy (relative to initial position)
    PE = -m1 * g * y + m2 * g * y  # m1 goes down, m2 goes up

    return t, y, v, a, KE_m1, KE_m2, KE_rot, PE


def simulate_atwood_numerical(m1, m2, I, R, t_max, dt=0.001, g=9.81):
    """
    Numerical simulation using RK4 (for verification and extension to friction).
    """
    t = np.arange(0, t_max, dt)
    n = len(t)

    y = np.zeros(n)  # position of m1
    v = np.zeros(n)  # velocity

    # Effective mass including pulley
    m_eff = m1 + m2 + I / R**2

    def dv_dt(v_curr):
        return g * (m1 - m2) / m_eff

    for i in range(1, n):
        # RK4 for velocity
        k1 = dv_dt(v[i-1])
        k2 = dv_dt(v[i-1] + 0.5 * dt * k1)
        k3 = dv_dt(v[i-1] + 0.5 * dt * k2)
        k4 = dv_dt(v[i-1] + dt * k3)
        v[i] = v[i-1] + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

        # Position update
        y[i] = y[i-1] + v[i-1] * dt + 0.5 * dv_dt(v[i-1]) * dt**2

    return t, y, v


def main():
    """Run Atwood machine experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Parameters
    m1 = 2.0  # kg (heavier mass)
    m2 = 1.5  # kg (lighter mass)
    g = 9.81  # m/s^2

    # Pulley parameters
    R = 0.1  # radius (m)
    M_pulley = 0.5  # pulley mass (kg)
    I_disk = 0.5 * M_pulley * R**2  # solid disk
    I_ring = M_pulley * R**2  # thin ring

    t_max = 3.0  # seconds

    # Plot 1: Position vs time for different pulley types
    ax1 = axes[0, 0]

    # No pulley inertia
    t, y_no_I, v_no_I, a_no_I, _, _, _, _ = simulate_atwood(m1, m2, 0, R, t_max)
    ax1.plot(t, y_no_I, 'b-', lw=2, label=f'Massless pulley (a={a_no_I:.3f} m/s$^2$)')

    # Disk pulley
    t, y_disk, v_disk, a_disk, _, _, _, _ = simulate_atwood(m1, m2, I_disk, R, t_max)
    ax1.plot(t, y_disk, 'r-', lw=2, label=f'Solid disk (a={a_disk:.3f} m/s$^2$)')

    # Ring pulley
    t, y_ring, v_ring, a_ring, _, _, _, _ = simulate_atwood(m1, m2, I_ring, R, t_max)
    ax1.plot(t, y_ring, 'g-', lw=2, label=f'Thin ring (a={a_ring:.3f} m/s$^2$)')

    # Numerical verification
    t_num, y_num, v_num = simulate_atwood_numerical(m1, m2, I_disk, R, t_max)
    ax1.plot(t_num[::100], y_num[::100], 'ko', markersize=4, alpha=0.5,
             label='Numerical (disk)')

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Displacement of $m_1$ (m)')
    ax1.set_title(f'Position vs Time ($m_1$={m1} kg, $m_2$={m2} kg)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Acceleration vs pulley moment of inertia
    ax2 = axes[0, 1]

    I_values = np.linspace(0, 2 * I_ring, 100)
    a_values = [atwood_acceleration_with_inertia(m1, m2, I, R, g) for I in I_values]

    ax2.plot(I_values * 1000, a_values, 'b-', lw=2)  # I in g*cm^2
    ax2.axhline(y=a_no_I, color='r', linestyle='--', label=f'No inertia: a={a_no_I:.3f} m/s$^2$')

    # Mark disk and ring
    ax2.plot(I_disk * 1000, a_disk, 'ro', markersize=10, label=f'Solid disk: a={a_disk:.3f}')
    ax2.plot(I_ring * 1000, a_ring, 'go', markersize=10, label=f'Thin ring: a={a_ring:.3f}')

    ax2.set_xlabel('Moment of Inertia (g$\\cdot$cm$^2$)')
    ax2.set_ylabel('Acceleration (m/s$^2$)')
    ax2.set_title('Acceleration vs Pulley Inertia')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Energy conservation
    ax3 = axes[1, 0]

    t, y, v, a, KE1, KE2, KE_rot, PE = simulate_atwood(m1, m2, I_disk, R, t_max)

    ax3.plot(t, KE1, 'b-', lw=2, label='KE of $m_1$')
    ax3.plot(t, KE2, 'r-', lw=2, label='KE of $m_2$')
    ax3.plot(t, KE_rot, 'g-', lw=2, label='KE of pulley')
    ax3.plot(t, PE, 'm-', lw=2, label='$\\Delta$PE (net)')

    total_E = KE1 + KE2 + KE_rot + PE
    ax3.plot(t, total_E, 'k--', lw=2, label='Total (should be 0)')

    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Energy (J)')
    ax3.set_title('Energy Conservation')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Tensions in the string
    ax4 = axes[1, 1]

    # Mass ratio sweep
    mass_ratios = np.linspace(1.0, 3.0, 50)
    m2_fixed = 1.0

    T1_no_I = []
    T2_no_I = []
    T1_disk = []
    T2_disk = []

    for ratio in mass_ratios:
        m1_var = ratio * m2_fixed

        # No inertia
        a = atwood_acceleration_no_inertia(m1_var, m2_fixed, g)
        T1, T2 = atwood_tensions(m1_var, m2_fixed, a, g)
        T1_no_I.append(T1)
        T2_no_I.append(T2)

        # With disk
        a = atwood_acceleration_with_inertia(m1_var, m2_fixed, I_disk, R, g)
        T1, T2 = atwood_tensions(m1_var, m2_fixed, a, g)
        T1_disk.append(T1)
        T2_disk.append(T2)

    ax4.plot(mass_ratios, T1_no_I, 'b-', lw=2, label='$T_1$ (no inertia)')
    ax4.plot(mass_ratios, T2_no_I, 'b--', lw=2, label='$T_2$ (no inertia)')
    ax4.plot(mass_ratios, T1_disk, 'r-', lw=2, label='$T_1$ (disk pulley)')
    ax4.plot(mass_ratios, T2_disk, 'r--', lw=2, label='$T_2$ (disk pulley)')

    # Equal masses limit (T = mg, a = 0)
    ax4.axhline(y=m2_fixed * g, color='gray', linestyle=':', alpha=0.5)
    ax4.text(1.05, m2_fixed * g + 0.5, f'$m_2 g$ = {m2_fixed * g:.2f} N', fontsize=9)

    ax4.set_xlabel('Mass Ratio $m_1/m_2$')
    ax4.set_ylabel('Tension (N)')
    ax4.set_title(f'String Tensions ($m_2$ = {m2_fixed} kg fixed)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Experiment 34: Atwood Machine with Pulley Inertia\n'
                 '$a = \\frac{(m_1 - m_2)g}{m_1 + m_2 + I/R^2}$',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save output
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'atwood_machine.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'atwood_machine.png')}")

    # Print summary
    print("\n=== Atwood Machine Summary ===")
    print(f"Masses: m1 = {m1} kg, m2 = {m2} kg")
    print(f"Pulley: R = {R*100} cm, M = {M_pulley} kg")
    print(f"\nAccelerations:")
    print(f"  Massless pulley: a = {a_no_I:.4f} m/s^2")
    print(f"  Solid disk:      a = {a_disk:.4f} m/s^2 ({100*(a_no_I-a_disk)/a_no_I:.1f}% reduction)")
    print(f"  Thin ring:       a = {a_ring:.4f} m/s^2 ({100*(a_no_I-a_ring)/a_no_I:.1f}% reduction)")


if __name__ == "__main__":
    main()
