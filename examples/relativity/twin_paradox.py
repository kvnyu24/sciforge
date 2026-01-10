"""
Experiment 191: Twin Paradox with Proper Time Calculation

This experiment demonstrates the twin paradox in special relativity,
showing how the traveling twin ages less than the stay-at-home twin.
The resolution involves understanding the role of acceleration.

Physical concepts:
- Proper time integral: tau = integral sqrt(1 - v^2/c^2) dt
- Time dilation for moving clocks
- Asymmetry due to acceleration phases
- Spacetime diagrams with worldlines
- Comparison of constant velocity vs constant acceleration profiles

The twin paradox is resolved by recognizing that:
1. The situation is NOT symmetric - the traveling twin accelerates
2. The proper time along a worldline is path-dependent
3. The straight worldline (stay-at-home) maximizes proper time
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, cumulative_trapezoid


def lorentz_factor(v, c=1.0):
    """
    Calculate the Lorentz factor gamma = 1/sqrt(1 - v^2/c^2).

    Args:
        v: Velocity (can be array)
        c: Speed of light

    Returns:
        Lorentz factor gamma
    """
    return 1.0 / np.sqrt(1 - (v / c) ** 2)


def proper_time_integrand(v, c=1.0):
    """
    Calculate the proper time integrand sqrt(1 - v^2/c^2) = 1/gamma.

    Args:
        v: Velocity
        c: Speed of light

    Returns:
        Proper time factor (dtau/dt)
    """
    return np.sqrt(1 - (v / c) ** 2)


def constant_velocity_journey(T_total, v, c=1.0, n_points=1000):
    """
    Calculate proper time for a constant velocity round trip.

    The twin travels at velocity v for half the time (outbound),
    then instantaneously reverses and returns at velocity -v.

    Args:
        T_total: Total coordinate time for round trip
        v: Travel velocity (magnitude)
        c: Speed of light
        n_points: Number of time points

    Returns:
        t: Coordinate time array
        x: Position array
        tau: Proper time array
        v_profile: Velocity profile
    """
    t = np.linspace(0, T_total, n_points)
    T_half = T_total / 2

    # Position: outbound then return
    x = np.where(t <= T_half, v * t, v * T_half - v * (t - T_half))

    # Velocity profile (constant magnitude, sign flip at midpoint)
    v_profile = np.where(t <= T_half, v, -v)

    # Proper time: integrate sqrt(1 - v^2/c^2) dt
    # For constant velocity, this is just (1/gamma) * t
    gamma = lorentz_factor(v, c)
    tau = t / gamma

    return t, x, tau, v_profile


def smooth_turnaround_journey(T_total, v_max, t_accel, c=1.0, n_points=1000):
    """
    Calculate proper time for a journey with smooth acceleration phases.

    The journey consists of:
    1. Acceleration from rest to v_max over time t_accel
    2. Coasting at v_max
    3. Deceleration and reversal over time 2*t_accel (at turnaround)
    4. Coasting at -v_max (return)
    5. Deceleration to rest over time t_accel

    Args:
        T_total: Total coordinate time
        v_max: Maximum cruise velocity
        t_accel: Duration of each acceleration/deceleration phase
        c: Speed of light
        n_points: Number of time points

    Returns:
        t: Coordinate time array
        x: Position array
        tau: Proper time array
        v_profile: Velocity profile
    """
    t = np.linspace(0, T_total, n_points)
    dt = t[1] - t[0]

    T_half = T_total / 2
    v_profile = np.zeros_like(t)

    for i, ti in enumerate(t):
        if ti < t_accel:
            # Initial acceleration
            v_profile[i] = v_max * (ti / t_accel)
        elif ti < T_half - t_accel:
            # Coast outbound
            v_profile[i] = v_max
        elif ti < T_half + t_accel:
            # Turnaround (decelerate, reverse, accelerate back)
            phase = (ti - (T_half - t_accel)) / (2 * t_accel)
            v_profile[i] = v_max * (1 - 2 * phase)
        elif ti < T_total - t_accel:
            # Coast inbound
            v_profile[i] = -v_max
        else:
            # Final deceleration
            phase = (ti - (T_total - t_accel)) / t_accel
            v_profile[i] = -v_max * (1 - phase)

    # Integrate velocity to get position
    x = cumulative_trapezoid(v_profile, t, initial=0)

    # Integrate proper time
    dtau_dt = proper_time_integrand(v_profile, c)
    tau = cumulative_trapezoid(dtau_dt, t, initial=0)

    return t, x, tau, v_profile


def hyperbolic_journey(T_total, a, c=1.0, n_points=1000):
    """
    Calculate proper time for a constant proper acceleration journey.

    This is the relativistically correct way to travel:
    1. Accelerate at proper acceleration 'a' toward destination
    2. At midpoint, reverse acceleration (decelerate)
    3. At destination, reverse and repeat for return

    For proper acceleration, the velocity is v = c * tanh(a * tau / c).

    Args:
        T_total: Total coordinate time
        a: Proper acceleration
        c: Speed of light
        n_points: Number of time points

    Returns:
        t: Coordinate time array
        x: Position array
        tau: Proper time array
        v_profile: Velocity profile
    """
    # For hyperbolic motion, we parameterize by proper time
    # Then find the corresponding coordinate time

    # First, find the proper time for the whole journey
    # For a 4-phase journey (accel, decel, accel back, decel)
    # each phase has the same proper time duration

    # Total coordinate time determines the journey
    T_quarter = T_total / 4

    # For hyperbolic motion: t = (c/a) * sinh(a * tau / c)
    # So tau = (c/a) * arcsinh(a * t / c) for each phase

    # But we need to work in coordinate time, so let's use numerical integration
    t = np.linspace(0, T_total, n_points)
    dt = t[1] - t[0]

    # We need to solve for the proper time as we go
    tau = np.zeros_like(t)
    v_profile = np.zeros_like(t)
    x = np.zeros_like(t)

    # Phase tracking
    current_tau = 0
    current_x = 0
    current_v = 0

    for i in range(1, len(t)):
        ti = t[i]

        # Determine which phase we're in and the proper acceleration direction
        if ti < T_quarter:
            # Phase 1: Accelerate outward
            accel_sign = 1
        elif ti < 2 * T_quarter:
            # Phase 2: Decelerate (still moving outward)
            accel_sign = -1
        elif ti < 3 * T_quarter:
            # Phase 3: Accelerate inward (moving back)
            accel_sign = -1
        else:
            # Phase 4: Decelerate (still moving inward)
            accel_sign = 1

        # Update proper time
        gamma = lorentz_factor(current_v, c)
        d_tau = dt / gamma
        current_tau += d_tau
        tau[i] = current_tau

        # Update velocity using relativistic equation of motion
        # dv/dt = a / gamma^3 (coordinate acceleration)
        # For proper acceleration a_proper, coordinate acceleration is a_proper / gamma^3
        a_coord = accel_sign * a / (gamma ** 3)
        current_v += a_coord * dt

        # Clamp velocity to avoid numerical issues
        if abs(current_v) >= 0.9999 * c:
            current_v = np.sign(current_v) * 0.9999 * c

        v_profile[i] = current_v

        # Update position
        current_x += current_v * dt
        x[i] = current_x

    return t, x, tau, v_profile


def analytic_constant_velocity(T, v, c=1.0):
    """
    Analytic proper time for constant velocity round trip.

    tau = T * sqrt(1 - v^2/c^2) = T / gamma

    Args:
        T: Coordinate time for journey
        v: Travel velocity
        c: Speed of light

    Returns:
        Proper time experienced by traveler
    """
    return T * proper_time_integrand(v, c)


def numerical_proper_time(t, v_profile, c=1.0):
    """
    Numerically integrate proper time from velocity profile.

    tau = integral sqrt(1 - v(t)^2/c^2) dt

    Args:
        t: Time array
        v_profile: Velocity at each time
        c: Speed of light

    Returns:
        Total proper time
    """
    dtau_dt = proper_time_integrand(v_profile, c)
    return np.trapezoid(dtau_dt, t)


def main():
    c = 1.0  # Speed of light (natural units)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # ==========================================================================
    # Plot 1: Spacetime diagram - Constant velocity journey
    # ==========================================================================
    ax1 = axes[0, 0]

    T_total = 20.0  # Total coordinate time
    v_travel = 0.8 * c  # Travel velocity

    t, x, tau, v_profile = constant_velocity_journey(T_total, v_travel, c)

    # Stay-at-home twin worldline (vertical line at x=0)
    ax1.plot([0, 0], [0, T_total], 'b-', lw=3, label='Stay-at-home twin')

    # Traveling twin worldline
    ax1.plot(x, t, 'r-', lw=3, label='Traveling twin')

    # Light cones from origin
    t_light = np.linspace(0, T_total, 100)
    ax1.plot(t_light, t_light, 'y--', lw=1.5, alpha=0.7, label='Light cone')
    ax1.plot(-t_light, t_light, 'y--', lw=1.5, alpha=0.7)

    # Mark departure and arrival
    ax1.plot(0, 0, 'go', markersize=12, zorder=5)
    ax1.plot(0, T_total, 'go', markersize=12, zorder=5)
    ax1.annotate('Departure', (0, 0), xytext=(0.5, 1), fontsize=9)
    ax1.annotate('Reunion', (0, T_total), xytext=(0.5, T_total - 1), fontsize=9)

    # Mark turnaround
    T_half = T_total / 2
    x_turn = v_travel * T_half
    ax1.plot(x_turn, T_half, 'r*', markersize=15, zorder=5)
    ax1.annotate('Turnaround', (x_turn, T_half), xytext=(x_turn + 0.5, T_half + 1), fontsize=9)

    ax1.set_xlabel('Position x (light-units)')
    ax1.set_ylabel('Coordinate Time t')
    ax1.set_title(f'Spacetime Diagram (v = {v_travel/c}c)')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # ==========================================================================
    # Plot 2: Proper time comparison
    # ==========================================================================
    ax2 = axes[0, 1]

    tau_traveler = tau[-1]
    tau_home = T_total  # Stay-at-home twin ages in coordinate time

    # Age comparison
    ages = [tau_home, tau_traveler]
    labels = ['Stay-at-home\nTwin', 'Traveling\nTwin']
    colors = ['blue', 'red']

    bars = ax2.bar(labels, ages, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

    ax2.set_ylabel('Proper Time (years)', fontsize=11)
    ax2.set_title('Age Comparison at Reunion')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, age in zip(bars, ages):
        height = bar.get_height()
        ax2.annotate(f'{age:.2f} years',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Age difference annotation
    age_diff = tau_home - tau_traveler
    ax2.annotate(f'Age difference:\n{age_diff:.2f} years',
                xy=(0.5, (tau_home + tau_traveler) / 2),
                xytext=(1.5, (tau_home + tau_traveler) / 2),
                fontsize=10, ha='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # ==========================================================================
    # Plot 3: Velocity profiles and proper time accumulation
    # ==========================================================================
    ax3 = axes[0, 2]

    # Compare different journey profiles
    T_total = 20.0
    v_max = 0.8 * c

    # Constant velocity journey
    t1, x1, tau1, v1 = constant_velocity_journey(T_total, v_max, c)

    # Smooth turnaround journey
    t2, x2, tau2, v2 = smooth_turnaround_journey(T_total, v_max, T_total / 10, c)

    ax3.plot(t1, tau1, 'r-', lw=2, label=f'Constant v (tau = {tau1[-1]:.2f})')
    ax3.plot(t2, tau2, 'g-', lw=2, label=f'Smooth turnaround (tau = {tau2[-1]:.2f})')
    ax3.plot(t1, t1, 'b--', lw=2, label='Stay-at-home (tau = t)')

    ax3.set_xlabel('Coordinate Time t')
    ax3.set_ylabel('Proper Time tau')
    ax3.set_title('Proper Time Accumulation')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # ==========================================================================
    # Plot 4: Resolution through acceleration - velocity profiles
    # ==========================================================================
    ax4 = axes[1, 0]

    ax4.plot(t1, v1 / c, 'r-', lw=2, label='Constant velocity')
    ax4.plot(t2, v2 / c, 'g-', lw=2, label='Smooth turnaround')

    ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax4.fill_between(t2, 0, v2 / c, alpha=0.2, color='green')

    ax4.set_xlabel('Coordinate Time t')
    ax4.set_ylabel('Velocity v/c')
    ax4.set_title('Velocity Profiles: The Role of Acceleration')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # Highlight acceleration phases
    ax4.axvspan(0, T_total / 10, alpha=0.1, color='orange', label='Acceleration')
    ax4.axvspan(T_total / 2 - T_total / 10, T_total / 2 + T_total / 10,
               alpha=0.1, color='orange')
    ax4.axvspan(T_total - T_total / 10, T_total, alpha=0.1, color='orange')

    ax4.annotate('Acceleration\nbreaks symmetry!',
                xy=(T_total / 2, 0), xytext=(T_total / 2 + 3, 0.5),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='orange'),
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # ==========================================================================
    # Plot 5: Numerical vs Analytic verification
    # ==========================================================================
    ax5 = axes[1, 1]

    velocities = np.linspace(0.1, 0.99, 50) * c
    analytic_tau = []
    numerical_tau = []

    for v in velocities:
        # Analytic result
        tau_a = analytic_constant_velocity(T_total, v, c)
        analytic_tau.append(tau_a)

        # Numerical result
        t, x, tau, v_prof = constant_velocity_journey(T_total, v, c, n_points=2000)
        numerical_tau.append(tau[-1])

    analytic_tau = np.array(analytic_tau)
    numerical_tau = np.array(numerical_tau)

    ax5.plot(velocities / c, analytic_tau, 'b-', lw=2, label='Analytic: T/gamma')
    ax5.plot(velocities / c, numerical_tau, 'r--', lw=2, label='Numerical integration')
    ax5.plot(velocities / c, [T_total] * len(velocities), 'g:', lw=2, label='Stay-at-home')

    ax5.set_xlabel('Travel Velocity v/c')
    ax5.set_ylabel('Proper Time (traveler)')
    ax5.set_title('Numerical vs Analytic Verification')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    # Relative error
    rel_error = np.abs(numerical_tau - analytic_tau) / analytic_tau
    ax5_inset = ax5.inset_axes([0.55, 0.55, 0.4, 0.35])
    ax5_inset.semilogy(velocities / c, rel_error, 'k-', lw=1)
    ax5_inset.set_xlabel('v/c', fontsize=8)
    ax5_inset.set_ylabel('Relative Error', fontsize=8)
    ax5_inset.set_title('Numerical Error', fontsize=9)
    ax5_inset.grid(True, alpha=0.3)
    ax5_inset.tick_params(labelsize=7)

    # ==========================================================================
    # Plot 6: Compare journey profiles - constant v vs constant acceleration
    # ==========================================================================
    ax6 = axes[1, 2]

    T_total = 20.0

    # Different journey types
    profiles = [
        ('Constant v=0.6c', 0.6, 'blue'),
        ('Constant v=0.8c', 0.8, 'red'),
        ('Constant v=0.9c', 0.9, 'green'),
        ('Constant v=0.95c', 0.95, 'purple'),
    ]

    proper_times = []
    labels = []

    # Stay-at-home baseline
    proper_times.append(T_total)
    labels.append('Stay-at-home')

    for name, v_frac, color in profiles:
        t, x, tau, v_prof = constant_velocity_journey(T_total, v_frac * c, c)
        proper_times.append(tau[-1])
        labels.append(name)

    # Hyperbolic (constant proper acceleration)
    # Choose acceleration to give reasonable velocity
    a = 0.5  # proper acceleration
    t_hyp, x_hyp, tau_hyp, v_hyp = hyperbolic_journey(T_total, a, c)
    proper_times.append(tau_hyp[-1])
    labels.append(f'Constant a={a}')

    colors = ['gray', 'blue', 'red', 'green', 'purple', 'orange']

    x_pos = np.arange(len(labels))
    bars = ax6.bar(x_pos, proper_times, color=colors, alpha=0.7, edgecolor='black')

    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax6.set_ylabel('Proper Time at Reunion')
    ax6.set_title('Comparing Journey Profiles')
    ax6.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, pt in zip(bars, proper_times):
        height = bar.get_height()
        ax6.annotate(f'{pt:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9)

    plt.suptitle('Twin Paradox: Proper Time and Time Dilation\n'
                 r'$\tau = \int \sqrt{1 - v^2/c^2} \, dt$ (Straight worldline maximizes proper time)',
                 fontsize=13, y=1.02)
    plt.tight_layout()

    # Print summary
    print("=" * 70)
    print("TWIN PARADOX ANALYSIS")
    print("=" * 70)
    print(f"\nJourney Parameters:")
    print(f"  Total coordinate time: {T_total} years")
    print(f"  Travel velocity: {v_travel/c}c")
    print(f"  Lorentz factor: gamma = {lorentz_factor(v_travel, c):.4f}")

    print(f"\nProper Time Results:")
    print(f"  Stay-at-home twin: {tau_home:.4f} years")
    print(f"  Traveling twin:    {tau_traveler:.4f} years")
    print(f"  Age difference:    {age_diff:.4f} years")

    print(f"\nVerification (constant velocity case):")
    tau_analytic = analytic_constant_velocity(T_total, v_travel, c)
    print(f"  Analytic:  tau = T/gamma = {tau_analytic:.6f} years")
    print(f"  Numerical: tau = {tau_traveler:.6f} years")
    print(f"  Relative error: {abs(tau_traveler - tau_analytic)/tau_analytic:.2e}")

    print(f"\nResolution of the Paradox:")
    print("  The situation is NOT symmetric because:")
    print("  1. The traveling twin ACCELERATES (changes reference frames)")
    print("  2. The stay-at-home twin remains in a single inertial frame")
    print("  3. Proper time is maximized along a geodesic (straight worldline)")

    print("\nProper Time for Different Velocities:")
    print("-" * 40)
    for v_frac in [0.5, 0.8, 0.9, 0.99, 0.999]:
        tau = analytic_constant_velocity(T_total, v_frac * c, c)
        gamma = lorentz_factor(v_frac * c, c)
        print(f"  v = {v_frac}c: gamma = {gamma:.3f}, tau = {tau:.4f} years")

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'twin_paradox.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")


if __name__ == "__main__":
    main()
