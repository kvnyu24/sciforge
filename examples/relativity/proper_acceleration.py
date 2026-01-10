"""
Experiment 190: Constant Proper Acceleration (Hyperbolic Motion)

This experiment demonstrates relativistic motion under constant proper
acceleration, showing the hyperbolic worldline and twin paradox resolution.

Physical concepts:
- Proper acceleration vs coordinate acceleration
- Hyperbolic motion in spacetime
- Rindler coordinates and horizon
- Relativistic rocket equation
- Interstellar travel scenarios
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt


def hyperbolic_motion(tau, a, c=1.0):
    """
    Calculate position and time for constant proper acceleration.

    x(tau) = (c^2/a) * (cosh(a*tau/c) - 1)
    t(tau) = (c/a) * sinh(a*tau/c)
    v(tau) = c * tanh(a*tau/c)

    Args:
        tau: Proper time
        a: Proper acceleration
        c: Speed of light

    Returns:
        (t, x, v) - coordinate time, position, velocity
    """
    t = (c / a) * np.sinh(a * tau / c)
    x = (c**2 / a) * (np.cosh(a * tau / c) - 1)
    v = c * np.tanh(a * tau / c)

    return t, x, v


def coordinate_acceleration(tau, a, c=1.0):
    """
    Calculate coordinate acceleration for constant proper acceleration.

    a_coord = a / gamma^3 = a * (1 - v^2/c^2)^(3/2)
    """
    v = c * np.tanh(a * tau / c)
    gamma = 1 / np.sqrt(1 - (v/c)**2)
    return a / gamma**3


def travel_time(d, a, c=1.0):
    """
    Calculate proper time and coordinate time to travel distance d
    with constant proper acceleration a (accelerate half, decelerate half).

    For one-way trip accelerating then decelerating:
    tau = (2c/a) * arcosh(1 + a*d/(2*c^2))
    t = (2/a) * sqrt((a*d/c)^2 + 2*a*d)

    Args:
        d: Distance to travel
        a: Proper acceleration
        c: Speed of light

    Returns:
        (tau_total, t_total) - proper time and coordinate time
    """
    # Accelerate for half distance, then decelerate
    d_half = d / 2

    # Proper time for half trip
    tau_half = (c / a) * np.arccosh(1 + a * d_half / c**2)

    # Coordinate time for half trip
    t_half = (c / a) * np.sqrt((1 + a * d_half / c**2)**2 - 1)

    return 2 * tau_half, 2 * t_half


def gamma_from_proper_time(tau, a, c=1.0):
    """Calculate Lorentz factor for hyperbolic motion."""
    return np.cosh(a * tau / c)


def main():
    c = 1.0  # Speed of light (natural units)
    g = 1.0  # Use g = 1 for nice plots (can interpret as 10 m/s^2)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ==========================================================================
    # Plot 1: Worldline in spacetime
    # ==========================================================================
    ax1 = axes[0, 0]

    tau_range = np.linspace(0, 3, 100)  # Proper time

    accelerations = [0.5*g, 1.0*g, 2.0*g]
    colors = ['blue', 'red', 'green']

    for a, color in zip(accelerations, colors):
        t, x, v = hyperbolic_motion(tau_range, a, c)
        ax1.plot(x, t, '-', color=color, lw=2, label=f'a = {a}g')

    # Light cone
    x_light = np.linspace(0, 3, 100)
    ax1.plot(x_light, x_light, 'y-', lw=2, alpha=0.7, label='Light cone')

    # Rindler horizon for a=g
    ax1.axvline(x=-c**2/g, color='purple', linestyle='--', lw=1.5,
               label='Rindler horizon (a=g)')

    ax1.set_xlabel('Position x')
    ax1.set_ylabel('Coordinate time t')
    ax1.set_title('Worldlines: Constant Proper Acceleration')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.5, 3)
    ax1.set_ylim(0, 3)

    # Annotate hyperbolic nature
    ax1.annotate('Hyperbolic\nworldline', xy=(1.5, 2), fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ==========================================================================
    # Plot 2: Velocity and acceleration vs proper time
    # ==========================================================================
    ax2 = axes[0, 1]

    a = g  # Use 1g acceleration

    t, x, v = hyperbolic_motion(tau_range, a, c)
    a_coord = coordinate_acceleration(tau_range, a, c)
    gamma = gamma_from_proper_time(tau_range, a, c)

    ax2.plot(tau_range, v/c, 'b-', lw=2, label='Velocity v/c')
    ax2.plot(tau_range, a_coord/a, 'r-', lw=2, label='Coord. acceleration / a')
    ax2.plot(tau_range, 1/gamma, 'g--', lw=2, label='1/gamma')

    ax2.axhline(y=1, color='gold', linestyle='--', alpha=0.7, label='Speed of light')
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    ax2.set_xlabel('Proper time tau')
    ax2.set_ylabel('Normalized values')
    ax2.set_title(f'Motion Parameters (a = {a}g)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ==========================================================================
    # Plot 3: Time dilation (twin paradox)
    # ==========================================================================
    ax3 = axes[1, 0]

    # Compare proper time and coordinate time
    ax3.plot(tau_range, t, 'b-', lw=2, label='Coordinate time t')
    ax3.plot(tau_range, tau_range, 'r--', lw=2, label='Proper time tau')

    # Show time dilation
    ax3.fill_between(tau_range, tau_range, t, alpha=0.2, color='blue',
                    label='Time dilation')

    ax3.set_xlabel('Proper time tau (traveler)')
    ax3.set_ylabel('Time')
    ax3.set_title(f'Twin Paradox: Time Dilation at a = {a}g')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Add annotation
    tau_mark = 2.5
    t_mark, _, _ = hyperbolic_motion(tau_mark, a, c)
    ax3.annotate(f'At tau = {tau_mark}:\nt = {t_mark:.2f}\nDifference = {t_mark - tau_mark:.2f}',
                xy=(tau_mark, t_mark), xytext=(1.5, 4),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # ==========================================================================
    # Plot 4: Interstellar travel scenarios
    # ==========================================================================
    ax4 = axes[1, 1]

    # Physical units: a = 10 m/s^2 (approximately 1g)
    # c = 3e8 m/s
    c_real = 3e8  # m/s
    a_real = 10.0  # m/s^2 (about 1g)
    year = 365.25 * 24 * 3600  # seconds
    ly = c_real * year  # light-year in meters

    # Destinations (distances in light-years)
    destinations = [
        ('Alpha Centauri', 4.4),
        ('Vega', 25),
        ('Galactic Center', 27000),
        ('Andromeda Galaxy', 2.5e6),
    ]

    distances_ly = np.array([d[1] for d in destinations])
    names = [d[0] for d in destinations]
    distances_m = distances_ly * ly

    # Calculate travel times
    proper_times = []
    coord_times = []

    for d in distances_m:
        tau, t = travel_time(d, a_real, c_real)
        proper_times.append(tau / year)  # Convert to years
        coord_times.append(t / year)

    x = np.arange(len(destinations))
    width = 0.35

    bars1 = ax4.bar(x - width/2, proper_times, width, label='Traveler time (proper)',
                   color='blue', alpha=0.7)
    bars2 = ax4.bar(x + width/2, coord_times, width, label='Earth time (coordinate)',
                   color='red', alpha=0.7)

    ax4.set_xlabel('Destination')
    ax4.set_ylabel('Travel time (years)')
    ax4.set_title('Interstellar Travel at 1g Constant Acceleration\n(Accelerate half, decelerate half)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(names, rotation=15, ha='right')
    ax4.legend()
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3, which='both')

    # Add distance labels
    for i, (name, d) in enumerate(destinations):
        ax4.text(i, coord_times[i] * 1.5, f'{d:.1e} ly' if d > 100 else f'{d} ly',
                ha='center', fontsize=8)

    plt.suptitle('Constant Proper Acceleration (Hyperbolic Motion)\n'
                 'x(tau) = (c^2/a)(cosh(a*tau/c) - 1), v = c*tanh(a*tau/c)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Print interstellar travel summary
    print("Interstellar Travel at 1g Constant Acceleration:")
    print("=" * 70)
    print(f"{'Destination':<25} {'Distance':<15} {'Traveler Time':<20} {'Earth Time':<15}")
    print("-" * 70)

    for i, (name, d) in enumerate(destinations):
        if proper_times[i] < 1:
            t_str = f"{proper_times[i]*12:.1f} months"
        elif proper_times[i] < 100:
            t_str = f"{proper_times[i]:.1f} years"
        else:
            t_str = f"{proper_times[i]:.0f} years"

        if coord_times[i] < 100:
            tc_str = f"{coord_times[i]:.1f} years"
        else:
            tc_str = f"{coord_times[i]:.0f} years"

        d_str = f"{d} ly" if d < 100 else f"{d:.2e} ly"
        print(f"{name:<25} {d_str:<15} {t_str:<20} {tc_str:<15}")

    # Maximum speed reached
    print("\nMaximum speeds reached at midpoint:")
    for i, (name, d) in enumerate(destinations):
        d_m = d * ly
        # At midpoint, tau = tau_total/2
        tau_total, _ = travel_time(d_m, a_real, c_real)
        tau_mid = tau_total / 2
        _, _, v_max = hyperbolic_motion(tau_mid, a_real, c_real)
        print(f"  {name}: v_max = {v_max/c_real:.6f}c")

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'proper_acceleration.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")


if __name__ == "__main__":
    main()
