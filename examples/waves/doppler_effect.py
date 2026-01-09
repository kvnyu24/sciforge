"""
Example demonstrating the Doppler effect for sound waves.

This example shows how the observed frequency changes when a source
or observer is in motion, with visualizations of wave compression/expansion.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.waves import Wave


def doppler_frequency(f_source, v_sound, v_source=0, v_observer=0):
    """
    Calculate observed frequency with Doppler effect.

    Args:
        f_source: Source frequency (Hz)
        v_sound: Speed of sound in medium (m/s)
        v_source: Source velocity (positive = toward observer)
        v_observer: Observer velocity (positive = toward source)

    Returns:
        Observed frequency (Hz)
    """
    return f_source * (v_sound + v_observer) / (v_sound - v_source)


def simulate_moving_source(v_source, f_source, v_sound, t_max):
    """
    Generate wave pattern from a moving source.

    Args:
        v_source: Source velocity (m/s)
        f_source: Source frequency (Hz)
        v_sound: Speed of sound (m/s)
        t_max: Simulation time (s)

    Returns:
        Arrays of wave front positions at final time
    """
    # Emit wave fronts at regular intervals
    period = 1.0 / f_source
    emission_times = np.arange(0, t_max, period)

    # Source position at each emission
    source_positions = v_source * emission_times

    # Wave fronts expand as circles
    # At final time, radius of each wave front is:
    radii = v_sound * (t_max - emission_times)

    return source_positions, radii


def main():
    # Physical parameters
    f_source = 440  # Hz (A4 note)
    v_sound = 343   # m/s (speed of sound in air at 20°C)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Wave front visualizations
    source_velocities = [0, 0.3 * v_sound, 0.8 * v_sound]
    titles = ['Stationary Source\n(v = 0)', 'Moving Source\n(v = 0.3c)',
              'Fast Moving Source\n(v = 0.8c)']

    for idx, (v_s, title) in enumerate(zip(source_velocities, titles)):
        ax = axes[0, idx]

        # Simulate moving source
        t_max = 0.05  # seconds
        source_positions, radii = simulate_moving_source(v_s, f_source, v_sound, t_max)

        # Plot circular wave fronts
        theta = np.linspace(0, 2*np.pi, 100)
        for pos, r in zip(source_positions[-20:], radii[-20:]):  # Last 20 wavefronts
            if r > 0:
                x = pos + r * np.cos(theta)
                y = r * np.sin(theta)
                ax.plot(x, y, 'b-', alpha=0.5, lw=0.5)

        # Mark source position at final time
        source_final = v_s * t_max
        ax.plot(source_final, 0, 'ro', markersize=10, label='Source')
        ax.arrow(source_final, 0, v_s * 0.002, 0, head_width=0.5, head_length=0.2,
                fc='red', ec='red') if v_s > 0 else None

        # Mark observer positions
        ax.plot(-15, 0, 'g^', markersize=10, label='Observer (behind)')
        ax.plot(25, 0, 'gv', markersize=10, label='Observer (ahead)')

        ax.set_xlim(-20, 30)
        ax.set_ylim(-15, 15)
        ax.set_aspect('equal')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title(title)
        if idx == 0:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Frequency shift plots
    v_source_range = np.linspace(-0.9 * v_sound, 0.9 * v_sound, 100)

    # Plot observed frequency vs source velocity
    ax3 = axes[1, 0]
    f_observed_ahead = doppler_frequency(f_source, v_sound, v_source_range, 0)
    f_observed_behind = doppler_frequency(f_source, v_sound, -v_source_range, 0)

    ax3.plot(v_source_range / v_sound, f_observed_ahead, 'b-', lw=2, label='Observer ahead')
    ax3.plot(v_source_range / v_sound, f_observed_behind, 'r--', lw=2, label='Observer behind')
    ax3.axhline(y=f_source, color='gray', linestyle=':', label='Source frequency')
    ax3.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    ax3.set_xlabel('Source Velocity / Speed of Sound')
    ax3.set_ylabel('Observed Frequency (Hz)')
    ax3.set_title('Doppler Shift vs Source Velocity')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 2000)

    # Musical pitch shift example
    ax4 = axes[1, 1]
    speeds = np.array([0, 10, 20, 30, 50, 100, 150])  # m/s
    speed_labels = ['0', '10', '20', '30', '50', '100', '150']

    f_approaching = [doppler_frequency(f_source, v_sound, v, 0) for v in speeds]
    f_receding = [doppler_frequency(f_source, v_sound, -v, 0) for v in speeds]

    x_pos = np.arange(len(speeds))
    width = 0.35

    bars1 = ax4.bar(x_pos - width/2, f_approaching, width, label='Approaching', color='blue', alpha=0.7)
    bars2 = ax4.bar(x_pos + width/2, f_receding, width, label='Receding', color='red', alpha=0.7)
    ax4.axhline(y=f_source, color='gray', linestyle='--', label=f'Source ({f_source} Hz)')

    ax4.set_xlabel('Source Speed (m/s)')
    ax4.set_ylabel('Observed Frequency (Hz)')
    ax4.set_title('Doppler Shift at Various Speeds')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(speed_labels)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # Percentage shift
    ax5 = axes[1, 2]
    percent_approaching = [(f - f_source) / f_source * 100 for f in f_approaching]
    percent_receding = [(f - f_source) / f_source * 100 for f in f_receding]

    ax5.plot(speeds, percent_approaching, 'b-o', lw=2, label='Approaching', markersize=6)
    ax5.plot(speeds, percent_receding, 'r-s', lw=2, label='Receding', markersize=6)
    ax5.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    ax5.set_xlabel('Source Speed (m/s)')
    ax5.set_ylabel('Frequency Shift (%)')
    ax5.set_title('Percentage Frequency Shift')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    plt.suptitle(f'Doppler Effect (f₀ = {f_source} Hz, v_sound = {v_sound} m/s)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'doppler_effect.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'doppler_effect.png')}")


if __name__ == "__main__":
    main()
