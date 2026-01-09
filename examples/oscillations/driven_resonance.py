"""
Example demonstrating driven harmonic oscillator resonance.

This example shows how a harmonically driven oscillator responds to different
driving frequencies, demonstrating resonance when the driving frequency
matches the natural frequency.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.oscillations import ResonantSystem


def simulate_frequency_sweep():
    """
    Simulate driven oscillator response across a range of frequencies.

    Returns:
        Dictionary with frequency response data
    """
    # System parameters
    k = 100.0       # Spring constant (N/m)
    m = 1.0         # Mass (kg)
    damping = 2.0   # Damping coefficient

    # Natural frequency
    omega_0 = np.sqrt(k / m)

    # Frequency range to scan (relative to natural frequency)
    freq_ratios = np.linspace(0.1, 3.0, 100)
    frequencies = freq_ratios * omega_0

    # Driving force amplitude
    F0 = 10.0

    results = {
        'frequencies': frequencies,
        'freq_ratios': freq_ratios,
        'amplitudes': [],
        'phases': [],
        'omega_0': omega_0
    }

    for omega in frequencies:
        system = ResonantSystem(
            mass=m,
            spring_constant=k,
            driving_frequency=omega,
            driving_amplitude=F0,
            damping=damping
        )

        # Get steady-state amplitude
        amplitude = system.resonance_amplitude()
        results['amplitudes'].append(amplitude)

        # Calculate phase lag
        gamma = damping / m
        phase = np.arctan2(gamma * omega, omega_0**2 - omega**2)
        results['phases'].append(phase)

    results['amplitudes'] = np.array(results['amplitudes'])
    results['phases'] = np.array(results['phases'])

    return results


def simulate_time_evolution():
    """Simulate time evolution at different driving frequencies."""
    k = 100.0
    m = 1.0
    damping = 2.0
    omega_0 = np.sqrt(k / m)
    F0 = 10.0

    # Three cases: below, at, and above resonance
    freq_cases = {
        'Below resonance (ω/ω₀ = 0.5)': 0.5 * omega_0,
        'At resonance (ω/ω₀ = 1.0)': omega_0,
        'Above resonance (ω/ω₀ = 2.0)': 2.0 * omega_0
    }

    dt = 0.001
    t_final = 5.0
    time_data = {}

    for name, omega in freq_cases.items():
        system = ResonantSystem(
            mass=m,
            spring_constant=k,
            driving_frequency=omega,
            driving_amplitude=F0,
            damping=damping
        )

        times = []
        positions = []
        t = 0

        while t < t_final:
            times.append(t)
            positions.append(system.position[0])  # x-component
            system.update(system.driving_force(t), dt)
            t += dt

        time_data[name] = {
            'time': np.array(times),
            'position': np.array(positions)
        }

    return time_data, omega_0


def plot_results(freq_results, time_data, omega_0):
    """Create comprehensive resonance visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Amplitude vs frequency (resonance curve)
    ax1 = axes[0, 0]
    ax1.plot(freq_results['freq_ratios'], freq_results['amplitudes'], 'b-', lw=2)
    ax1.axvline(x=1.0, color='r', linestyle='--', alpha=0.7, label='Natural frequency (ω₀)')

    # Find and mark resonance peak
    peak_idx = np.argmax(freq_results['amplitudes'])
    peak_ratio = freq_results['freq_ratios'][peak_idx]
    peak_amp = freq_results['amplitudes'][peak_idx]
    ax1.plot(peak_ratio, peak_amp, 'ro', markersize=10, label=f'Peak at ω/ω₀ = {peak_ratio:.2f}')

    ax1.set_xlabel('Frequency Ratio (ω/ω₀)')
    ax1.set_ylabel('Amplitude (m)')
    ax1.set_title('Resonance Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Phase vs frequency
    ax2 = axes[0, 1]
    ax2.plot(freq_results['freq_ratios'], np.degrees(freq_results['phases']), 'g-', lw=2)
    ax2.axvline(x=1.0, color='r', linestyle='--', alpha=0.7)
    ax2.axhline(y=90, color='gray', linestyle=':', alpha=0.7, label='90° phase lag')
    ax2.set_xlabel('Frequency Ratio (ω/ω₀)')
    ax2.set_ylabel('Phase Lag (degrees)')
    ax2.set_title('Phase Response')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Time evolution comparison
    ax3 = axes[1, 0]
    colors = ['blue', 'red', 'green']
    for (name, data), color in zip(time_data.items(), colors):
        ax3.plot(data['time'], data['position'], color=color, lw=1.5, label=name, alpha=0.8)

    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Position (m)')
    ax3.set_title('Time Evolution at Different Driving Frequencies')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Power absorbed (proportional to amplitude^2 * omega)
    ax4 = axes[1, 1]
    power = freq_results['amplitudes']**2 * freq_results['frequencies']
    power = power / np.max(power)  # Normalize
    ax4.fill_between(freq_results['freq_ratios'], power, alpha=0.3, color='orange')
    ax4.plot(freq_results['freq_ratios'], power, 'orange', lw=2)
    ax4.axvline(x=1.0, color='r', linestyle='--', alpha=0.7)
    ax4.set_xlabel('Frequency Ratio (ω/ω₀)')
    ax4.set_ylabel('Relative Power Absorbed')
    ax4.set_title('Power Absorption Spectrum')
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f'Driven Harmonic Oscillator Resonance (ω₀ = {omega_0:.2f} rad/s)', fontsize=14, y=1.02)
    plt.tight_layout()


def main():
    # Run simulations
    freq_results = simulate_frequency_sweep()
    time_data, omega_0 = simulate_time_evolution()

    # Plot results
    plot_results(freq_results, time_data, omega_0)

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'driven_resonance.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'driven_resonance.png')}")


if __name__ == "__main__":
    main()
