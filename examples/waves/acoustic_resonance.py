"""
Example demonstrating acoustic resonance in open and closed tubes.

This example shows the standing wave patterns and resonant frequencies
for tubes with different boundary conditions:
- Open-Open: pressure nodes at both ends
- Closed-Closed: displacement nodes at both ends
- Open-Closed: different boundary conditions create odd harmonics only
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.waves import StandingWave


def pressure_wave_open_open(x, t, n, L, c=343):
    """
    Pressure standing wave in open-open tube (n = 1, 2, 3, ...).

    Pressure nodes at x=0 and x=L (open ends).
    Wavelength: lambda_n = 2L/n
    Frequency: f_n = n*c/(2L)
    """
    k_n = n * np.pi / L
    omega_n = c * k_n
    return np.sin(k_n * x) * np.cos(omega_n * t)


def pressure_wave_closed_closed(x, t, n, L, c=343):
    """
    Pressure standing wave in closed-closed tube (n = 1, 2, 3, ...).

    Pressure antinodes at x=0 and x=L (closed ends).
    Same frequencies as open-open.
    """
    k_n = n * np.pi / L
    omega_n = c * k_n
    return np.cos(k_n * x) * np.cos(omega_n * t)


def pressure_wave_open_closed(x, t, n, L, c=343):
    """
    Pressure standing wave in open-closed tube (n = 1, 3, 5, ... odd only).

    Pressure node at x=0 (open), antinode at x=L (closed).
    Wavelength: lambda_n = 4L/n (n odd)
    Frequency: f_n = n*c/(4L) (n odd)
    """
    # n should be odd: 1, 3, 5, ...
    k_n = n * np.pi / (2 * L)
    omega_n = c * k_n
    return np.sin(k_n * x) * np.cos(omega_n * t)


def main():
    # Physical parameters
    L = 1.0          # Tube length (m)
    c = 343          # Speed of sound in air (m/s)

    # Spatial grid
    x = np.linspace(0, L, 500)

    fig = plt.figure(figsize=(16, 14))

    # =========================================================================
    # Panel 1: Open-Open Tube - First 4 harmonics
    # =========================================================================
    ax1 = fig.add_subplot(3, 3, 1)

    harmonics = [1, 2, 3, 4]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(harmonics)))

    for n, color in zip(harmonics, colors):
        # Show envelope (maximum amplitude)
        envelope = np.abs(np.sin(n * np.pi * x / L))
        f_n = n * c / (2 * L)

        ax1.fill_between(x, -envelope + (n-1)*2.5, envelope + (n-1)*2.5,
                         alpha=0.3, color=color)
        ax1.plot(x, envelope + (n-1)*2.5, color=color, lw=2,
                 label=f'n={n}, f={f_n:.1f} Hz')
        ax1.plot(x, -envelope + (n-1)*2.5, color=color, lw=2)

    # Mark open ends
    ax1.axvline(x=0, color='green', lw=3, linestyle='--')
    ax1.axvline(x=L, color='green', lw=3, linestyle='--')

    ax1.set_xlabel('Position in Tube (m)')
    ax1.set_ylabel('Pressure Amplitude (offset)')
    ax1.set_title('Open-Open Tube\n(All harmonics, pressure nodes at ends)')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.05, L+0.05)

    # =========================================================================
    # Panel 2: Closed-Closed Tube - First 4 harmonics
    # =========================================================================
    ax2 = fig.add_subplot(3, 3, 2)

    for n, color in zip(harmonics, colors):
        # Pressure antinodes at closed ends
        envelope = np.abs(np.cos(n * np.pi * x / L))
        f_n = n * c / (2 * L)

        ax2.fill_between(x, -envelope + (n-1)*2.5, envelope + (n-1)*2.5,
                         alpha=0.3, color=color)
        ax2.plot(x, envelope + (n-1)*2.5, color=color, lw=2,
                 label=f'n={n}, f={f_n:.1f} Hz')
        ax2.plot(x, -envelope + (n-1)*2.5, color=color, lw=2)

    # Mark closed ends
    ax2.axvline(x=0, color='black', lw=3)
    ax2.axvline(x=L, color='black', lw=3)

    ax2.set_xlabel('Position in Tube (m)')
    ax2.set_ylabel('Pressure Amplitude (offset)')
    ax2.set_title('Closed-Closed Tube\n(All harmonics, pressure antinodes at ends)')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.05, L+0.05)

    # =========================================================================
    # Panel 3: Open-Closed Tube - Odd harmonics only
    # =========================================================================
    ax3 = fig.add_subplot(3, 3, 3)

    odd_harmonics = [1, 3, 5, 7]
    colors_odd = plt.cm.plasma(np.linspace(0.1, 0.9, len(odd_harmonics)))

    for n, color in zip(odd_harmonics, colors_odd):
        # Pressure node at open end (x=0), antinode at closed end (x=L)
        envelope = np.abs(np.sin(n * np.pi * x / (2 * L)))
        f_n = n * c / (4 * L)

        idx = odd_harmonics.index(n)
        ax3.fill_between(x, -envelope + idx*2.5, envelope + idx*2.5,
                         alpha=0.3, color=color)
        ax3.plot(x, envelope + idx*2.5, color=color, lw=2,
                 label=f'n={n}, f={f_n:.1f} Hz')
        ax3.plot(x, -envelope + idx*2.5, color=color, lw=2)

    # Mark boundaries
    ax3.axvline(x=0, color='green', lw=3, linestyle='--', label='Open')
    ax3.axvline(x=L, color='black', lw=3, label='Closed')

    ax3.set_xlabel('Position in Tube (m)')
    ax3.set_ylabel('Pressure Amplitude (offset)')
    ax3.set_title('Open-Closed Tube\n(ODD harmonics only!)')
    ax3.legend(fontsize=8, loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-0.05, L+0.05)

    # =========================================================================
    # Panel 4: Frequency comparison
    # =========================================================================
    ax4 = fig.add_subplot(3, 3, 4)

    n_range = np.arange(1, 9)

    f_open_open = n_range * c / (2 * L)
    f_closed_closed = n_range * c / (2 * L)  # Same as open-open

    # For open-closed, only odd n
    n_odd = np.array([1, 3, 5, 7])
    f_open_closed = n_odd * c / (4 * L)

    width = 0.25
    x_pos = np.arange(len(n_range))

    ax4.bar(x_pos - width, f_open_open, width, label='Open-Open', color='blue', alpha=0.7)
    ax4.bar(x_pos, f_closed_closed, width, label='Closed-Closed', color='red', alpha=0.7)

    # Plot open-closed at odd positions
    x_pos_odd = [0, 2, 4, 6]  # Positions for n=1,3,5,7
    ax4.bar(np.array(x_pos_odd) + width, f_open_closed, width,
            label='Open-Closed', color='green', alpha=0.7)

    ax4.set_xlabel('Harmonic Number n')
    ax4.set_ylabel('Frequency (Hz)')
    ax4.set_title('Resonant Frequencies\n(Open-closed has only odd harmonics)')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(n_range)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # =========================================================================
    # Panel 5: Time evolution of standing wave
    # =========================================================================
    ax5 = fig.add_subplot(3, 3, 5)

    n = 2  # Second harmonic
    T = 2 * L / (n * c)  # Period
    times = np.linspace(0, T, 9)
    colors_time = plt.cm.coolwarm(np.linspace(0, 1, len(times)))

    for t, color in zip(times, colors_time):
        pressure = pressure_wave_open_open(x, t, n, L, c)
        ax5.plot(x, pressure, color=color, lw=1.5, alpha=0.8)

    # Envelope
    envelope = np.abs(np.sin(n * np.pi * x / L))
    ax5.plot(x, envelope, 'k--', lw=2, label='Envelope')
    ax5.plot(x, -envelope, 'k--', lw=2)

    ax5.set_xlabel('Position in Tube (m)')
    ax5.set_ylabel('Pressure')
    ax5.set_title(f'Time Evolution (Open-Open, n={n})\nT = {T*1000:.2f} ms')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0, L)

    # =========================================================================
    # Panel 6: Pressure vs Displacement
    # =========================================================================
    ax6 = fig.add_subplot(3, 3, 6)

    n = 1  # Fundamental
    t = 0  # At maximum

    # Pressure pattern
    pressure = np.sin(n * np.pi * x / L)

    # Displacement is 90 degrees out of phase with pressure
    # Where pressure is max, displacement is zero and vice versa
    displacement = np.cos(n * np.pi * x / L)

    ax6.plot(x, pressure, 'b-', lw=2, label='Pressure')
    ax6.plot(x, displacement, 'r--', lw=2, label='Displacement')

    # Mark nodes and antinodes
    ax6.axhline(y=0, color='gray', lw=0.5)

    ax6.set_xlabel('Position in Tube (m)')
    ax6.set_ylabel('Amplitude')
    ax6.set_title('Pressure vs Displacement Waves\n(90 degrees out of phase)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(0, L)

    # =========================================================================
    # Panel 7: Resonance curve
    # =========================================================================
    ax7 = fig.add_subplot(3, 3, 7)

    # Simulate driving frequency sweep
    f_drive = np.linspace(50, 2000, 1000)
    Q = 50  # Quality factor

    # Resonant frequencies for open-open tube
    f_res = np.array([n * c / (2 * L) for n in range(1, 6)])

    # Response amplitude (Lorentzian peaks)
    response = np.zeros_like(f_drive)
    for f_r in f_res:
        gamma = f_r / Q
        response += 1 / np.sqrt((f_drive**2 - f_r**2)**2 + (gamma * f_drive)**2)

    # Normalize
    response = response / np.max(response)

    ax7.plot(f_drive, response, 'b-', lw=2)

    for f_r in f_res:
        ax7.axvline(x=f_r, color='red', linestyle='--', alpha=0.5)

    ax7.set_xlabel('Driving Frequency (Hz)')
    ax7.set_ylabel('Response Amplitude')
    ax7.set_title(f'Resonance Curve (Open-Open)\nPeaks at f_n = n*c/(2L) = n*{c/(2*L):.1f} Hz')
    ax7.grid(True, alpha=0.3)
    ax7.set_xlim(50, 2000)

    # =========================================================================
    # Panel 8: Organ pipe comparison
    # =========================================================================
    ax8 = fig.add_subplot(3, 3, 8)

    # Different pipe lengths (musical notes)
    pipe_lengths = [1.0, 0.5, 0.25, 0.125]  # L in meters
    pipe_names = ['L', 'L/2', 'L/4', 'L/8']

    for i, (pipe_L, name) in enumerate(zip(pipe_lengths, pipe_names)):
        # Fundamental frequency
        f_fund = c / (2 * pipe_L)

        # Draw pipe
        ax8.barh(i, pipe_L, height=0.3, color='brown', alpha=0.7)
        ax8.text(pipe_L + 0.05, i, f'{name}\nf = {f_fund:.0f} Hz',
                 va='center', fontsize=9)

        # Draw standing wave inside
        x_pipe = np.linspace(0, pipe_L, 100)
        wave = 0.12 * np.sin(np.pi * x_pipe / pipe_L)
        ax8.plot(x_pipe, wave + i, 'b-', lw=2)

    ax8.set_xlabel('Pipe Length (m)')
    ax8.set_ylabel('Pipe')
    ax8.set_yticks(range(len(pipe_lengths)))
    ax8.set_yticklabels([f'Pipe {i+1}' for i in range(len(pipe_lengths))])
    ax8.set_title('Organ Pipes: Shorter = Higher Pitch\n(Fundamental shown)')
    ax8.grid(True, alpha=0.3, axis='x')
    ax8.set_xlim(0, 1.5)

    # =========================================================================
    # Panel 9: End correction
    # =========================================================================
    ax9 = fig.add_subplot(3, 3, 9)

    # For open ends, effective length > physical length
    # End correction ~ 0.6 * radius
    radii = np.linspace(0.01, 0.1, 100)
    end_correction = 0.6 * radii

    # Effective length ratio
    L_phys = 0.5  # 50 cm pipe
    L_eff = L_phys + 2 * end_correction  # Two open ends

    f_uncorrected = c / (2 * L_phys)
    f_corrected = c / (2 * L_eff)

    ax9.plot(radii * 100, f_corrected, 'b-', lw=2,
             label='With end correction')
    ax9.axhline(y=f_uncorrected, color='r', linestyle='--',
                label=f'Uncorrected = {f_uncorrected:.1f} Hz')

    ax9.set_xlabel('Tube Radius (cm)')
    ax9.set_ylabel('Fundamental Frequency (Hz)')
    ax9.set_title(f'End Correction Effect (L = {L_phys*100:.0f} cm)\n'
                  'Wider tubes have lower frequency')
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    # Add formula annotation
    ax9.text(5, f_uncorrected - 15,
             'End correction: delta_L = 0.6*R',
             fontsize=10, style='italic')

    plt.suptitle('Acoustic Resonance in Tubes\n'
                 'Boundary conditions determine allowed harmonics',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'acoustic_resonance.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'acoustic_resonance.png')}")


if __name__ == "__main__":
    main()
