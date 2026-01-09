"""
Example demonstrating standing wave patterns.

This example shows how standing waves form from superposition of
traveling waves, demonstrating nodes, antinodes, and harmonics.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.waves import Wave


def standing_wave(x, t, n, L, amplitude=1.0, c=1.0):
    """
    Calculate standing wave pattern for nth harmonic.

    Args:
        x: Position array
        t: Time
        n: Harmonic number (1 = fundamental)
        L: Length of medium
        amplitude: Wave amplitude
        c: Wave speed

    Returns:
        Wave displacement at each position
    """
    k = n * np.pi / L  # Wave number
    omega = c * k       # Angular frequency

    # Standing wave = 2A sin(kx) cos(ωt)
    return 2 * amplitude * np.sin(k * x) * np.cos(omega * t)


def main():
    # String parameters
    L = 1.0      # Length (m)
    c = 10.0     # Wave speed (m/s)
    A = 0.1      # Amplitude (m)

    # Create position array
    x = np.linspace(0, L, 500)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot first 6 harmonics
    for n in range(1, 7):
        ax = axes[(n-1) // 3, (n-1) % 3]

        # Fundamental frequency
        f_n = n * c / (2 * L)
        T = 1 / f_n

        # Plot at different times within one period
        times = np.linspace(0, T, 9)
        colors = plt.cm.viridis(np.linspace(0, 1, len(times)))

        for t, color in zip(times, colors):
            y = standing_wave(x, t, n, L, A, c)
            ax.plot(x, y, color=color, lw=1, alpha=0.6)

        # Plot envelope (maximum displacement)
        envelope = 2 * A * np.abs(np.sin(n * np.pi * x / L))
        ax.plot(x, envelope, 'k--', lw=2, label='Envelope')
        ax.plot(x, -envelope, 'k--', lw=2)

        # Mark nodes and antinodes
        for i in range(n + 1):
            node_x = i * L / n
            ax.axvline(x=node_x, color='red', linestyle=':', alpha=0.5)

        for i in range(n):
            antinode_x = (i + 0.5) * L / n
            if antinode_x < L:
                ax.plot(antinode_x, 2*A, 'g^', markersize=8)
                ax.plot(antinode_x, -2*A, 'gv', markersize=8)

        ax.set_xlim(0, L)
        ax.set_ylim(-0.25, 0.25)
        ax.set_xlabel('Position (m)')
        ax.set_ylabel('Displacement (m)')
        ax.set_title(f'n = {n} (f = {f_n:.1f} Hz)')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Standing Waves on a String (Fixed Ends)\n'
                 f'L = {L} m, c = {c} m/s, Fundamental f₁ = {c/(2*L):.1f} Hz',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save first figure
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'standing_waves_harmonics.png'), dpi=150, bbox_inches='tight')

    # Create animation-like figure showing time evolution
    fig2, axes2 = plt.subplots(2, 4, figsize=(16, 8))
    axes2 = axes2.flatten()

    n = 3  # Third harmonic
    f_n = n * c / (2 * L)
    T = 1 / f_n

    times = np.linspace(0, T * 0.875, 8)

    for idx, t in enumerate(times):
        ax = axes2[idx]
        y = standing_wave(x, t, n, L, A, c)

        ax.plot(x, y, 'b-', lw=2)
        ax.fill_between(x, y, alpha=0.3)

        # Envelope
        envelope = 2 * A * np.abs(np.sin(n * np.pi * x / L))
        ax.plot(x, envelope, 'k--', lw=1, alpha=0.5)
        ax.plot(x, -envelope, 'k--', lw=1, alpha=0.5)

        ax.set_xlim(0, L)
        ax.set_ylim(-0.25, 0.25)
        ax.set_xlabel('Position (m)')
        ax.set_ylabel('Displacement (m)')
        ax.set_title(f't = {t/T:.3f}T')
        ax.grid(True, alpha=0.3)

    fig2.suptitle(f'Time Evolution of Standing Wave (n={n}, T = {T:.3f} s)',
                  fontsize=14, y=1.02)
    plt.tight_layout()

    fig2.savefig(os.path.join(output_dir, 'standing_waves_time.png'), dpi=150, bbox_inches='tight')
    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
