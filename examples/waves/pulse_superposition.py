"""
Example demonstrating superposition and interference of pulses.

This example shows how two wave pulses interact when they meet,
demonstrating constructive and destructive interference depending
on their relative phases.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.waves import Wave


def gaussian_pulse(x, x0, sigma, amplitude=1.0):
    """Gaussian pulse centered at x0."""
    return amplitude * np.exp(-(x - x0)**2 / (2 * sigma**2))


def sech_pulse(x, x0, width, amplitude=1.0):
    """Hyperbolic secant pulse (soliton shape)."""
    arg = (x - x0) / width
    # Avoid overflow
    arg = np.clip(arg, -50, 50)
    return amplitude / np.cosh(arg)


def main():
    # Spatial domain
    x = np.linspace(-15, 15, 1000)
    dx = x[1] - x[0]

    fig = plt.figure(figsize=(16, 14))

    # =========================================================================
    # Panel 1: Constructive Interference (same polarity)
    # =========================================================================
    ax1 = fig.add_subplot(3, 3, 1)

    c = 1.0       # Wave speed
    sigma = 1.0   # Pulse width

    # Two pulses starting at opposite ends, same polarity
    x1_start, x2_start = -8, 8

    times = np.linspace(0, 8, 5)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(times)))

    for t, color in zip(times, colors):
        pulse1 = gaussian_pulse(x, x1_start + c*t, sigma)  # Moving right
        pulse2 = gaussian_pulse(x, x2_start - c*t, sigma)  # Moving left
        total = pulse1 + pulse2

        ax1.plot(x, total + 3*t, color=color, lw=2, label=f't = {t:.1f}')

    ax1.set_xlabel('Position')
    ax1.set_ylabel('Displacement (offset)')
    ax1.set_title('Constructive Interference\n(Same polarity pulses)')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-15, 15)

    # =========================================================================
    # Panel 2: Destructive Interference (opposite polarity)
    # =========================================================================
    ax2 = fig.add_subplot(3, 3, 2)

    for t, color in zip(times, colors):
        pulse1 = gaussian_pulse(x, x1_start + c*t, sigma)   # Moving right
        pulse2 = -gaussian_pulse(x, x2_start - c*t, sigma)  # Moving left, inverted
        total = pulse1 + pulse2

        ax2.plot(x, total + 3*t, color=color, lw=2, label=f't = {t:.1f}')

    ax2.set_xlabel('Position')
    ax2.set_ylabel('Displacement (offset)')
    ax2.set_title('Destructive Interference\n(Opposite polarity pulses)')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-15, 15)

    # =========================================================================
    # Panel 3: Detailed view at collision moment
    # =========================================================================
    ax3 = fig.add_subplot(3, 3, 3)

    # Show what happens at and around t = 4 (collision point)
    t_collision = np.linspace(3, 5, 9)
    colors_coll = plt.cm.plasma(np.linspace(0, 1, len(t_collision)))

    for t, color in zip(t_collision, colors_coll):
        pulse1 = gaussian_pulse(x, x1_start + c*t, sigma)
        pulse2 = gaussian_pulse(x, x2_start - c*t, sigma)
        total = pulse1 + pulse2

        ax3.plot(x, total, color=color, lw=1.5, alpha=0.7)

    # Highlight the moment of maximum constructive interference
    t_max = 4.0
    pulse1 = gaussian_pulse(x, x1_start + c*t_max, sigma)
    pulse2 = gaussian_pulse(x, x2_start - c*t_max, sigma)
    ax3.plot(x, pulse1 + pulse2, 'k-', lw=3, label=f'Max (t={t_max})')

    ax3.set_xlabel('Position')
    ax3.set_ylabel('Displacement')
    ax3.set_title('Collision Detail\n(Amplitude doubles at center)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-5, 5)
    ax3.set_ylim(-0.5, 2.5)

    # =========================================================================
    # Panel 4: Space-time diagram - Constructive
    # =========================================================================
    ax4 = fig.add_subplot(3, 3, 4)

    t_array = np.linspace(0, 10, 200)
    X, T = np.meshgrid(x, t_array)

    U_constructive = np.zeros_like(X)
    for i, t in enumerate(t_array):
        pulse1 = gaussian_pulse(x, x1_start + c*t, sigma)
        pulse2 = gaussian_pulse(x, x2_start - c*t, sigma)
        U_constructive[i, :] = pulse1 + pulse2

    im4 = ax4.imshow(U_constructive, aspect='auto',
                     extent=[x.min(), x.max(), t_array.max(), t_array.min()],
                     cmap='hot', vmin=0, vmax=2)
    ax4.set_xlabel('Position')
    ax4.set_ylabel('Time')
    ax4.set_title('Space-Time: Constructive')
    plt.colorbar(im4, ax=ax4, label='Amplitude')

    # =========================================================================
    # Panel 5: Space-time diagram - Destructive
    # =========================================================================
    ax5 = fig.add_subplot(3, 3, 5)

    U_destructive = np.zeros_like(X)
    for i, t in enumerate(t_array):
        pulse1 = gaussian_pulse(x, x1_start + c*t, sigma)
        pulse2 = -gaussian_pulse(x, x2_start - c*t, sigma)
        U_destructive[i, :] = pulse1 + pulse2

    im5 = ax5.imshow(U_destructive, aspect='auto',
                     extent=[x.min(), x.max(), t_array.max(), t_array.min()],
                     cmap='RdBu', vmin=-1, vmax=1)
    ax5.set_xlabel('Position')
    ax5.set_ylabel('Time')
    ax5.set_title('Space-Time: Destructive')
    plt.colorbar(im5, ax=ax5, label='Amplitude')

    # =========================================================================
    # Panel 6: Different pulse widths
    # =========================================================================
    ax6 = fig.add_subplot(3, 3, 6)

    # Wider pulse collides with narrower pulse
    sigma1, sigma2 = 2.0, 0.5
    t_diff = 4.0

    pulse1 = gaussian_pulse(x, x1_start + c*t_diff, sigma1)
    pulse2 = gaussian_pulse(x, x2_start - c*t_diff, sigma2)
    total = pulse1 + pulse2

    ax6.plot(x, pulse1, 'b--', lw=2, label=f'Wide pulse (sigma={sigma1})')
    ax6.plot(x, pulse2, 'r--', lw=2, label=f'Narrow pulse (sigma={sigma2})')
    ax6.plot(x, total, 'k-', lw=2, label='Superposition')
    ax6.fill_between(x, total, alpha=0.3)

    ax6.set_xlabel('Position')
    ax6.set_ylabel('Displacement')
    ax6.set_title('Different Width Pulses\nSuperposition at t = 4')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(-5, 5)

    # =========================================================================
    # Panel 7: Soliton collision (shape-preserving)
    # =========================================================================
    ax7 = fig.add_subplot(3, 3, 7)

    width = 1.0
    times_soliton = [0, 2, 4, 6, 8]
    colors_sol = plt.cm.cool(np.linspace(0.1, 0.9, len(times_soliton)))

    for t, color in zip(times_soliton, colors_sol):
        sol1 = sech_pulse(x, x1_start + c*t, width)
        sol2 = sech_pulse(x, x2_start - c*t, width)
        total = sol1 + sol2

        ax7.plot(x, total + 2.5*t, color=color, lw=2, label=f't = {t}')

    ax7.set_xlabel('Position')
    ax7.set_ylabel('Displacement (offset)')
    ax7.set_title('Sech Pulses (Soliton Shape)\nPass through each other unchanged')
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)
    ax7.set_xlim(-15, 15)

    # =========================================================================
    # Panel 8: Multiple pulse collision
    # =========================================================================
    ax8 = fig.add_subplot(3, 3, 8)

    # Three pulses
    positions = [-10, 0, 10]
    velocities = [1, 0, -1]

    t_multi = np.linspace(0, 5, 6)
    colors_multi = plt.cm.viridis(np.linspace(0.1, 0.9, len(t_multi)))

    for t, color in zip(t_multi, colors_multi):
        total = np.zeros_like(x)
        for x0, v in zip(positions, velocities):
            total += gaussian_pulse(x, x0 + v*t, sigma)

        ax8.plot(x, total + 2*t, color=color, lw=2, label=f't = {t:.1f}')

    ax8.set_xlabel('Position')
    ax8.set_ylabel('Displacement (offset)')
    ax8.set_title('Three-Pulse Collision\n(Linear superposition)')
    ax8.legend(fontsize=8)
    ax8.grid(True, alpha=0.3)
    ax8.set_xlim(-15, 15)

    # =========================================================================
    # Panel 9: Energy during collision
    # =========================================================================
    ax9 = fig.add_subplot(3, 3, 9)

    # Track energy through collision
    t_energy = np.linspace(0, 8, 100)
    energy_const = []
    energy_dest = []

    for t in t_energy:
        # Constructive case
        pulse1 = gaussian_pulse(x, x1_start + c*t, sigma)
        pulse2 = gaussian_pulse(x, x2_start - c*t, sigma)
        total_c = pulse1 + pulse2

        # Destructive case
        pulse1_d = gaussian_pulse(x, x1_start + c*t, sigma)
        pulse2_d = -gaussian_pulse(x, x2_start - c*t, sigma)
        total_d = pulse1_d + pulse2_d

        # Energy proportional to integral of amplitude squared
        energy_const.append(np.sum(total_c**2) * dx)
        energy_dest.append(np.sum(total_d**2) * dx)

    ax9.plot(t_energy, energy_const, 'b-', lw=2, label='Constructive')
    ax9.plot(t_energy, energy_dest, 'r--', lw=2, label='Destructive')

    # Individual pulse energies
    single_energy = np.sum(gaussian_pulse(x, 0, sigma)**2) * dx
    ax9.axhline(y=2*single_energy, color='gray', linestyle=':',
                label=f'2 x single = {2*single_energy:.2f}')

    ax9.axvline(x=4, color='green', linestyle='--', alpha=0.5, label='Collision')
    ax9.set_xlabel('Time')
    ax9.set_ylabel('Energy (integral of u^2)')
    ax9.set_title('Energy During Collision\n(Energy redistributes, not destroyed)')
    ax9.legend(fontsize=8)
    ax9.grid(True, alpha=0.3)

    plt.suptitle('Superposition Principle: Interference of Wave Pulses\n'
                 'u_total(x,t) = u_1(x,t) + u_2(x,t) (linear superposition)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'pulse_superposition.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'pulse_superposition.png')}")


if __name__ == "__main__":
    main()
