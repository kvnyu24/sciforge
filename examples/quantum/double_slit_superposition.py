"""
Experiment 166: Double Slit Two-Path Superposition

This experiment demonstrates the fundamental quantum superposition principle
in a double-slit setup, including:
- Path superposition and interference
- Probability amplitudes from each slit
- Effect of relative phase
- Single particle vs ensemble behavior
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def slit_amplitude(x: np.ndarray, slit_pos: float, slit_width: float,
                   wavelength: float, L: float) -> np.ndarray:
    """
    Calculate amplitude contribution from a single slit.

    Using Fraunhofer diffraction:
    psi(x) = sinc(k*a*(x-x_s)/(2*L)) * exp(i*k*r)

    where k = 2*pi/lambda, a = slit width, L = screen distance

    Args:
        x: Position on screen
        slit_pos: Position of slit center
        slit_width: Width of slit
        wavelength: de Broglie wavelength
        L: Distance from slits to screen

    Returns:
        Complex amplitude array
    """
    k = 2 * np.pi / wavelength

    # Path length from slit to screen point
    r = np.sqrt(L**2 + (x - slit_pos)**2)

    # Diffraction envelope (sinc)
    arg = k * slit_width * (x - slit_pos) / (2 * L)
    envelope = np.sinc(arg / np.pi)

    # Phase from path length
    phase = np.exp(1j * k * r)

    return envelope * phase


def double_slit_interference(x: np.ndarray, d: float, a: float,
                              wavelength: float, L: float,
                              relative_phase: float = 0) -> tuple:
    """
    Calculate double slit interference pattern.

    Args:
        x: Position array on screen
        d: Slit separation
        a: Slit width
        wavelength: de Broglie wavelength
        L: Screen distance
        relative_phase: Phase difference between slits

    Returns:
        Tuple of (total_probability, slit1_prob, slit2_prob, interference_term)
    """
    # Amplitudes from each slit
    psi_1 = slit_amplitude(x, -d/2, a, wavelength, L)
    psi_2 = slit_amplitude(x, d/2, a, wavelength, L) * np.exp(1j * relative_phase)

    # Total amplitude (superposition)
    psi_total = psi_1 + psi_2

    # Probabilities
    P_1 = np.abs(psi_1)**2
    P_2 = np.abs(psi_2)**2
    P_total = np.abs(psi_total)**2

    # Interference term: P_total = P_1 + P_2 + 2*Re(psi_1* psi_2)
    interference = P_total - P_1 - P_2

    return P_total, P_1, P_2, interference


def simulate_single_particles(x: np.ndarray, d: float, a: float,
                               wavelength: float, L: float,
                               n_particles: int) -> np.ndarray:
    """
    Simulate detection of individual particles.

    Each particle is detected at a random position according to |psi|^2.

    Args:
        x: Position array
        d: Slit separation
        a: Slit width
        wavelength: de Broglie wavelength
        L: Screen distance
        n_particles: Number of particles

    Returns:
        Array of detection positions
    """
    P_total, _, _, _ = double_slit_interference(x, d, a, wavelength, L)

    # Normalize to get probability distribution
    P_norm = P_total / np.sum(P_total)

    # Sample from distribution
    positions = np.random.choice(x, size=n_particles, p=P_norm)

    return positions


def main():
    # Parameters
    wavelength = 1.0  # de Broglie wavelength
    d = 5.0          # Slit separation
    a = 1.0          # Slit width
    L = 100.0        # Screen distance

    # Screen coordinates
    x = np.linspace(-30, 30, 1000)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Plot 1: Double slit pattern showing superposition
    ax1 = axes[0, 0]

    P_total, P_1, P_2, interference = double_slit_interference(x, d, a, wavelength, L)

    ax1.plot(x, P_total, 'b-', lw=2, label='|psi_1 + psi_2|^2')
    ax1.plot(x, P_1, 'r--', lw=1.5, alpha=0.7, label='|psi_1|^2 (slit 1 only)')
    ax1.plot(x, P_2, 'g--', lw=1.5, alpha=0.7, label='|psi_2|^2 (slit 2 only)')
    ax1.plot(x, P_1 + P_2, 'k:', lw=1.5, alpha=0.7, label='|psi_1|^2 + |psi_2|^2')

    ax1.set_xlabel('Position on Screen')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('Double Slit Superposition\nP != P_1 + P_2 (interference!)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Interference term
    ax2 = axes[0, 1]

    ax2.plot(x, P_1 + P_2, 'b-', lw=2, label='P_1 + P_2 (classical)')
    ax2.plot(x, interference, 'r-', lw=2, label='Interference term')
    ax2.plot(x, P_total, 'g--', lw=2, label='P_total (quantum)')

    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.fill_between(x, 0, interference, where=interference > 0,
                     alpha=0.3, color='red', label='Constructive')
    ax2.fill_between(x, 0, interference, where=interference < 0,
                     alpha=0.3, color='blue', label='Destructive')

    ax2.set_xlabel('Position on Screen')
    ax2.set_ylabel('Probability / Interference')
    ax2.set_title('Interference = 2*Re(psi_1* psi_2)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Effect of relative phase
    ax3 = axes[0, 2]

    phases = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(phases)))

    for phase, color in zip(phases, colors):
        P_phase, _, _, _ = double_slit_interference(x, d, a, wavelength, L, phase)
        ax3.plot(x, P_phase + phase, color=color, lw=1.5,
                label=f'phi = {phase/np.pi:.2f}*pi')

    ax3.set_xlabel('Position on Screen')
    ax3.set_ylabel('Probability (offset by phase)')
    ax3.set_title('Effect of Relative Phase Between Slits')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Single particle detection buildup
    ax4 = axes[1, 0]

    n_particles_list = [10, 100, 1000, 10000]
    colors_n = plt.cm.plasma(np.linspace(0.2, 0.9, len(n_particles_list)))

    np.random.seed(42)

    for n_particles, color in zip(n_particles_list, colors_n):
        positions = simulate_single_particles(x, d, a, wavelength, L, n_particles)
        hist, bins = np.histogram(positions, bins=50, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        ax4.plot(bin_centers, hist + len(n_particles_list) - n_particles_list.index(n_particles) - 1,
                color=color, lw=2, label=f'N = {n_particles}')

    ax4.set_xlabel('Position on Screen')
    ax4.set_ylabel('Detection frequency (offset)')
    ax4.set_title('Single Particle Detection Buildup\n(Pattern emerges statistically)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: 2D visualization
    ax5 = axes[1, 1]

    # Create 2D intensity pattern
    y_screen = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y_screen)

    # Gaussian envelope in y
    I_2d = np.zeros_like(X)
    for i, yi in enumerate(y_screen):
        P_x, _, _, _ = double_slit_interference(x, d, a, wavelength, L)
        # Add y-dependence (approximate as separable)
        envelope_y = np.exp(-yi**2 / 20)
        I_2d[i, :] = P_x * envelope_y

    im = ax5.imshow(I_2d, extent=[x.min(), x.max(), y_screen.min(), y_screen.max()],
                    aspect='auto', cmap='hot', origin='lower')
    plt.colorbar(im, ax=ax5, label='Intensity')

    ax5.set_xlabel('x position')
    ax5.set_ylabel('y position')
    ax5.set_title('2D Interference Pattern')

    # Plot 6: Path integral perspective
    ax6 = axes[1, 2]

    # Show how amplitudes add
    # Draw schematic of double slit setup

    # Slits
    ax6.plot([-0.5, -0.5], [d/2 - a/2, d/2 + a/2], 'k-', lw=10)
    ax6.plot([-0.5, -0.5], [-d/2 - a/2, -d/2 + a/2], 'k-', lw=10)

    # Barrier
    ax6.fill_between([-0.6, -0.4], -20, d/2 - a/2, color='gray', alpha=0.5)
    ax6.fill_between([-0.6, -0.4], d/2 + a/2, -d/2 - a/2, color='gray', alpha=0.5)
    ax6.fill_between([-0.6, -0.4], -d/2 + a/2, 20, color='gray', alpha=0.5)

    # Screen
    ax6.axvline(x=10, color='blue', lw=3)

    # Paths to a detection point
    x_detect = 10
    y_detect = 3

    ax6.plot([-2, -0.5, x_detect], [0, d/2, y_detect], 'r-', lw=2, alpha=0.7,
             label='Path 1')
    ax6.plot([-2, -0.5, x_detect], [0, -d/2, y_detect], 'g-', lw=2, alpha=0.7,
             label='Path 2')

    ax6.scatter([x_detect], [y_detect], s=100, c='purple', zorder=5)
    ax6.text(x_detect + 0.5, y_detect, 'Detection', fontsize=10)

    # Source
    ax6.scatter([-2], [0], s=100, c='orange', zorder=5)
    ax6.text(-2, 0.5, 'Source', fontsize=10)

    ax6.set_xlabel('z')
    ax6.set_ylabel('y')
    ax6.set_title('Path Amplitude Superposition\npsi = psi_1 + psi_2')
    ax6.legend()
    ax6.set_xlim(-4, 12)
    ax6.set_ylim(-10, 10)
    ax6.set_aspect('equal')

    # Add equation
    ax6.text(-3, -8, r'$\psi = A_1 e^{i\phi_1} + A_2 e^{i\phi_2}$', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax6.text(-3, -9.5, r'$P = |\psi|^2 \neq |A_1|^2 + |A_2|^2$', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle('Double Slit: Two-Path Quantum Superposition\n'
                 'Each particle goes through both slits!',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'double_slit_superposition.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'double_slit_superposition.png')}")


if __name__ == "__main__":
    main()
