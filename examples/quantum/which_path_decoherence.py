"""
Experiment 167: Which-Path Decoherence

This experiment demonstrates how which-path information destroys quantum
interference, including:
- Measurement-induced decoherence
- Partial which-path information
- Quantum eraser concept
- Complementarity principle
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def double_slit_with_detector(x: np.ndarray, d: float, a: float,
                               wavelength: float, L: float,
                               detector_efficiency: float = 0) -> tuple:
    """
    Calculate double slit pattern with partial which-path detection.

    When detector_efficiency = 0: full interference (no detection)
    When detector_efficiency = 1: no interference (complete detection)

    The visibility of interference fringes decreases as:
    V = sqrt(1 - detector_efficiency^2)

    Args:
        x: Position array on screen
        d: Slit separation
        a: Slit width
        wavelength: de Broglie wavelength
        L: Screen distance
        detector_efficiency: Probability of detecting which path (0 to 1)

    Returns:
        Tuple of (total_probability, visibility)
    """
    k = 2 * np.pi / wavelength

    # Single slit diffraction envelope
    sinc_arg = k * a * x / (2 * L)
    envelope = np.sinc(sinc_arg / np.pi)**2

    # Interference pattern
    path_diff = d * x / L  # Approximate for small angles
    phase_diff = k * path_diff

    # With partial which-path info, coherence is reduced
    coherence = np.sqrt(1 - detector_efficiency**2)

    # Interference pattern: 1 + V*cos(phase)
    interference = 1 + coherence * np.cos(phase_diff)

    P_total = envelope * interference

    # Calculate visibility from the pattern
    P_max = envelope * (1 + coherence)
    P_min = envelope * (1 - coherence)
    visibility = coherence

    return P_total, visibility


def density_matrix_evolution(rho_init: np.ndarray,
                              which_path_strength: float) -> np.ndarray:
    """
    Model decoherence as reduction of off-diagonal elements.

    rho_01 -> rho_01 * exp(-gamma)

    where gamma represents the which-path information gained.

    Args:
        rho_init: Initial 2x2 density matrix
        which_path_strength: Decoherence parameter (0 to inf)

    Returns:
        Decohered density matrix
    """
    rho = rho_init.copy()
    decay = np.exp(-which_path_strength)
    rho[0, 1] *= decay
    rho[1, 0] *= decay
    return rho


def quantum_eraser_pattern(x: np.ndarray, d: float, a: float,
                            wavelength: float, L: float,
                            erased: bool = False) -> np.ndarray:
    """
    Calculate pattern with quantum eraser.

    If erased=True, which-path info is erased and interference returns.

    Args:
        x: Position array
        d: Slit separation
        a: Slit width
        wavelength: de Broglie wavelength
        L: Screen distance
        erased: Whether which-path info is erased

    Returns:
        Probability pattern
    """
    k = 2 * np.pi / wavelength

    # Envelope
    sinc_arg = k * a * x / (2 * L)
    envelope = np.sinc(sinc_arg / np.pi)**2

    # Phase
    path_diff = d * x / L
    phase_diff = k * path_diff

    if erased:
        # Interference restored
        pattern = envelope * (1 + np.cos(phase_diff))
    else:
        # No interference (which-path known)
        pattern = 2 * envelope

    return pattern


def main():
    # Parameters
    wavelength = 1.0
    d = 5.0
    a = 1.0
    L = 100.0

    x = np.linspace(-30, 30, 1000)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Plot 1: Interference vs which-path detection
    ax1 = axes[0, 0]

    detection_levels = [0, 0.3, 0.6, 0.9, 1.0]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(detection_levels)))

    for det, color in zip(detection_levels, colors):
        P, vis = double_slit_with_detector(x, d, a, wavelength, L, det)
        ax1.plot(x, P / max(P), color=color, lw=2,
                label=f'Detection = {det:.1f}, V = {vis:.2f}')

    ax1.set_xlabel('Position on Screen')
    ax1.set_ylabel('Normalized Probability')
    ax1.set_title('Effect of Which-Path Detection\non Interference')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Visibility vs detection efficiency
    ax2 = axes[0, 1]

    det_range = np.linspace(0, 1, 100)
    visibility = np.sqrt(1 - det_range**2)

    ax2.plot(det_range, visibility, 'b-', lw=2)
    ax2.fill_between(det_range, 0, visibility, alpha=0.2)

    ax2.set_xlabel('Which-Path Detection Efficiency')
    ax2.set_ylabel('Fringe Visibility')
    ax2.set_title('Complementarity: V^2 + D^2 <= 1\nVisibility vs Distinguishability')
    ax2.grid(True, alpha=0.3)

    # Add complementarity relation
    ax2.text(0.5, 0.5, r'$V^2 + D^2 \leq 1$', fontsize=14,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Plot 3: Density matrix visualization
    ax3 = axes[0, 2]

    # Show how coherence (off-diagonal) decreases
    gamma_values = [0, 0.5, 1, 2, 5]
    colors_gamma = plt.cm.plasma(np.linspace(0.2, 0.9, len(gamma_values)))

    # Initial coherent superposition |+> = (|0> + |1>)/sqrt(2)
    rho_init = np.array([[0.5, 0.5], [0.5, 0.5]])

    bar_width = 0.15
    positions = np.arange(4)
    labels = ['rho_00', 'rho_11', 'Re(rho_01)', 'Im(rho_01)']

    for i, (gamma, color) in enumerate(zip(gamma_values, colors_gamma)):
        rho = density_matrix_evolution(rho_init, gamma)
        values = [np.real(rho[0,0]), np.real(rho[1,1]),
                  np.real(rho[0,1]), np.imag(rho[0,1])]
        ax3.bar(positions + i*bar_width, values, bar_width, color=color,
                label=f'gamma = {gamma}')

    ax3.set_xticks(positions + 2*bar_width)
    ax3.set_xticklabels(labels)
    ax3.set_ylabel('Matrix Element Value')
    ax3.set_title('Density Matrix Decoherence\n(Off-diagonal decay)')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Quantum eraser schematic pattern
    ax4 = axes[1, 0]

    # Three cases: no detector, detector with sorting, detector no sorting
    P_no_det, _ = double_slit_with_detector(x, d, a, wavelength, L, 0)
    P_det, _ = double_slit_with_detector(x, d, a, wavelength, L, 1)
    P_erased = quantum_eraser_pattern(x, d, a, wavelength, L, erased=True)

    ax4.plot(x, P_no_det / max(P_no_det), 'b-', lw=2,
             label='No detector (full interference)')
    ax4.plot(x, P_det / max(P_det), 'r-', lw=2,
             label='With detector (no interference)')
    ax4.plot(x, P_erased / max(P_erased) + 2, 'g-', lw=2,
             label='Quantum eraser (interference restored)')

    ax4.set_xlabel('Position on Screen')
    ax4.set_ylabel('Probability (offset for clarity)')
    ax4.set_title('Quantum Eraser Effect')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Continuous decoherence
    ax5 = axes[1, 1]

    # Time evolution of interference as decoherence increases
    T_dec = 1.0  # Decoherence time scale

    times = np.linspace(0, 3*T_dec, 50)

    # Store patterns for 2D plot
    patterns = []

    for t in times:
        gamma = t / T_dec
        det_eff = 1 - np.exp(-gamma)
        P, _ = double_slit_with_detector(x, d, a, wavelength, L, det_eff)
        patterns.append(P)

    patterns = np.array(patterns)

    im = ax5.imshow(patterns, extent=[x.min(), x.max(), 0, times.max()/T_dec],
                    aspect='auto', cmap='hot', origin='lower')
    plt.colorbar(im, ax=ax5, label='Probability')

    ax5.set_xlabel('Position on Screen')
    ax5.set_ylabel('Time / T_decoherence')
    ax5.set_title('Continuous Decoherence\n(Interference fades over time)')

    # Plot 6: Complementarity diagram
    ax6 = axes[1, 2]

    # Draw the D-V circle
    theta = np.linspace(0, np.pi/2, 100)
    V_circ = np.cos(theta)
    D_circ = np.sin(theta)

    ax6.plot(D_circ, V_circ, 'b-', lw=3, label='V^2 + D^2 = 1')
    ax6.fill_between(D_circ, 0, V_circ, alpha=0.2)

    # Mark specific points
    points = [
        (0, 1, 'No detection\n(full wave)'),
        (1, 0, 'Full detection\n(full particle)'),
        (1/np.sqrt(2), 1/np.sqrt(2), 'Intermediate'),
    ]

    for D, V, label in points:
        ax6.scatter([D], [V], s=100, zorder=5)
        ax6.annotate(label, (D, V), textcoords='offset points',
                     xytext=(10, 10), fontsize=9)

    ax6.set_xlabel('Distinguishability D (path information)')
    ax6.set_ylabel('Visibility V (interference)')
    ax6.set_title('Wave-Particle Complementarity')
    ax6.set_xlim(-0.1, 1.1)
    ax6.set_ylim(-0.1, 1.1)
    ax6.set_aspect('equal')
    ax6.grid(True, alpha=0.3)
    ax6.legend()

    # Add note
    ax6.text(0.5, -0.05, 'More path info => Less interference', ha='center',
             fontsize=10, style='italic')

    plt.suptitle('Which-Path Information and Decoherence\n'
                 'Observation destroys quantum superposition',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'which_path_decoherence.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'which_path_decoherence.png')}")


if __name__ == "__main__":
    main()
