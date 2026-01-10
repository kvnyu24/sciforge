"""
Example 115: Jones Calculus and Wave Plates

This example demonstrates Jones calculus for polarization optics,
showing how wave plates transform polarization states.

Physics:
    Jones vector: E = [Ex, Ey]^T represents polarization state

    Common Jones matrices:
    - Linear polarizer at angle theta: P(theta) = [cos^2, cos*sin; cos*sin, sin^2]
    - Quarter-wave plate: QWP = e^(i*pi/4) * [1, 0; 0, i] (fast axis horizontal)
    - Half-wave plate: HWP = [1, 0; 0, -1] (fast axis horizontal)
    - General retarder: R(delta, theta) with retardance delta and fast axis at theta

    Wave plate action: E_out = M * E_in
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D
from src.sciforge.physics.optics import JonesVector, JonesMatrix


class PolarizationEllipse:
    """Helper class to visualize polarization ellipse from Jones vector"""

    def __init__(self, jones_vector):
        """
        Args:
            jones_vector: JonesVector or array [Ex, Ey]
        """
        if hasattr(jones_vector, 'E'):
            self.Ex, self.Ey = jones_vector.E
        else:
            self.Ex, self.Ey = jones_vector

    def get_ellipse_params(self):
        """
        Calculate ellipse parameters (semi-axes, orientation, handedness).

        Returns:
            (a, b, theta, handedness) where a >= b are semi-axes,
            theta is orientation angle, handedness is +1 (RH) or -1 (LH)
        """
        # Stokes parameters
        Ex, Ey = self.Ex, self.Ey
        S0 = np.abs(Ex)**2 + np.abs(Ey)**2
        S1 = np.abs(Ex)**2 - np.abs(Ey)**2
        S2 = 2 * np.real(Ex * np.conj(Ey))
        S3 = 2 * np.imag(Ex * np.conj(Ey))

        # Normalize
        if S0 > 0:
            S1, S2, S3 = S1/S0, S2/S0, S3/S0

        # Ellipse orientation
        theta = 0.5 * np.arctan2(S2, S1)

        # Ellipticity angle
        chi = 0.5 * np.arcsin(S3)

        # Semi-axes
        a = np.cos(chi)
        b = np.abs(np.sin(chi))

        # Handedness
        handedness = np.sign(S3) if abs(S3) > 1e-10 else 0

        return a, b, theta, handedness

    def plot(self, ax, color='blue', label=None):
        """Plot polarization ellipse on axes"""
        a, b, theta, handedness = self.get_ellipse_params()

        # Generate ellipse points
        t = np.linspace(0, 2*np.pi, 100)
        x = a * np.cos(t)
        y = b * np.sin(t)

        # Rotate
        x_rot = x * np.cos(theta) - y * np.sin(theta)
        y_rot = x * np.sin(theta) + y * np.cos(theta)

        ax.plot(x_rot, y_rot, color=color, linewidth=2, label=label)

        # Add arrow to show direction
        idx = len(t) // 4
        dx = x_rot[idx+1] - x_rot[idx-1]
        dy = y_rot[idx+1] - y_rot[idx-1]
        if handedness < 0:
            dx, dy = -dx, -dy
        ax.annotate('', xy=(x_rot[idx] + dx*0.1, y_rot[idx] + dy*0.1),
                   xytext=(x_rot[idx], y_rot[idx]),
                   arrowprops=dict(arrowstyle='->', color=color))


def create_common_waveplates():
    """Create common waveplate configurations"""

    waveplates = {
        'QWP_0': JonesMatrix.quarter_wave_plate(0),
        'QWP_45': JonesMatrix.quarter_wave_plate(np.pi/4),
        'QWP_-45': JonesMatrix.quarter_wave_plate(-np.pi/4),
        'HWP_0': JonesMatrix.half_wave_plate(0),
        'HWP_22.5': JonesMatrix.half_wave_plate(np.pi/8),
        'HWP_45': JonesMatrix.half_wave_plate(np.pi/4),
    }

    return waveplates


def plot_jones_vectors():
    """Plot common polarization states as Jones vectors"""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    states = [
        (JonesVector.horizontal(), 'Horizontal Linear'),
        (JonesVector.vertical(), 'Vertical Linear'),
        (JonesVector.diagonal(np.pi/4), 'Diagonal (+45 deg)'),
        (JonesVector.right_circular(), 'Right Circular'),
        (JonesVector.left_circular(), 'Left Circular'),
        (JonesVector.elliptical(1.0, 0.5, np.pi/6), 'Elliptical'),
    ]

    for ax, (jones, title) in zip(axes, states):
        ellipse = PolarizationEllipse(jones)
        ellipse.plot(ax, color='blue')

        # Add coordinate axes
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.set_xlabel('Ex')
        ax.set_ylabel('Ey')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # Add Jones vector components
        Ex, Ey = jones.E
        ax.text(0.02, 0.98, f'Ex = {Ex:.2f}\nEy = {Ey:.2f}',
               transform=ax.transAxes, va='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    return fig


def plot_waveplate_transformations():
    """Plot how waveplates transform polarization states"""

    fig = plt.figure(figsize=(16, 12))

    # Quarter-wave plate transformations
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title('Quarter-Wave Plate (fast axis horizontal)')

    input_states = [
        (JonesVector.horizontal(), 'Horizontal', 'blue'),
        (JonesVector.vertical(), 'Vertical', 'red'),
        (JonesVector.diagonal(np.pi/4), '+45 deg', 'green'),
        (JonesVector.diagonal(-np.pi/4), '-45 deg', 'purple'),
    ]

    qwp = JonesMatrix.quarter_wave_plate(0)

    for jones_in, label, color in input_states:
        # Input state (dashed)
        ellipse_in = PolarizationEllipse(jones_in)
        a, b, theta, _ = ellipse_in.get_ellipse_params()
        t = np.linspace(0, 2*np.pi, 100)
        x = a * np.cos(t) * np.cos(theta) - b * np.sin(t) * np.sin(theta)
        y = a * np.cos(t) * np.sin(theta) + b * np.sin(t) * np.cos(theta)
        ax1.plot(x * 0.4, y * 0.4, '--', color=color, alpha=0.5)

        # Output state (solid)
        jones_out = qwp.apply(jones_in)
        ellipse_out = PolarizationEllipse(jones_out)
        ellipse_out.plot(ax1, color=color, label=f'{label} -> Output')

    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-1.2, 1.2)
    ax1.set_aspect('equal')
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.axvline(0, color='black', linewidth=0.5)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Half-wave plate transformations
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title('Half-Wave Plate (fast axis horizontal)')

    hwp = JonesMatrix.half_wave_plate(0)

    for jones_in, label, color in input_states:
        jones_out = hwp.apply(jones_in)
        ellipse_out = PolarizationEllipse(jones_out)
        ellipse_out.plot(ax2, color=color, label=f'{label}')

    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_aspect('equal')
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.axvline(0, color='black', linewidth=0.5)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Effect of QWP angle on circular polarization generation
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title('QWP Rotation: Linear to Circular Polarization')

    angles = np.linspace(0, np.pi/2, 5)
    colors = plt.cm.viridis(np.linspace(0, 1, len(angles)))

    linear_input = JonesVector.horizontal()

    for angle, color in zip(angles, colors):
        qwp_rotated = JonesMatrix.quarter_wave_plate(angle)
        jones_out = qwp_rotated.apply(linear_input)
        ellipse = PolarizationEllipse(jones_out)
        ellipse.plot(ax3, color=color, label=f'QWP at {np.degrees(angle):.0f} deg')

    ax3.set_xlim(-1.2, 1.2)
    ax3.set_ylim(-1.2, 1.2)
    ax3.set_aspect('equal')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Variable retardance plate
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title('Variable Retarder: Effect of Retardance')

    retardances = np.linspace(0, np.pi, 6)
    colors = plt.cm.plasma(np.linspace(0, 1, len(retardances)))

    diagonal_input = JonesVector.diagonal(np.pi/4)

    for retard, color in zip(retardances, colors):
        waveplate = JonesMatrix.waveplate(retard, 0)
        jones_out = waveplate.apply(diagonal_input)
        ellipse = PolarizationEllipse(jones_out)
        ellipse.plot(ax4, color=color, label=f'delta = {np.degrees(retard):.0f} deg')

    ax4.set_xlim(-1.2, 1.2)
    ax4.set_ylim(-1.2, 1.2)
    ax4.set_aspect('equal')
    ax4.legend(fontsize=8, loc='upper right')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_poincare_sphere():
    """Plot polarization states on Poincare sphere"""

    fig = plt.figure(figsize=(14, 6))

    # 3D Poincare sphere
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')

    # Draw sphere wireframe
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_wireframe(x, y, z, alpha=0.1, color='gray')

    # Mark special points
    special_states = [
        (JonesVector.horizontal(), 'H', 'red'),
        (JonesVector.vertical(), 'V', 'red'),
        (JonesVector.diagonal(np.pi/4), '+45', 'blue'),
        (JonesVector.diagonal(-np.pi/4), '-45', 'blue'),
        (JonesVector.right_circular(), 'RCP', 'green'),
        (JonesVector.left_circular(), 'LCP', 'green'),
    ]

    for jones, label, color in special_states:
        Ex, Ey = jones.E
        S0 = np.abs(Ex)**2 + np.abs(Ey)**2
        S1 = (np.abs(Ex)**2 - np.abs(Ey)**2) / S0
        S2 = 2 * np.real(Ex * np.conj(Ey)) / S0
        S3 = 2 * np.imag(Ex * np.conj(Ey)) / S0

        ax1.scatter([S1], [S2], [S3], s=100, color=color)
        ax1.text(S1*1.1, S2*1.1, S3*1.1, label, fontsize=10)

    # Draw axes
    ax1.plot([-1.2, 1.2], [0, 0], [0, 0], 'r-', linewidth=1)
    ax1.plot([0, 0], [-1.2, 1.2], [0, 0], 'b-', linewidth=1)
    ax1.plot([0, 0], [0, 0], [-1.2, 1.2], 'g-', linewidth=1)

    ax1.set_xlabel('S1 (H-V)')
    ax1.set_ylabel('S2 (+45/-45)')
    ax1.set_zlabel('S3 (RCP-LCP)')
    ax1.set_title('Poincare Sphere')

    # Trace QWP transformation on sphere
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_wireframe(x, y, z, alpha=0.1, color='gray')

    # Start with horizontal
    jones_start = JonesVector.horizontal()

    # Apply QWP at various angles and trace path
    qwp_angles = np.linspace(0, np.pi/4, 20)
    S1_path, S2_path, S3_path = [], [], []

    for angle in qwp_angles:
        qwp = JonesMatrix.quarter_wave_plate(angle)
        jones_out = qwp.apply(jones_start)
        Ex, Ey = jones_out.E
        S0 = np.abs(Ex)**2 + np.abs(Ey)**2
        S1_path.append((np.abs(Ex)**2 - np.abs(Ey)**2) / S0)
        S2_path.append(2 * np.real(Ex * np.conj(Ey)) / S0)
        S3_path.append(2 * np.imag(Ex * np.conj(Ey)) / S0)

    ax2.plot(S1_path, S2_path, S3_path, 'b-', linewidth=2, label='QWP rotation')
    ax2.scatter([S1_path[0]], [S2_path[0]], [S3_path[0]], s=100, color='green', label='Start (H)')
    ax2.scatter([S1_path[-1]], [S2_path[-1]], [S3_path[-1]], s=100, color='red', label='End (RCP)')

    # Draw axes
    ax2.plot([-1.2, 1.2], [0, 0], [0, 0], 'r-', linewidth=1, alpha=0.5)
    ax2.plot([0, 0], [-1.2, 1.2], [0, 0], 'b-', linewidth=1, alpha=0.5)
    ax2.plot([0, 0], [0, 0], [-1.2, 1.2], 'g-', linewidth=1, alpha=0.5)

    ax2.set_xlabel('S1')
    ax2.set_ylabel('S2')
    ax2.set_zlabel('S3')
    ax2.set_title('QWP Transformation Path on Poincare Sphere\n(H -> RCP as QWP rotates 0 to 45 deg)')
    ax2.legend()

    plt.tight_layout()
    return fig


def plot_optical_isolator():
    """Demonstrate optical isolator using polarizer and Faraday rotator"""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Optical isolator: Polarizer -> Faraday rotator (45 deg) -> Polarizer at 45 deg

    # Forward direction
    ax1, ax2, ax3 = axes[0, :]
    ax1.set_title('Forward: After Polarizer (V)')
    ax2.set_title('After Faraday Rotator (+45 deg)')
    ax3.set_title('After Output Polarizer (45 deg)')

    # Input: Unpolarized -> After V polarizer: Vertical
    input_state = JonesVector.vertical()
    ellipse = PolarizationEllipse(input_state)
    ellipse.plot(ax1, 'blue')

    # After Faraday rotator (45 degree rotation)
    faraday = JonesMatrix.rotator(np.pi/4)
    after_faraday = faraday.apply(input_state)
    ellipse = PolarizationEllipse(after_faraday)
    ellipse.plot(ax2, 'blue')

    # After output polarizer at 45 degrees (same as polarization direction)
    output_polarizer = JonesMatrix.linear_polarizer(np.pi/4)
    output_state = output_polarizer.apply(after_faraday)
    ellipse = PolarizationEllipse(output_state)
    ellipse.plot(ax3, 'blue')

    # Calculate transmission
    transmission_forward = np.abs(output_state.E[0])**2 + np.abs(output_state.E[1])**2
    ax3.text(0.5, -0.5, f'T = {transmission_forward:.2f}', fontsize=12)

    for ax in axes[0, :]:
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.grid(True, alpha=0.3)

    # Reverse direction
    ax4, ax5, ax6 = axes[1, :]
    ax4.set_title('Reverse: After Polarizer (45 deg)')
    ax5.set_title('After Faraday Rotator (+45 deg)')
    ax6.set_title('After Input Polarizer (V) - BLOCKED')

    # Input from reverse: 45 degree polarization
    reverse_input = JonesVector.diagonal(np.pi/4)
    ellipse = PolarizationEllipse(reverse_input)
    ellipse.plot(ax4, 'red')

    # After Faraday rotator (NON-RECIPROCAL: still +45 deg rotation!)
    after_faraday_rev = faraday.apply(reverse_input)
    ellipse = PolarizationEllipse(after_faraday_rev)
    ellipse.plot(ax5, 'red')
    # Now at 90 degrees (horizontal)

    # After input polarizer (vertical) - blocked!
    input_polarizer = JonesMatrix.linear_polarizer(np.pi/2)
    output_state_rev = input_polarizer.apply(after_faraday_rev)
    ellipse = PolarizationEllipse(output_state_rev)
    ellipse.plot(ax6, 'red')

    transmission_reverse = np.abs(output_state_rev.E[0])**2 + np.abs(output_state_rev.E[1])**2
    ax6.text(0.5, -0.5, f'T = {transmission_reverse:.2e}', fontsize=12)

    for ax in axes[1, :]:
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Optical Isolator: Non-Reciprocal Faraday Rotation', fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def plot_waveplate_retardance_effects():
    """Plot effect of retardance on polarization"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Retardance scan for diagonal input
    ax1 = axes[0, 0]

    retardances = np.linspace(0, 2*np.pi, 100)
    input_state = JonesVector.diagonal(np.pi/4)

    ellipticity = []
    orientation = []

    for delta in retardances:
        waveplate = JonesMatrix.waveplate(delta, 0)
        output = waveplate.apply(input_state)
        ellipse = PolarizationEllipse(output)
        a, b, theta, _ = ellipse.get_ellipse_params()
        ellipticity.append(b / a if a > 0 else 0)
        orientation.append(theta)

    ax1.plot(np.degrees(retardances), ellipticity, 'b-', linewidth=2, label='Ellipticity')
    ax1.axhline(1, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(90, color='red', linestyle='--', alpha=0.5, label='QWP (90 deg)')
    ax1.axvline(180, color='green', linestyle='--', alpha=0.5, label='HWP (180 deg)')
    ax1.axvline(270, color='red', linestyle='--', alpha=0.5)

    ax1.set_xlabel('Retardance (degrees)')
    ax1.set_ylabel('Ellipticity |b/a|')
    ax1.set_title('Ellipticity vs Retardance\n(Diagonal input, fast axis horizontal)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Output on Poincare sphere as retardance varies
    ax2 = axes[0, 1]

    S1_path, S3_path = [], []
    for delta in retardances:
        waveplate = JonesMatrix.waveplate(delta, 0)
        output = waveplate.apply(input_state)
        Ex, Ey = output.E
        S0 = np.abs(Ex)**2 + np.abs(Ey)**2
        S1_path.append((np.abs(Ex)**2 - np.abs(Ey)**2) / S0)
        S3_path.append(2 * np.imag(Ex * np.conj(Ey)) / S0)

    ax2.plot(np.degrees(retardances), S1_path, 'b-', linewidth=2, label='S1')
    ax2.plot(np.degrees(retardances), S3_path, 'r-', linewidth=2, label='S3')

    ax2.axvline(90, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(180, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(270, color='gray', linestyle='--', alpha=0.5)

    ax2.set_xlabel('Retardance (degrees)')
    ax2.set_ylabel('Stokes parameter')
    ax2.set_title('Stokes Parameters vs Retardance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Intensity after crossed polarizers with waveplate
    ax3 = axes[1, 0]

    # Setup: H-polarizer -> waveplate -> V-polarizer
    h_pol = JonesMatrix.linear_polarizer(0)
    v_pol = JonesMatrix.linear_polarizer(np.pi/2)

    input_light = JonesVector.horizontal()

    intensity_45 = []
    intensity_0 = []

    for delta in retardances:
        # Waveplate at 45 degrees
        wp_45 = JonesMatrix.waveplate(delta, np.pi/4)
        output_45 = v_pol.apply(wp_45.apply(input_light))
        I_45 = np.abs(output_45.E[0])**2 + np.abs(output_45.E[1])**2
        intensity_45.append(I_45)

        # Waveplate at 0 degrees (no effect on crossed polarizers)
        wp_0 = JonesMatrix.waveplate(delta, 0)
        output_0 = v_pol.apply(wp_0.apply(input_light))
        I_0 = np.abs(output_0.E[0])**2 + np.abs(output_0.E[1])**2
        intensity_0.append(I_0)

    ax3.plot(np.degrees(retardances), intensity_45, 'b-', linewidth=2, label='WP at 45 deg')
    ax3.plot(np.degrees(retardances), intensity_0, 'r--', linewidth=2, label='WP at 0 deg')

    ax3.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax3.axvline(180, color='green', linestyle='--', alpha=0.5, label='HWP: max')

    ax3.set_xlabel('Retardance (degrees)')
    ax3.set_ylabel('Transmitted intensity')
    ax3.set_title('Crossed Polarizers with Waveplate\n(Intensity = sin^2(delta/2) * sin^2(2*theta))')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Waveplate angle scan
    ax4 = axes[1, 1]

    angles = np.linspace(0, np.pi, 100)

    intensity_qwp = []
    intensity_hwp = []

    for theta in angles:
        # QWP (90 deg retardance)
        qwp = JonesMatrix.waveplate(np.pi/2, theta)
        output = v_pol.apply(qwp.apply(input_light))
        intensity_qwp.append(np.abs(output.E[0])**2 + np.abs(output.E[1])**2)

        # HWP (180 deg retardance)
        hwp = JonesMatrix.waveplate(np.pi, theta)
        output = v_pol.apply(hwp.apply(input_light))
        intensity_hwp.append(np.abs(output.E[0])**2 + np.abs(output.E[1])**2)

    ax4.plot(np.degrees(angles), intensity_qwp, 'b-', linewidth=2, label='QWP')
    ax4.plot(np.degrees(angles), intensity_hwp, 'r-', linewidth=2, label='HWP')

    ax4.set_xlabel('Waveplate angle (degrees)')
    ax4.set_ylabel('Transmitted intensity')
    ax4.set_title('Crossed Polarizers: Waveplate Angle Scan')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    """Main function to demonstrate Jones calculus and waveplates"""

    # Create figures
    fig1 = plot_jones_vectors()
    fig2 = plot_waveplate_transformations()
    fig3 = plot_poincare_sphere()
    fig4 = plot_optical_isolator()
    fig5 = plot_waveplate_retardance_effects()

    # Save plots
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)

    fig1.savefig(os.path.join(output_dir, 'jones_vectors.png'),
                 dpi=150, bbox_inches='tight')
    fig2.savefig(os.path.join(output_dir, 'jones_waveplate_transforms.png'),
                 dpi=150, bbox_inches='tight')
    fig3.savefig(os.path.join(output_dir, 'jones_poincare_sphere.png'),
                 dpi=150, bbox_inches='tight')
    fig4.savefig(os.path.join(output_dir, 'jones_optical_isolator.png'),
                 dpi=150, bbox_inches='tight')
    fig5.savefig(os.path.join(output_dir, 'jones_retardance_effects.png'),
                 dpi=150, bbox_inches='tight')

    print(f"Plots saved to {output_dir}/jones_*.png")

    # Print analysis
    print("\n=== Jones Calculus Analysis ===")
    print("\nCommon Jones matrices:")
    print("  Linear polarizer (H): [[1,0],[0,0]]")
    print("  Quarter-wave plate (fast axis H): [[1,0],[0,i]]")
    print("  Half-wave plate (fast axis H): [[1,0],[0,-1]]")
    print("\nKey transformations:")
    print("  Linear + QWP at 45 deg -> Circular")
    print("  Circular + HWP -> Opposite circular handedness")
    print("  Linear at theta + HWP at theta/2 -> Rotate by theta")


if __name__ == "__main__":
    main()
