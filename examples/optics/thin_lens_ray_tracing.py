"""
Example 102: Thin Lens Ray Tracing

This example demonstrates ray tracing through thin lenses, showing:
- Image formation by converging and diverging lenses
- ABCD matrix formalism for paraxial rays
- Principal planes and focal points
- Magnification and image location

Physics:
    Thin lens equation: 1/f = 1/do + 1/di
    Magnification: m = -di/do

    ABCD matrix for thin lens:
    [1    0  ]
    [-1/f  1 ]
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Arc, Ellipse
from src.sciforge.physics.optics import ThinLens, OpticalSystem, Ray


class LensRayTracer:
    """Ray tracer for thin lens systems"""

    def __init__(self, focal_length: float, lens_position: float = 0.0,
                 lens_diameter: float = 0.1):
        """
        Args:
            focal_length: Focal length (positive for converging)
            lens_position: Position of lens along optical axis
            lens_diameter: Diameter of lens aperture
        """
        self.f = focal_length
        self.z_lens = lens_position
        self.diameter = lens_diameter
        self.lens = ThinLens(focal_length, lens_diameter, lens_position)

    def trace_ray(self, y_in: float, theta_in: float, z_start: float,
                  z_end: float) -> tuple:
        """
        Trace a paraxial ray through the lens.

        Args:
            y_in: Initial height
            theta_in: Initial angle
            z_start: Starting z position
            z_end: Ending z position

        Returns:
            (z_coords, y_coords) arrays
        """
        z_coords = []
        y_coords = []

        # Propagate to lens
        z_before = np.linspace(z_start, self.z_lens, 50)
        for z in z_before:
            y = y_in + theta_in * (z - z_start)
            z_coords.append(z)
            y_coords.append(y)

        y_at_lens = y_in + theta_in * (self.z_lens - z_start)

        # Check if ray hits lens
        if abs(y_at_lens) > self.diameter / 2:
            return np.array(z_coords), np.array(y_coords), True  # Ray blocked

        # Apply lens transformation using ABCD matrix
        # theta_out = theta_in - y/f
        theta_out = theta_in - y_at_lens / self.f

        # Propagate after lens
        z_after = np.linspace(self.z_lens, z_end, 50)
        for z in z_after:
            y = y_at_lens + theta_out * (z - self.z_lens)
            z_coords.append(z)
            y_coords.append(y)

        return np.array(z_coords), np.array(y_coords), False


def draw_lens(ax, z_pos, height, lens_type='converging'):
    """Draw a lens symbol"""
    if lens_type == 'converging':
        # Converging lens (thicker in middle)
        lens = Ellipse((z_pos, 0), 0.02, height, facecolor='lightblue',
                       edgecolor='blue', alpha=0.7, linewidth=2)
    else:
        # Diverging lens (thinner in middle)
        lens = Ellipse((z_pos, 0), 0.02, height, facecolor='lightyellow',
                       edgecolor='orange', alpha=0.7, linewidth=2)
    ax.add_patch(lens)


def draw_object_and_image(ax, z_obj, h_obj, z_img, h_img):
    """Draw object and image arrows"""
    # Object arrow
    ax.annotate('', xy=(z_obj, h_obj), xytext=(z_obj, 0),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.text(z_obj, h_obj + 0.01, 'Object', ha='center', va='bottom',
            color='green', fontsize=10)

    # Image arrow
    if np.isfinite(z_img) and np.isfinite(h_img):
        linestyle = 'dashed' if h_img * h_obj < 0 else 'solid'
        color = 'red' if h_img < 0 else 'red'
        ax.annotate('', xy=(z_img, h_img), xytext=(z_img, 0),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2,
                                   linestyle=linestyle))
        label = 'Real Image' if z_img > 0 else 'Virtual Image'
        ax.text(z_img, h_img + np.sign(h_img) * 0.01, label,
                ha='center', va='bottom' if h_img > 0 else 'top',
                color=color, fontsize=10)


def plot_converging_lens_cases():
    """Plot different cases for converging lens"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    cases = [
        ('Object beyond 2f', -0.3, 0.05, 0.1),   # do > 2f
        ('Object at 2f', -0.2, 0.05, 0.1),        # do = 2f
        ('Object between f and 2f', -0.15, 0.05, 0.1),  # f < do < 2f
        ('Object inside f (virtual image)', -0.05, 0.05, 0.1),  # do < f
    ]

    for ax, (title, z_obj, h_obj, f) in zip(axes.flat, cases):
        lens = ThinLens(focal_length=f, diameter=0.15)

        # Calculate image position
        do = -z_obj  # Object distance (positive)
        di = lens.image_distance(do)
        m = lens.magnification(do)
        z_img = di
        h_img = m * h_obj

        # Draw optical axis
        ax.axhline(0, color='black', linewidth=0.5)

        # Draw lens
        draw_lens(ax, 0, 0.15, 'converging')

        # Mark focal points
        ax.plot([-f, f], [0, 0], 'ko', markersize=8)
        ax.text(-f, -0.02, 'F', ha='center', va='top', fontsize=10)
        ax.text(f, -0.02, 'F\'', ha='center', va='top', fontsize=10)

        # Draw object and image
        draw_object_and_image(ax, z_obj, h_obj, z_img, h_img)

        # Trace principal rays
        tracer = LensRayTracer(f, 0.0, 0.15)
        colors = ['red', 'blue', 'purple']

        # Ray 1: Parallel to axis, through focal point after lens
        z1, y1, blocked1 = tracer.trace_ray(h_obj, 0, z_obj, 0.4)
        if not blocked1:
            ax.plot(z1, y1, color=colors[0], linewidth=1.5, alpha=0.7)

        # Ray 2: Through center of lens (undeviated)
        theta = -h_obj / (-z_obj)
        z2 = np.array([z_obj, 0, 0.4])
        y2 = np.array([h_obj, 0, 0.4 * theta])
        ax.plot(z2, y2, color=colors[1], linewidth=1.5, alpha=0.7)

        # Ray 3: Through front focal point (emerges parallel)
        if z_obj < -f:
            # Ray passes through front focal point
            theta_to_focal = (h_obj - 0) / (z_obj - (-f))
            y_at_lens = 0 + theta_to_focal * (0 - (-f))
            z3 = np.array([z_obj, 0, 0.4])
            y3 = np.array([h_obj, y_at_lens, y_at_lens])
            ax.plot(z3, y3, color=colors[2], linewidth=1.5, alpha=0.7)

        # Virtual ray extensions for virtual images
        if z_img < 0:
            ax.plot([0, z_img], [y1[-1] - 0.4/f * (0.4), h_img],
                   'r--', linewidth=1, alpha=0.5)

        ax.set_xlim(-0.4, 0.4)
        ax.set_ylim(-0.1, 0.1)
        ax.set_xlabel('z (m)')
        ax.set_ylabel('y (m)')
        ax.set_title(f'{title}\ndo={do:.2f}m, di={di:.2f}m, m={m:.2f}')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    plt.tight_layout()
    return fig


def plot_diverging_lens():
    """Plot diverging lens ray tracing"""

    fig, ax = plt.subplots(figsize=(12, 6))

    f = -0.1  # Negative focal length
    z_obj = -0.2
    h_obj = 0.04

    lens = ThinLens(focal_length=f, diameter=0.12)

    # Calculate image
    do = -z_obj
    di = lens.image_distance(do)
    m = lens.magnification(do)
    z_img = di
    h_img = m * h_obj

    # Draw optical axis
    ax.axhline(0, color='black', linewidth=0.5)

    # Draw lens
    draw_lens(ax, 0, 0.12, 'diverging')

    # Mark focal points
    ax.plot([f, -f], [0, 0], 'ko', markersize=8)
    ax.text(f, -0.015, 'F', ha='center', va='top', fontsize=10)
    ax.text(-f, -0.015, 'F\'', ha='center', va='top', fontsize=10)

    # Draw object and image
    draw_object_and_image(ax, z_obj, h_obj, z_img, h_img)

    # Trace principal rays
    tracer = LensRayTracer(f, 0.0, 0.12)

    # Ray 1: Parallel to axis
    z1, y1, _ = tracer.trace_ray(h_obj, 0, z_obj, 0.2)
    ax.plot(z1, y1, 'r-', linewidth=1.5, alpha=0.7, label='Parallel ray')

    # Virtual extension back to focal point
    y_at_lens = h_obj
    ax.plot([0, f], [y_at_lens, 0], 'r--', linewidth=1, alpha=0.5)

    # Ray 2: Through center
    theta = -h_obj / (-z_obj)
    z2 = np.array([z_obj, 0, 0.2])
    y2 = np.array([h_obj, 0, 0.2 * theta])
    ax.plot(z2, y2, 'b-', linewidth=1.5, alpha=0.7, label='Central ray')

    # Ray 3: Directed toward far focal point (emerges parallel)
    y_at_lens_3 = h_obj + (0 - z_obj) * (0 - h_obj) / (-f - z_obj)
    z3 = np.array([z_obj, 0, 0.2])
    y3 = np.array([h_obj, y_at_lens_3, y_at_lens_3])
    ax.plot(z3, y3, 'purple', linewidth=1.5, alpha=0.7, label='Focal ray')
    ax.plot([z_obj, -f], [h_obj, 0], 'purple', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlim(-0.3, 0.25)
    ax.set_ylim(-0.08, 0.08)
    ax.set_xlabel('z (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(f'Diverging Lens (f={f}m): Always Virtual, Upright, Reduced Image\n'
                f'do={do:.2f}m, di={di:.3f}m, m={m:.2f}')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    return fig


def plot_two_lens_system():
    """Plot a two-lens optical system"""

    fig, ax = plt.subplots(figsize=(14, 6))

    # Two converging lenses
    f1 = 0.08
    f2 = 0.05
    z1 = 0.0
    z2 = 0.15
    z_obj = -0.12
    h_obj = 0.03

    # Create optical system using ABCD matrices
    system = OpticalSystem()
    system.add_propagation(z1 - z_obj)
    system.add_element('thin_lens', focal_length=f1)
    system.add_propagation(z2 - z1)
    system.add_element('thin_lens', focal_length=f2)

    # Draw optical axis
    ax.axhline(0, color='black', linewidth=0.5)

    # Draw lenses
    draw_lens(ax, z1, 0.1, 'converging')
    draw_lens(ax, z2, 0.08, 'converging')

    # Mark focal points
    ax.plot([z1 - f1, z1 + f1], [0, 0], 'ko', markersize=6)
    ax.plot([z2 - f2, z2 + f2], [0, 0], 'ko', markersize=6)
    ax.text(z1 - f1, -0.01, 'F1', ha='center', va='top', fontsize=8)
    ax.text(z1 + f1, -0.01, 'F1\'', ha='center', va='top', fontsize=8)
    ax.text(z2 - f2, -0.01, 'F2', ha='center', va='top', fontsize=8)
    ax.text(z2 + f2, -0.01, 'F2\'', ha='center', va='top', fontsize=8)

    # Trace rays using ABCD formalism
    angles = [0, 0.1, -0.1]
    colors = ['red', 'blue', 'green']

    for theta, color in zip(angles, colors):
        y_trace = [h_obj]
        z_trace = [z_obj]

        # Current position and angle
        y = h_obj
        angle = theta

        # To first lens
        for z in np.linspace(z_obj, z1, 20):
            y_current = h_obj + theta * (z - z_obj)
            z_trace.append(z)
            y_trace.append(y_current)

        y_at_lens1 = h_obj + theta * (z1 - z_obj)
        angle_after_lens1 = theta - y_at_lens1 / f1

        # To second lens
        for z in np.linspace(z1, z2, 20):
            y_current = y_at_lens1 + angle_after_lens1 * (z - z1)
            z_trace.append(z)
            y_trace.append(y_current)

        y_at_lens2 = y_at_lens1 + angle_after_lens1 * (z2 - z1)
        angle_after_lens2 = angle_after_lens1 - y_at_lens2 / f2

        # After second lens
        for z in np.linspace(z2, 0.35, 20):
            y_current = y_at_lens2 + angle_after_lens2 * (z - z2)
            z_trace.append(z)
            y_trace.append(y_current)

        ax.plot(z_trace, y_trace, color=color, linewidth=1.5, alpha=0.7)

    # Draw object
    ax.annotate('', xy=(z_obj, h_obj), xytext=(z_obj, 0),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.text(z_obj, h_obj + 0.005, 'Object', ha='center', va='bottom',
            color='green', fontsize=10)

    # Calculate effective focal length
    eff_f = system.effective_focal_length()

    ax.set_xlim(-0.15, 0.35)
    ax.set_ylim(-0.06, 0.06)
    ax.set_xlabel('z (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(f'Two-Lens System: f1={f1*100:.0f}cm, f2={f2*100:.0f}cm, '
                f'Effective f={eff_f*100:.1f}cm')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    return fig


def plot_lens_equation():
    """Plot thin lens equation relationships"""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Image distance vs object distance
    ax1 = axes[0]
    f_values = [0.05, 0.1, 0.15]

    for f in f_values:
        do_range = np.linspace(f * 1.01, 0.5, 100)
        di_values = [ThinLens(f).image_distance(do) for do in do_range]

        ax1.plot(do_range * 100, np.array(di_values) * 100,
                label=f'f = {f*100:.0f} cm', linewidth=2)

    # Mark special points
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.axvline(0, color='black', linewidth=0.5)

    ax1.set_xlabel('Object distance do (cm)')
    ax1.set_ylabel('Image distance di (cm)')
    ax1.set_title('Thin Lens Equation: 1/f = 1/do + 1/di')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 50)
    ax1.set_ylim(-30, 50)

    # Right: Magnification vs object distance
    ax2 = axes[1]

    for f in f_values:
        do_range = np.linspace(f * 1.01, 0.5, 100)
        m_values = [ThinLens(f).magnification(do) for do in do_range]

        ax2.plot(do_range * 100, m_values,
                label=f'f = {f*100:.0f} cm', linewidth=2)

    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.axhline(-1, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(1, color='gray', linestyle='--', alpha=0.5)

    ax2.set_xlabel('Object distance do (cm)')
    ax2.set_ylabel('Magnification m')
    ax2.set_title('Lateral Magnification: m = -di/do')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 50)
    ax2.set_ylim(-5, 1)

    plt.tight_layout()
    return fig


def main():
    """Main function to run thin lens ray tracing demonstration"""

    # Create figures
    fig1 = plot_converging_lens_cases()
    fig2 = plot_diverging_lens()
    fig3 = plot_two_lens_system()
    fig4 = plot_lens_equation()

    # Save plots
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)

    fig1.savefig(os.path.join(output_dir, 'thin_lens_converging.png'),
                 dpi=150, bbox_inches='tight')
    fig2.savefig(os.path.join(output_dir, 'thin_lens_diverging.png'),
                 dpi=150, bbox_inches='tight')
    fig3.savefig(os.path.join(output_dir, 'thin_lens_two_lens_system.png'),
                 dpi=150, bbox_inches='tight')
    fig4.savefig(os.path.join(output_dir, 'thin_lens_equation.png'),
                 dpi=150, bbox_inches='tight')

    print(f"Plots saved to {output_dir}/thin_lens_*.png")

    # Print analysis
    print("\n=== Thin Lens Analysis ===")
    print("\nImage characteristics for f=10cm converging lens:")
    lens = ThinLens(focal_length=0.1)
    for do in [0.3, 0.2, 0.15, 0.08]:
        di = lens.image_distance(do)
        m = lens.magnification(do)
        img_type = "Real" if di > 0 else "Virtual"
        orientation = "Inverted" if m < 0 else "Upright"
        size = "Magnified" if abs(m) > 1 else "Reduced"
        print(f"  do={do*100:.0f}cm: di={di*100:.1f}cm, m={m:.2f} ({img_type}, {orientation}, {size})")


if __name__ == "__main__":
    main()
