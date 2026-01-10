"""
Example 101: Snell's Law Ray Tracing

This example demonstrates ray tracing through multiple interfaces using Snell's law,
showing how light bends when passing through media with different refractive indices.

Physics:
    Snell's law: n1 * sin(theta1) = n2 * sin(theta2)

    The demonstration includes:
    - Ray propagation through layered media
    - Visualization of refraction at each interface
    - Comparison of rays at different incident angles
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from src.sciforge.physics.optics import SnellRefraction


class MultiLayerRayTracer:
    """Ray tracer for multiple planar interfaces"""

    def __init__(self, layers: list):
        """
        Args:
            layers: List of (thickness, refractive_index) tuples
                   First layer is semi-infinite (thickness ignored for entry)
                   Last layer is semi-infinite (thickness ignored for exit)
        """
        self.layers = layers
        self.n_layers = len(layers)

        # Calculate interface positions
        self.interfaces = [0.0]
        for i, (thickness, _) in enumerate(layers[:-1]):
            self.interfaces.append(self.interfaces[-1] + thickness)

    def trace_ray(self, y_start: float, angle_start: float) -> dict:
        """
        Trace a ray through all layers.

        Args:
            y_start: Starting y position
            angle_start: Starting angle (radians, from normal)

        Returns:
            Dictionary with ray path and properties
        """
        path_x = [self.interfaces[0] - 0.5]  # Start before first interface
        path_y = [y_start - 0.5 * np.tan(angle_start)]
        angles = [angle_start]
        current_angle = angle_start
        current_y = y_start

        reflected = False
        reflection_interface = None

        for i in range(len(self.interfaces)):
            # Add point at current interface
            path_x.append(self.interfaces[i])
            path_y.append(current_y)

            if i < len(self.interfaces) - 1:
                n1 = self.layers[i][1]
                n2 = self.layers[i + 1][1]

                try:
                    snell = SnellRefraction(n1, n2)
                    new_angle = snell.refraction_angle(current_angle)
                    angles.append(new_angle)
                    current_angle = new_angle

                    # Propagate to next interface
                    dx = self.interfaces[i + 1] - self.interfaces[i]
                    dy = dx * np.tan(current_angle)
                    current_y += dy

                except Exception:
                    # Total internal reflection
                    reflected = True
                    reflection_interface = i
                    break

        # Add final segment
        if not reflected:
            path_x.append(self.interfaces[-1] + 0.5)
            path_y.append(current_y + 0.5 * np.tan(current_angle))
        else:
            # Reflect the ray
            current_angle = -current_angle
            path_x.append(path_x[-1] - 0.3)
            path_y.append(path_y[-1] - 0.3 * np.tan(current_angle))

        return {
            'path_x': np.array(path_x),
            'path_y': np.array(path_y),
            'angles': angles,
            'reflected': reflected,
            'reflection_interface': reflection_interface
        }


def simulate_snells_law():
    """Demonstrate Snell's law with various configurations"""

    # Configuration 1: Glass slab in air
    layers_slab = [
        (0.5, 1.0),    # Air
        (1.0, 1.5),    # Glass
        (0.5, 1.0),    # Air
    ]

    # Configuration 2: Multiple layers (air -> water -> glass -> diamond)
    layers_multi = [
        (0.4, 1.0),     # Air
        (0.4, 1.33),    # Water
        (0.4, 1.5),     # Glass
        (0.4, 2.42),    # Diamond
        (0.4, 1.0),     # Air
    ]

    # Configuration 3: Prism-like structure
    layers_gradient = [
        (0.3, 1.0),
        (0.3, 1.2),
        (0.3, 1.4),
        (0.3, 1.6),
        (0.3, 1.4),
        (0.3, 1.2),
        (0.3, 1.0),
    ]

    return layers_slab, layers_multi, layers_gradient


def plot_ray_tracing(ax, layers, title, incident_angles):
    """Plot ray tracing through layers"""

    tracer = MultiLayerRayTracer(layers)

    # Draw layers
    colors = plt.cm.Blues(np.linspace(0.2, 0.8, len(layers)))
    x_start = tracer.interfaces[0]

    for i, (thickness, n) in enumerate(layers):
        if i < len(layers) - 1:
            x_end = tracer.interfaces[i + 1]
        else:
            x_end = tracer.interfaces[-1] + 0.5

        rect = Rectangle((x_start, -2), x_end - x_start, 4,
                         facecolor=colors[i], alpha=0.5, edgecolor='black')
        ax.add_patch(rect)

        # Label refractive index
        ax.text((x_start + x_end) / 2, 1.8, f'n={n}',
               ha='center', va='bottom', fontsize=9)

        x_start = x_end

    # Trace rays at different angles
    ray_colors = plt.cm.rainbow(np.linspace(0, 1, len(incident_angles)))

    for angle, color in zip(incident_angles, ray_colors):
        result = tracer.trace_ray(0.0, angle)

        linestyle = '--' if result['reflected'] else '-'
        ax.plot(result['path_x'], result['path_y'],
               color=color, linewidth=2, linestyle=linestyle,
               label=f'{np.degrees(angle):.0f} deg')

        # Add arrow to show direction
        mid_idx = len(result['path_x']) // 2
        if mid_idx > 0:
            dx = result['path_x'][mid_idx] - result['path_x'][mid_idx - 1]
            dy = result['path_y'][mid_idx] - result['path_y'][mid_idx - 1]
            ax.annotate('', xy=(result['path_x'][mid_idx], result['path_y'][mid_idx]),
                       xytext=(result['path_x'][mid_idx] - dx*0.3,
                              result['path_y'][mid_idx] - dy*0.3),
                       arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

    # Draw interface lines
    for x in tracer.interfaces:
        ax.axvline(x, color='black', linestyle=':', alpha=0.5)

    ax.set_xlim(-0.3, tracer.interfaces[-1] + 0.8)
    ax.set_ylim(-2, 2)
    ax.set_xlabel('x (arbitrary units)')
    ax.set_ylabel('y (arbitrary units)')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)


def plot_refraction_angles():
    """Plot refraction angle vs incident angle for different materials"""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left plot: Refracted angle vs incident angle
    ax1 = axes[0]
    incident_angles = np.linspace(0, np.pi/2 - 0.01, 100)

    materials = [
        ('Water (n=1.33)', 1.0, 1.33),
        ('Glass (n=1.5)', 1.0, 1.5),
        ('Diamond (n=2.42)', 1.0, 2.42),
        ('Glass to Air', 1.5, 1.0),
    ]

    for label, n1, n2 in materials:
        refracted = []
        critical_angle = None

        for theta in incident_angles:
            try:
                snell = SnellRefraction(n1, n2)
                refracted.append(snell.refraction_angle(theta))
            except:
                if critical_angle is None:
                    critical_angle = theta
                refracted.append(np.nan)

        ax1.plot(np.degrees(incident_angles), np.degrees(refracted),
                label=label, linewidth=2)

        if critical_angle is not None:
            ax1.axvline(np.degrees(critical_angle), linestyle='--', alpha=0.5)

    ax1.plot([0, 90], [0, 90], 'k--', alpha=0.3, label='No refraction')
    ax1.set_xlabel('Incident angle (degrees)')
    ax1.set_ylabel('Refracted angle (degrees)')
    ax1.set_title('Snell\'s Law: Refraction Angles')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 90)
    ax1.set_ylim(0, 90)

    # Right plot: Fresnel reflectance
    ax2 = axes[1]

    for label, n1, n2 in materials[:3]:
        snell = SnellRefraction(n1, n2)
        R_s = []
        R_p = []

        for theta in incident_angles:
            R_s.append(snell.reflectance(theta, 's'))
            R_p.append(snell.reflectance(theta, 'p'))

        ax2.plot(np.degrees(incident_angles), R_s,
                label=f'{label} (s-pol)', linewidth=2, linestyle='-')
        ax2.plot(np.degrees(incident_angles), R_p,
                label=f'{label} (p-pol)', linewidth=2, linestyle='--')

    ax2.set_xlabel('Incident angle (degrees)')
    ax2.set_ylabel('Reflectance')
    ax2.set_title('Fresnel Reflectance vs Angle')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 90)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    return fig


def main():
    """Main function to run Snell's law demonstration"""

    # Get layer configurations
    layers_slab, layers_multi, layers_gradient = simulate_snells_law()

    # Create figure for ray tracing
    fig1, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Incident angles to trace
    angles = [np.radians(a) for a in [0, 15, 30, 45, 60]]

    plot_ray_tracing(axes[0], layers_slab, 'Glass Slab in Air', angles)
    plot_ray_tracing(axes[1], layers_multi, 'Multiple Layers\n(Air-Water-Glass-Diamond-Air)', angles[:4])
    plot_ray_tracing(axes[2], layers_gradient, 'Graded Index Structure', angles)

    plt.tight_layout()

    # Create figure for refraction analysis
    fig2 = plot_refraction_angles()

    # Save plots
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)

    fig1.savefig(os.path.join(output_dir, 'snells_law_ray_tracing.png'),
                 dpi=150, bbox_inches='tight')
    fig2.savefig(os.path.join(output_dir, 'snells_law_analysis.png'),
                 dpi=150, bbox_inches='tight')

    print(f"Plots saved to {output_dir}/snells_law_*.png")

    # Print some analysis
    print("\n=== Snell's Law Analysis ===")
    print("\nCritical angles for total internal reflection:")
    for n1, label in [(1.5, 'Glass to Air'), (1.33, 'Water to Air'), (2.42, 'Diamond to Air')]:
        snell = SnellRefraction(n1, 1.0)
        critical = snell.critical_angle()
        print(f"  {label}: {np.degrees(critical):.1f} degrees")

    print("\nBrewster angles (no p-polarization reflection):")
    for n2, label in [(1.33, 'Air to Water'), (1.5, 'Air to Glass'), (2.42, 'Air to Diamond')]:
        snell = SnellRefraction(1.0, n2)
        brewster = snell.brewster_angle()
        print(f"  {label}: {np.degrees(brewster):.1f} degrees")


if __name__ == "__main__":
    main()
