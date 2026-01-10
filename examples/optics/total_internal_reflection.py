"""
Example 104: Total Internal Reflection

This example demonstrates total internal reflection (TIR), which occurs when
light travels from a higher to lower refractive index medium at angles
beyond the critical angle.

Physics:
    Critical angle: theta_c = arcsin(n2/n1), where n1 > n2

    At angles > theta_c, 100% of light is reflected
    At exactly theta_c, refracted ray grazes the interface

Applications:
    - Optical fibers
    - Prisms (Porro, Dove, pentaprism)
    - Diamond brilliance
    - Fingerprint sensors
    - Frustrated TIR for touch screens
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Circle, FancyArrowPatch
from matplotlib.lines import Line2D
from src.sciforge.physics.optics import SnellRefraction


class TIRDemonstrator:
    """Demonstrates total internal reflection at an interface"""

    def __init__(self, n1: float, n2: float):
        """
        Args:
            n1: Refractive index of denser medium (incident)
            n2: Refractive index of less dense medium (transmitted)
        """
        self.n1 = n1
        self.n2 = n2
        self.snell = SnellRefraction(n1, n2)
        self.critical_angle = self.snell.critical_angle()

    def trace_ray(self, incident_angle: float) -> dict:
        """
        Trace ray at interface, handling TIR.

        Args:
            incident_angle: Angle from normal (radians)

        Returns:
            Dictionary with ray paths and properties
        """
        result = {
            'incident_angle': incident_angle,
            'reflected': True,  # Always have reflected ray
            'transmitted': False,
            'tir': False
        }

        # Calculate reflectance
        R = self.snell.reflectance(incident_angle)
        result['reflectance'] = R

        # Reflected ray angle = incident angle (opposite side of normal)
        result['reflected_angle'] = incident_angle

        # Check for TIR
        sin_theta_t = self.n1 * np.sin(incident_angle) / self.n2

        if abs(sin_theta_t) > 1:
            # Total internal reflection
            result['tir'] = True
            result['transmitted'] = False
            result['transmittance'] = 0.0
        else:
            # Partial transmission
            result['transmitted'] = True
            result['transmitted_angle'] = np.arcsin(sin_theta_t)
            result['transmittance'] = 1 - R

        return result


class OpticalFiber:
    """Simple step-index optical fiber model"""

    def __init__(self, core_n: float, cladding_n: float, core_radius: float,
                 length: float):
        """
        Args:
            core_n: Core refractive index
            cladding_n: Cladding refractive index
            core_radius: Core radius
            length: Fiber length
        """
        self.n_core = core_n
        self.n_clad = cladding_n
        self.radius = core_radius
        self.length = length

        # Numerical aperture
        self.NA = np.sqrt(core_n**2 - cladding_n**2)

        # Critical angle at core-cladding interface
        self.critical_angle = np.arcsin(cladding_n / core_n)

        # Maximum acceptance angle
        self.acceptance_angle = np.arcsin(self.NA)

    def trace_ray(self, y_start: float, angle: float, n_bounces: int = 20) -> dict:
        """
        Trace ray through fiber with multiple TIR bounces.

        Args:
            y_start: Starting y position (within core)
            angle: Starting angle from axis
            n_bounces: Maximum bounces to trace

        Returns:
            Dictionary with ray path
        """
        if abs(y_start) > self.radius:
            return {'valid': False, 'reason': 'Start outside core'}

        # Check if ray will be guided
        if abs(angle) > np.pi/2 - self.critical_angle:
            return {'valid': False, 'reason': 'Exceeds critical angle'}

        path_z = [0.0]
        path_y = [y_start]

        y = y_start
        theta = angle  # Angle from z-axis
        z = 0.0

        for _ in range(n_bounces):
            # Propagate until hitting boundary
            if np.tan(theta) == 0:
                # Ray parallel to axis
                z = self.length
                path_z.append(z)
                path_y.append(y)
                break

            # Distance to boundary
            if np.tan(theta) > 0:
                dz = (self.radius - y) / np.tan(theta)
            else:
                dz = (-self.radius - y) / np.tan(theta)

            if z + dz > self.length:
                # Ray exits fiber
                z = self.length
                y = y + (z - path_z[-1]) * np.tan(theta)
                path_z.append(z)
                path_y.append(y)
                break

            z += dz
            y = self.radius if np.tan(theta) > 0 else -self.radius

            path_z.append(z)
            path_y.append(y)

            # Reflect (TIR)
            theta = -theta

        return {
            'valid': True,
            'path_z': np.array(path_z),
            'path_y': np.array(path_y)
        }


class PorroPrism:
    """Porro prism using TIR for 180-degree beam deflection"""

    def __init__(self, size: float, n: float = 1.52):
        """
        Args:
            size: Prism size (side of the 45-45-90 triangle)
            n: Refractive index
        """
        self.size = size
        self.n = n
        self.critical_angle = np.arcsin(1.0 / n)

    def trace_ray(self, y_in: float) -> dict:
        """
        Trace ray through Porro prism.

        Input ray: horizontal, entering hypotenuse
        Output ray: horizontal, exiting hypotenuse, displaced

        Args:
            y_in: Entry height (0 = center of hypotenuse)

        Returns:
            Ray path dictionary
        """
        # Prism geometry: 45-45-90 triangle
        # Entry on hypotenuse, TIR on two short sides

        path_x = []
        path_y = []

        # Entry point on hypotenuse
        x0 = -self.size / 2
        y0 = y_in
        path_x.append(x0)
        path_y.append(y0)

        # First TIR surface (right side, at 45 degrees)
        # Ray hits at 45 degrees, reflects
        x1 = 0
        y1 = y0 - self.size / 2
        path_x.append(x1)
        path_y.append(y1)

        # Second TIR surface (bottom)
        x2 = x1 + (y1 - (-self.size))
        y2 = -self.size + self.size/2
        path_x.append(x2)
        path_y.append(y2)

        # Exit through hypotenuse
        x3 = self.size / 2
        y3 = -y_in
        path_x.append(x3)
        path_y.append(y3)

        return {
            'path_x': np.array(path_x),
            'path_y': np.array(path_y),
            'inversion': True
        }


def plot_tir_at_interface():
    """Plot TIR phenomenon at a single interface"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Glass to air interface
    n1, n2 = 1.5, 1.0
    tir = TIRDemonstrator(n1, n2)

    # Plot 1: Ray diagram showing TIR transition
    ax1 = axes[0, 0]

    # Draw interface
    ax1.axhline(0, color='black', linewidth=2)
    ax1.fill_between([-2, 2], [-2, -2], [0, 0], color='lightblue', alpha=0.3)
    ax1.text(-1.8, -0.5, f'Glass (n={n1})', fontsize=12)
    ax1.text(-1.8, 0.5, f'Air (n={n2})', fontsize=12)

    angles = np.radians([20, 30, 41.8, 50, 60])  # 41.8 is critical angle
    colors = ['green', 'blue', 'orange', 'red', 'darkred']
    labels = ['20 deg', '30 deg', 'Critical', '50 deg (TIR)', '60 deg (TIR)']

    for angle, color, label in zip(angles, colors, labels):
        result = tir.trace_ray(angle)

        # Incident ray
        x_inc = [-1.5 * np.sin(angle), 0]
        y_inc = [-1.5 * np.cos(angle), 0]
        ax1.plot(x_inc, y_inc, color=color, linewidth=2)

        # Add arrow
        ax1.annotate('', xy=(0, 0),
                    xytext=(-0.3 * np.sin(angle), -0.3 * np.cos(angle)),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

        # Reflected ray
        x_ref = [0, 1.5 * np.sin(angle)]
        y_ref = [0, -1.5 * np.cos(angle)]
        linestyle = '-' if result['tir'] else '--'
        ax1.plot(x_ref, y_ref, color=color, linewidth=2, linestyle=linestyle,
                alpha=0.7 if not result['tir'] else 1.0)

        # Transmitted ray (if exists)
        if result['transmitted']:
            trans_angle = result['transmitted_angle']
            x_trans = [0, 1.5 * np.sin(trans_angle)]
            y_trans = [0, 1.5 * np.cos(trans_angle)]
            ax1.plot(x_trans, y_trans, color=color, linewidth=2, alpha=0.7)

    # Draw normal
    ax1.plot([0, 0], [-1.5, 1.5], 'k--', linewidth=1, alpha=0.5)

    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(f'Total Internal Reflection at Glass-Air Interface\n'
                 f'Critical angle = {np.degrees(tir.critical_angle):.1f} deg')
    ax1.set_aspect('equal')

    # Custom legend
    legend_elements = [Line2D([0], [0], color=c, linewidth=2, label=l)
                      for c, l in zip(colors, labels)]
    ax1.legend(handles=legend_elements, loc='upper right')

    # Plot 2: Reflectance vs angle
    ax2 = axes[0, 1]

    angles_plot = np.linspace(0, np.pi/2 - 0.01, 200)
    R_s = []
    R_p = []
    R_avg = []

    for angle in angles_plot:
        R_s.append(tir.snell.reflectance(angle, 's'))
        R_p.append(tir.snell.reflectance(angle, 'p'))
        R_avg.append(tir.snell.reflectance(angle, 'unpolarized'))

    ax2.plot(np.degrees(angles_plot), R_s, 'b-', linewidth=2, label='s-polarization')
    ax2.plot(np.degrees(angles_plot), R_p, 'r-', linewidth=2, label='p-polarization')
    ax2.plot(np.degrees(angles_plot), R_avg, 'k--', linewidth=2, label='Unpolarized')

    ax2.axvline(np.degrees(tir.critical_angle), color='orange', linestyle='--',
               linewidth=2, label=f'Critical angle')
    ax2.axvline(np.degrees(tir.snell.brewster_angle()), color='green', linestyle=':',
               linewidth=2, label=f'Brewster angle')

    ax2.set_xlabel('Incident angle (degrees)')
    ax2.set_ylabel('Reflectance')
    ax2.set_title('Fresnel Reflectance vs Incident Angle')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 90)
    ax2.set_ylim(0, 1.1)

    # Plot 3: Critical angles for different materials
    ax3 = axes[1, 0]

    materials = [
        ('Water', 1.33),
        ('Glass', 1.50),
        ('Sapphire', 1.77),
        ('Diamond', 2.42),
        ('Silicon', 3.42),
    ]

    x_pos = np.arange(len(materials))
    critical_angles = [np.degrees(np.arcsin(1.0 / n)) for _, n in materials]
    brewster_angles = [np.degrees(np.arctan(1.0 / n)) for _, n in materials]

    width = 0.35
    bars1 = ax3.bar(x_pos - width/2, critical_angles, width,
                   label='Critical angle', color='coral')
    bars2 = ax3.bar(x_pos + width/2, brewster_angles, width,
                   label='Brewster angle', color='skyblue')

    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([m[0] for m in materials])
    ax3.set_ylabel('Angle (degrees)')
    ax3.set_title('Critical and Brewster Angles for Various Materials\n(Interface with air)')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Add n values on bars
    for bar, (name, n) in zip(bars1, materials):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'n={n}', ha='center', va='bottom', fontsize=9)

    # Plot 4: Phase shift upon TIR
    ax4 = axes[1, 1]

    angles_tir = np.linspace(tir.critical_angle + 0.01, np.pi/2 - 0.01, 100)

    # Phase shifts for s and p polarization during TIR
    phase_s = []
    phase_p = []

    for theta in angles_tir:
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        n_ratio = n1 / n2

        # Evanescent wave parameter
        gamma = np.sqrt(sin_theta**2 - (n2/n1)**2)

        # Phase shifts (Fresnel equations for TIR)
        delta_s = 2 * np.arctan(gamma / cos_theta)
        delta_p = 2 * np.arctan(n_ratio**2 * gamma / cos_theta)

        phase_s.append(np.degrees(delta_s))
        phase_p.append(np.degrees(delta_p))

    ax4.plot(np.degrees(angles_tir), phase_s, 'b-', linewidth=2, label='s-polarization')
    ax4.plot(np.degrees(angles_tir), phase_p, 'r-', linewidth=2, label='p-polarization')
    ax4.plot(np.degrees(angles_tir), np.array(phase_p) - np.array(phase_s),
            'g--', linewidth=2, label='Phase difference (p-s)')

    ax4.axhline(45, color='gray', linestyle=':', alpha=0.5)
    ax4.axhline(90, color='gray', linestyle=':', alpha=0.5)

    ax4.set_xlabel('Incident angle (degrees)')
    ax4.set_ylabel('Phase shift (degrees)')
    ax4.set_title('Phase Shift Upon Total Internal Reflection')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(np.degrees(tir.critical_angle), 90)

    plt.tight_layout()
    return fig


def plot_optical_fiber():
    """Plot light propagation in optical fiber via TIR"""

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Create optical fiber
    fiber = OpticalFiber(
        core_n=1.48,
        cladding_n=1.46,
        core_radius=0.025,  # 25 micron core (scaled for visualization)
        length=1.0
    )

    ax1 = axes[0]

    # Draw fiber structure
    ax1.fill_between([0, fiber.length], [-fiber.radius*1.5, -fiber.radius*1.5],
                    [fiber.radius*1.5, fiber.radius*1.5],
                    color='lightblue', alpha=0.3, label='Cladding')
    ax1.fill_between([0, fiber.length], [-fiber.radius, -fiber.radius],
                    [fiber.radius, fiber.radius],
                    color='lightcoral', alpha=0.5, label='Core')

    # Trace rays at different angles
    entry_angles = np.radians([0, 3, 6, 8, 10])
    entry_positions = [0, 0.01, -0.01, 0.015, -0.015]
    colors = plt.cm.viridis(np.linspace(0, 1, len(entry_angles)))

    for angle, y0, color in zip(entry_angles, entry_positions, colors):
        result = fiber.trace_ray(y0, angle)
        if result['valid']:
            ax1.plot(result['path_z'], result['path_y'], color=color, linewidth=1.5)

    ax1.axhline(fiber.radius, color='black', linewidth=2)
    ax1.axhline(-fiber.radius, color='black', linewidth=2)

    ax1.set_xlabel('z (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_title(f'Step-Index Optical Fiber\n'
                 f'Core n={fiber.n_core}, Cladding n={fiber.n_clad}, '
                 f'NA={fiber.NA:.3f}, Acceptance angle={np.degrees(fiber.acceptance_angle):.1f} deg')
    ax1.legend(loc='upper right')
    ax1.set_xlim(-0.05, fiber.length + 0.05)
    ax1.set_ylim(-0.04, 0.04)
    ax1.grid(True, alpha=0.3)

    # Plot acceptance cone
    ax2 = axes[1]

    # Draw fiber face
    theta = np.linspace(0, 2*np.pi, 100)
    ax2.plot(fiber.radius * np.cos(theta), fiber.radius * np.sin(theta), 'r-', linewidth=2)
    ax2.fill(fiber.radius * np.cos(theta), fiber.radius * np.sin(theta),
            color='lightcoral', alpha=0.5, label='Fiber core')

    # Draw acceptance cone
    cone_length = 0.1
    for angle in np.linspace(-fiber.acceptance_angle, fiber.acceptance_angle, 10):
        x = [-cone_length * np.cos(angle), 0]
        y = [cone_length * np.sin(angle), 0]
        ax2.plot(x, y, 'g-', linewidth=1, alpha=0.5)

    # Draw cone boundary
    ax2.plot([-cone_length * np.cos(fiber.acceptance_angle), 0],
            [cone_length * np.sin(fiber.acceptance_angle), 0], 'g-', linewidth=2)
    ax2.plot([-cone_length * np.cos(fiber.acceptance_angle), 0],
            [-cone_length * np.sin(fiber.acceptance_angle), 0], 'g-', linewidth=2)

    # Show rejected ray
    ax2.plot([-cone_length * np.cos(1.5*fiber.acceptance_angle), 0],
            [cone_length * np.sin(1.5*fiber.acceptance_angle), 0],
            'r--', linewidth=2, label='Rejected ray')

    ax2.annotate(f'NA = {fiber.NA:.3f}', xy=(-0.05, 0.01), fontsize=12)
    ax2.annotate(f'Acceptance angle\n= {np.degrees(fiber.acceptance_angle):.1f} deg',
                xy=(-0.08, 0.025), fontsize=10)

    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')
    ax2.set_title('Fiber Acceptance Cone (Numerical Aperture)')
    ax2.set_xlim(-0.12, 0.04)
    ax2.set_ylim(-0.06, 0.06)
    ax2.set_aspect('equal')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_prism_tir():
    """Plot TIR in prisms"""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    n_glass = 1.52
    critical = np.degrees(np.arcsin(1.0 / n_glass))

    # Porro prism (45-45-90)
    ax1 = axes[0]

    # Draw prism
    vertices = np.array([[0, 0], [1, 0], [1, -1], [0, 0]])
    prism = Polygon(vertices, closed=True, fill=True, facecolor='lightblue',
                   edgecolor='blue', linewidth=2, alpha=0.5)
    ax1.add_patch(prism)

    # Trace rays
    y_positions = [0.2, 0.4, 0.6, 0.8]
    colors = plt.cm.Reds(np.linspace(0.3, 0.8, len(y_positions)))

    for y, color in zip(y_positions, colors):
        # Entry on hypotenuse
        x0 = -0.3
        y0 = -y
        # At hypotenuse
        x1 = y
        y1 = -y
        # First TIR (right wall)
        x2 = 1
        y2 = -y - (1 - y) / 1
        # Second TIR (bottom)
        x3 = 1 - abs(y2)
        y3 = -1
        # Exit
        x4 = 1.3
        y4 = -1 + (y - 0.5)

        ax1.plot([x0, x1, x2], [y0, y1, y2], color=color, linewidth=2)
        ax1.plot([x2, x3, x4], [y2, y3, y4], color=color, linewidth=2, linestyle='--')

    ax1.set_xlim(-0.5, 1.5)
    ax1.set_ylim(-1.5, 0.5)
    ax1.set_title(f'Porro Prism (45-45-90)\n180 deg beam deflection via 2 TIRs')
    ax1.set_aspect('equal')
    ax1.axis('off')

    # Right-angle prism
    ax2 = axes[1]

    vertices2 = np.array([[0, 0], [1, 0], [0, -1], [0, 0]])
    prism2 = Polygon(vertices2, closed=True, fill=True, facecolor='lightyellow',
                    edgecolor='orange', linewidth=2, alpha=0.5)
    ax2.add_patch(prism2)

    # Trace rays - 90 degree deflection
    for i, (x, color) in enumerate(zip([0.2, 0.4, 0.6], colors)):
        # Entry on top surface
        y0 = 0.3
        x0 = x
        # At hypotenuse - TIR
        x1 = x
        y1 = -(1 - x)
        # Exit through left surface
        x2 = -0.3
        y2 = y1

        ax2.plot([x0, x0], [y0, 0], color=color, linewidth=2)
        ax2.plot([x0, x1], [0, y1], color=color, linewidth=2)
        ax2.plot([x1, 0, x2], [y1, y1, y1], color=color, linewidth=2)

    ax2.set_xlim(-0.5, 1.5)
    ax2.set_ylim(-1.5, 0.5)
    ax2.set_title('Right-Angle Prism\n90 deg beam deflection via TIR')
    ax2.set_aspect('equal')
    ax2.axis('off')

    # Dove prism (truncated right-angle prism)
    ax3 = axes[2]

    vertices3 = np.array([[0, 0], [2, 0], [1.5, -0.5], [0.5, -0.5], [0, 0]])
    prism3 = Polygon(vertices3, closed=True, fill=True, facecolor='lightgreen',
                    edgecolor='green', linewidth=2, alpha=0.5)
    ax3.add_patch(prism3)

    # Trace rays - image inversion
    for y, color in zip([0.1, 0.2, 0.3], colors):
        # Entry on left face
        x0 = -0.3
        y0 = -0.1
        # Inside prism, hits bottom at TIR
        x1 = 0.2
        y1 = -0.15
        # At bottom surface
        x2 = 1.0
        y2 = -0.5 + 0.1
        # Exit
        x3 = 1.8
        y3 = -0.15
        x4 = 2.3
        y4 = -0.1

        ax3.plot([x0, x1, x2, x3, x4], [y0, y1, y2, y3, y4],
                color=color, linewidth=2)

    ax3.set_xlim(-0.5, 2.5)
    ax3.set_ylim(-1, 0.5)
    ax3.set_title('Dove Prism\nImage inversion via single TIR')
    ax3.set_aspect('equal')
    ax3.axis('off')

    plt.suptitle(f'Prisms Using Total Internal Reflection (Critical angle in glass = {critical:.1f} deg)',
                fontsize=12, y=0.02)
    plt.tight_layout()
    return fig


def main():
    """Main function to demonstrate total internal reflection"""

    # Create figures
    fig1 = plot_tir_at_interface()
    fig2 = plot_optical_fiber()
    fig3 = plot_prism_tir()

    # Save plots
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)

    fig1.savefig(os.path.join(output_dir, 'total_internal_reflection.png'),
                 dpi=150, bbox_inches='tight')
    fig2.savefig(os.path.join(output_dir, 'tir_optical_fiber.png'),
                 dpi=150, bbox_inches='tight')
    fig3.savefig(os.path.join(output_dir, 'tir_prisms.png'),
                 dpi=150, bbox_inches='tight')

    print(f"Plots saved to {output_dir}/total_internal_reflection*.png")

    # Print analysis
    print("\n=== Total Internal Reflection Analysis ===")
    print("\nCritical angles (interface with air):")
    materials = [('Water', 1.33), ('Glass', 1.50), ('Diamond', 2.42)]
    for name, n in materials:
        theta_c = np.degrees(np.arcsin(1.0 / n))
        print(f"  {name} (n={n}): {theta_c:.1f} degrees")

    print("\nApplications of TIR:")
    print("  - Optical fibers: Light guided by repeated TIR at core-cladding interface")
    print("  - Prisms: Porro (binoculars), pentaprism (cameras), Dove (image rotation)")
    print("  - Diamond brilliance: Low critical angle (24.4 deg) traps light")
    print("  - FTIR sensors: Frustrated TIR detects objects near surface")


if __name__ == "__main__":
    main()
