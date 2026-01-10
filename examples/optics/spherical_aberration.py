"""
Example 103: Spherical Aberration

This example demonstrates spherical aberration in lenses and mirrors,
showing how rays at different heights from the optical axis focus at
different points, causing image blur.

Physics:
    For a spherical surface, the focal length depends on the ray height:

    f(h) = f_paraxial * (1 - h^2 / (2*R^2) * ...)

    Longitudinal spherical aberration: Delta_z = z_marginal - z_paraxial
    Transverse spherical aberration: Delta_y = ray height at paraxial focus

    The third-order (Seidel) aberration gives:
    Delta_z ~ h^2 for marginal rays
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc, Wedge
from src.sciforge.physics.optics import ThickLens


class SphericalLens:
    """
    Lens with exact (non-paraxial) ray tracing to demonstrate spherical aberration.
    """

    def __init__(self, R1: float, R2: float, thickness: float, n: float,
                 diameter: float = 0.1):
        """
        Args:
            R1: First surface radius of curvature (positive = center to right)
            R2: Second surface radius of curvature (positive = center to right)
            thickness: Center thickness
            n: Refractive index
            diameter: Lens diameter
        """
        self.R1 = R1
        self.R2 = R2
        self.d = thickness
        self.n = n
        self.diameter = diameter

        # Calculate paraxial focal length using lensmaker's equation
        P1 = (n - 1) / R1 if R1 != 0 else 0
        P2 = (1 - n) / R2 if R2 != 0 else 0
        self.power = P1 + P2 - (thickness / n) * P1 * P2
        self.f_paraxial = 1.0 / self.power if self.power != 0 else np.inf

    def surface_sag(self, r: float, R: float) -> float:
        """Calculate surface sag (z-displacement) at radial height r"""
        if abs(R) == np.inf or R == 0:
            return 0.0
        if abs(r) > abs(R):
            return np.nan

        # z = R - sqrt(R^2 - r^2) for convex (R > 0)
        # z = -R + sqrt(R^2 - r^2) for concave (R < 0)
        return R - np.sign(R) * np.sqrt(R**2 - r**2)

    def refract_at_surface(self, y: float, angle: float, R: float,
                           n1: float, n2: float) -> tuple:
        """
        Exact refraction at spherical surface.

        Args:
            y: Ray height at surface
            angle: Ray angle (from z-axis)
            R: Surface radius of curvature
            n1: Incident medium refractive index
            n2: Exit medium refractive index

        Returns:
            (new_angle, hit_point_z, hit_point_y)
        """
        if abs(R) == np.inf:
            # Flat surface
            sin_theta2 = n1 * np.sin(angle) / n2
            if abs(sin_theta2) > 1:
                return None, None, None  # TIR
            return np.arcsin(sin_theta2), 0, y

        # Surface normal angle at height y
        if abs(y) > abs(R):
            return None, None, None  # Ray misses surface

        # Angle of surface normal from z-axis
        alpha = np.arcsin(y / R)

        # Incident angle relative to surface normal
        theta_i = angle - alpha

        # Snell's law
        sin_theta_t = n1 * np.sin(theta_i) / n2
        if abs(sin_theta_t) > 1:
            return None, None, None  # TIR

        theta_t = np.arcsin(sin_theta_t)

        # New ray angle relative to z-axis
        new_angle = theta_t + alpha

        # Hit point
        z_hit = self.surface_sag(y, R)

        return new_angle, z_hit, y

    def trace_ray(self, y_in: float, angle_in: float = 0.0) -> dict:
        """
        Trace a single ray through the lens.

        Args:
            y_in: Initial ray height (at z = -large)
            angle_in: Initial ray angle

        Returns:
            Dictionary with ray path and focal point
        """
        # Starting point (before lens)
        z_start = -0.2
        y_start = y_in + angle_in * z_start

        path_z = [z_start]
        path_y = [y_start]

        # First surface is at z = 0
        # Ray height at first surface
        y_at_s1 = y_in

        # First refraction
        result = self.refract_at_surface(y_at_s1, angle_in, self.R1, 1.0, self.n)
        if result[0] is None:
            return {'path_z': path_z, 'path_y': path_y, 'valid': False}

        angle_after_s1, z1, y1 = result
        path_z.append(z1)
        path_y.append(y1)

        # Propagate to second surface
        # Approximate: second surface at z = d
        z_s2 = self.d
        y_at_s2 = y1 + angle_after_s1 * (z_s2 - z1)

        if abs(y_at_s2) > self.diameter / 2:
            return {'path_z': path_z, 'path_y': path_y, 'valid': False}

        path_z.append(z_s2)
        path_y.append(y_at_s2)

        # Second refraction
        result = self.refract_at_surface(y_at_s2, angle_after_s1, self.R2, self.n, 1.0)
        if result[0] is None:
            return {'path_z': path_z, 'path_y': path_y, 'valid': False}

        angle_after_s2, z2, y2 = result

        # Propagate to focal region
        z_end = self.f_paraxial * 1.5
        y_end = y2 + angle_after_s2 * (z_end - z_s2)

        path_z.append(z_s2)
        path_y.append(y2)
        path_z.append(z_end)
        path_y.append(y_end)

        # Find z-intercept (where ray crosses axis)
        if abs(y2) > 1e-10 and abs(angle_after_s2) > 1e-10:
            z_focus = z_s2 - y2 / np.tan(angle_after_s2)
        else:
            z_focus = z_end

        return {
            'path_z': np.array(path_z),
            'path_y': np.array(path_y),
            'valid': True,
            'z_focus': z_focus,
            'y_at_focus': y2 + angle_after_s2 * (self.f_paraxial - z_s2)
        }


class SphericalMirror:
    """Spherical mirror with exact ray tracing for aberration analysis"""

    def __init__(self, R: float, diameter: float = 0.1):
        """
        Args:
            R: Radius of curvature (positive for concave)
            diameter: Mirror diameter
        """
        self.R = R
        self.diameter = diameter
        self.f_paraxial = R / 2

    def trace_ray(self, y_in: float, angle_in: float = 0.0) -> dict:
        """Trace ray reflecting from spherical mirror"""

        path_z = [-0.3]
        path_y = [y_in - 0.3 * np.tan(angle_in)]

        # Find intersection with spherical surface
        # Surface at z = R - sqrt(R^2 - y^2) for concave (R > 0)
        y_hit = y_in
        if abs(y_hit) > self.diameter / 2:
            return {'path_z': path_z, 'path_y': path_y, 'valid': False}

        z_hit = self.R - np.sqrt(self.R**2 - y_hit**2)

        path_z.append(z_hit)
        path_y.append(y_hit)

        # Surface normal angle
        alpha = np.arcsin(y_hit / self.R)

        # Incident angle relative to normal
        theta_i = angle_in - alpha

        # Reflection: theta_r = -theta_i (relative to normal)
        angle_out = -theta_i + alpha

        # Propagate reflected ray
        z_end = -0.3
        y_end = y_hit + np.tan(angle_out) * (z_end - z_hit)

        path_z.append(z_end)
        path_y.append(y_end)

        # Find focal point (axis crossing)
        if abs(y_hit) > 1e-10:
            z_focus = z_hit - y_hit / np.tan(angle_out)
        else:
            z_focus = self.f_paraxial

        return {
            'path_z': np.array(path_z),
            'path_y': np.array(path_y),
            'valid': True,
            'z_focus': z_focus
        }


def draw_lens_profile(ax, lens, z_offset=0):
    """Draw lens cross-section"""
    y = np.linspace(-lens.diameter/2, lens.diameter/2, 100)

    # First surface
    z1 = np.array([lens.surface_sag(yi, lens.R1) for yi in y])

    # Second surface
    z2 = np.array([lens.d + lens.surface_sag(yi, -lens.R2) for yi in y])

    # Draw lens outline
    ax.fill_betweenx(y, z1 + z_offset, z2 + z_offset, alpha=0.3, color='blue')
    ax.plot(z1 + z_offset, y, 'b-', linewidth=2)
    ax.plot(z2 + z_offset, y, 'b-', linewidth=2)


def draw_mirror_profile(ax, mirror, z_offset=0):
    """Draw spherical mirror cross-section"""
    y = np.linspace(-mirror.diameter/2, mirror.diameter/2, 100)
    z = np.array([mirror.R - np.sqrt(mirror.R**2 - yi**2) for yi in y])

    ax.plot(z + z_offset, y, 'gray', linewidth=4)
    ax.fill_betweenx(y, z + z_offset, z + z_offset + 0.005, color='gray', alpha=0.5)


def plot_lens_spherical_aberration():
    """Plot spherical aberration in a plano-convex lens"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plano-convex lens (flat side facing object - worst orientation)
    R1 = np.inf  # Flat
    R2 = -0.1    # Convex (center to left)
    lens = SphericalLens(R1, R2, 0.01, n=1.5, diameter=0.08)

    # Trace rays at different heights
    heights = np.linspace(0.001, 0.04, 20)
    colors = plt.cm.viridis(np.linspace(0, 1, len(heights)))

    ax1 = axes[0, 0]
    z_foci = []
    y_transverse = []

    for h, color in zip(heights, colors):
        result = lens.trace_ray(h, 0.0)
        if result['valid']:
            ax1.plot(result['path_z'], result['path_y'], color=color, linewidth=1)
            ax1.plot(result['path_z'], -result['path_y'], color=color, linewidth=1)
            z_foci.append(result['z_focus'])
            y_transverse.append(result['y_at_focus'])

    draw_lens_profile(ax1, lens)

    # Mark paraxial and marginal foci
    ax1.axvline(lens.f_paraxial, color='green', linestyle='--', label='Paraxial focus')
    if z_foci:
        ax1.axvline(z_foci[-1], color='red', linestyle='--', label='Marginal focus')

    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.set_xlabel('z (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_title(f'Plano-Convex Lens (wrong orientation)\nf_paraxial = {lens.f_paraxial*1000:.1f} mm')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.05, 0.2)
    ax1.set_ylim(-0.05, 0.05)

    # Better orientation: convex side facing object
    R1 = 0.1     # Convex (center to right)
    R2 = np.inf  # Flat
    lens2 = SphericalLens(R1, R2, 0.01, n=1.5, diameter=0.08)

    ax2 = axes[0, 1]
    z_foci2 = []

    for h, color in zip(heights, colors):
        result = lens2.trace_ray(h, 0.0)
        if result['valid']:
            ax2.plot(result['path_z'], result['path_y'], color=color, linewidth=1)
            ax2.plot(result['path_z'], -result['path_y'], color=color, linewidth=1)
            z_foci2.append(result['z_focus'])

    draw_lens_profile(ax2, lens2)
    ax2.axvline(lens2.f_paraxial, color='green', linestyle='--', label='Paraxial focus')
    if z_foci2:
        ax2.axvline(z_foci2[-1], color='red', linestyle='--', label='Marginal focus')

    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_xlabel('z (m)')
    ax2.set_ylabel('y (m)')
    ax2.set_title(f'Plano-Convex Lens (correct orientation)\nf_paraxial = {lens2.f_paraxial*1000:.1f} mm')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.05, 0.2)
    ax2.set_ylim(-0.05, 0.05)

    # Longitudinal spherical aberration plot
    ax3 = axes[1, 0]

    if z_foci:
        long_aberr = np.array(z_foci) - lens.f_paraxial
        ax3.plot(heights * 1000, long_aberr * 1000, 'b-', linewidth=2,
                label='Wrong orientation')
    if z_foci2:
        long_aberr2 = np.array(z_foci2) - lens2.f_paraxial
        ax3.plot(heights[:len(z_foci2)] * 1000, long_aberr2 * 1000, 'r-', linewidth=2,
                label='Correct orientation')

    ax3.set_xlabel('Ray height h (mm)')
    ax3.set_ylabel('Longitudinal aberration (mm)')
    ax3.set_title('Longitudinal Spherical Aberration')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(0, color='black', linewidth=0.5)

    # Transverse aberration
    ax4 = axes[1, 1]

    if y_transverse:
        ax4.plot(heights * 1000, np.array(y_transverse) * 1000, 'b-', linewidth=2)

    ax4.set_xlabel('Ray height h (mm)')
    ax4.set_ylabel('Transverse aberration (mm)')
    ax4.set_title('Transverse Spherical Aberration\n(at paraxial focal plane)')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(0, color='black', linewidth=0.5)

    plt.tight_layout()
    return fig


def plot_mirror_spherical_aberration():
    """Plot spherical aberration in concave mirror"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    R = 0.2  # Radius of curvature
    mirror = SphericalMirror(R, diameter=0.15)

    heights = np.linspace(0.001, 0.075, 25)
    colors = plt.cm.plasma(np.linspace(0, 1, len(heights)))

    ax1 = axes[0]
    z_foci = []

    for h, color in zip(heights, colors):
        result = mirror.trace_ray(h, 0.0)
        if result['valid']:
            ax1.plot(result['path_z'], result['path_y'], color=color, linewidth=1, alpha=0.7)
            ax1.plot(result['path_z'], -result['path_y'], color=color, linewidth=1, alpha=0.7)
            z_foci.append(result['z_focus'])

    draw_mirror_profile(ax1, mirror)

    ax1.axvline(mirror.f_paraxial, color='green', linestyle='--', label='Paraxial focus')
    if z_foci:
        ax1.axvline(z_foci[-1], color='red', linestyle='--', label='Marginal focus')

    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.set_xlabel('z (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_title(f'Spherical Mirror (R = {R*100:.0f} cm)\nParaxial f = R/2 = {mirror.f_paraxial*100:.0f} cm')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.15, 0.15)
    ax1.set_ylim(-0.1, 0.1)
    ax1.set_aspect('equal')

    # Caustic curve and aberration analysis
    ax2 = axes[1]

    if z_foci:
        # Longitudinal aberration
        ax2.plot(heights * 1000, (np.array(z_foci) - mirror.f_paraxial) * 1000,
                'b-', linewidth=2, label='Spherical mirror')

        # Theoretical: Delta_z ~ h^2 / (2R) for third-order
        h_theory = np.linspace(0, 0.075, 100)
        delta_z_theory = -h_theory**2 / (2 * R)
        ax2.plot(h_theory * 1000, delta_z_theory * 1000, 'r--', linewidth=2,
                label=r'Theory: $\Delta z = -h^2/(2R)$')

    ax2.set_xlabel('Ray height h (mm)')
    ax2.set_ylabel('Longitudinal aberration (mm)')
    ax2.set_title('Spherical Mirror Aberration')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='black', linewidth=0.5)

    plt.tight_layout()
    return fig


def plot_circle_of_least_confusion():
    """Plot the circle of least confusion"""

    fig, ax = plt.subplots(figsize=(10, 8))

    R1 = 0.1
    R2 = np.inf
    lens = SphericalLens(R1, R2, 0.01, n=1.5, diameter=0.08)

    heights = np.linspace(0.001, 0.04, 30)

    # Find marginal and paraxial foci
    z_foci = []
    y_final = []

    for h in heights:
        result = lens.trace_ray(h, 0.0)
        if result['valid']:
            z_foci.append(result['z_focus'])
            # Calculate y at different z positions
            y_final.append((result['z_focus'], result['y_at_focus']))

    z_paraxial = lens.f_paraxial
    z_marginal = z_foci[-1] if z_foci else z_paraxial

    # Plot rays
    colors = plt.cm.viridis(np.linspace(0, 1, len(heights)))
    for h, color in zip(heights, colors):
        result = lens.trace_ray(h, 0.0)
        if result['valid']:
            ax.plot(result['path_z'], result['path_y'], color=color, linewidth=0.8, alpha=0.6)
            ax.plot(result['path_z'], -result['path_y'], color=color, linewidth=0.8, alpha=0.6)

    draw_lens_profile(ax, lens)

    # Mark key planes
    ax.axvline(z_paraxial, color='green', linestyle='--', linewidth=2, label='Paraxial focus')
    ax.axvline(z_marginal, color='red', linestyle='--', linewidth=2, label='Marginal focus')

    # Circle of least confusion is at approximately 3/4 of the way from marginal to paraxial
    z_clc = z_marginal + 0.75 * (z_paraxial - z_marginal)
    ax.axvline(z_clc, color='purple', linestyle='--', linewidth=2,
              label='Circle of least confusion')

    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('z (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Spherical Aberration: Circle of Least Confusion\n'
                'Best focus is between marginal and paraxial foci')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 0.22)
    ax.set_ylim(-0.05, 0.05)

    # Add annotations
    ax.annotate('', xy=(z_paraxial, 0.04), xytext=(z_marginal, 0.04),
               arrowprops=dict(arrowstyle='<->', color='black'))
    ax.text((z_paraxial + z_marginal)/2, 0.042,
           f'Longitudinal SA = {(z_paraxial - z_marginal)*1000:.1f} mm',
           ha='center', fontsize=10)

    return fig


def main():
    """Main function to demonstrate spherical aberration"""

    # Create figures
    fig1 = plot_lens_spherical_aberration()
    fig2 = plot_mirror_spherical_aberration()
    fig3 = plot_circle_of_least_confusion()

    # Save plots
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)

    fig1.savefig(os.path.join(output_dir, 'spherical_aberration_lens.png'),
                 dpi=150, bbox_inches='tight')
    fig2.savefig(os.path.join(output_dir, 'spherical_aberration_mirror.png'),
                 dpi=150, bbox_inches='tight')
    fig3.savefig(os.path.join(output_dir, 'spherical_aberration_clc.png'),
                 dpi=150, bbox_inches='tight')

    print(f"Plots saved to {output_dir}/spherical_aberration_*.png")

    # Print analysis
    print("\n=== Spherical Aberration Analysis ===")
    print("\nKey concepts:")
    print("- Paraxial rays (near axis) focus at paraxial focal point f = R/2 (mirror)")
    print("- Marginal rays (edge of aperture) focus closer to the surface")
    print("- Longitudinal SA = z_marginal - z_paraxial")
    print("- Circle of least confusion: best focus plane")
    print("\nMitigation strategies:")
    print("- Use parabolic mirrors instead of spherical")
    print("- Correct orientation of plano-convex lenses")
    print("- Aspheric lens designs")
    print("- Lens combinations that cancel aberrations")


if __name__ == "__main__":
    main()
