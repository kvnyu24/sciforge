"""
Example 105: GRIN Lens Trajectories

This example demonstrates gradient-index (GRIN) optics, where the refractive
index varies continuously within the material, causing rays to follow curved
paths without discrete refraction events.

Physics:
    For a radial GRIN with parabolic profile:
    n(r) = n_0 * (1 - (g^2 * r^2) / 2)

    where g is the gradient constant and r is radial distance.

    Ray paths follow sinusoidal trajectories:
    r(z) = r_0 * cos(g*z) + (theta_0/g) * sin(g*z)

    The pitch length (one complete oscillation):
    P = 2*pi/g

Applications:
    - GRIN rod lenses for fiber coupling
    - Selfoc lenses in photocopiers
    - Gradient index optical fibers
    - Eye lens (crystalline lens)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D


class GRINLens:
    """
    Gradient-index lens with parabolic radial profile.

    n(r) = n_0 * sqrt(1 - (g*r)^2) ≈ n_0 * (1 - g^2*r^2/2) for small r
    """

    def __init__(self, n0: float, g: float, length: float, radius: float):
        """
        Args:
            n0: Central (axial) refractive index
            g: Gradient constant (1/mm)
            length: Lens length
            radius: Lens radius
        """
        self.n0 = n0
        self.g = g
        self.length = length
        self.radius = radius

        # Pitch length (one complete sinusoidal oscillation)
        self.pitch = 2 * np.pi / g

        # Numerical aperture
        self.NA = n0 * g * radius

    def refractive_index(self, r: float) -> float:
        """Calculate refractive index at radial position r"""
        if r >= self.radius:
            return 1.0  # Outside lens

        # Parabolic approximation (valid for small g*r)
        return self.n0 * np.sqrt(max(0, 1 - (self.g * r)**2))

    def trace_ray(self, r0: float, theta0: float, n_points: int = 500) -> dict:
        """
        Trace ray through GRIN lens using analytical solution.

        For parabolic GRIN profile, rays follow:
        r(z) = r0 * cos(g*z) + (theta0/g) * sin(g*z)
        theta(z) = -r0*g * sin(g*z) + theta0 * cos(g*z)

        Args:
            r0: Initial radial position
            theta0: Initial angle (radians, from axis)
            n_points: Number of points in path

        Returns:
            Dictionary with ray path
        """
        z = np.linspace(0, self.length, n_points)

        # Analytical ray trajectory (paraxial approximation)
        r = r0 * np.cos(self.g * z) + (theta0 / self.g) * np.sin(self.g * z)
        theta = -r0 * self.g * np.sin(self.g * z) + theta0 * np.cos(self.g * z)

        # Check if ray leaves lens
        valid = np.abs(r) < self.radius

        return {
            'z': z,
            'r': r,
            'theta': theta,
            'valid': valid
        }

    def trace_ray_numerical(self, r0: float, theta0: float, dz: float = 0.0001) -> dict:
        """
        Trace ray numerically using Fermat's principle.

        Ray equation in GRIN: d²r/dz² = (1/n) * (dn/dr)

        Args:
            r0: Initial radial position
            theta0: Initial angle
            dz: Step size

        Returns:
            Dictionary with ray path
        """
        z_vals = [0]
        r_vals = [r0]
        theta_vals = [theta0]

        r = r0
        theta = theta0
        z = 0

        while z < self.length and abs(r) < self.radius:
            # Get local refractive index and gradient
            n = self.refractive_index(abs(r))
            # dn/dr for parabolic profile
            if abs(r) < self.radius * 0.99:
                dn_dr = -self.n0 * self.g**2 * r / np.sqrt(max(0.01, 1 - (self.g * r)**2))
            else:
                dn_dr = 0

            # Update angle: d(theta)/dz = (1/n) * dn/dr
            d_theta = (1/n) * dn_dr * dz
            theta += d_theta

            # Update position
            r += theta * dz
            z += dz

            z_vals.append(z)
            r_vals.append(r)
            theta_vals.append(theta)

        return {
            'z': np.array(z_vals),
            'r': np.array(r_vals),
            'theta': np.array(theta_vals),
            'valid': np.ones(len(z_vals), dtype=bool)
        }

    def abcd_matrix(self) -> np.ndarray:
        """
        Calculate ABCD matrix for the GRIN lens.

        For length L and gradient g:
        A = cos(gL)
        B = sin(gL)/g
        C = -g*sin(gL)
        D = cos(gL)
        """
        gL = self.g * self.length
        A = np.cos(gL)
        B = np.sin(gL) / self.g
        C = -self.g * np.sin(gL)
        D = np.cos(gL)
        return np.array([[A, B], [C, D]])

    def focal_length(self) -> float:
        """Calculate effective focal length"""
        M = self.abcd_matrix()
        C = M[1, 0]
        if abs(C) < 1e-10:
            return np.inf
        return -1.0 / C


class GRINFiber:
    """Graded-index optical fiber"""

    def __init__(self, n_core: float, n_clad: float, core_radius: float,
                 length: float, alpha: float = 2.0):
        """
        Args:
            n_core: Core center refractive index
            n_clad: Cladding refractive index
            core_radius: Core radius
            length: Fiber length
            alpha: Profile parameter (2 = parabolic, inf = step index)
        """
        self.n_core = n_core
        self.n_clad = n_clad
        self.a = core_radius
        self.length = length
        self.alpha = alpha

        # Relative index difference
        self.delta = (n_core - n_clad) / n_core

        # Numerical aperture
        self.NA = np.sqrt(n_core**2 - n_clad**2)

    def refractive_index(self, r: float) -> float:
        """Calculate refractive index at radius r"""
        if r >= self.a:
            return self.n_clad

        # Power-law profile
        return self.n_core * np.sqrt(1 - 2*self.delta*(r/self.a)**self.alpha)

    def trace_ray(self, r0: float, theta0: float, n_points: int = 1000) -> dict:
        """Trace ray through graded-index fiber"""
        z = np.linspace(0, self.length, n_points)
        dz = z[1] - z[0]

        r_vals = [r0]
        theta_vals = [theta0]

        r = r0
        theta = theta0

        for _ in range(len(z) - 1):
            # Update using ray equation
            n = self.refractive_index(abs(r))

            # Gradient for parabolic profile
            if abs(r) < self.a * 0.99:
                dn_dr = -self.n_core * self.delta * self.alpha * \
                        (abs(r)/self.a)**(self.alpha-1) / self.a * np.sign(r)
                dn_dr /= np.sqrt(max(0.01, 1 - 2*self.delta*(abs(r)/self.a)**self.alpha))
            else:
                dn_dr = 0

            # Update angle and position
            theta += (1/n) * dn_dr * dz
            r += theta * dz

            # Reflect at cladding interface (simplified)
            if abs(r) > self.a:
                r = np.sign(r) * self.a
                theta = -theta

            r_vals.append(r)
            theta_vals.append(theta)

        return {
            'z': z,
            'r': np.array(r_vals),
            'theta': np.array(theta_vals)
        }


def plot_grin_lens_rays():
    """Plot ray trajectories through GRIN lens"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Create GRIN lens
    n0 = 1.6
    g = 20.0  # 1/m (gradient constant)
    length = 0.2  # 200 mm
    radius = 0.02  # 20 mm

    lens = GRINLens(n0, g, length, radius)

    print(f"GRIN Lens Parameters:")
    print(f"  n0 = {n0}, g = {g} /m")
    print(f"  Pitch length = {lens.pitch*1000:.1f} mm")
    print(f"  NA = {lens.NA:.3f}")

    # Plot 1: Different length lenses (quarter, half, full pitch)
    ax1 = axes[0, 0]

    pitch_fractions = [0.25, 0.5, 0.75, 1.0]
    colors = ['blue', 'green', 'orange', 'red']

    for frac, color in zip(pitch_fractions, colors):
        L = frac * lens.pitch
        temp_lens = GRINLens(n0, g, L, radius)

        # Draw lens
        rect = Rectangle((0, -radius), L, 2*radius,
                         facecolor=color, alpha=0.2, edgecolor=color, linewidth=2)
        ax1.add_patch(rect)

        # Trace rays
        for r0 in np.linspace(-0.015, 0.015, 5):
            result = temp_lens.trace_ray(r0, 0)
            ax1.plot(result['z'], result['r'], color=color, linewidth=1)

        # Mark entry and exit points
        ax1.axvline(L, color=color, linestyle='--', alpha=0.5)
        ax1.text(L/2, radius*1.1, f'{frac}P', ha='center', color=color, fontsize=10)

    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.set_xlabel('z (m)')
    ax1.set_ylabel('r (m)')
    ax1.set_title('GRIN Lens: Effect of Length\n(P = pitch length)')
    ax1.set_xlim(-0.02, lens.pitch + 0.02)
    ax1.set_ylim(-0.03, 0.03)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Ray tracing for different entry angles
    ax2 = axes[0, 1]

    half_pitch_lens = GRINLens(n0, g, lens.pitch/2, radius)

    # Draw lens
    rect = Rectangle((0, -radius), half_pitch_lens.length, 2*radius,
                     facecolor='lightblue', alpha=0.5, edgecolor='blue', linewidth=2)
    ax2.add_patch(rect)

    # Trace rays at different angles from same point
    angles = np.radians([-5, -2.5, 0, 2.5, 5])
    colors = plt.cm.viridis(np.linspace(0, 1, len(angles)))

    for angle, color in zip(angles, colors):
        result = half_pitch_lens.trace_ray(0.01, angle)
        ax2.plot(result['z'], result['r'], color=color, linewidth=1.5,
                label=f'{np.degrees(angle):.1f} deg')

    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_xlabel('z (m)')
    ax2.set_ylabel('r (m)')
    ax2.set_title('Half-Pitch GRIN Lens: Imaging Properties\n(Object at one surface imaged at other)')
    ax2.legend(title='Entry angle')
    ax2.set_xlim(-0.02, half_pitch_lens.length + 0.02)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Refractive index profile
    ax3 = axes[1, 0]

    r_range = np.linspace(-radius, radius, 200)
    n_values = [lens.refractive_index(abs(r)) for r in r_range]

    ax3.plot(r_range * 1000, n_values, 'b-', linewidth=2)
    ax3.axhline(n0, color='gray', linestyle='--', label=f'n_0 = {n0}')
    ax3.axvline(0, color='gray', linestyle='-', alpha=0.3)

    ax3.fill_between(r_range * 1000, 1.0, n_values, alpha=0.3)

    ax3.set_xlabel('Radial position r (mm)')
    ax3.set_ylabel('Refractive index n(r)')
    ax3.set_title('Parabolic GRIN Profile\n$n(r) = n_0 \\sqrt{1 - (gr)^2}$')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-radius*1000, radius*1000)

    # Plot 4: Collimation demonstration (quarter-pitch)
    ax4 = axes[1, 1]

    quarter_pitch_lens = GRINLens(n0, g, lens.pitch/4, radius)

    # Draw lens
    rect = Rectangle((0, -radius), quarter_pitch_lens.length, 2*radius,
                     facecolor='lightyellow', alpha=0.5, edgecolor='orange', linewidth=2)
    ax4.add_patch(rect)

    # Rays from a point source at the surface
    angles = np.linspace(-0.15, 0.15, 9)
    colors = plt.cm.plasma(np.linspace(0, 1, len(angles)))

    for angle, color in zip(angles, colors):
        # Ray starts at surface, at center
        result = quarter_pitch_lens.trace_ray(0, angle)

        # Extend before lens (virtual)
        z_before = np.linspace(-0.03, 0, 20)
        r_before = angle * z_before

        ax4.plot(z_before, r_before, color=color, linewidth=1, linestyle='--', alpha=0.5)
        ax4.plot(result['z'], result['r'], color=color, linewidth=1.5)

        # Extend after lens
        z_after = np.linspace(quarter_pitch_lens.length, quarter_pitch_lens.length + 0.03, 20)
        r_exit = result['r'][-1]
        theta_exit = result['theta'][-1]
        r_after = r_exit + theta_exit * (z_after - quarter_pitch_lens.length)
        ax4.plot(z_after, r_after, color=color, linewidth=1.5)

    ax4.axhline(0, color='black', linewidth=0.5)
    ax4.set_xlabel('z (m)')
    ax4.set_ylabel('r (m)')
    ax4.set_title('Quarter-Pitch GRIN Lens: Collimation\n(Point source at surface -> parallel beam)')
    ax4.set_xlim(-0.04, quarter_pitch_lens.length + 0.04)
    ax4.set_ylim(-0.025, 0.025)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_grin_fiber():
    """Plot graded-index fiber ray propagation"""

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Create graded-index fiber
    fiber = GRINFiber(
        n_core=1.48,
        n_clad=1.46,
        core_radius=0.025,  # 25 micron (scaled for visualization)
        length=0.5,  # 500 mm
        alpha=2.0  # Parabolic profile
    )

    # Plot 1: Ray trajectories
    ax1 = axes[0]

    # Draw fiber structure
    ax1.fill_between([0, fiber.length], [-fiber.a*1.5, -fiber.a*1.5],
                    [fiber.a*1.5, fiber.a*1.5], color='lightblue', alpha=0.3)
    ax1.fill_between([0, fiber.length], [-fiber.a, -fiber.a],
                    [fiber.a, fiber.a], color='lightcoral', alpha=0.5)

    # Trace rays
    entry_conditions = [
        (0.0, 0.05),
        (0.01, 0.02),
        (-0.015, 0.03),
        (0.02, -0.01),
        (-0.01, -0.04),
    ]
    colors = plt.cm.viridis(np.linspace(0, 1, len(entry_conditions)))

    for (r0, theta0), color in zip(entry_conditions, colors):
        result = fiber.trace_ray(r0, theta0)
        ax1.plot(result['z'], result['r'], color=color, linewidth=1,
                label=f'r0={r0*1000:.0f}um, theta={theta0:.2f}rad')

    ax1.axhline(fiber.a, color='black', linewidth=2)
    ax1.axhline(-fiber.a, color='black', linewidth=2)
    ax1.axhline(0, color='gray', linewidth=0.5, linestyle='--')

    ax1.set_xlabel('z (m)')
    ax1.set_ylabel('r (m)')
    ax1.set_title(f'Graded-Index Fiber (Parabolic Profile)\n'
                 f'n_core={fiber.n_core}, n_clad={fiber.n_clad}, NA={fiber.NA:.3f}')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.set_xlim(0, fiber.length)
    ax1.set_ylim(-0.04, 0.04)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Refractive index profiles comparison
    ax2 = axes[1]

    r_range = np.linspace(0, fiber.a * 1.2, 200)

    # Different profile parameters
    alphas = [1.5, 2.0, 3.0, np.inf]
    labels = ['alpha=1.5', 'alpha=2.0 (parabolic)', 'alpha=3.0', 'Step index']

    for alpha, label in zip(alphas, labels):
        if alpha == np.inf:
            n_vals = [fiber.n_core if r < fiber.a else fiber.n_clad for r in r_range]
        else:
            temp_fiber = GRINFiber(fiber.n_core, fiber.n_clad, fiber.a, fiber.length, alpha)
            n_vals = [temp_fiber.refractive_index(r) for r in r_range]

        ax2.plot(r_range / fiber.a, n_vals, linewidth=2, label=label)

    ax2.axvline(1.0, color='gray', linestyle='--', label='Core-cladding boundary')
    ax2.axhline(fiber.n_core, color='gray', linestyle=':', alpha=0.5)
    ax2.axhline(fiber.n_clad, color='gray', linestyle=':', alpha=0.5)

    ax2.set_xlabel('Normalized radius r/a')
    ax2.set_ylabel('Refractive index n(r)')
    ax2.set_title('Graded-Index Fiber Profiles\n$n(r) = n_{core}\\sqrt{1 - 2\\Delta(r/a)^\\alpha}$')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1.2)

    plt.tight_layout()
    return fig


def plot_grin_applications():
    """Plot various GRIN lens applications"""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Application 1: Fiber coupling (quarter-pitch)
    ax1 = axes[0]

    n0 = 1.6
    g = 25.0
    lens = GRINLens(n0, g, 2*np.pi/(4*g), 0.015)

    # Draw lens
    rect = Rectangle((0, -lens.radius), lens.length, 2*lens.radius,
                     facecolor='lightblue', alpha=0.5, edgecolor='blue', linewidth=2)
    ax1.add_patch(rect)

    # Draw fiber (left side)
    fiber_length = 0.03
    fiber_radius = 0.003
    rect_fiber = Rectangle((-fiber_length, -fiber_radius), fiber_length, 2*fiber_radius,
                           facecolor='lightcoral', alpha=0.5, edgecolor='red', linewidth=2)
    ax1.add_patch(rect_fiber)
    ax1.text(-fiber_length/2, -fiber_radius*1.5, 'Fiber', ha='center', fontsize=9)

    # Rays from fiber (diverging, then collimated)
    angles = np.linspace(-0.1, 0.1, 7)
    colors = plt.cm.Reds(np.linspace(0.3, 0.8, len(angles)))

    for angle, color in zip(angles, colors):
        # Before lens
        z_before = np.linspace(-fiber_length, 0, 20)
        r_before = angle * (z_before + fiber_length)

        # Through lens
        result = lens.trace_ray(0, angle)

        # After lens (collimated)
        z_after = np.linspace(lens.length, lens.length + 0.03, 20)
        r_after = result['r'][-1] * np.ones_like(z_after)

        ax1.plot(z_before, r_before, color=color, linewidth=1)
        ax1.plot(result['z'], result['r'], color=color, linewidth=1.5)
        ax1.plot(z_after, r_after, color=color, linewidth=1)

    ax1.set_xlabel('z (m)')
    ax1.set_ylabel('r (m)')
    ax1.set_title('Fiber Collimator (0.25P GRIN)\nDiverging fiber output -> Parallel beam')
    ax1.set_xlim(-0.04, lens.length + 0.04)
    ax1.set_ylim(-0.02, 0.02)
    ax1.grid(True, alpha=0.3)

    # Application 2: 1:1 Relay (half-pitch)
    ax2 = axes[1]

    lens2 = GRINLens(n0, g, 2*np.pi/(2*g), 0.015)

    rect2 = Rectangle((0, -lens2.radius), lens2.length, 2*lens2.radius,
                      facecolor='lightgreen', alpha=0.5, edgecolor='green', linewidth=2)
    ax2.add_patch(rect2)

    # Object at one surface, image at other
    for r0 in [-0.01, -0.005, 0, 0.005, 0.01]:
        for angle in np.linspace(-0.05, 0.05, 3):
            result = lens2.trace_ray(r0, angle)
            ax2.plot(result['z'], result['r'], 'purple', linewidth=0.8, alpha=0.6)

        # Mark object and image points
        ax2.plot(0, r0, 'go', markersize=8)
        ax2.plot(lens2.length, -r0, 'ro', markersize=8)

    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_xlabel('z (m)')
    ax2.set_ylabel('r (m)')
    ax2.set_title('1:1 Relay (0.5P GRIN)\nInverted image at same scale')
    ax2.set_xlim(-0.02, lens2.length + 0.02)
    ax2.set_ylim(-0.02, 0.02)
    ax2.grid(True, alpha=0.3)

    # Application 3: Endoscope relay
    ax3 = axes[2]

    # Multiple GRIN lenses in series
    lens3 = GRINLens(n0, g, 2*np.pi/(2*g), 0.012)
    n_lenses = 3
    gap = 0.005

    for i in range(n_lenses):
        z_start = i * (lens3.length + gap)
        rect3 = Rectangle((z_start, -lens3.radius), lens3.length, 2*lens3.radius,
                          facecolor='lightyellow', alpha=0.5, edgecolor='orange', linewidth=2)
        ax3.add_patch(rect3)

    # Trace ray through all lenses
    total_length = n_lenses * (lens3.length + gap)

    for r0 in [-0.008, -0.004, 0, 0.004, 0.008]:
        z_all = [0]
        r_all = [r0]

        r_current = r0
        theta_current = 0

        for i in range(n_lenses):
            z_start = i * (lens3.length + gap)

            # Through lens
            result = lens3.trace_ray(r_current, theta_current)
            z_all.extend((result['z'] + z_start).tolist())
            r_all.extend(result['r'].tolist())

            r_current = result['r'][-1]
            theta_current = result['theta'][-1]

            # Gap
            if i < n_lenses - 1:
                z_gap = np.linspace(z_start + lens3.length, z_start + lens3.length + gap, 10)
                r_gap = r_current + theta_current * (z_gap - z_start - lens3.length)
                z_all.extend(z_gap.tolist())
                r_all.extend(r_gap.tolist())
                r_current = r_gap[-1]

        ax3.plot(z_all, r_all, 'purple', linewidth=1, alpha=0.6)

    ax3.axhline(0, color='black', linewidth=0.5)
    ax3.set_xlabel('z (m)')
    ax3.set_ylabel('r (m)')
    ax3.set_title('Endoscope Relay (Multiple 0.5P GRIN)\nImage relayed through series')
    ax3.set_xlim(-0.02, total_length + 0.02)
    ax3.set_ylim(-0.015, 0.015)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    """Main function to demonstrate GRIN lens optics"""

    # Create figures
    fig1 = plot_grin_lens_rays()
    fig2 = plot_grin_fiber()
    fig3 = plot_grin_applications()

    # Save plots
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)

    fig1.savefig(os.path.join(output_dir, 'grin_lens_rays.png'),
                 dpi=150, bbox_inches='tight')
    fig2.savefig(os.path.join(output_dir, 'grin_fiber.png'),
                 dpi=150, bbox_inches='tight')
    fig3.savefig(os.path.join(output_dir, 'grin_applications.png'),
                 dpi=150, bbox_inches='tight')

    print(f"Plots saved to {output_dir}/grin_*.png")

    # Print analysis
    print("\n=== GRIN Lens Analysis ===")
    n0 = 1.6
    g = 20.0
    print(f"\nGRIN lens parameters: n0={n0}, g={g}/m")
    print(f"  Pitch length: {2*np.pi/g*1000:.1f} mm")
    print(f"  Quarter-pitch: {np.pi/(2*g)*1000:.1f} mm (collimation)")
    print(f"  Half-pitch: {np.pi/g*1000:.1f} mm (1:1 relay)")

    print("\nApplications:")
    print("  - Fiber collimators/couplers (quarter-pitch)")
    print("  - Image relay (half-pitch)")
    print("  - Endoscopes (multiple half-pitch lenses)")
    print("  - Compact imaging systems")


if __name__ == "__main__":
    main()
