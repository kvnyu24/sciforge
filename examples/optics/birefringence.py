"""
Example 116: Birefringence

This example demonstrates birefringence (double refraction) in anisotropic crystals,
showing how ordinary and extraordinary rays experience different refractive indices.

Physics:
    In a uniaxial crystal:
    - Ordinary ray: n_o (independent of direction)
    - Extraordinary ray: n_e(theta) = n_o * n_e / sqrt(n_o^2 * sin^2(theta) + n_e^2 * cos^2(theta))

    where theta is the angle between the ray and the optic axis.

    Key phenomena:
    - Double refraction (calcite)
    - Conical refraction (biaxial crystals)
    - Polarization-dependent phase shift (waveplates)
    - Optical rotation (circular birefringence)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from mpl_toolkits.mplot3d import Axes3D


class UniaxialCrystal:
    """Uniaxial birefringent crystal (e.g., calcite, quartz)"""

    def __init__(self, n_o: float, n_e: float, optic_axis_angle: float = 0.0):
        """
        Args:
            n_o: Ordinary refractive index
            n_e: Extraordinary refractive index
            optic_axis_angle: Angle of optic axis from surface normal (radians)
        """
        self.n_o = n_o
        self.n_e = n_e
        self.optic_axis_angle = optic_axis_angle

        # Crystal type
        self.type = 'positive' if n_e > n_o else 'negative'
        self.birefringence = abs(n_e - n_o)

    def extraordinary_index(self, theta: float) -> float:
        """
        Calculate extraordinary refractive index at angle theta from optic axis.

        n_e(theta) = n_o * n_e / sqrt(n_o^2 * sin^2(theta) + n_e^2 * cos^2(theta))
        """
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        denom = np.sqrt(self.n_o**2 * sin_theta**2 + self.n_e**2 * cos_theta**2)
        return self.n_o * self.n_e / denom

    def walk_off_angle(self, theta: float) -> float:
        """
        Calculate walk-off angle (angle between ray and wave vectors for e-ray).

        rho = arctan((n_o^2 - n_e^2) * sin(theta) * cos(theta) / (n_o^2 * sin^2(theta) + n_e^2 * cos^2(theta)))
        """
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        num = (self.n_o**2 - self.n_e**2) * sin_theta * cos_theta
        denom = self.n_o**2 * sin_theta**2 + self.n_e**2 * cos_theta**2

        return np.arctan(num / denom)

    def refract_ray(self, incident_angle: float) -> tuple:
        """
        Calculate refraction angles for o-ray and e-ray.

        Args:
            incident_angle: Angle from normal in air

        Returns:
            (theta_o, theta_e) refraction angles
        """
        sin_inc = np.sin(incident_angle)

        # Ordinary ray: simple Snell's law
        sin_o = sin_inc / self.n_o
        if abs(sin_o) > 1:
            theta_o = None
        else:
            theta_o = np.arcsin(sin_o)

        # Extraordinary ray: depends on direction relative to optic axis
        # Simplified for optic axis parallel to surface
        n_eff = self.extraordinary_index(np.pi/2 - incident_angle + self.optic_axis_angle)
        sin_e = sin_inc / n_eff
        if abs(sin_e) > 1:
            theta_e = None
        else:
            theta_e = np.arcsin(sin_e)

        return theta_o, theta_e

    def phase_difference(self, thickness: float, wavelength: float, theta: float = 0) -> float:
        """
        Calculate phase difference between o and e rays through crystal.

        delta = 2*pi * (n_e - n_o) * d / lambda (for theta = 0)
        """
        n_e_eff = self.extraordinary_index(theta)
        return 2 * np.pi * (n_e_eff - self.n_o) * thickness / wavelength


class CalciteDemo:
    """Double refraction demonstration using calcite crystal"""

    def __init__(self, thickness: float):
        """
        Args:
            thickness: Crystal thickness (m)
        """
        # Calcite parameters at 589 nm
        self.crystal = UniaxialCrystal(n_o=1.658, n_e=1.486)
        self.thickness = thickness

    def ray_displacement(self, wavelength: float = 589e-9) -> float:
        """Calculate lateral displacement between o and e rays"""
        # For light at normal incidence with optic axis at 45 degrees
        walk_off = self.crystal.walk_off_angle(np.pi/4)
        return self.thickness * np.tan(walk_off)


def plot_index_surface():
    """Plot refractive index surfaces for uniaxial crystals"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Positive uniaxial (quartz-like)
    ax1 = axes[0]
    n_o = 1.544
    n_e = 1.553

    theta = np.linspace(0, 2*np.pi, 200)

    # Ordinary surface (sphere)
    r_o = n_o * np.ones_like(theta)

    # Extraordinary surface (ellipsoid cross-section)
    crystal = UniaxialCrystal(n_o, n_e)
    r_e = np.array([crystal.extraordinary_index(t) for t in theta])

    ax1.plot(r_o * np.cos(theta), r_o * np.sin(theta), 'b-', linewidth=2, label='Ordinary (sphere)')
    ax1.plot(r_e * np.cos(theta), r_e * np.sin(theta), 'r-', linewidth=2, label='Extraordinary (ellipsoid)')

    # Mark optic axis
    ax1.arrow(0, 0, 0, 1.6, head_width=0.02, head_length=0.02, fc='green', ec='green')
    ax1.text(0.05, 1.55, 'Optic axis', fontsize=10, color='green')

    ax1.set_xlim(-1.7, 1.7)
    ax1.set_ylim(-1.7, 1.7)
    ax1.set_aspect('equal')
    ax1.set_xlabel('$n_x$')
    ax1.set_ylabel('$n_z$ (optic axis)')
    ax1.set_title(f'Positive Uniaxial Crystal (n_e > n_o)\n$n_o$ = {n_o}, $n_e$ = {n_e}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Negative uniaxial (calcite-like)
    ax2 = axes[1]
    n_o = 1.658
    n_e = 1.486

    r_o = n_o * np.ones_like(theta)
    crystal = UniaxialCrystal(n_o, n_e)
    r_e = np.array([crystal.extraordinary_index(t) for t in theta])

    ax2.plot(r_o * np.cos(theta), r_o * np.sin(theta), 'b-', linewidth=2, label='Ordinary')
    ax2.plot(r_e * np.cos(theta), r_e * np.sin(theta), 'r-', linewidth=2, label='Extraordinary')

    ax2.arrow(0, 0, 0, 1.75, head_width=0.02, head_length=0.02, fc='green', ec='green')
    ax2.text(0.05, 1.7, 'Optic axis', fontsize=10, color='green')

    ax2.set_xlim(-1.8, 1.8)
    ax2.set_ylim(-1.8, 1.8)
    ax2.set_aspect('equal')
    ax2.set_xlabel('$n_x$')
    ax2.set_ylabel('$n_z$ (optic axis)')
    ax2.set_title(f'Negative Uniaxial Crystal (n_e < n_o)\n$n_o$ = {n_o}, $n_e$ = {n_e}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_double_refraction():
    """Plot double refraction (ray splitting) in calcite"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Calcite crystal
    n_o = 1.658
    n_e = 1.486
    crystal = UniaxialCrystal(n_o, n_e, optic_axis_angle=np.pi/4)

    # Plot 1: Ray diagram
    ax1 = axes[0]

    thickness = 0.01  # 1 cm

    # Draw crystal
    crystal_rect = Rectangle((-0.02, 0), 0.04, thickness,
                             facecolor='lightblue', edgecolor='blue',
                             linewidth=2, alpha=0.5)
    ax1.add_patch(crystal_rect)

    # Optic axis (at 45 degrees in this example)
    ax1.arrow(0, thickness/2, 0.015, 0.015, head_width=0.001,
             head_length=0.001, fc='green', ec='green')
    ax1.text(0.018, thickness/2 + 0.015, 'Optic axis', fontsize=9, color='green')

    # Incident ray
    ax1.arrow(0, -0.005, 0, 0.004, head_width=0.001,
             head_length=0.001, fc='black', ec='black', linewidth=2)
    ax1.text(0.002, -0.003, 'Incident', fontsize=9)

    # O-ray (straight through for normal incidence)
    ax1.plot([0, 0], [0, thickness], 'b-', linewidth=2, label='O-ray')
    ax1.plot([0, 0], [thickness, thickness + 0.005], 'b-', linewidth=2)

    # E-ray (displaced due to walk-off)
    walk_off = crystal.walk_off_angle(np.pi/4)
    displacement = thickness * np.tan(walk_off)
    ax1.plot([0, displacement], [0, thickness], 'r-', linewidth=2, label='E-ray')
    ax1.plot([displacement, displacement], [thickness, thickness + 0.005], 'r-', linewidth=2)

    # Mark displacement
    ax1.annotate('', xy=(displacement, thickness + 0.002), xytext=(0, thickness + 0.002),
                arrowprops=dict(arrowstyle='<->', color='purple'))
    ax1.text(displacement/2, thickness + 0.003, f'd = {displacement*1e3:.2f} mm',
            ha='center', fontsize=9, color='purple')

    ax1.set_xlim(-0.025, 0.025)
    ax1.set_ylim(-0.008, thickness + 0.008)
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('z (m)')
    ax1.set_title(f'Double Refraction in Calcite\n'
                 f'n_o = {n_o}, n_e = {n_e}, thickness = {thickness*100:.0f} cm')
    ax1.legend()
    ax1.set_aspect('equal')

    # Plot 2: Displacement vs crystal thickness
    ax2 = axes[1]

    thicknesses = np.linspace(0.001, 0.02, 100)
    displacements = [t * np.tan(walk_off) * 1e3 for t in thicknesses]

    ax2.plot(thicknesses * 1e3, displacements, 'b-', linewidth=2)

    ax2.set_xlabel('Crystal thickness (mm)')
    ax2.set_ylabel('Ray displacement (mm)')
    ax2.set_title('O-ray to E-ray Displacement vs Thickness')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_birefringence_applications():
    """Plot applications of birefringence"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Waveplate retardance vs wavelength
    ax1 = axes[0, 0]

    # Quartz waveplate designed for 632.8 nm
    n_o_quartz = lambda wl: 1.5443 + 0.0031 / (wl*1e6)**2  # Sellmeier approx
    n_e_quartz = lambda wl: 1.5534 + 0.0031 / (wl*1e6)**2

    wavelengths = np.linspace(400e-9, 800e-9, 200)

    # Quarter-wave plate thickness for 632.8 nm
    wl_design = 632.8e-9
    delta_n = n_e_quartz(wl_design) - n_o_quartz(wl_design)
    qwp_thickness = wl_design / (4 * delta_n)

    retardance = []
    for wl in wavelengths:
        dn = n_e_quartz(wl) - n_o_quartz(wl)
        phase = 2 * np.pi * dn * qwp_thickness / wl
        retardance.append(np.degrees(phase))

    ax1.plot(wavelengths * 1e9, retardance, 'b-', linewidth=2)
    ax1.axhline(90, color='red', linestyle='--', label='Quarter-wave (90 deg)')
    ax1.axhline(180, color='green', linestyle='--', label='Half-wave (180 deg)')
    ax1.axvline(632.8, color='gray', linestyle=':', alpha=0.5)

    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Retardance (degrees)')
    ax1.set_title('Waveplate Retardance vs Wavelength\n(Designed as QWP at 632.8 nm)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Stress birefringence
    ax2 = axes[0, 1]

    # Simulate stressed glass plate
    x = np.linspace(-1, 1, 200)
    y = np.linspace(-1, 1, 200)
    X, Y = np.meshgrid(x, y)

    # Stress pattern (simplified: compression in center)
    stress = np.exp(-(X**2 + Y**2) / 0.3)

    # Birefringence proportional to stress
    delta_n_stress = 2e-5 * stress  # Stress-optic coefficient

    # Phase difference for 5mm thick glass at 632.8 nm
    thickness = 5e-3
    wavelength = 632.8e-9
    phase = 2 * np.pi * delta_n_stress * thickness / wavelength

    # Intensity between crossed polarizers
    intensity = np.sin(phase/2)**2

    im = ax2.imshow(intensity, extent=[-1, 1, -1, 1], cmap='rainbow',
                   origin='lower')
    plt.colorbar(im, ax=ax2, label='Intensity')

    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Stress Birefringence Pattern\n(Between crossed polarizers)')

    # Plot 3: Michel-Levy chart (simplified)
    ax3 = axes[1, 0]

    # Birefringence values
    birefringence_vals = [0.005, 0.01, 0.02, 0.04, 0.08]
    thicknesses = np.linspace(0, 50e-6, 100)

    for delta_n in birefringence_vals:
        # Path difference in nm
        path_diff = delta_n * thicknesses * 1e9
        ax3.plot(path_diff, thicknesses * 1e6, linewidth=2,
                label=f'$\Delta n$ = {delta_n}')

    ax3.set_xlabel('Path difference (nm)')
    ax3.set_ylabel('Thickness (um)')
    ax3.set_title('Simplified Michel-Levy Chart\n(Birefringence identification)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Add color bands (Newton's colors)
    colors = ['gray', 'yellow', 'red', 'purple', 'blue', 'green', 'yellow']
    color_positions = [0, 150, 500, 550, 650, 850, 1000]
    for i in range(len(colors)-1):
        ax3.axvspan(color_positions[i], color_positions[i+1], alpha=0.2, color=colors[i])

    # Plot 4: Liquid crystal cell
    ax4 = axes[1, 1]

    # Simulate LC cell twist
    z_positions = np.linspace(0, 1, 50)  # Normalized cell thickness

    # Different applied voltages (normalized)
    voltages = [0, 0.5, 1.0, 2.0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(voltages)))

    for V, color in zip(voltages, colors):
        # Director angle varies with z (twisted nematic)
        # At high voltage, director aligns with field (reduced twist)
        twist_angle = 90 * (1 - V/(V + 1))  # Simplified response
        angles = twist_angle * z_positions * np.pi / 180

        ax4.plot(z_positions, np.degrees(angles), color=color, linewidth=2,
                label=f'V = {V}')

    ax4.set_xlabel('Position in cell (normalized)')
    ax4.set_ylabel('Director angle (degrees)')
    ax4.set_title('Liquid Crystal Director Profile\n(Twisted nematic cell)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_optical_activity():
    """Plot optical activity (circular birefringence)"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Quartz optical rotation
    # Rotation: rho = pi * d * (n_L - n_R) / lambda
    # For quartz: ~21.7 deg/mm at 589 nm

    ax1 = axes[0]

    wavelengths = np.linspace(400e-9, 800e-9, 100)

    # Optical rotation dispersion (approximate)
    # rho ~ 1/lambda^2 (Drude model)
    rotation_per_mm = 21.7 * (589e-9 / wavelengths)**2  # deg/mm

    thicknesses = [1, 2, 5, 10]  # mm
    colors = plt.cm.viridis(np.linspace(0, 1, len(thicknesses)))

    for t, color in zip(thicknesses, colors):
        rotation = rotation_per_mm * t
        ax1.plot(wavelengths * 1e9, rotation, color=color, linewidth=2,
                label=f'{t} mm')

    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Rotation angle (degrees)')
    ax1.set_title('Optical Rotation in Quartz\n(Circular birefringence)')
    ax1.legend(title='Thickness')
    ax1.grid(True, alpha=0.3)

    # Faraday rotation
    ax2 = axes[1]

    # Verdet constant for heavy flint glass: ~30 rad/(T*m) at 589 nm
    verdet = 30  # rad/(T*m)

    magnetic_fields = np.linspace(0, 2, 100)  # Tesla
    lengths = [0.01, 0.02, 0.05, 0.1]  # meters

    for L, color in zip(lengths, colors):
        rotation = np.degrees(verdet * magnetic_fields * L)
        ax2.plot(magnetic_fields, rotation, color=color, linewidth=2,
                label=f'{L*100:.0f} cm')

    ax2.set_xlabel('Magnetic field (Tesla)')
    ax2.set_ylabel('Faraday rotation (degrees)')
    ax2.set_title('Faraday Rotation\n(Magnetically induced circular birefringence)')
    ax2.legend(title='Path length')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_conoscopic_figure():
    """Plot conoscopic interference figure"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Uniaxial crystal (optic axis perpendicular to surface)
    ax1 = axes[0]

    n_o = 1.544
    n_e = 1.553
    thickness = 1e-3  # 1 mm
    wavelength = 589e-9

    # Create angular grid
    theta_max = 0.1  # radians
    n_points = 200
    theta_x = np.linspace(-theta_max, theta_max, n_points)
    theta_y = np.linspace(-theta_max, theta_max, n_points)
    Tx, Ty = np.meshgrid(theta_x, theta_y)
    theta = np.sqrt(Tx**2 + Ty**2)

    # Phase difference
    crystal = UniaxialCrystal(n_o, n_e)
    n_e_eff = np.array([[crystal.extraordinary_index(t) for t in row] for row in theta])

    # Path difference
    delta = 2 * np.pi * (n_e_eff - n_o) * thickness / wavelength

    # Intensity between crossed polarizers
    # I = sin^2(2*phi) * sin^2(delta/2)
    # where phi is azimuthal angle
    phi = np.arctan2(Ty, Tx)
    intensity = np.sin(2 * phi)**2 * np.sin(delta / 2)**2

    im = ax1.imshow(intensity, extent=[-theta_max*180/np.pi, theta_max*180/np.pi,
                                       -theta_max*180/np.pi, theta_max*180/np.pi],
                   cmap='gray', origin='lower')
    plt.colorbar(im, ax=ax1, label='Intensity')

    ax1.set_xlabel('$\\theta_x$ (degrees)')
    ax1.set_ylabel('$\\theta_y$ (degrees)')
    ax1.set_title('Conoscopic Figure: Uniaxial Crystal\n(Optic axis perpendicular to surface)')

    # Biaxial crystal (simplified)
    ax2 = axes[1]

    # Two optic axes at angles +-V from z-axis
    V = 30  # degrees
    n_alpha = 1.52
    n_beta = 1.53
    n_gamma = 1.55

    # Create interference pattern (simplified)
    optic_axis_1 = np.array([np.sin(np.radians(V)), 0])
    optic_axis_2 = np.array([-np.sin(np.radians(V)), 0])

    # Distance from each optic axis
    dist1 = np.sqrt((Tx - optic_axis_1[0])**2 + Ty**2)
    dist2 = np.sqrt((Tx - optic_axis_2[0])**2 + Ty**2)

    # Simplified intensity pattern
    delta1 = 2 * np.pi * (n_gamma - n_alpha) * thickness * dist1 / wavelength
    delta2 = 2 * np.pi * (n_gamma - n_alpha) * thickness * dist2 / wavelength

    intensity2 = np.sin(2 * phi)**2 * np.sin((delta1 + delta2) / 4)**2

    im2 = ax2.imshow(intensity2, extent=[-theta_max*180/np.pi, theta_max*180/np.pi,
                                         -theta_max*180/np.pi, theta_max*180/np.pi],
                    cmap='gray', origin='lower')
    plt.colorbar(im2, ax=ax2, label='Intensity')

    # Mark optic axes
    ax2.plot([np.degrees(np.sin(np.radians(V)))], [0], 'rx', markersize=10)
    ax2.plot([-np.degrees(np.sin(np.radians(V)))], [0], 'rx', markersize=10)

    ax2.set_xlabel('$\\theta_x$ (degrees)')
    ax2.set_ylabel('$\\theta_y$ (degrees)')
    ax2.set_title('Conoscopic Figure: Biaxial Crystal\n(Two optic axes marked)')

    plt.tight_layout()
    return fig


def main():
    """Main function to demonstrate birefringence"""

    # Create figures
    fig1 = plot_index_surface()
    fig2 = plot_double_refraction()
    fig3 = plot_birefringence_applications()
    fig4 = plot_optical_activity()
    fig5 = plot_conoscopic_figure()

    # Save plots
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)

    fig1.savefig(os.path.join(output_dir, 'birefringence_index_surface.png'),
                 dpi=150, bbox_inches='tight')
    fig2.savefig(os.path.join(output_dir, 'birefringence_double_refraction.png'),
                 dpi=150, bbox_inches='tight')
    fig3.savefig(os.path.join(output_dir, 'birefringence_applications.png'),
                 dpi=150, bbox_inches='tight')
    fig4.savefig(os.path.join(output_dir, 'birefringence_optical_activity.png'),
                 dpi=150, bbox_inches='tight')
    fig5.savefig(os.path.join(output_dir, 'birefringence_conoscopic.png'),
                 dpi=150, bbox_inches='tight')

    print(f"Plots saved to {output_dir}/birefringence_*.png")

    # Print analysis
    print("\n=== Birefringence Analysis ===")
    print("\nCommon birefringent materials:")
    materials = [
        ('Calcite', 1.658, 1.486, 'negative'),
        ('Quartz', 1.544, 1.553, 'positive'),
        ('Sapphire', 1.768, 1.760, 'negative'),
        ('Rutile', 2.616, 2.903, 'positive'),
    ]

    for name, n_o, n_e, type_ in materials:
        print(f"  {name}: n_o = {n_o}, n_e = {n_e}, Delta_n = {abs(n_e-n_o):.3f} ({type_})")

    print("\nApplications:")
    print("  - Waveplates (QWP, HWP)")
    print("  - Polarizing beam splitters")
    print("  - Stress analysis (photoelasticity)")
    print("  - Liquid crystal displays")
    print("  - Optical modulators")


if __name__ == "__main__":
    main()
