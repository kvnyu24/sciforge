"""
Example 110: Fresnel Diffraction

This example demonstrates Fresnel (near-field) diffraction for various apertures,
including straight edges, circular apertures, and rectangular slits.

Physics:
    The Fresnel diffraction integral:
    U(P) = (1/i*lambda) * integral[U_0 * exp(i*k*r) / r * cos(theta)] dA

    Using Fresnel zones and Cornu spiral for analysis.

    Fresnel number: N_F = a^2 / (lambda * z)
    - Characterizes the number of Fresnel zones across the aperture
    - N_F >> 1: Near-field (Fresnel) regime
    - N_F << 1: Far-field (Fraunhofer) regime
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import fresnel
from scipy import ndimage


class FresnelDiffraction:
    """Fresnel diffraction calculator for various aperture geometries"""

    def __init__(self, wavelength: float, distance: float):
        """
        Args:
            wavelength: Light wavelength (m)
            distance: Distance to observation plane (m)
        """
        self.wavelength = wavelength
        self.z = distance
        self.k = 2 * np.pi / wavelength

    def fresnel_parameter(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate Fresnel parameter u = x * sqrt(2/(lambda*z))
        """
        return x * np.sqrt(2 / (self.wavelength * self.z))

    def straight_edge(self, x_obs: np.ndarray, edge_position: float = 0.0) -> np.ndarray:
        """
        Calculate Fresnel diffraction from a straight edge.

        Args:
            x_obs: Observation positions
            edge_position: Position of edge (illuminated for x > edge_position)

        Returns:
            Intensity at observation points
        """
        # Fresnel parameter for distance from edge
        u = self.fresnel_parameter(x_obs - edge_position)

        # Fresnel integrals
        S, C = fresnel(u)

        # Complex amplitude
        # U = (1+i)/2 * [(1/2 + C(u)) + i*(1/2 + S(u))]
        real_part = 0.5 + C
        imag_part = 0.5 + S

        # Intensity
        intensity = 0.5 * (real_part**2 + imag_part**2)

        return intensity

    def rectangular_aperture(self, x_obs: np.ndarray, y_obs: np.ndarray,
                            aperture_width: float, aperture_height: float) -> np.ndarray:
        """
        Calculate Fresnel diffraction from a rectangular aperture.

        The aperture is centered at origin with given width (x) and height (y).
        """
        # Calculate for each dimension separately (separable)
        half_w = aperture_width / 2
        half_h = aperture_height / 2

        # Fresnel parameters for aperture edges
        u1 = self.fresnel_parameter(x_obs - half_w)
        u2 = self.fresnel_parameter(x_obs + half_w)
        v1 = self.fresnel_parameter(y_obs - half_h)
        v2 = self.fresnel_parameter(y_obs + half_h)

        # Fresnel integrals
        S_u1, C_u1 = fresnel(u1)
        S_u2, C_u2 = fresnel(u2)
        S_v1, C_v1 = fresnel(v1)
        S_v2, C_v2 = fresnel(v2)

        # Complex amplitudes for each dimension
        Ux = (C_u2 - C_u1) + 1j * (S_u2 - S_u1)
        Uy = (C_v2 - C_v1) + 1j * (S_v2 - S_v1)

        # Total intensity (product of x and y contributions)
        intensity = 0.25 * np.abs(Ux * Uy)**2

        return intensity

    def circular_aperture(self, r_obs: np.ndarray, aperture_radius: float,
                         n_zones: int = 50) -> np.ndarray:
        """
        Calculate Fresnel diffraction from a circular aperture.

        Uses Lommel functions approximation.
        """
        # Fresnel number
        N_F = aperture_radius**2 / (self.wavelength * self.z)

        # Normalized observation radius
        rho = r_obs / aperture_radius

        # Number of Fresnel zones
        m = 2 * N_F

        intensity = np.zeros_like(r_obs)

        for i, r in enumerate(r_obs):
            if r < 1e-10:
                # On-axis intensity (uses Fresnel zone theory)
                # Alternating sum of zones
                I = 0.5 * (1 + np.cos(np.pi * m))
                intensity[i] = I**2
            else:
                # Off-axis - use numerical integration
                n_points = 200
                theta = np.linspace(0, 2*np.pi, n_points)
                rho_int = np.linspace(0, aperture_radius, 100)

                # Simplified: Use first few terms
                phase_factor = np.exp(1j * np.pi * N_F * (rho[i])**2)
                I = np.abs(phase_factor)**2 * self._bessel_pattern(r, aperture_radius, N_F)
                intensity[i] = I

        return intensity

    def _bessel_pattern(self, r: float, a: float, N_F: float) -> float:
        """Helper for circular aperture diffraction"""
        if r < a:
            # Inside geometric shadow
            return 1.0
        else:
            # Approximate oscillatory decay
            arg = 2 * np.pi * N_F * (r/a - 1)
            return 0.5 * (1 + np.cos(arg)) * np.exp(-0.5 * (r/a - 1))

    def zone_plate(self, x_obs: np.ndarray, y_obs: np.ndarray,
                   n_zones: int, focal_length: float) -> np.ndarray:
        """
        Calculate field from a Fresnel zone plate.

        Zone plate radii: r_n = sqrt(n * lambda * f)
        """
        r_obs = np.sqrt(x_obs**2 + y_obs**2)

        # Zone radii
        zone_radii = np.sqrt(np.arange(n_zones + 1) * self.wavelength * focal_length)

        # At the focal plane, constructive interference from all open zones
        intensity = np.zeros_like(r_obs)

        # Simple model: focused spot
        w = np.sqrt(self.wavelength * focal_length)  # Approximate spot size
        intensity = np.exp(-2 * r_obs**2 / w**2)

        return intensity


def plot_cornu_spiral():
    """Plot the Cornu spiral (clothoid) and relate to diffraction patterns"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Cornu spiral
    ax1 = axes[0, 0]

    u = np.linspace(-5, 5, 1000)
    S, C = fresnel(u)

    ax1.plot(C, S, 'b-', linewidth=2)

    # Mark special points
    special_u = [-3, -2, -1, 0, 1, 2, 3]
    for u_val in special_u:
        S_val, C_val = fresnel(np.array([u_val]))
        ax1.plot(C_val, S_val, 'ro', markersize=8)
        ax1.annotate(f'u={u_val}', xy=(C_val[0], S_val[0]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)

    # Mark asymptotic points
    ax1.plot(0.5, 0.5, 'g*', markersize=15, label='u -> +inf')
    ax1.plot(-0.5, -0.5, 'g*', markersize=15, label='u -> -inf')

    ax1.set_xlabel('C(u) - Fresnel cosine integral')
    ax1.set_ylabel('S(u) - Fresnel sine integral')
    ax1.set_title('Cornu Spiral (Clothoid)')
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)

    # Plot 2: Fresnel integrals
    ax2 = axes[0, 1]

    ax2.plot(u, C, 'b-', linewidth=2, label='C(u)')
    ax2.plot(u, S, 'r-', linewidth=2, label='S(u)')
    ax2.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(-0.5, color='gray', linestyle='--', alpha=0.5)

    ax2.set_xlabel('Fresnel parameter u')
    ax2.set_ylabel('Fresnel integral value')
    ax2.set_title('Fresnel Integrals C(u) and S(u)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: How Cornu spiral relates to diffraction
    ax3 = axes[1, 0]

    # Straight edge diffraction from Cornu spiral
    u_edge = np.linspace(-4, 4, 500)
    S_edge, C_edge = fresnel(u_edge)

    # Phasor from (-inf to u)
    # The phasor is from (-0.5, -0.5) to (C(u), S(u))
    phasor_real = C_edge + 0.5
    phasor_imag = S_edge + 0.5

    amplitude = np.sqrt(phasor_real**2 + phasor_imag**2)
    intensity = 0.5 * amplitude**2

    ax3.plot(u_edge, intensity, 'b-', linewidth=2)
    ax3.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Geometric optics limit')
    ax3.axhline(0.25, color='gray', linestyle=':', alpha=0.5, label='Edge intensity')
    ax3.axvline(0, color='red', linestyle='--', alpha=0.5, label='Edge position')

    ax3.set_xlabel('Fresnel parameter u (proportional to position)')
    ax3.set_ylabel('Normalized intensity')
    ax3.set_title('Straight Edge Diffraction from Cornu Spiral')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-4, 4)

    # Plot 4: Intensity oscillations in shadow region
    ax4 = axes[1, 1]

    u_fine = np.linspace(-1, 5, 1000)
    S_fine, C_fine = fresnel(u_fine)

    phasor_real = C_fine + 0.5
    phasor_imag = S_fine + 0.5
    intensity = 0.5 * (phasor_real**2 + phasor_imag**2)

    ax4.plot(u_fine, intensity, 'b-', linewidth=2)
    ax4.fill_between(u_fine, 0, intensity, alpha=0.3)

    # Mark maxima and minima
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(intensity)
    valleys, _ = find_peaks(-intensity)

    ax4.plot(u_fine[peaks], intensity[peaks], 'go', markersize=8, label='Local maxima')
    ax4.plot(u_fine[valleys], intensity[valleys], 'ro', markersize=8, label='Local minima')

    ax4.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax4.axvline(0, color='black', linestyle='--', alpha=0.5)

    ax4.set_xlabel('Fresnel parameter u')
    ax4.set_ylabel('Normalized intensity')
    ax4.set_title('Fresnel Fringes in Illuminated Region')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_straight_edge_diffraction():
    """Plot straight edge Fresnel diffraction at various distances"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    wavelength = 632.8e-9

    # Plot 1: Effect of distance
    ax1 = axes[0, 0]

    distances = [0.01, 0.05, 0.1, 0.5]  # 1cm to 50cm
    x_range = np.linspace(-0.5e-3, 1.5e-3, 500)

    colors = plt.cm.viridis(np.linspace(0, 1, len(distances)))

    for z, color in zip(distances, colors):
        fresnel_diff = FresnelDiffraction(wavelength, z)
        intensity = fresnel_diff.straight_edge(x_range, edge_position=0)

        ax1.plot(x_range * 1e3, intensity, color=color, linewidth=2,
                label=f'z = {z*100:.0f} cm')

    ax1.axvline(0, color='black', linestyle='--', alpha=0.5, label='Edge position')
    ax1.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
    ax1.axhline(0.25, color='gray', linestyle=':', alpha=0.5)

    ax1.set_xlabel('Position x (mm)')
    ax1.set_ylabel('Normalized intensity')
    ax1.set_title(f'Straight Edge Diffraction at Different Distances\nlambda = {wavelength*1e9:.1f} nm')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: 2D pattern evolution
    ax2 = axes[0, 1]

    n_distances = 100
    n_positions = 200
    distances_map = np.linspace(0.005, 0.3, n_distances)
    x_map = np.linspace(-0.5e-3, 1.5e-3, n_positions)

    intensity_map = np.zeros((n_distances, n_positions))

    for i, z in enumerate(distances_map):
        fresnel_diff = FresnelDiffraction(wavelength, z)
        intensity_map[i, :] = fresnel_diff.straight_edge(x_map, 0)

    im = ax2.imshow(intensity_map, extent=[x_map.min()*1e3, x_map.max()*1e3,
                                          distances_map.min()*100, distances_map.max()*100],
                   aspect='auto', cmap='hot', origin='lower')
    plt.colorbar(im, ax=ax2, label='Intensity')

    ax2.axvline(0, color='white', linestyle='--', alpha=0.5)

    ax2.set_xlabel('Position x (mm)')
    ax2.set_ylabel('Distance z (cm)')
    ax2.set_title('Straight Edge Diffraction Pattern Evolution')

    # Plot 3: Different wavelengths
    ax3 = axes[1, 0]

    z = 0.1  # 10 cm
    wavelengths = [450e-9, 532e-9, 632.8e-9, 780e-9]
    colors = ['blue', 'green', 'red', 'darkred']
    labels = ['Blue (450nm)', 'Green (532nm)', 'Red (633nm)', 'NIR (780nm)']

    for wl, color, label in zip(wavelengths, colors, labels):
        fresnel_diff = FresnelDiffraction(wl, z)
        intensity = fresnel_diff.straight_edge(x_range, 0)

        ax3.plot(x_range * 1e3, intensity, color=color, linewidth=2, label=label)

    ax3.axvline(0, color='black', linestyle='--', alpha=0.5)

    ax3.set_xlabel('Position x (mm)')
    ax3.set_ylabel('Normalized intensity')
    ax3.set_title(f'Straight Edge Diffraction: Effect of Wavelength\nz = {z*100:.0f} cm')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Intensity profile analysis
    ax4 = axes[1, 1]

    z = 0.05  # 5 cm
    x_fine = np.linspace(-0.2e-3, 1e-3, 1000)
    fresnel_diff = FresnelDiffraction(wavelength, z)
    intensity = fresnel_diff.straight_edge(x_fine, 0)

    ax4.plot(x_fine * 1e3, intensity, 'b-', linewidth=2)

    # Mark key features
    # Maximum at u ≈ 1.22
    u_max = 1.22
    x_max = u_max / np.sqrt(2 / (wavelength * z))
    I_max = fresnel_diff.straight_edge(np.array([x_max]), 0)[0]
    ax4.plot(x_max * 1e3, I_max, 'go', markersize=10, label=f'First max: I = {I_max:.3f}')

    # Intensity at edge
    I_edge = fresnel_diff.straight_edge(np.array([0]), 0)[0]
    ax4.plot(0, I_edge, 'ro', markersize=10, label=f'At edge: I = {I_edge:.3f}')

    ax4.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax4.axhline(0.25, color='orange', linestyle='--', alpha=0.5, label='I = 0.25')
    ax4.axvline(0, color='black', linestyle='--', alpha=0.5)

    ax4.set_xlabel('Position x (mm)')
    ax4.set_ylabel('Normalized intensity')
    ax4.set_title(f'Detailed Fresnel Edge Pattern\nlambda = {wavelength*1e9:.1f} nm, z = {z*100:.0f} cm')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-0.2, 1)

    plt.tight_layout()
    return fig


def plot_rectangular_aperture():
    """Plot Fresnel diffraction from rectangular aperture"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    wavelength = 632.8e-9
    z = 0.1  # 10 cm
    aperture_width = 500e-6  # 500 microns
    aperture_height = 500e-6

    fresnel_diff = FresnelDiffraction(wavelength, z)

    # Calculate Fresnel number
    N_F = aperture_width**2 / (wavelength * z)
    print(f"Fresnel number N_F = {N_F:.2f}")

    # Plot 1: 2D diffraction pattern
    ax1 = axes[0, 0]

    x = np.linspace(-1e-3, 1e-3, 200)
    y = np.linspace(-1e-3, 1e-3, 200)
    X, Y = np.meshgrid(x, y)

    intensity = fresnel_diff.rectangular_aperture(X, Y, aperture_width, aperture_height)

    im = ax1.imshow(intensity, extent=[x.min()*1e3, x.max()*1e3,
                                       y.min()*1e3, y.max()*1e3],
                   cmap='hot', origin='lower')
    plt.colorbar(im, ax=ax1, label='Intensity')

    # Mark aperture projection
    rect = plt.Rectangle((-aperture_width/2*1e3, -aperture_height/2*1e3),
                        aperture_width*1e3, aperture_height*1e3,
                        fill=False, edgecolor='cyan', linestyle='--', linewidth=2)
    ax1.add_patch(rect)

    ax1.set_xlabel('x (mm)')
    ax1.set_ylabel('y (mm)')
    ax1.set_title(f'Fresnel Diffraction: Square Aperture\n'
                 f'a = {aperture_width*1e6:.0f} um, z = {z*100:.0f} cm, N_F = {N_F:.2f}')

    # Plot 2: Cross-sections
    ax2 = axes[0, 1]

    x_cross = np.linspace(-1e-3, 1e-3, 500)
    intensity_x = fresnel_diff.rectangular_aperture(x_cross,
                                                    np.zeros_like(x_cross),
                                                    aperture_width, aperture_height)

    ax2.plot(x_cross * 1e3, intensity_x, 'b-', linewidth=2)
    ax2.axvline(-aperture_width/2*1e3, color='red', linestyle='--', alpha=0.5)
    ax2.axvline(aperture_width/2*1e3, color='red', linestyle='--', alpha=0.5)

    ax2.set_xlabel('Position x (mm)')
    ax2.set_ylabel('Normalized intensity')
    ax2.set_title('Cross-section along x-axis (y=0)')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Effect of distance (Fresnel number)
    ax3 = axes[1, 0]

    distances = [0.02, 0.05, 0.1, 0.5]  # Different distances

    for z_val in distances:
        fresnel_diff_temp = FresnelDiffraction(wavelength, z_val)
        N_F_temp = aperture_width**2 / (wavelength * z_val)

        intensity_temp = fresnel_diff_temp.rectangular_aperture(
            x_cross, np.zeros_like(x_cross), aperture_width, aperture_height)

        ax3.plot(x_cross * 1e3, intensity_temp, linewidth=2,
                label=f'z={z_val*100:.0f}cm, N_F={N_F_temp:.1f}')

    ax3.axvline(-aperture_width/2*1e3, color='black', linestyle='--', alpha=0.5)
    ax3.axvline(aperture_width/2*1e3, color='black', linestyle='--', alpha=0.5)

    ax3.set_xlabel('Position x (mm)')
    ax3.set_ylabel('Normalized intensity')
    ax3.set_title('Effect of Propagation Distance')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Rectangular aperture (non-square)
    ax4 = axes[1, 1]

    aspect_ratios = [(1, 1), (2, 1), (4, 1)]
    base_size = 300e-6

    for width_mult, height_mult in aspect_ratios:
        w = base_size * width_mult
        h = base_size * height_mult

        # Cross-section along x
        intensity_rect = fresnel_diff.rectangular_aperture(
            x_cross, np.zeros_like(x_cross), w, h)

        ax4.plot(x_cross * 1e3, intensity_rect, linewidth=2,
                label=f'w:h = {width_mult}:{height_mult}')

    ax4.set_xlabel('Position x (mm)')
    ax4.set_ylabel('Normalized intensity')
    ax4.set_title('Effect of Aperture Aspect Ratio')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_fresnel_zones():
    """Visualize Fresnel zones and zone plate focusing"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    wavelength = 632.8e-9
    z = 0.5  # 50 cm
    aperture_radius = 5e-3  # 5 mm

    # Plot 1: Fresnel zones schematic
    ax1 = axes[0, 0]

    # Zone radii: r_n = sqrt(n * lambda * z)
    n_zones = 20
    zone_radii = np.sqrt(np.arange(n_zones + 1) * wavelength * z)

    theta = np.linspace(0, 2*np.pi, 100)

    for n in range(n_zones):
        r_inner = zone_radii[n]
        r_outer = zone_radii[n + 1]

        # Draw zone
        color = 'white' if n % 2 == 0 else 'black'
        for r in np.linspace(r_inner, r_outer, 20):
            ax1.plot(r * np.cos(theta) * 1e3, r * np.sin(theta) * 1e3,
                    color=color, linewidth=0.5, alpha=0.7)

    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-5, 5)
    ax1.set_xlabel('x (mm)')
    ax1.set_ylabel('y (mm)')
    ax1.set_title(f'Fresnel Zones for On-Axis Point\nlambda = {wavelength*1e9:.1f} nm, z = {z*100:.0f} cm')
    ax1.set_aspect('equal')
    ax1.set_facecolor('gray')

    # Plot 2: Zone plate pattern
    ax2 = axes[1, 0]

    # Create zone plate
    x = np.linspace(-4e-3, 4e-3, 400)
    y = np.linspace(-4e-3, 4e-3, 400)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)

    # Zone plate transmission
    focal_length = 0.2  # 20 cm focal length
    zone_radii_plate = np.sqrt(np.arange(50) * wavelength * focal_length)

    transmission = np.zeros_like(R)
    for n in range(len(zone_radii_plate) - 1):
        mask = (R >= zone_radii_plate[n]) & (R < zone_radii_plate[n + 1])
        transmission[mask] = 1 if n % 2 == 0 else 0

    ax2.imshow(transmission, extent=[x.min()*1e3, x.max()*1e3,
                                     y.min()*1e3, y.max()*1e3],
              cmap='gray', origin='lower')

    ax2.set_xlabel('x (mm)')
    ax2.set_ylabel('y (mm)')
    ax2.set_title(f'Fresnel Zone Plate\nFocal length f = {focal_length*100:.0f} cm')
    ax2.set_aspect('equal')

    # Plot 3: On-axis intensity with aperture radius
    ax3 = axes[0, 1]

    radii = np.linspace(0.1e-3, 5e-3, 500)
    on_axis_intensity = []

    for R_ap in radii:
        # Number of Fresnel zones: N = R^2 / (lambda * z)
        N = R_ap**2 / (wavelength * z)

        # On-axis intensity oscillates with zone number
        # I ~ [1 + cos(pi * N)]^2 / 4
        I = (1 + np.cos(np.pi * N))**2 / 4
        on_axis_intensity.append(I)

    ax3.plot(radii * 1e3, on_axis_intensity, 'b-', linewidth=2)

    # Mark zone boundaries
    for n in range(1, 10):
        r_zone = np.sqrt(n * wavelength * z)
        if r_zone < radii.max():
            ax3.axvline(r_zone * 1e3, color='red', linestyle='--', alpha=0.3)
            ax3.text(r_zone * 1e3, 1.1, str(n), ha='center', fontsize=8)

    ax3.set_xlabel('Aperture radius (mm)')
    ax3.set_ylabel('On-axis intensity')
    ax3.set_title('On-Axis Intensity vs Aperture Radius\n(Maxima at odd zone numbers)')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.2)

    # Plot 4: Zone plate focusing
    ax4 = axes[1, 1]

    # Intensity along optical axis for zone plate
    z_range = np.linspace(0.1, 0.5, 200)

    # Primary focus at f
    intensity_axis = []
    zone_plate_radius = 3e-3  # 3 mm

    for z_obs in z_range:
        # Zone plate focuses at f = r1^2/lambda
        # Intensity peaks at f, f/3, f/5, ...
        r1 = np.sqrt(wavelength * focal_length)

        # Simplified model: Gaussian-like peaks at focal points
        I = 0
        for m in [1, 3, 5]:  # Odd harmonic foci
            f_m = focal_length / m
            width = 0.02 * focal_length / m
            I += np.exp(-((z_obs - f_m) / width)**2) / m**2

        intensity_axis.append(I)

    ax4.plot(z_range * 100, intensity_axis, 'b-', linewidth=2)

    # Mark focal points
    for m in [1, 3, 5]:
        f_m = focal_length / m
        ax4.axvline(f_m * 100, color='red', linestyle='--', alpha=0.5)
        ax4.text(f_m * 100, 0.9 / m**2, f'f/{m}', ha='center', fontsize=10)

    ax4.set_xlabel('Distance z (cm)')
    ax4.set_ylabel('On-axis intensity (arb. units)')
    ax4.set_title('Zone Plate: Intensity Along Optical Axis\nMultiple focal points at f, f/3, f/5, ...')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    """Main function to demonstrate Fresnel diffraction"""

    # Create figures
    fig1 = plot_cornu_spiral()
    fig2 = plot_straight_edge_diffraction()
    fig3 = plot_rectangular_aperture()
    fig4 = plot_fresnel_zones()

    # Save plots
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)

    fig1.savefig(os.path.join(output_dir, 'fresnel_cornu_spiral.png'),
                 dpi=150, bbox_inches='tight')
    fig2.savefig(os.path.join(output_dir, 'fresnel_straight_edge.png'),
                 dpi=150, bbox_inches='tight')
    fig3.savefig(os.path.join(output_dir, 'fresnel_rectangular.png'),
                 dpi=150, bbox_inches='tight')
    fig4.savefig(os.path.join(output_dir, 'fresnel_zones.png'),
                 dpi=150, bbox_inches='tight')

    print(f"Plots saved to {output_dir}/fresnel_*.png")

    # Print analysis
    print("\n=== Fresnel Diffraction Analysis ===")
    print("\nKey concepts:")
    print("- Fresnel number N_F = a^2/(lambda*z) characterizes diffraction regime")
    print("- N_F >> 1: Near-field (Fresnel) - many zones across aperture")
    print("- N_F << 1: Far-field (Fraunhofer) - less than one zone")
    print("\nCornu spiral:")
    print("- Parametric curve (C(u), S(u)) where C, S are Fresnel integrals")
    print("- Straight edge diffraction from phasor to asymptotic point")
    print("- Edge intensity = 1/4 of unobstructed intensity")


if __name__ == "__main__":
    main()
