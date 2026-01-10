"""
Example 114: Brewster's Angle and Fresnel Coefficients

This example demonstrates Brewster's angle and the Fresnel equations for
reflection and transmission at a dielectric interface.

Physics:
    Brewster's angle: tan(theta_B) = n2/n1
    At this angle, p-polarized light has zero reflection.

    Fresnel coefficients for amplitude reflection:
    r_s = (n1*cos(theta_i) - n2*cos(theta_t)) / (n1*cos(theta_i) + n2*cos(theta_t))
    r_p = (n2*cos(theta_i) - n1*cos(theta_t)) / (n2*cos(theta_i) + n1*cos(theta_t))

    Fresnel coefficients for amplitude transmission:
    t_s = 2*n1*cos(theta_i) / (n1*cos(theta_i) + n2*cos(theta_t))
    t_p = 2*n1*cos(theta_i) / (n2*cos(theta_i) + n1*cos(theta_t))

    Reflectivity (power): R = |r|^2
    Transmissivity (power): T = (n2*cos(theta_t))/(n1*cos(theta_i)) * |t|^2

    At Brewster's angle:
    - theta_i + theta_t = 90 degrees
    - Reflected and refracted rays are perpendicular
    - Only s-polarized light is reflected
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


class FresnelCoefficients:
    """Calculate Fresnel reflection and transmission coefficients"""

    def __init__(self, n1: float, n2: float):
        """
        Args:
            n1: Refractive index of incident medium
            n2: Refractive index of transmitted medium
        """
        self.n1 = n1
        self.n2 = n2

    def brewster_angle(self) -> float:
        """Calculate Brewster's angle in radians"""
        return np.arctan(self.n2 / self.n1)

    def critical_angle(self) -> float:
        """Calculate critical angle for total internal reflection"""
        if self.n1 > self.n2:
            return np.arcsin(self.n2 / self.n1)
        else:
            return np.pi / 2  # No TIR possible

    def snell_angle(self, theta_i: np.ndarray) -> np.ndarray:
        """Calculate refracted angle using Snell's law"""
        sin_theta_t = self.n1 * np.sin(theta_i) / self.n2
        # Handle total internal reflection
        sin_theta_t = np.clip(sin_theta_t, -1, 1)
        return np.arcsin(sin_theta_t)

    def r_s(self, theta_i: np.ndarray) -> np.ndarray:
        """S-polarization amplitude reflection coefficient"""
        theta_t = self.snell_angle(theta_i)

        numerator = self.n1 * np.cos(theta_i) - self.n2 * np.cos(theta_t)
        denominator = self.n1 * np.cos(theta_i) + self.n2 * np.cos(theta_t)

        return numerator / denominator

    def r_p(self, theta_i: np.ndarray) -> np.ndarray:
        """P-polarization amplitude reflection coefficient"""
        theta_t = self.snell_angle(theta_i)

        numerator = self.n2 * np.cos(theta_i) - self.n1 * np.cos(theta_t)
        denominator = self.n2 * np.cos(theta_i) + self.n1 * np.cos(theta_t)

        return numerator / denominator

    def t_s(self, theta_i: np.ndarray) -> np.ndarray:
        """S-polarization amplitude transmission coefficient"""
        theta_t = self.snell_angle(theta_i)

        numerator = 2 * self.n1 * np.cos(theta_i)
        denominator = self.n1 * np.cos(theta_i) + self.n2 * np.cos(theta_t)

        return numerator / denominator

    def t_p(self, theta_i: np.ndarray) -> np.ndarray:
        """P-polarization amplitude transmission coefficient"""
        theta_t = self.snell_angle(theta_i)

        numerator = 2 * self.n1 * np.cos(theta_i)
        denominator = self.n2 * np.cos(theta_i) + self.n1 * np.cos(theta_t)

        return numerator / denominator

    def R_s(self, theta_i: np.ndarray) -> np.ndarray:
        """S-polarization power reflectivity"""
        return np.abs(self.r_s(theta_i))**2

    def R_p(self, theta_i: np.ndarray) -> np.ndarray:
        """P-polarization power reflectivity"""
        return np.abs(self.r_p(theta_i))**2

    def T_s(self, theta_i: np.ndarray) -> np.ndarray:
        """S-polarization power transmissivity"""
        theta_t = self.snell_angle(theta_i)
        factor = (self.n2 * np.cos(theta_t)) / (self.n1 * np.cos(theta_i))
        return factor * np.abs(self.t_s(theta_i))**2

    def T_p(self, theta_i: np.ndarray) -> np.ndarray:
        """P-polarization power transmissivity"""
        theta_t = self.snell_angle(theta_i)
        factor = (self.n2 * np.cos(theta_t)) / (self.n1 * np.cos(theta_i))
        return factor * np.abs(self.t_p(theta_i))**2

    def R_unpolarized(self, theta_i: np.ndarray) -> np.ndarray:
        """Unpolarized light reflectivity (average of s and p)"""
        return 0.5 * (self.R_s(theta_i) + self.R_p(theta_i))


def plot_fresnel_reflectivity():
    """Plot Fresnel reflectivity for various interfaces"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Air-glass interface (external reflection)
    ax1 = axes[0, 0]

    n1, n2 = 1.0, 1.5  # Air to glass
    fc = FresnelCoefficients(n1, n2)

    theta_i = np.linspace(0, np.pi/2 - 0.001, 500)
    theta_deg = np.degrees(theta_i)

    R_s = fc.R_s(theta_i)
    R_p = fc.R_p(theta_i)
    R_avg = fc.R_unpolarized(theta_i)

    ax1.plot(theta_deg, R_s, 'b-', linewidth=2, label='R_s (s-polarized)')
    ax1.plot(theta_deg, R_p, 'r-', linewidth=2, label='R_p (p-polarized)')
    ax1.plot(theta_deg, R_avg, 'k--', linewidth=2, label='Unpolarized (avg)')

    # Mark Brewster's angle
    theta_B = fc.brewster_angle()
    ax1.axvline(np.degrees(theta_B), color='green', linestyle='--', alpha=0.7)
    ax1.plot(np.degrees(theta_B), fc.R_p(np.array([theta_B]))[0], 'go', markersize=10)
    ax1.text(np.degrees(theta_B) + 2, 0.1, f"Brewster's angle\n= {np.degrees(theta_B):.1f} deg",
             fontsize=10, color='green')

    ax1.set_xlabel('Angle of incidence (degrees)')
    ax1.set_ylabel('Reflectivity R')
    ax1.set_title(f'External Reflection: Air (n={n1}) to Glass (n={n2})\n'
                  r"At Brewster's angle: R_p = 0")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 90)
    ax1.set_ylim(0, 1)

    # Plot 2: Glass-air interface (internal reflection with TIR)
    ax2 = axes[0, 1]

    n1, n2 = 1.5, 1.0  # Glass to air
    fc = FresnelCoefficients(n1, n2)

    theta_i = np.linspace(0, np.pi/2 - 0.001, 500)
    theta_deg = np.degrees(theta_i)

    R_s = fc.R_s(theta_i)
    R_p = fc.R_p(theta_i)

    ax2.plot(theta_deg, R_s, 'b-', linewidth=2, label='R_s')
    ax2.plot(theta_deg, R_p, 'r-', linewidth=2, label='R_p')

    # Mark Brewster's angle
    theta_B = fc.brewster_angle()
    ax2.axvline(np.degrees(theta_B), color='green', linestyle='--', alpha=0.7)
    ax2.text(np.degrees(theta_B) + 2, 0.5, f"Brewster's\n{np.degrees(theta_B):.1f} deg",
             fontsize=9, color='green')

    # Mark critical angle
    theta_c = fc.critical_angle()
    ax2.axvline(np.degrees(theta_c), color='purple', linestyle='--', alpha=0.7)
    ax2.fill_betweenx([0, 1], np.degrees(theta_c), 90, alpha=0.2, color='purple')
    ax2.text(np.degrees(theta_c) + 2, 0.3, f'Critical\n{np.degrees(theta_c):.1f} deg\n(TIR)',
             fontsize=9, color='purple')

    ax2.set_xlabel('Angle of incidence (degrees)')
    ax2.set_ylabel('Reflectivity R')
    ax2.set_title(f'Internal Reflection: Glass (n={n1}) to Air (n={n2})\n'
                  'Total internal reflection beyond critical angle')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 90)
    ax2.set_ylim(0, 1.05)

    # Plot 3: Different materials
    ax3 = axes[1, 0]

    materials = [
        ('Water', 1.33),
        ('Glass (crown)', 1.52),
        ('Glass (flint)', 1.66),
        ('Diamond', 2.42),
    ]

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(materials)))

    for (name, n2), color in zip(materials, colors):
        fc = FresnelCoefficients(1.0, n2)
        R_p = fc.R_p(theta_i)
        theta_B = fc.brewster_angle()

        ax3.plot(theta_deg, R_p, color=color, linewidth=2,
                 label=f'{name} (n={n2:.2f}, theta_B={np.degrees(theta_B):.1f} deg)')
        ax3.plot(np.degrees(theta_B), 0, 'o', color=color, markersize=8)

    ax3.set_xlabel('Angle of incidence (degrees)')
    ax3.set_ylabel('P-polarization reflectivity R_p')
    ax3.set_title("Brewster's Angle for Various Materials\n"
                  'Higher index -> Larger Brewster angle')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 90)
    ax3.set_ylim(0, 0.5)

    # Plot 4: Brewster angle vs refractive index
    ax4 = axes[1, 1]

    n2_range = np.linspace(1.01, 4.0, 100)
    theta_B_range = np.degrees(np.arctan(n2_range))

    ax4.plot(n2_range, theta_B_range, 'b-', linewidth=2)

    # Mark common materials
    for name, n in materials:
        theta_B = np.degrees(np.arctan(n))
        ax4.plot(n, theta_B, 'ro', markersize=8)
        ax4.annotate(name, xy=(n, theta_B), xytext=(5, -10),
                    textcoords='offset points', fontsize=9)

    ax4.set_xlabel('Refractive index n2 (n1 = 1.0)')
    ax4.set_ylabel("Brewster's angle (degrees)")
    ax4.set_title("Brewster's Angle: tan(theta_B) = n2/n1")
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(1, 4)
    ax4.set_ylim(45, 80)

    plt.tight_layout()
    return fig


def plot_fresnel_amplitude_coefficients():
    """Plot Fresnel amplitude coefficients showing phase changes"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    n1, n2 = 1.0, 1.5
    fc = FresnelCoefficients(n1, n2)

    theta_i = np.linspace(0, np.pi/2 - 0.001, 500)
    theta_deg = np.degrees(theta_i)

    # Plot 1: Amplitude reflection coefficients
    ax1 = axes[0, 0]

    r_s = fc.r_s(theta_i)
    r_p = fc.r_p(theta_i)

    ax1.plot(theta_deg, r_s, 'b-', linewidth=2, label='r_s')
    ax1.plot(theta_deg, r_p, 'r-', linewidth=2, label='r_p')
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)

    # Mark Brewster's angle
    theta_B = fc.brewster_angle()
    ax1.axvline(np.degrees(theta_B), color='green', linestyle='--', alpha=0.7)
    ax1.plot(np.degrees(theta_B), 0, 'go', markersize=10)

    ax1.set_xlabel('Angle of incidence (degrees)')
    ax1.set_ylabel('Amplitude reflection coefficient r')
    ax1.set_title('Fresnel Amplitude Reflection Coefficients\n'
                  f'Air (n={n1}) to Glass (n={n2})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 90)

    # Annotate phase change
    ax1.annotate('r_s < 0:\npi phase shift', xy=(45, -0.3), fontsize=10,
                ha='center', color='blue')
    ax1.annotate('r_p sign change\nat Brewster angle', xy=(np.degrees(theta_B), 0),
                xytext=(np.degrees(theta_B)+10, 0.2),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')

    # Plot 2: Amplitude transmission coefficients
    ax2 = axes[0, 1]

    t_s = fc.t_s(theta_i)
    t_p = fc.t_p(theta_i)

    ax2.plot(theta_deg, t_s, 'b-', linewidth=2, label='t_s')
    ax2.plot(theta_deg, t_p, 'r-', linewidth=2, label='t_p')

    ax2.axvline(np.degrees(theta_B), color='green', linestyle='--', alpha=0.7,
                label=f"Brewster's angle = {np.degrees(theta_B):.1f} deg")

    ax2.set_xlabel('Angle of incidence (degrees)')
    ax2.set_ylabel('Amplitude transmission coefficient t')
    ax2.set_title('Fresnel Amplitude Transmission Coefficients')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 90)

    # Plot 3: Phase of reflection coefficients
    ax3 = axes[1, 0]

    phase_s = np.angle(r_s)
    phase_p = np.angle(r_p)

    ax3.plot(theta_deg, np.degrees(phase_s), 'b-', linewidth=2, label='Phase of r_s')
    ax3.plot(theta_deg, np.degrees(phase_p), 'r-', linewidth=2, label='Phase of r_p')

    ax3.axvline(np.degrees(theta_B), color='green', linestyle='--', alpha=0.7)
    ax3.axhline(180, color='gray', linestyle=':', alpha=0.5)
    ax3.axhline(0, color='gray', linestyle=':', alpha=0.5)

    ax3.set_xlabel('Angle of incidence (degrees)')
    ax3.set_ylabel('Phase (degrees)')
    ax3.set_title('Phase of Reflection Coefficients\n'
                  'Phase jump at Brewster angle for p-polarization')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 90)
    ax3.set_ylim(-10, 190)

    # Plot 4: Energy conservation R + T = 1
    ax4 = axes[1, 1]

    R_s = fc.R_s(theta_i)
    R_p = fc.R_p(theta_i)
    T_s = fc.T_s(theta_i)
    T_p = fc.T_p(theta_i)

    ax4.plot(theta_deg, R_s + T_s, 'b-', linewidth=2, label='R_s + T_s')
    ax4.plot(theta_deg, R_p + T_p, 'r--', linewidth=2, label='R_p + T_p')
    ax4.axhline(1.0, color='gray', linestyle=':', alpha=0.7)

    ax4.set_xlabel('Angle of incidence (degrees)')
    ax4.set_ylabel('R + T')
    ax4.set_title('Energy Conservation: R + T = 1')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 90)
    ax4.set_ylim(0.95, 1.05)

    plt.tight_layout()
    return fig


def plot_polarization_effects():
    """Demonstrate polarization effects at Brewster's angle"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    n1, n2 = 1.0, 1.5
    fc = FresnelCoefficients(n1, n2)
    theta_B = fc.brewster_angle()

    theta_i = np.linspace(0, np.pi/2 - 0.001, 500)
    theta_deg = np.degrees(theta_i)

    # Plot 1: Degree of polarization
    ax1 = axes[0, 0]

    R_s = fc.R_s(theta_i)
    R_p = fc.R_p(theta_i)

    # Degree of polarization for reflected light from unpolarized incident
    # DOP = (R_s - R_p) / (R_s + R_p)
    with np.errstate(divide='ignore', invalid='ignore'):
        DOP = (R_s - R_p) / (R_s + R_p)
        DOP = np.nan_to_num(DOP, nan=0.0)

    ax1.plot(theta_deg, DOP, 'b-', linewidth=2)
    ax1.axvline(np.degrees(theta_B), color='green', linestyle='--', alpha=0.7)
    ax1.axhline(1.0, color='red', linestyle=':', alpha=0.5)

    ax1.set_xlabel('Angle of incidence (degrees)')
    ax1.set_ylabel('Degree of polarization (s-preference)')
    ax1.set_title('Degree of Polarization of Reflected Light\n'
                  "100% s-polarized at Brewster's angle")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 90)
    ax1.set_ylim(0, 1.1)

    # Mark Brewster's angle
    ax1.annotate(f"Brewster's angle\n{np.degrees(theta_B):.1f} deg\nDOP = 100%",
                xy=(np.degrees(theta_B), 1.0), xytext=(np.degrees(theta_B)-15, 0.7),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10, color='green')

    # Plot 2: Stacked reflectivity showing polarization
    ax2 = axes[0, 1]

    ax2.fill_between(theta_deg, 0, R_p, alpha=0.5, color='red', label='R_p')
    ax2.fill_between(theta_deg, R_p, R_p + R_s, alpha=0.5, color='blue', label='R_s')
    ax2.plot(theta_deg, R_s + R_p, 'k-', linewidth=2, label='Total R (unpolarized)')

    ax2.axvline(np.degrees(theta_B), color='green', linestyle='--', linewidth=2)

    ax2.set_xlabel('Angle of incidence (degrees)')
    ax2.set_ylabel('Reflectivity')
    ax2.set_title("At Brewster's angle: Only s-polarized light reflects\n"
                  'Reflected light is linearly polarized')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 90)
    ax2.set_ylim(0, 1)

    # Plot 3: Polarization angle of reflected light
    ax3 = axes[1, 0]

    # If incident light has arbitrary polarization angle alpha
    # Reflected light polarization depends on r_s/r_p ratio
    alpha_incident = np.linspace(0, 90, 5)  # Polarization angles

    for alpha in alpha_incident:
        alpha_rad = np.radians(alpha)

        # E_s and E_p components
        E_s_inc = np.cos(alpha_rad)
        E_p_inc = np.sin(alpha_rad)

        # Reflected components
        r_s = fc.r_s(theta_i)
        r_p = fc.r_p(theta_i)

        E_s_ref = E_s_inc * r_s
        E_p_ref = E_p_inc * r_p

        # Polarization angle of reflected light
        alpha_ref = np.degrees(np.arctan2(np.abs(E_p_ref), np.abs(E_s_ref)))

        ax3.plot(theta_deg, alpha_ref, linewidth=2,
                 label=f'Incident: {alpha:.0f} deg')

    ax3.axvline(np.degrees(theta_B), color='green', linestyle='--', alpha=0.7)
    ax3.axhline(0, color='gray', linestyle=':', alpha=0.5)

    ax3.set_xlabel('Angle of incidence (degrees)')
    ax3.set_ylabel('Reflected polarization angle (degrees)')
    ax3.set_title('Polarization Angle Rotation on Reflection\n'
                  "All become s-polarized (0 deg) at Brewster's angle")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 90)
    ax3.set_ylim(-5, 95)

    # Plot 4: Multiple reflections (Brewster windows)
    ax4 = axes[1, 1]

    # For N parallel plates at Brewster's angle
    n_reflections = np.arange(1, 11)

    # At each surface, s-polarized light is partially reflected
    R_s_B = fc.R_s(np.array([theta_B]))[0]

    # After N surfaces (2N interfaces for N plates)
    T_s_N = (1 - R_s_B)**(2 * n_reflections)
    T_p_N = np.ones_like(n_reflections, dtype=float)  # p-pol passes perfectly

    ax4.plot(n_reflections, T_s_N * 100, 'b-o', linewidth=2, markersize=8,
             label='S-polarized transmission')
    ax4.plot(n_reflections, T_p_N * 100, 'r-s', linewidth=2, markersize=8,
             label='P-polarized transmission')

    ax4.set_xlabel('Number of Brewster plates')
    ax4.set_ylabel('Transmission (%)')
    ax4.set_title("Brewster's Angle Polarizer (Pile of Plates)\n"
                  f"R_s per surface = {R_s_B*100:.1f}%")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0.5, 10.5)
    ax4.set_ylim(0, 105)

    plt.tight_layout()
    return fig


def plot_physical_interpretation():
    """Physical interpretation of Brewster's angle"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    n1, n2 = 1.0, 1.5
    fc = FresnelCoefficients(n1, n2)
    theta_B = fc.brewster_angle()

    # Plot 1: Ray diagram at Brewster's angle
    ax1 = axes[0, 0]

    # Interface at y = 0
    ax1.axhline(0, color='black', linewidth=2)
    ax1.fill_between([-2, 2], [-1.5, -1.5], [0, 0], alpha=0.2, color='blue')

    # Incident ray
    x_inc = np.array([-1.5, 0])
    y_inc = np.array([1.5 * np.tan(theta_B), 0])
    ax1.plot(x_inc, y_inc, 'g-', linewidth=3, label='Incident ray')
    ax1.annotate('', xy=(0, 0), xytext=(-0.3, 0.3*np.tan(theta_B)),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))

    # Reflected ray (only for s-polarization)
    x_ref = np.array([0, 1.5])
    y_ref = np.array([0, 1.5 * np.tan(theta_B)])
    ax1.plot(x_ref, y_ref, 'b--', linewidth=2, alpha=0.5, label='Reflected ray (s only)')

    # Refracted ray
    theta_t = fc.snell_angle(np.array([theta_B]))[0]
    x_trans = np.array([0, 1.5])
    y_trans = np.array([0, -1.5 * np.tan(theta_t)])
    ax1.plot(x_trans, y_trans, 'r-', linewidth=3, label='Refracted ray')
    ax1.annotate('', xy=(0.3, -0.3*np.tan(theta_t)), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    # Mark angles
    arc1 = np.linspace(np.pi/2, np.pi/2 + theta_B, 50)
    ax1.plot(0.3*np.cos(arc1), 0.3*np.sin(arc1), 'g-', linewidth=1.5)
    ax1.text(0.1, 0.4, f'theta_B = {np.degrees(theta_B):.1f} deg', fontsize=10, color='green')

    arc2 = np.linspace(-np.pi/2, -np.pi/2 + theta_t, 50)
    ax1.plot(0.4*np.cos(arc2), 0.4*np.sin(arc2), 'r-', linewidth=1.5)
    ax1.text(0.2, -0.4, f'theta_t = {np.degrees(theta_t):.1f} deg', fontsize=10, color='red')

    # Show that reflected and refracted are perpendicular
    ax1.annotate('', xy=(0.8, 0.8*np.tan(theta_B)), xytext=(0.8, -0.8*np.tan(theta_t)),
                arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
    ax1.text(1.0, 0.3, '90 deg', fontsize=10, color='purple')

    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.2, 1.2)
    ax1.set_aspect('equal')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(f"Ray Diagram at Brewster's Angle\n"
                  f"theta_i + theta_t = 90 deg")
    ax1.legend(loc='upper right', fontsize=9)
    ax1.text(-1.3, 0.5, f'n = {n1}', fontsize=12)
    ax1.text(-1.3, -0.5, f'n = {n2}', fontsize=12)

    # Plot 2: Oscillating dipoles explanation
    ax2 = axes[0, 1]

    # Draw interface
    ax2.axhline(0, color='black', linewidth=2)
    ax2.fill_between([-2, 2], [-1.5, -1.5], [0, 0], alpha=0.2, color='blue')

    # Draw oscillating dipoles in the material
    dipole_positions = [(0.3, -0.2), (0.6, -0.4), (0.9, -0.6)]

    for x_d, y_d in dipole_positions:
        # Dipole oscillates along the refracted ray direction
        dx = 0.15 * np.cos(theta_t)
        dy = 0.15 * np.sin(theta_t)

        ax2.annotate('', xy=(x_d+dx, y_d-dy), xytext=(x_d-dx, y_d+dy),
                    arrowprops=dict(arrowstyle='<->', color='red', lw=2))

    # Show dipole radiation pattern (perpendicular to oscillation)
    ax2.annotate('No radiation\nalong dipole axis', xy=(0.3, -0.2),
                xytext=(1.0, 0.5), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
                color='blue')

    # Refracted ray direction
    ax2.annotate('', xy=(1.2, -0.8), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='orange', lw=2))
    ax2.text(0.7, -0.2, 'Dipoles oscillate\nalong this direction', fontsize=9)

    ax2.set_xlim(-0.5, 1.5)
    ax2.set_ylim(-1, 1)
    ax2.set_aspect('equal')
    ax2.set_title('Physical Explanation: Dipole Radiation\n'
                  'p-polarized dipoles cannot radiate along reflection direction')
    ax2.axis('off')

    # Plot 3: theta_i + theta_t = 90 deg verification
    ax3 = axes[1, 0]

    theta_i = np.linspace(0, np.pi/2 - 0.001, 500)
    theta_deg = np.degrees(theta_i)

    theta_t = fc.snell_angle(theta_i)
    sum_angles = theta_i + theta_t

    ax3.plot(theta_deg, np.degrees(sum_angles), 'b-', linewidth=2)
    ax3.axhline(90, color='red', linestyle='--', alpha=0.7)
    ax3.axvline(np.degrees(theta_B), color='green', linestyle='--', alpha=0.7)

    ax3.plot(np.degrees(theta_B), 90, 'go', markersize=12)
    ax3.annotate(f"Brewster's angle:\ntheta_i + theta_t = 90 deg",
                xy=(np.degrees(theta_B), 90), xytext=(30, 70),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10, color='green')

    ax3.set_xlabel('Angle of incidence (degrees)')
    ax3.set_ylabel('theta_i + theta_t (degrees)')
    ax3.set_title("Verification: theta_i + theta_t = 90 deg at Brewster's angle")
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 90)
    ax3.set_ylim(60, 110)

    # Plot 4: Applications
    ax4 = axes[1, 1]

    applications = [
        "Laser cavities\n(low-loss windows)",
        "Polarizers\n(pile of plates)",
        "Photography\n(polarizing filters)",
        "Optical\ninstruments",
    ]

    y_pos = np.arange(len(applications))
    ax4.barh(y_pos, [4, 3, 2, 1], color=['red', 'blue', 'green', 'orange'], alpha=0.7)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(applications, fontsize=11)
    ax4.set_xlabel('Relative importance')
    ax4.set_title("Applications of Brewster's Angle")

    # Add notes
    notes = """
Key Points:
- tan(theta_B) = n2/n1
- At Brewster's angle, reflected and
  refracted rays are perpendicular
- Only s-polarized light is reflected
- Used in lasers for minimal loss
"""
    ax4.text(2, -0.8, notes, fontsize=10, family='monospace',
            verticalalignment='top')

    plt.tight_layout()
    return fig


def main():
    """Main function to demonstrate Brewster's angle"""

    # Create figures
    fig1 = plot_fresnel_reflectivity()
    fig2 = plot_fresnel_amplitude_coefficients()
    fig3 = plot_polarization_effects()
    fig4 = plot_physical_interpretation()

    # Save plots
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)

    fig1.savefig(os.path.join(output_dir, 'brewster_angle.png'),
                 dpi=150, bbox_inches='tight')
    fig2.savefig(os.path.join(output_dir, 'fresnel_coefficients.png'),
                 dpi=150, bbox_inches='tight')
    fig3.savefig(os.path.join(output_dir, 'brewster_polarization.png'),
                 dpi=150, bbox_inches='tight')
    fig4.savefig(os.path.join(output_dir, 'brewster_interpretation.png'),
                 dpi=150, bbox_inches='tight')

    print(f"Plots saved to {output_dir}/brewster_*.png and fresnel_coefficients.png")

    # Print analysis
    print("\n=== Brewster's Angle Analysis ===")

    materials = [
        ('Air-Water', 1.0, 1.33),
        ('Air-Crown Glass', 1.0, 1.52),
        ('Air-Flint Glass', 1.0, 1.66),
        ('Air-Diamond', 1.0, 2.42),
        ('Glass-Air', 1.52, 1.0),
    ]

    print("\nBrewster's angles for common interfaces:")
    print("-" * 50)

    for name, n1, n2 in materials:
        fc = FresnelCoefficients(n1, n2)
        theta_B = fc.brewster_angle()

        print(f"{name}: theta_B = {np.degrees(theta_B):.2f} deg")

        if n1 > n2:
            theta_c = fc.critical_angle()
            print(f"  Critical angle: theta_c = {np.degrees(theta_c):.2f} deg")

    # Fresnel coefficients at various angles
    print("\n\nFresnel coefficients for Air-Glass (n=1.5):")
    print("-" * 50)

    fc = FresnelCoefficients(1.0, 1.5)
    angles = [0, 30, 45, 56.31, 60, 80]  # 56.31 is Brewster's angle

    print(f"{'Angle':>8} {'r_s':>10} {'r_p':>10} {'R_s':>10} {'R_p':>10}")
    print("-" * 50)

    for angle in angles:
        theta = np.radians(angle)
        r_s = fc.r_s(np.array([theta]))[0]
        r_p = fc.r_p(np.array([theta]))[0]
        R_s = fc.R_s(np.array([theta]))[0]
        R_p = fc.R_p(np.array([theta]))[0]

        marker = " *" if abs(angle - 56.31) < 0.1 else ""
        print(f"{angle:>8.1f} {r_s:>10.4f} {r_p:>10.4f} {R_s:>10.4f} {R_p:>10.4f}{marker}")

    print("\n* = Brewster's angle")


if __name__ == "__main__":
    main()
