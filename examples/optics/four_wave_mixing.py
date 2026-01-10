"""
Example 117: Four-Wave Mixing and Phase Conjugation

This example demonstrates four-wave mixing (FWM), a third-order nonlinear optical
process used for phase conjugation and optical signal processing.

Physics:
    Four-wave mixing involves the interaction of three waves to generate a fourth:
    omega_4 = omega_1 + omega_2 - omega_3 (degenerate: omega_1 = omega_2 = omega_3)

    Third-order nonlinear polarization:
    P^(3) = epsilon_0 * chi^(3) * E1 * E2 * E3*

    For degenerate FWM (phase conjugation):
    - Two counter-propagating pump beams (E_f, E_b)
    - Signal beam (E_s)
    - Generates phase-conjugate beam (E_c)

    Phase matching condition:
    k_4 = k_1 + k_2 - k_3

    The phase conjugate beam has:
    - Reversed wavevector: k_c = -k_s
    - Conjugate phase: phi_c = -phi_s
    - Time-reversed propagation

    Applications:
    - Aberration correction
    - Real-time holography
    - Optical phase conjugation
    - Signal processing
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


class FourWaveMixing:
    """Four-wave mixing in a chi^(3) nonlinear medium"""

    def __init__(
        self,
        chi3: float,
        wavelength: float,
        n0: float,
        length: float
    ):
        """
        Args:
            chi3: Third-order susceptibility (m^2/V^2)
            wavelength: Wavelength (m)
            n0: Linear refractive index
            length: Medium length (m)
        """
        self.chi3 = chi3
        self.wavelength = wavelength
        self.n0 = n0
        self.L = length
        self.k = 2 * np.pi * n0 / wavelength
        self.c = 3e8

    def coupling_coefficient(self, pump_intensity: float) -> float:
        """
        Calculate the nonlinear coupling coefficient kappa.

        kappa = (3 * omega * chi^(3) * I_pump) / (4 * n0^2 * c)
        """
        omega = 2 * np.pi * self.c / self.wavelength
        epsilon_0 = 8.854e-12

        # Effective nonlinear coefficient
        kappa = (3 * omega * self.chi3 * pump_intensity) / (4 * self.n0**2 * self.c)

        return kappa

    def phase_conjugate_reflectivity(
        self,
        pump_intensity: float,
        delta_k: float = 0.0
    ) -> float:
        """
        Calculate phase conjugate reflectivity for given pump intensity.

        R = tanh^2(kappa * L) for perfect phase matching
        """
        kappa = self.coupling_coefficient(pump_intensity)

        # Phase mismatch reduces efficiency
        if abs(delta_k) < 1e-10:
            # Perfect phase matching
            return np.tanh(kappa * self.L)**2
        else:
            # With phase mismatch
            gamma = np.sqrt(kappa**2 - (delta_k/2)**2 + 0j)
            if np.real(gamma) > 0:
                r = kappa * np.sinh(gamma * self.L) / \
                    (gamma * np.cosh(gamma * self.L) + 1j * delta_k/2 * np.sinh(gamma * self.L))
                return np.abs(r)**2
            else:
                return 0.0

    def propagate_fields(
        self,
        z_range: np.ndarray,
        E_pump_f: complex,
        E_pump_b: complex,
        E_signal: complex
    ) -> tuple:
        """
        Propagate fields through the medium using coupled wave equations.

        Returns signal and conjugate field amplitudes along z.
        """
        # Normalized pump intensity
        I_pump = np.abs(E_pump_f)**2 + np.abs(E_pump_b)**2
        kappa = self.coupling_coefficient(I_pump)

        n_points = len(z_range)
        E_s = np.zeros(n_points, dtype=complex)
        E_c = np.zeros(n_points, dtype=complex)

        # Initial conditions
        E_s[0] = E_signal
        E_c[-1] = 0  # No conjugate at output

        # Coupled equations:
        # dE_s/dz = i * kappa * E_pump_f * E_pump_b * E_c*
        # dE_c/dz = -i * kappa * E_pump_f* * E_pump_b* * E_s*

        # Solve using shooting method or analytical solution
        # For undepleted pumps, analytical solution:
        kL = kappa * self.L

        for i, z in enumerate(z_range):
            kz = kappa * z

            # Analytical solution for counter-propagating geometry
            E_s[i] = E_signal * np.cosh(kappa * (self.L - z)) / np.cosh(kL)
            E_c[i] = -1j * E_signal.conjugate() * np.sinh(kappa * (self.L - z)) / np.cosh(kL)

        return E_s, E_c

    def phase_matching_bandwidth(self) -> float:
        """Estimate phase matching bandwidth"""
        # delta_k_max ~ pi / L for 3dB reduction
        return np.pi / self.L


class DegenerateFWM:
    """Degenerate four-wave mixing for phase conjugation"""

    def __init__(self, chi3: float, wavelength: float, n0: float):
        """
        Args:
            chi3: Third-order susceptibility
            wavelength: Wavelength
            n0: Refractive index
        """
        self.chi3 = chi3
        self.wavelength = wavelength
        self.n0 = n0
        self.k = 2 * np.pi * n0 / wavelength
        self.c = 3e8
        self.omega = 2 * np.pi * self.c / wavelength

    def generate_signal_with_aberration(
        self,
        x: np.ndarray,
        y: np.ndarray,
        beam_width: float,
        aberration_type: str = 'random'
    ) -> tuple:
        """Generate a signal beam with aberrations"""
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)

        # Gaussian beam
        amplitude = np.exp(-R**2 / beam_width**2)

        # Add aberration to phase
        if aberration_type == 'random':
            np.random.seed(42)
            phase_aberration = 2 * np.random.rand(*X.shape) - 1
            # Smooth the phase
            from scipy.ndimage import gaussian_filter
            phase_aberration = gaussian_filter(phase_aberration, sigma=5)
            phase_aberration *= 2 * np.pi

        elif aberration_type == 'spherical':
            phase_aberration = 0.5 * (R / beam_width)**4

        elif aberration_type == 'astigmatism':
            phase_aberration = 0.3 * ((X / beam_width)**2 - (Y / beam_width)**2)

        elif aberration_type == 'coma':
            phase_aberration = 0.3 * (R / beam_width)**3 * np.cos(np.arctan2(Y, X))

        else:
            phase_aberration = np.zeros_like(R)

        E_signal = amplitude * np.exp(1j * phase_aberration)

        return E_signal, phase_aberration

    def phase_conjugate(self, E_signal: np.ndarray) -> np.ndarray:
        """
        Generate phase conjugate of input beam.

        E_conjugate ~ E_signal* (complex conjugate)
        """
        return np.conj(E_signal)

    def propagate_beam(
        self,
        E: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        distance: float
    ) -> np.ndarray:
        """Propagate beam using angular spectrum method"""
        dx = x[1] - x[0]
        dy = y[1] - y[0]

        # Spatial frequencies
        kx = np.fft.fftfreq(len(x), dx) * 2 * np.pi
        ky = np.fft.fftfreq(len(y), dy) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky)

        # Transfer function
        kz_sq = self.k**2 - KX**2 - KY**2
        kz = np.sqrt(np.maximum(kz_sq, 0) + 0j)

        H = np.exp(1j * kz * distance)

        # Propagate
        E_ft = np.fft.fft2(E)
        E_prop = np.fft.ifft2(E_ft * H)

        return E_prop


def plot_fwm_basics():
    """Plot basic four-wave mixing concepts"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Wave vector diagram
    ax1 = axes[0, 0]

    # Phase matching: k4 = k1 + k2 - k3
    angles = [0, 0, 0.3, -0.3]  # Angles for collinear and non-collinear

    # Draw vectors
    ax1.arrow(0, 0, 1, 0, head_width=0.05, head_length=0.05, fc='red', ec='red')
    ax1.text(0.5, 0.1, r'$\vec{k}_1$ (pump 1)', color='red', fontsize=10)

    ax1.arrow(0, 0, -1, 0, head_width=0.05, head_length=0.05, fc='blue', ec='blue')
    ax1.text(-0.7, 0.1, r'$\vec{k}_2$ (pump 2)', color='blue', fontsize=10)

    ax1.arrow(0, -0.5, 0.8*np.cos(0.3), 0.8*np.sin(0.3),
              head_width=0.05, head_length=0.05, fc='green', ec='green')
    ax1.text(0.5, -0.3, r'$\vec{k}_3$ (signal)', color='green', fontsize=10)

    ax1.arrow(0, -0.5, -0.8*np.cos(0.3), -0.8*np.sin(0.3),
              head_width=0.05, head_length=0.05, fc='purple', ec='purple')
    ax1.text(-0.8, -0.7, r'$\vec{k}_4$ (conjugate)', color='purple', fontsize=10)

    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1, 0.5)
    ax1.set_aspect('equal')
    ax1.set_title('Phase Matching: $\\vec{k}_4 = \\vec{k}_1 + \\vec{k}_2 - \\vec{k}_3$\n'
                  'Counter-propagating pump geometry')
    ax1.axis('off')

    # Plot 2: Energy level diagram
    ax2 = axes[0, 1]

    # Virtual levels
    levels = [0, 1, 2, 1, 0]
    x_pos = [0, 0.3, 0.6, 0.9, 1.2]

    for i in range(len(levels) - 1):
        ax2.plot([x_pos[i], x_pos[i+1]], [levels[i], levels[i+1]], 'b-', linewidth=2)
        mid_x = (x_pos[i] + x_pos[i+1]) / 2
        mid_y = (levels[i] + levels[i+1]) / 2

    # Arrows for photons
    ax2.annotate('', xy=(0.15, 0.5), xytext=(0.15, 0),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax2.text(0.05, 0.25, r'$\omega_1$', fontsize=12, color='red')

    ax2.annotate('', xy=(0.45, 1.5), xytext=(0.45, 1),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax2.text(0.35, 1.25, r'$\omega_2$', fontsize=12, color='blue')

    ax2.annotate('', xy=(0.75, 1), xytext=(0.75, 1.5),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax2.text(0.85, 1.25, r'$\omega_3$', fontsize=12, color='green')

    ax2.annotate('', xy=(1.05, 0), xytext=(1.05, 0.5),
                arrowprops=dict(arrowstyle='->', color='purple', lw=2))
    ax2.text(1.15, 0.25, r'$\omega_4$', fontsize=12, color='purple')

    ax2.axhline(0, color='black', linewidth=1)
    ax2.axhline(2, color='gray', linestyle='--', alpha=0.5)

    ax2.text(-0.2, 0, 'Ground', fontsize=10)
    ax2.text(-0.2, 2, 'Virtual', fontsize=10, color='gray')

    ax2.set_xlim(-0.3, 1.4)
    ax2.set_ylim(-0.3, 2.3)
    ax2.set_title('Energy Diagram for FWM\n'
                  r'$\omega_4 = \omega_1 + \omega_2 - \omega_3$')
    ax2.axis('off')

    # Plot 3: Phase conjugate reflectivity vs pump intensity
    ax3 = axes[1, 0]

    chi3 = 1e-20  # Typical value for CS2
    wavelength = 532e-9
    n0 = 1.5

    lengths = [1e-3, 5e-3, 10e-3]  # 1mm, 5mm, 10mm
    pump_intensities = np.linspace(0, 1e13, 100)  # W/m^2

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(lengths)))

    for L, color in zip(lengths, colors):
        fwm = FourWaveMixing(chi3, wavelength, n0, L)
        reflectivity = [fwm.phase_conjugate_reflectivity(I) for I in pump_intensities]

        ax3.plot(pump_intensities * 1e-9, reflectivity, color=color, linewidth=2,
                 label=f'L = {L*1e3:.0f} mm')

    ax3.axhline(1.0, color='gray', linestyle='--', alpha=0.5)

    ax3.set_xlabel('Pump intensity (GW/m^2)')
    ax3.set_ylabel('Phase conjugate reflectivity')
    ax3.set_title('Phase Conjugate Reflectivity vs Pump Intensity\n'
                  r'$R = \tanh^2(\kappa L)$')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 1.1)

    # Plot 4: Phase matching curve
    ax4 = axes[1, 1]

    L = 5e-3
    fwm = FourWaveMixing(chi3, wavelength, n0, L)

    delta_k = np.linspace(-5000, 5000, 500)  # Phase mismatch
    pump_intensity = 5e12

    reflectivity = [fwm.phase_conjugate_reflectivity(pump_intensity, dk)
                    for dk in delta_k]

    ax4.plot(delta_k * L, reflectivity, 'b-', linewidth=2)
    ax4.axvline(0, color='green', linestyle='--', alpha=0.7, label='Phase matched')
    ax4.axvline(np.pi, color='red', linestyle=':', alpha=0.7)
    ax4.axvline(-np.pi, color='red', linestyle=':', alpha=0.7, label='First null')

    ax4.set_xlabel(r'Phase mismatch $\Delta k \cdot L$')
    ax4.set_ylabel('Relative reflectivity')
    ax4.set_title('Phase Matching Curve\n'
                  'Efficiency decreases with phase mismatch')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-15, 15)

    plt.tight_layout()
    return fig


def plot_phase_conjugation():
    """Demonstrate phase conjugation and aberration correction"""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Setup
    chi3 = 1e-20
    wavelength = 532e-9
    n0 = 1.5

    fwm = DegenerateFWM(chi3, wavelength, n0)

    x = np.linspace(-5e-3, 5e-3, 200)
    y = np.linspace(-5e-3, 5e-3, 200)
    X, Y = np.meshgrid(x, y)
    beam_width = 2e-3

    # Generate aberrated signal
    E_signal, phase_aberration = fwm.generate_signal_with_aberration(
        x, y, beam_width, 'random')

    # Phase conjugate
    E_conjugate = fwm.phase_conjugate(E_signal)

    # Propagate through aberration again
    E_corrected = E_conjugate * np.exp(1j * phase_aberration)

    # Plot 1: Original beam intensity
    ax1 = axes[0, 0]
    im1 = ax1.imshow(np.abs(E_signal)**2, extent=[x.min()*1e3, x.max()*1e3,
                                                   y.min()*1e3, y.max()*1e3],
                     cmap='hot', origin='lower')
    plt.colorbar(im1, ax=ax1, label='Intensity')
    ax1.set_xlabel('x (mm)')
    ax1.set_ylabel('y (mm)')
    ax1.set_title('Original Signal Beam\nIntensity')

    # Plot 2: Original beam phase (aberrated)
    ax2 = axes[0, 1]
    im2 = ax2.imshow(np.angle(E_signal), extent=[x.min()*1e3, x.max()*1e3,
                                                  y.min()*1e3, y.max()*1e3],
                     cmap='twilight', origin='lower', vmin=-np.pi, vmax=np.pi)
    plt.colorbar(im2, ax=ax2, label='Phase (rad)')
    ax2.set_xlabel('x (mm)')
    ax2.set_ylabel('y (mm)')
    ax2.set_title('Original Signal Beam\nPhase (aberrated)')

    # Plot 3: Phase conjugate phase
    ax3 = axes[0, 2]
    im3 = ax3.imshow(np.angle(E_conjugate), extent=[x.min()*1e3, x.max()*1e3,
                                                     y.min()*1e3, y.max()*1e3],
                     cmap='twilight', origin='lower', vmin=-np.pi, vmax=np.pi)
    plt.colorbar(im3, ax=ax3, label='Phase (rad)')
    ax3.set_xlabel('x (mm)')
    ax3.set_ylabel('y (mm)')
    ax3.set_title('Phase Conjugate Beam\nPhase = -Phase_signal')

    # Plot 4: Corrected beam phase
    ax4 = axes[1, 0]
    im4 = ax4.imshow(np.angle(E_corrected), extent=[x.min()*1e3, x.max()*1e3,
                                                     y.min()*1e3, y.max()*1e3],
                     cmap='twilight', origin='lower', vmin=-np.pi, vmax=np.pi)
    plt.colorbar(im4, ax=ax4, label='Phase (rad)')
    ax4.set_xlabel('x (mm)')
    ax4.set_ylabel('y (mm)')
    ax4.set_title('After Double-Pass Through Aberration\n'
                  'Phase is corrected!')

    # Plot 5: Comparison of phases
    ax5 = axes[1, 1]

    center_idx = len(y) // 2
    ax5.plot(x * 1e3, phase_aberration[center_idx, :], 'r-', linewidth=2,
             label='Aberration phi')
    ax5.plot(x * 1e3, np.angle(E_conjugate)[center_idx, :], 'b--', linewidth=2,
             label='Conjugate phase = -phi')
    ax5.plot(x * 1e3, np.angle(E_corrected)[center_idx, :], 'g-', linewidth=2,
             label='Corrected (phi - phi = 0)')

    ax5.set_xlabel('x (mm)')
    ax5.set_ylabel('Phase (rad)')
    ax5.set_title('Phase Cross-Section (y=0)\n'
                  'Conjugate reverses aberration')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    # Plot 6: Schematic of aberration correction
    ax6 = axes[1, 2]

    # Draw schematic
    ax6.text(0.1, 0.9, 'Original beam', fontsize=11, transform=ax6.transAxes)
    ax6.arrow(0.1, 0.75, 0.2, 0, transform=ax6.transAxes,
              head_width=0.02, head_length=0.02, fc='blue', ec='blue')

    ax6.text(0.35, 0.78, 'Aberrating\nmedium', fontsize=9, transform=ax6.transAxes,
            ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    ax6.arrow(0.45, 0.75, 0.15, 0, transform=ax6.transAxes,
              head_width=0.02, head_length=0.02, fc='red', ec='red')
    ax6.text(0.52, 0.68, 'Aberrated', fontsize=9, transform=ax6.transAxes, color='red')

    ax6.text(0.65, 0.78, 'FWM\n(chi^3)', fontsize=9, transform=ax6.transAxes,
            ha='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    ax6.arrow(0.65, 0.65, -0.15, 0, transform=ax6.transAxes,
              head_width=0.02, head_length=0.02, fc='purple', ec='purple')
    ax6.text(0.52, 0.55, 'Conjugate', fontsize=9, transform=ax6.transAxes, color='purple')

    ax6.text(0.35, 0.52, 'Same\nmedium', fontsize=9, transform=ax6.transAxes,
            ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    ax6.arrow(0.25, 0.55, -0.15, 0, transform=ax6.transAxes,
              head_width=0.02, head_length=0.02, fc='green', ec='green')
    ax6.text(0.05, 0.48, 'Corrected!', fontsize=10, transform=ax6.transAxes,
            color='green', fontweight='bold')

    ax6.text(0.5, 0.3, 'Phase Conjugation Principle:\n'
             r'$E_{conj} \propto E_{signal}^*$' + '\n'
             r'$\phi_{conj} = -\phi_{signal}$' + '\n\n'
             'Double pass through aberration:\n'
             r'$\phi_{total} = \phi_{aberr} + (-\phi_{aberr}) = 0$',
            fontsize=10, transform=ax6.transAxes, ha='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.axis('off')
    ax6.set_title('Aberration Correction via Phase Conjugation')

    plt.tight_layout()
    return fig


def plot_field_evolution():
    """Plot field evolution through the FWM medium"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    chi3 = 1e-20
    wavelength = 532e-9
    n0 = 1.5
    L = 10e-3

    fwm = FourWaveMixing(chi3, wavelength, n0, L)

    z = np.linspace(0, L, 100)

    # Plot 1: Field amplitudes for different pump intensities
    ax1 = axes[0, 0]

    pump_intensities = [1e12, 3e12, 5e12, 8e12]
    colors = plt.cm.plasma(np.linspace(0, 0.8, len(pump_intensities)))

    for I_pump, color in zip(pump_intensities, colors):
        E_s, E_c = fwm.propagate_fields(z, 1.0, 1.0, 1.0)

        # Scale based on coupling
        kappa = fwm.coupling_coefficient(I_pump)
        kL = kappa * L

        E_s_scaled = np.cosh(kappa * (L - z)) / np.cosh(kL)
        E_c_scaled = np.sinh(kappa * (L - z)) / np.cosh(kL)

        ax1.plot(z * 1e3, np.abs(E_s_scaled), color=color, linewidth=2,
                 label=f'I_pump = {I_pump*1e-12:.0f} TW/m^2')
        ax1.plot(z * 1e3, np.abs(E_c_scaled), '--', color=color, linewidth=2)

    ax1.set_xlabel('Position z (mm)')
    ax1.set_ylabel('Normalized field amplitude')
    ax1.set_title('Signal (solid) and Conjugate (dashed) vs Position\n'
                  'Signal depleted, conjugate grows')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Reflectivity vs interaction length
    ax2 = axes[0, 1]

    I_pump = 5e12
    lengths = np.linspace(0.1e-3, 20e-3, 100)

    kappa = fwm.coupling_coefficient(I_pump)

    reflectivity = np.tanh(kappa * lengths)**2

    ax2.plot(lengths * 1e3, reflectivity, 'b-', linewidth=2)
    ax2.axhline(1.0, color='gray', linestyle='--', alpha=0.5)

    ax2.set_xlabel('Interaction length L (mm)')
    ax2.set_ylabel('Phase conjugate reflectivity')
    ax2.set_title(f'Reflectivity vs Length at I_pump = {I_pump*1e-12:.0f} TW/m^2\n'
                  r'$R = \tanh^2(\kappa L)$')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 20)
    ax2.set_ylim(0, 1.1)

    # Mark asymptotic approach
    L_sat = 2 / kappa  # Where tanh approaches 0.96
    if L_sat * 1e3 < 20:  # Only show if within plot range
        ax2.axvline(L_sat * 1e3, color='red', linestyle=':', alpha=0.7)
        ax2.text(L_sat * 1e3 + 1, 0.5, f'L_sat = {L_sat*1e3:.1f} mm\n(~saturation)',
                fontsize=10, color='red')

    # Plot 3: Chi^(3) materials comparison
    ax3 = axes[1, 0]

    materials = [
        ('CS2', 1e-20, 1.63),
        ('Silica fiber', 2.5e-22, 1.45),
        ('GaAs', 1e-17, 3.3),
        ('Chalcogenide', 5e-18, 2.4),
    ]

    L = 10e-3
    pump_intensity = np.linspace(0, 1e13, 100)

    for name, chi3, n0 in materials:
        fwm_mat = FourWaveMixing(chi3, wavelength, n0, L)
        R = [fwm_mat.phase_conjugate_reflectivity(I) for I in pump_intensity]

        ax3.plot(pump_intensity * 1e-12, R, linewidth=2,
                 label=f'{name} (chi3 = {chi3:.0e})')

    ax3.axhline(1.0, color='gray', linestyle='--', alpha=0.5)

    ax3.set_xlabel('Pump intensity (TW/m^2)')
    ax3.set_ylabel('Reflectivity')
    ax3.set_title('Reflectivity for Different Nonlinear Materials\n'
                  f'L = {L*1e3:.0f} mm')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 1.1)

    # Plot 4: Applications diagram
    ax4 = axes[1, 1]

    applications = [
        ("Aberration\nCorrection", 0.2, 0.8, 'Corrects wavefront distortions'),
        ("Optical\nPhase\nConjugation", 0.5, 0.8, 'Time-reversal of light'),
        ("Real-time\nHolography", 0.8, 0.8, 'Dynamic 3D imaging'),
        ("Optical\nSignal\nProcessing", 0.2, 0.3, 'Wavelength conversion'),
        ("Laser\nCavity\nStabilization", 0.5, 0.3, 'Self-aligning resonators'),
        ("Optical\nMemory", 0.8, 0.3, 'Photorefractive storage'),
    ]

    for name, x, y, desc in applications:
        ax4.text(x, y, name, fontsize=11, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        ax4.text(x, y - 0.12, desc, fontsize=8, ha='center', va='top', style='italic')

    ax4.text(0.5, 0.55, 'Four-Wave\nMixing', fontsize=14, ha='center', va='center',
            fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    # Draw arrows
    for name, x, y, desc in applications:
        ax4.annotate('', xy=(x, y - 0.08), xytext=(0.5, 0.55 + 0.08 if y > 0.5 else 0.55 - 0.08),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1))

    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Applications of Four-Wave Mixing')

    plt.tight_layout()
    return fig


def plot_spectral_properties():
    """Plot spectral and temporal properties of FWM"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    chi3 = 1e-20
    wavelength = 532e-9
    n0 = 1.5
    L = 5e-3

    fwm = FourWaveMixing(chi3, wavelength, n0, L)

    # Plot 1: Frequency diagram for non-degenerate FWM
    ax1 = axes[0, 0]

    # omega_4 = omega_1 + omega_2 - omega_3
    omega_0 = 2 * np.pi * 3e8 / wavelength

    # Show different FWM processes
    processes = [
        ('Degenerate', [omega_0, omega_0, omega_0, omega_0]),
        ('Partially degenerate', [omega_0*1.01, omega_0*0.99, omega_0, omega_0]),
        ('Non-degenerate', [omega_0*1.02, omega_0*0.98, omega_0*1.01, omega_0*0.99]),
    ]

    y_offset = 0
    for name, omegas in processes:
        colors = ['red', 'blue', 'green', 'purple']
        labels = [r'$\omega_1$', r'$\omega_2$', r'$\omega_3$', r'$\omega_4$']

        for i, (omega, color, label) in enumerate(zip(omegas, colors, labels)):
            ax1.arrow(i * 0.2, y_offset, 0, omega/omega_0 - 1 + 0.1,
                     head_width=0.02, head_length=0.02, fc=color, ec=color)
            if y_offset == 0:
                ax1.text(i * 0.2, 0.15, label, color=color, ha='center', fontsize=10)

        ax1.text(0.9, y_offset + 0.05, name, fontsize=10)
        y_offset += 0.3

    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.1, 1)
    ax1.set_xlabel('Process')
    ax1.set_ylabel('Relative frequency')
    ax1.set_title('FWM Frequency Conservation\n'
                  r'$\omega_4 = \omega_1 + \omega_2 - \omega_3$')
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.3)

    # Plot 2: Bandwidth of FWM process
    ax2 = axes[0, 1]

    # Signal wavelength detuning
    delta_lambda = np.linspace(-10e-9, 10e-9, 500)  # +/- 10 nm
    lambda_signal = wavelength + delta_lambda

    # Phase mismatch due to wavelength detuning (simplified)
    delta_k = 4 * np.pi * n0 * delta_lambda / wavelength**2

    pump_intensity = 5e12

    efficiency = []
    for dk in delta_k:
        R = fwm.phase_conjugate_reflectivity(pump_intensity, dk)
        efficiency.append(R)

    efficiency = np.array(efficiency)
    max_eff = efficiency.max()
    if max_eff > 0:
        efficiency /= max_eff
    else:
        efficiency = np.ones_like(efficiency)

    ax2.plot(delta_lambda * 1e9, efficiency, 'b-', linewidth=2)

    # Find bandwidth
    half_max_idx = np.where(efficiency > 0.5)[0]
    if len(half_max_idx) > 1:
        bw = delta_lambda[half_max_idx[-1]] - delta_lambda[half_max_idx[0]]
        ax2.axhline(0.5, color='red', linestyle='--', alpha=0.5)
        ax2.annotate('', xy=(delta_lambda[half_max_idx[0]]*1e9, 0.5),
                    xytext=(delta_lambda[half_max_idx[-1]]*1e9, 0.5),
                    arrowprops=dict(arrowstyle='<->', color='red'))
        ax2.text(0, 0.6, f'Bandwidth ~ {bw*1e9:.1f} nm', ha='center', fontsize=10, color='red')

    ax2.set_xlabel('Wavelength detuning (nm)')
    ax2.set_ylabel('Relative efficiency')
    ax2.set_title('FWM Bandwidth\n'
                  f'L = {L*1e3:.0f} mm')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Temporal response
    ax3 = axes[1, 0]

    # FWM response time determined by chi^(3) mechanism
    response_times = {
        'Electronic (bound)': 1e-15,
        'Molecular reorientation': 1e-12,
        'Thermal': 1e-6,
    }

    t = np.linspace(0, 5e-12, 1000)

    for mechanism, tau in response_times.items():
        if tau < 1e-13:
            response = np.exp(-t / (10 * tau))  # Fast electronic
        else:
            response = (1 - np.exp(-t / tau)) * np.exp(-t / (10 * tau))

        ax3.plot(t * 1e12, response, linewidth=2, label=f'{mechanism} (tau ~ {tau*1e12:.0f} ps)')

    ax3.set_xlabel('Time (ps)')
    ax3.set_ylabel('Response (arb. units)')
    ax3.set_title('Temporal Response of Chi^(3) Mechanisms\n'
                  'Electronic is fastest, thermal is slowest')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 5)

    # Plot 4: Pump depletion effects
    ax4 = axes[1, 1]

    # With pump depletion, efficiency saturates
    signal_intensity_ratio = np.linspace(0.001, 1, 100)

    # Undepleted pump
    undepleted = signal_intensity_ratio**2 * 4  # Simplified

    # With depletion (saturates)
    depleted = 4 * signal_intensity_ratio**2 / (1 + signal_intensity_ratio**2)

    ax4.plot(signal_intensity_ratio, undepleted, 'b--', linewidth=2,
             label='Undepleted pump')
    ax4.plot(signal_intensity_ratio, depleted, 'r-', linewidth=2,
             label='With pump depletion')

    ax4.set_xlabel('Signal / Pump intensity ratio')
    ax4.set_ylabel('Conjugate intensity (arb. units)')
    ax4.set_title('Pump Depletion Effects\n'
                  'Strong signal depletes pump, reducing efficiency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 1)

    plt.tight_layout()
    return fig


def main():
    """Main function to demonstrate four-wave mixing"""

    # Create figures
    fig1 = plot_fwm_basics()
    fig2 = plot_phase_conjugation()
    fig3 = plot_field_evolution()
    fig4 = plot_spectral_properties()

    # Save plots
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)

    fig1.savefig(os.path.join(output_dir, 'four_wave_mixing.png'),
                 dpi=150, bbox_inches='tight')
    fig2.savefig(os.path.join(output_dir, 'phase_conjugation.png'),
                 dpi=150, bbox_inches='tight')
    fig3.savefig(os.path.join(output_dir, 'fwm_field_evolution.png'),
                 dpi=150, bbox_inches='tight')
    fig4.savefig(os.path.join(output_dir, 'fwm_spectral.png'),
                 dpi=150, bbox_inches='tight')

    print(f"Plots saved to {output_dir}/four_wave_mixing*.png and related")

    # Print analysis
    print("\n=== Four-Wave Mixing Analysis ===")

    chi3 = 1e-20
    wavelength = 532e-9
    n0 = 1.5
    L = 10e-3

    fwm = FourWaveMixing(chi3, wavelength, n0, L)

    print(f"\nParameters:")
    print(f"  Chi^(3): {chi3:.0e} m^2/V^2")
    print(f"  Wavelength: {wavelength*1e9:.1f} nm")
    print(f"  Refractive index: {n0}")
    print(f"  Interaction length: {L*1e3:.0f} mm")

    pump_intensities = [1e12, 5e12, 1e13]
    print(f"\nPhase conjugate reflectivity:")
    for I in pump_intensities:
        R = fwm.phase_conjugate_reflectivity(I)
        kappa = fwm.coupling_coefficient(I)
        print(f"  I_pump = {I*1e-12:.0f} TW/m^2: R = {R*100:.1f}%, kappa*L = {kappa*L:.3f}")

    print("\nKey equations:")
    print("  Frequency: omega_4 = omega_1 + omega_2 - omega_3")
    print("  Wavevector: k_4 = k_1 + k_2 - k_3 (phase matching)")
    print("  Reflectivity: R = tanh^2(kappa*L)")
    print("  Phase conjugate: E_c ~ E_s* (complex conjugate)")


if __name__ == "__main__":
    main()
