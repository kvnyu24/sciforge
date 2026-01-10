"""
Example 118: Kerr Lens and Self-Focusing

This example demonstrates the optical Kerr effect and self-focusing of intense
laser beams due to the intensity-dependent refractive index.

Physics:
    Optical Kerr effect:
    n(I) = n0 + n2 * I

    where:
    - n0 is the linear refractive index
    - n2 is the nonlinear refractive index (m^2/W)
    - I is the optical intensity (W/m^2)

    For a Gaussian beam, the intensity profile induces a lens-like
    phase shift (higher index at center), causing self-focusing.

    Critical power for self-focusing:
    P_cr = 3.77 * lambda^2 / (8 * pi * n0 * n2)

    For P > P_cr, self-focusing overcomes diffraction and the beam collapses.

    For P < P_cr, diffraction dominates and beam spreads.

    The collapse distance (for P >> P_cr):
    z_f ~ z_R / sqrt(P/P_cr - 1)

    where z_R is the Rayleigh range.

    Applications:
    - Kerr-lens mode locking in ultrafast lasers
    - Optical limiting
    - All-optical switching
    - Pulse compression
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


class KerrMedium:
    """Kerr nonlinear medium with intensity-dependent refractive index"""

    def __init__(
        self,
        n0: float,
        n2: float,
        wavelength: float
    ):
        """
        Args:
            n0: Linear refractive index
            n2: Nonlinear refractive index (m^2/W)
            wavelength: Wavelength (m)
        """
        self.n0 = n0
        self.n2 = n2
        self.wavelength = wavelength
        self.k = 2 * np.pi * n0 / wavelength

    def refractive_index(self, intensity: np.ndarray) -> np.ndarray:
        """Calculate intensity-dependent refractive index"""
        return self.n0 + self.n2 * intensity

    def critical_power(self) -> float:
        """
        Calculate critical power for self-focusing.

        P_cr = 3.77 * lambda^2 / (8 * pi * n0 * n2)
        """
        return 3.77 * self.wavelength**2 / (8 * np.pi * self.n0 * self.n2)

    def kerr_lens_focal_length(self, beam_radius: float, power: float) -> float:
        """
        Calculate effective focal length of the Kerr lens.

        For a Gaussian beam in a thin Kerr medium of length L:
        1/f = n2 * I_peak * k * L / w^2

        Simplified version assuming thin lens.
        """
        # Peak intensity for Gaussian beam
        I_peak = 2 * power / (np.pi * beam_radius**2)

        # Approximate focal length
        f = beam_radius**2 / (self.n2 * I_peak * beam_radius)

        return f


class GaussianBeam:
    """Gaussian beam propagation with Kerr effect"""

    def __init__(
        self,
        wavelength: float,
        waist: float,
        power: float
    ):
        """
        Args:
            wavelength: Wavelength (m)
            waist: Beam waist radius (m)
            power: Beam power (W)
        """
        self.wavelength = wavelength
        self.w0 = waist
        self.power = power
        self.k = 2 * np.pi / wavelength

        # Rayleigh range
        self.z_R = np.pi * waist**2 / wavelength

    def spot_size(self, z: np.ndarray) -> np.ndarray:
        """Calculate beam spot size vs propagation distance (linear)"""
        return self.w0 * np.sqrt(1 + (z / self.z_R)**2)

    def peak_intensity(self, z: np.ndarray = None) -> float:
        """Calculate peak intensity at waist"""
        if z is None:
            return 2 * self.power / (np.pi * self.w0**2)
        else:
            w = self.spot_size(z)
            return 2 * self.power / (np.pi * w**2)

    def intensity_profile(self, r: np.ndarray, z: float = 0) -> np.ndarray:
        """Gaussian intensity profile at position z"""
        w = self.spot_size(np.array([z]))[0]
        I0 = self.peak_intensity(np.array([z]))
        return I0 * np.exp(-2 * r**2 / w**2)

    def propagate_with_kerr(
        self,
        kerr: KerrMedium,
        z_max: float,
        n_steps: int = 1000
    ) -> tuple:
        """
        Propagate Gaussian beam through Kerr medium.

        Uses the variational approach with Gaussian ansatz.

        Returns z, beam_width, on_axis_intensity
        """
        P_cr = kerr.critical_power()
        P_ratio = self.power / P_cr

        z = np.linspace(0, z_max, n_steps)

        if P_ratio >= 1:
            # Self-focusing regime
            # Collapse distance approximation
            if P_ratio > 1:
                z_f = 0.367 * self.z_R / np.sqrt((np.sqrt(P_ratio) - 0.858)**2 - 0.0219)
            else:
                z_f = np.inf

            # Beam width evolution (variational result)
            # d^2w/dz^2 = 4/(k^2*w^3) * (1 - P/P_cr)
            # For P > P_cr, this has no stable solution - collapse occurs

            # Approximate solution before collapse
            w = np.zeros_like(z)
            w[0] = self.w0

            dz = z[1] - z[0]
            dw_dz = 0  # Start at waist

            for i in range(1, len(z)):
                if z[i] < z_f * 0.9:  # Stop before singularity
                    d2w = 4 / (self.k**2 * w[i-1]**3) * (1 - P_ratio)
                    dw_dz += d2w * dz
                    w[i] = max(w[i-1] + dw_dz * dz, self.wavelength)  # Minimum is wavelength
                else:
                    w[i] = self.wavelength  # Collapse

        else:
            # Diffraction-dominated regime (modified by Kerr)
            # Effective Rayleigh range is increased
            z_R_eff = self.z_R / np.sqrt(1 - P_ratio)

            w = self.w0 * np.sqrt(1 + (z / z_R_eff)**2)

        # Calculate intensity
        I = 2 * self.power / (np.pi * w**2)

        return z, w, I


class SelfFocusingSimulation:
    """Numerical simulation of self-focusing using split-step method"""

    def __init__(
        self,
        wavelength: float,
        n0: float,
        n2: float,
        beam_waist: float
    ):
        """Initialize simulation parameters"""
        self.wavelength = wavelength
        self.n0 = n0
        self.n2 = n2
        self.w0 = beam_waist
        self.k = 2 * np.pi * n0 / wavelength

    def create_initial_field(
        self,
        x: np.ndarray,
        y: np.ndarray,
        power: float
    ) -> np.ndarray:
        """Create initial Gaussian field"""
        X, Y = np.meshgrid(x, y)
        R2 = X**2 + Y**2

        # Amplitude for given power
        # P = integral of |E|^2 = pi * w0^2 * |E0|^2 / 2
        E0 = np.sqrt(2 * power / (np.pi * self.w0**2))

        return E0 * np.exp(-R2 / self.w0**2)

    def propagate_step(
        self,
        E: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        dz: float
    ) -> np.ndarray:
        """
        Propagate field by dz using split-step Fourier method.

        1. Apply nonlinear phase (Kerr effect)
        2. Propagate in Fourier space (diffraction)
        """
        dx = x[1] - x[0]
        dy = y[1] - y[0]

        # Intensity
        I = np.abs(E)**2

        # Nonlinear phase
        phi_nl = self.k * self.n2 * I * dz
        E = E * np.exp(1j * phi_nl)

        # Fourier transform for diffraction
        kx = np.fft.fftfreq(len(x), dx) * 2 * np.pi
        ky = np.fft.fftfreq(len(y), dy) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky)

        # Diffraction operator
        H = np.exp(-1j * (KX**2 + KY**2) * dz / (2 * self.k))

        E_ft = np.fft.fft2(E)
        E = np.fft.ifft2(E_ft * H)

        return E

    def simulate(
        self,
        power: float,
        z_max: float,
        n_z: int = 100,
        n_xy: int = 128
    ) -> tuple:
        """
        Run self-focusing simulation.

        Returns z, x, intensity profiles at each z
        """
        # Spatial grid
        x_max = 10 * self.w0
        x = np.linspace(-x_max, x_max, n_xy)
        y = np.linspace(-x_max, x_max, n_xy)

        # Initial field
        E = self.create_initial_field(x, y, power)

        # Propagation
        z = np.linspace(0, z_max, n_z)
        dz = z[1] - z[0]

        # Store results
        intensity_profiles = np.zeros((n_z, n_xy))
        beam_widths = np.zeros(n_z)
        peak_intensities = np.zeros(n_z)

        center = n_xy // 2

        for i in range(n_z):
            I = np.abs(E)**2

            # Store cross-section
            intensity_profiles[i, :] = I[center, :]

            # Calculate beam width
            I_x = I[center, :]
            x_squared = np.sum(x**2 * I_x) / np.sum(I_x)
            beam_widths[i] = np.sqrt(2 * x_squared)  # 1/e^2 radius

            # Peak intensity
            peak_intensities[i] = I.max()

            # Propagate
            if i < n_z - 1:
                E = self.propagate_step(E, x, y, dz)

        return z, x, intensity_profiles, beam_widths, peak_intensities


def plot_kerr_effect_basics():
    """Plot basic Kerr effect concepts"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Parameters
    n0 = 1.5
    n2 = 3e-20  # m^2/W (typical for silica)
    wavelength = 800e-9  # Ti:Sapphire

    kerr = KerrMedium(n0, n2, wavelength)
    P_cr = kerr.critical_power()

    # Plot 1: Intensity-dependent refractive index
    ax1 = axes[0, 0]

    intensity = np.linspace(0, 1e16, 500)  # W/m^2
    n = kerr.refractive_index(intensity)

    ax1.plot(intensity * 1e-12, n, 'b-', linewidth=2)
    ax1.axhline(n0, color='gray', linestyle='--', alpha=0.5, label='n0')

    ax1.set_xlabel('Intensity (TW/m^2)')
    ax1.set_ylabel('Refractive index n')
    ax1.set_title('Optical Kerr Effect: n(I) = n0 + n2*I\n'
                  f'n2 = {n2:.0e} m^2/W')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Annotate delta_n
    I_example = 5e15
    n_example = kerr.refractive_index(I_example)
    ax1.annotate('', xy=(I_example*1e-12, n_example), xytext=(I_example*1e-12, n0),
                arrowprops=dict(arrowstyle='<->', color='red'))
    ax1.text(I_example*1e-12 + 0.5, (n0 + n_example)/2,
             f'delta_n = {(n_example-n0)*1e6:.1f}e-6', fontsize=10, color='red')

    # Plot 2: Gaussian beam creates lens
    ax2 = axes[0, 1]

    r = np.linspace(-3, 3, 100)  # Normalized to beam radius

    # Gaussian intensity profile
    I_profile = np.exp(-2 * r**2)

    ax2.plot(r, I_profile, 'b-', linewidth=2, label='Intensity I(r)')

    # Corresponding refractive index profile
    n_profile = 1 + 0.5 * I_profile  # Arbitrary scaling for visualization
    ax2.plot(r, n_profile - 0.5, 'r-', linewidth=2, label='n(r) - n0 (scaled)')

    # Equivalent lens shape
    lens = 1 - r**2 / 4
    ax2.fill_between(r, 0.4, 0.4 + 0.2 * lens, alpha=0.3, color='green',
                     label='Equivalent GRIN lens')

    ax2.set_xlabel('Radial position r/w0')
    ax2.set_ylabel('Normalized value')
    ax2.set_title('Self-Induced Lens Effect\n'
                  'Gaussian beam creates positive lens (n2 > 0)')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-3, 3)

    # Plot 3: Critical power
    ax3 = axes[1, 0]

    # Different materials
    materials = [
        ('Silica (glass fiber)', 1.45, 2.5e-20),
        ('Air (1 atm)', 1.0, 3e-23),
        ('CS2', 1.63, 3e-18),
        ('Water', 1.33, 2e-20),
    ]

    wavelengths = np.linspace(400e-9, 1600e-9, 100)

    for name, n0_mat, n2_mat in materials:
        P_cr_wl = 3.77 * wavelengths**2 / (8 * np.pi * n0_mat * n2_mat)
        ax3.semilogy(wavelengths * 1e9, P_cr_wl * 1e-6, linewidth=2,
                     label=f'{name}')

    ax3.set_xlabel('Wavelength (nm)')
    ax3.set_ylabel('Critical power P_cr (MW)')
    ax3.set_title(r'Critical Power: $P_{cr} = 3.77\lambda^2 / (8\pi n_0 n_2)$')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(400, 1600)

    # Plot 4: Self-focusing vs diffraction
    ax4 = axes[1, 1]

    # Power ratios
    P_ratios = [0.5, 0.9, 1.0, 1.5, 3.0]
    z = np.linspace(0, 5, 500)  # Normalized to Rayleigh range

    for P_ratio in P_ratios:
        if P_ratio < 1:
            # Diffraction dominates but reduced
            z_R_eff = 1 / np.sqrt(1 - P_ratio)
            w = np.sqrt(1 + (z / z_R_eff)**2)
            label = f'P/P_cr = {P_ratio:.1f}'
            ax4.plot(z, w, linewidth=2, label=label)

        elif P_ratio == 1:
            # Critical power - neither focuses nor defocuses
            w = np.ones_like(z)
            ax4.plot(z, w, 'k--', linewidth=2, label='P/P_cr = 1.0 (critical)')

        else:
            # Self-focusing
            z_f = 0.367 / np.sqrt((np.sqrt(P_ratio) - 0.858)**2 - 0.0219)
            z_valid = z < z_f * 0.95
            w = np.ones_like(z)
            w[z_valid] = np.sqrt(1 - (z[z_valid] / z_f)**2)
            w[~z_valid] = 0.1  # Collapsed

            ax4.plot(z, w, linewidth=2, label=f'P/P_cr = {P_ratio:.1f}')
            ax4.axvline(z_f, color='red', linestyle=':', alpha=0.5)

    ax4.axhline(1.0, color='gray', linestyle='--', alpha=0.5)

    ax4.set_xlabel('Propagation distance z/z_R')
    ax4.set_ylabel('Normalized beam width w/w0')
    ax4.set_title('Beam Evolution: Self-Focusing vs Diffraction\n'
                  'P > P_cr leads to collapse')
    ax4.legend(fontsize=9, loc='upper right')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 5)
    ax4.set_ylim(0, 3)

    plt.tight_layout()
    return fig


def plot_beam_collapse():
    """Plot beam collapse dynamics"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    wavelength = 800e-9
    n0 = 1.5
    n2 = 3e-20
    w0 = 100e-6  # 100 um waist

    kerr = KerrMedium(n0, n2, wavelength)
    P_cr = kerr.critical_power()

    # Plot 1: Collapse distance vs power
    ax1 = axes[0, 0]

    P_ratio = np.linspace(1.01, 10, 100)

    # Marburger formula for collapse distance
    z_f = np.zeros_like(P_ratio)
    z_R = np.pi * w0**2 / wavelength

    for i, Pr in enumerate(P_ratio):
        sqrt_Pr = np.sqrt(Pr)
        if (sqrt_Pr - 0.858)**2 > 0.0219:
            z_f[i] = 0.367 * z_R / np.sqrt((sqrt_Pr - 0.858)**2 - 0.0219)
        else:
            z_f[i] = np.inf

    ax1.plot(P_ratio, z_f * 1e2, 'b-', linewidth=2)
    ax1.axhline(z_R * 1e2, color='gray', linestyle='--', alpha=0.5,
                label=f'Rayleigh range = {z_R*1e2:.1f} cm')

    ax1.set_xlabel('Power ratio P/P_cr')
    ax1.set_ylabel('Collapse distance z_f (cm)')
    ax1.set_title('Self-Focusing Collapse Distance (Marburger Formula)\n'
                  r'$z_f = 0.367 z_R / \sqrt{(\sqrt{P/P_{cr}} - 0.858)^2 - 0.0219}$')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, 10)
    ax1.set_ylim(0, z_R * 5e2)

    # Plot 2: Intensity evolution
    ax2 = axes[0, 1]

    z_norm = np.linspace(0, 0.9, 500)  # Normalized to z_f

    for P_ratio_val in [1.5, 2, 3, 5]:
        # Approximate intensity increase as 1/w^2
        # w ~ sqrt(1 - (z/z_f)^2) approximately
        w_sq = 1 - z_norm**2
        I_norm = 1 / np.maximum(w_sq, 0.01)

        ax2.semilogy(z_norm, I_norm, linewidth=2,
                     label=f'P/P_cr = {P_ratio_val}')

    ax2.set_xlabel('Propagation distance z/z_f')
    ax2.set_ylabel('Normalized intensity I/I_0')
    ax2.set_title('Intensity Amplification During Collapse\n'
                  'Intensity diverges as beam collapses')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 0.9)
    ax2.set_ylim(1, 1000)

    # Plot 3: 2D beam profile evolution
    ax3 = axes[1, 0]

    sim = SelfFocusingSimulation(wavelength, n0, n2, w0)

    power = 3 * P_cr
    z_max = 0.3  # meters

    z, x, I_profiles, w, I_peak = sim.simulate(power, z_max, n_z=100, n_xy=128)

    # Normalize
    I_profiles_norm = I_profiles / I_profiles[0, :].max()

    im = ax3.imshow(I_profiles_norm.T, extent=[z.min()*100, z.max()*100,
                                                x.min()*1e3, x.max()*1e3],
                    aspect='auto', cmap='hot', origin='lower')
    plt.colorbar(im, ax=ax3, label='Normalized intensity')

    ax3.set_xlabel('Propagation distance z (cm)')
    ax3.set_ylabel('Radial position x (mm)')
    ax3.set_title(f'Beam Profile Evolution (P = {power/P_cr:.0f}*P_cr)\n'
                  'Beam focuses and intensity increases')

    # Plot 4: Beam width and peak intensity
    ax4 = axes[1, 1]

    ax4_twin = ax4.twinx()

    ax4.plot(z * 100, w * 1e6, 'b-', linewidth=2, label='Beam width')
    ax4_twin.plot(z * 100, I_peak / I_peak[0], 'r-', linewidth=2, label='Peak intensity')

    ax4.set_xlabel('Propagation distance z (cm)')
    ax4.set_ylabel('Beam width (um)', color='blue')
    ax4_twin.set_ylabel('Normalized peak intensity', color='red')

    ax4.tick_params(axis='y', labelcolor='blue')
    ax4_twin.tick_params(axis='y', labelcolor='red')

    ax4.set_title('Beam Width and Peak Intensity vs Propagation\n'
                  'Width decreases, intensity increases')
    ax4.grid(True, alpha=0.3)

    # Combine legends
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    return fig


def plot_kerr_lens_mode_locking():
    """Plot Kerr lens mode locking concept"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Kerr lens in laser cavity
    ax1 = axes[0, 0]

    # Draw simplified laser cavity
    ax1.axhline(0, color='gray', linestyle='-', linewidth=1)

    # Mirrors
    ax1.plot([0, 0], [-1, 1], 'b-', linewidth=3)
    ax1.plot([5, 5], [-1, 1], 'b-', linewidth=3)
    ax1.text(0, -1.3, 'Mirror 1', ha='center')
    ax1.text(5, -1.3, 'Mirror 2', ha='center')

    # Kerr medium
    rect = plt.Rectangle((2, -0.5), 1, 1, fill=True, facecolor='lightblue',
                          edgecolor='blue', linewidth=2)
    ax1.add_patch(rect)
    ax1.text(2.5, -0.8, 'Kerr\nmedium', ha='center', fontsize=9)

    # Aperture
    ax1.plot([4, 4], [-0.4, -0.15], 'k-', linewidth=3)
    ax1.plot([4, 4], [0.15, 0.4], 'k-', linewidth=3)
    ax1.text(4, -0.6, 'Aperture', ha='center', fontsize=9)

    # Beam profiles
    # CW mode (wide beam)
    z_cw = np.linspace(0, 5, 100)
    w_cw = 0.3 + 0.1 * np.sin(z_cw * np.pi / 2.5)**2
    ax1.fill_between(z_cw, -w_cw, w_cw, alpha=0.3, color='red', label='CW (low intensity)')

    # Mode-locked (focused)
    w_ml = 0.2 + 0.05 * np.sin(z_cw * np.pi / 2.5)**2
    # Extra focusing in Kerr medium
    w_ml[40:60] *= 0.7
    ax1.fill_between(z_cw, -w_ml, w_ml, alpha=0.3, color='green', label='Mode-locked (high intensity)')

    ax1.set_xlim(-0.5, 5.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect('equal')
    ax1.set_title('Kerr Lens Mode Locking (KLM)\n'
                  'High intensity -> self-focusing -> better transmission through aperture')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.axis('off')

    # Plot 2: Self-amplitude modulation
    ax2 = axes[0, 1]

    time = np.linspace(-2, 2, 500)

    # Pulse shape
    pulse = np.exp(-time**2)

    # Transmission depends on intensity (Kerr focusing)
    # Higher intensity -> better transmission through aperture
    T_baseline = 0.5
    T_kerr = T_baseline + 0.4 * pulse**2

    ax2.plot(time, pulse, 'b-', linewidth=2, label='Input pulse')
    ax2.plot(time, T_kerr * pulse, 'r-', linewidth=2, label='After Kerr + aperture')
    ax2.plot(time, T_kerr, 'g--', linewidth=2, label='Transmission T(I)')

    ax2.axhline(T_baseline, color='gray', linestyle=':', alpha=0.5)
    ax2.axhline(0.9, color='gray', linestyle=':', alpha=0.5)

    ax2.set_xlabel('Time (normalized)')
    ax2.set_ylabel('Intensity / Transmission')
    ax2.set_title('Self-Amplitude Modulation\n'
                  'Peak sees higher transmission -> pulse shortening')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Pulse shortening
    ax3 = axes[1, 0]

    n_roundtrips = [0, 10, 50, 100, 200]

    for n in n_roundtrips:
        # Pulse shortens with each round trip
        width = 1 / (1 + 0.02 * n)
        pulse = np.exp(-time**2 / width**2)

        ax3.plot(time, pulse + 0.2 * n_roundtrips.index(n), linewidth=2,
                 label=f'n = {n} round trips')

    ax3.set_xlabel('Time (normalized)')
    ax3.set_ylabel('Intensity (offset)')
    ax3.set_title('Pulse Evolution in KLM Laser\n'
                  'Pulses shorten with each round trip until limited by bandwidth')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Plot 4: KLM applications
    ax4 = axes[1, 1]

    applications = [
        ('Ti:Sapphire\nOscillators', 'Sub-10 fs pulses'),
        ('Yb:doped\nLasers', '< 50 fs pulses'),
        ('Cr:LiSAF\nLasers', 'Compact femtosecond'),
        ('Fiber\nLasers', 'Nonlinear loop mirrors'),
    ]

    y_pos = np.arange(len(applications))
    labels = [a[0] for a in applications]
    details = [a[1] for a in applications]

    ax4.barh(y_pos, [100, 80, 60, 70], color=['red', 'blue', 'green', 'orange'], alpha=0.7)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(labels)
    ax4.set_xlabel('Relative importance (%)')
    ax4.set_title('Kerr Lens Mode Locking Applications')

    for i, detail in enumerate(details):
        ax4.text(10, i, detail, va='center', fontsize=10)

    # Add note about parameters
    note = """KLM Parameters:
- Critical power: P_cr ~ 3 MW (silica at 800nm)
- n2 (silica) ~ 2.5e-20 m^2/W
- Typical pulse: ~50 fs, 5 nJ
- Peak power: ~100 kW"""
    ax4.text(0.6, 0.05, note, transform=ax4.transAxes, fontsize=9,
            verticalalignment='bottom', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    return fig


def plot_spatial_effects():
    """Plot spatial Kerr effect phenomena"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    wavelength = 800e-9
    n0 = 1.5
    n2 = 3e-20
    w0 = 100e-6

    kerr = KerrMedium(n0, n2, wavelength)
    P_cr = kerr.critical_power()

    # Plot 1: Self-phase modulation (radial)
    ax1 = axes[0, 0]

    r = np.linspace(-3, 3, 500)  # Normalized to beam radius

    powers = [0.1, 0.5, 1.0, 2.0]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(powers)))

    for P_ratio, color in zip(powers, colors):
        # Gaussian intensity
        I_norm = np.exp(-2 * r**2)

        # Phase shift
        # phi = k * n2 * I * L
        # Normalize to peak phase shift
        phi = P_ratio * I_norm * np.pi  # Peak is P_ratio * pi

        ax1.plot(r, phi, color=color, linewidth=2,
                 label=f'P/P_cr = {P_ratio}')

    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)

    ax1.set_xlabel('Radial position r/w0')
    ax1.set_ylabel('Nonlinear phase (rad)')
    ax1.set_title('Self-Phase Modulation (Spatial)\n'
                  'Higher intensity at center -> larger phase shift')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Wave front curvature
    ax2 = axes[0, 1]

    for P_ratio, color in zip(powers, colors):
        # Phase curvature creates equivalent lens
        I_norm = np.exp(-2 * r**2)
        phi = P_ratio * I_norm * np.pi

        # Wavefront
        wavefront = -phi / (2 * np.pi)  # In wavelengths

        ax2.plot(r, wavefront, color=color, linewidth=2,
                 label=f'P/P_cr = {P_ratio}')

    ax2.set_xlabel('Radial position r/w0')
    ax2.set_ylabel('Wavefront (wavelengths)')
    ax2.set_title('Induced Wavefront Curvature\n'
                  'Creates equivalent positive lens (converging)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Filamentation
    ax3 = axes[1, 0]

    # For very high power, beam breaks up into filaments
    x = np.linspace(-5, 5, 200)
    y = np.linspace(-5, 5, 200)
    X, Y = np.meshgrid(x, y)

    # Initial Gaussian
    I_initial = np.exp(-2 * (X**2 + Y**2))

    # Filaments (simplified model - random modulation)
    np.random.seed(42)
    modulation = 1 + 0.3 * np.random.randn(200, 200)
    from scipy.ndimage import gaussian_filter
    modulation = gaussian_filter(modulation, sigma=10)

    I_filament = I_initial * modulation**2
    I_filament = gaussian_filter(I_filament, sigma=3)  # Smooth

    im = ax3.imshow(I_filament, extent=[x.min(), x.max(), y.min(), y.max()],
                    cmap='hot', origin='lower')
    plt.colorbar(im, ax=ax3, label='Intensity')

    ax3.set_xlabel('x / w0')
    ax3.set_ylabel('y / w0')
    ax3.set_title('Beam Filamentation (P >> P_cr)\n'
                  'Beam breaks up due to modulation instability')

    # Plot 4: Power threshold diagram
    ax4 = axes[1, 1]

    P_over_Pcr = np.linspace(0, 5, 500)

    # Different regimes
    linear = P_over_Pcr < 0.1
    weakly_nl = (P_over_Pcr >= 0.1) & (P_over_Pcr < 1)
    critical = (P_over_Pcr >= 1) & (P_over_Pcr < 2)
    collapse = (P_over_Pcr >= 2) & (P_over_Pcr < 4)
    filament = P_over_Pcr >= 4

    ax4.axvspan(0, 0.1, alpha=0.3, color='green', label='Linear')
    ax4.axvspan(0.1, 1, alpha=0.3, color='blue', label='Weakly nonlinear')
    ax4.axvspan(1, 2, alpha=0.3, color='yellow', label='Self-focusing')
    ax4.axvspan(2, 4, alpha=0.3, color='orange', label='Collapse')
    ax4.axvspan(4, 5, alpha=0.3, color='red', label='Filamentation')

    ax4.axvline(1, color='black', linestyle='--', linewidth=2)
    ax4.text(1, 0.5, 'P_cr', fontsize=12, ha='center')

    ax4.set_xlabel('Power P / P_cr')
    ax4.set_ylabel('')
    ax4.set_title('Regimes of Kerr-Induced Self-Focusing')
    ax4.legend(loc='upper right')
    ax4.set_xlim(0, 5)
    ax4.set_ylim(0, 1)
    ax4.set_yticks([])

    plt.tight_layout()
    return fig


def main():
    """Main function to demonstrate Kerr lens effects"""

    # Create figures
    fig1 = plot_kerr_effect_basics()
    fig2 = plot_beam_collapse()
    fig3 = plot_kerr_lens_mode_locking()
    fig4 = plot_spatial_effects()

    # Save plots
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)

    fig1.savefig(os.path.join(output_dir, 'kerr_lens.png'),
                 dpi=150, bbox_inches='tight')
    fig2.savefig(os.path.join(output_dir, 'kerr_collapse.png'),
                 dpi=150, bbox_inches='tight')
    fig3.savefig(os.path.join(output_dir, 'kerr_mode_locking.png'),
                 dpi=150, bbox_inches='tight')
    fig4.savefig(os.path.join(output_dir, 'kerr_spatial.png'),
                 dpi=150, bbox_inches='tight')

    print(f"Plots saved to {output_dir}/kerr_*.png")

    # Print analysis
    print("\n=== Kerr Lens Analysis ===")

    materials = [
        ('Silica (glass)', 1.45, 2.5e-20),
        ('Air (1 atm)', 1.0, 3e-23),
        ('CS2', 1.63, 3e-18),
        ('Water', 1.33, 2e-20),
    ]

    wavelength = 800e-9

    print(f"\nCritical powers at lambda = {wavelength*1e9:.0f} nm:")
    print("-" * 50)
    print(f"{'Material':<20} {'n2 (m^2/W)':<15} {'P_cr (MW)':<10}")
    print("-" * 50)

    for name, n0, n2 in materials:
        kerr = KerrMedium(n0, n2, wavelength)
        P_cr = kerr.critical_power()
        print(f"{name:<20} {n2:<15.0e} {P_cr*1e-6:<10.3f}")

    print("\nKey equations:")
    print("  Kerr effect: n(I) = n0 + n2*I")
    print("  Critical power: P_cr = 3.77*lambda^2 / (8*pi*n0*n2)")
    print("  Collapse distance: z_f = 0.367*z_R / sqrt((sqrt(P/P_cr) - 0.858)^2 - 0.0219)")

    print("\nApplications:")
    print("  - Kerr lens mode locking (KLM) for ultrashort pulses")
    print("  - Self-focusing for laser materials processing")
    print("  - Optical limiting for eye protection")
    print("  - All-optical switching")


if __name__ == "__main__":
    main()
