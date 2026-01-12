"""
Example 113: Coherence Length and Fringe Visibility

This example demonstrates temporal coherence, showing how the coherence length
of a light source affects fringe visibility in interferometers.

Physics:
    Coherence length: L_c = c * tau_c = c / (Delta_nu) = lambda^2 / Delta_lambda

    For a Gaussian spectrum:
    g(tau) = exp(-pi * (tau/tau_c)^2) * exp(-i*2*pi*f_0*tau)

    Fringe visibility: V = |g(tau)| where tau = Delta_L / c

    The visibility decreases as path difference increases beyond coherence length.

    For different spectral shapes:
    - Lorentzian: V = exp(-|tau|/tau_c)
    - Gaussian: V = exp(-(pi*tau/tau_c)^2)
    - Rectangular: V = sinc(tau/tau_c)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


class CoherentSource:
    """Light source with spectral properties affecting coherence"""

    def __init__(self, center_wavelength: float, linewidth: float,
                 spectrum_type: str = 'gaussian'):
        """
        Args:
            center_wavelength: Center wavelength (m)
            linewidth: Spectral linewidth FWHM (m or Hz depending on type)
            spectrum_type: 'gaussian', 'lorentzian', or 'rectangular'
        """
        self.lambda_0 = center_wavelength
        self.c = 299792458.0
        self.f_0 = self.c / center_wavelength
        self.spectrum_type = spectrum_type

        # Convert wavelength linewidth to frequency linewidth
        # Delta_f ≈ c * Delta_lambda / lambda^2
        self.delta_lambda = linewidth
        self.delta_f = self.c * linewidth / center_wavelength**2

        # Coherence time and length
        if spectrum_type == 'gaussian':
            # For Gaussian: tau_c = 1/(pi * Delta_f) for FWHM definition
            self.tau_c = 1.0 / (np.pi * self.delta_f)
        elif spectrum_type == 'lorentzian':
            # For Lorentzian: tau_c = 1/(pi * Delta_f)
            self.tau_c = 1.0 / (np.pi * self.delta_f)
        else:  # rectangular
            self.tau_c = 1.0 / self.delta_f

        self.L_c = self.c * self.tau_c

    def spectrum(self, frequency: np.ndarray) -> np.ndarray:
        """Calculate spectral density S(f)"""
        f_rel = frequency - self.f_0

        if self.spectrum_type == 'gaussian':
            # Gaussian: S(f) = exp(-4*ln(2)*(f-f0)^2/Delta_f^2)
            sigma_f = self.delta_f / (2 * np.sqrt(2 * np.log(2)))
            return np.exp(-f_rel**2 / (2 * sigma_f**2))

        elif self.spectrum_type == 'lorentzian':
            # Lorentzian: S(f) = (Delta_f/2)^2 / ((f-f0)^2 + (Delta_f/2)^2)
            gamma = self.delta_f / 2
            return gamma**2 / (f_rel**2 + gamma**2)

        else:  # rectangular
            return np.where(np.abs(f_rel) < self.delta_f / 2, 1.0, 0.0)

    def coherence_function(self, tau: np.ndarray) -> np.ndarray:
        """
        Calculate normalized coherence function g(tau).
        This is the Fourier transform of the spectral density.
        """
        if self.spectrum_type == 'gaussian':
            # FT of Gaussian is Gaussian
            return np.exp(-(np.pi * tau / self.tau_c)**2)

        elif self.spectrum_type == 'lorentzian':
            # FT of Lorentzian is exponential
            return np.exp(-np.abs(tau) / self.tau_c)

        else:  # rectangular
            # FT of rectangular is sinc
            arg = np.pi * tau * self.delta_f
            return np.where(np.abs(arg) > 1e-10, np.sin(arg) / arg, 1.0)

    def visibility(self, path_difference: np.ndarray) -> np.ndarray:
        """Calculate fringe visibility for given path difference"""
        tau = path_difference / self.c
        return np.abs(self.coherence_function(tau))


class Interferometer:
    """Michelson or Mach-Zehnder interferometer"""

    def __init__(self, source: CoherentSource):
        """
        Args:
            source: Light source with coherence properties
        """
        self.source = source

    def interference_pattern(self, path_difference: float,
                            n_points: int = 200) -> tuple:
        """
        Calculate interference pattern with finite coherence.

        Args:
            path_difference: Optical path difference (m)
            n_points: Number of phase points

        Returns:
            (phase, intensity) arrays
        """
        # Phase from path difference
        phi = np.linspace(0, 4*np.pi, n_points)

        # Visibility at this path difference
        V = self.source.visibility(np.array([path_difference]))[0]

        # Intensity: I = I_0 * (1 + V * cos(phi + phi_0))
        # where phi_0 = 2*pi*f_0*Delta_L/c
        phi_0 = 2 * np.pi * path_difference / self.source.lambda_0

        intensity = 1 + V * np.cos(phi + phi_0)

        return phi, intensity, V


def plot_coherence_concepts():
    """Plot basic concepts of temporal coherence"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Spectral line shapes
    ax1 = axes[0, 0]

    lambda_0 = 632.8e-9  # HeNe wavelength
    delta_lambda = 0.002e-9  # 2 pm linewidth

    sources = [
        CoherentSource(lambda_0, delta_lambda, 'gaussian'),
        CoherentSource(lambda_0, delta_lambda, 'lorentzian'),
        CoherentSource(lambda_0, delta_lambda, 'rectangular'),
    ]
    labels = ['Gaussian', 'Lorentzian', 'Rectangular']
    colors = ['blue', 'red', 'green']

    f_center = sources[0].f_0
    f_range = np.linspace(f_center - 5*sources[0].delta_f,
                         f_center + 5*sources[0].delta_f, 500)

    for source, label, color in zip(sources, labels, colors):
        S = source.spectrum(f_range)
        ax1.plot((f_range - f_center)/1e9, S, color=color, linewidth=2,
                label=f'{label}, L_c = {source.L_c*100:.1f} cm')

    ax1.set_xlabel('Frequency detuning (GHz)')
    ax1.set_ylabel('Spectral density (normalized)')
    ax1.set_title('Spectral Line Shapes')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Coherence functions
    ax2 = axes[0, 1]

    tau_max = 3 * max(s.tau_c for s in sources)
    tau = np.linspace(-tau_max, tau_max, 500)

    for source, label, color in zip(sources, labels, colors):
        g = source.coherence_function(tau)
        ax2.plot(tau / source.tau_c, g, color=color, linewidth=2, label=label)

    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(-1, color='gray', linestyle=':', alpha=0.5)
    ax2.axvline(1, color='gray', linestyle=':', alpha=0.5)

    ax2.set_xlabel('Normalized delay tau/tau_c')
    ax2.set_ylabel('Coherence function |g(tau)|')
    ax2.set_title('Normalized Coherence Functions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Visibility vs path difference
    ax3 = axes[1, 0]

    L_range = np.linspace(0, 3 * sources[0].L_c, 500)

    for source, label, color in zip(sources, labels, colors):
        V = source.visibility(L_range)
        ax3.plot(L_range / source.L_c, V, color=color, linewidth=2, label=label)

    ax3.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='V = 0.5')
    ax3.axhline(1/np.e, color='gray', linestyle=':', alpha=0.5, label='V = 1/e')
    ax3.axvline(1, color='black', linestyle='--', alpha=0.5)

    ax3.set_xlabel('Path difference / coherence length')
    ax3.set_ylabel('Fringe visibility')
    ax3.set_title('Visibility vs Path Difference')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 3)

    # Plot 4: Interference patterns at different path differences
    ax4 = axes[1, 1]

    source = CoherentSource(lambda_0, delta_lambda, 'gaussian')
    interf = Interferometer(source)

    path_differences = [0, 0.5 * source.L_c, source.L_c, 2 * source.L_c]
    colors = plt.cm.viridis(np.linspace(0, 1, len(path_differences)))

    for delta_L, color in zip(path_differences, colors):
        phi, I, V = interf.interference_pattern(delta_L)
        ax4.plot(phi / np.pi, I, color=color, linewidth=2,
                label=f'Delta_L = {delta_L/source.L_c:.1f} L_c, V = {V:.2f}')

    ax4.set_xlabel('Phase (units of pi)')
    ax4.set_ylabel('Intensity (normalized)')
    ax4.set_title('Interference Patterns at Different Path Differences')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 4)

    plt.tight_layout()
    return fig


def plot_light_source_comparison():
    """Compare coherence properties of different light sources"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    c = 299792458.0

    # Different light sources
    sources_params = [
        ('HeNe laser', 632.8e-9, 1.5e9, 'lorentzian'),       # 1.5 GHz linewidth
        ('Diode laser', 780e-9, 50e9, 'lorentzian'),          # 50 GHz linewidth
        ('LED', 630e-9, 20e-9, 'gaussian'),                    # 20 nm = 15 THz
        ('Sodium lamp', 589e-9, 0.5e-9, 'lorentzian'),        # D line
        ('White light', 550e-9, 300e-9, 'gaussian'),          # ~300 nm bandwidth
    ]

    # Convert wavelength linewidth to frequency linewidth where needed
    sources = []
    labels = []
    for name, lambda_0, linewidth, shape in sources_params:
        if linewidth > 1e-6:  # Already in Hz
            delta_f = linewidth
            delta_lambda = lambda_0**2 * delta_f / c
        else:  # In meters (wavelength)
            delta_lambda = linewidth
            delta_f = c * linewidth / lambda_0**2

        # Create source with wavelength linewidth
        source = CoherentSource(lambda_0, delta_lambda, shape)
        sources.append(source)
        labels.append(name)

    # Plot 1: Coherence lengths comparison
    ax1 = axes[0, 0]

    L_c_values = [s.L_c for s in sources]
    colors = plt.cm.tab10(np.arange(len(sources)))

    bars = ax1.barh(np.arange(len(sources)), L_c_values, color=colors)

    ax1.set_yticks(np.arange(len(sources)))
    ax1.set_yticklabels(labels)
    ax1.set_xlabel('Coherence length')
    ax1.set_title('Coherence Lengths of Different Light Sources')
    ax1.set_xscale('log')

    # Add value labels
    for bar, L_c in zip(bars, L_c_values):
        if L_c >= 1:
            label = f'{L_c:.1f} m'
        elif L_c >= 1e-3:
            label = f'{L_c*1e3:.1f} mm'
        else:
            label = f'{L_c*1e6:.1f} um'
        ax1.text(bar.get_width() * 1.1, bar.get_y() + bar.get_height()/2,
                label, va='center', fontsize=9)

    # Plot 2: Visibility curves for each source
    ax2 = axes[0, 1]

    for source, label, color in zip(sources, labels, colors):
        L_range = np.linspace(0, min(3 * source.L_c, 1), 500)
        V = source.visibility(L_range)
        ax2.plot(L_range * 1e3, V, color=color, linewidth=2, label=label)

    ax2.set_xlabel('Path difference (mm)')
    ax2.set_ylabel('Fringe visibility')
    ax2.set_title('Visibility vs Path Difference')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)

    # Plot 3: Coherence time vs linewidth
    ax3 = axes[1, 0]

    delta_f_range = np.logspace(6, 15, 100)  # 1 MHz to 1 PHz
    tau_c = 1.0 / (np.pi * delta_f_range)

    ax3.loglog(delta_f_range, tau_c, 'b-', linewidth=2)

    # Mark source positions
    for source, label, color in zip(sources, labels, colors):
        ax3.plot(source.delta_f, source.tau_c, 'o', color=color,
                markersize=10, label=label)

    ax3.set_xlabel('Linewidth (Hz)')
    ax3.set_ylabel('Coherence time (s)')
    ax3.set_title('Coherence Time vs Spectral Linewidth\n(tau_c = 1/(pi*Delta_f))')
    ax3.legend(fontsize=8, loc='upper right')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Number of coherent oscillations
    ax4 = axes[1, 1]

    # N_coh = L_c / lambda = f_0 * tau_c
    N_coh_values = [s.f_0 * s.tau_c for s in sources]

    bars = ax4.barh(np.arange(len(sources)), N_coh_values, color=colors)

    ax4.set_yticks(np.arange(len(sources)))
    ax4.set_yticklabels(labels)
    ax4.set_xlabel('Number of coherent oscillations')
    ax4.set_title('Coherent Wave Cycles (N = L_c / lambda)')
    ax4.set_xscale('log')

    for bar, N in zip(bars, N_coh_values):
        label = f'{N:.0e}'
        ax4.text(bar.get_width() * 1.1, bar.get_y() + bar.get_height()/2,
                label, va='center', fontsize=9)

    plt.tight_layout()
    return fig


def plot_interferometer_demonstration():
    """Demonstrate visibility measurement in interferometer"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Simulate Michelson interferometer with HeNe laser
    lambda_0 = 632.8e-9
    delta_lambda = 1e-12  # 1 pm linewidth (high coherence)

    source = CoherentSource(lambda_0, delta_lambda, 'lorentzian')
    interf = Interferometer(source)

    print(f"HeNe laser coherence length: {source.L_c:.2f} m")

    # Plot 1: Interferogram at different path differences
    ax1 = axes[0, 0]

    mirror_positions = np.linspace(0, 2*source.L_c, 8)

    for i, delta_L in enumerate(mirror_positions):
        phi, I, V = interf.interference_pattern(delta_L, n_points=100)
        offset = i * 0.5
        ax1.plot(phi / np.pi, I + offset, 'b-', linewidth=1.5)
        ax1.text(4.2, 1 + offset, f'V={V:.2f}', fontsize=9)

    ax1.set_xlabel('Phase (units of pi)')
    ax1.set_ylabel('Intensity + offset')
    ax1.set_title(f'Michelson Interferometer Fringes\n'
                 f'HeNe laser, L_c = {source.L_c:.2f} m')
    ax1.set_xlim(0, 4.5)

    # Plot 2: Visibility measurement
    ax2 = axes[0, 1]

    L_range = np.linspace(0, 2*source.L_c, 100)
    V_measured = source.visibility(L_range)

    ax2.plot(L_range * 100, V_measured, 'b-', linewidth=2, label='Measured visibility')
    ax2.axhline(1/np.e, color='red', linestyle='--', label='V = 1/e')
    ax2.axvline(source.L_c * 100, color='green', linestyle='--',
               label=f'L_c = {source.L_c*100:.1f} cm')

    ax2.set_xlabel('Path difference (cm)')
    ax2.set_ylabel('Visibility')
    ax2.set_title('Visibility Curve Measurement')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Effect of source bandwidth on white-light fringes
    ax3 = axes[1, 0]

    # White light source
    delta_lambda_white = 300e-9  # 300 nm bandwidth
    white_source = CoherentSource(550e-9, delta_lambda_white, 'gaussian')

    print(f"White light coherence length: {white_source.L_c*1e6:.2f} um")

    L_range_white = np.linspace(-5*white_source.L_c, 5*white_source.L_c, 500)
    V_white = white_source.visibility(np.abs(L_range_white))

    # Simulate RGB components
    wavelengths = [450e-9, 550e-9, 650e-9]  # B, G, R
    colors = ['blue', 'green', 'red']

    for wl, color in zip(wavelengths, colors):
        # Phase varies with wavelength
        phi = 2 * np.pi * L_range_white / wl
        I = 1 + V_white * np.cos(phi)
        ax3.plot(L_range_white * 1e6, I, color=color, linewidth=1, alpha=0.5)

    # Combined intensity (white light pattern)
    I_total = np.zeros_like(L_range_white)
    for wl in wavelengths:
        phi = 2 * np.pi * L_range_white / wl
        I_total += 1 + V_white * np.cos(phi)
    I_total /= 3

    ax3.plot(L_range_white * 1e6, I_total, 'k-', linewidth=2, label='White light')

    ax3.set_xlabel('Path difference (um)')
    ax3.set_ylabel('Intensity')
    ax3.set_title(f'White Light Interference\nL_c = {white_source.L_c*1e6:.2f} um')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Extracting linewidth from visibility curve
    ax4 = axes[1, 1]

    # Simulated measurement with noise
    L_measured = np.linspace(0, 0.5, 50)  # 0 to 50 cm
    V_true = source.visibility(L_measured)
    V_noisy = V_true + 0.02 * np.random.randn(len(V_true))
    V_noisy = np.clip(V_noisy, 0, 1)

    ax4.plot(L_measured * 100, V_noisy, 'bo', markersize=6, label='Measured')
    ax4.plot(L_measured * 100, V_true, 'r-', linewidth=2, label='Fit (Lorentzian)')

    # Extract coherence length (where V = 1/e)
    L_c_extracted = source.L_c
    ax4.axhline(1/np.e, color='gray', linestyle='--')
    ax4.axvline(L_c_extracted * 100, color='green', linestyle='--')
    ax4.plot(L_c_extracted * 100, 1/np.e, 'g*', markersize=15,
            label=f'L_c = {L_c_extracted*100:.1f} cm')

    # Calculate linewidth from L_c
    delta_lambda_extracted = source.lambda_0**2 / (np.pi * L_c_extracted)
    ax4.text(0.05, 0.15, f'Extracted linewidth:\nDelta_lambda = {delta_lambda_extracted*1e12:.2f} pm',
            transform=ax4.transAxes, fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))

    ax4.set_xlabel('Path difference (cm)')
    ax4.set_ylabel('Visibility')
    ax4.set_title('Extracting Source Linewidth from Visibility Curve')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    """Main function to demonstrate coherence length and visibility"""

    # Create figures
    fig1 = plot_coherence_concepts()
    fig2 = plot_light_source_comparison()
    fig3 = plot_interferometer_demonstration()

    # Save plots
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)

    fig1.savefig(os.path.join(output_dir, 'coherence_concepts.png'),
                 dpi=150, bbox_inches='tight')
    fig2.savefig(os.path.join(output_dir, 'coherence_sources.png'),
                 dpi=150, bbox_inches='tight')
    fig3.savefig(os.path.join(output_dir, 'coherence_interferometer.png'),
                 dpi=150, bbox_inches='tight')

    print(f"Plots saved to {output_dir}/coherence_*.png")

    # Print analysis
    print("\n=== Coherence Length Analysis ===")
    print("\nRelationships:")
    print("  L_c = c * tau_c = lambda^2 / Delta_lambda")
    print("  tau_c = 1 / (pi * Delta_f) for Lorentzian")
    print("  V(Delta_L) = |g(Delta_L/c)| = exp(-Delta_L/L_c) for Lorentzian")

    print("\nTypical coherence lengths:")
    sources = [
        ('Single-mode HeNe', 632.8e-9, 1e-12),
        ('Multimode HeNe', 632.8e-9, 5e-12),
        ('Diode laser', 780e-9, 0.1e-9),
        ('LED', 630e-9, 20e-9),
        ('White light', 550e-9, 300e-9),
    ]

    for name, lambda_0, delta_lambda in sources:
        L_c = lambda_0**2 / delta_lambda
        if L_c >= 1:
            print(f"  {name}: L_c = {L_c:.2f} m")
        elif L_c >= 1e-3:
            print(f"  {name}: L_c = {L_c*1e3:.2f} mm")
        else:
            print(f"  {name}: L_c = {L_c*1e6:.2f} um")


if __name__ == "__main__":
    main()
