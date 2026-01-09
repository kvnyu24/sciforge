"""
Example demonstrating a Michelson interferometer simulation.

This example shows how a laser beam is split and recombined to create
interference patterns, including effects of:
- Path length differences 
- Mirror misalignment and surface imperfections
- Air turbulence and refractive index variations
- Temporal and spatial coherence
- Fringe visibility and contrast
- Beam divergence and mode structure
- Thermal effects and mechanical stability
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics import HermitePolynomial, RiceDistribution
from matplotlib.animation import FuncAnimation
class MichelsonInterferometer:
    """Class representing a sophisticated Michelson interferometer setup"""
    
    def __init__(self,
                 wavelength: float,      # Laser wavelength (m)
                 coherence_length: float, # Coherence length (m)
                 beam_radius: float,      # Initial beam radius (m)
                 arm_length1: float,      # Length of first arm (m) 
                 arm_length2: float,      # Length of second arm (m)
                 M2: float = 1.1,         # Beam quality factor
                 nx: int = 200,           # Grid points in x
                 ny: int = 200):          # Grid points in y
        
        # Store parameters
        self.wavelength = wavelength
        self.coherence_length = coherence_length
        self.beam_radius = beam_radius
        self.arm_length1 = arm_length1
        self.arm_length2 = arm_length2
        self.M2 = M2  # Beam quality factor (M² = 1 for perfect Gaussian)

        # Create high-resolution spatial grid
        x_max = 8 * beam_radius  # Wider field of view
        self.x = np.linspace(-x_max, x_max, nx)
        self.y = np.linspace(-x_max, x_max, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.R = np.sqrt(self.X**2 + self.Y**2)
        self.Theta = np.arctan2(self.Y, self.X)

        # Initialize laser beam with Hermite-Gaussian modes
        k = 2*np.pi/wavelength
        self.k = k
        self.z_R = np.pi * beam_radius**2 / wavelength  # Rayleigh range
        
        # Generate superposition of modes
        self.E0 = self._generate_laser_modes()
        
        # Initialize mirror parameters with surface roughness
        self.mirror1_angle = 0.0
        self.mirror2_angle = 0.0
        self.surface_roughness = wavelength/20  # λ/20 surface quality
        self._generate_mirror_surfaces()
        
        # Initialize environmental parameters
        self.time = 0.0
        self.temperature = 293.15  # Room temperature (K)
        self.pressure = 101325     # Atmospheric pressure (Pa)
        self.humidity = 0.4        # 40% relative humidity
        
        # Dynamic effects - increased amplitudes for more visible changes
        self.vibration_modes = [(5.0, wavelength/10),    # Building vibration
                               (35.0, wavelength/15),     # Acoustic noise
                               (150.0, wavelength/20)]    # High-frequency noise
        
        # Store comprehensive history
        self.history = {
            'pattern': [],
            'visibility': [],
            'phase_difference': [],
            'temperature': [],
            'coherence': [],
            'mode_structure': []
        }

    def _generate_laser_modes(self) -> np.ndarray:
        """Generate superposition of Hermite-Gaussian modes"""
        w0 = self.beam_radius
        z = 0  # Initial beam waist position
        
        # Calculate beam parameters
        w = w0 * np.sqrt(1 + (z/self.z_R)**2)
        R_c = z + self.z_R**2/z if z != 0 else np.inf
        gouy = np.arctan(z/self.z_R)
        
        # Generate fundamental mode (TEM00)
        E00 = np.exp(-self.R**2/w**2) * np.exp(-1j*self.k*z)
        
        # Add higher-order modes with larger amplitudes
        E = E00
        for m in range(1, 3):
            for n in range(1, 3):
                amplitude = 0.15 / (m+n)  # Increased contribution
                mode = self._hermite_gaussian_mode(m, n, w, R_c, gouy)
                E += amplitude * mode
        
        return E / np.max(np.abs(E))  # Normalize

    def _hermite_gaussian_mode(self, m: int, n: int, w: float, 
                             R_c: float, gouy: float) -> np.ndarray:
        """Calculate Hermite-Gaussian mode TEM_mn"""
        H_m = HermitePolynomial.evaluate(m, np.sqrt(2)*self.X/w)
        H_n = HermitePolynomial.evaluate(n, np.sqrt(2)*self.Y/w)
        
        phase = self.k*(self.R**2/(2*R_c)) - (m+n+1)*gouy
        return (H_m * H_n * np.exp(-self.R**2/w**2) * 
                np.exp(1j*phase))

    def _generate_mirror_surfaces(self):
        """Generate random mirror surface profiles"""
        self.mirror1_surface = self.surface_roughness * \
                              np.random.normal(0, 1, self.X.shape)
        self.mirror2_surface = self.surface_roughness * \
                              np.random.normal(0, 1, self.X.shape)

    def calculate_path_difference(self) -> float:
        """Calculate optical path difference including all effects"""
        # Basic geometric path difference with larger initial difference
        delta_L = 2 * (self.arm_length2 - self.arm_length1 + self.wavelength/4)
        
        # Add mechanical vibrations
        for freq, amp in self.vibration_modes:
            delta_L += amp * np.sin(2*np.pi*freq*self.time)
        
        # Add mirror misalignment and surface effects - take mean value
        delta_L += self.beam_radius * (np.tan(self.mirror1_angle) + 
                                     np.tan(self.mirror2_angle))
        delta_L += 2 * np.mean(self.mirror1_surface + self.mirror2_surface)  # Changed to mean
        
        # Add enhanced thermal expansion effects
        thermal_expansion = 50e-6  # Increased coefficient
        delta_T = self.temperature - 293.15
        delta_L += thermal_expansion * delta_T * (self.arm_length1 + self.arm_length2)
        
        return float(delta_L)  # Ensure we return a scalar

    def calculate_coherence_factor(self, path_diff: float) -> float:
        """Calculate complex coherence factor including temporal and spatial effects"""
        # Temporal coherence
        gamma_t = np.exp(-(path_diff/self.coherence_length)**2)
        
        # Spatial coherence (van Cittert-Zernike theorem)
        r_c = self.wavelength * self.arm_length1 / (2*np.pi*self.beam_radius)
        gamma_s = np.exp(-(self.R/(2*r_c))**2)
        
        return gamma_t * gamma_s

    def calculate_refractive_index(self) -> float:
        """Calculate air refractive index using Edlén equation"""
        # Enhanced refractive index variations
        n = 1.0 + (1.5e-3 * self.pressure/self.temperature - 
                   3e-11 * self.humidity * self.temperature**2)
        return n

    def get_interference_pattern(self, turbulence: float = 0.0) -> np.ndarray:
        """Calculate interference pattern including all physical effects"""
        # Get path difference and refractive index
        path_diff = self.calculate_path_difference()
        n = self.calculate_refractive_index()
        
        # Phase difference including turbulence and refractive index
        phi = self.k * n * path_diff
        if turbulence > 0:
            # Enhanced Kolmogorov turbulence model
            rice_dist = RiceDistribution(nu=2.0, sigma=1.0)
            phi += turbulence * rice_dist.rvs(size=self.X.shape)

        # Calculate fields from both arms with full beam propagation
        E1 = self.E0 * np.exp(1j * self.k * n * self.arm_length1)
        
        # Second arm with increased misalignment and thermal effects
        x_tilt = self.X * np.tan(2*self.mirror2_angle)
        thermal_distortion = 2e-6 * self.temperature * self.R**2
        E2 = self.E0 * np.exp(1j * self.k * n * (self.arm_length2 + x_tilt + thermal_distortion))

        # Apply coherence effects
        gamma = self.calculate_coherence_factor(path_diff)
        
        # Calculate interference with realistic detector response
        intensity = (np.abs(E1)**2 + np.abs(E2)**2 + 
                    2 * gamma * np.abs(E1 * E2) * np.cos(phi))
        
        # Add enhanced detector noise
        noise = 0.02 * np.random.normal(0, 1, intensity.shape)
        intensity = np.maximum(intensity + noise, 0)
        
        return intensity

    def update(self, delta_t: float, turbulence: float = 0.0):
        """Update interferometer state including environmental variations"""
        # Update time
        self.time += delta_t
        
        # Enhanced environmental variations
        self.temperature += 0.05 * np.sin(2*np.pi*0.05*self.time)  # Faster temperature drift
        
        # Calculate new interference pattern
        pattern = self.get_interference_pattern(turbulence)
        
        # Calculate advanced visibility metrics
        I_max = np.max(pattern)
        I_min = np.min(pattern)
        visibility = (I_max - I_min) / (I_max + I_min)
        
        # Calculate mode structure
        mode_purity = np.sum(np.abs(np.fft.fft2(pattern))**2)
        
        # Store comprehensive results
        self.history['pattern'].append(pattern)
        self.history['visibility'].append(visibility)
        self.history['phase_difference'].append(self.calculate_path_difference())
        self.history['temperature'].append(self.temperature)
        self.history['coherence'].append(np.mean(self.calculate_coherence_factor(
            self.calculate_path_difference())))
        self.history['mode_structure'].append(mode_purity)

def simulate_michelson():
    # Create interferometer with realistic HeNe laser parameters
    interferometer = MichelsonInterferometer(
        wavelength=632.8e-9,     # 632.8 nm (HeNe)
        coherence_length=30e-2,   # 30 cm coherence length
        beam_radius=1e-3,        # 1 mm beam radius
        arm_length1=10e-2,       # 10 cm first arm
        arm_length2=10e-2,       # 10 cm second arm
        M2=1.1                   # Typical HeNe beam quality
    )
    
    # Add increased misalignment
    interferometer.mirror2_angle = 5e-5  # 50 μrad tilt
    
    # Simulation parameters
    dt = 1e-3
    steps = 200  # More steps for better statistics
    turbulence = 0.2  # Increased air turbulence
    
    # Run simulation
    for _ in range(steps):
        interferometer.update(dt, turbulence)
    
    return interferometer

def plot_results(interferometer):
    """Create comprehensive visualization of interference patterns and dynamics"""
    plt.style.use('dark_background')  # Better visibility for interference patterns
    fig = plt.figure(figsize=(20, 12))
    
    # Create subplot grid
    gs = plt.GridSpec(3, 4, figure=fig)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[0, 3])
    ax4 = fig.add_subplot(gs[1, 2])
    ax5 = fig.add_subplot(gs[1, 3])
    ax6 = fig.add_subplot(gs[2, :], projection='3d')
    
    # Initialize plots
    times = np.arange(len(interferometer.history['pattern']))
    
    # Main interference pattern
    im = ax1.pcolormesh(interferometer.X*1e3, interferometer.Y*1e3,
                        interferometer.history['pattern'][0],
                        shading='gouraud', cmap='inferno')
    plt.colorbar(im, ax=ax1, label='Intensity (arb. units)')
    ax1.set_title('Interference Pattern')
    
    # Initialize other plots
    line2, = ax2.plot([], [], 'c-', lw=2)
    line3, = ax3.plot([], [], 'r-', lw=2)
    line4, = ax4.plot([], [], 'g-', lw=2)
    line5, = ax5.plot([], [], 'y-', lw=2)
    surf = ax6.plot_surface(interferometer.X*1e3, interferometer.Y*1e3,
                           interferometer.history['pattern'][0],
                           cmap='plasma', linewidth=0)
    
    # Set axis limits
    ax2.set_xlim(0, len(times))
    ax2.set_ylim(0, 1.1)
    ax3.set_xlim(0, len(times))
    ax3.set_ylim(min(interferometer.history['phase_difference'])*1e9,
                 max(interferometer.history['phase_difference'])*1e9)
    ax4.set_xlim(0, len(times))
    ax4.set_ylim(0, 1.1)
    ax5.set_xlim(0, len(times))
    ax5.set_ylim(min(interferometer.history['temperature']),
                 max(interferometer.history['temperature']))
    
    # Set titles and labels
    ax1.set_xlabel('x (mm)')
    ax1.set_ylabel('y (mm)')
    ax2.set_title('Fringe Visibility')
    ax3.set_title('Path Length Difference')
    ax4.set_title('Coherence Factor')
    ax5.set_title('Temperature')
    ax6.set_title('3D Intensity Distribution')
    
    for ax in [ax2, ax3, ax4, ax5]:
        ax.set_xlabel('Time (ms)')
        ax.grid(True, alpha=0.3)
    ax6.set_xlabel('x (mm)')
    ax6.set_ylabel('y (mm)')
    ax6.set_zlabel('Intensity')
    
    def animate(frame):
        # Update main interference pattern
        im.set_array(interferometer.history['pattern'][frame].ravel())
        
        # Update line plots
        current_times = times[:frame+1]
        line2.set_data(current_times, interferometer.history['visibility'][:frame+1])
        line3.set_data(current_times, np.array(interferometer.history['phase_difference'][:frame+1])*1e9)
        line4.set_data(current_times, interferometer.history['coherence'][:frame+1])
        line5.set_data(current_times, interferometer.history['temperature'][:frame+1])
        
        # Update 3D surface
        ax6.clear()
        ax6.plot_surface(interferometer.X*1e3, interferometer.Y*1e3,
                        interferometer.history['pattern'][frame],
                        cmap='plasma', linewidth=0)
        ax6.set_title('3D Intensity Distribution')
        
        return im, line2, line3, line4, line5
    
    anim = FuncAnimation(fig, animate, frames=len(times),
                        interval=0.01, blit=True)  
    plt.tight_layout()
    plt.show()

def plot_static_results(interferometer):
    """Create static visualization of interference patterns and dynamics"""
    fig = plt.figure(figsize=(16, 12))

    # Create subplot grid
    gs = plt.GridSpec(2, 3, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2], projection='3d')

    times = np.arange(len(interferometer.history['pattern']))

    # Main interference pattern (final state)
    final_idx = -1
    im = ax1.pcolormesh(interferometer.X*1e3, interferometer.Y*1e3,
                        interferometer.history['pattern'][final_idx],
                        shading='gouraud', cmap='inferno')
    plt.colorbar(im, ax=ax1, label='Intensity (arb. units)')
    ax1.set_title('Final Interference Pattern')
    ax1.set_xlabel('x (mm)')
    ax1.set_ylabel('y (mm)')

    # Fringe visibility over time
    ax2.plot(times, interferometer.history['visibility'], 'c-', lw=2)
    ax2.set_title('Fringe Visibility')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Visibility')
    ax2.grid(True, alpha=0.3)

    # Path length difference
    ax3.plot(times, np.array(interferometer.history['phase_difference'])*1e9, 'r-', lw=2)
    ax3.set_title('Path Length Difference')
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Path Difference (nm)')
    ax3.grid(True, alpha=0.3)

    # Coherence factor
    ax4.plot(times, interferometer.history['coherence'], 'g-', lw=2)
    ax4.set_title('Coherence Factor')
    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('Coherence')
    ax4.grid(True, alpha=0.3)

    # Temperature
    ax5.plot(times, interferometer.history['temperature'], 'y-', lw=2)
    ax5.set_title('Temperature')
    ax5.set_xlabel('Time (ms)')
    ax5.set_ylabel('Temperature (K)')
    ax5.grid(True, alpha=0.3)

    # 3D intensity distribution
    ax6.plot_surface(interferometer.X*1e3, interferometer.Y*1e3,
                    interferometer.history['pattern'][final_idx],
                    cmap='plasma', linewidth=0)
    ax6.set_title('3D Intensity Distribution')
    ax6.set_xlabel('x (mm)')
    ax6.set_ylabel('y (mm)')
    ax6.set_zlabel('Intensity')

    plt.tight_layout()

def main():
    # Run simulation
    interferometer = simulate_michelson()

    # Create static plots for saving
    plot_static_results(interferometer)

    # Save plot to output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'michelson_interferometer.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'michelson_interferometer.png')}")

if __name__ == "__main__":
    main()