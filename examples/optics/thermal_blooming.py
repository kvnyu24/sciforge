"""
Example demonstrating thermal blooming effects in high-power laser propagation.

This example simulates how a high-power laser beam heats the air it passes through,
creating thermal gradients that cause the beam to defocus and distort.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics import Wave, ThermalSystem
from matplotlib.animation import FuncAnimation

class ThermalBloomingBeam:
    """Class representing a laser beam with thermal blooming effects"""
    
    def __init__(self,
                 power: float,         # Laser power (W)
                 wavelength: float,    # Laser wavelength (m)
                 beam_radius: float,   # Initial beam radius (m)
                 absorption: float,    # Absorption coefficient (1/m)
                 n_temp: float = 1e-6, # dn/dT of medium (1/K)
                 nx: int = 50,         # Grid points in x
                 ny: int = 50):        # Grid points in y
        
        # Initialize laser parameters
        self.power = power
        self.wavelength = wavelength
        self.beam_radius = beam_radius
        self.absorption = absorption
        self.n_temp = n_temp
        
        # Create spatial grid
        self.x = np.linspace(-3*beam_radius, 3*beam_radius, nx)
        self.y = np.linspace(-3*beam_radius, 3*beam_radius, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Initialize beam as Gaussian
        self.intensity = self._gaussian_beam(self.X, self.Y)
        
        # Initialize temperature field
        self.temperature = ThermalSystem(
            temperature=293.15,  # Start at 20°C
            mass=1.2,           # Air density kg/m³
            specific_heat=1005, # Air specific heat J/(kg·K)
            thermal_conductivity=0.026  # Air thermal conductivity W/(m·K)
        )
        
        # Store history
        self.history = {
            'intensity': [self.intensity.copy()],
            'temperature': [293.15 * np.ones_like(self.X)],
            'phase': [np.zeros_like(self.X)]
        }
        
    def _gaussian_beam(self, x, y):
        """Calculate Gaussian beam intensity profile"""
        r2 = x**2 + y**2
        I0 = 2 * self.power / (np.pi * self.beam_radius**2)
        return I0 * np.exp(-2 * r2 / self.beam_radius**2)
    
    def update(self, dt: float):
        """Update beam and temperature distribution"""
        # Calculate heating from absorbed laser power
        dQ = self.absorption * self.intensity * dt
        
        # Update temperature field
        dT = dQ / (self.temperature.mass * self.temperature.specific_heat)
        T = self.history['temperature'][-1] + dT
        
        # Calculate thermal lensing phase shift
        phase = 2 * np.pi * self.n_temp * (T - 293.15) / self.wavelength
        
        # Propagate beam through thermal lens
        E = np.sqrt(self.intensity) * np.exp(1j * phase)
        self.intensity = np.abs(E)**2
        
        # Add thermal diffusion
        T = self._apply_thermal_diffusion(T, dt)
        
        # Store updated state
        self.history['intensity'].append(self.intensity.copy())
        self.history['temperature'].append(T)
        self.history['phase'].append(phase)
        
    def _apply_thermal_diffusion(self, T, dt):
        """Apply thermal diffusion using finite difference"""
        alpha = self.temperature.thermal_conductivity / \
               (self.temperature.mass * self.temperature.specific_heat)
        dx = self.x[1] - self.x[0]
        
        # 2D diffusion
        laplacian = (np.roll(T, 1, axis=0) + np.roll(T, -1, axis=0) + 
                    np.roll(T, 1, axis=1) + np.roll(T, -1, axis=1) - 4*T) / dx**2
        
        return T + alpha * dt * laplacian

def simulate_thermal_blooming():
    # Create laser beam
    beam = ThermalBloomingBeam(
        power=1000,        # 1 kW
        wavelength=1.06e-6,  # 1.06 μm (Nd:YAG)
        beam_radius=0.02,    # 2 cm
        absorption=0.1,      # 0.1 m⁻¹
    )
    
    # Simulation parameters
    dt = 1e-3
    steps = 100
    
    # Run simulation
    for _ in range(steps):
        beam.update(dt)
        
    return beam

def plot_results(beam):
    """Create animation of thermal blooming evolution"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    def update(frame):
        # Clear axes
        ax1.clear()
        ax2.clear()
        
        # Plot intensity
        im1 = ax1.pcolormesh(beam.X * 100, beam.Y * 100, 
                            beam.history['intensity'][frame],
                            shading='auto', cmap='hot')
        ax1.set_xlabel('x (cm)')
        ax1.set_ylabel('y (cm)')
        ax1.set_title(f'Beam Intensity (W/m²)\nt = {frame*1e-3:.1f} ms')
        plt.colorbar(im1, ax=ax1)
        
        # Plot temperature
        im2 = ax2.pcolormesh(beam.X * 100, beam.Y * 100,
                            beam.history['temperature'][frame] - 273.15,
                            shading='auto', cmap='plasma')
        ax2.set_xlabel('x (cm)')
        ax2.set_ylabel('y (cm)')
        ax2.set_title('Temperature (°C)')
        plt.colorbar(im2, ax=ax2)
        
        return im1, im2
    
    anim = FuncAnimation(fig, update, frames=len(beam.history['intensity']),
                        interval=50, blit=True)
    plt.tight_layout()
    return anim

def plot_static_results(beam):
    """Create static visualization of thermal blooming evolution"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot intensity and temperature at different time steps
    n_frames = len(beam.history['intensity'])
    frames_to_plot = [0, n_frames//4, n_frames-1]

    for idx, frame in enumerate(frames_to_plot):
        ax = axes[0, idx]
        im = ax.pcolormesh(beam.X * 100, beam.Y * 100,
                          beam.history['intensity'][frame],
                          shading='auto', cmap='hot')
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')
        ax.set_title(f'Intensity at t = {frame*1e-3:.1f} ms')
        plt.colorbar(im, ax=ax)

    for idx, frame in enumerate(frames_to_plot):
        ax = axes[1, idx]
        im = ax.pcolormesh(beam.X * 100, beam.Y * 100,
                          beam.history['temperature'][frame] - 273.15,
                          shading='auto', cmap='plasma')
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')
        ax.set_title(f'Temperature at t = {frame*1e-3:.1f} ms (°C)')
        plt.colorbar(im, ax=ax)

    plt.tight_layout()

def main():
    # Run simulation
    beam = simulate_thermal_blooming()

    # Create static plot for saving
    plot_static_results(beam)

    # Save plot to output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'thermal_blooming.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'thermal_blooming.png')}")

if __name__ == "__main__":
    main()