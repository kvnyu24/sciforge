"""
Example demonstrating the quantum double-slit experiment simulation.

This example shows how a quantum particle exhibits wave-like behavior when passing
through a double-slit apparatus, creating an interference pattern on the detection screen.
The simulation models:
- Wave packet evolution
- Diffraction and interference
- Probability density detection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics import Wavefunction
from matplotlib.animation import FuncAnimation

class DoubleSlit:
    """Class representing quantum double-slit experiment"""
    
    def __init__(self,
                 slit_separation: float,    # Distance between slits (m)
                 slit_width: float,         # Width of each slit (m)
                 screen_distance: float,     # Distance to detection screen (m)
                 particle_mass: float,       # Particle mass (kg)
                 wavelength: float,          # de Broglie wavelength (m)
                 nx: int = 200,             # Grid points in x
                 ny: int = 200):            # Grid points in y
        
        # Store parameters
        self.slit_separation = slit_separation
        self.slit_width = slit_width
        self.screen_distance = screen_distance
        
        # Create spatial grid
        x_max = 5 * slit_separation
        y_max = screen_distance
        self.x = np.linspace(-x_max, x_max, nx)
        self.y = np.linspace(0, y_max, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Create initial wave packet
        sigma = wavelength  # Spatial spread
        k0 = 2 * np.pi / wavelength
        psi0 = np.exp(-(self.X**2 + (self.Y - wavelength)**2)/(4*sigma**2)) * \
               np.exp(1j * k0 * self.Y)
        
        # Initialize wavefunction
        self.wavefunction = Wavefunction(psi0, self.x, particle_mass)
        
        # Create slit potential barrier
        self.potential = np.zeros((ny, nx))
        barrier_y = int(0.1 * ny)  # Barrier position
        self.potential[barrier_y, :] = 1e30  # Very high potential barrier
        
        # Create slits (zero potential in slit regions)
        slit1_center = int(nx/2 + slit_separation/(2*x_max/nx))
        slit2_center = int(nx/2 - slit_separation/(2*x_max/nx))
        slit_width_pixels = int(slit_width/(2*x_max/nx))
        
        self.potential[barrier_y, 
                      slit1_center-slit_width_pixels:slit1_center+slit_width_pixels] = 0
        self.potential[barrier_y, 
                      slit2_center-slit_width_pixels:slit2_center+slit_width_pixels] = 0
        
    def evolve(self, dt: float, steps: int):
        """Evolve quantum state through time"""
        for _ in range(steps):
            self.wavefunction.evolve(dt, self.potential)
            
    def get_detection_pattern(self) -> np.ndarray:
        """Get probability density at screen position"""
        screen_idx = int(0.9 * len(self.y))  # Screen near end of y domain
        return np.abs(self.wavefunction.psi[screen_idx, :])**2

def simulate_double_slit():
    # Create double-slit experiment (parameters for electrons)
    experiment = DoubleSlit(
        slit_separation=100e-9,    # 100 nm separation
        slit_width=30e-9,         # 30 nm slit width
        screen_distance=1e-6,     # 1 μm to screen
        particle_mass=9.1e-31,    # Electron mass
        wavelength=50e-9          # 50 nm de Broglie wavelength
    )
    
    # Simulation parameters
    dt = 1e-18  # 1 attosecond time step
    steps = 1000
    
    # Run simulation
    experiment.evolve(dt, steps)
    
    return experiment

def plot_results(experiment):
    """Create visualization of double-slit results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot probability density
    im1 = ax1.pcolormesh(experiment.X*1e9, experiment.Y*1e9, 
                        np.abs(experiment.wavefunction.psi)**2,
                        shading='auto', cmap='viridis')
    plt.colorbar(im1, ax=ax1, label='Probability Density')
    ax1.set_xlabel('x (nm)')
    ax1.set_ylabel('y (nm)')
    ax1.set_title('Wavefunction Evolution')
    
    # Plot detection pattern
    pattern = experiment.get_detection_pattern()
    ax2.plot(experiment.x*1e9, pattern/np.max(pattern))
    ax2.set_xlabel('Position on Screen (nm)')
    ax2.set_ylabel('Normalized Intensity')
    ax2.set_title('Interference Pattern')
    ax2.grid(True)
    
    plt.tight_layout()

def main():
    # Run simulation
    experiment = simulate_double_slit()
    
    # Plot results
    plot_results(experiment)
    plt.show()

if __name__ == "__main__":
    main()
