"""
Example demonstrating electromagnetic wave propagation and interference.

This example simulates the propagation of electromagnetic waves in space,
showing interference patterns and diffraction effects. It uses the Wave class
for wave mechanics and RK4 solver for time evolution.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics import Wave
from src.sciforge.differential import RungeKutta4
from matplotlib.animation import FuncAnimation

class EMWave:
    """Class representing electromagnetic wave propagation"""
    
    def __init__(self,
                 wavelength: float,    # Wavelength (m)
                 amplitude: float,      # Field amplitude (V/m)
                 domain_size: float,    # Spatial domain size (m)
                 nx: int = 200):        # Number of spatial points
        
        # Initialize spatial grid
        self.x = np.linspace(-domain_size/2, domain_size/2, nx)
        self.dx = self.x[1] - self.x[0]
        
        # Wave parameters
        self.wavelength = wavelength
        self.k = 2 * np.pi / wavelength  # Wavenumber
        self.omega = 3e8 * self.k        # Angular frequency (using c = 3e8)
        
        # Initialize E and B field components
        self.E = np.zeros(nx)  # Electric field
        self.B = np.zeros(nx)  # Magnetic field
        self.amplitude = amplitude
        
        # Store history for animation
        self.history = {
            'time': [0.0],
            'E': [self.E.copy()],
            'B': [self.B.copy()]
        }
    
    def initial_condition(self, gaussian_width: float):
        """Set initial Gaussian pulse"""
        envelope = np.exp(-(self.x/gaussian_width)**2)
        self.E = self.amplitude * envelope * np.sin(self.k * self.x)
        self.B = self.amplitude/3e8 * envelope * np.sin(self.k * self.x)
    
    def field_derivatives(self, t: float, y: np.ndarray) -> np.ndarray:
        """Calculate field derivatives for Maxwell's equations"""
        E = y[:len(self.x)]
        B = y[len(self.x):]
        
        # Maxwell's equations in 1D
        dE_dt = 3e8 * np.gradient(B, self.dx)
        dB_dt = 3e8 * np.gradient(E, self.dx)
        
        return np.concatenate([dE_dt, dB_dt])

def simulate_em_waves():
    # Create EM wave system
    wave = EMWave(
        wavelength=1e-6,    # 1 μm wavelength
        amplitude=1.0,      # 1 V/m amplitude
        domain_size=10e-6,  # 10 μm domain
        nx=200
    )
    
    # Set initial Gaussian pulse
    wave.initial_condition(gaussian_width=2e-6)
    
    # Simulation parameters
    dt = 1e-16  # 0.1 fs time step
    t_final = 5e-14  # 50 fs total time
    
    # Create RK4 solver
    solver = RungeKutta4()
    
    # Initial conditions
    y0 = np.concatenate([wave.E, wave.B])
    
    # Solve equations
    t, y = solver.solve(
        wave.field_derivatives,
        y0,
        (0, t_final),
        dt
    )
    
    # Store results
    for i in range(len(t)):
        wave.history['time'].append(t[i])
        wave.history['E'].append(y[i, :len(wave.x)])
        wave.history['B'].append(y[i, len(wave.x):])
    
    return wave

def plot_results(wave):
    """Create animation of wave propagation"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    def update(frame):
        # Clear axes
        ax1.clear()
        ax2.clear()
        
        # Plot E field
        ax1.plot(wave.x * 1e6, wave.history['E'][frame])
        ax1.set_ylabel('Electric Field (V/m)', fontsize=10)
        ax1.set_title(f'Electromagnetic Wave Propagation (t = {wave.history["time"][frame]*1e15:.1f} fs)', 
                     fontsize=12, pad=10)
        ax1.grid(True)
        ax1.set_ylim(-1.2*wave.amplitude, 1.2*wave.amplitude)
        ax1.set_xlabel('Position (μm)', fontsize=10)  # Add x-label for top plot
        
        # Plot B field
        ax2.plot(wave.x * 1e6, wave.history['B'][frame])
        ax2.set_xlabel('Position (μm)', fontsize=10)
        ax2.set_ylabel('Magnetic Field (T)', fontsize=10)
        ax2.grid(True)
        ax2.set_ylim(-1.2*wave.amplitude/3e8, 1.2*wave.amplitude/3e8)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Return artists that were updated
        return ax1.get_lines() + ax2.get_lines()
    
    anim = FuncAnimation(
        fig, update,
        frames=len(wave.history['time']),
        interval=50,
        blit=True
    )
    
    # Add padding to prevent label cutoff
    plt.subplots_adjust(left=0.12, bottom=0.1, top=0.95)
    return anim

def main():
    # Run simulation
    wave = simulate_em_waves()
    
    # Create animation
    anim = plot_results(wave)
    plt.show()

if __name__ == "__main__":
    main() 