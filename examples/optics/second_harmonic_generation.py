"""
Example demonstrating second harmonic generation in nonlinear optics.

This example simulates how an intense laser beam at frequency ω generates
a second harmonic at frequency 2ω when propagating through a nonlinear crystal.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics import Wave
from matplotlib.animation import FuncAnimation

class NonlinearCrystal:
    """Class representing a nonlinear optical crystal for SHG"""
    
    def __init__(self,
                 length: float,         # Crystal length (m)
                 chi2: float,           # Second-order susceptibility (m/V)
                 n1: float,             # Refractive index at ω
                 n2: float,             # Refractive index at 2ω
                 nx: int = 100):        # Grid points
        
        self.length = length
        self.chi2 = chi2
        self.n1 = n1
        self.n2 = n2
        
        # Create spatial grid
        self.x = np.linspace(0, length, nx)
        self.dx = self.x[1] - self.x[0]
        
        # Initialize fields
        self.E1 = np.zeros(nx, dtype=complex)  # Fundamental field
        self.E2 = np.zeros(nx, dtype=complex)  # Second harmonic field
        
        # Store history
        self.history = {
            'E1': [],
            'E2': []
        }
        
    def phase_mismatch(self, wavelength: float) -> float:
        """Calculate phase mismatch Δk"""
        k1 = 2 * np.pi * self.n1 / wavelength
        k2 = 2 * np.pi * self.n2 / (wavelength/2)
        return k2 - 2*k1
        
    def propagate(self, dt: float):
        """Propagate fields through crystal"""
        # Phase mismatch term
        dk = self.phase_mismatch(1.064e-6)  # For 1064nm input
        
        # Coupled wave equations for SHG
        dE1 = -1j * self.chi2 * np.conj(self.E1) * self.E2 * np.exp(1j * dk * self.x)
        dE2 = -1j * self.chi2 * self.E1**2 * np.exp(-1j * dk * self.x)
        
        # Update fields
        self.E1 += dE1 * dt
        self.E2 += dE2 * dt
        
        # Store history
        self.history['E1'].append(np.abs(self.E1)**2)
        self.history['E2'].append(np.abs(self.E2)**2)

def simulate_shg():
    # Create nonlinear crystal (KDP parameters)
    crystal = NonlinearCrystal(
        length=0.01,      # 1 cm
        chi2=8.5e-12,     # KDP nonlinear coefficient
        n1=1.494,         # Index at 1064nm
        n2=1.502,         # Index at 532nm
    )
    
    # Initial fundamental field (Gaussian)
    x = crystal.x
    w0 = 0.001  # 1mm beam width
    E0 = 1e6    # Field amplitude
    crystal.E1 = E0 * np.exp(-(x - crystal.length/2)**2 / w0**2).astype(complex)
    
    # Simulation parameters
    dt = 1e-15  # 1 fs time step
    steps = 100
    
    # Run simulation
    for _ in range(steps):
        crystal.propagate(dt)
        
    return crystal

def plot_results(crystal):
    """Plot fundamental and second harmonic intensities"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot fundamental field
    ax1.plot(crystal.x * 1e3, crystal.history['E1'][-1], 'b-', label='ω')
    ax1.set_xlabel('Position (mm)')
    ax1.set_ylabel('Intensity (W/m²)')
    ax1.set_title('Fundamental Field')
    ax1.grid(True)
    ax1.legend()
    
    # Plot second harmonic
    ax2.plot(crystal.x * 1e3, crystal.history['E2'][-1], 'r-', label='2ω')
    ax2.set_xlabel('Position (mm)')
    ax2.set_ylabel('Intensity (W/m²)')
    ax2.set_title('Second Harmonic Field')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()

def main():
    # Run simulation
    crystal = simulate_shg()

    # Plot results
    plot_results(crystal)

    # Save plot to output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'second_harmonic_generation.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'second_harmonic_generation.png')}")

if __name__ == "__main__":
    main() 