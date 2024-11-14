"""
Example demonstrating Corona discharge simulation.

This example shows how high electric fields near sharp conductors can ionize air molecules,
creating a visible glow discharge. The simulation models:
- Electric field distribution
- Ion generation and movement
- Space charge effects
- Current-voltage characteristics
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics import ElectricField, Particle
from matplotlib.animation import FuncAnimation

class CoronaDischarge:
    """Class representing corona discharge around a conductor"""
    
    def __init__(self,
                 voltage: float,        # Applied voltage (V)
                 radius: float,         # Conductor radius (m)
                 air_density: float,    # Air density (kg/m³)
                 ionization_field: float = 3e6,  # Breakdown field strength (V/m)
                 mobility: float = 2e-4,         # Ion mobility (m²/V·s)
                 nx: int = 100,         # Grid points in x
                 ny: int = 100):        # Grid points in y
        
        self.voltage = voltage
        self.radius = radius
        self.air_density = air_density
        self.ionization_field = ionization_field
        self.mobility = mobility
        
        # Create non-uniform spatial grid for better resolution near conductor
        r = np.logspace(np.log10(radius), np.log10(10*radius), nx//2)
        self.x = np.concatenate([-np.flip(r), r])
        self.y = np.concatenate([-np.flip(r), r])
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Initialize fields
        self.E = np.zeros((ny, nx, 2))  # Electric field components
        self.rho = np.zeros((ny, nx))   # Space charge density
        self.ions = []                  # List to store ion particles
        
        # Add random perturbations to initial field
        self.field_perturbation = 0.2 * np.random.randn(ny, nx)
        
        # Store history
        self.history = {
            'time': [0.0],
            'current': [0.0],
            'ion_count': [0],
            'field_strength': [self.calculate_max_field()]
        }
        
    def calculate_field(self):
        """Calculate electric field including space charge effects and perturbations"""
        # Base field from conductor with asymmetric perturbation
        r = np.sqrt(self.X**2 + self.Y**2)
        theta = np.arctan2(self.Y, self.X)
        E0 = self.voltage * self.radius / r**2
        E0 *= (1 + 0.3*np.sin(3*theta))  # Add angular variation
        E0 *= (1 + self.field_perturbation)  # Add spatial noise
        
        Ex = E0 * self.X / r
        Ey = E0 * self.Y / r
        
        # Add space charge contribution with non-linear effects
        rho_grad_x = np.gradient(self.rho**1.5, self.x[1]-self.x[0], axis=1)
        rho_grad_y = np.gradient(self.rho**1.5, self.y[1]-self.y[0], axis=0)
        Ex += rho_grad_x
        Ey += rho_grad_y
        
        self.E[:,:,0] = Ex
        self.E[:,:,1] = Ey
        
    def generate_ions(self, dt: float):
        """Generate new ions in high field regions with spatial variation"""
        E_mag = np.sqrt(self.E[:,:,0]**2 + self.E[:,:,1]**2)
        
        # Add time-varying ionization threshold
        local_threshold = self.ionization_field * (1 + 0.2*np.sin(2*np.pi*self.history['time'][-1]*1e6))
        mask = E_mag > local_threshold
        
        # Generate ions with non-uniform probability
        prob = dt * (E_mag[mask] - local_threshold) / local_threshold
        prob *= (1 + 0.5*np.sin(4*np.arctan2(self.Y[mask], self.X[mask])))  # Angular dependence
        new_ions = np.random.random(size=prob.shape) < prob
        
        # Create ion particles with initial velocity
        for x, y in zip(self.X[mask][new_ions], self.Y[mask][new_ions]):
            angle = np.random.uniform(0, 2*np.pi)
            v0 = np.array([np.cos(angle), np.sin(angle), 0]) * 1000  # Initial velocity
            self.ions.append(Particle(
                mass=1.67e-27,  # Approximate ion mass
                position=np.array([x, y, 0]),
                velocity=v0
            ))
            
    def update(self, dt: float):
        """Update corona discharge state"""
        # Update field perturbations
        self.field_perturbation += 0.1 * dt * np.random.randn(*self.field_perturbation.shape)
        self.field_perturbation *= 0.95  # Decay factor
        
        # Calculate electric field
        self.calculate_field()
        
        # Generate new ions
        self.generate_ions(dt)
        
        # Move existing ions with turbulent motion
        current = 0
        for ion in self.ions:
            # Get local field at ion position
            idx_x = np.searchsorted(self.x, ion.position[0])
            idx_y = np.searchsorted(self.y, ion.position[1])
            if 0 <= idx_x < len(self.x) and 0 <= idx_y < len(self.y):
                E_local = self.E[idx_y, idx_x]
                
                # Add random turbulent velocity component
                turbulence = np.random.randn(2) * 100
                ion.velocity[:2] = self.mobility * E_local + turbulence
                ion.position += ion.velocity * dt
                
                # Contribute to current
                current += np.sum(ion.velocity[:2] * E_local)
        
        # Update space charge density with diffusion
        self.rho = np.zeros_like(self.rho)
        for ion in self.ions:
            idx_x = np.searchsorted(self.x, ion.position[0])
            idx_y = np.searchsorted(self.y, ion.position[1])
            if 0 <= idx_x < len(self.x) and 0 <= idx_y < len(self.y):
                self.rho[idx_y, idx_x] += 1.6e-19  # Elementary charge
        
        # Apply diffusion
        self.rho = np.roll(self.rho, 1, axis=0) + np.roll(self.rho, -1, axis=0) + \
                  np.roll(self.rho, 1, axis=1) + np.roll(self.rho, -1, axis=1)
        self.rho *= 0.2
        
        # Update history
        self.history['time'].append(self.history['time'][-1] + dt)
        self.history['current'].append(current)
        self.history['ion_count'].append(len(self.ions))
        self.history['field_strength'].append(self.calculate_max_field())
        
    def calculate_max_field(self) -> float:
        """Calculate maximum electric field strength"""
        return np.max(np.sqrt(self.E[:,:,0]**2 + self.E[:,:,1]**2))

def simulate_corona():
    # Create corona discharge (e.g., around a thin wire)
    corona = CoronaDischarge(
        voltage=30000,     # 30 kV - higher voltage
        radius=5e-5,       # 50 μm radius - sharper point
        air_density=1.0    # Lower air density for more ionization
    )
    
    # Simulation parameters
    dt = 1e-9  # 1 ns time step
    steps = 2000  # More steps for better evolution
    
    # Run simulation
    for _ in range(steps):
        corona.update(dt)
        
    return corona

def plot_results(corona):
    """Plot corona discharge visualization and characteristics"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot electric field magnitude with log scale
    E_mag = np.sqrt(corona.E[:,:,0]**2 + corona.E[:,:,1]**2)
    im1 = ax1.pcolormesh(corona.X*1000, corona.Y*1000, np.log10(E_mag), 
                        shading='auto', cmap='plasma')
    plt.colorbar(im1, ax=ax1, label='Log Electric Field (V/m)')
    ax1.set_xlabel('x (mm)')
    ax1.set_ylabel('y (mm)')
    ax1.set_title('Electric Field Distribution')
    
    # Plot ion density with enhanced contrast
    im2 = ax2.pcolormesh(corona.X*1000, corona.Y*1000, np.sqrt(corona.rho),
                        shading='auto', cmap='viridis')
    plt.colorbar(im2, ax=ax2, label='√(Space Charge Density) (√(C/m³))')
    ax2.set_xlabel('x (mm)')
    ax2.set_ylabel('y (mm)')
    ax2.set_title('Ion Distribution')
    
    # Plot current vs time with moving average
    current = np.array(corona.history['current'])*1e6
    window = 50
    smoothed_current = np.convolve(current, np.ones(window)/window, mode='valid')
    ax3.plot(np.array(corona.history['time'][window-1:])*1e9, smoothed_current)
    ax3.set_xlabel('Time (ns)')
    ax3.set_ylabel('Current (μA)')
    ax3.set_title('Corona Current')
    ax3.grid(True)
    
    # Plot ion count vs time
    ax4.plot(np.array(corona.history['time'])*1e9, 
            corona.history['ion_count'])
    ax4.set_xlabel('Time (ns)')
    ax4.set_ylabel('Number of Ions')
    ax4.set_title('Ion Population')
    ax4.grid(True)
    ax4.set_yscale('log')
    
    plt.tight_layout()

def main():
    # Run simulation
    corona = simulate_corona()
    
    # Plot results
    plot_results(corona)
    plt.show()

if __name__ == "__main__":
    main() 