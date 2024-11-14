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
        
        # Create highly non-uniform spatial grid with much finer resolution near conductor
        r = np.logspace(np.log10(radius/2), np.log10(20*radius), nx//2)
        self.x = np.concatenate([-np.flip(r), r])
        self.y = np.concatenate([-np.flip(r), r])
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Initialize fields with strong asymmetric distribution
        theta = np.arctan2(self.Y, self.X)
        r = np.sqrt(self.X**2 + self.Y**2)
        
        # Add complex angular variations with sharp peaks
        asymmetry = (1 + 0.8*np.sin(5*theta) + 0.6*np.cos(3*theta) + 
                    0.4*np.sin(7*theta) * np.exp(-r/(2*radius)) +
                    0.5*np.cos(4*theta) * np.exp(-r/(1.5*radius)))
        
        # Create sharper field gradients near conductor
        base_field = voltage * (radius/r)**1.5 * asymmetry * np.exp(-r/(4*radius))
        
        self.E = np.zeros((ny, nx, 2))
        self.E[:,:,0] = base_field * self.X / (r + 1e-10)  # Radial component
        self.E[:,:,1] = base_field * self.Y / (r + 1e-10)
        
        # Initialize charge density with strong non-uniform background
        self.rho = (1e-9 * (1 + np.sin(4*theta) + 0.7*np.cos(6*theta)) * 
                   np.exp(-r/(3*radius)) * (1 + 0.5*np.random.randn(ny, nx)))
        
        self.ions = []  # List to store ion particles
        
        # Add initial ions in highly asymmetric pattern
        for i in range(50):
            # Create clusters of ions
            cluster_angle = 2*np.pi * i/10
            r_cluster = 2*radius * (1 + 0.5*np.sin(3*cluster_angle))
            
            for _ in range(5):
                # Add random offset within cluster
                dr = 0.2*radius*np.random.randn()
                dtheta = 0.2*np.random.randn()
                r_ion = r_cluster + dr
                theta_ion = cluster_angle + dtheta
                
                x = r_ion * np.cos(theta_ion)
                y = r_ion * np.sin(theta_ion)
                
                self.ions.append(Particle(
                    mass=1.67e-27,
                    position=np.array([x, y, 0]),
                    velocity=np.array([0, 0, 0])
                ))
        
        # Add strong spatially varying field perturbations
        self.field_perturbation = 0.4 * np.exp(-r/(5*radius)) * (
            np.sin(3*theta) + np.cos(4*theta)) * np.random.randn(ny, nx)
        
        # Add time-varying voltage component with spatial dependence
        self.base_voltage = voltage
        
        # Store history
        self.history = {
            'time': [0.0],
            'current': [1e-6],  # Start with non-zero current
            'ion_count': [len(self.ions)],
            'field_strength': [self.calculate_max_field()]
        }
        
    def calculate_field(self):
        """Calculate electric field including space charge effects and perturbations"""
        # Time-varying voltage with strong spatial modulation
        t = self.history['time'][-1]
        theta = np.arctan2(self.Y, self.X)
        r = np.sqrt(self.X**2 + self.Y**2)
        
        voltage_mod = self.base_voltage * (1 + 0.5*np.sin(2*np.pi*t*1e5) * 
                                         np.exp(-r/(3*self.radius)))
        
        # Base field with sharp gradients
        E0 = voltage_mod * (self.radius/r)**1.8
        
        # Add multiple strong angular variations
        angular_var = (1 + 0.8*np.sin(4*theta) + 0.6*np.sin(6*theta) + 
                      0.5*np.cos(3*theta) * np.exp(-r/(4*self.radius)) +
                      0.4*np.sin(8*theta) * np.exp(-r/(2*self.radius)))
        E0 *= angular_var
        
        # Add stronger spatially varying perturbations
        perturbation = (self.field_perturbation * 
                       (1 + 0.6*np.sin(5*theta)) * 
                       np.exp(-r/(6*self.radius)))
        E0 *= (1 + perturbation)
        
        Ex = E0 * self.X / (r + 1e-10)
        Ey = E0 * self.Y / (r + 1e-10)
        
        # Add enhanced non-linear space charge effects
        rho_grad_x = np.gradient(self.rho**1.8, self.x[1]-self.x[0], axis=1)
        rho_grad_y = np.gradient(self.rho**1.8, self.y[1]-self.y[0], axis=0)
        
        charge_mod = (1 + 0.7*np.sin(2*np.pi*t*1e4)) * np.exp(-r/(3*self.radius))
        Ex += rho_grad_x * charge_mod
        Ey += rho_grad_y * charge_mod
        
        self.E[:,:,0] = Ex
        self.E[:,:,1] = Ey
        
    def generate_ions(self, dt: float):
        """Generate new ions in high field regions with enhanced spatial variation"""
        E_mag = np.sqrt(self.E[:,:,0]**2 + self.E[:,:,1]**2)
        
        # Strongly varying ionization threshold
        theta = np.arctan2(self.Y, self.X)
        r = np.sqrt(self.X**2 + self.Y**2)
        local_threshold = self.ionization_field * (0.5 + 0.5*np.sin(3*np.pi*self.history['time'][-1]*1e5) + 
                                                 0.4*np.cos(4*theta)) * np.exp(-r/(5*self.radius))
        mask = E_mag > local_threshold
        
        # Non-uniform ion generation probability
        prob = 3 * dt * (E_mag[mask] - local_threshold[mask]) / local_threshold[mask]
        angular_term = (2.0 + np.sin(5*theta[mask]) + 0.8*np.cos(7*theta[mask])) * np.exp(-r[mask]/(4*self.radius))
        prob *= angular_term
        new_ions = np.random.random(size=prob.shape) < prob
        
        # Create ion particles with stronger position-dependent initial velocity
        for x, y in zip(self.X[mask][new_ions], self.Y[mask][new_ions]):
            r_local = np.sqrt(x**2 + y**2)
            angle = np.random.uniform(0, 2*np.pi)
            v0 = np.array([np.cos(angle), np.sin(angle), 0]) * 8000 * np.exp(-r_local/(3*self.radius))
            self.ions.append(Particle(
                mass=1.67e-27,
                position=np.array([x, y, 0]),
                velocity=v0
            ))
            
    def update(self, dt: float):
        """Update corona discharge state"""
        # Update field perturbations with stronger spatial variation
        r = np.sqrt(self.X**2 + self.Y**2)
        theta = np.arctan2(self.Y, self.X)
        spatial_noise = (0.5 * dt * np.random.randn(*self.field_perturbation.shape) * 
                        np.exp(-r/(8*self.radius)) * 
                        (1 + 0.4*np.sin(4*theta)))
        self.field_perturbation += spatial_noise
        self.field_perturbation *= 0.8
        
        # Calculate electric field
        self.calculate_field()
        
        # Generate new ions
        self.generate_ions(dt)
        
        # Move existing ions with enhanced position-dependent turbulent motion
        current = 1e-6  # Base current
        ions_to_remove = []
        
        for i, ion in enumerate(self.ions):
            idx_x = np.searchsorted(self.x, ion.position[0])
            idx_y = np.searchsorted(self.y, ion.position[1])
            
            if 0 <= idx_x < len(self.x)-1 and 0 <= idx_y < len(self.y)-1:
                E_local = self.E[idx_y, idx_x]
                r_local = np.sqrt(np.sum(ion.position[:2]**2))
                
                # Enhanced turbulent motion
                turbulence = np.random.randn(2) * 1000 * np.exp(-r_local/(2*self.radius))
                ion.velocity[:2] = self.mobility * E_local + turbulence
                ion.position += ion.velocity * dt
                
                # Non-linear current contribution
                current += np.sum(ion.velocity[:2] * E_local) * (2.0 + 0.5*np.exp(-r_local/(2*self.radius)))
                
                if r_local > 12*self.radius:
                    ions_to_remove.append(i)
            
        # Remove escaped ions
        for i in reversed(ions_to_remove):
            self.ions.pop(i)
        
        # Update space charge density with stronger spatial variation
        theta = np.arctan2(self.Y, self.X)
        r = np.sqrt(self.X**2 + self.Y**2)
        self.rho = 1e-9 * (1 + 0.6*np.sin(4*theta)) * np.exp(-r/(3*self.radius))
        
        for ion in self.ions:
            idx_x = np.searchsorted(self.x, ion.position[0])
            idx_y = np.searchsorted(self.y, ion.position[1])
            if 0 <= idx_x < len(self.x)-1 and 0 <= idx_y < len(self.y)-1:
                r_local = np.sqrt(ion.position[0]**2 + ion.position[1]**2)
                self.rho[idx_y, idx_x] += 3.2e-19 * np.exp(-r_local/(3*self.radius))
        
        # Apply anisotropic diffusion with enhanced spatial dependence
        diffusion_x = np.roll(self.rho, 1, axis=1) + np.roll(self.rho, -1, axis=1)
        diffusion_y = np.roll(self.rho, 1, axis=0) + np.roll(self.rho, -1, axis=0)
        diff_weight = np.exp(-r/(4*self.radius))
        self.rho = (0.3 * diffusion_x + 0.4 * diffusion_y) * diff_weight + self.rho * (1 - diff_weight)
        
        # Update history
        self.history['time'].append(self.history['time'][-1] + dt)
        self.history['current'].append(abs(current))  # Use absolute value
        self.history['ion_count'].append(max(1, len(self.ions)))  # Ensure non-zero
        self.history['field_strength'].append(self.calculate_max_field())
        
    def calculate_max_field(self) -> float:
        """Calculate maximum electric field strength"""
        return np.max(np.sqrt(self.E[:,:,0]**2 + self.E[:,:,1]**2))

def simulate_corona():
    # Create corona discharge with more extreme parameters
    corona = CoronaDischarge(
        voltage=80000,     # 80 kV - higher voltage
        radius=5e-6,       # 5 μm radius - sharper point
        air_density=0.7    # Lower air density for more ionization
    )
    
    # Simulation parameters
    dt = 1e-9  # 1 ns time step
    steps = 5000  # More steps for longer evolution
    
    # Run simulation
    for _ in range(steps):
        corona.update(dt)
        
    return corona

def plot_results(corona):
    """Create animation of corona discharge evolution"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    def update(frame):
        # Clear axes
        for ax in [ax1, ax2, ax3, ax4]:
            ax.clear()
            
        # Plot electric field magnitude with enhanced contrast
        E_mag = np.sqrt(corona.E[:,:,0]**2 + corona.E[:,:,1]**2)
        im1 = ax1.pcolormesh(corona.X*1000, corona.Y*1000, np.log10(E_mag), 
                            shading='auto', cmap='plasma', vmin=4, vmax=7)
        plt.colorbar(im1, ax=ax1, label='Log Electric Field (V/m)')
        ax1.set_xlabel('x (mm)')
        ax1.set_ylabel('y (mm)')
        ax1.set_title('Electric Field Distribution')
        
        # Plot ion density with enhanced contrast
        im2 = ax2.pcolormesh(corona.X*1000, corona.Y*1000, np.log10(corona.rho + 1e-20),
                            shading='auto', cmap='viridis', vmin=-19, vmax=-16)
        plt.colorbar(im2, ax=ax2, label='Log Space Charge Density (C/m³)')
        ax2.set_xlabel('x (mm)')
        ax2.set_ylabel('y (mm)')
        ax2.set_title('Ion Distribution')
        
        # Plot current vs time with moving window
        window = 100
        start_idx = max(0, frame - window)
        times = np.array(corona.history['time'][start_idx:frame])*1e9
        current = np.array(corona.history['current'][start_idx:frame])*1e6
        ax3.plot(times, current)
        ax3.set_xlabel('Time (ns)')
        ax3.set_ylabel('Current (μA)')
        ax3.set_title('Corona Current')
        ax3.grid(True)
        
        # Plot ion count vs time with moving window
        ax4.plot(times, corona.history['ion_count'][start_idx:frame])
        ax4.set_xlabel('Time (ns)')
        ax4.set_ylabel('Number of Ions')
        ax4.set_title('Ion Population')
        ax4.grid(True)
        ax4.set_yscale('log')
        
        plt.tight_layout()
        
        return ax1.get_lines() + ax2.get_lines() + ax3.get_lines() + ax4.get_lines()
    
    anim = FuncAnimation(
        fig, update,
        frames=len(corona.history['time']),
        interval=50,
        blit=True
    )
    
    return anim

def main():
    # Run simulation
    corona = simulate_corona()
    
    # Create and display animation
    anim = plot_results(corona)
    plt.show()

if __name__ == "__main__":
    main() 