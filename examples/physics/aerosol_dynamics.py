"""
Example demonstrating aerosol particle dynamics simulation.

This example shows how aerosol particles move under the influence of:
- Brownian motion (random thermal motion)
- Gravitational settling
- Evaporation/condensation effects
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics import ThermalSystem
from src.sciforge.stochastic import WienerProcess

class AerosolParticle(ThermalSystem):
    """Class representing an aerosol particle"""
    
    def __init__(self,
                 radius: float,         # Particle radius (m)
                 density: float,        # Particle density (kg/m³)
                 temperature: float,    # Initial temperature (K)
                 position: np.ndarray,  # Initial position [x,y,z] (m)
                 vapor_pressure: float, # Saturation vapor pressure (Pa)
                 surface_tension: float # Surface tension (N/m)
                ):
        # Calculate mass from density and radius
        mass = density * 4/3 * np.pi * radius**3
        
        super().__init__(
            temperature=temperature,
            mass=mass,
            specific_heat=1000,  # Approximate specific heat
            thermal_conductivity=0.1  # Approximate thermal conductivity
        )
        
        self.radius = radius
        self.density = density
        self.position = position
        self.vapor_pressure = vapor_pressure
        self.surface_tension = surface_tension
        
        # Initialize Brownian motion process
        self.brownian = WienerProcess()
        
        # Store history
        self.history = {
            'time': [0.0],
            'position': [position.copy()],
            'radius': [radius],
            'temperature': [temperature]
        }
    
    def update(self, dt: float, ambient_temp: float, ambient_rh: float):
        """Update particle state including all dynamic processes"""
        # Gravitational settling
        g = 9.81
        viscosity = 1.81e-5  # Air viscosity
        
        # Stokes settling velocity
        v_settle = 2 * self.radius**2 * self.density * g / (9 * viscosity)
        
        # Brownian motion displacement
        D = self.calculate_diffusion_coefficient()
        # Replace the WienerProcess with direct numpy random sampling
        dW = np.random.normal(0, 1, size=3)  # 3D random displacement
        dx_brownian = np.sqrt(2 * D * dt) * dW
        
        # Update position
        self.position[2] -= v_settle * dt  # Settling
        self.position += dx_brownian  # Brownian motion
        
        # Evaporation/condensation
        self.update_size(dt, ambient_temp, ambient_rh)
        
        # Heat transfer with environment
        self.temperature += (ambient_temp - self.temperature) * dt / 10
        
        # Update history
        self.history['time'].append(self.history['time'][-1] + dt)
        self.history['position'].append(self.position.copy())
        self.history['radius'].append(self.radius)
        self.history['temperature'].append(self.temperature)
    
    def calculate_diffusion_coefficient(self) -> float:
        """Calculate Brownian diffusion coefficient"""
        kb = 1.380649e-23  # Boltzmann constant
        viscosity = 1.81e-5  # Air viscosity
        return kb * self.temperature / (6 * np.pi * viscosity * self.radius)
    
    def update_size(self, dt: float, ambient_temp: float, ambient_rh: float):
        """Update particle size due to evaporation/condensation"""
        # Kelvin effect on vapor pressure
        p_kelvin = self.vapor_pressure * np.exp(
            2 * self.surface_tension / (self.density * self.radius * 8.314 * self.temperature)
        )
        
        # Mass transfer rate
        dm_dt = 4 * np.pi * self.radius * (ambient_rh * self.vapor_pressure - p_kelvin)
        
        # Update radius
        dV = dm_dt * dt / self.density
        self.radius = (3 * (4/3 * np.pi * self.radius**3 + dV) / (4 * np.pi))**(1/3)

def simulate_aerosol():
    # Create particle
    particle = AerosolParticle(
        radius=1e-6,        # 1 μm
        density=1000,       # 1 g/cm³
        temperature=293.15, # 20°C
        position=np.array([0.0, 0.0, 0.001]),  # Start at 1 mm height
        vapor_pressure=2300,  # Water vapor pressure at 20°C
        surface_tension=0.072 # Water surface tension
    )
    
    # Simulation parameters
    dt = 1e-4
    t_final = 0.1
    
    # Environmental conditions
    ambient_temp = 293.15  # 20°C
    ambient_rh = 0.8      # 80% relative humidity
    
    # Run simulation
    t = 0
    while t < t_final:
        particle.update(dt, ambient_temp, ambient_rh)
        t += dt
    
    return particle

def plot_results(particle):
    """Plot particle trajectory and properties"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Get history arrays
    t = np.array(particle.history['time'])
    pos = np.array(particle.history['position'])
    r = np.array(particle.history['radius'])
    T = np.array(particle.history['temperature'])
    
    # Plot 3D trajectory
    ax1.plot(pos[:,0]*1e6, pos[:,2]*1e6, 'b-', alpha=0.6)
    ax1.set_xlabel('x (μm)')
    ax1.set_ylabel('z (μm)')
    ax1.set_title('Particle Trajectory')
    ax1.grid(True)
    
    # Plot radius evolution
    ax2.plot(t*1000, r*1e6)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Radius (μm)')
    ax2.set_title('Particle Size Evolution')
    ax2.grid(True)
    
    # Plot vertical position
    ax3.plot(t*1000, pos[:,2]*1e3)
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Height (mm)')
    ax3.set_title('Settling Behavior')
    ax3.grid(True)
    
    # Plot temperature
    ax4.plot(t*1000, T-273.15)
    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('Temperature (°C)')
    ax4.set_title('Particle Temperature')
    ax4.grid(True)
    
    plt.tight_layout()

def main():
    # Run simulation
    particle = simulate_aerosol()

    # Plot results
    plot_results(particle)

    # Save plot to output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'aerosol_dynamics.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'aerosol_dynamics.png')}")

if __name__ == "__main__":
    main() 