"""
Example demonstrating the Coandă effect simulation.

This example shows how a fluid jet tends to follow a curved surface due to the 
pressure difference created by the fluid flow, a phenomenon known as the Coandă effect.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics import FluidJet
from src.sciforge.numerical.integration import trapezoid

def create_curved_surface():
    """Create a curved surface for the Coandă effect"""
    theta = np.linspace(0, np.pi/2, 100)
    radius = 0.05  # 5cm radius of curvature
    x = radius * np.cos(theta)
    y = radius * (1 - np.sin(theta))
    return x, y

def simulate_coanda_effect():
    # Initialize fluid jet
    jet = FluidJet(
        velocity=10.0,      # 10 m/s initial velocity
        density=1.225,      # air density kg/m³
        viscosity=1.81e-5,  # air viscosity
        width=0.005,        # 5mm jet width
        n_points=20         # Reduced number of points to avoid array shape mismatch
    )
    
    # Create curved surface
    surface_x, surface_y = create_curved_surface()
    
    # Time evolution parameters
    dt = 0.001
    t_final = 0.1
    
    # Run simulation
    t = 0
    while t < t_final:
        jet.update(dt, surface_x, surface_y)
        t += dt
        
    return jet, surface_x, surface_y

def plot_results(jet, surface_x, surface_y):
    """Plot the fluid jet trajectory and curved surface"""
    plt.figure(figsize=(10, 8))
    
    # Plot surface
    plt.plot(surface_x, surface_y, 'k-', linewidth=2, label='Curved Surface')
    
    # Plot jet streamlines
    streamlines = jet.get_streamlines()
    for streamline in streamlines:
        plt.plot(streamline[0, :], streamline[1, :], 'b-', alpha=0.3)
    
    # Plot velocity magnitude contours
    x, y = np.meshgrid(np.linspace(-0.02, 0.06, 50), np.linspace(0, 0.08, 50))
    vel_mag = jet.get_velocity_field(x, y)
    plt.contourf(x, y, vel_mag, levels=20, cmap='viridis', alpha=0.5)
    plt.colorbar(label='Velocity (m/s)')
    
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Coandă Effect Simulation')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()

def main():
    # Run simulation
    jet, surface_x, surface_y = simulate_coanda_effect()
    
    # Plot results
    plot_results(jet, surface_x, surface_y)
    plt.show()

if __name__ == "__main__":
    main()