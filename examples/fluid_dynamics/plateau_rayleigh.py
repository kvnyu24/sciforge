"""
Example demonstrating the Plateau-Rayleigh instability simulation.

This example shows how a fluid column breaks up into droplets due to 
surface tension effects, a phenomenon commonly observed in water jets
and falling streams of liquid.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics import FluidColumn

def simulate_plateau_rayleigh():
    # Create water column
    column = FluidColumn(
        radius=0.001,  # 1mm
        length=0.02,   # 2cm
        density=1000,  # water density
        surface_tension=0.072,  # water surface tension
        viscosity=0.001  # water viscosity
    )

    # Add small sinusoidal perturbation
    z = column.z
    column.r += 0.0001 * np.sin(2 * np.pi * z / column.length)

    # Time evolution parameters
    dt = 1e-6
    t_final = 0.001

    # Run simulation
    while column.history['time'][-1] < t_final:
        column.update(dt)
        
    return column

def plot_results(column):
    """Plot the initial and final states of the fluid column"""
    plt.figure(figsize=(10, 6))
    
    # Plot initial and final states
    times = [0, -1]  # Initial and final states
    for t_idx in times:
        r = column.history['radius'][t_idx]
        plt.plot(column.z * 1000, r * 1000, 
                label=f"t = {column.history['time'][t_idx]*1000:.1f} ms")
    
    plt.xlabel('z (mm)')
    plt.ylabel('r (mm)')
    plt.title('Plateau-Rayleigh Instability')
    plt.legend()
    plt.grid(True)
    
    # Add theoretical growth rate annotation
    omega = column.calculate_growth_rate()
    plt.annotate(f'Theoretical growth rate: {omega:.2f} s⁻¹', 
                xy=(0.02, 0.98), xycoords='axes fraction',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

def main():
    # Run simulation
    column = simulate_plateau_rayleigh()
    
    # Plot results
    plot_results(column)
    plt.show()

if __name__ == "__main__":
    main() 