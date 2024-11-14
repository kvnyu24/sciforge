"""
Example demonstrating heat dissipation between multiple thermal systems.

This example shows how heat transfers between three thermal systems with different
initial temperatures until they reach thermal equilibrium.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics import ThermalSystem

def simulate_heat_dissipation():
    # Create three thermal systems with different initial temperatures
    system1 = ThermalSystem(
        temperature=373.15,  # 100°C
        mass=1.0,           # 1 kg
        specific_heat=4186,  # Water specific heat J/(kg·K)
        thermal_conductivity=0.6  # Water thermal conductivity W/(m·K)
    )
    
    system2 = ThermalSystem(
        temperature=293.15,  # 20°C
        mass=2.0,           # 2 kg
        specific_heat=4186,  # Water
        thermal_conductivity=0.6
    )
    
    system3 = ThermalSystem(
        temperature=283.15,  # 10°C
        mass=1.5,           # 1.5 kg
        specific_heat=4186,  # Water
        thermal_conductivity=0.6
    )
    
    # Simulation parameters
    dt = 0.1  # Time step (s)
    t_final = 100  # Total simulation time (s)
    time = 0
    
    # Run simulation
    while time < t_final:
        # Calculate heat transfer between systems
        # Contact area and distance are arbitrary for this example
        system1.conductive_heat_transfer(system2, contact_area=0.01, distance=0.05, time=dt)
        system2.conductive_heat_transfer(system3, contact_area=0.01, distance=0.05, time=dt)
        
        # Update history
        system1.update_history(time)
        system2.update_history(time)
        system3.update_history(time)
        
        time += dt
        
    return system1, system2, system3

def plot_results(system1, system2, system3):
    """Plot temperature evolution of all systems"""
    plt.figure(figsize=(10, 6))
    
    # Convert temperatures to Celsius for plotting
    t1 = np.array(system1.history['temperature']) - 273.15
    t2 = np.array(system2.history['temperature']) - 273.15
    t3 = np.array(system3.history['temperature']) - 273.15
    
    plt.plot(system1.history['time'], t1, 'r-', label='System 1 (Initial: 100°C)')
    plt.plot(system2.history['time'], t2, 'b-', label='System 2 (Initial: 20°C)')
    plt.plot(system3.history['time'], t3, 'g-', label='System 3 (Initial: 10°C)')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (°C)')
    plt.title('Heat Dissipation Between Three Thermal Systems')
    plt.grid(True)
    plt.legend()
    
    # Add final temperature annotation
    final_temp = (t1[-1] + t2[-1] + t3[-1]) / 3
    plt.annotate(f'Final equilibrium temperature: {final_temp:.1f}°C',
                xy=(0.02, 0.98), xycoords='axes fraction',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

def main():
    # Run simulation
    system1, system2, system3 = simulate_heat_dissipation()
    
    # Plot results
    plot_results(system1, system2, system3)
    plt.show()

if __name__ == "__main__":
    main() 