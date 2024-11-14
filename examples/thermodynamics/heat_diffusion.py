"""
Example demonstrating heat diffusion using the heat equation.

This example solves the 1D heat equation:
∂T/∂t = α * ∂²T/∂x²

where α is the thermal diffusivity (k/(ρ*cp))
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics import ThermalSystem

def solve_heat_equation_1d(
    T0: np.ndarray,          # Initial temperature distribution
    dx: float,               # Spatial step size
    dt: float,               # Time step
    thermal_diffusivity: float,  # α = k/(ρ*cp)
    total_time: float,       # Total simulation time
    boundary_conditions: tuple = ("dirichlet", "dirichlet"),  # Boundary condition types
    boundary_values: tuple = (0, 0)  # Boundary values
) -> tuple:
    """
    Solve 1D heat equation using explicit finite difference method.
    
    Args:
        T0: Initial temperature distribution
        dx: Spatial step size (m)
        dt: Time step (s)
        thermal_diffusivity: Thermal diffusivity (m²/s)
        total_time: Total simulation time (s)
        boundary_conditions: Tuple of boundary condition types ("dirichlet" or "neumann")
        boundary_values: Tuple of boundary values (temperatures or fluxes)
        
    Returns:
        Tuple of (time points, temperature evolution)
    """
    # Stability check (CFL condition)
    stability_factor = thermal_diffusivity * dt / (dx * dx)
    if stability_factor > 0.5:
        raise ValueError(f"Solution will be unstable. Required: dt <= {0.5 * dx * dx / thermal_diffusivity}")
        
    # Initialize arrays
    nx = len(T0)
    nt = int(total_time / dt) + 1  # Add +1 to include the initial time
    T = np.zeros((nt, nx))  # Change array initialization to include time steps
    T[0] = T0
    
    # Time stepping
    for n in range(0, nt-1):
        # Interior points
        for i in range(1, nx-1):
            T[n+1,i] = T[n,i] + thermal_diffusivity * dt / (dx * dx) * \
                       (T[n,i+1] - 2*T[n,i] + T[n,i-1])
        
        # Boundary conditions
        if boundary_conditions[0] == "dirichlet":
            T[n+1,0] = boundary_values[0]
        elif boundary_conditions[0] == "neumann":
            # Forward difference for left boundary
            T[n+1,0] = T[n+1,1] - dx * boundary_values[0]
            
        if boundary_conditions[1] == "dirichlet":
            T[n+1,-1] = boundary_values[1]
        elif boundary_conditions[1] == "neumann":
            # Backward difference for right boundary
            T[n+1,-1] = T[n+1,-2] + dx * boundary_values[1]
    
    return np.linspace(0, total_time, nt), T

def simulate_heat_diffusion():
    # Material properties (copper)
    thermal_conductivity = 385.0  # W/(m·K)
    density = 8960.0             # kg/m³
    specific_heat = 386.0        # J/(kg·K)
    
    # Calculate thermal diffusivity
    thermal_diffusivity = thermal_conductivity / (density * specific_heat)
    
    # Simulation parameters
    L = 0.1                      # Length of rod (m)
    nx = 50                      # Number of spatial points
    dx = L / (nx - 1)           # Spatial step size
    dt = 0.001                  # Time step (s)
    total_time = 5.0            # Total simulation time (s)
    
    # Initial temperature distribution (Gaussian pulse)
    x = np.linspace(0, L, nx)
    T0 = 20 + 80 * np.exp(-(x - L/2)**2 / (0.01 * L)**2)  # Temperature in °C
    
    # Solve heat equation
    t, T = solve_heat_equation_1d(
        T0,
        dx,
        dt,
        thermal_diffusivity,
        total_time,
        boundary_conditions=("dirichlet", "dirichlet"),
        boundary_values=(20, 20)  # Fixed temperature at boundaries (20°C)
    )
    
    return x, t, T

def plot_results(x, t, T):
    """Plot temperature evolution"""
    plt.figure(figsize=(12, 8))
    
    # Plot temperature distribution at different times
    times_to_plot = [0, 0.1, 0.5, 1.0, 2.0, 5.0]
    for time in times_to_plot:
        idx = int(time / (t[1] - t[0]))
        if idx >= len(t):
            continue
        plt.plot(x * 100, T[idx], label=f't = {time:.1f} s')
    
    plt.xlabel('Position (cm)')
    plt.ylabel('Temperature (°C)')
    plt.title('Heat Diffusion in a Copper Rod')
    plt.grid(True)
    plt.legend()
    
    # Add colormap plot
    plt.figure(figsize=(12, 8))
    plt.pcolormesh(x * 100, t, T, shading='auto', cmap='hot')
    plt.colorbar(label='Temperature (°C)')
    plt.xlabel('Position (cm)')
    plt.ylabel('Time (s)')
    plt.title('Temperature Evolution Over Time')

def main():
    # Run simulation
    x, t, T = simulate_heat_diffusion()
    
    # Plot results
    plot_results(x, t, T)
    plt.show()

if __name__ == "__main__":
    main() 