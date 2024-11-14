"""
Example demonstrating Hamiltonian mechanics simulation of a pendulum.

This example shows how to simulate a pendulum using Hamilton's equations,
demonstrating conservation of energy and phase space trajectories.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.differential import RungeKutta4
from src.sciforge.physics import PhysicalSystem

class HamiltonianPendulum(PhysicalSystem):
    """Simple pendulum using Hamiltonian mechanics"""
    
    def __init__(self, length=1.0, mass=1.0, g=9.81):
        """
        Initialize pendulum
        
        Args:
            length: Pendulum length (m)
            mass: Bob mass (kg)
            g: Gravitational acceleration (m/s²)
        """
        super().__init__(mass=mass, position=np.array([0.0]))
        self.length = length
        self.g = g
        self.history = {
            'time': [],
            'theta': [],
            'p_theta': [],
            'energy': []
        }
        
    def hamiltonian(self, theta, p_theta):
        """Calculate system Hamiltonian (total energy)"""
        T = p_theta**2 / (2 * self.mass * self.length**2)  # Kinetic energy
        V = self.mass * self.g * self.length * (1 - np.cos(theta))  # Potential energy
        return T + V
        
    def hamilton_equations(self, t, state):
        """
        Implement Hamilton's equations of motion
        
        dθ/dt = ∂H/∂p
        dp/dt = -∂H/∂θ
        """
        theta, p_theta = state
        
        # dθ/dt = ∂H/∂p = p/(ml²)
        dtheta_dt = p_theta / (self.mass * self.length**2)
        
        # dp/dt = -∂H/∂θ = -mgl*sin(θ)
        dp_dt = -self.mass * self.g * self.length * np.sin(theta)
        
        return np.array([dtheta_dt, dp_dt])

def simulate_pendulum():
    # Create pendulum
    pendulum = HamiltonianPendulum(length=1.0, mass=1.0)
    
    # Initial conditions (angle and angular momentum)
    theta0 = np.pi/2  # 90 degrees
    p0 = 0.0         # Starting from rest
    y0 = np.array([theta0, p0])
    
    # Time parameters
    t_span = (0, 10)
    dt = 0.01
    
    # Create solver
    solver = RungeKutta4()
    
    # Solve Hamilton's equations
    t, y = solver.solve(pendulum.hamilton_equations, y0, t_span, dt)
    
    # Store results
    pendulum.history['time'] = t
    pendulum.history['theta'] = y[:, 0]
    pendulum.history['p_theta'] = y[:, 1]
    
    # Calculate energy at each point
    pendulum.history['energy'] = [
        pendulum.hamiltonian(theta, p) 
        for theta, p in zip(y[:, 0], y[:, 1])
    ]
    
    return pendulum

def plot_results(pendulum):
    """Plot pendulum motion and phase space trajectory"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot angle vs time
    ax1.plot(pendulum.history['time'], pendulum.history['theta'])
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('θ (rad)')
    ax1.set_title('Pendulum Angle vs Time')
    ax1.grid(True)
    
    # Plot phase space trajectory
    ax2.plot(pendulum.history['theta'], pendulum.history['p_theta'])
    ax2.set_xlabel('θ (rad)')
    ax2.set_ylabel('p_θ (kg⋅m²/s)')
    ax2.set_title('Phase Space Trajectory')
    ax2.grid(True)
    
    # Plot energy conservation
    ax3.plot(pendulum.history['time'], pendulum.history['energy'])
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Energy (J)')
    ax3.set_title('Total Energy vs Time')
    ax3.grid(True)
    
    plt.tight_layout()

def main():
    # Run simulation
    pendulum = simulate_pendulum()
    
    # Plot results
    plot_results(pendulum)
    plt.show()

if __name__ == "__main__":
    main() 