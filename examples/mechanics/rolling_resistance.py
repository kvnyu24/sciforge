"""
Example demonstrating rolling resistance simulation.

This example shows how rolling resistance affects the motion of a rolling object,
taking into account factors like material deformation and surface conditions.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics import Particle
from src.sciforge.differential import RungeKutta4

class RollingObject(Particle):
    """Class representing a rolling object with rolling resistance"""
    
    def __init__(self, mass, radius, crr, position=None, velocity=None):
        """
        Initialize rolling object
        
        Args:
            mass: Object mass (kg)
            radius: Object radius (m)
            crr: Coefficient of rolling resistance
            position: Initial position (default: origin)
            velocity: Initial velocity (default: at rest)
        """
        position = position if position is not None else np.zeros(3)
        velocity = velocity if velocity is not None else np.zeros(3)
        super().__init__(mass, position, velocity)
        
        self.radius = radius
        self.crr = crr
        
    def rolling_resistance_force(self):
        """Calculate rolling resistance force"""
        if np.allclose(self.velocity, 0):
            return np.zeros(3)
            
        # Direction opposite to motion
        direction = -self.velocity / np.linalg.norm(self.velocity)
        
        # Force magnitude = crr * N, where N is normal force (mg)
        magnitude = self.crr * self.mass * self.gravity
        
        return magnitude * direction

def simulate_rolling():
    # Create rolling object (e.g., a steel ball on concrete)
    ball = RollingObject(
        mass=1.0,      # 1 kg
        radius=0.05,   # 5 cm radius
        crr=0.001,     # Steel on concrete ≈ 0.001
        position=np.array([0.0, 0.0, 0.0]),
        velocity=np.array([5.0, 0.0, 0.0])  # Initial velocity 5 m/s in x direction
    )
    
    # Simulation parameters
    dt = 0.01
    t_final = 10.0
    times = np.arange(0, t_final, dt)
    
    # Store results
    positions = []
    velocities = []
    energies = []
    
    # Run simulation
    for t in times:
        # Calculate total force including rolling resistance
        rolling_force = ball.rolling_resistance_force()
        
        # Update particle state
        ball.update(rolling_force, dt)
        
        # Store results
        positions.append(ball.position.copy())
        velocities.append(ball.velocity.copy())
        energies.append(0.5 * ball.mass * np.linalg.norm(ball.velocity)**2)
    
    return times, np.array(positions), np.array(velocities), np.array(energies)

def plot_results(times, positions, velocities, energies):
    """Plot motion and energy of rolling object"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot position
    ax1.plot(times, positions[:, 0], label='x')
    ax1.plot(times, positions[:, 1], label='y')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (m)')
    ax1.set_title('Position vs Time')
    ax1.grid(True)
    ax1.legend()
    
    # Plot velocity
    ax2.plot(times, velocities[:, 0], label='v_x')
    ax2.plot(times, velocities[:, 1], label='v_y')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('Velocity vs Time')
    ax2.grid(True)
    ax2.legend()
    
    # Plot energy
    ax3.plot(times, energies)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Kinetic Energy (J)')
    ax3.set_title('Energy vs Time')
    ax3.grid(True)
    
    plt.tight_layout()

def main():
    # Run simulation
    times, positions, velocities, energies = simulate_rolling()
    
    # Plot results
    plot_results(times, positions, velocities, energies)
    plt.show()

if __name__ == "__main__":
    main() 