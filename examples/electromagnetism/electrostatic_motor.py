"""
Example demonstrating an electrostatic motor simulation.

This example shows how charged plates create electric fields that induce
rotational motion in a charged rotor, converting electrical potential
energy into mechanical kinetic energy. The simulation tracks various 
physical quantities including:
- Angular motion and acceleration
- Electric field strength and distribution
- Forces and torques on rotor elements
- Energy conversion and conservation
- System efficiency
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from src.sciforge.physics import ElectricField, Particle

class ElectrostaticMotor:
    """Class representing a simple electrostatic motor"""
    
    def __init__(self,
                 rotor_radius: float,      # Radius of rotor (m)
                 rotor_charge: float,      # Charge on rotor elements (C)
                 stator_charge: float,     # Charge on stator plates (C)
                 stator_radius: float,     # Radius of stator arrangement (m)
                 n_rotor_elements: int=4,  # Number of charged elements on rotor
                 n_stator_plates: int=4,   # Number of stator plates
                 damping: float=0.1,       # Damping coefficient
                 initial_velocity: float=5.0): # Initial angular velocity (rad/s)
        
        # Initialize geometry
        self.rotor_radius = rotor_radius
        self.stator_radius = stator_radius
        self.damping = damping
        
        # Create rotor elements as particles
        self.rotor_elements = []
        for i in range(n_rotor_elements):
            angle = 2 * np.pi * i / n_rotor_elements
            pos = rotor_radius * np.array([np.cos(angle), np.sin(angle), 0])
            self.rotor_elements.append(
                Particle(mass=1e-3, position=pos, velocity=np.zeros(3))
            )
            
        # Create stator electric fields with time-varying charges
        self.stator_fields = []
        for i in range(n_stator_plates):
            angle = 2 * np.pi * i / n_stator_plates
            # Alternate charges on stator plates
            charge = stator_charge if i % 2 == 0 else -stator_charge
            self.stator_fields.append(ElectricField(charge))
            
        # Store parameters
        self.rotor_charge = rotor_charge
        self.angular_velocity = initial_velocity  # Non-zero initial velocity
        self.angular_acceleration = 0.0
        self.angle = 0.0
        self.time = 0.0
        
        # Store history
        self.history = {
            'time': [0.0],
            'angle': [0.0],
            'angular_velocity': [initial_velocity],
            'angular_acceleration': [0.0],
            'kinetic_energy': [self.kinetic_energy()],
            'potential_energy': [self.potential_energy()],
            'total_energy': [self.total_energy()],
            'torque': [0.0],
            'power': [0.0],
            'field_strength': [self.average_field_strength()],
            'efficiency': [100.0]
        }
        
    def kinetic_energy(self) -> float:
        """Calculate rotational kinetic energy"""
        # Add time-varying moment of inertia
        moment_of_inertia = sum(
            (self.rotor_radius * (1 + 0.1 * np.sin(2*self.time)))**2 * p.mass 
            for p in self.rotor_elements
        )
        return 0.5 * moment_of_inertia * self.angular_velocity**2
        
    def potential_energy(self) -> float:
        """Calculate electric potential energy with time-varying field"""
        potential = 0.0
        for particle in self.rotor_elements:
            for i, field in enumerate(self.stator_fields):
                # Add oscillating component to field strength
                field_mod = field.charge * (1 + 0.2 * np.sin(3*self.time + i*np.pi/4))
                potential += self.rotor_charge * field_mod * field.potential(particle.position)
        return potential
        
    def total_energy(self) -> float:
        return self.kinetic_energy() + self.potential_energy()
        
    def average_field_strength(self) -> float:
        total_field = 0.0
        for particle in self.rotor_elements:
            field = np.zeros(3)
            for i, stator in enumerate(self.stator_fields):
                # Time-varying field strength
                field_mod = stator.charge * (1 + 0.2 * np.sin(3*self.time + i*np.pi/4))
                field += field_mod * stator.field(particle.position)
            total_field += np.linalg.norm(field)
        return total_field / len(self.rotor_elements)
        
    def update(self, dt: float):
        """Update motor state"""
        self.time += dt
        
        # Calculate total torque on rotor
        torque = 0.0
        for particle in self.rotor_elements:
            force = np.zeros(3)
            for i, field in enumerate(self.stator_fields):
                # Time-varying field strength
                field_mod = field.charge * (1 + 0.2 * np.sin(3*self.time + i*np.pi/4))
                force += self.rotor_charge * field_mod * field.field(particle.position)
            
            r = particle.position[:2]
            f = force[:2]
            torque += np.cross(r, f)
            
        # Add nonlinear damping
        damping_torque = -self.damping * (self.angular_velocity + 0.1*self.angular_velocity**2)
        torque += damping_torque
            
        # Update angular motion with time-varying moment of inertia
        moment_of_inertia = sum(
            (self.rotor_radius * (1 + 0.1 * np.sin(2*self.time)))**2 * p.mass 
            for p in self.rotor_elements
        )
        
        self.angular_acceleration = torque / moment_of_inertia
        self.angular_velocity += self.angular_acceleration * dt
        self.angle += self.angular_velocity * dt
        
        # Calculate power and efficiency
        power = torque * self.angular_velocity
        energy_change = self.total_energy() - self.history['total_energy'][-1]
        efficiency = 100 * (1 - abs(energy_change) / (power * dt + 1e-10))
        
        # Update rotor element positions
        for i, particle in enumerate(self.rotor_elements):
            base_angle = 2 * np.pi * i / len(self.rotor_elements)
            angle = base_angle + self.angle
            # Add radial oscillation
            r = self.rotor_radius * (1 + 0.1 * np.sin(2*self.time))
            particle.position = r * np.array([np.cos(angle), np.sin(angle), 0])
            
        # Store history
        self.history['time'].append(self.time)
        self.history['angle'].append(self.angle)
        self.history['angular_velocity'].append(self.angular_velocity)
        self.history['angular_acceleration'].append(self.angular_acceleration)
        self.history['kinetic_energy'].append(self.kinetic_energy())
        self.history['potential_energy'].append(self.potential_energy())
        self.history['total_energy'].append(self.total_energy())
        self.history['torque'].append(torque)
        self.history['power'].append(power)
        self.history['field_strength'].append(self.average_field_strength())
        self.history['efficiency'].append(efficiency)

def simulate_motor():
    # Create motor with higher initial velocity and lower damping
    motor = ElectrostaticMotor(
        rotor_radius=0.05,     # 5cm
        rotor_charge=1e-9,     # 1 nC
        stator_charge=1e-8,    # 10 nC
        stator_radius=0.07,    # 7cm
        n_rotor_elements=6,    # Increased number of elements
        n_stator_plates=8,     # Increased number of plates
        damping=0.01,         # Reduced damping
        initial_velocity=10.0  # Higher initial velocity
    )
    
    # Simulation parameters
    dt = 1e-4
    t_final = 2.0  # Longer simulation time
    
    # Run simulation
    t = 0
    while t < t_final:
        motor.update(dt)
        t += dt
        
    return motor

def plot_results(motor):
    """Create comprehensive plots of motor behavior"""
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 3)
    
    # Plot 1: Motor animation (3D)
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    
    def update(frame):
        ax1.clear()
        
        # Plot stator plates
        for i, field in enumerate(motor.stator_fields):
            angle = 2 * np.pi * i / len(motor.stator_fields)
            x = motor.stator_radius * np.cos(angle)
            y = motor.stator_radius * np.sin(angle)
            # Time-varying colors based on charge, but clamp alpha to [0,1]
            field_strength = 1 + 0.2 * np.sin(3*motor.history['time'][frame] + i*np.pi/4)
            color = 'r' if field.charge > 0 else 'b'
            alpha = min(1.0, max(0.0, abs(field_strength)))  # Clamp alpha between 0 and 1
            ax1.scatter(x, y, 0, c=color, s=100, alpha=alpha)
            
        # Plot rotor elements with time-varying radius
        angle = motor.history['angle'][frame]
        r = motor.rotor_radius * (1 + 0.1 * np.sin(2*motor.history['time'][frame]))
        for i in range(len(motor.rotor_elements)):
            base_angle = 2 * np.pi * i / len(motor.rotor_elements)
            theta = base_angle + angle
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            ax1.scatter(x, y, 0, c='k', s=50)
            
        ax1.set_xlim(-0.1, 0.1)
        ax1.set_ylim(-0.1, 0.1)
        ax1.set_zlim(-0.1, 0.1)
        ax1.set_title(f'Motor State (t = {motor.history["time"][frame]:.3f} s)')
        
    # Plot 2: Angular motion
    ax2 = fig.add_subplot(gs[0, 1:])
    t = motor.history['time']
    ax2.plot(t, motor.history['angle'], label='Angle (rad)')
    ax2.plot(t, motor.history['angular_velocity'], label='Angular Velocity (rad/s)')
    ax2.plot(t, motor.history['angular_acceleration'], label='Angular Acceleration (rad/s²)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Angular Motion')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Energy components
    ax3 = fig.add_subplot(gs[1, 0:2])
    ax3.plot(t, motor.history['kinetic_energy'], label='Kinetic')
    ax3.plot(t, motor.history['potential_energy'], label='Potential')
    ax3.plot(t, motor.history['total_energy'], label='Total')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Energy (J)')
    ax3.legend()
    ax3.grid(True)
    
    # Plot 4: Power and Efficiency
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(t, motor.history['power'], 'g-', label='Power')
    ax4_twin = ax4.twinx()
    ax4_twin.plot(t, motor.history['efficiency'], 'r--', label='Efficiency')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Power (W)')
    ax4_twin.set_ylabel('Efficiency (%)')
    ax4.grid(True)
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2)
    
    # Plot 5: Field Strength and Torque
    ax5 = fig.add_subplot(gs[2, :])
    ax5.plot(t, motor.history['field_strength'], label='Field Strength')
    ax5_twin = ax5.twinx()
    ax5_twin.plot(t, motor.history['torque'], 'r-', label='Torque')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Field Strength (N/C)')
    ax5_twin.set_ylabel('Torque (N⋅m)')
    ax5.grid(True)
    lines1, labels1 = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5_twin.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, labels1 + labels2)
    
    plt.tight_layout()
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(motor.history['time']),
                        interval=50, blit=False)
    return anim

def main():
    # Run simulation
    motor = simulate_motor()
    
    # Plot results
    anim = plot_results(motor)
    plt.show()

if __name__ == "__main__":
    main() 