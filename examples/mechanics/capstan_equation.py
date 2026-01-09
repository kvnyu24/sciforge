"""
Example demonstrating the Capstan equation for rope friction.

The Capstan equation describes how friction increases the holding force when a rope
is wrapped around a cylinder, explaining why a small force can hold a much larger load.
This example provides a comprehensive visualization of the physics involved.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc, Arrow
from src.sciforge.physics import PhysicalSystem

class CapstanSystem(PhysicalSystem):
    """Class representing a rope wrapped around a cylinder"""
    
    def __init__(self, 
                 cylinder_radius: float,
                 friction_coeff: float,
                 wrap_angle: float,
                 rope_tension: float):
        """
        Initialize Capstan system
        
        Args:
            cylinder_radius: Radius of cylinder (m)
            friction_coeff: Coefficient of friction between rope and cylinder
            wrap_angle: Angle of rope wrap around cylinder (radians)
            rope_tension: Initial tension in rope (N)
        """
        super().__init__(mass=1.0, position=np.zeros(3))
        self.radius = cylinder_radius
        self.mu = friction_coeff
        self.angle = wrap_angle
        self.T0 = rope_tension
        
        # Calculate holding force using Capstan equation
        self.T1 = self.T0 * np.exp(self.mu * self.angle)
        
        # Store data for plotting
        self.angles = np.linspace(0, self.angle, 100)
        self.tensions = self.T0 * np.exp(self.mu * self.angles)
        
        # Calculate rope coordinates for visualization
        theta = np.linspace(0, wrap_angle, 100)
        self.rope_x = cylinder_radius * np.cos(theta)
        self.rope_y = cylinder_radius * np.sin(theta)
        
    def calculate_tension(self, angle: float) -> float:
        """Calculate tension at given angle"""
        return self.T0 * np.exp(self.mu * angle)
    
    def mechanical_advantage(self) -> float:
        """Calculate mechanical advantage (ratio of output to input force)"""
        return np.exp(self.mu * self.angle)
    
    def get_force_vectors(self, scale=0.05):
        """Get force vectors for visualization"""
        # Input force vector
        start_x = self.rope_x[0]
        start_y = self.rope_y[0]
        angle_in = np.arctan2(self.rope_y[1] - self.rope_y[0], 
                             self.rope_x[1] - self.rope_x[0])
        
        # Output force vector
        end_x = self.rope_x[-1]
        end_y = self.rope_y[-1]
        angle_out = np.arctan2(self.rope_y[-1] - self.rope_y[-2],
                              self.rope_x[-1] - self.rope_x[-2])
        
        return {
            'input': (start_x, start_y, -scale * self.T0 * np.cos(angle_in),
                     -scale * self.T0 * np.sin(angle_in)),
            'output': (end_x, end_y, scale * self.T1 * np.cos(angle_out),
                      scale * self.T1 * np.sin(angle_out))
        }

def simulate_capstan():
    # Create system with typical values
    system = CapstanSystem(
        cylinder_radius=0.1,    # 10cm radius
        friction_coeff=0.3,     # Typical for rope on metal
        wrap_angle=2*np.pi,     # One full wrap
        rope_tension=100.0      # 100N input tension
    )
    
    return system

def plot_results(system):
    """Create comprehensive visualization of the Capstan system"""
    fig = plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(2, 2)
    
    # Physical system visualization
    ax1 = fig.add_subplot(gs[0, 0])
    plot_physical_system(ax1, system)
    
    # Tension distribution
    ax2 = fig.add_subplot(gs[0, 1])
    plot_tension_distribution(ax2, system)
    
    # Mechanical advantage plot
    ax3 = fig.add_subplot(gs[1, 0])
    plot_mechanical_advantage(ax3, system)
    
    # Parameter sensitivity plot
    ax4 = fig.add_subplot(gs[1, 1])
    plot_parameter_sensitivity(ax4, system)
    
    plt.tight_layout()

def plot_physical_system(ax, system):
    """Plot physical representation of the system"""
    # Draw cylinder
    circle = Circle((0, 0), system.radius, fill=False, color='black')
    ax.add_patch(circle)
    
    # Draw rope
    ax.plot(system.rope_x, system.rope_y, 'b-', linewidth=2, label='Rope')
    
    # Draw force vectors
    vectors = system.get_force_vectors()
    ax.quiver(vectors['input'][0], vectors['input'][1],
             vectors['input'][2], vectors['input'][3],
             color='red', scale=1, scale_units='xy',
             label=f'Input Force ({system.T0:.0f}N)')
    ax.quiver(vectors['output'][0], vectors['output'][1],
             vectors['output'][2], vectors['output'][3],
             color='green', scale=1, scale_units='xy',
             label=f'Output Force ({system.T1:.0f}N)')
    
    ax.set_aspect('equal')
    ax.set_title('Physical System')
    ax.legend()
    ax.grid(True)

def plot_tension_distribution(ax, system):
    """Plot tension distribution along rope"""
    ax.plot(system.angles, system.tensions, 'b-', linewidth=2)
    ax.fill_between(system.angles, system.tensions, alpha=0.2)
    
    ax.set_xlabel('Wrap Angle (radians)')
    ax.set_ylabel('Tension (N)')
    ax.set_title('Tension Distribution')
    ax.grid(True)
    
    # Add annotations
    ax.annotate(f'Input Tension: {system.T0:.1f} N',
                xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.annotate(f'Output Tension: {system.T1:.1f} N',
                xy=(0.05, 0.85), xycoords='axes fraction',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

def plot_mechanical_advantage(ax, system):
    """Plot mechanical advantage vs wrap angle"""
    test_angles = np.linspace(0, 4*np.pi, 100)
    mech_adv = np.exp(system.mu * test_angles)
    
    ax.plot(test_angles, mech_adv, 'g-', linewidth=2)
    ax.axvline(x=system.angle, color='r', linestyle='--',
               label=f'Current angle: {system.angle/np.pi:.1f}π')
    ax.axhline(y=system.mechanical_advantage(), color='r', linestyle='--')
    
    ax.set_xlabel('Wrap Angle (radians)')
    ax.set_ylabel('Mechanical Advantage')
    ax.set_title('Mechanical Advantage vs Wrap Angle')
    ax.grid(True)
    ax.legend()

def plot_parameter_sensitivity(ax, system):
    """Plot sensitivity to friction coefficient"""
    mu_range = np.linspace(0.1, 0.5, 5)
    angles = np.linspace(0, 2*np.pi, 100)
    
    for mu in mu_range:
        ma = np.exp(mu * angles)
        ax.plot(angles, ma, label=f'μ = {mu:.1f}')
    
    ax.set_xlabel('Wrap Angle (radians)')
    ax.set_ylabel('Mechanical Advantage')
    ax.set_title('Sensitivity to Friction Coefficient')
    ax.grid(True)
    ax.legend()

def main():
    # Run simulation
    system = simulate_capstan()

    # Plot results
    plot_results(system)

    # Save plot to output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'capstan_equation.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'capstan_equation.png')}")

if __name__ == "__main__":
    main() 