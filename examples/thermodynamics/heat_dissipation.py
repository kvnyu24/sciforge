"""
Example demonstrating dynamic heat dissipation between multiple thermal systems.

This example shows how heat transfers between three thermal systems with different
initial temperatures until they reach thermal equilibrium. The simulation includes:

- Conductive heat transfer with nonlinear thermal conductivity
- Temperature-dependent thermal properties with hysteresis effects
- Environmental heat exchange through convection and radiation
- Thermal mass effects with phase transitions
- Visualization of heat flow directions and temperature gradients
- Real-time temperature distribution and 3D heatmap
- Energy conservation tracking with dissipation
- Dynamic system parameters with coupling effects
- Advanced differential equation solvers for nonlinear evolution
- Enhanced visualization with interactive 3D plots and animations
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from src.sciforge.physics import ThermalSystem
from matplotlib.patches import Arrow, Circle, FancyArrowPatch
from matplotlib import colors
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

def differential_equations(t, y, systems, ambient_temp, air_flow_base, contact_area_base):
    """Define system of differential equations for temperature evolution with nonlinearities
    
    Args:
        t: Time variable
        y: Array of temperatures [T1, T2, T3]
        systems: List of ThermalSystem objects
        ambient_temp: Base ambient temperature
        air_flow_base: Base air flow rate
        contact_area_base: Base contact area between systems
    
    Returns:
        Array of temperature derivatives [dT1/dt, dT2/dt, dT3/dt]
    """
    # Handle the input array - take only first 3 elements
    y = np.asarray(y).flatten()[:3]
    
    T1, T2, T3 = y[0], y[1], y[2]
    
    # Update system temperatures
    systems[0].temperature = T1
    systems[1].temperature = T2 
    systems[2].temperature = T3
    
    # Dynamic conditions with chaotic variations
    ambient_temp_current = (ambient_temp + 
                          5*np.sin(2*np.pi*t/50) + 
                          2*np.cos(2*np.pi*t/25) +
                          3*np.sin(2*np.pi*t/15))
    
    air_flow = (air_flow_base * 
                (1 + 0.3*np.sin(2*np.pi*t/30) + 
                 0.1*np.cos(2*np.pi*t/15) +
                 0.2*np.sin(2*np.pi*t/10)))
    
    contact_area = (contact_area_base * 
                   (1 + 0.2*np.sin(t/5) + 
                    0.1*np.cos(t/10) +
                    0.15*np.sin(t/7)))
    
    # Nonlinear temperature-dependent properties
    for sys in systems:
        # Thermal conductivity with hysteresis
        delta_T = sys.temperature - 293.15
        sys.thermal_conductivity = (0.6 * (1 + 0.002*delta_T + 
                                         0.0001*delta_T**2 +
                                         0.00005*delta_T**3))
        
        # Specific heat with phase transition effects
        sys.specific_heat = (4186 * (1 + 0.001*delta_T + 
                                   0.0002*delta_T**2 +
                                   0.1*np.tanh((sys.temperature-350)/10)))
    
    # Heat transfer with nonlinear coupling
    distance = 0.05 * (1 + 0.1*np.sin(t/20) + 0.05*np.cos(t/15))
    
    # Nonlinear conduction with temperature-dependent resistance
    dQ12_dt = (systems[0].thermal_conductivity * contact_area * 
               (T1 - T2) / distance * (1 + 0.001*(T1 - T2)**2))
    dQ23_dt = (systems[1].thermal_conductivity * contact_area * 
               (T2 - T3) / distance * (1 + 0.001*(T2 - T3)**2))
    
    # Enhanced convection with turbulent flow effects
    conv_coeff = [(10 + 6*air_flow) * 
                  (1 + 0.2*np.sin(2*np.pi*t/10) + 
                   0.1*np.cos(2*np.pi*t/5) + 
                   0.05*np.sin(2*np.pi*t/2)) * 
                  (1 + 0.001*(temp - ambient_temp_current)**2)
                  for temp in y]
    
    # Add radiation effects
    stefan_boltzmann = 5.67e-8
    emissivity = 0.95
    radiation = [emissivity * stefan_boltzmann * sys.surface_area * 
                (ambient_temp_current**4 - T**4) for sys, T in zip(systems, y)]
    
    dQ1_conv = conv_coeff[0] * systems[0].surface_area * (ambient_temp_current - T1)
    dQ2_conv = conv_coeff[1] * systems[1].surface_area * (ambient_temp_current - T2)
    dQ3_conv = conv_coeff[2] * systems[2].surface_area * (ambient_temp_current - T3)
    
    # Temperature derivatives with coupled nonlinear effects
    dT1 = ((dQ1_conv - dQ12_dt + radiation[0]) / 
           (systems[0].mass * systems[0].specific_heat))
    dT2 = ((dQ2_conv + dQ12_dt - dQ23_dt + radiation[1]) / 
           (systems[1].mass * systems[1].specific_heat))
    dT3 = ((dQ3_conv + dQ23_dt + radiation[2]) / 
           (systems[2].mass * systems[2].specific_heat))
    
    return np.array([dT1, dT2, dT3])

def solve_temperature_evolution(systems, t_span, t_eval, ambient_temp, air_flow_base):
    """Solve temperature evolution using differential equation solver
    
    Args:
        systems: List of ThermalSystem objects
        t_span: Time span [t_start, t_end]
        t_eval: Times at which to evaluate solution
        ambient_temp: Base ambient temperature
        air_flow_base: Base air flow rate
        
    Returns:
        Solution object from solve_ivp
    """
    y0 = np.array([sys.temperature for sys in systems], dtype=float)
    contact_area_base = 0.01
    
    solution = solve_ivp(
        fun=lambda t, y: differential_equations(t, y, systems, ambient_temp, 
                                             air_flow_base, contact_area_base),
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-8,
        atol=1e-8
    )
    
    return solution

def simulate_heat_dissipation():
    # Create three thermal systems with enhanced nonlinear properties
    system1 = ThermalSystem(
        temperature=400.15,  # 127°C - Higher initial temperature
        mass=1.2,           # 1.2 kg
        specific_heat=4186,
        thermal_conductivity=0.8,  # Higher conductivity
        surface_area=0.06
    )
    
    system2 = ThermalSystem(
        temperature=310.15,  # 37°C
        mass=2.5,           # 2.5 kg
        specific_heat=4186,
        thermal_conductivity=0.5,
        surface_area=0.08
    )
    
    system3 = ThermalSystem(
        temperature=273.15,  # 0°C - Lower initial temperature
        mass=1.8,           # 1.8 kg
        specific_heat=4186,
        thermal_conductivity=0.4,  # Lower conductivity
        surface_area=0.07
    )
    
    # Environmental parameters with enhanced variations
    ambient_temp = 295.15   # 22°C ambient
    air_flow_base = 0.8     # Increased air flow
    humidity_base = 0.7     # Higher humidity
    
    # Simulation parameters with higher resolution
    t_final = 150  # Longer simulation
    t_eval = np.linspace(0, t_final, 3000)  # More time points
    
    # Solve temperature evolution
    systems = [system1, system2, system3]
    solution = solve_temperature_evolution(systems, [0, t_final], t_eval, 
                                        ambient_temp, air_flow_base)
    
    times = solution.t
    temps1, temps2, temps3 = solution.y
    
    # Initialize energy tracking
    total_initial_energy = sum(sys.mass * sys.specific_heat * sys.temperature 
                             for sys in systems)
    
    # Calculate energy ratios and spatial distributions
    energy_ratios = []
    spatial_temp_dist = []
    entropy_changes = []
    
    # Enhanced spatial grid for temperature distribution
    x = np.linspace(-0.2, 0.2, 120)  # Larger, higher resolution grid
    y = np.linspace(-0.2, 0.2, 120)
    X, Y = np.meshgrid(x, y)
    
    # Calculate spatial distributions and thermodynamic properties
    for t_idx, t in enumerate(times):
        # Update system temperatures
        system1.temperature = temps1[t_idx]
        system2.temperature = temps2[t_idx]
        system3.temperature = temps3[t_idx]
        
        # Calculate enhanced spatial temperature distribution with interference and nonlinear effects
        temp_dist = np.zeros_like(X)
        for i, sys in enumerate([system1, system2, system3]):
            center_x = -0.08 + i*0.08
            R = np.sqrt((X-center_x)**2 + Y**2)
            # Add nonlinear temperature spreading and interference
            temp_dist += ((sys.temperature - 273.15) * 
                         np.exp(-15*R**2) * 
                         (1 + 0.3*np.sin(12*R) + 0.2*np.cos(8*R)) * 
                         (1 + 0.1*np.sin(2*np.pi*t/20)))
        spatial_temp_dist.append(temp_dist)
        
        # Calculate current total energy and entropy change
        current_energy = sum(sys.mass * sys.specific_heat * sys.temperature 
                           for sys in systems)
        energy_ratios.append(current_energy/total_initial_energy)
        
        # Calculate entropy change with nonlinear effects
        entropy_change = sum(sys.mass * sys.specific_heat * 
                           (np.log(sys.temperature/temps1[0]) + 
                            0.01*(sys.temperature - temps1[0])**2)
                           for sys in systems)
        entropy_changes.append(entropy_change)
    
    # Package enhanced history data
    history = {
        'time': times,
        'temperature': [temps1, temps2, temps3],
        'energy_conservation': np.array(energy_ratios),
        'spatial_distribution': np.array(spatial_temp_dist),
        'spatial_grid': (X, Y),
        'entropy_changes': np.array(entropy_changes)
    }
        
    return system1, system2, system3, history

def create_animation(history):
    """Create enhanced animated visualization of heat transfer"""
    # Create figure with advanced layout
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])
    
    # Main temperature plot (top span)
    ax1 = fig.add_subplot(gs[0, :])
    # Energy conservation plot (bottom left)
    ax2 = fig.add_subplot(gs[1, 0])
    # Heatmap plot (bottom middle)
    ax3 = fig.add_subplot(gs[1, 1])
    # 3D surface plot (bottom right)
    ax4 = fig.add_subplot(gs[1, 2], projection='3d')
    
    fig.suptitle('Advanced Heat Dissipation Simulation', fontsize=16, y=0.95)
    
    # Enhanced temperature plot with gradient background
    temp_lines = []
    times = history['time']
    colors = ['#FF3366', '#3366FF', '#33FF66']
    for i, (color, label) in enumerate(zip(colors,
                                ['System 1 (Initial: 127°C)',
                                 'System 2 (Initial: 37°C)',
                                 'System 3 (Initial: 0°C)'])):
        line, = ax1.plot([], [], color=color, label=label, lw=2.5)
        temp_lines.append(line)
    
    # Add system markers
    markers = []
    for color in colors:
        marker = ax1.plot([], [], color=color, marker='o', markersize=10)[0]
        markers.append(marker)
    
    # Customize temperature plot
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Temperature (°C)', fontsize=12)
    ax1.set_xlim(0, max(times))
    ax1.set_ylim(-5, 130)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, loc='upper right')
    
    # Enhanced energy conservation plot
    energy_line, = ax2.plot([], [], color='#FF6B6B', label='Energy Conservation', lw=2)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Relative Total Energy', fontsize=12)
    ax2.set_xlim(0, max(times))
    ax2.set_ylim(0.9, 1.1)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # Enhanced heatmap plot
    X, Y = history['spatial_grid']
    temp_dist = history['spatial_distribution'][0]
    custom_cmap = LinearSegmentedColormap.from_list('custom', 
        ['#000033', '#0000FF', '#00FF00', '#FFFF00', '#FF0000', '#FF00FF'])
    heatmap = ax3.pcolormesh(X, Y, temp_dist, cmap=custom_cmap,
                            norm=plt.Normalize(vmin=-5, vmax=130))
    plt.colorbar(heatmap, ax=ax3, label='Temperature (°C)')
    ax3.set_xlabel('Position (m)', fontsize=12)
    ax3.set_ylabel('Position (m)', fontsize=12)
    ax3.set_title('Temperature Distribution', fontsize=12)
    
    # Initialize 3D surface plot with enhanced features
    surf = ax4.plot_surface(X, Y, temp_dist, cmap=custom_cmap,
                           norm=plt.Normalize(vmin=-5, vmax=130),
                           rstride=2, cstride=2,
                           linewidth=0.5, antialiased=True)
    ax4.set_xlabel('X (m)', fontsize=10)
    ax4.set_ylabel('Y (m)', fontsize=10)
    ax4.set_zlabel('Temperature (°C)', fontsize=10)
    ax4.set_title('3D Temperature Profile', fontsize=12)
    ax4.view_init(elev=30, azim=45)  # Set initial viewing angle
    
    # Heat flow arrows with enhanced styling
    arrow_props = dict(arrowstyle='->', lw=2, color='#FF6B6B', alpha=0.7)
    arrow1 = ax1.annotate('', xy=(0, 0), xytext=(0, 0),
                         arrowprops=arrow_props)
    arrow2 = ax1.annotate('', xy=(0, 0), xytext=(0, 0),
                         arrowprops=arrow_props)
    
    def animate(frame):
        frame_idx = int(frame)
        
        # Update temperature lines with smooth transitions
        for i, line in enumerate(temp_lines):
            line.set_data(times[:frame_idx],
                         history['temperature'][i][:frame_idx] - 273.15)
            
            # Update markers
            if frame_idx > 0:
                markers[i].set_data([times[frame_idx-1]], 
                                  [history['temperature'][i][frame_idx-1] - 273.15])
        
        # Update energy conservation with gradient effect
        energy_line.set_data(times[:frame_idx],
                           history['energy_conservation'][:frame_idx])
        
        # Update heatmap with enhanced interpolation
        temp_dist = history['spatial_distribution'][frame_idx]
        heatmap.set_array(temp_dist.ravel())
        
        # Update 3D surface with rotation
        ax4.clear()
        surf = ax4.plot_surface(X, Y, temp_dist,
                              cmap=custom_cmap,
                              norm=plt.Normalize(vmin=-5, vmax=130),
                              rstride=2, cstride=2,
                              linewidth=0.5, antialiased=True)
        ax4.set_xlabel('X (m)', fontsize=10)
        ax4.set_ylabel('Y (m)', fontsize=10)
        ax4.set_zlabel('Temperature (°C)', fontsize=10)
        ax4.set_title('3D Temperature Profile', fontsize=12)
        # Rotate view angle dynamically
        ax4.view_init(elev=30, azim=45 + frame/10)
        ax4.set_zlim(-5, 130)
        
        # Update heat flow arrows with dynamic sizing
        if frame_idx > 0:
            t1 = history['temperature'][0][frame_idx-1] - 273.15
            t2 = history['temperature'][1][frame_idx-1] - 273.15
            t3 = history['temperature'][2][frame_idx-1] - 273.15
            
            # Dynamic arrow positions and sizes based on temperature differences
            dt12 = abs(t1 - t2)
            dt23 = abs(t2 - t3)
            
            arrow1.set_position((times[frame_idx-1], (t1 + t2)/2))
            arrow2.set_position((times[frame_idx-1], (t2 + t3)/2))
            
            # Update arrow properties directly
            arrow1.arrowprops = arrow_props.copy()
            arrow1.arrowprops['mutation_scale'] = 10 + 20 * (dt12 / 100)
            arrow2.arrowprops = arrow_props.copy() 
            arrow2.arrowprops['mutation_scale'] = 10 + 20 * (dt23 / 100)
        
        # Update x-axis limits to show moving window
        window_size = 20  # Show 20 seconds of data
        if times[frame_idx] > window_size:
            ax1.set_xlim(times[frame_idx] - window_size, times[frame_idx])
            ax2.set_xlim(times[frame_idx] - window_size, times[frame_idx])
        
        return temp_lines + markers + [energy_line, arrow1, arrow2, heatmap]
    
    anim = FuncAnimation(fig, animate, frames=len(times),
                        interval=25, blit=True)
    plt.tight_layout()
    return anim

def main():
    # Run simulation
    system1, system2, system3, history = simulate_heat_dissipation()
    
    # Create and display animation
    anim = create_animation(history)
    plt.show()

if __name__ == "__main__":
    main()