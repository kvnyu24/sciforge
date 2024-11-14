"""
Fluid dynamics module implementing various fluid simulations
"""

import numpy as np
from typing import Optional, Tuple, List
from .base import PhysicalSystem
from ..numerical.integration import trapezoid
from ..core.constants import CONSTANTS

class FluidColumn(PhysicalSystem):
    """Class representing a fluid column for Plateau-Rayleigh instability simulation"""
    
    def __init__(self,
                 radius: float,
                 length: float,
                 density: float,
                 surface_tension: float,
                 viscosity: float,
                 n_points: int = 100):
        """
        Initialize fluid column
        
        Args:
            radius: Initial radius of fluid column (m)
            length: Length of fluid column (m)
            density: Fluid density (kg/m³)
            surface_tension: Surface tension coefficient (N/m)
            viscosity: Dynamic viscosity (Pa·s)
            n_points: Number of discretization points
        """
        super().__init__(mass=density * np.pi * radius**2 * length,
                        position=np.zeros(3))
        
        self.radius = radius
        self.length = length
        self.density = density
        self.surface_tension = surface_tension
        self.viscosity = viscosity
        
        # Discretize the column
        self.z = np.linspace(0, length, n_points)
        self.r = np.ones_like(self.z) * radius
        self.v_r = np.zeros_like(self.z)  # Radial velocity
        
        # Store history
        self.history = {
            'time': [0.0],
            'radius': [self.r.copy()],
            'velocity': [self.v_r.copy()]
        }
        
    def calculate_pressure(self) -> np.ndarray:
        """Calculate pressure due to surface tension"""
        # Curvature terms
        d2r_dz2 = np.gradient(np.gradient(self.r, self.z), self.z)
        axial_term = d2r_dz2 / (1 + np.gradient(self.r, self.z)**2)**(3/2)
        radial_term = 1 / self.r
        
        # Laplace pressure
        return self.surface_tension * (radial_term - axial_term)
    
    def calculate_growth_rate(self) -> float:
        """Calculate theoretical growth rate of fastest growing mode"""
        k = 0.697 / self.radius  # Wavenumber of fastest growing mode
        omega = np.sqrt(self.surface_tension / (self.density * self.radius**3))
        return omega
    
    def update(self, dt: float):
        """Update fluid column state"""
        # Calculate pressure
        pressure = self.calculate_pressure()
        
        # Update radial velocity using Navier-Stokes
        d2v_dz2 = np.gradient(np.gradient(self.v_r, self.z), self.z)
        acceleration = (-np.gradient(pressure, self.z) / self.density +
                       self.viscosity * d2v_dz2 / self.density)
        
        self.v_r += acceleration * dt
        
        # Update radius
        self.r += self.v_r * dt
        
        # Enforce volume conservation (approximately)
        volume = trapezoid(lambda z: np.pi * self.r**2, 0, self.length, len(self.z))
        scale = np.sqrt(self.length * np.pi * self.radius**2 / volume)
        self.r *= scale
        
        # Update history
        self.history['time'].append(self.history['time'][-1] + dt)
        self.history['radius'].append(self.r.copy())
        self.history['velocity'].append(self.v_r.copy()) 

class FluidJet(PhysicalSystem):
    """Class representing a fluid jet for Coandă effect simulation"""
    
    def __init__(self,
                 velocity: float,
                 density: float,
                 viscosity: float,
                 width: float,
                 n_points: int = 100):
        """
        Initialize fluid jet
        
        Args:
            velocity: Initial jet velocity (m/s)
            density: Fluid density (kg/m³)
            viscosity: Dynamic viscosity (Pa·s)
            width: Initial jet width (m)
            n_points: Number of discretization points
        """
        super().__init__(mass=density * width**2, position=np.zeros(3))
        
        self.velocity = velocity
        self.density = density
        self.viscosity = viscosity
        self.width = width
        self.n_points = n_points
        
        # Initialize streamlines
        self.streamlines = self._initialize_streamlines()
        
        # Store history
        self.history = {
            'time': [0.0],
            'streamlines': [self.streamlines.copy()]
        }
    
    def _initialize_streamlines(self):
        """Initialize streamline starting positions"""
        y_start = np.linspace(-self.width/2, self.width/2, self.n_points)
        streamlines = np.zeros((self.n_points, 2, 1))
        streamlines[:, 1, 0] = y_start
        return streamlines
    
    def update(self, dt: float, surface_x: np.ndarray, surface_y: np.ndarray):
        """Update fluid jet state considering Coandă effect"""
        # Calculate pressure gradient due to curved surface
        for i in range(self.n_points):
            # Find closest point on surface
            dx = surface_x - self.streamlines[i, 0, -1]
            dy = surface_y - self.streamlines[i, 1, -1]
            dist = np.sqrt(dx**2 + dy**2)
            closest_idx = np.argmin(dist)
            
            # Calculate surface curvature
            if closest_idx > 0 and closest_idx < len(surface_x) - 1:
                dx = surface_x[closest_idx+1] - surface_x[closest_idx-1]
                dy = surface_y[closest_idx+1] - surface_y[closest_idx-1]
                curvature = np.abs(dy/dx) / (1 + (dy/dx)**2)**1.5
            else:
                curvature = 0
            
            # Update velocity considering Coandă effect
            v_parallel = self.velocity * np.exp(-dist[closest_idx]/self.width)
            v_normal = -curvature * v_parallel**2 * dt

            # Add new point to streamline
            new_point = np.array([
                self.streamlines[i, 0, -1] + v_parallel * dt,
                self.streamlines[i, 1, -1] + v_normal * dt
            ]).reshape(2, 1)

            self.streamlines[i] = new_point


        # Update history
        self.history['time'].append(self.history['time'][-1] + dt)
        self.history['streamlines'].append(self.streamlines.copy())
    
    def get_streamlines(self):
        """Return current streamlines"""
        return self.streamlines
    
    def get_velocity_field(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate velocity magnitude at given points"""
        vel_mag = np.zeros_like(x)
        
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                # Find closest points on streamlines
                min_dist = float('inf')
                for streamline in self.streamlines:
                    dx = streamline[0, -1] - x[i, j]
                    dy = streamline[1, -1] - y[i, j]
                    dist = np.sqrt(dx**2 + dy**2)
                    min_dist = min(min_dist, dist)
                
                # Calculate velocity magnitude with distance decay
                vel_mag[i, j] = self.velocity * np.exp(-min_dist/self.width)
        
        return vel_mag