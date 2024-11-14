from typing import Optional, Tuple, List
import numpy as np
from numpy.typing import ArrayLike
from .base import PhysicalSystem
from .fields import ElectricField, MagneticField
from ..core.constants import CONSTANTS

class EMWave(PhysicalSystem):
    """Electromagnetic wave propagation in various media with advanced features"""
    def __init__(self,
                 wavelength: float,
                 amplitude: float,
                 polarization: ArrayLike = np.array([1, 0, 0]),
                 position: ArrayLike = np.array([0, 0, 0]),
                 direction: ArrayLike = np.array([0, 0, 1]),
                 medium_permittivity: float = CONSTANTS['eps0'],
                 medium_permeability: float = CONSTANTS['mu0'],
                 attenuation: float = 0.0,
                 dispersion: float = 0.0):
        
        super().__init__(0, position)  # Massless wave
        self.wavelength = wavelength
        self.amplitude = amplitude
        self.polarization = np.array(polarization) / np.linalg.norm(polarization)
        self.direction = np.array(direction) / np.linalg.norm(direction)
        self.eps = medium_permittivity
        self.mu = medium_permeability
        self.attenuation = attenuation
        self.dispersion = dispersion
        
        # Derived quantities
        self.n = np.sqrt(self.eps * self.mu / (CONSTANTS['eps0'] * CONSTANTS['mu0']))
        self.v = CONSTANTS['c'] / self.n
        self.frequency = self.v / wavelength
        self.omega = 2 * np.pi * self.frequency
        self.k = 2 * np.pi / wavelength
        
        # Complex wavevector including attenuation
        self.k_complex = self.k + 1j * self.attenuation
        
        # Initialize fields with proper scaling
        self.E_field = ElectricField(self.amplitude)
        self.B_field = MagneticField(self.amplitude * self.n / CONSTANTS['c'])
        
        # Track wave history for interference calculations
        self.path_history = [np.copy(position)]
        self.time_history = [0.0]
        
    def propagate(self, dt: float):
        """Propagate wave forward in time with dispersion effects"""
        # Update position with group velocity
        group_velocity = self.v * (1 - self.dispersion * self.wavelength)
        new_position = self.position + self.direction * group_velocity * dt
        
        # Store history
        self.path_history.append(np.copy(new_position))
        self.time_history.append(self.time_history[-1] + dt)
        
        # Update position
        self.position = new_position
        
    def get_fields(self, position: ArrayLike, time: float) -> Tuple[np.ndarray, np.ndarray]:
        """Get E and B fields at given position and time including attenuation"""
        r = np.array(position) - self.position
        r_mag = np.linalg.norm(r)
        
        # Complex phase with attenuation
        phase = self.k_complex * np.dot(r, self.direction) - self.omega * time
        
        # Include 1/r amplitude decay for spherical waves when far from source
        amplitude_factor = self.amplitude * np.exp(-self.attenuation * r_mag)
        if r_mag > self.wavelength:
            amplitude_factor *= self.wavelength / r_mag
            
        # Electric field with polarization
        E = amplitude_factor * self.polarization * np.exp(1j * phase).real
        
        # Magnetic field perpendicular to E and k
        B = (amplitude_factor * self.n / CONSTANTS['c']) * np.cross(self.direction, self.polarization) * np.exp(1j * phase).real
        
        return E, B
        
    def intensity(self, position: ArrayLike, time: float) -> float:
        """Calculate wave intensity with Poynting vector"""
        E, B = self.get_fields(position, time)
        S = np.cross(E, B) / self.mu  # Poynting vector
        return np.linalg.norm(S)
    
    def interference(self, other: 'EMWave', position: ArrayLike, time: float) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate interference between two waves"""
        E1, B1 = self.get_fields(position, time)
        E2, B2 = other.get_fields(position, time)
        return E1 + E2, B1 + B2
    
    def get_phase_difference(self, position1: ArrayLike, position2: ArrayLike) -> float:
        """Calculate phase difference between two points"""
        r1 = np.array(position1) - self.position
        r2 = np.array(position2) - self.position
        return self.k * (np.dot(r1, self.direction) - np.dot(r2, self.direction))
        
    def get_wavefront(self, time: float, points: int = 100) -> np.ndarray:
        """Calculate points on the wavefront surface"""
        # Create a plane perpendicular to propagation direction
        u = np.array([1, 0, 0]) if not np.allclose(self.direction, [1, 0, 0]) else np.array([0, 1, 0])
        v = np.cross(self.direction, u)
        u = np.cross(v, self.direction)
        
        # Generate points on wavefront
        theta = np.linspace(0, 2*np.pi, points)
        wavefront = self.position + self.v * time * self.direction
        return np.array([wavefront + self.wavelength * (u*np.cos(t) + v*np.sin(t)) for t in theta])