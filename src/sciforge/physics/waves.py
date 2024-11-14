"""
Wave mechanics and propagation module.

Provides classes for simulating classical waves and wave packets.
"""

import numpy as np
from typing import Optional, Union, Tuple
from numpy.typing import ArrayLike


class Wave:
    """Class representing a classical wave"""
    
    def __init__(self, 
                 amplitude: float,
                 wavelength: float,
                 frequency: float,
                 phase: float = 0.0):
        """
        Initialize wave parameters
        
        Args:
            amplitude: Wave amplitude
            wavelength: Wavelength (meters)
            frequency: Frequency (Hz) 
            phase: Initial phase (radians)
        """
        self.amplitude = amplitude
        self.wavelength = wavelength
        self.frequency = frequency
        self.phase = phase
        self.angular_freq = 2 * np.pi * frequency
        self.wavenumber = 2 * np.pi / wavelength
        
    def displacement(self, x: ArrayLike, t: float) -> np.ndarray:
        """
        Calculate wave displacement at position x and time t
        
        Args:
            x: Position array
            t: Time
            
        Returns:
            Wave displacement array
        """
        return self.amplitude * np.sin(self.wavenumber * x - self.angular_freq * t + self.phase)
    
    def velocity(self, x: ArrayLike, t: float) -> np.ndarray:
        """
        Calculate wave velocity at position x and time t
        
        Args:
            x: Position array
            t: Time
            
        Returns:
            Wave velocity array
        """
        return -self.amplitude * self.angular_freq * np.cos(self.wavenumber * x - self.angular_freq * t + self.phase)
    
    def energy(self) -> float:
        """Calculate wave energy density"""
        return 0.5 * self.amplitude**2 * self.angular_freq**2


class WavePacket:
    """Class representing a wave packet (group of waves)"""
    
    def __init__(self,
                 central_wavelength: float,
                 spread: float,
                 position: float = 0.0):
        """
        Initialize wave packet
        
        Args:
            central_wavelength: Central wavelength of packet
            spread: Spatial spread of packet
            position: Initial central position
        """
        self.central_wavelength = central_wavelength
        self.spread = spread
        self.position = position
        self.k0 = 2 * np.pi / central_wavelength
        
    def wavefunction(self, x: ArrayLike, t: float) -> np.ndarray:
        """
        Calculate wave packet amplitude at position x and time t
        
        Args:
            x: Position array
            t: Time
            
        Returns:
            Complex wave packet amplitude
        """
        # Group velocity
        v_g = self.k0 / (2 * np.pi)
        
        # Gaussian envelope
        envelope = np.exp(-(x - self.position - v_g * t)**2 / (4 * self.spread**2))
        
        # Carrier wave
        carrier = np.exp(1j * (self.k0 * x - self.k0 * v_g * t))
        
        return envelope * carrier
