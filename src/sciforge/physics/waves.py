"""
Wave mechanics and propagation module.

Provides classes for simulating classical waves and wave packets.
"""

import numpy as np
from typing import Optional, Union, Tuple
from numpy.typing import ArrayLike
from .base import PhysicalSystem


class Wave(PhysicalSystem):
    """Class representing a classical wave"""
    
    def __init__(self, 
                 amplitude: float,
                 wavelength: float,
                 frequency: float,
                 phase: float = 0.0,
                 position: ArrayLike = np.array([0.0]),
                 mass: float = 1.0):
        """
        Initialize wave parameters
        
        Args:
            amplitude: Wave amplitude
            wavelength: Wavelength (meters)
            frequency: Frequency (Hz) 
            phase: Initial phase (radians)
            position: Initial position array
            mass: Wave mass/energy density
        """
        super().__init__(mass, position)
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
        return 0.5 * self.mass * self.amplitude**2 * self.angular_freq**2


class WavePacket(PhysicalSystem):
    """Class representing a wave packet (group of waves)"""
    
    def __init__(self,
                 central_wavelength: float,
                 spread: float,
                 position: float = 0.0,
                 mass: float = 1.0):
        """
        Initialize wave packet
        
        Args:
            central_wavelength: Central wavelength of packet
            spread: Spatial spread of packet
            position: Initial central position
            mass: Wave packet mass/energy density
        """
        super().__init__(mass, np.array([position]))
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


class StandingWave(Wave):
    """Class representing a standing wave"""
    
    def __init__(self,
                 amplitude: float,
                 wavelength: float,
                 frequency: float,
                 length: float,
                 mode: int = 1,
                 phase: float = 0.0,
                 position: ArrayLike = np.array([0.0]),
                 mass: float = 1.0):
        """
        Initialize standing wave
        
        Args:
            amplitude: Wave amplitude
            wavelength: Wavelength (meters)
            frequency: Frequency (Hz)
            length: Length of medium
            mode: Mode number (1, 2, 3, ...)
            phase: Initial phase (radians)
            position: Initial position array
            mass: Wave mass/energy density
        """
        super().__init__(amplitude, wavelength, frequency, phase, position, mass)
        self.length = length
        self.mode = mode
        
    def displacement(self, x: ArrayLike, t: float) -> np.ndarray:
        """Calculate standing wave displacement"""
        return 2 * self.amplitude * np.sin(self.mode * np.pi * x / self.length) * np.cos(self.angular_freq * t)


class DampedWave(Wave):
    """Class representing a damped wave"""
    
    def __init__(self,
                 amplitude: float,
                 wavelength: float,
                 frequency: float,
                 damping: float,
                 phase: float = 0.0,
                 position: ArrayLike = np.array([0.0]),
                 mass: float = 1.0):
        """
        Initialize damped wave
        
        Args:
            amplitude: Initial amplitude
            wavelength: Wavelength (meters)
            frequency: Frequency (Hz)
            damping: Damping coefficient
            phase: Initial phase (radians)
            position: Initial position array
            mass: Wave mass/energy density
        """
        super().__init__(amplitude, wavelength, frequency, phase, position, mass)
        self.damping = damping
        
    def displacement(self, x: ArrayLike, t: float) -> np.ndarray:
        """Calculate damped wave displacement"""
        return self.amplitude * np.exp(-self.damping * t) * np.sin(self.wavenumber * x - self.angular_freq * t + self.phase)


class ShockWave(Wave):
    """Class representing a shock wave"""
    
    def __init__(self,
                 amplitude: float,
                 wavelength: float,
                 frequency: float,
                 shock_speed: float,
                 phase: float = 0.0,
                 position: ArrayLike = np.array([0.0]),
                 mass: float = 1.0):
        """
        Initialize shock wave
        
        Args:
            amplitude: Wave amplitude
            wavelength: Wavelength (meters)
            frequency: Frequency (Hz)
            shock_speed: Propagation speed of shock front
            phase: Initial phase (radians)
            position: Initial position array
            mass: Wave mass/energy density
        """
        super().__init__(amplitude, wavelength, frequency, phase, position, mass)
        self.shock_speed = shock_speed
        
    def displacement(self, x: ArrayLike, t: float) -> np.ndarray:
        """Calculate shock wave displacement with discontinuous front"""
        shock_front = self.shock_speed * t
        wave = np.where(x <= shock_front,
                       self.amplitude * np.sin(self.wavenumber * x - self.angular_freq * t + self.phase),
                       0)
        return wave
