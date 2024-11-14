"""
Module for attosecond optics and strong-field physics simulations.
Builds on existing quantum and optics frameworks.
"""

import numpy as np
from typing import Optional, Tuple
from numpy.typing import ArrayLike
from ..core.constants import CONSTANTS
from .quantum import QuantumSystem, Wavefunction
from .waves import WavePacket

class StrongFieldSystem(Wavefunction):
    """Base class for strong-field laser-matter interactions"""
    
    def __init__(self,
                 x: ArrayLike,
                 mass: float = CONSTANTS['me'],    # Default to electron mass
                 hbar: float = CONSTANTS['hbar']):  # Default to real hbar
        # Initialize with empty wavefunction array matching position grid size
        super().__init__(np.zeros_like(x), x, mass, hbar)
        
        # Initialize history
        self.history = {
            'time': [0],
            'wavefunction': [],
            'dipole_moment': [],
            'ionization_rate': []
        }
    
    def dipole_moment(self) -> float:
        """Calculate electric dipole moment"""
        return -CONSTANTS['e'] * self.expectation_value(self.position)
    
    def ionization_probability(self, field_strength: float) -> float:
        """Calculate tunneling ionization probability"""
        rate = self.tunnel_ionization_rate(field_strength)
        dt = self.history['time'][-1] - self.history['time'][-2] if len(self.history['time']) > 1 else 0
        return 1 - np.exp(-rate * dt)

class AttosecondPulseGenerator(StrongFieldSystem):
    """Class for simulating high harmonic generation and attosecond pulse production"""
    
    def __init__(self,
                 x: ArrayLike,
                 wavelength: float,     # Driving laser wavelength (m)
                 intensity: float,      # Peak intensity (W/m²)
                 pulse_duration: float, # FWHM duration (s)
                 ip: float = 13.6,      # Ionization potential (eV)
                 wavefunction: Optional[ArrayLike] = None):  # Initial wavefunction
        
        super().__init__(x)
        
        self.wavelength = wavelength
        self.intensity = intensity
        self.pulse_duration = pulse_duration
        self.ip = ip * CONSTANTS['e']  # Convert to Joules
        
        # Set initial wavefunction if provided
        if wavefunction is not None:
            self.psi = wavefunction
        
        # Derived parameters
        self.omega = 2 * CONSTANTS['pi'] * CONSTANTS['c'] / wavelength
        self.E0 = np.sqrt(2 * intensity / (CONSTANTS['c'] * CONSTANTS['eps0']))
        self.Up = CONSTANTS['e']**2 * self.E0**2 / (4 * CONSTANTS['me'] * self.omega**2)
        
        # Initialize driving laser as WavePacket
        self.driving_laser = WavePacket(
            central_wavelength=wavelength,
            spread=CONSTANTS['c'] * pulse_duration,
            position=0.0,
            mass=1.0  # Photon-like mass
        )
        
    def tunnel_ionization_rate(self, E: float) -> float:
        """Calculate ADK tunneling ionization rate"""
        kappa = np.sqrt(2 * self.ip) / self.hbar
        E_a = 5.14e11  # Atomic unit of field strength
        
        rate = (4 * kappa**3 * E_a / abs(E)) * \
               np.exp(-2 * kappa**3 * E_a / (3 * abs(E)))
        return rate
    
    def generate_attosecond_pulse(self, 
                                 t: ArrayLike,
                                 harmonic_range: Tuple[int, int] = (11, 31)
                                ) -> Tuple[ArrayLike, ArrayLike]:
        """
        Generate attosecond pulse through HHG process
        
        Args:
            t: Time array
            harmonic_range: (min, max) harmonic orders to include
            
        Returns:
            Tuple of (time array, attosecond pulse field)
        """
        # Get driving laser field
        # Reshape position to column vector and time to row vector for proper broadcasting
        x_reshaped = self.position[:, np.newaxis]  # Shape: (200, 1)
        t_reshaped = t[np.newaxis, :]             # Shape: (1, 1000)
        E_t = self.driving_laser.wavefunction(x_reshaped, t_reshaped)
        E_t = self.E0 * np.real(E_t[0, :])  # Take first spatial point for time evolution
        
        # Calculate dipole acceleration
        dipole_accel = np.zeros_like(t, dtype=complex)
        
        # Three-step model implementation
        for i in range(len(t)):
            # 1. Tunnel ionization
            ion_prob = self.ionization_probability(E_t[i])
            
            # 2. Classical electron acceleration
            if i > 0:
                dt = t[i] - t[i-1]
                v = -CONSTANTS['e'] * E_t[i] * dt / CONSTANTS['me']
                x = v * dt
                
                # 3. Recombination & emission
                if x < 0:  # electron returns to core
                    # Calculate recombination energy
                    E_kin = 0.5 * CONSTANTS['me'] * v**2
                    photon_energy = E_kin + self.ip
                    
                    # Add contribution to dipole acceleration
                    harmonic_order = int(photon_energy / (self.hbar * self.omega))
                    if harmonic_range[0] <= harmonic_order <= harmonic_range[1]:
                        dipole_accel[i] = -CONSTANTS['e'] * E_t[i] / CONSTANTS['me']
            
            # Update history
            self.history['time'].append(t[i])
            self.history['dipole_moment'].append(self.dipole_moment())
            self.history['ionization_rate'].append(ion_prob)
        
        # Generate attosecond pulse through Fourier filtering
        freq = np.fft.fftfreq(len(t), t[1]-t[0])
        dipole_spectrum = np.fft.fft(dipole_accel)
        
        # Filter harmonics
        mask = np.zeros_like(freq, dtype=bool)
        for n in range(harmonic_range[0], harmonic_range[1]+1, 2):
            mask |= (np.abs(freq) > (n-0.5)*self.omega/(2*CONSTANTS['pi'])) & \
                   (np.abs(freq) < (n+0.5)*self.omega/(2*CONSTANTS['pi']))
        
        filtered_spectrum = dipole_spectrum * mask
        attosecond_pulse = np.fft.ifft(filtered_spectrum)
        
        return t, np.real(attosecond_pulse)
