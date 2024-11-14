"""
Base components for physics simulations
"""

import numpy as np
from typing import Optional, Union, Tuple, List
from numpy.typing import ArrayLike
from ..core.base import BaseClass

class PhysicalSystem(BaseClass):
    """Base class for all physical systems"""
    def __init__(self, 
                 mass: float,
                 position: ArrayLike,
                 time: float = 0.0):
        super().__init__()
        self.mass = mass
        self.position = np.array(position)
        self.time = time
        self._energy = 0.0
        
    def energy(self) -> float:
        """Get total energy of system"""
        return self._energy
        
    def update(self, dt: float):
        """Update system state"""
        raise NotImplementedError

class DynamicalSystem(PhysicalSystem):
    """Base class for systems with dynamics"""
    def __init__(self,
                 mass: float, 
                 position: ArrayLike,
                 velocity: ArrayLike,
                 forces: Optional[List[callable]] = None):
        super().__init__(mass, position)
        self.velocity = np.array(velocity)
        self.forces = forces or []
        
    def add_force(self, force: callable):
        """Add force function to system"""
        self.forces.append(force)
        
    def total_force(self) -> np.ndarray:
        """Calculate total force on system"""
        return sum(f(self.position, self.velocity, self.time) 
                  for f in self.forces)

class ConservativeField(BaseClass):
    """Base class for conservative force fields"""
    def __init__(self, strength: float):
        super().__init__()
        self.strength = strength
        
    def potential(self, position: ArrayLike) -> float:
        """Calculate potential energy at position"""
        raise NotImplementedError
        
    def force(self, position: ArrayLike) -> np.ndarray:
        """Calculate force at position"""
        raise NotImplementedError

class QuantumSystem(PhysicalSystem):
    """Base class for quantum mechanical systems"""
    def __init__(self,
                 mass: float,
                 position: ArrayLike,
                 wavefunction: Union[callable, ArrayLike],
                 hbar: float = 1.0):
        super().__init__(mass, position)
        self.hbar = hbar
        self._init_wavefunction(wavefunction)
        
    def _init_wavefunction(self, psi):
        """Initialize and normalize wavefunction"""
        if callable(psi):
            self.psi = psi(self.position)
        else:
            self.psi = np.array(psi)
        self._normalize()
        
    def _normalize(self):
        """Normalize wavefunction"""
        dx = np.diff(self.position)[0]
        norm = np.sqrt(np.sum(np.abs(self.psi)**2) * dx)
        self.psi /= norm

class ThermodynamicSystem(PhysicalSystem):
    """Base class for thermodynamic systems"""
    def __init__(self,
                 mass: float,
                 position: ArrayLike,
                 temperature: float,
                 specific_heat: float):
        super().__init__(mass, position)
        self.temperature = temperature
        self.specific_heat = specific_heat
        self._energy = self.mass * specific_heat * temperature
        
    def heat_transfer(self, other: 'ThermodynamicSystem', 
                     coupling: float, dt: float):
        """Transfer heat between systems"""
        dQ = coupling * (other.temperature - self.temperature) * dt
        self.add_heat(dQ)
        other.add_heat(-dQ)
        
    def add_heat(self, dQ: float):
        """Add heat to system"""
        self.temperature += dQ / (self.mass * self.specific_heat)
        self._energy += dQ 