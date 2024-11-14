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
    
class Force(BaseClass):
    """Base class for forces in mechanical systems"""
    def __call__(self, position: ArrayLike, velocity: Optional[ArrayLike] = None, 
                time: Optional[float] = None) -> np.ndarray:
        """Calculate force at given position, velocity and time"""
        raise NotImplementedError

class Field(BaseClass):
    """Base class for all physical fields"""
    def __init__(self, strength: float):
        super().__init__()
        self.strength = strength

    def field(self, position: ArrayLike) -> np.ndarray:
        """Calculate field vector at position"""
        raise NotImplementedError

class ConservativeField(Field):
    """Base class for conservative force fields"""
    def potential(self, position: ArrayLike) -> float:
        """Calculate potential energy at position"""
        raise NotImplementedError
        
    def force(self, position: ArrayLike) -> np.ndarray:
        """Calculate force at position (negative gradient of potential)"""
        r = np.array(position)
        return -self.field(r)

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


class GravitationalSystem(BaseClass):
    """Universal gravitation between multiple bodies"""
    def __init__(self, G: float = 6.67430e-11):
        super().__init__()
        self.G = G
        self.bodies = []
        
    def add_body(self, mass: float, position: np.ndarray, velocity: np.ndarray):
        """Add a body to the gravitational system"""
        self.bodies.append({'mass': mass, 'position': position, 'velocity': velocity})
        
    def force_on_body(self, body_idx: int) -> np.ndarray:
        """Calculate net gravitational force on a specific body"""
        total_force = np.zeros(3)
        for i, other in enumerate(self.bodies):
            if i == body_idx:
                continue
            r = other['position'] - self.bodies[body_idx]['position']
            r_mag = np.linalg.norm(r)
            force_mag = self.G * self.bodies[body_idx]['mass'] * other['mass'] / (r_mag**2)
            total_force += force_mag * r / r_mag
        return total_force