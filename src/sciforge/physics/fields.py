import numpy as np
from typing import Tuple, Optional, Union
from numpy.typing import ArrayLike

class Field:
    """Base class for physical fields"""
    def __init__(self, strength: float):
        self.strength = strength
        
    def potential(self, position: ArrayLike) -> float:
        """Calculate potential at given position"""
        raise NotImplementedError
        
    def field(self, position: ArrayLike) -> np.ndarray:
        """Calculate field vector at given position"""
        raise NotImplementedError

class ElectricField(Field):
    """Electric field from point charge"""
    def __init__(self, charge: float):
        super().__init__(8.99e9 * charge) # Coulomb constant * charge
        self.charge = charge
        
    def potential(self, position: ArrayLike) -> float:
        """Calculate electric potential V at position r"""
        r = np.array(position)
        return self.strength / np.linalg.norm(r)
        
    def field(self, position: ArrayLike) -> np.ndarray:
        """Calculate electric field E at position r"""
        r = np.array(position)
        return self.strength * r / np.linalg.norm(r)**3

class MagneticField(Field):
    """Magnetic field from current element"""
    def __init__(self, current: float):
        super().__init__(1e-7 * current) # Magnetic constant * current
        self.current = current
        
    def potential(self, position: ArrayLike) -> float:
        """Calculate magnetic vector potential A magnitude at position r"""
        r = np.array(position)
        return self.strength / np.linalg.norm(r)
        
    def field(self, position: ArrayLike) -> np.ndarray:
        """Calculate magnetic field B at position r"""
        r = np.array(position)
        return self.strength * np.cross(np.array([0,0,1]), r) / np.linalg.norm(r)**3

class GravitationalField(Field):
    """Gravitational field from point mass"""
    def __init__(self, mass: float):
        super().__init__(6.67e-11 * mass) # Gravitational constant * mass
        self.mass = mass
        
    def potential(self, position: ArrayLike) -> float:
        """Calculate gravitational potential φ at position r"""
        r = np.array(position)
        return -self.strength / np.linalg.norm(r)
        
    def field(self, position: ArrayLike) -> np.ndarray:
        """Calculate gravitational field g at position r"""
        r = np.array(position)
        return -self.strength * r / np.linalg.norm(r)**3
