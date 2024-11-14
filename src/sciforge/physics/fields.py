import numpy as np
from typing import Tuple, Optional, Union
from numpy.typing import ArrayLike
from .base import Field, ConservativeField


class ElectricField(ConservativeField):
    """Electric field from point charge"""
    def __init__(self, charge: float):
        super().__init__(8.99e9 * charge)  # Coulomb constant * charge
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
        super().__init__(1e-7 * current)  # Magnetic constant * current
        self.current = current
        
    def vector_potential(self, position: ArrayLike) -> np.ndarray:
        """Calculate magnetic vector potential A at position r"""
        r = np.array(position)
        return self.strength * np.array([0,0,1]) / np.linalg.norm(r)
        
    def field(self, position: ArrayLike) -> np.ndarray:
        """Calculate magnetic field B at position r"""
        r = np.array(position)
        return self.strength * np.cross(np.array([0,0,1]), r) / np.linalg.norm(r)**3


class GravitationalField(ConservativeField):
    """Gravitational field from point mass"""
    def __init__(self, mass: float):
        super().__init__(6.67e-11 * mass)  # Gravitational constant * mass
        self.mass = mass
        
    def potential(self, position: ArrayLike) -> float:
        """Calculate gravitational potential φ at position r"""
        r = np.array(position)
        return -self.strength / np.linalg.norm(r)
        
    def field(self, position: ArrayLike) -> np.ndarray:
        """Calculate gravitational field g at position r"""
        r = np.array(position)
        return -self.strength * r / np.linalg.norm(r)**3


class UniformField(ConservativeField):
    """Uniform field with constant magnitude and direction"""
    def __init__(self, field_vector: ArrayLike):
        super().__init__(np.linalg.norm(field_vector))
        self.field_vector = np.array(field_vector)
        
    def field(self, position: ArrayLike) -> np.ndarray:
        """Return constant field vector"""
        return self.field_vector.copy()

    def potential(self, position: ArrayLike) -> float:
        """Calculate potential for uniform field"""
        return -np.dot(self.field_vector, np.array(position))


class DipoleField(Field):
    """Field from electric or magnetic dipole"""
    def __init__(self, moment: ArrayLike, is_electric: bool = True):
        self.moment = np.array(moment)
        strength = 8.99e9 if is_electric else 1e-7  # Electric/magnetic constant
        super().__init__(strength * np.linalg.norm(moment))
        
    def field(self, position: ArrayLike) -> np.ndarray:
        """Calculate dipole field at position"""
        r = np.array(position)
        r_mag = np.linalg.norm(r)
        r_hat = r / r_mag
        p_dot_r = np.dot(self.moment, r_hat)
        return (self.strength / r_mag**3) * (3 * p_dot_r * r_hat - self.moment)


class QuadrupoleField(Field):
    """Field from electric or magnetic quadrupole"""
    def __init__(self, quadrupole_tensor: ArrayLike, is_electric: bool = True):
        self.Q = np.array(quadrupole_tensor)
        strength = 8.99e9 if is_electric else 1e-7
        super().__init__(strength * np.linalg.norm(self.Q))
        
    def field(self, position: ArrayLike) -> np.ndarray:
        """Calculate quadrupole field at position"""
        r = np.array(position)
        r_mag = np.linalg.norm(r)
        r_hat = r / r_mag
        Q_dot_r = self.Q @ r_hat
        return (self.strength / r_mag**4) * (5 * np.dot(Q_dot_r, r_hat) * r_hat - 2 * Q_dot_r)


class SolenodalField(MagneticField):
    """Magnetic field from a solenoid"""
    def __init__(self, current: float, radius: float, turns_per_length: float):
        super().__init__(current)
        self.radius = radius
        self.n = turns_per_length
        
    def field(self, position: ArrayLike) -> np.ndarray:
        """Calculate magnetic field inside and outside solenoid"""
        r = np.array(position)
        rho = np.sqrt(r[0]**2 + r[1]**2)  # Radial distance
        
        if rho <= self.radius:
            # Inside solenoid - uniform field
            return np.array([0, 0, self.strength * self.n])
        else:
            # Outside solenoid - dipole-like field
            return super().field(position) * (self.radius**2 / rho**2)
