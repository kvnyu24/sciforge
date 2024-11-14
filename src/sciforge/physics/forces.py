import numpy as np
from typing import Optional, Union
from numpy.typing import ArrayLike
from .base import Force




class SpringForce(Force):
    """Hooke's law spring force"""
    def __init__(self, k: float, anchor: ArrayLike, rest_length: float = 0.0):
        self.k = k
        self.anchor = np.array(anchor)
        self.rest_length = rest_length
        
    def __call__(self, position: ArrayLike, velocity: Optional[ArrayLike] = None,
                time: Optional[float] = None) -> np.ndarray:
        displacement = np.array(position) - self.anchor
        stretch = np.linalg.norm(displacement) - self.rest_length
        if np.linalg.norm(displacement) < 1e-10:  # Avoid division by zero
            return np.zeros_like(displacement)
        return -self.k * stretch * displacement / np.linalg.norm(displacement)


class GravityForce(Force):
    """Constant gravitational force"""
    def __init__(self, g: float = 9.81):
        self.g = g
        
    def __call__(self, position: ArrayLike, velocity: Optional[ArrayLike] = None,
                time: Optional[float] = None) -> np.ndarray:
        return np.array([0, -self.g])  # 2D gravity


class DragForce(Force):
    """Quadratic drag force"""
    def __init__(self, drag_coeff: float):
        self.drag_coeff = drag_coeff
        
    def __call__(self, position: ArrayLike, velocity: ArrayLike,
                time: Optional[float] = None) -> np.ndarray:
        if velocity is None:
            return np.zeros_like(position)
        velocity = np.array(velocity)
        speed = np.linalg.norm(velocity)
        if speed < 1e-10:  # Avoid division by zero
            return np.zeros_like(velocity)
        return -0.5 * self.drag_coeff * speed * velocity


class CentralForce(Force):
    """Radial force field with 1/r^2 dependence"""
    def __init__(self, strength: float, center: ArrayLike = None):
        self.strength = strength
        self.center = np.zeros(2) if center is None else np.array(center)  # 2D default
        
    def __call__(self, position: ArrayLike, velocity: Optional[ArrayLike] = None,
                time: Optional[float] = None) -> np.ndarray:
        r = np.array(position) - self.center
        r_mag = np.linalg.norm(r)
        if r_mag < 1e-10:  # Avoid division by zero
            return np.zeros_like(r)
        return self.strength * r / r_mag**3


class FrictionForce(Force):
    """Combined static and kinetic friction"""
    def __init__(self, static_coeff: float, kinetic_coeff: float, normal_force: float):
        self.static_coeff = static_coeff
        self.kinetic_coeff = kinetic_coeff
        self.normal_force = normal_force
        
    def __call__(self, position: ArrayLike, velocity: ArrayLike,
                time: Optional[float] = None) -> np.ndarray:
        if velocity is None:
            return np.zeros_like(position)
        velocity = np.array(velocity)
        speed = np.linalg.norm(velocity)
        
        if speed < 1e-6:  # Static friction regime
            max_static_force = self.static_coeff * self.normal_force
            return -max_static_force * velocity if speed > 0 else np.zeros_like(velocity)
            
        # Kinetic friction regime
        return -self.kinetic_coeff * self.normal_force * velocity / speed
