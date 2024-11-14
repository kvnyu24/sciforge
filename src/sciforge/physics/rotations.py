import numpy as np
from typing import Optional, Tuple, List
from numpy.typing import ArrayLike
from ..core.constants import CONSTANTS
from .mechanics import DynamicalSystem

class RotationalSystem(DynamicalSystem):
    """System with rotational dynamics including angular momentum and torque"""
    def __init__(self, moment_of_inertia: float, angular_position: ArrayLike = None, 
                 angular_velocity: ArrayLike = None):
        super().__init__(moment_of_inertia, 
                        np.zeros(3) if angular_position is None else np.array(angular_position),
                        np.zeros(3) if angular_velocity is None else np.array(angular_velocity))
        self.I = moment_of_inertia
        self.angular_momentum = self.I * self.velocity

    def update(self, torque: ArrayLike, dt: float) -> None:
        """Update angular state using RK4 integration"""
        # RK4 integration for angular motion
        k1v = np.array(torque) / self.I
        k1x = self.velocity
        
        v_temp = self.velocity + 0.5 * dt * k1v
        k2v = np.array(torque) / self.I  
        k2x = v_temp
        
        v_temp = self.velocity + 0.5 * dt * k2v
        k3v = np.array(torque) / self.I
        k3x = v_temp
        
        v_temp = self.velocity + dt * k3v
        k4v = np.array(torque) / self.I
        k4x = v_temp
        
        # Update state
        self.velocity += (dt/6.0) * (k1v + 2*k2v + 2*k3v + k4v)
        self.position += (dt/6.0) * (k1x + 2*k2x + 2*k3x + k4x)
        self.angular_momentum = self.I * self.velocity

    @property
    def rotational_energy(self) -> float:
        """Calculate rotational kinetic energy"""
        return 0.5 * self.I * np.dot(self.velocity, self.velocity)

def calculate_center_of_mass(positions: List[ArrayLike], masses: List[float]) -> np.ndarray:
    """Calculate center of mass for system of particles"""
    total_mass = sum(masses)
    weighted_positions = sum(m * np.array(p) for m, p in zip(masses, positions))
    return weighted_positions / total_mass

def calculate_moment_of_inertia(shape: str, mass: float, dimensions: ArrayLike) -> float:
    """Calculate moment of inertia for common shapes
    
    Supported shapes:
    - solid_sphere: dimensions[0] = radius
    - hollow_sphere: dimensions[0] = radius
    - solid_cylinder: dimensions = [radius, height]
    - disk: dimensions[0] = radius
    - rod: dimensions[0] = length
    """
    if shape == "solid_sphere":
        return 0.4 * mass * dimensions[0]**2
    elif shape == "hollow_sphere":
        return (2/3) * mass * dimensions[0]**2
    elif shape == "solid_cylinder":
        return 0.5 * mass * dimensions[0]**2
    elif shape == "disk":
        return 0.5 * mass * dimensions[0]**2
    elif shape == "rod":
        return (1/12) * mass * dimensions[0]**2
    else:
        raise ValueError(f"Unknown shape: {shape}")

class Gyroscope(RotationalSystem):
    """Gyroscope with precession and nutation"""
    def __init__(self, moment_of_inertia: float, spin_rate: float, 
                 precession_angle: float = np.pi/6):
        super().__init__(moment_of_inertia)
        self.spin_rate = spin_rate
        self.precession_angle = precession_angle
        
    def calculate_precession_rate(self, torque_magnitude: float) -> float:
        """Calculate precession rate from applied torque"""
        return torque_magnitude / (self.I * self.spin_rate)
    
    def update_precession(self, external_torque: ArrayLike, dt: float) -> None:
        """Update gyroscope state including precession effects"""
        # Calculate precession rate
        torque_mag = np.linalg.norm(external_torque)
        precession_rate = self.calculate_precession_rate(torque_mag)
        
        # Update angular momentum including precession
        precession_axis = np.array([0, 0, 1])
        spin_axis = np.array([np.sin(self.precession_angle), 0, np.cos(self.precession_angle)])
        
        # Rotate spin axis around precession axis using Rodriguez rotation formula
        c = np.cos(precession_rate * dt)
        s = np.sin(precession_rate * dt)
        k = precession_axis
        K = np.array([[0, -k[2], k[1]], 
                     [k[2], 0, -k[0]], 
                     [-k[1], k[0], 0]])
        rotation_matrix = np.eye(3) + s * K + (1 - c) * (K @ K)
        
        new_spin_axis = rotation_matrix @ spin_axis
        self.angular_momentum = self.I * self.spin_rate * new_spin_axis
        self.velocity = self.angular_momentum / self.I
