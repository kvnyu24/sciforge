import numpy as np
from typing import Union, Tuple
from numpy.typing import ArrayLike

class LorentzTransform:
    """Class for special relativity calculations using Lorentz transformations"""
    
    def __init__(self, velocity: Union[float, ArrayLike]):
        """
        Initialize Lorentz transformation
        
        Args:
            velocity: Relative velocity between reference frames (as fraction of c)
        """
        self.c = 299792458  # Speed of light in m/s
        self.velocity = np.array(velocity)
        self.beta = self.velocity / self.c
        self.gamma = 1 / np.sqrt(1 - np.sum(self.beta**2))
        
    def transform_time(self, t: float, x: ArrayLike) -> float:
        """
        Transform time between reference frames
        
        Args:
            t: Time in original frame
            x: Position vector in original frame
            
        Returns:
            Time in new frame
        """
        x = np.array(x)
        return self.gamma * (t - np.dot(self.beta, x) / self.c)
        
    def transform_position(self, x: ArrayLike, t: float) -> np.ndarray:
        """
        Transform position between reference frames
        
        Args:
            x: Position vector in original frame
            t: Time in original frame
            
        Returns:
            Position vector in new frame
        """
        x = np.array(x)
        return x + (self.gamma - 1) * np.dot(self.beta, x) * self.beta / np.sum(self.beta**2) \
               - self.gamma * self.velocity * t
               
    def proper_time(self, t: float, v: ArrayLike) -> float:
        """
        Calculate proper time for moving object
        
        Args:
            t: Coordinate time
            v: Velocity vector of object
            
        Returns:
            Proper time
        """
        v = np.array(v)
        beta = np.linalg.norm(v) / self.c
        gamma = 1 / np.sqrt(1 - beta**2)
        return t / gamma
        
    def length_contraction(self, length: float) -> float:
        """
        Calculate contracted length along direction of motion
        
        Args:
            length: Proper length in rest frame
            
        Returns:
            Contracted length in moving frame
        """
        return length / self.gamma
        
    def time_dilation(self, time: float) -> float:
        """
        Calculate dilated time
        
        Args:
            time: Proper time in rest frame
            
        Returns:
            Dilated time in moving frame
        """
        return self.gamma * time
        
    def relativistic_mass(self, rest_mass: float) -> float:
        """
        Calculate relativistic mass
        
        Args:
            rest_mass: Mass in rest frame
            
        Returns:
            Relativistic mass
        """
        return self.gamma * rest_mass
        
    def relativistic_momentum(self, mass: float, velocity: ArrayLike) -> np.ndarray:
        """
        Calculate relativistic momentum
        
        Args:
            mass: Rest mass
            velocity: Velocity vector
            
        Returns:
            Relativistic momentum vector
        """
        velocity = np.array(velocity)
        beta = np.linalg.norm(velocity) / self.c
        gamma = 1 / np.sqrt(1 - beta**2)
        return mass * gamma * velocity
        
    def relativistic_energy(self, mass: float) -> float:
        """
        Calculate total relativistic energy
        
        Args:
            mass: Rest mass
            
        Returns:
            Total energy (including rest energy)
        """
        return self.gamma * mass * self.c**2


class MinkowskiSpacetime:
    """Class for handling 4D spacetime calculations"""
    
    def __init__(self):
        """Initialize Minkowski spacetime"""
        self.c = 299792458  # Speed of light in m/s
        self.metric = np.diag([1, -1, -1, -1])  # Metric tensor
        
    def interval(self, event1: ArrayLike, event2: ArrayLike) -> float:
        """
        Calculate spacetime interval between two events
        
        Args:
            event1: First event coordinates (t, x, y, z)
            event2: Second event coordinates (t, x, y, z)
            
        Returns:
            Spacetime interval
        """
        event1, event2 = np.array(event1), np.array(event2)
        delta = event1 - event2
        delta[0] *= self.c  # Convert time component
        return np.sqrt(np.dot(np.dot(delta, self.metric), delta))
    
    def proper_time(self, worldline: ArrayLike) -> float:
        """
        Calculate proper time along a worldline
        
        Args:
            worldline: Array of 4D spacetime points
            
        Returns:
            Total proper time along worldline
        """
        total_time = 0
        points = np.array(worldline)
        for i in range(len(points)-1):
            total_time += self.interval(points[i], points[i+1])
        return total_time / self.c


class RelativisticParticle:
    """Class representing a relativistic particle"""
    
    def __init__(self, mass: float, position: ArrayLike, velocity: ArrayLike):
        """
        Initialize relativistic particle
        
        Args:
            mass: Rest mass
            position: Initial position vector
            velocity: Initial velocity vector
        """
        self.c = 299792458
        self.mass = mass
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.proper_time = 0
        self.history = {'position': [self.position.copy()],
                       'velocity': [self.velocity.copy()],
                       'proper_time': [0]}
        
    def update(self, force: ArrayLike, dt: float):
        """
        Update particle state under relativistic force
        
        Args:
            force: Applied force vector
            dt: Time step
        """
        beta = np.linalg.norm(self.velocity) / self.c
        gamma = 1 / np.sqrt(1 - beta**2)
        
        # Relativistic acceleration
        acceleration = force / (gamma**3 * self.mass)
        
        # Update velocity and position
        self.velocity += acceleration * dt
        self.position += self.velocity * dt
        
        # Update proper time
        self.proper_time += dt / gamma
        
        # Store history
        self.history['position'].append(self.position.copy())
        self.history['velocity'].append(self.velocity.copy())
        self.history['proper_time'].append(self.proper_time)
