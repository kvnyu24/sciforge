import numpy as np
from typing import Optional, Tuple
from numpy.typing import ArrayLike
from ..core.constants import CONSTANTS
from .base import DynamicalSystem
from .fields import GravitationalField, UniformField

class Particle(DynamicalSystem):
    """Classical particle mechanics with advanced integration methods"""
    def __init__(self, mass, position, velocity, drag_coeff=0.0, gravity=CONSTANTS['G']):
        super().__init__(mass, position, velocity)
        self.drag_coeff = drag_coeff
        self.gravity = gravity
        self.grav_field = UniformField(np.array([0, 0, -gravity]))
        self.history = {'position': [], 'velocity': [], 'acceleration': []}
    
    def update(self, force, dt):
        """Update particle state using 4th order Runge-Kutta integration
        
        Includes:
        - Applied external forces
        - Gravitational force
        - Drag force proportional to velocity squared
        - Position and velocity history tracking
        """
        # Calculate total force including gravity and drag
        grav_force = self.mass * self.grav_field.field(self.position)
        drag_force = -0.5 * self.drag_coeff * np.linalg.norm(self.velocity) * self.velocity
        total_force = force + grav_force + drag_force
        
        # RK4 integration for position and velocity
        k1v = total_force / self.mass
        k1x = self.velocity
        
        v_temp = self.velocity + 0.5 * dt * k1v
        k2v = (force + grav_force - 0.5 * self.drag_coeff * np.linalg.norm(v_temp) * v_temp) / self.mass
        k2x = v_temp
        
        v_temp = self.velocity + 0.5 * dt * k2v
        k3v = (force + grav_force - 0.5 * self.drag_coeff * np.linalg.norm(v_temp) * v_temp) / self.mass
        k3x = v_temp
        
        v_temp = self.velocity + dt * k3v
        k4v = (force + grav_force - 0.5 * self.drag_coeff * np.linalg.norm(v_temp) * v_temp) / self.mass
        k4x = v_temp
        
        # Update state
        self.velocity += (dt/6.0) * (k1v + 2*k2v + 2*k3v + k4v)
        self.position += (dt/6.0) * (k1x + 2*k2x + 2*k3x + k4x)
        
        # Store history
        self.history['position'].append(self.position.copy())
        self.history['velocity'].append(self.velocity.copy())
        self.history['acceleration'].append(k1v.copy())

class Spring:
    """Simple harmonic oscillator with spring force"""
    def __init__(self, k: float, rest_length: float = 0.0):
        self.k = k  # Spring constant
        self.rest_length = rest_length
        
    def force(self, displacement: ArrayLike) -> np.ndarray:
        """Calculate spring force F = -kx"""
        stretch = np.linalg.norm(displacement) - self.rest_length
        return -self.k * stretch * displacement / np.linalg.norm(displacement)

class RigidBody(DynamicalSystem):
    """Rigid body dynamics with rotational motion"""
    def __init__(self, mass: float, position: ArrayLike, velocity: ArrayLike,
                 inertia_tensor: ArrayLike, orientation: Optional[ArrayLike] = None,
                 angular_velocity: Optional[ArrayLike] = None):
        super().__init__(mass, position, velocity)
        self.I = np.array(inertia_tensor)
        self.orientation = np.eye(3) if orientation is None else np.array(orientation)
        self.angular_velocity = np.zeros(3) if angular_velocity is None else np.array(angular_velocity)
        self.history = {
            'position': [], 'velocity': [], 'acceleration': [],
            'orientation': [], 'angular_velocity': [], 'angular_acceleration': []
        }
    
    def update(self, force: ArrayLike, torque: ArrayLike, dt: float):
        """Update rigid body state with translational and rotational motion"""
        # Translational motion
        acceleration = force / self.mass
        self.velocity += acceleration * dt
        self.position += self.velocity * dt
        
        # Rotational motion
        angular_acceleration = np.linalg.inv(self.I) @ (
            torque - np.cross(self.angular_velocity, self.I @ self.angular_velocity)
        )
        self.angular_velocity += angular_acceleration * dt
        
        # Update orientation using rotation matrix
        omega_matrix = np.array([
            [0, -self.angular_velocity[2], self.angular_velocity[1]],
            [self.angular_velocity[2], 0, -self.angular_velocity[0]],
            [-self.angular_velocity[1], self.angular_velocity[0], 0]
        ])
        self.orientation += omega_matrix @ self.orientation * dt
        
        # Store history
        self.history['position'].append(self.position.copy())
        self.history['velocity'].append(self.velocity.copy())
        self.history['acceleration'].append(acceleration.copy())
        self.history['orientation'].append(self.orientation.copy())
        self.history['angular_velocity'].append(self.angular_velocity.copy())
        self.history['angular_acceleration'].append(angular_acceleration.copy())

class Pendulum(DynamicalSystem):
    """Simple and double pendulum dynamics"""
    def __init__(self, mass: float, length: float, theta0: float, omega0: float = 0.0,
                 gravity: float = CONSTANTS['G'], damping: float = 0.0):
        position = np.array([length * np.sin(theta0), -length * np.cos(theta0)])
        velocity = np.array([length * omega0 * np.cos(theta0), length * omega0 * np.sin(theta0)])
        super().__init__(mass, position, velocity)
        self.length = length
        self.theta = theta0
        self.omega = omega0
        self.gravity = gravity
        self.damping = damping
        self.history = {'theta': [theta0], 'omega': [omega0]}
    
    def update(self, dt: float):
        """Update pendulum state using semi-implicit Euler method"""
        # Calculate acceleration
        alpha = (-self.gravity/self.length * np.sin(self.theta) - 
                self.damping * self.omega)
        
        # Update state
        self.omega += alpha * dt
        self.theta += self.omega * dt
        
        # Update position and velocity
        self.position[0] = self.length * np.sin(self.theta)
        self.position[1] = -self.length * np.cos(self.theta)
        self.velocity[0] = self.length * self.omega * np.cos(self.theta)
        self.velocity[1] = self.length * self.omega * np.sin(self.theta)
        
        # Store history
        self.history['theta'].append(self.theta)
        self.history['omega'].append(self.omega)

class Friction:
    """Models static and kinetic friction forces"""
    def __init__(self, static_coeff: float, kinetic_coeff: float):
        self.static_coeff = static_coeff
        self.kinetic_coeff = kinetic_coeff
        
    def force(self, normal_force: float, velocity: ArrayLike) -> np.ndarray:
        """Calculate friction force based on normal force and velocity"""
        speed = np.linalg.norm(velocity)
        if speed < 1e-6:  # Object approximately at rest
            return -self.static_coeff * normal_force * velocity
        return -self.kinetic_coeff * normal_force * velocity / speed

class MolecularForce:
    """Models molecular interactions using Lennard-Jones potential"""
    def __init__(self, epsilon: float, sigma: float):
        self.epsilon = epsilon  # Potential well depth
        self.sigma = sigma     # Distance at which potential is zero
        
    def force(self, distance: ArrayLike) -> np.ndarray:
        """Calculate molecular force using Lennard-Jones potential gradient"""
        r = np.linalg.norm(distance)
        magnitude = 24 * self.epsilon * (2 * (self.sigma**12 / r**13) - (self.sigma**6 / r**7))
        return magnitude * distance / r

class Collision:
    """Elastic and inelastic collision handler"""
    def __init__(self, restitution_coeff: float = 1.0):
        self.restitution_coeff = restitution_coeff
        
    def resolve(self, m1: float, m2: float, v1: ArrayLike, v2: ArrayLike, 
               normal: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        """Resolve collision between two particles"""
        # Normalize collision normal
        n = normal / np.linalg.norm(normal)
        
        # Relative velocity
        v_rel = v1 - v2
        
        # Normal impulse magnitude
        j = -(1 + self.restitution_coeff) * np.dot(v_rel, n)
        j /= (1/m1 + 1/m2)
        
        # New velocities
        v1_new = v1 + (j/m1) * n
        v2_new = v2 - (j/m2) * n
        
        return v1_new, v2_new

class Constraint:
    """Holonomic constraints for mechanical systems"""
    def __init__(self, constraint_func, jacobian_func):
        self.constraint = constraint_func
        self.jacobian = jacobian_func
        
    def apply_force(self, position: ArrayLike, velocity: ArrayLike) -> np.ndarray:
        """Calculate constraint force using Lagrange multipliers"""
        # Get constraint value and Jacobian
        c = self.constraint(position)
        J = self.jacobian(position)
        
        # Calculate Lagrange multiplier
        lambda_val = -np.linalg.inv(J @ J.T) @ (J @ velocity)
        
        # Return constraint force
        return J.T @ lambda_val

class RotationalSpring:
    """Torsional spring with angular restoring torque"""
    def __init__(self, k: float, equilibrium_angle: float = 0.0):
        self.k = k  # Torsional spring constant
        self.equilibrium = equilibrium_angle
        
    def torque(self, angle: float) -> float:
        """Calculate restoring torque τ = -kθ"""
        return -self.k * (angle - self.equilibrium)
