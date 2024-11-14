import numpy as np
from .base import DynamicalSystem

class Particle(DynamicalSystem):
    """Classical particle mechanics with advanced integration methods"""
    def __init__(self, mass, position, velocity, drag_coeff=0.0, gravity=9.81):
        super().__init__(mass, position, velocity)
        self.drag_coeff = drag_coeff
        self.gravity = gravity
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
        grav_force = -self.mass * self.gravity * np.array([0, 0, 1])
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
        
    def force(self, displacement: np.ndarray) -> np.ndarray:
        """Calculate spring force F = -kx"""
        stretch = np.linalg.norm(displacement) - self.rest_length
        return -self.k * stretch * displacement / np.linalg.norm(displacement)

class RigidBody(DynamicalSystem):
    """Rigid body dynamics with rotational motion"""
    def __init__(self, mass: float, position: np.ndarray, velocity: np.ndarray,
                 inertia_tensor: np.ndarray, orientation: np.ndarray = None,
                 angular_velocity: np.ndarray = None):
        super().__init__(mass, position, velocity)
        self.I = inertia_tensor
        self.orientation = np.eye(3) if orientation is None else orientation
        self.angular_velocity = np.zeros(3) if angular_velocity is None else angular_velocity
        self.history = {
            'position': [], 'velocity': [], 'acceleration': [],
            'orientation': [], 'angular_velocity': [], 'angular_acceleration': []
        }
    
    def update(self, force: np.ndarray, torque: np.ndarray, dt: float):
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
                 gravity: float = 9.81, damping: float = 0.0):
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


        