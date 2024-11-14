class Particle:
    """Classical particle mechanics with advanced integration methods"""
    def __init__(self, mass, position, velocity, drag_coeff=0.0, gravity=9.81):
        self.mass = mass
        self.position = position
        self.velocity = velocity
        self.drag_coeff = drag_coeff
        self.gravity = gravity
        self.forces = []
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