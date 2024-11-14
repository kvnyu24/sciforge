import numpy as np
from typing import Optional, Tuple, List
from numpy.typing import ArrayLike
from .base import DynamicalSystem
from .mechanics import Particle, Constraint, RotationalSpring
from .rotations import RotationalSystem, calculate_moment_of_inertia

class HarmonicOscillator(DynamicalSystem):
    """Simple harmonic oscillator with optional damping and forcing"""
    def __init__(self, mass: float, spring_constant: float, 
                 position: ArrayLike = None, velocity: ArrayLike = None,
                 damping: float = 0.0):
        super().__init__(mass, 
                        np.zeros(3) if position is None else np.array(position),
                        np.zeros(3) if velocity is None else np.array(velocity))
        self.k = spring_constant
        self.damping = damping
        self.natural_frequency = np.sqrt(self.k / self.mass)
        self.history = {'time': [], 'position': [], 'velocity': [], 'energy': []}
        
    def force(self, position: ArrayLike, velocity: ArrayLike, 
             external_force: Optional[ArrayLike] = None) -> np.ndarray:
        """Calculate total force including spring, damping and external forces"""
        spring_force = -self.k * position
        damping_force = -self.damping * velocity
        ext_force = np.zeros(3) if external_force is None else external_force
        return spring_force + damping_force + ext_force
        
    def update(self, external_force: Optional[ArrayLike], dt: float) -> None:
        """Update oscillator state using RK4 integration"""
        # RK4 integration steps
        k1v = self.force(self.position, self.velocity, external_force) / self.mass
        k1x = self.velocity
        
        v_temp = self.velocity + 0.5 * dt * k1v
        x_temp = self.position + 0.5 * dt * k1x
        k2v = self.force(x_temp, v_temp, external_force) / self.mass
        k2x = v_temp
        
        v_temp = self.velocity + 0.5 * dt * k2v
        x_temp = self.position + 0.5 * dt * k2x
        k3v = self.force(x_temp, v_temp, external_force) / self.mass
        k3x = v_temp
        
        v_temp = self.velocity + dt * k3v
        x_temp = self.position + dt * k3x
        k4v = self.force(x_temp, v_temp, external_force) / self.mass
        k4x = v_temp
        
        # Update state
        self.velocity += (dt/6.0) * (k1v + 2*k2v + 2*k3v + k4v)
        self.position += (dt/6.0) * (k1x + 2*k2x + 2*k3x + k4x)
        
        # Update history
        self.history['position'].append(self.position.copy())
        self.history['velocity'].append(self.velocity.copy())
        self.history['energy'].append(self.total_energy())
        
    def total_energy(self) -> float:
        """Calculate total mechanical energy"""
        kinetic = 0.5 * self.mass * np.dot(self.velocity, self.velocity)
        potential = 0.5 * self.k * np.dot(self.position, self.position)
        return kinetic + potential

    def get_phase_space_trajectory(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get phase space trajectory from history"""
        positions = np.array(self.history['position'])
        velocities = np.array(self.history['velocity'])
        return positions, velocities

class CoupledOscillator:
    """System of coupled harmonic oscillators"""
    def __init__(self, masses: ArrayLike, spring_constants: ArrayLike,
                 initial_positions: ArrayLike, initial_velocities: ArrayLike,
                 coupling_constants: Optional[ArrayLike] = None):
        self.n_oscillators = len(masses)
        self.oscillators = [
            HarmonicOscillator(m, k, pos, vel)
            for m, k, pos, vel in zip(masses, spring_constants, 
                                    initial_positions, initial_velocities)
        ]
        self.coupling_constants = coupling_constants if coupling_constants is not None else \
                                np.ones(self.n_oscillators - 1) * min(spring_constants)
        
    def coupling_force(self, i: int, j: int) -> np.ndarray:
        """Calculate coupling force between oscillators i and j"""
        displacement = self.oscillators[j].position - self.oscillators[i].position
        k_coupling = self.coupling_constants[min(i,j)]
        return k_coupling * displacement

    def normal_modes(self) -> List[float]:
        """Calculate normal mode frequencies"""
        # Construct mass and spring matrices
        M = np.diag([osc.mass for osc in self.oscillators])
        K = np.zeros((self.n_oscillators, self.n_oscillators))
        
        for i in range(self.n_oscillators):
            K[i,i] = self.oscillators[i].k
            if i > 0:
                K[i,i-1] = K[i-1,i] = -self.coupling_constants[i-1]
        
        # Solve eigenvalue problem
        eigenvals = np.linalg.eigvals(np.linalg.inv(M) @ K)
        return np.sqrt(np.abs(eigenvals))
        
    def update(self, dt: float) -> None:
        """Update all oscillators including coupling forces"""
        for i in range(self.n_oscillators):
            coupling_forces = np.zeros(3)
            for j in range(self.n_oscillators):
                if i != j:
                    coupling_forces += self.coupling_force(i, j)
            self.oscillators[i].update(coupling_forces, dt)

class TorsionalOscillator(RotationalSystem):
    """Rotational harmonic oscillator"""
    def __init__(self, moment_of_inertia: float, torsion_constant: float,
                 angular_position: Optional[ArrayLike] = None,
                 angular_velocity: Optional[ArrayLike] = None):
        super().__init__(moment_of_inertia, angular_position, angular_velocity)
        self.spring = RotationalSpring(torsion_constant)
        
    def update(self, external_torque: Optional[ArrayLike], dt: float) -> None:
        """Update using spring torque and external torque"""
        total_torque = self.spring.torque(self.position[2])
        if external_torque is not None:
            total_torque += external_torque
        super().update(total_torque, dt)

class ResonantSystem(HarmonicOscillator):
    """Harmonic oscillator with resonance analysis"""
    def __init__(self, mass: float, spring_constant: float,
                 driving_frequency: float, driving_amplitude: float,
                 damping: float = 0.0):
        super().__init__(mass, spring_constant, damping=damping)
        self.driving_freq = driving_frequency
        self.driving_amp = driving_amplitude
        self.phase = 0.0
        self.resonance_data = {'frequencies': [], 'amplitudes': []}
        
    def driving_force(self, t: float) -> np.ndarray:
        """Calculate time-dependent driving force"""
        force = self.driving_amp * np.sin(self.driving_freq * t + self.phase)
        return force * np.array([1, 0, 0])
        
    def resonance_amplitude(self) -> float:
        """Calculate theoretical resonance amplitude"""
        beta = self.damping / (2 * self.mass)
        omega = self.natural_frequency
        omega_d = self.driving_freq
        denominator = np.sqrt((omega**2 - omega_d**2)**2 + 4*beta**2*omega_d**2)
        return self.driving_amp / (self.mass * denominator)

    def scan_frequencies(self, freq_range: ArrayLike, duration: float, dt: float) -> None:
        """Scan through frequencies to find resonance peaks"""
        original_freq = self.driving_freq
        for freq in freq_range:
            self.driving_freq = freq
            self.reset_state()
            
            # Simulate for duration
            t = 0
            while t < duration:
                self.update(self.driving_force(t), dt)
                t += dt
            
            # Record steady state amplitude
            amplitude = np.max(np.abs(self.history['position'][-100:]))
            self.resonance_data['frequencies'].append(freq)
            self.resonance_data['amplitudes'].append(amplitude)
            
        self.driving_freq = original_freq

class ParametricOscillator(HarmonicOscillator):
    """Oscillator with time-varying parameters"""
    def __init__(self, mass: float, base_spring_constant: float,
                 modulation_amplitude: float, modulation_frequency: float):
        super().__init__(mass, base_spring_constant)
        self.k0 = base_spring_constant
        self.mod_amp = modulation_amplitude
        self.mod_freq = modulation_frequency
        
    def get_spring_constant(self, t: float) -> float:
        """Get time-dependent spring constant"""
        return self.k0 * (1 + self.mod_amp * np.cos(self.mod_freq * t))
        
    def force(self, position: ArrayLike, velocity: ArrayLike,
             external_force: Optional[ArrayLike] = None, t: float = 0) -> np.ndarray:
        """Calculate force with time-varying spring constant"""
        self.k = self.get_spring_constant(t)
        return super().force(position, velocity, external_force)
