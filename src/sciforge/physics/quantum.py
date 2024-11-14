import numpy as np
from typing import Optional, Union, List, Tuple
from numpy.typing import ArrayLike
from .base import QuantumSystem

class Wavefunction(QuantumSystem):
    """Class representing a quantum mechanical wavefunction"""
    
    def __init__(self,
                 psi0: Union[ArrayLike, callable],
                 x: ArrayLike,
                 mass: float = 1.0,
                 hbar: float = 1.0):
        """
        Initialize quantum wavefunction
        
        Args:
            psi0: Initial wavefunction (array or callable)
            x: Position space grid points
            mass: Particle mass (default=1)
            hbar: Reduced Planck constant (default=1)
        """
        super().__init__(mass, x, psi0, hbar)
        
        # Store history
        self.history = {'time': [0], 'psi': [self.psi.copy()]}
        
    def probability_density(self) -> np.ndarray:
        """Calculate probability density |ψ|²"""
        return np.abs(self.psi)**2
        
    def expectation_value(self, operator: np.ndarray) -> float:
        """Calculate quantum expectation value of an operator
        
        Args:
            operator: Array representing the quantum operator
            
        Returns:
            float: Expectation value <ψ|A|ψ>
        """
        return np.sum(np.conjugate(self.psi) * operator * self.psi).real
        
    def evolve(self, dt: float, potential: Optional[ArrayLike] = None):
        """
        Time evolve the wavefunction using split-operator method
        
        Args:
            dt: Time step
            potential: Optional potential energy function V(x)
        """
        # Momentum space grid
        dk = 2 * np.pi / (self.position[-1] - self.position[0])
        k = np.fft.fftfreq(len(self.position), self.position[1] - self.position[0]) * 2 * np.pi
        
        # Kinetic and potential operators
        T = np.exp(-1j * self.hbar * k**2 * dt / (4 * self.mass))
        if potential is not None:
            V = np.exp(-1j * potential * dt / (2 * self.hbar))
            self.psi *= V
            
        # Split-operator evolution
        self.psi = np.fft.ifft(T * np.fft.fft(self.psi))
        if potential is not None:
            self.psi *= V
            
        # Update history
        self.history['time'].append(self.history['time'][-1] + dt)
        self.history['psi'].append(self.psi.copy())

class HarmonicOscillator(QuantumSystem):
    """Quantum harmonic oscillator system"""
    
    def __init__(self, omega: float, x: ArrayLike, mass: float = 1.0, hbar: float = 1.0):
        """
        Initialize harmonic oscillator
        
        Args:
            omega: Angular frequency
            x: Position space grid points
            mass: Particle mass
            hbar: Reduced Planck constant
        """
        super().__init__(mass, x, None, hbar)
        self.omega = omega
        self.potential = 0.5 * mass * omega**2 * x**2
        
    def energy_levels(self, n_levels: int) -> np.ndarray:
        """Get first n energy levels"""
        return np.array([self.hbar * self.omega * (n + 0.5) for n in range(n_levels)])
        
    def eigenstate(self, n: int) -> np.ndarray:
        """Get nth eigenstate wavefunction"""
        x = self.position
        alpha = np.sqrt(self.mass * self.omega / self.hbar)
        prefactor = 1.0 / np.sqrt(2**n * np.math.factorial(n))
        hermite = np.polynomial.hermite.Hermite.basis(n)(alpha * x)
        return prefactor * (alpha/np.pi)**0.25 * np.exp(-alpha * x**2 / 2) * hermite

class ParticleInBox(QuantumSystem):
    """Particle in an infinite potential well"""
    
    def __init__(self, length: float, nx: int = 1000, mass: float = 1.0, hbar: float = 1.0):
        """
        Initialize particle in box
        
        Args:
            length: Box length
            nx: Number of grid points
            mass: Particle mass
            hbar: Reduced Planck constant
        """
        x = np.linspace(0, length, nx)
        super().__init__(mass, x, None, hbar)
        self.length = length
        
    def energy_levels(self, n_levels: int) -> np.ndarray:
        """Get first n energy levels"""
        n = np.arange(1, n_levels + 1)
        return (n * np.pi * self.hbar)**2 / (2 * self.mass * self.length**2)
        
    def eigenstate(self, n: int) -> np.ndarray:
        """Get nth eigenstate wavefunction"""
        return np.sqrt(2/self.length) * np.sin(n * np.pi * self.position / self.length)

class SpinSystem(QuantumSystem):
    """Quantum spin system"""
    
    def __init__(self, spin: float, B: ArrayLike = np.array([0, 0, 1]), g: float = 2.0):
        """
        Initialize spin system
        
        Args:
            spin: Spin quantum number
            B: Magnetic field vector
            g: g-factor
        """
        super().__init__(1.0, np.array([0]), None, 1.0)
        self.spin = spin
        self.B = np.array(B)
        self.g = g
        self.dim = int(2 * spin + 1)
        
    def spin_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get spin matrices Sx, Sy, Sz"""
        m = np.arange(self.spin, -self.spin-1, -1)
        Sz = np.diag(m)
        
        Sp = np.zeros((self.dim, self.dim))
        Sm = np.zeros((self.dim, self.dim))
        for i in range(self.dim-1):
            val = np.sqrt((self.spin - m[i]) * (self.spin + m[i] + 1))
            Sp[i, i+1] = val
            Sm[i+1, i] = val
            
        Sx = (Sp + Sm) / 2
        Sy = -1j * (Sp - Sm) / 2
        return Sx, Sy, Sz
