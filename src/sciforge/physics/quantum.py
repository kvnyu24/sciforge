import numpy as np
from typing import Optional, Union
from numpy.typing import ArrayLike

class Wavefunction:
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
        self.x = np.array(x)
        self.mass = mass
        self.hbar = hbar
        
        # Initialize wavefunction
        if callable(psi0):
            self.psi = psi0(self.x)
        else:
            self.psi = np.array(psi0)
            
        # Normalize
        self._normalize()
        
        # Store history
        self.history = {'time': [0], 'psi': [self.psi.copy()]}
        
    def _normalize(self):
        """Normalize the wavefunction"""
        dx = self.x[1] - self.x[0]
        norm = np.sqrt(np.sum(np.abs(self.psi)**2) * dx)
        self.psi /= norm
        
    def probability_density(self) -> np.ndarray:
        """Calculate probability density |ψ|²"""
        return np.abs(self.psi)**2
        
    def expectation_value(self, operator: np.ndarray) -> complex:
        """
        Calculate expectation value <ψ|A|ψ>
        
        Args:
            operator: Quantum operator matrix
            
        Returns:
            Complex expectation value
        """
        dx = self.x[1] - self.x[0]
        return np.sum(np.conj(self.psi) * operator @ self.psi) * dx
        
    def evolve(self, dt: float, potential: Optional[ArrayLike] = None):
        """
        Time evolve the wavefunction using split-operator method
        
        Args:
            dt: Time step
            potential: Optional potential energy function V(x)
        """
        # Momentum space grid
        dk = 2 * np.pi / (self.x[-1] - self.x[0])
        k = np.fft.fftfreq(len(self.x), self.x[1] - self.x[0]) * 2 * np.pi
        
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
