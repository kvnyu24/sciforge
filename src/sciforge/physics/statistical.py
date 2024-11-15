"""
Statistical functions used in physics calculations
"""

import numpy as np
from typing import Union, Optional
from numpy.typing import ArrayLike
from ..core.base import BaseClass

class HermitePolynomial(BaseClass):
    """Class for calculating Hermite polynomials used in quantum mechanics"""
    
    @staticmethod
    def evaluate(n: int, x: ArrayLike) -> np.ndarray:
        """
        Calculate nth Hermite polynomial H_n(x)
        
        Args:
            n: Order of Hermite polynomial
            x: Points at which to evaluate
            
        Returns:
            H_n(x) values
        """
        x = np.asarray(x)
        if n == 0:
            return np.ones_like(x)
        elif n == 1:
            return 2 * x
        else:
            h_prev = np.ones_like(x)  # H_0
            h_curr = 2 * x            # H_1
            
            for i in range(2, n + 1):
                h_next = 2 * x * h_curr - 2 * (i - 1) * h_prev
                h_prev = h_curr
                h_curr = h_next
                
            return h_curr

class RiceDistribution(BaseClass):
    """Class for Rice distribution calculations in physics"""
    
    def __init__(self, nu: float, sigma: float = 1.0):
        """
        Initialize Rice distribution
        
        Args:
            nu: Non-centrality parameter
            sigma: Scale parameter
        """
        self.nu = nu
        self.sigma = sigma
        
    def pdf(self, x: ArrayLike) -> np.ndarray:
        """
        Calculate Rice probability density function
        
        Args:
            x: Points at which to evaluate PDF
            
        Returns:
            PDF values
        """
        x = np.asarray(x)
        sigma2 = self.sigma**2
        
        # Handle x ≤ 0 to avoid warnings
        result = np.zeros_like(x)
        mask = x > 0
        
        x_valid = x[mask]
        result[mask] = (x_valid / sigma2) * np.exp(
            -(x_valid**2 + self.nu**2) / (2 * sigma2)
        ) * np.i0(x_valid * self.nu / sigma2)
        
        return result
        
    def rvs(self, size: Optional[Union[int, tuple]] = None) -> np.ndarray:
        """
        Generate random variates from Rice distribution
        
        Args:
            size: Output shape
            
        Returns:
            Random samples from distribution
        """
        # Generate complex normal variables
        z = (np.random.normal(size=size) + 1j * np.random.normal(size=size)) / np.sqrt(2)
        
        # Add non-centrality and scale
        z = self.sigma * (z + self.nu)
        
        return np.abs(z) 