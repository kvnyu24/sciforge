from ..core.base import BaseProcess
import numpy as np
from typing import Callable, Union, Optional, Tuple, Dict

class PoissonProcess(BaseProcess):
    """Implements Poisson process"""
    def simulate(self, rate, T, N):
        """
        Simulate Poisson process
        
        Args:
            rate: Average rate of events
            T: Final time
            N: Number of steps
        
        Returns:
            Time points and process values
        """
        dt = T/N
        t = np.linspace(0, T, N)
        X = np.zeros(N)
        X[0] = 0
        for i in range(1, N):
            # Generate Poisson increment
            dN = np.random.poisson(rate * dt)
            X[i] = X[i-1] + dN
        return t, X

class WienerProcess(BaseProcess):
    """Implements Wiener process (Brownian motion)"""
    def simulate(self, T, N):
        """
        Simulate Wiener process
        T: final time
        N: number of steps
        """
        dt = T/N
        dW = np.random.normal(0, np.sqrt(dt), N)
        W = np.cumsum(dW)
        return np.linspace(0, T, N), W

class OrnsteinUhlenbeck(BaseProcess):
    """Implements Ornstein-Uhlenbeck process"""
    def simulate(self, params, T, N):
        """
        Simulate OU process
        params: (theta, mu, sigma) parameters
        T: final time
        N: number of steps
        """
        theta, mu, sigma = params
        dt = T/N
        t = np.linspace(0, T, N)
        X = np.zeros(N)
        X[0] = mu
        for i in range(1, N):
            dW = np.random.normal(0, np.sqrt(dt))
            X[i] = X[i-1] + theta*(mu - X[i-1])*dt + sigma*dW
        return t, X

class GeometricBrownianMotion(BaseProcess):
    """Implements Geometric Brownian Motion (GBM)"""
    
    def sample_increment(self, dt: float) -> float:
        """Generate random normal increment"""
        return self.rng.normal(0, np.sqrt(dt))
    
    def simulate(self, params, T, N, initial_state=1.0):
        """
        Simulate GBM process
        params: (mu, sigma) drift and volatility parameters
        T: final time
        N: number of steps
        """
        mu, sigma = params
        dt = T/N
        t = np.linspace(0, T, N)
        X = np.zeros(N)
        X[0] = initial_state
        
        for i in range(1, N):
            dW = self.sample_increment(dt)
            X[i] = X[i-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*dW)
            
            if self.store_history:
                self.save_state(t[i], X[i])
                
        return t, X
    
    def mean(self, t: float) -> float:
        """Theoretical mean of GBM"""
        mu = self.params[0]
        return self.initial_state * np.exp(mu * t)
    
    def variance(self, t: float) -> float:
        """Theoretical variance of GBM"""
        mu, sigma = self.params
        m = self.mean(t)
        return m**2 * (np.exp(sigma**2 * t) - 1)

class CoxIngersollRoss(BaseProcess):
    """Implements Cox-Ingersoll-Ross (CIR) process"""
    def simulate(self, params, T, N):
        """
        Simulate CIR process
        params: (kappa, theta, sigma) mean reversion, long-term mean, and volatility
        T: final time
        N: number of steps
        """
        kappa, theta, sigma = params
        dt = T/N
        t = np.linspace(0, T, N)
        X = np.zeros(N)
        X[0] = theta
        for i in range(1, N):
            dW = np.random.normal(0, np.sqrt(dt))
            X[i] = X[i-1] + kappa*(theta - X[i-1])*dt + sigma*np.sqrt(max(X[i-1], 0))*dW
        return t, X

class VasicekModel(BaseProcess):
    """Implements Vasicek interest rate model"""
    def simulate(self, params, T, N):
        """
        Simulate Vasicek process
        params: (kappa, theta, sigma) mean reversion, long-term rate, and volatility
        T: final time
        N: number of steps
        """
        kappa, theta, sigma = params
        dt = T/N
        t = np.linspace(0, T, N)
        r = np.zeros(N)
        r[0] = theta
        for i in range(1, N):
            dW = np.random.normal(0, np.sqrt(dt))
            r[i] = r[i-1] + kappa*(theta - r[i-1])*dt + sigma*dW
        return t, r

class MetropolisHastings(BaseProcess):
    """
    Implements the Metropolis-Hastings algorithm for MCMC sampling.
    """
    
    def __init__(self, 
                 log_likelihood: Callable,
                 proposal: Callable,
                 initial_state: Union[np.ndarray, Dict],
                 seed: Optional[int] = None):
        """
        Initialize Metropolis-Hastings sampler
        
        Args:
            log_likelihood: Log likelihood function to evaluate states
            proposal: Function to generate proposal states
            initial_state: Initial state of the chain
            seed: Random seed for reproducibility
        """
        super().__init__(seed)
        self.log_likelihood = log_likelihood
        self.proposal = proposal
        self.state = initial_state
        self.history = {'states': [initial_state], 'log_likelihoods': []}
        
    def step(self, beta: float = 1.0) -> Tuple[Union[np.ndarray, Dict], float]:
        """
        Perform one step of the Metropolis-Hastings algorithm
        
        Args:
            beta: Inverse temperature parameter for simulated annealing
            
        Returns:
            Tuple of (new state, log likelihood)
        """
        # Generate proposal
        proposed_state = self.proposal(self.state)
        
        # Calculate log likelihoods
        current_ll = self.log_likelihood(self.state)
        proposed_ll = self.log_likelihood(proposed_state)
        
        # Calculate acceptance probability
        log_ratio = beta * (proposed_ll - current_ll)
        accept_prob = np.exp(min(0, log_ratio))
        
        # Accept or reject
        if self.rng.random() < accept_prob:
            self.state = proposed_state
            current_ll = proposed_ll
            
        # Update history
        self.history['states'].append(self.state)
        self.history['log_likelihoods'].append(current_ll)
        
        return self.state, current_ll
        
    def run(self,
            n_steps: int,
            beta: float = 1.0,
            cooling_rate: float = 0.0,
            early_stop: Optional[int] = None) -> Tuple[Union[np.ndarray, Dict], float]:
        """
        Run the Metropolis-Hastings chain
        
        Args:
            n_steps: Number of steps to run
            beta: Initial inverse temperature
            cooling_rate: Rate of temperature decrease
            early_stop: Early stopping threshold for no improvement
            
        Returns:
            Tuple of (best state, best log likelihood)
        """
        best_state = self.state
        best_ll = self.log_likelihood(self.state)
        no_improve = 0
        
        for i in range(n_steps):
            # Update state
            state, ll = self.step(beta)
            
            # Update best state
            if ll > best_ll:
                best_state = state
                best_ll = ll
                no_improve = 0
            else:
                no_improve += 1
                
            # Check early stopping
            if early_stop and no_improve >= early_stop:
                break
                
            # Update temperature
            beta += cooling_rate
            
        return best_state, best_ll