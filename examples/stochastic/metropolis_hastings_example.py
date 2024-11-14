"""
Example demonstrating the Metropolis-Hastings algorithm for optimization.

This example shows how to use MCMC sampling to find the global minimum
of a multi-modal function with noise.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.stochastic import MetropolisHastings

def target_function(x: np.ndarray) -> float:
    """
    Multi-modal target function to optimize.
    Has two local minima and one global minimum.
    """
    return (x[0]**2 + x[1]**2) * np.exp(-0.1 * (x[0]**2 + x[1]**2)) + \
           0.5 * np.sin(5*x[0]) * np.cos(5*x[1])

def proposal_function(state: np.ndarray) -> np.ndarray:
    """Generate proposal state using random walk"""
    return state + np.random.normal(0, 0.2, size=state.shape)

def log_likelihood(state: np.ndarray) -> float:
    """Convert target function to log likelihood"""
    return -target_function(state)  # Negative because we want to minimize

def main():
    # Initial state
    initial_state = np.array([2.0, 2.0])
    
    # Create Metropolis-Hastings sampler
    sampler = MetropolisHastings(
        log_likelihood=log_likelihood,
        proposal=proposal_function,
        initial_state=initial_state,
        seed=42
    )
    
    # Run chain with simulated annealing
    n_steps = 10000
    best_state, best_ll = sampler.run(
        n_steps=n_steps,
        beta=0.1,  # Initial temperature
        cooling_rate=1e-4,  # Temperature increase per step
        early_stop=1000  # Stop if no improvement in 1000 steps
    )
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    # Plot target function contours
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i,j] = target_function(np.array([X[i,j], Y[i,j]]))
    
    # Plot optimization trajectory
    states = np.array(sampler.history['states'])
    
    plt.subplot(121)
    plt.contour(X, Y, Z, levels=20)
    plt.colorbar(label='Target Function Value')
    plt.plot(states[:,0], states[:,1], 'r.', alpha=0.1, label='MCMC Samples')
    plt.plot(best_state[0], best_state[1], 'g*', markersize=15, label='Best Found')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Optimization Trajectory')
    plt.legend()
    
    # Plot log likelihood history
    plt.subplot(122)
    plt.plot(sampler.history['log_likelihoods'])
    plt.xlabel('Step')
    plt.ylabel('Log Likelihood')
    plt.title('Convergence History')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 