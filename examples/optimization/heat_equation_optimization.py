"""
Example comparing different optimization methods for finding optimal parameters
in a heat diffusion PDE problem.

This example tries to find the optimal thermal diffusivity parameter that best
matches experimental temperature data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.numerical.optimization import newton_optimize, gradient_descent, nelder_mead

# Reference the heat equation solver
from examples.thermodynamics.heat_diffusion import solve_heat_equation_1d

def generate_synthetic_data(noise_level=0.1):
    """Generate synthetic temperature data with noise"""
    # True parameters (copper)
    true_diffusivity = 1.11e-4  # m²/s
    
    # Simulation parameters
    L = 0.1  # Length (m)
    nx = 50  # Spatial points
    dx = L / (nx - 1)
    dt = 0.001
    total_time = 1.0
    
    # Initial temperature distribution
    x = np.linspace(0, L, nx)
    T0 = 20 + 80 * np.exp(-(x - L/2)**2 / (0.01 * L)**2)
    
    # Generate true solution
    t, T = solve_heat_equation_1d(
        T0, dx, dt, true_diffusivity, total_time,
        boundary_conditions=("dirichlet", "dirichlet"),
        boundary_values=(20, 20)
    )
    
    # Add noise
    T_noisy = T + noise_level * np.random.randn(*T.shape)
    
    return x, t, T_noisy, true_diffusivity

def objective_function(diffusivity, x, t, T_data):
    """Calculate mean squared error between model and data"""
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    T0 = T_data[0]
    
    # Solve heat equation with current diffusivity
    _, T_model = solve_heat_equation_1d(
        T0, dx, dt, diffusivity, t[-1],
        boundary_conditions=("dirichlet", "dirichlet"),
        boundary_values=(20, 20)
    )
    
    # Calculate MSE
    return np.mean((T_model - T_data)**2)

def gradient_objective(diffusivity, x, t, T_data, eps=1e-6):
    """Calculate numerical gradient of objective function"""
    f0 = objective_function(diffusivity, x, t, T_data)
    f1 = objective_function(diffusivity + eps, x, t, T_data)
    return (f1 - f0) / eps

def compare_optimizers(x, t, T_data, true_diffusivity):
    """Compare different optimization methods"""
    initial_guess = 5e-5  # Initial guess for diffusivity
    
    # Newton's method
    newton_result = newton_optimize(
        lambda d: objective_function(d, x, t, T_data),
        lambda d: gradient_objective(d, x, t, T_data),
        initial_guess,
        tol=1e-8
    )
    
    # Gradient descent
    grad_result = gradient_descent(
        lambda d: objective_function(d, x, t, T_data),
        lambda d: gradient_objective(d, x, t, T_data),
        np.array([initial_guess]),
        learning_rate=1e-6,
        tol=1e-8
    )
    
    # Nelder-Mead
    nelder_result = nelder_mead(
        lambda d: objective_function(d, x, t, T_data),
        np.array([initial_guess]),
        step=1e-5,
        tol=1e-8
    )
    
    return {
        'Newton': newton_result[0],
        'Gradient Descent': grad_result[0][0],
        'Nelder-Mead': nelder_result[0][0],
        'True': true_diffusivity
    }

def plot_results(results):
    """Plot comparison of optimization results"""
    methods = list(results.keys())
    values = [results[m] for m in methods]
    
    plt.figure(figsize=(10, 6))
    plt.bar(methods, values)
    plt.ylabel('Thermal Diffusivity (m²/s)')
    plt.title('Comparison of Optimization Methods')
    plt.xticks(rotation=45)
    
    # Add value labels
    for i, v in enumerate(values):
        plt.text(i, v, f'{v:.2e}', ha='center', va='bottom')
    
    plt.tight_layout()

def main():
    # Generate synthetic data
    x, t, T_data, true_diffusivity = generate_synthetic_data()
    
    # Compare optimization methods
    results = compare_optimizers(x, t, T_data, true_diffusivity)
    
    # Plot results
    plot_results(results)
    plt.show()

if __name__ == "__main__":
    main() 