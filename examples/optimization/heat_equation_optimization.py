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
    dt = 0.4 * dx * dx / true_diffusivity  # This was correct
    
    # Reduce total simulation time to match stability requirements
    total_time = 0.001  # Reduced from 1.0 to 0.001 seconds
    
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
    dt = 0.4 * dx * dx / diffusivity  # Calculate dt based on diffusivity
    T0 = T_data[0]
    
    # Solve heat equation with current diffusivity
    t_model, T_model = solve_heat_equation_1d(
        T0, dx, dt, diffusivity, t[-1],
        boundary_conditions=("dirichlet", "dirichlet"),
        boundary_values=(20, 20)
    )
    
    # Calculate MSE
    # Ensure arrays are same shape before calculating MSE
    min_length = min(len(T_model), len(T_data))
    T_model = T_model[:min_length]
    T_data = T_data[:min_length]
    
    return np.mean((T_model - T_data)**2)

def gradient_objective(diffusivity, x, t, T_data, eps=1e-6):
    """Calculate numerical gradient of objective function"""
    f0 = objective_function(diffusivity, x, t, T_data)
    f1 = objective_function(diffusivity + eps, x, t, T_data)
    return (f1 - f0) / eps

def compare_optimizers(x, t, T_data, true_diffusivity):
    """Compare different optimization methods"""
    initial_guess = 5e-5  # Initial guess for diffusivity
    
    # Lists to store optimization paths
    newton_path = [initial_guess]
    grad_path = [initial_guess]
    nelder_path = [initial_guess]
    
    # Newton's method
    newton_result = newton_optimize(
        lambda d: objective_function(d, x, t, T_data),
        lambda d: gradient_objective(d, x, t, T_data),
        initial_guess,
        tol=1e-8
    )
    newton_path.append(newton_result[0])  # Just add the final result
    
    # Gradient descent
    grad_result = gradient_descent(
        lambda d: objective_function(d, x, t, T_data),
        lambda d: gradient_objective(d, x, t, T_data),
        np.array([initial_guess]),
        learning_rate=1e-6,
        tol=1e-8,
    )
    grad_path.append(grad_result[0][0])
    
    # Nelder-Mead
    nelder_result = nelder_mead(
        lambda d: objective_function(d, x, t, T_data),
        np.array([initial_guess]),
        step=1e-5,
        tol=1e-8,
    )
    nelder_path.append(nelder_result[0][0])
    
    return {
        'Newton': (newton_result[0], newton_path),
        'Gradient Descent': (grad_result[0][0], grad_path),
        'Nelder-Mead': (nelder_result[0][0], nelder_path),
        'True': true_diffusivity
    }

def plot_results(results):
    """Plot comparison of optimization results and convergence paths"""
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Bar plot of final results
    methods = ['Newton', 'Gradient Descent', 'Nelder-Mead', 'True']
    values = [results[m][0] if isinstance(results[m], tuple) else results[m] for m in methods]
    
    ax1.bar(methods, values)
    ax1.set_ylabel('Thermal Diffusivity (m²/s)')
    ax1.set_title('Final Results of Optimization Methods')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for i, v in enumerate(values):
        ax1.text(i, v, f'{v:.2e}', ha='center', va='bottom')
    
    # Plot 2: Convergence paths
    for method in ['Newton', 'Gradient Descent', 'Nelder-Mead']:
        path = results[method][1]
        iterations = range(len(path))
        ax2.plot(iterations, path, marker='o', label=method, markersize=4)
    
    ax2.axhline(y=results['True'], color='r', linestyle='--', label='True Value')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Thermal Diffusivity (m²/s)')
    ax2.set_title('Optimization Convergence Paths')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()

def main():
    # Generate synthetic data
    x, t, T_data, true_diffusivity = generate_synthetic_data()

    # Compare optimization methods
    results = compare_optimizers(x, t, T_data, true_diffusivity)

    # Plot results
    plot_results(results)

    # Save plot to output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'heat_equation_optimization.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'heat_equation_optimization.png')}")

if __name__ == "__main__":
    main()