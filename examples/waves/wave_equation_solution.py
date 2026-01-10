"""
Example demonstrating solutions to the wave equation.

This example shows d'Alembert's solution to the 1D wave equation,
demonstrating traveling waves and their superposition.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.waves import Wave


def gaussian_pulse(x, x0, sigma):
    """Gaussian pulse centered at x0."""
    return np.exp(-(x - x0)**2 / (2 * sigma**2))


def dalembert_solution(x, t, f_func, g_func, c):
    """
    d'Alembert solution to wave equation.

    u(x,t) = (1/2)[f(x-ct) + f(x+ct)] + (1/2c)∫[x-ct to x+ct] g(s)ds

    For simplicity, assuming g(x) = 0 (zero initial velocity):
    u(x,t) = (1/2)[f(x-ct) + f(x+ct)]

    Args:
        x: Position array
        t: Time
        f_func: Initial displacement function
        g_func: Initial velocity function (assumed 0)
        c: Wave speed

    Returns:
        Displacement at time t
    """
    return 0.5 * (f_func(x - c*t) + f_func(x + c*t))


def main():
    # Parameters
    c = 1.0       # Wave speed
    L = 20.0      # Domain length
    sigma = 1.0   # Pulse width

    # Spatial grid
    x = np.linspace(-L/2, L/2, 1000)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Initial condition: Gaussian pulse at center
    def initial_condition(x):
        return gaussian_pulse(x, 0, sigma)

    # Plot 1: Wave evolution (snapshots)
    ax1 = axes[0, 0]

    times = [0, 2, 4, 6, 8]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(times)))

    for t, color in zip(times, colors):
        u = dalembert_solution(x, t, initial_condition, None, c)
        ax1.plot(x, u, color=color, lw=2, label=f't = {t}')

    ax1.set_xlabel('Position x')
    ax1.set_ylabel('Displacement u(x,t)')
    ax1.set_title('Wave Splitting (d\'Alembert Solution)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-L/2, L/2)

    # Plot 2: Space-time diagram
    ax2 = axes[0, 1]

    t_array = np.linspace(0, 10, 200)
    X, T = np.meshgrid(x, t_array)

    U = np.zeros_like(X)
    for i, t in enumerate(t_array):
        U[i, :] = dalembert_solution(x, t, initial_condition, None, c)

    im = ax2.imshow(U, aspect='auto', extent=[x.min(), x.max(), t_array.max(), t_array.min()],
                    cmap='RdBu', vmin=-1, vmax=1)
    ax2.plot([-L/2, L/2], [L/2/c, 0], 'k--', lw=1, alpha=0.5)  # Characteristic lines
    ax2.plot([-L/2, L/2], [0, L/2/c], 'k--', lw=1, alpha=0.5)

    ax2.set_xlabel('Position x')
    ax2.set_ylabel('Time t')
    ax2.set_title('Space-Time Diagram')
    plt.colorbar(im, ax=ax2, label='Displacement')

    # Plot 3: Standing wave from superposition
    ax3 = axes[0, 2]

    # Two counter-propagating waves
    k = 2 * np.pi / 4  # Wave number

    def traveling_right(x, t):
        return 0.5 * np.sin(k * x - c * k * t)

    def traveling_left(x, t):
        return 0.5 * np.sin(k * x + c * k * t)

    times_standing = np.linspace(0, 2*np.pi/(c*k), 8)
    colors_standing = plt.cm.plasma(np.linspace(0, 0.9, len(times_standing)))

    for t, color in zip(times_standing, colors_standing):
        standing = traveling_right(x, t) + traveling_left(x, t)
        ax3.plot(x, standing, color=color, lw=1.5, alpha=0.7)

    ax3.set_xlabel('Position x')
    ax3.set_ylabel('Displacement')
    ax3.set_title('Standing Wave = Two Traveling Waves')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-L/2, L/2)

    # Plot 4: Reflection at fixed boundary
    ax4 = axes[1, 0]

    def reflected_wave(x, t, x_boundary=5):
        """Wave with reflection at fixed boundary (inverted)."""
        # Incident wave
        incident = gaussian_pulse(x - c*t, -5, sigma) if (x - c*t) < x_boundary else 0

        # Reflected wave (inverted, traveling left)
        x_reflected = 2*x_boundary - x
        reflected = -gaussian_pulse(x_reflected - c*t, -5, sigma) if (x_reflected - c*t) < x_boundary else 0

        # Use image method
        forward = gaussian_pulse(x, -5 + c*t, sigma)
        backward = -gaussian_pulse(x, 2*5 - (-5 + c*t), sigma)

        return forward + backward

    times_reflect = [0, 3, 5, 7, 9]
    colors_reflect = plt.cm.viridis(np.linspace(0.1, 0.9, len(times_reflect)))

    for t, color in zip(times_reflect, colors_reflect):
        forward = gaussian_pulse(x, -5 + c*t, sigma)
        backward = -gaussian_pulse(x, 10 - (-5 + c*t), sigma)
        u = forward + backward
        u[x > 5] = 0  # Hard boundary

        ax4.plot(x, u, color=color, lw=2, label=f't = {t}')

    ax4.axvline(x=5, color='black', lw=3, label='Fixed boundary')
    ax4.set_xlabel('Position x')
    ax4.set_ylabel('Displacement')
    ax4.set_title('Reflection at Fixed Boundary (Inverted)')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-L/2, L/2)

    # Plot 5: Dispersion (different speeds for different frequencies)
    ax5 = axes[1, 1]

    # Non-dispersive (all frequencies same speed)
    def wave_packet_nondispersive(x, t, x0, sigma, k0, c):
        return gaussian_pulse(x - c*t, x0, sigma) * np.cos(k0 * (x - c*t))

    # Dispersive (speed depends on frequency)
    def wave_packet_dispersive(x, t, x0, sigma, k0, c0, alpha):
        # Phase velocity depends on k: c(k) = c0 + alpha*k
        # Group velocity: c_g = c0 + 2*alpha*k0
        c_phase = c0 + alpha * k0
        c_group = c0 + 2 * alpha * k0
        return gaussian_pulse(x - c_group*t, x0, sigma) * np.cos(k0 * (x - c_phase*t))

    x0 = -5
    k0 = 2 * np.pi / 2

    t_disp = 5
    u_nondisp = wave_packet_nondispersive(x, t_disp, x0, sigma, k0, c)
    u_disp = wave_packet_dispersive(x, t_disp, x0, sigma, k0, c, 0.1)

    ax5.plot(x, u_nondisp, 'b-', lw=2, label='Non-dispersive')
    ax5.plot(x, u_disp, 'r--', lw=2, label='Dispersive')
    ax5.set_xlabel('Position x')
    ax5.set_ylabel('Displacement')
    ax5.set_title(f'Dispersion Effects (t = {t_disp})')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(-L/2, L/2)

    # Plot 6: Energy propagation
    ax6 = axes[1, 2]

    # Energy density proportional to (du/dt)² + c²(du/dx)²
    t_energy = 5
    u = dalembert_solution(x, t_energy, initial_condition, None, c)
    dx = x[1] - x[0]

    # Numerical derivatives
    dudt = (dalembert_solution(x, t_energy + 0.01, initial_condition, None, c) -
            dalembert_solution(x, t_energy - 0.01, initial_condition, None, c)) / 0.02
    dudx = np.gradient(u, dx)

    energy_density = 0.5 * (dudt**2 + c**2 * dudx**2)

    ax6.fill_between(x, energy_density, alpha=0.5, label='Energy density')
    ax6.plot(x, energy_density, 'b-', lw=2)
    ax6.set_xlabel('Position x')
    ax6.set_ylabel('Energy Density')
    ax6.set_title(f'Energy Distribution (t = {t_energy})')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(-L/2, L/2)

    plt.suptitle('Wave Equation: ∂²u/∂t² = c²∂²u/∂x²\n'
                 'd\'Alembert Solution: u(x,t) = f(x-ct) + g(x+ct)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'wave_equation_solution.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'wave_equation_solution.png')}")


if __name__ == "__main__":
    main()
