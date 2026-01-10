"""
Example comparing steady-state and transient heat conduction.

Steady State:
- Temperature profile does not change with time
- For 1D: d^2T/dx^2 = 0 (Laplace equation)
- Linear temperature profile between boundaries

Transient (Time-dependent):
- Temperature evolves according to heat equation
- dT/dt = alpha * d^2T/dx^2
- Approaches steady state as t -> infinity

This example shows:
- Steady state temperature distribution
- Transient evolution toward steady state
- Comparison of different materials (thermal diffusivity)
- Effect of boundary conditions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from src.sciforge.core.constants import CONSTANTS


def steady_state_1d(x, T_left, T_right, L):
    """
    Calculate steady-state temperature distribution in 1D.

    For d^2T/dx^2 = 0: T(x) = T_left + (T_right - T_left) * x/L

    Args:
        x: Position array (m)
        T_left: Temperature at left boundary (K or C)
        T_right: Temperature at right boundary (K or C)
        L: Length of domain (m)

    Returns:
        Temperature array
    """
    return T_left + (T_right - T_left) * x / L


def solve_heat_equation_ftcs(T0, alpha, dx, dt, nt, T_left, T_right):
    """
    Solve 1D heat equation using Forward Time Central Space (FTCS) method.

    dT/dt = alpha * d^2T/dx^2

    Args:
        T0: Initial temperature distribution
        alpha: Thermal diffusivity (m^2/s)
        dx: Spatial step (m)
        dt: Time step (s)
        nt: Number of time steps
        T_left: Left boundary temperature
        T_right: Right boundary temperature

    Returns:
        Temperature evolution array (nt+1 x nx)
    """
    nx = len(T0)
    T = np.zeros((nt + 1, nx))
    T[0, :] = T0

    # Stability check
    r = alpha * dt / dx**2
    if r > 0.5:
        print(f"Warning: r = {r:.3f} > 0.5, solution may be unstable")

    for n in range(nt):
        # Interior points
        T[n+1, 1:-1] = T[n, 1:-1] + r * (T[n, 2:] - 2*T[n, 1:-1] + T[n, :-2])
        # Boundary conditions
        T[n+1, 0] = T_left
        T[n+1, -1] = T_right

    return T


def thermal_diffusivity(k, rho, cp):
    """
    Calculate thermal diffusivity: alpha = k / (rho * cp)

    Args:
        k: Thermal conductivity (W/(m*K))
        rho: Density (kg/m^3)
        cp: Specific heat capacity (J/(kg*K))

    Returns:
        Thermal diffusivity (m^2/s)
    """
    return k / (rho * cp)


def main():
    # Material properties
    materials = {
        'Copper': {'k': 401, 'rho': 8960, 'cp': 385, 'color': 'orange'},
        'Aluminum': {'k': 237, 'rho': 2700, 'cp': 897, 'color': 'silver'},
        'Steel': {'k': 50, 'rho': 7850, 'cp': 500, 'color': 'gray'},
        'Glass': {'k': 1.0, 'rho': 2500, 'cp': 840, 'color': 'lightblue'},
        'Wood': {'k': 0.15, 'rho': 600, 'cp': 1700, 'color': 'brown'},
    }

    # Calculate thermal diffusivities
    for name, props in materials.items():
        props['alpha'] = thermal_diffusivity(props['k'], props['rho'], props['cp'])
        print(f"{name}: alpha = {props['alpha']*1e6:.2f} x 10^-6 m^2/s")

    # Domain parameters
    L = 0.1          # 10 cm
    nx = 51
    x = np.linspace(0, L, nx)
    dx = L / (nx - 1)

    # Boundary conditions
    T_left = 100     # C
    T_right = 20     # C

    # Initial condition: uniform temperature
    T0 = np.ones(nx) * 20  # Start at room temperature

    # Steady state solution
    T_steady = steady_state_1d(x, T_left, T_right, L)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Steady state vs initial condition
    ax1 = axes[0, 0]

    ax1.plot(x * 100, T0, 'b--', lw=2, label='Initial condition (t=0)')
    ax1.plot(x * 100, T_steady, 'r-', lw=2, label='Steady state (t -> infinity)')

    ax1.fill_between(x * 100, T0, T_steady, alpha=0.2, color='purple')

    ax1.set_xlabel('Position (cm)', fontsize=12)
    ax1.set_ylabel('Temperature (C)', fontsize=12)
    ax1.set_title('Steady State vs Initial Condition', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add boundary condition annotations
    ax1.annotate(f'T = {T_left}C', xy=(0, T_left), xytext=(1, T_left + 5),
                fontsize=10, color='red')
    ax1.annotate(f'T = {T_right}C', xy=(10, T_right), xytext=(8, T_right - 10),
                fontsize=10, color='red')

    # Plot 2: Transient evolution for copper
    ax2 = axes[0, 1]

    # Time parameters for copper
    alpha_cu = materials['Copper']['alpha']
    dt = 0.01  # s
    total_time = 5  # s
    nt = int(total_time / dt)

    T_evolution = solve_heat_equation_ftcs(T0, alpha_cu, dx, dt, nt, T_left, T_right)

    # Plot at different times
    times = [0, 0.1, 0.5, 1.0, 2.0, 5.0]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(times)))

    for t, color in zip(times, colors):
        n = int(t / dt)
        if n < len(T_evolution):
            ax2.plot(x * 100, T_evolution[n], color=color, lw=2, label=f't = {t:.1f}s')

    ax2.plot(x * 100, T_steady, 'k--', lw=2, label='Steady state')

    ax2.set_xlabel('Position (cm)', fontsize=12)
    ax2.set_ylabel('Temperature (C)', fontsize=12)
    ax2.set_title(f'Transient Heat Conduction in Copper\n'
                  f'(alpha = {alpha_cu*1e6:.1f} x 10^-6 m^2/s)', fontsize=12)
    ax2.legend(loc='right')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Comparison of materials at same time
    ax3 = axes[1, 0]

    t_compare = 2.0  # Compare at t = 2s

    for name, props in list(materials.items())[:4]:  # Skip wood (too slow)
        alpha = props['alpha']
        dt_material = 0.0001 if alpha > 1e-5 else 0.01  # Adjust dt for stability
        nt_material = int(t_compare / dt_material)

        T_evo = solve_heat_equation_ftcs(T0, alpha, dx, dt_material, nt_material,
                                          T_left, T_right)

        ax3.plot(x * 100, T_evo[-1], color=props['color'], lw=2, label=name)

    ax3.plot(x * 100, T_steady, 'k--', lw=2, label='Steady state')

    ax3.set_xlabel('Position (cm)', fontsize=12)
    ax3.set_ylabel('Temperature (C)', fontsize=12)
    ax3.set_title(f'Temperature Profiles at t = {t_compare}s\n'
                  '(Different materials)', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax3.text(0.05, 0.05, 'Higher thermal diffusivity =\nfaster approach to steady state',
             transform=ax3.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Plot 4: Time to reach steady state
    ax4 = axes[1, 1]

    # Calculate characteristic time: tau ~ L^2 / alpha
    material_names = []
    tau_values = []
    alphas = []

    for name, props in materials.items():
        tau = L**2 / props['alpha']
        material_names.append(name)
        tau_values.append(tau)
        alphas.append(props['alpha'] * 1e6)

    # Sort by tau
    sorted_idx = np.argsort(tau_values)
    material_names = [material_names[i] for i in sorted_idx]
    tau_values = [tau_values[i] for i in sorted_idx]
    alphas = [alphas[i] for i in sorted_idx]

    colors = [materials[name]['color'] for name in material_names]

    bars = ax4.barh(material_names, tau_values, color=colors, edgecolor='black', alpha=0.7)

    # Add values
    for bar, tau in zip(bars, tau_values):
        width = bar.get_width()
        ax4.text(width + 10, bar.get_y() + bar.get_height()/2,
                f'{tau:.1f}s', ha='left', va='center', fontsize=10)

    ax4.set_xlabel('Characteristic Time tau = L^2/alpha (s)', fontsize=12)
    ax4.set_ylabel('Material', fontsize=12)
    ax4.set_title(f'Time to Reach Steady State (L = {L*100:.0f} cm)', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='x')

    # Add formula
    ax4.text(0.95, 0.05, r'$\tau \sim \frac{L^2}{\alpha}$',
             transform=ax4.transAxes, fontsize=14, ha='right',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('Steady State vs Transient Heat Conduction\n'
                 r'Heat equation: $\frac{\partial T}{\partial t} = \alpha \frac{\partial^2 T}{\partial x^2}$',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'heat_conduction_comparison.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'heat_conduction_comparison.png')}")


if __name__ == "__main__":
    main()
