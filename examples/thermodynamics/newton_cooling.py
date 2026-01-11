"""
Experiment 131: Newton's Law of Cooling

Compares lumped-system (Newton's law) cooling with full PDE heat diffusion.

Physical concepts:
- Newton's law: dT/dt = -h*A*(T - T_env) / (m*c) = -(T - T_env)/tau
- Solution: T(t) = T_env + (T_0 - T_env)*exp(-t/tau)
- Time constant: tau = m*c/(h*A)
- Valid when Bi = h*L/k << 1 (Biot number small)

Comparison with full heat equation shows when lumped model breaks down.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt


def newton_cooling_analytic(T0, T_env, tau, t):
    """
    Newton's law of cooling analytical solution.

    T(t) = T_env + (T_0 - T_env) * exp(-t/tau)
    """
    return T_env + (T0 - T_env) * np.exp(-t / tau)


def cooling_time_constant(mass, specific_heat, h, area):
    """
    Calculate cooling time constant.

    tau = m*c / (h*A)

    Args:
        mass: Object mass (kg)
        specific_heat: Specific heat capacity (J/kg/K)
        h: Heat transfer coefficient (W/m^2/K)
        area: Surface area (m^2)
    """
    return mass * specific_heat / (h * area)


def biot_number(h, L, k):
    """
    Calculate Biot number.

    Bi = h*L/k

    Args:
        h: Heat transfer coefficient (W/m^2/K)
        L: Characteristic length (m)
        k: Thermal conductivity (W/m/K)

    Bi << 0.1: Lumped system valid
    Bi > 0.1: Need to consider internal temperature gradients
    """
    return h * L / k


def solve_heat_equation_1d(T0, T_env, L, k, rho, cp, h, t_max, nx=50, dt=None):
    """
    Solve 1D heat equation with convective boundary condition.

    dT/dt = alpha * d^2T/dx^2

    Boundary: -k*dT/dx = h*(T - T_env) at surface
    """
    alpha = k / (rho * cp)  # Thermal diffusivity

    dx = L / (nx - 1)
    if dt is None:
        dt = 0.4 * dx**2 / alpha  # Stability condition

    x = np.linspace(0, L, nx)
    nt = int(t_max / dt)

    T = np.ones(nx) * T0
    T_history = [T.copy()]
    times = [0]

    # Time stepping (explicit)
    for n in range(nt):
        T_new = T.copy()

        # Interior points
        for i in range(1, nx-1):
            T_new[i] = T[i] + alpha * dt / dx**2 * (T[i+1] - 2*T[i] + T[i-1])

        # Boundary conditions
        # x=0: insulated (symmetry for sphere center)
        T_new[0] = T_new[1]

        # x=L: convective BC
        # -k*(T[nx-1] - T[nx-2])/dx = h*(T[nx-1] - T_env)
        T_new[nx-1] = (T_new[nx-2] + h*dx/k * T_env) / (1 + h*dx/k)

        T = T_new

        # Store at intervals
        if n % max(1, nt // 100) == 0:
            T_history.append(T.copy())
            times.append((n+1) * dt)

    return np.array(times), x, np.array(T_history)


def average_temperature(T_profile):
    """Calculate volume-averaged temperature (assuming uniform cross-section)."""
    return np.mean(T_profile)


def main():
    """Run Newton's cooling experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Parameters for a copper sphere
    T0 = 100.0  # Initial temperature (°C)
    T_env = 20.0  # Environment temperature (°C)

    # Material properties (copper)
    k = 400  # Thermal conductivity (W/m/K)
    rho = 8960  # Density (kg/m^3)
    cp = 385  # Specific heat (J/kg/K)

    # Geometry (sphere)
    radius = 0.01  # 1 cm radius
    L = radius  # Characteristic length for sphere = R/3, but use R for simplicity
    volume = (4/3) * np.pi * radius**3
    area = 4 * np.pi * radius**2
    mass = rho * volume

    # Heat transfer coefficients (different cooling scenarios)
    h_values = {
        'Natural convection (air)': 10,
        'Forced convection (air)': 50,
        'Natural convection (water)': 500,
        'Forced convection (water)': 3000,
    }

    t_max = 600  # seconds

    # Plot 1: Newton's cooling for different h values
    ax1 = axes[0, 0]

    t = np.linspace(0, t_max, 500)
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(h_values)))

    for (name, h), color in zip(h_values.items(), colors):
        tau = cooling_time_constant(mass, cp, h, area)
        T = newton_cooling_analytic(T0, T_env, tau, t)
        Bi = biot_number(h, L, k)

        ax1.plot(t, T, color=color, lw=2,
                label=f'{name}\nτ={tau:.1f}s, Bi={Bi:.4f}')

    ax1.axhline(y=T_env, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('Newton\'s Law of Cooling - Different Conditions')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Biot number regime map
    ax2 = axes[0, 1]

    h_range = np.logspace(0, 4, 100)
    L_range = np.logspace(-3, 0, 100)
    H, LL = np.meshgrid(h_range, L_range)

    # Different materials
    materials = {
        'Copper (k=400)': 400,
        'Steel (k=50)': 50,
        'Glass (k=1)': 1,
        'Wood (k=0.1)': 0.1,
    }

    for name, k_mat in materials.items():
        Bi_line = 0.1 * k_mat / L_range  # h where Bi = 0.1
        valid = (Bi_line >= h_range.min()) & (Bi_line <= h_range.max())
        ax2.loglog(L_range[valid] * 100, Bi_line[valid], lw=2, label=name)

    ax2.fill_between([0.1, 100], [1, 1], [1e4, 1e4], alpha=0.2, color='red',
                     label='Bi > 0.1 (PDE needed)')
    ax2.fill_between([0.1, 100], [0.01, 0.01], [1, 1], alpha=0.2, color='green',
                     label='Bi < 0.1 (lumped OK)')

    ax2.set_xlabel('Characteristic Length (cm)')
    ax2.set_ylabel('Heat Transfer Coefficient h (W/m²K)')
    ax2.set_title('Biot Number Regime Map')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_xlim(0.1, 100)
    ax2.set_ylim(1, 1e4)

    # Plot 3: Comparison - lumped vs PDE for low Bi
    ax3 = axes[1, 0]

    h_low = 10  # Low h -> low Bi
    tau_low = cooling_time_constant(mass, cp, h_low, area)
    Bi_low = biot_number(h_low, L, k)

    # Newton's law solution
    T_newton = newton_cooling_analytic(T0, T_env, tau_low, t)

    # PDE solution
    times_pde, x, T_history = solve_heat_equation_1d(
        T0, T_env, L, k, rho, cp, h_low, t_max, nx=30
    )
    T_avg_pde = [average_temperature(T_prof) for T_prof in T_history]

    ax3.plot(t, T_newton, 'b-', lw=2, label='Newton\'s law (lumped)')
    ax3.plot(times_pde, T_avg_pde, 'r--', lw=2, label='PDE (avg temperature)')

    # Also show surface vs center from PDE
    T_surface = [T_prof[-1] for T_prof in T_history]
    T_center = [T_prof[0] for T_prof in T_history]
    ax3.plot(times_pde, T_surface, 'g:', lw=1.5, alpha=0.7, label='PDE surface')
    ax3.plot(times_pde, T_center, 'm:', lw=1.5, alpha=0.7, label='PDE center')

    ax3.axhline(y=T_env, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Temperature (°C)')
    ax3.set_title(f'Low Biot Number (Bi = {Bi_low:.4f})\nLumped model valid')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Comparison - lumped vs PDE for high Bi
    ax4 = axes[1, 1]

    # Use a less conductive material for high Bi
    k_glass = 1.0
    rho_glass = 2500
    cp_glass = 840
    mass_glass = rho_glass * volume

    h_high = 1000  # High h for water
    tau_high = cooling_time_constant(mass_glass, cp_glass, h_high, area)
    Bi_high = biot_number(h_high, L, k_glass)

    # Newton's law solution
    T_newton_high = newton_cooling_analytic(T0, T_env, tau_high, t)

    # PDE solution
    times_pde_high, x_high, T_history_high = solve_heat_equation_1d(
        T0, T_env, L, k_glass, rho_glass, cp_glass, h_high, t_max, nx=30
    )
    T_avg_pde_high = [average_temperature(T_prof) for T_prof in T_history_high]

    ax4.plot(t, T_newton_high, 'b-', lw=2, label='Newton\'s law (lumped)')
    ax4.plot(times_pde_high, T_avg_pde_high, 'r--', lw=2, label='PDE (avg temperature)')

    # Surface vs center
    T_surface_high = [T_prof[-1] for T_prof in T_history_high]
    T_center_high = [T_prof[0] for T_prof in T_history_high]
    ax4.plot(times_pde_high, T_surface_high, 'g:', lw=1.5, alpha=0.7, label='PDE surface')
    ax4.plot(times_pde_high, T_center_high, 'm:', lw=1.5, alpha=0.7, label='PDE center')

    ax4.axhline(y=T_env, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Temperature (°C)')
    ax4.set_title(f'High Biot Number (Bi = {Bi_high:.1f})\nLumped model breaks down')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Highlight error
    error_region = (t < 200)
    ax4.fill_between(t[error_region],
                     T_newton_high[error_region],
                     np.interp(t[error_region], times_pde_high, T_avg_pde_high),
                     alpha=0.3, color='red', label='Error region')

    plt.suptitle('Experiment 131: Newton\'s Law of Cooling\n'
                 '$\\frac{dT}{dt} = -\\frac{T - T_{env}}{\\tau}$, valid when Bi $\\ll$ 0.1',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save output
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'newton_cooling.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'newton_cooling.png')}")

    # Print summary
    print("\n=== Newton's Cooling Summary ===")
    print(f"\nCopper sphere (r = {radius*100} cm):")
    for name, h in h_values.items():
        tau = cooling_time_constant(mass, cp, h, area)
        Bi = biot_number(h, L, k)
        print(f"  {name}: τ = {tau:.1f} s, Bi = {Bi:.5f}")

    print(f"\nBiot number criterion: Bi < 0.1 for lumped model")
    print(f"  Bi = h*L/k (ratio of convection to conduction)")


if __name__ == "__main__":
    main()
