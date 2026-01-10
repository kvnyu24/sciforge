"""
Experiment 265: Hydrostatic Equilibrium Polytrope

Demonstrates stellar structure through the Lane-Emden equation
for polytropic stars in hydrostatic equilibrium.

Physical concepts:
- Pressure balances gravity: dP/dr = -G*M(r)*rho/r^2
- Polytropic equation of state: P = K * rho^(1+1/n)
- Lane-Emden equation: (1/xi^2) d/dxi(xi^2 dtheta/dxi) + theta^n = 0
- Analytical solutions exist for n = 0, 1, 5
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.plasma import LaneEmden, HydrostaticStar

# Physical constants
G = 6.674e-11
M_sun = 1.989e30
R_sun = 6.96e8


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Lane-Emden solutions for different polytropic indices
    ax1 = axes[0, 0]

    polytropic_indices = [0, 0.5, 1, 1.5, 2, 3, 4]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(polytropic_indices)))

    surface_data = []

    for n, color in zip(polytropic_indices, colors):
        le = LaneEmden(n)
        sol = le.solve()

        xi = sol['xi']
        theta = sol['theta']
        xi_1 = sol['xi_1']

        ax1.plot(xi / xi_1, theta, color=color, lw=2, label=f'n = {n}')
        surface_data.append((n, xi_1))

    # Analytical n=1 for comparison
    xi_test = np.linspace(0.01, np.pi, 100)
    theta_n1_analytical = np.sin(xi_test) / xi_test
    ax1.plot(xi_test / np.pi, theta_n1_analytical, 'k--', lw=1, alpha=0.5)

    ax1.set_xlabel('Normalized radius $\\xi / \\xi_1$')
    ax1.set_ylabel('$\\theta$ (normalized density$^{1/n}$)')
    ax1.set_title('Lane-Emden Solutions: $\\frac{1}{\\xi^2}\\frac{d}{d\\xi}(\\xi^2 \\frac{d\\theta}{d\\xi}) + \\theta^n = 0$')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1.1)
    ax1.set_ylim(-0.1, 1.1)

    # Plot 2: Density profiles
    ax2 = axes[0, 1]

    for n, color in zip([0, 1, 1.5, 3], colors[::2]):
        le = LaneEmden(n)
        sol = le.solve()

        xi = sol['xi']
        theta = sol['theta']
        xi_1 = sol['xi_1']

        # Density ratio rho/rho_c = theta^n
        if n == 0:
            rho_ratio = np.ones_like(theta)
        else:
            rho_ratio = np.where(theta > 0, theta**n, 0)

        ax2.plot(xi / xi_1, rho_ratio, color=color, lw=2, label=f'n = {n}')

    ax2.set_xlabel('Normalized radius $r / R$')
    ax2.set_ylabel('$\\rho / \\rho_c$')
    ax2.set_title('Density Profiles for Polytropic Stars')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.1)

    # Shade core region
    ax2.axvspan(0, 0.25, alpha=0.1, color='red')
    ax2.text(0.12, 0.9, 'Core', fontsize=10, ha='center')

    # Plot 3: Surface radius xi_1 vs polytropic index
    ax3 = axes[1, 0]

    n_range = np.linspace(0.1, 4.9, 50)
    xi_1_values = []
    m_factor_values = []

    for n in n_range:
        le = LaneEmden(n)
        sol = le.solve()
        xi_1_values.append(sol['xi_1'])
        m_factor_values.append(sol['minus_xi2_dtheta_1'])

    ax3.plot(n_range, xi_1_values, 'b-', lw=2, label='$\\xi_1$ (surface)')
    ax3.axhline(y=np.pi, color='gray', linestyle='--', alpha=0.7)
    ax3.text(1, np.pi + 0.2, '$\\pi$ (n=1)', fontsize=9)

    # Mark special cases
    special_n = [0, 1, 5]
    special_xi1 = [np.sqrt(6), np.pi, np.inf]  # n=5 extends to infinity

    for n, xi in zip(special_n[:2], special_xi1[:2]):
        le = LaneEmden(n)
        sol = le.solve()
        ax3.plot(n, sol['xi_1'], 'ro', markersize=10)

    ax3.set_xlabel('Polytropic index n')
    ax3.set_ylabel('$\\xi_1$ (dimensionless surface radius)')
    ax3.set_title('Surface Radius vs Polytropic Index')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 5)
    ax3.set_ylim(0, 20)

    # Note about n >= 5
    ax3.annotate('$n \\geq 5$: infinite extent', xy=(4.5, 15),
                 fontsize=10, ha='center')

    # Plot 4: Mass-radius relation for different n
    ax4 = axes[1, 1]

    # For a given central density, calculate mass and radius
    rho_c_range = np.logspace(2, 6, 50)  # kg/m^3

    # Polytropic constant (choose to give Solar-like values at some point)
    K_values = {
        1.5: 4.5e13,  # Non-relativistic degenerate (white dwarf)
        3: 3.8e14,    # Relativistic degenerate
    }

    for n, K in K_values.items():
        le = LaneEmden(n)

        masses = []
        radii = []

        for rho_c in rho_c_range:
            result = le.mass_radius_relation(K, rho_c)
            masses.append(result['M'] / M_sun)
            radii.append(result['R'] / R_sun)

        masses = np.array(masses)
        radii = np.array(radii)

        # Filter physical values
        valid = (masses > 0) & (radii > 0) & (masses < 100) & (radii < 100)
        ax4.loglog(radii[valid], masses[valid], lw=2, label=f'n = {n}')

    # Mark reference points
    ax4.plot(1, 1, 'y*', markersize=15, label='Sun')
    ax4.plot(0.01, 0.6, 'wo', markersize=10, markeredgecolor='black', label='White dwarf')

    # Chandrasekhar limit
    ax4.axhline(y=1.44, color='red', linestyle='--', alpha=0.7)
    ax4.text(0.002, 1.5, 'Chandrasekhar limit', fontsize=9, color='red')

    ax4.set_xlabel('Radius ($R_\\odot$)')
    ax4.set_ylabel('Mass ($M_\\odot$)')
    ax4.set_title('Mass-Radius Relations for Polytropes')
    ax4.legend(loc='lower left')
    ax4.grid(True, alpha=0.3, which='both')
    ax4.set_xlim(0.001, 10)
    ax4.set_ylim(0.1, 10)

    plt.suptitle('Experiment 265: Hydrostatic Equilibrium and Polytropes\n'
                 'Stellar structure from the Lane-Emden equation',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'hydrostatic_equilibrium_polytrope.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'hydrostatic_equilibrium_polytrope.png')}")


if __name__ == "__main__":
    main()
