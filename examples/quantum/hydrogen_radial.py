"""
Experiment 159: Hydrogen Radial Wavefunctions

This experiment demonstrates the radial wavefunctions of the hydrogen atom,
including:
- Radial probability distributions P(r) = r^2 |R_nl(r)|^2
- Number of nodes and their dependence on quantum numbers
- Most probable radii and expectation values
- Comparison with Bohr model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial, genlaguerre
from scipy.integrate import trapezoid


def radial_wavefunction(n: int, l: int, r: np.ndarray, a0: float = 1.0) -> np.ndarray:
    """
    Calculate hydrogen radial wavefunction R_nl(r).

    R_nl(r) = sqrt((2/(n*a0))^3 * (n-l-1)! / (2n*((n+l)!)^3)) *
              exp(-rho/2) * rho^l * L_{n-l-1}^{2l+1}(rho)

    where rho = 2r/(n*a0) and L is associated Laguerre polynomial.

    Args:
        n: Principal quantum number (n = 1, 2, 3, ...)
        l: Angular momentum quantum number (l = 0, 1, ..., n-1)
        r: Radial coordinate array
        a0: Bohr radius

    Returns:
        Radial wavefunction R_nl(r)
    """
    if l >= n or l < 0:
        raise ValueError(f"Invalid quantum numbers: n={n}, l={l}")

    rho = 2 * r / (n * a0)

    # Normalization factor
    norm = np.sqrt((2 / (n * a0))**3 * factorial(n - l - 1) /
                   (2 * n * factorial(n + l)**3))

    # Associated Laguerre polynomial L_{n-l-1}^{2l+1}(rho)
    L_poly = genlaguerre(n - l - 1, 2 * l + 1)

    R = norm * np.exp(-rho / 2) * rho**l * L_poly(rho)

    return R


def radial_probability(n: int, l: int, r: np.ndarray, a0: float = 1.0) -> np.ndarray:
    """
    Calculate radial probability density P(r) = r^2 |R_nl(r)|^2.

    This is the probability of finding the electron between r and r+dr.
    """
    R = radial_wavefunction(n, l, r, a0)
    return r**2 * np.abs(R)**2


def expectation_r(n: int, l: int, a0: float = 1.0) -> float:
    """
    Calculate <r> for hydrogen atom.

    <r>_nl = (a0/2) * (3n^2 - l(l+1))
    """
    return (a0 / 2) * (3 * n**2 - l * (l + 1))


def expectation_r2(n: int, l: int, a0: float = 1.0) -> float:
    """
    Calculate <r^2> for hydrogen atom.

    <r^2>_nl = (a0^2 * n^2 / 2) * (5n^2 + 1 - 3l(l+1))
    """
    return (a0**2 * n**2 / 2) * (5 * n**2 + 1 - 3 * l * (l + 1))


def most_probable_radius(n: int, l: int, a0: float = 1.0) -> float:
    """
    Calculate most probable radius (where P(r) is maximum).

    For l = n - 1 states: r_mp = n^2 * a0
    General case requires numerical solution.
    """
    if l == n - 1:
        return n**2 * a0
    else:
        # Numerical calculation
        r = np.linspace(0.01 * a0, (3 * n**2 + 5) * a0, 10000)
        P = radial_probability(n, l, r, a0)
        return r[np.argmax(P)]


def main():
    # Bohr radius (natural units)
    a0 = 1.0

    # Radial grid
    r_max = 30 * a0
    N = 1000
    r = np.linspace(0.01 * a0, r_max, N)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Plot 1: Radial wavefunctions for n = 1, 2, 3 (s-states)
    ax1 = axes[0, 0]

    n_values = [1, 2, 3]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(n_values)))

    for n, color in zip(n_values, colors):
        l = 0  # s-states
        R = radial_wavefunction(n, l, r, a0)
        ax1.plot(r / a0, R * a0**(3/2), color=color, lw=2, label=f'{n}s (n={n}, l={l})')

    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('r / a0')
    ax1.set_ylabel('R_nl(r) * a0^(3/2)')
    ax1.set_title('Radial Wavefunctions (s-states, l=0)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 20)

    # Plot 2: Radial probability densities
    ax2 = axes[0, 1]

    for n, color in zip(n_values, colors):
        l = 0
        P = radial_probability(n, l, r, a0)
        ax2.plot(r / a0, P * a0, color=color, lw=2, label=f'{n}s')
        ax2.fill_between(r / a0, 0, P * a0, color=color, alpha=0.2)

        # Mark most probable radius
        r_mp = most_probable_radius(n, l, a0)
        P_mp = radial_probability(n, l, np.array([r_mp]), a0)[0]
        ax2.axvline(x=r_mp / a0, color=color, linestyle=':', alpha=0.7)

    ax2.set_xlabel('r / a0')
    ax2.set_ylabel('P(r) = r^2 |R_nl|^2 * a0')
    ax2.set_title('Radial Probability Densities')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 20)

    # Plot 3: Different l values for n=3
    ax3 = axes[0, 2]

    n = 3
    l_values = [0, 1, 2]
    labels = ['3s', '3p', '3d']
    colors_l = ['blue', 'green', 'red']

    for l, label, color in zip(l_values, labels, colors_l):
        P = radial_probability(n, l, r, a0)
        ax3.plot(r / a0, P * a0, color=color, lw=2, label=label)

        # Mark expectation value <r>
        r_expect = expectation_r(n, l, a0)
        ax3.axvline(x=r_expect / a0, color=color, linestyle='--', alpha=0.5)

    ax3.set_xlabel('r / a0')
    ax3.set_ylabel('P(r) * a0')
    ax3.set_title('n=3 States with Different l')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 30)

    # Plot 4: Nodes in radial wavefunction
    ax4 = axes[1, 0]

    # n=4, l=0 has 3 radial nodes
    n_node = 4
    l_node = 0
    R = radial_wavefunction(n_node, l_node, r, a0)

    ax4.plot(r / a0, R * a0**(3/2), 'b-', lw=2)
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Find and mark nodes
    nodes = []
    for i in range(len(R) - 1):
        if R[i] * R[i+1] < 0:
            # Linear interpolation
            r_node = r[i] - R[i] * (r[i+1] - r[i]) / (R[i+1] - R[i])
            nodes.append(r_node)
            ax4.axvline(x=r_node / a0, color='red', linestyle=':', alpha=0.7)

    ax4.set_xlabel('r / a0')
    ax4.set_ylabel('R_nl(r) * a0^(3/2)')
    ax4.set_title(f'{n_node}s Wavefunction: {len(nodes)} radial nodes (n-l-1 = {n_node-l_node-1})')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 40)

    # Plot 5: Comparison with Bohr model
    ax5 = axes[1, 1]

    # Bohr radii: r_n = n^2 * a0
    n_max = 5

    bohr_radii = []
    most_probable_radii = []
    expectation_radii = []

    for n in range(1, n_max + 1):
        l = n - 1  # Maximum l for given n
        bohr_radii.append(n**2 * a0)
        most_probable_radii.append(most_probable_radius(n, l, a0))
        expectation_radii.append(expectation_r(n, l, a0))

    n_arr = np.arange(1, n_max + 1)
    ax5.plot(n_arr, np.array(bohr_radii) / a0, 'bo-', lw=2, markersize=10,
             label='Bohr radius n^2 * a0')
    ax5.plot(n_arr, np.array(most_probable_radii) / a0, 'rs-', lw=2, markersize=8,
             label='Most probable (circular orbit)')
    ax5.plot(n_arr, np.array(expectation_radii) / a0, 'g^-', lw=2, markersize=8,
             label='<r> (circular orbit)')

    ax5.set_xlabel('Principal quantum number n')
    ax5.set_ylabel('Radius / a0')
    ax5.set_title('Quantum vs Bohr Model (l = n-1 states)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Energy level diagram with wavefunctions
    ax6 = axes[1, 2]

    # Draw energy levels
    for n in range(1, 5):
        E_n = -1 / (2 * n**2)  # In Rydberg units
        ax6.hlines(E_n, 0.1, 0.9, colors='blue', lw=2)
        ax6.text(0.92, E_n, f'n={n}', va='center', fontsize=10)

        # Show degeneracy
        l_values_n = range(n)
        l_labels = 'spdfg'
        for l in l_values_n:
            x_pos = 0.2 + l * 0.15
            ax6.text(x_pos, E_n + 0.02, f'{n}{l_labels[l]}', fontsize=8,
                    ha='center', alpha=0.7)

    ax6.set_xlim(0, 1)
    ax6.set_ylim(-0.6, 0.05)
    ax6.set_ylabel('Energy (Rydbergs)')
    ax6.set_title('Hydrogen Energy Levels')
    ax6.set_xticks([])

    # Add continuum
    ax6.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax6.text(0.5, 0.02, 'Ionization (E = 0)', ha='center', fontsize=9, color='red')

    # Add formula
    ax6.text(0.5, -0.55, r'$E_n = -\frac{13.6 \, \mathrm{eV}}{n^2}$',
             ha='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Hydrogen Atom Radial Wavefunctions\n'
                 r'$R_{nl}(r) = N_{nl} \cdot e^{-\rho/2} \cdot \rho^l \cdot L_{n-l-1}^{2l+1}(\rho)$, '
                 r'$\rho = 2r/(na_0)$',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'hydrogen_radial.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'hydrogen_radial.png')}")


if __name__ == "__main__":
    main()
