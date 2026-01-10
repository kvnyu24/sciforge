"""
Experiment 150: Finite Square Well Bound States

This experiment demonstrates bound state solutions in a finite square well potential,
including:
- Transcendental equation solutions for bound state energies
- Wavefunction penetration into classically forbidden regions
- Comparison with infinite well as depth increases
- Dependence of number of bound states on well parameters
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.integrate import trapezoid


def find_bound_state_energies(V0: float, a: float, m: float = 1.0, hbar: float = 1.0,
                               n_max: int = 50) -> tuple:
    """
    Find bound state energies for finite square well.

    The well is defined as V(x) = -V0 for |x| < a, V(x) = 0 otherwise.

    Args:
        V0: Well depth (positive)
        a: Half-width of well
        m: Particle mass
        hbar: Reduced Planck constant
        n_max: Maximum number of states to find

    Returns:
        Tuple of (even_energies, odd_energies) arrays
    """
    # Dimensionless parameter
    z0 = a * np.sqrt(2 * m * V0) / hbar

    def even_equation(z):
        """Transcendental equation for even states: z * tan(z) = sqrt(z0^2 - z^2)"""
        if z >= z0:
            return np.inf
        return z * np.tan(z) - np.sqrt(z0**2 - z**2)

    def odd_equation(z):
        """Transcendental equation for odd states: -z * cot(z) = sqrt(z0^2 - z^2)"""
        if z >= z0:
            return np.inf
        return -z / np.tan(z) - np.sqrt(z0**2 - z**2)

    even_energies = []
    odd_energies = []

    # Search for even states (in intervals (n*pi, (n+0.5)*pi) approximately)
    for n in range(n_max):
        z_low = n * np.pi + 0.001
        z_high = (n + 0.5) * np.pi - 0.001

        if z_low >= z0:
            break

        z_high = min(z_high, z0 - 0.001)

        try:
            if even_equation(z_low) * even_equation(z_high) < 0:
                z_root = brentq(even_equation, z_low, z_high)
                E = -V0 + (hbar**2 * z_root**2) / (2 * m * a**2)
                even_energies.append(E)
        except ValueError:
            pass

    # Search for odd states (in intervals ((n+0.5)*pi, (n+1)*pi) approximately)
    for n in range(n_max):
        z_low = (n + 0.5) * np.pi + 0.001
        z_high = (n + 1) * np.pi - 0.001

        if z_low >= z0:
            break

        z_high = min(z_high, z0 - 0.001)

        try:
            if odd_equation(z_low) * odd_equation(z_high) < 0:
                z_root = brentq(odd_equation, z_low, z_high)
                E = -V0 + (hbar**2 * z_root**2) / (2 * m * a**2)
                odd_energies.append(E)
        except ValueError:
            pass

    return np.array(even_energies), np.array(odd_energies)


def bound_state_wavefunction(x: np.ndarray, E: float, V0: float, a: float,
                              parity: str = 'even', m: float = 1.0,
                              hbar: float = 1.0) -> np.ndarray:
    """
    Calculate wavefunction for a bound state in finite square well.

    Args:
        x: Position array
        E: Bound state energy (negative)
        V0: Well depth (positive)
        a: Half-width of well
        parity: 'even' or 'odd'
        m: Particle mass
        hbar: Reduced Planck constant

    Returns:
        Normalized wavefunction
    """
    # Wave numbers
    k = np.sqrt(2 * m * (E + V0)) / hbar  # Inside well
    kappa = np.sqrt(-2 * m * E) / hbar    # Outside well (E < 0)

    psi = np.zeros_like(x)

    if parity == 'even':
        # Inside: cos(kx)
        # Outside: A * exp(-kappa|x|)
        A = np.cos(k * a) * np.exp(kappa * a)

        inside = np.abs(x) <= a
        psi[inside] = np.cos(k * x[inside])
        psi[~inside] = A * np.exp(-kappa * np.abs(x[~inside]))
    else:
        # Inside: sin(kx)
        # Outside: A * sign(x) * exp(-kappa|x|)
        A = np.sin(k * a) * np.exp(kappa * a)

        inside = np.abs(x) <= a
        psi[inside] = np.sin(k * x[inside])
        psi[~inside] = A * np.sign(x[~inside]) * np.exp(-kappa * np.abs(x[~inside]))

    # Normalize
    norm = np.sqrt(trapezoid(psi**2, x))
    psi /= norm

    return psi


def main():
    # Parameters (natural units)
    m = 1.0
    hbar = 1.0
    a = 1.0  # Half-width

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Plot 1: Graphical solution of transcendental equation
    ax1 = axes[0, 0]

    V0 = 10.0
    z0 = a * np.sqrt(2 * m * V0) / hbar

    z = np.linspace(0.01, z0 - 0.01, 1000)

    # Left-hand side for even states: z * tan(z)
    # Right-hand side: sqrt(z0^2 - z^2)
    lhs_even = z * np.tan(z)
    lhs_odd = -z / np.tan(z)
    rhs = np.sqrt(z0**2 - z**2)

    # Handle discontinuities
    lhs_even[lhs_even < 0] = np.nan
    lhs_even[lhs_even > 20] = np.nan
    lhs_odd[lhs_odd < 0] = np.nan
    lhs_odd[lhs_odd > 20] = np.nan

    ax1.plot(z, rhs, 'k-', lw=2, label=r'$\sqrt{z_0^2 - z^2}$')
    ax1.plot(z, lhs_even, 'b-', lw=2, label=r'$z \tan(z)$ (even)')
    ax1.plot(z, lhs_odd, 'r--', lw=2, label=r'$-z \cot(z)$ (odd)')

    # Mark intersections (bound states)
    even_E, odd_E = find_bound_state_energies(V0, a, m, hbar)

    for E in even_E:
        z_val = a * np.sqrt(2 * m * (E + V0)) / hbar
        ax1.plot(z_val, np.sqrt(z0**2 - z_val**2), 'bo', markersize=10)

    for E in odd_E:
        z_val = a * np.sqrt(2 * m * (E + V0)) / hbar
        ax1.plot(z_val, np.sqrt(z0**2 - z_val**2), 'rs', markersize=10)

    ax1.set_xlabel('z = ka')
    ax1.set_ylabel('Function value')
    ax1.set_title(f'Graphical Solution (V0 = {V0}, z0 = {z0:.2f})')
    ax1.legend()
    ax1.set_xlim(0, z0)
    ax1.set_ylim(0, z0)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Bound state wavefunctions
    ax2 = axes[0, 1]

    x = np.linspace(-4*a, 4*a, 1000)

    # Draw potential well
    V = np.zeros_like(x)
    V[np.abs(x) > a] = 0
    V[np.abs(x) <= a] = -V0
    ax2.fill_between(x, V, 0, alpha=0.2, color='gray', label='Potential')
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    # Combine and sort energies
    all_E = list(zip(even_E, ['even']*len(even_E))) + list(zip(odd_E, ['odd']*len(odd_E)))
    all_E.sort(key=lambda x: x[0])

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(all_E)))

    for i, (E, parity) in enumerate(all_E):
        psi = bound_state_wavefunction(x, E, V0, a, parity, m, hbar)
        # Scale for visualization
        scale = 3.0
        ax2.plot(x, scale * psi + E, color=colors[i], lw=2,
                label=f'n={i+1}, E={E:.2f}')
        ax2.axhline(y=E, color=colors[i], linestyle=':', alpha=0.5)

    ax2.axvline(x=-a, color='black', linestyle='--', alpha=0.5)
    ax2.axvline(x=a, color='black', linestyle='--', alpha=0.5)

    ax2.set_xlabel('Position x/a')
    ax2.set_ylabel('Energy / Wavefunction')
    ax2.set_title('Bound State Wavefunctions')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_xlim(-4*a, 4*a)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Probability density showing penetration
    ax3 = axes[0, 2]

    # Focus on ground state penetration
    E_ground = all_E[0][0]
    parity_ground = all_E[0][1]
    psi_ground = bound_state_wavefunction(x, E_ground, V0, a, parity_ground, m, hbar)
    prob = psi_ground**2

    ax3.fill_between(x, prob, alpha=0.5, color='blue', label='|psi|^2')
    ax3.plot(x, prob, 'b-', lw=2)

    # Highlight classically forbidden region
    ax3.axvspan(-4*a, -a, alpha=0.2, color='red', label='Classically forbidden')
    ax3.axvspan(a, 4*a, alpha=0.2, color='red')

    # Calculate penetration probability
    forbidden_prob = trapezoid(prob[np.abs(x) > a], x[np.abs(x) > a])
    ax3.text(0, max(prob)*0.8, f'Penetration probability: {forbidden_prob:.1%}',
             ha='center', fontsize=10)

    # Penetration depth
    kappa = np.sqrt(-2 * m * E_ground) / hbar
    penetration_depth = 1 / kappa
    ax3.axvline(x=a + penetration_depth, color='green', linestyle='--', alpha=0.7)
    ax3.axvline(x=-a - penetration_depth, color='green', linestyle='--', alpha=0.7,
               label=f'Penetration depth = {penetration_depth:.2f}')

    ax3.set_xlabel('Position x/a')
    ax3.set_ylabel('Probability Density')
    ax3.set_title('Ground State Penetration into Forbidden Region')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.set_xlim(-4*a, 4*a)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Number of bound states vs well depth
    ax4 = axes[1, 0]

    V0_range = np.linspace(0.5, 50, 100)
    n_states = []

    for V0_test in V0_range:
        even_E, odd_E = find_bound_state_energies(V0_test, a, m, hbar)
        n_states.append(len(even_E) + len(odd_E))

    ax4.plot(V0_range, n_states, 'b-', lw=2)
    ax4.set_xlabel('Well Depth V0')
    ax4.set_ylabel('Number of Bound States')
    ax4.set_title('Number of Bound States vs Well Depth')
    ax4.grid(True, alpha=0.3)

    # Add theoretical prediction: N ~ z0/pi + 1
    z0_range = a * np.sqrt(2 * m * V0_range) / hbar
    n_theory = np.floor(z0_range / (np.pi/2) + 1)
    ax4.plot(V0_range, n_theory, 'r--', lw=2, label='Theory: floor(2z0/pi + 1)')
    ax4.legend()

    # Plot 5: Comparison with infinite well
    ax5 = axes[1, 1]

    # Compare energies for different well depths
    well_depths = [5, 10, 20, 50, 100]
    colors_well = plt.cm.plasma(np.linspace(0.2, 0.9, len(well_depths)))

    for V0_test, color in zip(well_depths, colors_well):
        even_E, odd_E = find_bound_state_energies(V0_test, a, m, hbar)
        all_E_test = sorted(list(even_E) + list(odd_E))

        # Infinite well energies (shifted so ground state at -V0)
        n_arr = np.arange(1, len(all_E_test) + 1)
        E_infinite = n_arr**2 * np.pi**2 * hbar**2 / (2 * m * (2*a)**2) - V0_test

        ax5.scatter(n_arr, all_E_test, color=color, s=50, label=f'V0={V0_test}')
        ax5.plot(n_arr, E_infinite, color=color, linestyle='--', alpha=0.5)

    ax5.set_xlabel('State number n')
    ax5.set_ylabel('Energy E')
    ax5.set_title('Finite Well (dots) vs Infinite Well (dashed)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # Plot 6: Wave function decay length
    ax6 = axes[1, 2]

    V0 = 20.0
    even_E, odd_E = find_bound_state_energies(V0, a, m, hbar)
    all_E_decay = sorted(list(even_E) + list(odd_E))

    decay_lengths = []
    for E in all_E_decay:
        kappa = np.sqrt(-2 * m * E) / hbar
        decay_lengths.append(1 / kappa)

    ax6.bar(range(1, len(decay_lengths) + 1), decay_lengths, color='steelblue', alpha=0.7)
    ax6.set_xlabel('State number n')
    ax6.set_ylabel('Decay length 1/kappa')
    ax6.set_title(f'Evanescent Wave Decay Length (V0 = {V0})')
    ax6.grid(True, alpha=0.3, axis='y')

    # Show that deeper bound states have shorter decay length
    ax6_twin = ax6.twinx()
    ax6_twin.plot(range(1, len(all_E_decay) + 1), all_E_decay, 'ro-', lw=2)
    ax6_twin.set_ylabel('Energy E', color='red')
    ax6_twin.tick_params(axis='y', labelcolor='red')

    plt.suptitle('Finite Square Well: Bound State Analysis\n'
                 'V(x) = -V0 for |x| < a, V(x) = 0 otherwise',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'finite_square_well.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'finite_square_well.png')}")


if __name__ == "__main__":
    main()
