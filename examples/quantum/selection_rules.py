"""
Experiment 160: Selection Rules and Dipole Matrix Elements

This experiment demonstrates electric dipole selection rules in atomic physics,
including:
- Calculation of dipole matrix elements <nlm|r|n'l'm'>
- Angular momentum selection rules (Delta l = +/- 1, Delta m = 0, +/- 1)
- Radial integrals for hydrogen
- Transition rates and oscillator strengths
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial, genlaguerre, sph_harm
from scipy.integrate import trapezoid, dblquad


def radial_wavefunction(n: int, l: int, r: np.ndarray, a0: float = 1.0) -> np.ndarray:
    """Calculate hydrogen radial wavefunction R_nl(r)."""
    if l >= n or l < 0:
        return np.zeros_like(r)

    rho = 2 * r / (n * a0)

    norm = np.sqrt((2 / (n * a0))**3 * factorial(n - l - 1) /
                   (2 * n * factorial(n + l)**3))

    L_poly = genlaguerre(n - l - 1, 2 * l + 1)

    R = norm * np.exp(-rho / 2) * rho**l * L_poly(rho)

    return R


def radial_integral(n1: int, l1: int, n2: int, l2: int, a0: float = 1.0) -> float:
    """
    Calculate radial dipole integral <n1 l1|r|n2 l2>.

    Returns integral_0^infinity R_n1l1(r) * r * R_n2l2(r) * r^2 dr
    = integral_0^infinity R_n1l1(r) * r^3 * R_n2l2(r) dr
    """
    r_max = max(10 * max(n1, n2)**2 * a0, 100 * a0)
    r = np.linspace(0.001 * a0, r_max, 5000)

    R1 = radial_wavefunction(n1, l1, r, a0)
    R2 = radial_wavefunction(n2, l2, r, a0)

    integrand = R1 * r**3 * R2

    return trapezoid(integrand, r)


def angular_selection_rule(l1: int, m1: int, l2: int, m2: int, q: int) -> float:
    """
    Calculate angular matrix element for dipole transition.

    For electric dipole transitions:
    <l1, m1 | Y_1^q | l2, m2> where q = -1, 0, +1

    Selection rules:
    - Delta l = +/- 1
    - Delta m = q = -1, 0, +1

    Returns the angular factor (Clebsch-Gordan coefficient factor).
    """
    # Check selection rules
    if l1 != l2 + 1 and l1 != l2 - 1:
        return 0.0

    if m1 != m2 + q:
        return 0.0

    # Use 3j symbol relationship
    # <l1 m1 | Y_1^q | l2 m2> propto C(l2, 1, l1; m2, q, m1) * C(l2, 1, l1; 0, 0, 0)

    # Simplified formula for the relevant cases
    # For l1 = l2 + 1:
    if l1 == l2 + 1:
        l = l2
        if q == 0:
            return np.sqrt((l + 1 - m1) * (l + 1 + m1) / ((2*l + 1) * (2*l + 3)))
        elif q == 1:
            return -np.sqrt((l + 1 + m1) * (l + 2 + m1) / (2 * (2*l + 1) * (2*l + 3)))
        elif q == -1:
            return np.sqrt((l + 1 - m1) * (l + 2 - m1) / (2 * (2*l + 1) * (2*l + 3)))

    # For l1 = l2 - 1:
    elif l1 == l2 - 1:
        l = l2
        if q == 0:
            return np.sqrt((l - m1) * (l + m1) / ((2*l - 1) * (2*l + 1)))
        elif q == 1:
            return np.sqrt((l - m1) * (l - 1 - m1) / (2 * (2*l - 1) * (2*l + 1)))
        elif q == -1:
            return -np.sqrt((l + m1) * (l - 1 + m1) / (2 * (2*l - 1) * (2*l + 1)))

    return 0.0


def dipole_matrix_element(n1: int, l1: int, m1: int,
                           n2: int, l2: int, m2: int,
                           direction: str = 'z', a0: float = 1.0) -> float:
    """
    Calculate full dipole matrix element <n1 l1 m1|r_i|n2 l2 m2>.

    Args:
        n1, l1, m1: Final state quantum numbers
        n2, l2, m2: Initial state quantum numbers
        direction: 'x', 'y', or 'z'
        a0: Bohr radius

    Returns:
        Dipole matrix element in units of a0
    """
    # Get radial integral
    R = radial_integral(n1, l1, n2, l2, a0)

    # Get angular factor based on direction
    if direction == 'z':
        # z propto Y_1^0
        q = 0
        angular = angular_selection_rule(l1, m1, l2, m2, q)
        factor = np.sqrt(4 * np.pi / 3)
    elif direction == 'x':
        # x propto (Y_1^{-1} - Y_1^1)
        angular_m = angular_selection_rule(l1, m1, l2, m2, -1)
        angular_p = angular_selection_rule(l1, m1, l2, m2, 1)
        angular = (angular_m - angular_p) / np.sqrt(2)
        factor = np.sqrt(4 * np.pi / 3)
    elif direction == 'y':
        # y propto i(Y_1^{-1} + Y_1^1)
        angular_m = angular_selection_rule(l1, m1, l2, m2, -1)
        angular_p = angular_selection_rule(l1, m1, l2, m2, 1)
        angular = (angular_m + angular_p) / np.sqrt(2)
        factor = np.sqrt(4 * np.pi / 3)
    else:
        return 0.0

    return factor * R * angular


def transition_rate(n1: int, l1: int, n2: int, l2: int, a0: float = 1.0) -> float:
    """
    Calculate relative transition rate (summed over m values).

    Rate propto |<n1 l1 || r || n2 l2>|^2 * (2*l< + 1)

    Returns value proportional to Einstein A coefficient.
    """
    # Radial integral
    R = radial_integral(n1, l1, n2, l2, a0)

    # Angular factor (reduced matrix element squared)
    l_min = min(l1, l2)

    return R**2 * (2 * l_min + 1)


def main():
    a0 = 1.0  # Bohr radius

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Plot 1: Selection rule visualization
    ax1 = axes[0, 0]

    # Create grid showing allowed/forbidden transitions
    l_max = 5
    delta_l_range = range(-3, 4)

    allowed = np.zeros((len(delta_l_range), l_max))

    for i, delta_l in enumerate(delta_l_range):
        for l in range(l_max):
            l_final = l + delta_l
            if l_final >= 0:
                allowed[i, l] = 1 if abs(delta_l) == 1 else 0

    im = ax1.imshow(allowed, cmap='RdYlGn', aspect='auto',
                    extent=[-0.5, l_max-0.5, min(delta_l_range)-0.5, max(delta_l_range)+0.5])

    ax1.set_xlabel('Initial l')
    ax1.set_ylabel('Delta l')
    ax1.set_title('Selection Rule: Delta l = +/- 1')
    ax1.set_xticks(range(l_max))
    ax1.set_yticks(range(min(delta_l_range), max(delta_l_range)+1))

    # Add text annotations
    for i, delta_l in enumerate(delta_l_range):
        for l in range(l_max):
            status = 'A' if abs(delta_l) == 1 and l + delta_l >= 0 else 'F'
            ax1.text(l, delta_l, status, ha='center', va='center',
                    color='white' if status == 'A' else 'black', fontsize=10)

    # Plot 2: Radial integrals for specific transitions
    ax2 = axes[0, 1]

    transitions = [
        (2, 1, 1, 0, '2p -> 1s'),
        (3, 1, 1, 0, '3p -> 1s'),
        (3, 1, 2, 0, '3p -> 2s'),
        (3, 0, 2, 1, '3s -> 2p'),
        (3, 2, 2, 1, '3d -> 2p'),
        (4, 1, 1, 0, '4p -> 1s'),
    ]

    radial_values = []
    labels = []

    for n1, l1, n2, l2, label in transitions:
        R = radial_integral(n1, l1, n2, l2, a0)
        radial_values.append(abs(R))
        labels.append(label)

    ax2.barh(range(len(labels)), radial_values, color='steelblue', alpha=0.7)
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels)
    ax2.set_xlabel('|Radial Integral| (a0)')
    ax2.set_title('Radial Dipole Matrix Elements')
    ax2.grid(True, alpha=0.3, axis='x')

    # Plot 3: m selection rules
    ax3 = axes[0, 2]

    # Show allowed m transitions for l=1 -> l=0
    l_initial = 1
    l_final = 0

    m_matrix = np.zeros((3, 2 * l_initial + 1))
    m_initial_vals = range(-l_initial, l_initial + 1)

    for i, q in enumerate([-1, 0, 1]):  # polarization
        for j, m_i in enumerate(m_initial_vals):
            m_f = m_i + q
            if abs(m_f) <= l_final:
                m_matrix[i, j] = 1

    im3 = ax3.imshow(m_matrix, cmap='Blues', aspect='auto',
                     extent=[-l_initial-0.5, l_initial+0.5, -0.5, 2.5])

    ax3.set_xlabel('Initial m')
    ax3.set_ylabel('Polarization q')
    ax3.set_title(f'm Selection Rules ({l_initial}p -> {l_final}s)')
    ax3.set_xticks(range(-l_initial, l_initial + 1))
    ax3.set_yticks([0, 1, 2])
    ax3.set_yticklabels([r'$\sigma^-$ (q=-1)', r'$\pi$ (q=0)', r'$\sigma^+$ (q=+1)'])

    # Plot 4: Transition rates (oscillator strengths)
    ax4 = axes[1, 0]

    # Lyman series (n -> 1)
    n_values = range(2, 8)
    lyman_rates = []

    for n in n_values:
        rate = transition_rate(n, 1, 1, 0, a0)  # np -> 1s
        lyman_rates.append(rate)

    # Normalize
    lyman_rates = np.array(lyman_rates) / max(lyman_rates)

    ax4.bar([f'{n}p->1s' for n in n_values], lyman_rates, color='coral', alpha=0.7)
    ax4.set_ylabel('Relative Transition Rate')
    ax4.set_title('Lyman Series Transition Rates')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3, axis='y')

    # Plot 5: Dipole matrix element direction dependence
    ax5 = axes[1, 1]

    # 2p -> 1s transition with different m values
    directions = ['x', 'y', 'z']
    m_values = [-1, 0, 1]

    matrix_dir = np.zeros((3, 3))

    for i, direction in enumerate(directions):
        for j, m in enumerate(m_values):
            d = dipole_matrix_element(1, 0, 0, 2, 1, m, direction, a0)
            matrix_dir[i, j] = abs(d)

    im5 = ax5.imshow(matrix_dir, cmap='YlOrRd', aspect='auto')
    plt.colorbar(im5, ax=ax5, label='|<1s|r_i|2p,m>| (a0)')

    ax5.set_xticks([0, 1, 2])
    ax5.set_xticklabels(['m=-1', 'm=0', 'm=+1'])
    ax5.set_yticks([0, 1, 2])
    ax5.set_yticklabels(['x', 'y', 'z'])
    ax5.set_xlabel('Initial m (2p)')
    ax5.set_ylabel('Polarization Direction')
    ax5.set_title('Dipole Matrix Elements: 2p -> 1s')

    # Annotate values
    for i in range(3):
        for j in range(3):
            val = matrix_dir[i, j]
            ax5.text(j, i, f'{val:.3f}', ha='center', va='center',
                    color='white' if val > 0.5 else 'black')

    # Plot 6: Forbidden vs allowed transition illustration
    ax6 = axes[1, 2]

    r = np.linspace(0.01 * a0, 30 * a0, 1000)

    # 2s and 1s overlap (forbidden: Delta l = 0)
    R_2s = radial_wavefunction(2, 0, r, a0)
    R_1s = radial_wavefunction(1, 0, r, a0)
    integrand_forbidden = R_2s * r**3 * R_1s

    # 2p and 1s overlap (allowed: Delta l = 1)
    R_2p = radial_wavefunction(2, 1, r, a0)
    integrand_allowed = R_2p * r**3 * R_1s

    ax6.plot(r / a0, integrand_forbidden * a0**2, 'r-', lw=2,
             label='2s-1s (forbidden)')
    ax6.plot(r / a0, integrand_allowed * a0**2, 'b-', lw=2,
             label='2p-1s (allowed)')
    ax6.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    ax6.set_xlabel('r / a0')
    ax6.set_ylabel('R_1 * r^3 * R_2 * a0^2')
    ax6.set_title('Radial Integrands: Allowed vs Forbidden')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(0, 15)

    # Add integrals
    int_forbidden = trapezoid(integrand_forbidden, r)
    int_allowed = trapezoid(integrand_allowed, r)
    ax6.text(0.95, 0.95, f'2s-1s integral: {int_forbidden:.3f} a0\n'
                         f'2p-1s integral: {int_allowed:.3f} a0',
             transform=ax6.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle('Electric Dipole Selection Rules and Matrix Elements\n'
                 r'$\Delta l = \pm 1$, $\Delta m = 0, \pm 1$',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'selection_rules.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'selection_rules.png')}")


if __name__ == "__main__":
    main()
