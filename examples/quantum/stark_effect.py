"""
Experiment 170: Stark Effect

This experiment demonstrates the Stark effect - energy level shifts of atoms
in an external electric field.

Physics:
    The perturbation is V = -d.E = e*E*z (dipole interaction with field).

    Linear Stark Effect:
    - Occurs in degenerate systems (like hydrogen excited states)
    - Energy shift: Delta E = e*E*<psi|z|psi> (first order)
    - n=2 hydrogen: |2s> and |2p_z> mix, creating shift ~ E

    Quadratic Stark Effect:
    - Occurs in non-degenerate ground states
    - Energy shift: Delta E = -alpha*E^2/2 (second order)
    - alpha = polarizability = 2*sum |<n|z|0>|^2 / (E_n - E_0)
    - Ground state always lowered (polarizability positive)

    Selection Rules for dipole matrix elements:
    - Delta l = +/- 1 (parity change)
    - Delta m = 0 for z-polarized field
    - Delta m = +/- 1 for x,y polarized field
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial, genlaguerre, sph_harm


def hydrogen_radial_wavefunction(n, l, r, a0=1.0):
    """
    Radial wavefunction R_nl(r) for hydrogen atom.

    R_nl(r) = sqrt((2/n*a0)^3 * (n-l-1)! / (2n*(n+l)!)) *
              exp(-r/(n*a0)) * (2r/(n*a0))^l * L_{n-l-1}^{2l+1}(2r/(n*a0))

    Args:
        n: Principal quantum number
        l: Orbital angular momentum quantum number
        r: Radial coordinate (array)
        a0: Bohr radius

    Returns:
        Radial wavefunction values
    """
    rho = 2 * r / (n * a0)

    # Normalization
    norm = np.sqrt((2 / (n * a0))**3 * factorial(n - l - 1) /
                   (2 * n * factorial(n + l)))

    # Associated Laguerre polynomial
    L = genlaguerre(n - l - 1, 2 * l + 1)(rho)

    return norm * np.exp(-rho / 2) * rho**l * L


def hydrogen_energy(n, Ry=13.6):
    """
    Hydrogen energy level in eV.

    E_n = -Ry / n^2

    Args:
        n: Principal quantum number
        Ry: Rydberg energy in eV

    Returns:
        Energy in eV
    """
    return -Ry / n**2


def radial_dipole_matrix_element(n1, l1, n2, l2, a0=1.0):
    """
    Compute radial part of dipole matrix element.

    <n1,l1|r|n2,l2> = integral R_n1l1(r) * r * R_n2l2(r) * r^2 dr

    Args:
        n1, l1: Quantum numbers of bra state
        n2, l2: Quantum numbers of ket state
        a0: Bohr radius

    Returns:
        Radial matrix element (in units of a0)
    """
    # Numerical integration
    r_max = 50 * max(n1, n2) * a0
    r = np.linspace(0.001 * a0, r_max, 2000)
    dr = r[1] - r[0]

    R1 = hydrogen_radial_wavefunction(n1, l1, r, a0)
    R2 = hydrogen_radial_wavefunction(n2, l2, r, a0)

    # <r> matrix element includes extra r from z = r*cos(theta)
    integrand = R1 * r * R2 * r**2
    return np.trapz(integrand, r)


def angular_dipole_factor(l1, m1, l2, m2):
    """
    Angular part of z dipole matrix element.

    For z = r*cos(theta):
    <l1,m1|cos(theta)|l2,m2> is non-zero only if:
    - Delta l = +/- 1
    - Delta m = 0

    The factor is given by Clebsch-Gordan coefficients.

    Args:
        l1, m1: Angular quantum numbers of bra
        l2, m2: Angular quantum numbers of ket

    Returns:
        Angular factor
    """
    # Selection rules
    if m1 != m2:  # Delta m = 0 for z
        return 0.0
    if l1 != l2 + 1 and l1 != l2 - 1:  # Delta l = +/- 1
        return 0.0

    m = m1  # = m2 by selection rule

    if l1 == l2 + 1:
        # <l+1, m|cos(theta)|l, m>
        l = l2
        return np.sqrt((l + 1)**2 - m**2) / np.sqrt((2*l + 1) * (2*l + 3))
    else:
        # <l-1, m|cos(theta)|l, m>
        l = l2
        return np.sqrt(l**2 - m**2) / np.sqrt((2*l - 1) * (2*l + 1))


def hydrogen_dipole_matrix_element(n1, l1, m1, n2, l2, m2, a0=1.0):
    """
    Full dipole matrix element <n1,l1,m1|z|n2,l2,m2>.

    Args:
        n1, l1, m1: Quantum numbers of bra
        n2, l2, m2: Quantum numbers of ket
        a0: Bohr radius

    Returns:
        Dipole matrix element in units of e*a0
    """
    angular = angular_dipole_factor(l1, m1, l2, m2)
    if angular == 0:
        return 0.0

    radial = radial_dipole_matrix_element(n1, l1, n2, l2, a0)
    return radial * angular


def build_hydrogen_stark_matrix(n_max, m, E_field, a0=1.0, Ry=13.6):
    """
    Build Hamiltonian matrix for hydrogen in electric field.

    H = H_0 + e*E*z

    Only states with fixed m are coupled by z-polarized field.

    Args:
        n_max: Maximum principal quantum number
        m: Magnetic quantum number (fixed)
        E_field: Electric field strength (atomic units)
        a0: Bohr radius
        Ry: Rydberg energy

    Returns:
        Tuple of (H, state_labels)
    """
    # Generate states: (n, l) with l >= |m| and l < n
    states = []
    for n in range(1, n_max + 1):
        for l in range(abs(m), n):
            states.append((n, l))

    n_states = len(states)
    H = np.zeros((n_states, n_states), dtype=complex)

    labels = [f'|{n},{l},{m}>' for n, l in states]

    # Diagonal: unperturbed energies
    for i, (n, l) in enumerate(states):
        H[i, i] = hydrogen_energy(n, Ry)

    # Off-diagonal: dipole coupling
    for i, (n1, l1) in enumerate(states):
        for j, (n2, l2) in enumerate(states):
            if i != j:
                d = hydrogen_dipole_matrix_element(n1, l1, m, n2, l2, m, a0)
                # V = e*E*z in atomic units: e*a0*E * <z/a0>
                # Energy in eV: need to convert properly
                # Simplified: work in atomic units then convert
                H[i, j] = E_field * d * 27.2  # Convert Hartree to eV

    return H, labels


def hydrogen_polarizability(n, l, m, n_max=10, a0=1.0, Ry=13.6):
    """
    Calculate static polarizability for a hydrogen state.

    alpha = 2 * sum_{n' != n} |<n',l',m|z|n,l,m>|^2 / (E_n - E_n')

    Args:
        n, l, m: Quantum numbers of state
        n_max: Maximum n' for summation
        a0: Bohr radius
        Ry: Rydberg energy

    Returns:
        Polarizability in a0^3
    """
    E_n = hydrogen_energy(n, Ry)

    alpha = 0.0
    for n_prime in range(1, n_max + 1):
        for l_prime in [l - 1, l + 1]:  # Selection rule
            if l_prime >= 0 and l_prime < n_prime:
                d = hydrogen_dipole_matrix_element(n_prime, l_prime, m, n, l, m, a0)
                E_n_prime = hydrogen_energy(n_prime, Ry)

                if n_prime != n:
                    alpha += 2 * d**2 / (E_n - E_n_prime)

    return alpha


def main():
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    a0 = 1.0  # Bohr radius
    Ry = 13.6  # eV

    # ===== Plot 1: Linear Stark effect (n=2) =====
    ax1 = axes[0, 0]

    # n=2 states: 2s (l=0) and 2p (l=1)
    # For m=0: only |2,0,0> and |2,1,0> mix
    E_field_range = np.linspace(0, 0.01, 50)  # atomic units

    n2_energies = np.zeros((len(E_field_range), 4))

    for i, E in enumerate(E_field_range):
        # Build 2x2 matrix for 2s and 2p_z
        E0 = hydrogen_energy(2, Ry)

        d = hydrogen_dipole_matrix_element(2, 0, 0, 2, 1, 0, a0)
        V = E * d * 27.2  # eV

        H = np.array([[E0, V], [V, E0]])
        eigvals = np.linalg.eigvalsh(H)

        n2_energies[i, 0] = eigvals[0]
        n2_energies[i, 1] = eigvals[1]
        # 2p_x and 2p_y (m = +/-1) don't mix with anything at n=2
        n2_energies[i, 2] = E0
        n2_energies[i, 3] = E0

    E0_n2 = hydrogen_energy(2, Ry)
    ax1.plot(E_field_range * 5.14e9, (n2_energies[:, 0] - E0_n2) * 1000, 'b-', lw=2,
            label='|2s> - |2p_z> mix (lower)')
    ax1.plot(E_field_range * 5.14e9, (n2_energies[:, 1] - E0_n2) * 1000, 'r-', lw=2,
            label='|2s> + |2p_z> mix (upper)')
    ax1.plot(E_field_range * 5.14e9, (n2_energies[:, 2] - E0_n2) * 1000, 'gray', lw=2,
            linestyle='--', label='|2p_x>, |2p_y> (unshifted)')

    ax1.axhline(0, color='black', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Electric field (V/cm)')
    ax1.set_ylabel('Energy shift (meV)')
    ax1.set_title('Linear Stark Effect (n=2 Hydrogen)\nFirst-order shift ~ E')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ===== Plot 2: Quadratic Stark effect (ground state) =====
    ax2 = axes[0, 1]

    # Ground state polarizability
    alpha_1s = hydrogen_polarizability(1, 0, 0, n_max=15, a0=a0, Ry=Ry)

    E_field_quad = np.linspace(0, 0.005, 50)
    # Quadratic shift: Delta E = -alpha * E^2 / 2
    shift_perturbation = -0.5 * alpha_1s * E_field_quad**2

    # Exact calculation
    shift_exact = np.zeros_like(E_field_quad)
    for i, E in enumerate(E_field_quad):
        H, _ = build_hydrogen_stark_matrix(4, 0, E, a0, Ry)
        eigvals = np.linalg.eigvalsh(H)
        shift_exact[i] = eigvals[0] - hydrogen_energy(1, Ry)

    ax2.plot(E_field_quad * 5.14e9, shift_perturbation * 1e6, 'b--', lw=2,
            label='Perturbation: -alpha*E^2/2')
    ax2.plot(E_field_quad * 5.14e9, shift_exact * 1e6, 'r-', lw=2,
            label='Exact diagonalization')

    ax2.set_xlabel('Electric field (V/cm)')
    ax2.set_ylabel('Energy shift (microeV)')
    ax2.set_title(f'Quadratic Stark Effect (Ground State)\nPolarizability alpha = {alpha_1s:.1f} a0^3')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ===== Plot 3: Selection rules visualization =====
    ax3 = axes[0, 2]

    # Show allowed transitions
    n_max_vis = 4
    state_info = []

    for n in range(1, n_max_vis + 1):
        for l in range(n):
            for m in range(-l, l + 1):
                y_pos = -hydrogen_energy(n, Ry)
                x_pos = l + 0.2 * m
                state_info.append((n, l, m, x_pos, y_pos))

    # Plot energy levels
    for n, l, m, x, y in state_info:
        color = plt.cm.tab10(l / 4)
        ax3.scatter([x], [y], c=[color], s=100)
        ax3.annotate(f'{n}{["s","p","d","f"][l]}', (x, y), fontsize=7,
                    ha='center', va='bottom')

    # Draw allowed transitions from 2p (m=0)
    # 2p -> 1s and 2p -> 3s, 3d
    arrows = [
        ((1, 0.8), (0, 2.0)),   # 1s <- 2p
        ((1, 0.8), (2, 0.5)),   # 3d <- 2p
    ]

    for (x1, y1), (x2, y2) in arrows:
        ax3.annotate('', xy=(x1, y1), xytext=(x2, y2),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))

    ax3.set_xlabel('l + 0.2*m')
    ax3.set_ylabel('Binding energy (eV)')
    ax3.set_title('Selection Rules for z-Polarized Field\nDelta l = +/- 1, Delta m = 0')
    ax3.grid(True, alpha=0.3)

    # ===== Plot 4: Stark ladder for higher n =====
    ax4 = axes[1, 0]

    n = 3  # Consider n=3 manifold
    E_field_n3 = np.linspace(0, 0.008, 50)

    # For m=0, we have 3s, 3p, 3d
    n3_energies = np.zeros((len(E_field_n3), 3))

    for i, E in enumerate(E_field_n3):
        H, _ = build_hydrogen_stark_matrix(3, 0, E, a0, Ry)
        # Extract n=3 states (last 3 states for m=0: 3s, 3p, 3d)
        eigvals = np.linalg.eigvalsh(H)
        # n=3 energies are the 3 highest (least negative)
        n3_energies[i] = np.sort(eigvals)[-3:]

    E0_n3 = hydrogen_energy(3, Ry)
    for j in range(3):
        ax4.plot(E_field_n3 * 5.14e9, (n3_energies[:, j] - E0_n3) * 1000, lw=2,
                label=f'Level {j+1}')

    ax4.axhline(0, color='black', linestyle=':', alpha=0.5)
    ax4.set_xlabel('Electric field (V/cm)')
    ax4.set_ylabel('Energy shift (meV)')
    ax4.set_title('n=3 Stark Manifold\n(3s, 3p, 3d mixing)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # ===== Plot 5: Polarizability comparison =====
    ax5 = axes[1, 1]

    n_values = [1, 2, 3]
    l_values = [0, 0, 0]  # s states

    bar_positions = np.arange(len(n_values))
    alphas = []

    for n, l in zip(n_values, l_values):
        alpha = hydrogen_polarizability(n, l, 0, n_max=20, a0=a0, Ry=Ry)
        alphas.append(alpha)

    ax5.bar(bar_positions, alphas, alpha=0.7)
    ax5.set_xticks(bar_positions)
    ax5.set_xticklabels([f'{n}s' for n in n_values])
    ax5.set_ylabel('Polarizability (a0^3)')
    ax5.set_title('Static Polarizabilities\nalpha scales as n^7 for high n')
    ax5.grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for i, alpha in enumerate(alphas):
        ax5.text(i, alpha + 0.5, f'{alpha:.1f}', ha='center', fontsize=10)

    # ===== Plot 6: Ground state vs excited state comparison =====
    ax6 = axes[1, 2]

    E_field_compare = np.linspace(0, 0.003, 50)

    # Ground state (quadratic)
    shift_ground = -0.5 * alpha_1s * E_field_compare**2 * 1e6  # microeV

    # n=2 (linear) - use dipole matrix element
    d_n2 = hydrogen_dipole_matrix_element(2, 0, 0, 2, 1, 0, a0)
    shift_n2 = d_n2 * E_field_compare * 27.2 * 1e6  # microeV

    ax6.plot(E_field_compare * 5.14e9, shift_ground, 'b-', lw=2,
            label='1s: quadratic ~ E^2')
    ax6.plot(E_field_compare * 5.14e9, shift_n2, 'r-', lw=2,
            label='2s-2p: linear ~ E')
    ax6.plot(E_field_compare * 5.14e9, -shift_n2, 'r--', lw=2)

    ax6.axhline(0, color='black', linestyle=':', alpha=0.5)
    ax6.set_xlabel('Electric field (V/cm)')
    ax6.set_ylabel('Energy shift (microeV)')
    ax6.set_title('Linear vs Quadratic Stark Effect\nDegenerate (n=2) vs Non-degenerate (n=1)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.suptitle('Stark Effect in Hydrogen\n'
                 'Energy shifts in external electric field',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'stark_effect.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'stark_effect.png')}")

    # Print numerical results
    print("\n=== Stark Effect Results ===")

    print("\n1. Dipole matrix elements (in a0):")
    d_2s_2p = hydrogen_dipole_matrix_element(2, 0, 0, 2, 1, 0, a0)
    d_1s_2p = hydrogen_dipole_matrix_element(1, 0, 0, 2, 1, 0, a0)
    print(f"   <2s|z|2p_z> = {d_2s_2p:.4f} a0")
    print(f"   <1s|z|2p_z> = {d_1s_2p:.4f} a0")

    print("\n2. Polarizabilities (in a0^3):")
    for n in [1, 2, 3]:
        alpha = hydrogen_polarizability(n, 0, 0, n_max=20, a0=a0, Ry=Ry)
        print(f"   alpha({n}s) = {alpha:.2f} a0^3")
    print(f"   Analytical 1s: alpha = 4.5 a0^3 (9/2)")

    print("\n3. Energy shifts at E = 1000 V/cm:")
    E_au = 1000 / 5.14e9  # Convert to atomic units
    shift_1s = -0.5 * alpha_1s * E_au**2 * 27.2  # eV
    shift_n2 = d_2s_2p * E_au * 27.2  # eV
    print(f"   1s shift: {shift_1s*1e9:.3f} neV (quadratic)")
    print(f"   n=2 splitting: +/- {shift_n2*1e6:.3f} microeV (linear)")

    print("\n4. Selection rules:")
    print("   z-polarized field: Delta l = +/- 1, Delta m = 0")
    print("   x,y-polarized: Delta l = +/- 1, Delta m = +/- 1")


if __name__ == "__main__":
    main()
