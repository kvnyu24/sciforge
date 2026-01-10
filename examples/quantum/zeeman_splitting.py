"""
Experiment 161: Zeeman Splitting

This experiment demonstrates the Zeeman effect - the splitting of atomic energy
levels in a magnetic field, including:
- Normal Zeeman effect (spin-less case)
- Anomalous Zeeman effect (including electron spin)
- Selection rules for transitions
- Paschen-Back regime (strong field limit)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def lande_g_factor(l: int, s: float, j: float) -> float:
    """
    Calculate Lande g-factor.

    g_J = 1 + (J(J+1) + S(S+1) - L(L+1)) / (2*J(J+1))

    Args:
        l: Orbital angular momentum quantum number
        s: Spin quantum number
        j: Total angular momentum quantum number

    Returns:
        Lande g-factor
    """
    if j == 0:
        return 0

    numerator = j * (j + 1) + s * (s + 1) - l * (l + 1)
    denominator = 2 * j * (j + 1)

    return 1 + numerator / denominator


def zeeman_energy_shift(m_j: float, g_j: float, B: float,
                        mu_B: float = 1.0) -> float:
    """
    Calculate Zeeman energy shift.

    Delta E = g_J * mu_B * B * m_J

    Args:
        m_j: Magnetic quantum number
        g_j: Lande g-factor
        B: Magnetic field strength
        mu_B: Bohr magneton

    Returns:
        Energy shift
    """
    return g_j * mu_B * B * m_j


def get_allowed_j_values(l: int, s: float = 0.5) -> list:
    """Get allowed J values for given L and S."""
    j_min = abs(l - s)
    j_max = l + s

    if j_min == j_max:
        return [j_min]

    return [j_min, j_max]


def transition_allowed(m_j1: float, m_j2: float, polarization: str) -> bool:
    """
    Check if transition is allowed based on selection rules.

    Delta m_J = 0 (pi), +/-1 (sigma+/-)
    """
    delta_m = m_j1 - m_j2

    if polarization == 'pi':
        return abs(delta_m) < 1e-10
    elif polarization == 'sigma+':
        return abs(delta_m - 1) < 1e-10
    elif polarization == 'sigma-':
        return abs(delta_m + 1) < 1e-10

    return False


def main():
    # Physical constants (natural units)
    mu_B = 1.0  # Bohr magneton

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Plot 1: Normal Zeeman effect (L=1, S=0, J=1)
    ax1 = axes[0, 0]

    l = 1
    s = 0  # Spinless case
    j = 1
    g_j = lande_g_factor(l, s, j)  # Should be 1 for normal Zeeman

    B_values = np.linspace(0, 2, 100)

    m_j_values = np.arange(-j, j + 1)
    colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(m_j_values)))

    for m_j, color in zip(m_j_values, colors):
        E_shift = [zeeman_energy_shift(m_j, g_j, B, mu_B) for B in B_values]
        ax1.plot(B_values, E_shift, color=color, lw=2, label=f'm_J = {m_j:.0f}')

    ax1.set_xlabel('Magnetic Field B (mu_B units)')
    ax1.set_ylabel('Energy Shift (mu_B units)')
    ax1.set_title(f'Normal Zeeman Effect\nL={l}, S={s}, J={j}, g_J={g_j:.2f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Plot 2: Anomalous Zeeman effect (2P state: L=1, S=1/2)
    ax2 = axes[0, 1]

    l = 1
    s = 0.5
    j_values = get_allowed_j_values(l, s)  # J = 1/2, 3/2

    B = 1.0

    for j in j_values:
        g_j = lande_g_factor(l, s, j)
        m_j_values = np.arange(-j, j + 1)

        y_offset = (j - 0.5) * 2  # Offset for visualization

        for m_j in m_j_values:
            E_shift = zeeman_energy_shift(m_j, g_j, B, mu_B)
            ax2.hlines(y_offset + E_shift, 0.2, 0.8, colors='blue', lw=3)
            ax2.text(0.85, y_offset + E_shift, f'm_J={m_j:+.1f}', va='center', fontsize=9)

        ax2.text(0.1, y_offset, f'J={j}, g_J={g_j:.3f}', va='center', fontsize=10)

    ax2.set_xlim(0, 1)
    ax2.set_ylim(-3, 4)
    ax2.set_xticks([])
    ax2.set_ylabel('Energy (mu_B * B)')
    ax2.set_title(f'Anomalous Zeeman Effect\n2P State (L=1, S=1/2), B=1')

    # Plot 3: Zeeman splitting of D-line transitions
    ax3 = axes[0, 2]

    # Sodium D-lines: 3S_1/2 -> 3P_1/2, 3P_3/2
    # Initial state: 3S_1/2 (L=0, S=1/2, J=1/2)
    l_i = 0
    s_i = 0.5
    j_i = 0.5
    g_i = lande_g_factor(l_i, s_i, j_i)  # g = 2

    # Final states: 3P_1/2 and 3P_3/2
    l_f = 1
    s_f = 0.5

    B = 0.5

    # Draw transitions
    y_base_i = 0
    y_base_f1 = 2  # J = 1/2
    y_base_f2 = 4  # J = 3/2

    # Initial state levels
    for m_j in np.arange(-j_i, j_i + 1):
        E = zeeman_energy_shift(m_j, g_i, B, mu_B)
        ax3.hlines(y_base_i + E, 0.1, 0.3, colors='green', lw=3)
        ax3.text(0.05, y_base_i + E, f'{m_j:+.1f}', va='center', fontsize=8)

    ax3.text(0.2, y_base_i - 0.8, '3S$_{1/2}$', ha='center', fontsize=10)

    # Final state J=1/2
    j_f = 0.5
    g_f = lande_g_factor(l_f, s_f, j_f)

    for m_j in np.arange(-j_f, j_f + 1):
        E = zeeman_energy_shift(m_j, g_f, B, mu_B)
        ax3.hlines(y_base_f1 + E, 0.4, 0.6, colors='red', lw=3)
        ax3.text(0.65, y_base_f1 + E, f'{m_j:+.1f}', va='center', fontsize=8)

    ax3.text(0.5, y_base_f1 - 0.8, '3P$_{1/2}$', ha='center', fontsize=10)

    # Final state J=3/2
    j_f = 1.5
    g_f = lande_g_factor(l_f, s_f, j_f)

    for m_j in np.arange(-j_f, j_f + 1):
        E = zeeman_energy_shift(m_j, g_f, B, mu_B)
        ax3.hlines(y_base_f2 + E, 0.7, 0.9, colors='blue', lw=3)
        ax3.text(0.92, y_base_f2 + E, f'{m_j:+.1f}', va='center', fontsize=8)

    ax3.text(0.8, y_base_f2 - 1.2, '3P$_{3/2}$', ha='center', fontsize=10)

    ax3.set_xlim(0, 1)
    ax3.set_ylim(-1.5, 6.5)
    ax3.set_xticks([])
    ax3.set_ylabel('Energy (arb. units)')
    ax3.set_title(f'Sodium D-Line Zeeman Levels (B = {B})')

    # Plot 4: Transition spectrum (Normal Zeeman)
    ax4 = axes[1, 0]

    # D2 line: 3P_3/2 -> 3S_1/2
    j_i = 0.5  # 3S_1/2
    j_f = 1.5  # 3P_3/2
    g_i = lande_g_factor(0, 0.5, j_i)  # 2
    g_f = lande_g_factor(1, 0.5, j_f)  # 4/3

    B = 1.0
    E0 = 10  # Base transition energy

    transitions = []
    polarizations = []

    for m_i in np.arange(-j_i, j_i + 1):
        for m_f in np.arange(-j_f, j_f + 1):
            for pol in ['sigma-', 'pi', 'sigma+']:
                if transition_allowed(m_f, m_i, pol):
                    E_i = zeeman_energy_shift(m_i, g_i, B, mu_B)
                    E_f = zeeman_energy_shift(m_f, g_f, B, mu_B)
                    E_trans = E0 + E_f - E_i
                    transitions.append(E_trans)
                    polarizations.append(pol)

    # Plot transition lines
    colors_pol = {'sigma-': 'blue', 'pi': 'green', 'sigma+': 'red'}
    labels_used = set()

    for E, pol in zip(transitions, polarizations):
        label = pol if pol not in labels_used else None
        labels_used.add(pol)
        ax4.axvline(x=E - E0, color=colors_pol[pol], lw=2, alpha=0.7, label=label)

    ax4.set_xlabel('Energy Shift from Line Center (mu_B)')
    ax4.set_ylabel('Intensity (arb.)')
    ax4.set_title('Zeeman Transition Spectrum (D2 line)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Paschen-Back regime
    ax5 = axes[1, 1]

    # Compare weak and strong field
    l = 1
    s = 0.5

    B_weak = np.linspace(0, 0.3, 50)
    B_strong = np.linspace(0.3, 3, 50)
    B_full = np.concatenate([B_weak, B_strong])

    # In Paschen-Back: E = m_L * mu_B * B + 2 * m_S * mu_B * B + spin-orbit
    # Simplified: energies become linear in B with slope (m_L + 2*m_S)

    for j in [0.5, 1.5]:
        g_j = lande_g_factor(l, s, j)
        for m_j in np.arange(-j, j + 1):
            # Weak field (Zeeman)
            E_weak = zeeman_energy_shift(m_j, g_j, B_weak, mu_B)
            offset = (j - 1) * 3  # Spin-orbit splitting
            ax5.plot(B_weak, E_weak + offset, 'b-', lw=2)

    # Strong field approximation
    for m_l in [-1, 0, 1]:
        for m_s in [-0.5, 0.5]:
            E_strong = (m_l + 2 * m_s) * mu_B * B_strong
            offset = 0  # Ignore spin-orbit in Paschen-Back
            ax5.plot(B_strong, E_strong + offset, 'r--', lw=1.5, alpha=0.7)

    ax5.axvline(x=0.3, color='gray', linestyle=':', alpha=0.7)
    ax5.text(0.15, 3.5, 'Zeeman', ha='center', fontsize=10)
    ax5.text(1.5, 3.5, 'Paschen-Back', ha='center', fontsize=10)

    ax5.set_xlabel('Magnetic Field B (mu_B units)')
    ax5.set_ylabel('Energy (mu_B units)')
    ax5.set_title('Weak to Strong Field Transition')
    ax5.grid(True, alpha=0.3)

    # Plot 6: g-factor variation with J
    ax6 = axes[1, 2]

    s = 0.5

    l_range = range(0, 5)
    markers = ['o', 's', '^', 'D', 'v']

    for l, marker in zip(l_range, markers):
        j_vals = get_allowed_j_values(l, s)
        g_vals = [lande_g_factor(l, s, j) for j in j_vals]

        for j, g in zip(j_vals, g_vals):
            ax6.scatter(l, g, s=100, marker=marker, label=f'L={l}, J={j}')

    ax6.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='g=1 (orbital only)')
    ax6.axhline(y=2, color='gray', linestyle=':', alpha=0.5, label='g=2 (spin only)')

    ax6.set_xlabel('Orbital Angular Momentum L')
    ax6.set_ylabel('Lande g-factor')
    ax6.set_title('g-factor vs Angular Momentum (S=1/2)')
    ax6.grid(True, alpha=0.3)
    ax6.legend(fontsize=7, ncol=2)

    plt.suptitle('Zeeman Effect: Atomic Energy Level Splitting in Magnetic Field\n'
                 r'$\Delta E = g_J \mu_B B m_J$',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'zeeman_splitting.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'zeeman_splitting.png')}")


if __name__ == "__main__":
    main()
