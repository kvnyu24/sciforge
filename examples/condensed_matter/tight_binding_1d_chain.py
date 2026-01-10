"""
Experiment 221: Tight-Binding 1D Chain Dispersion

Demonstrates the tight-binding model for a 1D atomic chain, showing
the cosine dispersion relation E(k) = E0 - 2t*cos(ka) that arises
from nearest-neighbor hopping in a periodic lattice.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def tight_binding_1d(k, a, t, E0=0):
    """
    Tight-binding dispersion for 1D chain with nearest-neighbor hopping.

    Args:
        k: Wavevector (can be array)
        a: Lattice constant
        t: Hopping parameter (typically negative for attractive)
        E0: On-site energy

    Returns:
        Energy E(k)
    """
    return E0 - 2 * t * np.cos(k * a)


def tight_binding_1d_nnn(k, a, t1, t2, E0=0):
    """
    Tight-binding with nearest and next-nearest neighbor hopping.

    Args:
        k: Wavevector
        a: Lattice constant
        t1: Nearest-neighbor hopping
        t2: Next-nearest-neighbor hopping
        E0: On-site energy

    Returns:
        Energy E(k)
    """
    return E0 - 2 * t1 * np.cos(k * a) - 2 * t2 * np.cos(2 * k * a)


def group_velocity(k, a, t):
    """
    Group velocity v_g = (1/hbar) * dE/dk for tight-binding model.

    Args:
        k: Wavevector
        a: Lattice constant
        t: Hopping parameter

    Returns:
        Group velocity (in units where hbar = 1)
    """
    return 2 * t * a * np.sin(k * a)


def effective_mass(k, a, t):
    """
    Effective mass m* = hbar^2 / (d^2E/dk^2).

    Args:
        k: Wavevector
        a: Lattice constant
        t: Hopping parameter

    Returns:
        Effective mass (in units where hbar = 1, m_e = 1)
    """
    d2E_dk2 = 2 * t * a**2 * np.cos(k * a)
    # Avoid division by zero near band edges
    d2E_dk2 = np.where(np.abs(d2E_dk2) < 1e-10, 1e-10, d2E_dk2)
    return 1.0 / d2E_dk2


def main():
    """Main simulation and visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Parameters
    a = 1.0  # Lattice constant
    t = 1.0  # Hopping parameter

    # k-points in first Brillouin zone
    k = np.linspace(-np.pi/a, np.pi/a, 500)

    # Plot 1: Basic dispersion relation
    ax1 = axes[0, 0]

    E = tight_binding_1d(k, a, t)
    ax1.plot(k * a / np.pi, E, 'b-', lw=3, label='E(k) = -2t cos(ka)')

    # Free electron parabola for comparison
    E_free = k**2 / 2 - t  # Shifted to match minimum
    ax1.plot(k * a / np.pi, E_free, 'r--', lw=2, alpha=0.7, label='Free electron (parabola)')

    ax1.set_xlabel('ka / pi')
    ax1.set_ylabel('Energy (units of t)')
    ax1.set_title('Tight-Binding Dispersion: 1D Chain')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    ax1.set_xlim(-1, 1)

    # Mark zone boundaries
    ax1.axvline(x=-1, color='green', linestyle=':', lw=2, label='BZ boundary')
    ax1.axvline(x=1, color='green', linestyle=':', lw=2)

    # Plot 2: Effect of next-nearest neighbor hopping
    ax2 = axes[0, 1]

    t2_values = [0, 0.2, 0.4, -0.2]
    colors = ['blue', 'green', 'red', 'purple']

    for t2, color in zip(t2_values, colors):
        E_nnn = tight_binding_1d_nnn(k, a, t, t2)
        ax2.plot(k * a / np.pi, E_nnn, color=color, lw=2, label=f't2/t = {t2}')

    ax2.set_xlabel('ka / pi')
    ax2.set_ylabel('Energy (units of t)')
    ax2.set_title('Effect of Next-Nearest Neighbor Hopping')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-1, 1)

    # Plot 3: Group velocity
    ax3 = axes[1, 0]

    v_g = group_velocity(k, a, t)
    ax3.plot(k * a / np.pi, v_g, 'b-', lw=2, label='Group velocity')
    ax3.fill_between(k * a / np.pi, 0, v_g, alpha=0.3)

    ax3.set_xlabel('ka / pi')
    ax3.set_ylabel('Group velocity (units of ta/hbar)')
    ax3.set_title('Group Velocity v_g = dE/dk')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-1, 1)
    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)

    # Mark maximum velocity points
    ax3.axvline(x=0.5, color='red', linestyle='--', alpha=0.7)
    ax3.axvline(x=-0.5, color='red', linestyle='--', alpha=0.7)

    # Plot 4: Effective mass
    ax4 = axes[1, 1]

    m_eff = effective_mass(k, a, t)
    # Clip for visualization
    m_eff_clipped = np.clip(m_eff, -10, 10)

    ax4.plot(k * a / np.pi, m_eff_clipped, 'b-', lw=2, label='Effective mass')
    ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.5)

    # Shade regions of positive (electron-like) and negative (hole-like) mass
    ax4.fill_between(k * a / np.pi, 0, m_eff_clipped,
                     where=m_eff_clipped > 0, alpha=0.3, color='blue', label='Electron-like')
    ax4.fill_between(k * a / np.pi, 0, m_eff_clipped,
                     where=m_eff_clipped < 0, alpha=0.3, color='red', label='Hole-like')

    ax4.set_xlabel('ka / pi')
    ax4.set_ylabel('Effective mass (units of hbar^2/ta^2)')
    ax4.set_title('Effective Mass m* = hbar^2 / (d^2E/dk^2)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-1, 1)
    ax4.set_ylim(-10, 10)

    plt.suptitle('Tight-Binding Model for 1D Atomic Chain\n'
                 'Nearest-neighbor hopping leads to cosine band dispersion',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'tight_binding_1d_chain.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'tight_binding_1d_chain.png')}")


if __name__ == "__main__":
    main()
