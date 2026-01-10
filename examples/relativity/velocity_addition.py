"""
Experiment 187: Relativistic Velocity Addition

This experiment demonstrates how velocities add in special relativity,
showing that the sum of two subluminal velocities is always subluminal.

Physical concepts:
- Relativistic velocity addition formula
- Comparison with Galilean velocity addition
- Speed of light as cosmic speed limit
- Rapidity and its additive property
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt


def galilean_addition(u, v):
    """Galilean velocity addition: w = u + v"""
    return u + v


def relativistic_addition(u, v, c=1.0):
    """
    Relativistic velocity addition: w = (u + v) / (1 + uv/c^2)

    Args:
        u: Velocity of object in frame S'
        v: Velocity of frame S' relative to frame S
        c: Speed of light

    Returns:
        Velocity of object in frame S
    """
    return (u + v) / (1 + u * v / c**2)


def rapidity(v, c=1.0):
    """
    Calculate rapidity: phi = arctanh(v/c)

    Rapidities add linearly: phi_total = phi_1 + phi_2
    """
    beta = v / c
    return np.arctanh(beta)


def velocity_from_rapidity(phi, c=1.0):
    """Convert rapidity back to velocity: v = c * tanh(phi)"""
    return c * np.tanh(phi)


def main():
    c = 1.0  # Speed of light

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ==========================================================================
    # Plot 1: Comparison of Galilean and Relativistic addition
    # ==========================================================================
    ax1 = axes[0, 0]

    v = 0.6 * c  # Frame velocity

    u_range = np.linspace(-0.99*c, 0.99*c, 200)

    w_galilean = galilean_addition(u_range, v)
    w_relativistic = relativistic_addition(u_range, v, c)

    ax1.plot(u_range/c, w_galilean/c, 'b--', lw=2, label='Galilean: w = u + v')
    ax1.plot(u_range/c, w_relativistic/c, 'r-', lw=2, label='Relativistic: w = (u+v)/(1+uv/c^2)')

    # Speed of light limits
    ax1.axhline(y=1, color='gold', linestyle='-', lw=2, alpha=0.7, label='c (speed limit)')
    ax1.axhline(y=-1, color='gold', linestyle='-', lw=2, alpha=0.7)

    ax1.set_xlabel("Object velocity in S' frame (u/c)")
    ax1.set_ylabel('Object velocity in S frame (w/c)')
    ax1.set_title(f'Velocity Addition (frame velocity v = {v/c}c)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1.5, 1.5)

    # Mark key point: u = c
    ax1.annotate('Light speed\nis invariant!',
                xy=(0.9, 1.0), xytext=(0.5, 0.6),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10, color='green',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # ==========================================================================
    # Plot 2: Multiple successive additions
    # ==========================================================================
    ax2 = axes[0, 1]

    # Add same velocity multiple times
    delta_v = 0.3 * c  # Each step adds 0.3c

    n_steps = 20
    v_galilean = np.zeros(n_steps + 1)
    v_relativistic = np.zeros(n_steps + 1)
    steps = np.arange(n_steps + 1)

    for i in range(1, n_steps + 1):
        v_galilean[i] = galilean_addition(v_galilean[i-1], delta_v)
        v_relativistic[i] = relativistic_addition(v_relativistic[i-1], delta_v, c)

    ax2.plot(steps, v_galilean/c, 'b--', lw=2, marker='o', markersize=4,
            label='Galilean')
    ax2.plot(steps, v_relativistic/c, 'r-', lw=2, marker='s', markersize=4,
            label='Relativistic')

    ax2.axhline(y=1, color='gold', linestyle='-', lw=2, label='Speed of light')
    ax2.axhline(y=-1, color='gold', linestyle='-', lw=2)

    ax2.set_xlabel(f'Number of velocity additions (each +{delta_v/c}c)')
    ax2.set_ylabel('Cumulative velocity (v/c)')
    ax2.set_title('Successive Velocity Additions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.2, 3)

    # Annotate asymptotic behavior
    ax2.annotate('Approaches c\nasymptotically',
                xy=(n_steps, v_relativistic[-1]/c), xytext=(n_steps-5, 0.6),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')

    # ==========================================================================
    # Plot 3: Rapidity addition
    # ==========================================================================
    ax3 = axes[1, 0]

    v_range = np.linspace(0.01, 0.99, 100) * c
    phi_range = rapidity(v_range, c)

    # Compare velocity and rapidity
    ax3_twin = ax3.twinx()

    ax3.plot(v_range/c, v_range/c, 'b-', lw=2, label='Velocity v/c')
    ax3_twin.plot(v_range/c, phi_range, 'r-', lw=2, label='Rapidity phi')

    ax3.set_xlabel('Original velocity v/c')
    ax3.set_ylabel('Velocity v/c', color='blue')
    ax3_twin.set_ylabel('Rapidity phi', color='red')
    ax3.set_title('Velocity vs Rapidity')

    # Add rapidity addition example
    phi1 = rapidity(0.6*c, c)
    phi2 = rapidity(0.8*c, c)
    phi_sum = phi1 + phi2
    v_result = velocity_from_rapidity(phi_sum, c)

    ax3.axhline(y=0.6, color='blue', linestyle=':', alpha=0.5)
    ax3.axhline(y=0.8, color='blue', linestyle=':', alpha=0.5)

    textstr = (f'Rapidity addition:\n'
               f'v1 = 0.6c -> phi1 = {phi1:.3f}\n'
               f'v2 = 0.8c -> phi2 = {phi2:.3f}\n'
               f'phi_sum = {phi_sum:.3f}\n'
               f'v_result = {v_result/c:.4f}c')
    ax3.text(0.05, 0.95, textstr, transform=ax3.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='center left')
    ax3_twin.legend(loc='center right')

    # ==========================================================================
    # Plot 4: Contour plot of relativistic velocity addition
    # ==========================================================================
    ax4 = axes[1, 1]

    u_range = np.linspace(0, 0.99, 100) * c
    v_range = np.linspace(0, 0.99, 100) * c
    U, V = np.meshgrid(u_range, v_range)

    W = relativistic_addition(U, V, c)

    levels = np.array([0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]) * c
    contour = ax4.contourf(U/c, V/c, W/c, levels=levels/c, cmap='viridis', extend='both')
    cbar = plt.colorbar(contour, ax=ax4, label='Resultant velocity w/c')

    # Add contour lines
    ax4.contour(U/c, V/c, W/c, levels=levels/c, colors='white', linewidths=0.5)

    # Galilean diagonal (u + v = c)
    diag = np.linspace(0, 1, 100)
    ax4.plot(diag, 1 - diag, 'r--', lw=2, label='u + v = c (Galilean)')

    ax4.set_xlabel('Velocity u/c')
    ax4.set_ylabel('Velocity v/c')
    ax4.set_title('Relativistic Velocity Addition Contours')
    ax4.legend(loc='upper right')

    # Mark special cases
    special_cases = [
        (0.5, 0.5, 'Adding two\n0.5c velocities'),
        (0.9, 0.9, 'Adding two\n0.9c velocities'),
    ]

    for u, v, label in special_cases:
        w = relativistic_addition(u*c, v*c, c)
        ax4.plot(u, v, 'ro', markersize=8)
        ax4.annotate(f'{label}\nResult: {w/c:.3f}c',
                    xy=(u, v), xytext=(u-0.2, v-0.15),
                    fontsize=8, ha='center',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('Relativistic Velocity Addition\n'
                 'w = (u + v) / (1 + uv/c^2)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Print some numerical examples
    print("Relativistic Velocity Addition Examples:")
    print("-" * 50)
    examples = [
        (0.5, 0.5), (0.9, 0.9), (0.99, 0.99),
        (0.6, 0.8), (1.0, 0.5), (0.9, 0.99)
    ]
    for u, v in examples:
        w_gal = galilean_addition(u*c, v*c)
        w_rel = relativistic_addition(u*c, v*c, c)
        print(f"u = {u}c, v = {v}c:")
        print(f"  Galilean: w = {w_gal/c:.4f}c")
        print(f"  Relativistic: w = {w_rel/c:.6f}c")
        print()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'velocity_addition.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    main()
