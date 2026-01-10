"""
Example demonstrating special relativistic length contraction.

This example shows how objects appear contracted along the direction
of motion as they approach the speed of light.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.relativity import LorentzTransform


def lorentz_factor(v, c=1.0):
    """Calculate the Lorentz factor γ = 1/√(1-v²/c²)."""
    return 1.0 / np.sqrt(1 - (v/c)**2)


def contracted_length(L0, v, c=1.0):
    """Calculate contracted length L = L0/γ."""
    gamma = lorentz_factor(v, c)
    return L0 / gamma


def main():
    c = 1.0  # Speed of light (natural units)
    L0 = 1.0  # Rest length

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Length contraction vs velocity
    ax1 = axes[0, 0]

    v_range = np.linspace(0, 0.999*c, 500)
    L_contracted = contracted_length(L0, v_range, c)
    gamma_range = lorentz_factor(v_range, c)

    ax1.plot(v_range/c, L_contracted/L0, 'b-', lw=2, label='L/L₀')
    ax1.plot(v_range/c, 1/gamma_range, 'r--', lw=2, label='1/γ')
    ax1.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    ax1.set_xlabel('Velocity (v/c)')
    ax1.set_ylabel('Contracted Length (L/L₀)')
    ax1.set_title('Length Contraction vs Velocity')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.1)

    # Plot 2: Lorentz factor
    ax2 = axes[0, 1]

    ax2.semilogy(v_range/c, gamma_range, 'b-', lw=2)
    ax2.set_xlabel('Velocity (v/c)')
    ax2.set_ylabel('Lorentz Factor γ (log scale)')
    ax2.set_title('Lorentz Factor vs Velocity')
    ax2.grid(True, alpha=0.3, which='both')

    # Mark some notable velocities
    notable_v = [0.5, 0.9, 0.99, 0.999]
    for v in notable_v:
        gamma = lorentz_factor(v*c, c)
        ax2.plot(v, gamma, 'ro', markersize=8)
        ax2.annotate(f'v={v}c\nγ={gamma:.2f}', (v, gamma),
                    xytext=(5, 10), textcoords='offset points', fontsize=8)

    # Plot 3: Visual representation of a moving rod
    ax3 = axes[1, 0]

    velocities = [0, 0.5*c, 0.8*c, 0.9*c, 0.95*c, 0.99*c]
    y_positions = np.arange(len(velocities))

    for i, v in enumerate(velocities):
        L = contracted_length(L0, v, c)
        gamma = lorentz_factor(v, c) if v > 0 else 1.0

        # Draw rod centered at x=0
        rect = plt.Rectangle((-L/2, i - 0.3), L, 0.6,
                             facecolor=plt.cm.viridis(v/c),
                             edgecolor='black', lw=1)
        ax3.add_patch(rect)

        # Label
        if v == 0:
            label = f'v = 0 (rest)\nL = L₀'
        else:
            label = f'v = {v/c:.2f}c\nL = {L/L0:.3f}L₀'
        ax3.text(0.6, i, label, va='center', fontsize=9)

    ax3.set_xlim(-0.7, 1.2)
    ax3.set_ylim(-0.5, len(velocities) - 0.5)
    ax3.set_xlabel('Position (units of L₀)')
    ax3.set_ylabel('')
    ax3.set_title('Visual Length Contraction')
    ax3.set_yticks([])
    ax3.axvline(x=-L0/2, color='gray', linestyle='--', alpha=0.3)
    ax3.axvline(x=L0/2, color='gray', linestyle='--', alpha=0.3)

    # Plot 4: Muon decay example
    ax4 = axes[1, 1]

    # Muon parameters
    tau_0 = 2.2e-6  # Muon rest lifetime (s)
    v_muon = 0.9994 * c  # Typical cosmic ray muon speed
    gamma_muon = lorentz_factor(v_muon, c)

    # Distance traveled in lab frame
    d_lab = gamma_muon * tau_0 * v_muon * c * 3e8 / 1000  # km

    # In muon's frame, the atmosphere is contracted
    h_atmosphere = 10  # km (typical altitude where muons created)
    h_contracted = h_atmosphere / gamma_muon

    # Bar chart comparison
    labels = ['Atmosphere\n(lab frame)', 'Atmosphere\n(muon frame)', 'Distance muon\ncan travel']
    values = [h_atmosphere, h_contracted, d_lab]
    colors = ['blue', 'red', 'green']

    bars = ax4.bar(labels, values, color=colors, alpha=0.7, edgecolor='black')

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax4.annotate(f'{val:.2f} km',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10)

    ax4.set_ylabel('Distance (km)')
    ax4.set_title(f'Muon Example: v = {v_muon/c:.4f}c, γ = {gamma_muon:.1f}')
    ax4.grid(True, alpha=0.3, axis='y')

    # Add explanation
    ax4.text(0.5, 0.95, 'Why cosmic ray muons reach Earth:\n'
             'In muon frame, atmosphere is contracted so they can traverse it\n'
             'In lab frame, their lifetime is dilated so they travel further',
             transform=ax4.transAxes, fontsize=9, verticalalignment='top',
             ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Special Relativistic Length Contraction\n'
                 'L = L₀/γ = L₀√(1 - v²/c²)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'length_contraction.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'length_contraction.png')}")


if __name__ == "__main__":
    main()
