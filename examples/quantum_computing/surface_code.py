"""
Experiment 183: Surface Code Syndrome Demonstration

Demonstrates the surface code, a topological quantum error correction code
with high fault tolerance threshold.

Physics:
    The surface code is a 2D arrangement of data qubits with stabilizer measurements:

    Layout (d=3 distance code):
        D - Z - D - Z - D
        |   |   |   |   |
        X - D - X - D - X
        |   |   |   |   |
        D - Z - D - Z - D
        |   |   |   |   |
        X - D - X - D - X
        |   |   |   |   |
        D - Z - D - Z - D

    D = Data qubit
    X = X-stabilizer (measures X_1 X_2 X_3 X_4 product)
    Z = Z-stabilizer (measures Z_1 Z_2 Z_3 Z_4 product)

    Key properties:
    - Distance d: Can correct floor((d-1)/2) errors
    - Threshold: ~1% physical error rate
    - 2D local operations (nearest neighbor only)
    - Fault-tolerant syndrome extraction

    Error detection:
    - X errors flip Z-stabilizer values
    - Z errors flip X-stabilizer values
    - Error chains create syndrome patterns at endpoints
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.collections import PatchCollection


class SurfaceCode:
    """
    Simplified surface code implementation for demonstration.

    Uses a d x d grid of data qubits with stabilizers.
    """

    def __init__(self, distance=3):
        """
        Initialize surface code.

        Args:
            distance: Code distance (odd number)
        """
        if distance % 2 == 0:
            raise ValueError("Distance must be odd")

        self.d = distance

        # Number of data qubits
        self.n_data = distance * distance

        # Initialize data qubits (all |0>)
        # Store just classical bit values for syndrome demo
        self.data = np.zeros((distance, distance), dtype=int)

        # Z errors (bit flips in X basis = phase errors)
        self.z_errors = np.zeros((distance, distance), dtype=int)

        # X errors (bit flips in Z basis)
        self.x_errors = np.zeros((distance, distance), dtype=int)

    def apply_x_error(self, row, col):
        """Apply X (bit flip) error at position."""
        self.x_errors[row, col] ^= 1

    def apply_z_error(self, row, col):
        """Apply Z (phase flip) error at position."""
        self.z_errors[row, col] ^= 1

    def measure_z_stabilizers(self):
        """
        Measure Z stabilizers (detect X errors).

        Z stabilizers are on "white" plaquettes.
        Returns syndrome: 0 = no error detected, 1 = error detected

        For d=3: stabilizers at positions (0,0), (0,2), (2,0), (2,2) but offset
        """
        d = self.d
        # Z stabilizers detect X errors
        # Each stabilizer measures product of Z on surrounding data qubits
        # X error flips the Z measurement

        syndrome = []

        # Interior Z-stabilizers (plaquette centers)
        for i in range(d - 1):
            for j in range(d - 1):
                if (i + j) % 2 == 0:  # Checkerboard pattern
                    # Z stabilizer at plaquette (i, j)
                    # Measures qubits: (i,j), (i+1,j), (i,j+1), (i+1,j+1)
                    parity = (self.x_errors[i, j] ^
                             self.x_errors[i+1, j] ^
                             self.x_errors[i, j+1] ^
                             self.x_errors[i+1, j+1])
                    syndrome.append(((i, j), parity))

        return syndrome

    def measure_x_stabilizers(self):
        """
        Measure X stabilizers (detect Z errors).

        X stabilizers are on "black" plaquettes.
        """
        d = self.d
        syndrome = []

        for i in range(d - 1):
            for j in range(d - 1):
                if (i + j) % 2 == 1:  # Opposite checkerboard
                    parity = (self.z_errors[i, j] ^
                             self.z_errors[i+1, j] ^
                             self.z_errors[i, j+1] ^
                             self.z_errors[i+1, j+1])
                    syndrome.append(((i, j), parity))

        return syndrome

    def get_syndrome_pattern(self):
        """Get all syndrome measurements."""
        z_syn = self.measure_z_stabilizers()
        x_syn = self.measure_x_stabilizers()
        return z_syn, x_syn

    def reset(self):
        """Reset all errors."""
        self.x_errors = np.zeros((self.d, self.d), dtype=int)
        self.z_errors = np.zeros((self.d, self.d), dtype=int)


def visualize_surface_code(ax, code, title="Surface Code"):
    """
    Visualize the surface code with errors and syndromes.

    Args:
        ax: Matplotlib axis
        code: SurfaceCode instance
        title: Plot title
    """
    d = code.d
    ax.set_xlim(-0.5, d - 0.5)
    ax.set_ylim(-0.5, d - 0.5)

    # Draw data qubits
    for i in range(d):
        for j in range(d):
            color = 'lightblue'
            if code.x_errors[i, j]:
                color = 'red'  # X error
            elif code.z_errors[i, j]:
                color = 'orange'  # Z error

            circle = Circle((j, d - 1 - i), 0.2, facecolor=color, edgecolor='black')
            ax.add_patch(circle)

    # Get syndromes
    z_syn, x_syn = code.get_syndrome_pattern()

    # Draw Z-stabilizers (detect X errors)
    for (i, j), value in z_syn:
        x_pos = j + 0.5
        y_pos = d - 1 - i - 0.5

        color = 'green' if value == 0 else 'red'
        square = Rectangle((x_pos - 0.15, y_pos - 0.15), 0.3, 0.3,
                           facecolor=color, edgecolor='black', alpha=0.7)
        ax.add_patch(square)
        ax.text(x_pos, y_pos, 'Z', ha='center', va='center', fontsize=8, fontweight='bold')

    # Draw X-stabilizers (detect Z errors)
    for (i, j), value in x_syn:
        x_pos = j + 0.5
        y_pos = d - 1 - i - 0.5

        color = 'green' if value == 0 else 'red'
        square = Rectangle((x_pos - 0.15, y_pos - 0.15), 0.3, 0.3,
                           facecolor=color, edgecolor='black', alpha=0.7)
        ax.add_patch(square)
        ax.text(x_pos, y_pos, 'X', ha='center', va='center', fontsize=8, fontweight='bold')

    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xticks(range(d))
    ax.set_yticks(range(d))
    ax.grid(True, alpha=0.3)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue',
               markersize=10, label='Data qubit (no error)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markersize=10, label='X error'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
               markersize=10, label='Z error'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='green',
               markersize=10, label='Stabilizer OK'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='red',
               markersize=10, label='Stabilizer triggered'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=7)


def main():
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # ===== Plot 1: No errors =====
    ax1 = axes[0, 0]
    code = SurfaceCode(distance=3)
    visualize_surface_code(ax1, code, "No Errors\n(All stabilizers green)")

    # ===== Plot 2: Single X error =====
    ax2 = axes[0, 1]
    code = SurfaceCode(distance=3)
    code.apply_x_error(1, 1)  # Center qubit
    visualize_surface_code(ax2, code, "Single X Error at (1,1)\n(Z-stabilizers detect)")

    # ===== Plot 3: Two X errors (chain) =====
    ax3 = axes[0, 2]
    code = SurfaceCode(distance=3)
    code.apply_x_error(0, 1)
    code.apply_x_error(1, 1)
    visualize_surface_code(ax3, code, "X Error Chain\n(Syndromes at endpoints)")

    # ===== Plot 4: Single Z error =====
    ax4 = axes[1, 0]
    code = SurfaceCode(distance=3)
    code.apply_z_error(1, 1)
    visualize_surface_code(ax4, code, "Single Z Error at (1,1)\n(X-stabilizers detect)")

    # ===== Plot 5: Mixed errors =====
    ax5 = axes[1, 1]
    code = SurfaceCode(distance=3)
    code.apply_x_error(0, 0)
    code.apply_z_error(2, 2)
    visualize_surface_code(ax5, code, "Mixed Errors\n(X at (0,0), Z at (2,2))")

    # ===== Plot 6: Error threshold illustration =====
    ax6 = axes[1, 2]

    # Plot logical error rate vs physical error rate for different distances
    p_physical = np.linspace(0, 0.15, 100)

    for d in [3, 5, 7]:
        # Simplified model: P_logical ~ (p/p_th)^((d+1)/2) for p < p_th
        p_threshold = 0.01  # ~1% threshold
        p_logical = (p_physical / p_threshold) ** ((d + 1) / 2)
        p_logical = np.minimum(p_logical, 1.0)

        ax6.semilogy(p_physical * 100, p_logical, lw=2, label=f'd = {d}')

    ax6.axvline(1.0, color='gray', linestyle='--', label='Threshold ~1%')
    ax6.axhline(1e-6, color='gray', linestyle=':', alpha=0.5)

    ax6.set_xlabel('Physical Error Rate (%)')
    ax6.set_ylabel('Logical Error Rate')
    ax6.set_title('Surface Code: Logical vs Physical Error Rate\n(Higher distance = better correction)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(0, 15)
    ax6.set_ylim(1e-10, 1)

    plt.suptitle('Surface Code: Topological Quantum Error Correction\n'
                 '2D array of qubits with local stabilizer measurements',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'surface_code.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'surface_code.png')}")

    # Print explanation
    print("\n=== Surface Code Syndrome Demonstration ===")

    print("\nKey Concepts:")
    print("1. Data qubits arranged in 2D grid")
    print("2. Z-stabilizers: Detect X errors (measure Z_1 Z_2 Z_3 Z_4 product)")
    print("3. X-stabilizers: Detect Z errors (measure X_1 X_2 X_3 X_4 product)")
    print("4. Errors create syndromes at chain endpoints")
    print("5. Minimum weight matching for error correction")

    print("\nSurface Code Properties:")
    print(f"  Distance d: Number of qubits along edge")
    print(f"  Data qubits: d^2")
    print(f"  Correctable errors: floor((d-1)/2)")
    print(f"  Threshold: ~1% per gate operation")

    print("\nSyndrome Patterns:")
    print("  - Single error: Two adjacent stabilizers triggered")
    print("  - Error chain: Stabilizers at chain endpoints only")
    print("  - Logical error: Chain spanning the code (uncorrectable)")

    print("\nWhy Surface Code is Important:")
    print("  - Local operations only (2D nearest neighbor)")
    print("  - High threshold (~1%)")
    print("  - Scalable: Just add more qubits")
    print("  - Used by Google, IBM, and other quantum computing efforts")


if __name__ == "__main__":
    main()
