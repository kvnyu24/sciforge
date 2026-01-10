"""
Example demonstrating wave equation on a string with boundary reflection.

This example shows how waves reflect from fixed and free boundaries,
demonstrating phase inversion at fixed ends and no phase change at free ends.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.waves import Wave


def gaussian_pulse(x, x0, sigma):
    """Gaussian pulse centered at x0."""
    return np.exp(-(x - x0)**2 / (2 * sigma**2))


def simulate_wave_on_string(x, t_array, c, L, initial_pulse_pos, sigma,
                            left_bc='fixed', right_bc='fixed'):
    """
    Simulate wave propagation on a string using method of images.

    Args:
        x: Spatial grid
        t_array: Time array
        c: Wave speed
        L: String length
        initial_pulse_pos: Initial position of pulse
        sigma: Pulse width
        left_bc: 'fixed' or 'free' boundary condition at x=0
        right_bc: 'fixed' or 'free' boundary condition at x=L

    Returns:
        2D array of displacements [time, position]
    """
    U = np.zeros((len(t_array), len(x)))

    for i, t in enumerate(t_array):
        u = np.zeros_like(x)

        # Sum over multiple reflections (images)
        for n in range(-10, 11):
            # Positive traveling wave images
            x_pos = initial_pulse_pos + c * t + 2 * n * L
            sign_pos = 1.0

            # Apply reflection signs based on boundary conditions
            if left_bc == 'fixed' and n < 0:
                sign_pos *= (-1) ** abs(n)
            if right_bc == 'fixed' and n > 0:
                sign_pos *= (-1) ** abs(n)

            u += sign_pos * 0.5 * gaussian_pulse(x, x_pos, sigma)

            # Negative traveling wave images (from splitting)
            x_neg = initial_pulse_pos - c * t + 2 * n * L
            sign_neg = 1.0

            if left_bc == 'fixed' and n <= 0:
                sign_neg *= (-1) ** abs(n)
            if right_bc == 'fixed' and n > 0:
                sign_neg *= (-1) ** abs(n)

            u += sign_neg * 0.5 * gaussian_pulse(x, x_neg, sigma)

        # Apply boundary conditions by zeroing outside domain
        u[(x < 0) | (x > L)] = 0
        U[i, :] = u

    return U


def main():
    # String parameters
    L = 10.0       # String length
    c = 2.0        # Wave speed
    sigma = 0.5    # Pulse width

    # Spatial and time grids
    x = np.linspace(0, L, 500)
    t_max = 10.0
    t_array = np.linspace(0, t_max, 200)

    fig = plt.figure(figsize=(16, 12))

    # --- Fixed-Fixed Boundaries ---
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 4)

    # Simulate with initial pulse at x = 3
    initial_pos = 3.0
    times_snapshot = [0, 1.5, 3.5, 5.5, 7.5]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(times_snapshot)))

    for t, color in zip(times_snapshot, colors):
        # Simple d'Alembert with image method for fixed-fixed
        u = np.zeros_like(x)
        for n in range(-5, 6):
            # Right-going wave and its images
            u += ((-1)**abs(n)) * 0.5 * gaussian_pulse(x, initial_pos + c*t + 2*n*L, sigma)
            # Left-going wave and its images
            u += ((-1)**abs(n)) * 0.5 * gaussian_pulse(x, initial_pos - c*t + 2*n*L, sigma)

        ax1.plot(x, u, color=color, lw=2, label=f't = {t:.1f}')

    ax1.axvline(x=0, color='black', lw=3)
    ax1.axvline(x=L, color='black', lw=3)
    ax1.set_xlabel('Position (m)')
    ax1.set_ylabel('Displacement')
    ax1.set_title('Fixed-Fixed Boundaries\n(Phase inverts at reflection)')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.5, L + 0.5)
    ax1.set_ylim(-1.2, 1.2)

    # Space-time diagram for fixed-fixed
    t_fine = np.linspace(0, t_max, 200)
    U_ff = np.zeros((len(t_fine), len(x)))
    for i, t in enumerate(t_fine):
        u = np.zeros_like(x)
        for n in range(-5, 6):
            u += ((-1)**abs(n)) * 0.5 * gaussian_pulse(x, initial_pos + c*t + 2*n*L, sigma)
            u += ((-1)**abs(n)) * 0.5 * gaussian_pulse(x, initial_pos - c*t + 2*n*L, sigma)
        U_ff[i, :] = u

    im1 = ax2.imshow(U_ff, aspect='auto', extent=[0, L, t_max, 0],
                     cmap='RdBu', vmin=-1, vmax=1)
    ax2.set_xlabel('Position (m)')
    ax2.set_ylabel('Time (s)')
    ax2.set_title('Space-Time Diagram (Fixed-Fixed)')
    plt.colorbar(im1, ax=ax2, label='Displacement')

    # --- Fixed-Free Boundaries ---
    ax3 = fig.add_subplot(2, 3, 2)
    ax4 = fig.add_subplot(2, 3, 5)

    for t, color in zip(times_snapshot, colors):
        u = np.zeros_like(x)
        for n in range(-5, 6):
            # Fixed at x=0 (inverts), Free at x=L (no inversion)
            # The pattern depends on which boundary is hit
            sign_left = (-1) if n < 0 else 1
            sign_right = 1  # Free boundary doesn't invert

            # Right-going wave
            x_r = initial_pos + c*t + 2*n*L
            # Check number of left vs right reflections
            sign = 1.0
            if n < 0:
                sign *= (-1)**abs(n)  # Fixed reflections

            u += sign * 0.5 * gaussian_pulse(x, x_r, sigma)

            # Left-going wave
            x_l = initial_pos - c*t + 2*n*L
            u += sign * 0.5 * gaussian_pulse(x, x_l, sigma)

        ax3.plot(x, u, color=color, lw=2, label=f't = {t:.1f}')

    ax3.axvline(x=0, color='black', lw=3)
    ax3.axvline(x=L, color='green', lw=3, linestyle='--')
    ax3.set_xlabel('Position (m)')
    ax3.set_ylabel('Displacement')
    ax3.set_title('Fixed-Free Boundaries\n(Fixed: inverts, Free: no inversion)')
    ax3.legend(fontsize=8, loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-0.5, L + 0.5)
    ax3.set_ylim(-1.2, 1.2)

    # Space-time for fixed-free
    U_ff2 = np.zeros((len(t_fine), len(x)))
    for i, t in enumerate(t_fine):
        u = np.zeros_like(x)
        for n in range(-5, 6):
            sign = (-1)**abs(n) if n < 0 else 1
            u += sign * 0.5 * gaussian_pulse(x, initial_pos + c*t + 2*n*L, sigma)
            u += sign * 0.5 * gaussian_pulse(x, initial_pos - c*t + 2*n*L, sigma)
        U_ff2[i, :] = u

    im2 = ax4.imshow(U_ff2, aspect='auto', extent=[0, L, t_max, 0],
                     cmap='RdBu', vmin=-1, vmax=1)
    ax4.set_xlabel('Position (m)')
    ax4.set_ylabel('Time (s)')
    ax4.set_title('Space-Time Diagram (Fixed-Free)')
    plt.colorbar(im2, ax=ax4, label='Displacement')

    # --- Detailed reflection mechanics ---
    ax5 = fig.add_subplot(2, 3, 3)

    # Show a single reflection at fixed boundary in detail
    t_reflect = np.array([0, 1.0, 1.5, 2.0, 2.5, 3.0])
    x_short = np.linspace(0, 5, 300)
    pulse_start = 3.0

    colors_detail = plt.cm.plasma(np.linspace(0.1, 0.9, len(t_reflect)))

    for t, color in zip(t_reflect, colors_detail):
        # Incident wave (right to left)
        incident = gaussian_pulse(x_short, pulse_start - c*t, sigma)

        # Reflected wave (from image at x = -pulse_start)
        reflected = -gaussian_pulse(x_short, -pulse_start + c*t, sigma)

        # Total displacement (only physical region x >= 0)
        total = incident + reflected
        total[x_short < 0] = 0

        ax5.plot(x_short, total + 0.8*t, color=color, lw=2, label=f't = {t:.1f}')

    ax5.axvline(x=0, color='black', lw=3)
    ax5.set_xlabel('Position (m)')
    ax5.set_ylabel('Displacement (offset by time)')
    ax5.set_title('Fixed Boundary Reflection Detail\n(Wave inverts upon reflection)')
    ax5.legend(fontsize=8, loc='upper right')
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(-0.5, 5)

    # --- Energy conservation check ---
    ax6 = fig.add_subplot(2, 3, 6)

    # Calculate total energy over time for fixed-fixed case
    dx = x[1] - x[0]
    dt = t_fine[1] - t_fine[0]

    energies = []
    for i in range(len(t_fine) - 1):
        # Kinetic energy ~ (du/dt)^2
        dudt = (U_ff[i+1, :] - U_ff[i, :]) / dt

        # Potential energy ~ (du/dx)^2
        dudx = np.gradient(U_ff[i, :], dx)

        energy = 0.5 * np.sum(dudt**2 + c**2 * dudx**2) * dx
        energies.append(energy)

    ax6.plot(t_fine[:-1], energies, 'b-', lw=2)
    ax6.axhline(y=np.mean(energies), color='r', linestyle='--',
                label=f'Mean = {np.mean(energies):.3f}')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Total Energy (a.u.)')
    ax6.set_title('Energy Conservation\n(Fixed-Fixed Boundaries)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(0, max(energies) * 1.5)

    plt.suptitle('Wave Equation on String: Boundary Reflections\n'
                 'Black = Fixed (displacement = 0), Green dashed = Free (slope = 0)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'wave_boundary_reflection.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'wave_boundary_reflection.png')}")


if __name__ == "__main__":
    main()
