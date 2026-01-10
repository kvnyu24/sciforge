"""
Experiment 95: EM wave from Maxwell's equations.

This example demonstrates electromagnetic wave propagation derived
directly from Maxwell's equations, showing the coupled E and B fields.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Physical constants
C = 2.998e8      # Speed of light (m/s)
MU_0 = 4 * np.pi * 1e-7  # Permeability of free space
EPSILON_0 = 1 / (MU_0 * C**2)  # Permittivity of free space


def maxwell_fdtd_1d(nx, nt, dx, dt, E0=1.0, source_pos=0.1, wavelength=1e-6):
    """
    1D FDTD solution of Maxwell's equations.

    curl E = -dB/dt  =>  dEy/dx = -dBz/dt
    curl B = mu_0*eps_0*dE/dt  =>  dBz/dx = -mu_0*eps_0*dEy/dt

    For 1D wave propagating in x, Ey and Bz are nonzero.
    """
    # Fields
    Ey = np.zeros((nt, nx))
    Bz = np.zeros((nt, nx))

    # Source parameters
    omega = 2 * np.pi * C / wavelength
    source_idx = int(source_pos * nx)

    # Courant number (should be <= 1 for stability)
    S = C * dt / dx

    for n in range(nt - 1):
        # Time
        t = n * dt

        # Soft source (add to field rather than replace)
        Ey[n, source_idx] += E0 * np.sin(omega * t) * np.exp(-((t - 5/omega*2*np.pi) / (3/omega*2*np.pi))**2)

        # Update B (half-step staggered)
        for i in range(nx - 1):
            Bz[n+1, i] = Bz[n, i] - dt/dx * (Ey[n, i+1] - Ey[n, i])

        # Update E
        for i in range(1, nx):
            Ey[n+1, i] = Ey[n, i] - (C**2 * dt/dx) * (Bz[n+1, i] - Bz[n+1, i-1])

        # Absorbing boundary conditions (simple first-order)
        Ey[n+1, 0] = Ey[n, 1]
        Ey[n+1, -1] = Ey[n, -2]

    return Ey, Bz


def plane_wave_solution(x, t, E0, omega, k, phase=0):
    """
    Analytical plane wave solution.

    E_y = E0 * cos(kx - omega*t + phase)
    B_z = E0/c * cos(kx - omega*t + phase)
    """
    Ey = E0 * np.cos(k*x - omega*t + phase)
    Bz = E0/C * np.cos(k*x - omega*t + phase)
    return Ey, Bz


def poynting_vector(Ey, Bz):
    """
    Calculate Poynting vector S = E x B / mu_0.

    For 1D wave with Ey and Bz, S_x = Ey * Bz / mu_0
    """
    return Ey * Bz / MU_0


def main():
    fig = plt.figure(figsize=(16, 12))

    # Parameters
    wavelength = 1e-6  # 1 um (infrared)
    frequency = C / wavelength
    omega = 2 * np.pi * frequency
    k = omega / C
    E0 = 1.0  # 1 V/m

    # Simulation domain (in wavelengths)
    n_wavelengths = 5
    domain_size = n_wavelengths * wavelength
    nx = 500
    dx = domain_size / nx
    dt = 0.5 * dx / C  # CFL condition

    nt = 1000
    x = np.linspace(0, domain_size, nx)

    # Run FDTD simulation
    Ey, Bz = maxwell_fdtd_1d(nx, nt, dx, dt, E0, 0.1, wavelength)

    # Plot 1: E and B field snapshot
    ax1 = fig.add_subplot(2, 2, 1)

    t_snapshot = nt // 2
    time = t_snapshot * dt

    ax1.plot(x * 1e6, Ey[t_snapshot, :], 'b-', lw=2, label='E_y (V/m)')
    ax1.plot(x * 1e6, Bz[t_snapshot, :] * C * 1e9, 'r-', lw=2, label='B_z * c (nT * c)')

    ax1.set_xlabel('Position (um)')
    ax1.set_ylabel('Field amplitude')
    ax1.set_title(f'EM Wave: E and B Fields (t = {time*1e15:.1f} fs)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Space-time diagram
    ax2 = fig.add_subplot(2, 2, 2)

    t_arr = np.arange(nt) * dt * 1e15  # fs
    x_arr = x * 1e6  # um

    # Downsample for visualization
    stride = 4
    extent = [x_arr[0], x_arr[-1], t_arr[-1], t_arr[0]]
    im = ax2.imshow(Ey[::stride, :], aspect='auto', extent=extent, cmap='RdBu',
                    vmin=-E0, vmax=E0)
    plt.colorbar(im, ax=ax2, label='E_y (V/m)')

    # Add phase velocity line
    t_plot = np.array([0, t_arr[-1]])
    x_phase = C * t_plot * 1e-15 * 1e6 + 0.5  # Phase velocity trajectory
    ax2.plot(x_phase, t_plot, 'g--', lw=2, label='Phase velocity = c')

    ax2.set_xlabel('Position (um)')
    ax2.set_ylabel('Time (fs)')
    ax2.set_title('Space-Time Diagram (E_y)')
    ax2.legend()

    # Plot 3: Energy density and Poynting vector
    ax3 = fig.add_subplot(2, 2, 3)

    # Energy density: u = eps_0*E^2/2 + B^2/(2*mu_0)
    u_E = 0.5 * EPSILON_0 * Ey[t_snapshot, :]**2
    u_B = 0.5 * Bz[t_snapshot, :]**2 / MU_0
    u_total = u_E + u_B

    # Poynting vector
    S = poynting_vector(Ey[t_snapshot, :], Bz[t_snapshot, :])

    ax3.plot(x * 1e6, u_E * 1e12, 'b-', lw=2, label='Electric energy density')
    ax3.plot(x * 1e6, u_B * 1e12, 'r-', lw=2, label='Magnetic energy density')
    ax3.plot(x * 1e6, u_total * 1e12, 'k--', lw=1, label='Total energy density')

    ax3.set_xlabel('Position (um)')
    ax3.set_ylabel('Energy density (pJ/m^3)')
    ax3.set_title('Energy Distribution in EM Wave')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Verify equipartition
    ax3.text(0.95, 0.95, f'E energy / B energy = {np.mean(u_E)/np.mean(u_B):.3f}',
             transform=ax3.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 4: 3D visualization of E and B vectors
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')

    # Sample points along wave
    n_arrows = 30
    x_sample = x[::nx//n_arrows]
    Ey_sample = Ey[t_snapshot, ::nx//n_arrows]
    Bz_sample = Bz[t_snapshot, ::nx//n_arrows] * C  # Scale for visibility

    # Propagation direction (x)
    ax4.plot(x_sample * 1e6, np.zeros_like(x_sample), np.zeros_like(x_sample),
             'k-', lw=2, label='Propagation direction')

    # E field (y direction)
    for i in range(len(x_sample)):
        ax4.quiver(x_sample[i] * 1e6, 0, 0,
                   0, Ey_sample[i], 0,
                   color='blue', alpha=0.7, arrow_length_ratio=0.3)

    # B field (z direction)
    for i in range(len(x_sample)):
        ax4.quiver(x_sample[i] * 1e6, 0, 0,
                   0, 0, Bz_sample[i],
                   color='red', alpha=0.7, arrow_length_ratio=0.3)

    # Wave envelope
    ax4.plot(x_sample * 1e6, Ey_sample, np.zeros_like(x_sample), 'b-', lw=1, alpha=0.5)
    ax4.plot(x_sample * 1e6, np.zeros_like(x_sample), Bz_sample, 'r-', lw=1, alpha=0.5)

    ax4.set_xlabel('x (um)')
    ax4.set_ylabel('E_y (V/m)')
    ax4.set_zlabel('B_z * c (T * c)')
    ax4.set_title('3D Visualization: E and B Oscillations')

    # Add physics summary
    fig.text(0.5, 0.02,
             r"Maxwell's Equations in vacuum: "
             r"$\nabla \times \vec{E} = -\frac{\partial \vec{B}}{\partial t}$, "
             r"$\nabla \times \vec{B} = \mu_0\epsilon_0\frac{\partial \vec{E}}{\partial t}$"
             f'\nWave speed: c = 1/sqrt(mu_0*eps_0) = {C:.3e} m/s',
             ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.suptitle(f"Electromagnetic Wave from Maxwell's Equations\n"
                 f"wavelength = {wavelength*1e6:.1f} um, frequency = {frequency*1e-12:.1f} THz",
                 fontsize=14, y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)

    # Save plot
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'em_wave_maxwell.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
