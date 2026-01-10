"""
Experiment 262: Magnetic Reconnection Tearing Mode

Demonstrates magnetic reconnection and the tearing mode instability
that can disrupt current sheets.

Physical concepts:
- Reconnection changes magnetic field topology
- Sweet-Parker reconnection: slow, rate ~ S^(-1/2)
- Petschek reconnection: fast, rate ~ 1/ln(S)
- Tearing mode creates magnetic islands
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.plasma import MagneticReconnection

# Physical constants
mu_0 = 4 * np.pi * 1e-7
m_p = 1.673e-27


def harris_current_sheet(x, y, L, B0):
    """Harris current sheet equilibrium."""
    Bx = B0 * np.tanh(y / L)
    By = np.zeros_like(Bx)
    return Bx, By


def tearing_mode_perturbation(x, y, L, delta, k, amplitude):
    """Add tearing mode perturbation to Harris sheet."""
    # Psi perturbation
    psi = amplitude * np.exp(-(y / L)**2) * np.cos(k * x)

    # Magnetic field from psi
    delta_Bx = -np.gradient(psi, y, axis=0)
    delta_By = np.gradient(psi, x, axis=1)

    return delta_Bx, delta_By, psi


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Harris current sheet and X-point
    ax1 = axes[0, 0]

    L = 1.0  # Current sheet half-width
    B0 = 1.0  # Asymptotic field

    x = np.linspace(-5 * L, 5 * L, 100)
    y = np.linspace(-3 * L, 3 * L, 80)
    X, Y = np.meshgrid(x, y)

    Bx, By = harris_current_sheet(X, Y, L, B0)

    # Add small perturbation to show X-point formation
    k = 2 * np.pi / (4 * L)
    amplitude = 0.1 * B0
    delta_Bx, delta_By, psi = tearing_mode_perturbation(X, Y, L, L/10, k, amplitude)

    Bx_pert = Bx + delta_Bx
    By_pert = By + delta_By

    # Field magnitude
    B_mag = np.sqrt(Bx_pert**2 + By_pert**2)

    # Plot field lines
    contour = ax1.contour(X / L, Y / L, psi + B0 * L * np.log(np.cosh(Y / L)),
                          levels=20, colors='blue', linewidths=0.5)
    ax1.streamplot(X / L, Y / L, Bx_pert, By_pert, color='blue', density=1.5,
                   linewidth=0.5, arrowsize=0.5)

    # Mark X-point and O-point
    ax1.plot(0, 0, 'rx', markersize=15, markeredgewidth=3, label='X-point')
    ax1.plot(2, 0, 'go', markersize=10, label='O-point (island)')
    ax1.plot(-2, 0, 'go', markersize=10)

    # Current density (Jz ~ dBx/dy)
    Jz = np.gradient(Bx_pert, y, axis=0)
    cs = ax1.contourf(X / L, Y / L, Jz, levels=20, cmap='RdBu_r', alpha=0.3)
    plt.colorbar(cs, ax=ax1, label='Current density $J_z$')

    ax1.set_xlabel('$x / L$')
    ax1.set_ylabel('$y / L$')
    ax1.set_title('Magnetic Reconnection: X-point and Island Formation')
    ax1.legend(loc='upper right')
    ax1.set_aspect('equal')
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-3, 3)

    # Plot 2: Reconnection rate vs Lundquist number
    ax2 = axes[0, 1]

    # Typical solar corona parameters
    B_in = 1e-2  # 100 Gauss
    L_scale = 1e7  # 10 Mm
    n = 1e15  # m^-3
    rho = n * m_p

    # Resistivity range (anomalous to Spitzer)
    eta_range = np.logspace(-8, -2, 100)

    v_A = B_in / np.sqrt(mu_0 * rho)
    S_range = mu_0 * L_scale * v_A / eta_range

    sweet_parker = 1 / np.sqrt(S_range)
    petschek = np.pi / (8 * np.log(S_range))

    ax2.loglog(S_range, sweet_parker, 'b-', lw=2, label='Sweet-Parker: $M \\sim S^{-1/2}$')
    ax2.loglog(S_range, petschek, 'r--', lw=2, label='Petschek: $M \\sim (\\ln S)^{-1}$')

    # Observed rates
    ax2.axhline(y=0.1, color='green', linestyle=':', lw=2, label='Observed (~0.1)')

    # Mark typical S values
    S_solar = 1e12
    ax2.axvline(x=S_solar, color='gray', linestyle=':', alpha=0.7)
    ax2.text(S_solar, 1e-8, 'Solar corona', fontsize=9, rotation=90, va='bottom')

    ax2.set_xlabel('Lundquist Number $S = \\mu_0 L v_A / \\eta$')
    ax2.set_ylabel('Reconnection Rate $M = v_{in}/v_A$')
    ax2.set_title('Sweet-Parker vs Petschek Reconnection')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_xlim(1e4, 1e14)
    ax2.set_ylim(1e-8, 1)

    # Plot 3: Tearing mode island evolution
    ax3 = axes[1, 0]

    # Island width evolution
    times = [0.1, 0.5, 1.0, 2.0, 5.0]  # Normalized time
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(times)))

    x = np.linspace(-5 * L, 5 * L, 200)
    y = np.linspace(-2 * L, 2 * L, 100)
    X, Y = np.meshgrid(x, y)

    for t_norm, color in zip(times, colors):
        # Island width grows with time
        w = 0.1 * L * np.sqrt(t_norm)  # Linear phase growth

        # Simplified island structure
        amplitude = w * B0 / 2
        psi = amplitude * np.exp(-Y**2 / (L**2)) * np.cos(k * X)
        psi_eq = B0 * L * np.log(np.cosh(Y / L))

        # Plot separatrix
        ax3.contour(X / L, Y / L, psi_eq + psi, levels=[0],
                    colors=[color], linewidths=2)

    # Add colorbar-like legend
    sm = plt.cm.ScalarMappable(cmap='viridis',
                                norm=plt.Normalize(vmin=times[0], vmax=times[-1]))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax3)
    cbar.set_label('Normalized time')

    ax3.set_xlabel('$x / L$')
    ax3.set_ylabel('$y / L$')
    ax3.set_title('Magnetic Island Growth (Tearing Mode)')
    ax3.set_aspect('equal')
    ax3.set_xlim(-5, 5)
    ax3.set_ylim(-2, 2)

    # Plot 4: Energy release during reconnection
    ax4 = axes[1, 1]

    # Use actual reconnection model
    reconnection = MagneticReconnection(B_in, L_scale, rho, eta=1e-4)

    S = reconnection.lundquist_number
    rate_SP = reconnection.sweet_parker_rate()
    rate_P = reconnection.petschek_rate()

    # Energy conversion timeline
    t = np.linspace(0, 10, 100)  # Alfven times

    # Magnetic energy decay
    E_mag_SP = np.exp(-rate_SP * t)
    E_mag_P = np.exp(-rate_P * t)

    # Kinetic energy (bulk flow)
    E_kin_SP = 0.3 * (1 - E_mag_SP)
    E_kin_P = 0.3 * (1 - E_mag_P)

    # Thermal energy
    E_thermal_SP = 0.7 * (1 - E_mag_SP)
    E_thermal_P = 0.7 * (1 - E_mag_P)

    ax4.plot(t, E_mag_SP, 'b-', lw=2, label='Magnetic (SP)')
    ax4.plot(t, E_kin_SP, 'g-', lw=2, label='Kinetic (SP)')
    ax4.plot(t, E_thermal_SP, 'r-', lw=2, label='Thermal (SP)')

    ax4.plot(t, E_mag_P, 'b--', lw=2, label='Magnetic (Petschek)')
    ax4.plot(t, E_kin_P, 'g--', lw=2, label='Kinetic (Petschek)')
    ax4.plot(t, E_thermal_P, 'r--', lw=2, label='Thermal (Petschek)')

    ax4.set_xlabel('Time (Alfven times)')
    ax4.set_ylabel('Energy (normalized)')
    ax4.set_title('Energy Conversion During Reconnection')
    ax4.legend(loc='right', fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 1.1)

    # Add annotation
    textstr = f'$S$ = {S:.1e}\n$M_{{SP}}$ = {rate_SP:.2e}\n$M_P$ = {rate_P:.2e}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax4.text(0.02, 0.98, textstr, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    plt.suptitle('Experiment 262: Magnetic Reconnection and Tearing Mode\n'
                 'Topology change and energy release in current sheets',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'magnetic_reconnection_tearing.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'magnetic_reconnection_tearing.png')}")


if __name__ == "__main__":
    main()
