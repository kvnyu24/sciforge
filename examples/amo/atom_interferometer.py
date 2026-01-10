"""
Experiment 251: Atom Interferometer Phase Shift

This example demonstrates atom interferometry using light-pulse techniques.
We explore the phase shifts induced by various physical effects and show
how atom interferometers measure acceleration, rotation, and gravity. Topics:
- Mach-Zehnder atom interferometer geometry
- Phase accumulation from inertial effects
- Gravity measurement (gravimeter)
- Sagnac effect for rotation sensing

Key physics:
- Beam splitter: pi/2 pulse transfers superposition of momentum states
- Mirror: pi pulse swaps momentum states
- Free evolution: momentum states separate and accumulate phase
- Phase shift from acceleration: phi = k_eff * a * T^2
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.amo import HBAR, KB, C, M_PROTON

def simulate_atom_interferometer():
    """Simulate Mach-Zehnder atom interferometer."""

    # Rubidium-87 parameters
    wavelength = 780e-9  # m
    k = 2 * np.pi / wavelength  # Wave vector
    k_eff = 2 * k  # Effective wave vector for two-photon transition
    m_Rb = 87 * M_PROTON  # Rb-87 mass
    v_recoil = HBAR * k / m_Rb  # Recoil velocity

    results = {}
    results['k_eff'] = k_eff
    results['m'] = m_Rb
    results['v_recoil'] = v_recoil
    results['wavelength'] = wavelength

    print(f"Effective wave vector k_eff: {k_eff:.2e} m^-1")
    print(f"Recoil velocity: {v_recoil*1e3:.3f} mm/s")

    # 1. Interferometer geometry
    print("\nSimulating Mach-Zehnder geometry...")

    # Time between pulses
    T = 100e-3  # 100 ms interrogation time (typical for gravimeters)
    g = 9.81  # m/s^2

    # Phase shift from gravity
    phi_g = k_eff * g * T**2
    print(f"Gravitational phase shift: {phi_g:.2f} rad = {phi_g/(2*np.pi):.2f} fringes")

    results['T'] = T
    results['g'] = g
    results['phi_g'] = phi_g

    # Trajectory calculation
    t_total = 2 * T
    t = np.linspace(0, t_total, 1000)

    # Upper arm: receives +hbar*k_eff at t=0 (after pi/2 pulse)
    # Lower arm: stays in original momentum state

    # Initial position z0=0, initial velocity v0=1 m/s (upward launch)
    v0 = 2.0  # m/s upward
    z0 = 0

    # Lower arm trajectory (free fall, no recoil)
    z_lower = z0 + v0 * t - 0.5 * g * t**2

    # Upper arm trajectory (receives recoil kick at t=0)
    z_upper = z0 + (v0 + v_recoil) * t - 0.5 * g * t**2

    # At t=T (pi pulse), arms swap momentum
    # After pi pulse, upper arm now has lower momentum
    t_pi = T
    idx_pi = np.argmin(np.abs(t - t_pi))

    z_upper_after = np.zeros_like(t)
    z_lower_after = np.zeros_like(t)
    z_upper_after[:idx_pi+1] = z_upper[:idx_pi+1]
    z_lower_after[:idx_pi+1] = z_lower[:idx_pi+1]

    # After pi pulse: swap trajectories (with continuity)
    z_at_pi_upper = z_upper[idx_pi]
    z_at_pi_lower = z_lower[idx_pi]
    v_upper_before = v0 + v_recoil - g * t_pi
    v_lower_before = v0 - g * t_pi

    # After swap: upper arm now moves with lower velocity, lower moves with upper
    for i in range(idx_pi + 1, len(t)):
        dt = t[i] - t_pi
        z_upper_after[i] = z_at_pi_upper + v_lower_before * dt - 0.5 * g * dt**2
        z_lower_after[i] = z_at_pi_lower + v_upper_before * dt - 0.5 * g * dt**2

    results['trajectories'] = {
        't': t,
        'z_upper': np.where(t <= t_pi, z_upper, z_upper_after),
        'z_lower': np.where(t <= t_pi, z_lower, z_lower_after),
        't_pi': t_pi
    }

    # 2. Phase shift vs interrogation time
    print("Computing phase shift vs interrogation time...")
    T_range = np.linspace(10e-3, 200e-3, 100)
    phi_vs_T = k_eff * g * T_range**2

    results['phase_vs_T'] = {
        'T': T_range,
        'phi': phi_vs_T
    }

    # 3. Sensitivity calculation
    # Shot noise limited sensitivity: delta_g / g = 1 / (k_eff * g * T^2 * sqrt(N))
    N_atoms = 1e6  # Number of atoms per shot
    cycle_time = 1.0  # 1 second per measurement

    delta_phi = 1 / np.sqrt(N_atoms)  # Phase sensitivity
    delta_g = delta_phi / (k_eff * T_range**2)  # Gravity sensitivity

    results['sensitivity'] = {
        'T': T_range,
        'delta_g': delta_g,
        'N_atoms': N_atoms
    }

    # 4. Interferometer fringes
    print("Computing interference fringes...")

    # Scan applied phase (e.g., by varying laser phase)
    phi_scan = np.linspace(0, 4 * np.pi, 200)

    # At output, probability of being in upper state:
    # P = (1 + cos(phi_total)) / 2
    # where phi_total = phi_g + phi_applied

    P_output = (1 + np.cos(phi_g + phi_scan)) / 2

    results['fringes'] = {
        'phi_scan': phi_scan,
        'P': P_output
    }

    # 5. Sagnac phase for rotation sensing
    print("Computing Sagnac effect...")

    # Sagnac phase: phi_Sagnac = 4 * m * A * Omega / hbar
    # where A is enclosed area, Omega is rotation rate

    A_enclosed = 0.01  # m^2 (10 cm x 10 cm effective area)
    Omega_range = np.linspace(0, 1e-5, 100)  # Earth rotation ~ 7e-5 rad/s

    phi_Sagnac = 4 * m_Rb * A_enclosed * Omega_range / HBAR

    results['sagnac'] = {
        'Omega': Omega_range,
        'phi': phi_Sagnac,
        'A': A_enclosed
    }

    # 6. Gravity gradient measurement (gradiometer)
    print("Computing gravity gradient sensitivity...")

    # Two interferometers separated by baseline L
    L_baseline = 1.0  # m
    T_grad = 100e-3  # interrogation time

    # Gravity gradient measurement
    # Delta_phi = k_eff * grad_g * L * T^2
    grad_g_range = np.linspace(0, 3e-6, 100)  # s^-2 (typical ~ 3e-6)

    delta_phi_grad = k_eff * grad_g_range * L_baseline * T_grad**2

    results['gradiometer'] = {
        'grad_g': grad_g_range,
        'delta_phi': delta_phi_grad,
        'L': L_baseline,
        'T': T_grad
    }

    return results


def plot_results(results):
    """Create comprehensive visualization of atom interferometry."""

    fig = plt.figure(figsize=(14, 12))

    # Plot 1: Mach-Zehnder trajectories
    ax1 = fig.add_subplot(2, 2, 1)
    traj = results['trajectories']

    ax1.plot(traj['t'] * 1e3, traj['z_upper'] * 1e3, 'b-', linewidth=2,
            label='Upper arm')
    ax1.plot(traj['t'] * 1e3, traj['z_lower'] * 1e3, 'r-', linewidth=2,
            label='Lower arm')

    # Mark pulse positions
    T = results['T']
    ax1.axvline(x=0, color='green', linestyle='--', alpha=0.7, label=r'$\pi/2$ pulse')
    ax1.axvline(x=T * 1e3, color='purple', linestyle='--', alpha=0.7, label=r'$\pi$ pulse')
    ax1.axvline(x=2 * T * 1e3, color='green', linestyle='--', alpha=0.7)

    ax1.set_xlabel('Time (ms)', fontsize=11)
    ax1.set_ylabel('Position (mm)', fontsize=11)
    ax1.set_title('Mach-Zehnder Atom Interferometer Trajectories', fontsize=12)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Add annotation about momentum states
    ax1.annotate(r'$|p\rangle + |p+\hbar k\rangle$', xy=(T*1e3/2, 100),
                fontsize=9, ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Interference fringes
    ax2 = fig.add_subplot(2, 2, 2)
    fr = results['fringes']

    ax2.plot(fr['phi_scan'] / np.pi, fr['P'], 'b-', linewidth=2)
    ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)

    # Mark gravitational phase
    phi_g = results['phi_g']
    ax2.axvline(x=phi_g / np.pi, color='red', linestyle='--', alpha=0.7,
               label=f'Gravity: {phi_g/(2*np.pi):.1f} fringes')

    ax2.set_xlabel(r'Applied Phase ($\pi$ rad)', fontsize=11)
    ax2.set_ylabel('Output Probability', fontsize=11)
    ax2.set_title(f'Interference Fringes (T = {T*1e3:.0f} ms)', fontsize=12)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_xlim(0, 4)
    ax2.grid(True, alpha=0.3)

    # Add inset showing fringe contrast
    ax2.text(0.05, 0.05, 'Fringe contrast ~ 100%\n(ideal case)',
            transform=ax2.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 3: Phase shift and sensitivity vs T
    ax3 = fig.add_subplot(2, 2, 3)
    pv = results['phase_vs_T']
    sv = results['sensitivity']

    ax3_twin = ax3.twinx()

    l1, = ax3.plot(pv['T'] * 1e3, pv['phi'] / (2 * np.pi), 'b-', linewidth=2,
                  label='Phase shift')
    l2, = ax3_twin.semilogy(sv['T'] * 1e3, sv['delta_g'] / results['g'] * 1e9, 'r-',
                           linewidth=2, label=r'$\delta g / g$')

    ax3.set_xlabel('Interrogation Time T (ms)', fontsize=11)
    ax3.set_ylabel('Phase Shift (fringes)', fontsize=11, color='b')
    ax3_twin.set_ylabel(r'Relative Sensitivity $\delta g/g$ (ppb)', fontsize=11, color='r')
    ax3.set_title('Gravimeter Phase Shift and Sensitivity', fontsize=12)

    # Combined legend
    ax3.legend([l1, l2], ['Phase shift', 'Sensitivity (ppb)'],
              loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='y', labelcolor='b')
    ax3_twin.tick_params(axis='y', labelcolor='r')

    # Plot 4: Various applications
    ax4 = fig.add_subplot(2, 2, 4)

    # Sagnac effect for rotation sensing
    sg = results['sagnac']
    ax4.plot(sg['Omega'] * 1e6, sg['phi'], 'g-', linewidth=2,
            label=f'Gyroscope (A = {sg["A"]*1e4:.0f} cm$^2$)')

    # Mark Earth rotation
    Omega_Earth = 7.27e-5  # rad/s
    phi_Earth = 4 * results['m'] * sg['A'] * Omega_Earth / HBAR
    ax4.axvline(x=Omega_Earth * 1e6, color='orange', linestyle='--', alpha=0.7)
    ax4.text(Omega_Earth * 1e6 * 1.1, max(sg['phi']) * 0.8,
            'Earth\nrotation', fontsize=8, color='orange')

    ax4.set_xlabel(r'Rotation Rate ($\mu$rad/s)', fontsize=11)
    ax4.set_ylabel('Sagnac Phase (rad)', fontsize=11)
    ax4.set_title('Sagnac Effect for Rotation Sensing', fontsize=12)
    ax4.legend(loc='upper left', fontsize=9)
    ax4.grid(True, alpha=0.3)

    # Add text about applications
    textstr = '\n'.join([
        'Applications:',
        '- Gravimeters: $\\delta g/g \\sim 10^{-9}$',
        '- Gyroscopes: $\\delta\\Omega \\sim nrad/s$',
        '- Gradiometers: gravity mapping',
        '- Tests of general relativity'
    ])
    ax4.text(0.05, 0.05, textstr, transform=ax4.transAxes, fontsize=9,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig


def main():
    print("=" * 60)
    print("Experiment 251: Atom Interferometer Phase Shift")
    print("=" * 60)
    print()

    # Run simulation
    print("Running simulations...")
    results = simulate_atom_interferometer()

    # Create visualization
    print("\nCreating visualization...")
    fig = plot_results(results)

    # Save figure
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'atom_interferometer.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary (Rb-87 Mach-Zehnder Interferometer):")
    print("=" * 60)
    print(f"Wavelength: {results['wavelength']*1e9:.0f} nm")
    print(f"Effective wave vector: {results['k_eff']:.2e} m^-1")
    print(f"Recoil velocity: {results['v_recoil']*1e3:.3f} mm/s")
    print()
    print(f"Interrogation time: {results['T']*1e3:.0f} ms")
    print(f"Gravitational phase shift: {results['phi_g']:.2f} rad = {results['phi_g']/(2*np.pi):.2f} fringes")
    print()
    print("Key phase shift formulas:")
    print("  Gravity: phi = k_eff * g * T^2")
    print("  Rotation (Sagnac): phi = 4 * m * A * Omega / hbar")
    print("  Gravity gradient: phi = k_eff * grad_g * L * T^2")
    print()
    print("Achievable sensitivities:")
    print("  - Gravity: delta_g/g ~ 10^-9 (ppb level)")
    print("  - Rotation: delta_Omega ~ nrad/s")
    print("  - Equivalent to optical interferometers at lambda ~ nm scale!")

    plt.close()


if __name__ == "__main__":
    main()
