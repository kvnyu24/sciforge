"""
Experiment 250: Magneto-Optical Trap (MOT) Restoring Force

This example demonstrates the magneto-optical trap (MOT), which combines
optical molasses with a magnetic quadrupole field to create both velocity
and position-dependent forces. We explore:
- Zeeman-shifted resonance in magnetic field gradient
- Position-dependent restoring force
- MOT spring constant and trap frequency
- Atom cloud dynamics in the trap

Key physics:
- Magnetic field B(z) = B' * z (quadrupole gradient)
- Zeeman shift brings atom into resonance with appropriate beam
- Combined force: F = -kappa * x - beta * v (harmonic oscillator)
- Trap frequency: omega_trap = sqrt(kappa/m)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from src.sciforge.physics.amo import TwoLevelAtom, DopplerCooling, MagnetoOpticalTrap, HBAR, KB, C, M_PROTON

# Bohr magneton
MU_B = 9.274e-24  # J/T


def simulate_mot():
    """Simulate MOT restoring force and dynamics."""

    # Rubidium-87 D2 line parameters
    wavelength = 780e-9  # m
    omega_0 = 2 * np.pi * C / wavelength
    gamma = 2 * np.pi * 6.07e6  # Natural linewidth (rad/s)
    d = 2.5e-29  # Dipole moment (C*m)
    m_Rb = 87 * M_PROTON  # Rb-87 mass
    g_factor = 1.0  # Effective Lande g-factor

    # Create atom and cooling systems
    atom = TwoLevelAtom(omega_0=omega_0, d=d, gamma=gamma)
    cooling = DopplerCooling(atom=atom, wavelength=wavelength)
    k = cooling.k

    # MOT parameters
    dB_dz = 0.15  # T/m = 15 G/cm (typical)
    delta = -gamma / 2  # Red detuning
    s = 2.0  # Saturation parameter per beam

    # Create MOT
    mot = MagnetoOpticalTrap(cooling=cooling, dB_dz=dB_dz)

    results = {}
    results['gamma'] = gamma
    results['k'] = k
    results['m'] = m_Rb
    results['dB_dz'] = dB_dz
    results['delta'] = delta
    results['s'] = s

    # 1. Calculate MOT spring constant and trap frequency
    print("Computing MOT parameters...")
    kappa = mot.spring_constant(delta, s, g_factor)
    omega_trap = mot.trap_frequency(m_Rb, delta, s)

    results['kappa'] = kappa
    results['omega_trap'] = omega_trap
    results['T_trap'] = 2 * np.pi / omega_trap if omega_trap > 0 else np.inf

    print(f"Spring constant: {kappa:.2e} N/m")
    print(f"Trap frequency: {omega_trap/(2*np.pi):.2f} Hz")

    # 2. Position-dependent force in MOT
    print("\nComputing MOT force profile...")

    def mot_force(z, v, delta, s, dB_dz, g_factor):
        """
        Calculate MOT force including position and velocity dependence.

        The Zeeman shift modifies the effective detuning for each beam.
        """
        # Zeeman shift per unit position
        omega_Z = MU_B * g_factor * dB_dz / HBAR

        # Effective detuning for sigma+ and sigma- beams
        # sigma+ beam (from -z): sees atom at z with Zeeman shift
        # sigma- beam (from +z): opposite Zeeman shift
        delta_plus = delta - omega_Z * z - k * v
        delta_minus = delta + omega_Z * z + k * v

        # Scattering forces from each beam
        F_plus = (HBAR * k * gamma * s / 2) / (1 + s + (2 * delta_plus / gamma)**2)
        F_minus = (HBAR * k * gamma * s / 2) / (1 + s + (2 * delta_minus / gamma)**2)

        return F_plus - F_minus

    z_range = np.linspace(-5e-3, 5e-3, 500)  # -5 to 5 mm
    v = 0  # Static force

    F_mot = np.array([mot_force(z, v, delta, s, dB_dz, g_factor) for z in z_range])

    results['force_profile'] = {
        'z': z_range,
        'F': F_mot
    }

    # 3. Comparison with linear approximation
    F_linear = -kappa * z_range
    results['force_profile']['F_linear'] = F_linear

    # 4. Force vs position at various gradients
    print("Computing force at various gradients...")
    gradient_values = [0.05, 0.1, 0.15, 0.20, 0.25]  # T/m

    forces_gradient = []
    for grad in gradient_values:
        F = np.array([mot_force(z, 0, delta, s, grad, g_factor) for z in z_range])
        forces_gradient.append({
            'gradient': grad,
            'F': F
        })

    results['forces_gradient'] = forces_gradient

    # 5. Phase space trajectories (z, v)
    print("Simulating MOT dynamics...")

    def mot_dynamics(t, state):
        """Equations of motion in MOT."""
        z, v = state
        F = mot_force(z, v, delta, s, dB_dz, g_factor)
        a = F / m_Rb
        return [v, a]

    # Simulate atoms with different initial conditions
    initial_conditions = [
        (2e-3, 0),      # Displaced, stationary
        (0, 0.5),       # At center, moving
        (1e-3, 0.3),    # Mixed
        (-2e-3, -0.2),  # Other side
    ]

    t_max = 0.1  # 100 ms
    t_eval = np.linspace(0, t_max, 2000)

    trajectories = []
    for z0, v0 in initial_conditions:
        sol = solve_ivp(mot_dynamics, (0, t_max), [z0, v0],
                       t_eval=t_eval, method='RK45')
        trajectories.append({
            'z0': z0,
            'v0': v0,
            't': sol.t,
            'z': sol.y[0],
            'v': sol.y[1]
        })

    results['trajectories'] = trajectories

    # 6. Spring constant vs detuning
    print("Computing spring constant vs detuning...")
    delta_range = np.linspace(-5 * gamma, -0.1 * gamma, 100)

    kappa_vs_delta = []
    omega_vs_delta = []
    for d in delta_range:
        k_d = mot.spring_constant(d, s, g_factor)
        kappa_vs_delta.append(k_d)
        if k_d > 0:
            omega_vs_delta.append(np.sqrt(k_d / m_Rb))
        else:
            omega_vs_delta.append(0)

    results['kappa_vs_delta'] = {
        'delta': delta_range,
        'kappa': np.array(kappa_vs_delta),
        'omega': np.array(omega_vs_delta)
    }

    return results


def plot_results(results):
    """Create comprehensive visualization of MOT."""

    fig = plt.figure(figsize=(14, 12))
    gamma = results['gamma']

    # Plot 1: MOT force vs position
    ax1 = fig.add_subplot(2, 2, 1)
    fp = results['force_profile']

    # Scale for visibility
    F_max = np.max(np.abs(fp['F']))
    ax1.plot(fp['z'] * 1e3, fp['F'] / F_max, 'b-', linewidth=2, label='MOT force')
    ax1.plot(fp['z'] * 1e3, fp['F_linear'] / F_max, 'r--', linewidth=1.5,
            label='Linear: $F = -\\kappa z$')

    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

    ax1.set_xlabel('Position (mm)', fontsize=11)
    ax1.set_ylabel('Force (normalized)', fontsize=11)
    ax1.set_title(f'MOT Restoring Force ($\\delta = -\\gamma/2$, $dB/dz$ = 15 G/cm)', fontsize=12)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Add spring constant
    kappa = results['kappa']
    omega = results['omega_trap']
    textstr = f'$\\kappa$ = {kappa:.2e} N/m\n$\\omega/2\\pi$ = {omega/(2*np.pi):.1f} Hz'
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Force at various gradients
    ax2 = fig.add_subplot(2, 2, 2)
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(results['forces_gradient'])))

    for i, fg in enumerate(results['forces_gradient']):
        grad_Gcm = fg['gradient'] * 100  # Convert T/m to G/cm
        label = f'{grad_Gcm:.0f} G/cm'
        ax2.plot(fp['z'] * 1e3, fg['F'] * 1e23, color=colors[i], linewidth=1.5, label=label)

    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Position (mm)', fontsize=11)
    ax2.set_ylabel(r'Force ($10^{-23}$ N)', fontsize=11)
    ax2.set_title('MOT Force at Various Magnetic Field Gradients', fontsize=12)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Phase space trajectories
    ax3 = fig.add_subplot(2, 2, 3)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, traj in enumerate(results['trajectories']):
        ax3.plot(traj['z'] * 1e3, traj['v'] * 100, color=colors[i], linewidth=1,
                alpha=0.8, label=f'$z_0$={traj["z0"]*1e3:.0f}mm, $v_0$={traj["v0"]*100:.0f}cm/s')
        # Mark start
        ax3.plot(traj['z'][0] * 1e3, traj['v'][0] * 100, 'o', color=colors[i], markersize=8)
        # Mark end
        ax3.plot(traj['z'][-1] * 1e3, traj['v'][-1] * 100, 's', color=colors[i], markersize=6)

    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax3.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

    ax3.set_xlabel('Position (mm)', fontsize=11)
    ax3.set_ylabel('Velocity (cm/s)', fontsize=11)
    ax3.set_title('Phase Space Trajectories in MOT (100 ms)', fontsize=12)
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Add expected damped oscillation frequency
    T_trap = results['T_trap']
    ax3.text(0.05, 0.05, f'Expected period: {T_trap*1e3:.1f} ms',
            transform=ax3.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 4: Position vs time
    ax4 = fig.add_subplot(2, 2, 4)

    for i, traj in enumerate(results['trajectories']):
        ax4.plot(traj['t'] * 1e3, traj['z'] * 1e3, color=colors[i], linewidth=1.5,
                label=f'$z_0$={traj["z0"]*1e3:.0f}mm')

    ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax4.set_xlabel('Time (ms)', fontsize=11)
    ax4.set_ylabel('Position (mm)', fontsize=11)
    ax4.set_title('Atom Position vs Time in MOT', fontsize=12)
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.3)

    # Add inset showing spring constant vs detuning
    ax4_inset = ax4.inset_axes([0.55, 0.55, 0.4, 0.35])
    kvd = results['kappa_vs_delta']
    ax4_inset.plot(-kvd['delta'] / gamma, kvd['kappa'] * 1e16, 'b-', linewidth=1.5)
    ax4_inset.axvline(x=0.5, color='red', linestyle=':', alpha=0.7)
    ax4_inset.set_xlabel(r'$-\delta/\gamma$', fontsize=8)
    ax4_inset.set_ylabel(r'$\kappa$ ($10^{-16}$ N/m)', fontsize=8)
    ax4_inset.set_title('Spring constant vs detuning', fontsize=8)
    ax4_inset.grid(True, alpha=0.3)
    ax4_inset.set_xlim(0, 5)

    plt.tight_layout()
    return fig


def main():
    print("=" * 60)
    print("Experiment 250: Magneto-Optical Trap (MOT) Restoring Force")
    print("=" * 60)
    print()

    # Run simulation
    print("Running simulations...")
    results = simulate_mot()

    # Create visualization
    print("\nCreating visualization...")
    fig = plot_results(results)

    # Save figure
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'mot_restoring_force.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    # Print summary
    gamma = results['gamma']
    print("\n" + "=" * 60)
    print("Summary (Rb-87 D2 line MOT):")
    print("=" * 60)
    print(f"Magnetic field gradient: {results['dB_dz']*100:.0f} G/cm")
    print(f"Laser detuning: {results['delta']/gamma:.1f} * gamma")
    print(f"Saturation parameter: {results['s']:.0f}")
    print()
    print("MOT parameters:")
    print(f"  Spring constant kappa: {results['kappa']:.2e} N/m")
    print(f"  Trap frequency: {results['omega_trap']/(2*np.pi):.2f} Hz")
    print(f"  Oscillation period: {results['T_trap']*1e3:.1f} ms")
    print()
    print("The MOT provides:")
    print("  1. Position-dependent restoring force (from Zeeman shift)")
    print("  2. Velocity-dependent damping (from optical molasses)")
    print("  3. Resulting in damped harmonic oscillator dynamics")

    plt.close()


if __name__ == "__main__":
    main()
