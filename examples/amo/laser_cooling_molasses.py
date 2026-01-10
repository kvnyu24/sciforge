"""
Experiment 249: Laser Cooling - Optical Molasses

This example demonstrates optical molasses cooling, where counter-propagating
laser beams create a velocity-dependent friction force that slows atoms. We explore:
- The scattering force from red-detuned counter-propagating beams
- Velocity-dependent cooling force (optical molasses)
- Doppler cooling limit temperature
- Dynamics of cooling from thermal velocities
- Momentum diffusion and heating

Key physics:
- Red detuning (delta < 0): atoms moving toward a beam see it blue-shifted closer to resonance
- Net force F = -beta * v (friction at low velocities)
- Equilibrium: Doppler temperature T_D = hbar * gamma / (2 * k_B)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from src.sciforge.physics.amo import TwoLevelAtom, DopplerCooling, HBAR, KB, C, M_PROTON

def simulate_optical_molasses():
    """Simulate optical molasses cooling dynamics."""

    # Rubidium-87 D2 line parameters
    wavelength = 780e-9  # m
    omega_0 = 2 * np.pi * C / wavelength
    gamma = 2 * np.pi * 6.07e6  # Natural linewidth (rad/s)
    d = 2.5e-29  # Dipole moment (C*m)
    m_Rb = 87 * M_PROTON  # Rb-87 mass

    # Create atom and cooling system
    atom = TwoLevelAtom(omega_0=omega_0, d=d, gamma=gamma)
    cooling = DopplerCooling(atom=atom, wavelength=wavelength)

    # Doppler temperature limit
    T_D = cooling.doppler_temperature()
    v_D = np.sqrt(2 * KB * T_D / m_Rb)  # Corresponding velocity scale
    k = cooling.k  # Wave vector

    results = {}
    results['gamma'] = gamma
    results['T_D'] = T_D
    results['v_D'] = v_D
    results['k'] = k
    results['m'] = m_Rb
    results['wavelength'] = wavelength

    print(f"Doppler temperature: {T_D*1e6:.1f} uK")
    print(f"Doppler velocity: {v_D*100:.2f} cm/s")

    # 1. Scattering force vs velocity for single beam
    print("\nComputing scattering forces...")
    v_range = np.linspace(-20, 20, 500)  # m/s
    delta = -gamma / 2  # Optimal detuning for cooling
    s = 1.0  # Saturation parameter

    F_single = np.array([cooling.scattering_force(v, delta, s) for v in v_range])
    F_cooling = np.array([cooling.cooling_force(v, delta, s) for v in v_range])

    results['force_vs_velocity'] = {
        'v': v_range,
        'F_single': F_single,
        'F_cooling': F_cooling,
        'delta': delta,
        's': s
    }

    # 2. Force at various detunings
    print("Computing force for various detunings...")
    delta_values = np.array([-0.25, -0.5, -1, -2, -4]) * gamma
    forces_detuning = []

    for delta in delta_values:
        F = np.array([cooling.cooling_force(v, delta, s) for v in v_range])
        forces_detuning.append({
            'delta': delta,
            'F': F
        })

    results['forces_detuning'] = forces_detuning

    # 3. Cooling dynamics simulation
    print("Simulating cooling dynamics...")

    def equations_of_motion(t, state, delta, s):
        """Equations of motion for 1D cooling."""
        x, v = state

        # Deterministic cooling force
        F_cool = cooling.cooling_force(v, delta, s)

        # Acceleration
        a = F_cool / m_Rb

        return [v, a]

    # Initial conditions: thermal velocity distribution at 300 K
    T_initial = 300  # K
    v_thermal = np.sqrt(2 * KB * T_initial / m_Rb)  # ~240 m/s for Rb

    # Simulate multiple atoms with different initial velocities
    n_atoms = 20
    np.random.seed(42)
    v_initials = np.random.normal(0, v_thermal, n_atoms)

    # Time span - enough for significant cooling
    t_max = 1e-3  # 1 ms
    t_eval = np.linspace(0, t_max, 1000)

    delta_cool = -gamma / 2
    s_cool = 2.0

    trajectories = []
    for v0 in v_initials:
        sol = solve_ivp(equations_of_motion, (0, t_max), [0, v0],
                       t_eval=t_eval, args=(delta_cool, s_cool),
                       method='RK45')
        trajectories.append({
            'v0': v0,
            't': sol.t,
            'v': sol.y[1],
            'x': sol.y[0]
        })

    results['trajectories'] = trajectories
    results['t_eval'] = t_eval

    # 4. Temperature vs time (from velocity distribution)
    print("Computing temperature evolution...")

    # Calculate effective temperature at each time step
    velocities_t = np.array([traj['v'] for traj in trajectories])
    T_t = m_Rb * np.var(velocities_t, axis=0) / KB

    results['temperature'] = {
        't': t_eval,
        'T': T_t
    }

    # 5. Damping coefficient at optimal detuning
    print("Computing damping coefficients...")
    delta_range = np.linspace(-5 * gamma, -0.1 * gamma, 100)
    beta_values = np.array([cooling.damping_coefficient(d, s) for d in delta_range])

    results['damping'] = {
        'delta': delta_range,
        'beta': beta_values
    }

    # Find optimal detuning
    idx_max = np.argmax(beta_values)
    results['optimal_detuning'] = delta_range[idx_max]
    results['max_beta'] = beta_values[idx_max]

    return results


def plot_results(results):
    """Create comprehensive visualization of optical molasses."""

    fig = plt.figure(figsize=(14, 12))
    gamma = results['gamma']
    T_D = results['T_D']
    v_D = results['v_D']

    # Plot 1: Cooling force vs velocity
    ax1 = fig.add_subplot(2, 2, 1)
    fv = results['force_vs_velocity']

    # Scale force by maximum scattering force
    F_max = HBAR * results['k'] * gamma / 2

    ax1.plot(fv['v'], fv['F_single'] / F_max, 'b--', linewidth=1.5, alpha=0.7,
            label=r'Single beam ($+k$)')
    ax1.plot(fv['v'], -fv['F_single'] / F_max, 'r--', linewidth=1.5, alpha=0.7,
            label=r'Single beam ($-k$)')
    ax1.plot(fv['v'], fv['F_cooling'] / F_max, 'g-', linewidth=2.5,
            label='Net cooling force')

    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

    # Mark capture velocity
    v_cap = gamma / results['k']
    ax1.axvline(x=v_cap, color='purple', linestyle=':', alpha=0.7, label=f'$v_c = \\gamma/k$')
    ax1.axvline(x=-v_cap, color='purple', linestyle=':', alpha=0.7)

    ax1.set_xlabel('Velocity (m/s)', fontsize=11)
    ax1.set_ylabel(r'Force / $F_{max}$', fontsize=11)
    ax1.set_title(f'Optical Molasses Force ($\\delta = -\\gamma/2$, $s = 1$)', fontsize=12)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_xlim(-20, 20)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Force at various detunings
    ax2 = fig.add_subplot(2, 2, 2)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(results['forces_detuning'])))

    for i, fd in enumerate(results['forces_detuning']):
        label = f'$\\delta = {fd["delta"]/gamma:.2f}\\gamma$'
        ax2.plot(fv['v'], fd['F'] / F_max, color=colors[i], linewidth=1.5, label=label)

    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Velocity (m/s)', fontsize=11)
    ax2.set_ylabel(r'Force / $F_{max}$', fontsize=11)
    ax2.set_title('Cooling Force at Different Detunings', fontsize=12)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_xlim(-10, 10)
    ax2.grid(True, alpha=0.3)

    # Add text about optimal detuning
    ax2.text(0.05, 0.05, f'Optimal: $\\delta \\approx -\\gamma/2$\nfor max. damping',
            transform=ax2.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 3: Velocity trajectories during cooling
    ax3 = fig.add_subplot(2, 2, 3)
    for traj in results['trajectories']:
        ax3.plot(traj['t'] * 1e6, traj['v'], alpha=0.5, linewidth=0.8)

    # Mark capture velocity
    v_cap = gamma / results['k']
    ax3.axhline(y=v_cap, color='purple', linestyle='--', alpha=0.5, label='Capture velocity')
    ax3.axhline(y=-v_cap, color='purple', linestyle='--', alpha=0.5)
    ax3.axhline(y=v_D, color='red', linestyle=':', alpha=0.7, label=f'Doppler velocity')
    ax3.axhline(y=-v_D, color='red', linestyle=':', alpha=0.7)

    ax3.set_xlabel('Time ($\\mu$s)', fontsize=11)
    ax3.set_ylabel('Velocity (m/s)', fontsize=11)
    ax3.set_title('Cooling Trajectories (20 atoms from 300 K)', fontsize=12)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Temperature evolution
    ax4 = fig.add_subplot(2, 2, 4)
    temp = results['temperature']

    ax4.semilogy(temp['t'] * 1e6, temp['T'] * 1e6, 'b-', linewidth=2)
    ax4.axhline(y=T_D * 1e6, color='red', linestyle='--', linewidth=2,
               label=f'Doppler limit: {T_D*1e6:.1f} $\\mu$K')

    ax4.set_xlabel('Time ($\\mu$s)', fontsize=11)
    ax4.set_ylabel('Temperature ($\\mu$K)', fontsize=11)
    ax4.set_title('Temperature During Cooling', fontsize=12)
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.3, which='both')

    # Add inset showing damping coefficient
    ax4_inset = ax4.inset_axes([0.55, 0.55, 0.4, 0.35])
    damp = results['damping']
    ax4_inset.plot(-damp['delta'] / gamma, damp['beta'] * 1e24, 'g-', linewidth=2)
    ax4_inset.axvline(x=0.5, color='red', linestyle=':', alpha=0.7)
    ax4_inset.set_xlabel(r'$-\delta/\gamma$', fontsize=8)
    ax4_inset.set_ylabel(r'$\beta$ ($10^{-24}$ kg/s)', fontsize=8)
    ax4_inset.set_title('Damping coefficient', fontsize=8)
    ax4_inset.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    print("=" * 60)
    print("Experiment 249: Laser Cooling - Optical Molasses")
    print("=" * 60)
    print()

    # Run simulation
    print("Running simulations...")
    results = simulate_optical_molasses()

    # Create visualization
    print("\nCreating visualization...")
    fig = plot_results(results)

    # Save figure
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'laser_cooling_molasses.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    # Print summary
    gamma = results['gamma']
    T_D = results['T_D']
    print("\n" + "=" * 60)
    print("Summary (Rb-87 D2 line):")
    print("=" * 60)
    print(f"Wavelength: {results['wavelength']*1e9:.1f} nm")
    print(f"Natural linewidth: {gamma/(2*np.pi*1e6):.2f} MHz")
    print(f"Recoil velocity: {HBAR * results['k'] / results['m'] * 100:.3f} cm/s")
    print()
    print("Doppler cooling limit:")
    print(f"  Temperature: {T_D*1e6:.1f} uK")
    print(f"  RMS velocity: {results['v_D']*100:.2f} cm/s")
    print()
    print(f"Optimal detuning: {results['optimal_detuning']/gamma:.2f} * gamma")
    print(f"Capture velocity: {gamma/results['k']:.2f} m/s")
    print()
    print("Note: Real cooling requires 6 beams (3D) and includes")
    print("  momentum diffusion heating, leading to Doppler limit.")

    plt.close()


if __name__ == "__main__":
    main()
