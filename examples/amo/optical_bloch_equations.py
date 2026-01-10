"""
Experiment 246: Optical Bloch Equations

This example demonstrates the dynamics of a two-level atom interacting with a
coherent light field, described by the optical Bloch equations. We explore:
- Coherent Rabi oscillations at various detunings
- Damped dynamics due to spontaneous emission
- Steady-state populations and coherences
- Bloch sphere representation of the atomic state

The optical Bloch equations describe the time evolution of the density matrix
elements (populations and coherences) for a two-level atom driven by a laser.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.amo import TwoLevelAtom, BlochEquations, HBAR

def simulate_bloch_dynamics():
    """Simulate optical Bloch equation dynamics for various parameters."""

    # Rubidium D2 line parameters (780 nm)
    omega_0 = 2 * np.pi * 384e12  # Transition frequency (rad/s)
    gamma = 2 * np.pi * 6.07e6    # Natural linewidth (rad/s)
    d = 2.5e-29                    # Dipole moment (C*m)

    # Create two-level atom
    atom = TwoLevelAtom(omega_0=omega_0, d=d, gamma=gamma)
    bloch = BlochEquations(atom)

    # Time span in units of 1/gamma
    t_max = 10 / gamma
    n_points = 500

    # Initial state: ground state
    rho_0 = np.array([[1, 0], [0, 0]], dtype=complex)

    results = {}

    # Case 1: On-resonance driving at various Rabi frequencies
    print("Simulating on-resonance dynamics...")
    Omega_values = [0.5 * gamma, gamma, 2 * gamma]
    delta = 0  # On resonance

    results['on_resonance'] = []
    for Omega in Omega_values:
        t, rho_t = bloch.evolve(rho_0, Omega, delta, (0, t_max), n_points)
        # Extract excited state population
        P_e = np.array([np.real(rho[1, 1]) for rho in rho_t])
        results['on_resonance'].append({
            'Omega': Omega,
            't': t,
            'P_e': P_e,
            'rho': rho_t
        })

    # Case 2: Off-resonance driving
    print("Simulating off-resonance dynamics...")
    Omega = gamma
    delta_values = [0, gamma, 2 * gamma, 5 * gamma]

    results['off_resonance'] = []
    for delta in delta_values:
        t, rho_t = bloch.evolve(rho_0, Omega, delta, (0, t_max), n_points)
        P_e = np.array([np.real(rho[1, 1]) for rho in rho_t])
        results['off_resonance'].append({
            'delta': delta,
            't': t,
            'P_e': P_e,
            'rho': rho_t
        })

    # Case 3: Bloch vector evolution (u, v, w components)
    print("Computing Bloch vector evolution...")
    Omega = gamma
    delta = 0.5 * gamma
    t, rho_t = bloch.evolve(rho_0, Omega, delta, (0, t_max), n_points)

    # Bloch vector: u = Re(rho_ge), v = Im(rho_ge), w = (rho_ee - rho_gg)/2
    u = np.array([2 * np.real(rho[0, 1]) for rho in rho_t])
    v = np.array([2 * np.imag(rho[0, 1]) for rho in rho_t])
    w = np.array([np.real(rho[1, 1] - rho[0, 0]) for rho in rho_t])

    results['bloch_vector'] = {
        't': t,
        'u': u,
        'v': v,
        'w': w,
        'Omega': Omega,
        'delta': delta
    }

    # Case 4: Steady-state population vs detuning
    print("Computing steady-state spectrum...")
    delta_range = np.linspace(-10 * gamma, 10 * gamma, 200)
    Omega_ss = gamma  # Fixed Rabi frequency

    P_e_ss = np.array([bloch.steady_state_population(Omega_ss, d) for d in delta_range])

    results['steady_state'] = {
        'delta': delta_range,
        'P_e': P_e_ss,
        'Omega': Omega_ss
    }

    results['gamma'] = gamma

    return results


def plot_results(results):
    """Create comprehensive visualization of Bloch equation dynamics."""

    fig = plt.figure(figsize=(14, 12))
    gamma = results['gamma']

    # Plot 1: On-resonance Rabi oscillations
    ax1 = fig.add_subplot(2, 2, 1)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, data in enumerate(results['on_resonance']):
        Omega = data['Omega']
        label = f'$\\Omega = {Omega/gamma:.1f}\\gamma$'
        ax1.plot(data['t'] * gamma, data['P_e'], color=colors[i], label=label, linewidth=1.5)

    ax1.set_xlabel(r'Time ($\gamma^{-1}$)', fontsize=11)
    ax1.set_ylabel(r'Excited State Population $P_e$', fontsize=11)
    ax1.set_title('On-Resonance Rabi Oscillations with Damping', fontsize=12)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 0.6)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Off-resonance dynamics
    ax2 = fig.add_subplot(2, 2, 2)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, data in enumerate(results['off_resonance']):
        delta = data['delta']
        label = f'$\\delta = {delta/gamma:.0f}\\gamma$'
        ax2.plot(data['t'] * gamma, data['P_e'], color=colors[i], label=label, linewidth=1.5)

    ax2.set_xlabel(r'Time ($\gamma^{-1}$)', fontsize=11)
    ax2.set_ylabel(r'Excited State Population $P_e$', fontsize=11)
    ax2.set_title(r'Effect of Detuning ($\Omega = \gamma$)', fontsize=12)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 0.6)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Bloch vector evolution
    ax3 = fig.add_subplot(2, 2, 3)
    bv = results['bloch_vector']
    ax3.plot(bv['t'] * gamma, bv['u'], 'b-', label=r'$u$ (in-phase)', linewidth=1.5)
    ax3.plot(bv['t'] * gamma, bv['v'], 'r-', label=r'$v$ (quadrature)', linewidth=1.5)
    ax3.plot(bv['t'] * gamma, bv['w'], 'g-', label=r'$w$ (inversion)', linewidth=1.5)

    ax3.set_xlabel(r'Time ($\gamma^{-1}$)', fontsize=11)
    ax3.set_ylabel('Bloch Vector Components', fontsize=11)
    ax3.set_title(f'Bloch Vector Evolution ($\\Omega = \\gamma$, $\\delta = 0.5\\gamma$)', fontsize=12)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.set_xlim(0, 10)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Steady-state spectrum (Lorentzian)
    ax4 = fig.add_subplot(2, 2, 4)
    ss = results['steady_state']
    ax4.plot(ss['delta'] / gamma, ss['P_e'], 'b-', linewidth=2)

    # Add theoretical half-maximum width annotation
    max_pop = np.max(ss['P_e'])
    half_max = max_pop / 2
    ax4.axhline(y=half_max, color='r', linestyle='--', alpha=0.5, label='Half maximum')

    ax4.set_xlabel(r'Detuning $\delta/\gamma$', fontsize=11)
    ax4.set_ylabel(r'Steady-State Population $P_e$', fontsize=11)
    ax4.set_title(f'Steady-State Absorption Spectrum ($\\Omega = \\gamma$)', fontsize=12)
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.3)

    # Add text annotation about power broadening
    s = 2 * (ss['Omega'] / gamma)**2  # Saturation parameter
    broadened_width = gamma * np.sqrt(1 + s)
    ax4.annotate(f'Power-broadened FWHM: {broadened_width/gamma:.2f}$\\gamma$',
                xy=(0.05, 0.05), xycoords='axes fraction', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig


def main():
    print("=" * 60)
    print("Experiment 246: Optical Bloch Equations")
    print("=" * 60)
    print()

    # Run simulation
    print("Running simulations...")
    results = simulate_bloch_dynamics()

    # Create visualization
    print("\nCreating visualization...")
    fig = plot_results(results)

    # Save figure
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'optical_bloch_equations.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    # Print summary statistics
    gamma = results['gamma']
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"Natural linewidth gamma: {gamma/(2*np.pi)/1e6:.2f} MHz")
    print(f"Spontaneous lifetime: {1/gamma*1e9:.1f} ns")
    print(f"\nSteady-state maximum population: {np.max(results['steady_state']['P_e']):.3f}")
    print("  (Limited to 0.5 at saturation, lower due to finite Omega)")

    # Show on-resonance steady-state
    ss_on_res = results['steady_state']['P_e'][len(results['steady_state']['P_e'])//2]
    print(f"\nOn-resonance steady-state P_e: {ss_on_res:.3f}")

    plt.close()


if __name__ == "__main__":
    main()
