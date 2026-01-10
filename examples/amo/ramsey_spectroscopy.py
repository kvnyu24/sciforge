"""
Experiment 252: Ramsey Spectroscopy Linewidth

This example demonstrates Ramsey spectroscopy, where two separated pi/2 pulses
create an interference pattern with much narrower linewidth than a single pulse.
We explore:
- Ramsey fringe pattern and central fringe
- Linewidth narrowing: Delta_nu ~ 1/(2*T) vs 1/tau for Rabi
- Time-domain interpretation of frequency sensitivity
- Effects of decoherence and dephasing
- Application to atomic clocks

Key physics:
- First pi/2 pulse: creates superposition |g> + |e>
- Free evolution: phase accumulates phi = delta * T
- Second pi/2 pulse: converts phase to population
- Central fringe FWHM: Delta_nu = 1/(2*T)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.amo import HBAR, KB

def ramsey_fringes(delta, T, tau, contrast=1.0, decay_time=None):
    """
    Calculate Ramsey fringe pattern.

    Args:
        delta: Detuning from resonance (rad/s)
        T: Free evolution time (s)
        tau: Pulse duration (s)
        contrast: Fringe contrast (0 to 1)
        decay_time: T2 coherence time (optional)

    Returns:
        Excited state probability
    """
    # Phase accumulated during free evolution
    phi = delta * T

    # Ramsey signal (ideal case: perfect pi/2 pulses)
    # P_e = (1/2) * [1 + C * cos(delta * T)]
    signal = 0.5 * (1 + contrast * np.cos(phi))

    # Add decoherence if specified
    if decay_time is not None:
        decay = np.exp(-T / decay_time)
        signal = 0.5 * (1 + contrast * decay * np.cos(phi))

    return signal


def rabi_lineshape(delta, Omega, tau):
    """
    Calculate Rabi excitation probability.

    For a pi/2 pulse: tau = pi / (2 * Omega)

    Args:
        delta: Detuning (rad/s)
        Omega: Rabi frequency (rad/s)
        tau: Pulse duration (s)

    Returns:
        Excited state probability
    """
    Omega_eff = np.sqrt(Omega**2 + delta**2)
    return (Omega / Omega_eff)**2 * np.sin(Omega_eff * tau / 2)**2


def simulate_ramsey_spectroscopy():
    """Simulate Ramsey spectroscopy."""

    results = {}

    # Parameters for atomic clock transition
    # Example: Cs hyperfine transition at 9.192631770 GHz
    omega_0 = 2 * np.pi * 9.192631770e9  # Transition frequency
    gamma_nat = 0  # Effectively zero for clock transition

    # Typical atomic fountain parameters
    T = 0.5  # 500 ms free evolution (atomic fountain)
    tau = 1e-3  # 1 ms pulse duration
    Omega = np.pi / (2 * tau)  # Pi/2 pulse Rabi frequency

    results['T'] = T
    results['tau'] = tau
    results['Omega'] = Omega

    # 1. Ramsey fringes
    print("Computing Ramsey fringes...")
    delta_range_wide = np.linspace(-20 / T, 20 / T, 2000)  # +/- 20 fringes
    delta_range_narrow = np.linspace(-3 / T, 3 / T, 1000)  # Central region

    P_ramsey_wide = ramsey_fringes(delta_range_wide, T, tau)
    P_ramsey_narrow = ramsey_fringes(delta_range_narrow, T, tau)

    results['fringes'] = {
        'delta_wide': delta_range_wide,
        'P_wide': P_ramsey_wide,
        'delta_narrow': delta_range_narrow,
        'P_narrow': P_ramsey_narrow
    }

    # 2. Comparison with Rabi spectroscopy
    print("Computing Rabi comparison...")
    # Single pulse of same total duration
    tau_single = T + 2 * tau  # Approximately same total time

    # Rabi linewidth ~ Omega (for pi pulse)
    delta_rabi = np.linspace(-5 / tau, 5 / tau, 500)
    P_rabi = rabi_lineshape(delta_rabi, Omega, tau)

    results['rabi'] = {
        'delta': delta_rabi,
        'P': P_rabi,
        'tau': tau
    }

    # 3. Linewidth analysis
    print("Analyzing linewidths...")

    # Ramsey FWHM = 1/(2*T) in Hz, or pi/T in rad/s
    fwhm_ramsey = 1 / (2 * T)  # Hz
    fwhm_rabi = Omega / (2 * np.pi)  # Approximate FWHM for Rabi

    results['linewidths'] = {
        'ramsey': fwhm_ramsey,
        'rabi': fwhm_rabi,
        'ratio': fwhm_rabi / fwhm_ramsey
    }

    print(f"Ramsey FWHM: {fwhm_ramsey:.2f} Hz")
    print(f"Rabi FWHM: {fwhm_rabi:.2f} Hz")
    print(f"Narrowing ratio: {fwhm_rabi/fwhm_ramsey:.0f}x")

    # 4. Effect of evolution time T
    print("Computing T-dependence...")
    T_values = np.array([0.01, 0.05, 0.1, 0.5, 1.0])  # seconds

    T_dependence = []
    for T_val in T_values:
        delta_grid = np.linspace(-10 / T_val, 10 / T_val, 500)
        P = ramsey_fringes(delta_grid, T_val, tau)
        T_dependence.append({
            'T': T_val,
            'delta': delta_grid,
            'P': P,
            'fwhm': 1 / (2 * T_val)
        })

    results['T_dependence'] = T_dependence

    # 5. Decoherence effects
    print("Computing decoherence effects...")
    T2_values = [0.1, 0.5, 1.0, np.inf]  # T2 coherence times
    T_deco = 0.5  # Fixed evolution time

    decoherence = []
    for T2 in T2_values:
        P = ramsey_fringes(delta_range_narrow, T_deco, tau,
                          decay_time=T2 if T2 != np.inf else None)
        decoherence.append({
            'T2': T2,
            'P': P
        })

    results['decoherence'] = {
        'delta': delta_range_narrow,
        'T': T_deco,
        'data': decoherence
    }

    # 6. Clock stability
    print("Computing clock stability...")
    # Allan deviation sigma_y(tau) = 1/(omega_0 * sqrt(N) * pi * T)
    # where N is number of atoms

    N_atoms = 1e6
    cycle_time = 2 * T  # Approximately
    n_cycles_per_second = 1 / cycle_time

    sigma_single = 1 / (omega_0 * np.sqrt(N_atoms) * np.pi * T)

    # Averaging improves as 1/sqrt(tau)
    tau_avg = np.logspace(0, 5, 100)  # 1 s to 10^5 s
    n_measurements = tau_avg / cycle_time
    sigma_y = sigma_single / np.sqrt(n_measurements)

    results['stability'] = {
        'tau': tau_avg,
        'sigma_y': sigma_y,
        'sigma_single': sigma_single,
        'N_atoms': N_atoms
    }

    return results


def plot_results(results):
    """Create comprehensive visualization of Ramsey spectroscopy."""

    fig = plt.figure(figsize=(14, 12))
    T = results['T']

    # Plot 1: Ramsey fringes (wide view)
    ax1 = fig.add_subplot(2, 2, 1)
    fr = results['fringes']

    ax1.plot(fr['delta_wide'] * T / np.pi, fr['P_wide'], 'b-', linewidth=1)
    ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)

    ax1.set_xlabel(r'Detuning $\delta T / \pi$', fontsize=11)
    ax1.set_ylabel('Excited State Probability', fontsize=11)
    ax1.set_title(f'Ramsey Fringes (T = {T*1e3:.0f} ms)', fontsize=12)
    ax1.set_xlim(-20, 20)
    ax1.grid(True, alpha=0.3)

    # Highlight central fringe
    ax1.axvspan(-1, 1, alpha=0.2, color='red', label='Central fringe')
    ax1.legend(loc='upper right', fontsize=9)

    # Plot 2: Central fringe and Rabi comparison
    ax2 = fig.add_subplot(2, 2, 2)

    # Central Ramsey fringe
    ax2.plot(fr['delta_narrow'] / (2 * np.pi), fr['P_narrow'], 'b-', linewidth=2,
            label=f'Ramsey (T={T*1e3:.0f}ms)')

    # Rabi lineshape (scaled for visibility)
    rabi = results['rabi']
    ax2.plot(rabi['delta'] / (2 * np.pi), rabi['P'], 'r--', linewidth=2,
            label=f'Rabi ($\\tau$={rabi["tau"]*1e3:.0f}ms)')

    ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)

    # Mark linewidths
    fwhm_ramsey = results['linewidths']['ramsey']
    ax2.axvline(x=fwhm_ramsey/2, color='blue', linestyle=':', alpha=0.5)
    ax2.axvline(x=-fwhm_ramsey/2, color='blue', linestyle=':', alpha=0.5)

    ax2.set_xlabel('Detuning (Hz)', fontsize=11)
    ax2.set_ylabel('Excited State Probability', fontsize=11)
    ax2.set_title('Ramsey vs Rabi: Linewidth Comparison', fontsize=12)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Add linewidth text
    lw = results['linewidths']
    textstr = f'FWHM comparison:\nRamsey: {lw["ramsey"]:.2f} Hz\nRabi: {lw["rabi"]:.0f} Hz\nNarrowing: {lw["ratio"]:.0f}x'
    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 3: Effect of evolution time
    ax3 = fig.add_subplot(2, 2, 3)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(results['T_dependence'])))

    for i, td in enumerate(results['T_dependence']):
        label = f'T = {td["T"]*1e3:.0f} ms, FWHM = {td["fwhm"]:.1f} Hz'
        # Normalize delta for comparison
        ax3.plot(td['delta'] * td['T'], td['P'], color=colors[i],
                linewidth=1.5, label=label)

    ax3.set_xlabel(r'Normalized Detuning $\delta \cdot T$', fontsize=11)
    ax3.set_ylabel('Probability', fontsize=11)
    ax3.set_title(r'Ramsey Fringes: Linewidth $\propto 1/T$', fontsize=12)
    ax3.legend(loc='upper right', fontsize=8)
    ax3.set_xlim(-10, 10)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Decoherence effects
    ax4 = fig.add_subplot(2, 2, 4)
    deco = results['decoherence']

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, d in enumerate(deco['data']):
        if d['T2'] == np.inf:
            label = '$T_2 = \\infty$'
        else:
            label = f'$T_2 = {d["T2"]*1e3:.0f}$ ms'
        ax4.plot(deco['delta'] / (2 * np.pi), d['P'], color=colors[i],
                linewidth=1.5, label=label)

    ax4.set_xlabel('Detuning (Hz)', fontsize=11)
    ax4.set_ylabel('Probability', fontsize=11)
    ax4.set_title(f'Effect of Decoherence (T = {deco["T"]*1e3:.0f} ms)', fontsize=12)
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.3)

    # Add text about clock applications
    textstr = '\n'.join([
        'Atomic Clock Applications:',
        f'- NIST-F2 (Cs fountain): T ~ 0.5 s',
        f'- Optical lattice clocks: T ~ 0.1-1 s',
        f'- Fractional stability: $\\sigma_y \\sim 10^{{-16}}/\\sqrt{{\\tau}}$'
    ])
    ax4.text(0.05, 0.05, textstr, transform=ax4.transAxes, fontsize=9,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig


def main():
    print("=" * 60)
    print("Experiment 252: Ramsey Spectroscopy Linewidth")
    print("=" * 60)
    print()

    # Run simulation
    print("Running simulations...")
    results = simulate_ramsey_spectroscopy()

    # Create visualization
    print("\nCreating visualization...")
    fig = plot_results(results)

    # Save figure
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'ramsey_spectroscopy.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"Evolution time T: {results['T']*1e3:.0f} ms")
    print(f"Pulse duration tau: {results['tau']*1e3:.1f} ms")
    print()
    print("Linewidth comparison:")
    lw = results['linewidths']
    print(f"  Ramsey FWHM: {lw['ramsey']:.2f} Hz (= 1/(2T))")
    print(f"  Rabi FWHM: {lw['rabi']:.0f} Hz (~ Omega)")
    print(f"  Narrowing factor: {lw['ratio']:.0f}x")
    print()
    print("Key advantages of Ramsey method:")
    print("  1. Linewidth limited by T, not pulse duration")
    print("  2. Less sensitive to pulse area errors")
    print("  3. Enables precision spectroscopy with long T")
    print()
    print("Clock stability (single shot):")
    st = results['stability']
    print(f"  sigma_y(1 shot) = {st['sigma_single']:.2e}")
    print(f"  With {st['N_atoms']:.0e} atoms")

    plt.close()


if __name__ == "__main__":
    main()
