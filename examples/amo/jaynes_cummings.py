"""
Experiment 254: Jaynes-Cummings Model and Vacuum Rabi Splitting

This example demonstrates the Jaynes-Cummings model of cavity QED, where a
two-level atom couples to a quantized single-mode electromagnetic field.
We explore:
- Vacuum Rabi oscillations (single photon dynamics)
- Jaynes-Cummings ladder of energy levels
- Vacuum Rabi splitting in transmission spectrum
- Collapse and revival of Rabi oscillations
- Strong coupling regime: g >> kappa, gamma

Key physics:
- H = hbar*omega_c*a^dag*a + hbar*omega_a*sigma_z/2 + hbar*g*(a*sigma_+ + a^dag*sigma_-)
- Dressed states: |n, +> and |n, -> with splitting 2*g*sqrt(n+1)
- Vacuum Rabi splitting: 2*g (observed in transmission)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, eigh
from src.sciforge.physics.amo import HBAR

def create_operators(n_photons):
    """
    Create operators for Jaynes-Cummings model.

    Args:
        n_photons: Maximum photon number (Hilbert space truncation)

    Returns:
        Dictionary of operators
    """
    # Cavity operators (Fock basis: |0>, |1>, ..., |n_max>)
    dim_c = n_photons + 1

    # Annihilation operator
    a = np.zeros((dim_c, dim_c), dtype=complex)
    for n in range(1, dim_c):
        a[n-1, n] = np.sqrt(n)

    # Creation operator
    a_dag = a.T.conj()

    # Number operator
    n_op = a_dag @ a

    # Atom operators (|g>, |e> basis)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    sigma_plus = np.array([[0, 1], [0, 0]], dtype=complex)
    sigma_minus = np.array([[0, 0], [1, 0]], dtype=complex)

    # Identity matrices
    I_c = np.eye(dim_c, dtype=complex)
    I_a = np.eye(2, dtype=complex)

    # Full space operators (cavity x atom)
    # Convention: |n, e> = |n> x |e>, |n, g> = |n> x |g>

    A = np.kron(a, I_a)
    A_dag = np.kron(a_dag, I_a)
    N = np.kron(n_op, I_a)
    Sz = np.kron(I_c, sigma_z)
    Sp = np.kron(I_c, sigma_plus)
    Sm = np.kron(I_c, sigma_minus)

    return {
        'a': A,
        'a_dag': A_dag,
        'n': N,
        'sigma_z': Sz,
        'sigma_plus': Sp,
        'sigma_minus': Sm,
        'dim_c': dim_c,
        'dim_total': 2 * dim_c
    }


def jaynes_cummings_hamiltonian(ops, omega_c, omega_a, g):
    """
    Construct Jaynes-Cummings Hamiltonian.

    H = hbar*omega_c*a^dag*a + hbar*omega_a*sigma_z/2
        + hbar*g*(a*sigma_+ + a^dag*sigma_-)

    Args:
        ops: Dictionary of operators
        omega_c: Cavity frequency
        omega_a: Atom transition frequency
        g: Coupling strength

    Returns:
        Hamiltonian matrix
    """
    H_cavity = omega_c * ops['a_dag'] @ ops['a']
    H_atom = omega_a * ops['sigma_z'] / 2
    H_int = g * (ops['a'] @ ops['sigma_plus'] + ops['a_dag'] @ ops['sigma_minus'])

    return H_cavity + H_atom + H_int


def simulate_jaynes_cummings():
    """Simulate Jaynes-Cummings dynamics and spectrum."""

    results = {}

    # Parameters (in units where hbar = 1)
    omega_c = 1.0  # Cavity frequency (reference)
    omega_a = 1.0  # Atom frequency (resonant case)
    g = 0.05  # Coupling strength (strong coupling: g/omega >> 0)

    n_photons = 10  # Truncation

    results['omega_c'] = omega_c
    results['omega_a'] = omega_a
    results['g'] = g

    print(f"Coupling strength g/omega_c: {g/omega_c:.3f}")

    # Create operators
    print("Creating operators...")
    ops = create_operators(n_photons)
    dim = ops['dim_total']

    # 1. Energy spectrum
    print("Computing energy spectrum...")
    H = jaynes_cummings_hamiltonian(ops, omega_c, omega_a, g)
    eigenvalues, eigenvectors = eigh(H)

    results['eigenvalues'] = eigenvalues
    results['eigenvectors'] = eigenvectors

    # 2. Vacuum Rabi oscillations
    print("Simulating vacuum Rabi oscillations...")

    # Initial state: atom excited, cavity vacuum |0, e>
    # In our basis: |0,e>, |0,g>, |1,e>, |1,g>, ...
    psi_0 = np.zeros(dim, dtype=complex)
    psi_0[0] = 1  # |0, e>

    # Time evolution
    t_max = 5 * 2 * np.pi / g  # Several Rabi periods
    n_times = 500
    times = np.linspace(0, t_max, n_times)

    P_excited = []
    n_photon_avg = []

    for t in times:
        U = expm(-1j * H * t)
        psi_t = U @ psi_0

        # Excited state probability
        # Sum over all |n, e> states (even indices in our convention)
        P_e = sum(abs(psi_t[2*n])**2 for n in range(n_photons + 1))
        P_excited.append(P_e)

        # Average photon number
        n_avg = np.real(np.conj(psi_t) @ ops['n'] @ psi_t)
        n_photon_avg.append(n_avg)

    results['rabi'] = {
        't': times,
        'P_excited': np.array(P_excited),
        'n_photon': np.array(n_photon_avg)
    }

    # 3. Dressed states and Jaynes-Cummings ladder
    print("Analyzing dressed states...")

    # For each manifold n, the states are:
    # |n, +> = (|n-1, e> + |n, g>) / sqrt(2)  with energy n*omega + g*sqrt(n)
    # |n, -> = (|n-1, e> - |n, g>) / sqrt(2)  with energy n*omega - g*sqrt(n)

    n_manifolds = 5
    dressed_energies = []
    for n in range(n_manifolds + 1):
        if n == 0:
            # Ground state: |0, g>
            dressed_energies.append({'n': 0, 'E': -omega_a / 2, 'type': 'g'})
        else:
            E_plus = n * omega_c - omega_a / 2 + g * np.sqrt(n)
            E_minus = n * omega_c - omega_a / 2 - g * np.sqrt(n)
            dressed_energies.append({'n': n, 'E': E_plus, 'type': '+'})
            dressed_energies.append({'n': n, 'E': E_minus, 'type': '-'})

    results['dressed'] = dressed_energies

    # 4. Transmission spectrum (vacuum Rabi splitting)
    print("Computing transmission spectrum...")

    # Probe at various detunings
    delta_range = np.linspace(-4 * g, 4 * g, 500)
    omega_probe = omega_c + delta_range

    # Transmission ~ |1/(delta + i*kappa)|^2 with kappa = cavity decay
    kappa = 0.01 * omega_c  # Cavity linewidth
    gamma = 0.005 * omega_c  # Atom linewidth

    # In strong coupling, transmission shows two peaks at +/- g
    # Simple model: two Lorentzians
    transmission_empty = kappa**2 / (delta_range**2 + kappa**2)

    # With atom: dressed states at +/- g
    transmission_atom = (
        0.5 * kappa**2 / ((delta_range - g)**2 + ((kappa + gamma) / 2)**2) +
        0.5 * kappa**2 / ((delta_range + g)**2 + ((kappa + gamma) / 2)**2)
    )
    # Normalize
    transmission_atom /= np.max(transmission_atom)
    transmission_empty /= np.max(transmission_empty)

    results['transmission'] = {
        'delta': delta_range,
        'empty': transmission_empty,
        'atom': transmission_atom,
        'kappa': kappa,
        'gamma': gamma
    }

    # 5. Collapse and revival with coherent state
    print("Simulating collapse and revival...")

    # Initial state: atom excited, cavity in coherent state |alpha>
    alpha = 3.0  # Coherent state amplitude
    n_bar = abs(alpha)**2  # Mean photon number

    # Coherent state in Fock basis
    coherent_state = np.array([
        np.exp(-abs(alpha)**2 / 2) * alpha**n / np.sqrt(float(math.factorial(n)))
        for n in range(n_photons + 1)
    ], dtype=complex)

    # Initial state: |alpha, e>
    psi_coherent = np.zeros(dim, dtype=complex)
    for n in range(n_photons + 1):
        psi_coherent[2*n] = coherent_state[n]  # |n, e>

    psi_coherent /= np.linalg.norm(psi_coherent)

    # Time evolution for collapse and revival
    t_revival = 2 * np.pi / g * np.sqrt(n_bar)  # Revival time
    t_max_cr = 3 * t_revival
    n_times_cr = 1000
    times_cr = np.linspace(0, t_max_cr, n_times_cr)

    P_excited_cr = []
    for t in times_cr:
        U = expm(-1j * H * t)
        psi_t = U @ psi_coherent

        P_e = sum(abs(psi_t[2*n])**2 for n in range(n_photons + 1))
        P_excited_cr.append(P_e)

    results['revival'] = {
        't': times_cr,
        'P_excited': np.array(P_excited_cr),
        'alpha': alpha,
        't_revival': t_revival
    }

    return results


def plot_results(results):
    """Create comprehensive visualization of Jaynes-Cummings physics."""

    fig = plt.figure(figsize=(14, 12))
    g = results['g']

    # Plot 1: Vacuum Rabi oscillations
    ax1 = fig.add_subplot(2, 2, 1)
    rabi = results['rabi']

    ax1.plot(rabi['t'] * g / (2 * np.pi), rabi['P_excited'], 'b-', linewidth=2,
            label=r'$P_e$ (atom excited)')
    ax1.plot(rabi['t'] * g / (2 * np.pi), rabi['n_photon'], 'r--', linewidth=2,
            label=r'$\langle n \rangle$ (photon number)')

    ax1.set_xlabel(r'Time $gt / 2\pi$ (Rabi periods)', fontsize=11)
    ax1.set_ylabel('Probability / Photon number', fontsize=11)
    ax1.set_title('Vacuum Rabi Oscillations', fontsize=12)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_xlim(0, 5)
    ax1.grid(True, alpha=0.3)

    # Add text about exchange
    ax1.text(0.05, 0.95, 'Single photon: $|0,e\\rangle \\leftrightarrow |1,g\\rangle$\n'
            f'Period: $2\\pi / (2g) = \\pi / g$',
            transform=ax1.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Jaynes-Cummings ladder
    ax2 = fig.add_subplot(2, 2, 2)

    # Plot energy levels
    for level in results['dressed']:
        n = level['n']
        E = level['E']
        color = 'blue' if level['type'] == '+' else 'red' if level['type'] == '-' else 'green'

        # Draw level
        x = [n - 0.3, n + 0.3]
        ax2.plot(x, [E, E], color=color, linewidth=2)

        # Label
        if level['type'] == 'g':
            label = '$|0,g\\rangle$'
        else:
            label = f'$|{n},{level["type"]}\\rangle$'

    # Draw splittings
    for n in range(1, 4):
        E_plus = n * results['omega_c'] - results['omega_a'] / 2 + g * np.sqrt(n)
        E_minus = n * results['omega_c'] - results['omega_a'] / 2 - g * np.sqrt(n)
        ax2.annotate('', xy=(n + 0.35, E_plus), xytext=(n + 0.35, E_minus),
                    arrowprops=dict(arrowstyle='<->', color='green', lw=1.5))
        ax2.text(n + 0.4, (E_plus + E_minus) / 2, f'$2g\\sqrt{{{n}}}$',
                fontsize=8, color='green', verticalalignment='center')

    ax2.set_xlabel('Manifold number n', fontsize=11)
    ax2.set_ylabel('Energy (units of $\\omega_c$)', fontsize=11)
    ax2.set_title('Jaynes-Cummings Ladder (Dressed States)', fontsize=12)
    ax2.set_xlim(-0.5, 4.5)
    ax2.grid(True, alpha=0.3)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='blue', lw=2, label='$|n,+\\rangle$'),
                      Line2D([0], [0], color='red', lw=2, label='$|n,-\\rangle$'),
                      Line2D([0], [0], color='green', lw=2, label='Ground')]
    ax2.legend(handles=legend_elements, loc='upper left', fontsize=9)

    # Plot 3: Vacuum Rabi splitting in transmission
    ax3 = fig.add_subplot(2, 2, 3)
    trans = results['transmission']

    ax3.plot(trans['delta'] / g, trans['empty'], 'b--', linewidth=2,
            label='Empty cavity')
    ax3.plot(trans['delta'] / g, trans['atom'], 'r-', linewidth=2,
            label='With atom')

    # Mark splitting
    ax3.axvline(x=-1, color='gray', linestyle=':', alpha=0.5)
    ax3.axvline(x=1, color='gray', linestyle=':', alpha=0.5)
    ax3.annotate('', xy=(1, 0.5), xytext=(-1, 0.5),
                arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    ax3.text(0, 0.55, '2g', fontsize=12, color='green', ha='center', fontweight='bold')

    ax3.set_xlabel(r'Detuning $\delta / g$', fontsize=11)
    ax3.set_ylabel('Transmission (normalized)', fontsize=11)
    ax3.set_title('Vacuum Rabi Splitting in Cavity Transmission', fontsize=12)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.set_xlim(-4, 4)
    ax3.grid(True, alpha=0.3)

    # Add coupling regime text
    kappa = trans['kappa']
    gamma = trans['gamma']
    textstr = f'Strong coupling:\n$g / \\kappa = {g/kappa:.1f}$\n$g / \\gamma = {g/gamma:.1f}$'
    ax3.text(0.05, 0.95, textstr, transform=ax3.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 4: Collapse and revival
    ax4 = fig.add_subplot(2, 2, 4)
    rev = results['revival']

    ax4.plot(rev['t'] * g / (2 * np.pi), rev['P_excited'], 'b-', linewidth=1)

    # Mark collapse and revival times
    t_rev = rev['t_revival']
    ax4.axvline(x=t_rev * g / (2 * np.pi), color='red', linestyle='--', alpha=0.7,
               label=f'Revival: $t_R = 2\\pi\\sqrt{{\\bar{{n}}}} / g$')

    ax4.set_xlabel(r'Time $gt / 2\pi$', fontsize=11)
    ax4.set_ylabel('Excited State Probability', fontsize=11)
    ax4.set_title(f'Collapse and Revival ($|\\alpha|^2 = {rev["alpha"]**2:.0f}$ photons)', fontsize=12)
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.3)

    # Add text about physics
    ax4.text(0.05, 0.05, 'Collapse: dephasing of different\n   Fock state components\n'
            'Revival: rephasing at $t = t_R$',
            transform=ax4.transAxes, fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig


def main():
    print("=" * 60)
    print("Experiment 254: Jaynes-Cummings Model and Vacuum Rabi Splitting")
    print("=" * 60)
    print()

    # Run simulation
    print("Running simulations...")
    results = simulate_jaynes_cummings()

    # Create visualization
    print("\nCreating visualization...")
    fig = plot_results(results)

    # Save figure
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'jaynes_cummings.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"Cavity frequency: omega_c = {results['omega_c']:.2f}")
    print(f"Atom frequency: omega_a = {results['omega_a']:.2f}")
    print(f"Coupling strength: g = {results['g']:.3f}")
    print()
    print("Key features of Jaynes-Cummings model:")
    print("  1. Vacuum Rabi oscillations at frequency 2g")
    print("  2. Dressed state splitting: 2g*sqrt(n+1)")
    print("  3. Vacuum Rabi splitting in transmission: 2g")
    print("  4. Collapse and revival with coherent states")
    print()
    print("Strong coupling regime requirements:")
    print("  - g >> kappa (cavity decay)")
    print("  - g >> gamma (atomic decay)")
    print("  - Enables quantum state manipulation")
    print()
    print("Applications:")
    print("  - Quantum computing (cavity QED qubits)")
    print("  - Single-photon sources")
    print("  - Quantum networks")
    print("  - Fundamental tests of quantum mechanics")

    plt.close()


if __name__ == "__main__":
    main()
