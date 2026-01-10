"""
Experiment 175: Density Matrix Formalism

This experiment introduces the density matrix (density operator) formalism
for describing quantum states, including pure and mixed states.

Physics:
    Density matrix (operator) rho:
    - Pure state: rho = |psi><psi|
    - Mixed state: rho = sum_i p_i |psi_i><psi_i|  (classical mixture)

    Key properties:
    - Tr(rho) = 1 (normalization)
    - rho^dag = rho (Hermitian)
    - rho >= 0 (positive semidefinite)
    - <A> = Tr(rho * A) (expectation values)

    Purity:
    - P = Tr(rho^2)
    - P = 1 for pure states
    - P < 1 for mixed states
    - P = 1/d for maximally mixed state (d = dimension)

    Time evolution:
    - von Neumann equation: d(rho)/dt = -i/hbar [H, rho]

    Partial trace and reduced density matrix:
    - For composite system AB: rho_A = Tr_B(rho_AB)
    - Used to describe entanglement and decoherence
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
identity_2 = np.eye(2, dtype=complex)


def pure_state_density_matrix(psi):
    """
    Create density matrix from pure state: rho = |psi><psi|.

    Args:
        psi: State vector (normalized)

    Returns:
        Density matrix
    """
    psi = np.asarray(psi, dtype=complex)
    return np.outer(psi, np.conj(psi))


def mixed_state_density_matrix(states, probabilities):
    """
    Create density matrix for mixed state.

    rho = sum_i p_i |psi_i><psi_i|

    Args:
        states: List of state vectors
        probabilities: Classical probabilities (sum to 1)

    Returns:
        Density matrix
    """
    rho = np.zeros((len(states[0]), len(states[0])), dtype=complex)
    for psi, p in zip(states, probabilities):
        rho += p * pure_state_density_matrix(psi)
    return rho


def purity(rho):
    """
    Calculate purity: Tr(rho^2).

    Pure state: P = 1
    Maximally mixed: P = 1/d

    Args:
        rho: Density matrix

    Returns:
        Purity value
    """
    return np.real(np.trace(rho @ rho))


def von_neumann_entropy(rho):
    """
    Calculate von Neumann entropy: S = -Tr(rho * log(rho)).

    Uses eigenvalue decomposition for numerical stability.

    Args:
        rho: Density matrix

    Returns:
        Entropy in natural units (nats)
    """
    eigenvalues = np.linalg.eigvalsh(rho)
    # Filter out zero/negative eigenvalues for log
    eigenvalues = eigenvalues[eigenvalues > 1e-15]
    return -np.sum(eigenvalues * np.log(eigenvalues))


def expectation_value(rho, operator):
    """
    Calculate expectation value: <A> = Tr(rho * A).

    Args:
        rho: Density matrix
        operator: Observable operator

    Returns:
        Expectation value
    """
    return np.real(np.trace(rho @ operator))


def partial_trace(rho_AB, dim_A, dim_B, trace_out='B'):
    """
    Compute partial trace over subsystem.

    rho_A = Tr_B(rho_AB)

    Args:
        rho_AB: Density matrix of composite system
        dim_A: Dimension of subsystem A
        dim_B: Dimension of subsystem B
        trace_out: 'A' or 'B' - which subsystem to trace out

    Returns:
        Reduced density matrix
    """
    if trace_out == 'B':
        # Trace out B, keep A
        rho_A = np.zeros((dim_A, dim_A), dtype=complex)
        for i in range(dim_A):
            for j in range(dim_A):
                for k in range(dim_B):
                    rho_A[i, j] += rho_AB[i * dim_B + k, j * dim_B + k]
        return rho_A
    else:
        # Trace out A, keep B
        rho_B = np.zeros((dim_B, dim_B), dtype=complex)
        for i in range(dim_B):
            for j in range(dim_B):
                for k in range(dim_A):
                    rho_B[i, j] += rho_AB[k * dim_B + i, k * dim_B + j]
        return rho_B


def von_neumann_evolution(rho0, H, t_span, hbar=1.0, n_points=100):
    """
    Solve von Neumann equation: drho/dt = -i/hbar [H, rho].

    Args:
        rho0: Initial density matrix
        H: Hamiltonian
        t_span: (t_start, t_end)
        hbar: Reduced Planck constant
        n_points: Number of time points

    Returns:
        Tuple of (times, density_matrices)
    """
    dim = rho0.shape[0]

    def rhs(t, y):
        # Reshape flat array to matrix
        rho = y[:dim**2].reshape(dim, dim) + 1j * y[dim**2:].reshape(dim, dim)
        drho_dt = -1j / hbar * (H @ rho - rho @ H)
        return np.concatenate([np.real(drho_dt).flatten(),
                               np.imag(drho_dt).flatten()])

    y0 = np.concatenate([np.real(rho0).flatten(), np.imag(rho0).flatten()])

    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(rhs, t_span, y0, t_eval=t_eval, method='RK45')

    rhos = []
    for i in range(len(sol.t)):
        y = sol.y[:, i]
        rho = y[:dim**2].reshape(dim, dim) + 1j * y[dim**2:].reshape(dim, dim)
        rhos.append(rho)

    return sol.t, rhos


def bloch_vector(rho):
    """
    Extract Bloch vector from 2x2 density matrix.

    rho = (I + r.sigma)/2 where r = (r_x, r_y, r_z)

    Args:
        rho: 2x2 density matrix

    Returns:
        Bloch vector (r_x, r_y, r_z)
    """
    r_x = np.real(np.trace(rho @ sigma_x))
    r_y = np.real(np.trace(rho @ sigma_y))
    r_z = np.real(np.trace(rho @ sigma_z))
    return np.array([r_x, r_y, r_z])


def main():
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # ===== Plot 1: Pure vs Mixed states =====
    ax1 = axes[0, 0]

    # Pure state: |+x>
    psi_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho_pure = pure_state_density_matrix(psi_plus)

    # Mixed state: 50% |0> + 50% |1>
    psi_0 = np.array([1, 0], dtype=complex)
    psi_1 = np.array([0, 1], dtype=complex)
    rho_mixed = mixed_state_density_matrix([psi_0, psi_1], [0.5, 0.5])

    # Visualize as heatmaps
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(8, 3))

    im1 = ax1a.imshow(np.abs(rho_pure), cmap='Blues', vmin=0, vmax=0.5)
    ax1a.set_title(f'Pure State |+x>\nPurity = {purity(rho_pure):.3f}')
    ax1a.set_xticks([0, 1])
    ax1a.set_yticks([0, 1])
    ax1a.set_xticklabels(['|0>', '|1>'])
    ax1a.set_yticklabels(['|0>', '|1>'])
    plt.colorbar(im1, ax=ax1a)

    # Annotate matrix elements
    for i in range(2):
        for j in range(2):
            ax1a.text(j, i, f'{rho_pure[i,j]:.2f}', ha='center', va='center')

    im2 = ax1b.imshow(np.abs(rho_mixed), cmap='Blues', vmin=0, vmax=0.5)
    ax1b.set_title(f'Mixed State (50% |0> + 50% |1>)\nPurity = {purity(rho_mixed):.3f}')
    ax1b.set_xticks([0, 1])
    ax1b.set_yticks([0, 1])
    ax1b.set_xticklabels(['|0>', '|1>'])
    ax1b.set_yticklabels(['|0>', '|1>'])
    plt.colorbar(im2, ax=ax1b)

    for i in range(2):
        for j in range(2):
            ax1b.text(j, i, f'{rho_mixed[i,j]:.2f}', ha='center', va='center')

    plt.tight_layout()
    plt.close(fig1)  # Close this temporary figure

    # Now use ax1 for purity comparison
    mix_fractions = np.linspace(0, 1, 50)
    purities = []

    for f in mix_fractions:
        # Interpolate between pure and mixed
        rho = (1 - f) * rho_pure + f * rho_mixed
        purities.append(purity(rho))

    ax1.plot(mix_fractions, purities, 'b-', lw=2)
    ax1.axhline(1.0, color='green', linestyle='--', alpha=0.5, label='Pure state')
    ax1.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Maximally mixed')

    ax1.set_xlabel('Mixing fraction')
    ax1.set_ylabel('Purity Tr(rho^2)')
    ax1.set_title('Purity: Pure vs Mixed States')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ===== Plot 2: Bloch sphere representation =====
    ax2 = axes[0, 1]

    # Create various states
    states_bloch = {
        'Pure |0>': pure_state_density_matrix([1, 0]),
        'Pure |1>': pure_state_density_matrix([0, 1]),
        'Pure |+>': pure_state_density_matrix([1, 1] / np.sqrt(2)),
        'Mixed 50/50': mixed_state_density_matrix([[1, 0], [0, 1]], [0.5, 0.5]),
        'Mixed 75/25': mixed_state_density_matrix([[1, 0], [0, 1]], [0.75, 0.25]),
        'Partial mixed': mixed_state_density_matrix(
            [[1, 0], [1, 1] / np.sqrt(2)], [0.5, 0.5]),
    }

    # Draw unit circle (Bloch sphere cross-section)
    theta = np.linspace(0, 2*np.pi, 100)
    ax2.plot(np.cos(theta), np.sin(theta), 'k-', lw=1, alpha=0.3)

    colors = plt.cm.tab10(np.linspace(0, 0.6, len(states_bloch)))
    for (name, rho), color in zip(states_bloch.items(), colors):
        r = bloch_vector(rho)
        ax2.scatter([r[0]], [r[2]], c=[color], s=100, label=f'{name}: |r|={np.linalg.norm(r):.2f}')
        ax2.annotate(name, (r[0], r[2]), fontsize=8)

    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_aspect('equal')
    ax2.set_xlabel('r_x (x component)')
    ax2.set_ylabel('r_z (z component)')
    ax2.set_title('Bloch Sphere (x-z plane)\nPure states on surface, mixed inside')
    ax2.legend(fontsize=7, loc='upper left')
    ax2.grid(True, alpha=0.3)

    # ===== Plot 3: von Neumann evolution =====
    ax3 = axes[0, 2]

    # Two-level system with Hamiltonian H = omega * sigma_z
    omega = 1.0
    H = omega * sigma_z

    # Initial state: superposition
    psi0 = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho0 = pure_state_density_matrix(psi0)

    t_span = (0, 4 * np.pi)
    times, rhos = von_neumann_evolution(rho0, H, t_span, hbar=1.0, n_points=200)

    # Track populations and coherence
    pop_0 = [np.real(rho[0, 0]) for rho in rhos]
    pop_1 = [np.real(rho[1, 1]) for rho in rhos]
    coherence = [np.abs(rho[0, 1]) for rho in rhos]
    phase = [np.angle(rho[0, 1]) for rho in rhos]

    ax3.plot(times, pop_0, 'b-', lw=2, label='Population |0>')
    ax3.plot(times, pop_1, 'r-', lw=2, label='Population |1>')
    ax3.plot(times, coherence, 'g-', lw=2, label='|Coherence|')

    ax3.set_xlabel('Time')
    ax3.set_ylabel('Matrix elements')
    ax3.set_title('von Neumann Evolution\ndrho/dt = -i[H, rho]')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # ===== Plot 4: Partial trace and entanglement =====
    ax4 = axes[1, 0]

    # Bell state: |00> + |11>)/sqrt(2)
    psi_bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    rho_bell = pure_state_density_matrix(psi_bell)

    # Product state: |+>|+>
    psi_product = np.array([1, 1, 1, 1], dtype=complex) / 2
    rho_product = pure_state_density_matrix(psi_product)

    # Partial traces
    rho_A_bell = partial_trace(rho_bell, 2, 2, 'B')
    rho_A_product = partial_trace(rho_product, 2, 2, 'B')

    # Plot reduced density matrices
    width = 0.35
    x_pos = np.arange(2)

    ax4.bar(x_pos - width/2, np.real(np.diag(rho_A_bell)), width,
           label=f'Bell state (S={von_neumann_entropy(rho_A_bell):.3f})', alpha=0.7)
    ax4.bar(x_pos + width/2, np.real(np.diag(rho_A_product)), width,
           label=f'Product state (S={von_neumann_entropy(rho_A_product):.3f})', alpha=0.7)

    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(['|0>', '|1>'])
    ax4.set_ylabel('Population')
    ax4.set_title('Reduced Density Matrix (Subsystem A)\nEntangled: maximally mixed, Product: pure')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # ===== Plot 5: Entropy vs entanglement =====
    ax5 = axes[1, 1]

    # Create states with varying entanglement
    # |psi> = cos(theta)|00> + sin(theta)|11>
    theta_range = np.linspace(0, np.pi/2, 50)
    entropies = []
    purities_ent = []

    for theta in theta_range:
        psi = np.array([np.cos(theta), 0, 0, np.sin(theta)], dtype=complex)
        rho_AB = pure_state_density_matrix(psi)
        rho_A = partial_trace(rho_AB, 2, 2, 'B')

        entropies.append(von_neumann_entropy(rho_A))
        purities_ent.append(purity(rho_A))

    ax5.plot(theta_range / (np.pi/2), entropies, 'b-', lw=2, label='Entropy S_A')
    ax5.plot(theta_range / (np.pi/2), purities_ent, 'r-', lw=2, label='Purity Tr(rho_A^2)')

    ax5.axhline(np.log(2), color='green', linestyle='--', alpha=0.5,
               label=f'Max entropy = ln(2) = {np.log(2):.3f}')

    ax5.set_xlabel('Entanglement parameter theta / (pi/2)')
    ax5.set_ylabel('Entropy / Purity')
    ax5.set_title('Entanglement Entropy\n|psi> = cos(theta)|00> + sin(theta)|11>')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # ===== Plot 6: Thermal state =====
    ax6 = axes[1, 2]

    # Thermal state: rho = exp(-beta*H) / Z
    # For two-level system: H = E * sigma_z

    E_gap = 1.0
    H_thermal = E_gap * np.diag([1, -1])  # Energies +E and -E

    temperatures = np.linspace(0.1, 5, 50)
    pop_excited = []
    thermal_purities = []

    for T in temperatures:
        beta = 1 / T  # kB = 1
        rho_thermal = np.diag([np.exp(-beta * E_gap), np.exp(beta * E_gap)])
        Z = np.trace(rho_thermal)
        rho_thermal /= Z

        pop_excited.append(np.real(rho_thermal[0, 0]))
        thermal_purities.append(purity(rho_thermal))

    ax6.plot(temperatures, pop_excited, 'b-', lw=2, label='Excited population')
    ax6.plot(temperatures, thermal_purities, 'r-', lw=2, label='Purity')

    ax6.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Temperature (E/kB)')
    ax6.set_ylabel('Population / Purity')
    ax6.set_title('Thermal (Gibbs) State\nrho = exp(-H/kT) / Z')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.suptitle('Density Matrix Formalism\n'
                 'rho = |psi><psi| (pure) or sum_i p_i|psi_i><psi_i| (mixed)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'density_matrix_intro.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'density_matrix_intro.png')}")

    # Print numerical results
    print("\n=== Density Matrix Formalism Results ===")

    print("\n1. Pure vs Mixed states:")
    print(f"   Pure state |+x> purity: {purity(rho_pure):.4f}")
    print(f"   Mixed state purity: {purity(rho_mixed):.4f}")
    print(f"   Maximally mixed (1/d): {1/2:.4f}")

    print("\n2. Bloch vector representation:")
    for name, rho in states_bloch.items():
        r = bloch_vector(rho)
        print(f"   {name}: r = ({r[0]:.3f}, {r[1]:.3f}, {r[2]:.3f}), |r| = {np.linalg.norm(r):.3f}")

    print("\n3. Entanglement entropy:")
    print(f"   Bell state reduced entropy: {von_neumann_entropy(rho_A_bell):.4f} = ln(2)")
    print(f"   Product state reduced entropy: {von_neumann_entropy(rho_A_product):.4f}")

    print("\n4. Key formulas:")
    print("   Expectation value: <A> = Tr(rho * A)")
    print("   Time evolution: drho/dt = -i/hbar [H, rho]")
    print("   Partial trace: rho_A = Tr_B(rho_AB)")
    print("   Purity: P = Tr(rho^2), 1/d <= P <= 1")


if __name__ == "__main__":
    main()
