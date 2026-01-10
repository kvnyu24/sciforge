"""
Experiment 177: Quantum Trajectories Monte Carlo

Demonstrates the quantum trajectory (Monte Carlo wavefunction) method for
simulating open quantum system dynamics as an alternative to the master equation.

Physics:
    The quantum trajectory method unravels the master equation into stochastic
    pure-state evolutions (quantum jumps).

    Algorithm:
    1. Evolve with non-Hermitian effective Hamiltonian:
       H_eff = H - (i*hbar/2) * sum_k gamma_k * L_k^dag L_k

    2. At each step, compute jump probability:
       dp = dt * sum_k gamma_k * <psi| L_k^dag L_k |psi>

    3. With probability dp, apply jump:
       |psi> -> L_k |psi> / ||L_k |psi>||

    4. Otherwise, normalize after non-Hermitian evolution

    Ensemble average reproduces master equation:
    rho = E[|psi><psi|] (average over trajectories)

    Advantages:
    - Computational: scales as N vs N^2 for density matrix
    - Physical insight: individual detection events
    - Natural for photon counting experiments
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
sigma_plus = np.array([[0, 1], [0, 0]], dtype=complex)
sigma_minus = np.array([[0, 0], [1, 0]], dtype=complex)


def quantum_trajectory(psi0, H, jump_ops, gamma_list, t_max, dt, seed=None):
    """
    Single quantum trajectory simulation.

    Args:
        psi0: Initial state vector
        H: Hamiltonian
        jump_ops: List of jump operators L_k
        gamma_list: List of decay rates gamma_k
        t_max: Maximum time
        dt: Time step
        seed: Random seed

    Returns:
        times, trajectory of states, jump_times, jump_indices
    """
    if seed is not None:
        np.random.seed(seed)

    dim = len(psi0)
    psi = psi0.copy().astype(complex)
    psi = psi / np.linalg.norm(psi)

    # Build effective Hamiltonian
    H_eff = H.astype(complex).copy()
    for L, gamma in zip(jump_ops, gamma_list):
        H_eff -= 0.5j * gamma * (L.T.conj() @ L)

    times = [0]
    states = [psi.copy()]
    jump_times = []
    jump_which = []

    t = 0

    while t < t_max:
        # Calculate jump probabilities
        jump_probs = []
        for L, gamma in zip(jump_ops, gamma_list):
            L_psi = L @ psi
            prob = dt * gamma * np.real(np.conj(L_psi) @ L_psi)
            jump_probs.append(prob)

        total_jump_prob = sum(jump_probs)

        # Decide if jump occurs
        r = np.random.random()

        if r < total_jump_prob:
            # Jump occurs - determine which one
            r2 = np.random.random() * total_jump_prob
            cumsum = 0
            for i, (L, prob) in enumerate(zip(jump_ops, jump_probs)):
                cumsum += prob
                if r2 < cumsum:
                    # Apply jump operator
                    psi = L @ psi
                    psi = psi / np.linalg.norm(psi)
                    jump_times.append(t)
                    jump_which.append(i)
                    break
        else:
            # No jump - evolve with H_eff and renormalize
            dpsi = -1j * dt * H_eff @ psi
            psi = psi + dpsi
            psi = psi / np.linalg.norm(psi)

        t += dt
        times.append(t)
        states.append(psi.copy())

    return np.array(times), states, jump_times, jump_which


def ensemble_average(n_trajectories, psi0, H, jump_ops, gamma_list, t_max, dt):
    """
    Run ensemble of quantum trajectories and compute average density matrix.

    Args:
        n_trajectories: Number of trajectories
        psi0: Initial state
        H: Hamiltonian
        jump_ops: Jump operators
        gamma_list: Decay rates
        t_max: Maximum time
        dt: Time step

    Returns:
        times, list of average density matrices, all trajectories
    """
    all_times = None
    all_trajectories = []

    for i in range(n_trajectories):
        times, states, _, _ = quantum_trajectory(psi0, H, jump_ops, gamma_list, t_max, dt, seed=i)
        all_trajectories.append(states)
        if all_times is None:
            all_times = times

    # Compute average density matrix at each time
    n_times = len(all_times)
    rho_avg = []

    for t_idx in range(n_times):
        rho_t = np.zeros((len(psi0), len(psi0)), dtype=complex)
        for traj in all_trajectories:
            psi = traj[t_idx]
            rho_t += np.outer(psi, np.conj(psi))
        rho_t /= n_trajectories
        rho_avg.append(rho_t)

    return all_times, rho_avg, all_trajectories


def master_equation_solution(psi0, H, jump_ops, gamma_list, t_max, dt):
    """
    Solve Lindblad master equation directly for comparison.

    Args:
        psi0: Initial state
        H: Hamiltonian
        jump_ops: Jump operators
        gamma_list: Decay rates
        t_max: Maximum time
        dt: Time step

    Returns:
        times, density matrix history
    """
    rho = np.outer(psi0, np.conj(psi0))

    times = [0]
    rho_history = [rho.copy()]

    t = 0
    while t < t_max:
        # von Neumann term
        drho = -1j * (H @ rho - rho @ H)

        # Lindblad dissipator
        for L, gamma in zip(jump_ops, gamma_list):
            L_dag = L.T.conj()
            L_dag_L = L_dag @ L
            drho += gamma * (L @ rho @ L_dag - 0.5 * (L_dag_L @ rho + rho @ L_dag_L))

        rho = rho + drho * dt
        t += dt

        times.append(t)
        rho_history.append(rho.copy())

    return np.array(times), rho_history


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Parameters
    omega = 1.0  # Qubit frequency
    gamma = 0.3  # Decay rate
    t_max = 20.0
    dt = 0.05

    # System setup: driven damped qubit
    H = omega / 2 * sigma_z
    jump_ops = [sigma_minus]
    gamma_list = [gamma]

    # Initial state: |+> superposition
    psi0 = np.array([1, 1], dtype=complex) / np.sqrt(2)

    # ===== Plot 1: Individual trajectories =====
    ax1 = axes[0, 0]

    n_traj_show = 5
    colors = plt.cm.tab10(np.linspace(0, 1, n_traj_show))

    for i in range(n_traj_show):
        times, states, jumps, _ = quantum_trajectory(psi0, H, jump_ops, gamma_list, t_max, dt, seed=i)

        # Excited state population
        P_excited = [np.abs(psi[0])**2 for psi in states]

        ax1.plot(times, P_excited, color=colors[i], lw=1.5, alpha=0.7)

        # Mark quantum jumps
        for jump_t in jumps:
            ax1.axvline(jump_t, color=colors[i], linestyle=':', alpha=0.5, lw=0.5)

    ax1.set_xlabel('Time')
    ax1.set_ylabel('P(|0>)')
    ax1.set_title('Individual Quantum Trajectories\n(vertical lines = quantum jumps)')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, t_max)

    # ===== Plot 2: Ensemble average vs Master equation =====
    ax2 = axes[0, 1]

    n_trajectories = 100
    times_ens, rho_avg, all_traj = ensemble_average(n_trajectories, psi0, H, jump_ops, gamma_list, t_max, dt)
    times_me, rho_me = master_equation_solution(psi0, H, jump_ops, gamma_list, t_max, dt)

    # Population from ensemble
    P_ensemble = [np.real(rho[0, 0]) for rho in rho_avg]

    # Population from master equation
    P_master = [np.real(rho[0, 0]) for rho in rho_me]

    ax2.plot(times_ens, P_ensemble, 'b-', lw=2, label=f'Trajectory average (N={n_trajectories})')
    ax2.plot(times_me, P_master, 'r--', lw=2, label='Master equation')

    # Analytical solution for comparison (if applicable)
    P_analytical = np.abs(psi0[0])**2 * np.exp(-gamma * times_me) + 0 * (1 - np.exp(-gamma * times_me))
    ax2.plot(times_me, P_analytical, 'g:', lw=2, alpha=0.7, label='Analytical approx')

    ax2.set_xlabel('Time')
    ax2.set_ylabel('P(|0>)')
    ax2.set_title('Ensemble Average vs Master Equation\n(They should match)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, t_max)

    # ===== Plot 3: Coherence evolution =====
    ax3 = axes[1, 0]

    # Coherence from ensemble
    coherence_ens = [np.abs(rho[0, 1]) for rho in rho_avg]
    coherence_me = [np.abs(rho[0, 1]) for rho in rho_me]

    ax3.plot(times_ens, coherence_ens, 'b-', lw=2, label='Trajectory average')
    ax3.plot(times_me, coherence_me, 'r--', lw=2, label='Master equation')

    # Show individual trajectory coherences
    for i in range(min(5, len(all_traj))):
        coh_traj = [np.abs(np.outer(psi, np.conj(psi))[0, 1]) for psi in all_traj[i]]
        ax3.plot(times_ens, coh_traj, 'gray', lw=0.5, alpha=0.3)

    ax3.set_xlabel('Time')
    ax3.set_ylabel('Coherence |rho_01|')
    ax3.set_title('Coherence: Trajectories vs Average')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, t_max)

    # ===== Plot 4: Jump statistics =====
    ax4 = axes[1, 1]

    # Collect jump statistics
    n_stat_traj = 500
    all_jumps = []
    total_jumps = []

    for i in range(n_stat_traj):
        _, _, jumps, _ = quantum_trajectory(psi0, H, jump_ops, gamma_list, t_max, dt, seed=i + 1000)
        all_jumps.extend(jumps)
        total_jumps.append(len(jumps))

    # Histogram of jump times
    ax4.hist(all_jumps, bins=30, density=True, alpha=0.7, color='blue', label='Jump time distribution')

    # Expected rate: starts at gamma * P_excited, decays
    t_theory = np.linspace(0, t_max, 100)
    # Rate ~ gamma * P_excited(t) ~ gamma * exp(-gamma*t) for simple decay
    rate_theory = gamma * np.exp(-gamma * t_theory) / (1 - np.exp(-gamma * t_max))
    ax4.plot(t_theory, rate_theory * 2, 'r--', lw=2, label='Expected rate (scaled)')

    ax4.set_xlabel('Time of quantum jump')
    ax4.set_ylabel('Probability density')
    ax4.set_title(f'Quantum Jump Statistics\n(N={n_stat_traj} trajectories)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Inset: number of jumps distribution
    ax4_inset = ax4.inset_axes([0.6, 0.5, 0.35, 0.35])
    ax4_inset.hist(total_jumps, bins=range(max(total_jumps)+2), density=True, alpha=0.7, color='green')
    ax4_inset.set_xlabel('# jumps')
    ax4_inset.set_ylabel('P')
    ax4_inset.set_title(f'Mean={np.mean(total_jumps):.1f}', fontsize=9)

    plt.suptitle('Quantum Trajectories (Monte Carlo Wavefunction Method)\n'
                 'Stochastic unraveling of open system dynamics',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'quantum_trajectories.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'quantum_trajectories.png')}")

    # Print results
    print("\n=== Quantum Trajectories Results ===")
    print(f"\nParameters: omega = {omega}, gamma = {gamma}")
    print(f"Simulation: t_max = {t_max}, dt = {dt}")
    print(f"Ensemble size: {n_trajectories} trajectories")

    print(f"\nJump statistics ({n_stat_traj} trajectories):")
    print(f"  Mean jumps per trajectory: {np.mean(total_jumps):.2f}")
    print(f"  Std dev: {np.std(total_jumps):.2f}")
    print(f"  Total jumps observed: {len(all_jumps)}")

    # Compare final states
    final_P_ens = np.real(rho_avg[-1][0, 0])
    final_P_me = np.real(rho_me[-1][0, 0])
    print(f"\nFinal P(|0>) at t = {t_max}:")
    print(f"  Trajectory average: {final_P_ens:.4f}")
    print(f"  Master equation: {final_P_me:.4f}")
    print(f"  Difference: {abs(final_P_ens - final_P_me):.4f}")


if __name__ == "__main__":
    main()
