"""
Example 119: Optical Solitons in Fibers

This example demonstrates optical soliton propagation in optical fibers,
where the balance between group velocity dispersion (GVD) and self-phase
modulation (SPM) creates stable pulse shapes.

Physics:
    Nonlinear Schrodinger equation (NLSE) for optical pulses:
    i * dA/dz + (beta_2/2) * d^2A/dt^2 + gamma * |A|^2 * A = 0

    where:
    - A is the pulse envelope
    - beta_2 is the GVD parameter (ps^2/km)
    - gamma is the nonlinear parameter (1/(W*km))

    Soliton condition (anomalous dispersion, beta_2 < 0):
    N^2 = gamma * P0 * T0^2 / |beta_2|

    where N is the soliton order, P0 is peak power, T0 is pulse width.

    Fundamental soliton (N = 1):
    A(z, t) = sqrt(P0) * sech(t/T0) * exp(i*z/(2*L_D))

    The fundamental soliton maintains its shape during propagation.

    Higher-order solitons (N > 1):
    - Show periodic breathing behavior
    - Period: z_s = pi/2 * L_D
    - Shape oscillates but returns to original after one period

    Dispersion length: L_D = T0^2 / |beta_2|
    Nonlinear length: L_NL = 1 / (gamma * P0)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq, fftshift


class OpticalFiber:
    """Optical fiber with GVD and nonlinearity"""

    def __init__(
        self,
        beta2: float,
        gamma: float,
        length: float
    ):
        """
        Args:
            beta2: GVD parameter (ps^2/km)
            gamma: Nonlinear parameter (1/(W*km))
            length: Fiber length (km)
        """
        self.beta2 = beta2
        self.gamma = gamma
        self.L = length

    def dispersion_length(self, T0: float) -> float:
        """Calculate dispersion length L_D = T0^2 / |beta2|"""
        return T0**2 / abs(self.beta2)

    def nonlinear_length(self, P0: float) -> float:
        """Calculate nonlinear length L_NL = 1 / (gamma * P0)"""
        return 1 / (self.gamma * P0)

    def soliton_order(self, P0: float, T0: float) -> float:
        """Calculate soliton order N = sqrt(L_D / L_NL)"""
        L_D = self.dispersion_length(T0)
        L_NL = self.nonlinear_length(P0)
        return np.sqrt(L_D / L_NL)

    def soliton_power(self, T0: float, N: int = 1) -> float:
        """Calculate required power for N-th order soliton"""
        return N**2 * abs(self.beta2) / (self.gamma * T0**2)

    def soliton_period(self, T0: float) -> float:
        """Calculate soliton period z_s = pi/2 * L_D"""
        L_D = self.dispersion_length(T0)
        return np.pi / 2 * L_D


class NLSE_Solver:
    """Solve the Nonlinear Schrodinger Equation using split-step Fourier method"""

    def __init__(
        self,
        fiber: OpticalFiber,
        T_window: float,
        n_t: int = 2048
    ):
        """
        Args:
            fiber: OpticalFiber object
            T_window: Time window (ps)
            n_t: Number of time points
        """
        self.fiber = fiber
        self.n_t = n_t

        # Time grid
        self.dt = T_window / n_t
        self.t = np.linspace(-T_window/2, T_window/2, n_t)

        # Frequency grid
        self.omega = 2 * np.pi * fftfreq(n_t, self.dt)

    def propagate(
        self,
        A0: np.ndarray,
        n_z: int = 1000,
        store_interval: int = 10
    ) -> tuple:
        """
        Propagate pulse through fiber.

        Args:
            A0: Initial pulse envelope
            n_z: Number of z steps
            store_interval: Store every n-th step

        Returns:
            z, t, A(z, t) array
        """
        dz = self.fiber.L / n_z

        # Linear operator (dispersion)
        D_half = np.exp(1j * self.fiber.beta2 / 2 * self.omega**2 * dz / 2)

        # Storage
        n_store = n_z // store_interval + 1
        A_store = np.zeros((n_store, self.n_t), dtype=complex)
        z_store = np.zeros(n_store)

        A = A0.copy()
        store_idx = 0

        for i in range(n_z):
            # Store
            if i % store_interval == 0:
                A_store[store_idx, :] = A
                z_store[store_idx] = i * dz
                store_idx += 1

            # Split-step: D/2 -> N -> D/2
            # Half dispersion step
            A_freq = fft(A)
            A_freq *= D_half
            A = ifft(A_freq)

            # Full nonlinear step
            A *= np.exp(1j * self.fiber.gamma * np.abs(A)**2 * dz)

            # Half dispersion step
            A_freq = fft(A)
            A_freq *= D_half
            A = ifft(A_freq)

        # Store final
        if store_idx < n_store:
            A_store[store_idx, :] = A
            z_store[store_idx] = self.fiber.L

        return z_store[:store_idx+1], self.t, A_store[:store_idx+1, :]


def create_soliton(t: np.ndarray, T0: float, P0: float, chirp: float = 0) -> np.ndarray:
    """Create a soliton pulse (sech shape)"""
    return np.sqrt(P0) * (1 / np.cosh(t / T0)) * np.exp(-1j * chirp * (t / T0)**2 / 2)


def create_gaussian(t: np.ndarray, T0: float, P0: float, chirp: float = 0) -> np.ndarray:
    """Create a Gaussian pulse"""
    return np.sqrt(P0) * np.exp(-0.5 * (t / T0)**2) * np.exp(-1j * chirp * (t / T0)**2 / 2)


def plot_fundamental_soliton():
    """Plot fundamental soliton propagation"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Fiber parameters (standard SMF at 1550 nm, anomalous dispersion)
    beta2 = -20e-3  # ps^2/km (anomalous)
    gamma = 1.3e-3  # 1/(W*km)
    T0 = 1.0  # ps

    # Calculate soliton power for N=1
    P0 = abs(beta2) / (gamma * T0**2)  # About 15 mW

    fiber = OpticalFiber(beta2, gamma, length=10)  # 10 km
    L_D = fiber.dispersion_length(T0)

    print(f"Fundamental soliton parameters:")
    print(f"  T0 = {T0} ps")
    print(f"  P0 = {P0*1e3:.2f} mW")
    print(f"  L_D = {L_D:.2f} km")
    print(f"  N = {fiber.soliton_order(P0, T0):.2f}")

    # Solve NLSE
    solver = NLSE_Solver(fiber, T_window=20*T0, n_t=2048)
    A0 = create_soliton(solver.t, T0, P0)

    z, t, A = solver.propagate(A0, n_z=1000, store_interval=10)

    # Plot 1: 2D intensity evolution
    ax1 = axes[0, 0]

    I = np.abs(A)**2 / P0  # Normalized intensity

    im = ax1.imshow(I, extent=[t.min(), t.max(), z.min(), z.max()],
                    aspect='auto', cmap='hot', origin='lower')
    plt.colorbar(im, ax=ax1, label='Normalized intensity')

    ax1.set_xlabel('Time (ps)')
    ax1.set_ylabel('Distance (km)')
    ax1.set_title('Fundamental Soliton (N=1) Propagation\n'
                  'Shape unchanged during propagation')

    # Plot 2: Pulse shape at different positions
    ax2 = axes[0, 1]

    z_indices = [0, len(z)//4, len(z)//2, 3*len(z)//4, -1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(z_indices)))

    for idx, color in zip(z_indices, colors):
        ax2.plot(t, np.abs(A[idx])**2 / P0, color=color, linewidth=2,
                 label=f'z = {z[idx]:.1f} km')

    # Theoretical sech^2
    ax2.plot(t, 1 / np.cosh(t / T0)**2, 'k--', linewidth=2,
             alpha=0.5, label='sech^2(t/T0)')

    ax2.set_xlabel('Time (ps)')
    ax2.set_ylabel('Normalized intensity')
    ax2.set_title('Pulse Shape at Different Positions\n'
                  'All curves overlap - soliton is stable')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-5*T0, 5*T0)

    # Plot 3: Spectrum evolution
    ax3 = axes[1, 0]

    omega = fftshift(solver.omega)
    nu = omega / (2 * np.pi)  # THz

    for idx, color in zip(z_indices, colors):
        spectrum = np.abs(fftshift(fft(A[idx])))**2
        spectrum /= spectrum.max()
        ax3.plot(nu, spectrum, color=color, linewidth=2,
                 label=f'z = {z[idx]:.1f} km')

    ax3.set_xlabel('Frequency (THz)')
    ax3.set_ylabel('Normalized spectrum')
    ax3.set_title('Spectrum at Different Positions\n'
                  'Spectrum also unchanged')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-3, 3)

    # Plot 4: Peak power and width vs distance
    ax4 = axes[1, 1]

    peak_power = np.max(np.abs(A)**2, axis=1) / P0
    width = np.zeros(len(z))

    for i in range(len(z)):
        I_norm = np.abs(A[i])**2 / np.max(np.abs(A[i])**2)
        half_max = np.where(I_norm >= 0.5)[0]
        if len(half_max) > 1:
            width[i] = (half_max[-1] - half_max[0]) * solver.dt / (2 * np.log(1 + np.sqrt(2)))  # Correct for sech

    ax4.plot(z, peak_power, 'b-', linewidth=2, label='Peak power')
    ax4.plot(z, width / T0, 'r-', linewidth=2, label='Width / T0')
    ax4.axhline(1.0, color='gray', linestyle='--', alpha=0.5)

    ax4.set_xlabel('Distance (km)')
    ax4.set_ylabel('Normalized value')
    ax4.set_title('Peak Power and Width vs Distance\n'
                  'Both remain constant for fundamental soliton')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0.5, 1.5)

    plt.tight_layout()
    return fig


def plot_higher_order_solitons():
    """Plot higher-order soliton breathing"""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Fiber parameters
    beta2 = -20e-3  # ps^2/km
    gamma = 1.3e-3  # 1/(W*km)
    T0 = 1.0  # ps

    L_D = T0**2 / abs(beta2)
    z_s = np.pi / 2 * L_D  # Soliton period

    # Different soliton orders
    soliton_orders = [1, 2, 3]

    for col, N in enumerate(soliton_orders):
        # Soliton power
        P0 = N**2 * abs(beta2) / (gamma * T0**2)

        # Propagate for one soliton period
        fiber = OpticalFiber(beta2, gamma, length=z_s * 1.2)
        solver = NLSE_Solver(fiber, T_window=20*T0, n_t=2048)
        A0 = create_soliton(solver.t, T0, P0)

        z, t, A = solver.propagate(A0, n_z=2000, store_interval=20)

        # Top row: 2D evolution
        ax_top = axes[0, col]

        I = np.abs(A)**2 / P0

        im = ax_top.imshow(I, extent=[t.min(), t.max(), z.min()/L_D, z.max()/L_D],
                           aspect='auto', cmap='hot', origin='lower')

        ax_top.axhline(0.5 * np.pi, color='cyan', linestyle='--', alpha=0.7)
        ax_top.axhline(np.pi, color='cyan', linestyle='--', alpha=0.7)

        ax_top.set_xlabel('Time (ps)')
        ax_top.set_ylabel('Distance (z/L_D)')
        ax_top.set_title(f'N = {N} Soliton (P0 = {P0*1e3:.1f} mW)')

        if col == 2:
            plt.colorbar(im, ax=ax_top, label='I/P0')

        # Bottom row: Pulse shapes at different positions
        ax_bot = axes[1, col]

        z_fractions = [0, 0.25, 0.5, 0.75, 1.0]  # Fractions of soliton period
        colors = plt.cm.plasma(np.linspace(0, 0.8, len(z_fractions)))

        for frac, color in zip(z_fractions, colors):
            idx = int(frac * (len(z) - 1) * (z_s / fiber.L))
            idx = min(idx, len(z) - 1)

            ax_bot.plot(t, np.abs(A[idx])**2 / P0, color=color, linewidth=2,
                        label=f'z = {frac:.2f} * z_s')

        ax_bot.set_xlabel('Time (ps)')
        ax_bot.set_ylabel('Normalized intensity')
        ax_bot.set_title(f'N = {N}: Pulse Evolution Over One Period')
        ax_bot.legend(fontsize=8, loc='upper right')
        ax_bot.grid(True, alpha=0.3)
        ax_bot.set_xlim(-5*T0, 5*T0)

    plt.tight_layout()
    return fig


def plot_soliton_physics():
    """Plot the physics of soliton formation"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Parameters
    beta2 = -20e-3  # ps^2/km
    gamma = 1.3e-3  # 1/(W*km)
    T0 = 1.0  # ps
    P0 = abs(beta2) / (gamma * T0**2)  # N=1 power

    # Plot 1: Dispersion only
    ax1 = axes[0, 0]

    # Dispersive broadening of Gaussian
    fiber_disp = OpticalFiber(beta2, 0, length=5)  # No nonlinearity
    solver = NLSE_Solver(fiber_disp, T_window=30*T0, n_t=2048)
    A0 = create_gaussian(solver.t, T0, P0)

    z, t, A_disp = solver.propagate(A0, n_z=500, store_interval=10)

    z_indices = [0, len(z)//4, len(z)//2, 3*len(z)//4, -1]
    colors = plt.cm.cool(np.linspace(0, 1, len(z_indices)))

    for idx, color in zip(z_indices, colors):
        ax1.plot(t, np.abs(A_disp[idx])**2 / P0, color=color, linewidth=2,
                 label=f'z = {z[idx]:.1f} km')

    ax1.set_xlabel('Time (ps)')
    ax1.set_ylabel('Normalized intensity')
    ax1.set_title('Dispersion Only (GVD, no SPM)\n'
                  'Pulse broadens and develops chirp')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-10*T0, 10*T0)

    # Plot 2: Nonlinearity only
    ax2 = axes[0, 1]

    fiber_nl = OpticalFiber(0, gamma, length=5)  # No dispersion
    solver_nl = NLSE_Solver(fiber_nl, T_window=30*T0, n_t=2048)
    A0_nl = create_soliton(solver_nl.t, T0, P0)

    z_nl, t_nl, A_nl = solver_nl.propagate(A0_nl, n_z=500, store_interval=10)

    for idx, color in zip(z_indices, colors):
        if idx < len(z_nl):
            ax2.plot(t_nl, np.abs(A_nl[idx])**2 / P0, color=color, linewidth=2,
                     label=f'z = {z_nl[idx]:.1f} km')

    ax2.set_xlabel('Time (ps)')
    ax2.set_ylabel('Normalized intensity')
    ax2.set_title('Nonlinearity Only (SPM, no GVD)\n'
                  'Shape unchanged, only phase modulated')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-5*T0, 5*T0)

    # Plot 3: Phase evolution comparison
    ax3 = axes[1, 0]

    # Phase from dispersion (at pulse center)
    z_range = np.linspace(0, 5, 100)
    L_D = fiber_disp.dispersion_length(T0)

    # Dispersive phase
    phi_disp = z_range / L_D * np.pi / 4  # Approximate

    # SPM phase
    L_NL = 1 / (gamma * P0)
    phi_spm = z_range / L_NL  # At peak intensity

    ax3.plot(z_range, phi_disp, 'b-', linewidth=2, label='Dispersive phase')
    ax3.plot(z_range, phi_spm, 'r-', linewidth=2, label='SPM phase')
    ax3.plot(z_range, phi_spm - phi_disp, 'g--', linewidth=2, label='Net phase')

    ax3.axhline(0, color='gray', linestyle=':', alpha=0.5)

    ax3.set_xlabel('Distance (km)')
    ax3.set_ylabel('Phase (rad)')
    ax3.set_title('Phase Evolution: Dispersion vs SPM\n'
                  'For soliton: L_D = L_NL, phases balance')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Annotate balance
    ax3.annotate(f'L_D = {L_D:.2f} km\nL_NL = {L_NL:.2f} km',
                xy=(2.5, 1), fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Plot 4: Soliton energy diagram
    ax4 = axes[1, 1]

    # Plot N^2 vs P for different T0
    T0_values = [0.5, 1.0, 2.0, 5.0]
    P_range = np.logspace(-3, 0, 100)  # W

    for T0_val in T0_values:
        N_sq = gamma * P_range * T0_val**2 / abs(beta2)
        ax4.loglog(P_range * 1e3, N_sq, linewidth=2,
                   label=f'T0 = {T0_val:.1f} ps')

    # Mark soliton orders
    for N in [1, 2, 3]:
        ax4.axhline(N**2, color='gray', linestyle='--', alpha=0.5)
        ax4.text(0.15, N**2 * 1.1, f'N = {N}', fontsize=10)

    ax4.set_xlabel('Peak power (mW)')
    ax4.set_ylabel('Soliton order N^2')
    ax4.set_title('Soliton Order: N^2 = gamma * P0 * T0^2 / |beta2|\n'
                  'Shorter pulses need more power')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, which='both')
    ax4.set_xlim(0.1, 1000)
    ax4.set_ylim(0.1, 100)

    plt.tight_layout()
    return fig


def plot_soliton_applications():
    """Plot soliton applications and special cases"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    beta2 = -20e-3
    gamma = 1.3e-3
    T0 = 1.0

    # Plot 1: Soliton collision
    ax1 = axes[0, 0]

    P0 = abs(beta2) / (gamma * T0**2)
    fiber = OpticalFiber(beta2, gamma, length=15)
    solver = NLSE_Solver(fiber, T_window=40*T0, n_t=4096)

    # Two solitons with different velocities (frequency shifted)
    omega_shift = 1.0  # THz
    A0 = (create_soliton(solver.t - 10*T0, T0, P0) * np.exp(1j * 2*np.pi * omega_shift * solver.t) +
          create_soliton(solver.t + 10*T0, T0, P0) * np.exp(-1j * 2*np.pi * omega_shift * solver.t))

    z, t, A = solver.propagate(A0, n_z=2000, store_interval=20)

    I = np.abs(A)**2 / P0

    im1 = ax1.imshow(I, extent=[t.min(), t.max(), z.min(), z.max()],
                     aspect='auto', cmap='hot', origin='lower', vmax=2)
    plt.colorbar(im1, ax=ax1, label='I/P0')

    ax1.set_xlabel('Time (ps)')
    ax1.set_ylabel('Distance (km)')
    ax1.set_title('Soliton Collision\n'
                  'Solitons pass through each other unchanged')

    # Plot 2: Soliton fission from higher-order
    ax2 = axes[0, 1]

    N = 4
    P0_high = N**2 * abs(beta2) / (gamma * T0**2)
    fiber_long = OpticalFiber(beta2, gamma, length=20)
    solver_fiss = NLSE_Solver(fiber_long, T_window=60*T0, n_t=4096)

    A0_high = create_soliton(solver_fiss.t, T0, P0_high)
    z_f, t_f, A_f = solver_fiss.propagate(A0_high, n_z=3000, store_interval=30)

    I_f = np.abs(A_f)**2 / P0_high

    im2 = ax2.imshow(I_f, extent=[t_f.min(), t_f.max(), z_f.min(), z_f.max()],
                     aspect='auto', cmap='hot', origin='lower', vmax=5)
    plt.colorbar(im2, ax=ax2, label='I/P0')

    ax2.set_xlabel('Time (ps)')
    ax2.set_ylabel('Distance (km)')
    ax2.set_title(f'Higher-Order Soliton (N={N}) Dynamics\n'
                  'Periodic compression and evolution')

    # Plot 3: Soliton compression
    ax3 = axes[1, 0]

    # Show pulse compression via soliton effect
    fiber_compress = OpticalFiber(beta2, gamma, length=2)
    solver_comp = NLSE_Solver(fiber_compress, T_window=20*T0, n_t=2048)

    N_comp = 2
    P0_comp = N_comp**2 * abs(beta2) / (gamma * T0**2)
    A0_comp = create_soliton(solver_comp.t, T0, P0_comp)

    z_c, t_c, A_c = solver_comp.propagate(A0_comp, n_z=500, store_interval=5)

    # Find maximum compression
    peak_intensity = np.max(np.abs(A_c)**2, axis=1)
    max_idx = np.argmax(peak_intensity)

    ax3.plot(t_c, np.abs(A_c[0])**2 / P0_comp, 'b-', linewidth=2, label='Input')
    ax3.plot(t_c, np.abs(A_c[max_idx])**2 / P0_comp, 'r-', linewidth=2,
             label=f'Maximum compression (z={z_c[max_idx]:.2f} km)')

    ax3.set_xlabel('Time (ps)')
    ax3.set_ylabel('Normalized intensity')
    ax3.set_title('Soliton Compression\n'
                  f'N={N_comp} soliton at maximum compression')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-5*T0, 5*T0)

    # Plot 4: Soliton communication
    ax4 = axes[1, 1]

    # Draw conceptual diagram
    ax4.text(0.5, 0.95, 'Soliton-Based Optical Communication', fontsize=14,
             ha='center', va='top', fontweight='bold', transform=ax4.transAxes)

    # Advantages
    advantages = [
        "- No pulse broadening over long distances",
        "- Higher bit rates (shorter pulses possible)",
        "- No need for regenerators",
        "- Reduced signal distortion",
    ]

    for i, adv in enumerate(advantages):
        ax4.text(0.1, 0.75 - i*0.12, adv, fontsize=11, transform=ax4.transAxes)

    # Parameters
    params = """
Typical Parameters:
- Wavelength: 1550 nm (C-band)
- Pulse width: 10-50 ps
- Peak power: 1-10 mW
- Bit rate: 10-40 Gb/s
- Distance: >1000 km

Challenge: Gordon-Haus jitter
from amplifier noise
"""
    ax4.text(0.5, 0.4, params, fontsize=10, transform=ax4.transAxes,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')

    plt.tight_layout()
    return fig


def main():
    """Main function to demonstrate optical solitons"""

    # Create figures
    fig1 = plot_fundamental_soliton()
    fig2 = plot_higher_order_solitons()
    fig3 = plot_soliton_physics()
    fig4 = plot_soliton_applications()

    # Save plots
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)

    fig1.savefig(os.path.join(output_dir, 'optical_solitons.png'),
                 dpi=150, bbox_inches='tight')
    fig2.savefig(os.path.join(output_dir, 'higher_order_solitons.png'),
                 dpi=150, bbox_inches='tight')
    fig3.savefig(os.path.join(output_dir, 'soliton_physics.png'),
                 dpi=150, bbox_inches='tight')
    fig4.savefig(os.path.join(output_dir, 'soliton_applications.png'),
                 dpi=150, bbox_inches='tight')

    print(f"Plots saved to {output_dir}/optical_solitons*.png and related")

    # Print analysis
    print("\n=== Optical Soliton Analysis ===")

    beta2 = -20e-3  # ps^2/km
    gamma = 1.3e-3  # 1/(W*km)

    print(f"\nFiber parameters:")
    print(f"  beta2 = {beta2*1e3:.0f} ps^2/km (anomalous dispersion)")
    print(f"  gamma = {gamma*1e3:.2f} 1/(W*m)")

    fiber = OpticalFiber(beta2, gamma, 1.0)

    print(f"\nSoliton parameters for different pulse widths:")
    print("-" * 60)
    print(f"{'T0 (ps)':<10} {'P0 (mW)':<15} {'L_D (km)':<12} {'z_s (km)':<12}")
    print("-" * 60)

    for T0 in [0.5, 1.0, 2.0, 5.0, 10.0]:
        P0 = fiber.soliton_power(T0)
        L_D = fiber.dispersion_length(T0)
        z_s = fiber.soliton_period(T0)
        print(f"{T0:<10.1f} {P0*1e3:<15.2f} {L_D:<12.2f} {z_s:<12.2f}")

    print("\nKey equations:")
    print("  NLSE: i*dA/dz + (beta2/2)*d^2A/dt^2 + gamma*|A|^2*A = 0")
    print("  Soliton order: N^2 = gamma*P0*T0^2 / |beta2|")
    print("  Fundamental soliton: A = sqrt(P0)*sech(t/T0)*exp(i*z/(2*L_D))")
    print("  Dispersion length: L_D = T0^2 / |beta2|")
    print("  Soliton period: z_s = pi/2 * L_D")


if __name__ == "__main__":
    main()
