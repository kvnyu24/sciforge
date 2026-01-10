"""
Example 117: Laser Rate Equations

This example demonstrates laser dynamics through rate equations,
showing population inversion, threshold behavior, and relaxation oscillations.

Physics:
    Three-level and four-level laser rate equations:

    dN2/dt = R_pump - N2/tau_21 - sigma*c*phi*(N2 - N1)  (upper laser level)
    dN1/dt = N2/tau_21 - N1/tau_10 + sigma*c*phi*(N2 - N1)  (lower laser level)
    dphi/dt = Gamma*sigma*c*(N2 - N1)*phi - phi/tau_c + beta*N2/tau_21  (photon number)

    Key parameters:
    - R_pump: Pump rate
    - sigma: Stimulated emission cross-section
    - tau_21: Upper state lifetime
    - tau_c: Cavity photon lifetime
    - Gamma: Confinement factor
    - beta: Spontaneous emission factor

    Threshold condition: R_pump >= N_th / tau_21
    where N_th = 1 / (Gamma * sigma * c * tau_c)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


class ThreeLevelLaser:
    """Three-level laser system (e.g., Ruby laser)"""

    def __init__(self,
                 sigma: float = 2.5e-20,      # Stimulated emission cross-section (m^2)
                 tau_21: float = 3e-3,         # Upper state lifetime (s)
                 tau_c: float = 10e-9,         # Cavity photon lifetime (s)
                 gamma: float = 1.0,           # Confinement factor
                 beta: float = 1e-5,           # Spontaneous emission factor
                 n_total: float = 1.6e25):     # Total dopant density (m^-3)
        """
        Args:
            sigma: Stimulated emission cross-section
            tau_21: Upper state lifetime (fluorescence lifetime)
            tau_c: Cavity photon lifetime
            gamma: Mode confinement factor
            beta: Fraction of spontaneous emission into lasing mode
            n_total: Total dopant concentration
        """
        self.sigma = sigma
        self.tau_21 = tau_21
        self.tau_c = tau_c
        self.gamma = gamma
        self.beta = beta
        self.n_total = n_total
        self.c = 3e8  # Speed of light

        # Threshold population inversion
        self.n_th = 1.0 / (gamma * sigma * self.c * tau_c)

        # Threshold pump rate
        self.R_th = self.n_th / tau_21

    def rate_equations(self, y, t, R_pump):
        """
        Rate equations for three-level laser.

        Args:
            y: [N2, phi] - upper state population, photon number
            t: Time
            R_pump: Pump rate (may be function of t or constant)

        Returns:
            [dN2/dt, dphi/dt]
        """
        N2, phi = y

        # Ensure non-negative values
        N2 = max(0, N2)
        phi = max(0, phi)

        # Ground state population
        N1 = self.n_total - N2

        # Population inversion
        delta_N = N2 - N1

        # Rate equations
        dN2_dt = (R_pump - N2 / self.tau_21 -
                  self.sigma * self.c * phi * delta_N)

        dphi_dt = (self.gamma * self.sigma * self.c * delta_N * phi -
                   phi / self.tau_c +
                   self.beta * N2 / self.tau_21)

        return [dN2_dt, dphi_dt]

    def simulate(self, R_pump, t_span, y0=None):
        """
        Simulate laser dynamics.

        Args:
            R_pump: Pump rate (constant or array matching t_span)
            t_span: Time array
            y0: Initial conditions [N2_0, phi_0]

        Returns:
            Dictionary with time evolution
        """
        if y0 is None:
            y0 = [0, 1]  # Start with no inversion, one photon (noise)

        if callable(R_pump):
            solution = odeint(lambda y, t: self.rate_equations(y, t, R_pump(t)),
                            y0, t_span)
        else:
            solution = odeint(lambda y, t: self.rate_equations(y, t, R_pump),
                            y0, t_span)

        return {
            't': t_span,
            'N2': solution[:, 0],
            'phi': solution[:, 1],
            'N1': self.n_total - solution[:, 0],
            'delta_N': 2 * solution[:, 0] - self.n_total,
            'power': solution[:, 1] / self.tau_c  # Photons/s out of cavity
        }


class FourLevelLaser:
    """Four-level laser system (e.g., Nd:YAG)"""

    def __init__(self,
                 sigma: float = 6.5e-23,       # Stimulated emission cross-section (m^2)
                 tau_2: float = 230e-6,        # Upper state lifetime (s)
                 tau_1: float = 1e-12,         # Lower state lifetime (s) - very fast
                 tau_c: float = 10e-9,         # Cavity photon lifetime (s)
                 gamma: float = 1.0,           # Confinement factor
                 beta: float = 1e-5):          # Spontaneous emission factor
        """
        Four-level system with fast lower level relaxation.
        """
        self.sigma = sigma
        self.tau_2 = tau_2  # Upper laser level lifetime
        self.tau_1 = tau_1  # Lower laser level lifetime
        self.tau_c = tau_c
        self.gamma = gamma
        self.beta = beta
        self.c = 3e8

        # Threshold (N1 ≈ 0 for four-level)
        self.n_th = 1.0 / (gamma * sigma * self.c * tau_c)

    def rate_equations(self, y, t, R_pump):
        """
        Rate equations for four-level laser.
        Assumes lower level (N1) relaxes very fast, so N1 ≈ 0.

        Args:
            y: [N2, phi] - upper state population, photon number

        Returns:
            [dN2/dt, dphi/dt]
        """
        N2, phi = y
        N2 = max(0, N2)
        phi = max(0, phi)

        # N1 ≈ 0 for four-level system (fast relaxation)
        # So delta_N ≈ N2

        dN2_dt = (R_pump - N2 / self.tau_2 -
                  self.sigma * self.c * phi * N2)

        dphi_dt = (self.gamma * self.sigma * self.c * N2 * phi -
                   phi / self.tau_c +
                   self.beta * N2 / self.tau_2)

        return [dN2_dt, dphi_dt]

    def simulate(self, R_pump, t_span, y0=None):
        """Simulate laser dynamics."""
        if y0 is None:
            y0 = [0, 1]

        if callable(R_pump):
            solution = odeint(lambda y, t: self.rate_equations(y, t, R_pump(t)),
                            y0, t_span)
        else:
            solution = odeint(lambda y, t: self.rate_equations(y, t, R_pump),
                            y0, t_span)

        return {
            't': t_span,
            'N2': solution[:, 0],
            'phi': solution[:, 1],
            'power': solution[:, 1] / self.tau_c
        }

    def relaxation_frequency(self, R_pump):
        """
        Calculate relaxation oscillation frequency.

        omega_r = sqrt((R/R_th - 1) / (tau_c * tau_2))
        """
        R_th = self.n_th / self.tau_2
        if R_pump <= R_th:
            return 0

        omega_r = np.sqrt((R_pump / R_th - 1) / (self.tau_c * self.tau_2))
        return omega_r / (2 * np.pi)  # Return in Hz


def plot_steady_state():
    """Plot steady-state laser characteristics"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    laser = FourLevelLaser()

    # Plot 1: Output power vs pump rate (threshold curve)
    ax1 = axes[0, 0]

    R_th = laser.n_th / laser.tau_2
    R_range = np.linspace(0, 5 * R_th, 100)

    # Steady-state output power
    # Above threshold: phi_ss = (R - R_th) * tau_c * tau_2 / (sigma * c * tau_2)
    power_ss = np.zeros_like(R_range)
    above_th = R_range > R_th
    power_ss[above_th] = (R_range[above_th] - R_th) * laser.tau_c

    ax1.plot(R_range / R_th, power_ss / power_ss.max(), 'b-', linewidth=2)
    ax1.axvline(1, color='red', linestyle='--', label='Threshold')
    ax1.axhline(0, color='black', linewidth=0.5)

    ax1.set_xlabel('Pump rate (R / R_th)')
    ax1.set_ylabel('Output power (normalized)')
    ax1.set_title('Laser Output vs Pump Rate\n(Threshold behavior)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Population inversion at steady state
    ax2 = axes[0, 1]

    N2_ss = np.zeros_like(R_range)

    # Below threshold: N2 = R * tau_2
    # Above threshold: N2 ≈ N_th (clamped)
    below_th = R_range <= R_th
    N2_ss[below_th] = R_range[below_th] * laser.tau_2
    N2_ss[above_th] = laser.n_th

    ax2.plot(R_range / R_th, N2_ss / laser.n_th, 'b-', linewidth=2)
    ax2.axvline(1, color='red', linestyle='--', label='Threshold')
    ax2.axhline(1, color='green', linestyle=':', label='N_th')

    ax2.set_xlabel('Pump rate (R / R_th)')
    ax2.set_ylabel('Population inversion (N2 / N_th)')
    ax2.set_title('Population Inversion Clamping')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Slope efficiency
    ax3 = axes[1, 0]

    # Differential quantum efficiency
    # eta_d = (tau_c / tau_2) * (1 - R_th/R) for R > R_th
    eta_d = np.zeros_like(R_range)
    eta_d[above_th] = 1 - R_th / R_range[above_th]

    ax3.plot(R_range / R_th, eta_d * 100, 'b-', linewidth=2)
    ax3.axvline(1, color='red', linestyle='--')

    ax3.set_xlabel('Pump rate (R / R_th)')
    ax3.set_ylabel('Differential efficiency (%)')
    ax3.set_title('Laser Slope Efficiency')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 5)

    # Plot 4: Relaxation oscillation frequency
    ax4 = axes[1, 1]

    freq_relax = np.array([laser.relaxation_frequency(R) for R in R_range])

    ax4.plot(R_range / R_th, freq_relax / 1e3, 'b-', linewidth=2)
    ax4.axvline(1, color='red', linestyle='--', label='Threshold')

    ax4.set_xlabel('Pump rate (R / R_th)')
    ax4.set_ylabel('Relaxation oscillation frequency (kHz)')
    ax4.set_title('Relaxation Oscillation Frequency\n$f_r = \\sqrt{(R/R_{th}-1)/(\\tau_c \\tau_2)} / 2\\pi$')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 5)

    plt.tight_layout()
    return fig


def plot_laser_dynamics():
    """Plot laser turn-on dynamics and relaxation oscillations"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    laser = FourLevelLaser()
    R_th = laser.n_th / laser.tau_2

    # Plot 1: Turn-on dynamics
    ax1 = axes[0, 0]

    t_span = np.linspace(0, 50e-6, 5000)

    pump_levels = [1.5, 2.0, 3.0, 5.0]  # Times threshold
    colors = plt.cm.viridis(np.linspace(0, 1, len(pump_levels)))

    for pump_mult, color in zip(pump_levels, colors):
        R_pump = pump_mult * R_th
        result = laser.simulate(R_pump, t_span)

        # Normalize power
        power_norm = result['phi'] / result['phi'].max()
        ax1.plot(t_span * 1e6, power_norm, color=color, linewidth=1.5,
                label=f'R = {pump_mult}*R_th')

    ax1.set_xlabel('Time (us)')
    ax1.set_ylabel('Output power (normalized)')
    ax1.set_title('Laser Turn-On Dynamics')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Population dynamics
    ax2 = axes[0, 1]

    R_pump = 2.0 * R_th
    result = laser.simulate(R_pump, t_span)

    ax2.plot(t_span * 1e6, result['N2'] / laser.n_th, 'b-', linewidth=2, label='N2 / N_th')
    ax2.axhline(1, color='red', linestyle='--', label='Threshold inversion')

    ax2.set_xlabel('Time (us)')
    ax2.set_ylabel('Population (normalized)')
    ax2.set_title(f'Population Inversion Dynamics (R = 2*R_th)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Phase portrait
    ax3 = axes[1, 0]

    for pump_mult, color in zip(pump_levels[1:], colors[1:]):
        R_pump = pump_mult * R_th
        result = laser.simulate(R_pump, t_span)

        ax3.plot(result['N2'] / laser.n_th, result['phi'] / result['phi'].max(),
                color=color, linewidth=1, alpha=0.7,
                label=f'R = {pump_mult}*R_th')

    ax3.set_xlabel('Population inversion (N2 / N_th)')
    ax3.set_ylabel('Photon number (normalized)')
    ax3.set_title('Phase Portrait: Relaxation Oscillations')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Pump modulation response
    ax4 = axes[1, 1]

    t_long = np.linspace(0, 100e-6, 10000)

    # Modulated pump
    R_dc = 2.0 * R_th
    modulation_depth = 0.1
    mod_freq = 100e3  # 100 kHz

    def modulated_pump(t):
        return R_dc * (1 + modulation_depth * np.sin(2 * np.pi * mod_freq * t))

    result_mod = laser.simulate(modulated_pump, t_long, y0=[laser.n_th, 1e10])

    # Skip transient
    t_plot = t_long[5000:]
    phi_plot = result_mod['phi'][5000:]

    ax4.plot(t_plot * 1e6, phi_plot / phi_plot.mean(), 'b-', linewidth=1)

    ax4.set_xlabel('Time (us)')
    ax4.set_ylabel('Output power (normalized to mean)')
    ax4.set_title(f'Response to Pump Modulation\n(f_mod = {mod_freq/1e3:.0f} kHz, 10% depth)')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_three_level_comparison():
    """Compare three-level and four-level laser dynamics"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Create both laser types with comparable parameters
    three_level = ThreeLevelLaser(
        sigma=2.5e-20,
        tau_21=3e-3,
        tau_c=10e-9,
        n_total=1.6e25
    )

    four_level = FourLevelLaser(
        sigma=6.5e-23,
        tau_2=230e-6,
        tau_c=10e-9
    )

    # Plot 1: Threshold comparison
    ax1 = axes[0, 0]

    # Three-level threshold
    R_th_3 = three_level.R_th
    # Four-level threshold
    R_th_4 = four_level.n_th / four_level.tau_2

    labels = ['Three-level\n(Ruby-like)', 'Four-level\n(Nd:YAG-like)']
    thresholds = [R_th_3, R_th_4]

    bars = ax1.bar(labels, thresholds, color=['red', 'blue'], alpha=0.7)

    ax1.set_ylabel('Threshold pump rate (m^-3 s^-1)')
    ax1.set_title('Threshold Comparison')
    ax1.set_yscale('log')

    for bar, val in zip(bars, thresholds):
        ax1.text(bar.get_x() + bar.get_width()/2, val * 1.5,
                f'{val:.2e}', ha='center', fontsize=10)

    # Plot 2: Turn-on delay comparison
    ax2 = axes[0, 1]

    t_span = np.linspace(0, 1e-3, 10000)

    # Pump both at 2x threshold
    result_3 = three_level.simulate(2 * R_th_3, t_span)
    result_4 = four_level.simulate(2 * R_th_4, t_span)

    ax2.plot(t_span * 1e6, result_3['phi'] / result_3['phi'].max(),
            'r-', linewidth=2, label='Three-level')
    ax2.plot(t_span * 1e6, result_4['phi'] / result_4['phi'].max(),
            'b-', linewidth=2, label='Four-level')

    ax2.set_xlabel('Time (us)')
    ax2.set_ylabel('Output power (normalized)')
    ax2.set_title('Turn-On Dynamics at 2x Threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 100)

    # Plot 3: Efficiency comparison
    ax3 = axes[1, 0]

    pump_ratios = np.linspace(1.01, 5, 100)

    # Quantum efficiency (simplified)
    eta_3 = 1 - 1 / pump_ratios  # Three-level (needs 50% inversion)
    eta_4 = 1 - 1 / pump_ratios  # Four-level

    ax3.plot(pump_ratios, eta_3 * 100, 'r-', linewidth=2, label='Three-level')
    ax3.plot(pump_ratios, eta_4 * 100, 'b-', linewidth=2, label='Four-level')

    ax3.set_xlabel('Pump rate (R / R_th)')
    ax3.set_ylabel('Quantum efficiency (%)')
    ax3.set_title('Quantum Efficiency vs Pump Rate')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Energy level diagrams
    ax4 = axes[1, 1]

    # Three-level system
    levels_3 = [0, 0.6, 1.0]
    for i, E in enumerate(levels_3):
        ax4.hlines(E, 0.1, 0.4, colors='red', linewidth=3)
        ax4.text(0.05, E, f'E{i}', fontsize=10, va='center')

    # Transitions
    ax4.annotate('', xy=(0.25, levels_3[2]-0.02), xytext=(0.25, levels_3[0]+0.02),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax4.text(0.26, 0.5, 'Pump', fontsize=9, color='green')

    ax4.annotate('', xy=(0.35, levels_3[1]+0.02), xytext=(0.35, levels_3[2]-0.02),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax4.text(0.36, 0.75, 'Lasing', fontsize=9, color='red')

    ax4.text(0.25, 1.1, 'Three-Level', ha='center', fontsize=12, fontweight='bold')

    # Four-level system
    levels_4 = [0, 0.3, 0.7, 1.0]
    for i, E in enumerate(levels_4):
        ax4.hlines(E, 0.6, 0.9, colors='blue', linewidth=3)
        ax4.text(0.55, E, f'E{i}', fontsize=10, va='center')

    # Transitions
    ax4.annotate('', xy=(0.75, levels_4[3]-0.02), xytext=(0.75, levels_4[0]+0.02),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax4.text(0.76, 0.5, 'Pump', fontsize=9, color='green')

    ax4.annotate('', xy=(0.85, levels_4[1]+0.02), xytext=(0.85, levels_4[2]-0.02),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax4.text(0.86, 0.45, 'Lasing', fontsize=9, color='blue')

    ax4.annotate('', xy=(0.65, levels_4[0]+0.02), xytext=(0.65, levels_4[1]-0.02),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1))
    ax4.text(0.66, 0.1, 'Fast\ndecay', fontsize=8, color='gray')

    ax4.text(0.75, 1.1, 'Four-Level', ha='center', fontsize=12, fontweight='bold')

    ax4.set_xlim(0, 1)
    ax4.set_ylim(-0.1, 1.2)
    ax4.axis('off')
    ax4.set_title('Energy Level Diagrams')

    plt.tight_layout()
    return fig


def plot_q_switching():
    """Demonstrate Q-switched laser dynamics"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    laser = FourLevelLaser(
        sigma=6.5e-23,
        tau_2=230e-6,
        tau_c=10e-9
    )

    R_th = laser.n_th / laser.tau_2
    R_pump = 5 * R_th  # Strong pumping

    # Plot 1: Q-switch dynamics
    ax1 = axes[0, 0]

    t_span = np.linspace(0, 500e-6, 10000)

    # High losses during pump-up (low Q)
    # Then switch to high Q at t_switch
    t_switch = 200e-6

    def q_switched_rate(y, t, R_pump, t_switch, tau_c_low, tau_c_high):
        N2, phi = y
        N2 = max(0, N2)
        phi = max(0, phi)

        tau_c = tau_c_low if t < t_switch else tau_c_high

        dN2_dt = (R_pump - N2 / laser.tau_2 -
                  laser.sigma * laser.c * phi * N2)

        dphi_dt = (laser.gamma * laser.sigma * laser.c * N2 * phi -
                   phi / tau_c +
                   laser.beta * N2 / laser.tau_2)

        return [dN2_dt, dphi_dt]

    tau_c_low = 0.1e-9   # Low Q (high loss)
    tau_c_high = 10e-9   # High Q

    solution = odeint(q_switched_rate, [0, 1], t_span,
                     args=(R_pump, t_switch, tau_c_low, tau_c_high))

    N2 = solution[:, 0]
    phi = solution[:, 1]

    ax1.plot(t_span * 1e6, N2 / laser.n_th, 'b-', linewidth=2, label='Population N2')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(t_span * 1e6, phi / phi.max(), 'r-', linewidth=2, label='Photon number')

    ax1.axvline(t_switch * 1e6, color='green', linestyle='--', label='Q-switch')
    ax1.set_xlabel('Time (us)')
    ax1.set_ylabel('Population (N2 / N_th)', color='blue')
    ax1_twin.set_ylabel('Photon number (normalized)', color='red')
    ax1.set_title('Q-Switched Laser Dynamics')
    ax1.legend(loc='upper left')

    # Plot 2: Pulse close-up
    ax2 = axes[0, 1]

    # Zoom to pulse region
    pulse_region = (t_span > 199e-6) & (t_span < 210e-6)
    t_pulse = t_span[pulse_region]
    phi_pulse = phi[pulse_region]
    N2_pulse = N2[pulse_region]

    ax2.plot((t_pulse - t_switch) * 1e9, phi_pulse / phi_pulse.max(),
            'r-', linewidth=2, label='Photon number')
    ax2.plot((t_pulse - t_switch) * 1e9, N2_pulse / N2_pulse.max(),
            'b-', linewidth=2, label='Population')

    ax2.set_xlabel('Time after Q-switch (ns)')
    ax2.set_ylabel('Normalized value')
    ax2.set_title('Q-Switched Pulse Detail')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Pulse energy vs initial inversion
    ax3 = axes[1, 0]

    initial_inversions = np.linspace(1, 10, 50) * laser.n_th

    pulse_energies = []
    pulse_widths = []

    for N2_0 in initial_inversions:
        t_short = np.linspace(0, 200e-9, 2000)
        solution = odeint(lambda y, t: [
            -laser.sigma * laser.c * y[1] * y[0],
            laser.gamma * laser.sigma * laser.c * y[0] * y[1] - y[1] / tau_c_high
        ], [N2_0, 1], t_short)

        phi_t = solution[:, 1]
        pulse_energies.append(np.trapz(phi_t, t_short))

        # Estimate pulse width (FWHM)
        half_max = phi_t.max() / 2
        above_half = phi_t > half_max
        if above_half.any():
            fwhm_idx = np.where(above_half)[0]
            if len(fwhm_idx) > 1:
                pulse_widths.append((fwhm_idx[-1] - fwhm_idx[0]) * (t_short[1] - t_short[0]))
            else:
                pulse_widths.append(0)
        else:
            pulse_widths.append(0)

    ax3.plot(initial_inversions / laser.n_th, np.array(pulse_energies) / max(pulse_energies),
            'b-', linewidth=2)

    ax3.set_xlabel('Initial inversion (N2_0 / N_th)')
    ax3.set_ylabel('Pulse energy (normalized)')
    ax3.set_title('Pulse Energy vs Initial Inversion')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Pulse width vs initial inversion
    ax4 = axes[1, 1]

    ax4.plot(initial_inversions / laser.n_th, np.array(pulse_widths) * 1e9,
            'r-', linewidth=2)

    ax4.set_xlabel('Initial inversion (N2_0 / N_th)')
    ax4.set_ylabel('Pulse width FWHM (ns)')
    ax4.set_title('Pulse Width vs Initial Inversion')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    """Main function to demonstrate laser rate equations"""

    # Create figures
    fig1 = plot_steady_state()
    fig2 = plot_laser_dynamics()
    fig3 = plot_three_level_comparison()
    fig4 = plot_q_switching()

    # Save plots
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)

    fig1.savefig(os.path.join(output_dir, 'laser_rate_steady_state.png'),
                 dpi=150, bbox_inches='tight')
    fig2.savefig(os.path.join(output_dir, 'laser_rate_dynamics.png'),
                 dpi=150, bbox_inches='tight')
    fig3.savefig(os.path.join(output_dir, 'laser_rate_comparison.png'),
                 dpi=150, bbox_inches='tight')
    fig4.savefig(os.path.join(output_dir, 'laser_rate_q_switching.png'),
                 dpi=150, bbox_inches='tight')

    print(f"Plots saved to {output_dir}/laser_rate_*.png")

    # Print analysis
    print("\n=== Laser Rate Equations Analysis ===")

    laser = FourLevelLaser()
    R_th = laser.n_th / laser.tau_2

    print(f"\nFour-level laser parameters:")
    print(f"  Stimulated emission cross-section: {laser.sigma:.2e} m^2")
    print(f"  Upper state lifetime: {laser.tau_2*1e6:.0f} us")
    print(f"  Cavity photon lifetime: {laser.tau_c*1e9:.0f} ns")
    print(f"  Threshold inversion: {laser.n_th:.2e} m^-3")
    print(f"  Threshold pump rate: {R_th:.2e} m^-3 s^-1")

    print(f"\nRelaxation oscillation frequencies (at various pump levels):")
    for mult in [1.5, 2.0, 3.0, 5.0]:
        freq = laser.relaxation_frequency(mult * R_th)
        print(f"  R = {mult}*R_th: f_r = {freq/1e3:.1f} kHz")


if __name__ == "__main__":
    main()
