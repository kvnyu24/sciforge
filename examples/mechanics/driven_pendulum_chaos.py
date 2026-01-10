"""
Experiment 46: Driven Damped Pendulum - Chaos

This example demonstrates chaotic behavior in a driven damped pendulum.
Shows how periodic driving can lead to chaos, including sensitivity to
initial conditions, strange attractors, and Poincare sections.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


class DrivenDampedPendulum:
    """
    Driven damped pendulum with equation of motion:

    d^2(theta)/dt^2 + gamma*d(theta)/dt + omega_0^2*sin(theta) = A*cos(omega_d*t)

    where:
    - gamma is the damping coefficient
    - omega_0 = sqrt(g/L) is the natural frequency
    - A is the driving amplitude
    - omega_d is the driving frequency
    """

    def __init__(self, omega_0, gamma, A, omega_d, theta0, omega0):
        """
        Initialize driven damped pendulum.

        Args:
            omega_0: Natural frequency sqrt(g/L)
            gamma: Damping coefficient
            A: Driving amplitude
            omega_d: Driving frequency
            theta0: Initial angle
            omega0: Initial angular velocity
        """
        self.omega_0 = omega_0
        self.gamma = gamma
        self.A = A
        self.omega_d = omega_d
        self.theta = theta0
        self.omega = omega0
        self.t = 0

    def acceleration(self, theta, omega, t):
        """Calculate angular acceleration."""
        return -self.gamma * omega - self.omega_0**2 * np.sin(theta) + self.A * np.cos(self.omega_d * t)

    def update(self, dt):
        """Update pendulum state using RK4."""
        # RK4 integration
        k1_theta = self.omega
        k1_omega = self.acceleration(self.theta, self.omega, self.t)

        k2_theta = self.omega + 0.5 * dt * k1_omega
        k2_omega = self.acceleration(self.theta + 0.5 * dt * k1_theta,
                                       self.omega + 0.5 * dt * k1_omega,
                                       self.t + 0.5 * dt)

        k3_theta = self.omega + 0.5 * dt * k2_omega
        k3_omega = self.acceleration(self.theta + 0.5 * dt * k2_theta,
                                       self.omega + 0.5 * dt * k2_omega,
                                       self.t + 0.5 * dt)

        k4_theta = self.omega + dt * k3_omega
        k4_omega = self.acceleration(self.theta + dt * k3_theta,
                                       self.omega + dt * k3_omega,
                                       self.t + dt)

        self.theta += (dt / 6) * (k1_theta + 2*k2_theta + 2*k3_theta + k4_theta)
        self.omega += (dt / 6) * (k1_omega + 2*k2_omega + 2*k3_omega + k4_omega)
        self.t += dt

        # Keep theta in [-pi, pi]
        while self.theta > np.pi:
            self.theta -= 2 * np.pi
        while self.theta < -np.pi:
            self.theta += 2 * np.pi


def simulate_pendulum(omega_0, gamma, A, omega_d, theta0, omega0, t_final, dt):
    """Simulate driven damped pendulum motion."""
    pend = DrivenDampedPendulum(omega_0, gamma, A, omega_d, theta0, omega0)

    times = [0]
    thetas = [theta0]
    omegas = [omega0]

    while pend.t < t_final:
        pend.update(dt)
        times.append(pend.t)
        thetas.append(pend.theta)
        omegas.append(pend.omega)

    return {
        'time': np.array(times),
        'theta': np.array(thetas),
        'omega': np.array(omegas)
    }


def poincare_section(omega_0, gamma, A, omega_d, theta0, omega0, n_periods, dt):
    """
    Create Poincare section by sampling at each drive period.

    Returns points (theta, omega) at t = n * T_drive for integer n.
    """
    T_drive = 2 * np.pi / omega_d
    t_final = n_periods * T_drive

    pend = DrivenDampedPendulum(omega_0, gamma, A, omega_d, theta0, omega0)

    poincare_theta = []
    poincare_omega = []

    next_sample = T_drive
    transient_periods = min(100, n_periods // 4)  # Skip transient

    while pend.t < t_final:
        pend.update(dt)

        if pend.t >= next_sample:
            if next_sample >= transient_periods * T_drive:
                poincare_theta.append(pend.theta)
                poincare_omega.append(pend.omega)
            next_sample += T_drive

    return np.array(poincare_theta), np.array(poincare_omega)


def lyapunov_exponent(omega_0, gamma, A, omega_d, theta0, omega0, t_final, dt, delta0=1e-8):
    """
    Estimate the largest Lyapunov exponent by tracking nearby trajectories.
    """
    # Reference trajectory
    pend1 = DrivenDampedPendulum(omega_0, gamma, A, omega_d, theta0, omega0)

    # Perturbed trajectory
    pend2 = DrivenDampedPendulum(omega_0, gamma, A, omega_d, theta0 + delta0, omega0)

    lyap_sum = 0
    n_samples = 0
    renorm_interval = 1.0  # Renormalize every 1 second

    t_renorm = renorm_interval

    while pend1.t < t_final:
        pend1.update(dt)
        pend2.update(dt)

        if pend1.t >= t_renorm:
            # Calculate separation
            d_theta = pend2.theta - pend1.theta
            d_omega = pend2.omega - pend1.omega
            d = np.sqrt(d_theta**2 + d_omega**2)

            if d > 0:
                lyap_sum += np.log(d / delta0)
                n_samples += 1

                # Renormalize
                factor = delta0 / d
                pend2.theta = pend1.theta + d_theta * factor
                pend2.omega = pend1.omega + d_omega * factor

            t_renorm += renorm_interval

    if n_samples > 0:
        return lyap_sum / (n_samples * renorm_interval)
    return 0


def main():
    # Standard parameters (adjusted for chaos)
    omega_0 = 1.5  # Natural frequency
    gamma = 0.5    # Damping
    omega_d = 2/3  # Driving frequency (often 2/3 of natural for chaos)

    dt = 0.01
    t_final = 200.0

    # Create figure
    fig = plt.figure(figsize=(16, 14))

    # Case 1: Transition from periodic to chaotic with increasing drive
    ax1 = fig.add_subplot(3, 3, 1)

    A_values = [0.5, 0.9, 1.2, 1.5]
    colors = plt.cm.viridis(np.linspace(0, 1, len(A_values)))

    for A, color in zip(A_values, colors):
        results = simulate_pendulum(omega_0, gamma, A, omega_d, 0.1, 0, 100.0, dt)
        # Plot only last portion (after transient)
        mask = results['time'] > 60
        ax1.plot(results['time'][mask], results['theta'][mask],
                 color=color, lw=0.5, label=f'A = {A}')

    ax1.set_xlabel('Time')
    ax1.set_ylabel('Angle (rad)')
    ax1.set_title('Transition to Chaos with Increasing Drive')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Case 2: Sensitivity to initial conditions (butterfly effect)
    ax2 = fig.add_subplot(3, 3, 2)

    A_chaos = 1.2
    theta0_base = 0.1
    perturbations = [0, 1e-6, 1e-5, 1e-4]
    colors_pert = plt.cm.Reds(np.linspace(0.3, 1, len(perturbations)))

    for delta, color in zip(perturbations, colors_pert):
        results = simulate_pendulum(omega_0, gamma, A_chaos, omega_d,
                                     theta0_base + delta, 0, 100.0, dt)
        label = f'delta = {delta:.0e}' if delta > 0 else 'Reference'
        ax2.plot(results['time'], results['theta'],
                 color=color, lw=1, label=label)

    ax2.set_xlabel('Time')
    ax2.set_ylabel('Angle (rad)')
    ax2.set_title('Sensitivity to Initial Conditions\n(Chaotic regime: A=1.2)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 100)

    # Case 3: Phase space trajectory (chaotic)
    ax3 = fig.add_subplot(3, 3, 3)

    results_chaos = simulate_pendulum(omega_0, gamma, A_chaos, omega_d, 0.1, 0, 200.0, dt)
    mask = results_chaos['time'] > 50  # Skip transient
    ax3.plot(results_chaos['theta'][mask], results_chaos['omega'][mask],
             'b-', lw=0.3, alpha=0.5)
    ax3.set_xlabel('Angle (rad)')
    ax3.set_ylabel('Angular velocity (rad/s)')
    ax3.set_title('Chaotic Phase Space Trajectory')
    ax3.grid(True, alpha=0.3)

    # Case 4: Poincare section for different driving amplitudes
    ax4 = fig.add_subplot(3, 3, 4)

    A_poincare_values = [0.5, 0.9, 1.2, 1.5]
    colors_poincare = ['blue', 'green', 'orange', 'red']

    for A, color in zip(A_poincare_values, colors_poincare):
        theta_p, omega_p = poincare_section(omega_0, gamma, A, omega_d,
                                             0.1, 0, 1000, dt)
        ax4.scatter(theta_p, omega_p, s=1, c=color, alpha=0.5, label=f'A = {A}')

    ax4.set_xlabel('Angle (rad)')
    ax4.set_ylabel('Angular velocity (rad/s)')
    ax4.set_title('Poincare Sections')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # Case 5: Strange attractor detail
    ax5 = fig.add_subplot(3, 3, 5)

    theta_p, omega_p = poincare_section(omega_0, gamma, 1.2, omega_d,
                                         0.1, 0, 5000, dt*0.5)
    ax5.scatter(theta_p, omega_p, s=1, c='blue', alpha=0.3)
    ax5.set_xlabel('Angle (rad)')
    ax5.set_ylabel('Angular velocity (rad/s)')
    ax5.set_title('Strange Attractor (A=1.2, 5000 periods)')
    ax5.grid(True, alpha=0.3)

    # Case 6: Bifurcation diagram
    ax6 = fig.add_subplot(3, 3, 6)

    A_range = np.linspace(0.2, 1.6, 100)
    n_periods_bif = 500
    T_drive = 2 * np.pi / omega_d

    print("Computing bifurcation diagram...")
    for i, A in enumerate(A_range):
        if i % 20 == 0:
            print(f"  A = {A:.2f}")

        pend = DrivenDampedPendulum(omega_0, gamma, A, omega_d, 0.1, 0)

        # Skip transient
        for _ in range(int(300 * T_drive / dt)):
            pend.update(dt)

        # Collect Poincare points
        thetas_bif = []
        next_sample = pend.t + T_drive

        while len(thetas_bif) < 50:
            pend.update(dt)
            if pend.t >= next_sample:
                thetas_bif.append(pend.theta)
                next_sample += T_drive

        ax6.scatter([A] * len(thetas_bif), thetas_bif, s=0.5, c='black', alpha=0.3)

    ax6.set_xlabel('Driving Amplitude A')
    ax6.set_ylabel('Angle at t = nT (rad)')
    ax6.set_title('Bifurcation Diagram')
    ax6.grid(True, alpha=0.3)

    # Case 7: Power spectrum (regular vs chaotic)
    ax7 = fig.add_subplot(3, 3, 7)

    for A, label, color in [(0.5, 'Periodic', 'blue'), (1.2, 'Chaotic', 'red')]:
        results = simulate_pendulum(omega_0, gamma, A, omega_d, 0.1, 0, 500.0, dt)
        mask = results['time'] > 100
        theta_signal = results['theta'][mask]

        # FFT
        N = len(theta_signal)
        fft_vals = np.abs(np.fft.rfft(theta_signal))
        freqs = np.fft.rfftfreq(N, dt)

        ax7.semilogy(freqs[:N//4], fft_vals[:N//4], color=color, lw=1, label=label)

    ax7.axvline(x=omega_d/(2*np.pi), color='g', linestyle='--',
                alpha=0.5, label='Drive freq')
    ax7.set_xlabel('Frequency (Hz)')
    ax7.set_ylabel('Power')
    ax7.set_title('Power Spectrum')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.set_xlim(0, 2)

    # Case 8: Lyapunov exponent vs driving amplitude
    ax8 = fig.add_subplot(3, 3, 8)

    A_lyap_range = np.linspace(0.3, 1.6, 20)
    lyap_exponents = []

    print("Computing Lyapunov exponents...")
    for A in A_lyap_range:
        lyap = lyapunov_exponent(omega_0, gamma, A, omega_d, 0.1, 0, 200.0, dt)
        lyap_exponents.append(lyap)

    ax8.plot(A_lyap_range, lyap_exponents, 'b.-', lw=2, markersize=8)
    ax8.axhline(y=0, color='r', linestyle='--', label='Chaos threshold')
    ax8.fill_between(A_lyap_range, lyap_exponents, 0,
                     where=(np.array(lyap_exponents) > 0),
                     alpha=0.3, color='red', label='Chaotic region')
    ax8.set_xlabel('Driving Amplitude A')
    ax8.set_ylabel('Lyapunov Exponent')
    ax8.set_title('Lyapunov Exponent (positive = chaos)')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # Case 9: Time series comparison (periodic vs chaotic)
    ax9 = fig.add_subplot(3, 3, 9)

    results_periodic = simulate_pendulum(omega_0, gamma, 0.5, omega_d, 0.1, 0, 50.0, dt)
    results_chaotic = simulate_pendulum(omega_0, gamma, 1.2, omega_d, 0.1, 0, 50.0, dt)

    ax9.plot(results_periodic['time'], results_periodic['theta'] + 4,
             'b-', lw=1, label='Periodic (A=0.5)')
    ax9.plot(results_chaotic['time'], results_chaotic['theta'],
             'r-', lw=1, label='Chaotic (A=1.2)')
    ax9.set_xlabel('Time')
    ax9.set_ylabel('Angle (rad) + offset')
    ax9.set_title('Periodic vs Chaotic Motion')
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    plt.suptitle('Driven Damped Pendulum: Route to Chaos\n'
                 f'omega_0={omega_0}, gamma={gamma}, omega_d={omega_d}',
                 fontsize=12, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'driven_pendulum_chaos.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'driven_pendulum_chaos.png')}")


if __name__ == "__main__":
    main()
