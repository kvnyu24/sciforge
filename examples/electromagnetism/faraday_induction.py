"""
Experiment 89: Faraday induction.

This example demonstrates Faraday's law of electromagnetic induction,
showing how a changing magnetic flux through a loop induces an EMF.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt


# Constants
MU_0 = 4 * np.pi * 1e-7  # Permeability of free space (H/m)


def calculate_flux(B, A, theta=0):
    """
    Calculate magnetic flux through a loop.

    Phi = B * A * cos(theta)

    Args:
        B: Magnetic field magnitude (T)
        A: Loop area (m^2)
        theta: Angle between B and area normal (rad)

    Returns:
        Phi: Magnetic flux (Wb)
    """
    return B * A * np.cos(theta)


def induced_emf(dPhi_dt):
    """
    Calculate induced EMF from Faraday's law.

    EMF = -dPhi/dt
    """
    return -dPhi_dt


class RotatingLoop:
    """Simulation of a rotating loop in a uniform magnetic field."""

    def __init__(self, B, A, omega, N=1):
        """
        Args:
            B: Magnetic field magnitude (T)
            A: Loop area (m^2)
            omega: Angular velocity (rad/s)
            N: Number of turns
        """
        self.B = B
        self.A = A
        self.omega = omega
        self.N = N

    def flux(self, t):
        """Flux at time t."""
        theta = self.omega * t
        return self.N * self.B * self.A * np.cos(theta)

    def emf(self, t):
        """Induced EMF at time t."""
        # EMF = -dPhi/dt = N*B*A*omega*sin(omega*t)
        return self.N * self.B * self.A * self.omega * np.sin(self.omega * t)


class ApproachingMagnet:
    """Simulation of a magnet approaching a coil."""

    def __init__(self, m, R, v, N=100, A=0.01):
        """
        Args:
            m: Magnetic dipole moment (A*m^2)
            R: Coil radius (m)
            v: Approach velocity (m/s)
            N: Number of turns
            A: Coil area (m^2)
        """
        self.m = m
        self.R = R
        self.v = v
        self.N = N
        self.A = A

    def B_on_axis(self, z):
        """Magnetic field on axis of dipole at distance z."""
        z = np.maximum(np.abs(z), 0.01)
        return MU_0 * 2 * self.m / (4 * np.pi * z**3)

    def flux(self, t, z0=1.0):
        """Flux through coil as magnet approaches."""
        z = z0 - self.v * t
        z = np.maximum(z, 0.02)
        B = self.B_on_axis(z)
        return self.N * B * self.A

    def emf(self, t, z0=1.0, dt=1e-4):
        """EMF from numerical derivative of flux."""
        Phi_plus = self.flux(t + dt/2, z0)
        Phi_minus = self.flux(t - dt/2, z0)
        return -(Phi_plus - Phi_minus) / dt


def main():
    fig = plt.figure(figsize=(16, 12))

    # Simulation 1: Rotating loop (AC generator)
    ax1 = fig.add_subplot(2, 2, 1)

    B = 0.5       # 0.5 T magnetic field
    A = 0.01      # 100 cm^2 loop area
    omega = 2 * np.pi * 60  # 60 Hz rotation
    N = 100       # 100 turns

    generator = RotatingLoop(B, A, omega, N)

    t = np.linspace(0, 0.05, 1000)  # 50 ms = 3 cycles at 60 Hz
    flux = generator.flux(t)
    emf = generator.emf(t)

    ax1.plot(t * 1000, flux * 1000, 'b-', lw=2, label='Flux (mWb)')
    ax1.plot(t * 1000, emf, 'r-', lw=2, label='EMF (V)')

    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Flux (mWb) / EMF (V)')
    ax1.set_title('AC Generator: Rotating Loop in Uniform Field\n'
                  f'B = {B} T, A = {A*1e4} cm^2, N = {N}, f = 60 Hz')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Annotate peak EMF
    emf_peak = N * B * A * omega
    ax1.text(0.95, 0.95, f'Peak EMF = N*B*A*omega\n= {emf_peak:.1f} V',
             transform=ax1.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Simulation 2: Magnet approaching coil
    ax2 = fig.add_subplot(2, 2, 2)

    m = 1.0       # 1 A*m^2 dipole moment
    R = 0.02      # 2 cm coil radius
    v = 0.5       # 0.5 m/s approach velocity
    N_coil = 200
    A_coil = np.pi * R**2

    magnet = ApproachingMagnet(m, R, v, N_coil, A_coil)

    t2 = np.linspace(0, 1.5, 500)
    z0 = 1.0  # Start 1 m away

    flux2 = np.array([magnet.flux(ti, z0) for ti in t2])
    emf2 = np.array([magnet.emf(ti, z0) for ti in t2])

    ax2.plot(t2, flux2 * 1000, 'b-', lw=2, label='Flux (mWb)')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(t2, emf2 * 1000, 'r-', lw=2, label='EMF (mV)')

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Flux (mWb)', color='b')
    ax2_twin.set_ylabel('Induced EMF (mV)', color='r')
    ax2.set_title('Magnet Approaching Coil\n'
                  f'm = {m} A*m^2, v = {v} m/s, N = {N_coil}')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2_twin.tick_params(axis='y', labelcolor='r')

    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Simulation 3: Varying area (expanding loop)
    ax3 = fig.add_subplot(2, 2, 3)

    B3 = 0.1  # 0.1 T uniform field
    R0 = 0.05  # Initial radius 5 cm
    v_expand = 0.1  # Expansion velocity 10 cm/s

    t3 = np.linspace(0, 1, 500)
    R3 = R0 + v_expand * t3
    A3 = np.pi * R3**2
    flux3 = B3 * A3

    # dPhi/dt = B * dA/dt = B * 2*pi*R*dR/dt = B * 2*pi*R*v
    dPhi_dt3 = B3 * 2 * np.pi * R3 * v_expand
    emf3 = -dPhi_dt3

    ax3.plot(t3, R3 * 100, 'g-', lw=2, label='Radius (cm)')
    ax3.plot(t3, flux3 * 1000, 'b-', lw=2, label='Flux (mWb)')
    ax3.plot(t3, emf3 * 1000, 'r-', lw=2, label='EMF (mV)')

    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Various units')
    ax3.set_title('Expanding Loop in Uniform Field\n'
                  f'B = {B3} T, v_expand = {v_expand*100} cm/s')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Simulation 4: Faraday's law with multiple scenarios
    ax4 = fig.add_subplot(2, 2, 4)

    # Compare different scenarios at t=0
    scenarios = []

    # Scenario 1: Rotating loop
    omega_test = 100  # rad/s
    emf_rot = N * B * A * omega_test
    scenarios.append(('Rotating loop\n(60 Hz)', emf_peak))

    # Scenario 2: Changing B
    dB_dt = 10  # 10 T/s
    A_test = 0.001  # 10 cm^2
    N_test = 1000
    emf_dB = N_test * A_test * dB_dt
    scenarios.append(('Changing B\n(dB/dt=10 T/s)', emf_dB))

    # Scenario 3: Moving loop out of field
    v_move = 10  # 10 m/s
    B_test = 1  # 1 T
    L = 0.1  # 10 cm loop width
    emf_move = B_test * L * v_move
    scenarios.append(('Moving loop\n(v=10 m/s)', emf_move))

    # Scenario 4: Transformer
    N1, N2 = 100, 1000
    V1 = 120
    V2 = V1 * N2 / N1
    scenarios.append(('Transformer\n(100:1000)', V2))

    names = [s[0] for s in scenarios]
    values = [s[1] for s in scenarios]

    bars = ax4.bar(names, values, color=['blue', 'green', 'orange', 'red'], alpha=0.7)

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax4.annotate(f'{val:.1f} V',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10)

    ax4.set_ylabel('Induced EMF (V)')
    ax4.set_title('Comparison of Different Induction Scenarios')
    ax4.grid(True, alpha=0.3, axis='y')

    # Add physics summary
    fig.text(0.5, 0.02,
             r"Faraday's Law: $\mathcal{E} = -\frac{d\Phi_B}{dt} = "
             r"-\frac{d}{dt}\int \vec{B} \cdot d\vec{A}$" + '\n'
             r"For N turns: $\mathcal{E} = -N\frac{d\Phi_B}{dt}$",
             ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.suptitle('Faraday\'s Law of Electromagnetic Induction', fontsize=14, y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)

    # Save plot
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'faraday_induction.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
