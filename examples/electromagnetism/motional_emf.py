"""
Experiment 91: Motional EMF.

This example demonstrates motional EMF generated when a conducting rod
moves through a magnetic field, showing the EMF = BLv relationship,
Faraday's law for moving circuits, and the Lorentz force on charges.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt


# Physical constants
Q_E = 1.602e-19  # Elementary charge (C)
M_E = 9.109e-31  # Electron mass (kg)


def motional_emf(B, L, v):
    """
    Calculate motional EMF for a rod moving through a magnetic field.

    EMF = B * L * v (for perpendicular B, L, v)

    Args:
        B: Magnetic field strength (T)
        L: Length of rod (m)
        v: Velocity of rod (m/s)

    Returns:
        EMF: Induced electromotive force (V)
    """
    return B * L * v


def lorentz_force(q, v, B):
    """
    Calculate Lorentz force on a moving charge.

    F = q * v x B

    Args:
        q: Charge (C)
        v: Velocity vector (m/s)
        B: Magnetic field vector (T)

    Returns:
        F: Force vector (N)
    """
    return q * np.cross(v, B)


def hall_voltage(B, I, n, t, q=Q_E):
    """
    Calculate Hall voltage across a conductor.

    V_H = B * I / (n * q * t)

    Args:
        B: Magnetic field (T)
        I: Current (A)
        n: Charge carrier density (m^-3)
        t: Conductor thickness (m)
        q: Charge per carrier (C)

    Returns:
        V_H: Hall voltage (V)
    """
    return B * I / (n * q * t)


class MovingRodSimulation:
    """Simulation of a conducting rod moving on rails in a magnetic field."""

    def __init__(self, B, L, R, m, v0=0.0, F_ext=0.0):
        """
        Args:
            B: Magnetic field strength (T, into page)
            L: Length of rod (m)
            R: Circuit resistance (Ohm)
            m: Mass of rod (kg)
            v0: Initial velocity (m/s)
            F_ext: External force on rod (N)
        """
        self.B = B
        self.L = L
        self.R = R
        self.m = m
        self.F_ext = F_ext

        # State variables
        self.x = 0.0
        self.v = v0
        self.t = 0.0

        # History
        self.history = {'t': [], 'x': [], 'v': [], 'emf': [], 'I': [], 'F_mag': []}

    def emf(self):
        """Current EMF."""
        return self.B * self.L * self.v

    def current(self):
        """Current in circuit."""
        return self.emf() / self.R

    def magnetic_force(self):
        """Magnetic force on rod due to current (Lenz's law)."""
        I = self.current()
        return -I * self.L * self.B  # Opposes motion

    def update(self, dt):
        """Update rod position and velocity."""
        # Total force: external + magnetic braking
        F_total = self.F_ext + self.magnetic_force()

        # Update velocity and position
        a = F_total / self.m
        self.v += a * dt
        self.x += self.v * dt
        self.t += dt

        # Store history
        self.history['t'].append(self.t)
        self.history['x'].append(self.x)
        self.history['v'].append(self.v)
        self.history['emf'].append(self.emf())
        self.history['I'].append(self.current())
        self.history['F_mag'].append(self.magnetic_force())


class RotatingLoopGenerator:
    """AC generator: rotating loop in magnetic field."""

    def __init__(self, B, A, N, omega):
        """
        Args:
            B: Magnetic field strength (T)
            A: Loop area (m^2)
            N: Number of turns
            omega: Angular velocity (rad/s)
        """
        self.B = B
        self.A = A
        self.N = N
        self.omega = omega

    def flux(self, t):
        """Magnetic flux at time t."""
        return self.N * self.B * self.A * np.cos(self.omega * t)

    def emf(self, t):
        """Induced EMF at time t (derivative of flux)."""
        return self.N * self.B * self.A * self.omega * np.sin(self.omega * t)


def main():
    fig = plt.figure(figsize=(16, 12))

    # Plot 1: EMF vs velocity (basic relationship)
    ax1 = fig.add_subplot(2, 2, 1)

    B = 0.5  # 0.5 T magnetic field
    L = 0.1  # 10 cm rod length

    v_range = np.linspace(0, 10, 100)  # 0 to 10 m/s
    emf_values = motional_emf(B, L, v_range)

    ax1.plot(v_range, emf_values * 1000, 'b-', lw=2)

    ax1.set_xlabel('Velocity v (m/s)')
    ax1.set_ylabel('EMF (mV)')
    ax1.set_title(f'Motional EMF: EMF = BLv\nB = {B} T, L = {L*100:.0f} cm')
    ax1.grid(True, alpha=0.3)

    # Add annotation showing formula
    ax1.text(0.05, 0.95, r'$\mathcal{E} = BLv$',
             transform=ax1.transAxes, fontsize=14, va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Mark a specific point
    v_mark = 5
    emf_mark = motional_emf(B, L, v_mark)
    ax1.plot(v_mark, emf_mark * 1000, 'ro', markersize=10)
    ax1.annotate(f'v = {v_mark} m/s\nEMF = {emf_mark*1000:.0f} mV',
                xy=(v_mark, emf_mark * 1000), xytext=(v_mark + 1, emf_mark * 1000 + 100),
                arrowprops=dict(arrowstyle='->', color='red'))

    # Plot 2: Rod sliding on rails - velocity decay (magnetic braking)
    ax2 = fig.add_subplot(2, 2, 2)

    B = 1.0   # 1 T field
    L = 0.2   # 20 cm rod
    R = 0.1   # 0.1 Ohm resistance
    m = 0.05  # 50 g rod
    v0 = 5.0  # 5 m/s initial velocity

    rod = MovingRodSimulation(B, L, R, m, v0=v0)

    dt = 0.001  # 1 ms time step
    for _ in range(3000):
        rod.update(dt)

    t_hist = np.array(rod.history['t'])
    v_hist = np.array(rod.history['v'])
    emf_hist = np.array(rod.history['emf'])
    I_hist = np.array(rod.history['I'])

    # Analytical solution: v(t) = v0 * exp(-t/tau) where tau = m*R/(B*L)^2
    tau = m * R / (B * L)**2
    v_analytical = v0 * np.exp(-t_hist / tau)

    ax2.plot(t_hist * 1000, v_hist, 'b-', lw=2, label='Numerical')
    ax2.plot(t_hist * 1000, v_analytical, 'r--', lw=2, label='Analytical')

    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title(f'Magnetic Braking: Rod on Rails\n'
                  f'B = {B} T, L = {L*100:.0f} cm, R = {R} Ohm, m = {m*1000:.0f} g')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add time constant annotation
    ax2.axhline(y=v0/np.e, color='gray', linestyle=':', alpha=0.5)
    ax2.axvline(x=tau*1000, color='gray', linestyle=':', alpha=0.5)
    ax2.text(tau*1000 + 50, v0/np.e + 0.3, f'tau = {tau*1000:.1f} ms', fontsize=10)

    # Plot 3: Current and power dissipation
    ax3 = fig.add_subplot(2, 2, 3)

    power_dissipated = I_hist**2 * R
    kinetic_energy = 0.5 * m * v_hist**2
    initial_ke = 0.5 * m * v0**2

    ax3_twin = ax3.twinx()

    line1, = ax3.plot(t_hist * 1000, I_hist, 'b-', lw=2, label='Current (A)')
    line2, = ax3.plot(t_hist * 1000, power_dissipated, 'r-', lw=2, label='Power (W)')
    line3, = ax3_twin.plot(t_hist * 1000, kinetic_energy / initial_ke, 'g-', lw=2,
                           label='KE / KE_0')

    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Current (A) / Power (W)')
    ax3_twin.set_ylabel('Kinetic Energy Fraction', color='g')
    ax3.set_title('Energy Dissipation in Circuit')

    ax3.legend(handles=[line1, line2, line3], loc='upper right')
    ax3.grid(True, alpha=0.3)

    # Verify energy conservation
    energy_dissipated = np.trapz(power_dissipated, t_hist)
    ax3.text(0.5, 0.5, f'Initial KE: {initial_ke*1000:.2f} mJ\n'
                       f'Energy dissipated: {energy_dissipated*1000:.2f} mJ',
             transform=ax3.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 4: Lorentz force on charges in moving rod
    ax4 = fig.add_subplot(2, 2, 4)

    # Show charge separation in moving rod
    B = 1.0   # 1 T field (into page, -z direction)
    L = 0.1   # 10 cm rod (along y)
    v = 2.0   # 2 m/s (along x)

    # Electron in rod: v = (v, 0, 0), B = (0, 0, -B)
    v_vec = np.array([v, 0, 0])
    B_vec = np.array([0, 0, -B])

    F_electron = lorentz_force(-Q_E, v_vec, B_vec)
    E_field = v * B  # Electric field due to charge separation

    # Visualize rod with charge distribution
    y_rod = np.linspace(-L/2, L/2, 50)
    # Potential along rod: V(y) = E * y = vB * y
    V_rod = E_field * y_rod

    ax4.plot(y_rod * 100, V_rod * 1000, 'b-', lw=2)
    ax4.set_xlabel('Position along rod y (cm)')
    ax4.set_ylabel('Potential (mV)')

    # Mark ends
    ax4.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax4.plot([-L/2 * 100], [V_rod[0] * 1000], 'b+', markersize=20, mew=3, label='Negative end')
    ax4.plot([L/2 * 100], [V_rod[-1] * 1000], 'ro', markersize=10, label='Positive end')

    # Add voltage annotation
    V_emf = E_field * L
    ax4.annotate(f'EMF = {V_emf*1000:.0f} mV',
                xy=(L/2 * 100, V_rod[-1] * 1000), xytext=(L/2 * 100 - 2, V_rod[-1] * 1000 - 50),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10)

    ax4.set_title(f'Charge Separation in Moving Rod\n'
                  f'v = {v} m/s, B = {B} T, L = {L*100:.0f} cm')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Add Lorentz force annotation
    ax4.text(0.02, 0.95, f'Lorentz force on electron:\n'
                         f'F = qv x B\n'
                         f'F_y = {F_electron[1]:.2e} N',
             transform=ax4.transAxes, va='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Add physics summary
    fig.text(0.5, 0.02,
             r'Motional EMF: $\mathcal{E} = \oint (\vec{v} \times \vec{B}) \cdot d\vec{l} = BLv$'
             + '\n' +
             r'Faraday: $\mathcal{E} = -\frac{d\Phi_B}{dt}$, '
             r'Lorentz: $\vec{F} = q\vec{v} \times \vec{B}$',
             ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.suptitle('Motional EMF and Faraday\'s Law', fontsize=14, y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)

    # Save plot
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'motional_emf.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
