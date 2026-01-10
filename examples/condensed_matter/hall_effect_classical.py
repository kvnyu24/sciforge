"""
Experiment 229: Classical Hall Effect

Demonstrates the classical Hall effect in conductors:
- Hall voltage V_H = I*B / (n*e*t)
- Hall coefficient R_H = 1/(n*e)
- Determine carrier type and density
- Magnetic field dependence
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


# Physical constants
e = 1.602e-19       # Electron charge (C)
m_e = 9.109e-31     # Electron mass (kg)


def hall_voltage(I, B, n, t, carrier_type='electron'):
    """
    Calculate Hall voltage.

    V_H = I * B / (n * e * t)

    Sign depends on carrier type:
    - Electrons: V_H is positive when B points up and I flows right
    - Holes: V_H is negative (opposite sign)

    Args:
        I: Current (A)
        B: Magnetic field (T)
        n: Carrier density (m^-3)
        t: Sample thickness (m)
        carrier_type: 'electron' or 'hole'

    Returns:
        Hall voltage (V)
    """
    V_H = I * B / (n * e * t)
    if carrier_type == 'electron':
        return -V_H  # Electrons deflect opposite to positive charges
    else:
        return V_H


def hall_coefficient(n, carrier_type='electron'):
    """
    Hall coefficient.

    R_H = -1/(n*e) for electrons
    R_H = +1/(n*e) for holes

    Args:
        n: Carrier density (m^-3)
        carrier_type: 'electron' or 'hole'

    Returns:
        Hall coefficient (m^3/C)
    """
    if carrier_type == 'electron':
        return -1 / (n * e)
    else:
        return 1 / (n * e)


def hall_field(j_x, B, n, carrier_type='electron'):
    """
    Hall electric field.

    E_H = j_x * B / (n * e) = R_H * j_x * B

    Args:
        j_x: Current density in x direction (A/m^2)
        B: Magnetic field in z direction (T)
        n: Carrier density (m^-3)
        carrier_type: 'electron' or 'hole'

    Returns:
        Hall field (V/m)
    """
    R_H = hall_coefficient(n, carrier_type)
    return R_H * j_x * B


def hall_angle(mu, B):
    """
    Hall angle - angle between current and electric field.

    tan(theta_H) = mu * B

    Args:
        mu: Carrier mobility (m^2/(V*s))
        B: Magnetic field (T)

    Returns:
        Hall angle (radians)
    """
    return np.arctan(mu * B)


def mobility_from_hall(R_H, sigma):
    """
    Calculate mobility from Hall coefficient and conductivity.

    mu = |R_H| * sigma

    Args:
        R_H: Hall coefficient (m^3/C)
        sigma: Conductivity (S/m)

    Returns:
        Mobility (m^2/(V*s))
    """
    return np.abs(R_H) * sigma


def carrier_density_from_hall(R_H):
    """
    Calculate carrier density from Hall coefficient.

    n = 1 / (e * |R_H|)

    Args:
        R_H: Hall coefficient (m^3/C)

    Returns:
        Carrier density (m^-3)
    """
    return 1 / (e * np.abs(R_H))


def resistivity_tensor(rho_0, mu, B):
    """
    Resistivity tensor in magnetic field.

    rho_xx = rho_0
    rho_xy = -rho_0 * mu * B (for electrons)

    In matrix form:
    [rho_xx  rho_xy]   [1      -mu*B]
    [rho_yx  rho_yy] = [-mu*B    1  ] * rho_0

    Note: rho_yx = -rho_xy (antisymmetric)

    Args:
        rho_0: Zero-field resistivity (Ohm*m)
        mu: Mobility (m^2/(V*s))
        B: Magnetic field (T)

    Returns:
        2x2 resistivity tensor
    """
    rho_xx = rho_0
    rho_xy = rho_0 * mu * B
    return np.array([[rho_xx, rho_xy],
                     [-rho_xy, rho_xx]])


def conductivity_tensor(sigma_0, mu, B):
    """
    Conductivity tensor in magnetic field.

    sigma = rho^(-1)

    Args:
        sigma_0: Zero-field conductivity (S/m)
        mu: Mobility (m^2/(V*s))
        B: Magnetic field (T)

    Returns:
        2x2 conductivity tensor
    """
    omega_c_tau = mu * B  # Dimensionless
    denom = 1 + omega_c_tau**2

    sigma_xx = sigma_0 / denom
    sigma_xy = sigma_0 * omega_c_tau / denom

    return np.array([[sigma_xx, sigma_xy],
                     [-sigma_xy, sigma_xx]])


def simulate_hall_measurement(n, mu, B_range, I, w, t, L):
    """
    Simulate Hall measurement on a bar sample.

    Args:
        n: Carrier density (m^-3)
        mu: Mobility (m^2/(V*s))
        B_range: Array of magnetic field values (T)
        I: Current (A)
        w: Sample width (m)
        t: Sample thickness (m)
        L: Sample length (m)

    Returns:
        V_H: Hall voltage array (V)
        V_x: Longitudinal voltage array (V)
    """
    sigma_0 = n * e * mu

    V_H = np.zeros_like(B_range)
    V_x = np.zeros_like(B_range)

    for i, B in enumerate(B_range):
        # Current density
        j_x = I / (w * t)

        # Electric fields
        E_x = j_x / sigma_0  # Longitudinal (no magnetoresistance in simple Drude)
        E_y = hall_field(j_x, B, n)  # Hall field

        # Voltages
        V_x[i] = E_x * L
        V_H[i] = E_y * w

    return V_H, V_x


def main():
    """Main simulation and visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Sample parameters (typical semiconductor)
    n_Si = 1e22     # n-type silicon carrier density (m^-3)
    mu_Si = 0.14    # Electron mobility (m^2/(V*s))

    # Sample geometry
    L = 10e-3       # Length (10 mm)
    w = 2e-3        # Width (2 mm)
    t = 0.5e-3      # Thickness (0.5 mm)
    I = 1e-3        # Current (1 mA)

    # Plot 1: Hall voltage vs magnetic field
    ax1 = axes[0, 0]

    B_range = np.linspace(-2, 2, 100)

    # Different carrier types
    materials = [
        ('n-Si', n_Si, 'electron', 'blue'),
        ('p-Si', n_Si, 'hole', 'red'),
        ('Metal (Cu)', 8.5e28, 'electron', 'green')
    ]

    for name, n, carrier_type, color in materials:
        V_H = hall_voltage(I, B_range, n, t, carrier_type)
        if name == 'Metal (Cu)':
            V_H *= 1e6  # Scale up for visibility
            label = f'{name} (x10^6)'
        else:
            V_H *= 1e3  # Convert to mV
            label = name
        ax1.plot(B_range, V_H, color=color, lw=2, label=label)

    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

    ax1.set_xlabel('Magnetic Field (T)')
    ax1.set_ylabel('Hall Voltage (mV)')
    ax1.set_title('Hall Voltage vs Magnetic Field')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Annotate carrier type determination
    ax1.annotate('n-type:\nV_H < 0 for B > 0',
                xy=(1, -50), fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax1.annotate('p-type:\nV_H > 0 for B > 0',
                xy=(1, 50), fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightsalmon', alpha=0.8))

    # Plot 2: Hall coefficient vs carrier density
    ax2 = axes[0, 1]

    n_range = np.logspace(20, 29, 100)  # m^-3

    R_H_e = np.abs(hall_coefficient(n_range, 'electron'))

    ax2.loglog(n_range, R_H_e, 'b-', lw=2, label='|R_H| = 1/(n*e)')

    # Mark typical materials
    materials_rh = [
        ('Cu', 8.5e28, 'green'),
        ('Al', 1.8e29, 'orange'),
        ('n-Si (doped)', 1e22, 'blue'),
        ('n-GaAs', 1e23, 'red')
    ]

    for name, n, color in materials_rh:
        R_H = np.abs(hall_coefficient(n))
        ax2.scatter([n], [R_H], s=100, c=color, marker='o', zorder=5)
        ax2.annotate(name, xy=(n, R_H), xytext=(n*2, R_H*2),
                    fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))

    ax2.set_xlabel('Carrier Density (m^-3)')
    ax2.set_ylabel('|Hall Coefficient| (m^3/C)')
    ax2.set_title('Hall Coefficient vs Carrier Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')

    # Plot 3: Hall angle
    ax3 = axes[1, 0]

    B_range_angle = np.linspace(0, 10, 100)

    mobilities = [
        (0.01, 'Low mu (0.01 m^2/Vs)', 'blue'),
        (0.1, 'Medium mu (0.1 m^2/Vs)', 'green'),
        (1.0, 'High mu (1.0 m^2/Vs)', 'red'),
        (10.0, 'Very high mu (10 m^2/Vs)', 'purple')
    ]

    for mu, label, color in mobilities:
        theta_H = hall_angle(mu, B_range_angle) * 180 / np.pi  # Convert to degrees
        ax3.plot(B_range_angle, theta_H, color=color, lw=2, label=label)

    ax3.axhline(y=45, color='gray', linestyle='--', alpha=0.5, label='45 degrees')
    ax3.axhline(y=90, color='gray', linestyle=':', alpha=0.5)

    ax3.set_xlabel('Magnetic Field (T)')
    ax3.set_ylabel('Hall Angle (degrees)')
    ax3.set_title('Hall Angle: tan(theta_H) = mu * B')
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 90)

    # Plot 4: Complete Hall measurement simulation
    ax4 = axes[1, 1]

    B_meas = np.linspace(-1, 1, 50)
    V_H, V_x = simulate_hall_measurement(n_Si, mu_Si, B_meas, I, w, t, L)

    ax4_twin = ax4.twinx()

    line1, = ax4.plot(B_meas, V_H * 1e3, 'b-', lw=2, label='Hall voltage V_H')
    line2, = ax4_twin.plot(B_meas, V_x * 1e3, 'r--', lw=2, label='Longitudinal voltage V_x')

    ax4.set_xlabel('Magnetic Field (T)')
    ax4.set_ylabel('Hall Voltage (mV)', color='blue')
    ax4_twin.set_ylabel('Longitudinal Voltage (mV)', color='red')
    ax4.tick_params(axis='y', labelcolor='blue')
    ax4_twin.tick_params(axis='y', labelcolor='red')

    ax4.set_title('Simulated Hall Measurement')
    ax4.grid(True, alpha=0.3)

    # Add measurement info
    R_H = hall_coefficient(n_Si)
    sigma = n_Si * e * mu_Si
    rho = 1 / sigma

    ax4.text(0.02, 0.98, f'Sample: n-type Si\n'
                         f'n = {n_Si:.1e} m^-3\n'
                         f'mu = {mu_Si*1e4:.0f} cm^2/(V*s)\n'
                         f'R_H = {R_H:.2e} m^3/C\n'
                         f'rho = {rho*100:.2f} Ohm*cm',
             transform=ax4.transAxes, fontsize=10, va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # Combine legends
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='lower right')

    plt.suptitle('Classical Hall Effect\n'
                 r'$V_H = IB/(net)$, $R_H = 1/(ne)$',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'hall_effect_classical.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'hall_effect_classical.png')}")


if __name__ == "__main__":
    main()
