"""
Experiment 99: Dielectric slab in capacitor.

This example demonstrates the effect of inserting a dielectric
slab into a parallel plate capacitor, showing how it affects
the capacitance, field distribution, and stored energy.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt


# Physical constants
EPSILON_0 = 8.854e-12  # Permittivity of free space (F/m)


def capacitor_with_dielectric(d, A, d_dielectric, epsilon_r, x_dielectric=0):
    """
    Calculate properties of a capacitor with a partial dielectric slab.

    Args:
        d: Total plate separation (m)
        A: Plate area (m^2)
        d_dielectric: Thickness of dielectric slab (m)
        epsilon_r: Relative permittivity of dielectric
        x_dielectric: Position of dielectric (from bottom plate, m)

    Returns:
        C: Capacitance (F)
        E_vacuum: Field in vacuum region (V/m) for V=1V
        E_dielectric: Field in dielectric (V/m) for V=1V
    """
    # Capacitor with partial dielectric is like two capacitors in series:
    # Air gap 1 (below dielectric) + Dielectric + Air gap 2 (above dielectric)

    d_air = d - d_dielectric

    # Equivalent capacitance
    # 1/C = 1/C_air + 1/C_dielectric = d_air/(eps_0*A) + d_dielectric/(eps_0*eps_r*A)
    C = EPSILON_0 * A / (d_air + d_dielectric / epsilon_r)

    # For unit voltage:
    # V = E_air * d_air + E_dielectric * d_dielectric
    # D is continuous: eps_0 * E_air = eps_0 * eps_r * E_dielectric
    # Therefore: E_dielectric = E_air / eps_r

    # V = E_air * d_air + (E_air/eps_r) * d_dielectric
    # V = E_air * (d_air + d_dielectric/eps_r)
    E_vacuum = 1 / (d_air + d_dielectric / epsilon_r)  # For V = 1V
    E_dielectric = E_vacuum / epsilon_r

    return C, E_vacuum, E_dielectric


def field_profile(x, d, d_dielectric, x_start, epsilon_r, V):
    """
    Calculate electric field as a function of position.

    Args:
        x: Position array (m)
        d: Plate separation (m)
        d_dielectric: Dielectric thickness (m)
        x_start: Start position of dielectric (m)
        epsilon_r: Relative permittivity
        V: Applied voltage (V)

    Returns:
        E: Electric field at each position (V/m)
    """
    _, E_vacuum, E_dielectric = capacitor_with_dielectric(
        d, 1.0, d_dielectric, epsilon_r, x_start
    )

    E = np.zeros_like(x)

    for i, xi in enumerate(x):
        if x_start <= xi <= x_start + d_dielectric:
            E[i] = E_dielectric * V
        else:
            E[i] = E_vacuum * V

    return E


def potential_profile(x, d, d_dielectric, x_start, epsilon_r, V):
    """
    Calculate electric potential as a function of position.
    """
    _, E_vacuum, E_dielectric = capacitor_with_dielectric(
        d, 1.0, d_dielectric, epsilon_r, x_start
    )

    E_vacuum *= V
    E_dielectric *= V

    phi = np.zeros_like(x)

    for i, xi in enumerate(x):
        # Integrate E from bottom plate
        if xi <= x_start:
            phi[i] = V - E_vacuum * xi
        elif xi <= x_start + d_dielectric:
            phi[i] = V - E_vacuum * x_start - E_dielectric * (xi - x_start)
        else:
            phi_at_dielectric_end = (V - E_vacuum * x_start -
                                     E_dielectric * d_dielectric)
            phi[i] = phi_at_dielectric_end - E_vacuum * (xi - x_start - d_dielectric)

    return phi


def main():
    # Capacitor parameters
    d = 0.01       # 1 cm plate separation
    A = 0.01       # 100 cm^2 plate area
    V0 = 100       # 100 V applied voltage

    # Empty capacitor capacitance
    C0 = EPSILON_0 * A / d

    fig = plt.figure(figsize=(16, 12))

    # Plot 1: Capacitance vs dielectric thickness
    ax1 = fig.add_subplot(2, 2, 1)

    epsilon_r_values = [2, 4, 8, 80]  # Various dielectrics
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(epsilon_r_values)))

    d_dielectric_range = np.linspace(0, d * 0.99, 100)

    for eps_r, color in zip(epsilon_r_values, colors):
        C_values = []
        for d_d in d_dielectric_range:
            C, _, _ = capacitor_with_dielectric(d, A, d_d, eps_r)
            C_values.append(C)

        ax1.plot(d_dielectric_range / d * 100, np.array(C_values) / C0, color=color,
                 lw=2, label=f'eps_r = {eps_r}')

    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Empty capacitor')

    ax1.set_xlabel('Dielectric filling (% of gap)')
    ax1.set_ylabel('C / C_0 (normalized capacitance)')
    ax1.set_title('Capacitance vs Dielectric Thickness')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Field and potential profile
    ax2 = fig.add_subplot(2, 2, 2)

    x = np.linspace(0, d, 500)
    epsilon_r = 4  # Glass-like dielectric
    d_dielectric = 0.5 * d  # Fill half the gap
    x_start = 0.25 * d  # Centered dielectric

    E = field_profile(x, d, d_dielectric, x_start, epsilon_r, V0)
    phi = potential_profile(x, d, d_dielectric, x_start, epsilon_r, V0)

    ax2.plot(x * 1000, E / 1000, 'b-', lw=2, label='E field (kV/m)')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(x * 1000, phi, 'r-', lw=2, label='Potential (V)')

    # Mark dielectric region
    ax2.axvspan(x_start * 1000, (x_start + d_dielectric) * 1000,
                alpha=0.2, color='yellow', label='Dielectric')

    ax2.set_xlabel('Position (mm)')
    ax2.set_ylabel('Electric field (kV/m)', color='b')
    ax2_twin.set_ylabel('Potential (V)', color='r')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2_twin.tick_params(axis='y', labelcolor='r')

    ax2.set_title(f'Field and Potential Profile (eps_r = {epsilon_r})')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Energy storage
    ax3 = fig.add_subplot(2, 2, 3)

    # Compare energy with and without dielectric for:
    # 1. Constant voltage (capacitor connected to battery)
    # 2. Constant charge (isolated capacitor)

    fill_fractions = np.linspace(0, 0.99, 100)
    epsilon_r = 4

    U0 = 0.5 * C0 * V0**2  # Energy without dielectric

    U_const_V = []  # Energy at constant voltage
    U_const_Q = []  # Energy at constant charge

    Q0 = C0 * V0  # Initial charge

    for fill in fill_fractions:
        d_d = fill * d
        C, _, _ = capacitor_with_dielectric(d, A, d_d, epsilon_r)

        # Constant voltage: U = (1/2)*C*V^2, C increases, U increases
        U_const_V.append(0.5 * C * V0**2)

        # Constant charge: U = Q^2/(2C), C increases, U decreases
        U_const_Q.append(Q0**2 / (2 * C))

    ax3.plot(fill_fractions * 100, np.array(U_const_V) / U0, 'b-', lw=2,
             label='Constant voltage (battery connected)')
    ax3.plot(fill_fractions * 100, np.array(U_const_Q) / U0, 'r-', lw=2,
             label='Constant charge (isolated)')
    ax3.axhline(y=1, color='gray', linestyle='--', alpha=0.5)

    ax3.set_xlabel('Dielectric filling (%)')
    ax3.set_ylabel('U / U_0 (normalized energy)')
    ax3.set_title(f'Energy Storage (eps_r = {epsilon_r})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Annotate physical meaning
    ax3.text(0.5, 0.15, 'At constant V: More charge stored\n'
                        'At constant Q: Less voltage needed',
             transform=ax3.transAxes, fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 4: Force on dielectric
    ax4 = fig.add_subplot(2, 2, 4)

    # Force on dielectric being inserted (at constant voltage)
    # F = dU/dx = (1/2)*V^2*dC/dx

    # For a slab being inserted, C(x) where x is insertion depth
    width = 0.1  # 10 cm width of plates
    height = 0.1  # 10 cm height

    # Dielectric insertion from the side
    x_insert = np.linspace(0, width * 0.99, 100)
    epsilon_r = 4

    C_insertion = []
    for x_in in x_insert:
        # Part with dielectric: width = x_in
        C_dielectric = EPSILON_0 * epsilon_r * x_in * height / d
        # Part without: width = (width - x_in)
        C_air = EPSILON_0 * (width - x_in) * height / d
        C_total = C_dielectric + C_air
        C_insertion.append(C_total)

    C_insertion = np.array(C_insertion)

    # Force
    dC_dx = np.gradient(C_insertion, x_insert)
    Force = 0.5 * V0**2 * dC_dx

    ax4.plot(x_insert / width * 100, C_insertion / C_insertion[0], 'b-', lw=2,
             label='Capacitance')
    ax4_twin = ax4.twinx()
    ax4_twin.plot(x_insert / width * 100, Force * 1000, 'r-', lw=2, label='Force (mN)')

    ax4.set_xlabel('Dielectric insertion (%)')
    ax4.set_ylabel('C / C_0', color='b')
    ax4_twin.set_ylabel('Force (mN)', color='r')
    ax4.tick_params(axis='y', labelcolor='b')
    ax4_twin.tick_params(axis='y', labelcolor='r')

    ax4.set_title('Force on Dielectric Slab Being Inserted\n(constant voltage)')
    ax4.grid(True, alpha=0.3)

    # Note positive force = dielectric pulled in
    ax4_twin.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax4.text(50, 1.5, 'Dielectric is PULLED\ninto capacitor', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # Add physics summary
    fig.text(0.5, 0.02,
             r'Dielectric in capacitor: $C = \epsilon_0 \epsilon_r A / d$, '
             r'$E_{dielectric} = E_{vacuum}/\epsilon_r$' + '\n'
             r'Energy: $U = \frac{1}{2}CV^2$ (const V) or $U = \frac{Q^2}{2C}$ (const Q)',
             ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.suptitle(f'Dielectric Slab in Parallel Plate Capacitor\n'
                 f'A = {A*1e4:.0f} cm^2, d = {d*1000:.0f} mm, V = {V0} V',
                 fontsize=14, y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)

    # Save plot
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'dielectric_capacitor.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
