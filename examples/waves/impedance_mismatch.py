"""
Example demonstrating impedance mismatch reflection and transmission.

This example shows how waves reflect and transmit at boundaries between
media with different acoustic/mechanical impedances. The reflection and
transmission coefficients depend on the impedance ratio.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.waves import Wave


def gaussian_pulse(x, x0, sigma, amplitude=1.0):
    """Gaussian pulse centered at x0."""
    return amplitude * np.exp(-(x - x0)**2 / (2 * sigma**2))


def calculate_reflection_transmission(Z1, Z2):
    """
    Calculate reflection and transmission coefficients for wave at boundary.

    For a wave going from medium 1 to medium 2:
    - Reflection coefficient: r = (Z2 - Z1) / (Z2 + Z1)
    - Transmission coefficient: t = 2*Z2 / (Z2 + Z1)

    Args:
        Z1: Impedance of medium 1 (incident side)
        Z2: Impedance of medium 2 (transmitted side)

    Returns:
        r, t: Reflection and transmission coefficients
    """
    r = (Z2 - Z1) / (Z2 + Z1)
    t = 2 * Z2 / (Z2 + Z1)
    return r, t


def simulate_wave_at_boundary(x, t, x0_pulse, sigma, c1, c2, Z1, Z2, x_boundary):
    """
    Simulate wave pulse encountering an impedance boundary.

    Args:
        x: Spatial grid
        t: Current time
        x0_pulse: Initial pulse position
        sigma: Pulse width
        c1, c2: Wave speeds in media 1 and 2
        Z1, Z2: Impedances of media 1 and 2
        x_boundary: Position of boundary

    Returns:
        Wave displacement at time t
    """
    r, t_coef = calculate_reflection_transmission(Z1, Z2)

    # Initialize displacement
    u = np.zeros_like(x)

    # Incident wave (traveling right in medium 1)
    x_incident = x0_pulse + c1 * t
    incident = gaussian_pulse(x, x_incident, sigma)

    # Has the pulse reached the boundary?
    if x_incident >= x_boundary:
        # Time since hitting boundary
        t_after = (x_incident - x_boundary) / c1

        # Reflected wave (traveling left in medium 1)
        x_reflected = x_boundary - c1 * t_after
        reflected = r * gaussian_pulse(x, x_reflected, sigma)

        # Transmitted wave (traveling right in medium 2)
        x_transmitted = x_boundary + c2 * t_after
        transmitted = t_coef * gaussian_pulse(x, x_transmitted, sigma * c2/c1)

        # Add contributions in appropriate regions
        u[x < x_boundary] += reflected[x < x_boundary]
        u[x >= x_boundary] += transmitted[x >= x_boundary]

        # Add remaining incident wave if it hasn't fully passed
        u[x < x_boundary] += incident[x < x_boundary] * (x_incident < x_boundary + 3*sigma)
    else:
        # Pulse hasn't reached boundary yet
        u += incident

    return u


def main():
    # Spatial domain
    x = np.linspace(-10, 20, 1000)
    x_boundary = 0.0  # Boundary at x = 0

    fig = plt.figure(figsize=(16, 14))

    # =========================================================================
    # Panel 1: Impedance much higher in medium 2 (hard reflection)
    # =========================================================================
    ax1 = fig.add_subplot(3, 3, 1)

    Z1, Z2 = 1.0, 10.0  # Z2 >> Z1
    c1, c2 = 1.0, 0.5   # Different wave speeds
    r, t_coef = calculate_reflection_transmission(Z1, Z2)

    times = [0, 3, 6, 9, 12]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(times)))
    sigma = 1.0
    x0 = -5.0

    for t_val, color in zip(times, colors):
        u = simulate_wave_at_boundary(x, t_val, x0, sigma, c1, c2, Z1, Z2, x_boundary)
        ax1.plot(x, u + t_val * 0.3, color=color, lw=2, label=f't = {t_val}')

    ax1.axvline(x=x_boundary, color='black', lw=3)
    ax1.fill_betweenx([-2, 8], x_boundary, 20, color='gray', alpha=0.2)

    ax1.set_xlabel('Position x')
    ax1.set_ylabel('Displacement (offset)')
    ax1.set_title(f'High Impedance (Z2/Z1 = {Z2/Z1})\nr = {r:.2f}, t = {t_coef:.2f}')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-10, 15)

    # =========================================================================
    # Panel 2: Impedance much lower in medium 2 (soft reflection)
    # =========================================================================
    ax2 = fig.add_subplot(3, 3, 2)

    Z1, Z2 = 10.0, 1.0  # Z2 << Z1
    r, t_coef = calculate_reflection_transmission(Z1, Z2)

    for t_val, color in zip(times, colors):
        u = simulate_wave_at_boundary(x, t_val, x0, sigma, c1, c2, Z1, Z2, x_boundary)
        ax2.plot(x, u + t_val * 0.3, color=color, lw=2, label=f't = {t_val}')

    ax2.axvline(x=x_boundary, color='black', lw=3)
    ax2.fill_betweenx([-2, 8], x_boundary, 20, color='lightyellow', alpha=0.5)

    ax2.set_xlabel('Position x')
    ax2.set_ylabel('Displacement (offset)')
    ax2.set_title(f'Low Impedance (Z2/Z1 = {Z2/Z1})\nr = {r:.2f}, t = {t_coef:.2f}')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-10, 15)

    # =========================================================================
    # Panel 3: Matched impedance (no reflection)
    # =========================================================================
    ax3 = fig.add_subplot(3, 3, 3)

    Z1, Z2 = 1.0, 1.0  # Matched
    r, t_coef = calculate_reflection_transmission(Z1, Z2)

    for t_val, color in zip(times, colors):
        u = simulate_wave_at_boundary(x, t_val, x0, sigma, c1, c2, Z1, Z2, x_boundary)
        ax3.plot(x, u + t_val * 0.3, color=color, lw=2, label=f't = {t_val}')

    ax3.axvline(x=x_boundary, color='green', lw=3, linestyle='--')

    ax3.set_xlabel('Position x')
    ax3.set_ylabel('Displacement (offset)')
    ax3.set_title(f'Matched Impedance (Z2/Z1 = 1)\nr = {r:.2f}, t = {t_coef:.2f} (Perfect transmission)')
    ax3.legend(fontsize=8, loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-10, 15)

    # =========================================================================
    # Panel 4: Reflection and transmission vs impedance ratio
    # =========================================================================
    ax4 = fig.add_subplot(3, 3, 4)

    Z_ratio = np.logspace(-2, 2, 100)  # Z2/Z1 from 0.01 to 100

    r_values = (Z_ratio - 1) / (Z_ratio + 1)
    t_values = 2 * Z_ratio / (Z_ratio + 1)

    # Power/Energy coefficients (R and T where R + T = 1)
    R = r_values**2
    T = 1 - R  # Energy conservation

    ax4.semilogx(Z_ratio, r_values, 'b-', lw=2, label='r (amplitude)')
    ax4.semilogx(Z_ratio, t_values, 'r--', lw=2, label='t (amplitude)')
    ax4.semilogx(Z_ratio, R, 'b:', lw=2, label='R = r^2 (power)')
    ax4.semilogx(Z_ratio, T, 'r:', lw=2, label='T = 1-R (power)')

    ax4.axhline(y=0, color='gray', lw=0.5)
    ax4.axvline(x=1, color='green', linestyle='--', alpha=0.5, label='Matched (Z2=Z1)')

    ax4.set_xlabel('Impedance Ratio Z2/Z1')
    ax4.set_ylabel('Coefficient')
    ax4.set_title('Reflection & Transmission Coefficients\nvs Impedance Ratio')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-1.5, 2.5)

    # =========================================================================
    # Panel 5: Physical examples of impedance
    # =========================================================================
    ax5 = fig.add_subplot(3, 3, 5)

    # Acoustic impedance Z = rho * c
    materials = {
        'Air': 413,
        'Water': 1.48e6,
        'Soft Tissue': 1.63e6,
        'Bone': 7.8e6,
        'Steel': 45.7e6,
        'Aluminum': 17e6,
    }

    mat_names = list(materials.keys())
    mat_Z = list(materials.values())

    y_pos = np.arange(len(mat_names))
    bars = ax5.barh(y_pos, np.log10(mat_Z), color='steelblue', alpha=0.7)

    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(mat_names)
    ax5.set_xlabel('log10(Impedance) [kg/(m^2 s)]')
    ax5.set_title('Acoustic Impedances of Materials\nZ = density x speed of sound')
    ax5.grid(True, alpha=0.3, axis='x')

    # Add values as text
    for i, (name, Z) in enumerate(materials.items()):
        ax5.text(np.log10(Z) + 0.1, i, f'{Z:.2e}', va='center', fontsize=8)

    # =========================================================================
    # Panel 6: String with mass loading
    # =========================================================================
    ax6 = fig.add_subplot(3, 3, 6)

    # Mass loading creates impedance discontinuity
    x_string = np.linspace(-10, 10, 500)
    mass_pos = 0.0

    # Show multiple reflections for wave on string with bead
    Z_string = 1.0
    Z_bead = 3.0  # Heavy bead
    r_bead, t_bead = calculate_reflection_transmission(Z_string, Z_bead)

    times_string = [0, 2, 4, 6, 8]
    colors_str = plt.cm.plasma(np.linspace(0.1, 0.9, len(times_string)))

    for t_val, color in zip(times_string, colors_str):
        # Simplified: just show incident and first reflections
        incident = gaussian_pulse(x_string, -5 + t_val, 0.8)
        reflected = r_bead * gaussian_pulse(x_string, -t_val, 0.8) * (t_val > 5)
        transmitted = t_bead * gaussian_pulse(x_string, t_val - 5, 0.8) * (t_val > 5)

        total = incident + reflected + transmitted
        ax6.plot(x_string, total + t_val * 0.4, color=color, lw=2, label=f't = {t_val}')

    # Mark bead position
    ax6.plot(mass_pos, 0, 'ko', markersize=15, label='Bead (mass load)')

    ax6.set_xlabel('Position on String')
    ax6.set_ylabel('Displacement (offset)')
    ax6.set_title('Wave on String with Mass Loading\n(Bead creates partial reflection)')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(-10, 10)

    # =========================================================================
    # Panel 7: Multiple layer reflections (thin film)
    # =========================================================================
    ax7 = fig.add_subplot(3, 3, 7)

    # Simulate wave in three-layer system
    Z1, Z2, Z3 = 1.0, 2.0, 1.0  # Medium-Film-Medium
    d_film = 1.0  # Film thickness

    # Boundaries at x=0 and x=d_film
    x_layers = np.linspace(-5, 10, 500)

    times_layer = [0, 1.5, 3, 4.5, 6]
    colors_layer = plt.cm.viridis(np.linspace(0.1, 0.9, len(times_layer)))

    r12, t12 = calculate_reflection_transmission(Z1, Z2)
    r21, t21 = calculate_reflection_transmission(Z2, Z1)
    r23, t23 = calculate_reflection_transmission(Z2, Z3)

    for i, (t_val, color) in enumerate(zip(times_layer, colors_layer)):
        # Simplified visualization
        u = np.zeros_like(x_layers)

        # Primary incident
        u += gaussian_pulse(x_layers, -3 + t_val, 0.5)

        # Primary reflection from first boundary
        if t_val > 3:
            u += r12 * gaussian_pulse(x_layers, -(t_val - 3), 0.5)

        # Transmission into layer
        if t_val > 3:
            u += t12 * gaussian_pulse(x_layers, (t_val - 3) * 0.7, 0.35)

        ax7.plot(x_layers, u + i * 0.5, color=color, lw=2, label=f't = {t_val}')

    # Mark layer
    ax7.axvline(x=0, color='black', lw=2)
    ax7.axvline(x=d_film, color='black', lw=2)
    ax7.fill_betweenx([-1, 5], 0, d_film, color='lightblue', alpha=0.5, label='Thin film')

    ax7.set_xlabel('Position')
    ax7.set_ylabel('Displacement (offset)')
    ax7.set_title('Three-Layer System (Thin Film)\nMultiple reflections create interference')
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)
    ax7.set_xlim(-5, 8)

    # =========================================================================
    # Panel 8: Quarter-wave matching layer
    # =========================================================================
    ax8 = fig.add_subplot(3, 3, 8)

    # Anti-reflection coating: Z_film = sqrt(Z1 * Z3)
    Z1, Z3 = 1.0, 4.0  # Air to glass
    Z_optimal = np.sqrt(Z1 * Z3)  # Optimal matching layer

    Z_film_range = np.linspace(0.5, 4, 100)

    # Simplified model: net reflection for quarter-wave layer
    # R ~ ((Z1*Z3 - Z_film^2) / (Z1*Z3 + Z_film^2))^2
    R_net = ((Z1 * Z3 - Z_film_range**2) / (Z1 * Z3 + Z_film_range**2))**2

    ax8.plot(Z_film_range, R_net, 'b-', lw=2)
    ax8.axvline(x=Z_optimal, color='green', linestyle='--', lw=2,
                label=f'Optimal: Z_film = sqrt(Z1*Z3) = {Z_optimal:.2f}')
    ax8.axhline(y=0, color='gray', lw=0.5)

    # Without matching layer
    r_direct = (Z3 - Z1) / (Z3 + Z1)
    ax8.axhline(y=r_direct**2, color='red', linestyle=':',
                label=f'No layer: R = {r_direct**2:.2f}')

    ax8.set_xlabel('Matching Layer Impedance')
    ax8.set_ylabel('Net Reflectance R')
    ax8.set_title('Quarter-Wave Matching Layer\n(Minimum reflection at optimal Z)')
    ax8.legend(fontsize=8)
    ax8.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 9: Energy conservation at boundary
    # =========================================================================
    ax9 = fig.add_subplot(3, 3, 9)

    Z_ratio_check = np.logspace(-1, 1, 50)

    r_check = (Z_ratio_check - 1) / (Z_ratio_check + 1)
    t_check = 2 * Z_ratio_check / (Z_ratio_check + 1)

    # Energy flux: proportional to Z * |amplitude|^2
    # For continuity: incident = reflected + transmitted
    # Power: P_incident = Z1 * A_i^2, P_reflected = Z1 * |r|^2 * A_i^2
    # P_transmitted = Z2 * |t * (c1/c2)|^2 * A_i^2 (accounting for wavelength change)

    # Using proper energy coefficients
    R_power = r_check**2
    T_power = (4 * Z_ratio_check) / (Z_ratio_check + 1)**2  # Energy transmission

    sum_RT = R_power + T_power

    ax9.semilogx(Z_ratio_check, R_power, 'b-', lw=2, label='R (reflected power)')
    ax9.semilogx(Z_ratio_check, T_power, 'r--', lw=2, label='T (transmitted power)')
    ax9.semilogx(Z_ratio_check, sum_RT, 'k:', lw=3, label='R + T = 1')

    ax9.set_xlabel('Impedance Ratio Z2/Z1')
    ax9.set_ylabel('Power Fraction')
    ax9.set_title('Energy Conservation: R + T = 1\n(Power reflected + transmitted = incident)')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    ax9.set_ylim(0, 1.2)

    plt.suptitle('Impedance Mismatch: Wave Reflection and Transmission\n'
                 'r = (Z2-Z1)/(Z2+Z1), t = 2Z2/(Z2+Z1)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'impedance_mismatch.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'impedance_mismatch.png')}")


if __name__ == "__main__":
    main()
