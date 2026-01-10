"""
Example demonstrating wave packet dispersion.

This example shows how wave packets spread over time in a dispersive medium,
where the phase velocity depends on wavelength/frequency. This leads to
group velocity differing from phase velocity and eventual spreading.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.waves import WavePacket


def gaussian_envelope(x, x0, sigma):
    """Gaussian envelope centered at x0."""
    return np.exp(-(x - x0)**2 / (2 * sigma**2))


def wave_packet_nondispersive(x, t, x0, sigma0, k0, omega0):
    """
    Wave packet in non-dispersive medium.
    Envelope moves with phase velocity, no spreading.
    """
    c = omega0 / k0  # Phase velocity = group velocity
    envelope = gaussian_envelope(x, x0 + c * t, sigma0)
    carrier = np.cos(k0 * x - omega0 * t)
    return envelope * carrier


def wave_packet_dispersive(x, t, x0, sigma0, k0, omega0, dispersion):
    """
    Wave packet in dispersive medium with quadratic dispersion.

    omega(k) = omega0 + (k - k0) * v_g + 0.5 * dispersion * (k - k0)^2

    The group velocity is v_g = d(omega)/dk at k0
    The dispersion parameter affects spreading.
    """
    # Group velocity
    v_g = omega0 / k0  # Assuming linear term gives v_g = omega0/k0

    # For a Gaussian wave packet with dispersion, width evolves as:
    # sigma(t) = sigma0 * sqrt(1 + (dispersion * t / sigma0^2)^2)
    sigma_t = sigma0 * np.sqrt(1 + (dispersion * t / (2 * sigma0**2))**2)

    # Phase evolution
    phase_factor = np.arctan(dispersion * t / (2 * sigma0**2))

    # Envelope (spreads over time)
    envelope = (sigma0 / sigma_t) * gaussian_envelope(x, x0 + v_g * t, sigma_t)

    # Carrier wave (phase velocity differs from group velocity in dispersive medium)
    # For visualization, we'll use a simplified phase
    carrier = np.cos(k0 * (x - x0) - omega0 * t + phase_factor)

    return envelope * carrier


def main():
    # Spatial domain
    x = np.linspace(-50, 100, 2000)

    # Wave packet parameters
    x0 = 0.0       # Initial center position
    sigma0 = 5.0   # Initial width
    k0 = 1.0       # Central wavenumber
    omega0 = 1.0   # Central frequency
    dispersion = 0.5  # Dispersion parameter

    fig = plt.figure(figsize=(16, 14))

    # =========================================================================
    # Panel 1: Non-dispersive wave packet evolution
    # =========================================================================
    ax1 = fig.add_subplot(3, 3, 1)

    times = [0, 10, 20, 30, 40]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(times)))

    for t, color in zip(times, colors):
        wave = wave_packet_nondispersive(x, t, x0, sigma0, k0, omega0)
        ax1.plot(x, wave + 0.5 * t, color=color, lw=1.5, label=f't = {t}')

    ax1.set_xlabel('Position x')
    ax1.set_ylabel('Displacement (offset)')
    ax1.set_title('Non-Dispersive Medium\n(No spreading, shape preserved)')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-20, 80)

    # =========================================================================
    # Panel 2: Dispersive wave packet evolution
    # =========================================================================
    ax2 = fig.add_subplot(3, 3, 2)

    for t, color in zip(times, colors):
        wave = wave_packet_dispersive(x, t, x0, sigma0, k0, omega0, dispersion)
        ax2.plot(x, wave + 0.5 * t, color=color, lw=1.5, label=f't = {t}')

    ax2.set_xlabel('Position x')
    ax2.set_ylabel('Displacement (offset)')
    ax2.set_title('Dispersive Medium\n(Packet spreads over time)')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-20, 80)

    # =========================================================================
    # Panel 3: Width evolution comparison
    # =========================================================================
    ax3 = fig.add_subplot(3, 3, 3)

    t_array = np.linspace(0, 50, 100)

    # Width evolution for different dispersion strengths
    dispersions = [0, 0.25, 0.5, 1.0, 2.0]
    colors_disp = plt.cm.plasma(np.linspace(0.1, 0.9, len(dispersions)))

    for d, color in zip(dispersions, colors_disp):
        sigma_t = sigma0 * np.sqrt(1 + (d * t_array / (2 * sigma0**2))**2)
        ax3.plot(t_array, sigma_t, color=color, lw=2, label=f'D = {d}')

    ax3.set_xlabel('Time t')
    ax3.set_ylabel('Packet Width sigma(t)')
    ax3.set_title('Width Evolution\nsigma(t) = sigma_0 * sqrt(1 + (Dt/2sigma_0^2)^2)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 4: Space-time diagram - Non-dispersive
    # =========================================================================
    ax4 = fig.add_subplot(3, 3, 4)

    t_fine = np.linspace(0, 50, 200)
    X, T = np.meshgrid(x, t_fine)

    U_nondispersive = np.zeros_like(X)
    for i, t in enumerate(t_fine):
        U_nondispersive[i, :] = wave_packet_nondispersive(x, t, x0, sigma0, k0, omega0)

    im4 = ax4.imshow(U_nondispersive, aspect='auto',
                     extent=[x.min(), x.max(), t_fine.max(), t_fine.min()],
                     cmap='RdBu', vmin=-1, vmax=1)
    ax4.set_xlabel('Position x')
    ax4.set_ylabel('Time t')
    ax4.set_title('Space-Time: Non-Dispersive\n(Straight trajectory)')
    plt.colorbar(im4, ax=ax4, label='Amplitude')

    # Draw velocity line
    v_phase = omega0 / k0
    ax4.plot([x0, x0 + v_phase * t_fine.max()], [0, t_fine.max()],
             'k--', lw=2, alpha=0.5)

    # =========================================================================
    # Panel 5: Space-time diagram - Dispersive
    # =========================================================================
    ax5 = fig.add_subplot(3, 3, 5)

    U_dispersive = np.zeros_like(X)
    for i, t in enumerate(t_fine):
        U_dispersive[i, :] = wave_packet_dispersive(x, t, x0, sigma0, k0, omega0,
                                                    dispersion)

    im5 = ax5.imshow(U_dispersive, aspect='auto',
                     extent=[x.min(), x.max(), t_fine.max(), t_fine.min()],
                     cmap='RdBu', vmin=-1, vmax=1)
    ax5.set_xlabel('Position x')
    ax5.set_ylabel('Time t')
    ax5.set_title('Space-Time: Dispersive\n(Spreading evident)')
    plt.colorbar(im5, ax=ax5, label='Amplitude')

    # =========================================================================
    # Panel 6: Dispersion relation examples
    # =========================================================================
    ax6 = fig.add_subplot(3, 3, 6)

    k = np.linspace(0.1, 3, 100)

    # Different dispersion relations
    # 1. Non-dispersive (light in vacuum): omega = c*k
    omega_linear = k

    # 2. Deep water waves: omega = sqrt(g*k)
    g = 9.81
    omega_deep = np.sqrt(g * k)

    # 3. Shallow water waves: omega = c*k (also non-dispersive)
    c_shallow = 3.0
    omega_shallow = c_shallow * k

    # 4. Quantum particle: omega = hbar*k^2 / (2m)
    omega_quantum = 0.5 * k**2

    # 5. Electromagnetic waves in plasma: omega = sqrt(omega_p^2 + c^2*k^2)
    omega_p = 1.0
    omega_plasma = np.sqrt(omega_p**2 + k**2)

    ax6.plot(k, omega_linear, 'b-', lw=2, label='Linear (non-dispersive)')
    ax6.plot(k, omega_deep, 'r-', lw=2, label='Deep water: sqrt(gk)')
    ax6.plot(k, omega_quantum, 'g-', lw=2, label='Quantum: k^2')
    ax6.plot(k, omega_plasma, 'm-', lw=2, label='Plasma: sqrt(w_p^2 + k^2)')

    ax6.set_xlabel('Wavenumber k')
    ax6.set_ylabel('Angular frequency omega')
    ax6.set_title('Dispersion Relations\nomega(k)')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 7: Group velocity vs phase velocity
    # =========================================================================
    ax7 = fig.add_subplot(3, 3, 7)

    k = np.linspace(0.5, 3, 100)

    # Phase velocity v_p = omega/k
    # Group velocity v_g = d(omega)/dk

    # Deep water waves
    v_p_deep = np.sqrt(g / k)
    v_g_deep = 0.5 * np.sqrt(g / k)

    # Plasma waves
    v_p_plasma = np.sqrt(omega_p**2 / k**2 + 1)
    v_g_plasma = k / np.sqrt(omega_p**2 + k**2)

    ax7.plot(k, v_p_deep, 'r-', lw=2, label='Deep water: v_phase')
    ax7.plot(k, v_g_deep, 'r--', lw=2, label='Deep water: v_group')
    ax7.plot(k, v_p_plasma, 'm-', lw=2, label='Plasma: v_phase')
    ax7.plot(k, v_g_plasma, 'm--', lw=2, label='Plasma: v_group')

    ax7.axhline(y=1, color='b', linestyle=':', label='v_phase = v_group (non-disp)')

    ax7.set_xlabel('Wavenumber k')
    ax7.set_ylabel('Velocity')
    ax7.set_title('Phase vs Group Velocity\nv_g = d(omega)/dk, v_p = omega/k')
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)
    ax7.set_ylim(0, 5)

    # =========================================================================
    # Panel 8: Fourier decomposition showing dispersion
    # =========================================================================
    ax8 = fig.add_subplot(3, 3, 8)

    # Show individual frequency components and how they evolve
    x_detail = np.linspace(-30, 60, 500)
    t_show = 30

    # Initial wave packet in k-space is Gaussian centered at k0
    k_values = [0.7, 0.85, 1.0, 1.15, 1.3]
    colors_k = plt.cm.coolwarm(np.linspace(0, 1, len(k_values)))

    total = np.zeros_like(x_detail)

    for k_val, color in zip(k_values, colors_k):
        # In dispersive medium, omega depends on k
        omega_val = omega0 + (k_val - k0) + dispersion * (k_val - k0)**2
        v_p = omega_val / k_val

        # Component amplitude from Gaussian spectrum
        amp = np.exp(-(k_val - k0)**2 * sigma0**2 / 2)

        component = amp * np.cos(k_val * x_detail - omega_val * t_show)
        ax8.plot(x_detail, component, color=color, lw=1, alpha=0.7,
                 label=f'k = {k_val:.2f}')
        total += component

    ax8.plot(x_detail, total / len(k_values), 'k-', lw=2, label='Sum')
    ax8.set_xlabel('Position x')
    ax8.set_ylabel('Amplitude')
    ax8.set_title(f'Fourier Components at t = {t_show}\n(Different k travel at different speeds)')
    ax8.legend(fontsize=7)
    ax8.grid(True, alpha=0.3)
    ax8.set_xlim(-30, 60)

    # =========================================================================
    # Panel 9: Energy density evolution
    # =========================================================================
    ax9 = fig.add_subplot(3, 3, 9)

    # Energy density |u|^2 for dispersive packet
    for t, color in zip([0, 10, 20, 30, 40], colors):
        wave = wave_packet_dispersive(x, t, x0, sigma0, k0, omega0, dispersion)
        energy_density = wave**2

        # Normalize for visibility
        if np.max(energy_density) > 0:
            energy_density = energy_density / np.max(energy_density)

        ax9.plot(x, energy_density, color=color, lw=2, label=f't = {t}')

    ax9.set_xlabel('Position x')
    ax9.set_ylabel('Energy Density (normalized)')
    ax9.set_title('Energy Density Spreading\n(Total energy conserved, peak decreases)')
    ax9.legend(fontsize=8)
    ax9.grid(True, alpha=0.3)
    ax9.set_xlim(-30, 80)

    plt.suptitle('Wave Packet Dispersion\n'
                 'Different wavelengths travel at different speeds, causing spreading',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'wave_dispersion.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'wave_dispersion.png')}")


if __name__ == "__main__":
    main()
