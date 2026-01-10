"""
Experiment 188: Relativistic Doppler Effect and Aberration

This experiment demonstrates the relativistic Doppler effect (frequency shift)
and aberration (change in apparent direction of light sources).

Physical concepts:
- Longitudinal and transverse Doppler effects
- Relativistic aberration of light
- Beaming and headlight effect
- Applications to astrophysics (jets, cosmic rays)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt


def doppler_factor(v, theta, c=1.0):
    """
    Calculate relativistic Doppler factor.

    D = 1 / (gamma * (1 - beta * cos(theta)))

    Args:
        v: Source velocity
        theta: Angle between velocity and observation direction (in source frame)
        c: Speed of light

    Returns:
        Doppler factor D = f_observed / f_emitted
    """
    beta = v / c
    gamma = 1 / np.sqrt(1 - beta**2)
    return 1 / (gamma * (1 - beta * np.cos(theta)))


def longitudinal_doppler(v, approaching=True, c=1.0):
    """
    Longitudinal Doppler shift (source moving directly toward/away).

    f_obs/f_emit = sqrt((1 + beta) / (1 - beta)) for approaching
    f_obs/f_emit = sqrt((1 - beta) / (1 + beta)) for receding
    """
    beta = v / c
    if approaching:
        return np.sqrt((1 + beta) / (1 - beta))
    else:
        return np.sqrt((1 - beta) / (1 + beta))


def transverse_doppler(v, c=1.0):
    """
    Transverse Doppler shift (source moving perpendicular).

    f_obs/f_emit = 1/gamma (time dilation only)
    """
    beta = v / c
    gamma = 1 / np.sqrt(1 - beta**2)
    return 1 / gamma


def aberration(theta_emit, v, c=1.0):
    """
    Relativistic aberration of light.

    cos(theta_obs) = (cos(theta_emit) - beta) / (1 - beta * cos(theta_emit))

    Args:
        theta_emit: Emission angle in source rest frame
        v: Source velocity
        c: Speed of light

    Returns:
        Observed angle theta_obs
    """
    beta = v / c
    cos_emit = np.cos(theta_emit)
    cos_obs = (cos_emit - beta) / (1 - beta * cos_emit)
    # Handle numerical issues near +/- 1
    cos_obs = np.clip(cos_obs, -1, 1)
    return np.arccos(cos_obs)


def beaming_solid_angle_ratio(v, c=1.0):
    """
    Calculate the ratio of apparent solid angle (forward beaming).

    For isotropic emission in source frame, what fraction appears
    in forward hemisphere in observer frame?
    """
    beta = v / c
    gamma = 1 / np.sqrt(1 - beta**2)
    # Half of the emission (hemisphere in source frame) gets
    # compressed into a cone of half-angle ~1/gamma
    return 1 / gamma**2


def main():
    c = 1.0

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ==========================================================================
    # Plot 1: Doppler shift vs velocity
    # ==========================================================================
    ax1 = axes[0, 0]

    v_range = np.linspace(0.01, 0.99, 100) * c

    d_approaching = longitudinal_doppler(v_range, True, c)
    d_receding = longitudinal_doppler(v_range, False, c)
    d_transverse = transverse_doppler(v_range, c)

    ax1.semilogy(v_range/c, d_approaching, 'b-', lw=2, label='Approaching (blueshift)')
    ax1.semilogy(v_range/c, d_receding, 'r-', lw=2, label='Receding (redshift)')
    ax1.semilogy(v_range/c, d_transverse, 'g-', lw=2, label='Transverse (time dilation)')

    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5)

    ax1.set_xlabel('Source velocity v/c')
    ax1.set_ylabel('Frequency ratio f_obs / f_emit')
    ax1.set_title('Relativistic Doppler Effect')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.01, 100)

    # Mark some notable values
    v_mark = 0.9 * c
    ax1.axvline(x=0.9, color='purple', linestyle=':', alpha=0.5)
    ax1.annotate(f'At v=0.9c:\nBlueshift: {longitudinal_doppler(v_mark, True, c):.1f}x\n'
                f'Redshift: {longitudinal_doppler(v_mark, False, c):.2f}x',
                xy=(0.9, 1), xytext=(0.5, 5),
                fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ==========================================================================
    # Plot 2: Doppler factor vs angle
    # ==========================================================================
    ax2 = axes[0, 1]

    theta_range = np.linspace(0, np.pi, 200)
    velocities = [0.3, 0.6, 0.9, 0.99]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(velocities)))

    for v, color in zip(velocities, colors):
        D = doppler_factor(v * c, theta_range, c)
        ax2.plot(np.degrees(theta_range), D, '-', lw=2, color=color,
                label=f'v = {v}c')

    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=90, color='gray', linestyle=':', alpha=0.5)

    ax2.set_xlabel('Observation angle (degrees)')
    ax2.set_ylabel('Doppler factor D')
    ax2.set_title('Doppler Factor vs Observation Angle')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 180)
    ax2.set_yscale('log')
    ax2.set_ylim(0.01, 100)

    ax2.annotate('Approaching\n(blueshift)', xy=(10, 10), fontsize=10, color='blue')
    ax2.annotate('Receding\n(redshift)', xy=(160, 0.1), fontsize=10, color='red')

    # ==========================================================================
    # Plot 3: Relativistic aberration
    # ==========================================================================
    ax3 = axes[1, 0]

    theta_emit = np.linspace(0, np.pi, 200)

    for v, color in zip(velocities, colors):
        theta_obs = aberration(theta_emit, v * c, c)
        ax3.plot(np.degrees(theta_emit), np.degrees(theta_obs), '-', lw=2,
                color=color, label=f'v = {v}c')

    ax3.plot([0, 180], [0, 180], 'k--', lw=1, alpha=0.5, label='No aberration')

    ax3.set_xlabel('Emission angle in source frame (degrees)')
    ax3.set_ylabel('Observed angle in lab frame (degrees)')
    ax3.set_title('Relativistic Aberration')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Mark the beaming cone
    for v, color in zip([0.9, 0.99], ['green', 'red']):
        gamma = 1 / np.sqrt(1 - v**2)
        beam_angle = np.degrees(np.arcsin(1/gamma))
        ax3.axhline(y=beam_angle, color=color, linestyle=':', alpha=0.5)
        ax3.text(100, beam_angle + 3, f'Beam cone ~{beam_angle:.1f} deg (v={v}c)',
                fontsize=8, color=color)

    # ==========================================================================
    # Plot 4: Beaming / Headlight effect visualization
    # ==========================================================================
    ax4 = axes[1, 1]

    # Draw isotropic emission pattern (rest frame) and beamed pattern
    theta = np.linspace(0, 2*np.pi, 100)

    # Rest frame: isotropic
    r_rest = np.ones_like(theta)

    # Moving frame: beamed pattern (intensity ~ D^3 for optically thin source)
    v_beam = 0.9 * c
    D_beam = doppler_factor(v_beam, theta, c)
    r_beam = D_beam**3  # Intensity boost
    r_beam = r_beam / np.max(r_beam)  # Normalize

    ax4.plot(r_rest * np.cos(theta), r_rest * np.sin(theta), 'b-', lw=2,
            label='Rest frame (isotropic)')
    ax4.plot(r_beam * np.cos(theta) + 0.5, r_beam * np.sin(theta), 'r-', lw=2,
            label=f'Moving at v={v_beam/c}c (beamed)')

    # Direction of motion
    ax4.annotate('', xy=(1.5, 0), xytext=(0.5, 0),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax4.text(1.0, -0.15, 'Direction of motion', fontsize=10, ha='center')

    ax4.set_xlim(-1.5, 2)
    ax4.set_ylim(-1.5, 1.5)
    ax4.set_aspect('equal')
    ax4.set_title('Relativistic Beaming (Headlight Effect)')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)

    # Add physics explanation
    textstr = ('Beaming Effects:\n'
               f'- Forward intensity enhanced by D^3\n'
               f'- Emission concentrated in cone of angle ~1/gamma\n'
               f'- At v={v_beam/c}c: gamma={1/np.sqrt(1-v_beam**2):.1f}\n'
               f'- Beam half-angle ~{np.degrees(np.arcsin(np.sqrt(1-v_beam**2))):.0f} deg')
    ax4.text(-1.4, -0.8, textstr, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('Relativistic Doppler Effect and Aberration\n'
                 'Applications: Relativistic jets, cosmic rays, binary pulsars',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Print numerical examples
    print("Relativistic Doppler and Aberration Examples:")
    print("-" * 60)

    for v in [0.5, 0.9, 0.99, 0.999]:
        gamma = 1 / np.sqrt(1 - v**2)
        print(f"\nVelocity v = {v}c (gamma = {gamma:.2f}):")
        print(f"  Blueshift (approaching): {longitudinal_doppler(v*c, True, c):.3f}x")
        print(f"  Redshift (receding): {longitudinal_doppler(v*c, False, c):.4f}x")
        print(f"  Transverse (90 deg): {transverse_doppler(v*c, c):.4f}x")
        print(f"  Beam half-angle: {np.degrees(np.arcsin(1/gamma)):.2f} degrees")

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'doppler_aberration.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")


if __name__ == "__main__":
    main()
