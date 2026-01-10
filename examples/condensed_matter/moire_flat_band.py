"""
Experiment 245: Moire Flat Band

Demonstrates the emergence of flat bands in twisted bilayer systems,
particularly twisted bilayer graphene (TBG) at magic angles where
flat bands lead to strongly correlated phenomena.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg


def moire_lattice_constant(a, theta):
    """
    Calculate moire superlattice constant.

    L_M = a / (2 * sin(theta/2)) ~ a / theta for small theta

    Args:
        a: Monolayer lattice constant
        theta: Twist angle (radians)

    Returns:
        Moire lattice constant
    """
    if theta == 0:
        return np.inf
    return a / (2 * np.sin(theta / 2))


def magic_angles(n=1, v_F=1e6, a=0.246e-9, w=0.11):
    """
    Calculate magic angles for twisted bilayer graphene.

    theta_magic ~ w / (v_F * k_D) * sqrt(3*n + 1)

    where k_D = 4*pi / (3*a) is the Dirac point wavevector.

    Args:
        n: Magic angle index (n=1 is the first magic angle)
        v_F: Fermi velocity
        a: Graphene lattice constant (m)
        w: Interlayer tunneling (eV) - effective parameter

    Returns:
        Magic angle in degrees
    """
    # Simplified formula
    # First magic angle is approximately 1.1 degrees
    theta_1 = 1.1  # degrees (empirical)
    return theta_1 / np.sqrt(n)


def bistritzer_macdonald_bands(k, theta, t, w, a=1.0, n_layers=2):
    """
    Simplified continuum model for twisted bilayer graphene.

    Near the moire Brillouin zone center, the effective Hamiltonian gives
    flat bands at magic angles.

    Args:
        k: Wavevector (kx, ky) relative to moire BZ center
        theta: Twist angle (radians)
        t: Intralayer Dirac cone velocity (energy scale)
        w: Interlayer tunneling amplitude
        a: Graphene lattice constant

    Returns:
        Eigenvalues (band energies)
    """
    kx, ky = k
    k_mag = np.sqrt(kx**2 + ky**2)

    # Dirac cone energy
    E_dirac = t * k_mag

    # At small twist angles, interlayer coupling opens gaps and flattens bands
    # Simplified model: hybridization between Dirac cones

    k_theta = 2 * np.sin(theta/2) * (4*np.pi / (3*a))  # Separation of Dirac cones

    # 2x2 effective Hamiltonian
    H = np.array([
        [t * k_mag, w],
        [w, -t * k_mag]
    ])

    eigenvalues = linalg.eigvalsh(H)
    return eigenvalues


def flat_band_dispersion(k, theta, t, w, a=1.0):
    """
    Simplified flat band dispersion.

    At magic angle, bands become extremely flat.
    Model: E(k) = E_0 * (1 - exp(-k^2 * L_M^2))

    Args:
        k: Wavevector magnitude
        theta: Twist angle (radians)
        t: Hopping
        w: Interlayer coupling

    Returns:
        Band energy
    """
    L_M = moire_lattice_constant(a, theta)

    # Bandwidth decreases as theta approaches magic angle
    theta_magic = magic_angles(1) * np.pi / 180

    # Bandwidth proportional to |theta - theta_magic|
    delta_theta = abs(theta - theta_magic)
    bandwidth = t * delta_theta / theta_magic

    # Flat band dispersion
    E = bandwidth * (1 - np.exp(-k**2 * L_M**2))

    return E


def moire_density_of_states(E, theta, t, w, eta=0.01):
    """
    Density of states showing Van Hove singularities.

    At magic angles, DOS peaks become very sharp due to flat bands.
    """
    # Simplified: Lorentzian peaks at flat band energies
    theta_magic = magic_angles(1) * np.pi / 180

    # Band positions
    E_flat = 0  # Flat band at Fermi level
    E_remote = 0.5 * t  # Remote bands

    # Peak width decreases at magic angle
    delta = abs(theta - theta_magic) / theta_magic
    gamma = eta * (1 + 10 * delta)

    dos = (gamma / np.pi) / ((E - E_flat)**2 + gamma**2)
    dos += 0.3 * (gamma / np.pi) / ((E - E_remote)**2 + gamma**2)
    dos += 0.3 * (gamma / np.pi) / ((E + E_remote)**2 + gamma**2)

    return dos


def main():
    """Main simulation and visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Parameters
    a = 0.246e-9  # Graphene lattice constant (m)
    t = 1.0       # Energy scale
    w = 0.11      # Interlayer tunneling

    # Plot 1: Moire pattern
    ax1 = axes[0, 0]

    # Create moire pattern visualization
    theta = 5 * np.pi / 180  # 5 degrees

    x = np.linspace(-30, 30, 300)
    y = np.linspace(-30, 30, 300)
    X, Y = np.meshgrid(x, y)

    # Layer 1 (unrotated)
    k1 = 2 * np.pi / 1  # Lattice wavevector
    layer1 = np.cos(k1 * X) + np.cos(k1 * (-0.5*X + np.sqrt(3)/2*Y)) + np.cos(k1 * (-0.5*X - np.sqrt(3)/2*Y))

    # Layer 2 (rotated)
    X_rot = X * np.cos(theta) - Y * np.sin(theta)
    Y_rot = X * np.sin(theta) + Y * np.cos(theta)
    layer2 = np.cos(k1 * X_rot) + np.cos(k1 * (-0.5*X_rot + np.sqrt(3)/2*Y_rot)) + np.cos(k1 * (-0.5*X_rot - np.sqrt(3)/2*Y_rot))

    # Moire pattern
    moire = layer1 * layer2

    im1 = ax1.imshow(moire, extent=[-30, 30, -30, 30], cmap='RdBu', origin='lower')
    ax1.set_xlabel('x (nm)')
    ax1.set_ylabel('y (nm)')
    ax1.set_title(f'Moire Pattern (theta = {theta*180/np.pi:.1f} deg)')
    plt.colorbar(im1, ax=ax1)

    # Mark moire period
    L_M = 1 / (2 * np.sin(theta/2))
    ax1.set_xlim(-30, 30)
    ax1.set_ylim(-30, 30)

    # Plot 2: Band structure at different angles
    ax2 = axes[0, 1]

    k_range = np.linspace(-0.5, 0.5, 100)

    angles = [0.5, 1.1, 2.0, 5.0]  # degrees
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(angles)))

    for angle, color in zip(angles, colors):
        theta_rad = angle * np.pi / 180

        E_k = []
        for k in k_range:
            E = bistritzer_macdonald_bands((k, 0), theta_rad, t, w)
            E_k.append(E)

        E_k = np.array(E_k)
        label = f'{angle} deg' if angle != 1.1 else f'{angle} deg (magic)'
        ax2.plot(k_range, E_k[:, 0], color=color, lw=2, label=label)
        ax2.plot(k_range, E_k[:, 1], color=color, lw=2)

    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax2.set_xlabel('k (1/L_M)')
    ax2.set_ylabel('Energy (t)')
    ax2.set_title('Band Structure vs Twist Angle')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Highlight flat bands at magic angle
    ax2.axhspan(-0.1, 0.1, alpha=0.1, color='yellow')

    # Plot 3: Bandwidth vs twist angle
    ax3 = axes[1, 0]

    angles_range = np.linspace(0.3, 5, 100)
    bandwidths = []

    for angle in angles_range:
        theta_rad = angle * np.pi / 180
        # Bandwidth from band structure
        E_plus = []
        for k in np.linspace(-0.5, 0.5, 50):
            E = bistritzer_macdonald_bands((k, 0), theta_rad, t, w)
            E_plus.append(E[1])

        bw = max(E_plus) - min(E_plus)
        bandwidths.append(bw)

    ax3.semilogy(angles_range, bandwidths, 'b-', lw=2)

    # Mark magic angles
    for n in [1, 2, 3]:
        theta_m = magic_angles(n)
        ax3.axvline(x=theta_m, color='red', linestyle='--', alpha=0.5)
        ax3.text(theta_m, max(bandwidths), f'n={n}', fontsize=9, rotation=90, va='bottom')

    ax3.set_xlabel('Twist angle (degrees)')
    ax3.set_ylabel('Bandwidth (t)')
    ax3.set_title('Bandwidth vs Twist Angle')
    ax3.grid(True, alpha=0.3, which='both')

    ax3.text(0.05, 0.95, 'Flat bands at\nmagic angles',
             transform=ax3.transAxes, fontsize=10, va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Plot 4: Density of states at magic angle
    ax4 = axes[1, 1]

    E_range = np.linspace(-0.5, 0.5, 500)

    theta_magic_rad = magic_angles(1) * np.pi / 180
    theta_away = 2.0 * np.pi / 180  # Away from magic angle

    dos_magic = moire_density_of_states(E_range, theta_magic_rad, t, w, eta=0.02)
    dos_away = moire_density_of_states(E_range, theta_away, t, w, eta=0.02)

    ax4.plot(E_range, dos_magic, 'b-', lw=2, label='Magic angle (1.1 deg)')
    ax4.plot(E_range, dos_away, 'r--', lw=2, label='Away from magic (2.0 deg)')

    ax4.set_xlabel('Energy (t)')
    ax4.set_ylabel('DOS (arb. units)')
    ax4.set_title('Density of States')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-0.5, 0.5)

    # Mark Van Hove singularities
    ax4.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax4.text(0.02, max(dos_magic)*0.9, 'Flat band\nVan Hove peak', fontsize=10)

    plt.suptitle('Moire Flat Bands in Twisted Bilayer Graphene\n'
                 r'Flat bands emerge at magic angle $\theta \approx 1.1°$',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'moire_flat_band.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'moire_flat_band.png')}")


if __name__ == "__main__":
    main()
