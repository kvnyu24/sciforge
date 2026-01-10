"""
Experiment 153: WKB vs Exact Tunneling

This experiment compares the WKB (Wentzel-Kramers-Brillouin) approximation
for quantum tunneling with exact results, including:
- WKB transmission formula for rectangular barrier
- Comparison with exact analytical solution
- WKB for smooth barriers (triangular, parabolic)
- Validity of WKB approximation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


def exact_rectangular_transmission(E: float, V0: float, a: float,
                                    m: float = 1.0, hbar: float = 1.0) -> float:
    """
    Exact transmission coefficient for rectangular barrier.

    Barrier: V(x) = V0 for 0 < x < a, V(x) = 0 otherwise.

    Args:
        E: Particle energy
        V0: Barrier height
        a: Barrier width
        m: Particle mass
        hbar: Reduced Planck constant

    Returns:
        Transmission coefficient T
    """
    if E <= 0:
        return 0.0

    if E >= V0:
        # Above barrier
        k = np.sqrt(2 * m * E) / hbar
        k_prime = np.sqrt(2 * m * (E - V0)) / hbar
        sin_term = np.sin(k_prime * a)
        T = 1 / (1 + (k**2 + k_prime**2)**2 * sin_term**2 / (4 * k**2 * k_prime**2))
    else:
        # Below barrier - tunneling
        k = np.sqrt(2 * m * E) / hbar
        kappa = np.sqrt(2 * m * (V0 - E)) / hbar

        if kappa * a > 50:
            return 0.0

        sinh_term = np.sinh(kappa * a)
        T = 1 / (1 + (k**2 + kappa**2)**2 * sinh_term**2 / (4 * k**2 * kappa**2))

    return T


def wkb_rectangular_transmission(E: float, V0: float, a: float,
                                  m: float = 1.0, hbar: float = 1.0) -> float:
    """
    WKB approximation for rectangular barrier transmission.

    T_WKB = exp(-2 * integral of kappa(x) dx)
    For rectangular barrier: T_WKB = exp(-2 * kappa * a)

    Args:
        E: Particle energy
        V0: Barrier height
        a: Barrier width
        m: Particle mass
        hbar: Reduced Planck constant

    Returns:
        WKB transmission coefficient
    """
    if E >= V0 or E <= 0:
        return 1.0 if E >= V0 else 0.0

    kappa = np.sqrt(2 * m * (V0 - E)) / hbar
    gamma = 2 * kappa * a

    if gamma > 50:
        return 0.0

    return np.exp(-gamma)


def wkb_general_transmission(E: float, V_func, x1: float, x2: float,
                              m: float = 1.0, hbar: float = 1.0) -> float:
    """
    WKB transmission for general barrier shape.

    T_WKB = exp(-2/hbar * integral_{x1}^{x2} sqrt(2m(V(x) - E)) dx)

    where x1, x2 are classical turning points.

    Args:
        E: Particle energy
        V_func: Potential function V(x)
        x1: Left turning point
        x2: Right turning point
        m: Particle mass
        hbar: Reduced Planck constant

    Returns:
        WKB transmission coefficient
    """
    def integrand(x):
        V = V_func(x)
        if V > E:
            return np.sqrt(2 * m * (V - E))
        return 0.0

    gamma, _ = quad(integrand, x1, x2)
    gamma *= 2 / hbar

    if gamma > 50:
        return 0.0

    return np.exp(-gamma)


def triangular_barrier(x: float, V0: float, a: float) -> float:
    """Triangular barrier: V = V0(1 - x/a) for 0 < x < a."""
    if 0 < x < a:
        return V0 * (1 - x / a)
    return 0.0


def parabolic_barrier(x: float, V0: float, a: float) -> float:
    """Parabolic barrier: V = V0(1 - (2x/a - 1)^2) for 0 < x < a."""
    if 0 < x < a:
        return V0 * (1 - (2*x/a - 1)**2)
    return 0.0


def gaussian_barrier(x: float, V0: float, sigma: float) -> float:
    """Gaussian barrier: V = V0 * exp(-x^2 / (2*sigma^2))."""
    return V0 * np.exp(-x**2 / (2 * sigma**2))


def find_turning_points(E: float, V_func, x_min: float, x_max: float,
                        n_points: int = 1000) -> tuple:
    """Find classical turning points where E = V(x)."""
    x = np.linspace(x_min, x_max, n_points)
    V = np.array([V_func(xi) for xi in x])

    # Find where V crosses E
    crossings = []
    for i in range(len(x) - 1):
        if (V[i] - E) * (V[i+1] - E) < 0:
            # Linear interpolation for crossing point
            x_cross = x[i] + (x[i+1] - x[i]) * (E - V[i]) / (V[i+1] - V[i])
            crossings.append(x_cross)

    if len(crossings) >= 2:
        return crossings[0], crossings[-1]
    elif len(crossings) == 1:
        return crossings[0], crossings[0]
    else:
        return None, None


def main():
    # Parameters (natural units)
    m = 1.0
    hbar = 1.0
    V0 = 1.0
    a = 2.0  # Barrier width

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Plot 1: WKB vs Exact for rectangular barrier
    ax1 = axes[0, 0]

    energies = np.linspace(0.01, 0.99, 100) * V0

    T_exact = [exact_rectangular_transmission(E, V0, a, m, hbar) for E in energies]
    T_wkb = [wkb_rectangular_transmission(E, V0, a, m, hbar) for E in energies]

    ax1.semilogy(energies / V0, T_exact, 'b-', lw=2, label='Exact')
    ax1.semilogy(energies / V0, T_wkb, 'r--', lw=2, label='WKB')

    ax1.set_xlabel('Energy E / V0')
    ax1.set_ylabel('Transmission T (log scale)')
    ax1.set_title('Rectangular Barrier: WKB vs Exact')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)

    # Plot 2: Relative error
    ax2 = axes[0, 1]

    # Avoid division by zero
    T_exact_arr = np.array(T_exact)
    T_wkb_arr = np.array(T_wkb)
    valid = T_exact_arr > 1e-15
    relative_error = np.abs(T_wkb_arr[valid] - T_exact_arr[valid]) / T_exact_arr[valid]

    ax2.semilogy(energies[valid] / V0, relative_error, 'b-', lw=2)
    ax2.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='10% error')
    ax2.axhline(y=0.01, color='green', linestyle='--', alpha=0.7, label='1% error')

    ax2.set_xlabel('Energy E / V0')
    ax2.set_ylabel('Relative Error |T_WKB - T_exact| / T_exact')
    ax2.set_title('WKB Approximation Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Effect of barrier width
    ax3 = axes[0, 2]

    widths = [0.5, 1.0, 2.0, 4.0]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(widths)))

    for width, color in zip(widths, colors):
        E_test = 0.5 * V0
        T_exact_w = [exact_rectangular_transmission(E, V0, width, m, hbar)
                     for E in energies]
        T_wkb_w = [wkb_rectangular_transmission(E, V0, width, m, hbar)
                   for E in energies]

        ax3.semilogy(energies / V0, T_exact_w, '-', color=color, lw=2,
                     label=f'a = {width} (exact)')
        ax3.semilogy(energies / V0, T_wkb_w, '--', color=color, lw=1.5,
                     alpha=0.7)

    ax3.set_xlabel('Energy E / V0')
    ax3.set_ylabel('Transmission T (log scale)')
    ax3.set_title('WKB Improves for Thicker Barriers')
    ax3.legend(loc='lower right', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Different barrier shapes
    ax4 = axes[1, 0]

    x_plot = np.linspace(-1, 3, 500)

    # Define barrier functions with consistent parameters
    barriers = [
        ('Rectangular', lambda x: V0 if 0 < x < a else 0),
        ('Triangular', lambda x: triangular_barrier(x, V0, a)),
        ('Parabolic', lambda x: parabolic_barrier(x, V0, a)),
    ]

    colors_barrier = ['blue', 'green', 'red']

    for (name, V_func), color in zip(barriers, colors_barrier):
        V_vals = [V_func(xi) for xi in x_plot]
        ax4.plot(x_plot, V_vals, color=color, lw=2, label=name)

    ax4.axhline(y=0.5*V0, color='black', linestyle='--', alpha=0.5, label='E = 0.5 V0')
    ax4.set_xlabel('Position x')
    ax4.set_ylabel('Potential V(x)')
    ax4.set_title('Different Barrier Shapes')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-1, 3)

    # Plot 5: WKB for different barrier shapes
    ax5 = axes[1, 1]

    energies_shape = np.linspace(0.05, 0.95, 50) * V0

    for (name, V_func), color in zip(barriers, colors_barrier):
        T_wkb_shape = []
        for E in energies_shape:
            x1, x2 = find_turning_points(E, V_func, -0.5, a + 0.5)
            if x1 is not None and x2 is not None and x2 > x1:
                T = wkb_general_transmission(E, V_func, x1, x2, m, hbar)
            else:
                T = 1.0
            T_wkb_shape.append(T)

        ax5.semilogy(energies_shape / V0, T_wkb_shape, color=color, lw=2,
                     label=name)

    ax5.set_xlabel('Energy E / V0')
    ax5.set_ylabel('WKB Transmission T')
    ax5.set_title('WKB Transmission for Different Shapes')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Gaussian barrier (smooth)
    ax6 = axes[1, 2]

    sigma = 1.0
    x_gauss = np.linspace(-4*sigma, 4*sigma, 500)
    V_gauss = [gaussian_barrier(xi, V0, sigma) for xi in x_gauss]

    ax6_twin = ax6.twinx()
    ax6_twin.fill_between(x_gauss, 0, V_gauss, alpha=0.2, color='gray')
    ax6_twin.set_ylabel('Potential V(x)', color='gray')
    ax6_twin.set_ylim(0, 1.5*V0)

    energies_gauss = np.linspace(0.05, 0.95, 50) * V0
    T_gauss = []

    gaussian_func = lambda x: gaussian_barrier(x, V0, sigma)

    for E in energies_gauss:
        x1, x2 = find_turning_points(E, gaussian_func, -4*sigma, 4*sigma)
        if x1 is not None and x2 is not None and x2 > x1:
            T = wkb_general_transmission(E, gaussian_func, x1, x2, m, hbar)
        else:
            T = 1.0
        T_gauss.append(T)

    ax6.semilogy(energies_gauss / V0, T_gauss, 'b-', lw=2, label='WKB')

    # Inset showing potential shape
    ax6.set_xlabel('Energy E / V0')
    ax6.set_ylabel('WKB Transmission T')
    ax6.set_title(f'Gaussian Barrier (sigma = {sigma})')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Add text about WKB validity
    textstr = (r'WKB valid when: $\left|\frac{d\lambda}{dx}\right| \ll 1$'
               '\n'
               r'i.e., $\frac{\hbar}{(2m(V-E))^{3/2}} \left|\frac{dV}{dx}\right| \ll 1$')
    ax6.text(0.05, 0.02, textstr, transform=ax6.transAxes, fontsize=9,
             verticalalignment='bottom', bbox=dict(boxstyle='round',
             facecolor='wheat', alpha=0.5))

    plt.suptitle('WKB Approximation for Quantum Tunneling\n'
                 r'$T_{WKB} = \exp\left(-\frac{2}{\hbar}\int_{x_1}^{x_2}\sqrt{2m(V(x)-E)}dx\right)$',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'wkb_tunneling.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'wkb_tunneling.png')}")


if __name__ == "__main__":
    main()
