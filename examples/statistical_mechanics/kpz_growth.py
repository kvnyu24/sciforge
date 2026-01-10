"""
Experiment 144: KPZ Growth (Kardar-Parisi-Zhang)

This example demonstrates surface growth in the KPZ universality class
using a simple discrete model (ballistic deposition with relaxation).

The KPZ equation describes interface growth:
dh/dt = nu * nabla^2 h + (lambda/2) * (nabla h)^2 + eta(x,t)

where:
- h(x,t) = height at position x and time t
- nu = surface tension (smoothing)
- lambda = nonlinear coupling (lateral growth)
- eta = noise

The interface width scales as:
w(L,t) ~ L^alpha * f(t/L^z)

where:
- w = sqrt(<(h - <h>)^2>) is the interface width
- alpha = roughness exponent = 1/2 in 1+1D
- z = dynamic exponent = 3/2 in 1+1D
- beta = alpha/z = 1/3 (growth exponent)

For early times: w ~ t^beta
For saturation: w_sat ~ L^alpha
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# KPZ exponents in 1+1 dimensions
ALPHA = 0.5   # Roughness exponent
BETA = 1/3    # Growth exponent
Z = 3/2       # Dynamic exponent


def random_deposition(L, n_steps):
    """
    Random deposition model (not KPZ, for comparison).
    Particles drop vertically and stack.

    This gives alpha = infinity (uncorrelated), beta = 1/2.
    """
    heights = np.zeros(L)

    width_history = []
    for step in range(n_steps):
        # Deposit particle at random position
        x = np.random.randint(L)
        heights[x] += 1

        # Record width
        width = np.std(heights)
        width_history.append(width)

    return heights, np.array(width_history)


def ballistic_deposition(L, n_steps):
    """
    Ballistic deposition model (in KPZ universality class).
    Particles drop vertically and stick to first contact point,
    either on top of a column or on the side of a neighbor.
    """
    heights = np.zeros(L, dtype=int)

    width_history = []
    mean_height = []

    for step in range(n_steps):
        # Deposit particle at random position
        x = np.random.randint(L)

        # Height is max of current position and neighbors (side sticking)
        neighbor_heights = [heights[(x-1) % L], heights[x], heights[(x+1) % L]]
        heights[x] = max(neighbor_heights) + 1

        # Record statistics
        width = np.std(heights)
        width_history.append(width)
        mean_height.append(np.mean(heights))

    return heights, np.array(width_history), np.array(mean_height)


def kpz_discrete(L, n_steps, nu=1.0, lam=1.0, dt=0.01):
    """
    Direct integration of discretized KPZ equation.

    dh/dt = nu * (h_{i+1} - 2*h_i + h_{i-1})/dx^2
          + lambda/2 * ((h_{i+1} - h_{i-1})/(2*dx))^2
          + sqrt(D) * eta
    """
    heights = np.zeros(L)
    dx = 1.0
    D = 1.0  # Noise strength

    width_history = []
    mean_height = []

    for step in range(n_steps):
        # Laplacian (surface tension)
        laplacian = np.roll(heights, 1) - 2 * heights + np.roll(heights, -1)
        laplacian /= dx**2

        # Gradient squared (nonlinearity)
        gradient = (np.roll(heights, -1) - np.roll(heights, 1)) / (2 * dx)

        # Noise
        noise = np.random.normal(0, 1, L) * np.sqrt(D / dt)

        # Update
        heights += dt * (nu * laplacian + (lam / 2) * gradient**2 + noise)

        if step % 10 == 0:
            width_history.append(np.std(heights))
            mean_height.append(np.mean(heights))

    return heights, np.array(width_history), np.array(mean_height)


def eden_growth(L, n_steps):
    """
    Eden model: grow cluster by adding particles to perimeter.
    This is in the KPZ universality class for 1+1D interface.
    """
    # Simple 1D Eden model: interface grows by choosing random site
    # and advancing it if probability allows
    heights = np.zeros(L, dtype=int)

    width_history = []
    mean_height = []

    for step in range(n_steps):
        # Choose random position
        x = np.random.randint(L)

        # Advance height with probability based on local curvature
        # (smoother version with neighbor interaction)
        left_h = heights[(x-1) % L]
        right_h = heights[(x+1) % L]

        # Always advance
        heights[x] += 1

        # With some probability, also advance lagging neighbors
        if heights[x] > left_h + 1:
            if np.random.random() < 0.3:
                heights[(x-1) % L] += 1
        if heights[x] > right_h + 1:
            if np.random.random() < 0.3:
                heights[(x+1) % L] += 1

        if step % L == 0:
            width_history.append(np.std(heights))
            mean_height.append(np.mean(heights))

    return heights, np.array(width_history), np.array(mean_height)


def main():
    print("KPZ Surface Growth")
    print("=" * 50)
    print(f"1+1D KPZ exponents:")
    print(f"  alpha = {ALPHA} (roughness)")
    print(f"  beta = {BETA:.4f} (growth)")
    print(f"  z = {Z} (dynamic)")

    # Parameters
    L_values = [64, 128, 256, 512]
    n_steps = 50000

    print(f"\nRunning simulations...")

    # Run ballistic deposition for different L
    results = {}
    for L in L_values:
        print(f"  L = {L}...", end=' ')
        heights, widths, mean_h = ballistic_deposition(L, n_steps)
        results[L] = {'heights': heights, 'widths': widths, 'mean_h': mean_h}
        print(f"done (w_sat = {widths[-1]:.2f})")

    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Plot 1: Final interface profiles
    ax1 = axes[0, 0]
    for L in L_values[:3]:
        x = np.arange(L)
        h = results[L]['heights'] - np.mean(results[L]['heights'])
        ax1.plot(x, h, alpha=0.7, label=f'L = {L}')
    ax1.set_xlabel('Position x', fontsize=12)
    ax1.set_ylabel('Height h - <h>', fontsize=12)
    ax1.set_title('Interface Profiles (Ballistic Deposition)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Width evolution w(t)
    ax2 = axes[0, 1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(L_values)))

    for L, color in zip(L_values, colors):
        t = np.arange(len(results[L]['widths']))
        ax2.loglog(t, results[L]['widths'], color=color, alpha=0.7,
                   label=f'L = {L}')

    # Theoretical early-time growth
    t_theory = np.logspace(0, 3, 100)
    ax2.loglog(t_theory, 0.3 * t_theory**BETA, 'k--', lw=2, alpha=0.5,
               label=f'$t^{{\\beta}}$, $\\beta = {BETA:.2f}$')

    ax2.set_xlabel('Time t (particles deposited)', fontsize=12)
    ax2.set_ylabel('Interface width w', fontsize=12)
    ax2.set_title('Width Evolution', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')

    # Plot 3: Saturation width vs L
    ax3 = axes[0, 2]
    w_sat = [results[L]['widths'][-1000:].mean() for L in L_values]

    ax3.loglog(L_values, w_sat, 'bo-', markersize=8, label='Data')

    # Fit for alpha
    log_L = np.log(L_values)
    log_w = np.log(w_sat)
    slope, intercept, _, _, _ = linregress(log_L, log_w)

    L_fit = np.linspace(min(L_values), max(L_values), 50)
    ax3.loglog(L_fit, np.exp(intercept) * L_fit**slope, 'r--', lw=2,
               label=f'Fit: $\\alpha$ = {slope:.3f}')
    ax3.loglog(L_fit, L_fit**ALPHA * w_sat[0] / L_values[0]**ALPHA, 'g:',
               lw=2, alpha=0.5, label=f'Theory: $\\alpha$ = {ALPHA}')

    ax3.set_xlabel('System size L', fontsize=12)
    ax3.set_ylabel('Saturation width $w_{sat}$', fontsize=12)
    ax3.set_title(r'$w_{sat} \sim L^{\alpha}$', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3, which='both')

    # Plot 4: Data collapse w/L^alpha vs t/L^z
    ax4 = axes[1, 0]

    for L, color in zip(L_values, colors):
        t = np.arange(len(results[L]['widths']))
        x = t / L**Z
        y = results[L]['widths'] / L**ALPHA
        ax4.plot(x, y, color=color, alpha=0.7, label=f'L = {L}')

    ax4.set_xlabel(r'$t / L^z$', fontsize=12)
    ax4.set_ylabel(r'$w / L^{\alpha}$', fontsize=12)
    ax4.set_title('Data Collapse (KPZ Scaling)', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 5)

    # Plot 5: Height-height correlation function
    ax5 = axes[1, 1]
    L = 256
    h = results[L]['heights']
    h_centered = h - np.mean(h)

    # Compute C(r) = <(h(x+r) - h(x))^2>
    r_values = np.arange(1, L // 2)
    C_r = []
    for r in r_values:
        diff = np.roll(h_centered, -r) - h_centered
        C_r.append(np.mean(diff**2))

    ax5.loglog(r_values, C_r, 'b-', lw=2)

    # Theoretical: C(r) ~ r^(2*alpha)
    r_fit = np.logspace(0, np.log10(L//4), 50)
    ax5.loglog(r_fit, r_fit**(2*ALPHA) * C_r[0], 'r--', lw=2,
               label=f'$r^{{2\\alpha}}$, $\\alpha$ = {ALPHA}')

    ax5.set_xlabel('Distance r', fontsize=12)
    ax5.set_ylabel(r'$C(r) = \langle(h(x+r) - h(x))^2\rangle$', fontsize=12)
    ax5.set_title('Height-Height Correlation', fontsize=12)
    ax5.legend()
    ax5.grid(True, alpha=0.3, which='both')

    # Plot 6: Compare models
    ax6 = axes[1, 2]

    L = 256
    n_comp = 20000

    print("\nComparing growth models...")

    # Random deposition
    _, widths_rd = random_deposition(L, n_comp)
    t = np.arange(len(widths_rd))

    # Ballistic deposition (already computed)
    widths_bd = results[L]['widths'][:n_comp]

    ax6.loglog(t, widths_rd, 'b-', alpha=0.5, label='Random deposition')
    ax6.loglog(t[:len(widths_bd)], widths_bd, 'r-', alpha=0.7,
               label='Ballistic deposition (KPZ)')

    # Theoretical lines
    t_th = np.logspace(0, 4, 100)
    ax6.loglog(t_th, 0.5 * t_th**0.5, 'b--', alpha=0.5, label=r'$\beta = 0.5$ (RD)')
    ax6.loglog(t_th, 0.3 * t_th**BETA, 'r--', alpha=0.5, label=f'$\\beta = {BETA:.2f}$ (KPZ)')

    ax6.set_xlabel('Time t', fontsize=12)
    ax6.set_ylabel('Width w', fontsize=12)
    ax6.set_title('Model Comparison', fontsize=12)
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3, which='both')
    ax6.set_xlim(1, n_comp)

    plt.suptitle('KPZ Surface Growth: Ballistic Deposition Model',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Print summary
    print("\n" + "=" * 50)
    print("KPZ Growth Analysis")
    print("=" * 50)
    print(f"{'L':>8} {'w_sat':>12} {'Expected':>12}")
    print("-" * 35)
    for L in L_values:
        w_sat_measured = results[L]['widths'][-1000:].mean()
        w_sat_expected = L**ALPHA  # Proportionality constant varies
        print(f"{L:>8} {w_sat_measured:>12.3f}")

    print(f"\nMeasured alpha = {slope:.3f} (expected {ALPHA})")

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'kpz_growth.png'),
                dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {os.path.join(output_dir, 'kpz_growth.png')}")


if __name__ == "__main__":
    main()
