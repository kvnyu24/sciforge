"""
Experiment 139: 2D Ising Model Correlation Length

This example demonstrates the divergence of the correlation length at the
critical temperature in the 2D Ising model.

The spin-spin correlation function is:
G(r) = <s_i * s_j> - <s_i><s_j>

For large r and T > T_c:
G(r) ~ exp(-r / xi) / r^(d-2+eta)

where:
- xi = correlation length (diverges at T_c)
- eta = anomalous dimension = 1/4 for 2D Ising

Near T_c, the correlation length diverges as:
xi ~ |T - T_c|^(-nu) where nu = 1 (exact for 2D Ising)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Critical temperature for 2D Ising
T_C = 2.0 / np.log(1 + np.sqrt(2))  # ≈ 2.269

# Critical exponents for 2D Ising
NU = 1.0  # Correlation length exponent
ETA = 0.25  # Anomalous dimension


def initialize_lattice(L, state='random'):
    """Initialize square lattice of spins."""
    if state == 'random':
        return np.random.choice([-1, 1], size=(L, L))
    return np.ones((L, L), dtype=int)


def metropolis_sweep(spins, T, J=1.0):
    """Perform one Metropolis sweep."""
    L = spins.shape[0]

    for _ in range(L * L):
        i, j = np.random.randint(L), np.random.randint(L)
        s = spins[i, j]

        neighbors = (spins[(i+1) % L, j] + spins[(i-1) % L, j] +
                    spins[i, (j+1) % L] + spins[i, (j-1) % L])
        delta_E = 2 * J * s * neighbors

        if delta_E <= 0 or np.random.random() < np.exp(-delta_E / T):
            spins[i, j] = -s


def compute_correlation_function(spins, max_r=None):
    """
    Compute the spin-spin correlation function G(r).

    Uses translational invariance to average over all pairs at distance r.
    Computes correlation along x-direction for simplicity.

    Args:
        spins: 2D array of spins
        max_r: Maximum distance to compute (default: L/2)

    Returns:
        r_values, G_values: Arrays of distances and correlations
    """
    L = spins.shape[0]
    if max_r is None:
        max_r = L // 2

    # Average magnetization
    m = np.mean(spins)

    # Compute correlations
    r_values = np.arange(max_r + 1)
    G_values = np.zeros(max_r + 1)

    for r in range(max_r + 1):
        # Compute <s_i * s_{i+r}> averaged over all sites
        # Using np.roll for efficient periodic correlation
        shifted = np.roll(spins, -r, axis=0)
        G_values[r] = np.mean(spins * shifted) - m**2

    return r_values, G_values


def compute_correlation_fft(spins):
    """
    Compute correlation function using FFT (faster for large systems).
    """
    L = spins.shape[0]
    m = np.mean(spins)

    # Use FFT to compute correlation
    spins_centered = spins - m
    fft_spins = np.fft.fft2(spins_centered)
    power_spectrum = np.abs(fft_spins)**2 / (L * L)
    correlation = np.fft.ifft2(power_spectrum).real

    # Average over angles (radial average)
    max_r = L // 2
    r_values = np.arange(max_r)
    G_values = np.zeros(max_r)

    for r in range(max_r):
        # Take correlation at distance r along x-axis (averaged with y-axis)
        G_values[r] = (correlation[r, 0] + correlation[0, r]) / 2

    return r_values, G_values


def fit_correlation_length(r, G, r_min=1, r_max=None):
    """
    Fit correlation function to extract correlation length.

    Fits G(r) = A * exp(-r/xi) / r^(eta) for large r.

    Returns:
        xi: Correlation length
        A: Amplitude
    """
    if r_max is None:
        r_max = len(r) - 1

    mask = (r >= r_min) & (r <= r_max) & (G > 0)
    r_fit = r[mask]
    G_fit = G[mask]

    if len(r_fit) < 3:
        return np.nan, np.nan

    # Log-linear fit: log(G * r^eta) = log(A) - r/xi
    try:
        log_G = np.log(G_fit * r_fit**ETA)

        def model(r, log_A, inv_xi):
            return log_A - r * inv_xi

        popt, _ = curve_fit(model, r_fit, log_G, p0=[0, 0.5])
        xi = 1.0 / popt[1] if popt[1] > 0 else np.inf
        A = np.exp(popt[0])
        return xi, A
    except (RuntimeError, ValueError):
        return np.nan, np.nan


def measure_correlation_length(L, T, n_equilibrate, n_measure, n_samples=10):
    """
    Measure correlation length at temperature T.

    Args:
        L: Lattice size
        T: Temperature
        n_equilibrate: Equilibration sweeps
        n_measure: Measurement sweeps per sample
        n_samples: Number of independent samples

    Returns:
        xi_mean, xi_std, G_avg, r: Correlation length and averaged correlation function
    """
    spins = initialize_lattice(L)
    max_r = L // 2

    # Equilibration
    for _ in range(n_equilibrate):
        metropolis_sweep(spins, T)

    # Measurement
    G_sum = np.zeros(max_r + 1)
    xi_samples = []

    for _ in range(n_samples):
        for _ in range(n_measure):
            metropolis_sweep(spins, T)

        r, G = compute_correlation_function(spins, max_r)
        G_sum += G

        xi, _ = fit_correlation_length(r, G)
        if not np.isnan(xi):
            xi_samples.append(xi)

    G_avg = G_sum / n_samples
    xi_mean = np.mean(xi_samples) if xi_samples else np.nan
    xi_std = np.std(xi_samples) if len(xi_samples) > 1 else 0

    return xi_mean, xi_std, G_avg, r


def main():
    print("2D Ising Model: Correlation Length")
    print("=" * 50)
    print(f"Critical temperature T_c = {T_C:.4f}")
    print(f"Correlation length exponent nu = {NU}")
    print(f"Anomalous dimension eta = {ETA}")

    # Parameters
    L = 64  # Larger lattice for better correlation measurement
    n_equilibrate = 3000
    n_measure = 500
    n_samples = 20

    # Temperatures to study
    T_values = [1.8, 2.0, 2.1, 2.2, T_C, 2.3, 2.4, 2.6, 3.0, 3.5]

    print(f"\nLattice size: {L} x {L}")
    print(f"Samples per temperature: {n_samples}")

    # Run measurements
    results = {'T': [], 'xi': [], 'xi_err': [], 'G': [], 'r': []}

    for i, T in enumerate(T_values):
        print(f"  T = {T:.2f} ({i+1}/{len(T_values)})", end='')
        xi, xi_err, G, r = measure_correlation_length(L, T, n_equilibrate,
                                                       n_measure, n_samples)
        results['T'].append(T)
        results['xi'].append(xi)
        results['xi_err'].append(xi_err)
        results['G'].append(G)
        results['r'].append(r)
        print(f"  xi = {xi:.2f}")

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Correlation function at different temperatures
    ax1 = axes[0, 0]
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(T_values)))

    for i, (T, G, r) in enumerate(zip(results['T'], results['G'], results['r'])):
        mask = G > 1e-5
        label = f'T = {T:.2f}' + (' (T_c)' if abs(T - T_C) < 0.01 else '')
        ax1.semilogy(r[mask], G[mask], 'o-', color=colors[i], label=label,
                     markersize=4, alpha=0.8)

    ax1.set_xlabel('Distance r', fontsize=12)
    ax1.set_ylabel('G(r)', fontsize=12)
    ax1.set_title('Spin-Spin Correlation Function', fontsize=12)
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_xlim(0, L // 2)

    # Plot 2: Correlation length vs temperature
    ax2 = axes[0, 1]
    T_arr = np.array(results['T'])
    xi_arr = np.array(results['xi'])
    xi_err_arr = np.array(results['xi_err'])

    # Filter out NaN values
    valid = ~np.isnan(xi_arr)
    ax2.errorbar(T_arr[valid], xi_arr[valid], yerr=xi_err_arr[valid],
                 fmt='o', capsize=3, color='blue')
    ax2.axvline(T_C, color='red', linestyle='--', label=f'$T_c$ = {T_C:.3f}')
    ax2.set_xlabel('Temperature (J/k_B)', fontsize=12)
    ax2.set_ylabel(r'Correlation length $\xi$', fontsize=12)
    ax2.set_title('Correlation Length vs Temperature', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Log-log plot of xi vs |T - T_c|
    ax3 = axes[1, 0]

    # Separate points above and below T_c
    above_Tc = T_arr > T_C + 0.05
    below_Tc = T_arr < T_C - 0.05

    t_above = T_arr[above_Tc & valid] - T_C
    xi_above = xi_arr[above_Tc & valid]

    t_below = T_C - T_arr[below_Tc & valid]
    xi_below = xi_arr[below_Tc & valid]

    if len(t_above) > 0:
        ax3.loglog(t_above, xi_above, 'bo', markersize=8, label='T > T_c')
    if len(t_below) > 0:
        ax3.loglog(t_below, xi_below, 'rs', markersize=8, label='T < T_c')

    # Theoretical line
    t_theory = np.logspace(-1, 0, 50)
    ax3.loglog(t_theory, 1.0 * t_theory**(-NU), 'k--', label=r'$\xi \sim t^{-1}$')

    ax3.set_xlabel('|T - T_c| / (J/k_B)', fontsize=12)
    ax3.set_ylabel(r'Correlation length $\xi$', fontsize=12)
    ax3.set_title('Critical Scaling of Correlation Length', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3, which='both')

    # Plot 4: G(r) * r^eta at T_c (should be straight line on semilog)
    ax4 = axes[1, 1]

    # Find T closest to T_c
    T_c_idx = np.argmin(np.abs(T_arr - T_C))
    G_Tc = results['G'][T_c_idx]
    r_Tc = results['r'][T_c_idx]

    mask = (r_Tc > 0) & (G_Tc > 1e-5)
    r_plot = r_Tc[mask]
    G_plot = G_Tc[mask]

    # At T_c: G(r) ~ 1/r^eta, so G*r^eta should be constant
    ax4.semilogy(r_plot, G_plot * r_plot**ETA, 'b-', lw=2,
                 label=f'$G(r) \\cdot r^{{\\eta}}$ at T_c')

    # Compare with above and below T_c
    T_above_idx = np.argmin(np.abs(T_arr - 2.6))
    T_below_idx = np.argmin(np.abs(T_arr - 2.0))

    for idx, label, color in [(T_above_idx, 'T = 2.6', 'red'),
                               (T_below_idx, 'T = 2.0', 'green')]:
        G_T = results['G'][idx]
        r_T = results['r'][idx]
        mask = (r_T > 0) & (G_T > 1e-5)
        ax4.semilogy(r_T[mask], G_T[mask] * r_T[mask]**ETA, color=color,
                     alpha=0.7, lw=2, label=label)

    ax4.set_xlabel('Distance r', fontsize=12)
    ax4.set_ylabel(r'$G(r) \cdot r^{\eta}$', fontsize=12)
    ax4.set_title(f'Critical Correlation: $\\eta$ = {ETA}', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3, which='both')
    ax4.set_xlim(1, L // 2)

    plt.suptitle(f'2D Ising Model: Correlation Length Analysis (L = {L})',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Print summary
    print("\n" + "=" * 50)
    print("Results Summary")
    print("=" * 50)
    print(f"{'T':>8} {'xi':>12} {'xi_err':>12}")
    print("-" * 35)
    for T, xi, xi_err in zip(results['T'], results['xi'], results['xi_err']):
        if not np.isnan(xi):
            print(f"{T:>8.3f} {xi:>12.2f} {xi_err:>12.2f}")

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'ising_2d_correlation.png'),
                dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {os.path.join(output_dir, 'ising_2d_correlation.png')}")


if __name__ == "__main__":
    main()
