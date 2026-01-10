"""
Example demonstrating orbital precession from perturbation (toy GR model).

This example shows how a small perturbation to the Newtonian gravitational
potential leads to orbital precession, mimicking the effects of general
relativity. We use a 1/r^3 perturbation term which causes the perihelion
to advance with each orbit.

The GR correction to the effective potential is:
    V_eff = -GMm/r + L^2/(2mr^2) - GML^2/(mc^2 r^3)

The last term causes perihelion precession. For Mercury, this amounts to
43 arcseconds per century.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def simulate_precessing_orbit(m_central, m_orbiting, initial_r, initial_v,
                              perturbation_strength, G, t_final, dt):
    """
    Simulate orbit with 1/r^3 perturbation causing precession.

    The perturbation adds a term: F_pert = -alpha / r^4 * r_hat
    where alpha controls the precession rate.

    Args:
        m_central: Central mass
        m_orbiting: Orbiting mass
        initial_r: Initial position vector
        initial_v: Initial velocity vector
        perturbation_strength: Strength of 1/r^3 perturbation (alpha)
        G: Gravitational constant
        t_final: Total simulation time
        dt: Time step

    Returns:
        Dictionary with trajectory and orbital data
    """
    r = np.array(initial_r, dtype=float)
    v = np.array(initial_v, dtype=float)

    positions = [r.copy()]
    velocities = [v.copy()]
    times = [0.0]

    # Track perihelion positions
    perihelion_positions = []
    perihelion_times = []
    prev_r_mag = np.linalg.norm(r)
    decreasing = True

    def acceleration(pos):
        """Calculate total acceleration including perturbation."""
        r_vec = pos
        r_mag = np.linalg.norm(r_vec)
        r_hat = r_vec / r_mag

        # Newtonian gravity
        a_newton = -G * m_central * r_hat / r_mag**2

        # 1/r^3 perturbation (mimics GR correction)
        # F = -alpha / r^4 * r_hat => a = -alpha / (m * r^4) * r_hat
        a_pert = -perturbation_strength * r_hat / r_mag**4

        return a_newton + a_pert

    t = 0
    while t < t_final:
        # RK4 integration
        k1_v = acceleration(r)
        k1_r = v

        k2_v = acceleration(r + 0.5 * dt * k1_r)
        k2_r = v + 0.5 * dt * k1_v

        k3_v = acceleration(r + 0.5 * dt * k2_r)
        k3_r = v + 0.5 * dt * k2_v

        k4_v = acceleration(r + dt * k3_r)
        k4_r = v + dt * k3_v

        v = v + (dt / 6) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        r = r + (dt / 6) * (k1_r + 2*k2_r + 2*k3_r + k4_r)

        t += dt
        positions.append(r.copy())
        velocities.append(v.copy())
        times.append(t)

        # Detect perihelion (minimum distance)
        r_mag = np.linalg.norm(r)
        if decreasing and r_mag > prev_r_mag:
            # We just passed perihelion
            perihelion_positions.append(positions[-2].copy())
            perihelion_times.append(times[-2])
            decreasing = False
        elif not decreasing and r_mag < prev_r_mag:
            decreasing = True
        prev_r_mag = r_mag

    return {
        'positions': np.array(positions),
        'velocities': np.array(velocities),
        'times': np.array(times),
        'perihelion_positions': np.array(perihelion_positions),
        'perihelion_times': np.array(perihelion_times)
    }


def calculate_precession_rate(perihelion_positions, perihelion_times):
    """Calculate precession rate from perihelion positions."""
    if len(perihelion_positions) < 2:
        return 0, []

    angles = []
    for pos in perihelion_positions:
        angle = np.arctan2(pos[1], pos[0])
        angles.append(angle)

    # Unwrap angles to get cumulative precession
    angles = np.unwrap(angles)

    # Precession per orbit
    precession_per_orbit = []
    for i in range(1, len(angles)):
        # Subtract 2*pi for one full orbit
        precession = angles[i] - angles[i-1] - 2*np.pi
        precession_per_orbit.append(precession)

    if len(precession_per_orbit) > 0:
        avg_precession = np.mean(precession_per_orbit)
    else:
        avg_precession = 0

    return avg_precession, angles


def main():
    # Normalized units for clarity
    G = 1.0
    M = 1.0  # Central mass
    m = 0.001  # Small orbiting mass

    # Initial conditions for elliptical orbit
    # Start at perihelion
    a = 1.0  # Semi-major axis
    e = 0.5  # Eccentricity
    r_p = a * (1 - e)  # Perihelion distance

    # Velocity at perihelion for ellipse
    v_p = np.sqrt(G * M * (1 + e) / (a * (1 - e)))

    initial_r = np.array([r_p, 0.0, 0.0])
    initial_v = np.array([0.0, v_p, 0.0])

    # Orbital period (for reference)
    T = 2 * np.pi * np.sqrt(a**3 / (G * M))

    # Simulate multiple cases with different perturbation strengths
    perturbation_cases = {
        'No perturbation': 0.0,
        'Weak perturbation': 0.005,
        'Medium perturbation': 0.02,
        'Strong perturbation': 0.05
    }

    fig = plt.figure(figsize=(16, 12))

    # Plot 1: Orbital trajectories
    ax1 = fig.add_subplot(2, 2, 1)
    colors = ['black', 'blue', 'green', 'red']

    results = {}
    for idx, (name, alpha) in enumerate(perturbation_cases.items()):
        result = simulate_precessing_orbit(
            m_central=M,
            m_orbiting=m,
            initial_r=initial_r,
            initial_v=initial_v,
            perturbation_strength=alpha,
            G=G,
            t_final=10 * T,  # 10 orbits
            dt=T / 500
        )
        results[name] = result

        pos = result['positions']
        ax1.plot(pos[:, 0], pos[:, 1], color=colors[idx], lw=0.8,
                 label=name, alpha=0.7)

        # Mark perihelion positions
        if len(result['perihelion_positions']) > 0:
            perih = result['perihelion_positions']
            ax1.plot(perih[:, 0], perih[:, 1], 'o', color=colors[idx],
                     markersize=6, alpha=0.8)

    ax1.plot(0, 0, 'ko', markersize=12)  # Central mass
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Orbital Trajectories with Precession\n(10 orbits, dots mark perihelia)')
    ax1.set_aspect('equal')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Perihelion angle vs orbit number
    ax2 = fig.add_subplot(2, 2, 2)

    for idx, (name, alpha) in enumerate(perturbation_cases.items()):
        if name == 'No perturbation':
            continue

        result = results[name]
        if len(result['perihelion_positions']) > 1:
            _, angles = calculate_precession_rate(
                result['perihelion_positions'],
                result['perihelion_times']
            )
            # Convert to degrees and show deviation from unperturbed
            angles_deg = np.degrees(angles)
            orbit_numbers = np.arange(len(angles))

            # Subtract the base angle (would be 2*pi*n for no precession)
            base_angles = orbit_numbers * 360  # degrees per orbit
            precession = angles_deg - base_angles

            ax2.plot(orbit_numbers, precession, 'o-', color=colors[idx],
                     label=f'alpha = {alpha}', markersize=6)

    ax2.set_xlabel('Orbit Number')
    ax2.set_ylabel('Cumulative Precession (degrees)')
    ax2.set_title('Perihelion Precession vs Orbit Number')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Precession rate vs perturbation strength
    ax3 = fig.add_subplot(2, 2, 3)

    alphas = np.linspace(0.001, 0.1, 20)
    precession_rates = []

    for alpha in alphas:
        result = simulate_precessing_orbit(
            m_central=M,
            m_orbiting=m,
            initial_r=initial_r,
            initial_v=initial_v,
            perturbation_strength=alpha,
            G=G,
            t_final=5 * T,
            dt=T / 300
        )
        rate, _ = calculate_precession_rate(
            result['perihelion_positions'],
            result['perihelion_times']
        )
        precession_rates.append(np.degrees(rate))

    ax3.plot(alphas, precession_rates, 'bo-', markersize=6)
    ax3.set_xlabel('Perturbation Strength (alpha)')
    ax3.set_ylabel('Precession Rate (degrees/orbit)')
    ax3.set_title('Precession Rate vs Perturbation Strength')
    ax3.grid(True, alpha=0.3)

    # Add theoretical prediction for weak perturbation
    # For 1/r^3 perturbation, precession per orbit is approximately:
    # delta_phi ~ 6*pi*alpha / (L^2) where L is specific angular momentum
    L = r_p * v_p  # Specific angular momentum
    theoretical_rate = 6 * np.pi * alphas / L**2
    ax3.plot(alphas, np.degrees(theoretical_rate), 'r--', lw=2,
             label='Weak-field theory')
    ax3.legend()

    # Plot 4: Energy and angular momentum conservation
    ax4 = fig.add_subplot(2, 2, 4)

    # Use medium perturbation case
    result = results['Medium perturbation']
    pos = result['positions']
    vel = result['velocities']
    times = result['times']

    # Calculate energy (including perturbation potential)
    energies = []
    ang_mom = []
    alpha = 0.02

    for i in range(len(pos)):
        r_mag = np.linalg.norm(pos[i])
        v_mag = np.linalg.norm(vel[i])

        # Kinetic energy
        KE = 0.5 * m * v_mag**2

        # Potential energy (Newtonian + perturbation)
        PE_newton = -G * M * m / r_mag
        PE_pert = -alpha * m / (3 * r_mag**3)  # Integrated from force

        E = KE + PE_newton + PE_pert
        energies.append(E)

        # Angular momentum
        L_vec = m * np.cross(pos[i], vel[i])
        ang_mom.append(np.linalg.norm(L_vec))

    energies = np.array(energies) / energies[0]
    ang_mom = np.array(ang_mom) / ang_mom[0]
    times_norm = times / T

    ax4.plot(times_norm, energies, 'b-', lw=1.5, label='Energy E/E_0')
    ax4.plot(times_norm, ang_mom, 'r--', lw=1.5, label='Angular Momentum |L|/|L_0|')
    ax4.axhline(y=1.0, color='black', linestyle=':', alpha=0.5)
    ax4.set_xlabel('Time / Period')
    ax4.set_ylabel('Normalized Value')
    ax4.set_title('Conservation Laws (alpha=0.02)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0.99, 1.01)

    plt.suptitle('Orbital Precession from 1/r^3 Perturbation (Toy GR Model)\n'
                 'Mimics perihelion advance from general relativistic corrections',
                 fontsize=12, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'precession_perturbation.png'),
                dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(output_dir, 'precession_perturbation.png')}")

    # Print precession analysis
    print("\nPrecession Analysis:")
    print("-" * 50)
    for name, alpha in perturbation_cases.items():
        if name == 'No perturbation':
            continue
        result = results[name]
        rate, _ = calculate_precession_rate(
            result['perihelion_positions'],
            result['perihelion_times']
        )
        print(f"{name} (alpha={alpha}):")
        print(f"  Precession rate: {np.degrees(rate):.4f} degrees/orbit")
        print(f"  Precession rate: {np.degrees(rate)*3600:.2f} arcsec/orbit")


if __name__ == "__main__":
    main()
