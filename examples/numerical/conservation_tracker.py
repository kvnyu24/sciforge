"""
Experiment 10: Symmetry/invariant check harness - momentum/energy error trackers.

A general framework for tracking conserved quantities and detecting
numerical drift in physical simulations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


class ConservationTracker:
    """
    General framework for tracking conserved quantities.

    Can monitor energy, momentum, angular momentum, or any custom invariant.
    """

    def __init__(self, system_name="System"):
        self.system_name = system_name
        self.quantities = {}  # name -> (compute_func, values, initial_value)
        self.times = []
        self.tolerance_warnings = []

    def add_quantity(self, name, compute_func, tolerance=1e-6):
        """Add a conserved quantity to track."""
        self.quantities[name] = {
            'compute': compute_func,
            'values': [],
            'tolerance': tolerance,
            'initial': None
        }

    def record(self, t, state):
        """Record all quantities at time t."""
        self.times.append(t)
        for name, q in self.quantities.items():
            value = q['compute'](state)
            if q['initial'] is None:
                q['initial'] = value
            q['values'].append(value)

            # Check for tolerance violation
            if q['initial'] != 0:
                rel_error = abs(value - q['initial']) / abs(q['initial'])
                if rel_error > q['tolerance']:
                    self.tolerance_warnings.append({
                        'time': t,
                        'quantity': name,
                        'error': rel_error
                    })

    def get_errors(self, name, relative=True):
        """Get error history for a quantity."""
        q = self.quantities[name]
        values = np.array(q['values'])
        initial = q['initial']

        if relative and initial != 0:
            return (values - initial) / abs(initial)
        return values - initial

    def summary(self):
        """Generate summary statistics."""
        results = {}
        for name, q in self.quantities.items():
            values = np.array(q['values'])
            initial = q['initial']

            if initial != 0:
                rel_errors = (values - initial) / abs(initial)
                results[name] = {
                    'initial': initial,
                    'final': values[-1],
                    'max_error': np.max(np.abs(rel_errors)),
                    'mean_error': np.mean(np.abs(rel_errors)),
                    'drift_rate': rel_errors[-1] / self.times[-1] if self.times[-1] > 0 else 0
                }
            else:
                results[name] = {
                    'initial': initial,
                    'final': values[-1],
                    'max_error': np.max(np.abs(values)),
                    'mean_error': np.mean(np.abs(values)),
                    'drift_rate': 0
                }

        return results


def simulate_two_body(m1, m2, r1_0, v1_0, r2_0, v2_0, dt, n_steps, G=1.0):
    """Simulate gravitational two-body system."""
    r1, v1 = np.array(r1_0, dtype=float), np.array(v1_0, dtype=float)
    r2, v2 = np.array(r2_0, dtype=float), np.array(v2_0, dtype=float)

    def force(r1, r2, m1, m2, G):
        r = r2 - r1
        r_mag = np.linalg.norm(r)
        if r_mag < 1e-10:
            return np.zeros_like(r)
        return G * m1 * m2 * r / r_mag**3

    # Conservation trackers
    def kinetic_energy(state):
        r1, v1, r2, v2 = state
        return 0.5 * m1 * np.dot(v1, v1) + 0.5 * m2 * np.dot(v2, v2)

    def potential_energy(state):
        r1, v1, r2, v2 = state
        r = np.linalg.norm(r2 - r1)
        return -G * m1 * m2 / r if r > 1e-10 else 0

    def total_energy(state):
        return kinetic_energy(state) + potential_energy(state)

    def total_momentum(state):
        r1, v1, r2, v2 = state
        return m1 * v1 + m2 * v2

    def momentum_magnitude(state):
        return np.linalg.norm(total_momentum(state))

    def angular_momentum(state):
        r1, v1, r2, v2 = state
        L1 = m1 * np.cross(r1, v1)
        L2 = m2 * np.cross(r2, v2)
        return L1 + L2

    def angular_momentum_z(state):
        L = angular_momentum(state)
        return L if np.isscalar(L) else L[2] if len(L) > 2 else L

    def center_of_mass(state):
        r1, v1, r2, v2 = state
        return (m1 * r1 + m2 * r2) / (m1 + m2)

    def com_velocity(state):
        r1, v1, r2, v2 = state
        return np.linalg.norm((m1 * v1 + m2 * v2) / (m1 + m2))

    # Initialize tracker
    tracker = ConservationTracker("Two-Body Gravity")
    tracker.add_quantity("Total Energy", total_energy, tolerance=1e-8)
    tracker.add_quantity("Momentum |P|", momentum_magnitude, tolerance=1e-10)
    tracker.add_quantity("Angular Momentum Lz", angular_momentum_z, tolerance=1e-10)
    tracker.add_quantity("CoM Velocity", com_velocity, tolerance=1e-10)

    # Storage
    r1s, r2s = [r1.copy()], [r2.copy()]

    # Initial record
    state = (r1, v1, r2, v2)
    tracker.record(0, state)

    # Velocity Verlet integration
    for i in range(n_steps):
        # Accelerations
        F = force(r1, r2, m1, m2, G)
        a1 = F / m1
        a2 = -F / m2

        # Position update
        r1_new = r1 + v1 * dt + 0.5 * a1 * dt**2
        r2_new = r2 + v2 * dt + 0.5 * a2 * dt**2

        # New accelerations
        F_new = force(r1_new, r2_new, m1, m2, G)
        a1_new = F_new / m1
        a2_new = -F_new / m2

        # Velocity update
        v1_new = v1 + 0.5 * (a1 + a1_new) * dt
        v2_new = v2 + 0.5 * (a2 + a2_new) * dt

        r1, v1 = r1_new, v1_new
        r2, v2 = r2_new, v2_new

        r1s.append(r1.copy())
        r2s.append(r2.copy())

        state = (r1, v1, r2, v2)
        tracker.record((i + 1) * dt, state)

    return np.array(r1s), np.array(r2s), tracker


def main():
    # Two-body system (unequal masses)
    m1, m2 = 1.0, 0.5
    G = 1.0

    # Initial conditions (elliptical orbit)
    r1_0 = np.array([-0.25, 0.0, 0.0])
    r2_0 = np.array([0.5, 0.0, 0.0])
    v1_0 = np.array([0.0, -0.3, 0.0])
    v2_0 = np.array([0.0, 0.6, 0.0])

    dt = 0.01
    n_steps = 10000

    print("Simulating two-body system...")
    r1s, r2s, tracker = simulate_two_body(m1, m2, r1_0, v1_0, r2_0, v2_0, dt, n_steps, G)

    # Get summary
    summary = tracker.summary()

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    t = np.array(tracker.times)

    # Plot 1: Orbits
    ax = axes[0, 0]
    ax.plot(r1s[:, 0], r1s[:, 1], 'b-', lw=0.5, alpha=0.7, label=f'm₁ = {m1}')
    ax.plot(r2s[:, 0], r2s[:, 1], 'r-', lw=0.5, alpha=0.7, label=f'm₂ = {m2}')

    # Center of mass path
    com = (m1 * r1s + m2 * r2s) / (m1 + m2)
    ax.plot(com[:, 0], com[:, 1], 'k--', lw=1, alpha=0.5, label='CoM')

    ax.plot(r1s[0, 0], r1s[0, 1], 'bo', markersize=8)
    ax.plot(r2s[0, 0], r2s[0, 1], 'ro', markersize=8)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Two-Body Orbits')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Plot 2: Energy error
    ax = axes[0, 1]
    dE = tracker.get_errors("Total Energy", relative=True)
    ax.plot(t, dE * 100, 'b-', lw=1)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Energy error (%)')
    ax.set_title(f'Energy Conservation (max error: {summary["Total Energy"]["max_error"]*100:.2e}%)')
    ax.grid(True, alpha=0.3)

    # Plot 3: Momentum error
    ax = axes[1, 0]
    dP = tracker.get_errors("Momentum |P|", relative=True)
    dL = tracker.get_errors("Angular Momentum Lz", relative=True)

    ax.semilogy(t, np.abs(dP) + 1e-16, 'b-', lw=1, label='|P| error')
    ax.semilogy(t, np.abs(dL) + 1e-16, 'r-', lw=1, label='Lz error')

    ax.set_xlabel('Time')
    ax.set_ylabel('Relative error')
    ax.set_title('Momentum & Angular Momentum Conservation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Summary table
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = f"""Conservation Tracker Summary
============================
System: Two-Body Gravitational
Integration: Velocity Verlet
Time step: dt = {dt}
Steps: {n_steps}
Total time: {t[-1]:.1f}

Conserved Quantities:
---------------------
"""

    for name, stats in summary.items():
        summary_text += f"""
{name}:
  Initial:    {stats['initial']:.6e}
  Final:      {stats['final']:.6e}
  Max error:  {stats['max_error']:.2e}
  Mean error: {stats['mean_error']:.2e}
  Drift rate: {stats['drift_rate']:.2e}/t"""

    if tracker.tolerance_warnings:
        summary_text += f"\n\nWarnings: {len(tracker.tolerance_warnings)} tolerance violations"
    else:
        summary_text += "\n\nNo tolerance violations detected."

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    plt.suptitle('Conservation Tracker: Two-Body System Invariants',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'conservation_tracker.png'), dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/conservation_tracker.png")


if __name__ == "__main__":
    main()
