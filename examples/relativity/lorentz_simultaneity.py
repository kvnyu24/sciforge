"""
Experiment 184: Lorentz Transformation and Relativity of Simultaneity

This experiment demonstrates how the Lorentz transformation affects
simultaneity - events that are simultaneous in one reference frame
may not be simultaneous in another.

Physical concepts:
- Lorentz transformations between inertial frames
- Relativity of simultaneity
- Spacetime diagrams and worldlines
- Light cone structure
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics.relativity import LorentzTransform, FourVector


def lorentz_transform_event(t, x, v, c=1.0):
    """
    Apply Lorentz transformation to an event (t, x).

    Args:
        t: Time in original frame
        x: Position in original frame
        v: Relative velocity of new frame
        c: Speed of light

    Returns:
        (t', x') in new frame
    """
    gamma = 1 / np.sqrt(1 - (v/c)**2)
    t_prime = gamma * (t - v * x / c**2)
    x_prime = gamma * (x - v * t)
    return t_prime, x_prime


def main():
    c = 1.0  # Speed of light (natural units)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ==========================================================================
    # Plot 1: Spacetime diagram showing relativity of simultaneity
    # ==========================================================================
    ax1 = axes[0, 0]

    # Two events that are simultaneous in frame S (at t=0)
    # Event A at x=-1, Event B at x=+1
    events_S = [
        {'name': 'A', 't': 0, 'x': -1, 'color': 'red'},
        {'name': 'B', 't': 0, 'x': 1, 'color': 'blue'},
    ]

    # Draw worldlines for objects at rest in S
    t_range = np.linspace(-1.5, 1.5, 100)
    ax1.plot([-1]*len(t_range), t_range, 'r--', alpha=0.3, lw=1)
    ax1.plot([1]*len(t_range), t_range, 'b--', alpha=0.3, lw=1)

    # Draw light cones from origin
    ax1.plot(t_range, t_range, 'y-', lw=2, alpha=0.5, label='Light cone')
    ax1.plot(-t_range, t_range, 'y-', lw=2, alpha=0.5)
    ax1.fill_between(t_range, t_range, -t_range, alpha=0.1, color='yellow')

    # Plot events
    for e in events_S:
        ax1.plot(e['x'], e['t'], 'o', color=e['color'], markersize=12,
                label=f"Event {e['name']}: (t={e['t']}, x={e['x']})")

    # Draw line of simultaneity in S (horizontal line at t=0)
    ax1.axhline(y=0, color='green', linestyle='-', lw=2,
               label="Simultaneity line (S frame)")

    # Draw line of simultaneity for moving frame S' (v=0.5c)
    v = 0.5 * c
    gamma = 1 / np.sqrt(1 - v**2)
    x_range = np.linspace(-2, 2, 100)
    t_simult_Sprime = v * x_range / c**2  # t' = 0 line in S coordinates
    ax1.plot(x_range, t_simult_Sprime, 'm--', lw=2,
            label=f"Simultaneity line (S' frame, v={v}c)")

    ax1.set_xlabel('Position x')
    ax1.set_ylabel('Time t')
    ax1.set_title('Spacetime Diagram: Relativity of Simultaneity')
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-1.5, 1.5)
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Add annotation
    ax1.annotate('Events A and B are\nsimultaneous in S\nbut NOT in S\'',
                xy=(0, 0.3), fontsize=9, ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ==========================================================================
    # Plot 2: Transform events to different frames
    # ==========================================================================
    ax2 = axes[0, 1]

    velocities = np.linspace(-0.9, 0.9, 50) * c

    # Track time difference between events A and B in each frame
    delta_t_AB = []

    for v in velocities:
        t_A_prime, _ = lorentz_transform_event(0, -1, v, c)
        t_B_prime, _ = lorentz_transform_event(0, 1, v, c)
        delta_t_AB.append(t_B_prime - t_A_prime)

    ax2.plot(velocities/c, delta_t_AB, 'b-', lw=2)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    # Mark special points
    ax2.plot(0, 0, 'go', markersize=10, label='S frame (v=0): simultaneous')
    ax2.plot(0.5, delta_t_AB[int(len(velocities)*0.75)], 'ro', markersize=10,
            label=f'v=0.5c: B before A')
    ax2.plot(-0.5, delta_t_AB[int(len(velocities)*0.25)], 'mo', markersize=10,
            label=f'v=-0.5c: A before B')

    ax2.set_xlabel('Observer velocity v/c')
    ax2.set_ylabel("Time difference t'_B - t'_A")
    ax2.set_title('Time Ordering Depends on Observer')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Add regions
    ax2.fill_between(velocities/c, delta_t_AB, 0,
                    where=np.array(delta_t_AB) > 0, alpha=0.3, color='blue',
                    label='B occurs after A')
    ax2.fill_between(velocities/c, delta_t_AB, 0,
                    where=np.array(delta_t_AB) < 0, alpha=0.3, color='red')

    # ==========================================================================
    # Plot 3: Train-platform thought experiment
    # ==========================================================================
    ax3 = axes[1, 0]

    # Classic Einstein train thought experiment
    # Lightning strikes at front and back of train simultaneously in platform frame
    L = 1.0  # Train half-length
    v_train = 0.6 * c

    # In platform frame: lightning at t=0, x = -L and x = +L
    # Light from each strike travels to observer at center of train

    # Time for light to reach center (in platform frame)
    # Observer moves with train, so:
    # From rear: c * t = L - v * t => t_rear = L / (c + v)
    # From front: c * t = L + v * t => t_front = L / (c - v)
    t_rear = L / (c + v_train)
    t_front = L / (c - v_train)

    # Transform to train frame
    gamma = 1 / np.sqrt(1 - (v_train/c)**2)

    # In train frame, the observer receives light from rear first
    time_points = np.linspace(0, 2, 100)

    # Platform frame view
    ax3.axhline(y=0.7, color='gray', linestyle='-', lw=10, alpha=0.3)
    ax3.text(-1.5, 0.7, 'Platform', fontsize=10, va='center')

    # Train (moving right)
    train_x = np.array([-L, L])
    ax3.plot(train_x, [0.3, 0.3], 'b-', lw=15, alpha=0.5)
    ax3.text(0, 0.3, 'Train', fontsize=10, ha='center', va='center', color='white')

    # Lightning strikes
    ax3.plot([-L, L], [0.5, 0.5], 'y*', markersize=20, label='Lightning (simultaneous in platform)')

    # Light paths
    ax3.annotate('', xy=(0, 0.3), xytext=(-L, 0.5),
                arrowprops=dict(arrowstyle='->', color='yellow', lw=2))
    ax3.annotate('', xy=(0, 0.3), xytext=(L, 0.5),
                arrowprops=dict(arrowstyle='->', color='yellow', lw=2))

    ax3.set_xlim(-2, 2)
    ax3.set_ylim(0, 1)
    ax3.set_title("Einstein's Train: Lightning Thought Experiment")
    ax3.axis('off')

    # Add explanation
    explanation = (
        f"Train velocity: v = {v_train}c\n"
        f"Platform frame: Lightning simultaneous at t=0\n"
        f"Light from rear arrives: t = {t_rear:.3f}\n"
        f"Light from front arrives: t = {t_front:.3f}\n\n"
        f"Train observer sees rear strike FIRST!"
    )
    ax3.text(0, 0.1, explanation, fontsize=10, ha='center', va='bottom',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # ==========================================================================
    # Plot 4: Minkowski diagram with multiple frames
    # ==========================================================================
    ax4 = axes[1, 1]

    # Draw coordinate axes for frame S
    ax4.axhline(y=0, color='black', lw=1.5)
    ax4.axvline(x=0, color='black', lw=1.5)
    ax4.text(2.2, 0, 'x', fontsize=12)
    ax4.text(0, 2.2, 'ct', fontsize=12)

    # Draw axes for frame S' (v = 0.4c)
    v = 0.4 * c
    beta = v / c

    # x' axis (t'=0) has slope beta in (x, ct) coordinates
    # ct' axis (x'=0) has slope 1/beta
    x_range = np.linspace(-2, 2, 100)

    ax4.plot(x_range, beta * x_range, 'b-', lw=1.5, label=f"x' axis (v={beta}c)")
    ax4.plot(beta * x_range, x_range, 'b--', lw=1.5, label="ct' axis")

    # Draw light cone
    ax4.plot(x_range, x_range, 'y-', lw=2, alpha=0.7)
    ax4.plot(x_range, -x_range, 'y-', lw=2, alpha=0.7)

    # Mark events
    events = [
        {'x': 1.0, 't': 0.5, 'label': 'P', 'color': 'red'},
        {'x': -0.5, 't': 1.0, 'label': 'Q', 'color': 'green'},
        {'x': 0.8, 't': 1.2, 'label': 'R', 'color': 'purple'},
    ]

    for e in events:
        ax4.plot(e['x'], e['t'], 'o', color=e['color'], markersize=10)
        ax4.text(e['x'] + 0.1, e['t'] + 0.1, e['label'], fontsize=12, color=e['color'])

        # Show coordinates in S'
        t_prime, x_prime = lorentz_transform_event(e['t']/c, e['x'], v, c)
        print(f"Event {e['label']}: S=({e['x']:.2f}, {e['t']:.2f}) -> "
              f"S'=({x_prime:.2f}, {t_prime*c:.2f})")

    # Draw grid lines for S' frame
    for i in [-1, -0.5, 0.5, 1]:
        # Lines of constant t'
        ax4.plot(x_range, beta * x_range + i * np.sqrt(1 - beta**2),
                'b:', alpha=0.3, lw=0.5)
        # Lines of constant x'
        ax4.plot(beta * x_range + i * np.sqrt(1 - beta**2), x_range,
                'b:', alpha=0.3, lw=0.5)

    ax4.set_xlim(-2, 2.5)
    ax4.set_ylim(-0.5, 2.5)
    ax4.set_xlabel('x')
    ax4.set_ylabel('ct')
    ax4.set_title('Minkowski Diagram: Two Reference Frames')
    ax4.legend(loc='upper left', fontsize=9)
    ax4.grid(True, alpha=0.2)
    ax4.set_aspect('equal')

    plt.suptitle('Lorentz Transformations and Relativity of Simultaneity\n'
                 "t' = gamma(t - vx/c^2), x' = gamma(x - vt)",
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'lorentz_simultaneity.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    main()
