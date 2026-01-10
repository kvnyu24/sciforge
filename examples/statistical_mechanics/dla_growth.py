"""
Experiment 143: Diffusion-Limited Aggregation (DLA) Growth Patterns

This example demonstrates DLA, a process that creates fractal patterns
through random diffusion and aggregation.

DLA Algorithm:
1. Start with a seed particle at the origin
2. Release a random walker from far away
3. Walker performs random walk until it contacts the cluster
4. Walker sticks to cluster, becomes part of it
5. Repeat to grow the cluster

The resulting structure has fractal dimension D ≈ 1.71 in 2D.

DLA models various physical processes:
- Electrochemical deposition
- Bacterial colony growth
- Lightning patterns
- Mineral dendrites
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Fractal dimension of 2D DLA
D_FRACTAL = 1.71


def dla_simulation(n_particles, grid_size=500, seed=None):
    """
    Run DLA simulation.

    Args:
        n_particles: Number of particles to aggregate
        grid_size: Size of the grid
        seed: Random seed for reproducibility

    Returns:
        grid: Boolean array with cluster
        particle_times: Order in which particles were added
        radii: Radius of each added particle from origin
    """
    if seed is not None:
        np.random.seed(seed)

    # Grid to track occupied sites
    grid = np.zeros((grid_size, grid_size), dtype=bool)
    particle_times = np.zeros((grid_size, grid_size), dtype=int)

    center = grid_size // 2

    # Seed particle at center
    grid[center, center] = True
    particle_times[center, center] = 0

    # Track cluster radius
    max_radius = 1
    radii = [0]

    # Directions for random walk and neighbor check
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def has_neighbor(x, y):
        """Check if site has an occupied neighbor."""
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid_size and 0 <= ny < grid_size:
                if grid[nx, ny]:
                    return True
        return False

    for particle in range(1, n_particles):
        if particle % 500 == 0:
            print(f"  Particle {particle}/{n_particles}", end='\r')

        # Launch particle from circle around cluster
        launch_radius = max_radius + 10
        theta = np.random.uniform(0, 2 * np.pi)
        x = int(center + launch_radius * np.cos(theta))
        y = int(center + launch_radius * np.sin(theta))

        # Ensure within bounds
        x = max(1, min(grid_size - 2, x))
        y = max(1, min(grid_size - 2, y))

        # Random walk until sticking or escaping
        kill_radius = max_radius + 50
        stuck = False

        while not stuck:
            # Check if adjacent to cluster
            if has_neighbor(x, y):
                grid[x, y] = True
                particle_times[x, y] = particle
                r = np.sqrt((x - center)**2 + (y - center)**2)
                radii.append(r)
                max_radius = max(max_radius, r + 1)
                stuck = True
                break

            # Random step
            dx, dy = directions[np.random.randint(4)]
            x, y = x + dx, y + dy

            # Kill if too far
            if (x - center)**2 + (y - center)**2 > kill_radius**2:
                # Restart
                theta = np.random.uniform(0, 2 * np.pi)
                x = int(center + launch_radius * np.cos(theta))
                y = int(center + launch_radius * np.sin(theta))
                x = max(1, min(grid_size - 2, x))
                y = max(1, min(grid_size - 2, y))

            # Boundary check
            if x <= 0 or x >= grid_size - 1 or y <= 0 or y >= grid_size - 1:
                theta = np.random.uniform(0, 2 * np.pi)
                x = int(center + launch_radius * np.cos(theta))
                y = int(center + launch_radius * np.sin(theta))
                x = max(1, min(grid_size - 2, x))
                y = max(1, min(grid_size - 2, y))

    print(f"  Completed {n_particles} particles      ")
    return grid, particle_times, np.array(radii)


def compute_fractal_dimension(radii, n_particles_range=None):
    """
    Compute fractal dimension from N(r) ~ r^D relationship.

    The number of particles within radius r scales as:
    N(r) ~ r^D where D is the fractal dimension.
    """
    if n_particles_range is None:
        n_particles_range = np.arange(10, len(radii), 10)

    # For each N, find the maximum radius
    r_values = []
    n_values = []

    for n in n_particles_range:
        if n <= len(radii):
            r_max = np.max(radii[:n])
            if r_max > 0:
                r_values.append(r_max)
                n_values.append(n)

    r_values = np.array(r_values)
    n_values = np.array(n_values)

    # Fit log(N) vs log(R)
    log_r = np.log(r_values[r_values > 1])
    log_n = np.log(n_values[:len(log_r)])

    if len(log_r) > 2:
        from scipy.stats import linregress
        slope, intercept, r_value, _, _ = linregress(log_r, log_n)
        return slope, r_value**2, r_values, n_values
    return None, None, r_values, n_values


def compute_box_counting_dimension(grid, box_sizes=None):
    """
    Compute fractal dimension using box-counting method.

    N(epsilon) ~ epsilon^(-D)
    """
    if box_sizes is None:
        box_sizes = [2, 4, 8, 16, 32, 64]

    counts = []
    sizes = []

    for box_size in box_sizes:
        L = grid.shape[0]
        n_boxes = L // box_size
        count = 0

        for i in range(n_boxes):
            for j in range(n_boxes):
                region = grid[i*box_size:(i+1)*box_size,
                             j*box_size:(j+1)*box_size]
                if np.any(region):
                    count += 1

        if count > 0:
            counts.append(count)
            sizes.append(box_size)

    counts = np.array(counts)
    sizes = np.array(sizes)

    # Fit log(N) vs log(1/epsilon)
    from scipy.stats import linregress
    log_inv_eps = np.log(1.0 / sizes)
    log_n = np.log(counts)

    slope, intercept, r_value, _, _ = linregress(log_inv_eps, log_n)

    return slope, r_value**2, sizes, counts


def main():
    print("Diffusion-Limited Aggregation (DLA)")
    print("=" * 50)
    print(f"Expected fractal dimension D = {D_FRACTAL:.2f}")

    # Simulation parameters
    n_particles = 5000
    grid_size = 400

    print(f"\nGenerating cluster with {n_particles} particles...")
    grid, particle_times, radii = dla_simulation(n_particles, grid_size, seed=42)

    # Compute fractal dimensions
    print("\nComputing fractal dimension...")

    # From N(R) scaling
    D_nr, r2_nr, r_vals, n_vals = compute_fractal_dimension(radii)
    print(f"  From N(R) scaling: D = {D_nr:.3f} (R^2 = {r2_nr:.4f})")

    # From box counting
    D_box, r2_box, box_sizes, box_counts = compute_box_counting_dimension(grid)
    print(f"  From box counting: D = {D_box:.3f} (R^2 = {r2_box:.4f})")

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: DLA cluster
    ax1 = axes[0, 0]
    center = grid_size // 2
    extent = 150
    region = particle_times[center-extent:center+extent,
                           center-extent:center+extent]

    # Custom colormap for growth time
    cmap = plt.cm.hot_r.copy()
    cmap.set_under('white')
    im = ax1.imshow(region, cmap=cmap, vmin=1, origin='lower')
    plt.colorbar(im, ax=ax1, label='Particle order')
    ax1.set_title(f'DLA Cluster ({n_particles} particles)', fontsize=12)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)

    # Plot 2: Binary cluster view
    ax2 = axes[0, 1]
    ax2.imshow(grid[center-extent:center+extent, center-extent:center+extent],
               cmap='binary', origin='lower')
    ax2.set_title('DLA Cluster (Binary View)', fontsize=12)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)

    # Plot 3: N(R) scaling
    ax3 = axes[1, 0]
    ax3.loglog(r_vals, n_vals, 'bo', markersize=4, alpha=0.5, label='Data')

    # Fit line
    r_fit = np.linspace(r_vals.min(), r_vals.max(), 100)
    n_fit = r_fit**D_nr * n_vals[0] / r_vals[0]**D_nr
    ax3.loglog(r_fit, n_fit, 'r-', lw=2,
               label=f'Fit: N ~ R^{{{D_nr:.2f}}}')

    # Theoretical line
    ax3.loglog(r_fit, r_fit**D_FRACTAL * n_vals[0] / r_vals[0]**D_FRACTAL,
               'g--', lw=2, alpha=0.5, label=f'Theory: D = {D_FRACTAL}')

    ax3.set_xlabel('Cluster radius R', fontsize=12)
    ax3.set_ylabel('Number of particles N(R)', fontsize=12)
    ax3.set_title('Mass-Radius Scaling', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3, which='both')

    # Plot 4: Box counting
    ax4 = axes[1, 1]
    ax4.loglog(1.0/box_sizes, box_counts, 'gs-', markersize=8, label='Data')

    # Fit line
    eps_fit = np.linspace(1.0/box_sizes.max(), 1.0/box_sizes.min(), 100)
    n_box_fit = eps_fit**D_box * box_counts[-1] / (1.0/box_sizes[-1])**D_box
    ax4.loglog(eps_fit, n_box_fit, 'r-', lw=2,
               label=f'Fit: D = {D_box:.2f}')

    ax4.set_xlabel(r'$1/\epsilon$ (inverse box size)', fontsize=12)
    ax4.set_ylabel(r'$N(\epsilon)$ (box count)', fontsize=12)
    ax4.set_title('Box-Counting Dimension', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3, which='both')

    plt.suptitle('Diffusion-Limited Aggregation: Fractal Growth',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Additional analysis: radial density profile
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

    # Radial density
    ax5 = axes2[0]
    r_bins = np.linspace(0, radii.max(), 50)
    r_centers = 0.5 * (r_bins[1:] + r_bins[:-1])
    hist, _ = np.histogram(radii, bins=r_bins)

    # Density: particles per unit area in shell
    shell_area = np.pi * (r_bins[1:]**2 - r_bins[:-1]**2)
    density = hist / (shell_area + 1e-10)

    ax5.semilogy(r_centers, density, 'b-', lw=2)
    ax5.set_xlabel('Radius R', fontsize=12)
    ax5.set_ylabel('Particle density', fontsize=12)
    ax5.set_title('Radial Density Profile', fontsize=12)
    ax5.grid(True, alpha=0.3, which='both')

    # Growth dynamics
    ax6 = axes2[1]
    ax6.plot(np.arange(len(radii)), radii, 'b-', lw=0.5, alpha=0.7)
    ax6.set_xlabel('Particle number N', fontsize=12)
    ax6.set_ylabel('Cluster radius R', fontsize=12)
    ax6.set_title('Growth Dynamics: R(N)', fontsize=12)

    # Theoretical scaling: R ~ N^(1/D)
    n_theory = np.arange(10, len(radii))
    r_theory = n_theory**(1/D_FRACTAL) * radii[10] / 10**(1/D_FRACTAL)
    ax6.plot(n_theory, r_theory, 'r--', lw=2, alpha=0.7,
             label=f'$R \\sim N^{{1/D}}$, D = {D_FRACTAL}')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    # Print summary
    print("\n" + "=" * 50)
    print("DLA Analysis Summary")
    print("=" * 50)
    print(f"Number of particles: {n_particles}")
    print(f"Final cluster radius: {radii[-1]:.1f}")
    print(f"\nFractal dimension estimates:")
    print(f"  Mass-radius scaling: D = {D_nr:.3f}")
    print(f"  Box-counting: D = {D_box:.3f}")
    print(f"  Theoretical value: D = {D_FRACTAL:.2f}")

    # Save plots
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'dla_growth.png'),
                dpi=150, bbox_inches='tight')
    fig2.savefig(os.path.join(output_dir, 'dla_analysis.png'),
                 dpi=150, bbox_inches='tight')

    print(f"\nPlots saved to {output_dir}")


if __name__ == "__main__":
    main()
