# Create Physics Experiment

Create a new physics experiment that demonstrates a phenomenon using implemented SciForge primitives.

## Usage

```
/create-experiment <phenomenon-name>
```

For example:
```
/create-experiment doppler-effect
/create-experiment simple-pendulum
/create-experiment rc-circuit-decay
```

## Available Physics Primitives

Before creating an experiment, consider which primitives are available:

### Mechanics
- `Particle` - Point mass with position, velocity, forces
- `DynamicalSystem` - Base class for systems with equations of motion
- `RotationalSystem` - Rotational dynamics
- `Constraint` - Mechanical constraints
- `RotationalSpring` - Torsional springs

### Oscillations
- `HarmonicOscillator` - Simple harmonic motion
- `CoupledOscillator` - Coupled oscillator systems
- `ParametricOscillator` - Parametrically driven oscillators

### Fields
- `ElectricField` - Electric field calculations
- `MagneticField` - Magnetic field calculations
- `GravitationalField` - Gravitational field calculations

### Waves
- `Wave` - General wave mechanics
- `WavePacket` - Wave packet propagation
- `ElectromagneticWave` - EM wave propagation

### Thermodynamics
- `ThermalSystem` - Heat transfer and thermal dynamics

### Quantum
- `Wavefunction` - Quantum mechanical wavefunctions

### Fluids
- `FluidColumn` - Fluid dynamics
- `FluidJet` - Jet flow simulations

### Circuits
- `Circuit`, `Resistor`, `Capacitor`, `Inductor` - Circuit elements

### Optics
- `AttosecondPulseGenerator` - Strong-field optics
- `StrongFieldSystem` - High-intensity laser interactions

### Relativity
- `LorentzTransform` - Special relativity transformations

### Statistical
- `HermitePolynomial` - Hermite polynomial calculations
- `RiceDistribution` - Rice distribution sampling

## Steps to Create an Experiment

1. **Identify the phenomenon** to demonstrate (e.g., resonance, interference, decay)

2. **Select appropriate primitives** from the list above

3. **Create the experiment file** at `examples/<domain>/<experiment_name>.py`

4. **Use this template**:

```python
"""
Example demonstrating <phenomenon name>.

This example shows <brief description of the physics being demonstrated>.
The simulation includes:
- <Key feature 1>
- <Key feature 2>
- <Key feature 3>
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.sciforge.physics import <PrimitiveClass1>, <PrimitiveClass2>


def setup_experiment():
    """
    Set up the physical system for the experiment.

    Returns:
        Configured physics objects ready for simulation
    """
    # Initialize physics primitives with realistic parameters
    system = <PrimitiveClass>(
        # ... parameters with physical units in comments
    )
    return system


def run_simulation(system, dt: float, t_final: float):
    """
    Run the simulation and collect data.

    Args:
        system: The physical system to simulate
        dt: Time step in seconds
        t_final: Total simulation time in seconds

    Returns:
        Dictionary containing simulation results
    """
    results = {
        'time': [],
        'observable1': [],
        'observable2': [],
    }

    t = 0
    while t < t_final:
        # Record observables
        results['time'].append(t)
        results['observable1'].append(system.some_property)

        # Update system
        system.update(dt)
        t += dt

    # Convert to numpy arrays
    for key in results:
        results[key] = np.array(results[key])

    return results


def plot_results(results):
    """
    Create publication-quality plots of the phenomenon.

    Args:
        results: Dictionary of simulation results
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Primary observable vs time
    ax1 = axes[0, 0]
    ax1.plot(results['time'], results['observable1'], 'b-', lw=2)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Observable 1 (units)')
    ax1.set_title('Primary Observable')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Secondary observable or phase space
    ax2 = axes[0, 1]
    # ... second plot

    # Plot 3: Energy or conservation law verification
    ax3 = axes[1, 0]
    # ... third plot

    # Plot 4: Key physics insight (e.g., frequency spectrum, spatial distribution)
    ax4 = axes[1, 1]
    # ... fourth plot

    plt.suptitle('<Phenomenon Name> Demonstration', fontsize=14)
    plt.tight_layout()


def main():
    # Setup experiment
    system = setup_experiment()

    # Run simulation
    results = run_simulation(system, dt=0.001, t_final=10.0)

    # Plot results
    plot_results(results)

    # Save plot to output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, '<experiment_name>.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    main()
```

5. **Run the experiment** to verify it works:
```bash
python examples/<domain>/<experiment_name>.py
```

6. **Verify the output** in `examples/output/<experiment_name>.png`

## Example Phenomena to Demonstrate

### Mechanics
- Projectile motion with air resistance
- Coupled pendulums and normal modes
- Driven harmonic oscillator with resonance
- Gyroscopic precession

### Waves
- Standing waves and harmonics
- Wave interference and diffraction
- Doppler effect
- Beat frequencies

### Electromagnetism
- Charged particle in magnetic field (cyclotron motion)
- LC circuit oscillations
- RLC circuit resonance
- Electromagnetic induction

### Thermodynamics
- Newton's law of cooling
- Heat conduction in rods
- Thermal equilibration

### Quantum
- Quantum tunneling
- Particle in a box energy levels
- Wavefunction collapse simulation

### Fluids
- Bernoulli's principle demonstration
- Viscous flow and drag
- Surface tension effects

## Checklist

- [ ] Experiment file created with proper docstrings
- [ ] Uses existing SciForge primitives (no new physics classes needed)
- [ ] Simulation parameters are physically realistic
- [ ] Plots clearly demonstrate the phenomenon
- [ ] Axes labeled with proper units
- [ ] Output saved to `examples/output/` directory
- [ ] Script runs without errors
- [ ] Output plot is visually clear and informative

## Tips

1. **Choose appropriate time scales** - dt should be small enough to resolve the fastest dynamics
2. **Use physical units** - Keep everything in SI units, add unit comments
3. **Verify conservation laws** - Plot energy or momentum to validate simulation
4. **Add analytical comparison** - If available, overlay theoretical predictions
5. **Use descriptive variable names** - Make the physics clear in the code
