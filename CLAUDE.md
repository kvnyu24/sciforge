# CLAUDE.md - SciForge Development Guide

This file provides guidance for Claude Code and other AI assistants working on the SciForge codebase.

## Project Overview

SciForge is a Python scientific computing library built on NumPy that implements physics simulations, stochastic processes, and numerical methods. The project targets researchers, students, and developers who need robust scientific computing tools.

## Architecture

### Directory Structure

```
sciforge/
├── src/sciforge/           # Main package source
│   ├── core/               # Base classes, constants, utilities
│   │   ├── base.py         # BaseClass, BaseSolver, BaseProcess
│   │   ├── constants.py    # Physical and mathematical constants
│   │   ├── utils.py        # Utility functions
│   │   └── exceptions.py   # Custom exception hierarchy
│   ├── physics/            # Physics simulations (largest module)
│   │   ├── base.py         # PhysicalSystem, DynamicalSystem, Field, etc.
│   │   ├── mechanics.py    # Particle, RigidBody dynamics
│   │   ├── fields.py       # Electric, Magnetic, Gravitational fields
│   │   ├── waves.py        # Wave mechanics
│   │   ├── quantum.py      # Quantum systems
│   │   ├── thermodynamics.py
│   │   ├── fluids.py
│   │   ├── oscillations.py
│   │   ├── relativity.py
│   │   └── ...
│   ├── numerical/          # Numerical algorithms
│   │   ├── integration.py  # Trapezoid, Simpson's rule
│   │   ├── optimization.py # Newton, gradient descent
│   │   ├── root_finding.py
│   │   └── interpolation.py
│   ├── stochastic/         # Stochastic processes
│   │   └── processes.py    # Poisson, Wiener, OU, GBM, etc.
│   ├── differential/       # ODE solvers
│   │   └── ode.py          # Euler, RK2, RK4, adaptive methods
│   └── chaos/              # Fractals and chaos
│       └── fractals.py     # Mandelbrot, Julia sets (Numba-accelerated)
├── tests/                  # Test suite (pytest)
├── examples/               # Working example scripts by domain
├── docs/                   # Sphinx documentation
└── pyproject.toml          # Project configuration
```

### Class Hierarchy

```
BaseClass (core/base.py)
├── BaseSolver              # For numerical solvers
├── BaseProcess             # For stochastic processes
└── PhysicalSystem (physics/base.py)
    ├── DynamicalSystem     # Systems with velocity/forces
    │   ├── Particle
    │   ├── RigidBody
    │   └── Pendulum
    ├── QuantumSystem       # Quantum mechanics
    ├── ThermodynamicSystem # Heat transfer
    └── Field               # Force fields
        └── ConservativeField
```

### Key Design Patterns

1. **History Tracking**: All classes inheriting from `BaseClass` have `_history` dict for state tracking
2. **Composable Forces**: `DynamicalSystem` accepts list of force callables
3. **RK4 Integration**: Standard 4th-order Runge-Kutta used for ODE solving
4. **Type Hints**: 97% of functions have type annotations using `numpy.typing.ArrayLike`

## Development Commands

```bash
# Install in development mode
pip install -e .

# Run tests
pytest tests/ -v

# Run specific test file
pytest tests/test_physics/test_fluids.py -v

# Run with coverage
pytest --cov=sciforge tests/

# Format code
black src/ tests/

# Type checking
mypy src/sciforge/

# Build documentation
cd docs && make html
```

## Coding Standards

### Style Guidelines

- **Formatter**: Black with 88-character line length
- **Type Hints**: Required for all public functions
- **Docstrings**: NumPy-style docstrings with Args, Returns, Raises sections
- **Imports**: Standard library → third-party → local, each group alphabetized

### Example Function Template

```python
def calculate_field(
    position: ArrayLike,
    charge: float,
    epsilon_0: float = 8.854e-12
) -> np.ndarray:
    """
    Calculate electric field at a position due to a point charge.

    Args:
        position: 3D position vector [x, y, z] in meters
        charge: Point charge in Coulombs
        epsilon_0: Permittivity of free space (default: vacuum)

    Returns:
        Electric field vector [Ex, Ey, Ez] in V/m

    Raises:
        ValueError: If position is at the charge location (r=0)
        ValidationError: If charge is not a finite number

    Examples:
        >>> field = calculate_field([1, 0, 0], 1e-9)
        >>> np.linalg.norm(field)  # Field magnitude at 1m
        8.99...
    """
    position = np.asarray(position)
    validate_finite(charge, "charge")

    r = np.linalg.norm(position)
    if r < 1e-15:
        raise ValueError("Cannot calculate field at charge location")

    k = 1 / (4 * np.pi * epsilon_0)
    return k * charge * position / r**3
```

### Validation Pattern

Always validate physical parameters at class construction:

```python
class Particle(DynamicalSystem):
    def __init__(self, mass: float, position: ArrayLike, velocity: ArrayLike):
        validate_positive(mass, "mass")
        validate_array(position, expected_shape=(3,), name="position")
        validate_array(velocity, expected_shape=(3,), name="velocity")
        super().__init__(mass, position, velocity)
```

### Error Handling

Use the custom exception hierarchy:

```python
from sciforge.core.exceptions import (
    SciForgeError,      # Base exception
    ValidationError,    # Invalid parameters
    ConvergenceError,   # Solver didn't converge
    PhysicsError,       # Physical constraint violated
)
```

## Testing Guidelines

### Test Structure

```python
# tests/test_physics/test_mechanics.py

import pytest
import numpy as np
from sciforge.physics import Particle

class TestParticle:
    """Tests for Particle class"""

    def test_initialization(self):
        """Test particle can be created with valid parameters"""
        p = Particle(mass=1.0, position=[0, 0, 0], velocity=[1, 0, 0])
        assert p.mass == 1.0
        np.testing.assert_array_equal(p.position, [0, 0, 0])

    def test_invalid_mass_raises(self):
        """Test that negative mass raises ValidationError"""
        with pytest.raises(ValidationError):
            Particle(mass=-1.0, position=[0, 0, 0], velocity=[0, 0, 0])

    def test_energy_conservation(self):
        """Test that energy is conserved in free particle motion"""
        p = Particle(mass=1.0, position=[0, 0, 0], velocity=[1, 0, 0])
        initial_energy = p.kinetic_energy()

        for _ in range(100):
            p.update(dt=0.01)

        np.testing.assert_allclose(p.kinetic_energy(), initial_energy, rtol=1e-10)
```

### Test Categories

- **Unit tests**: Test individual functions/methods
- **Integration tests**: Test class interactions
- **Physics tests**: Verify conservation laws and known solutions
- **Numerical tests**: Check convergence and accuracy

## Common Tasks

### Adding a New Physics Module

1. Create file in `src/sciforge/physics/`
2. Inherit from appropriate base class (`PhysicalSystem`, `DynamicalSystem`, etc.)
3. Implement required methods (`update()`, `energy()`, etc.)
4. Add validation in `__init__`
5. Export in `physics/__init__.py`
6. Add tests in `tests/test_physics/`
7. Add example in `examples/`

### Adding a New Numerical Method

1. Create or extend file in `src/sciforge/numerical/`
2. Inherit from `BaseSolver` if applicable
3. Implement `solve()`, `validate_inputs()`, `estimate_error()`
4. Add comprehensive docstring with mathematical background
5. Add tests comparing to known analytical solutions

### Performance Optimization

- Use NumPy vectorization over Python loops
- Consider Numba JIT for computationally intensive functions (see `chaos/fractals.py`)
- Profile with `cProfile` before optimizing
- Add `@functools.lru_cache` for pure functions with repeated calls

## Known Issues and TODOs

- [ ] Two `HarmonicOscillator` classes exist (classical and quantum) - consider namespacing
- [ ] RK4 integration code duplicated in multiple classes - should use shared solver
- [ ] Test coverage is low (~2%) - priority to increase
- [ ] Some modules lack comprehensive error handling

## Dependencies

- **numpy** (>=1.20.0): Core numerical operations
- **scipy** (>=1.7.0): Advanced algorithms
- **matplotlib** (>=3.4.0): Visualization
- **numba** (>=0.55.0): JIT compilation for performance-critical code

## Versioning

This project follows [Semantic Versioning](https://semver.org/):
- MAJOR: Breaking API changes
- MINOR: New features, backward compatible
- PATCH: Bug fixes, backward compatible

Current version: 0.1.0 (initial development)
