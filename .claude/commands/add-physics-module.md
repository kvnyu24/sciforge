# Add Physics Module

Create a new physics module following SciForge conventions.

## Usage

```
/add-physics-module <module-name>
```

## Steps

1. **Create the module file** at `src/sciforge/physics/<module_name>.py`

2. **Use this template**:

```python
"""
<Module description>

This module implements <physics domain> calculations and simulations.
"""

import numpy as np
from typing import Optional, Tuple, List
from numpy.typing import ArrayLike

from .base import PhysicalSystem, DynamicalSystem
from ..core.exceptions import ValidationError, PhysicsError
from ..core.utils import validate_positive, validate_array


class <ClassName>(DynamicalSystem):
    """
    <Brief description of the physical system>

    This class models <detailed physics description>.

    Attributes:
        mass: Mass of the system in kg
        position: Position vector [x, y, z] in meters
        velocity: Velocity vector [vx, vy, vz] in m/s
        <additional attributes>

    Examples:
        >>> system = <ClassName>(mass=1.0, position=[0, 0, 0], velocity=[1, 0, 0])
        >>> system.update(dt=0.01)
        >>> print(system.position)
    """

    def __init__(
        self,
        mass: float,
        position: ArrayLike,
        velocity: ArrayLike,
        <additional_params>
    ):
        """
        Initialize the system.

        Args:
            mass: Mass in kg (must be positive)
            position: Initial position [x, y, z] in meters
            velocity: Initial velocity [vx, vy, vz] in m/s
            <additional args>

        Raises:
            ValidationError: If parameters are invalid
        """
        # Validate inputs
        validate_positive(mass, "mass")
        validate_array(position, expected_shape=(3,), name="position")
        validate_array(velocity, expected_shape=(3,), name="velocity")

        super().__init__(mass, position, velocity)

        # Initialize history tracking
        self.history = {
            'time': [],
            'position': [],
            'velocity': [],
            'energy': []
        }

    def update(self, dt: float) -> None:
        """
        Update system state using RK4 integration.

        Args:
            dt: Time step in seconds
        """
        # Record state before update
        self._record_state()

        # RK4 integration
        # ... implementation

        self.time += dt

    def energy(self) -> float:
        """
        Calculate total energy of the system.

        Returns:
            Total energy in Joules
        """
        kinetic = 0.5 * self.mass * np.dot(self.velocity, self.velocity)
        potential = self._potential_energy()
        return kinetic + potential

    def _potential_energy(self) -> float:
        """Calculate potential energy (override in subclasses)"""
        return 0.0

    def _record_state(self) -> None:
        """Record current state to history"""
        self.history['time'].append(self.time)
        self.history['position'].append(self.position.copy())
        self.history['velocity'].append(self.velocity.copy())
        self.history['energy'].append(self.energy())
```

3. **Export in `physics/__init__.py`**:

```python
from .<module_name> import <ClassName>
```

4. **Add to `__all__`**:

```python
__all__ = [
    # ... existing exports
    '<ClassName>',
]
```

5. **Create tests** at `tests/test_physics/test_<module_name>.py`

6. **Create example** at `examples/<domain>/<example_name>.py`

## Checklist

- [ ] Module file created with proper docstrings
- [ ] Inherits from appropriate base class
- [ ] Input validation in `__init__`
- [ ] History tracking implemented
- [ ] `update()` method uses RK4 or appropriate integrator
- [ ] `energy()` method implemented
- [ ] Exported in `physics/__init__.py`
- [ ] Unit tests written
- [ ] Example script created
