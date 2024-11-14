import pytest
import numpy as np
from sciforge.physics.fluids import FluidColumn

def test_fluid_column_initialization():
    column = FluidColumn(
        radius=0.001,  # 1mm
        length=0.01,   # 1cm
        density=1000,  # water
        surface_tension=0.072,  # water
        viscosity=0.001,  # water
        n_points=100
    )
    
    assert column.r.shape == (100,)
    assert np.allclose(column.r, 0.001)
    assert np.allclose(column.v_r, 0)

def test_plateau_rayleigh_growth_rate():
    column = FluidColumn(
        radius=0.001,
        length=0.01,
        density=1000,
        surface_tension=0.072,
        viscosity=0.001
    )
    
    # Theoretical growth rate
    omega = column.calculate_growth_rate()
    
    # Should be positive (unstable)
    assert omega > 0
    
    # Should match analytical solution within 5%
    omega_theoretical = np.sqrt(0.072 / (1000 * 0.001**3))
    assert abs(omega - omega_theoretical) / omega_theoretical < 0.05 