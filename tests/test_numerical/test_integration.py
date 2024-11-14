import pytest
import numpy as np
from sciforge.numerical import trapezoid, simpson

def test_trapezoid_integration():
    # Test integration of x^2 from 0 to 1
    f = lambda x: x**2
    result = trapezoid(f, 0, 1, 1000)
    assert abs(result - 1/3) < 1e-3

def test_simpson_integration():
    # Test integration of x^2 from 0 to 1
    f = lambda x: x**2
    result = simpson(f, 0, 1, 1001)
    assert abs(result - 1/3) < 1e-6 