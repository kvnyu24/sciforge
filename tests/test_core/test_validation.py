"""Tests for validation utilities."""

import pytest
import numpy as np

from sciforge.core import (
    validate_positive,
    validate_non_negative,
    validate_finite,
    validate_bounds,
    validate_array,
    validate_vector,
    ValidationError,
    DimensionError,
    BoundsError,
)


class TestValidatePositive:
    """Tests for validate_positive function."""

    def test_positive_value(self):
        """Test that positive values pass validation."""
        assert validate_positive(1.0, "mass") == 1.0
        assert validate_positive(0.001, "charge") == 0.001

    def test_zero_fails(self):
        """Test that zero fails validation by default."""
        with pytest.raises(ValidationError):
            validate_positive(0.0, "mass")

    def test_zero_allowed(self):
        """Test that zero passes when allow_zero=True."""
        assert validate_positive(0.0, "value", allow_zero=True) == 0.0

    def test_negative_fails(self):
        """Test that negative values fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            validate_positive(-1.0, "mass")
        assert "mass" in str(exc_info.value)

    def test_nan_fails(self):
        """Test that NaN fails validation."""
        with pytest.raises(ValidationError):
            validate_positive(float("nan"), "value")

    def test_inf_fails(self):
        """Test that infinity fails validation."""
        with pytest.raises(ValidationError):
            validate_positive(float("inf"), "value")


class TestValidateBounds:
    """Tests for validate_bounds function."""

    def test_within_bounds(self):
        """Test values within bounds pass."""
        assert validate_bounds(0.5, (0, 1), "prob") == 0.5
        assert validate_bounds(0.0, (0, 1), "prob") == 0.0
        assert validate_bounds(1.0, (0, 1), "prob") == 1.0

    def test_outside_bounds_fails(self):
        """Test values outside bounds fail."""
        with pytest.raises(BoundsError):
            validate_bounds(1.5, (0, 1), "prob")
        with pytest.raises(BoundsError):
            validate_bounds(-0.1, (0, 1), "prob")

    def test_exclusive_bounds(self):
        """Test exclusive bounds work correctly."""
        # Exclusive lower bound
        with pytest.raises(BoundsError):
            validate_bounds(0.0, (0, 1), "value", inclusive=(False, True))

        # Exclusive upper bound
        with pytest.raises(BoundsError):
            validate_bounds(1.0, (0, 1), "value", inclusive=(True, False))


class TestValidateArray:
    """Tests for validate_array function."""

    def test_list_converted(self):
        """Test that lists are converted to arrays."""
        result = validate_array([1, 2, 3], dim=1)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_dimension_check(self):
        """Test dimension checking."""
        validate_array([1, 2, 3], dim=1)  # Should pass
        validate_array([[1, 2], [3, 4]], dim=2)  # Should pass

        with pytest.raises(DimensionError):
            validate_array([1, 2, 3], dim=2)

    def test_shape_check(self):
        """Test exact shape checking."""
        validate_array([1, 2, 3], expected_shape=(3,))  # Should pass

        with pytest.raises(DimensionError):
            validate_array([1, 2], expected_shape=(3,))


class TestValidateVector:
    """Tests for validate_vector function."""

    def test_3d_vector(self):
        """Test 3D vector validation."""
        result = validate_vector([1, 0, 0])
        np.testing.assert_array_equal(result, [1, 0, 0])

    def test_wrong_size_fails(self):
        """Test that wrong size vectors fail."""
        with pytest.raises(DimensionError):
            validate_vector([1, 0], size=3)

    def test_custom_size(self):
        """Test custom vector sizes."""
        result = validate_vector([1, 0], size=2)
        np.testing.assert_array_equal(result, [1, 0])