"""Tests for error message quality and helpfulness.

This module tests that error messages follow the WHAT/WHY/HOW pattern:
- WHAT: Clear statement of the problem
- WHY: Brief explanation of the cause
- HOW: Specific, actionable recovery steps
"""

import numpy as np
import pytest

from spectral_connectivity import Connectivity, Multitaper
from spectral_connectivity.transforms import detrend


class TestDetrendErrorMessages:
    """Test detrend function error messages follow WHAT/WHY/HOW pattern."""

    def test_invalid_trend_type_error_message(self):
        """Test that invalid trend type error is helpful."""
        data = np.random.randn(100)

        with pytest.raises(ValueError) as excinfo:
            detrend(data, type="invalid")

        error_msg = str(excinfo.value)

        # WHAT: Should state the problem clearly
        assert (
            "trend type 'invalid' is not supported" in error_msg.lower()
            or "invalid trend type" in error_msg.lower()
        )

        # HOW: Should provide valid options
        assert "linear" in error_msg
        assert "constant" in error_msg

        # Message should be specific about what went wrong
        assert "invalid" in error_msg

    def test_breakpoint_validation_error_message(self):
        """Test that breakpoint validation error is helpful."""
        data = np.random.randn(100)

        with pytest.raises(ValueError) as excinfo:
            detrend(data, type="linear", bp=[150])  # Breakpoint beyond data length

        error_msg = str(excinfo.value)

        # WHAT: Should state the problem clearly
        assert "breakpoint" in error_msg.lower()

        # WHY: Should explain the constraint
        assert "length" in error_msg.lower() or "data" in error_msg.lower()

        # HOW: Should help user understand the valid range
        assert "100" in error_msg or "less" in error_msg.lower()


class TestExpectationTypeErrorMessages:
    """Test expectation_type parameter error messages."""

    def test_invalid_expectation_type_error_message(self):
        """Test that invalid expectation_type error is helpful."""
        # Create valid 5D fourier coefficients
        fourier_coefficients = np.random.randn(10, 5, 3, 50, 2) + 1j * np.random.randn(
            10, 5, 3, 50, 2
        )

        with pytest.raises(ValueError) as excinfo:
            Connectivity(fourier_coefficients, expectation_type="invalid_type")

        error_msg = str(excinfo.value)

        # WHAT: Should state what's wrong
        assert "invalid" in error_msg.lower() or "not supported" in error_msg.lower()
        assert "invalid_type" in error_msg

        # HOW: Should list all valid options
        assert "trials_tapers" in error_msg
        assert "trials" in error_msg
        assert "tapers" in error_msg

    def test_expectation_type_suggests_correct_order(self):
        """Test that wrong order in expectation_type gets helpful suggestion."""
        # Create valid 5D fourier coefficients
        fourier_coefficients = np.random.randn(10, 5, 3, 50, 2) + 1j * np.random.randn(
            10, 5, 3, 50, 2
        )

        # Try common mistake: wrong order (e.g., "tapers_trials" instead of "trials_tapers")
        with pytest.raises(ValueError) as excinfo:
            Connectivity(fourier_coefficients, expectation_type="tapers_trials")

        error_msg = str(excinfo.value)

        # Should detect the wrong order and suggest the correct one
        assert "tapers_trials" in error_msg
        assert "Did you mean 'trials_tapers'?" in error_msg


class TestMultitaperParameterErrorMessages:
    """Test Multitaper parameter validation error messages.

    Note: Many of these tests already exist in test_transforms.py.
    These tests focus on the quality and helpfulness of the error messages.
    """

    def test_negative_sampling_frequency_error_is_helpful(self):
        """Test that negative sampling frequency error guides the user."""
        time_series = np.random.randn(100, 1, 2)

        with pytest.raises(ValueError) as excinfo:
            Multitaper(time_series, sampling_frequency=-500.0)

        error_msg = str(excinfo.value)

        # WHAT: Clear statement of problem
        assert "sampling_frequency" in error_msg
        assert "positive" in error_msg.lower() or "greater than 0" in error_msg.lower()

        # WHY: Explains why it's invalid
        assert "-500" in error_msg or "negative" in error_msg.lower()

    def test_invalid_time_halfbandwidth_error_is_helpful(self):
        """Test that invalid time_halfbandwidth_product error explains the parameter."""
        time_series = np.random.randn(100, 1, 2)

        with pytest.raises(ValueError) as excinfo:
            Multitaper(
                time_series,
                sampling_frequency=500.0,
                time_halfbandwidth_product=0.5,  # Too small
            )

        error_msg = str(excinfo.value)

        # WHAT: States the problem
        assert "time_halfbandwidth_product" in error_msg

        # WHY/HOW: Should explain valid range or typical values
        # The error should be educational about this parameter
        assert "0.5" in error_msg or str(0.5) in error_msg


class TestGPUErrorMessages:
    """Test GPU-related error messages."""

    def test_gpu_import_error_is_actionable(self):
        """Test that GPU import error provides clear installation instructions.

        Note: This test verifies the error message content is helpful.
        The actual GPU import code is in transforms.py:20-24.
        """
        # We can't easily test this without mocking the import,
        # but we can read the code and verify it follows best practices
        # This is more of a documentation test

        # Read the error message from the code
        import inspect

        import spectral_connectivity.transforms as transforms_module

        source = inspect.getsource(transforms_module)

        # Verify the GPU error message exists and is helpful
        assert "CuPy is not installed" in source
        assert "pip install cupy" in source or "conda install cupy" in source

        # The error message should explain WHAT (CuPy not installed),
        # WHY (user requested GPU), and HOW (install commands)


class TestErrorMessagePatterns:
    """Test that error messages follow consistent patterns across the codebase."""

    def test_error_messages_provide_context(self):
        """Verify error messages include the problematic value."""
        # This is a meta-test that checks error messages include actual values

        # Example 1: Wrong expectation_type should show what was provided
        fourier_coefficients = np.random.randn(10, 5, 3, 50, 2) + 1j * np.random.randn(
            10, 5, 3, 50, 2
        )

        with pytest.raises(ValueError) as excinfo:
            Connectivity(fourier_coefficients, expectation_type="wrong")

        # Should include the actual wrong value
        assert "wrong" in str(excinfo.value)

    def test_error_messages_provide_solutions(self):
        """Verify error messages suggest how to fix the problem."""
        # Example: Invalid shape should suggest using Multitaper
        fourier_coefficients = np.random.randn(100, 2)  # Wrong shape

        with pytest.raises(ValueError) as excinfo:
            Connectivity(fourier_coefficients)

        error_msg = str(excinfo.value)

        # Should suggest the correct approach
        assert "Multitaper" in error_msg or "transform" in error_msg.lower()
