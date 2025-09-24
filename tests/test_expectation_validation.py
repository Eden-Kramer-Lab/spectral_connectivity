import numpy as np
import pytest

from spectral_connectivity import Connectivity


def test_valid_expectation_types():
    """Test that all valid expectation types work correctly."""
    # Create test data
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (1, 2, 3, 5, 2)
    fourier_coefficients = (
        np.random.randn(n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals)
        + 1j * np.random.randn(n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals)
    ).astype(np.complex128)

    # All these should work without error
    valid_types = [
        "time", "trials", "tapers",
        "time_trials", "time_tapers", "trials_tapers", "time_trials_tapers"
    ]

    for expectation_type in valid_types:
        conn = Connectivity(
            fourier_coefficients=fourier_coefficients,
            expectation_type=expectation_type
        )
        assert conn.expectation_type == expectation_type


def test_invalid_expectation_type_raises_error():
    """Test that invalid expectation_type raises ValueError with helpful message."""
    # Create test data
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (1, 2, 3, 5, 2)
    fourier_coefficients = (
        np.random.randn(n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals)
        + 1j * np.random.randn(n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals)
    ).astype(np.complex128)

    invalid_type = "invalid_option"

    with pytest.raises(ValueError) as exc_info:
        Connectivity(
            fourier_coefficients=fourier_coefficients,
            expectation_type=invalid_type
        )

    error_msg = str(exc_info.value)

    # Check error message contains key elements
    assert "Invalid expectation_type 'invalid_option'" in error_msg
    assert "Allowed values are:" in error_msg
    assert "'trials'" in error_msg  # Should list valid options
    assert "'trials_tapers'" in error_msg  # Should list valid options


def test_case_sensitive_expectation_type():
    """Test that expectation_type validation is case sensitive."""
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (1, 2, 3, 5, 2)
    fourier_coefficients = (
        np.random.randn(n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals)
        + 1j * np.random.randn(n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals)
    ).astype(np.complex128)

    # Should fail - case sensitivity
    with pytest.raises(ValueError):
        Connectivity(
            fourier_coefficients=fourier_coefficients,
            expectation_type="TRIALS"  # uppercase should fail
        )