"""Tests for the spectral Granger kernels in ``spectral_connectivity._granger``."""

import numpy as np
import pytest

from spectral_connectivity._granger import (
    _remove_instantaneous_causality,
    _sanitized_nonnegative_granger,
)


def test__remove_instantaneous_causality():
    noise_covariance = np.zeros((2, 2, 2))
    x1 = np.array([[1, 2], [2, 4]], dtype=float)
    x2 = np.array([[8, 4], [4, 16]], dtype=float)
    noise_covariance[0, ...] = x1
    noise_covariance[1, ...] = x2

    # x -> y: var(x) - (cov(x,y) ** 2 / var(y))
    # y -> x: var(y) - (cov(x,y) ** 2 / var(x))
    expected_rotated_noise_covariance = np.zeros((2, 2, 2))

    expected_rotated_noise_covariance[0, 0, 1] = x1[1, 1] - (x1[0, 1] ** 2 / x1[0, 0])
    expected_rotated_noise_covariance[0, 1, 0] = x1[0, 0] - (x1[1, 0] ** 2 / x1[1, 1])

    expected_rotated_noise_covariance[1, 0, 1] = x2[1, 1] - (x2[0, 1] ** 2 / x2[0, 0])
    expected_rotated_noise_covariance[1, 1, 0] = x2[0, 0] - (x2[1, 0] ** 2 / x2[1, 1])

    assert np.allclose(
        _remove_instantaneous_causality(noise_covariance),
        expected_rotated_noise_covariance,
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sanitized_nonnegative_granger_enforces_invariant(dtype):
    """The shared sanitizer clips roundoff, NaNs real negatives, keeps the rest."""
    eps = np.finfo(dtype).eps
    tiny_negative = -10 * eps  # inside the 100 * eps roundoff band
    material_negative = -1e-3  # far outside the roundoff band
    value = np.array([0.0, 0.5, tiny_negative, material_negative, np.nan], dtype=dtype)

    sanitized = _sanitized_nonnegative_granger(value)

    # Exact zero (no causality) is preserved, not discarded as NaN.
    assert sanitized[0] == 0.0
    # A genuine positive influence passes through untouched.
    assert sanitized[1] == dtype(0.5)
    # Roundoff around a true zero is clipped up to exactly zero.
    assert sanitized[2] == 0.0
    # A materially-negative (invalid) value becomes NaN.
    assert np.isnan(sanitized[3])
    # NaN propagates unchanged.
    assert np.isnan(sanitized[4])
