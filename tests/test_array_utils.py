"""Tests for the backend-neutral helpers in ``spectral_connectivity._array_utils``."""

import warnings

import numpy as np

from spectral_connectivity._array_utils import _conjugate_transpose, _divide_where


def test_conjugate_transpose_swaps_and_conjugates_the_last_two_axes():
    rng = np.random.default_rng(0)
    array = rng.standard_normal((3, 2, 4)) + 1j * rng.standard_normal((3, 2, 4))

    result = _conjugate_transpose(array)

    assert result.shape == (3, 4, 2)
    np.testing.assert_array_equal(result, np.conj(np.swapaxes(array, -1, -2)))


def test_divide_where_fills_masked_entries_without_warnings():
    numerator = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    denominator = np.array([2.0, 0.0, 4.0, 0.0], dtype=np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = _divide_where(numerator, denominator, denominator != 0, np.nan)

    np.testing.assert_array_equal(result, [0.5, np.nan, 0.75, np.nan])
    assert result.dtype == np.float32  # the fill does not upcast the quotient
