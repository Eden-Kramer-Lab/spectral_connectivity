"""Tests for the backend-neutral helpers in ``spectral_connectivity._array_utils``."""

import warnings

import numpy as np

from spectral_connectivity._array_utils import (
    _complex_inner_product,
    _conjugate_transpose,
    _divide_where,
    _squared_magnitude,
)
from tests._backend_helpers import to_device, to_host


def test_conjugate_transpose_swaps_and_conjugates_the_last_two_axes():
    rng = np.random.default_rng(0)
    array = rng.standard_normal((3, 2, 4)) + 1j * rng.standard_normal((3, 2, 4))

    result = _conjugate_transpose(to_device(array))

    assert result.shape == (3, 4, 2)
    np.testing.assert_array_equal(to_host(result), np.conj(np.swapaxes(array, -1, -2)))


def test_divide_where_fills_masked_entries_without_warnings():
    numerator = to_device(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    denominator = to_device(np.array([2.0, 0.0, 4.0, 0.0], dtype=np.float32))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = _divide_where(numerator, denominator, denominator != 0, np.nan)

    np.testing.assert_array_equal(to_host(result), [0.5, np.nan, 0.75, np.nan])
    assert result.dtype == np.float32  # the fill does not upcast the quotient


def test__squared_magnitude():
    test_array = np.array([[1, 2], [3, 4]])
    expected_array = np.array([[1, 4], [9, 16]])
    assert np.allclose(to_host(_squared_magnitude(to_device(test_array))), expected_array)


def test__complex_inner_product():
    """Test that the complex inner product is taken over the last two
    dimensions."""
    test_array1 = np.zeros((3, 2, 4), dtype=complex)
    test_array2 = np.zeros((3, 2, 4), dtype=complex)

    x1 = np.ones((2, 4)) * np.exp(1j * np.pi / 2)
    x2 = np.ones((2, 4)) * np.exp(1j * 0)

    test_array1[1, :, :] = x1
    test_array2[1, :, :] = x2

    test_array1[2, :, :] = x1
    test_array2[2, :, :] = x1

    expected_inner_product = np.zeros((3, 2, 2), dtype=complex)
    expected_inner_product[1, ...] = x1.dot(x2.T.conj())
    expected_inner_product[2, ...] = x1.dot(x1.T.conj())

    assert np.allclose(
        to_host(_complex_inner_product(to_device(test_array1), to_device(test_array2))),
        expected_inner_product,
    )
