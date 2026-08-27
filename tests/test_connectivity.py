import warnings
from unittest.mock import PropertyMock, patch

import numpy as np
import pytest
from pytest import mark

from spectral_connectivity import Multitaper
from spectral_connectivity.connectivity import (
    Connectivity,
    _bandpass,
    _complex_inner_product,
    _conjugate_transpose,
    _find_largest_independent_group,
    _find_largest_significant_group,
    _get_independent_frequencies,
    _get_independent_frequency_step,
    _max_psd_discrepancy,
    _remove_instantaneous_causality,
    _reshape,
    _set_diagonal_to_zero,
    _squared_magnitude,
    _total_inflow,
    _total_outflow,
)


@mark.parametrize("axis", [(0), (1), (2), (3)])
@mark.parametrize("dtype", [np.complex64, np.complex128])
def test_cross_spectrum(axis, dtype):
    """Test that the cross spectrum is correct for each dimension."""
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (2, 2, 2, 2, 2)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals), dtype=dtype
    )

    signal_fourier_coefficient = [
        2 * np.exp(1j * np.pi / 2),
        3 * np.exp(1j * -np.pi / 2),
    ]
    fourier_ind = [slice(0, 4)] * 5
    fourier_ind[-1] = slice(None)
    fourier_ind[axis] = slice(1, 2)
    fourier_coefficients[tuple(fourier_ind)] = signal_fourier_coefficient

    expected_cross_spectral_matrix = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals, n_signals),
        dtype=dtype,
    )

    expected_slice = np.array([[4, -6], [-6, 9]], dtype=dtype)
    expected_ind = [slice(0, 5)] * 6
    expected_ind[-1] = slice(None)
    expected_ind[-2] = slice(None)
    expected_ind[axis] = slice(1, 2)
    expected_cross_spectral_matrix[tuple(expected_ind)] = expected_slice

    this_Conn = Connectivity(fourier_coefficients=fourier_coefficients)
    assert np.allclose(expected_cross_spectral_matrix, this_Conn._cross_spectral_matrix)


def test_subset_cross_spectrum():
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (2, 2, 2, 2, 2)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals), dtype=complex
    )
    fourier_coefficients[..., :] = [
        2 * np.exp(1j * np.pi / 2),
        3 * np.exp(1j * -np.pi / 2),
    ]
    pairs = np.array([[0, 0], [0, 1]])
    this_Conn = Connectivity(fourier_coefficients=fourier_coefficients)
    full_csm = this_Conn._cross_spectral_matrix
    subset_csm = this_Conn._subset_cross_spectral_matrix(pairs)
    assert np.allclose(
        subset_csm[..., pairs[:, 0], pairs[:, 1]],
        full_csm[..., pairs[:, 0], pairs[:, 1]],
    )
    assert np.allclose(
        subset_csm[..., pairs[:, 1], pairs[:, 0]],
        full_csm[..., pairs[:, 1], pairs[:, 0]],
    )


@mark.parametrize("dtype", [np.complex64, np.complex128])
def test_power(dtype):
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (1, 1, 1, 1, 2)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals), dtype=dtype
    )

    fourier_coefficients[..., :] = [
        2 * np.exp(1j * np.pi / 2),
        3 * np.exp(1j * -np.pi / 2),
    ]

    expected_power = np.zeros((n_time_samples, n_fft_samples, n_signals))

    expected_power[..., :] = [4, 9]

    this_Conn = Connectivity(fourier_coefficients=fourier_coefficients)
    assert np.allclose(expected_power, this_Conn.power())


@mark.parametrize(
    "expectation_type, expected_shape",
    [("trials_tapers", (1, 4, 5)), ("trials", (1, 3, 4, 5)), ("tapers", (1, 2, 4, 5))],
)
def test_expectation(expectation_type, expected_shape):
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (1, 2, 3, 4, 5)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals), dtype=complex
    )

    this_Conn = Connectivity(
        fourier_coefficients=fourier_coefficients,
        expectation_type=expectation_type,
    )
    expectation_function = this_Conn._expectation
    assert np.allclose(expected_shape, expectation_function(fourier_coefficients).shape)


@mark.parametrize(
    "expectation_type, expected_n_observations",
    [("trials_tapers", 6), ("trials", 2), ("tapers", 3)],
)
def test_n_observations(expectation_type, expected_n_observations):
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (1, 2, 3, 4, 5)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals), dtype=complex
    )

    this_Conn = Connectivity(
        fourier_coefficients=fourier_coefficients,
        expectation_type=expectation_type,
    )
    assert this_Conn.n_observations == expected_n_observations


@mark.parametrize("dtype", [np.complex64, np.complex128])
def test_coherency(dtype):
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (1, 30, 1, 1, 2)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals), dtype=dtype
    )

    fourier_coefficients[..., :] = [
        2 * np.exp(1j * np.pi / 2),
        3 * np.exp(1j * -np.pi / 2),
    ]
    this_Conn = Connectivity(fourier_coefficients=fourier_coefficients)
    expected_coherence_magnitude = np.array([[np.nan, 1], [1, np.nan]])
    expected_phase = np.zeros((2, 2)) * np.nan
    expected_phase[0, 1] = np.pi
    expected_phase[1, 0] = -np.pi

    assert np.allclose(
        np.abs(this_Conn.coherency().squeeze()),
        expected_coherence_magnitude,
        equal_nan=True,
    )
    assert np.allclose(
        np.angle(this_Conn.coherency().squeeze()), expected_phase, equal_nan=True
    )


def test_imaginary_coherence():
    """Test that imaginary coherence sets signals with the same phase
    to zero."""
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (1, 30, 1, 1, 2)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals), dtype=complex
    )

    fourier_coefficients[..., :] = [2 * np.exp(1j * 0), 3 * np.exp(1j * 0)]
    expected_imaginary_coherence = np.zeros((2, 2))

    this_Conn = Connectivity(fourier_coefficients=fourier_coefficients)
    assert np.allclose(
        this_Conn.imaginary_coherence().squeeze(), expected_imaginary_coherence
    )


def test_phase_locking_value():
    """Make sure phase locking value ignores magnitudes."""
    rng = np.random.default_rng(42)
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (1, 30, 1, 1, 2)
    fourier_coefficients = rng.uniform(
        0, 2, (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals)
    ) * np.exp(1j * np.pi / 2)
    expected_phase_locking_value_magnitude = np.ones(fourier_coefficients.shape)
    expected_phase_locking_value_angle = np.zeros(fourier_coefficients.shape)
    this_Conn = Connectivity(fourier_coefficients=fourier_coefficients)

    assert np.allclose(
        np.abs(this_Conn.phase_locking_value()), expected_phase_locking_value_magnitude
    )
    assert np.allclose(
        np.angle(this_Conn.phase_locking_value()), expected_phase_locking_value_angle
    )


def test_phase_lag_index_sets_zero_phase_signals_to_zero():
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (1, 30, 1, 1, 2)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals), dtype=complex
    )

    fourier_coefficients[..., :] = [2 * np.exp(1j * 0), 3 * np.exp(1j * 0)]
    expected_phase_lag_index = np.zeros((2, 2))

    this_Conn = Connectivity(fourier_coefficients=fourier_coefficients)
    assert np.allclose(this_Conn.phase_lag_index().squeeze(), expected_phase_lag_index)


def test_phase_lag_index_sets_angles_up_to_pi_to_same_value():
    rng = np.random.default_rng(42)
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (1, 30, 1, 1, 2)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals), dtype=complex
    )
    fourier_coefficients[..., 0] = rng.uniform(
        0.1, 2, (n_time_samples, n_trials, n_tapers, n_fft_samples)
    ) * np.exp(1j * np.pi / 2)
    fourier_coefficients[..., 1] = rng.uniform(
        0.1, 2, (n_time_samples, n_trials, n_tapers, n_fft_samples)
    ) * np.exp(1j * np.pi / 4)

    expected_phase_lag_index = np.zeros((2, 2))
    expected_phase_lag_index[0, 1] = 1
    expected_phase_lag_index[1, 0] = -1

    this_Conn = Connectivity(fourier_coefficients=fourier_coefficients)

    assert np.allclose(this_Conn.phase_lag_index().squeeze(), expected_phase_lag_index)


def test_weighted_phase_lag_index_sets_zero_phase_signals_to_zero():
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (1, 30, 1, 1, 2)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals), dtype=complex
    )

    fourier_coefficients[..., :] = [2 * np.exp(1j * 0), 3 * np.exp(1j * 0)]
    expected_phase_lag_index = np.zeros((2, 2))

    this_Conn = Connectivity(fourier_coefficients=fourier_coefficients)
    assert np.allclose(
        this_Conn.weighted_phase_lag_index().squeeze(), expected_phase_lag_index
    )


def test_weighted_phase_lag_index_is_same_as_phase_lag_index():
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (1, 30, 1, 1, 2)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals), dtype=complex
    )

    fourier_coefficients[..., :] = [
        1 * np.exp(1j * 3 * np.pi / 4),
        1 * np.exp(1j * 5 * np.pi / 4),
    ]

    this_Conn = Connectivity(fourier_coefficients=fourier_coefficients)
    assert np.allclose(
        this_Conn.phase_lag_index(), this_Conn.weighted_phase_lag_index()
    )


def test_debiased_squared_phase_lag_index():
    """Test that incoherent signals produce near-zero values."""
    rng = np.random.default_rng(0)
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (1, 200, 5, 1, 2)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals), dtype=complex
    )

    angles1 = rng.uniform(
        0, 2 * np.pi, (n_time_samples, n_trials, n_tapers, n_fft_samples)
    )
    angles2 = rng.uniform(
        0, 2 * np.pi, (n_time_samples, n_trials, n_tapers, n_fft_samples)
    )

    fourier_coefficients[..., 0] = np.exp(1j * angles1)
    fourier_coefficients[..., 1] = np.exp(1j * angles2)

    this_Conn = Connectivity(fourier_coefficients=fourier_coefficients)

    # For truly random independent phases, expect values close to zero
    # (within statistical fluctuations, ~1/sqrt(n_samples))
    result = this_Conn.debiased_squared_phase_lag_index()
    assert np.all(np.abs(result) < 0.01)  # Reasonable threshold for random data


def test_debiased_squared_weighted_phase_lag_index():
    """Test that incoherent signals are set to zero or below."""
    rng = np.random.default_rng(0)
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (1, 200, 5, 1, 2)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals), dtype=complex
    )

    angles1 = rng.uniform(
        0, 2 * np.pi, (n_time_samples, n_trials, n_tapers, n_fft_samples)
    )
    angles2 = rng.uniform(
        0, 2 * np.pi, (n_time_samples, n_trials, n_tapers, n_fft_samples)
    )

    fourier_coefficients[..., 0] = np.exp(1j * angles1)
    fourier_coefficients[..., 1] = np.exp(1j * angles2)

    this_Conn = Connectivity(fourier_coefficients=fourier_coefficients)

    # set NaN to 0 so less than will work
    debiased_wPLI = this_Conn.debiased_squared_weighted_phase_lag_index()
    debiased_wPLI[np.isnan(debiased_wPLI)] = 0

    assert np.all(debiased_wPLI < np.finfo(float).eps)


def test_pairwise_phase_consistency():
    """Test that incoherent signals are set to zero or below
    and that differences in power are ignored."""
    rng = np.random.default_rng(0)
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (1, 200, 5, 1, 2)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals), dtype=complex
    )

    magnitude1 = rng.uniform(
        0.5, 3, (n_time_samples, n_trials, n_tapers, n_fft_samples)
    )
    angles1 = rng.uniform(
        0, 2 * np.pi, (n_time_samples, n_trials, n_tapers, n_fft_samples)
    )
    magnitude2 = rng.uniform(
        0.5, 3, (n_time_samples, n_trials, n_tapers, n_fft_samples)
    )
    angles2 = rng.uniform(
        0, 2 * np.pi, (n_time_samples, n_trials, n_tapers, n_fft_samples)
    )

    fourier_coefficients[..., 0] = magnitude1 * np.exp(1j * angles1)
    fourier_coefficients[..., 1] = magnitude2 * np.exp(1j * angles2)

    this_Conn = Connectivity(fourier_coefficients=fourier_coefficients)
    ppc = this_Conn.pairwise_phase_consistency()

    # set diagonal to zero because its always 1
    diagonal_ind = np.arange(0, n_signals)
    ppc[..., diagonal_ind, diagonal_ind] = 0

    assert np.all(ppc < np.finfo(float).eps)


def test__reshape():
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (20, 100, 3, 10, 2)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals), dtype=complex
    )
    expected_shape = (n_time_samples, n_fft_samples, n_signals, n_trials * n_tapers)
    assert np.allclose(_reshape(fourier_coefficients).shape, expected_shape)


def test__squared_magnitude():
    test_array = np.array([[1, 2], [3, 4]])
    expected_array = np.array([[1, 4], [9, 16]])
    assert np.allclose(_squared_magnitude(test_array), expected_array)


def test__conjugate_transpose():
    test_array = np.zeros((2, 2, 4), dtype=complex)
    test_array[1, ...] = [
        [1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j],
        [1 - 2j, 3 - 4j, 5 - 6j, 7 - 8j],
    ]
    expected_array = np.zeros((2, 4, 2), dtype=complex)
    expected_array[1, ...] = test_array[1, ...].conj().transpose()
    assert np.allclose(_conjugate_transpose(test_array), expected_array)


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
        _complex_inner_product(test_array1, test_array2), expected_inner_product
    )


def test__set_diagonal_to_zero():
    test_array = np.ones((2, 2, 2))
    expected_array = np.ones((2, 2, 2))
    expected_array[0, 0, 0] = 0
    expected_array[0, 1, 1] = 0
    expected_array[1, 0, 0] = 0
    expected_array[1, 1, 1] = 0
    assert np.allclose(_set_diagonal_to_zero(test_array), expected_array)


def test__bandpass():
    test_data = np.arange(0, 10).reshape((2, 5))
    labels = np.arange(0, 5) * 2
    labels_of_interest = [1, 5]

    expected_labels = np.array([2, 4])
    expected_data = np.array([[1, 2], [6, 7]])

    filtered_data, filtered_labels = _bandpass(
        test_data, labels, labels_of_interest, axis=-1
    )

    assert np.allclose(expected_data, filtered_data) & np.allclose(
        expected_labels, filtered_labels
    )


@mark.parametrize(
    "frequency_difference, frequency_resolution, expected_step",
    [(2.0, 5.0, 3), (5.0, 2.0, 1), (2.0, 2.0, 1)],
)
def test__get_independent_frequency_step(
    frequency_difference, frequency_resolution, expected_step
):
    step = _get_independent_frequency_step(frequency_difference, frequency_resolution)
    assert step == expected_step


@mark.parametrize(
    "is_significant, expected_is_significant",
    [
        (
            np.array([False, True, True, False, True, True, True, False]),
            np.array([False, False, False, False, True, True, True, False]),
        ),
        (np.ones((10,), dtype=bool), np.ones((10,), dtype=bool)),
    ],
)
def test__find_largest_significant_group(is_significant, expected_is_significant):
    assert np.allclose(
        _find_largest_significant_group(is_significant), expected_is_significant
    )


def test__find_largest_significant_group_with_no_significant():
    is_significant = np.zeros((10,), dtype=bool)
    expected_is_significant = np.zeros((10,), dtype=bool)

    assert np.allclose(
        _find_largest_significant_group(is_significant), expected_is_significant
    )


def test__get_independent_frequencies():
    is_significant = np.zeros((10,), dtype=bool)
    is_significant[3:7] = True
    frequency_step = 2

    expected_is_significant = np.zeros((10,), dtype=bool)
    expected_is_significant[3:7:frequency_step] = True

    assert np.allclose(
        _get_independent_frequencies(is_significant, frequency_step),
        expected_is_significant,
    )


@mark.parametrize(
    "min_group_size, expected_is_significant",
    [
        (3, np.zeros((10,), dtype=bool)),
        (
            1,
            np.array(
                [False, False, False, False, True, False, True, False, False, False],
                dtype=bool,
            ),
        ),
    ],
)
def test__find_largest_independent_group(min_group_size, expected_is_significant):
    is_significant = np.zeros((10,), dtype=bool)
    is_significant[1:3] = True
    is_significant[4:7] = True
    is_significant[8] = True
    frequency_step = 2

    assert np.allclose(
        _find_largest_independent_group(
            is_significant, frequency_step, min_group_size=min_group_size
        ),
        expected_is_significant,
    )


def test_largest_independent_group_vectorized_matches_reference():
    """The vectorized selection equals the per-slice reference exactly.

    ``_find_significant_frequencies`` selects the largest independent
    significant cluster per (batch, pair) slice with a single vectorized pass
    instead of ``np.apply_along_axis(_find_largest_independent_group, ...)``.
    The two must agree bit-for-bit (it is boolean logic) over random inputs and
    the edge cases (all/none significant, a single frequency, tied clusters).
    The chunked path (bounded memory) must match the single-pass path.
    """
    from unittest.mock import patch

    from spectral_connectivity import connectivity as conn_mod
    from spectral_connectivity.connectivity import (
        _largest_independent_group_along_frequency,
    )

    rng = np.random.default_rng(0)
    for _ in range(200):
        n_batch = int(rng.integers(1, 4))
        n_frequencies = int(rng.integers(1, 25))
        n_pairs = int(rng.integers(1, 6))
        is_significant = rng.random((n_batch, n_frequencies, n_pairs)) < rng.uniform(
            0.1, 0.9
        )
        frequency_step = int(rng.integers(1, 4))
        min_group_size = int(rng.integers(1, 5))
        reference = np.apply_along_axis(
            _find_largest_independent_group,
            -2,
            is_significant,
            frequency_step,
            min_group_size,
        )
        vectorized = _largest_independent_group_along_frequency(
            is_significant, frequency_step, min_group_size
        )
        np.testing.assert_array_equal(vectorized, reference)
        # A tiny chunk cap forces the slices through several bounded chunks and
        # must give the identical result.
        with patch.object(conn_mod, "_SIGNIFICANCE_SELECTION_CHUNK_ELEMENTS", 7):
            chunked = _largest_independent_group_along_frequency(
                is_significant, frequency_step, min_group_size
            )
        np.testing.assert_array_equal(chunked, reference)

    # An empty frequency band (n_frequencies == 0) must return an empty result,
    # not raise (the reshape cannot infer a -1 dimension at size 0).
    empty = np.zeros((2, 0, 3), dtype=bool)
    empty_result = _largest_independent_group_along_frequency(empty, 2, 3)
    assert empty_result.shape == (2, 0, 3)
    assert empty_result.dtype == bool
    assert empty_result.size == 0

    # Explicit edge cases, including two equal-size clusters (first is kept).
    tie = np.array([[True, True, False, True, True, False]]).reshape(1, 6, 1)
    for shape_case in (
        np.zeros((1, 8, 2), bool),
        np.ones((1, 8, 2), bool),
        np.ones((1, 1, 3), bool),
        tie,
    ):
        for frequency_step in (1, 2, 3):
            for min_group_size in (1, 3):
                np.testing.assert_array_equal(
                    _largest_independent_group_along_frequency(
                        shape_case, frequency_step, min_group_size
                    ),
                    np.apply_along_axis(
                        _find_largest_independent_group,
                        -2,
                        shape_case,
                        frequency_step,
                        min_group_size,
                    ),
                )


def test__total_inflow():
    transfer_function = np.ones((2, 3, 3))
    noise_variance = [4, 2, 3]
    expected_total_inflow = 3 * np.ones((2, 3, 1))

    assert np.allclose(
        _total_inflow(transfer_function, noise_variance), expected_total_inflow
    )


def test__total_outflow():
    MVAR_Fourier_coefficients = np.ones((2, 3, 3))
    noise_variance = np.array([0.25, 0.5, 1 / 3])
    expected_total_outflow = np.ones((2, 1, 3)) * np.sqrt(1.0 / noise_variance * 3)

    assert np.allclose(
        _total_outflow(MVAR_Fourier_coefficients, noise_variance),
        expected_total_outflow,
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


def test_directed_transfer_function():
    # Use proper 5D shape for fourier_coefficients
    c = Connectivity(fourier_coefficients=np.empty((1, 1, 1, 1, 2)))
    # Use patch context manager to avoid contaminating the class
    with patch.object(
        Connectivity, "_transfer_function", new_callable=PropertyMock
    ) as mock_prop:
        mock_prop.return_value = np.arange(1, 5).reshape((2, 2))
        dtf = c.directed_transfer_function()
        assert np.allclose(dtf.sum(axis=-1), 1.0)
        assert np.all((dtf >= 0.0) & (dtf <= 1.0))


def test_partial_directed_coherence():
    # Use proper 5D shape for fourier_coefficients
    c = Connectivity(fourier_coefficients=np.empty((1, 1, 1, 1, 2)))
    # Use patch context manager to avoid contaminating the class
    with patch.object(
        Connectivity, "_MVAR_Fourier_coefficients", new_callable=PropertyMock
    ) as mock_prop:
        mock_prop.return_value = np.arange(1, 5).reshape((2, 2))
        pdc = c.partial_directed_coherence()
        assert np.allclose(pdc.sum(axis=-2), 1.0)
        assert np.all((pdc >= 0.0) & (pdc <= 1.0))


def test_directed_coherence_is_bounded_and_normalized():
    """Directed coherence must stay in [0, 1] and normalize over sources.

    Regression test for a noise-variance broadcasting bug: the source noise
    variance was applied on the target axis (-2) instead of the source axis
    (-1), producing values > 1 whenever channels had unequal noise variances.
    The squared directed coherence sums to 1 over sources (like DTF).
    """
    c = Connectivity(fourier_coefficients=np.empty((1, 1, 1, 1, 2)))
    transfer_function = np.arange(1, 5).reshape((2, 2)).astype(float)  # [target, src]
    noise_covariance = np.diag([10.0, 1.0])  # unequal per-source noise variances
    with (
        patch.object(
            Connectivity, "_transfer_function", new_callable=PropertyMock
        ) as mock_transfer,
        patch.object(
            Connectivity, "_noise_covariance", new_callable=PropertyMock
        ) as mock_noise,
    ):
        mock_transfer.return_value = transfer_function
        mock_noise.return_value = noise_covariance
        dc = c.directed_coherence()
        assert np.all((dc >= 0.0) & (dc <= 1.0))
        assert np.allclose(dc.sum(axis=-1), 1.0)
        expected = np.array([[10 / 14, 4 / 14], [90 / 106, 16 / 106]])
        assert np.allclose(np.squeeze(dc), expected)


def test_max_psd_discrepancy():
    """The helper reports the relative gap between the diagonal and true PSD."""
    # Diagonal covariance: the diagonal denominator equals the true PSD.
    dense_H = np.arange(1, 5).reshape((1, 1, 2, 2)).astype(float)
    assert _max_psd_discrepancy(dense_H, np.diag([1.0, 2.0])[np.newaxis]) == 0.0
    # Single signal: not assessable.
    assert _max_psd_discrepancy(np.ones((1, 1, 1, 1)), np.array([[[3.0]]])) == 0.0
    # Aggregate cross-power a pairwise-correlation threshold would miss: 20
    # equicorrelated sources at rho=0.09 (every pair < 0.1) with an all-ones
    # transfer row omit ~63% of the true PSD.
    n, rho = 20, 0.09
    equicorrelated = ((1 - rho) * np.eye(n) + rho * np.ones((n, n)))[np.newaxis]
    all_ones = np.ones((1, 1, n, n))
    assert np.isclose(_max_psd_discrepancy(all_ones, equicorrelated), 0.631, atol=0.01)
    # True power exactly 0 (a rank-deficient but valid PSD covariance) with
    # nonzero diagonal power: the diagonal formula is infinitely wrong, so the
    # discrepancy is +inf and must not be dropped as unassessable.
    singular_cov = np.array([[1.0, -1.0], [-1.0, 1.0]])[np.newaxis]
    invertible_H = np.array([[1.0, 1.0], [1.0, 0.0]])[np.newaxis, np.newaxis]
    assert _max_psd_discrepancy(invertible_H, singular_cov) == np.inf
    # No power at all (0/0) is genuinely unassessable and stays 0.
    assert _max_psd_discrepancy(np.zeros((1, 1, 2, 2)), np.eye(2)[np.newaxis]) == 0.0


def test_directed_coherence_warns_on_material_cross_power():
    """directed_coherence warns when the diagonal denominator omits material power.

    The denominator ``sum_k nv_k|H_ik|^2`` equals the true PSD ``(H Cov H^H)_ii``
    only for uncorrelated innovations. It must warn when the omitted cross-power
    is a material fraction of the true PSD -- including the dimension-aware case
    where every pairwise correlation is below any fixed threshold but the sources
    jointly omit most of the power -- and stay silent for a diagonal covariance.
    """
    dense_H = np.arange(1, 5).reshape((1, 1, 2, 2)).astype(float)
    n, rho = 20, 0.09
    equicorrelated = ((1 - rho) * np.eye(n) + rho * np.ones((n, n)))[np.newaxis]
    # (transfer_function, noise_covariance, should_warn)
    cases = [
        (dense_H, np.array([[1.0, 0.9], [0.9, 1.0]])[np.newaxis], True),
        (dense_H, np.diag([1.0, 2.0])[np.newaxis], False),
        (np.ones((1, 1, n, n)), equicorrelated, True),  # weak pairwise, large sum
        # True power 0 vs diagonal power 2 -> infinite discrepancy -> must warn.
        (
            np.array([[1.0, 1.0], [1.0, 0.0]])[np.newaxis, np.newaxis],
            np.array([[1.0, -1.0], [-1.0, 1.0]])[np.newaxis],
            True,
        ),
    ]
    for transfer_function, noise_covariance, should_warn in cases:
        c = Connectivity(
            fourier_coefficients=np.empty((1, 1, 1, 1, transfer_function.shape[-1]))
        )
        with (
            patch.object(
                Connectivity, "_transfer_function", new_callable=PropertyMock
            ) as mock_transfer,
            patch.object(
                Connectivity, "_noise_covariance", new_callable=PropertyMock
            ) as mock_noise,
        ):
            mock_transfer.return_value = transfer_function
            mock_noise.return_value = noise_covariance
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                c.directed_coherence()
            warned = any(
                "uncorrelated MVAR innovations" in str(w.message) for w in caught
            )
            assert warned is should_warn


def test_single_signal_connectivity_raises_but_power_works():
    """Connectivity on a single signal raises; power() still works."""
    c = Connectivity(fourier_coefficients=np.ones((1, 1, 1, 4, 1), dtype=complex))
    for method in (c.coherence_magnitude, c.phase_locking_value, c.global_coherence):
        with pytest.raises(ValueError, match="at least 2 signals"):
            method()
    with pytest.raises(ValueError, match="at least 2 signals"):
        c.canonical_coherence(group_labels=np.array([0]))
    # Power is well-defined for a single signal.
    power = c.power()
    assert power.shape[-1] == 1
    assert np.all(np.isfinite(power))


def test_debiased_measures_require_multiple_observations():
    """Debiased PLI / PPC divide by (n_obs - 1); reject n_observations == 1."""
    # n_trials == n_tapers == 1 => n_observations == 1 with default expectation.
    c = Connectivity(fourier_coefficients=np.ones((1, 1, 1, 4, 2), dtype=complex))
    assert c.n_observations == 1
    with pytest.raises(ValueError, match="at least 2 observations"):
        c.debiased_squared_phase_lag_index()
    with pytest.raises(ValueError, match="at least 2 observations"):
        c.pairwise_phase_consistency()


def test_coherency_zero_power_returns_nan():
    """A dead (all-zero) channel yields NaN coherency, not huge values."""
    rng = np.random.default_rng(0)
    fourier = rng.standard_normal((1, 1, 2, 4, 2)) + 1j * rng.standard_normal(
        (1, 1, 2, 4, 2)
    )
    fourier[..., 1] = 0.0  # signal 1 is a flat/dead channel -> zero power
    c = Connectivity(fourier_coefficients=fourier)
    with pytest.warns(UserWarning, match="zero power"):
        coherency = c.coherency()
    # Pairs involving the dead channel are undefined -> NaN (not > 1).
    assert np.all(np.isnan(coherency[..., 0, 1]))
    assert np.all(np.isnan(coherency[..., 1, 0]))

    # imaginary_coherence has the same guard and must also return NaN (not a
    # clipped value) for the dead-channel pairs.
    c_imag = Connectivity(fourier_coefficients=fourier)
    with pytest.warns(UserWarning, match="zero power"):
        imag_coh = c_imag.imaginary_coherence()
    assert np.all(np.isnan(imag_coh[..., 0, 1]))
    assert np.all(np.isnan(imag_coh[..., 1, 0]))


def test_transfer_function_is_cached():
    """Expensive directed-connectivity intermediates are cached per instance."""
    rng = np.random.default_rng(1)
    fourier = rng.standard_normal((1, 3, 2, 8, 2)) + 1j * rng.standard_normal(
        (1, 3, 2, 8, 2)
    )
    c = Connectivity(fourier_coefficients=fourier)
    assert c._minimum_phase_factor is c._minimum_phase_factor
    assert c._transfer_function is c._transfer_function
    assert c._noise_covariance is c._noise_covariance


def test_changing_inputs_clears_cached_intermediates():
    """Reassigning fourier_coefficients / expectation_type must not serve stale cache."""
    rng = np.random.default_rng(2)
    fourier = rng.standard_normal((1, 4, 2, 8, 2)) + 1j * rng.standard_normal(
        (1, 4, 2, 8, 2)
    )
    c = Connectivity(fourier_coefficients=fourier, expectation_type="trials_tapers")
    transfer_default = c._transfer_function

    # "tapers" retains the trials dimension, so the recomputed transfer function
    # has a different shape -- proving the cache was invalidated, not reused.
    c.expectation_type = "tapers"
    transfer_tapers = c._transfer_function
    assert transfer_tapers.shape != transfer_default.shape

    # Reassigning fourier_coefficients also clears the cache.
    minimum_phase = c._minimum_phase_factor
    c.fourier_coefficients = fourier
    assert c._minimum_phase_factor is not minimum_phase


def test_power_and_cross_spectrum_caches_invalidate():
    """Cached _power / reduced cross-spectrum must not serve stale results.

    Both are cached per instance for reuse across measures; reassigning the
    inputs must drop them so a later access recomputes from the new data.
    """
    rng = np.random.default_rng(5)
    shape = (2, 4, 3, 8, 2)
    fourier = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    c = Connectivity(fourier_coefficients=fourier, expectation_type="trials_tapers")

    # Populate the caches: coherency reads power + reduced CSM; a phase-lag-index
    # measure populates the shared imaginary-cross-spectrum moments.
    c.coherency()
    c.weighted_phase_lag_index()
    assert "_power" in c.__dict__
    assert "_cached_reduced_cross_spectral_matrix" in c.__dict__
    assert "_imaginary_moment_cache" in c.__dict__
    power_before = c._power
    csm_before = c._cached_reduced_cross_spectral_matrix
    wpli_before = c.weighted_phase_lag_index()

    # Reassigning to *different* data must recompute, not serve the stale cache.
    other = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    c.fourier_coefficients = other
    assert "_power" not in c.__dict__
    assert "_cached_reduced_cross_spectral_matrix" not in c.__dict__
    assert "_imaginary_moment_cache" not in c.__dict__
    assert not np.allclose(c._power, power_before)
    assert not np.allclose(c._cached_reduced_cross_spectral_matrix, csm_before)
    # The fused phase-lag-index family recomputes from the new data and matches a
    # fresh instance (a stale moment cache would fail this).
    fresh_wpli = Connectivity(
        fourier_coefficients=other, expectation_type="trials_tapers"
    ).weighted_phase_lag_index()
    assert not np.allclose(c.weighted_phase_lag_index(), wpli_before)
    np.testing.assert_array_equal(c.weighted_phase_lag_index(), fresh_wpli)

    # Changing expectation_type also invalidates (and changes the averaged shape).
    power_trials_tapers = c._power
    c.expectation_type = "tapers"  # retains the trials axis
    assert "_power" not in c.__dict__
    assert c._power.shape != power_trials_tapers.shape


@pytest.mark.parametrize("expectation_type", ["trials_tapers", "tapers"])
def test_phase_lag_index_family_matches_per_fcn_reference(expectation_type):
    """The fused phase-lag-index family matches the per-fcn cross-spectrum path.

    phase_lag_index, weighted_phase_lag_index and
    debiased_squared_weighted_phase_lag_index now share one observation-level
    imaginary cross-spectrum (four cached moments) instead of re-forming it per
    ``fcn``. Each must equal the original per-fcn computation. Parametrized over
    ``expectation_type`` because ``debiased_squared_weighted_phase_lag_index``
    scales by ``n_observations``, which changes with it. Also checks that
    computing one measure does not corrupt a cached moment another relies on.
    """
    rng = np.random.default_rng(0)
    shape = (2, 8, 5, 32, 5)
    fc = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(
        np.complex128
    )

    def zero_diagonal_imag(x):
        imag = x.imag
        n_signals = imag.shape[-1]
        di = np.diag_indices(n_signals)
        imag[..., di[0], di[1]] = 0
        return imag

    def non_negative(a):  # mirror the @_non_negative_frequencies(-3) decorator
        return a[..., : a.shape[-3] // 2 + 1, :, :]

    conn = Connectivity(fc, expectation_type=expectation_type)
    n_observations = conn.n_observations
    # Reference moments computed independently by averaging a transform of the
    # per-observation cross-spectral matrix (a fresh matrix per call, since
    # zero_diagonal_imag mutates the imaginary view in place).
    mean_sign = conn._expectation(
        np.sign(zero_diagonal_imag(conn._cross_spectral_matrix))
    )
    mean_imag = conn._expectation(zero_diagonal_imag(conn._cross_spectral_matrix))
    mean_abs = conn._expectation(
        np.abs(zero_diagonal_imag(conn._cross_spectral_matrix))
    )
    mean_sq = conn._expectation(zero_diagonal_imag(conn._cross_spectral_matrix) ** 2)

    expected_pli = non_negative(mean_sign.real)
    weights = mean_abs.copy()
    weights[weights < np.finfo(float).eps] = 1
    expected_wpli = non_negative(mean_imag / weights)
    imag_sum = mean_imag * n_observations
    sq_sum = mean_sq * n_observations
    abs_sum = mean_abs * n_observations
    dwpli_weights = abs_sum**2 - sq_sum
    dwpli_weights[dwpli_weights == 0] = np.nan
    expected_dwpli = non_negative((imag_sum**2 - sq_sum) / dwpli_weights)

    np.testing.assert_array_equal(conn.phase_lag_index(), expected_pli)
    np.testing.assert_array_equal(conn.weighted_phase_lag_index(), expected_wpli)
    np.testing.assert_array_equal(
        conn.debiased_squared_weighted_phase_lag_index(), expected_dwpli
    )

    # Computing wpli (which guards its weights in place on a copy) must not
    # change a later debiased_squared_weighted_phase_lag_index result.
    warm = Connectivity(fc, expectation_type=expectation_type)
    warm.weighted_phase_lag_index()
    np.testing.assert_array_equal(
        warm.debiased_squared_weighted_phase_lag_index(), expected_dwpli
    )


def test_phase_lag_index_moments_are_computed_lazily():
    """A single-measure call computes only the moments that measure needs.

    The reduced imaginary-cross-spectrum moments are computed per key on demand,
    so a lone ``phase_lag_index`` does not also compute (or retain) the other
    measures' moments, while the family shares the ones already computed.
    """
    rng = np.random.default_rng(1)
    shape = (2, 6, 4, 16, 4)
    fc = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(
        np.complex128
    )

    pli_only = Connectivity(fc)
    pli_only.phase_lag_index()
    assert set(pli_only.__dict__["_imaginary_moment_cache"]) == {"sign"}

    wpli_only = Connectivity(fc)
    wpli_only.weighted_phase_lag_index()
    assert set(wpli_only.__dict__["_imaginary_moment_cache"]) == {
        "imaginary",
        "absolute",
    }

    # The family accumulates all four moments across measures.
    family = Connectivity(fc)
    family.phase_lag_index()
    family.weighted_phase_lag_index()
    family.debiased_squared_weighted_phase_lag_index()
    assert set(family.__dict__["_imaginary_moment_cache"]) == {
        "sign",
        "imaginary",
        "absolute",
        "squared",
    }

    # A repeated measure reuses the cached moment object (it is not recomputed).
    reuse = Connectivity(fc)
    reuse.phase_lag_index()
    cached_sign = reuse.__dict__["_imaginary_moment_cache"]["sign"]
    reuse.phase_lag_index()
    assert reuse.__dict__["_imaginary_moment_cache"]["sign"] is cached_sign


def test_phase_lag_index_family_fully_cached_path_matches_cold():
    """The fully-cached (no missing key) return path yields the cold result.

    When every requested moment is already cached, the method returns purely
    from the cache. Exercise that path with the reverse family order (dwpli
    caches imaginary/absolute/squared, so wpli then needs nothing new) and a
    repeated call, and require the values to match a fresh instance. This would
    catch a cache-return bug (e.g. iterating the cache instead of the requested
    keys) that the forward-order tests, which always add a key, cannot.
    """
    rng = np.random.default_rng(2)
    shape = (2, 6, 4, 16, 4)
    fc = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(
        np.complex128
    )

    warm = Connectivity(fc)
    warm.debiased_squared_weighted_phase_lag_index()  # caches imag/abs/squared
    wpli_from_cache = warm.weighted_phase_lag_index()  # all keys already cached
    pli_first = warm.phase_lag_index()
    pli_again = warm.phase_lag_index()  # sign already cached

    cold = Connectivity(fc)
    np.testing.assert_array_equal(wpli_from_cache, cold.weighted_phase_lag_index())
    np.testing.assert_array_equal(pli_first, cold.phase_lag_index())
    np.testing.assert_array_equal(pli_first, pli_again)


def test_fourier_coefficients_are_an_immutable_snapshot():
    """In-place edits must not silently bypass cache invalidation.

    The cached power / cross-spectrum assume the coefficients change only via
    assignment (which clears the caches). The constructor therefore stores a
    private snapshot, and the accessor never hands out a writable alias of it:
    mutating the caller's original array, or the array the property returns, must
    not serve stale scientific results.
    """
    rng = np.random.default_rng(9)
    shape = (1, 4, 2, 8, 3)
    fourier = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(
        np.complex128
    )

    c = Connectivity(fourier_coefficients=fourier)
    power_before = c.power().copy()
    coherence_before = c.coherence_magnitude().copy()

    # Mutating the caller's original array must not reach the warmed instance.
    fourier[...] = 0.0
    np.testing.assert_array_equal(c.power(), power_before)
    np.testing.assert_array_equal(c.coherence_magnitude(), coherence_before)

    # The getter returns an independent, read-only copy: an in-place edit raises
    # loudly rather than silently vanishing against a discarded object.
    returned = c.fourier_coefficients
    assert returned.base is None  # an owning copy, not a view of the snapshot
    assert returned.flags.writeable is False
    with pytest.raises(ValueError):
        returned[...] = 0.0
    # Even re-enabling writeability (permitted on an owning copy) and mutating it
    # cannot reach the instance: the copy shares no buffer with the snapshot.
    returned.flags.writeable = True
    returned[...] = 0.0
    np.testing.assert_array_equal(c.power(), power_before)
    np.testing.assert_array_equal(c.coherence_magnitude(), coherence_before)


def test_fourier_coefficients_getter_returns_fresh_independent_copy():
    """Each getter call returns a distinct copy disconnected from the snapshot."""
    rng = np.random.default_rng(11)
    shape = (1, 4, 2, 8, 3)
    fourier = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(
        np.complex128
    )

    c = Connectivity(fourier_coefficients=fourier)
    power_before = c.power().copy()

    first = c.fourier_coefficients
    second = c.fourier_coefficients
    # Distinct objects, neither aliasing the backing snapshot.
    assert first is not second
    assert first is not c._fourier_coefficients
    assert first.base is None and second.base is None
    np.testing.assert_array_equal(first, c._fourier_coefficients)

    # Re-enable and mutate one copy; the other copy, the snapshot, and the cache
    # are all untouched (independent buffers).
    first.flags.writeable = True
    first[...] = 0.0
    np.testing.assert_array_equal(c.power(), power_before)
    assert not np.array_equal(first, second)


def test_direct_construction_copies_and_is_isolated_from_caller_mutation():
    """Connectivity(fc) must defensively copy: mutating fc cannot reach it."""
    rng = np.random.default_rng(21)
    shape = (1, 4, 2, 8, 3)
    fourier = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(
        np.complex128
    )
    c = Connectivity(fourier_coefficients=fourier)
    # The stored snapshot owns its data (a copy), not a view of the caller's array.
    assert c._fourier_coefficients.base is None
    power_before = c.power().copy()
    fourier[...] = 0.0
    np.testing.assert_array_equal(c.power(), power_before)


def test_from_multitaper_adopts_without_copying():
    """from_multitaper takes ownership of fft() output without a full copy.

    Multitaper.fft() returns a swapaxes view that is referenced nowhere else, so
    Connectivity freezes it in place rather than copying the largest array in the
    pipeline. The stored array is that view (``base`` is not None), and the entire
    base chain is frozen so the backing buffer cannot be mutated.
    """
    rng = np.random.default_rng(22)
    m = Multitaper(
        rng.standard_normal((400, 6, 3)),
        sampling_frequency=400,
        time_halfbandwidth_product=3,
    )
    c = Connectivity.from_multitaper(m)

    stored = c._fourier_coefficients
    # Adopted (a view of fft() output), not an owning copy.
    assert stored.base is not None

    # The whole base chain is read-only, so the writable backing buffer of the
    # swapaxes view is unreachable for mutation.
    obj = stored
    while obj is not None:
        assert obj.flags.writeable is False
        obj = obj.base

    # Freezing only the outer view would leave the deepest base writable; confirm
    # the deepest base cannot be mutated.
    base = stored
    while base.base is not None:
        base = base.base
    with pytest.raises(ValueError):
        base[(0,) * base.ndim] = 0.0


@mark.parametrize(
    "make",
    [
        lambda m: Connectivity.from_multitaper(m),  # adoption path (frozen view)
        lambda m: Connectivity(np.asarray(m.fft())),  # defensive-copy path
    ],
    ids=["adopt", "copy"],
)
def test_getter_copy_defeats_base_reenable_attack(make):
    """The getter must not expose any writable path to the internal snapshot.

    A read-only view would still let a caller reach the owning base through
    ``.base``, re-enable its ``writeable`` flag (NumPy allows this on an array
    that owns its data), then re-enable and mutate the view -- corrupting the
    snapshot behind the warmed caches. The getter therefore returns an
    independent copy; walking to the root base and re-enabling it must not reach
    ``c._fourier_coefficients``.
    """
    rng = np.random.default_rng(23)
    m = Multitaper(
        rng.standard_normal((400, 6, 3)),
        sampling_frequency=400,
        time_halfbandwidth_product=3,
    )
    c = make(m)
    internal_before = np.asarray(c._fourier_coefficients).copy()
    power_before = c.power().copy()

    returned = c.fourier_coefficients
    assert returned.base is None  # an owning copy: no reachable snapshot base

    # Even performing the full attack on the returned copy's own root base leaves
    # the instance untouched, because the copy shares no buffer with the snapshot.
    root = returned
    while root.base is not None:
        root = root.base
    root.flags.writeable = True
    returned.flags.writeable = True
    returned[...] = 0.0

    np.testing.assert_array_equal(c._fourier_coefficients, internal_before)
    np.testing.assert_array_equal(c.power(), power_before)


def test_from_multitaper_adoption_matches_copy_numerically():
    """Adoption must not change results vs. the defensive-copy path."""
    rng = np.random.default_rng(24)
    m = Multitaper(
        rng.standard_normal((400, 6, 3)),
        sampling_frequency=400,
        time_halfbandwidth_product=3,
    )
    adopted = Connectivity.from_multitaper(m)
    copied = Connectivity(np.asarray(m.fft()))  # forces the copy path

    np.testing.assert_array_equal(adopted.power(), copied.power())
    for measure in ("coherence_magnitude", "imaginary_coherence"):
        a = getattr(adopted, measure)()
        b = getattr(copied, measure)()
        # equal including matching NaN positions (the coherence diagonal)
        assert np.array_equal(a, b, equal_nan=True)


def test_from_multitaper_adoption_invalidates_cache_on_reassignment():
    """Reassigning fourier_coefficients on an adopted instance clears caches."""
    rng = np.random.default_rng(25)
    m = Multitaper(
        rng.standard_normal((400, 6, 3)),
        sampling_frequency=400,
        time_halfbandwidth_product=3,
    )
    c = Connectivity.from_multitaper(m)
    _ = c.power()
    assert "_power" in c.__dict__  # cached
    c.fourier_coefficients = np.asarray(m.fft()) * 2.0
    assert "_power" not in c.__dict__  # cleared
    # New power reflects the reassigned (4x) data.
    np.testing.assert_allclose(c.power(), 4.0 * Connectivity.from_multitaper(m).power())


def test_debiased_weighted_pli_requires_multiple_observations():
    """debiased_squared_weighted_phase_lag_index guards n_observations < 2."""
    c = Connectivity(fourier_coefficients=np.ones((1, 1, 1, 4, 2), dtype=complex))
    assert c.n_observations == 1
    with pytest.raises(ValueError, match="at least 2 observations"):
        c.debiased_squared_weighted_phase_lag_index()


def test_subset_pairwise_granger_prediction():
    rng = np.random.default_rng(0)
    T = 64

    # Generate causal signals: x -> y
    x = rng.standard_normal((2, T))
    y = np.zeros_like(x)
    for t in range(1, T):
        y[:, t] = 0.8 * x[:, t - 1]

    # Stack to [trials, signals, time]
    data = np.stack([x, y], axis=1)

    fft_data = np.fft.rfft(data, axis=-1)
    fourier_coefficients = fft_data[None, :, None, :, :]
    c = Connectivity(fourier_coefficients=fourier_coefficients)
    pairs = np.array([[0, 0], [0, 1]])
    gp_subset = c.subset_pairwise_spectral_granger_prediction(pairs)
    gp_all = c.pairwise_spectral_granger_prediction()
    assert gp_subset.shape == gp_all.shape
    for i, j in pairs:
        assert np.allclose(gp_subset[..., i, j], gp_all[..., i, j], equal_nan=True)
        assert np.allclose(gp_subset[..., j, i], gp_all[..., j, i], equal_nan=True)


def test_nyquist_bin_even_n():
    """Test that Nyquist bin is included for even N FFT lengths."""
    # Create signal with even FFT length (N=1024)
    rng = np.random.default_rng(42)
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = 1, 1, 1, 1024, 2

    # Create random fourier coefficients with full frequency spectrum
    fourier_coefficients = rng.random(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals)
    ).astype(complex)

    c = Connectivity(fourier_coefficients=fourier_coefficients)

    # Test coherence which uses @_non_negative_frequencies decorator
    coherence = c.coherence_magnitude()

    # For even N=1024, should have N//2+1 = 513 frequencies (including Nyquist)
    expected_n_frequencies = n_fft_samples // 2 + 1
    assert coherence.shape[-3] == expected_n_frequencies, (
        f"Expected {expected_n_frequencies} frequencies, got {coherence.shape[-3]}"
    )


def test_nyquist_bin_odd_n():
    """Test that frequency indexing works correctly for odd N FFT lengths."""
    # Create signal with odd FFT length (N=1023)
    rng = np.random.default_rng(42)
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = 1, 1, 1, 1023, 2

    # Create random fourier coefficients with full frequency spectrum
    fourier_coefficients = rng.random(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals)
    ).astype(complex)

    c = Connectivity(fourier_coefficients=fourier_coefficients)

    # Test coherence which uses @_non_negative_frequencies decorator
    coherence = c.coherence_magnitude()

    # For odd N=1023, should have (N+1)//2 = 512 frequencies (no Nyquist)
    expected_n_frequencies = (n_fft_samples + 1) // 2
    assert coherence.shape[-3] == expected_n_frequencies, (
        f"Expected {expected_n_frequencies} frequencies, got {coherence.shape[-3]}"
    )


def test_nyquist_frequency_sign_even_n():
    """Test that Nyquist frequency has correct positive sign for even N.

    Regression test for issue where fftfreq() returns negative Nyquist
    for even N, causing frequency axis misalignment in spectrograms.
    """
    from spectral_connectivity.transforms import Multitaper, prepare_time_series

    # Create test signal with even N
    sampling_frequency = 1500
    n_samples = 1000  # Even N
    signal = np.random.randn(n_samples)

    # Transform to get frequencies
    signal_3d = prepare_time_series(signal)
    multitaper = Multitaper(signal_3d, sampling_frequency=sampling_frequency)
    connectivity = Connectivity.from_multitaper(multitaper)

    # Check that all frequencies are non-negative
    freqs = connectivity.frequencies
    assert freqs is not None, "Frequencies should not be None"
    assert len(freqs) == n_samples // 2 + 1, (
        f"Expected {n_samples // 2 + 1} frequencies, got {len(freqs)}"
    )
    assert np.all(freqs >= 0), (
        f"All frequencies should be non-negative, got min={freqs.min()}"
    )

    # Check Nyquist frequency specifically
    nyquist = sampling_frequency / 2
    assert np.isclose(freqs[-1], nyquist), (
        f"Last frequency should be Nyquist ({nyquist} Hz), got {freqs[-1]} Hz"
    )
    assert freqs[-1] > 0, f"Nyquist frequency should be positive, got {freqs[-1]}"


def test_nyquist_frequency_sign_odd_n():
    """Test that frequencies are correct for odd N FFT (no Nyquist bin)."""
    from spectral_connectivity.transforms import Multitaper, prepare_time_series

    # Create test signal and force odd FFT length
    sampling_frequency = 1500
    signal = np.random.randn(1023)  # Will result in odd n_fft_samples

    # Transform to get frequencies
    signal_3d = prepare_time_series(signal)
    multitaper = Multitaper(
        signal_3d, sampling_frequency=sampling_frequency, n_fft_samples=1023
    )
    connectivity = Connectivity.from_multitaper(multitaper)

    # Verify we have odd n_fft_samples
    n_fft = multitaper.n_fft_samples
    assert n_fft % 2 == 1, f"Expected odd n_fft_samples, got {n_fft}"

    # Check frequencies
    freqs = connectivity.frequencies
    assert freqs is not None, "Frequencies should not be None"
    expected_n_freqs = (n_fft + 1) // 2
    assert len(freqs) == expected_n_freqs, (
        f"Expected {expected_n_freqs} frequencies, got {len(freqs)}"
    )
    assert np.all(freqs >= 0), (
        f"All frequencies should be non-negative, got min={freqs.min()}"
    )

    # For odd N, last frequency should be less than Nyquist
    nyquist = sampling_frequency / 2
    assert freqs[-1] < nyquist, (
        f"For odd N, last frequency should be < Nyquist ({nyquist} Hz), got {freqs[-1]} Hz"
    )


def test_spectrogram_frequency_alignment():
    """Test that spectrogram power peaks align with correct frequencies.

    Regression test for frequency axis misalignment where negative Nyquist
    caused spectrograms to show power at wrong frequencies.
    """
    from spectral_connectivity.transforms import Multitaper, prepare_time_series

    # Create signal with known frequency content
    sampling_frequency = 500
    duration = 10
    time = np.arange(0, duration, 1 / sampling_frequency)
    n_time_samples = len(time)

    # Signal with 50 Hz that turns on at t=5s, plus constant 100 Hz
    signal_50 = np.sin(2 * np.pi * time * 50)
    signal_50[: n_time_samples // 2] = 0  # Turn on at t=5s
    signal_100 = np.sin(2 * np.pi * time * 100)
    signal = signal_50 + signal_100

    # Compute spectrogram
    signal_3d = prepare_time_series(signal)
    multitaper = Multitaper(
        signal_3d,
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=1,
        time_window_duration=1.0,
        time_window_step=0.5,
    )
    connectivity = Connectivity.from_multitaper(multitaper)
    power = connectivity.power()

    # Find frequency bins
    freqs = connectivity.frequencies
    freq_50_idx = np.argmin(np.abs(freqs - 50))
    freq_100_idx = np.argmin(np.abs(freqs - 100))

    # Verify frequencies are correct
    assert np.abs(freqs[freq_50_idx] - 50) < 2, (
        f"50 Hz bin at {freqs[freq_50_idx]} Hz, should be ~50 Hz"
    )
    assert np.abs(freqs[freq_100_idx] - 100) < 2, (
        f"100 Hz bin at {freqs[freq_100_idx]} Hz, should be ~100 Hz"
    )

    # Verify power dynamics
    power_50 = power[:, freq_50_idx, 0]
    power_100 = power[:, freq_100_idx, 0]

    # 50 Hz should increase dramatically after t=5s
    power_50_before = power_50[: len(power_50) // 2].mean()
    power_50_after = power_50[len(power_50) // 2 :].mean()
    assert power_50_after > 100 * power_50_before, (
        "50 Hz power should increase >100x after t=5s"
    )

    # 100 Hz should remain constant
    power_100_before = power_100[: len(power_100) // 2].mean()
    power_100_after = power_100[len(power_100) // 2 :].mean()
    ratio = power_100_after / power_100_before
    assert 0.5 < ratio < 2.0, (
        f"100 Hz power should be constant (ratio ~1.0), got {ratio:.2f}"
    )


def _near_singular_fourier(perturbation, seed=999):
    """Build highly correlated (near-singular) Fourier coefficients."""
    rng = np.random.default_rng(seed)
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (1, 10, 1, 5, 3)
    shape = (n_time_samples, n_trials, n_tapers, n_fft_samples)
    fourier_coefficients = np.zeros((*shape, n_signals), dtype=complex)
    base_signal = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    fourier_coefficients[..., 0] = base_signal
    for signal in (1, 2):
        fourier_coefficients[..., signal] = base_signal + perturbation * (
            rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        )
    return fourier_coefficients


def test_mvar_regularized_inverse_ill_conditioned_is_finite():
    """Tikhonov regularization keeps an ill-conditioned (full-rank) inverse finite."""
    conn = Connectivity(fourier_coefficients=_near_singular_fourier(1e-2))
    mvar_coeffs = conn._MVAR_Fourier_coefficients
    assert mvar_coeffs is not None
    assert np.all(np.isfinite(mvar_coeffs))
    assert np.all(np.isfinite(conn._transfer_function))


def test_mvar_rank_deficient_fails_gracefully_without_linalg_error():
    """A rank-deficient input must not raise LinAlgError.

    With near-identical channels the cross-spectral matrix is singular and the
    Wilson decomposition cannot converge; the result is NaN (with a convergence
    warning), but the regularized solve must not crash.
    """
    conn = Connectivity(fourier_coefficients=_near_singular_fourier(1e-10))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mvar_coeffs = conn._MVAR_Fourier_coefficients  # must not raise LinAlgError
        # Downstream directed measures must also not crash; for a rank-deficient
        # input that fails to converge they propagate NaN rather than raising.
        dtf = conn.directed_transfer_function()
    assert mvar_coeffs is not None
    assert dtf is not None


def test_regularized_solve_rhs_matches_batched_lhs():
    """RHS identity passed to xp.linalg.solve must match batched LHS shape.

    NumPy accepts an unbatched (M, M) identity against a batched LHS, but CuPy
    rejects the mismatch and crashes. We assert the contract on CPU so this
    class of bug is caught without GPU CI.
    """
    from spectral_connectivity import connectivity as conn_mod

    real_solve = conn_mod.xp.linalg.solve
    captured = []

    def recording_solve(a, b):
        captured.append((a.shape, b.shape))
        return real_solve(a, b)

    rng = np.random.default_rng(0)
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (2, 2, 2, 4, 3)
    fourier_coefficients = rng.standard_normal(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals)
    ) + 1j * rng.standard_normal(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals)
    )
    conn = Connectivity(fourier_coefficients=fourier_coefficients.astype(complex))

    with patch.object(conn_mod.xp.linalg, "solve", side_effect=recording_solve):
        # Touch both fixed code paths.
        _ = conn._transfer_function  # _estimate_transfer_function
        _ = conn._MVAR_Fourier_coefficients  # _MVAR_Fourier_coefficients

    assert captured, "expected xp.linalg.solve to be called"
    for a_shape, b_shape in captured:
        assert a_shape == b_shape, (
            f"solve received mismatched shapes a={a_shape}, b={b_shape}; "
            "RHS must be broadcast to LHS batched shape for CuPy compatibility"
        )


def test_connectivity_rejects_wrong_ndim():
    """Test that Connectivity rejects inputs with wrong number of dimensions."""
    import pytest

    # Test 1D array
    with pytest.raises(ValueError, match="must be 5-dimensional, got 1D"):
        fourier_1d = np.ones(10, dtype=np.complex128)
        Connectivity(fourier_coefficients=fourier_1d)

    # Test 2D array
    with pytest.raises(ValueError, match="must be 5-dimensional, got 2D"):
        fourier_2d = np.ones((10, 5), dtype=np.complex128)
        Connectivity(fourier_coefficients=fourier_2d)

    # Test 3D array
    with pytest.raises(ValueError, match="must be 5-dimensional, got 3D"):
        fourier_3d = np.ones((10, 5, 2), dtype=np.complex128)
        Connectivity(fourier_coefficients=fourier_3d)

    # Test 4D array
    with pytest.raises(ValueError, match="must be 5-dimensional, got 4D"):
        fourier_4d = np.ones((10, 5, 2, 100), dtype=np.complex128)
        Connectivity(fourier_coefficients=fourier_4d)

    # Test 6D array
    with pytest.raises(ValueError, match="must be 5-dimensional, got 6D"):
        fourier_6d = np.ones((10, 5, 2, 100, 3, 4), dtype=np.complex128)
        Connectivity(fourier_coefficients=fourier_6d)

    # Verify error message contains helpful information
    with pytest.raises(
        ValueError, match=r"Expected shape.*n_time_windows.*n_trials.*n_tapers"
    ):
        fourier_3d = np.ones((10, 5, 2), dtype=np.complex128)
        Connectivity(fourier_coefficients=fourier_3d)

    # Verify error message suggests using Multitaper
    with pytest.raises(ValueError, match="use the Multitaper class"):
        fourier_2d = np.ones((10, 5), dtype=np.complex128)
        Connectivity(fourier_coefficients=fourier_2d)


def test_connectivity_requires_multiple_signals():
    """Test that Connectivity allows single signals for power, but connectivity methods require >= 2."""

    # Single signal is now allowed (for power spectral density)
    fourier_1_signal = np.ones((2, 2, 2, 100, 1), dtype=np.complex128)
    conn = Connectivity(fourier_coefficients=fourier_1_signal)
    assert conn.fourier_coefficients.shape[-1] == 1

    # Power computation should work for single signal
    power = conn.power()
    assert power.shape[-1] == 1

    # Verify that 2 signals is accepted
    fourier_2_signals = np.ones((2, 2, 2, 100, 2), dtype=np.complex128)
    conn = Connectivity(fourier_coefficients=fourier_2_signals)
    assert conn.fourier_coefficients.shape[-1] == 2


def test_connectivity_warns_on_nan():
    """Test that Connectivity warns when fourier_coefficients contains NaN or Inf."""
    import warnings

    # Test NaN values
    fourier_with_nan = np.ones((2, 2, 2, 100, 2), dtype=np.complex128)
    fourier_with_nan[0, 0, 0, 0, 0] = np.nan

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Connectivity(fourier_coefficients=fourier_with_nan)
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert "NaN or Inf values" in str(w[0].message)
        assert "Check your input data" in str(w[0].message)

    # Test Inf values
    fourier_with_inf = np.ones((2, 2, 2, 100, 2), dtype=np.complex128)
    fourier_with_inf[0, 0, 0, 0, 0] = np.inf

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Connectivity(fourier_coefficients=fourier_with_inf)
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert "NaN or Inf values" in str(w[0].message)

    # Test complex Inf values
    fourier_with_complex_inf = np.ones((2, 2, 2, 100, 2), dtype=np.complex128)
    fourier_with_complex_inf[0, 0, 0, 0, 0] = complex(np.inf, 1.0)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Connectivity(fourier_coefficients=fourier_with_complex_inf)
        assert len(w) == 1
        assert "NaN or Inf values" in str(w[0].message)
        # Check for actionable suggestions
        assert "interpolating" in str(w[0].message) or "artifact removal" in str(
            w[0].message
        )

    # Test valid data (no warning)
    fourier_valid = np.ones((2, 2, 2, 100, 2), dtype=np.complex128)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Connectivity(fourier_coefficients=fourier_valid)
        # Filter out any warnings that are not from Connectivity
        connectivity_warnings = [
            warning for warning in w if "fourier_coefficients" in str(warning.message)
        ]
        assert len(connectivity_warnings) == 0


def test_reduced_cross_spectral_matrix_matches_outer_product():
    """The reduced (batched-matmul) CSM matches the full outer-product mean.

    ``_expectation_cross_spectral_matrix`` contracts the averaged observation
    axes directly instead of materializing the per-observation outer product. It
    must agree, to floating-point tolerance, with the explicit
    ``self._expectation(self._cross_spectral_matrix)`` for every expectation
    type, and must propagate NaNs the same way.
    """
    n_time_windows, n_trials, n_tapers, n_frequencies, n_signals = 4, 6, 5, 32, 4
    rng = np.random.default_rng(7)
    shape = (n_time_windows, n_trials, n_tapers, n_frequencies, n_signals)
    fourier_coefficients = (
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    ).astype(np.complex128)

    for expectation_type in [
        "time",
        "trials",
        "tapers",
        "time_trials",
        "time_tapers",
        "trials_tapers",
        "time_trials_tapers",
    ]:
        conn = Connectivity(
            fourier_coefficients=fourier_coefficients,
            expectation_type=expectation_type,
        )
        reduced = conn._expectation_cross_spectral_matrix()
        reference = conn._expectation(conn._cross_spectral_matrix)
        assert reduced.shape == reference.shape, expectation_type
        np.testing.assert_allclose(
            reduced, reference, rtol=1e-10, atol=1e-12, err_msg=expectation_type
        )

    # NaN in one observation/signal must poison the same rows/columns as the
    # explicit outer-product mean would.
    nan_coefficients = fourier_coefficients.copy()
    nan_coefficients[0, 0, 0, 0, 1] = np.nan
    conn = Connectivity(
        fourier_coefficients=nan_coefficients, expectation_type="trials_tapers"
    )
    reduced = conn._expectation_cross_spectral_matrix()
    reference = conn._expectation(conn._cross_spectral_matrix)
    np.testing.assert_array_equal(np.isnan(reduced), np.isnan(reference))


def _reference_normalized_cross_spectrum(conn):
    """Honest per-observation phase-locking cross-spectrum (materialized).

    Normalizes each per-observation cross-spectrum entry ``z_i conj(z_j)`` by its
    magnitude and averages, then restricts to non-negative frequencies -- the
    original implementation the factorized ``_phase_locking_value`` replaced.
    """
    csm = np.asarray(conn._cross_spectral_matrix)
    magnitude = np.abs(csm)
    with np.errstate(invalid="ignore", divide="ignore"):
        normalized = csm / magnitude
    normalized[magnitude == 0] = np.nan
    reduced = np.asarray(conn._expectation(normalized))
    return reduced[..., : reduced.shape[-3] // 2 + 1, :, :]


@mark.parametrize("dtype", [np.complex64, np.complex128])
@mark.parametrize("dead", [False, True])
@mark.parametrize(
    "expectation_type",
    [
        "time",
        "trials",
        "tapers",
        "time_trials",
        "time_tapers",
        "trials_tapers",
        "time_trials_tapers",
    ],
)
def test_phase_locking_value_matches_per_observation_reference(
    dtype, dead, expectation_type
):
    """Factorized PLV/PPC equal the materialized per-observation reference.

    ``phase_locking_value`` now unit-normalizes each Fourier coefficient and
    reuses the batched reduced cross-spectral matmul, using
    ``(z_i conj(z_j)) / |z_i conj(z_j)| = (z_i/|z_i|) conj(z_j/|z_j|)``. It must
    match the honest per-observation normalization-then-average across every
    expectation mode, both dtypes, and a dead (all-zero) channel that makes the
    normalization undefined (NaN) -- the case ``blocks`` used to serve.
    """
    rng = np.random.default_rng(0)
    shape = (3, 4, 5, 8, 4)
    fc = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dtype)
    if dead:
        fc[:, 1, :, :, 0] = 0.0  # dead channel 0 on trial 1

    conn = Connectivity(fourier_coefficients=fc, expectation_type=expectation_type)
    tol = (
        {"rtol": 2e-6, "atol": 2e-6}
        if dtype == np.complex64
        else {"rtol": 1e-9, "atol": 1e-11}
    )

    ref_complex = _reference_normalized_cross_spectrum(conn)
    ref_plv = np.abs(ref_complex)
    plv = conn.phase_locking_value()
    raw = np.asarray(conn._phase_locking_value())

    # NaN placement identical, values equal off the NaNs.
    np.testing.assert_array_equal(np.isnan(plv), np.isnan(ref_plv))
    np.testing.assert_array_equal(np.isnan(raw), np.isnan(ref_complex))
    finite = ~np.isnan(ref_plv)
    np.testing.assert_allclose(plv[finite], ref_plv[finite], **tol)
    np.testing.assert_allclose(raw[finite], ref_complex[finite], **tol)
    # Diagonal (self-consistency) is 1 where defined.
    diag = np.diagonal(plv, axis1=-2, axis2=-1)
    np.testing.assert_allclose(diag[~np.isnan(diag)], 1.0, **tol)

    # Pairwise phase consistency built from the same complex reference.
    n = conn.n_observations
    if n >= 2:
        plv_sum = ref_complex * n
        ref_ppc = ((plv_sum * plv_sum.conjugate() - n) / (n**2 - n)).real
        ppc = conn.pairwise_phase_consistency()
        np.testing.assert_array_equal(np.isnan(ppc), np.isnan(ref_ppc))
        fppc = ~np.isnan(ref_ppc)
        np.testing.assert_allclose(ppc[fppc], ref_ppc[fppc], **tol)
    else:
        with pytest.raises(ValueError, match="at least 2 observations"):
            conn.pairwise_phase_consistency()


def test_default_coordinates_created_when_omitted():
    """The constructor must populate frequencies/time defaults, as documented.

    Directly constructing Connectivity without frequencies/time should yield
    normalized frequencies and integer time indices rather than None, so that
    coordinate-dependent methods (e.g. group_delay) do not crash.
    """
    n_time_windows, n_trials, n_tapers, n_fft_samples, n_signals = 3, 1, 2, 8, 2
    rng = np.random.default_rng(0)
    fourier_coefficients = rng.standard_normal(
        (n_time_windows, n_trials, n_tapers, n_fft_samples, n_signals)
    ) + 1j * rng.standard_normal(
        (n_time_windows, n_trials, n_tapers, n_fft_samples, n_signals)
    )

    conn = Connectivity(fourier_coefficients=fourier_coefficients)

    assert conn.frequencies is not None
    assert conn.time is not None
    # Normalized frequencies: non-negative half of fftfreq(n_fft_samples).
    np.testing.assert_allclose(
        conn.frequencies,
        np.abs(np.fft.fftfreq(n_fft_samples))[: n_fft_samples // 2 + 1],
    )
    np.testing.assert_array_equal(conn.time, np.arange(n_time_windows))
    # A coordinate-dependent method must not raise.
    conn.group_delay()


@mark.parametrize(
    "measure",
    [
        "directed_transfer_function",
        "directed_coherence",
        "partial_directed_coherence",
        "generalized_partial_directed_coherence",
        "direct_directed_transfer_function",
    ],
)
def test_directed_measures_no_runtime_warning_on_nan_transfer_function(measure):
    """Directed measures must propagate NaN silently, not emit divide warnings.

    Force Wilson non-convergence with ``minimum_phase_max_iterations=1`` so the
    transfer function is deterministically NaN; normalizing by the
    inflow/outflow sum would emit ``invalid value encountered in divide`` unless
    scoped. The NaN is already reported via the convergence UserWarning.
    """
    from spectral_connectivity.transforms import Multitaper

    rng = np.random.default_rng(0)
    time_series = rng.standard_normal((256, 4, 3))
    conn = Connectivity.from_multitaper(
        Multitaper(time_series, sampling_frequency=256, time_halfbandwidth_product=3),
        minimum_phase_max_iterations=1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("error", RuntimeWarning)
        result = getattr(conn, measure)()
    # The forced non-convergence yields NaN; the point is it did so without a
    # RuntimeWarning (which would have been raised as an error above).
    assert np.isnan(result).any()


def test_power_one_sided_preserves_total_power():
    """The one-sided PSD must double interior bins to conserve total power.

    Summing the returned one-sided power (over frequency) must equal summing the
    full two-sided spectrum; otherwise integrating power() recovers only half
    the variance.
    """
    from spectral_connectivity.transforms import Multitaper

    rng = np.random.default_rng(0)
    for n_time in (512, 511):  # even and odd FFT lengths
        time_series = rng.standard_normal((n_time, 4, 2))
        conn = Connectivity.from_multitaper(
            Multitaper(time_series, sampling_frequency=256)
        )
        one_sided = conn.power()  # non-negative frequencies, interior doubled
        two_sided = conn._power  # full spectrum
        np.testing.assert_allclose(
            one_sided.sum(axis=-2), two_sided.sum(axis=-2), rtol=1e-10
        )


def test_power_preserves_float32_dtype():
    """power() must not upcast a float32 (complex64) spectrum to float64.

    Regression: the one-sided doubling multiplied by a float64 scale array,
    silently widening a complex64 spectrum back to float64 and defeating the
    memory/precision choice. The scale now matches the spectrum dtype.
    """
    rng = np.random.default_rng(0)
    fourier_coefficients = (
        rng.standard_normal((2, 3, 2, 8, 2)) + 1j * rng.standard_normal((2, 3, 2, 8, 2))
    ).astype(np.complex64)
    conn = Connectivity(fourier_coefficients=fourier_coefficients)
    assert conn._power.dtype == np.float32  # spectrum is genuinely float32
    assert conn.power().dtype == np.float32  # and power() must not widen it


def test_phase_slope_index_uses_adjacent_frequency_bins():
    """PSI must sum conj(C(f)) * C(f + df) over adjacent bins (Nolte 2008).

    The previous implementation summed over all i<j frequency-pair combinations,
    which is a different statistic. This checks the public result equals the
    adjacent-bin reference and differs from the all-pairs sum.
    """
    from spectral_connectivity.transforms import Multitaper

    rng = np.random.default_rng(0)
    time_series = rng.standard_normal((256, 12, 2))
    conn = Connectivity.from_multitaper(
        Multitaper(time_series, sampling_frequency=200, time_halfbandwidth_product=3)
    )

    # Reproduce the internal subsampling to build a faithful reference.
    frequencies = conn.frequencies
    bandpassed, band_freqs = _bandpass(conn.coherency(), frequencies, None)
    step = _get_independent_frequency_step(frequencies[1] - frequencies[0], None)
    idx = np.arange(0, band_freqs.shape[0], step)
    bandpassed = bandpassed[..., idx, :, :]

    adjacent_ref = (
        (np.conj(bandpassed[..., :-1, :, :]) * bandpassed[..., 1:, :, :])
        .sum(axis=-3)
        .imag
    )
    # All-pairs sum (the previous, incorrect statistic) for contrast.
    from itertools import combinations

    pair_index = np.array(list(combinations(range(bandpassed.shape[-3]), 2)))
    all_pairs_ref = (
        (
            np.conj(bandpassed[..., pair_index[:, 0], :, :])
            * bandpassed[..., pair_index[:, 1], :, :]
        )
        .sum(axis=-3)
        .imag
    )

    psi = conn.phase_slope_index()
    np.testing.assert_allclose(psi, adjacent_ref, atol=1e-12)
    # The two references genuinely differ, so this discriminates the fix.
    assert not np.allclose(adjacent_ref, all_pairs_ref)
    # PSI is antisymmetric in the signal pair.
    np.testing.assert_allclose(psi[..., 0, 1], -psi[..., 1, 0], atol=1e-12)


def _correlated_fixture():
    """Ordinary near-singular LFP-like fixture (highly correlated channels)."""
    from spectral_connectivity.transforms import Multitaper

    rng = np.random.default_rng(42)
    n_time, n_trials, n_signals, sf = 100, 5, 3, 500
    t = np.arange(n_time) / sf
    base = np.sin(2 * np.pi * 10 * t)
    sig = np.zeros((n_time, n_trials, n_signals))
    for k in range(n_trials):
        sig[:, k, 0] = base + 0.1 * rng.standard_normal(n_time)
        sig[:, k, 1] = np.sin(
            2 * np.pi * 10 * t + np.pi / 4
        ) + 0.1 * rng.standard_normal(n_time)
        sig[:, k, 2] = 0.1 * base + 0.9 * rng.standard_normal(n_time)
    return Multitaper(
        sig, sampling_frequency=sf, time_halfbandwidth_product=2, n_tapers=3
    )


@mark.parametrize(
    "measure",
    [
        "directed_transfer_function",
        "partial_directed_coherence",
        "pairwise_spectral_granger_prediction",
    ],
)
def test_directed_measures_finite_on_ordinary_correlated_data(measure):
    """Directed measures must return finite values for realistic correlated data.

    Regression: the Wilson relative-tolerance change with the old
    max_iterations=60 returned an all-NaN minimum-phase factor for near-singular
    cross-spectral matrices (highly correlated channels), which the range tests
    accepted vacuously. This is a non-vacuous finiteness check.
    """
    conn = Connectivity.from_multitaper(_correlated_fixture())
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # no non-convergence warning
        result = getattr(conn, measure)()
    assert np.isfinite(result).mean() > 0.5


def test_minimum_phase_max_iterations_is_configurable():
    """Users can raise max_iterations to recover from Wilson non-convergence."""
    import warnings

    m = _correlated_fixture()
    conn_low = Connectivity.from_multitaper(m, minimum_phase_max_iterations=1)
    with pytest.warns(UserWarning, match="did not converge"):
        dtf_low = conn_low.directed_transfer_function()
    assert np.isnan(dtf_low).all()

    conn_high = Connectivity.from_multitaper(m, minimum_phase_max_iterations=500)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        dtf_high = conn_high.directed_transfer_function()
    assert np.isfinite(dtf_high).mean() > 0.5


@mark.parametrize(
    "measure",
    ["directed_transfer_function", "pairwise_spectral_granger_prediction"],
)
def test_directed_measures_are_scale_invariant(measure):
    """Rescaling the signal must not change these scale-invariant measures.

    Regression: the Tikhonov diagonal loading used lambda proportional to
    mean(|H|**2) (amplitude-squared units) added to H (amplitude units), so the
    regularization strength was not scale-covariant. Rescaling the signal by a
    large factor then changed the transfer-function inverse and shifted DTF /
    spectral Granger by orders of magnitude. lambda now scales with the RMS
    magnitude, restoring invariance.
    """
    from spectral_connectivity.simulate import simulate_MVAR
    from spectral_connectivity.transforms import Multitaper

    coeffs = np.array([[[0.5, 0.3], [0.0, 0.4]], [[-0.2, 0.0], [0.1, -0.3]]])
    ts = simulate_MVAR(
        coeffs,
        noise_covariance=np.eye(2),
        n_time_samples=600,
        n_trials=8,
        random_state=np.random.default_rng(0),
    )

    def compute(scale):
        m = Multitaper(
            ts * scale,
            sampling_frequency=1.0,
            time_halfbandwidth_product=3,
            start_time=0,
        )
        return getattr(Connectivity.from_multitaper(m), measure)()

    base = compute(1.0)
    scaled = compute(1e12)
    finite = np.isfinite(base) & np.isfinite(scaled)
    assert finite.any()
    np.testing.assert_allclose(scaled[finite], base[finite], rtol=1e-6, atol=1e-9)


def test_reassigning_different_geometry_resets_coordinates():
    """A geometry change on reassignment must reset (not silently keep) coords.

    Reassigning 8 -> 10 FFT bins previously left 5 frequency coordinates for 6
    power bins, so phase_slope_index silently dropped a bin and group_delay
    could fail. The coordinates are now reset to the new geometry with a warning.
    """
    rng = np.random.default_rng(0)
    fc8 = rng.standard_normal((2, 3, 2, 8, 2)) + 1j * rng.standard_normal(
        (2, 3, 2, 8, 2)
    )
    conn = Connectivity(fourier_coefficients=fc8)
    assert conn.frequencies.size == 8 // 2 + 1
    assert conn.time.size == 2

    fc10 = rng.standard_normal((3, 3, 2, 10, 2)) + 1j * rng.standard_normal(
        (3, 3, 2, 10, 2)
    )
    with pytest.warns(UserWarning, match="changed the FFT/time geometry"):
        conn.fourier_coefficients = fc10
    assert conn.frequencies.size == 10 // 2 + 1
    assert conn.time.size == 3
    # Coordinate-dependent methods are now consistent (no silent bin drop).
    assert conn.group_delay()[0].shape[-2:] == (2, 2)


def test_reassigning_same_geometry_keeps_coordinates_without_warning():
    """Same-geometry reassignment (the reuse pattern) must not warn or reset."""
    rng = np.random.default_rng(0)
    fc = rng.standard_normal((2, 3, 2, 8, 2)) + 1j * rng.standard_normal(
        (2, 3, 2, 8, 2)
    )
    conn = Connectivity(fourier_coefficients=fc)
    freqs_before = conn.frequencies.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # no geometry warning
        conn.fourier_coefficients = fc
    np.testing.assert_array_equal(conn.frequencies, freqs_before)


def test_global_coherence_sparse_branch_orders_strongest_first():
    """global_coherence must order components strongest-first regardless of the
    order svds returns (which SciPy does not guarantee)."""
    from unittest.mock import patch

    from spectral_connectivity import connectivity as conn_mod

    real_svds = conn_mod.svds

    def ascending_svds(matrix, k):
        u, s, vh = real_svds(matrix, k)
        order = np.argsort(s)  # force ascending
        return u[:, order], s[order], vh[order]

    def descending_svds(matrix, k):
        u, s, vh = real_svds(matrix, k)
        order = np.argsort(s)[::-1]  # force descending
        return u[:, order], s[order], vh[order]

    rng = np.random.default_rng(0)
    fc = rng.standard_normal((1, 8, 1, 4, 6)) + 1j * rng.standard_normal(
        (1, 8, 1, 4, 6)
    )
    # Force the per-bin svds fallback (the moderate-n_signals default is the
    # batched eigendecomposition, which never calls svds) so the mock takes
    # effect and this exercises the svds ordering logic it is written for.
    with patch.object(conn_mod, "GLOBAL_COHERENCE_MAX_DENSE_COMPONENTS", 1):
        with patch.object(conn_mod, "svds", ascending_svds):
            gc_asc, _ = Connectivity(fourier_coefficients=fc).global_coherence(
                max_rank=3
            )
        with patch.object(conn_mod, "svds", descending_svds):
            gc_desc, _ = Connectivity(fourier_coefficients=fc).global_coherence(
                max_rank=3
            )
    # Same result regardless of the order svds returned, and strongest-first.
    np.testing.assert_allclose(gc_asc, gc_desc)
    assert np.all(gc_asc[..., 0] >= gc_asc[..., 1] - 1e-9)
    assert np.all(gc_asc[..., 1] >= gc_asc[..., 2] - 1e-9)


def test_global_coherence_batched_matches_per_bin_fallback():
    """The batched path matches the per-bin svds/svd path, thin and wide.

    global_coherence batches over bins with an ``eigh`` of the cross-spectral
    matrix (``n_estimates >= n_signals``) or the economy SVD of the thin matrix
    (``n_estimates < n_signals``), falling back to a per-bin loop for large
    square matrices. All must agree on the coherence fractions (the vectors need
    not agree: they are defined only up to a per-component phase, or an arbitrary
    unitary rotation within a degenerate subspace), including NaN placement for
    zero-power bins.
    """
    from unittest.mock import patch

    from spectral_connectivity import connectivity as conn_mod

    rng = np.random.default_rng(4)
    # wide: n_estimates (30) >= n_signals (8) -> eigh path
    # thin: n_estimates (2) <  n_signals (8) -> economy SVD path
    for shape in [(2, 10, 3, 20, 8), (2, 1, 2, 20, 8)]:
        fc = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(
            np.complex128
        )
        fc = fc.copy()
        fc[0, :, :, 5, :] = 0.0  # a zero-power bin -> NaN on both paths
        max_available = min(shape[4], shape[1] * shape[2])

        for max_rank in (1, min(3, max_available)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gc_batched, _ = Connectivity(fc).global_coherence(max_rank=max_rank)
                # Force the per-bin fallback by lowering the batching threshold.
                with patch.object(conn_mod, "GLOBAL_COHERENCE_MAX_DENSE_COMPONENTS", 0):
                    gc_loop, _ = Connectivity(fc).global_coherence(max_rank=max_rank)
            np.testing.assert_array_equal(np.isnan(gc_batched), np.isnan(gc_loop))
            np.testing.assert_allclose(
                gc_batched, gc_loop, rtol=1e-9, atol=1e-11, equal_nan=True
            )
            assert np.all(gc_batched[~np.isnan(gc_batched)] >= 0)
            assert np.all(gc_batched[~np.isnan(gc_batched)] <= 1)


def test_global_coherence_batched_matches_per_bin_ill_conditioned():
    """Batched eigh and per-bin SVD agree even for near-duplicate channels.

    The batched path diagonalizes ``A @ Aᴴ`` with ``eigh``, which squares the
    condition number relative to the per-bin ``svd(A)``. For a nearly
    rank-deficient cross-spectral matrix (near-duplicate channels) the weakest
    components can lose relative precision, but the coherence fractions must
    still agree to a tight absolute tolerance, and the dominant component -- the
    usual use of this measure -- must agree closely. This guards the documented
    eigh/SVD tradeoff against a regression that widens the gap; the existing
    equivalence test uses only well-conditioned Gaussian data.
    """
    from unittest.mock import patch

    from spectral_connectivity import connectivity as conn_mod

    rng = np.random.default_rng(20240827)
    n_time, n_trials, n_tapers, n_fft, n_signals = 2, 30, 2, 10, 4
    # Near-duplicate channels: a shared complex component broadcast across all
    # signals plus a tiny (1e-6) per-channel perturbation. The resulting per-bin
    # cross-spectral matrix is nearly rank-one (condition number ~1e6+), the
    # regime where eigh(A @ Aᴴ) and svd(A) diverge most.
    shared = rng.standard_normal(
        (n_time, n_trials, n_tapers, n_fft, 1)
    ) + 1j * rng.standard_normal((n_time, n_trials, n_tapers, n_fft, 1))
    perturbation = 1e-6 * (
        rng.standard_normal((n_time, n_trials, n_tapers, n_fft, n_signals))
        + 1j * rng.standard_normal((n_time, n_trials, n_tapers, n_fft, n_signals))
    )
    fc = (shared + perturbation).astype(np.complex128)

    for max_rank in (1, n_signals):
        gc_batched, _ = Connectivity(fc).global_coherence(max_rank=max_rank)
        # Force the per-bin svd/svds fallback (the well-conditioned reference).
        with patch.object(conn_mod, "GLOBAL_COHERENCE_MAX_DENSE_COMPONENTS", 0):
            gc_loop, _ = Connectivity(fc).global_coherence(max_rank=max_rank)

        np.testing.assert_array_equal(np.isnan(gc_batched), np.isnan(gc_loop))
        # Weak components may differ in relative terms but are ~machine-epsilon
        # in absolute terms, so compare all components on an absolute tolerance.
        np.testing.assert_allclose(gc_batched, gc_loop, atol=1e-8, equal_nan=True)
        # The dominant component holds essentially all the coherent power for
        # near-duplicate channels and must agree closely in relative terms.
        np.testing.assert_allclose(gc_batched[..., 0], gc_loop[..., 0], rtol=1e-6)
        assert np.all(gc_batched[..., 0] > 0.99)


def test_global_coherence_batched_chunking_matches_single_chunk():
    """The multi-chunk path matches processing all bins in one chunk.

    The default element cap keeps every test's bins in a single chunk, so the
    partial-last-chunk reshape/strided-write logic is otherwise unexercised.
    Force a tiny budget so bins are processed in several chunks (with a zero-power
    bin near a boundary) and require an identical result.
    """
    rng = np.random.default_rng(7)
    shape = (2, 10, 3, 20, 6)
    fc = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(
        np.complex128
    )
    fc = fc.copy()
    fc[0, :, :, 7, :] = 0.0  # zero-power bin (-> NaN), placed to straddle chunks

    gc_single, vec_single = Connectivity(fc).global_coherence(max_rank=2)
    # Budget chosen so `chunk` is only a few bins, forcing multiple iterations and
    # a partial final chunk (20 frequency bins do not divide evenly).
    gc_multi, vec_multi = Connectivity(fc).global_coherence(
        max_rank=2, max_workspace_elements=6 * 6 * 3
    )

    np.testing.assert_array_equal(np.isnan(gc_single), np.isnan(gc_multi))
    np.testing.assert_allclose(gc_single, gc_multi, equal_nan=True)
    np.testing.assert_allclose(vec_single, vec_multi, equal_nan=True)


def test_global_coherence_workspace_budget_is_configurable_and_result_invariant():
    """max_workspace_elements bounds peak memory without changing the result.

    Chunking is a memory-only concern: a tiny budget (many small chunks) and a
    huge budget (a single chunk) must yield the same fractions and vectors, and a
    non-positive budget must be rejected.
    """
    rng = np.random.default_rng(31)
    shape = (2, 10, 3, 20, 6)
    fc = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(
        np.complex128
    )
    c = Connectivity(fc)

    gc_default, vec_default = c.global_coherence(max_rank=2)
    gc_tiny, vec_tiny = c.global_coherence(max_rank=2, max_workspace_elements=5_000)
    gc_huge, vec_huge = c.global_coherence(max_rank=2, max_workspace_elements=10**9)

    np.testing.assert_array_equal(gc_default, gc_tiny)
    np.testing.assert_array_equal(gc_default, gc_huge)
    np.testing.assert_array_equal(vec_default, vec_tiny)
    np.testing.assert_array_equal(vec_default, vec_huge)

    # Must be a genuine positive integer: reject non-positive, fractional,
    # non-finite, and boolean budgets (a float would corrupt the chunk size and
    # crash later inside range(); bool is an int subclass).
    for bad in [0, -1, 2.5, np.nan, np.inf, True]:
        with pytest.raises(
            ValueError, match="max_workspace_elements must be a positive"
        ):
            c.global_coherence(max_workspace_elements=bad)


def test_global_coherence_vectors_are_orthonormal_eigenvectors():
    """The returned vectors are unit-norm eigenvectors, even ill-conditioned.

    The coherence vectors are a documented output but are only checked for shape
    elsewhere. Verify (including near-duplicate, ill-conditioned channels) that
    each returned vector has unit norm and that the leading vector is an
    eigenvector of the per-bin scaled cross-spectral matrix.
    """
    rng = np.random.default_rng(8)
    n_time, n_trials, n_tapers, n_fft, n_signals = 1, 12, 2, 8, 4
    base = rng.standard_normal(
        (n_time, n_trials, n_tapers, n_fft)
    ) + 1j * rng.standard_normal((n_time, n_trials, n_tapers, n_fft))
    fc = np.zeros((n_time, n_trials, n_tapers, n_fft, n_signals), dtype=complex)
    fc[..., 0] = base
    fc[..., 1] = base * (1 + 1e-10)  # near-duplicate -> ill-conditioned
    fc[..., 2] = 0.5 * base + 0.01 * (
        rng.standard_normal(base.shape) + 1j * rng.standard_normal(base.shape)
    )
    fc[..., 3] = rng.standard_normal(base.shape) + 1j * rng.standard_normal(base.shape)

    _gc, vectors = Connectivity(fc).global_coherence(max_rank=2)

    norms = np.linalg.norm(vectors, axis=-2)
    np.testing.assert_allclose(norms, 1.0, atol=1e-8)

    # Reconstruct the per-bin scaled cross-spectral matrix and check the leading
    # vector is an eigenvector (parallel to C @ v0).
    observations = fc.transpose(0, 3, 4, 1, 2).reshape(
        n_time * n_fft, n_signals, n_trials * n_tapers
    )
    scaled = observations / np.max(np.abs(observations), axis=(-2, -1), keepdims=True)
    cross_spectral = scaled @ np.conj(scaled).swapaxes(-1, -2)
    leading = vectors.reshape(n_time * n_fft, n_signals, 2)[:, :, 0]
    projected = np.einsum("bij,bj->bi", cross_spectral, leading)
    alignment = np.abs(np.sum(np.conj(leading) * projected, axis=-1)) / (
        np.linalg.norm(leading, axis=-1) * np.linalg.norm(projected, axis=-1)
    )
    np.testing.assert_allclose(alignment, 1.0, atol=1e-6)


def test_phase_slope_index_raises_with_fewer_than_two_bins():
    """Fewer than 2 frequency bins in the band must raise, not return a false 0."""
    from spectral_connectivity.transforms import Multitaper

    rng = np.random.default_rng(0)
    conn = Connectivity.from_multitaper(
        Multitaper(
            rng.standard_normal((256, 6, 2)),
            sampling_frequency=200,
            time_halfbandwidth_product=3,
        )
    )
    # A huge frequency_resolution subsamples the band to a single bin.
    with pytest.raises(ValueError, match="at least 2 frequency bins"):
        conn.phase_slope_index(frequency_resolution=1e6)


def test_connectivity_rejects_mismatched_coordinate_lengths():
    """Supplied frequencies/time must match the data geometry."""
    fc = np.zeros((3, 2, 1, 8, 2), dtype=complex)  # n_time=3, n_fft=8
    with pytest.raises(ValueError, match="frequencies must have length"):
        Connectivity(fourier_coefficients=fc, frequencies=np.arange(4))
    with pytest.raises(ValueError, match="time must have length"):
        Connectivity(fourier_coefficients=fc, time=np.arange(5))


@mark.parametrize("bad_resolution", [0.0, -1.0, np.nan, np.inf])
def test_frequency_resolution_must_be_finite_positive(bad_resolution):
    """delay/phase_slope_index reject an invalid frequency_resolution."""
    from spectral_connectivity.transforms import Multitaper

    conn = Connectivity.from_multitaper(
        Multitaper(
            np.random.default_rng(0).standard_normal((256, 6, 2)),
            sampling_frequency=200,
            time_halfbandwidth_product=3,
        )
    )
    with pytest.raises(ValueError, match="frequency_resolution must be a finite"):
        conn.phase_slope_index(frequency_resolution=bad_resolution)


def test_connectivity_rejects_non_1d_or_nonfinite_coordinates():
    """Coordinates must be exactly 1-D and finite, not merely the right length."""
    fc = np.zeros((3, 2, 1, 8, 2), dtype=complex)  # n_time=3, n_fft=8
    # (n, 1) frequency array has the right len but wrong shape.
    with pytest.raises(ValueError, match=r"frequencies must be a 1-D array"):
        Connectivity(fourier_coefficients=fc, frequencies=np.zeros((8, 1)))
    with pytest.raises(ValueError, match=r"time must be a 1-D array"):
        Connectivity(fourier_coefficients=fc, time=np.zeros((3, 1)))
    # Non-finite coordinate.
    bad_freqs = np.linspace(0, 1, 8)
    bad_freqs[2] = np.nan
    with pytest.raises(ValueError, match="frequencies must contain only finite"):
        Connectivity(fourier_coefficients=fc, frequencies=bad_freqs)


@mark.parametrize("measure", ["delay", "group_delay", "phase_slope_index"])
def test_single_frequency_bin_raises_clear_error(measure):
    """One frequency bin must raise a clear ValueError, not a raw IndexError."""
    # n_fft_samples = 1 -> a single non-negative frequency bin.
    fc = np.ones((2, 3, 2, 1, 2), dtype=complex)
    conn = Connectivity(fourier_coefficients=fc)
    assert conn.frequencies.size == 1
    with pytest.raises(ValueError, match="at least 2 frequency bins"):
        getattr(conn, measure)()


@mark.parametrize("measure", ["phase_locking_value", "pairwise_phase_consistency"])
def test_phase_locking_zero_power_is_nan_without_runtime_warning(measure):
    """A dead (zero) channel yields NaN with a UserWarning, not a RuntimeWarning."""
    import warnings

    fc = np.ones((1, 5, 2, 4, 2), dtype=complex)
    fc[..., 1] = 0.0  # dead second channel
    conn = Connectivity(fourier_coefficients=fc)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # no leaked divide warning
        with pytest.warns(UserWarning, match="zero magnitude"):
            result = getattr(conn, measure)()
    assert np.isnan(result).any()


def test_from_multitaper_connectivity_is_picklable():
    """A Connectivity from from_multitaper must pickle despite the provenance weakref.

    from_multitaper stores a weakref to its source Multitaper for provenance
    verification; weakrefs are not picklable, so __getstate__ drops it. The
    restored instance must round-trip and compute identical results (its
    provenance link is gone, as a live-object link cannot survive serialization).
    """
    import pickle

    rng = np.random.default_rng(0)
    m = Multitaper(rng.standard_normal((128, 2, 3)), sampling_frequency=500)
    conn = Connectivity.from_multitaper(m)

    restored = pickle.loads(pickle.dumps(conn))
    np.testing.assert_allclose(
        restored.coherence_magnitude(), conn.coherence_magnitude()
    )
    assert restored._source_multitaper is None


class _SlottedConnectivity(Connectivity):
    """Module-level slotted subclass so pickle can reference it by qualified name."""

    __slots__ = ("extra_metadata",)


class _StringSlottedConnectivity(Connectivity):
    """A subclass whose ``__slots__`` is a bare string (a single slot name)."""

    __slots__ = "extra_metadata"


@mark.parametrize("subclass", [_SlottedConnectivity, _StringSlottedConnectivity])
def test_pickle_and_copy_preserve_subclass_slots(subclass):
    """__getstate__/__setstate__ must not drop subclass __slots__ state.

    The base Connectivity keeps its state in __dict__, but a subclass may declare
    __slots__ (as a tuple *or* a bare string). Returning only __dict__ would
    silently lose those attributes on pickle / copy.copy / copy.deepcopy; the
    (dict, slots) state pair preserves them while still clearing the unpicklable
    provenance weakref.
    """
    import copy
    import pickle

    rng = np.random.default_rng(1)
    m = Multitaper(rng.standard_normal((128, 2, 3)), sampling_frequency=500)
    conn = subclass.from_multitaper(m)
    conn.extra_metadata = {"subject": "s1"}

    for clone in (
        pickle.loads(pickle.dumps(conn)),
        copy.copy(conn),
        copy.deepcopy(conn),
    ):
        assert clone.extra_metadata == {"subject": "s1"}
        np.testing.assert_allclose(
            clone.coherence_magnitude(), conn.coherence_magnitude()
        )
        assert clone._source_multitaper is None


def test_setstate_accepts_legacy_dict_state():
    """A plain-dict pickle (from before __getstate__ existed) must still load.

    Earlier releases had no custom __getstate__, so their pickles store a plain
    __dict__ rather than the new (dict, slots) pair. __setstate__ must accept
    that legacy form and initialize the new provenance fields to None instead of
    raising ``ValueError: too many values to unpack``.
    """
    rng = np.random.default_rng(2)
    fc = (
        rng.standard_normal((2, 3, 4, 8, 3)) + 1j * rng.standard_normal((2, 3, 4, 8, 3))
    ).astype(np.complex128)
    conn = Connectivity(fourier_coefficients=fc)

    # Emulate a legacy pickle payload: a plain __dict__ without provenance fields.
    legacy_state = dict(conn.__dict__)
    legacy_state.pop("_source_multitaper", None)
    legacy_state.pop("_source_parameters", None)

    restored = Connectivity.__new__(Connectivity)
    restored.__setstate__(legacy_state)  # plain dict, as older versions pickled
    assert restored._source_multitaper is None
    assert restored._source_parameters is None
    np.testing.assert_allclose(
        restored.coherence_magnitude(), conn.coherence_magnitude()
    )
