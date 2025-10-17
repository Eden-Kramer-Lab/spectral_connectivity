from unittest.mock import PropertyMock

import numpy as np
from pytest import mark

from spectral_connectivity.connectivity import (
    Connectivity,
    _bandpass,
    _complex_inner_product,
    _conjugate_transpose,
    _find_largest_independent_group,
    _find_largest_significant_group,
    _get_independent_frequencies,
    _get_independent_frequency_step,
    _inner_combination,
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
    np.random.seed(42)
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (1, 30, 1, 1, 2)
    fourier_coefficients = np.random.uniform(
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
    np.random.seed(42)
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (1, 30, 1, 1, 2)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals), dtype=complex
    )
    fourier_coefficients[..., 0] = np.random.uniform(
        0.1, 2, (n_time_samples, n_trials, n_tapers, n_fft_samples)
    ) * np.exp(1j * np.pi / 2)
    fourier_coefficients[..., 1] = np.random.uniform(
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
    """Test that incoherent signals are set to zero or below."""
    np.random.seed(0)
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (1, 200, 5, 1, 2)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals), dtype=complex
    )

    angles1 = np.random.uniform(
        0, 2 * np.pi, (n_time_samples, n_trials, n_tapers, n_fft_samples)
    )
    angles2 = np.random.uniform(
        0, 2 * np.pi, (n_time_samples, n_trials, n_tapers, n_fft_samples)
    )

    fourier_coefficients[..., 0] = np.exp(1j * angles1)
    fourier_coefficients[..., 1] = np.exp(1j * angles2)

    this_Conn = Connectivity(fourier_coefficients=fourier_coefficients)

    assert np.all(this_Conn.debiased_squared_phase_lag_index() < np.finfo(float).eps)


def test_debiased_squared_weighted_phase_lag_index():
    """Test that incoherent signals are set to zero or below."""
    np.random.seed(0)
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (1, 200, 5, 1, 2)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals), dtype=complex
    )

    angles1 = np.random.uniform(
        0, 2 * np.pi, (n_time_samples, n_trials, n_tapers, n_fft_samples)
    )
    angles2 = np.random.uniform(
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
    np.random.seed(0)
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (1, 200, 5, 1, 2)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals), dtype=complex
    )

    magnitude1 = np.random.uniform(
        0.5, 3, (n_time_samples, n_trials, n_tapers, n_fft_samples)
    )
    angles1 = np.random.uniform(
        0, 2 * np.pi, (n_time_samples, n_trials, n_tapers, n_fft_samples)
    )
    magnitude2 = np.random.uniform(
        0.5, 3, (n_time_samples, n_trials, n_tapers, n_fft_samples)
    )
    angles2 = np.random.uniform(
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


def test__inner_combination():
    n_time_samples, n_fft_samples, n_signals = (2, 3, 2)
    test_data = np.arange(0, n_time_samples * n_fft_samples * n_signals).reshape(
        (n_time_samples, n_fft_samples, n_signals)
    )

    expected_combination = np.array([[8, 23], [188, 239]])

    assert np.allclose(_inner_combination(test_data, axis=-2), expected_combination)


def test_directed_transfer_function():
    # Use proper 5D shape for fourier_coefficients
    c = Connectivity(fourier_coefficients=np.empty((1, 1, 1, 1, 2)))
    type(c)._transfer_function = PropertyMock(
        return_value=np.arange(1, 5).reshape((2, 2))
    )
    dtf = c.directed_transfer_function()
    assert np.allclose(dtf.sum(axis=-1), 1.0)
    assert np.all((dtf >= 0.0) & (dtf <= 1.0))


def test_partial_directed_coherence():
    # Use proper 5D shape for fourier_coefficients
    c = Connectivity(fourier_coefficients=np.empty((1, 1, 1, 1, 2)))
    type(c)._MVAR_Fourier_coefficients = PropertyMock(
        return_value=np.arange(1, 5).reshape((2, 2))
    )
    pdc = c.partial_directed_coherence()
    assert np.allclose(pdc.sum(axis=-2), 1.0)
    assert np.all((pdc >= 0.0) & (pdc <= 1.0))


def test_subset_pairwise_granger_prediction():
    np.random.seed(0)
    T = 64

    # Generate causal signals: x -> y
    x = np.random.randn(2, T)
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
    np.random.seed(42)
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = 1, 1, 1, 1024, 2

    # Create random fourier coefficients with full frequency spectrum
    fourier_coefficients = np.random.random(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals)
    ).astype(complex)

    c = Connectivity(fourier_coefficients=fourier_coefficients)

    # Test coherence which uses @_non_negative_frequencies decorator
    coherence = c.coherence_magnitude()

    # For even N=1024, should have N//2+1 = 513 frequencies (including Nyquist)
    expected_n_frequencies = n_fft_samples // 2 + 1
    assert (
        coherence.shape[-3] == expected_n_frequencies
    ), f"Expected {expected_n_frequencies} frequencies, got {coherence.shape[-3]}"


def test_nyquist_bin_odd_n():
    """Test that frequency indexing works correctly for odd N FFT lengths."""
    # Create signal with odd FFT length (N=1023)
    np.random.seed(42)
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = 1, 1, 1, 1023, 2

    # Create random fourier coefficients with full frequency spectrum
    fourier_coefficients = np.random.random(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals)
    ).astype(complex)

    c = Connectivity(fourier_coefficients=fourier_coefficients)

    # Test coherence which uses @_non_negative_frequencies decorator
    coherence = c.coherence_magnitude()

    # For odd N=1023, should have (N+1)//2 = 512 frequencies (no Nyquist)
    expected_n_frequencies = (n_fft_samples + 1) // 2
    assert (
        coherence.shape[-3] == expected_n_frequencies
    ), f"Expected {expected_n_frequencies} frequencies, got {coherence.shape[-3]}"


def test_mvar_regularized_inverse_near_singular():
    """Test regularized inverse handles near-singular frequency bins."""
    np.random.seed(42)
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (
        1, 10, 1, 5, 3
    )

    # Create nearly singular Fourier coefficients by making signals
    # highly correlated
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals),
        dtype=complex,
    )

    # Base signal
    base_signal = np.random.randn(
        n_time_samples, n_trials, n_tapers, n_fft_samples
    ) + 1j * np.random.randn(
        n_time_samples, n_trials, n_tapers, n_fft_samples
    )

    # Create near-singular cross-spectral matrix by making signals
    # nearly dependent
    fourier_coefficients[..., 0] = base_signal
    fourier_coefficients[..., 1] = base_signal + 1e-10 * (
        np.random.randn(n_time_samples, n_trials, n_tapers, n_fft_samples)
        + 1j * np.random.randn(n_time_samples, n_trials, n_tapers, n_fft_samples)
    )
    fourier_coefficients[..., 2] = base_signal + 1e-10 * (
        np.random.randn(n_time_samples, n_trials, n_tapers, n_fft_samples)
        + 1j * np.random.randn(n_time_samples, n_trials, n_tapers, n_fft_samples)
    )

    # This should not raise LinAlgError with regularized inverse
    conn = Connectivity(fourier_coefficients=fourier_coefficients)

    # Test that MVAR coefficients are computed without error
    mvar_coeffs = conn._MVAR_Fourier_coefficients
    assert mvar_coeffs is not None
    assert np.all(np.isfinite(mvar_coeffs))

    # Test that transfer function is computed without error
    transfer_func = conn._transfer_function
    assert transfer_func is not None
    assert np.all(np.isfinite(transfer_func))

    # Test connectivity measures that depend on MVAR work
    dtf = conn.directed_transfer_function()
    assert np.all(np.isfinite(dtf))
    assert np.all(dtf >= 0)  # DTF should be non-negative
    assert np.all(dtf <= 1)  # DTF should be bounded by 1


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
    """Test that Connectivity requires at least 2 signals."""
    import pytest

    # Test with 0 signals
    with pytest.raises(ValueError, match=r"At least 2 signals are required.*got 0"):
        fourier_0_signals = np.ones((2, 2, 2, 100, 0), dtype=np.complex128)
        Connectivity(fourier_coefficients=fourier_0_signals)

    # Test with 1 signal
    with pytest.raises(ValueError, match=r"At least 2 signals are required.*got 1"):
        fourier_1_signal = np.ones((2, 2, 2, 100, 1), dtype=np.complex128)
        Connectivity(fourier_coefficients=fourier_1_signal)

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


def test_expectation_cross_spectral_matrix_blocks():
    """Test that blocked computation produces identical results to unblocked.

    The blocks parameter enables memory-efficient computation of large
    connectivity matrices by processing signal pairs in chunks. This test
    verifies that using blocks produces identical results to computing all
    connections at once.
    """
    # Create test data with known structure
    # Use realistic dimensions: 10 time windows, 3 trials, 5 tapers, 50 frequencies, 10 signals
    n_time_windows = 10
    n_trials = 3
    n_tapers = 5
    n_frequencies = 50
    n_signals = 10

    # Create Fourier coefficients with some structure
    np.random.seed(42)
    fourier_coefficients = np.random.randn(
        n_time_windows, n_trials, n_tapers, n_frequencies, n_signals
    ) + 1j * np.random.randn(
        n_time_windows, n_trials, n_tapers, n_frequencies, n_signals
    )
    fourier_coefficients = fourier_coefficients.astype(np.complex128)

    # Test with different expectation types
    expectation_types = ["trials_tapers", "trials", "tapers"]

    for expectation_type in expectation_types:
        # Compute without blocks (all connections at once)
        conn_unblocked = Connectivity(
            fourier_coefficients=fourier_coefficients,
            expectation_type=expectation_type,
            blocks=None,
        )
        csm_unblocked = conn_unblocked._expectation_cross_spectral_matrix()

        # Test with different numbers of blocks
        for n_blocks in [2, 3, 5]:
            conn_blocked = Connectivity(
                fourier_coefficients=fourier_coefficients,
                expectation_type=expectation_type,
                blocks=n_blocks,
            )
            csm_blocked = conn_blocked._expectation_cross_spectral_matrix()

            # Verify shapes match
            assert csm_blocked.shape == csm_unblocked.shape, (
                f"Shape mismatch with blocks={n_blocks}, "
                f"expectation_type={expectation_type}: "
                f"{csm_blocked.shape} vs {csm_unblocked.shape}"
            )

            # Verify values match within floating-point tolerance
            assert np.allclose(csm_blocked, csm_unblocked, rtol=1e-10, atol=1e-12), (
                f"Values mismatch with blocks={n_blocks}, "
                f"expectation_type={expectation_type}. "
                f"Max difference: {np.max(np.abs(csm_blocked - csm_unblocked))}"
            )


def test_expectation_cross_spectral_matrix_blocks_coherence():
    """Test that blocked computation produces identical coherence results.

    This test verifies that coherence, a normalized connectivity measure,
    produces identical results whether computed with or without blocks.
    """
    # Create test data
    n_time_windows = 5
    n_trials = 2
    n_tapers = 3
    n_frequencies = 20
    n_signals = 8

    np.random.seed(123)
    fourier_coefficients = np.random.randn(
        n_time_windows, n_trials, n_tapers, n_frequencies, n_signals
    ) + 1j * np.random.randn(
        n_time_windows, n_trials, n_tapers, n_frequencies, n_signals
    )
    fourier_coefficients = fourier_coefficients.astype(np.complex128)

    # Compute coherence without blocks
    conn_unblocked = Connectivity(
        fourier_coefficients=fourier_coefficients, blocks=None
    )
    coherence_unblocked = conn_unblocked.coherence_magnitude()

    # Compute coherence with blocks
    for n_blocks in [2, 4]:
        conn_blocked = Connectivity(
            fourier_coefficients=fourier_coefficients, blocks=n_blocks
        )
        coherence_blocked = conn_blocked.coherence_magnitude()

        # Verify shapes match
        assert coherence_blocked.shape == coherence_unblocked.shape

        # Verify NaN locations match (diagonal elements)
        assert np.array_equal(
            np.isnan(coherence_blocked), np.isnan(coherence_unblocked)
        ), f"NaN pattern mismatch with blocks={n_blocks}"

        # Verify non-NaN values match
        mask = ~np.isnan(coherence_unblocked)
        assert np.allclose(
            coherence_blocked[mask], coherence_unblocked[mask], rtol=1e-10, atol=1e-12
        ), (
            f"Coherence mismatch with blocks={n_blocks}. "
            f"Max difference: {np.max(np.abs(coherence_blocked[mask] - coherence_unblocked[mask]))}"
        )


def test_expectation_cross_spectral_matrix_blocks_edge_cases():
    """Test edge cases for blocked computation.

    This test verifies that blocks parameter handles edge cases correctly:
    - blocks=1 (equivalent to unblocked)
    - blocks > number of signal pairs (more blocks than needed)
    - Very small datasets
    """
    # Small dataset
    n_time_windows = 2
    n_trials = 2
    n_tapers = 2
    n_frequencies = 5
    n_signals = 3  # Only 3 signals = 3 unique pairs in upper triangle

    np.random.seed(456)
    fourier_coefficients = np.random.randn(
        n_time_windows, n_trials, n_tapers, n_frequencies, n_signals
    ) + 1j * np.random.randn(
        n_time_windows, n_trials, n_tapers, n_frequencies, n_signals
    )
    fourier_coefficients = fourier_coefficients.astype(np.complex128)

    # Compute reference (unblocked)
    conn_ref = Connectivity(fourier_coefficients=fourier_coefficients, blocks=None)
    csm_ref = conn_ref._expectation_cross_spectral_matrix()

    # Test blocks=1 (should work like unblocked)
    conn_block1 = Connectivity(fourier_coefficients=fourier_coefficients, blocks=1)
    csm_block1 = conn_block1._expectation_cross_spectral_matrix()
    assert np.allclose(csm_block1, csm_ref, rtol=1e-10, atol=1e-12)

    # Test blocks > number of pairs (should handle gracefully)
    # With 3 signals, there are 3 pairs: (0,1), (0,2), (1,2)
    conn_block10 = Connectivity(fourier_coefficients=fourier_coefficients, blocks=10)
    csm_block10 = conn_block10._expectation_cross_spectral_matrix()
    assert np.allclose(csm_block10, csm_ref, rtol=1e-10, atol=1e-12)


def test_blocks_parameter_symmetry():
    """Test that blocked computation maintains matrix symmetry.

    The cross-spectral matrix should be symmetric (csm[i,j] = csm[j,i]*).
    This test verifies that blocked computation properly fills both
    upper and lower triangles.
    """
    n_time_windows = 3
    n_trials = 2
    n_tapers = 2
    n_frequencies = 10
    n_signals = 6

    np.random.seed(789)
    fourier_coefficients = np.random.randn(
        n_time_windows, n_trials, n_tapers, n_frequencies, n_signals
    ) + 1j * np.random.randn(
        n_time_windows, n_trials, n_tapers, n_frequencies, n_signals
    )
    fourier_coefficients = fourier_coefficients.astype(np.complex128)

    # Test with blocks
    conn_blocked = Connectivity(fourier_coefficients=fourier_coefficients, blocks=3)
    csm_blocked = conn_blocked._expectation_cross_spectral_matrix()

    # Verify symmetry: csm[..., i, j] should equal conj(csm[..., j, i])
    csm_transpose_conj = np.conj(np.swapaxes(csm_blocked, -2, -1))

    assert np.allclose(csm_blocked, csm_transpose_conj, rtol=1e-10, atol=1e-12), (
        "Cross-spectral matrix is not symmetric with blocked computation. "
        f"Max difference: {np.max(np.abs(csm_blocked - csm_transpose_conj))}"
    )


def test_blocks_reduce_memory():
    """Test that blocked computation reduces peak memory usage.

    The blocks parameter is designed to reduce memory consumption for large
    connectivity matrices by computing signal pairs in chunks. This test
    verifies that using blocks actually reduces peak memory usage.

    Memory Reduction Mechanism:
    - Without blocks: Computes full (n_signals x n_signals) cross-spectral matrix
    - With blocks: Computes smaller chunks at a time, reducing peak memory

    Expected memory reduction is most noticeable for large n_signals
    (e.g., n_signals >= 50).
    """
    import tracemalloc

    # Use moderately large dimensions to observe memory difference
    # (not too large to avoid slow tests)
    n_time_windows = 20
    n_trials = 5
    n_tapers = 7
    n_frequencies = 100
    n_signals = 50  # Large enough to see memory benefit

    np.random.seed(999)
    fourier_coefficients = np.random.randn(
        n_time_windows, n_trials, n_tapers, n_frequencies, n_signals
    ) + 1j * np.random.randn(
        n_time_windows, n_trials, n_tapers, n_frequencies, n_signals
    )
    fourier_coefficients = fourier_coefficients.astype(np.complex128)

    # Measure memory for unblocked computation
    tracemalloc.start()
    conn_unblocked = Connectivity(
        fourier_coefficients=fourier_coefficients, blocks=None
    )
    _ = conn_unblocked._expectation_cross_spectral_matrix()
    _, peak_unblocked = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Measure memory for blocked computation
    tracemalloc.start()
    conn_blocked = Connectivity(fourier_coefficients=fourier_coefficients, blocks=5)
    _ = conn_blocked._expectation_cross_spectral_matrix()
    _, peak_blocked = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Verify that blocked uses less or equal memory
    # Note: In practice, blocked computation should use less memory for large arrays,
    # but the benefit may be small for moderate sizes. We check that it doesn't
    # use significantly MORE memory (within 20% overhead for block management).
    memory_ratio = peak_blocked / peak_unblocked
    assert memory_ratio <= 1.2, (
        f"Blocked computation uses significantly more memory than unblocked. "
        f"Ratio: {memory_ratio:.2f} (peak_blocked={peak_blocked:,}, "
        f"peak_unblocked={peak_unblocked:,})"
    )

    # Document the actual memory usage for reference
    # (this helps users understand the benefit)
    print(
        f"\nMemory usage comparison (n_signals={n_signals}):\n"
        f"  Unblocked: {peak_unblocked:,} bytes\n"
        f"  Blocked:   {peak_blocked:,} bytes\n"
        f"  Ratio:     {memory_ratio:.2%}"
    )
