from unittest.mock import PropertyMock

import numpy as np
from pytest import mark
from spectral_connectivity.connectivity import (
    Connectivity, _bandpass, _complex_inner_product, _conjugate_transpose,
    _find_largest_independent_group, _find_largest_significant_group,
    _get_independent_frequencies, _get_independent_frequency_step,
    _inner_combination, _remove_instantaneous_causality, _reshape,
    _set_diagonal_to_zero, _squared_magnitude, _total_inflow, _total_outflow)


@mark.parametrize('axis', [(0), (1), (2), (3)])
def test_cross_spectrum(axis):
    '''Test that the cross spectrum is correct for each dimension.'''
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (
        2, 2, 2, 2, 2)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals),
        dtype=np.complex)

    signal_fourier_coefficient = [2 * np.exp(1j * np.pi / 2),
                                  3 * np.exp(1j * -np.pi / 2)]
    fourier_ind = [slice(0, 4)] * 5
    fourier_ind[-1] = slice(None)
    fourier_ind[axis] = slice(1, 2)
    fourier_coefficients[fourier_ind] = signal_fourier_coefficient

    expected_cross_spectral_matrix = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals,
         n_signals), dtype=np.complex)

    expected_slice = np.array([[4, -6], [-6, 9]], dtype=np.complex)
    expected_ind = [slice(0, 5)] * 6
    expected_ind[-1] = slice(None)
    expected_ind[-2] = slice(None)
    expected_ind[axis] = slice(1, 2)
    expected_cross_spectral_matrix[expected_ind] = expected_slice

    this_Conn = Connectivity(fourier_coefficients=fourier_coefficients)
    assert np.allclose(
        expected_cross_spectral_matrix, this_Conn._cross_spectral_matrix)


def test_power():
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (
        1, 1, 1, 1, 2)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals),
        dtype=np.complex)

    fourier_coefficients[..., :] = [2 * np.exp(1j * np.pi / 2),
                                    3 * np.exp(1j * -np.pi / 2)]

    expected_power = np.zeros((n_time_samples, n_fft_samples, n_signals))

    expected_power[..., :] = [4, 9]

    this_Conn = Connectivity(fourier_coefficients=fourier_coefficients)
    assert np.allclose(
        expected_power, this_Conn.power())


@mark.parametrize(
    'expectation_type, expected_shape',
    [('trials_tapers', (1, 4, 5)),
     ('trials', (1, 3, 4, 5)),
     ('tapers', (1, 2, 4, 5))])
def test_expectation(expectation_type, expected_shape):
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (
        1, 2, 3, 4, 5)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals),
        dtype=np.complex)

    this_Conn = Connectivity(
        fourier_coefficients=fourier_coefficients,
        expectation_type=expectation_type,
    )
    expectation_function = this_Conn._expectation
    assert np.allclose(
        expected_shape, expectation_function(fourier_coefficients).shape)


@mark.parametrize(
    'expectation_type, expected_n_observations',
    [('trials_tapers', 6),
     ('trials', 2),
     ('tapers', 3)])
def test_n_observations(expectation_type, expected_n_observations):
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (
        1, 2, 3, 4, 5)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals),
        dtype=np.complex)

    this_Conn = Connectivity(
        fourier_coefficients=fourier_coefficients,
        expectation_type=expectation_type,
    )
    assert this_Conn.n_observations == expected_n_observations


def test_coherency():
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (
        1, 30, 1, 1, 2)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals),
        dtype=np.complex)

    fourier_coefficients[..., :] = [2 * np.exp(1j * np.pi / 2),
                                    3 * np.exp(1j * -np.pi / 2)]
    this_Conn = Connectivity(fourier_coefficients=fourier_coefficients)
    expected_coherence_magnitude = np.array([[np.nan, 1], [1, np.nan]])
    expected_phase = np.zeros((2, 2)) * np.nan
    expected_phase[0, 1] = np.pi
    expected_phase[1, 0] = -np.pi

    assert np.allclose(
        np.abs(this_Conn.coherency().squeeze()),
        expected_coherence_magnitude, equal_nan=True)
    assert np.allclose(
        np.angle(this_Conn.coherency().squeeze()),
        expected_phase, equal_nan=True)


def test_imaginary_coherence():
    '''Test that imaginary coherence sets signals with the same phase
    to zero.'''
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (
        1, 30, 1, 1, 2)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals),
        dtype=np.complex)

    fourier_coefficients[..., :] = [2 * np.exp(1j * 0),
                                    3 * np.exp(1j * 0)]
    expected_imaginary_coherence = np.zeros((2, 2))

    this_Conn = Connectivity(fourier_coefficients=fourier_coefficients)
    assert np.allclose(
        this_Conn.imaginary_coherence().squeeze(),
        expected_imaginary_coherence)


def test_phase_locking_value():
    '''Make sure phase locking value ignores magnitudes.'''
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (
        1, 30, 1, 1, 2)
    fourier_coefficients = (
        np.random.uniform(0, 2, (n_time_samples, n_trials, n_tapers,
                                 n_fft_samples, n_signals)) *
        np.exp(1j * np.pi / 2))
    expected_phase_locking_value_magnitude = np.ones(
        fourier_coefficients.shape)
    expected_phase_locking_value_angle = np.zeros(
        fourier_coefficients.shape)
    this_Conn = Connectivity(fourier_coefficients=fourier_coefficients)

    assert np.allclose(
        np.abs(this_Conn.phase_locking_value()),
        expected_phase_locking_value_magnitude)
    assert np.allclose(
        np.angle(this_Conn.phase_locking_value()),
        expected_phase_locking_value_angle)


def test_phase_lag_index_sets_zero_phase_signals_to_zero():
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (
        1, 30, 1, 1, 2)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals),
        dtype=np.complex)

    fourier_coefficients[..., :] = [2 * np.exp(1j * 0),
                                    3 * np.exp(1j * 0)]
    expected_phase_lag_index = np.zeros((2, 2))

    this_Conn = Connectivity(fourier_coefficients=fourier_coefficients)
    assert np.allclose(
        this_Conn.phase_lag_index().squeeze(),
        expected_phase_lag_index)


def test_phase_lag_index_sets_angles_up_to_pi_to_same_value():
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (
        1, 30, 1, 1, 2)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals),
        dtype=np.complex)
    fourier_coefficients[..., 0] = (np.random.uniform(
        0.1, 2, (n_time_samples, n_trials, n_tapers, n_fft_samples)) *
        np.exp(1j * np.pi / 2))
    fourier_coefficients[..., 1] = (np.random.uniform(
        0.1, 2, (n_time_samples, n_trials, n_tapers, n_fft_samples)) *
        np.exp(1j * np.pi / 4))

    expected_phase_lag_index = np.zeros((2, 2))
    expected_phase_lag_index[0, 1] = 1
    expected_phase_lag_index[1, 0] = -1

    this_Conn = Connectivity(fourier_coefficients=fourier_coefficients)

    assert np.allclose(
        this_Conn.phase_lag_index().squeeze(),
        expected_phase_lag_index)


def test_weighted_phase_lag_index_sets_zero_phase_signals_to_zero():
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (
        1, 30, 1, 1, 2)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals),
        dtype=np.complex)

    fourier_coefficients[..., :] = [2 * np.exp(1j * 0),
                                    3 * np.exp(1j * 0)]
    expected_phase_lag_index = np.zeros((2, 2))

    this_Conn = Connectivity(fourier_coefficients=fourier_coefficients)
    assert np.allclose(
        this_Conn.weighted_phase_lag_index().squeeze(),
        expected_phase_lag_index)


def test_weighted_phase_lag_index_is_same_as_phase_lag_index():
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (
        1, 30, 1, 1, 2)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals),
        dtype=np.complex)

    fourier_coefficients[..., :] = [1 * np.exp(1j * 3 * np.pi / 4),
                                    1 * np.exp(1j * 5 * np.pi / 4)]

    this_Conn = Connectivity(fourier_coefficients=fourier_coefficients)
    assert np.allclose(
        this_Conn.phase_lag_index(),
        this_Conn.weighted_phase_lag_index())


def test_debiased_squared_phase_lag_index():
    '''Test that incoherent signals are set to zero or below.'''
    np.random.seed(0)
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (
        1, 200, 5, 1, 2)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals),
        dtype=np.complex)

    angles1 = np.random.uniform(
        0, 2 * np.pi, (n_time_samples, n_trials, n_tapers, n_fft_samples))
    angles2 = np.random.uniform(
        0, 2 * np.pi, (n_time_samples, n_trials, n_tapers, n_fft_samples))

    fourier_coefficients[..., 0] = np.exp(1j * angles1)
    fourier_coefficients[..., 1] = np.exp(1j * angles2)

    this_Conn = Connectivity(fourier_coefficients=fourier_coefficients)

    assert np.all(
        this_Conn.debiased_squared_phase_lag_index() < np.finfo(float).eps)


def test_debiased_squared_weighted_phase_lag_index():
    '''Test that incoherent signals are set to zero or below.'''
    np.random.seed(0)
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (
        1, 200, 5, 1, 2)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals),
        dtype=np.complex)

    angles1 = np.random.uniform(
        0, 2 * np.pi, (n_time_samples, n_trials, n_tapers, n_fft_samples))
    angles2 = np.random.uniform(
        0, 2 * np.pi, (n_time_samples, n_trials, n_tapers, n_fft_samples))

    fourier_coefficients[..., 0] = np.exp(1j * angles1)
    fourier_coefficients[..., 1] = np.exp(1j * angles2)

    this_Conn = Connectivity(fourier_coefficients=fourier_coefficients)

    # set NaN to 0 so less than will work
    debiased_wPLI = this_Conn.debiased_squared_weighted_phase_lag_index()
    debiased_wPLI[np.isnan(debiased_wPLI)] = 0

    assert np.all(debiased_wPLI < np.finfo(float).eps)


def test_pairwise_phase_consistency():
    '''Test that incoherent signals are set to zero or below
    and that differences in power are ignored.'''
    np.random.seed(0)
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (
        1, 200, 5, 1, 2)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals),
        dtype=np.complex)

    magnitude1 = np.random.uniform(
        0.5, 3, (n_time_samples, n_trials, n_tapers, n_fft_samples))
    angles1 = np.random.uniform(
        0, 2 * np.pi, (n_time_samples, n_trials, n_tapers, n_fft_samples))
    magnitude2 = np.random.uniform(
        0.5, 3, (n_time_samples, n_trials, n_tapers, n_fft_samples))
    angles2 = np.random.uniform(
        0, 2 * np.pi, (n_time_samples, n_trials, n_tapers, n_fft_samples))

    fourier_coefficients[..., 0] = magnitude1 * np.exp(1j * angles1)
    fourier_coefficients[..., 1] = magnitude2 * np.exp(1j * angles2)

    this_Conn = Connectivity(fourier_coefficients=fourier_coefficients)
    ppc = this_Conn.pairwise_phase_consistency()

    # set diagonal to zero because its always 1
    diagonal_ind = np.arange(0, n_signals)
    ppc[..., diagonal_ind, diagonal_ind] = 0

    assert np.all(ppc < np.finfo(float).eps)


def test__reshape():
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (
        20, 100, 3, 10, 2)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals),
        dtype=np.complex)
    expected_shape = (n_time_samples, n_fft_samples, n_signals,
                      n_trials * n_tapers)
    assert np.allclose(
        _reshape(fourier_coefficients).shape, expected_shape)


def test__squared_magnitude():
    test_array = np.array([[1, 2], [3, 4]])
    expected_array = np.array([[1, 4], [9, 16]])
    assert np.allclose(_squared_magnitude(test_array), expected_array)


def test__conjugate_transpose():
    test_array = np.zeros((2, 2, 4), dtype=np.complex)
    test_array[1, ...] = [[1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j],
                          [1 - 2j, 3 - 4j, 5 - 6j, 7 - 8j]]
    expected_array = np.zeros((2, 4, 2), dtype=np.complex)
    expected_array[1, ...] = test_array[1, ...].conj().transpose()
    assert np.allclose(_conjugate_transpose(test_array), expected_array)


def test__complex_inner_product():
    '''Test that the complex inner product is taken over the last two
    dimensions.'''
    test_array1 = np.zeros((3, 2, 4), dtype=np.complex)
    test_array2 = np.zeros((3, 2, 4), dtype=np.complex)

    x1 = np.ones((2, 4)) * np.exp(1j * np.pi / 2)
    x2 = np.ones((2, 4)) * np.exp(1j * 0)

    test_array1[1, :, :] = x1
    test_array2[1, :, :] = x2

    test_array1[2, :, :] = x1
    test_array2[2, :, :] = x1

    expected_inner_product = np.zeros((3, 2, 2), dtype=np.complex)
    expected_inner_product[1, ...] = x1.dot(x2.T.conj())
    expected_inner_product[2, ...] = x1.dot(x1.T.conj())

    assert np.allclose(
        _complex_inner_product(test_array1, test_array2),
        expected_inner_product)


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
        test_data, labels, labels_of_interest, axis=-1)

    assert (np.allclose(expected_data, filtered_data) &
            np.allclose(expected_labels, filtered_labels))


@mark.parametrize(
    'frequency_difference, frequency_resolution, expected_step',
    [(2.0, 5.0, 3),
     (5.0, 2.0, 1),
     (2.0, 2.0, 1)])
def test__get_independent_frequency_step(
        frequency_difference, frequency_resolution, expected_step):
    step = _get_independent_frequency_step(
        frequency_difference, frequency_resolution)
    assert step == expected_step


@mark.parametrize(
    'is_significant, expected_is_significant',
    [(np.array([False, True, True, False, True, True, True, False]),
      np.array([False, False, False, False, True, True, True, False])),
     (np.ones((10,), dtype=bool), np.ones((10,), dtype=bool))
     ])
def test__find_largest_significant_group(
        is_significant, expected_is_significant):

    assert np.allclose(
        _find_largest_significant_group(is_significant),
        expected_is_significant)


def test__find_largest_significant_group_with_no_significant():
    is_significant = np.zeros((10,), dtype=bool)
    expected_is_significant = np.zeros((10,), dtype=bool)

    assert np.allclose(
        _find_largest_significant_group(is_significant),
        expected_is_significant)


def test__get_independent_frequencies():
    is_significant = np.zeros((10,), dtype=bool)
    is_significant[3:7] = True
    frequency_step = 2

    expected_is_significant = np.zeros((10,), dtype=bool)
    expected_is_significant[3:7:frequency_step] = True

    assert np.allclose(
        _get_independent_frequencies(is_significant, frequency_step),
        expected_is_significant)


@mark.parametrize(
    'min_group_size, expected_is_significant',
    [(3, np.zeros((10,), dtype=bool)),
     (1, np.array(
         [False, False, False, False, True, False,  True, False,
          False, False], dtype=bool))])
def test__find_largest_independent_group(
        min_group_size, expected_is_significant):
    is_significant = np.zeros((10,), dtype=bool)
    is_significant[1:3] = True
    is_significant[4:7] = True
    is_significant[8] = True
    frequency_step = 2

    assert np.allclose(
        _find_largest_independent_group(is_significant, frequency_step,
                                        min_group_size=min_group_size),
        expected_is_significant)


def test__total_inflow():
    transfer_function = np.ones((2, 3, 3))
    noise_variance = [4, 2, 3]
    expected_total_inflow = 3 * np.ones((2, 3, 1))

    assert np.allclose(
        _total_inflow(transfer_function, noise_variance),
        expected_total_inflow)


def test__total_outflow():
    MVAR_Fourier_coefficients = np.ones((2, 3, 3))
    noise_variance = np.array([0.25, 0.5, 1 / 3])
    expected_total_outflow = (np.ones((2, 1, 3)) *
                              np.sqrt(1.0 / noise_variance * 3))

    assert np.allclose(
        _total_outflow(MVAR_Fourier_coefficients, noise_variance),
        expected_total_outflow)


def test__remove_instantaneous_causality():
    noise_covariance = np.zeros((2, 2, 2))
    x1 = np.array([[1, 2], [2, 4]], dtype=float)
    x2 = np.array([[8, 4], [4, 16]], dtype=float)
    noise_covariance[0, ...] = x1
    noise_covariance[1, ...] = x2

    # x -> y: var(x) - (cov(x,y) ** 2 / var(y))
    # y -> x: var(y) - (cov(x,y) ** 2 / var(x))
    expected_rotated_noise_covariance = np.zeros((2, 2, 2))

    expected_rotated_noise_covariance[0, 0, 1] = (
        x1[1, 1] - (x1[0, 1] ** 2 / x1[0, 0]))
    expected_rotated_noise_covariance[0, 1, 0] = (
        x1[0, 0] - (x1[1, 0] ** 2 / x1[1, 1]))

    expected_rotated_noise_covariance[1, 0, 1] = (
        x2[1, 1] - (x2[0, 1] ** 2 / x2[0, 0]))
    expected_rotated_noise_covariance[1, 1, 0] = (
        x2[0, 0] - (x2[1, 0] ** 2 / x2[1, 1]))

    assert np.allclose(
        _remove_instantaneous_causality(noise_covariance),
        expected_rotated_noise_covariance)


def test__inner_combination():
    n_time_samples, n_fft_samples, n_signals = (2, 3, 2)
    test_data = np.arange(
        0, n_time_samples * n_fft_samples * n_signals).reshape(
        (n_time_samples, n_fft_samples, n_signals))

    expected_combination = np.array([[8, 23], [188, 239]])

    assert np.allclose(
        _inner_combination(test_data, axis=-2), expected_combination)


def test_directed_transfer_function():
    c = Connectivity(fourier_coefficients=np.empty((1,)))
    type(c)._transfer_function = PropertyMock(
        return_value=np.arange(1, 5).reshape((2, 2)))
    dtf = c.directed_transfer_function()
    assert np.allclose(dtf.sum(axis=-1), 1.0)
    assert np.all((dtf >= 0.0) & (dtf <= 1.0))


def test_partial_directed_coherence():
    c = Connectivity(fourier_coefficients=np.empty((1,)))
    type(c)._MVAR_Fourier_coefficients = PropertyMock(
        return_value=np.arange(1, 5).reshape((2, 2)))
    pdc = c.partial_directed_coherence()
    assert np.allclose(pdc.sum(axis=-2), 1.0)
    assert np.all((pdc >= 0.0) & (pdc <= 1.0))
