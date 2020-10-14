import numpy as np
from scipy.fftpack import fft, ifft
from scipy.signal import freqz_zpk
from spectral_connectivity.minimum_phase_decomposition import (
    _check_convergence, _conjugate_transpose, _get_causal_signal,
    _get_intial_conditions, minimum_phase_decomposition)


def test__check_convergence():
    tolerance = 1e-8
    n_time_points = 5
    minimum_phase_factor = np.zeros((n_time_points, 4, 3))
    old_minimum_phase_factor = np.zeros((n_time_points, 4, 3))
    minimum_phase_factor[0, :, :] = 1e-9
    minimum_phase_factor[1, :, :] = 1e-7
    minimum_phase_factor[3, :] = 1
    minimum_phase_factor[4, :3, 1:2] = 1e-7

    expected_is_converged = np.array([True, False, True, False, False])

    is_converged = _check_convergence(
        minimum_phase_factor, old_minimum_phase_factor, tolerance)

    assert np.all(is_converged == expected_is_converged)


def test__conjugate_transpose():
    test_array = np.zeros((2, 2, 4), dtype=np.complex)
    test_array[1, ...] = [[1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j],
                          [1 - 2j, 3 - 4j, 5 - 6j, 7 - 8j]]
    expected_array = np.zeros((2, 4, 2), dtype=np.complex)
    expected_array[1, ...] = test_array[1, ...].conj().transpose()
    assert np.allclose(_conjugate_transpose(test_array), expected_array)


def test__get_initial_conditions():
    n_time_samples, n_fft_samples, n_signals = 3, 11, 2
    cross_spectral_matrix = np.ones(
        (n_time_samples, n_fft_samples, n_signals, n_signals),
        dtype=np.complex) * 4
    cross_spectral_matrix[..., 1, 0] = 0
    minimum_phase_factor = _get_intial_conditions(cross_spectral_matrix)
    expected_cross_spectral_matrix = np.zeros(
        (n_time_samples, 1, n_signals, n_signals),
        dtype=np.complex)
    expected_cross_spectral_matrix[..., :, :] = np.eye(n_signals) * 2
    assert np.allclose(
        minimum_phase_factor, expected_cross_spectral_matrix)


def test__get_causal_signal_removes_roots_outside_unit_circle():
    n_signals = 1
    _, transfer_function = freqz_zpk(4, 2, 1.00, whole=True)
    n_fft_samples = transfer_function.shape[0]
    linear_predictor = np.zeros(
        (1, n_fft_samples, n_signals, n_signals), dtype=np.complex)
    linear_predictor[0, :, 0, 0] = transfer_function

    expected_causal_signal = np.ones(
        (1, n_fft_samples, n_signals, n_signals), dtype=np.complex)

    causal_signal = _get_causal_signal(linear_predictor)

    assert np.allclose(causal_signal, expected_causal_signal)


def test__get_causal_signal_preserves_roots_inside_unit_circle():
    n_signals = 1
    _, transfer_function = freqz_zpk(0.25, 0.5, 1.00, whole=True)
    n_fft_samples = transfer_function.shape[0]
    linear_predictor = np.zeros(
        (1, n_fft_samples, n_signals, n_signals), dtype=np.complex)
    linear_predictor[0, :, 0, 0] = transfer_function

    _, expected_transfer_function = freqz_zpk(0.25, 0.5, 1.00, whole=True)
    linear_coef = ifft(expected_transfer_function)
    linear_coef[0] *= 0.5

    expected_causal_signal = np.zeros(
        (1, n_fft_samples, n_signals, n_signals), dtype=np.complex)
    expected_causal_signal[0, :, 0, 0] = fft(linear_coef)

    causal_signal = _get_causal_signal(linear_predictor)

    assert np.allclose(causal_signal, expected_causal_signal)


def test_minimum_phase_decomposition():
    n_signals = 1
    # minimum phase is all poles and zeros inside the unit circle
    _, transfer_function = freqz_zpk(0.25, 0.50, 1.00, whole=True)
    n_fft_samples = transfer_function.shape[0]
    expected_minimum_phase_factor = np.zeros(
        (2, n_fft_samples, n_signals, n_signals), dtype=np.complex)
    expected_minimum_phase_factor[0, :, 0, 0] = transfer_function

    _, transfer_function2 = freqz_zpk(0.125, 0.25, 1.00, whole=True)
    expected_minimum_phase_factor[1, :, 0, 0] = transfer_function2

    expected_cross_spectral_matrix = np.matmul(
        expected_minimum_phase_factor,
        _conjugate_transpose(expected_minimum_phase_factor))
    minimum_phase_factor = minimum_phase_decomposition(
        expected_cross_spectral_matrix)
    cross_spectral_matrix = (minimum_phase_factor *
                             _conjugate_transpose(minimum_phase_factor))

    assert np.allclose(minimum_phase_factor, expected_minimum_phase_factor)
    assert np.allclose(
        cross_spectral_matrix, expected_cross_spectral_matrix)
