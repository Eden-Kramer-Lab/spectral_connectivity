from logging import getLogger

import numpy as np
from scipy.fftpack import fft, ifft

logger = getLogger(__name__)


def _conjugate_transpose(x):
    '''Conjugate transpose of the last two dimensions of array x'''
    return x.swapaxes(-1, -2).conjugate()


def _get_intial_conditions(cross_spectral_matrix):
    '''Returns a guess for the minimum phase factor using the Cholesky
    factorization.

    Parameters
    ----------
    cross_spectral_matrix : array, shape (n_time_samples, ...,
                                          n_fft_samples, n_signals,
                                          n_signals)

    Returns
    -------
    minimum_phase_factor : array, shape (n_time_samples, ..., 1, n_signals,
                                         n_signals)
    '''
    try:
        return np.linalg.cholesky(
            ifft(cross_spectral_matrix, axis=-3)[..., 0:1, :, :].real
        ).swapaxes(-1, -2)
    except np.linalg.linalg.LinAlgError:
        logger.warning(
            'Computing the initial conditions using the Cholesky failed. '
            'Using a random initial condition.')

        new_shape = list(cross_spectral_matrix.shape)
        N_RAND = 1000
        new_shape[-3] = N_RAND
        random_start = np.random.standard_normal(
            size=new_shape)

        random_start = np.matmul(
            random_start, _conjugate_transpose(random_start)).mean(
                axis=-3, keepdims=True)

        return np.linalg.cholesky(random_start)


def _get_causal_signal(linear_predictor):
    '''Takes half the roots on the unit circle (zero lag) and all the roots
    inside the unit circle (positive lags).

    Gives you A_(t+1)(Z) / A_(t)(Z)
    This is the plus operator in [1]

    Parameters
    ----------
    linear_predictor : array, shape (..., n_fft_samples, n_signals,
                                     n_signals)

    Returns
    -------
    causal_part_of_linear_predictor : array, shape (..., n_fft_samples,
                                                    n_signals, n_signals)

    '''
    n_signals = linear_predictor.shape[-1]
    n_fft_samples = linear_predictor.shape[-3]
    linear_predictor_coefficients = ifft(linear_predictor, axis=-3)

    # Take half of the roots on the unit circle
    linear_predictor_coefficients[..., 0, :, :] *= 0.5

    # Make the unit circle roots upper triangular
    lower_triangular_ind = np.tril_indices(n_signals, k=-1)
    linear_predictor_coefficients[
        ..., 0, lower_triangular_ind[0], lower_triangular_ind[1]] = 0

    # Take only the roots inside the unit circle (positive lags)
    linear_predictor_coefficients[..., (n_fft_samples + 1) // 2:, :, :] = 0
    return fft(linear_predictor_coefficients, axis=-3)


def _check_convergence(current, old, tolerance=1E-8):
    '''Check convergence of Wilson algorithm at each time point.

    Parameters
    ----------
    current : array, shape (n_time_points, ...)
        Current guess.
    old : array, shape (n_time_points, ...)
        Previous guess.
    tolerance : float
        Largest difference between guesses for the matrix to be judged as
        similar.

    Returns
    -------
    is_converged : array, shape (n_time_points,)
        Boolean array that indicates whether the array has converged for
        each time point.
    '''
    n_time_points = current.shape[0]
    error = np.linalg.norm(
        np.reshape(current - old, (n_time_points, -1)), ord=np.inf, axis=1)
    return error < tolerance


def _get_linear_predictor(minimum_phase_factor, cross_spectral_matrix, I):
    '''Measure how close the minimum phase factor is to the original
    cross spectral matrix.

    Parameters
    ----------
    minimum_phase_factor : array, shape (n_time_samples, ...,
                                         n_fft_samples, n_signals,
                                         n_signals)
        The current minimum phase square root guess.
    cross_spectral_matrix : array, shape (n_time_samples, ...,
                                          n_fft_samples, n_signals,
                                          n_signals)
        The matrix to be factored.
    I : array, shape (n_signals, n_signals)
        Identity matrix.

    Returns
    -------
    linear_predictor : array, shape (n_time_samples, ..., n_fft_samples,
                                     n_signals, n_signals)
        How much to adjust for the next guess for minimum phase factor.

    '''
    covariance_sandwich_estimator = np.linalg.solve(
        minimum_phase_factor, cross_spectral_matrix)
    covariance_sandwich_estimator = np.linalg.solve(
        minimum_phase_factor,
        _conjugate_transpose(covariance_sandwich_estimator))
    return covariance_sandwich_estimator + I


def minimum_phase_decomposition(cross_spectral_matrix, tolerance=1E-8,
                                max_iterations=60):
    '''Find a minimum phase matrix square root of the cross spectral
    density using the Wilson algorithm.

    Parameters
    ----------
    cross_spectral_matrix : array, shape (n_time_samples, ...,
                                          n_fft_samples, n_signals,
                                          n_signals)
    tolerance : float
        The maximum difference between guesses.
    max_iterations : int
        The maximum number of iterations for the algorithm to converge.

    Returns
    -------
    minimum_phase_factor : array, shape (n_time_samples, ...,
                                         n_fft_samples, n_signals,
                                         n_signals)
        The square root of the `cross_spectral_matrix` where all the poles
        are inside the unit circle (minimum phase).

    '''
    n_time_points = cross_spectral_matrix.shape[0]
    n_signals = cross_spectral_matrix.shape[-1]
    I = np.eye(n_signals)
    is_converged = np.zeros(n_time_points, dtype=bool)
    minimum_phase_factor = np.zeros(cross_spectral_matrix.shape)
    minimum_phase_factor[..., :, :, :] = _get_intial_conditions(
        cross_spectral_matrix)

    for iteration in range(max_iterations):
        logger.debug(
            'iteration: {0}, {1} of {2} converged'.format(
                iteration, is_converged.sum(), len(is_converged)))
        old_minimum_phase_factor = minimum_phase_factor.copy()
        linear_predictor = _get_linear_predictor(
            minimum_phase_factor, cross_spectral_matrix, I)
        minimum_phase_factor = np.matmul(
            minimum_phase_factor, _get_causal_signal(linear_predictor))

        # If already converged at a time point, don't change.
        minimum_phase_factor[is_converged, ...] = old_minimum_phase_factor[
            is_converged, ...]
        is_converged = _check_convergence(
            minimum_phase_factor, old_minimum_phase_factor, tolerance)
        if np.all(is_converged):
            return minimum_phase_factor
    else:
        logger.warning(
            'Maximum iterations reached. {} of {} converged'.format(
                is_converged.sum(), len(is_converged)))
        return minimum_phase_factor
