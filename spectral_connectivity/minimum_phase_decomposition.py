"""Minimum phase decomposition for spectral density matrices.

A spectral density matrix can be decomposed into minimum phase functions
using the Wilson algorithm. This decomposition is used in computing
pairwise spectral Granger prediction and other directed connectivity measures.
"""

import os
from logging import getLogger
from typing import Any

import numpy as np
from numpy.typing import NDArray

if os.environ.get("SPECTRAL_CONNECTIVITY_ENABLE_GPU") == "true":
    try:
        import cupy as xp
        from cupyx.scipy.fft import fft, ifft
    except ImportError:
        import numpy as xp
        from scipy.fft import fft, ifft
else:
    import numpy as xp
    from scipy.fft import fft, ifft


logger = getLogger(__name__)


def _conjugate_transpose(x: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
    """Compute conjugate transpose of the last two dimensions.

    Parameters
    ----------
    x : NDArray[complexfloating], shape (..., M, N)
        Input array.

    Returns
    -------
    x_H : NDArray[complexfloating], shape (..., N, M)
        Conjugate transpose of last two dimensions.
    """
    return x.swapaxes(-1, -2).conjugate()


def _get_initial_conditions(
    cross_spectral_matrix: NDArray[np.complexfloating],
) -> NDArray[np.complexfloating]:
    """Generate initial guess for minimum phase factor using Cholesky decomposition.

    Provides an initial estimate for the Wilson algorithm by taking the Cholesky
    decomposition of the zero-lag cross-spectral matrix (real part of inverse FFT).
    Falls back to random initialization if Cholesky fails.

    Parameters
    ----------
    cross_spectral_matrix : NDArray[complexfloating], shape (n_time_samples, ..., n_fft_samples, n_signals, n_signals)
        Cross-spectral density matrix to be decomposed.

    Returns
    -------
    minimum_phase_factor : NDArray[complexfloating], shape (n_time_samples, ..., 1, n_signals, n_signals)
        Initial guess for minimum phase square root matrix.

    Notes
    -----
    If the Cholesky decomposition fails (matrix not positive definite),
    the function generates a random positive definite matrix as fallback.
    """
    try:
        return xp.linalg.cholesky(
            ifft(cross_spectral_matrix, axis=-3)[..., 0:1, :, :].real
        ).swapaxes(-1, -2)
    except xp.linalg.linalg.LinAlgError:
        logger.warning(
            "Computing the initial conditions using the Cholesky failed. "
            "Using a random initial condition."
        )

        new_shape = list(cross_spectral_matrix.shape)
        N_RAND = 1000
        new_shape[-3] = N_RAND
        random_start = xp.random.standard_normal(size=new_shape)

        random_start = xp.matmul(random_start, _conjugate_transpose(random_start)).mean(
            axis=-3, keepdims=True
        )

        return xp.linalg.cholesky(random_start)


def _get_causal_signal(
    linear_predictor: NDArray[np.complexfloating],
) -> NDArray[np.complexfloating]:
    """Extract causal part of linear predictor (plus operator).

    Implements the "plus" operator from the Wilson algorithm by:
    1. Taking half the roots on the unit circle (zero lag)
    2. Taking all roots inside the unit circle (positive lags)
    3. Making zero-lag term upper triangular

    This gives A_(t+1)(Z) / A_(t)(Z) in the Wilson algorithm notation.

    Parameters
    ----------
    linear_predictor : NDArray[complexfloating], shape (..., n_fft_samples, n_signals, n_signals)
        Linear predictor matrix in frequency domain.

    Returns
    -------
    causal_part_of_linear_predictor : NDArray[complexfloating], shape (..., n_fft_samples, n_signals, n_signals)
        Causal part of the linear predictor after plus operator.

    Notes
    -----
    The plus operator is a key component of the Wilson algorithm for
    minimum phase decomposition. It ensures causality by zeroing out
    negative lag components and enforcing upper triangular structure
    at zero lag.
    """
    n_signals = linear_predictor.shape[-1]
    n_fft_samples = linear_predictor.shape[-3]
    linear_predictor_coefficients = ifft(linear_predictor, axis=-3)

    # Take half of the roots on the unit circle
    linear_predictor_coefficients[..., 0, :, :] *= 0.5

    # Make the unit circle roots upper triangular
    lower_triangular_ind = np.tril_indices(n_signals, k=-1)
    linear_predictor_coefficients[
        ..., 0, lower_triangular_ind[0], lower_triangular_ind[1]
    ] = 0

    # Take only the roots inside the unit circle (positive lags)
    linear_predictor_coefficients[..., (n_fft_samples + 1) // 2 :, :, :] = 0
    return fft(linear_predictor_coefficients, axis=-3)


def _check_convergence(
    current: NDArray[np.complexfloating],
    old: NDArray[np.complexfloating],
    tolerance: float = 1e-8,
) -> NDArray[np.bool_]:
    """Check convergence of Wilson algorithm at each time point.

    Uses infinity norm to measure the maximum absolute difference between
    current and previous iterations for each time point.

    Parameters
    ----------
    current : NDArray[complexfloating], shape (n_time_points, ...)
        Current iteration's minimum phase factor estimates.
    old : NDArray[complexfloating], shape (n_time_points, ...)
        Previous iteration's minimum phase factor estimates.
    tolerance : float, default=1e-8
        Convergence tolerance. Time points with maximum difference below
        this value are considered converged.

    Returns
    -------
    is_converged : NDArray[bool], shape (n_time_points,)
        Boolean array indicating convergence status for each time point.

    Examples
    --------
    >>> import numpy as np
    >>> current = np.random.randn(10, 5, 5) + 1j * np.random.randn(10, 5, 5)
    >>> old = current + 1e-10 * np.random.randn(10, 5, 5)
    >>> converged = _check_convergence(current, old, tolerance=1e-8)
    """
    n_time_points = current.shape[0]
    error = xp.linalg.norm(
        xp.reshape(current - old, (n_time_points, -1)), ord=xp.inf, axis=1
    )
    return error < tolerance


def _get_linear_predictor(
    minimum_phase_factor: NDArray[np.complexfloating],
    cross_spectral_matrix: NDArray[np.complexfloating],
    I: NDArray[np.complexfloating],
) -> NDArray[np.complexfloating]:
    """Compute linear predictor for Wilson algorithm update step.

    Calculates how much to adjust the current minimum phase factor guess
    by solving: G^{-1} S G^{-H} + I, where G is the current guess, S is
    the cross-spectral matrix, and H denotes conjugate transpose.

    Parameters
    ----------
    minimum_phase_factor : NDArray[complexfloating], shape (n_time_samples, ..., n_fft_samples, n_signals, n_signals)
        Current minimum phase square root estimate.
    cross_spectral_matrix : NDArray[complexfloating], shape (n_time_samples, ..., n_fft_samples, n_signals, n_signals)
        Target cross-spectral matrix to be factored.
    I : NDArray[complexfloating], shape (n_signals, n_signals)
        Identity matrix.

    Returns
    -------
    linear_predictor : NDArray[complexfloating], shape (n_time_samples, ..., n_fft_samples, n_signals, n_signals)
        Adjustment matrix for updating minimum phase factor estimate.

    Notes
    -----
    This implements the core update step of the Wilson algorithm:
    computing the "covariance sandwich estimator" that measures the
    discrepancy between the current factorization and target matrix.
    """
    covariance_sandwich_estimator = xp.linalg.solve(
        minimum_phase_factor, cross_spectral_matrix
    )
    covariance_sandwich_estimator = xp.linalg.solve(
        minimum_phase_factor, _conjugate_transpose(covariance_sandwich_estimator)
    )
    return covariance_sandwich_estimator + I


def minimum_phase_decomposition(
    cross_spectral_matrix: NDArray[np.complexfloating],
    tolerance: float = 1e-8,
    max_iterations: int = 60,
) -> NDArray[np.complexfloating]:
    """Compute minimum phase decomposition using Wilson algorithm.

    Finds a minimum phase matrix square root G of the cross-spectral density
    matrix S such that S = G G^H, where all poles of G are inside the unit
    circle. This decomposition is essential for computing directed connectivity
    measures like spectral Granger causality.

    Parameters
    ----------
    cross_spectral_matrix : NDArray[complexfloating], shape (n_time_samples, ..., n_fft_samples, n_signals, n_signals)
        Cross-spectral density matrix to be decomposed. Must be Hermitian
        positive semidefinite for each frequency.
    tolerance : float, default=1e-8
        Convergence tolerance for Wilson algorithm iterations.
    max_iterations : int, default=60
        Maximum number of iterations before stopping algorithm.

    Returns
    -------
    minimum_phase_factor : NDArray[complexfloating], shape (n_time_samples, ..., n_fft_samples, n_signals, n_signals)
        Minimum phase square root of cross_spectral_matrix. All eigenvalues
        have negative real parts (minimum phase property).

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.fft import fft
    >>> # Create a simple 2x2 cross-spectral matrix
    >>> n_times, n_freqs, n_signals = 10, 32, 2
    >>> # Generate random coefficients for AR process
    >>> coeffs = np.random.randn(n_times, n_freqs, n_signals, n_signals)
    >>> cross_spec = np.matmul(coeffs, coeffs.conj().swapaxes(-1, -2))
    >>>
    >>> # Compute minimum phase decomposition
    >>> min_phase = minimum_phase_decomposition(cross_spec)
    >>>
    >>> # Verify decomposition: should reconstruct original matrix
    >>> reconstructed = np.matmul(min_phase, min_phase.conj().swapaxes(-1, -2))
    >>> error = np.abs(reconstructed - cross_spec).max()
    >>> print(f"Reconstruction error: {error:.2e}")

    Notes
    -----
    The Wilson algorithm iteratively refines an initial guess using the
    "plus" operator (causal projection) until convergence. The algorithm
    may not converge for all time points; warnings are issued when the
    maximum iteration count is reached.

    References
    ----------
    .. [1] Wilson, G. T. (1972). The factorization of matricial spectral
           densities. SIAM Journal on Applied Mathematics, 23(4), 420-426.
    .. [2] Dhamala, M., Rangarajan, G., & Ding, M. (2008). Analyzing
           information flow in brain networks with nonparametric Granger
           causality. NeuroImage, 41(2), 354-362.
    """
    n_time_points = cross_spectral_matrix.shape[0]
    n_signals = cross_spectral_matrix.shape[-1]
    I = xp.eye(n_signals)
    is_converged = xp.zeros(n_time_points, dtype=bool)
    minimum_phase_factor = xp.zeros(cross_spectral_matrix.shape)
    minimum_phase_factor[..., :, :, :] = _get_initial_conditions(cross_spectral_matrix)

    for iteration in range(max_iterations):
        logger.debug(
            "iteration: {0}, {1} of {2} converged".format(
                iteration, is_converged.sum(), len(is_converged)
            )
        )
        old_minimum_phase_factor = minimum_phase_factor.copy()
        linear_predictor = _get_linear_predictor(
            minimum_phase_factor, cross_spectral_matrix, I
        )
        minimum_phase_factor = xp.matmul(
            minimum_phase_factor, _get_causal_signal(linear_predictor)
        )

        # If already converged at a time point, don't change.
        minimum_phase_factor[is_converged, ...] = old_minimum_phase_factor[
            is_converged, ...
        ]
        is_converged = _check_convergence(
            minimum_phase_factor, old_minimum_phase_factor, tolerance
        )
        if xp.all(is_converged):
            return minimum_phase_factor
    else:
        logger.warning(
            "Maximum iterations reached. {} of {} converged".format(
                is_converged.sum(), len(is_converged)
            )
        )
        return minimum_phase_factor
